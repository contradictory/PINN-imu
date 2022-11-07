import csv
import math

import numpy
import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import time
from copy import deepcopy
import torch.optim as optim

# import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN(nn.Module):
    '''
    Neural Network Class
    net_layer: list with the number of neurons for each network layer, [n_imput, ..., n_output]
    '''

    def __init__(self,
                 # input,
                 layers_size=[46, 20, 20, 20, 20, 20, 20, 20, 20, 6],  ## input size 406
                 out_size=6,
                 params_list=None):

        super(PINN, self).__init__()

        #### Data
        # self.x = x
        # self.y = y
        # self.t = t
        #
        # self.u = u
        # self.v = v
        self.input = torch.tensor([], dtype=torch.float32)
        # self.input = input
        # self.pos
        # self.v_X = 0
        # self.v_Y = 0
        # self.v_Z = 0

        #### Initializing neural network

        self.layers = nn.ModuleList()
        #### Initialize parameters (we are going to learn lambda)
        # self.lambda_1 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.lambda_2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.jacobia = torch.tensor([], dtype=torch.float32)

        for k in range(len(layers_size) - 1):
            self.layers.append(nn.Linear(layers_size[k], layers_size[k + 1]))

        # Output layer
        # self.out = nn.Linear(layers_size[-1], out_size)
        self.out = self.layers[-1]

        m_aux = 0

        for m in self.layers:

            if params_list is None:
                nn.init.normal_(m.weight, mean=0, std=1 / np.sqrt(len(layers_size)))
                nn.init.constant_(m.bias, 0.0)
            else:
                m.weight = params_list[m_aux]
                m.bias = params_list[m_aux + 1]
                m_aux += 1

        if params_list is None:
            nn.init.normal_(self.out.weight, mean=0, std=1 / np.sqrt(len(layers_size)))
            nn.init.constant_(self.out.bias, 0.0)
        else:
            self.out.weight = params_list[-2]
            self.out.bias = params_list[-1]

        # self.model = nn.ModuleDict()

        self.optimizer = optim.SGD(self.parameters(),
                                  lr= 0.001,
                                  )

    #### Forward pass

    def forward(self, x):

        for layer in self.layers:
            # Activation function
            # print(x.shape, x.dtype)
            x = layer(x)
            x = torch.tanh(x)

        # Last layer: we could choose a different functionsoftmax
        # output= F.softmax(self.out(x), dim=1)
        output = x

        return output

    #### Net NS

    def net(self, batch_in, batch_size):


        pos = torch.arange(0, 100, step=1, dtype=torch.float32, requires_grad=True)
        pos = pos.unsqueeze(1).to(device)

        k = 20
        time_embed = torch.zeros(100, 40)
        for i in range(0, 100):
            for j in range(1, k + 1):
                time_embed[i][j - 1] = torch.sin(pos[i][0] / j)
            for j in range(k + 1, k + 21):
                time_embed[i][j - 1] = torch.cos(pos[i][0] / j)


        batch_in = np.transpose(batch_in)

        batch_in = torch.tensor(batch_in, dtype=torch.float32, requires_grad=True)
        print(batch_in.size())
        batch = torch.cat((batch_in, time_embed), 1).to(device)
        self.input = batch

        output = self.forward(batch)

        output_scalar = output.cpu().clone().detach().numpy()
        p = [output_scalar[:, 0], output_scalar[:, 1], output_scalar[:, 2]]
        theta = [output_scalar[:, 3], output_scalar[:, 4], output_scalar[:, 5]]

        theta_x = theta[0]  ##: theta for input signal is wrong
        theta_y = theta[1]
        theta_z = theta[2]


        g = torch.tensor([0, 0, -9.81], dtype=torch.float32)

        # requires_grad ???
        sx, cx, sy, cy, sz, cz = np.sin(theta_x), np.cos(theta_x), \
                                 np.sin(theta_y), np.cos(theta_y), \
                                 np.sin(theta_z), np.cos(theta_z)
        # sx, cx, sy, cy, sz, cz = np.sin(torch.detach(theta_x).numpy()), np.cos(torch.detach(theta_x).numpy()), \
        #                          np.sin(torch.detach(theta_y).numpy()), np.cos(torch.detach(theta_y).numpy()), \
        #                          np.sin(torch.detach(theta_z).numpy()), np.cos(torch.detach(theta_z).numpy())
        T = [[cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
             [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sz],
             [-sy, cy * sz, cx * cy]]
        T = torch.tensor(np.array(T), dtype=torch.float32).to(device)


        first_row = torch.tensor([1, 0, 0, 0, 0, 0]).to(device)
        second_row = torch.tensor([0, 1, 0, 0, 0, 0]).to(device)
        third_row = torch.tensor([0, 0, 1, 0, 0, 0]).to(device)
        four_row = torch.tensor([0, 0, 0, 1, 0, 0]).to(device)
        five_row = torch.tensor([0, 0, 0, 0, 1, 0]).to(device)
        six_row = torch.tensor([0, 0, 0, 0, 0, 1]).to(device)


        output = torch.transpose(output, 0, 1)
        # print(output[:, 1])
        d_res = torch.tensor([], dtype=torch.float32).to(device)
        tmp = torch.zeros(6).to(device)

        for t in range(0, 100):
            # tmp = output[:, t].backward(gradient=torch.ones_like(output[:, t]), retain_graph=True, create_graph=True)
            # l = torch.zeros_like(output[:, t])
            # l[t] = 1

            # d_veri = torch.autograd.grad(output[:, t], pos, grad_outputs=torch.ones_like(first_row), retain_graph=True)
            d0 = torch.autograd.grad(output[:, t], pos, grad_outputs=first_row, retain_graph=True)[0]
            d1 = torch.autograd.grad(output[:, t], pos, grad_outputs=second_row, retain_graph=True)[0]
            d2 = torch.autograd.grad(output[:, t], pos, grad_outputs=third_row, retain_graph=True)[0]
            d3 = torch.autograd.grad(output[:, t], pos, grad_outputs=four_row, retain_graph=True)[0]
            d4 = torch.autograd.grad(output[:, t], pos, grad_outputs=five_row, retain_graph=True)[0]
            d5 = torch.autograd.grad(output[:, t], pos, grad_outputs=six_row, retain_graph=True)[0]
            tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] = d0[t], d1[t], d2[t], d3[t], d4[t], d5[t]
            d_res = torch.cat((d_res, tmp.unsqueeze(dim=1)), dim=1)
            print(t)
            del(d0)
            del(d1)
            del(d2)
            del(d3)
            del(d4)
            del(d5)

        print(d_res.size())

        d_tt = torch.tensor([], dtype=torch.float32).to(device)
        # for t in range(0, 100):
        #     d_tt0 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=first_row,  retain_graph=True)[0]
        #     d_tt1 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=second_row, retain_graph=True)[0]
        #     d_tt2 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=third_row, retain_graph=True)[0]
        #     # d_tt3 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=four_row, retain_graph=True)[0]   #### zheli haikeyi zai youhua qu 0, huozhe zhijie buyao zhe 3 weidu
        #     # d_tt4 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=five_row, retain_graph=True)[0]
        #     # d_tt5 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=six_row, retain_graph=True)[0]
        #     d_tt3 = torch.zeros_like(d_tt0)
        #     d_tt4 = torch.zeros_like(d_tt0)
        #     d_tt5 = torch.zeros_like(d_tt0)
        #
        #     tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] = d_tt0[t], d_tt1[t], d_tt2[t], d_tt3[t], d_tt4[t], d_tt5[t]
        #     d_tt = torch.cat((d_tt, tmp.unsqueeze(dim=1)), dim=1)
        #     print(t)
        #     del(d_tt0)
        #     del(d_tt1)
        #     del(d_tt2)
        #     del(d_tt3)
        #     del(d_tt4)
        #     del(d_tt5)
        print("d_tt get")
        print(d_tt)


        # p_tt = torch.cat((d_tt[0, :].unsqueeze(dim=0), d_tt[1, :].unsqueeze(dim=0), d_tt[2, :].unsqueeze(dim=0)), 0)
        # print(p_tt)
        # theta_t = torch.cat((d_res[3, :].unsqueeze(dim=0), d_res[4, :].unsqueeze(dim=0), d_res[5, :].unsqueeze(dim=0)), 0).to(device)
        theta_t = d_res[3:6, :]

        print(theta_t.size())

        a_cur = torch.transpose(batch_in[:, 0 : 3], 0, 1).to(device)
        w_cur = torch.transpose(batch_in[:, 3 : 6], 0, 1).to(device)
        # print(w_cur)

        ##: Integrate acceleration here to get the velocity for loss

        # # g = torch.tensor([0, 0, -9.81])
        # v_X, v_Y, v_Z = 0, 0, 0
        # # a_T = T[:, :, 0].mm(a_cur[:, 0].unsqueeze(dim=1)) + g.reshape(3, 1)
        # # v_curX = v_X + a_T[0] * 0.02
        # # v_curY = v_Y + a_T[0] * 0.02
        # # v_curZ = v_Z + a_T[0] * 0.02
        # # v_X = v_curX
        # # v_Y = v_curY
        # # v_Z = v_curZ
        # v_cur = torch.zeros((3, 100), dtype=torch.float32)
        # for i in range(0, 100):
        #     # v_cur = torch.tensor(torch.zeros_like(a_T), dtype=torch.float32).to(device)
        #     a_T = T[:, :, i].mm(a_cur[:, i].unsqueeze(dim=1)) + g.reshape(3, 1)
        #     v_curX = v_X + a_T[0] * 0.02
        #     v_curY = v_Y + a_T[0] * 0.02
        #     v_curZ = v_Z + a_T[0] * 0.02
        #     v_cur[0][i] = v_curX
        #     v_cur[1][i] = v_curY
        #     v_cur[2][i] = v_curZ
        #     v_X = v_curX
        #     v_Y = v_curY
        #     v_Z = v_curZ
        #
        # # self.v_X, self.v_Y, self.v_Z = v_X, v_Y, v_Z
        # v_cur = v_cur.to(device)
        print(a_cur.size())
        return output, T, a_cur, w_cur, theta_t, d_res[0:3, :]


    def loss(self, output, T, a_cur, w_cur, theta_t, p_tt):

        # a f_a, w f_w
        # a_Physics = (v - v_Be) / self.t
        # w_physics = (e_angle - e_angleBe) / self.t
        # error_a = torch.mean(torch.square(a_deri - a_Physics))
        # error_w = torch.mean(torch.square(w_deri - w_deri))
        g = torch.tensor([0, 0, -9.81]).to(device)
        # T = torch.tensor(np.array(T)).to(device)

        a_T = T[:, :, 0].mm(a_cur[:, 0].unsqueeze(dim=1)) + g.reshape(3, 1)
        ## T mul is wrong here
        for i in range(1, 100):
            a_T = torch.concat((a_T, T[:, :, i].mm(a_cur[:, i].unsqueeze(dim=1)) + g.reshape(3, 1)), 1)
        print(a_T.size())

        # theta_t = torch.transpose(theta_t, 0, 1)
        # p_tt = torch.transpose(p_tt, 0, 1)
        print(p_tt.size())
        print(w_cur.shape)
        error_a = torch.mean(torch.square(p_tt - a_cur))
        error_w = torch.mean(torch.square(theta_t - w_cur))
        return error_a + error_w

    def init_optimizer(self, pos):
        optimizer = optim.SGD(params=pos, lr=0.001, weight_decay=0)
        return optimizer


    def get_gradient(self, f, x):
        """ computes gradient of tensor f with respect to tensor x """
        assert x.requires_grad

        x_shape = x.shape
        f_shape = f.shape
        f = f.view(-1)

        x_grads = []
        for f_val in f:
            if x.grad is not None:
                x.grad.data.zero_()
            f_val.backward(retain_graph=True)  ### manuelly get corresponding x gradient
            if x.grad is not None:
                x_grads.append(deepcopy(x.grad.data))
            else:
                # in case f isn't a function of x
                x_grads.append(torch.zeros(x.shape).to(x))
        output_shape = list(f_shape) + list(x_shape)
        return torch.cat((x_grads)).view(output_shape)



def main():
    dataloader = loadData('Sensorsimulate.csv')
    pinn = PINN()
    ## pos = []
    ## out = pinn(pos)
    ## out.gradient()
    epochs = 1
    pinn.to(device=device)
    csv_writer = csv.writer(open('Output.csv', 'a'))

    for epoch in range(epochs):
        i = 1

        for batch in dataloader:
            t0 = time.time()

            output, T, a_cur, w_cur, theta_t, p_tt = pinn.net(batch, 100)
            print(111122)
            # output.cpu()
            batch_loss = pinn.loss(output, T, a_cur, w_cur, theta_t, p_tt)
            del(output)

            pinn.optimizer.zero_grad()
            batch_loss.backward()
            pinn.optimizer.step()
            t1 = time.time()
            print('Batch= %d, Loss= %.10f, Time= %.4f' % (i, batch_loss, t1 - t0))
            # print('Epoch= %d' % epoch)
            i += 1

        print('Epoch= %d' % epoch)
        i = 0



def loadData(path):
    dataReader = csv.reader(open(path, 'r'))

    next(dataReader)

    a0, a1, a2, w0, w1, w2 = [], [], [], [], [], []
    dataloader = []
    batch = []

    for i in range(0, 10):
        j = 0
        for row in dataReader:
            a0.append(float(row[0]))
            a1.append(float(row[1]))
            a2.append(float(row[2]))
            w0.append(float(row[3]))
            w1.append(float(row[4]))
            w2.append(float(row[5]))
            j = j + 1
            if j == 100:
                break
        batch.append(a0.copy())
        batch.append(a1.copy())
        batch.append(a2.copy())
        batch.append(w0.copy())
        batch.append(w1.copy())
        batch.append(w2.copy())
        a0.clear()
        a1.clear()
        a2.clear()
        w0.clear()
        w1.clear()
        w2.clear()
        # print(np.shape(batch))
        dataloader.append(batch.copy())
        batch.clear()

    return dataloader

def initBatch(batch, pos):
    # pos = torch.arange(0, 400, step=1, dtype=torch.float32)
    # pos = pos.unsqueeze(1)
    k = 20

    time_embed = torch.zeros(400, 40)
    for i in range(0, 400):
        for j in range(1, 20):
            time_embed[i][j - 1] = torch.sin(pos[i][0] / j)
        for j in range(k, k + 21):
            time_embed[i][j - 1] = torch.cos(pos[i][0] / j)

    print(time_embed)

    batch = np.transpose(batch)
    batch = torch.tensor(batch, dtype=torch.float32, requires_grad=True)
    batch = torch.cat((batch, time_embed), 1)
    print(batch.shape)
    return batch, time_embed


if __name__ == '__main__':
    main()

