import csv
import math

import numpy
import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import time
from copy import deepcopy
from torch.autograd import Variable

# import torch.nn.functional as F

class PINN(nn.Module):
    '''
    Neural Network Class
    net_layer: list with the number of neurons for each network layer, [n_imput, ..., n_output]
    '''

    def __init__(self,
                 input,
                 layers_size=[46, 20, 20, 20, 20, 20, 20, 20, 20],  ## input size 406
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
        self.input = input

        #### Initializing neural network

        self.layers = nn.ModuleList()
        #### Initialize parameters (we are going to learn lambda)
        # self.lambda_1 = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.lambda_2 = nn.Parameter(torch.randn(1, requires_grad=True))

        for k in range(len(layers_size) - 1):
            self.layers.append(nn.Linear(layers_size[k], layers_size[k + 1]))

        # Output layer
        self.out = nn.Linear(layers_size[-1], out_size)

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

        # self.optimizer = optim.SGD(params= self.model.parameters(),
        #                           lr= 0.001,
        #                           weight_decay= 0.0)

    #### Forward pass

    def forward(self, x):

        for layer in self.layers:
            # Activation function
            x = torch.tanh(layer(x))

        # Last layer: we could choose a different functionsoftmax
        # output= F.softmax(self.out(x), dim=1)
        output = self.out(x)

        return output

    #### Net NS

    def net(self, batch, batch_size):

        # input = [[]]
        # torch.tensor([x[:, 0], x[:, 1], x[:, 2], w[:, 0], w[:, 1], w[:, 2]])
        time_embed1, time_embed2 = [], []

        init = [0]
        for k in range(1, 400):
            init.append(k)

        # pos = torch.tensor(init, dtype=torch.float32)
        pos = torch.tensor([0], dtype=torch.float32)
        for i in range(1, 400):
            pos = torch.concat((pos, torch.tensor([i], dtype=torch.float32)), 0)

        k = 20

        # for p in range(0, batch_size):
        #     pos[p] = p
        pos = Variable(pos, requires_grad=True)

        # row_embed1 = torch.empty()
        # row_embed2 = torch.empty()
        # first_embed = torch.empty()
        # second_embed = torch.empty()
        # time_embed = torch.empty()

        time_embed = torch.tensor([[torch.sin(pos[0] / torch.pow(torch.tensor(100), 0))]], dtype=torch.float32, requires_grad=False)
        print(torch.sin(pos[3]))

        for i in range(1, k):
            time_embed = torch.concat((time_embed, torch.tensor([[torch.sin(pos[0] / torch.pow(torch.tensor(100), 2 * i / 40))]], dtype=torch.float32)), 1)
        for i in range(k, 2 * k):
            time_embed = torch.concat((time_embed, torch.tensor([[torch.cos(pos[i] / torch.pow(torch.tensor(100), 2 * i / 40))]], dtype=torch.float32)), 1)


        for i in range(1, batch_size):
            time_start = torch.tensor([[torch.sin(pos[i] / torch.pow(torch.tensor(100), 0))]], dtype=torch.float32, requires_grad=False)
            for j in range(1, k):
                time_start = torch.concat((time_start, torch.tensor([[torch.sin(pos[i] / torch.pow(torch.tensor(100), 2 * j / 40))]], dtype=torch.float32)), 1)
            for j in range(k, k + 20):
                time_start = torch.concat((time_start, torch.tensor([[torch.cos(pos[i] / torch.pow(torch.tensor(100), 2 * j / 40))]], dtype=torch.float32)), 1)
            time_embed = torch.concat((time_embed, time_start), 0)


        # print(time_embed)

        # for i in range(0, batch_size):
        #     # input[i] = torch.tensor([x[i][0], x[i][1], x[i][2], w[i][0], w[i][1], w[i][2]])
        #     # time_embed1.append(np.sin(pos / math.pow(100, 2 * pos / batch_size)))
        #     # time_embed2.append(np.cos(pos / math.pow(100, 2 * pos / batch_size)))
        #     # pos[i][0] = torch.sin(torch.tensor(i / math.pow(100, 2 * i / batch_size), dtype=torch.float32))
        #     # pos[i][1] = torch.cos(torch.tensor(i / math.pow(100, 2 * i / batch_size), dtype=torch.float32))
        #
        #     for j in range(0, k):
        #         first_embed = torch.concat((row_embed1, torch.sin(torch.tensor(pos[i] / math.pow(100, 2 * j / 40)))), 1)
        #         second_embed = torch.concat((row_embed2, torch.cos(torch.tensor(pos[i] / math.pow(100, 2 * (20 + j) / 40)))), 1)
        #
        #     # if i == 0:
        #     #     time_embed = torch.concat((first_embed, second_embed), 1)
        #     tmp_embed = torch.concat((first_embed, second_embed), 1)
        #     if i == 0:
        #         time_embed = tmp_embed
        #     time_embed = torch.concat((time_embed, tmp_embed), 0)



        # pos = Variable(pos, requires_grad=True)

        ### sin, cos use torch sin cos
        # batch.append(time_embed1)
        # batch.append(time_embed2)


        # print(np.shape(batch))
        batch = np.transpose(batch)
        batch = torch.tensor(batch, dtype=torch.float32, requires_grad=False)

        batch = torch.cat((batch, time_embed), 1)
        # print(np.shape(batch))

        output = self.forward(batch)
        # concatenate t from here
        # t = [sin(pos / 100 ^ (2 * i / batch_size)),
        #      cos(pos / 100 ^ (2 * i / batch_size))]

        # # for i in range(0, len(x)):
        #     u[i].append(sin(pos / 100 ^ (2 * i / batch_size)))
        #     u[i].append(cos(pos / 100 ^ (2 * i / batch_size)))

        # input.append(np.sin(pos / 100 ^ (2 * pos / batch_size)))
        # input.append(np.cos(pos / 100 ^ (2 * pos / batch_size)))

        # output = torch.tensor(1, 8)
        # output[6] = np.sin(pos / 100 ^ (2 * pos / batch_size))
        # output[7] = np.cos(pos / 100 ^ (2 * pos / batch_size))

        # p = []
        # theta = []
        # for j in range(0, len(u)):
        #     p.append([output[j][0], output[j][1], output[j][2], output[j][6], output[j][7]])
        #     theta.append([output[j][3], output[j][4], output[j][5], output[j][6], output[j][7]])
        # p = [output[:, 0], output[:, 1], output[:, 2], output[:, 6], output[:, 7]]
        # theta = [output[:, 3], output[:, 4], output[:, 5], output[:, 6], output[:, 7]]

        # p = [output[:, 0], output[:, 1], output[:, 2], output[:, 6], output[:, 7]]
        # theta = [output[:, 3], output[:, 4], output[:, 5], output[:, 6], output[:, 7]]
        output_scalar = output.clone().detach().numpy()
        p = [output_scalar[:, 0], output_scalar[:, 1], output_scalar[:, 2]]
        theta = [output_scalar[:, 3], output_scalar[:, 4], output_scalar[:, 5]]

        theta_x = theta[0]
        theta_y = theta[1]
        theta_z = theta[2]


        g = [0, 0, -9.81]

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
        # a = [x[0], x[1], x[2]]
        # a0, a1, a2 = x[0], x[1], x[2]
        # wx, wy, wz = x[3], x[4], x[5]
        # w_cur = [(euler_angleX - euler_angleXBe) / dt, (euler_angleY - euler_angleYBe) / dt, (euler_angleZ - euler_angleZBe) / dt]

        # euler_angleX.backward()
        # w_cur_x = t.grad
        # euler_angleY.backward()
        # w_cur_y = t.grad
        # euler_angleZ.backward()
        # w_cur_z = t.grad

        ### wrong :  # v / dt == a

        # p position
        ### pos here ???
        print(np.shape(output_scalar))
        # p_t = autograd.grad(p, pos, create_graph=True, )
        # p_tt = autograd.grad(p_t, pos, )
        output_t = autograd.grad(output, pos, torch.ones_like(torch.ones(400, 6)), create_graph=True, allow_unused=True)

        # output_tt = autograd.grad(output_t, pos, torch.ones(400, 6))
        # p_t = torch.sum(self.get_gradient(p, pos))
        # p_tt = torch.sum(self.get_gradient(p_t, pos))
        # p_tt = torch.sum(self.get_gradient(p_t, t))
        # p_tt = a_cur * T + g  # calculate loss method

        # p_tt = a * T - g
        # p_t = torch.sum(self.get_gradient(p, pos))

        # theta_t = autograd.grad(theta, pos, )
        print(output_t)
        theta_t = output_t[:, 3:6]
        p_tt = output_t[:, 0:3]
        # theta_t = torch.sum(self.get_gradient(theta, pos))
        # theta_t = w  # calculate loss method

        # T = w

        # p , theta = f(a, w, t)

        #### T => Euler Angle
        # w = T / dt

        # zheli w bei chongyong de, chongfu youguanxi ma
        # ### dei dui shijian qiudao, cong P he W dechu acce he w
        # f_a
        # f_w

        a_cur = torch.tensor(batch[0][-1], batch[1][-1], batch[2][-1])
        w_cur = torch.tensor(batch[3][-1], batch[4][-1], batch[5][-1])

        p_tt = p_tt - torch.mul(a_cur, T) - g
        theta_t = theta_t -  w_cur

        return output, T, a_cur, w_cur, theta_t, p_tt

    def loss(self, output, T, a_cur, w_cur, theta_t, p_tt):

        # a f_a, w f_w
        # a_Physics = (v - v_Be) / self.t
        # w_physics = (e_angle - e_angleBe) / self.t
        # error_a = torch.mean(torch.square(a_deri - a_Physics))
        # error_w = torch.mean(torch.square(w_deri - w_deri))
        g = [0, 0, -9.81]
        error_a = torch.mean(torch.square(p_tt - a_cur * T - g))
        error_w = torch.mean(torch.square(theta_t - w_cur))
        return error_a + error_w

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
            f_val.backward(retain_graph=True)
            if x.grad is not None:
                x_grads.append(deepcopy(x.grad.data))
            else:
                # in case f isn't a function of x
                x_grads.append(torch.zeros(x.shape).to(x))
        output_shape = list(f_shape) + list(x_shape)
        return torch.cat((x_grads)).view(output_shape)

    def train_normal(self, epochs):

        t0 = time.time()
        input = self.input
        # input = torch.tensor(self.input)

        for epoch in range(epochs):
            # u_hat, v_hat, p_hat, f_u, f_v = self.net(self.x, self.y, self.t)
            # loss_ = self.loss(self.u, self.v, u_hat, v_hat, f_u, f_v)

            for batch in input:
                output, T, a_cur, w_cur, theta_t, p_tt = self.net(batch, 400)

                loss_ = self.loss(output, T, a_cur, w_cur, theta_t, p_tt)
                loss_print = loss_
                self.optimizer.zero_grad()  # Clear gradients for the next mini-batches

                loss_.backward()  # Backpropagation, compute gradients

                self.optimizer.step()
                # t1 = time.time()

                ### Training status
                print('Epoch %d, Loss= %.10f' % (epoch, loss_print))


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
    pinn = PINN(dataloader)

    pinn.train_normal(20)


def loadData(path):
    dataReader = csv.reader(open(path, 'r'))

    next(dataReader)

    a0, a1, a2, w0, w1, w2 = [], [], [], [], [], []
    dataloader = []
    batch = []

    for i in range(0, 25):
        j = 0
        for row in dataReader:
            a0.append(float(row[0]))
            a1.append(float(row[1]))
            a2.append(float(row[2]))
            w0.append(float(row[3]))
            w1.append(float(row[4]))
            w2.append(float(row[5]))
            j = j + 1
            if j == 400:
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





if __name__ == '__main__':
    main()
