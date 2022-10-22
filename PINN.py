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
                 # input,
                 layers_size=torch.tensor([46, 20, 20, 20, 20, 20, 20, 20, 20, 6]),  ## input size 406
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

        # self.optimizer = optim.SGD(params= self.model.parameters(),
        #                           lr= 0.001,
        #                           weight_decay= 0.0)

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

    def net(self, batch, batch_size, jacobian_t, jacobian_tt):

        # input = [[]]
        # torch.tensor([x[:, 0], x[:, 1], x[:, 2], w[:, 0], w[:, 1], w[:, 2]])
        time_embed1, time_embed2 = [], []



        # pos = torch.tensor(init, dtype=torch.float32)
        # self.pos = torch.from_numpy(np.array([*range(400)], dtype=np.float32))


        # pos = torch.tensor([0], dtype=torch.float32, requires_grad=True)
        pos = torch.arange(0, 400, step=1, dtype=torch.float32)
        # for i in range(1, 400):
        #     pos = torch.concat((pos, torch.tensor([i], dtype=torch.float32)), 0)
        pos = pos.unsqueeze(1)

        # p_e = numpy.arange(0, 400)
        # time_e = numpy.zeros(400, 40)
        # for i in range(0, 400):
        #     for j in range(0, 20):
        #         time_e[i][j].data = torch.sin(p_e[i] / j)
        #     for j in range(k, k + 20):
        #         time_e[i][j].data = torch.cos(p_e[i] / j)


        k = 20

        # for p in range(0, batch_size):
        #     pos[p] = p
        # pos = Variable(pos, requires_grad=True)

        # row_embed1 = torch.empty()
        # row_embed2 = torch.empty()
        # first_embed = torch.empty()
        # second_embed = torch.empty()
        # time_embed = torch.empty()

        # time_embed = torch.tensor([[torch.sin(pos[0] / torch.pow(torch.tensor(100), 0))]], dtype=torch.float32, requires_grad=True)
        # time_embed = torch.sin(pos[0] / torch.pow(torch.tensor(100), 0))
        time_embed = torch.zeros(400, 40)
        for i in range(0, 400):
            for j in range(1, 20):
                time_embed[i][j - 1] = torch.sin(pos[i][0] / j)
            for j in range(k, k + 21):
                time_embed[i][j - 1] = torch.cos(pos[i][0] / j)

        print(time_embed)

        # for i in range(1, k):
        #     time_embed = torch.concat((time_embed, torch.sin(pos[0] / torch.pow(torch.tensor(100), 2 * i / 40))), 1)
        # for i in range(k, 2 * k):
        #     time_embed = torch.concat((time_embed, torch.cos(pos[0] / torch.pow(torch.tensor(100), 2 * i / 40))), 1)
        #
        #
        # for i in range(1, batch_size):
        #     time_start = torch.tensor([[torch.sin(pos[i] / torch.pow(torch.tensor(100), 0))]], dtype=torch.float32, requires_grad=True)
        #     for j in range(1, k):
        #         time_start = torch.concat((time_start, torch.sin(pos[i] / torch.pow(torch.tensor(100), 2 * j / 40))), 1)
        #     for j in range(k, k + 20):
        #         time_start = torch.concat((time_start, torch.cos(pos[i] / torch.pow(torch.tensor(100), 2 * j / 40))), 1)
        #     time_embed = torch.concat((time_embed, time_start), 0)


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

        batch = torch.tensor(batch, dtype=torch.float32, requires_grad=True)

        batch = torch.cat((batch, time_embed), 1)
        # batch = torch.transpose(batch, 0, 1)
        # print(np.shape(batch))
        self.input = batch

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
        # print(np.shape(T))
        # p_t = autograd.grad(p, pos, create_graph=True, )
        # p_tt = autograd.grad(p_t, pos, )

        grad_outputs_p = torch.ones(400, 1, dtype=torch.float32)
        first_row = torch.tensor([1, 0, 0, 0, 0, 0])
        second_row = torch.tensor([0, 1, 0, 0, 0, 0])
        third_row = torch.tensor([0, 0, 1, 0, 0, 0])
        four_row = torch.tensor([0, 0, 0, 1, 0, 0])
        five_row = torch.tensor([0, 0, 0, 0, 1, 0])
        six_row = torch.tensor([0, 0, 0, 0, 0, 1])
        one = torch.ones((6, 6))

        print(pos.shape)
        # print(time_embed.shape)

        output = torch.transpose(output, 0, 1)
        print(output[:, 1])
        d_res = torch.tensor([], dtype=torch.float32)
        tmp = torch.zeros(6)
        # output_t_x0 = autograd.grad(output[:, 0].unsqueeze(1), pos, torch.ones_like(torch.ones(400, 1)), retain_graph=True, create_graph=True)[1]
        # output_t = autograd.grad(output, pos, grad_outputs=torch.ones_like(grad_outputs_p), retain_graph=True, create_graph=True)

        jacobian_t = autograd.functional.jacobian(lambda l: (l), pos)
        print(jacobian_t)
        for t in range(0, 200):
            # tmp = output[:, t].backward(gradient=torch.ones_like(output[:, t]), retain_graph=True, create_graph=True)
            # l = torch.zeros_like(output[:, t])
            # l[t] = 1



            d0 = torch.autograd.grad(output[:, t], pos, grad_outputs=first_row, retain_graph=True, create_graph=True)[0]
            d1 = torch.autograd.grad(output[:, t], pos, grad_outputs=second_row, retain_graph=True)[0]
            d2 = torch.autograd.grad(output[:, t], pos, grad_outputs=third_row, retain_graph=True)[0]
            d3 = torch.autograd.grad(output[:, t], pos, grad_outputs=four_row, retain_graph=True)[0]
            d4 = torch.autograd.grad(output[:, t], pos, grad_outputs=five_row, retain_graph=True)[0]
            d5 = torch.autograd.grad(output[:, t], pos, grad_outputs=six_row, retain_graph=True)[0]
            tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] = d0[t], d1[t], d2[t], d3[t], d4[t], d5[t]
            d_res = torch.cat((d_res, tmp.unsqueeze(dim=1)), dim=1)
            print(d0)
            # print(d2)

        d_tt = torch.tensor([], dtype=torch.float32)
        for t in range(0, 400):
            d_tt0 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=first_row, retain_graph=True)[0]
            d_tt1 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=second_row, retain_graph=True)[0]
            d_tt2 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=third_row, retain_graph=True)[0]
            d_tt3 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=four_row, retain_graph=True)[0]
            d_tt4 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=five_row, retain_graph=True)[0]
            d_tt5 = torch.autograd.grad(d_res[:, t], pos, grad_outputs=six_row, retain_graph=True)[0]

            tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] = d_tt0[t], d_tt1[t], d_tt2[t], d_tt3[t], d_tt4[t], d_tt5[t]
            d_tt = torch.cat((d_tt, d_tt0.unsqueeze(dim=1)), dim=1)


        # print(output_t_x0)
        # output_t_x1 = autograd.grad(output[:, 1].unsqueeze(1), pos, torch.ones_like(torch.ones(400, 1)), retain_graph=True, create_graph=True)
        # output_t_x2 = autograd.grad(output[:, 2].unsqueeze(1), pos, torch.ones_like(torch.ones(400, 1)), retain_graph=True, create_graph=True)
        # print(output_t_x0[0])
        # output_t = torch.sum(self.get_gradient(output, pos), 1)  # torch.gradient()
        # output_t_x0[0].requires_grad = True
        # output_t_x1[0].requires_grad = True
        # output_t_x2[0].requires_grad = True


        # output_tt_x0 = autograd.grad(output_t_x0[0], pos, torch.ones_like(torch.ones(400, 1)), create_graph=True, retain_graph=True)
        # output_tt_x1 = autograd.grad(output_t_x1[0], pos, torch.ones_like(torch.ones(400, 1)), create_graph=True)
        # output_tt_x2 = autograd.grad(output_t_x2[0], pos, torch.ones_like(torch.ones(400, 1)), create_graph=True)
        # print(output_tt_x0)

        theta_t_w0 = autograd.grad(output[:, 0].unsqueeze(1), pos, torch.ones_like(torch.ones(400, 1)), retain_graph=True)
        theta_t_w1 = autograd.grad(output[:, 1].unsqueeze(1), pos, torch.ones_like(torch.ones(400, 1)), retain_graph=True)
        theta_t_w2 = autograd.grad(output[:, 2].unsqueeze(1), pos, torch.ones_like(torch.ones(400, 1)), retain_graph=True)


        # p_t = torch.sum(self.get_gradient(p, pos))
        # p_tt = torch.sum(self.get_gradient(p_t, pos))
        # p_tt = torch.sum(self.get_gradient(p_t, t))
        # p_tt = a_cur * T + g  # calculate loss method

        # p_tt = a * T - g
        # p_t = torch.sum(self.get_gradient(p, pos))

        # theta_t = autograd.grad(theta, pos, )
        print(theta_t_w0[0])

        # theta_t = output_t[0][:, 3:6]
        # p_tt = output_t[0][:, 0:3]
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

        p_tt = torch.concat((output_tt_x0[0], output_tt_x1[0], output_tt_x2[0]), 1)
        theta_t = torch.concat((theta_t_w0[0], theta_t_w1[0], theta_t_w2[0]), 1)

        a_cur = torch.tensor([[batch[0][0], batch[0][1], batch[0][2]]])
        w_cur = torch.tensor([[batch[0][3], batch[0][4], batch[0][5]]])
        # print(output_t)
        for k in range(1, 400):
            a_cur = torch.concat((a_cur, torch.tensor([[batch[k][0], batch[k][1], batch[k][2]]])), 0)
            w_cur = torch.concat((w_cur, torch.tensor([[batch[k][3], batch[k][4], batch[k][5]]])), 0)

        print(a_cur.size())

        # p_tt = p_tt - torch.mul(a_cur, T) - g
        # theta_t = theta_t -  w_cur
        return output, T, a_cur, w_cur, theta_t, p_tt

    def loss(self, output, T, a_cur, w_cur, theta_t, p_tt):

        # a f_a, w f_w
        # a_Physics = (v - v_Be) / self.t
        # w_physics = (e_angle - e_angleBe) / self.t
        # error_a = torch.mean(torch.square(a_deri - a_Physics))
        # error_w = torch.mean(torch.square(w_deri - w_deri))
        g = torch.tensor([0, 0, -9.81])
        # print(p_tt.size())
        T = torch.tensor(np.array(T))
        print(a_cur[0, :].reshape(3, 1))

        a_T = T[:, :, 0].mm(a_cur[0, :].reshape(3, 1)) + g.reshape(3, 1)

        for i in range(1, 400):

            a_T = torch.concat((a_T, T[:, :, i].mm(a_cur[i, :].reshape(3, 1)) + g.reshape(3, 1)), 1)
        print(a_T.size())

        error_a = torch.mean(torch.square(p_tt - a_T))
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

    # def train_normal(self, epochs):
    #
    #     t0 = time.time()
    #     input = self.input
    #     # input = torch.tensor(self.input)
    #
    #     for epoch in range(epochs):
    #         # u_hat, v_hat, p_hat, f_u, f_v = self.net(self.x, self.y, self.t)
    #         # loss_ = self.loss(self.u, self.v, u_hat, v_hat, f_u, f_v)
    #
    #         for batch in input:
    #             output, T, a_cur, w_cur, theta_t, p_tt = self.net(batch, 400)
    #
    #             loss_ = self.loss(output, T, a_cur, w_cur, theta_t, p_tt)
    #             loss_print = loss_
    #             self.optimizer.zero_grad()  # Clear gradients for the next mini-batches
    #
    #             loss_.backward()  # Backpropagation, compute gradients
    #
    #             self.optimizer.step()
    #             # t1 = time.time()
    #
    #             ### Training status
    #             print('Epoch %d, Loss= %.10f' % (epoch, loss_print))


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
    # pos = []
    # out = pinn(pos)
    # out.gradient()
    # pinn = PINN()
    epochs = 20

    for epoch in range(epochs):
            # u_hat, v_hat, p_hat, f_u, f_v = self.net(self.x, self.y, self.t)
            # loss_ = self.loss(self.u, self.v, u_hat, v_hat, f_u, f_v)

        for batch in dataloader:
            pos = torch.arange(0, 400, step=1, dtype=torch.float32, requires_grad=True)
            pos = pos.unsqueeze(1)
            batch = initBatch(batch, pos)

            output = pinn(batch)
            jacobian_t = autograd.functional.jacobian(lambda l: pinn(l), pos, create_graph=True)
            jacobian_tt = autograd.functional.jacobian(lambda l: pinn(l), pos)

            output, T, a_cur, w_cur, theta_t, p_tt = net(batch, 400)

            loss_ = loss(output, T, a_cur, w_cur, theta_t, p_tt)
            loss_print = loss_
            optimizer.zero_grad()  # Clear gradients for the next mini-batches

            loss_.backward()  # Backpropagation, compute gradients

            optimizer.step()
                # t1 = time.time()

                ### Training status
            print('Epoch %d, Loss= %.10f' % (epoch, loss_print))




    # def train_normal(self, epochs):
    #
    #     t0 = time.time()
    #     input = self.input
    #     # input = torch.tensor(self.input)
    #
    #     for epoch in range(epochs):
    #         # u_hat, v_hat, p_hat, f_u, f_v = self.net(self.x, self.y, self.t)
    #         # loss_ = self.loss(self.u, self.v, u_hat, v_hat, f_u, f_v)
    #
    #         for batch in input:
    #             output, T, a_cur, w_cur, theta_t, p_tt = self.net(batch, 400)
    #
    #             loss_ = self.loss(output, T, a_cur, w_cur, theta_t, p_tt)
    #             loss_print = loss_
    #             self.optimizer.zero_grad()  # Clear gradients for the next mini-batches
    #
    #             loss_.backward()  # Backpropagation, compute gradients
    #
    #             self.optimizer.step()
    #             # t1 = time.time()
    #
    #             ### Training status
    #             print('Epoch %d, Loss= %.10f' % (epoch, loss_print))

    pinn.train_normal(20)


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
    return batch


if __name__ == '__main__':
    main()

