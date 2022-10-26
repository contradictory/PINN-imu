import numpy as np
import csv

def generateSensors():
    t = 0.02
    w_x, w_y, w_z = 0, 0, 0
    a_x, a_y, a_z = 0, 0, 0

    fileSimulater = open('Sensorsimulate.csv', 'a', newline='')
    sensor_writer = csv.writer(fileSimulater)
    sensor_writer.writerow(['a_x', 'a_y', 'a_z', 'w_x', 'w_y', 'w_z'])

    for i in range(0, 10000):
        w_x = w_x + t * 0.01
        w_y = w_y + t * 0.03
        w_z = w_z + t * 0.02
        a_x = a_x + t * 0.02
        a_y = a_y + t * 0.01
        a_z = a_z + t * 0.004
        sensor_writer.writerow([a_x, a_y, a_z, w_x, w_y, w_z])
        print([a_x, a_y, a_z, w_x, w_y, w_z])


def main():
    generateSensors()


if __name__ == '__main__':
    main()

    # l = torch.zeros_like(y)
    # l[:, i] = 1.
    # d = torch.autograd.grad(y, x, retain_graph=True, grad_outputs=l)[0]  #dydx: (batch_size, input_dim)
    # dydx3 = torch.concat((dydx3, d.unsqueeze(dim=1)), dim=1)