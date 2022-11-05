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

def integrateV():
    csv_reader = csv.reader(open('Sensorsimulate.csv', 'r'))
    next(csv_reader)
    a_x, a_y, a_z, w_x, w_y, w_z = [], [], [], [], [], []
    for row in csv_reader:
        a_x.append(row[0])
        a_y.append(row[1])
        a_z.append(row[2])
        w_x.append(row[3])
        w_y.append(row[4])
        w_z.append(row[5])

    delta_t = 0.02
    Q_before = np.array([[0], [0], [0], [1]])

    for i in range(0, len(a_x)):
        wx, wy, wz = w_x[i], w_y[i], w_z[i]
        anguToQ = [[0, -wx, -wy, -wz],
                   [wx, 0, wz, -wy],
                   [wy, -wz, 0, wx],
                   [wz, wy, -wx, 0]]
        qt = 0.5 * delta_t * np.dot(anguToQ, Q_before) + Q_before
        q0, q1, q2, q3 = qt[0][0], qt[1][0], qt[2][0], qt[3][0]
        Tbi = np.array([[1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                     [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
                     [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]])
        ax, ay, az = a_x[i], a_y[i], a_z[i]
        acce = np.array([[ax], [ay], [az]])



def main():
    generateSensors()


if __name__ == '__main__':
    main()

    # l = torch.zeros_like(y)
    # l[:, i] = 1.
    # d = torch.autograd.grad(y, x, retain_graph=True, grad_outputs=l)[0]  #dydx: (batch_size, input_dim)
    # dydx3 = torch.concat((dydx3, d.unsqueeze(dim=1)), dim=1)