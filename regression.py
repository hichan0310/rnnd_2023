import matplotlib.pyplot as plt
import numpy as np
import followLine
from followLine import followline as follow
from imgToNumpy import img_array
from math import sin, cos


def regression(x, y, n):
    default_x = lambda t: (x[-1] - x[0]) / (n - 1) * t * 160 + x[0]
    default_y = lambda t: (y[-1] - y[0]) / (n - 1) * t * 160 + y[0]
    ax, bx, cx, dx = 0, 0, 0, 0
    ay, by, cy, dy = 0, 0, 0, 0
    add_line_x = lambda t: t * (t - (n - 1)/160) * (ax * t * t * t + bx * t * t + cx * t + dx)
    add_line_y = lambda t: t * (t - (n - 1)/160) * (ay * t * t * t + by * t * t + cy * t + dy)
    line_x = lambda t: default_x(t) + add_line_x(t)
    line_y = lambda t: default_y(t) + add_line_y(t)

    e = 10000
    lr = 0.01
    for i in range(e):
        cost_x, cost_y = 0, 0
        for j in range(n):
            t = j / 160
            # x
            pred = line_x(np.array([t]))
            c = pred[0] - x[j]
            ax -= t * (t - (n - 1)/160) * t * t * t * c * lr
            bx -= t * (t - (n - 1)/160) * t * t * c * lr
            cx -= t * (t - (n - 1)/160) * t * c * lr
            dx -= t * (t - (n - 1)/160) * c * lr
            cost_x += c * c

            # y
            pred = line_y(np.array([t]))
            c = pred[0] - y[j]
            ay -= t * (t - (n - 1)/160) * t * t * t * c * lr
            by -= t * (t - (n - 1)/160) * t * t * c * lr
            cy -= t * (t - (n - 1)/160) * t * c * lr
            dy -= t * (t - (n - 1)/160) * c * lr
            cost_y += c * c

        print('\r', cost_x / n, cost_y / n, end='')

    # plt.plot(np.array([i for i in range(n)]), x, color='black')
    # plt.plot(np.array([i for i in range(n)]), y, color='black')
    # plt.plot(np.array([i for i in range(n * 100)]) / 100, line_x(np.array([i for i in range(n * 100)]) / 100 / 160),
    #          color='blue')
    # plt.plot(np.array([i for i in range(n * 100)]) / 100, line_y(np.array([i for i in range(n * 100)]) / 100 / 160),
    #          color='red')
    # plt.show()
    print()
    return lambda t:line_x(t/160), lambda t:line_y(t/160)
