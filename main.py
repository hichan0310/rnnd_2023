import matplotlib.pyplot as plt
import numpy as np
import followLine
from followLine import followline as follow
from imgToNumpy import img_array
from math import sin, cos
from regression import *

followLine.grad_rate = 0.8

conv_size = 9
image = img_array("graph.jpg", (200, 200), padding=conv_size)

color = [(i * 20, i * 20, i * 20) for i in range(9)]

index = 0

for x_ in range(conv_size, 200 + conv_size):
    for y_ in range(conv_size, 200 + conv_size):
        if image[x_][y_] == 1:
            result = []
            image[x_][y_] += 1

            # for line in image:
            #     print(*map(lambda a: '■' if a == 2 else ('□' if a == 1 else ' '), line), sep='')
            image[x_][y_] -= 1
            follow(
                result=result,
                image=image,
                position=(x_, y_),
                conv_size=conv_size,
                movement=1.5
            )
            n = len(result)
            print(n)
            if n<40:
                continue
            x = np.array(list(map(lambda a: -a[0], result))) / 10
            y = np.array(list(map(lambda a: a[1], result))) / 10
            # print(x, y)

            # a, b, c, d, e, f, g, h = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
            # line_x = lambda t: a[0] * t * t * t * t + b[0] * t * t * t + c[0] * t * t + d[0] * t + e[0] + f[
            #     0] * np.array(list(map(lambda tt: sin(g[0] * tt + h[0]), t)))
            # line_y = lambda t: a[1] * t * t * t * t + b[1] * t * t * t + c[1] * t * t + d[1] * t + e[1] + f[
            #     1] * np.array(list(map(lambda tt: sin(g[1] * tt + h[1]), t)))
            #
            # print(n)
            # lr = 0.00005
            # for j in range(10000):
            #     cost_x, cost_y = 0, 0
            #     for i in range(n):
            #         t = i / 20
            #         pred = line_x(np.array([t]))
            #         cost_x += (pred - x[i]) ** 2
            #         a[0] -= (pred - x[i]) * t * t * t * t * lr
            #         b[0] -= (pred - x[i]) * t * t * t * lr
            #         c[0] -= (pred - x[i]) * t * t * lr
            #         d[0] -= (pred - x[i]) * t * lr
            #         e[0] -= (pred - x[i]) * lr
            #         f[0] -= (pred - x[i]) * sin(g[0] * t + h[0]) * lr
            #         g[0] -= (pred - x[i]) * f[0] * t * cos(g[0] * t + h[0]) * lr
            #         h[0] -= (pred - x[i]) * f[0] * cos(g[0] * t + h[0]) * lr
            #         pred = line_y(np.array([t]))
            #         cost_y += (pred - y[i]) ** 2
            #         a[1] -= (pred - y[i]) * t * t * t * t * lr
            #         b[1] -= (pred - y[i]) * t * t * t * lr
            #         c[1] -= (pred - y[i]) * t * t * lr
            #         d[1] -= (pred - y[i]) * t * lr
            #         e[1] -= (pred - y[i]) * lr
            #         f[1] -= (pred - x[i]) * sin(g[1] * t + h[1]) * lr
            #         g[1] -= (pred - x[i]) * f[1] * t * cos(g[1] * t + h[1]) * lr
            #         h[1] -= (pred - x[i]) * f[1] * cos(g[1] * t + h[1]) * lr
            #     cost_x /= n
            #     cost_y /= n
            #     # if cost_x < 3:
            #     #     complete_x = True
            #     # if cost_y < 3:
            #     #     complete_y = True
            #     # if complete_x and complete_y:
            #     #     break
            #     print('\r'+'='*((j+1)//100)+' '*((9999-j)//100), cost_x, cost_y, end='')
            #     # print('.', end='')
            # t = np.array(list(range(n * 100))) / 100 / 20
            # plt.plot(line_x(t), line_y(t))



            l_x, l_y=regression(x, y, n)
            plt.plot(l_y(np.array([i for i in range(0, (n-1)*100)])/100),
                     l_x(np.array([i for i in range(0, (n-1)*100)])/100))
            # plt.plot(y, x)


plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.axis('off')
plt.savefig("regression_result.png", transparent=True)
plt.show()
