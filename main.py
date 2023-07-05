import numpy as np
import math
from math import pi, sin, cos
import matplotlib.pyplot as plt

size = 6
conv_size = 2 * size + 1
different_range = pi
movement = 10
result = []
grad = pi / 3
rate=0.9
#%%
conv = np.array([list(map(lambda x: x+0.00001 if abs(x) <= different_range / 2 else 0,
                          list(map(lambda x: x if x < pi else x - 2 * pi,
                                   [(0 if x - size == 0 and y - size == 0
                                     else (
                                              -pi / 2 if x - size == 0 and y - size > 0
                                              else (
                                                  pi / 2 if x - size == 0 and y - size < 0
                                                  else ((-math.atan(
                                                      (y - size) / (
                                                              x - size))) if x - size > 0
                                                        else (pi - math.atan(
                                                      (y - size) / (
                                                              x - size))
                                                        ))
                                              )
                                          )
                                          - grad + 4 * pi) % (2 * pi)
                                    for x in range(conv_size)])))) for y in
                 range(conv_size)])
#%%
conv
#%%
conv = (conv * 180 / pi).astype('int')
#%%
print(conv)
#%%
from PIL import Image
import numpy as np
import math
from math import pi, sin, cos
import cv2
#%%
img = Image.open("img2.png").convert('L')
img=cv2.imread("img2.png")
#%%
a = np.array((list(map(lambda a: a[100:700], np.array(img)))))[50:500]
#%%
a = np.array(Image.fromarray(255 - a).resize((100, 100)))
#%%
for i in np.array(Image.fromarray(a).convert('1')): print(i)
#%%
b = np.array(list(map(lambda a: np.array(list(map(lambda b: 255 if b else 0, a))), np.array(Image.fromarray(a).convert('1')))))
#%%
Image.fromarray(a).convert('1')
#%%
Image.fromarray(b)
#%%
mini = (100, 0)
for i in range(len(b)):
    for ii in range(len(b[i])):
        if b[i][ii] == 255:
            mini = min((ii, i), mini)
mini
#%%
size = 6
conv_size = 2 * size + 1
different_range = pi*2/3
movement = 2
result = []

def followline(image, grad, position, flag=True):
    result.append(position)
    conv = np.array([list(map(lambda x: x+0.00001 if abs(x) <= different_range / 2 else 0,
                          list(map(lambda x: x if x < pi else x - 2 * pi,
                                   [(0 if x - size == 0 and y - size == 0
                                     else (
                                              -pi / 2 if x - size == 0 and y - size > 0
                                              else (
                                                  pi / 2 if x - size == 0 and y - size < 0
                                                  else ((-math.atan(
                                                      (y - size) / (
                                                              x - size))) if x - size > 0
                                                        else (pi - math.atan(
                                                      (y - size) / (
                                                              x - size))
                                                        ))
                                              )
                                          )
                                          - grad + 4 * pi) % (2 * pi)
                                    for x in range(conv_size)])))) for y in
                 range(conv_size)])
    conv_result = image[position[1] - size:position[1] + size + 1, position[0] - size:position[0] + size + 1] * conv
    num, sum = 0, 0
    for x in conv_result:
        for y in x:
            if y != 0:
                num += 1
                sum += y
    if num == 0:
        print("end")
        print((b[max(position[1] - size-5, 0):min(position[1] + size + 1+5, 99),
              max(position[0] - size-5, 0):min(position[0] + size + 1+5, 99)]/255).astype('int'))
        print((conv*100).astype('int'))
        return
    new_grad_difference = sum / num
    new_grad = grad + new_grad_difference*(rate if flag else 1)
    print(position, num, sum, grad, new_grad)
    new_position = (int(position[0] + movement * cos(new_grad) + 0.5), int(position[1] - movement * sin(new_grad) + 0.5))
    followline(image, new_grad, new_position, False)
#%%
followline(b / 255, pi / 2, (15, 27))
#%%
x=list(map(lambda a:a[0], result))
y=list(map(lambda a:-a[1], result))
x=np.array(x)
y=np.array(y)
#%%
plt.plot(x, y)
plt.show()
# =-=-=-=-=-=-=
#%%
# 양끝을 고정시키고 다항회귀를 하자
# a(x-x_1)f(x)+b(x-x_2)g(x)+c
# abc는 그냥 방정식으로 해결
# 이렇게 하고 미분 때려서 하면 가능함
# 매개변수 방정식으로 하면 해결되겠네
#%%
x=x/10
y=y/10
#%%
start, end=result[0], result[-1]
a, b, c, d, e, f, g, h=[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
line_x=lambda t:a[0]*t*t*t*t+b[0]*t*t*t+c[0]*t*t+d[0]*t+e[0]+f[0]*np.array(list(map(lambda tt:sin(g[0]*tt+h[0]), t)))
line_y=lambda t:a[1]*t*t*t*t+b[1]*t*t*t+c[1]*t*t+d[1]*t+e[1]+f[1]*np.array(list(map(lambda tt:sin(g[1]*tt+h[1]), t)))
#%%
n=len(x)
#%%
n
#%%
lr=0.0001
ii=0
complete_x, complete_y=False, False
for j in range(10000):
    cost_x, cost_y=0, 0
    for i in range(n):
        t=i/20
        if not complete_x:
            pred=line_x(np.array([t]))
            cost_x+=(pred-x[i])**2
            a[0]-=(pred-x[i])*t*t*t*t*lr
            b[0]-=(pred-x[i])*t*t*t*lr
            c[0]-=(pred-x[i])*t*t*lr
            d[0]-=(pred-x[i])*t*lr
            e[0]-=(pred-x[i])*lr
            f[0]-=(pred-x[i])*sin(g[0]*t+h[0])*lr
            g[0]-=(pred-x[i])*f[0]*t*cos(g[0]*t+h[0])*lr
            h[0]-=(pred-x[i])*f[0]*cos(g[0]*t+h[0])*lr
        if not complete_y:
            pred=line_y(np.array([t]))
            cost_y+=(pred-y[i])**2
            a[1]-=(pred-y[i])*t*t*t*t*lr
            b[1]-=(pred-y[i])*t*t*t*lr
            c[1]-=(pred-y[i])*t*t*lr
            d[1]-=(pred-y[i])*t*lr
            e[1]-=(pred-y[i])*lr
            f[1]-=(pred-x[i])*sin(g[1]*t+h[1])*lr
            g[1]-=(pred-x[i])*f[1]*t*cos(g[1]*t+h[1])*lr
            h[1]-=(pred-x[i])*f[1]*cos(g[1]*t+h[1])*lr
    if cost_x<3:
        complete_x=True
    if cost_y<3:
        complete_y=True
    if complete_x and complete_y:
        break
    print(j, ii, cost_x, cost_y, "                          ")

#%%
t=np.array(list(range(n*100)))/100/20
plt.plot(t, line_x(t))
plt.plot(t, line_y(t))
plt.show()
#%%
t=np.array(list(range(n*100)))/100/20
plt.plot(line_x(t), line_y(t))
plt.savefig("regression_result.png")
plt.show()