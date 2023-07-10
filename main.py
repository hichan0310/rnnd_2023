import matplotlib.pyplot as plt
from filter import binary_filter
from PIL import Image
import numpy as np
import math
from math import pi, sin, cos
import cv2

rate=1
#%%
img=cv2.imread('graph.jpg', cv2.IMREAD_GRAYSCALE)
img=binary_filter(img)
#%%
a = img
#%%
a = np.array(Image.fromarray(255 - a).resize((200, 200)))
#%%
b = list(map(lambda a: list(map(lambda b: 1 if b else 0, a)), np.array(Image.fromarray(a).convert('1'))))
#%%
size = 9
conv_size = 2 * size + 1
different_range = pi
movement = 2
result = []
for i in range(len(a)):b[i]=b[i]+([0]*(size+movement))
b+=[[0]*(len(a)+size+movement)]*size

def followline(image:list[list[int]], position:tuple[int, int], grad:float=0, flag=True):
    global size, conv_size, different_range, movement, result
    result.append(position)
    if flag:
        t=different_range
        different_range=2*pi
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
    for x in (conv/pi*180).astype(int):
        print(list(x))
    for x in np.array(image)[position[0] - size:position[0] + size + 1,position[1] - size:position[1] + size + 1]:
        print(list(x))
    if flag:
        different_range=t
    try:
        conv_result = np.array(image)[position[1] - size:position[1] + size + 1,position[0] - size:position[0] + size + 1] * conv
        num, sum = 0, 0
        for x in conv_result:
            for y in x:
                if y != 0:
                    num += 1
                    sum += y
        if num == 0:
            print("end")
            return 0

        new_grad_difference = sum / num
        new_grad = grad + new_grad_difference

        print((position, new_grad / pi * 180), end='')
        new_position = (int(position[0] - movement * sin(new_grad) + 0.5), int(position[1] + movement * cos(new_grad) + 0.5))
        for aaa in range(position[1] - size//3, position[1] + size//3 + 1):
            for bbb in range(position[0] - size//3, position[0] + size//3 + 1):
                b[aaa][bbb]=1
        return 1+followline(image, new_position, new_grad, False)
    except:return 0


for xxxx in range(199, -1, -1):
    for yyyy in range(199, -1, -1):
        if b[xxxx][yyyy]==1:
            b[xxxx][yyyy]+=1
            for i in b: print(*map(lambda a: '■' if a==2 else ('□' if a==1 else ' '), i), sep='')
            b[xxxx][yyyy]-=1
            print(followline(b, (xxxx,yyyy), 0))
            #%%
            x=list(map(lambda a:a[0], result))
            y=list(map(lambda a:-a[1], result))
            #%%
            # 양끝을 고정시키고 다항회귀를 하자
            # a(x-x_1)f(x)+b(x-x_2)g(x)+c
            # abc는 그냥 방정식으로 해결
            # 이렇게 하고 미분 때려서 하면 가능함
            # 매개변수 방정식으로 하면 해결되겠네
            #%%
            #%%
            start, end=result[0], result[-1]
            aaa, bbb, ccc, ddd, eee, fff, ggg, hhh=[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
            line_x=lambda t:aaa[0]*t*t*t*t+bbb[0]*t*t*t+ccc[0]*t*t+ddd[0]*t+eee[0]+fff[0]*np.array(list(map(lambda tt:sin(ggg[0]*tt+hhh[0]), t)))
            line_y=lambda t:aaa[1]*t*t*t*t+bbb[1]*t*t*t+ccc[1]*t*t+ddd[1]*t+eee[1]+fff[1]*np.array(list(map(lambda tt:sin(ggg[1]*tt+hhh[1]), t)))
            #%%
            n=len(x)
            #%%
            n
            #%%
            lr=0.000001
            ii=0
            complete_x, complete_y=False, False
            for j in range(1000):
                cost_x, cost_y=0, 0
                for i in range(n):
                    t=i/20
                    if not complete_x:
                        pred=line_x(np.array([t]))
                        cost_x+=(pred-x[i])**2
                        aaa[0]-=(pred-x[i])*t*t*t*t*lr
                        bbb[0]-=(pred-x[i])*t*t*t*lr
                        ccc[0]-=(pred-x[i])*t*t*lr
                        ddd[0]-=(pred-x[i])*t*lr
                        eee[0]-=(pred-x[i])*lr
                        fff[0]-=(pred-x[i])*sin(ggg[0]*t+hhh[0])*lr
                        ggg[0]-=(pred-x[i])*fff[0]*t*cos(ggg[0]*t+hhh[0])*lr
                        hhh[0]-=(pred-x[i])*fff[0]*cos(ggg[0]*t+hhh[0])*lr
                    if not complete_y:
                        pred=line_y(np.array([t]))
                        cost_y+=(pred-y[i])**2
                        aaa[1]-=(pred-y[i])*t*t*t*t*lr
                        bbb[1]-=(pred-y[i])*t*t*t*lr
                        ccc[1]-=(pred-y[i])*t*t*lr
                        ddd[1]-=(pred-y[i])*t*lr
                        eee[1]-=(pred-y[i])*lr
                        fff[1]-=(pred-x[i])*sin(ggg[1]*t+hhh[1])*lr
                        ggg[1]-=(pred-x[i])*fff[1]*t*cos(ggg[1]*t+hhh[1])*lr
                        hhh[1]-=(pred-x[i])*fff[1]*cos(ggg[1]*t+hhh[1])*lr
                cost_x/=n
                cost_y/=n
                if cost_x<3:
                    complete_x=True
                if cost_y<3:
                    complete_y=True
                if complete_x and complete_y:
                    break
                print('\r', j, ii, cost_x, cost_y, "                          ", end='')
            #%%
            t=np.array(list(range(n*100)))/100/20
            plt.plot(line_x(t), line_y(t))
            result=[]
            print('end')
            input()
            print("start", xxxx, yyyy)
        else:pass
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.axis('off')
plt.savefig("regression_result.png", transparent=True)
plt.show()

