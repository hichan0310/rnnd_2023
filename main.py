# import numpy as np
# import math
# from math import pi, sin, cos
#
# size = 4
# conv_size = 2 * size + 1
# different_range = pi
# movement=10
# result = []
#
#
# def getline(image):
#     image = np.pad(image, ((size, size), (size, size)), 'constant', constant_values=0)
#     # ************** 여기에서 초기 좌표, 초기 각도 알아내고 followline 호출해야 함 **************
#     #                    추가로 result에 새 리스트 생성해서 초기 좌표 넣어야 함
#
#
# def followline(image, grad, position):
#     result[-1].append(position)
#     conv = np.array([list(map(lambda x: x if abs(x) <= different_range / 2 else 0,
#                               list(map(lambda x: x if x < pi else x - 2 * pi,
#                                        [(0 if x - size == 0 and y - size == 0
#                                          else (
#                                                   -pi / 2 if x - size == 0 and y - size > 0
#                                                   else (
#                                                       pi / 2 if x - size == 0 and y - size < 0
#                                                       else (-math.atan(
#                                                           (y - size) / (
#                                                                   x - size)) if x - size > 0
#                                                             else pi - math.atan(
#                                                           (y - size) / (
#                                                                   x - size)))))
#                                               - grad + 4 * pi) % (2 * pi)
#                                         for x in range(conv_size)])))) for y in
#                      range(conv_size)])
#     conv_result = image[position[0] - size:position[0] + size + 1, position[1] - size:position[1] + size + 1] * conv
#     num, sum = 0, 0
#     for x in conv_result:
#         for y in x:
#             if y != 0:
#                 num += 1
#                 sum += y
#     if num == 0:
#         exit()
#     new_grad_difference = sum / num
#     new_grad = grad + new_grad_difference
#     new_position = (position[0] + movement * cos(new_grad), position[1] + movement * sin(new_grad))
#     followline(image, new_grad, new_position)


#
#
# n,m,k=map(int,input().split())
# if n-m<k-1 or m*k<n:print(-1);exit()
# a,x,y,z=[],n,m,k
# def f():
#  global a,x,y,z
#  while True:
#   if x-z>y-1:a=list(range(x,x-z,-1))+a;x-=z;y-=1
#   else:a=list(range(x,y-1,-1))+a;a=list(range(1,y))+a;break
# f()
# print(*a)
# a,x,y,z=[],n,k,m
# f()
# print(*map(lambda x:n-x+1,a))


import random
import csv

import matplotlib.pyplot as plt


class Person:
    def __init__(self, learningRate):
        self.trust = random.random() * 0.5 + 0.5
        self.learningRate = learningRate

    def choose(self) -> bool:
        return random.random() < self.trust

    def feedback(self, result) -> None:
        self.trust += result * self.learningRate
        self.trust = min(1.0, max(0.0, self.trust))


x, y = [], []
num = 70000000
for i in range(num):
    t_p1 = []
    t_p2 = []
    print('\r[' + '=' * int(150 * (i + 1) / num) + '>' + '-' * (150 - int(150 * (i + 1) / num)) + f'] : {i + 1}/{num}',
          end='')
    p1 = Person(0.003)
    p2 = Person(0.003)
    n = 0
    while not (p1.trust == p2.trust == 0 or p1.trust == p2.trust == 1):
        t_p1.append(p1.trust)
        t_p2.append(p2.trust)
        n += 1
        a, b = p1.choose(), p2.choose()
        if a and b:
            p1.feedback(+3)
            p2.feedback(+3)
        elif a and not b:
            p1.feedback(-6)
            p2.feedback(+2)
        elif not a and b:
            p1.feedback(+2)
            p2.feedback(-6)
        else:
            p1.feedback(-5)
            p2.feedback(-5)
    x.append(n)
    y.append(int(p1.trust))
    if n > 500 and p1.trust == 1:
        plt.plot(list(range(n)), t_p1)
        plt.plot(list(range(n)), t_p2)
        plt.show()
        exit()
    if (i + 1) % 1000000 == 0 or i + 1 == num:
        print()
        with open('simulationdata.csv', 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            for ind in range(len(x)):
                print('\rsave data : [' + '=' * int(100 * (ind + 1) / len(x)) + '>' + '-' * (
                        100 - int(100 * (ind + 1) / len(x))) + f'] : {ind + 1}/{len(x)}', end='')
                wr.writerow([x[ind], y[ind]])
            x, y = [], []
        print()





t_p1 = []
t_p2 = []
p1 = Person(0.003)
p2 = Person(0.003)
n = 0
while not (p1.trust == p2.trust == 0 or p1.trust == p2.trust == 1):
    t_p1.append(p1.trust)
    t_p2.append(p2.trust)
    n += 1
    a, b = p1.choose(), p2.choose()
    if a and b:
        p1.feedback(+3)
        p2.feedback(+3)
    elif a and not b:
        p1.feedback(-6)
        p2.feedback(+2)
    elif not a and b:
        p1.feedback(+2)
        p2.feedback(-6)
    else:
        p1.feedback(-5)
        p2.feedback(-5)
plt.plot(list(range(n)), t_p1)
plt.plot(list(range(n)), t_p2)
plt.show()