import csv
import matplotlib.pyplot as plt

n=int(input('가져올 데이터 최대 개수 입력 : '))
x, y = [], []
with open('simulationdata.csv', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    i=1
    for line in rdr:
        print('\r[' + '=' * int(150 * i / n) + '>' + '-' * (150 - int(150 * i / n)) + f'] : {i}/{n}', end='')
        i+=1
        x.append(int(line[0]))
        y.append(int(line[1]))
        if i>n:break

num=len(x)

divv = 20
data = [[0, 0] for i in range(max(x) // divv + 1)]
for i in range(num):
    data[x[i] // divv][0] += y[i]
    data[x[i] // divv][1] += 1

data1 = list(map(lambda aa: aa[0] / (aa[1] + 0.00001), data))
data2 = list(map(lambda aa: aa[1] / num, data))

plt.plot(list(range(len(data1))), data1)
plt.plot(list(range(len(data2))), data2)
plt.title(f'{num} data used')
plt.show()
