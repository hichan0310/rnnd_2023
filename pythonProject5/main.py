import matplotlib.pyplot as plt

import os

def CountDir(dir_path):
    file_lst = os.listdir(dir_path)
    file_cnt = len(file_lst)
    return file_cnt



# 그래프 데이터 생성
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 그래프 생성
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Test Plot')



# 그래프를 PNG 파일로 저장
if CountDir("./saved") == 0:
    save_path = f'./saved/saved_plot.png'
else:
    save_path = f'./saved/saved_plot({CountDir("./saved")}).png'

plt.savefig(save_path, transparent=True)
