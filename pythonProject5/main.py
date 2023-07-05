import matplotlib.pyplot as plt
import os
def savePlot(plot, path, axis=False):
    file_lst = os.listdir(path)
    file_cnt = len(file_lst)

    plot.title('')
    plot.xlabel('')
    plot.ylabel('')
    if not axis:
        plot.axis('off')

    if file_cnt == 0:
        save_path = f'{path}/saved_plot.png'
    else:
        save_path = f'{path}/saved_plot({file_cnt}).png'

    plot.savefig(save_path, transparent=True)



# 그래프 데이터 생성
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 그래프 생성
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Test Plot')

savePlot(plt, './saved', axis=True)