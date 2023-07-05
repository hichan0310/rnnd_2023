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
