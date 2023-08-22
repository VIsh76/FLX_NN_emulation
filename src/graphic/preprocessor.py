import matplotlib.pyplot as plt

def save_figs(f, path, save):
    if save:
        f.savefig(path)


def plot_series(variables, matrix, nb_plot=10, graph_path='', title='', save=False):
    # INPUT VAR
    f = plt.figure(figsize=(15, len(variables)))
    f.suptitle(title)
    for i, var in enumerate(variables):
        ax = f.add_subplot(1+len(variables)//4, 4, i+1)
        ax.plot(matrix[:nb_plot, :, i].T)
        ax.set_title(var)
    save_figs(f, f'{graph_path}/{title}.jpg', save=save)
    plt.show()
