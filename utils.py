import psutil
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from generator import Basic_Generator
from preprocess import Zero_One, Normalizer, DictPrepross


# Show ram usage
def print_ram_usage():
    process = psutil.Process(os.getpid())
    ram_usage = round(process.memory_info().rss/float(2**30), 2)
    print("RAM usage: {}GB".format(ram_usage))

# Plot Fonctions :
class F_and_plots:
    def __init__(self,shape, figsize=(15,10)):
        f, axes = plt.subplots(shape[0], shape[1], figsize=figsize)
        self.f = f
        self.axes = axes

    def __getitem__(self,i):
        return self.axes.flatten()[i]

def Plot_Batch(x, y0, header, swap=True):
    """ Plot the 11 variables of a batch"""
    f=plt.figure( figsize=(15,10), dpi=80)
    if swap:
        x0 = x.swapaxes(1,2).copy()
    else:
        x0 = x.copy()
    for i in range(len(header)):
        ax= f.add_subplot(3,4,i+1)
        ax.set_title(header[i])
        for b in range(x0.shape[0]):
            ax.plot(np.flip(x0[b,i,:]), np.arange(len(x0[b,i,:])))
    ax= f.add_subplot(3,4,12)
    ax.set_title('flx')
    for b in range(y0.shape[0]):
        ax.plot(np.flip(y0[b]), np.arange(len(y0[b])))


def Plot_Histograms(F, w, header):
    f= plt.figure(figsize=(15,10))
    for i in range(11):
        F[i].hist(-abs(w[i,:]), bins=50, cumulative=True)
        F[i].set_title(header[i])


def Plot_one_profile(y):
    plt.plot(np.flip(y), np.arrange(len(y[0])) )
    plt.show()


def Plot_triple_diff_separated(F,y,y0, header_y, sep=0,  lev=72, j = 0):
    f = plt.figure( figsize=(15,8) )
    for i in range(3):
        F[i].plot(np.flip(y[:,:,i].T[:,j]) , np.arange(lev))
        F[i].plot(np.flip(y0[:,:,i].T[:,j]) , np.arange(lev))
        F[i].legend(["truth", "pred"])
        F[i].set_title(header_y[i]+' full column')
        if(sep>0):
            F[i+3].plot(np.flip(y[:,:,i].T[sep:, j]) , np.arange(lev-sep))
            F[i+3].plot(np.flip(y0[:,:,i].T[sep:, j]) , np.arange(lev-sep))
            F[i+3].legend(["truth", "pred"])
            F[i+3].set_title(header_y[i]+' low layers')
            #
            F[i+6].plot(np.flip(y[:,:,i].T[:sep,j]) , sep+np.arange(sep))
            F[i+6].plot(np.flip(y0[:,:,i].T[:sep,j]) , sep+np.arange(sep))
            F[i+6].legend(["truth", "pred"])
            F[i+6].set_title(header_y[i]+' high layers')

########## GET DICTIONNARY :
fct = []
for i in range(5):
    fct.append(Zero_One())
for j in range(5):
    fct.append(Normalizer())
hd = ['rl', 'ri', 'ql', 'qi', 'q', 'ts', 't', 'emis', 'o3', 'pl']

def Load_FLX_dict(header_dict = hd , path='DictPreprocess_fit.hdf5' , fct=fct):
    if os.path.isfile(path):
        Dhd = pd.read_hdf(path, key='s')
        D = DictPrepross([], [])
        D.load_from_pd(Dhd)
    else:
        print("Fitting Dict")
        B = Basic_Generator(data_folder)
        xdim, ydim = B.Xdim, B.Ydim
        B = Basic_Generator(data_folder, batch_size=xdim*ydim, shuffle=False)
        D = DictPrepross(header_dict, fct)
        D.fitonGen(B)
        Dhd = D.to_array_save()
        Dhd.to_hdf(path, key='s')
    return(D)
