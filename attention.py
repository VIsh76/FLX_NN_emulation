import psutil
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from generator import Basic_Generator
from preprocess import Zero_One, Normalizer, DictPrepross
# Contain utilitary functions

def print_ram_usage():
    process = psutil.Process(os.getpid())
    ram_usage = round(process.memory_info().rss/float(2**30), 2)
    print("RAM usage: {}GB".format(ram_usage))


def Plot_Batch(x0, y0, header):
    f=plt.figure( figsize=(15,10), dpi=80)
    for i in range(11):
        ax= f.add_subplot(3,4,i+1)
        ax.set_title(header[i])
        for b in range(x0.shape[0]):
            ax.plot(np.flip(x0[b,i,:]), np.arange(len(x0[b,i,:])))
    ax= f.add_subplot(3,4,12)
    ax.set_title('flx')
    for b in range(y0.shape[0]):
        ax.plot(np.flip(y0[b]), np.arange(len(y0[b])))




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
