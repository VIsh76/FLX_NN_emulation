import numpy as np
from netCDF4 import Dataset
import time
from preprocess import DictPrepross

def generate_conv(x, y, kernels, train_prop=0.9, header=['ql'], batch_size=16, preprocess=DictPrepross([], []),
                  maxbatch=-1, test_data=False):
    """"
    Generate a batch, randomly for a convolution NN, using only variable in header

    x : input data in NetCDF format
    y : output data in NetCDF format
    kernels : list of kernsl
    train_prop : proportion of the training set
    header : variable used
    batch_size : size of the generated batch
    preprocess : DictPrepross object to apply to the data
    maxbatch : number of batch produced, maxbacth<0 means an infinite number of batch
    """

    maxbatch = int(maxbatch)
    n = x['Xdim'].shape[0]
    nt = int(n * train_prop)  # id up to training
    p = x['Ydim'].shape[0]
    lev = x['lev'].shape[0]

    x_header = header
    x_header.sort()

    nbatch = 0
    while nbatch != maxbatch:
        y_shuffled = np.arange(p)  # Ydim id to be shuffled
        # Xdim id but to be shuffled
        if test_data:
            x_shuffled =  np.arange(n - nt)  # test
            x_max = n  # Xdim not to outgrow
        else:
            x_shuffled = np.arange(nt)  # train
            x_max = nt  # Xdim not to outgrow

#        np.random.shuffle(y_shuffled) #shuffling y divide the speed by 3
        np.random.shuffle(x_shuffled)

        id_batches_x = 0  # counting the id for batches coordinates
        id_batches_y = 0
        while id_batches_x < x_max and nbatch != maxbatch:

            nbatch += 1
            data_x = np.zeros((batch_size, lev, 1))  # batch data
            idn = x_shuffled[id_batches_x]  # chosen indice in X_dim
            idp = y_shuffled[id_batches_y + np.arange(batch_size)]  # chosen indices in Y_dim

            Y = y['flx'][0, :, idp, idn]
            Y = Y.swapaxes(0, 1)
            for k in x_header:
                if len(x[k].shape) == 4:
                    a = x[k][:, :, idp, idn]
                    a = a.reshape(1, lev, -1)
                    a = a.swapaxes(0, 2)
                elif len(x[k].shape) == 3:
                    a = x[k][:, idp, idn]
                    a = a.repeat(lev, 1).reshape(1, -1, lev)
                    a = a.swapaxes(0, 1)
                    a = a.swapaxes(1, 2)
                a = preprocess.apply(a, k)
                data_x = np.concatenate((data_x, a), axis=2)
            data_x = data_x[:, :, 1:]  # the first channel is full of 0, thus eliminated
            for k in kernels:
                data_x = k.apply(data_x, x_header)

            yield data_x, Y

            id_batches_y += batch_size
            if id_batches_y + batch_size >= p:
                id_batches_y = 0
                id_batches_x += 1


def generate_dense(x, y, kernels, train_prop=0.9, header=["o3",'ql',"t","ts",'emis'], batch_size=16):
    """"Generate a batch, randomly for a dense NN, using only variable in header"""
    n = x['Xdim'].shape[0]
    nt = int(n*train_prop) # id up to training
    p = x['Ydim'].shape[0]
    lev = x['lev'].shape[0]

    x_header = header
    x_header.sort()

    while True:
        y_shuffled = np.arange(p)  # y id but random
        x_shuffled = np.arange(nt)  # x id but random

#        np.random.shuffle(y_shuffled) #shuffling y divide the spead by 3
        np.random.shuffle(x_shuffled)
        id_batches_x = 0  # counting the id for batches coordinates
        id_batches_y = 0

        while id_batches_x < nt+1:

            data_x = np.zeros((batch_size, 1))  # batch data
            idn = x_shuffled[id_batches_x]
            idp = y_shuffled[id_batches_y + np.arange(batch_size)]
            Y = y['flx'][0, :, idp, idn]
            Y = Y.swapaxes(0, 1)
            for k in x_header:
                if len(x[k].shape) == 4:
                    a = x[k][:, :, idp, idn]
                    a = a.reshape(lev, -1)
                    a = a.swapaxes(0, 1)
                elif len(x[k].shape) == 3:
                    a = x[k][:, idp, idn]
                    a = a.swapaxes(0, 1)
                data_x = np.concatenate((data_x, a), axis=1)

            data_x = data_x[:, 1:]  # the first column is full of zeros
            for k in kernels:
                data_x = k.apply(data_x, x_header)
            yield data_x, Y
            id_batches_y += batch_size
            if id_batches_y+batch_size >= p:
                id_batches_y = 0
                id_batches_x += 1

##################### Preprocessing classes
