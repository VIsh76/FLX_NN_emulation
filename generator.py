import numpy as np
from netCDF4 import Dataset


def GenerateConv(x, y, kernel,train_prop=0.9, header=['ql'], batch_size=16, seed=0):
    "Generate a batch, randomly for a convolution NN, using only variable in header"

    n  = x['Xdim'].shape[0]
    nt = int(n*train_prop) # id up to training
    p  = x['Ydim'].shape[0]
    lev= x['lev'].shape[0]

    xheader = header
    xheader.sort()

    while True:
        yshuffled   = np.arange(p) # y id but random
        xshuffled   = np.arange(nt)# x id but random

        #np.random.shuffle(yshuffled) #shuffling y divide the spead by 3
        np.random.shuffle(xshuffled)
        id_batches_x = 0 #counting the id for batches coordinates
        id_batches_y = 0

        while (id_batches_x) < nt+1:

            dataX = np.zeros((batch_size,lev,1)) # batch data
            idn   = xshuffled[id_batches_x]
            idp   = yshuffled[id_batches_y + np.arange(batch_size)]
            Y = y['flx'][0,:, idp, idn]
            Y = Y.swapaxes(0,1)

            for k in xheader:
                if(len(x[k].shape)==4):
                    a = x[k][:,:,idp,idn]
                    a = a.reshape(1,lev, -1)
                    a = a.swapaxes(0,2)
                elif(len(x[k].shape)==3):
                    a = x[k][:,idp,idn]
                    a = a.repeat(lev,1).reshape(1,-1 , lev)
                    a = a.swapaxes(0,1)
                    a = a.swapaxes(1,2)
                dataX = np.concatenate((dataX,a),axis=2)
            dataX = dataX[:,:,1:] # the first channel is full of 0
            kernel.apply(dataX, xheader)
            print()
            yield dataX,Y
            id_batches_y += batch_size
            if(id_batches_y+batch_size >= p):
                id_batches_y = 0
                id_batches_x+= 1

def GenerateDense(x, y, kernel, train_prop=0.9, header=["o3",'ql',"t","ts",'emis'], batch_size=16, Dense=False, seed=0):
    "Generate a batch, randomly for a dense NN, using only variable in header"
    n  = x['Xdim'].shape[0]
    nt = int(n*train_prop) # id up to training
    p  = x['Ydim'].shape[0]
    lev= x['lev'].shape[0]

    np.random.seed(seed)
    xheader = header
    xheader.sort()

    while True:
        yshuffled   = np.arange(p) # y id but random
        xshuffled   = np.arange(nt)# x id but random

        #np.random.shuffle(yshuffled) #shuffling y divide the spead by 3
        np.random.shuffle(xshuffled)
        id_batches_x = 0 #counting the id for batches coordinates
        id_batches_y = 0

        while (id_batches_x) < nt+1:

            dataX = np.zeros((batch_size, 1)) # batch data
            idn   = xshuffled[id_batches_x]
            idp   = yshuffled[id_batches_y + np.arange(batch_size)]
            Y = y['flx'][0,:, idp, idn]
            Y = Y.swapaxes(0,1)
            for k in xheader:
                if(len(x[k].shape)==4):
                    a = x[k][:,:,idp,idn]
                    a = a.reshape(lev, -1)
                    a = a.swapaxes(0,1)
                elif(len(x[k].shape)==3):
                    a = x[k][:,idp,idn]
                    a = a.swapaxes(0,1)
                dataX = np.concatenate((dataX,a),axis=1)

            dataX = dataX[:,1:] # the first column is full of zeros
            kernel.apply(dataX, xheader)
            yield dataX,Y
            id_batches_y += batch_size
            if(id_batches_y+batch_size >= p):
                id_batches_y = 0
                id_batches_x+= 1
