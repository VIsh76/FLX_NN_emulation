import numpy as np

class Kernel(object):
    def __init__(self, ids):
        self.ids = ids

    def apply(self, x):
        return(x)

class ProdKernel(Kernel):
    def __init__(self, ids=[]):
        Kernel.__init__(self,ids)
        self.ids=ids
        self.header=[]

    def apply(self,x0, xheader):
        x=x0.copy()
        for ids in self.ids:
            id1=np.where(ids[0]==xheader)[0]
            id2=np.where(ids[1]==xheader)[0]
            k = x[:,:,id1]*x[:,:,id2]
            xheader.append(ids[0]+'*'+ids[1])
            x = np.concatenate((k,x),axis=-1)
        return(x)

class FKernel(Kernel):
    def __init__(self, func, gamma=1, ids=[]):
        Kernel.__init__(self,ids)
        self.func = func
        self.fname = str(func).split(' ')[1]
        self.ids  = ids

    def apply(self,x0, xheader0):
        xheader0 = xheader.copy()
        self.header=[]
        x=x0.copy()
        for ids in self.ids:
            id1=np.where(ids[0]==xheader)[0]
            k = self.func(x[:,:,id1])
            xheader.append(fname+ids[0])
            x = np.concatenate((k,x),axis=-1)
        self.xheader=xheader
        return(x)

# EXAMPLES OF KERNELS
Pk = ProdKernel()
nexp = lambda x: np.exp(-x)
Funcf = FKernel(nexp)
