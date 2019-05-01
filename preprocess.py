import numpy as np

class Preprocess(object):
    """
    Parent class of Preprocessing class
    """
    def __init__(self):
        self.fitted=False
        pass

    def apply(self,x):
        return x

    def fit(self,x):
        self.fitted=True
        return x

    def __str__(self):
        return 'type : {} \nfitted : {} \nvalues : {} \n  '

class Normalizer(Preprocess):
    """
    Transform the input into a variable of mean 0 and norm 1
    """
    def __init__(self):
        super().__init__()
        self.m = 0
        self.std = 1

    def apply(self,x):
        x-=self.m
        x/=self.std
        return x

    def fit(self,x):
        self.fitted=True
        self.m = np.mean(x)
        self.std = np.std(x)

    def __str__(self):
        return super().__str__().format("Normalizer", self.fitted, (self.m, self.std))

class Zero_One(Preprocess):
    """
    Match the input to a variable in [0,1]
    """
    def __init__(self):
        super().__init__()
        self.max=1
        self.min=1

    def apply(self,x):
        x-=self.min
        x/=(self.max-self.min)
        return x

    def fit(self,x):
        self.fitted=True
        self.min = np.min(x)
        self.max = np.max(x)

    def __str__(self):
        return super().__str__().format("Zero_One", self.fitted, (self.max, self.min))


###################################### Dict preprocess class

class DictPrepross(object):
    """
    Dictionnary of Preprocess
    """
    def __init__(self, header, functions):
        self.dict=dict()
        for i,h in enumerate(header):
            self.dict[h]=functions[i]

    def fitonNetCDF(self, data_in):
        for k in self.dict.keys():
            self[k].fit(data_in[k][:])

    def apply(self,x,h):
        if h in self.dict.keys():
            return self[h].apply(x)
        else:
            return x

    def __getitem__(self,k):
        return self.dict[k]

    def __str__(self):
        out =''
        for h in self.dict.keys():
            out = out + '{} : {}'.format(h, str(self[h]))+'\n'
        return out

################################### KERNEL class

class Kernel(object):
    def __init__(self, ids):
        self.ids = ids

    def apply(self, x, header):
        return x


class ProdKernel(Kernel):
    def __init__(self, ids=[]):
        Kernel.__init__(self, ids)
        self.ids = ids
        self.header = []

    def apply(self, x, header):
        for ids in self.ids:
            id1 = np.where(ids[0] == header)[0]
            id2 = np.where(ids[1] == header)[0]
            k = x[:, :, id1]*x[:, :, id2]
            header.append(ids[0]+'*'+ids[1])
            x = np.concatenate((k, x), axis=-1)
        return x


class FKernel(Kernel):
    def __init__(self, func, gamma=1, ids=[]):
        Kernel.__init__(self, ids)
        self.func = func
        self.gamma = gamma
        self.fname = str(func).split(' ')[1]
        self.ids = ids

    def apply(self, x, xheader):
        xheader0 = xheader.copy()
        self.header = []
        for ids in self.ids:
            id1 = np.where(ids[0] == xheader)[0]
            k = self.func(x[:, :, id1])
            xheader0.append(fname+ids[0])
            x = np.concatenate((k, x), axis=-1)
        self.xheader = xheader
        return x
