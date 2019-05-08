import numpy as np
import pandas as pd

class Preprocess(object):
    """
    Parent class of Preprocessing class
    """
    def __init__(self):
        self.fitted=False

    @property
    def new_vars(self):
        return []

    def __call__(self,x):
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

    def __call__(self,x):
        x-=self.m
        x/=self.std
        return x

    def fit(self,x):
        self.fitted=True
        self.m = np.mean(x)
        self.std = np.std(x)

    @property
    def params(self):
        return self.m, self.std

    def set_params(self, x,y):
        self.fitted=True
        self.m = x
        self.std = y

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

    def __call__(self,x):
        x-=self.min
        x/=(self.max-self.min)
        return x

    def fit(self,x):
        self.fitted=True
        self.min = np.min(x)
        self.max = np.max(x)

    def set_params(self, x,y):
        self.fitted=True
        self.min = x
        self.max = y

    @property
    def params(self):
        return self.min, self.max

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

    def load_from_pd(self, pd_file):
        self.dict=dict()
        for i,k in enumerate(pd_file['Name']):
            self.dict[k] = eval(pd_file['Type'][i].split('.')[-1])()
            self.dict[k].set_params( pd_file['P1'][i], pd_file['P2'][i] )

    def fitonGen(self, B):
        """ Generator must generate entire file """
        for k in self.dict.keys():
            print(k)
            x0 = np.array([0,1])
            for _,_ in B:
                x = np.array(B.X.copy()[k])
                x = x.flatten()
                x0 = np.hstack((x0,x))
                self[k].fit(x[1:])

    def add(self, F, v , params=[0,1]):
       self[v] = F
       self[v].set_params(params)


    def __call__(self,data_f):
        for h in self.dict.keys():
            data_f[h] = self[h](data_f[h])
        return(data_f)

    def __getitem__(self,k):
        return self.dict[k]

    def __len__(self):
        print(len(self.dict))

    def __str__(self):
        out =''
        l = list(self.dict.keys())
        l.sort()
        for h in l:
            out = out + '{} : {}'.format(h, str(self[h]))+'\n'
        return out

    def to_array_save(self):
        Type = []
        Name = []
        P1 = []
        P2 = []
        for k in self.dict:
            Name.append(k)
            Type.append(str(type(self[k]))[1:-2])
            P1.append(self[k].params[0])
            P2.append(self[k].params[1])
        data = {'Name':Name, 'P1':P1, 'P2': P2, 'Type':Type}
        return pd.DataFrame(data)


################################### KERNEL class

class Kernel(Preprocess):
    def __init__(self, var):
        super(Kernel, self).__init__()
        self.vars = var

    @property
    def new_vars(self):
        return([])

    def __call__(self, dataf):
        return dataf

class ProdKernel(Kernel):
    def __init__(self, var=[]):
        Kernel.__init__(self, var)

    def __call__(self, dataf):
        for v1,v2 in self.vars:
            Mu=dataf[v1].mul(dataf[v2])
            Mu.columns = pd.MultiIndex.from_product([[str(v1)+'*'+str(v2)], Mu.columns])
            dataf = dataf.join(Mu)
        return dataf

    @property
    def new_vars(self):
        return( [str(v1)+'*'+str(v2) for v1,v2 in self.vars])

class FKernel(Kernel):
    def __init__(self, func, gamma=1, var=[]):
        Kernel.__init__(self, var)
        self.func = func
        self.gamma = gamma
        self.fname = str(func).split(' ')[1]

    def __call__(self, dataf):
        for var in self.vars:
            Mu=self.func(self.gamma*dataf[var])
            Mu.columns = pd.MultiIndex.from_product( [[self.fname+str(var)], Mu.columns])
            dataf = dataf.join(Mu)
        return dataf

    @property
    def new_vars(self):
        return( [self.fname+str(var) for var in self.vars] )
