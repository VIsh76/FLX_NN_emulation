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

    @property
    def eliminated_vars(self):
        return []

    def __call__(self, x, headers=None):
        return x

    def fit(self, x):
        self.fitted = True
        return x

    def sub_vec(self,lev):
        return(np.zeros(lev))

    def prod_vec(self,lev):
        return(np.ones(lev))

    def __str__(self):
        return 'type : {} \nfitted : {} \nvalues : {} \n  '


class VarSuppression(Preprocess):
    """
    List of input variables are takken away from the input and unsed
    """
    def __init__(self, list_of_suppressed_vars):
        super().__init__()
        self._list_of_suppressed_vars = list_of_suppressed_vars

    @property
    def eliminated_vars(self):
        return(self._list_of_suppressed_vars)

    def __call__(self, x, header):
        id_var_used = [ header.index(var) for var in header if not var in self._list_of_suppressed_vars ]
        return(x[:,id_var_used])

    def __str__(self):
        return 'type : Var Supression \n values : {} \n'.format(self._list_of_suppressed_vars)

class SetToZero(Preprocess):
    """
    List of input variables are set to 0 and unsed for the training
    """
    def __init__(self, set_to_zero):
        super().__init__()
        self._list_of_suppressed_vars = list_of_suppressed_vars

    def __call__(self, x, header):
        id_to_zero = [ header.index(var) for var in header if var in self._list_of_suppressed_vars ]
        x[:,id_var_used]*=0
        return(x)


class Normalizer(Preprocess):
    """
    Transform the input into a variable of mean 0 and norm 1
    """
    def __init__(self):
        super().__init__()
        self.m = 0
        self.std = 1

    def __call__(self, x, headers=None):
        x -= self.m
        x /= self.std
        return x

    def fit(self, x):
        self.fitted=True
        self.m = np.mean(x)
        self.std = np.std(x)

    @property
    def params(self):
        return self.m, self.std

    def set_params(self, x, y):
        self.fitted=True
        self.m = x
        self.std = y

    def sub_vec(self, lev):
        return(np.zeros(lev)+self.m)

    def prod_vec(self, lev):
        return(np.ones(lev)/self.std)


    def __str__(self):
        return super().__str__().format("Normalizer", self.fitted, (self.m, self.std))


class Zero_One(Preprocess):
    """
    Match the input to a variable in [0,1]
    """
    def __init__(self):
        super().__init__()
        self.max = 1
        self.min = 1

    def __call__(self, x, headers=None):
        x -= self.min
        x /= (self.max-self.min)
        return x

    def fit(self, x):
        self.fitted=True
        self.min = np.min(x)
        self.max = np.max(x)

    def set_params(self, x, y):
        self.fitted = True
        self.min = x
        self.max = y

    def sub_vec(self, lev):
        return(np.zeros(lev)+self.min)

    def prod_vec(self, lev):
        return(np.ones(lev)/((self.max-self.min)))

    @property
    def params(self):
        return self.min, self.max

    def __str__(self):
        return super().__str__().format("Zero_One", self.fitted, (self.max, self.min))


class Binary(Preprocess):
    """
    Set value to 0 when x=min, 1 if x!=min
    """
    def __init__(self):
        super().__init__()
        self.min = 1

    def __call__(self, x, headers=None):
        x = (x - self.min) / x
        return x

    def fit(self, x):
        self.fitted=True
        self.min = np.min(x)
        self.max = np.max(x)

    def set_params(self, x, y):
        self.fitted = True
        self.min = x
        self.max = y

    @property
    def params(self):
        return self.min, self.max

    def __str__(self):
        return super().__str__().format("Zero_One", self.fitted, (self.max, self.min))


class Level_Normalizer(Preprocess):
    """
    Set the mean to 0 for every level
    """
    def __init__(self, use_renorm):
        super().__init__()
        self.L =0
        self.norm = 1
        self.use_renorm = use_renorm

    def __call__(self, x, headers=None):
        if self.fitted:
            x-= self.L/ self.norm
        else:
            x-= np.mean(x, axis=0)#.reshape(1,-1)
            norm = np.std(x, axis=0)[-1]
            if self.renorm:
                x/= norm
        return x

    def fit(self, x):
        self.L = np.mean(x, axis=0)
        self.fitted = True
        if self.renorm:
            self.norm = np.std(x, axis=0)[-1]

    def set_params(self, L):
        self.fitted=True
        self.L = L
        self.use_renorm = False

    @property
    def renorm(self):
        return self.use_renorm

    def sub_vec(self, lev):
        return np.zeros(lev) + self.L

    def prod_vec(self, lev):
        return np.ones(lev) / self.norm

    @property
    def params(self):
        return self.L, self.norm

    def __str__(self):
        return super().__str__().format("Level_Normalizer", self.fitted, self.L)

###################################### Dict preprocess class


class DictPrepross(Preprocess):
    """
    Dictionnary of Preprocess
    """
    def __init__(self, header, functions):
        super(DictPrepross, self).__init__()
        self.dict = dict()
        for i, h in enumerate(header):
            self.dict[h] = functions[i]

    def load_from_pd(self, pd_file):
        self.dict=dict()
        for i, k in enumerate(pd_file['Name']):
            self.dict[k] = eval(pd_file['Type'][i].split('.')[-1])()
            self.dict[k].set_params(pd_file['P1'][i], pd_file['P2'][i])

    def fitonGen(self, B, axis):
        """ Generator must generate entire file """
        header = B.variables
        if(axis==1):
            for k in self.dict.keys():
                id = header.index(k)
                x0 = B[0][0][:, id, :]
                for i,(x,y) in enumerate(B):
                    if i > 0:
                        x0 = np.concatenate( (x0, x[:, id, :]), axis=0)
                self[k].fit(x0)
        elif axis==2:
            for k in self.dict.keys():
                id = header.index(k)
                x0 = B[0][0][:, :, id]
                for i,(x,y) in enumerate(B):
                    if i > 0:
                        x0 = np.concatenate( (x0, x[:, :, id]), axis=0)
                self[k].fit(x0)
        else:
            assert(False)


    @property
    def new_vars(self):
        output = []
        for k in list(self.dict.keys()):
            output = output + self[k].new_vars
        return output

    def add(self, F, v, params=[0, 1]):
        self[v] = F
        self[v].set_params(params)

    def __call__(self, x, headers):
        for i, h in enumerate(headers):
            if h in self.dict.keys():
                x[:, i] = self[h](x[:,i])
        return x

    def call_on_pd(self,data_f):
        for h in self.dict.keys():
            data_f[h] = self[h](data_f[h])
        return data_f

    def __getitem__(self, k):
        return self.dict[k]

    def __len__(self):
        return len(self.dict)

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
        data = {'Name':Name, 'P1': P1, 'P2': P2, 'Type': Type}
        return pd.DataFrame(data)

################################### KERNEL class


class Kernel(Preprocess):
    def __init__(self, var):
        super(Kernel, self).__init__()
        self.vars = var

    @property
    def new_vars(self):
        return []

    def __call__(self, x, header):
        return x

    def __str__(self):
        return 'type : {} \nvariable : {} \nparams : {} \n  '


class ProdKernel(Kernel):
    def __init__(self, var=[]):
        Kernel.__init__(self, var)

    def __call__(self, x, header):
        """ header is the header of x"""
        for v1,v2 in self.vars:
            id1, id2 = header.index(v1), header.index(v2)
            xm = (x[:, id1, :]*x[:, id2, :]).reshape(x.shape[0], 1, x.shape[2])
            x = np.concatenate((x, xm), axis=1)
        return x

    def call_on_pd(self, dataf):
        for v1,v2 in self.vars:
            Mu=dataf[v1].mul(dataf[v2])
            Mu.columns = pd.MultiIndex.from_product([[str(v1)+'*'+str(v2)], Mu.columns])
            dataf = dataf.join(Mu)
        return dataf

    @property
    def new_vars(self):
        return [str(v1)+'*'+str(v2) for v1,v2 in self.vars]

    def __str__(self):
        return super().__str__().format("Prodkernel", self.vars, ' ')


class FKernel(Kernel):
    def __init__(self, func, var, gamma=1):
        Kernel.__init__(self, var=[])
        self.func = func
        self.gamma = gamma
        self.fname = str(func).split(' ')[1]
        self.vars = var

    def __call__(self, x, header):
        """ header is the header of x"""
        for i, v1 in enumerate(header):
            if v1 in self.vars:
                xm = self.func(self.gamma * x[:, i, :]).reshape(x.shape[0], 1, x.shape[2])
                x = np.concatenate((x, xm), axis=1)
        return x

    def call_on_pd(self, dataf):
        for var in self.vars:
            Mu=self.func(self.gamma*dataf[var])
            Mu.columns = pd.MultiIndex.from_product([[self.fname+str(var)], Mu.columns])
            dataf = dataf.join(Mu)
        return dataf

    @property
    def new_vars(self):
        return [self.fname+'_'+str(var) for var in self.vars]

    def __str__(self):
        return super().__str__().format("Fkernel", (self.vars, self.fname), self.gamma)

################### Dict Creation (to avoid recomputing)
