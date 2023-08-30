import numpy as np
import pandas as pd

# USED PREPROCESSORS :

class Preprocess(object):
    """
    Parent class of Preprocessing class
    """
    def __init__(self, input_var, output_var=None, supress=False, inplace=False):
        self.fitted=False
        self.input_var = input_var
        if output_var is None:
            output_var = input_var
        self.output_var = output_var
        self.inplace = inplace
        self.supress = supress        
        # Check integrity
        if not self.inplace:
            assert(self.output_var != self.input_var) # var names must ne different
        if self.supress:
            assert(not self.inplace) # if the var is suppress do not replace it inplace

    def __call__(self, x, headers=None):
        """Return a new vector fitted
        """
        return self.__call__( x.copy() )

    def fit(self, x):
        """X is size (None, lev)

        Args:
            x (_type_): _description_
        """
        self.fitted = True

    def apply(self, x):
        return x

    def sub_vec(self,lev):
        return(np.zeros(lev))

    def prod_vec(self,lev):
        return(np.ones(lev))

    @property
    def params(self):
        return []

    def __str__(self):
        return 'type : {} \nfitted : {} \nvalues : {} \n'


class Normalizer(Preprocess):
    """
    Transform the input into a variable of mean 0 and norm 1
    """
    def __init__(self,  input_var, output_var=None):
        super().__init__(input_var, output_var, supress=False, inplace=True)
        self.m = 0
        self.std = 1

    def __call__(self, x, headers=None):
        return (x - self.m) / self.std

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

    def __str__(self):
        return super().__str__().format("Normalizer", self.fitted, (self.m, self.std))


class Zero_One(Preprocess):
    """
    Match the input to a variable in [0,1] uses for cloud variables
    """
    def __init__(self,  input_var, output_var=None):
        super().__init__(input_var, output_var, supress=False, inplace=True)
        self.max = 1
        self.min = 1

    def __call__(self, x, headers=None):
        return (x-self.min) / (self.max - self.min)
    
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
    Substract the mean of each level
    If renorm is true, the lower level is 
    """
    def __init__(self,  input_var, output_var=None, normalisation_method='no'):
        """Construct a Level Normaliser

        Args:
            use_renorm (str): 
            - Anything or no : no normalisation (norm is one)
            - 'surface' : division by surface std value (usefull for pressure)
            - 'top' : division by surface std value (usefull for o3)
            - 'abs' : divided by mean of abs values(usefull for o3)
            - 'std' : divided by the std of the level
        """
        super().__init__(input_var, output_var, supress=False, inplace=True)
        self.L = 0. # becomes array of size lev when fitted
        self.norm = 1. # becomes arry of size when fitted
        self.normalisation_method = normalisation_method # how to normalize 

    def __call__(self, x, headers=None):
        if self.fitted:
            return (x - self.L) / self.norm
        else:
            return x

    def fit(self, x):
        self.L = np.mean(x, axis=0)
        self.fitted = True
        if self.normalisation_method == 'surface':
            self.norm = np.std(x, axis=0)[-1]
        elif self.normalisation_method == 'top':
            self.norm = np.std(x, axis=0)[0]
        elif self.normalisation_method == 'abs':
            self.norm = np.std(abs(x), axis=0)[-1] # last level
        elif self.normalisation_method == 'std':
            self.norm = np.std(x, axis=0)
        self.norm = np.maximum(self.norm, 0.001)

    def set_params(self, L):
        self.fitted=True
        self.L = L
        self.use_renorm = False

    @property
    def params(self):
        return self.L, self.norm

    def __str__(self):
        return super().__str__().format("Level_Normalizer", self.fitted, self.L)
    

class Rescaler(Level_Normalizer):
    """
    Just perform rescaling (set the Normalizer mean to 0)
    """
    def __init__(self, input_var, output_var=None, normalisation_method=False):
        super().__init__(input_var, output_var, normalisation_method=normalisation_method)

    def fit(self, x):
        super().fit(x)
        self.m = 0

    @property
    def params(self):
        return self.norm

    def __str__(self):
        return super().__str__().format("Rescaler", self.fitted, self.params)


class Log_Level_Normalizer(Preprocess):
    """
    Substract the mean of each level
    If renorm is true, the lower level is 
    """
    def __init__(self,  input_var, output_var=None, normalisation_method='no', inplace=False, supress=False):
        """Construct a Level Normaliser

        Args:
            use_renorm (str): 
            - Anything or no : no normalisation (norm is one)
            - 'surface' : division by surface std value (usefull for pressure)
            - 'top' : division by surface std value (usefull for o3)
            - 'abs' : divided by mean of abs values(usefull for o3)
        """
        super().__init__( input_var, output_var, inplace=inplace, supress=supress)
        self.L = 0. # becomes array of size lev when fitted
        self.norm = 1. # always float 
        self.normalisation_method = normalisation_method # if renorm is true, 

    def __call__(self, x, headers=None):
        if self.fitted:
            return (np.log(x + 0.00000001) - self.L) / self.norm
        else:
            return np.log(x + 0.00000001)

    def fit(self, x):
        self.L = np.mean(np.log(x + 0.00000001), axis=0)
        self.fitted = True
        if self.normalisation_method == 'surface':
            self.norm = np.std(np.log(x + 0.00000001), axis=0)[-1]
        elif self.normalisation_method == 'top':
            self.norm = np.std(np.log(x + 0.00000001), axis=0)[0]
        elif self.normalisation_method == 'abs':
            self.norm = np.std(np.log(x + 0.00000001), axis=0)[-1] # last level
        else:
            pass

    def set_params(self, L):
        self.fitted=True
        self.L = L
        self.use_renorm = False

    @property
    def params(self):
        return self.L, self.norm

    def __str__(self):
        return super().__str__().format("Level_Normalizer", self.fitted, self.L)

###################################### Dict preprocess class

class Kernel(Preprocess):
    def __init__(self, input_var, output_var, inplace=False, ):
        super(Kernel, self).__init__()

    @property
    def new_vars(self):
        return []

    def __call__(self, x):
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

################### Dict Creation (to avoid recomputing)

class Positive_Normalizer(Preprocess):
    """
    Transform the input into a variable of mean 0 and norm 1
    Then set the minimum to 0 for Jacobian
    """
    def __init__(self):
        super().__init__()
        self.m = (0,0) # mean and min
        self.std = 1

    def __call__(self, x, headers=None):
        x -= self.m[0]
        x /= self.std
        x -= self.m[1]
        return x

    def fit(self, x):
        self.fitted=True
        self.m = (np.mean(x), np.min( x - np.mean(x) )/np.std(x) )
        self.std = np.std(x)

    @property
    def params(self):
        return self.m, self.std

    def set_params(self, cst, std):
        """
        CST is a tuple (mean, min)
        """
        self.fitted=True
        self.m = cst
        self.std = std

    def sub_vec(self, lev):
        return(np.zeros(lev)+self.m)

    def prod_vec(self, lev):
        return(np.ones(lev)/self.std)

    def __str__(self):
        return super().__str__().format("Normalizer", self.fitted, (self.m, self.std))


class Kernel(Preprocess):
    def __init__(self, fct, input_var, output_var=None):
        super().__init__(input_var, output_var, supress=True, inplace=False)
        self.fct = np.vectorize(pyfunc = fct)

    def __call__(self, x):
        return self.fct(x)


# DICT PREPROCESSOR:
class DictPreprocess:
    def __init__(self, prep_dictionnary):
        self.dict = prep_dictionnary
        
    @property
    def keys(self):
        return self.dict.keys
        
    def __getitem__(self, v):
        return self.dict[v]

    @property
    def input_variable(self):
        l = []
        for v in self.keys:
            l.append(self[v].input_var)
        return l
            
    @property
    def output_variable(self):
        l = []
        for v in self.keys:
            l.append(self[v].output_var)
        return l

    @property
    def new_variables(self):
        new_variables = []
        for v in self.keys:
            if not self[v].new_var:
                new_variables.append(self[v].output_var)
        return new_variables

    @property
    def delete_var(self):
        delete_var = []
        for v in self.keys:
            if self[v].supress_var:
                delete_var.append(self[v].input_var)
        return set(delete_var)