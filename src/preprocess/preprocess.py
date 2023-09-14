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
    
    def inverse(self, x):
        return x

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
        
    def inverse(self, x):
        return x  * self.std + self.m

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

    def inverse(self, x):
        return x  * self.norm + self.L

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

