import numpy as np
import pandas as pd
import os
import keras

data_folder="Data"
class Basic_Generator(keras.utils.Sequence):
    """
    Use hdf5 datasets and simply return the desire variables
    To create a new Generator simply inherit this one and change '__init__' and __'data_generation
    """
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, custom_b_p_e = 0):
        # global parameters
        self._set_init_true()
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_b = custom_b_p_e
        # initialisation
        self._set_dirs(folder)
        self._initialise_parameters()
        # idx
        self.idx_folder = np.arange(self._nb_dir)
        self.idx_file = np.arange(self._div)
        self.idx_el = np.arange(self.Xdim*self.Ydim)
        self.current_b = 0
        self.current_folder = 0
        self.current_file = 0
        # randomize
        self.on_epoch_end()
        self._set_init_false()

    def _set_dirs(self, datafolder):
        self.List_of_dir = []
        folders = os.listdir(datafolder)
        folders.sort()
        for i in folders:
            if os.path.isdir(os.path.join(datafolder,i)) and i != '.ipynb_checkpoints':
                self.List_of_dir.append(os.path.join(datafolder,i))
        if(self.train):  # last folder is used as test
            self.List_of_dir = self.List_of_dir[:-1]
        else:
            self.List_of_dir = self.List_of_dir[-1:]
        self._nb_dir = len(self.List_of_dir)

    def _initialise_parameters(self):
        """ load one file to compute variables such as the dimensions, the name of var etc """
        x, y = self._load_a_couple0(self.load_a_path(0,0))
        self._div = int(len(os.listdir(self.List_of_dir[0]))/2)
        self.variables = list(x.columns.levels[0])
        self.variables_pred = list(y.columns.levels[0])
        self.Xdim = len(x.index.levels[0])
        self.Ydim = len(x.index.levels[1])
        self.lev = len(x.columns.levels[1])

    def load_a_path(self, id_fold, id_file):
        for f in os.listdir(self.List_of_dir[id_fold]):
            if f.split('.')[-1] == 'hdf5':
                if '_in' in f and '_'+str(id_file)+'.' in f:
                    input_path = os.path.join(self.List_of_dir[id_fold], f)
                if '_out' in f and '_'+str(id_file)+'.' in f:
                    output_path = os.path.join(self.List_of_dir[id_fold], f)
        return (input_path, output_path)

    def _load_a_couple0(self, path):
        """Load x and y only call once for initialisation"""
        assert(self._initialisation)
        X = pd.read_hdf(path[0], key='s')
        Y = pd.read_hdf(path[1], key='s')
        return X , Y

    def load_a_couple(self, path):
        """Load x and y files given by the two values of path"""
        return  pd.read_hdf(path[0], key='s'), pd.read_hdf(path[1], key='s')

    @property
    def dimensions(self):
        d=dict()
        d['div'] = (self._div)
        d['var'] = len(self.variables)
        d['x'] = self.Xdim
        d['y'] = self.Ydim
        d['lev'] = self.lev
        d['dir'] = self._nb_dir
        return(d)

    def __len__(self):
        'batch per size'
        l = int( (self.Xdim * self.Ydim // self.batch_size) * self._div * self._nb_dir)
        if (self.max_b > 0):
            return(min(self.max_b, l))
        return(l)

    def _set_init_false(self):
        self._initialisation=False

    def _set_init_true(self):
        self._initialisation=True

    def on_epoch_end(self, _initialisation=False):
        'Updates indexes after each epoch'
        self.current_b = 0
        if self.shuffle:
            np.random.shuffle(self.idx_folder)
            np.random.shuffle(self.idx_file)
            np.random.shuffle(self.idx_el)
        self.current_folder = self.idx_folder[0]
        self.current_file = self.idx_file[0]
        if(self._initialisation):
            self.X, self.Y = self._load_a_couple0(self.load_a_path(self.current_folder, self.current_file))
        else:
            self.X, self.Y = self.load_a_couple(self.load_a_path(self.current_folder, self.current_file))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        folder_id, file_id, el_id  = self.index_to_ids(index)
        # use the shuffled indices
        folder_id = self.idx_folder[folder_id]
        file_id = self.idx_file[file_id]
        el_ids = self.idx_el[el_id*self.batch_size + np.arange(self.batch_size)]
        self.current_b = el_ids
        return( self.__data_generation(folder_id, file_id, el_ids))

    def index_to_ids(self,index):
        index0 = index
        batch_per_file = (self.Xdim*self.Ydim // self.batch_size)
        el_id = index0 % batch_per_file
        index0 = index0 // batch_per_file
        file_id = index0 % self._div
        folder_id   = index0 // self._div
        return folder_id, file_id, el_id

    def reload(self,folder_id, file_id):
        """ Files are only loaded when the id of the file or folder is changed, this mutiply the speed by about 400"""
        if folder_id != self.current_folder or file_id != self.current_file:
            self.current_folder = folder_id
            self.current_file = file_id
            self.X, self.Y = self.load_a_couple(self.load_a_path(self.current_folder, self.current_file))

    def __data_generation(self, folder_id, file_id, el_ids):
        'Generates data containing batch_size samples'
        self.reload(folder_id, file_id)
        X = np.array(self.X.iloc[el_ids]).reshape(self.batch_size, len(self.variables), self.lev)
        Y = np.array(self.Y.iloc[el_ids]).reshape(self.batch_size, len(self.variables_pred), self.lev+1)
        return X,Y

######## Children Class, with preprocessing :

class Preprocessed_Generator(Basic_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[],preprocess_y=[]):
        super(Preprocessed_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e)
        self.preprocess_x = preprocess_x
        self.preprocess_y = preprocess_y
        self.reconfigured = False
        self._reconfigure_outputs()
        self.on_epoch_end()

    def _reconfigure_outputs(self):
        # preprocessing can generate new variables that have to be takken into account
        self._new_variables = []
        if not self.reconfigured:
            for p in self.preprocess_x:
                for var in p.new_vars:
                    self.variables.append(var)
                    self._new_variables.append(var)
        self.reconfigured = True

    def apply_preprocess_x(self,X):
        X = np.array(X).reshape(self.Xdim*self.Ydim , len(self.variables)-len(self._new_variables), self.lev)
        for p in self.preprocess_x:
            X = p(X, self.variables)
        return X

    def apply_preprocess_y(self,Y):
        Y = np.array(Y).reshape(self.Xdim*self.Ydim, len(self.variables_pred), self.lev+1)
        for p in self.preprocess_y:
            Y = p(Y, self.variables_pred)
        return Y

    def load_a_couple(self, path):
        """Load x and y files given by the two values of path"""
        X,Y = pd.read_hdf(path[0], key='s'), pd.read_hdf(path[1], key='s')
        X = self.apply_preprocess_x(X)
        Y = self.apply_preprocess_y(Y)
        return X,Y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        folder_id, file_id, el_id  = self.index_to_ids(index)
        # use the shuffled indices
        folder_id = self.idx_folder[folder_id]
        file_id = self.idx_file[file_id]
        el_ids = self.idx_el[el_id*self.batch_size + np.arange(self.batch_size)]
        return( self.__data_generation(folder_id, file_id, el_ids))

    def __data_generation(self, folder_id, file_id, el_ids):
        'Generates data containing batch_size samples'
        self.reload(folder_id, file_id)
        X = self.X[el_ids]
        Y = self.Y[el_ids]
        X=X.swapaxes(1,2)
        return X,Y

######## Differenciate Y and take only differences in flx as the output

class Full_Diff_Generator(Preprocessed_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[],  chosen_var = ['flxd','flxu','dfdts','flx']):
        self.new_variables_pred = chosen_var.copy()
        super(Full_Diff_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, \
                                                    preprocess_x=preprocess_x, preprocess_y=[])

    def apply_preprocess_y(self,Y):
        Y = np.array(Y).reshape(self.Xdim*self.Ydim, len(self.variables_pred), self.lev+1)
        idflx = np.array( [self.variables_pred.index(name) for name in self.new_variables_pred])
        Y = Y[:, idflx, 1:] - Y[:, idflx, :-1]
        Y = Y.swapaxes(1,2)
        return Y

#    def __data_generation(self, folder_id, file_id, el_ids):
#        'Generates data containing batch_size samples'
#        return super(Full_Diff_Generator, self).__data_generation(folder_id, file_id, el_ids)

class Diff_Generator(Full_Diff_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[]):
        super(Diff_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, \
                                                    preprocess_x=preprocess_x, chosen_var=['flx'])

    def apply_preprocess_y(self,Y):
        Y= super(Diff_Generator,self).apply_preprocess_y(Y)
        Y = Y[:,:,0]
        return Y

class Up_and_Down_Generator(Full_Diff_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[]):
        super(Up_and_Down_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, \
                                                    preprocess_x=preprocess_x, chosen_var= ['flxd','flxu','dfdts'])

class FC_Generator(Up_and_Down_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[], unique_var=['pl', 'emis', 'ts']):
        super(FC_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, preprocess_x=preprocess_x)
        self.unique_var = unique_var
        self._id_var_uni = [ self.variables.index(unique_var) for unique_var in self.unique_var ]
        self._id_var_lev = [ self.variables.index(var) for var in self.variables if not var in self.unique_var ]

    def __separate_uniques(self, X):
        X_u = X[:, -1, self._id_var_uni]
        X_l = X[:, :, self._id_var_lev]
        return[X_l, X_u]

    def apply_preprocess_y(self,Y):
        Y= super(FC_Generator,self).apply_preprocess_y(Y)
        Y = Y.swapaxes(1,2)
        Y = Y.reshape(Y.shape[0], -1)
        return Y

    def __getitem__(self,index):
        X,Y = super(FC_Generator, self).__getitem__(index)
        return self.__separate_uniques(X), Y

    def __data_generation(self, folder_id, file_id, el_ids):
        'Generates data containing batch_size samples'
        X,Y = super(FC_Generator, self).__data_generation(folder_id, file_id, el_ids)
        return self.__separate_uniques(X), Y
