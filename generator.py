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

    def _set_dirs(self, datafolder):
        self.List_of_dir = []
        folders = os.listdir(datafolder)
        folders.sort()
        for i in folders:
            if os.path.isdir(os.path.join(datafolder,i)):
                self.List_of_dir.append(os.path.join(datafolder,i))
        if(self.train):  # last folder is used as test
            self.List_of_dir = self.List_of_dir[:-1]
        else:
            self.List_of_dir = self.List_of_dir[-1:]
        self._nb_dir = len(self.List_of_dir)

    def _initialise_parameters(self):
        """ load one file to compute variables such as the dimensions, the name of var etc """
        x, y = self.load_a_couple(self.load_a_path(0,0))
        self._div = int(len(os.listdir(self.List_of_dir[0]))/2)
        self.variables = list(x.columns.levels[0])
        self.variables_pred = y.columns.levels[0]
        self.Xdim = len(x.index.levels[0])
        self.Ydim = len(x.index.levels[1])
        self.lev = len(x.columns.levels[1])

    def load_a_path(self, id_fold, id_file):
        for f in os.listdir(self.List_of_dir[id_fold]):
            if '_in' in f and '_'+str(id_file)+'.' in f:
                input_path = os.path.join(self.List_of_dir[id_fold], f)
            if '_out' in f and '_'+str(id_file)+'.' in f:
                output_path = os.path.join(self.List_of_dir[id_fold], f)
        return (input_path, output_path)

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

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.current_b = 0
        if self.shuffle:
            np.random.shuffle(self.idx_folder)
            np.random.shuffle(self.idx_file)
            np.random.shuffle(self.idx_el)
        self.current_folder = self.idx_folder[0]
        self.current_file = self.idx_file[0]
        self.X, self.Y = self.load_a_couple(self.load_a_path(self.current_folder, self.current_file))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        folder_id, file_id, el_id  = self.index_to_ids(index)
        # use the shuffled indices
        folder_id = self.idx_folder[folder_id]
        file_id = self.idx_file[file_id]
        el_ids = self.idx_el[el_id*self.batch_size + np.arange(self.batch_size)]
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

    def _reconfigure_outputs(self):
        # preprocessing can generate new variables that have to be takken into account
        if not self.reconfigured:
            for p in self.preprocess_x:
                for var in p.new_vars:
                    self.variables.append(var)
        self.reconfigured = True

    def apply_preprocess_x(self,X):
        for p in self.preprocess_x:
            X = p(X)
        X = np.array(X).reshape(self.batch_size, len(self.variables), self.lev)
        return(X)

    def apply_preprocess_y(self,Y):
        for p in self.preprocess_y:
            Y = p(Y)
        Y = np.array(Y).reshape(self.batch_size, len(self.variables_pred), self.lev+1)
        return(Y)

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
        X = self.apply_preprocess_x(self.X.iloc[el_ids].copy())
        Y = self.apply_preprocess_y(self.Y.iloc[el_ids].copy())
        return X,Y

######## Differenciate Y and take only differences in flx as the output

class Diff_Generator(Preprocessed_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[]):
        super(Preprocessed_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, \
                                                    preprocess_x=preprocess_x, preprocess_y=[])

    def apply_preprocess_y(self,Y):
        Y = (np.array(Y['flx'].iloc[:,1:]) - np.array(Y['flx'].iloc[:,:-1]))
        Y = np.array(Y).reshape(self.batch_size, self.lev)
        return(Y)

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
        X = self.apply_preprocess_x(self.X.iloc[el_ids].copy())
        Y = self.apply_preprocess_y(self.Y.iloc[el_ids].copy())
        return X,Y
