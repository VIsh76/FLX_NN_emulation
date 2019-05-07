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
            self.List_of_dir = self.List_of_dir[:-1]
        self._nb_dir = len(self.List_of_dir)
        
    def _initialise_parameters(self):
        """ load one file to compute variables such as the dimensions, the name of var etc """
        x, y = self.load_a_couple(self.load_a_path(0,0))
        self._div = int(len(os.listdir(self.List_of_dir[0]))/2)
        self.variables = x.columns.levels[0]
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
        if self.shuffle == True:
            np.random.shuffle(self.idx_folder)
            np.random.shuffle(self.idx_file)
            np.random.shuffle(self.idx_el)
        self.reload(self.idx_folder[0], self.idx_file[0])
        
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

