import numpy as np
import pandas as pd
import os
import keras
from netCDF4 import Dataset

data_folder="Data"
def convertion_name(idn):
    """
    return input and output name given the string of the common part
    """
    inputn = 'f522_dh.trainingdata_in.lcv.'+idn+'.hdf5'
    outputn = 'jacobian_'+idn+'.npy'
    return(inputn, outputn)

class Basic_Generator_J(keras.utils.Sequence):
    """
    Use hdf5 datasets and simply return the desire variables
    To create a new Generator simply inherit this one and change '__init__' and __'data_generation
    Note that 'Basic_Generator' output are transposed to all its children class
    """
    def __init__(self, folder, batch_size=64, shuffle=True, custom_b_p_e = 0):
        """
        folder : folder where all the folders are used
        batch_size : number of output
        shuffle : (0,1,2) 0 no shuffle every output is read in the same order
                          1 file order stays the same but the rest is shuffled
                          2 shuffle everything
        The generator shuffle itself, when used in "fit_generator", "fit_generator"'s shuffle argument MUST be set to 0
        otherwise training is slowed down by several factors 10.

        custom_b_p_e : if >0, fix len of the generator, if custom_b_p_e==self.batch_per_file,
                        during training, the generator will use 1 file for each epoch, allowing
                        more analysing of the test loss

        this is a parent function and shouldn't be used, instead use Preprocessed_Generator with preprocess_x=[]
        """
        # global parameters
        self.path = folder
        self._set_init_true() # initialisation variable set true
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_b = custom_b_p_e

        # initialisation
        self._set_dirs(folder) # set the path
        self._initialise_parameters() #

        # generate the idx (order of the variable to load)
        self.idx_folder = np.repeat(np.arange(self._nb_dir), self._div)
        self.idx_file = np.tile(np.arange(self._div), self._nb_dir)
        self.idx_el = np.arange(self.X_x_Y)
        # generate the current variable, what (folder, file, batch in file) the generator is currently at
        self.current_b = 0
        self.current_folder = 0
        self.current_file = 0

        # randomize
        self.reset()
        self.on_epoch_end()
        self._set_init_false() # stop initialisation

    def _set_dirs(self, datafolder):
        """
        Store the list of used directories and files
        """
        self.List_of_dir = []
        self.List_of_files = dict()
        folders = os.listdir(datafolder)
        folders.sort()
        for i in folders:
            if os.path.isdir(os.path.join(datafolder,i)) and i != '.ipynb_checkpoints': # ignore .ipynb_checkpoints, allowing the generator to work in Amazon
                self.List_of_dir.append(os.path.join(datafolder,i))
                self.List_of_files[os.path.join(datafolder,i)]=[]
            for file in os.listdir(os.path.join(datafolder, i, 'Input')):
                if file.split('.')[-1] == 'hdf5':
                    self.List_of_files[os.path.join(datafolder,i)].append(file.split('.')[-2])
        self._nb_dir = len(self.List_of_dir)

    def _initialise_parameters(self):
        """
        load one file to compute variables such as  : the dimensions, the name of variables etc
        is used once at initialisation
        """
        x, _ = self._load_a_couple0(self.load_a_path(0,0))

        self._div = int(len(os.listdir(os.path.join(self.List_of_dir[0],'Input'))))
        self.variables = list(x.columns.levels[0])
        self.used_variables = list(x.columns.levels[0])
        self.variables_pred = ['dflxdpl', 'dflxdt', 'dflxdq', 'dflxdqi', 'dflxdql', 'dflxdo3']
        self.Xdim = len(x.index.levels[0])
        self.Ydim = len(x.index.levels[1])
        self.X_x_Y = len(x.index)
        self.lev = len(x.columns.levels[1])
        self.all_files_idx = np.arange(self._div*self._nb_dir)
        if self.shuffle>1:
            np.random.shuffle(self.all_files_idx)


    def reset(self):
        """
        Reset the index to zero (call after epoch_end if the current file and folder are the last ones)
        """
        self.all_files_idx = np.arange(self._div*self._nb_dir)

        if self.shuffle>1:
            np.random.shuffle(self.all_files_idx)

        self.idx_folder = self.all_files_idx//self._div
        self.idx_file = self.all_files_idx % self._div
        self.current_folder = self.idx_folder[0]
        self.current_file = self.idx_file[0]

    def load_a_path(self, id_fold, id_file):
        """
        Given id of folder and file, return the corresponding path
        """
#        print(id_fold, id_file)
#        print(self.List_of_dir)
#        print(self.List_of_files)
        pair_name = self.List_of_files[self.List_of_dir[id_fold]][id_file]
        input_name, output_name = convertion_name(pair_name)
        input_path = os.path.join(self.List_of_dir[id_fold], 'Input', input_name)
        output_path = os.path.join(self.List_of_dir[id_fold], 'Output', output_name)
        return (input_path, output_path)

    def _load_a_couple0(self, path):
        """ Given a path, load x and y only call once for initialisation"""
        assert(self._initialisation)
        X = pd.read_hdf(path[0], key='s')
        Y = np.load(path[1])
        return X , Y

    def load_a_couple(self, path):
        """ Given a path, load x (input) and y (output) only call once for initialisation"""
        return  pd.read_hdf(path[0], key='s'), np.load(path[1])

    @property
    def dimensions(self):
        """
        Return intel on the dimensions of the generate input/output
        """
        d=dict()
        d['div'] = (self._div)
        d['var'] = len(self.used_variables)
        d['x'] = self.Xdim
        d['y'] = self.Ydim
        d['lev'] = self.lev
        d['dir'] = self._nb_dir
        return(d)

    def __len__(self):
        l = int( (self.X_x_Y // self.batch_size) * self._div * self._nb_dir)
        if (self.max_b > 0): # if the len is fixed
            return(min(self.max_b, l))
        return(l)

    def _set_init_false(self):
        self._initialisation=False
    def _set_init_true(self):
        self._initialisation=True

    def on_epoch_end(self, _initialisation=False):
        """Updates indexes after an epoch"""
        self.current_b = 0 # batch set to 0
        if self.shuffle>0: # shuffle the batches
            np.random.shuffle(self.idx_el)

        # id of the current file
        cidx =self.all_files_idx.tolist().index( self.current_file + self.current_folder*self._div)


        self.idx_folder = np.roll(self.idx_folder, -cidx-1)
        self.idx_file = np.roll(self.idx_file, -cidx-1)
        self.all_files_idx = np.roll(self.all_files_idx, -cidx-1)

        # Load new data
        if(self._initialisation):
            self.X, self.Y = self._load_a_couple0(self.load_a_path(self.current_folder, self.current_file))
        else:
            self.X, self.Y = self.load_a_couple(self.load_a_path(self.current_folder, self.current_file))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        folder_id, file_id, el_id  = self.index_to_ids(index)
        # use the shuffled indices
        folder_id = self.idx_folder[folder_id]
        file_id = self.idx_file[file_id]
        el_ids = self.idx_el[el_id*self.batch_size + np.arange(self.batch_size)]
        self.current_b = el_ids
        return( self.__data_generation(folder_id, file_id, el_ids))

    @property
    def batch_per_file(self):
        """
        Number of batch in a file
        """
        return(self.X_x_Y // self.batch_size)

    def index_to_ids(self,index):
        """
        Given an index return the corresponding folder_id, file_id, batch_id
        """
        index0 = index
        batch_per_file = self.batch_per_file
        el_id = index0 % batch_per_file
        file_id =  index0 // batch_per_file
        folder_id =  index0 // batch_per_file
        return folder_id, file_id, el_id

    def ids_to_index(self, ids):
        """
        Given folder_id, file_id, batch_id return the corresponding index
        """
        index = (ids[0]*self._div + ids[1])*self.batch_per_file +ids[2]
        return(index)

    def reload(self,folder_id, file_id):
        """
        Files are only loaded when the id of the file or folder is changed
        The shuffle argument of keras 'fit_generator' MUST be set to 0 otherwise
        this function would be called at every step
        """
        if folder_id != self.current_folder or file_id != self.current_file:
            self.current_folder = folder_id
            self.current_file = file_id
            self.X, self.Y = self.load_a_couple(self.load_a_path(self.current_folder, self.current_file))

    def __data_generation(self, folder_id, file_id, el_ids):
        """
        Generates data containing batch_size samples, called at the end of __getitem__
        """
        self.reload(folder_id, file_id)
        X = np.array(self.X.iloc[el_ids]).reshape(self.batch_size, len(self.used_variables), self.lev)
        Y = self.Y[:,:,:,el_ids]
        return X,Y
        
######## Children Class, with preprocessing :

class Preprocessed_Generator(Basic_Generator_J):
    """
    Child of Basic_Generator allows a preprocess of the data
    """
    def __init__(self, folder, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[],preprocess_y=[]):
        """
        preprocess_x : list of Preprocess class to apply to input
        preprocess_y : list of Preprocess class to apply to output
        """
        super(Preprocessed_Generator, self).__init__(folder,  batch_size, shuffle, custom_b_p_e)
        self.preprocess_x = preprocess_x
        self.preprocess_y = preprocess_y
        self.reconfigured = False
        self._reconfigure_outputs() # change intput names
        self.on_epoch_end()

    def _reconfigure_outputs(self):
        """
        preprocessing can generate or suppress variables
        This function synchronize the output with those change
        """
        self._new_variables = []
        self.used_variables = []
        if not self.reconfigured: # safety to call it once
            # Generate new vars
            for p in self.preprocess_x:
                for var in p.new_vars:
                    self.variables.append(var)
                    self._new_variables.append(var)
            self.used_variables = self.variables.copy()
            for p in self.preprocess_x:
                # Eliminate vars
                for var in p.eliminated_vars:
                    id = self.used_variables.index(var)
                    del(self.used_variables[id])
        self.reconfigured = True

    def apply_preprocess_x(self,X):
        """
        Apply all the preprocess in the input data
        """
        X = np.array(X).reshape(self.X_x_Y , len(self.variables)-len(self._new_variables), self.lev)
        for p in self.preprocess_x:
            X = p(X, self.variables)
        return X

    def apply_preprocess_y(self,Y):
        """
        Apply all the preprocess in the output data
        """
        for p in self.preprocess_y:
            Y = p(Y, self.variables_pred)
        return Y

    def load_a_couple(self, path):
        """
        Load x and y files given by the two values of path
        """
        X,Y = super(Preprocessed_Generator, self).load_a_couple(path)
        X = self.apply_preprocess_x(X)
        return X,Y

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        folder_id, file_id, el_id  = self.index_to_ids(index)
        folder_id = self.idx_folder[folder_id]
        file_id = self.idx_file[file_id]
        el_ids = self.idx_el[el_id*self.batch_size + np.arange(self.batch_size)]
        self.current_b = el_ids
        return self.__data_generation(folder_id, file_id, el_ids)

    def __data_generation(self, folder_id, file_id, el_ids):
        """
        Generates data containing batch_size samples, called at the end of __getitem__
        """
        self.reload(folder_id, file_id)
        X = self.X[el_ids]
        Y = self.apply_preprocess_y(self.Y[:,:,:,el_ids])
        Y = np.rollaxis(Y,-1) #6,72,72,BS
        Y = Y.swapaxes(-2,-1)
        return X,Y

######## Differenciate Y and take only differences in flx as the output
class Diff_Generator(Preprocessed_Generator):
    """
    Generate the cumulative FLX, and the bias at level 0
    """
    def __init__(self, folder,  batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[],preprocess_y=[]):
        super(Diff_Generator, self).__init__(folder,  batch_size, shuffle, custom_b_p_e, preprocess_x)

    def apply_preprocess_x(self,X):
        """
        Apply all the preprocess in the input data
        """
        X = super(Diff_Generator, self).apply_preprocess_x(X)
#        X = X[:,:,1:]
        return X

    def apply_preprocess_y(self,Y):
        """
        Apply all the preprocess in the output data
        """
        Y = super(Diff_Generator, self).apply_preprocess_y(Y)
        Y[:,1:,1:,:] = Y[:,1:,1:,:] - Y[:,1:,:-1,:] # 0 correspond to the upper layer and is always 0
        return Y

####### AE :
class AE_Generator(Diff_Generator):
    """
    Generate the cumulative FLX, and the bias at level 0
    """
    def __init__(self, folder, tmp_folder='TmpFolder', batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[],preprocess_y=[]):
        super(AE_Generator, self).__init__(folder, tmp_folder, batch_size, shuffle, custom_b_p_e, preprocess_x)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        folder_id, file_id, el_id  = self.index_to_ids(index)
        folder_id = self.idx_folder[folder_id]
        file_id = self.idx_file[file_id]
        el_ids = self.idx_el[el_id*self.batch_size + np.arange(self.batch_size)]
        self.current_b = el_ids
        return self.__data_generation(folder_id, file_id, el_ids)

    def __data_generation(self, folder_id, file_id, el_ids):
        """
        Generates data containing batch_size samples, called at the end of __getitem__
        """
        self.reload(folder_id, file_id)
        X = []
        Y = self.apply_preprocess_y(self.Y[:,:,:,el_ids])
        Y = np.rollaxis(Y,-1) #6,72,72,BS
        Y = Y.swapaxes(-2,-1)
        return Y,Y
