from torch.utils.data import DataLoader
import torch
import numpy as np
import itertools
import random
import xarray as xr

from .load_file import FullLoader, VarLoader
from .find_files import get_all_nc4_files


class BasicGenerator(object):
    """Hypothesis :
    - all tiles have the same shape

    Args:
        tf (_type_): _description_
    """
    def __init__(self,
                 data_path,
                 nb_portions,
                 batch_size,
                 input_variables,
                 output_variables,
                 file_keys=['X', 'Y'],
                 shuffle=True,
                 verbose = 0,
                 device='cpu',
                 _max_length=-1,
                 ):
        """Construct a generator that will subpart of files for all the files

        Args:
            data_path (str): path to where the input and output data are located
            nb_portions (int): number of portion to cut the files
            batch_size (int): number of element per batch
            input_variables (list): list of str, selected input variables
            output_variables (list): list of str, selected output variables
            shuffle (bool, optional): if true, select random elements instead of in order. Defaults to True.
        """
        self.file_keys = file_keys
        self.nb_portions = nb_portions        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self._max_length = _max_length
        self.device = device
        
        self.X_Loader = FullLoader(input_variables, nb_portions)
        self.Y_Loader = VarLoader(output_variables, nb_portions)
        self.input_variables =  self.X_Loader.variables
        self.output_variables = self.Y_Loader.variables    
        self.data_path = data_path
        

    def check(self):
        assert('X' in self.file_keys)
        assert('Y' in self.file_keys)

    def initialize(self):
        self.dict_of_files = get_all_nc4_files(self.data_path, self.file_keys)
        self.nb_files = len(self.dict_of_files[self.file_keys[0]])
        # Get the dimensions using any core file
        self.x, self.y, self.z = self.get_dimension()
        # set Loader
        self.X_Loader.set(self.y, self.z)
        self.Y_Loader.set(self.y, self.z)
        # idx :
        self.on_epoch_end()
    
    def on_epoch_end(self):
        if self.verbose>=1:
            print('on epoch end')
        self.file_portion_id = [ element for element in itertools.product(*[np.arange(self.nb_files),  np.arange(self.nb_portions)]) ]
        if self.shuffle:
            random.shuffle(self.file_portion_id)
        # Load the first file
        self.current_file_id, self.current_portion_id  = self.file_portion_id[0]
        self.reload(self.current_file_id , self.current_portion_id)
    
    # SIZE AND LEN
    @property
    def elements_per_file(self):
        return self.x * self.y // self.nb_portions
    @property
    def elements_per_portions(self):
        return self.x * self.y // self.nb_portions
    @property
    def batches_per_portions(self):
        return self.elements_per_portions // self.batch_size
    @property
    def batches_per_file(self):
        return self.nb_portions * self.batches_per_portions
    
    def __len__(self):
        if self._max_length > 0:
            return self._max_length
        return self.nb_files * self.batches_per_file

    def get_file_id(self, batch_id):
        return batch_id // self.batches_per_file
    
    def get_portion_id(self, batch_id):
        id_in_file = batch_id % self.batches_per_file
        return id_in_file // self.batches_per_portions

    def get_start(self, batch_id):
        """Given a batch_id return the lines in the data, end is start+batch_size

        Args:
            batch_id (int): _description_
        """
        id_in_portion = batch_id % self.batches_per_file % self.batches_per_portions
        return id_in_portion * self.batch_size

    def reload(self, file_id, portion_id):
        self.current_file_id = file_id
        self.current_portion_id = portion_id
        if self.verbose>=2:
            print('Generate X')
        self.generate_X(file_id, portion_id)
        if self.verbose>=2:
            print('Generate Y')
        self.generate_Y(file_id, portion_id)
        if self.shuffle:
            l = np.random.permutation(self.Y.shape[0])
            self.X = self.X[l]
            self.Y = self.Y[l]
        
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        file_id =  self.get_file_id(batches)
        portion_id = self.get_portion_id(batches)
        start = self.get_start(batches)

        shuffle_file_id, shuffle_portion_id = self.file_portion_id[file_id * self.nb_portions +  portion_id]

        if shuffle_file_id != self.current_file_id or  shuffle_portion_id != self.current_portion_id:
            if self.verbose>=1:
                print(f"Reload from {self.current_file_id}-{self.current_portion_id} to file {shuffle_file_id}-{shuffle_portion_id} ")
            self.reload(shuffle_file_id, shuffle_portion_id)
        # Pytorch requires channel first : i.e (bs, lev, vars))
        X = torch.from_numpy( np.swapaxes(self.X[start:start+self.batch_size], 1, 2).astype(np.float32)).to(self.device)
        Y = torch.from_numpy( np.swapaxes(self.Y[start:start+self.batch_size], 1, 2).astype(np.float32)).to(self.device)
        return X, Y

    def __getitem__(self, index):
        X, y = self.__get_data(index) 
        return X, y

# ABSTRACT :
    def get_dimension(self):
        # Open the first file and check the dimension
        file_0 = self.dict_of_files[self.file_keys[0]][0]
        data = xr.open_dataset(file_0)        
        x = len(data['Xdim'])
        y = len(data['Ydim'])
        z = len(data['lev'])
        return x, y, z
    
    def generate_X(self, file_id, portion_id):
        data_X = xr.open_dataset(self.dict_of_files['X'][file_id])
        self.X = self.X_Loader.load(data_X, id=portion_id, verbose=self.verbose)
        self.X = np.reshape(self.X, (-1, self.z, len(self.input_variables))) # reshape lev 
        del(data_X)

    def generate_Y(self, file_id, portion_id):
        data_Y = xr.open_dataset( self.dict_of_files['Y'][file_id])
        self.Y = self.Y_Loader.load(data_Y, id=portion_id, verbose=self.verbose)
        self.Y = np.reshape(self.Y, (-1, self.z, len(self.output_variables)))  # reshape lev+1 (output)
        del(data_Y)
        

class PhysicGenerator(BasicGenerator):
    def __init__(self,
                 data_path,
                 nb_portions,
                 batch_size,
                 input_variables,
                 output_variables,
                 file_keys=['X', 'Y'],
                 preprocessor_y=[],
                 preprocessor_x=[],
                 shuffle=True,
                 verbose=0,
                 device='cpu',
                 _max_length=-1):      
        super().__init__(data_path, 
                         nb_portions, 
                         batch_size, 
                         input_variables, 
                         output_variables,
                         file_keys, 
                         shuffle, 
                         verbose, 
                         device,
                         _max_length,)
        self.preprocessor_x = preprocessor_x
        self.preprocessor_y = preprocessor_y
        self.check()
        self.initialize()

    def reload(self, file_id, portion_id):
        super().reload(file_id, portion_id)
        self.preprocess_y()
        self.preprocess_x()
        
    def check(self):
        # Check file format
        assert('X' in self.file_keys)
        assert('Y' in self.file_keys)
        # Check if the input can be substracted from the output
        assert( len(self.output_variables) <= len(self.input_variables))
        for i, v in enumerate(self.output_variables):
            assert(v == self.input_variables[i])

    def get_dimension(self):
        # Open the first file and check the dimension
        file_0 = self.dict_of_files[self.file_keys[0]][0]
        data = xr.open_dataset(file_0)        
        x = len(data['Xdim'])
        y = len(data['Ydim']) * len(data['nf']) #
        z = len(data['lev'])
        return x, y, z

    def preprocess_x(self):
        for i, var in enumerate(self.input_variables):
            if var in self.preprocessor_x:
                if self.verbose>=2:
                    print(f"Preprocessing {var} - {i}")
                self.X[:, :, i] = self.preprocessor_x[var](self.X[:, :, i])

    def preprocess_y(self):
        # We compute the difference between 
        self.Y = (self.Y - self.X[:, :, :len(self.output_variables)])
        for i, var in enumerate(self.output_variables):
            if var in self.preprocessor_y:
                if self.verbose>=2:
                    print(f"Preprocessing {var} - {i}")
                self.Y[:, :, i] = self.preprocessor_y[var](self.Y[:, :, i])


class PreprocessColumnGenerator(BasicGenerator):
    def __init__(self,
                 data_path,
                 nb_portions,
                 batch_size,
                 input_variables,
                 output_variables,
                 file_keys=['X', 'Y'],
                 preprocessor_y=[],
                 preprocessor_x=[],
                 shuffle=True,
                 verbose=0,
                 _max_length=-1):
        self.preprocessor_x = preprocessor_x
        self.preprocessor_y = preprocessor_y
        super().__init__(data_path,
                 nb_portions,
                 batch_size,
                 input_variables,
                 output_variables,
                 file_keys,
                 shuffle,
                 verbose,
                 _max_length)
        self.initialize()

    def reload(self, file_id, portion_id):
        super().reload(file_id, portion_id)
        self.preprocess_x()
        self.preprocess_y()

    def preprocess_x(self):
        for i, var in enumerate(self.input_variables):
            if var in self.preprocessor_x:
                if self.verbose>=2:
                    print(f"Preprocessing {var} - {i}")
                self.X[:, :, i] = self.preprocessor_x[var](self.X[:, :, i])

    def preprocess_y(self):
        self.Y = self.Y[:, 1:] - self.Y[:, :-1]



