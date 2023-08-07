from torch.utils.data import DataLoader
import torch
import numpy as np
import itertools
import random
import xarray as xr

from .load_file import  get_dimension, load_nc4
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
                 shuffle=True,
                 test=False,
                 _max_length=-1):
        """Construct a generator that will subpart of files for all the files

        Args:
            data_path (str): path to where the input and output data are located
            nb_portions (int): number of portion to cut the files
            batch_size (int): number of element per batch
            input_variables (list): list of str, selected input variables
            output_variables (list): list of str, selected output variables
            shuffle (bool, optional): if true, select random elements instead of in order. Defaults to True.
        """
        self.list_of_files = get_all_nc4_files(data_path)
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.nb_files = len(self.list_of_files)
        self.test = test

        # Get the dimensions using any core file
        self.x, self.y, self.z = get_dimension(xr.open_dataset(self.list_of_files[0][0]))
        self.nb_portions = nb_portions        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._max_length = _max_length

        # idx :
        self.on_epoch_end()
    
    def on_epoch_end(self):
        if self.test:
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
        ################################################################
        if self.test:
            print("Loading X :")
        data_X = xr.open_dataset(self.list_of_files[file_id][0])
        self.X = load_nc4(data_X,  self.nb_portions, id=portion_id, vars=self.input_variables, verbose=self.test)
        self.X =np.reshape(self.X, (-1, self.z, len(self.input_variables)))
        del(data_X)
        if self.test:
            print("Loading Y :")
        data_Y = xr.open_dataset( self.list_of_files[file_id][1])
        self.Y = load_nc4(data_Y,  self.nb_portions, id=portion_id, vars=self.output_variables, verbose=self.test)
        self.Y = np.reshape(self.Y, (-1, self.z+1, len(self.output_variables)))
        del(data_Y)
        ################################################################
        if self.shuffle :
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
            if self.test:
                print(f"Reload from {self.current_file_id}-{self.current_portion_id} to file {shuffle_file_id}-{shuffle_portion_id} ")
            self.reload(shuffle_file_id, shuffle_portion_id)
        # Pytorch requires channel first :
        return torch.from_numpy(np.swapaxes(self.X[start:start+self.batch_size], 1, 2).astype(np.float32)), torch.from_numpy(np.swapaxes(self.Y[start:start+self.batch_size], 1, 2).astype(np.float32))

    def __getitem__(self, index):
        X, y = self.__get_data(index) 
        return X, y


class PreprocessGenerator(BasicGenerator):
    def __init__(self,
                 data_path,
                 nb_portions,
                 batch_size,
                 input_variables,
                 output_variables,
                 preprocessor_y=[],
                 preprocessor_x=[],
                 shuffle=True,
                 test=False,
                 _max_length=-1):
        self.preprocessor_x = preprocessor_x
        self.preprocessor_y = preprocessor_y
        super().__init__(data_path,
                 nb_portions,
                 batch_size,
                 input_variables,
                 output_variables,
                 shuffle,
                 test,
                 _max_length)


    def reload(self, file_id, portion_id):
        super().reload(file_id, portion_id)
        self.preprocess_x()
        self.preprocess_y()

    def preprocess_x(self):
        for i, var in enumerate(self.input_variables):
            if var in self.preprocessor_x:
                if self.test:
                    print(f"Preprocessing {var} - {i}")
                self.X[:, :, i] = self.preprocessor_x[var](self.X[:, :, i])

    def preprocess_y(self):
        self.Y = self.Y[:, 1:] - self.Y[:, :-1]


# LOAD Xarray (fast) -> dict of array 
# -> loading one var into an array

# -> load one var -> tensor
# -> concatenate 
