import numpy as np
import pandas as pd
import os
import keras

data_folder="Data"
class Basic_Generator(keras.utils.Sequence):
    """
    Use hdf5 datasets and simply return the desire variables
    To create a new Generator simply inherit this one and change '__init__' and __'data_generation
    Shuffle : 0,1,2 : 0 no shuffle, 1 shuffle the data only, shuffle data and dataset order
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
        self.idx_el = np.arange(self.X_x_Y)
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
        self.used_variables = list(x.columns.levels[0])
        self.variables_pred = list(y.columns.levels[0])
        self.Xdim = len(x.index.levels[0]) # doesn't work with randoms
        self.Ydim = len(x.index.levels[1]) # doesn't work with randoms
        self.X_x_Y = len(x.index)
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
        d['var'] = len(self.used_variables)
        d['x'] = self.Xdim
        d['y'] = self.Ydim
        d['lev'] = self.lev
        d['dir'] = self._nb_dir
        return(d)

    def __len__(self):
        'batch per size'
        l = int( (self.X_x_Y // self.batch_size) * self._div * self._nb_dir)
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
        if self.shuffle>0:
            np.random.shuffle(self.idx_el)
        if self.shuffle>1:
            np.random.shuffle(self.idx_folder)
            np.random.shuffle(self.idx_file)
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

    @property
    def batch_per_file(self):
        return(self.X_x_Y // self.batch_size)

    def index_to_ids(self,index):
        index0 = index
        batch_per_file = self.batch_per_file
        el_id = index0 % batch_per_file
        index0 = index0 // batch_per_file
        file_id = index0 % self._div
        folder_id   = index0 // self._div
        return folder_id, file_id, el_id

    def ids_to_index(self, ids):
        index = (ids[0]*self._div + ids[1])*self.batch_per_file +ids[2]
        return(index)

    def reload(self,folder_id, file_id):
        """ Files are only loaded when the id of the file or folder is changed, this mutiply the speed by about 400"""
        if folder_id != self.current_folder or file_id != self.current_file:
            self.current_folder = folder_id
            self.current_file = file_id
            self.X, self.Y = self.load_a_couple(self.load_a_path(self.current_folder, self.current_file))

    def __data_generation(self, folder_id, file_id, el_ids):
        'Generates data containing batch_size samples'
        self.reload(folder_id, file_id)
        X = np.array(self.X.iloc[el_ids]).reshape(self.batch_size, len(self.used_variables), self.lev)
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
        self.used_variables = []
        if not self.reconfigured:
            for p in self.preprocess_x:
                for var in p.new_vars:
                    self.variables.append(var)
                    self._new_variables.append(var)
            self.used_variables = self.variables.copy()
            for p in self.preprocess_x:
                for var in p.eliminated_vars:
                    id = self.used_variables.index(var)
                    del(self.used_variables[id])
        self.reconfigured = True

    def apply_preprocess_x(self,X):
        X = np.array(X).reshape(self.X_x_Y , len(self.variables)-len(self._new_variables), self.lev)
        for p in self.preprocess_x:
            X = p(X, self.variables)
        return X

    def apply_preprocess_y(self,Y):
        Y = np.array(Y).reshape(self.X_x_Y, len(self.variables_pred), self.lev+1)
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
        Y = np.array(Y).reshape(self.X_x_Y, len(self.variables_pred), self.lev+1)
        idflx = np.array( [ self.variables_pred.index(name) for name in self.new_variables_pred])
        Y = Y[:, idflx, 1:] - Y[:, idflx, :-1]
        Y = Y.swapaxes(1,2)
        return Y

######## AE :
class AE_Generator(Full_Diff_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[],  chosen_var = ['flxd','flxu'], stash_x=True):
        self.stash_x = stash_x # conserve input x variables or not
        super(AE_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, \
                                                    preprocess_x=preprocess_x, chosen_var=chosen_var)

    def __getitem__(self, index):
        X,Y = super(AE_Generator, self).__getitem__(index)
        if self.stash_x:
            X = Y.copy()
        else:
            X = np.concatenate((X,Y),axis=-1)
        if(Y.shape[-1]==1):
            Y=Y.reshape(Y.shape[0],Y.shape[1])
        return X,Y


class FLX_AE_Generator(Full_Diff_Generator):
    """
    Has flxu, flxd as input and flx as output, work as an AE
    """
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[],  chosen_var = ['flxd','flxu']):
        super(FLX_AE_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, \
                                                    preprocess_x=[], chosen_var=chosen_var)

    def change_y(self,Y):
        """
        Get FLX (the output into Y0)
        """
#        Y= super(FLX_AE_Generator,self).apply_preprocess_y(Y)
        Y0 = -Y[:,:,0] + Y[:,:,1]
        if(Y0.shape[-1]==1):
            Y0 = Y0.reshape(Y0.shape[0],Y0.shape[1])
        return Y,Y0

    def __getitem__(self, index):
        X,Y = super(FLX_AE_Generator, self).__getitem__(index)
        X,Y = self.change_y(Y)
        return X,Y


class FLX_Generator(Full_Diff_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[], chosen_var=['flxu', 'flxd']):
        super(FLX_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, \
                                                    preprocess_x=preprocess_x, chosen_var=['flxu', 'flxd'])

    def apply_preprocess_y(self,Y):
        Y= super(FLX_Generator,self).apply_preprocess_y(Y)
        Y = -Y[:,:,0] + Y[:,:,1]
        if(Y.shape[-1]==1):
            Y = Y.reshape(Y.shape[0],Y.shape[1])
        return Y


class FLX_Generator(Full_Diff_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[], chosen_var=['flxu', 'flxd']):
        super(FLX_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, \
                                                    preprocess_x=preprocess_x, chosen_var=['flxu', 'flxd'])

    def apply_preprocess_y(self,Y):
        Y= super(FLX_Generator,self).apply_preprocess_y(Y)
        Y = -Y[:,:,0] + Y[:,:,1]
        if(Y.shape[-1]==1):
            Y = Y.reshape(Y.shape[0],Y.shape[1])
        return Y

#### CloudyGenerator
class CloudyGenerator(FLX_Generator):
    """
    Can generate data with no cloud. Suffle is set to 1 to take the right number of element in each subset
    """
    def __init__(self, folder=data_folder, train=True, batch_size=64, \
                 custom_b_p_e = 0, preprocess_x=[], no_cloudi=True, no_cloudl=True,  chosen_var = ['flxd','flxu','dfdts','flx']):
        self.no_cloudi = no_cloudi
        self.no_cloudl = no_cloudl
        self._is_cloud_init = False
        super(CloudyGenerator, self).__init__(folder, train, batch_size, shuffle=1, custom_b_p_e=custom_b_p_e, \
                                                    chosen_var=chosen_var, preprocess_x=preprocess_x)
        len(self)
        self._is_cloud_init = True

    def apply_preprocess_x(self, X):
        X = super(CloudyGenerator,self).apply_preprocess_x(X)
        id_el = np.ones(X.shape[0]).astype(bool)

        if self.no_cloudl:
            id_el = id_el * (np.max(X[:, self.ql_id, :], axis=1)==0).astype(bool)
        if self.no_cloudi:
            id_el = id_el * (np.max(X[:, self.qi_id, :], axis=1)==0).astype(bool)
        X = X[id_el]

        if self.no_cloudi*self.no_cloudl:
            X = np.delete(X, [self.qi_id, self.ql_id], axis=1)
        elif self.no_cloudi:
            X = np.delete(X, [self.qi_id], axis=1)
        elif self.no_cloudl:
            X = np.delete(X, [self.ql_id], axis=1)

        self.id_cloud = id_el.copy()
        return X

    def apply_preprocess_y(self, Y):
        Y = super(CloudyGenerator, self).apply_preprocess_y(Y)
        Y = Y[self.id_cloud]
        return(Y)

    def _reconfigure_outputs(self):
        super(CloudyGenerator, self)._reconfigure_outputs()

        self.qi_id = self.used_variables.index('qi')
        self.ql_id = self.used_variables.index('ql')

        if self.no_cloudl:
            a = self.used_variables.index('ql')
            del(self.used_variables[a])
        if self.no_cloudi:
            a = self.used_variables.index('qi')
            del(self.used_variables[a])

    def on_epoch_end(self):
        super(CloudyGenerator, self).on_epoch_end()
        if self._is_cloud_init:
            self.idx_el = np.arange(self.batch_size*self.n_per_file[self.current_folder*self._div + self.current_file])

    def index_to_ids(self,index):
        index0 = index
        file=0
        while index0 >= self.n_per_file[file]:
            index0-=self.n_per_file[file]
            file+=1
        folder_id = file // self._div
        file_id = file % self._div
        el_id = index0
        self.reload(folder_id, file_id)
        return folder_id, file_id, el_id

    def reload(self, folder_id, file_id):
        if folder_id != self.current_folder or file_id != self.current_file:
            super(Full_Diff_Generator, self).reload(folder_id, file_id)
            self.idx_el = np.arange(self.batch_size*self.n_per_file[folder_id*self._div + file_id])
            if self.shuffle>0:
                np.random.shuffle(self.idx_el)

    def __getitem__(self, index):
        X,Y = super(CloudyGenerator, self).__getitem__(index)
        return X,Y

    def __len__(self):
        if(self.max_b==0):
            self.n_per_file = np.arange(self.batch_per_file)
            N = []
            for i in range(len(self.List_of_dir)):
                for j in range(self._div):
                    self.reload(i,j)
                    N.append(self.X.shape[0])
            N = [n//self.batch_size for n in N]
            self.n_per_file = N
            self.max_b = np.sum(np.array(N))
            self.reload(0,0)
        return(self.max_b)


class Up_and_Down_Generator(Full_Diff_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[]):
        super(Up_and_Down_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, \
                                                    preprocess_x=preprocess_x, chosen_var= ['flxd','flxu','dfdts'])

#### Fully Connected
def Regu_flx(y, M):
    y_bin = (y>M)
    y_pic = y_bin*y
    y_avg = (1-y_bin)*y + (y_bin)*np.mean(y, axis=1).reshape(-1,1)

    y_bin = np.expand_dims(y_bin, axis=2)
    y_avg = np.expand_dims(y_avg, axis=2)
    y_pic = np.expand_dims(y_pic, axis=2)
    return(y_bin, y_pic, y_avg)

# PIC GENERATOR
class Pic_Generator(CloudyGenerator):
    """
    Generate FLX, its pics,
    """
    def __init__(self, folder=data_folder, train=True, batch_size=64, \
                 custom_b_p_e = 0, preprocess_x=[], chosen_var = ['flxd','flxu'], threshold=50):
        self.threshold = threshold
        super(Pic_Generator, self).__init__(folder, train, batch_size, custom_b_p_e=custom_b_p_e, \
                                    chosen_var=chosen_var, preprocess_x=preprocess_x)

    def apply_preprocess_y(self, Y):
        Y = super(Pic_Generator, self).apply_preprocess_y(Y)
        y_bin, y_pic, y_avg = Regu_flx(Y, self.threshold)
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
        Y = np.concatenate( (Y,y_pic,y_avg,y_bin), axis=-1)
        return(Y)

    def _reconfigure_outputs(self):
        super(Pic_Generator,self)._reconfigure_outputs()
        self.true_variables_pred = ['flx', 'pics', 'avg', 'bin']

class Has_Pic_Generator(Pic_Generator):
    """
    Generate FLX, its pics,
    """
    def __init__(self, folder=data_folder, train=True, batch_size=64, \
                 custom_b_p_e = 0, preprocess_x=[], chosen_var = ['flxd','flxu'], threshold=50):
        self.threshold = threshold
        super(Has_Pic_Generator, self).__init__(folder, train, batch_size, custom_b_p_e=custom_b_p_e, \
                                    chosen_var=chosen_var, preprocess_x=preprocess_x)

    def apply_preprocess_y(self, Y):
        Y = super(Has_Pic_Generator, self).apply_preprocess_y(Y)
        Y = np.max(Y[:,:,-1], axis=1).reshape(Y.shape[0], 1)
        return(Y)

    def _reconfigure_outputs(self):
        super(Has_Pic_Generator,self)._reconfigure_outputs()
        self.true_variables_pred = ['bin']


class Max_Pic_Generator(Pic_Generator):
    """
    Generate the max value of FLX,
    """
    def __init__(self, folder=data_folder, train=True, batch_size=64, \
                 custom_b_p_e = 0, preprocess_x=[], chosen_var = ['flxd','flxu'], threshold=50):
        self.threshold = threshold
        super(Max_Pic_Generator, self).__init__(folder, train, batch_size, custom_b_p_e=custom_b_p_e, \
                                    chosen_var=chosen_var, preprocess_x=preprocess_x)

    def apply_preprocess_y(self, Y):
        Y = super(Max_Pic_Generator, self).apply_preprocess_y(Y)
        Y = np.max(Y[:,:,0], axis=1).reshape(Y.shape[0], 1)
        return(Y)

    def _reconfigure_outputs(self):
        super(Max_Pic_Generator,self)._reconfigure_outputs()
        self.true_variables_pred = ['M']

########### FULLY CONNECTED

class FC_Generator(Up_and_Down_Generator):
    def __init__(self, folder=data_folder, train=True, batch_size=64, shuffle=True, \
                 custom_b_p_e = 0, preprocess_x=[], unique_var=['pl', 'emis', 'ts']):
        super(FC_Generator, self).__init__(folder, train, batch_size, shuffle, custom_b_p_e, preprocess_x=preprocess_x)
        self.unique_var = unique_var
        self._id_var_uni = [ self.used_variables.index(unique_var) for unique_var in self.unique_var ]
        self._id_var_lev = [ self.used_variables.index(var) for var in self.used_variables if not var in self.unique_var ]

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
