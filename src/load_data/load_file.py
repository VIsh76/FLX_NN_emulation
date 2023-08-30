import numpy as np 
import datetime
import time

from pysolar.solar import get_altitude,  radiation

def subsection_select(data_values, tiles, id, lev):
    assert(len(data_values.shape) in [4,5])
    if len(data_values.shape) == 5:
        output = data_values.T[:, id::tiles, :, :]
    # if var is on the surface :
    elif len(data_values.shape) == 4:
        output  = np.repeat(np.expand_dims(data_values.T, axis=-1)[:, id::tiles, :], axis=-2, repeats=lev)
    return output

class VarLoader():
    def __init__(self, variables, portions):
        """_summary_

        Args:
            variables (_type_): _description_
         portions (int): number of total portions (divisions of the file)
        """
        self.portion = portions
        self._variables = variables.copy()
        self.is_set = False

    def set(self, y, lev):
        Ydim = y // self.portion     
        self.lev  = lev
        # Tile system (load subpart of the data)
        self.tiles = y // Ydim
        self.is_set = True
  
    @property
    def variables(self):
        return self._variables
    
    def load(self, data, id, verbose=0):
        """Load a portion of the data into a numpy array, form training purposes
        Build one array for each variable then concatenate, loading an array is O(1) but 
        concatenation get slower with the size of the portion.
        When loading a lot of porton > 100 load_nc4 get faster.
        Use this function when the number of portion is small or 1

        Args:
            data (xarray.Dataset): the dataset, must have all 'vars' as keys, with array of the same size
            (data has size : (X,Y,face,lev,time))
            id (int): the id of the portion, must be < to self.portion
            vars (list of str): list of the variables to consider
            verbose (int, optional): Printing function. Defaults to 0.
            add_geo_var (bool, False): add geographic variables (lat, lon)
            add_time_var (bool, False): add time variable (time of the day)

        Returns:
            np.array: a array of size (Xdim, Ydim//portions, faces, num_var)
        """
        T = time.time()
        id = min(self.portion-1, id)
        # Load physical variable
        all_vars = []
        for var in self._variables:
            t = time.time()
            if type(data[var]==np.ndarray):
                # ndarray
                output =  subsection_select(data[var], self.tiles, id, self.lev)
            else:
                # xarray (requires value)
                output =  subsection_select(data[var].values, self.tiles, id, self.lev)
            all_vars.append(output)
            if verbose>0:
                print(f"{var} - {data[var].shape}")
                print(var, time.time()-t)
        # Load Datetime Var
        X = np.concatenate(all_vars, axis=-1)
        if verbose>0:
            print('Concatenation time', time.time() -T)
        return X


class GeoLoader(VarLoader):
    def __init__(self, portion):
        super().__init__( ['lat_sin', 'lon_sin', 'lon_cos'], portion)

    def load(self, data, id, verbose=0):
        return super().load(self.add_geo_var(data), id, verbose)
    def add_geo_var(self, dataset):
        lbd_sin_lat = np.vectorize( lambda x : np.cos(2 * np.pi * x / 90))
        lbd_sin_lon = np.vectorize( lambda x : np.cos(2 * np.pi * x / 180))
        lbd_cos_lon = np.vectorize( lambda x : np.sin(2 * np.pi * x / 180))
        geo_dataset = {
        'lat_sin' : np.expand_dims(lbd_sin_lat(dataset['lats'].values), axis=0),
        'lon_sin' : np.expand_dims(lbd_sin_lon(dataset['lons'].values), axis=0),
        'lon_cos' : np.expand_dims(lbd_cos_lon(dataset['lons'].values), axis=0)
        }
        return geo_dataset


class TimeLoader(VarLoader):
    def __init__(self, portion):
        super().__init__( ['tao'], portion)

    def load(self, data, id, verbose=0):
        return super().load(self.add_date_var(data), id, verbose)
    
    def add_date_var(self, dataset):
        data_date = str(dataset['time'].values[0]).split('.')[0]
        data_date = datetime.datetime.strptime(data_date, '%Y-%m-%dT%H:%M:%S')
        data_date = datetime.datetime(data_date.year,
                                      data_date.month,
                                      data_date.day,
                                      data_date.hour,
                                      data_date.minute,
                                      tzinfo=datetime.timezone.utc
                                      )
        altitude_deg = get_altitude(dataset['lats'].values, dataset['lons'].values, data_date)
        return {'tao':np.expand_dims(altitude_deg, axis=0)}


class FullLoader(VarLoader):
    def __init__(self, variables, portions):
        super().__init__(variables, portions)
        self.geo = GeoLoader(portions)
        self.time = TimeLoader(portions)
    
    def set(self,  y, lev):
        super().set( y, lev)
        self.geo.set(y, lev)
        self.time.set(y, lev)
    
    @property
    def variables(self):
        return super().variables + self.geo.variables + self.time.variables
    
    def load(self, data, id, verbose=0):
        x1 = super().load(data, id, verbose)
        x2 = self.geo.load(data, id, verbose)
        x3 = self.time.load(data, id, verbose)
        return np.concatenate([x1,x2,x3], axis=-1)
