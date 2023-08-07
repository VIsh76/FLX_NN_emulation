import numpy as np 
import time

def get_dimension(data):
    x = len(data['Xdim'])
    y = len(data['Ydim'])
    z = len(data['lev'])
    return x, y, z

def load_slow(data, portions, id, vars, verbose=0):
    """Load a portion of the data into a numpy array, form training purposes
    Construct the array first and fill it, this is slow if the initial array is large

    Args:
        data (xarray.Dataset): the dataset, must have all 'vars' as keys, with array of the same size
        portions (int): number of total portions (divisions of the file)
        id (int): the id of the portion, must be < to portions
        vars (list of str): list of the variables to consider
        verbose (int, optional): Printing function. Defaults to 0.

    Returns:
        np.array: a array of size (Xdim, Ydim//portions, num_var)
    """
    # Load dims
    Xdim = len(data['Xdim'])
    Ydim = int(len(data['Ydim'])) // portions
    lev  = len(data['lev'])
    nb_var = len(vars)

    # Tile system (load subpart of the data)
    tiles = int(len(data['Ydim'])) // Ydim
    id = min(portions-1, id)

    # Load each variable
    output = np.zeros((Xdim, Ydim, lev, nb_var))
    for i, var in enumerate(vars):
        if verbose>0:
            print(f"{var} - {data[var].shape}")
        if len(data[var].shape) > 3:
            output[:,:,:,i] = data[var].values.T[:, id::tiles, :, 0]
        else:
            output[:,:,:,i]  = np.expand_dims(data[var].values.T, axis=-1)[:, id::tiles, :, 0]
    return output

def load_nc4(data, portions, id, vars, verbose=0):
    """Load a portion of the data into a numpy array, form training purposes
    Build one array for each variable then concatenate, loading an array is O(1) but 
    concatenation get slower with the size of the portion.
    When loading a lot of porton > 100 load_nc4 get faster.
    Use this function when the number of portion is small or 1

    Args:
        data (xarray.Dataset): the dataset, must have all 'vars' as keys, with array of the same size
        portions (int): number of total portions (divisions of the file)
        id (int): the id of the portion, must be < to portions
        vars (list of str): list of the variables to consider
        verbose (int, optional): Printing function. Defaults to 0.

    Returns:
        np.array: a array of size (Xdim, Ydim//portions, num_var)
    """

    Xdim = len(data['Xdim'])
    Ydim = int(len(data['Ydim'])) // portions
    lev  = len(data['lev'])
    nb_var = len(vars)

    # Tile system (load subpart of the data)
    tiles = int(len(data['Ydim'])) // Ydim
    id = min(portions-1, id)

    # Load each variable
    l = []
    for var in vars:
        t = time.time()
        # if var is a column:
        if len(data[var].shape) > 3:
            output = data[var].data.T[:, id::tiles, :]
        # if var is on the surface :
        else:
            output  = np.repeat(np.expand_dims(data[var].values.T, axis=-1)[:, id::tiles, :], axis=2, repeats=lev)
        l.append(output)
        if verbose>0:
            print(f"{var} - {data[var].shape}")
            print(var, time.time()-t)
    T = time.time()
    X = np.concatenate(l, axis=-1)
    if verbose>0:
        print('Concatenation time', time.time() -T)
    return X
