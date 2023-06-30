import numpy as np 

def get_dimension(data):
    x = len(data['Xdim'])
    y = len(data['Ydim'])
    z = len(data['lev'])
    return x, y, z

def load_nc4(data, portions, id, vars, verbose=0):
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