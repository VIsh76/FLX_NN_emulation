import pandas as pd
import numpy as np
import os
import tables
from netCDF4 import Dataset

### This file contains the function from_net_to_pd which convert a netcf4 file into a pd dataframe

def Convert_all_from(in_folder='Data', out_folder='Data'):
	data_folder = in_folder
	lst_files = os.listdir(data_folder)
	lst_files.sort()
	out_folder0 = out_folder
	for f_name in lst_files:
		if os.path.isfile(os.path.join(data_folder , f_name)):
			f = Dataset(os.path.join(data_folder, f_name))
			x = f.variables
			header = list(x.keys())[6:] # 6 first var are dims, X,Y,lat, lon, time,

			folder_name = f_name.split('.')[-2]
			out_folder = os.path.join( out_folder0, folder_name)

			if not os.path.isdir(out_folder):
			    print('Creating out_folder', out_folder)
			    os.mkdir(out_folder)

			input_name  = os.path.join(data_folder,f_name)
			output_name = os.path.join(out_folder, f_name)

			fully_convert_file(input_name, output_name, header = header, div=5)
			print('---------')

def select(data, n0, p0, n1,p1, lev):
    """
    Select elements in data
    """
    s = len(data.shape)
    if(s==4):
        v = data[0,:,n0:n1,p0:p1]
    elif(s==3):
        v = data[:,n0:n1,p0:p1].repeat(lev,axis=0)
    else:
        print("Variable require to have at least 3 dim and at most 4")
        assert(False)
    v=v.reshape(lev,-1)
    return(v.T)

def from_net_to_pd(x, n_beg, p_beg, n_end, p_end,  header, lev):
    """
    Convert parts of
    """
    VAR = header
    LEV = np.arange(lev)
    n0 = n_end - n_beg
    p0 = p_end - p_beg
    X = np.arange(n_beg,n_end); X = X.repeat(p0)
    Y = np.arange(p_beg,p_end); Y = np.tile(Y,n0)
    C = np.array([ select(x[k], p_beg, n_beg,p_end, n_end, lev=lev) for k in VAR ])

    C=C.swapaxes(1,2) # keep the dimension in place when we reshape
    C = C.reshape(len(VAR)*len(LEV), -1).T
    index = pd.MultiIndex.from_product([VAR,LEV], names=['Var', 'level'])
    return(pd.DataFrame(data=C, index=[X,Y], columns=index))


def fully_convert_file(src, out, div, header, extension='.hdf5'):
    x = Dataset(src)

    n   = x['Xdim'].shape[0]
    p   = x['Ydim'].shape[0]
    lev = x['lev'].shape[0]

    n_step = int(n/div)
    p_step = int(p/div)

    n0 = 0
    p0 = 0
    n1 = n_step
    p1 = p_step

    out_name = os.path.splitext(out)[0]

    for i in range(5):
        print(i)
        dataf = from_net_to_pd(x, n0,p0,n1, p1,header, lev=lev)
        dataf.to_hdf(out_name+'_'+str(i)+extension, key='s')
        n0 += n_step
        p0 += p_step
        n1 += n_step
        p1 += p_step
