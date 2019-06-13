import pandas as pd
import numpy as np
import os
import tables
from netCDF4 import Dataset

### This file contains the function from_net_to_pd which convert a netcf4 file into a pd dataframe with more random batches
# Sadly the execution time is too slow because of the randomisation (it should take more than 1h to convert 1 file)

def Convert_random_all_from(Xdim, Ydim, in_folder, out_folder, div=5):
	"""
	Convert all files in in_folder to hdf5, divide them into div and save them in out_folder
	each generated folder correspond to one file.
	use_selection = 0 : the entire file is converted (warning files are heavy)
	use_selection > 0 : 'use_selection' files are selected randomly from each dataset
	"""
	data_folder = in_folder
	lst_files = os.listdir(data_folder)
	lst_files.sort()
	out_folder0 = out_folder

	ids_n = np.arange(Xdim); np.random.shuffle(ids_n)
	ids_p = np.arange(Ydim); np.random.shuffle(ids_p)

	ndiv = Xdim // div
	pdiv = Ydim // div

	for i in range(div):
		for j in range(div):
			c_ids_n = ids_n[i*ndiv:(i+1)*ndiv]
			c_ids_p = ids_p[j*pdiv:(j+1)*pdiv]

			build = False
			for l in range(div):
				for k in range(div):
					used_n_ids = c_ids_n[np.where(np.logical_and(l*ndiv <= c_ids_n, c_ids_n < (l+1)*ndiv )]
					used_p_ids = c_ids_p[np.where(np.logical_and(k*ndiv <= c_ids_p, c_ids_p < (k+1)*ndiv )]
					X =  pd.read_hdf(lst_files[l*div+k])
					if(len(used_n_ids)*len(used_p_ids)>0):
						X = X.iloc[]
						if(build):
							Xf = Xf.append(X)
						else:
							build=True
							Xf = X.copy()

	for f_name in lst_files:
		if os.path.isfile(os.path.join(data_folder , f_name)) and '_in.lcv' in  f_name:
			in_f = True
			print(Xdim, Ydim)

			for i in range(2): # We load the in and out file at the same time
				if in_f:
					# Loading the 'in file'
					in_f = False
				else:
					f_name = f_name.split('_in')[0]+'_out'+f_name.split('_in')[1]
				f = Dataset(os.path.join(data_folder, f_name))
				x = f.variables
				print(f_name)
				header = list(x.keys())[6:] # 6 first var are dims, X,Y,lat, lon, time,

				folder_name = f_name.split('.')[-2]
				out_folder = os.path.join( out_folder0, folder_name)

				if not os.path.isdir(out_folder):
				    print('Creating out_folder', out_folder)
				    os.mkdir(out_folder)

				input_name  = os.path.join(data_folder,f_name)
				output_name = os.path.join(out_folder, f_name)

				fully_random_convert_file(input_name, output_name, div, ids_n, ids_p, header=header)
			print('---------')

def select_rng(data, rd_n, rd_p, lev):
    """
    Select elements in data
    """
    s = len(data.shape)
    print(data.shape)
    print(np.max(rd_n))
    print(np.max(rd_p))
    print(s)
    if(s==4):
        v = data[0,: ,rd_n, rd_p]
    elif(s==3):
        v = data[: ,rd_n, rd_p].repeat(lev,axis=0)
    else:
        print("Variable require to have at least 3 dim and at most 4")
        assert(False)
    v=v.reshape(lev,-1)
    return(v.T)

def from_net_to_pd_rng(x, ids_n, ids_p, header, lev):
    """
    Convert parts of
    """
    VAR = header
    LEV = np.arange(lev)
    n0 = len(ids_n)
    p0 = len(ids_p)

    X = ids_n; X = X.repeat(p0)
    Y = ids_p; Y = np.tile(Y,n0)
    C = np.array([ select_rng(x[k], ids_p, ids_n, lev=lev) for k in VAR ])

    C=C.swapaxes(1,2) # keep the dimension in place when we reshape
    C = C.reshape(len(VAR)*len(LEV), -1).T
    index = pd.MultiIndex.from_product([VAR,LEV], names=['Var', 'level'])
    return(pd.DataFrame(data=C, index=[X,Y], columns=index))

def fully_random_convert_file(src, out, div, n_ids_list, p_ids_list, header,  extension='.hdf5'):
	"""
	Takes random elements from the file
	# random_ids is a list of rd_n, rd_p, random elements to select
	"""
	x = Dataset(src)
	n = x['Xdim'].shape[0]
	p   = x['Ydim'].shape[0]
	lev = x['lev'].shape[0]

	n_step = int(n/div)
	p_step = int(p/div)
	n0 = 0
	p0 = 0
	n1 = n_step
	p1 = p_step

	out_name = os.path.splitext(out)[0]
	for k in range(div):
		i = n_ids_list[n0:n1]
		j = p_ids_list[p0:p1]
		print(np.max(i), np.max(j))
		dataf = from_net_to_pd_rng(x, i, j, header, lev=lev)
		dataf.to_hdf(out_name + '_' + str(k) + extension, key='s')
		n0 += n_step
		p0 += p_step
		n1 += n_step
		p1 += p_step
