import pandas as pd
import numpy as np
import os
import tables
from netCDF4 import Dataset

### This file contains the function Convert_all_from which convert all netcf4 file into a
### pd dataframe
NAMECST0=2  # use for naming 1...ndiv or 0001_0001 if not zero
def Convert_all_from(in_folder='Data', out_folder='Data', div=(5,5), in_and_out=False, use_selection=0):
	"""
	Convert all files in in_folder to hdf5, divide them into div and save them in out_folder
	each generated folder correspond to one file.
	use_selection = 0 : the entire file is converted
	use_selection > 0 : 'use_selection' files are selected randomly from each dataset
	"""
	data_folder = in_folder
	lst_files = os.listdir(data_folder)
	lst_files.sort()
	out_folder0 = out_folder
	for f_name in lst_files:
		if os.path.isfile(os.path.join(data_folder , f_name)) and '_in.lcv' in  f_name:
			in_f = True
			K = np.random.choice(np.arange(div[0]*div[1]), use_selection, replace=False)
			ilist = K // div[1]
			jlist = K % div[1]
			for i in range(2): # We load the in and out file at the same time
				if in_f:
					# Loading the 'in file'
					in_f = False
				else:
					f_name = f_name.split('_in')[0]+'_out'+f_name.split('_in')[1]
				f = Dataset(os.path.join(data_folder, f_name))
				x = f.variables
				print(f_name)
				header = list(x.keys())[6:] # 6 first var are [dims, X,Y,lat, lon, lev, time]

				folder_name = f_name.split('.')[-2]
				out_folder = os.path.join( out_folder0, folder_name)

				if not os.path.isdir(out_folder):
				    print('Creating out_folder', out_folder)
				    os.mkdir(out_folder)

				input_name  = os.path.join(data_folder,f_name)
				output_name = os.path.join(out_folder, f_name)

				if use_selection==0:
					fully_convert_file(input_name, output_name, header = header, div=div)
				if use_selection>0:

					random_convert_file(input_name, output_name, div=div, random_ids=(ilist, jlist), header=header)
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


def from_net_to_pd(x, n_beg, p_beg, n_end, p_end,  header, lev,	use_fraction=False):
    """
    Convert a part of an netCDF4 file into a hdf5
	use_fraction = False, means variables containing 'FR' (such as frland) are dropped to save memory
	j : jacobian data used
    """
    VAR = header
    # USE fraction
    LEV = np.arange(lev)
    if use_fraction:
        pass
    else:
		# Do not store fraction variables
        VAR2 = VAR.copy()
        for var in VAR:
            if 'fr' in var:
                del(VAR2[VAR2.index(var)])
                VAR = VAR2.copy()
        del(VAR2)
    n0 = n_end - n_beg
    p0 = p_end - p_beg
    X = np.arange(n_beg,n_end); X = X.repeat(p0)
    Y = np.arange(p_beg,p_end); Y = np.tile(Y,n0)
    C = np.array([ select(x[k], p_beg, n_beg, p_end, n_end, lev=lev) for k in VAR ])
    C=C.swapaxes(1,2) # keep the dimension in place when we reshape
    C = C.reshape(len(VAR)*len(LEV), -1).T
    index = pd.MultiIndex.from_product([VAR,LEV], names=['Var', 'level'])
    return(pd.DataFrame(data=C, index=[X,Y], columns=index))

def make_name_id(x,y):
    _is ='_is'
    _js = '_js'
    ext ='.nc4'
    xs,ys=str(x), str(y)
    while(len(xs)<4):
        xs='0' + xs
    while(len(ys)<4):
        ys='0' + ys
    return _is+xs+_js+ys+ext

def fully_convert_file(src, out, div, header, extension='.hdf5', name='id'):
	"""
	Divide the file into div[0], div[1] parts and save them
	j : jacobian file
	"""
	x = Dataset(src)
	n = x['Xdim'].shape[0]
	p   = x['Ydim'].shape[0]
	lev = x['lev'].shape[0]

	n_step = int(n/div[0])
	p_step = int(p/div[1])

	n0 = 0
	p0 = 0
	n1 = n_step
	p1 = p_step

	out_name = os.path.splitext(out)[0]
	for i in range(div[0]):
		for j in range(div[1]):
			print(i,j)
			dataf = from_net_to_pd(x, n0,p0,n1, p1,header, lev=lev)
			if name=='id':
				id_n = make_name_id(n1+1,p1+1)
				fullname = out_name+make_name_id(n0+1,p0+1)+extension
			if name=='coord':
				fullname = out_name+'_'+str(i*div[1]+j)+extension
			if name=='same':
				fullname = out.split('.')[0] + extension
			dataf.to_hdf(fullname, key='s')

			p1 += p_step
			p0 += p_step
		n0 += n_step
		n1 += n_step
		p1 = p_step
		p0 = 0

def random_convert_file(src, out, div, random_ids, header,  extension='.hdf5', naming=0):
	"""
	Divide the file into div[0]*div[1] parts and select n_select files randomly
	the id of the file will be in the name
	"""
	x = Dataset(src)
	n = x['Xdim'].shape[0]
	p   = x['Ydim'].shape[0]
	lev = x['lev'].shape[0]

	n_step = int(n/div[0])
	p_step = int(p/div[1])

	n0 = 0
	p0 = 0
	n1 = n_step
	p1 = p_step
	ilist, jlist = random_ids

	out_name = os.path.splitext(out)[0]
	for k in range(len(ilist)):
		i,j = ilist[k],jlist[k]
		print(i,j)
		p0 = p_step*j
		p1 = p0 + p_step

		n0 = n_step*i
		n1 = n0 + n_step

		dataf = from_net_to_pd(x, n0,p0,n1, p1,header, lev=lev)
		str_id = '(' + str(i) + ',' + str(j) +' )'
		if naming==0:
			full_out_name = out_name+'_' + str_id  + '_' + str(k)+extension
		if naming==1:
			full_out_name = out_name+'_' + str_id  + '_' + str(k)+extension
		dataf.to_hdf(full_out_name, key='s')

############ JACOBIAN

def Convert_all_jac_from(in_folder='Data', out_folder='Data'):
	"""
	Convert all jacobian '.nec4' files in in_folder to hdf5
	each generated folder correspond to one file.
	use_selection = 0 : the entire file is converted
	use_selection > 0 : 'use_selection' files are selected randomly from each dataset
	"""
	data_folder = in_folder
	lst_files = os.listdir(data_folder)
	lst_files.sort()
	out_folder0 = out_folder
	for f_name in lst_files:
		if os.path.isfile(os.path.join(data_folder , f_name)) and '.nc4' in  f_name:
			print(f_name)
			f = Dataset(os.path.join(data_folder, f_name))
			x = f.variables
			header = list(x.keys())[3:] # 3 first var are [lat, lon, lev]
			input_name  = os.path.join(data_folder,f_name)
			output_name = os.path.join(out_folder, f_name)
			fully_convert_file_j(input_name, output_name, header = header, name='same')
			print('---------')

def fully_convert_file_j(src, out, header, extension='.npy', name='coord'):
	"""
	Divide the file into div[0], div[1] parts and save them
	j : jacobian file
	"""
	x = Dataset(src)
	lev,_,n,p = x['dflxdpl'].shape
	out_name = os.path.splitext(out)[0]
	dataf = from_net_to_npy_j(x, header, lev=lev)
	fullname = out.split('.')[0] + extension
	np.save(out_name, dataf)

def from_net_to_npy_j(x, header, lev):
    """
    Convert a part of an netCDF4 file into a npy
    """
    VAR = header
    C = np.array([x[k] for k in VAR ])
    C2 = np.zeros((len(VAR), lev, lev, C.shape[-1]*C.shape[-1]))
    for i in range(C.shape[-1]):
        for j in range(C.shape[-2]):
            C2[:,:,:,i*C.shape[-1]+j] = C[:,:,:,i,j]
    return C2

def select_j(data, lev):
    """
    Select elements in data
    """
    s = len(data.shape)
    if(s==4):
        v = data[:,:,n0:n1,p0:p1]
    return(v.T)
