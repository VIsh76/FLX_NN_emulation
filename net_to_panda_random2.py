import pandas as pd
import numpy as np
import os
import tables

#Xdim=144, Ydim=864
### This file contains the function from_net_to_pd which convert a netcf4 file into a pd dataframe with more random batches
# Sadly the execution time is too slow because of the randomisation (it should take more than 1h to convert 1 file)

def Convert_random_all_from(Xdim, Ydim, in_folder, out_folder0, div=5, noutput=25):
    """
    Convert all files in in_folder to hdf5, divide them into div and save them in out_folder
    each generated folder correspond to one file.
    use_selection = 0 : the entire file is converted (warning files are heavy)
    use_selection > 0 : 'use_selection' files are selected randomly from each dataset
	Xdim : dimension of a division
	Ydim : dimension of a division
    """
    data_folder = in_folder
    lst_files = os.listdir(data_folder)
    lst_files.sort()

    N = lst_files[0].split('.')[-2]
    out_folder = os.path.join(out_folder0, N)
    print(out_folder)
    ids_n = np.arange(Xdim*div); np.random.shuffle(ids_n)
    ids_p = np.arange(Ydim*div); np.random.shuffle(ids_p)
    if not os.path.isdir(out_folder):
        print('Creating out_folder', out_folder)
        os.mkdir(out_folder)

    ndiv = Xdim
    pdiv = Ydim
    noutput = min(noutput, div*div)
    for i in range(div):
        for j in range(div):
            if(i*div+j)<noutput:
                name_out, extension = os.path.splitext(lst_files[i*div+j+25])
                name_in, extension = os.path.splitext(lst_files[i*div+j])

                name_out = os.path.join(out_folder, name_out+extension)
                name_in = os.path.join(out_folder, name_in+extension)
                print(name_in, name_out)

                c_ids_n = ids_n[i*ndiv:(i+1)*ndiv]
                c_ids_p = ids_p[j*pdiv:(j+1)*pdiv]
                build = False
                for l in range(div):
                    for k in range(div):
                        used_n_ids = c_ids_n[np.where(np.logical_and(l*ndiv <= c_ids_n, c_ids_n < (l+1)*ndiv ))]
                        used_p_ids = c_ids_p[np.where(np.logical_and(k*pdiv <= c_ids_p, c_ids_p < (k+1)*pdiv ))]
                        X =  pd.read_hdf(in_folder + lst_files[l*div+k])
                        Y =  pd.read_hdf(in_folder + lst_files[l*div+k+25])
                        if(len(used_n_ids)*len(used_p_ids)>0):
                            used_n_ids = used_n_ids%ndiv
                            used_p_ids = used_p_ids%pdiv
                            ID_flatten = np.dot(used_p_ids.reshape(-1,1), used_n_ids.reshape(1,-1)).flatten()
                            X = X.iloc[ID_flatten,  :]
                            Y = Y.iloc[ID_flatten,  :]
                            if(build):
                                Xf = Xf.append(X)
                                Yf = Yf.append(Y)
                            else:
                                build=True
                                Xf = X.copy()
                                Yf = Y.copy()
                Xf.to_hdf(name_in, key='s')
                Yf.to_hdf(name_out, key='s')
    return(Xf,Yf)

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
