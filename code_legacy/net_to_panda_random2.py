import pandas as pd
import numpy as np
import os
import tables

# Xdim=144, Ydim=864
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
import pandas as pd
import numpy as np
import os
import tables


def Shuffle_all_jac_from(Xdim, Ydim, in_folder, out_folder0, div2d, n_output):
    """
    Convert all files in in_folder to hdf5, divide them into div and save them in out_folder
    each generated folder correspond to one file.
    use_selection = 0 : the entire file is converted (warning files are heavy)
    use_selection > 0 : 'use_selection' files are selected randomly from each dataset
    Xdim : dimension of a division
    Ydim : dimension of a division
    in_folder : folder containing input and output
    """
    data_folder = in_folder
    lst_files_in = os.listdir(os.path.join(data_folder, 'Input'))
    lst_files_ou = os.listdir(os.path.join(data_folder, 'Output'))

    lst_files_in.sort()
    lst_files_ou.sort()

    lpynb = []
    if '.'==lst_files_in[0][0]:
        del(lst_files_in[0])
    if '.'==lst_files_ou[0][0]:
        del(lst_files_ou[0])

    N = '20190401_0000z'#lst_files_in[0].split('.')[-2] # NEED TO BE CHANGED
    out_folder = os.path.join(out_folder0, N)

    ids_n = np.arange(Xdim*div2d[0]); np.random.shuffle(ids_n)
    ids_p = np.arange(Ydim*div2d[1]); np.random.shuffle(ids_p)

    if not os.path.isdir(out_folder):
        print('Creating out_folder', out_folder)
        os.mkdir(out_folder0)
        os.mkdir(out_folder)
        os.mkdir(os.path.join(out_folder, 'Output'))
        os.mkdir(os.path.join(out_folder, 'Input'))
    ndiv = Xdim
    pdiv = Ydim
    noutput = min(n_output, div2d[0]*div2d[1])
    No = 'jac'
    Ni = 'input'
#    print(ids_n)
#    print(ids_p)
    for i in range(div2d[0]):
        for j in range(div2d[1]):
            c_id = i*div2d[1]+j
            if(c_id)<noutput:
                name_out, extension = os.path.splitext(No+'_'+str(c_id))
                name_in, extension = os.path.splitext(Ni+'_'+str(c_id))

                name_out = os.path.join(out_folder, 'Output', name_out+'.npy')
                name_in = os.path.join(out_folder, 'Input', name_in+'.hdf5')
                print(name_in, name_out)

                c_ids_n = ids_n[i*Xdim:(i+1)*Xdim]
                c_ids_p = ids_p[j*Ydim:(j+1)*Ydim]

                build = False
                for l in range(div2d[0]):
                    for k in range(div2d[1]):
                        used_n_ids = c_ids_n[np.where(np.logical_and(l*ndiv <= c_ids_n, c_ids_n < (l+1)*ndiv ))]
                        used_p_ids = c_ids_p[np.where(np.logical_and(k*pdiv <= c_ids_p, c_ids_p < (k+1)*pdiv ))]
                        X =  pd.read_hdf(os.path.join(in_folder, 'Input', lst_files_in[l*div2d[1]+k]))
                        Y =  np.load((os.path.join(in_folder, 'Output', lst_files_ou[l*div2d[1]+k])))
                        if(len(used_n_ids)*len(used_p_ids)>0):
                            used_n_ids = used_n_ids%ndiv
                            used_p_ids = used_p_ids%pdiv
                            ID_flatten = np.dot(used_p_ids.reshape(-1,1), used_n_ids.reshape(1,-1)).flatten()
                            X = X.iloc[ID_flatten,  :]
                            Y = Y[:, :, :, ID_flatten]
                            if(build):
                                Xf = Xf.append(X)
                                Yf = np.concatenate([Yf,Y], axis=-1)
                            else:
                                build=True
                                Xf = X.copy()
                                Yf = Y.copy()
                Xf.to_hdf(name_in, key='s')
                print(Yf.shape)
                Yf=np.save(name_out, np.array(Yf))
    return(Xf,Yf)

#f522_dh.trainingdata_in.lcv.20190401_0000z_is0001_js0001.hdf5
#Xdim=60
#Ydim=60
#Shuffle_all_jac_from(Xdim, Ydim, 'DataJac/Training/20190401_0000z/', 'DataJac/Shuffle', div2d=(12,72), n_output=1)
