import psutil
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from generator import Basic_Generator
from preprocess import Zero_One, Normalizer, DictPrepross


# Show ram usage
def print_ram_usage():
    process = psutil.Process(os.getpid())
    ram_usage = round(process.memory_info().rss/float(2**30), 2)
    print("RAM usage: {}GB".format(ram_usage))


def medium_pred(Mlist,x):
    for i,m in enumerate(Mlist):
        if i>0:
            y=y+m.predict(x)
        else:
            y=m.predict(x)
    return(y/len(Mlist))

# Plot Fonctions :
class F_and_plots:
    def __init__(self,shape, figsize=(15,10)):
        f, axes = plt.subplots(shape[0], shape[1], figsize=figsize)
        self.f = f
        self.shape = shape
        self.axes = axes

    def __getitem__(self,i):
        if(self.shape[0]*self.shape[1])==1:
            return(self.axes)
        else:
            return self.axes.flatten()[i]

def Plot_Batch(x, header, swap=True):
    """ Plot the 11 variables of a batch"""
    L = len(header)
    ligne = (L+1)//4 + 1
    f=plt.figure( figsize=(20, ligne*4), dpi=80)
    if swap:
        x0 = x.swapaxes(1,2).copy()
    else:
        x0 = x.copy()
    for i in range(len(header)):
        ax= f.add_subplot(ligne, 4, i+1)
        ax.set_title(header[i])
        for b in range(x0.shape[0]):
            ax.plot(np.flip(x0[b,i,:]), np.arange(len(x0[b,i,:]), 0, -1))
        ax.invert_yaxis()

def Plot_Histograms(F, w, header):
    f= plt.figure(figsize=(15,10))
    for i in range(11):
        F[i].hist(-abs(w[i,:]), bins=50, cumulative=True)
        F[i].set_title(header[i])


def Plot_one_profile(y):
    plt.plot(np.flip(y), np.arrange(len(y[0])) )
    plt.show()


def Plot_diff(F, y,y0, header_y, lev=72, J = [0], titles=False):
    for l in range(len(header_y)):
        for ind,i in enumerate(J):
            F[ind*len(header_y) + l].plot(np.flip(y[i,:,0].T) , np.arange(len(y[0,:,0])))
            F[ind*len(header_y) + l].plot(np.flip(y0[i,:,0].T) , np.arange(len(y0[0,:,0])))
            F[ind*len(header_y) + l].legend(["truth", "pred"])
            if titles !=False:
                F[ind*len(header_y) + l].set_title(header_y[l] + ' '+ str(titles[ind]))

def Plot_triple_diff_separated(F,y,y0, header_y, sep=0,  lev=72, j = 0):
    f = plt.figure( figsize=(15,8) )
    nv = len(header_y)
    for i in range(nv):
        F[i].plot(np.flip(y[:,:,i].T[:,j]) , np.arange(lev))
        F[i].plot(np.flip(y0[:,:,i].T[:,j]) , np.arange(lev))
        F[i].legend(["truth", "pred"])
        F[i].set_title(header_y[i]+' full column')
        if(sep>0):
            F[i+nv].plot(np.flip(y[:,:,i].T[sep:, j]) , np.arange(lev-sep))
            F[i+nv].plot(np.flip(y0[:,:,i].T[sep:, j]) , np.arange(lev-sep))
            F[i+nv].legend(["truth", "pred"])
            F[i+nv].set_title(header_y[i]+' low layers')
            #
            F[i+2*nv].plot(np.flip(y[:,:,i].T[:sep,j]) , sep+np.arange(sep))
            F[i+2*nv].plot(np.flip(y0[:,:,i].T[:sep,j]) , sep+np.arange(sep))
            F[i+2*nv].legend(["truth", "pred"])
            F[i+2*nv].set_title(header_y[i]+' high layers')

def Plot_FLXs(F, y, y0=[], header_y=['flx'], J = [0], titles=False):
    for ind,i in enumerate(J):
        F[0].plot(np.flip(y[i,:,0].T) , np.arange(len(y[0,:,0])))
        F[0].plot(np.flip(y[i,:,0].T) , np.arange(len(y[0,:,0])))
        F[0].legend(["truth", "pred"])
        if titles !=False:
            F[0].set_title(titles[0])
    if len(y0)!=0:
        for l in range(len(header_y)):
            for ind,i in enumerate(J):
                F[1].plot(np.flip(y0[i,:,0].T) , np.arange(len(y0[0,:,0])))
                F[1].plot(np.flip(y0[i,:,0].T) , np.arange(len(y0[0,:,0])))
                F[1].legend(["truth", "pred"])
                if titles !=False:
                    F[1].set_title(titles[1])


def reconstruct(T, div=5):
    _,y,x =T.shape
    T0 = np.zeros((div*x,div*y))
    for i in range(div):
        for j in range(div):
            if(i*div+j<len(T[:,0])):
                T0[i*x:(i+1)*x, j*y:(j+1)*y] = T[i*div+j].T
    return(T0)


def Get_Var(generator, header, var, op, y_v=False):
    idx = header.index(var)
    T=[]
    if(op==0):
        OP = lambda x: x[:,0, idx]
    if(op==-1):
        OP = lambda x: x[:,-1, idx]
    if(op==1):
        OP = lambda x: np.sum(abs(x[:,:, idx]), axis=1)
    if(op==2):
        OP = lambda x: np.mean(abs(x[:,:, idx]), axis=1)
    if(op==3):
        OP = lambda x: np.mean(np.square(x[:,:, idx]), axis=1)
    if(not y_v):
        for x, _ in generator:
            T.append(OP(x))
    else:
        for _, y in generator:
            T.append(OP(y))
    T = np.array(T)
    return(T)


# GRADIENT OPERATIONS:
def CumSum(dflx):
    """
    return flx given the output of the NN
    """
    flx = np.cumsum(np.flip(dflx), axis=1)
    flx = np.flip(flx)
    return(flx)

def Jacobian_NonCM(Mlist,x, dt):
    """
    Compute the Jacobian of x
    x has shape (1, lev, n_var)
    M product an output of size (1, lev)
    """
    _, lev, n_var= x.shape
    Jac = np.zeros((n_var, lev, lev))
    P0 = -medium_pred(Mlist, x)
    P1 = P0.copy()*0
    # could be more optimize [l steps instead of l*n_var]
    # Using one pred of size lev*n_var produce odd result, lev*header_x pred
    # which is not optimal
    for v in range(n_var):
        for l in range(lev):
            x0 = x.copy()
            pert = x[0,l,v]/dt
            x0[0, l, v] += pert
            P1 = -medium_pred(Mlist, x0)
            if abs(pert)>0:
                Jac[v, :, l] = (P1-P0)/pert
    return Jac

def Jacobian_Fortran(Mlist,x, dt):
    """
    Compute the Jacobian of x
    x has shape (1, lev, n_var)
    M product an output of size (1, lev)
    """
    _, lev, n_var= x.shape
    Jac = np.zeros((n_var, lev, lev))
    P0 = CumSum( -medium_pred(Mlist, x) )
    P1 = P0.copy()*0
    # could be more optimize [l steps instead of l*n_var]
    # Using one pred of size lev*n_var produce odd result, lev*header_x pred
    # which is not optimal
    for v in range(n_var):
        for l in range(lev):
            x0 = x.copy()
            pert = x[0,l,v]/dt
            x0[0, l, v] += pert
            P1 = CumSum(-medium_pred(Mlist, x0))
            if abs(pert)>0:
                Jac[v, :, l] = (P1-P0)/pert
    return Jac

def Check_Modif_Fortran(Mlist,x, dt):
    """
    Compute new outputs, for perturbations
    x has shape (1, lev, n_var)
    M product an output of size (1, lev)
    """
    _, lev, n_var= x.shape
    Jac = np.zeros((n_var, lev, lev))
    P0 = CumSum( medium_pred(Mlist, x) )
    P1 = P0.copy()*0
    # could be more optimize [l steps instead of l*n_var]
    # Using one pred of size lev*n_var produce odd result, lev*header_x pred
    # which is not optimal
    for v in range(n_var):
        for l in range(lev):
            x0 = x.copy()
            pert = x[0,l,v]/dt
            x0[0, l, v] += pert
            P1 = CumSum(medium_pred(Mlist, x0))
            Jac[v, :, l] = P1
    return Jac

def Jacobian_Single_Var_emis(Mlist,x, v_id, dt):
    """
    Compute the jacobian for 1 value vars such as emis
    v_id, id of the variable in the header_x (header.index(variable))
    """
    _, lev, n_var= x.shape
    J = np.zeros((lev))
    x0 = x.copy()
    x0[0,:,v_id]+=x0[0,0,0]/dt
    P1 = -CumSum(medium_pred(Mlist ,x))
    P2 = -CumSum(medium_pred(Mlist, x0))
    return (P1-P2)*dt

def Check_Modif_Fortran(Mlist,x, dt):
    """
    Compute new outputs, for perturbations
    x has shape (1, lev, n_var)
    M product an output of size (1, lev)
    """
    _, lev, n_var= x.shape
    Jac = np.zeros((n_var, lev, lev))
    P0 = CumSum( medium_pred(Mlist, x) )
    P1 = P0.copy()*0
    # could be more optimize [l steps instead of l*n_var]
    # Using one pred of size lev*n_var produce odd result, lev*header_x pred
    # which is not optimal
    for v in range(n_var):
        for l in range(lev):
            x0 = x.copy()
            pert = x[0,l,v]/dt
            x0[0, l, v] += pert
            P1 = CumSum(medium_pred(Mlist, x0))
            Jac[v, :, l] = P1
    return Jac
# Read NC4 files:

def Produce_netCDF4_jacobian(fh, header_x):
    z = np.arange(1, 73, 1)
    flag=True
    J2 = np.expand_dims(np.array(fh['dflxdpl']).T, axis=0)
    # first var is emis and a 2d var not shown anyway, so we use
    for v0 in header_x:
        v = 'dflxd'+v0
        if 'flx' in v and not 'emis' in v and not 'ts' in v:
            if flag:
                J2 = np.concatenate([J2 , np.expand_dims(np.array(fh[v]).T, axis=0)], axis=0)
            else:
                J2 = np.array(np.expand_dims(fh[v], axis=0))
                flag=True
    return(J2)
def Sep_Var_show_Log(F,J, header_x, T=True):
    """
    Show the Jacobian of each variable
    F : F and Plot class element of len len(header_x)
    header_x : list of variables
    J gradient of size (lev, n_var*lev)
    """
    c, l, _ = J.shape
    n_var = len(header_x)
    lev = l
    z = np.arange(1,73,1)
    for j in range(len(header_x)):
        if j>0:
            IMG = J[j]
            IMG = np.sign(J[j])*np.log(abs(J[j])+1)
            maxf = np.max(np.abs(IMG))
            if maxf == 0:
                maxf = 1.0
            incf = 2*maxf/51.
            clevs = np.arange(-maxf,maxf+incf,incf)
            im = F[j-1].contourf(z,z, IMG, clevs, cmap='PRGn')
            F[j-1].set_title(header_x[j])
            F[j-1].set_aspect('equal','box')
            F[j-1].invert_yaxis()
            F.f.colorbar(im, ax=F[j-1])

def Sep_Var_show(F,J, header_x, T=True):
    """
    Show the Jacobian of each variable
    F : F and Plot class element of len len(header_x)
    header_x : list of variables
    J gradient of size (lev, n_var*lev)
    """
    c, l, _ = J.shape
    n_var = len(header_x)
    lev = l
    z = np.arange(1,73,1)
    for j in range(len(header_x)):
        if j>0:
            IMG = J[j]
#            IMG = np.sign(J[j])*np.log(abs(J[j])+1)
            maxf = np.max(np.abs(IMG))
            if maxf == 0:
                maxf = 1.0
            incf = 2*maxf/51.
            clevs = np.arange(-maxf,maxf+incf,incf)
            im = F[j-1].contourf(z,z, IMG, clevs, cmap='PRGn')
            F[j-1].set_title(header_x[j])
            F[j-1].set_aspect('equal','box')
            F[j-1].invert_yaxis()
            F[j-1].invert_xaxis()
            F.f.colorbar(im, ax=F[j-1])


def Produce_x_nc4(fh, i0,j0, header_x):
    z = np.arange(1, 73, 1)
    flag=True
    x_0 = np.zeros((72, len(header_x)))
    for i,v in enumerate(header_x):
        if v=='emis':
            x_0[:,i] = fh.variables[v][0, j0-1, i0-1]
        if not 'emis' in v and not 'ts' in v:
            x_0[:,i] = fh.variables[v][0,:, j0-1, i0-1]
    return(x_0)

def file_name_to_id(fname):
#    fn, ex = fname.split('.')
    _, istr, jstr = fname.split('_')
    i = int(istr[1:])
    j = int(jstr[1:])
    return(i,j)


########## GET DICTIONNARY :
def read_Level_Norm(file):
    """
    Read a file with same '\n' parsing as 'dict0.txt' and load it into dictionnary
    Work as Load_FLX_dict for Level_Normalize preprocess class
    """
    d=dict()
    Flag=1
    with open(file) as f:
        all_of_it = f.read()
        G = all_of_it.split('\n')
        for line in G:
            if(len(line)==0):
                d[var]=values
                Flag=1
            elif(Flag==1):
                var = line
                values=[]
                Flag=2
            elif Flag==2:
                values=values+line.split(' ')
    for k in d.keys():
        delidx = []
        j=0
        while j< len(d[k]):
            if d[k][j]=='':
                del(d[k][j])
            else:
                d[k][j]=float(d[k][j])
                j+=1
        d[k]=np.array(d[k])
    return(d)


fct = []
for i in range(5):
    fct.append(Zero_One())
for j in range(5):
    fct.append(Normalizer())
hd = ['rl', 'ri', 'ql', 'qi', 'q', 'ts', 't', 'emis', 'o3', 'pl']

def Load_FLX_dict(header_dict = hd , path='DictPreprocess_fit.hdf5' , fct=fct):
    if os.path.isfile(path):
        Dhd = pd.read_hdf(path, key='s')
        D = DictPrepross([], [])
        D.load_from_pd(Dhd)
    else:
        print("Fitting Dict")
        B = Basic_Generator(data_folder)
        xdim, ydim = B.Xdim, B.Ydim
        B = Basic_Generator(data_folder, batch_size=xdim*ydim, shuffle=False)
        D = DictPrepross(header_dict, fct)
        D.fitonGen(B)
        Dhd = D.to_array_save()
        Dhd.to_hdf(path, key='s')
    return(D)
