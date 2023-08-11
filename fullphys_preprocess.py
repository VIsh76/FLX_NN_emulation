# %% [markdown]
# ### GENERATOR

# %% [markdown]
# Use a loader and then create preprocessors to be use in a latter loader with preprocessing

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
import pickle

# %%
# Parameters :

input_variables = ['u', 'v', 't', 'phis', 
             'frland','frlandice','frlake','frocean', # 'frseaice',
             'sphu','qitot','qltot','delp','ps_dyn',
              'dudtdyn', 'dvdtdyn', 'dtdtdyn']

pred_var = ['u', 'v', 't']

test=False # if test is True, a small batch of the data is instead loaded
save=True # if true overwrite last model

data_path = "/Users/vmarchai/Documents/ML_DATA/c48_XY_only_train"
graph_path = "Output/fullphys_0/Graph"
output_path = 'Output/fullphys_0/preprocess'
preprocess_name_X = f'{output_path}/X_fullphys_preprocess.pickle'
preprocess_name_Y = f'{output_path}/Y_fullphys_preprocess.pickle'

print("Creating folders")
os.makedirs(output_path, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)


# %%
from src.load_data.generator import PhysicGenerator

in_channel = len(input_variables)
out_channel = len(pred_var)

if test:
    nb_portion = 10
    prefix = "test"
else:
    nb_portion = 1 
    prefix = 'train'

PG_ex = PhysicGenerator(data_path = data_path,
                   nb_portions = nb_portion,
                   batch_size = 32,
                   input_variables=input_variables,
                   output_variables=['u','v','t'],
                   shuffle=False,
                   verbose=1)

# %% [markdown]
# # PLOTS :

# %%
from src.load_data.preprocess import Zero_One, Normalizer, Level_Normalizer, Log_Level_Normalizer, Rescaler

preprocess_X = {
# Surface :
'frlake'    : Zero_One(),
'frland'    : Zero_One(),
#'frlandice' : Zero_One(),
#psdyn
'frocean'   : Zero_One(),
'frseaice'  : Zero_One(),
# Colonnes Physics :
'u'        : Level_Normalizer(normalisation_method = False),
'v'        : Level_Normalizer(normalisation_method = False),
't'        : Level_Normalizer(normalisation_method = False),
'delp'     : Level_Normalizer(normalisation_method = False),
# CLOUDS
'qitot'        : Zero_One(),
'sphu'        :  Zero_One(),
'qltot'        : Zero_One(),

 'dudtdyn':Normalizer(),
 'dvdtdyn':Normalizer(),
 'dtdtdyn':Normalizer()
}

preprocess_Y = {
    'u':Rescaler(),
    'v':Rescaler(),
    't':Rescaler()
}

# %% [markdown]
# ### FIT - X

# %%
for i, var in enumerate(input_variables):
    t=time.time()
    if var in preprocess_X:
        PG0 = PhysicGenerator(data_path=data_path,
                   nb_portions=nb_portion,
                   batch_size=32,
                   input_variables=['u','v','t'] + [var],
                   output_variables=['u','v','t'],
                   shuffle=False,
                   verbose=1)
        print(f"Preprocessing {var} - {i} \t | {time.time() - t}")
        preprocess_X[var].fit(PG0.X[:, :, -1])        

# %% [markdown]
# ### FIT - Y

# %%
for i, var in enumerate(pred_var):
    t=time.time()
    if var in preprocess_Y:
        PG0 = PhysicGenerator(data_path=data_path,
                   nb_portions=nb_portion,
                   batch_size=32,
                   input_variables=[var],
                   output_variables=[var],
                   shuffle=False,
                   verbose=1)
        print(f"Preprocessing {var} - {i} \t | {time.time() - t}")
        preprocess_Y[var].fit(PG0.Y[:, :, 0])        

# %% [markdown]
# ### Retries

# %%
print('---- input ----')
X_prepross = np.zeros_like(PG_ex.X)
for i, var in enumerate(PG_ex.input_variables):
    if var in preprocess_X:
        print(f"Call {var} - {i}")
        X_prepross[:,:,i] = preprocess_X[var](PG_ex.X[:, :, i].copy())
        
print('---- output ----')
Y_prepross = np.zeros_like(PG_ex.Y)
for i, var in enumerate(PG_ex.output_variables):
    if var in preprocess_Y:
        print(f"Call {var} - {i}")
        Y_prepross[:,:,i] = preprocess_Y[var](PG_ex.Y[:, :, i].copy())

# %% [markdown]
# ### PLOTS

# %%
def save_figs(f, path, save=save):
    if save:
        f.savefig(path)

# INPUT VAR
f = plt.figure(figsize=(15, 15))
f.suptitle('input_var_orig')
for i, var in enumerate(PG_ex.input_variables):
    ax = f.add_subplot(1+len(PG_ex.input_variables)//4, 4, i+1)
    ax.plot(PG_ex.X[:10, :, i].T)
    ax.set_title(var)
save_figs(f, f'{graph_path}/input_var_orig.jpg')
plt.show()      

# PREPROCESS INPUT
f = plt.figure(figsize=(15,15))
f.suptitle('intput_var_preprocess')
for i, var in enumerate(PG_ex.input_variables):
    ax = f.add_subplot(1+len(PG_ex.input_variables)//4, 4, i+1)
    ax.plot(X_prepross[:10, :, i].T)
    ax.set_title(var)
save_figs(f, f'{graph_path}/intput_var_preprocess.jpg')
plt.show()


# OUTPUT VAR
f = plt.figure(figsize=(15,5))
f.suptitle('input_var_orig')
for i, var in enumerate(PG_ex.output_variables):
    ax = f.add_subplot(1+len(PG_ex.output_variables)//4, 4, i+1)
    ax.plot(  PG_ex.Y[:10, :, i].T)
    ax.set_title(var)
save_figs(f, f'{graph_path}/output_var_orig.jpg')
plt.show()  

f = plt.figure(figsize=(15,5))
f.suptitle('output_var_preprocess')
for i, var in enumerate(PG_ex.output_variables):
    ax = f.add_subplot(1+len(PG_ex.output_variables)//4, 4, i+1)
    ax.plot(Y_prepross[:10, :, i].T)
    ax.set_title(var)
save_figs(f, f'{graph_path}/output_var_preprocess.jpg')
plt.show()


# %%
PG_prep = PhysicGenerator(data_path = data_path,
                   nb_portions = nb_portion,
                   batch_size = 32,
                   input_variables=input_variables,
                   output_variables=['u','v','t'],
                   shuffle=False,
                   preprocessor_x=preprocess_X,
                   preprocessor_y=preprocess_Y,
                   verbose=1)

print('---- input ----')
X,Y = PG_prep[0]
X = X.numpy()
Y = Y.numpy()

# OUTPUT VAR
f = plt.figure(figsize=(15, 15))
f.suptitle('input_var_orig')
for i, var in enumerate(PG_prep.input_variables):
    ax = f.add_subplot(1+len(PG_prep.input_variables)//4, 4, i+1)
    ax.plot(  X[:10, i, :].T)
    ax.set_title(var)
#save_figs(f, f'{graph_path}/output_var_orig.jpg')
plt.show()  

f = plt.figure(figsize=(15,5))
f.suptitle('output_var_preprocess')
for i, var in enumerate(PG_prep.output_variables):
    ax = f.add_subplot(1+len(PG_prep.output_variables)//4, 4, i+1)
    ax.plot( Y[:10, i, :].T)
    ax.set_title(var)
#save_figs(f, f'{graph_path}/output_var_preprocess.jpg')
plt.show()

# %%
print( Y_prepross.shape)
print( np.std(Y_prepross, axis=(0,1)))

print( PG_ex.Y.shape)
print( np.std(PG_ex.Y, axis=(0,1)))

# %%
for v in preprocess_Y:
    print(preprocess_Y[v].params)

# %%
print( np.std(PG_ex.Y[:,:,0]) )
print( np.std(PG_ex.Y[:,:,1]) )
print( np.std(PG_ex.Y[:,:,2]) )

# %% [markdown]
# ### SAVE

# %%
if save:
    print("saving")
    path = f'{preprocess_name_X}'
    with open(path, 'wb') as handle:
        pickle.dump(preprocess_X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    path = f'{preprocess_name_Y}'
    with open(path, 'wb') as handle:
        pickle.dump(preprocess_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% [markdown]
# ## TESTS :

# %%
print('max : \t', np.max(PG_ex.Y, axis=(0,1)) )
print('min : \t', np.min(PG_ex.Y, axis=(0,1)) )
print('mean : \t', np.mean(PG_ex.Y, axis=(0,1)) )


print('abs-mean : \t', np.mean(abs(PG_ex.Y), axis=(0,1)) )
print('std : \t', np.std(PG_ex.Y, axis=(0,1)) )
print('pseudo-std : \t', np.sqrt(np.mean(   abs(PG_ex.Y)**2, axis=(0,1)) ))
