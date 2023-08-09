# %% [markdown]
# ### GENERATOR

# %% [markdown]
# Use a loader and then create preprocessors to be use in a latter loader with preprocessing

# %%
import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, psutil
import time

# %%
surface_var = ['emis', 'frlake', 'frland', 'frlandice', 'frocean', 'frseaice', 'ts']
pred_var = ['flcd', 'flcu', 'flxd', 'flxu']
col_var = ['fcld', 'o3', 'pl', 'q', 'qi', 'ql', 'ri', 'rl', 't']
preprocess_name = 'Data/preprocess/train_basic_preprocessor.pickle'

# test :
# if test is True, a small batch of the data is instead loaded
test=False
# Save :
# if true overwrite
save=False

if test:
    nb_portion = 10
    prefix = "test"
else:
    nb_portion = 1 
    prefix = 'train'
    
input_variables = surface_var + col_var

# %%
from src.load_data.generator import BasicGenerator

PG_ex = BasicGenerator(data_path="Data/train/",
                   nb_portions=40,
                   batch_size=32,
                   input_variables=input_variables,
                   output_variables=['flxu'],
                   shuffle=False,
                   test=1)

# %%
f = plt.figure(figsize=(15,15))
for i, var in enumerate(PG_ex.input_variables):
    ax = f.add_subplot(len(PG_ex.input_variables)//4, 4, i+1)
    ax.plot(PG_ex.X[:10, :, i].T)
    ax.set_title(var)

# %% [markdown]
# # PLOTS :

# %%
from src.load_data.preprocess import DeltaFlux
from src.load_data.preprocess import Zero_One, Normalizer, Level_Normalizer, Log_Level_Normalizer

preprocess_Y = {'flxu':DeltaFlux(), 
                "flxd":DeltaFlux(),
                'flCu':DeltaFlux(), 
                "flCd":DeltaFlux(),                
                }

preprocess_X = {
# Surface :
'emis'      : Zero_One(),
'frlake'    : Zero_One(),
'frland'    : Zero_One(),
'frlandice' : Zero_One(),
'frocean'   : Zero_One(),
'frseaice'  : Zero_One(),
'ts'        : Normalizer(),
# Colonnes Physics :
'fcld'      : Zero_One(),
'o3'        : Log_Level_Normalizer(normalisation_method = 'no'),
'pl'        : Level_Normalizer(normalisation_method = 'surface'),
't'         : Level_Normalizer(normalisation_method = False),
# CLOUDS
'q'         :  Log_Level_Normalizer(normalisation_method = 'no'),
'qi'        : Zero_One(),
'ql'        : Zero_One(),
'ri'        : Zero_One(),
'rl'        : Zero_One()
}

# %%
for i, var in enumerate(input_variables):
    t=time.time()
    if var in preprocess_X:
        PG0 = BasicGenerator(data_path="Data/train/",
                   nb_portions=nb_portion,
                   batch_size=32,
                   input_variables=[var],
                   output_variables=['flcd'],
                   shuffle=False,
                   test=1)
        print(f"Preprocessing {var} - {i} \t | {time.time() - t}")
        preprocess_X[var].fit(PG0.X[:, :, 0])        

# %%
PG0 = BasicGenerator(data_path="Data/train/",
           nb_portions=20,
           batch_size=32,
           input_variables=['pl'],
           output_variables=['flcd'],
           shuffle=False,
           test=1)

# %%
plt.plot(PG0.X.mean(axis=0))

# %%
X_prepross = np.zeros_like(PG_ex.X)
for i, var in enumerate(PG_ex.input_variables):
    if var in preprocess_X:
        print(f"Call {var} - {i}")
        X_prepross[:,:,i] = preprocess_X[var](PG_ex.X[:, :, i].copy())

# %%
plt.plot(PG_ex.X[:1, :, 9].T);
plt.plot(preprocess_X['pl'].L);

# %%
np.mean(PG_ex.X[:, :, 9], axis=0).shape

# %%
plt.plot(PG_ex.X[0, :, 9])
plt.plot(np.mean(PG_ex.X[:, :, 9], axis=0));

# %%
plt.plot(preprocess_X['pl'].L);

# %%
plt.plot( (PG_ex.X[:10, :, 9] - np.mean(PG_ex.X[:, :, 9], axis=0)).T);

# %%
f = plt.figure(figsize=(15,15))
for i, var in enumerate(PG_ex.input_variables):
    ax = f.add_subplot(len(PG_ex.input_variables)//4, 4, i+1)
    ax.plot(X_prepross[:10, :, i].T)
    ax.set_title(var)

# %%
import pickle

if save:
    print("saving")
    path = f'{preprocess_name}'
    with open(path, 'wb') as handle:
        pickle.dump(preprocess_X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"save at {path}")
    with open(path, 'rb') as handle:
        b = pickle.load(handle)


