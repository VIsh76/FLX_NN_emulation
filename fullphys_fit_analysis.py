# %% [markdown]
# # Main fit notebook

# %% [markdown]
# # Initialisation

# %% [markdown]
# ## Load Parameters

# %%
from src.load_data.generator import PhysicGenerator
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch 
import os
import logging
import time

# INITIALIZE:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)
np.random.seed(0)

# Parameters :
from params import input_variables, pred_variables
from params import preprocess_X_path, preprocess_Y_path
from params import data_path, data_path_test, graph_path, output_path

# Change parameters (EC2 path)
data_path = 'Data/c48_test_full_WITH_RAD_train'
data_path_test = 'Data/c48_test_full_WITH_RAD_test'

# Where everything is saved:
experiment_name = 'Unet_20230831'
torch.set_grad_enabled(False) 

# %%
test=True

# Parameters depending on test :
if test:
    generator_length = 10
    nb_epochs = 1
else:
    generator_length = -1
    nb_epochs = 50
    
output_path = f"Output/{experiment_name}"
graph_path = f"{output_path}/graphic"
checkpoint_path =  f"{output_path}/checkpoints"

# %% [markdown]
# ## Construct Generators :

# %%
with open(preprocess_X_path, 'rb') as handle:
     preprocessor_x = pickle.load(handle)
with open(preprocess_Y_path, 'rb') as handle:
     preprocessor_y = pickle.load(handle)

prec = 48
data_loader = PhysicGenerator(data_path=data_path,
                   nb_portions=1,
                   batch_size=48 * 48 * 6,
                   input_variables = input_variables,
                   output_variables = pred_variables,
                   file_keys=['X', 'Y'],
                   shuffle=False,
                   preprocessor_x = preprocessor_x,
                   preprocessor_y = preprocessor_y,
                   verbose=0,
                   device=device,
                   _max_length=-1)

batch_size_test = data_loader.x * data_loader.x * 6
data_loader_test = PhysicGenerator(data_path=data_path_test,
                   nb_portions=1,
                   batch_size=batch_size_test,
                   input_variables = input_variables,
                   output_variables = pred_variables,
                   file_keys=['X', 'Y'],
                   shuffle=True,
                   preprocessor_x = preprocessor_x,
                   preprocessor_y = preprocessor_y,
                   verbose=0,
                   device=device,
                   _max_length=-1)

# %%
data_loader_test.batch_size, data_loader.batch_size

# %% [markdown]
# ## Model and Training

# %%
from src.models.unet import UNet

model = UNet(lev=data_loader.z, 
             in_channels=len(data_loader.input_variables), 
             out_channels=len(data_loader.output_variables))

# %% [markdown]
# ### Loss

# %%
from torch import nn

huber_fn = nn.HuberLoss()
mse_fn = nn.MSELoss()
loss_fn = mse_fn

# %% [markdown]
# ### Load Checkpoints

# %%
model_id = 48
model = torch.load(f'Output/Unet_20230831/checkpoints/Unet_{model_id}.tch')

# %%
x,y  = data_loader[10]
ypred = model(x).to('cpu').numpy()
y = y.to('cpu').numpy()
(y - ypred).shape
plt.plot((y - ypred)[:10, 0].T);

# %%


# %% [markdown]
# ### Analyse global error

# %%
from tqdm import tqdm
max_length = 1


losses_train = []
for i in tqdm( range( min(max_length, len(data_loader)))):
    x, y = data_loader[i]
    output = model(x)
    losses_train.append( loss_fn(output, y).item() )
    
losses_test = []
for i in tqdm(range(len(data_loader_test))):
    x, y = data_loader_test[i]
    output = model(x)
    losses_test.append( loss_fn(output, y).item() )
    
plt.hist(losses_train, alpha=0.5, bins=50, density=False);
plt.hist(losses_test, alpha=0.5 , density=False);
plt.axvline(np.mean(losses_train))

# %%
plt.plot(losses_train);
plt.plot(losses_test);

# %% [markdown]
# ### Analysis

# %%
def analysis_per_level(pred, truth):
    error = (pred-truth)**2
    error = error.to('cpu').numpy()
    return error

def full_analysis_per_level(data_loader, max_length=0):
    nb_vars = len(data_loader.output_variables)
    nb_lev = data_loader.z
    losses = np.zeros((len(data_loader), nb_vars, nb_lev))
    for i in tqdm( range(   min(max_length, len(data_loader)))):
        x, y = data_loader[i]
        output = model(x)
        error = np.mean(analysis_per_level(output, y), axis=0)
        losses[i] = error
    return losses

full_error_train = full_analysis_per_level(data_loader)
full_error_test = full_analysis_per_level(data_loader_test)

# %%
plt.plot(full_error_train[:,0].T, color='black', alpha=0.5);
plt.plot(full_error_test[:,0].T, color='red', alpha=0.9);

# %%
#plt.plot(full_error_train[:,0], color=(0,3,35), alpha=0.5);
import matplotlib.pyplot as plt
import numpy as np

# Define the number of plots and create a list of colors
num_plots = len(full_error_train)
colors = [(1 - i / num_plots, i / num_plots, 0) for i in range(num_plots)]  # Gradually change the green component

# Create a figure and axis
fig, ax = plt.subplots()
# Plot each line with a different color
for i, color in enumerate(colors):
    ax.plot(full_error_train[i,0], color=color, label=f'Plot {i+1}', alpha=0.2)

# Add labels, legend, and title
ax.set_ylabel('Mean Error')
ax.set_xlabel('Level')
# Show the plot
plt.show()

# %%
num_plots = len(full_error_train.T)
colors = [(1 - i / num_plots, i / num_plots, 0) for i in range(num_plots)]  # Gradually change the green component

# Create a figure and axis
fig, ax = plt.subplots()
# Plot each line with a different color
for i, color in enumerate(colors):
    ax.plot(full_error_train.T[i,0], color=color, label=f'Plot {i+1}', alpha=0.2)

# Add labels, legend, and title
ax.set_ylabel('Mean Error')
ax.set_xlabel('Level')
# Show the plot
plt.show()

# %%
f = plt.figure(figsize=(6, 12))
ax = f.add_subplot(2,1,1)
ax.set_ylabel('Mean Error')
ax.set_xlabel('Level')
plt.plot(np.mean(full_error_train[:,0], axis=0).T);

ax = f.add_subplot(2,1,2)
ax.set_ylabel('Mean Error')
ax.set_xlabel('Time')
plt.plot(np.mean(full_error_train.T[:,0], axis=0));

# %%
u_std = data_loader.preprocessor_y['u'].params
v_std = data_loader.preprocessor_y['v'].params
plt.plot(u_std);
plt.plot(v_std);

# %% [markdown]
# ### Analysis of the targets :

# %%
def analysis_per_level(data_loader):
    nb_vars = len(data_loader.output_variables)
    nb_lev = data_loader.z
    losses = np.zeros((2, len(data_loader), nb_vars, nb_lev))
    for i in tqdm( range(len(data_loader))):
        x, y = data_loader[i]
        std = np.std(y, axis=0)
        mean = np.mean(y, axis=0)
        losses[0, i] = mean 
        losses[1, i] = std 
    return losses

# %%
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.graphic.utils import deconstruct_cube, reshaper

def imshow_continents(x, input_variables):
    land_id = input_variables.index('frland')
    ice_id  = input_variables.index('frlandice')
    d = x[:, land_id, 0] + x[:, ice_id, 0]
    cube_face = reshaper(d.to('cpu').numpy())   
    flat_face = deconstruct_cube(cube_face)
    plt.imshow(flat_face, alpha=0.1);

def imshow_toa(x, input_variables):
    toa_id = input_variables.index('tao')
    d = x[:, toa_id, 0]
    cube_face = reshaper(d.to('cpu').numpy())   
    flat_face = deconstruct_cube(cube_face)
    plt.imshow(-flat_face, alpha=0.3, cmap='binary');

def show_one_time(t, data_loader, var, lev, is_in_x):
    """Will iterate over time

    Args:
        t (int): id of the loader
        data_loader (_type_): _description_
        var (str): variable to plot
        lev (int): level to plot
        is_in_x (bool): indication if var is in input_variable or output_variable
    """
    x, y = data_loader[t]
    if is_in_x:
        id_var = data_loader.input_variables.index(var)
        cube_face = reshaper(x[:, id_var, lev].to('cpu').numpy() )
    else:
        id_var = data_loader.output_variables.index(var)
        cube_face = reshaper(y[:, id_var, lev].to('cpu').numpy())
        
    flat_face = deconstruct_cube(cube_face)
    plt.imshow(flat_face)
    plt.title(f'Time {t}, var {var}, level {lev}')

def show_one_lev(t, data_loader, var, lev, is_in_x):
    show_one_time(lev, data_loader, var, t, is_in_x)
    plt.title(f'Time {t}, var {var}, level {lev}')

def loss_over_time(t, data_loader, var, lev, model, use_mean=False, add_continents=False, add_toa=False):
    x, y = data_loader[t]
    id_var = data_loader.output_variables.index(var)
    y_pred = model(x).to('cpu').numpy()
    error = (y.to('cpu').numpy() - y_pred)**2
    if use_mean:
        error = np.mean(error[:, id_var], axis=-1)
        lev = 'mean'
    else:
        error = error[:, id_var, lev]    

    cube_face = reshaper(error)   
    M = np.quantile(cube_face.flatten(), 0.99)
    cube_face = np.minimum(cube_face, M)
    flat_face = deconstruct_cube(cube_face)
    plt.imshow(flat_face, cmap='cool')
    if add_toa:
         imshow_toa(x, data_loader.input_variables)
    if add_continents:
         imshow_continents(x, data_loader.input_variables)

    plt.title(f'Error time : {t}, var :{var}, level : {lev}')
    return cube_face


def loss_over_levels(lev, data_loader, var, t, model):
    x, y = data_loader[t]
    id_var = data_loader.output_variables.index(var)
    y_pred = model(x).to('cpu').numpy()
    error = (y.to('cpu').numpy() - y_pred)**2
    error = error[:, id_var, lev]    
    cube_face = reshaper(error)   
    flat_face = deconstruct_cube(cube_face)
    plt.imshow(flat_face)
    plt.title(f'Error time {t}, var {var}, level {lev}')

  
def animate_over_time(data_loader, updater_fct, figsize, length, filepath, **kwargs):
    fig = plt.figure(figsize=figsize)
    lbd_update = lambda x: updater_fct(x, data_loader, **kwargs)
    anim = FuncAnimation(fig, lbd_update, frames=length, interval=10)
    # Save the animation as a gif
    with Image.new('RGB', (1, 1)) as img:
        anim.save(filename=filepath, writer='pillow')
    # Show the animation
    plt.show();

# %%
def diff_over_time(t, data_loader, var, lev, model, add_continents=False, add_toa=False):
    x, y = data_loader[t]
    id_var = data_loader.output_variables.index(var)
    y_pred = model(x).to('cpu').numpy()
    error = (y.to('cpu').numpy() - y_pred)
    error = error[:, id_var, lev]    
    cube_face = reshaper(error)   
    M = np.quantile(cube_face.flatten(), 0.99)
    cube_face = np.minimum(cube_face, M)
    flat_face = deconstruct_cube(cube_face)
    plt.imshow(flat_face, cmap='coolwarm',  vmax= np.max(abs(cube_face)), vmin= -np.max(abs(cube_face)))
    if add_toa:
         imshow_toa(x, data_loader.input_variables)
    if add_continents:
         imshow_continents(x, data_loader.input_variables)

    plt.title(f'Error time : {t}, var :{var}, level : {lev}')
    return cube_face

# %% [markdown]
# ### STUFF TO SAE :

# %%
# U and V (mean)
animate_over_time(data_loader, 
                  loss_over_time, 
                  figsize=(10,10), 
                  length=80, 
                  filepath='u_mean.gif', 
                  # Kwargs
                  add_continents=True, 
                  add_toa=True, 
                  lev=0,
                  var='u',
                  model=model,
                  use_mean=True)

animate_over_time(data_loader, 
                  loss_over_time, 
                  figsize=(10,10), 
                  length=80, 
                  filepath='v_mean.gif', 
                  # Kwargs
                  add_continents=True, 
                  add_toa=True, 
                  lev=0,
                  var='v',
                  model=model,
                  use_mean=True)

# U and V (level 15)
animate_over_time(data_loader, 
                  loss_over_time, 
                  figsize=(10,10), 
                  length=80, 
                  filepath='u_15.gif', 
                  # Kwargs
                  add_continents=True, 
                  add_toa=True, 
                  lev=0,
                  var='u',
                  model=model,
                  use_mean=True)

animate_over_time(data_loader, 
                  loss_over_time, 
                  figsize=(10,10), 
                  length=80, 
                  filepath='v_15.gif', 
                  # Kwargs
                  add_continents=True, 
                  add_toa=True, 
                  lev=0,
                  var='v',
                  model=model,
                  use_mean=True)

# %%
# U and V (level 15 - diff)
animate_over_time(data_loader, 
                  diff_over_time, 
                  figsize=(10,10), 
                  length=80, 
                  filepath='u_15_diff.gif', 
                  # Kwargs
                  add_continents=True, 
                  add_toa=True, 
                  lev=0,
                  var='u',
                  model=model);

animate_over_time(data_loader, 
                  diff_over_time, 
                  figsize=(10,10), 
                  length=80, 
                  filepath='v_15_diff.gif', 
                  # Kwargs
                  add_continents=True, 
                  add_toa=True, 
                  lev=15,
                  var='v',
                  model=model);

# %%
# plot random profiles :
i = 10
random_id = np.random.randint(0, data_loader.elements_per_file, 10)
random_id = [8006, 11330,  4949,  8451,  6792,  3188,  6139,  3245, 10531, 11540]
x, y  = data_loader[i]

y_pred = model(x).to('cpu').numpy()
y_truth = y.to('cpu').numpy()

y_truth_normed = data_loader.deprocess_y(y_truth.copy())
y_pred_normed = data_loader.deprocess_y(y_pred.copy())

# %%
print(random_id)

# %%
f = plt.figure(figsize=(15,10))
ax = f.add_subplot(1,2,1)
plt.plot(y_truth_normed[random_id, 0, :].T);
ax = f.add_subplot(1,2,2)
plt.plot(y_pred_normed[random_id, 0, :].T);

# %%
f = plt.figure(figsize=(15,10))
ax = f.add_subplot(1,2,1)
plt.plot(y_truth[random_id, 0, :].T);
ax = f.add_subplot(1,2,2)
plt.plot(y_pred[random_id, 0, :].T);


