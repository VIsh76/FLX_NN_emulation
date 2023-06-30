# %% [markdown]
# # GRAPH :

# %% [markdown]
# This notebook loads and plot different information of profile and information related to te physical Data. It requires only 3 parameters :
# - the path of an X data
# - the path of an Y data
# - the prefix to save 
# 
# -> this notebook can be transform into a main python file

# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
data_path_X = "Data/train/f522_dh.trainingdata_in.lcv.20190216_0000z.nc4"
data_path_Y = "Data/train/f522_dh.trainingdata_out.lcv.20190216_0000z.nc4"
prefix = "20190216_0000z"

# Creation of output folder :
output_path = f"Graph/{prefix}"
os.makedirs(output_path, exist_ok=True)

# %%
# Open the data :
data_X = xr.open_dataset(data_path_X)
data_Y = xr.open_dataset(data_path_Y)

# %%


# %%
for var in data_X:
    for i in data_X[var]:
        data_X[var][i]
    break

# %% [markdown]
# ### Get the variables surface or colums

# %%
surface_var = []
col_var = []
for var in data_X:
    if len(data_X[var].shape) <= 3:
        surface_var.append(var) 
    else:
        col_var.append(var)

print("-----------")
print(f"Surface : {len(surface_var)}")
print("-----------")
for var in surface_var:
    print(var)
print("-----------")

print(f"Colonne : {len(col_var)}")
print("-----------")
for var in col_var:
    print(var)

pred_var=['flcd', 'flcu', 'flxd', 'flxu']

# %%
print(surface_var)
print(pred_var)
print(col_var)

# %%
x_id = np.random.randint(4320)
y_id = np.random.randint(720)

from src.graphic.data_information import plot_one_profile_column, plot_one_profile_surface
plot_one_profile_column(data_X, col_var, x_id, y_id);

# %% [markdown]
# # Question : Is lattitude important ?

# %%
f, ax = plt.subplots(1,3)
ax[0].imshow(data_Y['lats'].values);
ax[0].set_title('lats');
ax[1].imshow(data_Y['lons'].values);
ax[1].set_title('lons');
ax[2].imshow(data_X['frocean'].values[0]);
ax[2].set_title('frocean');

# %%
for var in data_Y:
    print(var, "\t", data_Y[var].shape) 

# %%
for var in data_X:
    print(var, "\t", data_X[var].shape) 

# %%
plot_one_profile_surface(data_Y, pred_var, 1);

# %%
plot_one_profile_surface(data_X,  col_var+surface_var, 16);

# %% [markdown]
# ### TEST DIFF BETWEEN LOWER LAYERS : DAY/NIGHT

# %%
z = 72
plt.hist(data_Y['flcd'].values[0,z].flatten(), bins=25);

# %%
z = 72
plt.hist(data_Y['flcu'].values[0,z].flatten(), bins=25, alpha=0.5);
plt.hist(data_Y['flcu'].values[0,0].flatten(), bins=25, alpha=0.5);

# %%
Z = data_Y['flcu'].values[0,72].flatten() + data_Y['flcd'].values[0,72].flatten()
plt.hist(Z, bins=25, alpha=0.5);

# %%
Z = data_Y['flcu'].values[0,72].flatten() - data_Y['flcu'].values[0,0].flatten()
plt.hist(Z, bins=25);

# %%
from PIL import Image
from matplotlib.animation import FuncAnimation

def update_hist(frame, datas):
    plt.hist(datas[frame], bins=20)
    plt.title(f"{frame}")
    plt.axis('off')

def animate_hist(ds, var, filepath=None):
    data = []
    for z in range(72):
        data.append(ds[var].values[0, z, ::5, ::5])  
        # Create the animation using FuncAnimation
    fig = plt.figure()
    lbd_update = lambda x: update_hist(x, datas=data)
    anim = FuncAnimation(fig, lbd_update, frames=len(data), interval=10)
    # Save the animation as a gif
    with Image.new('RGB', (1, 1)) as img:
        anim.save(filename=filepath, writer='pillow')
    # Show the animation
    plt.show()
    return data

# %% [markdown]
# # ANIMATE 

# %% [markdown]
# ## ANIMATE Y

# %% [markdown]
# ## SHOW INCONSTICENCIES

# %%
#plt.imshow(data_Y['flcd'].values[0,2,:,:], alpha=0.5)
#plt.imshow(data_X['frocean'].values[0,:,:])

Image1 = data_Y['flcd'].values[0,2,:,:]
Image2 = data_X['frocean'].values[0,:,:]

#plt.imshow(Image1, cmap='plasma') # I would add interpolation='none'
#plt.imshow(Image2, cmap='gray', alpha=0.5*(Image2>0)   ) # interpolation='none'

plt.imshow(Image2, cmap='gray') # I would add interpolation='none'
plt.imshow(Image1, cmap='plasma', alpha=0.7*(Image2>0)   ) # interpolation='none'
plt.savefig("Graph/down_sun.jpg")

plt.imshow(Image1, cmap='plasma') # I would add interpolation='none'
plt.savefig("Graph/down_discontinuous.jpg")

# %% [markdown]
# ## PLOT entire earth :

# %%
from src.graphic.data_information import plot_square_earth

plot_square_earth(data_X['frocean'].values[0]);
plot_square_earth(data_X['emis'].values[0]);
plot_square_earth(data_X['ts'].values[0]);

# %% [markdown]
# ### PLOT ON A BOX

# %% [markdown]
# # HISTOGRAMS :

# %% [markdown]
# ## SURFACE

# %%
print(surface_var)

# %%
for var in surface_var:
    print(var)
    plt.hist(data_X[var].values[0].flatten(), bins=20, density=True);
    plt.title(var)
    plt.show()

# %% [markdown]
# ## COLONNE

# %%
print(col_var)

# %% [markdown]
# ### NUAGES

# %%
for var in ["q", "qi", "ql", "ri", "rl"]:
    print(var)
    plt.hist(data_X[var].values[0].flatten(), bins=50, density=True);
    plt.title(var)
    plt.show()

# %%
for var in ["fcld", "o3", "t"]:
    print(var)
    plt.hist(data_X[var].values[0].flatten(), bins=50, density=True);
    plt.title(var)
    plt.show()

# %%
f, ax = plt.subplots( len(col_var), 2, figsize=(2*5,len(col_var)*5 )   )

for i,var in enumerate(col_var):
    print(var)
    ax[i, 0].plot(  np.mean(data_X[var].values[0], axis=(1,2)) );
    ax[i, 0].set_title(f"{var} - mean")
    ax[i, 1].plot(  np.std(data_X[var].values[0], axis=(1,2)), color='red' );
    ax[i, 1].set_title(f"{var} - std")

# %%
f, ax = plt.subplots( len(pred_var), 2, figsize=(2*5,len(pred_var)*5 )   )

for i,var in enumerate(pred_var):
    print(var)
    ax[i, 0].plot(  np.mean(data_Y[var].values[0], axis=(1,2)) );
    ax[i, 0].set_title(f"{var} - mean")
    ax[i, 1].plot(  np.std(data_Y[var].values[0], axis=(1,2)), color='red' );
    ax[i, 1].set_title(f"{var} - std")

# %%
f, ax = plt.subplots( len(pred_var), 2, figsize=(2*5,len(pred_var)*5 )   )

for i,var in enumerate(pred_var):
    print(var)
    delta = data_Y[var].values[0][1:] - data_Y[var].values[0][:-1]

    ax[i, 0].plot(  np.mean(delta, axis=(1,2)) );
    ax[i, 0].set_title(f"{var} - mean")
    ax[i, 1].plot(  np.std(delta, axis=(1,2)), color='red' );
    ax[i, 1].set_title(f"{var} - std")

# %%
data_X['emis'].values.min()

# %%
np.reshape(data_X[var].values[0], (72, -1))

# %% [markdown]
# plt.plot(da)

# %%


# %%



