from src.load_data.generator import PhysicGenerator
import matplotlib.pyplot as plt
import pickle

# Parameters :
from params import input_variables, pred_variables
from params import preprocess_X_path, preprocess_Y_path
from params import data_path, graph_path, output_path

with open(preprocess_X_path, 'rb') as handle:
     preprocessor_x = pickle.load(handle)
with open(preprocess_Y_path, 'rb') as handle:
     preprocessor_y = pickle.load(handle)


data_loader = PhysicGenerator(data_path = data_path,
                   nb_portions=1,
                   batch_size=32,
                   input_variables = input_variables,
                   output_variables = pred_variables,
                   file_keys=['X', 'Y'],
                   shuffle=True,
                   preprocessor_x = preprocessor_x,
                   preprocessor_y = preprocessor_y,
                   verbose=1,
                   _max_length=-1)

data_loader_test=0


plt.plot(data_loader.X[31, :, 1]);
plt.plot(data_loader[0][0][-1,1] - 1 );

import xarray as xr
from src.graphic.full_phys import show_all_faces

show_all_faces(data_loader.X, 0, id_var=2, resolution=48)

print("Show reshaping is made correctly")
l = xr.open_dataset(data_loader.dict_of_files['X'][0])
print(l['t'].data.shape)
id_lev=-1
f = plt.figure()
for i in range(6):
    ax = f.add_subplot(2,3,i+1)
    plt.imshow(l['t'].data[0, id_lev, i, :, :])
plt.show()


# %%
f = plt.figure(figsize=(15,15))
for i, var in enumerate(data_loader.input_variables):
    ax = f.add_subplot(1+len(data_loader.input_variables)//4, 4, i+1)
    ax.plot(data_loader.X[:10, :, i].T)
    ax.set_title(var)

# %%
f = plt.figure(figsize=(15,15))
for i, var in enumerate(data_loader.input_variables):
    ax = f.add_subplot(1+len(data_loader.input_variables)//4, 4, i+1)
    ax.plot(data_loader.X[:10, :, i].T)
    ax.set_title(var)

# %%
import torch
import numpy as np
from src.models.unet import UNet
import torch

start=0
print(data_loader.Y.shape)
#torch.from_numpy(np.swapaxes(self.X[start:start+self.batch_size], 1, 2).astype(np.float32)), 
y = np.swapaxes(data_loader.Y[start:start+data_loader.batch_size], 1, 2).astype(np.float32)
yt = torch.from_numpy(y)


model = UNet(lev=data_loader.z, in_channels=len(input_variables), out_channels=len(pred_variables))
with torch.no_grad():
    model(data_loader[0][0] );

# %% [markdown]
# ## LOSS

# %%
from torch import nn
import torch 

huber_fn = nn.HuberLoss()
mse_fn = nn.MSELoss()
loss_fn = huber_fn

# TESTING REDUCTION
loss_fn(torch.zeros((32, 2, 72)), 
        torch.ones((32, 2, 72)))

# %% [markdown]
# ## OPTIMIZER

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# %% [markdown]
# ### TESTING :

# %%
x, y = data_loader[0]
loss_fn(model(x), y)

# %% [markdown]
# ### TRAINING PROCEDURE

# %%
def train_n_epoch(num_epochs, test=False, saving_freq=1000, early_stopping=0):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    loss_list = []
    best_loss = -np.inf
    test_loss = -np.inf
    best_model_id = 0
    for e in range(num_epochs):
        print(f'epoch - {e}')
        for i in range(len(data_loader)):
            # Every data instance is an input + label pair
            data = data_loader[i]
            x, y = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(x)

            # Compute the loss and its gradients
            loss = loss_fn(outputs,y)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % saving_freq == (saving_freq-1):
                last_loss = running_loss / saving_freq # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                loss_list.append(last_loss)
                running_loss = 0.                
            if test:
                test_loss = run_test()
        data_loader.on_epoch_end()
    return loss_list


def run_test():
    loss_item = 0
    with torch.no_grad():
        for i in range(len(data_loader_test)):
            data = data_loader[i]
            x, y = data            
            outputs = model(x)
            loss_item += loss_fn(outputs,y).item()
    return loss_item

loss_list = train_n_epoch(1, saving_freq=100)
with torch.no_grad():
    x, y = data_loader[1]
    y_pred = model(x)

simple_pred = np.mean(data_loader.Y, axis=0)

ml_mse_err = ((y_pred -y).numpy())**2
mean_mse_err = (y.numpy().T - np.expand_dims(simple_pred, axis=-1))**2
ml_mae_err = abs((y_pred -y).numpy())
mean_mae_err = abs(y.numpy().T - np.expand_dims(simple_pred, axis=-1))

print('ml_mse_err'   , ': \t', ml_mse_err.mean())
print('mean_mse_err' , ': \t', mean_mse_err.mean())
print('ml_msa_err'   , ': \t', ml_mae_err.mean())
print('meam_msa_err' , ': \t', mean_mae_err.mean())

# %%
loss_fn

# %%
f = plt.figure(figsize=(15, len(pred_variables)*5))

for i, var in enumerate(pred_variables):
    ax_0 = f.add_subplot(len(pred_variables), 2, 2*i+1);
    ax_0.plot(y_pred[:, i].T);
    ax_0.set_title(var+' pred')
    ax_1 = f.add_subplot(len(pred_variables), 2, 2*i+2);
    ax_1.plot(y[:, i].T); 
    ax_1.set_title(var+' truth')

# %%
import datetime

time = datetime.datetime.now()
torch.save(model, f'Data/models/UNET_{time}.tch')

# %%
print(loss_fn(y_pred[:, 0], y[:, 0]))
print(loss_fn(y_pred[:, 1], y[:, 1]))
print(loss_fn(y_pred[:, 2], y[:, 2]))


