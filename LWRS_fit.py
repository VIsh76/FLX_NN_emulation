# %% [markdown]
# # FIT

# %%
# Parameters :

surface_var = ['emis', 'frlake', 'frland', 'frlandice', 'frocean', 'frseaice', 'ts']
pred_var = ['flcd', 'flcu', 'flxd', 'flxu']
pred_var = ['flxd', 'flxu']
col_var = ['fcld', 'o3', 'pl', 'q', 'qi', 'ql', 'ri', 'rl', 't']

in_channel = len(surface_var + col_var)
out_channel = len(pred_var)
preprocessor_path = "Data/preprocess/test_basic_preprocessor_0.pickle"

# %% [markdown]
# ### PREPROCESSOR

# %%
from src.load_data.generator import PreprocessColumnGenerator
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open(preprocessor_path, 'rb') as handle:
    preprocessor_x = pickle.load(handle)


data_loader = PreprocessColumnGenerator(data_path="Data/train/",
                   nb_portions=80,
                   batch_size=32,
                   input_variables=col_var+surface_var,
                   output_variables=pred_var,
                   file_keys=['X', 'Y'],
                   shuffle=False,
                   preprocessor_x= preprocessor_x,
                   test=False,
                   _max_length=1000) # 97_200

# %%
print(data_loader.X.shape)
print(np.min(data_loader.Y,axis=(0,1)))
print(np.max(data_loader.Y,axis=(0,1)))
print(np.amin(data_loader.Y,axis=(0,1)))

# %% [markdown]
# ### Check if the data are corrects

# %%
f = plt.figure(figsize=(15,15))
for i, var in enumerate(data_loader.input_variables):
    ax = f.add_subplot(len(data_loader.input_variables)//4, 4, i+1)
    ax.plot(data_loader.X[:10, :, i].T)
    ax.set_title(var)

# %% [markdown]
# ## ARCHITECTURE CREATION

# %%
from src.models.unet import UNet

model = UNet(lev=data_loader.z, in_channels=len(surface_var) + len(col_var), out_channels=2)

# %% [markdown]
# ## LOSS

# %%
from torch import nn
import torch 

loss_fn = nn.MSELoss()
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
def train_one_epoch(saving_freq=1000):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    loss_list = []
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
    return loss_list

# %% [markdown]
# ### TRACKERS :

# %%
loss_list = train_one_epoch(saving_freq=100)


