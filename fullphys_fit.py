
# # Main fit notebook


# # Initialisation


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
experiment_name = 'Unet_20230830'

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
# Construct folders:
os.makedirs(f"{output_path}", exist_ok=test)
os.makedirs(f"{graph_path}", exist_ok=test)
os.makedirs(f"{checkpoint_path}", exist_ok=test)
# Save Parameters:
os.system(f"cp params.py {output_path}/param.txt")

#if not test:
#    logging.basicConfig(filename=f'{output_path}/std.log', filemode='w') #, format='%(name)s - %(levelname)s - %(message)s')


# ## Construct Generators :

# %%
with open(preprocess_X_path, 'rb') as handle:
     preprocessor_x = pickle.load(handle)
with open(preprocess_Y_path, 'rb') as handle:
     preprocessor_y = pickle.load(handle)
     
data_loader = PhysicGenerator(data_path=data_path,
                   nb_portions=1,
                   batch_size=32,
                   input_variables = input_variables,
                   output_variables = pred_variables,
                   file_keys=['X', 'Y'],
                   shuffle=True,
                   preprocessor_x = preprocessor_x,
                   preprocessor_y = preprocessor_y,
                   verbose=0,
                   device=device,
                   _max_length=nb_epochs)

data_loader_test = PhysicGenerator(data_path=data_path_test,
                   nb_portions=1,
                   batch_size=32,
                   input_variables = input_variables,
                   output_variables = pred_variables,
                   file_keys=['X', 'Y'],
                   shuffle=True,
                   preprocessor_x = preprocessor_x,
                   preprocessor_y = preprocessor_y,
                   verbose=0,
                   device=device,
                   _max_length=nb_epochs)


# ### Graphic Check :

# %%
from src.graphic.full_phys import show_all_faces

print('Min of output : \t', np.min(data_loader.Y,axis=(0,1)))
print('Max of output : \t', np.max(data_loader.Y,axis=(0,1)))
print('Std of output : \t', np.std(data_loader.Y,axis=(0,1)))
print('Mean of output : \t', np.mean(data_loader.Y,axis=(0,1)))

for variable in ['tao', 'u', 'v', 't']:
    f = show_all_faces(data_loader.X, 
                       id_lev=-1, 
                       id_var=data_loader.input_variables.index(variable), 
                       resolution=data_loader.x, var=variable);
    f.savefig(f"{graph_path}/{variable}.jpg")
    plt.show();

f2 = plt.figure(figsize=(15,15))
for i, var in enumerate(data_loader.input_variables):
    ax = f2.add_subplot(1+len(data_loader.input_variables)//4, 4, i+1)
    ax.plot(data_loader.X[:5, :, i].T)
    ax.set_title(var)
f2.savefig(f"{graph_path}/input_var_profile.jpg")

f3 = plt.figure(figsize=(15,15))
for i, var in enumerate(data_loader.output_variables):
    ax = f3.add_subplot(1+len(data_loader.input_variables)//4, 4, i+1)
    ax.plot(data_loader.Y[:5, :, i].T)
    ax.set_title(var)
f3.savefig(f"{graph_path}/output_var_profile.jpg")



# ### Data Analysis :
# 
# fill with analysis on input 

# %%
# Test 1: output should have mean 0 and std 1


# ## Model and Training

# %%
from src.models.unet import UNet
from torch import nn

model = UNet(lev=data_loader.z, 
             in_channels=len(data_loader.input_variables), 
             out_channels=len(data_loader.output_variables))


# ### Loss

# %%
huber_fn = nn.HuberLoss()
mse_fn = nn.MSELoss()
loss_fn = mse_fn


# ### Optimizer

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.95))


# ### Testing

# %%
with torch.no_grad():
    x, y = data_loader[0]
    y0 = model(x)
    loss_fn(y0, y)


# ## Traning Procedure

# %%
def train_n_epoch(num_epochs, run_test=False, saving_freq=1000, check_points=False):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    loss_list = []
    best_loss = +np.inf
    test_loss = +np.inf
    best_model_id = 0
    for e in range(num_epochs):
        T = time.time()
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
        if run_test:
            test_loss = run_test_fct()
            print('test', test_loss)
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_id = e
        data_loader.on_epoch_end()
        if check_points:
            print(f"End of epoch {e} \t: test_loss : {test_loss} \t best loss : {best_loss} \t best model {best_model_id}")
            torch.save(model, f'{check_points}_{e}.tch')
        print(f'Epoch {e} ran in {round(time.time(), 1) - T} s')
            
    return loss_list, best_model_id, best_loss


def run_test_fct():
    loss_item = 0
    with torch.no_grad():
        for i in range(len(data_loader_test)):
            data = data_loader[i]
            x, y = data            
            outputs = model(x)
            loss_item += loss_fn(outputs,y).item()
    loss_item /= len(data_loader_test)
    return loss_item


# ## Runs :

# %%
start_time = time.time()
loss_list, best_model_id, best_loss = train_n_epoch(nb_epochs, saving_freq=1000, run_test=True, check_points='Data/models/Unet')
end_time = time.time()
delta_t = int(end_time - start_time)

# %%
hours   = delta_t//3600
minutes =(delta_t % 3600) //60
secs    = delta_t % 60

print(f"Training ran in {delta_t}")
print(f"Training ran in {hours}h:{minutes}m:{secs}s")


