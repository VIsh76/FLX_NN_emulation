# %% [markdown]
# # FIT

# %% [markdown]
# ### TEST

# %%
from keras.layers import Conv1D, UpSampling1D, MaxPooling1D, AveragePooling1D, SpatialDropout1D
from keras.layers import Lambda, Reshape, Dropout,  Dense, Flatten, Input, Concatenate


surface_var = ['emis', 'frlake', 'frland', 'frlandice', 'frocean', 'frseaice', 'ts']
pred_var = ['flcd', 'flcu', 'flxd', 'flxu']
pred_var = ['flxd', 'flxu']
col_var = ['fcld', 'o3', 'pl', 'q', 'qi', 'ql', 'ri', 'rl', 't']

in_channel = len(surface_var + col_var)
out_channel = len(pred_var)

lev = 72
input_shape = (lev, in_channel)
data_format = 'channels_last'


preprocessor_path = "Data/preprocess/test_basic_preprocessor_0.pickle"


# %% [markdown]
# ### PREPROCESSOR

# %%
from src.load_data.generator import PreprocessGenerator
import matplotlib.pyplot as plt
import pickle

with open(preprocessor_path, 'rb') as handle:
    preprocessor_x = pickle.load(handle)


PG = PreprocessGenerator(data_path="Data/train/",
                   nb_portions=40,
                   batch_size=32,
                   input_variables=col_var+surface_var,
                   output_variables=pred_var,
                   shuffle=False,
                   preprocessor_x= preprocessor_x,
                   test=False,
                   _max_length=10_000) # 97_200

print(len(PG))

# %% [markdown]
# ### Check if the data are corrects

# %%
f = plt.figure(figsize=(15,15))
for i, var in enumerate(PG.input_variables):
    ax = f.add_subplot(len(PG.input_variables)//4, 4, i+1)
    ax.plot(PG.X[:10, :, i].T)
    ax.set_title(var)

# %% [markdown]
# ## ARCHITECTURE CREATION

# %%
from src.models.architectures import Unet_Act_Simple,  Bidir_Casual_Conv, Upsampler
# from src.models.architectures_utils import total_loss, flxd_loss, flxu_loss, dfdts_loss, LossHistory,  Generate_Log, Activation_Generator

import keras
import keras.backend as K
from keras import optimizers
from keras.layers import Dropout
import datetime

list_of_filters_bdc =  [[64, 100, 32], [32, 20, 10], [100]]
list_of_kernel_bdc =  [[12, 5, 5], [10, 5, 3], [5, 3]]
list_of_activation_bdc =  [['elu', 'elu', 'elu'], ['elu', 'elu', 'elu'], ['relu']]

list_of_filters_unet =  [[50, 25, 100], [270], [50, 100, 200], [300, len(pred_var)]]
list_of_kernel_unet =  [[10, 5, 3], [], [6, 10, 20], [5, out_channel]]
list_of_activation_unet =  [['elu', 'elu', 'elu'], ['sigmoid'], ['elu', 'elu', 'elu'], ['elu', 'linear']]
params = [0.1]

M_Up = Upsampler(avg=5, pooling=22, input_shape=(lev, in_channel))
M_bd =  Bidir_Casual_Conv(list_of_kernel_bdc, list_of_filters_bdc, list_of_activation_bdc, params, in_channel, lev)

M_unet =  Unet_Act_Simple(list_of_kernel_unet, list_of_filters_unet, list_of_activation_unet,
                         params, Div=3, lev=lev, in_channel=in_channel)

# M = keras.models.Sequential()
# M.add(M_Up); 
# M.add(M_bd); 
# M.add(M_unet); 
#M.layers[-1].name = 'Unet'

# %%
from keras import optimizers
from keras.losses import mean_squared_error

# CALLBACKS :
CheckPoint = keras.callbacks.ModelCheckpoint(filepath='Data/model.{epoch:02d}.h5'),

#ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, restore_best_weights=True, baseline=35)
prefix = '20190530154223U'

# LH = LossHistory(frequency=train_generator.batch_per_file, losses=['flxu_loss', 'flxd_loss', 'loss'])
Adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1.e-6, amsgrad=False)
M_unet.compile(loss=mean_squared_error, optimizer=Adam)

# %%
M_unet.summary();

# %%
M_unet.fit(x=PG, epochs=9, verbose=2, shuffle=False, callbacks=[CheckPoint])

# %%
M_unet.save('model_1.hdf5')

# %%
Y_pred = M_unet(PG[0][0]).numpy()

f = plt.figure(figsize=(5,5))
for i, var in enumerate(PG.output_variables):
    ax = f.add_subplot(1, len(PG.output_variables), i+1)
    ax.plot(PG.Y[:10, :, i].T)
    ax.set_title(var)

# %%
Y_pred = M_unet(PG[0][0]).numpy()

f = plt.figure(figsize=(5,5))
for i, var in enumerate(PG.output_variables):
    ax = f.add_subplot(1, len(PG.output_variables), i+1)
    ax.plot(Y_pred[:10, :, i].T)
    ax.set_title(var)


