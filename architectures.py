import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Flatten, Input, TimeDistributed, Concatenate
from keras.layers import Conv1D, UpSampling1D, AveragePooling1D, SeparableConv1D
from keras.layers import Bidirectional, Lambda, Reshape
from keras.losses import mean_squared_error
from keras import backend as K
from keras import regularizers

import os
import numpy as np
from contextlib import redirect_stdout
from CST import CST
#### LOSSES :

def Name(layer,i):
    return layer+'_'+str(i)

def reshape(y, n_shape):
    y0=reshape(y.shape[0], n_shape[0], n_shape[1])
    return(y0)

def y_batch_reshape(y):
    return(reshape(y, CST.lev(CST)), CST.outputs_y(CST))


def one_loss(y_true, y_pred, i):
    E = mean_squared_error(y_true[:, :, i], y_pred[:, :, i])
    return E


# Normal Losses
def flxd_loss(y_true, y_pred):
    E = mean_squared_error(y_true[:, :, 0], y_pred[:, :, 0])
    return E
def flxu_loss(y_true, y_pred):
    E = mean_squared_error(y_true[:, :, 1], y_pred[:, :, 1])
    return E
def dfdts_loss(y_true, y_pred, coef=50):
    E = mean_squared_error(coef*y_true[:, :, 2], coef*y_pred[:, :, 2])
    return E

def total_loss(y_true, y_pred):
    E = flxd_loss(y_true, y_pred)
    E += flxu_loss(y_true, y_pred)
    E += dfdts_loss(y_true, y_pred)
    return E

# FC Losses
def total_loss_fc(y_true, y_pred, lev=72, n_input=3):
    E = flxd_loss_fc(y_true, y_pred)
    E += flxu_loss_fc(y_true, y_pred)
    E += dfdts_loss_fc(y_true, y_pred)
    return E

def flxd_loss_fc(y_true, y_pred, lev= CST.lev(CST)):
    E = mean_squared_error(y_true[:, :lev], y_pred[:, :lev])
    return E

def flxu_loss_fc(y_true, y_pred,lev= CST.lev(CST)):
    E = mean_squared_error(y_true[:, lev:(2*lev)], y_pred[:, lev:(2*lev)])
    return E

def dfdts_loss_fc(y_true, y_pred, coef=50, lev= CST.lev(CST)):
    E = mean_squared_error(coef*y_pred[:,(2*lev):(3*lev)], coef*y_pred[:, (2*lev):(3*lev)])
    return E



#### ARCHITECTURES

def Old_Bidir(in_channel=11, out_channel=3):
    modelbd = Sequential(name="Sequential_1")
    modelbd.add(Bidirectional(LSTM(128, return_sequences=True, use_bias=False),input_shape=(72, in_channel), name='Bidir'))
    modelbd.add(Conv1D(50, use_bias=False, kernel_size=8, padding='same', name='Conv_1'))
    modelbd.add(Activation('relu'))
    modelbd.add(Conv1D(20, kernel_size=5, padding='same', name='Conv_3'))
    modelbd.add(Activation('relu'))
    modelbd.add(Dense(out_channel, name='Dense'))
    return modelbd


def Classical_Bidir(in_channel=11, out_channel=3):
    modelbd = Sequential(name="Sequential_1")
    modelbd.add(Bidirectional(LSTM(128, return_sequences=True, use_bias=False),input_shape=(72, in_channel), name='Bidir'))
    modelbd.add(Conv1D(50, use_bias=False, kernel_size=8, padding='same', name='Conv_1'))
    modelbd.add(Activation('relu'))
    modelbd.add(Conv1D(50, kernel_size=5, padding='same', name='Conv_2'))
    modelbd.add(Activation('relu'))
    modelbd.add(Conv1D(20, kernel_size=3, padding='same', name='Conv_3'))
    modelbd.add(Activation('relu'))
    modelbd.add(Dense(out_channel, name='Dense'))
    return modelbd


def Averaged_Bidir(in_channel=11, out_channel=3):
    modelbd = Sequential(name="Sequential_1")
    modelbd.add(Bidirectional(LSTM(128, return_sequences=True, use_bias=False),input_shape=(72, in_channel), name='Bidir'))
    modelbd.add(Conv1D(50, use_bias=False, kernel_size=8, padding='same', name='Conv_1'))
    modelbd.add(Activation('relu'))
    modelbd.add(AveragePooling1D(7, padding='same', stride=1))
    modelbd.add(Conv1D(50, kernel_size=5, padding='same', name='Conv_2'))
    modelbd.add(Activation('relu'))
    modelbd.add(AveragePooling1D(4, padding='same', stride=1))
    modelbd.add(Conv1D(20, kernel_size=3, padding='same', name='Conv_3'))
    modelbd.add(Activation('relu'))
    modelbd.add(Dense(out_channel))
    return modelbd


def Weird_Averaged_Bidir(in_channel=11, out_channel=3):
    modelbd = Sequential(name="Sequential_1")
    modelbd.add(Bidirectional(LSTM(128, return_sequences=True, use_bias=False),input_shape=(72, in_channel), name='Bidir'))
    modelbd.add(Conv1D(50, use_bias=False, kernel_size=8, padding='same', name='Conv_1'))
    modelbd.add(AveragePooling1D(7, padding='same', stride=1))
    modelbd.add(Activation('relu'))
    modelbd.add(Conv1D(50, kernel_size=5, padding='same', name='Conv_2'))
    modelbd.add(AveragePooling1D(4, padding='same', stride=1))
    modelbd.add(Activation('relu'))
    modelbd.add(Conv1D(20, kernel_size=3, padding='same', name='Conv_3'))
    modelbd.add(Activation('relu'))
    modelbd.add(Dense(out_channel))
    return modelbd


def Upsampler(avg, pooling):
    M = Sequential(name="Upsampler")
    M.add(UpSampling1D(avg))
    M.add(AveragePooling1D(pooling, padding='same', stride=avg))
    return M


def Add_Upsampling(M_seq, shape, avg, pooling):
    newInput = Input( shape=shape, name="Input_1")
    M = Upsampler(avg, pooling)
    newOutputs  = M(newInput)
    newOutputs2 = M_seq(newOutputs)
    model2 = keras.Model(newInput, newOutputs2)
    return model2

#### Fully Conv :
def Bidir_Casual_Conv(list_of_kernel_s, list_of_filters, ups, pooling, in_channel, o_channel, lev=CST.lev(CST)):
    """
    used as input for the Unet
    """
    Input0 = Input(shape=(lev, in_channel), name=Name('Input',0), dtype='float32')

    Flip_layer = lambda x: K.reverse(x, axes=0)
    I_cp = UpSampling1D(ups, name=Name('Up',0))(Input0)
    I_avg = AveragePooling1D(pooling, padding='same', stride=ups, name='AVG_p')(I_cp)
    I_avg_flip = Lambda(Flip_layer, name=Name('Flip',0))(I_avg)

    Conv1u = [I_avg]
    Conv1d = [I_avg_flip]

    # Normal
    for i in range(len(list_of_filters[0])):
        Conv1u.append(Conv1D(filters = list_of_filters[0][i], kernel_size= list_of_kernel_s[0][i], \
                        padding='causal', activation='relu', name=Name("Conv_u",i+1), use_bias=True)(Conv1u[-1]))
    # Flipped
    for i in range(len(list_of_filters[1])):
        Conv1d.append(Conv1D(filters = list_of_filters[1][i], kernel_size= list_of_kernel_s[1][i],\
                        padding='causal', activation='relu', name=Name("Conv_d",i+1), use_bias=True)(Conv1d[-1]))

    C_flip = Lambda(Flip_layer,name=Name('Flip',1))(Conv1d[-1])
    C1d_prime = [Concatenate( name=Name('Concat',0))([Conv1u[-1], C_flip])]
    for i in range(len(list_of_filters[2])):
        if i==len(list_of_kernel_s)-1:
            C1d_prime.append(Conv1D(filters = list_of_filters[2][i], kernel_size=list_of_kernel_s[2][i], \
                            padding='causal', name=Name("Conv_conc",i), use_bias=False)(C1d_prime[-1]))
        else:
            C1d_prime.append(Conv1D(filters = list_of_filters[2][i], kernel_size=list_of_kernel_s[2][i], \
                            padding='causal', name=Name("Conv_conc",i), use_bias=False, activation='relu')(C1d_prime[-1]))
    return keras.Model(Input0, C1d_prime[-1])


def Unet(list_of_kernels_s, list_of_filters, list_of_pooling, Div=3, lev=CST.lev(CST), in_channel=11, o_channel=CST.output_y(CST)):
    """
    Generate a Unet-Archictecture
    list_of_kernels : list of 2 lists containing the kernel size for convolution
    list_of_filters : list of 2 lists containing the number of filters for convolution
    Div : number of downscaling
    in_channel : number of inputs
    """
    Concats_l = []
    Upsamplings_l = []
    Convs_l1 = []
    Convs_l2 = []
    Poolings_l = []
# DownScaling
    Convs_l1.append(Input(name = 'Origin_Input',  dtype='float32', shape=(lev, in_channel)))
    for i in range(Div):
        Poolings_l.append(AveragePooling1D(list_of_kernels_s[0][i]-1, padding='same', stride=2, name=Name('AVG', i+1))(Convs_l1[-1]))
        Convs_l1.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i], \
                                padding='same', activation='relu', name=Name('Conv1',i+1))( Poolings_l[-1] ))

# Operation done on the small dimension : here fc
    Convs_l2.append(Flatten()(Convs_l1[-1])  )
    Convs_l2.append(Dense( int(lev/2**Div) * list_of_filters[1][0]  )(Convs_l2[-1])  )
    Convs_l2.append(Reshape(name='Reshape',input_shape=Convs_l2[-1].shape ,\
                            target_shape=( int(lev/2**Div)  ,  list_of_filters[1][0] ))(Convs_l2[-1]))

# Upsampling and concats
    for i in range(Div):
        Concats_l.append(Concatenate( name=Name('Concat',i+1) )([Convs_l2[-1], Convs_l1[-i-1]]))
        Upsamplings_l.append(UpSampling1D(2, name=Name('Ups',i+1))(Concats_l[-1]))
        Convs_l2.append(Conv1D(filters=list_of_filters[2][i], kernel_size=list_of_kernels_s[2][i], \
                                padding='same', activation='relu', name=Name('Conv2',i+1))( Upsamplings_l[-1] ))
    Conv3 = Conv1D(filters=o_channel, kernel_size=1, padding='same', use_bias=False)(Convs_l2[-1])
    return keras.Model(Convs_l1[0],Conv3)
###### FC
def FixInputs_C(model, inputs):
    """
    Fix the ts variable in a FC architecture (considerer as the last variable)
    """
    first_input = K.constant(inputs[0])
    second_input = K.constant(inputs[1][:,:-1])

    Tensor_Input0 = Input(batch_shape = (model.input_shape[1][0], 1))

    n_input = keras.layers.Lambda(lambda x: K.concatenate([second_input,x],axis=-1))(Tensor_Input0)
    n2_input = keras.layers.Lambda(lambda x: [first_input, x])(n_input)
    Out1 = model(n2_input)
    Out2 = keras.layers.Lambda(lambda x : x)(Out1)
    M = keras.Model( Tensor_Input0, Out2  )
    return(M)


def FixInputsFC(model, inputs):
    """
    Fix the ts variable in a FC architecture (considerer as the last variable)
    """
    first_input = K.constant(inputs[0])
    second_input = K.constant(inputs[1][:,:-1])

    Tensor_Input0 = Input(batch_shape = (model.input_shape[1][0], 1))

    n_input = keras.layers.Lambda(lambda x: K.concatenate([second_input,x],axis=-1))(Tensor_Input0)
    n2_input = keras.layers.Lambda(lambda x: [first_input, x])(n_input)
    Out1 = model(n2_input)
#    Out2 = keras.layers.Lambda(lambda x : x[:,:,0] - x[:,:,1])(Out1)
    Out2 = keras.layers.Lambda(lambda x : x)(Out1)
    M = keras.Model( Tensor_Input0, Out2  )
    return(M)

def FC_archi(ups, pooling, list_of_units, BS, reg=0.0001, o_channel=3, lev= 72, unique_var = 3, level_var=8):
    """
    Create a FC neural network, with averaging
    ups : upsampling size
    pooling : int such that pooling/ups is the range of mean
    Reshape layer at the end stops any possiblity of test and is commented
    """
    # Two different types of input are made and merged
    All_Inputs = [Input(batch_shape=(BS, lev, level_var), name=Name('InputLev',0)), Input( batch_shape=(BS,unique_var), name=Name('InputU',1))]
    o1 = UpSampling1D(ups, name=Name('Up',0))(All_Inputs[0])
    o2 = AveragePooling1D(pooling, padding='same', stride=ups, name=Name('AVG',1))(o1)
    o3 = Flatten(name = Name('Flat',0))(o2)
    merged = Concatenate(name=Name('Concat',0))([o3, All_Inputs[1]])

    D = []
    D.append(Dense(list_of_units[0], use_bias=True, activation='relu', kernel_regularizer=regularizers.l2(reg), name=Name('Dense',0))(merged))
    for i, n_u in enumerate(list_of_units[1:]):
        D.append( Activation('relu', name=Name('Act',i) )(D[-1]))
        last_layer = (i!=len(list_of_units)-2)
        if(last_layer):
            D.append( Dense(units= n_u, use_bias=False, kernel_regularizer=regularizers.l2(reg), name=Name('Dense',i+1))(D[-1]))
        else:
            D.append( Dense(units= n_u, use_bias=True,  kernel_regularizer=regularizers.l2(reg), activation='relu', name=Name('Dense',i+1))(D[-1]))
#    D.append(Lambda(lambda x :return_print(x))(D[-1]))
#    print(lev)
#    D.append(keras.layers.Reshape((lev, o_channel))(D[-1]))
    return(keras.Model(All_Inputs, D[-1]))

#### GRAD METHOD
def Grad_ts(model_ffc, x):
    t0=time.time()
    model_fix = FixInputsFFC(model_ffc, x)
    ts = x[1][:,-1].reshape(-1,1)
    t = time.time()
    gradients = [ K.gradients(model_fix.get_output_at(0)[:,i], model_fix.input)[0] for i in range(72) ]
    grad0 = K.function( [model_fix.input] , gradients )
    o = np.array(grad0([ts]))[:,:,0]
    return(o)



### CALLBACKS

class SGDLearningRateTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, 'float32_ref'))))
        print('\nLR: {:.6f}\n'.format(lr))

class LossHistory(keras.callbacks.Callback):
    """
    Callback that keep an historic of the losses with more frequency
    """
    def __init__(self, frequency=1000):
        super(LossHistory, self).__init__()
        self.frequency=frequency

    @property
    def loss_name(self):
        return(['flxu_loss', 'flxd_loss', 'dfdts_loss', 'loss'])

    def __getitem__(self,i):
        return(self.losses.__getitem__(i))

    def on_train_begin(self, logs={}):
        self.losses = dict()
        for n in self.loss_name:
            self.losses[n] = []

    def on_batch_end(self, batch, logs={}):
        #print(logs['batch'])
        if(batch%self.frequency==0):
            for n in self.loss_name:
                self.losses[n].append( logs.get(n))

    def on_train_end(self, logs={}):
        for n in self.loss_name:
            self.losses[n] = np.array(self.losses[n])

def Return_print(x):
    """
    Used as lambda layer for printing shapes
    """
    print(x.shape)
    return(x)

def StructAssertion(L,Div):
    assert(len(L)==2)
    assert(len(L[0])==Div)
    assert(len(L[1])==Div)


def Generate_Log(models, history, callback, file ,seed):
    """
    Generate a log file, after the training
    models : the list of model used, whose summary are printed
    history : fit_generator output variables
    callback : instance of LossHistory used for training
    file : file where the log is written
    seed : used seed
    """
    with open(file, 'w') as f:
        with redirect_stdout(f):
            print('Seed {}'.format(seed))
            for model in models:
                model.summary()
            for k in history.history.keys():
                print(k,':', history.history[k])
            print('\n')
            for loss in callback.losses:
                print(loss)

#####
