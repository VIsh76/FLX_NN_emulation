import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Flatten, Input, TimeDistributed, Concatenate
from keras.layers import Conv1D, UpSampling1D, AveragePooling1D, SeparableConv1D
from keras.layers import Bidirectional, Lambda, Reshape
from keras.losses import mean_squared_error
from keras import backend as K
from keras import regularizers
from keras.layers import Layer
from architectures_utils import Name

import os
import numpy as np
from contextlib import redirect_stdout
from CST import CST

# Simples architecture are saved

def Upsampler(avg, pooling, input_shape):
    """
    Generate a Upsampler model, perform a
    avg : size of upsampling
    pooling : size of pooling
    input_shape : shape of the input
    adverage is perform on avg/pooling terms
    """
    Input0 = Input(shape=input_shape)
    Up = UpSampling1D(avg)(Input0)
    Avg = AveragePooling1D(pooling, padding='same', stride=avg)(Up)
    return keras.Model(Input0, Avg)


def Divide_Recombine(o_channel, in_channel,lev=CST.lev(CST), reg=0.001):
    """
    Generate a pseudo-fully connected layer where each layer of the profile is fully connected to its outputs only
    """
    Input0 = Input(shape=(in_channel,lev), name='Input_RC0')
    Input1 = Flatten(name='Last_flatten')(Input0)
    D = [ Dense(lev, kernel_regularizer = keras.regularizers.l2(reg), name=Name('Last_Dense',i))(Input1) for i in range(o_channel)]
    R = [Reshape(target_shape=(lev,1))(D[i]) for i in range(o_channel)]
    C = keras.layers.Concatenate(axis=-1, name='Last_Concat')(R)
    model = keras.Model(Input0, C)
    return(model)

def Divide_Substract(o_channel, in_channel,lev=CST.lev(CST), reg=0.00001):
    """
    Generate several Dense layer from the same input and combine them
    """
    Input0 = Input(shape=(in_channel,lev), name='Input_RC0')
    Input1 = Flatten(name='Last_flatten')(Input0)
    D = [ Dense(lev, kernel_regularizer = keras.regularizers.l2(reg), name=Name('Last_Dense',i))(Input1) for i in range(o_channel)]
    R = [Reshape(target_shape=(lev,1))(D[i]) for i in range(o_channel)]
    C = keras.layers.Concatenate(axis=-1, name='Last_Concat')(R)
    Sub_f = lambda x: x[:,:,0] - x[:,:,1]
    Sub = keras.layers.Lambda(Sub_f)(C)
    model = keras.Model(Input0, Sub)
    return(model)

# Dict Makers Models
def MakeDictMatrix(D, header, lev=72):
    """
    From a prepross dictionnary return two matrix, one for substraction one for multiplication
    Doing (X-Ms)*Mp is equivalent to applying the preproc normalization
    """
    Ms = np.zeros((1, lev, len(header)))
    Mp = np.ones((1, lev, len(header)))
    for var in D.dict.keys():
        i = header.index(var)
        Ms[0, :, i] = D[var].sub_vec(lev)
        Mp[0, :, i] = D[var].prod_vec(lev)
    return(Ms,Mp)

def MakeDictNet(Ms,Mp):
    """
    From the output of MakeDictMatrix, generate from the two matrix,
    a keras model performing (X-Ms)*Mp
    """
    Ts1 = tf.cast(Ms, dtype=tf.float32)
    Tp1 = tf.cast(Mp, dtype=tf.float32)
    D1s = lambda x : keras.layers.Subtract()([x, Ts1])
    D1m = lambda x : keras.layers.Multiply()([x, Tp1])
    lbd_D1s = keras.layers.Lambda(D1s)
    lbd_D1m = keras.layers.Lambda(D1m)
    M = keras.Sequential()
    M.add(lbd_D1s)
    M.add(lbd_D1m)
    return(M)

class ConvDP2(Layer):
    """
    Stacked Layers of :
    - Convolution, Activation function,
    can be used for more clarity but slow down the training (hence not used)
    """
    def __init__(self, nb_filter, ks, act, dpr, bias, reg, ide, **kwargs):
        """
        nb_filters : number of filter
        ks : (int) kernel_size
        act : (int) activation
        dpr : dropout rate
        bias : (bool) use_bias
        reg : reg weight
        id : (int) ID for layer's name
        """
        self.nb_filter = nb_filter
        self.ks = ks
        if(type(ks)==int):
            self.ks= (ks,ks)
        self.act = act
        self.dpr = dpr
        self.reg = reg
        self.ide = ide
        self.bias = bias
#        print('ks', ks, 'act',act,'dpr', dpr, 'reg',reg, 'ide',ide, 'bias',bias)
        super(ConvDP2, self).__init__(**kwargs)
        self.name = Name('ConvDP2', self.ide)

    def build(self, input_shape):
        AG = Activation_Generator()
        self.c2d = Conv2D(input_shape=input_shape, nb_filter=self.nb_filter,  padding='same', use_bias=self.bias,
                          kernel_size=self.ks, activity_regularizer=regularizers.l2(self.reg))
        self.activ =  AG(self.act, Name(self.act, self.ide), 0.01)
        self.d = Dropout(self.dpr)
        super(ConvDP2, self).build(input_shape)

    @property
    def trainable_weights(self):
        return self.c2d.trainable_weights

    def call(self,x):
        return self.d(self.activ(self.c2d(x)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.nb_filter)
# 2D convertion
class Perturbate(Layer):
    """
    Simulated perturbation (add a matrix to the input)
    """
    def __init__(self, **kwargs):
#        self.output_dim = output_dim
        self.SumL = keras.layers.Add()
        super(Perturbate, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], input_shape[2], input_shape[3]),
                                      initializer='uniform',
                                      trainable=True)
        super(Perturbate, self).build(input_shape)

    def call(self, x):
        print((x+self.kernel).shape)
        return x+self.kernel

    def compute_output_shape(self, input_shape):
         return (input_shape[1],input_shape[2], input_shape[3])

def Expander(lev):
    """
    Return a model that expand an input of size (lev, n_var) to (lev, lev, n_var)
    """
    expand = lambda x : K.expand_dims(x, axis=-1)
    repeat = lambda x : K.repeat_elements(x, lev, axis=-1)
    Expand = Lambda(expand)
    Repeat = Lambda(repeat)
    M = Sequential()
    M.add(Expand)
    M.add(Repeat)
    return(M)

#### OLD ARCHITECTURES, replace by Bidir_Casual_Conv

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



def Add_Upsampling(M_seq, shape, avg, pooling):
    newInput = Input( shape=shape, name="Input_1")
    M = Upsampler(avg, pooling)
    newOutputs  = M(newInput)
    newOutputs2 = M_seq(newOutputs)
    model2 = keras.Model(newInput, newOutputs2)
    return model2


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
