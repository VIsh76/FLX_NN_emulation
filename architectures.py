from CST import CST
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Flatten, Input, TimeDistributed, Concatenate
from keras.layers import Conv1D, UpSampling1D,  SeparableConv1D, MaxPooling1D, Dropout,AveragePooling1D
from keras.layers import UpSampling2D, Conv2D, AveragePooling2D

from keras.layers import Bidirectional, Lambda, Reshape, SpatialDropout1D
from keras.losses import mean_squared_error
from keras import backend as K
from keras import regularizers
import tensorflow as tf
import os
import numpy as np
from contextlib import redirect_stdout
from CST import CST
from architectures_utils import Name, Activation_Generator

"""
This file produce complex architectures such as the Bidirectional ConvNet and several Unet"
"""

##################### BD casual Conv
def Bidir_Casual_Conv(list_of_kernel_s, list_of_filters, list_of_activations, params, in_channel, lev=CST.lev(CST), reg=0.0001):
    """
    all list argument must be of length 3, each sublist of the same size as list_of_kernels' sublists
    list_xx[0] : parameters for the up-convolutions
    list_xx[1] : parameters for the down-Convolutions
    list_xx[2] : parameters for the concatenation of the up and down convs
    list_xx[2] : parameters for the concatenation of the up and down convs
    kernels : kernel size
    filters : number of filters
    activations : activation function names for Activation_Generator()
    params : parameters for activation function
    (in_channel, lev) : shape of input
    reg : regularizers weights for every layer
    """
    Input0 = Input(shape=(lev, in_channel), name=Name('Input',0), dtype='float32')
    Flip_layer = lambda x: K.reverse(x, axes=0)
    I_flip = Lambda(Flip_layer, name=Name('Flip',0))(Input0)

    Conv1u = [Input0]
    Conv1d = [I_flip]
    AG = Activation_Generator()

    # Normal causal
    for i in range(len(list_of_filters[0])):
        Conv1u.append(Conv1D(filters = list_of_filters[0][i], kernel_size= list_of_kernel_s[0][i], \
                        padding='causal', name=Name("Conv_u",i+1), use_bias=True, kernel_regularizer=regularizers.l2(reg))(Conv1u[-1]))
        Conv1u.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i]+'_u', i+1), params )(Conv1u[-1]) )

    # Flipped causal
    for i in range(len(list_of_filters[1])):
        Conv1d.append(Conv1D(filters = list_of_filters[1][i], kernel_size= list_of_kernel_s[1][i],\
                        padding='causal', name=Name("Conv_d",i+1), use_bias=True, kernel_regularizer=regularizers.l2(reg))(Conv1d[-1]))
        Conv1d.append( AG(list_of_activations[1][i], Name(list_of_activations[1][i]+'_d', i+1), params )(Conv1d[-1]) )

    C_flip = Lambda(Flip_layer,name=Name('Flip',1))(Conv1d[-1])
    C1d_prime = [Concatenate( name=Name('Concat',0))([Conv1u[-1], C_flip])]

    # Conv of the concatenation
    for i in range(len(list_of_filters[2])):
        C1d_prime.append(Conv1D(filters = list_of_filters[2][i], kernel_size=list_of_kernel_s[2][i], \
                            padding='same', name=Name("Conv_c",i), use_bias=True, kernel_regularizer=regularizers.l2(reg))(C1d_prime[-1]))
        C1d_prime.append( AG(list_of_activations[2][i], Name(list_of_activations[2][i]+'_c', 1+i), params )(C1d_prime[-1]) )
    return keras.Model(Input0, C1d_prime[-1])


##################### Unet Simple
def Unet_Act_Double(list_of_kernels_s, list_of_filters, list_of_activations, params=[], Div=3, lev=CST.lev(CST), in_channel=11, reg=1e-7 ):
    """
    Generate a Unet-Archictecture:
    list_xx[0] : parameters for the starting convolutions
    list_xx[1] : parameters for the downscaling convolutions
    list_xx[2] : parameters for small space convolutions
    list_xx[2] : parameters for upsampling convolutions
    list_xx[3] : parameters for the last convolutions
    lists must be the same size (expect for list_of_kernel_s[3] which is not used)

    kernel : size of kernels
    filter : number of filters
    activation : names of the activation
    dropout : dropout rate

    params : params used for activation
    Div : number of downscaling
    in_channel : input column number
    lev : input ligne number
    reg : weights loss

    # The number of convolution at each up/downsampling is 2 so :
    list_of_xxx[1]/list_of_xxx[3] must be of len Div*2

    # The third element of the list correspond to fully connected so :
    list_of_kernel_s[3] is not used
    list_of_filters[3] is the number of filter divided by lev/div**2 (9 with default params)
    # This is changed in Unet_Act_Double_DP
    """
    AG = Activation_Generator() # Generate activation layers
    Sizes = [ len(list_of_filters[i]) for i in range(len(list_of_filters)) ]
# First Set of Convolutions
    Conv_l0 = [Input(name='Origin_Input', dtype='float32', shape=(lev, in_channel))]
    for i in range(Sizes[0]):
        Conv_l0.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i],
                            padding='same', use_bias=False, name=Name('Conv', i),
                            kernel_regularizer=regularizers.l1(reg))(Conv_l0[-1]))
        Conv_l0.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], i), params)(Conv_l0[-1]))
# DownScaling (Div*2 Convolutions)
    Conv_l1 = [Conv_l0[-1]]
    for i in range(Sizes[1]//2):
        Conv_l1.append(AveragePooling1D(2, padding='same', stride=2, name=Name('AVG', i+100))(Conv_l1[-1]))
        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i], kernel_size=list_of_kernels_s[1][2*i],
                               padding='same', name=Name('Conv', i+100))(Conv_l1[-1] ))
        Conv_l1.append(AG(list_of_activations[1][2*i], Name(list_of_activations[1][2*i], 100+i), params )(Conv_l1[-1]) )

        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i+1], kernel_size=list_of_kernels_s[1][2*i+1],
                               padding='same', name=Name('Conv', 110+i))(Conv_l1[-1]) )
        Conv_l1.append(AG(list_of_activations[1][2*i+1], Name(list_of_activations[1][2*i+1], 110+i), params )(Conv_l1[-1]))

# Fully connection done on the smallest vectors
    Conv_l2 = [Flatten(name='Flatten')(Conv_l1[-1])]
    for i in range(Sizes[2]):
        Conv_l2.append( Dense( int(lev/2**Div) * list_of_filters[2][i], name=Name('Dense', i), kernel_regularizer=regularizers.l2(reg))(Conv_l2[-1]))
        Conv_l2.append(AG(list_of_activations[2][i], list_of_activations[2][i]+'_d_'+str(i), params)(Conv_l2[-1]))

    Conv_l2.append(Reshape(name='Reshape', input_shape=Conv_l2[-1].shape,
                            target_shape=(int(lev/2**Div),  list_of_filters[2][-1]))(Conv_l2[-1]))
    Conv_l3 = [Conv_l2[-1]]

# Upsampling and concats
    for i in range(Sizes[3]//2):
        Conv_l3.append( Concatenate( name=Name('Concat',i+300) )([Conv_l3[-1], Conv_l1[-(1+i*5)]]))
        Conv_l3.append(UpSampling1D(2, name=Name('Ups',i+200))(Conv_l3[-1]))
        Conv_l3.append(Conv1D(filters=list_of_filters[3][2*i], kernel_size=list_of_kernels_s[3][2*i], padding='same',
                               name=Name('Conv', i+200))(Conv_l3[-1] ))
        Conv_l3.append( AG(list_of_activations[3][2*i], Name(list_of_activations[3][2*i], 200+i), params )(Conv_l3[-1]) )

        Conv_l3.append(Conv1D(filters=list_of_filters[3][2*i+1], kernel_size=list_of_kernels_s[3][2*i+1], padding='same',
                               name=Name('Conv', i + 210))(Conv_l3[-1]))
        Conv_l3.append(AG(list_of_activations[3][2*i+1], Name(list_of_activations[3][2*i+1], 210 + i), params)(Conv_l3[-1]))

# Last Conv layers
    Conv_l4 = [Conv_l3[-1]]
    for i in range(Sizes[4]):
        Conv_l4.append(Conv1D(filters=list_of_filters[4][i], kernel_size=list_of_kernels_s[4][i],
                            padding='same', use_bias=False, name=Name('Conv3',i+300),
                            kernel_regularizer=regularizers.l2(reg))(Conv_l4[-1]))

        Conv_l4.append( AG(list_of_activations[4][i], Name(list_of_activations[4][i], 300+i), params )(Conv_l4[-1]) )
    return keras.Model(Conv_l0[0], Conv_l4[-1])

############Unet with DP and no fullyConnected
def Unet_Act_Double_DP(list_of_kernels_s, list_of_filters, list_of_activations, list_of_dropout, params=[], Div=3, lev=CST.lev(CST), in_channel=11, reg=0.001 ):
    """
    Generate a Unet-Archictecture:
    list_of_xxx : list of 5 list containing info for xxx
    list_xxx[0] : parameters for the starting convolutions
    list_xxx[1] : parameters for the downscaling convolutions
    list_xxx[2] : parameters for small space convolutions
    list_xxx[3] : parameters for upsampling convolutions
    list_xxx[4] : parameters for the last convolutions
    lists must be the same size (expect for list_of_kernel_s[3] which is not used)

    kernel : size of kernels
    filter : number of filters
    activation : names of the activation
    dropout : dropout rate

    params : params used for activation
    Div : number of downscaling
    in_channel : input column number
    lev : input ligne number
    reg : weights loss

    # The number of convolution at each up/downsampling is 2 so :
    list_of_xxx[1]/list_of_xxx[3] must be of len Div*2

    # The third element of the list correspond to a convolution with kernel_size=vector size so :
    list_of_kernel_s[3] is not used
    list_of_filters[3] is the number of filter (unlike previous Unet)
    """
    AG = Activation_Generator()
    Sizes = [ len(list_of_filters[i]) for i in range(len(list_of_filters)) ]
#First Convolutions
    Conv_l0 = [Input(name='Origin_Input', dtype='float32', shape=(lev, in_channel))]
    for i in range(Sizes[0]):
        Conv_l0.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i],
                            padding='same', use_bias=False, name=Name('Conv', i),
                            kernel_regularizer=regularizers.l1(reg))(Conv_l0[-1]))
        Conv_l0.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], i), params)(Conv_l0[-1]))
        Conv_l0.append( SpatialDropout1D(list_of_dropout[0][i])(Conv_l0[-1]))

# DownScaling
    Conv_l1 = [Conv_l0[-1]]
    for i in range(Sizes[1]//2):
        Conv_l1.append(AveragePooling1D(2, padding='same', stride=2, name=Name('AVG', i+100))(Conv_l1[-1]))
        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i], kernel_size=list_of_kernels_s[1][2*i],
                               padding='same', name=Name('Conv', i+100))(Conv_l1[-1] ))
        Conv_l1.append(AG(list_of_activations[1][2*i], Name(list_of_activations[1][2*i], 100+i), params )(Conv_l1[-1]) )
        Conv_l1.append( SpatialDropout1D(list_of_dropout[1][2*i])(Conv_l1[-1]))

        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i+1], kernel_size=list_of_kernels_s[1][2*i+1],
                               padding='same', name=Name('Conv', 110+i))(Conv_l1[-1]) )
        Conv_l1.append(AG(list_of_activations[1][2*i+1], Name(list_of_activations[1][2*i+1], 110+i), params )(Conv_l1[-1]))
        Conv_l1.append( SpatialDropout1D(list_of_dropout[1][2*i+1])(Conv_l1[-1]))

# Operation done on the small dimension : here fc
    Conv_l2 = [(Conv_l1[-1])]
    for i in range(Sizes[2]):
        Conv_l2.append(Conv1D(filters=list_of_filters[2][i], kernel_size=int(lev/2**Div),
                               padding='same', name=Name('Dense', i))(Conv_l2[-1]))
        Conv_l2.append(AG(list_of_activations[2][i], list_of_activations[2][i]+'_d_'+str(i), params)(Conv_l2[-1]))
        Conv_l2.append( SpatialDropout1D(list_of_dropout[2][i])(Conv_l2[-1]))
    Conv_l3 = [Conv_l2[-1]]

# Upsampling and concats
    for i in range(Sizes[3]//2):
        Conv_l3.append( Concatenate( name=Name('Concat',i+300) )([Conv_l3[-1], Conv_l1[-(1+i*7)]]))
        Conv_l3.append(UpSampling1D(2, name=Name('Ups',i+200))(Conv_l3[-1]))
        Conv_l3.append(Conv1D(filters=list_of_filters[3][2*i], kernel_size=list_of_kernels_s[3][2*i], padding='same',
                               name=Name('Conv', i+200))(Conv_l3[-1] ))
        Conv_l3.append( AG(list_of_activations[3][2*i], Name(list_of_activations[3][2*i], 200+i), params )(Conv_l3[-1]) )
        Conv_l3.append( SpatialDropout1D(list_of_dropout[3][2*i])(Conv_l3[-1]))

        Conv_l3.append(Conv1D(filters=list_of_filters[3][2*i+1], kernel_size=list_of_kernels_s[3][2*i+1], padding='same',
                               name=Name('Conv', i + 210))(Conv_l3[-1]))
        Conv_l3.append(AG(list_of_activations[3][2*i+1], Name(list_of_activations[3][2*i+1], 210 + i), params)(Conv_l3[-1]))
        Conv_l3.append( SpatialDropout1D(list_of_dropout[3][2*i+1])(Conv_l3[-1]))

# Last Conv layers
    Conv_l4 = [Conv_l3[-1]]
    for i in range(Sizes[4]):
        Conv_l4.append(Conv1D(filters=list_of_filters[4][i], kernel_size=list_of_kernels_s[4][i],
                            padding='same', use_bias=False, name=Name('Conv3',i+300),
                            activity_regularizer=regularizers.l2(reg))(Conv_l4[-1]))
        Conv_l4.append( AG(list_of_activations[4][i], Name(list_of_activations[4][i], 300+i), params )(Conv_l4[-1]) )
        Conv_l4.append( SpatialDropout1D(list_of_dropout[4][i])(Conv_l4[-1]))
    return keras.Model(Conv_l0[0], Conv_l4[-1])

##### Bias
def Unet_Act_Double_DP_Bias(list_of_kernels_s, list_of_filters, list_of_activations, list_of_dropout, params=[], Div=3, lev=CST.lev(CST), in_channel=11, reg=0.001 ):
    """
    Generate a Unet-Archictecture with bias output :
    outputs are size (lev+1)
    list_of_xxx : list of 6 list containing info for xxx
    list_xxx[0] : parameters for the starting convolutions
    list_xxx[1] : parameters for the downscaling convolutions
    list_xxx[2] : parameters for small space convolutions
    list_xxx[3] : parameters for upsampling convolutions
    list_xxx[4] : parameters for the last convolutions
    list_xxx[5] : parameters for the fully connected bias layers
    lists must be the same size (expect for list_of_kernel_s[3]/[5] which are not used)

    kernel : size of kernels
    filter : number of filters
    activation : names of the activation
    dropout : dropout rate

    params : params used for activation
    Div : number of downscaling
    in_channel : input column number
    lev : input ligne number
    reg : weights loss

    # The number of convolution at each up/downsampling is 2 so :
    list_of_xxx[1]/list_of_xxx[3] must be of len Div*2

    # The third element of the list correspond to a convolution with kernel_size=vector size so :
    list_of_kernel_s[3] is not used
    list_of_filters[3] is the number of filter (unlike previous Unet)
    """
    AG = Activation_Generator()
    Sizes = [ len(list_of_filters[i]) for i in range(len(list_of_filters)) ]
#First Convolutions
    Conv_l0 = [Input(name='Origin_Input', dtype='float32', shape=(lev, in_channel))]
    for i in range(Sizes[0]):
        Conv_l0.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i],
                            padding='same', use_bias=False, name=Name('Conv', i),
                            kernel_regularizer=regularizers.l1(reg))(Conv_l0[-1]))
        Conv_l0.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], i), params)(Conv_l0[-1]))
        Conv_l0.append( SpatialDropout1D(list_of_dropout[0][i])(Conv_l0[-1]))

# DownScaling
    Conv_l1 = [Conv_l0[-1]]
    for i in range(Sizes[1]//2):
        Conv_l1.append(AveragePooling1D(2, padding='same', stride=2, name=Name('AVG', i+100))(Conv_l1[-1]))
        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i], kernel_size=list_of_kernels_s[1][2*i],
                               padding='same', name=Name('Conv', i+100))(Conv_l1[-1] ))
        Conv_l1.append(AG(list_of_activations[1][2*i], Name(list_of_activations[1][2*i], 100+i), params )(Conv_l1[-1]) )
        Conv_l1.append( SpatialDropout1D(list_of_dropout[1][2*i])(Conv_l1[-1]))

        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i+1], kernel_size=list_of_kernels_s[1][2*i+1],
                               padding='same', name=Name('Conv', 110+i))(Conv_l1[-1]) )
        Conv_l1.append(AG(list_of_activations[1][2*i+1], Name(list_of_activations[1][2*i+1], 110+i), params )(Conv_l1[-1]))
        Conv_l1.append( SpatialDropout1D(list_of_dropout[1][2*i+1])(Conv_l1[-1]))

# Operation done on the small dimension : here fc
    Conv_l2 = [(Conv_l1[-1])]
    for i in range(Sizes[2]):
        Conv_l2.append(Conv1D(filters=list_of_filters[2][i], kernel_size=int(lev/2**Div),
                               padding='same', name=Name('Dense', i))(Conv_l2[-1]))
        Conv_l2.append(AG(list_of_activations[2][i], list_of_activations[2][i]+'_d_'+str(i), params)(Conv_l2[-1]))
        Conv_l2.append( SpatialDropout1D(list_of_dropout[2][i])(Conv_l2[-1]))
    Conv_l3 = [Conv_l2[-1]]

# Upsampling and concats
    for i in range(Sizes[3]//2):
        Conv_l3.append( Concatenate( name=Name('Concat',i+300) )([Conv_l3[-1], Conv_l1[-(1+i*7)]]))
        Conv_l3.append(UpSampling1D(2, name=Name('Ups',i+200))(Conv_l3[-1]))
        Conv_l3.append(Conv1D(filters=list_of_filters[3][2*i], kernel_size=list_of_kernels_s[3][2*i], padding='same',
                               name=Name('Conv', i+200))(Conv_l3[-1] ))
        Conv_l3.append( AG(list_of_activations[3][2*i], Name(list_of_activations[3][2*i], 200+i), params )(Conv_l3[-1]) )
        Conv_l3.append( SpatialDropout1D(list_of_dropout[3][2*i])(Conv_l3[-1]))

        Conv_l3.append(Conv1D(filters=list_of_filters[3][2*i+1], kernel_size=list_of_kernels_s[3][2*i+1], padding='same',
                               name=Name('Conv', i + 210))(Conv_l3[-1]))
        Conv_l3.append(AG(list_of_activations[3][2*i+1], Name(list_of_activations[3][2*i+1], 210 + i), params)(Conv_l3[-1]))
        Conv_l3.append( SpatialDropout1D(list_of_dropout[3][2*i+1])(Conv_l3[-1]))

# Last Conv layers
    Conv_l4 = [Conv_l3[-1]]
    for i in range(Sizes[4]):
        Conv_l4.append(Conv1D(filters=list_of_filters[4][i], kernel_size=list_of_kernels_s[4][i],
                            padding='same', use_bias=False, name=Name('Conv3',i+300),
                            kernel_regularizer=regularizers.l2(reg))(Conv_l4[-1]))
        Conv_l4.append( AG(list_of_activations[4][i], Name(list_of_activations[4][i], 300+i), params )(Conv_l4[-1]) )
        Conv_l4.append( SpatialDropout1D(list_of_dropout[4][i])(Conv_l4[-1]))

# Bias output
    D_l = [Conv_l2[-1]]
    D_l.append(Flatten(name=Name('Flat', 0))(D_l))
    for i in range(Sizes[5]):
        D_l.append(Dense(units=list_of_filters[5][i], kernel_regularizer=regularizers.l2(reg), name=Name('Dense', 200+i))(D_l[-1]))
        D_l.append(AG(list_of_activations[5][i], list_of_activations[5][i]+'_d2_'+str(i), params)(D_l[-1]))
        D_l.append( Dropout(list_of_dropout[5][i])(D_l[-1]))

    Sub_f = lambda x: x[:,:,0] - x[:,:,1]
    Sub = keras.layers.Lambda(Sub_f)

    Out1 = Sub(Conv_l4[-1])
    Combine = keras.layers.Concatenate(name='Concat')([Out1, D_l[-1]])

    return keras.Model(Conv_l0[0], Combine )

############## Unet_Act_Double_2D
def Unet_Act_Double_2D(list_of_kernels_s, list_of_filters, list_of_activations, list_of_dropout,
                       params=[], Div=3, lev=CST.lev(CST), in_channel=11, reg=0.0,
                       data_format='channels_first'):
    """
    Generate a Unet-Archictecture:
    list_of_kernels : list of 3 lists containing the kernel size for convolution
    list_of_filters : list of 3 lists containing the number of filters for convolution
    list_of_activations : list of 3 list containing the names of the activation function
    list_of_dropout : list of 3 list containing the proportion of the dropout
    params : params used for activation
    Div : number of downscaling
    in_channel : number of inputs
    reg : weights regularizers (should be 0 if you use dropout)
    """
    AG = Activation_Generator()
    Sizes = [ len(list_of_filters[i]) for i in range(len(list_of_filters)) ]
    if data_format=='channels_first':
        concat_axis=1
    else:
        concat_axis=-1
#First Convolutions
    Conv_l0 = [Input(name='Origin_Input', dtype='float32', shape=(in_channel, lev, lev))]
    for i in range(Sizes[0]):
        Conv_l0.append(Conv2D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i],
                            padding='same', use_bias=False, name=Name('Conv', i),
                            kernel_regularizer=regularizers.l1(reg),
                            data_format=data_format)(Conv_l0[-1]))
        Conv_l0.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], i), params)(Conv_l0[-1]))
        Conv_l0.append(Dropout(list_of_dropout[0][i])(Conv_l0[-1]))

# DownScaling
    Conv_l1 = [Conv_l0[-1]]
    for i in range(Sizes[1]//2):
        Conv_l1.append(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same', name=Name('AVG', i+100))(Conv_l1[-1]))
        Conv_l1.append(Conv2D(filters=list_of_filters[1][2*i], kernel_size=list_of_kernels_s[1][2*i],
                               padding='same', name=Name('Conv', i+100),data_format=data_format)(Conv_l1[-1] ))
        Conv_l1.append(AG(list_of_activations[1][2*i], Name(list_of_activations[1][2*i], 100+i), params )(Conv_l1[-1]) )
        Conv_l1.append(Dropout(list_of_dropout[1][2*i])(Conv_l1[-1]))

        Conv_l1.append(Conv2D(filters=list_of_filters[1][2*i+1], kernel_size=list_of_kernels_s[1][2*i+1],
                               padding='same', name=Name('Conv', 110+i),data_format=data_format)(Conv_l1[-1]) )
        Conv_l1.append(AG(list_of_activations[1][2*i+1], Name(list_of_activations[1][2*i+1], 110+i), params )(Conv_l1[-1]))
        Conv_l1.append(Dropout(list_of_dropout[1][2*i+1])(Conv_l1[-1]))

# Operation done on the small dimension : here fc
    Conv_l2 = [(Conv_l1[-1])]
    for i in range(Sizes[2]):
        Conv_l2.append(Conv2D(filters=list_of_filters[2][i], kernel_size=(int(lev/2**Div), int(lev/2**Div)),
                               padding='same', name=Name('Dense', i), data_format=data_format)(Conv_l2[-1]))
        Conv_l2.append(AG(list_of_activations[2][i], list_of_activations[2][i]+'_d_'+str(i), params)(Conv_l2[-1]))
        Conv_l2.append(Dropout(list_of_dropout[2][i])(Conv_l2[-1]))
    Conv_l3 = [Conv_l2[-1]]

# Upsampling and concats
    for i in range(Sizes[3]//2):
        Conv_l3.append( Concatenate( name=Name('Concat',i+300),axis=concat_axis )([Conv_l3[-1], Conv_l1[-(1+i*7)]]))
        Conv_l3.append(UpSampling2D(size=(2, 2), name=Name('Ups',i+200))(Conv_l3[-1]))
        Conv_l3.append(Conv2D(filters=list_of_filters[3][2*i], kernel_size=list_of_kernels_s[3][2*i], padding='same',
                               name=Name('Conv', i+200), data_format=data_format)(Conv_l3[-1] ))
        Conv_l3.append( AG(list_of_activations[3][2*i], Name(list_of_activations[3][2*i], 200+i), params )(Conv_l3[-1]) )
        Conv_l3.append(Dropout(list_of_dropout[3][2*i])(Conv_l3[-1]))

        Conv_l3.append(Conv2D(filters=list_of_filters[3][2*i+1], kernel_size=list_of_kernels_s[3][2*i+1], padding='same',
                               name=Name('Conv', i + 210), data_format=data_format)(Conv_l3[-1]))
        Conv_l3.append(AG(list_of_activations[3][2*i+1], Name(list_of_activations[3][2*i+1], 210 + i), params)(Conv_l3[-1]))
        Conv_l3.append(Dropout(list_of_dropout[3][2*i+1])(Conv_l3[-1]))

# Last Conv layers
    Conv_l4 = [Conv_l3[-1]]
    for i in range(Sizes[4]):
        Conv_l4.append(Conv2D(filters=list_of_filters[4][i], kernel_size=list_of_kernels_s[4][i],
                            padding='same', use_bias=False, name=Name('Conv3',i+300),
                            activity_regularizer=regularizers.l2(reg), data_format=data_format)(Conv_l4[-1]))

        Conv_l4.append( AG(list_of_activations[4][i], Name(list_of_activations[4][i], 300+i), params )(Conv_l4[-1]) )
        Conv_l4.append(Dropout(list_of_dropout[4][i])(Conv_l4[-1]))
    return keras.Model(Conv_l0[0], Conv_l4[-1])

def Unet_Act_Double_AE_2D(list_of_kernels_s, list_of_filters, list_of_activations, list_of_dropout,
                       params=[], Div=3, lev=CST.lev(CST), in_channel=11, reg=0.0,
                       data_format='channels_first'):
    """
    Generate a Unet-Archictecture:
    list_of_kernels : list of 3 lists containing the kernel size for convolution
    list_of_filters : list of 3 lists containing the number of filters for convolution
    list_of_activations : list of 3 list containing the names of the activation function
    list_of_dropout : list of 3 list containing the proportion of the dropout
    params : params used for activation
    Div : number of downscaling
    in_channel : number of inputs
    reg : weights regularizers (should be 0 if you use dropout)
    """
    AG = Activation_Generator()
    Sizes = [ len(list_of_filters[i]) for i in range(len(list_of_filters)) ]
    if data_format=='channels_first':
        concat_axis=1
    else:
        concat_axis=-1
#First Convolutions
    Conv_l0 = [Input(name='Origin_Input', dtype='float32', shape=(in_channel, lev, lev))]
    for i in range(Sizes[0]):
        Conv_l0.append(Conv2D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i],
                            padding='same', use_bias=False, name=Name('Conv', i),
                            kernel_regularizer=regularizers.l1(reg),
                            data_format=data_format)(Conv_l0[-1]))
        Conv_l0.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], i), params)(Conv_l0[-1]))
        Conv_l0.append(Dropout(list_of_dropout[0][i])(Conv_l0[-1]))
# DownScaling
    Conv_l1 = [Conv_l0[-1]]
    for i in range(Sizes[1]//2):
        Conv_l1.append(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same', name=Name('AVG', i+100))(Conv_l1[-1]))
        Conv_l1.append(Conv2D(filters=list_of_filters[1][2*i], kernel_size=list_of_kernels_s[1][2*i],
                               padding='same', name=Name('Conv', i+100),data_format=data_format)(Conv_l1[-1] ))
        Conv_l1.append(AG(list_of_activations[1][2*i], Name(list_of_activations[1][2*i], 100+i), params )(Conv_l1[-1]) )
        Conv_l1.append(Dropout(list_of_dropout[1][2*i])(Conv_l1[-1]))

        Conv_l1.append(Conv2D(filters=list_of_filters[1][2*i+1], kernel_size=list_of_kernels_s[1][2*i+1],
                               padding='same', name=Name('Conv', 110+i),data_format=data_format)(Conv_l1[-1]) )
        Conv_l1.append(AG(list_of_activations[1][2*i+1], Name(list_of_activations[1][2*i+1], 110+i), params )(Conv_l1[-1]))
        Conv_l1.append(Dropout(list_of_dropout[1][2*i+1])(Conv_l1[-1]))

# Operation done on the small dimension : here fc
    Conv_l2 = [(Conv_l1[-1])]
    for i in range(Sizes[2]):
        Conv_l2.append(Conv2D(filters=list_of_filters[2][i], kernel_size=(int(lev/2**Div), int(lev/2**Div)),
                               padding='same', name=Name('Dense', i), data_format=data_format)(Conv_l2[-1]))
        Conv_l2.append(AG(list_of_activations[2][i], list_of_activations[2][i]+'_d_'+str(i), params)(Conv_l2[-1]))
        Conv_l2.append(Dropout(list_of_dropout[2][i])(Conv_l2[-1]))

# Upsampling and concats
    Conv_l3 = [Conv_l2[-1]]
    for i in range(Sizes[3]//2):
        Conv_l3.append(UpSampling2D(size=(2, 2), name=Name('Ups',i+200))(Conv_l3[-1]))
        Conv_l3.append(Conv2D(filters=list_of_filters[3][2*i], kernel_size=list_of_kernels_s[3][2*i], padding='same',
                               name=Name('Conv', i+200), data_format=data_format)(Conv_l3[-1] ))
        Conv_l3.append( AG(list_of_activations[3][2*i], Name(list_of_activations[3][2*i], 200+i), params )(Conv_l3[-1]) )
        Conv_l3.append(Dropout(list_of_dropout[3][2*i])(Conv_l3[-1]))

        Conv_l3.append(Conv2D(filters=list_of_filters[3][2*i+1], kernel_size=list_of_kernels_s[3][2*i+1], padding='same',
                               name=Name('Conv', i + 210), data_format=data_format)(Conv_l3[-1]))
        Conv_l3.append(AG(list_of_activations[3][2*i+1], Name(list_of_activations[3][2*i+1], 210 + i), params)(Conv_l3[-1]))
        Conv_l3.append(Dropout(list_of_dropout[3][2*i+1])(Conv_l3[-1]))

# Last Conv layers
    Conv_l4 = [Conv_l3[-1]]
    for i in range(Sizes[4]):
        Conv_l4.append(Conv2D(filters=list_of_filters[4][i], kernel_size=list_of_kernels_s[4][i],
                            padding='same', use_bias=False, name=Name('Conv3',i+300),
                            activity_regularizer=regularizers.l2(reg), data_format=data_format)(Conv_l4[-1]))

        Conv_l4.append( AG(list_of_activations[4][i], Name(list_of_activations[4][i], 300+i), params )(Conv_l4[-1]) )
        Conv_l4.append(Dropout(list_of_dropout[4][i])(Conv_l4[-1]))
    return keras.Model(Conv_l0[0], Conv_l4[-1])

###############Other uNets
def Contraction(list_of_kernels_s, list_of_filters, list_of_activations,  list_of_dropout, params=[], Div=3, lev=CST.lev(CST), in_channel=11, reg=0.001):
    """
    Generate a Contracter Net (first half of the Unet)
    """
    AG = Activation_Generator()
    Sizes = [ len(list_of_filters[i]) for i in range(len(list_of_filters)) ]

#First Convolutions
    Conv_l0 = [Input(name='Origin_Input', dtype='float32', shape=(lev, in_channel))]
    for i in range(Sizes[0]):
        Conv_l0.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i],
                            padding='same', use_bias=False, name=Name('Conv', i),
                            kernel_regularizer=regularizers.l1(reg))(Conv_l0[-1]))
        Conv_l0.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], i), params)(Conv_l0[-1]))
# DownScaling
    Conv_l1 = [Conv_l0[-1]]
    for i in range(Sizes[1]//2):
        Conv_l1.append(AveragePooling1D(2, padding='same', stride=2, name=Name('AVG', i+100))(Conv_l1[-1]))
        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i], kernel_size=list_of_kernels_s[1][2*i],
                               padding='same', name=Name('Conv', i+100))(Conv_l1[-1] ))
        Conv_l1.append(AG(list_of_activations[1][2*i], Name(list_of_activations[1][2*i], 100+i), params )(Conv_l1[-1]) )

        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i+1], kernel_size=list_of_kernels_s[1][2*i+1],
                               padding='same', name=Name('Conv', 110+i))(Conv_l1[-1]) )
        Conv_l1.append(AG(list_of_activations[1][2*i+1], Name(list_of_activations[1][2*i+1], 110+i), params )(Conv_l1[-1]))

# Operation done on the small dimension : here fc
    Conv_l2 = [Flatten(name='Flatten')(Conv_l1[-1])]
    for i in range(Sizes[2]):
        Conv_l2.append( Dense( int(lev/2**Div) * list_of_filters[2][i], name=Name('Dense', i), activity_regularizer=regularizers.l2(reg))(Conv_l2[-1]))
        Conv_l2.append(AG(list_of_activations[2][i], list_of_activations[2][i]+'_d_'+str(i), params)(Conv_l2[-1]))

#    Conv_l2.append(Reshape(name='Reshape', input_shape=Conv_l2[-1].shape,
#                            target_shape=(int(lev/2**Div),  list_of_filters[2][-1]))(Conv_l2[-1]))
    return keras.Model(Conv_l0[0], Conv_l2[-1])



def AE(list_of_kernels_s, list_of_filters, list_of_activations, list_of_dropout, params=[], Div=3, lev=CST.lev(CST), in_channel=11, reg=0.001 ):
    """
    Generate a Unet-Archictecture
    list_of_kernels : list of 3 lists containing the kernel size for convolution
    list_of_filters : list of 3 lists containing the number of filters for convolution
    list_of_activations : list of 3 list containing the names of the activation function
    params : params used for activation
    Div : number of downscaling
    in_channel : number of inputs
    """
    AG = Activation_Generator()
    Sizes = [ len(list_of_filters[i]) for i in range(len(list_of_filters)) ]

#First Convolutions
    Conv_l0 = [Input(name='Origin_Input', dtype='float32', shape=(lev, in_channel))]
    for i in range(Sizes[0]):
        Conv_l0.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i],
                            padding='same', use_bias=False, name=Name('Conv', i),
                            activity_regularizer=regularizers.l2(reg))(Conv_l0[-1]))
        Conv_l0.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], i), params)(Conv_l0[-1]))
        Conv_l0.append( SpatialDropout1D(list_of_dropout[0][i])(Conv_l0[-1]))
# DownScaling
    Conv_l1 = [Conv_l0[-1]]
    for i in range(Sizes[1]//2):
        Conv_l1.append(MaxPooling1D(2, padding='same', stride=2, name=Name('AVG', i+100))(Conv_l1[-1]))
        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i], kernel_size=list_of_kernels_s[1][2*i],
                               padding='same', name=Name('Conv', i+100))(Conv_l1[-1] ))
        Conv_l1.append(AG(list_of_activations[1][2*i], Name(list_of_activations[1][2*i], 100+i), params )(Conv_l1[-1]) )
        Conv_l1.append( SpatialDropout1D(list_of_dropout[1][2*i])(Conv_l1[-1]))

        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i+1], kernel_size=list_of_kernels_s[1][2*i+1],
                               padding='same', name=Name('Conv', 110+i))(Conv_l1[-1]) )
        Conv_l1.append(AG(list_of_activations[1][2*i+1], Name(list_of_activations[1][2*i+1], 110+i), params )(Conv_l1[-1]))
        Conv_l1.append( SpatialDropout1D(list_of_dropout[1][2*i+1])(Conv_l1[-1]))

# Operation done on the small dimension : here fc
#    Conv_l2 = [Flatten(name='Flatten')(Conv_l1[-1])]
    Conv_l2 = [Conv_l1[-1] ]
    for i in range(Sizes[2]):
        Conv_l2.append( Conv1D(list_of_filters[2][i], kernel_size=lev//(2**Div), padding='same', name=Name('Dense', i), kernel_regularizer=regularizers.l2(reg))(Conv_l2[-1]))
        Conv_l2.append(AG(list_of_activations[2][i], list_of_activations[2][i]+'_d_'+str(i), params)(Conv_l2[-1]))
        Conv_l2.append( SpatialDropout1D(list_of_dropout[2][i])(Conv_l2[-1]))

#    Conv_l2.append(Reshape(name='Reshape', input_shape=Conv_l2[-1].shape,
#                            target_shape=(int(lev/2**Div),  list_of_filters[2][-1]))(Conv_l2[-1]))
    Conv_l3 = [Conv_l2[-1]]
# Upsampling and concats
    for i in range(Sizes[3]//2):
        Conv_l3.append(UpSampling1D(2, name=Name('Ups',i+200))(Conv_l3[-1]))
        Conv_l3.append(Conv1D(filters=list_of_filters[3][2*i], kernel_size=list_of_kernels_s[3][2*i], padding='same',
                               name=Name('Conv', i+200))(Conv_l3[-1] ))
        Conv_l3.append( AG(list_of_activations[3][2*i], Name(list_of_activations[3][2*i], 200+i), params )(Conv_l3[-1]) )
        Conv_l3.append( SpatialDropout1D(list_of_dropout[3][2*i])(Conv_l3[-1]))

        Conv_l3.append(Conv1D(filters=list_of_filters[3][2*i+1], kernel_size=list_of_kernels_s[3][2*i+1], padding='same',
                               name=Name('Conv', i + 210))(Conv_l3[-1]))
        Conv_l3.append(AG(list_of_activations[3][2*i+1], Name(list_of_activations[3][2*i+1], 210 + i), params)(Conv_l3[-1]))
        Conv_l3.append( SpatialDropout1D(list_of_dropout[3][2*i+1])(Conv_l3[-1]))

# Last Conv layers
    Conv_l4 = [Conv_l3[-1]]
    for i in range(Sizes[4]):
        Conv_l4.append(Conv1D(filters=list_of_filters[4][i], kernel_size=list_of_kernels_s[4][i],
                            padding='same', use_bias=False, name=Name('Conv3',i+300),
                            kernel_regularizer=regularizers.l2(reg))(Conv_l4[-1]))
        Conv_l4.append( AG(list_of_activations[4][i], Name(list_of_activations[4][i], 300+i), params )(Conv_l4[-1]) )
        Conv_l4.append( SpatialDropout1D(list_of_dropout[4][i])(Conv_l4[-1]))

    return keras.Model(Conv_l0[0], Conv_l4[-1])

################## CUSTOMS

def Unet_Double_XXXXX(list_of_kernels_s, list_of_filters, list_of_activations, params=[], Div=3, lev=CST.lev(CST), in_channel=11, reg=0.001 ):
    """
    Generate a Unet-Archictecture
    list_of_kernels : list of 3 lists containing the kernel size for convolution
    list_of_filters : list of 3 lists containing the number of filters for convolution
    list_of_activations : list of 3 list containing the names of the activation function
    params : params used for activation
    Div : number of downscaling
    in_channel : number of inputs
    """
    AG = Activation_Generator()
    Sizes = [ len(list_of_filters[i]) for i in range(len(list_of_filters)) ]

#First Convolutions
    Conv_l0 = [Input(name='Origin_Input', dtype='float32', shape=(lev, in_channel))]
    for i in range(Sizes[0]):
        Conv_l0.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i],
                            padding='same', use_bias=False, name=Name('Conv', i),
                            activity_regularizer=regularizers.l2(reg))(Conv_l0[-1]))
        Conv_l0.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], i), params)(Conv_l0[-1]))

# DownScaling
    Conv_l1 = [Conv_l0[-1]]
    for i in range(Sizes[1]//2):
        Conv_l1.append(MaxPooling1D(2, padding='same', stride=2, name=Name('AVG', i+100))(Conv_l1[-1]))
        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i], kernel_size=list_of_kernels_s[1][2*i],
                               padding='same', name=Name('Conv', i+100))(Conv_l1[-1] ))
        Conv_l1.append(AG(list_of_activations[1][2*i], Name(list_of_activations[1][2*i], 100+i), params )(Conv_l1[-1]) )

        Conv_l1.append(Conv1D(filters=list_of_filters[1][2*i+1], kernel_size=list_of_kernels_s[1][2*i+1],
                               padding='same', name=Name('Conv', 110+i))(Conv_l1[-1]) )
        Conv_l1.append(AG(list_of_activations[1][2*i+1], Name(list_of_activations[1][2*i+1], 110+i), params )(Conv_l1[-1]))

# Operation done on the small dimension : here fc
    Conv_l2 = [Flatten(name='Flatten')(Conv_l1[-1])]
    for i in range(Sizes[2]):
        Conv_l2.append( Dense( int(lev/2**Div) * list_of_filters[2][i], name=Name('Dense', i), kernel_regularizer=regularizers.l2(reg))(Conv_l2[-1]))
        Conv_l2.append(AG(list_of_activations[2][i], list_of_activations[2][i]+'_d_'+str(i), params)(Conv_l2[-1]))

    Conv_l2.append(Reshape(name='Reshape', input_shape=Conv_l2[-1].shape,
                            target_shape=(int(lev/2**Div),  list_of_filters[2][-1]))(Conv_l2[-1]))
    Conv_l3 = [Conv_l2[-1]]
# Upsampling and concats
    for i in range(Sizes[3]//2):
        Conv_l3.append( Concatenate( name=Name('Concat',i+300) )([Conv_l3[-1], Conv_l1[-(1+i*5)]]))
        Conv_l3.append(UpSampling1D(2, name=Name('Ups',i+200))(Conv_l3[-1]))
        Conv_l3.append(Conv1D(filters=list_of_filters[3][2*i], kernel_size=list_of_kernels_s[3][2*i], padding='same',
                               name=Name('Conv', i+200))(Conv_l3[-1] ))
        Conv_l3.append( AG(list_of_activations[3][2*i], Name(list_of_activations[3][2*i], 200+i), params )(Conv_l3[-1]) )

        Conv_l3.append(Conv1D(filters=list_of_filters[3][2*i+1], kernel_size=list_of_kernels_s[3][2*i+1], padding='same',
                               name=Name('Conv', i + 210))(Conv_l3[-1]))
        Conv_l3.append(AG(list_of_activations[3][2*i+1], Name(list_of_activations[3][2*i+1], 210 + i), params)(Conv_l3[-1]))

# Last Conv layers
    Conv_l4 = [Conv_l3[-1]]
    for i in range(Sizes[4]):
        Conv_l4.append(Conv1D(filters=list_of_filters[4][i], kernel_size=list_of_kernels_s[4][i],
                            padding='same', use_bias=False, name=Name('Conv3',i+300),
                            kernel_regularizer=regularizers.l2(reg))(Conv_l4[-1]))
        Conv_l4.append( AG(list_of_activations[4][i], Name(list_of_activations[4][i], 300+i), params )(Conv_l4[-1]) )
    return keras.Model(Conv_l0[0], Conv_l4[-1])


############# OLD AND UNUSED architectures ###################

def Bidir_Casual_Conv_L1(list_of_kernel_s, list_of_filters, list_of_activations, params, in_channel, lev=CST.lev(CST), reg=0.0001):
    """
    Same as Bidir_Casual_Conv but with L1 norm
    """
    Input0 = Input(shape=(lev, in_channel), name=Name('Input',0), dtype='float32')

    Flip_layer = lambda x: K.reverse(x, axes=0)
#    I_cp = UpSampling1D(ups, name=Name('Up',0))(Input0)
#    I_avg = AveragePooling1D(pooling, padding='same', stride=ups, name='AVG_p')(I_cp)
    I_flip = Lambda(Flip_layer, name=Name('Flip',0))(Input0)

    Conv1u = [Input0]
    Conv1d = [I_flip]
    AG = Activation_Generator()

    # Normal causal
    for i in range(len(list_of_filters[0])):
        Conv1u.append(Conv1D(filters = list_of_filters[0][i], kernel_size= list_of_kernel_s[0][i], \
                        padding='causal', name=Name("Conv_u",i+1), use_bias=True, activity_regularizer=regularizers.l1(reg))(Conv1u[-1]))
        Conv1u.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i]+'_u', i+1), params )(Conv1u[-1]) )

    # Flipped causal
    for i in range(len(list_of_filters[1])):
        Conv1d.append(Conv1D(filters = list_of_filters[1][i], kernel_size= list_of_kernel_s[1][i],\
                        padding='causal', name=Name("Conv_d",i+1), use_bias=True, activity_regularizer=regularizers.l1(reg))(Conv1d[-1]))
        Conv1d.append( AG(list_of_activations[1][i], Name(list_of_activations[1][i]+'_d', i+1), params )(Conv1d[-1]) )

    C_flip = Lambda(Flip_layer,name=Name('Flip',1))(Conv1d[-1])
    C1d_prime = [Concatenate( name=Name('Concat',0))([Conv1u[-1], C_flip])]

    # Conv of both
    for i in range(len(list_of_filters[2])):
        if(i!=len(list_of_filters[2])):
            C1d_prime.append(Conv1D(filters = list_of_filters[2][i], kernel_size=list_of_kernel_s[2][i], \
                                padding='same', name=Name("Conv_c",i), use_bias=True, activity_regularizer=regularizers.l1(reg))(C1d_prime[-1]))
            C1d_prime.append( AG(list_of_activations[2][i], Name(list_of_activations[2][i]+'_c', 1+i), params )(C1d_prime[-1]) )
        else:
            C1d_prime.append(Conv1D(filters = list_of_filters[2][i], kernel_size=list_of_kernel_s[2][i], \
                                padding='same', name=Name("Conv_c",i), use_bias=True)(C1d_prime[-1]))
            C1d_prime.append( AG(list_of_activations[2][i], Name(list_of_activations[2][i]+'_c', 1+i), params )(C1d_prime[-1]) )
    return keras.Model(Input0, C1d_prime[-1])


def Unet_Act_Simple(list_of_kernels_s, list_of_filters, list_of_activations, params=[], Div=3, lev=CST.lev(CST), in_channel=11, reg=0.001 ):
    """
    Generate a Unet-Archictecture
    list_of_kernels : list of 3 lists containing the kernel size for convolution
    list_of_filters : list of 3 lists containing the number of filters for convolution
    list_of_activations : list of 3 list containing the names of the activation function
    params : params used for activation
    Div : number of downscaling
    in_channel : number of inputs
    """
    AG = Activation_Generator()
    Concats_l = []
    Upsamplings_l = []
    Conv_l1 = []
    Conv_l2 = []
    Poolings_l = []

# DownScaling
    Conv_l1=[]
    ACT_l1 = []
    ACT_l1.append(Input(name = 'Origin_Input',  dtype='float32', shape=(lev, in_channel)))
    for i in range(Div):
        Poolings_l.append(AveragePooling1D(2, padding='same', \
                                           stride=2, name=Name('AVG', i+1))(ACT_l1[-1]))
        Conv_l1.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i],
                                padding='same', name=Name('Conv1',i+1), activity_regularizer=regularizers.l2(reg))( Poolings_l[-1] ))
        ACT_l1.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], 10+i), params  )(Conv_l1[-1]) )

# Operation done on the small dimension : here fc
    Conv_l2.append(Flatten()(ACT_l1[-1])  )
    Conv_l2.append(Dense( int(lev/2**Div) * list_of_filters[1][0], activity_regularizer=regularizers.l2(reg))(Conv_l2[-1])  )
    Conv_l2.append(  AG(list_of_activations[1][0], list_of_activations[1][0]+'_c', params)(Conv_l2[-1])  )
    Conv_l2.append(Reshape(name='Reshape',input_shape=Conv_l2[-1].shape ,\
                            target_shape=( int(lev/2**Div)  ,  list_of_filters[1][0] ))(Conv_l2[-1]))

# Upsampling and concats
    for i in range(Div):
        Concats_l.append(Concatenate( name=Name('Concat',i+1) )([Conv_l2[-1], ACT_l1[-i-1]]))
        Upsamplings_l.append(UpSampling1D(2, name=Name('Ups',i+1))(Concats_l[-1]))
        Conv_l2.append(Conv1D(filters=list_of_filters[2][i], kernel_size=list_of_kernels_s[2][i], use_bias=False,\
                               padding='same', name=Name('Conv2',i+1), kernel_regularizer=regularizers.l2(reg))( Upsamplings_l[-1] ))
        Conv_l2.append( AG(list_of_activations[2][i], Name(list_of_activations[2][i], 20+i), params )(Conv_l2[-1]) )
    Conv3 = [Conv_l2[-1]]
    for i in range(len(list_of_kernels_s[3])):
        Conv3.append(Conv1D(filters=list_of_filters[3][i], kernel_size=list_of_kernels_s[3][i],
                            padding='same', use_bias=False, name=Name('Conv3',i+30), activity_regularizer=regularizers.l2(reg))(Conv3[-1]))
        Conv3.append( AG(list_of_activations[3][i], Name(list_of_activations[3][i], 30+i), params )(Conv3[-1]) )
    return keras.Model(ACT_l1[0],Conv3[-1])
