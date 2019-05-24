from CST import CST
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
from architectures_utils import Name, Activation_Generator


##################### BD casual Conv1
def Bidir_Casual_Conv(list_of_kernel_s, list_of_filters, list_of_activations, params, ups, pooling, in_channel, lev=CST.lev(CST)):
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
    AG = Activation_Generator()
    # Normal
    for i in range(len(list_of_filters[0])):
        Conv1u.append(Conv1D(filters = list_of_filters[0][i], kernel_size= list_of_kernel_s[0][i], \
                        padding='causal', name=Name("Conv_u",i+1), use_bias=True)(Conv1u[-1]))
        Convlu.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], 100+i), params )(Conv1u[-1]) )

    # Flipped
    for i in range(len(list_of_filters[1])):
        Conv1d.append(Conv1D(filters = list_of_filters[1][i], kernel_size= list_of_kernel_s[1][i],\
                        padding='causal', name=Name("Conv_d",i+1), use_bias=True)(Conv1d[-1]))
        Convld.append( AG(list_of_activations[1][i], Name(list_of_activations[0][i], 200+i), params )(Conv1d[-1]) )

    C_flip = Lambda(Flip_layer,name=Name('Flip',1))(Conv1d[-1])
    C1d_prime = [Concatenate( name=Name('Concat',0))([Conv1u[-1], C_flip])]
    for i in range(len(list_of_filters[2])):
        if i==len(list_of_kernel_s)-1:
            C1d_prime.append(Conv1D(filters = list_of_filters[2][i], kernel_size=list_of_kernel_s[2][i], \
                            padding='causal', name=Name("Conv_conc",i), use_bias=True)(C1d_prime[-1]))
        else:
            C1d_prime.append(Conv1D(filters = list_of_filters[2][i], kernel_size=list_of_kernel_s[2][i], \
                            padding='causal', name=Name("Conv_conc",i), use_bias=True)(C1d_prime[-1]))
            C1d_prime.append( AG(list_of_activations[1][i], Name(list_of_activations[2][i], 200+i), params )(C1d_prime.[-1]) )
    return keras.Model(Input0, C1d_prime[-1])


##################### Unet Simple
def Unet_Act_Simple(list_of_kernels_s, list_of_filters, list_of_activations=[], params=[], Div=3, lev=CST.lev(CST), in_channel=11, o_channel=CST.output_y(CST) ):
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
    Convs_l1 = []
    Convs_l2 = []
    Poolings_l = []

# DownScaling
    ACT_l1 = []
    ACT_l1.append(Input(name = 'Origin_Input',  dtype='float32', shape=(lev, in_channel)))
    for i in range(Div):
        Poolings_l.append(AveragePooling1D(list_of_kernels_s[0][i]-1, padding='same', \
                                           stride=2, name=Name('AVG', i+1))(ACT_l1[-1]))
        Convs_l1.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i], \
                                padding='same', name=Name('Conv1',i+1))( Poolings_l[-1] ))
        ACT_l1.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], 10+i), params  )(Convs_l1[-1]) )

# Operation done on the small dimension : here fc
    Convs_l2.append(Flatten()(ACT_l1[-1])  )
    Convs_l2.append(Dense( int(lev/2**Div) * list_of_filters[1][0]  )(Convs_l2[-1])  )
    Convs_l2.append(  AG(list_of_activations[1][0], list_of_activations[1][0]+'_c', params)(Convs_l2[-1])  )
    Convs_l2.append(Reshape(name='Reshape',input_shape=Convs_l2[-1].shape ,\
                            target_shape=( int(lev/2**Div)  ,  list_of_filters[1][0] ))(Convs_l2[-1]))

# Upsampling and concats
    for i in range(Div):
        Concats_l.append(Concatenate( name=Name('Concat',i+1) )([Convs_l2[-1], ACT_l1[-i-1]]))
        Upsamplings_l.append(UpSampling1D(2, name=Name('Ups',i+1))(Concats_l[-1]))
        Convs_l2.append(Conv1D(filters=list_of_filters[2][i], kernel_size=list_of_kernels_s[2][i], padding='same', name=Name('Conv2',i+1))( Upsamplings_l[-1] ))
        Convs_l2.append( AG(list_of_activations[2][i], Name(list_of_activations[2][i], 20+i), params )(Convs_l2[-1]) )
    Conv3 = Conv1D(filters=o_channel, kernel_size=1, padding='same', use_bias=True)(Convs_l2[-1])
    return keras.Model(Convs_l1[0],Conv3)


##################### Two Layer of Conv at each upsampling and downsampling
def Unet_Act_Double(list_of_kernels_s, list_of_filters, list_of_activations=[], params=[], Div=3, lev=CST.lev(CST), in_channel=11, o_channel=CST.output_y(CST) ):
    """
    Generate a Unet-Archictecture
    list_of_kernels : list of 3 lists containing the kernel size for convolution
    list_of_filters : list of 3 lists containing the number of filters for convolution
    list_of_activations : list of 4 list containing the names of the activation function
    params : params used for activation
    Div : number of downscaling
    in_channel : number of inputs
    """
    AG = Activation_Generator()
    Concats_l = []
    Upsamplings_l = []
    Convs_l1 = []
    Convs_l2 = []
    Poolings_l = []

# DownScaling
    ACT_l1 = []
    ACT_l1.append(Input(name = 'Origin_Input',  dtype='float32', shape=(lev, in_channel)))
    for i in range(Div):
        Poolings_l.append(AveragePooling1D(list_of_kernels_s[0][i]-1, padding='same', stride=2, name=Name('AVG', i+1))(ACT_l1[-1]))
        Convs_l1.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i], padding='same', name=Name('Conv1',i+1))( Poolings_l[-1] ))
        Convs_l1.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], 10+i), params )(Convs_l1[-1]) )
        Convs_l1.append(Conv1D(filters=list_of_filters[0][i], kernel_size=list_of_kernels_s[0][i], padding='same', name=Name('Conv1',i+1))(Convs_l1[-1]) )
        ACT_l1.append( AG(list_of_activations[0][i], Name(list_of_activations[0][i], 10+i), params )(Convs_l1[-1]) )

# Operation done on the small dimension : here fc
    Convs_l2.append(Flatten()(ACT_l1[-1])  )
    Convs_l2.append(Dense( int(lev/2**Div) * list_of_filters[1][0]  )(Convs_l2[-1])  )
    Convs_l2.append(  AG(list_of_activations[1][0], list_of_activations[1][0]+'_c', params)(Convs_l2[-1])  )
    Convs_l2.append(Reshape(name='Reshape',input_shape=Convs_l2[-1].shape ,\
                            target_shape=( int(lev/2**Div)  ,  list_of_filters[1][0] ))(Convs_l2[-1]))
# Upsampling and concats
    for i in range(Div):
        Concats_l.append(Concatenate( name=Name('Concat',i+1) )([Convs_l2[-1], ACT_l1[-i-1]]))
        Upsamplings_l.append(UpSampling1D(2, name=Name('Ups',i+1))(Concats_l[-1]))
        Convs_l2.append(Conv1D(filters=list_of_filters[2][i], kernel_size=list_of_kernels_s[2][i], padding='same', name=Name('Conv2',i+1))( Upsamplings_l[-1] ))
        Convs_l2.append( AG(list_of_activations[2][i], Name(list_of_activations[2][i], 20+i), params )(Convs_l2[-1]) )
    Conv3.append(Conv1D(filters=o_channel, kernel_size=1, padding='same', use_bias=True)(Convs_l2[-1]))
    Conv3.append(AG(list_of_activations[1][0], list_of_activations[1][0]+'_c', params)(Convs_l2[-1])(Conv3[-1]))
    Conv3.append(Conv1D(filters=o_channel, kernel_size=1, padding='same', use_bias=True)(Conv3[-1]))
    return keras.Model(Convs_l1[0],Conv3[-1])
