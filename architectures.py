import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Flatten, Input, TimeDistributed
from keras.layers import Conv1D, UpSampling1D, AveragePooling1D, SeparableConv1D
from keras.layers import Bidirectional
from keras.losses import mean_squared_error
from keras import backend as K


def one_loss(y_true, y_pred, i):
    E = mean_squared_error(y_true[:, :, i], y_pred[:, :, i])
    return E


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
