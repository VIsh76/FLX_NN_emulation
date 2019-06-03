import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LeakyReLU, Activation, ELU
from keras.losses import mean_squared_error
from keras import backend as K
from keras import regularizers

from contextlib import redirect_stdout
from CST import CST
# Simples architecture are saved if the need to reused them is presented

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

def lower_loss(y_true, y_pred, lev = 40):
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

##########""# ACTIVATION GENERATOR
class Activation_Generator():
    def __init__(self):
        pass

    @property
    def keys(self):
        return(['sigmoid', 'elu', 'softplus', 'tanh', 'relu', 'leakyrelu','linear' ])

    def __call__(self, act,name, *arg):
        if act== 'sigmoid':
            la = Activation('sigmoid')
        elif act== 'softplus':
            la = Activation('softplus')
        elif act== 'relu':
            la = Activation('relu')
        elif act== 'sigmoid':
            la = Activation('sigmoid')
        elif act== 'selu':
            la = Activation('selu')
        elif act== 'tanh':
            la = Activation('tanh')
        elif act== 'linear':
            la = Activation('linear')
        elif act=='leakyrelu':
            la = LeakyReLU(arg)
        elif act=='elu':
            la = ELU(arg)
        else:
            print(act, "is not implemented")
            assert(False)
        la.name = name
        return la


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
        self.current_batch=0

    @property
    def loss_name(self):
        return(['flxu_loss', 'flxd_loss', 'dfdts_loss', 'loss'])

    def __getitem__(self,i):
        return(self.losses.__getitem__(i))

    def on_train_begin(self, logs={}):
        self.losses = dict()
        for n in self.loss_name:
            self.losses[n] = [0]

    def on_batch_end(self, batch, logs={}):
        #print(logs['batch'])
        if(batch%self.frequency!=0):
            for n in self.loss_name:
                self.losses[n][-1] += logs.get(n)
        else:
            for n in self.loss_name:
                self.losses[n][-1] /= self.frequency
                self.losses[n].append( logs.get(n))

#    def on_train_end(self, logs={}):
#        for n in self.loss_name:
#            self.losses[n] = np.array(self.losses[n])

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
