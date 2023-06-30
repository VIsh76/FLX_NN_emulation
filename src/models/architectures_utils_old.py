import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LeakyReLU, Activation, ELU
from keras.losses import mean_squared_error
from keras import backend as K
from keras import regularizers
from contextlib import redirect_stdout


# Various Utils fonction for architectures



# Reshaping
def reshape(y, n_shape):
    y0=reshape(y.shape[0], n_shape[0], n_shape[1])
    return(y0)

def y_batch_reshape(y):
    return(reshape(y, CST.lev(CST)), CST.outputs_y(CST))

# LOSSES
def one_loss(y_true, y_pred, i):
    """
    MSE on particular var
    """
    E = mean_squared_error(y_true[:, :, i], y_pred[:, :, i])
    return E

# OLD LOSS : flxu,flxd, dfdts
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

# Losse for full flx
def Up_Down_loss_sep(y_true, y_pred):
    """
    Compute the loss of flxd and flxu
    """
    E = flxd_loss(y_true, y_pred)
    E += flxu_loss(y_true, y_pred)
    return E

def lower_loss(y_true, y_pred, lev = 40):
    """
    Compute MSE in the lower levels only
    """
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

### Cross Enthorpy
def LogLoss(y_true, y_pred):
    y0 = tf.convert_to_tensor(y[:,:,-1], dtype=tf.int32)
    y0 = tf.one_hot(y0, depth=2,axis=-1)
    L = keras.losses.categorical_crossentropy(y0,
                                              tf.convert_to_tensor(y_pred, dtype=tf.float32))
    L = K.sum(L, axis=1)
    return(L)

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

########### ACTIVATION GENERATOR

# SWISH
from keras.utils.generic_utils import get_custom_objects
def swish_activation(x):
    return K.sigmoid(x)*x
get_custom_objects().update({'swish': Activation(swish_activation)})


# ACTIVATION
class Activation_Generator():
    """
    This class generates activation layers given one of the keys, allows to change the
    activation easiely in architectures
    """
    def __init__(self):
        """
        generate the class for calls
        """
        pass

    @property
    def keys(self):
        return(['sigmoid', 'elu', 'softplus', 'tanh', 'relu', 'leakyrelu','linear' ])

    def __call__(self, act,name, *arg):
        """
        Create an activation layer
        act : type of act
        name : name of layer
        *arg : additional argument for activation
        """
        if act== 'sigmoid':
            la = Activation('sigmoid')
        elif act== 'softplus':
            la = Activation('softplus')
        elif act== 'softmax':
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
        elif act== 'softmax':
            la = Activation('softmax')
        elif act=='leakyrelu':
            la = LeakyReLU(arg)
        elif act=='elu':
            la = ELU(arg)
        elif(act=='swish'):
            la=Activation('swish')
        else:
            print(act, "is not implemented")
            assert(False)
        la.name = name
        return la



### CALLBACKS
class SGDLearningRateTracker(keras.callbacks.Callback):
    """
    Return learning rate
    """
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, 'float32_ref'))))
        print('\nLR: {:.6f}\n'.format(lr))

class LossHistory(keras.callbacks.Callback):
    """
    Callback that keep an history of the training losses with more frequency
    """
    def __init__(self, frequency=1000, losses=['flxu_loss', 'flxd_loss', 'dfdts_loss', 'loss']):
        """
        frequency : size of the mean, should be set as generator.batch_per_epoch
        losses : list of the loss to keep track of, has to set as metrics when compile
        """
        super(LossHistory, self).__init__()
        self.frequency=frequency
        self.current_batch=0
        self._loss_name = losses

    @property
    def loss_name(self):
        return(self._loss_name)

    def __getitem__(self,i):
        return(self.losses.__getitem__(i))

    def on_train_begin(self, logs={}):
        self.losses = dict()
        for n in self.loss_name:
            self.losses[n] = [0]

    def on_batch_end(self, batch, logs={}):
        if(batch%self.frequency!=0):
            for n in self.loss_name:
                self.losses[n][-1] += logs.get(n)
        else:
            for n in self.loss_name:
                self.losses[n][-1] /= self.frequency
                self.losses[n].append( logs.get(n))

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

def Gradient_T(input, batch_size, model, header, lev=CST.lev(CST)):
    """
    Compute the impact of delta T (constant T add up) on the output
    """
    x_test = input
    id = header.index('t')
    k_constants = K.constant(x_test)

    input11 = keras.layers.Input(batch_shape=(batch_size,1))
    input12 = keras.layers.RepeatVector(n=lev)(input11)

    def Sum_T(x, id):
        np.repeat(x)

    out11 = keras.layers.Lambda(lambda x: x + k_constants, axis=-1)(input12)


    out12 = model_bd_avg_2(out11)
    out13 = keras.layers.Lambda(lambda y: y[:,:,1] - y[:,:,0])(out12)

    model1 = keras.models.Model(inputs=input11, outputs=out13)
    model1.summary()
    t=time.time()
    gradients = [K.gradients(model1.get_output_at(0)[:,i], model1.input) for i in range(72)]
    grad0 = K.function( [model1.input] , [gradients[i][0] for i in range(72)] )
    ts = x_test[:,0,-1].reshape(x_test.shape[0], 1)
    g = np.array(grad0([ts]))
    return(0)


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
