from keras.layers import LeakyReLU, Activation, ELU

#Naming
def Name(layer,i):
    """
    Set Name by combining layer name (string) and an id (int)
    """
    return layer+'_'+str(i)

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
        #la.name = name
        return la


