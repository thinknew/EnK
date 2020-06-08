
'''
EnK Layer
© Avinash K Singh 
https://github.com/thinknew/enk
Licensed under MIT License
'''

####################################################################
### Original code is from: https://github.com/vlawhern/arl-eegmodels
####################################################################




import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, \
    SeparableConv2D, DepthwiseConv2D, BatchNormalization, SpatialDropout2D, Input, Flatten, GaussianNoise, ConvLSTM2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
from EnKLayer import EnKLayer

tf.disable_v2_behavior()


# sess = tf.Session()
def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    # with tf.Graph().as_default() as g:
    input1 = Input(shape=(1, Chans, Samples))

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(1, Chans, Samples),
                    use_bias=False, data_format='channels_first')(input1)

    block1 = EnKLayer(filter=F1, EnK=EnK, InputData_shape=(1, Chans, Samples),
                      output_dimension=(1, 1, kernLength), conv2DOutput=block1)(input1)

    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_first',
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)  # changed by avinash
    block1 = Activation('elu')(block1)

    block1 = AveragePooling2D((1, 4), data_format='channels_first', )(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),  # changed by avinash
                             data_format='channels_first', use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8), data_format='channels_first', )(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten', data_format='channels_first')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def Gau_EEGNet(nb_classes, Chans=64, Samples=128,
               dropoutRate=0.5, kernLength=64, F1=8,
               D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    # with tf.Graph().as_default() as g:
    input1 = Input(shape=(1, Chans, Samples))

    gaus = GaussianNoise(0.1)(input1, training=True)

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(1, Chans, Samples),
                    use_bias=False, data_format='channels_first')(gaus)

    block1 = EnKLayer(filter=F1, EnK=EnK, InputData_shape=(1, Chans, Samples),
                      output_dimension=(1, 1, kernLength), conv2DOutput=block1)(gaus)

    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_first',
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)  # changed by avinash
    block1 = Activation('elu')(block1)

    block1 = AveragePooling2D((1, 4), data_format='channels_first', )(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),  # changed by avinash
                             data_format='channels_first', use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8), data_format='channels_first', )(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten', data_format='channels_first')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def ConvGau_EEGNet(nb_classes, Chans=64, Samples=128,
                   dropoutRate=0.5, kernLength=64, F1=8,
                   D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    # with tf.Graph().as_default() as g:
    input1 = Input(shape=(1, Chans, Samples))

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(1, Chans, Samples),
                    use_bias=False, data_format='channels_first')(input1)

    gaus = GaussianNoise(0.1)(block1, training=True)

    block1 = EnKLayer(filter=F1, EnK=EnK, InputData_shape=(1, Chans, Samples),
                      output_dimension=(1, 1, kernLength), conv2DOutput=gaus)(input1)

    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_first',
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)  # changed by avinash
    block1 = Activation('elu')(block1)

    block1 = AveragePooling2D((1, 4), data_format='channels_first', )(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),  # changed by avinash
                             data_format='channels_first', use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8), data_format='channels_first', )(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten', data_format='channels_first')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


# def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
def ShallowConvNet(nb_classes, Chans=64, Samples=128,
                   dropoutRate=0.5, kernLength=64, F1=8,
                   D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    # start the model
    input_main = Input((1, Chans, Samples))

    block1 = Conv2D(40, (1, 13),
                    input_shape=(1, Chans, Samples), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(input_main)

    # Including EnK Layer once
    # MyConv2D_TF(filter=F1, EnK=EnK, InputData_shape=(1, Chans, Samples), output_dimension=(1, 1, kernLength))(input1)
    block1 = EnKLayer(filter=40, EnK=EnK, InputData_shape=(1, Chans, Samples),
                      output_dimension=(1, 1, 13), conv2DOutput=block1)(input_main)

    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7), data_format='channels_first')(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten(data_format='channels_first')(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def Gau_ShallowConvNet(nb_classes, Chans=64, Samples=128,
                       dropoutRate=0.5, kernLength=64, F1=8,
                       D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    # start the model
    input_main = Input((1, Chans, Samples))

    gaus = GaussianNoise(0.1)(input_main, training=True)

    block1 = Conv2D(40, (1, 13),
                    input_shape=(1, Chans, Samples), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(gaus)

    # Including EnK Layer once
    # MyConv2D_TF(filter=F1, EnK=EnK, InputData_shape=(1, Chans, Samples), output_dimension=(1, 1, kernLength))(input1)
    block1 = EnKLayer(filter=40, EnK=EnK, InputData_shape=(1, Chans, Samples),
                      output_dimension=(1, 1, 13), conv2DOutput=block1)(gaus)

    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7), data_format='channels_first')(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten(data_format='channels_first')(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def ConvGau_ShallowConvNet(nb_classes, Chans=64, Samples=128,
                       dropoutRate=0.5, kernLength=64, F1=8,
                       D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    # start the model
    input_main = Input((1, Chans, Samples))



    block1 = Conv2D(40, (1, 13),
                    input_shape=(1, Chans, Samples), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(input_main)

    gaus = GaussianNoise(0.1)(block1, training=True)

    # Including EnK Layer once
    # MyConv2D_TF(filter=F1, EnK=EnK, InputData_shape=(1, Chans, Samples), output_dimension=(1, 1, kernLength))(input1)
    block1 = EnKLayer(filter=40, EnK=EnK, InputData_shape=(1, Chans, Samples),
                      output_dimension=(1, 1, 13), conv2DOutput=gaus)(input_main)

    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7), data_format='channels_first')(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten(data_format='channels_first')(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet(nb_classes, Chans=64, Samples=128,
                dropoutRate=0.5, kernLength=64, F1=8,
                D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    # start the model
    input_main = Input((1, Chans, Samples))

    block1 = Conv2D(25, (1, 5),
                    input_shape=(1, Chans, Samples), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(input_main)

    # Using EnK Layer
    block1 = EnKLayer(filter=25, EnK=EnK, InputData_shape=(1, Chans, Samples),
                      output_dimension=(1, 1, 5), conv2DOutput=block1)(input_main)
    block1 = Conv2D(25, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten(data_format='channels_first')(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def Gau_DeepConvNet(nb_classes, Chans=64, Samples=128,
                    dropoutRate=0.5, kernLength=64, F1=8,
                    D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    # start the model
    input_main = Input((1, Chans, Samples))

    gaus = GaussianNoise(0.1)(input_main, training=True)

    block1 = Conv2D(25, (1, 5),
                    input_shape=(1, Chans, Samples), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(gaus)

    # Using EnK Layer
    block1 = EnKLayer(filter=25, EnK=EnK, InputData_shape=(1, Chans, Samples),
                      output_dimension=(1, 1, 5), conv2DOutput=block1)(gaus)
    block1 = Conv2D(25, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten(data_format='channels_first')(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def ConvGau_DeepConvNet(nb_classes, Chans=64, Samples=128,
                    dropoutRate=0.5, kernLength=64, F1=8,
                    D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    # start the model
    input_main = Input((1, Chans, Samples))



    block1 = Conv2D(25, (1, 5),
                    input_shape=(1, Chans, Samples), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(input_main)

    gaus = GaussianNoise(0.1)(block1, training=True)

    # Using EnK Layer
    block1 = EnKLayer(filter=25, EnK=EnK, InputData_shape=(1, Chans, Samples),
                      output_dimension=(1, 1, 5), conv2DOutput=gaus)(input_main)

    block1 = Conv2D(25, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten(data_format='channels_first')(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

# def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
def CRNN(nb_classes, Chans=64, Samples=128,
                   dropoutRate=0.5, kernLength=64, F1=8,
                   D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):


    input_main = Input((1000, 1, Chans, Samples)) # None (BS) x time x Chan x H x W

    block1 =ConvLSTM2D(F1, (1, kernLength),padding="same",data_format='channels_first',
                               dropout=0.5,)(input_main)
    # Including EnK Layer once
    # MyConv2D_TF(filter=F1, EnK=EnK, InputData_shape=(1, Chans, Samples), output_dimension=(1, 1, kernLength))(input1)
    block1 = EnKLayer(filter=40, EnK=EnK, InputData_shape=(1, Chans, Samples),
                      output_dimension=(1, 1, 13), conv2DOutput=block1)(input_main)

    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7), data_format='channels_first')(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten(data_format='channels_first')(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)
