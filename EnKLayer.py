
'''
EnK Layer
© Avinash K Singh 
https://github.com/thinknew/enk
Licensed under MIT License
'''

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow.compat.v1 as tf
from im2col import *
from tensorflow.keras.layers import Conv2D


def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

# Using Tenssorflow Conv2d Function and adding EnKernel methodology
class EnKLayer(Layer):  # With Leanrable Parameters

    def __init__(self, filter, EnK,InputData_shape,output_dimension,conv2DOutput,dynamic=True):

        self.filter = filter
        self.output_dimension = output_dimension
        self.EnK =EnK
        self.InputData_shape=InputData_shape
        self.conv2DOutput=conv2DOutput
        # self.encoding_vector= encoding_vector
        super(EnKLayer, self).__init__()

    def build(self, input_shape):
        # Kernel dimension [filter_height, filter_width, in_channels, out_channels]
        self.kernel = self.add_weight(name='kernel', shape=(self.output_dimension[1],
                                                            self.output_dimension[2],self.output_dimension[0],self.filter),
                                      initializer='ones',trainable=False)

        if self.EnK:
            self.scaleFactor = self.add_weight(name='scaleFactor',shape=(1,), initializer='zeros', trainable=True)
        else:
            self.scaleFactor = 0

        super(EnKLayer, self).build(input_shape)

    def call(self, input):

        Part1 = self.conv2DOutput

        Part2 = tf.nn.conv2d(input, self.kernel, data_format='NCHW', padding='SAME') # output shape : Batch Size X Filter Size X H_out X W_Out

        # Tiling approach
        encoding_vector1 = tf.range(0, self.InputData_shape[2], 1, dtype=tf.float32)*self.scaleFactor # Defining a range from 0 to width of input tensor
        encoding_vector2 = tf.reshape(encoding_vector1,[1,1,1,-1]) # To Match with Input tensor shape


        Part3 = tf.multiply(Part2, encoding_vector2) # Element Wise Multiplication


        output =tf.math.add(Part1, Part3) # Adding the Conv2D output to fullfile the logic of EnKernel Approach

        return output

    def get_config(self):

        config = super(EnKLayer,self).get_config()
        config.update({'scaleFactor': self.scaleFactor})
        return config
