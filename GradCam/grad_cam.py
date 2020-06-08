

'''
EnK Layer
© Avinash K Singh 
https://github.com/thinknew/enk
Licensed under MIT License
'''


### Code taken from: https://github.com/jacobgil/keras-grad-cam


from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import tensorflow.keras
import sys
import cv2


tf.compat.v1.disable_eager_execution()

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == tensorflow.keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

def grad_cam(input_model, image, category_index, layer_name,nb_classes):
    # nb_classes = 2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    model.summary()
    loss = K.sum(model.output)
    # for l in model.layers:
    #     if l.name == layer_name:
    #         print('Layer Name',l.name)
    #         print('Layer output',l[0].output)
    conv_output =  [l for l in model.layers if l.name == layer_name][0].output
    # co = [l for l in model.layers if l.name is layer_name]
    # print(co)
    # conv_output=layer_name
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    scale_w,scale_h=4,1
    cam = cv2.resize(cam, (image.shape[2]*scale_w, image.shape[3]*scale_h))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # #Return to BGR [0..255] from the preprocessed image
    # image = image[0, :]
    # image -= np.min(image)
    # image = np.minimum(image, 255)

    reshaped_image=image.reshape(image.shape[3],image.shape[2],-1)
    reshaped_image=cv2.resize(reshaped_image, dsize=(image.shape[2]*scale_w, image.shape[3]*scale_h), interpolation=cv2.INTER_CUBIC)
    reshaped_image = reshaped_image.reshape(image.shape[3]*scale_h, image.shape[2]*scale_w, -1)


    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_HOT)
    print(reshaped_image.shape)
    print(cam.shape)
    cam = np.float32(cam) + np.float32(reshaped_image)
    # cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

