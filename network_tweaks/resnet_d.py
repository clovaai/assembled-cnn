from absl import flags
from absl import logging

import tensorflow as tf
from tensorflow.python.keras import backend

from conv_bn import fixed_padding
from conv_bn import conv2d_fixed_padding
from conv_bn import batch_norm

FLAGS = flags.FLAGS

layers = tf.keras.layers

def tweakC(x, num_filters):
  data_format = backend.image_data_format()
  x = conv2d_fixed_padding(x, num_filters // 2, 3, 2, 'dconv1')
  x = batch_norm(x, data_format, name='bn_dconv1')
  x = layers.Activation('relu')(x)

  x = conv2d_fixed_padding(x, num_filters // 2, 3, 1, 'dconv2')
  x = batch_norm(x, data_format, name='bn_dconv2')
  x = layers.Activation('relu')(x)

  x = conv2d_fixed_padding(x, num_filters, 3, 1, 'dconv3')

  return x

def tweakD(x, num_filters, strides, name):
  data_format = backend.image_data_format()
  # pool_size = 3 if FLAGS.use_bl else 2
  pool_size = 2
  if strides > 1:
    x = fixed_padding(x, pool_size)
    x = layers.AveragePooling2D(pool_size=pool_size,
                                strides=strides,
                                padding='SAME' if strides == 1 else 'VALID',
                                data_format=data_format)(x)
  x = conv2d_fixed_padding(x, num_filters, 1, 1, name + 'd1')

  return x
