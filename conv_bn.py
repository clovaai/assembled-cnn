from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend
from absl import flags
from tensorflow.python.keras import regularizers

layers = tf.keras.layers

FLAGS = flags.FLAGS


def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = layers.ZeroPadding2D(padding=((pad_beg, pad_end), (pad_beg, pad_end)))(inputs)
  return padded_inputs

def batch_norm(inputs, data_format, zero_gamma=False, epsilon=1e-5, name=None):
  """Performs a batch normalization using a standard set of parameters."""
  return layers.BatchNormalization(
      axis=1 if data_format == 'channels_first' else 3,
      momentum=FLAGS.bn_momentum,
      epsilon=epsilon,
      gamma_initializer='zeros' if zero_gamma else 'ones',
      name=name)(inputs)

  # return tf.layers.batch_normalization(
  #   inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
  #   momentum=momentum, epsilon=epsilon, center=True,
  #   scale=True, training=training, fused=True, gamma_initializer=gamma_initializer, name=name)

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, name):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return layers.Conv2D(
    filters, kernel_size,
    strides=strides,
    padding=('SAME' if strides == 1 else 'VALID'),
    use_bias=False,
    kernel_initializer=tf.keras.initializers.VarianceScaling(),
    kernel_regularizer=regularizers.l2(FLAGS.weight_decay) if not FLAGS.single_l2_loss_op else None,
    name=name)(inputs)

  # return tf.layers.conv2d(
  #   inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
  #   padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
  #   kernel_initializer=tf.variance_scaling_initializer(),
  #   data_format=data_format)
