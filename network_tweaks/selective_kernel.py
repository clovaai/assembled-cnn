
from absl import flags
import tensorflow as tf
from tensorflow.python.keras import backend
import resnet_model
FLAGS = flags.FLAGS
layers = tf.keras.layers
from conv_bn import conv2d_fixed_padding
from conv_bn import batch_norm
from tensorflow.python.keras import regularizers


class SKBlock(layers.Layer):

  def __init__(self, filters, strides=1, r=2, L=32, **kwargs):
    super(SKBlock, self).__init__(**kwargs)
    self.filters = filters
    self.strides = strides
    self.r = r
    self.L = L

    self.data_format = backend.image_data_format()
    self.channel_axis = 1 if self.data_format == 'channels_first' else 3
    if self.data_format == 'channels_last':
      self.bn_axis = 3
    else:
      self.bn_axis = 1

    self.pad = layers.ZeroPadding2D(padding=1)
    self.conv1 = layers.Conv2D(
      self.filters * 2, 3,
      strides=self.strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.keras.initializers.VarianceScaling(),
      kernel_regularizer=regularizers.l2(FLAGS.weight_decay) if not FLAGS.single_l2_loss_op else None,
      name='skconv1')
    self.bn1 = layers.BatchNormalization(
      axis=self.bn_axis,
      momentum=FLAGS.bn_momentum,
      epsilon=1e-5,
      name='bn_skconv1')

    d = max(int(self.filters / self.r), self.L)
    self.fc1 = layers.Conv2D(
      d, 1,
      strides=1,
      use_bias=False,
      kernel_initializer=tf.keras.initializers.VarianceScaling(),
      kernel_regularizer=regularizers.l2(FLAGS.weight_decay) if not FLAGS.single_l2_loss_op else None,
      name='sk_fc_1')

    self.bn2 = layers.BatchNormalization(
      axis=self.bn_axis,
      momentum=FLAGS.bn_momentum,
      epsilon=1e-5,
      name='bn_fc')

    self.fc2 = layers.Conv2D(
      self.filters * 2, 1,
      strides=1,
      use_bias=False,
      kernel_initializer=tf.keras.initializers.VarianceScaling(),
      kernel_regularizer=regularizers.l2(FLAGS.weight_decay) if not FLAGS.single_l2_loss_op else None,
      name='sk_fc_2')

    self.relu = layers.Activation('relu')

  def call(self, inputs, **kwargs):
    if self.strides > 1:
      inputs = self.pad(inputs)
    inputs = self.conv1(inputs)
    inputs = self.bn1(inputs)
    inputs = self.relu(inputs)

    feas = tf.split(inputs, num_or_size_splits=2, axis=self.channel_axis)
    fea_U = tf.reduce_sum(feas, axis=0)

    pooling_axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
    fea_s = tf.reduce_mean(fea_U, pooling_axes, keepdims=True)
    fea_z = self.fc1(fea_s)
    fea_z = self.bn2(fea_z)
    fea_z = self.relu(fea_z)

    attention = self.fc2(fea_z)
    attention = tf.split(attention, num_or_size_splits=2, axis=self.channel_axis)
    attention = tf.nn.softmax(attention, axis=0)
    fea_v = tf.reduce_sum(feas * attention, axis=0)

    return fea_v

