from absl import flags
import tensorflow as tf
from tensorflow.python.keras import backend
FLAGS = flags.FLAGS

layers = tf.keras.layers
from tensorflow.python.keras import regularizers

class SEBlock(layers.Layer):

  def __init__(self, channel, ratio=16, reduce_channels=None, **kwargs):
    super(SEBlock, self).__init__(**kwargs)
    self.ratio = ratio
    if not reduce_channels:
      reduce_channels = channel // self.ratio

    self.fc1 = layers.Conv2D(
      reduce_channels, 1,
      strides=1,
      use_bias=False,
      kernel_initializer=tf.keras.initializers.VarianceScaling(),
      kernel_regularizer=regularizers.l2(FLAGS.weight_decay) if not FLAGS.single_l2_loss_op else None,
      name='se_fc_1')

    self.fc2 = layers.Conv2D(
      channel, 1,
      strides=1,
      use_bias=False,
      kernel_initializer=tf.keras.initializers.VarianceScaling(),
      kernel_regularizer=regularizers.l2(FLAGS.weight_decay) if not FLAGS.single_l2_loss_op else None,
      name='se_fc_2')

  def call(self, x, **kwargs):
    """ https://arxiv.org/abs/1709.01507.
     """
    data_format = backend.image_data_format()
    if data_format == 'channels_last':
      reduce_spatial = [1, 2]
    elif data_format == 'channels_first':
      reduce_spatial = [2, 3]
    else:
      raise NotImplementedError

    squeeze = tf.reduce_mean(x, axis=reduce_spatial, keepdims=True)
    excitation = self.fc1(squeeze)
    excitation = layers.Activation('relu')(excitation)
    excitation = self.fc2(excitation)
    excitation = tf.nn.sigmoid(excitation)
    scale = x * excitation
    return scale
