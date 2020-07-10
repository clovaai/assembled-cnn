from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl import flags
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops

FLAGS = flags.FLAGS

layers = tf.keras.layers

def keep_prob_decay(starter_kp, end_kp, num_images, batch_size, train_epochs):
  batches_per_epoch = num_images / batch_size
  decay_steps = int(train_epochs * batches_per_epoch)
  def keep_prob_decay_fn(global_step):
    # global_step = (epoch+1) * batches_per_epoch
    kp = tf.compat.v1.train.polynomial_decay(starter_kp, global_step, decay_steps,
                                   end_kp, power=1.0,
                                   cycle=False)
    if not isinstance(kp, tf.Tensor):
      kp = kp()
    # if not isinstance(kp, float):
    #   kp = kp.numpy()
    # if global_step%100 == 0:
    #   tf.summary.scalar('kp', kp, step=global_step)
    return kp

  return keep_prob_decay_fn

# class DropblockKeepProbScheduler(tf.keras.callbacks.Callback):
#   def __init__(self, schedule, verbose=0):
#     super(DropblockKeepProbScheduler, self).__init__()
#     self.schedule = schedule
#     self.verbose = verbose
#
#   def on_epoch_end(self, epoch, logs=None):
#     kp = self.schedule(epoch)
#     if not isinstance(kp, (ops.Tensor, float, np.float32, np.float64)):
#       raise ValueError('The output of the "schedule" function '
#                        'should be float.')
#     if isinstance(kp, ops.Tensor) and not kp.dtype.is_floating:
#       raise ValueError('The dtype of Tensor should be float')
#
#     for layer in self.model.layers:
#       if isinstance(layer, DropBlock):
#         K.set_value(layer.keep_prob, K.get_value(kp))
#     if self.verbose > 0:
#       print('\nEapoch %05d: DropblockKeepProbScheduler reducing keep '
#             'prob to %s.' % (epoch + 1, kp))

class DropblockKeepProbScheduler(tf.keras.callbacks.Callback):
  def __init__(self, schedule, model):
    super(DropblockKeepProbScheduler, self).__init__()
    self.schedule = schedule
    # self.steps_per_epoch = steps_per_epoch
    # self.epochs = 0
    self.model = model


  # def on_epoch_begin(self, epoch, logs=None):
  #   self.epochs = epoch
  #   global_step = self.epochs * self.steps_per_epoch
  #   kp = self.schedule(global_step)
  #   if not isinstance(kp, (ops.Tensor, float, np.float32, np.float64)):
  #     raise ValueError('The output of the "schedule" function '
  #                      'should be float.')
  #   if isinstance(kp, ops.Tensor) and not kp.dtype.is_floating:
  #     raise ValueError('The dtype of Tensor should be float')
  #
  #   for layer in self.model.layers:
  #     if isinstance(layer, DropBlock):
  #       K.set_value(layer.keep_prob, K.get_value(kp))
  #       break

    # for layer in self.model.layers:
    #   if isinstance(layer, DropBlock):
    #     print('layer.keep_prob', layer.keep_prob)

    # print('\nEapoch %05d: DropblockKeepProbScheduler reducing keep '
    #       'prob to %s.' % (epoch + 1, kp))

  def on_batch_begin(self, batch, logs=None):
    global_step = batch
    kp = self.schedule(global_step)
    if not isinstance(kp, (ops.Tensor, float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    if isinstance(kp, ops.Tensor) and not kp.dtype.is_floating:
      raise ValueError('The dtype of Tensor should be float')

    # i = 0
    for layer in self.model.layers:
      if isinstance(layer, DropBlock):
        K.set_value(layer.keep_prob, K.get_value(kp))
        # if i == 0:
          # K.set_value(layer.keep_prob, K.get_value(kp))
          # layer.keep_prob = kp
          # logging.info('setting')
        # i += 1
        # logging.info(layer.keep_prob)
        # logging.info(layer.keep_prob)

        # logging.info(layer.keep_prob)
        # break

    # if batch % 100 == 0:
    #   for layer in self.model.layers:
    #     if isinstance(layer, DropBlock):
    #       K.set_value(layer.keep_prob, K.get_value(kp))
    #       K.set_value(layer.keep_prob, kp)
    #       logging.info(layer.keep_prob)
    #
    #     layer.keep_prob = kp
    #     break


class DropBlock(layers.Layer):

  def __init__(self, keep_prob, block_size, gamma_scale, seed=None, **kwargs):
    super(DropBlock, self).__init__(**kwargs)
    self.keep_prob = keep_prob
    self.block_size = block_size
    self.gamma_scale = gamma_scale
    self.seed = seed

  def _bernoulli(self, shape, mean, seed=None, dtype=tf.float32):
    return tf.nn.relu(tf.sign(mean - tf.random.uniform(shape, minval=0, maxval=1, dtype=dtype, seed=seed)))

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def dropped_inputs():
      return self._dropblock(inputs)

    output = tf_utils.smart_cond(training,
                                 dropped_inputs,
                                 lambda: tf.identity(inputs))
    return output

  def _dropblock(self, x):
    """
    Dropblock layer. For more details, refer to https://arxiv.org/abs/1810.12890
    """
    keep_prob = self.keep_prob
    block_size = self.block_size
    gamma_scale = self.gamma_scale
    seed = self.seed

    # Early return if nothing needs to be dropped.
    if (isinstance(keep_prob, float) and keep_prob == 1) or gamma_scale == 0:
      return x

    # with tf.name_scope(name, "dropblock", [x]) as name:
    if not x.dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
    if isinstance(keep_prob, float) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)

    br = (block_size - 1) // 2
    tl = (block_size - 1) - br
    data_format = K.image_data_format()

    if data_format == 'channels_last':
      _, h, w, c = tf.shape(x)
      sampling_mask_shape = tf.stack([1, h - block_size + 1, w - block_size + 1, c])
      pad_shape = [[0, 0], [tl, br], [tl, br], [0, 0]]
    elif data_format == 'channels_first':
      # _, c, h, w = \
      # c = tf.shape(x)
      shape = tf.shape(x)
      c = shape[1]
      h = shape[2]
      w = shape[3]
      sampling_mask_shape = tf.stack([1, c, h - block_size + 1, w - block_size + 1])
      pad_shape = [[0, 0], [0, 0], [tl, br], [tl, br]]
    else:
      raise NotImplementedError

    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)
    gamma = (1. - keep_prob) * (w * h) / (block_size ** 2) / ((w - block_size + 1) * (h - block_size + 1))
    gamma = gamma_scale * gamma
    mask = self._bernoulli(sampling_mask_shape, gamma, seed, tf.float32)
    mask = tf.pad(mask, pad_shape)

    xdtype_mask = tf.cast(mask, x.dtype)
    maxpool2d = layers.MaxPooling2D((block_size, block_size),
                                    strides=(1, 1),
                                    padding='same',
                                    data_format=data_format)
    xdtype_mask = maxpool2d(xdtype_mask)

    xdtype_mask = 1 - xdtype_mask
    fp32_mask = tf.cast(xdtype_mask, tf.float32)
    ret = tf.multiply(x, xdtype_mask)
    float32_mask_size = tf.cast(tf.size(fp32_mask), tf.float32)
    float32_mask_reduce_sum = tf.reduce_sum(fp32_mask)
    normalize_factor = tf.cast(float32_mask_size / (float32_mask_reduce_sum + 1e-8), x.dtype)
    ret = ret * normalize_factor
    return ret
