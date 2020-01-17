# coding=utf8
# ==============================================================================
# Copyright 2020-present NAVER Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from nets.model_helper import *
import numpy as np
import math


def generalized_mean_pooling(x, p=3, data_format='channels_first'):
  if data_format == 'channels_first':
    _, c, h, w = x.shape.as_list()
    reduce_axis = [2, 3]
  else:
    _, h, w, c = x.shape.as_list()
    reduce_axis = [1, 2]

  N = tf.to_float(tf.multiply(h, w))
  if x.dtype == tf.float16:
    # For numerical stability, cast to fp32, calculate, and then change back to fp16.
    x = tf.cast(x, tf.float32)

  epsilon = 1e-6
  x = tf.clip_by_value(x, epsilon, 1e12)
  x_p = tf.pow(x, p)
  x_p_sum = tf.maximum(tf.reduce_sum(x_p, axis=reduce_axis, keep_dims=True), epsilon)
  pooled_x = tf.pow(N, -1.0 / p) * tf.pow(x_p_sum, 1 / p)
  if x.dtype == tf.float16:
    pooled_x = tf.cast(pooled_x, tf.float16)
  return pooled_x


def anti_aliased_downsample(inp, data_format='channels_first',
                            filt_size=3, stride=2, name=None,
                            pad_off=0):
  pading_size = int(1. * (filt_size - 1) / 2)
  if data_format == 'channels_first':
    pad_sizes = [[0, 0], [0, 0], [pading_size + pad_off, pading_size + pad_off],
                 [pading_size + pad_off, pading_size + pad_off]]
  else:
    pad_sizes = [[0, 0], [pading_size + pad_off, pading_size + pad_off], [pading_size + pad_off, pading_size + pad_off],
                 [0, 0]]

  if filt_size == 1:
    a = np.array([1., ])
  elif filt_size == 2:
    a = np.array([1., 1.])
  elif filt_size == 3:
    a = np.array([1., 2., 1.])
  elif filt_size == 4:
    a = np.array([1., 3., 3., 1.])
  elif filt_size == 5:
    a = np.array([1., 4., 6., 4., 1.])
  elif filt_size == 6:
    a = np.array([1., 5., 10., 10., 5., 1.])
  elif filt_size == 7:
    a = np.array([1., 6., 15., 20., 15., 6., 1.])

  channel_axis = 1 if data_format == 'channels_first' else 3
  G = inp.shape[channel_axis]

  filt = tf.constant(a[:, None] * a[None, :], inp.dtype)
  filt = filt / tf.reduce_sum(filt)
  filt = tf.reshape(filt, [filt_size, filt_size, 1, 1])
  filt = tf.tile(filt, [1, 1, 1, G])

  if filt_size == 1:
    if pad_off == 0:
      return inp[:, :, ::stride, ::stride]
    else:
      padded = tf.pad(inp, pad_sizes, "REFLECT")
      return padded[:, :, ::stride, ::stride]
  else:
    inp = tf.pad(inp, pad_sizes, "REFLECT")
    data_format = "NCHW" if data_format == 'channels_first' else "NHWC"

    strides = [1, 1, stride, stride] if data_format == 'NCHW' else [1, stride, stride, 1]

    with tf.variable_scope(name, "anti_alias", [inp]) as name:
      try:
        output = tf.nn.conv2d(inp, filt,
                              strides=strides,
                              padding='VALID',
                              data_format=data_format)
      except:  # When not support group conv
        tf.logging.info('Group conv by looping')
        inp = tf.split(inp, G, axis=channel_axis)
        filters = tf.split(filt, G, axis=3)
        output = tf.concat(
          [tf.nn.conv2d(i, f,
                        strides=strides,
                        padding='VALID',
                        data_format=data_format) for i, f in zip(inp, filters)], axis=1 if data_format == 'NCHW' else 3)

    return output


def sk_conv2d(inputs, filters, strides, r=2, L=32, data_format='channels_first', training=True, name=None,
              bn_momentum=0.997):
  channel_axis = 1 if data_format == 'channels_first' else 3
  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters * 2, kernel_size=3, strides=strides,
    data_format=data_format)
  inputs = batch_norm(inputs, training, data_format, momentum=bn_momentum)
  inputs = tf.nn.relu(inputs)

  if data_format == 'channels_last':
    _, height, width, channel = inputs.shape.as_list()
  elif data_format == 'channels_first':
    _, channel, height, width = inputs.shape.as_list()
  else:
    raise NotImplementedError

  # feas_shape = [filters, height, width] if data_format == 'channels_first' else [height, width, filters]
  # feas = tf.reshape(inputs, [2, -1] + feas_shape)
  feas = tf.split(inputs, num_or_size_splits=2, axis=channel_axis)
  fea_U = tf.reduce_sum(feas, axis=0)

  pooling_axes = [2, 3] if data_format == 'channels_first' else [1, 2]
  fea_s = tf.reduce_mean(fea_U, pooling_axes, keepdims=True)

  d = max(int(filters / r), L)
  with tf.variable_scope(name, "sk_block", [inputs]) as name:
    fea_z = tf.layers.conv2d(
      inputs=fea_s, filters=d, kernel_size=1, strides=1,
      kernel_initializer=tf.variance_scaling_initializer(),
      use_bias=False,
      data_format=data_format, name='sk_fc_1')
    fea_z = batch_norm(fea_z, training, data_format, momentum=bn_momentum)
    fea_z = tf.nn.relu(fea_z)
    attention = tf.layers.conv2d(
      inputs=fea_z, filters=filters * 2, kernel_size=1, strides=1,
      kernel_initializer=tf.variance_scaling_initializer(),
      use_bias=False,
      data_format=data_format, name='sk_fc_2')
    # attention_shape = [filters, 1, 1] if data_format == 'channels_first' else [1, 1, filters]
    # attention = tf.reshape(attention, [2, -1] + attention_shape)
    attention = tf.split(attention, num_or_size_splits=2, axis=channel_axis)
    attention = tf.nn.softmax(attention, axis=0)
    fea_v = tf.reduce_sum(feas * attention, axis=0)

  return fea_v

def se_block(x, name=None, ratio=16, data_format='channels_last'):
  """ https://arxiv.org/abs/1709.01507.
  """
  # h, w = x.size()[-2:]
  if data_format == 'channels_last':
    _, height, width, channel = x.shape.as_list()
    reduce_spatial = [1, 2]
  elif data_format == 'channels_first':
    _, channel, height, width = x.shape.as_list()
    reduce_spatial = [2, 3]
  else:
    raise NotImplementedError

  with tf.variable_scope(name, "se_block", [x]) as name:
    squeeze = tf.reduce_mean(x, axis=reduce_spatial, keepdims=True)
    excitation = tf.layers.conv2d(
      inputs=squeeze, filters=channel // ratio, kernel_size=1, strides=1,
      kernel_initializer=tf.variance_scaling_initializer(),
      use_bias=False,
      data_format=data_format, name='seblock_dense_1')
    excitation = tf.nn.relu(excitation)
    excitation = tf.layers.conv2d(
      inputs=excitation, filters=channel, kernel_size=1, strides=1,
      kernel_initializer=tf.variance_scaling_initializer(),
      use_bias=False,
      data_format=data_format, name='seblock_dense_2')
    excitation = tf.nn.sigmoid(excitation)
    scale = x * excitation
  return scale


def _bernoulli(shape, mean, seed=None, dtype=tf.float32):
  return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=dtype, seed=seed)))


def dropblock(x, keep_prob, block_size, gamma_scale=1.0, seed=None, name=None,
              data_format='channels_last', is_training=True):  # pylint: disable=invalid-name
  """
  Dropblock layer. For more details, refer to https://arxiv.org/abs/1810.12890
  :param x: A floating point tensor.
  :param keep_prob: A scalar Tensor with the same type as x. The probability that each element is kept.
  :param block_size: The block size to drop
  :param gamma_scale: The multiplier to gamma.
  :param seed:  Python integer. Used to create random seeds.
  :param name: A name for this operation (optional)
  :param data_format: 'channels_last' or 'channels_first'
  :param is_training: If False, do nothing.
  :return: A Tensor of the same shape of x.
  """
  if not is_training:
    return x

  # Early return if nothing needs to be dropped.
  if (isinstance(keep_prob, float) and keep_prob == 1) or gamma_scale == 0:
    return x

  with tf.name_scope(name, "dropblock", [x]) as name:
    if not x.dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
    if isinstance(keep_prob, float) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)

    br = (block_size - 1) // 2
    tl = (block_size - 1) - br
    if data_format == 'channels_last':
      _, h, w, c = x.shape.as_list()
      sampling_mask_shape = tf.stack([1, h - block_size + 1, w - block_size + 1, c])
      pad_shape = [[0, 0], [tl, br], [tl, br], [0, 0]]
    elif data_format == 'channels_first':
      _, c, h, w = x.shape.as_list()
      sampling_mask_shape = tf.stack([1, c, h - block_size + 1, w - block_size + 1])
      pad_shape = [[0, 0], [0, 0], [tl, br], [tl, br]]
    else:
      raise NotImplementedError

    gamma = (1. - keep_prob) * (w * h) / (block_size ** 2) / ((w - block_size + 1) * (h - block_size + 1))
    gamma = gamma_scale * gamma
    mask = _bernoulli(sampling_mask_shape, gamma, seed, tf.float32)
    mask = tf.pad(mask, pad_shape)

    xdtype_mask = tf.cast(mask, x.dtype)
    xdtype_mask = tf.layers.max_pooling2d(
      inputs=xdtype_mask, pool_size=block_size,
      strides=1, padding='SAME',
      data_format=data_format)

    xdtype_mask = 1 - xdtype_mask
    fp32_mask = tf.cast(xdtype_mask, tf.float32)
    ret = tf.multiply(x, xdtype_mask)
    float32_mask_size = tf.cast(tf.size(fp32_mask), tf.float32)
    float32_mask_reduce_sum = tf.reduce_sum(fp32_mask)
    normalize_factor = tf.cast(float32_mask_size / (float32_mask_reduce_sum + 1e-8), x.dtype)
    ret = ret * normalize_factor
    return ret
