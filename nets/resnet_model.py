#!/usr/bin/env python
# coding=utf8
# This code is adapted from the https://github.com/tensorflow/models/tree/master/official/r1/resnet.
# ==========================================================================================
# NAVER’s modifications are Copyright 2020 NAVER corp. All rights reserved.
# ==========================================================================================
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets import blocks
from nets.model_helper import *
import numpy as np

DEFAULT_VERSION = 1
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format, zero_gamma=False,
                         dropblock_fn=None, se_block_fn=None, use_sk_block=False,
                         bn_momentum=0.997, anti_alias_filter_size=0, anti_alias_type="",
                         last_relu=True, block_expansion=4):
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format, momentum=bn_momentum)
    if dropblock_fn:
      shortcut = dropblock_fn(shortcut)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=1, strides=1,
    data_format=data_format)
  inputs = batch_norm(inputs, training, data_format, momentum=bn_momentum)
  if dropblock_fn:
    inputs = dropblock_fn(inputs)
  inputs = tf.nn.relu(inputs)

  if use_sk_block:
    inputs = blocks.sk_conv2d(inputs, filters, strides=1 if 'sconv' in anti_alias_type else strides,
                              training=training,
                              data_format=data_format,
                              bn_momentum=bn_momentum)
    if dropblock_fn:
      inputs = dropblock_fn(inputs)
  else:
    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1 if 'sconv' in anti_alias_type else strides,
      data_format=data_format)

    inputs = batch_norm(inputs, training, data_format, momentum=bn_momentum)
    if dropblock_fn:
      inputs = dropblock_fn(inputs)
    inputs = tf.nn.relu(inputs)

  if 'sconv' in anti_alias_type and strides != 1:
    tf.logging.info('Anti-Alias stridedConv2 On')
    inputs = blocks.anti_aliased_downsample(inputs,
                                            data_format=data_format,
                                            stride=strides,
                                            filt_size=anti_alias_filter_size)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=block_expansion * filters, kernel_size=1, strides=1,
    data_format=data_format)
  inputs = batch_norm(inputs, training, data_format, zero_gamma=zero_gamma, momentum=bn_momentum)

  if dropblock_fn:
    inputs = dropblock_fn(inputs)

  if se_block_fn:
    inputs = se_block_fn(inputs)

  inputs += shortcut

  if last_relu:
    inputs = tf.nn.relu(inputs)

  return inputs

def block_layer(inputs, filters, bottleneck, block_fn, num_blocks, strides,
                training, name, data_format, zero_gamma=False, use_resnet_d=False,
                dropblock_fn=None, se_block_fn=None, use_sk_block=False, bn_momentum=0.997,
                anti_alias_filter_size=0, anti_alias_type="", expansion=4,
                use_bl=False, last_relu=True):
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * expansion if bottleneck else filters

  def projection_shortcut(inputs):
    if 'proj' in anti_alias_type and strides != 1:
      tf.logging.info('Anti-Alias Projection Conv On')
      inputs = blocks.anti_aliased_downsample(inputs,
                                              data_format=data_format,
                                              stride=strides,
                                              filt_size=anti_alias_filter_size)
      return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=1,
        data_format=data_format)

    else:
      return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  def resnet_d_projection_shortcut(inputs):
    if strides > 1:
      inputs = fixed_padding(inputs, 2, data_format)
    inputs = tf.layers.average_pooling2d(inputs, pool_size=2, strides=strides,
                                         padding='SAME' if strides == 1 else 'VALID',
                                         data_format=data_format)
    return conv2d_fixed_padding(
      inputs=inputs, filters=filters_out, kernel_size=1, strides=1,
      data_format=data_format)

  def bl_projection_shortcut(inputs):
    if strides > 1:
      inputs = fixed_padding(inputs, 3, data_format)
      inputs = tf.layers.average_pooling2d(inputs, pool_size=3, strides=strides,
                                           padding='SAME' if strides == 1 else 'VALID',
                                           data_format=data_format)
    return conv2d_fixed_padding(
      inputs=inputs, filters=filters_out, kernel_size=1, strides=1,
      data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  if use_resnet_d:
    projection_shortcut_fn = resnet_d_projection_shortcut
  elif use_bl:  # resnet_d와 유사한데, average pooling의 kernel_size만 2에서 3으로 바뀜,
    projection_shortcut_fn = bl_projection_shortcut
  else:
    projection_shortcut_fn = projection_shortcut

  inputs = block_fn(inputs, filters, training, projection_shortcut_fn, strides,
                    data_format, zero_gamma=zero_gamma, dropblock_fn=dropblock_fn,
                    se_block_fn=se_block_fn, use_sk_block=use_sk_block, bn_momentum=bn_momentum,
                    anti_alias_filter_size=anti_alias_filter_size, anti_alias_type=anti_alias_type,
                    block_expansion=expansion)

  for i in range(1, num_blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format, zero_gamma,
                      dropblock_fn=dropblock_fn, se_block_fn=se_block_fn, use_sk_block=use_sk_block,
                      bn_momentum=bn_momentum, block_expansion=expansion,
                      last_relu=last_relu if i == num_blocks - 1 else True)

  return tf.identity(inputs, name)


class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self, resnet_size,
               bottleneck,
               num_classes,
               num_filters,
               kernel_size,
               conv_stride,
               first_pool_size,
               first_pool_stride,
               block_sizes,
               block_strides,
               zero_gamma=False,
               resnet_version=DEFAULT_VERSION,
               data_format=None,
               num_feature=None,
               use_se_block=False,
               use_sk_block=False,
               bn_momentum=0.997,
               embedding_size=0,
               anti_alias_filter_size=0,
               anti_alias_type="",
               pool_type='gap',
               bl_alpha=2,
               bl_beta=4,
               loss_type='softmax',
               dtype=DEFAULT_DTYPE):
    self.resnet_size = resnet_size

    if not data_format:
      data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
        'Resnet version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    if bottleneck:
      if resnet_version == 1 or resnet_version == 2:
        self.block_fn = _bottleneck_block_v1
      else:
        raise NotImplementedError
    else:
      raise NotImplementedError

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.dtype = dtype
    self.zero_gamma = zero_gamma
    self.embedding_size = num_feature
    self.use_se_block = use_se_block
    self.use_sk_block = use_sk_block
    self.bn_momentum = bn_momentum
    self.embedding_size = embedding_size
    self.anti_alias_filter_size = anti_alias_filter_size
    self.anti_alias_type = anti_alias_type
    self.pool_type = pool_type
    self.alpha = bl_alpha
    self.beta = bl_beta
    tf.logging.info('[loss type] {}'.format(loss_type))
    if loss_type == 'softmax':
      self.dense_bias_initializer = tf.zeros_initializer()
    elif loss_type == 'sigmoid' or loss_type == 'focal':
      tf.logging.info('[Dense layer bias initializer: -np.log(num_classes - 1)]')
      self.dense_bias_initializer = tf.constant_initializer(-np.log(num_classes - 1))
    elif loss_type == 'anchor':
      tf.logging.info('anchor loss warmup epoch: %d' % (anchor_loss_warmup_epochs))
      tf.logging.info('[Dense layer bias initializer: -np.log(num_classes - 1)]')
      self.dense_bias_initializer = tf.constant_initializer(-np.log(num_classes - 1))
    else:
      raise NotImplementedError

  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.

    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.

    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self, reuse=False):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.

    Returns:
      A variable scope for the model.
    """

    return tf.variable_scope('resnet_model', reuse=reuse,
                             custom_getter=self._custom_dtype_getter)

  def __call__(self, inputs,
               training,
               reuse=False,
               use_resnet_d=False,
               keep_prob=1.0,
               return_embedding=False):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.
      keep_prob: The keep_prob of Dropblock.
    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    with self._model_variable_scope(reuse):
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
      if use_resnet_d and self.resnet_version == 1:
        inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters // 2, kernel_size=3,
          strides=self.conv_stride, data_format=self.data_format)
        inputs = batch_norm(inputs, training, self.data_format, momentum=self.bn_momentum)
        inputs = tf.nn.relu(inputs)
        inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters // 2, kernel_size=3,
          strides=1, data_format=self.data_format)
        inputs = batch_norm(inputs, training, self.data_format, momentum=self.bn_momentum)
        inputs = tf.nn.relu(inputs)
        inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=3,
          strides=1, data_format=self.data_format)
      elif use_resnet_d and self.resnet_version == 2:
        tf.logging.warn("use_resnet_d + blresnet may causes models' predictive "
                        "performance degradation.")
        with tf.variable_scope(None, 'stage{}'.format(0)):
          inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters // 2, kernel_size=3,
            strides=self.conv_stride, data_format=self.data_format)
          inputs = batch_norm(inputs, training, self.data_format, momentum=self.bn_momentum)
          inputs = tf.nn.relu(inputs)
          inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters // 2, kernel_size=3,
            strides=1, data_format=self.data_format)
          inputs = batch_norm(inputs, training, self.data_format, momentum=self.bn_momentum)
          inputs = tf.nn.relu(inputs)
          inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=3,
            strides=1, data_format=self.data_format)
      elif self.resnet_version == 2:
        with tf.variable_scope(None, 'stage{}'.format(0)):
          inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
            strides=self.conv_stride, data_format=self.data_format)
      else:
        inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)

      inputs = tf.identity(inputs, 'initial_conv')

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.resnet_version == 1:
        inputs = batch_norm(inputs, training, self.data_format, momentum=self.bn_momentum)
        inputs = tf.nn.relu(inputs)
      elif self.resnet_version == 2:
        with tf.variable_scope(None, 'stage{}'.format(0)):
          inputs = batch_norm(inputs, training, self.data_format, momentum=self.bn_momentum)
          inputs = tf.nn.relu(inputs)


      if self.first_pool_size:
        if self.resnet_version == 2:
          # conv 구성
          # big, 3x3, 64, s2
          tf.logging.info('blModule0 On')
          with tf.variable_scope(None, 'stage{}/pool'.format(0)):
            big0 = conv2d_fixed_padding(
              inputs=inputs, filters=self.num_filters, kernel_size=3,
              strides=2, data_format=self.data_format)
            big0 = batch_norm(big0, training, self.data_format, momentum=self.bn_momentum)

            # conv 구성
            # little 3x3, 32
            # little 3x3, 32, s2
            # little 1x1, 64
            little0 = conv2d_fixed_padding(
              inputs=inputs, filters=self.num_filters // self.alpha, kernel_size=3,
              strides=1, data_format=self.data_format)
            little0 = batch_norm(little0, training, self.data_format, momentum=self.bn_momentum)
            little0 = tf.nn.relu(little0)
            little0 = conv2d_fixed_padding(
              inputs=little0, filters=self.num_filters // self.alpha, kernel_size=3,
              strides=2, data_format=self.data_format)
            little0 = batch_norm(little0, training, self.data_format, momentum=self.bn_momentum)
            little0 = tf.nn.relu(little0)
            little0 = conv2d_fixed_padding(
              inputs=little0, filters=self.num_filters, kernel_size=1,
              strides=1, data_format=self.data_format)
            little0 = batch_norm(little0, training, self.data_format, momentum=self.bn_momentum)

            inputs = tf.nn.relu(big0 + little0)
            inputs = conv2d_fixed_padding(
              inputs=inputs, filters=self.num_filters, kernel_size=1,
              strides=1, data_format=self.data_format)
            inputs = batch_norm(inputs, training, self.data_format, momentum=self.bn_momentum)
            inputs = tf.nn.relu(inputs)
        else:
          inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
          inputs = tf.identity(inputs, 'initial_max_pool')

      data_format = self.data_format

      def se_block(inputs):
        return blocks.se_block(inputs, ratio=16, data_format=data_format)

      def dropblock_for_group3(inputs):
        return blocks.dropblock(inputs, keep_prob, is_training=training,
                                block_size=7, gamma_scale=0.25, data_format=data_format)

      def dropblock_for_group4(inputs):
        return blocks.dropblock(inputs, keep_prob, is_training=training,
                                block_size=7, gamma_scale=1.0, data_format=data_format)

      if self.use_se_block:
        se_block_fn = se_block
      else:
        se_block_fn = None

      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2 ** i)

        if i == 2:
          dropblock_fn = dropblock_for_group3
        elif i == 3:
          dropblock_fn = dropblock_for_group4
        else:
          dropblock_fn = None

        if self.resnet_version == 2 and i < 3:
          tf.logging.info('blModule{} On'.format(i + 1))
          with tf.variable_scope(None, 'stage{}'.format(i + 1)):
            with tf.variable_scope(None, 'big{}'.format(i + 1)):
              big = block_layer(
                inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                block_fn=self.block_fn, num_blocks=num_blocks - 1,
                strides=2, training=training,
                name='big{}'.format(i + 1),
                data_format=self.data_format,
                zero_gamma=self.zero_gamma,
                dropblock_fn=dropblock_fn,
                se_block_fn=se_block_fn,
                use_sk_block=self.use_sk_block,
                bn_momentum=self.bn_momentum,
                anti_alias_filter_size=self.anti_alias_filter_size,
                anti_alias_type=self.anti_alias_type,
                last_relu=False,
                use_bl=True
              )

            with tf.variable_scope(None, 'little{}'.format(i + 1)):
              little = block_layer(
                inputs=inputs, filters=num_filters // self.alpha, bottleneck=self.bottleneck,
                block_fn=self.block_fn, num_blocks=max(1, num_blocks // self.beta - 1),
                strides=1, training=training,
                name='little{}'.format(i + 1),
                data_format=self.data_format,
                zero_gamma=self.zero_gamma,
                dropblock_fn=dropblock_fn,
                se_block_fn=se_block_fn,
                use_sk_block=self.use_sk_block,
                bn_momentum=self.bn_momentum,
                anti_alias_filter_size=self.anti_alias_filter_size,
                anti_alias_type=self.anti_alias_type,
                use_bl=True
              )

              little_e = conv2d_fixed_padding(
                inputs=little, filters=num_filters * 4, kernel_size=1,
                strides=1, data_format=self.data_format)
              little_e = batch_norm(little_e, training, self.data_format, momentum=self.bn_momentum)

            with tf.variable_scope(None, 'merge{}'.format(i + 1)):
              big_e = tf.keras.layers.UpSampling2D((2, 2), data_format=self.data_format)(big)

              inputs = tf.nn.relu(little_e + big_e)

              inputs = block_layer(
                inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                block_fn=self.block_fn, num_blocks=1,
                strides=self.block_strides[i], training=training,
                name='merge{}'.format(i + 1), data_format=self.data_format,
                zero_gamma=self.zero_gamma,
                dropblock_fn=dropblock_fn,
                se_block_fn=se_block_fn,
                use_sk_block=self.use_sk_block,
                bn_momentum=self.bn_momentum,
                anti_alias_filter_size=self.anti_alias_filter_size,
                anti_alias_type=self.anti_alias_type,
                use_bl=True
              )
            # print('x{}'.format(i + 1), inputs)
        elif self.resnet_version == 2 and i == 3:
          with tf.variable_scope(None, 'stage{}'.format(i + 1)):
            inputs = block_layer(
              inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
              block_fn=self.block_fn, num_blocks=num_blocks,
              strides=self.block_strides[i], training=training,
              name='block_layer{}'.format(i + 1), data_format=self.data_format,
              zero_gamma=self.zero_gamma,
              use_resnet_d=use_resnet_d,
              dropblock_fn=dropblock_fn,
              se_block_fn=se_block_fn,
              use_sk_block=self.use_sk_block,
              bn_momentum=self.bn_momentum,
              anti_alias_filter_size=self.anti_alias_filter_size,
              anti_alias_type=self.anti_alias_type,
              use_bl=True,
            )
        else:
          inputs = block_layer(
            inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
            block_fn=self.block_fn, num_blocks=num_blocks,
            strides=self.block_strides[i], training=training,
            name='block_layer{}'.format(i + 1), data_format=self.data_format,
            zero_gamma=self.zero_gamma,
            use_resnet_d=use_resnet_d,
            dropblock_fn=dropblock_fn,
            se_block_fn=se_block_fn,
            use_sk_block=self.use_sk_block,
            bn_momentum=self.bn_momentum,
            anti_alias_filter_size=self.anti_alias_filter_size,
            anti_alias_type=self.anti_alias_type,
          )
      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]

      if self.block_strides[-1] == 1:
        tf.logging.info('[No downsample at final stage]')

      if self.pool_type == 'gap':
        inputs = tf.reduce_mean(inputs, axes, keepdims=True)
      elif self.pool_type == 'gem':
        tf.logging.info('GeM POOLING ON')
        inputs = blocks.generalized_mean_pooling(inputs, data_format=data_format)
      elif self.pool_type == 'flatten':
        tf.logging.info('Use FLATTEN for POOLING')
        inputs = tf.layers.flatten(inputs)
        for axis_idx in axes:
          inputs = tf.expand_dims(inputs, axis=axis_idx)
      else:
        raise NotImplementedError
      global_pool = tf.identity(inputs, 'final_reduce_mean')

      if self.embedding_size > 0:
        tf.logging.info('[Added embedding layer before final dense layer]')
        embeddings = tf.layers.conv2d(
          inputs=global_pool, filters=self.embedding_size, kernel_size=1, strides=1,
          padding='SAME', use_bias=False,
          kernel_initializer=tf.variance_scaling_initializer(),
          data_format=data_format, name='embedding_dense')
        
        embeddings = batch_norm(embeddings, training, self.data_format,
                                momentum=self.bn_momentum,
                                name='embedding_dense_batch_normalization')
        squeezed_global_pool = tf.squeeze(embeddings, axes)
      else:
        squeezed_global_pool = tf.squeeze(global_pool, axes)

      if return_embedding:
        return squeezed_global_pool
      tf.logging.info('[Added final dense layer]')
      if self.embedding_size > 0:
        squeezed_global_pool = tf.nn.relu(squeezed_global_pool)

      final_dense = tf.layers.dense(inputs=squeezed_global_pool,
                                    units=self.num_classes,
                                    bias_initializer=self.dense_bias_initializer)
      final_dense = tf.identity(final_dense, 'final_dense')
      return final_dense
