# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().
This is ResNet model version 1.5.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl import flags

import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
import imagenet_preprocessing

from network_tweaks import resnet_d
from regularization.dropblock import DropBlock
from network_tweaks import selective_kernel, big_little
from network_tweaks import squeeze_excitation
from network_tweaks import antialias
from conv_bn import fixed_padding
from conv_bn import conv2d_fixed_padding
from conv_bn import batch_norm

import math

layers = tf.keras.layers

FLAGS = flags.FLAGS

def _gen_l2_regularizer(use_l2_regularizer=True):
  return regularizers.l2(FLAGS.weight_decay) if use_l2_regularizer else None


def bottleneck(input_tensor,
               kernel_size,
               filters,
               block,
               stage,
               strides,
               name,
               kp,
               projection=False):

  filters1, filters2, filters3 = filters
  conv_name_base = name + 'res' + str(stage) + "_" + block + '_branch'
  bn_name_base = name + 'bn' + str(stage) + "_" + block + '_branch'
  data_format = backend.image_data_format()

  x = conv2d_fixed_padding(input_tensor, filters1,
                           kernel_size=1,
                           strides=1,
                           name=conv_name_base + '2a')

  x = batch_norm(x, data_format, name=bn_name_base + '2a')
  if FLAGS.use_dropblock and (stage == 4 or stage == 5):
    x = DropBlock(kp, block_size=7, gamma_scale=0.25 if stage == 4 else 1.0)(x)
  x = layers.Activation('relu')(x)

  if FLAGS.use_sk_block:
    if projection:
      if 'sconv' in FLAGS.anti_alias_type or strides == 1:
        sk_strides = 1
      elif strides == 2:
        sk_strides = 2
      else:
        raise NotImplementedError('check out `strides`')
    else:
      sk_strides = 1

    x = selective_kernel.SKBlock(filters2, strides=sk_strides)(x)
    if FLAGS.use_dropblock and (stage == 4 or stage == 5):
      # kp = backend.variable(FLAGS.dropblock_kp[0])
      # kp._trainable = False
      x = DropBlock(kp, block_size=7, gamma_scale=0.25 if stage == 4 else 1.0)(x)
  else:
    x = conv2d_fixed_padding(x, filters2, kernel_size,
                             strides=1 if ('sconv' in FLAGS.anti_alias_type or not projection) else strides,
                             name=conv_name_base + '2b')
    x = batch_norm(x, data_format, name=bn_name_base + '2b')
    if FLAGS.use_dropblock and (stage == 4 or stage == 5):
      x = DropBlock(kp, block_size=7, gamma_scale=0.25 if stage == 4 else 1.0)(x)
    x = layers.Activation('relu')(x)

  if projection:
    if 'sconv' in FLAGS.anti_alias_type and strides == 2:
      logging.info('Anti-Alias stridedConv2 On')
      x = antialias.AntiAliasing(filt_size=FLAGS.anti_alias_filter_size)(x)

  if FLAGS.use_dedicated_se and stage is not 5:
    reduce_channels = max(filters3 // 8, 64)
    x = squeeze_excitation.SEBlock(filters2, reduce_channels=reduce_channels)(x)

  x = conv2d_fixed_padding(x, filters3,
                           kernel_size=1,
                           strides=1,
                           name=conv_name_base + '2c')

  x = batch_norm(x, data_format, zero_gamma=FLAGS.zero_gamma, name=bn_name_base + '2c')
  if FLAGS.use_dropblock and (stage == 4 or stage == 5):
    # kp = backend.variable(FLAGS.dropblock_kp[0])
    # kp._trainable = False
    x = DropBlock(kp, block_size=7, gamma_scale=0.25 if stage == 4 else 1.0)(x)

  if FLAGS.use_se_block:
    x = squeeze_excitation.SEBlock(filters3)(x)
  return  x


def basic(input_tensor,
          kernel_size,
          filters,
          block,
          stage,
          strides,
          name,
          kp,
          projection=False):
  logging.info('basic block on stage {}'.format(stage))
  _, filters , _ = filters
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
  conv_name_base = name + 'res' + str(stage) + "_" + block + '_branch'
  bn_name_base = name + 'bn' + str(stage) + "_" + block + '_branch'

  data_format = backend.image_data_format()

  if FLAGS.use_sk_block:

    if projection:
      if 'sconv' in FLAGS.anti_alias_type or strides == 1:
        sk_strides = 1
      elif strides == 2:
        sk_strides = 2
      else:
        raise NotImplementedError('check out `strides`')
    else:
      sk_strides = 1

    x = selective_kernel.SKBlock(filters, strides=sk_strides)(input_tensor)
    if FLAGS.use_dropblock and (stage == 4 or stage == 5):
      # kp = backend.variable(FLAGS.dropblock_kp[0])
      # kp._trainable = False
      x = DropBlock(kp, block_size=7, gamma_scale=0.25 if stage == 4 else 1.0)(x)
  else:
    x = conv2d_fixed_padding(input_tensor, filters,
                             kernel_size=kernel_size,
                             strides=1 if ('sconv' in FLAGS.anti_alias_type or not projection) else strides,
                             name=conv_name_base + '2basic_a')

    x = batch_norm(x, data_format, name=bn_name_base + '2basic_a')
    if FLAGS.use_dropblock and (stage == 4 or stage == 5):
      # kp = backend.variable(FLAGS.dropblock_kp[0])
      # kp._trainable = False
      x = DropBlock(kp, block_size=7, gamma_scale=0.25 if stage == 4 else 1.0)(x)
    x = layers.Activation('relu')(x)

  if projection:
    if 'sconv' in FLAGS.anti_alias_type and strides == 2:
      logging.info('Anti-Alias stridedConv2 On')
      x = antialias.AntiAliasing(filt_size=FLAGS.anti_alias_filter_size)(x)


  x = conv2d_fixed_padding(x, filters,
                           kernel_size=kernel_size,
                           strides=1,
                           name=conv_name_base + '2basic_b')

  x = batch_norm(x, data_format, zero_gamma=FLAGS.zero_gamma, name=bn_name_base + '2basic_b')
  if FLAGS.use_dropblock and (stage == 4 or stage == 5):
    # kp = backend.variable(FLAGS.dropblock_kp[0])
    # kp._trainable = False
    x = DropBlock(kp, block_size=7, gamma_scale=0.25 if stage == 4 else 1.0)(x)

  if FLAGS.use_dedicated_se:
    reduce_channels = max(filters // 4, 64)
    x = squeeze_excitation.SEBlock(filters, reduce_channels=reduce_channels)(x)
  return  x

def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   last_relu=True,
                   name='',
                   kp=None):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.

  Returns:
    Output tensor for the block.
  """
  block_fn = basic if FLAGS.use_bs and 2 <= stage <= 3 else bottleneck

  x = block_fn(input_tensor=input_tensor,
               kernel_size=kernel_size,
               filters=filters,
               block=block,
               stage=stage,
               strides=1,
               name=name,
               kp=kp,
               projection=False)

  x = layers.add([x, input_tensor])
  if last_relu:
    x = layers.Activation('relu')(x)
  return x

def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=2,
               last_relu=True,
               name='',
               kp=None):
  """A block that has a conv layer at shortcut.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    zero_gamma: Initialize γ = 0 for all BN layers that sit at the end of a residual block.
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.

  Returns:
    Output tensor for the block.
  """
  _, _, filters3 = filters
  if FLAGS.use_bs and 2 <= stage <= 3:
    ## basic block, no bottleneck
    filters3 = filters3 // 4

  conv_name_base = name + 'res' + str(stage) + "_" + block + '_branch'
  bn_name_base = name + 'bn' + str(stage) + "_" + block + '_branch'
  data_format = backend.image_data_format()

  if FLAGS.use_resnet_d:
    shortcut = resnet_d.tweakD(input_tensor, filters3, strides, conv_name_base)
  else:
    shortcut = conv2d_fixed_padding(input_tensor, filters3, 1, strides, conv_name_base + '1')
  shortcut = batch_norm(shortcut, data_format, name=bn_name_base + '1')

  if FLAGS.use_dropblock and (stage == 4 or stage == 5) :
    shortcut = DropBlock(kp, block_size=7, gamma_scale=0.25 if stage == 4 else 1.0)(shortcut)

  block_fn = basic if FLAGS.use_bs and 2 <= stage <= 3 else bottleneck

  x = block_fn(input_tensor=input_tensor,
               kernel_size=kernel_size,
               filters=filters,
               block=block,
               stage=stage,
               strides=strides,
               name=name,
               kp=kp,
               projection=True)

  x = layers.add([x, shortcut])
  if last_relu:
    x = layers.Activation('relu')(x)
  return x


def block_layer(x, stage, num_blocks, num_filters,
                conv_strides=2,
                use_l2_regularizer=True,
                kp=None,
                last_relu=True,
                name=''):
  x = conv_block(
    x,
    3, num_filters,
    stage=stage,
    block='0',
    strides=conv_strides,
    kp=kp,
    last_relu=True,
    name=name)
  for i in range(1, num_blocks):
    x = identity_block(
      x,
      3, num_filters,
      stage=stage,
      block=str(i),
      kp=kp,
      last_relu=last_relu if i == num_blocks - 1 else True,
      name=name)
  return x

def resnet50(num_classes,
             batch_size=None,
             use_l2_regularizer=True,
             rescale_inputs=False):
  """Instantiates the ResNet50 architecture.

  Args:
    num_classes: `int` number of classes for image classification.
    batch_size: Size of the batches for each step.
    use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
    rescale_inputs: whether to rescale inputs from 0 to 1.

  Returns:
      A Keras model instance.
  """
  ###############
  # net setting
  ###############
  if FLAGS.use_bl:
    block_configure = {
      50: [0, 0, 3, 4, 6, 3],
      101: [0, 0, 4, 8, 18, 3],
      152: [0, 0, 5, 12, 30, 3]
    }
  else:
    block_configure = {
      50: [0, 0, 3, 4, 6, 3],
      101: [0, 0, 3, 4, 23, 3],
      152: [0, 0, 3, 8, 36, 3],
      200: [0, 0, 3, 24, 36, 3]
    }

  if FLAGS.use_bs:
    block_configure = {
      50: [0, 0, 3, 4, 11, 3],
      101: [0, 0, 4, 5, 18, 3],
      152: [0, 0, 4, 5, 24, 3],
    }

  if FLAGS.use_bl:
    block_strides = [0, 0, 2, 2, 1, 2]
  else:
    block_strides = [0, 0, 1, 2, 2, 2]

  if FLAGS.no_downsample:
    block_strides[-1] = 1

  filter_configure = [64, 64, 256]

  ###############
  # start
  ###############

  if FLAGS.debug:
    input_shape = (224, 224, 3)
  else:
    input_shape = (None, None, 3)
  img_input = layers.Input(shape=input_shape, batch_size=batch_size)
  # label_input = layers.Input(shape=FLAGS.num_classes, batch_size=batch_size)
  if rescale_inputs:
    # Hub image modules expect inputs in the range [0, 1]. This rescales these
    # inputs to the range expected by the trained model.
    x = layers.Lambda(
        lambda x: x * 255.0 - backend.constant(
            imagenet_preprocessing.CHANNEL_MEANS,
            shape=[1, 1, 3],
            dtype=x.dtype),
        name='rescale')(
            img_input)
  else:
    x = img_input

  if backend.image_data_format() == 'channels_first':
    x = layers.Lambda(
        lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
        name='transpose')(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3

  #################
  # stage 1: Stem
  #################
  data_format = backend.image_data_format()
  if FLAGS.use_resnet_d and not FLAGS.use_bl and not FLAGS.use_space2depth:
    x = resnet_d.tweakC(x, 64)
  elif FLAGS.use_space2depth:
    logging.info('Space to Depth Stem On')
    data_format_four_letter = 'NCHW' if data_format == 'channels_first' else 'NHWC'
    x = tf.nn.space_to_depth(x, 4, data_format_four_letter)
    x = conv2d_fixed_padding(x, 64, 3, 1, 'conv1')
  else:
    x = conv2d_fixed_padding(x, 64, 7, 2, 'conv1')

  x = batch_norm(x, data_format, name='bn_conv1')
  x = layers.Activation('relu')(x)

  #######################
  # stage 1: max-pooling
  #######################
  if FLAGS.use_bl:
    x = big_little.bl_stage1(x, num_filters=filter_configure[0], strides=1 if FLAGS.use_space2depth else 2)
    logging.info("blstage 1 = {}".format(x))
  elif FLAGS.use_space2depth:
    pass
  else:
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    logging.info("stage 1 = {}".format(x))

  kp = backend.variable(float(FLAGS.dropblock_kp[0]))
  kp._trainable = False

  #######################
  # stage loop: 2,3,4,5
  #######################
  if FLAGS.use_bl:
    # if False:
    for stage in range(2, 6):
      filters = [f * int(math.pow(2,stage-2)) for f in filter_configure]
      if 2 <= stage <= 4:
        x = big_little.bl_block(x, stage=stage,
                                strides=block_strides[stage],
                                num_blocks=block_configure[FLAGS.resnet_size][stage],
                                kp=kp,
                                filters=filters)
      else:
        #  normal block layer at stage 5
        x = block_layer(x,
                        stage=stage,
                        num_blocks=block_configure[FLAGS.resnet_size][stage],
                        num_filters=filters,
                        conv_strides=block_strides[stage],
                        kp=kp)
      logging.info("blstage {}, stride {} block {} = {}".format(stage, block_strides[stage], block_configure[FLAGS.resnet_size][stage],x))

  else:
    for stage in range(2, 6):
      filters = [f * int(math.pow(2,stage-2)) for f in filter_configure]
      x = block_layer(x,conv_strides=block_strides[stage],
                      num_blocks=block_configure[FLAGS.resnet_size][stage],
                      stage=stage,
                      kp=kp,
                      num_filters=filters)
      logging.info("stage {} = {}".format(stage, x))

  rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
  x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)

  logit = layers.Dense(
    num_classes,
    kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
    bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
    name='fc1000')(x)

  out = tf.cast(logit, tf.float32)
  return models.Model(img_input, out, name='resnet' + str(FLAGS.resnet_size))
