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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from functions import data_config
from nets import resnet_model
from nets import run_loop_classification

def keep_prob_decay(starter_kp, end_kp, decay_steps):
  def keep_prob_decay_fn(global_step):
    kp = tf.train.polynomial_decay(starter_kp, global_step, decay_steps,
                                   end_kp, power=1.0,
                                   cycle=False)
    return kp

  return keep_prob_decay_fn


def learning_rate_with_decay(learning_rate_decay_type,
                             batch_size, batch_denom, num_images, num_epochs_per_decay,
                             learning_rate_decay_factor, end_learning_rate, piecewise_lr_boundary_epochs,
                             piecewise_lr_decay_rates, base_lr, warmup_epochs=0, train_epochs=None):
  """Get a learning rate that decays step-wise as training progresses.
  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    num_epochs_per_decay: number of epochs after which learning rate decays.
    learning_rate_decay_factor: number of epochs after which learning rate decays.
    end_learning_rate: the minimal end learning rate used by a polynomial decay learning rate.
    piecewise_lr_boundary_epochs: A list of ints with strictly increasing entries to reduce the learning rate at certain epochs.
    piecewise_lr_decay_rates: A list of floats that specifies the decay rates for the intervals defined by piecewise_lr_boundary_epochs. It should have one more element than piecewise_lr_boundary_epochs.
    base_lr: initial learning rate scaled based on batch_denom.
    warmup_epochs: Run the number of epoch warmup to the initial lr.
    train_epochs: The number of train epochs
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  tf.logging.debug("learning_rate_decay_type=({})".format(learning_rate_decay_type))
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size
  decay_steps = int(batches_per_epoch * num_epochs_per_decay)

  def learning_rate_fn(global_step):
    warmup_steps = int(batches_per_epoch * warmup_epochs)

    if learning_rate_decay_type == 'exponential':
      lr = tf.train.exponential_decay(initial_learning_rate, global_step - warmup_steps, decay_steps,
                                      learning_rate_decay_factor, staircase=True,
                                      name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
      lr = tf.constant(base_lr, name='fixed_learning_rate')
    elif learning_rate_decay_type == 'polynomial':
      lr = tf.train.polynomial_decay(initial_learning_rate, global_step - warmup_steps, decay_steps,
                                     end_learning_rate, power=1.0,
                                     cycle=False, name='polynomial_decay_learning_rate')
    elif learning_rate_decay_type == 'piecewise':
      boundaries = [int(batches_per_epoch * epoch) for epoch in piecewise_lr_boundary_epochs]
      vals = [initial_learning_rate * float(decay) for decay in piecewise_lr_decay_rates]
      lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    elif learning_rate_decay_type == 'cosine':
      total_batches = int(batches_per_epoch * train_epochs) - warmup_steps
      global_step_except_warmup_step = global_step - warmup_steps
      lr = tf.train.cosine_decay(initial_learning_rate, global_step_except_warmup_step, total_batches)
    else:
      raise NotImplementedError

    if warmup_steps > 0:
      warmup_lr = (initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
      return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

    return lr

  return learning_rate_fn


def get_block_sizes(resnet_size, resnet_version=1):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  # In the case of bl-resnet, the number of blocks in each stage is different from the original.
  if resnet_version == 2:
    choices = {
      50: [3, 4, 6, 3],
      101: [4, 8, 18, 3],
      152: [5, 12, 30, 3],
    }
  else:
    choices = {
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
    }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
      resnet_size, choices.keys()))
    raise ValueError(err)


class Model(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size,
               data_format=None,
               num_classes=None,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE,
               no_downsample=False,
               zero_gamma=False,
               use_se_block=False,
               use_sk_block=False,
               bn_momentum=0.997,
               embedding_size=0,
               anti_alias_filter_size=0,
               anti_alias_type="",
               pool_type='gap',
               loss_type='softmax',
               bl_alpha=2,
               bl_beta=4):

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
    else:
      bottleneck = True

    if resnet_version == 2:
      block_strides = [2, 2, 1, 2]
    else:
      block_strides = [1, 2, 2, 2]

    if no_downsample:
      block_strides[-1] = 1

    super(Model, self).__init__(
      resnet_size=resnet_size,
      bottleneck=bottleneck,
      num_classes=num_classes,
      num_filters=64,
      kernel_size=7,
      conv_stride=2,
      first_pool_size=3,
      first_pool_stride=2,
      block_sizes=get_block_sizes(resnet_size, resnet_version),
      block_strides=block_strides,
      resnet_version=resnet_version,
      data_format=data_format,
      dtype=dtype,
      zero_gamma=zero_gamma,
      use_se_block=use_se_block,
      use_sk_block=use_sk_block,
      bn_momentum=bn_momentum,
      embedding_size=embedding_size,
      anti_alias_filter_size=anti_alias_filter_size,
      anti_alias_type=anti_alias_type,
      pool_type=pool_type,
      bl_alpha=bl_alpha,
      bl_beta=bl_beta,
      loss_type=loss_type,
    )


def model_fn_cls(features, labels, mode, params):
  if int(params['resnet_size']) < 50:
    assert not params['use_dropblock']
    assert not params['use_se_block']
    assert not params['use_sk_block']
    assert not params['use_resnet_d']

  dataset = data_config.get_config(params['dataset_name'])
  learning_rate_fn = learning_rate_with_decay(
    learning_rate_decay_type=params['learning_rate_decay_type'],
    batch_size=params['batch_size'], batch_denom=params['batch_size'],
    num_images=dataset.num_images['train'], num_epochs_per_decay=params['num_epochs_per_decay'],
    learning_rate_decay_factor=params['learning_rate_decay_factor'],
    end_learning_rate=params['end_learning_rate'],
    piecewise_lr_boundary_epochs=params['piecewise_lr_boundary_epochs'],
    piecewise_lr_decay_rates=params['piecewise_lr_decay_rates'],
    base_lr=params['base_learning_rate'],
    train_epochs=params['train_epochs'],
    warmup_epochs=params['lr_warmup_epochs'])

  if params['use_dropblock']:
    starter_kp = params['dropblock_kp'][0]
    end_kp = params['dropblock_kp'][1]
    batches_per_epoch = dataset.num_images['train'] / params['batch_size']
    decay_steps = int(params['train_epochs'] * batches_per_epoch)
    keep_prob_fn = keep_prob_decay(starter_kp, end_kp, decay_steps)
  else:
    keep_prob_fn = None

  return run_loop_classification.resnet_model_fn(
    features=features,
    labels=labels,
    num_classes=dataset.num_classes,
    mode=mode,
    model_class=Model,
    learning_rate_fn=learning_rate_fn,
    keep_prob_fn=keep_prob_fn,
    loss_filter_fn=None,
    p=params)
