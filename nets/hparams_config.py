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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from official.utils.flags import core as flags_core

def define_metric_learning_flags(flags):
  define_common_flags(flags)

  flags.DEFINE_boolean(
    name='return_embedding', default=False,
    help=flags_core.help_wrap('Do not use the last fc.'))

def define_common_flags(flags):
  flags.DEFINE_string(
    name='dataset_name', default=None,
    help=flags_core.help_wrap('imagenet, food100, food101, cub_200_2011'))

  ################################################################################
  # Basic Hyperparameters
  ################################################################################
  flags.DEFINE_integer(
    name='val_batch_size', default=256,
    help=flags_core.help_wrap(
      'The number of steps to warmup learning rate.'))
  flags.DEFINE_string(
    name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
    help=flags_core.help_wrap(
      'If not None initialize all the network except the final layer with '
      'these values'))
  flags.DEFINE_float(
    name='num_epochs_per_decay', short_name='nepd', default=2.0,
    help=flags_core.help_wrap(
      'Number of epochs after which learning rate decays.'))
  flags.DEFINE_float(
    name='learning_rate_decay_factor', short_name='lrdf', default=0.94,
    help=flags_core.help_wrap(
      'Number of epochs after which learning rate decays.'))
  flags.DEFINE_float(
    name='base_learning_rate', short_name='blr', default=0.01,
    help=flags_core.help_wrap(
      'Base learning rate.'))
  flags.DEFINE_float(
    name='end_learning_rate', short_name='elr', default=0.0001,
    help=flags_core.help_wrap(
      'The minimal end learning rate used by a polynomial decay learning rate.'))
  flags.DEFINE_string(
    name='learning_rate_decay_type', short_name='lrdt', default='exponential',
    help=flags_core.help_wrap(
      'Specifies how the learning rate is decayed. One of '
      '"fixed", "exponential", "polynomial", "piecewise", "cosine"'))
  flags.DEFINE_float(
    name='momentum', short_name='mmt', default=0.9,
    help=flags_core.help_wrap(
      'The momentum for the MomentumOptimizer.'))
  flags.DEFINE_float(
    name='bn_momentum', default=0.997,
    help=flags_core.help_wrap(
      'batch normalization momentum.'))
  flags.DEFINE_integer(
    name='embedding_size', default=0,
    help=flags_core.help_wrap('When embedding_size> 0, add embedding between global average pool and fc.'))
  flags.DEFINE_list(
    name='piecewise_lr_boundary_epochs', short_name='pwlrb', default=[30, 60, 80, 90],
    help=flags_core.help_wrap(
      'A list of ints with strictly increasing entries to reduce the learning rate at certain epochs. '
      'It\'s for piecewise lr decay type purpose only.'))
  flags.DEFINE_list(
    name='piecewise_lr_decay_rates', short_name='pwlrd',
    default=[1, 0.1, 0.01, 0.001, 1e-4],
    help=flags_core.help_wrap(
      'A list of floats that specifies the decay rates for the intervals defined by piecewise_lr_boundary_epochs. '
      'It should have one more element than piecewise_lr_boundary_epochs. It\'s for piecewise lr decay type purpose only.'))
  flags.DEFINE_string(
    name='eval_similarity', default='cosine',
    help=flags_core.help_wrap('cosine or euclidean'))

  ################################################################################
  # Loss related
  ################################################################################
  flags.DEFINE_string(
    name='cls_loss_type', default='softmax',
    help=flags_core.help_wrap(
      '`softmax`, `sigmoid`'))

  flags.DEFINE_string(
    name='logit_type', default=None,
    help=flags_core.help_wrap('Logit type `arc_margin` or `None`.'))

  flags.DEFINE_float(
    name='arc_s', default=80.0,
    help=flags_core.help_wrap(
      's of arc-margin loss'))

  flags.DEFINE_float(
    name='arc_m', default=0.15,
    help=flags_core.help_wrap(
      'margin m of arc-margin loss'))

  flags.DEFINE_string(
    name='pool_type', default='gap',
    help=flags_core.help_wrap('`gap` or `gem` or `flatten`'))

  flags.DEFINE_boolean(
    name='no_downsample', default=False,
    help=flags_core.help_wrap('If true, remove downsample in group 4'))

  ################################################################################
  # Data augmentation
  ################################################################################
  flags.DEFINE_string(
    name='preprocessing_type', default='imagenet',
    help=flags_core.help_wrap(' "imagenet" or "cub"'))

  flags.DEFINE_string(
    name='autoaugment_type', default=None,
    help=flags_core.help_wrap(
      'Specifies auto augmentation type. One of "imagenet", "svhn", "cifar", "good"'
      'To use numpy implementation, prefix "np_" to the type.'))

  flags.DEFINE_integer(
    name='mixup_type', short_name='mixup_type', default=0,
    help=flags_core.help_wrap(
      'Use mixup data augmentation. For more details, refer to https://arxiv.org/abs/1710.09412'
      'If type is 0, do not use mixup.'
      'If the type is 1, mix up twice the batch_size to produce batch_size data.'
      'If the type is 2, it mixes up as much as batch_size and produces as much data as batch_size. '
      'Faster than Type 1, but may be less accurate'))

  ################################################################################
  # Network Tweak
  ################################################################################
  flags.DEFINE_boolean(
    name='use_resnet_d', default=False,
    help=flags_core.help_wrap('Use resnet_d architecture. '
                              'For more details, refer to https://arxiv.org/abs/1812.01187'))
  flags.DEFINE_boolean(
    name='use_se_block', default=False,
    help=flags_core.help_wrap('Use SE block. '
                              'For more details, refer to https://arxiv.org/abs/1709.01507'))
  flags.DEFINE_boolean(
    name='use_sk_block', default=False,
    help=flags_core.help_wrap('Use SK block.'))
  flags.DEFINE_integer(
    name='anti_alias_filter_size', default=0,
    help=flags_core.help_wrap('Anti-aliasing filter size, One of 2, 3, 4, 5, 6, 7'))

  flags.DEFINE_string(
    name='anti_alias_type', default="",
    help=flags_core.help_wrap(
      'Specifies auto anti alias type. For example,  "max,proj,sconv" is fully anti-alias, '
      '"sconv" means that only strided conv is applied. '))

  flags.DEFINE_enum(
    name='resnet_version', short_name='rv', default='1',
    enum_values=['1', '2'],
    help=flags_core.help_wrap(
      '1 is original ResNet structure.'
      '2 is BigLittleNet structure.'))

  flags.DEFINE_integer(
    name='bl_alpha', default=2,
    help=flags_core.help_wrap(''))
  flags.DEFINE_integer(
    name='bl_beta', default=4,
    help=flags_core.help_wrap(''))

  ################################################################################
  # Regularization
  ################################################################################
  flags.DEFINE_float(
    name='weight_decay', short_name='wd', default=0.00004,
    help=flags_core.help_wrap(
      'The weight decay on the model weights.'))
  flags.DEFINE_boolean(
    name='use_dropblock', default=False,
    help=flags_core.help_wrap('Use dropblock. '
                              'For more details, refer to https://arxiv.org/abs/1810.12890'))
  flags.DEFINE_list(
    name='dropblock_kp', short_name='drblkp',
    default=[1.0, 0.9],
    help=flags_core.help_wrap(
      'Initial keep_prob and end keep_prob of dropblock.'))
  flags.DEFINE_float(
    name='label_smoothing', short_name='lblsm', default=0.0,
    help=flags_core.help_wrap(
      'If greater than 0 then smooth the labels.'))
  flags.DEFINE_float(
    name='kd_temp', default=0,
    help=flags_core.help_wrap('Use knowledge distillation.'))

  ################################################################################
  # Tricks to Learn the Models
  ################################################################################
  flags.DEFINE_integer(
    name='lr_warmup_epochs', default=0,
    help=flags_core.help_wrap('The number of learning rate warmup epochs. If 0 do not use warmup'))
  flags.DEFINE_boolean(
    name='zero_gamma', default=False,
    help=flags_core.help_wrap(
      'If True, we initialize gamma = 0 for all BN layers that sit at the end of a residual block'))

  ################################################################################
  # Others
  ################################################################################
  flags.DEFINE_string(
    name='dct_method', short_name='dctm', default='INTEGER_ACCURATE',
    help=flags_core.help_wrap(
      'An optional `string`. '
      'Defaults to `""`. '
      'string specifying a hint about the algorithm used for decompression.  '
      'Defaults to "" which maps to a system-specific default.  '
      'Currently valid values are ["INTEGER_FAST", "INTEGER_ACCURATE"].  '
      'The hint may be ignored (e.g., the internal jpeg library changes to a version '
      'that does not have that specific option.)'))
  flags.DEFINE_float(
    name='use_ranking_loss', default=None, help=flags_core.help_wrap(
      'if use_ranking_loss  > 0 use softmax + ranking loss'))
  flags.DEFINE_boolean(
    name='with_drawing_bbox', default=False,
    help=flags_core.help_wrap('If True, display raw images with bounding box in tensorboard'))
  flags.DEFINE_boolean(
    name='use_hyperdash', default=True,
    help=flags_core.help_wrap('Use hyperdash(https://hyperdash.io/) '))
  flags.DEFINE_integer(
    name='num_best_ckpt_to_keep', short_name='nbck', default=3,
    help=flags_core.help_wrap(
      'The number of best performance checkpoint to keep.'))
  flags.DEFINE_boolean(
    name='keep_ckpt_every_eval', default=True,
    help=flags_core.help_wrap('If True, checkpoints are saved for each evaluation.'))
  flags.DEFINE_integer(
    name='keep_checkpoint_max', default=20,
    help=flags_core.help_wrap('keep checkpoint max.'))
  flags.DEFINE_boolean(
    name='eval_only', default=False,
    help=flags_core.help_wrap('Skip training and only perform evaluation on '
                              'the latest checkpoint.'))
  flags.DEFINE_boolean(
    name='training_random_crop', default=True,
    help=flags_core.help_wrap('Whether to randomly crop training images'))
  flags.DEFINE_boolean(
    name='export_only', default=False,
    help=flags_core.help_wrap('Skip training and evaluations.'
                              'Only export the latest checkpoint.'))
  flags.DEFINE_string(
    name='export_decoder_type', default='jpeg',
    help=flags_core.help_wrap('Specify the image decoder for binary input pb (jpeg | webp)'))
  flags.DEFINE_float(
    name='save_checkpoints_epochs', default=1.0,
    help=flags_core.help_wrap('Save checkpoint every save_checkpoints_epochs'))
  flags.DEFINE_float(
    name='ratio_fine_eval', default=1.0,
    help=flags_core.help_wrap('From `train_epochs` *` ratio_fine_eval`, '
                              'it evaluates every 1 epoch.'))
  flags.DEFINE_boolean(
    name='zeroshot_eval', default=False,
    help=flags_core.help_wrap('zeroshot evaluation'))
  flags.DEFINE_string(
    name='label_file', default=None,
    help=flags_core.help_wrap('use label ids which has been generated by building tfrecord. '
                              'If None, it put the label ids in alphabetical order.'))
  flags.DEFINE_string(
    name='train_regex', default='train-*',
    help=flags_core.help_wrap('use label ids which has been generated by building tfrecord. '
                              'If None, it put the label ids in alphabetical order.'))
  flags.DEFINE_string(
    name='val_regex', default='validation-*',
    help=flags_core.help_wrap('use label ids which has been generated by building tfrecord. '
                              'If None, it put the label ids in alphabetical order.'))
  flags.DEFINE_list(
    name='recall_at_k',
    default=[1, 5],
    help=flags_core.help_wrap(
      'A list of int that specifies the k values of the recall_at_k'))
  
