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

import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from official.utils.misc import distribution_utils
import json
import os

def _monkey_patch_org_assert_broadcastable():
  """Monkey-patch `assert_broadcast` op to avoid OOM when enabling XLA."""
  def no_op_assert_broadcastable(weights, values):
    del weights, values
    tf.logging.info(
        'Using monkey-patched version of assert_broadcastable op, which always '
        'returns an no_op. It should be removed after XLA OOM issue is fixed.')
    return tf.constant([], dtype=tf.float32)

  from tensorflow.python.ops import weights_broadcast_ops  # pylint: disable=g-import-not-at-top
  if not hasattr(weights_broadcast_ops, 'org_assert_broadcastable'):
    weights_broadcast_ops.org_assert_broadcastable = (
        weights_broadcast_ops.assert_broadcastable)
  weights_broadcast_ops.assert_broadcastable = no_op_assert_broadcastable


def get_session_config(flags_obj):
  """Return config proto according to flag settings, or None to use default."""
  config = tf.ConfigProto(
    inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
    intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
    allow_soft_placement=True)
  return config

def get_run_config(flags_obj, flags_core, session_config, num_images_train):
  distribution_strategy = distribution_utils.get_distribution_strategy(
    flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

  steps_per_epoch = flags_obj.save_checkpoints_epochs \
                    * int(num_images_train // int(flags_obj.batch_size))

  run_config = tf.estimator.RunConfig(
    train_distribute=distribution_strategy, session_config=session_config,
    keep_checkpoint_max=flags_obj.keep_checkpoint_max,
    save_checkpoints_steps=int(steps_per_epoch),
    save_checkpoints_secs=None,
  )
  return run_config


def get_epoch_schedule(flags_obj, schedule, num_images):
  if tf.gfile.Exists(flags_obj.model_dir):
    ckpt = tf.train.latest_checkpoint(flags_obj.model_dir)
    if not ckpt:
      cur_epoch = 0
    else:
      global_steps = int(ckpt.split("-")[-1])
      cur_epoch = global_steps // int(num_images['train'] / flags_obj.batch_size)
  else:
    cur_epoch = 0

  accumulated_epoch = 0
  fine_eval_start_epoch = int(flags_obj.train_epochs * flags_obj.ratio_fine_eval)
  # print('fine_eval_start_epoch', fine_eval_start_epoch)
  # print('cur_epoch', cur_epoch)
  new_schedule = []
  for num_train_epochs in schedule:
    #   print(num_train_epochs)
    accumulated_epoch += num_train_epochs
    if accumulated_epoch <= cur_epoch:
      continue
    if accumulated_epoch > fine_eval_start_epoch:
      for i in range(num_train_epochs):
        new_schedule.append(1)
    else:
      new_schedule.append(num_train_epochs)
  return new_schedule


def dump_hparam():
  flags_dict = tf.app.flags.FLAGS.flag_values_dict()
  tf.logging.info(flags_dict['model_dir'])
  tf.gfile.MakeDirs(flags_dict['model_dir'])
  with tf.gfile.Open(os.path.join(flags_dict['model_dir'], "hparams.json"), "w") as out:
    json.dump(flags_dict, out)
