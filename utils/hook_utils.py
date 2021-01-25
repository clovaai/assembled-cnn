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

import tensorflow as tf

try:
  from functions import model_fns
except:
  from functions import *
from utils import log_utils

class WarmStartHook(tf.train.SessionRunHook):
  def __init__(self, checkpoint_path):
    self.checkpoint_path = checkpoint_path
    self.initialized = False
    self.var_list_warm_start = []
    self.saver = None

  def begin(self):
    var_list_all = tf.contrib.framework.get_trainable_variables()

    for v in var_list_all:
      # Dense layers in se_block is included in warm-start variables.
      if 'dense' in v.name and not 'se_block' in v.name:
        continue
      else:
        self.var_list_warm_start.append(v)

    if self.checkpoint_path is not None and tf.gfile.IsDirectory(self.checkpoint_path):
      self.checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)

    self.saver = tf.train.Saver(self.var_list_warm_start)

  def after_create_session(self, session, coord=None):
    tf.logging.info('Session created.')
    if self.checkpoint_path and session.run(tf.train.get_or_create_global_step()) == 0:
      log_utils.log_var_list_by_line(self.var_list_warm_start, 'var_list_warm_start')
      tf.logging.info('Fine-tuning from %s' % self.checkpoint_path)
      self.saver.restore(session, self.checkpoint_path)

