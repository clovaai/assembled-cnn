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

def get_train_op(loss,
                 global_step,
                 learning_rate,
                 momentum,
                 loss_scale,
                 ):
  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
  scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)
  if loss_scale != 1:
    scaled_grad_vars = [(grad / loss_scale, var)
                        for grad, var in scaled_grad_vars]
  minimize_op = optimizer.apply_gradients(scaled_grad_vars, global_step)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  train_op = tf.group(minimize_op, update_ops)
  return train_op
