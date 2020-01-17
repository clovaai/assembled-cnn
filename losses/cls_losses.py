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

def get_sup_loss(logits, onehot_labels, global_step, num_classes, p):
  cross_entropy = None
  weights = 1.0

  tf.logging.info('[Classfication loss type; {}]'.format(p['cls_loss_type']))
  if p['cls_loss_type'] == 'softmax':
    if not isinstance(weights, float):
      weights = tf.reduce_mean(weights, axis=1)
    cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=onehot_labels,  # 128
      label_smoothing=p['label_smoothing'], weights=weights)
  elif p['cls_loss_type'] == 'sigmoid':
    cross_entropy = weights * tf.nn.sigmoid_cross_entropy_with_logits(
      labels=onehot_labels, logits=logits)
    # Normalize by the total number of positive samples.
    cross_entropy = tf.reduce_sum(cross_entropy) / tf.reduce_sum(onehot_labels)

  assert cross_entropy is not None
  return cross_entropy
