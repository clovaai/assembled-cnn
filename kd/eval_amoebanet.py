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

from absl import flags
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm

import sys

sys.path.append('./')
sys.path.append('../')

from functions.input_fns import *
from functions import data_config
from utils import data_util


import os

try:
  import urllib2 as urllib
except ImportError:
  import urllib.request as urllib

"""
Examples:
python kd/eval_amoebanet.py  
"""

flags.DEFINE_integer('batch_size', 32, 'The number of samples in each batch.')
flags.DEFINE_string('data_dir', 'imagenet', 'path to the root directory of images')
flags.DEFINE_string('gpu_to_use', '5', '')
flags.DEFINE_string('data_name', 'imagenet', 'dataset name to use.')
flags.DEFINE_string('preprocessing_type', 'inception_331', 'dataset name to use.')
flags.DEFINE_string('model_dir', 'tmp/tfmodel/', 'Create a flag for specifying the model file directory.')

FLAGS = flags.FLAGS


def input_fn_amoabanet(is_training,
                       use_random_crop,
                       data_dir,
                       batch_size,
                       num_epochs=1,
                       num_gpus=None,
                       dtype=tf.float32,
                       with_drawing_bbox=False,
                       autoaugment_type=None,
                       dataset_name=None,
                       drop_remainder=False,
                       preprocessing_type='imagenet',
                       return_logits=False,
                       dct_method="",
                       train_regex='train*',
                       val_regex='validation*'):
  filenames = data_util.get_filenames(is_training, data_dir,
                                      train_regex=train_regex,
                                      val_regex=val_regex)
  dataset = data_config.get_config(dataset_name)

  return input_fn(is_training, filenames, use_random_crop, batch_size,
                  dataset.num_train_files, dataset.num_images['train'],
                  dataset.shuffle_buffer, dataset.num_channels, num_epochs,
                  num_gpus, dtype,
                  autoaugment_type=autoaugment_type,
                  with_drawing_bbox=with_drawing_bbox,
                  drop_remainder=drop_remainder,
                  preprocessing_type=preprocessing_type,
                  return_logits=return_logits,
                  dct_method=dct_method)

def main(unused_argv):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_to_use
  dconf = data_config.get_config(FLAGS.data_name)
  dataset = input_fn_amoabanet(False, False, FLAGS.data_dir, FLAGS.batch_size,
                               dataset_name=FLAGS.data_name,
                               preprocessing_type=FLAGS.preprocessing_type)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  module = hub.Module("https://tfhub.dev/google/imagenet/amoebanet_a_n18_f448/classification/1")

  logits = module(images)  # Logits with shape [batch_size, num_classes].
  pred = tf.nn.softmax(logits)
  top1 = tf.argmax(logits, axis=1)

  np_preds = np.zeros(dconf.num_images['validation'], dtype=np.int64)
  np_labels = np.zeros(dconf.num_images['validation'], dtype=np.int64)

  np_i = 0
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    n_loop = dconf.num_images['validation'] // FLAGS.batch_size
    for _ in tqdm(range(n_loop + 1)):
      try:
        _pred, _top1, _labels = sess.run([pred, top1, labels])
        np_preds[np_i:np_i + _pred.shape[0]] = _top1
        np_labels[np_i:np_i + _labels.shape[0]] = _labels
        np_i += _pred.shape[0]
      except tf.errors.OutOfRangeError:
        break

  print('Accuracy:')
  print(np.count_nonzero(np_preds == np_labels) / dconf.num_images['validation'])


if __name__ == '__main__':
  tf.app.run()
