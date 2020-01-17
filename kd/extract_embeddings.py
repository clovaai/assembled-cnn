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

import sys

sys.path.append('./')
sys.path.append('../')

from functions.input_fns import *
import os
from tqdm import tqdm
import pandas as pd

try:
  import urllib2 as urllib
except ImportError:
  import urllib.request as urllib

flags.DEFINE_integer('batch_size', 64, 'The number of samples in each batch.')
flags.DEFINE_integer('total_num_shard', 128, '')
flags.DEFINE_integer('n_tfrecord', 1024, '')
flags.DEFINE_integer('no_shard', 0, '')
flags.DEFINE_boolean('is_training', True, '')
flags.DEFINE_string('data_dir', 'imagenet', 'path to the root directory of images')
flags.DEFINE_string('output_dir', 'amoebanet_logits', 'path to the root directory of images')
flags.DEFINE_string('gpu_to_use', '0', '')
flags.DEFINE_string('data_name', 'imagenet', 'dataset name to use.')
flags.DEFINE_string('net_name', 'amoebanet', '')
flags.DEFINE_integer('offset', 0, '')

FLAGS = flags.FLAGS


def input_fn_cls(is_training,
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
                 dct_method=""):
  tfr_range_low = FLAGS.no_shard * (FLAGS.n_tfrecord / FLAGS.total_num_shard)
  tfr_range_high = (FLAGS.no_shard + 1) * (FLAGS.n_tfrecord / FLAGS.total_num_shard)
  print('tfr_range_high', tfr_range_high)
  print('tfr_range_low', tfr_range_low)
  tf.logging.info('data_dir is {}'.format(data_dir))
  filenames = data_util.get_filenames(FLAGS.is_training, data_dir)

  filenames = [fn for fn in filenames if
               tfr_range_low <= int(fn.split('/')[-1].split("-")[1]) < tfr_range_high]
  print('len(filenames)', len(filenames))
  print('filenames', filenames)
  dataset = data_config.get_config(dataset_name)
  return input_fn(is_training, filenames, use_random_crop, batch_size,
                  dataset.num_train_files, dataset.num_images['train'], dataset.shuffle_buffer,
                  dataset.num_channels, num_epochs, num_gpus, dtype,
                  autoaugment_type=autoaugment_type,
                  with_drawing_bbox=with_drawing_bbox,
                  drop_remainder=drop_remainder,
                  preprocessing_type=preprocessing_type,
                  dct_method=dct_method,
                  return_filename=True)


def main(unused_argv):
  if FLAGS.net_name == 'amoebanet':
    hub_address = "https://tfhub.dev/google/imagenet/amoebanet_a_n18_f448/classification/1"
    preprocessing_type = 'inception_331'
  elif FLAGS.net_name == 'efficientnet':
    hub_address = "https://tfhub.dev/google/efficientnet/b7/classification/1"
    preprocessing_type = 'inception_600'
  else:
    raise NotImplementedError
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_to_use
  dataset = input_fn_cls(False, False, FLAGS.data_dir, FLAGS.batch_size,
                         dataset_name=FLAGS.data_name,
                         preprocessing_type=preprocessing_type)
  iterator = dataset.make_one_shot_iterator()
  images, labels, filename = iterator.get_next()
  print('images, labels, filename', images, labels, filename)

  module = hub.Module(hub_address)

  logits = module(images)  # Logits with shape [batch_size, num_classes].

  # Get the number of images
  num_images = 0
  with tf.Session() as sess:
    while True:
      try:
        _filename = sess.run(filename)
        num_images += _filename.shape[0]
      except tf.errors.OutOfRangeError:
        break

  file2emb_dict = {}
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    n_loop = num_images // (FLAGS.batch_size)
    for _ in tqdm(range(n_loop + 1)):

      try:
        _, _logits, _filename = sess.run([labels, logits, filename])

        for l, fn in zip(_logits, _filename):
          result = np.zeros(l.shape[0] + FLAGS.offset)
          result[FLAGS.offset:] = l
          file2emb_dict[fn.decode("utf-8")] = result

      except tf.errors.OutOfRangeError:
        break

  df = pd.DataFrame(data=file2emb_dict).T
  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  if FLAGS.is_training:
    output_file = os.path.join(FLAGS.output_dir, 'train-{}.csv'.format(str(FLAGS.no_shard)))
  else:
    output_file = os.path.join(FLAGS.output_dir, 'validation-{}.csv'.format(str(FLAGS.no_shard)))

  df.to_csv(output_file, sep=',', header=False)


if __name__ == '__main__':
  tf.app.run()
