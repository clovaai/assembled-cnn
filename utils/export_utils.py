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

import functools
import os

import tensorflow as tf

from official.utils.export import export
from utils import data_util
from functions import data_config
import numpy as np
from tqdm import tqdm


def export_test(bin_export_path, flags_obj, ir_eval):
  ds = tf.data.Dataset.list_files(flags_obj.data_dir + '/' + flags_obj.val_regex)
  ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=10)

  def parse_tfr(example_proto):
    feature_def = {'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                   'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')}
    features = tf.io.parse_single_example(serialized=example_proto, features=feature_def)
    return features['image/encoded'], features['image/class/label']

  ds = ds.map(parse_tfr)
  ds = ds.batch(flags_obj.val_batch_size)
  iterator = ds.make_one_shot_iterator()
  images, labels = iterator.get_next()
  dconf = data_config.get_config(flags_obj.dataset_name)
  num_val_images = dconf.num_images['validation']
  if flags_obj.zeroshot_eval or ir_eval:
    feature_dim = flags_obj.embedding_size if flags_obj.embedding_size > 0 else flags_obj.num_features
    np_features = np.zeros((num_val_images, feature_dim), dtype=np.float32)
    np_labels = np.zeros(num_val_images, dtype=np.int64)
    np_i = 0
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      tf.saved_model.load(sess=sess, export_dir=bin_export_path, tags={"serve"})
      for _ in tqdm(range(int(num_val_images / flags_obj.val_batch_size) + 1)):
        try:
          np_image, np_label = sess.run([images, labels])
          np_predict = sess.run('embedding_tensor:0',
                                feed_dict={'input_tensor:0': np_image})
          np_features[np_i:np_i + np_predict.shape[0], :] = np_predict
          np_labels[np_i:np_i + np_label.shape[0]] = np_label
          np_i += np_predict.shape[0]

        except tf.errors.OutOfRangeError:
          break
      assert np_i == num_val_images

    from sklearn.preprocessing import normalize

    x = normalize(np_features)
    np_sim = x.dot(x.T)
    np.fill_diagonal(np_sim, -10)  # removing similarity for query.
    num_correct = 0
    for i in range(num_val_images):
      cur_label = np_labels[i]
      rank1_label = np_labels[np.argmax(np_sim[i, :])]
      if rank1_label == cur_label:
        num_correct += 1
    recall_at_1 = num_correct / num_val_images
    metric = recall_at_1
  else:
    np_i = 0
    correct_cnt = 0
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      tf.saved_model.load(sess=sess, export_dir=bin_export_path, tags={"serve"})
      for _ in tqdm(range(int(num_val_images / flags_obj.val_batch_size) + 1)):
        try:
          np_image, np_label = sess.run([images, labels])
          np_predict = sess.run('ArgMax:0',
                                feed_dict={'input_tensor:0': np_image})
          np_i += np_predict.shape[0]
          correct_cnt += np.sum(np_predict == np_label)
        except tf.errors.OutOfRangeError:
          break
      assert np_i == num_val_images

      metric = correct_cnt / np_i
  return metric

def image_bytes_serving_input_fn(image_shape, decoder_name, dtype=tf.float32, pptype='imagenet'):
  """Serving input fn for raw jpeg images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    # Bounding box around the whole image.
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=dtype, shape=[1, 1, 4])
    _, _, num_channels = image_shape
    tf.logging.info("!!!!!!!!!! Preprocessing type for exporting pb: {} and decoder type: {}".format(pptype, decoder_name))
    image = data_util.preprocess_image(
      image_buffer=image_bytes, is_training=False, bbox=bbox,
      num_channels=num_channels, dtype=dtype, use_random_crop=False,
      decoder_name=decoder_name, dct_method='INTEGER_ACCURATE', preprocessing_type=pptype)
    return image

  image_bytes_list = tf.placeholder(
    shape=[None], dtype=tf.string, name='input_tensor')
  images = tf.map_fn(
    _preprocess_image, image_bytes_list, back_prop=False, dtype=dtype)
  return tf.estimator.export.TensorServingInputReceiver(
    images, {'image_bytes': image_bytes_list})


def export_pb(flags_core, flags_obj, shape, classifier, ir_eval=False):
  export_dtype = flags_core.get_tf_dtype(flags_obj)

  if not flags_obj.data_format:
    raise ValueError('The `data_format` must be specified: channels_first or channels_last ')

  bin_export_path = os.path.join(flags_obj.export_dir, flags_obj.data_format, 'binary_input')
  bin_input_receiver_fn = functools.partial(image_bytes_serving_input_fn, shape, flags_obj.export_decoder_type,
                                            dtype=export_dtype, pptype=flags_obj.preprocessing_type)

  pp_export_path = os.path.join(flags_obj.export_dir, flags_obj.data_format, 'preprocessed_input')
  pp_input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
    shape, batch_size=None, dtype=export_dtype)

  result_bin_export_path = classifier.export_savedmodel(bin_export_path, bin_input_receiver_fn)
  classifier.export_savedmodel(pp_export_path, pp_input_receiver_fn)

  if flags_obj.export_decoder_type == 'jpeg':
    metric = export_test(result_bin_export_path, flags_obj, ir_eval)
    msg = 'IMPOTANT! Evaluation metric of exported saved_model.pb is {}'.format(metric)
    tf.logging.info(msg)
    with tf.gfile.Open(result_bin_export_path.decode("utf-8") + '/model_performance.txt', 'w') as fp:
      fp.write(msg)
