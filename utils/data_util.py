#/usr/bin/env python
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

import os

import six
import re
import tensorflow as tf

from preprocessing import imagenet_preprocessing
from preprocessing import inception_preprocessing
from preprocessing import reid_preprocessing

def get_tf_version():
  tf_version = tf.VERSION
  tf_major_version, tf_minor_version, _ = tf_version.split('.')
  return int(tf_major_version), int(tf_minor_version)

def int64_feature(values):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(values, six.text_type):
    values = six.binary_type(values, encoding='utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Wrapper for inserting floats features into Example proto."""
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def convert_to_example(image_data, image_format, class_id, height, width, bbox=None):
  assert height > 0
  assert width > 0
  (xmin, ymin, xmax, ymax) = ([], [], [], [])
  bbox = [] if bbox is None else bbox
  for b in bbox:
    assert len(b) == 4
    [l.append(p) for l, p in zip([xmin, ymin, xmax, ymax], b)]

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/encoded': bytes_feature(image_data),
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/class/label': int64_feature(class_id),
    'image/format': bytes_feature(image_format),
    'image/object/bbox/xmin': float_feature(xmin),
    'image/object/bbox/xmax': float_feature(xmax),
    'image/object/bbox/ymin': float_feature(ymin),
    'image/object/bbox/ymax': float_feature(ymax)
  }))
  return example


def convert_to_example_without_bbox(image_data, image_format, class_id, height, width):
  assert height > 0
  assert width > 0
  example = tf.train.Example(features=tf.train.Features(feature={
    'image/encoded': bytes_feature(image_data),
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/class/label': int64_feature(class_id),
    'image/format': bytes_feature(image_format),
  }))
  return example


def mixup(x, y, alpha=0.2, keep_batch_size=True, y_t=None):
  dist = tf.contrib.distributions.Beta(alpha, alpha)

  _, h, w, c = x.get_shape().as_list()

  batch_size = tf.shape(x)[0]
  num_class = y.get_shape().as_list()[1]

  lam1 = dist.sample([batch_size // 2])

  if x.dtype == tf.float16:
    lam1 = tf.cast(lam1, dtype=tf.float16)
    y = tf.cast(y, dtype=tf.float16)
    if y_t is not None:
      y_t = tf.cast(y_t, dtype=tf.float16)

  x1, x2 = tf.split(x, 2, axis=0)
  y1, y2 = tf.split(y, 2, axis=0)

  lam1_x = tf.tile(tf.reshape(lam1, [batch_size // 2, 1, 1, 1]), [1, h, w, c])
  lam1_y = tf.tile(tf.reshape(lam1, [batch_size // 2, 1]), [1, num_class])

  mixed_sx1 = lam1_x * x1 + (1. - lam1_x) * x2
  mixed_sy1 = lam1_y * y1 + (1. - lam1_y) * y2
  mixed_sx1 = tf.stop_gradient(mixed_sx1)
  mixed_sy1 = tf.stop_gradient(mixed_sy1)

  if y_t is not None:
    y1_t, y2_t = tf.split(y_t, 2, axis=0)
    mixed_sy1_t = lam1_y * y1_t + (1. - lam1_y) * y2_t
    mixed_sy1_t = tf.stop_gradient(mixed_sy1_t)
  else:
    mixed_sy1_t = None

  if keep_batch_size:
    lam2 = dist.sample([batch_size // 2])

    if x.dtype == tf.float16:
      lam2 = tf.cast(lam2, dtype=tf.float16)

    lam2_x = tf.tile(tf.reshape(lam2, [batch_size // 2, 1, 1, 1]), [1, h, w, c])
    lam2_y = tf.tile(tf.reshape(lam2, [batch_size // 2, 1]), [1, num_class])

    x3 = tf.reverse(x2, [0])
    y3 = tf.reverse(y2, [0])

    mixed_sx2 = lam2_x * x1 + (1. - lam2_x) * x3
    mixed_sy2 = lam2_y * y1 + (1. - lam2_y) * y3

    mixed_sx2 = tf.stop_gradient(mixed_sx2)
    mixed_sy2 = tf.stop_gradient(mixed_sy2)

    mixed_sx1 = tf.concat([mixed_sx1, mixed_sx2], axis=0)
    mixed_sy1 = tf.concat([mixed_sy1, mixed_sy2], axis=0)

    if y_t is not None:
      y3_t = tf.reverse(y2_t, [0])
      mixed_sy2_t = lam2_y * y1 + (1. - lam2_y) * y3_t
      mixed_sy2_t = tf.stop_gradient(mixed_sy2_t)
      mixed_sy1_t = tf.concat([mixed_sy1_t, mixed_sy2_t], axis=0)

  return mixed_sx1, mixed_sy1, mixed_sy1_t


def get_filenames(is_training, data_dir, train_regex='train-*', val_regex='validation-*'):
  """Return filenames for dataset."""
  if is_training:
    path = os.path.join(data_dir, train_regex)
    matching_files = tf.gfile.Glob(path)
    return matching_files
  else:
    path = os.path.join(data_dir, val_regex)
    matching_files = tf.gfile.Glob(path)
    return matching_files


def parse_example_proto(example_serialized, ret_dict=False):
  # Dense features in Example proto.
  feature_map = {
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
    'image/logit': tf.VarLenFeature(dtype=tf.float32)
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
    {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                 'image/object/bbox/ymin',
                                 'image/object/bbox/xmax',
                                 'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  if ret_dict:
    return features
  else:
    return features['image/encoded'], label, bbox, features['image/filename'], features['image/logit'].values

def parse_record_sup(raw_record, is_training, num_channels, dtype,
                     use_random_crop=True, image_size=224,
                     autoaugment_type=None, with_drawing_bbox=False,
                     dct_method="", preprocessing_type='imagenet',
                     return_logits=False, num_classes=1001, return_filename=False):
  features = parse_example_proto(raw_record, ret_dict=True)
  results = {}
  sup_image = preprocess_image(image_buffer=features['image/encoded'],
                               is_training=is_training,
                               num_channels=num_channels,
                               dtype=dtype,
                               use_random_crop=use_random_crop,
                               image_size=image_size,
                               autoaugment_type=autoaugment_type,
                               dct_method=dct_method,
                               preprocessing_type=preprocessing_type)

  results['image'] = sup_image
  if return_logits:
    label = tf.one_hot(tf.cast(features['image/class/label'], dtype=tf.int32), num_classes)
    logit = tf.reshape(features['image/logit'].values, [num_classes])
    label = tf.concat([label, logit], axis=0)
  else:
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
  results['label'] = label
  return results

def parse_record(raw_record, is_training, num_channels, dtype,
                 use_random_crop=True, image_size=224,
                 autoaugment_type=None, with_drawing_bbox=False,
                 dct_method="", preprocessing_type='imagenet',
                 return_logits=False, num_classes=1001, return_filename=False):

  image_buffer, label, bbox, filename, logit = parse_example_proto(raw_record)
  if return_logits:
    assert num_classes == 1001, 'Only support ImageNet for Knowledge Distillation yet'
    label = tf.one_hot(label, num_classes)
    logit = tf.reshape(logit, [num_classes])
    label = tf.concat([label, logit], axis=0)

  image = preprocess_image(image_buffer=image_buffer,
                           is_training=is_training,
                           num_channels=num_channels,
                           dtype=dtype,
                           use_random_crop=use_random_crop,
                           image_size=image_size,
                           bbox=bbox,
                           autoaugment_type=autoaugment_type,
                           with_drawing_bbox=with_drawing_bbox,
                           dct_method=dct_method,
                           preprocessing_type=preprocessing_type)

  if return_filename:
    return image, label, filename
  else:
    return image, label

def preprocess_image(image_buffer, is_training, num_channels, dtype,
                     use_random_crop=True, image_size=224, bbox=None,
                     autoaugment_type=None, with_drawing_bbox=False,
                     dct_method="", decoder_name='jpeg',
                     preprocessing_type='imagenet'):
  raw_image_with_bbox = None
  bbox = tf.zeros([0, 0, 4], tf.float32) if bbox is None else bbox
  crop_type_check = re.compile('imagenet_[0-9]{3}')
  crop_type_check_a = re.compile('imagenet_[0-9]{3}a')
  if preprocessing_type == 'imagenet':
    image, raw_image_with_bbox = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=image_size,
      output_width=image_size,
      num_channels=num_channels,
      is_training=is_training,
      use_random_crop=use_random_crop,
      autoaugment_type=autoaugment_type,
      dct_method=dct_method,
      with_drawing_bbox=with_drawing_bbox)
  elif preprocessing_type == 'imagenet_224_256':
    # 224로 학습하고 256 해상도로 평가 type1
    image, raw_image_with_bbox = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=224 if is_training else 256,
      output_width=224 if is_training else 256,
      num_channels=num_channels,
      is_training=is_training,
      use_random_crop=use_random_crop,
      autoaugment_type=autoaugment_type,
      with_drawing_bbox=with_drawing_bbox,
      dct_method=dct_method,
      crop_type=0,
    )
  elif preprocessing_type == 'imagenet_224_256a':
    # 224로 학습하고 256 해상도로 평가 type2
    image, raw_image_with_bbox = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=224 if is_training else 256,
      output_width=224 if is_training else 256,
      num_channels=num_channels,
      is_training=is_training,
      use_random_crop=use_random_crop,
      autoaugment_type=autoaugment_type,
      with_drawing_bbox=with_drawing_bbox,
      dct_method=dct_method,
      crop_type=1,
    )
  elif crop_type_check_a.match(preprocessing_type):
    image_size = int(preprocessing_type.split("_")[1][0:3])
    image, raw_image_with_bbox = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=image_size,
      output_width=image_size,
      num_channels=num_channels,
      is_training=is_training,
      use_random_crop=use_random_crop,
      autoaugment_type=autoaugment_type,
      with_drawing_bbox=with_drawing_bbox,
      dct_method=dct_method,
      crop_type=1)
  elif crop_type_check.match(preprocessing_type):
    image_size = int(preprocessing_type.split("_")[1])
    image, raw_image_with_bbox = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=image_size,
      output_width=image_size,
      num_channels=num_channels,
      is_training=is_training,
      use_random_crop=use_random_crop,
      autoaugment_type=autoaugment_type,
      with_drawing_bbox=with_drawing_bbox,
      dct_method=dct_method,
      crop_type=0)
  elif preprocessing_type == 'reid':
    image = reid_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      output_height=image_size,
      output_width=image_size,
      num_channels=num_channels,
      is_training=is_training,
      dct_method=dct_method,
      autoaugment_type=autoaugment_type)
  elif preprocessing_type == 'reid_224':
    image = reid_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      output_height=image_size,
      output_width=image_size,
      num_channels=num_channels,
      is_training=is_training,
      autoaugment_type=autoaugment_type,
      dct_method=dct_method,
      eval_large_resolution=False)
  elif preprocessing_type == 'inception_331':
    image = inception_preprocessing.preprocess_image(
      image=image_buffer,
      output_height=331,
      output_width=331,
      scaled_images=False,
      is_training=is_training)
  elif preprocessing_type == 'inception_600':
    image = inception_preprocessing.preprocess_image(
      image=image_buffer,
      output_height=600,
      output_width=600,
      scaled_images=False,
      is_training=is_training)
  else:
    raise NotImplementedError

  image = tf.cast(image, dtype)
  if with_drawing_bbox and not raw_image_with_bbox:
    image = tf.stack([image, raw_image_with_bbox])

  return image


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer, num_channels, parse_record_fn, num_epochs,
                           num_gpus, examples_per_epoch, dtype, use_random_crop=False, with_drawing_bbox=False,
                           autoaugment_type=None, dct_method='', preprocessing_type='imagenet', drop_remainder=False,
                           return_logits=False, return_filename=False):

  # Disable intra-op parallelism to optimize for throughput instead of latency.
  tf_major_version, tf_minor_version = get_tf_version()
  if tf_major_version == 1 and tf_minor_version <= 12:
    pass
  else:
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)

  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  if is_training and num_gpus and examples_per_epoch:
    total_examples = num_epochs * examples_per_epoch
    # Force the number of batches to be divisible by the number of devices.
    # This prevents some devices from receiving batches while others do not,
    # which can lead to a lockup. This case will soon be handled directly by
    # distribution strategies, at which point this .take() operation will no
    # longer be needed.
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    dataset.take(total_batches * batch_size)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  if tf_major_version == 1 and tf_minor_version <= 12:
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        lambda value: parse_record_fn(value,
                                      is_training,
                                      num_channels,
                                      dtype,
                                      dct_method=dct_method,
                                      use_random_crop=use_random_crop,
                                      autoaugment_type=autoaugment_type,
                                      preprocessing_type=preprocessing_type,
                                      with_drawing_bbox=with_drawing_bbox,
                                      return_logits=return_logits,
                                      return_filename=return_filename),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=drop_remainder))
  else:
    dataset = dataset.map(
      lambda value: parse_record_fn(value,
                                    is_training,
                                    num_channels,
                                    dtype,
                                    dct_method=dct_method,
                                    use_random_crop=use_random_crop,
                                    autoaugment_type=autoaugment_type,
                                    preprocessing_type=preprocessing_type,
                                    with_drawing_bbox=with_drawing_bbox,
                                    return_logits=return_logits,
                                    return_filename=return_filename),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  if tf_major_version == 1 and tf_minor_version <= 12:
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  else:
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return dataset
