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
from official.utils.misc import distribution_utils
from official.utils.flags import core as flags_core
from functions import data_config
from utils import data_util


def get_tf_version():
  tf_version = tf.VERSION
  tf_major_version, tf_minor_version, _ = tf_version.split('.')
  return int(tf_major_version), int(tf_minor_version)

def input_fn(is_training,
             filenames,
             use_random_crop,
             batch_size,
             num_train_files,
             num_images,
             shuffle_buffer,
             num_channels,
             num_epochs=1,
             num_gpus=None,
             dtype=tf.float32,
             autoaugment_type=None,
             with_drawing_bbox=False,
             preprocessing_type='imagenet',
             drop_remainder=False,
             dct_method="",
             return_logits=False,
             return_filename=False,
             parse_record_fn=data_util.parse_record):
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=num_train_files)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  # dataset = dataset.apply(tf.data.experimental.parallel_interleave(
  #   tf.data.TFRecordDataset, cycle_length=20))

  tf_major_version, tf_minor_version = get_tf_version()
  if tf_major_version == 1 and tf_minor_version <= 12:
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=20))
  else:
    dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length=20,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return data_util.process_record_dataset(
    dataset=dataset,
    is_training=is_training,
    batch_size=batch_size,
    shuffle_buffer=shuffle_buffer,
    parse_record_fn=parse_record_fn,
    num_epochs=num_epochs,
    num_gpus=num_gpus,
    num_channels=num_channels,
    examples_per_epoch=num_images if is_training else None,
    dtype=dtype,
    use_random_crop=use_random_crop,
    dct_method=dct_method,
    autoaugment_type=autoaugment_type,
    preprocessing_type=preprocessing_type,
    drop_remainder=drop_remainder,
    with_drawing_bbox=with_drawing_bbox,
    return_logits=return_logits,
    return_filename=return_filename)


def input_fn_cls(is_training, use_random_crop, num_epochs, flags_obj):
  if flags_obj.mixup_type == 1 and is_training:
    batch_size = flags_obj.batch_size * 2
    num_epochs = num_epochs * 2
  else:
    batch_size = flags_obj.batch_size

  batch_size = distribution_utils.per_device_batch_size(batch_size, flags_core.get_num_gpus(flags_obj))
  filenames_sup = data_util.get_filenames(is_training, flags_obj.data_dir,
                                          train_regex=flags_obj.train_regex,
                                          val_regex=flags_obj.val_regex)
  tf.logging.info('The # of Supervised tfrecords: {}'.format(len(filenames_sup)))
  dataset_meta = data_config.get_config(flags_obj.dataset_name)
  datasets = []
  dataset_sup = input_fn(is_training,
                         filenames_sup,
                         use_random_crop,
                         batch_size,
                         dataset_meta.num_train_files,
                         dataset_meta.num_images['train'],
                         dataset_meta.shuffle_buffer,
                         dataset_meta.num_channels,
                         num_epochs,
                         flags_core.get_num_gpus(flags_obj),
                         flags_core.get_tf_dtype(flags_obj),
                         autoaugment_type=flags_obj.autoaugment_type,
                         with_drawing_bbox=flags_obj.with_drawing_bbox,
                         drop_remainder=False,
                         preprocessing_type=flags_obj.preprocessing_type,
                         return_logits=flags_obj.kd_temp > 0,
                         dct_method=flags_obj.dct_method,
                         parse_record_fn=data_util.parse_record_sup)
  datasets.append(dataset_sup)

  def flatten_input(*features):
    images_dict = {}
    for feature in features:
      for key in feature:
        if key == 'label':
          label = feature[key]
        else:
          images_dict[key] = feature[key]
    return images_dict, label

  dataset = tf.data.Dataset.zip(tuple(datasets))
  dataset = dataset.map(flatten_input)
  tf.logging.info('dataset = dataset.map(flatten_input)')
  tf.logging.info(dataset)
  return dataset


def input_fn_ir_eval(is_training,
                     data_dir,
                     batch_size,
                     num_epochs=1,
                     num_gpus=0,
                     dtype=tf.float32,
                     preprocessing_type='imagenet',
                     dataset_name=None,
                     dct_method="",
                     val_regex='validation-*'):
  filenames = data_util.get_filenames(is_training, data_dir, val_regex=val_regex)
  assert len(filenames) > 0
  dataset_config = data_config.get_config(dataset_name)

  return input_fn(is_training, filenames, False, batch_size,
                  dataset_config.num_train_files, dataset_config.num_images['validation'],
                  dataset_config.shuffle_buffer, dataset_config.num_channels, num_epochs, num_gpus, dtype,
                  preprocessing_type=preprocessing_type, dct_method=dct_method)
