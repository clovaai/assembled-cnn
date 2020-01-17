#!/usr/bin/env python
# coding=utf8
"""
Stanford Online Product 데이터 집합을 TFRecord 파일 포맷의 Example 객체로 변환한다.

Example:
python datasets/build_sop.py \
--input_dir=/home1/irteam/user/jklee/dataset/Stanford_Online_Products/Stanford_Online_Products \
--output_dir=sop_tfrecord
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys

sys.path.append('./')
sys.path.append('../')

from datetime import datetime

import numpy as np
import tensorflow as tf

from datasets import dataset_utils
from datasets import image_coder as coder
from utils import data_util

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_dir', type=str, default=None, help='Input data directory.')
parser.add_argument('-o', '--output_dir', type=str, default=None, help='Output data directory.')
parser.add_argument('--train_shards', type=int, default=128, help='Number of shards in training TFRecord files.')
parser.add_argument('--validation_shards', type=int, default=16, help='Number of shards in validation TFRecord files.')
parser.add_argument('--num_threads', type=int, default=8, help='Number of threads to preprocess the images.')


def _process_image(filename):
  """
  이미지 파일을 읽어들여 RGB 타입으로 변환하여 반환한다.
  :param filename: string, 읽어들일 이미지 파일 경로.
  :return:
    image_data: 이미지 데이터로 jpg 포맷의 데이터.
    height: 이미지 height
    width: 이미지 width
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()
    try:
      image = coder.decode_jpg(image_data)
      height = image.shape[0]
      width = image.shape[1]
    except tf.errors.InvalidArgumentError:
      raise ValueError("Invalid decode in {}".format(filename))
    return image_data, height, width


def _process_image_files_batch(thread_index, offsets, output_filenames, filenames, labels):
  """
  하나의 스레드 단위에서 이미지 리스트를 읽어 TRRecord 타입으로 변환하는 함수
  :param thread_index: 현재 작업중인 thread 번호.
  :param offsets: offset list. 이미지 목록 중 현재 스레드에서 처리해야 할 offset 값으로 shard 갯수만큼 리스트로 제공
  :param output_filenames: 출력 파일 이름으로 shard 갯수만큼 리스트로 제공.
  :param filenames: 처리해야 할 전체 이미지 파일 리스트
  :param labels: 처리해야 할 전체 이미지 레이블 리스트
  """
  assert len(offsets) == len(output_filenames)
  assert len(filenames) == len(labels)

  num_files_in_thread = offsets[-1][1] - offsets[0][0]
  counter = 0
  # 하나의 thread 에는 여러 개의 shard 가 할당될 수 있다.
  for offset, output_filename in zip(offsets, output_filenames):
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    # offset 에는 현재 shard 에 대한 (start, end) offset이 저장되어 있음.
    files_in_shard = np.arange(offset[0], offset[1], dtype=int)
    shard_counter = 0
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]

      try:
        image_data, height, width = _process_image(filename)
      except ValueError:
        dataset_utils.log('[thread %2d]: Invalid image found. %s - [skip].' % (thread_index, filename))
        continue

      example = data_util.convert_to_example_without_bbox(image_data, 'jpg', label, height, width)
      writer.write(example.SerializeToString())

      counter += 1
      shard_counter += 1
      if not counter % 1000:
        dataset_utils.log('%s [thread %2d]: Processed %d of %d images in thread batch.' %
                          (datetime.now(), thread_index, counter, num_files_in_thread))

    writer.close()
    dataset_utils.log('%s [thread %2d]: Wrote %d images to %s' %
                      (datetime.now(), thread_index, shard_counter, output_file))


def _process_dataset(name, filenames, labels, num_shards):
  """
  이미지 파일 목록을 읽어들여 TFRecord 객체로 변환하는 함수
  :param name: string, 데이터 고유 문자열 (train, validation 등)
  :param filenames: list of strings; 이미지 파일 경로 리스트.
  :param labels: list of integer; 이미지에 대한 정수화된 정답 레이블 리스트
  :param num_shards: 데이터 집합을 샤딩할 갯수.
  """
  assert len(filenames) == len(labels)

  shard_offsets = dataset_utils.make_shard_offsets(len(filenames), FLAGS.num_threads, num_shards)
  shard_output_filenames = dataset_utils.make_shard_filenames(name, len(filenames), FLAGS.num_threads, num_shards)

  def _process_batch(thread_index):
    offsets = shard_offsets[thread_index]
    output_filenames = shard_output_filenames[thread_index]
    _process_image_files_batch(thread_index, offsets, output_filenames, filenames, labels)

  dataset_utils.thread_execute(FLAGS.num_threads, _process_batch)
  dataset_utils.log('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))


def get_filenames_and_labels(data_dir, is_training=True):
  if is_training:
    txt = data_dir + '/Ebay_train.txt'
  else:
    txt = data_dir + '/Ebay_test.txt'

  with open(txt, 'r') as f:
    filenames = []
    labels = []
    for line in f:
      token = line.strip().split()
      if token[0] == 'image_id':
        continue
      class_id = int(token[1]) - 1

      path = token[3]
      path = os.path.join(data_dir, path)
      filenames.append(path)
      labels.append(class_id)

  if is_training:
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

  return filenames, labels


def main(_):
  if (FLAGS.input_dir is None) or (FLAGS.output_dir is None):
    parser.print_help()
    return

  assert not FLAGS.train_shards % FLAGS.num_threads, (
    'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
    'Please make the FLAGS.num_threads commensurate with FLAGS.validation_shards')
  print('Saving results to {}'.format(FLAGS.output_dir))

  dataset_utils.log('Make UEC-food100 TFRecord dataset.')

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  # train_txt = '/home1/irteam/user/jklee/dataset/Stanford_Online_Products/Stanford_Online_Products/Ebay_train.txt'
  # test_txt = '/home1/irteam/user/jklee/dataset/Stanford_Online_Products/Stanford_Online_Products/Ebay_test.txt'

  train_filenames, train_labels = get_filenames_and_labels(FLAGS.input_dir)
  validation_filenames, validation_labels = get_filenames_and_labels(FLAGS.input_dir, is_training=False)

  # train_filenames, train_labels = _get_filenames_and_labels(
  #   FLAGS.data_dir, train_path, label_path, shuffle=True)
  #
  # validation_filenames, validation_labels = _get_filenames_and_labels(
  #   FLAGS.data_dir, validation_path, label_path, shuffle=False)

  dataset_utils.log('Convert [train] dataset.')
  _process_dataset('train', train_filenames, train_labels, FLAGS.train_shards)

  dataset_utils.log('Convert [validation] dataset.')
  _process_dataset('validation', validation_filenames, validation_labels, FLAGS.validation_shards)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
