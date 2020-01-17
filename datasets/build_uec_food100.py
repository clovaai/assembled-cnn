#!/usr/bin/env python
# coding=utf8
"""
UEC-food100 데이터 집합을 TFRecord 파일 포맷의 Example 객체로 변환한다.
데이터는 `UECFOOD100/{label_no}/11500.jpg` 와 같은 형식으로 저장되어 있다.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

from datasets import dataset_utils
from datasets import image_coder as coder
from utils import data_util

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_UEC_FOOD100_ROOT_DIR = 'UECFOOD100'
_NUM_CLASSES = 100

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_dir', type=str, default=None, help='Input data directory.')
parser.add_argument('-o', '--output_dir', type=str, default=None, help='Output data directory.')
parser.add_argument('--train_shards', type=int, default=64, help='Number of shards in training TFRecord files.')
parser.add_argument('--validation_shards', type=int, default=16, help='Number of shards in validation TFRecord files.')
parser.add_argument('--num_threads', type=int, default=16, help='Number of threads to preprocess the images.')

parser.add_argument(
  '--use_bbox', type=dataset_utils.str2bool, nargs='?', const=True, default=False,
  help='Whether to use bounding boxes or not.')


def _get_bbox_info(input_dir):
  """
  bb_info.txt 파일을 읽어들여 각 이미지의 대한 bbox 정보를 읽어들인다.
  :param input_dir: 입력 파일 디렉토리.
  :return: double dict 타입의 객체로 bbox 정보를 가지고 있다. {label_id, {file_id, [x1, y1, x2, y2]}}
  """
  bbox_info = dict()
  for label_id in range(0, _NUM_CLASSES):
    label_dir = str(label_id + 1)  # from one-base index to zero-base
    box_file = os.path.join(input_dir, _UEC_FOOD100_ROOT_DIR, label_dir, 'bb_info.txt')
    bbox_info_by_label = dict()
    with tf.gfile.GFile(box_file, 'r') as f:
      next(f)  # skip first line (column info : 'img x1 y1 x2 y2')
      for line in f:
        sp = line.rstrip('\n').split(' ')
        if len(sp) == 0:
          continue
        file_id, bbox_location = sp[0], [int(i) for i in sp[1:]]
        assert len(bbox_location) == 4
        bbox_info_by_label[file_id] = bbox_location
      bbox_info[label_id] = bbox_info_by_label
  return bbox_info


def _get_filenames_and_labels(val_index, shuffle=True):
  """이미지 파일 이름과 이에 대한 레이블 정보를 반환한다."""
  output_filenames = []
  output_labels = []

  # make filenames
  for i in val_index:
    target_file = os.path.join(FLAGS.data_dir, 'val{}.txt'.format(str(i)))
    with open(target_file, 'r') as f:
      image_files = [os.path.join(FLAGS.data_dir, line.strip()) for line in f]
      output_filenames.extend(image_files)

  # make labels
  for filename in output_filenames:
    label = int(os.path.basename(os.path.dirname(filename))) - 1
    output_labels.append(label)

  if shuffle:
    shuffled_index = list(range(len(output_filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    output_filenames = [output_filenames[i] for i in shuffled_index]
    output_labels = [output_labels[i] for i in shuffled_index]

  return output_filenames, output_labels


def _process_image(filename, bbox):
  """
  이미지 파일을 읽어들여 RGB 타입으로 변환하여 반환한다.
  :param filename: string, 읽어들일 이미지 파일 경로. e.g., '/path/to/example.JPG'.
  :param bbox: [xmin, ymin, xmax, ymax] 형식의 bbox 데이터 또는 None
  :return:
    image_data: 이미지 데이터로 jpg 포맷의 데이터.
    height: 이미지 height
    width: 이미지 width
  """
  with tf.gfile.GFile(filename, 'rb') as f:
    image_data = f.read()
    try:
      image = coder.decode_jpg(image_data)

      height = image.shape[0]
      width = image.shape[1]
    except tf.errors.InvalidArgumentError:
      raise ValueError("Invalid decode in {}".format(filename))

  if bbox is None:
    return image_data, height, width
  else:
    # change bbox to [y, x, h, w]
    crop_window = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]

    # food100 bbox 정보 중 일부 bbox 정보가 잘못된 경우가 있다.
    # 전체 이미지 크기보다 bbox 영역이 넘어가는 경우 에러가 발생한다.
    # 이를 막기위한 보정 코드를 추가한다.
    h_gap = crop_window[2] + crop_window[0] - height
    w_gap = crop_window[3] + crop_window[1] - width
    if h_gap > 0:
      crop_window[2] -= h_gap
    if w_gap > 0:
      crop_window[3] -= w_gap
    assert crop_window[2] > 0
    assert crop_window[3] > 0

    image = coder.crop_bbox(image, crop_window)
    image_data = coder.encode_jpg(image)
    return image_data, crop_window[2], crop_window[3]


def _process_image_files_batch(thread_index, offsets, output_filenames, filenames, labels, bbox_info):
  """
  하나의 스레드 단위에서 이미지 리스트를 읽어 TRRecord 타입으로 변환하는 함수
  :param thread_index: 현재 작업중인 thread 번호.
  :param offsets: offset list. 이미지 목록 중 현재 스레드에서 처리해야 할 offset 값으로 shard 갯수만큼 리스트로 제공
  :param output_filenames: 출력 파일 이름으로 shard 갯수만큼 리스트로 제공.
  :param filenames: 처리해야 할 전체 이미지 파일 리스트
  :param labels: 처리해야 할 전체 이미지 레이블 리스트
  :param bbox_info: 전체 이미지에 대한 bbox 정보로 사용하지 않을 경우 None 입력
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

      file_id = os.path.splitext(os.path.basename(filename))[0]
      if bbox_info is None:
        bbox = None  # bbox_info 가 없는 경우 bbox 정보로 crop 하는 과정을 생략.
      else:
        bbox = bbox_info[label][file_id]
      try:
        image_data, height, width = _process_image(filename, bbox)
      except ValueError:
        dataset_utils.log('[thread %2d]: Invalid image found. %s - [skip].' % (thread_index, filename))
        continue

      # crop 이 필요한 경우 bbox 정보를 추가하는 대신 이미지를 직접 crop 뒤에 추가한다.
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


def _process_dataset(name, filenames, labels, bbox_info, num_shards):
  """
  이미지 파일 목록을 읽어들여 TFRecord 객체로 변환하는 함수
  :param name: string, 데이터 고유 문자열 (train, validation 등)
  :param filenames: list of strings; 이미지 파일 경로 리스트.
  :param labels: list of integer; 이미지에 대한 정수화된 정답 레이블 리스트
  :param bbox_info: double dict of label id to file id. 사용하지 않는 경우 None 을 입력
  :param num_shards: 데이터 집합을 샤딩할 갯수.
  """
  assert len(filenames) == len(labels)

  shard_offsets = dataset_utils.make_shard_offsets(len(filenames), FLAGS.num_threads, num_shards)
  shard_output_filenames = dataset_utils.make_shard_filenames(name, len(filenames), FLAGS.num_threads, num_shards)

  def _process_batch(thread_index):
    offsets = shard_offsets[thread_index]
    output_filenames = shard_output_filenames[thread_index]
    _process_image_files_batch(thread_index, offsets, output_filenames, filenames, labels, bbox_info)

  dataset_utils.thread_execute(FLAGS.num_threads, _process_batch)
  dataset_utils.log('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))


def main(_):
  if (FLAGS.data_dir is None) or (FLAGS.output_dir is None):
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

  if FLAGS.use_bbox:
    bbox_info = _get_bbox_info(FLAGS.data_dir)
    dataset_utils.log(' - Use bounding box info. (opt. ON)')
  else:
    bbox_info = None

  # val{}.txt 파일 중 0~3 은 train set 으로 사용하고 4는 validation set 으로 사용.
  train_filenames, train_labels = _get_filenames_and_labels([0, 1, 2, 3], shuffle=True)
  validation_filenames, validation_labels = _get_filenames_and_labels([4], shuffle=False)

  dataset_utils.log('Convert [train] dataset.')
  _process_dataset('train', train_filenames, train_labels, bbox_info, FLAGS.train_shards)

  dataset_utils.log('Convert [validation] dataset.')
  _process_dataset('validation', validation_filenames, validation_labels, bbox_info, FLAGS.validation_shards)

  dataset_utils.log('Make UEC-food100 TFRecord dataset. [OK]')


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
