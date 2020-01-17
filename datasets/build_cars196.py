#!/usr/bin/env python
# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""CAR196 를 생성한다.
example:
CUDA_VISIBLE_DEVICES="" python build_cars196.py \
--data_dir=cars196 \
--output_dir=cars196/tfrecord_cls \
--use_bbox=False 
"""

import os
import argparse
import sys

sys.path.append("../datasets")
sys.path.append("./datasets")
sys.path.append("..")
sys.path.append(".")
from datetime import datetime
from datasets import dataset_utils
from datasets import image_coder as coder
from utils import data_util
from scipy.io import loadmat
import tarfile
import subprocess

import tensorflow as tf
import numpy as np
import random

_NUM_CLASSES = 196

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_dir', type=str, default=None, help='Input data directory.')
parser.add_argument('-o', '--output_dir', type=str, default=None, help='Output data directory.')
parser.add_argument('--train_shards', type=int, default=128, help='Number of shards in training TFRecord files.')
parser.add_argument('--validation_shards', type=int, default=16, help='Number of shards in validation TFRecord files.')
parser.add_argument('--num_threads', type=int, default=8, help='Number of threads to preprocess the images.')
parser.add_argument('--use_bbox', type=dataset_utils.str2bool, nargs='?', const=True, default=False,
                    help='Whether to use bounding boxes or not.')


def _write_label_id_to_name(name, data_dir, id_to_name):
  output_filename = '%s_labels.txt' % (name)
  output_file = os.path.join(data_dir, output_filename)
  with open(output_file, 'w') as f:
    for index in sorted(id_to_name):
      f.write('%d:%s\n' % (index, id_to_name[index]))


def _get_bbox_info(dataset_name="cars196"):
  fuel_data_path = dataset_name
  label_filepath = os.path.join(fuel_data_path, "cars_annos.mat")
  # Extract class labels
  cars_annos = loadmat(label_filepath)
  annotations = cars_annos["annotations"].ravel()
  annotations = sorted(annotations, key=lambda a: str(a[0][0]))

  bbox_info = {}
  for i in range(len(annotations)):
    filename = annotations[i][0][0]
    file_id = os.path.splitext(os.path.basename(filename))[0]
    #   print(file_id)
    xmin = annotations[i][1][0][0]
    ymin = annotations[i][2][0][0]
    xmax = annotations[i][3][0][0]
    ymax = annotations[i][4][0][0]
    bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
    #   print(bbox)
    bbox_info[file_id] = bbox
  return bbox_info


def _find_image_files(dataset_name="cars196", archive_basename="car_ims"):
  fuel_data_path = dataset_name
  image_filepath = os.path.join(fuel_data_path, archive_basename + ".tgz")
  label_filepath = os.path.join(fuel_data_path, "cars_annos.mat")
  # Extract class labels
  cars_annos = loadmat(label_filepath)
  annotations = cars_annos["annotations"].ravel()
  annotations = sorted(annotations, key=lambda a: str(a[0][0]))
  class_labels = []
  for annotation in annotations:
    class_label = int(annotation[5])
    class_labels.append(class_label)

  if not os.path.exists(os.path.join(fuel_data_path, archive_basename)):
    subprocess.call(["tar", "zxvf", image_filepath.replace("\\", "/"),
                     "-C", fuel_data_path.replace("\\", "/"),
                     "--force-local"])

  train_filenames = []
  val_filenames = []
  train_labels = []
  val_labels = []

  def append_data(filenames, labels, label_idx, fn, fuel_data_path):
    # if not filenames[label_idx]:
    #   filenames[label_idx] = []
    # if not labels[label_idx]:
    #   labels[label_idx] = []

    filenames.append(os.path.join(os.getcwd(), fuel_data_path, fn))
    labels.append(label_idx)
    return filenames, labels

  for annotation in annotations:
    label_idx = int(annotation[5]) - 1
    fn = annotation[0][0]
    if int(annotation[6]) == 0:
      train_filenames, train_labels = append_data(train_filenames, train_labels, label_idx, fn, fuel_data_path)
    else:
      val_filenames, val_labels = append_data(val_filenames, val_labels, label_idx, fn, fuel_data_path)

  shuffled_index = list(range(len(train_filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)
  train_filenames = [train_filenames[i] for i in shuffled_index]
  train_labels = [train_labels[i] for i in shuffled_index]

  sum_train = len(train_filenames)
  sum_val = len(val_filenames)

  # sum_train += len(f)
  # for f in val_filenames:
  #   sum_val += len(f)

  return train_filenames, val_filenames, train_labels, val_labels, sum_train, sum_val


def _process_image(filename, bbox=None):
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
    image_format = dataset_utils.get_image_file_format(filename)

    # try:
    image = coder.decode_jpg(image_data)
    height = image.shape[0]
    width = image.shape[1]
    # except tf.errors.InvalidArgumentError:
    #   raise ValueError("Invalid decode in {}".format(filename))

  if bbox is None:
    return image_data, height, width, image_format
  else:
    # change bbox to [y, x, h, w]
    crop_window = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]

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
    image_format = 'jpg'
    return image_data, crop_window[2], crop_window[3], image_format


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

      # try:
      image_data, height, width, image_format = _process_image(filename)
      # except ValueError:
      #   dataset_utils.log('[thread %2d]: Invalid image found. %s - [skip].' % (thread_index, filename))
      #   continue

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


def main(unused_argv):
  if (FLAGS.data_dir is None) or (FLAGS.output_dir is None):
    parser.print_help()
    return

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  if FLAGS.use_bbox:
    bbox_info = _get_bbox_info()
    dataset_utils.log(' - Use bounding box info. (opt. ON)')
  else:
    bbox_info = None

  filenames_train, filenames_val, labels_train, labels_val, total_train, total_val \
    = _find_image_files(FLAGS.data_dir)
  # _process_dataset('train', filenames_train, labels_train, bbox_info)
  # # _write_label_id_to_name('train', FLAGS.output_dir, id_to_name)
  # dataset_utils.log('Finished writing all %d images in train data set.' % total_train)
  #
  # _process_dataset('validation', filenames_val, labels_val, bbox_info)
  # # _write_label_id_to_name('validation', FLAGS.output_dir, id_to_name)
  # dataset_utils.log('Finished writing all %d images in validation data set.' % total_val)

  print('filenames_train', len(filenames_train))
  dataset_utils.log('Convert [train] dataset.')
  _process_dataset('train', filenames_train, labels_train, FLAGS.train_shards)

  dataset_utils.log('Convert [validation] dataset.')
  _process_dataset('validation', filenames_val, labels_val, FLAGS.validation_shards)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
