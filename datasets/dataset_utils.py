#!/usr/bin/env python
# coding=utf8
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import shutil
import ssl
import sys
import tarfile
import threading
import zipfile
from datetime import datetime

import numpy as np
import tensorflow as tf
from six.moves import urllib


def log(msg, *args):
  msg = '[{}] ' + msg
  print(msg.format(datetime.now(), *args))
  sys.stdout.flush()


def str2bool(v):
  y = ['yes', 'true', 't', 'y', '1']
  n = ['no', 'false', 'f', 'n', '0']
  if v.lower() in y:
    return True
  elif v.lower() in n:
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_image_file_format(filename):
  image_name = filename.rsplit('.', 1)
  if len(image_name) <= 1:
    return 'jpg'  # default format
  image_format = image_name[-1].lower()
  if image_format in ['jpg', 'jpeg']:
    return 'jpg'
  elif image_format in ['bmp', 'png', 'gif']:
    return image_format
  return ""


def download_file(url, data_dir, filename):
  """
  URL로부터 데이터를 다운로드한다.
  :param url: 저장할 파일을 가리키는 URL.
  :param data_dir: 파일을 저장할 디렉토리 경로.
  :param filename: 저장할 파일 이름.
  :return: 저장된 파일의 경로 디렉토리.
  """
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  filepath = os.path.join(data_dir, filename)

  def _progress(count, block_size, total_size):
    if total_size > 0:
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
    else:
      sys.stdout.write('\r>> Downloading %s %s' % (filename, '.' * (count % 20)))
    sys.stdout.flush()

  # This is the way to allow unverified SSL
  ssl._create_default_https_context = ssl._create_unverified_context
  filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
  statinfo = os.stat(filepath)
  print()
  log('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def download_and_uncompress_tarball(tarball_url, data_dir, filename):
  """
  tar 형식으로 저장된 URL 파일을 다운받아 압축을 풀어 저장한다.
  :param tarball_url: tarball 파일 URL.
  :param data_dir: The directory where the output files are stored.
  :param filename: String, path to a output file.
  :return: The directory where the outputfiles are stored.
  """
  filepath = download_file(tarball_url, data_dir, filename)
  if filepath.endswith('tar'):
    tarfile.open(filepath, 'r:').extractall(data_dir)
  elif filepath.endswith('tar.gz'):
    tarfile.open(filepath, 'r:gz').extractall(data_dir)
  elif filepath.endswith('tgz'):
    tarfile.open(filepath, 'r:gz').extractall(data_dir)
  return data_dir


def download_and_uncompress_zip(zip_url, data_dir, zipped_filename):
  """
  zip 형식으로 저장된 URL 파일을 다운받아 압축을 풀어 저장한다.
  :param zip_url: The URL of zip file.
  :param data_dir: The directory where the output files are stored.
  :param zipped_filename: String, path to a output file.
  :return: Uncompredded file path.
  """
  zip_suffix = '.zip'
  zip_len = len(zip_suffix)
  assert len(zipped_filename) >= zip_len and zipped_filename[-zip_len:] == zip_suffix

  zipped_filepath = download_file(zip_url, data_dir, zipped_filename)
  zip_ref = zipfile.ZipFile(zipped_filepath, 'r')
  zip_ref.extractall(data_dir)
  zip_ref.close()
  return zipped_filepath


def download_and_uncompress_gzip(gzip_url, data_dir, zipped_filename):
  """
  Downloads the `gzip_url` and uncompresses it locally.
  :param gzip_url: The URL of gzip file.
  :param data_dir: The directory where the output files are stored.
  :param zipped_filename: String, path to a output file.
  :return: Uncompredded file path.
  """
  zip_suffix = '.gz'
  zip_len = len(zip_suffix)
  assert len(zipped_filename) >= zip_len and zipped_filename[-zip_len:] == zip_suffix

  zipped_filepath = download_file(gzip_url, data_dir, zipped_filename)
  filepath = zipped_filepath[:-zip_len]
  with gzip.open(zipped_filepath, 'rb') as f_in:
    # gzip only suppport single file.
    with tf.gfile.Open(filepath, 'wb') as f_out:
      shutil.copyfileobj(f_in, f_out)
  return filepath


def thread_execute(num_threads, fn):
  """
  Thread 단위로 fn 을 병렬 수행한다.
  :param num_threads: thread 갯수
  :param target_fn: thread 가 수행할 함수로 첫번째 인자에 thread index를 넘긴다.
  """
  assert num_threads > 0
  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()
  threads = []
  for idx in range(num_threads):
    t = threading.Thread(target=fn, args=(idx,))
    t.start()
    threads.append(t)
  coord.join(threads)  # Wait for all the threads to terminate.


def split(contents, num_split, start_index=0):
  """
  contents 를 num_split 값만큼 분할하여 리스트로 반환.
  :param contents: 분할하고자 하는 데이터 리스트
  :param num_split: 분할 갯수
  :param start_index: contents 시작 인덱스 번호로 default 값으로 0을 사용
  :return: split 갯수 크기의 더블 리스트. [[...],[...],...]
  """
  rs = np.linspace(start_index, len(contents), num_split + 1).astype(np.int)
  result = [contents[rs[i]:rs[i + 1]] for i in range(len(rs) - 1)]
  return result


def split_range(total, num_split, start_index=0):
  """
  정수 범위의 값을 num_split 값만큼 분할하여 해당 start, end 인덱스를 반환.
  :param total: 분할하고자 하는 max 값
  :param num_split: 분할 갯수
  :param start_index: contents 시작 인덱스 번호로 default 값으로 0을 사용
  :return: split 갯수 크기의 리스트로 start/end 인덱스 튜플을 원소로 가짐.  [(s,e),(s,e),...]
  """
  rs = np.linspace(start_index, total, num_split + 1).astype(np.int)
  result = [(rs[i], rs[i + 1]) for i in range(len(rs) - 1)]
  return result


def make_shard_offsets(total, num_threads, num_shards):
  """
  Thread 와 thread 내 shard 에서 사용할 인덱스 범위를 생성
  :param total: 분할하고자 하는 max 값
  :param num_threads: thread 갯수
  :param num_shards: 총 shard 수로 (num_threads * num_shards_per_thread)와 같다.
  :return: [[(s,e),(s,e)...],[()()...],...] 와 같은 형태의 더블 리스트.
  """
  assert total > 0
  assert num_threads > 0
  assert num_shards > 0
  assert not num_shards % num_threads

  num_shards_per_batch = int(num_shards / num_threads)
  thread_range = split_range(total, num_threads)
  offsets = []
  for start, end in thread_range:
    offsets.append(split_range(end, num_shards_per_batch, start_index=start))
  return offsets


def make_shard_filenames(name, total, num_threads, num_shards):
  assert total > 0
  assert num_threads > 0
  assert num_shards > 0

  offsets = make_shard_offsets(total, num_threads, num_shards)
  filenames = []
  shard_idx = 0
  for thread_offsets in offsets:
    shard_filenames = []
    for _ in thread_offsets:
      filename = '%s-%.5d-of-%.5d' % (name, shard_idx, num_shards)
      shard_idx += 1
      shard_filenames.append(filename)
    filenames.append(shard_filenames)
  return filenames


def make_label_id_to_name(data_dir, start_index=0):
  """
  레이블 이름이 디렉토리인 경우 이를 읽어 학습에 사용할 label id 로 매핑하는 함수.
  아래와 같은 형식의 데이터를 {0:labelA, 1:labelB} 와 같은 dict 객체로 변환한다.
    data_dir/labelA/xxx.jpg 
    data_dir/labelB/yyy.jpg
  :param data_dir: 레이블 디렉토리가 포함된 상위 디렉토리 이름.
  :return: name[id] 형식의 dict 객체.
  """
  id_to_name = {}
  label_index = 0 + start_index
  # os.listdir()은 순서를 보장하지 않으므로 반드시 sort하여 사용.
  for label_name in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, label_name)
    if os.path.isdir(path):
      image_file_path = '%s/*' % (path)
      matching_files = tf.gfile.Glob(image_file_path)
      id_to_name[label_index] = label_name
      label_index += 1
  return id_to_name


def make_label_name_to_id(data_dir, start_index=0):
  """
  레이블 이름이 디렉토리인 경우 이를 읽어 학습에 사용할 label id 로 매핑하는 함수.
  아래와 같은 형식의 데이터를 {labelA:0, labelB:1} 와 같은 dict 객체로 변환한다.
    data_dir/labelA/xxx.jpg 
    data_dir/labelB/yyy.jpg
  :param data_dir: 레이블 디렉토리가 포함된 상위 디렉토리 이름.
  :return: id[name] 형식의 dict 객체.
  """
  name_to_id = {}
  label_index = 0 + start_index
  # os.listdir()은 순서를 보장하지 않으므로 반드시 sort하여 사용.
  for label_name in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, label_name)
    if os.path.isdir(path):
      name_to_id[label_name] = label_index
      label_index += 1
  return name_to_id


def write_label_id_to_name(name, data_dir, output_dir=None, start_index=0):
  """
  id_to_name 정보를 data_dir 에 기록한다.
  :param name: 저장할 데이터의 타입. ex) train, validation
  :param data_dir: 레이블 디렉토리가 포함된 상위 디렉토리 이름.
  :param output_dir: 파일을 기록할 경로 디렉토리. None 인 경우 data_dir을 사용.
  """
  id_to_name = make_label_id_to_name(data_dir, start_index)
  output_filename = '%s_labels.txt' % (name)
  if not output_dir:
    output_dir = data_dir
  output_file = os.path.join(output_dir, output_filename)
  with open(output_file, 'w') as f:
    for index in sorted(id_to_name):
      f.write('%d:%s\n' % (index, id_to_name[index]))


def check_label_id_to_name_files(data_dir, train_name='train', validation_name='validation'):
  train_filename = os.path.join(data_dir, '%s_labels.txt' % (train_name))
  validation_filename = os.path.join(data_dir, '%s_labels.txt' % (validation_name))

  def _load_id_to_name(filename):
    id_to_name = {}
    with open(filename, 'r') as f:
      labels = f.readlines()
      for label in labels:
        label_id, name = label.strip().split(':', 2)
        id_to_name[int(label_id)] = name
      return id_to_name

  train = _load_id_to_name(train_filename)
  validation = _load_id_to_name(validation_filename)

  for key, val in train.items():
    if not key in validation:
      log("Warn: The label index (%d:%s) is not exist in validation but train." % (key, val))
    elif val != validation[key]:
      msg = "train(%d:%s) vs. validation(%d:%s)" % (key, val, key, validation[key])
      raise ValueError("Invalid label : {}".format(msg))

  for key, val in validation.items():
    if not key in train:
      msg = "valdation(%d:%s) is not exist in train label." % (key, val)
      raise ValueError("Invalid label : {}".format(msg))


def write_tfrecord_info(output_dir, num_train, num_validation):
  """
  train, validation 정보를 파일에 기록한다.
  :param output_dir: 파일을 기록할 경로 디렉토리. None 인 경우 data_dir을 사용.
  :param num_train: train 데이터 갯수
  :param num_validation: validation 데이터 갯수
  """
  id_to_name = make_label_id_to_name(output_dir)
  output_filename = 'tfrecord_info.txt'
  output_file = os.path.join(output_dir, output_filename)
  with open(output_file, 'w') as f:
    f.write('tfrecord info\n')
    f.write('- train: %s\n' % (num_train))
    f.write('- validation: %s\n' % (num_validation))
