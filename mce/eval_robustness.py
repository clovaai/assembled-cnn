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

import os
import sys

sys.path.append('./')
sys.path.append('../')

from absl import flags
import tensorflow as tf
import numpy as np
import glob

from functions import model_fns
from preprocessing import imagenet_preprocessing
import threading

flags.DEFINE_string(
  'robustness_type', 'ce',
  '[ce|fr]')

flags.DEFINE_string(
  'gpu_index', '0', 'The index of gpus.')

flags.DEFINE_integer(
  'label_offset', 1, '')

flags.DEFINE_integer(
  'num_classes', '1001', 'The number of classes.')

flags.DEFINE_integer(
  'batch_size', 128, 'The number of samples in each batch.')

flags.DEFINE_integer(
  'image_size', 224, 'The number of samples in each batch.')

flags.DEFINE_integer(
  'resnet_size', 50, 'The number of convolutional layers needed in the model.')

flags.DEFINE_integer(
  'resnet_version', 1, 'Resnet version')

flags.DEFINE_string(
  'data_format', 'channels_first',
  '[channels_fisrt|channels_last]')

flags.DEFINE_boolean(name='use_resnet_d', default=False, help='')
flags.DEFINE_boolean(name='use_se_block', default=False, help='')
flags.DEFINE_boolean(name='use_sk_block', default=False, help='')
flags.DEFINE_boolean(name='zero_gamma', default=True, help='')
flags.DEFINE_boolean(name='no_downsample', default=False, help='')
flags.DEFINE_integer(name='anti_alias_filter_size', default=0, help='')
flags.DEFINE_string(name='anti_alias_type', default="", help='')
flags.DEFINE_integer(name='bl_alpha', default=2, help='')
flags.DEFINE_integer(name='bl_beta', default=4, help='')
flags.DEFINE_integer(name='embedding_size', default=0, help='')
flags.DEFINE_string(name='pool_type', default='gap', help='')

flags.DEFINE_string(
  'data_dir', './imagenet-C',
  'path to the root directory of images')

flags.DEFINE_string(
  'label_file', './label.txt',
  'path to the label file of imagenet')

flags.DEFINE_string(
  'model_dir', 'tmp/tfmodel/',
  'Create a flag for specifying the model file directory.')

FLAGS = flags.FLAGS

ALEXNET_CE = {
  'gaussian_noise': 0.8864, 'shot_noise': 0.8945, 'impulse_noise': 0.9226,
  'defocus_blur': 0.8199, 'glass_blur': 0.8263, 'motion_blur': 0.7860, 'zoom_blur': 0.7984,
  'snow': 0.8668, 'frost': 0.8266, 'fog': 0.8193, 'brightness': 0.5646,
  'contrast': 0.8532, 'elastic_transform': 0.6461, 'pixelate': 0.7178, 'jpeg_compression': 0.6065,
  'speckle_noise': 0.8454, 'gaussian_blur': 0.7871, 'spatter': 0.7175, 'saturate': 0.6583
}

OURS_CE = {
  'gaussian_noise': 0.0, 'shot_noise': 0.0, 'impulse_noise': 0.0,
  'defocus_blur': 0.0, 'glass_blur': 0.0, 'motion_blur': 0.0, 'zoom_blur': 0.0,
  'snow': 0.0, 'frost': 0.0, 'fog': 0.0, 'brightness': 0.0,
  'contrast': 0.0, 'elastic_transform': 0.0, 'pixelate': 0.0, 'jpeg_compression': 0.0,
  'speckle_noise': 0.0, 'gaussian_blur': 0.0, 'spatter': 0.0, 'saturate': 0.0
}

ABSOLUTE_CE = {
  'gaussian_noise': 0.0, 'shot_noise': 0.0, 'impulse_noise': 0.0,
  'defocus_blur': 0.0, 'glass_blur': 0.0, 'motion_blur': 0.0, 'zoom_blur': 0.0,
  'snow': 0.0, 'frost': 0.0, 'fog': 0.0, 'brightness': 0.0,
  'contrast': 0.0, 'elastic_transform': 0.0, 'pixelate': 0.0, 'jpeg_compression': 0.0,
  'speckle_noise': 0.0, 'gaussian_blur': 0.0, 'spatter': 0.0, 'saturate': 0.0
}

ALEXNET_FP = {
  'gaussian_noise': 0.23653, 'shot_noise': 0.30062, 'motion_blur': 0.09297,
  'zoom_blur': 0.05942, 'spatter': 0.05053, 'brightness': 0.04889,
  'translate': 0.11007, 'rotate': 0.13104, 'tilt': 0.07050, 'scale': 0.23531,
  'speckle_noise': 0.18649, 'gaussian_blur': 0.02775, 'snow': 0.11928, 'shear': 0.10658
}


def input_fn_imagenet_c(image_files, batch_size):
  def _parse_function(image_file):
    image_buffer = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image_buffer, channels=3, dct_method="INTEGER_ACCURATE")

    def bypass(imagel):
      imagel = tf.cast(imagel, tf.float32)
      imagel.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
      imagel = imagenet_preprocessing.mean_image_subtraction(imagel, imagenet_preprocessing.CHANNEL_MEANS, 3)
      return imagel

    image = bypass(image)
    image = tf.cast(image, tf.float32)

    return image, image_file

  dataset = tf.data.Dataset.from_tensor_slices(image_files)
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(
    tf.contrib.data.map_and_batch(
      _parse_function,
      batch_size=batch_size,
      num_parallel_batches=1,
      drop_remainder=False))

  return dataset


def get_synsets2idx(labels_file):
  challenge_synsets = [l.strip().replace(' ', '_').lower() for l in
                       tf.gfile.GFile(labels_file, 'r').readlines()]
  synsets2idx = {}
  for i, synset in enumerate(challenge_synsets):
    synsets2idx[synset] = i

  return synsets2idx


def show_corruption_error_by_distortion(distortion_name, current_ckpt, gpu_id):
  synsets2idx = get_synsets2idx(FLAGS.label_file)

  distortion_dir = os.path.join(FLAGS.data_dir, distortion_name)

  with tf.device('/gpu:%d' % gpu_id):
    model = model_fns.Model(resnet_size=FLAGS.resnet_size,
                            num_classes=FLAGS.num_classes,
                            resnet_version=FLAGS.resnet_version,
                            use_se_block=FLAGS.use_se_block,
                            use_sk_block=FLAGS.use_sk_block,
                            zero_gamma=FLAGS.zero_gamma,
                            data_format=FLAGS.data_format,
                            no_downsample=FLAGS.no_downsample,
                            anti_alias_filter_size=FLAGS.anti_alias_filter_size,
                            anti_alias_type=FLAGS.anti_alias_type,
                            embedding_size=FLAGS.embedding_size,
                            pool_type=FLAGS.pool_type,
                            bl_alpha=FLAGS.bl_alpha,
                            bl_beta=FLAGS.bl_beta,
                            dtype=tf.float32)
    images = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
    logits = model(inputs=images, training=False,
                   use_resnet_d=FLAGS.use_resnet_d,
                   reuse=tf.AUTO_REUSE)
    softmax = tf.nn.softmax(logits)
    sm_top1 = tf.nn.top_k(softmax, 1)

  saver = tf.train.Saver()

  image_files = []
  for severity in range(1, 6):
    severity_dir = os.path.join(distortion_dir, str(severity))
    for d in os.listdir(severity_dir):
      for f in glob.glob(os.path.join(severity_dir, d) + '/*'):
        image_files.append(f)

  dataset = input_fn_imagenet_c(image_files, FLAGS.batch_size)

  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()

  with tf.Session() as sess:
    saver.restore(sess, current_ckpt)

    num_of_images = 0
    correct = 0

    while True:
      try:
        images_tensor, image_files = next_element
        images_input, image_files = sess.run([images_tensor, image_files])

        result = sess.run(sm_top1, feed_dict={images: images_input})

        for i in range(len(result.indices)):
          fn = os.path.splitext(image_files[i].decode("utf-8"))[0]
          synset = os.path.basename(os.path.dirname(fn))
          gt = synsets2idx[synset] + FLAGS.label_offset

          # top1
          pred = result.indices[i][0]

          num_of_images += 1
          if pred == gt:
            correct += 1

      except tf.errors.OutOfRangeError:
        break

  assert (num_of_images > 0)
  err = (1 - 1. * correct / num_of_images)

  return err


def thread_execute(num_threads, fn, current_ckpt):
  assert num_threads > 0
  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()
  threads = []
  for idx in range(num_threads):
    t = threading.Thread(target=fn, args=(idx, current_ckpt))
    t.start()
    threads.append(t)
  coord.join(threads)  # Wait for all the threads to terminate.


def process(no_thread, current_ckpt):
  print('!!!!!!!!!!!!!!!!!! start', no_thread)
  distortions = list(ALEXNET_CE.keys())
  num_distortions = len(distortions)

  num_gpus = len(FLAGS.gpu_index.split(','))
  quota_base = num_distortions // num_gpus
  overtime_task = num_distortions % num_gpus

  quota = []
  base = 0
  for i in range(num_gpus):
    if i < overtime_task:
      quota.append(base + quota_base + 1)
      base += quota_base + 1
    else:
      quota.append(base + quota_base)
      base += quota_base

  print('quota', quota)
  # assert sum(quota) == num_distortions

  start_idx = 0 if no_thread is 0 else quota[no_thread - 1]
  end_idx = quota[no_thread]
  quota_distortions = distortions[start_idx:end_idx]
  print('quota_distortions', quota_distortions)

  for distortion_name in quota_distortions:
    rate = show_corruption_error_by_distortion(distortion_name, current_ckpt, no_thread)
    ce = rate / ALEXNET_CE[distortion_name]
    ABSOLUTE_CE[distortion_name] = rate
    # error_rates.append(ce)
    OURS_CE[distortion_name] = ce
    print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}  | CE (normalized) (%): {:.2f}'.format(distortion_name,
                                                                                                      100 * rate,
                                                                                                      100 * ce))


def show_corruption_error():
  num_gpus = len(FLAGS.gpu_index.split(','))

  if tf.gfile.IsDirectory(FLAGS.model_dir):
    current_ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
  else:
    current_ckpt = FLAGS.model_dir

  thread_execute(num_gpus, process, current_ckpt)

  error_rates = list(OURS_CE.values())
  abs_error_rates = list(ABSOLUTE_CE.values())
  m_ce = np.mean(error_rates)
  abs_m_ce = np.mean(abs_error_rates)
  print('mCE (normalized by AlexNet errors) (%): {:.2f}'.format(100 * m_ce))
  print('mCE (unnormalized) (%): {:.2f}'.format(100 * abs_m_ce))


  tf.summary.scalar('mCE', tf.constant(m_ce))
  merged = tf.summary.merge_all()

  model_dir = os.path.dirname(os.path.splitext(current_ckpt)[0])
  test_writer = tf.summary.FileWriter(model_dir + '/eval')

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary = sess.run(merged)
    global_step = int(os.path.basename(current_ckpt).split('-')[1])
    test_writer.add_summary(summary, global_step)


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

  if FLAGS.robustness_type == 'ce':
    show_corruption_error()
  elif FLAGS.robustness_type == 'fr':
    raise ValueError('not yet supported ({}'.format(FLAGS.robustness_type))
  else:
    raise ValueError('invalid type ({})'.format(FLAGS.robustness_type))


if __name__ == '__main__':
  tf.app.run()
