#!/usr/bin/env python
# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# use eager execution
import tensorflow as tf

tf.enable_eager_execution()


# decorator
def with_tf_cpu(fn):
  def wrapper_fn(*args, **kwargs):
    with tf.device('CPU:0'):
      return fn(*args, **kwargs)

  return wrapper_fn


@with_tf_cpu
def decode_png(contents):
  image = tf.image.decode_png(contents, channels=3)
  return image.numpy()


@with_tf_cpu
def decode_jpg(contents):
  # 정확한 성능을 내기 위해서는 dct_method='INTEGER_ACCURATE' 를 사용해야 한다.
  image = tf.image.decode_jpeg(contents, channels=3, dct_method='INTEGER_ACCURATE')
  return image.numpy()


@with_tf_cpu
def decode_bmp(contents):
  image = tf.image.decode_bmp(contents, channels=3)
  return image.numpy()


@with_tf_cpu
def decode_gif(contents):
  image = tf.image.decode_gif(contents).numpy()
  if len(image.shape) == 4:
    return image[0,]
  return image


@with_tf_cpu
def decode(contents, image_type=None):
  if image_type == 'jpg':
    return decode_jpg(contents)
  elif image_type == 'png':
    return decode_png(contents)
  elif image_type == 'bmp':
    return decode_bmp(contents)
  elif image_type == 'gif':
    return decode_gif(contents)
  image = tf.image.decode_image(contents).numpy()
  if len(image.shape) == 4:
    return image[0,]
  return image


@with_tf_cpu
def encode_png(image):
  image_data = tf.image.encode_png(image)
  return image_data.numpy()


@with_tf_cpu
def encode_jpg(image):
  image_data = tf.image.encode_jpeg(image, format='rgb', quality=100)
  return image_data.numpy()


def crop(image, offset_height, offset_width, crop_height, crop_width):
  assert len(image.shape) == 3
  assert offset_height >= 0
  assert offset_width >= 0
  assert crop_height > 0
  assert crop_width > 0
  assert (offset_height + crop_height) <= image.shape[0]
  assert (offset_width + crop_width) <= image.shape[1]
  return image[offset_height:(offset_height + crop_height),
         offset_width:(offset_width + crop_width)]


def crop_bbox(image, bbox=None):
  if bbox is None:
    return image
  return crop(image, bbox[0], bbox[1], bbox[2], bbox[3])


@with_tf_cpu
def central_crop(image, central_fraction):
  cropped_image = tf.image.central_crop(image, central_fraction)
  return cropped_image.numpy()


def _smallest_size_at_least(height, width, smallest_side):
  """
  Computes new shape with the smallest side equal to `smallest_side`.
  :param height: A python integer indicating the current height.
  :param width: A python integer indicating the current width.
  :param smallest_side: A python integer indicating the size of the smallest side after resize.
  :return: tuple (new_height, new_width) of int type.
  """
  assert height > 0
  assert width > 0
  assert smallest_side > 0
  scale = float(smallest_side) / min(height, width)
  new_height = int(np.rint(height * scale))
  new_width = int(np.rint(width * scale))
  return (new_height, new_width)


@with_tf_cpu
def resize(image, height, width):
  image = tf.image.resize_images(
    image, [height, width],
    method=tf.image.ResizeMethod.BILINEAR,
    align_corners=False)
  return image.numpy().astype(np.uint8)


@with_tf_cpu
def aspect_preserving_resize(image, resize_min):
  """
  Resize images preserving the original aspect ratio.

  :param image: A 3-D image `Tensor`.
  :param resize_min: A python integer or scalar `Tensor` indicating the size ofthe smallest side after resize.
  :return: A 3-D tensor containing the resized image.
  """
  height, width = image.shape[0], image.shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, resize_min)
  return resize(image, new_height, new_width), new_height, new_width
