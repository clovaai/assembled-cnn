# This code is adapted from the https://github.com/tensorflow/models/tree/master/official/r1/resnet.
# ==========================================================================================
# NAVER’s modifications are Copyright 2020 NAVER corp. All rights reserved.
# ==========================================================================================
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Provides utilities to preprocess images.

Training images are sampled using the provided bounding boxes, and subsequently
cropped to the sampled bounding box. Images are additionally flipped randomly,
then resized to the target output size (without aspect-ratio preservation).

Images used during evaluation are resized (with aspect-ratio preservation) and
centrally cropped.

All images undergo mean color subtraction.

Note that these steps are colloquially referred to as "ResNet preprocessing,"
and they differ from "VGG preprocessing," which does not use bounding boxes
and instead does an aspect-preserving resize followed by random crop during
training. (These both differ from "Inception preprocessing," which introduces
color distortion steps.)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from preprocessing import autoaugment

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
# The lower bound for the smallest side of the image for aspect-preserving
# resizing. For example, if an image is 500 x 1000, it will be resized to
# _RESIZE_MIN x (_RESIZE_MIN * 2).
_RESIZE_MIN = 256

def central_crop(image, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image: a 3-D image tensor
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    3-D tensor with cropped image.
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  amount_to_be_cropped_h = (height - crop_height)
  crop_top = amount_to_be_cropped_h // 2
  amount_to_be_cropped_w = (width - crop_width)
  crop_left = amount_to_be_cropped_w // 2
  return tf.slice(
    image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _mean_image_subtraction(image, means, num_channels):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  # Note(b/130245863): we explicitly call `broadcast` instead of simply
  # expanding dimensions for better performance.
  means = tf.broadcast_to(means, tf.shape(image))

  return image - means


def _normalization(image, means, stds, num_channels):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  # Note(b/130245863): we explicitly call `broadcast` instead of simply
  # expanding dimensions for better performance.
  means = tf.broadcast_to(means, tf.shape(image))
  stds = tf.broadcast_to(stds, tf.shape(image))

  return (image - means) / stds


def _smallest_size_at_least(height, width, resize_min):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: an int32 scalar tensor indicating the new width.
  """
  resize_min = tf.cast(resize_min, tf.float32)

  # Convert to floats to make subsequent calculations go smoothly.
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

  smaller_dim = tf.minimum(height, width)
  scale_ratio = resize_min / smaller_dim

  # Convert back to ints to make heights and widths that TF ops will accept.
  new_height = tf.cast(height * scale_ratio, tf.int32)
  new_width = tf.cast(width * scale_ratio, tf.int32)

  return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  new_height, new_width = _smallest_size_at_least(height, width, resize_min)

  return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.

  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.

  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.

  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
  return tf.image.resize_images(
    image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
    align_corners=False)


def _ten_crop(image, crop_h, crop_w):
  def _crop(img, center_offset):
    # input img shape is [h,w,c]
    img = tf.image.extract_glimpse(
      [img], [crop_w, crop_h], offsets=tf.to_float([center_offset]),
      centered=False, normalized=False)
    return tf.squeeze(img, 0)

  def _crop5(img):
    # img shape is [h,w,c]
    im_shape = tf.shape(image)
    height, width = im_shape[0], im_shape[1]
    ch, cw = tf.to_int32(height / 2), tf.to_int32(width / 2)  # center offset
    hh, hw = tf.to_int32(crop_h / 2), tf.to_int32(crop_w / 2)  # half crop size
    ct = _crop(img, [ch, cw])
    lu = _crop(img, [hh, hw])
    ld = _crop(img, [height - hh, hw])
    ru = _crop(img, [hh, width - hw])
    rd = _crop(img, [height - hh, width - hw])
    return tf.stack([lu, ru, ld, rd, ct])

  lhs = _crop5(image)
  rhs = tf.image.flip_left_right(lhs)
  return tf.concat([lhs, rhs], axis=0)


def preprocess_image_ten_crop(image_buffer, output_height, output_width, num_channels):
  image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
  image = _aspect_preserving_resize(image, _RESIZE_MIN)
  images = _ten_crop(image, output_height, output_width)
  images.set_shape([10, output_height, output_width, num_channels])
  images = tf.map_fn(lambda x: _mean_image_subtraction(x, _CHANNEL_MEANS, num_channels), images)
  return images

def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
    tf.equal(tf.rank(image), 3),
    ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
    tf.logical_and(
      tf.greater_equal(original_shape[0], crop_height),
      tf.greater_equal(original_shape[1], crop_width)),
    ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def _get_random_crop_coord(image_list, crop_height, crop_width):
  """Crops the given list of images.
  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:
    image, depths, normals = _random_crop([image, depths, normals], 120, 150)
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.
  Returns:
    the image_list with cropped images.
  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
      tf.equal(image_rank, 3),
      ['Wrong rank for tensor  %s [expected] [actual]',
       image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  image_shape = control_flow_ops.with_dependencies(
    [rank_assertions[0]],
    tf.shape(image_list[0]))
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
    tf.logical_and(
      tf.greater_equal(image_height, crop_height),
      tf.greater_equal(image_width, crop_width)),
    ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    shape = control_flow_ops.with_dependencies([rank_assertions[i]],
                                               tf.shape(image))
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
      tf.equal(height, image_height),
      ['Wrong height for tensor %s [expected][actual]',
       image.name, height, image_height])
    width_assert = tf.Assert(
      tf.equal(width, image_width),
      ['Wrong width for tensor %s [expected][actual]',
       image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  max_offset_height = control_flow_ops.with_dependencies(
    asserts, tf.reshape(image_height - crop_height + 1, []))
  max_offset_width = control_flow_ops.with_dependencies(
    asserts, tf.reshape(image_width - crop_width + 1, []))
  offset_height = tf.random_uniform(
    [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
    [], maxval=max_offset_width, dtype=tf.int32)

  return tf.stack([offset_height, offset_width, crop_height, crop_width])


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.
  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:
    image, depths, normals = _random_crop([image, depths, normals], 120, 150)
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.
  Returns:
    the image_list with cropped images.
  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
      tf.equal(image_rank, 3),
      ['Wrong rank for tensor  %s [expected] [actual]',
       image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  image_shape = control_flow_ops.with_dependencies(
    [rank_assertions[0]],
    tf.shape(image_list[0]))
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
    tf.logical_and(
      tf.greater_equal(image_height, crop_height),
      tf.greater_equal(image_width, crop_width)),
    ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    shape = control_flow_ops.with_dependencies([rank_assertions[i]],
                                               tf.shape(image))
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
      tf.equal(height, image_height),
      ['Wrong height for tensor %s [expected][actual]',
       image.name, height, image_height])
    width_assert = tf.Assert(
      tf.equal(width, image_width),
      ['Wrong width for tensor %s [expected][actual]',
       image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  max_offset_height = control_flow_ops.with_dependencies(
    asserts, tf.reshape(image_height - crop_height + 1, []))
  max_offset_width = control_flow_ops.with_dependencies(
    asserts, tf.reshape(image_width - crop_width + 1, []))
  offset_height = tf.random_uniform(
    [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
    [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def pad_shorter(image):
  shape = tf.shape(image)
  height, width = shape[0], shape[1]
  larger_dim = tf.maximum(height, width)
  h1 = (larger_dim - height) // 2
  h2 = (larger_dim - height) - h1
  w1 = tf.maximum((larger_dim - width) // 2, 0)
  w2 = (larger_dim - width) - w1
  pad_shape = [[h1, h2], [w1, w2], [0, 0]]
  return tf.pad(image, pad_shape)


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
    func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
    for case in range(num_cases)])[0]


def resize_func(image, size, method):
  if method == 0:
    image = _resize_image(image, _RESIZE_MIN, _RESIZE_MIN)
    image = _random_crop([image], size[0], size[1])[0]
  else:
    image = _resize_image(image, size[0], size[1])
  return image

def preprocess_image(image_buffer,
                     output_height,
                     output_width,
                     num_channels,
                     dct_method='',
                     is_training=False,
                     autoaugment_type=None,
                     eval_large_resolution=True):

  if is_training:
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels, dct_method=dct_method)

    image = apply_with_random_selector(
      image,
      lambda x, method: resize_func(x, [output_height, output_width], method),
      num_cases=2)

    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)

    if autoaugment_type:
      tf.logging.info('Apply AutoAugment policy {}'.format(autoaugment_type))
      image = tf.clip_by_value(image, 0.0, 255.0)
      dtype = image.dtype
      image = tf.cast(image, dtype=tf.uint8)
      image = autoaugment.distort_image_with_autoaugment(
        image, autoaugment_type)
      image = tf.cast(image, dtype=dtype)

    image.set_shape([output_height, output_width, num_channels])

  else:
    if eval_large_resolution:
      output_height = int(output_height * (1.0 / 0.875))
      output_width = int(output_width * (1.0 / 0.875))
    # For validation, we want to decode, resize, then just crop the middle.
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels, dct_method=dct_method)

    image = _resize_image(image, output_height, output_width)
    image = tf.to_float(image)
    image.set_shape([output_height, output_width, num_channels])

  return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)
