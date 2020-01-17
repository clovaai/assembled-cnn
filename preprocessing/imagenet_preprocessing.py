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

from preprocessing import autoaugment

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

# The lower bound for the smallest side of the image for aspect-preserving
# resizing. For example, if an image is 500 x 1000, it will be resized to
# _RESIZE_MIN x (_RESIZE_MIN * 2).
_RESIZE_MIN = 256


def _decode_crop_and_flip(image_buffer, bbox, num_channels, use_random_crop,
                          with_drawing_bbox=False, dct_method=''):
  # A large fraction of image datasets contain a human-annotated bounding box
  # delineating the region of the image containing the object of interest.  We
  # choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an
  # allowed range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
  if use_random_crop:
    min_object_covered = 0.1
  else:
    min_object_covered = 1.0
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
    tf.image.extract_jpeg_shape(image_buffer),
    bounding_boxes=bbox,
    min_object_covered=min_object_covered,
    aspect_ratio_range=[0.75, 1.33],
    area_range=[0.05, 1.0],
    max_attempts=100,
    use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, bbox_for_draw = sample_distorted_bounding_box

  if with_drawing_bbox:
    image_float32 = tf.cast(tf.image.decode_jpeg(image_buffer, channels=num_channels), tf.float32)
    raw_image_with_bbox = tf.image.draw_bounding_boxes(tf.expand_dims(image_float32, 0), bbox_for_draw)[0]
  else:
    raw_image_with_bbox = None

  # Reassemble the bounding box in the format the crop op requires.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

  # Use the fused decode and crop op here, which is faster than each in series.
  cropped = tf.image.decode_and_crop_jpeg(
    image_buffer, crop_window, channels=num_channels, dct_method=dct_method)

  # Flip to add a little more random distortion in.
  cropped = tf.image.random_flip_left_right(cropped)
  return cropped, raw_image_with_bbox


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


def mean_image_subtraction(image, means, num_channels):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = mean_image_subtraction(image, means)

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

    org_images = tf.stack([lu, ru, ld, rd, ct])

    return org_images, tf.stack([lu, ru, ld, rd, ct])

  lhs, aa_lhs = _crop5(image)
  rhs = tf.image.flip_left_right(lhs)

  return tf.concat([lhs, rhs], axis=0)


def preprocess_image_ten_crop(image_buffer, output_height, output_width, num_channels):
  image = tf.image.decode_jpeg(image_buffer, channels=num_channels, dct_method='INTEGER_ACCURATE')
  image = _aspect_preserving_resize(image, _RESIZE_MIN)
  images = _ten_crop(image, output_height, output_width)
  num_crops = 10
  images.set_shape([num_crops, output_height, output_width, num_channels])
  images = tf.map_fn(lambda x: mean_image_subtraction(x, CHANNEL_MEANS, num_channels), images)

  return images


def preprocess_image(image_buffer, bbox, output_height,
                     output_width, num_channels, is_training=False,
                     use_random_crop=True, autoaugment_type=None,
                     with_drawing_bbox=False, crop_type=0, dct_method=''):
  if is_training:
    # For training, we want to randomize some of the distortions.
    image, image_with_bbox = _decode_crop_and_flip(image_buffer, bbox, num_channels,
                                                   use_random_crop,
                                                   dct_method=dct_method,
                                                   with_drawing_bbox=with_drawing_bbox)

    image = _resize_image(image, output_height, output_width)

    if autoaugment_type:
      tf.logging.info('Apply AutoAugment policy {}'.format(autoaugment_type))
      image = tf.clip_by_value(image, 0.0, 255.0)
      dtype = image.dtype
      image = tf.cast(image, dtype=tf.uint8)
      image = autoaugment.distort_image_with_autoaugment(
        image, autoaugment_type)
      image = tf.cast(image, dtype=dtype)

    if with_drawing_bbox:
      image_with_bbox = _resize_image(image_with_bbox, output_height, output_width)
      image_with_bbox.set_shape([output_height, output_width, num_channels])

  else:
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels, dct_method=dct_method)

    if with_drawing_bbox:
      image_with_bbox = image
      image_with_bbox = _resize_image(image_with_bbox, output_height, output_width)
      image_with_bbox.set_shape([output_height, output_width, num_channels])
    else:
      image_with_bbox = None

    if crop_type == 1:
      image = _aspect_preserving_resize(image, int(min(output_height, output_width) + 1))
      image = central_crop(image, output_height, output_width)
    else:
      image = _aspect_preserving_resize(image, int(min(output_height, output_width) * (1.0 / 0.875)))
      image = central_crop(image, output_height, output_width)

  image.set_shape([output_height, output_width, num_channels])
  return mean_image_subtraction(image, CHANNEL_MEANS, num_channels), image_with_bbox

