from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

FLAGS = flags.FLAGS

def mixup(images, labels, y_t=None, alpha=0.2):
  """Applies Mixup regularization to a batch of images and labels.

  [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412

  Arguments:
    batch_size: The input batch size for images and labels.
    alpha: Float that controls the strength of Mixup regularization.
    images: A batch of images of shape [batch_size, ...]
    labels: A batch of labels of shape [batch_size, num_classes]

  Returns:
    A tuple of (images, labels) with the same dimensions as the input with
    Mixup regularization applied.
  """
  batch_size = tf.shape(images)[0]
  mix_weight = tfd.Beta(alpha, alpha).sample([batch_size, 1])
  mix_weight = tf.maximum(mix_weight, 1. - mix_weight)

  if images.dtype == tf.float16:
    mix_weight = tf.cast(mix_weight, dtype=tf.float16)
    labels = tf.cast(labels, dtype=tf.float16)
    if y_t is not None:
      y_t = tf.cast(y_t, dtype=tf.float16)

  images_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
  # Mixup on a single batch is implemented by taking a weighted sum with the
  # same batch in reverse.
  images_mix = (
          images * images_mix_weight + images[::-1] * (1. - images_mix_weight))
  labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)

  if images.dtype == tf.float16:
    labels_mix = tf.cast(labels_mix, tf.float32)

  if y_t is not None:
    # y1_t, y2_t = tf.split(y_t, 2, axis=0)
    # mixed_sy1_t = lam1_y * y1_t + (1. - lam1_y) * y2_t
    # mixed_sy1_t = tf.stop_gradient(mixed_sy1_t)
    teacher_labels_mix = y_t * mix_weight + y_t[::-1] * (1. - mix_weight)
    if images.dtype == tf.float16:
      teacher_labels_mix = tf.cast(teacher_labels_mix, tf.float32)
    return images_mix, labels_mix, teacher_labels_mix
  else:

    return images_mix, labels_mix


# def mixup(x, y, y_t=None, alpha=0.2):
#   dist = tfd.Beta(alpha, alpha)
#
#   _, h, w, c = x.get_shape().as_list()
#
#   batch_size = tf.shape(x)[0]
#   num_class = y.get_shape().as_list()[1]
#
#   # lam1 = dist.sample([batch_size // 2])
#   lam1 = dist.sample([batch_size // 2])
#
#   if x.dtype == tf.float16:
#     lam1 = tf.cast(lam1, dtype=tf.float16)
#     y = tf.cast(y, dtype=tf.float16)
#
#
#   x1, x2 = tf.split(x, 2, axis=0)
#   y1, y2 = tf.split(y, 2, axis=0)
#
#   # lam1_x = tf.tile(tf.reshape(lam1, [batch_size // 2, 1, 1, 1]), [1, h, w, c])
#   lam1_y = tf.tile(tf.reshape(lam1, [batch_size // 2, 1]), [1, num_class])
#
#   # mixed_sx1 = lam1_x * x1 + (1. - lam1_x) * x2
#   mixed_sy1 = lam1_y * y1 + (1. - lam1_y) * y2
#   # mixed_sx1 = tf.stop_gradient(mixed_sx1)
#   mixed_sy1 = tf.stop_gradient(mixed_sy1)
#
#   if y_t is not None:
#     y1_t, y2_t = tf.split(y_t, 2, axis=0)
#     mixed_sy1_t = lam1_y * y1_t + (1. - lam1_y) * y2_t
#     mixed_sy1_t = tf.stop_gradient(mixed_sy1_t)
#     return mixed_sx1, mixed_sy1, mixed_sy1_t
#   else:
#     return (x1, x2, lam1), mixed_sy1

# def mixup(x, y, y_t=None, alpha=0.2):
#   dist = tfd.Beta(alpha, alpha)
#
#   _, h, w, c = x.get_shape().as_list()
#
#   batch_size = tf.shape(x)[0]
#
#   lam1 = dist.sample([batch_size // 2])
#
#   if x.dtype == tf.float16:
#     lam1 = tf.cast(lam1, dtype=tf.float16)
#     y = tf.cast(y, dtype=tf.float16)
#     if y_t is not None:
#       y_t = tf.cast(y_t, dtype=tf.float16)
#
#   x1, x2 = tf.split(x, 2, axis=0)
#   y1, y2 = tf.split(y, 2, axis=0)
#
#   # lam1_x = tf.tile(tf.reshape(lam1, [batch_size // 2, 1, 1, 1]), [1, h, w, c])
#   lam1_x = tf.reshape(lam1, [batch_size // 2, 1, 1, 1])
#   # lam1_y = tf.tile(tf.reshape(lam1, [batch_size // 2, 1]), [1, num_class])
#   lam1_y = tf.reshape(lam1, [batch_size // 2, 1])
#
#   mixed_sx1 = lam1_x * x1 + (1. - lam1_x) * x2
#   mixed_sy1 = lam1_y * y1 + (1. - lam1_y) * y2
#   mixed_sx1 = tf.stop_gradient(mixed_sx1)
#   mixed_sy1 = tf.stop_gradient(mixed_sy1)
#
#   if y_t is not None:
#     y1_t, y2_t = tf.split(y_t, 2, axis=0)
#     mixed_sy1_t = lam1_y * y1_t + (1. - lam1_y) * y2_t
#     mixed_sy1_t = tf.stop_gradient(mixed_sy1_t)
#     if x.dtype == tf.float16:
#       mixed_sy1 = tf.cast(mixed_sy1, dtype=tf.float32)
#       mixed_sy1_t = tf.cast(mixed_sy1_t, dtype=tf.float32)
#     return mixed_sx1, mixed_sy1, mixed_sy1_t
#   else:
#     if x.dtype == tf.float16:
#       mixed_sy1 = tf.cast(mixed_sy1, dtype=tf.float32)
#     return mixed_sx1, mixed_sy1, None
#     # return (mixed_sx1, mixed_sy1), mixed_sy1

