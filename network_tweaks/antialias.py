from absl import flags
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend

FLAGS = flags.FLAGS
layers = tf.keras.layers

class AntiAliasing(layers.Layer):
  def __init__(self, filt_size=3, stride=2, pad_off=0, **kwargs):
    super(AntiAliasing, self).__init__(**kwargs)
    self.filt_size = filt_size
    self.stride = stride
    self.pad_off = pad_off

  def call(self, inp, **kwargs):
    filt_size = self.filt_size
    pad_off = self.pad_off
    stride = self.stride
    data_format = backend.image_data_format()
    pading_size = int(1. * (filt_size - 1) / 2)
    if data_format == 'channels_first':
      pad_sizes = [[0, 0], [0, 0], [pading_size + pad_off, pading_size + pad_off],
                   [pading_size + pad_off, pading_size + pad_off]]
    else:
      pad_sizes = [[0, 0], [pading_size + pad_off, pading_size + pad_off],
                   [pading_size + pad_off, pading_size + pad_off],
                   [0, 0]]

    if filt_size == 1:
      a = np.array([1., ])
    elif filt_size == 2:
      a = np.array([1., 1.])
    elif filt_size == 3:
      a = np.array([1., 2., 1.])
    elif filt_size == 4:
      a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
      a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
      a = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
      a = np.array([1., 6., 15., 20., 15., 6., 1.])

    channel_axis = 1 if data_format == 'channels_first' else 3
    G = inp.shape[channel_axis]

    filt = tf.constant(a[:, None] * a[None, :], inp.dtype)
    filt = filt / tf.reduce_sum(filt)
    filt = tf.reshape(filt, [filt_size, filt_size, 1, 1])
    filt = tf.tile(filt, [1, 1, G, 1])

    if filt_size == 1:
      if pad_off == 0:
        return inp[:, :, ::stride, ::stride]
      else:
        padded = tf.pad(inp, pad_sizes, "REFLECT")
        return padded[:, :, ::stride, ::stride]
    else:
      inp = tf.pad(inp, pad_sizes, "REFLECT")
      data_format = "NCHW" if data_format == 'channels_first' else "NHWC"

      strides = (1, 1, stride, stride) if data_format == 'NCHW' else (1, stride, stride, 1)

      output = tf.nn.depthwise_conv2d(inp, filt, strides=strides,
                                      padding='VALID', data_format=data_format)
      return output
