from absl import flags
from tensorflow.python.keras import backend
import tensorflow as tf
import resnet_model as rm
from conv_bn import conv2d_fixed_padding
from conv_bn import batch_norm

FLAGS = flags.FLAGS

layers = tf.keras.layers

def bl_stage1(inputs, num_filters, strides=2):
  data_format = backend.image_data_format()

  big1 = conv2d_fixed_padding(inputs, num_filters, 3, strides, 'big1_conv')
  big1 = batch_norm(big1, data_format, name='big1_bn')

  little1 = conv2d_fixed_padding(inputs, num_filters // FLAGS.bl_alpha, 3, 1, 'little1_conv0')
  little1 = batch_norm(little1, data_format, name='little1_bn0')
  little1 = layers.Activation('relu')(little1)

  little1 = conv2d_fixed_padding(little1, num_filters // FLAGS.bl_alpha, 3, strides, 'little1_conv1')
  little1 = batch_norm(little1, data_format, name='little1_bn1')
  little1 = layers.Activation('relu')(little1)

  little1 = conv2d_fixed_padding(little1, num_filters, 1, 1, 'little1_conv2')
  little1 = batch_norm(little1, data_format, name='little1_bn2')

  merged = layers.Activation('relu')(big1 + little1)
  merged = conv2d_fixed_padding(merged, num_filters, 1, 1, 'merge1_conv0')
  merged = batch_norm(merged, data_format, name='merge1_bn0')
  merged = layers.Activation('relu')(merged)
  return merged


def bl_block(x, stage, strides, num_blocks, filters, kp):
  data_format = backend.image_data_format()
  big = rm.block_layer(x=x,
                       conv_strides=2,
                       num_blocks=num_blocks - 1,
                       stage=stage,
                       num_filters=filters,
                       kp=kp,
                       last_relu=False,
                       name='big{}_'.format(stage))
  big_e = layers.UpSampling2D(2, data_format=backend.image_data_format())(big)
  little = rm.block_layer(x=x,
                          conv_strides=1,
                          num_blocks=max(1, num_blocks // FLAGS.bl_beta - 1),
                          stage=stage,
                          num_filters=[f // FLAGS.bl_alpha for f in filters],
                          kp=kp,
                          name='little{}_'.format(stage))
  filters3 = filters[2] // 4 if FLAGS.use_bs and 2 <= stage <= 3 else filters[2]

  little_e = conv2d_fixed_padding(little, filters3, 1, 1, 'little_e_conv{}'.format(stage))
  little_e = batch_norm(little_e, data_format, name='little_e_bn{}'.format(stage))


  merge = layers.Activation('relu')(little_e + big_e)
  merge = rm.block_layer(x=merge,
                         conv_strides=strides,
                         num_blocks=1,
                         stage=stage,
                         num_filters=filters,
                         kp=kp,
                         name='merge{}_'.format(stage))
  return merge
