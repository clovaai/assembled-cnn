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


class Default(object):
  shuffle_buffer = 1000
  ir_shuffle_buffer = 100
  default_image_size = 224
  num_channels = 3
  num_train_files = 128
  num_val_files = 16


class Food101(Default):
  shuffle_buffer = 1000
  shuffle_buffer_unsup = 10000
  num_classes = 101
  num_images = {
    'train': 75750,
    'validation': 25250,
  }
  dataset_name = 'food101'


class ImageNet(Default):
  shuffle_buffer = 10000
  num_classes = 1001
  num_images = {
    'train': 1281167,
    'validation': 50000,
  }
  num_train_files = 1024
  dataset_name = 'imagenet'

class CUB_200_2011(Default):
  shuffle_buffer = 100
  num_classes = 100
  num_images = {
    'train': 5864,
    'validation': 5924,
  }
  num_train_files = 100
  dataset_name = 'cub_200_2011'

class iFood2019(Default):
  num_classes = 251
  num_images = {
    'train': 118475,
    'validation': 11994,
  }
  dataset_name = 'iFood2019'


class StanfordOnlineProduct(Default):
  shuffle_buffer = 50000
  num_classes = 11318
  num_images = {
    'train': 59551,
    'validation': 60502,
  }
  dataset_name = 'SOP'


class Flower102(Default):
  num_classes = 102
  num_images = {
    'train': 2040,
    'validation': 6149,
  }
  dataset_name = 'oxford_flowers102'


class Cars196(Default):
  num_classes = 196
  num_images = {
    'train': 8144,
    'validation': 8041,
  }
  dataset_name = 'cars196'


class Cars196ZeroShot(Default):
  num_classes = 196
  num_images = {
    'train': 8054,
    'validation': 8131,
  }
  dataset_name = 'cars196_zeroshot'


class OxfordPet(Default):
  num_classes = 37
  num_images = {
    'train': 3680,
    'validation': 3669,
  }
  dataset_name = 'oxford_iiit_pet'


class Aircraft(Default):
  num_classes = 100
  num_images = {
    'train': 6667,
    'validation': 3333,
  }
  dataset_name = 'fgvc_aircraft'


def get_config(data_name):
  if data_name == 'imagenet':
    return ImageNet()
  elif data_name == 'cars196':
    return Cars196()
  elif data_name == 'food101':
    return Food101()
  elif data_name == 'oxford_flowers102':
    return Flower102()
  elif data_name == 'oxford_iiit_pet':
    return OxfordPet()
  elif data_name == 'fgvc_aircraft':
    return Aircraft()
  elif data_name == 'cub_200_2011':
    return CUB_200_2011()
  elif data_name == 'cars196_zeroshot':
    return Cars196ZeroShot()
  elif data_name == 'SOP':
    return StanfordOnlineProduct()
  elif data_name == 'iFood2019':
    return iFood2019()
  else:
    raise ValueError("Unable to support {} dataset.".format(data_name))
