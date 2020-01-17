# coding=utf8
# This code is adapted from the https://github.com/tensorflow/models/tree/master/official/r1/resnet.
# ==========================================================================================
# NAVER’s modifications are Copyright 2020 NAVER corp. All rights reserved.
# ==========================================================================================
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags

from functions import input_fns, data_config, model_fns
from nets import run_loop_classification
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from utils import log_utils
from utils import config_utils

def define_flags():
  run_loop_classification.define_resnet_flags(
    resnet_size_choices=['18', '34', '50', '101', '152', '200'])
  flags.adopt_module_key_flags(run_loop_classification)
  flags_core.set_defaults(train_epochs=90)


def run(flags_obj):
  """Run ResNet ImageNet training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  if not flags_obj.eval_only:
    config_utils.dump_hparam()
  dataset = data_config.get_config(flags_obj.dataset_name)

  run_loop_classification.resnet_main(
    flags_obj, model_fns.model_fn_cls, input_fns.input_fn_cls, dataset.dataset_name,
    shape=[dataset.default_image_size, dataset.default_image_size, dataset.num_channels],
    num_images=dataset.num_images, zeroshot_eval=flags_obj.zeroshot_eval)

def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run(flags.FLAGS)

if __name__ == '__main__':
  # 로그 두번 나오는 것 fix
  log_utils.define_log_level()
  define_flags()
  absl_app.run(main)
