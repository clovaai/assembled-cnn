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

import logging
import sys

import tensorflow as tf


def log_var_list_by_line(var_list, var_list_name):
  tf.logging.info('*********** begin %s **************', var_list_name)
  for v in var_list:
    tf.logging.info(v.name)
  tf.logging.info('*********** end   %s **************', var_list_name)


def define_log_level():
  tf_logger = logging.getLogger('tensorflow')
  tf_logger.propagate = False
  handler = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter("%(asctime)s.%(msecs).3d %(levelname).1s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
  handler.setFormatter(formatter)
  tf_logger.handlers = [handler]
  tf_logger.setLevel(tf.logging.INFO)
