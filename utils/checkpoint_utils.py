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
import glob
import json
import os
from shutil import copyfile

import tensorflow as tf


class CheckpointKeeper(object):
  def __init__(self, save_dir, num_to_keep=1, keep_epoch=False, maximize=True):
    """Creates a `BestCheckpointSaver`

    Args:
        save_dir: The directory in which the checkpoint files will be saved
        num_to_keep: The number of best checkpoint files to retain
        keep_epoch: If True, checkpoints are saved for each evaluation.
        maximize: Define 'best' values to be the highest values.  For example,
          set this to True if selecting for the checkpoints with the highest
          given accuracy.  Or set to False to select for checkpoints with the
          lowest given error rate.
    """
    self._num_to_keep = num_to_keep
    self._save_dir = save_dir
    self._best_save_path = os.path.join(save_dir, 'best')
    self._periodical_save_path = os.path.join(save_dir, 'periodical')
    self._maximize = maximize
    self._keep_epoch = keep_epoch

    if not os.path.exists(self._best_save_path):
      os.makedirs(self._best_save_path)

    if self._keep_epoch:
      if not os.path.exists(self._periodical_save_path):
        os.makedirs(self._periodical_save_path)

    self.best_checkpoints_file = os.path.join(self._best_save_path, 'best_checkpoints')

  def _keep_ckpt(self, checkpoint_path, mode='best'):
    if mode == 'best':
      save_path = self._best_save_path
    elif mode == 'periodical':
      save_path = self._periodical_save_path
    else:
      raise AssertionError('mode must be "best" or "periodical"')

    for ckpt_file in glob.glob(os.path.join(self._save_dir, checkpoint_path) + "*"):
      file_name = os.path.basename(ckpt_file)
      copyfile(ckpt_file, os.path.join(save_path, file_name))

  def save(self, value, current_ckpt):
    """Updates the set of best checkpoints based on the given result.

    Args:
        value: The value by which to rank the checkpoint.
    """
    if tf.gfile.IsDirectory(current_ckpt):
      current_ckpt = tf.train.latest_checkpoint(current_ckpt)

    if current_ckpt.startswith('hdfs'):
      # Not support hdfs yet. @TODO
      return

    # The codes next is assumed to work only with the filename, so remove path except file name.
    current_ckpt = os.path.basename(current_ckpt)

    value = float(value)
    if not os.path.exists(self.best_checkpoints_file):
      self._save_best_checkpoints_file({current_ckpt: value})
      self._keep_ckpt(current_ckpt)
      return

    best_checkpoints = self._load_best_checkpoints_file()

    if len(best_checkpoints) < self._num_to_keep:
      best_checkpoints[current_ckpt] = value
      self._save_best_checkpoints_file(best_checkpoints)
      self._keep_ckpt(current_ckpt)
      return

    if self._maximize:
      should_save = not all(current_best >= value
                            for current_best in best_checkpoints.values())
    else:
      should_save = not all(current_best <= value
                            for current_best in best_checkpoints.values())
    if should_save:
      best_checkpoint_list = self._sort(best_checkpoints)

      worst_checkpoint = os.path.join(self._best_save_path,
                                      best_checkpoint_list.pop(-1)[0])
      tf.logging.debug(worst_checkpoint)
      self._remove_outdated_checkpoint_files(worst_checkpoint)

      best_checkpoints = dict(best_checkpoint_list)
      best_checkpoints[current_ckpt] = value
      self._save_best_checkpoints_file(best_checkpoints)
      self._keep_ckpt(current_ckpt)

    if self._keep_epoch:
      self._keep_ckpt(current_ckpt, mode='periodical')

  def _save_best_checkpoints_file(self, updated_best_checkpoints):
    with open(self.best_checkpoints_file, 'w') as f:
      json.dump(updated_best_checkpoints, f, indent=3)

  def _remove_outdated_checkpoint_files(self, worst_checkpoint):
    for ckpt_file in glob.glob(worst_checkpoint + '.*'):
      os.remove(ckpt_file)

  def _load_best_checkpoints_file(self):
    with open(self.best_checkpoints_file, 'r') as f:
      best_checkpoints = json.load(f)
    return best_checkpoints

  def _sort(self, best_checkpoints):
    best_checkpoints = [
      (ckpt, best_checkpoints[ckpt])
      for ckpt in sorted(best_checkpoints,
                         key=best_checkpoints.get,
                         reverse=self._maximize)
    ]
    return best_checkpoints
