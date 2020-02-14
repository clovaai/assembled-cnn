# coding=utf8
# This code is adapted from the https://github.com/tensorflow/models/tree/master/official/r1/resnet.
# ==========================================================================================
# NAVER’s modifications are Copyright 2020 NAVER corp. All rights reserved.
# ==========================================================================================

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import math
import os

# pylint: disable=g-bad-import-order
from absl import flags

from metric import ece_metric
from nets import hparams_config
from nets.optimizer_setting import *
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers
from utils import checkpoint_utils
from utils import data_util
from utils import export_utils
from utils.hook_utils import *
from utils import config_utils
from metric import recall_metric
from functions import data_config, model_fns, input_fns
from nets import blocks
from losses import cls_losses

try:
  from hyperdash import Experiment

  install_hyperdash = True
except:
  install_hyperdash = False


# pylint: enable=g-bad-import-order
################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def resnet_model_fn(features, labels, num_classes, mode, model_class,
                    learning_rate_fn, keep_prob_fn, loss_filter_fn, p):
  global_step = tf.train.get_or_create_global_step()

  if keep_prob_fn:
    keep_prob = keep_prob_fn(global_step)
  else:
    keep_prob = 1.0

  model = model_class(p['resnet_size'], p['data_format'],
                      num_classes=num_classes,
                      resnet_version=p['resnet_version'],
                      zero_gamma=p['zero_gamma'],
                      use_se_block=p['use_se_block'],
                      use_sk_block=p['use_sk_block'],
                      no_downsample=p['no_downsample'],
                      anti_alias_filter_size=p['anti_alias_filter_size'],
                      anti_alias_type=p['anti_alias_type'],
                      bn_momentum=p['bn_momentum'],
                      embedding_size=p['embedding_size'],
                      pool_type=p['pool_type'],
                      bl_alpha=p['bl_alpha'],
                      bl_beta=p['bl_beta'],
                      dtype=p['dtype'],
                      loss_type=p['cls_loss_type'])

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  if mode != tf.estimator.ModeKeys.PREDICT:
    features_sup_may_mixuped = features["image"]
    if p['kd_temp'] > 0:
      # The kd label contains the supervised label and the teacher label.
      # And unlike usual, it comes in one-hot form.
      # Split this and apply a temperatured softmax to the teacher label.
      onehot_labels, teacher_logits = tf.split(labels, 2, axis=1)
      teacher_labels = tf.nn.softmax(teacher_logits / p['kd_temp'])
      labels = tf.argmax(onehot_labels, axis=1)
    else:
      onehot_labels = tf.one_hot(labels, model.num_classes)
      teacher_labels = None

    if p['mixup_type'] == 1 and mode == tf.estimator.ModeKeys.TRAIN:
      tf.logging.info('MIXUP on: {}'.format(p['mixup_type']))
      features_sup_may_mixuped, onehot_labels, teacher_labels = data_util.mixup(features_sup_may_mixuped, onehot_labels,
                                                                                keep_batch_size=False,
                                                                                y_t=teacher_labels)
    elif p['mixup_type'] == 2 and mode == tf.estimator.ModeKeys.TRAIN:
      tf.logging.info('MIXUP on: {}'.format(p['mixup_type']))
      features_sup_may_mixuped, onehot_labels, teacher_labels = data_util.mixup(features_sup_may_mixuped, onehot_labels,
                                                                                y_t=teacher_labels)
  else:
    onehot_labels = None
    features_sup_may_mixuped = features

  tf.summary.image('image', features_sup_may_mixuped, max_outputs=3)
  all_images = features_sup_may_mixuped

  assert all_images.dtype == p['dtype']

  sup_bsz = tf.shape(features_sup_may_mixuped)[0]

  all_logits = model(all_images, mode == tf.estimator.ModeKeys.TRAIN, False,
                     use_resnet_d=p['use_resnet_d'], keep_prob=keep_prob)
  all_logits = tf.cast(all_logits, tf.float32)
  logits = all_logits[:sup_bsz]

  predictions = {
    'classes': tf.argmax(logits, axis=1),
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
    'probabilities_sigmoid': tf.nn.sigmoid(logits, name='sigmoid_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      export_outputs={
        'predict': tf.estimator.export.PredictOutput(predictions)
      })

  ###########################
  # Calculate cross entropy.
  ###########################
  cross_entropy = cls_losses.get_sup_loss(logits, onehot_labels, global_step, num_classes, p)
  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  sup_prob = tf.nn.softmax(logits, axis=-1)
  tf.summary.scalar("sup/pred_prob", tf.reduce_mean(tf.reduce_max(sup_prob, axis=-1)))


  ##############################
  # Knowledge Distillation loss
  ##############################
  if p['kd_temp'] > 0:
    cross_entropy_kd = p['kd_temp'] * p['kd_temp'] * tf.losses.softmax_cross_entropy(
      logits=logits / p['kd_temp'], onehot_labels=teacher_labels)
    tf.identity(cross_entropy_kd, name='cross_entropy_kd')
    tf.summary.scalar('cross_entropy_kd', cross_entropy_kd)
  else:
    cross_entropy_kd = 0

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name

  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  ###############################
  # Add weight decay to the loss.
  ###############################
  l2_loss = p['weight_decay'] * tf.add_n(
    # loss is computed using fp32 for numerical stability.
    [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
     if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss + cross_entropy_kd

  if mode == tf.estimator.ModeKeys.TRAIN:

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate and keep_prob for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    tf.identity(keep_prob, name='dropblock_kp')
    tf.summary.scalar('dropblock_kp', keep_prob)

    train_op = get_train_op(loss=loss,
                            global_step=global_step,
                            learning_rate=learning_rate,
                            momentum=p['momentum'],
                            loss_scale=p['loss_scale'],
                            )
  else:
    train_op = None

  if p['mixup_type'] > 0 and mode == tf.estimator.ModeKeys.TRAIN:
    # mixup 시에는 레이블 두개 섞여서 정답이 되기 때문에 train accuracy를 구할 수 없다.
    metrics = None
    tf.identity(0, name='train_accuracy')
    tf.identity(0, name='train_accuracy_top_5')
    tf.identity(0, name='train_ece')
  else:
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])

    accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
                                                    targets=labels,
                                                    k=5,
                                                    name='top_5_op'))
    conf = tf.reduce_max(predictions['probabilities'], axis=1)
    ece = ece_metric.ece(conf, predictions['classes'], labels)
    metrics = {'accuracy': accuracy,
               'accuracy_top_5': accuracy_top_5,
               'ece': ece,
               }

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.identity(ece[1], name='train_ece')
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])
    tf.summary.scalar('train_ece', ece[1])

  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=metrics)


def resnet_main(flags_obj,
                model_function,
                input_function,
                dataset_name,
                shape=None,
                num_images=None,
                zeroshot_eval=False):
  model_helpers.apply_clean(flags.FLAGS)

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Create session config based on values of inter_op_parallelism_threads and
  # intra_op_parallelism_threads. Note that we default to having
  # allow_soft_placement = True, which is required for multi-GPU and not
  # harmful for other modes.
  session_config = config_utils.get_session_config(flags_obj)
  run_config = config_utils.get_run_config(flags_obj, flags_core, session_config, num_images['train'])

  def gen_estimator(period=None):
    resnet_size = int(flags_obj.resnet_size)
    data_format = flags_obj.data_format
    batch_size = flags_obj.batch_size
    resnet_version = int(flags_obj.resnet_version)
    loss_scale = flags_core.get_loss_scale(flags_obj)
    dtype_tf = flags_core.get_tf_dtype(flags_obj)
    num_epochs_per_decay = flags_obj.num_epochs_per_decay
    learning_rate_decay_factor = flags_obj.learning_rate_decay_factor
    end_learning_rate = flags_obj.end_learning_rate
    learning_rate_decay_type = flags_obj.learning_rate_decay_type
    weight_decay = flags_obj.weight_decay
    zero_gamma = flags_obj.zero_gamma
    lr_warmup_epochs = flags_obj.lr_warmup_epochs
    base_learning_rate = flags_obj.base_learning_rate
    use_resnet_d = flags_obj.use_resnet_d
    use_dropblock = flags_obj.use_dropblock
    dropblock_kp = [float(be) for be in flags_obj.dropblock_kp]
    label_smoothing = flags_obj.label_smoothing
    momentum = flags_obj.momentum
    bn_momentum = flags_obj.bn_momentum
    train_epochs = flags_obj.train_epochs
    piecewise_lr_boundary_epochs = [int(be) for be in flags_obj.piecewise_lr_boundary_epochs]
    piecewise_lr_decay_rates = [float(dr) for dr in flags_obj.piecewise_lr_decay_rates]
    use_ranking_loss = flags_obj.use_ranking_loss
    use_se_block = flags_obj.use_se_block
    use_sk_block = flags_obj.use_sk_block
    mixup_type = flags_obj.mixup_type
    dataset_name = flags_obj.dataset_name
    kd_temp = flags_obj.kd_temp
    no_downsample = flags_obj.no_downsample
    anti_alias_filter_size = flags_obj.anti_alias_filter_size
    anti_alias_type = flags_obj.anti_alias_type
    cls_loss_type = flags_obj.cls_loss_type
    logit_type = flags_obj.logit_type
    embedding_size = flags_obj.embedding_size
    pool_type = flags_obj.pool_type
    arc_s = flags_obj.arc_s
    arc_m = flags_obj.arc_m
    bl_alpha = flags_obj.bl_alpha
    bl_beta = flags_obj.bl_beta
    exp = None

    if install_hyperdash and flags_obj.use_hyperdash:
      exp = Experiment(flags_obj.model_dir.split("/")[-1])
      resnet_size = exp.param("resnet_size", int(flags_obj.resnet_size))
      batch_size = exp.param("batch_size", flags_obj.batch_size)
      exp.param("dtype", flags_obj.dtype)
      learning_rate_decay_type = exp.param("learning_rate_decay_type", flags_obj.learning_rate_decay_type)
      weight_decay = exp.param("weight_decay", flags_obj.weight_decay)
      zero_gamma = exp.param("zero_gamma", flags_obj.zero_gamma)
      lr_warmup_epochs = exp.param("lr_warmup_epochs", flags_obj.lr_warmup_epochs)
      base_learning_rate = exp.param("base_learning_rate", flags_obj.base_learning_rate)
      use_dropblock = exp.param("use_dropblock", flags_obj.use_dropblock)
      dropblock_kp = exp.param("dropblock_kp", [float(be) for be in flags_obj.dropblock_kp])
      piecewise_lr_boundary_epochs = exp.param("piecewise_lr_boundary_epochs",
                                               [int(be) for be in flags_obj.piecewise_lr_boundary_epochs])
      piecewise_lr_decay_rates = exp.param("piecewise_lr_decay_rates",
                                           [float(dr) for dr in flags_obj.piecewise_lr_decay_rates])
      mixup_type = exp.param("mixup_type", flags_obj.mixup_type)
      dataset_name = exp.param("dataset_name", flags_obj.dataset_name)
      exp.param("autoaugment_type", flags_obj.autoaugment_type)

    classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags_obj.model_dir,
      config=run_config,
      params={
        'resnet_size': resnet_size,
        'data_format': data_format,
        'batch_size': batch_size,
        'resnet_version': resnet_version,
        'loss_scale': loss_scale,
        'dtype': dtype_tf,
        'num_epochs_per_decay': num_epochs_per_decay,
        'learning_rate_decay_factor': learning_rate_decay_factor,
        'end_learning_rate': end_learning_rate,
        'learning_rate_decay_type': learning_rate_decay_type,
        'weight_decay': weight_decay,
        'zero_gamma': zero_gamma,
        'lr_warmup_epochs': lr_warmup_epochs,
        'base_learning_rate': base_learning_rate,
        'use_resnet_d': use_resnet_d,
        'use_dropblock': use_dropblock,
        'dropblock_kp': dropblock_kp,
        'label_smoothing': label_smoothing,
        'momentum': momentum,
        'bn_momentum': bn_momentum,
        'embedding_size': embedding_size,
        'train_epochs': train_epochs,
        'piecewise_lr_boundary_epochs': piecewise_lr_boundary_epochs,
        'piecewise_lr_decay_rates': piecewise_lr_decay_rates,
        'with_drawing_bbox': flags_obj.with_drawing_bbox,
        'use_ranking_loss': use_ranking_loss,
        'use_se_block': use_se_block,
        'use_sk_block': use_sk_block,
        'mixup_type': mixup_type,
        'kd_temp': kd_temp,
        'no_downsample': no_downsample,
        'dataset_name': dataset_name,
        'anti_alias_filter_size': anti_alias_filter_size,
        'anti_alias_type': anti_alias_type,
        'cls_loss_type': cls_loss_type,
        'logit_type': logit_type,
        'arc_s': arc_s,
        'arc_m': arc_m,
        'pool_type': pool_type,
        'bl_alpha': bl_alpha,
        'bl_beta': bl_beta,
        'train_steps': total_train_steps,

      })
    return classifier, exp

  run_params = {
    'batch_size': flags_obj.batch_size,
    'dtype': flags_core.get_tf_dtype(flags_obj),
    'resnet_size': flags_obj.resnet_size,
    'resnet_version': flags_obj.resnet_version,
    'synthetic_data': flags_obj.use_synthetic_data,
    'train_epochs': flags_obj.train_epochs,
  }
  if flags_obj.use_synthetic_data:
    dataset_name = dataset_name + '-synthetic'

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info('resnet', dataset_name, run_params,
                                test_id=flags_obj.benchmark_test_id)

  train_hooks = hooks_helper.get_train_hooks(
    flags_obj.hooks,
    model_dir=flags_obj.model_dir,
    batch_size=flags_obj.batch_size)

  def input_fn_train(num_epochs):
    return input_function(
      is_training=True,
      use_random_crop=flags_obj.training_random_crop,
      num_epochs=num_epochs,
      flags_obj=flags_obj
    )

  def input_fn_eval():
    return input_function(
      is_training=False,
      use_random_crop=False,
      num_epochs=1,
      flags_obj=flags_obj)

  ckpt_keeper = checkpoint_utils.CheckpointKeeper(
    save_dir=flags_obj.model_dir,
    num_to_keep=flags_obj.num_best_ckpt_to_keep,
    keep_epoch=flags_obj.keep_ckpt_every_eval,
    maximize=True
  )

  if zeroshot_eval:
    dataset = data_config.get_config(dataset_name)
    model = model_fns.Model(int(flags_obj.resnet_size),
                            flags_obj.data_format,
                            resnet_version=int(flags_obj.resnet_version),
                            num_classes=dataset.num_classes,
                            zero_gamma=flags_obj.zero_gamma,
                            use_se_block=flags_obj.use_se_block,
                            use_sk_block=flags_obj.use_sk_block,
                            no_downsample=flags_obj.no_downsample,
                            anti_alias_filter_size=flags_obj.anti_alias_filter_size,
                            anti_alias_type=flags_obj.anti_alias_type,
                            bn_momentum=flags_obj.bn_momentum,
                            embedding_size=flags_obj.embedding_size,
                            pool_type=flags_obj.pool_type,
                            bl_alpha=flags_obj.bl_alpha,
                            bl_beta=flags_obj.bl_beta,
                            dtype=flags_core.get_tf_dtype(flags_obj),
                            loss_type=flags_obj.cls_loss_type)
  def train_and_evaluate(hooks):
    tf.logging.info('Starting cycle: %d/%d', cycle_index, int(n_loops))

    if num_train_epochs:
      classifier.train(input_fn=lambda: input_fn_train(num_train_epochs), hooks=hooks,
                       steps=flags_obj.max_train_steps)

    tf.logging.info('Starting to evaluate.')

    if zeroshot_eval:
      tf.reset_default_graph()
      eval_results = recall_metric.recall_at_k(flags_obj, flags_core,
                                               input_fns.input_fn_ir_eval, model,
                                               num_images['validation'],
                                               eval_similarity=flags_obj.eval_similarity,
                                               return_embedding=True)
    else:
      eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                         steps=flags_obj.max_train_steps)


    return eval_results

  total_train_steps = flags_obj.train_epochs * int(num_images['train'] / flags_obj.batch_size)

  if flags_obj.eval_only or not flags_obj.train_epochs:
    # If --eval_only is set, perform a single loop with zero train epochs.
    schedule, n_loops = [0], 1
  elif flags_obj.export_only:
    schedule, n_loops = [], 0
  else:
    n_loops = math.ceil(flags_obj.train_epochs / flags_obj.epochs_between_evals)
    schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
    schedule[-1] = flags_obj.train_epochs - sum(schedule[:-1])  # over counting.

    schedule = config_utils.get_epoch_schedule(flags_obj, schedule, num_images)
    tf.logging.info('epoch schedule:')
    tf.logging.info(schedule)

  classifier, exp = gen_estimator()
  if flags_obj.pretrained_model_checkpoint_path:
    warm_start_hook = WarmStartHook(flags_obj.pretrained_model_checkpoint_path)
    train_hooks.append(warm_start_hook)

  for cycle_index, num_train_epochs in enumerate(schedule):
    eval_results = train_and_evaluate(train_hooks)
    if zeroshot_eval:
      metric = eval_results['recall_at_1']
    else:
      metric = eval_results['accuracy']
    ckpt_keeper.save(metric, flags_obj.model_dir)
    if exp:
      exp.metric("accuracy", metric)
    benchmark_logger.log_evaluation_result(eval_results)
    if model_helpers.past_stop_threshold(
            flags_obj.stop_threshold, metric):
      break
    if model_helpers.past_stop_threshold(
            total_train_steps, eval_results['global_step']):
      break

  if exp:
    exp.end()

  if flags_obj.export_dir is not None:
    export_utils.export_pb(flags_core, flags_obj, shape, classifier)

def define_resnet_flags(resnet_size_choices=None):
  """Add flags and validators for ResNet."""
  flags_core.define_base()
  flags_core.define_performance(num_parallel_calls=False)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  hparams_config.define_common_flags(flags)

  choice_kwargs = dict(
    name='resnet_size', short_name='rs', default='50',
    help=flags_core.help_wrap('The size of the ResNet model to use.'))

  if resnet_size_choices is None:
    flags.DEFINE_string(**choice_kwargs)
  else:
    flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)
