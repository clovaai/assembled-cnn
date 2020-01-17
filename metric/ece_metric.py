from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

import tensorflow as tf

def get_tf_version():
  tf_version = tf.VERSION
  tf_major_version, tf_minor_version, _ = tf_version.split('.')
  return int(tf_major_version), int(tf_minor_version)

tf_major_version, tf_minor_version = get_tf_version()
if tf_major_version == 1 and tf_minor_version <= 12:
  from tensorflow.python.training import distribution_strategy_context
else:
  from tensorflow.python.distribute import distribution_strategy_context

def _remove_squeezable_dimensions(predictions, labels, weights):
  """Squeeze or expand last dim if needed.

  Squeezes last dim of `predictions` or `labels` if their rank differs by 1
  (using confusion_matrix.remove_squeezable_dimensions).
  Squeezes or expands last dim of `weights` if its rank differs by 1 from the
  new rank of `predictions`.

  If `weights` is scalar, it is kept scalar.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    labels: Optional label `Tensor` whose dimensions match `predictions`.
    weights: Optional weight scalar or `Tensor` whose dimensions match
      `predictions`.

  Returns:
    Tuple of `predictions`, `labels` and `weights`. Each of them possibly has
    the last dimension squeezed, `weights` could be extended by one dimension.
  """
  predictions = ops.convert_to_tensor(predictions)
  if labels is not None:
    labels, predictions = confusion_matrix.remove_squeezable_dimensions(
      labels, predictions)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  if weights is None:
    return predictions, labels, None

  weights = ops.convert_to_tensor(weights)
  weights_shape = weights.get_shape()
  weights_rank = weights_shape.ndims
  if weights_rank == 0:
    return predictions, labels, weights

  predictions_shape = predictions.get_shape()
  predictions_rank = predictions_shape.ndims
  if (predictions_rank is not None) and (weights_rank is not None):
    # Use static rank.
    if weights_rank - predictions_rank == 1:
      weights = array_ops.squeeze(weights, [-1])
    elif predictions_rank - weights_rank == 1:
      weights = array_ops.expand_dims(weights, [-1])
  else:
    # Use dynamic rank.
    weights_rank_tensor = array_ops.rank(weights)
    rank_diff = weights_rank_tensor - array_ops.rank(predictions)

    def _maybe_expand_weights():
      return control_flow_ops.cond(
        math_ops.equal(rank_diff, -1),
        lambda: array_ops.expand_dims(weights, [-1]), lambda: weights)

    # Don't attempt squeeze if it will fail based on static check.
    if ((weights_rank is not None) and
            (not weights_shape.dims[-1].is_compatible_with(1))):
      maybe_squeeze_weights = lambda: weights
    else:
      maybe_squeeze_weights = lambda: array_ops.squeeze(weights, [-1])

    def _maybe_adjust_weights():
      return control_flow_ops.cond(
        math_ops.equal(rank_diff, 1), maybe_squeeze_weights,
        _maybe_expand_weights)

    # If weights are scalar, do nothing. Otherwise, try to add or remove a
    # dimension to match predictions.
    weights = control_flow_ops.cond(
      math_ops.equal(weights_rank_tensor, 0), lambda: weights,
      _maybe_adjust_weights)
  return predictions, labels, weights

def _aggregate_across_towers(metrics_collections, metric_value_fn, *args):
  """Aggregate metric value across towers."""
  def fn(distribution, *a):
    """Call `metric_value_fn` in the correct control flow context."""
    if hasattr(distribution, '_outer_control_flow_context'):
      # If there was an outer context captured before this method was called,
      # then we enter that context to create the metric value op. If the
      # caputred context is `None`, ops.control_dependencies(None) gives the
      # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
      # captured context.
      # This special handling is needed because sometimes the metric is created
      # inside a while_loop (and perhaps a TPU rewrite context). But we don't
      # want the value op to be evaluated every step or on the TPU. So we
      # create it outside so that it can be evaluated at the end on the host,
      # once the update ops have been evaluted.

      # pylint: disable=protected-access
      if distribution._outer_control_flow_context is None:
        with ops.control_dependencies(None):
          metric_value = metric_value_fn(distribution, *a)
      else:
        distribution._outer_control_flow_context.Enter()
        metric_value = metric_value_fn(distribution, *a)
        distribution._outer_control_flow_context.Exit()
        # pylint: enable=protected-access
    else:
      metric_value = metric_value_fn(distribution, *a)
    if metrics_collections:
      ops.add_to_collections(metrics_collections, metric_value)
    return metric_value

  return distribution_strategy_context.get_tower_context().merge_call(fn, *args)

def _aggregate_across_replicas(metrics_collections, metric_value_fn, *args):
  """Aggregate metric value across replicas."""
  def fn(distribution, *a):
    """Call `metric_value_fn` in the correct control flow context."""
    if hasattr(distribution.extended, '_outer_control_flow_context'):
      # If there was an outer context captured before this method was called,
      # then we enter that context to create the metric value op. If the
      # caputred context is `None`, ops.control_dependencies(None) gives the
      # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
      # captured context.
      # This special handling is needed because sometimes the metric is created
      # inside a while_loop (and perhaps a TPU rewrite context). But we don't
      # want the value op to be evaluated every step or on the TPU. So we
      # create it outside so that it can be evaluated at the end on the host,
      # once the update ops have been evaluted.

      # pylint: disable=protected-access
      if distribution.extended._outer_control_flow_context is None:
        with ops.control_dependencies(None):
          metric_value = metric_value_fn(distribution, *a)
      else:
        distribution.extended._outer_control_flow_context.Enter()
        metric_value = metric_value_fn(distribution, *a)
        distribution.extended._outer_control_flow_context.Exit()
        # pylint: enable=protected-access
    else:
      metric_value = metric_value_fn(distribution, *a)
    if metrics_collections:
      ops.add_to_collections(metrics_collections, metric_value)
    return metric_value

  return distribution_strategy_context.get_replica_context().merge_call(
      fn, args=args)

def ece(conf, pred, label, num_thresholds=10,
        metrics_collections=None, updates_collections=None, name=None):
  """
  Calculate expected calibration error(ece).
  :param conf: The confidence values a `Tensor` of any shape.
  :param pred: The predicted values, a whose shape matches `conf`.
  :param label: The ground truth values, a `Tensor` whose shape matches `conf`.
  :param num_thresholds: The number of thresholds to use when discretizing reliability diagram.
  :param metrics_collections: An optional list of collections that `ece` should be added to.
  :param updates_collections: An optional list of collections that `update_op` should be added to.
  :param name: An optional variable_scope name.
  :return:
    ece: A scalar `Tensor` representing the current `ece` score
    update_op: An operation that increments the `ece` score
  """

  with variable_scope.variable_scope(
          name, 'ece', (conf, pred, label)):

    pred, label, conf = _remove_squeezable_dimensions(
      predictions=pred, labels=label, weights=conf)

    if pred.dtype != label.dtype:
      pred = math_ops.cast(pred, label.dtype)

    conf_2d = array_ops.reshape(conf, [-1, 1])
    pred_2d = array_ops.reshape(pred, [-1, 1])
    true_2d = array_ops.reshape(label, [-1, 1])

    # Use static shape if known.
    num_predictions = conf_2d.get_shape().as_list()[0]

    # Otherwise use dynamic shape.
    if num_predictions is None:
      num_predictions = array_ops.shape(conf_2d)[0]

    # To account for floating point imprecisions / avoid division by zero.
    epsilon = 1e-7
    thresholds = [(i + 1) * 1.0 / (num_thresholds)
                  for i in range(num_thresholds - 1)]
    thresholds = [0.0 - epsilon] + thresholds + [1.0 + epsilon]

    min_th = thresholds[0:num_thresholds]
    max_th = thresholds[1:num_thresholds + 1]

    min_thresh_tiled = array_ops.tile(
      array_ops.expand_dims(array_ops.constant(min_th), [1]),
      array_ops.stack([1, num_predictions]))
    max_thresh_tiled = array_ops.tile(
      array_ops.expand_dims(array_ops.constant(max_th), [1]),
      array_ops.stack([1, num_predictions]))

    conf_is_greater_th = math_ops.greater(
      array_ops.tile(array_ops.transpose(conf_2d), [num_thresholds, 1]),
      min_thresh_tiled)
    conf_is_less_equal_th = math_ops.less_equal(
      array_ops.tile(array_ops.transpose(conf_2d), [num_thresholds, 1]),
      max_thresh_tiled)

    # The `which_bin_conf_include` is num_thresholds x num_predictions (i, j) matrix
    # which if j-th prediction is included in i-th threshold, it is True
    which_bin_conf_include = math_ops.logical_and(conf_is_greater_th, conf_is_less_equal_th)

    # The `pred_2d_tiled` and `true_2d_tiled` is num_thresholds x num_predictions (i, j) matrix
    conf_2d_tiled = array_ops.tile(array_ops.transpose(conf_2d), [num_thresholds, 1])
    pred_2d_tiled = array_ops.tile(array_ops.transpose(pred_2d), [num_thresholds, 1])
    true_2d_tiled = array_ops.tile(array_ops.transpose(true_2d), [num_thresholds, 1])

    is_correct = math_ops.equal(pred_2d_tiled, true_2d_tiled)

    # The sum of correct answers count per threshold bin
    is_correct_per_bin = math_ops.reduce_sum(
      math_ops.to_float(math_ops.logical_and(is_correct, which_bin_conf_include)),
      1
    )

    # The sum of confidence per threshold bin
    conf_per_bin = math_ops.multiply(conf_2d_tiled, math_ops.cast(which_bin_conf_include, dtypes.float32))
    sum_conf_per_bin = math_ops.reduce_sum(conf_per_bin, 1)

    # The number of predictions per threshold bin
    len_per_bin = math_ops.reduce_sum(math_ops.to_float(which_bin_conf_include), 1)

    accumulated_correct = metrics_impl.metric_variable(
      [num_thresholds], dtypes.float32, name='accuracy_per_bin')
    accumulated_conf = metrics_impl.metric_variable(
      [num_thresholds], dtypes.float32, name='confidence_per_bin')
    accumulated_cnt = metrics_impl.metric_variable(
      [num_thresholds], dtypes.float32, name='count_per_bin')

    update_ops = {}
    update_ops['correct'] = state_ops.assign_add(accumulated_correct, is_correct_per_bin)
    update_ops['conf'] = state_ops.assign_add(accumulated_conf, sum_conf_per_bin)
    update_ops['cnt'] = state_ops.assign_add(accumulated_cnt, len_per_bin)

    values = {}
    values['correct'] = accumulated_correct
    values['conf'] = accumulated_conf
    values['cnt'] = accumulated_cnt

    def compute_ece(correct, conf, cnt, name):
      acc = math_ops.div(correct, epsilon + cnt, name='avg_acc_per_bin_' + name)
      avg_conf = math_ops.div(conf, epsilon + cnt, name='avg_conf_per_bin_' + name)
      abs_err = math_ops.abs(acc - avg_conf)
      sum_cnt = array_ops.reshape(math_ops.reduce_sum(cnt), [-1, ])
      sum_cnt_tiled = array_ops.tile(sum_cnt, [num_thresholds, ])
      weight = math_ops.div(cnt, sum_cnt_tiled)
      weighted_abs_err = math_ops.multiply(weight, abs_err)
      return math_ops.reduce_sum(weighted_abs_err)

    def ece_across_towers(_, correct, conf, cnt):
      ece = compute_ece(correct=correct, conf=conf,
                        cnt=cnt, name='value')
      return ece

    if tf_major_version == 1 and tf_minor_version <= 12:
      ece = _aggregate_across_towers(metrics_collections, ece_across_towers,
                                 values['correct'], values['conf'], values['cnt'])
    else:
      ece = _aggregate_across_replicas(metrics_collections, ece_across_towers,
                                 values['correct'], values['conf'], values['cnt'])

    update_op = compute_ece(correct=update_ops['correct'], conf=update_ops['conf'],
                            cnt=update_ops['cnt'], name='update')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return ece, update_op
