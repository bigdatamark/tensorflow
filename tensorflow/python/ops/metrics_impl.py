# [MODULE DOC]
# Graph-mode (TF1 style) streaming metrics implementation behind the v1 metrics API.
# Every public metric (mean, accuracy, auc, precision, recall, confusion-count metrics,
# top-k and sparse-at-k retrieval metrics, error metrics, sensitivity/specificity
# metrics, and so on) follows one pattern: it creates non-trainable local "metric"
# variables that accumulate statistics, returns a value tensor computed from those
# variables, and returns an update operation that folds a new batch of data into them.
# Most public functions are exported under the v1 metrics namespace via an export
# decorator, and they refuse to run (raise a runtime error) when eager execution is on.
# The module also holds private helpers for fixing up tensor ranks, building confusion
# matrices and counting conditions, selecting class ids, computing sparse top-k
# statistics, and aggregating values across replicas of a distribution strategy.
# Import-time side effects: only imports and the export/deprecation decorators are
# applied; no flags or global state are defined. Variables are registered in the local
# variables and metric variables graph collections at metric construction time.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of tf.metrics module."""

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export


def metric_variable(shape, dtype, validate_shape=True, name=None):
  # [FUNCTION DOC]
  # HELPER: creates the accumulator variables used by all metrics in this module.
  # Creates a zero-initialised, non-trainable variable and registers it in both the local
  # variables and metric variables graph collections. Inputs: shape (shape of the variable),
  # dtype (element type), validate_shape (optional, default true, whether the shape is
  # validated on assignment) and name (optional variable name). Returns the variable; under a
  # distribution strategy it is a sync-on-read container whose reads sum over all replicas
  # (synchronization on read, aggregation by sum). Side effects: adds the variable to graph
  # collections. Raises nothing itself.
  """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.

  If running in a `DistributionStrategy` context, the variable will be
  "sync on read". This means:

  *   The returned object will be a container with separate variables
      per replica of the model.

  *   When writing to the variable, e.g. using `assign_add` in a metric
      update, the update will be applied to the variable local to the
      replica.

  *   To get a metric's result value, we need to sum the variable values
      across the replicas before computing the final answer. Furthermore,
      the final answer should be computed once instead of in every
      replica. Both of these are accomplished by running the computation
      of the final result value inside
      `distribute_lib.get_replica_context().merge_call(fn)`.
      Inside the `merge_call()`, ops are only added to the graph once
      and access to a sync on read variable in a computation returns
      the sum across all replicas.

  Args:
    shape: Shape of the created variable.
    dtype: Type of the created variable.
    validate_shape: (Optional) Whether shape validation is enabled for
      the created variable.
    name: (Optional) String name of the created variable.

  Returns:
    A (non-trainable) variable initialized to zero, or if inside a
    `DistributionStrategy` scope a sync on read variable container.
  """
  # Note that synchronization "ON_READ" implies trainable=False.
  return variable_v1.VariableV1(
      lambda: array_ops.zeros(shape, dtype),
      trainable=False,
      collections=[
          ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
      ],
      validate_shape=validate_shape,
      synchronization=variables.VariableSynchronization.ON_READ,
      aggregation=variables.VariableAggregation.SUM,
      name=name)


def _remove_squeezable_dimensions(predictions, labels, weights):
  # [FUNCTION DOC]
  # HELPER: rank normalisation shared by most metrics before elementwise comparison.
  # Converts predictions to a tensor; if labels are given, drops a trailing size-1 dimension
  # from whichever of labels/predictions has one extra rank and asserts the shapes are
  # compatible. Weights (optional, may be nothing) are returned unchanged when absent or
  # scalar; otherwise a trailing dimension is squeezed or added so the weight rank matches
  # the prediction rank, using static ranks when known and conditional graph ops on dynamic
  # ranks otherwise. Returns a triple of predictions, labels (nothing if none were given) and
  # weights. Raises a shape error when labels and predictions are incompatible.
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
      # [FUNCTION DOC]
      # HELPER: dynamic-rank branch of the weight adjustment.
      # Returns the weights with a trailing dimension added when the weight rank is exactly
      # one less than the prediction rank (decided at graph run time), else the weights
      # unchanged. Takes no arguments; uses the enclosing weights and rank difference.
      return cond.cond(
          math_ops.equal(rank_diff, -1),
          lambda: array_ops.expand_dims(weights, [-1]), lambda: weights)

    # Don't attempt squeeze if it will fail based on static check.
    if ((weights_rank is not None) and
        (not weights_shape.dims[-1].is_compatible_with(1))):
      maybe_squeeze_weights = lambda: weights
    else:
      maybe_squeeze_weights = lambda: array_ops.squeeze(weights, [-1])

    def _maybe_adjust_weights():
      # [FUNCTION DOC]
      # HELPER: dynamic-rank branch of the weight adjustment.
      # At graph run time: if weights have one more rank than predictions, squeeze the last
      # dimension (or keep weights as is when a static check shows squeezing would fail);
      # otherwise fall through to the expand branch. Takes no arguments; returns a tensor.
      return cond.cond(
          math_ops.equal(rank_diff, 1), maybe_squeeze_weights,
          _maybe_expand_weights)

    # If weights are scalar, do nothing. Otherwise, try to add or remove a
    # dimension to match predictions.
    weights = cond.cond(
        math_ops.equal(weights_rank_tensor, 0), lambda: weights,
        _maybe_adjust_weights)
  return predictions, labels, weights


def _maybe_expand_labels(labels, predictions):
  # [FUNCTION DOC]
  # HELPER: aligns label rank with prediction rank for the sparse top-k metrics.
  # Inputs: labels (dense or sparse tensor) and predictions (dense tensor). Adds a trailing
  # dimension of size one to the labels when predictions have exactly one more rank. For
  # sparse labels the sparse shape is reshaped; for dense labels the static ranks are used
  # when known, otherwise a runtime conditional decides. Returns the possibly expanded labels.
  # Raises a value error when static ranks differ by anything other than zero or one.
  """If necessary, expand `labels` along last dimension to match `predictions`.

  Args:
    labels: `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN]. The latter implies
      num_labels=1, in which case the result is an expanded `labels` with shape
      [D1, ... DN, 1].
    predictions: `Tensor` with shape [D1, ... DN, num_classes].

  Returns:
    `labels` with the same rank as `predictions`.

  Raises:
    ValueError: if `labels` has invalid shape.
  """
  with ops.name_scope(None, 'expand_labels', (labels, predictions)) as scope:
    labels = sparse_tensor.convert_to_tensor_or_sparse_tensor(labels)

    # If sparse, expand sparse shape.
    if isinstance(labels, sparse_tensor.SparseTensor):
      return cond.cond(
          math_ops.equal(
              array_ops.rank(predictions),
              array_ops.size(labels.dense_shape) + 1),
          lambda: sparse_ops.sparse_reshape(  # pylint: disable=g-long-lambda
              labels,
              shape=array_ops.concat((labels.dense_shape, (1,)), 0),
              name=scope),
          lambda: labels)

    # Otherwise, try to use static shape.
    labels_rank = labels.get_shape().ndims
    if labels_rank is not None:
      predictions_rank = predictions.get_shape().ndims
      if predictions_rank is not None:
        if predictions_rank == labels_rank:
          return labels
        if predictions_rank == labels_rank + 1:
          return array_ops.expand_dims(labels, -1, name=scope)
        raise ValueError(
            f'Unexpected labels shape {labels.get_shape()} for predictions '
            f'shape {predictions.get_shape()}. Predictions rank should be the '
            'same rank as labels rank or labels rank plus one .')

    # Otherwise, use dynamic shape.
    return cond.cond(
        math_ops.equal(array_ops.rank(predictions),
                       array_ops.rank(labels) + 1),
        lambda: array_ops.expand_dims(labels, -1, name=scope), lambda: labels)


def _safe_scalar_div(numerator, denominator, name):
  # [FUNCTION DOC]
  # HELPER: division that yields zero instead of infinity or nan for a zero denominator.
  # Inputs: numerator and denominator tensors (rank at most one) and an op name. Verifies
  # both have rank of at most one (raises a shape error otherwise; the check results are
  # otherwise discarded), then returns the no-nan division result.
  """Divides two values, returning 0 if the denominator is 0.

  Args:
    numerator: A scalar `float64` `Tensor`.
    denominator: A scalar `float64` `Tensor`.
    name: Name for the returned op.

  Returns:
    0 if `denominator` == 0, else `numerator` / `denominator`
  """
  numerator.get_shape().with_rank_at_most(1)
  denominator.get_shape().with_rank_at_most(1)
  return math_ops.div_no_nan(numerator, denominator, name=name)


def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
  # [FUNCTION DOC]
  # HELPER: accumulator used by the mean-IoU metric.
  # Creates a float64 metric variable of size num_classes by num_classes, casts labels and
  # predictions to 64-bit integers, flattens labels, predictions and weights when their rank
  # exceeds one, builds the confusion matrix of this batch and adds it to the variable.
  # Weights are optional (may be nothing). Returns the pair of total confusion matrix
  # variable and the update operation. Side effect: creates a new metric variable.
  """Calculate a streaming confusion matrix.

  Calculates a confusion matrix. For estimation over a stream of data,
  the function creates an  `update_op` operation.

  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).

  Returns:
    total_cm: A `Tensor` representing the confusion matrix.
    update_op: An operation that increments the confusion matrix.
  """
  # Local variable to accumulate the predictions in the confusion matrix.
  total_cm = metric_variable(
      [num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')

  # Cast the type to int64 required by confusion_matrix_ops.
  predictions = math_ops.cast(predictions, dtypes.int64)
  labels = math_ops.cast(labels, dtypes.int64)
  num_classes = math_ops.cast(num_classes, dtypes.int64)

  # Flatten the input if its rank > 1.
  if predictions.get_shape().ndims > 1:
    predictions = array_ops.reshape(predictions, [-1])

  if labels.get_shape().ndims > 1:
    labels = array_ops.reshape(labels, [-1])

  if (weights is not None) and (weights.get_shape().ndims > 1):
    weights = array_ops.reshape(weights, [-1])

  # Accumulate the prediction to current confusion matrix.
  current_cm = confusion_matrix.confusion_matrix(
      labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
  update_op = state_ops.assign_add(total_cm, current_cm)
  return total_cm, update_op


def _aggregate_across_replicas(metrics_collections, metric_value_fn, *args):
  # [FUNCTION DOC]
  # HELPER: makes metric result values correct under distribution strategies.
  # Runs the value function once (not per replica) through a merge call on the current
  # replica context, passing the extra positional arguments (typically sync-on-read metric
  # variables). The result is added to the given graph collections if any are supplied.
  # Inputs: metrics_collections (list of collection names or nothing), metric_value_fn
  # (callback taking the distribution strategy followed by the variable values), and the
  # variadic arguments. Returns the metric value tensor. Needs an active replica context.
  """Aggregate metric value across replicas."""
  def fn(distribution, *a):
    # [FUNCTION DOC]
    # HELPER: merge-call body for the replica aggregation.
    # Calls the metric value function in the right control flow context: if the strategy
    # captured an outer control flow context, the call is made outside it (no dependencies
    # when the captured context is nothing, otherwise by entering and exiting that context)
    # so that the value op is created once outside loops or device rewrites. Afterwards the
    # value is added to the metric collections if given. Returns the metric value.
    """Call `metric_value_fn` in the correct control flow context."""
    if hasattr(distribution.extended, '_outer_control_flow_context'):
      # If there was an outer context captured before this method was called,
      # then we enter that context to create the metric value op. If the
      # captured context is `None`, ops.control_dependencies(None) gives the
      # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
      # captured context.
      # This special handling is needed because sometimes the metric is created
      # inside a while_loop (and perhaps a TPU rewrite context). But we don't
      # want the value op to be evaluated every step or on the TPU. So we
      # create it outside so that it can be evaluated at the end on the host,
      # once the update ops have been evaluated.

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

  return distribute_lib.get_replica_context().merge_call(
      fn, args=args)


@tf_export(v1=['metrics.mean'])
def mean(values,
         weights=None,
         metrics_collections=None,
         updates_collections=None,
         name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the update operation's value matches "mean_value", but
  # the returned value tensor is named mean; the migration table also names update_collections
  # inconsistently with the real parameter updates_collections.
  # Streaming weighted mean (v1 metrics API). Inputs: values (tensor of any shape, cast to
  # float32), weights (optional tensor, rank zero or same rank as values, broadcastable),
  # metrics_collections and updates_collections (optional lists of graph collection names to
  # receive the value tensor and update op), name (optional variable scope name). Creates two
  # local float32 variables, total and count; the update op adds the weighted sum of values to
  # total and the sum of weights (or the element count when weights are nothing) to count, and
  # returns total over count with negative counts clamped to zero and zero division giving zero.
  # Returns the pair of mean tensor (aggregated across replicas) and update op. Raises a
  # runtime error when eager execution is enabled; raises value errors for incompatible
  # weight shapes or non-list collections.
  """Computes the (weighted) mean of the given values.

  The `mean` function creates two local variables, `total` and `count`
  that are used to compute the average of `values`. This average is ultimately
  returned as `mean` which is an idempotent operation that simply divides
  `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean`.
  `update_op` increments `total` with the reduced sum of the product of `values`
  and `weights`, and it increments `count` with the reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `Tensor` of arbitrary dimensions.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `values`, and must be broadcastable to `values` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `values` dimension).
    metrics_collections: An optional list of collections that `mean`
      should be added to.
    updates_collections: An optional list of collections that `update_op`
      should be added to.
    name: An optional variable_scope name.

  Returns:
    mean: A `Tensor` representing the current mean, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_value`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
    RuntimeError: If eager execution is enabled.

  @compatibility(TF2)
  `tf.compat.v1.metrics.mean` is not compatible with eager
  execution or `tf.function`.
  Please use `tf.keras.metrics.Mean` instead for TF2 migration. After
  instantiating a `tf.keras.metrics.Mean` object, you can first call the
  `update_state()` method to record the new values, and then call the
  `result()` method to get the mean eagerly. You can also attach it to a
  Keras model with the `add_metric` method.  Please refer to the [migration
  guide](https://www.tensorflow.org/guide/migrate#new-style_metrics_and_losses)
  for more details.

  #### Structural Mapping to TF2

  Before:

  ```python
  mean, update_op = tf.compat.v1.metrics.mean(
    values=values,
    weights=weights,
    metrics_collections=metrics_collections,
    update_collections=update_collections,
    name=name)
  ```

  After:

  ```python
   m = tf.keras.metrics.Mean(
     name=name)

   m.update_state(
     values=values,
     sample_weight=weights)

   mean = m.result()
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name    | Note                       |
  | :-------------------- | :-------------- | :------------------------- |
  | `values`              | `values`        | In `update_state()` method |
  | `weights`             | `sample_weight` | In `update_state()` method |
  | `metrics_collections` | Not supported   | Metrics should be tracked  |
  :                       :                 : explicitly or with Keras   :
  :                       :                 : APIs, for example,         :
  :                       :                 : [add_metric][add_metric],  :
  :                       :                 : instead of via collections :
  | `updates_collections` | Not supported   | -                          |
  | `name`                | `name`          | In constructor             |

  [add_metric]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_metric


  #### Before & After Usage Example

  Before:

  >>> g = tf.Graph()
  >>> with g.as_default():
  ...   values = [1, 2, 3]
  ...   mean, update_op = tf.compat.v1.metrics.mean(values)
  ...   global_init = tf.compat.v1.global_variables_initializer()
  ...   local_init = tf.compat.v1.local_variables_initializer()
  >>> sess = tf.compat.v1.Session(graph=g)
  >>> sess.run([global_init, local_init])
  >>> sess.run(update_op)
  >>> sess.run(mean)
  2.0


  After:

  >>> m = tf.keras.metrics.Mean()
  >>> m.update_state([1, 2, 3])
  >>> m.result().numpy()
  2.0

  ```python
  # Used within Keras model
  model.add_metric(tf.keras.metrics.Mean()(values))
  ```

  @end_compatibility
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean is not supported when eager execution '
                       'is enabled.')

  with variable_scope.variable_scope(name, 'mean', (values, weights)):
    values = math_ops.cast(values, dtypes.float32)

    total = metric_variable([], dtypes.float32, name='total')
    count = metric_variable([], dtypes.float32, name='count')

    if weights is None:
      num_values = math_ops.cast(array_ops.size(values), dtypes.float32)
    else:
      values, _, weights = _remove_squeezable_dimensions(
          predictions=values, labels=None, weights=weights)
      weights = weights_broadcast_ops.broadcast_weights(
          math_ops.cast(weights, dtypes.float32), values)
      values = math_ops.multiply(values, weights)
      num_values = math_ops.reduce_sum(weights)

    update_total_op = state_ops.assign_add(total, math_ops.reduce_sum(values))
    with ops.control_dependencies([values]):
      update_count_op = state_ops.assign_add(count, num_values)

    def compute_mean(_, t, c):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for the mean metric.
      # Ignores the distribution argument; divides total t by count c (floored at zero),
      # giving zero when the count is zero. Returns the mean tensor named value.
      return math_ops.div_no_nan(t, math_ops.maximum(c, 0), name='value')

    mean_t = _aggregate_across_replicas(
        metrics_collections, compute_mean, total, count)
    update_op = math_ops.div_no_nan(
        update_total_op, math_ops.maximum(update_count_op, 0), name='update_op')

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_t, update_op


@tf_export(v1=['metrics.accuracy'])
def accuracy(labels,
             predictions,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the migration table in the docstring maps the predictions argument to the
  # Keras y_true name (should be y_pred), and its before-example passes logits first and
  # labels second, the reverse of this function's parameter order.
  # Streaming accuracy (v1 metrics API). Inputs: labels (ground truth), predictions (any
  # shape, same shape as labels after squeezing), weights (optional), metrics_collections and
  # updates_collections (optional lists of collection names), name (optional scope, defaults
  # to accuracy). Squeezes trailing dimensions, checks shape compatibility, casts predictions
  # to the labels type, computes elementwise equality as float values and delegates to the
  # streaming mean. Returns the pair of accuracy tensor and update op. Raises a runtime error
  # in eager mode, and value errors for mismatched shapes or bad collection arguments.
  """Calculates how often `predictions` matches `labels`.

  The `accuracy` function creates two local variables, `total` and
  `count` that are used to compute the frequency with which `predictions`
  matches `labels`. This frequency is ultimately returned as `accuracy`: an
  idempotent operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `accuracy`.
  Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
  where the corresponding elements of `predictions` and `labels` match and 0.0
  otherwise. Then `update_op` increments `total` with the reduced sum of the
  product of `weights` and `is_correct`, and it increments `count` with the
  reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose shape matches
      `predictions`.
    predictions: The predicted values, a `Tensor` of any shape.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `accuracy` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    accuracy: A `Tensor` representing the accuracy, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `accuracy`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.

  @compatibility(TF2)
  `tf.compat.v1.metrics.accuracy` is not compatible with eager
  execution or `tf.function`.
  Please use `tf.keras.metrics.Accuracy` instead for TF2 migration. After
  instantiating a `tf.keras.metrics.Accuracy` object, you can first call the
  `update_state()` method to record the prediction/labels, and then call the
  `result()` method to get the accuracy eagerly. You can also attach it to a
  Keras model when calling the `compile` method. Please refer to [this
  guide](https://www.tensorflow.org/guide/migrate#new-style_metrics_and_losses)
  for more details.

  #### Structural Mapping to Native TF2

  Before:

  ```python
  accuracy, update_op = tf.compat.v1.metrics.accuracy(
    labels=labels,
    predictions=predictions,
    weights=weights,
    metrics_collections=metrics_collections,
    update_collections=update_collections,
    name=name)
  ```

  After:

  ```python
   m = tf.keras.metrics.Accuracy(
     name=name,
     dtype=None)

   m.update_state(
   y_true=labels,
   y_pred=predictions,
   sample_weight=weights)

   accuracy = m.result()
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name    | Note                       |
  | :-------------------- | :-------------- | :------------------------- |
  | `label`               | `y_true`        | In `update_state()` method |
  | `predictions`         | `y_true`        | In `update_state()` method |
  | `weights`             | `sample_weight` | In `update_state()` method |
  | `metrics_collections` | Not supported   | Metrics should be tracked  |
  :                       :                 : explicitly or with Keras   :
  :                       :                 : APIs, for example,         :
  :                       :                 : [add_metric][add_metric],  :
  :                       :                 : instead of via collections :
  | `updates_collections` | Not supported   | -                          |
  | `name`                | `name`          | In constructor             |

  [add_metric]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_metric


  #### Before & After Usage Example

  Before:

  >>> g = tf.Graph()
  >>> with g.as_default():
  ...   logits = [1, 2, 3]
  ...   labels = [0, 2, 3]
  ...   acc, acc_op = tf.compat.v1.metrics.accuracy(logits, labels)
  ...   global_init = tf.compat.v1.global_variables_initializer()
  ...   local_init = tf.compat.v1.local_variables_initializer()
  >>> sess = tf.compat.v1.Session(graph=g)
  >>> sess.run([global_init, local_init])
  >>> print(sess.run([acc, acc_op]))
  [0.0, 0.66667]


  After:

  >>> m = tf.keras.metrics.Accuracy()
  >>> m.update_state([1, 2, 3], [0, 2, 3])
  >>> m.result().numpy()
  0.66667

  ```python
  # Used within Keras model
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.Accuracy()])
  ```

  @end_compatibility
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.accuracy is not supported when eager '
                       'execution is enabled.')

  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions=predictions, labels=labels, weights=weights)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  if labels.dtype != predictions.dtype:
    predictions = math_ops.cast(predictions, labels.dtype)
  is_correct = math_ops.cast(
      math_ops.equal(predictions, labels), dtypes.float32)
  return mean(is_correct, weights, metrics_collections, updates_collections,
              name or 'accuracy')


def _confusion_matrix_at_thresholds(labels,
                                    predictions,
                                    thresholds,
                                    weights=None,
                                    includes=None):
  # [FUNCTION DOC]
  # HELPER: shared engine for the threshold based metrics (auc, precision/recall at
  # thresholds, sensitivity/specificity, and the count metrics at thresholds).
  # Inputs: labels (cast to boolean), predictions (floating tensor with values in zero to
  # one, asserted at graph run time), thresholds (list or tuple of floats), weights (optional)
  # and includes (optional tuple of keys among tp, fn, tn, fp; default all four). For each
  # requested key a float32 metric variable of length equal to the number of thresholds is
  # created and an op adding the (weighted) count of that outcome per threshold is built;
  # a prediction counts as positive when strictly above the threshold. Returns two mappings
  # keyed by the requested names: the variables, and their update ops. Raises a value error
  # for an unknown key in includes or for incompatible shapes.
  """Computes true_positives, false_negatives, true_negatives, false_positives.

  This function creates up to four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives`.
  `true_positive[i]` is defined as the total weight of values in `predictions`
  above `thresholds[i]` whose corresponding entry in `labels` is `True`.
  `false_negatives[i]` is defined as the total weight of values in `predictions`
  at most `thresholds[i]` whose corresponding entry in `labels` is `True`.
  `true_negatives[i]` is defined as the total weight of values in `predictions`
  at most `thresholds[i]` whose corresponding entry in `labels` is `False`.
  `false_positives[i]` is defined as the total weight of values in `predictions`
  above `thresholds[i]` whose corresponding entry in `labels` is `False`.

  For estimation of these metrics over a stream of data, for each metric the
  function respectively creates an `update_op` operation that updates the
  variable and returns its value.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    includes: Tuple of keys to return, from 'tp', 'fn', 'tn', fp'. If `None`,
        default to all four.

  Returns:
    values: Dict of variables of shape `[len(thresholds)]`. Keys are from
        `includes`.
    update_ops: Dict of operations that increments the `values`. Keys are from
        `includes`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `includes` contains invalid keys.
  """
  all_includes = ('tp', 'fn', 'tn', 'fp')
  if includes is None:
    includes = all_includes
  else:
    for include in includes:
      if include not in all_includes:
        raise ValueError(f'Invalid key: {include}')

  with ops.control_dependencies([
      check_ops.assert_greater_equal(
          predictions,
          math_ops.cast(0.0, dtype=predictions.dtype),
          message='predictions must be in [0, 1]'),
      check_ops.assert_less_equal(
          predictions,
          math_ops.cast(1.0, dtype=predictions.dtype),
          message='predictions must be in [0, 1]')
  ]):
    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtypes.float32),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)

  num_thresholds = len(thresholds)

  # Reshape predictions and labels.
  predictions_2d = array_ops.reshape(predictions, [-1, 1])
  labels_2d = array_ops.reshape(
      math_ops.cast(labels, dtype=dtypes.bool), [1, -1])

  # Use static shape if known.
  num_predictions = predictions_2d.get_shape().as_list()[0]

  # Otherwise use dynamic shape.
  if num_predictions is None:
    num_predictions = array_ops.shape(predictions_2d)[0]
  thresh_tiled = array_ops.tile(
      array_ops.expand_dims(array_ops.constant(thresholds), [1]),
      array_ops_stack.stack([1, num_predictions]))

  # Tile the predictions after thresholding them across different thresholds.
  pred_is_pos = math_ops.greater(
      array_ops.tile(array_ops.transpose(predictions_2d), [num_thresholds, 1]),
      thresh_tiled)
  if ('fn' in includes) or ('tn' in includes):
    pred_is_neg = math_ops.logical_not(pred_is_pos)

  # Tile labels by number of thresholds
  label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])
  if ('fp' in includes) or ('tn' in includes):
    label_is_neg = math_ops.logical_not(label_is_pos)

  if weights is not None:
    weights = weights_broadcast_ops.broadcast_weights(
        math_ops.cast(weights, dtypes.float32), predictions)
    weights_tiled = array_ops.tile(
        array_ops.reshape(weights, [1, -1]), [num_thresholds, 1])
    thresh_tiled.get_shape().assert_is_compatible_with(
        weights_tiled.get_shape())
  else:
    weights_tiled = None

  values = {}
  update_ops = {}

  if 'tp' in includes:
    true_p = metric_variable(
        [num_thresholds], dtypes.float32, name='true_positives')
    is_true_positive = math_ops.cast(
        math_ops.logical_and(label_is_pos, pred_is_pos), dtypes.float32)
    if weights_tiled is not None:
      is_true_positive *= weights_tiled
    update_ops['tp'] = state_ops.assign_add(true_p,
                                            math_ops.reduce_sum(
                                                is_true_positive, 1))
    values['tp'] = true_p

  if 'fn' in includes:
    false_n = metric_variable(
        [num_thresholds], dtypes.float32, name='false_negatives')
    is_false_negative = math_ops.cast(
        math_ops.logical_and(label_is_pos, pred_is_neg), dtypes.float32)
    if weights_tiled is not None:
      is_false_negative *= weights_tiled
    update_ops['fn'] = state_ops.assign_add(false_n,
                                            math_ops.reduce_sum(
                                                is_false_negative, 1))
    values['fn'] = false_n

  if 'tn' in includes:
    true_n = metric_variable(
        [num_thresholds], dtypes.float32, name='true_negatives')
    is_true_negative = math_ops.cast(
        math_ops.logical_and(label_is_neg, pred_is_neg), dtypes.float32)
    if weights_tiled is not None:
      is_true_negative *= weights_tiled
    update_ops['tn'] = state_ops.assign_add(true_n,
                                            math_ops.reduce_sum(
                                                is_true_negative, 1))
    values['tn'] = true_n

  if 'fp' in includes:
    false_p = metric_variable(
        [num_thresholds], dtypes.float32, name='false_positives')
    is_false_positive = math_ops.cast(
        math_ops.logical_and(label_is_neg, pred_is_pos), dtypes.float32)
    if weights_tiled is not None:
      is_false_positive *= weights_tiled
    update_ops['fp'] = state_ops.assign_add(false_p,
                                            math_ops.reduce_sum(
                                                is_false_positive, 1))
    values['fp'] = false_p

  return values, update_ops


def _aggregate_variable(v, collections):
  # [FUNCTION DOC]
  # HELPER: reads a metric variable in a way that is correct under distribution strategies.
  # Inputs: v (a metric variable, possibly a sync-on-read container) and collections (list of
  # collection names or nothing to which the read value is added). Returns the aggregated
  # (summed across replicas) value of the variable via the replica aggregation helper.
  f = lambda distribution, value: distribution.extended.read_var(value)
  return _aggregate_across_replicas(collections, f, v)


@tf_export(v1=['metrics.auc'])
@deprecated(None,
            'The value of AUC returned by this may race with the update so '
            'this is deprecated. Please use tf.keras.metrics.AUC instead.')
def auc(labels,
        predictions,
        weights=None,
        num_thresholds=200,
        metrics_collections=None,
        updates_collections=None,
        curve='ROC',
        name=None,
        summation_method='trapezoidal',
        thresholds=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the default summation method (trapezoidal) is documented as the default, yet
  # for the precision-recall curve it logs a warning at every call saying it yields incorrect
  # areas; the function is also marked deprecated because the value may race with the update.
  # Streaming approximate area under the ROC or precision-recall curve. Inputs: labels (cast to
  # boolean), predictions (floats in zero to one), weights (optional), num_thresholds (default
  # 200, ignored when thresholds is given), metrics_collections / updates_collections (optional
  # lists of collection names), curve (ROC, the default, or PR), name (optional scope),
  # summation_method (trapezoidal, careful_interpolation, minoring or majoring) and thresholds
  # (optional list of floats; sorted, then padded with an entry just below zero and just above
  # one). Builds the threshold confusion counts, then sums the areas of the segments between
  # successive thresholds using the chosen rule (recall and false positive rate for ROC,
  # precision versus recall for PR, with a closed-form interpolation for careful_interpolation).
  # Returns the pair of auc tensor and update op (the update op value is computed from the
  # post-update counts). Raises a runtime error in eager mode and value errors for an unknown
  # curve, an invalid summation method, or mismatched shapes.
  """Computes the approximate AUC via a Riemann sum.

  The `auc` function creates four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives` that are used to
  compute the AUC. To discretize the AUC curve, a linearly spaced set of
  thresholds is used to compute pairs of recall and precision values. The area
  under the ROC-curve is therefore computed using the height of the recall
  values by the false positive rate, while the area under the PR-curve is the
  computed using the height of the precision values by the recall.

  This value is ultimately returned as `auc`, an idempotent operation that
  computes the area under a discretized curve of precision versus recall values
  (computed using the aforementioned variables). The `num_thresholds` variable
  controls the degree of discretization with larger numbers of thresholds more
  closely approximating the true AUC. The quality of the approximation may vary
  dramatically depending on `num_thresholds`.

  For best results, `predictions` should be distributed approximately uniformly
  in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC
  approximation may be poor if this is not the case. Setting `summation_method`
  to 'minoring' or 'majoring' can help quantify the error in the approximation
  by providing lower or upper bound estimate of the AUC. The `thresholds`
  parameter can be used to manually specify thresholds which split the
  predictions more evenly.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `auc`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use when discretizing the roc
      curve.
    metrics_collections: An optional list of collections that `auc` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    curve: Specifies the name of the curve to be computed, 'ROC' [default] or
      'PR' for the Precision-Recall-curve.
    name: An optional variable_scope name.
    summation_method: Specifies the Riemann summation method used
      (https://en.wikipedia.org/wiki/Riemann_sum): 'trapezoidal' [default] that
      applies the trapezoidal rule; 'careful_interpolation', a variant of it
      differing only by a more correct interpolation scheme for PR-AUC -
      interpolating (true/false) positives but not the ratio that is precision;
      'minoring' that applies left summation for increasing intervals and right
      summation for decreasing intervals; 'majoring' that does the opposite.
      Note that 'careful_interpolation' is strictly preferred to 'trapezoidal'
      (to be deprecated soon) as it applies the same method for ROC, and a
      better one (see Davis & Goadrich 2006 for details) for the PR curve.
    thresholds: An optional list of floating point values to use as the
      thresholds for discretizing the curve. If set, the `num_thresholds`
      parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
      equal to {-epsilon, 1+epsilon} for a small positive epsilon value will be
      automatically included with these to correctly handle predictions equal to
       exactly 0 or 1.

  Returns:
    auc: A scalar `Tensor` representing the current area-under-curve.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `auc`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.auc is not supported when eager execution '
                       'is enabled.')

  with variable_scope.variable_scope(name, 'auc',
                                     (labels, predictions, weights)):
    if curve != 'ROC' and curve != 'PR':
      raise ValueError(f'Curve must be either ROC or PR. Curve {curve} is '
                       'unknown.')

    kepsilon = 1e-7  # To account for floating point imprecisions.
    if thresholds is not None:
      # If specified, use the supplied thresholds.
      thresholds = sorted(thresholds)
      num_thresholds = len(thresholds) + 2
    else:
      # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
      # (0, 1).
      thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                    for i in range(num_thresholds - 2)]

    # Add an endpoint "threshold" below zero and above one for either threshold
    # method.
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

    values, update_ops = _confusion_matrix_at_thresholds(
        labels, predictions, thresholds, weights)

    # Add epsilons to avoid dividing by 0.
    epsilon = 1.0e-6

    def interpolate_pr_auc(tp, fp, fn):
      # [FUNCTION DOC]
      # HELPER: PR-curve area for the careful_interpolation summation method of auc.
      # Inputs: tp, fp, fn count vectors (one entry per threshold). Assumes precision varies
      # via linearly varying true and false positive weights inside each threshold interval
      # and sums the closed-form area per interval, normalised by the total positive weight;
      # divisions are zero-safe. Returns a scalar tensor with the approximated area.
      """Interpolation formula inspired by section 4 of (Davis et al., 2006).

      Note here we derive & use a closed formula not present in the paper
      - as follows:
      Modeling all of TP (true positive weight),
      FP (false positive weight) and their sum P = TP + FP (positive weight)
      as varying linearly within each interval [A, B] between successive
      thresholds, we get
        Precision = (TP_A + slope * (P - P_A)) / P
      with slope = dTP / dP = (TP_B - TP_A) / (P_B - P_A).
      The area within the interval is thus (slope / total_pos_weight) times
        int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
        int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}
      where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in
        int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)
      Bringing back the factor (slope / total_pos_weight) we'd put aside, we get
         slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight
      where dTP == TP_B - TP_A.
      Note that when P_A == 0 the above calculation simplifies into
        int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)
      which is really equivalent to imputing constant precision throughout the
      first bucket having >0 true positives.

      Args:
        tp: true positive counts
        fp: false positive counts
        fn: false negative counts

      Returns:
        pr_auc: an approximation of the area under the P-R curve.

      References:
        The Relationship Between Precision-Recall and ROC Curves:
          [Davis et al., 2006](https://dl.acm.org/citation.cfm?id=1143874)
          ([pdf](https://www.biostat.wisc.edu/~page/rocpr.pdf))
      """
      dtp = tp[:num_thresholds - 1] - tp[1:]
      p = tp + fp
      prec_slope = math_ops.div_no_nan(
          dtp,
          math_ops.maximum(p[:num_thresholds - 1] - p[1:], 0),
          name='prec_slope')
      intercept = tp[1:] - math_ops.multiply(prec_slope, p[1:])
      safe_p_ratio = array_ops.where(
          math_ops.logical_and(p[:num_thresholds - 1] > 0, p[1:] > 0),
          math_ops.div_no_nan(
              p[:num_thresholds - 1],
              math_ops.maximum(p[1:], 0),
              name='recall_relative_ratio'), array_ops.ones_like(p[1:]))
      return math_ops.reduce_sum(
          math_ops.div_no_nan(
              prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)),
              math_ops.maximum(tp[1:] + fn[1:], 0),
              name='pr_auc_increment'),
          name='interpolate_pr_auc')

    def compute_auc(tp, fn, tn, fp, name):
      # [FUNCTION DOC]
      # HELPER: turns confusion counts into the curve area for auc.
      # Inputs: per-threshold true positive, false negative, true negative and false positive
      # vectors, and a name for the result op. Uses the enclosing curve and summation method.
      # For PR with trapezoidal it logs a warning; with careful_interpolation it delegates to
      # the interpolation helper. Otherwise computes recall and (false positive rate or
      # precision) with a small epsilon and sums trapezoids, lower sums or upper sums.
      # Returns a scalar tensor; raises a value error for an unknown summation method.
      """Computes the roc-auc or pr-auc based on confusion counts."""
      if curve == 'PR':
        if summation_method == 'trapezoidal':
          logging.warning(
              'Trapezoidal rule is known to produce incorrect PR-AUCs; '
              'please switch to "careful_interpolation" instead.')
        elif summation_method == 'careful_interpolation':
          # This one is a bit tricky and is handled separately.
          return interpolate_pr_auc(tp, fp, fn)
      rec = math_ops.divide(tp + epsilon, tp + fn + epsilon)
      if curve == 'ROC':
        fp_rate = math_ops.divide(fp, fp + tn + epsilon)
        x = fp_rate
        y = rec
      else:  # curve == 'PR'.
        prec = math_ops.divide(tp + epsilon, tp + fp + epsilon)
        x = rec
        y = prec
      if summation_method in ('trapezoidal', 'careful_interpolation'):
        # Note that the case ('PR', 'careful_interpolation') has been handled
        # above.
        return math_ops.reduce_sum(
            math_ops.multiply(x[:num_thresholds - 1] - x[1:],
                              (y[:num_thresholds - 1] + y[1:]) / 2.),
            name=name)
      elif summation_method == 'minoring':
        return math_ops.reduce_sum(
            math_ops.multiply(x[:num_thresholds - 1] - x[1:],
                              math_ops.minimum(y[:num_thresholds - 1], y[1:])),
            name=name)
      elif summation_method == 'majoring':
        return math_ops.reduce_sum(
            math_ops.multiply(x[:num_thresholds - 1] - x[1:],
                              math_ops.maximum(y[:num_thresholds - 1], y[1:])),
            name=name)
      else:
        raise ValueError(f'Invalid summation_method: {summation_method} '
                         'summation_method should be \'trapezoidal\', '
                         '\'careful_interpolation\', \'minoring\', or '
                         '\'majoring\'.')

    # sum up the areas of all the trapeziums
    def compute_auc_value(_, values):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for auc.
      # Ignores the distribution argument and computes the area from the mapping of count
      # variables (keys tp, fn, tn, fp), naming the result value. Returns a scalar tensor.
      return compute_auc(values['tp'], values['fn'], values['tn'], values['fp'],
                         'value')

    auc_value = _aggregate_across_replicas(
        metrics_collections, compute_auc_value, values)
    update_op = compute_auc(update_ops['tp'], update_ops['fn'],
                            update_ops['tn'], update_ops['fp'], 'update_op')

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return auc_value, update_op


@tf_export(v1=['metrics.mean_absolute_error'])
def mean_absolute_error(labels,
                        predictions,
                        weights=None,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None):
  # [FUNCTION DOC]
  # Streaming mean absolute error. Inputs: labels and predictions (same shape after
  # squeezing), weights (optional), metrics_collections / updates_collections (optional lists
  # of collection names), name (optional, defaults to mean_absolute_error). Computes the
  # absolute difference elementwise and feeds it to the streaming mean. Returns the pair of
  # value tensor and update op. Raises a runtime error in eager mode and value errors for
  # mismatched shapes or bad collection arguments.
  """Computes the mean absolute error between the labels and predictions.

  The `mean_absolute_error` function creates two local variables,
  `total` and `count` that are used to compute the mean absolute error. This
  average is weighted by `weights`, and it is ultimately returned as
  `mean_absolute_error`: an idempotent operation that simply divides `total` by
  `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `mean_absolute_error`. Internally, an `absolute_errors` operation computes the
  absolute value of the differences between `predictions` and `labels`. Then
  `update_op` increments `total` with the reduced sum of the product of
  `weights` and `absolute_errors`, and it increments `count` with the reduced
  sum of `weights`

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` of arbitrary shape.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `mean_absolute_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    mean_absolute_error: A `Tensor` representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_absolute_error`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_absolute_error is not supported '
                       'when eager execution is enabled.')

  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions=predictions, labels=labels, weights=weights)
  absolute_errors = math_ops.abs(predictions - labels)
  return mean(absolute_errors, weights, metrics_collections,
              updates_collections, name or 'mean_absolute_error')


@tf_export(v1=['metrics.mean_cosine_distance'])
def mean_cosine_distance(labels,
                         predictions,
                         dim,
                         weights=None,
                         metrics_collections=None,
                         updates_collections=None,
                         name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the result is one minus the mean of the plain dot product along the given
  # dimension; it is only a cosine distance when labels and predictions are already unit
  # normalised, which is neither checked nor enforced.
  # Streaming mean cosine distance. Inputs: labels, predictions (same shape), dim (axis along
  # which the dot product is taken, kept as size one), weights (optional; dimension dim must be
  # one), metrics_collections / updates_collections (optional lists), name (optional).
  # Multiplies predictions with labels, sums along dim, takes the streaming mean and subtracts
  # it (and the update value) from one; collections are filled after this transformation.
  # Returns the pair of distance tensor and update op. Raises a runtime error in eager mode.
  """Computes the cosine distance between the labels and predictions.

  The `mean_cosine_distance` function creates two local variables,
  `total` and `count` that are used to compute the average cosine distance
  between `predictions` and `labels`. This average is weighted by `weights`,
  and it is ultimately returned as `mean_distance`, which is an idempotent
  operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `mean_distance`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of arbitrary shape.
    predictions: A `Tensor` of the same shape as `labels`.
    dim: The dimension along which the cosine distance is computed.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension). Also,
      dimension `dim` must be `1`.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    mean_distance: A `Tensor` representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_cosine_distance is not supported when '
                       'eager execution is enabled.')

  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions=predictions, labels=labels, weights=weights)
  radial_diffs = math_ops.multiply(predictions, labels)
  radial_diffs = math_ops.reduce_sum(
      radial_diffs, axis=[
          dim,
      ], keepdims=True)
  mean_distance, update_op = mean(radial_diffs, weights, None, None, name or
                                  'mean_cosine_distance')
  mean_distance = math_ops.subtract(1.0, mean_distance)
  update_op = math_ops.subtract(1.0, update_op)

  if metrics_collections:
    ops.add_to_collections(metrics_collections, mean_distance)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return mean_distance, update_op


@tf_export(v1=['metrics.mean_per_class_accuracy'])
def mean_per_class_accuracy(labels,
                            predictions,
                            num_classes,
                            weights=None,
                            metrics_collections=None,
                            updates_collections=None,
                            name=None):
  # [FUNCTION DOC]
  # Streaming mean of per-class accuracies. Inputs: labels and predictions (integer class
  # indices, flattened when rank exceeds one; predictions are cast to the labels type),
  # num_classes (size of the two per-class accumulators), weights (optional, flattened),
  # metrics_collections / updates_collections (optional lists), name (optional scope, default
  # mean_accuracy). Creates float32 total and count variables of length num_classes and uses
  # scatter-add keyed by label: total gets the weights, count gets the weighted correctness.
  # The value is the mean over classes of count divided by total (zero-safe); the update op
  # is the same ratio computed from the updated variables. Returns the pair of value tensor
  # and update op. Raises a runtime error in eager mode and shape errors on mismatch.
  """Calculates the mean of the per-class accuracies.

  Calculates the accuracy for each class, then takes the mean of that.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates the accuracy of each class and returns
  them.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since two variables with shape =
      [num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `mean_per_class_accuracy'
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.

  Returns:
    mean_accuracy: A `Tensor` representing the mean per class accuracy.
    update_op: An operation that updates the accuracy tensor.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_per_class_accuracy is not supported '
                       'when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'mean_accuracy',
                                     (predictions, labels, weights)):
    labels = math_ops.cast(labels, dtypes.int64)

    # Flatten the input if its rank > 1.
    if labels.get_shape().ndims > 1:
      labels = array_ops.reshape(labels, [-1])

    if predictions.get_shape().ndims > 1:
      predictions = array_ops.reshape(predictions, [-1])

    # Check if shape is compatible.
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    total = metric_variable([num_classes], dtypes.float32, name='total')
    count = metric_variable([num_classes], dtypes.float32, name='count')

    ones = array_ops.ones([array_ops.size(labels)], dtypes.float32)

    if labels.dtype != predictions.dtype:
      predictions = math_ops.cast(predictions, labels.dtype)
    is_correct = math_ops.cast(
        math_ops.equal(predictions, labels), dtypes.float32)

    if weights is not None:
      if weights.get_shape().ndims > 1:
        weights = array_ops.reshape(weights, [-1])
      weights = math_ops.cast(weights, dtypes.float32)

      is_correct *= weights
      ones *= weights

    update_total_op = state_ops.scatter_add(total, labels, ones)
    update_count_op = state_ops.scatter_add(count, labels, is_correct)

    def compute_mean_accuracy(_, count, total):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for mean per-class accuracy.
      # Ignores the distribution argument; divides per-class correct counts by per-class
      # totals (zero-safe, totals floored at zero) and returns their mean as a scalar tensor.
      per_class_accuracy = math_ops.div_no_nan(
          count, math_ops.maximum(total, 0), name=None)
      mean_accuracy_v = math_ops.reduce_mean(
          per_class_accuracy, name='mean_accuracy')
      return mean_accuracy_v

    mean_accuracy_v = _aggregate_across_replicas(
        metrics_collections, compute_mean_accuracy, count, total)

    update_op = math_ops.div_no_nan(
        update_count_op, math_ops.maximum(update_total_op, 0), name='update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_accuracy_v, update_op


@tf_export(v1=['metrics.mean_iou'])
def mean_iou(labels,
             predictions,
             num_classes,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):
  # [FUNCTION DOC]
  # Streaming mean intersection-over-union for semantic segmentation. Inputs: labels and
  # predictions (integer class indices, flattened), num_classes (size of the square confusion
  # matrix), weights (optional), metrics_collections / updates_collections (optional lists),
  # name (optional scope, default mean_iou). Accumulates a float64 confusion matrix; the value
  # is the mean over classes present of true positives divided by the union (row sum plus
  # column sum minus diagonal), with classes of zero union excluded from the mean. Returns the
  # pair of mean-iou tensor and update op. Raises a runtime error in eager mode.
  """Calculate per-step mean Intersection-Over-Union (mIOU).

  Mean Intersection-Over-Union is a common evaluation metric for
  semantic image segmentation, which first computes the IOU for each
  semantic class and then computes the average over classes.
  IOU is defined as follows:
    IOU = true_positive / (true_positive + false_positive + false_negative).
  The predictions are accumulated in a confusion matrix, weighted by `weights`,
  and mIOU is then calculated from it.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean_iou`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `mean_iou`
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.

  Returns:
    mean_iou: A `Tensor` representing the mean intersection-over-union.
    update_op: An operation that increments the confusion matrix.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_iou is not supported when '
                       'eager execution is enabled.')

  with variable_scope.variable_scope(name, 'mean_iou',
                                     (predictions, labels, weights)):
    # Check if shape is compatible.
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    total_cm, update_op = _streaming_confusion_matrix(labels, predictions,
                                                      num_classes, weights)

    def compute_mean_iou(_, total_cm):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for mean_iou.
      # Ignores the distribution argument. From the accumulated confusion matrix computes, per
      # class, diagonal divided by (row sum plus column sum minus diagonal), skipping classes
      # whose union is zero, and returns the mean over the remaining classes (zero when no
      # class is valid) as a scalar tensor.
      """Compute the mean intersection-over-union via the confusion matrix."""
      sum_over_row = math_ops.cast(
          math_ops.reduce_sum(total_cm, 0), dtypes.float32)
      sum_over_col = math_ops.cast(
          math_ops.reduce_sum(total_cm, 1), dtypes.float32)
      cm_diag = math_ops.cast(array_ops.diag_part(total_cm), dtypes.float32)
      denominator = sum_over_row + sum_over_col - cm_diag

      # The mean is only computed over classes that appear in the
      # label or prediction tensor. If the denominator is 0, we need to
      # ignore the class.
      num_valid_entries = math_ops.reduce_sum(
          math_ops.cast(
              math_ops.not_equal(denominator, 0), dtype=dtypes.float32))

      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
      denominator = array_ops.where(
          math_ops.greater(denominator, 0), denominator,
          array_ops.ones_like(denominator))
      iou = math_ops.divide(cm_diag, denominator)

      # If the number of valid entries is 0 (no classes) we return 0.
      result = array_ops.where(
          math_ops.greater(num_valid_entries, 0),
          math_ops.reduce_sum(iou, name='mean_iou') / num_valid_entries, 0)
      return result

    # TODO(priyag): Use outside_compilation if in TPU context.
    mean_iou_v = _aggregate_across_replicas(
        metrics_collections, compute_mean_iou, total_cm)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_iou_v, update_op


@tf_export(v1=['metrics.mean_relative_error'])
def mean_relative_error(labels,
                        predictions,
                        normalizer,
                        weights=None,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None):
  # [FUNCTION DOC]
  # Streaming mean relative error. Inputs: labels, predictions and normalizer (same shape;
  # trailing singleton dimensions are squeezed), weights (optional), metrics_collections /
  # updates_collections (optional lists), name (optional, default mean_relative_error). Each
  # element error is the absolute difference divided by the normalizer, defined as zero where
  # the normalizer is zero; errors are fed to the streaming mean. Returns the pair of value
  # tensor and update op. Raises a runtime error in eager mode and shape errors on mismatch.
  """Computes the mean relative error by normalizing with the given values.

  The `mean_relative_error` function creates two local variables,
  `total` and `count` that are used to compute the mean relative absolute error.
  This average is weighted by `weights`, and it is ultimately returned as
  `mean_relative_error`: an idempotent operation that simply divides `total` by
  `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `mean_reative_error`. Internally, a `relative_errors` operation divides the
  absolute value of the differences between `predictions` and `labels` by the
  `normalizer`. Then `update_op` increments `total` with the reduced sum of the
  product of `weights` and `relative_errors`, and it increments `count` with the
  reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` of arbitrary shape.
    normalizer: A `Tensor` of the same shape as `predictions`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `mean_relative_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    mean_relative_error: A `Tensor` representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_relative_error`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_relative_error is not supported when '
                       'eager execution is enabled.')

  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions=predictions, labels=labels, weights=weights)

  predictions, normalizer = confusion_matrix.remove_squeezable_dimensions(
      predictions, normalizer)
  predictions.get_shape().assert_is_compatible_with(normalizer.get_shape())
  relative_errors = array_ops.where(
      math_ops.equal(normalizer, 0.0), array_ops.zeros_like(labels),
      math_ops.divide(math_ops.abs(labels - predictions), normalizer))
  return mean(relative_errors, weights, metrics_collections,
              updates_collections, name or 'mean_relative_error')


@tf_export(v1=['metrics.mean_squared_error'])
def mean_squared_error(labels,
                       predictions,
                       weights=None,
                       metrics_collections=None,
                       updates_collections=None,
                       name=None):
  # [FUNCTION DOC]
  # Streaming mean squared error. Inputs: labels and predictions (same shape after
  # squeezing), weights (optional), metrics_collections / updates_collections (optional
  # lists), name (optional, default mean_squared_error). Squares the elementwise difference and
  # feeds it to the streaming mean. Returns the pair of value tensor and update op. Raises a
  # runtime error in eager mode and value errors for mismatched shapes or bad collections.
  """Computes the mean squared error between the labels and predictions.

  The `mean_squared_error` function creates two local variables,
  `total` and `count` that are used to compute the mean squared error.
  This average is weighted by `weights`, and it is ultimately returned as
  `mean_squared_error`: an idempotent operation that simply divides `total` by
  `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `mean_squared_error`. Internally, a `squared_error` operation computes the
  element-wise square of the difference between `predictions` and `labels`. Then
  `update_op` increments `total` with the reduced sum of the product of
  `weights` and `squared_error`, and it increments `count` with the reduced sum
  of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` of arbitrary shape.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `mean_squared_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    mean_squared_error: A `Tensor` representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_squared_error`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_squared_error is not supported when '
                       'eager execution is enabled.')

  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions=predictions, labels=labels, weights=weights)
  squared_error = math_ops.squared_difference(labels, predictions)
  return mean(squared_error, weights, metrics_collections, updates_collections,
              name or 'mean_squared_error')


@tf_export(v1=['metrics.mean_tensor'])
def mean_tensor(values,
                weights=None,
                metrics_collections=None,
                updates_collections=None,
                name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the update op value matches "mean_value" and refers to
  # total and count; the real accumulators are float32 variables named total_tensor and
  # count_tensor, and the variables take their shape from the static shape of values, so the
  # input shape must be fully known.
  # Streaming element-wise weighted mean that keeps the input shape. Inputs: values (any
  # numeric tensor, cast to float32), weights (optional), metrics_collections /
  # updates_collections (optional lists), name (optional scope, default mean). Accumulates
  # elementwise sums of values times weights and of weights (ones when no weights). Returns the
  # pair of mean tensor (total over count, zero-safe) and update op. Raises a runtime error in
  # eager mode and value errors for bad weights or collections.
  """Computes the element-wise (weighted) mean of the given tensors.

  In contrast to the `mean` function which returns a scalar with the
  mean,  this function returns an average tensor with the same shape as the
  input tensors.

  The `mean_tensor` function creates two local variables,
  `total_tensor` and `count_tensor` that are used to compute the average of
  `values`. This average is ultimately returned as `mean` which is an idempotent
  operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean`.
  `update_op` increments `total` with the reduced sum of the product of `values`
  and `weights`, and it increments `count` with the reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `Tensor` of arbitrary dimensions.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `values`, and must be broadcastable to `values` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `values` dimension).
    metrics_collections: An optional list of collections that `mean`
      should be added to.
    updates_collections: An optional list of collections that `update_op`
      should be added to.
    name: An optional variable_scope name.

  Returns:
    mean: A float `Tensor` representing the current mean, the value of `total`
      divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_value`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_tensor is not supported when '
                       'eager execution is enabled.')

  with variable_scope.variable_scope(name, 'mean', (values, weights)):
    values = math_ops.cast(values, dtypes.float32)
    total = metric_variable(
        values.get_shape(), dtypes.float32, name='total_tensor')
    count = metric_variable(
        values.get_shape(), dtypes.float32, name='count_tensor')

    num_values = array_ops.ones_like(values)
    if weights is not None:
      values, _, weights = _remove_squeezable_dimensions(
          predictions=values, labels=None, weights=weights)
      weights = weights_broadcast_ops.broadcast_weights(
          math_ops.cast(weights, dtypes.float32), values)
      values = math_ops.multiply(values, weights)
      num_values = math_ops.multiply(num_values, weights)

    update_total_op = state_ops.assign_add(total, values)
    with ops.control_dependencies([values]):
      update_count_op = state_ops.assign_add(count, num_values)

    compute_mean = lambda _, t, c: math_ops.div_no_nan(  # pylint: disable=g-long-lambda
        t, math_ops.maximum(c, 0), name='value')

    mean_t = _aggregate_across_replicas(
        metrics_collections, compute_mean, total, count)

    update_op = math_ops.div_no_nan(
        update_total_op, math_ops.maximum(update_count_op, 0), name='update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_t, update_op


@tf_export(v1=['metrics.percentage_below'])
def percentage_below(values,
                     threshold,
                     weights=None,
                     metrics_collections=None,
                     updates_collections=None,
                     name=None):
  # [FUNCTION DOC]
  # Streaming fraction of values strictly below a threshold. Inputs: values (numeric tensor),
  # threshold (scalar), weights (optional), metrics_collections / updates_collections
  # (optional lists), name (optional, default percentage_below_threshold). Marks each value
  # below the threshold with one, others zero, and delegates to the streaming mean. Returns the
  # pair of fraction tensor and update op. Raises a runtime error in eager mode.
  """Computes the percentage of values less than the given threshold.

  The `percentage_below` function creates two local variables,
  `total` and `count` that are used to compute the percentage of `values` that
  fall below `threshold`. This rate is weighted by `weights`, and it is
  ultimately returned as `percentage` which is an idempotent operation that
  simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `percentage`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A numeric `Tensor` of arbitrary size.
    threshold: A scalar threshold.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `values`, and must be broadcastable to `values` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `values` dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    percentage: A `Tensor` representing the current mean, the value of `total`
      divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.percentage_below is not supported when '
                       'eager execution is enabled.')

  is_below_threshold = math_ops.cast(
      math_ops.less(values, threshold), dtypes.float32)
  return mean(is_below_threshold, weights, metrics_collections,
              updates_collections, name or 'percentage_below_threshold')


def _count_condition(values,
                     weights=None,
                     metrics_collections=None,
                     updates_collections=None):
  # [FUNCTION DOC]
  # HELPER: shared implementation of the false/true positive/negative count metrics.
  # Inputs: values (boolean tensor marking where the condition holds), weights (optional;
  # rank zero or equal to the rank of values, checked at graph run time), metrics_collections
  # and updates_collections (optional lists). Creates one float32 count variable, adds the
  # (weighted) number of true entries on each update. Returns the pair of aggregated count
  # value and update op. Raises a type error if values is not boolean.
  """Sums the weights of cases where the given values are True.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `bool` `Tensor` of arbitrary size.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `values`, and must be broadcastable to `values` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `values` dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.

  Returns:
    value_tensor: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  check_ops.assert_type(values, dtypes.bool)
  count = metric_variable([], dtypes.float32, name='count')

  values = math_ops.cast(values, dtypes.float32)
  if weights is not None:
    with ops.control_dependencies((check_ops.assert_rank_in(
        weights, (0, array_ops.rank(values))),)):
      weights = math_ops.cast(weights, dtypes.float32)
      values = math_ops.multiply(values, weights)

  value_tensor = _aggregate_variable(count, metrics_collections)

  update_op = state_ops.assign_add(count, math_ops.reduce_sum(values))
  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return value_tensor, update_op


@tf_export(v1=['metrics.false_negatives'])
def false_negatives(labels,
                    predictions,
                    weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    name=None):
  # [FUNCTION DOC]
  # Streaming weighted count of false negatives (label true, prediction false). Inputs: labels
  # and predictions (cast to boolean, same shape), weights (optional), metrics_collections /
  # updates_collections (optional lists), name (optional scope, default false_negatives).
  # Returns the pair of count tensor and update op. Raises a runtime error in eager mode and
  # shape errors on mismatch.
  """Computes the total number of false negatives.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    value_tensor: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.false_negatives is not supported when '
                       'eager execution is enabled.')

  with variable_scope.variable_scope(name, 'false_negatives',
                                     (predictions, labels, weights)):

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)
    is_false_negative = math_ops.logical_and(
        math_ops.equal(labels, True), math_ops.equal(predictions, False))
    return _count_condition(is_false_negative, weights, metrics_collections,
                            updates_collections)


@tf_export(v1=['metrics.false_negatives_at_thresholds'])
def false_negatives_at_thresholds(labels,
                                  predictions,
                                  thresholds,
                                  weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  # [FUNCTION DOC]
  # Streaming weighted false negative counts, one per threshold. Inputs: labels (cast to
  # boolean), predictions (floats in zero to one), thresholds (list or tuple of floats),
  # weights (optional), metrics_collections / updates_collections (optional lists), name
  # (optional scope). Uses the threshold confusion helper requesting only false negatives.
  # Returns the pair of vector tensor (length equal to number of thresholds) and update op.
  # Raises a runtime error in eager mode and value errors for mismatched shapes.
  """Computes false negatives at provided threshold values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `false_negatives`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    false_negatives:  A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that updates the `false_negatives` variable and
      returns its current value.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.false_negatives_at_thresholds is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'false_negatives',
                                     (predictions, labels, weights)):
    values, update_ops = _confusion_matrix_at_thresholds(
        labels, predictions, thresholds, weights=weights, includes=('fn',))

    fn_value = _aggregate_variable(values['fn'], metrics_collections)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_ops['fn'])

    return fn_value, update_ops['fn']


@tf_export(v1=['metrics.false_positives'])
def false_positives(labels,
                    predictions,
                    weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    name=None):
  # [FUNCTION DOC]
  # Streaming weighted count of false positives (label false, prediction true). Inputs: labels
  # and predictions (cast to boolean, same shape), weights (optional), metrics_collections /
  # updates_collections (optional lists), name (optional scope, default false_positives).
  # Returns the pair of count tensor and update op. Raises a runtime error in eager mode and
  # shape errors on mismatch.
  """Sum the weights of false positives.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    value_tensor: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.false_positives is not supported when '
                       'eager execution is enabled.')

  with variable_scope.variable_scope(name, 'false_positives',
                                     (predictions, labels, weights)):

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)
    is_false_positive = math_ops.logical_and(
        math_ops.equal(labels, False), math_ops.equal(predictions, True))
    return _count_condition(is_false_positive, weights, metrics_collections,
                            updates_collections)


@tf_export(v1=['metrics.false_positives_at_thresholds'])
def false_positives_at_thresholds(labels,
                                  predictions,
                                  thresholds,
                                  weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  # [FUNCTION DOC]
  # Streaming weighted false positive counts, one per threshold. Inputs: labels (cast to
  # boolean), predictions (floats in zero to one), thresholds (list or tuple of floats),
  # weights (optional), metrics_collections / updates_collections (optional lists), name
  # (optional scope). Uses the threshold confusion helper requesting only false positives.
  # Returns the pair of vector tensor and update op. Raises a runtime error in eager mode and
  # value errors for mismatched shapes.
  """Computes false positives at provided threshold values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `false_positives`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    false_positives:  A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that updates the `false_positives` variable and
      returns its current value.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.false_positives_at_thresholds is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'false_positives',
                                     (predictions, labels, weights)):
    values, update_ops = _confusion_matrix_at_thresholds(
        labels, predictions, thresholds, weights=weights, includes=('fp',))

    fp_value = _aggregate_variable(values['fp'], metrics_collections)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_ops['fp'])

    return fp_value, update_ops['fp']


@tf_export(v1=['metrics.true_negatives'])
def true_negatives(labels,
                   predictions,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   name=None):
  # [FUNCTION DOC]
  # Streaming weighted count of true negatives (label false, prediction false). Inputs: labels
  # and predictions (cast to boolean, same shape), weights (optional), metrics_collections /
  # updates_collections (optional lists), name (optional scope, default true_negatives).
  # Returns the pair of count tensor and update op. Raises a runtime error in eager mode and
  # shape errors on mismatch.
  """Sum the weights of true_negatives.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    value_tensor: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.true_negatives is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'true_negatives',
                                     (predictions, labels, weights)):

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)
    is_true_negative = math_ops.logical_and(
        math_ops.equal(labels, False), math_ops.equal(predictions, False))
    return _count_condition(is_true_negative, weights, metrics_collections,
                            updates_collections)


@tf_export(v1=['metrics.true_negatives_at_thresholds'])
def true_negatives_at_thresholds(labels,
                                 predictions,
                                 thresholds,
                                 weights=None,
                                 metrics_collections=None,
                                 updates_collections=None,
                                 name=None):
  # [FUNCTION DOC]
  # Streaming weighted true negative counts, one per threshold. Inputs: labels (cast to
  # boolean), predictions (floats in zero to one), thresholds (list or tuple of floats),
  # weights (optional), metrics_collections / updates_collections (optional lists), name
  # (optional scope). Uses the threshold confusion helper requesting only true negatives.
  # Returns the pair of vector tensor and update op. Raises a runtime error in eager mode and
  # value errors for mismatched shapes.
  """Computes true negatives at provided threshold values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `true_negatives`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    true_negatives:  A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that updates the `true_negatives` variable and
      returns its current value.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.true_negatives_at_thresholds is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'true_negatives',
                                     (predictions, labels, weights)):
    values, update_ops = _confusion_matrix_at_thresholds(
        labels, predictions, thresholds, weights=weights, includes=('tn',))

    tn_value = _aggregate_variable(values['tn'], metrics_collections)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_ops['tn'])

    return tn_value, update_ops['tn']


@tf_export(v1=['metrics.true_positives'])
def true_positives(labels,
                   predictions,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   name=None):
  # [FUNCTION DOC]
  # Streaming weighted count of true positives (label true, prediction true). Inputs: labels
  # and predictions (cast to boolean, same shape), weights (optional), metrics_collections /
  # updates_collections (optional lists), name (optional scope, default true_positives).
  # Returns the pair of count tensor and update op. Raises a runtime error in eager mode and
  # shape errors on mismatch.
  """Sum the weights of true_positives.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    value_tensor: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.true_positives is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'true_positives',
                                     (predictions, labels, weights)):

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)
    is_true_positive = math_ops.logical_and(
        math_ops.equal(labels, True), math_ops.equal(predictions, True))
    return _count_condition(is_true_positive, weights, metrics_collections,
                            updates_collections)


@tf_export(v1=['metrics.true_positives_at_thresholds'])
def true_positives_at_thresholds(labels,
                                 predictions,
                                 thresholds,
                                 weights=None,
                                 metrics_collections=None,
                                 updates_collections=None,
                                 name=None):
  # [FUNCTION DOC]
  # Streaming weighted true positive counts, one per threshold. Inputs: labels (cast to
  # boolean), predictions (floats in zero to one), thresholds (list or tuple of floats),
  # weights (optional), metrics_collections / updates_collections (optional lists), name
  # (optional scope). Uses the threshold confusion helper requesting only true positives.
  # Returns the pair of vector tensor and update op. Raises a runtime error in eager mode and
  # value errors for mismatched shapes.
  """Computes true positives at provided threshold values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `true_positives`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    true_positives:  A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that updates the `true_positives` variable and
      returns its current value.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.true_positives_at_thresholds is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'true_positives',
                                     (predictions, labels, weights)):
    values, update_ops = _confusion_matrix_at_thresholds(
        labels, predictions, thresholds, weights=weights, includes=('tp',))

    tp_value = _aggregate_variable(values['tp'], metrics_collections)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_ops['tp'])

    return tp_value, update_ops['tp']


@tf_export(v1=['metrics.precision'])
def precision(labels,
              predictions,
              weights=None,
              metrics_collections=None,
              updates_collections=None,
              name=None):
  # [FUNCTION DOC]
  # Streaming precision: true positives divided by (true plus false positives). Inputs: labels
  # and predictions (cast to boolean, same shape), weights (optional), metrics_collections /
  # updates_collections (optional lists), name (optional scope, default precision). Builds the
  # true and false positive counters through the corresponding count metrics (without
  # collections) and combines them; the result is zero when no positives were predicted.
  # Returns the pair of precision scalar (aggregated across replicas) and update op. Raises a
  # runtime error in eager mode and shape errors on mismatch.
  """Computes the precision of the predictions with respect to the labels.

  The `precision` function creates two local variables,
  `true_positives` and `false_positives`, that are used to compute the
  precision. This value is ultimately returned as `precision`, an idempotent
  operation that simply divides `true_positives` by the sum of `true_positives`
  and `false_positives`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision`. `update_op` weights each prediction by the corresponding value in
  `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `precision` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    precision: Scalar float `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately and whose value matches
      `precision`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.precision is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'precision',
                                     (predictions, labels, weights)):

    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)

    true_p, true_positives_update_op = true_positives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)
    false_p, false_positives_update_op = false_positives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)

    def compute_precision(tp, fp, name):
      # [FUNCTION DOC]
      # HELPER: precision formula for the precision metric.
      # Inputs: true positive and false positive counts and an op name. Returns tp divided by
      # tp plus fp where that sum is positive, otherwise zero.
      return array_ops.where(
          math_ops.greater(tp + fp, 0), math_ops.divide(tp, tp + fp), 0, name)

    def once_across_replicas(_, true_p, false_p):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for precision.
      # Ignores the distribution argument; returns the precision computed from the aggregated
      # true and false positive counts, named value.
      return compute_precision(true_p, false_p, 'value')

    p = _aggregate_across_replicas(metrics_collections, once_across_replicas,
                                   true_p, false_p)

    update_op = compute_precision(true_positives_update_op,
                                  false_positives_update_op, 'update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return p, update_op


@tf_export(v1=['metrics.precision_at_thresholds'])
def precision_at_thresholds(labels,
                            predictions,
                            thresholds,
                            weights=None,
                            metrics_collections=None,
                            updates_collections=None,
                            name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring describes metrics_collections as the collection for "auc"; the
  # value actually added there is the precision vector. Also precision is divided by the sum
  # plus a tiny epsilon, so it is slightly below the exact ratio and zero (not undefined) when
  # nothing is above the threshold.
  # Streaming precision per threshold. Inputs: labels (cast to boolean), predictions (floats in
  # zero to one), thresholds (list or tuple of floats), weights (optional), metrics_collections
  # / updates_collections (optional lists), name (optional scope). Accumulates true and false
  # positive counts per threshold. Returns the pair of vector tensor (one entry per threshold)
  # and update op. Raises a runtime error in eager mode and value errors for bad shapes.
  """Computes precision values for different `thresholds` on `predictions`.

  The `precision_at_thresholds` function creates four local variables,
  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`
  for various values of thresholds. `precision[i]` is defined as the total
  weight of values in `predictions` above `thresholds[i]` whose corresponding
  entry in `labels` is `True`, divided by the total weight of values in
  `predictions` above `thresholds[i]` (`true_positives[i] / (true_positives[i] +
  false_positives[i])`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `auc` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    precision: A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables that
      are used in the computation of `precision`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.precision_at_thresholds is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'precision_at_thresholds',
                                     (predictions, labels, weights)):
    values, update_ops = _confusion_matrix_at_thresholds(
        labels, predictions, thresholds, weights, includes=('tp', 'fp'))

    # Avoid division by zero.
    epsilon = 1e-7

    def compute_precision(tp, fp, name):
      # [FUNCTION DOC]
      # HELPER: precision formula for precision_at_thresholds.
      # Inputs: true positive and false positive count vectors and a name suffix. Returns tp
      # divided by (tp + fp + small epsilon), named with the given suffix.
      return math_ops.divide(tp, epsilon + tp + fp, name='precision_' + name)

    def precision_across_replicas(_, values):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for precision_at_thresholds.
      # Ignores the distribution argument; takes the mapping of count variables and returns the
      # per-threshold precision vector named value.
      return compute_precision(values['tp'], values['fp'], 'value')

    prec = _aggregate_across_replicas(
        metrics_collections, precision_across_replicas, values)

    update_op = compute_precision(update_ops['tp'], update_ops['fp'],
                                  'update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return prec, update_op


@tf_export(v1=['metrics.recall'])
def recall(labels,
           predictions,
           weights=None,
           metrics_collections=None,
           updates_collections=None,
           name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the eager-mode error message repeats itself ("is not supported is not
  # supported"), a harmless wording bug.
  # Streaming recall: true positives divided by (true positives plus false negatives). Inputs:
  # labels and predictions (cast to boolean, same shape), weights (optional), metrics_collections
  # / updates_collections (optional lists), name (optional scope, default recall). Builds the
  # counters through the true positive and false negative count metrics (without collections);
  # the result is zero when there are no actual positives. Returns the pair of recall scalar
  # (aggregated across replicas) and update op. Raises a runtime error in eager mode.
  """Computes the recall of the predictions with respect to the labels.

  The `recall` function creates two local variables, `true_positives`
  and `false_negatives`, that are used to compute the recall. This value is
  ultimately returned as `recall`, an idempotent operation that simply divides
  `true_positives` by the sum of `true_positives` and `false_negatives`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` that updates these variables and returns the `recall`. `update_op`
  weights each prediction by the corresponding value in `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `recall` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    recall: Scalar float `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately and whose value matches
      `recall`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.recall is not supported is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'recall',
                                     (predictions, labels, weights)):
    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)

    true_p, true_positives_update_op = true_positives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)
    false_n, false_negatives_update_op = false_negatives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)

    def compute_recall(true_p, false_n, name):
      # [FUNCTION DOC]
      # HELPER: recall formula for the recall metric.
      # Inputs: true positive and false negative counts and an op name. Returns tp divided by
      # tp plus fn where that sum is positive, otherwise zero.
      return array_ops.where(
          math_ops.greater(true_p + false_n, 0),
          math_ops.divide(true_p, true_p + false_n), 0, name)

    def once_across_replicas(_, true_p, false_n):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for recall.
      # Ignores the distribution argument; returns recall computed from the aggregated true
      # positive and false negative counts, named value.
      return compute_recall(true_p, false_n, 'value')

    rec = _aggregate_across_replicas(
        metrics_collections, once_across_replicas, true_p, false_n)

    update_op = compute_recall(true_positives_update_op,
                               false_negatives_update_op, 'update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return rec, update_op


def _at_k_name(name, k=None, class_id=None):
  # [FUNCTION DOC]
  # HELPER: builds metric names for the at-k metrics.
  # Inputs: name (base string), k (optional integer cutoff) and class_id (optional integer).
  # Appends "_at_" and the cutoff (or "_at_k" when no cutoff is given) and then, when a class
  # id is given, "_class" and the id. Returns the resulting string. Raises a type error if k
  # or class_id is not an integer-like number.
  if k is not None:
    name = '%s_at_%d' % (name, k)
  else:
    name = '%s_at_k' % (name)
  if class_id is not None:
    name = '%s_class%d' % (name, class_id)
  return name


def _select_class_id(ids, selected_id):
  # [FUNCTION DOC]
  # HELPER: restricts class id tensors to a single class for per-class at-k metrics.
  # Inputs: ids (int64 dense or sparse tensor of class ids) and selected_id (integer class).
  # For sparse ids retains only entries equal to the class; for dense ids intersects each row
  # with the class as a set. Returns a sparse tensor with the same dense shape as the input
  # holding only entries equal to the selected class.
  """Filter all but `selected_id` out of `ids`.

  Args:
    ids: `int64` `Tensor` or `SparseTensor` of IDs.
    selected_id: Int id to select.

  Returns:
    `SparseTensor` of same dimensions as `ids`. This contains only the entries
    equal to `selected_id`.
  """
  ids = sparse_tensor.convert_to_tensor_or_sparse_tensor(ids)
  if isinstance(ids, sparse_tensor.SparseTensor):
    return sparse_ops.sparse_retain(ids, math_ops.equal(ids.values,
                                                        selected_id))

  # TODO(ptucker): Make this more efficient, maybe add a sparse version of
  # tf.equal and tf.reduce_any?

  # Shape of filled IDs is the same as `ids` with the last dim collapsed to 1.
  ids_shape = array_ops.shape(ids, out_type=dtypes.int64)
  ids_last_dim = array_ops.size(ids_shape) - 1
  filled_selected_id_shape = math_ops.reduced_shape(ids_shape,
                                                    array_ops.reshape(
                                                        ids_last_dim, [1]))

  # Intersect `ids` with the selected ID.
  filled_selected_id = array_ops.fill(filled_selected_id_shape,
                                      math_ops.cast(selected_id, dtypes.int64))
  result = sets.set_intersection(filled_selected_id, ids)
  return sparse_tensor.SparseTensor(
      indices=result.indices, values=result.values, dense_shape=ids_shape)


def _maybe_select_class_id(labels, predictions_idx, selected_id=None):
  # [FUNCTION DOC]
  # HELPER: optional per-class filter for the at-k metrics.
  # Inputs: labels and predictions_idx (int64 class id tensors, dense or sparse) and
  # selected_id (optional integer). With no selected class both inputs are returned
  # untouched; otherwise both are filtered to that class. Returns the pair of labels and
  # predictions_idx.
  """If class ID is specified, filter all other classes.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    predictions_idx: `int64` `Tensor` of class IDs, with shape [D1, ... DN, k]
      where N >= 1. Commonly, N=1 and `predictions_idx` has shape
      [batch size, k].
    selected_id: Int id to select.

  Returns:
    Tuple of `labels` and `predictions_idx`, possibly with classes removed.
  """
  if selected_id is None:
    return labels, predictions_idx
  return (_select_class_id(labels, selected_id),
          _select_class_id(predictions_idx, selected_id))


def _sparse_true_positive_at_k(labels,
                               predictions_idx,
                               class_id=None,
                               weights=None,
                               name=None):
  # [FUNCTION DOC]
  # HELPER: per-step true positive counts for recall at k and precision at k.
  # Inputs: labels (int64 dense or sparse class ids per example), predictions_idx (int64 top-k
  # predicted class ids, same leading dimensions as labels), class_id (optional; restrict to
  # one class), weights (optional; rank zero or one less than labels, must be broadcastable
  # to the result), name (optional scope). Counts the size of the set intersection of the
  # predicted and label ids for each example (as float64), multiplied by the weights.
  # Returns a float64 tensor with the leading dimensions of the inputs. Raises an error if
  # weights are not broadcastable to the counts.
  """Calculates true positives for recall@k and precision@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels_sparse`.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    name: Name of operation.

  Returns:
    A [D1, ... DN] `Tensor` of true positive counts.
  """
  with ops.name_scope(name, 'true_positives',
                      (predictions_idx, labels, weights)):
    labels, predictions_idx = _maybe_select_class_id(labels, predictions_idx,
                                                     class_id)
    tp = sets.set_size(sets.set_intersection(predictions_idx, labels))
    tp = math_ops.cast(tp, dtypes.float64)
    if weights is not None:
      with ops.control_dependencies((weights_broadcast_ops.assert_broadcastable(
          weights, tp),)):
        weights = math_ops.cast(weights, dtypes.float64)
        tp = math_ops.multiply(tp, weights)
    return tp


def _streaming_sparse_true_positive_at_k(labels,
                                         predictions_idx,
                                         k=None,
                                         class_id=None,
                                         weights=None,
                                         name=None):
  # [FUNCTION DOC]
  # HELPER: streaming accumulator of true positives for recall at k and precision at k.
  # Inputs: labels and predictions_idx (int64 class id tensors; see the per-step helper), k
  # (optional integer, only used to build the default scope name), class_id (optional single
  # class), weights (optional), name (optional scope name, also used as the variable name).
  # Computes per-example true positive counts, sums them over the batch (float64) and adds the
  # sum to a new float64 scalar metric variable. Returns the pair of that variable and the
  # update operation. Raises a value error if the weights shape is incompatible.
  """Calculates weighted per step true positives for recall@k and precision@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    k: Integer, k for @k metric. This is only used for default op name.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    name: Name of new variable, and namespace for other dependent ops.

  Returns:
    A tuple of `Variable` and update `Operation`.

  Raises:
    ValueError: If `weights` is not `None` and has an incompatible shape.
  """
  with ops.name_scope(name, _at_k_name('true_positive', k, class_id=class_id),
                      (predictions_idx, labels, weights)) as scope:
    tp = _sparse_true_positive_at_k(
        predictions_idx=predictions_idx,
        labels=labels,
        class_id=class_id,
        weights=weights)
    batch_total_tp = math_ops.cast(math_ops.reduce_sum(tp), dtypes.float64)

    var = metric_variable([], dtypes.float64, name=scope)
    return var, state_ops.assign_add(var, batch_total_tp, name='update')


def _sparse_false_negative_at_k(labels,
                                predictions_idx,
                                class_id=None,
                                weights=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring text about class_id talks about "true positives" and about
  # labels_sparse (copied from the true positive helper); this function counts false negatives.
  # HELPER: per-step false negative counts for recall at k.
  # Inputs: labels (int64 dense or sparse class ids), predictions_idx (int64 top-k predicted
  # ids), class_id (optional single class) and weights (optional, rank zero or one less than
  # labels, must be broadcastable). Counts, per example, the label ids that are not among the
  # predicted ids (set difference of labels minus predictions, as float64), times the weights.
  # Returns a float64 tensor with the leading dimensions of the inputs.
  """Calculates false negatives for recall@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels_sparse`.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).

  Returns:
    A [D1, ... DN] `Tensor` of false negative counts.
  """
  with ops.name_scope(None, 'false_negatives',
                      (predictions_idx, labels, weights)):
    labels, predictions_idx = _maybe_select_class_id(labels, predictions_idx,
                                                     class_id)
    fn = sets.set_size(
        sets.set_difference(predictions_idx, labels, aminusb=False))
    fn = math_ops.cast(fn, dtypes.float64)
    if weights is not None:
      with ops.control_dependencies((weights_broadcast_ops.assert_broadcastable(
          weights, fn),)):
        weights = math_ops.cast(weights, dtypes.float64)
        fn = math_ops.multiply(fn, weights)
    return fn


def _streaming_sparse_false_negative_at_k(labels,
                                          predictions_idx,
                                          k,
                                          class_id=None,
                                          weights=None,
                                          name=None):
  # [FUNCTION DOC]
  # HELPER: streaming accumulator of false negatives for recall at k.
  # Inputs: labels, predictions_idx (int64 class id tensors), k (required integer, only used
  # to build the default scope name), class_id (optional), weights (optional), name (optional
  # scope, also used as the variable name). Sums the per-example false negative counts over the
  # batch and adds them to a new float64 scalar metric variable. Returns the pair of that
  # variable and the update operation. Raises a value error if weights are incompatible.
  """Calculates weighted per step false negatives for recall@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    k: Integer, k for @k metric. This is only used for default op name.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    name: Name of new variable, and namespace for other dependent ops.

  Returns:
    A tuple of `Variable` and update `Operation`.

  Raises:
    ValueError: If `weights` is not `None` and has an incompatible shape.
  """
  with ops.name_scope(name, _at_k_name('false_negative', k, class_id=class_id),
                      (predictions_idx, labels, weights)) as scope:
    fn = _sparse_false_negative_at_k(
        predictions_idx=predictions_idx,
        labels=labels,
        class_id=class_id,
        weights=weights)
    batch_total_fn = math_ops.cast(math_ops.reduce_sum(fn), dtypes.float64)

    var = metric_variable([], dtypes.float64, name=scope)
    return var, state_ops.assign_add(var, batch_total_fn, name='update')


@tf_export(v1=['metrics.recall_at_k'])
def recall_at_k(labels,
                predictions,
                k,
                class_id=None,
                weights=None,
                metrics_collections=None,
                updates_collections=None,
                name=None):
  # [FUNCTION DOC]
  # Streaming recall at k from raw prediction scores. Inputs: labels (int64 dense or sparse
  # class ids per example), predictions (float scores or logits with the class dimension
  # last), k (integer), class_id (optional single class), weights (optional),
  # metrics_collections / updates_collections (optional lists), name (optional scope, default
  # recall_at_k with the cutoff and class). Takes the top k class indices of the predictions
  # and delegates to the top-k variant. Returns the pair of recall scalar (float64) and update
  # op. Raises a runtime error in eager mode and value errors on bad weights or collections.
  """Computes recall@k of the predictions with respect to sparse labels.

  If `class_id` is specified, we calculate recall by considering only the
      entries in the batch for which `class_id` is in the label, and computing
      the fraction of them for which `class_id` is in the top-k `predictions`.
  If `class_id` is not specified, we'll calculate recall as how often on
      average a class among the labels of a batch entry is in the top-k
      `predictions`.

  `sparse_recall_at_k` creates two local variables,
  `true_positive_at_<k>` and `false_negative_at_<k>`, that are used to compute
  the recall_at_k frequency. This frequency is ultimately returned as
  `recall_at_<k>`: an idempotent operation that simply divides
  `true_positive_at_<k>` by total (`true_positive_at_<k>` +
  `false_negative_at_<k>`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `recall_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
  indicating the top `k` `predictions`. Set operations applied to `top_k` and
  `labels` calculate the true positives and false negatives weighted by
  `weights`. Then `update_op` increments `true_positive_at_<k>` and
  `false_negative_at_<k>` using these values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values
      should be in range [0, num_classes), where num_classes is the last
      dimension of `predictions`. Values outside this range always count
      towards `false_negative_at_<k>`.
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
      The final dimension contains the logit values for each class. [D1, ... DN]
      must match `labels`.
    k: Integer, k for @k metric.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes), where num_classes is the last dimension of
      `predictions`. If class_id is outside this range, the method returns NAN.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    recall: Scalar `float64` `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately, and whose value matches
      `recall`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
    `predictions`, or if either `metrics_collections` or `updates_collections`
    are not a list or tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.recall_at_k is not '
                       'supported when eager execution is enabled.')

  with ops.name_scope(name, _at_k_name('recall', k, class_id=class_id),
                      (predictions, labels, weights)) as scope:
    _, top_k_idx = nn.top_k(predictions, k)
    return recall_at_top_k(
        labels=labels,
        predictions_idx=top_k_idx,
        k=k,
        class_id=class_id,
        weights=weights,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections,
        name=scope)


@tf_export(v1=['metrics.recall_at_top_k'])
def recall_at_top_k(labels,
                    predictions_idx,
                    k=None,
                    class_id=None,
                    weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: unlike the other public metrics it does not refuse to run under eager
  # execution on its own (only the logits variant checks), and the value is a plain division
  # of the two counters, so it yields not-a-number when there are no relevant labels at all.
  # Streaming recall at k from already selected top-k class indices. Inputs: labels (int64
  # dense or sparse class ids), predictions_idx (integer tensor whose last dimension holds the
  # top k class ids), k (optional integer used only for naming), class_id (optional single
  # class), weights (optional), metrics_collections / updates_collections (optional lists),
  # name (optional scope). Expands labels to match the rank of predictions, accumulates true
  # positives and false negatives in float64 variables, and divides. Returns the pair of recall
  # scalar and update op. Raises value errors on incompatible weights.
  """Computes recall@k of top-k predictions with respect to sparse labels.

  Differs from `recall_at_k` in that predictions must be in the form of top `k`
  class indices, whereas `recall_at_k` expects logits. Refer to `recall_at_k`
  for more details.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values
      should be in range [0, num_classes), where num_classes is the last
      dimension of `predictions`. Values outside this range always count
      towards `false_negative_at_<k>`.
    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where N >= 1.
      Commonly, N=1 and predictions has shape [batch size, k]. The final
      dimension contains the top `k` predicted class indices. [D1, ... DN] must
      match `labels`.
    k: Integer, k for @k metric. Only used for the default op name.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes), where num_classes is the last dimension of
      `predictions`. If class_id is outside this range, the method returns NAN.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    recall: Scalar `float64` `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately, and whose value matches
      `recall`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
    `predictions`, or if either `metrics_collections` or `updates_collections`
    are not a list or tuple.
  """
  with ops.name_scope(name, _at_k_name('recall', k, class_id=class_id),
                      (predictions_idx, labels, weights)) as scope:
    labels = _maybe_expand_labels(labels, predictions_idx)
    top_k_idx = math_ops.cast(predictions_idx, dtypes.int64)
    tp, tp_update = _streaming_sparse_true_positive_at_k(
        predictions_idx=top_k_idx,
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)
    fn, fn_update = _streaming_sparse_false_negative_at_k(
        predictions_idx=top_k_idx,
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)

    def compute_recall(_, tp, fn):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for recall at top k.
      # Ignores the distribution argument; returns true positives divided by the sum of true
      # positives and false negatives (no zero guard), named with the enclosing scope.
      return math_ops.divide(tp, math_ops.add(tp, fn), name=scope)

    metric = _aggregate_across_replicas(
        metrics_collections, compute_recall, tp, fn)

    update = math_ops.divide(
        tp_update, math_ops.add(tp_update, fn_update), name='update')
    if updates_collections:
      ops.add_to_collections(updates_collections, update)
    return metric, update


@tf_export(v1=['metrics.recall_at_thresholds'])
def recall_at_thresholds(labels,
                         predictions,
                         thresholds,
                         weights=None,
                         metrics_collections=None,
                         updates_collections=None,
                         name=None):
  # [FUNCTION DOC]
  # Streaming recall per threshold. Inputs: labels (cast to boolean), predictions (floats in
  # zero to one), thresholds (list or tuple of floats), weights (optional), metrics_collections
  # / updates_collections (optional lists), name (optional scope). Accumulates true positive and
  # false negative counts per threshold; recall is tp divided by (tp + fn + tiny epsilon).
  # Returns the pair of vector tensor (one entry per threshold) and update op. Raises a runtime
  # error in eager mode and value errors for mismatched shapes.
  """Computes various recall values for different `thresholds` on `predictions`.

  The `recall_at_thresholds` function creates four local variables,
  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`
  for various values of thresholds. `recall[i]` is defined as the total weight
  of values in `predictions` above `thresholds[i]` whose corresponding entry in
  `labels` is `True`, divided by the total weight of `True` values in `labels`
  (`true_positives[i] / (true_positives[i] + false_negatives[i])`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `recall`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `recall` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    recall: A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables that
      are used in the computation of `recall`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.recall_at_thresholds is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'recall_at_thresholds',
                                     (predictions, labels, weights)):
    values, update_ops = _confusion_matrix_at_thresholds(
        labels, predictions, thresholds, weights, includes=('tp', 'fn'))

    # Avoid division by zero.
    epsilon = 1e-7

    def compute_recall(tp, fn, name):
      # [FUNCTION DOC]
      # HELPER: recall formula for recall_at_thresholds.
      # Inputs: true positive and false negative count vectors and a name suffix. Returns tp
      # divided by (tp + fn + small epsilon), named with the suffix.
      return math_ops.divide(tp, epsilon + tp + fn, name='recall_' + name)

    def recall_across_replicas(_, values):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for recall_at_thresholds.
      # Ignores the distribution argument; takes the mapping of count variables and returns the
      # per-threshold recall vector named value.
      return compute_recall(values['tp'], values['fn'], 'value')

    rec = _aggregate_across_replicas(
        metrics_collections, recall_across_replicas, values)

    update_op = compute_recall(update_ops['tp'], update_ops['fn'], 'update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return rec, update_op


@tf_export(v1=['metrics.root_mean_squared_error'])
def root_mean_squared_error(labels,
                            predictions,
                            weights=None,
                            metrics_collections=None,
                            updates_collections=None,
                            name=None):
  # [FUNCTION DOC]
  # Streaming root mean squared error. Inputs: labels and predictions (same shape after
  # squeezing), weights (optional), metrics_collections / updates_collections (optional lists),
  # name (optional, default root_mean_squared_error). Computes the streaming mean squared error
  # (without collections) and takes its square root for both the value (aggregated across
  # replicas) and the update op. Returns the pair of value tensor and update op. Raises a
  # runtime error in eager mode and value errors for mismatched shapes.
  """Computes the root mean squared error between the labels and predictions.

  The `root_mean_squared_error` function creates two local variables,
  `total` and `count` that are used to compute the root mean squared error.
  This average is weighted by `weights`, and it is ultimately returned as
  `root_mean_squared_error`: an idempotent operation that takes the square root
  of the division of `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `root_mean_squared_error`. Internally, a `squared_error` operation computes
  the element-wise square of the difference between `predictions` and `labels`.
  Then `update_op` increments `total` with the reduced sum of the product of
  `weights` and `squared_error`, and it increments `count` with the reduced sum
  of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of the same shape as `predictions`.
    predictions: A `Tensor` of arbitrary shape.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `root_mean_squared_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    root_mean_squared_error: A `Tensor` representing the current mean, the value
      of `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `root_mean_squared_error`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.root_mean_squared_error is not '
                       'supported when eager execution is enabled.')

  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions=predictions, labels=labels, weights=weights)
  mse, update_mse_op = mean_squared_error(labels, predictions, weights, None,
                                          None, name or
                                          'root_mean_squared_error')

  once_across_replicas = lambda _, mse: math_ops.sqrt(mse)
  rmse = _aggregate_across_replicas(
      metrics_collections, once_across_replicas, mse)

  update_rmse_op = math_ops.sqrt(update_mse_op)
  if updates_collections:
    ops.add_to_collections(updates_collections, update_rmse_op)

  return rmse, update_rmse_op


@tf_export(v1=['metrics.sensitivity_at_specificity'])
def sensitivity_at_specificity(labels,
                               predictions,
                               specificity,
                               weights=None,
                               num_thresholds=200,
                               metrics_collections=None,
                               updates_collections=None,
                               name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the first line of the docstring says it computes "the specificity at a given
  # sensitivity", but the function computes the sensitivity at a given specificity.
  # Streaming sensitivity (recall) at the threshold whose specificity is closest to a target.
  # Inputs: labels (cast to boolean), predictions (floats in zero to one), specificity (scalar
  # target in zero to one; checked eagerly at graph construction), weights (optional),
  # num_thresholds (default 200; linearly spaced thresholds padded with one just below zero
  # and one just above one), metrics_collections / updates_collections (optional lists), name
  # (optional scope). Accumulates all four confusion counts per threshold, picks the threshold
  # whose specificity is closest to the target and returns tp / (tp + fn + epsilon) there.
  # Returns the pair of scalar tensor and update op. Raises a runtime error in eager mode and a
  # value error when specificity is outside zero to one.
  """Computes the specificity at a given sensitivity.

  The `sensitivity_at_specificity` function creates four local
  variables, `true_positives`, `true_negatives`, `false_positives` and
  `false_negatives` that are used to compute the sensitivity at the given
  specificity value. The threshold for the given specificity value is computed
  and used to evaluate the corresponding sensitivity.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `sensitivity`. `update_op` increments the `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` counts with the weight of each case
  found in the `predictions` and `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  For additional information about specificity and sensitivity, see the
  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    specificity: A scalar value in range `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use for matching the given
      specificity.
    metrics_collections: An optional list of collections that `sensitivity`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    sensitivity: A scalar `Tensor` representing the sensitivity at the given
      `specificity` value.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `sensitivity`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `specificity` is not between 0 and 1, or if either `metrics_collections`
      or `updates_collections` are not a list or tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.sensitivity_at_specificity is not '
                       'supported when eager execution is enabled.')

  if specificity < 0 or specificity > 1:
    raise ValueError('`specificity` must be in the range [0, 1]. Currently, '
                     f'`specificity` got {specificity}.')

  with variable_scope.variable_scope(name, 'sensitivity_at_specificity',
                                     (predictions, labels, weights)):
    kepsilon = 1e-7  # to account for floating point imprecisions
    thresholds = [
        (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
    ]
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

    values, update_ops = _confusion_matrix_at_thresholds(
        labels, predictions, thresholds, weights)

    def compute_sensitivity_at_specificity(tp, tn, fp, fn, name):
      # [FUNCTION DOC]
      # HELPER: selects the threshold and computes the sensitivity for the metric.
      # Inputs: per-threshold count vectors tp, tn, fp, fn and an op name. Computes the
      # specificity at each threshold, finds the index whose specificity is closest to the
      # target, and returns tp over (tp + fn + epsilon) at that index, named by the argument.
      specificities = math_ops.divide(tn, tn + fp + kepsilon)
      tf_index = math_ops.argmin(math_ops.abs(specificities - specificity), 0)
      tf_index = math_ops.cast(tf_index, dtypes.int32)

      # Now, we have the implicit threshold, so compute the sensitivity:
      return math_ops.divide(tp[tf_index],
                             tp[tf_index] + fn[tf_index] + kepsilon, name)

    def sensitivity_across_replicas(_, values):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for sensitivity_at_specificity.
      # Ignores the distribution argument; takes the mapping of count variables and returns the
      # sensitivity scalar named value.
      return compute_sensitivity_at_specificity(
          values['tp'], values['tn'], values['fp'], values['fn'], 'value')

    sensitivity = _aggregate_across_replicas(
        metrics_collections, sensitivity_across_replicas, values)

    update_op = compute_sensitivity_at_specificity(
        update_ops['tp'], update_ops['tn'], update_ops['fp'], update_ops['fn'],
        'update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return sensitivity, update_op


def _expand_and_tile(tensor, multiple, dim=0, name=None):
  # [FUNCTION DOC]
  # HELPER: replicates a tensor along a freshly inserted dimension (used by the sparse average
  # precision computation).
  # Inputs: tensor (dense or sparse), multiple (integer, at least one, the repeat count), dim
  # (integer position where the new dimension is inserted, default 0), name (optional scope).
  # Returns the expanded and tiled tensor. Raises a value error when multiple is below one or
  # dim is outside the valid range for the tensor rank.
  """Slice `tensor` shape in 2, then tile along the sliced dimension.

  A new dimension is inserted in shape of `tensor` before `dim`, then values are
  tiled `multiple` times along the new dimension.

  Args:
    tensor: Input `Tensor` or `SparseTensor`.
    multiple: Integer, number of times to tile.
    dim: Integer, dimension along which to tile.
    name: Name of operation.

  Returns:
    `Tensor` result of expanding and tiling `tensor`.

  Raises:
    ValueError: if `multiple` is less than 1, or `dim` is not in
    `[-rank(tensor), rank(tensor)]`.
  """
  if multiple < 1:
    raise ValueError(f'Invalid argument multiple={multiple} for '
                     'expand_and_tile  call. `multiple` must be an integer > 0')
  with ops.name_scope(name, 'expand_and_tile',
                      (tensor, multiple, dim)) as scope:
    # Sparse.
    tensor = sparse_tensor.convert_to_tensor_or_sparse_tensor(tensor)
    if isinstance(tensor, sparse_tensor.SparseTensor):
      if dim < 0:
        expand_dims = array_ops.reshape(
            array_ops.size(tensor.dense_shape) + dim, [1])
      else:
        expand_dims = [dim]
      expanded_shape = array_ops.concat(
          (array_ops.slice(tensor.dense_shape, [0], expand_dims), [1],
           array_ops.slice(tensor.dense_shape, expand_dims, [-1])),
          0,
          name='expanded_shape')
      expanded = sparse_ops.sparse_reshape(
          tensor, shape=expanded_shape, name='expand')
      if multiple == 1:
        return expanded
      return sparse_ops.sparse_concat(
          dim - 1 if dim < 0 else dim, [expanded] * multiple, name=scope)

    # Dense.
    expanded = array_ops.expand_dims(
        tensor, dim if (dim >= 0) else (dim - 1), name='expand')
    if multiple == 1:
      return expanded
    ones = array_ops.ones_like(array_ops.shape(tensor))
    tile_multiples = array_ops.concat(
        (ones[:dim], (multiple,), ones[dim:]), 0, name='multiples')
    return array_ops.tile(expanded, tile_multiples, name=scope)


def _num_relevant(labels, k):
  # [FUNCTION DOC]
  # HELPER: relevant-item count per example for the average precision at k computation.
  # Inputs: labels (int64 dense or sparse tensor with the label list in the last dimension)
  # and k (integer, at least one). For sparse labels the count is the set size of each row; for
  # dense labels it is the number of non-negative entries along the last dimension (negative
  # entries are treated as padding). In both cases the result is capped at k. Returns an
  # integer tensor with the leading dimensions of labels. Raises a value error if k is below one.
  """Computes number of relevant values for each row in labels.

  For labels with shape [D1, ... DN, num_labels], this is the minimum of
  `num_labels` and `k`.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels].
    k: Integer, k for @k metric.

  Returns:
    Integer `Tensor` of shape [D1, ... DN], where each value is the number of
    relevant values for that row.

  Raises:
    ValueError: if inputs have invalid dtypes or values.
  """
  if k < 1:
    raise ValueError(f'Invalid k={k}')
  with ops.name_scope(None, 'num_relevant', (labels,)) as scope:
    # For SparseTensor, calculate separate count for each row.
    labels = sparse_tensor.convert_to_tensor_or_sparse_tensor(labels)
    if isinstance(labels, sparse_tensor.SparseTensor):
      return math_ops.minimum(sets.set_size(labels), k, name=scope)

    # The relevant values for each (d1, ... dN) is the minimum of k and the
    # number of labels along the last dimension that are non-negative.
    num_labels = math_ops.reduce_sum(
        array_ops.where_v2(math_ops.greater_equal(labels, 0),
                           array_ops.ones_like(labels),
                           array_ops.zeros_like(labels)),
        axis=-1)
    return math_ops.minimum(num_labels, k, name=scope)


def _sparse_average_precision_at_top_k(labels, predictions_idx):
  # [FUNCTION DOC]
  # HELPER: per-example average precision at k for the sparse retrieval metrics.
  # Inputs: labels (int64 dense or sparse class ids; negative values ignored) and
  # predictions_idx (integer tensor whose last dimension, which must be statically known, holds
  # the top k predicted ids). Expands the labels to the label rank of predictions, tiles the
  # labels once per rank position, computes whether each ranked prediction is relevant,
  # cumulative precision at each rank, sums the precision at relevant ranks and divides by the
  # number of relevant items (capped at k). Returns a float64 tensor with the leading
  # dimensions of the inputs; rows with no relevant labels divide by zero. Raises a value error
  # if predictions_idx is a scalar or its last dimension is unknown.
  """Computes average precision@k of predictions with respect to sparse labels.

  From en.wikipedia.org/wiki/Information_retrieval#Average_precision, formula
  for each row is:

    AveP = sum_{i=1...k} P_{i} * rel_{i} / num_relevant_items

  A "row" is the elements in dimension [D1, ... DN] of `predictions_idx`,
  `labels`, and the result `Tensors`. In the common case, this is [batch_size].
  Each row of the results contains the average precision for that row.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `predictions_idx`.
      Values should be non-negative. Negative values are ignored.
    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where N >= 1.
      Commonly, N=1 and `predictions_idx` has shape [batch size, k]. The final
      dimension must be set and contains the top `k` predicted class indices.
      [D1, ... DN] must match `labels`. Values should be in range
      [0, num_classes).

  Returns:
    `float64` `Tensor` of shape [D1, ... DN], where each value is the average
    precision for that row.

  Raises:
    ValueError: if the last dimension of predictions_idx is not set.
  """
  with ops.name_scope(None, 'average_precision',
                      (predictions_idx, labels)) as scope:
    predictions_idx = math_ops.cast(
        predictions_idx, dtypes.int64, name='predictions_idx')
    if predictions_idx.get_shape().ndims == 0:
      raise ValueError('The rank of `predictions_idx` must be at least 1.')
    k = predictions_idx.get_shape().as_list()[-1]
    if k is None:
      raise ValueError('The last dimension of predictions_idx must be set. '
                       'Currently, it is None.')
    labels = _maybe_expand_labels(labels, predictions_idx)

    # Expand dims to produce [D1, ... DN, k, 1] tensor. This gives us a separate
    # prediction for each k, so we can calculate separate true positive values
    # for each k.
    predictions_idx_per_k = array_ops.expand_dims(
        predictions_idx, -1, name='predictions_idx_per_k')

    # Replicate labels k times to produce [D1, ... DN, k, num_labels] tensor.
    labels_per_k = _expand_and_tile(
        labels, multiple=k, dim=-1, name='labels_per_k')

    # The following tensors are all of shape [D1, ... DN, k], containing values
    # per row, per k value.
    # `relevant_per_k` (int32) - Relevance indicator, 1 if the prediction at
    #     that k value is correct, 0 otherwise. This is the "rel_{i}" term from
    #     the formula above.
    # `tp_per_k` (int32) - True positive counts.
    # `retrieved_per_k` (int32) - Number of predicted values at each k. This is
    #     the precision denominator.
    # `precision_per_k` (float64) - Precision at each k. This is the "P_{i}"
    #     term from the formula above.
    # `relevant_precision_per_k` (float64) - Relevant precisions; i.e.,
    #     precisions at all k for which relevance indicator is true.
    relevant_per_k = _sparse_true_positive_at_k(
        labels_per_k, predictions_idx_per_k, name='relevant_per_k')
    tp_per_k = math_ops.cumsum(relevant_per_k, axis=-1, name='tp_per_k')
    retrieved_per_k = math_ops.cumsum(
        array_ops.ones_like(relevant_per_k), axis=-1, name='retrieved_per_k')
    precision_per_k = math_ops.divide(
        math_ops.cast(tp_per_k, dtypes.float64),
        math_ops.cast(retrieved_per_k, dtypes.float64),
        name='precision_per_k')
    relevant_precision_per_k = math_ops.multiply(
        precision_per_k,
        math_ops.cast(relevant_per_k, dtypes.float64),
        name='relevant_precision_per_k')

    # Reduce along k dimension to get the sum, yielding a [D1, ... DN] tensor.
    precision_sum = math_ops.reduce_sum(
        relevant_precision_per_k, axis=(-1,), name='precision_sum')

    # Divide by number of relevant items to get average precision. These are
    # the "num_relevant_items" and "AveP" terms from the formula above.
    num_relevant_items = math_ops.cast(_num_relevant(labels, k), dtypes.float64)
    return math_ops.divide(precision_sum, num_relevant_items, name=scope)


def _streaming_sparse_average_precision_at_top_k(labels,
                                                 predictions_idx,
                                                 weights=None,
                                                 metrics_collections=None,
                                                 updates_collections=None,
                                                 name=None):
  # [FUNCTION DOC]
  # HELPER: streaming mean-average-precision accumulator behind average_precision_at_k.
  # Inputs: labels and predictions_idx (see the per-example helper), weights (optional; rank
  # zero or one less than labels, broadcast to the per-example values), metrics_collections /
  # updates_collections (optional lists), name (optional scope). Computes per-example average
  # precision (times weights), then accumulates two float64 metric variables: the total of the
  # average precisions and the maximum attainable total (row count, or sum of weights). The
  # value is total divided by maximum with zero-safe division. Returns the pair of mean average
  # precision scalar (aggregated across replicas) and update op.
  """Computes average precision@k of predictions with respect to sparse labels.

  `sparse_average_precision_at_top_k` creates two local variables,
  `average_precision_at_<k>/total` and `average_precision_at_<k>/max`, that
  are used to compute the frequency. This frequency is ultimately returned as
  `average_precision_at_<k>`: an idempotent operation that simply divides
  `average_precision_at_<k>/total` by `average_precision_at_<k>/max`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision_at_<k>`. Set operations applied to `top_k` and `labels` calculate
  the true positives and false positives weighted by `weights`. Then `update_op`
  increments `true_positive_at_<k>` and `false_positive_at_<k>` using these
  values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `predictions_idx`.
      Values should be non-negative. Negative values are ignored.
    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where N >= 1.
      Commonly, N=1 and `predictions_idx` has shape [batch size, k]. The final
      dimension contains the top `k` predicted class indices. [D1, ... DN] must
      match `labels`. Values should be in range [0, num_classes).
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    mean_average_precision: Scalar `float64` `Tensor` with the mean average
      precision values.
    update: `Operation` that increments variables appropriately, and whose
      value matches `metric`.
  """
  with ops.name_scope(name, 'average_precision_at_top_k',
                      (predictions_idx, labels, weights)) as scope:
    # Calculate per-example average precision, and apply weights.
    average_precision = _sparse_average_precision_at_top_k(
        predictions_idx=predictions_idx, labels=labels)
    if weights is not None:
      weights = weights_broadcast_ops.broadcast_weights(
          math_ops.cast(weights, dtypes.float64), average_precision)
      average_precision = math_ops.multiply(average_precision, weights)

    # Create accumulation variables and update ops for max average precision and
    # total average precision.
    with ops.name_scope(None, 'max', (average_precision,)) as max_scope:
      # `max` is the max possible precision. Since max for any row is 1.0:
      # - For the unweighted case, this is just the number of rows.
      # - For the weighted case, it's the sum of the weights broadcast across
      #   `average_precision` rows.
      max_var = metric_variable([], dtypes.float64, name=max_scope)
      if weights is None:
        batch_max = math_ops.cast(
            array_ops.size(average_precision, name='batch_max'), dtypes.float64)
      else:
        batch_max = math_ops.reduce_sum(weights, name='batch_max')
      max_update = state_ops.assign_add(max_var, batch_max, name='update')
    with ops.name_scope(None, 'total', (average_precision,)) as total_scope:
      total_var = metric_variable([], dtypes.float64, name=total_scope)
      batch_total = math_ops.reduce_sum(average_precision, name='batch_total')
      total_update = state_ops.assign_add(total_var, batch_total, name='update')

    # Divide total by max to get mean, for both vars and the update ops.
    def precision_across_replicas(_, total_var, max_var):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for mean average precision.
      # Ignores the distribution argument; returns the total divided by the maximum with a zero
      # result when the maximum is zero, named mean.
      return _safe_scalar_div(total_var, max_var, name='mean')

    mean_average_precision = _aggregate_across_replicas(
        metrics_collections, precision_across_replicas, total_var, max_var)

    update = _safe_scalar_div(total_update, max_update, name=scope)
    if updates_collections:
      ops.add_to_collections(updates_collections, update)

    return mean_average_precision, update


def _clean_out_of_range_indices(labels, num_classes):
  # [FUNCTION DOC]
  # HELPER: normalises out-of-range class labels before top-k retrieval metrics.
  # Inputs: labels (int64 dense tensor, sparse tensor or sparse tensor value) and num_classes
  # (int64 scalar tensor). If the maximum label is at least num_classes, every such value is
  # replaced by minus one (which the metrics ignore); otherwise labels are returned unchanged
  # (the check is a runtime conditional). Sparse inputs keep their indices and shape. Returns
  # the cleaned labels of the same kind as the input.
  """Replaces large out-of-range labels by small out-of-range labels.

  Replaces any value in `labels` that is greater or equal to `num_classes` by
  -1. Do this conditionally for efficiency in case there are no such values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor`.
    num_classes: `int64` scalar `Tensor`.
  Returns:
    An `int64` `Tensor` or `SparseTensor` as `labels` with indices greater
    or equal to num_classes replaced by -1.
  """

  def _labels_is_sparse():
    # [FUNCTION DOC]
    # HELPER: type test for the label cleaning code.
    # Takes no arguments; returns true when the enclosing labels are a sparse tensor or a
    # sparse tensor value, false for dense inputs.
    """Returns true is `labels` is a sparse tensor."""
    return isinstance(labels, (sparse_tensor.SparseTensor,
                               sparse_tensor.SparseTensorValue))

  def _clean_out_of_range(values):
    # [FUNCTION DOC]
    # HELPER: elementwise replacement used by the label cleaning code.
    # Input: values (integer tensor). Returns a tensor of the same shape where every value at
    # least as large as the enclosing num_classes is replaced by minus one.
    """Replaces by -1 any large out-of-range `values`."""
    return array_ops.where_v2(math_ops.greater_equal(values, num_classes),
                              -1 * array_ops.ones_like(values), values)

  def _clean_labels_out_of_range():
    # [FUNCTION DOC]
    # HELPER: conditional branch that cleans the enclosing labels.
    # Takes no arguments. For sparse labels rebuilds a tensor of the same class with cleaned
    # values and unchanged indices and dense shape; for dense labels cleans the tensor directly.
    # Returns the cleaned labels.
    """Replaces by -1 ane large out-of-range values in `labels`."""
    if _labels_is_sparse():
      return type(labels)(indices=labels.indices,
                          values=_clean_out_of_range(labels.values),
                          dense_shape=labels.dense_shape)
    else:
      return _clean_out_of_range(labels)

  max_labels = math_ops.reduce_max(
      labels.values if _labels_is_sparse() else labels)
  return cond.cond(
      math_ops.greater_equal(max_labels, num_classes),
      _clean_labels_out_of_range,
      lambda: labels)


@tf_export(v1=['metrics.sparse_average_precision_at_k'])
@deprecated(None, 'Use average_precision_at_k instead')
def sparse_average_precision_at_k(labels,
                                  predictions,
                                  k,
                                  weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  # [FUNCTION DOC]
  # Deprecated alias kept for backward compatibility. Forwards all arguments (labels,
  # predictions, k, weights, metrics_collections, updates_collections, name) unchanged to the
  # average precision at k function and returns its result (value tensor and update op). Any
  # exception, including the eager-mode runtime error, comes from the target function.
  """Renamed to `average_precision_at_k`, please use that method instead."""
  return average_precision_at_k(
      labels=labels,
      predictions=predictions,
      k=k,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


@tf_export(v1=['metrics.average_precision_at_k'])
def average_precision_at_k(labels,
                           predictions,
                           k,
                           weights=None,
                           metrics_collections=None,
                           updates_collections=None,
                           name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the eager-mode error message names the old function name
  # (sparse_average_precision_at_k) instead of this function.
  # Streaming mean average precision at k from raw prediction scores. Inputs: labels (int64
  # dense or sparse class ids; values outside zero to number of classes are ignored),
  # predictions (float scores with the class dimension last), k (integer, at least one),
  # weights (optional), metrics_collections / updates_collections (optional lists), name
  # (optional scope, default average_precision_at_k with the cutoff). Takes the top k indices,
  # replaces labels at or above the class count by minus one, and delegates to the streaming
  # accumulator. Returns the pair of mean average precision scalar (float64) and update op.
  # Raises a runtime error in eager mode and a value error when k is below one.
  """Computes average precision@k of predictions with respect to sparse labels.

  `average_precision_at_k` creates two local variables,
  `average_precision_at_<k>/total` and `average_precision_at_<k>/max`, that
  are used to compute the frequency. This frequency is ultimately returned as
  `average_precision_at_<k>`: an idempotent operation that simply divides
  `average_precision_at_<k>/total` by `average_precision_at_<k>/max`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
  indicating the top `k` `predictions`. Set operations applied to `top_k` and
  `labels` calculate the true positives and false positives weighted by
  `weights`. Then `update_op` increments `true_positive_at_<k>` and
  `false_positive_at_<k>` using these values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values
      should be in range [0, num_classes), where num_classes is the last
      dimension of `predictions`. Values outside this range are ignored.
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and `predictions` has shape
      [batch size, num_classes]. The final dimension contains the logit values
      for each class. [D1, ... DN] must match `labels`.
    k: Integer, k for @k metric. This will calculate an average precision for
      range `[1,k]`, as documented above.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    mean_average_precision: Scalar `float64` `Tensor` with the mean average
      precision values.
    update: `Operation` that increments variables appropriately, and whose
      value matches `metric`.

  Raises:
    ValueError: if k is invalid.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.sparse_average_precision_at_k is not '
                       'supported when eager execution is enabled.')

  if k < 1:
    raise ValueError(f'Invalid k={k}. `k` should be >= 1.')
  with ops.name_scope(name, _at_k_name('average_precision', k),
                      (predictions, labels, weights)) as scope:
    # Calculate top k indices to produce [D1, ... DN, k] tensor.
    _, predictions_idx = nn.top_k(predictions, k)
    # The documentation states that labels should be in [0, ..., num_classes),
    # but num_classes is lost when predictions_idx replaces predictions.
    # For conformity with the documentation, any label >= num_classes, which is
    # ignored, is replaced by -1.
    labels = _clean_out_of_range_indices(
        labels, math_ops.cast(array_ops.shape(predictions)[-1], dtypes.int64))
    return _streaming_sparse_average_precision_at_top_k(
        labels=labels,
        predictions_idx=predictions_idx,
        weights=weights,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections,
        name=scope)


def _sparse_false_positive_at_k(labels,
                                predictions_idx,
                                class_id=None,
                                weights=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring text about class_id talks about "true positives" and about
  # labels_sparse (copied from the true positive helper); this function counts false positives.
  # HELPER: per-step false positive counts for precision at k.
  # Inputs: labels (int64 dense or sparse class ids), predictions_idx (int64 top-k predicted
  # ids), class_id (optional single class) and weights (optional, rank zero or one less than
  # labels, must be broadcastable). Counts per example the predicted ids that are not among
  # the labels (set difference of predictions minus labels, as float64), times the weights.
  # Returns a float64 tensor with the leading dimensions of the inputs.
  """Calculates false positives for precision@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels_sparse`.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).

  Returns:
    A [D1, ... DN] `Tensor` of false positive counts.
  """
  with ops.name_scope(None, 'false_positives',
                      (predictions_idx, labels, weights)):
    labels, predictions_idx = _maybe_select_class_id(labels, predictions_idx,
                                                     class_id)
    fp = sets.set_size(
        sets.set_difference(predictions_idx, labels, aminusb=True))
    fp = math_ops.cast(fp, dtypes.float64)
    if weights is not None:
      with ops.control_dependencies((weights_broadcast_ops.assert_broadcastable(
          weights, fp),)):
        weights = math_ops.cast(weights, dtypes.float64)
        fp = math_ops.multiply(fp, weights)
    return fp


def _streaming_sparse_false_positive_at_k(labels,
                                          predictions_idx,
                                          k=None,
                                          class_id=None,
                                          weights=None,
                                          name=None):
  # [FUNCTION DOC]
  # HELPER: streaming accumulator of false positives for precision at k.
  # Inputs: labels, predictions_idx (int64 class id tensors), k (optional integer used only
  # for naming), class_id (optional), weights (optional), name (optional scope, also used as
  # the variable name). Sums per-example false positive counts over the batch and adds them to
  # a new float64 scalar metric variable. Returns the pair of that variable and the update
  # operation. Raises a value error if the weights shape is incompatible.
  """Calculates weighted per step false positives for precision@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    k: Integer, k for @k metric. This is only used for default op name.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    name: Name of new variable, and namespace for other dependent ops.

  Returns:
    A tuple of `Variable` and update `Operation`.

  Raises:
    ValueError: If `weights` is not `None` and has an incompatible shape.
  """
  with ops.name_scope(name, _at_k_name('false_positive', k, class_id=class_id),
                      (predictions_idx, labels, weights)) as scope:
    fp = _sparse_false_positive_at_k(
        predictions_idx=predictions_idx,
        labels=labels,
        class_id=class_id,
        weights=weights)
    batch_total_fp = math_ops.cast(math_ops.reduce_sum(fp), dtypes.float64)

    var = metric_variable([], dtypes.float64, name=scope)
    return var, state_ops.assign_add(var, batch_total_fp, name='update')


@tf_export(v1=['metrics.precision_at_top_k'])
def precision_at_top_k(labels,
                       predictions_idx,
                       k=None,
                       class_id=None,
                       weights=None,
                       metrics_collections=None,
                       updates_collections=None,
                       name=None):
  # [FUNCTION DOC]
  # Streaming precision at k from already selected top-k class indices. Inputs: labels (int64
  # dense or sparse class ids), predictions_idx (integer tensor whose last dimension holds the
  # top k ids), k (optional integer used only for naming), class_id (optional single class),
  # weights (optional), metrics_collections / updates_collections (optional lists), name
  # (optional scope). Expands labels to the rank of predictions, casts the indices to 64-bit
  # integers, and accumulates true and false positive counts in float64 metric variables
  # (optionally restricted to one class). The value is true positives over the sum of true and
  # false positives with no zero guard, so it is not-a-number when nothing was predicted.
  # Returns the pair of precision scalar (aggregated across replicas) and update op. Raises a
  # runtime error in eager mode and value errors on incompatible weights.
  """Computes precision@k of the predictions with respect to sparse labels.

  Differs from `sparse_precision_at_k` in that predictions must be in the form
  of top `k` class indices, whereas `sparse_precision_at_k` expects logits.
  Refer to `sparse_precision_at_k` for more details.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values
      should be in range [0, num_classes), where num_classes is the last
      dimension of `predictions`. Values outside this range are ignored.
    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where
      N >= 1. Commonly, N=1 and predictions has shape [batch size, k].
      The final dimension contains the top `k` predicted class indices.
      [D1, ... DN] must match `labels`.
    k: Integer, k for @k metric. Only used for the default op name.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes], where num_classes is the last dimension of
      `predictions`. If `class_id` is outside this range, the method returns
      NAN.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    precision: Scalar `float64` `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately, and whose value matches
      `precision`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
      `predictions`, or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.precision_at_top_k is not '
                       'supported when eager execution is enabled.')

  with ops.name_scope(name, _at_k_name('precision', k, class_id=class_id),
                      (predictions_idx, labels, weights)) as scope:
    labels = _maybe_expand_labels(labels, predictions_idx)
    top_k_idx = math_ops.cast(predictions_idx, dtypes.int64)
    tp, tp_update = _streaming_sparse_true_positive_at_k(
        predictions_idx=top_k_idx,
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)
    fp, fp_update = _streaming_sparse_false_positive_at_k(
        predictions_idx=top_k_idx,
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)

    def precision_across_replicas(_, tp, fp):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for precision at top k.
      # Ignores the distribution argument; returns true positives divided by the sum of true and
      # false positives (no zero guard), named with the enclosing scope.
      return math_ops.divide(tp, math_ops.add(tp, fp), name=scope)

    metric = _aggregate_across_replicas(
        metrics_collections, precision_across_replicas, tp, fp)

    update = math_ops.divide(
        tp_update, math_ops.add(tp_update, fp_update), name='update')
    if updates_collections:
      ops.add_to_collections(updates_collections, update)
    return metric, update


@tf_export(v1=['metrics.sparse_precision_at_k'])
@deprecated(None, 'Use precision_at_k instead')
def sparse_precision_at_k(labels,
                          predictions,
                          k,
                          class_id=None,
                          weights=None,
                          metrics_collections=None,
                          updates_collections=None,
                          name=None):
  # [FUNCTION DOC]
  # Deprecated alias kept for backward compatibility. Forwards all arguments (labels,
  # predictions, k, class_id, weights, metrics_collections, updates_collections, name)
  # unchanged to the precision at k function and returns its result (value tensor and update
  # op). Any exception, including the eager-mode runtime error, comes from the target function.
  """Renamed to `precision_at_k`, please use that method instead."""
  return precision_at_k(
      labels=labels,
      predictions=predictions,
      k=k,
      class_id=class_id,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


@tf_export(v1=['metrics.precision_at_k'])
def precision_at_k(labels,
                   predictions,
                   k,
                   class_id=None,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the eager-mode error message names the old function name
  # (sparse_precision_at_k) instead of precision_at_k.
  # Streaming precision at k from raw prediction scores. Inputs: labels (int64 dense or sparse
  # class ids; values outside the class range are ignored), predictions (float scores with the
  # class dimension last), k (integer), class_id (optional single class), weights (optional),
  # metrics_collections / updates_collections (optional lists), name (optional scope, default
  # precision_at_k with the cutoff and class). Takes the top k indices of the predictions and
  # delegates to the top-k variant. Returns the pair of precision scalar (float64) and update
  # op. Raises a runtime error in eager mode and value errors on bad weights or collections.
  """Computes precision@k of the predictions with respect to sparse labels.

  If `class_id` is specified, we calculate precision by considering only the
      entries in the batch for which `class_id` is in the top-k highest
      `predictions`, and computing the fraction of them for which `class_id` is
      indeed a correct label.
  If `class_id` is not specified, we'll calculate precision as how often on
      average a class among the top-k classes with the highest predicted values
      of a batch entry is correct and can be found in the label for that entry.

  `precision_at_k` creates two local variables,
  `true_positive_at_<k>` and `false_positive_at_<k>`, that are used to compute
  the precision@k frequency. This frequency is ultimately returned as
  `precision_at_<k>`: an idempotent operation that simply divides
  `true_positive_at_<k>` by total (`true_positive_at_<k>` +
  `false_positive_at_<k>`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
  indicating the top `k` `predictions`. Set operations applied to `top_k` and
  `labels` calculate the true positives and false positives weighted by
  `weights`. Then `update_op` increments `true_positive_at_<k>` and
  `false_positive_at_<k>` using these values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values
      should be in range [0, num_classes), where num_classes is the last
      dimension of `predictions`. Values outside this range are ignored.
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
      The final dimension contains the logit values for each class. [D1, ... DN]
      must match `labels`.
    k: Integer, k for @k metric.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes], where num_classes is the last dimension of
      `predictions`. If `class_id` is outside this range, the method returns
      NAN.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    precision: Scalar `float64` `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately, and whose value matches
      `precision`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
      `predictions`, or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.sparse_precision_at_k is not '
                       'supported when eager execution is enabled.')

  with ops.name_scope(name, _at_k_name('precision', k, class_id=class_id),
                      (predictions, labels, weights)) as scope:
    _, top_k_idx = nn.top_k(predictions, k)
    return precision_at_top_k(
        labels=labels,
        predictions_idx=top_k_idx,
        k=k,
        class_id=class_id,
        weights=weights,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections,
        name=scope)


@tf_export(v1=['metrics.specificity_at_sensitivity'])
def specificity_at_sensitivity(labels,
                               predictions,
                               sensitivity,
                               weights=None,
                               num_thresholds=200,
                               metrics_collections=None,
                               updates_collections=None,
                               name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the upper padding threshold is one minus a small epsilon, whereas the sibling
  # sensitivity-at-specificity metric uses one plus epsilon; predictions exactly equal to one
  # are therefore counted as positive at every threshold, which looks like a typo.
  # Streaming specificity at the threshold whose sensitivity is closest to a target. Inputs:
  # labels (cast to boolean), predictions (floats in zero to one), sensitivity (scalar target
  # in zero to one), weights (optional), num_thresholds (default 200, linearly spaced
  # thresholds padded at both ends), metrics_collections / updates_collections (optional
  # lists), name (optional scope). Accumulates the four confusion counts per threshold, picks
  # the threshold with sensitivity nearest the target (the last such one in case of ties) and
  # returns tn / (tn + fp + epsilon) there. Returns the pair of scalar tensor and update op.
  # Raises a runtime error in eager mode and a value error if sensitivity is outside zero to one.
  """Computes the specificity at a given sensitivity.

  The `specificity_at_sensitivity` function creates four local
  variables, `true_positives`, `true_negatives`, `false_positives` and
  `false_negatives` that are used to compute the specificity at the given
  sensitivity value. The threshold for the given sensitivity value is computed
  and used to evaluate the corresponding specificity.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `specificity`. `update_op` increments the `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` counts with the weight of each case
  found in the `predictions` and `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  For additional information about specificity and sensitivity, see the
  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    sensitivity: A scalar value in range `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use for matching the given
      sensitivity.
    metrics_collections: An optional list of collections that `specificity`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    specificity: A scalar `Tensor` representing the specificity at the given
      `sensitivity` value.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `specificity`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `sensitivity` is not between 0 and 1, or if either `metrics_collections`
      or `updates_collections` are not a list or tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.specificity_at_sensitivity is not '
                       'supported when eager execution is enabled.')

  if sensitivity < 0 or sensitivity > 1:
    raise ValueError('`sensitivity` must be in the range [0, 1]. Currently, '
                     f'`sensitivity` is {sensitivity}.')

  with variable_scope.variable_scope(name, 'specificity_at_sensitivity',
                                     (predictions, labels, weights)):
    kepsilon = 1e-7  # to account for floating point imprecisions
    thresholds = [
        (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
    ]
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 - kepsilon]

    values, update_ops = _confusion_matrix_at_thresholds(
        labels, predictions, thresholds, weights)

    def compute_specificity_at_sensitivity(tp, tn, fp, fn, name):
      # [FUNCTION DOC]
      # HELPER: threshold selection and specificity formula for the metric.
      # Inputs: per-threshold count vectors tp, tn, fp, fn and an op name. Computes the
      # sensitivity at each threshold, chooses the last index whose sensitivity is closest to
      # the target, and returns tn over (tn + fp + epsilon) at that index, named by the argument.
      """Computes the specificity at the given sensitivity.

      Args:
        tp: True positives.
        tn: True negatives.
        fp: False positives.
        fn: False negatives.
        name: The name of the operation.

      Returns:
        The specificity using the aggregated values.
      """
      sensitivities = math_ops.divide(tp, tp + fn + kepsilon)

      # We'll need to use this trick until tf.argmax allows us to specify
      # whether we should use the first or last index in case of ties.
      min_val = math_ops.reduce_min(math_ops.abs(sensitivities - sensitivity))
      indices_at_minval = math_ops.equal(
          math_ops.abs(sensitivities - sensitivity), min_val)
      indices_at_minval = math_ops.cast(indices_at_minval, dtypes.int64)
      indices_at_minval = math_ops.cumsum(indices_at_minval)
      tf_index = math_ops.argmax(indices_at_minval, 0)
      tf_index = math_ops.cast(tf_index, dtypes.int32)

      # Now, we have the implicit threshold, so compute the specificity:
      return math_ops.divide(tn[tf_index],
                             tn[tf_index] + fp[tf_index] + kepsilon, name)

    def specificity_across_replicas(_, values):
      # [FUNCTION DOC]
      # HELPER: value function handed to the replica aggregation for specificity_at_sensitivity.
      # Ignores the distribution argument; takes the mapping of count variables and returns the
      # specificity scalar named value.
      return compute_specificity_at_sensitivity(
          values['tp'], values['tn'], values['fp'], values['fn'], 'value')

    specificity = _aggregate_across_replicas(
        metrics_collections, specificity_across_replicas, values)

    update_op = compute_specificity_at_sensitivity(
        update_ops['tp'], update_ops['tn'], update_ops['fp'], update_ops['fn'],
        'update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return specificity, update_op
