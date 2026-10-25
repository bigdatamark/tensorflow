# [MODULE DOC]
# Implementation of the legacy (TF1-style) loss functions: absolute difference, cosine distance,
# hinge, Huber, log loss, mean pairwise squared error, mean squared error, sigmoid and softmax
# cross entropy and sparse softmax cross entropy, plus the reduction-type constants class and the
# shared weighting and reduction logic. Every public loss casts inputs to floating point, builds an
# element-wise loss, then calls the weighted-loss function, which applies weights, reduces
# according to the chosen reduction, and registers the scalar result in a graph collection (the
# collection is skipped when executing eagerly). Public functions are exported under the v1 loss
# API names. The package module re-exports everything here through a wildcard import.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of Loss operations for use in neural networks."""

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["losses.Reduction"])
class Reduction:
  # [CLASS DOC]
  # Namespace of string constants naming the supported loss reduction modes (none, sum, mean,
  # sum over batch size, sum by or over nonzero weights; the last two names are aliases) with
  # two class methods to list all valid keys and to validate a key. Never instantiated; holds
  # no per-instance state. Exported as the v1 reduction class.
  """Types of loss reduction.

  Contains the following values:

  * `NONE`: Un-reduced weighted losses with the same shape as input.
  * `SUM`: Scalar sum of weighted losses.
  * `MEAN`: Scalar `SUM` divided by sum of weights. DEPRECATED.
  * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
  * `SUM_OVER_NONZERO_WEIGHTS`: Scalar `SUM` divided by number of non-zero
     weights. DEPRECATED.
  * `SUM_BY_NONZERO_WEIGHTS`: Same as `SUM_OVER_NONZERO_WEIGHTS`. DEPRECATED.
  """

  NONE = "none"
  SUM = "weighted_sum"
  SUM_OVER_BATCH_SIZE = "weighted_sum_over_batch_size"
  MEAN = "weighted_mean"
  SUM_BY_NONZERO_WEIGHTS = "weighted_sum_by_nonzero_weights"
  SUM_OVER_NONZERO_WEIGHTS = SUM_BY_NONZERO_WEIGHTS

  @classmethod
  def all(cls):
    # [FUNCTION DOC]
    # Class method returning a tuple of every valid reduction key string, in the order none,
    # sum, mean, sum over batch size, sum over nonzero weights, sum by nonzero weights.
    return (
        cls.NONE,
        cls.SUM,
        cls.MEAN,
        cls.SUM_OVER_BATCH_SIZE,
        cls.SUM_OVER_NONZERO_WEIGHTS,
        cls.SUM_BY_NONZERO_WEIGHTS)

  @classmethod
  def validate(cls, key):
    # [FUNCTION DOC]
    # Class method that checks key is one of the valid reduction strings. Returns nothing;
    # raises a value error listing the valid keys if the key is not recognised.
    if key not in cls.all():
      raise ValueError(f"Invalid Reduction Key {key}. Key should be one of "
                       f"{cls.all()}.")


def _safe_mean(losses, num_present):
  # [FUNCTION DOC]
  # HELPER: Used by the weighted-loss reductions to compute means.
  # Sums the losses tensor and divides by num_present, returning zero instead of an error or
  # infinity when num_present is zero. Output is a scalar tensor named value.
  """Computes a safe mean of the losses.

  Args:
    losses: `Tensor` whose elements contain individual loss measurements.
    num_present: The number of measurable elements in `losses`.

  Returns:
    A scalar representing the mean of `losses`. If `num_present` is zero,
      then zero is returned.
  """
  total_loss = math_ops.reduce_sum(losses)
  return math_ops.div_no_nan(total_loss, num_present, name="value")


def _num_present(losses, weights, per_batch=False):
  # [FUNCTION DOC]
  # HELPER: Used by the weighted-loss reduction and the pairwise squared error to normalise.
  # KNOWN ISSUE: The eager shortcut reads a private rank method of weights, which fails for
  # weights that are Python integers or other non-tensor, non-float values when executing
  # eagerly. Also the shortcut ignores the per_batch flag, so with per_batch set and a nonzero
  # scalar weight it returns the total element count instead of a per-batch tensor.
  # Counts the elements of losses that have nonzero weight after broadcasting weights to the
  # shape of losses. If weights is a nonzero Python float, or (eagerly) a nonzero scalar tensor,
  # returns the total number of elements. Otherwise builds a 0/1 presence tensor from weights
  # cast to float32, broadcasts it, and sums it: over everything (scalar result) or, if per_batch
  # is true, over all dimensions except the first (shape batch-size with kept dimensions).
  """Computes the number of elements in the loss function induced by `weights`.

  A given weights tensor induces different numbers of usable elements in the
  `losses` tensor. The `weights` tensor is broadcast across `losses` for all
  possible dimensions. For example, if `losses` is a tensor of dimension
  `[4, 5, 6, 3]` and `weights` is a tensor of shape `[4, 5]`, then `weights` is,
  in effect, tiled to match the shape of `losses`. Following this effective
  tile, the total number of present elements is the number of non-zero weights.

  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: `Tensor` of shape `[]`, `[batch_size]` or
      `[batch_size, d1, ... dK]`, where K < N.
    per_batch: Whether to return the number of elements per batch or as a sum
      total.

  Returns:
    The number of present (non-zero) elements in the losses tensor. If
      `per_batch` is `True`, the value is returned as a tensor of size
      `[batch_size]`. Otherwise, a single scalar tensor is returned.
  """
  if ((isinstance(weights, float) and weights != 0.0) or
      (context.executing_eagerly() and weights._rank() == 0  # pylint: disable=protected-access
       and not math_ops.equal(weights, 0.0))):
    return _num_elements(losses)
  with ops.name_scope(None, "num_present", (losses, weights)) as scope:
    weights = math_ops.cast(weights, dtype=dtypes.float32)
    present = array_ops.where(
        math_ops.equal(weights, 0.0),
        array_ops.zeros_like(weights),
        array_ops.ones_like(weights))
    present = weights_broadcast_ops.broadcast_weights(present, losses)
    if per_batch:
      return math_ops.reduce_sum(
          present,
          axis=math_ops.range(1, array_ops.rank(present)),
          keepdims=True,
          name=scope)
    return math_ops.reduce_sum(present, name=scope)


def _num_elements(losses):
  # [FUNCTION DOC]
  # HELPER: Used by reductions that divide by the element count.
  # Returns the total number of elements of losses as a scalar tensor cast to the losses dtype.
  """Computes the number of elements in `losses` tensor."""
  with ops.name_scope(None, "num_elements", values=[losses]) as scope:
    return math_ops.cast(array_ops.size(losses, name=scope), dtype=losses.dtype)


@tf_export(v1=["losses.compute_weighted_loss"])
@dispatch.add_dispatch_support
def compute_weighted_loss(
    losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  # [FUNCTION DOC]
  # KNOWN ISSUE: The docstring says a missing weights value raises a value error, but nothing
  # checks for that; a none weights value fails later with a different error. Results are
  # computed in float32 and cast back, so higher precision inputs lose precision.
  # Core of all the losses: multiplies the elementwise losses by weights (both cast to float32)
  # and reduces according to reduction (validated first; invalid keys raise a value error):
  # none keeps the weighted losses, sum sums, mean divides by the sum of weights, nonzero-weight
  # modes divide by the count of nonzero weights, batch-size mode divides by the element
  # count (all divisions are safe against zero). The result is cast back to the input dtype and
  # added to loss_collection via the add-loss helper (skipped eagerly). Parameters: losses
  # tensor, weights (scalar or broadcastable tensor, default one), scope name, collection name,
  # and reduction. Side effect: records the reduction on the default graph for later loss
  # normalisation by distribution code. Unless in an XLA context, a control dependency asserts
  # that weights is broadcastable to losses. Returns the reduced scalar or unreduced tensor.
  """Computes the weighted loss.

  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `losses`, and must be broadcastable to `losses` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: the scope for the operations performed in computing the loss.
    loss_collection: the loss will be added to these collections.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss `Tensor` of the same type as `losses`. If `reduction` is
    `NONE`, this has the same shape as `losses`; otherwise, it is scalar.

  Raises:
    ValueError: If `weights` is `None` or the shape is not compatible with
      `losses`, or if the number of dimensions (rank) of either `losses` or
      `weights` is missing.

  Note:
    When calculating the gradient of a weighted loss contributions from
    both `losses` and `weights` are considered. If your `weights` depend
    on some model parameters but you do not want this to affect the loss
    gradient, you need to apply `tf.stop_gradient` to `weights` before
    passing them to `compute_weighted_loss`.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  Reduction.validate(reduction)
  with ops.name_scope(scope, "weighted_loss", (losses, weights)):
    # Save the `reduction` argument for loss normalization when distributing
    # to multiple replicas. Used only for estimator + v1 optimizer flow.
    ops.get_default_graph()._last_loss_reduction = reduction  # pylint: disable=protected-access

    def compute_loss(losses, weights, loss_collection, reduction):
      # [FUNCTION DOC]
      # HELPER: Nested worker of the weighted-loss function, split out so it can run with or
      # without the broadcast-check dependency.
      # Converts losses to a tensor, casts losses and weights to float32, multiplies them,
      # applies the requested reduction (none, sum, mean, nonzero-weight mean, or batch-size
      # mean), casts the result back to the original dtype, registers it in loss_collection and
      # returns it.
      losses = ops.convert_to_tensor(losses)
      input_dtype = losses.dtype
      losses = math_ops.cast(losses, dtype=dtypes.float32)
      weights = math_ops.cast(weights, dtype=dtypes.float32)
      weighted_losses = math_ops.multiply(losses, weights)
      if reduction == Reduction.NONE:
        loss = weighted_losses
      else:
        loss = math_ops.reduce_sum(weighted_losses)
        if reduction == Reduction.MEAN:
          loss = _safe_mean(
              loss, math_ops.reduce_sum(array_ops.ones_like(losses) * weights))
        elif (reduction == Reduction.SUM_BY_NONZERO_WEIGHTS or
              reduction == Reduction.SUM_OVER_NONZERO_WEIGHTS):
          loss = _safe_mean(loss, _num_present(losses, weights))
        elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
          loss = _safe_mean(loss, _num_elements(losses))

      # Convert the result back to the input type.
      loss = math_ops.cast(loss, input_dtype)
      util.add_loss(loss, loss_collection)
      return loss

    # Skip the assert_broadcastable in XLA context because asserts are not
    # supported so it only causes unnecessary ops. Also skip it because it uses
    # a DenseToDenseSetOperation op that is incompatible with XLA when
    # the shape(s) are dynamic.
    if control_flow_ops.get_enclosing_xla_context() is not None:
      return compute_loss(losses, weights, loss_collection, reduction)
    else:
      with ops.control_dependencies(
          (weights_broadcast_ops.assert_broadcastable(weights, losses),)):
        return compute_loss(losses, weights, loss_collection, reduction)


@tf_export(v1=["losses.absolute_difference"])
@dispatch.add_dispatch_support
def absolute_difference(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  # [FUNCTION DOC]
  # Absolute difference (L1) loss. Takes ground truth labels, predictions, optional weights,
  # scope name, loss collection and reduction. Raises a value error if labels or predictions is
  # none, or if their static shapes are incompatible. Casts both to float32, computes the
  # elementwise absolute value of predictions minus labels and passes it to the weighted-loss
  # function. Returns the reduced loss (scalar unless reduction is none). Side effect: adds
  # the loss to the collection in graph mode.
  """Adds an Absolute Difference loss to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a `Tensor` of
  shape `[batch_size]`, then the total loss for each sample of the batch is
  rescaled by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of
      `labels` or if the shape of `weights` is invalid or if `labels`
      or `predictions` is None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if labels is None:
    raise ValueError("Argument `labels` must not be None.")
  if predictions is None:
    raise ValueError("Argument `predictions` must not be None.")
  with ops.name_scope(scope, "absolute_difference",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.cast(predictions, dtype=dtypes.float32)
    labels = math_ops.cast(labels, dtype=dtypes.float32)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = math_ops.abs(math_ops.subtract(predictions, labels))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


@tf_export(v1=["losses.cosine_distance"])
@dispatch.add_dispatch_support
@deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def cosine_distance(
    labels, predictions, axis=None, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS,
    dim=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: The dim argument is deprecated and merged into axis; the axis default of none
  # is not actually optional, since a missing axis raises a value error. The function does not
  # normalise inputs, despite the loss name; it assumes unit-normalised vectors.
  # Cosine distance loss. Resolves axis from the deprecated dim alias, then raises a value error
  # if axis, labels or predictions is none or shapes are incompatible. Casts to float32, takes
  # one minus the sum over axis (dimension kept) of the elementwise product of predictions and
  # labels, and passes that to the weighted-loss function with the given weights, scope,
  # collection and reduction. Returns the reduced loss; adds it to the collection in graph mode.
  """Adds a cosine-distance loss to the training procedure.

  Note that the function assumes that `predictions` and `labels` are already
  unit-normalized.

  Args:
    labels: `Tensor` whose shape matches 'predictions'
    predictions: An arbitrary matrix.
    axis: The dimension along which the cosine distance is computed.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: Type of reduction to apply to loss.
    dim: The old (deprecated) name for `axis`.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If `predictions` shape doesn't match `labels` shape, or
      `axis`, `labels`, `predictions` or `weights` is `None`.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  axis = deprecated_argument_lookup("axis", axis, "dim", dim)
  if axis is None:
    raise ValueError("You must specify argument `axis`.")
  if labels is None:
    raise ValueError("Argument `labels` must not be None.")
  if predictions is None:
    raise ValueError("Argument `predictions` must not be None.")
  with ops.name_scope(scope, "cosine_distance_loss",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.cast(predictions, dtype=dtypes.float32)
    labels = math_ops.cast(labels, dtype=dtypes.float32)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    radial_diffs = math_ops.multiply(predictions, labels)
    losses = 1 - math_ops.reduce_sum(radial_diffs, axis=(axis,), keepdims=True)
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


@tf_export(v1=["losses.hinge_loss"])
@dispatch.add_dispatch_support
def hinge_loss(labels, logits, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  # [FUNCTION DOC]
  # Hinge loss for binary classification. Inputs: labels (values zero or one), logits (float
  # scores), weights, scope, loss collection, reduction. Raises a value error if labels or
  # logits is none or their static shapes are incompatible. Converts labels from zero/one to
  # minus one/plus one, computes the rectified value of one minus labels times logits, and
  # passes it to the weighted-loss function. Returns the reduced loss; graph-mode side effect
  # of adding it to the collection.
  """Adds a hinge loss to the training procedure.

  Args:
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0. Internally
      the {0,1} labels are converted to {-1,1} when calculating the hinge loss.
    logits: The logits, a float tensor. Note that logits are assumed to be
      unbounded and 0-centered. A value > 0 (resp. < 0) is considered a positive
      (resp. negative) binary prediction.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shapes of `logits` and `labels` don't match or
      if `labels` or `logits` is None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if labels is None:
    raise ValueError("Argument `labels` must not be None.")
  if logits is None:
    raise ValueError("Argument `logits` must not be None.")
  with ops.name_scope(scope, "hinge_loss", (logits, labels, weights)) as scope:
    logits = math_ops.cast(logits, dtype=dtypes.float32)
    labels = math_ops.cast(labels, dtype=dtypes.float32)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    all_ones = array_ops.ones_like(labels)
    labels = math_ops.subtract(2 * labels, all_ones)
    losses = nn_ops.relu(
        math_ops.subtract(all_ones, math_ops.multiply(labels, logits)))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


@tf_export(v1=["losses.huber_loss"])
@dispatch.add_dispatch_support
def huber_loss(labels, predictions, weights=1.0, delta=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  # [FUNCTION DOC]
  # Huber loss: quadratic for errors up to delta and linear beyond. Inputs: labels, predictions,
  # weights, delta threshold (default one), scope, collection, reduction. Raises a value error
  # if labels or predictions is none or shapes are incompatible. Computes the absolute error,
  # splits it into a quadratic part capped at delta and the linear remainder (written so the
  # gradient at exactly delta is not doubled), and combines half the squared quadratic part
  # with delta times the linear part. Then applies the weighted-loss function and returns the
  # reduced loss; adds it to the collection in graph mode.
  """Adds a [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss) term to the training procedure.

  For each value x in `error=labels-predictions`, the following is calculated:

  ```
    0.5 * x^2                  if |x| <= d
    0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```

  where d is `delta`.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    delta: `float`, the point where the huber loss function changes from a
      quadratic to linear.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or
     `predictions` is None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if labels is None:
    raise ValueError("Argument `labels` must not be None.")
  if predictions is None:
    raise ValueError("Argument `predictions` must not be None.")
  with ops.name_scope(scope, "huber_loss",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.cast(predictions, dtype=dtypes.float32)
    labels = math_ops.cast(labels, dtype=dtypes.float32)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    error = math_ops.subtract(predictions, labels)
    abs_error = math_ops.abs(error)
    quadratic = math_ops.minimum(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = math_ops.subtract(abs_error, quadratic)
    losses = math_ops.add(
        math_ops.multiply(
            ops.convert_to_tensor(0.5, dtype=quadratic.dtype),
            math_ops.multiply(quadratic, quadratic)),
        math_ops.multiply(delta, linear))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


@tf_export(v1=["losses.log_loss"])
@dispatch.add_dispatch_support
def log_loss(labels, predictions, weights=1.0, epsilon=1e-7, scope=None,
             loss_collection=ops.GraphKeys.LOSSES,
             reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  # [FUNCTION DOC]
  # Log loss (binary cross entropy on probabilities). Inputs: labels, predictions (expected
  # probabilities between zero and one), weights, epsilon added inside the logarithms to avoid
  # taking the log of zero (default 1e-7), scope, collection, reduction. Raises a value error if
  # labels or predictions is none or shapes are incompatible. Computes minus labels times log of
  # predictions plus epsilon, minus one-minus-labels times log of one minus predictions plus
  # epsilon, and passes it to the weighted-loss function. Returns the reduced loss.
  """Adds a Log Loss term to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    epsilon: A small increment to add to avoid taking a log of zero.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions`
      is None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if labels is None:
    raise ValueError("Argument `labels` must not be None.")
  if predictions is None:
    raise ValueError("Argument `predictions` must not be None.")
  with ops.name_scope(scope, "log_loss",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.cast(predictions, dtype=dtypes.float32)
    labels = math_ops.cast(labels, dtype=dtypes.float32)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = -math_ops.multiply(
        labels,
        math_ops.log(predictions + epsilon)) - math_ops.multiply(
            (1 - labels), math_ops.log(1 - predictions + epsilon))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


# TODO(b/37208492): Add reduction arg.
@tf_export(v1=["losses.mean_pairwise_squared_error"])
@dispatch.add_dispatch_support
def mean_pairwise_squared_error(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES):
  # [FUNCTION DOC]
  # Mean pairwise squared error: for each batch sample, averages the squared differences of
  # prediction-minus-label errors over all pairs of elements within that sample. Inputs: labels,
  # predictions, weights (scalar, per-sample or per-element), scope, collection. There is no
  # reduction argument; the result is always a scalar. Raises a value error if labels or
  # predictions is none or shapes are incompatible. Casts to float32 and, unless in an XLA
  # context, requires weights to be broadcastable to labels. Registers the loss in the
  # collection (graph mode) and returns it; yields zero if no elements are present.
  """Adds a pairwise-errors-squared loss to the training procedure.

  Unlike `mean_squared_error`, which is a measure of the differences between
  corresponding elements of `predictions` and `labels`,
  `mean_pairwise_squared_error` is a measure of the differences between pairs of
  corresponding elements of `predictions` and `labels`.

  For example, if `labels`=[a, b, c] and `predictions`=[x, y, z], there are
  three pairs of differences are summed to compute the loss:
    loss = [ ((a-b) - (x-y)).^2 + ((a-c) - (x-z)).^2 + ((b-c) - (y-z)).^2 ] / 3

  Note that since the inputs are of shape `[batch_size, d0, ... dN]`, the
  corresponding pairs are computed within each batch sample but not across
  samples within a batch. For example, if `predictions` represents a batch of
  16 grayscale images of dimension [batch_size, 100, 200], then the set of pairs
  is drawn from each image, but not across images.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector.

  Args:
    labels: The ground truth output tensor, whose shape must match the shape of
      `predictions`.
    predictions: The predicted outputs, a tensor of size
      `[batch_size, d0, .. dN]` where N+1 is the total number of dimensions in
      `predictions`.
    weights: Coefficients for the loss a scalar, a tensor of shape
      `[batch_size]` or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.

  Returns:
    A scalar `Tensor` that returns the weighted loss.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions`
      is None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if labels is None:
    raise ValueError("Argument `labels` must not be None.")
  if predictions is None:
    raise ValueError("Argument `predictions` must not be None.")
  with ops.name_scope(scope, "mean_pairwise_squared_error",
                      (predictions, labels, weights)) as scope:
    weights = math_ops.cast(weights, dtype=dtypes.float32)
    labels = math_ops.cast(labels, dtype=dtypes.float32)

    def compute_loss(labels, predictions, weights, loss_collection):
      # [FUNCTION DOC]
      # HELPER: Nested worker of the pairwise squared error, run with or without the
      # broadcast-check dependency.
      # Computes per-sample sums of squared differences and sums of differences, normalises them
      # by the count of present elements and by that count times itself minus one (safe
      # division), forms the pairwise term as twice the first minus twice the second, multiplies
      # by weights and sums. Returns zero if no elements are present. Adds the result to
      # loss_collection and returns it.
      predictions = math_ops.cast(predictions, dtype=dtypes.float32)
      predictions.get_shape().assert_is_compatible_with(labels.get_shape())

      diffs = math_ops.subtract(predictions, labels)

      axis = math_ops.range(1, array_ops.rank(diffs))

      sum_squares_diff_per_batch = math_ops.reduce_sum(
          math_ops.square(diffs), axis=axis, keepdims=True)
      num_present_per_batch = _num_present(diffs, weights, per_batch=True)

      term1 = 2.0 * math_ops.div_no_nan(
          sum_squares_diff_per_batch,
          math_ops.maximum(num_present_per_batch - 1, 0),
          name="value")

      sum_diff = math_ops.reduce_sum(diffs, axis=axis, keepdims=True)
      term2 = 2.0 * math_ops.div_no_nan(
          math_ops.square(sum_diff),
          math_ops.maximum(
              math_ops.multiply(num_present_per_batch,
                                num_present_per_batch - 1), 0),
          name="value")

      weighted_losses = math_ops.multiply(term1 - term2, weights)
      loss = math_ops.reduce_sum(weighted_losses)

      mean_loss = array_ops.where(
          math_ops.reduce_sum(num_present_per_batch) > 0,
          loss,
          array_ops.zeros_like(loss),
          name="value")
      util.add_loss(mean_loss, loss_collection)
      return mean_loss

    # Skip the assert_broadcastable in XLA context because asserts are not
    # supported so it only causes unnecessary ops. Also skip it because it uses
    # a DenseToDenseSetOperation op that is incompatible with XLA when
    # the shape(s) are dynamic.
    if control_flow_ops.get_enclosing_xla_context() is not None:
      return compute_loss(labels, predictions, weights, loss_collection)
    else:
      with ops.control_dependencies(
          (weights_broadcast_ops.assert_broadcastable(weights, labels),)):
        return compute_loss(labels, predictions, weights, loss_collection)


@tf_export(v1=["losses.mean_squared_error"])
@dispatch.add_dispatch_support
def mean_squared_error(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  # [FUNCTION DOC]
  # Mean squared error (L2) loss. Inputs: labels, predictions, weights, scope, loss collection
  # and reduction. Raises a value error if labels or predictions is none or their static shapes
  # are incompatible. Casts both to float32, computes the elementwise squared difference of
  # predictions and labels, and passes it to the weighted-loss function. Returns the reduced
  # loss (scalar unless reduction is none); adds it to the collection in graph mode.
  """Adds a Sum-of-Squares loss to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions`
      is None.

  @compatibility(TF2)

  `tf.compat.v1.losses.mean_squared_error` is mostly compatible with eager
  execution and `tf.function`. But, the `loss_collection` argument is
  ignored when executing eagerly and no loss will be written to the loss
  collections. You will need to either hold on to the return value manually
  or rely on `tf.keras.Model` loss tracking.


  To switch to native TF2 style, instantiate the
   `tf.keras.losses.MeanSquaredError` class and call the object instead.


  #### Structural Mapping to Native TF2

  Before:

  ```python
  loss = tf.compat.v1.losses.mean_squared_error(
    labels=labels,
    predictions=predictions,
    weights=weights,
    reduction=reduction)
  ```

  After:

  ```python
  loss_fn = tf.keras.losses.MeanSquaredError(
    reduction=reduction)
  loss = loss_fn(
    y_true=labels,
    y_pred=predictions,
    sample_weight=weights)
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name     | Note                       |
  | :-------------------- | :--------------- | :------------------------- |
  | `labels`              | `y_true`         | In `__call__()` method     |
  | `predictions`         | `y_pred`         | In `__call__()` method     |
  | `weights`             | `sample_weight`  | In `__call__()` method.    |
  : : : The shape requirements for `sample_weight` is different from      :
  : : : `weights`. Please check the [argument definition][api_docs] for   :
  : : : details.                                                          :
  | `scope`               | Not supported    | -                          |
  | `loss_collection`     | Not supported    | Losses should be tracked   |
  : : : explicitly or with Keras APIs, for example, [add_loss][add_loss], :
  : : : instead of via collections                                        :
  | `reduction`           | `reduction`      | In constructor. Value of   |
  : : : `tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE`,              :
  : : : `tf.compat.v1.losses.Reduction.SUM`,                              :
  : : : `tf.compat.v1.losses.Reduction.NONE` in                           :
  : : : `tf.compat.v1.losses.softmax_cross_entropy` correspond to         :
  : : : `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE`,                  :
  : : : `tf.keras.losses.Reduction.SUM`,                                  :
  : : : `tf.keras.losses.Reduction.NONE`, respectively. If you            :
  : : : used other value for `reduction`, including the default value     :
  : : :  `tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS`, there is :
  : : : no directly corresponding value. Please modify the loss           :
  : : : implementation manually.                                          :

  [add_loss]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss
  [api_docs]:https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError#__call__


  #### Before & After Usage Example

  Before:

  >>> y_true = [1, 2, 3]
  >>> y_pred = [1, 3, 5]
  >>> weights = [0, 1, 0.25]
  >>> # samples with zero-weight are excluded from calculation when `reduction`
  >>> # argument is set to default value `Reduction.SUM_BY_NONZERO_WEIGHTS`
  >>> tf.compat.v1.losses.mean_squared_error(
  ...    labels=y_true,
  ...    predictions=y_pred,
  ...    weights=weights).numpy()
  1.0

  >>> tf.compat.v1.losses.mean_squared_error(
  ...    labels=y_true,
  ...    predictions=y_pred,
  ...    weights=weights,
  ...    reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE).numpy()
  0.66667

  After:

  >>> y_true = [[1.0], [2.0], [3.0]]
  >>> y_pred = [[1.0], [3.0], [5.0]]
  >>> weights = [1, 1, 0.25]
  >>> mse = tf.keras.losses.MeanSquaredError(
  ...    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  >>> mse(y_true=y_true, y_pred=y_pred, sample_weight=weights).numpy()
  0.66667

  @end_compatibility
  """
  if labels is None:
    raise ValueError("Argument `labels` must not be None.")
  if predictions is None:
    raise ValueError("Argument `predictions` must not be None.")
  with ops.name_scope(scope, "mean_squared_error",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.cast(predictions, dtype=dtypes.float32)
    labels = math_ops.cast(labels, dtype=dtypes.float32)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = math_ops.squared_difference(predictions, labels)
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


@tf_export(v1=["losses.sigmoid_cross_entropy"])
@dispatch.add_dispatch_support
def sigmoid_cross_entropy(
    multi_class_labels, logits, weights=1.0, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  # [FUNCTION DOC]
  # Sigmoid cross entropy loss for multi-label or binary classification. Inputs:
  # multi_class_labels (values in zero to one, same shape as logits), logits, weights, a label
  # smoothing amount (default zero; when positive, labels are pulled towards one half by
  # that fraction), scope, collection and reduction. Raises a value error if labels or logits is
  # none or static shapes are incompatible. Casts labels to the logits dtype, computes the
  # elementwise sigmoid cross entropy with logits, and passes it to the weighted-loss function.
  # Returns the reduced loss; adds it to the collection in graph mode.
  """Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/2:

      new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                              + 0.5 * label_smoothing

  Args:
    multi_class_labels: `[batch_size, num_classes]` target integer labels in
      `{0, 1}`.
    logits: Float `[batch_size, num_classes]` logits outputs of the network.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
    `multi_class_labels`, and must be broadcastable to `multi_class_labels`
    (i.e., all dimensions must be either `1`, or the same as the
    corresponding `losses` dimension).
    label_smoothing: If greater than `0` then smooth the labels.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has the same shape as `logits`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of
      `multi_class_labels` or if the shape of `weights` is invalid, or if
      `weights` is None.  Also if `multi_class_labels` or `logits` is None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if multi_class_labels is None:
    raise ValueError("Argument `multi_class_labels` must not be None.")
  if logits is None:
    raise ValueError("Argument `logits` must not be None.")
  with ops.name_scope(scope, "sigmoid_cross_entropy_loss",
                      (logits, multi_class_labels, weights)) as scope:
    logits = ops.convert_to_tensor(logits)
    multi_class_labels = math_ops.cast(multi_class_labels, logits.dtype)
    logits.get_shape().assert_is_compatible_with(multi_class_labels.get_shape())

    if label_smoothing > 0:
      multi_class_labels = (multi_class_labels * (1 - label_smoothing) +
                            0.5 * label_smoothing)

    losses = nn.sigmoid_cross_entropy_with_logits(labels=multi_class_labels,
                                                  logits=logits,
                                                  name="xentropy")
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


@tf_export(v1=["losses.softmax_cross_entropy"])
@dispatch.add_dispatch_support
def softmax_cross_entropy(
    onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  # [FUNCTION DOC]
  # Softmax cross entropy loss with one-hot labels. Inputs: onehot_labels, logits (same shape),
  # weights, label smoothing amount (default zero; positive values move the one-hot targets
  # towards a uniform distribution over classes), scope, collection and reduction. Raises a value
  # error if labels or logits is none or shapes are incompatible. Casts labels to the logits
  # dtype, applies smoothing, blocks gradients flowing into the labels, computes the per-example
  # softmax cross entropy with logits (version two op) and passes it to the weighted-loss
  # function. Returns the reduced loss; adds it to the collection in graph mode.
  r"""Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits_v2.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes

  Note that `onehot_labels` and `logits` must have the same shape,
  e.g. `[batch_size, num_classes]`. The shape of `weights` must be
  broadcastable to loss, whose shape is decided by the shape of `logits`.
  In case the shape of `logits` is `[batch_size, num_classes]`, loss is
  a `Tensor` of shape `[batch_size]`.

  Args:
    onehot_labels: One-hot-encoded labels.
    logits: Logits outputs of the network.
    weights: Optional `Tensor` that is broadcastable to loss.
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has shape `[batch_size]`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`
      or if the shape of `weights` is invalid or if `weights` is None.  Also if
      `onehot_labels` or `logits` is None.

  @compatibility(TF2)

  `tf.compat.v1.losses.softmax_cross_entropy` is mostly compatible with eager
  execution and `tf.function`. But, the `loss_collection` argument is
  ignored when executing eagerly and no loss will be written to the loss
  collections. You will need to either hold on to the return value manually
  or rely on `tf.keras.Model` loss tracking.


  To switch to native TF2 style, instantiate the
   `tf.keras.losses.CategoricalCrossentropy` class with `from_logits` set
  as `True` and call the object instead.


  #### Structural Mapping to Native TF2

  Before:

  ```python
  loss = tf.compat.v1.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels,
    logits=logits,
    weights=weights,
    label_smoothing=smoothing)
  ```

  After:

  ```python
  loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=smoothing)
  loss = loss_fn(
    y_true=onehot_labels,
    y_pred=logits,
    sample_weight=weights)
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name     | Note                       |
  | :-------------------- | :--------------- | :------------------------- |
  |  -                    | `from_logits`    | Set `from_logits` as True  |
  :                       :                  : to have identical behavior :
  | `onehot_labels`       | `y_true`         | In `__call__()` method     |
  | `logits`              | `y_pred`         | In `__call__()` method     |
  | `weights`             | `sample_weight`  | In `__call__()` method     |
  | `label_smoothing`     | `label_smoothing`| In constructor             |
  | `scope`               | Not supported    | -                          |
  | `loss_collection`     | Not supported    | Losses should be tracked   |
  :                       :                  : explicitly or with Keras   :
  :                       :                  : APIs, for example,         :
  :                       :                  : [add_loss][add_loss],      :
  :                       :                  : instead of via collections :
  | `reduction`           | `reduction`      | In constructor. Value of   |
  : : : `tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE`,              :
  : : : `tf.compat.v1.losses.Reduction.SUM`,                              :
  : : : `tf.compat.v1.losses.Reduction.NONE` in                           :
  : : : `tf.compat.v1.losses.softmax_cross_entropy` correspond to         :
  : : : `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE`,                  :
  : : : `tf.keras.losses.Reduction.SUM`,                                  :
  : : : `tf.keras.losses.Reduction.NONE`, respectively. If you            :
  : : : used other value for `reduction`, including the default value     :
  : : :  `tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS`, there is :
  : : : no directly corresponding value. Please modify the loss           :
  : : : implementation manually.                                          :

  [add_loss]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss


  #### Before & After Usage Example

  Before:

  >>> y_true = [[0, 1, 0], [0, 0, 1]]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> weights = [0.3, 0.7]
  >>> smoothing = 0.2
  >>> tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred, weights=weights,
  ...   label_smoothing=smoothing).numpy()
  0.57618

  After:

  >>> cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
  ...   label_smoothing=smoothing)
  >>> cce(y_true, y_pred, sample_weight=weights).numpy()
  0.57618

  @end_compatibility
  """
  if onehot_labels is None:
    raise ValueError("Argument `onehot_labels` must not be None.")
  if logits is None:
    raise ValueError("Argument `logits` must not be None.")
  with ops.name_scope(scope, "softmax_cross_entropy_loss",
                      (logits, onehot_labels, weights)) as scope:
    logits = ops.convert_to_tensor(logits)
    onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
    logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

    if label_smoothing > 0:
      num_classes = math_ops.cast(
          array_ops.shape(onehot_labels)[-1], logits.dtype)
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    onehot_labels = array_ops.stop_gradient(
        onehot_labels, name="labels_stop_gradient")
    losses = nn.softmax_cross_entropy_with_logits_v2(
        labels=onehot_labels, logits=logits, name="xentropy")

    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


# TODO(ptucker): Merge this with similar method in metrics_impl.
def _remove_squeezable_dimensions(
    labels, predictions, weights=None, expected_rank_diff=0):
  # [FUNCTION DOC]
  # HELPER: Used by the sparse softmax cross entropy to align label, prediction and weight ranks.
  # KNOWN ISSUE: The docstring says it returns predictions, labels and weights, but the code
  # returns labels, predictions, weights in that order. A TODO notes it duplicates a similar
  # helper in the metrics implementation.
  # Squeezes labels or predictions through the confusion-matrix helper when their ranks differ
  # from expected_rank_diff by one. If weights is given it is converted to a tensor and squeezed
  # on the last dimension when its rank exceeds that of labels by one, using static ranks if
  # known and otherwise a runtime conditional. Returns the (labels, predictions, weights) tuple.
  """Internal version of _remove_squeezable_dimensions which handles weights.

  Squeezes `predictions` and `labels` if their ranks differ from expected by
  exactly 1.
  Squeezes `weights` if its rank is 1 more than the new rank of `predictions`

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    labels: Label values, a `Tensor` whose dimensions match `predictions`.
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    weights: Optional weight `Tensor`. It will be squeezed if it's not scalar,
      and its rank is 1 more than the new rank of `labels`.
    expected_rank_diff: Expected result of `rank(predictions) - rank(labels)`.

  Returns:
    Tuple of `predictions`, `labels` and `weights`, possibly with the last
    dimension squeezed.
  """
  labels, predictions = confusion_matrix.remove_squeezable_dimensions(
      labels, predictions, expected_rank_diff=expected_rank_diff)

  if weights is not None:
    weights = ops.convert_to_tensor(weights)
    labels_rank = labels.get_shape().ndims
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims

    if (labels_rank is not None) and (weights_rank is not None):
      # Use static rank.
      rank_diff = weights_rank - labels_rank
      if rank_diff == 1:
        weights = array_ops.squeeze(weights, [-1])
      return labels, predictions, weights

    # Use dynamic rank.
    rank_diff = array_ops.rank(weights) - array_ops.rank(labels)
    if (weights_rank is None) or (
        weights_rank > 0 and weights_shape.dims[-1].is_compatible_with(1)):
      weights = cond.cond(
          math_ops.equal(1, rank_diff),
          lambda: array_ops.squeeze(weights, [-1]),
          lambda: weights)

  return labels, predictions, weights


@tf_export(v1=["losses.sparse_softmax_cross_entropy"])
@dispatch.add_dispatch_support
def sparse_softmax_cross_entropy(
    labels, logits, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  # [FUNCTION DOC]
  # Sparse softmax cross entropy loss with integer class-id labels. Inputs: labels (class ids,
  # rank one less than logits), logits (unscaled scores), weights, scope, collection and
  # reduction. Raises a value error if labels or logits is none. Aligns ranks of labels, logits
  # and weights (expecting logits to have exactly one more dimension), computes per-example
  # sparse softmax cross entropy and passes it to the weighted-loss function. Returns the reduced
  # loss; adds it to the collection in graph mode.
  """Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.

  Args:
    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
      `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
      must be an index in `[0, num_classes)`. Other values will raise an
      exception when this op is run on CPU, and return `NaN` for corresponding
      loss and gradient rows on GPU.
    logits: Unscaled log probabilities of shape
      `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float16`, `float32` or
      `float64`.
    weights: Coefficients for the loss. This must be scalar or broadcastable to
      `labels` (i.e. same rank and each dimension is either 1 or the same).
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has the same shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shapes of `logits`, `labels`, and `weights` are
      incompatible, or if any of them are None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if labels is None:
    raise ValueError("Argument `labels` must not be None.")
  if logits is None:
    raise ValueError("Argument `logits` must not be None.")
  with ops.name_scope(scope, "sparse_softmax_cross_entropy_loss",
                      (logits, labels, weights)) as scope:
    # As documented above in Args, labels contain class IDs and logits contains
    # 1 probability per class ID, so we expect rank(logits) - rank(labels) == 1;
    # therefore, expected_rank_diff=1.
    labels, logits, weights = _remove_squeezable_dimensions(
        labels, logits, weights, expected_rank_diff=1)
    losses = nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                         logits=logits,
                                                         name="xentropy")
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)
