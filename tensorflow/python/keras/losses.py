# [MODULE DOC]
# Built-in Keras loss functions and loss classes (legacy tensorflow.python.keras implementation).
# Defines the Loss base class and LossFunctionWrapper (turns a plain function into a configurable
# object with a reduction mode and name), one class per built-in loss (mean squared/absolute/
# percentage/logarithmic error, binary/categorical/sparse crossentropy, hinge variants, Poisson,
# log-cosh, KL divergence, Huber, cosine similarity) and the matching plain functions that compute
# per-sample losses. Several functions also register RaggedTensor implementations through the
# dispatch mechanism, which apply the dense loss row by row. Also provides the lookup helpers
# serialize, deserialize and get, lower-case and upper-case aliases (mse, MAE, bce, ...), and a
# mapping from loss functions to the label dtype they expect.
# Import-time side effects: the dispatch decorators register the ragged-tensor handlers and
# dispatch support for the loss functions; no flags are defined.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-classes-have-attributes
"""Built-in loss functions."""

import abc
import functools

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.tools.docs import doc_controls


class Loss:
  # [CLASS DOC]
  # KNOWN ISSUE: the class does not derive from an abstract base class, so the abstract-method
  # marker on call is not enforced; a subclass that forgets call only fails with a
  # not-implemented error when invoked.
  # Base class of all loss objects. Holds the reduction mode (how per-sample losses are combined,
  # default automatic) and an optional name, plus a name scope string derived from the name or
  # the class name. Instances are callable: they compute per-sample losses via the subclass
  # call method and then apply sample weighting and the reduction. Supports round-tripping
  # through a configuration mapping (get_config and from_config). Subclasses implement call.
  """Loss base class.

  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

  Example subclass implementation:

  ```python
  class MeanSquaredError(Loss):

    def call(self, y_true, y_pred):
      y_pred = tf.convert_to_tensor_v2(y_pred)
      y_true = tf.cast(y_true, y_pred.dtype)
      return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
  ```

  When used with `tf.distribute.Strategy`, outside of built-in training loops
  such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
  types, and reduce losses explicitly in your training loop. Using 'AUTO' or
  'SUM_OVER_BATCH_SIZE' will raise an error.

  Please see this custom training [tutorial](
    https://www.tensorflow.org/tutorials/distribute/custom_training) for more
  details on this.

  You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:

  ```python
  with strategy.scope():
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ....
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
            (1. / global_batch_size))
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (a reduction mode from the reduction enumeration, default
    # automatic) and name (optional string). Validates the reduction (raises a value error for an
    # unknown mode), stores both, disables the "sum over batch size outside built-in training"
    # allowance (a private flag that Keras training code may switch on) and computes the name
    # scope. Returns nothing.
    """Initializes `Loss` class.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance.
    """
    losses_utils.ReductionV2.validate(reduction)
    self.reduction = reduction
    self.name = name
    # SUM_OVER_BATCH is only allowed in losses managed by `fit` or
    # CannedEstimators.
    self._allow_sum_over_batch_size = False
    self._set_name_scope()

  def _set_name_scope(self):
    # [FUNCTION DOC]
    # HELPER: internal helper of the Loss constructor that prepares the name scope string.
    # Sets the private name scope attribute: the class name when no name was given, the word
    # lambda when the name is the angle-bracket lambda name, otherwise the name with leading and
    # trailing underscores removed. Takes no inputs besides the instance and returns nothing.
    """Creates a valid `name_scope` name."""
    if self.name is None:
      self._name_scope = self.__class__.__name__
    elif self.name == '<lambda>':
      self._name_scope = 'lambda'
    else:
      # E.g. '_my_loss' => 'my_loss'
      self._name_scope = self.name.strip('_')

  def __call__(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment about stripping angle brackets from lambda names sits above code
    # that does not do that; the stripping is done in the name scope helper at construction.
    # Computes the weighted, reduced loss. Inputs: y_true (ground truth values), y_pred
    # (predictions) and optional sample_weight (scalar, per-sample vector, or shape broadcastable
    # to the loss shape). Enters a backend name scope (plus a graph context when any input is a
    # symbolic tensor), calls the subclass call method (eagerly, or converted with autograph when
    # building a graph) to get per-sample losses, then applies sample weights and the reduction
    # through the shared weighted-loss utility. Returns a float tensor: per-sample losses when the
    # reduction is none, otherwise a scalar. Raises a value error if used under a distribution
    # strategy with automatic or sum-over-batch-size reduction outside built-in training loops, or
    # if the sample weight shape is invalid.
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
      sample_weight: Optional `sample_weight` acts as a coefficient for the
        loss. If a scalar is provided, then the loss is simply scaled by the
        given value. If `sample_weight` is a tensor of size `[batch_size]`, then
        the total loss for each sample of the batch is rescaled by the
        corresponding element in the `sample_weight` vector. If the shape of
        `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to
        this shape), then each loss element of `y_pred` is scaled
        by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
          functions reduce by 1 dimension, usually axis=-1.)

    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
        shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
        because all loss functions reduce by 1 dimension, usually axis=-1.)

    Raises:
      ValueError: If the shape of `sample_weight` is invalid.
    """
    # If we are wrapping a lambda function strip '<>' from the name as it is not
    # accepted in scope name.
    graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
        y_true, y_pred, sample_weight)
    with backend.name_scope(self._name_scope), graph_ctx:
      if context.executing_eagerly():
        call_fn = self.call
      else:
        call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
      losses = call_fn(y_true, y_pred)
      return losses_utils.compute_weighted_loss(
          losses, sample_weight, reduction=self._get_reduction())

  @classmethod
  def from_config(cls, config):
    # [FUNCTION DOC]
    # Class method that builds a new instance from a configuration mapping (as produced by
    # get_config) by passing its entries as constructor keyword arguments. Returns the new
    # instance; raises a type error if the mapping has keys the constructor does not accept.
    """Instantiates a `Loss` from its config (output of `get_config()`).

    Args:
        config: Output of `get_config()`.

    Returns:
        A `Loss` instance.
    """
    return cls(**config)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a new mapping holding the reduction mode and the name of this loss, enough to
    # recreate it with from_config. No side effects.
    """Returns the config dictionary for a `Loss` instance."""
    return {'reduction': self.reduction, 'name': self.name}

  @abc.abstractmethod
  @doc_controls.for_subclass_implementers
  def call(self, y_true, y_pred):
    # [FUNCTION DOC]
    # KNOWN ISSUE: marked abstract but the class is not an abstract base class, so nothing stops
    # instantiation; the body always raises a not-implemented error.
    # Extension point for subclasses: given ground truth and predictions, return per-sample loss
    # values with the last dimension reduced. The base version only raises a not-implemented
    # error.
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

    Returns:
      Loss values with the shape `[batch_size, d0, .. dN-1]`.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  def _get_reduction(self):
    # [FUNCTION DOC]
    # HELPER: resolves the effective reduction mode for the Loss call path.
    # Raises a value error when a distribution strategy is active, the private allowance flag is
    # off and the reduction is automatic or sum-over-batch-size (users must then pick sum or none
    # and reduce themselves). Otherwise returns sum-over-batch-size for the automatic mode, or
    # the configured reduction unchanged.
    """Handles `AUTO` reduction cases and returns the reduction value."""
    if (not self._allow_sum_over_batch_size and
        distribute_lib.has_strategy() and
        (self.reduction == losses_utils.ReductionV2.AUTO or
         self.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)):
      raise ValueError(
          'Please use `tf.keras.losses.Reduction.SUM` or '
          '`tf.keras.losses.Reduction.NONE` for loss reduction when losses are '
          'used with `tf.distribute.Strategy` outside of the built-in training '
          'loops. You can implement '
          '`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch '
          'size like:\n```\nwith strategy.scope():\n'
          '    loss_obj = tf.keras.losses.CategoricalCrossentropy('
          'reduction=tf.keras.losses.Reduction.NONE)\n....\n'
          '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * '
          '(1. / global_batch_size)\n```\nPlease see '
          'https://www.tensorflow.org/tutorials/distribute/custom_training'
          ' for more details.')

    if self.reduction == losses_utils.ReductionV2.AUTO:
      return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
    return self.reduction


class LossFunctionWrapper(Loss):
  # [CLASS DOC]
  # HELPER: shared base of the built-in loss classes that simply wrap a loss function.
  # Stores the wrapped function (public attribute fn) and extra keyword arguments (private
  # mapping) that are forwarded on every evaluation. Its call method aligns ranks of tensor inputs
  # and applies the function; its configuration includes the extra keyword arguments.
  """Wraps a loss function in the `Loss` class."""

  def __init__(self,
               fn,
               reduction=losses_utils.ReductionV2.AUTO,
               name=None,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: fn (loss function taking true values, predicted values and optional
    # keyword arguments), reduction (default automatic), name (optional) and any extra keyword
    # arguments, which are stored and passed to fn on every call. Calls the base constructor and
    # stores fn and the keyword arguments. Returns nothing.
    """Initializes `LossFunctionWrapper` class.

    Args:
      fn: The loss function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super().__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    # [FUNCTION DOC]
    # Evaluates the wrapped function. Inputs: y_true and y_pred. When both are tensors, their
    # ranks are reconciled by squeezing or expanding a trailing dimension of size one. The wrapped
    # function is converted with autograph (honoring the current autograph control status) and
    # called with the inputs plus the stored keyword arguments. Returns per-sample loss values.
    """Invokes the `LossFunctionWrapper` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Loss values per sample.
    """
    if tensor_util.is_tf_type(y_pred) and tensor_util.is_tf_type(y_true):
      y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

    ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
    return ag_fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration (reduction and name) merged with the stored extra keyword
    # arguments; any value that is a tensor or variable is evaluated to a plain value first
    # (this requires a usable backend session or eager mode). Returns a new mapping.
    config = {}
    for k, v in self._fn_kwargs.items():
      config[k] = backend.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MeanSquaredError(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for mean squared error: wraps the mean squared error function so it can be used
  # with a reduction mode, optional sample weights and the compile API. Holds no extra state.
  """Computes the mean of squares of errors between labels and predictions.

  `loss = square(y_true - y_pred)`

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> mse = tf.keras.losses.MeanSquaredError()
  >>> mse(y_true, y_pred).numpy()
  0.5

  >>> # Calling with 'sample_weight'.
  >>> mse(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  0.25

  >>> # Using 'sum' reduction type.
  >>> mse = tf.keras.losses.MeanSquaredError(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> mse(y_true, y_pred).numpy()
  1.0

  >>> # Using 'none' reduction type.
  >>> mse = tf.keras.losses.MeanSquaredError(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> mse(y_true, y_pred).numpy()
  array([0.5, 0.5], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_squared_error'):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (default automatic) and name (default mean squared error
    # name). Delegates to the wrapper constructor with the mean squared error function. Returns
    # nothing; an invalid reduction raises a value error.
    """Initializes `MeanSquaredError` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'mean_squared_error'.
    """
    super().__init__(mean_squared_error, name=name, reduction=reduction)


class MeanAbsoluteError(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for mean absolute error: wraps the mean absolute error function with reduction
  # and sample weight support. Holds no extra state.
  """Computes the mean of absolute difference between labels and predictions.

  `loss = abs(y_true - y_pred)`

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> mae = tf.keras.losses.MeanAbsoluteError()
  >>> mae(y_true, y_pred).numpy()
  0.5

  >>> # Calling with 'sample_weight'.
  >>> mae(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  0.25

  >>> # Using 'sum' reduction type.
  >>> mae = tf.keras.losses.MeanAbsoluteError(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> mae(y_true, y_pred).numpy()
  1.0

  >>> # Using 'none' reduction type.
  >>> mae = tf.keras.losses.MeanAbsoluteError(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> mae(y_true, y_pred).numpy()
  array([0.5, 0.5], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.MeanAbsoluteError())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_absolute_error'):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (default automatic) and name (default mean absolute error
    # name). Delegates to the wrapper constructor with the mean absolute error function. Returns
    # nothing; an invalid reduction raises a value error.
    """Initializes `MeanAbsoluteError` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'mean_absolute_error'.
    """
    super().__init__(mean_absolute_error, name=name, reduction=reduction)


class MeanAbsolutePercentageError(LossFunctionWrapper):
  # [CLASS DOC]
  # KNOWN ISSUE: the formula in the docstring omits the mean over the last axis that the wrapped
  # function actually applies; the denominator is also clipped to a small epsilon, not used raw.
  # Loss object for mean absolute percentage error: wraps the matching function with reduction and
  # sample weight support. Holds no extra state.
  """Computes the mean absolute percentage error between `y_true` and `y_pred`.

  `loss = 100 * abs(y_true - y_pred) / y_true`

  Standalone usage:

  >>> y_true = [[2., 1.], [2., 3.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> mape = tf.keras.losses.MeanAbsolutePercentageError()
  >>> mape(y_true, y_pred).numpy()
  50.

  >>> # Calling with 'sample_weight'.
  >>> mape(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  20.

  >>> # Using 'sum' reduction type.
  >>> mape = tf.keras.losses.MeanAbsolutePercentageError(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> mape(y_true, y_pred).numpy()
  100.

  >>> # Using 'none' reduction type.
  >>> mape = tf.keras.losses.MeanAbsolutePercentageError(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> mape(y_true, y_pred).numpy()
  array([25., 75.], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.MeanAbsolutePercentageError())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_absolute_percentage_error'):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (default automatic) and name (default mean absolute
    # percentage error name). Delegates to the wrapper constructor with the percentage error
    # function. Returns nothing; an invalid reduction raises a value error.
    """Initializes `MeanAbsolutePercentageError` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to
        'mean_absolute_percentage_error'.
    """
    super().__init__(
        mean_absolute_percentage_error, name=name, reduction=reduction)


class MeanSquaredLogarithmicError(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for mean squared logarithmic error: wraps the matching function with reduction and
  # sample weight support. Holds no extra state.
  """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

  `loss = square(log(y_true + 1.) - log(y_pred + 1.))`

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError()
  >>> msle(y_true, y_pred).numpy()
  0.240

  >>> # Calling with 'sample_weight'.
  >>> msle(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  0.120

  >>> # Using 'sum' reduction type.
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> msle(y_true, y_pred).numpy()
  0.480

  >>> # Using 'none' reduction type.
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> msle(y_true, y_pred).numpy()
  array([0.240, 0.240], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.MeanSquaredLogarithmicError())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_squared_logarithmic_error'):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (default automatic) and name (default mean squared
    # logarithmic error name). Delegates to the wrapper constructor with the logarithmic error
    # function. Returns nothing; an invalid reduction raises a value error.
    """Initializes `MeanSquaredLogarithmicError` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to
        'mean_squared_logarithmic_error'.
    """
    super().__init__(
        mean_squared_logarithmic_error, name=name, reduction=reduction)


class BinaryCrossentropy(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for binary crossentropy between 0/1 labels and predictions given as probabilities
  # or logits. Forwards from_logits, label smoothing and the averaging axis to the binary
  # crossentropy function and additionally keeps the from_logits flag as a public attribute.
  """Computes the cross-entropy loss between true labels and predicted labels.

  Use this cross-entropy loss for binary (0 or 1) classification applications.
  The loss function requires the following inputs:

  - `y_true` (true label): This is either 0 or 1.
  - `y_pred` (predicted value): This is the model's prediction, i.e, a single
    floating-point value which either represents a
    [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
    when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
    `from_logits=False`).

  **Recommended Usage:** (set `from_logits=True`)

  With `tf.keras` API:

  ```python
  model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    ....
  )
  ```

  As a standalone function:

  >>> # Example 1: (batch_size = 1, number of samples = 4)
  >>> y_true = [0, 1, 0, 0]
  >>> y_pred = [-18.6, 0.51, 2.94, -12.8]
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  >>> bce(y_true, y_pred).numpy()
  0.865

  >>> # Example 2: (batch_size = 2, number of samples = 4)
  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
  >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  >>> bce(y_true, y_pred).numpy()
  0.865
  >>> # Using 'sample_weight' attribute
  >>> bce(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.243
  >>> # Using 'sum' reduction` type.
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> bce(y_true, y_pred).numpy()
  1.730
  >>> # Using 'none' reduction type.
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> bce(y_true, y_pred).numpy()
  array([0.235, 1.496], dtype=float32)

  **Default Usage:** (set `from_logits=False`)

  >>> # Make the following updates to the above "Recommended Usage" section
  >>> # 1. Set `from_logits=False`
  >>> tf.keras.losses.BinaryCrossentropy() # OR ...('from_logits=False')
  >>> # 2. Update `y_pred` to use probabilities instead of logits
  >>> y_pred = [0.6, 0.3, 0.2, 0.8] # OR [[0.6, 0.3], [0.2, 0.8]]
  """

  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='binary_crossentropy'):
    # [FUNCTION DOC]
    # Constructor. Inputs: from_logits (boolean, default false: predictions are probabilities),
    # label_smoothing (number from 0 to 1, default 0), axis (axis averaged over, default last),
    # reduction (default automatic) and name. Passes these to the wrapper constructor together with
    # the binary crossentropy function and also stores from_logits on the instance. Returns
    # nothing.
    """Initializes `BinaryCrossentropy` instance.

    Args:
      from_logits: Whether to interpret `y_pred` as a tensor of
        [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
          assume that `y_pred` contains probabilities (i.e., values in [0, 1]).
      label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0,
        we compute the loss between the predicted labels and a smoothed version
        of the true labels, where the smoothing squeezes the labels towards 0.5.
        Larger values of `label_smoothing` correspond to heavier smoothing.
      axis: The axis along which to compute crossentropy (the features axis).
        Defaults to -1.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Name for the op. Defaults to 'binary_crossentropy'.
    """
    super().__init__(
        binary_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        axis=axis)
    self.from_logits = from_logits


class CategoricalCrossentropy(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for crossentropy between one-hot labels and predicted class distributions (or
  # logits). Forwards from_logits, label smoothing and axis to the categorical crossentropy
  # function. Holds no extra state.
  """Computes the crossentropy loss between the labels and predictions.

  Use this crossentropy loss function when there are two or more label classes.
  We expect labels to be provided in a `one_hot` representation. If you want to
  provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
  There should be `# classes` floating point values per feature.

  In the snippet below, there is `# classes` floating pointing values per
  example. The shape of both `y_pred` and `y_true` are
  `[batch_size, num_classes]`.

  Standalone usage:

  >>> y_true = [[0, 1, 0], [0, 0, 1]]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> cce = tf.keras.losses.CategoricalCrossentropy()
  >>> cce(y_true, y_pred).numpy()
  1.177

  >>> # Calling with 'sample_weight'.
  >>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
  0.814

  >>> # Using 'sum' reduction type.
  >>> cce = tf.keras.losses.CategoricalCrossentropy(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> cce(y_true, y_pred).numpy()
  2.354

  >>> # Using 'none' reduction type.
  >>> cce = tf.keras.losses.CategoricalCrossentropy(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> cce(y_true, y_pred).numpy()
  array([0.0513, 2.303], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
  ```
  """

  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='categorical_crossentropy'):
    # [FUNCTION DOC]
    # Constructor. Inputs: from_logits (boolean, default false), label_smoothing (number from 0 to
    # 1, default 0), axis (class axis, default last), reduction (default automatic) and name.
    # Delegates to the wrapper constructor with the categorical crossentropy function. Returns
    # nothing.
    """Initializes `CategoricalCrossentropy` instance.

    Args:
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
      label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
        meaning the confidence on label values are relaxed. For example, if
        `0.1`, use `0.1 / num_classes` for non-target labels and
        `0.9 + 0.1 / num_classes` for target labels.
      axis: The axis along which to compute crossentropy (the features axis).
        Defaults to -1.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance.
        Defaults to 'categorical_crossentropy'.
    """
    super().__init__(
        categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        axis=axis)


class SparseCategoricalCrossentropy(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for crossentropy between integer class labels and predicted class distributions
  # (or logits). Forwards the from_logits flag to the sparse categorical crossentropy function.
  # Holds no extra state.
  """Computes the crossentropy loss between the labels and predictions.

  Use this crossentropy loss function when there are two or more label classes.
  We expect labels to be provided as integers. If you want to provide labels
  using `one-hot` representation, please use `CategoricalCrossentropy` loss.
  There should be `# classes` floating point values per feature for `y_pred`
  and a single floating point value per feature for `y_true`.

  In the snippet below, there is a single floating point value per example for
  `y_true` and `# classes` floating pointing values per example for `y_pred`.
  The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
  `[batch_size, num_classes]`.

  Standalone usage:

  >>> y_true = [1, 2]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> scce = tf.keras.losses.SparseCategoricalCrossentropy()
  >>> scce(y_true, y_pred).numpy()
  1.177

  >>> # Calling with 'sample_weight'.
  >>> scce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
  0.814

  >>> # Using 'sum' reduction type.
  >>> scce = tf.keras.losses.SparseCategoricalCrossentropy(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> scce(y_true, y_pred).numpy()
  2.354

  >>> # Using 'none' reduction type.
  >>> scce = tf.keras.losses.SparseCategoricalCrossentropy(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> scce(y_true, y_pred).numpy()
  array([0.0513, 2.303], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.SparseCategoricalCrossentropy())
  ```
  """

  def __init__(self,
               from_logits=False,
               reduction=losses_utils.ReductionV2.AUTO,
               name='sparse_categorical_crossentropy'):
    # [FUNCTION DOC]
    # Constructor. Inputs: from_logits (boolean, default false), reduction (default automatic) and
    # name. Delegates to the wrapper constructor with the sparse categorical crossentropy
    # function. Note there is no axis or label smoothing option here. Returns nothing.
    """Initializes `SparseCategoricalCrossentropy` instance.

    Args:
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to
        'sparse_categorical_crossentropy'.
    """
    super().__init__(
        sparse_categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits)


class Hinge(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for the hinge loss; labels given as 0/1 are converted to -1/1 by the wrapped
  # function. Holds no extra state.
  """Computes the hinge loss between `y_true` and `y_pred`.

  `loss = maximum(1 - y_true * y_pred, 0)`

  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
  provided we will convert them to -1 or 1.

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> h = tf.keras.losses.Hinge()
  >>> h(y_true, y_pred).numpy()
  1.3

  >>> # Calling with 'sample_weight'.
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
  0.55

  >>> # Using 'sum' reduction type.
  >>> h = tf.keras.losses.Hinge(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> h(y_true, y_pred).numpy()
  2.6

  >>> # Using 'none' reduction type.
  >>> h = tf.keras.losses.Hinge(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> h(y_true, y_pred).numpy()
  array([1.1, 1.5], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.Hinge())
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='hinge'):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (default automatic) and name (default hinge). Delegates to
    # the wrapper constructor with the hinge function. Returns nothing.
    """Initializes `Hinge` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'hinge'.
    """
    super().__init__(hinge, name=name, reduction=reduction)


class SquaredHinge(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for the squared hinge loss; 0/1 labels are converted to -1/1 by the wrapped
  # function. Holds no extra state.
  """Computes the squared hinge loss between `y_true` and `y_pred`.

  `loss = square(maximum(1 - y_true * y_pred, 0))`

  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
  provided we will convert them to -1 or 1.

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> h = tf.keras.losses.SquaredHinge()
  >>> h(y_true, y_pred).numpy()
  1.86

  >>> # Calling with 'sample_weight'.
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
  0.73

  >>> # Using 'sum' reduction type.
  >>> h = tf.keras.losses.SquaredHinge(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> h(y_true, y_pred).numpy()
  3.72

  >>> # Using 'none' reduction type.
  >>> h = tf.keras.losses.SquaredHinge(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> h(y_true, y_pred).numpy()
  array([1.46, 2.26], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.SquaredHinge())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='squared_hinge'):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (default automatic) and name (default squared hinge).
    # Delegates to the wrapper constructor with the squared hinge function. Returns nothing.
    """Initializes `SquaredHinge` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'squared_hinge'.
    """
    super().__init__(squared_hinge, name=name, reduction=reduction)


class CategoricalHinge(LossFunctionWrapper):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring formula says the negative term is a maximum without naming its
  # reduction; the wrapped function takes the maximum over the last axis of the wrong-class
  # scores and the sum over the last axis of the true-class score.
  # Loss object for categorical hinge loss on one-hot labels. Holds no extra state.
  """Computes the categorical hinge loss between `y_true` and `y_pred`.

  `loss = maximum(neg - pos + 1, 0)`
  where `neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)`

  Standalone usage:

  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> h = tf.keras.losses.CategoricalHinge()
  >>> h(y_true, y_pred).numpy()
  1.4

  >>> # Calling with 'sample_weight'.
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
  0.6

  >>> # Using 'sum' reduction type.
  >>> h = tf.keras.losses.CategoricalHinge(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> h(y_true, y_pred).numpy()
  2.8

  >>> # Using 'none' reduction type.
  >>> h = tf.keras.losses.CategoricalHinge(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> h(y_true, y_pred).numpy()
  array([1.2, 1.6], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalHinge())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='categorical_hinge'):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (default automatic) and name (default categorical hinge).
    # Delegates to the wrapper constructor with the categorical hinge function. Returns nothing.
    """Initializes `CategoricalHinge` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'categorical_hinge'.
    """
    super().__init__(categorical_hinge, name=name, reduction=reduction)


class Poisson(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for the Poisson loss between true counts and predicted rates. Holds no extra
  # state.
  """Computes the Poisson loss between `y_true` and `y_pred`.

  `loss = y_pred - y_true * log(y_pred)`

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [0., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> p = tf.keras.losses.Poisson()
  >>> p(y_true, y_pred).numpy()
  0.5

  >>> # Calling with 'sample_weight'.
  >>> p(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.4

  >>> # Using 'sum' reduction type.
  >>> p = tf.keras.losses.Poisson(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> p(y_true, y_pred).numpy()
  0.999

  >>> # Using 'none' reduction type.
  >>> p = tf.keras.losses.Poisson(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> p(y_true, y_pred).numpy()
  array([0.999, 0.], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.Poisson())
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='poisson'):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (default automatic) and name (default poisson). Delegates to
    # the wrapper constructor with the Poisson function. Returns nothing.
    """Initializes `Poisson` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'poisson'.
    """
    super().__init__(poisson, name=name, reduction=reduction)


class LogCosh(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for the logarithm of the hyperbolic cosine of the prediction error; behaves like
  # squared error for small errors and like absolute error for large ones. Holds no extra state.
  """Computes the logarithm of the hyperbolic cosine of the prediction error.

  `logcosh = log((exp(x) + exp(-x))/2)`,
  where x is the error `y_pred - y_true`.

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [0., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> l = tf.keras.losses.LogCosh()
  >>> l(y_true, y_pred).numpy()
  0.108

  >>> # Calling with 'sample_weight'.
  >>> l(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.087

  >>> # Using 'sum' reduction type.
  >>> l = tf.keras.losses.LogCosh(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> l(y_true, y_pred).numpy()
  0.217

  >>> # Using 'none' reduction type.
  >>> l = tf.keras.losses.LogCosh(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> l(y_true, y_pred).numpy()
  array([0.217, 0.], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.LogCosh())
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='log_cosh'):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (default automatic) and name (default log cosh). Delegates to
    # the wrapper constructor with the log-cosh function. Returns nothing.
    """Initializes `LogCosh` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'log_cosh'.
    """
    super().__init__(log_cosh, name=name, reduction=reduction)


class KLDivergence(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for Kullback-Leibler divergence between true and predicted distributions. Holds
  # no extra state.
  """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

  `loss = y_true * log(y_true / y_pred)`

  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

  Standalone usage:

  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> kl = tf.keras.losses.KLDivergence()
  >>> kl(y_true, y_pred).numpy()
  0.458

  >>> # Calling with 'sample_weight'.
  >>> kl(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.366

  >>> # Using 'sum' reduction type.
  >>> kl = tf.keras.losses.KLDivergence(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> kl(y_true, y_pred).numpy()
  0.916

  >>> # Using 'none' reduction type.
  >>> kl = tf.keras.losses.KLDivergence(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> kl(y_true, y_pred).numpy()
  array([0.916, -3.08e-06], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.KLDivergence())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='kl_divergence'):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (default automatic) and name (default kl divergence).
    # Delegates to the wrapper constructor with the KL divergence function. Returns nothing.
    """Initializes `KLDivergence` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'kl_divergence'.
    """
    super().__init__(kl_divergence, name=name, reduction=reduction)


class Huber(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for the Huber loss, quadratic for small errors and linear for large ones, with
  # the switch point set by a delta parameter that is forwarded to the huber function.
  """Computes the Huber loss between `y_true` and `y_pred`.

  For each value x in `error = y_true - y_pred`:

  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Standalone usage:

  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> h = tf.keras.losses.Huber()
  >>> h(y_true, y_pred).numpy()
  0.155

  >>> # Calling with 'sample_weight'.
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
  0.09

  >>> # Using 'sum' reduction type.
  >>> h = tf.keras.losses.Huber(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> h(y_true, y_pred).numpy()
  0.31

  >>> # Using 'none' reduction type.
  >>> h = tf.keras.losses.Huber(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> h(y_true, y_pred).numpy()
  array([0.18, 0.13], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.Huber())
  ```
  """

  def __init__(self,
               delta=1.0,
               reduction=losses_utils.ReductionV2.AUTO,
               name='huber_loss'):
    # [FUNCTION DOC]
    # Constructor. Inputs: delta (number, default 1.0, where the loss turns from quadratic to
    # linear), reduction (default automatic) and name (default huber loss). Delegates to the
    # wrapper constructor with the huber function and delta. Returns nothing.
    """Initializes `Huber` instance.

    Args:
      delta: A float, the point where the Huber loss function changes from a
        quadratic to linear.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'huber_loss'.
    """
    super().__init__(huber, name=name, reduction=reduction, delta=delta)


@dispatch.add_dispatch_support
def mean_squared_error(y_true, y_pred):
  # [FUNCTION DOC]
  # Per-sample mean squared error. Inputs: y_true (ground truth, any numeric type) and y_pred
  # (predictions, converted to a tensor). The truth is cast to the prediction dtype, squared
  # differences are computed and averaged over the last axis. Returns a tensor with the last
  # dimension removed. Shape mismatches surface as errors from the underlying operations.
  """Computes the mean squared error between labels and predictions.

  After computing the squared distance between the inputs, the mean value over
  the last dimension is returned.

  `loss = mean(square(y_true - y_pred), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return backend.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)


def _ragged_tensor_apply_loss(loss_fn, y_true, y_pred, y_pred_extra_dim=False):
  # [FUNCTION DOC]
  # HELPER: shared engine behind the RaggedTensor versions of the loss functions.
  # Applies a dense loss function to ragged inputs one batch element at a time. Inputs: loss_fn
  # (the dense loss function), y_true and y_pred (ragged tensors; if y_true is not ragged the
  # prediction is simply densified and the loss applied directly), and y_pred_extra_dim (true when
  # the prediction has an extra trailing class dimension compared with the truth, so the last
  # split level of the prediction is ignored when checking that both have matching row splits).
  # Verifies that the nested row splits of both inputs match (assertion dependency), then maps
  # over the batch, converting rows to dense form when all their row lengths are equal. Returns a
  # dense tensor when the per-element result is a scalar, otherwise a ragged tensor. Raises
  # an invalid-argument error at run time when row splits differ.
  """Apply a loss function on a per batch basis.

  Args:
    loss_fn: The loss function
    y_true: truth values (RaggedTensor)
    y_pred: predicted values (RaggedTensor)
    y_pred_extra_dim: whether y_pred has an additional dimension compared to
      y_true

  Returns:
    Loss-function result. A dense tensor if the output has a single dimension
    (per-batch loss value); a ragged tensor otherwise.
  """

  def rt_is_equiv_dense(rt):
    # [FUNCTION DOC]
    # HELPER: nested check inside the ragged loss engine.
    # Input: rt, a ragged tensor. Returns a boolean scalar tensor that is true when, at every
    # nested ragged level, the variance of the row lengths is zero (all rows equally long), meaning
    # the tensor can be converted to a dense one without losing information.
    """Returns true if this RaggedTensor has the same row_lengths across

       all ragged dimensions and thus can be converted to a dense tensor
       without loss of information.

    Args:
      rt: RaggedTensor.
    """
    return math_ops.reduce_all([
        math_ops.equal(
            math_ops.reduce_variance(math_ops.cast(row_lens, backend.floatx())),
            constant_op.constant([0.])) for row_lens in rt.nested_row_lengths()
    ])

  def _convert_to_dense(inputs):
    # [FUNCTION DOC]
    # HELPER: nested utility of the ragged loss engine. Input: a sequence of tensors. Returns a
    # tuple in which every ragged tensor was converted to a dense tensor and other values are
    # kept unchanged.
    return tuple(
        rt.to_tensor() if isinstance(rt, ragged_tensor.RaggedTensor) else rt
        for rt in inputs)

  def _call_loss(inputs, ragged_output):
    # [FUNCTION DOC]
    # HELPER: nested utility of the ragged loss engine. Inputs: inputs (sequence of the true and
    # predicted values for one batch element) and ragged_output (boolean, whether the result must
    # be ragged). Calls the loss function and then converts the result to a ragged tensor or to a
    # dense tensor as requested, so every branch of the map returns the same type. Returns the
    # converted loss value.
    """ Adapt the result to ragged or dense tensor according to the expected

        output type. This is done so that all the return values of the map
        operation have the same type.
    """
    r = loss_fn(*inputs)
    if ragged_output and not isinstance(r, ragged_tensor.RaggedTensor):
      r = ragged_tensor.RaggedTensor.from_tensor(r)
    elif not ragged_output and isinstance(r, ragged_tensor.RaggedTensor):
      r = r.to_tensor()
    return r

  def _wrapper(inputs, ragged_output):
    # [FUNCTION DOC]
    # HELPER: per-element function passed to the batch map of the ragged loss engine.
    # Inputs: inputs (pair of true and predicted values for one batch element) and ragged_output
    # (boolean, whether a ragged result is expected). If the prediction is ragged it picks, with a
    # runtime conditional, between a dense-converted evaluation (when all rows are equally long)
    # and a direct evaluation, both normalised to the expected output type. Otherwise it calls
    # the loss function directly. Returns the loss for that element.
    _, y_pred = inputs
    if isinstance(y_pred, ragged_tensor.RaggedTensor):
      return cond.cond(
          rt_is_equiv_dense(y_pred),
          lambda: _call_loss(_convert_to_dense(inputs), ragged_output),
          lambda: _call_loss(inputs, ragged_output))

    return loss_fn(*inputs)

  if not isinstance(y_true, ragged_tensor.RaggedTensor):
    return loss_fn(y_true, y_pred.to_tensor())

  lshape = y_pred.shape.as_list()[1:-1]
  if len(lshape) > 0:
    spec = ragged_tensor.RaggedTensorSpec(shape=lshape, dtype=y_pred.dtype)
  else:
    spec = tensor_spec.TensorSpec(shape=[], dtype=y_pred.dtype)

  nested_splits_list = [rt.nested_row_splits for rt in (y_true, y_pred)]
  if y_pred_extra_dim:
    # The last dimension of a categorical prediction may be ragged or not.
    rdims = [len(slist) for slist in nested_splits_list]
    if rdims[0] == rdims[1] - 1:
      nested_splits_list[1] = nested_splits_list[1][:-1]

  map_fn = functools.partial(_wrapper, ragged_output=len(lshape) > 1)

  assertion_list = ragged_util.assert_splits_match(nested_splits_list)
  with ops.control_dependencies(assertion_list):
    return ragged_map_ops.map_fn(map_fn, elems=(y_true, y_pred), dtype=spec)


@dispatch.dispatch_for_types(mean_squared_error, ragged_tensor.RaggedTensor)
def _ragged_tensor_mse(y_true, y_pred):
  # [FUNCTION DOC]
  # HELPER: dispatch handler that makes the mean squared error function accept ragged tensors
  # (registered through the dispatch decorator). Inputs: ragged truth and prediction. Delegates to
  # the ragged loss engine. Returns a dense per-batch tensor, or a ragged tensor when the feature
  # shape has more than one dimension.
  """Implements support for handling RaggedTensors.

  Args:
    y_true: RaggedTensor truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: RaggedTensor predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    When the number of dimensions of the batch feature vector [d0, .. dN] is
    greater than one the return value is a RaggedTensor. Otherwise a Dense
    tensor with dimensions [batch_size] is returned.
  """
  return _ragged_tensor_apply_loss(mean_squared_error, y_true, y_pred)


@dispatch.add_dispatch_support
def mean_absolute_error(y_true, y_pred):
  # [FUNCTION DOC]
  # Per-sample mean absolute error. Inputs: y_true and y_pred. The prediction is converted to a
  # tensor, the truth cast to its dtype, and absolute differences are averaged over the last
  # axis. Returns a tensor with the last dimension removed.
  """Computes the mean absolute error between labels and predictions.

  `loss = mean(abs(y_true - y_pred), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(), np.mean(np.abs(y_true - y_pred), axis=-1))

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return backend.mean(math_ops.abs(y_pred - y_true), axis=-1)


@dispatch.dispatch_for_types(mean_absolute_error, ragged_tensor.RaggedTensor)
def _ragged_tensor_mae(y_true, y_pred):
  # [FUNCTION DOC]
  # HELPER: dispatch handler that lets the mean absolute error function accept ragged tensors by
  # delegating to the ragged loss engine. Inputs: ragged truth and prediction. Returns the
  # per-batch (dense) or ragged loss.
  """RaggedTensor adapter for mean_absolute_error."""
  return _ragged_tensor_apply_loss(mean_absolute_error, y_true, y_pred)


@dispatch.add_dispatch_support
def mean_absolute_percentage_error(y_true, y_pred):
  # [FUNCTION DOC]
  # Per-sample mean absolute percentage error. Inputs: y_true and y_pred. The truth is cast to
  # the prediction dtype; the absolute difference is divided by the absolute truth clipped from
  # below to a small epsilon (so zero targets do not cause division by zero), averaged over the
  # last axis and multiplied by one hundred. Returns a tensor with the last dimension removed.
  """Computes the mean absolute percentage error between `y_true` and `y_pred`.

  `loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.random(size=(2, 3))
  >>> y_true = np.maximum(y_true, 1e-7)  # Prevent division by zero
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(),
  ...     100. * np.mean(np.abs((y_true - y_pred) / y_true), axis=-1))

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  diff = math_ops.abs(
      (y_true - y_pred) / backend.maximum(math_ops.abs(y_true),
                                          backend.epsilon()))
  return 100. * backend.mean(diff, axis=-1)


@dispatch.dispatch_for_types(mean_absolute_percentage_error,
                             ragged_tensor.RaggedTensor)
def _ragged_tensor_mape(y_true, y_pred):
  # [FUNCTION DOC]
  # HELPER: dispatch handler that lets the mean absolute percentage error function accept ragged
  # tensors by delegating to the ragged loss engine. Inputs: ragged truth and prediction.
  # Returns the per-batch or ragged loss.
  """Support RaggedTensors."""
  return _ragged_tensor_apply_loss(mean_absolute_percentage_error, y_true,
                                   y_pred)


@dispatch.add_dispatch_support
def mean_squared_logarithmic_error(y_true, y_pred):
  # [FUNCTION DOC]
  # KNOWN ISSUE: both truth and prediction are clipped from below to a small epsilon before the
  # logarithm, which the docstring does not state; negative inputs are therefore treated as
  # near zero instead of producing undefined values.
  # Per-sample mean squared logarithmic error. Inputs: y_true and y_pred. Truth is cast to the
  # prediction dtype, each side is clipped, shifted by one and passed through the natural
  # logarithm, then squared differences are averaged over the last axis. Returns a tensor with the
  # last dimension removed.
  """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

  `loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> y_true = np.maximum(y_true, 1e-7)
  >>> y_pred = np.maximum(y_pred, 1e-7)
  >>> assert np.allclose(
  ...     loss.numpy(),
  ...     np.mean(
  ...         np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean squared logarithmic error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  first_log = math_ops.log(backend.maximum(y_pred, backend.epsilon()) + 1.)
  second_log = math_ops.log(backend.maximum(y_true, backend.epsilon()) + 1.)
  return backend.mean(
      math_ops.squared_difference(first_log, second_log), axis=-1)


@dispatch.dispatch_for_types(mean_squared_logarithmic_error,
                             ragged_tensor.RaggedTensor)
def _ragged_tensor_msle(y_true, y_pred):
  # [FUNCTION DOC]
  # HELPER: dispatch handler that lets the mean squared logarithmic error function accept ragged
  # tensors by delegating to the ragged loss engine. Inputs: ragged truth and prediction.
  # Returns the per-batch or ragged loss.
  """Implements support for handling RaggedTensors."""
  return _ragged_tensor_apply_loss(mean_squared_logarithmic_error, y_true,
                                   y_pred)


def _maybe_convert_labels(y_true):
  # [FUNCTION DOC]
  # HELPER: label preprocessing shared by the hinge and squared hinge functions.
  # Input: y_true, a numeric tensor. If every element is exactly zero or one, the labels are
  # mapped to minus one and one; otherwise they are returned unchanged. The check is done with a
  # runtime-aware conditional so it works in eager and graph mode. Returns the possibly converted
  # tensor.
  """Converts binary labels into -1/1."""
  are_zeros = math_ops.equal(y_true, 0)
  are_ones = math_ops.equal(y_true, 1)
  is_binary = math_ops.reduce_all(math_ops.logical_or(are_zeros, are_ones))

  def _convert_binary_labels():
    # [FUNCTION DOC]
    # HELPER: nested branch of the label conversion helper. Maps zero/one labels from the
    # enclosing scope to minus one/one and returns the result.
    # Convert the binary labels to -1 or 1.
    return 2. * y_true - 1.

  updated_y_true = smart_cond.smart_cond(is_binary, _convert_binary_labels,
                                         lambda: y_true)
  return updated_y_true


@dispatch.add_dispatch_support
def squared_hinge(y_true, y_pred):
  # [FUNCTION DOC]
  # Per-sample squared hinge loss. Inputs: y_true (labels of minus one/one, or zero/one which are
  # converted) and y_pred. Computes the square of the positive part of one minus label times
  # prediction and averages over the last axis. Returns a tensor with the last dimension removed.
  """Computes the squared hinge loss between `y_true` and `y_pred`.

  `loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.choice([-1, 1], size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.squared_hinge(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(),
  ...     np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1))

  Args:
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
      If binary (0 or 1) labels are provided we will convert them to -1 or 1.
      shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
     Squared hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = _maybe_convert_labels(y_true)
  return backend.mean(
      math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)


@dispatch.add_dispatch_support
def hinge(y_true, y_pred):
  # [FUNCTION DOC]
  # Per-sample hinge loss. Inputs: y_true (labels of minus one/one, or zero/one which are
  # converted) and y_pred. Computes the positive part of one minus label times prediction and
  # averages over the last axis. Returns a tensor with the last dimension removed.
  """Computes the hinge loss between `y_true` and `y_pred`.

  `loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.choice([-1, 1], size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.hinge(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(),
  ...     np.mean(np.maximum(1. - y_true * y_pred, 0.), axis=-1))

  Args:
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
      If binary (0 or 1) labels are provided they will be converted to -1 or 1.
      shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = _maybe_convert_labels(y_true)
  return backend.mean(math_ops.maximum(1. - y_true * y_pred, 0.), axis=-1)


@dispatch.add_dispatch_support
def categorical_hinge(y_true, y_pred):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says labels may be minus one/one or zero/one, but the code never
  # converts labels; the formula only works for one-hot (zero/one) labels. The docstring also
  # calls the negative term a maximum without saying it is taken over the last axis.
  # Per-sample categorical hinge loss. Inputs: y_true (one-hot labels) and y_pred (scores). The
  # truth is cast to the prediction dtype; the positive term is the sum over the last axis of
  # truth times prediction, the negative term the maximum over the last axis of the prediction
  # restricted to wrong classes. Returns the positive part of negative minus positive plus one,
  # with the last dimension removed.
  """Computes the categorical hinge loss between `y_true` and `y_pred`.

  `loss = maximum(neg - pos + 1, 0)`
  where `neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)`

  Standalone usage:

  >>> y_true = np.random.randint(0, 3, size=(2,))
  >>> y_true = tf.keras.utils.to_categorical(y_true, num_classes=3)
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.categorical_hinge(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> pos = np.sum(y_true * y_pred, axis=-1)
  >>> neg = np.amax((1. - y_true) * y_pred, axis=-1)
  >>> assert np.array_equal(loss.numpy(), np.maximum(0., neg - pos + 1.))

  Args:
    y_true: The ground truth values. `y_true` values are expected to be
    either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor).
    y_pred: The predicted values.

  Returns:
    Categorical hinge loss values.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  pos = math_ops.reduce_sum(y_true * y_pred, axis=-1)
  neg = math_ops.reduce_max((1. - y_true) * y_pred, axis=-1)
  zero = math_ops.cast(0., y_pred.dtype)
  return math_ops.maximum(neg - pos + 1., zero)


@dispatch.add_dispatch_support
def huber(y_true, y_pred, delta=1.0):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns one scalar entry per sample, but the result is the
  # mean over the last axis only, so higher-rank inputs give a tensor with one fewer dimension.
  # Huber loss. Inputs: y_true, y_pred (both cast to the default float type of the backend) and
  # delta (number, default 1.0, the threshold between the quadratic and the linear region). The
  # error magnitude is compared with delta: quadratic below or equal, linear above. The elementwise
  # results are averaged over the last axis. Returns the loss tensor.
  """Computes Huber loss value.

  For each value x in `error = y_true - y_pred`:

  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = d * |x| - 0.5 * d^2        if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    delta: A float, the point where the Huber loss function changes from a
      quadratic to linear.

  Returns:
    Tensor with one scalar loss entry per sample.
  """
  y_pred = math_ops.cast(y_pred, dtype=backend.floatx())
  y_true = math_ops.cast(y_true, dtype=backend.floatx())
  delta = math_ops.cast(delta, dtype=backend.floatx())
  error = math_ops.subtract(y_pred, y_true)
  abs_error = math_ops.abs(error)
  half = tensor_conversion.convert_to_tensor_v2_with_dispatch(
      0.5, dtype=abs_error.dtype
  )
  return backend.mean(
      array_ops.where_v2(abs_error <= delta, half * math_ops.square(error),
                         delta * abs_error - half * math_ops.square(delta)),
      axis=-1)


@dispatch.add_dispatch_support
def log_cosh(y_true, y_pred):
  # [FUNCTION DOC]
  # Per-sample logarithm of the hyperbolic cosine of the error. Inputs: y_true and y_pred. The
  # error is computed in a numerically stable form (using a softplus) and averaged over the last
  # axis. Returns a tensor with the last dimension removed.
  """Logarithm of the hyperbolic cosine of the prediction error.

  `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
  to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
  like the mean squared error, but will not be so strongly affected by the
  occasional wildly incorrect prediction.

  Standalone usage:

  >>> y_true = np.random.random(size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.logcosh(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> x = y_pred - y_true
  >>> assert np.allclose(
  ...     loss.numpy(),
  ...     np.mean(x + np.log(np.exp(-2. * x) + 1.) - math_ops.log(2.), axis=-1),
  ...     atol=1e-5)

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Logcosh error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)

  def _logcosh(x):
    # [FUNCTION DOC]
    # HELPER: nested elementwise helper of the log-cosh function. Input: x, an error tensor.
    # Returns the stable log-cosh value of each element, with the constant natural log of two
    # cast to the dtype of x.
    return x + math_ops.softplus(-2. * x) - math_ops.cast(
        math_ops.log(2.), x.dtype)

  return backend.mean(_logcosh(y_pred - y_true), axis=-1)


@dispatch.add_dispatch_support
def categorical_crossentropy(y_true,
                             y_pred,
                             from_logits=False,
                             label_smoothing=0,
                             axis=-1):
  # [FUNCTION DOC]
  # Categorical crossentropy between one-hot labels and predictions. Inputs: y_true (one-hot
  # targets), y_pred (probabilities, or logits when from_logits is true), from_logits (boolean,
  # default false), label_smoothing (number from 0 to 1, default 0) and axis (class axis, default
  # last). The truth is cast to the prediction dtype; when smoothing is non-zero (decided with a
  # runtime-aware conditional) labels are blended towards a uniform distribution over the classes.
  # Delegates to the backend crossentropy. Returns the per-sample loss with the class axis
  # removed.
  """Computes the categorical crossentropy loss.

  Standalone usage:

  >>> y_true = [[0, 1, 0], [0, 0, 1]]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> loss.numpy()
  array([0.0513, 2.303], dtype=float32)

  Args:
    y_true: Tensor of one-hot true targets.
    y_pred: Tensor of predicted targets.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
      example, if `0.1`, use `0.1 / num_classes` for non-target labels
      and `0.9 + 0.1 / num_classes` for target labels.
    axis: Defaults to -1. The dimension along which the entropy is
      computed.

  Returns:
    Categorical crossentropy loss value.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  label_smoothing = tensor_conversion.convert_to_tensor_v2_with_dispatch(
      label_smoothing, dtype=backend.floatx()
  )

  def _smooth_labels():
    # [FUNCTION DOC]
    # HELPER: nested branch of categorical crossentropy. Uses the label smoothing value and truth
    # from the enclosing scope and returns labels scaled by one minus the smoothing, plus the
    # smoothing divided by the number of classes (size of the last axis).
    num_classes = math_ops.cast(array_ops.shape(y_true)[-1], y_pred.dtype)
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

  y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels,
                                 lambda: y_true)

  return backend.categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits, axis=axis)


@dispatch.dispatch_for_types(categorical_crossentropy,
                             ragged_tensor.RaggedTensor)
def _ragged_tensor_categorical_crossentropy(y_true,
                                            y_pred,
                                            from_logits=False,
                                            label_smoothing=0,
                                            axis=-1):
  # [FUNCTION DOC]
  # HELPER: dispatch handler that lets the categorical crossentropy function accept ragged
  # tensors. Inputs: ragged truth and prediction, from_logits, label_smoothing and axis, which
  # are bound to the dense function and run through the ragged loss engine. Returns a loss with
  # shape batch by sequence length (dense or ragged).
  """Implements support for handling RaggedTensors.

  Args:
    y_true: Tensor of one-hot true targets.
    y_pred: Tensor of predicted targets.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
      example, if `0.1`, use `0.1 / num_classes` for non-target labels and `0.9
      + 0.1 / num_classes` for target labels.
    axis: The axis along which to compute crossentropy (the features axis).
      Defaults to -1.

  Returns:
    Categorical crossentropy loss value.

  Expected shape: (batch, sequence_len, n_classes) with sequence_len
  being variable per batch.
  Return shape: (batch, sequence_len).

  When used by CategoricalCrossentropy() with the default reduction
  (SUM_OVER_BATCH_SIZE), the reduction averages the loss over the
  number of elements independent of the batch. E.g. if the RaggedTensor
  has 2 batches with [2, 1] values respectively the resulting loss is
  the sum of the individual loss values divided by 3.
  """
  fn = functools.partial(
      categorical_crossentropy,
      from_logits=from_logits,
      label_smoothing=label_smoothing,
      axis=axis)
  return _ragged_tensor_apply_loss(fn, y_true, y_pred)


@dispatch.add_dispatch_support
def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
  # [FUNCTION DOC]
  # Crossentropy between integer class labels and predictions. Inputs: y_true (integer class
  # indices), y_pred (probabilities or logits), from_logits (boolean, default false) and axis
  # (class axis, default last). The prediction is converted to a tensor and the truth cast to
  # its dtype, then the backend sparse crossentropy is used. Returns per-sample loss values.
  """Computes the sparse categorical crossentropy loss.

  Standalone usage:

  >>> y_true = [1, 2]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> loss.numpy()
  array([0.0513, 2.303], dtype=float32)

  Args:
    y_true: Ground truth values.
    y_pred: The predicted values.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    axis: Defaults to -1. The dimension along which the entropy is
      computed.

  Returns:
    Sparse categorical crossentropy loss value.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return backend.sparse_categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits, axis=axis)


@dispatch.dispatch_for_types(sparse_categorical_crossentropy,
                             ragged_tensor.RaggedTensor)
def _ragged_tensor_sparse_categorical_crossentropy(y_true,
                                                   y_pred,
                                                   from_logits=False,
                                                   axis=-1):
  # [FUNCTION DOC]
  # HELPER: dispatch handler that lets the sparse categorical crossentropy function accept ragged
  # tensors. Inputs: ragged integer labels and ragged predictions with a class dimension,
  # from_logits and axis. Runs the dense function through the ragged loss engine, telling it that
  # the prediction has an extra class dimension. Returns a loss with shape batch by sequence
  # length.
  """ Implements support for handling RaggedTensors.

      Expected y_pred shape: (batch, sequence_len, n_classes) with sequence_len
      being variable per batch.
      Return shape: (batch, sequence_len).

      When used by SparseCategoricalCrossentropy() with the default reduction
      (SUM_OVER_BATCH_SIZE), the reduction averages the loss over the
      number of elements independent of the batch. E.g. if the RaggedTensor
      has 2 batches with [2, 1] values respectively, the resulting loss is
      the sum of the individual loss values divided by 3.
  """
  fn = functools.partial(
      sparse_categorical_crossentropy, from_logits=from_logits, axis=axis)
  return _ragged_tensor_apply_loss(fn, y_true, y_pred, y_pred_extra_dim=True)


@dispatch.add_dispatch_support
def binary_crossentropy(y_true,
                        y_pred,
                        from_logits=False,
                        label_smoothing=0,
                        axis=-1):
  # [FUNCTION DOC]
  # Binary crossentropy. Inputs: y_true (0/1 targets), y_pred (probabilities, or logits when
  # from_logits is true), from_logits (boolean, default false), label_smoothing (number from 0 to
  # 1, default 0, squeezing labels towards one half) and axis (axis over which the mean is taken,
  # default last). Prediction is converted to a tensor, truth cast to its dtype, labels smoothed
  # when the smoothing is non-zero, then the elementwise backend crossentropy is averaged over
  # axis. Returns a tensor with that axis removed.
  """Computes the binary crossentropy loss.

  Standalone usage:

  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> loss.numpy()
  array([0.916 , 0.714], dtype=float32)

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels by
      squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing`
      for the target class and `0.5 * label_smoothing` for the non-target class.
    axis: The axis along which the mean is computed. Defaults to -1.

  Returns:
    Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  label_smoothing = tensor_conversion.convert_to_tensor_v2_with_dispatch(
      label_smoothing, dtype=backend.floatx()
  )

  def _smooth_labels():
    # [FUNCTION DOC]
    # HELPER: nested branch of binary crossentropy. Uses truth and smoothing from the enclosing
    # scope and returns labels scaled by one minus the smoothing plus half the smoothing.
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

  y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels,
                                 lambda: y_true)

  return backend.mean(
      backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits),
      axis=axis)


@dispatch.dispatch_for_types(binary_crossentropy, ragged_tensor.RaggedTensor)
def _ragged_tensor_binary_crossentropy(y_true,
                                       y_pred,
                                       from_logits=False,
                                       label_smoothing=0,
                                       axis=-1):
  # [FUNCTION DOC]
  # HELPER: dispatch handler that lets the binary crossentropy function accept ragged tensors.
  # Inputs: ragged truth and prediction (batch by variable sequence length), from_logits,
  # label_smoothing and axis, bound to the dense function and run through the ragged loss engine.
  # Returns the per-batch mean loss values (shape: batch).
  """Implements support for handling RaggedTensors.

  Args:
    y_true: Tensor of one-hot true targets.
    y_pred: Tensor of predicted targets.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
      example, if `0.1`, use `0.1 / num_classes` for non-target labels
      and `0.9 + 0.1 / num_classes` for target labels.
    axis: Axis along which to compute crossentropy.

  Returns:
    Binary crossentropy loss value.

  Expected shape: (batch, sequence_len) with sequence_len being variable
  per batch.
  Return shape: (batch,); returns the per batch mean of the loss values.

  When used by BinaryCrossentropy() with the default reduction
  (SUM_OVER_BATCH_SIZE), the reduction averages the per batch losses over
  the number of batches.
  """
  fn = functools.partial(
      binary_crossentropy,
      from_logits=from_logits,
      label_smoothing=label_smoothing,
      axis=axis)
  return _ragged_tensor_apply_loss(fn, y_true, y_pred)


@dispatch.add_dispatch_support
def kl_divergence(y_true, y_pred):
  # [FUNCTION DOC]
  # Kullback-Leibler divergence. Inputs: y_true and y_pred (distributions). The prediction is
  # converted to a tensor, the truth cast to its dtype, both are clipped to the range from a small
  # epsilon to one, and the sum over the last axis of truth times log of truth over prediction is
  # returned (a tensor without the last dimension). Casting failures raise a type error.
  """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

  `loss = y_true * log(y_true / y_pred)`

  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float64)
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> y_true = tf.keras.backend.clip(y_true, 1e-7, 1)
  >>> y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1)
  >>> assert np.array_equal(
  ...     loss.numpy(), np.sum(y_true * np.log(y_true / y_pred), axis=-1))

  Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.

  Returns:
    A `Tensor` with loss.

  Raises:
    TypeError: If `y_true` cannot be cast to the `y_pred.dtype`.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = backend.clip(y_true, backend.epsilon(), 1)
  y_pred = backend.clip(y_pred, backend.epsilon(), 1)
  return math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)


@dispatch.add_dispatch_support
def poisson(y_true, y_pred):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring lists an invalid-argument error for incompatible shapes, but this
  # function raises nothing itself; any shape error comes from the underlying tensor operations.
  # Per-sample Poisson loss. Inputs: y_true and y_pred. Computes prediction minus truth times
  # the log of the prediction plus a small epsilon, averaged over the last axis. Returns a tensor
  # with the last dimension removed.
  """Computes the Poisson loss between y_true and y_pred.

  The Poisson loss is the mean of the elements of the `Tensor`
  `y_pred - y_true * log(y_pred)`.

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.poisson(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> y_pred = y_pred + 1e-7
  >>> assert np.allclose(
  ...     loss.numpy(), np.mean(y_pred - y_true * np.log(y_pred), axis=-1),
  ...     atol=1e-5)

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
     Poisson loss value. shape = `[batch_size, d0, .. dN-1]`.

  Raises:
    InvalidArgumentError: If `y_true` and `y_pred` have incompatible shapes.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return backend.mean(
      y_pred - y_true * math_ops.log(y_pred + backend.epsilon()), axis=-1)


@dispatch.add_dispatch_support
def cosine_similarity(y_true, y_pred, axis=-1):
  # [FUNCTION DOC]
  # KNOWN ISSUE: despite the name, it returns the NEGATIVE cosine similarity (a loss to minimise),
  # and unlike its siblings it does not convert or cast its inputs first.
  # Inputs: y_true, y_pred and axis (default last). Both are normalised to unit L2 length along
  # the axis and the negated sum of their elementwise product over that axis is returned
  # (zero vectors give zero). The result has the axis removed.
  """Computes the cosine similarity between labels and predictions.

  Note that it is a number between -1 and 1. When it is a negative number
  between -1 and 0, 0 indicates orthogonality and values closer to -1
  indicate greater similarity. The values closer to 1 indicate greater
  dissimilarity. This makes it usable as a loss function in a setting
  where you try to maximize the proximity between predictions and
  targets. If either `y_true` or `y_pred` is a zero vector, cosine
  similarity will be 0 regardless of the proximity between predictions
  and targets.

  `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`

  Standalone usage:

  >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
  >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
  >>> loss = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=1)
  >>> loss.numpy()
  array([-0., -0.999, 0.999], dtype=float32)

  Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.
    axis: Axis along which to determine similarity.

  Returns:
    Cosine similarity tensor.
  """
  y_true = nn.l2_normalize(y_true, axis=axis)
  y_pred = nn.l2_normalize(y_pred, axis=axis)
  return -math_ops.reduce_sum(y_true * y_pred, axis=axis)


class CosineSimilarity(LossFunctionWrapper):
  # [CLASS DOC]
  # Loss object for negative cosine similarity between truth and predictions along an axis. The
  # axis is forwarded to the cosine similarity function as an extra keyword argument. Holds no
  # other state.
  """Computes the cosine similarity between labels and predictions.

  Note that it is a number between -1 and 1. When it is a negative number
  between -1 and 0, 0 indicates orthogonality and values closer to -1
  indicate greater similarity. The values closer to 1 indicate greater
  dissimilarity. This makes it usable as a loss function in a setting
  where you try to maximize the proximity between predictions and targets.
  If either `y_true` or `y_pred` is a zero vector, cosine similarity will be 0
  regardless of the proximity between predictions and targets.

  `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`

  Standalone usage:

  >>> y_true = [[0., 1.], [1., 1.]]
  >>> y_pred = [[1., 0.], [1., 1.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
  >>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
  >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
  >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
  >>> # loss = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
  >>> #       = -((0. + 0.) +  (0.5 + 0.5)) / 2
  >>> cosine_loss(y_true, y_pred).numpy()
  -0.5

  >>> # Calling with 'sample_weight'.
  >>> cosine_loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  -0.0999

  >>> # Using 'sum' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> cosine_loss(y_true, y_pred).numpy()
  -0.999

  >>> # Using 'none' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> cosine_loss(y_true, y_pred).numpy()
  array([-0., -0.999], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.CosineSimilarity(axis=1))
  ```

  Args:
    axis: The axis along which the cosine similarity is computed
      (the features axis). Defaults to -1.
    reduction: Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`. When used with `tf.distribute.Strategy`, outside of
      built-in training loops such as `tf.keras` `compile` and `fit`, using
      `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error. Please see this
      custom training [tutorial]
      (https://www.tensorflow.org/tutorials/distribute/custom_training) for more
        details.
    name: Optional name for the instance.
  """

  def __init__(self,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='cosine_similarity'):
    # [FUNCTION DOC]
    # Constructor. Inputs: axis (axis along which similarity is computed, default last),
    # reduction (default automatic) and name (default cosine similarity). Delegates to the
    # wrapper constructor with the cosine similarity function and the axis. Returns nothing.
    super().__init__(
        cosine_similarity, reduction=reduction, name=name, axis=axis)


# Aliases.

bce = BCE = binary_crossentropy
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence = kl_divergence
logcosh = log_cosh
huber_loss = huber


def is_categorical_crossentropy(loss):
  # [FUNCTION DOC]
  # HELPER: predicate used by Keras compile/training code to detect categorical crossentropy.
  # Input: loss (a loss object, function or string). Returns true when it is a categorical
  # crossentropy instance, a wrapper around the categorical crossentropy function, has that
  # function name, or equals the string name. Has no side effects.
  result = ((isinstance(loss, CategoricalCrossentropy) or
             (isinstance(loss, LossFunctionWrapper) and
              loss.fn == categorical_crossentropy) or
             (hasattr(loss, '__name__') and
              loss.__name__ == 'categorical_crossentropy') or
             (loss == 'categorical_crossentropy')))
  return result


def serialize(loss):
  # [FUNCTION DOC]
  # Converts a loss object or loss function into its configuration mapping (class or function
  # name plus config) through the shared Keras serialization utility. Input: loss. Returns the
  # mapping.
  """Serializes loss function or `Loss` instance.

  Args:
    loss: A Keras `Loss` instance or a loss function.

  Returns:
    Loss configuration dictionary.
  """
  return serialize_keras_object(loss)


def deserialize(name, custom_objects=None):
  # [FUNCTION DOC]
  # Rebuilds a loss from a name string or configuration mapping. Inputs: name (string or
  # mapping) and custom_objects (optional mapping from names to user classes or functions, used
  # in addition to everything defined in this module). Uses the shared Keras deserialization
  # utility with this module's global names as the lookup table. Returns a loss instance or
  # function; raises a value error for unknown names.
  """Deserializes a serialized loss class/function instance.

  Args:
      name: Loss configuration.
      custom_objects: Optional dictionary mapping names (strings) to custom
        objects (classes and functions) to be considered during deserialization.

  Returns:
      A Keras `Loss` instance or a loss function.
  """
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='loss function')


def get(identifier):
  # [FUNCTION DOC]
  # Resolves a loss identifier. Input: identifier, which may be nothing (returns nothing), a
  # string name or a configuration mapping (deserialized), or a callable (returned unchanged).
  # Any other type raises a value error naming the bad identifier.
  """Retrieves a Keras loss as a `function`/`Loss` class instance.

  The `identifier` may be the string name of a loss function or `Loss` class.

  >>> loss = tf.keras.losses.get("categorical_crossentropy")
  >>> type(loss)
  <class 'function'>
  >>> loss = tf.keras.losses.get("CategoricalCrossentropy")
  >>> type(loss)
  <class '...keras.losses.CategoricalCrossentropy'>

  You can also specify `config` of the loss to this function by passing dict
  containing `class_name` and `config` as an identifier. Also note that the
  `class_name` must map to a `Loss` class

  >>> identifier = {"class_name": "CategoricalCrossentropy",
  ...               "config": {"from_logits": True}}
  >>> loss = tf.keras.losses.get(identifier)
  >>> type(loss)
  <class '...keras.losses.CategoricalCrossentropy'>

  Args:
    identifier: A loss identifier. One of None or string name of a loss
      function/class or loss configuration dictionary or a loss function or a
      loss class instance.

  Returns:
    A Keras loss as a `function`/ `Loss` class instance.

  Raises:
    ValueError: If `identifier` cannot be interpreted.
  """
  if identifier is None:
    return None
  if isinstance(identifier, str):
    identifier = str(identifier)
    return deserialize(identifier)
  if isinstance(identifier, dict):
    return deserialize(identifier)
  if callable(identifier):
    return identifier
  raise ValueError(
      f'Could not interpret loss function identifier: {identifier}')


LABEL_DTYPES_FOR_LOSSES = {
    losses_impl.sparse_softmax_cross_entropy: 'int32',
    sparse_categorical_crossentropy: 'int32'
}
