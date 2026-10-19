# [MODULE DOC]
# Built-in Keras metrics (legacy tensorflow.python.keras implementation). Defines the abstract
# Metric base class (a stateful Layer that accumulates statistics in non-trainable variables and
# reports a result), reduction-based metrics (Sum, Mean, MeanMetricWrapper), accuracy metrics,
# confusion-matrix based metrics (counts, precision, recall, sensitivity/specificity, AUC),
# regression/loss-style metrics wrapped around functions from the losses module, IoU metrics,
# plus the plain functional metric helpers (binary_accuracy, categorical_accuracy, top-k
# accuracy, cosine similarity, etc.) and the serialize / deserialize / get lookup functions.
# Import-time side effects: only imports and definitions plus aliases (for example mse, mae);
# no flags or registrations are created. Public Keras metric names are exported elsewhere.
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
# pylint: disable=g-doc-return-or-yield
"""Built-in metrics."""

import abc
import types
import warnings

import numpy as np

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import categorical_hinge
from tensorflow.python.keras.losses import hinge
from tensorflow.python.keras.losses import kullback_leibler_divergence
from tensorflow.python.keras.losses import logcosh
from tensorflow.python.keras.losses import mean_absolute_error
from tensorflow.python.keras.losses import mean_absolute_percentage_error
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import mean_squared_logarithmic_error
from tensorflow.python.keras.losses import poisson
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.losses import squared_hinge
from tensorflow.python.keras.saving.saved_model import metric_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls


class Metric(base_layer.Layer, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class for all Keras metrics. A metric is a stateful, non-trainable Layer whose
  # state variables (created with add_weight in the constructor of a subclass) are updated by
  # update_state and turned into a value by result. Subclasses must implement update_state and
  # result (both abstract); reset_state clears the variables. Calling an instance updates the
  # state with a batch and returns the current result. Construction wraps update_state and result
  # of each instance (see the allocator) so they work under autograph, graph mode and
  # distribution strategies. Instances can be serialized with get_config.
  """Encapsulates metric logic and state.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    **kwargs: Additional layer keywords arguments.

  Standalone usage:

  ```python
  m = SomeMetric(...)
  for input in ...:
    m.update_state(input)
  print('Final result: ', m.result().numpy())
  ```

  Usage with `compile()` API:

  ```python
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))

  model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

  data = np.random.random((1000, 32))
  labels = np.random.random((1000, 10))

  dataset = tf.data.Dataset.from_tensor_slices((data, labels))
  dataset = dataset.batch(32)

  model.fit(dataset, epochs=10)
  ```

  To be implemented by subclasses:
  * `__init__()`: All state variables should be created in this method by
    calling `self.add_weight()` like: `self.var = self.add_weight(...)`
  * `update_state()`: Has all updates to the state variables like:
    self.var.assign_add(...).
  * `result()`: Computes and returns a value for the metric
    from the state variables.

  Example subclass implementation:

  ```python
  class BinaryTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
      super(BinaryTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = tf.cast(y_true, tf.bool)
      y_pred = tf.cast(y_pred, tf.bool)

      values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
      values = tf.cast(values, self.dtype)
      if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, self.dtype)
        sample_weight = tf.broadcast_to(sample_weight, values.shape)
        values = tf.multiply(values, sample_weight)
      self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
      return self.true_positives
  ```
  """

  def __init__(self, name=None, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: name (optional string name of the metric), dtype (optional data type of
    # the result and of the state variables) and extra layer keyword arguments passed to the
    # Layer constructor. Marks the layer as stateful and already built. When the V2 dtype
    # behavior is disabled, the stored dtype is set to the backend float type if dtype is not
    # given, otherwise to the name of the given dtype. Returns nothing.
    super(Metric, self).__init__(name=name, dtype=dtype, **kwargs)
    self.stateful = True  # All metric layers are stateful.
    self.built = True
    if not base_layer_utils.v2_dtype_behavior_enabled():
      # We only do this when the V2 behavior is not enabled, as when it is
      # enabled, the dtype already defaults to floatx.
      self._dtype = (backend.floatx() if dtype is None
                     else dtypes.as_dtype(dtype).name)

  def __new__(cls, *args, **kwargs):
    # [FUNCTION DOC]
    # Allocator that creates a new metric instance and then wraps its update_state and result
    # methods on that instance. Inputs: the class being instantiated plus positional and keyword
    # arguments, which are ignored here (they are consumed by the constructor afterwards).
    # If running inside eager or a tf.function, or the class is a built-in metric, update_state is
    # wrapped so it is converted by autograph according to the current autograph control status.
    # Otherwise (graph-mode custom metric) update_state is wrapped in a tf.function unless it
    # already is one. The result is then passed through the update-state wrapper utility and
    # bound to the instance; result is likewise autograph-converted and wrapped by the result
    # wrapper. Returns the new instance with instance-level update_state and result overrides.
    obj = super(Metric, cls).__new__(cls)

    # If `update_state` is not in eager/tf.function and it is not from a
    # built-in metric, wrap it in `tf.function`. This is so that users writing
    # custom metrics in v1 need not worry about control dependencies and
    # return ops.
    if (base_layer_utils.is_in_eager_or_tf_function() or
        is_built_in(cls)):
      obj_update_state = obj.update_state

      def update_state_fn(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Nested adapter used by the allocator to wrap a metric's update_state.
        # Reads the current autograph control status, converts the captured update_state method
        # accordingly and calls it with the given positional and keyword arguments. Returns
        # whatever update_state returns (typically an update operation or nothing).
        control_status = ag_ctx.control_status_ctx()
        ag_update_state = autograph.tf_convert(obj_update_state, control_status)
        return ag_update_state(*args, **kwargs)
    else:
      if isinstance(obj.update_state, def_function.Function):
        update_state_fn = obj.update_state
      else:
        update_state_fn = def_function.function(obj.update_state)

    obj.update_state = types.MethodType(
        metrics_utils.update_state_wrapper(update_state_fn), obj)

    obj_result = obj.result

    def result_fn(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: Nested adapter used by the allocator to wrap a metric's result method.
      # Reads the current autograph control status, converts the captured result method
      # accordingly and calls it with the given arguments. Returns the metric value tensor.
      control_status = ag_ctx.control_status_ctx()
      ag_result = autograph.tf_convert(obj_result, control_status)
      return ag_result(*args, **kwargs)

    obj.result = types.MethodType(metrics_utils.result_wrapper(result_fn), obj)

    return obj

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists the positional arguments as an empty entry; in fact both
    # positional and keyword arguments are forwarded unchanged to update_state.
    # Updates the metric state with a mini-batch and returns the current metric value. Inputs are
    # arbitrary positional and keyword arguments (typically labels, predictions and optional
    # sample weights). The work is done through a replica-local helper invoked via the
    # distributed-training utility, so under a distribution strategy it runs in each replica.
    # Returns the result tensor, which carries a reference to this metric object as metadata.
    # Side effects: updates state variables. Imports the distributed training utilities lazily.
    """Accumulates statistics and then computes metric result value.

    Args:
      *args:
      **kwargs: A mini-batch of inputs to the Metric,
        passed on to `update_state()`.

    Returns:
      The metric value tensor.
    """

    def replica_local_fn(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: Nested helper of the call method that runs once per replica.
      # Inputs are the batch arguments. If any argument is a symbolic Keras tensor, no state
      # update is performed (update op is nothing); otherwise update_state is invoked. Then the
      # result is computed under control dependencies on the update op, the metric object is
      # attached to the result tensor as metadata (used by add_metric in graph mode to reset
      # state per epoch) and the result tensor is returned.
      """Updates the state of the metric in a replica-local context."""
      if any(
          isinstance(arg, keras_tensor.KerasTensor)
          for arg in nest.flatten((args, kwargs))):
        update_op = None
      else:
        update_op = self.update_state(*args, **kwargs)  # pylint: disable=not-callable
      update_ops = []
      if update_op is not None:
        update_ops.append(update_op)
      with ops.control_dependencies(update_ops):
        result_t = self.result()  # pylint: disable=not-callable

        # We are adding the metric object as metadata on the result tensor.
        # This is required when we want to use a metric with `add_metric` API on
        # a Model/Layer in graph mode. This metric instance will later be used
        # to reset variable state after each epoch of training.
        # Example:
        #   model = Model()
        #   mean = Mean()
        #   model.add_metric(mean(values), name='mean')
        result_t._metric_obj = self  # pylint: disable=protected-access
        return result_t

    from tensorflow.python.keras.distribute import distributed_training_utils  # pylint:disable=g-import-not-at-top
    return distributed_training_utils.call_replica_local_fn(
        replica_local_fn, *args, **kwargs)

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Read-only property returning the data type (as stored internally, normally a name string)
    # used for the metric's result and state variables. Takes no inputs; no side effects.
    return self._dtype

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a mapping with the metric's name and dtype, which is the minimal configuration
    # needed to recreate it. Subclasses extend it with their own constructor arguments.
    """Returns the serializable config of the metric."""
    return {'name': self.name, 'dtype': self.dtype}

  def reset_state(self):
    # [FUNCTION DOC]
    # Resets every state variable of the metric to zero. If a subclass overrides the deprecated
    # reset_states method (detected because it is no longer the default implementation), a
    # warning is emitted asking to rename it and that method is called instead, and its return
    # value is returned. Otherwise all variables are set to zero in one backend batch update.
    # Returns nothing in the default path. Side effects: modifies variables, may warn.
    """Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.
    """
    if not generic_utils.is_default(self.reset_states):
      warnings.warn('Metric %s implements a `reset_states()` method; rename it '
                    'to `reset_state()` (without the final "s"). The name '
                    '`reset_states()` has been deprecated to improve API '
                    'consistency.' % (self.__class__.__name__,))
      return self.reset_states()
    else:
      backend.batch_set_value([(v, 0) for v in self.variables])

  @abc.abstractmethod
  def update_state(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Abstract method that subclasses implement to fold a mini-batch of inputs into the state
    # variables. Inputs are arbitrary positional and keyword arguments. This base version always
    # raises a not-implemented error. In graph mode the implementation runs as a graph function.
    """Accumulates statistics for the metric.

    Note: This function is executed as a graph function in graph mode.
    This means:
      a) Operations on the same resource are executed in textual order.
         This should make it easier to do things like add the updated
         value of a variable to another, for example.
      b) You don't need to worry about collecting the update ops to execute.
         All update ops added to the graph by this function will be executed.
      As a result, code should generally work the same way with graph or
      eager execution.

    Args:
      *args:
      **kwargs: A mini-batch of inputs to the Metric.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def result(self):
    # [FUNCTION DOC]
    # Abstract method that subclasses implement to compute the metric value from the state
    # variables without modifying them. Takes no inputs. This base version always raises a
    # not-implemented error.
    """Computes and returns the metric value tensor.

    Result computation is an idempotent operation that simply calculates the
    metric value using the state variables.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  ### For use by subclasses ###
  @doc_controls.for_subclass_implementers
  def add_weight(
      self,
      name,
      shape=(),
      aggregation=variables_module.VariableAggregation.SUM,
      synchronization=variables_module.VariableSynchronization.ON_READ,
      initializer=None,
      dtype=None):
    # [FUNCTION DOC]
    # Creates a non-trainable state variable owned by the metric; intended for subclass
    # constructors. Inputs: name of the variable; shape (defaults to a scalar); aggregation mode
    # across replicas (defaults to sum); synchronization mode (defaults to on-read, but forced to
    # on-write when the active distribution strategy is a TPU strategy); initializer (optional);
    # dtype (optional, defaults to the metric dtype). Variable creation happens in the init scope
    # so it escapes any function graph. Returns the new variable and registers it with the layer.
    """Adds state variable. Only for use by subclasses."""
    if distribute_lib.has_strategy():
      strategy = distribute_lib.get_strategy()
    else:
      strategy = None

    # TODO(b/120571621): Make `ON_READ` work with Keras metrics on TPU.
    if backend.is_tpu_strategy(strategy):
      synchronization = variables_module.VariableSynchronization.ON_WRITE

    with ops.init_scope():
      return super(Metric, self).add_weight(
          name=name,
          shape=shape,
          dtype=self._dtype if dtype is None else dtype,
          trainable=False,
          initializer=initializer,
          collections=[],
          synchronization=synchronization,
          aggregation=aggregation)

  ### End: For use by subclasses ###

  @property
  def trainable_weights(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the list that is extended with the sub-metric weights is the layer's own
    # internal trainable-weights list (not a copy), so repeated access mutates internal state.
    # Property returning the trainable weights of this metric including those of nested metrics,
    # de-duplicated; returns an empty list if the metric is not trainable. Metric variables are
    # created non-trainable, so this is normally empty.
    # Overridden from Layer class to track submetric weights.
    if self.trainable:
      trainable_weights = self._trainable_weights
      for m in self._metrics:
        trainable_weights += m.trainable_weights
      return self._dedup_weights(trainable_weights)
    else:
      return []

  @property
  def non_trainable_weights(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when the metric is trainable, the internal non-trainable-weights list is
    # extended in place with the sub-metric weights (not a copy), which mutates internal state on
    # each access.
    # Property returning the non-trainable weights (where metric state variables live), including
    # those of nested metrics, de-duplicated. If the metric is marked not trainable, its trainable
    # weights are included as well, and all weights of sub-metrics are added.
    # Overridden from Layer class to track submetric weights.
    if self.trainable:
      non_trainable_weights = self._non_trainable_weights
      for m in self._metrics:
        non_trainable_weights += m.non_trainable_weights
    else:
      non_trainable_weights = (
          self._non_trainable_weights + self._trainable_weights)
      for m in self._metrics:
        non_trainable_weights += m.weights
    return self._dedup_weights(non_trainable_weights)

  @property
  def _trackable_saved_model_saver(self):
    # [FUNCTION DOC]
    # HELPER: Hook used by SavedModel serialization.
    # Property returning a metric-specific saved-model saver object created for this metric.
    return metric_serialization.MetricSavedModelSaver(self)

  @generic_utils.default
  @doc_controls.do_not_generate_docs
  def reset_states(self):
    # [FUNCTION DOC]
    # HELPER: Deprecated backwards-compatibility alias of reset_state (marked as default so the
    # base class can detect overrides). Takes no inputs and returns the result of reset_state.
    # Backwards compatibility alias of `reset_state`. New classes should
    # only implement `reset_state`.
    return self.reset_state()


class Reduce(Metric):
  # [CLASS DOC]
  # Base class for metrics that reduce a stream of values to one number (sum, sum over batch size
  # or weighted mean). Holds a reduction mode, a "total" accumulator and, for the mean-like
  # reductions, a "count" accumulator. Used directly as the parent of Sum and Mean.
  """Encapsulates metrics that perform a reduce operation on the values.

  Args:
    reduction: a `tf.keras.metrics.Reduction` enum value.
    name: string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

  def __init__(self, reduction, name, dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: reduction (a reduction enum value: sum, sum over batch size or weighted
    # mean), name (string name of the metric, required) and optional dtype. Stores the reduction,
    # creates a zero-initialized scalar "total" variable and, for sum-over-batch-size and
    # weighted-mean reductions, a zero-initialized "count" variable. Returns nothing.
    super(Reduce, self).__init__(name=name, dtype=dtype)
    self.reduction = reduction
    self.total = self.add_weight(
        'total', initializer=init_ops.zeros_initializer)
    if reduction in [metrics_utils.Reduction.SUM_OVER_BATCH_SIZE,
                     metrics_utils.Reduction.WEIGHTED_MEAN]:
      self.count = self.add_weight(
          'count', initializer=init_ops.zeros_initializer)

  def update_state(self, values, sample_weight=None):
    # [FUNCTION DOC]
    # Accumulates a batch of values. Inputs: values (tensor, possibly ragged, of per-example
    # values) and optional sample_weight (weights broadcastable to the values). Values are cast to
    # the metric dtype; if casting fails a runtime error is raised stating a metric function must
    # return a single tensor (with a hint for dict outputs). Weights are cast, dimensions are
    # squeezed/expanded to match, then broadcast to the values; if broadcasting fails the values
    # are reduced (sum for the sum reduction, mean otherwise) over the extra trailing axes to the
    # weight rank. Values are multiplied by weights, summed and added to total. For the sum
    # reduction the total update op is returned. Otherwise count is increased by the number of
    # elements (sum over batch size, or weighted mean without weights) or by the sum of weights
    # (weighted mean with weights), and that update op is returned. Raises a not-implemented error
    # for an unknown reduction. Side effects: updates the variables.
    """Accumulates statistics for computing the metric.

    Args:
      values: Per-example value.
      sample_weight: Optional weighting of each example. Defaults to 1.

    Returns:
      Update op.
    """
    [values], sample_weight = \
        metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [values], sample_weight)
    try:
      values = math_ops.cast(values, self._dtype)
    except (ValueError, TypeError):
      msg = ('The output of a metric function can only be a single Tensor. '
             'Got: %s' % (values,))
      if isinstance(values, dict):
        msg += ('. To return a dict of values, implement a custom Metric '
                'subclass.')
      raise RuntimeError(msg)
    if sample_weight is not None:
      sample_weight = math_ops.cast(sample_weight, self._dtype)
      # Update dimensions of weights to match with values if possible.
      values, _, sample_weight = losses_utils.squeeze_or_expand_dimensions(
          values, sample_weight=sample_weight)
      try:
        # Broadcast weights if possible.
        sample_weight = weights_broadcast_ops.broadcast_weights(
            sample_weight, values)
      except ValueError:
        # Reduce values to same ndim as weight array
        ndim = backend.ndim(values)
        weight_ndim = backend.ndim(sample_weight)
        if self.reduction == metrics_utils.Reduction.SUM:
          values = math_ops.reduce_sum(
              values, axis=list(range(weight_ndim, ndim)))
        else:
          values = math_ops.reduce_mean(
              values, axis=list(range(weight_ndim, ndim)))
      values = math_ops.multiply(values, sample_weight)

    value_sum = math_ops.reduce_sum(values)
    with ops.control_dependencies([value_sum]):
      update_total_op = self.total.assign_add(value_sum)

    # Exit early if the reduction doesn't have a denominator.
    if self.reduction == metrics_utils.Reduction.SUM:
      return update_total_op

    # Update `count` for reductions that require a denominator.
    if self.reduction == metrics_utils.Reduction.SUM_OVER_BATCH_SIZE:
      num_values = math_ops.cast(array_ops.size(values), self._dtype)
    elif self.reduction == metrics_utils.Reduction.WEIGHTED_MEAN:
      if sample_weight is None:
        num_values = math_ops.cast(array_ops.size(values), self._dtype)
      else:
        num_values = math_ops.reduce_sum(sample_weight)
    else:
      raise NotImplementedError(
          'reduction [%s] not implemented' % self.reduction)

    with ops.control_dependencies([update_total_op]):
      return self.count.assign_add(num_values)

  def result(self):
    # [FUNCTION DOC]
    # Computes the metric value. For the sum reduction returns a copy of total; for weighted mean
    # and sum over batch size returns total divided by count, giving zero when count is zero.
    # Raises a not-implemented error for any other reduction. No inputs, no side effects.
    if self.reduction == metrics_utils.Reduction.SUM:
      return array_ops.identity(self.total)
    elif self.reduction in [
        metrics_utils.Reduction.WEIGHTED_MEAN,
        metrics_utils.Reduction.SUM_OVER_BATCH_SIZE
    ]:
      return math_ops.div_no_nan(self.total, self.count)
    else:
      raise NotImplementedError(
          'reduction [%s] not implemented' % self.reduction)


class Sum(Reduce):
  # [CLASS DOC]
  # Metric that computes the (optionally weighted) sum of all values seen. A thin subclass of
  # Reduce fixed to the sum reduction; state is a single "total" variable.
  """Computes the (weighted) sum of the given values.

  For example, if values is [1, 3, 5, 7] then the sum is 16.
  If the weights were specified as [1, 1, 0, 0] then the sum would be 4.

  This metric creates one variable, `total`, that is used to compute the sum of
  `values`. This is ultimately returned as `sum`.

  If `sample_weight` is `None`, weights default to 1.  Use `sample_weight` of 0
  to mask values.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.Sum()
  >>> m.update_state([1, 3, 5, 7])
  >>> m.result().numpy()
  16.0

  Usage with `compile()` API:

  ```python
  model.add_metric(tf.keras.metrics.Sum(name='sum_1')(outputs))
  model.compile(optimizer='sgd', loss='mse')
  ```
  """

  def __init__(self, name='sum', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "sum") and optional dtype. Initializes the
    # parent with the sum reduction. Returns nothing.
    super(Sum, self).__init__(reduction=metrics_utils.Reduction.SUM,
                              name=name, dtype=dtype)


class Mean(Reduce):
  # [CLASS DOC]
  # Metric that computes the (optionally weighted) mean of all values seen, using "total" and
  # "count" variables. A thin subclass of Reduce fixed to the weighted-mean reduction; it is also
  # the parent of several other metrics.
  """Computes the (weighted) mean of the given values.

  For example, if values is [1, 3, 5, 7] then the mean is 4.
  If the weights were specified as [1, 1, 0, 0] then the mean would be 2.

  This metric creates two variables, `total` and `count` that are used to
  compute the average of `values`. This average is ultimately returned as `mean`
  which is an idempotent operation that simply divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.Mean()
  >>> m.update_state([1, 3, 5, 7])
  >>> m.result().numpy()
  4.0
  >>> m.reset_state()
  >>> m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
  >>> m.result().numpy()
  2.0

  Usage with `compile()` API:

  ```python
  model.add_metric(tf.keras.metrics.Mean(name='mean_1')(outputs))
  model.compile(optimizer='sgd', loss='mse')
  ```
  """

  def __init__(self, name='mean', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "mean") and optional dtype. Initializes the
    # parent with the weighted-mean reduction. Returns nothing.
    super(Mean, self).__init__(
        reduction=metrics_utils.Reduction.WEIGHTED_MEAN, name=name, dtype=dtype)


class MeanRelativeError(Mean):
  # [CLASS DOC]
  # Metric computing the mean of |y_true - y_pred| divided by a user-supplied normalizer (with
  # zero denominators giving zero). Keeps the normalizer as a tensor attribute, and otherwise
  # relies on the Mean accumulators.
  """Computes the mean relative error by normalizing with the given values.

  This metric creates two local variables, `total` and `count` that are used to
  compute the mean relative error. This is weighted by `sample_weight`, and
  it is ultimately returned as `mean_relative_error`:
  an idempotent operation that simply divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    normalizer: The normalizer values with same shape as predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.MeanRelativeError(normalizer=[1, 3, 2, 3])
  >>> m.update_state([1, 3, 2, 3], [2, 4, 6, 8])

  >>> # metric = mean(|y_pred - y_true| / normalizer)
  >>> #        = mean([1, 1, 4, 5] / [1, 3, 2, 3]) = mean([1, 1/3, 2, 5/3])
  >>> #        = 5/4 = 1.25
  >>> m.result().numpy()
  1.25

  Usage with `compile()` API:

  ```python
  model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.MeanRelativeError(normalizer=[1, 3])])
  ```
  """

  def __init__(self, normalizer, name=None, dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: normalizer (tensor or array-like with the same shape as predictions,
    # or broadcast-compatible), optional name and dtype. Casts the normalizer to the metric dtype
    # and stores it. Returns nothing.
    super(MeanRelativeError, self).__init__(name=name, dtype=dtype)
    normalizer = math_ops.cast(normalizer, self._dtype)
    self.normalizer = normalizer

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # Accumulates relative errors. Inputs: y_true and y_pred (ground truth and predictions),
    # optional sample_weight. Casts both to the metric dtype, flattens ragged inputs, aligns
    # dimensions, squeezes the stored normalizer to match the predictions (this reassigns the
    # normalizer attribute), asserts the prediction shape is compatible with the labels (raises a
    # shape error otherwise), computes absolute error divided by the normalizer (zero where the
    # normalizer is zero) and passes it to the parent update. Returns the update op.
    """Accumulates metric statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    [y_pred, y_true], sample_weight = \
        metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_pred, y_true], sample_weight)
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
        y_pred, y_true)

    y_pred, self.normalizer = losses_utils.remove_squeezable_dimensions(
        y_pred, self.normalizer)
    y_pred.shape.assert_is_compatible_with(y_true.shape)
    relative_errors = math_ops.div_no_nan(
        math_ops.abs(y_true - y_pred), self.normalizer)

    return super(MeanRelativeError, self).update_state(
        relative_errors, sample_weight=sample_weight)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration plus the normalizer, evaluated to a concrete value if it is
    # a tensor or variable. Returns a mapping.
    n = self.normalizer
    config = {'normalizer': backend.eval(n) if is_tensor_or_variable(n) else n}
    base_config = super(MeanRelativeError, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MeanMetricWrapper(Mean):
  # [CLASS DOC]
  # Adapter that turns a stateless metric function (taking labels and predictions and returning
  # per-sample values) into a stateful mean metric. Holds the wrapped function and extra keyword
  # arguments for it. Parent of most accuracy and loss-style metrics in this module.
  """Wraps a stateless metric function with the Mean metric.

  You could use this class to quickly build a mean metric from a function. The
  function needs to have the signature `fn(y_true, y_pred)` and return a
  per-sample loss array. `MeanMetricWrapper.result()` will return
  the average metric value across all samples seen so far.

  For example:

  ```python
  def accuracy(y_true, y_pred):
    return tf.cast(tf.math.equal(y_true, y_pred), tf.float32)

  accuracy_metric = tf.keras.metrics.MeanMetricWrapper(fn=accuracy)

  keras_model.compile(..., metrics=accuracy_metric)
  ```

  Args:
    fn: The metric function to wrap, with signature `fn(y_true, y_pred,
      **kwargs)`.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    **kwargs: Keyword arguments to pass on to `fn`.
  """

  def __init__(self, fn, name=None, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: fn (callable taking labels, predictions and optional keyword
    # arguments), optional name and dtype, and extra keyword arguments that are stored and passed
    # to fn on every update. Returns nothing.
    super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # Accumulates the mean of the wrapped function's output. Inputs: y_true, y_pred and optional
    # sample_weight. Casts labels and predictions to the metric dtype, flattens ragged inputs,
    # aligns dimensions, calls the wrapped function (autograph-converted) with the stored keyword
    # arguments and feeds the per-sample values and weights to the parent update. Returns the
    # update op.
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
      sample_weight: Optional `sample_weight` acts as a
        coefficient for the metric. If a scalar is provided, then the metric is
        simply scaled by the given value. If `sample_weight` is a tensor of size
        `[batch_size]`, then the metric for each sample of the batch is rescaled
        by the corresponding element in the `sample_weight` vector. If the shape
        of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted
        to this shape), then each metric element of `y_pred` is scaled by the
        corresponding value of `sample_weight`. (Note on `dN-1`: all metric
        functions reduce by 1 dimension, usually the last axis (-1)).

    Returns:
      Update op.
    """
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    [y_true, y_pred], sample_weight = (
        metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight))
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
        y_pred, y_true)

    ag_fn = autograph.tf_convert(self._fn, ag_ctx.control_status_ctx())
    matches = ag_fn(y_true, y_pred, **self._fn_kwargs)
    return super(MeanMetricWrapper, self).update_state(
        matches, sample_weight=sample_weight)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the configuration mapping: the wrapped function only when the instance is exactly
    # this class (not a subclass), the stored keyword arguments (tensors/variables evaluated to
    # values), plus the base configuration.
    config = {}

    if type(self) is MeanMetricWrapper:  # pylint: disable=unidiomatic-typecheck
      # Only include function argument when the object is a MeanMetricWrapper
      # and not a subclass.
      config['fn'] = self._fn

    for k, v in self._fn_kwargs.items():
      config[k] = backend.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(MeanMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    # [FUNCTION DOC]
    # Class method that rebuilds a metric from a configuration mapping. The function entry is
    # removed from the (mutated) mapping; when called on this class itself, the function is
    # looked up by name/identifier through the module-level get function and the rest of the
    # mapping becomes constructor arguments; for subclasses the parent implementation is used.
    # Returns a new metric instance.
    # Note that while MeanMetricWrapper itself isn't public, objects of this
    # class may be created and added to the model by calling model.compile.
    fn = config.pop('fn', None)
    if cls is MeanMetricWrapper:
      return cls(get(fn), **config)
    return super(MeanMetricWrapper, cls).from_config(config)


class Accuracy(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting how often predictions exactly equal labels (mean of elementwise equality),
  # built on the mean wrapper around the module's accuracy function.
  """Calculates how often predictions equal labels.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `binary accuracy`: an idempotent operation that simply
  divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.Accuracy()
  >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
  >>> m.result().numpy()
  0.75

  >>> m.reset_state()
  >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]],
  ...                sample_weight=[1, 1, 0, 0])
  >>> m.result().numpy()
  0.5

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.Accuracy()])
  ```
  """

  def __init__(self, name='accuracy', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "accuracy") and dtype. Wraps the module's
    # accuracy function. Returns nothing.
    super(Accuracy, self).__init__(accuracy, name, dtype=dtype)


class BinaryAccuracy(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting how often thresholded predictions match binary labels; mean wrapper around
  # the module's binary accuracy function with a configurable threshold.
  """Calculates how often predictions match binary labels.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `binary accuracy`: an idempotent operation that simply
  divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    threshold: (Optional) Float representing the threshold for deciding
    whether prediction values are 1 or 0.

  Standalone usage:

  >>> m = tf.keras.metrics.BinaryAccuracy()
  >>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]])
  >>> m.result().numpy()
  0.75

  >>> m.reset_state()
  >>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]],
  ...                sample_weight=[1, 0, 0, 1])
  >>> m.result().numpy()
  0.5

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.BinaryAccuracy()])
  ```
  """

  def __init__(self, name='binary_accuracy', dtype=None, threshold=0.5):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "binary_accuracy"), dtype, and threshold
    # (float, default 0.5) above which a prediction counts as one. Returns nothing.
    super(BinaryAccuracy, self).__init__(
        binary_accuracy, name, dtype=dtype, threshold=threshold)


class CategoricalAccuracy(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting how often the arg-max of predictions equals the arg-max of one-hot labels;
  # mean wrapper around the module's categorical accuracy function.
  """Calculates how often predictions match one-hot labels.

  You can provide logits of classes as `y_pred`, since argmax of
  logits and probabilities are same.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `categorical accuracy`: an idempotent operation that
  simply divides `total` by `count`.

  `y_pred` and `y_true` should be passed in as vectors of probabilities, rather
  than as labels. If necessary, use `tf.one_hot` to expand `y_true` as a vector.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.CategoricalAccuracy()
  >>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
  ...                 [0.05, 0.95, 0]])
  >>> m.result().numpy()
  0.5

  >>> m.reset_state()
  >>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
  ...                 [0.05, 0.95, 0]],
  ...                sample_weight=[0.7, 0.3])
  >>> m.result().numpy()
  0.3

  Usage with `compile()` API:

  ```python
  model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.CategoricalAccuracy()])
  ```
  """

  def __init__(self, name='categorical_accuracy', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "categorical_accuracy") and dtype. Wraps the
    # categorical accuracy function. Returns nothing.
    super(CategoricalAccuracy, self).__init__(
        categorical_accuracy, name, dtype=dtype)


class SparseCategoricalAccuracy(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting how often the arg-max of predictions equals integer class labels; mean
  # wrapper around the module's sparse categorical accuracy function.
  """Calculates how often predictions match integer labels.

  ```python
  acc = np.dot(sample_weight, np.equal(y_true, np.argmax(y_pred, axis=1))
  ```

  You can provide logits of classes as `y_pred`, since argmax of
  logits and probabilities are same.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `sparse categorical accuracy`: an idempotent operation
  that simply divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.SparseCategoricalAccuracy()
  >>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
  >>> m.result().numpy()
  0.5

  >>> m.reset_state()
  >>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]],
  ...                sample_weight=[0.7, 0.3])
  >>> m.result().numpy()
  0.3

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  ```
  """

  def __init__(self, name='sparse_categorical_accuracy', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "sparse_categorical_accuracy") and dtype. Wraps
    # the sparse categorical accuracy function. Returns nothing.
    super(SparseCategoricalAccuracy, self).__init__(
        sparse_categorical_accuracy, name, dtype=dtype)


class TopKCategoricalAccuracy(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting how often the true class (one-hot labels) is among the k highest
  # predictions; mean wrapper around the module's top-k categorical accuracy function.
  """Computes how often targets are in the top `K` predictions.

  Args:
    k: (Optional) Number of top elements to look at for computing accuracy.
      Defaults to 5.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
  >>> m.update_state([[0, 0, 1], [0, 1, 0]],
  ...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
  >>> m.result().numpy()
  0.5

  >>> m.reset_state()
  >>> m.update_state([[0, 0, 1], [0, 1, 0]],
  ...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
  ...                sample_weight=[0.7, 0.3])
  >>> m.result().numpy()
  0.3

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.TopKCategoricalAccuracy()])
  ```
  """

  def __init__(self, k=5, name='top_k_categorical_accuracy', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: k (integer number of top predictions considered, default 5), optional
    # name (default "top_k_categorical_accuracy") and dtype. Returns nothing.
    super(TopKCategoricalAccuracy, self).__init__(
        top_k_categorical_accuracy, name, dtype=dtype, k=k)


class SparseTopKCategoricalAccuracy(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting how often the integer true class is among the k highest predictions; mean
  # wrapper around the module's sparse top-k categorical accuracy function.
  """Computes how often integer targets are in the top `K` predictions.

  Args:
    k: (Optional) Number of top elements to look at for computing accuracy.
      Defaults to 5.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
  >>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
  >>> m.result().numpy()
  0.5

  >>> m.reset_state()
  >>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
  ...                sample_weight=[0.7, 0.3])
  >>> m.result().numpy()
  0.3

  Usage with `compile()` API:

  ```python
  model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy()])
  ```
  """

  def __init__(self, k=5, name='sparse_top_k_categorical_accuracy', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: k (integer number of top predictions considered, default 5), optional
    # name (default "sparse_top_k_categorical_accuracy") and dtype. Returns nothing.
    super(SparseTopKCategoricalAccuracy, self).__init__(
        sparse_top_k_categorical_accuracy, name, dtype=dtype, k=k)


class _ConfusionMatrixConditionCount(Metric):
  # [CLASS DOC]
  # HELPER: Shared base of FalsePositives, FalseNegatives, TrueNegatives and TruePositives.
  # Counts (weighted) occurrences of one confusion-matrix condition for each of one or more
  # thresholds in a single "accumulator" vector variable. The result is a scalar when there is a
  # single threshold, otherwise a vector.
  """Calculates the number of the given confusion matrix condition.

  Args:
    confusion_matrix_cond: One of `metrics_utils.ConfusionMatrix` conditions.
    thresholds: (Optional) Defaults to 0.5. A float value or a python list/tuple
      of float threshold values in [0, 1]. A threshold is compared with
      prediction values to determine the truth value of predictions (i.e., above
      the threshold is `true`, below is `false`). One metric value is generated
      for each threshold value.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

  def __init__(self,
               confusion_matrix_cond,
               thresholds=None,
               name=None,
               dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: confusion_matrix_cond (which confusion-matrix condition to count),
    # thresholds (optional float or list/tuple of floats in 0 to 1; default 0.5), name, dtype.
    # Parses and stores the thresholds, records whether they are evenly distributed, and creates a
    # zero-initialized accumulator variable with one entry per threshold. Returns nothing.
    super(_ConfusionMatrixConditionCount, self).__init__(name=name, dtype=dtype)
    self._confusion_matrix_cond = confusion_matrix_cond
    self.init_thresholds = thresholds
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=0.5)
    self._thresholds_distributed_evenly = (
        metrics_utils.is_evenly_distributed_thresholds(self.thresholds))
    self.accumulator = self.add_weight(
        'accumulator',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # Adds the weighted count of the configured condition, per threshold, to the accumulator.
    # Inputs: y_true, y_pred and optional sample_weight. Delegates to the confusion-matrix update
    # utility and returns its update op.
    """Accumulates the metric statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    return metrics_utils.update_confusion_matrix_variables(
        {self._confusion_matrix_cond: self.accumulator},
        y_true,
        y_pred,
        thresholds=self.thresholds,
        thresholds_distributed_evenly=self._thresholds_distributed_evenly,
        sample_weight=sample_weight)

  def result(self):
    # [FUNCTION DOC]
    # Returns the accumulator as a tensor: its first element when there is one threshold,
    # otherwise the whole vector. No side effects.
    if len(self.thresholds) == 1:
      result = self.accumulator[0]
    else:
      result = self.accumulator
    return tensor_conversion.convert_to_tensor_v2_with_dispatch(result)

  def reset_state(self):
    # [FUNCTION DOC]
    # Sets every variable of the metric to a zero vector with one entry per threshold. Returns
    # nothing; side effect is the variable update.
    num_thresholds = len(to_list(self.thresholds))
    backend.batch_set_value(
        [(v, np.zeros((num_thresholds,))) for v in self.variables])

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration plus the thresholds exactly as originally passed to the
    # constructor (possibly nothing).
    config = {'thresholds': self.init_thresholds}
    base_config = super(_ConfusionMatrixConditionCount, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class FalsePositives(_ConfusionMatrixConditionCount):
  # [CLASS DOC]
  # Metric counting (weighted) false positives per threshold; specialization of the shared
  # confusion-condition counter.
  """Calculates the number of false positives.

  If `sample_weight` is given, calculates the sum of the weights of
  false positives. This metric creates one local variable, `accumulator`
  that is used to keep track of the number of false positives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    thresholds: (Optional) Defaults to 0.5. A float value or a python
      list/tuple of float threshold values in [0, 1]. A threshold is compared
      with prediction values to determine the truth value of predictions
      (i.e., above the threshold is `true`, below is `false`). One metric
      value is generated for each threshold value.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.FalsePositives()
  >>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1])
  >>> m.result().numpy()
  2.0

  >>> m.reset_state()
  >>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1], sample_weight=[0, 0, 1, 0])
  >>> m.result().numpy()
  1.0

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.FalsePositives()])
  ```
  """

  def __init__(self, thresholds=None, name=None, dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional thresholds (float or list/tuple of floats in 0 to 1, default
    # 0.5), name and dtype. Configures the parent to count false positives. Returns nothing.
    super(FalsePositives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_POSITIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)


class FalseNegatives(_ConfusionMatrixConditionCount):
  # [CLASS DOC]
  # Metric counting (weighted) false negatives per threshold; specialization of the shared
  # confusion-condition counter.
  """Calculates the number of false negatives.

  If `sample_weight` is given, calculates the sum of the weights of
  false negatives. This metric creates one local variable, `accumulator`
  that is used to keep track of the number of false negatives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    thresholds: (Optional) Defaults to 0.5. A float value or a python
      list/tuple of float threshold values in [0, 1]. A threshold is compared
      with prediction values to determine the truth value of predictions
      (i.e., above the threshold is `true`, below is `false`). One metric
      value is generated for each threshold value.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.FalseNegatives()
  >>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
  >>> m.result().numpy()
  2.0

  >>> m.reset_state()
  >>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0], sample_weight=[0, 0, 1, 0])
  >>> m.result().numpy()
  1.0

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.FalseNegatives()])
  ```
  """

  def __init__(self, thresholds=None, name=None, dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional thresholds (float or list/tuple of floats in 0 to 1, default
    # 0.5), name and dtype. Configures the parent to count false negatives. Returns nothing.
    super(FalseNegatives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_NEGATIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)


class TrueNegatives(_ConfusionMatrixConditionCount):
  # [CLASS DOC]
  # Metric counting (weighted) true negatives per threshold; specialization of the shared
  # confusion-condition counter.
  """Calculates the number of true negatives.

  If `sample_weight` is given, calculates the sum of the weights of
  true negatives. This metric creates one local variable, `accumulator`
  that is used to keep track of the number of true negatives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    thresholds: (Optional) Defaults to 0.5. A float value or a python
      list/tuple of float threshold values in [0, 1]. A threshold is compared
      with prediction values to determine the truth value of predictions
      (i.e., above the threshold is `true`, below is `false`). One metric
      value is generated for each threshold value.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.TrueNegatives()
  >>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0])
  >>> m.result().numpy()
  2.0

  >>> m.reset_state()
  >>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0], sample_weight=[0, 0, 1, 0])
  >>> m.result().numpy()
  1.0

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.TrueNegatives()])
  ```
  """

  def __init__(self, thresholds=None, name=None, dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional thresholds (float or list/tuple of floats in 0 to 1, default
    # 0.5), name and dtype. Configures the parent to count true negatives. Returns nothing.
    super(TrueNegatives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_NEGATIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)


class TruePositives(_ConfusionMatrixConditionCount):
  # [CLASS DOC]
  # Metric counting (weighted) true positives per threshold; specialization of the shared
  # confusion-condition counter.
  """Calculates the number of true positives.

  If `sample_weight` is given, calculates the sum of the weights of
  true positives. This metric creates one local variable, `true_positives`
  that is used to keep track of the number of true positives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    thresholds: (Optional) Defaults to 0.5. A float value or a python
      list/tuple of float threshold values in [0, 1]. A threshold is compared
      with prediction values to determine the truth value of predictions
      (i.e., above the threshold is `true`, below is `false`). One metric
      value is generated for each threshold value.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.TruePositives()
  >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
  >>> m.result().numpy()
  2.0

  >>> m.reset_state()
  >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
  >>> m.result().numpy()
  1.0

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.TruePositives()])
  ```
  """

  def __init__(self, thresholds=None, name=None, dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional thresholds (float or list/tuple of floats in 0 to 1, default
    # 0.5), name and dtype. Configures the parent to count true positives. Returns nothing.
    super(TruePositives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_POSITIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)


class Precision(Metric):
  # [CLASS DOC]
  # Metric computing precision (true positives divided by true plus false positives) for one or
  # more thresholds, optionally restricted to the top-k predictions and/or one class id. State is
  # two vector variables, true_positives and false_positives, one entry per threshold.
  """Computes the precision of the predictions with respect to the labels.

  The metric creates two local variables, `true_positives` and `false_positives`
  that are used to compute the precision. This value is ultimately returned as
  `precision`, an idempotent operation that simply divides `true_positives`
  by the sum of `true_positives` and `false_positives`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  If `top_k` is set, we'll calculate precision as how often on average a class
  among the top-k classes with the highest predicted values of a batch entry is
  correct and can be found in the label for that entry.

  If `class_id` is specified, we calculate precision by considering only the
  entries in the batch for which `class_id` is above the threshold and/or in the
  top-k highest predictions, and computing the fraction of them for which
  `class_id` is indeed a correct label.

  Args:
    thresholds: (Optional) A float value or a python list/tuple of float
      threshold values in [0, 1]. A threshold is compared with prediction
      values to determine the truth value of predictions (i.e., above the
      threshold is `true`, below is `false`). One metric value is generated
      for each threshold value. If neither thresholds nor top_k are set, the
      default is to calculate precision with `thresholds=0.5`.
    top_k: (Optional) Unset by default. An int value specifying the top-k
      predictions to consider when calculating precision.
    class_id: (Optional) Integer class ID for which we want binary metrics.
      This must be in the half-open interval `[0, num_classes)`, where
      `num_classes` is the last dimension of predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.Precision()
  >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
  >>> m.result().numpy()
  0.6666667

  >>> m.reset_state()
  >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
  >>> m.result().numpy()
  1.0

  >>> # With top_k=2, it will calculate precision over y_true[:2] and y_pred[:2]
  >>> m = tf.keras.metrics.Precision(top_k=2)
  >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
  >>> m.result().numpy()
  0.0

  >>> # With top_k=4, it will calculate precision over y_true[:4] and y_pred[:4]
  >>> m = tf.keras.metrics.Precision(top_k=4)
  >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
  >>> m.result().numpy()
  0.5

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.Precision()])
  ```
  """

  def __init__(self,
               thresholds=None,
               top_k=None,
               class_id=None,
               name=None,
               dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: thresholds (optional float or list/tuple in 0 to 1), top_k (optional
    # integer), class_id (optional integer class index), name, dtype. The default threshold is 0.5
    # unless top_k is given, in which case it is negative infinity (no threshold filtering).
    # Creates zero-initialized true_positives and false_positives variables, one entry per
    # threshold. Returns nothing.
    super(Precision, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.top_k = top_k
    self.class_id = class_id

    default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=default_threshold)
    self._thresholds_distributed_evenly = (
        metrics_utils.is_evenly_distributed_thresholds(self.thresholds))
    self.true_positives = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.false_positives = self.add_weight(
        'false_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # Accumulates true-positive and false-positive counts per threshold, honoring top_k and
    # class_id. Inputs: y_true (cast to boolean), y_pred (values in 0 to 1), optional
    # sample_weight. Delegates to the confusion-matrix update utility and returns its update op.
    """Accumulates true positive and false positive statistics.

    Args:
      y_true: The ground truth values, with the same dimensions as `y_pred`.
        Will be cast to `bool`.
      y_pred: The predicted values. Each element must be in the range `[0, 1]`.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
        },
        y_true,
        y_pred,
        thresholds=self.thresholds,
        thresholds_distributed_evenly=self._thresholds_distributed_evenly,
        top_k=self.top_k,
        class_id=self.class_id,
        sample_weight=sample_weight)

  def result(self):
    # [FUNCTION DOC]
    # Returns true positives divided by (true plus false positives), zero where the denominator
    # is zero. A scalar for a single threshold, otherwise a vector with one entry per threshold.
    result = math_ops.div_no_nan(self.true_positives,
                                 self.true_positives + self.false_positives)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_state(self):
    # [FUNCTION DOC]
    # Sets the true_positives and false_positives variables to zero vectors with one entry per
    # threshold. Returns nothing.
    num_thresholds = len(to_list(self.thresholds))
    backend.batch_set_value([(v, np.zeros((num_thresholds,)))
                             for v in (self.true_positives,
                                       self.false_positives)])

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration plus the original thresholds, top_k and class_id.
    config = {
        'thresholds': self.init_thresholds,
        'top_k': self.top_k,
        'class_id': self.class_id
    }
    base_config = super(Precision, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Recall(Metric):
  # [CLASS DOC]
  # Metric computing recall (true positives divided by true positives plus false negatives) for
  # one or more thresholds, optionally restricted to top-k predictions and/or one class id.
  # State is two vector variables, true_positives and false_negatives.
  """Computes the recall of the predictions with respect to the labels.

  This metric creates two local variables, `true_positives` and
  `false_negatives`, that are used to compute the recall. This value is
  ultimately returned as `recall`, an idempotent operation that simply divides
  `true_positives` by the sum of `true_positives` and `false_negatives`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  If `top_k` is set, recall will be computed as how often on average a class
  among the labels of a batch entry is in the top-k predictions.

  If `class_id` is specified, we calculate recall by considering only the
  entries in the batch for which `class_id` is in the label, and computing the
  fraction of them for which `class_id` is above the threshold and/or in the
  top-k predictions.

  Args:
    thresholds: (Optional) A float value or a python list/tuple of float
      threshold values in [0, 1]. A threshold is compared with prediction
      values to determine the truth value of predictions (i.e., above the
      threshold is `true`, below is `false`). One metric value is generated
      for each threshold value. If neither thresholds nor top_k are set, the
      default is to calculate recall with `thresholds=0.5`.
    top_k: (Optional) Unset by default. An int value specifying the top-k
      predictions to consider when calculating recall.
    class_id: (Optional) Integer class ID for which we want binary metrics.
      This must be in the half-open interval `[0, num_classes)`, where
      `num_classes` is the last dimension of predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.Recall()
  >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
  >>> m.result().numpy()
  0.6666667

  >>> m.reset_state()
  >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
  >>> m.result().numpy()
  1.0

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.Recall()])
  ```
  """

  def __init__(self,
               thresholds=None,
               top_k=None,
               class_id=None,
               name=None,
               dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: thresholds (optional float or list/tuple in 0 to 1), top_k (optional
    # integer), class_id (optional integer class index), name, dtype. Default threshold is 0.5
    # unless top_k is given, in which case it is negative infinity. Creates zero-initialized
    # true_positives and false_negatives variables, one entry per threshold. Returns nothing.
    super(Recall, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.top_k = top_k
    self.class_id = class_id

    default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=default_threshold)
    self._thresholds_distributed_evenly = (
        metrics_utils.is_evenly_distributed_thresholds(self.thresholds))
    self.true_positives = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.false_negatives = self.add_weight(
        'false_negatives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # Accumulates true-positive and false-negative counts per threshold, honoring top_k and
    # class_id. Inputs: y_true (cast to boolean), y_pred (values in 0 to 1), optional
    # sample_weight. Delegates to the confusion-matrix update utility and returns its update op.
    """Accumulates true positive and false negative statistics.

    Args:
      y_true: The ground truth values, with the same dimensions as `y_pred`.
        Will be cast to `bool`.
      y_pred: The predicted values. Each element must be in the range `[0, 1]`.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
        },
        y_true,
        y_pred,
        thresholds=self.thresholds,
        thresholds_distributed_evenly=self._thresholds_distributed_evenly,
        top_k=self.top_k,
        class_id=self.class_id,
        sample_weight=sample_weight)

  def result(self):
    # [FUNCTION DOC]
    # Returns true positives divided by (true positives plus false negatives), zero where the
    # denominator is zero. A scalar for a single threshold, otherwise a vector.
    result = math_ops.div_no_nan(self.true_positives,
                                 self.true_positives + self.false_negatives)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_state(self):
    # [FUNCTION DOC]
    # Sets the true_positives and false_negatives variables to zero vectors with one entry per
    # threshold. Returns nothing.
    num_thresholds = len(to_list(self.thresholds))
    backend.batch_set_value([(v, np.zeros((num_thresholds,)))
                             for v in (self.true_positives,
                                       self.false_negatives)])

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration plus the original thresholds, top_k and class_id.
    config = {
        'thresholds': self.init_thresholds,
        'top_k': self.top_k,
        'class_id': self.class_id
    }
    base_config = super(Recall, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SensitivitySpecificityBase(Metric, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base for metrics that evaluate one of sensitivity/specificity subject to a
  # constraint on the other. Keeps four vector variables (true/false positives/negatives) over a
  # fixed grid of num_thresholds thresholds between 0 and 1, the constraint value and an optional
  # class id. Subclasses implement the result computation using the constrained-maximum helper.
  """Abstract base class for computing sensitivity and specificity.

  For additional information about specificity and sensitivity, see
  [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
  """

  def __init__(self,
               value,
               num_thresholds=200,
               class_id=None,
               name=None,
               dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: value (the constraint value), num_thresholds (positive integer,
    # default 200), class_id (optional), name, dtype. Raises a value error if num_thresholds is
    # not positive. Creates four zero-initialized variables (true positives, true negatives, false
    # positives, false negatives), one entry per threshold. Thresholds are a single 0.5 when
    # num_thresholds is one; otherwise 0, evenly spaced interior values and 1 (flagged as evenly
    # distributed). Returns nothing.
    super(SensitivitySpecificityBase, self).__init__(name=name, dtype=dtype)
    if num_thresholds <= 0:
      raise ValueError('`num_thresholds` must be > 0.')
    self.value = value
    self.class_id = class_id
    self.true_positives = self.add_weight(
        'true_positives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.true_negatives = self.add_weight(
        'true_negatives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.false_positives = self.add_weight(
        'false_positives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.false_negatives = self.add_weight(
        'false_negatives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)

    # Compute `num_thresholds` thresholds in [0, 1]
    if num_thresholds == 1:
      self.thresholds = [0.5]
      self._thresholds_distributed_evenly = False
    else:
      thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                    for i in range(num_thresholds - 2)]
      self.thresholds = [0.0] + thresholds + [1.0]
      self._thresholds_distributed_evenly = True

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # Accumulates all four confusion-matrix counts per threshold (restricted to class_id if set).
    # Inputs: y_true, y_pred, optional sample_weight. Delegates to the confusion-matrix update
    # utility and returns its update op.
    """Accumulates confusion matrix statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    return metrics_utils.update_confusion_matrix_variables(
        {
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
        },
        y_true,
        y_pred,
        thresholds=self.thresholds,
        thresholds_distributed_evenly=self._thresholds_distributed_evenly,
        class_id=self.class_id,
        sample_weight=sample_weight)

  def reset_state(self):
    # [FUNCTION DOC]
    # Sets all four confusion-matrix variables to zero vectors with one entry per threshold.
    # Returns nothing.
    num_thresholds = len(self.thresholds)
    confusion_matrix_variables = (self.true_positives, self.true_negatives,
                                  self.false_positives, self.false_negatives)
    backend.batch_set_value([
        (v, np.zeros((num_thresholds,))) for v in confusion_matrix_variables
    ])

  def get_config(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the configuration omits num_thresholds and the constraint value, so a metric
    # rebuilt from this base configuration alone would not be equivalent (subclasses add them).
    # Returns the base configuration plus the class id.
    config = {'class_id': self.class_id}
    base_config = super(SensitivitySpecificityBase, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _find_max_under_constraint(self, constrained, dependent, predicate):
    # [FUNCTION DOC]
    # HELPER: Shared by the sensitivity-at-specificity and specificity-at-sensitivity results.
    # Inputs: constrained (rank-1 tensor of values the constraint applies to), dependent (tensor
    # of the same shape from which the maximum is taken, aligned by threshold) and predicate
    # (binary boolean function compared against the stored constraint value). Selects thresholds
    # where the predicate holds and returns the maximum dependent value among them, or 0.0 if no
    # threshold satisfies it.
    """Returns the maximum of dependent_statistic that satisfies the constraint.

    Args:
      constrained: Over these values the constraint
        is specified. A rank-1 tensor.
      dependent: From these values the maximum that satiesfies the
        constraint is selected. Values in this tensor and in
        `constrained` are linked by having the same threshold at each
        position, hence this tensor must have the same shape.
      predicate: A binary boolean functor to be applied to arguments
      `constrained` and `self.value`, e.g. `tf.greater`.

    Returns maximal dependent value, if no value satiesfies the constraint 0.0.
    """
    feasible = array_ops.where_v2(predicate(constrained, self.value))
    feasible_exists = math_ops.greater(array_ops.size(feasible), 0)
    max_dependent = math_ops.reduce_max(array_ops.gather(dependent, feasible))

    return array_ops.where_v2(feasible_exists, max_dependent, 0.0)


class SensitivityAtSpecificity(SensitivitySpecificityBase):
  # [CLASS DOC]
  # Metric returning the best sensitivity among thresholds whose specificity is at least a given
  # value. Stores the specificity and number of thresholds; uses the base class variables.
  """Computes best sensitivity where specificity is >= specified value.

  the sensitivity at a given specificity.

  `Sensitivity` measures the proportion of actual positives that are correctly
  identified as such (tp / (tp + fn)).
  `Specificity` measures the proportion of actual negatives that are correctly
  identified as such (tn / (tn + fp)).

  This metric creates four local variables, `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` that are used to compute the
  sensitivity at the given specificity. The threshold for the given specificity
  value is computed and used to evaluate the corresponding sensitivity.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  If `class_id` is specified, we calculate precision by considering only the
  entries in the batch for which `class_id` is above the threshold predictions,
  and computing the fraction of them for which `class_id` is indeed a correct
  label.

  For additional information about specificity and sensitivity, see
  [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

  Args:
    specificity: A scalar value in range `[0, 1]`.
    num_thresholds: (Optional) Defaults to 200. The number of thresholds to
      use for matching the given specificity.
    class_id: (Optional) Integer class ID for which we want binary metrics.
      This must be in the half-open interval `[0, num_classes)`, where
      `num_classes` is the last dimension of predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.SensitivityAtSpecificity(0.5)
  >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
  >>> m.result().numpy()
  0.5

  >>> m.reset_state()
  >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
  ...                sample_weight=[1, 1, 2, 2, 1])
  >>> m.result().numpy()
  0.333333

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SensitivityAtSpecificity()])
  ```
  """

  def __init__(self,
               specificity,
               num_thresholds=200,
               class_id=None,
               name=None,
               dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: specificity (scalar in 0 to 1), num_thresholds (default 200), class_id
    # (optional), name, dtype. Raises a value error if specificity is outside 0 to 1. Stores
    # specificity and num_thresholds before calling the parent constructor with specificity as
    # the constraint value. Returns nothing.
    if specificity < 0 or specificity > 1:
      raise ValueError('`specificity` must be in the range [0, 1].')
    self.specificity = specificity
    self.num_thresholds = num_thresholds
    super(SensitivityAtSpecificity, self).__init__(
        specificity,
        num_thresholds=num_thresholds,
        class_id=class_id,
        name=name,
        dtype=dtype)

  def result(self):
    # [FUNCTION DOC]
    # Computes specificity (true negatives over true negatives plus false positives) and
    # sensitivity (true positives over true positives plus false negatives) per threshold and
    # returns the largest sensitivity among thresholds whose specificity is greater than or equal
    # to the configured value, or 0.0 if none qualify.
    specificities = math_ops.div_no_nan(
        self.true_negatives, self.true_negatives + self.false_positives)
    sensitivities = math_ops.div_no_nan(
        self.true_positives, self.true_positives + self.false_negatives)
    return self._find_max_under_constraint(
        specificities, sensitivities, math_ops.greater_equal)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration plus num_thresholds and specificity.
    config = {
        'num_thresholds': self.num_thresholds,
        'specificity': self.specificity
    }
    base_config = super(SensitivityAtSpecificity, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SpecificityAtSensitivity(SensitivitySpecificityBase):
  # [CLASS DOC]
  # Metric returning the best specificity among thresholds whose sensitivity is at least a given
  # value. Stores the sensitivity and number of thresholds; uses the base class variables.
  """Computes best specificity where sensitivity is >= specified value.

  `Sensitivity` measures the proportion of actual positives that are correctly
  identified as such (tp / (tp + fn)).
  `Specificity` measures the proportion of actual negatives that are correctly
  identified as such (tn / (tn + fp)).

  This metric creates four local variables, `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` that are used to compute the
  specificity at the given sensitivity. The threshold for the given sensitivity
  value is computed and used to evaluate the corresponding specificity.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  If `class_id` is specified, we calculate precision by considering only the
  entries in the batch for which `class_id` is above the threshold predictions,
  and computing the fraction of them for which `class_id` is indeed a correct
  label.

  For additional information about specificity and sensitivity, see
  [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

  Args:
    sensitivity: A scalar value in range `[0, 1]`.
    num_thresholds: (Optional) Defaults to 200. The number of thresholds to
      use for matching the given sensitivity.
    class_id: (Optional) Integer class ID for which we want binary metrics.
      This must be in the half-open interval `[0, num_classes)`, where
      `num_classes` is the last dimension of predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.SpecificityAtSensitivity(0.5)
  >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
  >>> m.result().numpy()
  0.66666667

  >>> m.reset_state()
  >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
  ...                sample_weight=[1, 1, 2, 2, 2])
  >>> m.result().numpy()
  0.5

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SpecificityAtSensitivity()])
  ```
  """

  def __init__(self,
               sensitivity,
               num_thresholds=200,
               class_id=None,
               name=None,
               dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: sensitivity (scalar in 0 to 1), num_thresholds (default 200), class_id
    # (optional), name, dtype. Raises a value error if sensitivity is outside 0 to 1. Stores
    # sensitivity and num_thresholds, then calls the parent constructor with sensitivity as the
    # constraint value. Returns nothing.
    if sensitivity < 0 or sensitivity > 1:
      raise ValueError('`sensitivity` must be in the range [0, 1].')
    self.sensitivity = sensitivity
    self.num_thresholds = num_thresholds
    super(SpecificityAtSensitivity, self).__init__(
        sensitivity,
        num_thresholds=num_thresholds,
        class_id=class_id,
        name=name,
        dtype=dtype)

  def result(self):
    # [FUNCTION DOC]
    # Computes sensitivity and specificity per threshold and returns the largest specificity
    # among thresholds whose sensitivity is greater than or equal to the configured value, or 0.0
    # if none qualify.
    sensitivities = math_ops.div_no_nan(
        self.true_positives, self.true_positives + self.false_negatives)
    specificities = math_ops.div_no_nan(
        self.true_negatives, self.true_negatives + self.false_positives)
    return self._find_max_under_constraint(
        sensitivities, specificities, math_ops.greater_equal)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration plus num_thresholds and sensitivity.
    config = {
        'num_thresholds': self.num_thresholds,
        'sensitivity': self.sensitivity
    }
    base_config = super(SpecificityAtSensitivity, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class PrecisionAtRecall(SensitivitySpecificityBase):
  # [CLASS DOC]
  # Metric returning the best precision among thresholds whose recall is at least a given value.
  # Stores the recall target and number of thresholds; uses the base class variables.
  """Computes best precision where recall is >= specified value.

  This metric creates four local variables, `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` that are used to compute the
  precision at the given recall. The threshold for the given recall
  value is computed and used to evaluate the corresponding precision.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  If `class_id` is specified, we calculate precision by considering only the
  entries in the batch for which `class_id` is above the threshold predictions,
  and computing the fraction of them for which `class_id` is indeed a correct
  label.

  Args:
    recall: A scalar value in range `[0, 1]`.
    num_thresholds: (Optional) Defaults to 200. The number of thresholds to
      use for matching the given recall.
    class_id: (Optional) Integer class ID for which we want binary metrics.
      This must be in the half-open interval `[0, num_classes)`, where
      `num_classes` is the last dimension of predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.PrecisionAtRecall(0.5)
  >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
  >>> m.result().numpy()
  0.5

  >>> m.reset_state()
  >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
  ...                sample_weight=[2, 2, 2, 1, 1])
  >>> m.result().numpy()
  0.33333333

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
  ```
  """

  def __init__(self,
               recall,
               num_thresholds=200,
               class_id=None,
               name=None,
               dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: recall (scalar in 0 to 1), num_thresholds (default 200), class_id
    # (optional), name, dtype. Raises a value error if recall is outside 0 to 1. Stores recall and
    # num_thresholds, then calls the parent constructor with recall as the constraint value.
    # Returns nothing.
    if recall < 0 or recall > 1:
      raise ValueError('`recall` must be in the range [0, 1].')
    self.recall = recall
    self.num_thresholds = num_thresholds
    super(PrecisionAtRecall, self).__init__(
        value=recall,
        num_thresholds=num_thresholds,
        class_id=class_id,
        name=name,
        dtype=dtype)

  def result(self):
    # [FUNCTION DOC]
    # Computes recall and precision per threshold and returns the largest precision among
    # thresholds whose recall is greater than or equal to the configured value, or 0.0 if none
    # qualify.
    recalls = math_ops.div_no_nan(
        self.true_positives, self.true_positives + self.false_negatives)
    precisions = math_ops.div_no_nan(
        self.true_positives, self.true_positives + self.false_positives)
    return self._find_max_under_constraint(
        recalls, precisions, math_ops.greater_equal)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration plus num_thresholds and recall.
    config = {'num_thresholds': self.num_thresholds, 'recall': self.recall}
    base_config = super(PrecisionAtRecall, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class RecallAtPrecision(SensitivitySpecificityBase):
  # [CLASS DOC]
  # Metric returning the best recall among thresholds whose precision is at least a given value
  # (0.0 if the precision cannot be reached). Stores the precision target and number of
  # thresholds; uses the base class variables.
  """Computes best recall where precision is >= specified value.

  For a given score-label-distribution the required precision might not
  be achievable, in this case 0.0 is returned as recall.

  This metric creates four local variables, `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` that are used to compute the
  recall at the given precision. The threshold for the given precision
  value is computed and used to evaluate the corresponding recall.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  If `class_id` is specified, we calculate precision by considering only the
  entries in the batch for which `class_id` is above the threshold predictions,
  and computing the fraction of them for which `class_id` is indeed a correct
  label.

  Args:
    precision: A scalar value in range `[0, 1]`.
    num_thresholds: (Optional) Defaults to 200. The number of thresholds to
      use for matching the given precision.
    class_id: (Optional) Integer class ID for which we want binary metrics.
      This must be in the half-open interval `[0, num_classes)`, where
      `num_classes` is the last dimension of predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.RecallAtPrecision(0.8)
  >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
  >>> m.result().numpy()
  0.5

  >>> m.reset_state()
  >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
  ...                sample_weight=[1, 0, 0, 1])
  >>> m.result().numpy()
  1.0

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.8)])
  ```
  """

  def __init__(self,
               precision,
               num_thresholds=200,
               class_id=None,
               name=None,
               dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: precision (scalar in 0 to 1), num_thresholds (default 200), class_id
    # (optional), name, dtype. Raises a value error if precision is outside 0 to 1. Stores
    # precision and num_thresholds, then calls the parent constructor with precision as the
    # constraint value. Returns nothing.
    if precision < 0 or precision > 1:
      raise ValueError('`precision` must be in the range [0, 1].')
    self.precision = precision
    self.num_thresholds = num_thresholds
    super(RecallAtPrecision, self).__init__(
        value=precision,
        num_thresholds=num_thresholds,
        class_id=class_id,
        name=name,
        dtype=dtype)

  def result(self):
    # [FUNCTION DOC]
    # Computes precision and recall per threshold and returns the largest recall among thresholds
    # whose precision is greater than or equal to the configured value, or 0.0 if none qualify.
    precisions = math_ops.div_no_nan(
        self.true_positives, self.true_positives + self.false_positives)
    recalls = math_ops.div_no_nan(
        self.true_positives, self.true_positives + self.false_negatives)
    return self._find_max_under_constraint(
        precisions, recalls, math_ops.greater_equal)

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration plus num_thresholds and precision.
    config = {'num_thresholds': self.num_thresholds,
              'precision': self.precision}
    base_config = super(RecallAtPrecision, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class AUC(Metric):
  # [CLASS DOC]
  # Metric approximating the area under the ROC or precision-recall curve with a Riemann sum over
  # a fixed set of thresholds (evenly spaced, or user supplied, plus endpoint thresholds just
  # below zero and above one). State is four variables (true/false positives/negatives) of shape
  # thresholds (by labels for multi-label data); they may be created lazily on the first update
  # when the number of labels is unknown. Supports label weights, multi-label averaging and
  # prediction logits.
  """Approximates the AUC (Area under the curve) of the ROC or PR curves.

  The AUC (Area under the curve) of the ROC (Receiver operating
  characteristic; default) or PR (Precision Recall) curves are quality measures
  of binary classifiers. Unlike the accuracy, and like cross-entropy
  losses, ROC-AUC and PR-AUC evaluate all the operational points of a model.

  This class approximates AUCs using a Riemann sum. During the metric
  accumulation phrase, predictions are accumulated within predefined buckets
  by value. The AUC is then computed by interpolating per-bucket averages. These
  buckets define the evaluated operational points.

  This metric creates four local variables, `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` that are used to compute the AUC.
  To discretize the AUC curve, a linearly spaced set of thresholds is used to
  compute pairs of recall and precision values. The area under the ROC-curve is
  therefore computed using the height of the recall values by the false positive
  rate, while the area under the PR-curve is the computed using the height of
  the precision values by the recall.

  This value is ultimately returned as `auc`, an idempotent operation that
  computes the area under a discretized curve of precision versus recall values
  (computed using the aforementioned variables). The `num_thresholds` variable
  controls the degree of discretization with larger numbers of thresholds more
  closely approximating the true AUC. The quality of the approximation may vary
  dramatically depending on `num_thresholds`. The `thresholds` parameter can be
  used to manually specify thresholds which split the predictions more evenly.

  For a best approximation of the real AUC, `predictions` should be distributed
  approximately uniformly in the range [0, 1] (if `from_logits=False`). The
  quality of the AUC approximation may be poor if this is not the case. Setting
  `summation_method` to 'minoring' or 'majoring' can help quantify the error in
  the approximation by providing lower or upper bound estimate of the AUC.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    num_thresholds: (Optional) Defaults to 200. The number of thresholds to use
      when discretizing the roc curve. Values must be > 1.
    curve: (Optional) Specifies the name of the curve to be computed, 'ROC'
      [default] or 'PR' for the Precision-Recall-curve.
    summation_method: (Optional) Specifies the [Riemann summation method](
        https://en.wikipedia.org/wiki/Riemann_sum) used. 'interpolation'
          (default) applies mid-point summation scheme for `ROC`. For PR-AUC,
          interpolates (true/false) positives but not the ratio that is
          precision (see Davis & Goadrich 2006 for details); 'minoring' applies
          left summation for increasing intervals and right summation for
          decreasing intervals; 'majoring' does the opposite.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    thresholds: (Optional) A list of floating point values to use as the
      thresholds for discretizing the curve. If set, the `num_thresholds`
      parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
      equal to {-epsilon, 1+epsilon} for a small positive epsilon value will be
      automatically included with these to correctly handle predictions equal to
      exactly 0 or 1.
    multi_label: boolean indicating whether multilabel data should be treated as
      such, wherein AUC is computed separately for each label and then averaged
      across labels, or (when False) if the data should be flattened into a
      single label before AUC computation. In the latter case, when multilabel
      data is passed to AUC, each label-prediction pair is treated as an
      individual data point. Should be set to False for multi-class data.
    num_labels: (Optional) The number of labels, used when `multi_label` is
      True. If `num_labels` is not specified, then state variables get created
      on the first call to `update_state`.
    label_weights: (Optional) list, array, or tensor of non-negative weights
      used to compute AUCs for multilabel data. When `multi_label` is True, the
      weights are applied to the individual label AUCs when they are averaged to
      produce the multi-label AUC. When it's False, they are used to weight the
      individual label predictions in computing the confusion matrix on the
      flattened data. Note that this is unlike class_weights in that
      class_weights weights the example depending on the value of its label,
      whereas label_weights depends only on the index of that label before
      flattening; therefore `label_weights` should not be used for multi-class
      data.
    from_logits: boolean indicating whether the predictions (`y_pred` in
      `update_state`) are probabilities or sigmoid logits. As a rule of thumb,
      when using a keras loss, the `from_logits` constructor argument of the
      loss should match the AUC `from_logits` constructor argument.

  Standalone usage:

  >>> m = tf.keras.metrics.AUC(num_thresholds=3)
  >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
  >>> # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
  >>> # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
  >>> # tp_rate = recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
  >>> # auc = ((((1+0.5)/2)*(1-0)) + (((0.5+0)/2)*(0-0))) = 0.75
  >>> m.result().numpy()
  0.75

  >>> m.reset_state()
  >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
  ...                sample_weight=[1, 0, 0, 1])
  >>> m.result().numpy()
  1.0

  Usage with `compile()` API:

  ```python
  # Reports the AUC of a model outputting a probability.
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC()])

  # Reports the AUC of a model outputting a logit.
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.AUC(from_logits=True)])
  ```
  """

  def __init__(self,
               num_thresholds=200,
               curve='ROC',
               summation_method='interpolation',
               name=None,
               dtype=None,
               thresholds=None,
               multi_label=False,
               num_labels=None,
               label_weights=None,
               from_logits=False):
    # [FUNCTION DOC]
    # Constructor. Inputs: num_thresholds (default 200, must exceed one, ignored if thresholds is
    # given), curve (ROC by default, or PR; enum or string), summation_method (interpolation by
    # default, minoring or majoring; enum or string), name, dtype, thresholds (optional list of
    # floats in 0 to 1, sorted internally), multi_label (boolean), num_labels (optional integer,
    # only valid with multi_label), label_weights (optional non-negative weights per label) and
    # from_logits (boolean, apply sigmoid to predictions). Raises a value error for an invalid
    # curve or summation-method enum, for num_thresholds not above one (when no thresholds given),
    # when num_labels is given without multi_label, and (through an assertion) for negative label
    # weights. Builds the threshold array with endpoints just outside the unit interval. Creates
    # the state variables immediately, unless multi_label is set without num_labels, in which case
    # creation is deferred to the first update. Returns nothing.
    # Validate configurations.
    if isinstance(curve, metrics_utils.AUCCurve) and curve not in list(
        metrics_utils.AUCCurve):
      raise ValueError('Invalid curve: "{}". Valid options are: "{}"'.format(
          curve, list(metrics_utils.AUCCurve)))
    if isinstance(
        summation_method,
        metrics_utils.AUCSummationMethod) and summation_method not in list(
            metrics_utils.AUCSummationMethod):
      raise ValueError(
          'Invalid summation method: "{}". Valid options are: "{}"'.format(
              summation_method, list(metrics_utils.AUCSummationMethod)))

    # Update properties.
    if thresholds is not None:
      # If specified, use the supplied thresholds.
      self.num_thresholds = len(thresholds) + 2
      thresholds = sorted(thresholds)
      self._thresholds_distributed_evenly = (
          metrics_utils.is_evenly_distributed_thresholds(
              np.array([0.0] + thresholds + [1.0])))
    else:
      if num_thresholds <= 1:
        raise ValueError('`num_thresholds` must be > 1.')

      # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
      # (0, 1).
      self.num_thresholds = num_thresholds
      thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                    for i in range(num_thresholds - 2)]
      self._thresholds_distributed_evenly = True

    # Add an endpoint "threshold" below zero and above one for either
    # threshold method to account for floating point imprecisions.
    self._thresholds = np.array([0.0 - backend.epsilon()] + thresholds +
                                [1.0 + backend.epsilon()])

    if isinstance(curve, metrics_utils.AUCCurve):
      self.curve = curve
    else:
      self.curve = metrics_utils.AUCCurve.from_str(curve)
    if isinstance(summation_method, metrics_utils.AUCSummationMethod):
      self.summation_method = summation_method
    else:
      self.summation_method = metrics_utils.AUCSummationMethod.from_str(
          summation_method)
    super(AUC, self).__init__(name=name, dtype=dtype)

    # Handle multilabel arguments.
    self.multi_label = multi_label
    if label_weights is not None:
      label_weights = constant_op.constant(label_weights, dtype=self.dtype)
      checks = [
          check_ops.assert_non_negative(
              label_weights,
              message='All values of `label_weights` must be non-negative.')
      ]
      with ops.control_dependencies(checks):
        self.label_weights = label_weights

    else:
      self.label_weights = None

    self._from_logits = from_logits

    self._built = False
    if self.multi_label:
      if num_labels:
        shape = tensor_shape.TensorShape([None, num_labels])
        self._build(shape)
    else:
      if num_labels:
        raise ValueError(
            '`num_labels` is needed only when `multi_label` is True.')
      self._build(None)

  @property
  def thresholds(self):
    # [FUNCTION DOC]
    # Read-only property returning the thresholds, including the two endpoint thresholds, as a
    # list.
    """The thresholds used for evaluating AUC."""
    return list(self._thresholds)

  def _build(self, shape):
    # [FUNCTION DOC]
    # HELPER: Creates the four state variables for the AUC metric.
    # Input: shape of the data (a tensor shape; nothing is needed for the single-label case). For
    # multi-label data the shape must have rank two, otherwise a value error is raised; the number
    # of labels is taken from its second dimension and the variables have shape thresholds by
    # labels, else just thresholds. Creates zero-initialized true/false positives/negatives and
    # marks the metric as built. For multi-label metrics in graph (non-eager) mode the backend
    # variables are initialized in the initialization scope. Returns nothing.
    """Initialize TP, FP, TN, and FN tensors, given the shape of the data."""
    if self.multi_label:
      if shape.ndims != 2:
        raise ValueError('`y_true` must have rank=2 when `multi_label` is '
                         'True. Found rank %s.' % shape.ndims)
      self._num_labels = shape[1]
      variable_shape = tensor_shape.TensorShape(
          [tensor_shape.Dimension(self.num_thresholds), self._num_labels])

    else:
      variable_shape = tensor_shape.TensorShape(
          [tensor_shape.Dimension(self.num_thresholds)])
    self._build_input_shape = shape
    # Create metric variables
    self.true_positives = self.add_weight(
        'true_positives',
        shape=variable_shape,
        initializer=init_ops.zeros_initializer)
    self.true_negatives = self.add_weight(
        'true_negatives',
        shape=variable_shape,
        initializer=init_ops.zeros_initializer)
    self.false_positives = self.add_weight(
        'false_positives',
        shape=variable_shape,
        initializer=init_ops.zeros_initializer)
    self.false_negatives = self.add_weight(
        'false_negatives',
        shape=variable_shape,
        initializer=init_ops.zeros_initializer)

    if self.multi_label:
      with ops.init_scope():
        # This should only be necessary for handling v1 behavior. In v2, AUC
        # should be initialized outside of any tf.functions, and therefore in
        # eager mode.
        if not context.executing_eagerly():
          backend._initialize_variables(backend._get_session())  # pylint: disable=protected-access

    self._built = True

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # Accumulates confusion-matrix counts for every threshold. Inputs: y_true, y_pred (probabilities,
    # or logits when from_logits is set; logits are passed through a sigmoid) and optional
    # sample_weight. Builds the variables from the prediction shape if not yet built. For
    # multi-label data or when label weights exist, asserts the label and variable shapes are
    # consistent (an assertion error with a message about the number of labels otherwise). Label
    # weights are forwarded to the update utility only for single-label data; for multi-label data
    # they are applied later when averaging in the result. Returns the update op.
    """Accumulates confusion matrix statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    deps = []
    if not self._built:
      self._build(tensor_shape.TensorShape(y_pred.shape))

    if self.multi_label or (self.label_weights is not None):
      # y_true should have shape (number of examples, number of labels).
      shapes = [
          (y_true, ('N', 'L'))
      ]
      if self.multi_label:
        # TP, TN, FP, and FN should all have shape
        # (number of thresholds, number of labels).
        shapes.extend([(self.true_positives, ('T', 'L')),
                       (self.true_negatives, ('T', 'L')),
                       (self.false_positives, ('T', 'L')),
                       (self.false_negatives, ('T', 'L'))])
      if self.label_weights is not None:
        # label_weights should be of length equal to the number of labels.
        shapes.append((self.label_weights, ('L',)))
      deps = [
          check_ops.assert_shapes(
              shapes, message='Number of labels is not consistent.')
      ]

    # Only forward label_weights to update_confusion_matrix_variables when
    # multi_label is False. Otherwise the averaging of individual label AUCs is
    # handled in AUC.result
    label_weights = None if self.multi_label else self.label_weights

    if self._from_logits:
      y_pred = activations.sigmoid(y_pred)

    with ops.control_dependencies(deps):
      return metrics_utils.update_confusion_matrix_variables(
          {
              metrics_utils.ConfusionMatrix.TRUE_POSITIVES:
                  self.true_positives,
              metrics_utils.ConfusionMatrix.TRUE_NEGATIVES:
                  self.true_negatives,
              metrics_utils.ConfusionMatrix.FALSE_POSITIVES:
                  self.false_positives,
              metrics_utils.ConfusionMatrix.FALSE_NEGATIVES:
                  self.false_negatives,
          },
          y_true,
          y_pred,
          self._thresholds,
          thresholds_distributed_evenly=self._thresholds_distributed_evenly,
          sample_weight=sample_weight,
          multi_label=self.multi_label,
          label_weights=label_weights)

  def interpolate_pr_auc(self):
    # [FUNCTION DOC]
    # Computes the area under the precision-recall curve using the closed-form interpolation of
    # Davis and Goadrich: per interval between successive thresholds it derives the precision
    # slope and intercept from the true-positive and predicted-positive counts, integrates with a
    # logarithmic term (guarding against zero or non-positive counts) and normalizes by the total
    # positives. For multi-label data it sums per label and returns the plain mean over labels, or
    # the label-weighted mean when label weights are set. Otherwise returns the sum of all
    # increments. No inputs; returns a scalar tensor.
    """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

    https://www.biostat.wisc.edu/~page/rocpr.pdf

    Note here we derive & use a closed formula not present in the paper
    as follows:

      Precision = TP / (TP + FP) = TP / P

    Modeling all of TP (true positive), FP (false positive) and their sum
    P = TP + FP (predicted positive) as varying linearly within each interval
    [A, B] between successive thresholds, we get

      Precision slope = dTP / dP
                      = (TP_B - TP_A) / (P_B - P_A)
                      = (TP - TP_A) / (P - P_A)
      Precision = (TP_A + slope * (P - P_A)) / P

    The area within the interval is (slope / total_pos_weight) times

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

    Returns:
      pr_auc: an approximation of the area under the P-R curve.
    """
    dtp = self.true_positives[:self.num_thresholds -
                              1] - self.true_positives[1:]
    p = self.true_positives + self.false_positives
    dp = p[:self.num_thresholds - 1] - p[1:]
    prec_slope = math_ops.div_no_nan(
        dtp, math_ops.maximum(dp, 0), name='prec_slope')
    intercept = self.true_positives[1:] - math_ops.multiply(prec_slope, p[1:])

    safe_p_ratio = array_ops.where(
        math_ops.logical_and(p[:self.num_thresholds - 1] > 0, p[1:] > 0),
        math_ops.div_no_nan(
            p[:self.num_thresholds - 1],
            math_ops.maximum(p[1:], 0),
            name='recall_relative_ratio'),
        array_ops.ones_like(p[1:]))

    pr_auc_increment = math_ops.div_no_nan(
        prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)),
        math_ops.maximum(self.true_positives[1:] + self.false_negatives[1:], 0),
        name='pr_auc_increment')

    if self.multi_label:
      by_label_auc = math_ops.reduce_sum(
          pr_auc_increment, name=self.name + '_by_label', axis=0)
      if self.label_weights is None:
        # Evenly weighted average of the label AUCs.
        return math_ops.reduce_mean(by_label_auc, name=self.name)
      else:
        # Weighted average of the label AUCs.
        return math_ops.div_no_nan(
            math_ops.reduce_sum(
                math_ops.multiply(by_label_auc, self.label_weights)),
            math_ops.reduce_sum(self.label_weights),
            name=self.name)
    else:
      return math_ops.reduce_sum(pr_auc_increment, name='interpolate_pr_auc')

  def result(self):
    # [FUNCTION DOC]
    # Computes the AUC. For the PR curve with interpolation summation it delegates to the PR
    # interpolation routine. Otherwise it computes recall plus either the false positive rate
    # (ROC) or precision (PR) as the x/y coordinates, picks rectangle heights by summation method
    # (average of neighbours for interpolation, minimum for minoring, maximum for majoring) and
    # sums height times x-step over adjacent thresholds. For multi-label data the per-label sums
    # are averaged, using label weights if given. Returns a scalar tensor; no side effects.
    if (self.curve == metrics_utils.AUCCurve.PR and
        self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION
       ):
      # This use case is different and is handled separately.
      return self.interpolate_pr_auc()

    # Set `x` and `y` values for the curves based on `curve` config.
    recall = math_ops.div_no_nan(self.true_positives,
                                 self.true_positives + self.false_negatives)
    if self.curve == metrics_utils.AUCCurve.ROC:
      fp_rate = math_ops.div_no_nan(self.false_positives,
                                    self.false_positives + self.true_negatives)
      x = fp_rate
      y = recall
    else:  # curve == 'PR'.
      precision = math_ops.div_no_nan(
          self.true_positives, self.true_positives + self.false_positives)
      x = recall
      y = precision

    # Find the rectangle heights based on `summation_method`.
    if self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION:
      # Note: the case ('PR', 'interpolation') has been handled above.
      heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.
    elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
      heights = math_ops.minimum(y[:self.num_thresholds - 1], y[1:])
    else:  # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
      heights = math_ops.maximum(y[:self.num_thresholds - 1], y[1:])

    # Sum up the areas of all the rectangles.
    if self.multi_label:
      riemann_terms = math_ops.multiply(x[:self.num_thresholds - 1] - x[1:],
                                        heights)
      by_label_auc = math_ops.reduce_sum(
          riemann_terms, name=self.name + '_by_label', axis=0)

      if self.label_weights is None:
        # Unweighted average of the label AUCs.
        return math_ops.reduce_mean(by_label_auc, name=self.name)
      else:
        # Weighted average of the label AUCs.
        return math_ops.div_no_nan(
            math_ops.reduce_sum(
                math_ops.multiply(by_label_auc, self.label_weights)),
            math_ops.reduce_sum(self.label_weights),
            name=self.name)
    else:
      return math_ops.reduce_sum(
          math_ops.multiply(x[:self.num_thresholds - 1] - x[1:], heights),
          name=self.name)

  def reset_state(self):
    # [FUNCTION DOC]
    # Sets all four state variables to zeros (thresholds by labels for multi-label data, else
    # thresholds). Does nothing if the variables have not been created yet.
    if self._built:
      confusion_matrix_variables = (self.true_positives, self.true_negatives,
                                    self.false_positives, self.false_negatives)
      if self.multi_label:
        backend.batch_set_value(
            [(v, np.zeros((self.num_thresholds, self._num_labels)))
             for v in confusion_matrix_variables])
      else:
        backend.batch_set_value([(v, np.zeros((self.num_thresholds,)))
                                 for v in confusion_matrix_variables])

  def get_config(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the configuration omits from_logits and num_labels, so a metric rebuilt from it
    # loses the logits setting and, for multi-label data, builds its variables lazily.
    # Returns the base configuration plus number of thresholds, curve and summation-method values,
    # the user thresholds (endpoint thresholds removed), multi_label and label weights (evaluated
    # to values if tensors).
    if is_tensor_or_variable(self.label_weights):
      label_weights = backend.eval(self.label_weights)
    else:
      label_weights = self.label_weights
    config = {
        'num_thresholds': self.num_thresholds,
        'curve': self.curve.value,
        'summation_method': self.summation_method.value,
        # We remove the endpoint thresholds as an inverse of how the thresholds
        # were initialized. This ensures that a metric initialized from this
        # config has the same thresholds.
        'thresholds': self.thresholds[1:-1],
        'multi_label': self.multi_label,
        'label_weights': label_weights
    }
    base_config = super(AUC, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class CosineSimilarity(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean cosine similarity between labels and predictions along a
  # given axis; mean wrapper around the module's cosine similarity function.
  """Computes the cosine similarity between the labels and predictions.

  `cosine similarity = (a . b) / ||a|| ||b||`

  See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

  This metric keeps the average cosine similarity between `predictions` and
  `labels` over a stream of data.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    axis: (Optional) Defaults to -1. The dimension along which the cosine
      similarity is computed.

  Standalone usage:

  >>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
  >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
  >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
  >>> # result = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
  >>> #        = ((0. + 0.) +  (0.5 + 0.5)) / 2
  >>> m = tf.keras.metrics.CosineSimilarity(axis=1)
  >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
  >>> m.result().numpy()
  0.49999997

  >>> m.reset_state()
  >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]],
  ...                sample_weight=[0.3, 0.7])
  >>> m.result().numpy()
  0.6999999

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
  ```
  """

  def __init__(self, name='cosine_similarity', dtype=None, axis=-1):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "cosine_similarity"), dtype, and axis (integer,
    # default last axis) along which similarity is computed. Returns nothing.
    super(CosineSimilarity, self).__init__(
        cosine_similarity, name, dtype=dtype, axis=axis)


class MeanAbsoluteError(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean absolute error between labels and predictions; mean wrapper
  # around the mean absolute error loss function.
  """Computes the mean absolute error between the labels and predictions.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.MeanAbsoluteError()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
  >>> m.result().numpy()
  0.25

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  0.5

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanAbsoluteError()])
  ```
  """

  def __init__(self, name='mean_absolute_error', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "mean_absolute_error") and dtype. Returns
    # nothing.
    super(MeanAbsoluteError, self).__init__(
        mean_absolute_error, name, dtype=dtype)


class MeanAbsolutePercentageError(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean absolute percentage error between labels and predictions;
  # mean wrapper around the corresponding loss function.
  """Computes the mean absolute percentage error between `y_true` and `y_pred`.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.MeanAbsolutePercentageError()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
  >>> m.result().numpy()
  250000000.0

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  500000000.0

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
  ```
  """

  def __init__(self, name='mean_absolute_percentage_error', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "mean_absolute_percentage_error") and dtype.
    # Returns nothing.
    super(MeanAbsolutePercentageError, self).__init__(
        mean_absolute_percentage_error, name, dtype=dtype)


class MeanSquaredError(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean squared error between labels and predictions; mean wrapper
  # around the mean squared error loss function.
  """Computes the mean squared error between `y_true` and `y_pred`.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.MeanSquaredError()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
  >>> m.result().numpy()
  0.25

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  0.5

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanSquaredError()])
  ```
  """

  def __init__(self, name='mean_squared_error', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "mean_squared_error") and dtype. Returns
    # nothing.
    super(MeanSquaredError, self).__init__(
        mean_squared_error, name, dtype=dtype)


class MeanSquaredLogarithmicError(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean squared logarithmic error between labels and predictions;
  # mean wrapper around the corresponding loss function.
  """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.MeanSquaredLogarithmicError()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
  >>> m.result().numpy()
  0.12011322

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  0.24022643

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])
  ```
  """

  def __init__(self, name='mean_squared_logarithmic_error', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "mean_squared_logarithmic_error") and dtype.
    # Returns nothing.
    super(MeanSquaredLogarithmicError, self).__init__(
        mean_squared_logarithmic_error, name, dtype=dtype)


class Hinge(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean hinge loss between labels (minus one or one; zero/one labels
  # are converted) and predictions; mean wrapper around the hinge loss function.
  """Computes the hinge metric between `y_true` and `y_pred`.

  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
  provided we will convert them to -1 or 1.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.Hinge()
  >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
  >>> m.result().numpy()
  1.3

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  1.1

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss='mse', metrics=[tf.keras.metrics.Hinge()])
  ```
  """

  def __init__(self, name='hinge', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "hinge") and dtype. Returns nothing.
    super(Hinge, self).__init__(hinge, name, dtype=dtype)


class SquaredHinge(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean squared hinge loss between labels and predictions; mean
  # wrapper around the squared hinge loss function.
  """Computes the squared hinge metric between `y_true` and `y_pred`.

  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
  provided we will convert them to -1 or 1.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.SquaredHinge()
  >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
  >>> m.result().numpy()
  1.86

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  1.46

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SquaredHinge()])
  ```
  """

  def __init__(self, name='squared_hinge', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "squared_hinge") and dtype. Returns nothing.
    super(SquaredHinge, self).__init__(squared_hinge, name, dtype=dtype)


class CategoricalHinge(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean categorical hinge loss between labels and predictions; mean
  # wrapper around the categorical hinge loss function.
  """Computes the categorical hinge metric between `y_true` and `y_pred`.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.CategoricalHinge()
  >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
  >>> m.result().numpy()
  1.4000001

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  1.2

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.CategoricalHinge()])
  ```
  """

  def __init__(self, name='categorical_hinge', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "categorical_hinge") and dtype. Returns
    # nothing.
    super(CategoricalHinge, self).__init__(categorical_hinge, name, dtype=dtype)


class RootMeanSquaredError(Mean):
  # [CLASS DOC]
  # Metric reporting the root of the running mean squared error. Accumulates squared differences
  # using the Mean total/count variables and takes the square root in the result.
  """Computes root mean squared error metric between `y_true` and `y_pred`.

  Standalone usage:

  >>> m = tf.keras.metrics.RootMeanSquaredError()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
  >>> m.result().numpy()
  0.5

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  0.70710677

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.RootMeanSquaredError()])
  ```
  """

  def __init__(self, name='root_mean_squared_error', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "root_mean_squared_error") and dtype. Returns
    # nothing.
    super(RootMeanSquaredError, self).__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # Accumulates squared errors. Inputs: y_true, y_pred (cast to the metric dtype, dimensions
    # aligned) and optional sample_weight. Computes elementwise squared differences and passes
    # them to the parent mean update. Returns the update op.
    """Accumulates root mean squared error statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
        y_pred, y_true)
    error_sq = math_ops.squared_difference(y_pred, y_true)
    return super(RootMeanSquaredError, self).update_state(
        error_sq, sample_weight=sample_weight)

  def result(self):
    # [FUNCTION DOC]
    # Returns the square root of total divided by count (zero when count is zero).
    return math_ops.sqrt(math_ops.div_no_nan(self.total, self.count))


class LogCoshError(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean of the log of the hyperbolic cosine of the prediction
  # error; mean wrapper around the log-cosh loss function.
  """Computes the logarithm of the hyperbolic cosine of the prediction error.

  `logcosh = log((exp(x) + exp(-x))/2)`, where x is the error (y_pred - y_true)

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.LogCoshError()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
  >>> m.result().numpy()
  0.10844523

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  0.21689045

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.LogCoshError()])
  ```
  """

  def __init__(self, name='logcosh', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "logcosh") and dtype. Returns nothing.
    super(LogCoshError, self).__init__(logcosh, name, dtype=dtype)


class Poisson(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean Poisson loss between labels and predictions; mean wrapper
  # around the Poisson loss function.
  """Computes the Poisson metric between `y_true` and `y_pred`.

  `metric = y_pred - y_true * log(y_pred)`

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.Poisson()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
  >>> m.result().numpy()
  0.49999997

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  0.99999994

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.Poisson()])
  ```
  """

  def __init__(self, name='poisson', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "poisson") and dtype. Returns nothing.
    super(Poisson, self).__init__(poisson, name, dtype=dtype)


class KLDivergence(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean Kullback-Leibler divergence between labels and predictions;
  # mean wrapper around the KL divergence loss function.
  """Computes Kullback-Leibler divergence metric between `y_true` and `y_pred`.

  `metric = y_true * log(y_true / y_pred)`

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.KLDivergence()
  >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
  >>> m.result().numpy()
  0.45814306

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  0.9162892

  Usage with `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss='mse',
                metrics=[tf.keras.metrics.KLDivergence()])
  ```
  """

  def __init__(self, name='kullback_leibler_divergence', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "kullback_leibler_divergence") and dtype.
    # Returns nothing.
    super(KLDivergence, self).__init__(
        kullback_leibler_divergence, name, dtype=dtype)


class MeanIoU(Metric):
  # [CLASS DOC]
  # Metric computing mean intersection-over-union for semantic segmentation. Accumulates a
  # num_classes by num_classes confusion matrix (weighted by sample weights) in one variable and
  # derives per-class IOU (true positives over row sum plus column sum minus true positives),
  # averaged over classes that actually appear.
  """Computes the mean Intersection-Over-Union metric.

  Mean Intersection-Over-Union is a common evaluation metric for semantic image
  segmentation, which first computes the IOU for each semantic class and then
  computes the average over classes. IOU is defined as follows:
    IOU = true_positive / (true_positive + false_positive + false_negative).
  The predictions are accumulated in a confusion matrix, weighted by
  `sample_weight` and the metric is then calculated from it.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    num_classes: The possible number of labels the prediction task can have.
      This value must be provided, since a confusion matrix of dimension =
      [num_classes, num_classes] will be allocated.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> # cm = [[1, 1],
  >>> #        [1, 1]]
  >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
  >>> # iou = true_positives / (sum_row + sum_col - true_positives))
  >>> # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
  >>> m = tf.keras.metrics.MeanIoU(num_classes=2)
  >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
  >>> m.result().numpy()
  0.33333334

  >>> m.reset_state()
  >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
  ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
  >>> m.result().numpy()
  0.23809525

  Usage with `compile()` API:

  ```python
  model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
  ```
  """

  def __init__(self, num_classes, name=None, dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: num_classes (integer number of possible labels; required), optional
    # name and dtype. Stores num_classes and creates a zero-initialized square confusion-matrix
    # variable of that size. Returns nothing.
    super(MeanIoU, self).__init__(name=name, dtype=dtype)
    self.num_classes = num_classes

    # Variable to accumulate the predictions in the confusion matrix.
    self.total_cm = self.add_weight(
        'total_confusion_matrix',
        shape=(num_classes, num_classes),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # Adds the confusion matrix of the batch to the accumulated one. Inputs: y_true and y_pred
    # (integer class labels, cast to the metric dtype and flattened if their rank exceeds one) and
    # optional sample_weight (cast and flattened likewise). Returns the update op of the variable.
    """Accumulates the confusion matrix statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """

    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)

    # Flatten the input if its rank > 1.
    if y_pred.shape.ndims > 1:
      y_pred = array_ops.reshape(y_pred, [-1])

    if y_true.shape.ndims > 1:
      y_true = array_ops.reshape(y_true, [-1])

    if sample_weight is not None:
      sample_weight = math_ops.cast(sample_weight, self._dtype)
      if sample_weight.shape.ndims > 1:
        sample_weight = array_ops.reshape(sample_weight, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = confusion_matrix.confusion_matrix(
        y_true,
        y_pred,
        self.num_classes,
        weights=sample_weight,
        dtype=self._dtype)
    return self.total_cm.assign_add(current_cm)

  def result(self):
    # [FUNCTION DOC]
    # Computes row sums, column sums and the diagonal of the confusion matrix, derives per-class
    # IOU (zero where the denominator is zero), and returns the sum of IOUs divided by the number
    # of classes with a non-zero denominator (zero if there are none). Scalar tensor result.
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = math_ops.cast(
        math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
    sum_over_col = math_ops.cast(
        math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
    true_positives = math_ops.cast(
        array_ops.tensor_diag_part(self.total_cm), dtype=self._dtype)

    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = math_ops.reduce_sum(
        math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype))

    iou = math_ops.div_no_nan(true_positives, denominator)

    return math_ops.div_no_nan(
        math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)

  def reset_state(self):
    # [FUNCTION DOC]
    # Sets the confusion-matrix variable to a zero matrix. Returns nothing.
    backend.set_value(
        self.total_cm, np.zeros((self.num_classes, self.num_classes)))

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the base configuration plus num_classes.
    config = {'num_classes': self.num_classes}
    base_config = super(MeanIoU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MeanTensor(Metric):
  # [CLASS DOC]
  # Metric computing the elementwise (weighted) mean of tensors of a fixed shape, keeping
  # same-shaped "total" and "count" variables. The shape may be given at construction or inferred
  # from the first update, in which case the variables are created lazily.
  """Computes the element-wise (weighted) mean of the given tensors.

  `MeanTensor` returns a tensor with the same shape of the input tensors. The
  mean value is updated by keeping local variables `total` and `count`. The
  `total` tracks the sum of the weighted values, and `count` stores the sum of
  the weighted counts.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    shape: (Optional) A list of integers, a tuple of integers, or a 1-D Tensor
      of type int32. If not specified, the shape is inferred from the values at
      the first call of update_state.

  Standalone usage:

  >>> m = tf.keras.metrics.MeanTensor()
  >>> m.update_state([0, 1, 2, 3])
  >>> m.update_state([4, 5, 6, 7])
  >>> m.result().numpy()
  array([2., 3., 4., 5.], dtype=float32)

  >>> m.update_state([12, 10, 8, 6], sample_weight= [0, 0.2, 0.5, 1])
  >>> m.result().numpy()
  array([2.       , 3.6363635, 4.8      , 5.3333335], dtype=float32)

  >>> m = tf.keras.metrics.MeanTensor(dtype=tf.float64, shape=(1, 4))
  >>> m.result().numpy()
  array([[0., 0., 0., 0.]])
  >>> m.update_state([[0, 1, 2, 3]])
  >>> m.update_state([[4, 5, 6, 7]])
  >>> m.result().numpy()
  array([[2., 3., 4., 5.]])
  """

  def __init__(self, name='mean_tensor', dtype=None, shape=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "mean_tensor"), dtype, and shape (optional list
    # or tuple of integers, or integer vector tensor). If a shape is given the state variables are
    # created immediately; otherwise they are created on the first update. Returns nothing.
    super(MeanTensor, self).__init__(name=name, dtype=dtype)
    self._shape = None
    self._total = None
    self._count = None
    self._built = False
    if shape is not None:
      self._build(shape)

  def _build(self, shape):
    # [FUNCTION DOC]
    # HELPER: Lazily creates the state variables of the tensor-mean metric.
    # Input: shape of the values. Stores it, creates zero-initialized "total" and "count"
    # variables of that shape and, in non-eager (graph) mode, initializes backend variables in the
    # init scope. Marks the metric built. Returns nothing.
    self._shape = tensor_shape.TensorShape(shape)
    self._build_input_shape = self._shape
    # Create new state variables
    self._total = self.add_weight(
        'total', shape=shape, initializer=init_ops.zeros_initializer)
    self._count = self.add_weight(
        'count', shape=shape, initializer=init_ops.zeros_initializer)
    with ops.init_scope():
      if not context.executing_eagerly():
        backend._initialize_variables(backend._get_session())  # pylint: disable=protected-access
    self._built = True

  @property
  def total(self):
    # [FUNCTION DOC]
    # Read-only property returning the total variable, or nothing if variables are not yet built.
    return self._total if self._built else None

  @property
  def count(self):
    # [FUNCTION DOC]
    # Read-only property returning the count variable, or nothing if variables are not yet built.
    return self._count if self._built else None

  def update_state(self, values, sample_weight=None):
    # [FUNCTION DOC]
    # Accumulates a tensor into the elementwise total and count. Inputs: values (cast to the
    # metric dtype) and optional sample_weight. On first use builds the variables from the values
    # shape; later values must have exactly that shape or a value error is raised. Weights are
    # aligned and broadcast to the values (if broadcasting fails, values are averaged down to the
    # weight rank); count increases by the weights (or ones) and total by weighted values. Returns
    # the count update op.
    """Accumulates statistics for computing the element-wise mean.

    Args:
      values: Per-example value.
      sample_weight: Optional weighting of each example. Defaults to 1.

    Returns:
      Update op.
    """
    values = math_ops.cast(values, self._dtype)
    if not self._built:
      self._build(values.shape)
    elif values.shape != self._shape:
      raise ValueError('MeanTensor input values must always have the same '
                       'shape. Expected shape (set during the first call): {}. '
                       'Got: {}'.format(self._shape, values.shape))

    num_values = array_ops.ones_like(values)
    if sample_weight is not None:
      sample_weight = math_ops.cast(sample_weight, self._dtype)

      # Update dimensions of weights to match with values if possible.
      values, _, sample_weight = losses_utils.squeeze_or_expand_dimensions(
          values, sample_weight=sample_weight)
      try:
        # Broadcast weights if possible.
        sample_weight = weights_broadcast_ops.broadcast_weights(
            sample_weight, values)
      except ValueError:
        # Reduce values to same ndim as weight array
        ndim = backend.ndim(values)
        weight_ndim = backend.ndim(sample_weight)
        values = math_ops.reduce_mean(
            values, axis=list(range(weight_ndim, ndim)))

      num_values = math_ops.multiply(num_values, sample_weight)
      values = math_ops.multiply(values, sample_weight)

    update_total_op = self._total.assign_add(values)
    with ops.control_dependencies([update_total_op]):
      return self._count.assign_add(num_values)

  def result(self):
    # [FUNCTION DOC]
    # Returns total divided by count elementwise (zero where count is zero). Raises a value error
    # if no values have been seen and no shape was given (variables not built).
    if not self._built:
      raise ValueError(
          'MeanTensor does not have any result yet. Please call the MeanTensor '
          'instance or use `.update_state(value)` before retrieving the result.'
          )
    return math_ops.div_no_nan(self.total, self.count)

  def reset_state(self):
    # [FUNCTION DOC]
    # Sets all variables to zeros of the stored shape. Does nothing if not yet built.
    if self._built:
      backend.batch_set_value(
          [(v, np.zeros(self._shape.as_list())) for v in self.variables])


class BinaryCrossentropy(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean binary crossentropy between labels and predictions; mean
  # wrapper around the binary crossentropy loss function with logits and label smoothing options.
  """Computes the crossentropy metric between the labels and predictions.

  This is the crossentropy metric class to be used when there are only two
  label classes (0 and 1).

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    from_logits: (Optional )Whether output is expected to be a logits tensor.
      By default, we consider that output encodes a probability distribution.
    label_smoothing: (Optional) Float in [0, 1]. When > 0, label values are
      smoothed, meaning the confidence on label values are relaxed.
      e.g. `label_smoothing=0.2` means that we will use a value of `0.1` for
      label `0` and `0.9` for label `1`".

  Standalone usage:

  >>> m = tf.keras.metrics.BinaryCrossentropy()
  >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
  >>> m.result().numpy()
  0.81492424

  >>> m.reset_state()
  >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
  ...                sample_weight=[1, 0])
  >>> m.result().numpy()
  0.9162905

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.BinaryCrossentropy()])
  ```
  """

  def __init__(self,
               name='binary_crossentropy',
               dtype=None,
               from_logits=False,
               label_smoothing=0):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "binary_crossentropy"), dtype, from_logits
    # (boolean, default false: predictions are probabilities) and label_smoothing (float in 0 to
    # 1, default 0) forwarded to the loss function. Returns nothing.
    super(BinaryCrossentropy, self).__init__(
        binary_crossentropy,
        name,
        dtype=dtype,
        from_logits=from_logits,
        label_smoothing=label_smoothing)


class CategoricalCrossentropy(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean categorical crossentropy between one-hot labels and
  # predictions; mean wrapper around the categorical crossentropy loss function.
  """Computes the crossentropy metric between the labels and predictions.

  This is the crossentropy metric class to be used when there are multiple
  label classes (2 or more). Here we assume that labels are given as a `one_hot`
  representation. eg., When labels values are [2, 0, 1],
   `y_true` = [[0, 0, 1], [1, 0, 0], [0, 1, 0]].

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    from_logits: (Optional) Whether output is expected to be a logits tensor.
      By default, we consider that output encodes a probability distribution.
    label_smoothing: (Optional) Float in [0, 1]. When > 0, label values are
      smoothed, meaning the confidence on label values are relaxed. e.g.
      `label_smoothing=0.2` means that we will use a value of `0.1` for label
      `0` and `0.9` for label `1`"

  Standalone usage:

  >>> # EPSILON = 1e-7, y = y_true, y` = y_pred
  >>> # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
  >>> # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
  >>> # xent = -sum(y * log(y'), axis = -1)
  >>> #      = -((log 0.95), (log 0.1))
  >>> #      = [0.051, 2.302]
  >>> # Reduced xent = (0.051 + 2.302) / 2
  >>> m = tf.keras.metrics.CategoricalCrossentropy()
  >>> m.update_state([[0, 1, 0], [0, 0, 1]],
  ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
  >>> m.result().numpy()
  1.1769392

  >>> m.reset_state()
  >>> m.update_state([[0, 1, 0], [0, 0, 1]],
  ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
  ...                sample_weight=tf.constant([0.3, 0.7]))
  >>> m.result().numpy()
  1.6271976

  Usage with `compile()` API:

  ```python
  model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.CategoricalCrossentropy()])
  ```
  """

  def __init__(self,
               name='categorical_crossentropy',
               dtype=None,
               from_logits=False,
               label_smoothing=0):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "categorical_crossentropy"), dtype, from_logits
    # (boolean, default false) and label_smoothing (float in 0 to 1, default 0) forwarded to the
    # loss function. Returns nothing.
    super(CategoricalCrossentropy, self).__init__(
        categorical_crossentropy,
        name,
        dtype=dtype,
        from_logits=from_logits,
        label_smoothing=label_smoothing)


class SparseCategoricalCrossentropy(MeanMetricWrapper):
  # [CLASS DOC]
  # Metric reporting the running mean sparse categorical crossentropy between integer labels and
  # predictions; mean wrapper around the sparse categorical crossentropy loss function.
  """Computes the crossentropy metric between the labels and predictions.

  Use this crossentropy metric when there are two or more label classes.
  We expect labels to be provided as integers. If you want to provide labels
  using `one-hot` representation, please use `CategoricalCrossentropy` metric.
  There should be `# classes` floating point values per feature for `y_pred`
  and a single floating point value per feature for `y_true`.

  In the snippet below, there is a single floating point value per example for
  `y_true` and `# classes` floating pointing values per example for `y_pred`.
  The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
  `[batch_size, num_classes]`.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    from_logits: (Optional) Whether output is expected to be a logits tensor.
      By default, we consider that output encodes a probability distribution.
    axis: (Optional) Defaults to -1. The dimension along which the metric is
      computed.

  Standalone usage:

  >>> # y_true = one_hot(y_true) = [[0, 1, 0], [0, 0, 1]]
  >>> # logits = log(y_pred)
  >>> # softmax = exp(logits) / sum(exp(logits), axis=-1)
  >>> # softmax = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
  >>> # xent = -sum(y * log(softmax), 1)
  >>> # log(softmax) = [[-2.9957, -0.0513, -16.1181],
  >>> #                [-2.3026, -0.2231, -2.3026]]
  >>> # y_true * log(softmax) = [[0, -0.0513, 0], [0, 0, -2.3026]]
  >>> # xent = [0.0513, 2.3026]
  >>> # Reduced xent = (0.0513 + 2.3026) / 2
  >>> m = tf.keras.metrics.SparseCategoricalCrossentropy()
  >>> m.update_state([1, 2],
  ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
  >>> m.result().numpy()
  1.1769392

  >>> m.reset_state()
  >>> m.update_state([1, 2],
  ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
  ...                sample_weight=tf.constant([0.3, 0.7]))
  >>> m.result().numpy()
  1.6271976

  Usage with `compile()` API:

  ```python
  model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])
  ```
  """

  def __init__(self,
               name='sparse_categorical_crossentropy',
               dtype=None,
               from_logits=False,
               axis=-1):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "sparse_categorical_crossentropy"), dtype,
    # from_logits (boolean, default false) and axis (integer, default last axis) forwarded to the
    # loss function. Returns nothing.
    super(SparseCategoricalCrossentropy, self).__init__(
        sparse_categorical_crossentropy,
        name,
        dtype=dtype,
        from_logits=from_logits,
        axis=axis)


class SumOverBatchSize(Reduce):
  # [CLASS DOC]
  # Metric computing the weighted sum of values divided by the number of values seen. A subclass
  # of Reduce fixed to the sum-over-batch-size reduction, with total and count variables.
  """Computes the weighted sum over batch size of the given values.

  For example, if values is [1, 3, 5, 7] then the metric value is 4.
  If the weights were specified as [1, 1, 0, 0] then the value would be 1.

  This metric creates two variables, `total` and `count` that are used to
  compute the average of `values`. This average is ultimately returned as sum
  over batch size which is an idempotent operation that simply divides `total`
  by `count`.

  If `sample_weight` is `None`, weights default to 1.  Use `sample_weight` of 0
  to mask values.
  """

  def __init__(self, name='sum_over_batch_size', dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: optional name (default "sum_over_batch_size") and dtype. Initializes
    # the parent with the sum-over-batch-size reduction. Returns nothing.
    super(SumOverBatchSize, self).__init__(
        reduction=metrics_utils.Reduction.SUM_OVER_BATCH_SIZE,
        name=name,
        dtype=dtype)


class SumOverBatchSizeMetricWrapper(SumOverBatchSize):
  # [CLASS DOC]
  # Adapter that wraps a stateless metric function (labels, predictions, keyword arguments) as a
  # sum-over-batch-size metric. Holds the function and its extra keyword arguments.
  """Wraps a function with the `SumOverBatchSizeMetricWrapper` metric."""

  def __init__(self, fn, name=None, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: fn (metric callable), optional name and dtype, and extra keyword
    # arguments stored and passed to fn on every update. Returns nothing.
    """Creates a `SumOverBatchSizeMetricWrapper` instance.

    Args:
      fn: The metric function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super(SumOverBatchSizeMetricWrapper, self).__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the mean wrapper, ragged inputs are not flattened before the wrapped
    # function is called.
    # Accumulates the wrapped function's output. Inputs: y_true, y_pred (cast to the metric dtype,
    # dimensions aligned) and optional sample_weight. Calls the wrapped function (autograph
    # converted) with the stored keyword arguments and passes the per-sample values and weights to
    # the parent update. Returns the update op.
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
        y_pred, y_true)

    ag_fn = autograph.tf_convert(self._fn, ag_ctx.control_status_ctx())
    matches = ag_fn(y_true, y_pred, **self._fn_kwargs)
    return super(SumOverBatchSizeMetricWrapper, self).update_state(
        matches, sample_weight=sample_weight)

  def get_config(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the wrapped function itself is not included in the configuration (unlike the
    # mean wrapper for its own class), so the metric cannot be rebuilt from it alone.
    # Returns the base configuration plus the stored keyword arguments (tensors or variables are
    # evaluated to concrete values).
    config = {}
    for k, v in self._fn_kwargs.items():
      config[k] = backend.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(SumOverBatchSizeMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def accuracy(y_true, y_pred):
  # [FUNCTION DOC]
  # Elementwise equality of labels and predictions. Inputs: y_true and y_pred (tensors, possibly
  # ragged; flattened if ragged). Asserts the shapes are compatible (raises a shape error
  # otherwise), casts predictions to the label dtype when dtypes differ, and returns a tensor of
  # one/zero values (backend float type) marking where they are equal.
  [y_pred, y_true], _ = \
      metrics_utils.ragged_assert_compatible_and_get_flat_values(
          [y_pred, y_true])
  y_true.shape.assert_is_compatible_with(y_pred.shape)
  if y_true.dtype != y_pred.dtype:
    y_pred = math_ops.cast(y_pred, y_true.dtype)
  return math_ops.cast(math_ops.equal(y_true, y_pred), backend.floatx())


@dispatch.add_dispatch_support
def binary_accuracy(y_true, y_pred, threshold=0.5):
  # [FUNCTION DOC]
  # Per-example binary accuracy. Inputs: y_true (labels), y_pred (values), threshold (float,
  # default 0.5) above which a prediction counts as one. Converts predictions to a tensor, applies
  # the threshold and returns the mean equality with the labels over the last axis (so the output
  # has one fewer dimension). Registered for dispatch support on custom types.
  """Calculates how often predictions match binary labels.

  Standalone usage:
  >>> y_true = [[1], [1], [0], [0]]
  >>> y_pred = [[1], [1], [0], [0]]
  >>> m = tf.keras.metrics.binary_accuracy(y_true, y_pred)
  >>> assert m.shape == (4,)
  >>> m.numpy()
  array([1., 1., 1., 1.], dtype=float32)

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    threshold: (Optional) Float representing the threshold for deciding whether
      prediction values are 1 or 0.

  Returns:
    Binary accuracy values. shape = `[batch_size, d0, .. dN-1]`
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  threshold = math_ops.cast(threshold, y_pred.dtype)
  y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
  return backend.mean(math_ops.equal(y_true, y_pred), axis=-1)


@dispatch.add_dispatch_support
def categorical_accuracy(y_true, y_pred):
  # [FUNCTION DOC]
  # Per-example categorical accuracy. Inputs: one-hot (or probability) labels and predictions
  # (probabilities or logits). Returns one/zero (backend float type) values marking where the
  # arg-max class over the last axis agrees. Registered for dispatch support.
  """Calculates how often predictions match one-hot labels.

  Standalone usage:
  >>> y_true = [[0, 0, 1], [0, 1, 0]]
  >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
  >>> m = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
  >>> assert m.shape == (2,)
  >>> m.numpy()
  array([0., 1.], dtype=float32)

  You can provide logits of classes as `y_pred`, since argmax of
  logits and probabilities are same.

  Args:
    y_true: One-hot ground truth values.
    y_pred: The prediction values.

  Returns:
    Categorical accuracy values.
  """
  return math_ops.cast(
      math_ops.equal(
          math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)),
      backend.floatx())


@dispatch.add_dispatch_support
def sparse_categorical_accuracy(y_true, y_pred):
  # [FUNCTION DOC]
  # Per-example sparse categorical accuracy. Inputs: integer labels and predictions (probabilities
  # or logits over classes). Converts both to tensors, removes a trailing size-one label axis when
  # label and prediction ranks match, takes the arg-max of predictions, casts it to the label
  # dtype if different and returns one/zero (backend float type) values where they agree.
  # Registered for dispatch support.
  """Calculates how often predictions match integer labels.

  Standalone usage:
  >>> y_true = [2, 1]
  >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
  >>> m = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
  >>> assert m.shape == (2,)
  >>> m.numpy()
  array([0., 1.], dtype=float32)

  You can provide logits of classes as `y_pred`, since argmax of
  logits and probabilities are same.

  Args:
    y_true: Integer ground truth values.
    y_pred: The prediction values.

  Returns:
    Sparse categorical accuracy values.
  """
  y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_true)
  y_pred_rank = y_pred.shape.ndims
  y_true_rank = y_true.shape.ndims
  # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
  if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
      backend.int_shape(y_true)) == len(backend.int_shape(y_pred))):
    y_true = array_ops.squeeze(y_true, [-1])
  y_pred = math_ops.argmax(y_pred, axis=-1)

  # If the predicted output and actual output types don't match, force cast them
  # to match.
  if backend.dtype(y_pred) != backend.dtype(y_true):
    y_pred = math_ops.cast(y_pred, backend.dtype(y_true))

  return math_ops.cast(math_ops.equal(y_true, y_pred), backend.floatx())


@dispatch.add_dispatch_support
def top_k_categorical_accuracy(y_true, y_pred, k=5):
  # [FUNCTION DOC]
  # Per-example top-k accuracy with one-hot labels. Inputs: y_true (one-hot labels), y_pred
  # (scores) and k (integer, default 5). Returns one/zero (backend float type) values marking
  # where the label's arg-max class is among the k highest predictions. Registered for dispatch
  # support.
  """Computes how often targets are in the top `K` predictions.

  Standalone usage:
  >>> y_true = [[0, 0, 1], [0, 1, 0]]
  >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
  >>> m = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
  >>> assert m.shape == (2,)
  >>> m.numpy()
  array([1., 1.], dtype=float32)

  Args:
    y_true: The ground truth values.
    y_pred: The prediction values.
    k: (Optional) Number of top elements to look at for computing accuracy.
      Defaults to 5.

  Returns:
    Top K categorical accuracy value.
  """
  return math_ops.cast(
      nn.in_top_k(
          y_pred, math_ops.argmax(y_true, axis=-1), k), backend.floatx())


@dispatch.add_dispatch_support
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
  # [FUNCTION DOC]
  # Per-example top-k accuracy with integer labels. Inputs: y_true (integer labels), y_pred
  # (scores) and k (integer, default 5). When ranks are known, flattens predictions to two
  # dimensions and labels to one. Returns one/zero (backend float type) values marking where the
  # label is among the k highest predictions (labels are cast to 32-bit integers). Registered for
  # dispatch support.
  """Computes how often integer targets are in the top `K` predictions.

  Standalone usage:
  >>> y_true = [2, 1]
  >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
  >>> m = tf.keras.metrics.sparse_top_k_categorical_accuracy(
  ...     y_true, y_pred, k=3)
  >>> assert m.shape == (2,)
  >>> m.numpy()
  array([1., 1.], dtype=float32)

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    k: (Optional) Number of top elements to look at for computing accuracy.
      Defaults to 5.

  Returns:
    Sparse top K categorical accuracy value.
  """
  y_pred_rank = tensor_conversion.convert_to_tensor_v2_with_dispatch(
      y_pred
  ).shape.ndims
  y_true_rank = tensor_conversion.convert_to_tensor_v2_with_dispatch(
      y_true
  ).shape.ndims
  # Flatten y_pred to (batch_size, num_samples) and y_true to (num_samples,)
  if (y_true_rank is not None) and (y_pred_rank is not None):
    if y_pred_rank > 2:
      y_pred = array_ops.reshape(y_pred, [-1, y_pred.shape[-1]])
    if y_true_rank > 1:
      y_true = array_ops.reshape(y_true, [-1])

  return math_ops.cast(
      nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), k), backend.floatx())


def cosine_proximity(y_true, y_pred, axis=-1):
  # [FUNCTION DOC]
  # KNOWN ISSUE: despite the name "proximity", it returns the plain (positive) cosine similarity,
  # not its negation as the corresponding loss does.
  # Per-example cosine similarity. Inputs: y_true and y_pred (tensors) and axis (integer, default
  # last axis). Both are L2-normalized along the axis, multiplied elementwise and summed over the
  # axis, which removes that dimension. Also exported under the cosine similarity alias.
  """Computes the cosine similarity between labels and predictions.

  Args:
    y_true: The ground truth values.
    y_pred: The prediction values.
    axis: (Optional) Defaults to -1. The dimension along which the cosine
      similarity is computed.

  Returns:
    Cosine similarity value.
  """
  y_true = nn.l2_normalize(y_true, axis=axis)
  y_pred = nn.l2_normalize(y_pred, axis=axis)
  return math_ops.reduce_sum(y_true * y_pred, axis=axis)

# Aliases

acc = ACC = accuracy
bce = BCE = binary_crossentropy
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine_similarity = cosine_proximity
log_cosh = logcosh


def clone_metric(metric):
  # [FUNCTION DOC]
  # Creates a fresh copy of a stateful metric. Input: a metric object or any other value. If it is
  # a Metric instance, a new instance of the same class is built from its configuration (inside
  # the init scope so variables are created outside any function graph); otherwise the input is
  # returned unchanged.
  """Returns a clone of the metric if stateful, otherwise returns it as is."""
  if isinstance(metric, Metric):
    with ops.init_scope():
      return metric.__class__.from_config(metric.get_config())
  return metric


def clone_metrics(metrics):
  # [FUNCTION DOC]
  # Applies the single-metric cloning function to every leaf of a nested structure (list, tuple or
  # mapping) of metrics. Input: the structure. Returns a structure of the same shape holding
  # clones (non-Metric leaves are passed through).
  """Clones the given metric list/dict."""
  return nest.map_structure(clone_metric, metrics)


def serialize(metric):
  # [FUNCTION DOC]
  # Converts a metric function or Metric instance into its serializable configuration mapping via
  # the generic Keras object serializer. Input: the metric. Returns the configuration (class name
  # and config for objects, or the function name).
  """Serializes metric function or `Metric` instance.

  Args:
    metric: A Keras `Metric` instance or a metric function.

  Returns:
    Metric configuration dictionary.
  """
  return serialize_keras_object(metric)


def deserialize(config, custom_objects=None):
  # [FUNCTION DOC]
  # Rebuilds a metric from a serialized configuration (mapping or name string). Inputs: config and
  # optional custom_objects (mapping from names to custom classes or functions). Names are looked
  # up among this module's globals first plus the custom objects. Returns a Metric instance or
  # metric function; the generic deserializer raises a value error for unknown names.
  """Deserializes a serialized metric class/function instance.

  Args:
    config: Metric configuration.
    custom_objects: Optional dictionary mapping names (strings) to custom
      objects (classes and functions) to be considered during deserialization.

  Returns:
      A Keras `Metric` instance or a metric function.
  """
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='metric function')


def get(identifier):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the identifier may be None, but None is not callable, so it
  # raises a value error like any other uninterpretable identifier.
  # Resolves an identifier to a metric. Input: a configuration mapping (deserialized), a string
  # name (deserialized, so it returns a function or class looked up by name), or any callable
  # (returned unchanged). Raises a value error for anything else.
  """Retrieves a Keras metric as a `function`/`Metric` class instance.

  The `identifier` may be the string name of a metric function or class.

  >>> metric = tf.keras.metrics.get("categorical_crossentropy")
  >>> type(metric)
  <class 'function'>
  >>> metric = tf.keras.metrics.get("CategoricalCrossentropy")
  >>> type(metric)
  <class '...keras.metrics.CategoricalCrossentropy'>

  You can also specify `config` of the metric to this function by passing dict
  containing `class_name` and `config` as an identifier. Also note that the
  `class_name` must map to a `Metric` class

  >>> identifier = {"class_name": "CategoricalCrossentropy",
  ...               "config": {"from_logits": True}}
  >>> metric = tf.keras.metrics.get(identifier)
  >>> type(metric)
  <class '...keras.metrics.CategoricalCrossentropy'>

  Args:
    identifier: A metric identifier. One of None or string name of a metric
      function/class or metric configuration dictionary or a metric function or
      a metric class instance

  Returns:
    A Keras metric as a `function`/ `Metric` class instance.

  Raises:
    ValueError: If `identifier` cannot be interpreted.
  """
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError(
        'Could not interpret metric function identifier: {}'.format(identifier))


def is_built_in(cls):
  # [FUNCTION DOC]
  # HELPER: Used by the Metric allocator to decide how to wrap update_state.
  # Input: a class. Returns true if the class is defined in the same module as the Metric base
  # class (that is, a built-in metric from this file), otherwise false.
  return cls.__module__ == Metric.__module__
