# [MODULE DOC]
# Distributed value wrappers used by ParameterServerStrategy (and CentralStorageStrategy).
# Defines AggregatingVariable (wraps a variable and aggregates updates across replicas),
# CachingVariable (wraps a variable and caches its read value inside a caching scope),
# PerWorkerVariable and its spec (unsynchronized variables living on each worker),
# DistributedTable and RestoredDistributedTable (static hash tables replicated on workers), and
# a thread-local context used while restoring such tables from a saved model.
# Import-time side effects: registers tensor conversion functions for AggregatingVariable and
# CachingVariable, and installs operator overloads on CachingVariable.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Various classes representing distributed values for PS."""

import contextlib
import copy
import functools
import threading
import weakref

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import numpy_compat


TRACKABLE_RESOURCE_METHODS = [
    "_create_resource", "_initialize", "_destroy_resource"
]


# Variable used in PSStrategy TF 1, TF2 and CentralStorageStrategy.
class AggregatingVariable(resource_variable_ops.BaseResourceVariable,
                          core.Tensor):
  # [CLASS DOC]
  # Wrapper around a single variable that aggregates updates coming from several replicas
  # according to a chosen aggregation method. Holds the strategy, the wrapped variable and the
  # aggregation. Most reads and properties are forwarded to the wrapped variable; assignments
  # are routed through the strategy so they are reduced first. Used by PS and central storage
  # strategies.
  """A wrapper around a variable that aggregates updates across replicas."""

  def __init__(self, strategy, v, aggregation):
    # [FUNCTION DOC]
    # Stores the strategy, the wrapped variable and the aggregation method. Registers a weak
    # reference to this wrapper on the wrapped variable as its aggregating container (not the
    # distributed container, to avoid regroup behavior). Does not call the base constructor.
    # Inputs: strategy, v (variable to wrap), aggregation (aggregation method). Returns nothing.
    self._distribute_strategy = strategy
    self._v = v
    # NOTE: We don't use "_distributed_container" here because we don't want
    # to trigger that code path in regroup().
    v._aggregating_container = weakref.ref(self)  # pylint: disable=protected-access
    self._aggregation = aggregation

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it raises a runtime error for a different strategy; this
    # method raises nothing itself, the check is delegated to entering/asserting the strategy.
    # Deep copies the wrapped variable inside the originating strategy scope and builds a new
    # wrapper with the same strategy and aggregation, recording it in the memo mapping.
    # Input: memo, the deepcopy memoization mapping. Returns the new wrapper.
    """Perform a deepcopy of the `AggregatingVariable`.

    Unlike the deepcopy of a regular tf.Variable, this keeps the original
    strategy and devices of the `AggregatingVariable`.  To avoid confusion
    with the behavior of deepcopy on a regular `Variable` (which does
    copy into new devices), we only allow a deepcopy of a `AggregatingVariable`
    within its originating strategy scope.

    Args:
      memo: The memoization object for `deepcopy`.

    Returns:
      A deep copy of the current `AggregatingVariable`.

    Raises:
      RuntimeError: If trying to deepcopy into a different strategy.
    """
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      v = copy.deepcopy(self._v, memo)

    copied_variable = type(self)(
        strategy=self._distribute_strategy,
        v=v,
        aggregation=self._aggregation)

    memo[id(self)] = copied_variable

    return copied_variable

  def get(self):
    # [FUNCTION DOC]
    # Returns the wrapped variable.
    return self._v

  @property
  def distribute_strategy(self):
    # [FUNCTION DOC]
    # Property returning the distribution strategy this variable belongs to.
    return self._distribute_strategy

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Forwards lookup of any attribute not found on the wrapper to the wrapped variable.
    # Input: name of the attribute. Returns the attribute value.
    return getattr(self._v, name)

  def _assign_func(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Shared implementation of assign, assign_add and assign_sub. Takes the update function from
    # the keyword arguments (key f) and the remaining positional/keyword arguments. Inside the
    # strategy scope: in cross-replica context it applies the function directly when already in an
    # update context, otherwise wraps it in a strategy update call. In replica context it raises
    # a value error if the aggregation is none; otherwise performs a merge call that reduces the
    # value by the aggregation and then applies the update. Returns the update result.
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      f = kwargs.pop("f")
      if distribute_lib.in_cross_replica_context():
        if distribute_lib.get_update_replica_id() is not None:
          # We are calling an assign function in an update context.
          return f(self._v, *args, **kwargs)

        # We are calling an assign function in cross replica context, wrap it in
        # an update call.
        return self._distribute_strategy.extended.update(
            self, f, args=args, kwargs=kwargs)
      else:
        replica_context = distribute_lib.get_replica_context()
        assert replica_context
        # We are calling an assign function in replica context.
        # We reduce the value we want to assign/add/sub. More details about how
        # we handle the different use cases can be found in the _reduce method.
        # We call the function with the reduced value.
        if self._aggregation == vs.VariableAggregation.NONE:
          raise ValueError(
              values_util.aggregation_error_msg.format(
                  variable_type="AggregatingVariable"))

        def merge_fn(strategy,
                     value,
                     use_locking=False,
                     name=None,
                     read_value=True):
          # [FUNCTION DOC]
          # Merge function run in cross-replica context: aggregates the per-replica value with the
          # variable's aggregation, unwraps a per-replica name to its first value, then calls the
          # strategy update with the update function and options (locking, name, read_value).
          # Inputs: strategy, value, use_locking (default false), name (optional), read_value
          # (default true). Returns the update result.
          v = values_util.apply_aggregation(strategy, value, self._aggregation,
                                            self)
          if name and isinstance(name, values.PerReplica):
            name = name.values[0]
          return strategy.extended.update(
              self,
              f,
              args=(v,),
              kwargs={
                  "use_locking": use_locking,
                  "name": name,
                  "read_value": read_value
              })
        return replica_context.merge_call(merge_fn, args=args, kwargs=kwargs)

  def assign_sub(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Subtracts a value from the variable through the aggregating assign path. Arguments are
    # passed through. Returns the update result.
    assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
    return self._assign_func(f=assign_sub_fn, *args, **kwargs)

  def assign_add(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Adds a value to the variable through the aggregating assign path. Arguments are passed
    # through. Returns the update result.
    assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
    return self._assign_func(f=assign_add_fn, *args, **kwargs)

  def assign(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Assigns a value to the variable through the aggregating assign path. Arguments are passed
    # through. Returns the update result.
    assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
    return self._assign_func(f=assign_fn, *args, **kwargs)

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Property: initializer operation of the wrapped variable.
    return self._v.initializer

  def initialized_value(self):
    return self._v.initialized_value()
    # [FUNCTION DOC]
    # Returns the initialized value of the wrapped variable.

  @property
  def initial_value(self):
    # [FUNCTION DOC]
    # Property: initial value of the wrapped variable.
    return self._v.initial_value

  @property
  def op(self) -> ops.Operation:
    # [FUNCTION DOC]
    # Property: the operation of the wrapped variable.
    return self._v.op

  def value(self):
    # [FUNCTION DOC]
    # Returns the current value of the wrapped variable.
    return self._v.value()

  def read_value(self):
    # [FUNCTION DOC]
    # Reads and returns the wrapped variable's value.
    return self._v.read_value()

  def sparse_read(self, indices, name=None):
    # [FUNCTION DOC]
    # Sparse read of the wrapped variable at the given indices; name is optional.
    return self._v.sparse_read(indices, name=name)

  def eval(self, session=None):
    # [FUNCTION DOC]
    # Evaluates the wrapped variable, optionally in the given session. Returns the value.
    return self._v.eval(session)

  @property
  def graph(self):
    # [FUNCTION DOC]
    # Property: graph of the wrapped variable.
    return self._v.graph

  @property
  def device(self):
    # [FUNCTION DOC]
    # Property: device of the wrapped variable.
    return self._v.device

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property: shape of the wrapped variable.
    return self._v.shape

  @property
  def aggregation(self):
    # [FUNCTION DOC]
    # Property: the aggregation method of this wrapper.
    return self._aggregation

  @property
  def synchronization(self):
    # [FUNCTION DOC]
    # Property: synchronization mode of the wrapped variable.
    return self._v.synchronization

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: name of the wrapped variable.
    return self._v.name

  @property
  def trainable(self):
    # [FUNCTION DOC]
    # Property: whether the wrapped variable is trainable.
    return self._v.trainable

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property: data type of the wrapped variable.
    return self._v.dtype

  # TODO(josh11b): Test saving & restoring.
  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # Checkpoint hook. If the wrapped variable is a caching variable, delegates to it; otherwise
    # returns a mapping from the standard variable-value key to the wrapped variable.
    if isinstance(self._v, CachingVariable):
      return self._v._gather_saveables_for_checkpoint()  # pylint:disable=protected-access
    return {trackable.VARIABLE_VALUE_KEY: self._v}

  def _export_to_saved_model_graph(self, object_map, tensor_map,
                                   options, **kwargs):
    # [FUNCTION DOC]
    # Saved-model export hook: delegates to the wrapped variable and then maps this wrapper to the
    # wrapped variable's entry so the saved model looks like one with a normal variable.
    # Inputs: object map, tensor map, options, extra keyword arguments. Returns the resource list.
    """For implementing `Trackable`."""
    # By delegating this method to the wrapped variable, SavedModel with
    # AggregatingVariable are identical to SavedModel with normal variables.
    resource_list = self._v._export_to_saved_model_graph(object_map, tensor_map,  # pylint:disable=protected-access
                                                         options, **kwargs)
    object_map[self] = object_map[self._v]
    return resource_list

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # Copies this variable to the CPU for checkpointing: copies the wrapped variable into the object
    # map, then creates a new wrapper around that copy if none exists yet. Mutates the object map.
    """For implementing `Trackable`."""
    # Create a copy of `self._v` to object_map, then create a new copy of self
    # that wraps the copy of `self._v`.
    # When updating value, only the lowest-level variable will actually do that,
    # the copy of `AggregatingVariable` is more like a shell.
    self._v._copy_trackable_to_cpu(object_map)  # pylint:disable=protected-access
    if self not in object_map:
      # If copy of `self` not populated yet, initialize one.
      object_map[self] = AggregatingVariable(self._distribute_strategy,
                                             object_map[self._v],
                                             self._aggregation)

  # pylint: disable=multiple-statements
  def __add__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: addition.
    return self._v + o

  def __radd__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: reflected addition.
    return o + self._v

  def __sub__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: subtraction.
    return self._v - o

  def __rsub__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: reflected subtraction.
    return o - self._v

  def __mul__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: multiplication.
    return self._v * o

  def __rmul__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: reflected multiplication.
    return o * self._v

  def __truediv__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: true division.
    return self._v / o

  def __rtruediv__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: reflected true division.
    return o / self._v

  def __floordiv__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: floor division.
    return self._v // o

  def __rfloordiv__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: reflected floor division.
    return o // self._v

  def __mod__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: modulo.
    return self._v % o

  def __rmod__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: reflected modulo.
    return o % self._v

  def __lt__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: less than.
    return self._v < o

  def __le__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: less than or equal.
    return self._v <= o

  def __gt__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: greater than.
    return self._v > o

  def __ge__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: greater than or equal.
    return self._v >= o

  def __and__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: bitwise and.
    return self._v & o

  def __rand__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: reflected bitwise and.
    return o & self._v

  def __or__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: bitwise or.
    return self._v | o

  def __ror__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: reflected bitwise or.
    return o | self._v

  def __xor__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: bitwise exclusive or.
    return self._v ^ o

  def __rxor__(self, o):
    # [FUNCTION DOC]
    # Operator forwarding to the wrapped variable: reflected bitwise exclusive or.
    return o ^ self._v

  def __getitem__(self, o):
    # [FUNCTION DOC]
    # Indexing forwarded to the wrapped variable.
    return self._v[o]

  def __pow__(self, o, modulo=None):
    # [FUNCTION DOC]
    # Power operator forwarding to the wrapped variable, with an optional modulo (default none).
    return pow(self._v, o, modulo)

  def __rpow__(self, o):
    # [FUNCTION DOC]
    # Reflected power operator: raises the other operand to the wrapped variable.
    return pow(o, self._v)

  def __invert__(self):
    # [FUNCTION DOC]
    # Bitwise inversion of the wrapped variable.
    return ~self._v

  def __neg__(self):
    # [FUNCTION DOC]
    # Negation of the wrapped variable.
    return -self._v

  def __abs__(self):
    # [FUNCTION DOC]
    # Absolute value of the wrapped variable.
    return abs(self._v)

  def __div__(self, o):
    # [FUNCTION DOC]
    # Legacy division: forwards to the wrapped variable's legacy division; returns the not-implemented
    # marker if the wrapped variable lacks it.
    try:
      return self._v.__div__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rdiv__(self, o):
    # [FUNCTION DOC]
    # Legacy reflected division: forwards to the wrapped variable; returns the not-implemented
    # marker if it lacks the method.
    try:
      return self._v.__rdiv__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __matmul__(self, o):
    # [FUNCTION DOC]
    # Matrix multiplication forwarded to the wrapped variable; returns the not-implemented marker
    # if the wrapped variable lacks the method.
    try:
      return self._v.__matmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rmatmul__(self, o):
    # [FUNCTION DOC]
    # Reflected matrix multiplication forwarded to the wrapped variable; returns the
    # not-implemented marker if it lacks the method.
    try:
      return self._v.__rmatmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __str__(self):
    # [FUNCTION DOC]
    # String form: that of the wrapped variable.
    return str(self._v)

  def __repr__(self):
    # [FUNCTION DOC]
    # Representation: that of the wrapped variable.
    return repr(self._v)

  def _should_act_as_resource_variable(self):
    # [FUNCTION DOC]
    # HELPER: marker so resource-variable checks accept this wrapper.
    # Does nothing; its presence makes the resource-variable type check pass.
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # Converts to a dense tensor by delegating to the wrapped variable.
    # Inputs: dtype (optional), name (optional), as_ref (default false). Returns a tensor.
    return self._v._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


class CachingVariable(resource_variable_ops.BaseResourceVariable, core.Tensor):
  # [CLASS DOC]
  # Wrapper around a variable that caches its read value on the CPU while a caching scope is
  # active. Holds the wrapped variable, the cached tensor and a counter of cache scopes seen so
  # the cache is refreshed when a new scope starts. Writes and most properties are forwarded to
  # the wrapped variable. Operator overloads are installed on the class after its definition.
  """A wrapper around a variable that caches read value locally."""

  def __init__(self, v):
    # [FUNCTION DOC]
    # Stores the wrapped variable, empties the cache and resets the seen-scope counter.
    # Input: v, the variable to wrap. Returns nothing.
    self._v = v
    self._cache = None
    self._current_new_cache_scope_count = 0

  def get(self):
    # [FUNCTION DOC]
    # Returns the wrapped variable.
    return self._v

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Forwards lookup of attributes missing on the wrapper to the wrapped variable.
    return getattr(self._v, name)

  def read_value(self):
    # [FUNCTION DOC]
    # Returns the cached value when inside a caching scope, otherwise reads the wrapped variable.
    if distribute_utils.caching_scope_local.in_caching_scope():
      return self.cached_read_value()
    return self._v.read_value()

  def sparse_read(self, indices, name=None):
    # [FUNCTION DOC]
    # Sparse read of the wrapped variable at the given indices; name is optional.
    return self._v.sparse_read(indices, name=name)

  def cached_read_value(self):
    # [FUNCTION DOC]
    # Returns the locally cached value of the variable. If a newer caching scope has begun than
    # the one seen, the cache is cleared first. On a miss it copies the variable on CPU into the
    # cache. Side effect: updates the cache and scope counter.
    if (distribute_utils.caching_scope_local.new_cache_scope_count >
        self._current_new_cache_scope_count):
      self._current_new_cache_scope_count += 1
      self._cache = None

    with ops.device("CPU:0"):
      if self._cache is not None:
        return self._cache
      else:
        self._cache = array_ops.identity(self._v)
        return self._cache

  def assign_sub(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Subtracts from the wrapped variable; arguments passed through. Does not invalidate the cache.
    return self._v.assign_sub(*args, **kwargs)

  def assign_add(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Adds to the wrapped variable; arguments passed through. Does not invalidate the cache.
    return self._v.assign_add(*args, **kwargs)

  def assign(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Assigns to the wrapped variable; arguments passed through. Does not invalidate the cache.
    return self._v.assign(*args, **kwargs)

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Property: initializer operation of the wrapped variable.
    return self._v.initializer

  def initialized_value(self):
    # [FUNCTION DOC]
    # Returns the initialized value of the wrapped variable.
    return self._v.initialized_value()

  @property
  def initial_value(self):
    # [FUNCTION DOC]
    # Property: initial value of the wrapped variable.
    return self._v.initial_value

  @property
  def op(self) -> ops.Operation:
    # [FUNCTION DOC]
    # Property: the operation of the wrapped variable.
    return self._v.op

  def value(self):
    # [FUNCTION DOC]
    # Returns the cached value inside a caching scope, otherwise the wrapped variable's value.
    if distribute_utils.caching_scope_local.in_caching_scope():
      return self.cached_read_value()
    return self._v.value()

  def eval(self, session=None):
    # [FUNCTION DOC]
    # Evaluates the wrapped variable, optionally in the given session.
    return self._v.eval(session)

  @property
  def graph(self):
    # [FUNCTION DOC]
    # Property: graph of the wrapped variable.
    return self._v.graph

  @property
  def device(self):
    # [FUNCTION DOC]
    # Property: device of the wrapped variable.
    return self._v.device

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property: shape of the wrapped variable.
    return self._v.shape

  @property
  def synchronization(self):
    # [FUNCTION DOC]
    # Property: synchronization mode of the wrapped variable.
    return self._v.synchronization

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: name of the wrapped variable.
    return self._v.name

  @property
  def trainable(self):
    # [FUNCTION DOC]
    # Property: whether the wrapped variable is trainable.
    return self._v.trainable

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property: data type of the wrapped variable.
    return self._v.dtype

  @property
  def constraint(self):
    # [FUNCTION DOC]
    # Property: constraint of the wrapped variable.
    return self._v.constraint

  def __array__(self, dtype=None):
    # [FUNCTION DOC]
    # Converts to a numpy array using the numpy value; dtype is optional.
    return numpy_compat.np_asarray(self.numpy(), dtype=dtype)

  def __complex__(self):
    # [FUNCTION DOC]
    # Converts the current value to a complex number.
    return complex(self.value().numpy())

  def __int__(self):
    # [FUNCTION DOC]
    # Converts the current value to an integer.
    return int(self.value().numpy())

  def __float__(self):
    # [FUNCTION DOC]
    # Converts the current value to a floating point number.
    return float(self.value().numpy())

  def numpy(self):
    # [FUNCTION DOC]
    # Returns the value as a numpy array when executing eagerly; otherwise raises a
    # not-implemented error.
    if context.executing_eagerly():
      return self.read_value().numpy()
    else:
      raise NotImplementedError(
          "numpy() is only available when eager execution is enabled.")

  def __str__(self):
    # [FUNCTION DOC]
    # String form: that of the wrapped variable.
    return str(self._v)

  def __repr__(self):
    # [FUNCTION DOC]
    # Representation: that of the wrapped variable.
    return repr(self._v)

  def _should_act_as_resource_variable(self):
    # [FUNCTION DOC]
    # HELPER: marker so resource-variable checks accept this wrapper. Does nothing.
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when not in a caching scope the as_ref argument is ignored and always passed as
    # false to the wrapped variable.
    # Converts to a dense tensor: the cached value in a caching scope, else the wrapped variable's.
    # Inputs: dtype (optional), name (optional), as_ref (default false).
    if distribute_utils.caching_scope_local.in_caching_scope():
      return self.cached_read_value()
    return self._v._dense_var_to_tensor(dtype=dtype, name=name, as_ref=False)  # pylint: disable=protected-access

  @classmethod
  def _overload_overloadable_operators(cls):
    # [FUNCTION DOC]
    # HELPER: sets up operator overloading for the class.
    # Class method that installs an overload for every overloadable tensor operator except
    # equality and inequality, each delegating to the tensor implementation on the current value.
    """Register overloads for all operators."""
    for operator in tensor.Tensor.OVERLOADABLE_OPERATORS:
      # Overloading __eq__ or __ne__ does not work as expected.
      if operator == "__eq__" or operator == "__ne__":
        continue
      cls._tensor_overload_operator(operator)

  @classmethod
  def _tensor_overload_operator(cls, operator):
    # [FUNCTION DOC]
    # HELPER: installs one operator overload on the class.
    # Looks up the operator on the tensor type and sets a function on the class that applies it to
    # the variable's (possibly cached) value. Input: operator name.
    """Delegate an operator overload to `tensor.Tensor`."""
    tensor_operator = getattr(tensor.Tensor, operator)

    def _operator(v, *args, **kwargs):
      # [FUNCTION DOC]
      # The installed overload: applies the tensor operator to the variable's current value with
      # the extra arguments, returning the result.
      return tensor_operator(v.value(), *args, **kwargs)  # pylint: disable=protected-access
    setattr(cls, operator, _operator)

  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # Checkpoint hook: returns a mapping from the standard variable-value key to the wrapped variable.
    return {trackable.VARIABLE_VALUE_KEY: self._v}

  def _export_to_saved_model_graph(self, object_map, tensor_map,
                                   options, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inner comment mentions AggregatingVariable (copy-paste); this class is the
    # caching variable.
    # Saved-model export hook: delegates to the wrapped variable, then maps this wrapper to the
    # wrapped variable's entry. Returns the resource list.
    """For implementing `Trackable`."""
    # By delegating this method to the wrapped variable, SavedModel with
    # AggregatingVariable are identical to SavedModel with normal variables.
    resource_list = self._v._export_to_saved_model_graph(object_map, tensor_map,  # pylint:disable=protected-access
                                                         options, **kwargs)
    object_map[self] = object_map[self._v]
    return resource_list

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # Copies the wrapped variable to the CPU into the object map and creates a new caching
    # wrapper around the copy if none exists yet. Mutates the object map.
    """For implementing `Trackable`."""
    # Create a copy of `self._v` to object_map, then create a new copy of self
    # that wraps the copy of `self._v`.
    # When updating value, only the lowest-level variable will actually do that,
    # the copy of `CachingVariable` is more like a shell.
    self._v._copy_trackable_to_cpu(object_map)  # pylint:disable=protected-access
    if self not in object_map:
      # If copy of `self` not populated yet, initialize one.
      object_map[self] = CachingVariable(object_map[self._v])


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_aggregate(var, dtype=None, name=None, as_ref=False):
  # [FUNCTION DOC]
  # HELPER: tensor conversion function registered for AggregatingVariable.
  # Converts the variable to a tensor by its dense conversion. Inputs: var, dtype, name, as_ref.
  return var._dense_var_to_tensor(dtype, name, as_ref)  # pylint: disable=protected-access


tensor_conversion_registry.register_tensor_conversion_function(
    AggregatingVariable, _tensor_conversion_aggregate)


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_caching(var, dtype=None, name=None, as_ref=False):
  # [FUNCTION DOC]
  # HELPER: tensor conversion function registered for CachingVariable.
  # Converts the variable to a tensor by its dense conversion. Inputs: var, dtype, name, as_ref.
  return var._dense_var_to_tensor(dtype, name, as_ref)  # pylint: disable=protected-access


tensor_conversion_registry.register_tensor_conversion_function(
    CachingVariable, _tensor_conversion_caching)

CachingVariable._overload_overloadable_operators()  # pylint: disable=protected-access


class PerWorkerVariable(resource_variable_ops.BaseResourceVariable):
  # [CLASS DOC]
  # Variable wrapper for unsynchronized variables that exist independently on each worker. Holds
  # the coordinator, a creator for per-worker variables, the lazily created per-worker values and
  # a coordinator-side instance used for tracing and saving. Graph handles are captured as values
  # resolved at call time so the worker-local variable is used. Experimental; parameter server
  # strategy only.
  """A wrapper around unsynced variables created on workers.

  `PerWorkerVariable`s are variables that are stored on workers and not
  synchronized. A `PerWorkerVariable` is really a wrapper around multiple
  independent `Variable`s stored on independent worker machines. 
  `PerWorkerVariable` is currently only tested and supported when used with
  `ParameterServerStrategy`. A `PerWorkerVariable` can be created by creating a
  `Variable` within strategy scope and using the `per_worker_variable` flag,
  e.g.:

  ```
  with strategy.scope():
    var = tf.Variable(initial_value=0.0, per_worker_variable=True)
  ```

  The implementation modifies the graph to ensure that a worker's local version
  of the variable is used for computation at call time, while needing only one
  function trace and requiring no code changes beyond the `per_worker_variable`
  flag. `PerWorkerVariable`s can thus be treated like a standard `Variable`, but
  support is experimental and not all ops have been tested.

  All per-worker values can be retrieved and read into a list via
  `PerWorkerVariable.read_all()`.

  Caveats:
    - `PerWorkerVariable`s should not be used as direct inputs to a
      `tf.function`. That is, they should not appear in a tf.function header as
      an input argument. However they can still be read and manipulated in a
      `tf.function`.
    - The `shape` argument must be fully-defined (no `None` entries) or left
      empty. Partially-defined shapes are not yet supported.
    - Automatic control dependencies do not work with `PerWorkerVariable`s, so
      returning a `PerWorkerVariable` is not supported, and `read_all()` should 
      be used to retrieve values. (TODO: b/286052052)
    - `PerWorkerVariable`s should not be created within a `tf.function`.
  """

  def __init__(self, strategy, next_creator, **kwargs):
    # [FUNCTION DOC]
    # Builds the coordinator-side variable instance and records its shape, dtype and metadata.
    # Inputs: strategy (with a cluster coordinator), next_creator (callable creating the real
    # variable), extra keyword arguments passed to the creator (including optional graph-mode flag,
    # unique id, handle name and shape validation). Not trainable. Returns nothing.
    self._coordinator = strategy._cluster_coordinator
    self._per_worker_vars = None
    self._var_creator = functools.partial(next_creator, **kwargs)

    self._coordinator_instance = next_creator(**kwargs)

    # Set ResourceVariable attributes based on kwargs
    if kwargs.get("in_graph_mode") is None:
      with ops.init_scope():
        self._in_graph_mode = not context.executing_eagerly()
    else:
      self._in_graph_mode = kwargs["in_graph_mode"]

    self._cached_value = None
    self._shape = self._coordinator_instance.shape
    self._dtype = self._coordinator_instance.dtype
    self._trainable = False  # not supported
    self._unique_id = kwargs.get("unique_id")
    if kwargs.get("handle_name") is None:
      self._handle_name = "Variable:0"
    else:
      self._handle_name = kwargs["handle_name"] + ":0"
    self._validate_shape = kwargs.get("validate_shape", True)

  @classmethod
  def _variable_call(cls, *args, **kwargs):
    # [FUNCTION DOC]
    # Override that does nothing and returns nothing, preventing the metaclass from creating
    # resource variables.
    """Override to be a no-op to avoid metaclass creating ResourceVariables."""
    return None

  @property
  def handle(self):
    # [FUNCTION DOC]
    # Property: the resource handle. Eagerly or when saving, returns the coordinator instance's
    # handle. Otherwise creates per-worker variables if needed and captures a call-time value in
    # the default graph.
    if context.executing_eagerly() or save_context.in_save_context():
      return self._coordinator_instance.handle
    else:
      self._maybe_create_per_worker_vars()
      closure, spec = self.handle_call_time_value()
      return ops.get_default_graph().capture_call_time_value(
          closure,
          spec)

  def handle_call_time_value(self):
    # [FUNCTION DOC]
    # Returns a pair of a closure and a spec for the handle placeholder. The closure returns the
    # handle of the variable on the current dispatch worker, or the coordinator's handle when
    # tracing outside a dispatch.
    """Returns a closure to run for a handle at call time and its spec.

    This function is called in self.handle to create a placeholder
    which returns a handle on some worker or on the coordinator.
    """

    def closure():
      # [FUNCTION DOC]
      # Closure evaluated at call time: gets the worker's remote value and returns its handle, or
      # the coordinator instance's handle when there is no dispatch context.
      dispatch_context = coordinator_context.get_current_dispatch_context()
      if dispatch_context:
        remote_value = self._per_worker_vars._values[  # pylint: disable=protected-access
            dispatch_context.worker_index]
        ret = dispatch_context.maybe_get_remote_value(remote_value)
        return ret.handle
      else:
        # Only needed for tracing
        return self._coordinator_instance.handle
    return closure, PerWorkerVariableSpec(
        value=self._coordinator_instance.handle)

  def _maybe_create_per_worker_vars(self):
    # [FUNCTION DOC]
    # Creates the variable on every worker through the coordinator if not already created.
    """Create variable on each worker if it hasn't been created."""
    if not self._per_worker_vars:
      self._per_worker_vars = (
          self._coordinator._create_per_worker_variables(self._var_creator))  # pylint: disable=protected-access

  def read_all(self):
    # [FUNCTION DOC]
    # Synchronously reads the variable from all workers. Returns a list of tensors.
    """Synchronously read variables from all workers into a list of Tensors."""
    return [wv.get() for wv in self._per_worker_vars._values]  # pylint: disable=protected-access


class PerWorkerVariableSpec(tensor.TensorSpec):
  # [CLASS DOC]
  # Tensor spec for a per-worker variable handle; keeps the coordinator-side value so handle data
  # can be copied onto the placeholder.
  def __init__(self, value=None, name=None):
    # [FUNCTION DOC]
    # Creates a spec with the value's shape and dtype. Inputs: value (variable-like with shape and
    # dtype), name (optional).
    super().__init__(value.shape, value.dtype, name=name)
    self._value = value

  def placeholder_value(self, placeholder_context):
    # [FUNCTION DOC]
    # Creates the placeholder and copies the stored value's handle data onto it. Returns it.
    placeholder = super().placeholder_value(placeholder_context)
    handle_data_util.set_handle_data(placeholder, self._value._handle_data)  # pylint: disable=protected-access
    return placeholder


class DistributedTable(lookup_ops.StaticHashTable):
  # [CLASS DOC]
  # Static hash table replicated on each worker for ParameterServerStrategy. Keeps a coordinator
  # copy, the creator, the coordinator and a lazily built per-worker mapping of resource handles.
  # Calls resolve to the consumer worker's resource through call-time capture. Saved as a plain
  # table on the coordinator; restoring is unsupported.
  """A distributed StaticHashTable for ParameterServerStrategy.

  An instance of DistributedTable has copies of a StaticHashTable and its
  resource handle on the coordinator of each worker, created at the
  DistributedTable instance initialization time with initializers on each
  worker. Users can call methods on a DistributedTable as if it were a
  StaticHashTable, which leads to execution with the resource local to the
  consumer worker (or the coordinator, if calling from the coordinator). This
  implementation relies on the fact that the methods of StaticHashTable are
  queried with the resource handle (instead of the python object).

  Currently, at saving time, a DistributedTable is saved as a StaticHashTable on
  the coordinator, and restoring a DistributedTable from SavedModel is not
  supported.
  """

  def __init__(self, strategy, wrapped_creator):
    # [FUNCTION DOC]
    # Increments the input API usage counter, creates the coordinator copy, and builds the
    # per-worker tables unless in a save context. Inputs: strategy, wrapped_creator (callable
    # making a table).
    distribute_lib.distribution_strategy_input_api_counter.get_cell(
        self.__class__.__name__, "PSSDistributedLookupTable").increase_by(1)
    self._coordinator_instance = wrapped_creator()
    self._wrapped_creator = wrapped_creator
    self._coordinator = strategy._cluster_coordinator
    # self._distributed_table is a RemoteValue mapping worker_index to
    # RemoteValue that wraps a resource handle on the worker
    self._distributed_table = None
    self._distributed_table_creation_lock = threading.Lock()

    if not save_context.in_save_context():
      self._maybe_build_distributed_table()

  def __getattr__(self, attr):
    # [FUNCTION DOC]
    # Attribute fallback delegating to the coordinator copy; callable attributes found in its own
    # dictionary are rebound to this object. Raises an attribute error for the coordinator
    # instance itself, avoiding infinite recursion during shallow copies. Input: attribute name.
    # This allows copy.copy(DistributedTable), e.g. at saving time.
    # (DistributedVariable uses the same fix.) When copying an object, copy.copy
    # doesn't invoke its __init__ method, instead it makes a new empty object,
    # then copies the attributes over. copy.copy looks for attributes like
    # "__setstate__" in case the object implements its custom unpickling. Since
    # DistributedTable doesn't have those attributes defined, __getattr__ will
    # be invoked, which tries to access the `_coordinator_instance` attribute.
    # But that doesn't exist either because this is an empty object, and again
    # __getattr__ is invoked, leading to an infinite recursion.
    if attr == "_coordinator_instance":
      raise AttributeError()

    if attr in self._coordinator_instance.__dict__:
      attr_value = self._coordinator_instance.__dict__[attr]
      if callable(attr_value):

        def wrapper(*args, **kwargs):
          # [FUNCTION DOC]
          # Calls the delegated function with this table as the first argument plus given arguments.
          return attr_value(self, *args, **kwargs)

        return wrapper
      elif isinstance(attr_value, property):
        return attr_value
      else:
        return getattr(self._coordinator_instance, attr)
    else:
      return getattr(self._coordinator_instance, attr)

  def resource_handle_call_time_value(self):
    # [FUNCTION DOC]
    # Returns a closure and a tensor spec for the resource handle. The closure returns the handle
    # of the table on the current dispatch worker, or the coordinator's handle outside dispatch.
    """Returns a closure to run for a resource handle at call time and its spec.

    This function is called in self.resource_handle to create a placeholder
    which returns a resource handle on some worker or on the coordinator.
    """

    def closure():
      # [FUNCTION DOC]
      # Closure evaluated at call time: returns the worker's resource handle or the coordinator
      # instance's handle when there is no dispatch context.
      # function to be evaluated at function call time, returning a nest of
      # tensors compatible with `spec`.
      dispatch_context = coordinator_context.get_current_dispatch_context()
      if dispatch_context:
        remote_value = self._distributed_table._values[  # pylint: disable=protected-access
            dispatch_context.worker_index]
        ret = dispatch_context.maybe_get_remote_value(remote_value)
        return ret

      else:
        return self._coordinator_instance.resource_handle

    return closure, tensor.TensorSpec([], dtype=dtypes.resource)

  def _maybe_build_distributed_table(self):
    # [FUNCTION DOC]
    # Under a lock, creates the table and its resource on each worker through the coordinator if
    # not yet created, storing the result.
    """Create table objects and resources on each worker if hasn't been created."""
    with self._distributed_table_creation_lock:
      if not self._distributed_table:

        def create_copy():
          # [FUNCTION DOC]
          # Creates a new table and returns its resource handle; run on each worker.
          new_table = self._wrapped_creator()
          ret = new_table.resource_handle
          return ret

        self._distributed_table = (
            self._coordinator._create_per_worker_resources(create_copy))  # pylint: disable=protected-access

  @property
  def resource_handle(self):
    # [FUNCTION DOC]
    # Property: resource handle. Eagerly or when saving, the coordinator's handle; otherwise builds
    # the per-worker tables if needed and captures a call-time value with the coordinator handle
    # as default.
    if context.executing_eagerly() or save_context.in_save_context():
      return self._coordinator_instance.resource_handle
    else:
      self._maybe_build_distributed_table()
      closure, spec = self.resource_handle_call_time_value()
      return ops.get_default_graph().capture_call_time_value(
          closure,
          spec,
          default_value=self._coordinator_instance.resource_handle)

  @property
  def is_distributed_table(self):
    # [FUNCTION DOC]
    # Property: always true, marking this as a distributed table.
    return True

  def __tf_experimental_restore_capture__(
      self, concrete_function, internal_capture):
    # [FUNCTION DOC]
    # Restore-capture hook: replaces the internal capture in the concrete function's graph with a
    # deferred capture resolved per worker. Inputs: concrete_function, internal_capture
    # placeholder. Returns the newly added deferred external capture.
    closure, spec = self.resource_handle_call_time_value()
    concrete_function.graph.replace_capture_with_deferred_capture(
        self._coordinator_instance.resource_handle,
        closure,
        spec,
        default_value=self._coordinator_instance.resource_handle,
        placeholder=internal_capture)
    return concrete_function.graph.deferred_external_captures[-1]


_local_resource_restore_context = threading.local()


def get_current_local_resource_restore_context():
  # [FUNCTION DOC]
  # Returns the current local resource restore context of this thread, or nothing if none is set.
  try:
    return _local_resource_restore_context.current
  except AttributeError:
    return None


@contextlib.contextmanager
def with_local_resource_restore_context(instance):
  # [FUNCTION DOC]
  # Context manager: sets a restore context for the given instance on this thread for the body and
  # restores the previous one afterwards (not restored if the body raises). Input: instance.
  previous_context = getattr(_local_resource_restore_context, "current", None)
  _local_resource_restore_context.current = LocalResourceRestoreContext(
      instance)
  yield
  _local_resource_restore_context.current = previous_context


class LocalResourceRestoreContext(object):
  # [CLASS DOC]
  # Holds the distributed instance (such as a table) being restored; used with the context
  # manager so code can find the instance without a worker-to-handle mapping.
  """Class holding information of a distributed instance, e.g. StaticHashTable.

  Pairing use with context manager `with_local_resource_restore_context` allows
  operations under this context manager to conveniently gets information of a
  component of the `RestoredDistributedTable` (and other restored distributed
  `CapturableResource` if we're supporting their distribution in the future),
  instead of looking it up from the mapping of the worker-to-resource handle.
  This is especially useful when we know which instance the operations should
  execute with and the mapping is not available yet.
  """

  def __init__(self, instance):
    # [FUNCTION DOC]
    # Stores the instance. Returns nothing.
    self.instance = instance


class RestoredDistributedTable(DistributedTable):
  # [CLASS DOC]
  # Distributed table for tables restored from a saved model. Intercepts setting of the resource
  # methods, stores them to apply to worker tables and waits until all are available before
  # building per-worker tables.
  """A restored and distributed StaticHashTable for ParameterServerStrategy."""

  def __init__(self, strategy, wrapped_creator):
    # [FUNCTION DOC]
    # Creates the condition variable used to wait for restored resource functions, then runs the
    # base constructor. Inputs: strategy, wrapped_creator.
    # Wait for all resource functions to have been set before building the table
    self._has_resource_functions = threading.Condition()
    super().__init__(strategy, wrapped_creator)

  def resource_handle_call_time_value(self):
    # [FUNCTION DOC]
    # Like the base version but, inside a restore context, returns the handle of the table being
    # restored instead of looking up the per-worker mapping. Returns closure and spec.
    """Returns a closure to run for a resource handle at call time and its spec.

    This function is called in self.resource_handle to create a placeholder
    which returns a resource handle on some worker or on the coordinator.
    """

    def closure():
      # [FUNCTION DOC]
      # Closure evaluated at call time: uses the restore context's instance handle if present,
      # otherwise the worker's table; with no dispatch context returns the coordinator handle.
      # function to be evaluated at function call time, returning a nest of
      # tensors compatible with `spec`.
      dispatch_context = coordinator_context.get_current_dispatch_context()
      if dispatch_context:
        local_resource_restore_context = (
            get_current_local_resource_restore_context())

        # A LocalResourceRestoreContext is entered in the process of remote
        # table creation and initialization if we're in the process of loading
        # from a SavedModel. A LocalResourceRestoreContext carries the
        # information regarding which table is being created and initialized. In
        # order to initialize a table, we need the restored `_initialize`
        # function, which captures this closure as table resource. And when this
        # closure is executed, we will read the table info from the
        # LocalResourceRestoreContext and return its handle, rather than
        # following the normal procedure of fetching from
        # `self._distributed_table`, because we're still in the middle of
        # building `self._distributed_table`.
        if local_resource_restore_context:
          remote_value = local_resource_restore_context.instance.resource_handle

        else:
          remote_value = self._distributed_table._values[  # pylint: disable=protected-access
              dispatch_context.worker_index]

        ret = dispatch_context.maybe_get_remote_value(remote_value)
        return ret

      else:

        return self._coordinator_instance.resource_handle

    return closure, tensor.TensorSpec(shape=(), dtype=dtypes.resource)

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Attribute setter. For the resource methods, stores them for worker tables, wakes waiters once
    # all are present and also sets them on the coordinator copy. Other names use normal setting.
    # Inputs: name, value.
    if name in TRACKABLE_RESOURCE_METHODS:
      # When a StaticHashTable is loaded with `tf.saved_model.load`, it becomes
      # a RestoredResource with dummy `_create_resource`, `_initialize`, and
      # `_destroy_resource" methods. Similarly, when loaded with
      # `tf.keras.models.load_model`, its initializer becomes a dummy one. In
      # both cases, these methods needs to be set to some RestoredFunctions
      # through `__setattr__`. Thus we need to store and set these methods for
      # the distributed tables (a.k.a. `self._distributed_table`) on the
      # workers too, besides setting for the coordinator instance. However, we
      # cannot set them at this point, since the distributed tables have not
      # been created. We store them in '_restored_function' and set them to the
      # distributed tables when they're created in
      # `self._maybe_build_distributed_table.create_copy`.
      if not hasattr(self, "_restored_function"):
        self._restored_function = {}
      self._restored_function[name] = value
      if all(method in self._restored_function
             for method in TRACKABLE_RESOURCE_METHODS):
        with self._has_resource_functions:
          self._has_resource_functions.notify_all()
      return self._coordinator_instance.__setattr__(name, value)
    else:
      return super(RestoredDistributedTable, self).__setattr__(name, value)

  def _create_resource(self):
    # [FUNCTION DOC]
    # Creates a resource handle by delegating to the coordinator copy.
    """A function that creates a resource handle for a table on coordinator."""
    return self._coordinator_instance._create_resource()  # pylint: disable=protected-access

  def _initialize(self):
    # [FUNCTION DOC]
    # Initializes the resource by delegating to the coordinator copy.
    """A function that initializes the resource."""
    return self._coordinator_instance._initialize()  # pylint: disable=protected-access

  def _destroy_resource(self):
    # [FUNCTION DOC]
    # Destroys the resource by delegating to the coordinator copy.
    """A function that destroys the resource."""
    return self._coordinator_instance._destroy_resource()  # pylint: disable=protected-access

  def _maybe_build_distributed_table(self):
    # [FUNCTION DOC]
    # Under a lock, creates restored tables on each worker if not yet created, storing the result.
    """Create table objects and resources on each worker if hasn't been created."""
    with self._distributed_table_creation_lock:
      if not self._distributed_table:

        def create_copy():
          # [FUNCTION DOC]
          # Creates a table, waits until all restored resource functions are available, sets them on
          # it inside a restore context, initializes it (registering the initializer in graph mode)
          # and returns its resource handle.
          new_table = self._wrapped_creator()
          # Wait until all resource functions are available before setting them
          # on new_table.
          with self._has_resource_functions:
            while not hasattr(self, "_restored_function") or any(
                method not in self._restored_function
                for method in TRACKABLE_RESOURCE_METHODS):
              self._has_resource_functions.wait()

          if hasattr(self, "_restored_function"):
            with with_local_resource_restore_context(new_table):
              for name, tf_function in self._restored_function.items():
                setattr(new_table, name, tf_function)
              init_op = new_table._initialize()  # pylint: disable=protected-access
              if not context.executing_eagerly():
                ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)

          ret = new_table.resource_handle
          return ret

        self._distributed_table = (
            self._coordinator._create_per_worker_resources(create_copy))  # pylint: disable=protected-access
