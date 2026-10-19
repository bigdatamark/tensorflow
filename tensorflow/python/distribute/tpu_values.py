# [MODULE DOC]
# Defines the TPU-specific flavours of distributed variables used by the TPU distribution
# strategy. It provides a mixin (TPUVariableMixin) that changes how handles, reads and
# conversions behave when code runs inside a TPU rewrite context versus outside of it, the
# concrete distributed, mirrored, lazily initialised and sync-on-read variable classes built on
# that mixin, module-level assign helpers shared by the mirrored variable and the write policy,
# and the on-write / on-read synchronization policies for TPU. Outside a TPU context most calls
# fall back to the generic behavior in the values module; inside a TPU context they use the
# replicated variable handle and raw resource-variable operations. No import-time side effects
# other than defining a module-level error message template for scatter operations. The tests
# for this module live in the values test file.
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
"""Various classes representing TPU distributed values.

Note that the tests are in values_test.py .

"""

from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


_scatter_error_msg = ("{op_name} is only supported for distributed "
                      "variable (variable created within certain "
                      "`tf.distribute.Strategy` scope) with NONE "
                      " aggregation, got: {aggregation}.")


class TPUVariableMixin(object):
  # [CLASS DOC]
  # Mixin that adapts a distributed variable class to TPUs. It is combined with a concrete
  # distributed-variable base class (listed after it in the inheritance order) and overrides
  # handle lookup, reading, tensor conversion and graph-element access so that inside a TPU
  # context the variable is represented by a replicated variable handle, while outside it
  # delegates to the base class. State: a handle identifier used as a cache key for the
  # replicated handle. It relies on attributes supplied by the base class (common name,
  # primary variable, values list, packed variable, mirrored check).
  """Mixin for TPU variables."""

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Forwards all positional and keyword arguments to the next class in the
    # inheritance order, then computes and stores a handle identifier. When eager execution is
    # active outside functions, the identifier is the variable's common name joined with the
    # identity number of the primary variable (because different eager variables can share a
    # name); otherwise it is just the common name. Returns nothing.
    super(TPUVariableMixin, self).__init__(*args, **kwargs)

    # Handle ID is needed for `get_replicated_var_handle` to cache the variables
    # correctly since in eager mode different variables can have the same name.
    if ops.executing_eagerly_outside_functions():
      self._handle_id = self._common_name + "_" + str(id(self._primary))
    else:
      self._handle_id = self._common_name

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Fallback attribute lookup. Outside a TPU context it defers to the base class lookup for
    # the given attribute name and returns its result. Inside a TPU context it raises an
    # attribute error saying the named attribute is not accessible there.
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self).__getattr__(name)
    else:
      raise AttributeError(
          f"`TPUVariableMixin.{name}` not accessible within a TPU context.")

  def get(self):
    # [FUNCTION DOC]
    # Returns the component variable for the current replica/device by delegating to the base
    # class, but only outside a TPU context. Inside a TPU context it raises a not-implemented
    # error because per-replica component lookup is unsupported there. Takes no arguments.
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self).get()
    else:
      raise NotImplementedError(
          "`TPUVariableMixin.get()` is not supported within a TPU context.")

  def _get_as_operand(self):
    # [FUNCTION DOC]
    # HELPER: used by tensor-conversion / operator-overloading glue to obtain the value to
    # compute with.
    # Returns the result of reading the variable's current value.
    return self.read_value()

  @property
  def handle(self):
    # [FUNCTION DOC]
    # Property giving the resource handle used to access this variable. Outside a TPU context,
    # or when executing eagerly, it picks the component on the current device (or the primary
    # one); if that component is a packed variable-and-device wrapper it returns the handle
    # for the current device, otherwise the component's own handle. Inside a TPU rewrite in
    # graph mode it asks the TPU context for the replicated variable handle, passing the common
    # name, the handle identifier, the list of component variables (or just the packed variable
    # when one exists), whether the variable is mirrored, and whether it is packed.
    """The handle by which this variable can be accessed."""
    # If we're in a tpu.rewrite(), return the replicated handle.
    tpu_context = tpu_util.enclosing_tpu_context()
    if tpu_context is None or context.executing_eagerly():
      var = self._get_on_device_or_primary()
      if isinstance(var, packed.PackedVarAndDevice):
        return var.on_device_handle()
      else:
        return var.handle
    else:
      is_packed = self._packed_var is not None
      val = self._values
      if is_packed:
        val = [self._packed_var]

      return tpu_context.get_replicated_var_handle(self._common_name,
                                                   self._handle_id, val,
                                                   self._is_mirrored(),
                                                   is_packed)

  @property
  def device(self):
    # [FUNCTION DOC]
    # Property returning the device string of this variable's handle (note that this evaluates
    # the handle property, which depends on the TPU/eager context).
    return self.handle.device

  def _read_variable_op(self):
    # [FUNCTION DOC]
    # HELPER: shared by read_value and value when running inside a TPU context.
    # Reads the variable using the raw read-variable operation on its handle and returns the
    # resulting tensor. If the variable is trainable, records the access on the gradient tape
    # first. When the handle is a packed handle, the read is placed under a device scope for
    # the current-device (or primary) component.
    """Reads the value of this variable."""
    if self.trainable:
      tape.variable_accessed(self)

    handle = self.handle
    if getattr(handle, "is_packed", False):
      # Add a device scope for a packed variable handle.
      with ops.device(self._get_on_device_or_primary().device):
        return gen_resource_variable_ops.read_variable_op(handle, self.dtype)
    else:
      return gen_resource_variable_ops.read_variable_op(handle, self.dtype)

  def read_value(self):
    # [FUNCTION DOC]
    # Returns the current value as a tensor. Outside a TPU context it uses the base class
    # behavior; inside one it performs a raw variable read on the replicated handle.
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self).read_value()
    else:
      return self._read_variable_op()

  def value(self):
    # [FUNCTION DOC]
    # Returns the current value as a tensor. Outside a TPU context it uses the base class
    # behavior; inside one it performs a raw variable read on the replicated handle.
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self).value()
    else:
      return self._read_variable_op()

  def _as_graph_element(self):
    # [FUNCTION DOC]
    # Returns the graph element for this variable. Outside a TPU context this is whatever the
    # base class returns; inside a TPU context it returns nothing (null value).
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self)._as_graph_element()  # pylint: disable=protected-access
    else:
      return None

  @property
  def op(self):
    # [FUNCTION DOC]
    # Property returning the operation associated with the variable. While saving in a
    # non-distributed fashion it returns the primary variable's operation directly; otherwise
    # it builds a distributed-variable operation wrapper from the primary operation's name,
    # graph, traceback and type.
    if values_util.is_saving_non_distributed():
      return self._primary.op
    return values.DistributedVarOp(self._primary.op.name,
                                   self._primary.op.graph,
                                   self._primary.op.traceback,
                                   self._primary.op.type)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # HELPER: tensor-conversion hook used when the variable is passed where a tensor is expected.
    # Parameters: dtype (optional requested element type), name (optional op name), as_ref
    # (whether to return the handle as a reference instead of a value). Outside a TPU context it
    # delegates to the base class with the same arguments. Inside a TPU context it returns a
    # cast of the read value when a different dtype is requested, otherwise the handle when
    # as_ref is true or the read value when false.
    """Converts a variable to a tensor."""
    # pylint: disable=protected-access
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUVariableMixin, self)._dense_var_to_tensor(
          dtype=dtype, name=name, as_ref=as_ref)
    # pylint: enable=protected-access
    elif dtype is not None and dtype != self.dtype:
      return math_ops.cast(self.read_value(), dtype)
    else:
      return self.handle if as_ref else self.read_value()


class TPUDistributedVariable(TPUVariableMixin, values.DistributedVariable):
  # [CLASS DOC]
  # Distributed variable for the TPU strategy: combines the TPU mixin with the generic
  # distributed variable. All assignment and scatter methods route through the variable's
  # synchronization policy, except while saving in non-distributed mode, when they act directly
  # on the primary component variable.
  """DistributedVariable subclass for TPUStrategy."""

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Subtracts value from the variable. Parameters: value (amount to subtract), use_locking
    # (optional flag, default false), name (optional op name), read_value (default true; whether
    # to return the updated value). While saving non-distributed it applies the operation to the
    # primary variable; otherwise it delegates to the policy. Returns what that call returns.
    if values_util.is_saving_non_distributed():
      return self._primary.assign_sub(value, use_locking, name, read_value)
    return self._policy.assign_sub(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Adds value to the variable. Parameters: value (amount to add), use_locking (optional flag,
    # default false), name (optional op name), read_value (default true; whether to return the
    # updated value). While saving non-distributed it applies the operation to the primary
    # variable; otherwise it delegates to the policy. Returns what that call returns.
    if values_util.is_saving_non_distributed():
      return self._primary.assign_add(value, use_locking, name, read_value)
    return self._policy.assign_add(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Overwrites the variable with value. Parameters: value (new value), use_locking (optional
    # flag, default false), name (optional op name), read_value (default true; whether to
    # return the updated value). While saving non-distributed it assigns on the primary
    # variable; otherwise it delegates to the policy. Returns what that call returns.
    if values_util.is_saving_non_distributed():
      return self._primary.assign(value, use_locking, name, read_value)
    return self._policy.assign(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Subtracts a sparse (indexed-slices) update from the variable. Parameters: sparse_delta
    # (the sparse update), use_locking (optional, default false), name (optional). While saving
    # non-distributed it acts on the primary variable; otherwise it delegates to the policy.
    # Returns the result of that call.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_sub(sparse_delta, use_locking, name)
    return self._policy.scatter_sub(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Adds a sparse (indexed-slices) update to the variable. Parameters: sparse_delta (the
    # sparse update), use_locking (optional, default false), name (optional). While saving
    # non-distributed it acts on the primary variable; otherwise it delegates to the policy.
    # Returns the result of that call.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_add(sparse_delta, use_locking, name)
    return self._policy.scatter_add(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Multiplies the variable by a sparse (indexed-slices) update. Parameters: sparse_delta
    # (the sparse update), use_locking (optional, default false), name (optional). While saving
    # non-distributed it acts on the primary variable; otherwise it delegates to the policy.
    # Returns the result of that call.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_mul(sparse_delta, use_locking, name)
    return self._policy.scatter_mul(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Divides the variable by a sparse (indexed-slices) update. Parameters: sparse_delta (the
    # sparse update), use_locking (optional, default false), name (optional). While saving
    # non-distributed it acts on the primary variable; otherwise it delegates to the policy.
    # Returns the result of that call.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_div(sparse_delta, use_locking, name)
    return self._policy.scatter_div(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Takes the element-wise minimum of the variable and a sparse (indexed-slices) update.
    # Parameters: sparse_delta, use_locking (optional, default false), name (optional). While
    # saving non-distributed it acts on the primary variable; otherwise it delegates to the
    # policy. Returns the result of that call.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_min(sparse_delta, use_locking, name)
    return self._policy.scatter_min(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Takes the element-wise maximum of the variable and a sparse (indexed-slices) update.
    # Parameters: sparse_delta, use_locking (optional, default false), name (optional). While
    # saving non-distributed it acts on the primary variable; otherwise it delegates to the
    # policy. Returns the result of that call.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_max(sparse_delta, use_locking, name)
    return self._policy.scatter_max(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Overwrites the selected slices of the variable with a sparse (indexed-slices) update.
    # Parameters: sparse_delta, use_locking (optional, default false), name (optional). While
    # saving non-distributed it acts on the primary variable; otherwise it delegates to the
    # policy. Returns the result of that call.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_update(sparse_delta, use_locking, name)
    return self._policy.scatter_update(
        self, sparse_delta, use_locking=use_locking, name=name)


class TPUMirroredVariable(TPUVariableMixin, values.MirroredVariable):
  # [CLASS DOC]
  # Mirrored variable for the TPU strategy: a set of per-replica component variables kept in
  # sync, with the TPU mixin behavior. It can be backed by TPU replicated variables (replicated
  # or sharded to logical cores), in which case outside a TPU context updates go through the
  # generic update mechanism. Inside a TPU context with no aggregation, raw assign operations
  # are used; scatter operations are only supported while saving non-distributed.
  """Holds a map from replica to TPU variables whose values are kept in sync."""

  def _is_replicated_or_sharded_to_logical_cores(self):
    # [FUNCTION DOC]
    # HELPER: used by the assign methods and the device property to choose a code path.
    # Returns true when the primary component is a TPU replicated variable, false otherwise.
    """Returns whether each of the underlying variables is replicated or sharded to logical cores.

    If True, the handles of the underlying variables are not available outside a
    TPU context.
    """
    return isinstance(self._primary,
                      tpu_replicated_variable.TPUReplicatedVariable)

  @property
  def device(self):
    # [FUNCTION DOC]
    # Property giving the device of the variable. If the components are replicated or sharded
    # to logical cores and no TPU context is active, returns the primary component's device;
    # otherwise returns the device computed by the parent classes (from the handle).
    if (self._is_replicated_or_sharded_to_logical_cores() and
        tpu_util.enclosing_tpu_context() is None):
      return self._primary.device
    return super(TPUMirroredVariable, self).device

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Subtracts value from the variable. Parameters: value, use_locking (default false), name
    # (optional), read_value (default true). Three paths: (1) replicated/sharded components
    # outside a TPU context: applies a subtraction to each component through the generic update
    # mechanism; (2) inside a TPU context with aggregation NONE: uses the raw
    # assign-subtract-variable operation directly; (3) otherwise: calls the module-level
    # assign-subtract helper. Returns the result of the chosen path.
    tpu_context = tpu_util.enclosing_tpu_context()
    if (self._is_replicated_or_sharded_to_logical_cores() and
        tpu_context is None):
      assign_sub_fn = lambda v, *a, **ka: v.assign_sub(*a, **ka)
      return self._update(
          update_fn=assign_sub_fn,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)

    if (tpu_context and
        self.aggregation == variable_scope.VariableAggregation.NONE):
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)
    return assign_sub(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Adds value to the variable. Parameters: value, use_locking (default false), name
    # (optional), read_value (default true). Three paths: (1) replicated/sharded components
    # outside a TPU context: applies an addition to each component through the generic update
    # mechanism; (2) inside a TPU context with aggregation NONE: uses the raw
    # assign-add-variable operation directly; (3) otherwise: calls the module-level assign-add
    # helper. Returns the result of the chosen path.
    tpu_context = tpu_util.enclosing_tpu_context()
    if (self._is_replicated_or_sharded_to_logical_cores() and
        tpu_context is None):
      assign_add_fn = lambda v, *a, **ka: v.assign_add(*a, **ka)
      return self._update(
          update_fn=assign_add_fn,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)

    if (tpu_context and
        self.aggregation == variable_scope.VariableAggregation.NONE):
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)
    return assign_add(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Overwrites the variable with value. Parameters: value, use_locking (default false), name
    # (optional), read_value (default true). Three paths: (1) replicated/sharded components
    # outside a TPU context: assigns each component through the generic update mechanism;
    # (2) inside a TPU context with aggregation NONE: uses the raw assign-variable operation
    # directly; (3) otherwise: calls the module-level assign helper. Returns the result of the
    # chosen path.
    tpu_context = tpu_util.enclosing_tpu_context()
    if (self._is_replicated_or_sharded_to_logical_cores() and
        tpu_context is None):
      assign_fn = lambda v, *a, **ka: v.assign(*a, **ka)
      return self._update(
          update_fn=assign_fn,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)

    if (tpu_util.enclosing_tpu_context() and
        self.aggregation == variable_scope.VariableAggregation.NONE):
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)
    return assign(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def scatter_sub(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse subtraction. Only supported while saving in non-distributed mode, where the call
    # is forwarded with all arguments to the primary variable and its result returned.
    # Otherwise raises a not-implemented error.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_sub(*args, **kwargs)
    raise NotImplementedError

  def scatter_add(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse addition. Only supported while saving in non-distributed mode, where the call is
    # forwarded with all arguments to the primary variable and its result returned. Otherwise
    # raises a not-implemented error.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_add(*args, **kwargs)
    raise NotImplementedError

  def scatter_max(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse element-wise maximum. Only supported while saving in non-distributed mode, where
    # the call is forwarded with all arguments to the primary variable and its result returned.
    # Otherwise raises a not-implemented error.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_max(*args, **kwargs)
    raise NotImplementedError

  def scatter_min(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse element-wise minimum. Only supported while saving in non-distributed mode, where
    # the call is forwarded with all arguments to the primary variable and its result returned.
    # Otherwise raises a not-implemented error.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_min(*args, **kwargs)
    raise NotImplementedError

  def scatter_mul(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse multiplication. Only supported while saving in non-distributed mode, where the
    # call is forwarded with all arguments to the primary variable and its result returned.
    # Otherwise raises a not-implemented error.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_mul(*args, **kwargs)
    raise NotImplementedError

  def scatter_div(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse division. Only supported while saving in non-distributed mode, where the call is
    # forwarded with all arguments to the primary variable and its result returned. Otherwise
    # raises a not-implemented error.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_div(*args, **kwargs)
    raise NotImplementedError

  def scatter_update(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse overwrite of selected slices. Only supported while saving in non-distributed mode,
    # where the call is forwarded with all arguments to the primary variable and its result
    # returned. Otherwise raises a not-implemented error.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_update(*args, **kwargs)
    raise NotImplementedError


class TPULazyDistributedVariable(TPUDistributedVariable):
  # [CLASS DOC]
  # TPU distributed variable whose component variables are created lazily: before the first
  # assignment or read it triggers initialization of all lazily created variables through a
  # lazy scope object (supplied by the strategy) and records that this has happened.
  """TPU Mirrored variable to be initialized lazily in a batch."""

  def _initialize_if_uninitialized(self):
    # [FUNCTION DOC]
    # HELPER: called at the start of every assign and read of the lazy variable.
    # Does nothing if the variable was already marked as lazily initialized. Otherwise asks the
    # lazy scope to initialize all pending variables and marks this variable as initialized.
    # Returns nothing.
    if getattr(self, "_is_lazily_initialized", False):
      return
    self._lazy_scope.initialize_all()

    self._is_lazily_initialized = True

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Ensures lazy initialization has happened, then performs the parent class subtraction with
    # the same arguments (value, use_locking, name, read_value) and returns its result.
    self._initialize_if_uninitialized()
    return super().assign_sub(
        value, use_locking, name, read_value
    )

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Ensures lazy initialization has happened, then performs the parent class addition with the
    # same arguments (value, use_locking, name, read_value) and returns its result.
    self._initialize_if_uninitialized()
    return super().assign_add(
        value, use_locking, name, read_value
    )

  def assign(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Ensures lazy initialization has happened, then performs the parent class assignment with
    # the same arguments (value, use_locking, name, read_value) and returns its result.
    self._initialize_if_uninitialized()

    return super().assign(
        value, use_locking, name, read_value
    )

  def read_value(self):
    # [FUNCTION DOC]
    # Ensures lazy initialization has happened, then returns the parent class read value.
    self._initialize_if_uninitialized()
    return super().read_value()


class TPUSyncOnReadVariable(TPUVariableMixin, values.SyncOnReadVariable):
  # [CLASS DOC]
  # TPU variable with sync-on-read semantics: each replica keeps its own value and the values
  # are reduced only when saved or read in cross-replica context. Assignments inside a TPU
  # context use raw resource-variable assign operations; outside they use the generic
  # sync-on-read implementation.
  """Holds a map from replica to variables whose values are reduced on save."""

  def assign_sub(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Subtracts from the variable, forwarding all arguments. Outside a TPU context it calls the
    # generic sync-on-read subtraction; inside one it uses the raw assign-subtract-variable
    # operation. Returns the result.
    if tpu_util.enclosing_tpu_context() is None:
      return values.SyncOnReadVariable.assign_sub(self, *args, **kwargs)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(self, *args,
                                                            **kwargs)

  def assign_add(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Adds to the variable, forwarding all arguments. Outside a TPU context it calls the
    # generic sync-on-read addition; inside one it uses the raw assign-add-variable operation.
    # Returns the result.
    if tpu_util.enclosing_tpu_context() is None:
      return values.SyncOnReadVariable.assign_add(self, *args, **kwargs)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(self, *args,
                                                            **kwargs)

  def assign(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Overwrites the variable, forwarding all arguments. Outside a TPU context it calls the
    # generic sync-on-read assignment; inside one it uses the raw assign-variable operation.
    # Returns the result.
    if tpu_util.enclosing_tpu_context() is None:
      return values.SyncOnReadVariable.assign(self, *args, **kwargs)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_variable_op)(self, *args, **kwargs)


# Common method between OnWrite and Mirrored variables.
def assign_sub(var, value, use_locking=False, name=None, read_value=True):
  # [FUNCTION DOC]
  # HELPER: shared subtraction routine for the TPU mirrored variable and the TPU on-write
  # policy.
  # Parameters: var (a distributed variable), value (amount to subtract), use_locking (default
  # false), name (optional), read_value (default true). Builds a raw assign-subtract function
  # and applies it to every component through the variable's update mechanism. Returns the
  # result of that update.
  assign_sub_fn = tpu_util.make_raw_assign_fn(
      gen_resource_variable_ops.assign_sub_variable_op)
  return var._update(  # pylint: disable=protected-access
      update_fn=assign_sub_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)


def assign_add(var, value, use_locking=False, name=None, read_value=True):
  # [FUNCTION DOC]
  # HELPER: shared addition routine for the TPU mirrored variable and the TPU on-write policy.
  # Parameters: var (a distributed variable), value (amount to add), use_locking (default
  # false), name (optional), read_value (default true). Builds a raw assign-add function and
  # applies it to every component through the variable's update mechanism. Returns the result
  # of that update.
  assign_add_fn = tpu_util.make_raw_assign_fn(
      gen_resource_variable_ops.assign_add_variable_op)
  return var._update(  # pylint: disable=protected-access
      update_fn=assign_add_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)


def assign(var, value, use_locking=False, name=None, read_value=True):
  # [FUNCTION DOC]
  # HELPER: shared assignment routine for the TPU mirrored variable and the TPU on-write
  # policy.
  # Parameters: var (a distributed variable), value (new value), use_locking (default false),
  # name (optional), read_value (default true). Builds a raw assign function and applies it to
  # every component through the variable's update mechanism. Returns the result of that update.
  assign_fn = tpu_util.make_raw_assign_fn(
      gen_resource_variable_ops.assign_variable_op)
  return var._update(  # pylint: disable=protected-access
      update_fn=assign_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)


class TPUOnWritePolicy(values.OnWritePolicy):
  # [CLASS DOC]
  # Synchronization policy for variables that are kept in sync on every write. It overrides the
  # generic policy so that inside a TPU context with aggregation NONE, assignments use raw
  # resource-variable operations, and scatter operations are supported only with aggregation
  # NONE (otherwise a not-implemented error is raised inside a TPU context). Outside a TPU
  # context assignments go through the module-level assign helpers and scatters go through the
  # variable's generic update mechanism.
  """Policy defined for `tf.VariableSynchronization.ON_WRITE` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.AUTO` or `tf.VariableSynchronization.ON_WRITE`.
  """

  def assign_sub(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    # [FUNCTION DOC]
    # Subtracts value from the given variable var. Parameters: var, value, use_locking (default
    # false), name (optional), read_value (default true). Inside a TPU context with variable
    # aggregation NONE it applies the raw assign-subtract-variable operation directly;
    # otherwise it calls the module-level assign-subtract helper. Returns the result.
    if (tpu_util.enclosing_tpu_context() and
        var.aggregation == variable_scope.VariableAggregation.NONE):
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(
              var,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)
    return assign_sub(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    # [FUNCTION DOC]
    # Adds value to the given variable var. Parameters: var, value, use_locking (default
    # false), name (optional), read_value (default true). Inside a TPU context with variable
    # aggregation NONE it applies the raw assign-add-variable operation directly; otherwise it
    # calls the module-level assign-add helper. Returns the result.
    if (tpu_util.enclosing_tpu_context() and
        var.aggregation == variable_scope.VariableAggregation.NONE):
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(
              var,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)
    return assign_add(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign(self, var, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Overwrites the given variable var with value. Parameters: var, value, use_locking
    # (default false), name (optional), read_value (default true). Inside a TPU context with
    # variable aggregation NONE it applies the raw assign-variable operation directly;
    # otherwise it calls the module-level assign helper. Returns the result.
    if (tpu_util.enclosing_tpu_context() and
        var.aggregation == variable_scope.VariableAggregation.NONE):
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_variable_op)(
              var,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)
    return assign(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def _scatter_xxx(self,
                   raw_scater_xxx_fn,
                   op_name,
                   var,
                   sparse_delta,
                   use_locking=False,
                   name=None):
    # [FUNCTION DOC]
    # HELPER: common implementation behind all scatter methods of this policy.
    # Parameters: raw_scater_xxx_fn (the raw resource scatter operation to use), op_name (name
    # used in the error message), var (the distributed variable), sparse_delta (sparse update),
    # use_locking (default false), name (optional). Wraps the raw operation into a scatter
    # function. Inside a TPU context: raises a not-implemented error (formatted with the
    # operation name and aggregation) if the policy's aggregation is not NONE, else applies the
    # scatter directly to var. Outside a TPU context: applies the scatter to all components
    # via the variable's update mechanism. Returns the result of the scatter/update.
    scater_xxx_fn = tpu_util.make_raw_scatter_xxx_fn(raw_scater_xxx_fn)
    if tpu_util.enclosing_tpu_context():
      if self._aggregation != variable_scope.VariableAggregation.NONE:
        raise NotImplementedError(
            _scatter_error_msg.format(
                op_name=op_name, aggregation=self._aggregation))
      return scater_xxx_fn(
          var, sparse_delta=sparse_delta, use_locking=use_locking, name=name)
    else:
      return var._update(  # pylint: disable=protected-access
          update_fn=scater_xxx_fn,
          value=sparse_delta,
          use_locking=use_locking,
          name=name)

  def scatter_sub(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse subtraction on var using the resource scatter-subtract operation, via the common
    # scatter routine. Parameters: var, sparse_delta, use_locking (default false), name
    # (optional). Returns the routine's result.
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_sub,
                             "scatter_sub", var, sparse_delta, use_locking,
                             name)

  def scatter_add(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse addition on var using the resource scatter-add operation, via the common scatter
    # routine. Parameters: var, sparse_delta, use_locking (default false), name (optional).
    # Returns the routine's result.
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_add,
                             "scatter_add", var, sparse_delta, use_locking,
                             name)

  def scatter_max(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse element-wise maximum on var using the resource scatter-max operation, via the
    # common scatter routine. Parameters: var, sparse_delta, use_locking (default false), name
    # (optional). Returns the routine's result.
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_max,
                             "scatter_max", var, sparse_delta, use_locking,
                             name)

  def scatter_min(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse element-wise minimum on var using the resource scatter-min operation, via the
    # common scatter routine. Parameters: var, sparse_delta, use_locking (default false), name
    # (optional). Returns the routine's result.
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_min,
                             "scatter_min", var, sparse_delta, use_locking,
                             name)

  def scatter_mul(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse multiplication on var using the resource scatter-multiply operation, via the
    # common scatter routine. Parameters: var, sparse_delta, use_locking (default false), name
    # (optional). Returns the routine's result.
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_mul,
                             "scatter_mul", var, sparse_delta, use_locking,
                             name)

  def scatter_div(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse division on var using the resource scatter-divide operation, via the common
    # scatter routine. Parameters: var, sparse_delta, use_locking (default false), name
    # (optional). Returns the routine's result.
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_div,
                             "scatter_div", var, sparse_delta, use_locking,
                             name)

  def scatter_update(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse overwrite of selected slices of var using the resource scatter-update operation,
    # via the common scatter routine. Parameters: var, sparse_delta, use_locking (default
    # false), name (optional). Returns the routine's result.
    return self._scatter_xxx(gen_resource_variable_ops.resource_scatter_update,
                             "scatter_update", var, sparse_delta, use_locking,
                             name)


class TPUOnReadPolicy(values.OnReadPolicy):
  # [CLASS DOC]
  # Synchronization policy for variables whose replicas are kept separately and reduced on
  # read (sync-on-read). Inside a TPU context assignments use raw resource-variable operations;
  # outside they defer to the parent policy. All scatter operations are unsupported and raise a
  # not-implemented error.
  """Policy defined for `tf.VariableSynchronization.ON_READ` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.ON_READ` and `aggregation` is set to any of the
  values allowed by the `tf.VariableAggregation` enum such as `NONE`, `SUM`,
  `MEAN` or `ONLY_FIRST_REPLICA`when creating a `tf.Variable` in `tf.distribute`
  scope.
  """

  def assign_sub(self, var, *args, **kwargs):
    # [FUNCTION DOC]
    # Subtracts from variable var, forwarding the remaining arguments. Outside a TPU context
    # defers to the parent policy; inside one uses the raw assign-subtract-variable operation.
    # Returns the result.
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUOnReadPolicy, self).assign_sub(var, *args, **kwargs)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(var, *args,
                                                            **kwargs)

  def assign_add(self, var, *args, **kwargs):
    # [FUNCTION DOC]
    # Adds to variable var, forwarding the remaining arguments. Outside a TPU context defers to
    # the parent policy; inside one uses the raw assign-add-variable operation. Returns the
    # result.
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUOnReadPolicy, self).assign_add(var, *args, **kwargs)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(var, *args,
                                                            **kwargs)

  def assign(self, var, *args, **kwargs):
    # [FUNCTION DOC]
    # Overwrites variable var, forwarding the remaining arguments. Outside a TPU context defers
    # to the parent policy; inside one uses the raw assign-variable operation. Returns the
    # result.
    if tpu_util.enclosing_tpu_context() is None:
      return super(TPUOnReadPolicy, self).assign(var, *args, **kwargs)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_variable_op)(var, *args, **kwargs)

  def scatter_sub(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported for this policy: always raises a not-implemented error.
    raise NotImplementedError

  def scatter_add(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported for this policy: always raises a not-implemented error.
    raise NotImplementedError

  def scatter_max(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported for this policy: always raises a not-implemented error.
    raise NotImplementedError

  def scatter_min(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported for this policy: always raises a not-implemented error.
    raise NotImplementedError

  def scatter_mul(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported for this policy: always raises a not-implemented error.
    raise NotImplementedError

  def scatter_div(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported for this policy: always raises a not-implemented error.
    raise NotImplementedError

  def scatter_update(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported for this policy: always raises a not-implemented error.
    raise NotImplementedError
