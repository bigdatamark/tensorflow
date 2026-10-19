# [MODULE DOC]
# Defines a resource variable that packs several per-device variables into one object, plus a
# small wrapper that pins accesses to one device. Exposes PackedDistributedVariable and
# PackedVarAndDevice. Used by distribution strategies to reduce function-call overhead when a
# variable is replicated across devices (eager mode only). Import-time side effect: registers a
# tensor conversion function so the device-pinned wrapper can be converted to a tensor.
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
"""A variable which packs a list of variables distributed across devices."""

from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops


class PackedDistributedVariable(resource_variable_ops.BaseResourceVariable):
  # [CLASS DOC]
  # A resource variable wrapping a list of per-device resource variables behind a single packed
  # handle. Holds the list of underlying variables and the list of their devices. In eager
  # op-by-op execution, reads and updates are forwarded to the variable on the current device;
  # inside functions (non-eager) the packed handle is used. Subclasses the base resource variable.
  """A variable which packs multiple variables distributed across devices.

  It's only supported when eager execution is enabled.
  For op-by-op execution, use an unpacked handle on the current device; for
  function execution, use the packed handle to reduce the overhead of function
  calls.
  """

  def __init__(self, distributed_variables=None, name=None, **unused_kwargs):
    # [FUNCTION DOC]
    # Packs the given resource variables into one variable. distributed_variables is a non-empty
    # list of resource variables (one per device); name is an optional variable name (defaults to
    # "Variable", made unique by the name scope). Extra keyword arguments are ignored.
    # Raises a value error if not executing eagerly outside functions, if the list is empty or
    # null, or if any element is not a resource variable. Packs the eager handles into one packed
    # handle (inside the init scope) and initializes the base class with trainable flag, shape,
    # dtype, synchronization, constraint, aggregation and strategy taken from the first variable.
    # Records the variables and their devices on the instance.
    """Packs a list of variables which are distributed across devices.

    Args:
      distributed_variables: A list of distributed Variables to pack.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
    """
    if not ops.executing_eagerly_outside_functions():
      raise ValueError(
          "PackedDistributedVariable should be created in eager mode.")
    if not distributed_variables:
      raise ValueError("Expect a non-empty list of variables to pack.")
    for i, var in enumerate(distributed_variables):
      if not resource_variable_ops.is_resource_variable(var):
        raise ValueError("Expect a list of ResourceVariables to pack, "
                         "but the %d-th variable is %s" % (i, type(var)))

    self._distributed_variables = distributed_variables
    self._devices = [v.device for v in distributed_variables]
    with ops.init_scope():
      with ops.name_scope(name, "Variable", skip_on_eager=False) as name:
        handle = ops.pack_eager_tensors(
            [var.handle for var in distributed_variables])
        handle_name = ops.name_from_scope_name(name)
        unique_id = "%s_%d" % (handle_name, ops.uid())
        super(PackedDistributedVariable, self).__init__(
            trainable=distributed_variables[0].trainable,
            shape=distributed_variables[0].shape,
            dtype=distributed_variables[0].dtype,
            handle=handle,
            synchronization=distributed_variables[0].synchronization,
            constraint=distributed_variables[0].constraint,
            aggregation=distributed_variables[0].aggregation,
            distribute_strategy=distributed_variables[0]._distribute_strategy,  # pylint: disable=protected-access
            name=name,
            unique_id=unique_id,
            handle_name=handle_name,
            graph_element=None,
            initial_value=None,
            initializer_op=None,
            is_initialized_op=None,
            cached_value=None,
            caching_device=None,
            is_distributed_variables=True)

  @property
  def devices(self):
    # [FUNCTION DOC]
    # Returns the list of device names, one per packed variable, in packing order.
    return self._devices

  def on_device(self, device):
    # [FUNCTION DOC]
    # Returns a wrapper object that forwards all variable operations under the given device.
    return PackedVarAndDevice(self, device)

  def get_var_on_device(self, device):
    # [FUNCTION DOC]
    # Returns the packed variable located on the given device name. First looks for an exact
    # device match; as a fallback matches when the host of a packed variable's device equals the
    # given device. Raises a value error if neither match is found.
    for i, d in enumerate(self._devices):
      if d == device:
        return self._distributed_variables[i]
    # Device didn't match, check if device on same host as a fallback.
    for i, d in enumerate(self._devices):
      if device_util.get_host_for_device(d) == device:
        return self._distributed_variables[i]
    raise ValueError("Device %s is not found" % device)

  def get_var_on_current_device(self):
    # [FUNCTION DOC]
    # Returns the packed variable on the device currently in scope (canonicalized current device).
    # Raises a value error if no packed variable matches.
    current_device = device_util.canonicalize(device_util.current())
    return self.get_var_on_device(current_device)

  def initial_value(self, device):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the base class, this is a regular method taking a device argument
    # rather than a property; callers must pass a device name.
    # Returns the initial value tensor of the packed variable located on the given device.
    """Returns the Tensor used as the initial value for the variable."""
    return self.get_var_on_device(device).initial_value

  @property
  def handle(self):
    # [FUNCTION DOC]
    # Returns the resource handle. When executing eagerly, this is the unpacked handle of the
    # variable on the current device; otherwise it is the packed handle.
    if context.executing_eagerly():
      return self.get_var_on_current_device().handle
    else:
      return self._handle

  @property
  def packed_handle(self):
    # [FUNCTION DOC]
    # Returns the packed handle that combines all per-device handles.
    return self._handle

  def _read_variable_op(self, no_copy=False):
    # [FUNCTION DOC]
    # Reads the variable value. When eager, reads the value of the variable on the current device;
    # otherwise defers to the base class read, passing along the no_copy flag (default false).
    if context.executing_eagerly():
      return self.get_var_on_current_device().value()
    else:
      return super()._read_variable_op(no_copy)

  def value(self):
    # [FUNCTION DOC]
    # Returns the current value of the variable via the read operation.
    return self._read_variable_op()

  def is_initialized(self, name=None):
    # [FUNCTION DOC]
    # Returns a boolean tensor that is true only if all packed variables are initialized, built by
    # combining the per-variable checks with logical and. When eager, queries each underlying
    # variable; otherwise queries the base class check under each device in turn. The optional
    # name is applied to the final combined operation (and to the per-device base checks).
    if context.executing_eagerly():
      result = self._distributed_variables[0].is_initialized()
      for v in self._distributed_variables[1:-1]:
        result = math_ops.logical_and(result, v.is_initialized())
      result = math_ops.logical_and(
          result, self._distributed_variables[-1].is_initialized(), name=name)
    else:
      with ops.device(self._devices[0]):
        result = super(PackedDistributedVariable, self).is_initialized(name)
      for d in self._devices[1:-1]:
        with ops.device(d):
          initialized = super(PackedDistributedVariable,
                              self).is_initialized(name)
        result = math_ops.logical_and(result, initialized)
      with ops.device(self._devices[-1]):
        initialized = super(PackedDistributedVariable,
                            self).is_initialized(name)
      result = math_ops.logical_and(result, initialized, name=name)
    return result

  def _update(self, update_fn, value, **kwargs):
    # [FUNCTION DOC]
    # HELPER: shared implementation for the assign and scatter methods of this class.
    # Applies update_fn (a callback taking a variable, a value and extra keyword arguments) with
    # the given value and keyword arguments. When eager, applies it to the variable on the current
    # device; otherwise applies it to the base class view of this variable. Returns the callback's
    # result.
    if context.executing_eagerly():
      return update_fn(self.get_var_on_current_device(), value, **kwargs)
    else:
      return update_fn(super(PackedDistributedVariable, self), value, **kwargs)

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Subtracts delta from the variable (on the current device when eager). use_locking, name and
    # read_value (default true) are passed through. Returns the update result.
    assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
    return self._update(
        update_fn=assign_sub_fn,
        value=delta,
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Adds delta to the variable (on the current device when eager). use_locking, name and
    # read_value (default true) are passed through. Returns the update result.
    assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
    return self._update(
        update_fn=assign_add_fn,
        value=delta,
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Assigns value to the variable (on the current device when eager). use_locking, name and
    # read_value (default true) are passed through. Returns the update result.
    assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
    return self._update(
        update_fn=assign_fn,
        value=value,
        use_locking=use_locking,
        name=name,
        read_value=read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Subtracts a sparse (indexed) delta from the variable on the current device (when eager).
    # use_locking defaults to false; name is optional. Returns the update result.
    scatter_sub_fn = lambda var, *a, **kw: var.scatter_sub(*a, **kw)
    return self._update(
        update_fn=scatter_sub_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Adds a sparse (indexed) delta to the variable on the current device (when eager).
    # use_locking defaults to false; name is optional. Returns the update result.
    scatter_add_fn = lambda var, *a, **kw: var.scatter_add(*a, **kw)
    return self._update(
        update_fn=scatter_add_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Multiplies the variable by a sparse (indexed) delta on the current device (when eager).
    # use_locking defaults to false; name is optional. Returns the update result.
    scatter_mul_fn = lambda var, *a, **kw: var.scatter_mul(*a, **kw)
    return self._update(
        update_fn=scatter_mul_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Divides the variable by a sparse (indexed) delta on the current device (when eager).
    # use_locking defaults to false; name is optional. Returns the update result.
    scatter_div_fn = lambda var, *a, **kw: var.scatter_div(*a, **kw)
    return self._update(
        update_fn=scatter_div_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Takes the element-wise minimum of the variable and a sparse (indexed) delta on the current
    # device (when eager). use_locking defaults to false; name is optional. Returns the result.
    scatter_min_fn = lambda var, *a, **kw: var.scatter_min(*a, **kw)
    return self._update(
        update_fn=scatter_min_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Takes the element-wise maximum of the variable and a sparse (indexed) delta on the current
    # device (when eager). use_locking defaults to false; name is optional. Returns the result.
    scatter_max_fn = lambda var, *a, **kw: var.scatter_max(*a, **kw)
    return self._update(
        update_fn=scatter_max_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Overwrites the indexed slices of the variable with a sparse delta on the current device
    # (when eager). use_locking defaults to false; name is optional. Returns the update result.
    scatter_update_fn = lambda var, *a, **kw: var.scatter_update(*a, **kw)
    return self._update(
        update_fn=scatter_update_fn,
        value=sparse_delta,
        use_locking=use_locking,
        name=name)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # Converts the variable to a dense tensor. When eager, delegates to the variable on the current
    # device; otherwise to the base class. dtype, name and as_ref (default false) are passed
    # through. Returns the tensor.
    if context.executing_eagerly():
      return self.get_var_on_current_device()._dense_var_to_tensor(  # pylint: disable=protected-access
          dtype=dtype,
          name=name,
          as_ref=as_ref)
    else:
      return super(PackedDistributedVariable, self)._dense_var_to_tensor(  # pylint: disable=protected-access
          dtype=dtype,
          name=name,
          as_ref=as_ref)


class PackedVarAndDevice(object):
  # [CLASS DOC]
  # Lightweight wrapper pairing a packed distributed variable with a device. Every operation is
  # run under that device scope and forwarded to the wrapped variable. Created by the packed
  # variable's on-device method. Holds the variable and the device name.
  """Holds a packed distributed variable and a device."""

  def __init__(self, var, device):
    # [FUNCTION DOC]
    # Stores the packed variable and the device name; returns nothing.
    self._var = var
    self._device = device

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Fallback attribute lookup: fetches the named attribute from the wrapped variable while the
    # stored device scope is active. The try/re-raise exists to work around a reference cycle
    # created when exceptions propagate through a context manager (see comment below), which can
    # delay finalizers of resources such as iterators. Exceptions are re-raised unchanged.
    # Exceptions raised inside the contextmanager can cause a reference
    # cycle.[1] The cycle involves the current frame, which holds the reference
    # to the outer frame. Tensorflow, e.g. iterators, relies on object
    # finalizers to clean up resources. Such references prevents the resource
    # from being deleted and can cause leaks and errors. One corner the case is
    # that iterators are kept alive and the garbage collector happens to run
    # after auto control dependencies; this causes the deletion to lose the
    # control dependencies to operations that uses such resources.
    #
    # Catch and re-raise the exception seems to workaround the issue.
    #
    # [1] https://bugs.python.org/issue43533
    try:
      with ops.device(self._device):
        return getattr(self._var, name)
    except:  # pylint: disable=try-except-raise
      raise

  def var(self):
    # [FUNCTION DOC]
    # Returns the wrapped packed variable.
    return self._var

  def value(self):
    # [FUNCTION DOC]
    # Returns the wrapped variable's value, read under the stored device.
    with ops.device(self._device):
      return self._var.value()

  def read_value(self):
    # [FUNCTION DOC]
    # Returns the wrapped variable's read-value result, computed under the stored device.
    with ops.device(self._device):
      return self._var.read_value()

  @property
  def initial_value(self):
    # [FUNCTION DOC]
    # Returns the initial value of the wrapped variable on the stored device.
    return self._var.initial_value(self._device)

  def initialized_value(self):
    # [FUNCTION DOC]
    # Returns the wrapped variable's initialized value, computed under the stored device.
    with ops.device(self._device):
      return self._var.initialized_value()

  @property
  def device(self):
    # [FUNCTION DOC]
    # Returns the stored device name.
    return self._device

  @property
  def handle(self):
    # [FUNCTION DOC]
    # Returns the wrapped variable's handle, fetched under the stored device scope.
    with ops.device(self._device):
      return self._var.handle

  def on_device_handle(self):
    # [FUNCTION DOC]
    # Returns the handle of the underlying variable on the current device, evaluated while the
    # stored device scope is active (so the current device is the stored one).
    with ops.device(self._device):
      return self._var.get_var_on_current_device().handle

  @property
  def op(self) -> ops.Operation:
    # [FUNCTION DOC]
    # Returns the wrapped variable's operation, fetched under the stored device scope.
    with ops.device(self._device):
      return self._var.op

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Subtracts delta from the wrapped variable under the stored device; returns the result.
    with ops.device(self._device):
      return self._var.assign_sub(delta, use_locking, name, read_value)

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Adds delta to the wrapped variable under the stored device; returns the result.
    with ops.device(self._device):
      return self._var.assign_add(delta, use_locking, name, read_value)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Assigns value to the wrapped variable under the stored device; returns the result.
    with ops.device(self._device):
      return self._var.assign(value, use_locking, name, read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse subtract on the wrapped variable under the stored device; returns the result.
    with ops.device(self._device):
      return self._var.scatter_sub(sparse_delta, use_locking, name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse add on the wrapped variable under the stored device; returns the result.
    with ops.device(self._device):
      return self._var.scatter_add(sparse_delta, use_locking, name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse multiply on the wrapped variable under the stored device; returns the result.
    with ops.device(self._device):
      return self._var.scatter_mul(sparse_delta, use_locking, name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse divide on the wrapped variable under the stored device; returns the result.
    with ops.device(self._device):
      return self._var.scatter_div(sparse_delta, use_locking, name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse minimum on the wrapped variable under the stored device; returns the result.
    with ops.device(self._device):
      return self._var.scatter_min(sparse_delta, use_locking, name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse maximum on the wrapped variable under the stored device; returns the result.
    with ops.device(self._device):
      return self._var.scatter_max(sparse_delta, use_locking, name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse overwrite on the wrapped variable under the stored device; returns the result.
    with ops.device(self._device):
      return self._var.scatter_update(sparse_delta, use_locking, name)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # Converts the wrapped variable to a dense tensor under the stored device. dtype, name and
    # as_ref (default false) are passed through. Returns the tensor.
    with ops.device(self._device):
      return self._var._dense_var_to_tensor(  # pylint: disable=protected-access
          dtype=dtype,
          name=name,
          as_ref=as_ref)

  def _as_graph_element(self):
    # [FUNCTION DOC]
    # Returns the wrapped variable's graph element representation (used by graph conversion).
    return self._var._as_graph_element()  # pylint: disable=protected-access


def _tensor_conversion_packed_var_and_device(var,
                                             dtype=None,
                                             name=None,
                                             as_ref=False):
  # [FUNCTION DOC]
  # HELPER: tensor conversion callback registered for the device-pinned wrapper class.
  # Converts the wrapper to a dense tensor by delegating to its dense conversion method; dtype,
  # name and as_ref (default false) are passed through. Returns the tensor.
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


tensor_conversion_registry.register_tensor_conversion_function(
    PackedVarAndDevice, _tensor_conversion_packed_var_and_device)
