# [MODULE DOC]
# Defines DistributedVariable (the newer variant of the replicated-variable wrapper that does not
# synchronize automatically, unlike the older class in values.py) and registers a tensor
# conversion for it. The wrapper holds several per-device resource variables and picks the right
# handle depending on context (replica device, TPU replicate scope, or primary replica); it can
# optionally pack the handles into one eager tensor.
# Import-time side effect: registers the tensor conversion function for DistributedVariable.

# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Various classes representing distributed values."""

import copy
import weakref

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib


# pylint: disable=protected-access


class DistributedVariable(resource_variable_ops.BaseResourceVariable):
  # [CLASS DOC]
  # A resource variable that wraps a list of per-device component variables and behaves like a
  # single variable. State: the list of components, an optional packed handle (one eager tensor
  # packing all component handles), a mapping from device name to handle, the primary (first)
  # handle, and a unique id. The handle used depends on context: the current device's handle,
  # the replicated handle inside a TPU replicate scope, the packed handle in graph mode when
  # packing is on, or the primary handle (also when saving a non-distributed model). It does not
  # synchronize components automatically. Read and write methods run under a device scope so
  # packed handles always have explicit device annotations. Serialization to and from proto is
  # not supported. Created with the list of components; constructor registers itself as
  # the distributed container on every component handle.
  """Represents variables that are replicated.

  It behaves exactly as a normal variable, but uses corresponding variable
  handle based on the context.
  - In each replica, it uses the handle from that replica.
  - In tpu.replicate(), it uses the replicated handle.
  - Otherwise, it uses the handle from the primary replica.

  Note that it doesn't synchronize automatically as the old DistributedVariable
  in values.py.
  """

  def __init__(self, variables, *, enable_packed_handle=False):
    # [FUNCTION DOC]
    # Builds the wrapper. Parameters: variables is a non-empty list of component resource
    # variables (one per device); enable_packed_handle (keyword only, default false) packs the
    # component handles into one eager tensor. Raises a value error if packing is requested but
    # eager execution is not enabled outside functions. Side effects: marks each component handle
    # with a weak reference back to this object, records a device-to-handle mapping, picks the
    # first component's handle as primary, creates a unique id from a name scope, and calls the
    # base constructor with attributes copied from the first component (trainable, shape, dtype,
    # synchronization, constraint, aggregation, strategy, name). In graph mode it also uses the
    # first component's initial value and groups all component initializers.
    if enable_packed_handle and not ops.executing_eagerly_outside_functions():
      raise ValueError(
          "Argument `enable_packed_handle` is true, but packed handle is only "
          "supported in eager mode. Please make sure eager execution is "
          "enabled.")
    self._variables = variables
    if enable_packed_handle:
      self._packed_handle = ops.pack_eager_tensors(
          [v.handle for v in variables])
    else:
      self._packed_handle = None
    for v in variables:
      v.handle._distributed_container = weakref.ref(self)  # pylint: disable=protected-access
    self._device_to_handle = {v.device: v.handle for v in variables}
    self._primary_handle = variables[0].handle
    with ops.init_scope(), \
         ops.name_scope("DistributedVariable", skip_on_eager=False) as name:
      handle_name = ops.name_from_scope_name(name)
      self._unique_id = "%s_%d" % (handle_name, ops.uid())
      if context.executing_eagerly():
        initial_value = None
        initializer = None
      else:
        initial_value = variables[0].initial_value
        initializer = control_flow_ops.group([v.initializer for v in variables])
      super().__init__(
          trainable=variables[0].trainable,
          shape=variables[0].shape,
          dtype=variables[0].dtype,
          handle=None,
          synchronization=variables[0].synchronization,
          constraint=variables[0].constraint,
          aggregation=variables[0].aggregation,
          distribute_strategy=variables[0]._distribute_strategy,
          name=variables[0].name,
          unique_id=self._unique_id,
          handle_name=handle_name,
          graph_element=variables[0]._graph_element,
          initial_value=initial_value,
          initializer_op=initializer,
          is_initialized_op=None,
          cached_value=None,
          caching_device=None,
          is_variables=True)

  @property
  def handle(self):
    # [FUNCTION DOC]
    # Property returning the resource handle appropriate for the current context. When saving a
    # non-distributed model: the primary handle. Inside a TPU context in graph mode: the
    # replicated variable handle obtained from the TPU context (using the packed handle or all
    # handles, mirrored unless the variable is sync-on-read, with the handle name stripped of any
    # output-index suffix). In graph mode with a packed handle: the packed handle. Otherwise the
    # handle for the current canonical device, falling back to the primary handle.
    if values_util.is_saving_non_distributed():
      return self._primary_handle
    tpu_context = tpu_util.enclosing_tpu_context()
    if tpu_context and not context.executing_eagerly():
      is_mirrored = (
          self._variables[0].synchronization !=
          variables_lib.VariableSynchronization.ON_READ)
      if self._packed_handle is None:
        handles = [v.handle for v in self._variables]
        is_packed = False
      else:
        handles = [self._packed_handle]
        is_packed = True
      common_name = self._handle_name
      # BaseResourceVariable appends ":0" to the handle name, which makes it not
      # a valid root scope name.
      if ":" in common_name:
        common_name = common_name.split(":")[0]
      return tpu_context.get_replicated_var_handle(common_name, self._unique_id,
                                                   handles, is_mirrored,
                                                   is_packed)
    if self._packed_handle is not None and not context.executing_eagerly():
      return self._packed_handle
    device = device_util.canonicalize(device_util.current())
    return self._device_to_handle.get(device, self._primary_handle)

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the variable name: the first component's name when saving a
    # non-distributed model, otherwise the base class name.
    if values_util.is_saving_non_distributed():
      return self._variables[0].name
    return super().name

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Property returning the initializer operation: the first component's initializer when saving a
    # non-distributed model, otherwise the base class initializer (the grouped one).
    if values_util.is_saving_non_distributed():
      return self._variables[0].initializer
    return super().initializer

  def _lazy_read(self, op):
    # [FUNCTION DOC]
    # HELPER: override of the base lazy-read hook; lazy reads are not supported here.
    # Parameter op is an operation to depend on. Returns a freshly read value computed after op.
    # Lazy read is not supported.
    with ops.control_dependencies([op]):
      return self.read_value()

  # Begin overrides of read/write methods to satisfy the requirement of using
  # packed handle, i.e. there must be explicit device annotations.

  def _device_scope(self):
    # [FUNCTION DOC]
    # HELPER: picks a context manager for read/write methods. Returns a do-nothing context when
    # there is no packed handle, when saving a non-distributed model, inside a TPU context, or when
    # the current device already has a component. Otherwise returns a device scope for the primary
    # handle's device so packed-handle operations have an explicit device.
    if (self._packed_handle is None or
        values_util.is_saving_non_distributed() or
        tpu_util.enclosing_tpu_context() is not None):
      return ops.NullContextmanager()
    device = device_util.canonicalize(device_util.current())
    if device in self._device_to_handle:
      return ops.NullContextmanager()
    return ops.device(self._primary_handle.device)

  def value(self):
    # [FUNCTION DOC]
    # Returns the current value by always doing a fresh read (never a cached value), because this
    # may be called on different devices.
    # We always force a read_value() instead of using the cached_value, as
    # value() can be called on different devices.
    return self.read_value()

  def read_value(self):
    # [FUNCTION DOC]
    # Reads the variable value under the device scope; returns the tensor from the base class.
    with self._device_scope():
      return super().read_value()

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Subtracts delta from the variable via the base class, under the device scope. Parameters:
    # delta amount; use_locking optional; name optional operation name; read_value whether to
    # return the new value (default true). Returns what the base class returns.
    with self._device_scope():
      return super().assign_sub(delta, use_locking, name, read_value)

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Adds delta to the variable via the base class, under the device scope. Parameters: delta
    # amount; use_locking optional; name optional operation name; read_value whether to return
    # the new value (default true). Returns what the base class returns.
    with self._device_scope():
      return super().assign_add(delta, use_locking, name, read_value)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Assigns value to the variable via the base class, under the device scope. Parameters: value
    # new value; use_locking optional; name optional operation name; read_value whether to return
    # the new value (default true). Returns what the base class returns.
    with self._device_scope():
      return super().assign(value, use_locking, name, read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Scatter-subtract of sparse_delta, delegated to the base class under the device scope.
    # use_locking and name are optional. Returns the base class result.
    with self._device_scope():
      return super().scatter_sub(sparse_delta, use_locking, name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Scatter-add of sparse_delta, delegated to the base class under the device scope.
    # use_locking and name are optional. Returns the base class result.
    with self._device_scope():
      return super().scatter_add(sparse_delta, use_locking, name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Scatter-multiply by sparse_delta, delegated to the base class under the device scope.
    # use_locking and name are optional. Returns the base class result.
    with self._device_scope():
      return super().scatter_mul(sparse_delta, use_locking, name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Scatter-divide by sparse_delta, delegated to the base class under the device scope.
    # use_locking and name are optional. Returns the base class result.
    with self._device_scope():
      return super().scatter_div(sparse_delta, use_locking, name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Scatter-minimum with sparse_delta, delegated to the base class under the device scope.
    # use_locking and name are optional. Returns the base class result.
    with self._device_scope():
      return super().scatter_min(sparse_delta, use_locking, name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Scatter-maximum with sparse_delta, delegated to the base class under the device scope.
    # use_locking and name are optional. Returns the base class result.
    with self._device_scope():
      return super().scatter_max(sparse_delta, use_locking, name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Scatter-update (overwrite) with sparse_delta, delegated to the base class under the device
    # scope. use_locking and name are optional. Returns the base class result.
    with self._device_scope():
      return super().scatter_update(sparse_delta, use_locking, name)

  def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Batched scatter-update with sparse_delta, delegated to the base class under the device scope.
    # use_locking and name are optional. Returns the base class result.
    with self._device_scope():
      return super().batch_scatter_update(sparse_delta, use_locking, name)

  def scatter_nd_sub(self, indices, updates, name=None):
    # [FUNCTION DOC]
    # N-dimensional scatter-subtract of updates at indices, delegated to the base class under the
    # device scope. name is optional. Returns the base class result.
    with self._device_scope():
      return super().scatter_nd_sub(indices, updates, name)

  def scatter_nd_add(self, indices, updates, name=None):
    # [FUNCTION DOC]
    # N-dimensional scatter-add of updates at indices, delegated to the base class under the
    # device scope. name is optional. Returns the base class result.
    with self._device_scope():
      return super().scatter_nd_add(indices, updates, name)

  def scatter_nd_update(self, indices, updates, name=None):
    # [FUNCTION DOC]
    # N-dimensional scatter-update of updates at indices, delegated to the base class under the
    # device scope. name is optional. Returns the base class result.
    with self._device_scope():
      return super().scatter_nd_update(indices, updates, name)

  def sparse_read(self, indices, name=None):
    # [FUNCTION DOC]
    # Gathers rows at indices via the base class under the device scope. name is optional.
    # Returns the gathered tensor.
    with self._device_scope():
      return super().sparse_read(indices, name)

  def gather_nd(self, indices, name=None):
    # [FUNCTION DOC]
    # N-dimensional gather at indices via the base class under the device scope. name is optional.
    # Returns the gathered tensor.
    with self._device_scope():
      return super().gather_nd(indices, name)

  def to_proto(self, export_scope=None):
    # [FUNCTION DOC]
    # Unsupported. Always raises a type error saying the variable cannot be converted to a proto;
    # export_scope is ignored.
    del self
    raise TypeError("DistributedVariable doesn't support to_proto")

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    # [FUNCTION DOC]
    # Static method, unsupported. Always raises a type error saying creating from a proto is not
    # supported; both arguments are ignored.
    raise TypeError("DistributedVariable doesn't support from_proto")

  def _as_graph_element(self):
    # [FUNCTION DOC]
    # HELPER: graph-element hook. If the default graph is finalized (no new ops can be added),
    # returns the first component's graph element; otherwise returns a freshly read value.
    if ops.get_default_graph().finalized:
      return self._variables[0]._graph_element
    return self.read_value()

  def _strided_slice_assign(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: strided-slice assignment hook. Passes all positional and keyword arguments to the
    # base class implementation under the device scope and returns its result.
    with self._device_scope():
      return super()._strided_slice_assign(*args, **kwargs)

  def __str__(self):
    # [FUNCTION DOC]
    # Returns text with the class name followed by a braced, comma-separated, indexed listing of
    # the string form of every component variable.
    debug_str = ",\n".join(
        "  %d: %s" % (i, v) for i, v in enumerate(self._variables))
    return "%s:{\n%s\n}" % (self.__class__.__name__, debug_str)

  def __repr__(self):
    # [FUNCTION DOC]
    # Same layout as the string form, but lists the debug representation of each component.
    debug_repr = ",\n".join(
        "  %d: %r" % (i, v) for i, v in enumerate(self._variables))
    return "%s:{\n%s\n}" % (self.__class__.__name__, debug_repr)

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Deep-copies the component variables using the given memo mapping and returns a new
    # DistributedVariable over the copies, with packed handles enabled iff this one has one.
    copied_variables = copy.deepcopy(self._variables, memo)
    return DistributedVariable(
        copied_variables, enable_packed_handle=self._packed_handle is not None)


def _tensor_conversion(var, dtype=None, name=None, as_ref=False):
  # [FUNCTION DOC]
  # HELPER: tensor conversion function registered for DistributedVariable.
  # Parameters: var the variable; dtype optional target type; name optional name; as_ref whether a
  # reference is requested. Raises a value error if as_ref is true (old graph control flow is not
  # supported). Otherwise converts the variable's freshly read value to a tensor and returns it.
  if as_ref:
    raise ValueError(
        "You may be using variable created under distribute strategy in TF "
        "1.x control flows. Try explicitly converting the variable to Tensor "
        "using variable.read_value(), or switch to TF 2.x.")
  return ops.convert_to_tensor(
      var.read_value(), dtype=dtype, name=name, as_ref=as_ref)


tensor_conversion_registry.register_tensor_conversion_function(
    DistributedVariable, _tensor_conversion)
