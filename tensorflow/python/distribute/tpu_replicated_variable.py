# [MODULE DOC]
# Defines TPUReplicatedVariable, a variable container that treats a list of identical variables
# placed on different logical TPU cores (model parallelism) as one variable, plus a small helper
# for per-device updates. Builds on TPU utility functions, XLA sharding and partitioned-input ops
# and integrates with checkpointing and saved-model export. Import-time side effect: registers a
# tensor-conversion function for the class so instances can be used where tensors are expected.
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
"""A Variable class that is replicated to logical cores for model parallelism."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import abc
import contextlib

from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_tpu_partition_ops as tpu_partition_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable


def _on_device_update(update_fn, var, value, **kwargs):
  # [FUNCTION DOC]
  # HELPER: glue used by TPUReplicatedVariable to run an update on the device of one component.
  # Inputs: update_fn (callback taking a variable, a value and extra keyword options), var (one
  # variable), value (the already-converted tensor) and extra keyword arguments forwarded to the
  # callback. Enters a device scope for var's device, calls the callback and returns its result
  # (typically an update operation or variable).
  with ops.device(var.device):
    return update_fn(var, value, **kwargs)


class TPUReplicatedVariable(variables_lib.Variable):
  # [CLASS DOC]
  # A variable-like container wrapping a list of equally shaped and typed variables that live on
  # separate logical TPU devices (model parallelism) so that the TF2XLA bridge sees one single
  # variable. State: the list of component variables, the container name, a common name (name
  # without any output-index suffix) and an optional cached value used to order assignments.
  # Most read-only properties (dtype, shape, device, trainable, constraint, graph, ...) delegate
  # to the first component; assignments are applied to every component. The handle is a
  # partitioned/replicated tensor inside TPU contexts. Subclass of the generic Variable class.
  # Import-time side effect: a tensor conversion function for this class is registered at the
  # bottom of the module.
  """Container for replicated `Variables` that are treated as a single variable.

  This class maintains a list of replicated variables that are stored on
  separate logic TPU devices. TF2XLA bridge accesses these variables as
  if they were a single variable.
  """

  def __init__(self, variables, name='TPUReplicatedVariable'):
    # [FUNCTION DOC]
    # Constructor. Inputs: variables (a non-empty sequence of variable objects, ideally resource
    # variables, not shared with other containers) and name (string, default
    # "TPUReplicatedVariable"). Raises a type error if variables is not a non-empty sequence of
    # variables, and a value error if the components differ in dtype or in shape. Stores the
    # components, the name, the name stripped of any colon suffix, and sets the cached value to
    # nothing.
    """Treats `variables` as a replicated list of `tf.Variable`s.

    Example:

    ```
    variables = [
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
    ]
    replicated_variable = TPUReplicatedVariable(variables)
    assert replicated_variable.shape.as_list() == [10, 100]
    ```

    Args:
      variables: A list of `ResourceVariable`s that comprise this replicated
        variable. Variables should not be shared between different
        `TPUReplicatedVariable` objects.
      name: String. Name of this container. Defaults to "TPUReplicatedVariable".
    """
    if not isinstance(variables, abc.Sequence) or not variables or any(
        not isinstance(v, variables_lib.Variable) for v in variables):
      raise TypeError('Argument `variables` should be a non-empty list of '
                      f'`variables.Variable`s. Received {variables}')

    if any(v.dtype != variables[0].dtype for v in variables):
      raise ValueError(
          'All elements in argument `variables` must have the same dtype. '
          f'Received dtypes: {[v.dtype for v in variables]}')

    if any(v.shape != variables[0].shape for v in variables):
      raise ValueError(
          'All elements in argument `variables` must have the same shape. '
          f'Received shapes: {[v.shape for v in variables]}')

    self._vars = variables
    self._name = name
    self._common_name = self._name.split(':')[0]
    self._cached_value = None

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the underlying component variables (the full list, regardless of
    # save context).
    """Return an iterable for accessing the underlying sharded variables."""
    return iter(self._vars)

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property: the container name given at construction; used for checkpointing.
    """The name of this object. Used for checkpointing."""
    return self._name

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property: the data type shared by all components (taken from the first one).
    """The dtype of all `Variable`s in this object."""
    return self._vars[0].dtype

  @property
  def is_initialized(self):
    # [FUNCTION DOC]
    # Property: returns the first component's initialization indicator as is (for resource
    # variables this is a callable producing a boolean tensor, not a plain boolean). Other
    # components are not consulted.
    return self._vars[0].is_initialized

  @property
  def trainable(self):
    # [FUNCTION DOC]
    # Property: whether the variable is trainable, taken from the first component.
    return self._vars[0].trainable

  @property
  def device(self):
    # [FUNCTION DOC]
    # Property: the device string of the first component (the other components live elsewhere).
    """The device this variable is on."""
    return self._vars[0].device

  @contextlib.contextmanager
  def _handle_graph(self):
    # [FUNCTION DOC]
    # HELPER: context manager (generator) that makes the graph of this variable's handle the
    # default graph for the duration of the block. Takes nothing, yields nothing. Appears unused
    # within this module.
    with self.handle.graph.as_default():
      yield

  @contextlib.contextmanager
  def _assign_dependencies(self):
    # [FUNCTION DOC]
    # HELPER: context manager (generator) that, when a cached value exists, adds a control
    # dependency on it for the block; otherwise does nothing special. Yields nothing. Appears
    # unused within this module, and the cached value is never set to anything but nothing here.
    if self._cached_value is not None:
      with ops.control_dependencies([self._cached_value]):
        yield
    else:
      yield

  @property
  def constraint(self):
    # [FUNCTION DOC]
    # Property: the optional constraint of the first component.
    return self._vars[0].constraint

  @property
  def _in_graph_mode(self):
    # [FUNCTION DOC]
    # HELPER: property reporting whether the first component was created in graph mode (reads a
    # private attribute of that component).
    return self._vars[0]._in_graph_mode  # pylint: disable=protected-access

  @property
  def _unique_id(self):
    # [FUNCTION DOC]
    # HELPER: property returning the unique id of the first component (private attribute).
    return self._vars[0]._unique_id  # pylint: disable=protected-access

  @property
  def graph(self):
    # [FUNCTION DOC]
    # Property: the graph of the first component.
    return self._vars[0].graph

  @property
  def _shared_name(self):
    # [FUNCTION DOC]
    # HELPER: property returning the common name (container name without colon suffix) used as
    # the shared name of the resource.
    return self._common_name

  @property
  def synchronization(self):
    # [FUNCTION DOC]
    # Property: always reports no synchronization, regardless of the components.
    return variable_scope.VariableSynchronization.NONE

  @property
  def aggregation(self):
    # [FUNCTION DOC]
    # Property: always reports no aggregation, regardless of the components.
    return variable_scope.VariableAggregation.NONE

  @property
  def variables(self):
    # [FUNCTION DOC]
    # Property: the component variables. Inside a saving context returns a list with only the
    # first component (so that only one copy is saved); otherwise returns the full list. Note
    # that updates iterate over this property, so in a saving context only the first component
    # would be updated.
    """The list of `Variables`."""
    if save_context.in_save_context():
      return [self._vars[0]]
    return self._vars

  def _export_to_saved_model_graph(self, object_map, tensor_map,
                                   options, **kwargs):
    # [FUNCTION DOC]
    # HELPER: implements the trackable saved-model export hook. Inputs: object_map and
    # tensor_map (mutable mappings from objects/handles to their exported counterparts), options
    # (save options) and extra keyword arguments. Exports the first component through its own
    # method, then maps every other component, and its handle, to the first component's exported
    # object and handle, maps this container to the first component too, and appends the other
    # handles and this container to the resource list. Returns the resource list; mutates both
    # mappings.
    """For implementing `Trackable`."""
    first_var = self._vars[0]
    resource_list = first_var._export_to_saved_model_graph(  # pylint:disable=protected-access
        object_map, tensor_map, options, **kwargs)
    for v in self._vars[1:]:
      object_map[v] = object_map[first_var]
      tensor_map[v.handle] = tensor_map[first_var.handle]
      resource_list.append(v.handle)
    object_map[self] = object_map[first_var]
    tensor_map[self] = tensor_map[first_var.handle]
    resource_list.append(self)
    return resource_list

  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # HELPER: trackable checkpoint hook. Returns a mapping from the standard variable-value key
    # to the first component variable only (so just one copy is saved).
    return {trackable.VARIABLE_VALUE_KEY: self._vars[0]}

  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # HELPER: trackable checkpoint hook. Input: restored_tensors, a mapping containing the value
    # under the standard variable-value key. Assigns that value to the container (hence to all
    # components) and returns the result of the assignment.
    restored_tensor = restored_tensors[trackable.VARIABLE_VALUE_KEY]
    return self.assign(restored_tensor)

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # HELPER: trackable hook used by asynchronous checkpointing to copy state to host memory.
    # Input: object_map, a mutable mapping from original objects to their CPU copies. If this
    # container is already in the mapping, asks each component to refresh its copy. Otherwise
    # asks each component to create its copy, builds a new container (same name) from the copies
    # and records it in the mapping. Returns nothing.
    """For implementing `Trackable`."""
    if self in object_map:
      # If populated already, just update the values to the copy.
      for v in self._vars:
        v._copy_trackable_to_cpu(object_map)  # pylint: disable=protected-access
    else:
      # If not populated, populate first, then copy over the values.
      copied_vars = []
      for v in self._vars:
        v._copy_trackable_to_cpu(object_map)  # pylint: disable=protected-access
        copied_vars.append(object_map[v])
      new_var = TPUReplicatedVariable(copied_vars, name=self.name)
      object_map[self] = new_var

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property: the static shape of the first component (all components share it).
    return self._vars[0].shape

  @property
  def handle(self):
    # [FUNCTION DOC]
    # Property: the resource handle. In a saving context or eager execution returns the first
    # component's handle. Otherwise, outside any TPU context raises a not-implemented error.
    # Inside a TPU context, temporarily leaves the TPU context and builds a partitioned-input
    # tensor: either from the packed handle of an attached packed variable (when one exists and
    # soft device placement is off) or from all component handles unpacked. The result is then
    # marked as replicated for XLA sharding and returned.
    if save_context.in_save_context() or context.executing_eagerly():
      return self._vars[0].handle

    if tpu_util.enclosing_tpu_context() is None:
      raise NotImplementedError('TPUReplicatedVariable.handle is not available '
                                'outside tpu context or save context')
    else:
      with tpu_util.outside_or_skip_tpu_context():
        packed_var = getattr(self, '_packed_var', None)

        # TODO(b/202047549): Enable packed variables with soft device placement
        if packed_var is None or config.get_soft_device_placement():
          tensor = tpu_partition_ops.tpu_partitioned_input_v2(
              [v.handle for v in self._vars],
              partition_dims=[], is_packed=False)
        else:
          tensor = tpu_partition_ops.tpu_partitioned_input_v2(
              [packed_var.packed_handle], partition_dims=[], is_packed=True)

      return xla_sharding.replicate(tensor)

  def _read_variable_op(self):
    # [FUNCTION DOC]
    # HELPER: emits a raw read-variable operation on this container's handle with its dtype and
    # returns the resulting tensor (used inside TPU computations).
    return gen_resource_variable_ops.read_variable_op(self.handle, self.dtype)

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # HELPER: tensor conversion hook registered for this class.
    # KNOWN ISSUE: the dtype, name and as_ref arguments are accepted but silently ignored (no
    # cast or reference semantics).
    # Returns the first component's value when not inside a TPU context, otherwise a raw read
    # of the replicated handle.
    """Converts a variable to a tensor."""
    # pylint: disable=protected-access
    if tpu_util.enclosing_tpu_context() is None:
      return self.read_value()
    else:
      return self._read_variable_op()

  def read_value(self):
    # [FUNCTION DOC]
    # Returns the current value of the first component as a tensor (all components are kept in
    # sync by assignment).
    return self._vars[0].read_value()

  def _update(self, update_fn, value, **kwargs):
    # [FUNCTION DOC]
    # HELPER: shared implementation of the assignment methods. Inputs: update_fn (callback that
    # applies an update to one variable), value (anything convertible to a tensor of this
    # variable's dtype) and extra keyword arguments passed to the callback. Converts value to a
    # tensor, applies the callback to every component on that component's device, and returns a
    # grouped operation covering all updates.
    """Converts the value to tensor and updates the variable list."""
    input_tensor = ops.convert_to_tensor(
        value, name='value_in_tensor', dtype=self.dtype)

    return control_flow_ops.group(
        *tuple(
            _on_device_update(update_fn, v, input_tensor, **kwargs)
            for v in self.variables))

  def assign(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Assigns a new value. Inputs: value, use_locking (boolean, default false), name (optional
    # string) and read_value (boolean, default true: whether to return the new value). Outside a
    # TPU context, or when executing eagerly, assigns to every component and returns the grouped
    # update. Inside a TPU graph context, emits a raw assign-variable operation on the
    # replicated handle and returns its result.
    if tpu_util.enclosing_tpu_context() is None or context.executing_eagerly():
      assign_fn = lambda var, *a, **ka: var.assign(*a, **ka)
      return self._update(
          assign_fn,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Subtracts a value from the variable; same inputs and branching as assign (all components
    # outside TPU context or in eager mode; raw assign-subtract on the replicated handle inside a
    # TPU graph).
    if tpu_util.enclosing_tpu_context() is None or context.executing_eagerly():
      assign_sub_fn = lambda var, *a, **ka: var.assign_sub(*a, **ka)
      return self._update(
          assign_sub_fn,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_sub_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Adds a value to the variable; same inputs and branching as assign (all components outside
    # TPU context or in eager mode; raw assign-add on the replicated handle inside a TPU graph).
    if tpu_util.enclosing_tpu_context() is None or context.executing_eagerly():
      assign_add_fn = lambda var, *a, **ka: var.assign_add(*a, **ka)
      return self._update(
          assign_add_fn,
          value=value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    else:
      return tpu_util.make_raw_assign_fn(
          gen_resource_variable_ops.assign_add_variable_op)(
              self,
              value=value,
              use_locking=use_locking,
              name=name,
              read_value=read_value)

  def __str__(self):
    # [FUNCTION DOC]
    # Returns a multi-line string with the class name followed by each component's string form,
    # numbered by index.
    debug_str = ',\n'.join(
        '  %d: %s' % (i, v) for i, v in enumerate(self._vars))
    return '%s:{\n%s\n}' % (self.__class__.__name__, debug_str)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a multi-line string with the class name followed by each component's repr,
    # numbered by index.
    debug_repr = ',\n'.join(
        '  %d: %r' % (i, v) for i, v in enumerate(self._vars))
    return '%s:{\n%s\n}' % (self.__class__.__name__, debug_repr)


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
def _tensor_conversion_tpu_replicated_var(var,
                                          dtype=None,
                                          name=None,
                                          as_ref=False):
  # [FUNCTION DOC]
  # HELPER: conversion function registered so a replicated variable can be used as a tensor.
  # Inputs: var (the replicated variable), dtype, name and as_ref (all forwarded). Returns the
  # result of the variable's own dense-conversion hook.
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


tensor_conversion_registry.register_tensor_conversion_function(
    TPUReplicatedVariable, _tensor_conversion_tpu_replicated_var)
