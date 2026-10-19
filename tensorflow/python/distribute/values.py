# [MODULE DOC]
# Core value containers of tf.distribute. A "distributed value" is a tuple of per-device
# component values wrapped in one object. This module defines: the base DistributedValues and
# the DistributedDelegate wrapper that forwards attributes and arithmetic operators to the
# component for the current replica; PerReplica (unsynchronized values, a composite tensor) with
# its type spec and a registered tensor conversion that only works inside replica context;
# Mirrored (identical values on all replicas); DistributedVariable (a variable made of one
# component variable per replica, delegating updates to a policy object), its trace type, a
# fake operation object, and checkpoint saveable helpers; MirroredVariable; SyncOnReadVariable;
# a policy hierarchy (VariablePolicy base, OnReadPolicy and OnWritePolicy) that defines how
# reads, writes, scatters and checkpointing behave for each synchronization mode; and
# PerWorkerResource, a per-host resource proxy. Also private helpers for replica-context
# updates and tensor conversion.
# Import-time side effects: registers tensor conversion functions for PerReplica,
# DistributedVariable, MirroredVariable, Mirrored and SyncOnReadVariable, and registers a
# type-spec codec for PerReplicaSpec with the nested-structure coder. Used by strategy
# implementations (mirrored, TPU, parameter server, etc.) and by the TPU-specific subclasses in
# the TPU values module. Tests live in the values test files.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
from typing import Optional
import weakref

from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.types import trace


def _on_write_update_replica(var, update_fn, value, **kwargs):
  # [FUNCTION DOC]
  # HELPER: shared by MirroredVariable and OnWritePolicy to apply updates in replica context.
  # Parameters: var (a distributed variable with ON_WRITE synchronization), update_fn (callable
  # that updates one component variable given the component and the value), value (the update
  # value, possibly a tensor), extra keyword arguments passed through to update_fn. Behavior:
  # with aggregation NONE it simply updates the component on the current device (or primary).
  # Otherwise, if the strategy does not use merge calls, it rejects mean aggregation on
  # non-float variables (raising a value error when the value is a tensor), aggregates the value
  # with the replica-context helper, marks the program as not saveable, and issues a grouped
  # replica-context update. If merge calls are used, it defines an inner merge function and
  # runs it through a merge call so that the aggregation happens in cross-replica context.
  # Returns the update result (updated variable or operation group). Raises a value error for
  # mean aggregation on non-float variables.
  """Updates variables with ON_WRITE synchronization in replica context."""
  """Updates variables with ON_WRITE synchronization in replica context."""
  if var.aggregation == vs.VariableAggregation.NONE:
    return update_fn(var._get_on_device_or_primary(), value, **kwargs)  # pylint: disable=protected-access

  if not distribute_lib.get_strategy().extended._use_merge_call():  # pylint: disable=protected-access
    # Don't allow MEAN with non float dtype, since it may cause unexpected
    # precision loss. Python3 and NumPy automatically upcast integers to
    # float in division, but we should always preserve the type.
    if var.aggregation == vs.VariableAggregation.MEAN and (
        not var.dtype.is_floating) and tensor_util.is_tf_type(value):
      raise ValueError(
          "Cannot update non-float variables with "
          "tf.VariableAggregation.MEAN aggregation in replica context. "
          "Either change the variable dtype to float or update it in "
          "cross-replica context.")

    aggregated_value = apply_aggregation_replica_context(
        value, var.aggregation, var)
    values_util.mark_as_unsaveable()

    return distribute_lib.get_replica_context()._update(  # pylint: disable=protected-access
        var,
        update_fn,
        args=(aggregated_value,),
        kwargs=kwargs,
        group=True)

  else:

    def merge_fn(strategy, value, **kwargs):
      # [FUNCTION DOC]
      # HELPER: nested merge function of _on_write_update_replica, run in cross-replica context.
      # Parameters: strategy (the active strategy, must equal the variable's strategy), value
      # (the per-replica value, possibly a PerReplica), keyword arguments for the update.
      # Rejects mean aggregation on non-float variables when the value differs per replica
      # (a PerReplica) by raising a value error; asserts the strategy matches; aggregates the
      # value according to the variable's aggregation; then performs a cross-replica update of
      # the variable. Returns the update result.
      """Aggregate values and update all variables in cross replica context."""
      # Don't allow MEAN with non float dtype, since it may cause unexpected
      # precision loss. Python3 and NumPy automatically upcast integers to
      # float in division, but we should always preserve the type.
      #
      # Note that to be backward compatible we allow the case when the value
      # is *always* the same on each replica. I.E. value is not a
      # PerReplica. Refer to regroup() to see how values are grouped.
      if var.aggregation == vs.VariableAggregation.MEAN and (
          not var.dtype.is_floating) and isinstance(value, PerReplica):
        raise ValueError(
            "Cannot update non-float variables with "
            "tf.VariableAggregation.MEAN aggregation in replica context. "
            "Either change the variable dtype to float or update it in "
            "cross-replica context.")

      assert strategy == var.distribute_strategy
      v = values_util.apply_aggregation(strategy, value, var.aggregation, var)
      return var._update_cross_replica(update_fn, v, **kwargs)  # pylint: disable=protected-access

    return distribute_lib.get_replica_context().merge_call(
        merge_fn, args=(value,), kwargs=kwargs)


def apply_aggregation_replica_context(value, aggregation, destinations):
  # [FUNCTION DOC]
  # Aggregates a value across replicas from inside replica context. Parameters: value (a
  # tensor-like value or a plain python literal), aggregation (a variable aggregation mode),
  # destinations (where the result should live, normally the variable). Raises a type error if
  # the value is a DistributedValues. Non-tensor values are returned unchanged. For
  # only-first-replica aggregation it switches to cross-replica context via a merge call and
  # broadcasts the first replica's value to the destinations; otherwise it converts the
  # aggregation to a reduce operation and performs an all-reduce through the strategy's
  # extended API. Returns the aggregated value.
  """Aggregate `value` to `destinations` as specified by `aggregation`."""
  # if it is a python literal, return without aggregation
  if isinstance(value, DistributedValues):
    raise TypeError(
        "Cannot use DistributedValues to update variables in replica context.")
  if not tensor_util.is_tf_type(value):
    return value

  if aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
    # Switch to cross-replica context to broadcast
    def merge_fn(strategy, value):
      # [FUNCTION DOC]
      # HELPER: nested merge function of apply_aggregation_replica_context.
      # Runs in cross-replica context; takes the strategy and the per-replica value, and
      # broadcasts the first replica's local value to the destinations of the enclosing call.
      # Returns the broadcast result.
      return strategy.extended.broadcast_to(
          strategy.experimental_local_results(value)[0],
          destinations=destinations)

    return distribute_lib.get_replica_context().merge_call(
        merge_fn, args=(value,))

  else:
    reduce_op = reduce_util.ReduceOp.from_variable_aggregation(aggregation)
    aggregated_value = distribute_lib.get_strategy(  # pylint: disable=protected-access
    ).extended._replica_ctx_all_reduce(reduce_op, value)
    return aggregated_value


class DistributedValues(ds_types.DistributedValues):
  # [CLASS DOC]
  # Base class for a collection of per-replica component values stored as a tuple, one entry
  # per replica or device. It offers lookup of the component for the current replica, a
  # representative "primary" component, and readable string forms. Subclasses decide what a
  # cross-replica access means.
  """Base class for representing distributed values."""

  def __init__(self, values):
    # [FUNCTION DOC]
    # Constructor. Takes values, any iterable of component values, and stores it as a tuple in
    # the internal values attribute. Intended to be called only from subclass constructors.
    """Should only be called by subclass __init__."""
    self._values = tuple(values)

  def _get(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it raises a value error, but this base implementation
    # raises nothing itself; in cross-replica context it calls the cross-replica hook, which in
    # this base class raises a not-implemented error.
    # Returns the component for the current replica: if there is no current replica id (cross-
    # replica context) it returns the result of the cross-replica hook, otherwise the component
    # at the current replica's index.
    """Returns the value for the current device or raises a ValueError."""
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      return self._get_cross_replica()
    else:
      return self._values[replica_id]

  def _get_cross_replica(self):
    # [FUNCTION DOC]
    # Hook for subclasses that support access from cross-replica context. The base version
    # always raises a not-implemented error that mentions the concrete type name.
    raise NotImplementedError(
        "DistributedValues._get_cross_replica should be implemented by "
        "sub-classes which support cross-replica accesses. "
        f"Type name is {type(self)}"
    )

  def _get_on_device_or_primary(self):
    # [FUNCTION DOC]
    # Returns the component belonging to the current replica if there is a current replica id.
    # Otherwise it searches for a component whose device matches the current device (after
    # canonicalizing both names) and returns it, falling back to the primary component when no
    # device matches.
    """Returns value in same replica or device if possible, else the _primary."""
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      # Try to find a value on the current device.
      current_device = device_util.canonicalize(device_util.current())
      for value in self._values:
        if device_util.canonicalize(value.device) == current_device:
          return value
      return self._primary
    else:
      return self._values[replica_id]

  @property
  def _primary(self):
    # [FUNCTION DOC]
    # Property returning the first component value, used as the representative component.
    """Returns a representative component."""
    return self._values[0]

  @property
  def _devices(self):
    # [FUNCTION DOC]
    # Property returning a tuple with the device name of every component, in replica order.
    return tuple(v.device for v in self._values)

  def __str__(self):
    # [FUNCTION DOC]
    # Human-readable text: the class name followed by a braced, multi-line list of the replica
    # index and the plain string form of each component.
    debug_str = ",\n".join(
        "  %d: %s" % (i, v) for i, v in enumerate(self._values))
    return "%s:{\n%s\n}" % (self.__class__.__name__, debug_str)

  def __repr__(self):
    # [FUNCTION DOC]
    # Debug text: the class name followed by a braced, multi-line list of the replica index and
    # the representation of each component.
    debug_repr = ",\n".join(
        "  %d: %r" % (i, v) for i, v in enumerate(self._values))
    return "%s:{\n%s\n}" % (self.__class__.__name__, debug_repr)


# NOTE(josh11b,apassos): It would be great if we could inspect the values this was
# initialized with and use that to generate the overloaded operators here.
# Unfortunately, Python's rules for special methods don't allow this, see
# https://docs.python.org/3/reference/datamodel.html#special-method-names
# "if a class defines a method named __getitem__(), and x is an instance of
# this class, then x[i] is roughly equivalent to type(x).__getitem__(x, i)."
# In particular, these special methods don't go through __getattr__, and
# it will only use those methods if they are defined in the class, not the
# object.
class DistributedDelegate(DistributedValues):
  # [CLASS DOC]
  # Wrapper that makes a set of per-replica values behave like one of the values: unknown
  # attribute lookups and the arithmetic, comparison, bitwise and indexing operators are
  # forwarded to the component for the current replica (or the operand form supplied by
  # subclasses). Base of Mirrored and DistributedVariable. Operators must be defined explicitly
  # on the class because Python looks special methods up on the type, not through attribute
  # fallback.
  """A map from device to values; acts as the same type as the values."""

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # Fallback attribute lookup, invoked only when normal lookup fails. Parameter: name, the
    # attribute name. Names starting with a private "self" prefix, and a few bookkeeping names
    # used while restoring saved models and tracking layers, are NOT forwarded: they are passed
    # to the parent lookup (which has no such fallback, so an attribute error results). The
    # internal values name raises an attribute error directly to avoid infinite recursion when
    # an object is copied without running its constructor. Every other name is looked up on the
    # component for the current replica and its value returned.
    # The '_use_resource_variables' and the attrs starts with '_self' are used
    # for restoring the saved_model proto, and '_attribute_sentinel' is used for
    # Layer tracking. At the point these attrs are queried, the variable has not
    # been initialized. Thus it should not query those of the underlying
    # components.
    if name.startswith("_self_") or name in ("_use_resource_variables",
                                             "_attribute_sentinel",
                                             "_distributed_container"):
      return super(DistributedDelegate, self).__getattr__(name)

    # This allows copy.copy(DistributedDelegate). When copying an object,
    # copy.copy doesn't invoke its __init__ method, instead it makes a new
    # empty object, then copies the attributes over. copy.copy looks for
    # attributes like "__getstate__" in case the object implements its custom
    # copying. Since DistributedDelegate doesn't have those attributes defined,
    # __getattr__ will be invoked, which tries to access "_values" attributes,
    # but that doesn't exist either because this is an empty object, and again
    # __getattr__ is invoked, leading to an infinite recursion.
    if name == "_values":
      raise AttributeError()

    # TODO(priyag): This needs to be made robust against pitfalls from mix use
    # __getattr__ and @property. See b/120402273.
    return getattr(self._get(), name)

  @property
  def values(self):
    # [FUNCTION DOC]
    # Property returning the tuple of per-replica component values.
    """Returns the per replica values."""
    return self._values

  def _get_as_operand(self):
    # [FUNCTION DOC]
    # HELPER: used by all forwarded operators below to fetch the value to compute with.
    # Returns the component for the current replica; subclasses may override it to return
    # something usable in operations when the component type is not directly accessible.
    """Returns the value for operations for the current device.

    Some implementations, e.g. `TPUMirroredVariable`, are not able to return the
    value type within a replica context. They can, however, return a value that
    can be used by the operations below.
    """
    return self._get()

  # pylint: disable=multiple-statements
  def __add__(self, o):
    # [FUNCTION DOC]
    # Addition operator: adds the other operand o to the operand-form of this value (left side).
    return self._get_as_operand() + o

  def __radd__(self, o):
    # [FUNCTION DOC]
    # Reflected addition: adds the operand-form of this value to o (this value on the right).
    return o + self._get_as_operand()

  def __sub__(self, o):
    # [FUNCTION DOC]
    # Subtraction operator: operand-form of this value minus o.
    return self._get_as_operand() - o

  def __rsub__(self, o):
    # [FUNCTION DOC]
    # Reflected subtraction: o minus the operand-form of this value.
    return o - self._get_as_operand()

  def __mul__(self, o):
    # [FUNCTION DOC]
    # Multiplication operator: operand-form of this value times o.
    return self._get_as_operand() * o

  def __rmul__(self, o):
    # [FUNCTION DOC]
    # Reflected multiplication: o times the operand-form of this value.
    return o * self._get_as_operand()

  def __truediv__(self, o):
    # [FUNCTION DOC]
    # True division operator: operand-form of this value divided by o.
    return self._get_as_operand() / o

  def __rtruediv__(self, o):
    # [FUNCTION DOC]
    # Reflected true division: o divided by the operand-form of this value.
    return o / self._get_as_operand()

  def __floordiv__(self, o):
    # [FUNCTION DOC]
    # Floor division operator: operand-form of this value floor-divided by o.
    return self._get_as_operand() // o

  def __rfloordiv__(self, o):
    # [FUNCTION DOC]
    # Reflected floor division: o floor-divided by the operand-form of this value.
    return o // self._get_as_operand()

  def __mod__(self, o):
    # [FUNCTION DOC]
    # Modulo operator: remainder of the operand-form of this value divided by o.
    return self._get_as_operand() % o

  def __rmod__(self, o):
    # [FUNCTION DOC]
    # Reflected modulo: remainder of o divided by the operand-form of this value.
    return o % self._get_as_operand()

  def __lt__(self, o):
    # [FUNCTION DOC]
    # Less-than comparison of the operand-form of this value against o.
    return self._get_as_operand() < o

  def __le__(self, o):
    # [FUNCTION DOC]
    # Less-than-or-equal comparison of the operand-form of this value against o.
    return self._get_as_operand() <= o

  def __gt__(self, o):
    # [FUNCTION DOC]
    # Greater-than comparison of the operand-form of this value against o.
    return self._get_as_operand() > o

  def __ge__(self, o):
    # [FUNCTION DOC]
    # Greater-than-or-equal comparison of the operand-form of this value against o.
    return self._get_as_operand() >= o

  def __and__(self, o):
    # [FUNCTION DOC]
    # Bitwise/logical and: operand-form of this value combined with o.
    return self._get_as_operand() & o

  def __rand__(self, o):
    # [FUNCTION DOC]
    # Reflected bitwise/logical and: o combined with the operand-form of this value.
    return o & self._get_as_operand()

  def __or__(self, o):
    # [FUNCTION DOC]
    # Bitwise/logical or: operand-form of this value combined with o.
    return self._get_as_operand() | o

  def __ror__(self, o):
    # [FUNCTION DOC]
    # Reflected bitwise/logical or: o combined with the operand-form of this value.
    return o | self._get_as_operand()

  def __xor__(self, o):
    # [FUNCTION DOC]
    # Bitwise exclusive-or: operand-form of this value combined with o.
    return self._get_as_operand() ^ o

  def __rxor__(self, o):
    # [FUNCTION DOC]
    # Reflected bitwise exclusive-or: o combined with the operand-form of this value.
    return o ^ self._get_as_operand()

  def __getitem__(self, o):
    # [FUNCTION DOC]
    # Indexing/slicing: applies the index or slice o to the operand-form of this value and
    # returns the result.
    return self._get_as_operand()[o]

  def __pow__(self, o, modulo=None):
    # [FUNCTION DOC]
    # Power operator: raises the operand-form of this value to the exponent o, with an optional
    # modulo (default none) passed to the built-in power function.
    return pow(self._get_as_operand(), o, modulo)

  def __rpow__(self, o):
    # [FUNCTION DOC]
    # Reflected power: raises o to the power of the operand-form of this value.
    return pow(o, self._get_as_operand())

  def __invert__(self):
    # [FUNCTION DOC]
    # Bitwise inversion of the operand-form of this value.
    return ~self._get_as_operand()

  def __neg__(self):
    # [FUNCTION DOC]
    # Numeric negation of the operand-form of this value.
    return -self._get_as_operand()

  def __abs__(self):
    # [FUNCTION DOC]
    # Absolute value of the operand-form of this value.
    return abs(self._get_as_operand())

  def __div__(self, o):
    # [FUNCTION DOC]
    # KNOWN ISSUE: legacy Python 2 division hook; Python 3 never calls it (it uses true
    # division), so this is effectively dead code.
    # Tries to call the legacy division method of the operand-form of this value with o; if that
    # method does not exist it returns the not-implemented marker so Python can try other
    # options.
    try:
      return self._get_as_operand().__div__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rdiv__(self, o):
    # [FUNCTION DOC]
    # KNOWN ISSUE: legacy Python 2 reflected division hook; Python 3 never calls it, so this is
    # effectively dead code.
    # Tries to call the legacy reflected division method of the operand-form of this value with
    # o; if that method does not exist it returns the not-implemented marker.
    try:
      return self._get_as_operand().__rdiv__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __matmul__(self, o):
    # [FUNCTION DOC]
    # Matrix multiplication operator: calls the matrix-multiply method of the operand-form of
    # this value with o; if that method does not exist it returns the not-implemented marker.
    try:
      return self._get_as_operand().__matmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  def __rmatmul__(self, o):
    # [FUNCTION DOC]
    # Reflected matrix multiplication: calls the reflected matrix-multiply method of the
    # operand-form of this value with o; if missing it returns the not-implemented marker.
    try:
      return self._get_as_operand().__rmatmul__(o)
    except AttributeError:
      # See https://docs.python.org/3/library/constants.html#NotImplemented
      return NotImplemented

  # TODO(josh11b): Even more operator overloads.


class PerReplica(DistributedValues, composite_tensor.CompositeTensor,
                 ds_types.PerReplica):
  # [CLASS DOC]
  # Holds one independent (unsynchronized) value per replica. Unlike Mirrored it has no
  # cross-replica view and cannot be used directly in operators. It is a composite tensor whose
  # type spec is a PerReplicaSpec built from the specs of the components. Produced for example
  # by running a function on all replicas or by distributing datasets.
  """Holds a map from replica to unsynchronized values."""

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Property returning the type specification: a PerReplicaSpec built from the type spec of
    # each component value.
    return PerReplicaSpec(
        *(type_spec.type_spec_from_value(v) for v in self._values))

  @property
  def values(self):
    # [FUNCTION DOC]
    # Property returning the tuple of per-replica values.
    """Returns the per replica values."""
    return self._values


def _per_replica_to_tensor(var, dtype=None, name=None, as_ref=False):
  # [FUNCTION DOC]
  # HELPER: registered as the tensor conversion function for PerReplica to give a useful error.
  # KNOWN ISSUE: when a dtype is requested, the check reads a dtype attribute from the PerReplica
  # object, but PerReplica is not a delegate and defines no such attribute, so that path would
  # fail with an attribute error instead of the intended value error.
  # Parameters: var (a PerReplica), dtype (optional requested type), name (ignored), as_ref
  # (must be false). Raises a value error if dtype is incompatible with the variable's dtype,
  # a not-implemented error if as_ref is true, and a value error if called outside replica
  # context (cross-replica context or no strategy). Inside replica context it returns the
  # component belonging to the current replica.
  """Converts a `PerReplica` to a `Tensor`."""
  del name
  if dtype is not None and not dtype.is_compatible_with(var.dtype):
    raise ValueError(
        "Incompatible type conversion requested to type {!r} for variable "
        "of type {!r}".format(dtype.name, var.dtype.name))
  if as_ref:
    raise NotImplementedError(
        "PerReplica doesn't support being used as a reference.")
  if (distribute_lib.in_cross_replica_context() or
      not distribute_lib.has_strategy()):
    raise ValueError("It looks like you are using a PerReplica object while "
                     "not inside a replica context, which is not supported. "
                     "Try running your op or function inside a replica context "
                     "by using `strategy.run`")
  else:
    replica_id = values_util.get_current_replica_id_as_int()
    return var.values[replica_id]

# Register a conversion function to provide a useful error message when users
# try to use PerReplica values in the wrong contexts
tensor_conversion_registry.register_tensor_conversion_function(
    PerReplica, _per_replica_to_tensor)


class PerReplicaSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # Type specification of a PerReplica: holds one component type spec per replica. Used for
  # tracing and for flattening a PerReplica into its component values. Registered with the
  # nested-structure coder below so it can be serialized.
  """Type specification for a `PerReplica`."""

  __slots__ = ["_value_specs"]

  value_type = property(lambda self: PerReplica)

  def __init__(self, *value_specs):
    # [FUNCTION DOC]
    # Constructor. Takes any number of component type specs (one per replica) and stores them
    # as a tuple.
    self._value_specs = tuple(value_specs)

  def _serialize(self):
    # [FUNCTION DOC]
    # Returns the tuple of component specs as the serialized form of this spec.
    return self._value_specs

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Property returning the tuple of component type specs.
    return self._value_specs

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Flattens a PerReplica value into its tuple of component values. Raises a value error if
    # called inside a replica context that has more than one replica in sync. Returns the
    # internal tuple of values of the given value.
    replica_context = distribute_lib.get_replica_context()
    if replica_context is not None and replica_context.num_replicas_in_sync > 1:
      raise ValueError(
          "Flattening a PerReplica to components is not supported in replica "
          "context.")
    return value._values  # pylint: disable=protected-access

  def _from_components(self, tensor_list):
    # [FUNCTION DOC]
    # Rebuilds a PerReplica from a sequence of component values (tensor_list) and returns it.
    return PerReplica(tensor_list)


nested_structure_coder.register_codec(
    nested_structure_coder.BuiltInTypeSpecCodec(
        PerReplicaSpec, struct_pb2.TypeSpecProto.PER_REPLICA_SPEC
    )
)


# Note that unlike PerReplica, Mirrored values inherit from
# DistributedDelegate and so can be used directly in cross-replica mode.
# TODO(tomhennigan) Should this extend CompositeTensor?
class Mirrored(DistributedDelegate, ds_types.Mirrored):
  # [CLASS DOC]
  # Holds one value per replica where all values are kept identical. Because it derives from
  # DistributedDelegate it can also be used directly in cross-replica context, where it
  # resolves to the component on the current device (or the primary one).
  """Holds a map from replica to values which are kept in sync."""

  def _get_cross_replica(self):
    # [FUNCTION DOC]
    # Cross-replica access: returns the component on the current device, or the primary one.
    return self._get_on_device_or_primary()

  def _as_graph_element(self):
    # [FUNCTION DOC]
    # Returns the graph element for the current component: if that component has a callable
    # graph-element conversion, its result is returned, otherwise the component itself.
    obj = self._get()
    conv_fn = getattr(obj, "_as_graph_element", None)
    if conv_fn and callable(conv_fn):
      return conv_fn()
    return obj

  def _is_mirrored(self):
    # [FUNCTION DOC]
    # Always returns true: values of this class are mirrored.
    return True


class DistributedVarOp(object):
  # [CLASS DOC]
  # Lightweight stand-in that looks like an operation, exposing name, graph, traceback and
  # type of the primary component's operation. Used as the "op" of distributed variables in
  # cross-replica context. Supports equality and hashing on those four fields.
  """A class that looks like `tf.Operation`."""

  def __init__(self, name, graph, traceback, typ):
    # [FUNCTION DOC]
    # Constructor. Stores the operation name, its graph, its traceback (sequence of frames) and
    # its type name (typ) as public attributes.
    self.name = name
    self.graph = graph
    self.traceback = traceback
    self.type = typ

  def __eq__(self, o):
    # [FUNCTION DOC]
    # KNOWN ISSUE: comparing with an object of a different class raises a not-implemented error
    # instead of returning false or the not-implemented marker, so equality checks against
    # arbitrary objects (for example inside containers) can raise.
    # Returns true when name, graph, traceback and type all match those of o.
    if not isinstance(o, self.__class__):
      raise NotImplementedError
    return (self.name == o.name and self.graph == o.graph and
            self.traceback == o.traceback and self.type == o.type)

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash combining name, graph, the traceback converted to a tuple, and type.
    return hash((self.name, self.graph, tuple(self.traceback), self.type))


# TODO(b/209081027): Remove this once Variable is a CompositeTensor.
class DistributedVariableTraceType(trace.TraceType):
  # [CLASS DOC]
  # Trace type used when a distributed variable is an argument of a traced function. Two trace
  # types are equal when the variables have the same shape and dtype. Holds the variable and a
  # components pair (shape as a tuple, dtype).
  """TraceType of DistributedVariable objects."""

  def __init__(self, distributed_variable):
    # [FUNCTION DOC]
    # Constructor. Stores the distributed variable and a components pair made of its shape
    # (as a tuple of dimensions) and its dtype, which defines equality and hashing.
    self.distributed_variable = distributed_variable
    self.components = (tuple(distributed_variable.shape.as_list()),
                       distributed_variable.dtype)

  def is_subtype_of(self, other):
    # [FUNCTION DOC]
    # A trace type is a subtype only of an equal one: returns whether self equals other.
    return self == other

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # Returns this trace type if it equals every one in the others collection, otherwise
    # nothing (null value), meaning there is no common supertype.
    return self if all(self == other for other in others) else None

  def placeholder_value(self, placeholder_context=None):
    # [FUNCTION DOC]
    # Returns the distributed variable itself as its placeholder; the optional placeholder
    # context is ignored.
    return self.distributed_variable

  def to_tensors(self, value):
    # [FUNCTION DOC]
    # Returns an empty list: a distributed variable contributes no tensors to flattening.
    return []

  def cast(self, value, _):
    # [FUNCTION DOC]
    # Returns the value unchanged; the second (casting context) argument is ignored.
    return value

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Hash of the components pair (shape and dtype).
    return hash(self.components)

  def __eq__(self, other) -> bool:
    # [FUNCTION DOC]
    # Equal only to another DistributedVariableTraceType with equal shape and dtype; other
    # types give false.
    if not isinstance(other, DistributedVariableTraceType):
      return False

    return self.components == other.components


class DistributedVariable(DistributedDelegate, variables_lib.Variable,
                          core.Tensor):
  # [CLASS DOC]
  # A variable made of one component variable per replica/device, behaving as a single
  # tf.Variable. It records the owning strategy, the aggregation mode, an optional packed
  # variable (one handle for all components, eager mode only) and an optional variable policy.
  # Reads and writes are delegated to the policy when present (or to utility functions);
  # while saving non-distributed, everything is redirected to the primary component. Each
  # component gets a weak reference back to this container. Subclasses: MirroredVariable,
  # SyncOnReadVariable and the TPU variants.
  """Holds a map from replica to variables."""

  def __init__(self, strategy, values, aggregation, var_policy=None):
    # [FUNCTION DOC]
    # Constructor. Parameters: strategy (the distribution strategy that owns the variable),
    # values (sequence of component variables, one per replica), aggregation (variable
    # aggregation mode), var_policy (optional policy object deciding synchronization behavior).
    # Raises a value error when mean aggregation is combined with a non-floating dtype. Stores
    # strategy and aggregation, initializes the base values and the common name (first
    # component's name without the output index), gives each component a weak reference to this
    # container (stored on the component's handle for composite-tensor resource variables),
    # creates a packed variable when running eagerly and the strategy enables packing (for
    # nested components, packs their inner variables and attaches the packed variable to each
    # component instead), and initializes the keras-initialized flag to false, the initializer
    # operation to nothing and the policy.
    if (aggregation == variables_lib.VariableAggregation.MEAN and
        not values[0].dtype.is_floating):
      raise ValueError(
          "creating distributed tf.Variable with aggregation=MEAN and a "
          "non-floating dtype is not supported, please use a different "
          "aggregation or dtype")
    self._distribute_strategy = strategy
    self._aggregation = aggregation
    super(DistributedVariable, self).__init__(values)
    self._common_name = self._primary.name.split(":")[0]
    # Use a weakref to make it easy to map from the contained values
    # to the container without introducing a reference cycle.
    for v in values:
      # ResourceVariable is a CompositeTensor. Attributes added to
      # CompositeTensors will get lost through tf.nest packing and unpacking.
      if isinstance(v, composite_tensor.CompositeTensor) and hasattr(
          v, "handle"):
        v.handle._distributed_container = weakref.ref(self)  # pylint: disable=protected-access
      else:
        v._distributed_container = weakref.ref(self)  # pylint: disable=protected-access

    # Packed variable is used to reduce the overhead of function execution.
    # For a DistributedVariable, only one variable handle is captured into a
    # function graph. It's only supported in eager mode.
    if ops.executing_eagerly_outside_functions() and getattr(
        strategy, "_enable_packed_variable_in_eager_mode", False):
      name = "%s/packed/" % self._common_name
      if hasattr(values[0], "_vars"):
        # Handle when the resource variables are "nested" underneath another
        # layer of values, e.g., TPUReplicatedVariable, by packing all them
        # together and pushing the packed var down a level
        # pylint: disable=protected-access
        packed_var = packed.PackedDistributedVariable(
            sum((value._vars for value in values), []), name=name)
        for value in values:
          value._packed_var = packed_var
        self._packed_var = None
        # pylint: enable=protected-access
      else:
        self._packed_var = packed.PackedDistributedVariable(values, name=name)
    else:
      self._packed_var = None

    # tf.keras keeps track of variables initialized using this attribute. When
    # tf.keras gets the default session, it initializes all uninitialized vars.
    # We need to make _keras_initialized a member of DistributedVariable because
    # without this it will use `__getattr__` which will delegate to a component
    # variable.
    self._keras_initialized = False
    # Typically, a `DistributedVariable`'s initializer is composed of the
    # initializers of the components variables. However, in some cases, such as
    # when restoring from a checkpoint, we may set the _initializer_op
    # property on the entire `DistributedVariable`.
    self._initializer_op = None
    # Set a VariablePolicy which decides how we replicate/aggregate the given
    # variable.
    self._policy = var_policy

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Deep copy hook. Parameter: memo, the copy memoization mapping. Inside the originating
    # strategy's scope (asserted or entered), copies every component on its own device, then
    # builds a new instance of the same class with the same strategy and aggregation and a deep
    # copy of the policy, records it in memo, and returns it. Raises a runtime error (from the
    # strategy scope assertion) when invoked under a different strategy.
    """Perform a deepcopy of the `DistributedVariable`.

    Unlike the deepcopy of a regular tf.Variable, this keeps the original
    strategy and devices of the `DistributedVariable`.  To avoid confusion
    with the behavior of deepcopy on a regular `Variable` (which does
    copy into new devices), we only allow a deepcopy of a `DistributedVariable`
    within its originating strategy scope.

    Args:
      memo: The memoization object for `deepcopy`.

    Returns:
      A deep copy of the current `DistributedVariable`.

    Raises:
      RuntimeError: If trying to deepcopy into a different strategy.
    """
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      new_values = []

      for value in self._values:
        with ops.device(value.device):
          new_values.append(copy.deepcopy(value, memo))

    copied_variable = type(self)(
        strategy=self._distribute_strategy,
        values=new_values,
        aggregation=self._aggregation,
        var_policy=copy.deepcopy(self._policy, memo))

    memo[id(self)] = copied_variable

    return copied_variable

  def _use_packed_variable(self):
    # [FUNCTION DOC]
    # HELPER: decides whether the packed variable should be used. Returns true when a packed
    # variable exists and the code is not saving in non-distributed mode (to avoid explicit
    # device placement on consuming ops).
    # Don't use packed variable when under a SaveContext to avoid explicit
    # device placement on variable consuming ops.
    return self._packed_var is not None and (
        not values_util.is_saving_non_distributed())

  def is_initialized(self, name=None):
    # [FUNCTION DOC]
    # Returns an op that is true when all component variables are initialized. Parameter: name,
    # optional name for the final logical-and operation. When saving non-distributed it returns
    # the primary's check; with a packed variable it returns the packed variable's check;
    # otherwise it chains logical-and operations over all components, naming the last one.
    """Identifies if all the component variables are initialized.

    Args:
      name: Name of the final `logical_and` op.

    Returns:
      The op that evaluates to True or False depending on if all the
      component variables are initialized.
    """
    if values_util.is_saving_non_distributed():
      return self._primary.is_initialized()
    if self._use_packed_variable():
      return self._packed_var.is_initialized()
    result = self._primary.is_initialized()
    # We iterate through the list of values except the last one to allow us to
    # name the final `logical_and` op the same name that is passed by the user
    # to the `is_initialized` op. For distributed variables, the
    # `is_initialized` op is a `logical_and` op.
    for v in self._values[1:-1]:
      result = math_ops.logical_and(result, v.is_initialized())
    result = math_ops.logical_and(
        result, self._values[-1].is_initialized(), name=name)
    return result

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Property returning the initializer operation. Primary's initializer when saving
    # non-distributed; else the explicitly set initializer operation if one exists; else a
    # grouped operation of all component initializers.
    if values_util.is_saving_non_distributed():
      return self._primary.initializer
    if self._initializer_op:
      init_op = self._initializer_op
    else:
      # return grouped ops of all the var initializations of component values of
      # the mirrored variable
      init_op = control_flow_ops.group(
          tuple(v.initializer for v in self._values))
    return init_op

  def initialized_value(self):
    # [FUNCTION DOC]
    # Returns the initialized value of the component on the current device (or the primary).
    return self._get_on_device_or_primary().initialized_value()

  def _is_mirrored(self):
    # [FUNCTION DOC]
    # Returns true only when a policy is set and that policy reports being mirrored.
    return (self._policy is not None) and (self._policy._is_mirrored())  # pylint: disable=protected-access

  @property
  def initial_value(self):
    # [FUNCTION DOC]
    # Property returning the initial value of the component on the current device (or primary).
    return self._get_on_device_or_primary().initial_value

  @property
  def constraint(self):
    # [FUNCTION DOC]
    # Property returning the primary component's constraint.
    return self._primary.constraint

  @property
  def graph(self):
    # [FUNCTION DOC]
    # Property returning the graph of the primary component.
    return self._primary.graph

  @property
  def _shared_name(self):
    # [FUNCTION DOC]
    # Property returning the common name (primary component name without output index).
    return self._common_name

  @property
  def _unique_id(self):
    # [FUNCTION DOC]
    # Property returning the primary component's unique identifier.
    return self._primary._unique_id  # pylint: disable=protected-access

  @property
  def _graph_key(self):
    # [FUNCTION DOC]
    # Property returning the primary component's graph key, so optimizers can tell which graph
    # the variable belongs to.
    """Lets Optimizers know which graph this variable is from."""
    return self._primary._graph_key  # pylint: disable=protected-access

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the primary component's full name.
    return self._primary.name

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the primary component's element type.
    return self._primary.dtype

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property returning the primary component's shape.
    return self._primary.shape

  @property
  def synchronization(self):
    # [FUNCTION DOC]
    # Property returning the primary component's synchronization mode.
    return self._primary.synchronization

  @property
  def aggregation(self):
    # [FUNCTION DOC]
    # Property returning the aggregation mode given at construction (stored on this container,
    # not read from the components).
    return self._aggregation

  @property
  def _packed_variable(self):
    # [FUNCTION DOC]
    # Property returning the packed variable when it should be used, otherwise nothing.
    if self._use_packed_variable():
      return self._packed_var
    return None

  @property
  def handle(self):
    # [FUNCTION DOC]
    # Property returning the resource handle. When saving non-distributed it is the primary's
    # handle. Raises a value error outside replica context (and outside an update call), since
    # no single handle exists there. In replica context returns the packed variable's handle if
    # packing is in use, otherwise the handle of the current replica's component.
    if values_util.is_saving_non_distributed():
      return self._primary.handle
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      raise ValueError(
          "DistributedVariable.handle is not available outside the replica "
          "context or a `tf.distribute.Strategy.update()` call.")
    else:
      if self._use_packed_variable():
        return self._packed_var.handle
      return self._values[replica_id].handle

  def eval(self, session=None):
    # [FUNCTION DOC]
    # Evaluates the variable in a session (optional; default session when none) using the
    # component on the current device or the primary. Returns the evaluated value.
    return self._get_on_device_or_primary().eval(session)

  @property
  def _save_slice_info(self):
    # [FUNCTION DOC]
    # Property returning the primary component's save-slice information.
    return self._primary._save_slice_info  # pylint: disable=protected-access

  def _get_save_slice_info(self):
    # [FUNCTION DOC]
    # Returns the primary component's save-slice information (via its getter).
    return self._primary._get_save_slice_info()  # pylint: disable=protected-access

  def _set_save_slice_info(self, save_slice_info):
    # [FUNCTION DOC]
    # Sets the given save-slice information on every component variable. Returns nothing.
    for v in self._values:
      v._set_save_slice_info(save_slice_info)  # pylint: disable=protected-access

  @property
  def device(self):
    # [FUNCTION DOC]
    # Property returning the device of the component on the current device (or the primary).
    return self._get_on_device_or_primary().device

  @property
  def trainable(self):
    # [FUNCTION DOC]
    # Property returning whether the primary component is trainable.
    return self._primary.trainable

  @property
  def distribute_strategy(self):
    # [FUNCTION DOC]
    # Property returning the distribution strategy that created this variable.
    return self._distribute_strategy

  def get_shape(self) -> tensor_shape.TensorShape:
    # [FUNCTION DOC]
    # Returns the primary component's static tensor shape.
    return self._primary.get_shape()

  def to_proto(self, export_scope=None):
    # [FUNCTION DOC]
    # Serializes the primary component to its variable protocol-buffer message. Parameter:
    # export_scope, optional name scope to strip. Returns the message.
    return self._primary.to_proto(export_scope=export_scope)

  @property
  def op(self) -> ops.Operation:
    # [FUNCTION DOC]
    # Property returning the operation associated with the variable. Primary's op when saving
    # non-distributed; in cross-replica context a DistributedVarOp built from the primary op's
    # name, graph, traceback and type (so reading attributes works from any device); in replica
    # context the op of the current replica's component.
    if values_util.is_saving_non_distributed():
      return self._primary.op
    # We want cross-replica code that does some var.op.X calls
    # to work (even if the current device isn't in self._devices), but
    # other uses of var.op in a cross-replica context to fail.
    if distribute_lib.in_cross_replica_context():
      return DistributedVarOp(self._primary.op.name, self._primary.op.graph,
                              self._primary.op.traceback, self._primary.op.type)
    return self._get().op

  @property
  def _in_graph_mode(self):
    # [FUNCTION DOC]
    # Property returning whether the primary component lives in a graph (not eager).
    return self._primary._in_graph_mode  # pylint: disable=protected-access

  def _get_replica(self, replica_id):
    # [FUNCTION DOC]
    # Returns the component for the given replica index; if the packed variable is in use, it
    # returns the packed variable's view for that component's device instead.
    """Returns the value on a device with the given replica_id."""
    value = self._values[replica_id]
    if self._use_packed_variable():
      return self._packed_var.on_device(value.device)
    else:
      return value

  def _get(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring claims a value error is raised, but this method raises nothing
    # itself; in cross-replica context it defers to the cross-replica hook, which may raise a
    # not-implemented error when no policy is set.
    # Returns the primary when saving non-distributed; in cross-replica context the
    # cross-replica hook's value; otherwise the current replica's component (packed view if
    # packing is in use).
    """Returns the value for the current device or raises a ValueError."""
    if values_util.is_saving_non_distributed():
      return self._primary
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      return self._get_cross_replica()
    else:
      return self._get_replica(replica_id)

  def _get_on_device_or_primary(self):
    # [FUNCTION DOC]
    # Returns the primary when saving non-distributed. With a current replica id it returns that
    # replica's component. Otherwise it returns the component whose canonical device equals the
    # current device, falling back to the first component (index zero); packed views are used
    # when packing is enabled.
    """Returns value in same replica or device if possible, else the _primary."""
    if values_util.is_saving_non_distributed():
      return self._primary
    replica_id = values_util.get_current_replica_id_as_int()
    if replica_id is None:
      # Try to find a value on the current device.
      current_device = device_util.canonicalize(device_util.current())
      for i, value in enumerate(self._values):
        if device_util.canonicalize(value.device) == current_device:
          return self._get_replica(i)
      return self._get_replica(0)
    else:
      return self._get_replica(replica_id)

  def read_value(self):
    # [FUNCTION DOC]
    # Returns the current value as a tensor. Primary's value when saving non-distributed;
    # otherwise an identity of the current component (resolved through the cross-replica hook
    # in cross-replica context), computed inside the variable's strategy scope.
    if values_util.is_saving_non_distributed():
      return self._primary.read_value()
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      return array_ops.identity(self._get())

  def value(self):
    # [FUNCTION DOC]
    # Returns the variable's value. Primary's value when saving non-distributed; the policy's
    # value computation when a policy is set; otherwise the value of the component on the
    # current device or primary.
    if values_util.is_saving_non_distributed():
      return self._primary.value()
    if self._policy:
      return self._policy.value(self)
    return self._get_on_device_or_primary().value()

  def numpy(self):
    # [FUNCTION DOC]
    # Returns the value as a numpy array by reading it and converting. Raises a
    # not-implemented error when eager execution is not enabled.
    if context.executing_eagerly():
      return self.read_value().numpy()
    else:
      raise NotImplementedError("DistributedVariable.numpy() is only available "
                                "when eager execution is enabled.")

  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Subtracts value from the variable. Parameters: value, use_locking (default false), name
    # (optional), read_value (default true: return the updated variable, else the operation).
    # Primary's subtraction when saving non-distributed; the policy's subtraction when a policy
    # is set; otherwise the generic on-write subtraction helper.
    if values_util.is_saving_non_distributed():
      return self._primary.assign_sub(value, use_locking, name, read_value)
    if self._policy:
      return self._policy.assign_sub(
          self,
          value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    return values_util.on_write_assign_sub(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Adds value to the variable. Parameters: value, use_locking (default false), name
    # (optional), read_value (default true: return the updated variable, else the operation).
    # Primary's addition when saving non-distributed; the policy's addition when a policy is
    # set; otherwise the generic on-write addition helper.
    if values_util.is_saving_non_distributed():
      return self._primary.assign_add(value, use_locking, name, read_value)
    if self._policy:
      return self._policy.assign_add(
          self,
          value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    return values_util.on_write_assign_add(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Overwrites the variable with value. Parameters: value, use_locking (default false), name
    # (optional), read_value (default true: return the updated variable, else the operation).
    # Primary's assignment when saving non-distributed; the policy's assignment when a policy is
    # set; otherwise the generic on-write assignment helper.
    if values_util.is_saving_non_distributed():
      return self._primary.assign(value, use_locking, name, read_value)
    if self._policy:
      return self._policy.assign(
          self,
          value,
          use_locking=use_locking,
          name=name,
          read_value=read_value)
    return values_util.on_write_assign(
        self, value, use_locking=use_locking, name=name, read_value=read_value)

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Subtracts a sparse (indexed-slices) update. Parameters: sparse_delta, use_locking (default
    # false), name (optional). Primary's scatter when saving non-distributed; policy's scatter
    # when a policy is set; otherwise the generic scatter helper. Returns its result.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_sub(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_sub(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_sub(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Adds a sparse (indexed-slices) update. Parameters: sparse_delta, use_locking (default
    # false), name (optional). Primary's scatter when saving non-distributed; policy's scatter
    # when a policy is set; otherwise the generic scatter helper. Returns its result.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_add(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_add(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_add(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Multiplies by a sparse (indexed-slices) update. Parameters: sparse_delta, use_locking
    # (default false), name (optional). Primary's scatter when saving non-distributed; policy's
    # scatter when a policy is set; otherwise the generic scatter helper. Returns its result.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_mul(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_mul(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_mul(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Divides by a sparse (indexed-slices) update. Parameters: sparse_delta, use_locking
    # (default false), name (optional). Primary's scatter when saving non-distributed; policy's
    # scatter when a policy is set; otherwise the generic scatter helper. Returns its result.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_div(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_div(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_div(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Element-wise minimum with a sparse (indexed-slices) update. Parameters: sparse_delta,
    # use_locking (default false), name (optional). Primary's scatter when saving
    # non-distributed; policy's scatter when a policy is set; otherwise the generic scatter
    # helper. Returns its result.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_min(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_min(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_min(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Element-wise maximum with a sparse (indexed-slices) update. Parameters: sparse_delta,
    # use_locking (default false), name (optional). Primary's scatter when saving
    # non-distributed; policy's scatter when a policy is set; otherwise the generic scatter
    # helper. Returns its result.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_max(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_max(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_max(
        self, sparse_delta, use_locking=use_locking, name=name)

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Overwrites selected slices with a sparse (indexed-slices) update. Parameters:
    # sparse_delta, use_locking (default false), name (optional). Primary's scatter when saving
    # non-distributed; policy's scatter when a policy is set; otherwise the generic scatter
    # helper. Returns its result.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_update(sparse_delta, use_locking, name)
    if self._policy:
      return self._policy.scatter_update(
          self, sparse_delta, use_locking=use_locking, name=name)
    return values_util.scatter_update(
        self, sparse_delta, use_locking=use_locking, name=name)

  def __tf_tracing_type__(self, _):
    # [FUNCTION DOC]
    # Tracing hook: returns a DistributedVariableTraceType for this variable. The argument (a
    # tracing context) is ignored.
    return DistributedVariableTraceType(self)

  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # Checkpointing hook (overrides the trackable method). Returns a mapping with a single entry
    # from the variable-value key to a factory that creates a saveable object for this
    # distributed variable, using the primary component and the given or common name.
    """Overrides Trackable method.

    This allows both name-based and object-based save and restore of
    DistributedVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """

    def _saveable_factory(name=self._common_name):
      # [FUNCTION DOC]
      # HELPER: factory for the checkpoint mapping; takes an optional name (default: the common
      # name) and returns a saveable object wrapping this variable, its primary and the name.
      return _DistributedVariableSaveable(self, self._primary, name)

    return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _as_graph_element(self):
    # [FUNCTION DOC]
    # Returns the graph element for the variable: the primary's when saving non-distributed,
    # the policy's when a policy exists, else raises a not-implemented error saying that a valid
    # policy is required (or a subclass override).
    if values_util.is_saving_non_distributed():
      return self._primary._as_graph_element()  # pylint: disable=protected-access
    if self._policy:
      return self._policy._as_graph_element(self)  # pylint: disable=protected-access

    raise NotImplementedError(
        "DistributedVariable._as_graph_element requires a valid "
        "VariablePolicy. Please set the policy via the `var_policy` argument "
        "in the constructor, or override this method in sub-classes which "
        "support cross-replica accesses. "
        f"Type name is {type(self)}"
    )

  def _get_cross_replica(self):
    # [FUNCTION DOC]
    # Cross-replica access hook. Returns the primary when saving non-distributed, the policy's
    # cross-replica value when a policy exists, else raises a not-implemented error saying a
    # valid policy is required (or a subclass override).
    if values_util.is_saving_non_distributed():
      return self._primary
    if self._policy:
      return self._policy._get_cross_replica(self)  # pylint: disable=protected-access

    raise NotImplementedError(
        "DistributedVariable._get_cross_replica requires a valid "
        "VariablePolicy. Please set the policy via the `var_policy` argument "
        "in the constructor, or override this method in sub-classes which "
        "support cross-replica accesses. "
        f"Type name is {type(self)}"
    )

  def _update_cross_replica(self, update_fn, value, **kwargs):
    # [FUNCTION DOC]
    # Applies an update in cross-replica context. Parameters: update_fn (callable updating a
    # component), value (value to pass), keyword arguments for update_fn. Marks the program as
    # not saveable (side effect) and runs the update on all components through the strategy's
    # extended update, grouped. Returns the update result (variable or operation).
    """Applies updates across replicas.

    Args:
      update_fn: A callable to pass to `strategy.extended.update` to update the
        variable. It should has the same signature as `Variable.assign()`.
      value: value to be passed to `update_fn`.
      **kwargs: remaining arguments to `update_fn`.

    Returns:
      Updated variable or `tf.Operation`.
    """
    values_util.mark_as_unsaveable()
    return self.distribute_strategy.extended.update(
        self, update_fn, args=(value,), kwargs=kwargs, group=True)

  def _update_replica(self, update_fn, value, **kwargs):
    # [FUNCTION DOC]
    # Applies an update in replica context by delegating to the policy. Parameters as for the
    # cross-replica version. Raises a not-implemented error when no policy is set (subclasses
    # may override). Returns the update result.
    """Applies updates in one replica.

    Args:
      update_fn: A callable to update the variable. It should has the same
        signature as `Variable.assign()`.
      value: value to be passed to `update_fn`.
      **kwargs: remaining arguments to `update_fn`.

    Returns:
      Updated variable or `tf.Operation`.
    """
    if self._policy:
      return self._policy._update_replica(self, update_fn, value, **kwargs)  # pylint: disable=protected-access
    raise NotImplementedError(
        "DistributedVariable._update_replica requires a valid VariablePolicy. "
        "Please set the policy via the `var_policy` argument in the "
        "constructor, or override this method in sub-classes which support "
        "cross-replica accesses. "
        f"Type name is {type(self)}"
    )

  def _update(self, update_fn, value, **kwargs):
    # [FUNCTION DOC]
    # Applies an update according to the current context. Parameters: update_fn (callable that
    # updates one component), value (value to pass), keyword arguments for update_fn. When
    # saving non-distributed it updates the primary directly. Otherwise, inside the variable's
    # strategy scope: in cross-replica context, if an update replica id is active it updates
    # just that replica's component, else it performs a cross-replica update on all components;
    # in replica context it asserts the replica context is valid and performs a replica update.
    # Returns the update result (variable or operation).
    """Applies updates depending on the context.

    The method calls `_update_replica` in replica context,
    `_update_cross_replica` in cross replica context, and `update_fn` in update
    context.

    If `read_value` is True, the method returns the updated Variable. If
    `read_value` is False, the method returns the update `tf.Operation`.

    Args:
      update_fn: A callable to pass to `strategy.extended.update` to update the
        variable. It should have the same signature as `Variable.assign()`.
      value: value to be passed to `update_fn`.
      **kwargs: keyword arguments to `update_fn`.

    Returns:
      Updated variable or `tf.Operation`.

    """
    if values_util.is_saving_non_distributed():
      return update_fn(self._primary, value, **kwargs)
    with distribute_lib.enter_or_assert_strategy(self.distribute_strategy):
      if distribute_lib.in_cross_replica_context():
        update_replica_id = distribute_lib.get_update_replica_id()
        if update_replica_id is not None:
          replica_value = self._get_replica(update_replica_id)
          return update_fn(replica_value, value, **kwargs)
        return self._update_cross_replica(update_fn, value, **kwargs)
      else:
        values_util.assert_replica_context(self.distribute_strategy)
        return self._update_replica(update_fn, value, **kwargs)

  def _should_act_as_resource_variable(self):
    # [FUNCTION DOC]
    # HELPER: marker method whose presence makes the resource-variable check succeed for this
    # class; it does nothing and returns nothing.
    """Pass resource_variable_ops.is_resource_variable check."""
    pass

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # Converts the variable to a tensor. Parameters: dtype (optional), name (optional), as_ref
    # (return reference form). Converts the primary when saving non-distributed; otherwise
    # converts the current component inside the strategy scope. Returns the tensor.
    """Converts a variable to a tensor."""
    if values_util.is_saving_non_distributed():
      return ops.convert_to_tensor(
          self._primary, dtype=dtype, name=name, as_ref=as_ref)
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      return ops.convert_to_tensor(
          self._get(), dtype=dtype, name=name, as_ref=as_ref)

  def __tf_tensor__(self,
                    dtype: Optional[dtypes.DType] = None,
                    name: Optional[str] = None) -> tensor_lib.Tensor:
    # [FUNCTION DOC]
    # Tensor-conversion protocol hook: converts the variable to a tensor with an optional
    # requested element type and optional name by calling the dense conversion. Returns the
    # tensor.
    return self._dense_var_to_tensor(dtype, name)

  def _export_to_saved_model_graph(self,
                                   object_map=None,
                                   tensor_map=None,
                                   options=None,
                                   **kwargs):
    # [FUNCTION DOC]
    # SavedModel export hook. Parameters: object_map and tensor_map (mappings being filled to
    # relate original objects/tensors to their exported counterparts), options (save options,
    # whose variable policy decides whether components are expanded), extra keyword arguments
    # passed through. Exports the primary first, then for each other component either exports it
    # too (when the policy expands distributed variables) or maps it and its handle onto the
    # primary's. Also maps this variable, and the packed handle when present, to the primary's
    # entries. Returns the list of resources (handles and this variable) that were produced.
    # Initialize for self._primary first, so that obj_map[self._primary] and
    # resource_map[self._primary.handle] contain mapped values.
    resource_list = self._primary._export_to_saved_model_graph(  # pylint:disable=protected-access
        object_map=object_map,
        tensor_map=tensor_map,
        options=options,
        **kwargs)
    for v in [v for v in self._values if v != self._primary]:
      if (options.experimental_variable_policy  # pylint:disable=protected-access
          ._expand_distributed_variables()):
        resource_list.extend(
            v._export_to_saved_model_graph(  # pylint:disable=protected-access
                object_map=object_map,
                tensor_map=tensor_map,
                options=options,
                **kwargs))  # pylint:disable=protected-access
      else:
        object_map[v] = object_map[self._primary]
        tensor_map[v.handle] = tensor_map[self._primary.handle]
        resource_list.append(v.handle)
    object_map[self] = object_map[self._primary]
    tensor_map[self] = tensor_map[self._primary.handle]
    resource_list.append(self)
    if self._packed_var is not None:
      tensor_map[self._packed_var.packed_handle] = tensor_map[
          self._primary.handle]
      resource_list.append(self._packed_var.packed_handle)
    return resource_list

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # Trackable hook that copies this variable's value to a CPU copy. Parameter: object_map, a
    # mapping from objects to their CPU copies; if this variable is missing it creates an
    # uninitialized variable with the same trainable flag, shape, dtype, shared name, strategy
    # and aggregation on the CPU device and registers it. Then assigns the current value into
    # the copy on the copy's device. Returns nothing; mutates object_map and the copy.
    """For implementing `Trackable`."""
    if self not in object_map:
      # If not populated, initialize the cpu copy first.
      op_device = pydev.DeviceSpec.from_string(self.device).replace(
          device_type="CPU", device_index=0).to_string()
      with ops.device(op_device):
        new_var = resource_variable_ops.UninitializedVariable(
            trainable=self.trainable,
            shape=self.shape,
            dtype=self.dtype,
            name=self._shared_name,
            distribute_strategy=self._distribute_strategy,
            aggregation=self._aggregation)  # pylint: disable=protected-access
      object_map[self] = new_var

    # Then copy value of self to the copy.
    destination_var = object_map[self]
    with ops.device(destination_var.device):
      destination_var.assign(self.read_value())

  def _write_object_proto(self, proto, options):
    # [FUNCTION DOC]
    # Fills in the SavedObject message when saving. Parameters: proto (pre-built SavedVariable
    # message to modify), options (save options). First writes the standard resource-variable
    # fields, then writes distributed-variable specifics (component information and device
    # placement) through the utility module. Returns nothing; mutates proto.
    """Update a SavedObject proto for the caller.

    If a DistributedVariable object supports this method, it will be called when
    saving with a pre-built `SavedObject` proto representing the object, plus an
    instance of `SaveOptions`. This method is then free to modify that proto
    instance.

    `DistributedVariable` with `AUTO` or `ON_WRITE` synchronization optionally
    write out information about their components to the
    `experimental_distributed_variable_components` field of a
    `SavedVariable` (depending on the `SaveOptions` variable policy).

    Args:
      proto: A pre-built `SavedObject` proto for this object. It is assumed this
        will be a `SavedVariable` instance.
      options: A `SaveOptions` instance.
    """
    resource_variable_ops.write_object_proto_for_resource_variable(
        self, proto, options)
    # Set protos in the saved model such that distributed variables are
    # correctly restored on COMPOSITE devices (otherwise, task:0/TPU:0).
    values_util.write_object_proto(self, proto, options)

  @property
  def is_distributed_variable(self):
    # [FUNCTION DOC]
    # Property that always returns true, identifying this object as a distributed variable.
    return True

  def __tf_experimental_restore_capture__(
      self, concrete_function, internal_capture):
    # [FUNCTION DOC]
    # Hook used when restoring a saved function: replaces the capture of this variable in the
    # concrete function's graph with the given internal capture placeholder, and records a
    # pass-through "captured value" operation for gradient purposes. Parameters:
    # concrete_function (restored function), internal_capture (placeholder tensor). Returns this
    # variable.
    graph = concrete_function.graph
    # Add given distributed variable to captures with given placeholder.
    graph.replace_capture(self, internal_capture)
    record.record_operation(
        "captured_value", [internal_capture], [self],
        backward_function=lambda x: [x],
        forward_function=lambda x: [x])
    return self


# We extend from `saveable_object.SaveableObject` instead of
# `saveable_object_util.ResourceVariableSaveable` since we need to read the
# value of ONREAD variables when saving. `SaveableObject` provides a way to
# specify the function to run to get the value of the variable or tensor at
# saving time. We can use this for both ON_READ and ON_WRITE variables.
# TODO(b/164586507): Consolidate ON_WRITE and ON_READ saving/restoring logic
# if possible.
class _DistributedVariableSaveable(saveable_object.SaveableObject):
  # [CLASS DOC]
  # HELPER: private checkpoint saveable for a DistributedVariable that has a policy; asks the
  # policy for the tensor and specification to save and for the restore operations. Created by
  # the variable's checkpoint-gathering hook.
  """Class for defining how to restore a DistributedVariable."""

  def __init__(self, distributed_variable, primary_variable, name):
    # [FUNCTION DOC]
    # Constructor. Parameters: distributed_variable (the variable to save), primary_variable
    # (its primary component), name (checkpoint key). Raises a value error when the variable has
    # no policy. Obtains the tensor and save specification from the policy and initializes the
    # base saveable with them.
    self._distributed_variable = distributed_variable
    if not self._distributed_variable._policy:
      raise ValueError(
          "The VariablePolicy of the argument `distributed_variable` must be "
          "set to create a _DistributedVariableSaveable. Please set it via "
          "the `var_policy` argument in the constructor of DistributedVariable."
      )
    tensor, spec = distributed_variable._policy.get_saveable(
        distributed_variable, primary_variable, name)
    super(_DistributedVariableSaveable, self).__init__(tensor, spec, name)

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # Restores the checkpointed value into all components. Parameters: restored_tensors (a
    # sequence holding exactly one tensor), restored_shapes (ignored). Returns the restore
    # operations built by the policy.
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return self._distributed_variable._policy.get_restore_ops(  # pylint: disable=protected-access
        self._distributed_variable, tensor)


class _MirroredSaveable(saveable_object.SaveableObject):
  # [CLASS DOC]
  # HELPER: private checkpoint saveable for a MirroredVariable, using the on-write save and
  # restore utilities (one value saved, restored into every replica).
  """Class for defining how to restore a MirroredVariable."""

  def __init__(self, mirrored_variable, primary_variable, name):
    # [FUNCTION DOC]
    # Constructor. Parameters: mirrored_variable, primary_variable (its primary component),
    # name (checkpoint key). Obtains the tensor and save specification from the on-write
    # utilities and initializes the base saveable with them.
    self._mirrored_variable = mirrored_variable
    tensor, spec = values_util.get_on_write_saveable(self._mirrored_variable,
                                                     primary_variable, name)
    super(_MirroredSaveable, self).__init__(tensor, spec, name)

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # Restores the checkpointed value into every replica of the mirrored variable. Parameters:
    # restored_tensors (a sequence with exactly one tensor), restored_shapes (ignored). Returns
    # the restore operations.
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return values_util.get_on_write_restore_ops(self._mirrored_variable, tensor)


class MirroredVariable(DistributedVariable, Mirrored):
  # [CLASS DOC]
  # Distributed variable whose per-replica component variables are kept in sync (ON_WRITE /
  # AUTO synchronization), also usable in cross-replica context. Updates in replica context use
  # the on-write update helper, and scatter min/max/update are only allowed with aggregation
  # none or only-first-replica.
  """Holds a map from replica to variables whose values are kept in sync."""

  def _is_mirrored(self):
    # [FUNCTION DOC]
    # Always true (via the Mirrored parent), chosen explicitly because of the multiple
    # inheritance with DistributedVariable.
    return Mirrored._is_mirrored(self)  # Use correct parent class.

  def _update_replica(self, update_fn, value, **kwargs):
    # [FUNCTION DOC]
    # Applies an update in replica context using the on-write update helper. Parameters:
    # update_fn, value, keyword arguments for update_fn. Returns the update result.
    return _on_write_update_replica(self, update_fn, value, **kwargs)

  def scatter_min(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse element-wise minimum; arguments are forwarded. Primary's scatter when saving
    # non-distributed. Raises a not-implemented error unless aggregation is none or
    # only-first-replica. Otherwise defers to the parent implementation and returns its result.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_min(*args, **kwargs)
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_min", aggregation=self._aggregation))
    return super(MirroredVariable, self).scatter_min(*args, **kwargs)

  def scatter_max(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse element-wise maximum; arguments are forwarded. Primary's scatter when saving
    # non-distributed. Raises a not-implemented error unless aggregation is none or
    # only-first-replica. Otherwise defers to the parent implementation and returns its result.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_max(*args, **kwargs)
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_max", aggregation=self._aggregation))
    return super(MirroredVariable, self).scatter_max(*args, **kwargs)

  def scatter_update(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse overwrite; arguments are forwarded. Primary's scatter when saving non-distributed.
    # Raises a not-implemented error unless aggregation is none or only-first-replica.
    # Otherwise defers to the parent implementation and returns its result.
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_update(*args, **kwargs)
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_update", aggregation=self._aggregation))
    return super(MirroredVariable, self).scatter_update(*args, **kwargs)

  def _get_cross_replica(self):
    # [FUNCTION DOC]
    # Cross-replica access: returns an identity tensor of the component on the current device
    # (or primary), so the variable itself is not exposed and cannot be modified by mistake.
    # Return identity, to avoid directly exposing the variable to the user and
    # allowing it to be modified by mistake.
    return array_ops.identity(Mirrored._get_cross_replica(self))

  def _as_graph_element(self):
    # [FUNCTION DOC]
    # Returns the graph element of the component on the current device (or the primary).
    return self._get_on_device_or_primary()._as_graph_element()  # pylint: disable=protected-access

  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # Checkpointing hook. Returns a mapping with one entry from the variable-value key to a
    # factory creating a mirrored-variable saveable for this variable.
    """Overrides Trackable method.

    This allows both name-based and object-based save and restore of
    MirroredVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """

    def _saveable_factory(name=self._common_name):
      # [FUNCTION DOC]
      # HELPER: factory for the checkpoint mapping; takes an optional name (default: common
      # name) and returns a saveable for this mirrored variable and its primary.
      return _MirroredSaveable(self, self._primary, name)

    return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # Converts the variable to a tensor. Parameters: dtype (optional), name (optional), as_ref
    # (must be false). Raises a value error when as_ref is true (typical for legacy control-flow
    # use). Otherwise converts the current component to a tensor and returns it.
    """Converts a variable to a tensor."""
    # TODO(b/154017756): Make _dense_var_to_tensor consistent between ON_READ
    # and ON_WRITE.
    # Try to avoid assignments to and other mutations of MirroredVariable
    # state except through a DistributionStrategy.extended.update() or any of
    # the `assign*` and `scatter*` calls.
    if as_ref:
      # A TF 1.x case where the variable is a boolean variable and used like:
      # tf.cond(v, true_fn, false_fn).
      raise ValueError(
          "You may be using variable created under distribute strategy in TF "
          "1.x control flows. Try explicitly converting the variable to Tensor "
          "using variable.read_value(), or switch to TF 2.x.")
    return ops.convert_to_tensor(
        self._get(), dtype=dtype, name=name, as_ref=as_ref)


class _SyncOnReadSaveable(saveable_object.SaveableObject):
  # [CLASS DOC]
  # HELPER: private checkpoint saveable for a SyncOnReadVariable; saves the aggregated value and
  # restores through the on-read utilities, which take the aggregation mode into account.
  """Class for defining how to restore a SyncOnReadVariable."""

  def __init__(self, sync_on_read_variable, name):
    # [FUNCTION DOC]
    # Constructor. Parameters: sync_on_read_variable (the variable to save), name (checkpoint
    # key). Obtains the tensor and save specification from the on-read utilities (using the
    # variable's primary component) and initializes the base saveable with them.
    self._sync_on_read_variable = sync_on_read_variable
    tensor, spec = values_util.get_on_read_saveable(
        sync_on_read_variable, sync_on_read_variable._primary, name)

    super(_SyncOnReadSaveable, self).__init__(tensor, spec, name)

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # Restores the checkpointed value into the sync-on-read variable's components, taking its
    # aggregation mode into account. Parameters: restored_tensors (sequence with exactly one
    # tensor), restored_shapes (ignored). Returns the restore operations.
    """Restore the same value into all variables."""
    tensor, = restored_tensors
    return values_util.get_on_read_restore_ops(
        self._sync_on_read_variable, tensor,
        self._sync_on_read_variable.aggregation)


class SyncOnReadVariable(DistributedVariable):
  # [CLASS DOC]
  # Distributed variable whose per-replica components hold independent values that are only
  # combined (reduced according to the aggregation mode) when read in cross-replica context or
  # saved (ON_READ synchronization). Assignments in cross-replica context are broadcast through
  # on-read helpers; scatter operations are not supported.
  """Holds a map from replica to variables whose values are reduced on save."""

  def _update_replica(self, update_fn, value, **kwargs):
    # [FUNCTION DOC]
    # Applies an update in replica context to the component on the current device (or primary)
    # only. Parameters: update_fn, value, keyword arguments. Returns the update result.
    return update_fn(self._get_on_device_or_primary(), value, **kwargs)

  def _get(self):
    # [FUNCTION DOC]
    # Returns the variable's value for the current context inside the strategy scope: the
    # component for the current replica in a non-default replica context, or the synchronized
    # (reduced) value in default replica or cross-replica context, via the parent lookup.
    """Returns the value of SyncOnReadVariable based on surrounding context.

    If called under a non-default replica-context, returns the corresponding
    variable on that replica.
    If called under default replica-context or cross-replica context, returns
    the synced value.
    """
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      return super(SyncOnReadVariable, self)._get()

  # TODO(b/154017756): Make assign behavior in cross replica context consistent
  # with MirroredVariable.
  def assign_sub(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Subtracts value. Parameters: value, use_locking (default false), name (optional),
    # read_value (default true). Primary's subtraction when saving non-distributed. In
    # cross-replica context (outside a replica update) it marks the program unsaveable and uses
    # the on-read cross-replica subtraction; otherwise it uses the parent implementation.
    if values_util.is_saving_non_distributed():
      return self._primary.assign_sub(value, use_locking, name, read_value)
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      if (distribute_lib.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_sub_cross_replica(
            self, value, read_value=read_value)
      else:
        return super(SyncOnReadVariable,
                     self).assign_sub(value, use_locking, name, read_value)

  def assign_add(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Adds value. Parameters: value, use_locking (default false), name (optional), read_value
    # (default true). Primary's addition when saving non-distributed. In cross-replica context
    # (outside a replica update) it marks the program unsaveable and uses the on-read
    # cross-replica addition; otherwise it uses the parent implementation.
    if values_util.is_saving_non_distributed():
      return self._primary.assign_add(value, use_locking, name, read_value)
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      if (distribute_lib.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_add_cross_replica(
            self, value, read_value=read_value)
      else:
        return super(SyncOnReadVariable,
                     self).assign_add(value, use_locking, name, read_value)

  def assign(self, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Overwrites the variable. Parameters: value, use_locking (default false), name (optional),
    # read_value (default true). Primary's assignment when saving non-distributed. In
    # cross-replica context (outside a replica update) it marks the program unsaveable and uses
    # the on-read cross-replica assignment; otherwise it uses the parent implementation.
    if values_util.is_saving_non_distributed():
      return self._primary.assign(value, use_locking, name, read_value)
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      if (distribute_lib.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_cross_replica(
            self, value, read_value=read_value)
      else:
        return super(SyncOnReadVariable, self).assign(value, use_locking, name,
                                                      read_value)

  def _scatter_not_implemented(self, method):
    # [FUNCTION DOC]
    # HELPER: shared by the scatter methods of this class. Always raises a not-implemented error
    # saying that ON_READ variables do not support the named method (parameter: method name).
    raise NotImplementedError(
        f"Variables with `synchronization=ON_READ` doesn't support `{method}`")

  def scatter_sub(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse subtraction: forwarded to the primary when saving non-distributed; otherwise
    # raises a not-implemented error (unsupported for ON_READ variables).
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_sub(*args, **kwargs)
    self._scatter_not_implemented("scatter_sub")

  def scatter_add(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse addition: forwarded to the primary when saving non-distributed; otherwise raises a
    # not-implemented error (unsupported for ON_READ variables).
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_add(*args, **kwargs)
    self._scatter_not_implemented("scatter_add")

  def scatter_mul(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse multiplication: forwarded to the primary when saving non-distributed; otherwise
    # raises a not-implemented error (unsupported for ON_READ variables).
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_mul(*args, **kwargs)
    self._scatter_not_implemented("scatter_mul")

  def scatter_div(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse division: forwarded to the primary when saving non-distributed; otherwise raises a
    # not-implemented error (unsupported for ON_READ variables).
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_div(*args, **kwargs)
    self._scatter_not_implemented("scatter_div")

  def scatter_min(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse element-wise minimum: forwarded to the primary when saving non-distributed;
    # otherwise raises a not-implemented error (unsupported for ON_READ variables).
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_min(*args, **kwargs)
    self._scatter_not_implemented("scatter_min")

  def scatter_max(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse element-wise maximum: forwarded to the primary when saving non-distributed;
    # otherwise raises a not-implemented error (unsupported for ON_READ variables).
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_max(*args, **kwargs)
    self._scatter_not_implemented("scatter_max")

  def scatter_update(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Sparse overwrite: forwarded to the primary when saving non-distributed; otherwise raises a
    # not-implemented error (unsupported for ON_READ variables).
    if values_util.is_saving_non_distributed():
      return self._primary.scatter_update(*args, **kwargs)
    self._scatter_not_implemented("scatter_update")

  def value(self):
    # [FUNCTION DOC]
    # Returns the value. Raises a not-implemented error inside the sync-on-read variable
    # context. Primary's value when saving non-distributed. In cross-replica context (outside a
    # replica update) it returns the first replica's value for only-first-replica aggregation,
    # else the reduced cross-replica value; otherwise the value of the current component.
    if distribute_lib.in_variable_sync_on_read_context():
      raise NotImplementedError(
          "call `variable.value()` inside variable_sync_on_read_context is not "
          "supported")
    if values_util.is_saving_non_distributed():
      return self._primary.value()
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      if (distribute_lib.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
          return self._get_replica(0).value()
        return self._get_cross_replica()
      else:
        # _get_on_device_or_primary() returns a Variable.
        return self._get_on_device_or_primary().value()

  def read_value(self):
    # [FUNCTION DOC]
    # Returns the current value as a tensor via the parent implementation. Raises a
    # not-implemented error when called inside the sync-on-read variable context.
    if distribute_lib.in_variable_sync_on_read_context():
      raise NotImplementedError(
          "call `variable.read_value()` inside variable_sync_on_read_context is"
          " not supported")
    return super().read_value()

  def _get_cross_replica(self):
    # [FUNCTION DOC]
    # Cross-replica value: for only-first-replica aggregation returns the first component; for
    # other modes reduces the variable across replicas with the matching reduce operation
    # (marking the program unsaveable when the mode is sum) inside the strategy scope, and
    # returns the reduced value.
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
      # Consider returning a tensor value here to make the return value of
      # _get_cross_replica consistent.
      return self._get_replica(0)
    if self._aggregation == vs.VariableAggregation.SUM:
      values_util.mark_as_unsaveable()
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      return self._distribute_strategy.reduce(
          reduce_util.ReduceOp.from_variable_aggregation(self._aggregation),
          self,
          axis=None)

  def _as_graph_element(self):
    # [FUNCTION DOC]
    # Returns the graph element: the primary's when saving non-distributed; in cross-replica
    # context the cross-replica value converted to a tensor; otherwise the current component's
    # graph element.
    if values_util.is_saving_non_distributed():
      return self._primary._as_graph_element()  # pylint: disable=protected-access
    # pylint: disable=protected-access
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      if distribute_lib.in_cross_replica_context():
        return ops.convert_to_tensor(self._get_cross_replica())
    return self._get()._as_graph_element()

  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # Checkpointing hook. Returns a mapping with one entry from the variable-value key to a
    # factory creating a sync-on-read saveable for this variable.
    """Overrides Trackable method.

    This allows both name-based and object-based save and restore of
    `SyncOnReadVariable`s.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """

    def _saveable_factory(name=self._common_name):
      # [FUNCTION DOC]
      # HELPER: factory for the checkpoint mapping; takes an optional name (default: common
      # name) and returns a saveable for this sync-on-read variable.
      return _SyncOnReadSaveable(self, name)

    return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    # [FUNCTION DOC]
    # Converts the variable to a tensor. Parameters: dtype (optional), name (optional), as_ref.
    # Primary's conversion when saving non-distributed. Inside a replica context within the
    # sync-on-read variable context, it returns the first replica's value for
    # only-first-replica aggregation, otherwise an all-reduce of the current component's value
    # using the aggregation's reduce operation (marking the program unsaveable for sum).
    # Otherwise converts the current component normally.
    """Converts a SyncOnReadVariable to a tensor."""
    if values_util.is_saving_non_distributed():
      return ops.convert_to_tensor(
          self._primary, dtype=dtype, name=name, as_ref=as_ref)
    with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
      replica_context = distribute_lib.get_replica_context()
      if (replica_context is not None and
          distribute_lib.in_variable_sync_on_read_context()):
        if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
          return ops.convert_to_tensor(
              self._get_replica(0), dtype=dtype, name=name, as_ref=as_ref)
        if self._aggregation == vs.VariableAggregation.SUM:
          values_util.mark_as_unsaveable()
        # pylint: disable=protected-access
        reduced = (
            replica_context.strategy.extended._replica_ctx_all_reduce(
                reduce_util.ReduceOp.from_variable_aggregation(
                    self._aggregation),
                self._get().read_value()))
        return ops.convert_to_tensor(
            reduced, dtype=dtype, name=name, as_ref=as_ref)

      return ops.convert_to_tensor(
          self._get(), dtype=dtype, name=name, as_ref=as_ref)


# Register a conversion functions which reads the value of the variable,
# allowing instances of the class to be used as tensors.
# DistributedVariable
def _tensor_conversion_distributed_var(var,
                                       dtype=None,
                                       name=None,
                                       as_ref=False):
  # [FUNCTION DOC]
  # HELPER: tensor conversion function registered for DistributedVariable. Parameters: var,
  # dtype (optional), name (optional), as_ref (default false). Returns the variable's own dense
  # tensor conversion with the same arguments.
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


tensor_conversion_registry.register_tensor_conversion_function(
    DistributedVariable, _tensor_conversion_distributed_var)


# MirroredVariables
def _tensor_conversion_mirrored(var, dtype=None, name=None, as_ref=False):
  # [FUNCTION DOC]
  # HELPER: tensor conversion function registered for MirroredVariable. Parameters: var, dtype
  # (optional), name (optional), as_ref (default false). Returns the variable's own dense
  # tensor conversion with the same arguments.
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


tensor_conversion_registry.register_tensor_conversion_function(
    MirroredVariable, _tensor_conversion_mirrored)


# Mirrored Values
def _tensor_conversion_mirrored_val(value, dtype=None, name=None, as_ref=False):
  # [FUNCTION DOC]
  # HELPER: tensor conversion function registered for Mirrored values. Parameters: value (a
  # Mirrored), dtype (optional), name (optional), as_ref (default false). Converts the
  # component for the current replica to a tensor and returns it.
  return ops.convert_to_tensor(
      value._get(), dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


tensor_conversion_registry.register_tensor_conversion_function(
    Mirrored, _tensor_conversion_mirrored_val)


# SyncOnReadVariables
def _tensor_conversion_sync_on_read(var, dtype=None, name=None, as_ref=False):
  # [FUNCTION DOC]
  # HELPER: tensor conversion function registered for SyncOnReadVariable. Parameters: var,
  # dtype (optional), name (optional), as_ref (default false). Returns the variable's own dense
  # tensor conversion with the same arguments.
  return var._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)  # pylint: disable=protected-access


tensor_conversion_registry.register_tensor_conversion_function(
    SyncOnReadVariable, _tensor_conversion_sync_on_read)


class VariablePolicy(object):
  # [CLASS DOC]
  # Abstract base of the policies that define how a distributed variable synchronizes and
  # aggregates (reads, writes, cross-replica access, graph element, update in replica context).
  # Holds the aggregation mode. Concrete subclasses are OnReadPolicy and OnWritePolicy.
  """Policy defining synchronization and aggregation of a distributed variable.

  Given `synchronization` and `aggregation` parameters set on a `tf.Variable`
  during variable creation within `tf.distribute` scope, `tf.distribute` creates
  an appropriate policy object and assigns it to the distributed variable. All
  variable operations are delegated to the respective policy object.
  """

  def __init__(self, aggregation):
    # [FUNCTION DOC]
    # Constructor. Stores the aggregation mode (a variable aggregation value) used by the
    # policy's reduce and scatter checks.
    self._aggregation = aggregation

  def value(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the base signature takes no variable argument, but callers (the distributed
    # variable and the subclasses) pass the variable, so calling the base method as called by
    # DistributedVariable would fail with an argument-count error instead of the intended
    # not-implemented error.
    # Abstract: always raises a not-implemented error saying subclasses must override it.
    raise NotImplementedError(
        "VariablePolicy.value should be overridden by sub-classes. "
        f"Type name is {type(self)}"
    )

  def _is_mirrored(self):
    # [FUNCTION DOC]
    # Abstract: whether variables under this policy are mirrored. The base version always raises
    # a not-implemented error; subclasses return true or false.
    raise NotImplementedError(
        "VariablePolicy._is_mirrored should be overridden by sub-classes. "
        f"Type name is {type(self)}"
    )

  def _as_graph_element(self, _):
    # [FUNCTION DOC]
    # Abstract: returns the graph element of a variable (the argument). The base version always
    # raises a not-implemented error.
    raise NotImplementedError(
        "VariablePolicy._as_graph_element should be overridden by sub-classes. "
        f"Type name is {type(self)}"
    )

  def _get_cross_replica(self, var):
    # [FUNCTION DOC]
    # Abstract: returns the cross-replica value of the variable var. The base version always
    # raises a not-implemented error.
    raise NotImplementedError(
        "VariablePolicy._get_cross_replica should be overridden by"
        f" sub-classes. Type name is {type(self)}"
    )

  def _update_replica(self, var, update_fn, value, **kwargs):
    # [FUNCTION DOC]
    # Abstract: applies an update to variable var in replica context. Parameters: var, update_fn,
    # value, keyword arguments. The base version always raises a not-implemented error.
    raise NotImplementedError(
        "VariablePolicy._update_replica should be overridden by sub-classes. "
        f"Type name is {type(self)}"
    )


class OnReadPolicy(VariablePolicy):
  # [CLASS DOC]
  # Policy for ON_READ (sync-on-read) variables: replicas keep independent values that are
  # reduced according to the aggregation when read in cross-replica context or saved.
  # Assignments in cross-replica context are applied through on-read helpers; scatter
  # operations are unsupported (raise not-implemented errors). Created by the variable creation
  # logic and held by the distributed variable.
  """Policy defined for `tf.VariableSynchronization.ON_READ` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.ON_READ` and `aggregation` is set to any of the
  values allowed by the `tf.VariableAggregation` enum such as `NONE`, `SUM`,
  `MEAN` or `ONLY_FIRST_REPLICA`when creating a `tf.Variable` in `tf.distribute`
  scope.
  """

  def _is_mirrored(self):
    # [FUNCTION DOC]
    # Always returns false: ON_READ variables are not mirrored.
    return False

  def value(self, var):
    # [FUNCTION DOC]
    # Returns the value of variable var. In cross-replica context (outside a replica update) it
    # returns the first replica's value for only-first-replica aggregation, else the variable's
    # cross-replica (reduced) value; otherwise the value of the component on the current device
    # or primary. Runs inside the variable's strategy scope.
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
      if (distribute_lib.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
          return var._get_replica(0).value()  # pylint: disable=protected-access
        return var._get_cross_replica()  # pylint: disable=protected-access
      else:
        return var._get_on_device_or_primary().value()  # pylint: disable=protected-access

  def _as_graph_element(self, var):
    # [FUNCTION DOC]
    # Returns the graph element of variable var: in cross-replica context the cross-replica
    # value converted to a tensor; otherwise the current component's graph element.
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
      if distribute_lib.in_cross_replica_context():
        return ops.convert_to_tensor(var._get_cross_replica())  # pylint: disable=protected-access
    return var._get()._as_graph_element()  # pylint: disable=protected-access

  def _get_cross_replica(self, var):
    # [FUNCTION DOC]
    # Cross-replica value of variable var: the first component for only-first-replica
    # aggregation; otherwise the variable reduced across replicas using the reduce operation
    # matching the aggregation (marking the program unsaveable for sum), inside the strategy
    # scope.
    if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
      return var._get_replica(0)  # pylint: disable=protected-access
    if self._aggregation == vs.VariableAggregation.SUM:
      values_util.mark_as_unsaveable()
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
      return var.distribute_strategy.reduce(
          reduce_util.ReduceOp.from_variable_aggregation(self._aggregation),
          var,
          axis=None)

  def _update_replica(self, var, update_fn, value, **kwargs):
    # [FUNCTION DOC]
    # Applies an update in replica context to only the component of var on the current device
    # (or primary). Parameters: var, update_fn, value, keyword arguments. Returns the result.
    return update_fn(var._get_on_device_or_primary(), value, **kwargs)  # pylint: disable=protected-access

  def _scatter_not_implemented(self, method):
    # [FUNCTION DOC]
    # HELPER: shared by the scatter methods of this policy. Always raises a not-implemented
    # error saying that ON_READ variables do not support the named method in cross-replica
    # context (parameter: method name).
    raise NotImplementedError(f"ON_READ variables doesn't support `{method}` "
                              "in cross replica context")

  def assign_sub(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    # [FUNCTION DOC]
    # Subtracts value from variable var. Parameters: var, value, use_locking (default false),
    # name (optional), read_value (default true). In cross-replica context (outside a replica
    # update) it marks the program unsaveable and uses the on-read cross-replica subtraction;
    # otherwise it uses the on-write subtraction helper. Returns the result.
    """Subtracts a value from this variable."""
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
      if (distribute_lib.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_sub_cross_replica(
            var, value, read_value=read_value)
      else:
        return values_util.on_write_assign_sub(
            var,
            value,
            use_locking=use_locking,
            name=name,
            read_value=read_value)

  def assign_add(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    # [FUNCTION DOC]
    # Adds value to variable var. Parameters: var, value, use_locking (default false), name
    # (optional), read_value (default true). In cross-replica context (outside a replica
    # update) it marks the program unsaveable and uses the on-read cross-replica addition;
    # otherwise it uses the on-write addition helper. Returns the result.
    """Adds a value to this variable."""
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
      if (distribute_lib.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_add_cross_replica(
            var, value, read_value=read_value)
      else:
        return values_util.on_write_assign_add(
            var,
            value,
            use_locking=use_locking,
            name=name,
            read_value=read_value)

  def assign(self, var, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Overwrites variable var with value. Parameters: var, value, use_locking (default false),
    # name (optional), read_value (default true). In cross-replica context (outside a replica
    # update) it marks the program unsaveable and uses the on-read cross-replica assignment;
    # otherwise it uses the on-write assignment helper. Returns the result.
    with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
      if (distribute_lib.in_cross_replica_context() and
          not values_util.in_replica_update_context()):
        values_util.mark_as_unsaveable()
        return values_util.on_read_assign_cross_replica(
            var, value, read_value=read_value)
      else:
        return values_util.on_write_assign(
            var,
            value,
            use_locking=use_locking,
            name=name,
            read_value=read_value)

  def scatter_sub(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported: ignores its arguments and raises a not-implemented error.
    del args, kwargs
    self._scatter_not_implemented("scatter_sub")

  def scatter_add(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported: ignores its arguments and raises a not-implemented error.
    del args, kwargs
    self._scatter_not_implemented("scatter_add")

  def scatter_mul(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported: ignores its arguments and raises a not-implemented error.
    del args, kwargs
    self._scatter_not_implemented("scatter_mul")

  def scatter_div(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported: ignores its arguments and raises a not-implemented error.
    del args, kwargs
    self._scatter_not_implemented("scatter_div")

  def scatter_min(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported: ignores its arguments and raises a not-implemented error.
    del args, kwargs
    self._scatter_not_implemented("scatter_min")

  def scatter_max(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported: ignores its arguments and raises a not-implemented error.
    del args, kwargs
    self._scatter_not_implemented("scatter_max")

  def scatter_update(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Unsupported: ignores its arguments and raises a not-implemented error.
    del args, kwargs
    self._scatter_not_implemented("scatter_update")

  def get_saveable(self, var, primary_var, name):
    # [FUNCTION DOC]
    # Creates the checkpoint save information for variable var using the on-read utilities.
    # Parameters: var, primary_var (its primary component), name (checkpoint key). Returns the
    # tensor and save specification pair.
    """Create a saveable object for the given variable."""
    return values_util.get_on_read_saveable(var, primary_var, name)

  def get_restore_ops(self, var, tensor):
    # [FUNCTION DOC]
    # Builds the operations that restore a checkpointed tensor into variable var using the
    # on-read utilities and this policy's aggregation. Returns the restore operations.
    """Restore the same value into all variables."""
    return values_util.get_on_read_restore_ops(var, tensor, self._aggregation)


class OnWritePolicy(VariablePolicy):
  # [CLASS DOC]
  # Policy for ON_WRITE / AUTO synchronization: every write is applied to all replicas so the
  # components stay identical (mirrored). Reads use the component on the current device;
  # cross-replica reads return an identity copy. Assignments and scatters delegate to the
  # on-write utilities; scatter min, max and update additionally require aggregation none or
  # only-first-replica.
  """Policy defined for `tf.VariableSynchronization.ON_WRITE` synchronization.

  This policy is created when the following `synchronization` and `aggregation`
  parameters are specified when creating a `tf.Variable` in `tf.distribute`
  scope and `synchronization` is equal to `tf.VariableSynchronization.ON_WRITE`
  or `tf.VariableSynchronization.AUTO`.
  """

  def _is_mirrored(self):
    # [FUNCTION DOC]
    # Always returns true: ON_WRITE variables are mirrored.
    return True

  def value(self, var):
    # [FUNCTION DOC]
    # Returns the value of the component of var on the current device (or primary).
    return var._get_on_device_or_primary().value()  # pylint: disable=protected-access

  def _as_graph_element(self, var):
    # [FUNCTION DOC]
    # Returns the graph element of the component of var on the current device (or primary).
    return var._get_on_device_or_primary()._as_graph_element()  # pylint: disable=protected-access

  def _get_cross_replica(self, var):
    # [FUNCTION DOC]
    # Cross-replica value of var: an identity tensor of the component on the current device (or
    # primary), so the variable is not exposed directly.
    # Return identity, to avoid directly exposing the variable to the user and
    # allowing it to be modified by mistake.
    return array_ops.identity(var._get_on_device_or_primary())  # pylint: disable=protected-access

  def _update_replica(self, var, update_fn, value, **kwargs):
    # [FUNCTION DOC]
    # Applies an update in replica context. Parameters: var, update_fn, value, keyword
    # arguments. With aggregation NONE it updates only the current device's (or primary)
    # component; otherwise it uses the on-write replica update helper that aggregates first.
    # Returns the update result.
    if var.aggregation == variables_lib.VariableAggregation.NONE:
      return update_fn(var._get_on_device_or_primary(), value, **kwargs)  # pylint: disable=protected-access
    return _on_write_update_replica(var, update_fn, value, **kwargs)

  def assign(self, var, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Overwrites var with value using the on-write assignment utility. Parameters: var, value,
    # use_locking (default false), name (optional), read_value (default true). Returns its
    # result.
    return values_util.on_write_assign(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    # [FUNCTION DOC]
    # Adds value to var using the on-write addition utility. Parameters: var, value,
    # use_locking (default false), name (optional), read_value (default true). Returns its
    # result.
    return values_util.on_write_assign_add(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_sub(self,
                 var,
                 value,
                 use_locking=False,
                 name=None,
                 read_value=True):
    # [FUNCTION DOC]
    # Subtracts value from var using the on-write subtraction utility. Parameters: var, value,
    # use_locking (default false), name (optional), read_value (default true). Returns its
    # result.
    return values_util.on_write_assign_sub(
        var, value, use_locking=use_locking, name=name, read_value=read_value)

  def scatter_sub(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse subtraction on var through the scatter utility. Parameters: var, sparse_delta,
    # use_locking (default false), name (optional). Returns its result.
    return values_util.scatter_sub(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_add(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse addition on var through the scatter utility. Parameters: var, sparse_delta,
    # use_locking (default false), name (optional). Returns its result.
    return values_util.scatter_add(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_mul(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse multiplication on var through the scatter utility. Parameters: var, sparse_delta,
    # use_locking (default false), name (optional). Returns its result.
    return values_util.scatter_mul(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_div(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse division on var through the scatter utility. Parameters: var, sparse_delta,
    # use_locking (default false), name (optional). Returns its result.
    return values_util.scatter_div(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_min(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse element-wise minimum on var. Parameters: var, sparse_delta, use_locking (default
    # false), name (optional). Raises a not-implemented error unless the policy's aggregation is
    # none or only-first-replica; otherwise returns the scatter utility's result.
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_min", aggregation=self._aggregation))
    return values_util.scatter_min(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_max(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse element-wise maximum on var. Parameters: var, sparse_delta, use_locking (default
    # false), name (optional). Raises a not-implemented error unless the policy's aggregation is
    # none or only-first-replica; otherwise returns the scatter utility's result.
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_max", aggregation=self._aggregation))
    return values_util.scatter_max(
        var, sparse_delta, use_locking=use_locking, name=name)

  def scatter_update(self, var, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse overwrite on var. Parameters: var, sparse_delta, use_locking (default false), name
    # (optional). Raises a not-implemented error unless the policy's aggregation is none or
    # only-first-replica; otherwise returns the scatter utility's result.
    if (self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and
        self._aggregation != vs.VariableAggregation.NONE):
      raise NotImplementedError(
          values_util.scatter_error_msg.format(
              op_name="scatter_update", aggregation=self._aggregation))
    return values_util.scatter_update(
        var, sparse_delta, use_locking=use_locking, name=name)

  def get_saveable(self, var, primary_var, name):
    # [FUNCTION DOC]
    # Creates checkpoint save information for var using the on-write utility. Parameters: var,
    # primary_var (its primary component), name (checkpoint key). Returns the tensor and save
    # specification pair.
    """Saveable ops for AUTO variables."""
    return values_util.get_on_write_saveable(var, primary_var, name)

  def get_restore_ops(self, var, tensor):
    # [FUNCTION DOC]
    # Builds the operations restoring a checkpointed tensor into every replica of var using the
    # on-write utility. Returns the restore operations.
    return values_util.get_on_write_restore_ops(var, tensor)


class PerWorkerResource():
  # [CLASS DOC]
  # Proxy for a resource that exists once per worker host (for example a lookup table under the
  # TPU strategy). It stores the strategy and a mapping from host device names to resources;
  # nearly all attribute reads and writes are redirected to the resource on the current
  # (local) host. Creating one increments a usage counter metric.
  """A per-worker CapturableResource class for non-ParameterServer strategy.

  Resources that populate `host_to_resources` should be instances of classes
  subclassing CapturableResource, although currently it's only used and tested
  for StaticHashTable with TPUStrategy.
  """

  def __init__(self, strategy, host_to_resources):
    # [FUNCTION DOC]
    # Constructor. Parameters: strategy (the distribution strategy), host_to_resources (mapping
    # from canonical host device names to per-host resource objects). Side effect: increments
    # the distribution-strategy input API usage counter for this class. Stores both arguments.
    distribute_lib.distribution_strategy_input_api_counter.get_cell(
        "PerWorkerResource", "TPUDistributedLookupTable").increase_by(1)
    self._strategy = strategy
    self._host_to_resources = host_to_resources

  def __getattribute__(self, name):
    # [FUNCTION DOC]
    # Attribute read hook. Parameter: name. A small set of own attributes (constructor, this
    # hook, the stored mapping, the strategy and the local-resource method) is read normally;
    # every other attribute is read from the resource on the local host.
    if name not in ("__init__", "__getattribute__", "_host_to_resources",
                    "_strategy", "local_resource"):
      return getattr(self.local_resource(), name)
    return super(PerWorkerResource, self).__getattribute__(name)

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Attribute write hook. Parameters: name, value. Only the strategy and the stored mapping
    # are set on this proxy; every other attribute is set on the resource on the local host.
    if name not in ("_strategy", "_host_to_resources"):
      return setattr(self.local_resource(), name, value)
    return super(PerWorkerResource, self).__setattr__(name, value)

  def local_resource(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the fallback (first resource in the mapping) is computed eagerly on every
    # call, so an empty mapping raises a key error even when the host would match; and when the
    # current host is not in the mapping the first entry is silently used instead of failing.
    # Returns the resource for the host of the current device (canonicalized device names), or
    # the first registered resource if there is none for that host.
    """Returns the resource on the local worker."""
    current_device = device_util.canonicalize(device_util.current())
    host_device = device_util.canonicalize(
        device_util.get_host_for_device(current_device))
    return self._host_to_resources.get(
        host_device,
        self._host_to_resources[next(iter(self._host_to_resources))])
