# [MODULE DOC]
# HELPER: internal utility module used mainly by optimizers and moving-average code.
# Provides functions to create "slot" variables: non-trainable variables tied to a primary
# variable or tensor (for example momentum accumulators), scoped under the primary's name and
# by default placed on the same device. Public functions are create_slot (from an initial
# value tensor), create_slot_with_initializer (from an initializer plus shape and type) and
# create_zeros_slot (zero filled, same shape as the primary); all share the private helper
# that builds the variable, handles partitioned primaries and optionally copies XLA sharding.
# No import-time side effects beyond imports.
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

"""Standard functions for creating slots.

A slot is a `Variable` created with the same first m-dimension as a primary
variable or `Tensor`. A slot is always scoped in the namespace of the primary
object and typically has the same device and type.

Slots are typically used as accumulators to track values associated with
the primary object:

```python
# Optimizers can create a slot for each variable to track accumulators
accumulators = {var : create_zeros_slot(var, "momentum") for var in vs}
for var in vs:
  apply_momentum(var, accumulators[var], lr, grad, momentum_tensor)

# Slots can also be used for moving averages
mavg = create_slot(var, var.initialized_value(), "exponential_moving_avg")
update_mavg = mavg.assign_sub((mavg - var) * (1 - decay))
```
"""
# pylint: disable=g-bad-name

from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables


def _create_slot_var(primary,
                     val,
                     scope,
                     validate_shape,
                     shape,
                     dtype,
                     *,
                     copy_xla_sharding=False):
  # [FUNCTION DOC]
  # HELPER: shared builder behind the public slot creation functions.
  # Creates one non-trainable variable in the current variable scope. Inputs: primary (the
  # variable or tensor the slot belongs to), val (initial value tensor or callable
  # initializer), scope (name passed to the variable getter, callers pass an empty string
  # because they already opened a scope), validate_shape (whether the initial shape is checked),
  # shape (used only when val is callable, otherwise forced to nothing), dtype (element type,
  # may be nothing to infer), and the keyword-only copy_xla_sharding flag. The kind of variable
  # follows the primary: resource variable gives resource, reference variable gives reference,
  # otherwise the scope default. Any partitioner on the current scope is temporarily removed
  # and restored afterwards (not restored if variable creation raises). If the primary has
  # partition save-slice information, the slot gets matching slice information truncated to its
  # own rank, skipped for scalar slots. With copy_xla_sharding and equal known ranks, the
  # primary's XLA sharding is copied onto the slot without a sharding op. Returns the slot
  # variable. Errors from the variable getter propagate.
  """Helper function for creating a slot variable."""

  # TODO(lukaszkaiser): Consider allowing partitioners to be set in the current
  # scope.
  current_partitioner = variable_scope.get_variable_scope().partitioner
  variable_scope.get_variable_scope().set_partitioner(None)
  # When init from val instead of callable initializer, the shape is expected to
  # be None, not <unknown> or any fully defined shape.
  shape = shape if callable(val) else None
  if resource_variable_ops.is_resource_variable(primary):
    use_resource = True
  elif isinstance(primary, ref_variable.RefVariable):
    use_resource = False
  else:
    use_resource = None
  slot = variable_scope.get_variable(
      scope,
      initializer=val,
      trainable=False,
      use_resource=use_resource,
      shape=shape,
      dtype=dtype,
      validate_shape=validate_shape)
  variable_scope.get_variable_scope().set_partitioner(current_partitioner)

  # pylint: disable=protected-access
  if isinstance(primary, variables.Variable) and primary._save_slice_info:
    # Primary is a partitioned variable, so we need to also indicate that
    # the slot is a partitioned variable.  Slots have the same partitioning
    # as their primaries.
    # For examples when using AdamOptimizer in linear model, slot.name
    # here can be "linear//weights/Adam:0", while primary.op.name is
    # "linear//weight". We want to get 'Adam' as real_slot_name, so we
    # remove "'linear//weight' + '/'" and ':0'.
    real_slot_name = slot.name[len(primary.op.name + "/"):-2]
    slice_info = primary._save_slice_info
    # support slot's shape not same as primary's shape
    # example: primary's shape = [10, 20, 30], slot's shape =
    # None, [], [10], [10, 20] or [10, 20, 30] is allowed
    # slot's shape = None or [10, 20, 30], set slot's slice_info same as primary
    # slot's shape = [], don't set slot's slice_info
    # slot's shape = [10] or [10, 20], set slot's slice_info according to ndims
    n = slot.shape.ndims
    if n is None or n > 0:
      slot._set_save_slice_info(
          variables.Variable.SaveSliceInfo(
              slice_info.full_name + "/" + real_slot_name,
              slice_info.full_shape[:n], slice_info.var_offset[:n],
              slice_info.var_shape[:n]))
  # pylint: enable=protected-access

  # Copy XLA sharding attributes from the primary if the slot variable has the
  # same rank as the primary.
  def _has_same_rank(primary_shape, slot_shape):
    # [FUNCTION DOC]
    # HELPER: guard used when deciding whether to copy XLA sharding to a slot.
    # Takes two shape objects and returns true only when both ranks are known and equal; returns
    # false if either rank is unknown. No side effects.
    return (primary_shape.rank is not None and slot_shape.rank is not None and
            primary_shape.rank == slot_shape.rank)

  if copy_xla_sharding and _has_same_rank(primary.shape, slot.shape):
    slot = xla_sharding.copy_sharding(primary, slot, use_sharding_op=False)
  return slot


def create_slot(primary,
                val,
                name,
                colocate_with_primary=True,
                *,
                copy_xla_sharding=False):
  # [FUNCTION DOC]
  # Creates a non-trainable slot variable initialized from a tensor value. Inputs: primary
  # (variable or tensor owning the slot), val (tensor giving the initial value, its dtype and
  # shape define the slot; shape validation is on only if the shape is fully defined), name
  # (slot name suffix), colocate_with_primary (default true: place on the primary's device via
  # the current distribution strategy), and keyword-only copy_xla_sharding (default false).
  # The slot is created under a scope named after the primary's shared name (variables) or
  # operation name (tensors) joined with the slot name; a unique suffix is added if reused.
  # Returns the new variable. Side effects: adds a variable to the graph and the variable
  # scope store.
  """Create a slot initialized to the given value.

  The type of the slot is determined by the given value.

  Args:
    primary: The primary `Variable` or `Tensor`.
    val: A `Tensor` specifying the initial value of the slot.
    name: Name to use for the slot variable.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.
    copy_xla_sharding: Boolean. If True also copies XLA sharding
      from primary.

  Returns:
    A `Variable` object.
  """
  # Scope the slot name in the namespace of the primary variable.
  # Set primary's name + '/' + name as default name, so the scope name of
  # optimizer can be shared when reuse is True. Meanwhile when reuse is False
  # and the same name has been previously used, the scope name will add '_N'
  # as suffix for unique identifications.
  validate_shape = val.get_shape().is_fully_defined()
  if isinstance(primary, variables.Variable):
    prefix = primary._shared_name  # pylint: disable=protected-access
  else:
    prefix = primary.op.name
  with variable_scope.variable_scope(None, prefix + "/" + name):
    if colocate_with_primary:
      distribution_strategy = distribute_lib.get_strategy()
      with distribution_strategy.extended.colocate_vars_with(primary):
        return _create_slot_var(
            primary,
            val,
            "",
            validate_shape,
            None,
            None,
            copy_xla_sharding=copy_xla_sharding)
    else:
      return _create_slot_var(
          primary,
          val,
          "",
          validate_shape,
          None,
          None,
          copy_xla_sharding=copy_xla_sharding)


def create_slot_with_initializer(primary,
                                 initializer,
                                 shape,
                                 dtype,
                                 name,
                                 colocate_with_primary=True,
                                 *,
                                 copy_xla_sharding=False):
  # [FUNCTION DOC]
  # Creates a non-trainable slot variable whose value comes from an initializer. Inputs:
  # primary (owning variable or tensor), initializer (initializer object), shape (a shape
  # object for the slot; must support the fully-defined query, validation happens only when it
  # is fully defined), dtype (element type), name (slot name suffix), colocate_with_primary
  # (default true), and keyword-only copy_xla_sharding (default false). Uses the same scoping
  # as the value-based slot creator. Returns the new variable. Side effects: adds a variable
  # to the graph and variable scope store.
  """Creates a slot initialized using an `Initializer`.

  The type of the slot is determined by the given value.

  Args:
    primary: The primary `Variable` or `Tensor`.
    initializer: An `Initializer`.  The initial value of the slot.
    shape: Shape of the initial value of the slot.
    dtype: Type of the value of the slot.
    name: Name to use for the slot variable.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.
    copy_xla_sharding: Boolean. If True also copies XLA sharding
      from primary.

  Returns:
    A `Variable` object.
  """
  # Scope the slot name in the namespace of the primary variable.
  # Set "primary.op.name + '/' + name" as default name, so the scope name of
  # optimizer can be shared when reuse is True. Meanwhile when reuse is False
  # and the same name has been previously used, the scope name will add '_N'
  # as suffix for unique identifications.
  validate_shape = shape.is_fully_defined()
  if isinstance(primary, variables.Variable):
    prefix = primary._shared_name  # pylint: disable=protected-access
  else:
    prefix = primary.op.name
  with variable_scope.variable_scope(None, prefix + "/" + name):
    if colocate_with_primary:
      distribution_strategy = distribute_lib.get_strategy()
      with distribution_strategy.extended.colocate_vars_with(primary):
        return _create_slot_var(
            primary,
            initializer,
            "",
            validate_shape,
            shape,
            dtype,
            copy_xla_sharding=copy_xla_sharding)
    else:
      return _create_slot_var(
          primary,
          initializer,
          "",
          validate_shape,
          shape,
          dtype,
          copy_xla_sharding=copy_xla_sharding)


def create_zeros_slot(primary,
                      name,
                      dtype=None,
                      colocate_with_primary=True,
                      *,
                      copy_xla_sharding=False):
  # [FUNCTION DOC]
  # Creates a zero filled slot with the same shape as the primary. Inputs: primary (owning
  # variable or tensor), name (slot name suffix), dtype (optional, defaults to the primary's
  # type), colocate_with_primary (default true), keyword-only copy_xla_sharding (default
  # false). If the primary's static shape is fully defined, delegates to the initializer-based
  # creator with a zeros initializer. Otherwise it computes the shape dynamically (for a
  # variable, conditionally from its current value if initialized or from its initial value
  # if not; for a tensor, from the tensor), builds a zeros tensor of that shape and delegates
  # to the value-based creator. Returns the new variable. Side effects: adds ops and a variable
  # to the graph.
  """Create a slot initialized to 0 with same shape as the primary object.

  Args:
    primary: The primary `Variable` or `Tensor`.
    name: Name to use for the slot variable.
    dtype: Type of the slot variable.  Defaults to the type of `primary`.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.
    copy_xla_sharding: Boolean. If True also copies XLA sharding
      from primary.

  Returns:
    A `Variable` object.
  """
  if dtype is None:
    dtype = primary.dtype
  slot_shape = primary.get_shape()
  if slot_shape.is_fully_defined():
    initializer = init_ops.zeros_initializer()
    return create_slot_with_initializer(
        primary,
        initializer,
        slot_shape,
        dtype,
        name,
        colocate_with_primary=colocate_with_primary,
        copy_xla_sharding=copy_xla_sharding)
  else:
    if isinstance(primary, variables.Variable):
      slot_shape = array_ops.shape(
          cond.cond(
              variable_v1.is_variable_initialized(primary), primary.read_value,
              lambda: primary.initial_value))
    else:
      slot_shape = array_ops.shape(primary)
    val = array_ops.zeros(slot_shape, dtype=dtype)
    return create_slot(
        primary,
        val,
        name,
        colocate_with_primary=colocate_with_primary,
        copy_xla_sharding=copy_xla_sharding)
