# [MODULE DOC]
# HELPER: helper module shared by the distributed-variable implementations (values.py, ps_values.py).
# Collection of free functions that implement the common behavior of distributed variables:
# building save/restore specs for mirrored (AUTO / ON_WRITE) and sync-on-read (ON_READ) variables,
# assigning to every component of a variable on its own device, forwarding assign / scatter
# operations to the variable's update mechanism, replica-context checks, aggregation of per-replica
# values, and save-context queries. It also defines two error message templates (aggregation and
# scatter) that callers format with variable type / operation names.
# No import-time side effects other than defining these functions and strings.

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
"""Utility functions used by values.py and ps_values.py."""

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.training.saving import saveable_object


def write_object_proto(var, proto, options):
  # [FUNCTION DOC]
  # HELPER: used by distributed variables while saving to record per-component information.
  # KNOWN ISSUE: the loop variable reuses the name of the parameter var, shadowing the distributed
  # variable inside the loop; harmless because the values are read once before iteration starts.
  # Fills in the pre-built saved-object proto for a distributed variable. Parameters: var is the
  # distributed variable, proto is the saved-object proto (assumed to describe a variable) that is
  # modified in place, options is the save-options object. If the variable policy of the options
  # says distributed variables should be expanded, one component entry is appended to the proto
  # for each component variable, holding its name (without the output-index suffix) and device.
  # Otherwise nothing is changed. Returns nothing.
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
    var: The DistributedVariable object.
    proto: A pre-built `SavedObject` proto for this object. It is assumed this
      will be a `SavedVariable` instance.
    options: A `SaveOptions` instance.
  """
  if options.experimental_variable_policy._expand_distributed_variables(  # pylint: disable=protected-access
  ):
    for var in var.values:
      var_proto = (
          proto.variable.experimental_distributed_variable_components.add())
      var_proto.name = var.name.split(":")[0]
      var_proto.device = var.device


def get_on_write_saveable(var, primary_var, name):
  # [FUNCTION DOC]
  # HELPER: builds the save specification used when checkpointing mirrored variables.
  # Parameters: var is the distributed variable, primary_var is the component variable whose
  # device and initialization state are used, name is the checkpoint key. Returns a pair: a
  # callable producing the tensor to save, and a list holding one save-spec built from that
  # callable (empty slice spec, the variable's dtype, the primary component's device).
  # The callable is lazy so nothing is read when restoring. Creates no side effects itself.
  """Return saveable spec for AUTO and ON_WRITE variables."""
  # We use a callable so that we don't have to evaluate this expression
  # in the case where we are trying to restore instead of save.
  def tensor():
    # [FUNCTION DOC]
    # HELPER: lazy value getter for the save spec.
    # Returns nothing/null value when running eagerly and the primary component is not yet
    # initialized (signals an uninitialized variable to the saver); otherwise reads the variable
    # value through the strategy's extended read function and returns that tensor.
    if context.executing_eagerly() and not primary_var.is_initialized():
      # A SaveSpec tensor value of `None` indicates that the variable is
      # uninitialized.
      return None
    strategy = var.distribute_strategy
    return strategy.extended.read_var(var)

  spec = saveable_object.SaveSpec(
      tensor=tensor,
      slice_spec="",
      name=name,
      dtype=var.dtype,
      device=primary_var.device)

  return tensor, [spec]


def get_on_write_restore_ops(var, tensor):
  # [FUNCTION DOC]
  # HELPER: builds the restore operation for mirrored variables.
  # Parameters: var is the distributed variable, tensor is the restored value. Assigns the value
  # to every component (using the packed variable once per device when the variable is packed,
  # otherwise each component on its own device) and returns a single grouped operation that
  # completes when all assignments are done.
  """Return restore ops for AUTO and ON_WRITE variables."""
  packed_var = var._packed_variable  # pylint: disable=protected-access
  if packed_var is not None:
    return control_flow_ops.group(
        tuple(
            assign_on_device(d, packed_var, tensor)
            for d in packed_var.devices))
  return control_flow_ops.group(
      tuple(
          assign_on_device(v.device, v, tensor)
          for v in var.values))


def get_on_read_saveable(var, primary_var, name):
  # [FUNCTION DOC]
  # HELPER: builds the save specification used when checkpointing sync-on-read variables.
  # Parameters: var is the distributed variable, primary_var supplies the device, name is the
  # checkpoint key. Returns a pair: a callable producing the aggregated (cross-replica) value, and
  # a list with one save-spec built from it (empty slice spec, the variable's dtype).
  """Return saveables for ON_READ variable."""

  # We use a callable so that we don't have to evaluate this expression
  # in the case where we are trying to restore instead of save.
  def tensor():
    # [FUNCTION DOC]
    # HELPER: lazy value getter for the save spec.
    # Returns the cross-replica (aggregated) value of the variable.
    return var._get_cross_replica()  # pylint: disable=protected-access

  spec = saveable_object.SaveSpec(
      tensor=tensor,
      slice_spec="",
      name=name,
      dtype=var.dtype,
      device=primary_var.device)

  return tensor, [spec]


def get_on_read_restore_ops(var, tensor, aggregation):
  # [FUNCTION DOC]
  # HELPER: builds the restore operation for sync-on-read variables.
  # Parameters: var is the distributed variable, tensor is the restored (aggregated) value,
  # aggregation is the variable aggregation mode. When the mode is sum, the value is divided by
  # the number of replicas in sync (and cast back to the variable dtype) so the sum is preserved
  # across save and restore. The value is then assigned to every component on its own device;
  # returns one grouped operation covering all assignments.
  """Return restore ops for ON_READ variables."""
  # To preserve the sum across save and restore, we have to divide the
  # total across all devices when restoring a variable that was summed
  # when saving.
  if aggregation == vs.VariableAggregation.SUM:
    strategy = var.distribute_strategy
    tensor = math_ops.cast(tensor / strategy.num_replicas_in_sync,
                           var.dtype)
  return control_flow_ops.group(
      tuple(
          assign_on_device(v.device, v, tensor)
          for v in var.values))


# Utility function that indicates if you are in an UpdateContext when running
# in a replica fn.
def in_replica_update_context():
  # [FUNCTION DOC]
  # HELPER: context query used by distributed variables.
  # Returns true when the code is running inside an update context entered from a replica function
  # (an update replica id is set), false otherwise. Takes no parameters.
  return distribute_lib.get_update_replica_id() is not None


def on_write_assign(var, value, use_locking=False, name=None, read_value=True):
  # [FUNCTION DOC]
  # HELPER: shared implementation of assign for mirrored (on-write) variables.
  # Forwards an assign of value to the variable's update mechanism, which applies it to the
  # components. Parameters: var distributed variable; value new value; use_locking optional
  # flag (default false); name optional operation name; read_value whether to return the new
  # value (default true). Returns whatever the variable's update returns.
  assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
  return var._update(  # pylint: disable=protected-access
      update_fn=assign_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)


def on_write_assign_add(var, value, use_locking=False, name=None,
                        read_value=True):
  # [FUNCTION DOC]
  # HELPER: shared implementation of assign-add for mirrored (on-write) variables.
  # Same parameters and behavior as the plain assign helper, but adds value to the variable
  # through its update mechanism. Returns the update result.
  assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
  return var._update(  # pylint: disable=protected-access
      update_fn=assign_add_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)


def on_write_assign_sub(var, value, use_locking=False, name=None,
                        read_value=True):
  # [FUNCTION DOC]
  # HELPER: shared implementation of assign-sub for mirrored (on-write) variables.
  # Same parameters and behavior as the plain assign helper, but subtracts value from the variable
  # through its update mechanism. Returns the update result.
  assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
  return var._update(  # pylint: disable=protected-access
      update_fn=assign_sub_fn,
      value=value,
      use_locking=use_locking,
      name=name,
      read_value=read_value)


def assign_on_each_device(var, assign_func, value, read_value):
  # [FUNCTION DOC]
  # HELPER: applies an assignment to every component of a distributed variable.
  # Parameters: var distributed variable; assign_func callback taking a device, a variable and a
  # value; value the value to apply; read_value whether to return the resulting value. Uses the
  # packed variable once per device when present, otherwise each component on its own device,
  # and groups the resulting operations. Returns the grouped update operation when read_value is
  # false, otherwise the variable's read value computed after the update completes.
  """Update the variable on each replica with the given assign_func and value."""
  if var._packed_variable is not None:  # pylint: disable=protected-access
    update = control_flow_ops.group(
        tuple(
            assign_func(d, var._packed_variable, value) for d in var._devices))  # pylint: disable=protected-access
  else:
    update = control_flow_ops.group(
        tuple(assign_func(v.device, v, value) for v in var._values))  # pylint: disable=protected-access
  if not read_value:
    return update
  with ops.control_dependencies([update] if update else []):
    return var.read_value()


def on_read_assign_sub_cross_replica(var, value, read_value=True):
  # [FUNCTION DOC]
  # HELPER: assign-sub for sync-on-read variables in cross-replica context.
  # KNOWN ISSUE: returns nothing (implicit null) when not in cross-replica context, silently
  # skipping the update instead of raising.
  # Enters or asserts the variable's strategy; in cross-replica context, raises a value error if the
  # aggregation is sum (unsupported), else subtracts value on each device. Parameters: var, value,
  # read_value (default true). Returns the update or the new value as in the each-device helper.
  with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
    if distribute_lib.in_cross_replica_context():
      if var.aggregation == vs.VariableAggregation.SUM:
        raise ValueError(
            "SyncOnReadVariable does not support `assign_sub` in "
            "cross-replica context when aggregation is set to "
            "`tf.VariableAggregation.SUM`.")
      return assign_on_each_device(var, assign_sub_on_device,
                                   value, read_value)


def on_read_assign_add_cross_replica(var, value, read_value=True):
  # [FUNCTION DOC]
  # HELPER: assign-add for sync-on-read variables in cross-replica context.
  # KNOWN ISSUE: returns nothing (implicit null) when not in cross-replica context, silently
  # skipping the update instead of raising.
  # Enters or asserts the variable's strategy; in cross-replica context, raises a value error if the
  # aggregation is sum (unsupported), else adds value on each device. Parameters: var, value,
  # read_value (default true). Returns the update or the new value as in the each-device helper.
  with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
    if distribute_lib.in_cross_replica_context():
      if var.aggregation == vs.VariableAggregation.SUM:
        raise ValueError(
            "SyncOnReadVariable does not support `assign_add` in "
            "cross-replica context when aggregation is set to "
            "`tf.VariableAggregation.SUM`.")
      return assign_on_each_device(var, assign_add_on_device,
                                   value, read_value)


def on_read_assign_cross_replica(var, value, read_value=True):
  # [FUNCTION DOC]
  # HELPER: assign for sync-on-read variables in cross-replica context.
  # KNOWN ISSUE: the docstring says it returns the value of the variable, but it actually assigns
  # value to every component; it also returns nothing (implicit null) outside cross-replica context.
  # In cross-replica context, if aggregation is sum the value is divided by the number of replicas
  # in sync (cast to the variable dtype) so the total is preserved, then assigned on each device.
  # Parameters: var, value, read_value (default true). Returns the update or the new value.
  """Return the value of the variable in cross replica context."""
  with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
    if distribute_lib.in_cross_replica_context():
      # To preserve the sum across save and restore, we have to divide the
      # total across all devices when restoring a variable that was summed
      # when saving.
      tensor = value
      if var.aggregation == vs.VariableAggregation.SUM:
        strategy = var._distribute_strategy  # pylint: disable=protected-access
        tensor = math_ops.cast(tensor / strategy.num_replicas_in_sync,
                               var.dtype)
      return assign_on_each_device(var, assign_on_device, tensor,
                                   read_value)


def scatter_sub(var, sparse_delta, use_locking=False, name=None):
  # [FUNCTION DOC]
  # HELPER: forwards a scatter-subtract to the variable's update mechanism.
  # Parameters: var distributed variable; sparse_delta indexed-slices style update; use_locking
  # optional flag (default false); name optional operation name. Returns the update result.
  scatter_sub_fn = lambda var, *a, **kw: var.scatter_sub(*a, **kw)
  return var._update(  # pylint: disable=protected-access
      update_fn=scatter_sub_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)


def scatter_add(var, sparse_delta, use_locking=False, name=None):
  # [FUNCTION DOC]
  # HELPER: forwards a scatter-add to the variable's update mechanism.
  # Parameters: var distributed variable; sparse_delta indexed-slices style update; use_locking
  # optional flag (default false); name optional operation name. Returns the update result.
  scatter_add_fn = lambda var, *a, **kw: var.scatter_add(*a, **kw)
  return var._update(  # pylint: disable=protected-access
      update_fn=scatter_add_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)


def scatter_mul(var, sparse_delta, use_locking=False, name=None):
  # [FUNCTION DOC]
  # HELPER: forwards a scatter-multiply to the variable's update mechanism.
  # Parameters: var distributed variable; sparse_delta indexed-slices style update; use_locking
  # optional flag (default false); name optional operation name. Returns the update result.
  scatter_mul_fn = lambda var, *a, **kw: var.scatter_mul(*a, **kw)
  return var._update(  # pylint: disable=protected-access
      update_fn=scatter_mul_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)


def scatter_div(var, sparse_delta, use_locking=False, name=None):
  # [FUNCTION DOC]
  # HELPER: forwards a scatter-divide to the variable's update mechanism.
  # Parameters: var distributed variable; sparse_delta indexed-slices style update; use_locking
  # optional flag (default false); name optional operation name. Returns the update result.
  scatter_div_fn = lambda var, *a, **kw: var.scatter_div(*a, **kw)
  return var._update(  # pylint: disable=protected-access
      update_fn=scatter_div_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)


def scatter_min(var, sparse_delta, use_locking=False, name=None):
  # [FUNCTION DOC]
  # HELPER: forwards a scatter-minimum to the variable's update mechanism.
  # Parameters: var distributed variable; sparse_delta indexed-slices style update; use_locking
  # optional flag (default false); name optional operation name. Returns the update result.
  scatter_min_fn = lambda var, *a, **kw: var.scatter_min(*a, **kw)
  return var._update(  # pylint: disable=protected-access
      update_fn=scatter_min_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)


def scatter_max(var, sparse_delta, use_locking=False, name=None):
  # [FUNCTION DOC]
  # HELPER: forwards a scatter-maximum to the variable's update mechanism.
  # Parameters: var distributed variable; sparse_delta indexed-slices style update; use_locking
  # optional flag (default false); name optional operation name. Returns the update result.
  scatter_max_fn = lambda var, *a, **kw: var.scatter_max(*a, **kw)
  return var._update(  # pylint: disable=protected-access
      update_fn=scatter_max_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)


def scatter_update(var, sparse_delta, use_locking=False, name=None):
  # [FUNCTION DOC]
  # HELPER: forwards a scatter-update (overwrite) to the variable's update mechanism.
  # Parameters: var distributed variable; sparse_delta indexed-slices style update; use_locking
  # optional flag (default false); name optional operation name. Returns the update result.
  scatter_update_fn = lambda var, *a, **kw: var.scatter_update(*a, **kw)
  return var._update(  # pylint: disable=protected-access
      update_fn=scatter_update_fn,
      value=sparse_delta,
      use_locking=use_locking,
      name=name)


def get_current_replica_id_as_int():
  # [FUNCTION DOC]
  # HELPER: replica-id lookup for distributed variables.
  # Returns the current replica id as a plain integer. In a replica context the id is taken from the
  # context and, if it is a tensor, converted with constant folding (may give null if not constant).
  # Outside a replica context returns the update replica id, which may be null. Takes no parameters.
  """Returns the current replica ID as an integer, or `None`."""
  replica_context = distribute_lib.get_replica_context()
  if replica_context:
    replica_id = replica_context._replica_id  # pylint: disable=protected-access
    if not isinstance(replica_id, int):
      replica_id = tensor_util.constant_value(replica_id)
  else:
    replica_id = distribute_lib.get_update_replica_id()
  return replica_id


def assign_on_device(device, variable, tensor):
  # [FUNCTION DOC]
  # HELPER: places an assignment on a specific device.
  # Under a device scope for device, assigns tensor to variable and returns the assign result.
  with ops.device(device):
    return variable.assign(tensor)


def assign_add_on_device(device, variable, tensor):
  # [FUNCTION DOC]
  # HELPER: places an add-assignment on a specific device.
  # Under a device scope for device, adds tensor to variable and returns the result.
  with ops.device(device):
    return variable.assign_add(tensor)


def assign_sub_on_device(device, variable, tensor):
  # [FUNCTION DOC]
  # HELPER: places a subtract-assignment on a specific device.
  # Under a device scope for device, subtracts tensor from variable and returns the result.
  with ops.device(device):
    return variable.assign_sub(tensor)


def assert_replica_context(strategy):
  # [FUNCTION DOC]
  # HELPER: validation for assigning to replica-local variables.
  # Raises a runtime error if there is no current replica context, or if the current replica
  # context belongs to a different strategy than the given one. Returns nothing.
  replica_context = distribute_lib.get_replica_context()
  if not replica_context:
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")
  if replica_context.strategy is not strategy:
    raise RuntimeError(
        "Replica-local variables may only be assigned in a replica context.")


def apply_aggregation(strategy, value, aggregation, destinations):
  # [FUNCTION DOC]
  # HELPER: combines per-replica values before updating a variable from replica context.
  # Parameters: strategy distribution strategy; value per-replica value; aggregation variable
  # aggregation mode; destinations where the result should live. For only-first-replica, the first
  # local result is broadcast to the destinations; otherwise the value is reduced to the
  # destinations with the reduce operation matching the aggregation. Returns the resulting value.
  if aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
    return strategy.extended.broadcast_to(
        strategy.experimental_local_results(value)[0],
        destinations=destinations)
  reduce_op = reduce_util.ReduceOp.from_variable_aggregation(aggregation)
  return strategy.extended.reduce_to(reduce_op, value, destinations)


aggregation_error_msg = (
    "You must specify an aggregation method to update a "
    "{variable_type} in Replica Context. You can do so by passing "
    "an explicit value for argument `aggregation` to tf.Variable(..)."
    "e.g. `tf.Variable(..., aggregation=tf.VariableAggregation.SUM)`"
    "`tf.VariableAggregation` lists the possible aggregation methods."
    "This is required because {variable_type} should always be "
    "kept in sync. When updating them or assigning to them in a "
    "replica context, we automatically try to aggregate the values "
    "before updating the variable. For this aggregation, we need to "
    "know the aggregation method. "
    "Another alternative is to not try to update such "
    "{variable_type} in replica context, but in cross replica "
    "context. You can enter cross replica context by calling "
    "`tf.distribute.get_replica_context().merge_call(merge_fn, ..)`."
    "Inside `merge_fn`, you can then update the {variable_type} "
    "using `tf.distribute.StrategyExtended.update()`.")


scatter_error_msg = ("{op_name} is only supported for mirrored "
                     "variable (variable created within certain "
                     "`tf.distribute.Strategy` scope) with NONE or "
                     "`ONLY_FIRST_REPLICA` aggregation, got: {aggregation}.")


def is_saving_non_distributed():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it is true iff the policy is NONE, but the code returns true
  # for any policy other than expand-distributed-variables.
  # Returns true only inside a save context whose variable policy is not the one that expands
  # distributed variables (so a non-distributed model is being saved); false outside saving.
  # Takes no parameters.
  """Returns whether we're saving a non-distributed version of the model.

  It returns True iff we are in saving context and are saving a non-distributed
  version of the model. That is, SaveOptions.experimental_variable_policy is
  NONE.

  Returns:
    A boolean.
  """
  if not save_context.in_save_context():
    return False
  options = save_context.get_save_options()
  return (options.experimental_variable_policy !=
          save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES)


def mark_as_unsaveable():
  # [FUNCTION DOC]
  # HELPER: guards against saving concrete functions that use distributed variables unsafely.
  # When called while building a function (graph) and not inside a save context, marks the default
  # graph as unsaveable with an explanatory message suggesting an input signature. Otherwise does
  # nothing. Takes no parameters, returns nothing.
  """Marks the function as unsaveable if not inside save context."""
  if ops.inside_function() and not save_context.in_save_context():
    ops.get_default_graph().mark_as_unsaveable("""
ConcreteFunction that uses distributed variables in certain way cannot be saved.
If you're saving with

tf.saved_model.save(..., signatures=f.get_concrete_function())

do

@tf.function(input_signature=...)
def f_with_input_signature():
  ...

tf.saved_model.save(..., signatures=f_with_input_signature)`

instead.""")
