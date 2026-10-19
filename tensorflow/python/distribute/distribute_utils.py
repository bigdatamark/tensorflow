# [MODULE DOC]
# KNOWN ISSUE: the module docstring says "Class implementing utilities", but the module is a
# collection of functions plus one small thread-local class.
# Utility functions shared by the tf.distribute strategy implementations. It provides: regrouping
# of per-replica values into per-replica or mirrored wrappers (and the inverse selection of one
# replica's value), lookup of the container of a component of a distributed variable, validation
# of variable synchronization, aggregation and colocation arguments, a shared routine that creates
# mirrored/distributed variables for synchronous strategies, predicates for kinds of distributed
# values, a thread-local variable-read caching scope, and constant mappings from synchronization
# modes to variable classes and policies (for regular and TPU devices).
# Import-time side effects: registers one public API symbol (the loss reduction getter) and
# creates the module-level thread-local caching state.
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
"""Class implementing utilities used by tf.distribute.Strategy."""

from collections import abc
import contextlib
import threading

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_values as tpu_values_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.reduce_util import ReduceOp
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["distribute.get_loss_reduction"])
def get_loss_reduction():
  # [FUNCTION DOC]
  # Returns the reduce op matching the loss reduction last recorded on the default graph. No
  # parameters. If the current strategy does not scale loss for the estimator, returns the sum
  # op. Otherwise reads the last loss reduction from the default graph and returns sum when it
  # is the sum reduction (enum or the string "sum"), else mean.
  """`tf.distribute.ReduceOp` corresponding to the last loss reduction.

  Returns:
    `tf.distribute.ReduceOp` corresponding to the last loss reduction for
    estimator and v1 optimizer use case. `tf.distribute.ReduceOp.SUM` otherwise.
  """
  if not distribute_lib.get_strategy()._scale_loss_for_estimator:  # pylint: disable=protected-access
    # If we are not in Estimator context then return 'SUM'. We do not need to
    # scale loss in the optimizer.
    return ReduceOp.SUM
  last_reduction = ops.get_default_graph()._last_loss_reduction  # pylint: disable=protected-access
  if (last_reduction == losses_impl.Reduction.SUM or
      last_reduction == "sum"):  # Check for tf.keras.losses.Reduction.SUM
    return ReduceOp.SUM
  return ReduceOp.MEAN


def regroup(values, wrap_class=values_lib.PerReplica, always_wrap=False):
  # [FUNCTION DOC]
  # Converts a list (one entry per replica) of identically structured nests into a single nest
  # whose leaves combine the per-replica leaves. Parameters: values, a non-empty list with one
  # nest per replica; wrap_class, the class used to wrap leaf lists (default per-replica value);
  # always_wrap, when true wraps even if all replicas hold the very same object (default false;
  # distributed variables are still returned unwrapped).
  # Behavior: recurses through lists, tuples (named tuples are rebuilt as the same type) and
  # mappings (same type as the first), asserting that all replicas have the same lengths/keys
  # (assertion errors otherwise). At a leaf: if every replica has the identical object, it is
  # returned (distributed variables always; others unless always_wrap and the object has no
  # container). If the leaves are components of one distributed variable, returns that variable's
  # container after checking all share it. Otherwise wraps the leaf list in wrap_class.
  # Returns the regrouped nest. No side effects.
  """Makes a nest per-replica into a nest of PerReplica/Mirrored values.

  Args:
    values: Values to regroup
    wrap_class: Class that `values` be wrapped in.
    always_wrap: Always wrap the `values` in `wrap_class` even if the values
        are the same except for DistributeVariable.
  Returns:
    Wrapped `values`.
  """
  v0 = values[0]

  if isinstance(v0, list):
    for v in values[1:]:
      assert isinstance(v, list)
      assert len(v) == len(v0), ("len(v) == %d, len(v0) == %d, v: %s, v0: %s" %
                                 (len(v), len(v0), v, v0))
    return [
        regroup(tuple(v[i] for v in values), wrap_class, always_wrap)
        for i in range(len(v0))
    ]

  if isinstance(v0, tuple):
    for v in values[1:]:
      assert isinstance(v, tuple)
      assert len(v) == len(v0), ("Values to regroup had different lengths: "
                                 f"len(v) == {len(v)}, len(v0) == {len(v0)}, "
                                 f"v: {v}, v0: {v0}")
    regrouped_tuple = tuple(
        regroup(tuple(v[i] for v in values), wrap_class, always_wrap)
        for i in range(len(v0)))
    if hasattr(v0, "_fields"):
      # This tuple is in fact a namedtuple! Create a new namedtuple instance
      # and initialize it with the regrouped values:
      assert hasattr(v0, "_make")
      return v0._make(regrouped_tuple)
    else:
      return regrouped_tuple

  if isinstance(v0, abc.Mapping):
    v0keys = v0.keys()
    for v in values[1:]:
      assert isinstance(v, abc.Mapping), ("v[0]: %r  v[i]: %r" % (v0, v))
      assert set(v.keys()) == set(v0keys), ("v[0].keys: %s  v[i].keys: %s" %
                                            (set(v0keys), set(v.keys())))
    # Use the actual type in case it is a class inherited from a dict.
    return type(v0)({
        key: regroup(tuple(v[key] for v in values),
                     wrap_class, always_wrap)
        for key in v0keys
    })

  # If exactly the same object across all devices, return it unwrapped.
  same_id = True
  for v in values[1:]:
    if v is not v0:
      same_id = False
      break
  # Consider three cases where same_id is true:
  # * If v0 is a DistributedVariable (a MirroredVariable or
  #   SyncOnReadVariable, and same_id means it is the same across all
  #   devices), we want to return it. We check DistributedVariable
  #   specifically since it can look like it has a
  #   _distributed_container member since its members do.
  if same_id and isinstance(v0, values_lib.DistributedVariable):
    return v0
  # * If v0 is a member of a distributed variable, in which case
  #   value_container(v0) is not v0 itself, we want to
  #   return the DistributedVariable that contains it using the
  #   _distributed_container logic below. This case can trigger
  #   same_id when there is only one device.
  # * In any other situation, same_id means we return v0 unless `always_wrap` is
  #   true.
  if same_id and not always_wrap and value_container(v0) is v0:
    return v0

  # Detect the case where each device has a parallel component of the
  # same MirroredVariable (or SyncOnReadVariable). In this case we
  # want to return the containing MirroredVariable, after a bunch of
  # sanity checking. In particular, each component should have the
  # same container, and the devices of the variables should match the
  # keys of the per-replica dictionary. For _UnreadVariables, use the wrap_class
  # path, which calls tf.identity on them.
  if (not isinstance(v0, resource_variable_ops._UnreadVariable) and  # pylint: disable=protected-access
      value_container(v0) is not v0):
    # pylint: disable=protected-access
    assert not isinstance(v0, values_lib.MirroredVariable), (
        "ids = %s, values = %s" % ([id(v) for v in values], values))
    distributed_container = value_container(v0)
    assert distributed_container is not None
    for v in values[1:]:
      assert distributed_container is value_container(v)
    return distributed_container
  # pylint: enable=protected-access

  return wrap_class(values)


def select_replica(replica_id, structured):
  # [FUNCTION DOC]
  # Picks the values belonging to one replica from a nest. Parameters: replica_id, integer
  # index of the replica; structured, a nest of regular and distributed values. Returns a nest of
  # the same structure where each distributed value (but not a distributed variable, which is
  # kept whole) is replaced by its component for that replica and other values are unchanged.
  """Specialize a nest of regular & per-replica values for one replica."""

  def _get(x):
    # [FUNCTION DOC]
    # HELPER: per-leaf mapper for the replica selection above. Returns the leaf itself if it is a
    # distributed variable or not a distributed value; otherwise returns the component for the
    # captured replica id.
    # `DistributedValues` would be sliced according to replica unless it is a
    # `DistributedVariable` because `DistributedVariable` can be handled
    # directly in the replica context.
    if (isinstance(x, values_lib.DistributedVariable) or
        not isinstance(x, values_lib.DistributedValues)):
      return x
    else:
      return x.values[replica_id]

  return nest.map_structure(_get, structured)


def select_replica_mirrored(replica_id, structured):
  # [FUNCTION DOC]
  # Like the replica selection function but first asserts that every distributed value in the
  # nest is mirrored (raising a type error otherwise). Parameters: replica_id (integer) and
  # structured (nest). Returns the nest specialized for that replica.
  """Specialize a nest of regular & mirrored values for one replica."""
  assert_mirrored(structured)
  return select_replica(replica_id, structured)


def assert_mirrored(structured):
  # [FUNCTION DOC]
  # Validates a nest: every distributed value in it must be mirrored (regular values are fine).
  # Parameter: structured, a nest. Raises a type error naming the offending value when a
  # non-mirrored distributed value is found. Returns nothing.
  """Raises if the structured is not composed of mirrored or regular values."""

  def _assert_mirrored(x):
    # [FUNCTION DOC]
    # HELPER: per-leaf check used by the nest validation above; raises a type error if the leaf
    # is a distributed value that is not mirrored. Returns nothing.
    if isinstance(x, values_lib.DistributedValues) and not is_mirrored(x):
      raise TypeError(
          "Expected value to be mirrored across replicas: %s in %s." %
          (x, structured))

  nest.map_structure(_assert_mirrored, structured)


def update_regroup(extended, updates, group):
  # [FUNCTION DOC]
  # Combines the per-replica results of an update. Parameters: extended, the strategy extension
  # (used for its local-results method); updates, list of per-replica update results (nests);
  # group, boolean. If group is false, regroups the updates into mirrored values and returns the
  # nest of local results (lists of per-replica values). If true, returns a regrouped nest of
  # mirrored values where, when more than one replica exists, each value is made to depend on a
  # grouped op of all updates so running it executes every replica's update (if the values are
  # not all tensors, the grouped op itself is returned for them).
  """Regroup for an update, with dependencies to ensure all updates execute."""
  if not group:
    regrouped = regroup(updates, values_lib.Mirrored)
    return nest.map_structure(extended._local_results, regrouped)  # pylint: disable=protected-access

  def _make_grouped_mirrored(values):
    # [FUNCTION DOC]
    # HELPER: wrapper function for the update regrouping above. Takes the list of per-replica
    # leaves; a single replica is just wrapped as mirrored. Otherwise groups all leaves into one
    # op so every update runs, returns that op if the leaves are not tensors, else returns a
    # mirrored value of identity copies placed on each leaf's device with a control dependency on
    # the group.
    """Convert per-replica list `values` into Mirrored type with grouping."""
    if len(values) == 1:
      return values_lib.Mirrored(values)

    # Make sure we run all updates. Without this, something like
    # session.run(extended.update(...)) may only update one replica.
    g = control_flow_ops.group(values)

    # If values is just ops, the grouping is enough. Everything in values
    # should have the same type, since we expect every replica to be performing
    # the same computation.
    if not all(tensor_util.is_tf_type(v) for v in values):
      return g

    # Otherwise we need tensors with the same values as `values`, but
    # that have a dependency on `g`.
    with_dep = []
    for v in values:
      with ops.device(v.device), ops.control_dependencies([g]):
        with_dep.append(array_ops.identity(v))

    return values_lib.Mirrored(with_dep)

  return regroup(updates, _make_grouped_mirrored)


def value_container(val):
  # [FUNCTION DOC]
  # Finds the distributed container (e.g. the mirrored variable) that a per-replica component
  # belongs to. Parameter: val, any value. Distributed variables themselves are never mapped.
  # For other values, uses the stored weak container reference on the value, or on the handle of
  # a composite-tensor resource variable; returns the container if still alive, otherwise (no
  # container or already destroyed) returns the value itself.
  """Returns the container that this per-replica `value` belongs to.

  Args:
    val: A value returned by `call_for_each_replica()` or a variable created in
      `scope()`.

  Returns:
    A container that `value` belongs to.
    If value does not belong to any container (including the case of
    container having been destroyed), returns the value itself.
  """
  # DistributedVariable has _distributed_container defined but we don't want to
  # return it.
  container = None
  if not isinstance(val, values_lib.DistributedVariable):
    if hasattr(val, "_distributed_container"):
      container = val._distributed_container()  # pylint: disable=protected-access
    elif (isinstance(val, composite_tensor.CompositeTensor) and
          hasattr(val, "handle") and
          hasattr(val.handle, "_distributed_container")):
      # For ResourceVariables, the _distributed_container attribute
      # is added to their handle tensors.
      container = val.handle._distributed_container()  # pylint: disable=protected-access
  return container if container is not None else val


def is_distributed_variable(v):
  # [FUNCTION DOC]
  # Returns true when the object advertises the distributed-variable marker attribute (true
  # value); false when the attribute is missing or false. Parameter: v, any object.
  """Determine if a variable is ds variable or TPU mirrored variable."""
  return getattr(v, "is_distributed_variable", False)


def is_distributed_table(v):
  # [FUNCTION DOC]
  # Returns true when the object advertises the distributed-table marker attribute (true value);
  # false when missing or false. Parameter: v, any object.
  """Determine if an object is a DistributedTable."""
  return getattr(v, "is_distributed_table", False)


def _validate_colocate_extended(v, extended):
  # [FUNCTION DOC]
  # HELPER: shared check for the colocation validators below.
  # Verifies the variable was created under the strategy owning the given extension. Parameters:
  # v, a variable with a strategy attribute; extended, the strategy extension. Raises a value
  # error if the variable has no strategy or its strategy's extension is a different one.
  variable_strategy = v._distribute_strategy  # pylint: disable=protected-access
  if not variable_strategy or variable_strategy.extended is not extended:
    raise ValueError(
        "`colocate_vars_with` must only be passed a variable created in this "
        "tf.distribute.Strategy.scope(), not %s created in scope: %s" %
        (v, variable_strategy))


def validate_colocate_distributed_variable(v, extended):
  # [FUNCTION DOC]
  # Validates a colocation request for a variable that must be a distributed variable.
  # Parameters: v, the object to colocate with; extended, the strategy extension. Raises a value
  # error if v is not a distributed variable or was not created in the extension's strategy.
  # Returns nothing.
  if not isinstance(v, values_lib.DistributedVariable):
    raise ValueError(
        "`colocate_vars_with` must only be passed a variable created in this "
        "tf.distribute.Strategy.scope(), not: %r" % (v,))
  _validate_colocate_extended(v, extended)


def validate_colocate(v, extended):
  # [FUNCTION DOC]
  # Validates a colocation request for any object carrying a strategy attribute. Parameters: v,
  # the object to colocate with; extended, the strategy extension. Raises a value error if v has
  # no strategy attribute or its strategy belongs to a different extension. Returns nothing.
  if not hasattr(v, "_distribute_strategy"):
    raise ValueError(
        "`colocate_vars_with` must only be passed a variable created in this "
        "tf.distribute.Strategy.scope(), not: %r" % (v,))
  _validate_colocate_extended(v, extended)


# Variable creation function for sync strategies.
def _validate_synchronization(kwargs):
  # [FUNCTION DOC]
  # HELPER: variable-creation argument check for create_mirrored_variable.
  # Reads the synchronization mode from the mapping of variable-creation arguments (default
  # auto). Raises a value error if it is the none mode (unsupported, message names the variable)
  # or not one of on-read, on-write, auto. Returns on-write for auto, otherwise the mode itself.
  """Validate that given synchronization value is valid."""
  synchronization = kwargs.get("synchronization",
                               vs.VariableSynchronization.AUTO)
  if synchronization == vs.VariableSynchronization.NONE:
    raise ValueError(
        "`NONE` variable synchronization mode is not supported with "
        "tf.distribute strategy. Please change the `synchronization` for "
        "variable: " + str(kwargs["name"]))
  if synchronization not in (vs.VariableSynchronization.ON_READ,
                             vs.VariableSynchronization.ON_WRITE,
                             vs.VariableSynchronization.AUTO):
    raise ValueError(
        "Invalid variable synchronization mode: %s for variable: %s" %
        (synchronization, kwargs["name"]))
  if synchronization == vs.VariableSynchronization.AUTO:
    return vs.VariableSynchronization.ON_WRITE
  return synchronization


def _validate_aggregation(kwargs):
  # [FUNCTION DOC]
  # HELPER: variable-creation argument check for create_mirrored_variable.
  # Reads the aggregation mode from the mapping of variable-creation arguments (default none).
  # Raises a value error if it is not one of none, sum, mean, only-first-replica. Returns it.
  aggregation = kwargs.get("aggregation", vs.VariableAggregation.NONE)

  if aggregation not in (vs.VariableAggregation.NONE,
                         vs.VariableAggregation.SUM,
                         vs.VariableAggregation.MEAN,
                         vs.VariableAggregation.ONLY_FIRST_REPLICA):
    raise ValueError("Invalid variable aggregation mode: %s for variable: %s" %
                     (aggregation, kwargs["name"]))
  return aggregation


def create_mirrored_variable(strategy, real_mirrored_creator, class_mapping,
                             policy_mapping, **kwargs):
  # [FUNCTION DOC]
  # Creates one distributed (mirrored or sync-on-read) variable from per-device component
  # variables. Parameters: strategy, the owning strategy; real_mirrored_creator, callback taking
  # the variable-creation keyword arguments and returning the list of per-device variables;
  # class_mapping, mapping from synchronization mode (or the variable-class keys) to the
  # distributed variable class; policy_mapping, mapping from synchronization mode to policy
  # class; keyword arguments are the variable-creation arguments (collections, synchronization,
  # aggregation, trainable, caching device, batch-initialization flag, and others passed to the
  # creator).
  # Behavior: pops the batch-initialization flag (selects the lazy variable class key) and the
  # collections (default global variables), then passes an empty collections list to the creator;
  # validates and normalizes synchronization (auto becomes on-write) and aggregation; drops the
  # caching device. With recording on the gradient tape paused, calls the creator; component
  # variables lacking an initializer op get a no-op one. If the extension uses variable policies,
  # builds the policy and wraps with the variable class; else wraps with the class chosen by
  # synchronization. In graph mode, removes the components from the trainable collection when
  # trainable (default true) and registers the wrapper in the requested collections (adding
  # trainable); in eager mode only the global-step collection is honored.
  # Returns the wrapper variable. Raises a value error for invalid synchronization/aggregation.
  """Create distributed variables with given synchronization and aggregation."""
  # Figure out what collections this variable should be added to.
  # We'll add the MirroredVariable to those collections instead.

  if kwargs.pop("experimental_batch_initialization", None):
    variable_class_key = "LazyVariableClass"
  else:
    variable_class_key = "VariableClass"

  var_collections = kwargs.pop("collections", None)
  if var_collections is None:
    var_collections = [ops.GraphKeys.GLOBAL_VARIABLES]
  kwargs["collections"] = []

  synchronization = _validate_synchronization(kwargs)
  # Update synchronization in kwargs in case it's AUTO, which is converted to
  # ON_WRITE.
  kwargs["synchronization"] = synchronization
  aggregation = _validate_aggregation(kwargs)
  use_var_policy = getattr(strategy.extended, "_use_var_policy", False)

  # Ignore user-specified caching device, not needed for mirrored variables.
  kwargs.pop("caching_device", None)

  # TODO(josh11b,apassos): It would be better if variable initialization
  # was never recorded on the tape instead of having to do this manually
  # here.
  with record.stop_recording():
    value_list = real_mirrored_creator(**kwargs)
    # MirroredVariable is recreated during saved_model loading, and its
    # component variables (value_list) will have None initializer. We
    # set their initializers to no_op so that consumer like
    # `global_variables_initializer` wouldn't complain, as it groups all
    # variables' initializers thus all variables have to have initializers.
    for v in value_list:
      # pylint:disable=protected-access
      if hasattr(v, "_initializer_op") and v._initializer_op is None:
        v._initializer_op = control_flow_ops.no_op()
      # pylint:enable=protected-access
    if use_var_policy:
      var_policy_cls = policy_mapping.get(synchronization)
      var_policy = var_policy_cls(aggregation=aggregation)
      var_cls = class_mapping.get(variable_class_key)
      result = var_cls(strategy, value_list, aggregation, var_policy=var_policy)
    else:
      var_cls = class_mapping.get(synchronization)
      result = var_cls(strategy, value_list, aggregation)

  # Add the wrapped variable to the requested collections.
  # The handling of eager mode and the global step matches
  # ResourceVariable._init_from_args().
  if not context.executing_eagerly():
    g = ops.get_default_graph()
    # If "trainable" is True, next_creator() will add the member variables
    # to the TRAINABLE_VARIABLES collection, so we manually remove
    # them and replace with the MirroredVariable. We can't set
    # "trainable" to False for next_creator() since that causes functions
    # like implicit_gradients to skip those variables.
    if kwargs.get("trainable", True):
      var_collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
      l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
      for value in value_list:
        for i, trainable_variable in enumerate(l):
          if value is trainable_variable:
            del l[i]
            break

    g.add_to_collections(var_collections, result)
  elif ops.GraphKeys.GLOBAL_STEP in var_collections:
    ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, result)

  return result


# Utility functions
# Return True if the Value is Mirrored or the Variable is replicated and kept in
# sync.
def is_mirrored(val):
  # [FUNCTION DOC]
  # Returns true if the value reports itself as mirrored (replicated and kept in sync) through
  # its mirrored-check method; values lacking that method count as not mirrored.
  return (getattr(val, "_is_mirrored", lambda: False))()


def is_sync_on_read(val):
  # [FUNCTION DOC]
  # Returns true when the value is not mirrored (treated as sync-on-read). Note this is only the
  # negation of the mirrored check, so any non-mirrored value (including ordinary values) counts.
  return not is_mirrored(val)


class CachingScopeLocal(threading.local):
  # [CLASS DOC]
  # Thread-local state tracking whether the current thread is inside a variable-read caching
  # scope. Holds two counters: scopes entered and scopes exited; the thread is inside a scope
  # when entered exceeds exited. A single module-level instance is used by the caching context
  # manager below. Each thread sees its own counters starting at zero.
  """Class for maintaining thread local state for caching scope."""

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the counters of entered and exited caching scopes to zero (runs once per thread
    # that touches the instance).
    super(CachingScopeLocal, self).__init__()
    self.new_cache_scope_count = 0
    self.cache_scope_exited_count = 0

  def enter_scope(self):
    # [FUNCTION DOC]
    # Records entering a caching scope by incrementing the entered counter for this thread.
    self.new_cache_scope_count += 1

  def exit_scope(self):
    # [FUNCTION DOC]
    # Records leaving a caching scope by incrementing the exited counter for this thread.
    self.cache_scope_exited_count += 1

  def in_caching_scope(self):
    # [FUNCTION DOC]
    # Returns true when this thread has entered more caching scopes than it has exited.
    return self.new_cache_scope_count > self.cache_scope_exited_count


caching_scope_local = CachingScopeLocal()


@contextlib.contextmanager
def cache_variable_reads():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the cleanup step that records scope exit runs even when the nested-scope error
  # is raised before this scope was ever entered, so a rejected nested attempt still increments
  # the exited counter; the counters then no longer match the real nesting depth (the outer scope
  # appears closed early).
  # Context manager enabling variable-read caching for the current thread. No parameters. On
  # entry, raises a value error if the thread is already in a caching scope (nesting is
  # unsupported); otherwise records entry and yields nothing. On exit (including errors) records
  # the exit. Side effects: mutates the thread-local module-level counters.
  """Scope for caching variable reads for AggregatingVariable.

  The variable reads for AggregatingVariable inside this scope are cached. i.e.
  the first read of variable reads the value from possibly remote handle, but
  subsequent reads are returned using local cached value.

  For example:
  strategy = ParameterServerStrategy...
  with strategy.scope():
    # Variable v is of AggregatingVariable type with actual variable residing
    # on PS.
    v = tf.Variable(1.0)

  with distribute_utils.cache_variable_reads():
    v.read_value()  # Reads value 1.0
    v.assign(constant_op.constant(5.0))  # v changes to 5.0
    t1 = v.read_value()
    t2 = v.read_value()  # Both t1 & t2 return cached value 1.0 from local CPU.

  Notes about cache_variable_reads scope:
  1. Nesting of scope cache_variable_reads() is not supported
  2. And when caching scope is enabled, the thread enabling the cache and
    mirrored_run._MirroredReplicaThread threads spawned from it will have
    caching enabled.

  Yields:
    A context for caching variables.
  """

  try:
    if caching_scope_local.in_caching_scope():
      # There is nested cache scope, which is not supported.
      raise ValueError("cache_variable_reads scope cannot be nested")
    caching_scope_local.enter_scope()
    yield
  finally:
    caching_scope_local.exit_scope()


# The following mapping indicates the policy that you must use for a given
# variable `synchronization` and `aggregation` pair.
# OnWritePolicy is used for:
# (synchronization=Auto, aggregation=NONE,SUM,MEAN,ONLY_FIRST_REPLICA)
# (synchronization=ON_WRITE, aggregation=NONE,SUM,MEAN,ONLY_FIRST_REPLICA)
# OnReadPolicy is used for:
# (synchronization=ON_READ, aggregation=NONE,SUM,MEAN,ONLY_FIRST_REPLICA)
VARIABLE_POLICY_MAPPING = {
    vs.VariableSynchronization.ON_WRITE: values_lib.OnWritePolicy,
    vs.VariableSynchronization.ON_READ: values_lib.OnReadPolicy,
}

VARIABLE_CLASS_MAPPING = {
    "VariableClass": values_lib.DistributedVariable,
    vs.VariableSynchronization.ON_WRITE: values_lib.MirroredVariable,
    vs.VariableSynchronization.ON_READ: values_lib.SyncOnReadVariable,
}

TPU_VARIABLE_POLICY_MAPPING = {
    vs.VariableSynchronization.ON_WRITE: tpu_values_lib.TPUOnWritePolicy,
    vs.VariableSynchronization.ON_READ: tpu_values_lib.TPUOnReadPolicy,
}

TPU_VARIABLE_CLASS_MAPPING = {
    "VariableClass": tpu_values_lib.TPUDistributedVariable,
    "LazyVariableClass": tpu_values_lib.TPULazyDistributedVariable,
    vs.VariableSynchronization.ON_WRITE: tpu_values_lib.TPUMirroredVariable,
    vs.VariableSynchronization.ON_READ: tpu_values_lib.TPUSyncOnReadVariable,
}
