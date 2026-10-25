# [MODULE DOC]
# Core of the trackable system: defines the Trackable base class that checkpoints and saved
# models traverse, plus its supporting pieces: named (strong and weak) dependency references,
# initial-value wrappers for restore-on-create variables, a saveable that never restores, and
# helpers (decorator and context managers) that temporarily disable automatic or manual
# dependency tracking. Re-exports the checkpoint key constants and the save-type enumeration from
# the constants module. Import-time side effect: several classes and functions are registered
# as public internal API symbols through the export decorator.
"""An object-local variable management scheme."""
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
import collections
import weakref

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.trackable import constants
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export

OBJECT_GRAPH_PROTO_KEY = constants.OBJECT_GRAPH_PROTO_KEY
VARIABLE_VALUE_KEY = constants.VARIABLE_VALUE_KEY
OBJECT_CONFIG_JSON_KEY = constants.OBJECT_CONFIG_JSON_KEY
SaveType = constants.SaveType


@tf_export("__internal__.tracking.TrackableReference", v1=[])
class TrackableReference(object):
  # [CLASS DOC]
  # Immutable (slotted) pair of a local dependency name and the Trackable object it refers to.
  # Held in the dependency lists of Trackable objects and produced when overriding the checkpoint
  # dependencies. Behaves like a two-element tuple: iterable as (name, ref), and equal to a plain
  # pair or another reference with the same name and referent. Exported as an internal API.
  """A named reference to a trackable object for use with the `Trackable` class.

  These references mark named `Trackable` dependencies of a `Trackable` object
  and should be created when overriding `Trackable._checkpoint_dependencies`.

  Attributes:
    name: The local name for this dependency.
    ref: The `Trackable` object being referenced.
  """

  __slots__ = ("_name", "_ref")

  def __init__(self, name, ref):
    # [FUNCTION DOC]
    # Constructor. Inputs: name, the local dependency name (string); ref, the referenced
    # Trackable object. Stores both in private slots. Returns nothing.
    self._name = name
    self._ref = ref

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the local dependency name given at construction.
    return self._name

  @property
  def ref(self):
    # [FUNCTION DOC]
    # Property returning the referenced Trackable object (strongly held in this class).
    return self._ref

  def __iter__(self):
    # [FUNCTION DOC]
    # Generator that yields the name and then the referenced object, so a reference can be
    # unpacked like a two-element tuple.
    yield self.name
    yield self.ref

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a readable string showing the concrete class name, the dependency name and the
    # referenced object.
    return f"{self.__class__.__name__}(name={self.name}, ref={self.ref})"

  def __eq__(self, o):
    # [FUNCTION DOC]
    # Equality check. Input o: any object. If o is a tuple, compares it with the (name, ref) pair;
    # if o is a TrackableReference, compares names and referents; any other type is unequal.
    # Returns a boolean.
    # KNOWN ISSUE: defining equality without a hash method makes instances unhashable, so they
    # cannot be placed in sets or used as mapping keys.
    if isinstance(o, tuple):
      return (self.name, self.ref) == o
    elif isinstance(o, TrackableReference):
      return self.name == o.name and self.ref == o.ref
    else:
      return False


class WeakTrackableReference(TrackableReference):
  # [CLASS DOC]
  # Variant of TrackableReference that keeps only a weak reference to the referenced object, so
  # the dependency does not keep it alive. The ref property dereferences it and yields nothing
  # once the object has been garbage collected. Uses no extra slots.
  """TrackableReference that stores weak references."""
  __slots__ = ()

  def __init__(self, name, ref):
    # [FUNCTION DOC]
    # Constructor. Inputs: name, the dependency name; ref, either a Trackable or an existing weak
    # reference to one. A plain object is converted into a weak reference before delegating to the
    # parent constructor. Returns nothing.
    if not isinstance(ref, weakref.ref):
      ref = weakref.ref(ref)
    super(WeakTrackableReference, self).__init__(name=name, ref=ref)

  @property
  def ref(self):
    # [FUNCTION DOC]
    # Property that dereferences the stored weak reference and returns the live object, or
    # nothing if it has already been collected.
    return self._ref()


# TODO(bfontain):  Update once sharded initialization interface is finalized.
ShardInfo = collections.namedtuple("CheckpointInitialValueShardInfo",
                                   ["shape", "offset"])


@tf_export("__internal__.tracking.CheckpointInitialValueCallable", v1=[])
class CheckpointInitialValueCallable(object):
  # [CLASS DOC]
  # Initializer-like callable that wraps a pending checkpoint restoration position. When called
  # like a normal variable initializer it produces a CheckpointInitialValue so the variable is
  # created directly with the checkpointed value (restore-on-create). Exposes the position and
  # the restore identifier used to order competing restorations.
  """A callable object that returns a CheckpointInitialValue.

  See CheckpointInitialValue for more information.
  """

  def __init__(self, checkpoint_position):
    # [FUNCTION DOC]
    # Constructor. Input: checkpoint_position, the object describing which checkpoint entry to
    # read the variable value from. Stores it. Returns nothing.
    self._checkpoint_position = checkpoint_position

  @property
  def checkpoint_position(self):
    # [FUNCTION DOC]
    # Property returning the checkpoint position given at construction.
    return self._checkpoint_position

  def __call__(self, shape=None, dtype=None, shard_info=None):
    # [FUNCTION DOC]
    # Makes the object usable as a variable initializer. Inputs: shape, the full variable shape
    # (optional, needed only for sharded values); dtype, accepted for compatibility but ignored;
    # shard_info, optional description of the shard (offset and shape) to read. Returns a new
    # CheckpointInitialValue reading the value from the stored checkpoint position. Side effect:
    # builds restore ops that read the checkpoint.
    # Note that the signature here is for compatibility with normal callable
    # initializers which take shape and dtype. Although dtype isn't used, it
    # will get passed in by a functool.partial_wrapper in places like
    # base_layer_utils.py's make_variable.
    return CheckpointInitialValue(
        self._checkpoint_position, shape, shard_info=shard_info)

  @property
  def restore_uid(self):
    # [FUNCTION DOC]
    # Property returning the restore identifier of the stored checkpoint position; larger values
    # mean the restore was requested later and should win.
    return self._checkpoint_position.restore_uid


@tf_export("__internal__.tracking.CheckpointInitialValue", v1=[])
class CheckpointInitialValue(object):
  # [CLASS DOC]
  # Wrapper around the tensor holding a variable's value read from a checkpoint, used as a
  # variable's initial value. Holds the wrapped tensor (wrapped_value) and the checkpoint position
  # it came from so the variable can learn the restore identifier and sequence deferred restores.
  # Convertible to a tensor through the tensor-conversion protocol.
  """Tensor wrapper for managing update UIDs in `Variables`.

  When supplied as an initial value, objects of this type let a `Variable`
  (`Variable`, `ResourceVariable`, etc.) know the UID of the restore the initial
  value came from. This allows deferred restorations to be sequenced in the
  order the user specified them, and lets us fall back on assignment if an
  initial value is not set (e.g. due to a custom getter interfering).

  See comments in _add_variable_with_custom_getter for more information about
  how `CheckpointInitialValue` is used.
  """

  def __init__(self, checkpoint_position, shape=None, shard_info=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: checkpoint_position, the position in the checkpoint holding the
    # variable value; shape, the full variable shape (list of integers), required when shard_info
    # is given; shard_info, optional shard description with offset and shape sequences. Builds a
    # shape-and-slice specification string (empty when not sharded) and asks the position for the
    # value tensor stored under the variable-value key, keeping it as wrapped_value. Returns
    # nothing. Side effect: creates checkpoint-reading ops. A missing shape with shard_info set
    # raises a type error from iterating over nothing.
    if shard_info:
      full_shape_str = " ".join("%d" % d for d in shape) + " "
      slice_spec = ":".join(
          "%d,%d" % (o, s) for o, s in zip(shard_info.offset, shard_info.shape))
      shape_and_slice = full_shape_str + slice_spec
    else:
      shape_and_slice = ""
    self.wrapped_value = checkpoint_position.value_tensors(
        {VARIABLE_VALUE_KEY: shape_and_slice})[VARIABLE_VALUE_KEY]
    self._checkpoint_position = checkpoint_position

  def __tf_tensor__(self, dtype=None, name=None):
    # [FUNCTION DOC]
    # Tensor-conversion protocol hook. Inputs dtype and name are accepted for compatibility and
    # deliberately discarded. Returns the wrapped tensor holding the checkpointed value.
    del dtype
    del name
    return self.wrapped_value

  @property
  def checkpoint_position(self):
    # [FUNCTION DOC]
    # Property returning the checkpoint position this initial value was read from.
    return self._checkpoint_position


class NoRestoreSaveable(saveable_object.SaveableObject):
  # [CLASS DOC]
  # Saveable object that writes a tensor to a checkpoint under a given name but does nothing on
  # restore (its restore step returns a no-op). Used to embed values that are saved but never
  # read back into the object. Subclass of the generic saveable object with a single save spec.
  """Embeds a tensor in a checkpoint with no restore ops."""

  def __init__(self, tensor, name, dtype=None, device=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: tensor, the value to save; name, the checkpoint key; dtype and device,
    # optional overrides for the save spec. Builds one save spec with an empty slice and passes it
    # to the parent constructor. Returns nothing.
    spec = saveable_object.SaveSpec(
        tensor, "", name, dtype=dtype, device=device)
    super(NoRestoreSaveable, self).__init__(tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # Restore hook. Inputs restored_tensors and restored_shapes are ignored. Returns a no-op
    # graph operation, so restoring never changes any state.
    return gen_control_flow_ops.no_op()


_SlotVariableRestoration = collections.namedtuple(
    "_SlotVariableRestoration",
    [
        # The checkpoint proto id of the optimizer object.
        "optimizer_id",
        # The checkpoint proto id of the slot variable.
        "slot_variable_id",
        "slot_name",
    ])


@tf_export("__internal__.tracking.no_automatic_dependency_tracking", v1=[])
def no_automatic_dependency_tracking(method):
  # [FUNCTION DOC]
  # Decorator factory. Input: method, a method of a Trackable object. Returns a decorated version
  # (built with the decorator utility so the original is preserved as the target) which, around
  # each call, turns off attribute-assignment dependency tracking on the instance, calls the
  # method with all arguments, and always restores the previous tracking value, even if the
  # method raises. Exceptions from the method propagate. Not thread safe.
  """Disables automatic dependency tracking on attribute assignment.

  Use to decorate any method of a Trackable object. Attribute assignment in
  that method will not add dependencies (also respected in Model). Harmless if
  used in a class which does not do automatic dependency tracking (which means
  it's safe to use in base classes which may have subclasses which also inherit
  from Trackable).

  Args:
    method: The method to decorate.

  Returns:
    A decorated method which sets and un-sets automatic dependency tracking for
    the object the method is called on (not thread safe).
  """

  def _method_wrapper(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: inner wrapper created by the no-automatic-dependency-tracking decorator.
    # Saves the instance's current tracking flag (default enabled), disables tracking, calls the
    # wrapped method with the given arguments, and restores the flag in a finally step. Returns
    # the wrapped method's result; exceptions propagate after restoring the flag. Side effect:
    # temporarily mutates the instance's tracking flag.
    previous_value = getattr(self, "_self_setattr_tracking", True)
    self._self_setattr_tracking = False  # pylint: disable=protected-access
    try:
      result = method(self, *args, **kwargs)
    finally:
      self._self_setattr_tracking = previous_value  # pylint: disable=protected-access
    return result

  return tf_decorator.make_decorator(
      target=method, decorator_func=_method_wrapper)


@tf_contextlib.contextmanager
def no_manual_dependency_tracking_scope(obj):
  # [FUNCTION DOC]
  # Context manager (generator based). Input: obj, a trackable object. On entry records the
  # object's current manual-tracking flag (default enabled) and sets it to disabled; yields
  # control to the body; on exit, even after an exception, restores the previous flag. Yields
  # nothing useful. Side effect: temporarily mutates the object's manual tracking flag.
  """A context that disables manual dependency tracking for the given `obj`.

  Sometimes library methods might track objects on their own and we might want
  to disable that and do the tracking on our own. One can then use this context
  manager to disable the tracking the library method does and do your own
  tracking.

  For example:

  class TestLayer(tf.keras.Layer):
    def build():
      with no_manual_dependency_tracking_scope(self):
        var = self.add_variable("name1")  # Creates a var and doesn't track it
      self._track_trackable("name2", var)  # We track variable with name `name2`

  Args:
    obj: A trackable object.

  Yields:
    a scope in which the object doesn't track dependencies manually.
  """
  # pylint: disable=protected-access
  previous_value = getattr(obj, "_manual_tracking", True)
  obj._manual_tracking = False
  try:
    yield
  finally:
    obj._manual_tracking = previous_value


@tf_contextlib.contextmanager
def no_automatic_dependency_tracking_scope(obj):
  # [FUNCTION DOC]
  # Context manager (generator based). Input: obj, a trackable object. On entry records the
  # object's attribute-assignment tracking flag (default enabled) and disables it; yields to the
  # body; on exit, even after an exception, restores the earlier value. Works like the decorator
  # version but for a block of code. Side effect: temporarily mutates the object's flag.
  """A context that disables automatic dependency tracking when assigning attrs.

  Objects that inherit from Autotrackable automatically creates dependencies
  to trackable objects through attribute assignments, and wraps data structures
  (lists or dicts) with trackable classes. This scope may be used to temporarily
  disable this behavior. This works similar to the decorator
  `no_automatic_dependency_tracking`.

  Example usage:
  ```
  model = tf.keras.Model()
  model.arr1 = []  # Creates a ListWrapper object
  with no_automatic_dependency_tracking_scope(model):
    model.arr2 = []  # Creates a regular, untracked python list
  ```

  Args:
    obj: A trackable object.

  Yields:
    a scope in which the object doesn't track dependencies.
  """
  previous_value = getattr(obj, "_setattr_tracking", True)
  obj._setattr_tracking = False  # pylint: disable=protected-access
  try:
    yield
  finally:
    obj._setattr_tracking = previous_value  # pylint: disable=protected-access


@tf_export("__internal__.tracking.Trackable", v1=[])
class Trackable(object):
  # [CLASS DOC]
  # Root base class of everything that can be saved to checkpoints and saved models. Keeps
  # dependency bookkeeping lazily in attributes prefixed with an internal marker (list and
  # name-to-object mapping of dependencies, deferred restorations, an update identifier, name-based
  # restores and legacy saveable factories), created on first use rather than in a constructor.
  # Provides tracking of named dependencies, restore-on-create variable creation, and the hooks
  # (children, tensors serialization, saved-model export) that subclasses override. Does not track
  # attributes automatically; see the auto-tracking subclass for that. Exported as an internal API.
  """Base class for `Trackable` objects without automatic dependencies.

  This class has no __setattr__ override for performance reasons. Dependencies
  must be added explicitly. Unless attribute assignment is performance-critical,
  use `AutoTrackable` instead. Use `Trackable` for `isinstance`
  checks.
  """

  # For compatibility with wrapt.ObjectProxy, attributes are all prefixed with
  # _self_. We have some properties to forward semi-public attributes to their
  # _self_ equivalents.

  @property
  def _setattr_tracking(self):
    # [FUNCTION DOC]
    # Getter for the attribute-assignment tracking flag. Lazily creates the underlying attribute
    # as enabled (true) if it does not exist yet, then returns it. Side effect: may set it.
    if not hasattr(self, "_self_setattr_tracking"):
      self._self_setattr_tracking = True
    return self._self_setattr_tracking

  @_setattr_tracking.setter
  def _setattr_tracking(self, value):
    # [FUNCTION DOC]
    # Setter. Input value: boolean enabling or disabling attribute-assignment tracking. Stores it
    # in the underlying attribute. Returns nothing.
    self._self_setattr_tracking = value

  @property
  def _update_uid(self):
    # [FUNCTION DOC]
    # Getter returning the identifier of the latest restore applied to this object.
    # Raises an attribute error if the trackable bookkeeping has not been initialized yet.
    return self._self_update_uid

  @_update_uid.setter
  def _update_uid(self, value):
    # [FUNCTION DOC]
    # Setter. Input value: integer restore identifier. Stores it. Returns nothing.
    self._self_update_uid = value

  @property
  def _unconditional_checkpoint_dependencies(self):
    # [FUNCTION DOC]
    # Returns the raw list of dependency references. Raises an attribute error if bookkeeping has
    # not been initialized yet.
    return self._self_unconditional_checkpoint_dependencies

  @property
  def _unconditional_dependency_names(self):
    # [FUNCTION DOC]
    # Returns the raw mapping from dependency names to trackables. Raises an attribute error if
    # bookkeeping has not been initialized yet.
    return self._self_unconditional_dependency_names

  @property
  def _name_based_restores(self):
    # [FUNCTION DOC]
    # Returns the set of name-based restore coordinators applied to this object. Raises an
    # attribute error if bookkeeping has not been initialized yet.
    return self._self_name_based_restores

  # Trackable does not do automatic dependency tracking, but uses the
  # no_automatic_dependency_tracking decorator so it can avoid adding
  # dependencies if a subclass is Trackable / inherits from Model (both of
  # which have __setattr__ overrides).
  @no_automatic_dependency_tracking
  def _maybe_initialize_trackable(self):
    # [FUNCTION DOC]
    # Lazily creates the dependency bookkeeping on this object; does nothing if already done
    # (detected by the presence of the dependency list). Creates the dependency list, the
    # name-to-object mapping, the deferred-dependency mapping, the update identifier (set to
    # minus one), the set of name-based restores and the mapping of legacy saveable factories.
    # Takes no inputs and returns nothing. Raises an assertion error if an update identifier
    # already exists before initialization. Wrapped so attribute assignments here are not tracked.
    """Initialize dependency management.

    Not __init__, since most objects will forget to call it.
    """
    if hasattr(self, "_self_unconditional_checkpoint_dependencies"):
      # __init__ already called. This check means that we don't need
      # Trackable.__init__() in the constructor of every TensorFlow object.
      return
    # A list of TrackableReference objects. Some classes implementing
    # `Trackable`, notably `Optimizer`s, may override the
    # _checkpoint_dependencies property with conditional dependencies
    # (e.g. based on the current graph when saving).
    self._self_unconditional_checkpoint_dependencies = []
    # Maps names -> Trackable objects
    self._self_unconditional_dependency_names = {}
    # Restorations for other Trackable objects on which this object may
    # eventually depend. Maps local name -> CheckpointPosition list. Optimizers
    # tack on conditional dependencies, and so need separate management of
    # deferred dependencies too.
    self._self_unconditional_deferred_dependencies = {}
    # The UID of the highest assignment to this object. Used to ensure that the
    # last requested assignment determines the final value of an object.
    if hasattr(self, "_self_update_uid"):
      raise AssertionError(
          "Internal error: the object had an update UID set before its "
          "initialization code was run.")
    self._self_update_uid = -1
    # When executing eagerly, holds a collection of _NameBasedRestoreCoordinator
    # instances, which should be checked when creating variables or other
    # saveables. These are passed on recursively to all dependencies, since
    # unlike object-based checkpoint restores we don't know which subgraph is
    # being restored in advance. This mechanism is only necessary for
    # restore-on-create when executing eagerly, and so is unused when graph
    # building.
    self._self_name_based_restores = set()

    # Dictionary of SaveableObjects factories. This dictionary is defined when
    # the object is loaded from the SavedModel. When writing a custom class,
    # prefer overriding "_gather_saveables_from_checkpoint" to using this
    # attribute.
    self._self_saveable_object_factories = {}

  @property
  def _object_identifier(self):
    # [FUNCTION DOC]
    # Property returning the constant string "_generic_user_object", the default identifier for
    # saved-model serialization. Deprecated in favor of the serializable registration name.
    """String used to identify this object in a SavedModel.

    THIS FIELD HAS BEEN DEPRECATED IN FAVOR OF THE NAME REGISTERED WITH
    `register_serializable`.

    Generally, the object identifier is constant across objects of the same
    class, while the metadata field is used for instance-specific data.

    Returns:
      String object identifier.
    """
    return "_generic_user_object"

  def _no_dependency(self, value):
    # [FUNCTION DOC]
    # Hook to mark a value as not to be tracked. Input: value, any object. In this base class
    # (no automatic tracking) it simply returns the value unchanged; auto-tracking subclasses
    # override it to wrap the value.
    """If automatic dependency tracking is enabled, ignores `value`."""
    return value

  def _name_based_attribute_restore(self, checkpoint):
    # [FUNCTION DOC]
    # Restores this object from a legacy name-based checkpoint. Input: checkpoint, the name-based
    # restore coordinator with a restore identifier and an eager-restore operation. Records the
    # coordinator in the set of name-based restores; if this object's update identifier is older
    # than the coordinator's, performs the eager restore for this object and updates the
    # identifier. Returns nothing. Requires the bookkeeping to be initialized.
    """Restore the object's attributes from a name-based checkpoint."""
    self._self_name_based_restores.add(checkpoint)
    if self._self_update_uid < checkpoint.restore_uid:
      checkpoint.eager_restore(self)
      self._self_update_uid = checkpoint.restore_uid

  @property
  def _checkpoint_dependencies(self):
    # [FUNCTION DOC]
    # Property returning the list of named dependency references to save with this object; the
    # base version returns the unconditional list. Subclasses may override to add conditional
    # dependencies.
    """All dependencies of this object.

    May be overridden to include conditional dependencies.

    Returns:
      A list of `TrackableReference` objects indicating named
      `Trackable` dependencies which should be saved along with this
      object.
    """
    return self._self_unconditional_checkpoint_dependencies

  @property
  def _deferred_dependencies(self):
    # [FUNCTION DOC]
    # Property returning the mapping from local dependency names to lists of pending checkpoint
    # positions for dependencies that did not exist yet when a restore was requested. The base
    # version returns the unconditional mapping. Raises an attribute error if the unconditional
    # deferred mapping was never created.
    """A dictionary with deferred dependencies.

    Stores restorations for other Trackable objects on which this object
    may eventually depend. May be overridden by sub-classes (e.g. Optimizers use
    conditional dependencies based the current graph, and so need separate
    management of deferred dependencies too).

    Returns:
      A dictionary mapping from local name to a list of CheckpointPosition
      objects.
    """
    return self._self_unconditional_deferred_dependencies

  def _lookup_dependency(self, name, cached_dependencies=None):
    # [FUNCTION DOC]
    # Finds a dependency by local name. Inputs: name (string); cached_dependencies, optional
    # precomputed mapping of children. If that mapping is non-empty the name is looked up there,
    # otherwise in the unconditional name mapping. Returns the Trackable or nothing if absent.
    # Raises an attribute error if bookkeeping is uninitialized and no cache is given.
    """Look up a dependency by name.

    May be overridden to include conditional dependencies.

    Args:
      name: The local name of the dependency.
      cached_dependencies: Optional dict containing all computed dependencies
        returned by `self._trackable_children()`.

    Returns:
      A `Trackable` object, or `None` if no dependency by this name was
      found.
    """
    if cached_dependencies:
      return cached_dependencies.get(name)
    return self._self_unconditional_dependency_names.get(name)

  def _add_variable_with_custom_getter(self,
                                       name,
                                       shape=None,
                                       dtype=dtypes.float32,
                                       initializer=None,
                                       getter=None,
                                       overwrite=False,
                                       **kwargs_for_getter):
    # [FUNCTION DOC]
    # Creates a variable through a getter and tracks it as a dependency, using a pending
    # checkpoint value as initializer when one exists (restore-on-create). Inputs: name, unique
    # dependency name; shape; dtype (default float32); initializer; getter, the callable that
    # actually builds the variable (required in practice); overwrite, if true skips unique-name
    # checks; extra keyword arguments are passed to the getter. Initializes bookkeeping, and,
    # when executing eagerly, looks for a simple preloaded restoration which replaces the
    # initializer unless the given initializer is a checkpoint-initial-value callable with a
    # later restore identifier. Calls the getter; then, unless overwrite is set with a non-
    # trackable result, tracks the variable. Returns the new variable. Raises a value error for
    # duplicate names (from tracking). Side effect: adds a dependency.
    """Restore-on-create for a variable be saved with this `Trackable`.

    If the user has requested that this object or another `Trackable` which
    depends on this object be restored from a checkpoint (deferred loading
    before variable object creation), `initializer` may be ignored and the value
    from the checkpoint used instead.

    Args:
      name: A name for the variable. Must be unique within this object.
      shape: The shape of the variable.
      dtype: The data type of the variable.
      initializer: The initializer to use. Ignored if there is a deferred
        restoration stored in the Trackable.
      getter: The getter to wrap which actually fetches the variable.
      overwrite: If True, disables unique name and type checks.
      **kwargs_for_getter: Passed to the getter.

    Returns:
      The new variable object.

    Raises:
      ValueError: If the variable name is not unique.
    """
    self._maybe_initialize_trackable()
    with ops.init_scope():
      if context.executing_eagerly():
        # If this is a variable with a single Tensor stored in the checkpoint,
        # we can set that value as an initializer rather than initializing and
        # then assigning (when executing eagerly). This call returns None if
        # there is nothing to restore.
        checkpoint_initializer = self._preload_simple_restoration(name=name)
      else:
        checkpoint_initializer = None
      if (checkpoint_initializer is not None and
          not (isinstance(initializer, CheckpointInitialValueCallable) and
               (initializer.restore_uid > checkpoint_initializer.restore_uid))):
        # If multiple Trackable objects are "creating" the same variable
        # via the magic of custom getters, the one with the highest restore UID
        # (the one called last) has to make the final initializer. If another
        # custom getter interrupts this process by overwriting the initializer,
        # then we'll catch that when we call _track_trackable. So this is
        # "best effort" to set the initializer with the highest restore UID.
        initializer = checkpoint_initializer
    new_variable = getter(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        **kwargs_for_getter)

    # If we set an initializer and the variable processed it, tracking will not
    # assign again. It will add this variable to our dependencies, and if there
    # is a non-trivial restoration queued, it will handle that. This also
    # handles slot variables.
    if not overwrite or isinstance(new_variable, Trackable):
      return self._track_trackable(new_variable, name=name, overwrite=overwrite)
    else:
      # TODO(allenl): Some variable types are not yet supported. Remove this
      # fallback once all get_variable() return types are Trackable.
      return new_variable

  def _preload_simple_restoration(self, name):
    # [FUNCTION DOC]
    # Looks for a queued restoration that can serve as a variable's initial value. Input: name,
    # the local dependency name. Returns nothing if no restore is queued for the name or if any
    # queued restore is not a simple single-variable one; otherwise picks the position with the
    # highest restore identifier and returns a checkpoint-initial-value callable for it. The
    # queued restoration is not removed.
    """Return a dependency's value for restore-on-create.

    Note the restoration is not deleted; if for some reason preload is called
    and then not assigned to the variable (for example because a custom getter
    overrides the initializer), the assignment will still happen once the
    variable is tracked (determined based on checkpoint.restore_uid).

    Args:
      name: The object-local name of the dependency holding the variable's
        value.

    Returns:
      An callable for use as a variable's initializer/initial_value, or None if
      one should not be set (either because there was no variable with this name
      in the checkpoint or because it needs more complex deserialization). Any
      non-trivial deserialization will happen when the variable object is
      tracked.
    """
    deferred_dependencies_list = self._deferred_dependencies.get(name, ())
    if not deferred_dependencies_list:
      # Nothing to do; we don't have a restore for this dependency queued up.
      return
    for checkpoint_position in deferred_dependencies_list:
      if not checkpoint_position.is_simple_variable():
        # If _any_ pending restoration is too complicated to fit in an
        # initializer (because it has dependencies, or because there are
        # multiple Tensors to restore), bail and let the general tracking code
        # handle it.
        return None
    checkpoint_position = max(
        deferred_dependencies_list,
        key=lambda restore: restore.checkpoint.restore_uid)
    return CheckpointInitialValueCallable(
        checkpoint_position=checkpoint_position)

  def _track_trackable(self, trackable, name, overwrite=False):
    # [FUNCTION DOC]
    # Declares a named dependency of this object on another Trackable so that checkpoints include
    # it. Inputs: trackable, the object depended on; name, the local dependency name (string);
    # overwrite, boolean allowing replacement of an existing different dependency of that name
    # (default false). Initializes bookkeeping. Raises a type error if trackable is not a
    # Trackable. If manual tracking is disabled on this object, returns trackable without
    # recording anything. Otherwise: if a different object is already registered under the name,
    # raises a value error unless overwrite is set, in which case the existing reference is
    # replaced in the dependency list; if nothing is registered, appends a new reference and
    # applies any deferred restores waiting for that name. Always stores the name mapping.
    # Returns the trackable. Side effect: mutates the dependency list and mapping.
    """Declare a dependency on another `Trackable` object.

    Indicates that checkpoints for this object should include variables from
    `trackable`.

    Variables in a checkpoint are mapped to `Trackable`s based on the names
    provided when the checkpoint was written. To avoid breaking existing
    checkpoints when modifying a class, neither variable names nor dependency
    names (the names passed to `_track_trackable`) may change.

    Args:
      trackable: A `Trackable` which this object depends on.
      name: A local name for `trackable`, used for loading checkpoints into the
        correct objects.
      overwrite: Boolean, whether silently replacing dependencies is OK. Used
        for __setattr__, where throwing an error on attribute reassignment would
        be inappropriate.

    Returns:
      `trackable`, for convenience when declaring a dependency and
      assigning to a member variable in one statement.

    Raises:
      TypeError: If `trackable` does not inherit from `Trackable`.
      ValueError: If another object is already tracked by this name.
    """
    self._maybe_initialize_trackable()
    if not isinstance(trackable, Trackable):
      raise TypeError(
          "Trackable._track_trackable() can only be used to track objects of "
          f"type Trackable. Got type {type(trackable)}.")
    if not getattr(self, "_manual_tracking", True):
      return trackable
    new_reference = TrackableReference(name=name, ref=trackable)
    current_object = self._lookup_dependency(name)
    if (current_object is not None and current_object is not trackable):
      if not overwrite:
        raise ValueError(
            f"Called Trackable._track_trackable() with name='{name}', "
            "but a Trackable with this name is already declared as a "
            "dependency. Names must be unique (or overwrite=True).")
      # This is a weird thing to do, but we're not going to stop people from
      # using __setattr__.
      for index, (old_name, _) in enumerate(
          self._self_unconditional_checkpoint_dependencies):
        if name == old_name:
          self._self_unconditional_checkpoint_dependencies[
              index] = new_reference
    elif current_object is None:
      self._self_unconditional_checkpoint_dependencies.append(new_reference)
      self._handle_deferred_dependencies(name=name, trackable=trackable)
    self._self_unconditional_dependency_names[name] = trackable
    return trackable

  def _handle_deferred_dependencies(self, name, trackable):
    # [FUNCTION DOC]
    # Applies queued restores for a dependency. Inputs: name, the dependency name; trackable,
    # the object now associated with that name. Initializes bookkeeping on both objects, removes
    # the deferred restore positions queued under the name and applies them to the trackable in
    # order of descending restore identifier, then passes all name-based restores of this object
    # on to the trackable, newest first. Returns nothing. Side effects: removes deferred entries
    # and triggers restore operations (value assignments) on the trackable.
    """Pop and load any deferred checkpoint restores into `trackable`.

    This method does not add a new dependency on `trackable`, but it does
    check if any outstanding/deferred dependencies have been queued waiting for
    this dependency to be added (matched based on `name`). If so,
    `trackable` and its dependencies are restored. The restorations are
    considered fulfilled and so are deleted.

    `_track_trackable` is more appropriate for adding a
    normal/unconditional dependency, and includes handling for deferred
    restorations. This method allows objects such as `Optimizer` to use the same
    restoration logic while managing conditional dependencies themselves, by
    overriding `_checkpoint_dependencies` and `_lookup_dependency` to change the
    object's dependencies based on the context it is saved/restored in (a single
    optimizer instance can have state associated with multiple graphs).

    Args:
      name: The name of the dependency within this object (`self`), used to
        match `trackable` with values saved in a checkpoint.
      trackable: The Trackable object to restore (inheriting from `Trackable`).
    """
    self._maybe_initialize_trackable()
    trackable._maybe_initialize_trackable()  # pylint: disable=protected-access
    deferred_dependencies_list = self._deferred_dependencies.pop(name, ())
    for checkpoint_position in sorted(
        deferred_dependencies_list,
        key=lambda restore: restore.checkpoint.restore_uid,
        reverse=True):
      checkpoint_position.restore(trackable)

    # Pass on any name-based restores queued in this object.
    for name_based_restore in sorted(
        self._self_name_based_restores,
        key=lambda checkpoint: checkpoint.restore_uid,
        reverse=True):
      trackable._name_based_attribute_restore(name_based_restore)  # pylint: disable=protected-access

  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # Deprecated legacy hook used by the old name-based saver. Takes no inputs. Returns the
    # mapping of checkpoint keys to saveable factories stored on this object (an empty mapping
    # if none were ever set). Subclasses override it to supply their own saveables.
    """Returns a dictionary of values to checkpoint with this object.

    NOTE: This method is deprecated, prefer implementing `_serialize_to_tensors`
    and `_restore_from_tensors` instead. This method is only used in the
    deprecated `tf.compat.v1.train.Saver`.

    Keys in the returned dictionary are local to this object and in a separate
    namespace from dependencies. Values may either be `SaveableObject` factories
    or variables easily converted to `SaveableObject`s (as in
    `tf.compat.v1.train.Saver`'s
    `var_list` constructor argument).

    `SaveableObjects` have a name set, which Trackable needs to generate
    itself. So rather than returning `SaveableObjects` directly, this method
    should return a dictionary of callables which take `name` arguments and
    return `SaveableObjects` with that name.

    If this object may also be passed to the global-name-based
    `tf.compat.v1.train.Saver`,
    the returned callables should have a default value for their name argument
    (i.e. be callable with no arguments).

    Returned values must be saved only by this object; if any value may be
    shared, it should instead be a dependency. For example, variable objects
    save their own values with the key `VARIABLE_VALUE_KEY`, but objects which
    reference variables simply add a dependency.

    **AsyncCheckpoint Support**
    If your Trackable implements `_gather_saveables_for_checkpoint`,
    `_copy_trackable_to_cpu` needs to be implemented as well to support
    asynchronous checkpoint.

    Returns:
      The dictionary mapping attribute names to `SaveableObject` factories
      described above. For example:
      {VARIABLE_VALUE_KEY:
       lambda name="global_name_for_this_object":
       SaveableObject(name=name, ...)}
    """
    return getattr(self, "_self_saveable_object_factories", {})

  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # Extension point: should return a mapping from names to the tensors that make up this
    # object's checkpointed state. The base implementation always raises a not-implemented error,
    # so only objects with custom resources or ops override it. Takes no inputs.
    """Gathers tensors to save to the checkpoint.

    You should only override `_serialize_to_tensors` and `_restore_from_tensors`
    if you are defining a custom resource or variable with custom ops.

    Otherwise, please store the state of your trackable in `tf.Variable` objects
    and add them to Trackable object hierarchy using `setattr` (for subclasses
    of `AutoTrackable`) or overriding the `_trackable_children` method.

    For an example of a valid implementation of these two methods, please see
    `DenseHashTable`.

    **Invalid implementation**

    ````
    class NamedTrackable(Trackable):
      def __init__(self, name: str):
        self.name = name
      def _serialize_to_tensors(self):
        return {"name": self.name}
      def _restore_from_tensors(self, restored_tensors):
        self.name = restored_tensors["name"]
    ```

    In this example, `NamedTrackable` can be saved and restored from
    checkpoints, but is incompatible with SavedModel, which tries to convert
    the serialize/restore functions into tf.functions. This fails because
    attribute assignment (`self.attr = new_value`) is not graph-friendly.

    **Suggested fix**

    ```
    class NamedTrackable(Trackable):
      def __init__(self, name: str):
        self.name = tf.Variable(name)

      def _trackable_children(self):
        return {"name": self.name}
    ```

    If the `name` attribute should be saved to the checkpoint, then convert it
    a `tf.Variable`.

    **TF1 Saver Compatibility**
    If your Trackable needs to be comatible with `tf.compat.v1.train.Saver`,
    implement `_gather_saveables_from_checkpoint`.

    **AsyncCheckpoint Support**
    If your Trackable implements `_serialize_to_tensors`,
    `_copy_trackable_to_cpu` needs to be implemented as well to support
    asynchronous checkpoint.

    Returns:
      A dictionary mapping names to tensors.
    """
    raise NotImplementedError

  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # Extension point paired with the serialize-to-tensors hook. Input: restored_tensors, a
    # mapping from the saved names to tensors. Subclasses should assign them and return the
    # restoring operation; the base implementation always raises a not-implemented error.
    """Restores checkpointed values to this `Trackable`.

    Please see the documentation for `Trackable._serialize_to_tensors`.

    Args:
      restored_tensors: A dictionary mapping names to tensors. The keys to this
        dictionary matches the names passed to _serialize_to_tensors.

    Returns:
      An op that runs the restoration.
    """
    raise NotImplementedError

  def _serialize_to_proto(self, object_proto=None, **kwargs):
    # [FUNCTION DOC]
    # Saved-model hook for registered serializable classes. Inputs: object_proto, the saved
    # object message that core types may fill (ignored here); extra keyword arguments (ignored).
    # The base implementation discards both and returns nothing, meaning no extra metadata is saved.
    """Returns a proto of any type to be saved into the SavedModel.

    Trackable classes decorated with `register_serializable` should overwrite
    this method to save metadata for this object to the SavedModel. The proto
    returned by this function will be passed to `_deserialize_from_proto` in the
    form of a `google.protobuf.Any` proto.

    This data is only saved and used by the Python API. Existing C++ loading
    APIs such as `tensorflow::LoadSavedModel` will not read this field at all.

    Args:
      object_proto: A `SavedObject` proto that may be filled by this function.
        Only the core serializable types (Variable, Function, Constant, Asset)
        should modify this argument.
      **kwargs: Future keyword arguments passed to the object during saving.

    Returns:
      A proto that serializes this class's type.
    """
    del object_proto, kwargs  # Unused.

    return None

  @classmethod
  def _deserialize_from_proto(cls,
                              proto=None,
                              dependencies=None,
                              object_proto=None,
                              export_dir=None,
                              asset_file_def=None,
                              operation_attributes=None,
                              **kwargs):
    # [FUNCTION DOC]
    # Class method hook for loading a registered serializable class from a saved model. Inputs
    # (all optional and ignored by this default): proto, the stored metadata message; dependencies,
    # mapping of names to already loaded dependencies; object_proto; export_dir; asset file
    # definitions; operation attributes; extra keyword arguments. Returns a new instance created
    # by calling the class constructor with no arguments; the constructor's own errors propagate.
    """Returns a new object restored by the SavedModel.

    Trackable classes decorated with `register_serializable` should overwrite
    this method to change how the object is loaded from SavedModel. By default,
    the object is initialized with no arguments.

    Example:

    ```
    def _serialize_to_proto(self, **unused_kwargs):
      return Message(name="a")

    @classmethod
    def _deserialize_from_proto(cls, proto, **unused_kwargs):
      if proto.Is(Message.DESCRIPTOR):
        unpacked = Message()
        proto.Unpack(unpacked)
        return cls(unpacked.name)
      else:
        return cls()
    ```

    This function is only used by the Python API. C++ and TensorFlow Serving do
    not have access to your registered class and cannot execute any of the
    non-tf.functions attached to the Python class. However, all signatures and
    tf.functions are still accessible.

    **Avoid creating duplicate trackables**

    SavedModel is saved by recursively gathering all of the trackables and their
    children. SavedModel loading reverses those steps by creating all
    trackables, then reconnecting the children trackables to their parents using
    `Trackable._add_trackable_child`.

    That means that if `_deserialize_from_proto` calls the `__init__` function,
    which creates all of the children trackables, then those children end up
    being created *twice*.

    To avoid this, structure your code so that Trackables are not created
    when deserialized from SavedModel:

    ```
    @register_serializable()
    class Serializable(trackable):
      def __init __(self, from_proto=False):
        create_non_trackable_objects()
        if not from_proto:
          create_variables_and_other_trackables()

      def _deserialize_from_proto(cls, **kwargs):
        return cls(from_proto=True)

      def _add_trackable_child(self, name, value):
        self.__setattr__(name, value)
    ```

    Args:
      proto: A `google.protobuf.Any` proto read from the `SavedModel`.
      dependencies: A dictionary mapping names to dependencies (see
        `_deserialization_dependencies`)
      object_proto: The `SavedObject` proto for this object.
      export_dir: The `SavedModel` directory
      asset_file_def: The `MetaGraphDef`'s `asset_file_def` field.
      operation_attributes: Dictionary mapping nodes to attribute from the
        imported `GraphDef`.
      **kwargs: Future keyword arguments passed to the object when loading.

    Returns:
      A new object.
    """
    del (proto, dependencies, object_proto, export_dir, asset_file_def,
         operation_attributes, kwargs)

    return cls()

  def _add_trackable_child(self, name, value):
    # [FUNCTION DOC]
    # Used when loading a saved model to reconnect a child to its parent. Inputs: name, the
    # connection name; value, the child Trackable. Tracks the child under that name, replacing
    # any existing dependency of the same name. Returns nothing. Raises a type error if value is
    # not a Trackable. Side effect: updates the dependency bookkeeping.
    """Restores a connection between trackables when loading from SavedModel.

    SavedModel stores both the object metadata and its list of children. When
    loading, this function is used along with `_deserialize_from_proto` to load
    objects from the SavedModel: First, all saved objects are created with
    `_deserialize_from_proto`. After that is complete, the children are
    connected using `_add_trackable_child`.

    **Example**

    `tf.Module`, `tf.keras.Model` and Keras layers use `__setattr__` to track
    children. This is why users can call `model.v = tf.Variable(...)`, and the
    variable will be automatically saved to the checkpoint. The implementation
    of this method for the listed objects is:

    ```
    def _add_trackable_child(self, name, value):
      self.__setattr__(name, value)
    ```

    Args:
      name: The name of the connection between the parent and child `Trackable`.
      value: The child `Trackable` object.
    """
    self._track_trackable(value, name, overwrite=True)

  def _deserialization_dependencies(self, children):
    # [FUNCTION DOC]
    # Declares objects that must be created before this one when loading a saved model. Input:
    # children, the mapping produced by the trackable-children hook (ignored here). The base
    # implementation returns an empty mapping, i.e. no extra ordering constraints.
    """Returns a dictionary containing `Trackables` that this object depends on.

    Dependencies define the order to serialize and deserialize objects in the
    SavedModel. For example:

    class A(Trackable):
      b = B()
      def _deserialization_dependencies(self, children):
        return {'b': self.b}

    class B(Trackable):
      pass

    We say that object `a=A()` depends on `a.b`.

    Dependencies are guaranteed to be serialized and deserialized before the
    object depending on them. The following methods use dependencies:
      - `_deserialize_from_proto` [loading]

    SavedModel loads with the bottom-up approach, by first creating all objects
    in the order defined by the dependencies, then connecting the children.

    Unlike `_trackable_children`, this function does not define the
    `SavedObjectGraph`. It only changes the order in which things are
    saved/loaded. Therefore, if there are dependencies that are not in the
    `SavedObjectGraph`, saving will fail.

    Args:
      children: Dict returned from `_trackable_children`.

    Returns:
      A dictionary mapping names to `Trackable`.
    """
    del children  # Unused.
    return {}

  def _trackable_children(self,
                          save_type=SaveType.CHECKPOINT,
                          cache=None,
                          **kwargs):
    # [FUNCTION DOC]
    # Returns the children used to build the object graph for checkpoints and saved models.
    # Inputs: save_type, which target is being saved (default checkpoint); cache, optional
    # mapping shared across a saved-model export (ignored here); extra keyword arguments
    # (ignored). Initializes bookkeeping, then returns a new mapping from dependency names to
    # the tracked objects, built from the checkpoint dependencies. Side effect: lazily creates
    # bookkeeping attributes. Subclasses override this to define their own children.
    """Returns this object's `Trackable` attributes.

    This method is used to build the object graph (or the object hierarchy,
    in pickling terms) for checkpoint save/restore, and `SavedModel` export.

    Override this method to define the children of this instance. Please read
    the implementation restrictions:

    **Rule 1: All children must be convertable to `Trackable`.**

    Must pass `isinstance` check or `converter.convert_to_trackable`.

    **Rule 2: [Checkpoint-only] Do not create new objects.**

    When saving to a `SavedModel`, this method is called *exactly once* for each
    `Trackable` in the object graph. When saving or restoring from a checkpoint,
    this method may be called *multiple times*. Thus, this method may create
    new Trackables when `save_type == SaveType.SAVEDMODEL` but not when
    `save_type == SaveType.CHECKPOINT`.

    When saving to `SavedModel`, new `Trackable` children can be created to save
    non-Trackable attributes to the `SavedModel`. In the example below, `hyper`
    is a regular python float hyperparameter. To save this value, a new Variable
    is created to store the value of `hyper`:

    ```
    def __init__(self):
      self.hyper = 1e-5

    def _trackable_children(self, save_type, **unused_kwargs):
      # Correct implementation
      children = {}
      if format == 'saved_model':
        children['hyper'] = tf.Variable(self.hyper)
      return children
    ```

    An incorrect implementation of `_trackable_children` is shown below. This
    function would cause failures when loading the checkpoint, and calling
    `load_status.assert_consumed()` or
    `load_status.assert_existing_objects_matched`. If you want a value to be
    saved in the checkpoint, hyper must be defined as a `tf.Variable` from the
    start.

    ```
    def _trackable_children(self, save_type, **unused_kwargs):
      # Incorrect implementation
      return {'hyper': tf.Variable(self.hyper)}
    ```

    **Rule 3: [`SavedModel`-only] Watch out for un-traced tf.functions.**

    At the begining of `_trackable_children`, always call
    `get_concrete_function()` for any `tf.function` that has an input signature.

    When `tf.functions` are saved to `SavedModel`, any `tf.functions` that have
    an input signature and has never been called is traced at export time in
    order to copy the op graph into the `SavedModel`. `tf.functions` that are
    traced for the first time are allowed to create new state:


    ```
    @tf.function(input_signature=[]):
    def fn(self);
      if self.v is None:
        self.v = tf.Variable(1.)
      return self.v
    ```

    A problem occurs when there is a `Trackable` that returns `fn` as one of its
    children and `self.v` has not been created yet. When `fn` is traced,
    `self.v` is added to the `Trackable`, but `SavedModel` does not see this
    modification since the `Trackable`'s children have already been gathered.

    Therefore, as a precaution, call `get_concrete_function()` at the very
    start of `_trackable_children` to ensure that the function is traced:


    ```
    def _trackable_children(self):
      self.fn.get_concrete_function()
      return {"v": self.v, "fn": self.fn}
    ```

    Args:
      save_type: A string, can be 'savedmodel' or 'checkpoint'. Defaults to
        SaveType.CHECKPOINT.
      cache: May be `None`, or a dictionary. When `save_type == savedmodel`, a
        new cache is created at the start of the SavedModel export, and shared
        between all `Trackables` in the same object graph. This cache may be
        used for advanced saving functionality.
      **kwargs: Additional kwargs that may be added at a later time.

    Returns:
      Dictionary mapping names to child trackables.
    """
    del save_type, cache, kwargs  # Unused.

    self._maybe_initialize_trackable()
    return {name: ref for name, ref in self._checkpoint_dependencies}

  def _export_to_saved_model_graph(self,
                                   object_map,
                                   tensor_map,
                                   options,
                                   **kwargs):
    # [FUNCTION DOC]
    # Saved-model export hook. Inputs: object_map, mapping from original to copied trackables;
    # tensor_map, mapping from original to copied tensors; options, the save options; extra
    # keyword arguments. The base implementation changes nothing and returns an empty list of
    # copied original tensors; subclasses holding tensors override it and update the maps.
    """Creates a copy of this object's tensors onto SavedModel graph.

    Needs to be overridden if the class contains tensors that must be saved
    into the graph. This method should update the `object_map` and `tensor_map`
    dictionaries.

    This method is called on all nodes in the Trackable Graph (generated by
    `_trackable_children`). The nodes are traversed in the order defined by
    `_deserialization_dependencies`

    All usages of _map_resources should be migrated to this method.

    Args:
      object_map: A dictionary that maps original Trackables to the copied
        Trackables. This only needs to be updated if the object is a
        tf.function, or if the copied tensors are necessary for checkpointing
        this object.
      tensor_map: Dictionary mapping original tensors to copied tensors.
      options: A `tf.saved_model.SaveOptions` object.
      **kwargs: Additional kwargs that may be added at a later time.

    Returns:
      Flat list of original tensors that have been copied.
    """
    _, _, _ = object_map, tensor_map, options
    del kwargs
    return []

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # Async-checkpoint hook. Input: object_map, mapping from original trackables to their CPU
    # copies, to be updated in place by overriding classes. The base implementation ignores the
    # input and always raises a not-implemented error, so objects that must support
    # asynchronous checkpointing have to override it.
    """Creates a copy of this object onto CPU, also copies values over.

    Needs to be overridden if the `Trackable` requires AsyncCheckpoint support.
    The method first checks whether a copy of `self` is already created in
    `object_map`, and creates one if not already created. Then the method copies
    the **values** of itself over to its copy mapped by `object_map`.

    Args:
      object_map: A dictionary that maps original Trackables to the copied
        Trackables, which reside in the CPU.
    """
    del object_map  # Unused
    raise NotImplementedError("Need to implement _copy_trackable_to_cpu() if "
                              "the Trackable requires AsyncCheckpoint support.")

  def _checkpoint_adapter(self, path: str):
    # [FUNCTION DOC]
    # Hook returning a checkpoint adapter that customizes how this object is restored. Input:
    # path, the checkpoint path (string, ignored by default). The base implementation returns
    # nothing, meaning no adapter is used.
    """Returns a checkpoint adapter for this object.

    Needs to be overridden if the `Trackable` requires adapter at restore.
    Override this method to define callbacks for checkpoint positions to be
    applied at restore time.

    Args:
      path: Checkpoint path.
    Returns:
      A subclass of AbstractCheckpointAdapter that defines callbacks at restore
      for this trackable.
    """
    del path
    return None
