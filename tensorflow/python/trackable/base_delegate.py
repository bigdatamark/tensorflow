# [MODULE DOC]
# Defines the delegating Trackable mixin, used by wrapper classes (notably the Keras loss scale
# optimizer) that must look like another trackable object to checkpoint and saved-model code.
# Import-time side effect: the mixin class is registered as a public internal API symbol via the
# export decorator. Depends only on the API-export utility.

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
"""A mixin class that delegates another Trackable to be used when saving.

This is intended to be used with wrapper classes that cannot directly proxy the
wrapped object (e.g. with wrapt.ObjectProxy), because there are inner attributes
that cannot be exposed.

The Wrapper class itself cannot contain any Trackable children, as only the
delegated Trackable will be saved to checkpoint and SavedModel.

This class will "disappear" and be replaced with the wrapped inner Trackable
after a cycle of SavedModel saving and loading, unless the object is registered
and loaded with Keras.
"""

from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.tracking.DelegatingTrackableMixin", v1=[])
class DelegatingTrackableMixin(object):
  # [CLASS DOC]
  # Mixin for wrapper classes that also derive from Trackable. Holds one attribute, _trackable,
  # the delegate given at construction. Every internal Trackable property and method is
  # overridden to forward to the delegate, so checkpoints and saved models see only the delegate's
  # structure. Exported publicly as an internal tracking API; intended solely for the Keras loss
  # scale optimizer. Must come before Trackable in the base-class order so its overrides win.
  """A mixin that delegates all Trackable methods to another trackable object.

  DO NOT USE THIS UNLESS YOU ARE THE KERAS LOSS SCALE OPTIMIZER.

  This class must be used with multiple inheritance. A class that subclasses
  Trackable can also subclass this class, which causes all Trackable methods to
  be delegated to the trackable object passed in the constructor.

  A subclass can use this mixin to appear as if it were the trackable passed to
  the constructor, from a Checkpoint's perspective. LossScaleOptimizer uses this
  mixin, so that the checkpoint format for a LossScaleOptimizer is identical to
  the checkpoint format for a normal optimizer. This allows a model to be saved
  with a normal Optimizer and restored with a LossScaleOptimizer, or vice versa.
  The only difference in checkpoint format is that the loss scale is also saved
  with a LossScaleOptimizer.
  """

  def __init__(self, trackable_obj):
    # [FUNCTION DOC]
    # Constructor: stores the trackable object to which every Trackable operation is delegated
    # in the attribute named _trackable. Input: trackable_obj, any Trackable instance (required).
    # Returns nothing. Note it does not call the Trackable constructor of the other base class.
    self._trackable = trackable_obj

  # pylint: disable=protected-access
  @property
  def _setattr_tracking(self):
    # [FUNCTION DOC]
    # Getter: returns the attribute-assignment tracking flag of the delegate trackable.
    return self._trackable._setattr_tracking

  @_setattr_tracking.setter
  def _setattr_tracking(self, value):
    # [FUNCTION DOC]
    # Setter: input value is a boolean; stores it as the attribute-assignment tracking flag of
    # the delegate trackable (side effect on the delegate, not on this wrapper). Returns nothing.
    self._trackable._setattr_tracking = value

  @property
  def _update_uid(self):
    # [FUNCTION DOC]
    # Getter: returns the checkpoint update identifier recorded on the delegate trackable.
    return self._trackable._update_uid

  @_update_uid.setter
  def _update_uid(self, value):
    # [FUNCTION DOC]
    # Setter: input value is the new update identifier (integer); stores it on the delegate
    # trackable. Returns nothing.
    self._trackable._update_uid = value

  @property
  def _unconditional_checkpoint_dependencies(self):
    # [FUNCTION DOC]
    # Returns the delegate's list of checkpoint dependencies, ignoring any save-type filtering.
    return self._trackable._unconditional_checkpoint_dependencies

  @property
  def _unconditional_dependency_names(self):
    # [FUNCTION DOC]
    # Returns the delegate's mapping from dependency names to dependent trackables.
    return self._trackable._unconditional_dependency_names

  @property
  def _name_based_restores(self):
    # [FUNCTION DOC]
    # Returns the delegate's collection of name-based checkpoint restores recorded so far.
    return self._trackable._name_based_restores

  def _maybe_initialize_trackable(self):
    # [FUNCTION DOC]
    # Asks the delegate to lazily create its tracking bookkeeping state; returns whatever the
    # delegate returns (normally nothing). Side effect: initializes state on the delegate.
    return self._trackable._maybe_initialize_trackable()

  @property
  def _object_identifier(self):
    # [FUNCTION DOC]
    # Returns the delegate's string identifier used when serializing the object to a saved model.
    return self._trackable._object_identifier

  @property
  def _tracking_metadata(self):
    # [FUNCTION DOC]
    # Returns the delegate's metadata string stored alongside the object in saved models.
    return self._trackable._tracking_metadata

  def _no_dependency(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all positional and keyword arguments to the delegate's no-dependency helper,
    # which marks a value so that assigning it as an attribute is not tracked. Returns the result.
    return self._trackable._no_dependency(*args, **kwargs)

  def _name_based_attribute_restore(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments to the delegate's name-based (legacy) restore handler, which
    # restores values by attribute name from a checkpoint. Returns the delegate's result.
    return self._trackable._name_based_attribute_restore(*args, **kwargs)

  @property
  def _checkpoint_dependencies(self):
    # [FUNCTION DOC]
    # Returns the delegate's list of checkpoint dependencies (name, reference pairs).
    return self._trackable._checkpoint_dependencies

  @property
  def _deferred_dependencies(self):
    # [FUNCTION DOC]
    # Returns the delegate's mapping of dependencies that were requested by a restore before the
    # corresponding objects existed.
    return self._trackable._deferred_dependencies

  def _lookup_dependency(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments (a dependency name and optional extras) to the delegate and returns
    # the dependency it finds, or nothing if there is none.
    return self._trackable._lookup_dependency(*args, **kwargs)

  def _add_variable_with_custom_getter(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments (name, shape, dtype, initializer, getter, overwrite and so on) to
    # the delegate, which creates a variable via the getter and tracks it on the delegate.
    # Returns the created variable. Side effect: adds a dependency to the delegate.
    return self._trackable._add_variable_with_custom_getter(*args, **kwargs)

  def _preload_simple_restoration(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments to the delegate's hook that applies a pending simple restoration
    # (a value awaiting a variable of a given name) before the variable is used. Returns its result.
    return self._trackable._preload_simple_restoration(*args, **kwargs)

  def _track_trackable(self, *args, **kwargs):  # pylint: disable=redefined-outer-name
    # [FUNCTION DOC]
    # Forwards all arguments (trackable, name, optional overwrite) to the delegate, so the
    # dependency is recorded on the delegate rather than on this wrapper. Returns the tracked
    # object. Raises whatever the delegate raises (for example for duplicate names).
    return self._trackable._track_trackable(*args, **kwargs)

  def _handle_deferred_dependencies(self, name, trackable):  # pylint: disable=redefined-outer-name
    # [FUNCTION DOC]
    # Inputs: name, the dependency name, and trackable, the object just created for it. Forwards
    # them to the delegate so pending checkpoint restores for that name are applied. Returns the
    # delegate's result (normally nothing).
    return self._trackable._handle_deferred_dependencies(name, trackable)

  def _gather_saveables_for_checkpoint(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments to the delegate's legacy saveable gathering routine and returns the
    # mapping of checkpoint keys to saveable factories it produces.
    return self._trackable._gather_saveables_for_checkpoint(*args, **kwargs)

  def _trackable_children(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments (such as the save type) to the delegate and returns its mapping from
    # child names to child trackables. This is what makes only the delegate's children appear
    # in a checkpoint.
    return self._trackable._trackable_children(*args, **kwargs)

  def _deserialization_dependencies(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments to the delegate and returns the mapping of dependencies that must
    # be loaded before this object when restoring from a saved model.
    return self._trackable._deserialization_dependencies(*args, **kwargs)

  def _export_to_saved_model_graph(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments (object map, tensor map, options) to the delegate, which registers
    # its exportable pieces in the saved-model graph. Returns the delegate's result. Because the
    # delegate is exported, the wrapper itself disappears from the saved model.
    return self._trackable._export_to_saved_model_graph(*args, **kwargs)

  def _serialize_to_tensors(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments to the delegate and returns its mapping from checkpoint keys to the
    # tensors that represent the object's state.
    return self._trackable._serialize_to_tensors(*args, **kwargs)

  def _restore_from_tensors(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments (a mapping from checkpoint keys to restored tensors) to the delegate,
    # which assigns them to its state. Returns the delegate's result, typically restore operations.
    return self._trackable._restore_from_tensors(*args, **kwargs)

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the copy is a bare mixin instance, not an instance of the wrapper's actual
    # (sub)class, so subclass attributes are not reproduced in the CPU copy.
    # Hook for async checkpointing. Input: object_map, a mapping from original objects to their
    # CPU copies, updated in place. First asks the delegate to add its CPU copy. Then, if this
    # wrapper has no entry yet, adds one holding a new plain mixin instance that wraps the
    # delegate's copy. Returns nothing.
    self._trackable._copy_trackable_to_cpu(object_map)
    if self not in object_map:
      object_map[self] = DelegatingTrackableMixin(object_map[self._trackable])
  # pylint: enable=protected-access

