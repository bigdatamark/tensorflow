# [MODULE DOC]
# Defines AutoTrackable, a Trackable base class that automatically records dependency edges when
# trackable values are assigned as attributes (wrapping lists, dicts and tuples in trackable data
# structures), and also reports tf.function attributes as children for SavedModel export. Exposed
# publicly through the internal tracking namespace. Has no import-time side effects other than
# the export registration.
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
"""Dependency tracking for trackable objects."""

import warnings

from absl import logging

from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.types import core as core_types
from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.tracking.AutoTrackable", v1=[])
class AutoTrackable(base.Trackable):
  # [CLASS DOC]
  # Trackable subclass with implicit dependency tracking. Assigning an attribute passes the value
  # through the sticky attribute assignment helper, which creates a named dependency when the
  # value is trackable (or wraps containers); deleting an attribute removes the dependency. For
  # SavedModel export its children also include function attributes (tf.function objects and
  # concrete functions), which are traced to force side effects. State lives in the Trackable
  # base class dependency lists and name maps. Intended to be subclassed or instantiated directly.
  """Manages dependencies on other objects.

  `Trackable` objects may have dependencies: other `Trackable` objects
  which should be saved if the object declaring the dependency is saved. A
  correctly saveable program has a dependency graph such that if changing a
  global variable affects an object (e.g. changes the behavior of any of its
  methods) then there is a chain of dependencies from the influenced object to
  the variable.

  Dependency edges have names, and are created implicitly when a
  `Trackable` object is assigned to an attribute of another
  `Trackable` object. For example:

  ```
  obj = Trackable()
  obj.v = ResourceVariable(0.)
  ```

  The `Trackable` object `obj` now has a dependency named "v" on a
  variable.

  `Trackable` objects may specify `Tensor`s to be saved and restored
  directly (e.g. a `Variable` indicating how to save itself) rather than through
  dependencies on other objects. See
  `Trackable._gather_saveables_for_checkpoint` for details.
  """

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Attribute assignment hook. Inputs: name and value. If the attribute already exists and is
    # the identical object, returns immediately (no re-tracking). Otherwise, unless the instance
    # flag for setattr tracking has been switched off (it defaults to on), the value is passed
    # through the sticky attribute assignment helper, which may register a dependency on it and
    # may return a wrapped value (for example a trackable list or mapping). Finally stores the
    # (possibly wrapped) value with the standard attribute mechanism. Returns nothing. Errors from
    # the helper, such as a name already declared as a dependency, propagate.
    """Support self.foo = trackable syntax."""
    try:
      if getattr(self, name) is value:
        # Short circuit for `self.$x = self.$x`.
        return
    except AttributeError:
      pass

    if getattr(self, "_self_setattr_tracking", True):
      value = data_structures.sticky_attribute_assignment(
          trackable=self, value=value, name=name)
    super(AutoTrackable, self).__setattr__(name, value)

  def __delattr__(self, name):
    # [FUNCTION DOC]
    # Attribute deletion hook. Input: name. First removes any tracked dependency with that name,
    # then deletes the attribute normally (raising an attribute error if it does not exist,
    # after the dependency removal has already been attempted). Returns nothing.
    self._delete_tracking(name)
    super(AutoTrackable, self).__delattr__(name)

  def _no_dependency(self, value):
    # [FUNCTION DOC]
    # HELPER: lets callers mark a value so assigning it does not create a dependency. Input:
    # value, any object. Returns it wrapped in the no-dependency wrapper from the data structures
    # module. No side effects.
    """Override to allow TrackableBase to disable dependency tracking."""
    return data_structures.NoDependency(value)

  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    # [FUNCTION DOC]
    # Returns a mapping from child names to child objects. Inputs: save_type (checkpoint by
    # default, or SavedModel) and extra keyword arguments forwarded to the base implementation.
    # For anything but SavedModel it simply delegates to the base class. For SavedModel it scans
    # every attribute name of the object (temporarily raising the logging threshold to fatal and
    # ignoring warnings, and treating any attribute whose access raises as nothing), collects
    # tf.function and concrete function attributes, and traces all tf.function objects to
    # enumerate their concrete functions (forcing variable creation and other side effects).
    # It then builds the result from the tracked checkpoint dependencies, skipping function-typed
    # dependencies, and merges in the collected functions. Raises a value error if a tracked
    # dependency and a function attribute share a name but are different objects. Restores the
    # logging verbosity even on errors.
    """Returns all children of a trackable, including functions."""
    if save_type != base.SaveType.SAVEDMODEL:
      return super(AutoTrackable, self)._trackable_children(
          save_type, **kwargs)

    functions = {}
    try:
      # We get the attributes, suppressing warnings and exceptions.
      logging_verbosity = logging.get_verbosity()
      logging.set_verbosity(logging.FATAL)
      for attribute_name in dir(self):
        try:
          with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attribute_value = getattr(self, attribute_name, None)
        except Exception:  # pylint: disable=broad-except
          # NOTE: If we make the exception catching here less broad, we might
          # need to revisit `finally` block below.
          # We really don't want to throw an exception just because some
          # object's attribute accessor is broken.
          attribute_value = None
        if isinstance(attribute_value, (def_function.Function,
                                        defun.ConcreteFunction)):
          functions[attribute_name] = attribute_value
    finally:
      logging.set_verbosity(logging_verbosity)

    # Trace concrete functions to force side-effects:
    #   1. populate the cache for functions that have an input_signature
    #      and have not been called
    #   2. force side effects of creation of concrete functions, e.g. create
    #      variables on first run.
    for fn in functions.values():
      if isinstance(fn, def_function.Function):
        fn._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access

    # Additional dependencies may have been generated during function tracing
    # (e.g. captured variables). Make sure we return those too.
    children = {}
    for name, child in self._checkpoint_dependencies:
      if isinstance(child, (core_types.PolymorphicFunction,
                            core_types.ConcreteFunction)):
        # Skip "tracked" functions for now since there may be objects that
        # automatically track functions that should not be saved.
        # TODO(kathywu): remove once `_list_functions_for_serialization` has
        # been fully deprecated.
        continue

      if name in functions and child is not functions[name]:
        raise ValueError(
            "Can't save object because it has multiple children with the same "
            f"name. Object: {self}, attribute name: {name}, child 1: "
            f"{child}, child 2: {functions[name]}")

      children[name] = child

    children.update(functions)
    return children

  def _delete_tracking(self, name):
    # [FUNCTION DOC]
    # HELPER: used by attribute deletion to drop a dependency. Input: name of the dependency.
    # Makes sure the trackable bookkeeping is initialized, and if the name is a registered
    # unconditional dependency, removes it from the name map and removes the first matching entry
    # from the ordered unconditional checkpoint dependency list. Does nothing for unknown names.
    # Returns nothing.
    """Removes the tracking of name."""
    self._maybe_initialize_trackable()
    if name in self._unconditional_dependency_names:
      del self._unconditional_dependency_names[name]
      for index, (dep_name, _) in enumerate(
          self._unconditional_checkpoint_dependencies):
        if dep_name == name:
          del self._unconditional_checkpoint_dependencies[index]
          break

  def _add_trackable_child(self, name, value):
    # [FUNCTION DOC]
    # HELPER: hook used by checkpoint and SavedModel loading to attach a restored child. Inputs:
    # name and value. Assigns the value as an attribute through the tracking attribute assignment,
    # so a dependency is created. Returns nothing.
    self.__setattr__(name, value)
