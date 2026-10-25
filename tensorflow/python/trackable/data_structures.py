# [MODULE DOC]
# Trackable data structures: wrappers and containers (lists, mappings, dicts, tuples) that let
# plain Python containers holding trackable objects take part in checkpoint dependency tracking.
# Exposes NoDependency (opt-out marker), wrap_or_unwrap and sticky_attribute_assignment (used when
# attributes are assigned on trackable objects), the base TrackableDataStructure, the append-only
# List and Mapping containers, the mutation-tracking wrappers ListWrapper, _DictWrapper and
# _TupleWrapper, and small helpers used by checkpoint restoration (set_list_item, set_tuple_item).
# Relates to the trackable base module (dependency tracking) and layer_utils (layer discovery).
# Import-time side effects: registers a few functions with the public API export machinery; it
# imports the wrapt package, falling back to a bundled copy if the system package is missing.
"""Trackable data structures."""
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
import collections
import copy
import sys

try:
  import wrapt
except ImportError:
  # Fall back to the build-time dependency if the system package is not available.
  from .....third_party import wrapt  # pylint: disable=relative-beyond-top-level

from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


class NoDependency:
  # [CLASS DOC]
  # Marker wrapper around a value. When such an object is assigned as an attribute of a trackable
  # object, no checkpoint dependency is created and the attribute stores the unwrapped value.
  # Uses slots and holds a single attribute, value, with the wrapped object. Constructed directly
  # by users; unwrapped by wrap_or_unwrap and sticky_attribute_assignment.
  """Allows attribute assignment to `Trackable` objects with no dependency.

  Example usage:
  ```python
  obj = Trackable()
  obj.has_dependency = tf.Variable(0., name="dep")
  obj.no_dependency = NoDependency(tf.Variable(1., name="nodep"))
  assert obj.no_dependency.name == "nodep:0"
  ```

  `obj` in this example has a dependency on the variable "dep", and both
  attributes contain un-wrapped `Variable` objects.

  `NoDependency` also works with `tf.keras.Model`, but only for checkpoint
  dependencies: wrapping a `Layer` in `NoDependency` will assign the (unwrapped)
  `Layer` to the attribute without a checkpoint dependency, but the `Model` will
  still track the `Layer` (so it will appear in `Model.layers`, and its
  variables will appear in `Model.variables`).
  """

  __slots__ = ["value"]

  def __init__(self, value):
    # [FUNCTION DOC]
    # Constructor. Stores the given value (any object) in the value attribute so it can later be
    # unwrapped. Returns nothing, raises nothing and has no other side effects.
    self.value = value


def _should_wrap_tuple(t):
  # [FUNCTION DOC]
  # HELPER: decides whether wrap_or_unwrap must wrap a tuple in a trackable tuple wrapper.
  # Input: a tuple. Returns true if any element is a NoDependency marker, a trackable object, an
  # exact dict, ordered dict or list (subclasses are deliberately excluded), or a nested tuple
  # that itself needs wrapping (checked recursively). Otherwise returns false, since immutable
  # tuples of plain values need no tracking. Pure function, raises nothing.
  """Determine if a tuple has any trackable components."""
  # pylint: disable=unidiomatic-typecheck
  # Exact type checking to avoid mucking up custom logic in list/dict
  # subclasses, e.g. collections.Counter.
  for element in t:
    if isinstance(element, NoDependency):
      return True  # We should remove the NoDependency object from the tuple.
    if isinstance(element, base.Trackable):
      return True
    if type(element) == dict:
      return True
    if type(element) == collections.OrderedDict:
      return True
    if type(element) == list:
      return True
    if isinstance(element, tuple) and _should_wrap_tuple(element):
      return True
  # There are no trackable elements or data structures. Tuples are immutable, so
  # mutation isn't a concern. Don't wrap.
  return False
  # pylint: enable=unidiomatic-typecheck


@tf_export("__internal__.tracking.wrap", v1=[])
def wrap_or_unwrap(value):
  # [FUNCTION DOC]
  # Converts a value into its trackable form. Input: any object. Returns: the inner value if the
  # input is a NoDependency marker; the input itself if already trackable; a dict wrapper for an
  # exact dict or ordered dict; a list wrapper for an exact list; a tuple wrapper for a tuple
  # that contains trackable parts; otherwise the input unchanged. Subclasses of list and dict are
  # intentionally left alone. Raises nothing itself. Exported publicly as an internal tracking API.
  """Wraps input value into trackable data structures.

  This is mostly useful for containers like list, dict, etc, which could contain
  trackable objects in it. Wrapped data structure will be tracked when
  associated with a `tf.Module`, so that save model/checkpoint can properly
  track the dependency.

  It will also unwrap NoDependency objects.

  Args:
    value: the input object to be wrapped.

  Returns:
    Wrapped trackable data structure.
  """
  # pylint: disable=unidiomatic-typecheck
  # Exact type checking to avoid mucking up custom logic in list/dict
  # subclasses, e.g. collections.Counter.
  if isinstance(value, NoDependency):
    return value.value
  if isinstance(value, base.Trackable):
    return value  # Skip conversion for already trackable objects.
  elif type(value) == dict:
    return _DictWrapper(value)
  elif type(value) == collections.OrderedDict:
    return _DictWrapper(value)
  elif type(value) == list:
    return ListWrapper(value)
  elif isinstance(value, tuple) and _should_wrap_tuple(value):
    # There are trackable elements or data structures. Wrap the tuple.
    return _TupleWrapper(value)
  else:
    return value
  # pylint: enable=unidiomatic-typecheck


@tf_export("__internal__.tracking.sticky_attribute_assignment", v1=[])
def sticky_attribute_assignment(trackable, name, value):
  # [FUNCTION DOC]
  # Prepares a value being assigned to an attribute and records the checkpoint dependency.
  # Inputs: trackable (the object receiving the attribute), name (attribute name string used as
  # the dependency name), value (any object). The value is wrapped or unwrapped via wrap_or_unwrap.
  # If the value was a NoDependency marker, no dependency is added. If the resulting value is
  # trackable, it is registered on the owner as a dependency under the name, allowing an existing
  # dependency of that name to be overwritten. Returns the value that should actually be stored
  # in the attribute. Side effect: mutates the dependency table of the owner object.
  """Adds dependencies, generally called from __setattr__.

  This behavior is shared between Trackable and Model.

  Respects NoDependency indicators, but otherwise makes trackable objects
  out of common data structures and tracks objects by their attribute names.

  Args:
    trackable: The object to add dependencies to (generally the one having
      an attribute assigned).
    name: The attribute name being assigned.
    value: The value being assigned. Not necessarily a trackable object.

  Returns:
    The value which should be stored in the attribute (unwrapped from a
    NoDependency object if necessary).
  """
  if isinstance(value, NoDependency):
    add_dependency = False
  else:
    add_dependency = True
  value = wrap_or_unwrap(value)
  if not add_dependency:
    return value
  if isinstance(value, base.Trackable):
    trackable._track_trackable(  # pylint: disable=protected-access
        value, name=name,
        # Allow the user to switch the Trackable which is tracked by this
        # name, since assigning a new variable to an attribute has
        # historically been fine (e.g. Adam did this).
        overwrite=True)
  return value


class _UntrackableError(ValueError):
  # [CLASS DOC]
  # HELPER: private error type used by TrackableDataStructure when a non-trackable value is
  # stored in a container that only accepts trackable objects. Subclass of the value error type;
  # keeps the offending value so the message can describe it. Caught by the wrapper classes,
  # which then store the value untracked.

  def __init__(self, value):  # pylint: disable=super-init-not-called
    # [FUNCTION DOC]
    # Constructor. Remembers the offending value (any object). Deliberately does not call the
    # parent constructor, so the exception args are empty and only the string form is meaningful.
    self._value = value

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the error message text: says only trackable objects (such as layers or optimizers)
    # may be stored in a list and names the offending value, which does not inherit from the
    # trackable base class.
    return ("Only trackable objects (such as Layers or Optimizers) may be "
            f"stored in a List object. Got {self._value}, which does not "
            "inherit from Trackable.")


@tf_export("__internal__.tracking.TrackableDataStructure", v1=[])
class TrackableDataStructure(base.Trackable):
  # [CLASS DOC]
  # Base class for containers (lists, mappings, tuples and their wrappers) that hold trackable
  # objects. Provides value tracking with checkpoint dependencies, layer and variable aggregation
  # (layers, weights, trainable and non-trainable variables, updates, losses), a trainable flag,
  # an attribute-cache sentinel, and identity-based hashing and equality. State is kept in
  # attributes prefixed with _self_ so that subclasses can also derive from the wrapt proxy.
  # Subclasses must supply the _values property (the contained objects).
  """Base class for data structures which contain trackable objects."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Initializes the trainable flag to true, an empty list of extra (directly stored)
    # variables, and an attribute sentinel from layer_utils used to invalidate cached attribute
    # values. Attribute names use the _self_ prefix for compatibility with proxy objects.
    # Takes no arguments, returns nothing.
    # Attributes prefixed with "_self_" for compatibility with
    # wrapt.ObjectProxy. All additional attrs MUST conform to this pattern, as
    # extending `__slots__` on a subclass of ObjectProxy breaks in a variety of
    # ways.
    self._self_trainable = True
    self._self_extra_variables = []
    self._self_attribute_sentinel = layer_utils.AttributeSentinel(True)

  @property
  def _attribute_sentinel(self):
    # [FUNCTION DOC]
    # HELPER: accessor used by attribute caching and mutation tracking.
    # Property returning the attribute sentinel object created in the constructor.
    return self._self_attribute_sentinel

  @property
  def trainable(self):
    # [FUNCTION DOC]
    # Property returning whether the variables of this container are considered trainable.
    # Defaults to true; when false, trainable_weights is empty and all variables are non-trainable.
    return self._self_trainable

  @trainable.setter
  def trainable(self, value):
    # [FUNCTION DOC]
    # Setter for the trainable flag. Input: a boolean. Only stores the flag; it is not propagated
    # to contained layers. Returns nothing.
    self._self_trainable = value

  def _track_value(self, value, name):
    # [FUNCTION DOC]
    # HELPER: used by container subclasses to register each element.
    # Adds a checkpoint dependency on a value. Inputs: value (object to store) and name (string
    # dependency name). The value is passed through sticky_attribute_assignment; if the result is a
    # variable it is also remembered in the extra-variables list. Raises the private untrackable
    # error (a value error) if the result is not trackable. If the value has a resource-variable
    # flag attribute it is forced to true (legacy layers must use resource variables). If the
    # value has an attribute sentinel, this container's sentinel is registered as its parent.
    # Returns the (possibly unwrapped or wrapped) value to store.
    """Add a dependency on `value`."""
    # pylint: disable=g-import-not-at-top
    from tensorflow.python.ops import variables
    value = sticky_attribute_assignment(
        trackable=self, value=value, name=name)
    if isinstance(value, variables.Variable):
      self._self_extra_variables.append(value)
    if not isinstance(value, base.Trackable):
      raise _UntrackableError(value)
    if hasattr(value, "_use_resource_variables"):
      # In subclassed models, legacy layers (tf.layers) must always use
      # resource variables.
      value._use_resource_variables = True  # pylint: disable=protected-access
    value_attribute_sentinel = getattr(value, "_attribute_sentinel", None)
    if value_attribute_sentinel:
      value_attribute_sentinel.add_parent(self._attribute_sentinel)
    return value

  @property
  def _values(self):
    # [FUNCTION DOC]
    # Abstract property: subclasses return the iterable of contained objects. The base version
    # always raises a not-implemented error.
    """An iterable/sequence which may contain trackable objects."""
    raise NotImplementedError("Abstract method")

  @property
  def _layers(self):
    # [FUNCTION DOC]
    # HELPER: feeds the layers property.
    # Returns a new list of the contained values that are data structures, layers, or objects with
    # weights. Computed on demand each time so wrappers reflect their wrapped content. Includes
    # empty containers.
    """All Layers and Layer containers, including empty containers."""
    # Filter objects on demand so that wrapper objects use values from the thing
    # they're wrapping if out of sync.
    collected = []
    for obj in self._values:
      if (isinstance(obj, TrackableDataStructure)
          or layer_utils.is_layer(obj)
          or layer_utils.has_weights(obj)):
        collected.append(obj)
    return collected

  @property
  def layers(self):
    # [FUNCTION DOC]
    # Property returning a list of the contained layers, with nested layer containers flattened
    # and empty containers filtered out (via layer_utils).
    return list(layer_utils.filter_empty_layer_containers(self._layers))

  @property
  def trainable_weights(self):
    # [FUNCTION DOC]
    # Property returning the list of trainable variables. Empty if the container's trainable flag
    # is false. Otherwise concatenates trainable_variables of each trackable contained value that
    # has them, followed by directly stored variables that are trainable.
    if not self._self_trainable:
      return []
    trainable_variables = []
    for obj in self._values:
      if isinstance(obj, base.Trackable) and hasattr(
          obj, "trainable_variables"):
        trainable_variables += obj.trainable_variables
    trainable_extra_variables = [
        v for v in self._self_extra_variables if v.trainable
    ]
    return trainable_variables + trainable_extra_variables

  @property
  def non_trainable_weights(self):
    # [FUNCTION DOC]
    # Property returning the list of non-trainable variables: those from contained values plus
    # directly stored non-trainable variables. If the container itself is marked not trainable,
    # all its trainable variables are appended first, in the order: trainable from contained
    # values, trainable extras, non-trainable from contained values, non-trainable extras.
    trainable_extra_variables = [
        v for v in self._self_extra_variables if v.trainable
    ]
    non_trainable_extra_variables = [
        v for v in self._self_extra_variables if not v.trainable
    ]
    non_trainable_variables = []
    for obj in self._values:
      if isinstance(obj, base.Trackable) and hasattr(
          obj, "non_trainable_variables"):
        non_trainable_variables += obj.non_trainable_variables

    if not self._self_trainable:
      # Return order is all trainable vars, then all non-trainable vars.
      trainable_variables = []
      for obj in self._values:
        if isinstance(obj, base.Trackable) and hasattr(
            obj, "trainable_variables"):
          trainable_variables += obj.trainable_variables

      non_trainable_variables = (
          trainable_variables + trainable_extra_variables +
          non_trainable_variables + non_trainable_extra_variables)
    else:
      non_trainable_variables = (
          non_trainable_variables + non_trainable_extra_variables)

    return non_trainable_variables

  @property
  def weights(self):
    # [FUNCTION DOC]
    # Property returning all variables: trainable weights followed by non-trainable weights.
    return self.trainable_weights + self.non_trainable_weights

  @property
  def trainable_variables(self):
    # [FUNCTION DOC]
    # Property alias of trainable_weights.
    return self.trainable_weights

  @property
  def non_trainable_variables(self):
    # [FUNCTION DOC]
    # Property alias of non_trainable_weights.
    return self.non_trainable_weights

  @property
  def variables(self):
    # [FUNCTION DOC]
    # Property alias of weights (all variables, trainable first).
    return self.weights

  @property
  def updates(self):
    # [FUNCTION DOC]
    # Property returning the concatenated update operations of all contained layers that expose
    # an updates attribute. Not filtered by inputs since a container has no inputs.
    """Aggregate updates from any `Layer` instances."""
    # Updates and conditional losses are forwarded as-is rather than being
    # filtered based on inputs, since this is just a container and won't ever
    # have any inputs.
    aggregated = []
    for layer in self.layers:
      if hasattr(layer, "updates"):
        aggregated += layer.updates
    return aggregated

  @property
  def losses(self):
    # [FUNCTION DOC]
    # Property returning the concatenated losses of all contained layers that expose a losses
    # attribute.
    """Aggregate losses from any `Layer` instances."""
    aggregated = []
    for layer in self.layers:
      if hasattr(layer, "losses"):
        aggregated += layer.losses
    return aggregated

  def __hash__(self):
    # [FUNCTION DOC]
    # Identity-based hash (object id) so containers can be used in sets and as dictionary keys.
    # Support object-identity hashing, so these structures can be used as keys
    # in sets/dicts.
    return id(self)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Identity-based equality: true only when other is the very same object. Mirrors tensor
    # behavior so containers work in set and dictionary membership.
    # Similar to Tensors, trackable data structures use object-identity
    # equality to support set/dict membership.
    return self is other


class List(TrackableDataStructure, collections_abc.Sequence):
  # [CLASS DOC]
  # Append-only trackable sequence. Keeps its elements in a backing list (storage), names each
  # element by its decimal index as the checkpoint dependency name, and requires elements to be
  # trackable. Supports reading, length, concatenation and repetition (which return plain lists),
  # copying, append, extend and in-place add and multiply. Is a pure container: it does not call
  # the layers it holds. Base class of ListWrapper.
  """An append-only sequence type which is trackable.

  Maintains checkpoint dependencies on its contents (which must also be
  trackable), and forwards any `Layer` metadata such as updates and losses.

  Note that `List` is purely a container. It lets a `tf.keras.Model` or
  other trackable object know about its contents, but does not call any
  `Layer` instances which are added to it. To indicate a sequence of `Layer`
  instances which should be called sequentially, use `tf.keras.Sequential`.

  Example usage:
  ```python
  class HasList(tf.keras.Model):

    def __init__(self):
      super().__init__()
      self.layer_list = List([layers.Dense(3)])
      self.layer_list.append(layers.Dense(4))

    def call(self, x):
      aggregation = 0.
      for l in self.layer_list:
        x = l(x)
        aggregation += tf.reduce_sum(x)
      return aggregation
  ```

  This kind of wrapping is necessary because `Trackable` objects do not
  (yet) deeply inspect regular Python data structures, so for example assigning
  a regular list (`self.layer_list = [layers.Dense(3)]`) does not create a
  checkpoint dependency and does not add the `Layer` instance's weights to its
  parent `Model`.
  """

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Arguments are forwarded to the storage factory (by default the built-in list
    # constructor, so an optional iterable of initial elements). Creates the backing storage, then
    # registers each initial element as a tracked dependency named by its index, replacing it with
    # the tracked form. Raises the untrackable value error if an element is not trackable
    # (ListWrapper overrides tracking to allow such elements).
    """Construct a new sequence. Arguments are passed to `list()`."""
    super().__init__()
    self._storage = self._make_storage(*args, **kwargs)
    for index, element in enumerate(self._storage):
      self._storage[index] = self._track_value(
          element, name=self._name_element(index))

  def copy(self):
    # [FUNCTION DOC]
    # Returns a new object of the same type built from a shallow copy of the backing storage; the
    # elements are re-tracked by the new container. Takes no arguments.
    return type(self)(copy.copy(self._storage))

  def __copy__(self):
    # [FUNCTION DOC]
    # Support for the shallow copy protocol; delegates to the copy method.
    return self.copy()

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Support for deep copying. Input: memo, the mapping of already copied objects. Returns a new
    # object of the same type built from a deep copy of the backing storage.
    return type(self)(copy.deepcopy(self._storage, memo))

  def _make_storage(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: factory hook for the constructor, overridden by subclasses.
    # Creates the backing storage by calling the built-in list constructor with the given
    # arguments. Returns the new list.
    """Determines the backing storage (overridden in subclasses)."""
    return list(*args, **kwargs)

  def _name_element(self, index):
    # [FUNCTION DOC]
    # HELPER: naming hook for dependency tracking.
    # Input: integer index. Returns the index formatted as a decimal string, used as the
    # checkpoint dependency name of that element.
    return "%d" % (index,)

  @property
  def _values(self):
    # [FUNCTION DOC]
    # Property telling the base class which objects to inspect: the list itself (iterating over
    # its elements).
    """Collect values for TrackableDataStructure."""
    return self

  def append(self, value):
    # [FUNCTION DOC]
    # Appends one value. Input: the value, which must be trackable. It is registered as a
    # dependency named by its new index and the tracked form is added to the storage. Raises the
    # untrackable value error if it is not trackable. Returns nothing.
    """Add a new trackable value."""
    value = self._track_value(value, self._name_element(len(self._storage)))
    self._storage.append(value)

  def extend(self, values):
    # [FUNCTION DOC]
    # Appends every item of an iterable, one at a time, via append. Raises the untrackable value
    # error on the first non-trackable item (earlier items stay added). Returns nothing.
    """Add a sequence of trackable values."""
    for value in values:
      self.append(value)

  def __iadd__(self, values):
    # [FUNCTION DOC]
    # In-place addition: extends with the given iterable and returns this same object.
    self.extend(values)
    return self

  def __add__(self, other):
    # [FUNCTION DOC]
    # Concatenation. Input: another sequence (or another container of this kind, whose storage is
    # used). Returns a new plain list, not a tracked container.
    return self._storage + getattr(other, "_storage", other)

  def __imul__(self, y):
    # [FUNCTION DOC]
    # In-place repetition. Input: integer y. Raises a value error if y is zero or negative
    # (the list is append-only and cannot shrink). Otherwise appends y minus one additional copies
    # of the current elements (re-tracked via append) and returns this object.
    if y <= 0:
      raise ValueError(
          f"List only supports append, multiplying in place by {y} removes "
          "elements.")

    n = len(self._storage)
    for _ in range(y - 1):
      for i in range(n):
        self.append(self._storage[i])

    return self

  def __mul__(self, n):
    # [FUNCTION DOC]
    # Repetition. Input: integer n. Returns a new plain list with the elements repeated n times.
    return self._storage * n

  def __rmul__(self, n):
    # [FUNCTION DOC]
    # Reflected repetition (integer times container); same result as multiplication.
    return self * n

  def __radd__(self, other):
    # [FUNCTION DOC]
    # Reflected concatenation. Input: a sequence on the left. Returns a new plain list of that
    # sequence followed by this container's elements.
    return other + self._storage

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Indexing. Input: integer index or slice. Returns the element or a plain list slice from the
    # storage. Raises an index error for out-of-range indexes.
    return self._storage[key]

  def __getslice__(self, i, j):
    # [FUNCTION DOC]
    # KNOWN ISSUE: legacy Python 2 slice hook; Python 3 never calls it (slicing goes through
    # indexing), so this method is effectively dead code kept for compatibility.
    # Returns the plain list slice of the storage between the start and end positions.
    return self._storage[slice(i, j)]

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of elements in the storage.
    return len(self._storage)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text form showing the class label and the representation of the backing storage.
    return "List(%s)" % (repr(self._storage),)

  def __sizeof__(self):
    # [FUNCTION DOC]
    # Returns the memory size in bytes: the base object size plus the size of the backing storage
    # list (not counting the elements).
    return super().__sizeof__() + sys.getsizeof(self._storage)


# TODO(tomhennigan) Update to collections.UserList?
# TODO(allenl): Try switching this to wrapt.ObjectProxy again when we drop
# Python 3.4 support (may still be tricky).
class ListWrapper(
    List,
    collections_abc.MutableSequence,
    # Shadowed, but there for isinstance checks.
    list):
  # [CLASS DOC]
  # Mutable trackable list used when a plain list is assigned to an attribute of a trackable
  # object. Behaves like a built-in list but records "bad" mutations: replacing, deleting,
  # inserting or sorting that touch trackables set a non-append-mutation flag, and changes made
  # to the original wrapped list behind the wrapper's back set an external-modification flag.
  # Either flag makes saving (listing trackable children) raise a value error. Keeps a snapshot
  # of the storage for detecting external changes. Also inherits from the built-in list only for
  # type checks. Extends List.
  """Wraps the built-in `list` to support restore-on-create for variables.

  Unlike `List`, this sequence type is mutable in the same ways built-in lists
  are. Instead of throwing an error immediately like `List`, it records
  problematic mutations (e.g. assigning a new element to a position already
  occupied, meaning both elements get the same names at different times) and
  refuses to save.

  On assignment to an attribute of a Model or Trackable object, Python
  lists are replaced with ListWrapper. Wrapping a list in a
  `NoDependency` object prevents this.
  """

  def __init__(self, wrapped_list):
    # [FUNCTION DOC]
    # Constructor. Input: the list to wrap, which is used directly as the backing storage (not
    # copied). Clears the two mutation flags, runs the List constructor (tracking each element,
    # allowing non-trackable ones) and stores a shallow snapshot of the storage used later to
    # detect modification outside the wrapper. Returns nothing.
    """Construct a new list wrapper.

    Args:
      wrapped_list: The initial value of the data structure. A shallow copy may
        be maintained for error checking. `wrapped_list` itself should not be
        modified directly after constructing the `ListWrapper`, and if changes
        are detected the `ListWrapper` will throw an exception on save.
    """
    # Monotonic flags which indicate this object would not be restored properly,
    # and therefore should throw an error on save to avoid giving the impression
    # that restoring it will work.
    self._non_append_mutation_value = False
    self._external_modification_value = False
    super().__init__(wrapped_list)
    self._last_wrapped_list_snapshot = list(self._storage)

  @property
  def _non_append_mutation(self):
    # [FUNCTION DOC]
    # Property: true if a mutation other than appending (replace, delete, insert, sort) has
    # happened, meaning the list can no longer be saved.
    return self._non_append_mutation_value

  @_non_append_mutation.setter
  def _non_append_mutation(self, value):
    # [FUNCTION DOC]
    # Setter for the non-append-mutation flag. Input: boolean. Side effect: invalidates all cached
    # attribute values via the attribute sentinel immediately, then stores the flag.
    # Trackable only cares that a mutation occurred at some point; when
    # attempting to save it checks whether a mutation occurred and the object is
    # in a "dirty" state but otherwise the specifics of how it got to that state
    # are ignored. By contrast, the attribute cache needs to signal the mutation
    # immediately since a caller could query the value of an attribute (And
    # should not hit the cached value since the mutation may have affected the
    # result.)
    self._attribute_sentinel.invalidate_all()
    self._non_append_mutation_value = value

  @property
  def _external_modification(self):
    # [FUNCTION DOC]
    # Property: true if the wrapped list was found to have been changed outside the wrapper.
    return self._external_modification_value

  @_external_modification.setter
  def _external_modification(self, value):
    # [FUNCTION DOC]
    # Setter for the external-modification flag. Input: boolean. Invalidates all cached attribute
    # values via the attribute sentinel, then stores the flag.
    # Invalidate for the same reason as `_non_append_mutation`
    self._attribute_sentinel.invalidate_all()
    self._external_modification_value = value

  # pylint: disable=protected-access
  def __copy__(self):
    # [FUNCTION DOC]
    # Shallow copy. Builds the copy through the parent implementation and carries over both
    # mutation flags so a dirty list stays dirty. Returns the new wrapper.
    copied = super().__copy__()
    copied._non_append_mutation = self._non_append_mutation
    copied._external_modification = self._external_modification
    return copied

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Deep copy. Input: memo mapping of already copied objects. Uses the parent implementation and
    # copies both mutation flags onto the result, which is returned.
    copied = super().__deepcopy__(memo)
    copied._non_append_mutation = self._non_append_mutation
    copied._external_modification = self._external_modification
    return copied
  # pylint: enable=protected-access

  def __reduce_ex__(self, protocol):
    # [FUNCTION DOC]
    # Pickle support. Input: pickle protocol number (unused). Returns a recipe that re-creates the
    # wrapper from its class and the backing storage list. Mutation flags are not preserved.
    return (self.__class__,
            (self._storage,))

  def _make_storage(self, wrapped_list):
    # [FUNCTION DOC]
    # HELPER: storage factory override used by the List constructor.
    # Input: the user's list. Returns that very list, so the wrapper shares it rather than
    # copying it.
    """Use the user's original list for storage."""
    return wrapped_list

  def _check_external_modification(self):
    # [FUNCTION DOC]
    # HELPER: called before every mutation and before saving.
    # Does nothing if a flag is already set. Otherwise compares the storage with the last snapshot;
    # if they differ, sets the external-modification flag and discards the snapshot. Returns nothing.
    """Checks for any changes to the wrapped list not through the wrapper."""
    if self._external_modification or self._non_append_mutation:
      return
    if self._storage != self._last_wrapped_list_snapshot:
      self._external_modification = True
      self._last_wrapped_list_snapshot = None

  def _update_snapshot(self):
    # [FUNCTION DOC]
    # HELPER: called after every mutation made through the wrapper.
    # Invalidates cached attributes, and, unless the list is already marked dirty, stores a fresh
    # shallow snapshot of the storage as the new baseline. Returns nothing.
    """Acknowledges tracked changes to the wrapped list."""

    # Mutation tracking for attributes reuses the same infrastructure as
    # Trackable mutation tracking.
    self._attribute_sentinel.invalidate_all()
    if self._external_modification or self._non_append_mutation:
      return
    self._last_wrapped_list_snapshot = list(self._storage)

  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    # [FUNCTION DOC]
    # Lists the dependencies to save. Inputs: save_type (checkpoint by default, or saved-model)
    # and extra keyword options forwarded to the parent. First checks for external modification.
    # Raises a value error if a non-append mutation occurred (an element was replaced, deleted or
    # moved) or if the wrapped list was modified outside the wrapper, explaining the object cannot
    # be restored on creation. Otherwise returns the mapping of names to children from the parent;
    # for saved-model saving it also adds function elements keyed by their index string.
    self._check_external_modification()
    if self._non_append_mutation:
      raise ValueError(
          f"Unable to save the object {self} (a list wrapper constructed to "
          "track trackable TensorFlow objects). A list element was replaced "
          "(__setitem__, __setslice__), deleted (__delitem__, __delslice__), "
          "or moved (sort). In order to support restoration on object "
          "creation, tracking is exclusively for append-only data structures."
          "\n\nIf you don't need this list checkpointed, wrap it in a "
          "non-trackable object; it will be subsequently ignored.")
    if self._external_modification:
      raise ValueError(
          f"Unable to save the object {self} (a list wrapper constructed to "
          "track trackable TensorFlow objects). The wrapped list was modified "
          f"outside the wrapper (its final value was {self._storage}, its value"
          " when a checkpoint dependency was added was "
          f"{self._last_wrapped_list_snapshot}), which breaks "
          "restoration on object creation.\n\nIf you don't need this list "
          "checkpointed, wrap it in a NoDependency object; it will be "
          "subsequently ignored.")
    children = super()._trackable_children(save_type, **kwargs)

    if save_type == base.SaveType.SAVEDMODEL:
      # Add functions to be serialized.
      children.update({
          str(key): value
          for key, value in enumerate(self)
          if _is_function(value)
      })

    return children

  def _has_mutation_or_trackable(self):
    # [FUNCTION DOC]
    # HELPER: guards destructive operations (delete, insert, sort, shrinking multiply).
    # Returns true if a non-append mutation was already recorded, otherwise true if any stored
    # element is a trackable object. Used to decide whether an operation must poison the list.
    """Short-circuits a check for trackables if there's already a mutation."""
    if self._non_append_mutation:
      return True
    return any(isinstance(element, base.Trackable) for element in self._storage)

  def __delitem__(self, key):
    # [FUNCTION DOC]
    # Deletes an element or slice. Input: index or slice. Checks for external modification, marks
    # the list as non-append-mutated if it holds trackables or was already mutated, removes the
    # item from storage (index error if out of range) and refreshes the snapshot.
    self._check_external_modification()
    if self._has_mutation_or_trackable():
      self._non_append_mutation = True
    del self._storage[key]
    self._update_snapshot()

  def __setitem__(self, key, value):
    # [FUNCTION DOC]
    # Assigns an element or slice. Inputs: key (integer index or slice) and value (a single item,
    # or an iterable for a slice). Checks for external modification first. For a slice, it
    # compares the storage before and after position by position: if any replaced old item was
    # trackable the list is marked non-append-mutated, and any new or changed item is tracked
    # under its position name. For an index, overwriting a trackable marks the list mutated and the
    # new value is tracked. Refreshes the snapshot. Raises an index error for a bad index.
    self._check_external_modification()

    if isinstance(key, slice):
      # Note: this is quite inefficient, but the list API supports a broad range
      # of slice setters (e.g. truncate, extend, replace) and imitating this
      # for a range of Python versions is non-trivial.
      storage_copy = list(self._storage)
      self._storage[key] = value

      len_before = len(storage_copy)
      len_now = len(self._storage)
      for i in range(max(len_before, len_now)):
        value_now = self._storage[i] if i < len_now else None
        value_before = storage_copy[i] if i < len_before else None

        if isinstance(value_before, base.Trackable):
          self._non_append_mutation = True

        if value_now is not None and value_now != value_before:
          self._storage[i] = self._track_value(self._storage[i],
                                               self._name_element(i))

    else:
      if isinstance(self._storage[key], base.Trackable):
        self._non_append_mutation = True
      self._storage[key] = self._track_value(value, self._name_element(key))

    self._update_snapshot()

  def append(self, value):
    # [FUNCTION DOC]
    # Appends a value (trackable or not). Checks for external modification, appends via the
    # parent (tracking it under its index name) and refreshes the snapshot. Returns nothing.
    """Add a new trackable value."""
    self._check_external_modification()
    super().append(value)
    self._update_snapshot()

  def extend(self, values):
    # [FUNCTION DOC]
    # Appends each item of an iterable. Checks for external modification, extends via the parent
    # and refreshes the snapshot once at the end. Returns nothing.
    """Add a sequence of trackable values."""
    self._check_external_modification()
    super().extend(values)
    self._update_snapshot()

  def __imul__(self, y):
    # [FUNCTION DOC]
    # In-place repetition. Input: integer y. For y of zero or less the list is emptied (or
    # repeated negatively), after an external-modification check; if it holds trackables it is
    # marked non-append-mutated; the snapshot is refreshed and this object returned. For
    # positive y it defers to the parent, which appends copies of the elements.
    if y <= 0:
      self._check_external_modification()
      if self._has_mutation_or_trackable():
        self._non_append_mutation = True
      self._storage *= y
      self._update_snapshot()
      return self

    # Relies on super() calling append, which updates the snapshot.
    return super().__imul__(y)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality by value like a built-in list. Input: another wrapper (its storage is used) or any
    # sequence. Returns a boolean.
    return self._storage == getattr(other, "_storage", other)

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Inequality by value; negation of the list comparison against other's storage or other.
    return self._storage != getattr(other, "_storage", other)

  def __lt__(self, other):
    # [FUNCTION DOC]
    # Ordering comparison (less than) of storage against another wrapper's storage or a sequence.
    return self._storage < getattr(other, "_storage", other)

  def __le__(self, other):
    # [FUNCTION DOC]
    # Ordering comparison (less than or equal) against another wrapper's storage or a sequence.
    return self._storage <= getattr(other, "_storage", other)

  def __gt__(self, other):
    # [FUNCTION DOC]
    # Ordering comparison (greater than) against another wrapper's storage or a sequence.
    return self._storage > getattr(other, "_storage", other)

  def __ge__(self, other):
    # [FUNCTION DOC]
    # Ordering comparison (greater than or equal) against another wrapper's storage or a sequence.
    return self._storage >= getattr(other, "_storage", other)

  def __hash__(self):
    # [FUNCTION DOC]
    # Hashing is disabled: always raises a type error, like built-in lists, because the wrapper
    # compares by value and is mutable.
    # List wrappers need to compare like regular lists, and so like regular
    # lists they don't belong in hash tables.
    raise TypeError("unhashable type: 'ListWrapper'")

  def insert(self, index, obj):
    # [FUNCTION DOC]
    # Inserts an object before a position. Inputs: index (integer) and obj (any object). Checks for
    # external modification; marks the list non-append-mutated if it already holds trackables, was
    # mutated, or if the new object is trackable; inserts into storage and refreshes the snapshot.
    self._check_external_modification()
    if (self._has_mutation_or_trackable() or isinstance(obj, base.Trackable)):
      self._non_append_mutation = True
    self._storage.insert(index, obj)
    self._update_snapshot()

  def sort(self):
    # [FUNCTION DOC]
    # Sorts the storage in place in natural order (no key or reverse options are accepted).
    # Marks the list non-append-mutated if it holds trackables or was already mutated, since
    # reordering changes names. Refreshes the snapshot. Raises a type error if items are not
    # comparable.
    self._check_external_modification()
    if self._has_mutation_or_trackable():
      self._non_append_mutation = True
    self._storage.sort()
    self._update_snapshot()

  def __setslice__(self, i, j, y):
    # [FUNCTION DOC]
    # KNOWN ISSUE: legacy Python 2 slice-assignment hook; Python 3 does not call it, so it is
    # effectively dead code.
    # Forwards to item assignment with a slice from start i to end j and replacement y.
    self.__setitem__(slice(i, j), y)

  def __delslice__(self, i, j):
    # [FUNCTION DOC]
    # KNOWN ISSUE: legacy Python 2 slice-deletion hook; Python 3 does not call it, so it is
    # effectively dead code.
    # Deletes the slice between start i and end j: checks external modification, marks the list
    # non-append-mutated if it has trackables, deletes and refreshes the snapshot.
    self._check_external_modification()
    if self._has_mutation_or_trackable():
      self._non_append_mutation = True
    del self._storage[slice(i, j)]
    self._update_snapshot()

  def _track_value(self, value, name):
    # [FUNCTION DOC]
    # HELPER: tracking hook used for each stored element.
    # Inputs: value and dependency name. Tries the parent tracking; if the value is not trackable
    # (the untrackable error is a value error) falls back to only unwrapping NoDependency markers
    # through sticky_attribute_assignment. Returns the value to store. Unlike the parent, never
    # raises for non-trackable values.
    """Allows storage of non-trackable objects."""
    try:
      value = super()._track_value(value=value, name=name)
    except ValueError:
      # Even if this value isn't trackable, we need to make sure
      # NoDependency objects get unwrapped.
      value = sticky_attribute_assignment(
          trackable=self, value=value, name=name)
    return value

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text form showing the class label and the representation of the storage.
    return "ListWrapper(%s)" % (repr(self._storage),)


class Mapping(TrackableDataStructure, collections_abc.Mapping):
  # [CLASS DOC]
  # Append-only trackable mapping with string keys. Keeps a backing dictionary (storage), names
  # each value by its key as the checkpoint dependency name, requires trackable values, and
  # forbids replacing or deleting keys. Read access follows the standard read-only mapping
  # interface, iterating in insertion order; the tracked values are inspected in key-sorted order.
  """An append-only trackable mapping data structure with string keys.

  Maintains checkpoint dependencies on its contents (which must also be
  trackable), named based on its keys.

  Note that once a key has been added, it may not be deleted or replaced.
  """

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says "Construct a new sequence" but this builds a mapping; the
    # arguments are passed to the dict constructor, not to a list.
    # Constructor. Arguments are forwarded to the dictionary constructor (optional mapping or
    # key-value pairs plus keyword entries). Every initial value is registered as a tracked
    # dependency named by its key and replaced by the tracked form. Raises a type error for a
    # non-string key and the untrackable value error for a non-trackable value.
    """Construct a new sequence. Arguments are passed to `dict()`."""
    super().__init__()
    self._storage = self._make_storage(*args, **kwargs)
    self._storage.update(
        {key: self._track_value(
            value, name=self._name_element(key))
         for key, value in self._storage.items()})

  def __copy__(self):
    # [FUNCTION DOC]
    # Shallow copy: returns a new object of the same type built from a shallow copy of the
    # storage (values are re-tracked by the new mapping).
    return type(self)(copy.copy(self._storage))

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Deep copy. Input: memo mapping of already copied objects. Returns a new object of the same
    # type built from a deep copy of the storage.
    return type(self)(copy.deepcopy(self._storage, memo))

  def _make_storage(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: storage factory used by the constructor. Builds and returns a new dictionary from
    # the given arguments.
    return dict(*args, **kwargs)

  @property
  def _values(self):
    # [FUNCTION DOC]
    # Property giving the base class its contained objects. Returns the values as a tuple ordered
    # by key (deterministic ordering), or an empty list when the mapping is empty. Note the return
    # type differs (tuple versus list) between the two cases.
    """Collect values for TrackableDataStructure."""
    # Sort items deterministically by key
    ordered = list(zip(*sorted(self.items(), key=lambda it: it[0])))
    if ordered:
      return ordered[1]
    return []

  def _name_element(self, key):
    # [FUNCTION DOC]
    # HELPER: naming hook for dependency tracking. Input: a key. Raises a type error if the key is
    # not a string; otherwise returns the key as the dependency name.
    if not isinstance(key, str):
      raise TypeError(
          f"Mapping accepts only string keys, but got a key {repr(key)}.")
    return str(key)

  def __setitem__(self, key, value):
    # [FUNCTION DOC]
    # Adds a new entry. Inputs: key (string) and value (trackable). Tracks the value under the key
    # name, then stores it only if the key was absent. Raises a type error for non-string keys, the
    # untrackable value error for non-trackable values, and a value error if the key already exists
    # with a different object (overwrite is forbidden; re-adding the same object is allowed).
    name = self._name_element(key)
    value = self._track_value(value, name=name)
    current_value = self._storage.setdefault(key, value)
    if current_value is not value:
      raise ValueError(
          "Mappings are an append-only data structure. Tried to overwrite the "
          f"key '{key}' with value {value}, but it already contains "
          f"{current_value}")

  def update(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Adds many entries. Arguments follow the dictionary constructor (mapping, pairs, keywords).
    # Each entry is added through item assignment, so the same errors apply (non-string key,
    # non-trackable value, overwrite of an existing different value). Earlier entries stay added
    # if a later one fails. Returns nothing.
    for key, value in dict(*args, **kwargs).items():
      self[key] = value

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Looks up a value by key in the storage; raises a key error if missing.
    return self._storage[key]

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of entries.
    return len(self._storage)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text form showing the class label and the representation of the storage.
    return "Mapping(%s)" % (repr(self._storage),)

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the keys in insertion order.
    return iter(self._storage)


class _DictWrapper(TrackableDataStructure, wrapt.ObjectProxy):
  # [CLASS DOC]
  # Private proxy wrapping a built-in dict (or ordered dict) assigned to an attribute of a
  # trackable object. It is to Mapping what ListWrapper is to List: allows any mutation, but
  # records problems (a non-string key mapped to a trackable, or changes to the wrapped dict made
  # outside the wrapper) and then raises on save. Keeps a snapshot of the content to detect
  # external changes. Combines the trackable data structure base with the wrapt object proxy, so
  # its own state uses _self_ prefixed names and most operations forward to the wrapped dict.
  """Wraps built-in dicts to support restore-on-create for variables.

  _DictWrapper is to Mapping as ListWrapper is to List. Unlike Mapping,
  _DictWrapper allows non-string keys and values and arbitrary mutations (delete
  keys, reassign values). Like ListWrapper, these mutations mean that
  _DictWrapper will raise an exception on save.
  """

  def __init__(self, wrapped_dict=None):
    # [FUNCTION DOC]
    # Constructor. Input: optional dict to wrap (nothing means a new empty dict; a non-mapping
    # such as a sequence of pairs is converted to a dict, which then is a copy). Initializes the
    # proxy and the trackable base, clears the non-string-key and external-modification flags,
    # tracks every existing value under its key name (replacing values with tracked forms in the
    # wrapped dict) and takes the first snapshot. Returns nothing.
    if wrapped_dict is None:
      # Allow zero-argument construction, e.g. from session.run's re-wrapping.
      wrapped_dict = {}
    if not isinstance(wrapped_dict, collections_abc.Mapping):
      # Allow construction from a sequence, e.g. from nest.pack_sequence_as.
      wrapped_dict = dict(wrapped_dict)
    wrapt.ObjectProxy.__init__(self, wrapped_dict)
    TrackableDataStructure.__init__(self)
    self._self_non_string_key = False
    self._self_external_modification = False
    self.__wrapped__.update(
        {key: self._track_value(
            value, name=self._name_element(key))
         for key, value in self.__wrapped__.items()})
    self._update_snapshot()

  def __reduce_ex__(self, protocol):
    # [FUNCTION DOC]
    # Pickle support. Input: protocol number (unused). Returns a recipe that rebuilds the wrapper
    # from its class and the wrapped dict.
    return (self.__class__,
            (self.__wrapped__,))

  def __getattribute__(self, name):
    # [FUNCTION DOC]
    # Attribute lookup override. Input: attribute name. For the instance dictionary name it
    # returns the proxy's own dictionary, or an empty dictionary if that lookup fails. For names
    # that are properties on this class, bypasses the proxy and reads them from the wrapper itself.
    # Otherwise uses the proxy lookup, converting a type error (seen with newer wrapt versions)
    # into an attribute error chained to the original.
    if name == "__dict__":
      # Returns __dict__ from wrapt.ObjectProxy
      try:
        return object.__getattribute__(self, "__dict__")
      except (AttributeError, TypeError):
        return {}
    if (hasattr(type(self), name)
        and isinstance(getattr(type(self), name), property)):
      # Bypass ObjectProxy for properties. Whether this workaround is necessary
      # appears to depend on the Python version but not the wrapt version: 3.4
      # in particular seems to look up properties on the wrapped object instead
      # of the wrapper without this logic.
      return object.__getattribute__(self, name)
    else:
      # Raise TypeError as AttributeError to fix breakage in wrapt 1.15 for
      # `__getattribute__` as suggested in discussion with library author in
      # GitHub https://github.com/GrahamDumpleton/wrapt/issues/231
      try:
        return super().__getattribute__(name)
      except TypeError as e:
        raise AttributeError from e

  def copy(self):
    # [FUNCTION DOC]
    # Returns a shallow copy of the wrapper (using the shallow copy protocol), which is again a
    # wrapper rather than a plain dict.
    return copy.copy(self)

  # pylint: disable=protected-access
  def __copy__(self):
    # [FUNCTION DOC]
    # Shallow copy: wraps a shallow copy of the wrapped dict in a new wrapper and carries over the
    # external-modification and non-string-key flags. Returns the new wrapper.
    copied = _DictWrapper(copy.copy(self.__wrapped__))
    copied._self_external_modification = self._self_external_modification
    copied._self_non_string_key = self._self_non_string_key
    return copied

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Deep copy. Input: memo mapping of copied objects. Wraps a deep copy of the wrapped dict in a
    # new wrapper and carries over both flags. Returns the new wrapper.
    copied = _DictWrapper(copy.deepcopy(self.__wrapped__, memo))
    copied._self_external_modification = self._self_external_modification
    copied._self_non_string_key = self._self_non_string_key
    return copied
  # pylint: enable=protected-access

  @property
  def _values(self):
    # [FUNCTION DOC]
    # Property giving the base class its contained objects: the values ordered by key, as a tuple,
    # or an empty list if there are none. Sorting requires keys to be mutually comparable and
    # raises a type error for mixed key types.
    """Collect values for TrackableDataStructure."""
    # Sort items deterministically by key
    ordered = list(zip(*sorted(self.items(), key=lambda it: it[0])))
    if ordered:
      return ordered[1]
    return []

  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    # [FUNCTION DOC]
    # Lists the dependencies to save. Inputs: save_type (checkpoint by default, or saved-model) and
    # extra options forwarded to the parent. Checks for external modification, then raises a value
    # error if a non-string key maps to a trackable, or if the wrapped dict was modified outside the
    # wrapper. Asserts the wrapper is not dirty. Returns the mapping of names to children from the
    # parent; for saved-model saving also adds function values keyed by their dictionary keys.
    """Check that the object is saveable before listing its dependencies."""
    self._check_self_external_modification()
    if self._self_non_string_key:
      raise ValueError(
          f"Unable to save the object {self} (a dictionary wrapper constructed "
          "automatically on attribute assignment). The wrapped dictionary "
          "contains a non-string key which maps to a trackable object or "
          "mutable data structure.\n\nIf you don't need this dictionary "
          "checkpointed, wrap it in a non-trackable "
          "object; it will be subsequently ignored.")
    if self._self_external_modification:
      raise ValueError(
          f"Unable to save the object {self} (a dictionary wrapper constructed "
          "automatically on attribute assignment). The wrapped dictionary was "
          f"modified outside the wrapper (its final value was {self}, its value"
          " when a checkpoint dependency was added was "
          f"{self._self_last_wrapped_dict_snapshot}), which breaks "
          "restoration on object creation.\n\nIf you don't need this "
          "dictionary checkpointed, wrap it in a "
          "non-trackable object; it will be subsequently ignored.")
    assert not self._dirty  # Any reason for dirtiness should have an exception.
    children = super()._trackable_children(save_type, **kwargs)

    if save_type == base.SaveType.SAVEDMODEL:
      # Add functions to be serialized.
      children.update(
          {key: value for key, value in self.items() if _is_function(value)})

    return children

  @property
  def _dirty(self):
    # [FUNCTION DOC]
    # Property: true if the wrapper already has a mutation that prevents saving (external
    # modification or a non-string key mapping to a trackable).
    """Check if there has already been a mutation which prevents saving."""
    return (self._self_external_modification
            or self._self_non_string_key)

  def _check_self_external_modification(self):
    # [FUNCTION DOC]
    # HELPER: called before mutations and saving.
    # Does nothing if already dirty. Otherwise compares the current content with the last snapshot;
    # if different, sets the external-modification flag and drops the snapshot. Returns nothing.
    """Checks for any changes to the wrapped dict not through the wrapper."""
    if self._dirty:
      return
    if self != self._self_last_wrapped_dict_snapshot:
      self._self_external_modification = True
      self._self_last_wrapped_dict_snapshot = None

  def _update_snapshot(self):
    # [FUNCTION DOC]
    # HELPER: called after each mutation through the wrapper.
    # Invalidates cached attributes and, unless dirty, stores a shallow copy of the content as the
    # new baseline snapshot. Returns nothing.
    """Acknowledges tracked changes to the wrapped dict."""
    self._attribute_sentinel.invalidate_all()
    if self._dirty:
      return
    self._self_last_wrapped_dict_snapshot = dict(self)

  def _track_value(self, value, name):
    # [FUNCTION DOC]
    # HELPER: tracking hook for each stored value. Inputs: value and key (used as the name). A
    # non-string key is replaced by a fixed placeholder name. Tries the parent tracking; if the
    # key is not a string and the value is tracked (and not a NoDependency marker), flags the
    # wrapper as having a non-string key, which blocks saving. If the value is not trackable,
    # falls back to only unwrapping NoDependency markers. Returns the value to store.
    """Allows storage of non-trackable objects."""
    if isinstance(name, str):
      string_key = True
    else:
      name = "-non_string_key"
      string_key = False
    try:
      no_dependency = isinstance(value, NoDependency)
      value = super()._track_value(value=value, name=name)
      if not (string_key or no_dependency):
        # A non-string key maps to a trackable value. This data structure
        # is not saveable.
        self._self_non_string_key = True
      return value
    except ValueError:
      # Even if this value isn't trackable, we need to make sure
      # NoDependency objects get unwrapped.
      return sticky_attribute_assignment(
          trackable=self, value=value, name=name)

  def _name_element(self, key):
    # [FUNCTION DOC]
    # HELPER: naming hook. Returns the key unchanged (no string validation, unlike Mapping).
    """Tells TrackableDataStructure to use keys as names as-is."""
    return key

  def __setitem__(self, key, value):
    # [FUNCTION DOC]
    # Assigns an entry. Inputs: key (any hashable) and value. Checks for external modification and
    # makes sure trackable bookkeeping is initialized. With a string key the value is tracked under
    # that name; otherwise the value is only wrapped/unwrapped, and if it is trackable (not
    # exempted by NoDependency) the wrapper is flagged as having a non-string key, which blocks
    # saving. Stores the result in the wrapped dict and refreshes the snapshot. Returns nothing.
    """Allow any modifications, but possibly mark the wrapper as unsaveable."""
    self._check_self_external_modification()
    self._maybe_initialize_trackable()
    no_dep = isinstance(value, NoDependency)
    if isinstance(key, str):
      value = self._track_value(value, name=key)
    else:
      value = wrap_or_unwrap(value)
      if not no_dep and isinstance(value, base.Trackable):
        # Non-string keys are OK as long as we have no reason to add a
        # dependency on the value (either because the value is not
        # trackable, or because it was wrapped in a NoDependency object).
        self._self_non_string_key = True
    self.__wrapped__[key] = value

    self._update_snapshot()

  def __delitem__(self, key):
    # [FUNCTION DOC]
    # Deletes an entry. Input: key. Checks for external modification, deletes from the wrapped dict
    # (key error if missing) and refreshes the snapshot. Does not flag the wrapper as unsaveable.
    self._check_self_external_modification()
    del self.__wrapped__[key]
    self._update_snapshot()

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text form with the class label and the representation of the wrapped dict.
    return "DictWrapper(%s)" % (repr(self.__wrapped__),)

  def __hash__(self):
    # [FUNCTION DOC]
    # Hashing is disabled: always raises a type error, like built-in dicts.
    raise TypeError("unhashable type: 'DictWrapper'")

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality by value: compares the wrapped dict with other, overriding the identity-based
    # equality of the base class. Returns a boolean.
    # Override the TrackableDataStructure "== -> is" forwarding and go back to
    # the wrapt implementation.
    return self.__wrapped__ == other

  def update(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Adds or replaces many entries. Arguments follow the dictionary constructor. Each pair goes
    # through item assignment so values are tracked and the snapshot refreshed per entry.
    # Returns nothing.
    for key, value in dict(*args, **kwargs).items():
      self[key] = value


class _TupleWrapper(TrackableDataStructure, wrapt.ObjectProxy):
  # [CLASS DOC]
  # Private proxy for tuples and namedtuples that contain trackable parts. Wraps (a possibly
  # re-built version of) the tuple, tracking elements by index, and for namedtuples also by field
  # name. If a namedtuple cannot be rebuilt from its fields, saving raises an error. Attribute
  # lookups prefer the wrapped tuple; equality and hashing follow the wrapped tuple.
  """Trackable wrapper for tuples and namedtuples."""

  def __init__(self, original_wrapped_tuple=()):
    # [FUNCTION DOC]
    # Constructor. Input: a tuple or namedtuple (default empty). Each element is passed through
    # wrap_or_unwrap (NoDependency markers are removed and remembered as not to be tracked). A
    # namedtuple is rebuilt from its fields; if its constructor rejects that (a type error), the
    # original tuple is wrapped as-is, a not-constructable flag is set and nothing is tracked.
    # Otherwise the wrapped tuple is rebuilt with its original type. After initializing the proxy
    # and trackable base, each element that should be tracked is registered by field name (for
    # namedtuples) and by index. Returns nothing.
    add_dependency = []
    substituted_wrapped_tuple = []
    for element in original_wrapped_tuple:
      if isinstance(element, NoDependency):
        add_dependency.append(False)
      else:
        add_dependency.append(True)
      substituted_wrapped_tuple.append(wrap_or_unwrap(element))
    try:
      fields = original_wrapped_tuple._fields
    except AttributeError:
      # Not a namedtuple
      is_namedtuple = False
    else:
      is_namedtuple = True
    original_type = type(original_wrapped_tuple)
    # Flag to poison saving if we can't re-construct a namedtupled because its
    # __new__ takes different keyword arguments than its _fields.
    self._self_tuple_is_constructable = True
    if is_namedtuple:
      try:
        # NamedTuples take N arguments, unlike tuple which takes a sequence.
        substituted_wrapped_tuple = original_type(
            **dict(zip(fields, substituted_wrapped_tuple)))
      except TypeError:
        wrapt.ObjectProxy.__init__(self, original_wrapped_tuple)
        TrackableDataStructure.__init__(self)
        self._self_tuple_is_constructable = False
        return
    else:
      substituted_wrapped_tuple = original_type(substituted_wrapped_tuple)
    wrapt.ObjectProxy.__init__(self, substituted_wrapped_tuple)
    TrackableDataStructure.__init__(self)

    if is_namedtuple:
      # For namedtuples, also track by names for compatibility with
      # dictionaries.
      for name, should_depend, element in zip(
          fields, add_dependency, substituted_wrapped_tuple):
        if should_depend:
          self._track_value(element, name=name)

    # Track by index as well, for compatibility with lists.
    for index, (should_depend, element) in enumerate(
        zip(add_dependency, substituted_wrapped_tuple)):
      if should_depend:
        self._track_value(element, name="%d" % (index,))

  @property
  def _values(self):
    # [FUNCTION DOC]
    # Property giving the base class its contained objects: the wrapper itself (iterating the
    # wrapped tuple).
    """Collect values for TrackableDataStructure."""
    return self

  def _track_value(self, value, name):
    # [FUNCTION DOC]
    # HELPER: tracking hook used per element. Inputs: value and dependency name. Tries the parent
    # tracking; on a value error (value not trackable) falls back to only unwrapping NoDependency
    # markers. Returns the value to store; never raises for non-trackable values.
    """Allows storage of non-trackable objects."""
    try:
      value = super()._track_value(value=value, name=name)
    except ValueError:
      # Even if this value isn't trackable, we need to make sure
      # NoDependency objects get unwrapped.
      value = sticky_attribute_assignment(
          trackable=self, value=value, name=name)
    return value

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text form with the class label and the representation of the wrapped tuple.
    return "_TupleWrapper(%s)" % (repr(self.__wrapped__),)

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash of the wrapped tuple, overriding the identity-based hash of the base class.
    # Override the TrackableDataStructure hash forwarding and go back to
    # the wrapt implementation.
    return hash(self.__wrapped__)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality by value: compares the wrapped tuple with other, overriding the identity-based
    # equality of the base class. Returns a boolean.
    # Override the TrackableDataStructure "== -> is" forwarding and go back to
    # the wrapt implementation.
    return self.__wrapped__ == other

  def __copy__(self):
    # [FUNCTION DOC]
    # Shallow copy: returns a new wrapper around a shallow copy of the wrapped tuple.
    return _TupleWrapper(copy.copy(self.__wrapped__))

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Deep copy. Input: memo mapping of copied objects. Returns a new wrapper around a deep copy of
    # the wrapped tuple.
    return _TupleWrapper(copy.deepcopy(self.__wrapped__, memo))

  @property
  def __dict__(self):
    # [FUNCTION DOC]
    # Property forwarding the instance-dictionary lookup to the wrapped object, so introspection
    # (notably in Python 3.12) gets an attribute error from objects like tuples that have no
    # dictionary instead of a type error from the proxy.
    # Python 3.12 inspect._check_instance() method only expects and handles
    # AttributeError but TypeError was raised when the method looks for
    # `__dict__` on the data structure proxy wrapper. Thus we overrides the
    # `__dict__` property and forwarding the `__dict__` lookup to the underlying
    # wrapped TrackalbeDataStructure. AttributeError will be raised when
    # TrackableDataStructure does not support `__dict__` and thus will be
    # handled properly.
    return self.__wrapped__.__dict__

  def __reduce_ex__(self, protocol):
    # [FUNCTION DOC]
    # Pickle support. Input: protocol number (unused). Returns a recipe that rebuilds the wrapper
    # from its class and the wrapped tuple.
    return (self.__class__,
            (self.__wrapped__,))

  # imul and iadd are the only tuple-relevant in-place operators. They need to
  # be special-cased to avoid mutating the original proxy object.
  def __imul__(self, y):
    # [FUNCTION DOC]
    # In-place repetition. Input: integer y. Returns a new plain tuple (the wrapped tuple repeated),
    # not the wrapper, so the proxy is not mutated.
    """Avoid running self.__wrapped__ *= y, which mutates `self`."""
    return self.__wrapped__ * y

  def __iadd__(self, y):
    # [FUNCTION DOC]
    # In-place concatenation. Input: tuple y. Returns a new plain tuple (wrapped tuple plus y),
    # not the wrapper, so the proxy is not mutated.
    """Avoid running self.__wrapped__ += y, which mutates `self`."""
    return self.__wrapped__ + y

  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    # [FUNCTION DOC]
    # Lists the dependencies to save. Inputs: save_type (checkpoint by default) and extra options
    # forwarded to the parent. Raises a value error if the wrapped namedtuple could not be rebuilt
    # from its fields (its constructor is overridden). Otherwise returns the parent's mapping of
    # names to children.
    if not self._self_tuple_is_constructable:
      raise ValueError(
          f"Unable to save because the namedtuple {self.__wrapped__} is not "
          "constructable from its _fields (i.e. __new__ is overridden). "
          f"Expected keyword arguments {self.__wrapped__._fields}. If you do "
          "not need to save this object, consider wrapping it in a custom "
          "object that does not inherit from tuple.")
    return super()._trackable_children(save_type, **kwargs)

  def __getattribute__(self, name):
    # [FUNCTION DOC]
    # Attribute lookup override. Input: attribute name. Any name other than the wrapped-object
    # slot that exists on the wrapped tuple is read from the wrapped tuple (it wins over the
    # wrapper's own). Names that are properties on this class bypass the proxy. Everything else
    # uses the normal proxy lookup. Raises an attribute error when not found.
    if name != "__wrapped__" and hasattr(self.__wrapped__, name):
      # Prefer attributes on the wrapped object when they conflict with
      # attributes on the wrapper object.
      return getattr(self.__wrapped__, name)

    if (hasattr(type(self), name)
        and isinstance(getattr(type(self), name), property)):
      # Bypass ObjectProxy for properties. Whether this workaround is necessary
      # appears to depend on the Python version but not the wrapt version: 3.4
      # in particular seems to look up properties on the wrapped object instead
      # of the wrapper without this logic.
      return object.__getattribute__(self, name)
    else:
      return super().__getattribute__(name)


def _is_function(x):
  # [FUNCTION DOC]
  # HELPER: used when saving as a saved-model to find function children.
  # Input: any object. Returns true if it is a polymorphic function or a concrete function from
  # the eager runtime. Imports the eager modules lazily to avoid circular imports.
  # pylint: disable=g-import-not-at-top
  from tensorflow.python.eager import def_function
  from tensorflow.python.eager import function as defun
  return isinstance(x, (def_function.Function, defun.ConcreteFunction))


def set_list_item(list_object, index_string, value):
  # [FUNCTION DOC]
  # HELPER: used when restoring checkpoint/saved-model dependencies into list-like objects.
  # Inputs: list_object (a growable list), index_string (decimal text of the position) and value.
  # Pads the list with null values if it is too short, then stores the value at that position.
  # Raises a value error if the index text is not an integer. Mutates the list; returns nothing.
  item_index = int(index_string)
  if len(list_object) <= item_index:
    list_object.extend([None] * (1 + item_index - len(list_object)))
  list_object[item_index] = value


def set_tuple_item(list_object, index_string, value):
  # [FUNCTION DOC]
  # HELPER: used when restoring dependencies into a list standing in for a tuple.
  # Inputs: list_object (growable list), index_string (decimal text of a position, or a namedtuple
  # field name) and value. A non-integer index text (namedtuple field) is silently ignored.
  # Otherwise pads the list with null values as needed and stores the value. Mutates the list;
  # returns nothing.
  try:
    item_index = int(index_string)
  except ValueError:
    # Ignore namedtuple fields.
    return
  if len(list_object) <= item_index:
    list_object.extend([None] * (1 + item_index - len(list_object)))
  list_object[item_index] = value
