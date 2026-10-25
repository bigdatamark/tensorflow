# [MODULE DOC]
# HELPER: helper module shared by the trackable data structures and Keras-style layers.
# Provides duck-typing checks for layer-like objects (is_layer, has_weights), two small
# cache-state tracking classes (MutationSentinel and AttributeSentinel) used to invalidate cached
# attribute values when objects or their children change, and a generator that flattens nested
# layer containers (filter_empty_layer_containers). Lives here so it does not need to import the
# Keras package. No import-time side effects beyond imports.
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
"""Utilities related to layer/model functionality."""

# TODO(b/110718070): Move these functions back to tensorflow/python/keras/utils
# once __init__ files no longer require all of tf.keras to be imported together.

import collections
import weakref

from tensorflow.python.util import object_identity

try:
  # typing module is only used for comment type annotations.
  import typing  # pylint: disable=g-import-not-at-top, unused-import
except ImportError:
  pass


def is_layer(obj):
  # [FUNCTION DOC]
  # HELPER: duck-typing test used when collecting layers from containers.
  # Input: any object. Returns true if the object has the layer marker attribute and is not a
  # class itself (so layer classes are excluded, instances are accepted). Never raises.
  """Implicit check for Layer-like objects."""
  # TODO(b/110718070): Replace with isinstance(obj, base_layer.Layer).
  return hasattr(obj, "_is_layer") and not isinstance(obj, type)


def has_weights(obj):
  # [FUNCTION DOC]
  # HELPER: duck-typing test used when collecting layers from containers.
  # KNOWN ISSUE: the docstring is copied from is_layer; this function actually checks whether the
  # object's type defines both trainable and non-trainable weight attributes, not layer-ness.
  # Input: any object. Returns true if its type has both attributes and it is not a class itself.
  """Implicit check for Layer-like objects."""
  # TODO(b/110718070): Replace with isinstance(obj, base_layer.Layer).
  has_weight = (hasattr(type(obj), "trainable_weights")
                and hasattr(type(obj), "non_trainable_weights"))

  return has_weight and not isinstance(obj, type)


class MutationSentinel(object):
  # [CLASS DOC]
  # HELPER: building block of AttributeSentinel.
  # Holds a single boolean, whether one property is currently in a cached (valid) state. The flag
  # is a class-level default of false that instances override when marked.
  """Container for tracking whether a property is in a cached state."""
  _in_cached_state = False

  def mark_as(self, value):  # type: (MutationSentinel, bool) -> bool
    # [FUNCTION DOC]
    # Sets the cached-state flag. Input: boolean new state. Returns true if the new value differs
    # from the previous one (meaning upstream parents may be affected), otherwise false.
    may_affect_upstream = (value != self._in_cached_state)
    self._in_cached_state = value
    return may_affect_upstream

  @property
  def in_cached_state(self):
    # [FUNCTION DOC]
    # Property returning the current cached-state flag (true when the cached value is valid).
    return self._in_cached_state


class AttributeSentinel(object):
  # [CLASS DOC]
  # Tracks, per attribute name, whether a cached value is still valid, for a layer or container.
  # Holds a mapping from attribute names to mutation sentinels (created on demand), a weak set of
  # parent sentinels that must be told when this one changes, and a flag forcing propagation to
  # parents on every change. Invalidating a key notifies parents; invalidating everything also
  # clears all local keys and recursively all parents. Used by trackable data structures and
  # layers.
  """Container for managing attribute cache state within a Layer.

  The cache can be invalidated either on an individual basis (for instance when
  an attribute is mutated) or a layer-wide basis (such as when a new dependency
  is added).
  """

  def __init__(self, always_propagate=False):
    # [FUNCTION DOC]
    # Constructor. Input: always_propagate (boolean, default false); when true every change is
    # passed on to parents even if the local state did not change (used by pass-through
    # containers, which are always considered cached). Creates an empty weak set of parents and an
    # empty on-demand mapping of attribute names to mutation sentinels.
    self._parents = weakref.WeakSet()
    self.attributes = collections.defaultdict(MutationSentinel)

    # The trackable data structure containers are simple pass throughs. They
    # don't know or care about particular attributes. As a result, they will
    # consider themselves to be in a cached state, so it's up to the Layer
    # which contains them to terminate propagation.
    self.always_propagate = always_propagate

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a multi-line text form: the default object description, then a newline and a mapping
    # of each known attribute name to whether it is in a cached state.
    return "{}\n  {}".format(
        super(AttributeSentinel, self).__repr__(),
        {k: v.in_cached_state for k, v in self.attributes.items()})

  def add_parent(self, node):
    # type: (AttributeSentinel, AttributeSentinel) -> None
    # [FUNCTION DOC]
    # Registers another sentinel as a parent (held weakly) that must hear about changes here.
    # Input: the parent sentinel. Side effect: invalidates all of the parent's cached attributes
    # (and, recursively, its own parents), since it has gained a child. Removal is not tracked,
    # which is safe because the sentinels only drive cache invalidation. Returns nothing.

    # Properly tracking removal is quite challenging; however since this is only
    # used to invalidate a cache it's alright to be overly conservative. We need
    # to invalidate the cache of `node` (since it has implicitly gained a child)
    # but we don't need to invalidate self since attributes should not depend on
    # parent Layers.
    self._parents.add(node)
    node.invalidate_all()

  def get(self, key):
    # type: (AttributeSentinel, str) -> bool
    # [FUNCTION DOC]
    # Input: attribute name. Returns whether that attribute is currently in a cached state; an
    # unknown name is created on demand and reports false.
    return self.attributes[key].in_cached_state

  def _set(self, key, value):
    # type: (AttributeSentinel, str, bool) -> None
    # [FUNCTION DOC]
    # HELPER: shared implementation of mark_cached and invalidate.
    # Inputs: attribute name and new boolean state. Updates the local sentinel; if the state
    # changed, or always-propagate is set, tells every parent to invalidate the same name.
    # Returns nothing.
    may_affect_upstream = self.attributes[key].mark_as(value)
    if may_affect_upstream or self.always_propagate:
      for node in self._parents:  # type: AttributeSentinel
        node.invalidate(key)

  def mark_cached(self, key):
    # type: (AttributeSentinel, str) -> None
    # [FUNCTION DOC]
    # Marks the named attribute as holding a valid cached value. Input: attribute name.
    # Propagates to parents per the rules of the internal setter. Returns nothing.
    self._set(key, True)

  def invalidate(self, key):
    # type: (AttributeSentinel, str) -> None
    # [FUNCTION DOC]
    # Marks the named attribute's cached value as stale. Input: attribute name. Propagates to
    # parents per the rules of the internal setter. Returns nothing.
    self._set(key, False)

  def invalidate_all(self):
    # [FUNCTION DOC]
    # Marks every known attribute stale locally (without notifying parents per key) and then calls
    # invalidate-all on each parent, because parents may track different keys. Returns nothing.
    # Parents may have different keys than their children, so we locally
    # invalidate but use the `invalidate_all` method of parents.
    for key in self.attributes.keys():
      self.attributes[key].mark_as(False)

    for node in self._parents:
      node.invalidate_all()


def filter_empty_layer_containers(layer_list):
  # [FUNCTION DOC]
  # Generator that flattens nested layer containers. Input: a list of layer-like objects or
  # containers. Walks depth first, keeping the original order, visiting each object (by identity)
  # once. Yields every layer-like object; for anything else it follows its layers attribute (empty
  # if absent), so containers contribute only the layers they hold and empty containers vanish.
  # Does not modify the input list. Raises nothing itself.
  """Filter out empty Layer-like containers and uniquify."""
  # TODO(b/130381733): Make this an attribute in base_layer.Layer.
  existing = object_identity.ObjectIdentitySet()
  to_visit = layer_list[::-1]
  while to_visit:
    obj = to_visit.pop()
    if obj in existing:
      continue
    existing.add(obj)
    if is_layer(obj):
      yield obj
    else:
      sub_layers = getattr(obj, "layers", None) or []

      # Trackable data structures will not show up in ".layers" lists, but
      # the layers they contain will.
      to_visit.extend(sub_layers[::-1])
