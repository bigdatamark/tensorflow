# [MODULE DOC]
# HELPER: identity-based collections used by TensorFlow tracking, Keras and graph code.
# Provides wrapper objects whose equality and hash follow object identity (the "is" relation)
# instead of value equality, so unhashable or value-compared objects (for example trackable list
# wrappers) can be stored. Exposes Reference, ObjectIdentityDictionary, ObjectIdentityWeakKeyDictionary,
# ObjectIdentitySet and ObjectIdentityWeakSet (weak variants hold weak references to their members).
# The code is mirrored by a copy in the Keras utils package (see the lint markers). No import-time
# side effects.
"""Utilities for collecting objects based on "is" comparison."""
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
from typing import Any, Set
import weakref

from tensorflow.python.util.compat import collections_abc


# LINT.IfChange
class _ObjectIdentityWrapper:
  # [CLASS DOC]
  # HELPER: base building block for the identity-based collections in this module.
  # Holds one object in a private slot. Equality means "same wrapped object", ordering compares
  # the identity numbers, and the hash is the identity number, so instances work as keys of
  # dictionaries and sets. Comparison with a non-wrapper object (other than none) raises a type
  # error. Supports weak references itself.
  """Wraps an object, mapping __eq__ on wrapper to "is" on wrapped.

  Since __eq__ is based on object identity, it's safe to also define __hash__
  based on object ids. This lets us add unhashable types like trackable
  _ListWrapper objects to object-identity collections.
  """

  __slots__ = ["_wrapped", "__weakref__"]

  def __init__(self, wrapped):
    # [FUNCTION DOC]
    # Stores the object to wrap. Input: wrapped (any object). Returns nothing.
    self._wrapped = wrapped

  @property
  def unwrapped(self):
    # [FUNCTION DOC]
    # Property returning the wrapped object (the stored value). No inputs, no side effects.
    return self._wrapped

  def _assert_type(self, other):
    # [FUNCTION DOC]
    # HELPER: guard for the comparison methods.
    # Input: other (any object). Returns nothing; raises a type error unless other is an identity
    # wrapper.
    if not isinstance(other, _ObjectIdentityWrapper):
      raise TypeError("Cannot compare wrapped object with unwrapped object")

  def __lt__(self, other):
    # [FUNCTION DOC]
    # Orders wrappers by the identity numbers of the wrapped objects (arbitrary but consistent).
    # Input: other (must be a wrapper, else type error). Returns a boolean.
    self._assert_type(other)
    return id(self._wrapped) < id(other._wrapped)  # pylint: disable=protected-access

  def __gt__(self, other):
    # [FUNCTION DOC]
    # Greater-than ordering by identity numbers of the wrapped objects.
    # Input: other (must be a wrapper, else type error). Returns a boolean.
    self._assert_type(other)
    return id(self._wrapped) > id(other._wrapped)  # pylint: disable=protected-access

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Identity equality of the wrapped objects.
    # Input: other. Returns false for none, true when both wrap the very same object, false
    # otherwise; raises a type error when other is neither none nor a wrapper.
    if other is None:
      return False
    self._assert_type(other)
    return self._wrapped is other._wrapped  # pylint: disable=protected-access

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Negation of the equality method; same input and exceptions. Returns a boolean.
    return not self.__eq__(other)

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash equal to the identity number of the wrapped object, consistent with identity equality.
    # For the weak subclass the wrapped object is the weak reference, which is cached per target.
    # Wrapper id() is also fine for weakrefs. In fact, we rely on
    # id(weakref.ref(a)) == id(weakref.ref(a)) and weakref.ref(a) is
    # weakref.ref(a) in _WeakObjectIdentityWrapper.
    return id(self._wrapped)

  def __repr__(self):
    # [FUNCTION DOC]
    # Debug text naming the wrapper class and the representation of the wrapped object.
    return "<{} wrapping {!r}>".format(type(self).__name__, self._wrapped)


class _WeakObjectIdentityWrapper(_ObjectIdentityWrapper):
  # [CLASS DOC]
  # HELPER: identity wrapper that holds only a weak reference to its object, used as key type of the
  # weak collections. The object can be garbage collected; unwrapping then yields none.

  __slots__ = ()

  def __init__(self, wrapped):
    # [FUNCTION DOC]
    # Wraps a weak reference to the given object. Input: wrapped (must be weak-referenceable, else
    # a type error is raised). Returns nothing.
    super(_WeakObjectIdentityWrapper, self).__init__(weakref.ref(wrapped))

  @property
  def unwrapped(self):
    # [FUNCTION DOC]
    # Property returning the referenced object, or none if it has been garbage collected.
    return self._wrapped()


class Reference(_ObjectIdentityWrapper):
  # [CLASS DOC]
  # Public identity wrapper: two references are equal only if they refer to the same object.
  # Disables the inherited unwrapped property and exposes the object through deref instead.
  """Reference that refers an object.

  ```python
  x = [1]
  y = [1]

  x_ref1 = Reference(x)
  x_ref2 = Reference(x)
  y_ref2 = Reference(y)

  print(x_ref1 == x_ref2)
  ==> True

  print(x_ref1 == y)
  ==> False
  ```
  """

  __slots__ = ()

  # Disabling super class' unwrapped field.
  unwrapped = property()

  def deref(self):
    # [FUNCTION DOC]
    # Returns the referenced object. No inputs, no side effects.
    """Returns the referenced object.

    ```python
    x_ref = Reference(x)
    print(x is x_ref.deref())
    ==> True
    ```
    """
    return self._wrapped


class ObjectIdentityDictionary(collections_abc.MutableMapping):
  # [CLASS DOC]
  # Mutable mapping whose keys are compared by identity. Keys are wrapped before storage in an
  # internal dictionary (the only state); iteration yields the original key objects. Subclassed by
  # the weak-key variant, which overrides the key wrapping.
  """A mutable mapping data structure which compares using "is".

  This is necessary because we have trackable objects (_ListWrapper) which
  have behavior identical to built-in Python lists (including being unhashable
  and comparing based on the equality of their contents by default).
  """

  __slots__ = ["_storage"]

  def __init__(self):
    # [FUNCTION DOC]
    # Creates an empty dictionary. Takes no inputs; initializes the internal storage mapping.
    self._storage = {}

  def _wrap_key(self, key):
    # [FUNCTION DOC]
    # HELPER: key wrapping hook, overridden by the weak variant.
    # Input: key (any object). Returns a strong identity wrapper around it.
    return _ObjectIdentityWrapper(key)

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Looks up the value stored for the identical key object. Raises a key error if absent.
    return self._storage[self._wrap_key(key)]

  def __setitem__(self, key, value):
    # [FUNCTION DOC]
    # Stores a value under the key object (identity based), replacing any previous value.
    self._storage[self._wrap_key(key)] = value

  def __delitem__(self, key):
    # [FUNCTION DOC]
    # Removes the entry for the identical key object. Raises a key error if absent.
    del self._storage[self._wrap_key(key)]

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of stored entries.
    return len(self._storage)

  def __iter__(self):
    # [FUNCTION DOC]
    # Generator yielding the original (unwrapped) key objects in insertion order.
    for key in self._storage:
      yield key.unwrapped

  def __repr__(self):
    # [FUNCTION DOC]
    # Debug text showing the class name and the internal storage representation (wrapped keys).
    return "ObjectIdentityDictionary(%s)" % repr(self._storage)


class ObjectIdentityWeakKeyDictionary(ObjectIdentityDictionary):
  # [CLASS DOC]
  # Identity-keyed dictionary that holds its keys weakly, like a weak-key dictionary. Keys must be
  # weak-referenceable. Entries whose key was collected are meant to be dropped lazily on iteration.
  """Like weakref.WeakKeyDictionary, but compares objects with "is"."""

  __slots__ = ["__weakref__"]

  def _wrap_key(self, key):
    # [FUNCTION DOC]
    # HELPER: key wrapping hook. Input: key (weak-referenceable object). Returns a weak identity
    # wrapper; raises a type error if the key cannot be weakly referenced.
    return _WeakObjectIdentityWrapper(key)

  def __len__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment claims dead weak references are discarded, but the code only counts
    # the stored keys without removing any, so entries of collected keys are still counted.
    # Returns the number of stored entries.
    # Iterate, discarding old weak refs
    return len(list(self._storage))

  def __iter__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the cleanup path deletes by passing a wrapper to the item-deletion method, which
    # wraps it again, so the dead entry is not found (key error), and it also mutates the storage
    # while iterating its keys view (runtime error). Works only while no key has died.
    # Generator yielding live key objects; intends to drop entries whose key was collected.
    keys = self._storage.keys()
    for key in keys:
      unwrapped = key.unwrapped
      if unwrapped is None:
        del self[key]
      else:
        yield unwrapped


class ObjectIdentitySet(collections_abc.MutableSet):
  # [CLASS DOC]
  # Mutable set whose members are compared by identity. Members are wrapped before being stored in
  # an internal set (the only state); iteration yields the original objects. Subclassed by the
  # weak set, which overrides key wrapping.
  """Like the built-in set, but compares objects with "is"."""

  __slots__ = ["_storage", "__weakref__"]

  def __init__(self, *args):
    # [FUNCTION DOC]
    # Creates the set. Input: optionally one iterable of members (more than one positional
    # argument raises a type error from list conversion). Each member is wrapped and stored.
    self._storage = set(self._wrap_key(obj) for obj in list(*args))

  def __le__(self, other: Set[Any]) -> bool:
    # [FUNCTION DOC]
    # Subset test. Input: other (a set-like). Returns not-implemented if other is not a set type;
    # false if this set is larger; otherwise true only if every stored wrapper is contained in
    # other. KNOWN ISSUE: it tests wrappers, not the original objects, against other, so a plain
    # set of objects will not match; works reliably only against another identity set.
    if not isinstance(other, Set):
      return NotImplemented
    if len(self) > len(other):
      return False
    for item in self._storage:
      if item not in other:
        return False
    return True

  def __ge__(self, other: Set[Any]) -> bool:
    # [FUNCTION DOC]
    # Superset test. Input: other (a set-like). Returns not-implemented if other is not a set
    # type; false if this set is smaller; otherwise true only if every item of other is contained
    # in this set (by identity).
    if not isinstance(other, Set):
      return NotImplemented
    if len(self) < len(other):
      return False
    for item in other:
      if item not in self:
        return False
    return True

  @staticmethod
  def _from_storage(storage):
    # [FUNCTION DOC]
    # HELPER: builds an identity set directly from an already-wrapped storage set (used by the
    # difference method). Input: storage (set of wrappers). Returns a new identity set sharing that
    # storage object.
    result = ObjectIdentitySet()
    result._storage = storage  # pylint: disable=protected-access
    return result

  def _wrap_key(self, key):
    # [FUNCTION DOC]
    # HELPER: key wrapping hook, overridden by the weak set. Input: key. Returns a strong wrapper.
    return _ObjectIdentityWrapper(key)

  def __contains__(self, key):
    # [FUNCTION DOC]
    # Membership by identity. Input: key (any object). Returns a boolean.
    return self._wrap_key(key) in self._storage

  def discard(self, key):
    # [FUNCTION DOC]
    # Removes the identical object if present; silent if absent. Input: key. Returns nothing.
    self._storage.discard(self._wrap_key(key))

  def add(self, key):
    # [FUNCTION DOC]
    # Adds an object (by identity); no effect if already present. Input: key. Returns nothing.
    self._storage.add(self._wrap_key(key))

  def update(self, items):
    # [FUNCTION DOC]
    # Adds every object of an iterable. Input: items (iterable of objects). Returns nothing.
    self._storage.update([self._wrap_key(item) for item in items])

  def clear(self):
    # [FUNCTION DOC]
    # Removes all members. Takes no inputs; returns nothing.
    self._storage.clear()

  def intersection(self, items):
    # [FUNCTION DOC]
    # KNOWN ISSUE: returns a plain set of internal wrapper objects, not an identity set of the
    # original objects, unlike difference; callers get wrappers instead of their objects.
    # Input: items (iterable of objects). Returns the wrapper set of members also in items.
    return self._storage.intersection([self._wrap_key(item) for item in items])

  def difference(self, items):
    # [FUNCTION DOC]
    # Members not present in items. Input: items (iterable of objects). Returns a new identity set;
    # this set is not modified.
    return ObjectIdentitySet._from_storage(
        self._storage.difference([self._wrap_key(item) for item in items]))

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of members.
    return len(self._storage)

  def __iter__(self):
    # [FUNCTION DOC]
    # Generator yielding the original member objects, iterating over a snapshot of the storage.
    keys = list(self._storage)
    for key in keys:
      yield key.unwrapped


class ObjectIdentityWeakSet(ObjectIdentitySet):
  # [CLASS DOC]
  # Identity set that holds members weakly; members must be weak-referenceable. Dead members are
  # meant to be dropped lazily when the set is iterated or measured.
  """Like weakref.WeakSet, but compares objects with "is"."""

  __slots__ = ()

  def _wrap_key(self, key):
    # [FUNCTION DOC]
    # HELPER: key wrapping hook. Input: key (weak-referenceable). Returns a weak wrapper; raises a
    # type error for objects that cannot be weakly referenced.
    return _WeakObjectIdentityWrapper(key)

  def __len__(self):
    # [FUNCTION DOC]
    # Counts live members by iterating (which also tries to discard dead entries). Returns an int.
    # Iterate, discarding old weak refs
    return len([_ for _ in self])

  def __iter__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the cleanup passes an already-wrapped key to the discard method, which wraps it
    # again (and weakly references the wrapper), so dead entries are never actually removed.
    # Generator yielding live member objects from a snapshot of the storage.
    keys = list(self._storage)
    for key in keys:
      unwrapped = key.unwrapped
      if unwrapped is None:
        self.discard(key)
      else:
        yield unwrapped
# LINT.ThenChange(//tensorflow/python/keras/utils/object_identity.py)
