# [MODULE DOC]
# HELPER: Helper module of identity-based containers used by Keras to track layers, variables
# and other objects (including unhashable ones) without relying on equality or hashing.
# Provides wrapper classes whose equality and hash are based on object identity, a reference
# class, a mutable mapping and a set keyed by identity, and weak-reference variants of the
# mapping and set that drop entries whose objects were garbage collected. This is a copy of the
# same utilities in the core util package and must be kept in sync with it. No import-time side
# effects.
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

import collections
from typing import Any, Set
import weakref


# LINT.IfChange
class _ObjectIdentityWrapper(object):
  # [CLASS DOC]
  # HELPER: Internal building block for the identity-based dictionary, set and reference classes.
  # Holds one object in a private slot. Two wrappers are equal only if they wrap the very same
  # object, ordering compares the wrapped objects' identity numbers, and the hash is the wrapped
  # object's identity number, so unhashable objects can be used as keys. Comparing a wrapper
  # with a non-wrapper raises a type error (except equality against nothing, which is false).
  """Wraps an object, mapping __eq__ on wrapper to "is" on wrapped.

  Since __eq__ is based on object identity, it's safe to also define __hash__
  based on object ids. This lets us add unhashable types like trackable
  _ListWrapper objects to object-identity collections.
  """

  __slots__ = ["_wrapped", "__weakref__"]

  def __init__(self, wrapped):
    # [FUNCTION DOC]
    # Stores the object to wrap (any object, hashable or not) in the private slot. Returns nothing.
    self._wrapped = wrapped

  @property
  def unwrapped(self):
    # [FUNCTION DOC]
    # Property returning the wrapped object itself (in the weak subclass this is the weak
    # reference, overridden there to dereference it).
    return self._wrapped

  def _assert_type(self, other):
    # [FUNCTION DOC]
    # HELPER: Type guard shared by the comparison methods.
    # Raises a type error if other is not an identity wrapper; otherwise returns nothing.
    if not isinstance(other, _ObjectIdentityWrapper):
      raise TypeError("Cannot compare wrapped object with unwrapped object")

  def __lt__(self, other):
    # [FUNCTION DOC]
    # Less-than based on the identity numbers of the wrapped objects. Raises a type error if other
    # is not a wrapper. Returns a boolean.
    self._assert_type(other)
    return id(self._wrapped) < id(other._wrapped)  # pylint: disable=protected-access

  def __gt__(self, other):
    # [FUNCTION DOC]
    # Greater-than based on the identity numbers of the wrapped objects. Raises a type error if
    # other is not a wrapper. Returns a boolean.
    self._assert_type(other)
    return id(self._wrapped) > id(other._wrapped)  # pylint: disable=protected-access

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality means both wrappers wrap the very same object. Comparing with nothing returns
    # false; comparing with any other non-wrapper raises a type error. Returns a boolean.
    if other is None:
      return False
    self._assert_type(other)
    return self._wrapped is other._wrapped  # pylint: disable=protected-access

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Inverse of equality, with the same type-error behavior for non-wrapper operands.
    return not self.__eq__(other)

  def __hash__(self):
    # [FUNCTION DOC]
    # Returns the identity number of the wrapped object, consistent with identity-based equality.
    # (The wrapped value may be a weak reference in the weak subclass, whose identity is stable.)
    # Wrapper id() is also fine for weakrefs. In fact, we rely on
    # id(weakref.ref(a)) == id(weakref.ref(a)) and weakref.ref(a) is
    # weakref.ref(a) in _WeakObjectIdentityWrapper.
    return id(self._wrapped)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a debug string naming the concrete wrapper class and the representation of the
    # wrapped object.
    return "<{} wrapping {!r}>".format(type(self).__name__, self._wrapped)


class _WeakObjectIdentityWrapper(_ObjectIdentityWrapper):
  # [CLASS DOC]
  # HELPER: Internal building block for the weak-key dictionary and weak set.
  # Identity wrapper that stores only a weak reference to the object, so it does not keep the
  # object alive. Once the object is collected, unwrapped yields nothing/null.

  __slots__ = ()

  def __init__(self, wrapped):
    # [FUNCTION DOC]
    # Wraps a weak reference to the given object (which must support weak references; otherwise
    # a type error is raised by the weak reference creation). Returns nothing.
    super(_WeakObjectIdentityWrapper, self).__init__(weakref.ref(wrapped))

  @property
  def unwrapped(self):
    # [FUNCTION DOC]
    # Property that dereferences the weak reference: returns the live object, or nothing/null
    # if it has been garbage collected.
    return self._wrapped()


class Reference(_ObjectIdentityWrapper):
  # [CLASS DOC]
  # Public identity-based reference to an object: two references are equal only when they refer to
  # the same object, so it can be used as a dictionary key or set member even for unhashable
  # objects. Disables the inherited unwrapped property; use deref to get the object back.
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
    # Returns the object this reference was created with. Takes no inputs, no side effects.
    """Returns the referenced object.

    ```python
    x_ref = Reference(x)
    print(x is x_ref.deref())
    ==> True
    ```
    """
    return self._wrapped


class ObjectIdentityDictionary(collections.abc.MutableMapping):
  # [CLASS DOC]
  # Mutable mapping whose keys are compared by identity rather than equality or hash, so
  # unhashable objects (such as tracked list wrappers) can be keys. Internally keeps a plain
  # dictionary from identity wrappers to values (private storage slot). Iteration yields the
  # original key objects. Created empty with no arguments.
  """A mutable mapping data structure which compares using "is".

  This is necessary because we have trackable objects (_ListWrapper) which
  have behavior identical to built-in Python lists (including being unhashable
  and comparing based on the equality of their contents by default).
  """

  __slots__ = ["_storage"]

  def __init__(self):
    # [FUNCTION DOC]
    # Creates an empty mapping (no arguments, unlike a normal dictionary). Returns nothing.
    self._storage = {}

  def _wrap_key(self, key):
    # [FUNCTION DOC]
    # HELPER: Key-wrapping hook for the mapping operations; subclasses override it to change the
    # wrapper kind. Returns a strong identity wrapper around the key.
    return _ObjectIdentityWrapper(key)

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Looks up the value stored for the very same key object. Raises a key error if absent.
    return self._storage[self._wrap_key(key)]

  def __setitem__(self, key, value):
    # [FUNCTION DOC]
    # Stores value under the identity of key, replacing any previous value. Mutates the mapping.
    self._storage[self._wrap_key(key)] = value

  def __delitem__(self, key):
    # [FUNCTION DOC]
    # Removes the entry for the very same key object. Raises a key error if absent.
    del self._storage[self._wrap_key(key)]

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of stored entries.
    return len(self._storage)

  def __iter__(self):
    # [FUNCTION DOC]
    # Generator yielding the original key objects (unwrapped), in insertion order.
    for key in self._storage:
      yield key.unwrapped

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a debug string showing the class name and the internal wrapped-key storage.
    return "ObjectIdentityDictionary(%s)" % repr(self._storage)


class ObjectIdentityWeakKeyDictionary(ObjectIdentityDictionary):
  # [CLASS DOC]
  # Identity-keyed mapping that holds keys weakly, like the standard weak key dictionary but
  # comparing keys with identity. Keys must support weak references. Entries are meant to vanish
  # when their key object is collected. Subclass of the strong identity dictionary overriding the
  # key wrapping, length and iteration.
  """Like weakref.WeakKeyDictionary, but compares objects with "is"."""

  __slots__ = ["__weakref__"]

  def _wrap_key(self, key):
    # [FUNCTION DOC]
    # HELPER: Key-wrapping hook; returns a weak identity wrapper so keys are not kept alive.
    return _WeakObjectIdentityWrapper(key)

  def __len__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The comment claims old weak refs are discarded, but this only lists the stored
    # wrapper keys; it does not iterate this mapping, so dead entries are still counted.
    # Returns the number of stored entries.
    # Iterate, discarding old weak refs
    return len(list(self._storage))

  def __iter__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Dead entries are removed by deleting with the wrapper itself, which gets
    # wrapped again (a weak reference to a wrapper), so the stored entry is not matched; it
    # likely raises a key error, and it also mutates storage while iterating its key view.
    # Generator yielding live key objects and attempting to drop entries whose key was collected.
    keys = self._storage.keys()
    for key in keys:
      unwrapped = key.unwrapped
      if unwrapped is None:
        del self[key]
      else:
        yield unwrapped


class ObjectIdentitySet(collections.abc.MutableSet):
  # [CLASS DOC]
  # Mutable set whose members are compared by identity instead of equality, so unhashable objects
  # can be members. Internally holds a plain set of identity wrappers (private storage slot).
  # Created from an optional iterable of objects; iteration yields the original objects.
  """Like the built-in set, but compares objects with "is"."""

  __slots__ = ["_storage", "__weakref__"]

  def __init__(self, *args):
    # [FUNCTION DOC]
    # Creates the set, optionally from a single iterable of objects (positional argument; passing
    # more than one argument raises a type error from list conversion). Each object is wrapped by
    # identity. Returns nothing.
    self._storage = set(self._wrap_key(obj) for obj in list(*args))

  def __le__(self, other: Set[Any]) -> bool:
    # [FUNCTION DOC]
    # KNOWN ISSUE: Membership is tested with the stored wrapper objects against other, not the
    # unwrapped items, so subset results are only right when other is itself an identity-wrapper
    # based collection; for ordinary sets this tends to return false.
    # Subset test. Returns the not-implemented marker if other is not a set type, false if this
    # set is larger than other, else true when every item is found in other.
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
    # Superset test. Returns the not-implemented marker if other is not a set type, false if this
    # set is smaller than other, else true when every item of other is a member (by identity).
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
    # HELPER: Static factory used by difference. Builds an empty identity set and installs the
    # given set of already-wrapped items as its storage (no copy). Returns the new set.
    result = ObjectIdentitySet()
    result._storage = storage  # pylint: disable=protected-access
    return result

  def _wrap_key(self, key):
    # [FUNCTION DOC]
    # HELPER: Wrapping hook; returns a strong identity wrapper (overridden by the weak set).
    return _ObjectIdentityWrapper(key)

  def __contains__(self, key):
    # [FUNCTION DOC]
    # Membership test by identity. Returns a boolean.
    return self._wrap_key(key) in self._storage

  def discard(self, key):
    # [FUNCTION DOC]
    # Removes the object if present; does nothing otherwise. Mutates the set.
    self._storage.discard(self._wrap_key(key))

  def add(self, key):
    # [FUNCTION DOC]
    # Adds the object (by identity). Mutates the set; returns nothing.
    self._storage.add(self._wrap_key(key))

  def update(self, items):
    # [FUNCTION DOC]
    # Adds every object of the given iterable. Mutates the set; returns nothing.
    self._storage.update([self._wrap_key(item) for item in items])

  def clear(self):
    # [FUNCTION DOC]
    # Removes all members. Mutates the set; returns nothing.
    self._storage.clear()

  def intersection(self, items):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Returns a plain set of internal identity wrappers rather than an identity set
    # of the original objects, unlike difference; callers get wrappers, not the objects.
    # Takes an iterable of objects and returns the members also present in it.
    return self._storage.intersection([self._wrap_key(item) for item in items])

  def difference(self, items):
    # [FUNCTION DOC]
    # Takes an iterable of objects and returns a new identity set holding the members not in it.
    # Does not mutate this set.
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
  # Identity-compared set that holds its members weakly, like the standard weak set. Members
  # must support weak references; dead members are meant to be dropped lazily when the set is
  # iterated or measured. Subclass of the strong identity set overriding wrapping, length, iteration.
  """Like weakref.WeakSet, but compares objects with "is"."""

  __slots__ = ()

  def _wrap_key(self, key):
    # [FUNCTION DOC]
    # HELPER: Wrapping hook; returns a weak identity wrapper so members are not kept alive.
    return _WeakObjectIdentityWrapper(key)

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of live members, found by iterating (which also tries to discard dead
    # entries).
    # Iterate, discarding old weak refs
    return len([_ for _ in self])

  def __iter__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Dead entries are discarded by passing the stored wrapper to discard, which
    # wraps it again (a weak reference to the wrapper), so the dead entry is not actually removed.
    # Generator yielding live member objects from a snapshot of the storage.
    keys = list(self._storage)
    for key in keys:
      unwrapped = key.unwrapped
      if unwrapped is None:
        self.discard(key)
      else:
        yield unwrapped
# LINT.ThenChange(//tensorflow/python/util/object_identity.py)
