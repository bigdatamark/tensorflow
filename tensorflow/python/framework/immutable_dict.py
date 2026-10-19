# [MODULE DOC]
# Defines a read-only mapping type wrapping a regular dictionary. It is used internally by
# extension types and is otherwise a small standalone utility. No import-time side effects.

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
"""Immutable mapping."""

import collections.abc


# WARNING: this class is used internally by extension types (tf.ExtensionType),
# and may be deleted if/when extension types transition to a different encoding
# in the future.
class ImmutableDict(collections.abc.Mapping):
  # [CLASS DOC]
  # Read-only mapping built on the abstract mapping base class. Holds a private copy of the
  # supplied key/value pairs in an internal dictionary and exposes only lookup, membership,
  # iteration, length and representation; item assignment and deletion raise a type error
  # because the base class defines no mutators. Keys, values, items, get and equality come from
  # the base class. Contents are only shallowly immutable: mutable values can still change.
  # Marked as supported by the nest utilities to suppress a warning they would otherwise emit.
  """Immutable `Mapping`."""
  # Note: keys, items, values, get, __eq__, and __ne__ are implemented by
  # the `Mapping` base class.

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Accepts the same arguments as the built-in dictionary constructor (an
    # optional mapping or iterable of key/value pairs, and/or keyword entries) and stores a
    # fresh internal dictionary copy. Raises whatever the dictionary constructor raises for
    # invalid arguments (for example a type error or value error).
    self._dict = dict(*args, **kwargs)

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Returns the value stored for the given key. Raises a key error if the key is absent.
    return self._dict[key]

  def __contains__(self, key):
    # [FUNCTION DOC]
    # Membership test: returns True if the key is present in the internal dictionary, else False.
    # Direct lookup, faster than the base class fallback.
    return key in self._dict

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the keys, in the internal dictionary's insertion order.
    return iter(self._dict)

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of key/value entries held.
    return len(self._dict)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a string made of the class name wrapping the repr of the internal dictionary.
    return f'ImmutableDict({self._dict})'

  # This suppresses a warning that tf.nest would otherwise generate.
  __supported_by_tf_nest__ = True
