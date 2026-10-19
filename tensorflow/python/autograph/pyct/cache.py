# [MODULE DOC]
# HELPER: caching utilities used by the autograph code-conversion machinery.
# Defines a generic weak-key, two-level cache for transformed functions and two concrete
# variants: one keyed by code object and one keyed by the unbound function. No import-time
# side effects.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Caching utilities."""

import inspect
import weakref


# TODO(mdan): Add a garbage collection hook for cleaning up modules.
class _TransformedFnCache(object):
  # [CLASS DOC]
  # HELPER: abstract base for the concrete caches in this module.
  # Two-level cache: weakly referenced keys derived from an entity by the key hook (to be
  # overridden), each mapping to an ordinary mapping of strongly held subkeys to values.
  # State is a single weak-key mapping (slot restricted). Entries vanish when the key dies.
  """Generic hierarchical cache for transformed functions.

  The keys are soft references (i.e. they are discarded when the key is
  destroyed) created from the source function by `_get_key`. The subkeys are
  strong references and can be any value. Typically they identify different
  kinds of transformation.
  """

  __slots__ = ('_cache',)

  def __init__(self):
    # [FUNCTION DOC]
    # Creates an empty cache backed by a weak-key mapping; takes no arguments, returns nothing.
    self._cache = weakref.WeakKeyDictionary()

  def _get_key(self, entity):
    # [FUNCTION DOC]
    # HELPER: key derivation hook for the cache lookup methods of this class.
    # Abstract: maps an entity to its cache key. Subclasses must override; this base version
    # always raises a not-implemented error.
    raise NotImplementedError('subclasses must override')

  def has(self, entity, subkey):
    # [FUNCTION DOC]
    # Returns true if the entity has an entry containing the given subkey, else false. Does not
    # create an entry for unknown entities.
    key = self._get_key(entity)
    parent = self._cache.get(key, None)
    if parent is None:
      return False
    return subkey in parent

  def __getitem__(self, entity):
    # [FUNCTION DOC]
    # Returns the mutable mapping (subkey to value) for the given entity, creating and storing an
    # empty one if absent, so that item assignment on the result persists in the cache.
    # Side effect: lookup of a missing entity inserts a new empty entry (which counts in length).
    key = self._get_key(entity)
    parent = self._cache.get(key, None)
    if parent is None:
      # The bucket is initialized to support this usage:
      #   cache[key][subkey] = value
      self._cache[key] = parent = {}
    return parent

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of live top-level keys (entities) currently held in the cache.
    return len(self._cache)


class CodeObjectCache(_TransformedFnCache):
  # [CLASS DOC]
  # Transformed-function cache keyed by the function's code object (falling back to the entity
  # itself when it has none). Functions that share code, such as closures created in a loop or
  # bound methods, share one entry. Keys are weakly referenced.
  """A function cache based on code objects.

  Code objects are good proxies for the source code of a function.

  This cache efficiently handles functions that share code objects, such as
  functions defined in a loop, bound methods, etc.

  The cache falls back to the function object, if it doesn't have a code object.
  """

  def _get_key(self, entity):
    # [FUNCTION DOC]
    # Computes the cache key for an entity: its code object if it has one, otherwise the entity
    # itself. Functions sharing a code object therefore share a cache entry.
    if hasattr(entity, '__code__'):
      return entity.__code__
    else:
      return entity


class UnboundInstanceCache(_TransformedFnCache):
  # [CLASS DOC]
  # Transformed-function cache keyed by the unbound function object. Bound methods are mapped
  # to their underlying function, so all instances of a class share one entry, while distinct
  # functions with identical code stay separate. Keys are weakly referenced.
  """A function cache based on unbound function objects.

  Using the function for the cache key allows efficient handling of object
  methods.

  Unlike the _CodeObjectCache, this discriminates between different functions
  even if they have the same code. This is needed for decorators that may
  masquerade as another function.
  """

  def _get_key(self, entity):
    # [FUNCTION DOC]
    # Computes the cache key for an entity: for a bound method returns the underlying plain
    # function (so all instances share one entry); otherwise returns the entity itself.
    if inspect.ismethod(entity):
      return entity.__func__
    return entity


