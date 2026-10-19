# [MODULE DOC]
# Polymorphic type dispatch. Provides TypeDispatchTable, which stores a set of target function
# types in insertion order and, for a request function type, selects the closest (most specific)
# stored target that the request is a supertype of, with a bounded LRU cache of lookups. It also
# offers a generalization heuristic used to reduce retracing. Defines a module constant limiting
# the dispatch cache size. No import-time side effects.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Polymorphic Type Dispatch."""

import collections
from typing import Optional, Iterable

from tensorflow.core.function.polymorphism import function_type

# The maximum number of dispatch lookups to cache.
_MAX_DISPATCH_CACHE = 1024


class TypeDispatchTable:
  # [CLASS DOC]
  # Ordered collection of target function types with a dispatch operation. State: an ordered
  # mapping used as an insertion-ordered set of targets, and an ordered mapping serving as an
  # LRU cache from request types to the chosen non-exact target. Created empty; targets are
  # added, deleted or cleared by callers (typically a function cache), and dispatch is queried.
  """Type dispatch table implementation.

  A type dispatch table is a list, L, of target types. Given a request type, R,
  the table selects a target type, T, according to the following dispatch rules:
    1. R == T or R is supertype of T (functions are contravariant on args)
    2. There does not exist O in L such that R is supertype of O and O is a
       supertype of T (in other words, T is the closest to R, within list L).
    3. If the above two rules are satisfied by multiple targets, the earliest
       inserted one is chosen.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no arguments; initializes the empty target table and the empty dispatch
    # cache. Returns nothing.
    """Creates a TypeDispatchTable object."""
    # Holds all inserted types as keys mapping to None.
    # (Using OrderedDict as a set for determinism)
    self._dispatch_table = collections.OrderedDict()

    # LRU cache for dispatch results.
    # Maps request types to target types (see class description).
    # Does not contain exact matches, i.e, if cache[a] is b then a is not b.
    self._dispatch_cache = collections.OrderedDict()

  def add_target(self, target: function_type.FunctionType) -> None:
    # [FUNCTION DOC]
    # Inserts the given function type as a target (re-adding an existing one keeps its original
    # position). Then updates the cache: every cached request whose cached target is a subtype
    # of the new target is repointed to the new target. Returns nothing.
    """Adds a new target type."""
    self._dispatch_table[target] = None
    for request in self._dispatch_cache:
      if target.is_supertype_of(self._dispatch_cache[request]):
        self._dispatch_cache[request] = target

  @property
  def targets(self) -> Iterable[function_type.FunctionType]:
    # [FUNCTION DOC]
    # Read-only property returning a live view of all targets in insertion order.
    """Returns an iterable to all targets in the table."""
    return self._dispatch_table.keys()

  def delete(self, target: function_type.FunctionType) -> None:
    # [FUNCTION DOC]
    # Removes the target from the table if present (otherwise does nothing) and evicts every
    # cache entry that pointed to it. Returns nothing.
    """Deletes a target in the table if it exists."""
    if target in self._dispatch_table:
      del self._dispatch_table[target]
      for request in list(self._dispatch_cache.keys()):
        if self._dispatch_cache[request] == target:
          del self._dispatch_cache[request]

  # TODO(b/205971333): remove once FunctionCache 'clear' is removed.
  def clear(self) -> None:
    # [FUNCTION DOC]
    # Empties both the target table and the dispatch cache. Returns nothing.
    """Deletes all targets in the table."""
    self._dispatch_table.clear()
    self._dispatch_cache.clear()

  def dispatch(
      self, request: function_type.FunctionType
  ) -> Optional[function_type.FunctionType]:
    # [FUNCTION DOC]
    # Finds the target for a request function type. Exact matches return the request itself;
    # a cache hit moves the entry to the most-recent end and returns the cached target.
    # Otherwise scans targets in insertion order for those the request is a supertype of,
    # keeping the closest one (an earlier target wins ties), stores a non-empty result in the
    # cache and returns it. Returns nothing/null if no target matches (misses are not cached).
    # Note: despite the docstring wording, the chosen target is a subtype of the request.
    """Returns the most specific supertype target if it exists in the table."""
    # For known exact matches.
    if request in self._dispatch_table:
      return request

    # For known non-exact matches.
    # (self._dispatch cache does not contain exact matches)
    if request in self._dispatch_cache:
      # Move to the front of LRU cache.
      result = self._dispatch_cache.pop(request)
      self._dispatch_cache[request] = result
      return result

    most_specific_supertype = None
    for other in self._dispatch_table:
      if request.is_supertype_of(other):
        if most_specific_supertype is None or other.is_supertype_of(
            most_specific_supertype):
          most_specific_supertype = other

    self._cache_dispatch(request, most_specific_supertype)
    return most_specific_supertype

  def _cache_dispatch(self, request, target):
    # [FUNCTION DOC]
    # HELPER: records dispatch results for the dispatch method.
    # If the target is not null, evicts the oldest cache entry when the cache already exceeds
    # the maximum size, then stores the request-to-target mapping. Does nothing for a null
    # target. Returns nothing.
    """Caches the dispatch lookup result for a target."""
    if target is not None:
      # LRU Cache removes oldest item
      if len(self._dispatch_cache) > _MAX_DISPATCH_CACHE:
        self._dispatch_cache.popitem(last=False)
      self._dispatch_cache[request] = target

  def try_generalizing_function_type(
      self, target: function_type.FunctionType) -> function_type.FunctionType:
    # [FUNCTION DOC]
    # Starting from the given function type, iterates over all targets and, for each, replaces
    # the current candidate with its most specific common subtype with that target whenever one
    # exists. Returns the final candidate, which is the input itself if nothing generalizes.
    # Does not modify the table.
    """Returns a generalized subtype of the one given.

    This heuristic aims to reduce the number of future traces by computing a
    type that represents more general function inputs.

    The original "experimental_relax_shapes" heuristic identified a known type
    which shared a common subtype with the current unknown type and then
    traced with that common subtype. However, the notion of "common subtype"
    was only limited to shapes. This heuristic extends that to FunctionType.

    Returns `target` if a generalized subtype can not be found.

    Args:
      target: The FunctionType to generalize
    """
    relaxed = target
    for other in self._dispatch_table:
      subtype = relaxed.most_specific_common_subtype([other])
      if subtype is not None:
        relaxed = subtype
    return relaxed
