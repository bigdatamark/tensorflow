# [MODULE DOC]
# Defines the cache that stores traced functions keyed by their function type, grouped by an
# execution context. Exposes FunctionContext (a simple context key) and FunctionCache (add,
# look up with subtype dispatch, delete, generalize, clear and list functions). Relies on the
# type dispatch table for choosing the most specific matching function type. No import-time
# side effects.
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
"""Cache to manage functions based on their FunctionType."""

import collections
from typing import Any, NamedTuple, Optional

from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.function.polymorphism import type_dispatch


class FunctionContext(NamedTuple):
  # [CLASS DOC]
  # Immutable, hashable record describing the execution context of a function: an arbitrary
  # context value and an arbitrary scope type, both defaulting to nothing. Used as part of cache
  # keys so functions traced under different contexts are kept apart.
  """Contains information regarding tf.function execution context."""
  context: Any = None
  scope_type: Any = None


class FunctionCache:
  # [CLASS DOC]
  # Container mapping (context, function type) pairs to functions. Holds an insertion-ordered
  # primary mapping from the pair to the function, and a per-context type dispatch table used to
  # find the most specific stored function type that accepts a requested type. Created empty and
  # filled with add; queried with lookup.
  # KNOWN ISSUE: the slots list includes a garbage-collector attribute that is never assigned.
  """A container for managing functions."""

  __slots__ = ["_primary", "_dispatch_dict", "_garbage_collectors"]

  def __init__(self):
    # [FUNCTION DOC]
    # Creates an empty cache: an ordered primary mapping and an empty mapping from contexts to
    # type dispatch tables. Takes no arguments and returns nothing.
    # Maps (FunctionContext, FunctionType) to a function.
    self._primary = collections.OrderedDict()

    # Maps FunctionContext to a TypeDispatchTable containing FunctionTypes of
    # that particular context.
    self._dispatch_dict = {}

  def lookup(self, function_type: function_type_lib.FunctionType,
             context: Optional[FunctionContext] = None) -> Optional[Any]:
    # [FUNCTION DOC]
    # Finds the stored function whose type best matches the requested one. Inputs: function_type,
    # the type to look for; context, optional context (a default empty context if omitted).
    # If the context is known, the dispatch table for it picks the most specific compatible
    # stored type, and the function stored under it is returned. Returns nothing/null when the
    # context is unknown or no stored type matches. Does not modify the primary mapping.
    """Looks up a function based on the context and type."""
    context = context or FunctionContext()
    if context in self._dispatch_dict:
      dispatch_type = self._dispatch_dict[context].dispatch(function_type)
      if dispatch_type:
        return self._primary[(context, dispatch_type)]

    return None

  def delete(self, function_type: function_type_lib.FunctionType,
             context: Optional[FunctionContext] = None,
             ) -> bool:
    # [FUNCTION DOC]
    # Removes the function stored under exactly the given type and context. Inputs: function_type
    # (exact stored type) and optional context (default empty context). Returns false if no such
    # entry exists; otherwise removes it from the primary mapping and from the context's dispatch
    # table and returns true.
    """Deletes a function given the context and type."""
    context = context or FunctionContext()
    if (context, function_type) not in self._primary:
      return False

    del self._primary[(context, function_type)]
    self._dispatch_dict[context].delete(function_type)

    return True

  def add(self, fn: Any, context: Optional[FunctionContext] = None) -> None:
    # [FUNCTION DOC]
    # Stores a function keyed by its own function type attribute and the context. Inputs: fn, any
    # object exposing a function type; context, optional (default empty context). Replaces any
    # entry with the same key, creates a dispatch table for a new context, and registers the
    # function type as a dispatch target. Returns nothing.
    """Adds a new function using its function_type.

    Args:
      fn: The function to be added to the cache.
      context: A FunctionContext representing the current context.
    """
    context = context or FunctionContext()
    self._primary[(context, fn.function_type)] = fn
    if context not in self._dispatch_dict:
      self._dispatch_dict[context] = type_dispatch.TypeDispatchTable()

    self._dispatch_dict[context].add_target(fn.function_type)

  def generalize(
      self, context: FunctionContext,
      function_type: function_type_lib.FunctionType
  ) -> function_type_lib.FunctionType:
    # [FUNCTION DOC]
    # Tries to produce a more general function type from the given one using the types already
    # stored for the context. Inputs: context and function_type. If the context has a dispatch
    # table, returns the result of its generalization attempt; otherwise returns the input type
    # unchanged. Does not modify the cache.
    """Try to generalize a FunctionType within a FunctionContext."""
    if context in self._dispatch_dict:
      return self._dispatch_dict[context].try_generalizing_function_type(
          function_type)
    else:
      return function_type

  # TODO(b/205971333): Remove this function.
  def clear(self):
    # [FUNCTION DOC]
    # Empties the cache by clearing both the primary mapping and all dispatch tables. Takes no
    # arguments and returns nothing. Marked in a comment as slated for removal.
    """Removes all functions from the cache."""
    self._primary.clear()
    self._dispatch_dict.clear()

  def values(self):
    # [FUNCTION DOC]
    # Returns a new list of all stored functions, in insertion order. No side effects.
    """Returns a list of all functions held by this cache."""
    return list(self._primary.values())

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the number of functions stored in the cache.
    return len(self._primary)
