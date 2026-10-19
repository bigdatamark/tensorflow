# [MODULE DOC]
# Builds TraceTypes (the cache-key types of function arguments) from arbitrary Python values.
# Exposes three context classes shared during tracing - a tracing context (aliasing of global
# ids and placeholder bookkeeping), a placeholder context (graph, naming scope, flags and
# alias-to-placeholder mapping) and a cast context (whether specs may be cast) - plus the main
# function that converts a value into a TraceType by dispatching on its kind: tracing protocol,
# wrapped objects, lists, tuples, named tuples, mappings, attrs instances, array-likes, custom
# nest protocol objects, weak-referenceable objects and finally plain literals. Relies on the
# default types, custom nest trace type and util modules. No import-time side effects.
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
"""Utitiles for Cache Key generation based on Function Trace Type."""

import collections.abc
from typing import Any, Dict, Hashable, Optional
import weakref

from tensorflow.core.function.trace_type import custom_nest_trace_type
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
from tensorflow.python.util import custom_nest_protocol


class InternalTracingContext(trace.TracingContext):
  # [CLASS DOC]
  # Default tracing context used while converting values into TraceTypes. Holds a mapping from
  # global ids to small local ids (assigned in first-seen order), a mapping from alias ids to
  # placeholders, and a flag telling whether the value comes from a legacy signature (such as an
  # input signature). Created by the builder function when no context is supplied.
  """Container for variables and flags shared across TraceType generation."""

  def __init__(self, is_legacy_signature: bool = False):
    # [FUNCTION DOC]
    # Takes an optional flag (default false) saying whether the traced values come from a legacy
    # signature representation. Starts with empty id and placeholder mappings.
    self._global_to_local_id = {}
    self._alias_id_to_placeholder = {}
    self._is_legacy_signature = is_legacy_signature

  def alias_global_id(self, global_id: Hashable) -> Hashable:
    # [FUNCTION DOC]
    # Maps a hashable global id to a local alias id: the first unseen id gets the next integer
    # (0, 1, 2, ...), repeated ids get the same alias. Returns the alias. Mutates the internal
    # mapping on first sight of an id.
    if global_id not in self._global_to_local_id:
      self._global_to_local_id[global_id] = len(self._global_to_local_id)

    return self._global_to_local_id[global_id]

  def add_placeholder(self, alias_id: Hashable, variable) -> None:
    # [FUNCTION DOC]
    # Records a placeholder (variable) under an alias id, silently replacing any earlier entry.
    # Returns nothing.
    self._alias_id_to_placeholder[alias_id] = variable

  def get_placeholder_mapping(self) -> Dict[Hashable, Any]:
    # [FUNCTION DOC]
    # Returns the live (not copied) mapping from alias ids to placeholders.
    return self._alias_id_to_placeholder

  @property
  def is_legacy_signature(self) -> bool:
    # [FUNCTION DOC]
    # Property returning whether the values come from a legacy signature representation.
    """If the value is from a legacy signature representation.

    Legacy signature representations include tf.function.input_signature and
    ConcreteFunction.structured_input_signature.
    """
    return self._is_legacy_signature


class InternalPlaceholderContext(trace.PlaceholderContext):
  # [CLASS DOC]
  # Default placeholder context passed to TraceTypes when they create placeholder values. Holds
  # the alias-to-placeholder mapping (so repeated aliases share one placeholder), the current
  # naming scope, the graph in which placeholders are created, and flags for unnest-only
  # behavior, none-control-dependency use and a composite device name. Mostly read-only
  # accessors, with methods to add placeholders and change the naming scope.
  """Container with mappings shared across TraceTypes for placeholder values."""

  def __init__(self,
               context_graph=None,
               placeholder_mapping=None,
               unnest_only=False,
               with_none_control_dependencies=False,
               composite_device_name=None):
    # [FUNCTION DOC]
    # All parameters are optional. context_graph: the graph to create placeholders in.
    # placeholder_mapping: an existing mapping from alias ids to placeholders (an empty one is
    # used if missing or empty; a supplied non-empty mapping is shared, not copied). unnest_only:
    # flag, default false. with_none_control_dependencies: flag, default false.
    # composite_device_name: optional device name. The naming scope starts unset.
    self._alias_id_to_placeholder = placeholder_mapping or {}
    self._naming_scope = None
    self._context_graph = context_graph
    self._unnest_only = unnest_only
    self._with_none_control_dependencies = with_none_control_dependencies
    self._composite_device_name = composite_device_name

  def has_placeholder(self, alias_id: Hashable) -> bool:
    # [FUNCTION DOC]
    # Returns true if a placeholder has been stored for the alias id.
    return alias_id in self._alias_id_to_placeholder

  def get_placeholder(self, alias_id: Hashable) -> Hashable:
    # [FUNCTION DOC]
    # Returns the placeholder stored for the alias id. Raises a key error if none is stored.
    if not self.has_placeholder(alias_id):
      raise KeyError(f"alias_id: {alias_id} not found in this instance of "
                     "placeholder context.")
    return self._alias_id_to_placeholder[alias_id]

  def add_placeholder(self, alias_id: Hashable, placeholder: Hashable) -> None:
    # [FUNCTION DOC]
    # Stores a placeholder under an alias id. Unlike the tracing context, raises a key error if
    # the alias id already has a placeholder. Returns nothing.
    if alias_id in self._alias_id_to_placeholder:
      raise KeyError(f"alias id: {alias_id} is already stored in this "
                     "instance of placeholder context.")
    self._alias_id_to_placeholder[alias_id] = placeholder

  def update_naming_scope(self, naming_scope: Optional[str]) -> None:
    # [FUNCTION DOC]
    # Sets the current naming scope (a string or nothing to clear it). Returns nothing.
    self._naming_scope = naming_scope

  @property
  def naming_scope(self) -> Optional[str]:
    # [FUNCTION DOC]
    # Property returning the current naming scope, or nothing if unset.
    return self._naming_scope

  @property
  def context_graph(self):
    # [FUNCTION DOC]
    # Property returning the graph in which placeholders are created (may be nothing).
    return self._context_graph

  @property
  def unnest_only(self) -> bool:
    # [FUNCTION DOC]
    # Property returning the unnest-only flag.
    return self._unnest_only

  @property
  def with_none_control_dependencies(self) -> bool:
    # [FUNCTION DOC]
    # Property returning whether placeholders are created with no control dependencies.
    return self._with_none_control_dependencies

  @property
  def composite_device_name(self) -> Any:
    # [FUNCTION DOC]
    # Property returning the composite device name, or nothing if unset.
    return self._composite_device_name


class InternalCastContext(trace.CastContext):
  # [CLASS DOC]
  # Default context passed to TraceType cast operations. Holds a single flag saying whether
  # TypeSpec objects may be cast in place of actual composite tensors.
  """Default casting behaviors."""

  def __init__(self, allow_specs=False):
    # [FUNCTION DOC]
    # Takes an optional flag (default false) allowing type specs to be cast, and stores it.
    self._allow_specs = allow_specs

  @property
  def allow_specs(self) -> bool:
    # [FUNCTION DOC]
    # Property returning the flag that allows type specs to be cast instead of actual composite
    # tensors (needed for public APIs that accept specs).
    """Allow TypeSpecs to be casted (instead of the actual CompositeTensors)."""
    # Public APIs like get_concrete_function allow users to pass in specs
    # instead which need to pass through input binding etc.
    return self._allow_specs


def from_value(value: Any,
               context: trace.TracingContext = None) -> trace.TraceType:
  # [FUNCTION DOC]
  # Main entry point: converts a Python value into a TraceType. Parameters: value (anything) and
  # an optional tracing context (a default one is created if missing). Dispatch order: if the
  # context is a legacy signature and the value is already a TraceType, return it as is; if the
  # value supports the tracing protocol, call it (raises a type error if the result is not a
  # TraceType); a weak reference input raises a type error; objects with a wrapped attribute are
  # unwrapped and converted recursively; lists, tuples, named tuples, mappings and attrs
  # instances become the matching container TraceType built recursively from their items;
  # array-likes become the registered tensor type with their shape and dtype; custom nest
  # protocol objects become a custom nest TraceType built from their flatten metadata and
  # components; other objects become a weak-reference TraceType, or if they cannot be weakly
  # referenced a Literal. Raises a type error if no generic TraceType can be produced (for
  # example an unhashable value). Returns the TraceType.
  """Returns a TraceType corresponding to the value based on the context.

  Args:
    value: The value to generate a TraceType for.
    context: The TracingContext to be shared during protocol calls.

  Returns:
    A TraceType object representing the given value.
  """

  if context is None:
    context = InternalTracingContext()

  if context.is_legacy_signature and isinstance(value, trace.TraceType):
    return value
  elif isinstance(value, trace.SupportsTracingProtocol):
    generated_type = value.__tf_tracing_type__(context)
    if not isinstance(generated_type, trace.TraceType):
      raise TypeError(
          "Expected an instance of TraceType for Tracing Protocol call to " +
          str(value) + " but got " + str(generated_type))
    return generated_type

  # TODO(b/183107079): Allow these once they're handled properly.
  if isinstance(value, weakref.ref):
    raise TypeError(
        f"weakref input {value} not supported for tf.function."
    )

  if hasattr(value, "__wrapped__"):
    return from_value(value.__wrapped__, context)

  if isinstance(value, list):
    return default_types.List(*(from_value(c, context) for c in value))

  if isinstance(value, tuple):
    if util.is_namedtuple(value):
      named_tuple_type = type(value)
      return default_types.NamedTuple.from_type_and_attributes(
          named_tuple_type, tuple(from_value(c, context) for c in value))
    else:
      return default_types.Tuple(*(from_value(c, context) for c in value))

  if isinstance(value, collections.abc.Mapping):
    mapping_type = type(value)
    return default_types.Dict(
        {k: from_value(value[k], context) for k in value}, mapping_type)

  if util.is_attrs(value):
    return default_types.Attrs.from_type_and_attributes(
        type(value),
        tuple(
            from_value(getattr(value, a.name), context)
            for a in value.__attrs_attrs__))

  if util.is_np_ndarray(value):
    ndarray = value.__array__()
    return default_types.TENSOR(ndarray.shape, ndarray.dtype)

  if isinstance(value, custom_nest_protocol.CustomNestProtocol):
    metadata, components = value.__tf_flatten__()
    return custom_nest_trace_type.CustomNestTraceType(
        type(value), metadata, tuple(from_value(c, context) for c in components)
    )

  try:
    ref = weakref.ref(value)
    if ref is None:
      raise TypeError(
          f"Deleted objects are not valid tf.function arguments, Got {value!r}")
    else:
      return default_types.Weakref(ref)
  except TypeError:
    try:
      return default_types.Literal(value)
    except:
      raise TypeError(  # pylint: disable=raise-missing-from
          f"Could not generate a generic TraceType for {value!r}."
          f"Please verify that it is immutable/hashable. Otherwise, consider "
          f"implementing the Tracing Protocol for it.")
