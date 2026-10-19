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
# [MODULE DOC]
# Bookkeeping for values captured by a traced function graph. Exposes an ordered dictionary
# that tracks whether it has been modified, and a container class that records captures made
# by value (tensors copied into placeholders or constants) and by reference (callbacks that
# re-read a value on each call), together with their trace types. Used by the function graph
# machinery. No import-time side effects apart from a module constant giving the maximum
# element count for which small eager tensors are captured as constants.
# ==============================================================================
"""FuncGraph and related functionality."""

import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional

from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity


_EAGER_CONST_THRESHOLD = 128


class MutationAwareDict(py_collections.OrderedDict):
  # [CLASS DOC]
  # Ordered dictionary that keeps a boolean flag saying whether it has been changed since the
  # flag was last cleared. Mutating operations (set, delete, pop, clear) set the flag. Used by
  # the capture container to know when its cached derived data must be recomputed. The flag
  # starts true on creation. Note that other mutators (update, setdefault, popitem) are not
  # overridden and do not set the flag.
  """A dict with a mutation flag."""

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructs the ordered dictionary from any positional and keyword arguments accepted by
    # an ordered dictionary, and marks it as mutated.
    super().__init__(*args, **kwargs)
    self._mutated = True

  def pop(self, key, default=None):
    # [FUNCTION DOC]
    # Removes the entry for key and returns its value, or returns default (null by default) if
    # the key is absent; never raises for a missing key. Always marks the dictionary mutated.
    self._mutated = True
    return super().pop(key, default)

  def __setitem__(self, key, value):
    # [FUNCTION DOC]
    # Stores value under key and marks the dictionary mutated. Returns whatever the parent
    # implementation returns (nothing).
    self._mutated = True
    return super().__setitem__(key, value)

  def __delitem__(self, key):
    # [FUNCTION DOC]
    # Deletes the entry for key and marks the dictionary mutated. Raises a missing-key error if
    # the key is absent (the flag is already set by then).
    self._mutated = True
    return super().__delitem__(key)

  def clear(self):
    # [FUNCTION DOC]
    # Removes all entries and marks the dictionary mutated.
    self._mutated = True
    return super().clear()

  @property
  def mutated(self):
    # [FUNCTION DOC]
    # Property getter: returns true if the dictionary was modified since the flag was cleared.
    return self._mutated

  @mutated.setter
  def mutated(self, value):
    # [FUNCTION DOC]
    # Property setter: sets the mutation flag to the given boolean value (used to clear it
    # after cached data is rebuilt). Returns nothing.
    self._mutated = value


class FunctionCaptures(object):
  # [CLASS DOC]
  # Holds every capture of a function graph. By-value captures map a key (the identity of the
  # external tensor) to the external value, the internal placeholder or constant, and a trace
  # type. By-reference captures map a key to a callback returning the current external value,
  # the internal placeholder, and a trace type. Also holds a set of external operations the graph
  # has control dependencies on (public attribute control) and lazily cached views (list of
  # external/internal pairs, and combined capture types) rebuilt when the by-value dictionaries
  # report mutation. Created with no arguments by the graph that owns it.
  """A container for all capture usages within FuncGraph."""

  def __init__(self):
    # [FUNCTION DOC]
    # Creates empty by-reference and by-value dictionaries for external values, internal
    # placeholders and trace types (the by-value external/internal ones track mutation), an
    # empty identity-based control-dependency set, and empty caches. Takes no inputs.
    self._by_ref_internal = py_collections.OrderedDict()
    self._by_ref_external = py_collections.OrderedDict()
    self._by_ref_tracetype = py_collections.OrderedDict()
    self._by_val_internal = MutationAwareDict()
    self._by_val_external = MutationAwareDict()
    self._by_val_tracetype = py_collections.OrderedDict()

    # Set of external ops on which the graph has a control dependency
    self.control = object_identity.ObjectIdentitySet()

    # Cached properties derived from the above.
    self._cached_by_val_capture_tuples = []
    self._cached_capture_types = py_collections.OrderedDict()

  def clear(self):
    # [FUNCTION DOC]
    # Empties all by-reference and by-value capture dictionaries (external, internal and trace
    # type). Does not clear the control-dependency set or the cached derived views, though the
    # by-value dictionaries become flagged as mutated so caches are rebuilt on next access.
    self._by_ref_internal.clear()
    self._by_ref_external.clear()
    self._by_ref_tracetype.clear()
    self._by_val_internal.clear()
    self._by_val_external.clear()

  def capture_by_value(
      self, graph: Any, tensor: core.Tensor, name: Optional[str] = None
  ) -> core.Tensor:
    # [FUNCTION DOC]
    # Captures a tensor into a function graph by value and returns the tensor to use inside it.
    # Inputs: graph (the function graph doing the capturing), tensor (eager value, resource, or
    # graph tensor), name (optional; used for the placeholder; when absent an eager tensor gets a
    # unique integer string and a graph tensor uses its operation name).
    # For eager values: small tensors (element count at most the module threshold, with a
    # value dtype) are captured as constants, reusing an existing capture of the same tensor
    # identity if present; if the tensor cannot become a constant a placeholder is made instead.
    # Newly created constants are recorded as captures and appended to the graph inputs, and the
    # tensor is recorded on the gradient tape. Larger eager tensors and resources get
    # placeholders via the placeholder helper.
    # For graph tensors from another graph: validates the tensor is in scope and delegates to the
    # graph's own capture helper (which cond/while graphs override). A tensor already belonging
    # to the graph is returned unchanged.
    # Side effects: mutates the capture dictionaries and the graph inputs list. May raise an
    # inaccessible-tensor error from the scope validation.
    """Captures `tensor` if it's external to this graph.

    If `tensor` is from a different graph, returns a placeholder for it.
    `tensor` and the placeholder will appear in self.captures, and the
    placeholder will appear in self.inputs.  Multiple calls to this method with
    the same `tensor` argument will return the same placeholder. If `tensor` is
    from this graph, returns `tensor`.

    Args:
      graph: The FuncGraph that captures this tensor.
      tensor: Tensor. May be from this FuncGraph or a different graph.
      name: Optional name if a placeholder is created.

    Returns:
      Tensor from this FuncGraph.

    Raises:
      InaccessibleTensorError: if any tensors are accessed in a manner that
      bypasses the mechanisms required for the data dependencies to be correctly
      wired.
    """
    if isinstance(tensor, core.Value):
      if name is None:
        # A unique (within the program execution) integer.
        name = str(pywrap_tfe.TFE_Py_UID())

      # Small EagerTensors are captured with Const ops
      if (
          tensor.dtype in dtypes.TF_VALUE_DTYPES
          and functools.reduce(lambda a, b: a * b, tensor.shape, 1)
          <= _EAGER_CONST_THRESHOLD
      ):
        graph_const = self.by_val_internal.get(id(tensor))
        if graph_const is None:
          graph_const = tensor._capture_as_const(name)  # pylint: disable=protected-access
          if graph_const is None:
            # Some eager tensors, e.g. parallel tensors, are not convertible to
            # a single constant. We'll use a placeholder for this case.
            graph_const = self._create_placeholder_helper(graph, tensor, name)
          self.add_or_replace(
              key=id(tensor),
              external=tensor,
              internal=graph_const,
              is_by_ref=False,
          )
          graph.inputs.append(graph_const)
        graph_const._record_tape(tensor)  # pylint: disable=protected-access
        return graph_const

      # Large EagerTensors and resources are captured with Placeholder ops
      return self._create_placeholder_helper(graph, tensor, name)

    if tensor.graph is not graph:
      graph._validate_in_scope(tensor)  # pylint: disable=protected-access
      if name is None:
        assert tensor.op is not None, (
            tensor.__class__,
            dir(tensor),
            tensor.__class__.__name__,
        )
        name = tensor.op.name
      # cond/while graphs override _capture_helper() so cannot call
      # self.create_placeholder_helper() here directly.
      return graph._capture_helper(tensor, name)  # pylint: disable=protected-access
    return tensor

  def add_or_replace(
      self,
      key: Hashable,
      external: Any,
      internal: core.Tensor,
      tracetype: Any = None,
      is_by_ref: bool = False,
  ) -> None:
    # [FUNCTION DOC]
    # Records a capture, overwriting any existing capture with the same key. Inputs: key
    # (hashable identifier), external (the outer value, or for by-reference captures a callback
    # producing it), internal (the tensor used inside the graph), tracetype (optional type
    # description), is_by_ref (default false, selecting by-reference storage).
    # By-reference: stores external, internal and tracetype directly (tracetype may be null).
    # By-value: stores external and internal; if tracetype is null it is computed from the
    # external value. Returns nothing.
    """Replace a already exsiting capture, otherwise add it."""
    if is_by_ref:
      self._by_ref_external[key] = external
      self._by_ref_internal[key] = internal
      self._by_ref_tracetype[key] = tracetype
    else:
      self._by_val_internal[key] = internal
      self._by_val_external[key] = external
      if tracetype is not None:
        self._by_val_tracetype[key] = tracetype
      else:
        self._by_val_tracetype[key] = trace_type.from_value(external)

  def pop(self, key: Hashable, is_by_ref: bool = False) -> Any:
    # [FUNCTION DOC]
    # Removes the capture stored under key from the by-reference storage (when is_by_ref is
    # true) or by-value storage (default). Returns a triple of the removed external value,
    # internal tensor and trace type, each being null if it was missing. Never raises for an
    # absent key.
    if is_by_ref:
      return (
          self._by_ref_external.pop(key, None),
          self._by_ref_internal.pop(key, None),
          self._by_ref_tracetype.pop(key, None),
      )
    else:
      return (
          self._by_val_external.pop(key, None),
          self._by_val_internal.pop(key, None),
          self._by_val_tracetype.pop(key, None),
      )

  def reset_captures(self, tensors, placeholders):
    # [FUNCTION DOC]
    # Replaces all by-value captures. Inputs: tensors (list of external values) and placeholders
    # (list of internal tensors, paired by position; extra items in the longer list are ignored).
    # Creates fresh external, internal and trace-type dictionaries keyed by the identity of each
    # external value, computing trace types from the values. Returns nothing.
    # KNOWN ISSUE: the by-value trace-type dictionary is replaced with the mutation-tracking
    # type while add_or_replace and the constructor use a plain ordered dictionary; harmless but
    # inconsistent. By-reference captures are left untouched.
    """Set the captures with the provided list of captures & placeholder."""
    self._by_val_external = MutationAwareDict()
    self._by_val_internal = MutationAwareDict()
    self._by_val_tracetype = MutationAwareDict()
    for external, internal in zip(tensors, placeholders):
      key = id(external)
      self._by_val_external[key] = external
      self._by_val_internal[key] = internal
      self._by_val_tracetype[key] = trace_type.from_value(external)

  # TODO(panzf): make the method public after supporting lam() returns
  # non-tensor values. Currently, this method is only used by
  # FuncGraph._experimental_capture_side_input_by_ref(), which contains the
  # logics for converting non-tensor values to tensor.
  def _capture_by_ref(
      self, graph: Any, lam: Callable[[], Any], key: Hashable = None
  ) -> Any:
    # [FUNCTION DOC]
    # HELPER: internal support for the graph's experimental capture-side-input-by-reference.
    # Creates or retrieves a by-reference capture. Inputs: graph (owning function graph), lam
    # (callback with no arguments returning the value to capture), key (optional hashable id).
    # If key is given and already captured, returns the existing internal placeholder. If no key
    # is given, picks the first unused integer starting at the current capture count. Otherwise
    # calls the callback once, derives a trace type from the value, builds a placeholder for it in
    # the graph, and stores a wrapper callback (which re-invokes the callback and flattens the
    # result to tensors), the placeholder and the trace type. Returns the internal placeholder
    # (possibly a nested structure). Side effect: mutates the by-reference dictionaries.
    """Used during tracing process to create/retrive by-ref captures.

    Args:
      graph: The FuncGraph that captures this tensor.
      lam: A callable that takes no arguments and returns tensor captures.
      key: A hashable identifier.

    Returns:
      Tensor from this FuncGraph.
    """
    # Check if the capture exists in self._by_ref
    if key is not None and key in self._by_ref_internal:
      return self._by_ref_internal[key]
    if key is None:
      key = len(self._by_ref_internal)
      while key in self._by_ref_internal:
        key += 1

    value_nested = lam()
    capture_trace_type = trace_type.from_value(value_nested)
    ctx = trace_type.InternalPlaceholderContext(graph)
    internal = capture_trace_type.placeholder_value(ctx)

    def lam_fn():
      # [FUNCTION DOC]
      # HELPER: wrapper stored as the external side of a by-reference capture. Calls the user
      # callback again and converts its result to a flat list of tensors using the trace type
      # captured at creation. Takes no inputs; errors from the callback propagate.
      # pytype: disable=attribute-error
      value = lam()
      return capture_trace_type.to_tensors(value)
      # pytype: enable=attribute-error

    self._by_ref_external[key] = lam_fn
    self._by_ref_internal[key] = internal
    self._by_ref_tracetype[key] = capture_trace_type
    return self._by_ref_internal[key]

  def merge_by_ref_with(self, other: "FunctionCaptures") -> None:
    # [FUNCTION DOC]
    # Copies by-reference captures from another container into this one, for keys not already
    # present: the external callback and trace type are copied, but the internal placeholder is
    # not. Input: other, which must be a capture container (asserted). Returns nothing.
    """Add by-ref captures from `other` to `self` if not exist."""
    assert isinstance(other, FunctionCaptures)
    for key in other.by_ref_external:
      if key not in self._by_ref_external:
        self._by_ref_external[key] = other.by_ref_external[key]
        self._by_ref_tracetype[key] = other.by_ref_tracetype[key]

  # TODO(panzf): Return structured values instead of flat tensors.
  def get_by_ref_snapshot(self) -> Mapping[Hashable, Any]:
    # [FUNCTION DOC]
    # Returns a new mapping from each by-reference capture key to its current value, obtained by
    # calling the stored callback. If the callback raises an attribute error or runtime error
    # (value unavailable at dispatch time), the stored trace type is used for that key instead.
    # Other exceptions propagate. Does not modify the container.
    """Get a snapshot of current values of by-ref captures."""
    snapshot = {}
    for key in self._by_ref_external:
      func = self._by_ref_external[key]
      try:
        value = func()
      except (AttributeError, RuntimeError):
        # b/269680071 In case of by-ref captures are unavailable at dispatch
        # time, use the predefined trace_type instead.
        value = self._by_ref_tracetype[key]
      snapshot[key] = value
    return snapshot

  def _create_placeholder_helper(
      self, graph: Any, tensor: core.Tensor, name: str
  ):
    # [FUNCTION DOC]
    # HELPER: creates (or reuses) the placeholder for a by-value capture of a tensor.
    # Inputs: graph (function graph), tensor (eager value, resource or graph tensor), name
    # (placeholder name). If a placeholder for this tensor identity already exists it is reused.
    # Otherwise builds a trace type for the tensor, sets its name, uses the tensor's device as a
    # composite device name when the tensor is a packed eager value, creates the placeholder
    # in the graph with no control dependencies, registers the capture and appends the
    # placeholder to the graph inputs. In both cases records the tensor on the gradient tape.
    # Returns the placeholder.
    """A helper function to create capture placeholder."""
    placeholder = self._by_val_internal.get(id(tensor))
    if placeholder is None:
      tracing_ctx = trace_type.InternalTracingContext()
      spec = trace_type.from_value(tensor, tracing_ctx)
      spec._name = name  # pylint: disable=protected-access
      if isinstance(tensor, core.Value) and tensor.is_packed:
        composite_device_name = tensor.device
      else:
        composite_device_name = None
      placeholder_ctx = trace_type.InternalPlaceholderContext(
          graph,
          with_none_control_dependencies=True,
          composite_device_name=composite_device_name,
      )
      placeholder = spec.placeholder_value(placeholder_ctx)
      self.add_or_replace(
          key=id(tensor), external=tensor, internal=placeholder, is_by_ref=False
      )
      graph.inputs.append(placeholder)
    placeholder._record_tape(tensor)  # pylint: disable=protected-access
    return placeholder

  def _recompute_cached_properties(self):
    # [FUNCTION DOC]
    # HELPER: rebuilds the cached views when by-value captures changed. Clears the mutation
    # flags, asserts the by-value external and internal dictionaries have equal size and keys,
    # rebuilds the list of (external, internal) pairs, and rebuilds the combined capture-type
    # mapping (by-value types followed by by-reference types). Returns nothing.
    """Regenerates cached properties if there have been mutations."""
    self._by_val_internal.mutated = False
    self._by_val_external.mutated = False
    assert len(self._by_val_internal) == len(self._by_val_external)
    self._cached_by_val_capture_tuples = []
    for key in self._by_val_internal:
      assert key in self._by_val_external
      internal = self._by_val_internal[key]
      external = self._by_val_external[key]
      self._cached_by_val_capture_tuples.append((external, internal))

    self._cached_capture_types = py_collections.OrderedDict(
        list(self._by_val_tracetype.items())
        + list(self._by_ref_tracetype.items())
    )

  @property
  def capture_types(self):
    # [FUNCTION DOC]
    # Property: ordered mapping from capture key to trace type, by-value entries first and then
    # by-reference ones. Refreshes the cache first if by-value captures changed.
    # KNOWN ISSUE: only by-value mutation triggers a refresh, so by-reference changes made after
    # the last refresh may not appear until a by-value mutation occurs.
    if self._by_val_internal.mutated or self._by_val_external.mutated:
      self._recompute_cached_properties()
    return self._cached_capture_types

  @property
  def by_val_capture_tuples(self):
    # [FUNCTION DOC]
    # Property: list of (external value, internal tensor) pairs for by-value captures in
    # insertion order, refreshed first if the by-value dictionaries were mutated.
    if self._by_val_internal.mutated or self._by_val_external.mutated:
      self._recompute_cached_properties()
    return self._cached_by_val_capture_tuples

  @property
  def by_ref_internal(self):
    # [FUNCTION DOC]
    # Property: the live dictionary from key to internal placeholder for by-reference captures.
    return self._by_ref_internal

  @property
  def by_ref_external(self):
    # [FUNCTION DOC]
    # Property: the live dictionary from key to value-producing callback for by-reference captures.
    return self._by_ref_external

  @property
  def by_ref_tracetype(self):
    # [FUNCTION DOC]
    # Property: the live dictionary from key to trace type for by-reference captures.
    return self._by_ref_tracetype

  @property
  def by_val_internal(self):
    # [FUNCTION DOC]
    # Property: the live mutation-tracking dictionary from key to internal tensor for by-value
    # captures.
    return self._by_val_internal

  @property
  def by_val_external(self):
    # [FUNCTION DOC]
    # Property: the live mutation-tracking dictionary from key to external value for by-value
    # captures.
    return self._by_val_external

  @property
  def by_val_tracetype(self):
    # [FUNCTION DOC]
    # Property: the live dictionary from key to trace type for by-value captures.
    return self._by_val_tracetype
