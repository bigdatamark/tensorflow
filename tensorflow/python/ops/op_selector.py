# [MODULE DOC]
# HELPER: internal graph-inspection utilities used by gradient, lifting and function-building code.
# Utilities for selecting and converting graph elements. It offers conversion of tensors, operations
# or whole graphs into lists of tensors or operations, lookup of generating and consuming ops,
# graph-uniqueness checks, a backward graph walk from seed ops, a path finder used for error
# messages, and a subgraph mapper used when lifting tensors out of a graph. Also defines the
# error type raised when a tensor cannot be lifted. No import-time side effects besides imports.
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
"""Tools for selecting ops in a graph."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity


def is_differentiable(op):
  # [FUNCTION DOC]
  # Reports whether the given operation has a registered gradient function.
  # Input: op, an operation whose definition name is looked up in the global gradient registry.
  # Output: true if a gradient entry is found and is not null; false if the registry lookup fails
  # with a lookup error. Note that ops registered as not differentiable may still be found by the
  # registry (with a null entry), in which case the result is false. No side effects.
  try:
    return ops._gradient_registry.lookup(op.op_def.name) is not None  # pylint: disable=protected-access
  except LookupError:
    return False


def is_iterable(obj):
  # [FUNCTION DOC]
  # Tells whether an object can be iterated over, treating tensors as not iterable.
  # Input: obj, any object. Output: false for tensors, false if asking for an iterator raises any
  # exception, true otherwise. Exceptions are swallowed. No side effects.
  """Return true if the object is iterable."""
  if isinstance(obj, tensor_lib.Tensor):
    return False
  try:
    _ = iter(obj)
  except Exception:  # pylint: disable=broad-except
    return False
  return True


def concatenate_unique(la, lb):
  # [FUNCTION DOC]
  # Appends to list la every element of lb not already in la, preserving lb order.
  # Inputs: la, a list of hashable objects (mutated in place); lb, an iterable of hashable objects.
  # Output: the same list la, now extended. Elements are compared by hash and equality; duplicates
  # inside lb are added only once. Raises a type error if elements are unhashable.
  """Add all the elements of `lb` to `la` if they are not there already.

  The elements added to `la` maintain ordering with respect to `lb`.

  Args:
    la: List of Python objects.
    lb: List of Python objects.
  Returns:
    `la`: The list `la` with missing elements from `lb`.
  """
  la_set = set(la)
  for l in lb:
    if l not in la_set:
      la.append(l)
      la_set.add(l)
  return la


def get_tensors(graph):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns tensors that are inputs or outputs of ops, but the
  # code only collects the output tensors of every operation (inputs are not added separately).
  # Input: graph, a graph object. Output: list of the outputs of all operations in graph order.
  # Raises a type error if the argument is not a graph. No side effects.
  """get all the tensors which are input or output of an op in the graph.

  Args:
    graph: a `tf.Graph`.
  Returns:
    A list of `tf.Tensor`.
  Raises:
    TypeError: if graph is not a `tf.Graph`.
  """
  if not isinstance(graph, ops.Graph):
    raise TypeError("Expected a graph, got: {}".format(type(graph)))
  ts = []
  for op in graph.get_operations():
    ts += op.outputs
  return ts


def get_unique_graph(tops, check_types=None, none_if_empty=False):
  # [FUNCTION DOC]
  # Finds the single graph that all given elements belong to.
  # Inputs: tops, a graph (returned as is) or an iterable of operations/tensors; check_types,
  # optional type or iterable of types every element must match (default operation and tensor);
  # none_if_empty, if true an empty input yields nothing/null instead of an error.
  # Output: the common graph. Raises a type error if tops is not iterable or an element has the
  # wrong type; raises a value error if elements come from different graphs (compared by graph
  # key) or if tops is empty and none_if_empty is false. No side effects.
  """Return the unique graph used by the all the elements in tops.

  Args:
    tops: iterable of elements to check (usually a list of tf.Operation and/or
      tf.Tensor). Or a tf.Graph.
    check_types: check that the element in tops are of given type(s). If None,
      the types (tf.Operation, tf.Tensor) are used.
    none_if_empty: don't raise an error if tops is an empty list, just return
      None.
  Returns:
    The unique graph used by all the tops.
  Raises:
    TypeError: if tops is not a iterable of tf.Operation.
    ValueError: if the graph is not unique.
  """
  if isinstance(tops, ops.Graph):
    return tops
  if not is_iterable(tops):
    raise TypeError("{} is not iterable".format(type(tops)))
  if check_types is None:
    check_types = (ops.Operation, tensor_lib.Tensor)
  elif not is_iterable(check_types):
    check_types = (check_types,)
  g = None
  for op in tops:
    if not isinstance(op, check_types):
      raise TypeError("Expected a type in ({}), got: {}".format(", ".join([str(
          t) for t in check_types]), type(op)))
    if g is None:
      g = op.graph
    elif g._graph_key != op.graph._graph_key:  # pylint: disable=protected-access
      raise ValueError("Operation {} does not belong to given graph".format(op))
  if g is None and not none_if_empty:
    raise ValueError("Can't find the unique graph of an empty list")
  return g


def check_graphs(*args):
  # [FUNCTION DOC]
  # Verifies that all arguments that have a graph share the same graph object.
  # Input: any number of objects exposing a graph attribute; objects whose graph is null are
  # ignored. Output: returns nothing. Raises a value error naming the offending argument index
  # when a graph differs (identity comparison) from the first one found. No side effects.
  """Check that all the element in args belong to the same graph.

  Args:
    *args: a list of object with a obj.graph property.
  Raises:
    ValueError: if all the elements do not belong to the same graph.
  """
  graph = None
  for i, sgv in enumerate(args):
    if graph is None and sgv.graph is not None:
      graph = sgv.graph
    elif sgv.graph is not None and sgv.graph is not graph:
      raise ValueError(f"args[{i}] does not belong to the same graph as "
                       "other arguments.")


def make_list_of_t(ts, check_graph=True, allow_graph=True, ignore_ops=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says mixed graphs raise a type error, but the graph check raises a
  # value error for elements from different graphs.
  # Normalizes a graph, a single tensor or an iterable into a new list of tensors.
  # Inputs: ts, the thing to convert; check_graph, verify all elements share one graph; allow_graph,
  # whether a graph argument may be expanded to all its tensors; ignore_ops, if true operations
  # in the input are tolerated by the graph check and then dropped.
  # Output: list of tensors (empty for empty input); non-tensor elements are filtered out.
  # Raises a type error if a graph is given while disallowed, or if the graph check finds an
  # element of the wrong type; the graph check may raise a value error for mixed graphs.
  """Convert ts to a list of `tf.Tensor`.

  Args:
    ts: can be an iterable of `tf.Tensor`, a `tf.Graph` or a single tensor.
    check_graph: if `True` check if all the tensors belong to the same graph.
    allow_graph: if `False` a `tf.Graph` cannot be converted.
    ignore_ops: if `True`, silently ignore `tf.Operation`.
  Returns:
    A newly created list of `tf.Tensor`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `tf.Tensor` or,
     if `check_graph` is `True`, if all the ops do not belong to the same graph.
  """
  if isinstance(ts, ops.Graph):
    if allow_graph:
      return get_tensors(ts)
    else:
      raise TypeError("allow_graph is False: cannot convert a tf.Graph.")
  else:
    if not is_iterable(ts):
      ts = [ts]
    if not ts:
      return []
    if check_graph:
      check_types = None if ignore_ops else tensor_lib.Tensor
      get_unique_graph(ts, check_types=check_types)
    return [t for t in ts if isinstance(t, tensor_lib.Tensor)]


def get_generating_ops(ts):
  # [FUNCTION DOC]
  # Maps tensors to the operations that produce them.
  # Input: ts, a tensor or iterable of tensors (graphs are not allowed). Output: list of producing
  # operations, one per tensor, in order, possibly with repeats. Raises a type error if the input
  # cannot be converted to tensors. No side effects.
  """Return all the generating ops of the tensors in `ts`.

  Args:
    ts: a list of `tf.Tensor`
  Returns:
    A list of all the generating `tf.Operation` of the tensors in `ts`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `tf.Tensor`.
  """
  ts = make_list_of_t(ts, allow_graph=False)
  return [t.op for t in ts]


def get_consuming_ops(ts):
  # [FUNCTION DOC]
  # Collects the operations that consume any of the given tensors.
  # Input: ts, a tensor or iterable of tensors (graphs not allowed). Output: list of distinct
  # consumer operations in first-seen order. Raises a type error if the input cannot be converted
  # to tensors. No side effects.
  """Return all the consuming ops of the tensors in ts.

  Args:
    ts: a list of `tf.Tensor`
  Returns:
    A list of all the consuming `tf.Operation` of the tensors in `ts`.
  Raises:
    TypeError: if ts cannot be converted to a list of `tf.Tensor`.
  """
  ts = make_list_of_t(ts, allow_graph=False)
  tops = []
  for t in ts:
    for op in t.consumers():
      if op not in tops:
        tops.append(op)
  return tops


def make_list_of_op(tops, check_graph=True, allow_graph=True, ignore_ts=False):
  # [FUNCTION DOC]
  # Normalizes a graph, a single operation or an iterable into a new list of operations.
  # Inputs: tops, the thing to convert; check_graph, verify all elements share one graph;
  # allow_graph, whether a graph argument is expanded to all its operations; ignore_ts, if true
  # tensors are tolerated by the graph check and then dropped.
  # Output: list of operations (empty for empty input). Raises a type error if a graph is given
  # while disallowed or an element has the wrong type; mixed graphs raise a value error from the
  # graph check. No side effects.
  """Convert ops to a list of `tf.Operation`.

  Args:
    tops: can be an iterable of `tf.Operation`, a `tf.Graph` or a single
      operation.
    check_graph: if `True` check if all the operations belong to the same graph.
    allow_graph: if `False` a `tf.Graph` cannot be converted.
    ignore_ts: if True, silently ignore `tf.Tensor`.
  Returns:
    A newly created list of `tf.Operation`.
  Raises:
    TypeError: if tops cannot be converted to a list of `tf.Operation` or,
     if `check_graph` is `True`, if all the ops do not belong to the
     same graph.
  """
  if isinstance(tops, ops.Graph):
    if allow_graph:
      return tops.get_operations()
    else:
      raise TypeError("allow_graph is False: cannot convert a tf.Graph.")
  else:
    if not is_iterable(tops):
      tops = [tops]
    if not tops:
      return []
    if check_graph:
      check_types = None if ignore_ts else ops.Operation
      get_unique_graph(tops, check_types=check_types)
    return [op for op in tops if isinstance(op, ops.Operation)]


def _get_inputs(op, only_differentiable):
  # [FUNCTION DOC]
  # HELPER: used by the backward graph walk to choose which input tensors to follow.
  # Inputs: op, an operation; only_differentiable, whether to restrict to differentiable ops.
  # Output: the op's input tensors, or an empty list if restriction is on and the op has no
  # registered gradient. No side effects.
  op_inputs = op.inputs
  if only_differentiable:
    return op_inputs if is_differentiable(op) else []
  else:
    return op_inputs


def get_backward_walk_ops(seed_ops,
                          inclusive=True,
                          within_ops=None,
                          within_ops_fn=None,
                          stop_at_ts=(),
                          control_inputs=False,
                          only_differentiable=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a Python set, but the code returns a list (and an
  # empty list for empty seeds). Also, control_inputs is silently turned off when
  # only_differentiable is true.
  # Walks the graph backwards from seed ops through input tensors (and optionally control inputs),
  # breadth-first, and returns the visited operations.
  # Inputs: seed_ops, an operation, tensor, or iterable of either (tensors are replaced by their
  # producers); inclusive, keep the seeds in the result; within_ops, optional restriction set of
  # operations (seeds are intersected with it); within_ops_fn, optional predicate that an op must
  # satisfy; stop_at_ts, tensors at which traversal does not continue; control_inputs, also
  # follow control dependencies; only_differentiable, only expand ops with a registered gradient.
  # Output: list of operations, seeds first then discovered ones. Raises a type error if inputs
  # cannot be converted to operations or tensors. No side effects.
  """Do a backward graph walk and return all the visited ops.

  Args:
    seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    inclusive: if True the given seed_ops are also part of the resulting set.
    within_ops: an iterable of `tf.Operation` within which the search is
      restricted. If `within_ops` is `None`, the search is performed within
      the whole graph.
    within_ops_fn: if provided, a function on ops that should return True iff
      the op is within the graph traversal. This can be used along within_ops,
      in which case an op is within if it is also in within_ops.
    stop_at_ts: an iterable of tensors at which the graph walk stops.
    control_inputs: if True, control inputs will be used while moving backward.
    only_differentiable: if True, only traverse ops which are differentiable.
      This includes natively differentiable ops, or ops with custom gradients.
  Returns:
    A Python set of all the `tf.Operation` behind `seed_ops`.
  Raises:
    TypeError: if `seed_ops` or `within_ops` cannot be converted to a list of
      `tf.Operation`.
  """
  control_inputs = control_inputs and (not only_differentiable)

  if not is_iterable(seed_ops):
    seed_ops = [seed_ops]

  try:
    first_seed_op = next(iter(seed_ops))
  except StopIteration:
    # Empty iterable.
    return []

  if isinstance(first_seed_op, tensor_lib.Tensor):
    ts = make_list_of_t(seed_ops, allow_graph=False)
    seed_ops = get_generating_ops(ts)
  else:
    seed_ops = make_list_of_op(seed_ops, allow_graph=False)

  stop_at_ts = object_identity.ObjectIdentitySet(make_list_of_t(stop_at_ts))
  seed_ops = object_identity.ObjectIdentitySet(make_list_of_op(seed_ops))
  if within_ops:
    within_ops = make_list_of_op(within_ops, allow_graph=False)
    within_ops = object_identity.ObjectIdentitySet(within_ops)
    seed_ops &= within_ops

  def is_within(op):
    # [FUNCTION DOC]
    # HELPER: membership filter for the backward walk.
    # Returns true if the op passes both the optional allowed-set test and the optional
    # predicate test. Reads the enclosing within_ops and within_ops_fn. No side effects.
    return (within_ops is None or op in within_ops) and (
        within_ops_fn is None or within_ops_fn(op))

  result = list(seed_ops)
  wave = set(seed_ops)
  while wave:
    new_wave = set()
    for op in wave:
      for new_t in _get_inputs(op, only_differentiable=only_differentiable):
        if new_t in stop_at_ts:
          continue
        if new_t.op not in result and is_within(new_t.op):
          new_wave.add(new_t.op)
      if control_inputs:
        for new_op in op.control_inputs:
          if new_op not in result and is_within(new_op):
            new_wave.add(new_op)
    concatenate_unique(result, new_wave)
    wave = new_wave
  if not inclusive:
    result = [op for op in result if op not in seed_ops]
  return result


class UnliftableError(Exception):
  # [CLASS DOC]
  # Error raised when a tensor cannot be lifted out of a graph because it depends on a disallowed
  # placeholder. Carries no extra state; a class flag tells autograph to leave it unconverted.
  """Raised if a Tensor cannot be lifted from the graph."""

  # Prevent autograph from rewriting this error.
  ag_pass_through = True


def _as_operation(op_or_tensor):
  # [FUNCTION DOC]
  # HELPER: coerces tensors to their producing operation for the path and subgraph walkers.
  # Input: a tensor or operation. Output: the producing op for a tensor, otherwise the input
  # unchanged. No side effects.
  if isinstance(op_or_tensor, tensor_lib.Tensor):
    return op_or_tensor.op
  return op_or_tensor


def graph_inputs(op):
  # [FUNCTION DOC]
  # HELPER: lists the dependencies of an op for graph walks.
  # Input: op, an operation. Output: list of producer operations of its input tensors followed by
  # its control-input operations (duplicates possible). No side effects.
  return [x.op for x in op.inputs] + list(op.control_inputs)


def show_path(from_op, tensors, sources):
  # [FUNCTION DOC]
  # HELPER: builds human-readable diagnostics for lifting errors.
  # KNOWN ISSUE: the docstring says the path goes from from_op to tensors, but the search starts
  # at the tensors' ops and walks backwards through inputs until from_op is reached; the printed
  # string lists from_op first. If from_op is unreachable, a placeholder string is returned.
  # Inputs: from_op, an operation (a tensor is converted to its op); tensors, one or a list of
  # tensors/operations; sources, tensors whose producers are not traversed.
  # Output: string of op names and types joined by arrows, or a question-mark placeholder.
  # No side effects.
  """Find one path from `from_op` to any of `tensors`, ignoring `sources`.

  Args:
    from_op: A `tf.Operation`.
    tensors: A `tf.Operation`, a `tf.Tensor`, or a list thereof.
    sources: A list of `tf.Tensor`.

  Returns:
    A python string containing the path, or "??" if none is found.
  """
  if isinstance(from_op, tensor_lib.Tensor):
    from_op = from_op.op

  if not isinstance(tensors, list):
    tensors = [tensors]

  final_ops = [_as_operation(tensor) for tensor in tensors]

  visited_ops = set(x.op for x in sources)
  ops_to_visit = list(final_ops)
  some_op_output = {}
  while ops_to_visit:
    op = ops_to_visit.pop()
    if op in visited_ops:
      continue
    visited_ops.add(op)
    if op == from_op:
      path_op = op
      path = [path_op]
      while path_op not in final_ops:
        path_op = some_op_output[path_op]
        path.append(path_op)
      return " <- ".join("%s (%s)" % (x.name, x.type) for x in reversed(path))
    else:
      for inp in graph_inputs(op):
        if inp not in visited_ops and inp not in sources:
          some_op_output[inp] = op
          ops_to_visit.append(inp)
  return "??"


# TODO(jmenick) - there is considerable duplication of functionality between
# this function and get_backward_walk_ops(). Need to deduplicate.
def map_subgraph(init_tensor, sources, disallowed_placeholders, visited_ops,
                 op_outputs, add_sources):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring calls disallowed_placeholders optional with a default of all
  # placeholders, but there is no default; None means any placeholder not allowed by add_sources
  # is an error. A TODO notes duplication with the backward-walk function.
  # Walks backwards from init_tensor, recording for each visited input op the ops that consume
  # it, stopping at sources and at discovered placeholders.
  # Inputs: init_tensor, tensor or op to start from; sources, collection of tensors or ops where
  # traversal stops (may be null); disallowed_placeholders, set of ops forbidden in the subgraph
  # or null; visited_ops, set mutated to include visited ops; op_outputs, mapping from op to a
  # set, mutated to record consumers; add_sources, whether unexpected placeholders are accepted.
  # Output: set of placeholder output tensors found that are not sources.
  # Raises UnliftableError if a forbidden placeholder is reached, with a message including a path.
  """Walk a Graph and capture the subgraph between init_tensor and sources.

  Note: This function mutates visited_ops and op_outputs.

  Args:
    init_tensor:  A Tensor or Operation where the subgraph terminates.
    sources:  A set of Tensors where subgraph extraction should stop.
    disallowed_placeholders: An optional set of ops which may not appear in the
      lifted graph. Defaults to all placeholders.
    visited_ops: A set of operations which were visited in a prior pass.
    op_outputs: A defaultdict containing the outputs of an op which are to be
      copied into the new subgraph.
    add_sources: A boolean indicating whether placeholders which are not in
      sources should be allowed.

  Returns:
    The set of placeholders upon which init_tensor depends and are not in
    sources.

  Raises:
    UnliftableError: if init_tensor depends on a placeholder which is not in
      sources and add_sources is False.
  """
  ops_to_visit = [_as_operation(init_tensor)]
  extra_sources = object_identity.ObjectIdentitySet()
  while ops_to_visit:
    op = ops_to_visit.pop()
    if op in visited_ops:
      continue
    visited_ops.add(op)

    should_raise = False
    if disallowed_placeholders is not None and op in disallowed_placeholders:
      should_raise = True
    elif op.type == "Placeholder":
      if disallowed_placeholders is None and not add_sources:
        should_raise = True
      extra_sources.update(op.outputs)

    if should_raise:
      raise UnliftableError(
          "Unable to lift tensor %s because it depends transitively on "
          "placeholder %s via at least one path, e.g.: %s" %
          (repr(init_tensor), repr(op), show_path(op, init_tensor, sources)))
    for inp in graph_inputs(op):
      op_outputs[inp].add(op)
      if inp not in visited_ops and inp not in (sources or extra_sources):
        ops_to_visit.append(inp)

  return extra_sources
