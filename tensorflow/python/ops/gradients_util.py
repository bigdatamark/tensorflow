# [MODULE DOC]
# HELPER: internal implementation module behind the public graph-mode gradients function and
# related gradient utilities; not intended to be called directly by users.
# Implements symbolic (graph mode) backpropagation: given output tensors, input tensors and optional
# initial gradients, it finds the subgraph of operations between them, counts pending backprop
# inputs per operation, walks the operations in dependency order, looks up each operation's
# registered gradient function (or a symbolic gradient for function calls), aggregates the
# gradients arriving from several consumers, supports while loops through the control-flow state
# helper, and optionally colocates gradient ops and compiles them with XLA attributes. It also
# defines the public aggregation-method constants class and a small helper to ask whether tapes may
# need gradients, plus constants for the possible tape gradient types. Raises an error if used in
# eager mode. Importing registers the aggregation-method class in the public API; no other import
# time side effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Implements the graph generation for computation of gradients."""

import collections
import contextlib

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_state
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


def _MarkReachedOps(from_ops, reached_ops, func_graphs):
  # [FUNCTION DOC]
  # HELPER: forward-reachability step of the gradient computation (used by _PendingCount).
  # Starting from the operations in from_ops, performs a breadth-first traversal over the
  # consumers of every trainable output (crossing into the given function graphs that capture a
  # tensor), and adds each visited operation to the reached_ops set. Inputs: from_ops (list of
  # operations), reached_ops (set that is filled in place), func_graphs (list of function graphs
  # to search for captures). Returns nothing; mutates reached_ops.
  """Mark all ops reached from "from_ops".

  Args:
    from_ops: list of Operations.
    reached_ops: set of Operations.
    func_graphs: list of FuncGraphs. This method will traverse through
      these functions if they capture from_ops or any reachable ops.
  """
  queue = collections.deque()
  queue.extend(from_ops)
  while queue:
    op = queue.popleft()
    if op not in reached_ops:
      reached_ops.add(op)
      for output in op.outputs:
        if backprop_util.IsTrainable(output):
          queue.extend(_Consumers(output, func_graphs))


def _PendingCount(
    to_ops: list[ops.Operation],
    from_ops: list[ops.Operation],
    colocate_gradients_with_ops,
    func_graphs,
    xs_set,
):
  # [FUNCTION DOC]
  # HELPER: setup phase of the backward pass (called by _GradientsHelper).
  # Marks all operations reachable from from_ops, then walks backwards from to_ops to collect the
  # "between" operations that lie on a path of trainable tensors from from_ops to to_ops (each
  # visited once). Creates the while-loop control-flow state if any between-op belongs to a loop.
  # Then counts, for every between operation, how many of its consumers inside the between set
  # will send gradients back to it. Inputs: to_ops and from_ops (lists of operations),
  # colocate_gradients_with_ops (boolean forwarded to the loop state), func_graphs (function
  # graphs to traverse for captures) and xs_set (identity set of tensors being differentiated).
  # Returns a tuple: the set of to_ops reachable from from_ops, a mapping (default zero) from
  # operation to pending count, and the loop state object or null if there are no while loops.
  """Initialize the pending count for ops between two lists of Operations.

  'pending_count[op]' indicates the number of backprop inputs
  to this operation.

  Args:
    to_ops: list of Operations.
    from_ops: list of Operations.
    colocate_gradients_with_ops: Python bool.  See docstring of gradients().
    func_graphs: list of FuncGraphs. This method will traverse through
      these functions if they capture from_ops or any reachable ops. This is
      useful if to_ops occur in a function and from_ops are in an outer function
      or graph.
    xs_set: ObjectIdentitySet of Tensors.

  Returns:
    A tuple containing: (1) the subset of to_ops reachable from from_ops by a
    path of zero or more backpropagatable tensors, (2) a mapping from operation
    to the number of backprop inputs to that op, and (3) a ControlFlowState
    object which is not None if the ops between from_ops and to_ops contain
    control flow loops.
  """
  # Mark reachable ops from from_ops.
  reached_ops = set()
  _MarkReachedOps(from_ops, reached_ops, func_graphs)
  # X in reached_ops iff X is reachable from from_ops by a path of zero or more
  # backpropagatable tensors.

  reachable_to_ops = set(op for op in to_ops if op in reached_ops)

  # Mark between ops.
  between_ops = set()
  between_op_list = []
  queue = collections.deque()
  queue.extend(to_ops)
  while queue:
    op = queue.popleft()
    # We are interested in this op.
    if op in reached_ops:
      between_ops.add(op)
      between_op_list.append(op)
      # Clear the boolean so we won't add the inputs again.
      reached_ops.remove(op)
      for inp in _NonEagerInputs(op, xs_set):
        queue.append(inp.op)
  # X in between_ops iff X is on a path of zero or more backpropagatable tensors
  # between from_ops and to_ops

  # 'loop_state' is None if there are no while loops.
  loop_state = control_flow_state.MaybeCreateControlFlowState(
      between_op_list, between_ops, colocate_gradients_with_ops)

  # Initialize pending count for between ops.
  pending_count = collections.defaultdict(int)
  for op in between_op_list:
    for x in _NonEagerInputs(op, xs_set):
      if x.op in between_ops:
        pending_count[x.op] += 1

  return reachable_to_ops, pending_count, loop_state


def _AsList(x):
  # [FUNCTION DOC]
  # HELPER: normalizes an argument that may be a single item or a sequence. Returns x unchanged
  # if it is a list or tuple, otherwise a new one-element list containing x. No side effects.
  return x if isinstance(x, (list, tuple)) else [x]


def _DefaultGradYs(grad_ys,
                   ys,
                   colocate_gradients_with_ops,
                   gradient_uid="__unsupported__"):
  # [FUNCTION DOC]
  # HELPER: prepares the initial upstream gradients for gradients computation.
  # KNOWN ISSUE: when y is complex and the supplied gradient is not complex, the raised message says
  # the gradient "must be real", although the check actually requires a complex gradient.
  # Inputs: grad_ys (list of gradient values, entries may be null), ys (list of tensors; must have
  # the same length), colocate_gradients_with_ops (boolean; colocate created ops with each y's op),
  # gradient_uid (identifier of this gradients call, default is a placeholder string). Missing
  # gradients become tensors of ones shaped like y (not allowed for complex y). Provided gradients
  # are converted to tensors or sparse slices, validated against y's dtype (real or integer needs
  # real or integer, complex needs complex, variant needs variant, resource needs a non-resource,
  # anything else is rejected) and wrapped in identity ops named after their index so they live in
  # the gradient name scope (sparse slices have each tensor component wrapped). Returns the new
  # list without null entries. Raises a value error on length mismatch and a type error on dtype
  # problems or complex y with no gradient. Creates graph operations.
  """Fill in default values for grad_ys.

  Args:
    grad_ys: List of gradients, can contain None.
    ys: List of tensors.
    colocate_gradients_with_ops: If True, try colocating gradients with
      the corresponding op.
    gradient_uid: A unique identifier within the graph indicating
      which invocation of gradients is being executed. Used to cluster
      ops for compilation.

  Returns:
    A list of gradients to use, without None.

  Raises:
    ValueError: If sizes of gradients and inputs don't match
    TypeError: If type of any gradient is not valid for its input.
  """
  if len(grad_ys) != len(ys):
    raise ValueError(f"Length mismatch. Passed {len(grad_ys)} grad_ys for "
                     f"{len(ys)} ys")
  grad_ys = indexed_slices.convert_n_to_tensor_or_indexed_slices(
      grad_ys, name="grad_y")
  new_grad_ys = []
  for i, (y, grad_y) in enumerate(zip(ys, grad_ys)):
    with _maybe_colocate_with(y.op, gradient_uid, colocate_gradients_with_ops):
      if grad_y is None:
        if y.dtype.is_complex:
          raise TypeError(
              f"Gradients of complex tensors ({y}) must set grad_ys (y.dtype = "
              f"{dtypes.as_dtype(y.dtype).name})"
          )
        new_grad_ys.append(
            array_ops.ones(
                array_ops.shape(y), dtype=y.dtype, name="grad_ys_%d" % i
            )
        )
        continue
      if y.dtype.is_floating or y.dtype.is_integer:
        if not grad_y.dtype.is_floating and not grad_y.dtype.is_integer:
          raise TypeError(
              f"Gradient type {dtypes.as_dtype(grad_y.dtype).name} generated "
              f"for real or integer-valued tensor {y} with type "
              f"{dtypes.as_dtype(y.dtype).name} must be real or integer"
          )
      elif y.dtype.is_complex:
        if not grad_y.dtype.is_complex:
          raise TypeError(
              f"Gradient type {dtypes.as_dtype(grad_y.dtype).name} generated "
              f"for complex-valued tensor {y} with type "
              f"{dtypes.as_dtype(y.dtype).name} must be real"
          )
      elif y.dtype == dtypes.variant:
        if grad_y.dtype != dtypes.variant:
          raise TypeError(
              f"Gradient type {dtypes.as_dtype(grad_y.dtype).name} generated "
              f"for variant tensor {y} with type "
              f"{dtypes.as_dtype(y.dtype).name} must be variant"
          )
      elif y.dtype == dtypes.resource:
        # We assume y is the handle of a ResourceVariable. The gradient of a
        # ResourceVariable should be a numeric value, not another resource.
        if grad_y.dtype == dtypes.resource:
          raise TypeError(
              f"Input gradient {grad_y} for resource tensor {y} "
              "should not be a resource"
          )
      else:
        raise TypeError(
            f"Tensor {y} with type {dtypes.as_dtype(y.dtype).name} must be "
            "numeric to obtain a default gradient"
        )
      # Create a grad_y tensor in the name scope of the gradient.
      # Required for TensorArrays to identify which gradient call a
      # grad_y value is coming from.
      if isinstance(grad_y, indexed_slices.IndexedSlices):
        new_grad_ys.append(
            indexed_slices.IndexedSlices(
                indices=(
                    array_ops.identity(
                        grad_y.indices, name="grad_ys_%d_indices" % i
                    )
                    if isinstance(grad_y.indices, tensor_lib.Tensor)
                    else grad_y.indices
                ),
                values=(
                    array_ops.identity(
                        grad_y.values, name="grad_ys_%d_values" % i
                    )
                    if isinstance(grad_y.values, tensor_lib.Tensor)
                    else grad_y.values
                ),
                dense_shape=(
                    array_ops.identity(
                        grad_y.dense_shape, name="grad_ys_%d_shape" % i
                    )
                    if isinstance(grad_y.dense_shape, tensor_lib.Tensor)
                    else grad_y.dense_shape
                ),
            )
        )
      else:
        new_grad_ys.append(array_ops.identity(grad_y, name="grad_ys_%d" % i))

  return new_grad_ys


def _VerifyGeneratedGradients(grads, op: ops.Operation):
  # [FUNCTION DOC]
  # HELPER: sanity check on the output of a gradient function.
  # KNOWN ISSUE: the docstring claims to verify both number and type of gradients and to raise a
  # type error, but the code only compares the number of gradients with the number of op inputs
  # and never checks types.
  # Inputs: grads (list of generated gradients) and op (the forward operation). Operations of type
  # While or StatelessWhile are skipped because their inputs grow during gradient construction.
  # Otherwise raises a value error if the count differs from the number of op inputs. Returns
  # nothing.
  """Verify that gradients are valid in number and type.

  Args:
    grads: List of generated gradients.
    op: Operation for which the gradients where generated.

  Raises:
    ValueError: if sizes of gradients and inputs don't match.
    TypeError: if type of any gradient is not valid for its input.
  """
  # While ops have inputs added to them during the gradient computation, so we
  # skip the below check. See while_v2 for details.
  if op.type == "While" or op.type == "StatelessWhile":
    return

  if len(grads) != len(op.inputs):
    raise ValueError(
        f"Num gradients {len(grads)} generated for op "
        f"{op.node_def} do not match num inputs {len(op.inputs)}"
    )


def _StopOps(
    from_ops: list[ops.Operation],
    stop_gradient_ops: list[ops.Operation],
    pending_count,
    xs_set,
):
  # [FUNCTION DOC]
  # HELPER: computes where backpropagation must stop (used by _GradientsHelper).
  # An operation from from_ops is a stop operation if none of its non-eager inputs belongs to an
  # operation with a positive pending count (it has no predecessor among the from ops). All ops in
  # stop_gradient_ops are added as well. Inputs: from_ops, stop_gradient_ops (lists of operations),
  # pending_count (mapping from operation to pending backprop count) and xs_set (identity set of
  # tensors being differentiated). Returns the set of stop operations. No mutation of arguments.
  """The set of ops that terminate the gradient computation.

  This computes the frontier of the forward graph *before* which backprop
  should stop. Operations in the returned set will not be differentiated.
  This set is defined as the subset of `from_ops` containing ops that have
  no predecessor in `from_ops`. `pending_count` is the result of
  `_PendingCount(xs, from_ops)`. An 'op' has predecessors in `from_ops`
  iff pending_count[op] > 0.

  In addition, none of `stop_gradient_ops` will be differentiated.

  Args:
    from_ops: list of Operations.
    stop_gradient_ops: list of Operations never to backprop through.
    pending_count: mapping from operation to number of backprop inputs.
    xs_set: ObjectIdentitySet of Tensors.

  Returns:
    The set of operations.
  """
  stop_ops = set()
  for op in from_ops:
    is_stop_op = True
    for inp in _NonEagerInputs(op, xs_set):
      if pending_count[inp.op] > 0:
        is_stop_op = False
        break
    if is_stop_op:
      stop_ops.add(op)
  stop_ops.update(op for op in stop_gradient_ops)
  return stop_ops


@contextlib.contextmanager
def _maybe_colocate_with(  # pylint: disable=invalid-name
    op: ops.Operation,
    gradient_uid,
    colocate_gradients_with_ops,
):
  # [FUNCTION DOC]
  # HELPER: context manager used while building gradient operations.
  # If colocate_gradients_with_ops is true, the enclosed block runs with new operations colocated
  # with op (using the gradient uid to group them); otherwise it simply runs the block unchanged.
  # Yields nothing. Side effect: changes the graph's colocation stack for the duration.
  """Context to colocate with `op` if `colocate_gradients_with_ops`."""
  if colocate_gradients_with_ops:
    with ops._colocate_with_for_gradient(op, gradient_uid):  # pylint: disable=protected-access
      yield
  else:
    yield


def _IsPartitionedCall(op: ops.Operation):
  # [FUNCTION DOC]
  # HELPER: predicate returning true when the operation's type is the partitioned-call or the
  # stateful-partitioned-call function invocation op, false otherwise. No side effects.
  return op.type == "PartitionedCall" or op.type == "StatefulPartitionedCall"


def _SymGrad(op: ops.Operation, out_grads):
  # [FUNCTION DOC]
  # HELPER: fallback gradient for function-call operations that have no registered gradient.
  # Builds a function reference naming the called function (taken from the "f" attribute for
  # partitioned calls, otherwise the op type) and copies all of the op's attributes onto it. Then
  # adds a symbolic-gradient op taking the op's inputs followed by out_grads, with output types
  # derived from the zero-gradient dtype of each input. Inputs: op (the call operation) and
  # out_grads (list of gradient tensors for its outputs). Returns the list of input gradients.
  # Side effect: adds a node to the current graph.
  """Backprop through a function call node op given its outputs' gradients."""
  f_in = [x for x in op.inputs] + out_grads
  f_types = [default_gradient.get_zeros_dtype(x) for x in op.inputs]
  f = attr_value_pb2.NameAttrList()
  if _IsPartitionedCall(op):
    f.name = op.get_attr("f").name
  else:
    f.name = op.type
  for k in op.node_def.attr:
    f.attr[k].CopyFrom(op.node_def.attr[k])
  in_grads = gen_functional_ops.symbolic_gradient(input=f_in, Tout=f_types, f=f)
  return in_grads


def _MaybeCompile(scope, op: ops.Operation, func, grad_fn):
  # [FUNCTION DOC]
  # HELPER: wraps the construction of gradient ops so XLA compilation attributes carry over.
  # Inputs: scope (name scope of the gradients call, slashes turned into underscores), op (forward
  # operation), func (function definition holder for function-call ops, or null) and grad_fn
  # (callback with no arguments that builds the gradient ops). Reads the XLA compile flag, the
  # separate-gradient flag and the XLA scope from the function definition when given, otherwise
  # from op attributes (a missing attribute means no compilation). If compilation is not requested
  # it just calls grad_fn. Otherwise it calls grad_fn inside an attribute scope that sets the compile
  # flag and an XLA scope (suffixed with the gradient scope if gradients are compiled separately).
  # Returns whatever grad_fn returns. Side effect: gradient ops get XLA attributes.
  """Compile the calculation in grad_fn if op was marked as compiled."""
  scope = scope.rstrip("/").replace("/", "_")
  if func is not None:
    xla_compile = func.cached_definition.attr["_XlaCompile"].b
    xla_separate_compiled_gradients = func.cached_definition.attr[
        "_XlaSeparateCompiledGradients"].b
    xla_scope = func.cached_definition.attr["_XlaScope"].s.decode()
  else:
    try:
      xla_compile = op.get_attr("_XlaCompile")
      xla_separate_compiled_gradients = op.get_attr(
          "_XlaSeparateCompiledGradients")
      xla_scope = op.get_attr("_XlaScope").decode()
    except ValueError:
      xla_compile = False

  if not xla_compile:
    return grad_fn()  # Exit early

  # If the gradients are supposed to be compiled separately, we give them a
  # _XlaScope name that is based on the name_scope of the gradients.  Otherwise
  # they just inherit the existing _XlaScope name, which lets them be merged
  # together with the non-gradient computation.
  if xla_separate_compiled_gradients:
    xla_grad_scope = "%s_grad_%s" % (xla_scope, scope)
  else:
    xla_grad_scope = xla_scope

  attrs = {
      "_XlaCompile": attr_value_pb2.AttrValue(b=xla_compile),
      "_XlaScope": attr_value_pb2.AttrValue(s=xla_grad_scope.encode())
  }
  with ops.get_default_graph()._attr_scope(attrs):  # pylint: disable=protected-access
    return grad_fn()


def _RaiseNoGradWrtInitialLoopValError(
    op: ops.Operation,
    from_ops: list[ops.Operation],
    xs_set,
):
  # [FUNCTION DOC]
  # HELPER: error reporter used when a loop-variable switch is hit during gradient construction.
  # Searches backwards from op (through non-eager inputs) for the nearest operation contained in
  # from_ops so the message can name it, asserts that one is found, and always raises a value
  # error stating that gradients with respect to or through the initial value of a loop variable
  # are unsupported. Inputs: op, from_ops (list of operations), xs_set (identity set of tensors).
  # Never returns normally.
  """Raises an error if we backprop through a loop var."""

  # Find the nearest 'to_op' reachable from 'op' to provide a more helpful error
  # message.
  target_op = None
  queue = collections.deque([op])
  visited = set()
  while queue:
    curr_op = queue.popleft()
    if curr_op in visited: continue
    visited.add(curr_op)
    if curr_op in from_ops:
      target_op = curr_op
      break
    queue.extend(t.op for t in _NonEagerInputs(curr_op, xs_set))
  assert target_op
  raise ValueError(
      "Cannot compute gradient inside while loop with respect to op "
      f"'{target_op.name}'. We do not support taking the gradient wrt or "
      "through the initial value of a loop variable. Gradients can be computed "
      "through loop invariants or wrt the input parameters to the loop body.")


def _IsFunction(graph):
  # [FUNCTION DOC]
  # HELPER: returns true if the given object is a graph that is currently building a function body
  # (a function graph), false otherwise. Avoids importing the function-graph module directly.
  # No side effects.
  # isinstance check for FuncGraphs that avoids the explicit dependency
  # on func_graph.py and function.py
  return isinstance(graph, ops.Graph) and graph._building_function  # pylint: disable=protected-access


def _Captures(func_graph):
  # [FUNCTION DOC]
  # HELPER: returns the captured-value pairs (outer tensor and its placeholder inside the function)
  # of a function graph. Asserts that the argument really is a function graph. No side effects.
  assert _IsFunction(func_graph)
  return func_graph.captures


def _MaybeCaptured(t):
  # [FUNCTION DOC]
  # HELPER: resolves captured placeholders back to the original outer tensor.
  # If t is a symbolic placeholder inside a function graph that stands for a captured value, the
  # function follows the capture (recursively, through nested function graphs) and returns the
  # outer tensor; otherwise it returns t itself. Eager tensors are returned unchanged.
  # No side effects.
  """If t is a captured value placeholder, returns the original captured value.

  Args:
    t: Tensor

  Returns:
    A tensor, potentially from a different Graph/FuncGraph.
  """
  # pylint: disable=protected-access
  if (not isinstance(t, ops.EagerTensor) and
      _IsFunction(t.op.graph) and t.op.type == "Placeholder"):
    for input_t, placeholder_t in _Captures(t.op.graph):
      if t is placeholder_t:
        return _MaybeCaptured(input_t)
  # pylint: enable=protected-access
  return t


def _NonEagerInputs(op: ops.Operation, xs_set):
  # [FUNCTION DOC]
  # HELPER: like _Inputs but drops eager tensors (captured constants), so the result may be
  # shorter than the op's real input list. Inputs: op and xs_set (identity set of tensors being
  # differentiated). Returns a list of tensors, possibly from several graphs. No side effects.
  """Returns the inputs of op, crossing closure boundaries where necessary.

  Does not return any captured EagerTensors, i.e., the number of tensors
  returned may be less than the actual number of inputs.

  Args:
    op: Operation
    xs_set: ObjectIdentitySet of Tensors we are differentiating w.r.t.

  Returns:
    A list of tensors. The tensors may be from multiple Graph/FuncGraphs if op
    is in a FuncGraph and has captured inputs.
  """
  return [t for t in _Inputs(op, xs_set) if not isinstance(t, ops.EagerTensor)]


# TODO(skyewm): plumbing xs through everywhere is ugly, consider making
# _GradientsHelper a class with xs as a member variable.
def _Inputs(op: ops.Operation, xs_set):
  # [FUNCTION DOC]
  # HELPER: returns the inputs of op for graph traversal. For an op in a function graph, each input
  # that is not itself being differentiated is replaced by the outer captured tensor if it is a
  # captured placeholder; inputs in xs_set are kept as they are. For ops in ordinary graphs the
  # op's inputs are returned directly. Returns a list-like of tensors. No side effects.
  """Returns the inputs of op, crossing closure boundaries where necessary.

  Args:
    op: Operation
    xs_set: ObjectIdentitySet of Tensors we are differentiating w.r.t.

  Returns:
    A list of tensors. The tensors may be from multiple Graph/FuncGraphs if op
    is in a FuncGraph and has captured inputs.
  """
  if _IsFunction(op.graph):  # pylint: disable=protected-access
    inputs = []
    for t in op.inputs:
      # If we're differentiating w.r.t. `t`, do not attempt to traverse through
      # it to a captured value. The algorithm needs to "see" `t` in this case,
      # even if it's a function input for a captured value, whereas usually we'd
      # like to traverse through these closures as if the captured value was the
      # direct input to op.
      if t not in xs_set:
        t = _MaybeCaptured(t)
      inputs.append(t)
    return inputs
  else:
    return op.inputs


def _Consumers(t, func_graphs):
  # [FUNCTION DOC]
  # HELPER: returns the operations that consume tensor t, including consumers inside the given
  # function graphs that capture t (found recursively through their placeholders).
  # KNOWN ISSUE: the docstring says it returns tensors, but the consumers are operations.
  # Input t is a tensor; func_graphs is a list of function graphs. Extends the list returned by
  # the tensor's consumer query (which is a fresh list) and returns it. No other side effects.
  """Returns the consumers of t, crossing closure boundaries where necessary.

  Args:
    t: Tensor
    func_graphs: a list of FuncGraphs that may have captured t.

  Returns:
    A list of tensors. The tensors will be from the current graph and/or
    func_graphs.
  """
  consumers = t.consumers()
  for func in func_graphs:
    for input_t, placeholder in _Captures(func):
      if input_t is t:
        consumers.extend(_Consumers(placeholder, func_graphs))
  return consumers


def _GradientsHelper(
    ys,
    xs,
    grad_ys=None,
    name="gradients",
    colocate_gradients_with_ops=False,
    gate_gradients=False,
    aggregation_method=None,
    stop_gradients=None,
    unconnected_gradients=UnconnectedGradients.NONE,
    src_graph=None,
    optimize_identity_n=False,
):
  # [FUNCTION DOC]
  # KNOWN ISSUE: when ys or xs contain composite tensors the function recurses on the flattened
  # tensors but does not pass optimize_identity_n along, so that option silently reverts to false
  # on that path. Also, the "no gradient defined" error message lacks a space before the quoted
  # op name.
  # Core implementation of graph-mode gradients (the public gradients function wraps it).
  # Inputs: ys and xs (tensors or lists of them; variables are replaced by their handles), grad_ys
  # (optional initial gradients per y), name (name scope, default "gradients"),
  # colocate_gradients_with_ops (colocate gradient ops with forward ops), gate_gradients (group the
  # gradients of one op with a tuple so they are computed together), aggregation_method (one of the
  # aggregation constants, default add-n), stop_gradients (tensors not to backprop through),
  # unconnected_gradients (none or zero for xs unreachable from ys), src_graph (graph to use,
  # default current default graph) and optimize_identity_n (do not create zero gradients for
  # missing outputs of identity-n ops).
  # Algorithm: flatten composite tensors (recursing), gather the function graphs enclosing the
  # source graph, convert inputs, build default grad_ys, compute pending counts and stop ops, seed
  # the gradient map with grad_ys, and process ops from a queue: aggregate each op's output
  # gradients, find its gradient function (registered, else a function's python gradient, else a
  # symbolic gradient for function calls, else raise a lookup error), fill zeros for missing
  # trainable outputs, call it inside a name scope named after the op with the XLA attributes,
  # verify the number of results, optionally gate them, set static shapes of each input gradient
  # (raising a value error on incompatible shapes), store them for the inputs and update pending
  # counts. While-loop contexts are entered and exited around these steps. Finally returns a list
  # with one gradient (or null / zeros) per x.
  # Raises a runtime error when eager execution is enabled, a value error for an unknown
  # unconnected_gradients value, a lookup error when an op has no gradient, and the errors of
  # helpers. Side effects: adds many operations to the graph; logs gradients at verbosity 1.
  """Implementation of gradients()."""

  if context.executing_eagerly():
    raise RuntimeError("tf.gradients is not supported when eager execution "
                       "is enabled. Use tf.GradientTape instead.")
  ys = variable_utils.convert_variables_to_tensors(_AsList(ys))
  xs = [
      x.handle if resource_variable_ops.is_resource_variable(x) else x
      for x in _AsList(xs)
  ]
  if grad_ys is not None:
    grad_ys = _AsList(grad_ys)

  # Handle CompositeTensors.
  if (any(isinstance(x, composite_tensor.CompositeTensor) for x in xs) or
      any(isinstance(y, composite_tensor.CompositeTensor) for y in ys)):
    flat_xs = composite_tensor_gradient.get_flat_tensors_for_gradients(xs)
    flat_ys = composite_tensor_gradient.get_flat_tensors_for_gradients(ys)
    flat_grad_ys = (
        None if grad_ys is None else
        composite_tensor_gradient.get_flat_tensors_for_gradients(grad_ys))
    flat_grads = _GradientsHelper(flat_ys, flat_xs, flat_grad_ys, name,
                                  colocate_gradients_with_ops, gate_gradients,
                                  aggregation_method, stop_gradients,
                                  unconnected_gradients, src_graph)
    return composite_tensor_gradient.replace_flat_tensors_for_gradients(
        xs, flat_grads)

  if src_graph is None:
    src_graph = ops.get_default_graph()
  try:
    unconnected_gradients = UnconnectedGradients(unconnected_gradients)
  except ValueError:
    raise ValueError(
        f"Unknown value for unconnected_gradients: '{unconnected_gradients}'")

  # If src_graph is a _FuncGraph (i.e. a function body), gather it and all
  # ancestor graphs. This is necessary for correctly handling captured values.
  func_graphs = []
  curr_graph = src_graph
  while _IsFunction(curr_graph):
    func_graphs.append(curr_graph)
    curr_graph = curr_graph.outer_graph

  stop_gradients = [] if stop_gradients is None else _AsList(stop_gradients)
  if grad_ys is None:
    grad_ys = [None] * len(ys)

  with ops.name_scope(
      name, "gradients",
      list(ys) + list(xs) + list(stop_gradients) + list(grad_ys)) as grad_scope:
    # Get a uid for this call to gradients that can be used to help
    # cluster ops for compilation.
    gradient_uid = ops.get_default_graph().unique_name("uid")
    ys = indexed_slices.convert_n_to_tensor_or_indexed_slices(ys, name="y")
    xs = indexed_slices.internal_convert_n_to_tensor_or_indexed_slices(
        xs, name="x", as_ref=True)
    xs_set = object_identity.ObjectIdentitySet(xs)
    grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops,
                             gradient_uid)

    # The approach we take here is as follows: Create a list of all ops in the
    # subgraph between the ys and xs.  Visit these ops in reverse order of ids
    # to ensure that when we visit an op the gradients w.r.t its outputs have
    # been collected.  Then aggregate these gradients if needed, call the op's
    # gradient function, and add the generated gradients to the gradients for
    # its input.

    # Initialize the pending count for ops in the connected subgraph from ys
    # to the xs.
    to_ops = [t.op for t in ys]
    from_ops = [t.op for t in xs]
    stop_gradient_ops = [t.op for t in stop_gradients]
    reachable_to_ops, pending_count, loop_state = _PendingCount(
        to_ops, from_ops, colocate_gradients_with_ops, func_graphs, xs_set)

    # Iterate over the collected ops.
    #
    # grads: op => list of gradients received on each output endpoint of the
    # op.  The gradients for each endpoint are initially collected as a list.
    # When it is time to call the op's gradient function, for each endpoint we
    # aggregate the list of received gradients into a Add() Operation if there
    # is more than one.
    grads = {}

    # Add the initial gradients for the ys.
    for y, grad_y in zip(ys, grad_ys):
      _SetGrad(grads, y, grad_y)

    # Initialize queue with to_ops.
    queue = collections.deque()
    # Add the ops in 'to_ops' into the queue.
    to_ops_set = set()
    for op in to_ops:
      # 'ready' handles the case where one output gradient relies on
      # another output's gradient.
      ready = (pending_count[op] == 0)
      if ready and op not in to_ops_set and op in reachable_to_ops:
        to_ops_set.add(op)
        queue.append(op)

    if loop_state:
      loop_exits = loop_state.ProcessUnusedLoopExits(pending_count, to_ops_set)
      for y in loop_exits:
        if backprop_util.IsTrainable(y):
          _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
          queue.append(y.op)

    stop_ops = _StopOps(from_ops, stop_gradient_ops, pending_count, xs_set)
    while queue:
      # generate gradient subgraph for op.
      op = queue.popleft()
      with _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops):
        if loop_state:
          loop_state.EnterGradWhileContext(op, before=True)
        out_grads = _AggregatedGrads(grads, op, gradient_uid, loop_state,
                                     aggregation_method)
        if loop_state:
          loop_state.ExitGradWhileContext(op, before=True)

        grad_fn = None
        func_call = None
        is_partitioned_call = _IsPartitionedCall(op)
        # pylint: disable=protected-access
        is_func_call = src_graph._is_function(op.type) or is_partitioned_call
        # pylint: enable=protected-access
        has_out_grads = any(
            isinstance(g, tensor_lib.Tensor) or g for g in out_grads
        )
        if has_out_grads and (op not in stop_ops):
          try:
            grad_fn = ops.get_gradient_function(op)
          except LookupError:
            if is_func_call:
              if is_partitioned_call:
                func_name = compat.as_bytes(op.get_attr("f").name)
                func_call = src_graph._get_function(  # pylint: disable=protected-access
                    func_name)
                # When a graph is imported, the FunctionDefs are not copied over
                # to each sub-graph so we recursively search the outer graphs
                # for the FunctionDef.
                if not func_call and hasattr(src_graph, "outer_graph"):
                  graph = src_graph.outer_graph
                  while graph is not None:
                    func_call = graph._get_function(func_name)  # pylint: disable=protected-access
                    if func_call  is not None:
                      break
                    if hasattr(graph, "outer_graph"):
                      graph = graph.outer_graph
                    else:
                      break
              else:
                func_call = src_graph._get_function(op.type)  # pylint: disable=protected-access
              # Note that __defun is not set if the graph is
              # imported. If it's set, we prefer to access the original
              # defun.
              func_call = getattr(op, "__defun", func_call)
              grad_fn = func_call.python_grad_func
            else:
              raise LookupError(
                  "No gradient defined for operation"
                  f"'{op.name}' (op type: {op.type}). "
                  "In general every operation must have an associated "
                  "`@tf.RegisterGradient` for correct autodiff, which this "
                  "op is lacking. If you want to pretend this "
                  "operation is a constant in your program, you may insert "
                  "`tf.stop_gradient`. This can be useful to silence the "
                  "error in cases where you know gradients are not needed, "
                  "e.g. the forward pass of tf.custom_gradient. "
                  "Please see more details in "
                  "https://www.tensorflow.org/api_docs/python/tf/custom_gradient.")  # pylint: disable=line-too-long
        if loop_state:
          loop_state.EnterGradWhileContext(op, before=False)

        # NOTE(skyewm): We don't support computing gradients wrt a loop variable
        # unless it's within the context of a single iteration (i.e. the
        # gradient is wrt to the loop parameter in the body function, not wrt or
        # through the initial value). This means if we're in a while loop
        # context, we should never see a switch node from this context.
        # pylint: disable=protected-access
        if (control_flow_util.IsSwitch(op) and
            op._control_flow_context is not None and
            op._control_flow_context.IsWhileContext() and
            op._control_flow_context ==
            ops.get_default_graph()._get_control_flow_context()):
          _RaiseNoGradWrtInitialLoopValError(op, from_ops, xs_set)
        # pylint: enable=protected-access

        if (grad_fn or is_func_call) and has_out_grads:
          # NOTE: If _AggregatedGrads didn't compute a value for the i'th
          # output, it means that the cost does not depend on output[i],
          # therefore dC/doutput[i] is 0.
          #
          # NOTE: "IdentityN" acts as a pass-through node, so we don't need to
          # instantiate zeros for its inputs with corresponding `None` output
          # gradients.
          for i, out_grad in enumerate(out_grads):
            if (
                (not isinstance(out_grad, tensor_lib.Tensor) and not out_grad)
                and (
                    (not grad_fn and is_func_call)
                    or backprop_util.IsTrainable(op.outputs[i])
                )
                and not (optimize_identity_n and op.type == "IdentityN")
            ):
              # Only trainable outputs or outputs for a function call that
              # will use SymbolicGradient get a zero gradient. Gradient
              # functions should ignore the gradient for other outputs.
              # TODO(apassos) gradients of resource handles might be an
              # issue here because of zeros.
              if loop_state:
                out_grads[i] = loop_state.ZerosLikeV1WhileLoop(op, i)
              elif default_gradient.supports_default_grad(op.outputs[i]):
                # TODO(b/143286622): The supports_default_grad check is needed
                # because While op emits non-differentiable resource tensors
                # as outputs. Remove this check when that is not the case.
                out_grads[i] = control_flow_state.ZerosLike(op, i)
          with ops.name_scope(op.name + "_grad"):
            # pylint: disable=protected-access
            with src_graph._original_op(op):
              # pylint: enable=protected-access
              if grad_fn:
                # If grad_fn was found, do not use SymbolicGradient even for
                # functions.
                in_grads = _MaybeCompile(grad_scope, op, func_call,
                                         lambda: grad_fn(op, *out_grads))
              else:
                # For function call ops, we add a 'SymbolicGradient'
                # node to the graph to compute gradients.
                in_grads = _MaybeCompile(grad_scope, op, func_call,
                                         lambda: _SymGrad(op, out_grads))
              in_grads = _AsList(in_grads)
              _VerifyGeneratedGradients(in_grads, op)
              if gate_gradients and len([x for x in in_grads
                                         if x is not None]) > 1:
                with ops.device(None):
                  with ops._colocate_with_for_gradient(  # pylint: disable=protected-access
                      None,
                      gradient_uid,
                      ignore_existing=True):
                    in_grads = control_flow_ops.tuple(in_grads)
          _LogOpGradients(op, out_grads, in_grads)
        else:
          # If no grad_fn is defined or none of out_grads is available,
          # just propagate a list of None backwards.
          in_grads = [None] * len(_Inputs(op, xs_set))
        # Note: we don't filter out eager inputs here because the inputs need to
        # line up with in_grads.
        for i, (t_in, in_grad) in enumerate(zip(_Inputs(op, xs_set), in_grads)):
          if in_grad is not None:
            if (isinstance(in_grad, tensor_lib.Tensor) and
                t_in.dtype != dtypes.resource):
              try:
                in_grad.set_shape(t_in.get_shape())
              except ValueError:
                raise ValueError(
                    "Incompatible shapes between op input and calculated "
                    f"input gradient. Forward operation: {op.name}. Input "
                    f"index: {i}. Original input shape: {t_in.shape}. "
                    f"Calculated input gradient shape: {in_grad.shape}")
            if not isinstance(t_in, ops.EagerTensor):
              _SetGrad(grads, t_in, in_grad)
        if loop_state:
          loop_state.ExitGradWhileContext(op, before=False)

      # Update pending count for the inputs of op and enqueue ready ops.
      _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state,
                                    xs_set)

  if loop_state:
    loop_state.PostProcessing()
  return [_GetGrad(grads, x, unconnected_gradients) for x in xs]


def _HasAnyNotNoneGrads(grads, op: ops.Operation):
  # [FUNCTION DOC]
  # HELPER: reports whether any output of op has received a real gradient. Looks at the stored
  # gradients for op and returns true if any is a tensor or sparse slices, or a non-empty sequence
  # containing a non-null item; otherwise false. Inputs: grads (mapping from op to per-output
  # gradient entries) and op. No side effects.
  """Return true iff op has real gradient."""
  out_grads = _GetGrads(grads, op)
  for out_grad in out_grads:
    if isinstance(out_grad, (tensor_lib.Tensor, indexed_slices.IndexedSlices)):
      return True
    if out_grad and isinstance(out_grad, collections_abc.Sequence):
      if any(g is not None for g in out_grad):
        return True
  return False


def _UpdatePendingAndEnqueueReady(
    grads, op: ops.Operation, queue, pending_count, loop_state, xs_set
):
  # [FUNCTION DOC]
  # HELPER: bookkeeping step after an op's gradients were produced.
  # For each non-eager input op of op, decrements its pending count and enqueues it when the count
  # reaches zero (in loop mode, also loop switches that still have a positive count). Loop exits
  # are not enqueued directly: they are collected in the loop gradient state, and once all exits of
  # a loop are collected, exits with real gradients are enqueued; unused exits get zero gradients
  # (if trainable and at least one exit has a real gradient) or none, and are enqueued too.
  # Inputs: grads (gradient map, may be extended with zeros), op, queue (work queue, mutated),
  # pending_count (mapping, mutated), loop_state (loop state or null) and xs_set. Returns nothing.
  """Update pending count for the inputs of op and enqueue ready ops."""
  for x in _NonEagerInputs(op, xs_set):
    pending_count[x.op] -= 1
    ready = pending_count[x.op] == 0
    if loop_state and not ready:
      ready = pending_count[x.op] > 0 and control_flow_util.IsLoopSwitch(x.op)
    if ready:
      if control_flow_util.IsLoopExit(x.op):
        # if x is an exit without real gradient, defer processing them.
        grad_state = loop_state.GetGradState(x.op, before=False)
        grad_state.deferred_exits.append(x)
        grad_state.pending_exits_count -= 1
        if grad_state.pending_exits_count == 0:
          # We now have all the exits so process them.
          has_not_none_grad = False
          for y in grad_state.deferred_exits:
            if _HasAnyNotNoneGrads(grads, y.op):
              has_not_none_grad = True
              queue.append(y.op)
            else:
              grad_state.unused_exits.append(y)
          if has_not_none_grad:
            # For an unused exit, if it has trainable outputs, backprop
            # a zero gradient. Otherwise, just ignore it.
            for y in grad_state.unused_exits:
              if backprop_util.IsTrainable(y):
                _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
              queue.append(y.op)
          else:
            # All exits are "unused" so use None as gradient.
            for y in grad_state.unused_exits:
              queue.append(y.op)
      else:
        queue.append(x.op)


def _SetGrad(grads, t, grad):
  # [FUNCTION DOC]
  # HELPER: records a gradient for tensor t in the gradient map. Creates, if absent, a list of
  # empty per-output lists for t's operation, then appends grad to the list for t's output index.
  # For loop-switch ops whose entry is not a list it overwrites the entry instead (asserting the
  # op is a loop switch). Returns nothing; mutates grads.
  """Sets gradient "grad" in "grads" for tensor "t"."""
  op = t.op
  op_grads = grads.get(op)
  if not op_grads:
    op_grads = [[] for _ in range(len(op.outputs))]
    grads[op] = op_grads
  t_grads = op_grads[t.value_index]
  if isinstance(t_grads, list):
    t_grads.append(grad)
  else:
    assert control_flow_util.IsLoopSwitch(op)
    op_grads[t.value_index] = grad


def _ZerosLike(t):
  # [FUNCTION DOC]
  # HELPER: builds a zero gradient for tensor t. The dtype comes from the default-gradient helper;
  # for resource tensors (variable handles) a zeros tensor with the variable's shape is created,
  # for other tensors a zeros-like tensor. Returns the new tensor; adds graph operations.
  t_dtype = default_gradient.get_zeros_dtype(t)
  if t.dtype == dtypes.resource:
    return array_ops.zeros(
        resource_variable_ops.variable_shape(t), dtype=t_dtype)
  else:
    return array_ops.zeros_like(t, dtype=t_dtype)


def _GetGrad(grads, t, unconnected_gradients):
  # [FUNCTION DOC]
  # HELPER: reads the final gradient for tensor t from the gradient map.
  # If nothing was recorded for t's operation, returns a zeros tensor when unconnected_gradients is
  # the zero option, null for the none option, and raises a value error for any other value. If an
  # entry exists but is null and zeros were requested, returns zeros. Otherwise asserts that the
  # entry has already been aggregated (is not a list) and returns it.
  """Gets gradient for tensor "t"."""
  op = t.op
  op_grads = grads.get(op)
  if not op_grads:
    if unconnected_gradients == UnconnectedGradients.ZERO:
      return _ZerosLike(t)
    elif unconnected_gradients == UnconnectedGradients.NONE:
      return None
    else:
      raise ValueError(
          f"Unknown value for unconnected_gradients: '{unconnected_gradients}'")

  t_grad = op_grads[t.value_index]
  # This can happen if some other output of `t.op` has non-None grad.
  if unconnected_gradients == UnconnectedGradients.ZERO and t_grad is None:
    return _ZerosLike(t)

  assert not isinstance(
      t_grad, list), ("gradients list should have been aggregated by now.")
  return t_grad


def _GetGrads(grads, op: ops.Operation):
  # [FUNCTION DOC]
  # HELPER: returns the stored list of per-output gradient entries for op, or, if none exist,
  # a fresh list with one empty list per output of op. Does not insert into grads.
  """Gets all gradients for op."""
  if op in grads:
    return grads[op]
  else:
    return [[] for _ in range(len(op.outputs))]


def _AccumulatorShape(inputs):
  # [FUNCTION DOC]
  # HELPER: computes a shape for logging gradient sums. Starts from an unknown shape and merges in
  # the static shape of every tensor in inputs (non-tensors are skipped). Returns the merged shape;
  # may raise a value error if shapes are incompatible.
  shape = tensor_shape.unknown_shape()
  for i in inputs:
    if isinstance(i, tensor_lib.Tensor):
      shape = shape.merge_with(i.get_shape())
  return shape


def _LogOpGradients(op: ops.Operation, out_grads, in_grads):
  # [FUNCTION DOC]
  # HELPER: verbose logging of gradient flow for one op.
  # KNOWN ISSUE: the log labels are from the gradient op's viewpoint: the line labelled "in" lists
  # names of out_grads (incoming gradients) and the line labelled "out" lists in_grads, which may
  # read as reversed. Items must have a name attribute, so sparse-slices entries that are not
  # lists could fail in this logging.
  # Writes three verbose log lines (level 1): the op name, the names of non-empty incoming
  # gradients and the names of non-empty produced gradients. Null and empty entries are filtered
  # by a nested predicate. Returns nothing.
  """Log the in and out grads of an op."""
  logging.vlog(1, "Gradient for '" + op.name + "'")

  def _FilterGrad(x):
    # [FUNCTION DOC]
    # HELPER: predicate for logging. Returns false for null, the truthiness (non-empty) for lists
    # and tuples, and true for anything else.
    if x is None:
      return False
    if isinstance(x, (list, tuple)):
      return bool(x)
    else:
      return True

  logging.vlog(1, "  in  --> %s",
               ", ".join(x.name for x in out_grads if _FilterGrad(x)))
  logging.vlog(1, "  out --> %s",
               ", ".join(x.name for x in in_grads if _FilterGrad(x)))


def _MultiDeviceAddN(tensor_list, gradient_uid):
  # [FUNCTION DOC]
  # HELPER: sums gradient tensors that may live on different devices.
  # Groups the tensors by device, adds each group with an add-n op created colocated with the
  # first tensor of the group (devices visited in sorted order, a missing device sorting first),
  # then adds the per-device partial sums with a final add-n. Inputs: tensor_list (list of
  # tensors) and gradient_uid (identifier used for colocation grouping). Returns the total sum
  # tensor. Side effect: adds graph operations.
  """Adds tensors from potentially multiple devices."""
  # Basic function structure comes from control_flow_ops.group().
  # Sort tensors according to their devices.
  tensors_on_device = collections.defaultdict(lambda: [])
  for tensor in tensor_list:
    tensors_on_device[tensor.device].append(tensor)

  # For each device, add the tensors on that device first.
  # Then gather the partial sums from multiple devices.
  # TODO(sjhwang): Create hierarchical aggregation tree as pbar's suggestion.
  # E.g., aggregate per GPU, then per task, and so on.
  summands = []

  def DeviceKey(dev):
    # [FUNCTION DOC]
    # HELPER: sort key for devices; maps a missing device (null) to the empty string so it can be
    # compared with device name strings, and returns other names unchanged.
    return "" if dev is None else dev

  for dev in sorted(tensors_on_device, key=DeviceKey):
    tensors = tensors_on_device[dev]
    with ops._colocate_with_for_gradient(  # pylint: disable=protected-access
        tensors[0].op,
        gradient_uid,
        ignore_existing=True):
      summands.append(math_ops.add_n(tensors))

  return math_ops.add_n(summands)


@tf_export("AggregationMethod")
class AggregationMethod:
  # [CLASS DOC]
  # Public namespace of integer constants naming the strategies used to combine several gradient
  # contributions to the same tensor: add-n (also the default), an experimental tree of pairwise
  # sums, and an experimental accumulate-n alias of the tree. Holds only class attributes, is never
  # instantiated, and is exported to the public API under the name AggregationMethod.
  # Consumed by the aggregation logic in this module and by callers of the gradients function.
  """A class listing aggregation methods used to combine gradients.

  Computing partial derivatives can require aggregating gradient
  contributions. This class lists the various methods that can
  be used to combine gradients in the graph.

  The following aggregation methods are part of the stable API for
  aggregating gradients:

  *  `ADD_N`: All of the gradient terms are summed as part of one
     operation using the "AddN" op (see `tf.add_n`). This
     method has the property that all gradients must be ready and
     buffered separately in memory before any aggregation is performed.
  *  `DEFAULT`: The system-chosen default aggregation method.

  The following aggregation methods are experimental and may not
  be supported in future releases:

  * `EXPERIMENTAL_TREE`: Gradient terms are summed in pairs using
    the "AddN" op. This method of summing gradients may reduce
    performance, but it can improve memory utilization because the
    gradients can be released earlier.
  * `EXPERIMENTAL_ACCUMULATE_N`: Same as `EXPERIMENTAL_TREE`.

  Example usage when computing gradient:

  >>> @tf.function
  ... def example():
  ...   x = tf.constant(1.0)
  ...   y = x * 2.0
  ...   z = y + y + y + y
  ...   return tf.gradients(z, [x, y],
  ...     aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
  >>> example()
  [<tf.Tensor: shape=(), dtype=float32, numpy=8.0>,
   <tf.Tensor: shape=(), dtype=float32, numpy=4.0>]

  """
  ADD_N = 0
  DEFAULT = ADD_N
  # The following are experimental and may not be supported in future releases.
  EXPERIMENTAL_TREE = 1
  EXPERIMENTAL_ACCUMULATE_N = 2  # An alias for EXPERIMENTAL_ADD_N = 1


def _AggregatedGrads(grads,
                     op,
                     gradient_uid,
                     loop_state,
                     aggregation_method=None):
  # [FUNCTION DOC]
  # HELPER: combines the gradient contributions collected for each output of op.
  # KNOWN ISSUE: the docstring says it raises a value error only for invalid arguments, and the
  # code also contains asserts that loop-state tensors belong to loop switches; the local variable
  # holding the accumulated shape shadows the imported shape module inside this function (harmless
  # here, but misleading).
  # Inputs: grads (map from operation to per-output gradient lists), op, gradient_uid (identifier
  # for colocation grouping), loop_state (loop state or null) and aggregation_method (constant or
  # null for the default; an unknown value raises a value error). For each output: loop switch
  # outputs holding a single tensor are kept; a sequence containing items that are neither tensors
  # nor sparse slices raises a type error; an empty list becomes null; a single item is used as is;
  # several dense tensors are summed (add-n across devices by default, or a running pairwise sum
  # within a name scope for the tree / accumulate options); a mixture involving sparse slices is
  # aggregated by the sparse aggregation helper. Logs the aggregation at verbosity 2. Returns the
  # list of aggregated gradients (the list from the map is modified in place).
  """Get the aggregated gradients for op.

  Args:
    grads: The map of memoized gradients.
    op: The op to get gradients for.
    gradient_uid: A unique identifier within the graph indicating
      which invocation of gradients is being executed. Used to cluster
      ops for compilation.
    loop_state: An object for maintaining the state of the while loops in the
                graph. It is of type ControlFlowState. None if the graph
                contains no while loops.
    aggregation_method: Specifies the method used to combine gradient terms.
      Accepted values are constants defined in the class `AggregationMethod`.

  Returns:
    A list of gradients, one per each output of `op`. If the gradients
      for a particular output is a list, this function aggregates it
      before returning.

  Raises:
    TypeError: if the incoming grads are not Tensors or IndexedSlices.
    ValueError: if the arguments are invalid.

  """
  if aggregation_method is None:
    aggregation_method = AggregationMethod.DEFAULT
  valid_aggregation_methods = [
      AggregationMethod.ADD_N, AggregationMethod.EXPERIMENTAL_TREE,
      AggregationMethod.EXPERIMENTAL_ACCUMULATE_N]
  if aggregation_method not in valid_aggregation_methods:
    raise ValueError(
        f"Invalid `aggregation_method` specified {aggregation_method}. "
        f"Accepted values are {valid_aggregation_methods}.")
  out_grads = _GetGrads(grads, op)
  for i, out_grad in enumerate(out_grads):
    if loop_state:
      if isinstance(
          out_grad, (tensor_lib.Tensor, indexed_slices.IndexedSlices)):
        assert control_flow_util.IsLoopSwitch(op)
        continue
    # Grads have to be Tensors or IndexedSlices
    if (isinstance(out_grad, collections_abc.Sequence) and not all(
        isinstance(g, (tensor_lib.Tensor, indexed_slices.IndexedSlices))
        for g in out_grad
        if g is not None)):
      raise TypeError(f"Invalid gradient {out_grad} [index = {i}]. Gradients "
                      "have to be either all Tensors or all IndexedSlices")
    # Aggregate multiple gradients, and convert [] to None.
    if out_grad:
      if len(out_grad) < 2:
        used = "nop"
        out_grads[i] = out_grad[0]
      elif all(
          isinstance(g, tensor_lib.Tensor) for g in out_grad if g is not None):
        tensor_shape = _AccumulatorShape(out_grad)
        if aggregation_method in [
            AggregationMethod.EXPERIMENTAL_TREE,
            AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
        ]:
          # Aggregate all gradients by doing pairwise sums: this may
          # reduce performance, but it can improve memory because the
          # gradients can be released earlier.
          #
          # TODO(vrv): Consider replacing this with a version of
          # tf.AddN() that eagerly frees its inputs as soon as they are
          # ready, so the order of this tree does not become a problem.
          used = "tree"
          with ops.name_scope(op.name + "_gradient_sum"):
            running_sum = out_grad[0]
            for grad in out_grad[1:]:
              running_sum = math_ops.add_n([running_sum, grad])
            out_grads[i] = running_sum
        else:
          used = "add_n"
          out_grads[i] = _MultiDeviceAddN(out_grad, gradient_uid)
        logging.vlog(2, "  _AggregatedGrads %d x %s using %s", len(out_grad),
                     tensor_shape, used)
      else:
        out_grads[i] = backprop_util.AggregateIndexedSlicesGradients(out_grad)  # pylint: disable=protected-access
    else:  # not out_grad
      # out_grads[i] is [], thus its aggregation is simply None.
      out_grads[i] = None
  return out_grads


# Represents the output of TFE_Py_TapeSetPossibleGradientTypes. Real enums are
# unfortunately too slow to use here.
POSSIBLE_GRADIENT_TYPES_NONE = 0
POSSIBLE_GRADIENT_TYPES_FIRST_ORDER = 1
POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER = 2


def PossibleTapeGradientTypes(tensors):
  # [FUNCTION DOC]
  # Asks the native tape machinery whether any active gradient tape could need gradients for the
  # given tensors. Input tensors is a list of tensors. Returns one of the module constants: none,
  # first-order or higher-order. No side effects besides querying the native tape set.
  """Determines whether and how `args` may require tape gradients."""
  return pywrap_tfe.TFE_Py_TapeSetPossibleGradientTypes(tensors)
