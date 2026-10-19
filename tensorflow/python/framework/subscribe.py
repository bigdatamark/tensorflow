# [MODULE DOC]
# Implements graph-mode tensor subscription: attaching side-effect subgraphs (for example
# printing or logging) to existing tensors without changing how downstream code uses them.
# The public entry point is subscribe, which accepts a tensor or nested structure of tensors and
# one or more side-effect constructor functions, and returns identity tensors that are control
# dependent on the side-effect ops; consumers of the original tensors are rewired to them.
# The rest are private helpers: structure traversal, a cache of control outputs per graph,
# creation of new subscriptions, extension of existing ones, detection of subscription identity
# ops by naming convention, and control-flow and device scoping. No import-time side effects.
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
"""Subscribe function."""

import contextlib
import re

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging


def _recursive_apply(tensors, apply_fn):
  # [FUNCTION DOC]
  # HELPER: Structure traversal used by subscribe.
  # Applies apply_fn to every tensor in a structure (a single tensor, list, tuple, named tuple
  # or plain mapping, nested arbitrarily) and rebuilds the same structure of results. Variables
  # are replaced by applying the function to their current value tensor. Raises a type error for
  # any other element type (including subclasses of mapping other than plain dict). Pure apart
  # from whatever apply_fn does.
  """Helper method to recursively apply a function to structure of tensors.

  The structure of the tensors should take the form similar to fetches in
  `tf.compat.v1.Session` and includes single `Tensor`, `list`, nested `list`,
  `tuple`,
  `namedtuple`, or `dict`.

  Args:
    tensors: Single `Tensor`, `list`, nested `list, `tuple`, `namedtuple`, or
      `dict`.
    apply_fn: Function to apply to each `Tensor` and should return a `Tensor`.

  Returns:
    Returns the modified tensors with the same structure.
  Raises:
    `TypeError` if undefined type in the tensors structure.
  """
  tensors_type = type(tensors)
  if isinstance(tensors, tensor_lib.Tensor):
    return apply_fn(tensors)
  elif isinstance(tensors, variables.Variable):
    return apply_fn(tensors.value())
  elif isinstance(tensors, (list, tuple)):
    tensors = [_recursive_apply(t, apply_fn) for t in tensors]
    if tensors_type is list:
      return list(tensors)
    elif tensors_type is tuple:
      return tuple(tensors)
    return tensors_type(*tensors)  # collections.namedtuple
  elif tensors_type is dict:
    return dict((k, _recursive_apply(v, apply_fn)) for k, v in tensors.items())
  else:
    raise TypeError(f'_recursive_apply argument {tensors!r} has invalid type '
                    f'{tensors_type!r}')


class _ControlOutputCache(object):
  # [CLASS DOC]
  # HELPER: Per-call cache used by subscribe. Holds a mapping from graph to a mapping from each
  # operation to the set of operations that have it as a control input, so the reverse
  # control-dependency lookup is computed once per graph rather than per tensor.
  """Helper class to manage calculating and caching control_outputs in graph."""

  __slots__ = ['cache']

  def __init__(self):
    # [FUNCTION DOC]
    # Creates the cache with an empty mapping.
    self.cache = {}

  def calc_control_outputs(self, graph):
    # [FUNCTION DOC]
    # Scans every operation of the given graph and returns a mapping from each operation that
    # is used as a control input to the set of operations depending on it. Read-only.
    """Returns the map of control_outputs for a given graph.

    Args:
      graph: The graph to parse.

    Returns:
      A map of the control outputs.
    """
    control_outputs = {}
    for op in graph.get_operations():
      for control_input in op.control_inputs:
        if control_input not in control_outputs:
          control_outputs[control_input] = set()
        control_outputs[control_input].add(op)
    return control_outputs

  def get_control_outputs(self, op):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the cache is never invalidated, so results can be stale if the graph gains
    # control dependencies after the first lookup (it is only used within one subscribe call).
    # Returns the operations that have the given op as a control input, computing and caching
    # the map for the op's graph on first use. Returns an empty list if there are none.
    """Return the control outputs for a given op.

    Args:
      op: The op to fetch control outputs for.

    Returns:
      Iterable of control output ops.
    """
    if op.graph not in self.cache:
      control_outputs = self.calc_control_outputs(op.graph)
      self.cache[op.graph] = control_outputs
    else:
      control_outputs = self.cache[op.graph]
    return control_outputs.get(op, [])


def _subscribe_new(tensor, side_effects, control_cache):
  # [FUNCTION DOC]
  # HELPER: Creates a brand-new subscription for subscribe.
  # Inputs: tensor to subscribe; side_effects, a list of functions each taking the tensor and
  # returning a list of ops or tensors to use as control dependencies; control_cache, the
  # control-output cache. Records the tensor's consumers, builds the side-effect subgraphs under
  # a name scope ending in subscription, creates an identity of the tensor that depends on them,
  # and rewires every recorded consumer input to the identity. Ops that had a control dependency
  # on the tensor's op are changed to depend on the identity op instead. Mutates the graph.
  # Returns the identity tensor.
  """Helper method that subscribes a single tensor to a list of side_effects.

  Args:
    tensor: `tf.Tensor`
    side_effects: List of side_effect functions see subscribe for details.
    control_cache: `_ControlOutputCache` helper to get control_outputs faster.

  Returns:
    The modified replacement to the passed in tensor which triggers the side
    effects.
  """
  update_input = []
  for consumer_op in list(tensor.consumers()):  # explicit copy
    update_input.append((consumer_op, list(consumer_op.inputs).index(tensor)))

  update_control_input = control_cache.get_control_outputs(tensor.op)

  # Trailing slash on name scope to replace the scope.
  name_scope = tensor.op.name + '/subscription/'
  with ops.name_scope(name_scope):
    outs = []
    for s in side_effects:
      outs += s(tensor)

    with ops.control_dependencies(outs):
      out = array_ops.identity(tensor)

  for consumer_op, index in update_input:
    consumer_op._update_input(index, out)  # pylint: disable=protected-access

  for consumer_op in update_control_input:
    # If an op has more than one output and two or more of its output tensors
    # are subscribed at the same time, we remove the control dependency from
    # the original op only once and we add the dependencies to all the
    # new identities.
    new_control_inputs = consumer_op.control_inputs
    if tensor.op in new_control_inputs:
      new_control_inputs.remove(tensor.op)
    new_control_inputs.append(out.op)
    # pylint: disable=protected-access
    consumer_op._remove_all_control_inputs()
    consumer_op._add_control_inputs(new_control_inputs)
    # pylint: enable=protected-access
  return out


def _subscribe_extend(tensor, side_effects):
  # [FUNCTION DOC]
  # HELPER: Adds more side effects to an existing subscription identity tensor.
  # Inputs: tensor, a subscription identity (must have exactly one input, checked by assertion);
  # side_effects, list of side-effect constructor functions. Builds each subgraph from the
  # identity's source tensor and adds the results as extra control inputs of the identity op.
  # Mutates the graph. Returns the same tensor.
  """Helper method to extend the list of side_effects for a subscribed tensor.

  Args:
    tensor: A `tf.Tensor` as returned by subscribe().
    side_effects: List of side_effect functions, see subscribe for details.

  Returns:
    The given subscribed tensor (for API consistency).
  """
  assert len(tensor.op.inputs) == 1, 'Op {} must only have one input'.format(
      tensor.op.name)
  source_tensor = tensor.op.inputs[0]

  # Build the side effect graphs and add their outputs to the list of control
  # dependencies for the subscribed tensor.
  outs = []
  name_scope = source_tensor.op.name + '/subscription/'
  with ops.name_scope(name_scope):
    for s in side_effects:
      outs += s(source_tensor)

  out_ops = [
      out.op if isinstance(out, tensor_lib.Tensor) else out for out in outs
  ]
  tensor.op._add_control_inputs(out_ops)  # pylint: disable=protected-access

  return tensor


def _is_subscribed_identity(tensor):
  # [FUNCTION DOC]
  # HELPER: Detects subscription identities for subscribe.
  # Returns true if the tensor's op is an identity whose name follows the subscription naming
  # convention and whose prefix equals the name of its single input's op; false otherwise.
  # Asserts (assertion error) that the op has exactly one input once the name check passed.
  # Read-only.
  """Checks if the given tensor is an identity op returned by `subscribe()`.

  Args:
    tensor: A `tf.Tensor` to check.

  Returns:
    True if the given tensor matches the criteria for subscription identities:
    its op type is `Identity`, its name matches the name of its input and
    conforms to the convention for subscribed nodes.
    False otherwise.
  """
  # Subscribed tensor are assumed to be identity ops.
  if tensor.op.type != 'Identity':
    return False

  # Check that the tensor name matches the convention in place for identity ops
  # created by subscribe().
  match = re.match(r'(?P<prefix_name>^.*?)/subscription/Identity[^/]+',
                   tensor.name)
  if match is None or len(match.groups()) != 1:
    return False
  prefix_name = match.group('prefix_name')

  # Get a reference to the source tensor and check that it has a matching name.
  assert len(tensor.op.inputs) == 1, 'Op {} must only have one input'.format(
      tensor.op.name)
  source_tensor = tensor.op.inputs[0]
  if prefix_name != source_tensor.op.name:
    return False

  return True


def _subscribe(tensor, side_effects, control_cache):
  # [FUNCTION DOC]
  # HELPER: Dispatcher for subscribe. Chooses between returning the tensor unchanged, extending
  # an existing subscription, or creating a new one.
  # Inputs: tensor, list of side-effect functions, and the control-output cache. Tensors whose
  # data type cannot be represented in numpy are returned unchanged after a debug log message.
  # If the tensor is itself a subscription identity, or has exactly one consumer that is one, that
  # identity is extended. Finding more than one subscription consumer triggers an assertion error.
  # Otherwise a new subscription is created. Mutates the graph; returns the resulting tensor.
  """Helper method that subscribes a single tensor to a list of side_effects.

  This method will check if the given tensor has already been subscribed or if
  it's a tensor returned by a previous call to `subscribe()` and, if so, will
  reuse the existing identity op, appending the given side effects to the list
  of existing ones.

  Args:
    tensor: The `tf.Tensor` to be subscribed.
    side_effects: List of side_effect functions, see subscribe for details.
    control_cache: `_ControlOutputCache` helper to get control_outputs faster.

  Returns:
    The modified replacement to the passed in tensor which triggers the side
    effects or the given tensor, if it was already been subscribed.
  """
  # Check if the given tensor has a numpy compatible type (see dtypes.py).
  # If not, we cannot subscribe it, so we just return the original tensor.
  if not tensor.dtype.is_numpy_compatible:
    logging.debug(('Tensor {} has an un-supported {} type and cannot be '
                   'subscribed.').format(tensor.name, tensor.dtype))
    return tensor

  if _is_subscribed_identity(tensor):
    return _subscribe_extend(tensor, side_effects)

  # Check if the given tensor has already been subscribed by inspecting its
  # outputs.
  name_scope = tensor.op.name + '/subscription/Identity'
  consumers = tensor.consumers()
  matching_ops = [op for op in consumers if op.name.startswith(name_scope)]
  assert len(matching_ops) <= 1, ('Op {} must only have one subscription '
                                  'op connected to it').format(tensor.op.name)
  if len(matching_ops) == 1:
    candidate_tensor = matching_ops[0].outputs[0]
    if _is_subscribed_identity(candidate_tensor):
      return _subscribe_extend(candidate_tensor, side_effects)

  return _subscribe_new(tensor, side_effects, control_cache)


@contextlib.contextmanager
def _preserve_control_flow_context(tensor):
  # [FUNCTION DOC]
  # HELPER: Context manager used by _scoped_subscribe.
  # Enters the control flow context (conditional or loop) of the given tensor's op, if any, for
  # the duration of the block so new ops are created inside it, and exits it afterwards even on
  # error. Yields nothing. Input: tensor whose context should be preserved.
  """Preserve the control flow context for the given tensor.

  Sets the graph context to the tensor's context so that side effect ops are
  added under the same context.

  This is needed when subscribing to tensors defined within a conditional
  block or a while loop. In these cases we need that the side-effect ops
  are created within the same control flow context as that of the tensor
  they are attached to.

  Args:
    tensor: tensor whose context should be preserved.

  Yields:
    None
  """

  # pylint: disable=protected-access
  context = tensor.op._get_control_flow_context()
  # pylint: enable=protected-access
  if context:
    context.Enter()
  try:
    yield
  finally:
    if context:
      context.Exit()


def _scoped_subscribe(tensor, side_effects, control_cache):
  # [FUNCTION DOC]
  # HELPER: Wrapper used by subscribe for each tensor. Runs the subscription dispatcher with the
  # tensor's device and control flow context active so side-effect ops are colocated and placed
  # in the same context. Inputs and return value are the same as for the dispatcher.
  """Helper method that subscribes a single tensor to a list of side_effects.

  This is a thin wrapper around `_subscribe` and ensures that the side effect
  ops are added within the same device and control flow context of the
  subscribed tensor.

  Args:
    tensor: The `tf.Tensor` to be subscribed.
    side_effects: List of side_effect functions, see subscribe for details.
    control_cache: `_ControlOutputCache` helper to get control_outputs faster.

  Returns:
    The modified replacement to the passed in tensor which triggers the side
    effects or the given tensor, if it was already been subscribed.
  """

  with ops.device(tensor.device):
    with _preserve_control_flow_context(tensor):
      return _subscribe(tensor, side_effects, control_cache)


def subscribe(tensors, side_effects):
  # [FUNCTION DOC]
  # Public entry point: attaches side-effect subgraphs to one or more tensors.
  # Inputs: tensors, a tensor or nested list, tuple, named tuple or mapping of tensors (and
  # variables); side_effects, a function or an iterable of functions, each taking a tensor and
  # returning a nonempty list of ops or tensors that become control dependencies. A single
  # callable is wrapped into a list. Returns a structure of the same shape holding identity
  # tensors that must be used in place of the originals. Raises a type error for unsupported
  # structure elements. Side effects: modifies the current graph, rewiring consumers of the
  # tensors to the identities; reuses and extends existing subscriptions on repeated calls.
  """Subscribe to a tensor.

  This method will attach side effect graphs to a given set
  of tensors. Set of tensors follows from session.run and supports
  single `Tensor`, `list`, nested `list`, `tuple`, `namedtuple`, or `dict`. It
  returns the tensors in the same passed in structure, but as clones with
  side effects applied. The supplied side effect graphs are specified
  as a constructor function which takes the target tensor and
  constructs a side effect graph and returns a list of ops that should
  be control dependencies on fetching the tensor. It will append
  'subscription' to the name scope of the tensor for every node in
  the side effect graph. These control dependencies are what trigger
  the side effects. Subscribe will construct the additions to your
  graph and return the created identity tensor downstream of the control
  dependencies. Use these tensors as you would normally in the rest of
  your tensorflow code. If a given tensor has already been subscribed or a
  tensor returned by a call to subscribe is passed, the previously created
  identity tensor will be reused and the side effect graphs will be added to
  the existing ones.

  Args:
    tensors: `Tensor` or set of tensors to subscribe to. Set of tensors format
      follows from `Session.run` and supports single `Tensor`, `list`, nested
      `list`, `tuple`, `namedtuple`, or `dict`.
    side_effects: Function(s) that takes a `Tensor`, construct a subgraph, and
      return a nonempty list of control dependencies. This can be a single
      function or list of functions.

  Returns:
    Subscribed tensors, which are identity copies of the passed in tensors
      in the same passed in structure, but the graph has been modified
      such that these are downstream of the control dependencies for
      the side effect graphs. Use these functionally equivalent tensors
      instead of the passed in tensors for further construction or running.
  """
  if not hasattr(side_effects, '__iter__'):
    side_effects = [side_effects]

  control_outputs = _ControlOutputCache()
  result = _recursive_apply(
      tensors, lambda t: _scoped_subscribe(t, side_effects, control_outputs))
  return result
