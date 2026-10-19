# [MODULE DOC]
# TensorFlow Debugger (tfdbg) tools for retrieving gradient tensors. Provides GradientsDebugger,
# which inserts debug-gradient identity ops into a graph and, through a registered gradient
# function, records the gradient tensors computed by automatic differentiation, plus a helper that
# reads the values of such gradients from a debug dump directory object. Import-time side effects:
# creates a module-level registry of live debugger instances (keyed by unique id) and registers
# gradient functions for the two debug-gradient identity op types.
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
"""TensorFlow Debugger: Tools for debugging gradients."""

import re
import uuid

from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables

_GRADIENT_DEBUG_TAG = "gradient_debug_"

_gradient_debuggers = {}


def _tensor_to_grad_debug_op_name(tensor, grad_debugger_uuid):
  # [FUNCTION DOC]
  # HELPER: builds op names for GradientsDebugger.identify_gradient.
  # Takes a tensor and a debugger id string; returns a name made of the tensor's op name, its
  # output slot, a slash, a fixed gradient-debug tag and the id. Inverse of the parse helper below.
  op_name, slot = debug_graphs.parse_node_or_tensor_name(tensor.name)
  return "%s_%d/%s%s" % (op_name, slot, _GRADIENT_DEBUG_TAG, grad_debugger_uuid)


def _parse_grad_debug_op_name(op_name):
  # [FUNCTION DOC]
  # HELPER: inverse of the debug-op name builder, used by the gradient function.
  # Takes the name of a debug gradient op (must contain a slash and end with the gradient-debug tag
  # followed by an id; asserts otherwise). Returns a pair: the debugger id (truncated at any
  # underscore, e.g. a uniquifying suffix) and the name of the original tensor (op name plus colon
  # and slot, recovered from the second-to-last name component). Raises assertion errors for
  # malformed names; no side effects.
  """Parse the name of a debug gradient op.

  Args:
    op_name: the name of the debug gradient op.

  Returns:
    1) The UUID of the GradientsDebugger that created the debug gradient op.
    2) Name of the original tensor whose gradient is debugged by the debug
       gradient op.
  """
  name_items = op_name.split("/")
  assert len(name_items) > 1
  assert name_items[-1].startswith(_GRADIENT_DEBUG_TAG)

  grad_debugger_uuid = name_items[-1][len(_GRADIENT_DEBUG_TAG):]
  if "_" in grad_debugger_uuid:
    grad_debugger_uuid = grad_debugger_uuid[:grad_debugger_uuid.index("_")]
  orig_tensor_slot = int(name_items[-2][name_items[-2].rfind("_") + 1:])
  orig_base_op_name = name_items[-2][:name_items[-2].rfind("_")]
  orig_tensor_name = ("/".join(name_items[:-2] + [orig_base_op_name]) +
                      ":%d" % orig_tensor_slot)

  return grad_debugger_uuid, orig_tensor_name


class GradientsDebugger:
  # [CLASS DOC]
  # Lets users retrieve the gradient tensors that automatic differentiation creates for chosen
  # tensors. Each instance has a unique id, registers itself in the module-level registry, and holds
  # a mapping from x-tensor names to gradient tensors, the optional y tensor, the graph being
  # debugged, and a flag telling whether it is the active context (entered with a with-statement).
  # Gradient tensors are captured by identity ops inserted via identify_gradient or the watch
  # methods; the registered gradient function then reports back to the matching instance.
  """Gradients Debugger.

  Allows retrieval of gradient tensors created by TensorFlow's automatic
  differentiation algorithm, i.e., `tf.gradients` and optimizer classes that
  use it.
  """
  # TODO(cais): Add examples code in the doc string?

  def __init__(self, y_tensor=None):
    # [FUNCTION DOC]
    # Creates a debugger. Parameter y_tensor: optional tensor being differentiated; if given, its
    # graph becomes the debugger's graph. Generates a unique hexadecimal id and registers the
    # instance in the module-level registry (side effect), and initializes the empty mapping of
    # gradient tensors and the inactive-context flag.
    """Constructor of GradientsDebugger.

    Args:
      y_tensor: optional: the `tf.Tensor` to be differentiated, i.e., the tensor
        on the numerator of the differentiation.
    """

    self._uuid = uuid.uuid4().hex
    _gradient_debuggers[self._uuid] = self

    # A dict mapping x-tensor names to gradient tensor. x-tensor refers to the
    # independent tf.Tensor, i.e., the tensor on the denominator of the
    # differentiation.
    self._gradient_tensors = {}
    self._y_tensor = y_tensor

    self._graph = None
    if y_tensor:
      self._graph = y_tensor.graph

    self._is_active_context = False

  @property
  def y_tensor(self):
    # [FUNCTION DOC]
    # Property: returns the y tensor given at construction (may be nothing/null).
    return self._y_tensor

  @property
  def graph(self):
    # [FUNCTION DOC]
    # Property: returns the graph being debugged, taken from the y tensor or the first registered
    # gradient tensor; nothing/null if not yet known.
    return self._graph

  def __enter__(self):
    # [FUNCTION DOC]
    # Context-manager entry: marks this debugger as the active context so gradient registration is
    # accepted. Returns nothing (so a bound name in a with-statement would be null).
    self._is_active_context = True

  def __exit__(self, unused_type, unused_value, unused_traceback):
    # [FUNCTION DOC]
    # Context-manager exit: clears the active-context flag. Arguments are ignored; does not
    # suppress exceptions (returns nothing).
    self._is_active_context = False

  def identify_gradient(self, input_tensor):
    # [FUNCTION DOC]
    # Inserts a debug-gradient identity op (the reference variant for reference dtypes) on the given
    # tensor, named after the tensor and this debugger's id, so that gradients flowing through it
    # are registered with this debugger. Parameter input_tensor: the tensor to wrap.
    # Returns the forwarded identity tensor. Raises a value error if the graph already contains an
    # op with the chosen name (the created op got a different name). Adds an op to the graph.
    """Create a debug identity tensor that registers and forwards gradients.

    The side effect of this method is that when gradient tensor(s) are created
    with respect to the any paths that include the `input_tensor`, the gradient
    tensor(s) with respect to `input_tensor` will be registered with this
    this `GradientsDebugger` instance and can later be retrieved, with the
    methods `gradient_tensor` and `gradient_tensors`.

    Example:

    ```python
    x = tf.Variable(1.0)
    y = tf.add(x, x)

    grad_debugger = tf_debug.GradientsDebugger()
    debug_y = grad_debugger.identify_gradient(y)
    z = tf.square(debug_y)

    # Create a train op under the grad_debugger context.
    with grad_debugger:
      train_op = tf.compat.v1.train.GradientDescentOptimizer(z)

    # Now we can reflect through grad_debugger to get the gradient tensor
    # with respect to y.
    y_grad = grad_debugger.gradient_tensor(y)
    ```

    Args:
      input_tensor: the input `tf.Tensor` object whose related gradient tensors
        are to be registered with this `GradientsDebugger` instance when they
        are created, e.g., during `tf.gradients` calls or the construction
        of optimization (training) op that uses `tf.gradients`.

    Returns:
      A forwarded identity of `input_tensor`, as a `tf.Tensor`.

    Raises:
      ValueError: If an op with name that duplicates the gradient-debugging op
        already exists in the graph (highly unlikely).
    """
    # TODO(cais): Allow overriding gradient.
    # TODO(cais): Implement value_stack.
    grad_debug_op_name = _tensor_to_grad_debug_op_name(input_tensor, self._uuid)
    # pylint: disable=protected-access
    identity_op = (
        gen_array_ops.debug_gradient_ref_identity
        if input_tensor.dtype._is_ref_dtype else
        gen_array_ops.debug_gradient_identity)
    # pylint: enable=protected-access
    debug_grad_identity = identity_op(input_tensor, name=grad_debug_op_name)
    assert debug_grad_identity.dtype == input_tensor.dtype
    if debug_grad_identity.op.name != grad_debug_op_name:
      raise ValueError(
          "The graph already contains an op named %s" % grad_debug_op_name)
    return debug_grad_identity

  def watch_gradients_by_tensors(self, graph, tensors):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring refers to a method named for x-tensor names that does not exist
    # (the real one is the by-tensor-names method), and its example uses an undefined variable.
    # Watches gradients of the given tensor or variable (or list of them) in the graph. Builds an
    # anchored regular expression from the escaped tensor names and delegates to the by-names
    # method. Parameters: graph (the graph to modify), tensors (one object or a list).
    # Returns this debugger.
    """Watch gradient tensors by x-tensor(s).

    The side effect of this method is that when gradient tensor(s) are created
    with respect to the any paths that include the `x_tensor`s, the gradient
    tensor(s) with respect to the tensor will be registered with this
    this `GradientsDebugger` instance and can later be retrieved, with the
    methods `gradient_tensor` and `gradient_tensors`.

    Unlike the method `identify_gradient`, this method is used to retrieve
    gradient tensors after the construction of the forward subgraph has
    completed (but before the construction of the backward subgraph).

    This method is the same as `watch_gradients_by_x_tensor_names` except that
    the tensors are specified by the Python `tf.Tensor` or `tf.Variable`
    objects, instead by name patterns.

    Example:

    ```python
    x = tf.Variable(1.0)
    y = tf.add(x, x, name="y")
    z = tf.square(debug_y)

    # Create a train op under the grad_debugger context.
    grad_debugger = tf_debug.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensors(y):
      train_op = tf.compat.v1.train.GradientDescentOptimizer(z)

    # Now we can reflect through grad_debugger to get the gradient tensor
    # with respect to y.
    y_grad = grad_debugger.gradient_tensor(y)
    # or
    y_grad = grad_debugger.gradient_tensor("y:0")
    ```

    Args:
      graph: the `tf.Graph` to watch the gradients on.
      tensors: a `tf.Tensor` or `tf.Variable` object, or a list of such objects.

    Returns:
      The GradientsDebugger instance itself.
    """

    if not isinstance(tensors, list):
      tensors = [tensors]

    tensor_name_regex = []
    for tensor in tensors:
      tensor_name_regex.append(re.escape(tensor.name) + "$")
    tensor_name_regex = "(" + "|".join(tensor_name_regex) + ")"
    return self.watch_gradients_by_tensor_names(graph, tensor_name_regex)

  def watch_gradients_by_tensor_names(self, graph, tensor_name_regex):
    # [FUNCTION DOC]
    # Watches gradients of every tensor in the graph whose name matches a regular expression
    # (matched from the start of the name). For each match it inserts a debug-gradient identity op
    # and rewires every other consumer of the original tensor to read from the identity instead.
    # Parameters: graph (graph to modify), tensor_name_regex (pattern text). Returns this debugger.
    # Side effect: mutates the graph by adding ops and updating consumer inputs.
    """Watch gradient tensors by name(s) of the x-tensor(s).

    The side effect of this method is that when gradient tensor(s) are created
    with respect to the x-tensors, the gradient tensor(s) will be registered
    with this `GradientsDebugger` instance and can later be retrieved.

    Unlike the `identify_gradient` method, this method is used after the
    construction of the forward graph has completed. Unlike the
    `watch_gradients_by_tensor` method, this method does not use handles to the
    tensors of interest; it uses their names.

    This method is the same as `watch_gradients_by_tensors` except that the
    x-tensors are specified by name patterns, instead of `tf.Tensor` or
    `tf.Variable` objects.

    Example:

    ```python
    x = tf.Variable(1.0, name="x")
    y = tf.add(x, x, name="y")
    z = tf.square(debug_y)

    # Create a train op under the grad_debugger context.
    grad_debugger = tf_debug.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensor_names(r"(x|y):0$"):
      train_op = tf.compat.v1.train.GradientDescentOptimizer(z)

    # Now we can reflect through grad_debugger to get the gradient tensor
    # with respect to x and y.
    x_grad = grad_debugger.gradient_tensor("x:0")
    y_grad = grad_debugger.gradient_tensor("y:0")
    ```

    Args:
      graph: the `tf.Graph` to watch the gradients on.
      tensor_name_regex: the regular-expression pattern of the name(s) of the
        x-tensor(s) to watch. x-tensor refers to the tensors on the denominator
        of the differentiation.

    Returns:
      The GradientsDebugger instance itself.
    """
    tensor_name_pattern = re.compile(tensor_name_regex)
    with graph.as_default():
      for op in graph.get_operations():
        for output in op.outputs:
          if tensor_name_pattern.match(output.name):
            debug_op = self.identify_gradient(output)

            # Make a copy of output.consumers() since we'll modify the consumers
            # TODO(skyewm): this is unnecessary once the C API is enabled
            for consumer in list(output.consumers()):
              if consumer == debug_op.op:
                continue

              # Locate the slot index of the original input.
              for i, consumer_input in enumerate(consumer.inputs):
                if consumer_input == output:
                  consumer._update_input(i, debug_op)  # pylint: disable=protected-access
    return self

  def _check_same_graph(self, tensor):
    # [FUNCTION DOC]
    # HELPER: keeps the debugger bound to a single graph. If no graph is known yet, adopts the
    # tensor's graph; otherwise raises a value error when the tensor belongs to a different graph.
    if self._graph is None:
      self._graph = tensor.graph
    elif self._graph != tensor.graph:
      raise ValueError(
          "The graph of the value (%s) is not the same as the graph %s" %
          (tensor.graph, self._graph))

  def register_gradient_tensor(self,
                               x_tensor_name,
                               gradient_tensor):
    # [FUNCTION DOC]
    # Called by the registered gradient function to record a gradient tensor. Parameters:
    # x_tensor_name (name of the independent tensor), gradient_tensor (its gradient). The record is
    # stored only if this is the sole existing debugger or it is the active context; otherwise it
    # is silently ignored. Checks the graph matches (value error if not). Returns nothing.
    """Register the gradient tensor for an x-tensor.

    Args:
      x_tensor_name: (`str`) the name of the independent `tf.Tensor`, i.e.,
        the tensor on the denominator of the differentiation.
      gradient_tensor: the gradient `tf.Tensor`.
    """
    if len(_gradient_debuggers) == 1 or self._is_active_context:
      self._check_same_graph(gradient_tensor)
      self._gradient_tensors[x_tensor_name] = gradient_tensor

  def gradient_tensor(self, x_tensor):
    # [FUNCTION DOC]
    # Returns the gradient tensor recorded for an x-tensor given as a tensor, variable or name.
    # Raises a type error for other types (via the name helper) and a lookup error if no gradient
    # has been recorded for it.
    """Get the gradient tensor of an x-tensor.

    Args:
      x_tensor: (`tf.Tensor`, `tf.Variable` or `str`) The x-tensor object or its
        name. x-tensor refers to the independent `tf.Tensor`, i.e., the tensor
        on the denominator of the differentiation.

    Returns:
      If found, the gradient tensor.

    Raises:
      TypeError: If `x_tensor` is not a `tf.Tensor`, `tf.Variable` or `str`.
      LookupError: If the `x_tensor` has not been registered with a gradient
        tensor.
    """
    x_tensor_name = self._get_tensor_name(x_tensor)
    if x_tensor_name not in self._gradient_tensors:
      raise LookupError(
          "This GradientsDebugger has not received any gradient tensor for "
          "x-tensor %s" % x_tensor_name)
    return self._gradient_tensors[x_tensor_name]

  def gradient_tensors(self):
    # [FUNCTION DOC]
    # Returns the internal mapping from x-tensor names to gradient tensors (the live mapping, not a
    # copy).
    """Get the gradient tensors that this object is aware of.

    Returns:
      A dict mapping x-tensor names to gradient tensor objects. x-tensor refers
      to the tensors on the denominator of the differentation.
    """
    return self._gradient_tensors

  def _get_tensor_name(self, tensor):
    # [FUNCTION DOC]
    # HELPER: converts a tensor, variable or string into a tensor name. Returns the name; raises a
    # type error for any other type.
    if isinstance(tensor, (tensor_lib.Tensor, variables.Variable)):
      return tensor.name
    elif isinstance(tensor, str):
      return tensor
    else:
      raise TypeError(
          "x_tensor must be a str or tf.Tensor or tf.Variable, "
          "but instead has type %s" % type(tensor))


def clear_gradient_debuggers():
  # [FUNCTION DOC]
  # Empties the module-level registry of gradient debuggers (side effect on global state). Takes no
  # arguments and returns nothing.
  """Clear all globally registered gradient debuggers."""
  _gradient_debuggers.clear()


@ops.RegisterGradient("DebugGradientIdentity")
def _identify_gradient_grad(op, dy):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the debug-gradient identity op.
  # KNOWN ISSUE: the docstring calls it the gradient of the DebugIdentity op, but it is registered
  # for the debug-gradient identity op type.
  # Parses the op name to find the owning debugger and original tensor, registers the incoming
  # gradient with that debugger (looked up in the global registry; a missing id raises a lookup
  # error), and returns the gradient unchanged. Parameters: op (the op), dy (incoming gradient).
  """Gradient function for the DebugIdentity op."""
  # TODO(cais): Allow overriding gradient.
  grad_debugger_uuid, orig_tensor_name = _parse_grad_debug_op_name(op.name)
  grad_debugger = _gradient_debuggers[grad_debugger_uuid]
  grad_debugger.register_gradient_tensor(orig_tensor_name, dy)
  return dy


@ops.RegisterGradient("DebugGradientRefIdentity")
def _identify_gradient_grad_ref(op, dy):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the reference variant of the debug-gradient identity
  # op. Simply delegates to the non-reference gradient function and returns its result.
  # KNOWN ISSUE: the docstring says DebugIdentity op, but the registered op is the reference
  # debug-gradient identity.
  """Gradient function for the DebugIdentity op."""
  return _identify_gradient_grad(op, dy)


def gradient_values_from_dump(grad_debugger, x_tensor, dump):
  # [FUNCTION DOC]
  # Reads values of an x-tensor's gradient from a debug dump directory object. Parameters:
  # grad_debugger (a GradientsDebugger), x_tensor (tensor, variable or name), dump (the dump
  # directory object). If both the dump and debugger know a Python graph and they differ, raises a
  # value error. Looks up the gradient tensor (lookup/type errors propagate) and returns the list
  # of values dumped by the identity debug op for it; returns an empty list if the watch key does
  # not exist in the dump (gradient not executed).
  """Find gradient values from a `DebugDumpDir` object.

  Args:
    grad_debugger: the `tf_debug.GradientsDebugger` instance to be used.
    x_tensor: (`tf.Tensor`, `tf.Variable` or `str`) The x-tensor object or its
      name. x-tensor refers to the independent `tf.Tensor`, i.e., the tensor
      on the denominator of the differentiation.
    dump: A `tfdbg.DebugDumpDir` object.

  Returns:
    If this `GradientsDebugger` instance has the gradient tensor of `x_tensor`
      registered: a list of `numpy.ndarray` representing the value of the
      gradient tensor from `dump`. The list could be empty, if the gradient
      tensor is not executed in the `tf.Session.run()` call that generated
      the `dump`. The list could also contain multiple values of the gradient
      tensor, e.g., if gradient tensor is computed repeatedly in a
      `tf.while_loop` during the run that generated the `dump`.

  Raises:
    LookupError: If this `GradientsDebugger` instance does not have the
      gradient tensor of `x_tensor` registered.
    ValueError: If this `GradientsDebugger` has a `tf.Graph` object that
      does not match the `tf.Graph` object of the `dump`.
    TypeError: If `x_tensor` is not a `tf.Tensor`, `tf.Variable` or `str`.
  """
  # TODO(cais): Use this method in LocalCLIDebugWrapperSession to present the
  # gradient tensors to the TFDBG CLI.

  # If possible, verify that the Python graph of the dump and that of this
  # GradientsDebugger match.
  if (dump.python_graph and grad_debugger.graph and
      dump.python_graph != grad_debugger.graph):
    raise ValueError(
        "This GradientsDebugger instance has a graph (%s) that differs from "
        "the graph of the DebugDumpDir object (%s)." %
        (grad_debugger.graph, dump.python_graph))

  gradient_tensor = grad_debugger.gradient_tensor(x_tensor)
  node_name, output_slot = debug_graphs.parse_node_or_tensor_name(
      gradient_tensor.name)

  try:
    return dump.get_tensors(node_name, output_slot, "DebugIdentity")
  except debug_data.WatchKeyDoesNotExistInDebugDumpDirError:
    return []
