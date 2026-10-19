# [MODULE DOC]
# Classes and functions for processing debugger-decorated graph definitions (tfdbg). Provides
# helpers to parse node/tensor names and debug-node names, a depth-first input tracer, the
# DebugGraph class that indexes a partition GraphDef (inputs, control inputs, recipients, devices,
# op types, attributes) while hiding the Copy and Debug nodes inserted by the debugger, and a
# function reconstructing the original non-debug GraphDef. Used by the debug data readers. No
# import-time side effects beyond imports.
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
"""Classes and methods for processing debugger-decorated graphs."""
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging


def parse_node_or_tensor_name(name):
  # [FUNCTION DOC]
  # Splits a graph element name into node name and output slot. Parameter name: a node name or a
  # tensor name with a trailing colon and slot number. Returns a pair: node name and integer slot,
  # or the whole name and nothing/null when there is no colon or the name ends with a colon. A
  # non-numeric text after the last colon raises a value error from integer conversion.
  """Get the node name from a string that can be node or tensor name.

  Args:
    name: An input node name (e.g., "node_a") or tensor name (e.g.,
      "node_a:0"), as a str.

  Returns:
    1) The node name, as a str. If the input name is a tensor name, i.e.,
      consists of a colon, the final colon and the following output slot
      will be stripped.
    2) If the input name is a tensor name, the output slot, as an int. If
      the input name is not a tensor name, None.
  """

  if ":" in name and not name.endswith(":"):
    node_name = name[:name.rfind(":")]
    output_slot = int(name[name.rfind(":") + 1:])

    return node_name, output_slot
  else:
    return name, None


def get_node_name(element_name):
  # [FUNCTION DOC]
  # HELPER: returns just the node name of a node or tensor name (drops any output slot).
  node_name, _ = parse_node_or_tensor_name(element_name)
  return node_name


def get_output_slot(element_name):
  # [FUNCTION DOC]
  # Returns the integer output slot of a graph element name, or 0 if the name is a plain node name.
  """Get the output slot number from the name of a graph element.

  If element_name is a node name without output slot at the end, 0 will be
  assumed.

  Args:
    element_name: (`str`) name of the graph element in question.

  Returns:
    (`int`) output slot number.
  """
  _, output_slot = parse_node_or_tensor_name(element_name)
  return output_slot if output_slot is not None else 0


def is_copy_node(node_name):
  # [FUNCTION DOC]
  # Returns true when the node name starts with the prefix used for debugger-inserted Copy nodes.
  """Determine whether a node name is that of a debug Copy node.

  Such nodes are inserted by TensorFlow core upon request in
  RunOptions.debug_options.debug_tensor_watch_opts.

  Args:
    node_name: Name of the node.

  Returns:
    A bool indicating whether the input argument is the name of a debug Copy
    node.
  """
  return node_name.startswith("__copy_")


def is_debug_node(node_name):
  # [FUNCTION DOC]
  # Returns true when the node name starts with the prefix used for debugger-inserted debug nodes.
  """Determine whether a node name is that of a debug node.

  Such nodes are inserted by TensorFlow core upon request in
  RunOptions.debug_options.debug_tensor_watch_opts.

  Args:
    node_name: Name of the node.

  Returns:
    A bool indicating whether the input argument is the name of a debug node.
  """
  return node_name.startswith("__dbg_")


def parse_debug_node_name(node_name):
  # [FUNCTION DOC]
  # Parses a debug node name of the form prefix, watched tensor name, debug-op index and debug op
  # name, separated by underscores. Returns four values: watched node name, watched output slot
  # (int), debug op index (int) and debug op name. Raises a value error for a wrong prefix, fewer
  # than two underscores after the prefix, or a watched tensor name without exactly one colon.
  # Non-numeric index or slot text also raises a value error.
  """Parse the name of a debug node.

  Args:
    node_name: Name of the debug node.

  Returns:
    1. Name of the watched node, as a str.
    2. Output slot index of the watched tensor, as an int.
    3. Index of the debug node, as an int.
    4. Name of the debug op, as a str, e.g, "DebugIdentity".

  Raises:
    ValueError: If the input node name is not a valid debug node name.
  """
  prefix = "__dbg_"

  name = node_name
  if not name.startswith(prefix):
    raise ValueError("Invalid prefix in debug node name: '%s'" % node_name)

  name = name[len(prefix):]

  if name.count("_") < 2:
    raise ValueError("Invalid debug node name: '%s'" % node_name)

  debug_op = name[name.rindex("_") + 1:]
  name = name[:name.rindex("_")]

  debug_op_index = int(name[name.rindex("_") + 1:])
  name = name[:name.rindex("_")]

  if name.count(":") != 1:
    raise ValueError("Invalid tensor name in debug node name: '%s'" % node_name)

  watched_node_name = name[:name.index(":")]
  watched_output_slot = int(name[name.index(":") + 1:])

  return watched_node_name, watched_output_slot, debug_op_index, debug_op


class GraphTracingReachedDestination(Exception):
  # [CLASS DOC]
  # Exception signalling that a graph trace reached its configured destination node. Carries no
  # extra data; raised by the depth-first tracer to stop the search.
  pass


class DFSGraphTracer(object):
  # [CLASS DOC]
  # Traces the transitive inputs of a graph element by depth-first search over one or more input
  # adjacency maps. Holds the maps, nodes to skip, optional destination node, and the traversal
  # state: the visited nodes, the discovered inputs, the current depth and a list of the depth at
  # which each input was found. Use trace to run it, then read inputs and depth_list.
  """Graph input tracer using depth-first search."""

  def __init__(self,
               input_lists,
               skip_node_names=None,
               destination_node_name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the constructor raises when a stop node is reached and names a
    # nonexistent parameter; the constructor only stores state, and raising happens in trace.
    # Parameters: input_lists (list of mappings from node name to list of input names),
    # skip_node_names (optional list of nodes not to trace; the tracer does not handle nothing/null
    # here, so trace fails if omitted), destination_node_name (optional node at which tracing
    # raises the reached-destination exception). Initializes empty results and depth counters.
    """Constructor of _DFSGraphTracer.

    Args:
      input_lists: A list of dicts. Each dict is an adjacency (input) map from
        the recipient node name as the key and the list of input node names
        as the value.
      skip_node_names: Optional: a list of node names to skip tracing.
      destination_node_name: Optional: destination node name. If not `None`, it
        should be the name of a destination not as a str and the graph tracing
        will raise GraphTracingReachedDestination as soon as the node has been
        reached.

    Raises:
      GraphTracingReachedDestination: if stop_at_node_name is not None and
        the specified node is reached.
    """

    self._input_lists = input_lists
    self._skip_node_names = skip_node_names

    self._inputs = []
    self._visited_nodes = []
    self._depth_count = 0
    self._depth_list = []

    self._destination_node_name = destination_node_name

  def trace(self, graph_element_name):
    # [FUNCTION DOC]
    # KNOWN ISSUE: skip_node_names defaults to nothing/null in the constructor, but it is used in a
    # membership test here, so tracing raises a type error unless a list was supplied.
    # Recursively traces inputs of a node or tensor name. Increments the depth, raises the
    # reached-destination exception if the node is the destination, returns early for skipped or
    # already visited nodes, otherwise marks it visited and, for each input list containing it,
    # appends each not-yet-visited input to the inputs result with its depth and recurses. Decrements
    # the depth at the end. Returns nothing; results accumulate on the instance.
    """Trace inputs.

    Args:
      graph_element_name: Name of the node or an output tensor of the node, as a
        str.

    Raises:
      GraphTracingReachedDestination: if destination_node_name of this tracer
        object is not None and the specified node is reached.
    """
    self._depth_count += 1

    node_name = get_node_name(graph_element_name)
    if node_name == self._destination_node_name:
      raise GraphTracingReachedDestination()

    if node_name in self._skip_node_names:
      return
    if node_name in self._visited_nodes:
      return

    self._visited_nodes.append(node_name)

    for input_list in self._input_lists:
      if node_name not in input_list:
        continue
      for inp in input_list[node_name]:
        if get_node_name(inp) in self._visited_nodes:
          continue
        self._inputs.append(inp)
        self._depth_list.append(self._depth_count)
        self.trace(inp)

    self._depth_count -= 1

  def inputs(self):
    # [FUNCTION DOC]
    # Returns the list of input names discovered so far, in discovery order.
    return self._inputs

  def depth_list(self):
    # [FUNCTION DOC]
    # Returns the list of depths parallel to the inputs list (depth at which each was found).
    return self._depth_list


def _infer_device_name(graph_def):
  # [FUNCTION DOC]
  # HELPER: infers the device name of a partition GraphDef from the first node that has a non-empty
  # device. Returns it, or nothing/null after logging a warning when no node has a device.
  """Infer device name from a partition GraphDef."""
  device_name = None
  for node in graph_def.node:
    if node.device:
      device_name = node.device
      break
  if device_name is None:
    logging.warn(
        "Failed to infer device name from partition GraphDef: none of the "
        "nodes of the GraphDef has a non-empty device name.")
  return device_name


class DebugGraph(object):
  # [CLASS DOC]
  # Indexed view of a debugger-decorated partition GraphDef. Built from the GraphDef (and optional
  # device name, inferred if absent), it builds mappings by node name for attributes, inputs,
  # control inputs, recipients, control recipients, devices, op types, reversed reference-edge
  # inputs and reference-type output names, with debugger Copy and Debug nodes pruned out so the
  # structure matches the original graph. Also lazily builds the non-debug GraphDef.
  """Represents a debugger-decorated graph."""

  def __init__(self, debug_graph_def, device_name=None):
    # [FUNCTION DOC]
    # Builds the indexes. Parameters: debug_graph_def (the decorated partition GraphDef),
    # device_name (optional; inferred from the nodes when empty). Processes every node (skipping
    # debug nodes), prunes non-control and control edges that go through debug ops, removes Copy
    # nodes from the maps, and populates recipient maps. Raises a value error on duplicate node
    # names.
    self._debug_graph_def = debug_graph_def
    self._non_debug_graph_def = None

    self._node_attributes = {}
    self._node_inputs = {}
    self._node_reversed_ref_inputs = {}
    self._node_ctrl_inputs = {}
    self._node_recipients = {}
    self._node_ctrl_recipients = {}
    self._node_devices = {}
    self._node_op_types = {}
    self._copy_send_nodes = []
    self._ref_args = {}

    self._device_name = device_name
    if not self._device_name:
      self._device_name = _infer_device_name(debug_graph_def)

    for node in debug_graph_def.node:
      self._process_debug_graph_node(node)

    self._prune_non_control_edges_of_debug_ops()
    self._prune_control_edges_of_debug_ops()
    self._prune_nodes_from_input_and_recipient_maps(self._get_copy_nodes())

    self._populate_recipient_maps()

  def _process_debug_graph_node(self, node):
    # [FUNCTION DOC]
    # HELPER: registers one node in the index maps. Parameter node: a node definition. Debug nodes
    # are skipped entirely. Raises a value error for a duplicate node name. Otherwise records its
    # attributes, devices (falling back to the graph device), op type and reference output names,
    # splits inputs into regular and control (those starting with a caret), and remembers nodes
    # that are send/return ops fed by a Copy node.
    # KNOWN ISSUE: the comment in the debug-node branch says it parses the name and retrieves watch
    # information, but the code just returns without doing so.
    """Process a node from the debug GraphDef.

    Args:
      node: (NodeDef) A partition-graph node to be processed.

    Raises:
      ValueError: If duplicate node names are encountered.
    """
    if is_debug_node(node.name):
      # This is a debug node. Parse the node name and retrieve the
      # information about debug watches on tensors. But do not include
      # the node in the graph.
      return

    if node.name in self._node_inputs:
      raise ValueError("Duplicate node name on device %s: '%s'" %
                       (self._device_name, node.name))

    self._node_attributes[node.name] = node.attr

    self._node_inputs[node.name] = []
    self._node_ctrl_inputs[node.name] = []
    self._node_recipients[node.name] = []
    self._node_ctrl_recipients[node.name] = []

    if node.name not in self._node_devices:
      self._node_devices[node.name] = set()
    self._node_devices[node.name].add(
        node.device if node.device else self._device_name)
    self._node_op_types[node.name] = node.op
    self._ref_args[node.name] = self._get_ref_args(node)

    for inp in node.input:
      if is_copy_node(inp) and (node.op == "_Send" or node.op == "_Retval"):
        self._copy_send_nodes.append(node.name)

      if inp.startswith("^"):
        cinp = inp[1:]
        self._node_ctrl_inputs[node.name].append(cinp)
      else:
        self._node_inputs[node.name].append(inp)

  def _get_ref_args(self, node):
    # [FUNCTION DOC]
    # HELPER: lists the output names of a node that are reference-typed. Looks up the op
    # definition in the registry (returns an empty list if unknown); output i is named by the node
    # name for index 0, or the node name, a colon and the index otherwise.
    # KNOWN ISSUE: the docstring speaks of inputs, but the code inspects output arguments.
    """Determine whether an input of an op is ref-type.

    Args:
      node: A `NodeDef`.

    Returns:
      A list of the arg names (as strs) that are ref-type.
    """
    op_def = op_def_registry.get(node.op)
    if op_def is None:
      return []

    ref_args = []
    for i, output_arg in enumerate(op_def.output_arg):
      if output_arg.is_ref:
        arg_name = node.name if i == 0 else ("%s:%d" % (node.name, i))
        ref_args.append(arg_name)
    return ref_args

  def _get_copy_nodes(self):
    # [FUNCTION DOC]
    # HELPER: returns the list of node names in the input map that are debugger Copy nodes.
    """Find all Copy nodes in the loaded graph."""
    copy_nodes = []
    for node in self._node_inputs:
      if is_copy_node(node):
        copy_nodes.append(node)
    return copy_nodes

  def _prune_non_control_edges_of_debug_ops(self):
    # [FUNCTION DOC]
    # HELPER: for each regular input that is a Copy node, replaces it in place by the Copy node's
    # own first input, restoring the original input. Mutates the input map; returns nothing.
    """Prune (non-control) edges related to debug ops.

    Prune the Copy ops and associated _Send ops inserted by the debugger out
    from the non-control inputs and output recipients map. Replace the inputs
    and recipients with original ones.
    """
    for node in self._node_inputs:
      inputs = self._node_inputs[node]

      for i, inp in enumerate(inputs):
        if is_copy_node(inp):
          # Find the input to the Copy node, which should be the original
          # input to the node.
          orig_inp = self._node_inputs[inp][0]
          inputs[i] = orig_inp

  def _prune_control_edges_of_debug_ops(self):
    # [FUNCTION DOC]
    # HELPER: removes control inputs that are debug nodes from every node's control input list.
    # Mutates the control input map; returns nothing.
    """Prune control edges related to the debug ops."""
    for node in self._node_ctrl_inputs:
      ctrl_inputs = self._node_ctrl_inputs[node]
      debug_op_inputs = []
      for ctrl_inp in ctrl_inputs:
        if is_debug_node(ctrl_inp):
          debug_op_inputs.append(ctrl_inp)
      for debug_op_inp in debug_op_inputs:
        ctrl_inputs.remove(debug_op_inp)

  def _populate_recipient_maps(self):
    # [FUNCTION DOC]
    # HELPER: fills the recipient maps from the input maps. For each regular input (by node name)
    # adds the consuming node as recipient, and also to the reversed reference input map when the
    # input is a reference-typed output. For control inputs (skipping those from copy-send nodes)
    # adds control recipients. Mutates internal maps; returns nothing.
    """Populate the map from node name to recipient(s) of its output(s).

    This method also populates the input map based on reversed ref edges.
    """
    for node in self._node_inputs:
      inputs = self._node_inputs[node]
      for inp in inputs:
        inp = get_node_name(inp)
        if inp not in self._node_recipients:
          self._node_recipients[inp] = []
        self._node_recipients[inp].append(node)

        if inp in self._ref_args:
          if inp not in self._node_reversed_ref_inputs:
            self._node_reversed_ref_inputs[inp] = []
          self._node_reversed_ref_inputs[inp].append(node)

    for node in self._node_ctrl_inputs:
      ctrl_inputs = self._node_ctrl_inputs[node]
      for ctrl_inp in ctrl_inputs:
        if ctrl_inp in self._copy_send_nodes:
          continue

        if ctrl_inp not in self._node_ctrl_recipients:
          self._node_ctrl_recipients[ctrl_inp] = []
        self._node_ctrl_recipients[ctrl_inp].append(node)

  def _prune_nodes_from_input_and_recipient_maps(self, nodes_to_prune):
    # [FUNCTION DOC]
    # HELPER: deletes the given node names from the input, control input, recipient and control
    # recipient maps. Parameter nodes_to_prune: list of names; missing keys raise a lookup error.
    """Prune nodes out of input and recipient maps.

    Args:
      nodes_to_prune: (`list` of `str`) Names of the nodes to be pruned.
    """
    for node in nodes_to_prune:
      del self._node_inputs[node]
      del self._node_ctrl_inputs[node]
      del self._node_recipients[node]
      del self._node_ctrl_recipients[node]

  def _reconstruct_non_debug_graph_def(self):
    # [FUNCTION DOC]
    # HELPER: lazily builds the GraphDef without Copy and Debug nodes (cached; does nothing on later
    # calls). Copies each non-debug node and replaces its input list by the pruned regular inputs
    # followed by control inputs marked with a caret. Stores the result on the instance.
    """Reconstruct non-debug GraphDef.

    Non-debug GraphDef means the original GraphDef without the Copy* and Debug
    nodes inserted by the debugger.
    """
    if self._non_debug_graph_def:
      return

    self._non_debug_graph_def = graph_pb2.GraphDef()
    for node in self._debug_graph_def.node:
      if is_copy_node(node.name) or is_debug_node(node.name):
        continue

      new_node = self._non_debug_graph_def.node.add()
      new_node.CopyFrom(node)

      # Redo the list of inputs, because in _debug_graph_def, the list can
      # consist of Copy* and Debug* nodes inserted by the debugger. Those will
      # be replaced with the original inputs here.
      del new_node.input[:]
      for inp in self._node_inputs[node.name]:
        new_node.input.append(inp)
      for ctrl_inp in self._node_ctrl_inputs[node.name]:
        new_node.input.append("^" + ctrl_inp)

  @property
  def device_name(self):
    # [FUNCTION DOC]
    # Property: the device name of this partition graph (given or inferred; may be null).
    return self._device_name

  @property
  def debug_graph_def(self):
    # [FUNCTION DOC]
    # Property: the original debugger-decorated GraphDef passed in.
    """The debugger-decorated GraphDef."""
    return self._debug_graph_def

  @property
  def non_debug_graph_def(self):
    # [FUNCTION DOC]
    # Property: the reconstructed GraphDef without Copy and Debug nodes, built on first access.
    """The GraphDef without the Copy* and Debug* nodes added by the debugger."""
    self._reconstruct_non_debug_graph_def()
    return self._non_debug_graph_def

  @property
  def node_devices(self):
    # [FUNCTION DOC]
    # Property: mapping from node name to the set of device names for that node.
    return self._node_devices

  @property
  def node_op_types(self):
    # [FUNCTION DOC]
    # Property: mapping from node name to op type.
    return self._node_op_types

  @property
  def node_attributes(self):
    # [FUNCTION DOC]
    # Property: mapping from node name to the node's attribute mapping.
    return self._node_attributes

  @property
  def node_inputs(self):
    # [FUNCTION DOC]
    # Property: mapping from node name to list of its regular input names (debug ops pruned).
    return self._node_inputs

  @property
  def node_ctrl_inputs(self):
    # [FUNCTION DOC]
    # Property: mapping from node name to list of its control input node names.
    return self._node_ctrl_inputs

  @property
  def node_reversed_ref_inputs(self):
    # [FUNCTION DOC]
    # Property: mapping from a node producing a reference-typed output to the nodes consuming it,
    # which count as its inputs in the reversed direction.
    return self._node_reversed_ref_inputs

  @property
  def node_recipients(self):
    # [FUNCTION DOC]
    # Property: mapping from node name to list of nodes consuming its outputs through regular edges.
    return self._node_recipients

  @property
  def node_ctrl_recipients(self):
    # [FUNCTION DOC]
    # Property: mapping from node name to list of nodes depending on it through control edges.
    return self._node_ctrl_recipients


def reconstruct_non_debug_graph_def(debug_graph_def):
  # [FUNCTION DOC]
  # Strips debugger-inserted Copy and Debug nodes from a decorated partition GraphDef. Parameter
  # debug_graph_def: the decorated GraphDef. Returns the reconstructed GraphDef (built through a
  # DebugGraph). It may differ from the true original in names of runtime send/receive nodes and
  # in while-loop parallel iterations being 1. Raises a value error on duplicate node names.
  """Reconstruct original (non-debugger-decorated) partition GraphDef.

  This method strips the input `tf.compat.v1.GraphDef` of the Copy* and
  Debug*-type nodes inserted by the debugger.

  The reconstructed partition graph is identical to the original (i.e.,
    non-debugger-decorated) partition graph except in the following respects:
      1) The exact names of the runtime-inserted internal nodes may differ.
         These include _Send, _Recv, _HostSend, _HostRecv, _Retval ops.
      2) As a consequence of 1, the nodes that receive input directly from such
         send- and recv-type ops will have different input names.
      3) The parallel_iteration attribute of while-loop Enter ops are set to 1.

  Args:
    debug_graph_def: The debugger-decorated `tf.compat.v1.GraphDef`, with the
      debugger-inserted Copy* and Debug* nodes.

  Returns:
    The reconstructed `tf.compat.v1.GraphDef` stripped of the debugger-inserted
    nodes.
  """
  return DebugGraph(debug_graph_def).non_debug_graph_def
