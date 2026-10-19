# [MODULE DOC]
# Utilities that freeze a TensorFlow graph by replacing variables with constant nodes holding the
# same values. It offers public entry points for ConcreteFunctions (eager and graph-mode variants,
# one also returning the intermediate graph definition) and for session-based version 1 graphs
# plus a deprecated version 1 wrapper. Internally it builds a wrapper graph of "convertible"
# objects (graph definition, library functions, nodes and specialised node kinds such as variable
# handles, variable reads, gathers, control flow and function-calling ops) connected by edges, and
# propagates a variable-to-constant conversion along those edges so that dtypes, shapes and ops of
# dependent nodes and called functions are rewritten consistently. Converter-data classes collect
# the variable values (from a function or a session) before conversion.
# Import-time side effects: only defines constants and classes and a deprecated/exported
# function registration for the version 1 API; no flags are defined.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Helpers to convert variables to constants in TensorFlow 2.0."""

import collections
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export


# Used in _FunctionConverterDataInGraph().
VAR_ASSIGN_COLLECTION = "extra_var_assign_ops"
_CONDITIONAL_OPS = set(["If", "StatelessIf"])
_LOOP_OPS = set(["While", "StatelessWhile"])
_CONTROL_FLOW_OPS = _CONDITIONAL_OPS.union(_LOOP_OPS)


class _TensorData(
    collections.namedtuple("_TensorData", ["numpy", "dtype", "index"])):
  # [CLASS DOC]
  # HELPER: immutable record used by the constant-conversion machinery in this module.
  # Named tuple with three fields: the numpy array holding the value of a variable or captured
  # tensor, its dtype as a TensorFlow data type enum number, and the index of the corresponding
  # captured function input (nothing when the value did not come from a function capture).
  # Has no extra per-instance storage. Created by the converter-data classes and consumed when
  # nodes are rewritten into constants.
  """Data about a tensor that was converted to a constant."""
  __slots__ = ()

  @property
  def dtype_attr(self):
    # [FUNCTION DOC]
    # HELPER: builds node attribute values for the constant-conversion code.
    # Property returning a node attribute protocol buffer whose type field is set to this
    # record's dtype enum. No inputs, no side effects, no exceptions.
    return attr_value_pb2.AttrValue(type=self.dtype)


class _EndPoint(collections.namedtuple("_EndPoint", ["convertible", "index"])):
  # [CLASS DOC]
  # HELPER: graph-edge building block for the constant-conversion machinery.
  # Immutable named tuple identifying one endpoint of an edge: the convertible object (node,
  # function or graph) and an integer index (an input position or output position, depending on
  # which side of the edge it sits). Used as source and destination of edge objects.
  """An endpoint in a graph."""
  __slots__ = ()

  def __str__(self):
    # [FUNCTION DOC]
    # HELPER: debugging/logging text for edges and endpoints.
    # Returns the text form of the endpoint: the convertible's text, then the index in square
    # brackets. No side effects.
    return "{}[{}]".format(self.convertible, self.index)


class _Edge(collections.namedtuple("_Edge", ["source", "destination"])):
  # [CLASS DOC]
  # HELPER: graph-edge building block for the constant-conversion machinery.
  # Immutable named tuple with a source endpoint and a destination endpoint, describing a
  # directed dependency between two convertibles. Stored in the outgoing edge lists of the
  # source convertible and followed when a conversion is propagated.
  """A directed graph edge."""
  __slots__ = ()

  def __str__(self):
    # [FUNCTION DOC]
    # HELPER: debugging/logging text for edges.
    # Returns the source text, an arrow, then the destination text. No side effects.
    return "{} -> {}".format(self.source, self.destination)


class _Convertible(object):
  # [CLASS DOC]
  # HELPER: abstract base for the wrapper objects used when freezing a graph.
  # Represents anything that can take part in variable-to-constant conversion (a graph
  # definition, a library function or a node). Holds the enclosing graph wrapper, the list of
  # outgoing edges and a lazily created "converted copy" that is modified during conversion.
  # Subclasses must implement the converted-copy accessor, the conversion step and edge creation.
  """An entity that can have variables converted to constants."""

  def __init__(self, enclosing_graph):
    # [FUNCTION DOC]
    # Stores the wrapper of the graph that contains this object (nothing for the graph wrapper
    # itself), starts with an empty outgoing edge list and no converted copy yet.
    self._enclosing_graph = enclosing_graph
    self._outgoing_edges = []
    self._converted_self = None

  def converted_self(self):
    # [FUNCTION DOC]
    # HELPER: abstract hook of the conversion machinery.
    # Abstract: subclasses return the copy of this object that lives in the converted graph and
    # is the one modified during conversion. This base version always raises a
    # not-implemented error.
    """A copy of this Convertible to be modified during conversion.

    Returns:
      Implementations should return the copied instance, which in turn should
      be contained in converted_enclosing_graph(). This instance is the one that
      will be modified during conversion. Its main use will be in the
      implementations of convert_variable_to_constant().
    """
    raise NotImplementedError

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # HELPER: abstract hook of the conversion machinery.
    # Abstract: subclasses rewrite themselves for an incoming edge whose source was turned into
    # a constant (incoming_edge) with the constant's value record (tensor_data) and propagate
    # to dependents. This base version always raises a not-implemented error.
    """Converts a variable in this Convertible and its dependencies.

    This method should make sure that a converted copy of itself is present in
    the converted graph, and that all Convertibles depending on this one also go
    through the same process.

    Args:
      incoming_edge: The graph edge into this Convertible that is being
        converted to a constant.
      tensor_data: The tensor representing the constant.
    """
    raise NotImplementedError

  def create_edges(self):
    # [FUNCTION DOC]
    # HELPER: abstract hook used while building the dependency graph.
    # Abstract: subclasses register, on the convertibles that feed them, the edges that
    # describe their dependencies. This base version always raises a not-implemented error.
    """Calls add_outgoing_edge for all edges known to this Convertible.

    This is used to build the graph dependencies, so that conversion of
    variables to constants can be properly propagated through the graph. Usually
    this method will call add_outgoing_edge() to all the Convertible inputs.
    """
    raise NotImplementedError

  def add_outgoing_edge(self, edge):
    # [FUNCTION DOC]
    # HELPER: edge bookkeeping for the conversion machinery.
    # Appends the given edge (whose source should be this object) to this object's outgoing
    # edge list. Mutates the list; returns nothing; no validation is performed.
    """Adds an outgoing edge to the Convertible's list of edges.

    Args:
      edge: The outgoing edge (its source should be 'self').
    """
    self._outgoing_edges.append(edge)

  @property
  def converted_enclosing_graph(self):
    # [FUNCTION DOC]
    # HELPER: shortcut used by convertibles during conversion.
    # Property returning the converted copy of the enclosing graph wrapper (created on first
    # use). Fails if there is no enclosing graph (for example on the graph wrapper itself).
    """The graph being converted."""
    return self._enclosing_graph.converted_self()

  @property
  def outgoing_edges(self):
    # [FUNCTION DOC]
    # HELPER: accessor for the conversion machinery.
    # Property returning the live list of edges that start at this object (not a copy).
    """The list of edges starting at this Convertible."""
    return self._outgoing_edges


class _Function(_Convertible):
  # [CLASS DOC]
  # HELPER: wrapper for one library function of a graph being frozen.
  # Wraps a function definition from the graph's function library and a mapping from node
  # names to node wrappers for the function body. Edges into a function go from the caller's
  # inputs to the function's inputs. Converting an argument changes the function signature type
  # and clears shape information, in a renamed copy of the function inside the converted graph.
  """A library function Convertible.

  Edges into functions are edges from node _inputs_ into function _inputs_:
  Functions get their input from their callers, not from node outputs, and the
  callers in turn get those values as inputs.
  """

  def __init__(self, function, enclosing_graph):
    # [FUNCTION DOC]
    # Inputs: function (a function definition protocol buffer) and enclosing_graph (the graph
    # wrapper that owns the library). Stores them and builds a mapping from node name to a node
    # wrapper (of the right specialised kind) for every node in the function body.
    super(_Function, self).__init__(enclosing_graph)
    self._function = function
    self._nodes = {
        n.name:
        _Node.new(node=n, function=self, enclosing_graph=enclosing_graph)
        for n in function.node_def
    }

  def __str__(self):
    # [FUNCTION DOC]
    # HELPER: debugging/logging text. Returns the function's signature name.
    return self.function.signature.name

  @property
  def function(self):
    # [FUNCTION DOC]
    # HELPER: accessor. Property returning the wrapped function definition.
    return self._function

  @property
  def nodes(self):
    # [FUNCTION DOC]
    # HELPER: accessor. Property returning the mapping from node name to node wrapper for the
    # nodes of this function body.
    return self._nodes

  def converted_self(self):
    # [FUNCTION DOC]
    # Returns the converted copy of this function. On first use it looks up the new unique name
    # for this function from the enclosing graph, asks the converted graph to rename its copy of
    # the function (mutating the converted graph's function mapping and the copy's signature
    # name) and caches that copy. Later calls return the cached copy.
    """The Function copy to be converted.

    The copy will be renamed according to the graph's converted_function_name
    map, to ensure the name does not match anything currently in TensorFlow's
    function cache.

    Returns:
      The function instance to be converted.
    """
    if self._converted_self is None:
      old_name = self.function.signature.name
      new_name = self._enclosing_graph.converted_function_names[old_name]
      self.converted_enclosing_graph.rename_function(old_name, new_name)
      self._converted_self = self.converted_enclosing_graph.functions[new_name]
    return self._converted_self

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # Converts one function argument (the destination index of incoming_edge) into a constant of
    # the type in tensor_data. First propagates the conversion along this function's outgoing
    # edges that start at that argument, then, in the converted copy, sets the argument's type
    # in the signature and marks the recorded input shape and per-argument output shape (when
    # present) as unknown rank with no dimensions. Mutates the converted function; returns
    # nothing.
    """Converts one function argument into a constant.

    Args:
      incoming_edge: The edge into the argument to be converted.
      tensor_data: The constant value.
    """
    index = incoming_edge.destination.index
    for edge in self.outgoing_edges:
      if edge.source.index == index:
        edge.destination.convertible.convert_variable_to_constant(
            edge, tensor_data)

    function = self.converted_self().function
    function.signature.input_arg[index].type = tensor_data.dtype
    # TODO(b/176982859): Find a more satisfying way to update shape information
    # than clearing it, or migrate users to a workflow that does not require
    # freezing.
    if "_input_shapes" in function.attr:
      function.attr["_input_shapes"].list.shape[index].unknown_rank = True
      del function.attr["_input_shapes"].list.shape[index].dim[:]
    arg_attrs = function.arg_attr[index].attr
    if "_output_shapes" in arg_attrs:
      arg_attrs["_output_shapes"].list.shape[0].unknown_rank = True
      del arg_attrs["_output_shapes"].list.shape[0].dim[:]

  def create_edges(self):
    # [FUNCTION DOC]
    # Asks every node wrapper of this function body to create its edges. Returns nothing; adds
    # edges to the convertibles that feed those nodes.
    for n in self._nodes.values():
      n.create_edges()


class _Node(_Convertible):
  # [CLASS DOC]
  # HELPER: wrapper for one node definition, either in the main graph or inside a function.
  # Holds the node definition and the function wrapper it belongs to (nothing for main-graph
  # nodes). The static factory picks a specialised subclass by operation type. The base class
  # does nothing on conversion; subclasses rewrite the node. Provides input resolution and
  # dtype updating helpers.
  """A Convertible NodeDef."""

  def __init__(self, node, function, enclosing_graph):
    # [FUNCTION DOC]
    # Inputs: node (a node definition), function (the function wrapper containing it, or nothing
    # for a main-graph node) and enclosing_graph (the graph wrapper). Stores them.
    super(_Node, self).__init__(enclosing_graph)
    self._node = node
    self._function = function

  def __str__(self):
    # [FUNCTION DOC]
    # HELPER: debugging/logging text. Returns the wrapped node's name.
    return self._node.name

  @staticmethod
  def new(node, function, enclosing_graph):
    # [FUNCTION DOC]
    # HELPER: factory used when wrapping graph and function nodes.
    # Static factory. Inputs: node definition, containing function wrapper (or nothing) and
    # enclosing graph wrapper. Returns an instance of the subclass matching the node's operation:
    # variable handles for variable, variable-handle and placeholder ops; case, merge, partitioned
    # call (stateless and stateful), variable read, resource gather, resource gather-nd, if-like,
    # while-like specialisations; intermediate for enter/exit/identity/next-iteration/switch ops;
    # and the plain base class otherwise. No side effects.
    """Creates a new _Node base on its operation type."""
    if node.op in ["VariableV2", "VarHandleOp", "Placeholder"]:
      return _VarHandle(node, function, enclosing_graph)
    elif node.op == "Case":
      return _Case(node, function, enclosing_graph)
    elif node.op == "Merge":
      return _Merge(node, function, enclosing_graph)
    elif node.op == "PartitionedCall":
      return _PartitionedCall(node, function, enclosing_graph)
    elif node.op == "StatefulPartitionedCall":
      return _PartitionedCall(node, function, enclosing_graph)
    elif node.op == "ReadVariableOp":
      return _ReadVariable(node, function, enclosing_graph)
    elif node.op == "ResourceGather":
      return _ResourceGather(node, function, enclosing_graph)
    elif node.op == "ResourceGatherNd":
      return _ResourceGatherNd(node, function, enclosing_graph)
    elif node.op in ["If", "StatelessIf"]:
      return _If(node, function, enclosing_graph)
    elif node.op in ["While", "StatelessWhile"]:
      return _While(node, function, enclosing_graph)
    elif node.op in [
        "Enter", "Exit", "Identity", "NextIteration", "Switch", "_SwitchN"]:
      return _Intermediate(node, function, enclosing_graph)
    else:
      return _Node(node, function, enclosing_graph)

  @property
  def node(self):
    # [FUNCTION DOC]
    # HELPER: accessor. Property returning the wrapped node definition.
    return self._node

  @property
  def container(self):
    # [FUNCTION DOC]
    # HELPER: accessor. Property returning the function definition holding this node when the
    # node is inside a function, otherwise the graph definition of the enclosing graph.
    """The node container (either a graph or a function)."""
    if self._function is not None:
      return self._function.function
    return self._enclosing_graph.graph_def

  def converted_self(self):
    # [FUNCTION DOC]
    # Returns the wrapper of this node inside the converted graph or converted function (looked
    # up by node name in the converted copy of its source), caching it on first use. Subclasses
    # call this through the parent class to obtain the node they are going to modify.
    """The NodeDef to be converted.

    Returns:
      The NodeDef to be converted, which can come from either a graph for a
      function. Derived classes should call this (via 'super') to make sure the
      node is retrieved from the right place.
    """
    if self._converted_self is None:
      source = self._function or self._enclosing_graph
      self._converted_self = source.converted_self().nodes[self._node.name]
    return self._converted_self

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # Default conversion step for ordinary nodes: intentionally does nothing (arguments are
    # ignored). Subclasses override it to rewrite themselves.
    pass

  def create_edges(self):
    # [FUNCTION DOC]
    # Registers edges for this node: for each data input (control inputs, whose names start with
    # a caret, are skipped) it resolves the input name to its source endpoint and adds an
    # outgoing edge on the source's convertible going from that source to this node's input
    # position. Mutates the source convertibles' edge lists; returns nothing.
    for index, name in enumerate(self._node.input):
      # Discard edges from control inputs.
      if name[0] == "^":
        continue
      source = self.resolve_input(name)
      source.convertible.add_outgoing_edge(
          _Edge(source, _EndPoint(self, index)))

  def resolve_input(self, input_name):
    # [FUNCTION DOC]
    # HELPER: input-name parsing for edge creation.
    # Input: input_name, a node input string that may start with a caret and may end with a
    # colon and a numeric output index (a missing index means output zero). Returns an endpoint
    # (convertible plus output index): for main-graph nodes the graph node of that name; for
    # nodes in a function, the function's own node when the index is nonzero or the name is a
    # node of the function, otherwise the function argument with that name (its position in the
    # signature). Raises a key error or value error (from lookup) when the name is unknown.
    """Resolves an input into its _EndPoint.

    A NodeDef's input name can refer to either global NodeDefs (in the
    GraphDef's node list), a NodeDef in a function's node list, or a Function
    (in the GraphDef's function library). The name can also carry semantic
    information, depending on whether it starts with "^". This method handles
    all that logic in order to find the object to which the input name refers
    to.

    Args:
      input_name: The input name to resolve.

    Returns:
      The object referred to by 'input_name'.
    """

    # The logic below oversimplifies the semantics, but is good enough for the
    # purposes of converting to constants. The introduction of new types of
    # operations may change this, forcing the code to be more generic.
    #
    # In particular, we are assuming that the lack of an index suffix means
    # ":0", when it could mean "all the outputs of a node." This works now
    # because converting to constants relies very little on output types, and
    # when it does it specializes its treatment in dedicated classes.
    name_elts = input_name.split(":")
    source_name = name_elts[0]
    if source_name[0] == "^":
      source_name = source_name[1:]
    source_index = 0
    if len(name_elts) > 1 and name_elts[-1].isnumeric():
      source_index = int(name_elts[-1])

    if self._function is None:
      return _EndPoint(self._enclosing_graph.nodes[source_name], source_index)

    if source_index != 0 or source_name in self._function.nodes:
      return _EndPoint(self._function.nodes[source_name], source_index)

    inputs = [i.name for i in self._function.function.signature.input_arg]
    return _EndPoint(self._function, inputs.index(source_name))

  def update_dtype(self, attr_name, index, dtype):
    # [FUNCTION DOC]
    # HELPER: type rewriting for converted nodes.
    # Inputs: attr_name (node attribute holding types), index (which type entry), dtype (new
    # type enum). If the attribute is a list with enough entries, or a single type and the index
    # is zero, it is overwritten in place. Otherwise raises a value error stating the index is out
    # of range for that attribute. Mutates the wrapped node; returns nothing.
    """Changes the type of a given input.

    Args:
      attr_name: The NodeDef attribute containing the type to change.
      index: The index of the input type to change.
      dtype: The type to change to.
    """
    attr = self._node.attr[attr_name]
    num_types = 0
    # Check for various 'oneof' possibilities, and update the type if
    # index in range.
    if attr.HasField("list"):
      types = attr.list.type
      num_types = len(types)
      if num_types > index:
        types[index] = dtype
        return
    elif attr.HasField("type"):
      num_types = 1
      if index == 0:
        attr.type = dtype
        return
    raise ValueError(f"`index` {index:d} is out of range for "
                     f"node({self._node.name}).attr({attr_name}), which has "
                     f"{num_types:d} elements.")


class _Intermediate(_Node):
  # [CLASS DOC]
  # HELPER: node specialisation for pass-through ops (enter, exit, identity, next-iteration,
  # switch). Their output type follows their input, so conversion retypes them and forwards the
  # conversion to their dependents.
  """Specialization of _Node to intermediate ops."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # In the converted copy, sets the "T" type attribute at the incoming edge's destination
    # index to the constant's dtype, deletes any recorded output shapes attribute, and then
    # propagates the conversion along every outgoing edge. Mutates the converted node; returns
    # nothing.
    node = self.converted_self()
    node.update_dtype("T", incoming_edge.destination.index, tensor_data.dtype)
    if "_output_shapes" in node.node.attr:
      del node.node.attr["_output_shapes"]
    for edge in self.outgoing_edges:
      edge.destination.convertible.convert_variable_to_constant(
          edge, tensor_data)


class _Merge(_Node):
  # [CLASS DOC]
  # HELPER: node specialisation for merge ops, which share one type across all inputs.
  """Specialization of _Node to Merge ops."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the replacement destination is built as an edge wrapping an edge (the code
    # nests an edge object where an endpoint is expected) and the parent implementation does
    # nothing, so this method has no real effect and its comment about changing all inputs at once
    # is not what actually happens.
    # Intended to retype all merge inputs at once: forwards a rebuilt edge (same source, input
    # index replaced by zero) and the constant data to the parent class. Returns nothing.
    # The Merge operation has a single type for all its inputs, the number of
    # which is reflected in the "N" attribute. For the time being, we assume
    # that unilaterally changing all of them at once is ok.
    super(_Merge, self).convert_variable_to_constant(
        _Edge(incoming_edge.source,
              _Edge(incoming_edge.destination.convertible, 0)), tensor_data)


class _VarHandle(_Node):
  # [CLASS DOC]
  # HELPER: node specialisation for variable handles (variable, variable-handle and placeholder
  # ops); these are the nodes that actually become constants.
  """Specialization of _Node to VarHandleOp."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # Turns the converted copy of this node into a constant node: builds a tensor value from the
    # numpy data and dtype in tensor_data, clears the node, keeps its name, sets the operation to
    # constant, the dtype attribute and the value attribute, then propagates the conversion along
    # all outgoing edges. Mutates the converted node; returns nothing.
    tensor_proto = tensor_util.make_tensor_proto(tensor_data.numpy,
                                                 tensor_data.dtype,
                                                 tensor_data.numpy.shape)

    node = self.converted_self().node
    node.Clear()
    node.name = self._node.name
    node.op = "Const"
    node.attr["dtype"].CopyFrom(tensor_data.dtype_attr)
    node.attr["value"].tensor.CopyFrom(tensor_proto)

    for edge in self.outgoing_edges:
      edge.destination.convertible.convert_variable_to_constant(
          edge, tensor_data)


class _ResourceGather(_Node):
  # [CLASS DOC]
  # HELPER: node specialisation that rewrites resource gather ops into plain gather ops.
  """Specialization of _Node to ResourceGather."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # Rewrites a resource gather into a version-2 gather on the converted graph. Does nothing
    # when the node is inside a function. Raises a value error if the batch dimensions attribute
    # is not zero. Otherwise adds (or reuses) a constant axis node named after this node with an
    # axis suffix holding the batch dimension value, clears the converted node, and makes it a
    # gather with the original two inputs plus the axis, copying element and index types and any
    # class attribute. The tensor_data argument is not used. Mutates the converted graph.
    # We currently skip the conversion if this is inside a function.
    if self._function is not None:
      return
    if self._node.attr["batch_dims"].i != 0:
      raise ValueError("batch_dims must be 0 for freeze_graph, but got "
                       f"node({self._node.name}).attr('batch_dims') = "
                       f"{self._node.attr['batch_dims'].i}.")
    axis_node_name = self._node.name + "/axis"
    axis_dtype = self._node.attr["Tindices"]
    axis_data = np.array(self._node.attr["batch_dims"].i)
    converted_graph = self._enclosing_graph.converted_self()
    # Add Const axis node, or get it if it exists to avoid duplicates.
    if axis_node_name not in converted_graph.nodes:
      converted_graph.nodes[axis_node_name] = _Node.new(
          node=converted_graph.graph_def.node.add(),
          function=self._function,
          enclosing_graph=converted_graph)
    output_axis_node = converted_graph.nodes[axis_node_name].node
    output_axis_node.name = axis_node_name
    output_axis_node.op = "Const"
    output_axis_node.attr["dtype"].CopyFrom(axis_dtype)
    tensor = tensor_util.make_tensor_proto(
        axis_data, dtype=axis_dtype.type, shape=axis_data.shape)
    output_axis_node.attr["value"].tensor.CopyFrom(tensor)

    output_node = self.converted_self().node
    output_node.Clear()
    output_node.name = self._node.name
    output_node.op = "GatherV2"
    output_node.input.extend(
        [self._node.input[0], self._node.input[1], axis_node_name])
    output_node.attr["Tparams"].CopyFrom(self._node.attr["dtype"])
    output_node.attr["Tindices"].CopyFrom(self._node.attr["Tindices"])
    output_node.attr["Taxis"].CopyFrom(axis_dtype)
    if "_class" in self._node.attr:
      output_node.attr["_class"].CopyFrom(self._node.attr["_class"])


class _ResourceGatherNd(_Node):
  # [CLASS DOC]
  # HELPER: node specialisation that rewrites resource gather-nd ops into plain gather-nd ops.
  """Specialization of _Node to ResourceGatherNd."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # Clears the converted node and turns it into a gather-nd op that keeps the original name and
    # first two inputs, copying the element type, index type and any class attribute from the
    # original. Arguments are not used. Mutates the converted node; returns nothing; does not
    # propagate further.
    output_node = self.converted_self().node
    output_node.Clear()
    output_node.name = self._node.name
    output_node.op = "GatherNd"
    output_node.input.extend([self._node.input[0], self._node.input[1]])
    output_node.attr["Tparams"].CopyFrom(self._node.attr["dtype"])
    output_node.attr["Tindices"].CopyFrom(self._node.attr["Tindices"])
    if "_class" in self._node.attr:
      output_node.attr["_class"].CopyFrom(self._node.attr["_class"])


class _ReadVariable(_Node):
  # [CLASS DOC]
  # HELPER: node specialisation that rewrites variable-read ops into identity ops.
  """Specialization of _Node to ReadVariableOp."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # Clears the converted node and makes it an identity op with the original name and first
    # input, copying the dtype into the type attribute and any class attribute. When the node is
    # inside a function, dependents that refer to its output with a value suffix are changed to
    # use an output suffix instead. Arguments other than the edge target are unused. Mutates the
    # converted graph; returns nothing.
    node = self.converted_self().node
    node.Clear()
    node.name = self._node.name
    node.op = "Identity"

    node.input.append(self._node.input[0])
    node.attr["T"].CopyFrom(self._node.attr["dtype"])
    if "_class" in self._node.attr:
      node.attr["_class"].CopyFrom(self._node.attr["_class"])

    # If the ReadVariableOp is part of a function, then every node having the
    # ReadVariableOp one as its input will refer to it using a ":value"
    # syntax. We need to change that to ":output".
    if self._function is not None:
      for edge in self.outgoing_edges:
        index = edge.destination.index
        dest = edge.destination.convertible.converted_self()
        if isinstance(dest, _Node):
          input_name_parts = dest.node.input[index].split(":")
          if len(input_name_parts) > 1 and input_name_parts[1] == "value":
            input_name_parts[1] = "output"
            dest.node.input[index] = ":".join(input_name_parts)


class _FunctionCaller(_Node):
  # [CLASS DOC]
  # HELPER: base for node wrappers whose operation calls library functions (if, case, while and
  # partitioned calls). Stores which node input is the first one passed to functions, which
  # attribute lists input types, and which attributes name the called functions. Conversion
  # forwards into the called functions and retypes the call node's input types.
  """A base class for Convertibles that reference functions."""

  def __init__(self, node, function, enclosing_graph, first_function_input,
               type_attribute, function_attributes):
    # [FUNCTION DOC]
    # Passes node, function and enclosing_graph to the parent and stores first_function_input
    # (index of the first node input that maps to function inputs), type_attribute (name of the
    # attribute listing input types) and function_attributes (names of attributes referencing
    # functions). No exceptions.
    """Initializes a _FunctionCaller.

    Args:
      node: As in _Node.
      function: As in _Node.
      enclosing_graph: As in _Node.
      first_function_input: The index of the first NodeDef input that is tied to
        the function inputs. It is assumed that the rest of the NodeDef inputs
        map one to one to function inputs.
      type_attribute: The name of the NodeDef attribute that defines the input
        types. It is assumed that the types listed here map one-to-one with the
        function inputs (that is, they do _not_ specify types for inputs that
        are not passed to functions).
      function_attributes: The names of the NodeDef attributes containing
        references to functions.
    """
    super(_FunctionCaller, self).__init__(node, function, enclosing_graph)
    self._first_function_input = first_function_input
    self._type_attribute = type_attribute
    self._function_attributes = function_attributes

  def converted_self(self):
    # [FUNCTION DOC]
    # Returns the converted copy of this node. On first use, gets it from the parent and rewrites
    # every referenced function name (single function attribute or list of functions) that
    # belongs to a converted function to its new frozen name. Mutates the converted node.
    if self._converted_self is None:
      node = super(_FunctionCaller, self).converted_self().node
      converted_names = self._enclosing_graph.converted_function_names
      for attr_name in self._function_attributes:
        attr = node.attr[attr_name]
        if attr.HasField(
            "func") and self._enclosing_graph.is_converted_function(
                attr.func.name):
          attr.func.name = converted_names[attr.func.name]
        elif attr.HasField("list"):
          for func in attr.list.func:
            if self._enclosing_graph.is_converted_function(func.name):
              func.name = converted_names[func.name]
    return self._converted_self

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # KNOWN ISSUE: only edges into called functions are followed; edges into other nodes are
    # deliberately skipped (documented limitation in the comments), so outputs mapped by the body
    # may keep stale types.
    # Propagates the conversion of the input at the edge's destination index to each called
    # function reached by an outgoing edge from that input, then, if that input is one of the
    # inputs passed to functions, retypes the matching entry of the input-type attribute in the
    # converted node. Mutates the converted node and functions; returns nothing.
    index = incoming_edge.destination.index
    # The loop below is reasonable but not correct in general:
    # The outgoing edges going into the functions are correct, because the
    # inputs map to the function inputs. But the edges going into other nodes do
    # not take into account the logic of the body function, which may do
    # arbitrary things to the node's output:
    #
    #   while x < 0:
    #     return y
    #
    # In this case, the node's ":0" output may map to its ":1 input". For the
    # time being, then, we only process edges into functions.
    for edge in self.outgoing_edges:
      dest = edge.destination.convertible
      if edge.source.index == index and isinstance(dest, _Function):
        dest.convert_variable_to_constant(edge, tensor_data)

    node = self.converted_self()
    if index >= self._first_function_input:
      node.update_dtype(self._type_attribute,
                        index - self._first_function_input, tensor_data.dtype)

  def create_edges(self):
    # [FUNCTION DOC]
    # Calls the parent to create ordinary input edges, then for every function referenced by the
    # configured function attributes (a single function or a list), adds outgoing edges from each
    # call input that is passed to the function (offset by the first function input) to the
    # corresponding function input. Looks functions up in the graph library, so a missing name
    # raises a key error. Returns nothing.
    """Creates edges related to a function caller.

    Edges from a function caller to its called functions are always edges from
    _inputs_ to _inputs_: a FunctionDef input is given by the caller, based on
    its own inputs.
    """
    super(_FunctionCaller, self).create_edges()
    for attr_name in self._function_attributes:
      attr = self._node.attr[attr_name]
      if attr.HasField("func"):
        function = self._enclosing_graph.functions[attr.func.name]
        for index in range(len(self._node.input) - self._first_function_input):
          self.add_outgoing_edge(
              _Edge(
                  _EndPoint(self, index + self._first_function_input),
                  _EndPoint(function, index)))
      elif attr.HasField("list"):
        for func in attr.list.func:
          function = self._enclosing_graph.functions[func.name]
          for index in range(
              len(self._node.input) - self._first_function_input):
            self.add_outgoing_edge(
                _Edge(
                    _EndPoint(self, index + self._first_function_input),
                    _EndPoint(function, index)))


class _If(_FunctionCaller):
  # [CLASS DOC]
  # HELPER: function-caller specialisation for if and stateless-if ops: functions are in the
  # then and else branch attributes, the first input (condition) is not passed to functions and
  # the input types live in the input-types attribute.
  """Specialization of _Node to If-like operations."""

  def __init__(self, node, function, enclosing_graph):
    # [FUNCTION DOC]
    # Initialises the function-caller base with first function input one, the input-types
    # attribute, and the then-branch and else-branch function attributes.
    super(_If, self).__init__(
        node,
        function,
        enclosing_graph,
        first_function_input=1,
        type_attribute="Tin",
        function_attributes=["then_branch", "else_branch"])


class _Case(_FunctionCaller):
  # [CLASS DOC]
  # HELPER: function-caller specialisation for case ops: functions are in the branches list
  # attribute and the first input (branch index) is not passed to functions.
  """Specialization of _Node to Case-like operations."""

  def __init__(self, node, function, enclosing_graph):
    # [FUNCTION DOC]
    # Initialises the function-caller base with first function input one, the input-types
    # attribute, and the branches function attribute.
    super(_Case, self).__init__(
        node,
        function,
        enclosing_graph,
        first_function_input=1,
        type_attribute="Tin",
        function_attributes=["branches"])


class _PartitionedCall(_FunctionCaller):
  # [CLASS DOC]
  # HELPER: function-caller specialisation for partitioned-call ops (stateless and stateful):
  # every input is passed to the single function named by the function attribute.
  """Specialization of _Node to PartitionedCall-like operations."""

  def __init__(self, node, function, enclosing_graph):
    # [FUNCTION DOC]
    # Initialises the function-caller base with first function input zero, the input-types
    # attribute, and the single function attribute.
    super(_PartitionedCall, self).__init__(
        node,
        function,
        enclosing_graph,
        first_function_input=0,
        type_attribute="Tin",
        function_attributes=["f"])


class _While(_FunctionCaller):
  # [CLASS DOC]
  # HELPER: function-caller specialisation for while and stateless-while ops: every input is a
  # loop variable, the types live in the type attribute, and the body and condition functions
  # are referenced by function attributes. Also updates output shapes and the body output type.
  """Specialization of _Node to While-like operations."""

  def __init__(self, node, function, enclosing_graph):
    # [FUNCTION DOC]
    # Initialises the function-caller base with first function input zero, the single-type
    # attribute, and the body and condition function attributes.
    super(_While, self).__init__(
        node,
        function,
        enclosing_graph,
        first_function_input=0,
        type_attribute="T",
        function_attributes=["body", "cond"])

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # [FUNCTION DOC]
    # Runs the function-caller conversion, then, if the converted node records output shapes,
    # replaces the shape at the converted input's index with the shape of the constant's array.
    # Finally sets the output type of the converted body function at the same index to the
    # constant's dtype (loop variables keep their type across iterations). Mutates the converted
    # node and body function; returns nothing. Raises a key error if the body is not in the
    # library.
    super(_While, self).convert_variable_to_constant(incoming_edge, tensor_data)
    node = self.converted_self()
    if node.node.attr["output_shapes"].list.shape:
      node.node.attr["output_shapes"].list.shape[
          incoming_edge.destination.index].CopyFrom(
              tensor_shape_pb2.TensorShapeProto(dim=[
                  tensor_shape_pb2.TensorShapeProto.Dim(size=dim)
                  for dim in tensor_data.numpy.shape
              ]))

    # The while's body inputs and outputs have the same type, so here we can go
    # ahead and change that function's output type.
    body_name = self._node.attr["body"].func.name
    body = self._enclosing_graph.functions[body_name].converted_self().function
    body.signature.output_arg[
        incoming_edge.destination.index].type = tensor_data.dtype


class _GraphDef(_Convertible):
  # [CLASS DOC]
  # HELPER: wrapper for a whole graph definition used by the freezing machinery.
  # Holds the graph definition, a mapping from node names to node wrappers, a mapping from
  # function names to function wrappers, and the lazily computed mapping of old to new function
  # names. Creating it wraps all nodes and functions and builds all edges. Its converted copy is
  # another instance built from a copy of the graph definition.
  """A convertible GraphDef."""

  def __init__(self, graph_def):
    # [FUNCTION DOC]
    # Input: graph_def, a graph definition protocol buffer. Calls the parent with no enclosing
    # graph, wraps every node (specialised by op type) and every library function, then creates
    # all edges and leaves the converted-function-name mapping unset. Mutates nothing external.
    super(_GraphDef, self).__init__(enclosing_graph=None)
    self._graph_def = graph_def
    self._nodes = {
        n.name: _Node.new(node=n, function=None, enclosing_graph=self)
        for n in graph_def.node
    }
    self._functions = {
        f.signature.name: _Function(f, enclosing_graph=self)
        for f in graph_def.library.function
    }
    self.create_edges()
    self._converted_function_names = None

  @property
  def graph_def(self):
    # [FUNCTION DOC]
    # HELPER: accessor. Property returning the wrapped graph definition.
    return self._graph_def

  @property
  def nodes(self):
    # [FUNCTION DOC]
    # HELPER: accessor. Property returning the mapping from node name to node wrapper for the
    # main graph.
    return self._nodes

  @property
  def functions(self):
    # [FUNCTION DOC]
    # HELPER: accessor. Property returning the mapping from function name to function wrapper.
    return self._functions

  @property
  def converted_function_names(self):
    # [FUNCTION DOC]
    # Property returning, and caching on first use, a mapping from each library function name to
    # a new unique name made of its base name (a trailing numeric suffix after an underscore is
    # stripped), the word frozen and a fresh unique id. Names are processed in sorted order of
    # their parsed numeric id. Side effect: consumes unique ids from the global id counter.
    """Map from original to new function names.

    In order to avoid conflicts (two functions with the same name, one converted
    and one not), we need to change the name of every converted function to
    something that is hopefully unique.

    Returns:
      Map from original to new suggested function names.
    """
    if self._converted_function_names is None:
      parsed_names = []  # List of (id, base_name, original_name)
      for name in self.functions:
        elements = name.rsplit("_", 1)
        if len(elements) == 2 and elements[1].isnumeric():
          parsed_names.append((int(elements[1]), elements[0], name))
        else:
          parsed_names.append((-1, name, name))
      self._converted_function_names = {
          name: "{}_frozen_{}".format(base_name, ops.uid())
          for (_, base_name, name) in sorted(parsed_names)
      }

    return self._converted_function_names

  def rename_function(self, old_name, new_name):
    # [FUNCTION DOC]
    # HELPER: used by function wrappers when creating their converted copy.
    # Inputs: old_name and new_name of a library function. Removes the wrapper registered under
    # the old name (raises a key error if missing), sets its signature name to the new name and
    # registers it under the new name. Mutates the function mapping and the function definition.
    func = self.functions.pop(old_name)
    func.function.signature.name = new_name
    self.functions[new_name] = func

  def is_converted_function(self, function_name):
    # [FUNCTION DOC]
    # HELPER: used by function callers when rewriting function references.
    # Input: function_name. Returns true when the name is not yet present in the converted
    # graph's functions (it was renamed away) and has a new name in the converted-name mapping.
    # Triggers creation of the converted graph copy and the name mapping as a side effect.
    # Only converted functions will be renamed.
    return (function_name not in self.converted_self().functions) and (
        function_name in self.converted_function_names)

  def converted_self(self):
    # [FUNCTION DOC]
    # Returns the converted copy of this graph wrapper, creating on first use a new wrapper over
    # a deep copy of the graph definition (which rebuilds nodes, functions and edges). Caches it.
    if self._converted_self is None:
      copied_graph = graph_pb2.GraphDef()
      copied_graph.CopyFrom(self._graph_def)
      self._converted_self = _GraphDef(copied_graph)
    return self._converted_self

  def create_edges(self):
    # [FUNCTION DOC]
    # Asks every main-graph node wrapper and then every function wrapper to create its edges.
    # Returns nothing; fills the outgoing edge lists of the convertibles.
    for n in self._nodes.values():
      n.create_edges()
    for f in self._functions.values():
      f.create_edges()


class _ConverterData(object):
  # [CLASS DOC]
  # HELPER: base container of the data needed to freeze a graph.
  # Holds the graph definition, a mapping from node name to node definition (main graph nodes
  # plus control flow nodes inside functions), a mapping from tensor name to its value record,
  # and optional allow and deny lists of variable names. Subclasses fill the tensor data from a
  # concrete function or from a session.
  """Container for constant conversion supporting data.

  The data includes the graph being converted, and the pre-converted
  tensors. This class will be specialized for ConcreteFunction and Session-based
  conversions, as the means to obtain that data is different for each case.
  """

  def __init__(self,
               graph_def,
               variable_names_allowlist=None,
               variable_names_denylist=None):
    # [FUNCTION DOC]
    # Inputs: graph_def (graph definition to convert), variable_names_allowlist (optional set of
    # names to convert; nothing means all) and variable_names_denylist (optional set of names to
    # skip). Stores them, starts with an empty tensor data mapping and builds the node
    # definition mapping.
    self._graph_def = graph_def
    self._tensor_data = {}
    self._build_node_defs_list()
    self._variable_names_allowlist = variable_names_allowlist
    self._variable_names_denylist = variable_names_denylist

  @property
  def graph_def(self):
    # [FUNCTION DOC]
    # HELPER: accessor. Property returning the graph definition being converted.
    """The graph to be converted."""
    return self._graph_def

  @property
  def node_defs(self):
    # [FUNCTION DOC]
    # HELPER: accessor. Property returning the mapping from node name to node definition for
    # main-graph nodes and for control flow nodes inside library functions.
    """All the node defs in the graph to be converted.

    Returns:
      A map from node name to the NodeDef for all NodeDefs in the graph, as well
      as all control flow NodeDefs in the functions.
    """
    return self._node_defs

  @property
  def tensor_data(self):
    # [FUNCTION DOC]
    # HELPER: accessor. Property returning the live mapping from tensor name to its value record.
    """A map from tensor name to its converted _TensorData."""
    return self._tensor_data

  def _should_convert(self, name):
    # [FUNCTION DOC]
    # HELPER: filter used by subclasses when collecting variable values.
    # Input: name of a variable. Returns true when the allow list is unset or contains the name,
    # and the deny list is unset or does not contain it. No side effects.
    """Checks whether to convert the given variable name to a constant."""
    return (self._variable_names_allowlist is None or
            name in self._variable_names_allowlist) and (
                self._variable_names_denylist is None or
                name not in self._variable_names_denylist)

  def _build_node_defs_list(self):
    # [FUNCTION DOC]
    # HELPER: initialisation step of the converter data.
    # KNOWN ISSUE: the docstring calls the result a list but a mapping from names to node
    # definitions is built. Also, control flow nodes inside different functions with equal names
    # overwrite each other in the mapping.
    # Builds the mapping of node name to node definition from the main graph nodes, then adds
    # control flow nodes found in library functions. Sets the internal node definition mapping.
    """Builds the list of NodeDefs in the GraphDef.

    This list consists of all NodeDefs in the main graph as well as all control
    flow NodeDefs in the functions.

    The remaining NodeDefs in the functions are not included because the op
    names
    are not unique and the variables are handled differently than the main
    graph.
    The control flow ops need to be extracted because they are need their
    attributes to be updated similar to the control flow ops in the main graph.
    """
    self._node_defs = {node.name: node for node in self._graph_def.node}

    if self._graph_def.library:
      for func in self._graph_def.library.function:
        self._node_defs.update({
            node.name: node
            for node in func.node_def
            if node.op in _CONTROL_FLOW_OPS
        })


class _FunctionConverterData(_ConverterData):
  # [CLASS DOC]
  # HELPER: abstract converter-data class for concrete functions.
  # Runs graph inlining on the function's graph and collects values of captured variables and
  # tensors. Subclasses supply how a tensor is evaluated (eagerly or through a session).
  """Container for ConcreteFunction-based conversion data."""

  def __init__(self,
               func,
               lower_control_flow,
               aggressive_inlining,
               variable_names_allowlist=None,
               variable_names_denylist=None):
    # [FUNCTION DOC]
    # Inputs: func (a concrete function), lower_control_flow (whether control flow is lowered),
    # aggressive_inlining (whether function inlining is aggressive), plus optional allow and deny
    # lists of variable names. Stores the function, runs the inlining optimisation to obtain the
    # graph definition, initialises the parent with it and then fills tensor data for captures
    # and reference variables. Side effects: runs Grappler, evaluates tensors.
    """Creates the conversion data for the given function.

    Args:
      func: ConcreteFunction.
      lower_control_flow: Boolean indicating whether or not to lower control
        flow ops such as If and While.
      aggressive_inlining: Boolean indicating whether or not to do aggressive
        function inlining (might be unsafe if function has stateful ops, not
        properly connected to control outputs).
      variable_names_allowlist: The set of variable names to convert (by
        default, all variables are converted).
      variable_names_denylist: The set of variable names to omit converting to
        constants.
    """

    self._func = func
    # Inline the graph in order to remove functions when possible.
    graph_def = _run_inline_graph_optimization(func, lower_control_flow,
                                               aggressive_inlining)
    super(_FunctionConverterData, self).__init__(
        graph_def,
        variable_names_allowlist=variable_names_allowlist,
        variable_names_denylist=variable_names_denylist)

    self._build_tensor_data()

  def _eval(self, tensor):
    # [FUNCTION DOC]
    # HELPER: abstract hook for reading values during tensor data collection.
    # Abstract: always raises an unimplemented error; subclasses return the value of a given
    # tensor or variable.
    """Returns the value in the tensor. Must be implemented in sub-classes."""
    raise errors.UnimplementedError(
        "The evaluation method should be implemented in sub-classes.")

  def _build_tensor_data(self):
    # [FUNCTION DOC]
    # HELPER: initialisation step of the function converter data.
    # Maps captured inputs to the function's variables, then for each captured placeholder that
    # passes the allow/deny filter evaluates the variable or captured tensor (skipping resource
    # typed captures that are not variables, with a verbose log) and stores a value record keyed
    # by the placeholder name. Then for reference-variable nodes that were not captured, evaluates
    # them through a pruned identity in the function graph and stores the value with no index.
    # Side effects: evaluates tensors, may add an identity op to the function graph, logs.
    """Caches the tensor data for all Placeholders in the given function."""
    map_index_to_variable = {}
    for var in self._func.graph.variables:
      for idx, captured_input in enumerate(self._func.captured_inputs):
        if var.handle is captured_input:  # pylint: disable=protected-access
          map_index_to_variable[idx] = var
          break

    # Iterates through all captures which are represented as Placeholders.
    for idx, (val_tensor, name_tensor) in enumerate(self._func.graph.captures):
      tensor_name = name_tensor.name.split(":")[0]
      if not self._should_convert(tensor_name):
        continue
      if idx in map_index_to_variable:
        data = self._eval(map_index_to_variable[idx])
      else:
        if val_tensor.dtype == dtypes.resource:
          logging.vlog(1, "Skip converting resource tensor %s" % tensor_name)
          continue
        data = np.array(self._eval(val_tensor))

      self._tensor_data[tensor_name] = _TensorData(
          numpy=data,
          dtype=dtypes.as_dtype(data.dtype).as_datatype_enum,
          index=idx)

    # Get data for VariableV2 ops (reference variables) that cannot be lifted.
    for node in self.node_defs.values():
      if node.op == "VariableV2":
        if not self._should_convert(node.name):
          continue
        if node.name not in self.tensor_data:
          with self._func.graph.as_default():
            identity_node = array_ops.identity(
                self._func.graph.as_graph_element(node.name + ":0"))
          pruned_graph = self._func.prune([], [identity_node.name])()[0]
          self._tensor_data[node.name] = _TensorData(
              numpy=pruned_graph.numpy(),
              dtype=node.attr["dtype"].type,
              index=None)


class _FunctionConverterDataInEager(_FunctionConverterData):
  # [CLASS DOC]
  # HELPER: function converter data for eager execution; values are read directly from tensors.
  """Container for ConcreteFunction-based conversion data in Eager mode."""

  def _eval(self, tensor):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says this must be implemented in sub-classes, but this is the
    # concrete implementation.
    # Returns the numpy value of the given eager tensor or variable.
    """Returns the value in the tensor. Must be implemented in sub-classes."""
    return tensor.numpy()


class _FunctionConverterDataInGraph(_FunctionConverterData):
  # [CLASS DOC]
  # HELPER: function converter data for graph mode; values are read by running a session.
  """Container for ConcreteFunction-based conversion data in Graph mode."""

  def __init__(self,
               func,
               lower_control_flow,
               aggressive_inlining,
               variable_names_allowlist=None,
               variable_names_denylist=None,
               session=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: session is optional in the signature but is used unconditionally, so passing
    # nothing fails with an attribute error.
    # Inputs as in the parent class, plus session (the session used to evaluate values). Stores
    # the session, runs the global variable initialiser, runs each extra assignment op registered
    # in the default graph's extra-assign collection one by one, then calls the parent
    # constructor. Side effects: executes initialisation and assignment ops in the session.
    """Creates the conversion data for the given function.

    Args:
      func: ConcreteFunction.
      lower_control_flow: Boolean indicating whether or not to lower control
        flow ops such as If and While.
      aggressive_inlining: Boolean indicating whether or not to do aggressive
        function inlining (might be unsafe if function has stateful ops, not
        properly connected to control outputs).
      variable_names_allowlist: The set of variable names to convert (by
        default, all variables are converted).
      variable_names_denylist: The set of variable names to omit converting to
        constants.
      session: Session object.
    """
    self._session = session

    session.run(variables.global_variables_initializer())
    # Run extra assignment ops if needed.
    # These assignments are run sequentially to ensure order.
    for op in ops.get_default_graph().get_collection(VAR_ASSIGN_COLLECTION):
      session.run(op)

    super(_FunctionConverterDataInGraph, self).__init__(
        func,
        lower_control_flow,
        aggressive_inlining,
        variable_names_allowlist,
        variable_names_denylist)

  def _eval(self, tensor):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says this must be implemented in sub-classes, but this is the
    # concrete implementation.
    # Returns the value of the given tensor or variable by running it in the stored session.
    """Returns the value in the tensor. Must be implemented in sub-classes."""
    return self._session.run(tensor)


class _SessionConverterData(_ConverterData):
  # [CLASS DOC]
  # HELPER: converter data for session-based (version 1 style) graphs; variable values are read
  # by running the variables in a session.
  """Container for Session-based conversion data."""

  def __init__(self,
               session,
               graph_def,
               output_node_names,
               variable_names_allowlist=None,
               variable_names_denylist=None):
    # [FUNCTION DOC]
    # Inputs: session (active session holding variable values), graph_def (graph definition),
    # output_node_names (list of result node names used to prune the graph) and optional allow
    # and deny lists of names. Extracts the sub graph needed for the outputs, initialises the
    # parent, then for each variable, reference variable or variable handle node that passes the
    # filter (handles are read through their read op) runs all of them in one session call and
    # stores the resulting values with the node's dtype and no index. Side effect: runs the
    # session.
    graph_def = graph_util.extract_sub_graph(graph_def, output_node_names)
    super(_SessionConverterData, self).__init__(
        graph_def,
        variable_names_allowlist=variable_names_allowlist,
        variable_names_denylist=variable_names_denylist)

    nodes_to_convert = []
    tensor_names_to_convert = []
    for node in self.graph_def.node:
      if node.op in ["Variable", "VariableV2", "VarHandleOp"]:
        tensor_name = node.name
        if not self._should_convert(tensor_name):
          continue
        if node.op == "VarHandleOp":
          tensor_name = tensor_name + "/Read/ReadVariableOp"
        nodes_to_convert.append(node)
        tensor_names_to_convert.append(tensor_name + ":0")

    if tensor_names_to_convert:
      converted_tensors = session.run(tensor_names_to_convert)
      for node, tensor_value in zip(nodes_to_convert, converted_tensors):
        self._tensor_data[node.name] = _TensorData(
            numpy=tensor_value, dtype=node.attr["dtype"].type, index=None)


def disable_lower_using_switch_merge(graph_def):
  # [FUNCTION DOC]
  # Input: graph_def, a graph definition. Returns a copy in which every control flow node (if,
  # stateless if, while, stateless while) in the main graph and in every library function has
  # its attribute that allows lowering into switch/merge ops set to false. The input is not
  # modified. No exceptions are raised.
  """Set '_lower_using_switch_merge' attributes to False.

  Sets the attribute to False in the NodeDefs in the main graph and the NodeDefs
  in each function's graph.

  Args:
    graph_def: GraphDef proto.

  Returns:
    GraphDef
  """
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.CopyFrom(graph_def)

  def disable_control_flow_lowering(node):
    # [FUNCTION DOC]
    # HELPER: nested helper of disabling lowering. Input: a node definition. If it is a control
    # flow node, sets its lowering attribute to false in place; otherwise does nothing.
    if node.op in _CONTROL_FLOW_OPS:
      node.attr["_lower_using_switch_merge"].b = False

  for node in output_graph_def.node:
    disable_control_flow_lowering(node)

  if output_graph_def.library:
    for func in output_graph_def.library.function:
      for node in func.node_def:
        disable_control_flow_lowering(node)
  return output_graph_def


def _run_inline_graph_optimization(func, lower_control_flow,
                                   aggressive_inlining):
  # [FUNCTION DOC]
  # HELPER: graph preparation for the version 2 freezing functions.
  # KNOWN ISSUE: the loop over the four collection names always reads the byte list of the
  # variables collection instead of the one for the current name, so the other three collections
  # are overwritten with the variables collection contents. The docstring also says
  # lower_control_flow defaults to true, but this function has no defaults.
  # Inputs: func (concrete function), lower_control_flow (when false, lowering of control flow
  # is disabled in the graph definition) and aggressive_inlining (use aggressive function
  # optimisation). Exports the function's graph as a meta graph after removing the
  # implementation-selection attribute from library functions, clears initializer names of
  # variables, adds a training-op collection listing function inputs and outputs so Grappler
  # knows the fetches, and runs only the function-inlining optimiser (small graphs are not
  # skipped). Returns the optimised graph definition. Side effects: runs Grappler.
  """Apply function inline optimization to the graph.

  Returns the GraphDef after Grappler's function inlining optimization is
  applied. This optimization does not work on models with control flow.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to do aggressive
      function inlining (might be unsafe if function has stateful ops not
      properly connected to control outputs).

  Returns:
    GraphDef
  """
  graph_def = func.graph.as_graph_def()
  if not lower_control_flow:
    graph_def = disable_lower_using_switch_merge(graph_def)

  # In some cases, a secondary implementation of the function (e.g. for GPU) is
  # written to the "api_implements" attribute. (e.g. `tf.keras.layers.LSTM` in
  # TF2 produces a CuDNN-based RNN for GPU).
  # This function suppose to inline all functions calls, but "api_implements"
  # prevents this from happening. Removing the attribute solves the problem.
  # To learn more about "api_implements", see:
  #   tensorflow/core/grappler/optimizers/implementation_selector.h
  for function in graph_def.library.function:
    if "api_implements" in function.attr:
      del function.attr["api_implements"]

  meta_graph = export_meta_graph(graph_def=graph_def, graph=func.graph)

  # Clear the initializer_name for the variables collections, since they are not
  # needed after saved to saved_model.
  for name in [
      "variables", "model_variables", "trainable_variables", "local_variables"
  ]:
    raw_list = []
    for raw in meta_graph.collection_def["variables"].bytes_list.value:
      variable = variable_pb2.VariableDef()
      variable.ParseFromString(raw)
      variable.ClearField("initializer_name")
      raw_list.append(variable.SerializeToString())
    meta_graph.collection_def[name].bytes_list.value[:] = raw_list

  # Add a collection 'train_op' so that Grappler knows the outputs.
  fetch_collection = meta_graph_pb2.CollectionDef()
  for array in func.inputs + func.outputs:
    fetch_collection.node_list.value.append(array.name)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

  # Initialize RewriterConfig with everything disabled except function inlining.
  config = config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  rewrite_options.min_graph_nodes = -1  # do not skip small graphs
  rewrite_options.optimizers.append("function")
  if aggressive_inlining:
    rewrite_options.function_optimization =\
      rewriter_config_pb2.RewriterConfig.AGGRESSIVE
  return tf_optimizer.OptimizeGraph(config, meta_graph)


def _construct_concrete_function(func, output_graph_def,
                                 converted_input_indices):
  # [FUNCTION DOC]
  # HELPER: final step of the version 2 freezing functions.
  # Inputs: func (original concrete function), output_graph_def (frozen graph definition) and
  # converted_input_indices (set of captured-input positions that became constants). Computes
  # the input and output tensor names that remain, removes from the eager context any function
  # already registered under the names in the new library (so the new versions are used), wraps
  # the graph definition as a concrete function and copies the original shapes onto its inputs
  # (scalar shapes would otherwise be lost). Returns the new concrete function. Side effect:
  # removes functions from the global eager context.
  """Constructs a concrete function from the `output_graph_def`.

  Args:
    func: ConcreteFunction
    output_graph_def: GraphDef proto.
    converted_input_indices: Set of integers of input indices that were
      converted to constants.

  Returns:
    ConcreteFunction.
  """
  # Create a ConcreteFunction from the new GraphDef.
  input_tensors = func.graph.internal_captures
  converted_inputs = object_identity.ObjectIdentitySet(
      [input_tensors[index] for index in converted_input_indices])
  not_converted_inputs = [
      tensor for tensor in func.inputs if tensor not in converted_inputs
  ]
  not_converted_inputs_map = {
      tensor.name: tensor for tensor in not_converted_inputs
  }

  new_input_names = [tensor.name for tensor in not_converted_inputs]
  new_output_names = [tensor.name for tensor in func.outputs]

  # Remove old functions to use updated functions from graph def.
  for f in output_graph_def.library.function:
    if context.context().has_function(f.signature.name):
      context.context().remove_function(f.signature.name)

  new_func = wrap_function.function_from_graph_def(output_graph_def,
                                                   new_input_names,
                                                   new_output_names)

  # Manually propagate shape for input tensors where the shape is not correctly
  # propagated. Scalars shapes are lost when wrapping the function.
  for input_tensor in new_func.inputs:
    input_tensor.set_shape(not_converted_inputs_map[input_tensor.name].shape)
  return new_func


def _replace_variables_by_constants(converter_data):
  # [FUNCTION DOC]
  # HELPER: core conversion step shared by all public freezing functions.
  # KNOWN ISSUE: the docstring says it returns the converted graph, but it returns a pair: the
  # converted graph definition and the set of captured-input indices that were converted.
  # Input: converter_data, a populated converter data object. Wraps its graph, converts the node
  # for each collected tensor name (raises a key error if a name has no node) and returns the
  # converted graph definition and the set of non-empty input indices of the collected tensors.
  """Replaces variables by constants on a given graph.

  Given a _ConverterData instance with converted variables in its tensor_data
  field, create a new graph where the respective variables are replaced with the
  converted constants.

  Args:
    converter_data: A pre-populated _ConverterData instance.

  Returns:
    The converted graph.
  """
  input_graph = _GraphDef(converter_data.graph_def)

  for tensor_name, tensor_data in converter_data.tensor_data.items():
    input_graph.nodes[tensor_name].convert_variable_to_constant(
        None, tensor_data)

  converted_graph = input_graph.converted_self().graph_def

  converted_input_indices = {
      t.index
      for t in converter_data.tensor_data.values()
      if t.index is not None
  }

  return converted_graph, converted_input_indices


def convert_variables_to_constants_v2(func,
                                      lower_control_flow=True,
                                      aggressive_inlining=False):
  # [FUNCTION DOC]
  # Public eager-mode freezing function. Inputs: func (concrete function), lower_control_flow
  # (default true) and aggressive_inlining (default false). Collects variable values eagerly,
  # converts variables to constants and returns a new concrete function with the constants
  # embedded and the converted captures removed from its inputs. Side effects: runs Grappler,
  # evaluates variables, removes superseded functions from the eager context.
  """Replaces all the variables in a graph with constants of the same values.

  TensorFlow 2.0 function for converting all Variable ops into Const ops holding
  the same values. This makes it possible to describe the network fully with a
  single GraphDef file, and allows the removal of a lot of ops related to
  loading and saving the variables. This function runs Grappler's function
  inlining optimization in order to return a single subgraph.

  The current implementation only works for graphs that do not contain any
  control flow or embedding related ops.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to do aggressive
      function inlining (might be unsafe if function has stateful ops, not
      properly connected to control outputs). (default False)

  Returns:
    ConcreteFunction containing a simplified version of the original.
  """

  converter_data = _FunctionConverterDataInEager(
      func=func,
      lower_control_flow=lower_control_flow,
      aggressive_inlining=aggressive_inlining)

  output_graph_def, converted_input_indices = _replace_variables_by_constants(
      converter_data=converter_data)

  return _construct_concrete_function(func, output_graph_def,
                                      converted_input_indices)


def convert_var_to_const_function_in_v1(func,
                                        lower_control_flow=True,
                                        aggressive_inlining=False):
  # [FUNCTION DOC]
  # Graph-mode variant of the version 2 freezing function. Inputs as in that function. Requires
  # a default session; raises a runtime error if none is active. Runs initialisers and extra
  # assignment ops in that session, reads variable values through it, converts and returns a
  # new concrete function. Side effects: executes ops in the session, runs Grappler.
  """Replaces all the variables in a graph with constants of the same values.

  This function works as same as convert_variables_to_constants_v2, but it
  should be used in Graph mode. It is a temporary solution when users want to
  integrate their models written in TF2 with infra that requires TF1 mode.

  The current implementation only works for graphs that do not contain any
  control flow or embedding related ops.

  The function must be called in a Session context.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to do aggressive
      function inlining (might be unsafe if function has stateful ops, not
      properly connected to control outputs). (default False)

  Raises:
      RuntimeError: If no Session context is present.

  Returns:
    ConcreteFunction containing a simplified version of the original.
  """

  session = ops.get_default_session()
  if session is None:
    raise RuntimeError(
        "The conversion must be carried out in a Session context.")

  converter_data = _FunctionConverterDataInGraph(
      func=func,
      lower_control_flow=lower_control_flow,
      aggressive_inlining=aggressive_inlining,
      session=session)

  output_graph_def, converted_input_indices = _replace_variables_by_constants(
      converter_data=converter_data)

  return _construct_concrete_function(func, output_graph_def,
                                      converted_input_indices)


def convert_variables_to_constants_v2_as_graph(func,
                                               lower_control_flow=True,
                                               aggressive_inlining=False):
  # [FUNCTION DOC]
  # Same as the eager version 2 freezing function but also returns the frozen graph definition.
  # Inputs: func (concrete function), lower_control_flow (default true), aggressive_inlining
  # (default false). Returns a pair: the frozen concrete function and the converted graph
  # definition. Same side effects as the eager function.
  """Replaces all the variables in a graph with constants of the same values.

  This function works as same as convert_variables_to_constants_v2, but it
  returns the intermediate `GraphDef` as well. This `GraphDef` contains all the
  debug information after all the transformations in the frozen phase.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to do aggressive
      function inlining (might be unsafe if function has stateful ops, not
      properly connected to control outputs).

  Returns:
    ConcreteFunction containing a simplified version of the original, and also
    the intermediate GraphDef containing the node debug information for the
    transformations in the frozen phase.
  """
  converter_data = _FunctionConverterDataInEager(
      func=func,
      lower_control_flow=lower_control_flow,
      aggressive_inlining=aggressive_inlining)

  output_graph_def, converted_input_indices = _replace_variables_by_constants(
      converter_data=converter_data)

  frozen_func = _construct_concrete_function(func, output_graph_def,
                                             converted_input_indices)
  return frozen_func, output_graph_def


def convert_variables_to_constants_from_session_graph(
    session,
    graph_def,
    output_node_names,
    variable_names_allowlist=None,
    variable_names_denylist=None):
  # [FUNCTION DOC]
  # Session-based freezing for version 1 graphs. Inputs: session (active session with variable
  # values), graph_def (graph definition), output_node_names (list of result node names, used to
  # prune the graph), optional allow and deny lists of variable names. Reads variable values by
  # running them in the session, converts them and returns the converted graph definition.
  # Side effect: runs the session.
  """Replaces all the variables in a graph with constants of the same values.

  This function works similarly to convert_variables_to_constants_v2, but it
  retrieves the constant values from a Session instead of from a
  ConcreteFunction. This is useful when converting graphs generated from
  TensorFlow V1, where ConcreteFunctions are not available. This also differs
  from graph_util.convert_variables_to_constants in that it supports resource
  variables when V2 control flow constructions are present.

  Args:
    session: Active TensorFlow session containing the variables.
    graph_def: A GraphDef to convert.
    output_node_names: List of name strings for the result nodes of the graph.
    variable_names_allowlist: The set of variable names to convert (by default,
      all variables are converted).
    variable_names_denylist: The set of variable names to omit converting to
      constants.

  Returns:
    An optimized GraphDef.
  """
  graph_def, _ = _replace_variables_by_constants(
      converter_data=_SessionConverterData(
          session=session,
          graph_def=graph_def,
          output_node_names=output_node_names,
          variable_names_allowlist=variable_names_allowlist,
          variable_names_denylist=variable_names_denylist))
  return graph_def


@deprecation.deprecated(
    date=None,
    instructions="This API was designed for TensorFlow v1. See "
    "https://www.tensorflow.org/guide/migrate for instructions on how to "
    "migrate your code to TensorFlow v2."
)
@tf_export(v1=["graph_util.convert_variables_to_constants"])
def convert_variables_to_constants(sess,
                                   input_graph_def,
                                   output_node_names,
                                   variable_names_whitelist=None,
                                   variable_names_blacklist=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring's Raises section says a runtime error occurs for conflicting
  # allow and deny listing, but this wrapper raises nothing itself and no such check exists in the
  # session conversion path. Deprecated and exported only in the version 1 API.
  # Deprecated wrapper over the session-based conversion. Inputs: sess (active session),
  # input_graph_def (graph definition), output_node_names (list of result names), and optional
  # variable_names_whitelist and variable_names_blacklist (renamed allow and deny lists).
  # Returns the converted graph definition.
  """Replaces all the variables in a graph with constants of the same values.

  If you have a trained graph containing Variable ops, it can be convenient to
  convert them all to Const ops holding the same values. This makes it possible
  to describe the network fully with a single GraphDef file, and allows the
  removal of a lot of ops related to loading and saving the variables.

  Args:
    sess: Active TensorFlow session containing the variables.
    input_graph_def: GraphDef object holding the network.
    output_node_names: List of name strings for the result nodes of the graph.
    variable_names_whitelist: The set of variable names to convert (by default,
      all variables are converted).
    variable_names_blacklist: The set of variable names to omit converting to
      constants.

  Returns:
    GraphDef containing a simplified version of the original.

  Raises:
    RuntimeError: if a DT_RESOURCE op is found whose ancestor Variables are both
      denylisted AND whitelisted for freezing.
  """
  ret = convert_variables_to_constants_from_session_graph(
      session=sess,
      graph_def=input_graph_def,
      output_node_names=output_node_names,
      variable_names_allowlist=variable_names_whitelist,
      variable_names_denylist=variable_names_blacklist)
  return ret
