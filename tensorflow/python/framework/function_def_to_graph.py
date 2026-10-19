# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
# [MODULE DOC]
# Utilities that convert a FunctionDef protocol buffer into a GraphDef or a FuncGraph. It exposes
# function_def_to_graph (full conversion), function_def_to_graph_def (proto-level conversion with
# nested-to-flat tensor name mapping), plus lookup helpers for function definitions in the eager
# context or in graph/outer graphs and a recursive library copier. Used by function loading,
# saved model import and function inspection code. No import-time side effects beyond imports.
"""Utility to convert FunctionDef to GraphDef and Graph."""

import itertools


from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import versions
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import resource_variable_ops


def function_def_to_graph(
    fdef,
    structured_input_signature=None,
    structured_outputs=None,
    input_shapes=None,
    propagate_device_spec=False,
    include_library_functions=False,
):
  # [FUNCTION DOC]
  # Converts a function definition into a function graph (a graph subclass) with inputs, outputs,
  # control outputs and output names populated.
  # Inputs: fdef is the function definition; structured_input_signature and structured_outputs are
  # optional structures forwarded to the function graph constructor; input_shapes is an optional
  # list of shapes (default: taken from the definition's input-shapes attribute, where resource
  # handle inputs with handle data get unknown shape to avoid shape inference problems);
  # propagate_device_spec (default false) keeps assigned devices when importing nodes;
  # include_library_functions (default false) copies referenced functions into the library.
  # Behavior: converts to a graph definition, imports it into the new graph, sets inputs, outputs
  # and control outputs by tensor/operation name, applies handle data for resources, applies the
  # recorded output-shapes attribute to op outputs (extra entries are tolerated), and records a
  # mapping from output tensor identity to the output argument names.
  # Returns the function graph. Raises a value error on invalid shapes or missing functions.
  # Side effects: temporarily enters the new graph as default; touches a private attribute.
  """Converts a FunctionDef to a FuncGraph (sub-class Graph).

  The returned FuncGraph's `name`, `inputs` and `outputs` fields will be set.
  The input tensors are represented as placeholders.

  Note: `FuncGraph.inputs` and `FuncGraph.captures` are not set and may be set
  by the caller.

  Args:
    fdef: FunctionDef.
    structured_input_signature: Optional. The structured input signature to use
      for initializing the FuncGraph. See the docstring for FuncGraph for more
      information.
    structured_outputs: Optional. The structured outputs to use for initializing
      the FuncGraph. See the docstring for FuncGraph for more information.
    input_shapes: Optional. A list of TensorShape objects of the shapes of
      function inputs. Defaults to the function's "_input_shapes" attribute. If
      specified, its length must match length of `fdef.signature.input_arg`. If
      a shape is None, the corresponding input placeholder will have unknown
      shape.
    propagate_device_spec: Optional. Whether to propagate assigned device
      information when constructing a new Graph from a FunctionDef.
    include_library_functions: Optional. Whether to include library functions in
      the output FuncGraph. In graph mode, the library functions will be found
      from outer graph. In eager mode, the library functions will be found from
      eager context.

  Returns:
    A FuncGraph.
  """
  func_graph = FuncGraph(fdef.signature.name,
                         structured_input_signature=structured_input_signature,
                         structured_outputs=structured_outputs)
  if input_shapes is None:
    input_shapes_attr = fdef.attr.get("_input_shapes", None)
    if input_shapes_attr is not None:
      raw_input_shapes = input_shapes_attr.list.shape

      # Replace resource handle shapes in the inputs to disable shape inference.
      # Setting the shape to either the variable handle shape (which is always
      # `[]`) or the variable shape can cause shape inference issues.
      input_shapes = []
      for input_shape, arg_def in zip(raw_input_shapes,
                                      fdef.signature.input_arg):
        if arg_def.type == types_pb2.DT_RESOURCE and arg_def.handle_data:
          input_shapes.append(None)
        else:
          input_shapes.append(input_shape)

  graph_def, nested_to_flat_tensor_name = function_def_to_graph_def(
      fdef, input_shapes, include_library_functions=include_library_functions
  )

  with func_graph.as_default():
    # Add all function nodes to the graph.
    importer.import_graph_def_for_function(
        graph_def, name="", propagate_device_spec=propagate_device_spec)

    # Initialize fields specific to FuncGraph.

    # inputs
    input_tensor_names = [
        nested_to_flat_tensor_name[arg.name] for arg in fdef.signature.input_arg
    ]
    func_graph.inputs = [
        func_graph.get_tensor_by_name(name) for name in input_tensor_names
    ]

    # outputs
    output_tensor_names = [
        nested_to_flat_tensor_name[fdef.ret[arg.name]]
        for arg in fdef.signature.output_arg
    ]
    func_graph.outputs = [
        func_graph.get_tensor_by_name(name) for name in output_tensor_names
    ]
    func_graph.control_outputs = [
        func_graph.get_operation_by_name(fdef.control_ret[ret_name])
        for ret_name in fdef.signature.control_output
    ]

    _set_handle_data(func_graph, fdef)

    for node in graph_def.node:
      output_shapes = node.attr.get("_output_shapes", None)
      if output_shapes is not None:
        op = func_graph.get_operation_by_name(node.name)
        # _output_shapes for functions can sometimes be too long because the
        # output-intermediates-for-gradients version of the function was
        # substituted before saving. We'll accept that here. (See b/133666530).
        for output_index, shape in enumerate(
            output_shapes.list.shape[:len(op.outputs)]):
          op.outputs[output_index].set_shape(shape)
    output_names = {}
    for ret_arg_def, tensor_name in zip(
        fdef.signature.output_arg, output_tensor_names):
      output_names[ops.tensor_id(
          func_graph.get_tensor_by_name(tensor_name))] = (
              ret_arg_def.name)
    func_graph._output_names = output_names  # pylint: disable=protected-access
  return func_graph


def is_function(fname, graph):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the return value is not always a boolean; in graph mode it returns true, false
  # (when a graph lacks an outer graph) or an implicit null value when the walk runs off the end.
  # Reports whether a function with the given name is defined. In eager mode the eager context is
  # the only source of truth and the graph argument is ignored. In graph mode, the given graph and
  # then its chain of outer graphs are searched. No side effects.
  """Checks for a function definition with `fname` in the current context."""
  if context.executing_eagerly():
    # Eager mode: use eager context as the single source of truth.
    return context.context().has_function(fname)
  else:
    # Graph mode: use outer graphs as the single source of truth.
    while graph is not None:
      if graph._is_function(fname):  # pylint: disable=protected-access
        return True
      if hasattr(graph, "outer_graph"):
        graph = graph.outer_graph
      else:
        return False


def get_function_def(fname, graph):
  # [FUNCTION DOC]
  # KNOWN ISSUE: returns nothing (null value) implicitly when the function is not found, instead
  # of raising; callers must handle the null result.
  # Looks up a function definition by name. In eager mode it asks the eager context (the graph
  # argument is ignored); in graph mode it walks from the given graph through its outer graphs
  # until one defines the function. Returns the function definition proto, or nothing if absent.
  # No side effects.
  """Gets a function definition with `fname` in the current context."""
  if context.executing_eagerly():
    # Eager mode: use eager context as the single source of truth.
    if context.context().has_function(fname):
      return context.context().get_function_def(fname)
  else:
    # Graph mode: use outer graphs as the single source of truth.
    while graph is not None:
      if graph._is_function(fname):  # pylint: disable=protected-access
        return graph._get_function(fname).cached_definition  # pylint: disable=protected-access
      graph = getattr(graph, "outer_graph", None)


def copy_function_def_to_graph_def_recursively(
    func_name, graph_def, copied_functions, default_graph=None):
  # [FUNCTION DOC]
  # HELPER: internal step of function-to-graph conversion that fills the graph definition library.
  # KNOWN ISSUE: the docstring says default_graph is not used in eager mode, but it is used
  # unconditionally to look up op definitions for each node; passing nothing (the default) would
  # fail with an attribute error whenever the function has body nodes.
  # Copies the named function definition and, recursively, every function referenced through
  # function-valued or function-list attributes of its nodes into the graph definition library.
  # Inputs: func_name is the function signature name; graph_def is the graph definition that is
  # mutated; copied_functions is a set of already-copied names that is updated so each function is
  # copied at most once; default_graph is the graph used for lookups in graph mode.
  # Raises a value error if a non-empty function name cannot be found. Empty names (custom ops)
  # are tolerated. Returns nothing.
  """Recursively copies `FunctionDef`s to `GraphDef`.

  It copies the outermost `FunctionDef` and all nested `FunctionDef`s to
  `graph_def`. The `copied_function` enforces that every `FunctionDef` will be
  copied at most once. The `FunctionDef`s will be found from `default_graph` if
  this function was called in graph mode or from eager context if this function
  was called in eager mode.

  Args:
    func_name: The signature name of FunctionDef to be copied to `graph_def`.
    graph_def: The GraphDef that will contain all `FunctionDef`s in its library.
    copied_functions: A set contains all copied function names.
    default_graph: The `tf.Graph` where all `FunctionDef`s will be found
      in graph mode. Not used in eager mode.
  """
  # Custom ops may contain a func attr with an empty fname.
  if func_name and not is_function(func_name, default_graph):
    raise ValueError(f"Function {func_name} was not found. Please make "
                     "sure the FunctionDef `fdef` is correct.")

  # If `copied_functions` contains `func_name`, the FunctionDef has already
  # been added to GraphDef so we simply return here.
  if func_name in copied_functions:
    return

  copied_functions.add(func_name)
  func_def = get_function_def(func_name, default_graph)
  graph_def.library.function.add().CopyFrom(func_def)

  for node_def in func_def.node_def:
    op_def = default_graph.op_def_for_type(node_def.op)
    for attr in op_def.attr:
      if attr.type == "func":
        func_name = node_def.attr[attr.name].func.name
        copy_function_def_to_graph_def_recursively(
            func_name, graph_def, copied_functions, default_graph)

      elif attr.type == "list(func)":
        for fn in node_def.attr[attr.name].list.func:
          func_name = fn.name
          copy_function_def_to_graph_def_recursively(
              func_name, graph_def, copied_functions, default_graph)


def function_def_to_graph_def(
    fdef, input_shapes=None, include_library_functions=False
):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the parameter fdef is rebound inside the node loop to the cached definition of a
  # referenced function; this shadows the argument and is misleading (it is not used afterwards,
  # and the loop iterates over the original node list, so results are unaffected).
  # KNOWN ISSUE: the docstring says a ValueError is raised for an invalid FunctionDef, but
  # missing tensor names in the mapping would surface as a lookup (key) error instead.
  # Converts a function definition into a graph definition plus a tensor name mapping.
  # Inputs: fdef is the function definition; input_shapes is an optional list of shapes (shape
  # objects or protos, entries may be nothing meaning unknown shape) whose length must equal the
  # number of input arguments; include_library_functions (default false) controls whether the
  # function and all functions it references are copied into the graph definition library.
  # Behavior: sets the graph version fields; creates one placeholder node per input argument
  # (copying dtype, the given shape, and internal underscore-prefixed argument attributes, with
  # the output-shapes attribute mapped to the placeholder shape); appends the body nodes; builds
  # a mapping from nested names (node:output_arg:index, plus control names) to flat names
  # (node:index); for nodes whose op is itself a function found in this or an outer graph, copies
  # that function (and its gradient definition if any) into the library; verifies referenced
  # function attributes exist, and finally rewrites every node input using the mapping.
  # Returns a pair: the graph definition and the nested-to-flat name mapping.
  # Raises a value error if the input shapes length mismatches, a referenced function is not
  # found, or an argument definition is invalid. Uses the current default graph and, when
  # copying libraries, the eager context; the function definitions are copied, not modified.
  """Convert a FunctionDef to a GraphDef.

  Steps:
  1. Creates placeholder nodes corresponding to inputs in
     `FunctionDef.signature.input_arg`.
  2. Adds NodeDefs in `FunctionDef.node_def` to `GraphDef.node`.
  3. Renames inputs of all nodes to use the convention of GraphDef instead of
     FunctionDef. See comment on `FunctionDef.node_def` on how the tensor naming
     in FunctionDefs is different from GraphDefs.

  Args:
    fdef: FunctionDef.
    input_shapes: Optional. A list of TensorShape objects of the shapes of
      function inputs. If specified, its length must match length of
      `fdef.signature.input_arg`. If a shape is None, the corresponding input
      placeholder will have unknown shape.
    include_library_functions: Optional. If enabled, copy `fdef` and its
      nested `FunctionDef`s to the library functions of the returned `GraphDef`.
      In graph mode, the functions will be found from outer graph. In eager
      mode, the functions will be found from eager context.

  Returns:
    A tuple of (GraphDef, dict<string, string>). The dict contains a mapping
    from nested tensor names (in FunctionDef) to flattened names (in GraphDef).

  Raises:
    ValueError: If the length of input_shapes does not match the number of
      input_args or if the FunctionDef is invalid.
  """
  graph_def = graph_pb2.GraphDef()
  graph_def.versions.CopyFrom(
      versions_pb2.VersionDef(
          producer=versions.GRAPH_DEF_VERSION,
          min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER))

  default_graph = ops.get_default_graph()

  copied_functions = set()

  if input_shapes and len(input_shapes) != len(fdef.signature.input_arg):
    raise ValueError("Length of `input_shapes` must match the number "
                     f"of `input_arg`s in `fdef`. Got "
                     f"{len(input_shapes)} `input_shapes` and "
                     f"{len(fdef.signature.input_arg)} `input_arg`s.")

  # 1. Create placeholders for input nodes.
  for i, arg_def in enumerate(fdef.signature.input_arg):
    node_def = graph_def.node.add()
    node_def.name = arg_def.name
    node_def.op = "Placeholder"
    node_def.attr["dtype"].type = arg_def.type
    if input_shapes and input_shapes[i] is not None:
      input_shape = input_shapes[i]
      if not isinstance(input_shape, tensor_shape_pb2.TensorShapeProto):
        input_shape = input_shape.as_proto()
      node_def.attr["shape"].shape.CopyFrom(input_shape)
    arg_attrs = fdef.arg_attr[i].attr
    for k in arg_attrs:
      # Only copy internal attributes. Normal attributes for nodes cannot be
      # applied to these Placeholder nodes.
      if k == "_output_shapes":
        if arg_attrs[k].WhichOneof("value") == "list":
          node_def.attr["shape"].shape.CopyFrom(arg_attrs[k].list.shape[0])
        elif arg_attrs[k].WhichOneof("value") == "shape":
          node_def.attr["shape"].shape.CopyFrom(arg_attrs[k].shape)
      elif k.startswith("_"):
        node_def.attr[k].CopyFrom(arg_attrs[k])

  # 2. Copy all body NodeDefs to the GraphDef.
  graph_def.node.extend(fdef.node_def)

  # 3. Perform the renaming.

  # Build the tensor name mapping then flatten the tensor names.
  # See comment on `FunctionDef.node_def` on how the tensor naming in
  # FunctionDefs is different from GraphDefs.
  nested_to_flat_tensor_name = {}

  for arg_def in fdef.signature.input_arg:
    nested_to_flat_tensor_name[arg_def.name] = "{}:0".format(arg_def.name)
    control_name = "^" + arg_def.name
    nested_to_flat_tensor_name[control_name] = control_name

  for node_def in fdef.node_def:
    graph = default_graph
    while True:
      f = graph._functions.get(node_def.op, None)  # pylint: disable=protected-access
      if f is not None or not hasattr(graph, "outer_graph"):
        break
      graph = graph.outer_graph

    if f is not None:
      fdef = f.cached_definition
      op_def = fdef.signature
      if node_def.op not in copied_functions:
        # Since this function is referenced as an op type, we have no choice but
        # to copy it into the GraphDef if we want downstream tools to process
        # it.
        graph_def.library.function.add().CopyFrom(fdef)
        copied_functions.add(node_def.op)
        if getattr(f, "grad_func_name", None):
          grad_def = function_pb2.GradientDef()
          grad_def.function_name = f.name
          grad_def.gradient_func = f.grad_func_name
          graph_def.library.gradient.extend([grad_def])
    else:
      op_def = default_graph.op_def_for_type(node_def.op)  # pylint: disable=protected-access

    for attr in op_def.attr:
      if attr.type == "func":
        fname = node_def.attr[attr.name].func.name
        # Custom ops may contain a func attr with an empty fname.
        if fname and not is_function(fname, default_graph):
          raise ValueError(f"Function {fname} was not found. Please make sure "
                           "the FunctionDef `fdef` is correct.")
        if include_library_functions:
          copy_function_def_to_graph_def_recursively(
              fname, graph_def, copied_functions, default_graph)

      elif attr.type == "list(func)":
        for fn in node_def.attr[attr.name].list.func:
          fname = fn.name
          # Custom ops may contain a func attr with an empty fname.
          if fname and not is_function(fname, default_graph):
            raise ValueError(f"Function {fname} was not found. Please make "
                             "sure the FunctionDef `fdef` is correct.")
          if include_library_functions:
            copy_function_def_to_graph_def_recursively(
                fname, graph_def, copied_functions, default_graph)

    # Iterate over output_args in op_def to build the map.
    # Index of the output tensor in the flattened list of *all* output
    # tensors of the op.
    flattened_index = 0
    for arg_def in op_def.output_arg:
      num_args = _get_num_args(arg_def, node_def)
      for i in range(num_args):
        # Map tensor names from "node_name:output_arg_name:index" to
        # "node_name:flattened_index".
        nested_name = "{}:{}:{}".format(node_def.name, arg_def.name, i)
        flat_name = "{}:{}".format(node_def.name, flattened_index)
        nested_to_flat_tensor_name[nested_name] = flat_name
        flattened_index += 1
    control_name = "^" + node_def.name
    nested_to_flat_tensor_name[control_name] = control_name

  # Update inputs of all nodes in graph.
  for node_def in graph_def.node:
    for i in range(len(node_def.input)):
      node_def.input[i] = nested_to_flat_tensor_name[node_def.input[i]]

  return graph_def, nested_to_flat_tensor_name


# Based on implementation in core/framework/node_def_util.cc::ComputeArgRange.
def _get_num_args(arg_def, node_def):
  # [FUNCTION DOC]
  # HELPER: internal utility for building the nested-to-flat tensor name mapping.
  # Computes how many tensors an operation argument definition expands to for a given node.
  # Inputs: arg_def is an output argument definition from an op definition; node_def is the node
  # whose attributes supply the counts.
  # Returns the integer attribute value when the argument uses a number attribute, the length of
  # the type list when it uses a type-list attribute, and one when it has a single type attribute
  # or a fixed valid type.
  # Raises a value error if the argument definition specifies none of these (invalid definition).
  # Mirrors the argument range computation in the C++ node definition utilities.
  if arg_def.number_attr:
    return node_def.attr[arg_def.number_attr].i
  elif arg_def.type_list_attr:
    return len(node_def.attr[arg_def.type_list_attr].list.type)
  elif arg_def.type_attr or arg_def.type != types_pb2.DT_INVALID:
    return 1
  else:
    raise ValueError(f"Invalid arg_def:\n\n{arg_def}. Please make sure the "
                     "FunctionDef `fdef` is correct.")


def _set_handle_data(func_graph, fdef):
  # [FUNCTION DOC]
  # HELPER: internal step of converting a function definition into a function graph.
  # Restores shape and dtype information for resource-typed (variable handle) inputs and outputs.
  # Inputs: func_graph is the function graph whose inputs and outputs lists are already populated;
  # fdef is the function definition whose signature input and output argument definitions may carry
  # handle data. Pairs each graph input/output tensor with its argument definition, in order.
  # For each argument that has handle data, the tensor's own static shape is set to scalar and the
  # first handle shape-and-type entry is stored as the tensor's resource handle data (shape and
  # dtype of the pointed-to variable) via the resource variable helper.
  # Returns nothing; mutates the tensors in place. Arguments without handle data are untouched.
  # Extra tensors beyond the shorter of the two paired sequences are ignored.
  """Adds handle data for resource type inputs and outputs."""
  # The shape of the handle itself is [], while the variable shape is
  # saved in `handle_data`. Previously, the shape of the resource handle
  # was set to `None`. Correct both shapes here.
  for tensor, arg_def in itertools.chain(
      zip(func_graph.inputs, fdef.signature.input_arg),
      zip(func_graph.outputs, fdef.signature.output_arg)):
    if arg_def.handle_data:
      tensor.set_shape([])

      shape_and_dtype = arg_def.handle_data[0]
      handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
      handle_data.is_set = True
      handle_data.shape_and_type.append(
          cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
              shape=shape_and_dtype.shape, dtype=shape_and_dtype.dtype))
      resource_variable_ops._set_handle_shapes_and_types(  # pylint: disable=protected-access
          tensor, handle_data, True)
