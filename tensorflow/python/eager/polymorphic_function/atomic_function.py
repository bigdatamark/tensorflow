# [MODULE DOC]
# Implements AtomicFunction, the Python callable around a function registered in the TensorFlow
# runtime, with its call options, factory functions (from function definitions or function graphs),
# graph-call helpers, conversion back to function graphs, and a context manager that rewrites
# runtime errors with Python stack traces. Import-time side effect: defines the module-level
# reference-count mapping for runtime functions. Used by the polymorphic function machinery.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation for AtomicFunction."""

import dataclasses
import traceback
import typing
from typing import Any, Dict, List, Optional, Sequence, Union

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_stack


# TODO(fmuham): Should be lowered to FunctionDef/FunctionRecord.
@dataclasses.dataclass(frozen=True)
class CallOptions:
  # [CLASS DOC]
  # Immutable record of extra configuration for calling an AtomicFunction: ids of collective
  # managers used (list of integers), control captures (list of ops/tensors/callables that must run
  # first) and a stateful flag selecting the kind of partitioned call. All fields have defaults.
  """Specifies additional configuration for an AtomicFunction call."""

  # Used by ACD to identify the CollectiveManager this function is scoped in.
  collective_manager_ids_used: List[int] = dataclasses.field(
      default_factory=list
  )

  # Used by ACD to list Ops/Tensors/Callables that must be called in advance.
  control_captures: List[Any] = dataclasses.field(default_factory=list)

  # Determines what kind of partitioned call is used for this function.
  is_stateful: bool = False


# Maps the (scope_id, name) in runtime to associated AtomicFunctions.
RUNTIME_FUNCTION_REFS = {}


class AtomicFunction(core.AtomicFunction):
  # [CLASS DOC]
  # Python callable wrapping a function registered in the TensorFlow runtime context. Holds name,
  # bound context, function type, child dependencies, call options and lazily cached definition and
  # graph (restricted by slots). Created directly, or via the module's factory functions from a
  # function definition or function graph. Calls work in eager and graph mode, with error
  # interpolation. Reference counts in a module-level mapping ensure the runtime function is
  # removed only after the last wrapper is destroyed.
  """A Python callable for functions in the TF Runtime.

  Provides core functionality for tf.function including:
    - automatic lifecycle management of runtime functions
    - structured inputs (including captures) and structured outputs
    - calls from both eager and graph mode
    - dependency tracking of children functions
    - runtime error interpolation to identify user code stack traces
    - control dependencies (including automatic)
  """

  __slots__ = [
      "_name",
      "_bound_context",
      "_function_type",
      "_children",
      "_call_options",
      "_cached_definition",
      "_cached_graph",
      "_generated_graph",
  ]

  def __init__(
      self,
      name: Union[str, bytes],
      bound_context: context.Context,
      function_type: function_type_lib.FunctionType,
      children: Optional[List["AtomicFunction"]] = None,
      call_options: CallOptions = CallOptions(),
      cached_graph: Optional[func_graph_module.FuncGraph] = None,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the default call options value is a single shared instance created at definition
    # time, reused by every function constructed without explicit options.
    # Creates the wrapper. Inputs: name (text or bytes, stored as bytes), bound context, function
    # type, optional children list (empty if none), optional call options, optional cached graph.
    # Side effect: increments the module-level reference count for (context scope id, name) so the
    # runtime function is removed only when the last wrapper is destroyed. Returns nothing.
    """Construct a new AtomicFunction.

    Args:
      name: str/bytes name of the runtime function in the bound context.
      bound_context: interface to the runtime for the AtomicFunction.
      function_type: input/output contract for the AtomicFunction
      children: list of AtomicFunctions that are needed to call this one.
      call_options: extra configuration options for the call.
      cached_graph: FuncGraph that this AtomicFunction was generated from (if
        known). Otherwise it will lazily construct a new corresponding FuncGraph
        if ever needed.
    """
    self._name = compat.as_bytes(name)
    self._bound_context = bound_context
    self._function_type = function_type
    self._children = children if children else []
    self._call_options = call_options
    self._cached_definition = None

    self._cached_graph = cached_graph
    self._generated_graph = None

    ref_key = (self._bound_context.function_scope_id, self.name)
    if ref_key not in RUNTIME_FUNCTION_REFS:
      RUNTIME_FUNCTION_REFS[ref_key] = 1
    else:
      RUNTIME_FUNCTION_REFS[ref_key] += 1

  @property
  def name(self) -> bytes:
    # [FUNCTION DOC]
    # Property returning the runtime function name as bytes.
    """Name represented in UTF-8 encoded bytes."""
    return self._name

  @property
  def function_type(self) -> function_type_lib.FunctionType:
    # [FUNCTION DOC]
    # Property returning the input/output contract given at construction.
    """Represents the input/output contract of this function."""
    return self._function_type

  @property
  def children(self) -> List["AtomicFunction"]:
    # [FUNCTION DOC]
    # Property returning the list of dependency functions (the stored list, not a copy).
    """AtomicFunctions needed as dependencies for this one."""
    return self._children

  @property
  def definition(self) -> function_pb2.FunctionDef:
    # [FUNCTION DOC]
    # Property fetching the current function definition from the runtime by name (not cached).
    """Current FunctionDef in the Runtime."""
    return self._bound_context.get_function_def(self.name)

  @property
  def attributes(self) -> Any:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the runtime-construction-context entry is popped from the attribute mapping of
    # the definition returned by the context, which may mutate that definition's own attributes
    # rather than a copy.
    # Property returning the attribute mapping of the current runtime definition without the eager
    # runtime construction context entry.
    """Returns FunctionDef attributes in the Runtime."""
    attrs = self.definition.attr
    # Remove construction context since it is specific to runtime and this fn.
    attrs.pop(attributes_lib.EAGER_RUNTIME_CONSTRUCTION_CONTEXT, None)
    return attrs

  @property
  def graph_debug_info(self) -> graph_debug_info_pb2.GraphDebugInfo:
    # [FUNCTION DOC]
    # Property fetching, from the bound context, the debug info (node to stack trace mapping) of
    # this function.
    """A GraphDebugInfo proto mapping nodes to corresponding stack traces."""
    return self._bound_context.get_graph_debug_info(self.name)

  @property
  def call_options(self) -> CallOptions:
    # [FUNCTION DOC]
    # Property returning the call options given at construction.
    """Call options declared for this AtomicFunction."""
    return self._call_options

  @property
  def graph_call_attrs(self) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Property returning a new mapping of attributes needed to add a call op in a graph: the
    # stateful flag, the list of output type enums, the XLA compile attribute (null if absent),
    # plus the context's function call options (config and executor type).
    """Returns a dictionary of attributes needed to add a call in graph."""
    attrs = {
        "is_stateful": self.call_options.is_stateful,
        "tout": [
            o.dtype.as_datatype_enum for o in self.function_type.flat_outputs
        ],
        "xla_compile_attr": self.cached_definition.attr.get(
            attributes_lib.XLA_COMPILE, None
        ),
    }
    attrs.update(self._bound_context.function_call_options.as_attrs())
    return attrs

  @property
  def _c_func(self) -> Any:
    # [FUNCTION DOC]
    # HELPER: private accessor for runtime-level code.
    # Property returning the native function record for this function's name from the bound context.
    """Returns a scoped pybind object containing FunctionRecord in runtime."""
    return self._bound_context.get_c_function(self.name)

  # TODO(fmuham): Move caching to dependent code and remove method.
  @property
  def cached_definition(self) -> function_pb2.FunctionDef:
    # [FUNCTION DOC]
    # Property returning the function definition, fetched from the runtime once and then reused,
    # so it may be stale if the runtime copy changes.
    """Cached FunctionDef (not guaranteed to be fresh)."""
    if self._cached_definition is None:
      self._cached_definition = self.definition

    return self._cached_definition

  @property
  def graph(self) -> func_graph_module.FuncGraph:
    # [FUNCTION DOC]
    # Property returning the function graph: the cached one if provided at construction, otherwise
    # one generated lazily from the runtime definition on first access and remembered.
    """Returns a FuncGraph corresponding to the AtomicFunction."""
    if self._cached_graph:
      return self._cached_graph

    # Lazily generate the graph if one is not specified.
    if not self._generated_graph:
      self._generated_graph = to_func_graph(self)

    return self._generated_graph

  def call_with_captures(
      self, args: Sequence[Any], kwargs: Dict[str, Any], captures: Sequence[Any]
  ) -> Any:
    # [FUNCTION DOC]
    # Binds positional and keyword arguments to the function type, flattens them into tensors,
    # appends the unpacked captured values, and calls the preflattened entry point. Inputs: args
    # sequence, kwargs mapping, captures sequence. Returns the structured output. Binding errors
    # from the function type propagate.
    """Calls with args, kwargs, captures and returns structured output."""
    bound_parameters = self.function_type.bind(*args, **kwargs)
    tensor_inputs = self.function_type.unpack_inputs(bound_parameters)
    capture_inputs = self.function_type.unpack_captures(captures)
    return self.call_preflattened(tensor_inputs + capture_inputs)

  def call_preflattened(self, args: Sequence[core.Tensor]) -> Any:
    # [FUNCTION DOC]
    # Calls with an already flattened list of tensors (arguments followed by captures) and packs the
    # flat result into the structured output defined by the function type. Returns that output.
    """Calls with flattened tensor inputs and returns the structured output."""
    flat_outputs = self.call_flat(*args)
    return self.function_type.pack_output(flat_outputs)

  def call_flat(self, *args: core.Tensor) -> Sequence[core.Tensor]:
    # [FUNCTION DOC]
    # Executes the runtime function on flat tensors. Validates the argument count against the
    # definition's signature (value error on mismatch). The call runs inside the error
    # interpolation guard, under the function's control-capture dependencies, with gradient
    # recording paused. In eager mode it calls the function in the bound context; in graph mode it
    # adds a call op. Afterwards resource handle data is propagated to outputs and, in graph mode,
    # output shapes are set from the function type. Returns the flat outputs.
    """Calls with flat tensor inputs and returns flat tensor outputs.

    Args:
      *args: arguments to call this function with.

    Returns:
      The outputs of the function call.

    Raises:
      ValueError: if the number of arguments is incorrect.
      FunctionAlreadyGarbageCollectedError: if the function is no longer
        available to be called because it has been garbage collected.
    """
    expected_len = len(self.cached_definition.signature.input_arg)
    if len(args) != expected_len:
      raise ValueError(
          f"Signature specifies {expected_len} arguments, got: {len(args)}."
          f" Expected inputs: {self.cached_definition.signature.input_arg}."
          f" Received inputs: {args}."
          f" Function Type: {self.function_type!r}"
      )

    with InterpolateRuntimeError(self):
      with ops.control_dependencies(self._call_options.control_captures):
        # The caller must use record_operation to record this operation in the
        # eager case, so we enforce the same requirement for the non-eager
        # case by explicitly pausing recording. We don't have a gradient
        # registered for PartitionedCall, so recording this operation confuses
        # forwardprop code (GradientTape manages to ignore it).
        with record.stop_recording():
          if self._bound_context.executing_eagerly():
            outputs = self._bound_context.call_function(
                self.name,
                list(args),
                len(self.function_type.flat_outputs),
            )
          else:
            outputs = make_call_op_in_graph(
                self,
                list(args),
                self._bound_context.function_call_options.as_attrs(),
            )

    for i, output_type in enumerate(self.function_type.flat_outputs):
      handle_data = output_type.dtype._handle_data  # pylint: disable=protected-access
      if handle_data:
        handle_data_util.set_handle_data(
            outputs[i], handle_data.shape_inference
        )

    # TODO(fmuham): Use FunctionType cast here for all cases.
    if not self._bound_context.executing_eagerly():
      for i, output_type in enumerate(self.function_type.flat_outputs):
        outputs[i].set_shape(output_type.shape)

    return outputs

  def __call__(self, *args, **kwargs) -> Any:
    # [FUNCTION DOC]
    # Calls the function with positional and keyword arguments and returns its structured output.
    # Only valid for functions without captured inputs; otherwise raises a value error telling the
    # caller to use the captures-aware method. Delegates with an empty captures list.
    if self.function_type.captures:
      raise ValueError(
          "The FunctionType defines captured inputs. Use call_with_captures"
          " instead."
      )

    return self.call_with_captures(args, kwargs, [])

  def __del__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: raises a runtime error from a finalizer if the reference count goes negative;
    # exceptions in finalizers are only printed as warnings, never propagated.
    # Finalizer. Dismantles any lazily generated graph, then decrements the module-level reference
    # count for this function's (scope id, name); when it reaches zero the function is removed from
    # the bound context and the count entry is dropped. Does nothing if module state is already
    # torn down; type and attribute errors during interpreter shutdown are silently ignored.
    # Raises a runtime error when the count would become negative.
    if self._generated_graph:
      func_graph_module.dismantle_func_graph(self._generated_graph)

    if RUNTIME_FUNCTION_REFS is None:
      return

    key = (self._bound_context.function_scope_id, self.name)
    RUNTIME_FUNCTION_REFS[key] -= 1
    if RUNTIME_FUNCTION_REFS[key] < 0:
      raise RuntimeError(
          f"AtomicFunction Refcounting for {self.name} is invalid."
      )

    if RUNTIME_FUNCTION_REFS[key] == 0:
      try:
        self._bound_context.remove_function(self.name)
        RUNTIME_FUNCTION_REFS.pop(key)
      except TypeError:
        # Suppress some exceptions, mainly for the case when we're running on
        # module deletion. Things that can go wrong include the context module
        # already being unloaded, self._handle._handle_data no longer being
        # valid, and so on. Printing warnings in these cases is silly
        # (exceptions raised from __del__ are printed as warnings to stderr).
        pass  # 'NoneType' object is not callable when the handle has been
        # partially unloaded.
      except AttributeError:
        pass  # 'NoneType' object has no attribute 'eager_mode' when context has
        # been unloaded. Will catch other module unloads as well.

  def __str__(self):
    # [FUNCTION DOC]
    # Returns a short string made of a marker, the function name and its function type.
    return f"<AtomicFunction> {compat.as_str(self.name)}{self.function_type}"

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a multi-line debugging string listing name, bound context, function type, children,
    # call options and cached graph.
    return (
        f"AtomicFunction(name={self.name},\n"
        f"bound_context={self._bound_context},\n"
        f"function_type={self.function_type!r},\n"
        f"children={self._children!s},\n"
        f"call_options={self._call_options},\n"
        f"cached_graph={self._cached_graph})"
    )


def _set_read_only_resource_inputs_attr(
    op: ops.Operation, func_graph: func_graph_module.FuncGraph
):
  # [FUNCTION DOC]
  # HELPER: used by make_call_op_in_graph for automatic control dependency analysis.
  # Computes the indices of resource inputs of the function graph that are only read, and stores
  # them as an integer-list attribute on the given call op. Inputs: the op and the function
  # graph. Returns nothing; mutates the op.
  """Sets the list of resource inputs which are read-only.

  This is used by AutomaticControlDependencies.

  Args:
    op: PartitionedCall Operation.
    func_graph: FuncGraph.
  """
  read_only_indices = acd.get_read_only_resource_input_indices_graph(func_graph)
  ops.set_int_list_attr(
      op, acd.READ_ONLY_RESOURCE_INPUTS_ATTR, read_only_indices
  )


def partitioned_call_op(
    name: str,
    args: Sequence[core.Tensor],
    is_stateful: bool,
    tout: Sequence[Any],
    config: Any = None,
    executor_type: Optional[str] = None,
    xla_compile_attr: Any = None,
) -> ops.Operation:
  # [FUNCTION DOC]
  # Creates a PartitionedCall or StatefulPartitionedCall op in the default graph. Inputs: function
  # name, argument values (converted to tensors), whether it is stateful, list of output type
  # enums, optional serialized config (defaults to one disabling optimizations), optional executor
  # type (defaults to empty), optional XLA compile attribute (copied onto the call if given).
  # Uses direct op creation because bindings return nothing for functions without outputs.
  # Returns the created operation. Side effect: adds the op to the default graph.
  """Generates a function call op respecting device annotations.

  Args:
    name: Name of the function to call.
    args: The arguments of the function, including captured inputs.
    is_stateful: If the function is stateful.
    tout: a list containing the output dtypes enums
    config: (Optional) A `tensorflow::ConfigProto` proto, serialized. If `None`,
      all optimizations are disabled. Currently only handled for eager defined
      functions.
    executor_type: (Optional) A string for the name of the executor to be used
      in the function call. If not set, or set to an empty string, the default
      tensorflow executor will be used.
    xla_compile_attr: (Optional) value of the XLA compilation attribute.

  Returns:
    Returns the operation.
  """
  if config is None:
    config = function_utils.get_disabled_rewriter_config()

  if executor_type is None:
    executor_type = ""

  # The generated binding returns an empty list for functions that don't
  # return any Tensors, hence the need to use `create_op` directly.
  args = [ops.convert_to_tensor(x) for x in args]
  tin_attr = attr_value_pb2.AttrValue(
      list=attr_value_pb2.AttrValue.ListValue(
          type=[x.dtype.as_datatype_enum for x in args]
      )
  )
  tout_attr = attr_value_pb2.AttrValue(
      list=attr_value_pb2.AttrValue.ListValue(type=tout)
  )
  func_attr = attr_value_pb2.AttrValue(
      func=attr_value_pb2.NameAttrList(name=name)
  )
  executor_type_attr = attr_value_pb2.AttrValue(
      s=compat.as_bytes(executor_type)
  )

  # When running in graph mode, the graph and function graphs are optimized
  # (i.e. run through grappler) per the session options, so we can disable any
  # eager-specific rewriting.
  config_proto = attr_value_pb2.AttrValue(s=config)

  op_name = "StatefulPartitionedCall" if is_stateful else "PartitionedCall"

  # Propagate the attribute indicating the need to compile from function to the
  # call itself.
  op_attrs = {
      "Tin": tin_attr,
      "Tout": tout_attr,
      "f": func_attr,
      "config_proto": config_proto,
      "executor_type": executor_type_attr,
  }
  if xla_compile_attr is not None:
    op_attrs[attributes_lib.XLA_COMPILE] = xla_compile_attr

  op = ops.get_default_graph().create_op(
      op_name, args, tout, name=op_name, attrs=op_attrs
  )
  return op


def make_call_op_in_graph(
    atomic: AtomicFunction,
    tensor_inputs: Sequence[core.Tensor],
    context_call_attrs: Dict[str, Any],
):
  # [FUNCTION DOC]
  # Adds a call to an AtomicFunction into the current default graph. Inputs: the atomic function,
  # the flat tensor inputs and the context call attributes (needs config and executor entries).
  # Side effects: recursively adds the function and its dependencies to the graph, creates a
  # (stateful) partitioned call op, sets its read-only resource inputs attribute and collective
  # manager ids attribute. Returns the op's output tensors.
  """Adds an AtomicFunction to graph."""
  graph = ops.get_default_graph()
  graph._add_function_recursive(atomic)  # pylint: disable=protected-access

  op = partitioned_call_op(  # pytype: disable=wrong-arg-types  # always-use-property-annotation
      name=atomic.name,
      args=tensor_inputs,
      is_stateful=atomic.call_options.is_stateful,
      tout=[
          o.dtype.as_datatype_enum for o in atomic.function_type.flat_outputs
      ],
      config=context_call_attrs["config_proto"],
      executor_type=context_call_attrs["executor_type"],
      xla_compile_attr=atomic.cached_definition.attr.get(
          attributes_lib.XLA_COMPILE, None
      ),
  )
  _set_read_only_resource_inputs_attr(op, atomic.graph)

  ops.set_int_list_attr(
      op,
      acd.COLLECTIVE_MANAGER_IDS,
      atomic._call_options.collective_manager_ids_used,  # pylint: disable=protected-access
  )

  return op.outputs


def from_function_def(
    function_def: function_pb2.FunctionDef,
    function_type: function_type_lib.FunctionType,
) -> AtomicFunction:
  # [FUNCTION DOC]
  # Wraps a function definition already built elsewhere. Inputs: a FunctionDef proto and its
  # FunctionType. Registers the definition in the current eager context and returns a new
  # AtomicFunction. Raises a value error if a function with the same name is already registered.
  """Create a new AtomicFunction from FunctionDef + FunctionType."""
  bound_context = context.context()
  if bound_context.has_function(compat.as_bytes(function_def.signature.name)):
    raise ValueError("Function already registered in context.")

  bound_context.add_function_def(function_def)
  return AtomicFunction(
      function_def.signature.name, bound_context, function_type
  )


def from_func_graph(
    name: Union[str, bytes],
    graph: func_graph_module.FuncGraph,
    attrs: Dict[str, attr_value_pb2.AttrValue],
    function_type: Optional[function_type_lib.FunctionType] = None,
    overwrite: bool = False,
) -> AtomicFunction:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the error text for the implements-attribute check contains duplicated words
  # ("use  use") and is raised through an assertion, which is skipped when assertions are disabled.
  # Converts a FuncGraph into a runtime function and wraps it. Inputs: name of the function,
  # the graph, a mapping of attribute names to attribute values, an optional function type (derived
  # from the graph if absent) and an overwrite flag. If the implements attribute is present, it
  # asserts the function has no resource-variable inputs and no captures. Collects non-input
  # operations, derives output names (falling back to automatic naming on missing or duplicate
  # names), builds the C function, sets parsed attributes, optionally removes an existing function
  # of the same name, registers the function in the current context, and frees the temporary C
  # function. Call options (collective ids, control captures, statefulness) are taken from the
  # graph. Returns the new AtomicFunction. Raises an assertion error on the implements violation.
  """Initializes an AtomicFunction from FuncGraph.

  Args:
    name: str, the name for the created function.
    graph: Graph, the graph containing the operations in the function
    attrs: dict mapping names of attributes to their AttrValue values
    function_type: known FunctionType to use, otherwise one is derived.
    overwrite: overwrites function definition in the current context if needed

  Returns:
    An AtomicFunction instance.
  """
  if attrs and attributes_lib.IMPLEMENTS in attrs:
    # The alternative is to silently drop "implements" tag
    # but it seems likely it would lead to hard to catch bugs.
    # Another alternative is to make func_body to preserve the order
    # of arguments if variables are present. Yet another option
    # is to automatically replace variables as arguments to functions
    # to v.read_value() whenever "implements" tag is present
    # Anytime we annotate existing function we probably want to wrap
    # it with safe read_value for backward compatibility.
    has_resource_vars = any(
        inp.dtype == dtypes.resource for inp in graph.inputs
    )

    captured_inputs = graph.external_captures + graph.deferred_external_captures
    assert not any(
        (has_resource_vars, captured_inputs)
    ), (
        'Function {name} has "{attr}={value}" attribute and thus can not '
        "depend on any tensors outside of its signature or modify variables. "
        "\n\nNote: variables are always captured and cause function "
        "re-tracing for every variable called.\n"
        "  inputs: {inputs}\n  captures: {captured}\n\n"
        "To pass a variable to such function use  "
        "use variable.read_value().".format(
            name=graph.name,
            attr=attributes_lib.IMPLEMENTS,
            value=attrs[attributes_lib.IMPLEMENTS],
            inputs=graph.inputs,
            captured=captured_inputs,
        )
    )

  input_ops = set(arg.op for arg in graph.inputs)
  operations = [op for op in graph.get_operations() if op not in input_ops]

  graph_output_names = graph._output_names  # pylint: disable=protected-access
  if graph_output_names is not None and all(
      ops.tensor_id(t) in graph_output_names for t in graph.outputs
  ):
    output_names = [
        compat.as_bytes(graph_output_names[ops.tensor_id(t)])
        for t in graph.outputs
    ]
    if len(set(output_names)) != len(output_names):
      # There are duplicate names for some reason, probably an invalid
      # signature. Revert to auto-naming.
      output_names = []
  else:
    output_names = []
  with graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
    fn = pywrap_tf_session.TF_GraphToFunction_wrapper(
        c_graph,
        compat.as_str(name),
        False,
        [o._c_op for o in operations],  # pylint: disable=protected-access
        [t._as_tf_output() for t in graph.inputs],  # pylint: disable=protected-access
        [t._as_tf_output() for t in graph.outputs],  # pylint: disable=protected-access
        output_names,
        [o._c_op for o in graph.control_outputs],  # pylint: disable=protected-access
        [],  # control_output_names
        None,
        compat.as_str(""),
    )

  attrs = attributes_lib.parse_func_attrs(attrs or {})
  for attr_name, attr_value in attrs.items():
    serialized = attr_value.SerializeToString()
    pywrap_tf_session.TF_FunctionSetAttrValueProto(
        fn, compat.as_str(attr_name), serialized
    )

  name = compat.as_bytes(name)
  bound_context = context.context()

  if overwrite and bound_context.has_function(name):
    bound_context.remove_function(name)

  bound_context.add_c_function(fn)
  pywrap_tf_session.TF_DeleteFunction(fn)

  call_options = CallOptions(
      collective_manager_ids_used=getattr(
          graph, "collective_manager_ids_used", []
      ),
      control_captures=graph.function_captures.control,
      is_stateful=any(op._is_stateful for op in operations),  # pylint: disable=protected-access
  )

  if not function_type:
    function_type = function_type_utils.derive_from_graph(graph)

  return AtomicFunction(
      name,
      bound_context,
      function_type,
      list(graph._functions.values()),  # pylint: disable=protected-access,
      call_options,
      cached_graph=graph,
  )


def to_func_graph(atomic: AtomicFunction) -> func_graph_module.FuncGraph:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the collective manager ids attribute is assigned a one-element tuple wrapping the
  # list (trailing comma), not the list itself, so the resulting graph holds a nested value that
  # differs from the type used elsewhere (a plain list of integers).
  # Rebuilds a FuncGraph from an AtomicFunction. Input: the atomic function. Converts its function
  # type to structured input/output signatures, then, in a fresh default graph, registers all
  # child functions and converts the runtime function definition to a graph (propagating device
  # specs, excluding library functions). Children are added to the result, input and output
  # shapes and resource handle data are restored from the function type, and the collective
  # manager ids are copied. Returns the new function graph.
  """Generate a FuncGraph from an AtomicFunction."""
  # pylint: disable=protected-access
  input_signature, output_signature = function_type_lib.to_structured_signature(
      atomic.function_type
  )

  with ops.Graph().as_default():
    # Insert dependencies in the default graph so the new graph can pull them.
    for f in atomic.children:
      ops.get_default_graph()._add_function(f)

    result = function_def_to_graph.function_def_to_graph(
        atomic.definition,
        structured_input_signature=input_signature,
        structured_outputs=output_signature,
        propagate_device_spec=True,
        include_library_functions=False,
    )
    for f in atomic.children:
      result._add_function(f)

  # Set input shapes and handle data
  for i, input_type in enumerate(atomic.function_type.flat_inputs):
    handle_data = input_type.dtype._handle_data
    if handle_data:
      handle_data_util.set_handle_data(
          result.inputs[i], handle_data.shape_inference
      )
    result.inputs[i].set_shape(input_type.shape)

  # Set output shapes and handle data
  for i, output_type in enumerate(atomic.function_type.flat_outputs):
    handle_data = output_type.dtype._handle_data
    if handle_data:
      handle_data_util.set_handle_data(
          result.outputs[i], handle_data.shape_inference
      )
    result.outputs[i].set_shape(output_type.shape)

  result.collective_manager_ids_used = (
      atomic.call_options.collective_manager_ids_used,
  )

  # pylint: enable=protected-access
  return result


class InterpolateRuntimeError(object):
  # [CLASS DOC]
  # HELPER: internal glue used by AtomicFunction.call_flat to improve runtime error messages.
  # Context manager wrapped around a function call. If the call raises a runtime op error, the
  # error message is rewritten to show Python stack traces of the nodes that failed, using the
  # function (and nested child function) debug info. Holds the top-level function; has a class
  # level list of phrases that cause stack frames to be omitted. Never suppresses exceptions.
  """Context Manager that interpolates exceptions received by AtomicFunction."""

  DENY_LIST_PHRASES = ["<embedded"]

  def __init__(self, top_level_func):
    # [FUNCTION DOC]
    # Stores the top-level AtomicFunction whose call is being guarded. Input: that function.
    # Returns nothing.
    self._func = top_level_func

  def interpolate(self, message, node_names, graph_debug_info):
    # [FUNCTION DOC]
    # Builds a human-readable "graph execution error" message with user stack traces.
    # Inputs: message (the original error text, stripped and appended last), node_names (list of
    # graph node names where the error was detected) and graph_debug_info (debug-info proto used
    # to load stack traces). For each node it adds a "detected at node" header followed by the
    # formatted frames of its trace, skipping frames containing any denied phrase, or a note that
    # traces are unavailable. Returns the joined multi-line string. No side effects.
    """Uses the GraphDebugInfo to generate an error message."""
    error_message = ["Graph execution error:", ""]
    traces = tf_stack.LoadTracesFromDebugInfo(graph_debug_info)

    for node_name in node_names:
      error_message.append(
          f"Detected at node {node_name} defined at (most recent call last):"
      )
      if node_name in traces:
        stack_trace = traces[node_name]
        for formatted_frame in traceback.format_list(stack_trace):
          if not any(p in formatted_frame for p in self.DENY_LIST_PHRASES):
            error_message.append(formatted_frame)
      else:
        error_message.append("<stack traces unavailable>")

    error_message.append(message.strip())
    return "\n".join(error_message)

  def __enter__(self):
    # [FUNCTION DOC]
    # Context-manager entry hook. Does nothing and returns nothing (so a with-as target would be
    # the null value); all work happens on exit.
    pass

  def __exit__(self, typ, exc, tb):
    # [FUNCTION DOC]
    # Context-manager exit hook that rewrites the message of runtime op errors.
    # Inputs: exception type, exception instance and traceback (any may be empty when no error
    # occurred). If there is no exception, or it is not an op error, nothing is changed.
    # Otherwise the message is parsed into function tags and node tags; the tags are matched
    # against the wrapped function and its children (walking down the children chain) to find the
    # deepest function, and the exception's private message is replaced by an interpolated message
    # built from that function's debug info. Side effect: mutates the exception message.
    # Always returns false, so the exception is never suppressed.
    if not exc or not isinstance(exc, errors.OpError):
      return False
    exc = typing.cast(errors.OpError, exc)
    message = compat.as_text(exc.message)
    parsed_message, func_tags, node_tags = error_interpolation.parse_message(
        message
    )
    deepest_func = None
    for func_tag in func_tags:
      if func_tag.name == compat.as_str(self._func.name):
        deepest_func = self._func
      elif deepest_func:
        next_func = None
        for child_func in deepest_func.children:
          if func_tag.name == compat.as_str(child_func.name):
            next_func = child_func
            break
        if next_func is not None and isinstance(next_func, AtomicFunction):
          deepest_func = next_func
    if deepest_func:
      exc._message = self.interpolate(
          parsed_message,
          [t.name for t in node_tags],
          deepest_func.graph_debug_info,
      )
    return False
