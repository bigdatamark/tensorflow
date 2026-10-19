# [MODULE DOC]
# Core of compiling Python functions into graphs by tracing. Exposes an enumeration of tracing
# scopes, a dataclass of tracing options, a function that traces then calls, a function that
# traces and returns a concrete function, and private helpers that look up the function cache,
# build a new graph, and set argument keywords. Used by the polymorphic function wrapper and
# related code. Import-time side effect: creates a monitoring counter for graph building time.
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
# ==============================================================================
"""Compile Python functions to TF graphs using tracing."""

import contextlib
import dataclasses
import enum
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache as function_cache_lib
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import concrete_function as concrete_function_lib
from tensorflow.python.eager.polymorphic_function import function_context
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.util import compat

_graph_building_time_counter = monitoring.Counter(
    "/tensorflow/core/tf_function/graph_building_time_usecs",
    "Time for tf.function to build a graph (us).",
)


class ScopeType(enum.Enum):
  # [CLASS DOC]
  # Enumeration of the scopes under which a function may be traced: no special scope, a scope in
  # which variable creation is allowed, and a scope in which variable creation is disallowed.
  # The value is part of the function context used as a cache key, so traces made under
  # different scopes are kept separate.
  """Enumerate scopes under which functions might be traced."""
  NO_SCOPE = 1
  VARIABLE_CREATION = 2
  NO_VARIABLE_CREATION = 3


@dataclasses.dataclass
class TracingOptions:
  # [CLASS DOC]
  # Plain data container (a dataclass) holding every setting for tracing a Python function into
  # a graph: the function itself, its name, its polymorphic type and default parameter values,
  # the tracing scope, attributes for the generated function definition, autograph settings,
  # whether to reduce retracing, whether the graph lifetime is bound to the function, and the
  # optional shared function cache, capture tracker and lock. Created by callers (such as the
  # polymorphic function wrapper) and passed to the tracing entry points. After construction it
  # validates attributes and fills in type and defaults when absent, and derives the input
  # signature.
  """Configuration options for tracing."""
  # Python function to trace.
  python_function: Callable[[Any], Any] = lambda *args, **kwargs: None

  # Name given to the traced function.
  name: str = "function"

  # Known FunctionType of the python function.
  polymorphic_type: Optional[function_type_lib.FunctionType] = None

  # Known default values for the python function parameters.
  default_values: Optional[Dict[str, Any]] = None

  # Identifies effecting scope under which the function is traced.
  scope_type: ScopeType = ScopeType.NO_SCOPE

  # FunctionDef attributes for traced function.
  attributes: Optional[Dict[str, Any]] = None

  # See https://www.tensorflow.org/guide/autograph for more information.
  # If autograph is enabled.
  autograph: bool = True
  # Optional tuple of `tf.autograph.experimental.Feature` values.
  autograph_options: Optional[Tuple[Any, ...]] = None

  # Trace generalized functions where possible to avoid future retracing.
  reduce_retracing: bool = False

  # If true, graph of generated Function will be destroyed with the function.
  bind_graph_to_function: bool = False

  # A FunctionCache object that holds existing traced functions.
  function_cache: Optional[function_cache_lib.FunctionCache] = None

  # A FunctionCaptures object that tracks by-ref captures.
  function_captures: Optional[capture_container.FunctionCaptures] = None

  # If specified, guards tracing and function lookup
  lock: Optional[threading.Lock] = None

  def __post_init__(self):
    # [FUNCTION DOC]
    # Runs after dataclass construction. Validates that every requested function attribute is in
    # the allowed list for tracing compilation, raising a value error naming the first
    # unsupported one. Then, if the polymorphic type is missing or the default values are
    # nothing, both are derived from the Python function's signature (overwriting any partially
    # supplied one of the two). Finally computes and stores the input signature from the type.
    # Mutates the instance; returns nothing.
    if self.attributes:
      for attribute in self.attributes:
        if attribute not in attributes_lib.TRACING_COMPILATION_ALLOWLIST:
          raise ValueError(
              f"Tracing compilation does not support `{attribute}` as an"
              " attribute."
          )

    if not self.polymorphic_type or self.default_values is None:
      self.polymorphic_type = function_type_lib.FunctionType.from_callable(
          self.python_function
      )
      self.default_values = function_type_lib.FunctionType.get_default_values(
          self.python_function
      )

    self._input_signature = function_type_utils.to_input_signature(
        self.polymorphic_type
    )

  @property
  def is_pure(self):
    # [FUNCTION DOC]
    # Property telling whether the function is considered pure, which is the case when the
    # attributes mapping exists and contains the implements marker. KNOWN ISSUE: returns the
    # attributes value itself when it is empty or missing (nothing or an empty mapping) rather
    # than a strict true/false value, so callers should only rely on truthiness.
    return self.attributes and attributes_lib.IMPLEMENTS in self.attributes

  @property
  def input_signature(self):
    # [FUNCTION DOC]
    # Read-only property returning the input signature computed during construction from the
    # polymorphic type (nothing if the type does not define a fixed signature).
    return self._input_signature


def call_function(args=None, kwargs=None, tracing_options=None):
  # [FUNCTION DOC]
  # Traces the Python function for the given arguments and then immediately executes the
  # resulting concrete function. Inputs: positional arguments (sequence, default empty), keyword
  # arguments (mapping, default empty) and tracing options (default: a freshly built default
  # options object). Obtains a concrete function via the trace entry point, binds the arguments
  # directly against its type (skipping the usual canonicalization), flattens them and calls the
  # concrete function with its captured inputs. Returns the call's output. Errors from tracing
  # or binding propagate. Side effects: may add a trace to the function cache and run ops.
  """Traces a function for args and kwargs and calls it after."""
  if not tracing_options:
    tracing_options = TracingOptions()

  args = args if args else ()
  kwargs = kwargs if kwargs else {}
  function = trace_function(
      args=args, kwargs=kwargs, tracing_options=tracing_options
  )

  # Bind it ourselves to skip unnecessary canonicalization of default call.
  bound_args = function.function_type.bind(*args, **kwargs)
  flat_inputs = function.function_type.unpack_inputs(bound_args)
  return function._call_flat(  # pylint: disable=protected-access
      flat_inputs, captured_inputs=function.captured_inputs
  )


def trace_function(args=None, kwargs=None, tracing_options=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring has no Returns or Raises section; it actually returns the
  # concrete function and may raise errors from input binding or tracing.
  # Public tracing entry point. Inputs: positional arguments, keyword arguments (both optional,
  # defaulting to empty) and tracing options (default: a fresh default options object). If the
  # options carry an input signature and arguments were given, the arguments are first bound
  # against the type and defaults to validate them. Under the options' lock (if any), when a
  # signature exists and no arguments were given, the signature stands in for the arguments;
  # then the function is looked up or traced. Unless the graph is bound to the function, the
  # concrete function's garbage collector is released so the graph is not destroyed with it.
  # Returns the concrete function. Side effects: may populate the function cache.
  """Returns a `ConcreteFunction` specialized to inputs and execution context.

  Compiles a Graph corresponding to the Python function logic and uses that
  to generate a differentiable ConcreteFunction.

  Args:
    args: inputs to specialize on. Can be concrete values (e.g. 1) or
      `tf.Tensor` or `tf.TensorSpec`.
    kwargs: keyword inputs to specialize on. Concrete values (e.g. 1) or
      `tf.Tensor` or `tf.TensorSpec`.
    tracing_options: TracingOptions for the tracing process.
  """
  if not tracing_options:
    tracing_options = TracingOptions()

  args = args if args else ()
  kwargs = kwargs if kwargs else {}

  if tracing_options.input_signature and (args or kwargs):
    # Check to see if a valid type can be generated from the args, kwargs
    bound_args = function_type_utils.bind_function_inputs(
        args,
        kwargs,
        tracing_options.polymorphic_type,
        tracing_options.default_values,
    )
    args, kwargs = bound_args.args, bound_args.kwargs

  with tracing_options.lock or contextlib.nullcontext():
    if tracing_options.input_signature and not args and not kwargs:
      args = tracing_options.input_signature
      kwargs = {}

    concrete_function = _maybe_define_function(
        args, kwargs, tracing_options
    )

  if not tracing_options.bind_graph_to_function:
    concrete_function._garbage_collector.release()  # pylint: disable=protected-access

  return concrete_function


def _maybe_define_function(args, kwargs, tracing_options):
  # [FUNCTION DOC]
  # HELPER: cache lookup and tracing core for the tracing entry point.
  # Canonicalizes the arguments against the polymorphic type and defaults, substitutes the fixed
  # input signature for leading positional arguments when one exists, computes the current
  # function context and a canonical monomorphic type (taking capture types into account), and
  # looks it up in the function cache if present. On a hit returns the cached concrete function.
  # On a miss builds a new graph under the autograph status from the options, optionally
  # generalizes the type when reducing retracing (only without an input signature and with a
  # cache), creates the concrete function, adds it to the cache, and returns it. Graph building
  # time is recorded with a timer only when not already inside another function, and verbose
  # logging is emitted. Raises value, type or runtime errors as listed in its docstring.
  """Gets a function for these inputs, defining it if necessary.

  Args:
    args: The varargs for the Python function.
    kwargs: The keyword args for the Python function.
    tracing_options: TracingOptions for the tracing process.

  Returns:
    A ConcreteFunction generated based on args, kwargs and tracing_options.

  Raises:
    ValueError: If inputs are incompatible with the input signature.
    TypeError: If the function inputs include non-hashable objects
    RuntimeError: If there's an internal bug (inconsistency) in handling
      shape relaxation retracing.
  """
  bound_args = function_type_utils.canonicalize_function_inputs(
      args,
      kwargs,
      tracing_options.polymorphic_type,
      tracing_options.default_values,
      tracing_options.is_pure,
  )
  args, kwargs = bound_args.args, bound_args.kwargs

  if tracing_options.input_signature is not None:
    args = (
        *tracing_options.input_signature,
        *args[len(tracing_options.input_signature) :],
    )

  current_func_context = function_context.make_function_context(
      tracing_options.scope_type
  )

  capture_types = (
      tracing_options.function_captures.capture_types
      if tracing_options.function_captures
      else {}
  )
  lookup_func_type, lookup_func_context = (
      function_type_utils.make_canonicalized_monomorphic_type(
          args,
          kwargs,
          capture_types,
          tracing_options.polymorphic_type,
      )
  )

  if tracing_options.function_cache is not None:
    concrete_function = tracing_options.function_cache.lookup(
        lookup_func_type, current_func_context
    )
  else:
    concrete_function = None

  if concrete_function is not None:
    return concrete_function

  # Use a timer for graph building only if not already inside a function. This
  # avoids double counting graph building time for nested functions.
  with monitoring.MonitoredTimer(
      _graph_building_time_counter.get_cell()
  ) if not ops.inside_function() else contextlib.nullcontext():
    with trace.Trace("tf.function-graph_building"):
      logging.vlog(
          1,
          "Creating new FuncGraph for Python function %r (key: %r, %r)",
          tracing_options.python_function,
          current_func_context,
          lookup_func_type,
      )
      logging.vlog(
          2, "Python function signature [args: %s] [kwargs: %s]", args, kwargs
      )
      ag_status = (
          ag_ctx.Status.ENABLED
          if tracing_options.autograph
          else ag_ctx.Status.DISABLED
      )
      with ag_ctx.ControlStatusCtx(
          status=ag_status, options=tracing_options.autograph_options
      ):
        func_graph = func_graph_module.FuncGraph(tracing_options.name)
        if (
            tracing_options.input_signature is None
            and tracing_options.reduce_retracing
            and tracing_options.function_cache
        ):
          target_func_type = tracing_options.function_cache.generalize(
              current_func_context, lookup_func_type
          )
        else:
          target_func_type = lookup_func_type
        concrete_function = _create_concrete_function(
            target_func_type, lookup_func_context, func_graph, tracing_options
        )

        if tracing_options.function_cache is not None:
          tracing_options.function_cache.add(
              concrete_function, current_func_context
          )

        return concrete_function


def _create_concrete_function(
    function_type, type_context, func_graph, tracing_options
):
  # [FUNCTION DOC]
  # HELPER: graph construction step of the tracing core.
  # KNOWN ISSUE: the docstring says the function is created from arguments and keyword
  # arguments, but the function actually takes a function type, a type context, an empty graph
  # and tracing options.
  # Creates placeholders for the type's arguments inside the graph, traces the Python function
  # into the graph (adding automatic control dependencies unless disabled by the attributes),
  # applies registered graph transforms, merges by-reference captures into the options' capture
  # tracker when present, and builds a new function type including captures and the output type.
  # Wraps the graph in a concrete function that cleans up its graph when it goes out of scope,
  # sets the argument keywords, runs concrete function callbacks, and returns it.
  """Create a `ConcreteFunction` from `args`, `kwargs`, and `func_graph`."""
  placeholder_context = trace_type.InternalPlaceholderContext(
      func_graph, type_context.get_placeholder_mapping()
  )
  with func_graph.as_default():
    placeholder_bound_args = function_type.placeholder_arguments(
        placeholder_context
    )

  disable_acd = tracing_options.attributes and tracing_options.attributes.get(
      attributes_lib.DISABLE_ACD, False
  )
  traced_func_graph = func_graph_module.func_graph_from_py_func(
      tracing_options.name,
      tracing_options.python_function,
      placeholder_bound_args.args,
      placeholder_bound_args.kwargs,
      None,
      func_graph=func_graph,
      add_control_dependencies=not disable_acd,
      arg_names=function_type_utils.to_arg_names(function_type),
      create_placeholders=False,
  )

  transform.apply_func_graph_transforms(traced_func_graph)

  graph_capture_container = traced_func_graph.function_captures

  if tracing_options.function_captures:
    # Maintain the list of all captures
    tracing_options.function_captures.merge_by_ref_with(graph_capture_container)

  # Create a new FunctionType including captures and outputs.
  output_type = trace_type.from_value(
      traced_func_graph.structured_outputs, type_context
  )
  traced_func_type = function_type_lib.FunctionType(
      function_type.parameters.values(),
      traced_func_graph.function_captures.capture_types,
      return_annotation=output_type,
  )

  concrete_function = concrete_function_lib.ConcreteFunction.from_func_graph(
      traced_func_graph,
      traced_func_type,
      tracing_options.attributes,
      # Tell the ConcreteFunction to clean up its graph once it goes out of
      # scope. This is not the default behavior since it gets used in some
      # places (like Keras) where the FuncGraph lives longer than the
      # ConcreteFunction.
      shared_func_graph=False,
  )
  _set_arg_keywords(concrete_function)
  transform.call_concrete_function_callbacks(concrete_function)

  return concrete_function


def _set_arg_keywords(concrete_function):
  # [FUNCTION DOC]
  # HELPER: post-processing for newly created concrete functions.
  # Mutates the given concrete function: sets its argument keyword list, one unique name per
  # non-captured graph input, taken from each input's user-specified name (default generic
  # tensor name when absent); duplicates get a numeric suffix. Also records the number of
  # positional arguments (graph inputs excluding captures). Returns nothing.
  """Sets arg keywords for ConcreteFunction."""
  seen_names = set()
  concrete_function._arg_keywords = []  # pylint: disable=protected-access
  prefix_counts = {}
  graph = concrete_function.graph
  num_captures = len(graph.internal_captures + graph.deferred_internal_captures)
  num_positional = len(graph.inputs) - num_captures
  for arg in concrete_function.graph.inputs[:num_positional]:
    try:
      user_arg_name = compat.as_str(arg.op.get_attr("_user_specified_name"))
    except ValueError:
      user_arg_name = "tensor_arg"
    proposal = user_arg_name
    while proposal in seen_names:
      index = prefix_counts.get(user_arg_name, 1)
      proposal = "{}_{}".format(user_arg_name, index)
      prefix_counts[user_arg_name] = index + 1
    seen_names.add(proposal)
    concrete_function._arg_keywords.append(proposal)  # pylint: disable=protected-access
  # Anything can be a positional argument, in the same order as .inputs
  concrete_function._num_positional_args = (  # pylint: disable=protected-access
      num_positional
  )
