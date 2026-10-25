# [MODULE DOC]
# Tools for deserializing saved functions. Rebuilds polymorphic functions (restored functions that
# dispatch among saved concrete functions), bare concrete functions, and whole function libraries
# from SavedModel protocol buffers, renaming functions and shared resource names so loads do not
# collide, re-registering custom gradients, and restoring function types from saved function specs.
# Used by the SavedModel loader. Import-time side effects: none beyond imports and a regular
# expression constant for cleaning inference-wrapper function names.
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
# ==============================================================================
"""Tools for deserializing `Function`s."""

import collections
import pprint
import re

from absl import logging

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_lib
from tensorflow.python.framework import function_def_to_graph as function_def_lib
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect


def _is_tensor(t):
  # [FUNCTION DOC]
  # HELPER: type check used when matching call arguments against restored function signatures.
  # Returns true if the value t is a tensor or a resource variable, false otherwise.
  return isinstance(
      t, (tensor.Tensor, resource_variable_ops.BaseResourceVariable))


# TODO(b/205016027): Update this to just use ConcreteFunction.__call__ with the
# structured signature.
def _call_concrete_function(function, inputs):
  # [FUNCTION DOC]
  # HELPER: executes the chosen concrete function for restored polymorphic functions.
  # Calls a restored concrete function with structured inputs (a pair of positional and keyword
  # arguments matching the function's structured input signature). The inputs are flattened
  # against the expected signature; tensor-spec slots are converted to tensors (using the spec
  # dtype as a hint), variable-spec slots contribute the variable handle, other slots are
  # dropped. The flat call is made with the function's captured inputs. Returns the function
  # output, or nothing/null if the result is an operation. Does not verify that non-tensor
  # inputs match. Raises whatever structure flattening or tensor conversion raises on mismatch.
  """Calls a restored Function with structured inputs.

  This differs from `function.__call__` in that inputs and outputs are
  structured and that it casts inputs to tensors if needed.

  Note: this does not checks that non-tensor inputs match. That should be
  done before via `_concrete_function_callable_with`.

  Args:
    function: ConcreteFunction to call.
    inputs: Structured inputs compatible with
      `function.graph.structured_input_signature`.

  Returns:
    The structured function output.
  """
  expected_structure = function.graph.structured_input_signature
  flatten_inputs = nest.flatten_up_to(
      expected_structure, inputs, expand_composites=True)
  flatten_expected = nest.flatten(expected_structure, expand_composites=True)
  tensor_inputs = []
  for arg, expected in zip(flatten_inputs, flatten_expected):
    if isinstance(expected, tensor.TensorSpec):
      tensor_inputs.append(
          ops.convert_to_tensor(arg, dtype_hint=expected.dtype))
    elif isinstance(expected, resource_variable_ops.VariableSpec):
      tensor_inputs.append(arg.handle)
  result = function._call_flat(tensor_inputs, function.captured_inputs)  # pylint: disable=protected-access
  if isinstance(result, ops.Operation):
    return None
  return result


def _try_convert_to_tensor_spec(arg, dtype_hint):
  # [FUNCTION DOC]
  # HELPER: lets signature matching accept values convertible to tensors.
  # Tries to convert arg to a tensor (with dtype_hint as preferred type) inside a scratch function
  # graph so the current context is not polluted, and returns a tensor spec with the resulting
  # shape and dtype. Returns nothing/null if conversion raises a type or value error.
  """Returns None or TensorSpec obtained if `arg` is converted to tensor."""
  try:
    # Note: try conversion in a FuncGraph to avoid polluting current context.
    with func_graph_lib.FuncGraph(name="guess_conversion").as_default():
      result = ops.convert_to_tensor(arg, dtype_hint=dtype_hint)
      return tensor.TensorSpec(shape=result.shape, dtype=result.dtype)
  except (TypeError, ValueError):
    return None


def _concrete_function_callable_with(function, inputs, allow_conversion):
  # [FUNCTION DOC]
  # HELPER: overload resolution for restored functions.
  # Decides whether a concrete function can be called with inputs (positional/keyword pair).
  # The inputs must be structurally compatible with the function's structured input signature.
  # For tensor specs: when allow_conversion is true the argument is first converted to a spec;
  # it must then be a tensor/variable/spec with equal dtype and a compatible shape. For other
  # type specs the spec must be compatible with the argument. For tensor-valued arguments where
  # a plain value was expected, identity must match; other values must compare equal.
  # Returns true or false; never raises for structure mismatches. No side effects.
  """Returns whether concrete `function` can be called with `inputs`."""
  expected_structure = function.graph.structured_input_signature
  try:
    flatten_inputs = nest.flatten_up_to(expected_structure, inputs)
  except (TypeError, ValueError):
    return False

  for arg, expected in zip(flatten_inputs, nest.flatten(expected_structure)):
    if isinstance(expected, tensor.TensorSpec):
      if allow_conversion:
        arg = _try_convert_to_tensor_spec(arg, dtype_hint=expected.dtype)
      if not _is_tensor(arg) and not isinstance(arg, tensor.TensorSpec):
        return False
      if arg.dtype != expected.dtype:
        return False
      if not expected.shape.is_compatible_with(arg.shape):
        return False
    elif isinstance(expected, type_spec.TypeSpec):
      if not expected.is_compatible_with(arg):
        return False
    elif _is_tensor(arg):
      if id(arg) != id(expected):
        return False
    else:
      if arg != expected:
        return False
  return True


def _deserialize_function_spec_as_nonmethod(function_spec_proto):
  # [FUNCTION DOC]
  # HELPER: rebuilds function specs while loading a saved model.
  # Decodes a function spec protocol buffer into a function spec object. The argument spec and
  # input signature are decoded; if the proto says it was a method (or the first argument is named
  # self) the leading self argument is removed so the function behaves as a non-method. The jit
  # compile enum is mapped to unset/true/false. Returns the new function spec. Raises a
  # not-implemented error if it is a method but has no named arguments.
  """Deserialize a FunctionSpec object from its proto representation."""
  typeless_fullargspec = nested_structure_coder.decode_proto(
      function_spec_proto.fullargspec)

  # Convert a method function into a non method.
  if function_spec_proto.is_method or (
      typeless_fullargspec.args and typeless_fullargspec.args[0] == "self"
  ):
    if not typeless_fullargspec.args:
      raise NotImplementedError(
          "Cannot deserialize a method function without a named "
          "'self' argument.")
    args = typeless_fullargspec.args[1:]
  else:
    args = typeless_fullargspec.args

  fullargspec = tf_inspect.FullArgSpec(
      args=args,
      varargs=typeless_fullargspec.varargs,
      varkw=typeless_fullargspec.varkw,
      defaults=typeless_fullargspec.defaults,
      kwonlyargs=typeless_fullargspec.kwonlyargs,
      kwonlydefaults=typeless_fullargspec.kwonlydefaults,
      annotations=typeless_fullargspec.annotations)
  input_signature = nested_structure_coder.decode_proto(
      function_spec_proto.input_signature)

  # See `tf.function` and the JitCompile proto for details.
  jit_compile = {
      saved_object_graph_pb2.FunctionSpec.JitCompile.DEFAULT: None,
      saved_object_graph_pb2.FunctionSpec.JitCompile.ON: True,
      saved_object_graph_pb2.FunctionSpec.JitCompile.OFF: False,
  }.get(function_spec_proto.jit_compile)

  return function_type_utils.FunctionSpec.from_fullargspec_and_signature(
      fullargspec=fullargspec,
      input_signature=input_signature,
      jit_compile=jit_compile)


# TODO(b/203440205): Set FunctionType with ConcreteFunction constructor.
def set_preinitialized_function_spec(concrete_fn, spec):
  # [FUNCTION DOC]
  # Sets the function type of a restored concrete function from a function spec. If spec is
  # nothing/null the function type is cleared. Otherwise the spec's parameters are made
  # unconstrained (no type), the concrete function's structured input signature is canonicalized
  # against it using the spec's default values, and the output type is derived from the graph's
  # structured outputs. The resulting function type replaces the private function type of
  # concrete_fn (mutated in place). Returns nothing. Captures are added later at restore time.
  """Set the FunctionType of the ConcreteFunction using FunctionSpec."""
  if spec is None:
    concrete_fn._function_type = None  # pylint: disable=protected-access
    return

  unconstrained_type = function_type_lib.FunctionType(
      [
          function_type_lib.Parameter(p.name, p.kind, p.optional, None)
          for p in spec.function_type.parameters.values()
      ]
  )
  arg_specs, kwarg_specs = concrete_fn.structured_input_signature

  input_function_type, _ = function_type_lib.canonicalize_to_monomorphic(
      arg_specs,
      {
          function_type_lib.sanitize_arg_name(k): v
          for k, v in kwarg_specs.items()
      },
      spec.default_values,
      {},
      unconstrained_type,
  )

  output_type = trace_type.from_value(concrete_fn.graph.structured_outputs)
  # Captures are restored later so we will update it then.
  function_type = function_type_lib.FunctionType(
      input_function_type.parameters.values(),
      return_annotation=output_type,
  )
  concrete_fn._function_type = function_type  # pylint: disable=protected-access


# TODO(b/205016761): The fact that we can't derive ConcreteFunction calling
# conventions from the serialized input spec right now is unfortunate. Merging
# these would be good, maybe by adding TensorSpec names to cache keys so renamed
# keyword arguments would yield different ConcreteFunctions.
def setup_bare_concrete_function(saved_bare_concrete_function,
                                 concrete_functions):
  # [FUNCTION DOC]
  # Makes a restored bare concrete function (one saved without a wrapping polymorphic function)
  # callable. Inputs: the saved bare-concrete-function proto and a mapping from concrete function
  # names to restored concrete functions. Looks up the function by name, sets its argument
  # keywords and number of positional arguments from the proto, applies the saved function spec
  # (if present) as its function type, and adds it to the current graph. Returns that concrete
  # function. Mutates the looked-up function. Raises a key error if the name is missing.
  """Makes a restored bare concrete function callable."""
  concrete_function = concrete_functions[
      saved_bare_concrete_function.concrete_function_name]
  # pylint: disable=protected-access
  concrete_function._arg_keywords = (
      saved_bare_concrete_function.argument_keywords)
  concrete_function._num_positional_args = (
      saved_bare_concrete_function.allowed_positional_arguments)
  if saved_bare_concrete_function.HasField("function_spec"):
    function_spec = _deserialize_function_spec_as_nonmethod(
        saved_bare_concrete_function.function_spec)
    set_preinitialized_function_spec(concrete_function, function_spec)
  # pylint: enable=protected-access
  concrete_function.add_to_graph()
  return concrete_function


class RestoredFunction(def_function.Function):
  # [CLASS DOC]
  # A polymorphic function recreated from a saved model. It wraps a generated Python body that
  # dispatches to the already-built concrete functions, holds the list of concrete functions,
  # the function type and the default values from the saved function spec. Autograph is disabled
  # and frequent-tracing warnings are suppressed. Created by recreate_function.
  """Wrapper class for a function that has been restored from saved state.

  See `def_function.Function`.
  """

  def __init__(self, python_function, name, function_spec, concrete_functions):
    # [FUNCTION DOC]
    # Constructs the restored function. python_function is the dispatching body, name its name,
    # function_spec supplies the jit-compile setting, function type and default values, and
    # concrete_functions is the list of concrete functions it may dispatch to. Calls the parent
    # constructor with autograph off, stores the three values, and disables the frequent tracing
    # warning. Returns nothing.
    # TODO(b/205016819): We may enable autograph once exceptions are supported.
    super(RestoredFunction, self).__init__(
        python_function,
        name,
        autograph=False,
        jit_compile=function_spec.jit_compile)
    self.concrete_functions = concrete_functions
    self._function_type = function_spec.function_type
    self._default_values = function_spec.default_values

    # Prevent RestoredFunction from spamming users with frequent tracing
    # warnings.
    self._omit_frequent_tracing_warning = True

  @property
  def _run_functions_eagerly(self):
    # [FUNCTION DOC]
    # Property override that always reports false, so the global run-functions-eagerly setting is
    # ignored and calls always go through the stored concrete function graphs (the original Python
    # function is unavailable).
    # We do not have access to the original python function, and thus, we
    # cannot meaningfully do anything but call our concrete function graphs
    # under the hood.
    #
    # Attempting to call our bespoke python function (i.e.
    # `restored_function_body`) will work so long as the user passes in all
    # required and optional arguments. If an optional argument is missing,
    # however, the call will break. For this reason, we instead skip the
    # eager call path altogether if a user has enabled eager function execution
    # via `tf.config.run_functions_eagerly`.
    return False

  def _list_all_concrete_functions(self):
    # [FUNCTION DOC]
    # Returns the concrete functions held by this restored function (no tracing is done).
    return self.concrete_functions

  def _list_all_concrete_functions_for_serialization(self):
    # [FUNCTION DOC]
    # Returns the same stored concrete functions, used when re-saving a loaded function.
    return self.concrete_functions


def recreate_function(saved_function, concrete_functions):
  # [FUNCTION DOC]
  # Rebuilds a callable polymorphic function from a saved-function proto. concrete_functions maps
  # names to restored concrete functions; the saved function spec is converted to a non-method
  # spec and applied to each referenced concrete function (mutating them). A dispatching body is
  # created that picks, at call time, the first concrete function whose signature accepts the
  # arguments (first without conversions, then allowing tensor conversion) and calls it. The body
  # is wrapped in a RestoredFunction and decorated to carry the saved argument spec. Returns the
  # decorated function. Calling it raises a value error when no concrete functions exist, when a
  # function has a missing captured input (unsupported parameter-server partitioned variables),
  # or when no signature matches (the message lists all options).
  """Creates a `Function` from a `SavedFunction`.

  Args:
    saved_function: `SavedFunction` proto.
    concrete_functions: map from function name to `ConcreteFunction`. As a side
      effect of this function, the `FunctionSpec` from `saved_function` is added
      to each `ConcreteFunction` in this map.

  Returns:
    A `Function`.
  """
  # TODO(b/205017389): Construct a `Function` with the cache populated
  # instead of creating a new `Function` backed by a Python layer to
  # glue things together. Current approach is nesting functions deeper for each
  # serialization cycle.

  # Note: handling method functions is tricky since make_decorator does not
  # allows control of "ismethod". Additionally since restored functions do
  # not behave as methods i.e. they always use the same captured tensors
  # independent of the object they are bound to, there is little value on
  # propagating that correctly.
  #
  # Ideally this conversion should happen at serialization time. But since
  # there are SavedModels which have "ismethod" populated and have an extra
  # argument that they expect to be ignored, we do it at deserialization.
  function_spec = _deserialize_function_spec_as_nonmethod(
      saved_function.function_spec)

  def restored_function_body(*args, **kwargs):
    # [FUNCTION DOC]
    # Dispatcher invoked when the restored function is called with arbitrary positional and keyword
    # arguments (already canonicalized). Tries each saved concrete function in order, first without
    # and then with argument conversion, and returns the result of the first compatible one.
    # Raises a value error if there are no concrete functions, if any function has an
    # unavailable (null) captured input, or if nothing matches (message lists every option).
    """Calls a restored function or raises an error if no matching function."""
    if not saved_function.concrete_functions:
      raise ValueError("Found zero restored functions for caller function.")
    # This is the format of function.graph.structured_input_signature. At this
    # point, the args and kwargs have already been canonicalized.
    inputs = (args, kwargs)

    # First try to find a concrete function that can be called without input
    # conversions. This allows one to pick a more specific trace in case there
    # was also a more expensive one that supported tensors.
    for allow_conversion in [False, True]:
      for function_name in saved_function.concrete_functions:
        function = concrete_functions[function_name]
        if any([inp is None for inp in function.captured_inputs]):
          raise ValueError("Looks like you are trying to run a loaded "
                           "non-Keras model that was trained using "
                           "tf.distribute.experimental.ParameterServerStrategy "
                           "with variable partitioning, which is not currently "
                           "supported. Try using Keras to define your model "
                           "if possible.")
        if _concrete_function_callable_with(function, inputs, allow_conversion):
          return _call_concrete_function(function, inputs)

    signature_descriptions = []

    def _pretty_format_positional(positional):
      # [FUNCTION DOC]
      # HELPER: builds the error message for restored_function_body.
      # Formats a sequence of positional arguments as a count line followed by one pretty-printed
      # bullet per argument; returns the string.
      return "Positional arguments ({} total):\n    * {}".format(
          len(positional),
          "\n    * ".join(pprint.pformat(a) for a in positional))

    for index, function_name in enumerate(saved_function.concrete_functions):
      concrete_function = concrete_functions[function_name]
      positional, keyword = concrete_function.structured_input_signature
      signature_descriptions.append(
          "Option {}:\n  {}\n  Keyword arguments: {}".format(
              index + 1, _pretty_format_positional(positional), keyword))
    raise ValueError(
        "Could not find matching concrete function to call loaded from the "
        f"SavedModel. Got:\n  {_pretty_format_positional(args)}\n  Keyword "
        f"arguments: {kwargs}\n\n Expected these arguments to match one of the "
        f"following {len(saved_function.concrete_functions)} option(s):\n\n"
        f"{(chr(10)+chr(10)).join(signature_descriptions)}")

  concrete_function_objects = []
  for concrete_function_name in saved_function.concrete_functions:
    concrete_function_objects.append(concrete_functions[concrete_function_name])

  for cf in concrete_function_objects:
    set_preinitialized_function_spec(cf, function_spec)

  restored_function = RestoredFunction(restored_function_body,
                                       restored_function_body.__name__,
                                       function_spec, concrete_function_objects)

  return tf_decorator.make_decorator(
      restored_function_body,
      restored_function,
      decorator_argspec=function_spec.fullargspec)


def load_function_def_library(library,
                              saved_object_graph=None,
                              load_shared_name_suffix=None,
                              wrapper_function=None):
  # [FUNCTION DOC]
  # Loads every function of a function library proto as a concrete function without captured
  # inputs. Inputs: library (function definition library), saved_object_graph (optional; supplies
  # structured input signatures and outputs by original name), load_shared_name_suffix (optional
  # string making shared names unique; generated from a unique id if omitted), wrapper_function
  # (optional callable applied to each created concrete function). Steps: re-register custom
  # gradients under new generated op types, sort functions topologically by dependency, fix each
  # function definition in place (rename functions, uniquify shared names), build a graph for it
  # (in a temporary graph when executing eagerly), restore gradient functions, attach dependencies,
  # drop the input-shapes attribute, build the concrete function, optionally wrap it, add it to the
  # graph, and register its gradient if one was saved. Functions containing TensorRT engine ops are
  # also added to the default graph (workaround). Returns a mapping from original function names to
  # concrete functions. Raises a value error on cyclic dependencies. Side effects: mutates the
  # library's function definitions, registers gradients globally, adds functions to graphs.
  """Load a set of functions as concrete functions without captured inputs.

  Functions names are manipulated during load such that they do not overlap
  with previously created ones.

  Gradients are re-registered under new names. Ops that reference the gradients
  are updated to reflect the new registered names.

  Args:
    library: FunctionDefLibrary proto message.
    saved_object_graph: SavedObjectGraph proto message. If not passed in,
      concrete function structured signatures and outputs will not be set.
    load_shared_name_suffix: If specified, used to uniquify shared names.
      Otherwise, a unique name is generated.
    wrapper_function: An object that will be wrapped on newly created functions.

  Returns:
    Map of original function names in the library to instances of
    `ConcreteFunction` without captured inputs.

  Raises:
    ValueError: if functions dependencies have a cycle.
  """
  library_function_names = set(fdef.signature.name for fdef in library.function)
  functions = {}
  renamed_functions = {}

  # Our graph building code currently requires functions to be registered with
  # some tf.Graph in order to import functions using the
  # op-name-is-function-name calling convention. To avoid leaking memory into
  # the global default graph when executing eagerly, we create a temporary
  # Graph.
  #
  # TODO(b/205023033): Make this Graph creation unnecessary when executing
  # eagerly by fixing function_def_to_graph_def.
  if ops.executing_eagerly_outside_functions():
    graph = ops.Graph()
  else:
    graph = ops.get_default_graph()

  if load_shared_name_suffix is None:
    load_shared_name_suffix = "_load_{}".format(ops.uid())

  # Custom gradient functions must be re-registered under new UIDs.
  library_gradient_names = {}  # Maps old op type to old function name
  new_gradient_op_types = {}  # Maps old gradient op type to new op type.
  gradients_to_register = {}  # Maps old function name to new op type
  for gdef in library.registered_gradients:
    if gdef.registered_op_type:
      new_op_type = custom_gradient.generate_name()
      old_op_type = compat.as_bytes(gdef.registered_op_type)

      library_gradient_names[old_op_type] = gdef.gradient_func
      new_gradient_op_types[old_op_type] = new_op_type
      gradients_to_register[gdef.gradient_func] = new_op_type

  function_deps = {}
  for fdef in library.function:
    function_deps[fdef.signature.name] = _list_function_deps(
        fdef, library_function_names, library_gradient_names)

  loaded_gradients = {}
  for fdef in _sort_function_defs(library, function_deps):
    orig_name = _fix_fdef_in_place(fdef, functions, load_shared_name_suffix,
                                   new_gradient_op_types)

    # Setup function signatures and outputs
    #
    # When concrete functions are created normally (i.e. when they're originally
    # created and not loaded via saved model), the inputs and outputs are
    # calculated based on the values passed in by the user and returned from the
    # original function, respectively. We don't have access to those anymore at
    # restore time, so we must instead pass them to the FuncGraph explicitly.
    structured_input_signature = None
    structured_outputs = None
    if (saved_object_graph is not None and
        orig_name in saved_object_graph.concrete_functions):
      # TODO(b/204324043): Offload the deserialization of the protos to the
      # first class objects by passing the actual protos. This is blocked on
      # importing `nested_structure_coder` in function.py causing a circular
      # dependency.
      proto = saved_object_graph.concrete_functions[orig_name]
      structured_input_signature = nested_structure_coder.decode_proto(
          proto.canonicalized_input_signature)
      structured_outputs = nested_structure_coder.decode_proto(
          proto.output_signature)

    # There is no need to copy all functions into the function def graph. It
    # leads to a O(n^2) increase of memory when importing functions and the
    # extra function definitions are a no-op since they already imported as a
    # function before and passed in explicitly (due to the topologic sort
    # import).
    with graph.as_default():
      func_graph = function_def_lib.function_def_to_graph(
          fdef,
          structured_input_signature=structured_input_signature,
          structured_outputs=structured_outputs)
    # Restores gradients for function-call ops (not the same as ops that use
    # custom gradients)
    _restore_gradient_functions(func_graph, renamed_functions, loaded_gradients)

    for dep in function_deps[orig_name]:
      functions[dep].add_to_graph(func_graph)

    # We do not initialize the new ConcreteFunction's function_spec and/or
    # arg_keywords here (which are used to parse the structured and flat
    # signatures, respectively). ConcreteFunction that are part of a saved
    # function is set up later by recreate_function(); and bare ConcreteFunction
    # is set up by by setup_bare_concrete_function().
    # However, we copy the FunctionDef attributes to the new ConcreteFunction,
    # excluding the "_input_shapes", which may cause an error during input shape
    # initialization at a later stage.
    if "_input_shapes" in fdef.attr:
      del fdef.attr["_input_shapes"]
    function_type = function_type_lib.from_structured_signature(
        func_graph.structured_input_signature,
        func_graph.structured_outputs,
        func_graph.function_captures.capture_types,
    )
    func = function_lib.ConcreteFunction.from_func_graph(
        func_graph, function_type, attrs=fdef.attr)
    if wrapper_function:
      func = wrapper_function(func)
    func.add_to_graph(graph)

    functions[orig_name] = func
    renamed_functions[func.name] = func
    if any(op.type == "TRTEngineOp" for op in func_graph.get_operations()):
      # TODO(b/150708051): Remove this hack once TensorRT SavedModel integration
      # is fixed. Currently it's leaking memory to maintain bug compatibility
      # with previous behavior.
      func.add_to_graph(ops.get_default_graph())

    if orig_name in gradients_to_register:
      gradient_op_type = gradients_to_register[orig_name]
      loaded_gradients[compat.as_bytes(gradient_op_type)] = func
      ops.RegisterGradient(gradient_op_type)(_gen_gradient_func(func))

  return functions


def _gen_gradient_func(func):
  # [FUNCTION DOC]
  # HELPER: gradient registration in load_function_def_library.
  # Builds a gradient function for a restored function func: the returned callable ignores the
  # op, replaces missing (null) incoming gradients with zeros shaped like the function inputs,
  # and calls func on them. Returns that callable.
  """Wraps a deserialized function."""

  def gradient_func(unused_op, *result_grads):
    # [FUNCTION DOC]
    # Gradient callable registered for a restored function. The op argument is unused; result_grads
    # are the incoming gradients, with null entries replaced by zeros (using the matching function
    # input for shape and type). Returns the outputs of calling the function on the gradients.
    # Replace all `None` arguments, because the traced custom gradient function
    # expects tensors. Replacing with zeros is correct since the `None` values
    # occur when the gradient is unconnected, and thus the gradient is
    # "statically proven to be zero." See `tf.UnconnectedGradients` for details.

    def none_to_zero(x, t):
      # [FUNCTION DOC]
      # HELPER: fills unconnected gradients. Returns x unchanged unless it is null; then returns
      # zeros shaped like tensor t (fully defined shape), or zeros with unknown dimensions set
      # to one (when the rank is known), or a scalar zero when the rank is unknown, in t's dtype.
      if x is not None:
        return x

      shape, dtype = default_gradient.shape_and_dtype(t)

      if shape.is_fully_defined():
        return default_gradient.zeros_like(t)

      dims = []
      if shape.rank is not None:
        dims = [1 if d is None else d for d in shape.as_list()]

      return array_ops.zeros(dims, dtype)

    result_grads = [
        none_to_zero(x, t) for (x, t) in zip(result_grads, func.graph.inputs)
    ]

    return func(*result_grads)

  return gradient_func


def _restore_gradient_functions(func_graph, renamed_functions,
                                loaded_gradients):
  # [FUNCTION DOC]
  # HELPER: gradient restoration step of load_function_def_library.
  # For every op in func_graph: function-call ops get the default gradient function of the renamed
  # function they call (looked up by name in renamed_functions); ops carrying a custom gradient
  # op type found in loaded_gradients make the corresponding function record the op's input count
  # and input names as its positional-argument count and keywords. Mutates ops and functions.
  # Returns nothing. Raises a key error if a called function is not in renamed_functions.
  # KNOWN ISSUE: the existing docstring below says default gradient only, but the code also
  # updates argument metadata for custom gradient functions.
  """Populate function op's _gradient_function with default gradient."""
  for op in func_graph.get_operations():
    # TODO(b/205024208): This code assumes that the gradient registered for this
    # function call is the default gradient for the function and not a custom
    # one.
    if op.type in ["StatefulPartitionedCall", "PartitionedCall"]:
      function = renamed_functions[compat.as_bytes(
          op.node_def.attr["f"].func.name)]
      op._gradient_function = function._get_gradient_function()  # pylint: disable=protected-access
    try:
      gradient_op_type = op.get_attr("_gradient_op_type")
    except ValueError:
      pass
    else:
      if gradient_op_type in loaded_gradients:
        grad_fn = loaded_gradients[gradient_op_type]
        grad_fn._num_positional_args = len(op.inputs)  # pylint: disable=protected-access
        grad_fn._arg_keywords = [inp.name for inp in op.inputs]  # pylint: disable=protected-access


def _sort_function_defs(library, function_deps):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the cycle error is raised with two separate arguments (message and detail) rather
  # than one formatted message, so the error text renders as a pair of strings.
  # HELPER: ordering step of load_function_def_library.
  # Returns the function definitions of the library as a list in topological order, dependencies
  # first, given function_deps (mapping from function name to the names it depends on).
  # Raises a value error if a cycle prevents all functions from being ordered.
  """Return a topologic sort of FunctionDefs in a library."""
  edges = collections.defaultdict(list)
  in_count = collections.defaultdict(lambda: 0)

  for fname, deps in function_deps.items():
    for dep in deps:
      edges[dep].append(fname)
      in_count[fname] += 1
  ready = [
      fdef.signature.name
      for fdef in library.function
      if in_count[fdef.signature.name] == 0
  ]
  output = []
  while ready:
    node = ready.pop()
    output.append(node)
    for dest in edges[node]:
      in_count[dest] -= 1
      if not in_count[dest]:
        ready.append(dest)

  if len(output) != len(library.function):
    failed_to_resolve = sorted(set(in_count.keys()) - set(output))
    raise ValueError("There is a cyclic dependency between functions. ",
                     f"Could not resolve {failed_to_resolve}.")

  reverse = {fdef.signature.name: fdef for fdef in library.function}
  return [reverse[x] for x in output]


def _get_gradient_op_type(node_def):
  # [FUNCTION DOC]
  # HELPER: gradient discovery for function loading.
  # Returns the custom gradient op type (bytes) stored in the node definition's gradient attribute,
  # or nothing/null if absent or if the node is a function-call op (stateful partitioned call or
  # partitioned call).
  """Returns the custom gradient op type."""
  if ("_gradient_op_type" in node_def.attr and
      node_def.op not in ["StatefulPartitionedCall", "PartitionedCall"]):
    return node_def.attr["_gradient_op_type"].s
  return None


def fix_node_def(node_def, functions, shared_name_suffix):
  # [FUNCTION DOC]
  # Rewrites a node definition in place for loading. Inputs: node_def, functions (mapping from
  # original function name to concrete function), shared_name_suffix (string). If the op itself
  # is a function it is renamed; function-valued attributes (single or list) are renamed. Hash
  # table ops lacking node-name sharing get it turned on and the suffix extended with a unique id.
  # If the op definition has a shared-name attribute, the node's shared name (existing, default or
  # node name) gets the suffix appended so loads do not share resources. Returns nothing.
  # Raises a key error if a referenced function is missing from functions.
  """Replace functions calls and shared names in `node_def`."""
  if node_def.op in functions:
    node_def.op = functions[node_def.op].name
  for _, attr_value in node_def.attr.items():
    if attr_value.WhichOneof("value") == "func":
      attr_value.func.name = functions[attr_value.func.name].name
    elif attr_value.WhichOneof("value") == "list":
      for fn in attr_value.list.func:
        fn.name = functions[fn.name].name

  # Fix old table creation bug.
  if node_def.op == "HashTableV2":
    if ("use_node_name_sharing" not in node_def.attr or
        not node_def.attr["use_node_name_sharing"].b):
      node_def.attr["use_node_name_sharing"].b = True
      # We are turning on node mame sharing, so have to make sure we don't
      # accidentally share a table resource.
      shared_name_suffix += "_{}".format(ops.uid())

  # TODO(b/124205571): Avoid accidental sharing and destruction of restored
  # resources. For now uniquify "shared_name" when loading functions to avoid
  # sharing.
  # TODO: Add regression test for b/150826922.
  op_def = op_def_registry.get(node_def.op)
  if op_def:
    attr = next((a for a in op_def.attr if a.name == "shared_name"), None)
    if attr:
      shared_name = None
      if "shared_name" in node_def.attr and node_def.attr["shared_name"].s:
        shared_name = node_def.attr["shared_name"].s
      elif attr.default_value.s:
        shared_name = compat.as_bytes(attr.default_value.s)
      if not shared_name:
        shared_name = compat.as_bytes(node_def.name)

      node_def.attr["shared_name"].s = (
          shared_name + compat.as_bytes(shared_name_suffix))


def _fix_fdef_in_place(fdef, functions, shared_name_suffix,
                       new_gradient_op_types):
  # [FUNCTION DOC]
  # HELPER: per-function preparation in load_function_def_library.
  # Mutates function definition fdef: fixes every node (see fix_node_def), replaces gradient op
  # types using new_gradient_op_types, logs a warning if some node has a custom gradient that was
  # not saved, and renames the function to a cleaned name. Returns the original function name.
  """Fixes a FunctionDef proto to be loaded in current context.

  In particular, when loading a function library into an eager context, one
  must rename the functions to avoid conflicts with existent functions.

  Args:
    fdef: FunctionDef proto to fix. It is mutated in-place.
    functions: map from function name to a ConcreteFunction instance.
    shared_name_suffix: A unique string for this load which helps to avoid
      `shared_name` collisions across loads. Two functions from the same load
      using the same `shared_name` still need to share, but functions from
      different loads with the same `shared_name` should not.
    new_gradient_op_types: map from old gradient op type to newly generated op
      type.

  Returns:
    orig_name: original value of fdef.signature.name
  """
  orig_name = fdef.signature.name
  contains_unsaved_custom_gradients = False

  for node_def in fdef.node_def:
    fix_node_def(node_def, functions, shared_name_suffix)
    op_type = _get_gradient_op_type(node_def)
    if op_type is not None:
      if op_type in new_gradient_op_types:
        node_def.attr["_gradient_op_type"].s = compat.as_bytes(
            new_gradient_op_types[op_type])
      else:
        contains_unsaved_custom_gradients = True
  if contains_unsaved_custom_gradients:
    logging.warning(
        "Importing a function (%s) with ops with unsaved custom gradients. Will"
        " likely fail if a gradient is requested.", fdef.signature.name)

  fdef.signature.name = _clean_function_name(fdef.signature.name)
  return orig_name


def _list_function_deps(fdef, library_function_names, library_gradient_names):
  # [FUNCTION DOC]
  # HELPER: dependency discovery for load_function_def_library.
  # Returns the set of function names that fdef depends on: functions called directly as ops,
  # gradient functions of nodes with custom gradient types, and otherwise function-valued
  # attributes (single or list).
  # KNOWN ISSUE: attribute-based functions are only scanned for nodes that are neither library
  # functions nor library gradients (else-branch), so such attributes on those nodes are missed.
  """Find functions referenced in `fdef`."""
  # TODO(b/205023953): Recurse into list attributes and into NameAttrList attrs
  # both when listing deps and when fixing them. `function_def_to_graph` also
  # requires fixes.
  deps = set()
  for node_def in fdef.node_def:
    grad_op_type = _get_gradient_op_type(node_def)
    if node_def.op in library_function_names:
      deps.add(node_def.op)
    elif grad_op_type and grad_op_type in library_gradient_names:
      deps.add(library_gradient_names[grad_op_type])
    else:
      for _, attr_value in node_def.attr.items():
        if attr_value.WhichOneof("value") == "func":
          deps.add(attr_value.func.name)
        elif attr_value.WhichOneof("value") == "list":
          for fn in attr_value.list.func:
            deps.add(fn.name)

  return deps


_FUNCTION_WRAPPER_NAME_REGEX = r"^%s(.*)_\d+$" % (function_lib._INFERENCE_PREFIX
                                                 )  # pylint:disable=protected-access


def _clean_function_name(name):
  # [FUNCTION DOC]
  # HELPER: cosmetic name cleanup. If name looks like an inference wrapper name (prefix, original
  # name, underscore, digits), returns the original name part; otherwise returns name unchanged.
  """Vanity function to keep the function names comprehensible."""
  # Note: each time a function is wrapped into `function_lib.ConcreteFunction`
  # its name becomes "__inference_<orig>_xyz".
  match = re.search(_FUNCTION_WRAPPER_NAME_REGEX, name)
  if match:
    return match.group(1)
  else:
    return name
