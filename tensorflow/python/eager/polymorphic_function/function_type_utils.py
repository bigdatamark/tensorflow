# [MODULE DOC]
# Utilities that bridge tf.function with the FunctionType machinery. Provides conversion
# between FunctionType plus default values and the legacy full argument spec, input signature
# and argument name lists; the deprecated FunctionSpec class describing how to bind arguments to
# a Python function; helpers that build and canonicalize function types from callables and
# call arguments; input signature validation (only possibly nested TensorSpec objects are
# allowed); derivation of a function type from a graph; and a legacy structure comparison.
# Part of the polymorphic_function package. No import-time side effects.
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
"""Utilities for using FunctionType with tf.function."""

import functools
import inspect
from typing import Any, Dict, Tuple

import six

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest


def to_fullargspec(
    function_type: function_type_lib.FunctionType,
    default_values: Dict[str, Any],
) -> inspect.FullArgSpec:
  # [FUNCTION DOC]
  # Converts a FunctionType into a legacy full argument spec. Positional-only and
  # positional-or-keyword parameters go to the argument list (their defaults, looked up by name
  # in default_values, form the defaults tuple); keyword-only parameters go to the keyword-only
  # list with a defaults mapping; variadic positional and variadic keyword parameter names become
  # varargs and varkw. Inputs: function_type, the signature to convert; default_values, mapping
  # from parameter names to default values. Output: the full argument spec, with defaults and
  # keyword-only defaults set to nothing when empty and annotations always empty. Raises a key
  # error if a parameter marked as having a default is missing from default_values. No side effects.
  """Generates backwards compatible FullArgSpec from FunctionType."""
  args = []
  varargs = None
  varkw = None
  defaults = []
  kwonlyargs = []
  kwonlydefaults = {}

  for parameter in function_type.parameters.values():
    if parameter.kind in [
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ]:
      args.append(parameter.name)
      if parameter.default is not inspect.Parameter.empty:
        defaults.append(default_values[parameter.name])
    elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
      kwonlyargs.append(parameter.name)
      if parameter.default is not inspect.Parameter.empty:
        kwonlydefaults[parameter.name] = default_values[parameter.name]
    elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
      varargs = parameter.name
    elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
      varkw = parameter.name

  return inspect.FullArgSpec(
      args,
      varargs,
      varkw,
      tuple(defaults) if defaults else None,
      kwonlyargs,
      kwonlydefaults if kwonlydefaults else None,
      annotations={},
  )


def _to_default_values(fullargspec):
  # [FUNCTION DOC]
  # HELPER: used by to_function_type to extract defaults from a legacy full argument spec.
  # Builds a mapping from argument name to default value by pairing the trailing positional
  # arguments with the defaults tuple, then adds keyword-only defaults. All names are passed
  # through argument-name sanitization. Input: a full argument spec. Output: the mapping (empty
  # when there are no defaults). No side effects.
  """Returns default values from the function's inspected fullargspec."""
  if fullargspec.defaults is not None:
    defaults = {
        name: value
        for name, value in zip(
            fullargspec.args[-len(fullargspec.defaults) :], fullargspec.defaults
        )
    }
  else:
    defaults = {}

  if fullargspec.kwonlydefaults is not None:
    defaults.update(fullargspec.kwonlydefaults)

  defaults = {
      function_type_lib.sanitize_arg_name(name): value
      for name, value in defaults.items()
  }

  return defaults


def to_function_type(fullargspec):
  # [FUNCTION DOC]
  # Converts a legacy full argument spec into a FunctionType and its default values.
  # Each positional argument becomes a positional-or-keyword parameter (sanitized name, optional
  # if it has a default), followed by the variadic positional parameter, keyword-only parameters
  # and the variadic keyword parameter. No type constraints are attached (all are nothing).
  # Input: a full argument spec. Output: a pair of the FunctionType and the mapping from
  # sanitized names to default values. No side effects.
  # KNOWN ISSUE: the variadic positional and variadic keyword names are not sanitized, unlike
  # all other names, so they may differ from names used in the default-values mapping.
  """Generates FunctionType and default values from fullargspec."""
  default_values = _to_default_values(fullargspec)
  parameters = []

  for arg in fullargspec.args:
    arg_name = function_type_lib.sanitize_arg_name(arg)
    parameters.append(
        function_type_lib.Parameter(
            arg_name,
            function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
            arg_name in default_values,
            None,
        )
    )

  if fullargspec.varargs is not None:
    parameters.append(
        function_type_lib.Parameter(
            fullargspec.varargs,
            function_type_lib.Parameter.VAR_POSITIONAL,
            False,
            None,
        )
    )

  for kwarg in fullargspec.kwonlyargs:
    parameters.append(
        function_type_lib.Parameter(
            function_type_lib.sanitize_arg_name(kwarg),
            function_type_lib.Parameter.KEYWORD_ONLY,
            kwarg in default_values,
            None,
        )
    )

  if fullargspec.varkw is not None:
    parameters.append(
        function_type_lib.Parameter(
            fullargspec.varkw,
            function_type_lib.Parameter.VAR_KEYWORD,
            False,
            None,
        )
    )

  return function_type_lib.FunctionType(parameters), default_values


def to_input_signature(function_type):
  # [FUNCTION DOC]
  # Extracts a legacy input signature (tuple of tensor specs) from a FunctionType. A leading
  # parameter named self is skipped. For every remaining parameter with a type constraint, the
  # constraint is converted to its legacy placeholder form; if any component is not a tensor
  # spec, the scan stops and the signature is treated as auto-generated. Input: function_type.
  # Output: an empty tuple when there are no parameters to constrain, or when constraints were
  # auto-generated and none were collected before stopping; otherwise the tuple of collected
  # constraints; or nothing when no parameter had a constraint. Parameters without a constraint
  # are skipped, so the result can be shorter than the parameter list. No side effects.
  """Extracts an input_signature from function_type instance."""
  constrained_parameters = list(function_type.parameters.keys())

  # self does not have a constraint in input_signature
  if "self" in constrained_parameters:
    constrained_parameters.pop(0)

  # There are no parameters to constrain.
  if not constrained_parameters:
    return tuple()

  constraints = []
  is_auto_constrained = False

  for parameter_name in constrained_parameters:
    parameter = function_type.parameters[parameter_name]
    constraint = None
    if parameter.type_constraint:
      # Generate legacy constraint representation.
      constraint = parameter.type_constraint.placeholder_value(
          trace_type.InternalPlaceholderContext(unnest_only=True)
      )
      if any(
          not isinstance(arg, tensor.TensorSpec)
          for arg in nest.flatten([constraint], expand_composites=True)
      ):
        # input_signature only supports contiguous TensorSpec composites
        is_auto_constrained = True
        break
      else:
        constraints.append(constraint)

  # All constraints were generated by FunctionType
  if is_auto_constrained and not constraints:
    return tuple()

  # If the list is empty then there was no input_signature specified.
  return tuple(constraints) if constraints else None


def to_arg_names(function_type):
  # [FUNCTION DOC]
  # Returns the list of names of the positional-only and positional-or-keyword parameters of a
  # FunctionType, in declaration order. Variadic and keyword-only parameters are excluded.
  # No side effects.
  """Generates a list of arg names from a FunctionType."""
  arg_names = []
  for p in function_type.parameters.values():
    if p.kind in {
        function_type_lib.Parameter.POSITIONAL_ONLY,
        function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
    }:
      arg_names.append(p.name)
  return arg_names


# TODO(b/214462107): Minimize API surface for FunctionSpec.
class FunctionSpec(object):
  # [CLASS DOC]
  # Deprecated wrapper describing how arguments bind to a Python function, kept for backwards
  # compatibility with code that predates FunctionType. Holds the function type, the mapping of
  # default values, a derived legacy full argument spec and input signature, a purity flag
  # (convert all inputs to tensors and forbid variable changes), a jit-compile setting and a
  # function name (default "f"). Created through the two class-level factory methods or the
  # constructor, and read through read-only properties. No class-level side effects.
  """Specification of how to bind arguments to a function.

  Deprecated. Please use FunctionType instead.
  """

  @classmethod
  def from_function_and_signature(
      cls, python_function, input_signature, is_pure=False, jit_compile=None
  ):
    # [FUNCTION DOC]
    # Factory that builds a FunctionSpec by inspecting a Python callable. Inputs: python_function,
    # the callable (possibly wrapped in partial objects); input_signature, a tuple or list of
    # tensor specs or nothing for a variable signature; is_pure, optional flag defaulting to
    # false; jit_compile, optional compilation setting. Derives the function type and defaults
    # via make_function_type, strips partial wrappers to read the function name (falling back to
    # "f"), and returns a new FunctionSpec. Raises a type error if the input signature is
    # invalid. No side effects.
    # KNOWN ISSUE: the code constructs FunctionSpec directly rather than the class this method is
    # called on, so subclasses get a base-class instance.
    """Creates a FunctionSpec instance given a python function and signature.

    Args:
      python_function: a function to inspect
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      jit_compile: see `tf.function`

    Returns:
      instance of FunctionSpec
    """
    function_type, default_values = make_function_type(
        python_function, input_signature)
    # Get the function's name.  Remove functools.partial wrappers if necessary.
    while isinstance(python_function, functools.partial):
      python_function = python_function.func
    name = getattr(python_function, "__name__", "f")

    return FunctionSpec(
        function_type,
        default_values,
        is_pure=is_pure,
        jit_compile=jit_compile,
        name=name,
    )

  @classmethod
  def from_fullargspec_and_signature(
      cls,
      fullargspec,
      input_signature,
      is_pure=False,
      name=None,
      jit_compile=None,
  ):
    # [FUNCTION DOC]
    # Factory that builds a FunctionSpec from a legacy full argument spec. Inputs: fullargspec,
    # the legacy spec; input_signature, a sequence of tensor specs or nothing/empty; is_pure,
    # optional flag defaulting to false; name, optional function name; jit_compile, optional
    # setting. Converts the spec to a function type and defaults; if a non-empty input signature
    # is given it is turned into a tuple, validated, and added as type constraints. Returns a
    # new FunctionSpec. Raises a type error if the signature is invalid. No other side effects.
    # KNOWN ISSUE: constructs FunctionSpec directly rather than the class this method is called
    # on, so subclasses get a base-class instance.
    """Construct FunctionSpec from legacy FullArgSpec format."""
    function_type, default_values = to_function_type(fullargspec)
    if input_signature:
      input_signature = tuple(input_signature)
      _validate_signature(input_signature)
      function_type = function_type_lib.add_type_constraints(
          function_type, input_signature, default_values
      )

    return FunctionSpec(
        function_type, default_values, is_pure, name, jit_compile
    )

  def __init__(
      self,
      function_type,
      default_values,
      is_pure=False,
      name=None,
      jit_compile=None,
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: function_type, the FunctionType of the Python function;
    # default_values, mapping from parameter names to defaults; is_pure, optional flag
    # (default false); name, optional function name (falls back to "f" when empty or nothing);
    # jit_compile, optional compilation setting. Stores them, and derives and stores the legacy
    # full argument spec and the input signature from the function type. Returns nothing.
    """Constructs a FunctionSpec describing a python function.

    Args:
      function_type: A FunctionType describing the python function signature.
      default_values: Dictionary mapping parameter names to default values.
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      name: Name of the function
      jit_compile: see `tf.function`.
    """
    self._function_type = function_type
    self._default_values = default_values
    self._fullargspec = to_fullargspec(function_type, default_values)
    self._is_pure = is_pure
    self._jit_compile = jit_compile

    # TODO(edloper): Include name when serializing for SavedModel?
    self._name = name or "f"
    self._input_signature = to_input_signature(function_type)

  @property
  def default_values(self):
    # [FUNCTION DOC]
    # Property returning the stored mapping from parameter names to default values (the same
    # object, not a copy).
    """Returns dict mapping parameter names to default values."""
    return self._default_values

  @property
  def function_type(self):
    # [FUNCTION DOC]
    # Property returning the FunctionType describing the Python function signature.
    """Returns a FunctionType representing the Python function signature."""
    return self._function_type

  @property
  def fullargspec(self):
    # [FUNCTION DOC]
    # Property returning the legacy full argument spec computed at construction time.
    return self._fullargspec

  # TODO(fmuham): Replace usages with FunctionType and remove.
  @property
  def input_signature(self):
    # [FUNCTION DOC]
    # Property returning the legacy input signature derived from the function type at
    # construction time: a tuple of tensor specs, an empty tuple, or nothing if unconstrained.
    return self._input_signature

  # TODO(fmuham): Replace usages with FunctionType and remove.
  @property
  def flat_input_signature(self):
    # [FUNCTION DOC]
    # Property returning the input signature flattened into a single tuple, expanding composite
    # type specs into their components.
    # KNOWN ISSUE: if the input signature is nothing (unconstrained), flattening yields a tuple
    # holding that single null value rather than an empty tuple.
    return tuple(nest.flatten(self.input_signature, expand_composites=True))

  @property
  def is_pure(self):
    # [FUNCTION DOC]
    # Property returning whether all inputs are converted to tensors and variable changes are
    # disallowed.
    return self._is_pure

  @property
  def jit_compile(self):
    # [FUNCTION DOC]
    # Property returning the stored jit-compile setting (may be nothing if unspecified).
    return self._jit_compile

  # TODO(fmuham): Replace usages and remove.
  @property
  def arg_names(self):
    # [FUNCTION DOC]
    # Property returning the list of positional (positional-only and positional-or-keyword)
    # parameter names of the function type, in order.
    return to_arg_names(self.function_type)

  def signature_summary(self, default_values=False):
    # [FUNCTION DOC]
    # Returns a text summary of the signature: the printable representation of the function
    # type. Input: default_values, optional flag (default false); when true a "Defaults:"
    # section is appended listing each parameter name with its default value representation,
    # or "None" when there are no defaults. Output: the summary string. No side effects.
    """Returns a string summarizing this function's signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
    summary = f"{self._function_type!r}"
    if default_values:
      summary += "\nDefaults:"
      if self.default_values:
        for name, value in self.default_values.items():
          summary += f"\n  {name}: {value!r}"
      else:
        summary += "\n  None"
    return summary


def make_function_type(python_function, input_signature):
  # [FUNCTION DOC]
  # Builds the FunctionType and default values of a Python callable. Validates the input
  # signature first (raising a type error if invalid), then inspects the callable for its type
  # and defaults; if an input signature is given it is converted to a tuple and added as type
  # constraints. Inputs: python_function, the callable; input_signature, sequence of tensor
  # specs or nothing. Output: pair of FunctionType and mapping of default values. No side effects.
  """Generates a FunctionType for python_function."""
  _validate_signature(input_signature)

  function_type = function_type_lib.FunctionType.from_callable(
      python_function
  )
  default_values = function_type_lib.FunctionType.get_default_values(
      python_function
  )

  if input_signature is not None:
    input_signature = tuple(input_signature)
    function_type = function_type_lib.add_type_constraints(
        function_type, input_signature, default_values
    )

  return function_type, default_values


def make_canonicalized_monomorphic_type(
    args: Any,
    kwargs: Any,
    capture_types: Any,
    polymorphic_type,
) -> Tuple[function_type_lib.FunctionType, trace_type.InternalTracingContext]:
  # [FUNCTION DOC]
  # Generates a monomorphic function type from concrete call arguments. Keyword argument names
  # are sanitized, then the arguments are canonicalized against the polymorphic type together
  # with the capture types (no extra bound parameters are passed). Inputs: args, positional call
  # arguments; kwargs, mapping of keyword arguments; capture_types, types of captured values;
  # polymorphic_type, the function type to canonicalize against. Output: pair of the monomorphic
  # FunctionType and the tracing context used. No side effects.
  """Generates function type given the function arguments."""
  kwargs = {
      function_type_lib.sanitize_arg_name(name): value
      for name, value in kwargs.items()
  }

  function_type, type_context = (
      function_type_lib.canonicalize_to_monomorphic(
          args, kwargs, {}, capture_types, polymorphic_type
      )
  )

  return function_type, type_context


def canonicalize_function_inputs(
    args, kwargs, function_type, default_values=None, is_pure=False
):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring describes a returned four-element tuple and a value error, but
  # the code returns only the bound arguments object from binding, and binding failures are
  # raised as type errors (or a value error on sanitized name collisions).
  # Canonicalizes call arguments against a FunctionType. Inputs: args, positional arguments;
  # kwargs, mapping of keyword arguments; function_type, the signature to bind to;
  # default_values, optional mapping of defaults (treated as empty when nothing); is_pure,
  # optional flag; when true, every argument is first converted to a tensor (tensors and tensor
  # specs are kept). Output: the bound arguments with defaults filled in. Raises a value error on
  # name collisions after sanitization and a type error if binding fails. Converting values to
  # tensors may create tensors in the current context.
  """Canonicalizes `args` and `kwargs`.

  Canonicalize the inputs to the Python function using FunctionType.
  In particular, we parse the varargs and kwargs that the
  original function was called with into a tuple corresponding to the
  Python function's positional (named) arguments and a dictionary
  corresponding to its kwargs.  Missing default arguments are added.

  If the FunctionType has an type constraints, then they are used to convert
  arguments to tensors; otherwise, any inputs containing numpy arrays are
  converted to tensors.


  Args:
    args: The varargs this object was called with.
    kwargs: The keyword args this function was called with.
    function_type: FunctionType to canonicalize against.
    default_values: Default values to use.
    is_pure: Force variable inputs to Tensors.

  Returns:
    A canonicalized ordering of the inputs, as well as full and filtered
    (Tensors and Variables only) versions of their concatenated flattened
    representations, represented by a tuple in the form (args, kwargs,
    flat_args, filtered_flat_args). Here: `args` is a full list of bound
    arguments, and `kwargs` contains only true keyword arguments, as opposed
    to named arguments called in a keyword-like fashion.

  Raises:
    ValueError: If a keyword in `kwargs` cannot be matched with a positional
      argument when an input signature is specified, or when the inputs
      do not conform to the input signature.
  """
  default_values = {} if not default_values else default_values
  if is_pure:
    args, kwargs = _convert_variables_to_tensors(args, kwargs)
  bound_arguments = bind_function_inputs(
      args, kwargs, function_type, default_values
  )
  return bound_arguments


def bind_function_inputs(args, kwargs, function_type, default_values):
  # [FUNCTION DOC]
  # HELPER: used by canonicalize_function_inputs to do the actual binding.
  # Sanitizes keyword argument names and binds positional and keyword arguments to the function
  # type, filling in defaults. Inputs: args, kwargs, function_type, default_values (mapping).
  # Output: the bound arguments. Raises a value error if sanitizing the names makes two keyword
  # arguments collide, and a type error (chained from the original error) if binding fails for
  # any reason. No side effects.
  """Bind `args` and `kwargs` into a canonicalized signature args, kwargs."""
  sanitized_kwargs = {
      function_type_lib.sanitize_arg_name(k): v for k, v in kwargs.items()
  }
  if len(kwargs) != len(sanitized_kwargs):
    raise ValueError(
        "Name collision after sanitization. Please rename "
        "tf.function input parameters. Original: "
        f"{sorted(kwargs.keys())}, Sanitized: "
        f"{sorted(sanitized_kwargs.keys())}"
    )

  try:
    bound_arguments = function_type.bind_with_defaults(
        args, sanitized_kwargs, default_values
    )
  except Exception as e:
    raise TypeError(
        f"Binding inputs to tf.function failed due to `{e}`. "
        f"Received args: {args} and kwargs: {sanitized_kwargs} for signature:"
        f" {function_type}."
    ) from e
  return bound_arguments


def _validate_signature(signature):
  # [FUNCTION DOC]
  # HELPER: validates user-provided input signatures before building function types.
  # Does nothing for a null signature. Otherwise raises a type error if the signature is not a
  # tuple or list, if it contains variable specs, or if any flattened element (composites
  # expanded) is not a tensor spec; the message lists the offending arguments. Returns nothing.
  """Checks the input_signature to be valid."""
  if signature is None:
    return

  if not isinstance(signature, (tuple, list)):
    raise TypeError(
        "input_signature must be either a tuple or a list, got "
        f"{type(signature)}."
    )

  # TODO(xjun): Allow VariableSpec once we figure out API for de-aliasing.
  variable_specs = _get_variable_specs(signature)
  if variable_specs:
    raise TypeError(
        f"input_signature doesn't support VariableSpec, got {variable_specs}"
    )

  if any(
      not isinstance(arg, tensor.TensorSpec)
      for arg in nest.flatten(signature, expand_composites=True)
  ):
    bad_args = [
        arg
        for arg in nest.flatten(signature, expand_composites=True)
        if not isinstance(arg, tensor.TensorSpec)
    ]
    raise TypeError(
        "input_signature must be a possibly nested sequence of "
        f"TensorSpec objects, got invalid args {bad_args} with "
        f"types {list(six.moves.map(type, bad_args))}."
    )


def _to_tensor_or_tensor_spec(x):
  # [FUNCTION DOC]
  # HELPER: element converter for _convert_variables_to_tensors.
  # Returns the value unchanged if it is already a tensor or tensor spec; otherwise converts it
  # to a tensor in the current context (which may raise if conversion is impossible).
  return (
      x
      if isinstance(x, (tensor.Tensor, tensor.TensorSpec))
      else ops.convert_to_tensor(x)
  )


def _convert_variables_to_tensors(args, kwargs):
  # [FUNCTION DOC]
  # HELPER: used by canonicalize_function_inputs for pure functions.
  # Converts every positional argument and every keyword argument value to a tensor (leaving
  # tensors and tensor specs alone). Inputs: args, sequence; kwargs, mapping. Output: pair of a
  # tuple of converted positional arguments and a mapping of converted keyword arguments.
  # Variables are converted by the general tensor conversion.
  args = [_to_tensor_or_tensor_spec(x) for x in args]
  kwargs = {kw: _to_tensor_or_tensor_spec(x) for kw, x in kwargs.items()}
  return tuple(args), kwargs


def _get_variable_specs(args):
  # [FUNCTION DOC]
  # HELPER: used by _validate_signature to reject variable specs.
  # Flattens the argument structure and collects every variable spec found, recursing into the
  # component specs of composite (non-tensor) type specs. Non-spec elements are ignored.
  # Input: a nested structure. Output: list of variable specs (empty if none).
  """Returns `VariableSpecs` from `args`."""
  variable_specs = []
  for arg in nest.flatten(args):
    if not isinstance(arg, type_spec.TypeSpec):
      continue
    if isinstance(arg, resource_variable_ops.VariableSpec):
      variable_specs.append(arg)
    elif not isinstance(arg, tensor.TensorSpec):
      # arg is a CompositeTensor spec.
      variable_specs.extend(_get_variable_specs(arg._component_specs))  # pylint: disable=protected-access
  return variable_specs


def derive_from_graph(func_graph):
  # [FUNCTION DOC]
  # Derives a FunctionType from a function graph. The input side is a tuple of trace types of
  # the graph's input tensors plus an empty keyword mapping; the output side is a tuple of trace
  # types of the graph's outputs. Both are combined with the graph's capture types into a
  # function type built from a structured signature. Input: func_graph. Output: the
  # FunctionType. Structure information of the original inputs and outputs is not preserved.
  """Derives a FunctionType from FuncGraph."""
  # TODO(fmuham): Include structure info from structured_inputs
  input_signature = (
      tuple(trace_type.from_value(i) for i in func_graph.inputs),
      {},
  )

  # TODO(fmuham): Include output structure info from structured_outputs
  output_signature = tuple(trace_type.from_value(o) for o in func_graph.outputs)

  return function_type_lib.from_structured_signature(
      input_signature,
      output_signature,
      func_graph.function_captures.capture_types,
  )


# TODO(fmuham): Replace usages with TraceType and remove.
def is_same_structure(structure1, structure2, check_values=False):
  # [FUNCTION DOC]
  # Compares two nested structures (composites expanded). Returns false if the structures
  # differ (the underlying comparison raised a value or type error). When check_values is true
  # (default false), it additionally requires the flattened elements to have identical types
  # pairwise and to be equal as lists. Otherwise returns true. No side effects.
  # KNOWN ISSUE: with check_values, equality of flattened lists may be ambiguous or raise for
  # elements such as arrays whose comparison is not a single boolean.
  """Check two structures for equality, optionally of types and of values."""
  try:
    nest.assert_same_structure(structure1, structure2, expand_composites=True)
  except (ValueError, TypeError):
    return False
  if check_values:
    flattened1 = nest.flatten(structure1, expand_composites=True)
    flattened2 = nest.flatten(structure2, expand_composites=True)
    # First check the types to avoid AttributeErrors.
    if any(type(f1) is not type(f2) for f1, f2 in zip(flattened1, flattened2)):
      return False
    return flattened1 == flattened2
  return True
