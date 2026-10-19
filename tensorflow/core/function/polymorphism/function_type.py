# [MODULE DOC]
# Defines FunctionType, the canonical description of the input/output contract of tf.function
# related functions, together with Parameter (a typed extension of the standard signature
# parameter). Provides conversion from callables and legacy signatures, proto (de)serialization,
# argument binding with defaults, subtype/supertype relations, placeholder generation, flattening
# and packing of inputs, captures and outputs, argument-name sanitization and canonicalization to
# monomorphic types. Import-time state: a sentinel for captured default values, enum mapping
# tables between proto and Python parameter kinds, and a counter limiting sanitization warnings.
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
"""Represents the types of TF functions."""

import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from absl import logging

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace


# Represents a defined parameter default value that is saved alongside the
# function's captures.
class CapturedDefaultValue:
  # [CLASS DOC]
  # HELPER: sentinel type marking that a parameter has a default value whose real value is stored
  # alongside the function's captures. A single module-level instance is used as the default of
  # optional parameters.
  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the fixed text marker for a captured default value.
    return "<captured_default_value>"

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the same fixed text marker as the representation.
    return "<captured_default_value>"

CAPTURED_DEFAULT_VALUE = CapturedDefaultValue()

PROTO_TO_PY_ENUM = {
    function_type_pb2.Parameter.Kind.POSITIONAL_ONLY:
        inspect.Parameter.POSITIONAL_ONLY,
    function_type_pb2.Parameter.Kind.POSITIONAL_OR_KEYWORD:
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    function_type_pb2.Parameter.Kind.VAR_POSITIONAL:
        inspect.Parameter.VAR_POSITIONAL,
    function_type_pb2.Parameter.Kind.KEYWORD_ONLY:
        inspect.Parameter.KEYWORD_ONLY,
    function_type_pb2.Parameter.Kind.VAR_KEYWORD:
        inspect.Parameter.VAR_KEYWORD,
}

PY_TO_PROTO_ENUM = {v: k for k, v in PROTO_TO_PY_ENUM.items()}


class Parameter(inspect.Parameter):
  # [CLASS DOC]
  # A function parameter extending the standard signature parameter with an optional trace type
  # constraint (stored as the annotation) and an optional flag (stored as a default equal to a
  # sentinel marker). Supports proto conversion, subtype checks and common-supertype computation,
  # equality, hashing and pickling by constructor arguments.
  """Represents a parameter to a function."""

  def __init__(self, name: str, kind: Any, optional: bool,
               type_constraint: Optional[trace.TraceType]):
    # [FUNCTION DOC]
    # Builds a parameter. Inputs: name (string); kind (one of the standard parameter kinds);
    # optional (boolean: whether a value may be omitted); type_constraint (a trace type or
    # nothing). Raises a value error if optional but the kind is not positional-only,
    # keyword-only or positional-or-keyword; raises a type error if a variable positional/keyword
    # parameter has a constraint, or if the constraint is neither a trace type nor nothing.
    # Optional parameters get the captured-default sentinel as default; the constraint becomes
    # the annotation.
    if optional and kind not in [
        self.POSITIONAL_ONLY, self.KEYWORD_ONLY, self.POSITIONAL_OR_KEYWORD
    ]:
      raise ValueError(
          "Parameter " + name +
          " is optional and its kind must be one of {POSITIONAL_ONLY, " +
          "KEYWORD_ONLY, POSITIONAL_OR_KEYWORD}. Got: " + str(kind))

    if type_constraint and kind in [self.VAR_POSITIONAL, self.VAR_KEYWORD]:
      raise TypeError("Variable args/kwargs can not have type constraints.")

    if not isinstance(type_constraint, (trace.TraceType, type(None))):
      raise TypeError(
          "Type constraints can only be an instance of a TraceType but got " +
          "type_constraint=" + str(type_constraint) + " for Parameter " + name)

    super().__init__(
        name,
        kind,
        default=CAPTURED_DEFAULT_VALUE if optional else self.empty,
        annotation=type_constraint
        if type_constraint is not None else self.empty)

  @classmethod
  def from_proto(cls, proto: Any) -> "Parameter":
    # [FUNCTION DOC]
    # Builds a Parameter from its serialized message. Input: proto, a parameter message with a
    # name, kind, optional flag and optional serialized type constraint. The constraint is
    # deserialized when present. Returns the new Parameter (always the base class, not a
    # subclass).
    """Generate a Parameter from the proto representation."""
    deserialized_type_constraint = serialization.deserialize(
        proto.type_constraint) if proto.HasField("type_constraint") else None
    return Parameter(proto.name, PROTO_TO_PY_ENUM[proto.kind],
                     proto.is_optional, deserialized_type_constraint)

  def to_proto(self) -> function_type_pb2.Parameter:
    # [FUNCTION DOC]
    # Serializes this parameter into a parameter message holding its name, kind, optional flag
    # and (if a constraint exists) the serialized type constraint. Takes no other input.
    """Generate a proto representation of the Parameter."""
    serialized_type_constraint = serialization.serialize(
        self.type_constraint) if self.type_constraint else None
    return function_type_pb2.Parameter(
        name=self.name,
        kind=PY_TO_PROTO_ENUM[self.kind],
        is_optional=self.optional,
        type_constraint=serialized_type_constraint)

  @property
  def optional(self) -> bool:
    # [FUNCTION DOC]
    # Property: true when the parameter has a default (i.e. it may be omitted in a call).
    """If this parameter might not be supplied for a call."""
    return self.default is not self.empty

  @property
  def type_constraint(self) -> Optional[trace.TraceType]:
    # [FUNCTION DOC]
    # Property: the trace type constraint stored as the annotation, or nothing if unspecified.
    """A supertype that the parameter's type must subtype for validity."""
    return self.annotation if self.annotation is not self.empty else None

  def is_subtype_of(self, other: "Parameter") -> bool:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns true if self is a supertype of other, but the
    # code checks that self's constraint is a subtype of the other's constraint.
    # Input: other, another Parameter. Raises a type error if either lacks a type constraint.
    # Returns false if name, kind or optional flag differ; otherwise returns whether this
    # parameter's constraint is a subtype of the other's constraint.
    """Returns True if self is a supertype of other Parameter."""
    if not self.type_constraint or not other.type_constraint:
      raise TypeError(
          "Can not determine relationship between partially specified types.")

    if ((self.name, self.kind, self.optional) !=
        (other.name, other.kind, other.optional)):
      return False

    return self.type_constraint.is_subtype_of(other.type_constraint)

  def most_specific_common_supertype(
      self, others: Sequence["Parameter"]) -> Optional["Parameter"]:
    # [FUNCTION DOC]
    # Computes a parameter whose constraint is the most specific common supertype of this and
    # the other parameters' constraints. Input: others, a sequence of Parameters. Raises a type
    # error if any parameter lacks a constraint. Returns nothing if any name, kind or optional
    # flag differs, or the constraints have no common supertype; otherwise a new Parameter with
    # the same name, kind and optional flag and the supertype constraint.
    """Returns a common supertype (if exists)."""
    if not self.type_constraint or any(
        not other.type_constraint for other in others):
      raise TypeError(
          "Can not determine relationship between partially specified types.")

    for other in others:
      if ((self.name, self.kind, self.optional) !=
          (other.name, other.kind, other.optional)):
        return None

    supertyped_constraint = self.type_constraint.most_specific_common_supertype(
        [other.type_constraint for other in others])
    if supertyped_constraint:
      return Parameter(self.name, self.kind, self.optional,
                       supertyped_constraint)
    else:
      return None

  def __eq__(self, other: Any) -> bool:
    # [FUNCTION DOC]
    # Equality: defers (not-implemented marker) for non-Parameters; otherwise compares name,
    # kind, optional flag and type constraint.
    if not isinstance(other, Parameter):
      return NotImplemented

    return ((self.name, self.kind, self.optional,
             self.type_constraint) == (other.name, other.kind, other.optional,
                                       other.type_constraint))

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash of name, kind, optional flag and type constraint (consistent with equality).
    return hash((self.name, self.kind, self.optional, self.type_constraint))

  def __repr__(self):
    # [FUNCTION DOC]
    # Text form listing the name, kind, optional flag and type constraint.
    return ("Parameter(name=" + self.name + ", kind=" + str(self.kind) +
            ", optional=" + repr(self.optional) + ", type_constraint=" +
            repr(self.type_constraint) + ")")

  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickling support: rebuilds the object by calling its class with name, kind, optional flag
    # and type constraint.
    return (self.__class__, (self.name, self.kind, self.optional,
                             self.type_constraint))


class FunctionType(core.FunctionType):
  # [CLASS DOC]
  # Canonical immutable-style description of a function's signature: an ordered mapping of
  # Parameters, an ordered mapping of capture ids to trace types, and an optional return
  # annotation (output trace type). Extends the core function type and the standard signature
  # class. Offers construction from callables, proto and legacy signatures, binding with
  # defaults, supertype/subtype logic, placeholder generation, and flatten/unpack/pack helpers.
  # Lazily caches flattened inputs, captures, outputs, sorted parameters and its text form in
  # private attributes.
  """Represents the type of a TensorFlow function.

  FunctionType is the canonical way to represent the input/output contract of
  all kinds of functions within the tf.function domain, including:
    - Polymorphic Function
    - Concrete Function
    - Atomic Function

  It provides consistent, centralized and layered logic for:
    - Canonicalization of Python input arguments
    - Type-based dispatch to monomorphic functions
    - Packing/unpacking structured python values to Tensors
    - Generation of structured placeholder values for tracing

  Additionaly, it also provides:
    - Lossless serialization
    - Native integration with Python function signature representation
    - Seamless migration from older representation formats
  """

  def __init__(self,
               parameters: Sequence[inspect.Parameter],
               captures: Optional[collections.OrderedDict] = None,
               **kwargs):
    # [FUNCTION DOC]
    # Builds a function type. Inputs: parameters, a sequence of parameters; captures, optional
    # ordered mapping of capture id to trace type (empty mapping if omitted or empty); extra
    # keyword arguments such as the return annotation are forwarded to the base signature class.
    super().__init__(parameters, **kwargs)
    self._captures = captures if captures else collections.OrderedDict()

  @property
  def parameters(self) -> Mapping[str, Any]:
    # [FUNCTION DOC]
    # Property: ordered mapping from parameter name to Parameter (delegates to the base class).
    """Returns an ordered mapping of parameter name to specification."""
    return super().parameters

  @property
  def captures(self) -> collections.OrderedDict:
    # [FUNCTION DOC]
    # Property: ordered mapping from capture id to the capture's trace type.
    """Returns an ordered mapping of capture id to type."""
    return self._captures

  @property
  def output(self) -> Optional[trace.TraceType]:
    # [FUNCTION DOC]
    # Property: the return annotation (output trace type) or nothing if not specified.
    """Return the output TraceType if specified."""
    return (
        self.return_annotation
        if self.return_annotation is not self.empty
        else None
    )

  @classmethod
  def from_callable(cls,
                    obj: Callable[..., Any],
                    *,
                    follow_wrapped: bool = True) -> "FunctionType":
    # [FUNCTION DOC]
    # Builds a function type from a Python callable. Inputs: obj, the callable; follow_wrapped
    # (keyword-only, default true), whether to unwrap decorated callables. Each parameter of the
    # callable's signature becomes a Parameter with the same name and kind, marked optional if it
    # has a default, and with no type constraint. Returns a FunctionType without captures or
    # output. Does not keep the actual default values.
    """Generate FunctionType from a python Callable."""
    signature = super().from_callable(obj, follow_wrapped=follow_wrapped)
    # TODO(fmuham): Support TraceType-based annotations.
    parameters = [
        Parameter(p.name, p.kind, p.default is not p.empty, None)
        for p in signature.parameters.values()
    ]

    return FunctionType(parameters)

  @classmethod
  def get_default_values(cls,
                         obj: Callable[..., Any],
                         *,
                         follow_wrapped: bool = True) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Collects default values of a callable. Inputs: obj, the callable; follow_wrapped
    # (keyword-only, default true). Returns a mapping from parameter name to default value for
    # every parameter that has a default.
    """Inspects and returns a dictionary of default values."""
    signature = super().from_callable(obj, follow_wrapped=follow_wrapped)
    default_values = {}
    for p in signature.parameters.values():
      if p.default is not p.empty:
        default_values[p.name] = p.default
    return default_values

  @classmethod
  def from_proto(cls, proto: Any) -> "FunctionType":
    # [FUNCTION DOC]
    # Builds a FunctionType from its serialized message. Input: proto with a list of parameter
    # messages and a list of capture messages (name and serialized type). Returns a FunctionType
    # with the deserialized parameters and an ordered mapping of capture names to types. The
    # output type is not restored.
    """Generate a FunctionType from the proto representation."""
    return FunctionType([Parameter.from_proto(p) for p in proto.parameters],
                        collections.OrderedDict([
                            (c.name,
                             serialization.deserialize(c.type_constraint))
                            for c in proto.captures
                        ]))

  def to_proto(self) -> Any:
    # [FUNCTION DOC]
    # Serializes to a function-type message holding each parameter's message and each capture's
    # name and serialized type. The output type is not serialized.
    """Generate a proto representation from the FunctionType."""
    return function_type_pb2.FunctionType(
        parameters=[p.to_proto() for p in self.parameters.values()],
        captures=[
            function_type_pb2.Capture(
                name=n, type_constraint=serialization.serialize(t))
            for n, t in self.captures.items()
        ])

  def bind_with_defaults(self, args, kwargs, default_values):
    # [FUNCTION DOC]
    # Binds call arguments to this signature, fills in defaults and casts values to constraints.
    # Inputs: args (positional values), kwargs (mapping of keyword values), default_values
    # (mapping from parameter name to real default value). Raises the standard binding error on
    # mismatched arguments. Parameters holding the captured-default sentinel are replaced by the
    # real defaults. Each value whose parameter has a type constraint is cast with that
    # constraint (allowing specs). Returns the bound arguments object.
    """Returns BoundArguments with default values filled in."""
    bound_arguments = self.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    with_default_args = collections.OrderedDict()
    for name, value in bound_arguments.arguments.items():
      if value is CAPTURED_DEFAULT_VALUE:
        with_default_args[name] = default_values[name]
      else:
        with_default_args[name] = value

    for arg_name in with_default_args:
      constraint = self.parameters[arg_name].type_constraint
      if constraint:
        with_default_args[arg_name] = constraint.cast(
            with_default_args[arg_name],
            trace_type.InternalCastContext(allow_specs=True),
        )
    bound_arguments = inspect.BoundArguments(self, with_default_args)
    return bound_arguments

  def is_supertype_of(self, other: "FunctionType") -> bool:
    # [FUNCTION DOC]
    # Checks whether this type is a supertype of another function type. Returns false if the
    # parameter counts differ, if any parameter of this type is not a subtype of the matching
    # parameter of the other (in order), or if the other lacks any capture name of this type.
    # Finally requires each capture type of this to be a subtype of the other's capture type.
    # Parameters without constraints raise a type error via the parameter check.
    """Returns True if self is a supertype of other FunctionType."""
    if len(self.parameters) != len(other.parameters):
      return False

    for self_param, other_param in zip(self.parameters.values(),
                                       other.parameters.values()):
      # Functions are contravariant on their parameter types.
      if not self_param.is_subtype_of(other_param):
        return False

    # Other must have all capture names of self.
    if not all(name in other.captures for name in self.captures):
      return False

    # Functions are contravariant upon the capture types.
    return all(capture_type.is_subtype_of(other.captures[name])
               for name, capture_type in self.captures.items())

  def most_specific_common_subtype(
      self, others: Sequence["FunctionType"]) -> Optional["FunctionType"]:
    # [FUNCTION DOC]
    # Computes a common subtype of this and other function types. Input: others, a sequence of
    # FunctionTypes (assumed to have at least as many parameters as this one). Each parameter is
    # replaced by the common supertype of the corresponding parameters (functions are
    # contravariant); captures are the union of all capture names, each typed as the common
    # supertype of the types from the function types that have it. Returns nothing if any
    # parameter or capture has no common type; otherwise a new FunctionType (without output).
    """Returns a common subtype (if exists)."""
    subtyped_parameters = []

    for i, parameter in enumerate(self.parameters.values()):
      # Functions are contravariant on their parameter types.
      subtyped_parameter = parameter.most_specific_common_supertype(
          [list(other.parameters.values())[i] for other in others])
      if subtyped_parameter is None:
        return None
      subtyped_parameters.append(subtyped_parameter)

    if not all(subtyped_parameters):
      return None

    # Common subtype has superset of all captures.
    capture_names = set(self.captures.keys())
    for other in others:
      capture_names = capture_names.union(other.captures.keys())

    subtyped_captures = collections.OrderedDict()
    for name in capture_names:
      containing = [t for t in [self, *others] if name in t.captures]
      # Pick the first type that has the capture as the base.
      base = containing[0]
      relevant_others = containing[1:]

      # Functions are contravariant upon the capture types.
      common_type = base.captures[name].most_specific_common_supertype(
          [other.captures[name] for other in relevant_others]
      )
      if common_type is None:
        return None
      else:
        subtyped_captures[name] = common_type

    return FunctionType(subtyped_parameters, subtyped_captures)

  def placeholder_arguments(
      self, placeholder_context: trace.PlaceholderContext
  ) -> inspect.BoundArguments:
    # [FUNCTION DOC]
    # Generates placeholder values for tracing. Input: placeholder_context, which tracks naming
    # scope (updated with each parameter name as a side effect). Raises a value error for
    # variable-length parameters or parameters without a type constraint. Returns bound
    # arguments mapping each parameter name to the placeholder from its constraint.
    """Returns BoundArguments of values that can be used for tracing."""
    arguments = collections.OrderedDict()
    for parameter in self.parameters.values():
      if parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
        raise ValueError("Can not generate placeholder values for "
                         "variable length function type.")

      if not parameter.type_constraint:
        raise ValueError("Can not generate placeholder value for "
                         "partially defined function type.")
      placeholder_context.update_naming_scope(parameter.name)
      arguments[parameter.name] = parameter.type_constraint.placeholder_value(
          placeholder_context)

    return inspect.BoundArguments(self, arguments)

  @property
  def _sorted_parameters(self) -> List[trace.TraceType]:
    # [FUNCTION DOC]
    # HELPER: ordering utility for flat_inputs and unpack_inputs.
    # Property returning (and caching) the parameters with keyword-only ones moved to the end and
    # sorted by name; others keep their order. The annotation says trace types but the elements
    # are Parameters.
    if not hasattr(self, "_cached_sorted_parameters"):
      # Sort keyword-only parameters by name.
      sorted_parameters = []
      kwonly_parameters = []
      for p in self.parameters.values():
        if p.kind is Parameter.KEYWORD_ONLY:
          kwonly_parameters.append(p)
        else:
          sorted_parameters.append(p)
      sorted_parameters = sorted_parameters + sorted(
          kwonly_parameters, key=lambda p: p.name
      )
      self._cached_sorted_parameters = sorted_parameters
    return self._cached_sorted_parameters

  @property
  def flat_inputs(self) -> List[trace.TraceType]:
    # [FUNCTION DOC]
    # Property: cached flat list of tensor trace types of all parameters, in the sorted parameter
    # order, obtained by flattening each parameter's constraint. Requires every parameter to
    # have a constraint.
    """Flat tensor inputs accepted by this FunctionType."""
    if not hasattr(self, "_cached_flat_inputs"):
      cached_flat_inputs = []
      for p in self._sorted_parameters:
        cached_flat_inputs.extend(p.type_constraint.flatten())
      self._cached_flat_inputs = cached_flat_inputs
    return self._cached_flat_inputs

  def unpack_inputs(
      self, bound_parameters: inspect.BoundArguments
  ) -> List[core.Tensor]:
    # [FUNCTION DOC]
    # Flattens bound call arguments into tensors. Input: bound_parameters, bound arguments with a
    # value for every parameter. Values are converted to tensors by each parameter's constraint
    # in sorted parameter order, then tensors whose input type reports an alias id already seen
    # are dropped (de-aliasing). Returns the list of kept tensors.
    """Unpacks python arguments to flat tensor inputs accepted by this type."""
    flat = []
    for p in self._sorted_parameters:
      flat.extend(
          p.type_constraint.to_tensors(bound_parameters.arguments[p.name])
      )

    dealiased_inputs = []
    ids_used = set()
    for tensor, input_type in zip(flat, self.flat_inputs):
      alias_id = input_type._alias_id()  # pylint: disable=protected-access
      if alias_id is None or alias_id not in ids_used:
        dealiased_inputs.append(tensor)

      if alias_id is not None:
        ids_used.add(alias_id)

    return dealiased_inputs

  @property
  def flat_captures(self) -> List[trace.TraceType]:
    # [FUNCTION DOC]
    # Property: cached flat list of tensor trace types of all captures, in capture order.
    """Flat tensor captures needed by this FunctionType."""
    if not hasattr(self, "_cached_flat_captures"):
      cached_flat_captures = []
      for t in self.captures.values():
        cached_flat_captures.extend(t.flatten())
      self._cached_flat_captures = cached_flat_captures

    return self._cached_flat_captures

  def unpack_captures(self, captures) -> List[core.Tensor]:
    # [FUNCTION DOC]
    # Flattens capture values into tensors. Input: captures, a sequence of values matching the
    # capture types in order. Each value is converted by its capture type. Raises a type error if
    # the resulting tensor count differs from the expected flat capture count. Returns the list.
    """Unpacks captures to flat tensors."""
    flat = []
    for v, t in zip(captures, self.captures.values()):
      flat.extend(t.to_tensors(v))
    if len(flat) != len(self.flat_captures):
      raise TypeError(
          f"Flattening captures {captures} with type {self!r} produced"
          f" {len(flat)} tensors instead of {len(self.flat_captures)}"
      )
    return flat

  @property
  def flat_outputs(self) -> List[trace.TraceType]:
    # [FUNCTION DOC]
    # KNOWN ISSUE: when no output type is set, the cache attribute is never assigned, so the
    # final return raises an attribute error instead of returning an empty list or nothing.
    # Property: cached flat list of tensor trace types of the output type.
    """Flat tensor outputs returned by this FunctionType."""
    if not hasattr(self, "_cached_flat_outputs"):
      if self.output is not None:
        self._cached_flat_outputs = self.output.flatten()

    return self._cached_flat_outputs

  def pack_output(self, flat_values: Sequence[core.Tensor]) -> Any:
    # [FUNCTION DOC]
    # Rebuilds a structured output value from flat tensors. Input: flat_values, a sequence of
    # tensors (nothing is treated as empty). Raises a value error if no output type is defined.
    # Returns the value reconstructed by the output type.
    """Packs flat tensors to generate a value of the output type."""
    if flat_values is None:
      flat_values = []

    if self.output is None:
      raise ValueError("Can not pack outputs for undefined output type.")
    else:
      return self.output.from_tensors(iter(flat_values))

  def __eq__(self, other: Any) -> bool:
    # [FUNCTION DOC]
    # Equality: defers (not-implemented marker) for non-FunctionTypes; otherwise compares the
    # parameter mappings and capture mappings. The output type is ignored.
    if not isinstance(other, FunctionType):
      return NotImplemented

    return (self.parameters, self.captures) == (other.parameters,
                                                other.captures)

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Hash of the parameter items and capture items (consistent with equality).
    return hash((tuple(self.parameters.items()), tuple(self.captures.items())))

  def __repr__(self):
    # [FUNCTION DOC]
    # Multi-line text form listing input parameters (name, kind, constraint), the output type and
    # the captures (or "None"). The result is cached on the instance after first computation.
    if hasattr(self, "_cached_repr"):
      return self._cached_repr

    lines = ["Input Parameters:"]
    for parameter in self.parameters.values():
      lines.append(
          f"  {parameter.name} ({parameter.kind}): {parameter.type_constraint}"
      )

    lines.append("Output Type:")
    lines.append(f"  {self.output}")

    lines.append("Captures:")
    if self.captures:
      for capture_id, capture_type in self.captures.items():
        lines.append(f"  {capture_id}: {capture_type}")
    else:
      lines.append("  None")

    self._cached_repr = "\n".join(lines)
    return self._cached_repr


MAX_SANITIZATION_WARNINGS = 5
sanitization_warnings_given = 0


# TODO(fmuham): In future, replace warning with exception.
# TODO(fmuham): Sanitize to graph node conventions.
def sanitize_arg_name(name: str) -> str:
  # [FUNCTION DOC]
  # Makes a legal parameter name. Input: name, a non-empty string. Non-alphanumeric characters
  # become underscores, and a prefix is added if the first character is not a letter. Returns
  # the sanitized name. Side effect: logs a warning (and increments a module-level counter)
  # when the name changed, up to a fixed maximum number of warnings. An empty name would
  # cause an index error.
  """Sanitizes function argument names.

  Matches Python symbol naming rules.

  Without sanitization, names that are not legal Python parameter names can be
  set which makes it challenging to represent callables supporting the named
  calling capability.

  Args:
    name: The name to sanitize.

  Returns:
    A string that meets Python parameter conventions.
  """
  # Replace non-alphanumeric chars with '_'
  swapped = "".join([c if c.isalnum() else "_" for c in name])
  result = swapped if swapped[0].isalpha() else "arg_" + swapped

  global sanitization_warnings_given
  if name != result and sanitization_warnings_given < MAX_SANITIZATION_WARNINGS:
    logging.warning(
        "`%s` is not a valid tf.function parameter name. Sanitizing to `%s`.",
        name, result)
    sanitization_warnings_given += 1

  return result


# TODO(fmuham): Consider forcing kind to be always POSITIONAL_OR_KEYWORD.
def _make_validated_mono_param(
    name, value, kind, type_context, poly_type
) -> Parameter:
  # [FUNCTION DOC]
  # HELPER: builds parameters for canonicalize_to_monomorphic.
  # Inputs: name, value (an actual argument), kind, type_context (tracing context) and
  # poly_type (the polymorphic constraint or nothing). Traces the value into a type; raises a
  # type error if the type is not a subtype of the polymorphic constraint. Returns a
  # non-optional Parameter with the traced type.
  """Generates and validates a parameter for Monomorphic FunctionType."""
  mono_type = trace_type.from_value(value, type_context)

  if poly_type and not mono_type.is_subtype_of(poly_type):
    raise TypeError(f"Parameter `{name}` was expected to be of type "
                    f"{poly_type} but is {mono_type}")

  return Parameter(name, kind, False, mono_type)


def canonicalize_to_monomorphic(
    args: Tuple[Any, ...], kwargs: Dict[Any, Any], default_values: Dict[Any,
                                                                        Any],
    capture_types: collections.OrderedDict, polymorphic_type: FunctionType
) -> Tuple[FunctionType, trace_type.InternalTracingContext]:
  # [FUNCTION DOC]
  # Specializes a polymorphic function type to concrete arguments. Inputs: args and kwargs (the
  # call arguments), default_values (mapping from name to real default; when non-empty, defaults
  # are applied and sentinels replaced), capture_types (ordered mapping of captures) and
  # polymorphic_type (the type to bind against). Binding errors propagate. Variable positional
  # parameters become individually named positional-only parameters; when one exists, all
  # positional-or-keyword parameters become positional-only; variable keyword parameters become
  # keyword-only parameters in sorted key order. Every value is traced and validated against
  # the polymorphic constraint (type error on mismatch). Returns the monomorphic FunctionType
  # and the tracing context used.
  """Generates a monomorphic type out of polymorphic type for given args."""
  poly_bound_arguments = polymorphic_type.bind(*args, **kwargs)

  # Inject Default Values.
  if default_values:
    poly_bound_arguments.apply_defaults()
    default_values_injected = poly_bound_arguments.arguments
    for name, value in default_values_injected.items():
      if value is CAPTURED_DEFAULT_VALUE:
        default_values_injected[name] = default_values[name]
    poly_bound_arguments = inspect.BoundArguments(
        poly_bound_arguments.signature, default_values_injected
    )

  parameters = []
  type_context = trace_type.InternalTracingContext()
  has_var_positional = any(p.kind is Parameter.VAR_POSITIONAL
                           for p in polymorphic_type.parameters.values())

  for name, arg in poly_bound_arguments.arguments.items():
    poly_parameter = polymorphic_type.parameters[name]
    if (has_var_positional and
        poly_parameter.kind is Parameter.POSITIONAL_OR_KEYWORD):
      # If there is a VAR_POSITIONAL, all POSITIONAL_OR_KEYWORD become
      # POSITIONAL_ONLY.
      parameters.append(
          _make_validated_mono_param(name, arg, Parameter.POSITIONAL_ONLY,
                                     type_context,
                                     poly_parameter.type_constraint))

    elif poly_parameter.kind is Parameter.VAR_POSITIONAL:
      # Unbundle VAR_POSITIONAL into individual POSITIONAL_ONLY args.
      for i, value in enumerate(arg):
        parameters.append(
            _make_validated_mono_param(f"{poly_parameter.name}_{i}", value,
                                       Parameter.POSITIONAL_ONLY, type_context,
                                       poly_parameter.type_constraint))

    elif poly_parameter.kind is Parameter.VAR_KEYWORD:
      # Unbundle VAR_KEYWORD into individual KEYWORD_ONLY args.
      for kwarg_name in sorted(arg.keys()):
        parameters.append(
            _make_validated_mono_param(kwarg_name, arg[kwarg_name],
                                       Parameter.KEYWORD_ONLY, type_context,
                                       poly_parameter.type_constraint))
    else:
      parameters.append(
          _make_validated_mono_param(name, arg, poly_parameter.kind,
                                     type_context,
                                     poly_parameter.type_constraint))

  return FunctionType(parameters, capture_types), type_context


# TODO(fmuham): Share code with canonicalize_to_monomorphic.
# TODO(fmuham): Lift unnecessary restrictions on input_signature validity.
def add_type_constraints(function_type: FunctionType, input_signature: Any,
                         default_values: Dict[str, Any]) -> FunctionType:
  # [FUNCTION DOC]
  # Adds type constraints from an explicit input signature. Inputs: function_type (base type),
  # input_signature (sequence of specs, one per argument), default_values (mapping of name to
  # default). Specs are traced with legacy-signature mode. The "self" parameter stays
  # unconstrained; variable-keyword parameters are dropped; a variable-positional parameter is
  # expanded into positional-only parameters consuming the remaining specs; other parameters
  # consume specs in order, falling back to the type of their default. Raises a type error
  # for a keyword-only parameter without default, a missing constraint, or leftover specs.
  # Returns a new FunctionType (captures and output are not carried over).
  """Adds type constraints to a FunctionType based on the input_signature."""
  context = trace_type.InternalTracingContext(is_legacy_signature=True)
  constraints = [trace_type.from_value(c, context) for c in input_signature]
  parameters = []

  has_var_pos = any(
      p.kind is p.VAR_POSITIONAL for p in function_type.parameters.values())

  for param in function_type.parameters.values():
    # VAR_POSITIONAL does not allow POSITIONAL_OR_KEYWORD args.
    sanitized_kind = (
        param.POSITIONAL_ONLY if has_var_pos and
        param.kind is param.POSITIONAL_OR_KEYWORD else param.kind)

    if param.name == "self":
      # Type constraints do not apply on them.
      parameters.append(Parameter("self", sanitized_kind, param.optional, None))

    elif param.kind is param.VAR_KEYWORD:
      # Disabled when input_signature is specified.
      continue

    elif param.kind is param.VAR_POSITIONAL:
      # Convert into Positional Only args based on length of constraints.
      for i in range(len(constraints)):
        parameters.append(
            Parameter(param.name + "_" + str(i), Parameter.POSITIONAL_ONLY,
                      False, constraints.pop(0)))

    elif (param.kind in [
        param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY
    ]):
      if param.kind is param.KEYWORD_ONLY and param.name not in default_values:
        raise TypeError(
            "Since input_signature is defined, keyword-only parameter"
            f" `{param.name}` must have a default value"
        )

      if constraints:
        parameters.append(
            Parameter(param.name, sanitized_kind, param.optional,
                      constraints.pop(0)))
      elif param.name in default_values:
        type_constraint = trace_type.from_value(default_values[param.name])
        parameters.append(
            Parameter(param.name, sanitized_kind, param.optional,
                      type_constraint))
      else:
        raise TypeError(
            f"input_signature missing type constraint for {param.name}")

  if constraints:
    raise TypeError(
        f"input_signature contains {len(constraints)} extra type constraints.")

  return FunctionType(parameters)


def from_structured_signature(
    input_signature=None,
    output_signature=None,
    capture_types=None,
    are_keyword_args_also_positional=False,
) -> FunctionType:
  # [FUNCTION DOC]
  # Builds a FunctionType from a legacy structured signature. Inputs: input_signature (pair of
  # positional specs and keyword specs mapping; default is empty), output_signature (spec of the
  # outputs), capture_types (optional mapping of captures; empty if omitted), and
  # are_keyword_args_also_positional (boolean, default false). Positional specs become
  # positional-only parameters named with an arg prefix and index; keyword specs become
  # sanitized-name parameters of kind keyword-only, or positional-or-keyword if the flag is set.
  # Returns a FunctionType with the traced output type as return annotation.
  """Generates a FunctionType from legacy signature representation."""
  if input_signature is None:
    input_signature = ((), {})

  args, kwargs = input_signature
  parameters = []

  for i, arg in enumerate(args):
    parameters.append(
        Parameter(
            "arg_" + str(i),
            Parameter.POSITIONAL_ONLY,
            False,
            trace_type.from_value(
                arg, trace_type.InternalTracingContext(is_legacy_signature=True)
            ),
        )
    )

  keyword_arg_kind = (
      Parameter.POSITIONAL_OR_KEYWORD
      if are_keyword_args_also_positional
      else Parameter.KEYWORD_ONLY
  )

  for name, kwarg in kwargs.items():
    parameters.append(
        Parameter(
            sanitize_arg_name(name),
            keyword_arg_kind,
            False,
            trace_type.from_value(
                kwarg,
                trace_type.InternalTracingContext(is_legacy_signature=True),
            ),
        )
    )

  return_type = trace_type.from_value(
      output_signature,
      trace_type.InternalTracingContext(is_legacy_signature=True),
  )

  return FunctionType(
      parameters, capture_types or {}, return_annotation=return_type
  )


def to_structured_signature(function_type: FunctionType) -> Tuple[Any, Any]:
  # [FUNCTION DOC]
  # Converts a fully specified FunctionType into a legacy structured signature. Input:
  # function_type. Each constraint is turned into a spec through unnest-only placeholders.
  # Positional-only parameters go into the positional tuple, all others into a mapping by name.
  # Raises a type error if any constraint or the output type is missing. Returns a pair of
  # (positional tuple, keyword mapping) and the output spec.
  """Returns structured input and output signatures from a FunctionType."""
  def to_signature(x_type):
    # [FUNCTION DOC]
    # HELPER: converts one trace type to a spec for to_structured_signature. Raises a type error
    # if the type is missing; otherwise returns its unnest-only placeholder value.
    if x_type is None:
      raise TypeError(
          "Can not generate structured signature if FunctionType is not fully"
          f" specified. Received {function_type}"
      )
    return x_type.placeholder_value(
        trace_type.InternalPlaceholderContext(unnest_only=True)
    )

  args_signature = []
  kwargs_signature = {}
  for p in function_type.parameters.values():
    if p.kind == Parameter.POSITIONAL_ONLY:
      args_signature.append(to_signature(p.type_constraint))
    else:
      kwargs_signature[p.name] = to_signature(p.type_constraint)

  input_signature = (tuple(args_signature), kwargs_signature)
  output_signature = to_signature(function_type.output)

  return input_signature, output_signature
