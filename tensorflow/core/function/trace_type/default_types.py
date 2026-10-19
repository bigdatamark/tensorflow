# [MODULE DOC]
# Default TraceType implementations for common Python values, used by the function tracing cache.
# A TraceType describes the "type" of a function argument for the purpose of deciding whether a
# previously traced function can be reused. This module provides TraceTypes for literals
# (bool, int, float, string, None, ranges, etc.), weak references to arbitrary objects, tuples,
# lists, named tuples, attrs-decorated classes and dictionaries. Each supports subtype checks,
# most-specific-common-supertype computation, placeholder value generation, casting of concrete
# values, flattening to tensors and back, and (for most) protobuf serialization.
# Exposes a one-time registration hook for the Tensor TraceType (to avoid a cyclic dependency), a
# NaN helper, and the classes Literal, Weakref, Tuple, List, NamedTuple, Attrs and Dict.
# Import-time side effects: the serializable classes Literal, Tuple, List, NamedTuple, Attrs and
# Dict are registered with the serialization registry at the bottom of the module. A module-level
# TENSOR variable starts empty and is set once by register_tensor_type.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""TraceType implementations for common Python types."""

import collections
import math
import numbers
from typing import Any, Dict as PythonDict, Hashable, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type
import weakref

from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace

# Register the TraceType of Tensor (aka TensorSpec) to avoid cyclic dependency.
TENSOR = None


def register_tensor_type(tensor_type):
  # [FUNCTION DOC]
  # HELPER: Lets the tensor spec module plug its TraceType in without a cyclic import.
  # Stores the given Tensor TraceType class in the module-level TENSOR variable. Takes the class to
  # register. Returns nothing. Raises an assertion error if a tensor type was already registered
  # (registration is allowed only once). Side effect: mutates module-global state.
  global TENSOR
  if not TENSOR:
    TENSOR = tensor_type
  else:
    raise AssertionError("Tensor type is already registered.")

NanMarker = object()


def is_nan(x):
  # [FUNCTION DOC]
  # HELPER: Used by Literal to treat NaN values as equal to each other.
  # Takes any value. Returns true only if it is a number and is not-a-number (for complex numbers,
  # if either the real or imaginary part is NaN); false for non-numbers and ordinary numbers.
  # No side effects, raises nothing for non-numeric input.
  """Checks if given value is a Python NaN."""
  if not isinstance(x, numbers.Number):
    return False

  if isinstance(x, complex):
    return math.isnan(x.real) or math.isnan(x.imag)
  else:
    return math.isnan(x)


class Literal(trace.TraceType, serialization.Serializable):
  # [CLASS DOC]
  # TraceType for a concrete constant Python value (bool, int, float, string, None, range, ...).
  # Two Literals are the same type only if their values compare equal; there is no proper
  # subtype relation beyond equality. NaN values are all normalized to one marker so that NaN
  # matches NaN. Holds the value and its precomputed hash. Serializable to protobuf for
  # bool, int, float, string and None values only. Registered with the serialization registry.
  """Represents a Literal type like bool, int or string."""

  def __init__(self, value: Any):
    # [FUNCTION DOC]
    # Creates a Literal from any hashable value. NaN values are replaced by an internal marker so
    # they compare equal. Stores the value and caches its hash. Raises a type error if the value
    # is unhashable.
    # We match nan values against each other even though Python doesn't.
    if is_nan(value):
      value = NanMarker

    self.value = value
    self._value_hash = hash(value)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    # [FUNCTION DOC]
    # Returns true only when the other TraceType equals this Literal (same value).
    return self == other

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["Literal"]:
    # [FUNCTION DOC]
    # Given a sequence of TraceTypes, returns this Literal if all of them equal it, otherwise
    # returns nothing (no common supertype exists).
    return self if all(self == other for other in types) else None

  @classmethod
  def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedLiteral]:
    # [FUNCTION DOC]
    # Class method returning the protobuf message class used to serialize a Literal.
    return default_types_pb2.SerializedLiteral

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedLiteral) -> "Literal":
    # [FUNCTION DOC]
    # Class method that rebuilds a Literal from a serialized literal message. Checks which of the
    # bool, int, float, string or none fields is set and wraps that value. Raises a value error
    # if none of the fields is set (malformed message).
    if proto.HasField("bool_value"):
      return Literal(proto.bool_value)

    if proto.HasField("int_value"):
      return Literal(proto.int_value)

    if proto.HasField("float_value"):
      return Literal(proto.float_value)

    if proto.HasField("str_value"):
      return Literal(proto.str_value)

    if proto.HasField("none_value"):
      return Literal(None)

    raise ValueError("Malformed Literal proto can not be deserialized")

  def experimental_as_proto(self) -> default_types_pb2.SerializedLiteral:
    # [FUNCTION DOC]
    # Serializes this Literal into a literal message, choosing the field by the value's type
    # (bool is checked before int; none uses an empty none marker). Raises a value error for any
    # other value type (including the NaN marker, which is not serializable).
    if isinstance(self.value, bool):
      return default_types_pb2.SerializedLiteral(bool_value=self.value)

    if isinstance(self.value, int):
      return default_types_pb2.SerializedLiteral(int_value=self.value)

    if isinstance(self.value, float):
      return default_types_pb2.SerializedLiteral(float_value=self.value)

    if isinstance(self.value, str):
      return default_types_pb2.SerializedLiteral(str_value=self.value)

    if self.value is None:
      return default_types_pb2.SerializedLiteral(
          none_value=default_types_pb2.SerializedLiteral.NoneValue())

    raise ValueError("Can not serialize Literal of type " +
                     type(self.value).__name__)

  def placeholder_value(self, placeholder_context) -> Any:
    # [FUNCTION DOC]
    # Returns the concrete value to use as a placeholder during tracing: the stored value itself,
    # except that a range is converted to a list and the NaN marker becomes a real NaN float. The
    # placeholder context argument is unused.
    # TODO(b/263505796): Remove this check when a range's placeholder output
    # is expected to be a range and not a list.
    if isinstance(self.value, range):
      return list(self.value)

    if self.value is NanMarker:
      return float("nan")

    return self.value

  def cast(self, value: Any, casting_context: Any) -> Any:
    # [FUNCTION DOC]
    # Checks that a concrete value matches this Literal and returns it unchanged. NaN matches a
    # NaN Literal; otherwise the value must equal the stored value. Raises a value error when it
    # does not match. The casting context is unused.
    if self.value is NanMarker and is_nan(value):
      return value

    if value == self.value:
      return value
    else:
      raise ValueError(f"Can not cast {value!r} to {self!r}")

  def __eq__(self, other) -> bool:
    # [FUNCTION DOC]
    # Equality: signals not-implemented for non-TraceType objects; otherwise true only if the
    # other is a Literal with an equal value.
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    return isinstance(other, Literal) and self.value == other.value

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Returns the hash precomputed at construction from the (possibly NaN-normalized) value.
    return self._value_hash

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns a readable string of the form class name followed by the value's representation.
    return f"{self.__class__.__name__}[{self.value!r}]"


class Weakref(trace.TraceType):
  # [CLASS DOC]
  # TraceType for an arbitrary Python object that is held only through a weak reference, so the
  # function cache does not keep the object alive. Equality requires both referents to be alive
  # and identical or equal references. Holds the weak reference and its precomputed hash. Not
  # serializable.
  """Represents weakref of an arbitrary Python object.

  When a function argument is a custom class, instead of making a copy of it
  just for the sake of function cache, a weakref is instead kept to save memory.
  """

  def __init__(self, ref: weakref.ReferenceType):
    # [FUNCTION DOC]
    # Takes a weak reference object, stores it and caches its hash (which raises a type error if
    # the reference cannot be hashed).
    self._ref = ref
    self._ref_hash = hash(ref)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    # [FUNCTION DOC]
    # Returns true only when the other TraceType equals this one (same live referent).
    return self == other

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["Weakref"]:
    # [FUNCTION DOC]
    # Returns this Weakref if every given TraceType equals it, otherwise nothing.
    return self if all(self == other for other in types) else None

  def placeholder_value(self, placeholder_context) -> Any:
    # [FUNCTION DOC]
    # Returns the referenced object (or nothing if it has been garbage collected). The context
    # argument is unused.
    return self._ref()

  def cast(self, value, _):
    # [FUNCTION DOC]
    # Checks that a value is the referenced object (identical or equal) and returns it. Also
    # accepts values that wrap the referenced object through repeatedly unwrapping the wrapped
    # attribute, returning the unwrapped object in that case. Raises a value error otherwise. The
    # second argument (casting context) is unused.
    if value is self._ref() or value == self._ref():
      return value

    # We unwrap objects when generating the TraceType so we allow matching now.
    while hasattr(value, "__wrapped__"):
      value = value.__wrapped__
      if value is self._ref():
        return value

    raise ValueError(f"Can not cast {value!r} to {self!r}")

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality: not-implemented for non-TraceType objects; false for other TraceType classes or
    # when either referent is dead; true if both refer to the identical object or the weak
    # references compare equal.
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Weakref):
      return False

    if self._ref() is None or other._ref() is None:
      return False

    if self._ref() is other._ref():
      return True

    return self._ref == other._ref

  def __hash__(self):
    # [FUNCTION DOC]
    # Returns the hash of the weak reference computed at construction.
    return self._ref_hash

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the returned text has an unbalanced extra closing parenthesis at the end.
    # Returns a string with the class name and the weak reference's representation.
    return f"{self.__class__.__name__}[{self._ref!r}])"


class Tuple(trace.TraceType, serialization.Serializable):
  # [CLASS DOC]
  # TraceType for a Python tuple: an ordered sequence of component TraceTypes. Subtype and
  # supertype relations are computed component-wise and require equal length. Holds the
  # components as a tuple. Serializable to protobuf and registered with the serialization
  # registry. Also the building block used by List, NamedTuple and Attrs.
  """Represents a tuple of TraceType objects."""

  def __init__(self, *components: trace.TraceType):
    # [FUNCTION DOC]
    # Takes any number of component TraceTypes (positional arguments) and stores them as the
    # components tuple.
    self.components = components

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    # [FUNCTION DOC]
    # Returns true if the other is a Tuple of the same length and every component of this tuple is
    # a subtype of the corresponding component of the other; false otherwise.
    if (not isinstance(other, Tuple) or
        len(self.components) != len(other.components)):
      return False

    return all(
        self_component.is_subtype_of(other_component) for self_component,
        other_component in zip(self.components, other.components))

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["Tuple"]:
    # [FUNCTION DOC]
    # Computes the component-wise most specific common supertype of this tuple and the given
    # sequence of TraceTypes. Returns nothing if any of them is not a Tuple of equal length or if
    # any component position has no common supertype; otherwise returns a new Tuple of the
    # per-position supertypes.
    """See base class."""
    if not all(
        isinstance(other, Tuple) and
        len(self.components) == len(other.components) for other in others):
      return None

    supertyped_components = []
    for i, component in enumerate(self.components):
      supertyped_component = component.most_specific_common_supertype(
          [other.components[i] for other in others])
      if supertyped_component is None:
        return None
      supertyped_components.append(supertyped_component)

    return Tuple(*supertyped_components)

  @classmethod
  def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedTuple]:
    # [FUNCTION DOC]
    # Class method returning the protobuf message class used to serialize a Tuple.
    return default_types_pb2.SerializedTuple

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedTuple) -> "Tuple":
    # [FUNCTION DOC]
    # Class method rebuilding a Tuple from its serialized message by deserializing each component
    # through the serialization registry.
    return Tuple(*[serialization.deserialize(c) for c in proto.components])

  def experimental_as_proto(self) -> default_types_pb2.SerializedTuple:
    # [FUNCTION DOC]
    # Serializes this Tuple into a message holding each serialized component.
    return default_types_pb2.SerializedTuple(
        components=[serialization.serialize(c) for c in self.components])

  def placeholder_value(self, placeholder_context) -> Any:
    # [FUNCTION DOC]
    # Builds a placeholder tuple by asking each component for its placeholder value with the
    # given placeholder context, and returns them as a Python tuple.
    components = [
        component.placeholder_value(placeholder_context)
        for component in self.components
    ]
    return tuple(components)

  def to_tensors(self, value) -> Any:
    # [FUNCTION DOC]
    # Flattens a concrete tuple value into a list of tensors by concatenating each component's
    # tensors, pairing values and component types in order. Asserts the value is a tuple.
    assert isinstance(value, tuple)
    flattened_values = []
    for comp_value, comp_type in zip(value, self.components):
      flattened_values.extend(comp_type.to_tensors(comp_value))
    return flattened_values

  def from_tensors(self, tensors) -> Any:
    # [FUNCTION DOC]
    # Rebuilds a tuple value by letting each component consume what it needs from the given
    # tensor iterator, in order. Returns a Python tuple.
    return tuple(c.from_tensors(tensors) for c in self.components)

  def flatten(self) -> PythonList[trace.TraceType]:
    # [FUNCTION DOC]
    # Returns a flat list of the leaf TraceTypes of all components, in order.
    flattened_types = []
    for component in self.components:
      flattened_types.extend(component.flatten())
    return flattened_types

  def cast(self, value: Any, casting_context) -> Any:
    # [FUNCTION DOC]
    # Casts a concrete tuple to this type by casting each component. Asserts the value is a tuple
    # of the same length. Returns a new tuple if any component changed, otherwise the original
    # value.
    assert isinstance(value, tuple), f"Can not cast {value!r} to tuple type."
    assert len(value) == len(
        self.components
    ), f"Expected {value} to have length of {len(self.components)}"

    casted_values, was_casted = util.cast_and_return_whether_casted(
        self.components, value, casting_context
    )
    if was_casted:
      return tuple(casted_values)
    else:
      return value

  def __eq__(self, other: Any) -> bool:
    # [FUNCTION DOC]
    # Equality: not-implemented for non-TraceType objects; false for non-Tuple TraceTypes;
    # otherwise true if the component tuples are equal.
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Tuple):
      return False

    return self.components == other.components

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Returns the hash of the components tuple.
    return hash(self.components)

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns a readable string listing the representations of all components.
    return f"Tuple[{', '.join(map(repr, self.components))}]"


class List(trace.TraceType, serialization.Serializable):
  # [CLASS DOC]
  # TraceType for a Python list. Implemented by wrapping a Tuple of the element TraceTypes
  # (stored as components_tuple) and delegating most behavior to it, while distinguishing itself
  # from tuples by type. Serializable and registered with the serialization registry.
  """Represents a list of TraceType objects."""

  def __init__(self, *components: trace.TraceType):
    # [FUNCTION DOC]
    # Takes any number of element TraceTypes and stores them wrapped in a Tuple TraceType.
    self.components_tuple = Tuple(*components)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    # [FUNCTION DOC]
    # Returns true if the other is a List and this list's components are component-wise subtypes
    # of the other's; false otherwise.
    if not isinstance(other, List):
      return False

    return self.components_tuple.is_subtype_of(other.components_tuple)

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["Tuple"]:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the return annotation says a Tuple, but a List (or nothing) is actually
    # returned.
    # Returns nothing if any given TraceType is not a List or if the component-wise supertype
    # does not exist; otherwise returns a new List built from the supertype components.
    """See base class."""
    if not all(isinstance(other, List) for other in others):
      return None

    supertyped_components_tuple = (
        self.components_tuple.most_specific_common_supertype(
            [other.components_tuple for other in others]
        )
    )

    if supertyped_components_tuple is None:
      return None

    return List(*supertyped_components_tuple.components)

  @classmethod
  def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedList]:
    # [FUNCTION DOC]
    # Class method returning the protobuf message class used to serialize a List.
    return default_types_pb2.SerializedList

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedList) -> "List":
    # [FUNCTION DOC]
    # Class method rebuilding a List from its message by deserializing the wrapped tuple and
    # using its components.
    return List(
        *Tuple.experimental_from_proto(proto.components_tuple).components)

  def experimental_as_proto(self) -> default_types_pb2.SerializedList:
    # [FUNCTION DOC]
    # Serializes this List as a message holding the serialized wrapped Tuple.
    return default_types_pb2.SerializedList(
        components_tuple=self.components_tuple.experimental_as_proto())

  def placeholder_value(self, placeholder_context) -> Any:
    # [FUNCTION DOC]
    # Returns a Python list of the component placeholder values, built via the wrapped Tuple.
    return list(self.components_tuple.placeholder_value(placeholder_context))

  def to_tensors(self, value):
    # [FUNCTION DOC]
    # Flattens a concrete list into a list of tensors by converting it to a tuple and delegating
    # to the wrapped Tuple. Asserts the value is a list.
    assert isinstance(value, list)
    return self.components_tuple.to_tensors(tuple(value))

  def from_tensors(self, tensors) -> Any:
    # [FUNCTION DOC]
    # Rebuilds a list value from the tensor iterator via the wrapped Tuple and returns it as a
    # Python list.
    return list(self.components_tuple.from_tensors(tensors))

  def flatten(self) -> PythonList[trace.TraceType]:
    # [FUNCTION DOC]
    # Returns the flat list of leaf TraceTypes of the elements, delegated to the wrapped Tuple.
    return self.components_tuple.flatten()

  def cast(self, value: Any, casting_context) -> Any:
    # [FUNCTION DOC]
    # Casts a concrete list by casting each element. Asserts the value is a list of the same
    # length. Returns a new list if any element changed, else the original value.
    assert isinstance(value, list), f"Can not cast {value!r} to list type."
    assert len(value) == len(
        self.components_tuple.components
    ), f"Expected {value} to have length of {len(self.components_tuple)}"

    casted_values, was_casted = util.cast_and_return_whether_casted(
        self.components_tuple.components, value, casting_context
    )
    if was_casted:
      return list(casted_values)
    else:
      return value

  def __eq__(self, other: Any) -> bool:
    # [FUNCTION DOC]
    # Equality: not-implemented for non-TraceType objects; false for non-List TraceTypes;
    # otherwise compares the wrapped tuples.
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, List):
      return False

    return self.components_tuple == other.components_tuple

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Returns the hash of the wrapped tuple.
    return hash(self.components_tuple)

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns a readable string listing the representations of all element types.
    return f"List[{', '.join(map(repr, self.components_tuple.components))}]"


class NamedTuple(trace.TraceType, serialization.Serializable):
  # [CLASS DOC]
  # TraceType for a Python named tuple. Records the type name, the ordered attribute names, a
  # Tuple of attribute TraceTypes, and optionally the concrete named tuple class used to build
  # placeholders and repacked values (this class is lost on serialization). Two NamedTuples are
  # related only if type name and attribute names match. Serializable and registered.
  """Represents a NamedTuple of TraceType objects."""

  def __init__(self,
               type_name: str,
               attribute_names: PythonTuple[str],
               attributes: PythonTuple[trace.TraceType],
               placeholder_type: Optional[Type[Any]] = None):
    # [FUNCTION DOC]
    # Takes the type name (string), a tuple of attribute names, a tuple of attribute TraceTypes
    # and an optional concrete class (default nothing) used to create placeholders. Stores them;
    # the attribute types are wrapped in a Tuple TraceType.
    self.type_name = type_name
    self.attribute_names = attribute_names
    self.attributes = Tuple(*attributes)
    self._placeholder_type = placeholder_type

  @classmethod
  def from_type_and_attributes(
      cls, named_tuple_type: Any,
      attributes: PythonTuple[trace.TraceType]) -> "NamedTuple":
    # [FUNCTION DOC]
    # Class method factory: builds a NamedTuple TraceType from a concrete named tuple class and a
    # tuple of attribute TraceTypes, taking the type name and field names from the class and
    # remembering the class as the placeholder type.
    return NamedTuple(named_tuple_type.__name__, named_tuple_type._fields,
                      attributes, named_tuple_type)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    # [FUNCTION DOC]
    # Returns true if the other is a NamedTuple with the same type name and attribute names and
    # this one's attributes are component-wise subtypes of the other's; false otherwise.
    if not isinstance(other, NamedTuple):
      return False

    return (self.type_name == other.type_name and
            self.attribute_names == other.attribute_names and
            self.attributes.is_subtype_of(other.attributes))

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["NamedTuple"]:
    # [FUNCTION DOC]
    # Returns nothing if any given TraceType is not a NamedTuple with the same type name and
    # attribute names, or if the attributes have no common supertype. Otherwise returns a new
    # NamedTuple with the supertype attributes, keeping this one's placeholder type.
    """See base class."""
    if not all(
        isinstance(other, NamedTuple) and self.type_name == other.type_name and
        self.attribute_names == other.attribute_names for other in others):
      return None

    supertyped_attributes = self.attributes.most_specific_common_supertype(
        [other.attributes for other in others])

    if supertyped_attributes is None:
      return None

    return NamedTuple(self.type_name, self.attribute_names,
                      supertyped_attributes.components, self._placeholder_type)

  @classmethod
  def experimental_type_proto(
      cls) -> Type[default_types_pb2.SerializedNamedTuple]:
    # [FUNCTION DOC]
    # Class method returning the protobuf message class used to serialize a NamedTuple.
    return default_types_pb2.SerializedNamedTuple

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedNamedTuple) -> "NamedTuple":
    # [FUNCTION DOC]
    # Class method rebuilding a NamedTuple from its message (type name, attribute names and
    # serialized attributes). The placeholder type is not stored, so the result has none.
    return NamedTuple(
        proto.type_name, tuple(proto.attribute_names),
        Tuple.experimental_from_proto(proto.attributes).components)

  def experimental_as_proto(self) -> default_types_pb2.SerializedNamedTuple:
    # [FUNCTION DOC]
    # Serializes the type name, attribute names and attribute types into a message. The
    # placeholder type is not serialized.
    return default_types_pb2.SerializedNamedTuple(
        type_name=self.type_name,
        attribute_names=list(self.attribute_names),
        attributes=self.attributes.experimental_as_proto())

  def placeholder_value(self, placeholder_context) -> Any:
    # [FUNCTION DOC]
    # Builds a placeholder instance of the concrete named tuple class from the attributes'
    # placeholder values. Raises a value error if no placeholder type is known (for example after
    # deserialization).
    if self._placeholder_type is None:
      # We don't need to trace after serialization so it is not needed but we
      # can generate a placeholder type using the description if ever needed.
      raise ValueError("Can not generate placeholder value for NamedTuple with"
                       " unspecified placeholder_type. Note: placeholder_type "
                       "is lost during serialization.")
    attribute_placeholders = [
        attribute.placeholder_value(placeholder_context)
        for attribute in self.attributes.components
    ]
    return self._placeholder_type(*attribute_placeholders)

  def to_tensors(self, value: Any):
    # [FUNCTION DOC]
    # Flattens a concrete named tuple into a list of tensors by reading each named attribute and
    # converting it with the matching attribute type. Asserts the value is a named tuple.
    assert util.is_namedtuple(value)
    flattened_values = []
    for attribute_name, attribute_type in zip(
        self.attribute_names, self.attributes.components):
      attribute_value = getattr(value, attribute_name)
      flattened_values.extend(attribute_type.to_tensors(attribute_value))
    return flattened_values

  def from_tensors(self, tensors) -> Any:
    # [FUNCTION DOC]
    # Rebuilds a named tuple instance from the tensor iterator, one attribute at a time. Raises a
    # value error if there is no placeholder type (e.g. the type was deserialized).
    if self._placeholder_type is None:
      raise ValueError("Packing serialized NamedTuples is not supported.")

    return self._placeholder_type(
        *[c.from_tensors(tensors) for c in self.attributes.components]
    )

  def flatten(self) -> PythonList[trace.TraceType]:
    # [FUNCTION DOC]
    # Returns the flat list of leaf TraceTypes of all attributes, in attribute order.
    flattened_types = []

    for component in self.attributes.components:
      flattened_types.extend(component.flatten())

    return flattened_types

  def cast(self, value: Any, casting_context) -> Any:
    # [FUNCTION DOC]
    # Casts a concrete named tuple by casting each attribute. Asserts the value is a named tuple
    # whose field names equal this type's attribute names. Returns a new instance of the
    # placeholder type if any attribute changed, otherwise the original value.
    # Value must have same attributes with the TraceType
    assert util.is_namedtuple(
        value
    ), f"Cannot cast {value!r} to type {self._placeholder_type!r}."
    value_dict = value._asdict()
    assert set(value_dict.keys()) == set(
        self.attribute_names
    ), f"{value!r} has different attributes with the TraceType {self!r}"

    casted_values, was_casted = util.cast_and_return_whether_casted(
        self.attributes.components,
        [getattr(value, name) for name in self.attribute_names],
        casting_context,
    )
    if was_casted:
      return self._placeholder_type(*casted_values)
    else:
      return value

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Returns a hash combining the type name, attribute names and attribute types.
    return hash((self.type_name, self.attribute_names, self.attributes))

  def __eq__(self, other: Any) -> bool:
    # [FUNCTION DOC]
    # Equality: not-implemented for non-TraceType objects; false for non-NamedTuple TraceTypes;
    # otherwise true if type name, attribute names and attribute types are all equal.
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, NamedTuple):
      return False

    return (self.type_name == other.type_name and
            self.attribute_names == other.attribute_names and
            self.attributes == other.attributes)

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns a readable string: the type name followed by name and type pairs of the attributes.
    paired = [
        f"[{n!r}, {c!r}]"
        for n, c in zip(self.attribute_names, self.attributes.components)
    ]
    return f"{self.type_name}[{', '.join(paired)}]"


class Attrs(trace.TraceType):
  # [CLASS DOC]
  # TraceType for an instance of a class decorated with the attrs library. Internally stores the
  # type name, attribute names and attribute TraceTypes as a NamedTuple TraceType, plus the
  # concrete class used for placeholders and repacking. Not registered as serializable by the
  # registry in this module, although it defines the protobuf conversion methods.
  """Represents a class annotated by attr.s."""

  def __init__(self,
               type_name: str,
               attribute_names: PythonTuple[str],
               attributes: PythonTuple[trace.TraceType],
               placeholder_type: Optional[Type[Any]] = None):
    # [FUNCTION DOC]
    # Takes the class name, tuple of attribute names, tuple of attribute TraceTypes and an
    # optional concrete attrs class (default nothing). Wraps the first three in a NamedTuple
    # TraceType and stores the concrete class for placeholder creation.
    self.named_attributes = NamedTuple(type_name, attribute_names, attributes)
    self._placeholder_type = placeholder_type

  @classmethod
  def from_type_and_attributes(
      cls, attrs_type: Any,
      attributes: PythonTuple[trace.TraceType]) -> "Attrs":
    # [FUNCTION DOC]
    # Class method factory: builds an Attrs TraceType from an attrs-decorated class and a tuple of
    # attribute TraceTypes, reading the class name and the attribute names from the class's attrs
    # metadata and remembering the class as the placeholder type.
    return Attrs(attrs_type.__name__,
                 tuple(attr.name for attr in attrs_type.__attrs_attrs__),
                 attributes, attrs_type)

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    # [FUNCTION DOC]
    # Returns true if the other is an Attrs and the wrapped named-tuple types are in a subtype
    # relation; false otherwise.
    if not isinstance(other, Attrs):
      return False

    return self.named_attributes.is_subtype_of(other.named_attributes)

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["Attrs"]:
    # [FUNCTION DOC]
    # Returns nothing if any given TraceType is not an Attrs or the wrapped types have no common
    # supertype; otherwise a new Attrs with the supertype attributes and this one's placeholder
    # type.
    """See base class."""
    if not all(isinstance(other, Attrs) for other in others):
      return None

    supertyped_attributes = (
        self.named_attributes.most_specific_common_supertype(
            [other.named_attributes for other in others]
        )
    )

    if supertyped_attributes is None:
      return None

    return Attrs(self.named_attributes.type_name,
                 self.named_attributes.attribute_names,
                 supertyped_attributes.attributes.components,
                 self._placeholder_type)

  @classmethod
  def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedAttrs]:
    # [FUNCTION DOC]
    # Class method returning the protobuf message class for an Attrs.
    return default_types_pb2.SerializedAttrs

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedAttrs) -> "Attrs":
    # [FUNCTION DOC]
    # Class method rebuilding an Attrs from its message; the placeholder type is not stored so
    # the result has none.
    return Attrs(
        proto.named_attributes.type_name,
        tuple(proto.named_attributes.attribute_names),
        Tuple.experimental_from_proto(
            proto.named_attributes.attributes).components)

  def experimental_as_proto(self) -> default_types_pb2.SerializedAttrs:
    # [FUNCTION DOC]
    # Serializes the wrapped named-tuple description into an Attrs message.
    return default_types_pb2.SerializedAttrs(
        named_attributes=self.named_attributes.experimental_as_proto())

  def placeholder_value(self, placeholder_context) -> Any:
    # [FUNCTION DOC]
    # Builds a placeholder instance of the attrs class from the attributes' placeholder values.
    # Raises a value error if no placeholder type is known (e.g. after deserialization).
    if self._placeholder_type is None:
      # We don't need to trace after serialization so it is not needed but we
      # can generate a placeholder type using the description if ever needed.
      raise ValueError("Can not generate placeholder value for Attrs with"
                       " unspecified placeholder_type. Note: placeholder_type "
                       "is lost during serialization.")
    attribute_placeholders = [
        attribute.placeholder_value(placeholder_context)
        for attribute in self.named_attributes.attributes.components
    ]
    return self._placeholder_type(*attribute_placeholders)

  def to_tensors(self, value: Any):
    # [FUNCTION DOC]
    # Flattens a concrete attrs instance into a list of tensors by reading each attribute and
    # converting it with the matching attribute type. Asserts the value is an attrs instance.
    assert util.is_attrs(value)
    flattened_values = []
    for attribute_name, attribute_type in zip(
        self.named_attributes.attribute_names,
        self.named_attributes.attributes.components):
      attribute_value = getattr(value, attribute_name)
      flattened_values.extend(attribute_type.to_tensors(attribute_value))
    return flattened_values

  def from_tensors(self, tensors):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message mentions NamedTuples although this is the attrs type.
    # Rebuilds an attrs instance from the tensor iterator, one attribute at a time. Raises a value
    # error if there is no placeholder type.
    if self._placeholder_type is None:
      raise ValueError("Packing serialized NamedTuples is not supported.")

    return self._placeholder_type(
        *[
            c.from_tensors(tensors)
            for c in self.named_attributes.attributes.components
        ]
    )

  def flatten(self) -> PythonList[trace.TraceType]:
    # [FUNCTION DOC]
    # Returns the flat list of leaf TraceTypes of all attributes, in attribute order.
    flattened_types = []

    for component in self.named_attributes.attributes.components:
      flattened_types.extend(component.flatten())

    return flattened_types

  def cast(self, value: Any, casting_context) -> Any:
    # [FUNCTION DOC]
    # Casts a concrete attrs instance by casting each attribute. Asserts the value is an attrs
    # instance. Returns a new instance of the placeholder type if any attribute changed, otherwise
    # the original value.
    assert util.is_attrs(value)

    attr_names = self.named_attributes.attribute_names
    casted_values, was_casted = util.cast_and_return_whether_casted(
        self.named_attributes.attributes.components,
        [getattr(value, name) for name in attr_names],
        casting_context,
    )

    if was_casted:
      return self._placeholder_type(*casted_values)
    else:
      return value

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Returns the hash of the wrapped named-tuple description.
    return hash(self.named_attributes)

  def __eq__(self, other: Any) -> bool:
    # [FUNCTION DOC]
    # Equality: not-implemented for non-TraceType objects; false for non-Attrs TraceTypes;
    # otherwise compares the wrapped named-tuple descriptions.
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Attrs):
      return False

    return self.named_attributes == other.named_attributes

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns a readable string: the type name followed by name and type pairs of the attributes.
    name_component_zip = zip(
        self.named_attributes.attribute_names,
        self.named_attributes.attributes.components,
    )
    paired = [f"[{n!r}, {c!r}]" for n, c in name_component_zip]
    return f"{self.named_attributes.type_name}[{', '.join(paired)}]"


class Dict(trace.TraceType, serialization.Serializable):
  # [CLASS DOC]
  # TraceType for a Python mapping. Holds a mapping from keys to the TraceTypes of the values,
  # plus an optional concrete mapping class used to build placeholders and repacked values (lost
  # on serialization). Two Dicts are related only if they have exactly the same key set. Tensor
  # flattening visits keys in sorted order. Serializable and registered.
  """Represents a dictionary of TraceType objects.

  Attributes:
    mapping: A mapping from keys to corresponding TraceTypes of the dict values.
  """

  def __init__(self,
               mapping: PythonDict[Hashable, trace.TraceType],
               placeholder_type: Optional[Type[Any]] = None):
    # [FUNCTION DOC]
    # Takes a mapping from hashable keys to value TraceTypes and an optional concrete mapping
    # class (default nothing) used to create placeholders. Stores both.
    self.mapping = mapping
    self._placeholder_type = placeholder_type

  def _has_same_structure(self, other):
    # [FUNCTION DOC]
    # HELPER: Structure check used by the subtype and supertype computations of Dict.
    # Returns true only if the other is a Dict with exactly the same set of keys.
    if not isinstance(other, Dict):
      return False

    return self.mapping.keys() == other.mapping.keys()

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    # [FUNCTION DOC]
    # Returns true if the other is a Dict with the same keys and every value type here is a
    # subtype of the other's value type for that key; false otherwise.
    """See base class."""
    if not self._has_same_structure(other):
      return False

    # We need all keys to be present because there can be logic relying on
    # their existence or lack thereof and hence can not guarantee subtype based
    # on a subset or superset of keys.
    # Only the tracing code can explicitly check for key dependencies and inform
    # that decision.
    return all(self.mapping[key].is_subtype_of(other.mapping[key])
               for key in self.mapping)

  def most_specific_common_supertype(
      self, types: Sequence[trace.TraceType]) -> Optional["Dict"]:
    # [FUNCTION DOC]
    # Returns nothing unless all given TraceTypes are Dicts with the same keys and each key has a
    # common value supertype; otherwise returns a new Dict of the per-key supertypes keeping this
    # one's placeholder type.
    """See base class."""
    if not all(self._has_same_structure(other) for other in types):
      return None

    new_mapping = {}
    for key in self.mapping.keys():
      common = self.mapping[key].most_specific_common_supertype(
          [other.mapping[key] for other in types])
      if common is None:
        return None
      else:
        new_mapping[key] = common

    return Dict(new_mapping, self._placeholder_type)

  @classmethod
  def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedDict]:
    # [FUNCTION DOC]
    # Class method returning the protobuf message class used to serialize a Dict.
    return default_types_pb2.SerializedDict

  @classmethod
  def experimental_from_proto(
      cls, proto: default_types_pb2.SerializedDict) -> "Dict":
    # [FUNCTION DOC]
    # Class method rebuilding a Dict from its message: keys are decoded as Literals, values are
    # deserialized through the registry. The placeholder type is not stored, so none is set.
    return Dict({
        Literal.experimental_from_proto(k).value: serialization.deserialize(v)
        for k, v in zip(proto.keys, proto.values)
    })

  def experimental_as_proto(self) -> default_types_pb2.SerializedDict:
    # [FUNCTION DOC]
    # Serializes keys (as Literals, so they must be serializable literal values; otherwise a
    # value error is raised) and values (via the registry) into parallel lists in the message.
    return default_types_pb2.SerializedDict(
        keys=[Literal(k).experimental_as_proto() for k in self.mapping.keys()],
        values=[serialization.serialize(v) for v in self.mapping.values()])

  def placeholder_value(self, placeholder_context) -> Any:
    # [FUNCTION DOC]
    # Builds a placeholder mapping from the values' placeholder values, keeping key order. Uses
    # a plain dict when the placeholder type is a default-dict, otherwise constructs the
    # placeholder type from key and value pairs. Raises a value error if no placeholder type is
    # known.
    if self._placeholder_type is None:
      raise ValueError("Can not generate placeholder value for Dict with"
                       " unspecified placeholder_type. Note: placeholder_type "
                       "is lost during serialization.")
    attribute_placeholders = [
        (key, value.placeholder_value(placeholder_context))
        for key, value in self.mapping.items()
    ]
    if self._placeholder_type is collections.defaultdict:
      return dict(attribute_placeholders)
    return self._placeholder_type(attribute_placeholders)

  def to_tensors(self, value: Any):
    # [FUNCTION DOC]
    # Flattens a concrete mapping into a list of tensors, visiting keys in sorted order and
    # converting each value with its value type. Asserts the value is a mapping.
    assert isinstance(value, collections.abc.Mapping)
    flattened_values = []
    for key in sorted(self.mapping.keys()):
      comp_value, comp_type = value[key], self.mapping[key]
      flattened_values.extend(comp_type.to_tensors(comp_value))
    return flattened_values

  def from_tensors(self, tensors):
    # [FUNCTION DOC]
    # Rebuilds a mapping from the tensor iterator: values are consumed in sorted key order, then
    # the result is assembled in original key order using the placeholder type (a plain dict for
    # a default-dict). Raises a value error if no placeholder type is known.
    if self._placeholder_type is None:
      raise ValueError("Packing serialized Dict is not supported.")

    sorted_traversal = {
        key: self.mapping[key].from_tensors(tensors)
        for key in sorted(self.mapping)
    }

    if self._placeholder_type is collections.defaultdict:
      return {key: sorted_traversal[key] for key in self.mapping}

    return self._placeholder_type(
        (key, sorted_traversal[key]) for key in self.mapping
    )

  def flatten(self) -> PythonList[trace.TraceType]:
    # [FUNCTION DOC]
    # Returns the flat list of leaf TraceTypes of all values, ordered by sorted key.
    flattened_types = []

    for key in sorted(self.mapping.keys()):
      flattened_types.extend(self.mapping[key].flatten())

    return flattened_types

  def cast(self, value: Any, casting_context) -> Any:
    # [FUNCTION DOC]
    # KNOWN ISSUE: when a change is needed it builds the result by calling the placeholder type
    # with keyword arguments, which fails if no placeholder type is set or keys are not strings.
    # Casts a concrete mapping by casting each value. Asserts the value is a mapping with exactly
    # this type's keys. Returns a new mapping of the placeholder type if any value changed,
    # otherwise the original value.
    # Value must have same keys with the TraceType
    assert isinstance(
        value, collections.abc.Mapping
    ), f"Can not cast {value!r} to a Dict type."
    assert set(value.keys()) == set(
        self.mapping.keys()
    ), f"{value!r} has different keys with the TraceType {self!r}."

    casted_values, was_casted = util.cast_and_return_whether_casted(
        self.mapping.values(),
        [value[k] for k in self.mapping.keys()],
        casting_context,
    )

    if was_casted:
      return self._placeholder_type(
          **{k: v for k, v in zip(self.mapping.keys(), casted_values)}
      )
    else:
      return value

  def __eq__(self, other) -> bool:
    # [FUNCTION DOC]
    # Equality: not-implemented for non-TraceType objects; false for non-Dict TraceTypes;
    # otherwise compares the key to value-type mappings.
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    if not isinstance(other, Dict):
      return False

    return self.mapping == other.mapping

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Returns a hash based only on the set of keys (values are ignored), which is consistent with
    # equality.
    return hash(frozenset(self.mapping.keys()))

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns a readable string listing key and value-type pairs.
    paired = [f"[{n!r}, {t!r}]" for n, t in self.mapping.items()]
    return f"{self.__class__.__name__}[{', '.join(paired)}]"


serialization.register_serializable(Literal)
serialization.register_serializable(Tuple)
serialization.register_serializable(List)
serialization.register_serializable(NamedTuple)
serialization.register_serializable(Attrs)
serialization.register_serializable(Dict)
