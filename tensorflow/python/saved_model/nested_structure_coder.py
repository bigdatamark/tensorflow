# [MODULE DOC]
# Encodes nested Python structures (lists, tuples, named tuples, dicts, strings, floats, ints,
# booleans, null, tensor data types and type specs) into structured-value protos and decodes
# them back. Used when saving and restoring function signatures and specs in a SavedModel.
# Exposes encode_structure, decode_proto, can_encode and register_codec, plus codec classes.
# Import-time side effect: builds the module-level codec list, ordered so that later entries
# take priority. Other modules register more codecs with register_codec.
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
"""Module that encodes (decodes) nested structures into (from) protos.

The intended use is to serialize everything needed to restore a `Function` that
was saved into a SavedModel. This may include concrete function inputs and
outputs, signatures, function specs, etc.

Example use:
# Encode into proto.
signature_proto = nested_structure_coder.encode_structure(
    function.input_signature)
# Decode into a Python object.
restored_signature = nested_structure_coder.decode_proto(signature_proto)
"""

import collections
import functools
import warnings

from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


class NotEncodableError(Exception):
  # [CLASS DOC]
  # Exception type raised when no registered codec can encode (or decode) a given object. It
  # carries only the message text; it adds no attributes or behavior to the base exception.
  """Error raised when a coder cannot encode an object."""


def register_codec(x):
  # [FUNCTION DOC]
  # Adds a codec object to the end of the module-level codec list. Later registrations take
  # priority over earlier ones because lookups scan the list in reverse. Input x is any object
  # providing the can-encode, do-encode, can-decode and do-decode operations; it is not
  # validated. Returns nothing. Side effect: mutates the global codec registry.
  """Registers a codec to use for encoding/decoding.

  Args:
    x: The codec object to register. The object must implement can_encode,
      do_encode, can_decode, and do_decode. See the various _*Codec classes for
      examples.
  """
  _codecs.append(x)


def _get_encoders():
  # [FUNCTION DOC]
  # HELPER: supplies the encoder table for the structure-mapping routine.
  # Returns a new list with one pair (can-encode predicate, do-encode function) per registered
  # codec, in registration order. No inputs, no side effects.
  return [(c.can_encode, c.do_encode) for c in _codecs]


def _get_decoders():
  # [FUNCTION DOC]
  # HELPER: supplies the decoder table for the structure-mapping routine.
  # Returns a new list with one pair (can-decode predicate, do-decode function) per registered
  # codec, in registration order. No inputs, no side effects.
  return [(c.can_decode, c.do_decode) for c in _codecs]


def _map_structure(pyobj, coders):
  # [FUNCTION DOC]
  # HELPER: shared recursive engine behind encoding and decoding.
  # Inputs: pyobj, the object (or proto) to convert; coders, a list of (predicate, converter)
  # pairs. Scans the pairs from last to first and applies the first converter whose predicate
  # accepts pyobj, passing it a recursion callback bound to the same pairs so nested elements
  # are converted the same way. Returns the converter result. Raises NotEncodableError when no
  # predicate accepts the object.
  # Iterate through the codecs in the reverse order they were registered in,
  #   as the most specific codec should be checked first.
  for can, do in reversed(coders):
    if can(pyobj):
      recursion_fn = functools.partial(_map_structure, coders=coders)
      return do(pyobj, recursion_fn)
  raise NotEncodableError(
      f"No encoder for object {str(pyobj)} of type {type(pyobj)}.")


@tf_export("__internal__.saved_model.encode_structure", v1=[])
def encode_structure(nested_structure):
  # [FUNCTION DOC]
  # Public entry point (exported as an internal saved-model API) that encodes a nested structure
  # of lists, tuples, named tuples, dicts, strings, floats, integers, booleans, nothing values,
  # data types and type specs into a structured-value proto. Input: the structure. Returns the
  # proto. Raises NotEncodableError if any element has no codec. No side effects except possible
  # warnings from type spec encoding.
  """Encodes nested structures composed of encodable types into a proto.

  Args:
    nested_structure: Structure to encode.

  Returns:
    Encoded proto.

  Raises:
    NotEncodableError: For values for which there are no encoders.
  """
  return _map_structure(nested_structure, _get_encoders())


def can_encode(nested_structure):
  # [FUNCTION DOC]
  # Tests encodability by actually attempting the full encoding and discarding the result.
  # Input: the nested structure. Returns true on success and false if the encoder raises
  # NotEncodableError; other exceptions propagate. Side effect: may emit the type spec warning,
  # and costs a full encode.
  """Determines whether a nested structure can be encoded into a proto.

  Args:
    nested_structure: Structure to encode.

  Returns:
    True if the nested structured can be encoded.
  """
  try:
    encode_structure(nested_structure)
  except NotEncodableError:
    return False
  return True


@tf_export("__internal__.saved_model.decode_proto", v1=[])
def decode_proto(proto):
  # [FUNCTION DOC]
  # Public entry point (exported as an internal saved-model API) that turns a structured-value
  # proto back into the nested Python structure. Input: the proto. Returns the decoded object
  # (named tuples come back as freshly created tuple types, dicts as plain dicts). Raises
  # NotEncodableError if no codec recognizes the proto, and ValueError for unregistered or
  # unsupported type spec classes.
  """Decodes proto representing a nested structure.

  Args:
    proto: Proto to decode.

  Returns:
    Decoded structure.

  Raises:
    NotEncodableError: For values for which there are no encoders.
  """
  return _map_structure(proto, _get_decoders())


class _ListCodec:
  # [CLASS DOC]
  # HELPER: codec for Python lists, registered in the module codec list. Encodes to the list
  # value field of a structured value, recursing on every element. Stateless.
  """Codec for lists."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if pyobj is a list (including subclasses).
    return isinstance(pyobj, list)

  def do_encode(self, list_value, encode_fn):
    # [FUNCTION DOC]
    # Builds a structured value holding a list value whose entries are the encodings of each
    # element, produced with the recursion callback encode_fn. Returns the new proto.
    encoded_list = struct_pb2.StructuredValue()
    encoded_list.list_value.CopyFrom(struct_pb2.ListValue())
    for element in list_value:
      encoded_list.list_value.values.add().CopyFrom(encode_fn(element))
    return encoded_list

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured value proto has its list value field set.
    return value.HasField("list_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Returns a Python list made by decoding each stored element with the recursion callback.
    return [decode_fn(element) for element in value.list_value.values]


def _is_tuple(obj):
  # [FUNCTION DOC]
  # HELPER: lets the tuple codec skip named tuples.
  # Returns true if obj is a tuple that is not a named tuple.
  return not _is_named_tuple(obj) and isinstance(obj, tuple)


def _is_named_tuple(instance):
  # [FUNCTION DOC]
  # HELPER: duck-typed named tuple detection for the tuple and named tuple codecs.
  # Returns true if instance is a tuple having a fields attribute that is a sequence whose
  # entries are all strings. Otherwise false. No side effects.
  """Returns True iff `instance` is a `namedtuple`.

  Args:
    instance: An instance of a Python object.

  Returns:
    True if `instance` is a `namedtuple`.
  """
  if not isinstance(instance, tuple):
    return False
  return (hasattr(instance, "_fields") and
          isinstance(instance._fields, collections_abc.Sequence) and
          all(isinstance(f, str) for f in instance._fields))


class _TupleCodec:
  # [CLASS DOC]
  # HELPER: codec for plain (non-named) tuples, registered in the module codec list. Uses the
  # tuple value field of a structured value and recurses on elements. Stateless.
  """Codec for tuples."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if pyobj is a tuple that is not a named tuple.
    return _is_tuple(pyobj)

  def do_encode(self, tuple_value, encode_fn):
    # [FUNCTION DOC]
    # Builds a structured value with a tuple value holding the encoding of every element, made
    # with the recursion callback encode_fn. Returns the new proto.
    encoded_tuple = struct_pb2.StructuredValue()
    encoded_tuple.tuple_value.CopyFrom(struct_pb2.TupleValue())
    for element in tuple_value:
      encoded_tuple.tuple_value.values.add().CopyFrom(encode_fn(element))
    return encoded_tuple

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured value proto has its tuple value field set.
    return value.HasField("tuple_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Returns a Python tuple built by decoding each stored element with the recursion callback.
    return tuple(decode_fn(element) for element in value.tuple_value.values)


class _DictCodec:
  # [CLASS DOC]
  # HELPER: codec for mappings, registered in the module codec list. Uses the dict value field
  # of a structured value (field name to encoded value). Stateless.
  """Codec for dicts."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if pyobj is any mapping (not only plain dicts).
    return isinstance(pyobj, collections_abc.Mapping)

  def do_encode(self, dict_value, encode_fn):
    # [FUNCTION DOC]
    # KNOWN ISSUE: keys are used directly as proto map keys, so non-string keys fail inside the
    # protocol buffer library rather than raising NotEncodableError; mapping subclasses are
    # decoded back as plain dicts.
    # Builds a structured value with a dict value whose fields are the encodings of each value,
    # made with the callback encode_fn. Returns the new proto.
    encoded_dict = struct_pb2.StructuredValue()
    encoded_dict.dict_value.CopyFrom(struct_pb2.DictValue())
    for key, value in dict_value.items():
      encoded_dict.dict_value.fields[key].CopyFrom(encode_fn(value))
    return encoded_dict

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured value proto has its dict value field set.
    return value.HasField("dict_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Returns a plain dict from field names to the decoded values (decoded with the callback).
    return {key: decode_fn(val) for key, val in value.dict_value.fields.items()}


class _NamedTupleCodec:
  # [CLASS DOC]
  # HELPER: codec for named tuples, registered in the module codec list. Stores the type name
  # and an ordered list of key/encoded-value pairs in the named tuple value field. Decoding
  # creates a brand-new named tuple type each time. Stateless.
  """Codec for namedtuples.

  Encoding and decoding a namedtuple reconstructs a namedtuple with a different
  actual Python type, but with the same `typename` and `fields`.
  """

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if pyobj looks like a named tuple (tuple with string field names).
    return _is_named_tuple(pyobj)

  def do_encode(self, named_tuple_value, encode_fn):
    # [FUNCTION DOC]
    # Builds a structured value with a named tuple value: the type's class name plus one
    # key/value pair per field, each value encoded with the recursion callback encode_fn.
    # Returns the new proto.
    encoded_named_tuple = struct_pb2.StructuredValue()
    encoded_named_tuple.named_tuple_value.CopyFrom(struct_pb2.NamedTupleValue())
    encoded_named_tuple.named_tuple_value.name = \
      named_tuple_value.__class__.__name__
    for key in named_tuple_value._fields:
      pair = encoded_named_tuple.named_tuple_value.values.add()
      pair.key = key
      pair.value.CopyFrom(encode_fn(named_tuple_value._asdict()[key]))
    return encoded_named_tuple

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured value proto has its named tuple value field set.
    return value.HasField("named_tuple_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Decodes each stored key/value pair with the recursion callback, creates a new named tuple
    # type with the stored name and the keys in order, and returns an instance filled with the
    # decoded values. The result is not an instance of the original class. Side effect: creates
    # a new type on every call.
    key_value_pairs = value.named_tuple_value.values
    items = [(pair.key, decode_fn(pair.value)) for pair in key_value_pairs]
    named_tuple_type = collections.namedtuple(value.named_tuple_value.name,
                                              [item[0] for item in items])
    return named_tuple_type(**dict(items))


class _Float64Codec:
  # [CLASS DOC]
  # HELPER: codec for Python floats, stored in the 64-bit float field. Stateless leaf codec.
  """Codec for floats."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if pyobj is a float.
    return isinstance(pyobj, float)

  def do_encode(self, float64_value, encode_fn):
    # [FUNCTION DOC]
    # Returns a structured value with the 64-bit float field set. The recursion callback is
    # unused (leaf value).
    del encode_fn
    value = struct_pb2.StructuredValue()
    value.float64_value = float64_value
    return value

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured value proto has its 64-bit float field set.
    return value.HasField("float64_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Returns the stored float; the callback is unused.
    del decode_fn
    return value.float64_value


class _Int64Codec:
  # [CLASS DOC]
  # HELPER: codec for Python integers, stored in the 64-bit integer field. Booleans are
  # excluded so the boolean codec handles them. Stateless leaf codec.
  """Codec for Python integers (limited to 64 bit values)."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says integers are limited to 64 bits, but nothing checks the
    # range here; out-of-range values only fail later when assigned into the proto field.
    # Returns true if pyobj is an integer and not a boolean.
    return not isinstance(pyobj, bool) and isinstance(pyobj, int)

  def do_encode(self, int_value, encode_fn):
    # [FUNCTION DOC]
    # Returns a structured value with the 64-bit integer field set; the callback is unused.
    # The protocol buffer library raises an error for values outside the 64-bit range.
    del encode_fn
    value = struct_pb2.StructuredValue()
    value.int64_value = int_value
    return value

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured value proto has its 64-bit integer field set.
    return value.HasField("int64_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Returns the stored value converted to a Python integer; the callback is unused.
    del decode_fn
    return int(value.int64_value)


class _StringCodec:
  # [CLASS DOC]
  # HELPER: codec for text strings, stored in the string field of a structured value.
  # Stateless leaf codec.
  """Codec for strings.

  See StructuredValue.string_value in proto/struct.proto for more detailed
  explanation.
  """

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if pyobj is a text string (byte strings are not accepted).
    return isinstance(pyobj, str)

  def do_encode(self, string_value, encode_fn):
    # [FUNCTION DOC]
    # Returns a structured value with the string field set; the callback is unused.
    del encode_fn
    value = struct_pb2.StructuredValue()
    value.string_value = string_value
    return value

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured value proto has its string field set.
    return value.HasField("string_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Returns the stored string converted to a native text string; the callback is unused.
    del decode_fn
    return compat.as_str(value.string_value)


class _NoneCodec:
  # [CLASS DOC]
  # HELPER: codec for the null value, stored as an empty none value message. Stateless.
  """Codec for None."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true only if pyobj is the null value.
    return pyobj is None

  def do_encode(self, none_value, encode_fn):
    # [FUNCTION DOC]
    # Returns a structured value with the empty none value message set; both arguments are
    # ignored.
    del encode_fn, none_value
    value = struct_pb2.StructuredValue()
    value.none_value.CopyFrom(struct_pb2.NoneValue())
    return value

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured value proto has its none value field set.
    return value.HasField("none_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Always returns the null value; both arguments are ignored.
    del decode_fn, value
    return None


class _BoolCodec:
  # [CLASS DOC]
  # HELPER: codec for booleans, stored in the boolean field. Stateless leaf codec.
  """Codec for booleans."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if pyobj is a boolean.
    return isinstance(pyobj, bool)

  def do_encode(self, bool_value, encode_fn):
    # [FUNCTION DOC]
    # Returns a structured value with the boolean field set; the callback is unused.
    del encode_fn
    value = struct_pb2.StructuredValue()
    value.bool_value = bool_value
    return value

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured value proto has its boolean field set.
    return value.HasField("bool_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Returns the stored boolean; the callback is unused.
    del decode_fn
    return value.bool_value


class _TensorTypeCodec:
  # [CLASS DOC]
  # HELPER: codec for tensor data types, stored as the numeric data type enum value in the
  # tensor dtype field. Stateless leaf codec.
  """Codec for `TensorType`."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if pyobj is a tensor data type object.
    return isinstance(pyobj, dtypes.DType)

  def do_encode(self, tensor_dtype_value, encode_fn):
    # [FUNCTION DOC]
    # Returns a structured value whose tensor dtype field holds the data type's enum number; the
    # callback is unused.
    del encode_fn
    encoded_tensor_type = struct_pb2.StructuredValue()
    encoded_tensor_type.tensor_dtype_value = tensor_dtype_value.as_datatype_enum
    return encoded_tensor_type

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured value proto has its tensor dtype field set.
    return value.HasField("tensor_dtype_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Returns the data type object for the stored enum number; the callback is unused.
    del decode_fn
    return dtypes.DType(value.tensor_dtype_value)


class BuiltInTypeSpecCodec:
  # [CLASS DOC]
  # Codec for one built-in type spec class, identified by a small proto enum number. Each
  # built-in type spec module creates one instance and registers it. Class-level lists record
  # every class and enum number already claimed (shared by all instances, never cleared), which
  # prevents duplicate registration. Instance attributes: type_spec_class and
  # type_spec_proto_enum. Encoding stores the serialized spec state and the flat component count.
  """Codec for built-in `TypeSpec` classes.

  Built-in TypeSpec's that do not require a custom codec implementation
  register themselves by instantiating this class and passing it to
  register_codec.

  Attributes:
    type_spec_class: The built-in TypeSpec class that the
      codec is instantiated for.
    type_spec_proto_enum: The proto enum value for the built-in TypeSpec class.
  """

  _BUILT_IN_TYPE_SPEC_PROTOS = []
  _BUILT_IN_TYPE_SPECS = []

  def __init__(self, type_spec_class, type_spec_proto_enum):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the valid enum range (1 to 10) is hard-coded here, so adding more built-in
    # type specs requires editing this check. The duplicate checks use shared class-level lists
    # that are appended to only after all validation passes.
    # Inputs: type_spec_class, a subclass of the internal type spec base; type_spec_proto_enum,
    # an integer from 1 to 10 identifying it in the proto. Raises ValueError if the class is not
    # a type spec subclass, already has a codec, or the enum is already used, not an integer, or
    # out of range. Side effect: records both values in the shared class-level lists.
    if not issubclass(type_spec_class, internal.TypeSpec):
      raise ValueError(
          f"The type '{type_spec_class}' does not subclass tf.TypeSpec.")

    if type_spec_class in self._BUILT_IN_TYPE_SPECS:
      raise ValueError(
          f"The type '{type_spec_class}' already has an instantiated codec.")

    if type_spec_proto_enum in self._BUILT_IN_TYPE_SPEC_PROTOS:
      raise ValueError(
          f"The proto value '{type_spec_proto_enum}' is already registered."
      )

    if (not isinstance(type_spec_proto_enum, int)
        or type_spec_proto_enum <= 0
        or type_spec_proto_enum > 10):
      raise ValueError(f"The proto value '{type_spec_proto_enum}' is invalid.")

    self.type_spec_class = type_spec_class
    self.type_spec_proto_enum = type_spec_proto_enum

    self._BUILT_IN_TYPE_SPECS.append(type_spec_class)
    self._BUILT_IN_TYPE_SPEC_PROTOS.append(type_spec_proto_enum)

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if pyobj is an instance of the type spec class this codec was built for.
    """Returns true if `pyobj` can be encoded as the built-in TypeSpec."""
    return isinstance(pyobj, self.type_spec_class)

  def do_encode(self, type_spec_value, encode_fn):
    # [FUNCTION DOC]
    # Serializes the spec's state, counts its flattened component specs (expanding composites),
    # and returns a structured value holding a type spec message with this codec's enum number,
    # the state encoded through the callback encode_fn, the class name and the component count.
    """Returns an encoded proto for the given built-in TypeSpec."""
    type_state = type_spec_value._serialize()  # pylint: disable=protected-access
    num_flat_components = len(nest.flatten(
        type_spec_value._component_specs, expand_composites=True))  # pylint: disable=protected-access
    encoded_type_spec = struct_pb2.StructuredValue()
    encoded_type_spec.type_spec_value.CopyFrom(
        struct_pb2.TypeSpecProto(
            type_spec_class=self.type_spec_proto_enum,
            type_state=encode_fn(type_state),
            type_spec_class_name=self.type_spec_class.__name__,
            num_flat_components=num_flat_components))
    return encoded_type_spec

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the proto has a type spec field whose class enum equals this codec's enum
    # number; false otherwise.
    """Returns true if `value` can be decoded into its built-in TypeSpec."""
    if value.HasField("type_spec_value"):
      type_spec_class_enum = value.type_spec_value.type_spec_class
      return type_spec_class_enum == self.type_spec_proto_enum
    return False

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Decodes the stored type state with the callback decode_fn and returns the result of the
    # type spec class's deserialize operation applied to it.
    """Returns the built in `TypeSpec` encoded by the proto `value`."""
    type_spec_proto = value.type_spec_value
    # pylint: disable=protected-access
    return self.type_spec_class._deserialize(
        decode_fn(type_spec_proto.type_state)
    )


# TODO(b/238903802): Use TraceType serialization and specific protos.
class _TypeSpecCodec:
  # [CLASS DOC]
  # HELPER: generic codec for type specs whose class was registered by name in the type spec
  # registry. Stored as a registered type spec message with the registered name; decoding needs
  # the class to be imported and registered. Stateless.
  """Codec for `tf.TypeSpec`."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if pyobj is a type spec whose class has a registered name; false for other
    # objects or unregistered type specs.
    """Returns true if `pyobj` can be encoded as a TypeSpec."""
    # Check if it's a registered type.
    if isinstance(pyobj, internal.TypeSpec):
      try:
        type_spec_registry.get_name(type(pyobj))
        return True
      except ValueError:
        return False

    return False

  def do_encode(self, type_spec_value, encode_fn):
    # [FUNCTION DOC]
    # Looks up the registered name of the spec's class, issues a user warning that loading will
    # require the type to be imported and registered, then builds a structured value with a
    # registered type spec message holding the name, the state encoded with the callback, and
    # the flat component count. Raises ValueError if the class is not registered.
    """Returns an encoded proto for the given `tf.TypeSpec`."""
    type_spec_class_name = type_spec_registry.get_name(type(type_spec_value))
    type_spec_class = struct_pb2.TypeSpecProto.REGISTERED_TYPE_SPEC
    # Support for saving registered TypeSpecs is currently experimental.
    # Issue a warning to indicate the limitations.
    warnings.warn("Encoding a StructuredValue with type %s; loading this "
                  "StructuredValue will require that this type be "
                  "imported and registered." % type_spec_class_name)

    type_state = type_spec_value._serialize()  # pylint: disable=protected-access
    num_flat_components = len(
        nest.flatten(type_spec_value._component_specs, expand_composites=True))  # pylint: disable=protected-access
    encoded_type_spec = struct_pb2.StructuredValue()
    encoded_type_spec.type_spec_value.CopyFrom(
        struct_pb2.TypeSpecProto(
            type_spec_class=type_spec_class,
            type_state=encode_fn(type_state),
            type_spec_class_name=type_spec_class_name,
            num_flat_components=num_flat_components))
    return encoded_type_spec

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the proto has a type spec field set, regardless of its class enum.
    """Returns true if `value` can be decoded into a `tf.TypeSpec`."""
    return value.HasField("type_spec_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Handles only the registered type spec kind: looks up the stored class name in the
    # registry and returns the class's deserialize result for the decoded state. Raises
    # ValueError if the name is not registered (asking the user to import its module) or if the
    # stored kind is another kind, which means it came from a newer TensorFlow version.
    """Returns the `tf.TypeSpec` encoded by the proto `value`."""
    type_spec_proto = value.type_spec_value
    type_spec_class_enum = type_spec_proto.type_spec_class
    class_name = type_spec_proto.type_spec_class_name

    if type_spec_class_enum == struct_pb2.TypeSpecProto.REGISTERED_TYPE_SPEC:
      try:
        type_spec_class = type_spec_registry.lookup(class_name)
      except ValueError as e:
        raise ValueError(
            f"The type '{class_name}' has not been registered.  It must be "
            "registered before you load this object (typically by importing "
            "its module).") from e
    else:
      raise ValueError(
          f"The type '{class_name}' is not supported by this version of "
          "TensorFlow. (The object you are loading must have been created "
          "with a newer version of TensorFlow.)")

    # pylint: disable=protected-access
    return type_spec_class._deserialize(decode_fn(type_spec_proto.type_state))


_codecs = [
    _ListCodec(),
    _TupleCodec(),
    _NamedTupleCodec(),
    _StringCodec(),
    _Float64Codec(),
    _NoneCodec(),
    _Int64Codec(),
    _BoolCodec(),
    _DictCodec(),
    _TypeSpecCodec(),
    _TensorTypeCodec(),
]
