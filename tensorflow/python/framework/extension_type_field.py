# [MODULE DOC]
# Field metadata and value validation/conversion support for user-defined extension types.
# Defines the list of reserved field names, a small sentinel class, the field description
# record (name, declared type, default) and the routines that check that a declared type
# annotation is supported and that convert or type-check concrete field values (or
# type-specification values) against those annotations. Supported annotations are basic
# scalars, dtypes, tensors, tensor shapes, composite tensors, generic tuples, mappings and
# unions of those. Used by the extension type machinery elsewhere in the framework package.
# No import-time side effects beyond defining module-level constants.
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
"""Metadata about fields for user-defined ExtensionType classes."""

import collections
import collections.abc
import enum
import typing

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.util import type_annotations

# These names may not be used as the name for a ExtensionType field (to prevent
# name clashes).  All names beginning with `'_tf_extension_type'` are also
# reserved.
RESERVED_FIELD_NAMES = [
    'self',
    # Name of the nested TypeSpec class.
    'Spec',
    # Names defined by the CompositeTensor base class.
    '_type_spec',
    '_shape_invariant_to_type_spec',
    '_consumers',
    # Names defined by the TypeSpec base class.
    'value_type',
    'is_compatible_with',
    'most_specific_compatible_type',
    '_with_tensor_ranks_only',
    '_to_components',
    '_from_components',
    '_component_specs',
    '_to_tensor_list',
    '_from_tensor_list',
    '_from_compatible_tensor_list',
    '_flat_tensor_specs',
    '_serialize',
    '_deserialize',
    '_to_legacy_output_types',
    '_to_legacy_output_shapes',
    '_to_legacy_output_classes',
    # Used by Keras
    '_keras_mask'
]


class Sentinel(object):
  # [CLASS DOC]
  # Marker object used as a unique placeholder (for example "no default given"). Each instance
  # holds a name used only for its printed form; identity comparison distinguishes it from any
  # user-supplied value.
  """Sentinel value that's not equal (w/ `is`) to any user value."""

  def __init__(self, name):
    # [FUNCTION DOC]
    # Creates a sentinel. Input: name, a string stored and later used as the printed form.
    # Returns nothing; no side effects beyond storing the name.
    self._name = name

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the stored name string as the printable form of the sentinel.
    return self._name


_NoneType = type(None)


def _issubclass(cls, clsinfo):
  # [FUNCTION DOC]
  # HELPER: safe subclass test used by the type validation and conversion routines in this module.
  # Inputs: cls, the candidate class or generic alias; clsinfo, the class (or tuple of classes) to
  # test against. Returns true if cls is a subclass, false otherwise, including the case where
  # the check itself raises a type error (for example for parameterized generic aliases).
  """Internal issubclass that doesn't raise TypeError."""
  try:
    return issubclass(cls, clsinfo)
  except TypeError:
    # issubclass with GenericAlias instances raises TypeError. For example,
    # `issubclass(tuple[int], composite_tensor.CompositeTensor)`.
    return False


# ==============================================================================
# ExtensionTypeField
# ==============================================================================
class ExtensionTypeField(
    collections.namedtuple('ExtensionTypeField',
                           ['name', 'value_type', 'default'])):
  # [CLASS DOC]
  # Immutable record (a named tuple with fields name, value_type and default) describing one
  # field of an extension type: its name, declared type annotation, and default value (or the
  # NO_DEFAULT sentinel when the field is required). Instances are built by the extension type
  # machinery from class annotations. Class attribute NO_DEFAULT is the "no default" marker.
  """Metadata about a single field in a `tf.ExtensionType` object."""

  NO_DEFAULT = Sentinel('ExtensionTypeField.NO_DEFAULT')

  def __new__(cls, name, value_type, default=NO_DEFAULT):
    # [FUNCTION DOC]
    # Creates a field record. Inputs: name, the field name string; value_type, the type
    # annotation constraining values (forward references allowed here); default, optional
    # default value, with the no-default marker meaning the field is required.
    # Validates the annotation; raises a type error (prefixed with the field name) if it is
    # unsupported. If a default is given it is converted/checked against value_type in the
    # default-value context and may raise a type error on mismatch. Returns the new record.
    """Constructs a new ExtensionTypeField containing metadata for a single field.

    Args:
      name: The name of the new field (`str`).  May not be a reserved name.
      value_type: A python type expression constraining what values this field
        can take.
      default: The default value for the new field, or `NO_DEFAULT` if this
        field has no default value.

    Returns:
      A new `ExtensionTypeField`.

    Raises:
      TypeError: If the type described by `value_type` is not currently
          supported by `tf.ExtensionType`.
      TypeError: If `default` is specified and its type does not match
        `value_type`.
    """
    try:
      validate_field_value_type(value_type, allow_forward_references=True)
    except TypeError as e:
      raise TypeError(f'In field {name!r}: {e}') from e

    if default is not cls.NO_DEFAULT:
      default = _convert_value(default, value_type,
                               (f'default value for {name}',),
                               _ConversionContext.DEFAULT)
    return super(ExtensionTypeField, cls).__new__(cls, name, value_type,
                                                  default)

  @staticmethod
  def is_reserved_name(name):
    # [FUNCTION DOC]
    # Static method. Input: name, a string. Returns true if the name is in the reserved field
    # name list or, ignoring case, begins with the reserved extension-type prefix; false otherwise.
    # No side effects.
    """Returns true if `name` is a reserved name."""
    return name in RESERVED_FIELD_NAMES or name.lower().startswith(
        '_tf_extension_type')


def validate_field_value_type(value_type,
                              in_mapping_key=False,
                              allow_forward_references=False):
  # [FUNCTION DOC]
  # Recursively validates a type annotation. Inputs: value_type, the annotation; in_mapping_key,
  # true when nested inside a mapping key (where tensor and composite-tensor types are not
  # allowed); allow_forward_references, whether string or forward-reference annotations are
  # accepted (they are then not checked further). Accepted: int, float, str, bytes, bool, none,
  # dtype, tensor, tensor shape, composite tensor subclasses, generic tuples (including the
  # variable-length form), unions and mappings (keys validated in key mode).
  # Returns nothing; raises a type error for unresolved forward references (when disallowed),
  # tensor-like mapping keys, or any unsupported annotation.
  """Checks that `value_type` contains only supported type annotations.

  Args:
    value_type: The type annotation to check.
    in_mapping_key: True if `value_type` is nested in the key of a mapping.
    allow_forward_references: If false, then raise an exception if a
      `value_type` contains a forward reference (i.e., a string literal).

  Raises:
    TypeError: If `value_type` contains an unsupported type annotation.
  """
  if isinstance(value_type, str) or type_annotations.is_forward_ref(value_type):
    if allow_forward_references:
      return
    else:
      raise TypeError(f'Unresolved forward reference {value_type!r}')

  if value_type in (int, float, str, bytes, bool, None, _NoneType,
                    dtypes.DType):
    return
  elif (value_type in (tensor.Tensor, tensor_shape.TensorShape) or
        (isinstance(value_type, type) and
         _issubclass(value_type, composite_tensor.CompositeTensor))):
    if in_mapping_key:
      raise TypeError(f'Mapping had a key {value_type.__name__!r} with type '
                      f'{type(value_type).__name__!r}')
  elif (type_annotations.is_generic_tuple(value_type) or
        type_annotations.is_generic_union(value_type)):
    type_args = type_annotations.get_generic_type_args(value_type)
    if (len(type_args) == 2 and type_args[1] is Ellipsis and
        type_annotations.is_generic_tuple(value_type)):  # `Tuple[X, ...]`
      validate_field_value_type(type_args[0], in_mapping_key,
                                allow_forward_references)
    else:
      for arg in type_annotations.get_generic_type_args(value_type):
        validate_field_value_type(arg, in_mapping_key, allow_forward_references)
  elif type_annotations.is_generic_mapping(value_type):
    key_type, value_type = type_annotations.get_generic_type_args(value_type)
    validate_field_value_type(key_type, True, allow_forward_references)
    validate_field_value_type(value_type, in_mapping_key,
                              allow_forward_references)
  elif isinstance(value_type, type):
    raise TypeError(f'Unsupported type annotation {value_type.__name__!r}')
  else:
    raise TypeError(f'Unsupported type annotation {value_type!r}')


# ==============================================================================
# Type-checking & conversion for ExtensionTypeField values
# ==============================================================================


class _ConversionContext(enum.Enum):
  # [CLASS DOC]
  # HELPER: context flag for the value conversion routines in this module.
  # Enumeration of what kind of value is being converted: an extension type field value, a
  # type-specification field value (tensor-like fields hold specs), or a default value for the
  # constructor (tensor-like values are left unconverted).
  """Enum to indicate what kind of value is being converted.

  Used by `_convert_fields` and `_convert_value` and their helper methods.
  """
  VALUE = 1  # Converting an ExtensionType field
  SPEC = 2  # Converting an ExtensionType.Spec field
  DEFAULT = 3  # Converting a default value for __init__


def convert_fields(fields, field_values):
  # [FUNCTION DOC]
  # Public entry point for converting real field values. Inputs: fields, list of field records;
  # field_values, mutable mapping from field names to values containing exactly the declared
  # names. Converts each value to its declared type in the value context and replaces the
  # entries in field_values in place. Returns nothing. Raises a value error if the key set does
  # not match the field names, and a type error if a value cannot be converted.
  """Type-checks and converts each field in `field_values` (in place).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.

  Raises:
    ValueError: If the keys of `field_values` do not match the names of
      the fields in `fields`.
    TypeError: If any value in `field_values` does not have the type indicated
      by the corresponding `ExtensionTypeField` object.
  """
  _convert_fields(fields, field_values, context=_ConversionContext.VALUE)


def convert_fields_for_spec(fields, field_values):
  # [FUNCTION DOC]
  # Same as the public value conversion but in the type-specification context: for tensor and
  # composite-tensor fields the supplied value must itself be a type specification of the
  # right kind. Inputs: fields (list of field records) and field_values (mutable mapping from
  # names to values, with exactly the declared names). Updates field_values in place; returns
  # nothing. Raises a value error on key mismatch and a type error on a wrongly typed value.
  """Type-checks and converts field values for a TypeSpec (in place).

  This is similar to `convert_fields`, except that we expect a `TypeSpec` for
  tensor-like types.  In particular, if the `value_type` of a field is
  `tf.Tensor` or a `CompositeTensor` subclass, then the corresponding value in
  `fields` is expected to contain a `TypeSpec` (rather than a value described by
  that `TypeSpec`).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.

  Raises:
    ValueError: If the keys of `field_values` do not match the names of
      the fields in `fields`.
    TypeError: If any value in `field_values` does not have the type indicated
      by the corresponding `ExtensionTypeField` object.
  """
  _convert_fields(fields, field_values, context=_ConversionContext.SPEC)


def _convert_fields(fields, field_values, context):
  # [FUNCTION DOC]
  # HELPER: shared implementation of the two public field conversion entry points.
  # Inputs: fields, list of field records; field_values, mutable mapping from names to values;
  # context, the conversion context. Checks the number of values and each name against the
  # fields (reporting mismatches via the mismatch helper), converts every value into a new
  # temporary mapping, then updates field_values in place with the converted values (so a
  # failure leaves it unchanged). Returns nothing; raises value error or type error.
  """Type-checks and converts each field in `field_values` (in place).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.
    context: _ConversionContext, indicates what kind of value we are converting.

  Raises:
    ValueError: If the keys of `field_values` do not match the names of
      the fields in `fields`.
    TypeError: If any value in `field_values` does not have the type indicated
      by the corresponding `ExtensionTypeField` object.
  """
  converted = {}
  if len(fields) != len(field_values):
    _report_field_mismatches(fields, field_values)
  for field in fields:
    if field.name not in field_values:
      _report_field_mismatches(fields, field_values)
    field_value = field_values[field.name]
    converted[field.name] = _convert_value(field_value, field.value_type,
                                           (field.name,), context)
  field_values.update(converted)


def _convert_value(value, expected_type, path,
                   context=_ConversionContext.VALUE):
  # [FUNCTION DOC]
  # HELPER: recursive dispatcher that converts one value to an expected type annotation.
  # Inputs: value; expected_type, the annotation (none means the none type); path, tuple of
  # strings naming the value location for error messages (must be a tuple, asserted);
  # context, conversion context (default: value). Dispatches to tensor, composite tensor,
  # tensor shape, dtype, plain scalar, tuple, mapping or union handling. Plain scalar types
  # are checked with an instance test only (no coercion). Returns the converted value.
  # Raises a type error describing the path when conversion is impossible or the annotation is
  # unsupported.
  """Type-checks and converts a value.

  Args:
    value: The value to type-check.
    expected_type: The expected type for the value.
    path: Tuple of `str` naming the value (used for exception messages).
    context: _ConversionContext, indicates what kind of value we are converting.

  Returns:
    A copy of `value`, converted to the expected type.

  Raises:
    TypeError: If `value` can not be converted to the expected type.
  """
  assert isinstance(path, tuple)

  if expected_type is None:
    expected_type = _NoneType

  if expected_type is tensor.Tensor:
    return _convert_tensor(value, path, context)
  elif (isinstance(expected_type, type) and
        _issubclass(expected_type, composite_tensor.CompositeTensor)):
    return _convert_composite_tensor(value, expected_type, path, context)
  elif expected_type is tensor_shape.TensorShape:
    try:
      return tensor_shape.as_shape(value)
    except TypeError as e:
      raise TypeError(f"{''.join(path)}: expected 'tf.TensorShape', got "
                      f'{type(value).__name__!r}') from e
  elif expected_type is dtypes.DType:
    try:
      return dtypes.as_dtype(value)
    except TypeError as e:
      raise TypeError(f"{''.join(path)}: expected 'tf.DType', got "
                      f'{type(value).__name__!r}') from e
  elif expected_type in (int, float, bool, str, bytes, _NoneType):
    if not isinstance(value, expected_type):
      raise TypeError(f'{"".join(path)}: expected {expected_type.__name__!r}, '
                      f'got {type(value).__name__!r}')
    return value
  elif type_annotations.is_generic_tuple(expected_type):
    return _convert_tuple(value, expected_type, path, context)
  elif type_annotations.is_generic_mapping(expected_type):
    return _convert_mapping(value, expected_type, path, context)
  elif type_annotations.is_generic_union(expected_type):
    return _convert_union(value, expected_type, path, context)
  else:
    raise TypeError(f'{"".join(path)}: Unsupported type annotation '
                    f'{expected_type!r}')


def _convert_tensor(value, path, context):
  # [FUNCTION DOC]
  # HELPER: tensor-typed branch of the value conversion.
  # KNOWN ISSUE: the docstring only mentions conversion, but spec and default contexts return
  # the value without producing a tensor.
  # Inputs: value, path (tuple of strings
  # for messages), context. In the spec context the value must be a type specification whose
  # value type is a tensor, and is returned unchanged. Otherwise a non-tensor value is converted
  # to a tensor, except in the default-value context where it is returned unconverted.
  # Raises a type error if the spec is wrong or tensor conversion fails.
  """Converts `value` to a `Tensor`."""
  if context == _ConversionContext.SPEC:
    if not (isinstance(value, type_spec.TypeSpec) and
            value.value_type is tensor.Tensor):
      raise TypeError(
          f'{"".join(path)}: expected a TensorSpec, got '
          f'{type(value).__name__!r}')
    return value

  if not isinstance(value, tensor.Tensor):
    if context == _ConversionContext.DEFAULT:
      # TODO(edloper): Convert the value to a numpy array?  (Note: we can't just
      # use `np.array(value)`, since the default dtypes for TF and numpy are
      # different -- e.g., int->np.int64 but int->tf.int32.
      return value
    try:
      value = ops.convert_to_tensor(value)
    except (ValueError, TypeError) as e:
      raise TypeError(f'{"".join(path)}: expected a Tensor, '
                      f'got {type(value).__name__!r}') from e
  return value


def _convert_composite_tensor(value, expected_type, path, context):
  # [FUNCTION DOC]
  # HELPER: composite-tensor-typed branch of the value conversion. Inputs: value;
  # expected_type, a composite tensor subclass; path; context. In the spec context the value
  # must be a type specification whose value type is a subclass of expected_type. Otherwise
  # the value must be an instance of expected_type. No coercion happens; the same value is
  # returned. Raises a type error naming the path when the check fails.
  """Converts `value` to a value of type `expected_type`."""
  if context == _ConversionContext.SPEC:
    if not (isinstance(value, type_spec.TypeSpec) and
            _issubclass(value.value_type, expected_type)):
      raise TypeError(f'{"".join(path)}: expected a TypeSpec for '
                      f'{expected_type.__name__!r}, got '
                      f'{type(value).__name__!r}')
    return value

  if not isinstance(value, expected_type):
    raise TypeError(f'{"".join(path)}: expected {expected_type.__name__!r}, '
                    f'got {type(value).__name__!r}')
  return value


def _convert_tuple(value, expected_type, path, context):
  # [FUNCTION DOC]
  # HELPER: tuple-typed branch of the value conversion.
  # KNOWN ISSUE: the length-mismatch message reports the type name of the value rather than its
  # actual length and ends with a stray closing parenthesis.
  # Inputs: value, a sequence; expected_type,
  # a generic tuple annotation (fixed-length or variable-length form); path; context.
  # Each element is converted recursively with an index appended to the path; returns a new
  # tuple. Raises a type error if value is not a sequence or, for the fixed-length form, if the
  # length differs.
  """Converts `value` to a tuple with type `expected_type`."""
  if not isinstance(value, typing.Sequence):
    raise TypeError(f'{"".join(path)}: expected tuple, got '
                    f'{type(value).__name__!r}')
  element_types = type_annotations.get_generic_type_args(expected_type)
  if len(element_types) == 2 and element_types[1] is Ellipsis:
    return tuple([
        _convert_value(v, element_types[0], path + (f'[{i}]',), context)
        for (i, v) in enumerate(value)
    ])
  else:
    if len(value) != len(element_types):
      raise TypeError(f'{"".join(path)}: expected tuple with length '
                      f'{len(element_types)}, got {type(value).__name__!r})')
    return tuple([
        _convert_value(v, t, path + (f'[{i}]',), context)
        for (i, (v, t)) in enumerate(zip(value, element_types))
    ])


def _convert_mapping(value, expected_type, path, context):
  # [FUNCTION DOC]
  # HELPER: mapping-typed branch of the value conversion. Inputs: value, any mapping;
  # expected_type, a generic mapping annotation giving key and value types; path; context.
  # Converts every key and value recursively and returns a new immutable dictionary.
  # Raises a type error if value is not a mapping or an entry fails conversion.
  """Converts `value` to a mapping with type `expected_type`."""
  if not isinstance(value, typing.Mapping):
    raise TypeError(f'{"".join(path)}: expected mapping, got '
                    f'{type(value).__name__!r}')
  key_type, value_type = type_annotations.get_generic_type_args(expected_type)
  return immutable_dict.ImmutableDict([
      (_convert_value(k, key_type, path + ('[<key>]',), context),
       _convert_value(v, value_type, path + (f'[{k!r}]',), context))
      for (k, v) in value.items()
  ])


def _convert_union(value, expected_type, path, context):
  # [FUNCTION DOC]
  # HELPER: union-typed branch of the value conversion. Inputs: value; expected_type, a union
  # annotation; path; context. Tries each member type in order and returns the first successful
  # conversion (earlier members win). Raises a type error naming the path if none succeeds.
  """Converts `value` to a value with any of the types in `expected_type`."""
  for type_option in type_annotations.get_generic_type_args(expected_type):
    try:
      return _convert_value(value, type_option, path, context)
    except TypeError:
      pass
  raise TypeError(f'{"".join(path)}: expected {expected_type!r}, got '
                  f'{type(value).__name__!r}')


def _report_field_mismatches(fields, field_values):
  # [FUNCTION DOC]
  # HELPER: error reporter for the field conversion routines.
  # KNOWN ISSUE: if the names match exactly nothing is raised and it silently returns, despite
  # the docstring implying an exception; callers only use it on a detected mismatch.
  # Inputs: fields, list of field
  # records; field_values, mapping of supplied values. Raises a value error: first listing
  # unexpected extra names if any, otherwise listing missing required names. Returns nothing.
  """Raises an exception with mismatches between fields and field_values."""
  expected = set(f.name for f in fields)
  actual = set(field_values)
  extra = actual - expected
  if extra:
    raise ValueError(f'Got unexpected fields: {extra}')
  missing = expected - actual
  if missing:
    raise ValueError(f'Missing required fields: {missing}')
