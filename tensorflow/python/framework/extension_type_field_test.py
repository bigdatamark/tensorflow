# [MODULE DOC]
# Unit tests for the extension type field module: construction and representation of field
# records, reserved-name detection, validation of supported/unsupported type annotations, and
# conversion of values (and type-specification values) to declared field types, including
# error messages. Tests run in graph and eager modes where decorated. Both the typing-module
# generics and the built-in generic tuple forms are exercised (the latter on Python 3.9+).
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
"""Tests for tf.framework.extension_type_field."""

import sys
import typing
from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


if sys.version_info >= (3, 9):
  _TUPLE = tuple
else:
  # Remove this branch once TF drops support for Python < 3.9.
  _TUPLE = typing.Tuple


@test_util.run_all_in_graph_and_eager_modes
class ExtensionTypeFieldTest(test_util.TensorFlowTestCase,
                             parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the field record class: constructing fields with and without defaults, error
  # reporting for bad annotations or defaults, printed form, and reserved-name checks. Runs
  # every test in both graph and eager modes.

  @parameterized.parameters([
      # Without default values:
      ('x', int),
      ('f', float),
      ('t', tensor.Tensor),
      # With default values:
      ('x', int, 33),
      ('y', float, 33.8),
      ('t', tensor.Tensor, [[1, 2], [3, 4]]),
      ('t', tensor.Tensor, lambda: constant_op.constant([[1, 2], [3, 4]])),
      ('r', ragged_tensor.RaggedTensor,
       lambda: ragged_factory_ops.constant([[1, 2], [3]])),
      ('seq', typing.Tuple[typing.Union[int, float], ...], (33, 12.8, 9, 0)),
      ('seq', typing.Tuple[typing.Union[int, float],
                           ...], [33, 12.8, 9, 0], (33, 12.8, 9, 0)),
      ('seq', _TUPLE[typing.Union[int, float], ...], (33, 12.8, 9, 0)),
      ('seq', _TUPLE[typing.Union[int, float], ...], (33, 12.8, 9, 0)),
      ('s', tensor_shape.TensorShape, [1, 2], tensor_shape.TensorShape([1, 2])),
      ('dtype', dtypes.DType, np.int32, dtypes.int32),
  ])
  def testConstruction(
      self,
      name,
      value_type,
      default=extension_type_field.ExtensionTypeField.NO_DEFAULT,
      converted_default=None):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: field name, declared type, optional default (a deferred
    # builder is called first when callable, so tensors are created inside the test), and an
    # optional expected converted default. Builds a field and asserts its name, declared type
    # and default (compared as array-like for tensors, otherwise by equality; the converted
    # default is expected when given, e.g. lists turned into tuples, shapes, dtypes).
    if callable(default):
      default = default()  # deferred construction (contains tensor)
    field = extension_type_field.ExtensionTypeField(name, value_type, default)
    if converted_default is not None:
      default = converted_default
    self.assertEqual(field.name, name)
    self.assertEqual(field.value_type, value_type)
    if isinstance(default, (tensor.Tensor, ragged_tensor.RaggedTensor)):
      self.assertAllEqual(field.default, default)
    else:
      self.assertEqual(field.default, default)

  @parameterized.parameters([
      ('i', int, 8.3, "default value for i: expected 'int', got 'float'"),
      ('f', float, 8, "default value for f: expected 'float', got 'int'"),
      ('x', int, 'hello world',
       "default value for x: expected 'int', got 'str'"),
      ('seq', typing.Tuple[typing.Union[int, float], ...], [33, 12.8, 'zero'],
       (r'default value for seq\[2\]: expected '
        r"typing.Union\[int, float\], got 'str'")),
      ('seq', _TUPLE[typing.Union[int, float], ...], [33, 12.8, 'zero'],
       (r'default value for seq\[2\]: expected '
        r"typing.Union\[int, float\], got 'str'")),
      ('t', tensor.TensorSpec(None, dtypes.int32),
       lambda: constant_op.constant(0.0),
       'Unsupported type annotation TensorSpec.*'),
      ('x', dict, {}, "In field 'x': Unsupported type annotation 'dict'"),
      ('y', typing.Union[int, list], 3,
       "In field 'y': Unsupported type annotation 'list'"),
      ('z', typing.Mapping[tensor.Tensor, int], {},
       "In field 'z': Mapping had a key 'Tensor' with type 'type'"),
  ])
  def testConstructionError(self, name, value_type, default, error):
    # [FUNCTION DOC]
    # Parameterized test of failure cases. Inputs: name, type, default (possibly a deferred
    # builder) and an expected error message pattern. Asserts that building the field raises
    # a type error matching the pattern: wrong-typed defaults, unsupported annotations such as
    # dict, list inside a union, specs, and tensor-typed mapping keys.
    if callable(default):
      default = default()  # deferred construction (contains tensor)
    with self.assertRaisesRegex(TypeError, error):
      extension_type_field.ExtensionTypeField(name, value_type, default)

  @parameterized.parameters([
      ("ExtensionTypeField(name='i', value_type=<class 'int'>, "
       'default=ExtensionTypeField.NO_DEFAULT)', 'i', int),
      ("ExtensionTypeField(name='x', value_type=typing.Tuple"
       '[typing.Union[int, str], ...], default=ExtensionTypeField.NO_DEFAULT)',
       'x', typing.Tuple[typing.Union[int, str], ...]),
      ("ExtensionTypeField(name='j', value_type=<class 'int'>, default=3)", 'j',
       int, 3),
  ])
  def testRepr(self,
               expected,
               name,
               value_type,
               default=extension_type_field.ExtensionTypeField.NO_DEFAULT):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: expected printed text, field name, type and optional default.
    # Builds the field and asserts that its printed form equals the expected text, including the
    # no-default marker when no default is given.
    field = extension_type_field.ExtensionTypeField(name, value_type, default)
    self.assertEqual(repr(field), expected)

  @parameterized.parameters([
      ('Spec', True),
      ('_type_spec', True),
      ('self', True),
      ('x', False),
      ('_tf_extension_type_foo_bar', True),
  ])
  def testIsReservedName(self, name, expected):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: a candidate name and the expected boolean. Asserts that the
    # reserved-name check agrees: Spec, a base-class attribute, self and names with the reserved
    # prefix are reserved; an ordinary name is not.
    self.assertEqual(
        extension_type_field.ExtensionTypeField.is_reserved_name(name),
        expected)


class ValidateFieldPyTypeTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the type annotation validator: a list of supported annotations that must be
  # accepted and a list of unsupported ones that must be rejected with specific messages.

  @parameterized.parameters([
      # Simple types
      dict(tp=int),
      dict(tp=float),
      dict(tp=str),
      dict(tp=bytes),
      dict(tp=bool),
      dict(tp=None),
      dict(tp=type(None)),
      dict(tp=dtypes.DType),
      dict(tp=tensor_shape.TensorShape),
      dict(tp=tensor.Tensor),
      dict(tp='A', allow_forward_references=True),
      # Generic types
      dict(tp=typing.Union[int, float]),
      dict(tp=typing.Tuple[int, ...]),
      dict(tp=typing.Tuple[int, int]),
      dict(tp=_TUPLE[int, ...]),
      dict(tp=_TUPLE[int, int]),
      dict(tp=typing.Mapping[int, int]),
      dict(tp=typing.Mapping[str, int]),
      dict(tp=typing.Union[int, 'A'], allow_forward_references=True),
      dict(tp=typing.Mapping['A', int], allow_forward_references=True),
      dict(tp=typing.Union[int, typing.Tuple[typing.Tuple[int, int], ...]]),
      dict(tp=typing.Union[int, _TUPLE[_TUPLE[int, int], ...]]),
      dict(tp=typing.Union[int, _TUPLE[typing.Tuple[int, int], ...]]),
      dict(tp=typing.Union[int, typing.Tuple[_TUPLE[int, int], ...]]),
  ])
  def testValidPytype(self, tp, allow_forward_references=False):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: an annotation and whether forward references are allowed
    # (default no). Asserts that validation completes without raising for scalars, dtype,
    # tensor, shape, unions, tuples, mappings and nested combinations.
    extension_type_field.validate_field_value_type(
        tp, allow_forward_references=allow_forward_references)

  @parameterized.parameters([
      dict(tp=dict, error="Unsupported type annotation 'dict'"),
      dict(tp=list, error="Unsupported type annotation 'list'"),
      dict(
          tp=typing.Union[int, list],
          error="Unsupported type annotation 'list'"),
      dict(
          tp=typing.Tuple[typing.Tuple[int, int, dict], ...],
          error="Unsupported type annotation 'dict'"),
      dict(
          tp=_TUPLE[_TUPLE[int, int, dict], ...],
          error="Unsupported type annotation 'dict'"),
      dict(tp='A', error='Unresolved forward reference .*'),
      dict(tp=typing.Union[int, 'A'], error='Unresolved forward reference .*'),
      dict(tp=typing.Mapping[tensor.Tensor, int],
           error="Mapping had a key 'Tensor' with type 'type'"),
      dict(
          tp=typing.Mapping[tensor_shape.TensorShape, int],
          error="Mapping had a key 'TensorShape' with type 'ABCMeta'"),
  ])
  def testInvalidPytype(self, tp, error):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: an annotation and an expected message pattern. Asserts that
    # validation raises a type error matching the pattern: unsupported classes, nested
    # unsupported members, unresolved forward references, and tensor-like mapping keys.
    with self.assertRaisesRegex(TypeError, error):
      extension_type_field.validate_field_value_type(tp)


class FieldValueConverterTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the value conversion routines: mismatched field sets, conversion of single values
  # in value and spec contexts, conversion errors, and in-place conversion of whole field
  # mappings.

  @parameterized.parameters([
      ({
          'x': 1
      }, "Missing required fields: {'y'}"),
      ({
          'x': 1,
          'y': 2.0,
          'z': 3
      }, "Got unexpected fields: {'z'}"),
  ])
  def testConvertFieldsMismatch(self, field_values, error):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: a value mapping and an expected message. With two declared
    # fields (an integer and a float), asserts that converting a mapping with a missing or an
    # extra key raises a value error naming the missing or unexpected field.
    fields = [
        extension_type_field.ExtensionTypeField('x', int),
        extension_type_field.ExtensionTypeField('y', float)
    ]
    with self.assertRaisesRegex(ValueError, error):
      extension_type_field.convert_fields(fields, field_values)

  @parameterized.parameters([
      (12, int),
      (5.3, float),
      ('foo', str),
      (None, None),
      (True, bool),
      ([1, 2, 3], tensor.Tensor),
      (lambda: constant_op.constant([1, 2, 3]), tensor.Tensor),
      (lambda: ragged_factory_ops.constant([[1, 2], [3]]),
       ragged_tensor.RaggedTensor),
      ([1, 2, 3], typing.Tuple[int, ...], (1, 2, 3)),
      ((1, 2, 3), typing.Tuple[int, int, int], (1, 2, 3)),
      ([1, 2, 3], _TUPLE[int, ...], (1, 2, 3)),
      ((1, 2, 3), _TUPLE[int, int, int], (1, 2, 3)),
      ({
          'a': 12
      }, typing.Mapping[str, int]),
      ({
          'a': (12, 3.0)
      }, typing.Mapping[str, typing.Tuple[int, float]]),
      ({
          'a': (12, 3.0)
      }, typing.Mapping[str, _TUPLE[int, float]]),
      (tensor_shape.TensorShape([1, 2]), tensor_shape.TensorShape,
       tensor_shape.TensorShape([1, 2])),
      ([1, 2], tensor_shape.TensorShape, tensor_shape.TensorShape([1, 2])),
      (dtypes.int32, dtypes.DType, dtypes.int32),
      (np.int32, dtypes.DType, dtypes.int32),
  ])
  def testConvertValue(self, value, value_type, expected=None):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: a value (or deferred builder), declared type, and optional
    # expected result (defaults to the input). Converts in the default value context and
    # asserts the result equals the expectation (array-like comparison for tensors): lists
    # become tuples, shapes and dtypes are normalized, scalars are unchanged.
    if callable(value):
      value = value()  # deferred construction (contains tensor)
    if expected is None:
      expected = value
    converted = extension_type_field._convert_value(value, value_type, ('x',))
    if isinstance(converted, (tensor.Tensor, ragged_tensor.RaggedTensor)):
      self.assertAllEqual(converted, expected)
    else:
      self.assertEqual(converted, expected)

  @parameterized.parameters([
      (12, int),
      (5.3, float),
      ('foo', str),
      (None, None),
      (True, bool),
      (tensor.TensorSpec([5]), tensor.Tensor),
      (ragged_tensor.RaggedTensorSpec([5, None]), ragged_tensor.RaggedTensor),
      ([1, 2, 3], typing.Tuple[int, ...], (1, 2, 3)),
      ((1, 2, 3), typing.Tuple[int, int, int], (1, 2, 3)),
      ([1, 2, 3], _TUPLE[int, ...], (1, 2, 3)),
      ((1, 2, 3), _TUPLE[int, int, int], (1, 2, 3)),
      ({
          'a': 12
      }, typing.Mapping[str, int]),
      ({
          'a': (12, 3.0)
      }, typing.Mapping[str, typing.Tuple[int, float]]),
      ({
          'a': (12, 3.0)
      }, typing.Mapping[str, _TUPLE[int, float]]),
      (tensor_shape.TensorShape([1, 2]), tensor_shape.TensorShape,
       tensor_shape.TensorShape([1, 2])),
      ([1, 2], tensor_shape.TensorShape, tensor_shape.TensorShape([1, 2])),
      (dtypes.int32, dtypes.DType, dtypes.int32),
      (np.int32, dtypes.DType, dtypes.int32),
  ])
  def testConvertValueForSpec(self, value, value_type, expected=None):
    # [FUNCTION DOC]
    # Parameterized test like the value conversion test but in the type-specification context.
    # Inputs: value, declared type, optional expected result. Tensor-like fields are given
    # specs rather than values. Asserts the converted result equals the expectation.
    if callable(value):
      value = value()  # deferred construction (contains tensor)
    if expected is None:
      expected = value
    converted = extension_type_field._convert_value(
        value, value_type, ('x',),
        extension_type_field._ConversionContext.SPEC)
    if isinstance(converted, (tensor.Tensor, ragged_tensor.RaggedTensor)):
      self.assertAllEqual(converted, expected)
    else:
      self.assertEqual(converted, expected)

  @parameterized.parameters([
      (12.3, int, "x: expected 'int', got 'float'"),
      (12, float, "x: expected 'float', got 'int'"),
      ([1, 2, 3.0], typing.Tuple[int, ...],
       r"x\[2\]: expected 'int', got 'float'"),
      ([1, 2, 3.0], _TUPLE[int, ...],
       r"x\[2\]: expected 'int', got 'float'"),
      ('foo', tensor_shape.TensorShape,
       "x: expected 'tf.TensorShape', got 'str'"),
      ('foo', dtypes.DType, "x: expected 'tf.DType', got 'str'"),
  ])
  def testConvertValueError(self, value, value_type, error):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: a value, declared type and expected message pattern. Asserts
    # that converting raises a type error whose message names the path and the expected and
    # actual type, including indexed paths for tuple elements, shapes and dtypes.
    if callable(value):
      value = value()  # deferred construction (contains tensor)
    with self.assertRaisesRegex(TypeError, error):
      extension_type_field._convert_value(value, value_type, ('x',))

  def testConvertFields(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the field list declares the name y twice (once with the typing generic tuple
    # and once with the built-in one), apparently a copy and paste artifact; the extra entry
    # only passes because the mismatch reporter does not raise when the name sets agree.
    # Converts a mapping with an integer, a list (to a tuple of ints/bools) and a nested list
    # (to a tensor) in place, then asserts the key set is unchanged, values are converted
    # as expected and the third entry is a tensor with the original contents.
    fields = [
        extension_type_field.ExtensionTypeField('x', int),
        extension_type_field.ExtensionTypeField(
            'y', typing.Tuple[typing.Union[int, bool], ...]),
        extension_type_field.ExtensionTypeField(
            'y', _TUPLE[typing.Union[int, bool], ...]),
        extension_type_field.ExtensionTypeField('z', tensor.Tensor)
    ]
    field_values = {'x': 1, 'y': [1, True, 3], 'z': [[1, 2], [3, 4], [5, 6]]}
    extension_type_field.convert_fields(fields, field_values)
    self.assertEqual(set(field_values), set(['x', 'y', 'z']))
    self.assertEqual(field_values['x'], 1)
    self.assertEqual(field_values['y'], (1, True, 3))
    self.assertIsInstance(field_values['z'], tensor.Tensor)
    self.assertAllEqual(field_values['z'], [[1, 2], [3, 4], [5, 6]])

  def testConvertFieldsForSpec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the field list declares the name y twice, as in the value conversion test.
    # Converts a mapping in the type-specification context in place, where the tensor field
    # holds a tensor spec. Asserts the key set is unchanged, the integer is kept, the list
    # becomes a tuple, and the spec is left equal to the original spec.
    fields = [
        extension_type_field.ExtensionTypeField('x', int),
        extension_type_field.ExtensionTypeField(
            'y', typing.Tuple[typing.Union[int, bool], ...]),
        extension_type_field.ExtensionTypeField(
            'y', _TUPLE[typing.Union[int, bool], ...]),
        extension_type_field.ExtensionTypeField('z', tensor.Tensor)
    ]
    field_values = {
        'x': 1,
        'y': [1, True, 3],
        'z': tensor.TensorSpec([5, 3])
    }
    extension_type_field.convert_fields_for_spec(fields, field_values)
    self.assertEqual(set(field_values), set(['x', 'y', 'z']))
    self.assertEqual(field_values['x'], 1)
    self.assertEqual(field_values['y'], (1, True, 3))
    self.assertEqual(field_values['z'], tensor.TensorSpec([5, 3]))


if __name__ == '__main__':
  googletest.main()
