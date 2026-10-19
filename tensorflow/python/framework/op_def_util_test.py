# [MODULE DOC]
# Unit tests for the native op-definition utility extension: conversion of Python values to typed
# attribute values by attribute type name, and conversion of serialized attribute-value protocol
# buffers back to Python objects (including data types and tensor shapes). Contains one
# parameterized test class and runs under the standard test runner when executed as a script.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.ops.op_def_library."""

from absl.testing import parameterized

import numpy as np

from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import _op_def_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class OpDefUtilTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case covering the native op-def utility functions that convert Python
  # objects to attribute values and serialized attribute values to Python objects. It has no
  # shared fixtures; each test method is driven by tables of attribute types, inputs and expected
  # outputs.

  @parameterized.parameters([
      ("any", "Foo", "Foo"),
      ("any", 12, 12),
      ("any", {2: 3}, {2: 3}),
      ("string", "Foo", "Foo"),
      ("string", b"Foo", b"Foo"),
      ("int", 12, 12),
      ("int", 12.3, 12),
      ("float", 12, 12.0),
      ("float", 12.3, 12.3),
      ("bool", True, True),
      ("shape", tensor_shape.TensorShape([3]), tensor_shape.TensorShape([3])),
      ("shape", [3], tensor_shape.TensorShape([3])),
      ("type", dtypes.int32, dtypes.int32),
      ("type", np.int32, dtypes.int32),
      ("type", "int32", dtypes.int32),
      ("tensor", tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_FLOAT),
       tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_FLOAT)),
      ("tensor", "dtype: DT_FLOAT",
       tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_FLOAT)),
      ("list(any)", [1, "foo", 7.3, dtypes.int32],
       [1, "foo", 7.3, dtypes.int32]),
      ("list(any)", (1, "foo"), [1, "foo"]),
      ("list(string)", ["foo", "bar"], ["foo", "bar"]),
      ("list(string)", ("foo", "bar"), ["foo", "bar"]),
      ("list(string)", iter("abcd"), ["a", "b", "c", "d"]),
      ("list(int)", (1, 2.3), [1, 2]),
      ("list(float)", (1, 2.3), [1.0, 2.3]),
      ("list(bool)", [True, False], [True, False]),
      ("list(type)", [dtypes.int32, dtypes.bool], [dtypes.int32, dtypes.bool]),
      ("list(shape)", [tensor_shape.TensorShape([3]), [4, 5]],
       [tensor_shape.TensorShape([3]), tensor_shape.TensorShape([4, 5])]),
      ("list(tensor)",
       [tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_FLOAT),
        "dtype: DT_INT32"],
       [tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_FLOAT),
        tensor_pb2.TensorProto(dtype=types_pb2.DataType.DT_INT32)]),
  ])  # pyformat: disable
  def testConvert(self, attr_type, value, expected):
    # [FUNCTION DOC]
    # Parameterized over (attribute type name, input value, expected result). Converts the input
    # to the named attribute type (any, string, int, float, bool, shape, type, tensor and list
    # variants, accepting strings, numbers, tuples, iterators, dtype-like objects and text-format
    # tensors). Asserts the result equals the expected value, that its type equals the expected
    # type (for example floats stay floats and ints are truncated), and for lists that each item
    # has the expected item type. Returns nothing.
    result = _op_def_util.ConvertPyObjectToAttributeType(value, attr_type)

    # Check that we get the expected value(s).
    self.assertEqual(expected, result)

    # Check that we get the expected type(s).
    self.assertEqual(type(expected), type(result))
    if isinstance(result, list):
      for expected_item, result_item in zip(expected, result):
        self.assertEqual(type(expected_item), type(result_item))

  @parameterized.parameters([
      ("string", 12),
      ("int", "foo"),
      ("float", "foo"),
      ("bool", 1),
      ("dtype", None),
      ("shape", 12.0),
      ("tensor", [1, 2, 3]),
      ("list(any)", 12),
      ("list(int)", [1, "two"]),
      ("list(string)", [1, "two"]),
      ("tensor", "string that is not a text-formatted TensorProto"),
  ])
  def testConvertError(self, attr_type, value):
    # [FUNCTION DOC]
    # KNOWN ISSUE: one case uses the attribute type name "dtype", which is not a valid attribute
    # type name (the real name is "type"), so that case may pass because of the unknown type
    # rather than the null value.
    # Parameterized over (attribute type name, invalid value) pairs such as a number for a string,
    # text for a number, a number for a boolean, a null value, a list for a tensor, and a list
    # with a bad element. Asserts the conversion raises a type error whose message says it failed
    # to convert the value. Returns nothing.
    with self.assertRaisesRegex(TypeError, "Failed to convert value"):
      _op_def_util.ConvertPyObjectToAttributeType(value, attr_type)

  # Test AttrValueToPyObject().  Note: this test also exercises the code in
  # DataTypeToPyObject() and TensorShapeToPyObject(), since those are used
  # when the AttrValue contains a DataType or TensorShape.
  @parameterized.parameters([
      ("s: 'foo'", "foo"),
      ("i: 5", 5),
      ("f: 8", 8.0),
      ("b: True", True),
      ("type: DT_INT32", dtypes.int32),
      ("shape { dim: [{size: 3}, {size: 4}] }",
       tensor_shape.TensorShape([3, 4])),
      ("list { }", []),
      ("list { s: [] }", []),
      ("list { s: ['a', 'b', 'c'] }", ["a", "b", "c"]),
      ("list { i: [1, 2, 3] }", [1, 2, 3]),
      ("list { f: [2.0, 4.0] }", [2.0, 4.0]),
  ])  # pyformat: disable
  def testAttrValueToPyObject(self, pbtxt, expected):
    # [FUNCTION DOC]
    # Parameterized over (text-format attribute value, expected Python object). Parses the text
    # into an attribute-value message, serializes it, converts it to a Python object with the
    # native utility and asserts equality with the expected string, int, float, bool, data type,
    # tensor shape or list. Also exercises the data type and shape conversion code paths.
    # Returns nothing.
    proto = attr_value_pb2.AttrValue()
    text_format.Parse(pbtxt, proto)
    result = _op_def_util.SerializedAttrValueToPyObject(
        proto.SerializeToString())

    self.assertEqual(expected, result)

  @parameterized.parameters([
      "",                           # Empty value (oneof not set)
      "tensor {}",                  # 'TensorProto' not supported (yet).
      "func {}",                    # 'func' not supported.
      "placeholder: ''",            # 'placeholder' not supported.
      "list { tensor [{}] }",       # 'TensorProto' not supported (yet).
      "list { func [{}] }",         # 'func' not supported.
  ])  # pyformat: disable
  def testAttrValueToPyObjectError(self, pbtxt):
    # [FUNCTION DOC]
    # Parameterized over text-format attribute values that cannot be converted: an unset value,
    # tensor, function, placeholder, and lists of tensors or functions. Parses and serializes
    # each, then asserts the native conversion raises a type error or a value error. Returns
    # nothing.
    proto = attr_value_pb2.AttrValue()
    text_format.Parse(pbtxt, proto)
    with self.assertRaises((TypeError, ValueError)):
      _op_def_util.SerializedAttrValueToPyObject(proto.SerializeToString())


if __name__ == "__main__":
  googletest.main()
