# [MODULE DOC]
# Unit tests for the conversion helper module of the dataset utilities: conversion of optional
# arguments to tensors with defaults, and conversion of shape-like values to 1-D int64 tensors,
# including error cases. Runs the tests when executed as a program.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utilities working with user input."""

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.util import convert
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ConvertTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test class covering the conversion helpers for optional arguments and partial shapes. Uses
  # the dataset test base class and runs in the default (eager and graph) test combinations,
  # except where a test is restricted to graph mode.

  @combinations.generate(test_base.default_test_combinations())
  def testInteger(self):
    # [FUNCTION DOC]
    # Test: a provided integer value is converted to a tensor and evaluates to the same value.
    resp = convert.optional_param_to_tensor("foo", 3)
    self.assertEqual(3, self.evaluate(resp))

  @combinations.generate(test_base.default_test_combinations())
  def testIntegerDefault(self):
    # [FUNCTION DOC]
    # Test: with a null value and no explicit default, the result is the integer zero.
    resp = convert.optional_param_to_tensor("foo", None)
    self.assertEqual(0, self.evaluate(resp))

  @combinations.generate(test_base.default_test_combinations())
  def testStringDefault(self):
    # [FUNCTION DOC]
    # Test: with a null value and a string type, the supplied default string is used; asserts
    # the evaluated tensor equals the default as bytes.
    resp = convert.optional_param_to_tensor("bar", None, "default",
                                            dtypes.string)
    self.assertEqual(compat.as_bytes("default"), self.evaluate(resp))

  @combinations.generate(test_base.default_test_combinations())
  def testString(self):
    # [FUNCTION DOC]
    # Test: a provided string value is used instead of the string default; asserts the evaluated
    # tensor equals the provided value as bytes.
    resp = convert.optional_param_to_tensor("bar", "value", "default",
                                            dtypes.string)
    self.assertEqual(compat.as_bytes("value"), self.evaluate(resp))

  @combinations.generate(test_base.default_test_combinations())
  def testPartialShapeToTensorKnownDimension(self):
    # [FUNCTION DOC]
    # Test: a one-dimensional known shape (size one) given as shape object, tuple, list or int64
    # tensor converts to a tensor holding that single value.
    self.assertAllEqual([1],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                tensor_shape.TensorShape([1]))))
    self.assertAllEqual([1], self.evaluate(
        convert.partial_shape_to_tensor((1,))))
    self.assertAllEqual([1], self.evaluate(
        convert.partial_shape_to_tensor([1])))
    self.assertAllEqual([1],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                constant_op.constant([1], dtype=dtypes.int64))))

  @combinations.generate(test_base.graph_only_combinations())
  def testPartialShapeToTensorUnknownDimension(self):
    # [FUNCTION DOC]
    # Test (graph mode only): a one-dimensional shape with unknown size in various input forms
    # converts to a tensor holding minus one. Also asserts a value error when the tensor input is
    # 2-D (message about needing a 1-D int64 tensor, shape (2, 2)) and a type error when the
    # tensor has float32 elements.
    self.assertAllEqual([-1],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                tensor_shape.TensorShape([None]))))
    self.assertAllEqual([-1],
                        self.evaluate(convert.partial_shape_to_tensor((None,))))
    self.assertAllEqual([-1],
                        self.evaluate(convert.partial_shape_to_tensor([None])))
    self.assertAllEqual([-1],
                        self.evaluate(convert.partial_shape_to_tensor([-1])))
    self.assertAllEqual([-1],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                constant_op.constant([-1],
                                                     dtype=dtypes.int64))))

    with self.assertRaisesRegex(
        ValueError, r"The given shape .* must be a 1-D tensor of `tf.int64` "
        r"values, but the shape was \(2, 2\)."):
      convert.partial_shape_to_tensor(constant_op.constant(
          [[1, 1], [1, 1]], dtype=dtypes.int64))

    with self.assertRaisesRegex(
        TypeError, r"The given shape .* must be a 1-D tensor of `tf.int64` "
        r"values, but the element type was float32."):
      convert.partial_shape_to_tensor(constant_op.constant([1., 1.]))

  @combinations.generate(test_base.default_test_combinations())
  def testPartialShapeToTensorMultipleDimensions(self):
    # [FUNCTION DOC]
    # Test: multi-dimensional shapes given as shape object, tuple, list or int64 tensor convert
    # to the matching 1-D tensor. Covers fully known (3, 6), partly unknown (3, unknown) and
    # fully unknown dimensions, where unknown (or minus one) becomes minus one.
    self.assertAllEqual([3, 6],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                tensor_shape.TensorShape([3, 6]))))
    self.assertAllEqual([3, 6],
                        self.evaluate(convert.partial_shape_to_tensor((3, 6))))
    self.assertAllEqual([3, 6],
                        self.evaluate(convert.partial_shape_to_tensor([3, 6])))
    self.assertAllEqual([3, 6],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                constant_op.constant([3, 6],
                                                     dtype=dtypes.int64))))

    self.assertAllEqual([3, -1],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                tensor_shape.TensorShape([3, None]))))
    self.assertAllEqual([3, -1],
                        self.evaluate(
                            convert.partial_shape_to_tensor((3, None))))
    self.assertAllEqual([3, -1],
                        self.evaluate(
                            convert.partial_shape_to_tensor([3, None])))
    self.assertAllEqual([3, -1],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                constant_op.constant([3, -1],
                                                     dtype=dtypes.int64))))

    self.assertAllEqual([-1, -1],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                tensor_shape.TensorShape([None, None]))))
    self.assertAllEqual([-1, -1],
                        self.evaluate(
                            convert.partial_shape_to_tensor((None, None))))
    self.assertAllEqual([-1, -1],
                        self.evaluate(
                            convert.partial_shape_to_tensor([None, None])))
    self.assertAllEqual([-1, -1],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                constant_op.constant([-1, -1],
                                                     dtype=dtypes.int64))))

  @combinations.generate(test_base.default_test_combinations())
  def testPartialShapeToTensorScalar(self):
    # [FUNCTION DOC]
    # Test: empty shapes (empty shape object, tuple, list, empty int64 tensor) must all convert
    # to an empty 1-D tensor; asserts each result equals an empty list.
    self.assertAllEqual([],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                tensor_shape.TensorShape([]))))
    self.assertAllEqual([], self.evaluate(convert.partial_shape_to_tensor(())))
    self.assertAllEqual([], self.evaluate(convert.partial_shape_to_tensor([])))
    self.assertAllEqual([],
                        self.evaluate(
                            convert.partial_shape_to_tensor(
                                constant_op.constant([], dtype=dtypes.int64))))


if __name__ == "__main__":
  test.main()
