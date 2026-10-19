# [MODULE DOC]
# Tests for the native tensor converter, which converts a Python value (number, tensor, nested
# list, numpy array, indexed slices) to a tensor given an optional requested dtype, returning the
# tensor, the resulting dtype enum and a flag saying whether the slower Python fallback conversion
# was needed. Tests are grouped by input kind (int, tensor, list, numpy array, indexed slices)
# and by dtype situation (inferred, explicit, incompatible). Runs in graph and eager modes;
# running the file starts the googletest runner.
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
"""Tests for tensorflow.python.framework.python_tensor_converter."""

from absl.testing import parameterized

import numpy as np

from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import _pywrap_python_tensor_converter
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class PythonTensorConverterTest(test_util.TensorFlowTestCase,
                                parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the native tensor converter. Each test creates a converter bound to the current
  # eager context and converts one kind of value with an inferred, explicit or incompatible
  # dtype. Runs in graph and eager modes; in graph mode the converter is expected to use the
  # fallback path for plain Python values.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: makes sure the eager context is initialized, then runs the base class
    # setup. Returns nothing.
    context.ensure_initialized()
    super(PythonTensorConverterTest, self).setUp()

  def makePythonTensorConverter(self):
    # [FUNCTION DOC]
    # HELPER: creates a native tensor converter bound to the current eager context. No inputs.
    # Returns the converter.
    return _pywrap_python_tensor_converter.PythonTensorConverter(
        context.context())

  #=============================================================================
  # Convert int to tensor.

  def testConvertIntWithInferredDType(self):
    # [FUNCTION DOC]
    # Scenario: convert the Python integer twelve with no requested dtype. Asserts the result is a
    # tensor equal to twelve, the dtype is 32-bit integer, and the fallback flag is set exactly
    # when not executing eagerly.
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert(12, types_pb2.DT_INVALID)
    self.assertIsInstance(result, tensor.Tensor)
    self.assertAllEqual(result, 12)
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertIntWithExplicitDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert the integer twelve requesting 64-bit integer. Asserts a tensor equal to
    # twelve, the dtype 64-bit integer, and the fallback flag set exactly when not eager.
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert(12, types_pb2.DT_INT64)
    self.assertIsInstance(result, tensor.Tensor)
    self.assertAllEqual(result, 12)
    self.assertEqual(dtype, types_pb2.DT_INT64)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertIntWithIncompatibleDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert the integer three requesting a string dtype. Asserts a type error whose
    # message says a string was expected (graph path) or that the value cannot be converted to a
    # string eager tensor (eager path).
    converter = self.makePythonTensorConverter()
    with self.assertRaisesRegex(
        TypeError, "Expected string, but got 3 of type 'int'"
        "|Cannot convert 3 to EagerTensor of dtype string"):
      converter.Convert(3, types_pb2.DT_STRING)

  #=============================================================================
  # Convert tensor to tensor.

  def testConvertTensorWithInferredDType(self):
    # [FUNCTION DOC]
    # Scenario: convert an existing 32-bit integer tensor with no requested dtype. Asserts a
    # tensor with the same values, dtype 32-bit integer, and that no fallback was used.
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert(
        constant_op.constant([1, 2, 3]), types_pb2.DT_INVALID)
    self.assertIsInstance(result, tensor.Tensor)
    self.assertAllEqual(result, [1, 2, 3])
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertFalse(used_fallback)

  def testConvertTensorWithExplicitDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert a 64-bit integer tensor requesting 64-bit integer. Asserts a tensor with
    # the same values, dtype 64-bit integer, and no fallback used.
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert(
        constant_op.constant([1, 2, 3], dtypes.int64), types_pb2.DT_INT64)
    self.assertIsInstance(result, tensor.Tensor)
    self.assertAllEqual(result, [1, 2, 3])
    self.assertEqual(dtype, types_pb2.DT_INT64)
    self.assertFalse(used_fallback)

  def testConvertTensorWithIncorrectDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert a 32-bit integer tensor requesting 64-bit integer. Asserts that a type
    # error or value error is raised (tensors are not silently cast).
    converter = self.makePythonTensorConverter()
    with self.assertRaises((TypeError, ValueError)):
      converter.Convert(
          constant_op.constant([1, 2, 3], dtypes.int32), types_pb2.DT_INT64)

  #=============================================================================
  # Convert list to tensor.

  def testConvertListWithInferredDType(self):
    # [FUNCTION DOC]
    # Scenario: convert a nested list of small integers with no requested dtype. Asserts a tensor
    # with the matching 2 by 3 values, dtype 32-bit integer, and fallback used exactly when not
    # eager.
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert([[1, 2, 3], [4, 5, 6]],
                                                     types_pb2.DT_INVALID)
    self.assertIsInstance(result, tensor.Tensor)
    self.assertAllEqual(result, [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertListWithExplicitDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert the same nested integer list requesting 64-bit integer. Asserts matching
    # values, dtype 64-bit integer, and fallback used exactly when not eager.
    converter = self.makePythonTensorConverter()
    result, dtype, used_fallback = converter.Convert([[1, 2, 3], [4, 5, 6]],
                                                     types_pb2.DT_INT64)
    self.assertIsInstance(result, tensor.Tensor)
    self.assertAllEqual(result, [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(dtype, types_pb2.DT_INT64)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertListWithIncompatibleDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert a nested integer list requesting a string dtype. Asserts a type error
    # saying a string was expected but an integer was found (graph path), or that the value cannot
    # be converted to a string eager tensor (eager path).
    converter = self.makePythonTensorConverter()
    with self.assertRaisesRegex(
        TypeError, "Expected string, but got .* of type 'int'"
        "|Cannot convert .* to EagerTensor of dtype string"):
      converter.Convert([[1, 2, 3], [4, 5, 6]], types_pb2.DT_STRING)

  def testConvertListWithInconsistentDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert a nested list mixing integers and strings with no requested dtype.
    # Asserts a type error or value error complaining about mixed types or a failed conversion.
    converter = self.makePythonTensorConverter()
    with self.assertRaisesRegex(
        (TypeError, ValueError),
        "Can't convert Python sequence with mixed types to Tensor."
        "|Failed to convert"):
      converter.Convert([[1, 2], ["a", "b"]], types_pb2.DT_INVALID)

  #=============================================================================
  # Convert np.array to tensor.

  def testConvertNumpyArrayWithInferredDType(self):
    # [FUNCTION DOC]
    # Scenario: convert a 2 by 3 numpy array of 32-bit integers with no requested dtype. Asserts
    # a tensor with the same values, dtype 32-bit integer, and fallback used exactly when not
    # eager.
    converter = self.makePythonTensorConverter()
    x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    result, dtype, used_fallback = converter.Convert(x, types_pb2.DT_INVALID)
    self.assertIsInstance(result, tensor.Tensor)
    self.assertAllEqual(result, [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertNumpyArrayWithExplicitDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert a 32-bit integer numpy array requesting 64-bit integer. Asserts the
    # values are preserved, dtype 64-bit integer, and fallback used exactly when not eager
    # (so, unlike tensors, arrays may be cast).
    converter = self.makePythonTensorConverter()
    x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    result, dtype, used_fallback = converter.Convert(x, types_pb2.DT_INT64)
    self.assertIsInstance(result, tensor.Tensor)
    self.assertAllEqual(result, [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(dtype, types_pb2.DT_INT64)
    self.assertEqual(used_fallback, not context.executing_eagerly())

  def testConvertNumpyArrayWithIncompatibleDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert a 32-bit integer numpy array requesting a string dtype. Asserts a value
    # error or type error is raised.
    converter = self.makePythonTensorConverter()
    x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    with self.assertRaises((ValueError, TypeError)):
      converter.Convert(x, types_pb2.DT_STRING)

  def testConvertNumpyArrayWithUnsupportedDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert a numpy array of generic objects (mixed numbers and strings) with no
    # requested dtype. Asserts a value error or type error is raised because the element type
    # is unsupported.
    converter = self.makePythonTensorConverter()
    x = np.array([[1, 2], ["a", "b"]], np.object_)
    with self.assertRaises((ValueError, TypeError)):
      converter.Convert(x, types_pb2.DT_INVALID)

  #=============================================================================
  # Convert IndexedSlices to tensor.

  def testConvertIndexedSlicesWithInferredDType(self):
    # [FUNCTION DOC]
    # Scenario: convert a sparse row representation (values, indices, dense shape of 3 by 3, one
    # nonzero row) with no requested dtype. Asserts the result is a dense tensor with that row
    # filled and zeros elsewhere, dtype 32-bit integer, and that the fallback was used.
    converter = self.makePythonTensorConverter()
    x = indexed_slices.IndexedSlices(
        constant_op.constant([[1, 2, 3]], dtypes.int32, name="x_values"),
        constant_op.constant([1], dtypes.int64, name="x_indices"),
        constant_op.constant([3, 3], dtypes.int64, name="x_shape"))
    result, dtype, used_fallback = converter.Convert(x, types_pb2.DT_INVALID)
    self.assertIsInstance(result, tensor.Tensor)
    self.assertAllEqual(result, [[0, 0, 0], [1, 2, 3], [0, 0, 0]])
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertTrue(used_fallback)

  def testConvertIndexedSlicesWithExplicitDtype(self):
    # [FUNCTION DOC]
    # Scenario: same sparse row value as the inferred case but requesting 32-bit integer
    # explicitly. Asserts the dense result, dtype 32-bit integer, and that the fallback was used.
    converter = self.makePythonTensorConverter()
    x = indexed_slices.IndexedSlices(
        constant_op.constant([[1, 2, 3]], dtypes.int32, name="x_values"),
        constant_op.constant([1], dtypes.int64, name="x_indices"),
        constant_op.constant([3, 3], dtypes.int64, name="x_shape"))
    result, dtype, used_fallback = converter.Convert(x, types_pb2.DT_INT32)
    self.assertIsInstance(result, tensor.Tensor)
    self.assertAllEqual(result, [[0, 0, 0], [1, 2, 3], [0, 0, 0]])
    self.assertEqual(dtype, types_pb2.DT_INT32)
    self.assertTrue(used_fallback)

  def testConvertIndexedSlicesWithIncorrectDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert the sparse row value of 32-bit integers requesting float. Asserts a value
    # error or type error is raised.
    converter = self.makePythonTensorConverter()
    x = indexed_slices.IndexedSlices(
        constant_op.constant([[1, 2, 3]], dtypes.int32, name="x_values"),
        constant_op.constant([1], dtypes.int64, name="x_indices"),
        constant_op.constant([3, 3], dtypes.int64, name="x_shape"))
    with self.assertRaises((ValueError, TypeError)):
      converter.Convert(x, types_pb2.DT_FLOAT)


if __name__ == "__main__":
  googletest.main()
