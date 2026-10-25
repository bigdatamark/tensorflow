# [MODULE DOC]
# Tests for TF NumPy logical operations (currently only element-wise equality from np_math_ops),
# comparing results against NumPy over many input array representations. Contains the test case
# plus a small helper that converts results for NumPy. Run as a script it enables eager execution and
# NumPy methods on tensors, then starts the test runner.
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
"""Tests for tf numpy logical methods."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test


class LogicTest(test.TestCase):
  # [CLASS DOC]
  # Test case for logical operations. Its setUp builds a list of conversion callables (identity,
  # tensor conversion, NumPy arrays of several dtypes, TF NumPy arrays of several dtypes) that tests
  # apply to inputs; it also provides matching helpers comparing shape, dtype and values.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture that calls the base setUp and stores on the instance a list named array_transforms of
    # callables, each turning a Python value into an array: identity, tensor conversion, NumPy array
    # with inferred dtype or int32/int64/float32/float64, and the TF NumPy array function with inferred
    # dtype or int32/int64/float32/float64 (the latter via TF dtypes). Returns nothing.
    super(LogicTest, self).setUp()
    self.array_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.int32),
        lambda x: np.array(x, dtype=np.int64),
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=dtypes.int32),
        lambda x: np_array_ops.array(x, dtype=dtypes.int64),
        lambda x: np_array_ops.array(x, dtype=dtypes.float32),
        lambda x: np_array_ops.array(x, dtype=dtypes.float64),
    ]

  def testEqual(self):
    # [FUNCTION DOC]
    # Tests the equal operation. For each pair of transforms applied to each of two inputs, computes
    # equality with the TF NumPy function and with NumPy (after making inputs NumPy compatible) and
    # asserts via match that dtype, shape and values are identical.

    def run_test(x1, x2=None):
      # [FUNCTION DOC]
      # HELPER: inner driver of testEqual that runs one input case.
      # Inputs: x1 is the first operand, x2 the optional second operand (defaults to x1). Loops over all
      # pairs of array transforms, converts each operand, and asserts the TF NumPy equal matches NumPy's
      # equal. Called with scalars, vectors and nested lists, including broadcasting cases and unequal
      # elements. Returns nothing; assertion failure raises a test failure.
      if x2 is None:
        x2 = x1
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arg1 = fn1(x1)
          arg2 = fn2(x2)
          self.match(
              np_math_ops.equal(arg1, arg2),
              np.equal(
                  make_numpy_compatible(arg1), make_numpy_compatible(arg2)))

    run_test(1)
    run_test(1, 2)
    run_test([1, 2])
    run_test([1, 2, 3], [2])
    run_test([[1, 2], [3, 4]], [1, 2])
    run_test([[1, 2], [1, 4]], [1, 2])
    run_test([1, 2], [[1, 2], [1, 4]])
    run_test([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    run_test([[1, 2], [3, 4]], [[1, 3], [3, 4]])

  def match_shape(self, actual, expected, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the tests in this file.
    # Asserts actual and expected have the same shape. Inputs: actual and expected are array-like
    # objects with a shape; msg is an optional message prefix. If msg is given it is replaced by a
    # formatted shape-mismatch message; otherwise the default assertion message is used.
    if msg:
      msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
          msg, expected.shape, actual.shape)
    self.assertEqual(actual.shape, expected.shape, msg=msg)

  def match_dtype(self, actual, expected, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the tests in this file.
    # Asserts actual and expected have the same dtype. Inputs: actual and expected are array-like
    # objects with a dtype; msg is an optional message. If msg is given it is replaced by a formatted
    # dtype-mismatch message; otherwise the default assertion message is used.
    if msg:
      msg = 'Dtype match failed for: {}. Expected: {} Actual: {}.'.format(
          msg, expected.dtype, actual.dtype)
    self.assertEqual(actual.dtype, expected.dtype, msg=msg)

  def match(self, actual, expected, msg=None):
    # [FUNCTION DOC]
    # HELPER: main comparison assertion for the tests in this file.
    # Inputs: actual (result under test), expected (reference NumPy value), msg optional extra text.
    # Builds a message with expected and actual values. Asserts actual is a TF NumPy array, then checks
    # dtype and shape match, then compares values: as plain lists for scalars (rank zero), or as
    # sequences otherwise. Raises a test assertion failure on any mismatch.
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, np_arrays.ndarray)
    self.match_dtype(actual, expected, msg)
    self.match_shape(actual, expected, msg)
    if not actual.shape.rank:
      self.assertEqual(actual.tolist(), expected.tolist())
    else:
      self.assertSequenceEqual(actual.tolist(), expected.tolist())


def make_numpy_compatible(s):
  # [FUNCTION DOC]
  # HELPER: module-level utility for the tests (converts results for NumPy comparison).
  # Input s is any value. If it is a TF NumPy array (tensor), returns its NumPy value; otherwise
  # returns it unchanged. No side effects.
  return s if not isinstance(s, np_arrays.ndarray) else s.numpy()


if __name__ == '__main__':
  ops.enable_eager_execution()
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()
