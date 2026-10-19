# [MODULE DOC]
# XLA compiler test for the explicit XLA reduce-window operation exposed through the tf2xla
# Python bindings. Checks windowed sum and product reductions with various window sizes, strides
# and padding on one- and two-dimensional inputs. No import-time side effects.
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
"""Tests for xla.reduce_window."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class ReduceWindowTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the XLA reduce-window op. Contains a helper that runs the op on the test
  # device and one test that exercises sum and product reducers.
  """Test cases for xla.reduce_window."""

  def _reduce_window(self, operand, init, reducer, **kwargs):
    # [FUNCTION DOC]
    # HELPER: runs the XLA reduce-window op on the test device. Takes the operand array, the
    # initial scalar value, the reducer function (a defined binary function) and extra keyword
    # arguments (window dimensions, window strides, padding, ...) forwarded to the op. Feeds the
    # operand through a placeholder in a fresh session and returns the evaluated output array.
    with self.session():
      placeholder = array_ops.placeholder(operand.dtype)
      with self.test_scope():
        output = xla.reduce_window(placeholder, init, reducer, **kwargs)
      return output.eval(feed_dict={placeholder: operand})

  def testReduceWindow(self):
    # [FUNCTION DOC]
    # For bfloat16 and float32 (when supported; float16 and float64 are unimplemented), defines
    # sum and multiply reducers and checks five cases: window 2 sums over a 7-element vector;
    # window 2 with stride 2; window 1 with stride 3; a 2x2 window product with stride 1 over a
    # 3x4 matrix (initial value 1); and a 2x2 window sum with stride 2 and padding of (2,3) rows
    # and (1,2) columns, whose expected output has zero rows from the padding.

    # TODO(b/77644762): float16 and float64 ReduceWindow are unimplemented.
    for dtype in set(self.numeric_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])):

      @function.Defun(dtype, dtype)
      def sum_reducer(x, y):
        # [FUNCTION DOC]
        # HELPER: binary reducer (declared as a graph function over the loop dtype) that returns
        # the sum of its two scalar arguments.
        return x + y

      @function.Defun(dtype, dtype)
      def mul_reducer(x, y):
        # [FUNCTION DOC]
        # HELPER: binary reducer (declared as a graph function over the loop dtype) that returns
        # the product of its two scalar arguments.
        return x * y

      self.assertAllClose(
          np.array([3, 5, 7, 9, 11, 13], dtype=dtype),
          self._reduce_window(
              np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype),
              0.0,
              sum_reducer,
              window_dimensions=[2]))

      self.assertAllClose(
          np.array([3, 7, 11], dtype=dtype),
          self._reduce_window(
              np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype),
              0.0,
              sum_reducer,
              window_dimensions=[2],
              window_strides=[2]))

      self.assertAllClose(
          np.array([1, 4, 7], dtype=dtype),
          self._reduce_window(
              np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype),
              0.0,
              sum_reducer,
              window_dimensions=[1],
              window_strides=[3]))

      self.assertAllClose(
          np.array([[24, 36, 24], [96, 0, 0]], dtype=dtype),
          self._reduce_window(
              np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 0, 1]], dtype=dtype),
              1.0,
              mul_reducer,
              window_dimensions=[2, 2],
              window_strides=[1, 1]))

      self.assertAllClose(
          np.array([[0, 0, 0], [5, 10, 5], [2, 4, 1], [0, 0, 0]], dtype=dtype),
          self._reduce_window(
              np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 0, 1]], dtype=dtype),
              0.0,
              sum_reducer,
              window_dimensions=[2, 2],
              window_strides=[2, 2],
              padding=[[2, 3], [1, 2]]))


if __name__ == '__main__':
  googletest.main()
