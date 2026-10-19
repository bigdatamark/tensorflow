# [MODULE DOC]
# Test module for the argmin and argmax operations on an XLA test device. Provides a small
# comparison helper and one test sweeping input dtypes and output index types. It exposes no
# public API; running it as a script starts the test runner. No import-time side effects.
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
"""Functional tests for ArgMin and ArgMax Ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ArgMinMaxTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the argmin and argmax operations. Holds a helper that runs one operator on
  # the XLA test device and a single test sweeping dtypes. No instance state of its own.

  def _assertOpOutputMatchesExpected(self, op, axis, output_type, op_input,
                                     expected):
    # [FUNCTION DOC]
    # HELPER: comparison helper for the argmin/argmax test.
    # Runs the given operator (argmin or argmax) on the XLA test device with a placeholder shaped
    # like op_input, reducing along axis (integer) and producing indices of output_type (numpy
    # type). Feeds op_input (numpy array) and asserts the result equals expected (numpy array or
    # scalar). Returns nothing; fails the test on mismatch.
    """Verifies that 'op' produces 'expected' when fed input 'op_input' .

    Args:
      op: argmin or argmax operator to test.
      axis: integer axis to reduce across.
      output_type: numpy datatype of the output to produce.
      op_input: numpy input array to use as input to 'op'.
      expected: numpy array representing the expected output of 'op'.
    """
    with self.session() as session:
      with self.test_scope():
        pinp = array_ops.placeholder(
            dtypes.as_dtype(op_input.dtype), op_input.shape, name="a")
        output = op(pinp, axis=axis, output_type=output_type)
      result = session.run(output, {pinp: op_input})
      self.assertAllEqual(result, expected)

  def testArgMinMax(self):
    # [FUNCTION DOC]
    # For every integer and float input type and for each output index type among 32 and 64 bit
    # integers, checks argmax on a vector (index 2), on a matrix along axis 0 and along axis 1,
    # and argmin on a vector (index 4) and on a matrix along both axes. Complex types are excluded because they do not support these ops.
    # Complex numbers do not support argmin/argmax.
    minmax_types = self.all_types & {np.int32, np.int64}
    for dtype in self.int_types | self.float_types:
      # output_type is a numpy data type that is used to specify the desired
      # output type of the op as well as to convert the Python number to the
      # array scalar of the type.
      for output_type in minmax_types:
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array([1, 10, 27, 3, 3, 4], dtype=dtype),
            expected=output_type(2))
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=0,
            output_type=output_type,
            op_input=np.array([[4, 1, 7], [3, 2, 4]], dtype=dtype),
            expected=np.array([0, 1, 0], dtype=output_type))
        self._assertOpOutputMatchesExpected(
            math_ops.argmax,
            axis=1,
            output_type=output_type,
            op_input=np.array([[4, 1], [3, 2]], dtype=dtype),
            expected=np.array([0, 0], dtype=output_type))

        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array([3, 10, 27, 3, 2, 4], dtype=dtype),
            expected=output_type(4))
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=0,
            output_type=output_type,
            op_input=np.array([[4, 1, 7], [3, 2, 4]], dtype=dtype),
            expected=np.array([1, 0, 1], dtype=output_type))
        self._assertOpOutputMatchesExpected(
            math_ops.argmin,
            axis=1,
            output_type=output_type,
            op_input=np.array([[4, 1], [3, 2]], dtype=dtype),
            expected=np.array([1, 1], dtype=output_type))


if __name__ == "__main__":
  test.main()
