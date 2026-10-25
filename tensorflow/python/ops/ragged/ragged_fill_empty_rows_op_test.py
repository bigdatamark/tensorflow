# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Test module for ragged_array_ops.fill_empty_rows, which inserts a default value into each empty
# row of a ragged tensor and reports which rows were empty. Covers integer, float (with gradient
# checks, function tracing), string and no-empty-row inputs. No import-time side effects.
"""Tests tf.ragged.fill_empty_rows."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


class RaggedFillEmptyRowsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the ragged fill_empty_rows operation. Each test builds a small ragged tensor
  # with a default value, runs the op and asserts the filled rows, row lengths, flat values and
  # the empty-row indicator. No shared fixtures.

  def testFillInt(self):
    # [FUNCTION DOC]
    # Int32 case on GPU when available: rows [empty, four values, empty, four values, empty]
    # with default -1. Asserts each empty row becomes a single -1, row lengths are 1,4,1,4,1,
    # flat values are as expected and the indicator marks rows 0, 2, 4 as empty. Returns nothing.
    with test_util.use_gpu():
      default_value = constant_op.constant(-1, dtype=dtypes.int32)
      ragged_input = ragged_factory_ops.constant(
          [[], [1, 3, 5, 7], [], [2, 4, 6, 8], []], dtype=dtypes.int32
      )
      ragged_output, empty_row_indicator = ragged_array_ops.fill_empty_rows(
          ragged_input, default_value
      )

      self.assertAllEqual(
          ragged_output, [[-1], [1, 3, 5, 7], [-1], [2, 4, 6, 8], [-1]]
      )
      self.assertAllEqual(ragged_output.row_lengths(), [1, 4, 1, 4, 1])
      self.assertAllEqual(
          ragged_output.values, [-1, 1, 3, 5, 7, -1, 2, 4, 6, 8, -1]
      )
      self.assertAllEqual(empty_row_indicator, [True, False, True, False, True])

  @test_util.run_deprecated_v1
  def testFillFloat(self):
    # [FUNCTION DOC]
    # Graph-mode (deprecated v1) float64 test. Builds a ragged tensor from float values and row
    # lengths with default -1.0, asserts row lengths, flat values and empty-row indicator, then
    # numerically checks the gradient with respect to the values and with respect to the default
    # value: each error must be positive and below 1e-8. Returns nothing.
    with self.session():
      values = constant_op.constant(
          [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0], dtype=dtypes.float64
      )
      default_value = constant_op.constant(-1.0, dtype=dtypes.float64)
      ragged_input = ragged_tensor.RaggedTensor.from_row_lengths(
          values=values, row_lengths=[0, 4, 0, 4, 0]
      )
      ragged_output, empty_row_indicator = ragged_array_ops.fill_empty_rows(
          ragged_input, default_value
      )

      self.assertAllEqual(ragged_output.row_lengths(), [1, 4, 1, 4, 1])
      self.assertAllEqual(
          ragged_output.values,
          [-1.0, 1.0, 3.0, 5.0, 7.0, -1.0, 2.0, 4.0, 6.0, 8.0, -1.0],
      )
      self.assertAllEqual(empty_row_indicator, [True, False, True, False, True])

      values_grad_err = gradient_checker.compute_gradient_error(
          values, values.shape.as_list(), ragged_output.values, [11], delta=1e-8
      )
      self.assertGreater(values_grad_err, 0)
      self.assertLess(values_grad_err, 1e-8)

      default_value_grad_err = gradient_checker.compute_gradient_error(
          default_value,
          default_value.shape.as_list(),
          ragged_output.values,
          [11],
          delta=1e-8,
      )
      self.assertGreater(default_value_grad_err, 0)
      self.assertLess(default_value_grad_err, 1e-8)

  def testFillFloatFunction(self):
    # [FUNCTION DOC]
    # Checks fill_empty_rows works inside a compiled function with float64 ragged input: calls it
    # and asserts the row lengths, flat values and empty-row indicator. Returns nothing.
    @def_function.function
    def func(ragged_input):
      # [FUNCTION DOC]
      # HELPER: compiled wrapper for the test. Fills empty rows of the given ragged tensor with
      # the float64 default -1.0 and returns the filled tensor and the empty-row indicator.
      default_value = constant_op.constant(-1.0, dtype=dtypes.float64)
      return ragged_array_ops.fill_empty_rows(ragged_input, default_value)

    ragged_input = ragged_factory_ops.constant(
        [[], [1.0, 3.0, 5.0, 7.0], [], [2.0, 4.0, 6.0, 8.0], []],
        dtype=dtypes.float64,
    )
    ragged_output, empty_row_indicator = func(ragged_input)
    self.assertAllEqual(ragged_output.row_lengths(), [1, 4, 1, 4, 1])
    self.assertAllEqual(
        ragged_output.values,
        [-1.0, 1.0, 3.0, 5.0, 7.0, -1.0, 2.0, 4.0, 6.0, 8.0, -1.0],
    )
    self.assertAllEqual(empty_row_indicator, [True, False, True, False, True])

  def testFillString(self):
    # [FUNCTION DOC]
    # String case forced onto the CPU: default value "x" fills empty rows; asserts row lengths,
    # flat byte-string values and the empty-row indicator. Returns nothing.
    with test_util.force_cpu():
      values = constant_op.constant(
          ["a", "c", "e", "g", "b", "d", "f", "h"], dtype=dtypes.string
      )
      default_value = constant_op.constant("x", dtype=dtypes.string)
      ragged_input = ragged_tensor.RaggedTensor.from_row_lengths(
          values=values, row_lengths=[0, 4, 0, 4, 0]
      )
      ragged_output, empty_row_indicator = ragged_array_ops.fill_empty_rows(
          ragged_input, default_value
      )

      self.assertAllEqual(ragged_output.row_lengths(), [1, 4, 1, 4, 1])
      self.assertAllEqual(
          ragged_output.values,
          [b"x", b"a", b"c", b"e", b"g", b"x", b"b", b"d", b"f", b"h", b"x"],
      )
      self.assertAllEqual(
          empty_row_indicator, np.array([1, 0, 1, 0, 1]).astype(np.bool_)
      )

  def testNoEmptyRows(self):
    # [FUNCTION DOC]
    # Int32 input whose rows are all non-empty (two rows of four): asserts the output is
    # unchanged (same row lengths and values) and the indicator is all false. Returns nothing.
    with test_util.use_gpu():
      values = constant_op.constant(
          [1, 3, 5, 7, 2, 4, 6, 8], dtype=dtypes.int32
      )
      default_value = constant_op.constant(-1, dtype=dtypes.int32)
      ragged_input = ragged_tensor.RaggedTensor.from_row_lengths(
          values=values, row_lengths=[4, 4]
      )
      ragged_output, empty_row_indicator = ragged_array_ops.fill_empty_rows(
          ragged_input, default_value
      )

      self.assertAllEqual(ragged_output.row_lengths(), [4, 4])
      self.assertAllEqual(ragged_output.values, values)
      self.assertAllEqual(empty_row_indicator, np.zeros(2).astype(np.bool_))


if __name__ == "__main__":
  googletest.main()
