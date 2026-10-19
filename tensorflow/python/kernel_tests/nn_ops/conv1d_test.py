# [MODULE DOC]
# Kernel tests for the 1-D convolution entry points of the nn ops: the argument handling of
# conv1d (basic and with an extra leading batch dimension) and a numeric check of conv1d_transpose.
# Tests use tiny hand-computable inputs and run on GPU when one is available. No import-time side
# effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for convolution related functionality in tensorflow.ops.nn."""
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class Conv1DTest(test.TestCase):
  # [CLASS DOC]
  # Test case for conv1d and conv1d_transpose with small constant inputs. No shared fixtures.

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: a one-by-four-by-one input holding one to four and a two-tap filter holding two
    # and one, for float16, float32 and (except on ROCm builds) float64, with strides one and two
    # and VALID padding. Runs conv1d in a session on GPU if available and asserts the squeezed
    # output has the expected length (three or two) and the hand-computed values.
    """Test that argument passing to conv1d is handled properly."""
    # double datatype is currently not supported for convolution ops
    # on the ROCm platform
    optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
    for dtype in [dtypes.float16, dtypes.float32] + optional_float64:
      x = constant_op.constant([1, 2, 3, 4], dtype=dtype)
      x = array_ops.expand_dims(x, 0)  # Add batch dimension
      x = array_ops.expand_dims(x, 2)  # And depth dimension
      filters = constant_op.constant([2, 1], dtype=dtype)
      filters = array_ops.expand_dims(filters, 1)  # in_channels
      filters = array_ops.expand_dims(filters, 2)  # out_channels
      # Filters is 2x1x1
      for stride in [1, 2]:
        with self.cached_session(use_gpu=test.is_gpu_available()):
          c = nn_ops.conv1d(x, filters, stride, padding="VALID")
          reduced = array_ops.squeeze(c)
          output = self.evaluate(reduced)
          if stride == 1:
            self.assertEqual(len(output), 3)
            self.assertAllClose(output,
                                [2 * 1 + 1 * 2, 2 * 2 + 1 * 3, 2 * 3 + 1 * 4])
          else:
            self.assertEqual(len(output), 2)
            self.assertAllClose(output, [2 * 1 + 1 * 2, 2 * 3 + 1 * 4])

  def testExpandedBatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is a copy of the basic test's and the ROCm float64 comment above
    # the body is stale; only float32 is used here.
    # Scenario: same data as the basic test but stacked into a batch shape of two by one (extra
    # leading batch dimension), float32, strides one and two, VALID padding. Asserts conv1d
    # accepts the expanded batch and the squeezed output has the hand-computed values for both
    # batch entries.
    """Test that argument passing to conv1d is handled properly."""
    # double datatype is currently not supported for convolution ops
    # on the ROCm platform
    x = constant_op.constant([1, 2, 3, 4], dtype=dtypes.float32)
    x = array_ops.expand_dims(x, 0)  # Add batch dimension
    x = array_ops.expand_dims(x, 2)  # And depth dimension
    x = array_ops_stack.stack([x, x])  # Make batch shape [2, 1]
    filters = constant_op.constant([2, 1], dtype=dtypes.float32)
    filters = array_ops.expand_dims(filters, 1)  # in_channels
    filters = array_ops.expand_dims(filters, 2)  # out_channels
    # Filters is 2x1x1
    for stride in [1, 2]:
      with self.cached_session(use_gpu=test.is_gpu_available()):
        c = nn_ops.conv1d(x, filters, stride, padding="VALID")
        reduced = array_ops.squeeze(c)  # Sequeeze out dims 1 and 3.
        output = self.evaluate(reduced)
        if stride == 1:
          self.assertAllClose(output,
                              [[2 * 1 + 1 * 2, 2 * 2 + 1 * 3, 2 * 3 + 1 * 4],
                               [2 * 1 + 1 * 2, 2 * 2 + 1 * 3, 2 * 3 + 1 * 4]])
        else:
          self.assertAllClose(
              output,
              [[2 * 1 + 1 * 2, 2 * 3 + 1 * 4], [2 * 1 + 1 * 2, 2 * 3 + 1 * 4]])

  def testConv1DTranspose(self):
    # [FUNCTION DOC]
    # Scenario: all-ones float32 input (batch two, width four, depth three) and all-ones filter
    # (width three, output depth two, input depth three) with stride two, VALID padding and
    # requested output width nine. Builds the expected array by hand: interior positions get
    # three, doubled at interior stride-aligned positions, and the border copies its neighbour.
    # Asserts the transposed convolution matches that array.
    with self.cached_session():
      stride = 2

      # Input, output: [batch, width, depth]
      x_shape = [2, 4, 3]
      y_shape = [2, 9, 2]

      # Filter: [kernel_width, output_depth, input_depth]
      f_shape = [3, 2, 3]

      x = constant_op.constant(
          1.0, shape=x_shape, name="x", dtype=dtypes.float32)
      f = constant_op.constant(
          1.0, shape=f_shape, name="filter", dtype=dtypes.float32)
      output = nn_ops.conv1d_transpose(
          x, f, y_shape, strides=stride, padding="VALID")
      value = self.evaluate(output)

      cache_values = np.zeros(y_shape, dtype=np.float32)

      # The amount of padding added
      pad = 1

      for n in range(x_shape[0]):
        for k in range(f_shape[1]):
          for w in range(pad, y_shape[1] - pad):
            target = 3.0
            # We add a case for locations divisible by the stride.
            w_in = w % stride == 0 and w > pad and w < y_shape[1] - 1 - pad
            if w_in:
              target += 3.0
            cache_values[n, w, k] = target

          # copy values in the border
          cache_values[n, 0, k] = cache_values[n, 1, k]
          cache_values[n, -1, k] = cache_values[n, -2, k]

    self.assertAllClose(cache_values, value)


if __name__ == "__main__":
  test.main()
