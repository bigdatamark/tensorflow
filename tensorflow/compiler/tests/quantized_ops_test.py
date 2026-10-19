# [MODULE DOC]
# XLA compiler tests for quantized types: that quantized tensors can be cast, reshaped and
# clustered by XLA, and that the XLA dequantize op correctly dequantizes data packed four bytes
# per 32-bit word. No import-time side effects.
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
"""Tests for quantized operations."""

import math

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class QuantizedOpsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case checking that XLA can handle the quantized tensor types supported by the test
  # device. Holds no fixtures.

  # Verify that quantized types can be clustered by XLA.
  def testQuantizedTypeRoundtrip(self):
    # [FUNCTION DOC]
    # For each quantized type the device supports, casts six int32 values to that type on the
    # XLA device, reshapes to 3x2, and asserts the values fed back equal the original layout.
    with self.session() as session:
      for dtype in self.quantized_tf_types:
        in_values = np.array([1, 2, 3, 4, 5, 6])
        expected = [[1, 2], [3, 4], [5, 6]]
        with self.test_scope():
          p = array_ops.placeholder(dtype=dtypes.int32)
          x = math_ops.cast(p, dtype)
          x = array_ops.reshape(x, [3, 2])

        value = session.run(x, {p: in_values})
        self.assertAllEqual(value, expected)


class DequantizedOpsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the XLA dequantize op. Provides a helper to pack unsigned 8-bit matrices into
  # 32-bit words and one test for the unsigned 8-bit quantized type.

  def pack_uint8_r2_to_uint32(self, test_input):
    # [FUNCTION DOC]
    # HELPER: packs a rank-2 tensor of 8-bit values into 32-bit words, four columns per word.
    # Casts the input to 8 bits, zero-pads the column count up to a multiple of four, then for
    # each position within a group of four gathers every fourth column, shifts it left so that
    # the first column ends up in the most significant byte, and ORs the results together.
    # Input must have a statically known shape. Returns a tensor with the same rows and
    # ceil(columns/4) unsigned 32-bit columns.
    num_rows, num_columns = test_input.get_shape().as_list()
    num_output_columns = int(math.ceil(num_columns / 4.0))
    padding_input = array_ops.pad(
        math_ops.cast(test_input, dtype=dtypes.uint8),
        constant_op.constant([[
            0,
            0,
        ], [0, num_output_columns * 4 - num_columns]]))
    output = array_ops.zeros([num_rows, num_output_columns],
                             dtype=dtypes.uint32)
    num_elements_per_pack = 4
    shift_bits = 8

    iota_r1 = math_ops.range(num_output_columns * num_elements_per_pack)

    for p in range(num_elements_per_pack):
      selected_index = math_ops.equal(
          math_ops.mod(iota_r1, num_elements_per_pack), p)
      gather_index = array_ops.boolean_mask(iota_r1, selected_index)
      gathered_input = array_ops.gather(padding_input, gather_index, axis=1)
      total_shift_bits = shift_bits * (num_elements_per_pack - p - 1)
      left_shift_input = bitwise_ops.left_shift(
          math_ops.cast(gathered_input, dtype=dtypes.uint32), total_shift_bits)
      output = bitwise_ops.bitwise_or(output, left_shift_input)
    return output

  def testDequantizeQuint8(self):
    # [FUNCTION DOC]
    # Draws a 100x3547 normal random matrix (mean 128, stddev 10), and on the CPU transposes and
    # quantizes it to unsigned 8-bit over the range 0 to 255, then packs it into 32-bit words.
    # On the XLA device dequantizes with the min-combined mode (transposed output) and slices
    # back to the original shape. Asserts the result matches the original within tolerance 1.0.
    num_rows = 100
    num_columns = 3547
    random_input = np.random.normal(128.0, 10.0, [num_rows, num_columns])
    with self.session() as session:
      with ops.device("CPU"):
        test_input = ops.convert_to_tensor(random_input, dtype=dtypes.float32)
        transposed_input = array_ops.transpose(test_input, [1, 0])
        quantized_input = array_ops.quantize(transposed_input, 0.0, 255.0,
                                             dtypes.quint8)
        packed_input = self.pack_uint8_r2_to_uint32(quantized_input.output)
      with self.test_scope():
        transposed_quantized_output = xla.dequantize(packed_input, 0.0, 255.0,
                                                     "MIN_COMBINED", True)
        quantized_output = array_ops.slice(transposed_quantized_output, [0, 0],
                                           [num_rows, num_columns])

      value = session.run(quantized_output)
    self.assertAllClose(value, random_input, 1.0)


if __name__ == "__main__":
  googletest.main()
