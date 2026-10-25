# [MODULE DOC]
# Unit tests for the ragged bitcast op (reinterpreting a ragged tensor's element bits as
# another dtype). Covers same-size casts (including signed/unsigned reinterpretation),
# casts to a larger dtype (which fold an inner dimension of size equal to the size ratio
# away) and casts to a smaller dtype (which add an inner dimension), plus an error case
# for a ragged inner dimension. Run via the standard test main; no import-time side effects.
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
"""Tests for ragged_array_ops.bitcast."""

from absl.testing import parameterized

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedSplitOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the class is named as a split-op test, but it actually tests the ragged
  # bitcast op; the name is a leftover.
  # Test case for the ragged bitcast op, run in both graph and eager modes. Tests are
  # parameterized with input and expected output ragged values; no shared fixtures.

  @parameterized.parameters([
      #=========================================================================
      # Cast to same-size dtype.
      #=========================================================================
      dict(
          descr='int32 to int32 cast',
          inputs=ragged_factory_ops.constant_value(
              [[1, 2], [3]],
              dtype=dtypes.int32,
          ),
          outputs=ragged_factory_ops.constant_value(
              [[1, 2], [3]],
              dtype=dtypes.int32,
          )),
      dict(
          descr='int32 to uint32 cast',
          inputs=ragged_factory_ops.constant_value(
              [[1, 2], [-1]],
              dtype=dtypes.int32,
          ),
          outputs=ragged_factory_ops.constant_value(
              [[1, 2], [4294967295]],
              dtype=dtypes.uint32,
          )),
      dict(
          descr='uint32 to int32 cast',
          inputs=ragged_factory_ops.constant_value(
              [[1, 2], [4294967295]],
              dtype=dtypes.uint32,
          ),
          outputs=ragged_factory_ops.constant_value(
              [[1, 2], [-1]],
              dtype=dtypes.int32,
          )),
      #=========================================================================
      # Cast to larger dtype.
      #=========================================================================
      dict(
          descr='int32 to int64 cast',
          inputs=ragged_factory_ops.constant_value(
              [[[1, 0], [2, 0]], [[3, 0]]],
              dtype=dtypes.int32,
              ragged_rank=1,
          ),
          outputs=ragged_factory_ops.constant_value(
              [[1, 2], [3]],
              dtype=dtypes.int64,
          )),
      #=========================================================================
      # Cast to smaller dtype.
      #=========================================================================
      dict(
          descr='int64 to int32 cast',
          inputs=ragged_factory_ops.constant_value(
              [[1, 2], [3]],
              dtype=dtypes.int64,
          ),
          outputs=ragged_factory_ops.constant_value(
              [[[1, 0], [2, 0]], [[3, 0]]],
              dtype=dtypes.int32,
              ragged_rank=1,
          )),
  ])  # pyformat: disable
  def testBitcast(self, descr, inputs, outputs, name=None):
    # [FUNCTION DOC]
    # Parameterized success test. Inputs: descr (case label), inputs (ragged value to cast),
    # outputs (expected ragged result, whose dtype is the target dtype) and optional name.
    # Cases: int32 to int32, int32 to uint32 and back, int32 pairs to int64, and int64 to a
    # pair of int32. Runs the bitcast and asserts the result dtype, ragged rank and values
    # equal the expected output.
    result = ragged_array_ops.bitcast(inputs, outputs.dtype, name)
    self.assertEqual(result.dtype, outputs.dtype)
    self.assertEqual(result.ragged_rank, outputs.ragged_rank)
    self.assertAllEqual(result, outputs)

  @parameterized.parameters([
      dict(
          descr='Upcast requires uniform inner dimension',
          inputs=ragged_factory_ops.constant_value(
              [[[1, 0], [2, 0]], [[3, 0]]],
              dtype=dtypes.int32,
              ragged_rank=2,
          ),
          cast_to_dtype=dtypes.int64,
          exception=ValueError,
          message='`input.flat_values` is required to have rank >= 2'),
  ])  # pyformat: disable
  def testBitcastError(self,
                       descr,
                       inputs,
                       cast_to_dtype,
                       exception,
                       message,
                       name=None):
    # [FUNCTION DOC]
    # Parameterized failure test. Inputs: descr (label), inputs (ragged value), cast_to_dtype
    # (target dtype), exception (expected error class), message (expected message pattern)
    # and optional name. Scenario: casting to a larger dtype when the innermost dimension
    # is itself ragged. Expects a value error saying the flat values need rank at least 2.
    with self.assertRaisesRegex(exception, message):
      result = ragged_array_ops.bitcast(inputs, cast_to_dtype, name)
      self.evaluate(result)


if __name__ == '__main__':
  googletest.main()
