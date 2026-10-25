# [MODULE DOC]
# Unit tests for the clipping ops in the clip_ops module: clip by norm (dense and indexed-slices),
# clip by global norm (including mixed dtypes and non-finite norms). Run through the standard
# test main entry point; no import-time side effects.
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
"""Tests for Clip Operations."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import numerics
from tensorflow.python.platform import test


class ClipOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the norm-based clipping ops. Holds private check helpers shared by the
  # test methods; no persistent fixtures.

  def __init__(self, method_name="runTest"):
    # [FUNCTION DOC]
    # Constructor that forwards the test method name (default is the run-test method) to the
    # base test case; adds no state.
    super(ClipOpsTest, self).__init__(method_name)

  def _testClipTensorByNorm(self, inputs, max_norm, expected):
    # [FUNCTION DOC]
    # HELPER: shared check for the clip-by-norm tests.
    # Clips a constant tensor built from inputs to max_norm, runs numeric checks on all ops in
    # the graph, evaluates, and asserts the result is close to expected.
    input_op = constant_op.constant(inputs)
    clipped = clip_ops.clip_by_norm(input_op, max_norm)
    check_op = numerics.add_check_numerics_ops()
    result, _ = self.evaluate([clipped, check_op])
    self.assertAllClose(result, expected)

  def _testClipTensorByGlobalNorm(self, inputs, max_norm, expected):
    # [FUNCTION DOC]
    # HELPER: shared check for the clip-by-global-norm tests.
    # Clips the list inputs by global norm to max_norm, evaluates, and asserts the clipped
    # values are close to expected (the returned norm is ignored).
    clipped = clip_ops.clip_by_global_norm(inputs, max_norm)
    result, _ = self.evaluate(clipped)
    self.assertAllClose(result, expected)

  def _testNonFiniteClippingByGlobalNorm(self, inputs, max_norm):
    # [FUNCTION DOC]
    # HELPER: shared check for non-finite inputs.
    # Clips inputs by global norm and asserts every clipped value is not-a-number.
    clipped = clip_ops.clip_by_global_norm(inputs, max_norm)
    result, _ = self.evaluate(clipped)
    self.assertTrue(np.all(np.isnan(result)))

  def _testClipIndexedSlicesByNorm(self, values, indices, shape, max_norm,
                                   axes):
    # [FUNCTION DOC]
    # HELPER: shared check for indexed-slices clipping.
    # Builds an indexed-slices from values, indices and dense shape, clips it by norm over axes,
    # asserts the result is still indexed-slices, then clips the equivalent dense tensor and
    # asserts both converted results are close.
    values = constant_op.constant(values)
    indices = constant_op.constant(indices)
    shape = constant_op.constant(shape)
    # IndexedSlices mode
    indexed_slices = indexed_slices_lib.IndexedSlices(values, indices, shape)
    clipped = clip_ops.clip_by_norm(indexed_slices, max_norm, axes)
    # clipped should be IndexedSlices
    self.assertIsInstance(clipped, indexed_slices_lib.IndexedSlices)
    clipped = ops.convert_to_tensor(clipped)

    # Tensor mode
    dense_tensor = ops.convert_to_tensor(indexed_slices)
    dense_clipped = clip_ops.clip_by_norm(dense_tensor, max_norm, axes)
    result, expected = self.evaluate([clipped, dense_clipped])
    self.assertAllClose(result, expected)

  @test_util.run_deprecated_v1
  def testClipTensorByNorm(self):
    # [FUNCTION DOC]
    # Checks clip by norm on a 2x3 matrix for three cases: a norm above the limit (scaled down),
    # a norm below the limit (unchanged), and all zeros (stays zero without NaN).
    # Simple example
    self._testClipTensorByNorm([[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], 4.0,
                               [[-2.4, 0.0, 0.0], [3.2, 0.0, 0.0]])
    # No clipping.
    self._testClipTensorByNorm([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 4.0,
                               [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # Zero norm
    self._testClipTensorByNorm([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 4.0,
                               [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

  @test_util.run_deprecated_v1
  def testClipTensorByGlobalNorm(self):
    # [FUNCTION DOC]
    # Checks clip by global norm for the same three cases: scaled down, unchanged, and all zeros.
    # Simple example
    self._testClipTensorByGlobalNorm([[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], 4.0,
                                     [[-2.4, 0.0, 0.0], [3.2, 0.0, 0.0]])
    # No clipping.
    self._testClipTensorByGlobalNorm([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 4.0,
                                     [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # Zero norm.
    self._testClipTensorByGlobalNorm([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 4.0,
                                     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

  @test_util.run_deprecated_v1
  def testClipTensorByGlobalNormMultipleDtypes(self):
    # [FUNCTION DOC]
    # Clips a float32 tensor and a bfloat16 tensor together with a supplied norm of 5 and limit 4.
    # Asserts the values are scaled by 0.8 (bfloat16 with its rounding) and each output keeps its
    # input dtype.
    (a, b), _ = self.evaluate(
        clip_ops.clip_by_global_norm(
            [
                ops.convert_to_tensor([-3.0, 0.0, 0.0], dtype=dtypes.float32),
                ops.convert_to_tensor([4.0, 0.0, 0.0], dtype=dtypes.bfloat16),
            ],
            clip_norm=4.0,
            use_norm=ops.convert_to_tensor(5.0, dtype=dtypes.float32),
        )
    )
    self.assertAllClose(a, [-2.4, 0.0, 0.0])
    self.assertEqual(a.dtype, dtypes.float32)
    self.assertAllClose(b, [3.203125, 0.0, 0.0])
    self.assertEqual(b.dtype, dtypes.bfloat16)

  @test_util.run_deprecated_v1
  def testGlobalClipWithNonfinite(self):
    # [FUNCTION DOC]
    # Runs the non-finite helper with positive infinity, negative infinity and not-a-number in one
    # input; asserts all outputs become not-a-number.
    self._testNonFiniteClippingByGlobalNorm(
        [[-3.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]], 4.0)
    self._testNonFiniteClippingByGlobalNorm(
        [[-3.0, 0.0, 0.0], [float("-inf"), 0.0, 0.0]], 4.0)
    self._testNonFiniteClippingByGlobalNorm(
        [[-3.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]], 4.0)

  def testClipIndexedSlicesByNorm(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: several comments mislabel the axes (the "Axes == 2" case actually passes 1, and
    # the repeated "[0, 1]" comments cover other axes lists), so one case duplicates another.
    # Runs the indexed-slices helper on a small 3-D example with a limit of 4 over axes nothing,
    # 0, 1, and several lists of axes, asserting sparse and dense clipping agree each time.
    values = [[[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
              [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]]
    indices = [2, 6]
    shape = [10, 2, 3]
    # Axes == None
    self._testClipIndexedSlicesByNorm(values, indices, shape, 4.0, None)

    # Axes == 0
    self._testClipIndexedSlicesByNorm(values, indices, shape, 4.0, 0)

    # Axes == 1
    self._testClipIndexedSlicesByNorm(values, indices, shape, 4.0, 1)

    # Axes == 2
    self._testClipIndexedSlicesByNorm(values, indices, shape, 4.0, 1)

    # Axes == [0, 1]
    self._testClipIndexedSlicesByNorm(values, indices, shape, 4.0, [0, 1])

    # Axes == [0, 1]
    self._testClipIndexedSlicesByNorm(values, indices, shape, 4.0, [0, 2])

    # Axes == [0, 1]
    self._testClipIndexedSlicesByNorm(values, indices, shape, 4.0, [1, 2])

    # Axes == [0, 1]
    self._testClipIndexedSlicesByNorm(values, indices, shape, 4.0, [0, 1, 2])


if __name__ == "__main__":
  test.main()
