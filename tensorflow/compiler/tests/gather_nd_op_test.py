# [MODULE DOC]
# Tests for the gather-nd op on the XLA test device, covering numeric types, empty inputs,
# scalar and slice indexing, higher-rank params and indices, and the ignore policy for
# out-of-range indices. Runs the test main when executed directly.
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
"""Tests for tensorflow.ops.tf.gather_nd."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class GatherNdTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the gather-nd op; every test builds numpy params and indices and runs
  # them through a shared helper. Stateless.

  def _runGather(self, params, indices, bad_indices_policy=""):
    # [FUNCTION DOC]
    # HELPER: runs gather-nd on the test device.
    # Inputs: params array, integer indices array, and an optional bad-indices policy string
    # (default empty, i.e. the op default). Feeds both through placeholders and returns the
    # evaluated gathered array.
    with self.session():
      paramsp = array_ops.placeholder(params.dtype)
      indicesp = array_ops.placeholder(indices.dtype)
      with self.test_scope():
        gather_nd_t = array_ops.gather_nd(
            paramsp, indicesp, bad_indices_policy=bad_indices_policy
        )
      feed_dict = {paramsp: params, indicesp: indices}
      return gather_nd_t.eval(feed_dict=feed_dict)

  def testSimpleDtype(self):
    # [FUNCTION DOC]
    # For each numeric type, gathers three entries from a 1-D array by index lists and checks
    # the expected values (7, 7, 8).
    for dtype in self.numeric_types:
      self.assertAllEqual(
          np.array([7, 7, 8], dtype=dtype),
          self._runGather(
              np.array([8, 1, 2, 3, 7, 5], dtype=dtype),
              np.array([[4], [4], [0]], np.int32)))

  @test_util.disable_mlir_bridge("Error handling")
  def testEmptyIndicesAndParamstAndEmptyParamsOk(self):
    # [FUNCTION DOC]
    # Skipped for the MLIR bridge (error handling). Checks gather-nd with empty indices on
    # non-empty params, with empty params and empty indices, and with empty params and non-empty
    # indices (result is zeros), asserting the expected empty or zero results.
    with self.session():
      params = np.ones((3, 3), dtype=np.float32)

      indices_empty = np.empty((0, 2), dtype=np.int32)
      gather_nd_ok_val = self._runGather(params, indices_empty)
      self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)

      indices_empty = np.empty((0, 1), dtype=np.int32)
      gather_nd_ok_val = self._runGather(params, indices_empty)
      self.assertAllClose(np.empty((0, 3), dtype=np.float32), gather_nd_ok_val)

      params_empty = np.empty((0, 3), dtype=np.float32)
      indices_empty = np.empty((0, 2), dtype=np.int32)
      gather_nd_ok_val = self._runGather(params_empty, indices_empty)
      self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)

      # Zero sized indices results in a constant of 0
      params_empty = np.empty((0, 3), dtype=np.float32)
      indices_nonempty = np.zeros((1, 2), dtype=np.int32)
      gather_nd_ok_val = self._runGather(params_empty, indices_nonempty)
      self.assertAllEqual(gather_nd_ok_val, np.zeros([3]))

  def testIndexScalar(self):
    # [FUNCTION DOC]
    # Gathers one scalar from a 2-D params using a full-rank index vector; expects 7.
    params = np.array(
        [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
    indices = np.array([4, 1], dtype=np.int32)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(np.array(7), gather_nd_val)

  def testParamsRankLargerThanIndexIndexScalarSlices(self):
    # [FUNCTION DOC]
    # Uses a one-element index vector on 2-D params so a whole row is gathered; expects the
    # row (-7, 7).
    params = np.array(
        [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
    indices = np.array(
        [
            4,
        ], dtype=np.int32)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(np.array([-7, 7]), gather_nd_val)

  def testParamsRankLargerThanIndexSlices(self):
    # [FUNCTION DOC]
    # Gathers three rows from 2-D params with a list of single-element indices; expects the
    # matching rows in order.
    params = np.array(
        [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
    indices = np.array([[4], [4], [0]], np.int32)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(np.array([[-7, 7], [-7, 7], [-8, 8]]), gather_nd_val)

  def testHigherRankParamsLargerThanIndexSlices(self):
    # [FUNCTION DOC]
    # Gathers slices along the first axis of a 3-D params using single-element indices; expects
    # the same as numpy fancy indexing on the first axis.
    params = np.array(
        [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
         [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
        dtype=np.float32).T
    indices = np.array([[4], [4], [0]], np.int32)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(params[[4, 4, 0]], gather_nd_val)

  def testEmptyIndicesLastRankMeansCopyEntireTensor(self):
    # [FUNCTION DOC]
    # Uses indices of shape two by zero (last dimension empty) on a 3-D params; expects the
    # whole params tensor copied once per index row, stacked into a new leading axis.
    params = np.array(
        [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
         [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
        dtype=np.float32).T
    indices = np.array([[], []], dtype=np.int32)  # Size (2, 0)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(
        np.vstack((params[np.newaxis, :], params[np.newaxis, :])),
        gather_nd_val)

  def testHigherRankParamsAndIndicesLargerThanIndexSlices(self):
    # [FUNCTION DOC]
    # Uses 3-D indices with single-element last dimension on 3-D params; expects the gathered
    # slices reshaped to the index batch shape followed by the slice shape.
    params = np.array(
        [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
         [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
        dtype=np.float32).T
    indices = np.array([[[3], [2], [1]], [[4], [4], [0]]], np.int32)
    gather_nd_val = self._runGather(params, indices)
    self.assertAllEqual(params[[3, 2, 1, 4, 4, 0]].reshape(2, 3, 2, 2),
                        gather_nd_val)

  def testHigherRankParams(self):
    # [FUNCTION DOC]
    # Random 5-D params with 2000 random full-rank indices (random data, unseeded here);
    # expects equality with numpy tuple-of-columns indexing.
    shape = (10, 20, 5, 1, 17)
    params = np.random.rand(*shape).astype(np.float32)
    indices = np.vstack(
        [np.random.randint(0, s, size=2000, dtype=np.int32) for s in shape]).T
    gather_nd_val = self._runGather(params, indices)

    expected = params[tuple(indices.T)]
    self.assertAllEqual(expected, gather_nd_val)

  def testHigherRankParamsAndIndices(self):
    # [FUNCTION DOC]
    # Same as the higher-rank params test but the 2000 index rows are reshaped into a 4-D
    # indices tensor; expects the numpy result reshaped to the batch shape.
    shape = (10, 20, 5, 1, 17)
    params = np.random.rand(*shape).astype(np.float32)
    indices = np.vstack(
        [np.random.randint(0, s, size=2000, dtype=np.int32) for s in shape]).T
    indices_reshaped = indices.reshape([10, 10, 20, 5])
    gather_nd_val = self._runGather(params, indices_reshaped)
    expected = params[tuple(indices.T)]
    self.assertAllEqual(expected.reshape([10, 10, 20]), gather_nd_val)

  def testIgnoreBadIndices(self):
    # [FUNCTION DOC]
    # Runs gather-nd with the ignore policy on 3-D params and indices that include negative and
    # out-of-range entries; expects valid indices to return their slices and bad ones to return
    # zeros.
    shape = (3, 4, 5)
    params = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    indices = np.array([[[0, 0], [-1, 3]], [[2, 4], [2, 3]]], dtype=np.int32)
    gather_nd_val = self._runGather(
        params, indices, bad_indices_policy="IGNORE"
    )
    expected = np.array(
        [
            [[0, 1, 2, 3, 4], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [55, 56, 57, 58, 59]],
        ],
        dtype=np.int32,
    )
    self.assertAllEqual(expected, gather_nd_val)

  def testIgnoreBadIndicesNoFeatureDim(self):
    # [FUNCTION DOC]
    # Ignore policy on 2-D params with full-rank indices (scalar results); bad indices yield
    # zero, valid ones yield the element (expected zeros and 11 in the second row).
    shape = (3, 4)
    params = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    indices = np.array([[[0, 0], [-1, 3]], [[2, 4], [2, 3]]], dtype=np.int32)
    gather_nd_val = self._runGather(
        params, indices, bad_indices_policy="IGNORE"
    )
    expected = np.array(
        [[0, 0], [0, 11]],
        dtype=np.int32,
    )
    self.assertAllEqual(expected, gather_nd_val)


if __name__ == "__main__":
  test.main()
