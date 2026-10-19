# [MODULE DOC]
# Unit tests for the batch gather operation in the array ops module. They exercise gathering
# slices along the leading batch dimensions for several value types (64-bit integer, 32-bit float,
# complex), 32/64-bit index types, higher-rank inputs, strings, unknown-shape indices, bad indices
# on the CPU and empty slices. Defines one test case class and a module-level tuple of the value
# types under test. No import-time side effects; running the file as a script launches the
# TensorFlow test runner.
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
"""Tests for tensorflow.ops.tf.gather."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

_TEST_TYPES = (dtypes.int64, dtypes.float32,
               dtypes.complex64, dtypes.complex128)


class GatherTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the batch gather op across value types, index types, ranks, strings,
  # unknown shapes, invalid indices and empty slices. Combines the TensorFlow test base class with
  # the parameterized-test mixin so some tests run once per index type. Holds no state; one small
  # helper builds numpy inputs.

  def _buildParams(self, data, dtype):
    # [FUNCTION DOC]
    # HELPER: builds numpy test inputs for the gather tests.
    # Converts the numpy array data to the numpy equivalent of the given TensorFlow dtype. For
    # complex dtypes, adds ten times the data as the imaginary part so that results can be
    # distinguished from real-only values. Returns the converted array; no side effects.
    data = data.astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  @parameterized.parameters(dtypes.int32, dtypes.int64)
  def testSimpleGather(self, indices_dtype):
    # [FUNCTION DOC]
    # Parameterized by index dtype (32 or 64-bit integer). Gathers indices 3 and 4 from a 1-D
    # array for every value type in the test list and asserts that the result equals the
    # expected values [3, 7] and that the static shape matches. Returns nothing.
    data = np.array([0, 1, 2, 3, 7, 5, 8, 9, 10, 11, 15, 13])
    indices = [3, 4]
    with self.session():
      for dtype in _TEST_TYPES:
        params_np = self._buildParams(data, dtype)
        params = constant_op.constant(params_np)
        indices_tf = constant_op.constant(indices, dtype=indices_dtype)
        gather_t = array_ops.batch_gather(params, indices_tf)
        expected_result = np.array([3, 7])
        np_val = self._buildParams(expected_result, dtype)
        gather_val = self.evaluate(gather_t)
        self.assertAllEqual(np_val, gather_val)
        self.assertEqual(np_val.shape, gather_t.get_shape())

  @parameterized.parameters(dtypes.int32, dtypes.int64)
  def test2DArray(self, indices_dtype):
    # [FUNCTION DOC]
    # Parameterized by index dtype. Batch-gathers one column per row from a 2x6 array with
    # indices of shape 2x1 for each value type, and asserts the values are [[3], [15]] with
    # matching static shape. Returns nothing.
    data = np.array([[0, 1, 2, 3, 7, 5], [8, 9, 10, 11, 15, 13]])
    indices = [[3], [4]]
    with self.session():
      for dtype in _TEST_TYPES:
        params_np = self._buildParams(data, dtype)
        params = constant_op.constant(params_np)
        indices_tf = constant_op.constant(indices, dtype=indices_dtype)
        gather_t = array_ops.batch_gather(params, indices_tf)
        expected_result = np.array([[3], [15]])
        np_val = self._buildParams(expected_result, dtype)
        gather_val = self.evaluate(gather_t)
        self.assertAllEqual(np_val, gather_val)
        self.assertEqual(np_val.shape, gather_t.get_shape())

  def testHigherRank(self):
    # [FUNCTION DOC]
    # Gathers along the last axis of a rank-3 array using rank-3 indices, for each value type,
    # and asserts the gathered values and the static shape equal the hand-computed expectation.
    # Returns nothing.
    data = np.array([[[0, 1, 2], [3, 7, 5]], [[8, 9, 10], [11, 15, 13]]])
    indices = [[[2, 0], [1, 2]], [[2, 0], [0, 1]]]
    with self.session():
      for dtype in _TEST_TYPES:
        params_np = self._buildParams(data, dtype)
        params = constant_op.constant(params_np)
        indices_tf = constant_op.constant(indices)
        gather_t = array_ops.batch_gather(params, indices_tf)
        gather_val = self.evaluate(gather_t)
        expected_result = np.array([[[2, 0], [7, 5]], [[10, 8], [11, 15]]])
        np_val = self._buildParams(expected_result, dtype)
        self.assertAllEqual(np_val, gather_val)
        self.assertEqual(np_val.shape, gather_t.get_shape())

  def testString(self):
    # [FUNCTION DOC]
    # Gathers row 1 from a 2x2 numpy array of byte strings and asserts the result is the second
    # row wrapped in a batch of one. Returns nothing.
    params = np.array([[b"asdf", b"zxcv"], [b"qwer", b"uiop"]])
    with self.cached_session():
      indices_tf = constant_op.constant([1])
      self.assertAllEqual(
          [[b"qwer", b"uiop"]],
          self.evaluate(array_ops.batch_gather(params, indices_tf)))

  def testUnknownIndices(self):
    # [FUNCTION DOC]
    # In a fresh graph, uses a placeholder index tensor of unknown shape (rank 2) with a 1x3
    # constant params and asserts the inferred static shape of the gather result is a known
    # leading dimension of 1 and an unknown second dimension. Returns nothing.
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      params = constant_op.constant([[0, 1, 2]])
      indices = array_ops.placeholder(dtypes.int32, shape=[None, None])
      gather_t = array_ops.batch_gather(params, indices)
      self.assertEqual([1, None], gather_t.get_shape().as_list())

  @test_util.disable_xla("Cannot force cpu placement for xla_gpu test")
  def testBadIndicesCPU(self):
    # [FUNCTION DOC]
    # Skipped under XLA. Pinned to the CPU device, gathers with an out-of-range index 7 from a
    # 2x3 params and asserts an op error stating that the index is not in the valid range.
    # Returns nothing.
    with ops.device_v2("cpu:0"):
      params = [[0, 1, 2], [3, 4, 5]]
      with self.assertRaisesOpError(r"indices\[0\] = 7 is not in \[0, 2\)"):
        self.evaluate(array_ops.batch_gather(params, [7]))

  def testEmptySlices(self):
    # [FUNCTION DOC]
    # For each value type and both index types, gathers two entries from a params array of shape
    # 7x0x0 and asserts the result equals zeros of shape 2x0x0. Returns nothing.
    with self.session():
      for dtype in _TEST_TYPES:
        for itype in np.int32, np.int64:
          params = np.zeros((7, 0, 0), dtype=dtype.as_numpy_dtype)
          indices = np.array([3, 4], dtype=itype)
          self.assertAllEqual(
              self.evaluate(array_ops.batch_gather(params, indices)),
              np.zeros((2, 0, 0)))

if __name__ == "__main__":
  test.main()
