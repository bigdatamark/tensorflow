# [MODULE DOC]
# Unit tests for the cross_device_utils module of the distribute package.
# Covers aggregation, division and copying of dense tensors and indexed slices, and the
# size-based grouping of values into packs. Contains two test classes; copy tests require
# one GPU. Runs under the standard TensorFlow test main entry point; no import-time side effects.

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
"""Tests for cross_device_utils."""

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class IndexedSlicesUtilsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the helpers in cross_device_utils that aggregate, divide, identify and copy
  # dense tensors and indexed slices. Provides a shared helper comparing two values by
  # converting both to dense tensors.

  def _assert_values_equal(self, left, right):
    # [FUNCTION DOC]
    # HELPER: shared assertion helper for the tests in this class.
    # Converts both arguments (tensors or indexed slices) to dense tensors, evaluates them and
    # asserts the resulting arrays are equal. Returns nothing; fails the test on mismatch.
    self.assertAllEqual(
        self.evaluate(ops.convert_to_tensor(left)),
        self.evaluate(ops.convert_to_tensor(right)))

  @test_util.run_in_graph_and_eager_modes
  def testAggregateTensors(self):
    # [FUNCTION DOC]
    # Aggregating two dense tensors of equal shape should yield their elementwise sum.
    # Runs in graph and eager modes; asserts the result equals the expected total.
    t0 = constant_op.constant([[1., 2.], [0, 0], [3., 4.]])
    t1 = constant_op.constant([[0., 0.], [5, 6], [7., 8.]])
    total = constant_op.constant([[1., 2.], [5, 6], [10., 12.]])
    result = cross_device_utils.aggregate_tensors_or_indexed_slices([t0, t1])
    self._assert_values_equal(total, result)

  @test_util.run_in_graph_and_eager_modes
  def testAggregateIndexedSlices(self):
    # [FUNCTION DOC]
    # Aggregating two indexed-slices values should produce an indexed-slices result whose dense
    # form equals the elementwise sum. Runs in graph and eager modes; asserts both type and value.
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(
        constant_op.constant([[0., 0.], [5, 6], [7., 8.]]))
    total = constant_op.constant([[1., 2.], [5, 6], [10., 12.]])
    result = cross_device_utils.aggregate_tensors_or_indexed_slices([t0, t1])
    self.assertIsInstance(result, indexed_slices.IndexedSlices)
    self._assert_values_equal(total, result)

  @test_util.run_in_graph_and_eager_modes
  def testDivideTensor(self):
    # [FUNCTION DOC]
    # Dividing a dense tensor by an integer count (2) should divide every element by that count.
    # Runs in graph and eager modes; asserts the result equals the expected halved tensor.
    t = constant_op.constant([[1., 2.], [0, 0], [3., 4.]])
    n = 2
    expected = constant_op.constant([[0.5, 1.], [0, 0], [1.5, 2.]])
    result = cross_device_utils.divide_by_n_tensors_or_indexed_slices(t, n)
    self._assert_values_equal(expected, result)

  @test_util.run_in_graph_and_eager_modes
  def testDivideIndexedSlices(self):
    # [FUNCTION DOC]
    # Dividing an indexed-slices value by a count (2) should keep the indexed-slices type and
    # divide all values. Runs in graph and eager modes; asserts type and dense value.
    t = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    n = 2
    expected = constant_op.constant([[0.5, 1.], [0, 0], [1.5, 2.]])
    result = cross_device_utils.divide_by_n_tensors_or_indexed_slices(t, n)
    self.assertIsInstance(result, indexed_slices.IndexedSlices)
    self._assert_values_equal(expected, result)

  @test_util.run_in_graph_and_eager_modes
  def testIsIndexedSlices(self):
    # [FUNCTION DOC]
    # Builds an indexed-slices value from a dense tensor and asserts the type-check helper
    # reports true. Runs in graph and eager modes.
    t = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    self.assertTrue(cross_device_utils.is_indexed_slices(t))

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      required_gpus=1))
  def testCopyTensor(self):
    # [FUNCTION DOC]
    # Creates a dense tensor on the CPU and copies it to the first GPU. Requires one GPU; runs in
    # graph and eager modes. Asserts values are preserved and the result's device resolves to
    # the destination device.
    with ops.device("/cpu:0"):
      t = constant_op.constant([[1., 2.], [0, 0], [3., 4.]])
    destination = "/gpu:0"
    result = cross_device_utils.copy_tensor_or_indexed_slices_to_device(
        t, destination)

    self._assert_values_equal(t, result)
    self.assertEqual(device_util.resolve(destination),
                     device_util.resolve(result.device))

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      required_gpus=1))
  def testCopyIndexedSlices(self):
    # [FUNCTION DOC]
    # Creates an indexed-slices value on the CPU and copies it to the first GPU. Requires one
    # GPU; runs in graph and eager modes. Asserts the result is still indexed slices, values are
    # equal, and the device resolves to the destination.
    with ops.device("/cpu:0"):
      t = math_ops._as_indexed_slices(
          constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    destination = "/gpu:0"
    result = cross_device_utils.copy_tensor_or_indexed_slices_to_device(
        t, destination)

    self.assertIsInstance(result, indexed_slices.IndexedSlices)
    self._assert_values_equal(t, result)
    self.assertEqual(
        device_util.resolve(destination), device_util.resolve(result.device))

  @combinations.generate(
      combinations.combine(mode=["graph", "eager"], required_gpus=1))
  def testCopyIndexedSlicesNoDenseShape(self):
    # [FUNCTION DOC]
    # Copies an indexed-slices value that has no dense shape (only indices and values) from the
    # CPU to the first GPU. Requires one GPU; graph and eager modes. Asserts indices and values
    # are preserved, type is indexed slices, and the device matches the destination.
    with ops.device("/cpu:0"):
      t = indexed_slices.IndexedSlices(
          indices=array_ops.identity([0]), values=array_ops.identity([1.]))
    destination = "/gpu:0"
    result = cross_device_utils.copy_tensor_or_indexed_slices_to_device(
        t, destination)

    self.assertIsInstance(result, indexed_slices.IndexedSlices)
    self.assertAllEqual(t.indices, result.indices)
    self.assertAllEqual(t.values, result.values)
    self.assertEqual(
        device_util.resolve(destination), device_util.resolve(result.device))


class GroupBySizeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the size-based grouping of tensors into packs performed by the group-by-size
  # helper in cross_device_utils. No shared fixtures.

  def testPreferLargerPack(self):
    # [FUNCTION DOC]
    # Groups four tensors of differing byte sizes with a 200-byte pack target. Asserts every pack
    # except the last reaches at least the target, giving two packs: the first holds the first
    # three tensors in order and the second holds the last tensor.
    # Each packs except the last one should be equal or larger than
    # bytes_per_pack.
    values = [
        # size = 2 * 4 * 4 * 4 = 128
        array_ops.ones([2, 4, 4], dtype=dtypes.float32),
        # size = 8 * 4 = 32
        array_ops.ones([8], dtype=dtypes.int32),
        # size = 10 * 10 * 8 = 800
        array_ops.ones([10, 10], dtype=dtypes.int64),
        # size = 1 * 4 = 4
        array_ops.ones([1], dtype=dtypes.int32),
    ]
    packs = cross_device_utils.group_by_size(values, bytes_per_pack=200)
    self.assertLen(packs, 2)
    self.assertLen(packs[0], 3)
    self.assertEqual(packs[0][0].shape, [2, 4, 4])
    self.assertEqual(packs[0][1].shape, [8])
    self.assertEqual(packs[0][2].shape, [10, 10])
    self.assertLen(packs[1], 1)
    self.assertEqual(packs[1][0].shape, [1])

  def testZeroBytesPerPack(self):
    # [FUNCTION DOC]
    # With a pack size of zero bytes, grouping should put all tensors into a single pack,
    # preserving order. Asserts one pack with two tensors of the expected shapes.
    values = [
        array_ops.ones([1], dtype=dtypes.float32),
        array_ops.ones([2], dtype=dtypes.float32),
    ]
    packs = cross_device_utils.group_by_size(values, bytes_per_pack=0)
    self.assertLen(packs, 1)
    self.assertLen(packs[0], 2)
    self.assertEqual(packs[0][0].shape, [1])
    self.assertEqual(packs[0][1].shape, [2])

  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Groups a tensor of known shape together with a placeholder of partially unknown shape using
    # a 1-byte pack target. Asserts everything ends up in a single pack equal to the input list.
    def create_placeholder(shape, dtype):
      # [FUNCTION DOC]
      # HELPER: builds the unknown-shape input for testUnknownShape.
      # Creates a placeholder of the given shape (may contain unknown dimensions) and dtype
      # inside a fresh graph and returns it.
      with ops.Graph().as_default():
        return array_ops.placeholder(dtype=dtype, shape=shape)

    values = [
        array_ops.ones([10, 10], dtype=dtypes.float32),
        create_placeholder([None, 10], dtype=dtypes.float32),
    ]
    packs = cross_device_utils.group_by_size(values, bytes_per_pack=1)
    self.assertLen(packs, 1)
    self.assertEqual(packs[0], values)


if __name__ == "__main__":
  test.main()
