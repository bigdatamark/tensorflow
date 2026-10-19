# [MODULE DOC]
# Unit tests for the DynamicPartition op (data_flow_ops.dynamic_partition) and its gradient
# (the data-flow gradient module is imported for its registration side effect). Covers one- and
# two-dimensional data, large inputs, complex and empty data, scalar and higher-rank partition
# indices, out-of-range index errors, wrong index dimensions and multi-GPU placement.
# Not a helper module.
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
"""Tests for the DynamicPartition op."""

import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_impl
import tensorflow.python.ops.data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class DynamicPartitionTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (also parameterized over dtypes for one test) for dynamic_partition. Each test builds
  # constant data and integer partition indices, partitions into a given number of outputs,
  # evaluates them and compares per-partition contents and static shapes. Holds no instance state.

  @parameterized.parameters(dtypes.float32, dtypes.bfloat16)
  @test_util.run_deprecated_v1
  def testSimpleOneDimensional(self, dtype):
    # [FUNCTION DOC]
    # Test, parameterized over float32 and bfloat16: partitions a 6-element vector into 4 parts and
    # checks each part's values, and that each output has static shape of one unknown dimension.
    # Input: the dtype used for the data.
    with self.session():
      data = constant_op.constant([0, 13, 2, 39, 4, 17], dtype=dtype)
      indices = constant_op.constant([0, 0, 2, 3, 2, 1])
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=4)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(4, len(partition_vals))
    self.assertAllEqual([0, 13], partition_vals[0])
    self.assertAllEqual([17], partition_vals[1])
    self.assertAllEqual([2, 4], partition_vals[2])
    self.assertAllEqual([39], partition_vals[3])
    # Vector data input to DynamicPartition results in
    # `num_partitions` vectors of unknown length.
    self.assertEqual([None], partitions[0].get_shape().as_list())
    self.assertEqual([None], partitions[1].get_shape().as_list())
    self.assertEqual([None], partitions[2].get_shape().as_list())
    self.assertEqual([None], partitions[3].get_shape().as_list())

  @test_util.run_deprecated_v1
  def testSimpleTwoDimensional(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment in the body says "vector data input" but the data is a matrix
    # of 6 rows and 3 columns; the assertions (unknown rows, 3 columns) are correct for a matrix.
    # Test: partitions a 6x3 float matrix into 4 parts, checks the rows of each part and that each
    # output has static shape of unknown rows by 3 columns.
    with self.session():
      data = constant_op.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                                   [12, 13, 14], [15, 16, 17]],
                                  dtype=dtypes.float32)
      indices = constant_op.constant([0, 0, 2, 3, 2, 1])
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=4)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(4, len(partition_vals))
    self.assertAllEqual([[0, 1, 2], [3, 4, 5]], partition_vals[0])
    self.assertAllEqual([[15, 16, 17]], partition_vals[1])
    self.assertAllEqual([[6, 7, 8], [12, 13, 14]], partition_vals[2])
    self.assertAllEqual([[9, 10, 11]], partition_vals[3])
    # Vector data input to DynamicPartition results in
    # `num_partitions` matrices with an unknown number of rows, and 3 columns.
    self.assertEqual([None, 3], partitions[0].get_shape().as_list())
    self.assertEqual([None, 3], partitions[1].get_shape().as_list())
    self.assertEqual([None, 3], partitions[2].get_shape().as_list())
    self.assertEqual([None, 3], partitions[3].get_shape().as_list())

  def testLargeOneDimensional(self):
    # [FUNCTION DOC]
    # Test: partitions 100000 consecutive numbers into even and odd indices (2 partitions) and
    # checks the two outputs equal the expected even and odd lists.
    num = 100000
    data_list = [x for x in range(num)]
    indices_list = [x % 2 for x in range(num)]
    part1 = [x for x in range(num) if x % 2 == 0]
    part2 = [x for x in range(num) if x % 2 == 1]
    with self.session():
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=2)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(2, len(partition_vals))
    self.assertAllEqual(part1, partition_vals[0])
    self.assertAllEqual(part2, partition_vals[1])

  def testLargeTwoDimensional(self):
    # [FUNCTION DOC]
    # Test: partitions a 100000 by 100 matrix (each row filled with its row number) into 97 parts
    # using the row number squared modulo 97 as index; every part must equal the expected rows
    # (reshaped so empty parts have 100 columns).
    rows = 100000
    cols = 100
    data_list = [None] * rows
    for i in range(rows):
      data_list[i] = [i for _ in range(cols)]
    num_partitions = 97
    indices_list = [(i**2) % num_partitions for i in range(rows)]
    parts = [[] for _ in range(num_partitions)]
    for i in range(rows):
      parts[(i**2) % num_partitions].append(data_list[i])
    with self.session():
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=num_partitions)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(num_partitions, len(partition_vals))
    for i in range(num_partitions):
      # reshape because of empty parts
      parts_np = np.array(parts[i], dtype=np.float64).reshape(-1, cols)
      self.assertAllEqual(parts_np, partition_vals[i])

  def testSimpleComplex(self):
    # [FUNCTION DOC]
    # Test: partitions four complex64 numbers into 2 parts using indices 1,0,1,0 and checks the
    # values of both parts.
    data_list = [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j]
    indices_list = [1, 0, 1, 0]
    with self.session():
      data = constant_op.constant(data_list, dtype=dtypes.complex64)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=2)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(2, len(partition_vals))
    self.assertAllEqual([3 + 4j, 7 + 8j], partition_vals[0])
    self.assertAllEqual([1 + 2j, 5 + 6j], partition_vals[1])

  def testScalarPartitions(self):
    # [FUNCTION DOC]
    # Test: the partition index is a plain scalar (3) with 4 partitions; the whole data vector goes
    # to partition 3 (shaped as one row of 4) and the other three outputs are empty with 4 columns.
    data_list = [10, 13, 12, 11]
    with self.session():
      data = constant_op.constant(data_list, dtype=dtypes.float64)
      indices = 3
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=4)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(4, len(partition_vals))
    self.assertAllEqual(
        np.array([], dtype=np.float64).reshape(-1, 4), partition_vals[0])
    self.assertAllEqual(
        np.array([], dtype=np.float64).reshape(-1, 4), partition_vals[1])
    self.assertAllEqual(
        np.array([], dtype=np.float64).reshape(-1, 4), partition_vals[2])
    self.assertAllEqual(
        np.array([10, 13, 12, 11], dtype=np.float64).reshape(-1, 4),
        partition_vals[3])

  @test_util.run_deprecated_v1
  def testHigherRank(self):
    # [FUNCTION DOC]
    # Test (graph mode): with a fixed random seed, random partition indices of several ranks and
    # extra data dimensions are used with 2 or 3 partitions. Each output must equal the data selected
    # by the mask for that partition. Also checks gradients: the index gradient is nothing, and
    # the data gradient of scaled outputs equals the data scaled by 7.
    np.random.seed(7)
    with self.session() as sess:
      for n in 2, 3:
        for shape in (4,), (4, 5), (4, 5, 2):
          partitions = np.random.randint(n, size=np.prod(shape)).reshape(shape)
          for extra_shape in (), (6,), (6, 7):
            data = np.random.randn(*(shape + extra_shape))
            partitions_t = constant_op.constant(partitions, dtype=dtypes.int32)
            data_t = constant_op.constant(data)
            outputs = data_flow_ops.dynamic_partition(
                data_t, partitions_t, num_partitions=n)
            self.assertEqual(n, len(outputs))
            outputs_val = self.evaluate(outputs)
            for i, output in enumerate(outputs_val):
              self.assertAllEqual(output, data[partitions == i])

            # Test gradients
            outputs_grad = [7 * output for output in outputs_val]
            grads = gradients_impl.gradients(outputs, [data_t, partitions_t],
                                             outputs_grad)
            self.assertEqual(grads[1], None)  # Partitions has no gradients
            self.assertAllEqual(7 * data, sess.run(grads[0]))

  def testEmptyParts(self):
    # [FUNCTION DOC]
    # Test: with 4 partitions but indices only 1 and 3, partitions 0 and 2 must be empty and 1 and 3
    # hold the expected values.
    data_list = [1, 2, 3, 4]
    indices_list = [1, 3, 1, 3]
    with self.session():
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=4)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(4, len(partition_vals))
    self.assertAllEqual([], partition_vals[0])
    self.assertAllEqual([1, 3], partition_vals[1])
    self.assertAllEqual([], partition_vals[2])
    self.assertAllEqual([2, 4], partition_vals[3])

  def testEmptyDataTwoDimensional(self):
    # [FUNCTION DOC]
    # Test: data of two empty rows with indices 0 and 1 and 3 partitions; checks the resulting
    # partitions (empty rows kept for 0 and 1, an empty 0 by 0 array for partition 2).
    data_list = [[], []]
    indices_list = [0, 1]
    with self.session():
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=3)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(3, len(partition_vals))
    self.assertAllEqual([[]], partition_vals[0])
    self.assertAllEqual([[]], partition_vals[1])
    self.assertAllEqual(
        np.array([], dtype=np.float64).reshape(0, 0), partition_vals[2])

  def testEmptyPartitions(self):
    # [FUNCTION DOC]
    # Test: empty data and indices with 2 partitions yield two empty outputs.
    data_list = []
    indices_list = []
    with self.session():
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=2)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(2, len(partition_vals))
    self.assertAllEqual([], partition_vals[0])
    self.assertAllEqual([], partition_vals[1])

  @unittest.skip("Fails on windows.")
  def testGPUTooManyParts(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unconditionally skipped (decorated as failing on Windows), so the body never runs;
    # it also silently passes (returns early) when no GPU is available.
    # Test intent: on GPU, indices beyond the partition count are discarded without error. Uses
    # 2 partitions and indices larger than 2 and expects only the in-range entries to appear.
    # This test only makes sense on the GPU. There we do not check
    # for errors. In this case, we should discard all but the first
    # num_partitions indices.
    if not test.is_gpu_available():
      return

    data_list = [1, 2, 3, 4, 5, 6]
    indices_list = [6, 5, 4, 3, 1, 0]
    with self.session():
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=2)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(2, len(partition_vals))
    self.assertAllEqual([6], partition_vals[0])
    self.assertAllEqual([5], partition_vals[1])

  @unittest.skip("Fails on windows.")
  def testGPUPartsTooLarge(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unconditionally skipped (decorated as failing on Windows), so the body never runs;
    # it also silently passes (returns early) when no GPU is available.
    # Test intent: on GPU, index values larger than the partition count are dropped without error;
    # with 5 partitions only the in-range entries land in outputs.
    # This test only makes sense on the GPU. There we do not check
    # for errors. In this case, we should discard all the values
    # larger than num_partitions.
    if not test.is_gpu_available():
      return

    data_list = [1, 2, 3, 4, 5, 6]
    indices_list = [10, 11, 2, 12, 0, 1000]
    with self.session():
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=5)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(5, len(partition_vals))
    self.assertAllEqual([5], partition_vals[0])
    self.assertAllEqual([], partition_vals[1])
    self.assertAllEqual([3], partition_vals[2])
    self.assertAllEqual([], partition_vals[3])
    self.assertAllEqual([], partition_vals[4])

  @unittest.skip("Fails on windows.")
  def testGPUAllIndicesBig(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unconditionally skipped (decorated as failing on Windows), so the body never runs;
    # it also silently passes (returns early) when no GPU is available.
    # Test intent: on GPU, when every index is out of range, all 40 outputs are empty.
    # This test only makes sense on the GPU. There we do not check
    # for errors. In this case, we should discard all the values
    # and have an empty output.
    if not test.is_gpu_available():
      return

    data_list = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    indices_list = [90, 70, 60, 100, 110, 40]
    with self.session():
      data = constant_op.constant(data_list, dtype=dtypes.float32)
      indices = constant_op.constant(indices_list, dtype=dtypes.int32)
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=40)
      partition_vals = self.evaluate(partitions)

    self.assertEqual(40, len(partition_vals))
    for i in range(40):
      self.assertAllEqual([], partition_vals[i])

  @test_util.run_deprecated_v1
  @test_util.no_xla_auto_jit("xla doesn't raise out-of-range exceptions")
  def testErrorIndexOutOfRange(self):
    # [FUNCTION DOC]
    # Test (graph mode, CPU only, skipped under XLA auto-jit): an index of 99 with 4 partitions must
    # raise an op error saying the partition entry is not in the range 0 to 4.
    # GPU kernels don't throw exceptions.
    with self.cached_session(use_gpu=False):
      data = constant_op.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                                   [12, 13, 14]])
      indices = constant_op.constant([0, 2, 99, 2, 2])
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=4)
      with self.assertRaisesOpError(r"partitions\[2\] = 99 is not in \[0, 4\)"):
        self.evaluate(partitions)

  @test_util.run_deprecated_v1
  @test_util.no_xla_auto_jit("xla doesn't raise out-of-range exceptions")
  def testScalarIndexOutOfRange(self):
    # [FUNCTION DOC]
    # Test (graph mode, CPU only, skipped under XLA auto-jit): a scalar index 17 with 7 partitions
    # raises an op error saying partitions = 17 is not in range 0 to 7.
    # GPU kernels don't throw exceptions.
    with self.cached_session(use_gpu=False):
      bad = 17
      data = np.zeros(5)
      partitions = data_flow_ops.dynamic_partition(data, bad, num_partitions=7)
      with self.assertRaisesOpError(r"partitions = 17 is not in \[0, 7\)"):
        self.evaluate(partitions)

  @test_util.run_deprecated_v1
  @test_util.no_xla_auto_jit("xla doesn't raise out-of-range exceptions")
  def testHigherRankIndexOutOfRange(self):
    # [FUNCTION DOC]
    # Test (graph mode, CPU only, skipped under XLA auto-jit): with a 2x3 placeholder of indices fed
    # at run time, each single position in turn is set to 17 (others zero) and running must raise an
    # op error naming that position and the range 0 to 7.
    # GPU kernels don't throw exceptions.
    with self.cached_session(use_gpu=False) as sess:
      shape = (2, 3)
      indices = array_ops.placeholder(shape=shape, dtype=np.int32)
      data = np.zeros(shape + (5,))
      partitions = data_flow_ops.dynamic_partition(
          data, indices, num_partitions=7)
      for i in range(2):
        for j in range(3):
          bad = np.zeros(shape, dtype=np.int32)
          bad[i, j] = 17
          with self.assertRaisesOpError(
              r"partitions\[%d,%d\] = 17 is not in \[0, 7\)" % (i, j)):
            sess.run(partitions, feed_dict={indices: bad})

  @test_util.run_deprecated_v1
  def testErrorWrongDimsIndices(self):
    # [FUNCTION DOC]
    # Test (graph mode): indices whose shape is not a prefix of the data shape (2x1 indices vs 3x1
    # data) must raise a value error when the op is built.
    data = constant_op.constant([[0], [1], [2]])
    indices = constant_op.constant([[0], [0]])
    with self.assertRaises(ValueError):
      data_flow_ops.dynamic_partition(data, indices, num_partitions=4)

  #  see https://github.com/tensorflow/tensorflow/issues/17106
  def testCUBBug(self):
    # [FUNCTION DOC]
    # Test: regression for a GPU CUB library issue (public issue 17106). Partitions 3072 random
    # values into 16 parts with a hand-built, unevenly sized index list and asserts the last part
    # has 192 elements. The index list length is also checked against the data length.
    x = constant_op.constant(np.random.randn(3072))
    inds = [0] * 189 + [1] * 184 + [2] * 184 + [3] * 191 + [4] * 192 + [
        5
    ] * 195 + [6] * 195
    inds += [7] * 195 + [8] * 188 + [9] * 195 + [10] * 188 + [11] * 202 + [
        12
    ] * 194
    inds += [13] * 194 + [14] * 194 + [15] * 192
    self.assertEqual(len(inds), x.shape[0])
    partitioned = data_flow_ops.dynamic_partition(x, inds, 16)
    with self.cached_session():
      res = self.evaluate(partitioned)
    self.assertEqual(res[-1].shape[0], 192)

  #  see https://github.com/tensorflow/tensorflow/issues/42500
  def testMultiGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when no GPU is present the device list is empty and the test asserts nothing.
    # Test: regression for a multi-GPU issue (public issue 42500). On each logical GPU device
    # partitions 1000 zeros into 10 parts by index modulo 10 and, if devices exist, checks every
    # result has 10 parts of 100 zeros.
    device_list = config.list_logical_devices("GPU")
    results = []
    for device in device_list:
      with ops.device(device.name):
        data = constant_op.constant(np.zeros((1000,)))
        partitions = constant_op.constant(np.arange(1000, dtype=np.int32) % 10)
        result = data_flow_ops.dynamic_partition(data, partitions, 10)
        results.append(self.evaluate(result))
    if device_list:
      self.assertAllEqual(results, np.zeros((len(device_list), 10, 100)))


if __name__ == "__main__":
  test.main()
