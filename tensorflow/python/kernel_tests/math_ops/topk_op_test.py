# [MODULE DOC]
# Tests and a benchmark for the top-k operation of the neural-network ops module. Contains a test
# case that checks values and indices of the k largest entries for sorted and unsorted output,
# different index and k dtypes, large and medium inputs, ties (stability), NaNs, empty inputs,
# error handling and gradients, plus a benchmark case measuring throughput over input sizes, k
# fractions and devices. Imports the neural-network gradient module for its registration side
# effect. Not a helper module.
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
"""Tests for TopK op."""

import itertools
import sys

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class TopKTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the top-k operation. Provides a shared validation helper that runs the op and
  # compares against expected values and indices, and many scenario tests that use it. No
  # per-test fixtures.

  def _validateTopK(
      self,
      inputs,
      k,
      expected_values,
      expected_indices,
      sorted=True,
      index_type=dtypes.int32,
  ):  # pylint: disable=redefined-builtin
    # [FUNCTION DOC]
    # HELPER: shared checker used by the top-k tests. Inputs: the input data (nested list or
    # array), k (integer or scalar tensor), expected values, expected indices, a flag for sorted
    # output (default true), and the index dtype (default 32-bit integer). Runs the op in a cached
    # session, then asserts the index dtype and the shapes of both outputs. For sorted output it
    # asserts values close and indices equal (for non-floating dtypes), while for floating dtypes
    # it only requires values at mismatching indices to be close. For unsorted output it checks
    # each returned index points at the returned value, that equal neighbours have increasing
    # indices, and that the sorted results match the sorted expectations. Returns nothing.
    # KNOWN ISSUE: in the sorted floating-point branch the expected values are indexed with a
    # boolean mask, which fails if the caller passed a plain list instead of an array.
    np_expected_values = np.array(expected_values)
    np_expected_indices = np.array(expected_indices)
    with self.cached_session():
      values_op, indices_op = nn_ops.top_k(
          inputs, k, sorted=sorted, index_type=index_type
      )
      values, indices = self.evaluate([values_op, indices_op])

      self.assertEqual(indices.dtype, index_type)
      self.assertShapeEqual(np_expected_values, values_op)
      self.assertShapeEqual(np_expected_indices, indices_op)

      if sorted:
        self.assertAllClose(np_expected_values, values)
        # Do some special casing of equality of indices: if indices
        # are not the same, but values are floating type, ensure that
        # the values are within epsilon of each other.
        if not np.issubdtype(np_expected_values.dtype, np.floating) and \
            np_expected_values.dtype != dtypes.bfloat16.as_numpy_dtype:
          # Values are not floating point type; check indices exactly
          self.assertAllEqual(np_expected_indices, indices)
        else:
          # Values are floating point; indices may be swapped for
          # values near each other.
          indices_not_equal = np_expected_indices != indices
          if np.any(indices_not_equal):
            values_unsure = values[indices_not_equal]
            expected_values_unsure = expected_values[indices_not_equal]
            self.assertAllClose(expected_values_unsure, values_unsure)
      else:
        np_inputs = np.array(inputs)

        # Check that the indices are valid.
        for result_index, src_index in np.ndenumerate(indices):
          value = values[result_index]
          expected_value = np_inputs[result_index[0], src_index]
          np.testing.assert_almost_equal(value, expected_value)

        # Check that if two elements are equal, the lower-index element appears
        # first.
        shape = values.shape
        for batch_index in range(shape[0]):
          for index in range(shape[1] - 1):
            if np.isclose(values[batch_index, index],
                          values[batch_index, index + 1]):
              self.assertLess(indices[batch_index, index],
                              indices[batch_index, index + 1])

        # Now check the results, ignoring order.
        self.assertAllEqual(np.sort(np_expected_indices), np.sort(indices))
        self.assertAllClose(np.sort(np_expected_values), np.sort(values))

  def testTop1(self):
    # [FUNCTION DOC]
    # Test: top one of two rows of four floats (second row has a tie) must give the maximum and
    # its first index per row. Returns nothing.
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.3, 0.2]]
    self._validateTopK(inputs, 1, [[0.4], [0.3]], [[3], [1]])

  def testTop2(self):
    # [FUNCTION DOC]
    # Test: top two entries of two rows of four floats must be returned in descending order with
    # the matching indices. Returns nothing.
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2]]
    self._validateTopK(inputs, 2, [[0.4, 0.3], [0.4, 0.3]], [[3, 1], [2, 1]])

  def testOutputIndexType(self):
    # [FUNCTION DOC]
    # Test: repeats the top-two check requesting 16-, 32- and 64-bit integer index outputs; the
    # validator asserts the returned index dtype equals the request. Returns nothing.
    for index_type in [dtypes.int16, dtypes.int32, dtypes.int64]:
      inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2]]
      self._validateTopK(
          inputs,
          2,
          [[0.4, 0.3], [0.4, 0.3]],
          [[3, 1], [2, 1]],
          index_type=index_type,
      )

  def testKType(self):
    # [FUNCTION DOC]
    # Test: passes k as a scalar constant tensor of 32-, 64- and 16-bit integer dtype and expects
    # the same top-two result each time. Returns nothing.
    for ktype in [dtypes.int32, dtypes.int64, dtypes.int16]:
      inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2]]
      self._validateTopK(
          inputs,
          constant_op.constant(2, dtype=ktype),
          [[0.4, 0.3], [0.4, 0.3]],
          [[3, 1], [2, 1]],
      )

  def testTop3(self):
    # [FUNCTION DOC]
    # Test: for odd k from 3 to 9 and vector lengths from 512 up to about 12000 in steps of 512,
    # shuffles evenly spaced doubles and checks top-k against values and indices from NumPy
    # sorting. Uses global random state. Returns nothing.
    for k in range(3, 11, 2):
      for dim in range(512, 12288, 512):
        inputs = np.random.permutation(
            np.linspace(0, 100, dim, dtype=np.float64))
        indices = np.argsort(-inputs)[:k]
        values = -np.sort(-inputs)[:k]
        self._validateTopK(inputs, k, values, indices)

  def testTop1AllNan(self):
    # [FUNCTION DOC]
    # Test: a two-by-two input of all NaN with k of one must return NaN values and index zero in
    # each row. Returns nothing.
    inputs = [[np.nan, np.nan], [np.nan, np.nan]]
    self._validateTopK(inputs, 1, [[np.nan], [np.nan]], [[0], [0]])

  def _testLargeSort(self, dtype):
    # [FUNCTION DOC]
    # HELPER: for testLargeSort. Builds a shuffled 10 by 5000 array of the given numpy dtype and
    # checks that top-k with k equal to the row length gives a full descending sort (reference
    # values and indices from NumPy). Uses random state. Returns nothing.
    b = 10
    n = 5000
    inputs = np.random.permutation(
        np.linspace(0, 100, b * n, dtype=dtype)).reshape(b, n)
    indices = np.argsort(-inputs, axis=1)
    values = -np.sort(-inputs, axis=1)
    self._validateTopK(inputs, n, values, indices)

  def testLargeSort(self):
    # [FUNCTION DOC]
    # Test: runs the large full-sort helper for single, half and bfloat16 precision. Returns
    # nothing.
    self._testLargeSort(np.float32)
    self._testLargeSort(np.float16)
    self._testLargeSort(dtypes.bfloat16.as_numpy_dtype)

  def _testLargeTopK(self, dtype):
    # [FUNCTION DOC]
    # HELPER: for testLargeTopK. Like the large sort helper but requests k equal to one less than
    # the row length of a shuffled 10 by 5000 array of the given numpy dtype, comparing with the
    # first k NumPy-sorted entries. Returns nothing.
    b = 10
    n = 5000
    k = n - 1
    inputs = np.random.permutation(
        np.linspace(0, 100, b * n, dtype=dtype)).reshape(b, n)
    indices = np.argsort(-inputs, axis=1)[:, :k]
    values = -np.sort(-inputs, axis=1)[:, :k]
    self._validateTopK(inputs, k, values, indices)

  def testLargeTopK(self):
    # [FUNCTION DOC]
    # Test: runs the large top-k helper for single, half and bfloat16 precision. Returns nothing.
    self._testLargeTopK(np.float32)
    self._testLargeTopK(np.float16)
    self._testLargeTopK(dtypes.bfloat16.as_numpy_dtype)

  def _testMediumTopK(self, dtype):
    # [FUNCTION DOC]
    # HELPER: for testMediumTopK. Checks top fifty of a shuffled 5 by 500 array of the given
    # numpy dtype against NumPy sort results. Returns nothing.
    b = 5
    n = 500
    k = 50
    inputs = np.random.permutation(
        np.linspace(0, 100, b * n, dtype=dtype)).reshape(b, n)
    indices = np.argsort(-inputs, axis=1)[:, :k]
    values = -np.sort(-inputs, axis=1)[:, :k]
    self._validateTopK(inputs, k, values, indices)

  def testMediumTopK(self):
    # [FUNCTION DOC]
    # Test: runs the medium top-k helper for single, half and bfloat16 precision. Returns
    # nothing.
    self._testMediumTopK(np.float32)
    self._testMediumTopK(np.float16)
    self._testMediumTopK(dtypes.bfloat16.as_numpy_dtype)

  def testStableSort(self):
    # [FUNCTION DOC]
    # Test: shuffled 5 by 500 integer data with many repeated values (range zero to three), for k
    # of 1, 5, 50 and 500. Expected indices come from a stable merge sort, so top-k must break
    # ties by lower index first. Returns nothing.
    b = 5
    n = 500
    for k in [1, 5, 50, 500]:
      # Lots of repeated integers taking values in [0, 3]
      inputs = np.random.permutation(
          np.linspace(0, 3, b * n, dtype=np.int32)).reshape(b, n)
      # Use mergesort, a stable sort, to get the indices.
      indices = np.argsort(-inputs, axis=1, kind="mergesort")[:, :k]
      values = -np.sort(-inputs, axis=1)[:, :k]
      self._validateTopK(inputs, k, values, indices)

  def testTopAll(self):
    # [FUNCTION DOC]
    # Test: k equal to the row length (four) must return each full row sorted descending, with
    # ties resolved by lower index. Returns nothing.
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.3, 0.2]]
    self._validateTopK(inputs, 4, [[0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.2, 0.1]],
                       [[3, 1, 2, 0], [1, 2, 3, 0]])

  def testTop3Unsorted(self):
    # [FUNCTION DOC]
    # Test: requests the top three with sorted output disabled; the validator checks indices are
    # valid, ties are in index order, and results match the expectation once sorted. Returns
    # nothing.
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.4, 0.3, 0.2]]
    self._validateTopK(
        inputs,
        3, [[0.2, 0.3, 0.4], [0.2, 0.4, 0.3]], [[2, 1, 3], [3, 1, 2]],
        sorted=False)

  def testTop3Vector(self):
    # [FUNCTION DOC]
    # Test: top three of a one-dimensional integer list must be the three largest values with
    # their positions. Returns nothing.
    inputs = [3, 6, 15, 18, 6, 12, 1, 17, 3, 0, 4, 19, 1, 6]
    self._validateTopK(inputs, 3, [19, 18, 17], [11, 3, 7])

  def testTensorK(self):
    # [FUNCTION DOC]
    # Test: same as the vector case but k is passed as a constant tensor. Returns nothing.
    inputs = [3, 6, 15, 18, 6, 12, 1, 17, 3, 0, 4, 19, 1, 6]
    k = constant_op.constant(3)
    self._validateTopK(inputs, k, [19, 18, 17], [11, 3, 7])

  def testTop3ZeroRows(self):
    # [FUNCTION DOC]
    # Test: an input with zero rows and ten columns, k of three, must yield empty outputs of
    # shape zero by three for both values and indices. Returns nothing.
    inputs = np.zeros([0, 10], dtype=np.float32)
    self._validateTopK(inputs, 3, np.zeros([0, 3], dtype=np.float32),
                       np.zeros([0, 3], dtype=np.int32))

  def testKNegative(self):
    # [FUNCTION DOC]
    # Test: a negative k (minus seven) must raise a value error or invalid-argument error whose
    # message says k must be non-negative. Returns nothing.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "Need k >= 0, got -7|non-negative",
    ):
      self.evaluate(nn_ops.top_k([[0.1, 0.2], [0.3, 0.4]], -7))

  def testKTooLarge(self):
    # [FUNCTION DOC]
    # Test: k of four on rows with only two columns must raise a value error or invalid-argument
    # error saying the last dimension must be at least k. Returns nothing.
    inputs = [[0.1, 0.2], [0.3, 0.4]]
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        r"must have last dimension >= k = 4|must have at least k",
    ):
      self.evaluate(nn_ops.top_k(inputs, 4))

  @test_util.run_deprecated_v1
  def testTopKGradients(self):
    # [FUNCTION DOC]
    # Test (graph mode, deprecated v1 style): top three of a placeholder of shape two by five,
    # with upstream gradients supplied, fed with concrete data. Asserts the gradient with respect
    # to the input scatters the upstream gradients back to the selected positions and is zero
    # elsewhere. Returns nothing.
    with self.session() as sess:
      inputs = array_ops.placeholder(dtypes.float32, shape=[2, 5])
      values, _ = nn_ops.top_k(inputs, 3)
      grad = sess.run(
          gradients_impl.gradients(
              values, inputs, grad_ys=[[[1., 2., 3.], [4., 5., 6.]]]),
          feed_dict={inputs: [[2., -1., 1000., 3., 4.],
                              [1., 5., 2., 4., 3.]]})[0]
    self.assertEqual(
        grad.tolist(), [[0., 0., 1., 3., 2.], [0., 4., 0., 5., 6.]])


class TopKBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark case measuring the top-k operation throughput; holds no state.

  def benchmarkTopK(self):
    # [FUNCTION DOC]
    # Benchmark: for 128 rows, several row lengths, several fractions of the row length as k, and
    # CPU and GPU placement, builds a graph with a random resource variable and the top-k op,
    # runs the benchmark harness for at least 100 iterations, and prints wall time and throughput
    # in gigabytes per second. Combinations with k of zero are skipped. Side effects: prints and
    # flushes standard output, reports benchmark results. Returns nothing.
    # KNOWN ISSUE: the benchmark name formats k with a general float format and the throughput
    # counts only input elements (not bytes), so the labeled GB/s is an element rate in billions.
    for (m, n, p, use_gpu) in itertools.product(
        [128],
        [10, 100, 1000, 10000, 100000],
        [0.001, 0.01, 0.5, 0.99, 1.0],
        [False, True]):
      k = int(p * n)
      if k == 0:
        continue
      name = "m_%d_n_%d_k_%g_use_gpu_%s" % (m, n, k, use_gpu)
      device = "/%s:0" % ("gpu" if use_gpu else "cpu")
      with ops.Graph().as_default():
        with ops.device(device):
          x = random_ops.random_uniform((m, n))
          v = resource_variable_ops.ResourceVariable(x)
          op = nn_ops.top_k(v, k)
        with session.Session() as sess:
          self.evaluate(v.initializer)
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          gb_processed_input = m * n / 1.0e9
          throughput = gb_processed_input / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()


if __name__ == "__main__":
  test.main()
