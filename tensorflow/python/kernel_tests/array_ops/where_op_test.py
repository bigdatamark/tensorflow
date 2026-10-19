# [MODULE DOC]
# KNOWN ISSUE: the module docstring says it tests the reverse sequence op, but the file tests the
# where op (copy-paste mismatch).
# Tests and benchmarks for the where operation (v1 single-argument coordinates form and
# three-argument select form, plus the v2 broadcasting variant). Defines a test case that checks
# vector, matrix, 3-D, random multi-dtype and batch-select inputs against numpy, and a benchmark
# class measuring throughput over CPU and GPU.
# No import-time side effects; running as a script
# starts the TensorFlow test runner.
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
"""Tests for tensorflow.ops.reverse_sequence_op."""

import itertools
import sys

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class WhereOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the where op in both its v1 and v2 (broadcasting) forms. Private helper methods
  # implement each scenario parameterized by the op function under test; public test methods call
  # them for the v1 and v2 implementations and for various dtypes.

  def _testWhere(self, x, truth, expected_err_re=None, fn=array_ops.where):
    # [FUNCTION DOC]
    # HELPER: core check for the single-argument where form.
    # Inputs: array x (condition), truth (expected coordinates array), optional expected error
    # regular expression, and the op function to test (default v1 where). In a cached session
    # applies the op, asserts the static shape is compatible with unknown count by rank-of-x
    # columns; if no error is expected, asserts the result is close to truth; otherwise asserts
    # evaluation raises an op error matching the expression. Returns nothing.
    with self.cached_session():
      ans = fn(x)
      self.assertTrue(ans.get_shape().is_compatible_with([None, x.ndim]))
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        self.assertAllClose(tf_ans, truth, atol=1e-10)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testWrongNumbers(self, fn=array_ops.where):
    # [FUNCTION DOC]
    # HELPER: checks argument validation of the three-argument form.
    # In a session asserts that supplying only the x argument without y, or only y without x,
    # raises a value error. Input: the op function under test (default v1 where). Returns nothing.
    with self.session():
      with self.assertRaises(ValueError):
        fn([False, True], [1, 2], None)
      with self.assertRaises(ValueError):
        fn([False, True], None, [1, 2])

  def _testBasicVec(self, fn=array_ops.where):
    # [FUNCTION DOC]
    # HELPER: checks where on three small boolean vectors with one or two true entries, expecting
    # column-vector coordinates of the true positions (64-bit ints). Input: op function under
    # test. Returns nothing.
    x = np.asarray([True, False])
    truth = np.asarray([[0]], dtype=np.int64)
    self._testWhere(x, truth, None, fn)

    x = np.asarray([False, True, False])
    truth = np.asarray([[1]], dtype=np.int64)
    self._testWhere(x, truth, None, fn)

    x = np.asarray([False, False, True, False, True])
    truth = np.asarray([[2], [4]], dtype=np.int64)
    self._testWhere(x, truth, None, fn)

  def _testRandomVec(self, fn=array_ops.where):
    # [FUNCTION DOC]
    # HELPER: checks where on a random boolean vector of one million entries against numpy's
    # coordinates of true elements. Input: op function under test. Uses the global numpy random
    # state. Returns nothing.
    x = np.random.rand(1000000) > 0.5
    truth = np.vstack([np.where(x)[0].astype(np.int64)]).T
    self._testWhere(x, truth, None, fn)

  def _testBasicMat(self, fn=array_ops.where):
    # [FUNCTION DOC]
    # HELPER: checks where on a 2x2 boolean matrix and verifies coordinates come out in row-major
    # order. Input: op function under test. Returns nothing.
    x = np.asarray([[True, False], [True, False]])

    # Ensure RowMajor mode
    truth = np.asarray([[0, 0], [1, 0]], dtype=np.int64)

    self._testWhere(x, truth, None, fn)

  def _testBasic3Tensor(self, fn=array_ops.where):
    # [FUNCTION DOC]
    # HELPER: checks where on a 3x2x2 boolean tensor, expecting five row-major coordinate triples.
    # Input: op function under test. Returns nothing.
    x = np.asarray([[[True, False], [True, False]],
                    [[False, True], [False, True]],
                    [[False, False], [False, True]]])

    # Ensure RowMajor mode
    truth = np.asarray(
        [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [2, 1, 1]], dtype=np.int64)

    self._testWhere(x, truth, None, fn)

  def _testRandom(self, dtype, expected_err_re=None, fn=array_ops.where):
    # [FUNCTION DOC]
    # HELPER: shared random-data check for where, used by the per-dtype tests.
    # KNOWN ISSUE: the complex-valued array built on the first line is immediately overwritten by
    # the second assignment, so it is dead code; the data is real-valued regardless of dtype.
    # Checks where on random 127x33x53 data converted to the given dtype (non-zero counts
    # as true), comparing with numpy's coordinates of non-zero elements stacked as columns.
    # Inputs: numpy dtype, optional expected error regular expression, op function under test.
    # Uses the global numpy random state. Returns nothing.
    shape = [127, 33, 53]
    x = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    x = (np.random.randn(*shape) > 0).astype(dtype)
    truth = np.where(np.abs(x) > 0)  # Tuples of indices by axis.
    truth = np.vstack(truth).T  # Convert to [num_true, indices].
    self._testWhere(x, truth, expected_err_re, fn)

  def _testThreeArgument(self, fn=array_ops.where):
    # [FUNCTION DOC]
    # HELPER: checks the three-argument select form on a small 2x3 integer array, choosing the
    # square where positive and the negation otherwise, and compares with numpy. Input: op
    # function under test. Returns nothing.
    x = np.array([[-2, 3, -1], [1, -3, -3]])
    np_val = np.where(x > 0, x * x, -x)
    with self.test_session():
      tf_val = self.evaluate(fn(constant_op.constant(x) > 0, x * x, -x))
    self.assertAllEqual(tf_val, np_val)

  def testWrongNumbers(self):
    # [FUNCTION DOC]
    # Runs the wrong-argument-count check for the v1 op. Returns nothing.
    self._testWrongNumbers()

  @test_util.run_deprecated_v1
  def testBasicVec(self):
    # [FUNCTION DOC]
    # Runs the basic vector scenario for the v1 op (graph mode). Returns nothing.
    self._testBasicVec()

  @test_util.run_deprecated_v1
  def testRandomVec(self):
    # [FUNCTION DOC]
    # Runs the large random vector scenario for the v1 op (graph mode). Returns nothing.
    self._testRandomVec()

  @test_util.run_deprecated_v1
  def testBasicMat(self):
    # [FUNCTION DOC]
    # Runs the 2x2 matrix row-major scenario for the v1 op (graph mode). Returns nothing.
    self._testBasicMat()

  @test_util.run_deprecated_v1
  def testBasic3Tensor(self):
    # [FUNCTION DOC]
    # Runs the 3-D tensor scenario for the v1 op (graph mode). Returns nothing.
    self._testBasic3Tensor()

  @test_util.run_deprecated_v1
  def testRandomBool(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with boolean data for the v1 op (graph mode). Returns nothing.
    self._testRandom(np.bool_)

  @test_util.run_deprecated_v1
  def testRandomInt32(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with 32-bit integer data for the v1 op (graph mode). Returns
    # nothing.
    self._testRandom(np.int32)

  @test_util.run_deprecated_v1
  def testRandomInt64(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with 64-bit integer data for the v1 op (graph mode). Returns
    # nothing.
    self._testRandom(np.int64)

  @test_util.run_deprecated_v1
  def testRandomFloat(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with 32-bit float data for the v1 op (graph mode). Returns
    # nothing.
    self._testRandom(np.float32)

  @test_util.run_deprecated_v1
  def testRandomDouble(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with 64-bit float data for the v1 op (graph mode). Returns
    # nothing.
    self._testRandom(np.float64)

  @test_util.run_deprecated_v1
  def testRandomComplex64(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with complex64 data for the v1 op (graph mode). Returns
    # nothing.
    self._testRandom(np.complex64)

  @test_util.run_deprecated_v1
  def testRandomComplex128(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with complex128 data for the v1 op (graph mode). Returns
    # nothing.
    self._testRandom(np.complex128)

  @test_util.run_deprecated_v1
  def testRandomUint8(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with unsigned 8-bit data for the v1 op (graph mode). Returns
    # nothing.
    self._testRandom(np.uint8)

  @test_util.run_deprecated_v1
  def testRandomInt8(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with signed 8-bit data for the v1 op (graph mode). Returns
    # nothing.
    self._testRandom(np.int8)

  @test_util.run_deprecated_v1
  def testRandomInt16(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with signed 16-bit data for the v1 op (graph mode). Returns
    # nothing.
    self._testRandom(np.int16)

  @test_util.run_deprecated_v1
  def testThreeArgument(self):
    # [FUNCTION DOC]
    # Runs the three-argument select scenario for the v1 op (graph mode). Returns nothing.
    self._testThreeArgument()

  def testV2WrongNumbers(self):
    # [FUNCTION DOC]
    # Runs the wrong-argument-count check for the v2 op. Returns nothing.
    self._testWrongNumbers(array_ops.where_v2)

  def testV2BasicVec(self):
    # [FUNCTION DOC]
    # Runs the basic vector scenario for the v2 op. Returns nothing.
    self._testBasicVec(array_ops.where_v2)

  def testV2RandomVec(self):
    # [FUNCTION DOC]
    # Runs the large random vector scenario for the v2 op. Returns nothing.
    self._testRandomVec(array_ops.where_v2)

  def testV2BasicMat(self):
    # [FUNCTION DOC]
    # Runs the 2x2 matrix row-major scenario for the v2 op. Returns nothing.
    self._testBasicMat(array_ops.where_v2)

  def testV2Basic3Tensor(self):
    # [FUNCTION DOC]
    # Runs the 3-D tensor scenario for the v2 op. Returns nothing.
    self._testBasic3Tensor(array_ops.where_v2)

  def testV2RandomBool(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with boolean data for the v2 op. Returns nothing.
    self._testRandom(np.bool_, None, array_ops.where_v2)

  def testV2RandomInt32(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with 32-bit integer data for the v2 op. Returns nothing.
    self._testRandom(np.int32, None, array_ops.where_v2)

  def testV2RandomInt64(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with 64-bit integer data for the v2 op. Returns nothing.
    self._testRandom(np.int64, None, array_ops.where_v2)

  def testV2RandomFloat(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with 32-bit float data for the v2 op. Returns nothing.
    self._testRandom(np.float32, None, array_ops.where_v2)

  def testV2RandomDouble(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with 64-bit float data for the v2 op. Returns nothing.
    self._testRandom(np.float64, None, array_ops.where_v2)

  def testV2RandomComplex64(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with complex64 data for the v2 op. Returns nothing.
    self._testRandom(np.complex64, None, array_ops.where_v2)

  def testV2RandomComplex128(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with complex128 data for the v2 op. Returns nothing.
    self._testRandom(np.complex128, None, array_ops.where_v2)

  def testV2RandomUint8(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with unsigned 8-bit data for the v2 op. Returns nothing.
    self._testRandom(np.uint8, None, array_ops.where_v2)

  def testV2RandomInt8(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with signed 8-bit data for the v2 op. Returns nothing.
    self._testRandom(np.int8, None, array_ops.where_v2)

  def testV2RandomInt16(self):
    # [FUNCTION DOC]
    # Runs the random-data scenario with signed 16-bit data for the v2 op. Returns nothing.
    self._testRandom(np.int16, None, array_ops.where_v2)

  def testV2ThreeArgument(self):
    # [FUNCTION DOC]
    # Runs the three-argument select scenario for the v2 op. Returns nothing.
    self._testThreeArgument(array_ops.where_v2)

  def testV2Broadcasting(self):
    # [FUNCTION DOC]
    # Scenario: the v2 op with a random 4-D condition of shape 3x5x1x1 and 7x11 zeros and ones as
    # the two branches; the three inputs broadcast together. Asserts the result equals numpy's
    # selection. Returns nothing.
    f = np.random.normal(0, 1, (3, 5, 1, 1))
    x = np.zeros((7, 11))
    y = np.ones((7, 11))
    np_val = np.where(f < 0, x, y)
    with self.test_session():
      tf_val = self.evaluate(
          array_ops.where_v2(constant_op.constant(f) < 0, x, y))
    self.assertAllEqual(tf_val, np_val)

  def testV2ScalarBroadcasting(self):
    # [FUNCTION DOC]
    # Scenario: the v2 op with a scalar true condition and 7x11 zeros and ones; asserts the result
    # equals numpy's selection (all zeros). Returns nothing.
    x = np.zeros((7, 11))
    y = np.ones((7, 11))
    np_val = np.where(True, x, y)
    with self.test_session():
      tf_val = self.evaluate(
          array_ops.where_v2(
              constant_op.constant(True, dtype=dtypes.bool), x, y))
    self.assertAllEqual(tf_val, np_val)

  def testV2VectorBroadcasting(self):
    # [FUNCTION DOC]
    # Scenario: the v2 op with a one-element true condition vector broadcast against length-7
    # zeros and ones; asserts the result equals numpy's selection. Returns nothing.
    x = np.zeros(7)
    y = np.ones(7)
    np_val = np.where([True], x, y)
    with self.test_session():
      tf_val = self.evaluate(
          array_ops.where_v2(
              constant_op.constant([True], dtype=dtypes.bool), x, y))
    self.assertAllEqual(tf_val, np_val)

  def testV2PredBroadcasting(self):
    # [FUNCTION DOC]
    # Scenario: the v2 op with a 3x1 integer predicate column broadcast against random 3x4
    # branches; asserts the result is close to numpy's selection. Uses the global numpy random
    # state. Returns nothing.
    pred = np.array([1, 0, 0]).reshape((3, 1))
    x = np.random.randn(3, 4)
    y = np.random.randn(3, 4)
    np_val = np.where(pred, x, y)
    with self.test_session():
      tf_val = self.evaluate(array_ops.where_v2(pred, x, y))
    self.assertAllClose(tf_val, np_val)

  @test_util.run_deprecated_v1
  def testBatchSelect(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): the v1 three-argument form with a 16384-element boolean vector
    # selecting whole rows of a 16384x192 matrix (square versus negation). Asserts it equals
    # numpy's selection using the equivalent full-matrix condition. Returns nothing.
    x = np.array([[-2, 3, -1] * 64, [1, -3, -3] * 64] * 8192)  # [16384, 192]
    c_mat = np.array([[False] * 192, [True] * 192] * 8192)  # [16384, 192]
    c_vec = np.array([False, True] * 8192)  # [16384]
    np_val = np.where(c_mat, x * x, -x)
    with self.session():
      tf_val = array_ops.where(c_vec, x * x, -x).eval()
    self.assertAllEqual(tf_val, np_val)


class WhereBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmarks for the where op: one for coordinate extraction over a grid of sizes, densities and
  # devices, and one for batch row selection. Results are reported via the benchmark framework and
  # printed to standard output.

  def benchmarkWhere(self):
    # [FUNCTION DOC]
    # Benchmarks the single-argument where op over combinations of row count, column count, true
    # probability and device (CPU or GPU). For each, builds a graph with a boolean resource
    # variable, runs the op for at least 100 iterations in a session, and prints the benchmark name,
    # wall time and estimated throughput in gigabytes per second (input plus approximate output
    # size). Side effects: prints and flushes standard output, records benchmark results. Returns
    # nothing.
    for (m, n, p, use_gpu) in itertools.product(
        [10],
        [10, 100, 1000, 10000, 100000, 1000000],
        [0.01, 0.5, 0.99],
        [False, True]):
      name = "m_%d_n_%d_p_%g_use_gpu_%s" % (m, n, p, use_gpu)
      device = "/%s:0" % ("gpu" if use_gpu else "cpu")
      with ops.Graph().as_default():
        with ops.device(device):
          x = random_ops.random_uniform((m, n), dtype=dtypes.float32) <= p
          v = resource_variable_ops.ResourceVariable(x)
          op = array_ops.where(v)
        with session.Session(config=benchmark.benchmark_config()) as sess:
          self.evaluate(v.initializer)
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          gb_processed_input = m * n / 1.0e9
          # approximate size of output: m*n*p int64s for each axis.
          gb_processed_output = 2 * 8 * m * n * p / 1.0e9
          gb_processed = gb_processed_input + gb_processed_output
          throughput = gb_processed / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()

  def benchmarkBatchSelect(self):
    # [FUNCTION DOC]
    # Benchmarks the three-argument where with a per-row boolean condition over combinations of
    # row count, column count and device. For each, builds variables for both branches and the
    # condition, runs the op for at least 100 iterations, and prints name, wall time and estimated
    # throughput in gigabytes per second. Side effects: prints and flushes standard output,
    # records benchmark results. Returns nothing.
    for (m, n, use_gpu) in itertools.product([1000, 10000, 100000],
                                             [10, 100, 1000], [False, True]):
      name = "m_%d_n_%d_use_gpu_%s" % (m, n, use_gpu)
      device = "/%s:0" % ("gpu" if use_gpu else "cpu")
      with ops.Graph().as_default():
        with ops.device(device):
          x_gen = random_ops.random_uniform([m, n], dtype=dtypes.float32)
          y_gen = random_ops.random_uniform([m, n], dtype=dtypes.float32)
          c_gen = random_ops.random_uniform([m], dtype=dtypes.float32) <= 0.5
          x = resource_variable_ops.ResourceVariable(x_gen)
          y = resource_variable_ops.ResourceVariable(y_gen)
          c = resource_variable_ops.ResourceVariable(c_gen)
          op = array_ops.where(c, x, y)
        with session.Session(config=benchmark.benchmark_config()) as sess:
          self.evaluate(x.initializer)
          self.evaluate(y.initializer)
          self.evaluate(c.initializer)
          r = self.run_op_benchmark(sess, op, min_iters=100, name=name)
          # approximate size of output: m*n*2 floats for each axis.
          gb_processed = m * n * 8 / 1.0e9
          throughput = gb_processed / r["wall_time"]
          print("Benchmark: %s \t wall_time: %0.03g s \t "
                "Throughput: %0.03g GB/s" % (name, r["wall_time"], throughput))
          sys.stdout.flush()


if __name__ == "__main__":
  test.main()
