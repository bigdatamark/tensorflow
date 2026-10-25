# [MODULE DOC]
# Unit tests and a benchmark for sparse_ops.sparse_add. Tests cover sparse+sparse addition
# (self, with negation, small-value thresholding), sparse+dense addition in both argument orders
# and several dtypes, gradients, and rejection of invalid indices or mismatched shapes.
# The benchmark times sparse_add against converting to dense and adding. Importing the module
# also imports the sparse gradient registrations (needed for the gradient tests).
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SparseAdd."""

import timeit

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  # [FUNCTION DOC]
  # HELPER: converts dense random arrays into sparse tensors for the tests and benchmark.
  # Inputs: a two-dimensional NumPy array x; thresh (default 0.5), values below it are zeroed;
  # index_dtype (default 64-bit integer) type of the index array. Side effect: x is modified in
  # place (small entries set to zero). Returns a pair: a SparseTensor holding the remaining
  # non-zero entries (row-major order), and the count of non-zero values.
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values)


class SparseAddTest(test.TestCase):
  # [CLASS DOC]
  # Test case for sparse_ops.sparse_add covering sparse-sparse and sparse-dense addition,
  # thresholding, gradients and invalid input handling. Provides fixture builders for small 3x3
  # sparse tensors and random sparse tensors.

  def _randomTensor(self, size, np_dtype, sparse=True):
    # [FUNCTION DOC]
    # HELPER: random fixture builder. Takes a pair (rows, columns), a NumPy dtype and a sparse
    # flag (default true). Draws normal random values; if sparse, returns the pair from
    # sparsifying them (sparse tensor and nonzero count), otherwise the dense array.
    n, m = size
    x = np.random.randn(n, m).astype(np_dtype)
    return _sparsify(x) if sparse else x

  def _SparseTensorValue_3x3(self, negate=False):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a SparseTensorValue (NumPy-backed) of dense shape 3x3 with four
    # float entries 1, 2, 3, 4 at positions (0,1), (1,0), (2,0), (2,1). If negate is true
    # (default false) all values are negated.
    # [    1]
    # [2    ]
    # [3   4]
    # ...or its cwise negation, if `negate`
    ind = np.array([[0, 1], [1, 0], [2, 0], [2, 1]])
    val = np.array([1, 2, 3, 4])
    if negate:
      val = -np.array([1, 2, 3, 4])
    shape = np.array([3, 3])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(val, np.float32), np.array(shape, np.int64))

  def _SparseTensor_3x3(self, negate=False):
    # [FUNCTION DOC]
    # HELPER: fixture. Wraps the 3x3 sparse value fixture (optionally negated, default false)
    # into a SparseTensor and returns it.
    return sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_3x3(negate))

  def _SparseTensor_3x3_v2(self):
    # [FUNCTION DOC]
    # HELPER: fixture. Returns a 3x3 SparseTensor built from constants with the same positions as
    # the basic fixture but values 1, -1.9, 3, -4.2, used so sums with the basic fixture produce
    # small values that threshold tests can drop.
    # [           1]
    # [-1.9        ]
    # [   3    -4.2]
    ind = np.array([[0, 1], [1, 0], [2, 0], [2, 1]])
    val = np.array([1, -1.9, 3, -4.2])
    shape = np.array([3, 3])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

  def testAddSelf(self):
    # [FUNCTION DOC]
    # On CPU, adds the 3x3 fixture to itself for all combinations of value and tensor
    # representations and asserts static shape, dense shape vector shape, indices unchanged and
    # values doubled (2, 4, 6, 8). Returns nothing.
    with test_util.force_cpu():
      for sp_a in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
        for sp_b in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
          sp_sum = sparse_ops.sparse_add(sp_a, sp_b)
          self.assertAllEqual((3, 3), sp_sum.get_shape())

          sum_out = self.evaluate(sp_sum)

          self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
          self.assertAllEqual(sum_out.indices, [[0, 1], [1, 0], [2, 0], [2, 1]])
          self.assertAllEqual(sum_out.values, [2, 4, 6, 8])
          self.assertAllEqual(sum_out.dense_shape, [3, 3])

  def testAddSelfAndNegation(self):
    # [FUNCTION DOC]
    # On CPU, adds the fixture to its negation with threshold 0.1 and asserts the result has no
    # entries (empty indices and values) and dense shape 3x3. Returns nothing.
    with test_util.force_cpu():
      sp_a = self._SparseTensor_3x3()
      sp_b = self._SparseTensor_3x3(negate=True)

      sp_sum = sparse_ops.sparse_add(sp_a, sp_b, 0.1)
      sum_out = self.evaluate(sp_sum)

      self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
      self.assertAllEqual(sum_out.indices, np.empty([0, 2]))
      self.assertAllEqual(sum_out.values, [])
      self.assertAllEqual(sum_out.dense_shape, [3, 3])

  def testSmallValuesShouldVanish(self):
    # [FUNCTION DOC]
    # On CPU, adds two different 3x3 fixtures with threshold 0.21 (both small sums, 0.1 and -0.2,
    # vanish, leaving values 2 and 6) and with threshold 0.11 (only 0.1 vanishes, leaving 2, 6
    # and -0.2). Asserts indices, values and dense shape. Returns nothing.
    with test_util.force_cpu():
      sp_a = self._SparseTensor_3x3()
      sp_b = self._SparseTensor_3x3_v2()

      # sum:
      # [       2]
      # [.1      ]
      # [ 6   -.2]

      # two values should vanish: |.1| < .21, and |-.2| < .21
      sp_sum = sparse_ops.sparse_add(sp_a, sp_b, thresh=0.21)
      sum_out = self.evaluate(sp_sum)

      self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
      self.assertAllEqual(sum_out.indices, [[0, 1], [2, 0]])
      self.assertAllEqual(sum_out.values, [2, 6])
      self.assertAllEqual(sum_out.dense_shape, [3, 3])

      # only .1 vanishes
      sp_sum = sparse_ops.sparse_add(sp_a, sp_b, thresh=0.11)
      sum_out = self.evaluate(sp_sum)

      self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
      self.assertAllEqual(sum_out.indices, [[0, 1], [2, 0], [2, 1]])
      self.assertAllClose(sum_out.values, [2, 6, -.2])
      self.assertAllEqual(sum_out.dense_shape, [3, 3])

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Deprecated-v1 (graph) test with a fixed random seed. For several sizes, adds two random
    # sparse tensors and asserts the numerical gradient error with respect to both value vectors
    # is below 1e-3. Uses a CPU-only session. Returns nothing.
    np.random.seed(1618)  # Make it reproducible.
    with self.session(use_gpu=False):
      for n in [10, 31]:
        for m in [4, 17]:
          sp_a, nnz_a = self._randomTensor([n, m], np.float32)
          sp_b, nnz_b = self._randomTensor([n, m], np.float32)
          sp_sum = sparse_ops.sparse_add(sp_a, sp_b)
          nnz_sum = len(self.evaluate(sp_sum.values))

          err = gradient_checker.compute_gradient_error(
              [sp_a.values, sp_b.values], [(nnz_a,), (nnz_b,)], sp_sum.values,
              (nnz_sum,))
          self.assertLess(err, 1e-3)

  def testAddSparseDense(self):
    # [FUNCTION DOC]
    # With a fixed seed and random sizes, for several value dtypes and both index dtypes, adds
    # a sparsified array to a dense constant on CPU, in both argument orders, asserting the
    # result equals the dense sum and keeps the value dtype. Returns nothing.
    np.random.seed(1618)  # Make it reproducible.
    n, m = np.random.randint(30, size=2)
    for dtype in [np.float32, np.float64, np.int64, np.complex64]:
      for index_dtype in [np.int32, np.int64]:
        rand_vals_np = np.random.randn(n, m).astype(dtype)
        dense_np = np.random.randn(n, m).astype(dtype)

        with test_util.force_cpu():
          sparse, unused_nnz = _sparsify(rand_vals_np, index_dtype=index_dtype)
          s = self.evaluate(
              sparse_ops.sparse_add(sparse, constant_op.constant(dense_np)))
          self.assertAllEqual(dense_np + rand_vals_np, s)
          self.assertTrue(s.dtype == dtype)

          # check commutativity
          s = self.evaluate(
              sparse_ops.sparse_add(constant_op.constant(dense_np), sparse))
          self.assertAllEqual(dense_np + rand_vals_np, s)
          self.assertTrue(s.dtype == dtype)

  @test_util.run_deprecated_v1
  def testSparseTensorDenseAddGradients(self):
    # [FUNCTION DOC]
    # Deprecated-v1 (graph) test with a fixed seed. Adds a random sparse tensor to a random
    # dense constant and asserts the numerical gradient error with respect to the sparse values
    # and the dense input is below 1e-3. Returns nothing.
    np.random.seed(1618)  # Make it reproducible.
    n, m = np.random.randint(30, size=2)
    rand_vals_np = np.random.randn(n, m).astype(np.float32)
    dense_np = np.random.randn(n, m).astype(np.float32)

    with self.session(use_gpu=False):
      sparse, nnz = _sparsify(rand_vals_np)
      dense = constant_op.constant(dense_np, dtype=dtypes.float32)
      s = sparse_ops.sparse_add(sparse, dense)

      err = gradient_checker.compute_gradient_error([sparse.values, dense],
                                                    [(nnz,), (n, m)], s, (n, m))
      self.assertLess(err, 1e-3)

  def testInvalidSparseTensor(self):
    # [FUNCTION DOC]
    # On CPU, adds a dense 2x2 tensor to a sparse value with an out-of-range index (negative or
    # beyond the shape) and asserts a value error or invalid-argument error whose message
    # mentions an invalid index. Returns nothing.
    with test_util.force_cpu():
      shape = [2, 2]
      val = [0]
      dense = constant_op.constant(np.zeros(shape, dtype=np.int32))

      for bad_idx in [
          [[-1, 0]],  # -1 is invalid.
          [[1, 3]],  # ...so is 3.
      ]:
        sparse = sparse_tensor.SparseTensorValue(bad_idx, val, shape)
        with self.assertRaisesRegex(
            (ValueError, errors_impl.InvalidArgumentError), "invalid index"):
          s = sparse_ops.sparse_add(sparse, dense)
          self.evaluate(s)

  def _testSparseDenseInvalidInputs(self,
                                    a_indices,
                                    a_values,
                                    a_shape,
                                    b,
                                    expected_error=""):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the invalid-input test. Takes sparse indices, values, dense
    # shape, a dense tensor b and an expected error message pattern (default empty, matches
    # anything). Asserts a value error or invalid-argument error is raised both through the
    # public sparse_add (which builds a SparseTensor) and by calling the generated sparse-dense
    # add kernel directly, which bypasses SparseTensor validation. Returns nothing.
    # Public API call to sparse-dense add.
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                expected_error):
      a = sparse_tensor.SparseTensor(a_indices, a_values, a_shape)
      self.evaluate(sparse_ops.sparse_add(a, b))
    # Directly call generated kernel, by-passing SparseTensor validation.
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                expected_error):
      self.evaluate(
          sparse_ops.gen_sparse_ops.sparse_tensor_dense_add(
              a_indices, a_values, a_shape, b))

  def testSparseDenseInvalidInputs(self):
    # [FUNCTION DOC]
    # Runs the invalid-input helper for three cases: indices and values with mismatched counts
    # (17 vs 5), indices of rank 4 against a rank-2 shape, and out-of-range indices (value 7).
    # Expects the corresponding dimension-incompatible or invalid index errors. Returns nothing.
    self._testSparseDenseInvalidInputs(
        a_indices=constant_op.constant(0, shape=[17, 2], dtype=dtypes.int64),
        a_values=constant_op.constant(0, shape=[5], dtype=dtypes.float32),
        a_shape=constant_op.constant([3, 4], dtype=dtypes.int64),
        b=constant_op.constant(1, shape=[3, 4], dtype=dtypes.float32),
        expected_error="Dimensions 17 and 5 are not compatible")
    self._testSparseDenseInvalidInputs(
        a_indices=constant_op.constant(0, shape=[17, 4], dtype=dtypes.int64),
        a_values=constant_op.constant(0, shape=[17], dtype=dtypes.float32),
        a_shape=constant_op.constant([3, 4], dtype=dtypes.int64),
        b=constant_op.constant(1, shape=[3, 4], dtype=dtypes.float32),
        expected_error="Dimensions 4 and 2 are not compatible")
    self._testSparseDenseInvalidInputs(
        a_indices=constant_op.constant(7, shape=[17, 2], dtype=dtypes.int64),
        a_values=constant_op.constant(0, shape=[17], dtype=dtypes.float32),
        a_shape=constant_op.constant([3, 4], dtype=dtypes.int64),
        b=constant_op.constant(1, shape=[3, 4], dtype=dtypes.float32),
        expected_error="invalid index")

######################## Benchmarking code


def _s2d_add_vs_sparse_add(sparsity, n, m, num_iters=50):
  # [FUNCTION DOC]
  # HELPER: timing routine for the sparse add benchmark.
  # Inputs: sparsity threshold (random values below it are zeroed, so larger means sparser),
  # matrix size n by m, and iteration count (default 50). In a fresh graph and session, builds
  # a sparse tensor and a dense random matrix, then times converting-to-dense-and-adding versus
  # sparse_add (after a 3-run warm-up each). Returns a pair of average per-iteration latencies
  # in milliseconds: dense route first, sparse_add second. Seeds the NumPy generator (fixed).
  np.random.seed(1618)

  with session.Session(graph=ops.Graph()) as sess:
    sp_vals = np.random.rand(n, m).astype(np.float32)
    sp_t, unused_nnz = _sparsify(sp_vals, thresh=sparsity, index_dtype=np.int32)
    vals = np.random.rand(n, m).astype(np.float32)

    s2d = math_ops.add(
        sparse_ops.sparse_tensor_to_dense(sp_t), constant_op.constant(vals))
    sa = sparse_ops.sparse_add(sp_t, constant_op.constant(vals))

    timeit.timeit(lambda: sess.run(s2d), number=3)
    timeit.timeit(lambda: sess.run(sa), number=3)

    s2d_total = timeit.timeit(lambda: sess.run(s2d), number=num_iters)
    sa_total = timeit.timeit(lambda: sess.run(sa), number=num_iters)

  # per-iter latency; secs to millis
  return s2d_total * 1e3 / num_iters, sa_total * 1e3 / num_iters


class SparseAddBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class comparing sparse_add against dense conversion plus addition. Only prints
  # results; has no state.

  def benchmarkSparseAddDense(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the printed header labels the first column as the non-zero percentage, but
    # the value printed is the thresholding parameter passed to the timing helper (random
    # values below it are zeroed), so larger numbers mean fewer non-zeros, not more.
    # Sweeps sparsity thresholds, row counts and column counts, timing both approaches and
    # printing one line per combination with latencies and speedup (dense time over sparse_add
    # time). Prints to standard output; returns nothing and reports no benchmark entries.

    print("SparseAddDense: add with sparse_to_dense vs. sparse_add")
    print("%nnz \t n \t m \t millis(s2d) \t millis(sparse_add) \t speedup")

    for sparsity in [0.99, 0.5, 0.01]:
      for n in [1, 256, 50000]:
        for m in [100, 1000]:
          s2d_dt, sa_dt = _s2d_add_vs_sparse_add(sparsity, n, m)
          print("%.2f \t %d \t %d \t %.4f \t %.4f \t %.2f" % (sparsity, n, m,
                                                              s2d_dt, sa_dt,
                                                              s2d_dt / sa_dt))


if __name__ == "__main__":
  test.main()
