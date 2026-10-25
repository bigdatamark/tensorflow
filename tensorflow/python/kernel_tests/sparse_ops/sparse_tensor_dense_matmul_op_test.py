# [MODULE DOC]
# Functional tests and a benchmark for the sparse-times-dense matrix multiplication op. The test
# class compares the op's results against numpy for many value types, index types and adjoint
# flags, checks static shape inference, out-of-bounds index errors on CPU, the different
# behavior on GPU (no error, results marked invalid), unordered indices, and large or random
# sized matrices. The module also holds benchmark helpers that time the op against a dense matmul
# with a sparsity hint over a while loop, and a main function that prints a comparison table.
# Running the script with the benchmarks switch on the command line runs the benchmark through the
# application runner; otherwise the normal TensorFlow test runner starts.
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
"""Tests for sparse_ops.sparse_tensor_dense_matmul."""

import sys
import time

from absl import app
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


def _maybe_complex(x):
  # [FUNCTION DOC]
  # HELPER: gives complex test arrays a non-zero imaginary part. Input x is a numpy array. For a
  # complex type, returns the array plus the same array times the imaginary unit (so real and
  # imaginary parts are equal); otherwise returns x unchanged. No side effects.
  if x.dtype.kind == "c":  # complex
    return x + 1j * x
  return x


class SparseTensorDenseMatMulTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the sparse-times-dense matmul op. It holds no state; helper methods compare the
  # op against numpy for given operands, and test methods supply random operands of various types,
  # sizes and sparsity, plus error and shape inference scenarios.

  def _testMatmul(self,
                  x,
                  y,
                  adjoint_a=False,
                  adjoint_b=False,
                  indices_dtype=np.int64):
    # [FUNCTION DOC]
    # HELPER: checks the sparse-dense matmul op against numpy for one pair of operands.
    # KNOWN ISSUE: the half-precision branch of the tolerance selection uses the same tolerances
    # as the final fallback branch, so it is redundant.
    # Inputs: x, dense numpy array whose non-zero entries become the sparse operand; y, dense
    # operand; adjoint_a and adjoint_b, whether to conjugate-transpose each operand in the
    # reference result and in the op (default false); indices_dtype, numpy index type (64-bit by
    # default). Computes the reference product with numpy, then runs the op both with a raw
    # sparse value and a sparse tensor, asserts the static column count of the result is known and
    # right, and asserts closeness to the reference with tolerances that depend on the value type.
    x_mat = np.array(x)
    if adjoint_a:
      x_mat = x_mat.T.conj()
    y_mat = np.array(y)
    if adjoint_b:
      y_mat = y_mat.T.conj()

    np_ans = x_mat.dot(y_mat)

    x_indices = np.vstack(np.where(x)).astype(indices_dtype).T
    x_values = x[np.where(x)]
    x_shape = x.shape

    with self.cached_session():
      sp_x_value = sparse_tensor.SparseTensorValue(
          indices=x_indices, values=x_values, dense_shape=x_shape)
      tf_value_ans = sparse_ops.sparse_tensor_dense_matmul(
          sp_x_value, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
      tf_tensor_ans = sparse_ops.sparse_tensor_dense_matmul(
          sparse_tensor.SparseTensor.from_value(sp_x_value),
          y,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)

      # Ensure that the RHS shape is known at least.
      self.assertEqual(tf_value_ans.get_shape()[1], np_ans.shape[1])
      self.assertEqual(tf_tensor_ans.get_shape()[1], np_ans.shape[1])

      for out in (self.evaluate(tf_value_ans), self.evaluate(tf_tensor_ans)):
        if x.dtype == np.float32:
          self.assertAllClose(np_ans, out, rtol=1e-4, atol=1e-4)
        elif x.dtype == np.float64:
          self.assertAllClose(np_ans, out, rtol=1e-6, atol=1e-6)
        elif x.dtype == np.float16:
          self.assertAllClose(np_ans, out, rtol=1e-3, atol=1e-3)
        else:
          self.assertAllClose(np_ans, out, rtol=1e-3, atol=1e-3)

  def _testBasic(self, value_dtype, indices_dtype=np.int64):
    # [FUNCTION DOC]
    # HELPER: runs the matmul comparison on random 10x10 sparse and 10x20 dense operands.
    # Inputs: value_dtype, numpy value type; indices_dtype, numpy index type (default 64-bit).
    # The left operand has entries with magnitude under one half zeroed to make it sparse. Uses
    # numpy's global random state. Returns nothing; assertions happen in the comparison helper.
    x = _maybe_complex(np.random.rand(10, 10).astype(value_dtype))
    x[np.abs(x) < 0.5] = 0  # Make it sparse

    y = _maybe_complex(np.random.randn(10, 20).astype(value_dtype))

    self._testMatmul(x, y, indices_dtype=indices_dtype)

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: seed numpy randomness and run the basic comparison for 32-bit integer, half,
    # single, double, complex64 and complex128 values with 64-bit indices, and for integer and
    # single values with 32-bit indices.
    # Expected: results match the numpy product within tolerance.
    np.random.seed(127)  # Repeatable results
    self._testBasic(np.int32)
    self._testBasic(np.float16)
    self._testBasic(np.float32)
    self._testBasic(np.float64)
    self._testBasic(np.complex64)
    self._testBasic(np.complex128)
    self._testBasic(np.int32, indices_dtype=np.int32)
    self._testBasic(np.float32, indices_dtype=np.int32)

  def testShapeInference(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph, multiply a 10x10 sparse tensor by a 10x20 dense array; repeat
    # with the sparse dense-shape being a placeholder of unknown shape; then with a declared
    # sparse shape of 10 by 15 that disagrees with the dense operand.
    # Expected: static result shape 10 by 20; unknown row count with 20 columns when the sparse
    # shape is unknown; a value error saying dimensions must be equal for the mismatched case.
    x = np.random.rand(10, 10)
    x[np.abs(x) < 0.5] = 0  # Make it sparse
    y = np.random.randn(10, 20)
    x_indices = np.vstack(np.where(x)).astype(np.int64).T
    x_values = x[np.where(x)]
    x_shape = x.shape

    with ops.Graph().as_default():
      x_st = sparse_tensor.SparseTensor(x_indices, x_values, x_shape)
      result = sparse_ops.sparse_tensor_dense_matmul(x_st, y)
      self.assertEqual(result.get_shape(), (10, 20))

      x_shape_unknown = array_ops.placeholder(dtype=dtypes.int64, shape=None)
      x_st_shape_unknown = sparse_tensor.SparseTensor(x_indices, x_values,
                                                      x_shape_unknown)
      result_left_shape_unknown = sparse_ops.sparse_tensor_dense_matmul(
          x_st_shape_unknown, y)
      self.assertEqual(result_left_shape_unknown.get_shape().as_list(),
                       [None, 20])

      x_shape_inconsistent = [10, 15]
      x_st_shape_inconsistent = sparse_tensor.SparseTensor(
          x_indices, x_values, x_shape_inconsistent)
      with self.assertRaisesRegex(ValueError, "Dimensions must be equal"):
        sparse_ops.sparse_tensor_dense_matmul(x_st_shape_inconsistent, y)

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testInvalidIndicesForSparseTensorDenseMatmul(self):
    # [FUNCTION DOC]
    # Scenario (CPU only, graph and eager): a 3x2 sparse tensor has one entry whose column index
    # of 10 is out of bounds. Multiply by dense operands with 5 and with 500 columns (to hit both
    # the small and large kernel paths), then repeat with the left operand adjointed.
    # Expected: op errors reporting that the k index (or, with adjoint, the m index) of the entry
    # is out of bounds. A TODO notes the GPU kernel should return similarly nice errors.
    # TODO(b/169813429): Make GPU kernel return nice errors too.
    indices = np.array([[1, 10]]).astype(np.int64)
    values = np.array([10]).astype(np.float32)
    shape = [3, 2]
    sparse_t = sparse_tensor.SparseTensor(indices, values, shape)

    # Test multiplying by both a small and large dense matrix, to hit
    # both cases in the kernel.
    dense_t = np.array([[1] * 5, [2] * 5], dtype=np.float32)
    with self.assertRaisesOpError("k .10. from index.0,1. out of bounds .>=2."):
      self.evaluate(sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))
    dense_t = np.array([[1] * 500, [2] * 500], dtype=np.float32)
    with self.assertRaisesOpError("k .10. from index.0,1. out of bounds .>=2."):
      self.evaluate(sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))

    # Repeat with adjoint_a, to get a different error.
    dense_t = np.array([[1] * 5, [2] * 5, [3] * 5], dtype=np.float32)
    with self.assertRaisesOpError("m .10. from index.0,1. out of bounds .>=2."):
      self.evaluate(
          sparse_ops.sparse_tensor_dense_matmul(
              sparse_t, dense_t, adjoint_a=True))
    dense_t = np.array([[1] * 500, [2] * 500, [3] * 500], dtype=np.float32)
    with self.assertRaisesOpError("m .10. from index.0,1. out of bounds .>=2."):
      self.evaluate(
          sparse_ops.sparse_tensor_dense_matmul(
              sparse_t, dense_t, adjoint_a=True))

  def testUnorderedIndicesForSparseTensorDenseMatmul(self):
    # [FUNCTION DOC]
    # Scenario: a 3x2 sparse tensor whose two entries are listed out of row-major order is
    # multiplied by a 2x500 dense matrix.
    # Expected: the op still produces the right product (rows with 11, 0 and 20 repeated 500
    # times), i.e. index ordering does not matter.
    indices = np.array([(2, 1), (0, 0)]).astype(np.int64)
    values = np.array([10, 11]).astype(np.float32)
    shape = [3, 2]
    sparse_t = sparse_tensor.SparseTensor(indices, values, shape)

    dense_t = np.array([[1] * 500, [2] * 500], dtype=np.float32)
    expected_t = np.array([[11] * 500, [0] * 500, [20] * 500], dtype=np.float32)

    self.assertAllClose(
        expected_t, sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))

  @test_util.run_gpu_only
  def testInvalidIndicesForSparseTensorDenseMatmulOnGPU(self):
    # [FUNCTION DOC]
    # Scenario (GPU only): same out-of-bounds sparse entry as the CPU test, with dense operands of
    # 5 and 500 columns, plain and with the left operand adjointed.
    # Expected: no error is raised; the plain case gives a result whose second row is not-a-number
    # and other rows zero, while the adjoint case gives all zeros because the kernel cannot
    # accumulate an entry that is out of range for the output.
    indices = np.array([[1, 10]]).astype(np.int64)
    values = np.array([10]).astype(np.float32)
    shape = [3, 2]
    sparse_t = sparse_tensor.SparseTensor(indices, values, shape)

    # Test multiplying by both a small and large dense matrix, to hit
    # both cases in the kernel.
    dense_t = np.array([[1] * 5, [2] * 5], dtype=np.float32)
    expected_t = np.array([[0] * 5, [np.nan] * 5, [0] * 5], dtype=np.float32)
    self.assertAllClose(
        expected_t, sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))
    dense_t = np.array([[1] * 500, [2] * 500], dtype=np.float32)
    expected_t = np.array([[0] * 500, [np.nan] * 500, [0] * 500],
                          dtype=np.float32)
    self.assertAllClose(
        expected_t, sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))

    # Repeat with adjoint_a, now the error is that the sparse index
    # is OOO w.r.t. the output.  The GPU kernel can't do much here,
    # so it just doesn't accumulate.

    dense_t = np.array([[1] * 5, [2] * 5, [3] * 5], dtype=np.float32)
    expected_t = np.array([[0] * 5, [0] * 5], dtype=np.float32)
    self.assertAllClose(
        expected_t,
        sparse_ops.sparse_tensor_dense_matmul(
            sparse_t, dense_t, adjoint_a=True))

    dense_t = np.array([[1] * 500, [2] * 500, [3] * 500], dtype=np.float32)
    expected_t = np.array([[0] * 500, [0] * 500], dtype=np.float32)
    self.assertAllClose(
        expected_t,
        sparse_ops.sparse_tensor_dense_matmul(
            sparse_t, dense_t, adjoint_a=True))

  def _testLarge(self, np_dtype):
    # [FUNCTION DOC]
    # HELPER: compares the matmul against numpy for shapes where one dimension is very large.
    # Input: np_dtype, numpy value type. Draws one large size (6000 to 19999) and two small ones
    # (1 to 9), then for each placement of the large size in (rows, inner, columns) builds a very
    # sparse left operand (entries under 0.8 zeroed) and a dense right operand, and checks all four
    # adjoint combinations. Uses numpy's global random state.
    r1 = np.random.randint(6000, 20000)
    r2 = np.random.randint(1, 10)
    r3 = np.random.randint(1, 10)

    for m, k, n in [(r1, r2, r3),
                    (r2, r1, r3),
                    (r2, r3, r1)]:
      x = _maybe_complex(np.random.rand(m, k).astype(np_dtype))
      x[np.abs(x) < 0.8] = 0

      y = _maybe_complex(np.random.randn(k, n).astype(np_dtype))

      self._testMatmul(x, y, adjoint_a=False, adjoint_b=False)
      self._testMatmul(x.transpose(), y, adjoint_a=True, adjoint_b=False)
      self._testMatmul(x, y.transpose(), adjoint_a=False, adjoint_b=True)
      self._testMatmul(
          x.transpose(), y.transpose(), adjoint_a=True, adjoint_b=True)

  # Tests setting one dimension to be a high value.
  def testLarge(self):
    # [FUNCTION DOC]
    # Scenario: seed numpy randomness, then run the large-dimension comparison for single, double,
    # complex64 and complex128 values.
    # Expected: results match numpy within tolerance for all adjoint combinations.
    np.random.seed(127)  # Repeatable results
    self._testLarge(np.float32)
    self._testLarge(np.float64)
    self._testLarge(np.complex64)
    self._testLarge(np.complex128)

  # Tests random sized matrices.
  def testFloatRandom(self):
    # [FUNCTION DOC]
    # Scenario: seed numpy randomness; eight rounds over all adjoint_a and adjoint_b combinations
    # and sparsity thresholds 0, 0.2, 0.8 and 1; each draws random dimensions from 1 to 99 and
    # single-precision operands (left zeroed below the threshold), transposed when the matching
    # adjoint flag is set, and compares the op with numpy.
    # Expected: results match numpy within tolerance, including the all-zero (threshold 1) case.
    np.random.seed(127)  # Repeatable results
    for _ in range(8):
      for adjoint_a in [True, False]:
        for adjoint_b in [True, False]:
          for thresh in [0.0, 0.2, 0.8, 1.0]:
            n, k, m = np.random.randint(1, 100, size=3)
            x = np.random.rand(n, k).astype(np.float32)
            x[x < thresh] = 0  # Make it sparse
            y = np.random.randn(k, m).astype(np.float32)
            x = x.transpose() if adjoint_a else x
            y = y.transpose() if adjoint_b else y
            self._testMatmul(x, y, adjoint_a, adjoint_b)


def _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(x, y, adjoint_a,
                                                         adjoint_b):
  # [FUNCTION DOC]
  # HELPER: benchmark helper that builds the timing loop for the regular dense matmul with the
  # left operand flagged as sparse.
  # Inputs: x and y, the two operand tensors; adjoint_a and adjoint_b, booleans giving the
  # transpose flags. Returns a function that, given an iteration count and a session (ignored),
  # builds a sequential while loop (no parallelism, no backprop) repeating the matmul with each
  # iteration depending on the previous result, and returns a one-element list holding the final
  # value to run.

  def body(t, prev):
    # [FUNCTION DOC]
    # HELPER: loop body for the dense benchmark: takes the counter t and previous result prev,
    # and after prev is computed returns the incremented counter and a new dense matmul of the
    # captured operands with sparse hint on the left operand.
    with ops.control_dependencies([prev]):
      return (t + 1, math_ops.matmul(
          x,
          y,
          transpose_a=adjoint_a,
          transpose_b=adjoint_b,
          a_is_sparse=True,
          b_is_sparse=False))

  t0 = constant_op.constant(0)
  v0 = constant_op.constant(0.0)

  def _timeit(iterations, _):
    # [FUNCTION DOC]
    # HELPER: builds the benchmark loop. Inputs: iterations, number of loop iterations (a value
    # usable in the loop condition); the second argument (a session) is unused. Returns a
    # one-element list with the final loop value.
    (_, final) = while_loop.while_loop(
        lambda t, _: t < iterations,
        body, (t0, v0),
        parallel_iterations=1,
        back_prop=False,
        shape_invariants=(tensor_shape.TensorShape(
            ()), tensor_shape.TensorShape(None)))
    return [final]

  return _timeit


def _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(x_ind, x_val, x_shape,
                                                          y, adjoint_a,
                                                          adjoint_b):
  # [FUNCTION DOC]
  # HELPER: benchmark helper that builds the timing loop for the sparse-dense matmul op.
  # Inputs: x_ind, x_val, x_shape, the indices, values and dense shape of the sparse operand; y,
  # the dense operand; adjoint_a and adjoint_b, booleans for the adjoint flags. Returns a function
  # that, given an iteration count and a session (ignored), builds a sequential while loop (no
  # parallelism, no backprop) repeating the op with each iteration depending on the previous
  # result, and returns a one-element list holding the final value to run.
  sp_x = sparse_tensor.SparseTensor(
      indices=x_ind, values=x_val, dense_shape=x_shape)

  def body(t, prev):
    # [FUNCTION DOC]
    # HELPER: loop body for the sparse benchmark: takes the counter t and previous result prev,
    # and after prev is computed returns the incremented counter and a new sparse-dense matmul of
    # the captured operands.
    with ops.control_dependencies([prev]):
      return (t + 1, sparse_ops.sparse_tensor_dense_matmul(
          sp_x, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b))

  t0 = constant_op.constant(0)
  v0 = constant_op.constant(0.0)

  def _timeit(iterations, _):
    # [FUNCTION DOC]
    # HELPER: builds the benchmark loop. Inputs: iterations, number of loop iterations (a value
    # usable in the loop condition); the second argument (a session) is unused. Returns a
    # one-element list with the final loop value.
    (_, final) = while_loop.while_loop(
        lambda t, _: t < iterations,
        body, (t0, v0),
        parallel_iterations=1,
        back_prop=False,
        shape_invariants=(tensor_shape.TensorShape(
            ()), tensor_shape.TensorShape(None)))
    return [final]

  return _timeit


def sparse_tensor_dense_vs_dense_matmul_benchmark(thresh,
                                                  m,
                                                  k,
                                                  n,
                                                  adjoint_a,
                                                  adjoint_b,
                                                  use_gpu,
                                                  skip_dense=False):
  # [FUNCTION DOC]
  # Benchmark driver (not a test): times a dense matmul with a sparse hint against the
  # sparse-dense matmul op for one configuration and prints one result line.
  # Inputs: thresh, fraction of random entries zeroed in the left operand (sparsity); m, k, n,
  # the matrix dimensions (left m by k, right k by n); adjoint_a and adjoint_b, booleans that
  # transpose the operands and set the op flags; use_gpu, whether to place ops on the first GPU
  # (otherwise the CPU); skip_dense, when true the dense measurement is skipped and reported as
  # not-a-number.
  # Seeds numpy randomness with a fixed pair, builds single-precision operands, creates a session
  # with soft placement, warms up for 10 iterations and times 200 iterations, and prints a line
  # with the density, n, GPU flag, m, k, dense time, sparse time and their ratio. Returns nothing.
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Configurable for benchmarking:
  # config.intra_op_parallelism_threads = 100
  # config.gpu_options.per_process_gpu_memory_fraction = 0.3

  np.random.seed([6, 117])  # Reproducibility
  x = np.random.rand(m, k).astype(np.float32)
  x[x < thresh] = 0
  y = np.random.randn(k, n).astype(np.float32)
  if adjoint_a:
    x = x.T
  if adjoint_b:
    y = y.T

  def _timer(sess, ops_fn, iterations):
    # [FUNCTION DOC]
    # HELPER: measures average run time per iteration. Inputs: sess, a session; ops_fn, the
    # function made by the benchmark helpers; iterations, number of loop iterations to time.
    # Runs a 10-iteration warm-up, then times one run of the requested iterations, and returns
    # the elapsed wall-clock seconds divided by the iteration count.
    # Warm in
    sess.run(ops_fn(10, sess))

    # Timing run
    start = time.time()
    sess.run(ops_fn(iterations, sess))
    end = time.time()

    return (end - start) / (1.0 * iterations)  # Average runtime per iteration

  # Using regular matmul, marking one of the matrices as dense.
  if skip_dense:
    delta_dense = float("nan")
  else:
    with session.Session(config=config, graph=ops.Graph()) as sess:
      if not use_gpu:
        with ops.device("/cpu:0"):
          x_t = constant_op.constant(x)
          y_t = constant_op.constant(y)
          ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(
              x_t, y_t, adjoint_a, adjoint_b)
      else:
        with ops.device("/device:GPU:0"):
          x_t = constant_op.constant(x)
          y_t = constant_op.constant(y)
          ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(
              x_t, y_t, adjoint_a, adjoint_b)
      delta_dense = _timer(sess, ops_fn, 200)

  # Using sparse_tensor_dense_matmul.
  with session.Session("", config=config, graph=ops.Graph()) as sess:
    if not use_gpu:
      with ops.device("/cpu:0"):
        x_ind = constant_op.constant(np.vstack(np.where(x)).astype(np.int64).T)
        x_val = constant_op.constant(x[np.where(x)])
        x_shape = constant_op.constant(np.array(x.shape).astype(np.int64))
        y_t = constant_op.constant(y)
        ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(
            x_ind, x_val, x_shape, y_t, adjoint_a, adjoint_b)
    else:
      with ops.device("/device:GPU:0"):
        x_ind = constant_op.constant(np.vstack(np.where(x)).astype(np.int64).T)
        x_val = constant_op.constant(x[np.where(x)])
        x_shape = constant_op.constant(np.array(x.shape).astype(np.int64))
        y_t = constant_op.constant(y)
        ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(
            x_ind, x_val, x_shape, y_t, adjoint_a, adjoint_b)
    delta_sparse = _timer(sess, ops_fn, 200)

  print("%g \t %d \t %s \t %d \t %d \t %g \t %g \t %g" %
        (1 - thresh, n, use_gpu, m, k, delta_dense, delta_sparse,
         delta_sparse / delta_dense))


def main(_):
  # [FUNCTION DOC]
  # Entry point for the benchmark mode. The single argument is the unused command-line argument
  # list. Prints a table header, then loops over sparsity thresholds (0.99, 0.8, 0.5, 0.2), right
  # operand widths (50, 100), CPU and GPU, and sizes 100 and 1000 for the row and inner
  # dimensions, running the benchmark driver with no adjoints and printing each result line.
  # Very large benchmarks are left commented out. Returns nothing.
  print("DenseDense MatMul (w/ Sparse Flag) vs. SparseTensorDense MatMul")
  print("Matrix sizes:")
  print("  A sparse [m, k] with % nonzero values between 1% and 80%")
  print("  B dense [k, n]")
  print("")
  print("% nnz \t n \t gpu \t m \t k \t dt(dense) \t dt(sparse) "
        "\t dt(sparse)/dt(dense)")

  for thresh in (0.99, 0.8, 0.5, 0.2):
    for n in (50, 100):
      for use_gpu in (True, False):
        for m in (100, 1000):
          for k in (100, 1000):
            sparse_tensor_dense_vs_dense_matmul_benchmark(
                thresh, m, k, n, False, False, use_gpu=use_gpu)

  # Enable for large scale benchmarks, these ones take a long time to run.
  #
  # for use_gpu in (True, False):
  #   sparse_tensor_dense_vs_dense_matmul_benchmark(
  #       thresh=0.99, m=1000000, k=1000, n=100, adjoint_a=False,
  #       adjoint_b=False, use_gpu=use_gpu, skip_dense=True)


if __name__ == "__main__":
  if "--benchmarks" in sys.argv:
    sys.argv.remove("--benchmarks")
    app.run()  # pylint: disable=no-value-for-parameter
  else:
    test.main()
