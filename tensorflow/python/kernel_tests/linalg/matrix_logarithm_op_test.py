# [MODULE DOC]
# Tests and a benchmark for the matrix logarithm op (generated linear algebra op). Correctness is
# checked by applying the matrix exponential to the logarithm and comparing with the input, for
# complex types only (the op takes complex input), covering nonsymmetric and symmetric matrices,
# batches, empty inputs, random small and large matrices, error cases and repeated execution.
# The benchmark times the op on CPU. No import-time side effects other than imports.
#
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
"""Tests for tensorflow.ops.gen_linalg_ops.matrix_logarithm."""

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class LogarithmOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the matrix logarithm op: verification helpers and tests for assorted complex
  # matrices, batches, empty input, error cases and repeated execution.

  def _verifyLogarithm(self, x, np_type):
    # [FUNCTION DOC]
    # HELPER: checks the logarithm op through a round trip. Inputs: x (numpy array of matrices)
    # and np_type (complex dtype to cast to). Computes the logarithm, then its matrix exponential
    # (on GPU if available), and asserts the result is close to the input with relative tolerance
    # 1e-4 and absolute tolerance 1e-3. Returns nothing.
    inp = x.astype(np_type)
    with test_util.use_gpu():
      # Verify that expm(logm(A)) == A.
      tf_ans = linalg_impl.matrix_exponential(
          gen_linalg_ops.matrix_logarithm(inp))
      out = self.evaluate(tf_ans)
      self.assertAllClose(inp, out, rtol=1e-4, atol=1e-3)

  def _verifyLogarithmComplex(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the logarithm round-trip verification on x for both complex dtypes.
    for np_type in [np.complex64, np.complex128]:
      self._verifyLogarithm(x, np_type)

  def _makeBatch(self, matrix1, matrix2):
    # [FUNCTION DOC]
    # HELPER: stacks two matrices into a pair and tiles it into a four-dimensional batch of
    # shape 2 by 6 by matrix dimensions. Returns the numpy array.
    matrix_batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    return matrix_batch

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testNonsymmetric(self):
    # [FUNCTION DOC]
    # Test: complex versions of two nonsymmetric 2x2 matrices, and a batch of them, pass the
    # logarithm round trip for both complex dtypes. Runs in graph and eager modes, GPU allowed.
    # 2x2 matrices
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    matrix1 = matrix1.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 = matrix2.astype(np.complex64)
    matrix2 += 1j * matrix2
    self._verifyLogarithmComplex(matrix1)
    self._verifyLogarithmComplex(matrix2)
    # Complex batch
    self._verifyLogarithmComplex(self._makeBatch(matrix1, matrix2))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testSymmetricPositiveDefinite(self):
    # [FUNCTION DOC]
    # Test: complex versions of two symmetric positive definite 2x2 matrices, and a batch of
    # them, pass the logarithm round trip for both complex dtypes.
    # 2x2 matrices
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    matrix1 = matrix1.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 = matrix2.astype(np.complex64)
    matrix2 += 1j * matrix2
    self._verifyLogarithmComplex(matrix1)
    self._verifyLogarithmComplex(matrix2)
    # Complex batch
    self._verifyLogarithmComplex(self._makeBatch(matrix1, matrix2))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testNonSquareMatrix(self):
    # [FUNCTION DOC]
    # Test: the logarithm of a 2x3 complex matrix raises either a value error (graph mode) or an
    # invalid-argument error (eager mode).
    # When the logarithm of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      gen_linalg_ops.matrix_logarithm(
          np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.complex64))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variable is named tensor3 but holds a one-dimensional tensor.
    # Test: the logarithm of a one-dimensional complex tensor raises a value error or an
    # invalid-argument error.
    # The input to the logarithm should be at least a 2-dimensional tensor.
    tensor3 = constant_op.constant([1., 2.], dtype=dtypes.complex64)
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      gen_linalg_ops.matrix_logarithm(tensor3)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testEmpty(self):
    # [FUNCTION DOC]
    # Test: empty complex inputs (zero-size batch of 2x2, and batch of two 0x0 matrices) pass the
    # round-trip verification.
    self._verifyLogarithmComplex(np.empty([0, 2, 2], dtype=np.complex64))
    self._verifyLogarithmComplex(np.empty([2, 0, 0], dtype=np.complex64))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testRandomSmallAndLargeComplex64(self):
    # [FUNCTION DOC]
    # Test: seeds the global numpy random state (42), generates uniform random matrices cast to
    # 64-bit complex for batch shapes (none, 1, 3, 2 by 2) and sizes 8, 31 and 32, and verifies
    # the logarithm round trip (for both complex dtypes via the helper).
    np.random.seed(42)
    for batch_dims in [(), (1,), (3,), (2, 2)]:
      for size in 8, 31, 32:
        shape = batch_dims + (size, size)
        matrix = np.random.uniform(
            low=-1.0, high=1.0,
            size=np.prod(shape)).reshape(shape).astype(np.complex64)
        self._verifyLogarithmComplex(matrix)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testRandomSmallAndLargeComplex128(self):
    # [FUNCTION DOC]
    # Test: same as the 64-bit complex random test but the generated matrices are cast to 128-bit
    # complex. Seeds the global numpy random state with 42.
    np.random.seed(42)
    for batch_dims in [(), (1,), (3,), (2, 2)]:
      for size in 8, 31, 32:
        shape = batch_dims + (size, size)
        matrix = np.random.uniform(
            low=-1.0, high=1.0,
            size=np.prod(shape)).reshape(shape).astype(np.complex128)
        self._verifyLogarithmComplex(matrix)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testConcurrentExecutesWithoutError(self):
    # [FUNCTION DOC]
    # Test: two identical random 5x5 matrices (same stateless seed, cast to 64-bit complex) have
    # their logarithms computed in one evaluation; asserts the inputs are equal and the outputs
    # are identical.
    matrix_shape = [5, 5]
    seed = [42, 24]
    matrix1 = math_ops.cast(
        stateless_random_ops.stateless_random_normal(matrix_shape, seed=seed),
        dtypes.complex64)
    matrix2 = math_ops.cast(
        stateless_random_ops.stateless_random_normal(matrix_shape, seed=seed),
        dtypes.complex64)
    self.assertAllEqual(matrix1, matrix2)
    logm1 = gen_linalg_ops.matrix_logarithm(matrix1)
    logm2 = gen_linalg_ops.matrix_logarithm(matrix2)
    logm = self.evaluate([logm1, logm2])
    self.assertAllEqual(logm[0], logm[1])


class MatrixLogarithmBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark for the matrix logarithm op. The class attribute shapes lists the square matrix
  # shapes (with optional batch dimension) to time.

  shapes = [
      (4, 4),
      (10, 10),
      (16, 16),
      (101, 101),
      (256, 256),
      (1000, 1000),
      (1024, 1024),
      (2048, 2048),
      (513, 4, 4),
      (513, 16, 16),
      (513, 256, 256),
  ]

  def _GenerateMatrix(self, shape):
    # [FUNCTION DOC]
    # HELPER: builds benchmark input. Input: shape tuple with equal last two entries and optional
    # leading batch entries. Returns a variable holding 64-bit complex well-conditioned matrices
    # (identity plus a small constant), tiled over the batch. Raises an assertion error for
    # non-square shapes.
    batch_shape = shape[:-2]
    shape = shape[-2:]
    assert shape[0] == shape[1]
    n = shape[0]
    matrix = np.ones(shape).astype(np.complex64) / (2.0 * n) + np.diag(
        np.ones(n).astype(np.complex64))
    return variables.Variable(np.tile(matrix, batch_shape + (1, 1)))

  def benchmarkMatrixLogarithmOp(self):
    # [FUNCTION DOC]
    # Benchmark: for each shape, in a fresh graph and session pinned to CPU, builds the matrix
    # variable and times the logarithm op for at least 25 iterations. Side effects: creates graphs
    # and sessions and emits benchmark reports. No GPU variant, unlike the sibling benchmarks.
    for shape in self.shapes:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix = self._GenerateMatrix(shape)
        logm = gen_linalg_ops.matrix_logarithm(matrix)
        self.evaluate(variables.global_variables_initializer())
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(logm),
            min_iters=25,
            name="matrix_logarithm_cpu_{shape}".format(shape=shape))


if __name__ == "__main__":
  test.main()
