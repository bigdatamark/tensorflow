# [MODULE DOC]
# Tests and a benchmark for the matrix inverse op in the linear algebra ops module. The tests
# check that the inverse (optionally adjoint) multiplied by the input gives the identity for real
# (half, single, double) and complex types, symmetric and nonsymmetric matrices, batches and empty
# inputs, and check the errors for non-square, too-low-rank and singular inputs, plus repeated
# execution consistency. The benchmark times the op over many shapes, with and without adjoint,
# on CPU and (if available) GPU. No import-time side effects other than imports.
#
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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class InverseOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the matrix inverse op: verification helpers plus tests for various matrix
  # kinds, batches, empty input, error cases and repeated execution.

  def _high_precision_matmul(self, a, b, adjoint_b):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it casts to float32 or float64, but the code only upcasts
    # half-precision inputs to float32; all other dtypes are multiplied as is, without TF32.
    # HELPER: matrix product for the verification. Inputs: a and b (tensors), adjoint_b (whether
    # to use the adjoint of b). Returns the product cast back to the dtype of a.
    """Do a higher-precision matmul, casting either to float32 or float64."""
    if a.dtype == dtypes.float16:
      a = math_ops.cast(a, dtypes.float32)
      b = math_ops.cast(b, dtypes.float32)

    ret = test_util.matmul_without_tf32(a, b, adjoint_b=adjoint_b)
    return math_ops.cast(ret, a.dtype)

  def _verifyInverse(self, x, np_type):
    # [FUNCTION DOC]
    # HELPER: checks the matrix inverse op. Inputs: x (numpy array of matrices, maybe batched or
    # empty) and np_type (dtype to cast to). For both adjoint settings, computes the inverse
    # (adjoint form when requested), multiplies it by the input (with adjoint of the second
    # operand when requested), and asserts the product is close to a tiled identity (tolerance by
    # type, looser for half precision) and that the product shape equals the input shape.
    # Returns nothing.
    for adjoint in False, True:
      y = x.astype(np_type)
      with self.cached_session(use_gpu=test_util.is_gpu_available()):
        # Verify that x^{-1} * x == Identity matrix.
        inv = linalg_ops.matrix_inverse(y, adjoint=adjoint)
        tf_ans = self._high_precision_matmul(inv, y, adjoint_b=adjoint)
        np_ans = np.identity(y.shape[-1]).astype(np_type)
        if x.ndim > 2:
          tiling = list(y.shape)
          tiling[-2:] = [1, 1]
          np_ans = np.tile(np_ans, tiling)
        out = self.evaluate(tf_ans)
        self.assertAllCloseAccordingToType(
            np_ans, out, atol=0.001, half_atol=0.1)
        self.assertShapeEqual(y, tf_ans)

  def _verifyInverseReal(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the inverse verification on x for 16-, 32- and 64-bit float types. Casting a
    # complex input to these types would drop the imaginary part.
    for np_type in [np.float16, np.float32, np.float64]:
      self._verifyInverse(x, np_type)

  def _verifyInverseComplex(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the inverse verification on x for the two complex dtypes.
    for np_type in [np.complex64, np.complex128]:
      self._verifyInverse(x, np_type)

  def _makeBatch(self, matrix1, matrix2):
    # [FUNCTION DOC]
    # HELPER: stacks two matrices into a pair and tiles it into a four-dimensional batch of
    # shape 2 by 6 by matrix dimensions. Returns the numpy array.
    matrix_batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    return matrix_batch

  def testNonsymmetric(self):
    # [FUNCTION DOC]
    # Test: inverses of two nonsymmetric 2x2 matrices and a batch of them verify for real dtypes;
    # then the same matrices made complex (adding the real part as imaginary part) verify for
    # complex dtypes.
    # 2x2 matrices
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    self._verifyInverseReal(matrix1)
    self._verifyInverseReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyInverseReal(self._makeBatch(matrix1, matrix2))
    matrix1 = matrix1.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 = matrix2.astype(np.complex64)
    matrix2 += 1j * matrix2
    self._verifyInverseComplex(matrix1)
    self._verifyInverseComplex(matrix2)
    # Complex batch
    self._verifyInverseComplex(self._makeBatch(matrix1, matrix2))

  def testSymmetricPositiveDefinite(self):
    # [FUNCTION DOC]
    # Test: same as the nonsymmetric test but with two symmetric positive definite 2x2 matrices
    # (real, batched, then complex versions).
    # 2x2 matrices
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    self._verifyInverseReal(matrix1)
    self._verifyInverseReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyInverseReal(self._makeBatch(matrix1, matrix2))
    matrix1 = matrix1.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 = matrix2.astype(np.complex64)
    matrix2 += 1j * matrix2
    self._verifyInverseComplex(matrix1)
    self._verifyInverseComplex(matrix2)
    # Complex batch
    self._verifyInverseComplex(self._makeBatch(matrix1, matrix2))

  @test_util.deprecated_graph_mode_only
  def testNonSquareMatrix(self):
    # [FUNCTION DOC]
    # Test: inverting a 2x3 matrix raises a value error. Deprecated graph mode only.
    # When the inverse of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises(ValueError):
      linalg_ops.matrix_inverse(np.array([[1., 2., 3.], [3., 4., 5.]]))

  @test_util.deprecated_graph_mode_only
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variable is named tensor3 but holds a one-dimensional tensor.
    # Test: inverting a one-dimensional tensor raises a value error. Deprecated graph mode only.
    # The input to the inverse should be at least a 2-dimensional tensor.
    tensor3 = constant_op.constant([1., 2.])
    with self.assertRaises(ValueError):
      linalg_ops.matrix_inverse(tensor3)

  def testNotInvertible(self):
    # [FUNCTION DOC]
    # Test: a 3x3 matrix whose rows sum to zero (singular) makes the inverse op fail at
    # evaluation with an op error saying the input is not invertible.
    # The input should be invertible.
    with self.cached_session():
      with self.assertRaisesOpError("Input is not invertible."):
        # All rows of the matrix below add to zero.
        tensor3 = constant_op.constant([[1., 0., -1.], [-1., 1., 0.],
                                        [0., -1., 1.]])
        linalg_ops.matrix_inverse(tensor3).eval()

  def testEmpty(self):
    # [FUNCTION DOC]
    # Test: empty inputs (zero-size batch of 2x2, and batch of two 0x0 matrices) verify for real
    # dtypes.
    self._verifyInverseReal(np.empty([0, 2, 2]))
    self._verifyInverseReal(np.empty([2, 0, 0]))

  def testRandomSmallAndLarge(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: loops over complex dtypes when generating the matrices but verifies with the
    # real-only helper, so complex data is cast to real types and the imaginary part is dropped.
    # Test: seeds the global numpy random state (42), generates uniform random matrices for four
    # dtypes, batch shapes (none, 1, 3, 2 by 2) and sizes 8, 31 and 32, and verifies the inverse.
    np.random.seed(42)
    for dtype in np.float32, np.float64, np.complex64, np.complex128:
      for batch_dims in [(), (1,), (3,), (2, 2)]:
        for size in 8, 31, 32:
          shape = batch_dims + (size, size)
          matrix = np.random.uniform(
              low=-1.0, high=1.0,
              size=np.prod(shape)).reshape(shape).astype(dtype)
          self._verifyInverseReal(matrix)

  @test_util.deprecated_graph_mode_only
  def testConcurrentExecutesWithoutError(self):
    # [FUNCTION DOC]
    # Test: for adjoint and non-adjoint, inverts two identical (same seed) random 5x5 matrices in
    # one evaluation and asserts each pair of results is identical. Deprecated graph mode only.
    with self.session() as sess:
      all_ops = []
      for adjoint_ in True, False:
        matrix1 = random_ops.random_normal([5, 5], seed=42)
        matrix2 = random_ops.random_normal([5, 5], seed=42)
        inv1 = linalg_ops.matrix_inverse(matrix1, adjoint=adjoint_)
        inv2 = linalg_ops.matrix_inverse(matrix2, adjoint=adjoint_)
        all_ops += [inv1, inv2]
      inv = self.evaluate(all_ops)
      self.assertAllEqual(inv[0], inv[1])
      self.assertAllEqual(inv[2], inv[3])


class MatrixInverseBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark for the matrix inverse op. The class attribute shapes lists the square matrix
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
    # leading batch entries. Returns a variable holding float32 well-conditioned matrices
    # (identity plus a small constant), tiled over the batch. Raises an assertion error for
    # non-square shapes.
    batch_shape = shape[:-2]
    shape = shape[-2:]
    assert shape[0] == shape[1]
    n = shape[0]
    matrix = np.ones(shape).astype(np.float32) / (
        2.0 * n) + np.diag(np.ones(n).astype(np.float32))
    return variables.Variable(np.tile(matrix, batch_shape + (1, 1)))

  def benchmarkMatrixInverseOp(self):
    # [FUNCTION DOC]
    # Benchmark: for each adjoint setting and shape, in a fresh graph and session on CPU, builds
    # the matrix variable and times the inverse for at least 25 iterations; repeats on the GPU
    # when available. Side effects: creates graphs and sessions and emits benchmark reports.
    for adjoint in False, True:
      for shape in self.shapes:
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/cpu:0"):
          matrix = self._GenerateMatrix(shape)
          inv = linalg_ops.matrix_inverse(matrix, adjoint=adjoint)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(inv),
              min_iters=25,
              name="matrix_inverse_cpu_{shape}_adjoint_{adjoint}".format(
                  shape=shape, adjoint=adjoint))

        if test.is_gpu_available(True):
          with ops.Graph().as_default(), \
              session.Session(config=benchmark.benchmark_config()) as sess, \
              ops.device("/gpu:0"):
            matrix = self._GenerateMatrix(shape)
            inv = linalg_ops.matrix_inverse(matrix, adjoint=adjoint)
            self.evaluate(variables.global_variables_initializer())
            self.run_op_benchmark(
                sess,
                control_flow_ops.group(inv),
                min_iters=25,
                name="matrix_inverse_gpu_{shape}_adjoint_{adjoint}".format(
                    shape=shape, adjoint=adjoint))


if __name__ == "__main__":
  test.main()
