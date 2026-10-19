# [MODULE DOC]
# Unit tests for the matrix square root linear algebra op (gen_linalg_ops.matrix_square_root).
# Covers real and complex dtypes, batches, symmetric positive definite, asymmetric, identity and
# empty inputs, invalid shapes, and repeated concurrent execution. Verification is done by
# squaring the computed root and comparing with the original matrix. Importing has no side
# effects beyond defining the test class; running the file as a script starts the test runner.
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
"""Tests for tensorflow.ops.math_ops.matrix_square_root."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test


class SquareRootOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the matrix square root op. Has no shared fixtures; helper methods build
  # inputs and check that multiplying the computed root by itself reproduces the input.

  def _verifySquareRoot(self, matrix, np_type):
    # [FUNCTION DOC]
    # HELPER: core check used by the real and complex verification helpers and the tests.
    # Converts the given matrix (numpy array, possibly batched) to the given numpy dtype, computes
    # its matrix square root with the op, multiplies the root by itself (with TF32 disabled) and
    # asserts the shape matches the input and the values are close (relative tolerance 1e-4,
    # absolute 1e-3). Returns nothing; raises an assertion failure on mismatch.
    matrix = matrix.astype(np_type)

    # Verify that matmul(sqrtm(A), sqrtm(A)) = A
    sqrt = gen_linalg_ops.matrix_square_root(matrix)
    square = test_util.matmul_without_tf32(sqrt, sqrt)
    self.assertShapeEqual(matrix, square)
    self.assertAllClose(matrix, square, rtol=1e-4, atol=1e-3)

  def _verifySquareRootReal(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the square root verification on the given numpy array for both float32 and
    # float64. Returns nothing; assertion failures propagate.
    for np_type in [np.float32, np.float64]:
      self._verifySquareRoot(x, np_type)

  def _verifySquareRootComplex(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the square root verification on the given numpy array for both complex64 and
    # complex128. Returns nothing; assertion failures propagate.
    for np_type in [np.complex64, np.complex128]:
      self._verifySquareRoot(x, np_type)

  def _makeBatch(self, matrix1, matrix2):
    # [FUNCTION DOC]
    # HELPER: builds a batch input for the tests. Stacks the two given square matrices into a
    # leading axis of size two and tiles the result so the output has batch shape two by three
    # followed by the matrix dimensions. Returns the numpy array; no side effects.
    matrix_batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    return matrix_batch

  def _testMatrices(self, matrix1, matrix2):
    # [FUNCTION DOC]
    # HELPER: shared driver for the symmetric and asymmetric tests. Verifies the square root of
    # each of the two given matrices and of their batch, as real inputs. Then converts both to
    # complex64, adds the same matrix times the imaginary unit (making them complex), and verifies
    # each and the complex batch again. Returns nothing; assertion failures propagate.
    # Real
    self._verifySquareRootReal(matrix1)
    self._verifySquareRootReal(matrix2)
    self._verifySquareRootReal(self._makeBatch(matrix1, matrix2))
    matrix1 = matrix1.astype(np.complex64)
    matrix2 = matrix2.astype(np.complex64)
    matrix1 += 1j * matrix1
    matrix2 += 1j * matrix2
    self._verifySquareRootComplex(matrix1)
    self._verifySquareRootComplex(matrix2)
    self._verifySquareRootComplex(self._makeBatch(matrix1, matrix2))

  @test_util.run_without_tensor_float_32
  def testSymmetricPositiveDefinite(self):
    # [FUNCTION DOC]
    # Scenario: two 2x2 symmetric positive definite matrices. Expects the square root, squared,
    # to reproduce each matrix for real, complex and batched variants. TF32 is disabled.
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    self._testMatrices(matrix1, matrix2)

  @test_util.run_without_tensor_float_32
  def testAsymmetric(self):
    # [FUNCTION DOC]
    # Scenario: two 2x2 non-symmetric matrices that have a square root. Expects the root squared
    # to equal the input for real, complex and batched variants. TF32 is disabled.
    matrix1 = np.array([[0., 4.], [-1., 5.]])
    matrix2 = np.array([[33., 24.], [48., 57.]])
    self._testMatrices(matrix1, matrix2)

  @test_util.run_without_tensor_float_32
  def testIdentityMatrix(self):
    # [FUNCTION DOC]
    # Scenario: 2x2 and 3x3 identity matrices (real dtypes only). Expects the root squared to
    # equal the identity. TF32 is disabled.
    # 2x2
    identity = np.array([[1., 0], [0, 1.]])
    self._verifySquareRootReal(identity)
    # 3x3
    identity = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    self._verifySquareRootReal(identity)

  @test_util.run_without_tensor_float_32
  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: empty inputs, namely zero matrices of size two by two and two empty zero-by-zero
    # matrices. Expects the op to succeed and the squared output to match the empty input shape.
    self._verifySquareRootReal(np.empty([0, 2, 2]))
    self._verifySquareRootReal(np.empty([2, 0, 0]))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # Scenario: a rank-one tensor is passed to the op. Expects a value error (graph construction)
    # or an invalid-argument error (runtime), since at least two dimensions are required.
    # Runs in graph and eager modes, GPU allowed.
    # The input to the square root should be at least a 2-dimensional tensor.
    tensor = constant_op.constant([1., 2.])
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      gen_linalg_ops.matrix_square_root(tensor)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32
  def testNotSquare(self):
    # [FUNCTION DOC]
    # Scenario: a 2x3 (non-square) matrix is passed to the op and evaluated. Expects a value
    # error or an invalid-argument error. Runs in graph and eager modes, GPU allowed.
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      tensor = constant_op.constant([[1., 0., -1.], [-1., 1., 0.]])
      self.evaluate(gen_linalg_ops.matrix_square_root(tensor))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32
  def testConcurrentExecutesWithoutError(self):
    # [FUNCTION DOC]
    # Scenario: two identical 5x5 random matrices are generated from the same fixed seed, squared,
    # and their square roots computed as two independent ops evaluated together. Expects the
    # inputs to be equal and the two results to be close, showing concurrent kernels do not
    # interfere. Runs in graph and eager modes, GPU allowed.
    matrix_shape = [5, 5]
    seed = [42, 24]
    matrix1 = stateless_random_ops.stateless_random_normal(
        shape=matrix_shape, seed=seed)
    matrix2 = stateless_random_ops.stateless_random_normal(
        shape=matrix_shape, seed=seed)
    self.assertAllEqual(matrix1, matrix2)
    square1 = math_ops.matmul(matrix1, matrix1)
    square2 = math_ops.matmul(matrix2, matrix2)
    sqrt1 = gen_linalg_ops.matrix_square_root(square1)
    sqrt2 = gen_linalg_ops.matrix_square_root(square2)
    all_ops = [sqrt1, sqrt2]
    sqrt = self.evaluate(all_ops)
    self.assertAllClose(sqrt[0], sqrt[1])


if __name__ == "__main__":
  test.main()
