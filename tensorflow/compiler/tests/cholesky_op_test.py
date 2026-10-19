# [MODULE DOC]
# Test module for the Cholesky decomposition op on the XLA test device. Verifies factors against
# the input by multiplying back, checks batching, large and ill-conditioned matrices and error
# handling for bad shapes (separate tests for graph-mode and v2 behavior). No import-time side effects.
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
"""Tests for tensorflow.ops.tf.Cholesky."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


class CholeskyOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the Cholesky decomposition. Overrides the supported float type set to the
  # types the op is defined for, and offers helpers that verify a factor reproduces its input.

  # Cholesky defined for float64, float32, complex64, complex128
  # (https://www.tensorflow.org/api_docs/python/tf/cholesky)
  @property
  def float_types(self):
    # [FUNCTION DOC]
    # Property override: returns the set of float types supported by the base test class
    # restricted to float64, float32, complex64 and complex128.
    return set(super(CholeskyOpTest, self).float_types).intersection(
        (np.float64, np.float32, np.complex64, np.complex128))

  def _verifyCholeskyBase(self, sess, placeholder, x, chol, verification, atol):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the Cholesky tests.
    # Runs the factor and the verification product in the session with the input fed to the
    # placeholder. Asserts the product equals the input within the absolute tolerance, the factor
    # has the input's shape, and (if non-empty) every matrix in the batch is lower triangular with
    # a strictly positive diagonal. Returns nothing.
    chol_np, verification_np = sess.run([chol, verification], {placeholder: x})
    self.assertAllClose(x, verification_np, atol=atol)
    self.assertShapeEqual(x, chol)
    # Check that the cholesky is lower triangular, and has positive diagonal
    # elements.
    if chol_np.shape[-1] > 0:
      chol_reshaped = np.reshape(chol_np, (-1, chol_np.shape[-2],
                                           chol_np.shape[-1]))
      for chol_matrix in chol_reshaped:
        self.assertAllClose(chol_matrix, np.tril(chol_matrix), atol=atol)
        self.assertTrue((np.diag(chol_matrix) > 0.0).all())

  def _verifyCholesky(self, x, atol=1e-6):
    # [FUNCTION DOC]
    # HELPER: end-to-end check used by most tests.
    # Takes a numpy array x (matrix or batch of matrices) and an optional absolute tolerance
    # (default 1e-6). Computes the Cholesky factor on the test device, multiplies it by its own
    # conjugate transpose (avoiding reduced-precision matmul), and delegates to the base
    # verifier. Returns nothing.
    # Verify that LL^T == x.
    with self.session() as sess:
      placeholder = array_ops.placeholder(
          dtypes.as_dtype(x.dtype), shape=x.shape)
      with self.test_scope():
        chol = linalg_ops.cholesky(placeholder)
      verification = test_util.matmul_without_tf32(chol, chol, adjoint_b=True)
      self._verifyCholeskyBase(sess, placeholder, x, chol, verification, atol)

  def testBasic(self):
    # [FUNCTION DOC]
    # Verifies the factorization of a fixed 3x3 symmetric positive-definite matrix for every
    # supported type. Returns nothing.
    data = np.array([[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]])
    for dtype in self.float_types:
      self._verifyCholesky(data.astype(dtype))

  def testBatch(self):
    # [FUNCTION DOC]
    # For each supported type verifies batched factorization: a batch of one 2x2 matrix, a stack
    # of two, a stack of two 3x3 matrices, and ten random positive-definite 5x5 matrices
    # (built as transpose times itself, tolerance 1e-4). Uses unseeded numpy randomness. Returns nothing.
    for dtype in self.float_types:
      simple_array = np.array(
          [[[1., 0.], [0., 5.]]], dtype=dtype)  # shape (1, 2, 2)
      self._verifyCholesky(simple_array)
      self._verifyCholesky(np.vstack((simple_array, simple_array)))
      odd_sized_array = np.array(
          [[[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]], dtype=dtype)
      self._verifyCholesky(np.vstack((odd_sized_array, odd_sized_array)))

      # Generate random positive-definite matrices.
      matrices = np.random.rand(10, 5, 5).astype(dtype)
      for i in range(10):
        matrices[i] = np.dot(matrices[i].T, matrices[i])
      self._verifyCholesky(matrices, atol=1e-4)

  @test_util.run_v2_only
  def testNonSquareMatrixV2(self):
    # [FUNCTION DOC]
    # Only runs in v2 behavior. For each supported type, asserts that factorizing a 2x3 matrix and
    # a batch of 2x3 matrices raises an invalid-argument error. Returns nothing.
    for dtype in self.float_types:
      with self.assertRaises(errors.InvalidArgumentError):
        linalg_ops.cholesky(np.array([[1., 2., 3.], [3., 4., 5.]], dtype=dtype))
      with self.assertRaises(errors.InvalidArgumentError):
        linalg_ops.cholesky(
            np.array(
                [[[1., 2., 3.], [3., 4., 5.]], [[1., 2., 3.], [3., 4., 5.]]],
                dtype=dtype))

  @test_util.run_v1_only("Different error types")
  def testNonSquareMatrixV1(self):
    # [FUNCTION DOC]
    # Only runs in v1 behavior. Same inputs as the v2 variant (non-square matrix and batch) but
    # expects a value error instead of an invalid-argument error. Returns nothing.
    for dtype in self.float_types:
      with self.assertRaises(ValueError):
        linalg_ops.cholesky(np.array([[1., 2., 3.], [3., 4., 5.]], dtype=dtype))
      with self.assertRaises(ValueError):
        linalg_ops.cholesky(
            np.array(
                [[[1., 2., 3.], [3., 4., 5.]], [[1., 2., 3.], [3., 4., 5.]]],
                dtype=dtype))

  @test_util.run_v2_only
  def testWrongDimensionsV2(self):
    # [FUNCTION DOC]
    # Only runs in v2 behavior. For each supported type, asserts that factorizing a rank-1 tensor
    # raises an invalid-argument error (the same check is performed twice). The variable is named
    # tensor3 although it is rank 1. Returns nothing.
    for dtype in self.float_types:
      tensor3 = constant_op.constant([1., 2.], dtype=dtype)
      with self.assertRaises(errors.InvalidArgumentError):
        linalg_ops.cholesky(tensor3)
      with self.assertRaises(errors.InvalidArgumentError):
        linalg_ops.cholesky(tensor3)

  @test_util.run_v1_only("Different error types")
  def testWrongDimensionsV1(self):
    # [FUNCTION DOC]
    # Only runs in v1 behavior. Same as the v2 variant but expects a value error (checked twice).
    # Returns nothing.
    for dtype in self.float_types:
      tensor3 = constant_op.constant([1., 2.], dtype=dtype)
      with self.assertRaises(ValueError):
        linalg_ops.cholesky(tensor3)
      with self.assertRaises(ValueError):
        linalg_ops.cholesky(tensor3)

  def testLarge2000x2000(self):
    # [FUNCTION DOC]
    # Builds a 2000x2000 float32 matrix with a diagonal of 1 plus 1/4000 everywhere (symmetric
    # positive definite) and verifies its factorization with tolerance 1e-4. Returns nothing.
    n = 2000
    shape = (n, n)
    data = np.ones(shape).astype(np.float32) / (2.0 * n) + np.diag(
        np.ones(n).astype(np.float32))
    self._verifyCholesky(data, atol=1e-4)

  def testMatrixConditionNumbers(self):
    # [FUNCTION DOC]
    # For each supported type, builds a 20x20 symmetric matrix with a condition number of 1000:
    # takes eigenvectors of a random positive-definite matrix and recombines them with
    # exponentially spaced eigenvalues from 1 down to 1/1000. Verifies its factorization with
    # tolerance 1e-4. Uses unseeded numpy randomness. Returns nothing.
    for dtype in self.float_types:
      condition_number = 1000
      size = 20

      # Generate random positive-definite symmetric matrices, and take their
      # Eigendecomposition.
      matrix = np.random.rand(size, size)
      matrix = np.dot(matrix.T, matrix)
      _, w = np.linalg.eigh(matrix)

      # Build new Eigenvalues exponentially distributed between 1 and
      # 1/condition_number
      v = np.exp(-np.log(condition_number) * np.linspace(0, size, size) / size)
      matrix = np.dot(np.dot(w, np.diag(v)), w.T).astype(dtype)
      self._verifyCholesky(matrix, atol=1e-4)

if __name__ == "__main__":
  test.main()
