# [MODULE DOC]
# Test module for the banded triangular solve op (linear algebra ops). One test class builds
# dense triangular matrices from compact band storage, solves with numpy, and compares with the op
# for real/complex types, batches, adjoints and unknown-shape placeholders; also error cases.
# No import-time side effects; running as a script starts the test runner.
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
"""Tests for tensorflow.ops.math_ops.banded_triangular_solve."""

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


class BandedTriangularSolveOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the banded triangular solve op. Matrices are given in compact band form (one
  # row per diagonal); helpers expand them to dense triangular matrices, solve with numpy and
  # compare with the op for real and complex types, batching, placeholders and adjoint. Also
  # covers wrong dimensions and singular matrices.

  def _verifySolveAllWays(self, x, y, dtypes, batch_dims=None):
    # [FUNCTION DOC]
    # HELPER: runs the single-solve verifier for every combination of adjoint (false/true) and
    # placeholder use (true/false), always with the upper-triangular setting (lower is only
    # false). Takes banded matrix x, right-hand side y, a tuple of numpy dtypes and optional
    # list of batch dimensions. Returns nothing.
    for lower in (False,):
      for adjoint in (False, True):
        for use_placeholder in True, False:
          self._verifySolve(
              x,
              y,
              lower=lower,
              adjoint=adjoint,
              batch_dims=batch_dims,
              use_placeholder=use_placeholder,
              dtypes=dtypes)

  def _verifySolveAllWaysReal(self, x, y, batch_dims=None):
    # [FUNCTION DOC]
    # HELPER: runs all-ways verification using 32-bit and 64-bit real floating types.
    self._verifySolveAllWays(x, y, (np.float32, np.float64), batch_dims)

  def _verifySolveAllWaysComplex(self, x, y, batch_dims=None):
    # [FUNCTION DOC]
    # HELPER: runs all-ways verification using 64-bit and 128-bit complex types.
    self._verifySolveAllWays(x, y, (np.complex64, np.complex128), batch_dims)

  def _verifySolve(self,
                   x,
                   y,
                   lower=True,
                   adjoint=False,
                   batch_dims=None,
                   use_placeholder=False,
                   dtypes=(np.float32, np.float64)):
    # [FUNCTION DOC]
    # HELPER: core verifier. For each numpy dtype, casts the banded matrix x and right-hand side
    # y, expands x into a dense triangular matrix (lower or upper per the lower flag; empty
    # input is used as is), takes the conjugate transpose if adjoint is set, and tiles all
    # inputs over the optional batch dimensions. Then evaluates the banded triangular solve op
    # (optionally via placeholders with unknown shape) and asserts the result has the same shape
    # as, and is close to, numpy's dense linear solve. Defaults: lower true, adjoint false, no
    # batch, no placeholder. Op errors propagate to the caller.
    for np_type in dtypes:
      a = x.astype(np_type)
      b = y.astype(np_type)

      # Now we need to convert a to a dense triangular matrix.
      def make_diags(diags, lower=True):
        # [FUNCTION DOC]
        # HELPER: converts a compact band matrix (one row per diagonal, each of length n) into
        # a dense n by n triangular numpy matrix. In lower mode, band row i is placed on the
        # i-th sub-diagonal; in upper mode the rows are flipped and placed on super-diagonals,
        # using only the leading entries. Returns the dense matrix.
        n = len(diags[0])
        a = np.zeros(n * n, dtype=diags.dtype)
        if lower:
          for i, diag in enumerate(diags):
            a[n * i:n * n:n + 1] = diag[i:]
        else:
          diags_flip = np.flip(diags, 0)
          for i, diag in enumerate(diags_flip):
            a[i:(n - i) * n:n + 1] = diag[:(n - i)]
        return a.reshape(n, n)

      # For numpy.solve we have to explicitly zero out the strictly
      # upper or lower triangle.
      if a.size > 0:
        a_np = make_diags(a, lower=lower)
      else:
        a_np = a
      if adjoint:
        a_np = np.conj(np.transpose(a_np))

      if batch_dims is not None:
        a = np.tile(a, batch_dims + [1, 1])
        a_np = np.tile(a_np, batch_dims + [1, 1])
        b = np.tile(b, batch_dims + [1, 1])

      with self.cached_session():
        a_tf = a
        b_tf = b
        if use_placeholder:
          a_tf = array_ops.placeholder_with_default(a_tf, shape=None)
          b_tf = array_ops.placeholder_with_default(b_tf, shape=None)
        tf_ans = linalg_ops.banded_triangular_solve(
            a_tf, b_tf, lower=lower, adjoint=adjoint)
        tf_val = self.evaluate(tf_ans)
        np_ans = np.linalg.solve(a_np, b)
        self.assertEqual(np_ans.shape, tf_val.shape)
        self.assertAllClose(np_ans, tf_val)

  @test_util.run_deprecated_v1
  def testSolve(self):
    # [FUNCTION DOC]
    # Graph-mode test of real solves with one-by-one, two-by-two (one and three right-hand
    # sides) and four-by-four banded matrices; each case is verified against numpy in all
    # adjoint/placeholder combinations.
    # 1x1 matrix, single rhs.
    matrix = np.array([[0.1]])
    rhs0 = np.array([[1.]])
    self._verifySolveAllWaysReal(matrix, rhs0)
    # 2x2 matrix with 2 bands, single right-hand side.
    # Corresponds to the lower triangular
    # [[1., 0.], [3., 4.]]
    # and upper triangular
    # [[2., 1.], [0., 3.]]
    matrix = np.array([[1., 4.], [2., 3.]])
    rhs0 = np.array([[1.], [1.]])
    self._verifySolveAllWaysReal(matrix, rhs0)
    # 2x2 matrix with 2 bands, 3 right-hand sides.
    rhs1 = np.array([[1., 0., 1.], [0., 1., 1.]])
    self._verifySolveAllWaysReal(matrix, rhs1)
    # 4 x 4 matrix with 2 bands, 3 right hand sides.
    # Corresponds to the lower triangular
    # [[1.,  0., 0., 0.],
    #  [-1., 2., 0., 0.],
    #  [0., -2., 3., 0.],
    #  [0., 0., -3., 4.]]
    # and upper triangular
    # [[1.,  1., 0., 0.],
    #  [0., -1., 2., 0.],
    #  [0., 0., -2., 3.],
    #  [0., 0., 0., -3.]]
    matrix = np.array([[1., 2., 3., 4.], [1., -1., -2., -3.]])
    rhs0 = np.array([[1., 0., 1.], [0., 1., 1.], [-1., 2., 1.], [0., -1., -1.]])
    self._verifySolveAllWaysReal(matrix, rhs0)

  def testSolveBandSizeSmaller(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: both cases in the test build a matrix with three bands although the comment
    # on the first says two bands, so the two-band case is not actually exercised.
    # Uses random six-by-six banded matrices whose band count (3) is smaller than the size, with
    # random right-hand sides, and verifies against numpy.
    rhs0 = np.random.randn(6, 4)

    # 6 x 6 matrix with 2 bands. Ensure all non-zero entries.
    matrix = 2. * np.random.uniform(size=[3, 6]) + 1.
    self._verifySolveAllWaysReal(matrix, rhs0)

    # 6 x 6 matrix with 3 bands. Ensure all non-zero entries.
    matrix = 2. * np.random.uniform(size=[3, 6]) + 1.
    self._verifySolveAllWaysReal(matrix, rhs0)

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message="ROCm does not support BLAS operations for complex types")
  @test_util.run_deprecated_v1
  def testSolveComplex(self):
    # [FUNCTION DOC]
    # Graph-mode test of complex solves (skipped on ROCm builds): one-by-one and two-by-two
    # banded matrices with one or three complex right-hand sides, verified against numpy.
    # 1x1 matrix, single rhs.
    matrix = np.array([[0.1 + 1j * 0.1]])
    rhs0 = np.array([[1. + 1j]])
    self._verifySolveAllWaysComplex(matrix, rhs0)
    # 2x2 matrix with 2 bands, single right-hand side.
    # Corresponds to
    # [[1. + 1j, 0.], [4 + 1j, 2 + 1j]]
    matrix = np.array([[1., 2.], [3., 4.]]).astype(np.complex64)
    matrix += 1j * matrix
    rhs0 = np.array([[1.], [1.]]).astype(np.complex64)
    rhs0 += 1j * rhs0
    self._verifySolveAllWaysComplex(matrix, rhs0)
    # 2x2 matrix with 2 bands, 3 right-hand sides.
    rhs1 = np.array([[1., 0., 1.], [0., 1., 1.]]).astype(np.complex64)
    rhs1 += 1j * rhs1
    self._verifySolveAllWaysComplex(matrix, rhs1)

  @test_util.run_deprecated_v1
  def testSolveBatch(self):
    # [FUNCTION DOC]
    # Graph-mode test of batched real solves: two-by-two and four-by-four (three-band) matrices
    # tiled over batch shapes two-by-three and three-by-two, verified against numpy.
    matrix = np.array([[1., 2.], [3., 4.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]])
    # Batch of 2x3x2x2 matrices, 2x3x2x3 right-hand sides.
    self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[2, 3])
    # Batch of 3x2x2x2 matrices, 3x2x2x3 right-hand sides.
    self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[3, 2])

    matrix = np.array([[1., 2., 3., 4.], [-1., -2., -3., -4.],
                       [-1., 1., 2., 3.]])
    rhs = np.array([[-1., 2.], [1., 1.], [0., 1.], [2., 3.]])
    # Batch of 2x3x4x4 matrices with 3 bands, 2x3x4x2 right-hand sides.
    self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[2, 3])
    # Batch of 3x2x4x4 matrices with 3 bands, 3x2x4x2 right-hand sides.
    self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[3, 2])

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message="ROCm does not support BLAS operations for complex types")
  @test_util.run_deprecated_v1
  def testSolveBatchComplex(self):
    # [FUNCTION DOC]
    # Graph-mode test of batched complex solves (skipped on ROCm) with a two-by-two complex
    # matrix over batch shapes two-by-three and three-by-two, verified against numpy.
    matrix = np.array([[1., 2.], [3., 4.]]).astype(np.complex64)
    matrix += 1j * matrix
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]]).astype(np.complex64)
    rhs += 1j * rhs
    # Batch of 2x3x2x2 matrices, 2x3x2x3 right-hand sides.
    self._verifySolveAllWaysComplex(matrix, rhs, batch_dims=[2, 3])
    # Batch of 3x2x2x2 matrices, 3x2x2x3 right-hand sides.
    self._verifySolveAllWaysComplex(matrix, rhs, batch_dims=[3, 2])

  @test_util.run_deprecated_v1
  def testWrongDimensions(self):
    # [FUNCTION DOC]
    # Graph-mode negative test: a matrix whose size does not match the right-hand side rows, and
    # a matrix with more bands than its dimension, must each raise a value error, both without
    # and with batch dimensions.
    # The matrix should have the same number of rows as the
    # right-hand sides.
    matrix = np.array([[1., 1.], [1., 1.]])
    rhs = np.array([[1., 0.]])
    with self.cached_session():
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs)
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs, batch_dims=[2, 3])

    # Number of bands exceeds the dimension of the matrix.
    matrix = np.ones((6, 4))
    rhs = np.ones((4, 2))
    with self.cached_session():
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs)
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs, batch_dims=[2, 3])

  @test_util.run_deprecated_v1
  @test_util.disable_xla("XLA cannot throw assertion errors during a kernel.")
  def testNotInvertible(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment in the code says the GPU kernel does not check for singularity, so
    # on GPU this expectation may not hold; the test is also disabled under XLA.
    # Graph-mode test: a singular matrix (zero on the diagonal) as both matrix and right-hand
    # side must raise an op error saying the input matrix is not invertible, with and without
    # batch dimensions.
    # The input should be invertible.
    # The matrix is singular because it has a zero on the diagonal.
    # FIXME(rmlarsen): The GPU kernel does not check for singularity.
    singular_matrix = np.array([[1., 0., -1.], [-1., 0., 1.], [0., -1., 1.]])
    with self.cached_session():
      with self.assertRaisesOpError("Input matrix is not invertible."):
        self._verifySolve(singular_matrix, singular_matrix)
      with self.assertRaisesOpError("Input matrix is not invertible."):
        self._verifySolve(singular_matrix, singular_matrix, batch_dims=[2, 3])


if __name__ == "__main__":
  test.main()
