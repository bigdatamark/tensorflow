# [MODULE DOC]
# XLA compiler test module for the matrix triangular solve op: verifies solutions by
# multiplying back, for lower/upper and adjoint modes, batches, broadcasting, large, bfloat16
# and complex inputs, and shape errors. Runs the test runner when executed as a script.
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
"""Tests for tensorflow.ops.tf.MatrixTriangularSolve."""

import itertools

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


def MakePlaceholder(x, dtype=None):
  # [FUNCTION DOC]
  # HELPER: creates a graph placeholder shaped like the numpy array x. The element type is the
  # given dtype if provided, otherwise the type of x. Returns the placeholder tensor.
  return array_ops.placeholder(
      dtypes.as_dtype(x.dtype) if dtype is None else dtype, shape=x.shape)


class MatrixTriangularSolveOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the triangular solve op on the XLA test device. Verifies solutions by
  # multiplying the (cleaned, truly triangular) coefficient matrix by the solution and
  # comparing to the right-hand side, over lower/upper and adjoint combinations, batches,
  # broadcasting, a large matrix, low-precision and complex types, and error handling.

  #  MatrixTriangularSolve defined for float64, float32, complex64, complex128
  # (https://www.tensorflow.org/api_docs/python/tf/matrix_triangular_solve)
  @property
  def float_types(self):
    # [FUNCTION DOC]
    # Property overriding the base class set of floating types: returns the base set
    # restricted to float64, float32, complex64 and complex128 (the types the op supports).
    return set(super(MatrixTriangularSolveOpTest,
                     self).float_types).intersection(
                         (np.float64, np.float32, np.complex64, np.complex128))

  def _VerifyTriangularSolveBase(self, sess, placeholder_a, placeholder_ca,
                                 placeholder_b, a, clean_a, b, verification,
                                 atol):
    # [FUNCTION DOC]
    # HELPER: final comparison step. Inputs: a session, placeholders for the original matrix,
    # cleaned matrix and right-hand side, their numpy values, the verification tensor
    # (cleaned matrix times solution) and atol (absolute tolerance). Runs the verification
    # with the feeds, broadcasts the right-hand side to the batch shape of the matrix, and
    # asserts closeness within atol.
    feed_dict = {placeholder_a: a, placeholder_ca: clean_a, placeholder_b: b}
    verification_np = sess.run(verification, feed_dict)
    broadcasted_shape = a.shape[:-2] + (b.shape[-2], b.shape[-1])
    broadcasted_b = b + np.zeros(shape=broadcasted_shape, dtype=b.dtype)
    self.assertAllClose(broadcasted_b, verification_np, atol=atol)

  def _VerifyTriangularSolve(self, a, b, lower, adjoint, atol, dtype=None):
    # [FUNCTION DOC]
    # HELPER: solves one configuration. Inputs: a (matrix or batch), b (right-hand side),
    # lower (boolean: use lower triangle, otherwise upper), adjoint (boolean), atol, and an
    # optional dtype override for placeholders. Zeroes the other triangle of a to form the
    # clean matrix, runs the triangular solve op in the test scope on the unmodified a (the op
    # only reads the chosen triangle), multiplies the clean matrix (adjoint if requested) by
    # the solution with TF32 disabled, and compares to b via the base verification helper.
    clean_a = np.tril(a) if lower else np.triu(a)
    with self.session() as sess:
      placeholder_a = MakePlaceholder(a, dtype)
      placeholder_ca = MakePlaceholder(clean_a, dtype)
      placeholder_b = MakePlaceholder(b, dtype)
      with self.test_scope():
        x = linalg_ops.matrix_triangular_solve(
            placeholder_a, placeholder_b, lower=lower, adjoint=adjoint)
      verification = test_util.matmul_without_tf32(
          placeholder_ca, x, adjoint_a=adjoint)
      self._VerifyTriangularSolveBase(sess, placeholder_a, placeholder_ca,
                                      placeholder_b, a, clean_a, b,
                                      verification, atol)

  def _VerifyTriangularSolveCombo(self, a, b, atol=1e-4, dtype=None):
    # [FUNCTION DOC]
    # HELPER: runs the solve check for all four combinations of lower/upper and adjoint.
    # Inputs: a, b, atol (default 0.0001) and optional dtype override. For the upper case the
    # last two axes of a are swapped so that the matrix passed in is upper triangular.
    transp = lambda x: np.swapaxes(x, -1, -2)
    for lower, adjoint in itertools.product([True, False], repeat=2):
      self._VerifyTriangularSolve(
          a if lower else transp(a), b, lower, adjoint, atol, dtype=dtype)

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: random five-by-five lower triangular matrix and five-by-seven right-hand side
    # (fixed seed), solved for each supported float type in all lower/adjoint combinations.
    rng = np.random.RandomState(0)
    a = np.tril(rng.randn(5, 5))
    b = rng.randn(5, 7)
    for dtype in self.float_types:
      self._VerifyTriangularSolveCombo(a.astype(dtype), b.astype(dtype))

  def testBfloat16(self):
    # [FUNCTION DOC]
    # Scenario: same data as the basic test but with bfloat16 placeholders and a loose
    # tolerance of 0.05, in all lower/adjoint combinations.
    rng = np.random.RandomState(0)
    a = np.tril(rng.randn(5, 5))
    b = rng.randn(5, 7)
    self._VerifyTriangularSolveCombo(a, b, atol=5e-2, dtype=dtypes.bfloat16)

  def testBasicNotActuallyTriangular(self):
    # [FUNCTION DOC]
    # Scenario: a full random five-by-five matrix (not triangular) is passed; the op must only
    # use the requested triangle, which the verification (using the cleaned matrix) confirms.
    rng = np.random.RandomState(0)
    a = rng.randn(5, 5)  # the `a` matrix is not lower-triangular
    b = rng.randn(5, 7)
    for dtype in self.float_types:
      self._VerifyTriangularSolveCombo(a.astype(dtype), b.astype(dtype))

  def testBasicComplexDtypes(self):
    # [FUNCTION DOC]
    # Scenario: complex lower triangular matrix and complex right-hand side solved for every
    # complex type in all combinations. Skipped on ROCm builds because the underlying BLAS
    # triangular solve for complex types is unsupported there.

    if xla_test.test.is_built_with_rocm():
      # The following subtest invokes the call to "BlasTrsm"
      # That operation is currently not supported on the ROCm platform
      self.skipTest("BlasTrsm op for complex types is not supported in ROCm")

    rng = np.random.RandomState(0)
    a = np.tril(rng.randn(5, 5) + rng.randn(5, 5) * 1j)
    b = rng.randn(5, 7) + rng.randn(5, 7) * 1j
    for dtype in self.complex_types:
      self._VerifyTriangularSolveCombo(a.astype(dtype), b.astype(dtype))

  def testBatch(self):
    # [FUNCTION DOC]
    # Scenario: batched solves with matching batch shapes (several shapes of matrix and
    # right-hand side) for each float type, using well-conditioned near-identity triangular
    # matrices and tolerance 0.001, in all combinations.
    rng = np.random.RandomState(0)
    shapes = [((4, 3, 3), (4, 3, 5)), ((1, 2, 2), (1, 2, 1)),
              ((1, 1, 1), (1, 1, 2)), ((2, 3, 4, 4), (2, 3, 4, 1))]
    tuples = itertools.product(self.float_types, shapes)
    for dtype, (a_shape, b_shape) in tuples:
      n = a_shape[-1]
      a = np.tril(rng.rand(*a_shape) - 0.5) / (2.0 * n) + np.eye(n)
      b = rng.randn(*b_shape)
      self._VerifyTriangularSolveCombo(
          a.astype(dtype), b.astype(dtype), atol=1e-3)

  def testBatchBroadcast(self):
    # [FUNCTION DOC]
    # Scenario: batched solves where the matrix and right-hand side batch shapes differ and
    # must broadcast. Same construction and tolerance as the batch test.
    rng = np.random.RandomState(0)
    shapes = [((3, 3), (4, 3, 5)), ((1, 2, 2), (3, 2, 1)), ((1, 1), (1, 1, 2)),
              ((1, 3, 4, 4), (2, 1, 4, 1))]
    tuples = itertools.product(self.float_types, shapes)
    for dtype, (a_shape, b_shape) in tuples:
      n = a_shape[-1]
      a = np.tril(rng.rand(*a_shape) - 0.5) / (2.0 * n) + np.eye(n)
      b = rng.randn(*b_shape)
      self._VerifyTriangularSolveCombo(
          a.astype(dtype), b.astype(dtype), atol=1e-3)

  def testLarge(self):
    # [FUNCTION DOC]
    # Scenario: a 1024-by-1024 lower triangular float32 system with a square right-hand side,
    # solved once (lower, no adjoint) with tolerance 0.0001.
    n = 1024
    rng = np.random.RandomState(0)
    a = np.tril(rng.rand(n, n) - 0.5) / (2.0 * n) + np.eye(n)
    b = rng.randn(n, n)
    self._VerifyTriangularSolve(
        a.astype(np.float32), b.astype(np.float32), True, False, 1e-4)

  @test_util.disable_mlir_bridge("Error handling")
  def testNonSquareCoefficientMatrix(self):
    # [FUNCTION DOC]
    # Scenario: a three-by-four coefficient matrix with a four-by-four right-hand side for each
    # float type. Expects a value error or an invalid-argument error. Disabled for the MLIR
    # bridge because of differing error handling.
    rng = np.random.RandomState(0)
    for dtype in self.float_types:
      a = rng.randn(3, 4).astype(dtype)
      b = rng.randn(4, 4).astype(dtype)
      with self.test_scope():
        with self.assertRaises((ValueError, errors.InvalidArgumentError)):
          linalg_ops.matrix_triangular_solve(a, b)

  @test_util.run_v2_only  # Different error types
  @test_util.disable_mlir_bridge("Error handling")
  def testWrongDimensionsV2(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the same failing call is asserted twice in a row; the second check is
    # redundant.
    # Scenario (version two behavior only): a three-by-three matrix with a four-by-three
    # right-hand side must raise an invalid-argument error.
    randn = np.random.RandomState(0).randn
    for dtype in self.float_types:
      lhs = constant_op.constant(randn(3, 3), dtype=dtype)
      rhs = constant_op.constant(randn(4, 3), dtype=dtype)
      with self.assertRaises(errors.InvalidArgumentError):
        linalg_ops.matrix_triangular_solve(lhs, rhs)
      with self.assertRaises(errors.InvalidArgumentError):
        linalg_ops.matrix_triangular_solve(lhs, rhs)

  @test_util.run_v1_only("Different error types")
  @test_util.disable_mlir_bridge("Error handling")
  def testWrongDimensionsV1(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the same failing call is asserted twice in a row; the second check is
    # redundant.
    # Scenario (version one behavior only): a three-by-three matrix with a four-by-three
    # right-hand side must raise a value error.
    randn = np.random.RandomState(0).randn
    for dtype in self.float_types:
      lhs = constant_op.constant(randn(3, 3), dtype=dtype)
      rhs = constant_op.constant(randn(4, 3), dtype=dtype)
      with self.assertRaises(ValueError):
        linalg_ops.matrix_triangular_solve(lhs, rhs)
      with self.assertRaises(ValueError):
        linalg_ops.matrix_triangular_solve(lhs, rhs)


if __name__ == "__main__":
  test.main()
