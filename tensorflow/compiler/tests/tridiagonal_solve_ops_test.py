# [MODULE DOC]
# Tests for the tridiagonal solve op running on an XLA device. Covers solving with one or several
# right-hand sides, the compact, sequence and matrix diagonal formats, batching, transposed rhs,
# gradients, and invalid-shape errors. Many cases adapted from the generic (non-XLA) tridiagonal
# solve test; complex numbers, pivoting and runtime invertibility errors are skipped as unsupported.
# Defines module-level sample data and two tiny constant/ones helpers; no import-time side effects
# beyond that.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tridiagonal solve ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import test

_sample_diags = np.array([[2, 1, 4, 0], [1, 3, 2, 2], [0, 1, -1, 1]],
                         dtype=np.float32)
_sample_rhs = np.array([1, 2, 3, 4], dtype=np.float32)
_sample_result = np.array([-9, 5, -4, 4], dtype=np.float32)


def _tfconst(array):
  # [FUNCTION DOC]
  # HELPER: converts array-like data into a 32-bit float constant tensor for the tests.
  return constant_op.constant(array, dtype=dtypes.float32)


def _tf_ones(shape):
  # [FUNCTION DOC]
  # HELPER: returns a 64-bit float tensor of ones with the given shape, used for shape-error tests.
  return array_ops.ones(shape, dtype=dtypes.float64)


class TridiagonalSolveOpsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the tridiagonal solve op. Contains tests with random data, many adapted
  # fixed-value tests, gradient tests and invalid-shape tests, plus private helpers that run the
  # solver on the XLA test device inside a session. Uses module-level sample diagonals, rhs and
  # expected result.
  """Test for tri-diagonal matrix related ops."""

  def testTridiagonalSolverSolves1Rhs(self):
    # [FUNCTION DOC]
    # Scenario: 8 random systems of size 11 (fixed seed) with one right-hand side each, in
    # compact format and without pivoting. Solves with placeholders on the XLA device, then
    # multiplies the tridiagonal matrix by the solution in numpy and asserts it reproduces rhs
    # within 1e-4 tolerances.
    np.random.seed(19)

    batch_size = 8
    num_dims = 11

    diagonals_np = np.random.normal(size=(batch_size, 3,
                                          num_dims)).astype(np.float32)
    rhs_np = np.random.normal(size=(batch_size, num_dims, 1)).astype(np.float32)

    with self.session() as sess, self.test_scope():
      diags = array_ops.placeholder(
          shape=(batch_size, 3, num_dims), dtype=dtypes.float32)
      rhs = array_ops.placeholder(
          shape=(batch_size, num_dims, 1), dtype=dtypes.float32)
      x_np = sess.run(
          linalg_impl.tridiagonal_solve(diags, rhs, partial_pivoting=False),
          feed_dict={
              diags: diagonals_np,
              rhs: rhs_np
          })[:, :, 0]

    superdiag_np = diagonals_np[:, 0]
    diag_np = diagonals_np[:, 1]
    subdiag_np = diagonals_np[:, 2]

    y = np.zeros((batch_size, num_dims), dtype=np.float32)

    for i in range(num_dims):
      if i == 0:
        y[:, i] = (
            diag_np[:, i] * x_np[:, i] + superdiag_np[:, i] * x_np[:, i + 1])
      elif i == num_dims - 1:
        y[:, i] = (
            subdiag_np[:, i] * x_np[:, i - 1] + diag_np[:, i] * x_np[:, i])
      else:
        y[:, i] = (
            subdiag_np[:, i] * x_np[:, i - 1] + diag_np[:, i] * x_np[:, i] +
            superdiag_np[:, i] * x_np[:, i + 1])

    self.assertAllClose(y, rhs_np[:, :, 0], rtol=1e-4, atol=1e-4)

  def testTridiagonalSolverSolvesKRhs(self):
    # [FUNCTION DOC]
    # Scenario: like the single-rhs test but with 5 right-hand sides per system. Verifies, for
    # each rhs column, that multiplying the matrix by the computed solution gives back the
    # original column within 1e-4 tolerances.
    np.random.seed(19)

    batch_size = 8
    num_dims = 11
    num_rhs = 5

    diagonals_np = np.random.normal(size=(batch_size, 3,
                                          num_dims)).astype(np.float32)
    rhs_np = np.random.normal(size=(batch_size, num_dims,
                                    num_rhs)).astype(np.float32)

    with self.session() as sess, self.test_scope():
      diags = array_ops.placeholder(
          shape=(batch_size, 3, num_dims), dtype=dtypes.float32)
      rhs = array_ops.placeholder(
          shape=(batch_size, num_dims, num_rhs), dtype=dtypes.float32)
      x_np = sess.run(
          linalg_impl.tridiagonal_solve(diags, rhs, partial_pivoting=False),
          feed_dict={
              diags: diagonals_np,
              rhs: rhs_np
          })

    superdiag_np = diagonals_np[:, 0]
    diag_np = diagonals_np[:, 1]
    subdiag_np = diagonals_np[:, 2]

    for eq in range(num_rhs):
      y = np.zeros((batch_size, num_dims), dtype=np.float32)
      for i in range(num_dims):
        if i == 0:
          y[:, i] = (
              diag_np[:, i] * x_np[:, i, eq] +
              superdiag_np[:, i] * x_np[:, i + 1, eq])
        elif i == num_dims - 1:
          y[:, i] = (
              subdiag_np[:, i] * x_np[:, i - 1, eq] +
              diag_np[:, i] * x_np[:, i, eq])
        else:
          y[:, i] = (
              subdiag_np[:, i] * x_np[:, i - 1, eq] +
              diag_np[:, i] * x_np[:, i, eq] +
              superdiag_np[:, i] * x_np[:, i + 1, eq])

      self.assertAllClose(y, rhs_np[:, :, eq], rtol=1e-4, atol=1e-4)

  # All the following is adapted from tridiagonal_solve_op_test.py
  def _test(self,
            diags,
            rhs,
            expected,
            diags_format="compact",
            transpose_rhs=False):
    # [FUNCTION DOC]
    # HELPER: main assertion helper for fixed-value tests. Inputs: diags and rhs (array-like,
    # converted to 32-bit float constants), expected (array-like result), diags_format (default
    # compact), transpose_rhs (default false). Solves without pivoting or conjugation on the XLA
    # device and asserts the result is close to expected. Returns nothing.
    with self.session() as sess, self.test_scope():
      self.assertAllClose(
          sess.run(
              linalg_impl.tridiagonal_solve(
                  _tfconst(diags),
                  _tfconst(rhs),
                  diags_format,
                  transpose_rhs,
                  conjugate_rhs=False,
                  partial_pivoting=False)),
          np.asarray(expected, dtype=np.float32))

  def _testWithDiagonalLists(self,
                             diags,
                             rhs,
                             expected,
                             diags_format="compact",
                             transpose_rhs=False):
    # [FUNCTION DOC]
    # HELPER: like the main assertion helper, but diags is a list of separate diagonal arrays,
    # each converted to a constant and passed as a sequence. Expected is converted to a constant
    # too and compared with the solution. Same optional format and transpose arguments.
    with self.session() as sess, self.test_scope():
      self.assertAllClose(
          sess.run(
              linalg_impl.tridiagonal_solve([_tfconst(x) for x in diags],
                                            _tfconst(rhs),
                                            diags_format,
                                            transpose_rhs,
                                            conjugate_rhs=False,
                                            partial_pivoting=False)),
          sess.run(_tfconst(expected)))

  def testReal(self):
    # [FUNCTION DOC]
    # Scenario: the shared 4x4 sample system in compact format; expects the shared sample result.
    self._test(diags=_sample_diags, rhs=_sample_rhs, expected=_sample_result)

  # testComplex is skipped as complex type is not yet supported.

  def test3x3(self):
    # [FUNCTION DOC]
    # Scenario: a fixed 3x3 compact system; expects the hand-computed solution.
    self._test(
        diags=[[2.0, -1.0, 0.0], [1.0, 3.0, 1.0], [0.0, -1.0, -2.0]],
        rhs=[1.0, 2.0, 3.0],
        expected=[-3.0, 2.0, 7.0])

  def test2x2(self):
    # [FUNCTION DOC]
    # Scenario: a fixed 2x2 compact system with one rhs; expects the hand-computed solution.
    self._test(
        diags=[[2.0, 0.0], [1.0, 3.0], [0.0, 1.0]],
        rhs=[1.0, 4.0],
        expected=[-5.0, 3.0])

  def test1x1(self):
    # [FUNCTION DOC]
    # Scenario: a 1x1 system (main diagonal 3, rhs 6); expects the solution 2.
    self._test(diags=[[0], [3], [0]], rhs=[6], expected=[2])

  def test0x0(self):
    # [FUNCTION DOC]
    # Scenario: an empty (zero-size) system with empty diagonals and a 0-by-1 rhs; expects an
    # empty 0-by-1 result without error.
    self._test(
        diags=np.zeros(shape=(3, 0), dtype=np.float32),
        rhs=np.zeros(shape=(0, 1), dtype=np.float32),
        expected=np.zeros(shape=(0, 1), dtype=np.float32))

  def test2x2WithMultipleRhs(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 system with three right-hand-side columns; expects the matching three
    # solution columns.
    self._test(
        diags=[[2, 0], [1, 3], [0, 1]],
        rhs=[[1, 2, 3], [4, 8, 12]],
        expected=[[-5, -10, -15], [3, 6, 9]])

  def test1x1WithMultipleRhs(self):
    # [FUNCTION DOC]
    # Scenario: a 1x1 system with a row of three right-hand sides; expects each divided by 3.
    self._test(diags=[[0], [3], [0]], rhs=[[6, 9, 12]], expected=[[2, 3, 4]])

  # test1x1NotInvertible is skipped as runtime error not raised for now.

  # test2x2NotInvertible is skipped as runtime error not raised for now.

  @test_util.disable_mlir_bridge("Error messages differ")
  def testPartialPivotingRaises(self):
    # [FUNCTION DOC]
    # Scenario: random batched system (8 systems, size 11, 5 rhs) solved with partial pivoting
    # requested. Expects an unimplemented-error whose message says pivoting is not yet supported.
    # Skipped when the MLIR bridge is used, because its error messages differ.
    np.random.seed(0)
    batch_size = 8
    num_dims = 11
    num_rhs = 5

    diagonals_np = np.random.normal(size=(batch_size, 3,
                                          num_dims)).astype(np.float32)
    rhs_np = np.random.normal(size=(batch_size, num_dims,
                                    num_rhs)).astype(np.float32)

    with self.session() as sess, self.test_scope():
      with self.assertRaisesRegex(
          errors_impl.UnimplementedError,
          "Current implementation does not yet support pivoting."):
        diags = array_ops.placeholder(
            shape=(batch_size, 3, num_dims), dtype=dtypes.float32)
        rhs = array_ops.placeholder(
            shape=(batch_size, num_dims, num_rhs), dtype=dtypes.float32)
        sess.run(
            linalg_impl.tridiagonal_solve(diags, rhs, partial_pivoting=True),
            feed_dict={
                diags: diagonals_np,
                rhs: rhs_np
            })

  # testCaseRequiringPivotingLastRows is skipped as pivoting is not supported
  # for now.

  # testNotInvertible is skipped as runtime error not raised for now.

  def testDiagonal(self):
    # [FUNCTION DOC]
    # Scenario: purely diagonal matrix (off-diagonals zero); expects rhs divided by the diagonal.
    self._test(
        diags=[[0, 0, 0, 0], [1, 2, -1, -2], [0, 0, 0, 0]],
        rhs=[1, 2, 3, 4],
        expected=[1, 1, -3, -2])

  def testUpperTriangular(self):
    # [FUNCTION DOC]
    # Scenario: upper-triangular bidiagonal matrix (sub-diagonal zero); expects the hand-computed
    # back-substitution solution.
    self._test(
        diags=[[2, 4, -1, 0], [1, 3, 1, 2], [0, 0, 0, 0]],
        rhs=[1, 6, 4, 4],
        expected=[13, -6, 6, 2])

  def testLowerTriangular(self):
    # [FUNCTION DOC]
    # Scenario: lower-triangular bidiagonal matrix (super-diagonal zero); expects the
    # hand-computed forward-substitution solution.
    self._test(
        diags=[[0, 0, 0, 0], [2, -1, 3, 1], [0, 1, 4, 2]],
        rhs=[4, 5, 6, 1],
        expected=[2, -3, 6, -11])

  def testWithTwoRightHandSides(self):
    # [FUNCTION DOC]
    # Scenario: sample system with two rhs columns (the second is double the first); expects
    # solution columns in the same ratio.
    self._test(
        diags=_sample_diags,
        rhs=np.transpose([_sample_rhs, 2 * _sample_rhs]),
        expected=np.transpose([_sample_result, 2 * _sample_result]))

  def testBatching(self):
    # [FUNCTION DOC]
    # Scenario: batch of two systems, the second with negated diagonals and doubled rhs;
    # expects the second solution to be minus two times the sample result.
    self._test(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, -2 * _sample_result]))

  def testWithTwoBatchingDimensions(self):
    # [FUNCTION DOC]
    # Scenario: a 2-by-3 grid of batched systems with varied signs and rhs scales; expects the
    # solutions scaled and signed accordingly relative to the sample result.
    self._test(
        diags=np.array([[_sample_diags, -_sample_diags, _sample_diags],
                        [-_sample_diags, _sample_diags, -_sample_diags]]),
        rhs=np.array([[_sample_rhs, 2 * _sample_rhs, 3 * _sample_rhs],
                      [4 * _sample_rhs, 5 * _sample_rhs, 6 * _sample_rhs]]),
        expected=np.array(
            [[_sample_result, -2 * _sample_result, 3 * _sample_result],
             [-4 * _sample_result, 5 * _sample_result, -6 * _sample_result]]))

  def testBatchingAndTwoRightHandSides(self):
    # [FUNCTION DOC]
    # Scenario: batch of two systems, each with two rhs columns. Expects the matching batched
    # two-column solutions (second batch entry scaled by minus two).
    rhs = np.transpose([_sample_rhs, 2 * _sample_rhs])
    expected_result = np.transpose([_sample_result, 2 * _sample_result])
    self._test(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([rhs, 2 * rhs]),
        expected=np.array([expected_result, -2 * expected_result]))

  def testSequenceFormat(self):
    # [FUNCTION DOC]
    # Scenario: diagonals passed as three separate lists (super and sub one element shorter);
    # expects the sample result.
    self._testWithDiagonalLists(
        diags=[[2, 1, 4], [1, 3, 2, 2], [1, -1, 1]],
        rhs=[1, 2, 3, 4],
        expected=[-9, 5, -4, 4],
        diags_format="sequence")

  def testSequenceFormatWithDummyElements(self):
    # [FUNCTION DOC]
    # Scenario: sequence format with padded, full-length super and sub diagonals whose unused
    # end elements hold a dummy value; expects the solver to ignore them and give the sample result.
    dummy = 20  # Should be ignored by the solver.
    self._testWithDiagonalLists(
        diags=[
            [2, 1, 4, dummy],
            [1, 3, 2, 2],
            [dummy, 1, -1, 1],
        ],
        rhs=[1, 2, 3, 4],
        expected=[-9, 5, -4, 4],
        diags_format="sequence")

  def testSequenceFormatWithBatching(self):
    # [FUNCTION DOC]
    # Scenario: sequence format with a batch of two systems (second has negated diagonals);
    # expects solutions that are negatives of each other.
    self._testWithDiagonalLists(
        diags=[[[2, 1, 4], [-2, -1, -4]], [[1, 3, 2, 2], [-1, -3, -2, -2]],
               [[1, -1, 1], [-1, 1, -1]]],
        rhs=[[1, 2, 3, 4], [1, 2, 3, 4]],
        expected=[[-9, 5, -4, 4], [9, -5, 4, -4]],
        diags_format="sequence")

  def testMatrixFormat(self):
    # [FUNCTION DOC]
    # Scenario: dense 4x4 tridiagonal matrix in matrix format; expects the sample result.
    self._test(
        diags=[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
        rhs=[1, 2, 3, 4],
        expected=[-9, 5, -4, 4],
        diags_format="matrix")

  def testMatrixFormatWithMultipleRightHandSides(self):
    # [FUNCTION DOC]
    # Scenario: matrix format with two rhs columns (second is the negation of the first);
    # expects negated solution columns.
    self._test(
        diags=[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
        rhs=[[1, -1], [2, -2], [3, -3], [4, -4]],
        expected=[[-9, 9], [5, -5], [-4, 4], [4, -4]],
        diags_format="matrix")

  def testMatrixFormatWithBatching(self):
    # [FUNCTION DOC]
    # Scenario: matrix format with a batch of two dense matrices (second is the negation of the
    # first); expects negated solutions.
    self._test(
        diags=[[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
               [[-1, -2, 0, 0], [-1, -3, -1, 0], [0, 1, -2, -4], [0, 0, -1,
                                                                  -2]]],
        rhs=[[1, 2, 3, 4], [1, 2, 3, 4]],
        expected=[[-9, 5, -4, 4], [9, -5, 4, -4]],
        diags_format="matrix")

  def testRightHandSideAsColumn(self):
    # [FUNCTION DOC]
    # Scenario: rhs given as a single-column matrix instead of a vector; expects a column result.
    self._test(
        diags=_sample_diags,
        rhs=np.transpose([_sample_rhs]),
        expected=np.transpose([_sample_result]),
        diags_format="compact")

  def testTransposeRhs(self):
    # [FUNCTION DOC]
    # Scenario: rhs supplied as two rows with transposition requested; expects the solution as the
    # transpose of the two-column sample results.
    self._test(
        diags=_sample_diags,
        rhs=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, 2 * _sample_result]).T,
        transpose_rhs=True)

  # testConjugateRhs is skipped as complex type is not yet supported.

  # testAjointRhs is skipped as complex type is not yet supported

  def testTransposeRhsWithRhsAsVector(self):
    # [FUNCTION DOC]
    # Scenario: transposition requested while rhs is a plain vector; expects the sample result.
    self._test(
        diags=_sample_diags,
        rhs=_sample_rhs,
        expected=_sample_result,
        transpose_rhs=True)

  # testConjugateRhsWithRhsAsVector is skipped as complex type is not yet
  # supported.

  def testTransposeRhsWithRhsAsVectorAndBatching(self):
    # [FUNCTION DOC]
    # Scenario: transposition requested with batched vector rhs (two systems, second negated
    # diagonals and doubled rhs); expects sample result and minus two times it.
    self._test(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, -2 * _sample_result]),
        transpose_rhs=True)

  # Gradient tests

  def _gradientTest(
      self,
      diags,
      rhs,
      y,  # output = reduce_sum(y * tridiag_solve(diags, rhs))
      expected_grad_diags,  # expected gradient of output w.r.t. diags
      expected_grad_rhs,  # expected gradient of output w.r.t. rhs
      diags_format="compact",
      transpose_rhs=False,
      feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: shared gradient assertion. Inputs: diags and rhs (array-like), y (weights), the
    # expected gradients of the weighted sum of the solution with respect to diags and rhs,
    # diags_format (default compact), transpose_rhs (default false), feed_dict (optional values
    # for placeholders). Builds the scalar sum of solution times y on the XLA device, evaluates
    # both gradients in a session and asserts they are close to the expected values.
    expected_grad_diags = np.array(expected_grad_diags).astype(np.float32)
    expected_grad_rhs = np.array(expected_grad_rhs).astype(np.float32)
    with self.session() as sess, self.test_scope():
      diags = _tfconst(diags)
      rhs = _tfconst(rhs)
      y = _tfconst(y)

      x = linalg_impl.tridiagonal_solve(
          diags,
          rhs,
          diagonals_format=diags_format,
          transpose_rhs=transpose_rhs,
          conjugate_rhs=False,
          partial_pivoting=False)

      res = math_ops.reduce_sum(x * y)
      actual_grad_diags = sess.run(
          gradient_ops.gradients(res, diags)[0], feed_dict=feed_dict)
      actual_rhs_diags = sess.run(
          gradient_ops.gradients(res, rhs)[0], feed_dict=feed_dict)
    self.assertAllClose(expected_grad_diags, actual_grad_diags)
    self.assertAllClose(expected_grad_rhs, actual_rhs_diags)

  def testGradientSimple(self):
    # [FUNCTION DOC]
    # Scenario: gradient of the weighted solution sum for the sample system with a vector rhs;
    # expects fixed hand-computed gradients for diagonals and rhs.
    self._gradientTest(
        diags=_sample_diags,
        rhs=_sample_rhs,
        y=[1, 3, 2, 4],
        expected_grad_diags=[[-5, 0, 4, 0], [9, 0, -4, -16], [0, 0, 5, 16]],
        expected_grad_rhs=[1, 0, -1, 4])

  def testGradientWithMultipleRhs(self):
    # [FUNCTION DOC]
    # Scenario: gradient check with two rhs columns and matching weights; expects fixed
    # hand-computed gradients.
    self._gradientTest(
        diags=_sample_diags,
        rhs=[[1, 2], [2, 4], [3, 6], [4, 8]],
        y=[[1, 5], [2, 6], [3, 7], [4, 8]],
        expected_grad_diags=[[-20, 28, -60, 0], [36, -35, 60, 80],
                             [0, 63, -75, -80]],
        expected_grad_rhs=[[0, 2], [1, 3], [1, 7], [0, -10]])

  def _makeDataForGradientWithBatching(self):
    # [FUNCTION DOC]
    # HELPER: builds inputs for the batched gradient test. Takes no arguments. Returns a tuple of
    # five arrays: weights, batched diagonals, batched rhs, expected diagonal gradients and expected
    # rhs gradients, all 32-bit floats with a 2-by-3 batch grid made by scaling the simple case.
    y = np.array([1, 3, 2, 4]).astype(np.float32)
    grad_diags = np.array([[-5, 0, 4, 0], [9, 0, -4, -16],
                           [0, 0, 5, 16]]).astype(np.float32)
    grad_rhs = np.array([1, 0, -1, 4]).astype(np.float32)

    diags_batched = np.array(
        [[_sample_diags, 2 * _sample_diags, 3 * _sample_diags],
         [4 * _sample_diags, 5 * _sample_diags,
          6 * _sample_diags]]).astype(np.float32)
    rhs_batched = np.array([[_sample_rhs, -_sample_rhs, _sample_rhs],
                            [-_sample_rhs, _sample_rhs,
                             -_sample_rhs]]).astype(np.float32)
    y_batched = np.array([[y, y, y], [y, y, y]]).astype(np.float32)
    expected_grad_diags_batched = np.array(
        [[grad_diags, -grad_diags / 4, grad_diags / 9],
         [-grad_diags / 16, grad_diags / 25,
          -grad_diags / 36]]).astype(np.float32)
    expected_grad_rhs_batched = np.array(
        [[grad_rhs, grad_rhs / 2, grad_rhs / 3],
         [grad_rhs / 4, grad_rhs / 5, grad_rhs / 6]]).astype(np.float32)

    return (y_batched, diags_batched, rhs_batched, expected_grad_diags_batched,
            expected_grad_rhs_batched)

  def testGradientWithBatchDims(self):
    # [FUNCTION DOC]
    # Scenario: gradient check over a 2-by-3 grid of batched systems using the generated batch
    # data; expects the stored batched gradients.
    y, diags, rhs, expected_grad_diags, expected_grad_rhs = (
        self._makeDataForGradientWithBatching())

    self._gradientTest(
        diags=diags,
        rhs=rhs,
        y=y,
        expected_grad_diags=expected_grad_diags,
        expected_grad_rhs=expected_grad_rhs)

  # testGradientWithUnknownShapes is skipped as shapes should be fully known.

  def _assertRaises(self, diags, rhs, diags_format="compact"):
    # [FUNCTION DOC]
    # HELPER: asserts that building the solve with the given diagonals, rhs and format (default
    # compact) raises a value error at graph-construction time. Returns nothing.
    with self.assertRaises(ValueError):
      linalg_impl.tridiagonal_solve(diags, rhs, diags_format)

  # Invalid input shapes
  def testInvalidShapesCompactFormat(self):
    # [FUNCTION DOC]
    # Scenario: several inconsistent shape combinations for compact-format diagonals and rhs
    # (wrong row count, mismatched sizes, wrong ranks); each must raise a value error.

    def test_raises(diags_shape, rhs_shape):
      # [FUNCTION DOC]
      # HELPER: builds ones tensors of the given shapes and asserts the solve raises in compact format.
      self._assertRaises(_tf_ones(diags_shape), _tf_ones(rhs_shape), "compact")

    test_raises((5, 4, 4), (5, 4))
    test_raises((5, 3, 4), (4, 5))
    test_raises((5, 3, 4), (5))
    test_raises((5), (5, 4))

  def testInvalidShapesSequenceFormat(self):
    # [FUNCTION DOC]
    # Scenario: inconsistent sequence-format inputs (wrong number of diagonals, wrong lengths,
    # mismatched batch sizes or rhs size); each must raise a value error.

    def test_raises(diags_tuple_shapes, rhs_shape):
      # [FUNCTION DOC]
      # HELPER: builds a tuple of ones tensors from the list of shapes and asserts the solve in
      # sequence format raises a value error.
      diagonals = tuple(_tf_ones(shape) for shape in diags_tuple_shapes)
      self._assertRaises(diagonals, _tf_ones(rhs_shape), "sequence")

    test_raises(((5, 4), (5, 4)), (5, 4))
    test_raises(((5, 4), (5, 4), (5, 6)), (5, 4))
    test_raises(((5, 3), (5, 4), (5, 6)), (5, 4))
    test_raises(((5, 6), (5, 4), (5, 3)), (5, 4))
    test_raises(((5, 4), (7, 4), (5, 4)), (5, 4))
    test_raises(((5, 4), (7, 4), (5, 4)), (3, 4))

  def testInvalidShapesMatrixFormat(self):
    # [FUNCTION DOC]
    # Scenario: non-square matrices or rhs sizes that do not match the matrix; each must raise
    # a value error in matrix format.

    def test_raises(diags_shape, rhs_shape):
      # [FUNCTION DOC]
      # HELPER: builds ones tensors of the given shapes and asserts the solve raises in matrix format.
      self._assertRaises(_tf_ones(diags_shape), _tf_ones(rhs_shape), "matrix")

    test_raises((5, 4, 7), (5, 4))
    test_raises((5, 4, 4), (3, 4))
    test_raises((5, 4, 4), (5, 3))

  # Tests involving placeholder with an unknown dimension are all skipped.
  # Dimensions have to be all known statically.


if __name__ == "__main__":
  test.main()
