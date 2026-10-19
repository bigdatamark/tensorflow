# [MODULE DOC]
# Unit tests and benchmarks for the tridiagonal linear solver exposed by the linalg implementation
# module (tridiagonal_solve). The tests cover real and complex dtypes, tiny matrices (0x0, 1x1,
# 2x2, 3x3), singular inputs, cases that need partial pivoting, diagonal and triangular systems,
# multiple right-hand sides, batching, the three diagonal input formats (compact, sequence, matrix),
# transposed and conjugated right-hand sides, gradients with respect to both inputs, invalid shape
# detection, and graph-mode placeholders with partially unknown shapes. A benchmark class times the
# solver on CPU (and GPU when present) with and without pivoting.
# Because the main block rewrites the test class at run time, every parameterizable test is
# replaced by a "_pivoting" variant and (unless it requires pivoting) a "_noPivoting" variant that
# sets an instance attribute telling the helpers which mode to use, then the test runner starts.
# Import-time side effects: defines module-level sample data and two flag-name constants. Not a
# helper module; it is a standalone test file.
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
"""Tests for tensorflow.ops.linalg.linalg_impl.tridiagonal_solve."""

import itertools

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

_sample_diags = np.array([[2, 1, 4, 0], [1, 3, 2, 2], [0, 1, -1, 1]])
_sample_rhs = np.array([1, 2, 3, 4])
_sample_result = np.array([-9, 5, -4, 4])

# Flag, indicating that test should be run only with partial_pivoting=True
FLAG_REQUIRES_PIVOTING = "FLAG_REQUIRES_PIVOT"

# Flag, indicating that test shouldn't be parameterized by different values of
# partial_pivoting, etc.
FLAG_NO_PARAMETERIZATION = "FLAG_NO_PARAMETERIZATION"


def flags(*args):
  # [FUNCTION DOC]
  # HELPER: test-marking utility for the pivoting parameterization done in the main block.
  # Decorator factory. Takes any number of flag names (strings) and returns a decorator that sets
  # each flag name as a true attribute on the decorated test function and returns that same function
  # unchanged. The main block later inspects these attributes to decide whether a test is skipped
  # from parameterization or only run with pivoting. Raises nothing; mutates the decorated function.

  def decorator(f):
    # [FUNCTION DOC]
    # HELPER: inner decorator produced by the flags factory.
    # Receives a test function, marks it with every flag collected by the enclosing factory (sets an
    # attribute named after each flag to true) and returns the same function. Side effect: mutates the
    # function object.
    for flag in args:
      setattr(f, flag, True)
    return f

  return decorator


def _tfconst(array):
  # [FUNCTION DOC]
  # HELPER: test data conversion utility.
  # Converts an array-like value (nested list or numpy array) into a 64-bit floating point constant
  # tensor. If the input is the null value, nothing is created and the null value (implicitly) is
  # returned. Used so that expected values and inputs share one dtype.
  if array is not None:
    return constant_op.constant(array, dtypes.float64)


def _tf_ones(shape):
  # [FUNCTION DOC]
  # HELPER: builds dummy inputs for the invalid-shape tests.
  # Takes a shape (tuple of dimension sizes) and returns a 64-bit floating point tensor of that shape
  # filled with ones. No side effects.
  return array_ops.ones(shape, dtype=dtypes.float64)


class TridiagonalSolveOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the tridiagonal solve function of the linalg implementation module.
  # Holds helper methods that run the solver and compare with expected values (_test, _testWithLists,
  # _assertRaises, _gradientTest, _gradientTestWithLists, _testWithPlaceholders) and many test methods.
  # An optional instance attribute named pivoting (set by the main block's wrapper) selects whether
  # partial pivoting is used; when it is absent the value defaults to true in some helpers and is read
  # directly (and so would fail) in others. Also nests a benchmark class. Derives from the standard
  # TensorFlow test case; instances are created by the test runner.

  def _test(self,
            diags,
            rhs,
            expected,
            diags_format="compact",
            transpose_rhs=False,
            conjugate_rhs=False):
    # [FUNCTION DOC]
    # HELPER: core assertion helper for the solver tests.
    # Inputs: diags (tensor of diagonals), rhs (right-hand side tensor), expected (expected solution
    # tensor, or the null value meaning that the system is singular and every result entry must be
    # non-finite), diags_format (name of the diagonal layout, default compact), transpose_rhs and
    # conjugate_rhs (booleans, default false). Runs the solver inside a cached session using the
    # pivoting mode stored on the test instance (default true) and evaluates the result. When the XLA
    # backend is enabled and pivoting is requested, it silently returns without checking anything
    # because XLA does not support pivoting. Otherwise compares with expected using approximate
    # equality, or when expected is null asserts that no result entry is finite. Returns nothing; fails
    # the test through assertions.
    with self.cached_session():
      pivoting = True
      if hasattr(self, "pivoting"):
        pivoting = self.pivoting
      if test_util.is_xla_enabled() and pivoting:
        # Pivoting is not supported by xla backends.
        return
      result = linalg_impl.tridiagonal_solve(
          diags,
          rhs,
          diags_format,
          transpose_rhs,
          conjugate_rhs,
          partial_pivoting=pivoting)
      result = self.evaluate(result)
      if expected is None:
        self.assertAllEqual(
            np.zeros_like(result, dtype=np.bool_), np.isfinite(result))
      else:
        self.assertAllClose(result, expected)

  def _testWithLists(self,
                     diags,
                     rhs,
                     expected=None,
                     diags_format="compact",
                     transpose_rhs=False,
                     conjugate_rhs=False):
    # [FUNCTION DOC]
    # HELPER: convenience wrapper around _test for plain Python/numpy data.
    # Converts diags, rhs and expected (expected may be the null value, default) to 64-bit float
    # constant tensors and forwards them together with diags_format (default compact), transpose_rhs
    # and conjugate_rhs to _test. Returns nothing.
    self._test(
        _tfconst(diags), _tfconst(rhs), _tfconst(expected), diags_format,
        transpose_rhs, conjugate_rhs)

  def _assertRaises(self, diags, rhs, diags_format="compact"):
    # [FUNCTION DOC]
    # HELPER: asserts that the solver rejects malformed inputs.
    # Inputs: diags, rhs and diags_format (default compact). Uses the pivoting mode on the test instance
    # (default true). If XLA is enabled with pivoting it returns immediately without testing. Otherwise
    # it expects the solver call to raise a value error and fails the test if it does not. Returns
    # nothing.
    pivoting = True
    if hasattr(self, "pivoting"):
      pivoting = self.pivoting
    if test_util.is_xla_enabled() and pivoting:
      # Pivoting is not supported by xla backends.
      return
    with self.assertRaises(ValueError):
      linalg_impl.tridiagonal_solve(
          diags, rhs, diags_format, partial_pivoting=pivoting)

  # Tests with various dtypes

  def testReal(self):
    # [FUNCTION DOC]
    # Scenario: the sample 4x4 system solved in single and double precision floating point.
    # Builds the compact diagonals, right-hand side and expected solution as constants of each dtype and
    # asserts the solver result matches the expected solution.
    for dtype in dtypes.float32, dtypes.float64:
      self._test(
          diags=constant_op.constant(_sample_diags, dtype),
          rhs=constant_op.constant(_sample_rhs, dtype),
          expected=constant_op.constant(_sample_result, dtype))

  def testComplex(self):
    # [FUNCTION DOC]
    # Scenario: the sample system with complex dtypes (complex64 and complex128).
    # Diagonals are scaled by one plus i and the right-hand side by one minus i, so the expected
    # solution is the real sample solution scaled by the ratio of those factors. Asserts the solver
    # result matches.
    for dtype in dtypes.complex64, dtypes.complex128:
      self._test(
          diags=constant_op.constant(_sample_diags, dtype) * (1 + 1j),
          rhs=constant_op.constant(_sample_rhs, dtype) * (1 - 1j),
          expected=constant_op.constant(_sample_result, dtype) * (1 - 1j) /
          (1 + 1j))

  # Tests with small matrix sizes

  def test3x3(self):
    # [FUNCTION DOC]
    # Scenario: a 3x3 system given in compact format with a known solution. Asserts the solver output
    # equals the expected vector.
    self._testWithLists(
        diags=[[2, -1, 0], [1, 3, 1], [0, -1, -2]],
        rhs=[1, 2, 3],
        expected=[-3, 2, 7])

  def test2x2(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 system in compact format with one right-hand side vector. Asserts the solver
    # returns the expected solution.
    self._testWithLists(
        diags=[[2, 0], [1, 3], [0, 1]], rhs=[1, 4], expected=[-5, 3])

  def test2x2Complex(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 purely imaginary system for both complex dtypes. Asserts the solver returns the
    # expected complex solution.
    for dtype in dtypes.complex64, dtypes.complex128:
      self._test(
          diags=constant_op.constant([[2j, 0j], [1j, 3j], [0j, 1j]], dtype),
          rhs=constant_op.constant([1 - 1j, 4 - 4j], dtype),
          expected=constant_op.constant([5 + 5j, -3 - 3j], dtype))

  def test1x1(self):
    # [FUNCTION DOC]
    # Scenario: a 1x1 system (single diagonal entry three, right-hand side six). Asserts the solution
    # is two.
    self._testWithLists(diags=[[0], [3], [0]], rhs=[6], expected=[2])

  def test0x0(self):
    # [FUNCTION DOC]
    # Scenario: an empty (zero-size) system with 32-bit float inputs. Skipped (returns early) when XLA
    # is enabled because slicing zero-length tensors crashes there. Otherwise asserts the solver returns
    # an empty zero-by-one result without error.
    if test_util.is_xla_enabled():
      # The following test crashes with XLA due to slicing 0 length tensors.
      return
    self._test(
        diags=constant_op.constant(0, shape=(3, 0), dtype=dtypes.float32),
        rhs=constant_op.constant(0, shape=(0, 1), dtype=dtypes.float32),
        expected=constant_op.constant(0, shape=(0, 1), dtype=dtypes.float32))

  def test2x2WithMultipleRhs(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 system with three right-hand side columns. Asserts each column of the result
    # matches the expected solution.
    self._testWithLists(
        diags=[[2, 0], [1, 3], [0, 1]],
        rhs=[[1, 2, 3], [4, 8, 12]],
        expected=[[-5, -10, -15], [3, 6, 9]])

  def test1x1WithMultipleRhs(self):
    # [FUNCTION DOC]
    # Scenario: a 1x1 system with a row of three right-hand sides. Asserts the three solutions are
    # two, three and four.
    self._testWithLists(
        diags=[[0], [3], [0]], rhs=[[6, 9, 12]], expected=[[2, 3, 4]])

  def test1x1NotInvertible(self):
    # [FUNCTION DOC]
    # Scenario: a 1x1 zero matrix (singular). Skipped when XLA is enabled because XLA does not check
    # invertibility. Otherwise asserts that all entries of the result are non-finite.
    if test_util.is_xla_enabled():
      # XLA implementation does not check invertibility.
      return
    self._testWithLists(diags=[[0], [0], [0]], rhs=[[6, 9, 12]])

  def test2x2NotInvertible(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 singular system. Skipped under XLA (no invertibility check there). Otherwise
    # asserts that the result contains no finite entries.
    if test_util.is_xla_enabled():
      # XLA implementation does not check invertibility.
      return
    self._testWithLists(diags=[[3, 0], [1, 3], [0, 1]], rhs=[1, 4])

  # Other edge cases

  @flags(FLAG_REQUIRES_PIVOTING)
  def testCaseRequiringPivoting(self):
    # [FUNCTION DOC]
    # Scenario: a 4x4 system whose leading pivot would make a pivot-free method (such as the Thomas
    # algorithm) fail. Marked with the requires-pivoting flag so only the pivoting variant is generated.
    # Asserts the solver returns the expected solution.
    # Without partial pivoting (e.g. Thomas algorithm) this would fail.
    self._testWithLists(
        diags=[[2, -1, 1, 0], [1, 4, 1, -1], [0, 2, -2, 3]],
        rhs=[1, 2, 3, 4],
        expected=[8, -3.5, 0, -4])

  @flags(FLAG_REQUIRES_PIVOTING)
  def testCaseRequiringPivotingLastRows(self):
    # [FUNCTION DOC]
    # Scenario: a 4x4 system where pivoting is required in the last rows. Marked requires-pivoting.
    # Asserts the solver returns the expected solution.
    self._testWithLists(
        diags=[[2, 1, -1, 0], [1, -1, 2, 1], [0, 1, -6, 1]],
        rhs=[1, 2, -1, -2],
        expected=[5, -2, -5, 3])

  def testNotInvertible(self):
    # [FUNCTION DOC]
    # Scenario: a 4x4 singular system. Skipped under XLA. Otherwise asserts that every result entry is
    # non-finite (no expected value is given).
    if test_util.is_xla_enabled():
      return
    self._testWithLists(
        diags=[[2, -1, 1, 0], [1, 4, 1, -1], [0, 2, 0, 3]], rhs=[1, 2, 3, 4])

  def testDiagonal(self):
    # [FUNCTION DOC]
    # Scenario: a matrix with only a main diagonal (off-diagonals zero). Asserts the solution is the
    # right-hand side divided elementwise by the diagonal.
    self._testWithLists(
        diags=[[0, 0, 0, 0], [1, 2, -1, -2], [0, 0, 0, 0]],
        rhs=[1, 2, 3, 4],
        expected=[1, 1, -3, -2])

  def testUpperTriangular(self):
    # [FUNCTION DOC]
    # Scenario: an upper bidiagonal system (lower diagonal zero). Asserts the expected solution.
    self._testWithLists(
        diags=[[2, 4, -1, 0], [1, 3, 1, 2], [0, 0, 0, 0]],
        rhs=[1, 6, 4, 4],
        expected=[13, -6, 6, 2])

  def testLowerTriangular(self):
    # [FUNCTION DOC]
    # Scenario: a lower bidiagonal system (upper diagonal zero). Asserts the expected solution.
    self._testWithLists(
        diags=[[0, 0, 0, 0], [2, -1, 3, 1], [0, 1, 4, 2]],
        rhs=[4, 5, 6, 1],
        expected=[2, -3, 6, -11])

  # Multiple right-hand sides and batching

  def testWithTwoRightHandSides(self):
    # [FUNCTION DOC]
    # Scenario: the sample matrix with two right-hand side columns (the sample and its double).
    # Asserts the result columns are the sample solution and its double.
    self._testWithLists(
        diags=_sample_diags,
        rhs=np.transpose([_sample_rhs, 2 * _sample_rhs]),
        expected=np.transpose([_sample_result, 2 * _sample_result]))

  def testBatching(self):
    # [FUNCTION DOC]
    # Scenario: a batch of two systems, the sample and its negation, with right-hand sides of one and
    # two times the sample. Asserts the batch of solutions (sample solution and minus two times it).
    self._testWithLists(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, -2 * _sample_result]))

  def testWithTwoBatchingDimensions(self):
    # [FUNCTION DOC]
    # Scenario: inputs with a two-dimensional batch (2 by 3) of scaled sample systems with scaled
    # right-hand sides. Asserts each batch entry's solution matches the correspondingly scaled sample
    # solution.
    self._testWithLists(
        diags=np.array([[_sample_diags, -_sample_diags, _sample_diags],
                        [-_sample_diags, _sample_diags, -_sample_diags]]),
        rhs=np.array([[_sample_rhs, 2 * _sample_rhs, 3 * _sample_rhs],
                      [4 * _sample_rhs, 5 * _sample_rhs, 6 * _sample_rhs]]),
        expected=np.array(
            [[_sample_result, -2 * _sample_result, 3 * _sample_result],
             [-4 * _sample_result, 5 * _sample_result, -6 * _sample_result]]))

  def testBatchingAndTwoRightHandSides(self):
    # [FUNCTION DOC]
    # Scenario: a batch of two systems each with two right-hand side columns. Asserts the batched
    # solutions match the expected scaled results.
    rhs = np.transpose([_sample_rhs, 2 * _sample_rhs])
    expected_result = np.transpose([_sample_result, 2 * _sample_result])
    self._testWithLists(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([rhs, 2 * rhs]),
        expected=np.array([expected_result, -2 * expected_result]))

  # Various input formats

  def testSequenceFormat(self):
    # [FUNCTION DOC]
    # Scenario: diagonals supplied in sequence format, a tuple of (superdiagonal with three entries,
    # main diagonal, subdiagonal with three entries). Asserts the solution equals the sample solution.
    self._test(
        diags=(_tfconst([2, 1, 4]), _tfconst([1, 3, 2, 2]), _tfconst([1, -1,
                                                                      1])),
        rhs=_tfconst([1, 2, 3, 4]),
        expected=_tfconst([-9, 5, -4, 4]),
        diags_format="sequence")

  def testSequenceFormatWithDummyElements(self):
    # [FUNCTION DOC]
    # Scenario: sequence format where the super and sub diagonals are padded to full length with a
    # dummy value (twenty). Asserts the padding is ignored and the sample solution is returned.
    dummy = 20
    self._test(
        diags=(_tfconst([2, 1, 4,
                         dummy]), _tfconst([1, 3, 2,
                                            2]), _tfconst([dummy, 1, -1, 1])),
        rhs=_tfconst([1, 2, 3, 4]),
        expected=_tfconst([-9, 5, -4, 4]),
        diags_format="sequence")

  def testSequenceFormatWithBatching(self):
    # [FUNCTION DOC]
    # Scenario: sequence format with a batch of two systems (second is the negation of the first).
    # Asserts the batched solutions.
    self._test(
        diags=(_tfconst([[2, 1, 4], [-2, -1, -4]]),
               _tfconst([[1, 3, 2, 2],
                         [-1, -3, -2, -2]]), _tfconst([[1, -1, 1], [-1, 1,
                                                                    -1]])),
        rhs=_tfconst([[1, 2, 3, 4], [1, 2, 3, 4]]),
        expected=_tfconst([[-9, 5, -4, 4], [9, -5, 4, -4]]),
        diags_format="sequence")

  def testMatrixFormat(self):
    # [FUNCTION DOC]
    # Scenario: the sample system given as a full 4x4 dense tridiagonal matrix using the matrix
    # format. Asserts the sample solution.
    self._testWithLists(
        diags=[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
        rhs=[1, 2, 3, 4],
        expected=[-9, 5, -4, 4],
        diags_format="matrix")

  def testMatrixFormatWithMultipleRightHandSides(self):
    # [FUNCTION DOC]
    # Scenario: matrix format with two right-hand side columns (one and its negation). Asserts the
    # solution columns match.
    self._testWithLists(
        diags=[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
        rhs=[[1, -1], [2, -2], [3, -3], [4, -4]],
        expected=[[-9, 9], [5, -5], [-4, 4], [4, -4]],
        diags_format="matrix")

  def testMatrixFormatWithBatching(self):
    # [FUNCTION DOC]
    # Scenario: matrix format with a batch of two matrices (second is the negation of the first).
    # Asserts the batched solutions.
    self._testWithLists(
        diags=[[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]],
               [[-1, -2, 0, 0], [-1, -3, -1, 0], [0, 1, -2, -4], [0, 0, -1,
                                                                  -2]]],
        rhs=[[1, 2, 3, 4], [1, 2, 3, 4]],
        expected=[[-9, 5, -4, 4], [9, -5, 4, -4]],
        diags_format="matrix")

  def testRightHandSideAsColumn(self):
    # [FUNCTION DOC]
    # Scenario: compact format with the right-hand side given as an explicit single column (4 by 1).
    # Asserts the result is the corresponding column.
    self._testWithLists(
        diags=_sample_diags,
        rhs=np.transpose([_sample_rhs]),
        expected=np.transpose([_sample_result]),
        diags_format="compact")

  # Tests with transpose and adjoint

  def testTransposeRhs(self):
    # [FUNCTION DOC]
    # Scenario: the right-hand side is given as rows (two by four) and transpose_rhs is enabled.
    # Asserts the solution has the transposed layout (columns are solutions).
    self._testWithLists(
        diags=_sample_diags,
        rhs=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, 2 * _sample_result]).T,
        transpose_rhs=True)

  def testConjugateRhs(self):
    # [FUNCTION DOC]
    # Scenario: complex right-hand side columns with conjugate_rhs enabled. Asserts the solution equals
    # the real sample solution scaled by the complex conjugate factors.
    self._testWithLists(
        diags=_sample_diags,
        rhs=np.transpose([_sample_rhs * (1 + 1j), _sample_rhs * (1 - 2j)]),
        expected=np.transpose(
            [_sample_result * (1 - 1j), _sample_result * (1 + 2j)]),
        conjugate_rhs=True)

  def testAdjointRhs(self):
    # [FUNCTION DOC]
    # Scenario: complex right-hand sides given as rows with both transpose_rhs and conjugate_rhs set
    # (adjoint). Asserts the expected transposed, conjugated solutions.
    self._testWithLists(
        diags=_sample_diags,
        rhs=np.array([_sample_rhs * (1 + 1j), _sample_rhs * (1 - 2j)]),
        expected=np.array(
            [_sample_result * (1 - 1j), _sample_result * (1 + 2j)]).T,
        transpose_rhs=True,
        conjugate_rhs=True)

  def testTransposeRhsWithBatching(self):
    # [FUNCTION DOC]
    # Scenario: batched systems with a batch of two-row right-hand sides and transpose_rhs enabled.
    # Asserts the result has the expected batched transposed layout.
    self._testWithLists(
        diags=np.array([_sample_diags, -_sample_diags]),
        rhs=np.array([[_sample_rhs, 2 * _sample_rhs],
                      [3 * _sample_rhs, 4 * _sample_rhs]]),
        expected=np.array([[_sample_result, 2 * _sample_result],
                           [-3 * _sample_result,
                            -4 * _sample_result]]).transpose(0, 2, 1),
        transpose_rhs=True)

  def testTransposeRhsWithRhsAsVector(self):
    # [FUNCTION DOC]
    # Scenario: transpose_rhs enabled but the right-hand side is a plain vector. Asserts the solver
    # still returns the sample solution (transposing a vector has no effect).
    self._testWithLists(
        diags=_sample_diags,
        rhs=_sample_rhs,
        expected=_sample_result,
        transpose_rhs=True)

  def testConjugateRhsWithRhsAsVector(self):
    # [FUNCTION DOC]
    # Scenario: conjugate_rhs enabled with a complex vector right-hand side. Asserts the conjugated
    # scaled solution.
    self._testWithLists(
        diags=_sample_diags,
        rhs=_sample_rhs * (1 + 1j),
        expected=_sample_result * (1 - 1j),
        conjugate_rhs=True)

  def testTransposeRhsWithRhsAsVectorAndBatching(self):
    # [FUNCTION DOC]
    # Scenario: transpose_rhs enabled, batched systems whose right-hand sides are vectors. Asserts the
    # batched sample solutions.
    self._testWithLists(
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
      conjugate_rhs=False,
      feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: shared gradient checker for the gradient tests.
    # KNOWN ISSUE: under XLA the function returns from inside the nested tape contexts before the
    # solver is even called, so nothing is checked (the comment says pivoting is unsupported, but the
    # check does not depend on pivoting); it also never passes a pivoting option to the solver.
    # Inputs: diags, rhs and y (tensors), expected_grad_diags and expected_grad_rhs (array-likes of
    # expected gradients), diags_format (default compact), transpose_rhs, conjugate_rhs (booleans),
    # feed_dict (optional mapping used when running graph placeholders, default null). Converts the
    # expected gradients to constants, records the solve with two nested gradient tapes watching diags
    # and rhs, defines the scalar output as the sum of y times the solution, then in a session computes
    # the gradients of that scalar with respect to each input and asserts approximate equality with the
    # expectations. Returns nothing.
    expected_grad_diags = _tfconst(expected_grad_diags)
    expected_grad_rhs = _tfconst(expected_grad_rhs)
    with backprop.GradientTape() as tape_diags:
      with backprop.GradientTape() as tape_rhs:
        tape_diags.watch(diags)
        tape_rhs.watch(rhs)
        if test_util.is_xla_enabled():
          # Pivoting is not supported by xla backends.
          return
        x = linalg_impl.tridiagonal_solve(
            diags,
            rhs,
            diagonals_format=diags_format,
            transpose_rhs=transpose_rhs,
            conjugate_rhs=conjugate_rhs)
        res = math_ops.reduce_sum(x * y)
    with self.cached_session() as sess:
      actual_grad_diags = sess.run(
          tape_diags.gradient(res, diags), feed_dict=feed_dict)
      actual_rhs_diags = sess.run(
          tape_rhs.gradient(res, rhs), feed_dict=feed_dict)
    self.assertAllClose(expected_grad_diags, actual_grad_diags)
    self.assertAllClose(expected_grad_rhs, actual_rhs_diags)

  def _gradientTestWithLists(self,
                             diags,
                             rhs,
                             y,
                             expected_grad_diags,
                             expected_grad_rhs,
                             diags_format="compact",
                             transpose_rhs=False,
                             conjugate_rhs=False):
    # [FUNCTION DOC]
    # HELPER: wrapper around _gradientTest for plain data.
    # Converts diags, rhs and y to 64-bit float constants and forwards them with the expected gradients,
    # diags_format, transpose_rhs and conjugate_rhs. Returns nothing.
    self._gradientTest(
        _tfconst(diags), _tfconst(rhs), _tfconst(y), expected_grad_diags,
        expected_grad_rhs, diags_format, transpose_rhs, conjugate_rhs)

  def testGradientSimple(self):
    # [FUNCTION DOC]
    # Scenario: gradients of a weighted sum of the solution of the sample system with a single
    # right-hand side. Asserts the gradients with respect to diagonals and right-hand side equal fixed
    # expected values.
    self._gradientTestWithLists(
        diags=_sample_diags,
        rhs=_sample_rhs,
        y=[1, 3, 2, 4],
        expected_grad_diags=[[-5, 0, 4, 0], [9, 0, -4, -16], [0, 0, 5, 16]],
        expected_grad_rhs=[1, 0, -1, 4])

  def testGradientWithMultipleRhs(self):
    # [FUNCTION DOC]
    # Scenario: gradient check with two right-hand side columns. Asserts the expected gradients with
    # respect to the diagonals and the right-hand sides.
    self._gradientTestWithLists(
        diags=_sample_diags,
        rhs=[[1, 2], [2, 4], [3, 6], [4, 8]],
        y=[[1, 5], [2, 6], [3, 7], [4, 8]],
        expected_grad_diags=([[-20, 28, -60, 0], [36, -35, 60, 80],
                              [0, 63, -75, -80]]),
        expected_grad_rhs=[[0, 2], [1, 3], [1, 7], [0, -10]])

  def _makeDataForGradientWithBatching(self):
    # [FUNCTION DOC]
    # HELPER: builds batched data for the batched gradient tests.
    # Takes no arguments. Builds a two-by-three batch of scaled sample systems and right-hand sides, a
    # matching batch of weight vectors, and the expected gradients obtained by scaling the single-system
    # gradients appropriately. Returns a tuple of five arrays: weights, diagonals, right-hand sides,
    # expected diagonal gradients and expected right-hand side gradients.
    y = np.array([1, 3, 2, 4])
    grad_diags = np.array([[-5, 0, 4, 0], [9, 0, -4, -16], [0, 0, 5, 16]])
    grad_rhs = np.array([1, 0, -1, 4])

    diags_batched = np.array(
        [[_sample_diags, 2 * _sample_diags, 3 * _sample_diags],
         [4 * _sample_diags, 5 * _sample_diags, 6 * _sample_diags]])
    rhs_batched = np.array([[_sample_rhs, -_sample_rhs, _sample_rhs],
                            [-_sample_rhs, _sample_rhs, -_sample_rhs]])
    y_batched = np.array([[y, y, y], [y, y, y]])
    expected_grad_diags_batched = np.array(
        [[grad_diags, -grad_diags / 4, grad_diags / 9],
         [-grad_diags / 16, grad_diags / 25, -grad_diags / 36]])
    expected_grad_rhs_batched = np.array(
        [[grad_rhs, grad_rhs / 2, grad_rhs / 3],
         [grad_rhs / 4, grad_rhs / 5, grad_rhs / 6]])

    return (y_batched, diags_batched, rhs_batched, expected_grad_diags_batched,
            expected_grad_rhs_batched)

  def testGradientWithBatchDims(self):
    # [FUNCTION DOC]
    # Scenario: gradient check with a two-dimensional batch of systems using the data from the batching
    # helper. Asserts the gradients with respect to diagonals and right-hand sides match.
    y, diags, rhs, expected_grad_diags, expected_grad_rhs = \
      self._makeDataForGradientWithBatching()

    self._gradientTestWithLists(
        diags=diags,
        rhs=rhs,
        y=y,
        expected_grad_diags=expected_grad_diags,
        expected_grad_rhs=expected_grad_rhs)

  @test_util.run_deprecated_v1
  def testGradientWithUnknownShapes(self):
    # [FUNCTION DOC]
    # Scenario: graph-mode gradient check where diags, rhs and y are placeholders of known rank (4, 3
    # and 3) but entirely unknown dimensions. Values are fed through the feed dictionary and the
    # expected batched gradients are asserted. Decorated to run only in deprecated graph mode.

    def placeholder(rank):
      # [FUNCTION DOC]
      # HELPER: creates a 64-bit float placeholder of a given rank.
      # KNOWN ISSUE: the shape is built from a generator of nothing values rather than a list or tuple,
      # so the placeholder shape may not be interpreted as a fully unknown shape of that rank as the
      # intent suggests.
      # Takes rank (integer) and returns a placeholder whose shape is meant to have that many unknown
      # dimensions.
      return array_ops.placeholder(
          dtypes.float64, shape=(None for _ in range(rank)))

    y, diags, rhs, expected_grad_diags, expected_grad_rhs = \
      self._makeDataForGradientWithBatching()

    diags_placeholder = placeholder(rank=4)
    rhs_placeholder = placeholder(rank=3)
    y_placeholder = placeholder(rank=3)

    self._gradientTest(
        diags=diags_placeholder,
        rhs=rhs_placeholder,
        y=y_placeholder,
        expected_grad_diags=expected_grad_diags,
        expected_grad_rhs=expected_grad_rhs,
        feed_dict={
            diags_placeholder: diags,
            rhs_placeholder: rhs,
            y_placeholder: y
        })

  # Invalid input shapes

  @flags(FLAG_NO_PARAMETERIZATION)
  def testInvalidShapesCompactFormat(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: shapes written as a parenthesized single number (for example five) are plain integers
    # rather than one-element tuples, so these cases pass an integer shape to the ones constructor.
    # Scenario: compact-format inputs with inconsistent shapes. Defines a local checker and runs four
    # bad combinations: wrong number of diagonal rows, mismatched matrix sizes, a right-hand side of
    # wrong rank, and a diagonal array of wrong rank. Asserts a value error each time. Marked as not
    # parameterized by pivoting.

    def test_raises(diags_shape, rhs_shape):
      # [FUNCTION DOC]
      # HELPER: local checker for the compact format test.
      # Takes a diagonal shape and a right-hand side shape, builds all-ones tensors of those shapes and
      # asserts that the solver raises a value error using the compact format.
      self._assertRaises(_tf_ones(diags_shape), _tf_ones(rhs_shape), "compact")

    test_raises((5, 4, 4), (5, 4))
    test_raises((5, 3, 4), (4, 5))
    test_raises((5, 3, 4), (5))
    test_raises((5), (5, 4))

  @flags(FLAG_NO_PARAMETERIZATION)
  def testInvalidShapesSequenceFormat(self):
    # [FUNCTION DOC]
    # Scenario: sequence-format inputs with inconsistent shapes (wrong tuple length, wrong diagonal
    # lengths, mismatched batch sizes, mismatched right-hand side). Asserts a value error for each.
    # Marked as not parameterized by pivoting.

    def test_raises(diags_tuple_shapes, rhs_shape):
      # [FUNCTION DOC]
      # HELPER: local checker for the sequence format test.
      # Takes a tuple of shapes (one per diagonal) and a right-hand side shape, builds all-ones tensors,
      # and asserts that the solver raises a value error using the sequence format.
      diagonals = tuple(_tf_ones(shape) for shape in diags_tuple_shapes)
      self._assertRaises(diagonals, _tf_ones(rhs_shape), "sequence")

    test_raises(((5, 4), (5, 4)), (5, 4))
    test_raises(((5, 4), (5, 4), (5, 6)), (5, 4))
    test_raises(((5, 3), (5, 4), (5, 6)), (5, 4))
    test_raises(((5, 6), (5, 4), (5, 3)), (5, 4))
    test_raises(((5, 4), (7, 4), (5, 4)), (5, 4))
    test_raises(((5, 4), (7, 4), (5, 4)), (3, 4))

  @flags(FLAG_NO_PARAMETERIZATION)
  def testInvalidShapesMatrixFormat(self):
    # [FUNCTION DOC]
    # Scenario: matrix-format inputs that are non-square or whose size does not match the right-hand
    # side. Asserts a value error for each combination. Marked as not parameterized by pivoting.

    def test_raises(diags_shape, rhs_shape):
      # [FUNCTION DOC]
      # HELPER: local checker for the matrix format test.
      # Takes a matrix shape and a right-hand side shape, builds all-ones tensors and asserts that the
      # solver raises a value error using the matrix format.
      self._assertRaises(_tf_ones(diags_shape), _tf_ones(rhs_shape), "matrix")

    test_raises((5, 4, 7), (5, 4))
    test_raises((5, 4, 4), (3, 4))
    test_raises((5, 4, 4), (5, 3))

  # Tests with placeholders

  def _testWithPlaceholders(self,
                            diags_shape,
                            rhs_shape,
                            diags_feed,
                            rhs_feed,
                            expected,
                            diags_format="compact"):
    # [FUNCTION DOC]
    # HELPER: runs the solver on placeholders with possibly unknown shapes (graph mode only).
    # KNOWN ISSUE: reads the pivoting attribute directly from the test instance, so it raises an
    # attribute error if the test is run without the main-block wrapper that sets it.
    # Inputs: diags_shape and rhs_shape (static shapes for the placeholders, possibly with unknown
    # dimensions), diags_feed and rhs_feed (concrete arrays to feed), expected (expected result),
    # diags_format (default compact). Returns immediately without checking when executing eagerly, or
    # when XLA with pivoting is active. Otherwise builds the placeholders, calls the solver with the
    # instance's pivoting mode, runs it in a session with the fed values and asserts approximate
    # equality with expected. Returns nothing.
    if context.executing_eagerly():
      return
    diags = array_ops.placeholder(dtypes.float64, shape=diags_shape)
    rhs = array_ops.placeholder(dtypes.float64, shape=rhs_shape)
    if test_util.is_xla_enabled() and self.pivoting:
      # Pivoting is not supported by xla backends.
      return
    x = linalg_impl.tridiagonal_solve(
        diags, rhs, diags_format, partial_pivoting=self.pivoting)
    with self.cached_session() as sess:
      result = sess.run(x, feed_dict={diags: diags_feed, rhs: rhs_feed})
      self.assertAllClose(result, expected)

  @test_util.run_deprecated_v1
  def testCompactFormatAllDimsUnknown(self):
    # [FUNCTION DOC]
    # Scenario: compact format with completely unknown diagonal and right-hand side dimensions.
    # Asserts the sample solution when values are fed. Graph mode only.
    self._testWithPlaceholders(
        diags_shape=[None, None],
        rhs_shape=[None],
        diags_feed=_sample_diags,
        rhs_feed=_sample_rhs,
        expected=_sample_result)

  @test_util.run_deprecated_v1
  def testCompactFormatUnknownMatrixSize(self):
    # [FUNCTION DOC]
    # Scenario: compact format where the matrix size is unknown but the number of rows is three.
    # Asserts the sample solution. Graph mode only.
    self._testWithPlaceholders(
        diags_shape=[3, None],
        rhs_shape=[4],
        diags_feed=_sample_diags,
        rhs_feed=_sample_rhs,
        expected=_sample_result)

  @test_util.run_deprecated_v1
  def testCompactFormatUnknownRhsCount(self):
    # [FUNCTION DOC]
    # Scenario: compact format where the number of right-hand side columns is unknown. Asserts the
    # two-column solution. Graph mode only.
    self._testWithPlaceholders(
        diags_shape=[3, 4],
        rhs_shape=[4, None],
        diags_feed=_sample_diags,
        rhs_feed=np.transpose([_sample_rhs, 2 * _sample_rhs]),
        expected=np.transpose([_sample_result, 2 * _sample_result]))

  @test_util.run_deprecated_v1
  def testCompactFormatUnknownBatchSize(self):
    # [FUNCTION DOC]
    # Scenario: compact format with an unknown batch size. Asserts the batched solutions. Graph mode
    # only.
    self._testWithPlaceholders(
        diags_shape=[None, 3, 4],
        rhs_shape=[None, 4],
        diags_feed=np.array([_sample_diags, -_sample_diags]),
        rhs_feed=np.array([_sample_rhs, 2 * _sample_rhs]),
        expected=np.array([_sample_result, -2 * _sample_result]))

  @test_util.run_deprecated_v1
  def testMatrixFormatWithUnknownDims(self):
    # [FUNCTION DOC]
    # Scenario: matrix format with every combination of known and unknown matrix dimensions, and with
    # known, partly unknown or entirely unknown right-hand side shape. Does nothing in eager mode.
    # Asserts the two-column solution each time. Graph mode only.
    if context.executing_eagerly():
      return

    def test_with_matrix_shapes(matrix_shape, rhs_shape=None):
      # [FUNCTION DOC]
      # HELPER: local runner for the unknown-dimension matrix test.
      # Takes matrix_shape (static shape for the matrix placeholder) and rhs_shape (static shape for the
      # right-hand side, default null). Builds the sample matrix, a two-column right-hand side and its
      # expected solution, and delegates to the placeholder helper using the matrix format.
      matrix = np.array([[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4],
                         [0, 0, 1, 2]])
      rhs = np.array([1, 2, 3, 4])
      x = np.array([-9, 5, -4, 4])
      self._testWithPlaceholders(
          diags_shape=matrix_shape,
          rhs_shape=rhs_shape,
          diags_feed=matrix,
          rhs_feed=np.transpose([rhs, 2 * rhs]),
          expected=np.transpose([x, 2 * x]),
          diags_format="matrix")

    test_with_matrix_shapes(matrix_shape=[4, 4], rhs_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=[None, 4], rhs_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=[4, None], rhs_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=[None, None], rhs_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=[4, 4])
    test_with_matrix_shapes(matrix_shape=[None, 4])
    test_with_matrix_shapes(matrix_shape=[4, None])
    test_with_matrix_shapes(matrix_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=None, rhs_shape=[None, None])
    test_with_matrix_shapes(matrix_shape=None)

  @test_util.run_deprecated_v1
  def testSequenceFormatWithUnknownDims(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: reads the pivoting attribute directly, failing if the main-block wrapper has not set it.
    # Scenario: sequence format with placeholders of fully unknown length for the three diagonals and
    # the right-hand side. Does nothing in eager mode or under XLA with pivoting. Runs the solver with
    # the instance's pivoting mode, feeds padded diagonals and asserts the sample solution.
    if context.executing_eagerly():
      return
    if test_util.is_xla_enabled() and self.pivoting:
      # Pivoting is not supported by xla backends.
      return
    superdiag = array_ops.placeholder(dtypes.float64, shape=[None])
    diag = array_ops.placeholder(dtypes.float64, shape=[None])
    subdiag = array_ops.placeholder(dtypes.float64, shape=[None])
    rhs = array_ops.placeholder(dtypes.float64, shape=[None])

    x = linalg_impl.tridiagonal_solve((superdiag, diag, subdiag),
                                      rhs,
                                      diagonals_format="sequence",
                                      partial_pivoting=self.pivoting)
    with self.cached_session() as sess:
      result = sess.run(
          x,
          feed_dict={
              subdiag: [20, 1, -1, 1],
              diag: [1, 3, 2, 2],
              superdiag: [2, 1, 4, 20],
              rhs: [1, 2, 3, 4]
          })
      self.assertAllClose(result, [-9, 5, -4, 4])

  # Benchmark

  class TridiagonalSolveBenchmark(test.Benchmark):
    # [CLASS DOC]
    # Benchmark class (nested in the test class) measuring the tridiagonal solver. Class attributes:
    # sizes, a list of (matrix size, batch size, right-hand side count) triples, and pivoting_options,
    # a list of (flag, label) pairs. Derives from the TensorFlow benchmark base class; benchmark methods
    # are discovered by name by the benchmark runner.
    sizes = [(100000, 1, 1), (1000000, 1, 1), (10000000, 1, 1), (100000, 10, 1),
             (100000, 100, 1), (10000, 1, 10), (10000, 1, 100)]

    pivoting_options = [(True, "pivoting"), (False, "no_pivoting")]

    def _generateData(self, matrix_size, batch_size, num_rhs, seed=42):
      # [FUNCTION DOC]
      # HELPER: builds random benchmark inputs in compact diagonal format.
      # KNOWN ISSUE: this generator is never used by the benchmark methods in this file (both use the
      # matrix data generator).
      # Inputs: matrix_size, batch_size, num_rhs (integers) and seed (integer, default 42). Seeds the
      # global numpy random generator (side effect), draws normal random data, and returns a tuple of two
      # 64-bit float variables: the stacked diagonals and the right-hand sides.
      np.random.seed(seed)
      data = np.random.normal(size=(batch_size, matrix_size, 3 + num_rhs))
      diags = np.stack([data[:, :, 0], data[:, :, 1], data[:, :, 2]], axis=-2)
      rhs = data[:, :, 3:]
      return (variables.Variable(diags, dtype=dtypes.float64),
              variables.Variable(rhs, dtype=dtypes.float64))

    def _generateMatrixData(self, matrix_size, batch_size, num_rhs, seed=42):
      # [FUNCTION DOC]
      # HELPER: builds random strictly diagonally dominant tridiagonal matrices for benchmarks.
      # KNOWN ISSUE: despite being used by benchmarks that name compact or non-matrix formats, it
      # returns full dense matrices; and the comment text has a stray character after "invertibility".
      # Inputs: matrix_size, batch_size, num_rhs (integers) and seed (default 42). Seeds the global numpy
      # random generator, imports scipy sparse at call time, builds a dominant tridiagonal matrix, tiles
      # it across the batch and generates random right-hand sides. Returns a tuple of two variables
      # (matrices, right-hand sides). Raises an import error if scipy is missing.
      np.random.seed(seed)
      import scipy.sparse as sparse  # pylint:disable=g-import-not-at-top
      # By being strictly diagonally dominant, we guarantee invertibility.d
      diag = 2 * np.abs(np.random.randn(matrix_size)) + 4.1
      subdiag = 2 * np.abs(np.random.randn(matrix_size - 1))
      superdiag = 2 * np.abs(np.random.randn(matrix_size - 1))
      matrix = sparse.diags([superdiag, diag, subdiag], [1, 0, -1]).toarray()
      vector = np.random.randn(batch_size, matrix_size, num_rhs)
      return (variables.Variable(np.tile(matrix, (batch_size, 1, 1))),
              variables.Variable(vector))

    def _benchmark(self, generate_data_fn, test_name_format_string):
      # [FUNCTION DOC]
      # HELPER: shared benchmark driver for the benchmark methods.
      # KNOWN ISSUE: when XLA is enabled and pivoting is requested the function returns, abandoning all
      # remaining device/size combinations instead of skipping just that one; it also calls an evaluate
      # method that the benchmark base class does not provide.
      # Inputs: generate_data_fn (callback taking matrix size, batch size and rhs count and returning
      # diagonals and right-hand sides) and test_name_format_string (format template for the benchmark
      # name with five slots: device, matrix size, batch size, rhs count, pivoting label). Iterates over
      # CPU (plus GPU when CUDA is available) crossed with pivoting options and sizes; for each builds a
      # fresh graph and session on the device, builds data, runs the solver, initializes variables and
      # runs a timed op benchmark with at least ten iterations, reporting under the formatted name.
      # Returns nothing; side effects are benchmark result reporting.
      devices = [("/cpu:0", "cpu")]
      if test.is_gpu_available(cuda_only=True):
        devices += [("/gpu:0", "gpu")]

      for device_option, pivoting_option, size_option in \
          itertools.product(devices, self.pivoting_options, self.sizes):

        device_id, device_name = device_option
        pivoting, pivoting_name = pivoting_option
        matrix_size, batch_size, num_rhs = size_option

        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device(device_id):
          diags, rhs = generate_data_fn(matrix_size, batch_size, num_rhs)
          # Pivoting is not supported by XLA backends.
          if test.is_xla_enabled() and pivoting:
            return
          x = linalg_impl.tridiagonal_solve(
              diags, rhs, partial_pivoting=pivoting)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(x),
              min_iters=10,
              store_memory_usage=False,
              name=test_name_format_string.format(device_name, matrix_size,
                                                  batch_size, num_rhs,
                                                  pivoting_name))

    def benchmarkTridiagonalSolveOp_WithMatrixInput(self):
      # [FUNCTION DOC]
      # KNOWN ISSUE: the data is passed to the solver in its default compact format, so the "matrix
      # format" naming does not reflect the format actually used.
      # Benchmark entry point using generated matrix data and a name template that mentions the matrix
      # format. Delegates to the shared benchmark driver.
      self._benchmark(
          self._generateMatrixData,
          test_name_format_string=(
              "tridiagonal_solve_matrix_format_{}_matrix_size_{}_"
              "batch_size_{}_num_rhs_{}_{}"))

    def benchmarkTridiagonalSolveOp(self):
      # [FUNCTION DOC]
      # KNOWN ISSUE: it uses the matrix data generator rather than the compact diagonals generator, so it
      # benchmarks essentially the same inputs as the other benchmark.
      # Benchmark entry point using the matrix data generator and a general name template. Delegates to
      # the shared benchmark driver.
      self._benchmark(
          self._generateMatrixData,
          test_name_format_string=("tridiagonal_solve_{}_matrix_size_{}_"
                                   "batch_size_{}_num_rhs_{}_{}"))


if __name__ == "__main__":
  for name, fun in dict(TridiagonalSolveOpTest.__dict__).items():
    if not name.startswith("test"):
      continue
    if hasattr(fun, FLAG_NO_PARAMETERIZATION):
      continue

    # Replace testFoo with testFoo_pivoting and testFoo_noPivoting, setting
    # self.pivoting to corresponding value.
    delattr(TridiagonalSolveOpTest, name)

    def decor(test_fun, pivoting):
      # [FUNCTION DOC]
      # HELPER: main-block utility creating the parameterized test variants.
      # Takes a test function and a boolean pivoting; returns a wrapper function that takes a test
      # instance, sets its pivoting attribute to that boolean and then runs the original test with the
      # instance.

      def wrapped(instance):
        # [FUNCTION DOC]
        # HELPER: wrapper produced by the decor utility.
        # Takes a test instance, sets its pivoting attribute to the captured value and calls the wrapped
        # test function with the instance. Returns nothing.
        instance.pivoting = pivoting
        test_fun(instance)

      return wrapped

    setattr(TridiagonalSolveOpTest, name + "_pivoting",
            decor(fun, pivoting=True))
    if not hasattr(fun, FLAG_REQUIRES_PIVOTING):
      setattr(TridiagonalSolveOpTest, name + "_noPivoting",
              decor(fun, pivoting=False))

  test.main()
