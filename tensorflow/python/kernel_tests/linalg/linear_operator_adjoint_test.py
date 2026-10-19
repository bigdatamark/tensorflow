# [MODULE DOC]
# Tests for the linear operator adjoint wrapper. Defines a square-operator test class and a
# non-square test class derived from the shared linear operator test utilities (which supply
# most of the generic tests), plus extra tests for hint handling, naming, matmul, solve, matvec
# and gradient-tape safety. When run as a program it registers the generated derived-class
# tests for the square case (see the known issue on the non-square class) and starts the runner.
#
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

import numpy as np

from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_adjoint
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib

LinearOperatorAdjoint = linear_operator_adjoint.LinearOperatorAdjoint  # pylint: disable=invalid-name


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorAdjointTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Tests the adjoint operator wrapper on square matrices. Inherits the generic square-operator
  # test suite from the shared test utilities, which relies on operator_and_matrix to build the
  # operator under test and a dense reference. Disables TensorFloat-32 during each test and
  # loosens complex64 tolerances.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture teardown: restores the global TensorFloat-32 execution setting saved in setUp.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: saves whether TensorFloat-32 execution is enabled, disables it globally
    # (side effect on global configuration, undone in tearDown), and sets absolute and relative
    # tolerances for complex64 to 1e-5.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    self._atol[dtypes.complex64] = 1e-5
    self._rtol[dtypes.complex64] = 1e-5

  def operator_and_matrix(self,
                          build_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Hook required by the shared test base class. Inputs: build_info (has the shape to build),
    # dtype, use_placeholder (hide static shape behind a placeholder with default), and
    # ensure_self_adjoint_and_pd (default false). Draws a well-conditioned random matrix: positive
    # definite when the flag is set, otherwise lower triangular. Returns a pair: an adjoint
    # operator wrapping a full-matrix operator (flagged positive definite and self-adjoint) or a
    # lower-triangular operator, and the dense adjoint of the matrix as reference.
    shape = list(build_info.shape)

    if ensure_self_adjoint_and_pd:
      matrix = linear_operator_test_util.random_positive_definite_matrix(
          shape, dtype, force_well_conditioned=True)
    else:
      matrix = linear_operator_test_util.random_tril_matrix(
          shape, dtype, force_well_conditioned=True, remove_upper=True)

    lin_op_matrix = matrix

    if use_placeholder:
      lin_op_matrix = array_ops.placeholder_with_default(matrix, shape=None)

    if ensure_self_adjoint_and_pd:
      operator = LinearOperatorAdjoint(
          linalg.LinearOperatorFullMatrix(
              lin_op_matrix, is_positive_definite=True, is_self_adjoint=True))
    else:
      operator = LinearOperatorAdjoint(
          linalg.LinearOperatorLowerTriangular(lin_op_matrix))

    return operator, linalg.adjoint(matrix)

  def test_base_operator_hint_used(self):
    # [FUNCTION DOC]
    # Test scenario: builds a full-matrix operator with positive-definite and non-singular true
    # and self-adjoint false hints, takes its adjoint. Asserts the result is an adjoint wrapper
    # that inherits the same hint values.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    operator_adjoint = operator.adjoint()
    self.assertIsInstance(operator_adjoint, LinearOperatorAdjoint)
    self.assertTrue(operator_adjoint.is_positive_definite)
    self.assertTrue(operator_adjoint.is_non_singular)
    self.assertFalse(operator_adjoint.is_self_adjoint)

  def test_adjoint_of_adjoint_is_operator(self):
    # [FUNCTION DOC]
    # Test scenario: takes the adjoint of a full-matrix operator, then the adjoint of that.
    # Asserts the first is an adjoint wrapper and the second is back to a full-matrix operator.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(matrix)
    operator_adjoint = operator.adjoint()
    self.assertIsInstance(operator_adjoint, LinearOperatorAdjoint)
    adjoint_of_op_adjoint = operator_adjoint.adjoint()
    self.assertIsInstance(adjoint_of_op_adjoint,
                          linalg.LinearOperatorFullMatrix)

  def test_supplied_hint_used(self):
    # [FUNCTION DOC]
    # Test scenario: wraps a full-matrix operator in an adjoint wrapper supplying explicit hints
    # (positive definite and non-singular true, self-adjoint false). Asserts the wrapper reports
    # exactly those hints.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(matrix)
    operator_adjoint = LinearOperatorAdjoint(
        operator,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator_adjoint.is_positive_definite)
    self.assertTrue(operator_adjoint.is_non_singular)
    self.assertFalse(operator_adjoint.is_self_adjoint)

  def test_contradicting_hints_raise(self):
    # [FUNCTION DOC]
    # Test scenario: wraps an operator hinted not positive definite with a wrapper claiming it is
    # positive definite, and an operator hinted not self-adjoint with a wrapper claiming it is.
    # Asserts a value error mentioning positive-definite and self-adjoint respectively.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, is_positive_definite=False)
    with self.assertRaisesRegex(ValueError, "positive-definite"):
      LinearOperatorAdjoint(operator, is_positive_definite=True)

    operator = linalg.LinearOperatorFullMatrix(matrix, is_self_adjoint=False)
    with self.assertRaisesRegex(ValueError, "self-adjoint"):
      LinearOperatorAdjoint(operator, is_self_adjoint=True)

  def test_name(self):
    # [FUNCTION DOC]
    # Test scenario: wraps a named full-matrix operator in an adjoint wrapper without a name.
    # Asserts the wrapper's name is the base name with an adjoint suffix.
    matrix = [[11., 0.], [1., 8.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, name="my_operator", is_non_singular=True)

    operator = LinearOperatorAdjoint(operator)

    self.assertEqual("my_operator_adjoint", operator.name)

  def test_matmul_adjoint_operator(self):
    # [FUNCTION DOC]
    # Test scenario: two random real four-by-four full-matrix operators (unseeded randomness).
    # Multiplies them with the argument adjointed, the first adjointed, and both adjointed.
    # Asserts each dense result equals the numpy product with the corresponding transposes.
    matrix1 = np.random.randn(4, 4)
    matrix2 = np.random.randn(4, 4)
    full_matrix1 = linalg.LinearOperatorFullMatrix(matrix1)
    full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)

    self.assertAllClose(
        np.matmul(matrix1, matrix2.T),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint_arg=True).to_dense()))

    self.assertAllClose(
        np.matmul(matrix1.T, matrix2),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint=True).to_dense()))

    self.assertAllClose(
        np.matmul(matrix1.T, matrix2.T),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint=True,
                                adjoint_arg=True).to_dense()))

  def test_matmul_adjoint_complex_operator(self):
    # [FUNCTION DOC]
    # Test scenario: same as the real matmul test but with complex random matrices. Asserts the
    # operator products with adjoint flags equal numpy products using conjugate transposes.
    matrix1 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    matrix2 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    full_matrix1 = linalg.LinearOperatorFullMatrix(matrix1)
    full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)

    self.assertAllClose(
        np.matmul(matrix1,
                  matrix2.conj().T),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint_arg=True).to_dense()))

    self.assertAllClose(
        np.matmul(matrix1.conj().T, matrix2),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint=True).to_dense()))

    self.assertAllClose(
        np.matmul(matrix1.conj().T,
                  matrix2.conj().T),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint=True,
                                adjoint_arg=True).to_dense()))

  def test_matvec(self):
    # [FUNCTION DOC]
    # Test scenario: a fixed two-by-two matrix and vector. Asserts matrix-vector product of the
    # operator equals the numpy product, and that the operator's adjoint shorthand gives the
    # product with the transpose.
    matrix = np.array([[1., 2.], [3., 4.]])
    x = np.array([1., 2.])
    operator = linalg.LinearOperatorFullMatrix(matrix)
    self.assertAllClose(matrix.dot(x), self.evaluate(operator.matvec(x)))
    self.assertAllClose(matrix.T.dot(x), self.evaluate(operator.H.matvec(x)))

  def test_solve_adjoint_operator(self):
    # [FUNCTION DOC]
    # Test scenario: a well-conditioned random lower-triangular double matrix as a
    # lower-triangular operator and a random real full-matrix argument. Solves with the argument
    # adjointed, the operator adjointed, and both. Asserts the dense results equal triangular
    # solves with transposed inputs and the matching lower or upper flag.
    matrix1 = self.evaluate(
        linear_operator_test_util.random_tril_matrix(
            [4, 4], dtype=dtypes.float64, force_well_conditioned=True))
    matrix2 = np.random.randn(4, 4)
    full_matrix1 = linalg.LinearOperatorLowerTriangular(
        matrix1, is_non_singular=True)
    full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)

    self.assertAllClose(
        self.evaluate(linalg.triangular_solve(matrix1, matrix2.T)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint_arg=True).to_dense()))

    self.assertAllClose(
        self.evaluate(linalg.triangular_solve(matrix1.T, matrix2, lower=False)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint=True).to_dense()))

    self.assertAllClose(
        self.evaluate(
            linalg.triangular_solve(matrix1.T, matrix2.T, lower=False)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint=True,
                               adjoint_arg=True).to_dense()))

  def test_solve_adjoint_complex_operator(self):
    # [FUNCTION DOC]
    # Test scenario: same as the real solve test with complex data. The complex lower-triangular
    # matrix is built from two random triangular matrices plus an imaginary multiple.
    # Asserts the three adjoint-flag solves match triangular solves using conjugate transposes.
    matrix1 = self.evaluate(
        linear_operator_test_util.random_tril_matrix(
            [4, 4], dtype=dtypes.complex128, force_well_conditioned=True) +
        1j * linear_operator_test_util.random_tril_matrix(
            [4, 4], dtype=dtypes.complex128, force_well_conditioned=True))
    matrix2 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)

    full_matrix1 = linalg.LinearOperatorLowerTriangular(
        matrix1, is_non_singular=True)
    full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)

    self.assertAllClose(
        self.evaluate(linalg.triangular_solve(matrix1,
                                              matrix2.conj().T)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint_arg=True).to_dense()))

    self.assertAllClose(
        self.evaluate(
            linalg.triangular_solve(matrix1.conj().T, matrix2, lower=False)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint=True).to_dense()))

    self.assertAllClose(
        self.evaluate(
            linalg.triangular_solve(
                matrix1.conj().T, matrix2.conj().T, lower=False)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint=True,
                               adjoint_arg=True).to_dense()))

  def test_solvevec(self):
    # [FUNCTION DOC]
    # Test scenario: a fixed invertible two-by-two matrix and vector. Asserts solving a vector
    # with the operator equals the inverse times the vector, and with the adjoint shorthand
    # equals the transposed inverse times the vector.
    matrix = np.array([[1., 2.], [3., 4.]])
    inv_matrix = np.linalg.inv(matrix)
    x = np.array([1., 2.])
    operator = linalg.LinearOperatorFullMatrix(matrix)
    self.assertAllClose(inv_matrix.dot(x), self.evaluate(operator.solvevec(x)))
    self.assertAllClose(
        inv_matrix.T.dot(x), self.evaluate(operator.H.solvevec(x)))

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Test scenario: wraps a full-matrix operator backed by a trainable variable in an adjoint
    # wrapper and runs the shared gradient-tape safety check, which verifies operator methods
    # are differentiable with respect to the variable.
    matrix = variables_module.Variable([[1., 2.], [3., 4.]])
    operator = LinearOperatorAdjoint(linalg.LinearOperatorFullMatrix(matrix))
    self.check_tape_safe(operator)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorAdjointNonSquareTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # KNOWN ISSUE: the main block registers generated tests only for the square test class, so the
  # non-square class contributes no generated tests when run as a program.
  # Tests the adjoint operator wrapper on non-square matrices, inheriting the generic non-square
  # test suite from the shared test utilities.
  """Tests done in the base class NonSquareLinearOperatorDerivedClassTest."""

  def operator_and_matrix(self, build_info, dtype, use_placeholder):
    # [FUNCTION DOC]
    # Hook required by the shared test base class. Inputs: build_info (has the target operator
    # shape), dtype, use_placeholder (hide static shape). Builds a random normal matrix with the
    # last two dimensions swapped, wraps it in a full-matrix operator and then an adjoint
    # wrapper. Returns a pair: the adjoint operator and the dense adjoint of the matrix.
    shape_before_adjoint = list(build_info.shape)
    # We need to swap the last two dimensions because we are taking the adjoint
    # of this operator
    shape_before_adjoint[-1], shape_before_adjoint[-2] = (
        shape_before_adjoint[-2], shape_before_adjoint[-1])
    matrix = linear_operator_test_util.random_normal(
        shape_before_adjoint, dtype=dtype)

    lin_op_matrix = matrix

    if use_placeholder:
      lin_op_matrix = array_ops.placeholder_with_default(matrix, shape=None)

    operator = LinearOperatorAdjoint(
        linalg.LinearOperatorFullMatrix(lin_op_matrix))

    return operator, linalg.adjoint(matrix)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorAdjointTest)
  test.main()
