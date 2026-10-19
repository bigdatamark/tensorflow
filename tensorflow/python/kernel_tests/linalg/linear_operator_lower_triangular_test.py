# [MODULE DOC]
# Test module for the lower triangular linear operator in the linalg package. Defines one test
# class derived from the shared square linear-operator test base class, which builds a lower
# triangular operator from a random well-conditioned matrix (whose upper triangle is non-zero but
# must be ignored) and compares it with the dense lower-triangular part. Adds tests for the
# singularity assertion, flag handling, input validation, closure of matmul with diagonal
# operators, tape safety, variable conversion and the Cholesky shortcut for products of a lower
# triangular operator with its adjoint. When run as a script it first disables TensorFloat-32
# globally, registers the generated base-class tests and starts the test runner.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorLowerTriangularTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the lower triangular linear operator. Inherits generic square-operator checks
  # from the shared base class, skips the Cholesky test (meaningless for triangular matrices),
  # supplies the operator/matrix factory and adds tests for assertions, flags, validation,
  # matmul with diagonal operators, tape safety, variable conversion and Cholesky of products of
  # a triangular operator with its adjoint. Runs in graph and eager modes.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the names of generated tests to skip, here only the
    # Cholesky test, because a Cholesky factorization does not make sense for triangular
    # matrices. Takes no inputs.
    # Cholesky does not make sense for triangular matrices.
    return ["cholesky"]

  def operator_and_matrix(self, build_info, dtype, use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory hook used by the base class. Inputs: build_info (its shape is the full
    # batch-plus-matrix shape), dtype, use_placeholder (feed the matrix through a placeholder of
    # unknown shape) and ensure_self_adjoint_and_pd. Draws a well-conditioned random matrix
    # whose upper triangle is left non-zero (the operator must ignore it). When the flag is set,
    # replaces the matrix by a diagonal matrix of the absolute diagonal values plus 0.1 (so it
    # is self-adjoint and positive definite) and tells the operator so. Builds the lower
    # triangular operator and the expected dense matrix, obtained by zeroing everything above the
    # diagonal. Returns (operator, dense matrix).
    shape = list(build_info.shape)
    # Upper triangle will be nonzero, but ignored.
    # Use a diagonal that ensures this matrix is well conditioned.
    tril = linear_operator_test_util.random_tril_matrix(
        shape, dtype=dtype, force_well_conditioned=True, remove_upper=False)
    if ensure_self_adjoint_and_pd:
      # Get the diagonal and make the matrix out of it.
      tril = array_ops.matrix_diag_part(tril)
      tril = math_ops.abs(tril) + 1e-1
      tril = array_ops.matrix_diag(tril)

    lin_op_tril = tril

    if use_placeholder:
      lin_op_tril = array_ops.placeholder_with_default(lin_op_tril, shape=None)

    operator = linalg.LinearOperatorLowerTriangular(
        lin_op_tril,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None)

    matrix = array_ops.matrix_band_part(tril, -1, 0)

    return operator, matrix

  def test_assert_non_singular(self):
    # [FUNCTION DOC]
    # Test: a lower triangular operator with a zero on its diagonal is singular, so running its
    # non-singularity assertion op raises an op error mentioning a singular operator.
    # KNOWN ISSUE: the assertion op is executed through its run method, which only works for
    # graph-mode operations; in eager mode (the class runs tests in both modes) the assertion
    # would have to be evaluated differently, so this test body is graph-oriented.
    # Singular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.cached_session():
      tril = [[1., 0.], [1., 0.]]
      operator = linalg.LinearOperatorLowerTriangular(tril)
      with self.assertRaisesOpError("Singular operator"):
        operator.assert_non_singular().run()

  def test_is_x_flags(self):
    # [FUNCTION DOC]
    # Test: explicit flags passed to the operator (positive definite, non-singular, not
    # self-adjoint) on a 2 by 2 lower triangular matrix are reported unchanged by the operator's
    # properties.
    # Matrix with two positive eigenvalues.
    tril = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorLowerTriangular(
        tril,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_tril_must_have_at_least_two_dims_or_raises(self):
    # [FUNCTION DOC]
    # Test: constructing the operator from a one-dimensional list raises a value error saying the
    # input must have at least 2 dimensions.
    with self.assertRaisesRegex(ValueError, "at least 2 dimensions"):
      linalg.LinearOperatorLowerTriangular([1.])

  def test_triangular_diag_matmul(self):
    # [FUNCTION DOC]
    # Test: multiplying a 3 by 3 lower triangular operator with a diagonal operator, in either
    # order, yields a lower triangular operator (closure under matmul), and its dense form equals
    # the product of the dense forms of the two operands in the same order.
    operator1 = linalg_lib.LinearOperatorLowerTriangular(
        [[1., 0., 0.], [2., 1., 0.], [2., 3., 3.]])
    operator2 = linalg_lib.LinearOperatorDiag([2., 2., 3.])
    operator_matmul = operator1.matmul(operator2)
    self.assertTrue(isinstance(
        operator_matmul,
        linalg_lib.LinearOperatorLowerTriangular))
    self.assertAllClose(
        math_ops.matmul(
            operator1.to_dense(),
            operator2.to_dense()),
        self.evaluate(operator_matmul.to_dense()))

    operator_matmul = operator2.matmul(operator1)
    self.assertTrue(isinstance(
        operator_matmul,
        linalg_lib.LinearOperatorLowerTriangular))
    self.assertAllClose(
        math_ops.matmul(
            operator2.to_dense(),
            operator1.to_dense()),
        self.evaluate(operator_matmul.to_dense()))

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Test: builds a non-singular lower triangular operator from a trainable variable and runs
    # the shared gradient-tape safety check on it.
    tril = variables_module.Variable([[1., 0.], [0., 1.]])
    operator = linalg_lib.LinearOperatorLowerTriangular(
        tril, is_non_singular=True)
    self.check_tape_safe(operator)

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Test: builds a non-singular lower triangular operator from a variable, initializes the
    # variable in a session and runs the shared check that converting the operator's variables to
    # tensors yields an equivalent operator.
    tril = variables_module.Variable([[1., 0.], [0., 1.]])
    operator = linalg_lib.LinearOperatorLowerTriangular(
        tril, is_non_singular=True)
    with self.cached_session() as sess:
      sess.run([tril.initializer])
      self.check_convert_variables_to_tensors(operator)

  def test_llt_composition_with_pd_l(self):
    # [FUNCTION DOC]
    # Test: for a lower triangular operator flagged non-singular and positive definite, the
    # product of it with its adjoint has a Cholesky factor that is the very same operator object
    # (identity check), so the factorization is a shortcut and does no computation.
    l = linalg_lib.LinearOperatorLowerTriangular(
        [[1., 0.], [0.5, 0.2]], is_non_singular=True, is_positive_definite=True)
    self.assertIs(l, (l @ l.H).cholesky())

  def test_llt_composition_with_non_pd_l(self):
    # [FUNCTION DOC]
    # Test: for a real lower triangular operator with a negative diagonal entry (not positive
    # definite), the Cholesky factor of the product of it with its adjoint is a lower triangular
    # operator with a strictly positive diagonal, and multiplying that factor with its adjoint
    # reproduces the original product's dense matrix.
    # The tril matrix here is selected so that multiplying the rows by the sign
    # (the correct thing to do) is different than multiplying the columns.
    l = linalg_lib.LinearOperatorLowerTriangular(
        [[-1., 0., 0.], [0.5, 0.2, 0.], [0.1, 0.1, 1.]], is_non_singular=True)
    llt = l @ l.H
    chol = llt.cholesky()
    self.assertIsInstance(chol, linalg_lib.LinearOperatorLowerTriangular)
    self.assertGreater(self.evaluate(chol.diag_part()).min(), 0)
    self.assertAllClose(
        self.evaluate(llt.to_dense()), self.evaluate(
            (chol @ chol.H).to_dense()))

  def test_llt_composition_with_non_pd_complex_l(self):
    # [FUNCTION DOC]
    # Test: complex-valued version of the previous test, with complex diagonal entries. The
    # Cholesky factor of the product of the operator with its adjoint must be a lower
    # triangular operator whose diagonal has strictly positive real parts and (numerically) zero
    # imaginary parts, and multiplying the factor with its adjoint reproduces the original
    # product's dense matrix.
    # The tril matrix here is selected so that multiplying the rows by the sign
    # (the correct thing to do) is different than multiplying the columns.
    i = math_ops.complex(0., 1.)
    l = linalg_lib.LinearOperatorLowerTriangular(
        [[-1. + i, 0., 0.], [0.5, 0.2 - 2 * i, 0.], [0.1, 0.1, 1.]],
        is_non_singular=True)
    llt = l @ l.H
    chol = llt.cholesky()
    self.assertIsInstance(chol, linalg_lib.LinearOperatorLowerTriangular)
    self.assertGreater(self.evaluate(math_ops.real(chol.diag_part())).min(), 0)
    self.assertAllClose(
        self.evaluate(math_ops.imag(chol.diag_part())).min(), 0)
    self.assertAllClose(
        self.evaluate(llt.to_dense()), self.evaluate(
            (chol @ chol.H).to_dense()))


if __name__ == "__main__":
  config.enable_tensor_float_32_execution(False)
  linear_operator_test_util.add_tests(LinearOperatorLowerTriangularTest)
  test.main()
