# [MODULE DOC]
# Test module for the linear operator inversion wrapper in the linalg package. Defines one test
# class, derived from the shared square linear-operator test base class, which builds an inversion
# operator around a base operator (a lower triangular operator, or a full-matrix operator when
# self-adjoint positive-definite structure is required) and compares it with the dense matrix
# inverse. Also tests flag (hint) propagation, hint contradictions, singular-operator rejection,
# naming, inverse-of-inverse simplification and gradient tape safety. When run as a script it
# registers the generated base-class tests and starts the test runner. No import-time side
# effects besides module-level aliases for the linalg package and the inversion operator class.
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

from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib

LinearOperatorInversion = linear_operator_inversion.LinearOperatorInversion  # pylint: disable=invalid-name


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorInversionTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the inversion linear operator (the operator representing the inverse of a base
  # operator). Inherits generic square-operator checks from the shared base class, supplies the
  # operator/matrix factory (reference matrix is the dense inverse), loosens complex64
  # tolerances, skips the saved-model test, and adds tests for flag hints, contradictions,
  # singular rejection, naming, inversion algebra and tape safety. Runs in graph and eager modes;
  # TensorFloat-32 is disabled per test and restored afterwards.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  # TODO: b/311343496 - Re-enable this test.
  @staticmethod
  def skip_these_tests() -> list[str]:
    # [FUNCTION DOC]
    # Hook for the base test class: returns the list of generated test names to skip, here only
    # the saved-model test, which is disabled pending a tracked bug. Takes no inputs.
    return [
        "test_saved_model",
    ]

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: restores the global TensorFloat-32 execution setting saved in
    # setUp. Returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: saves the current TensorFloat-32 setting on the instance, disables
    # TensorFloat-32 globally, and loosens the absolute and relative comparison tolerances for
    # the complex64 data type to 1e-5 (the tolerance tables are provided by the base class).
    # Returns nothing.
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
    # Factory hook used by the base class. Inputs: build_info (its shape field is the full
    # batch-plus-matrix shape), dtype, use_placeholder (feed the matrix through a placeholder of
    # unknown shape) and ensure_self_adjoint_and_pd. If that flag is set, draws a well
    # conditioned random positive-definite matrix and wraps a full-matrix operator flagged
    # positive definite and self-adjoint in an inversion operator; otherwise draws a well
    # conditioned random lower triangular matrix and wraps a lower triangular operator. Returns
    # (inversion operator, dense inverse of the drawn matrix).
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
      operator = LinearOperatorInversion(
          linalg.LinearOperatorFullMatrix(
              lin_op_matrix, is_positive_definite=True, is_self_adjoint=True))
    else:
      operator = LinearOperatorInversion(
          linalg.LinearOperatorLowerTriangular(lin_op_matrix))

    return operator, linalg.inv(matrix)

  def test_base_operator_hint_used(self):
    # [FUNCTION DOC]
    # Test: flags set on the base full-matrix operator (positive definite, non-singular, not
    # self-adjoint) are inherited by the inversion operator when it gets no explicit flags.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    operator_inv = LinearOperatorInversion(operator)
    self.assertTrue(operator_inv.is_positive_definite)
    self.assertTrue(operator_inv.is_non_singular)
    self.assertFalse(operator_inv.is_self_adjoint)

  def test_supplied_hint_used(self):
    # [FUNCTION DOC]
    # Test: flags passed explicitly to the inversion operator (positive definite, non-singular,
    # not self-adjoint) are honored when the base operator carries no flags.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(matrix)
    operator_inv = LinearOperatorInversion(
        operator,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator_inv.is_positive_definite)
    self.assertTrue(operator_inv.is_non_singular)
    self.assertFalse(operator_inv.is_self_adjoint)

  def test_solve_on_inverse(self):
    # [FUNCTION DOC]
    # Test: taking the inverse of a full-matrix operator and then solving it against the original
    # operator returns a composition operator (lazy product) rather than a dense result.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(matrix)
    operator_inv = operator.inverse()
    solved_result = operator_inv.solve(operator)
    self.assertIsInstance(
        solved_result, linalg.LinearOperatorComposition)

  def test_inverse_of_inverse(self):
    # [FUNCTION DOC]
    # Test: the inverse of a full-matrix operator is an inversion operator, and inverting that
    # again yields the original kind of operator (a full-matrix operator), i.e. double inversion
    # is simplified away.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(matrix)
    operator_inv = operator.inverse()
    self.assertIsInstance(operator_inv, LinearOperatorInversion)
    inverse_of_op_inverse = operator_inv.inverse()
    self.assertIsInstance(
        inverse_of_op_inverse, linalg.LinearOperatorFullMatrix)

  def test_contradicting_hints_raise(self):
    # [FUNCTION DOC]
    # Test: asking for a flag that contradicts the base operator's flag makes the inversion
    # constructor raise a value error. Covered cases: base not positive definite but inversion
    # claims positive definite (error mentions positive-definite); base not self-adjoint but
    # inversion claims self-adjoint (error mentions self-adjoint).
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, is_positive_definite=False)
    with self.assertRaisesRegex(ValueError, "positive-definite"):
      LinearOperatorInversion(operator, is_positive_definite=True)

    operator = linalg.LinearOperatorFullMatrix(matrix, is_self_adjoint=False)
    with self.assertRaisesRegex(ValueError, "self-adjoint"):
      LinearOperatorInversion(operator, is_self_adjoint=True)

  def test_singular_raises(self):
    # [FUNCTION DOC]
    # Test: inverting an operator that is declared singular raises a value error mentioning the
    # non-singular flag, both when the base operator is flagged not non-singular and when the
    # inversion is created with an explicit not-non-singular flag.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 1.], [1., 1.]]

    operator = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=False)
    with self.assertRaisesRegex(ValueError, "is_non_singular"):
      LinearOperatorInversion(operator)

    operator = linalg.LinearOperatorFullMatrix(matrix)
    with self.assertRaisesRegex(ValueError, "is_non_singular"):
      LinearOperatorInversion(operator, is_non_singular=False)

  def test_name(self):
    # [FUNCTION DOC]
    # Test: the default name of an inversion operator is the base operator's name with an
    # inverse suffix added; for a base named "my_operator" it must be "my_operator_inv".
    matrix = [[11., 0.], [1., 8.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, name="my_operator", is_non_singular=True)

    operator = LinearOperatorInversion(operator)

    self.assertEqual("my_operator_inv", operator.name)

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Test: wraps a full-matrix operator built from a trainable variable in an inversion operator
    # and runs the shared gradient-tape safety check on it.
    matrix = variables_module.Variable([[1., 2.], [3., 4.]])
    operator = LinearOperatorInversion(linalg.LinearOperatorFullMatrix(matrix))
    self.check_tape_safe(operator)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorInversionTest)
  test.main()
