# [MODULE DOC]
# Tests for the Householder reflection linear operator. Contains one test case derived from the
# shared square operator test base class (which checks matmul, solve, etc. against a dense
# reference reflection matrix), plus direct tests for constructor validation, result types of
# adjoint and inverse, tape safety (skipping quantities that are hard-coded) and variable
# conversion. Import-time side effect: defines a module-level alias for the tape-safety skip
# options. When run as a script it registers the generated tests and starts the test runner.
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

from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_householder as householder
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib
CheckTapeSafeSkipOptions = linear_operator_test_util.CheckTapeSafeSkipOptions


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorHouseholderTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the Householder reflection operator, derived from the shared square operator
  # test base class. Runs in graph and eager modes, disables TF32 per test, supplies the shapes
  # to test, the cholesky skip and the operator factory built from a random unit-norm reflection
  # axis, and adds direct tests.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the base class tearDown, so parent cleanup is skipped.
    # Restores the TensorFloat-32 execution setting saved in setUp. Takes no inputs, returns
    # nothing; side effect is changing the global TF32 flag back to its original value.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the base class setUp, so parent fixture setup is skipped.
    # Test fixture setup. Saves the current TensorFloat-32 execution flag on the instance and then
    # disables TF32. Takes no inputs and returns nothing; side effect is changing the global TF32
    # flag.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Returns shape-info records for the operator shapes to
    # test: 1x1, and batched 3x3, 4x4 and 4x4 with two-level batch (no empty shape).
    shape_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shape_info((1, 1)),
        shape_info((1, 3, 3)),
        shape_info((3, 4, 4)),
        shape_info((2, 1, 4, 4))]

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Returns the generated test names to skip: only
    # cholesky, since a Householder reflection is never positive definite.
    # This linear operator is never positive definite.
    return ["cholesky"]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the ensure_self_adjoint_and_pd argument is accepted but silently ignored (not
    # deleted or used); the Householder operator is always self-adjoint but never positive
    # definite, so the flag cannot be honored.
    # Factory for the base test utility. Inputs: build_info (record with the operator shape),
    # dtype, and use_placeholder (wrap the reflection axis in a placeholder with unknown shape).
    # Draws a random reflection axis with magnitudes one to two, normalizes it to unit norm along
    # the last axis, builds the Householder operator, and builds the dense reference as the
    # identity minus twice the outer product of the axis with itself. Returns the pair
    # (operator, dense matrix).
    shape = list(build_info.shape)
    reflection_axis = linear_operator_test_util.random_sign_uniform(
        shape[:-1], minval=1., maxval=2., dtype=dtype)
    # Make sure unit norm.
    reflection_axis = reflection_axis / linalg_ops.norm(
        reflection_axis, axis=-1, keepdims=True)

    lin_op_reflection_axis = reflection_axis

    if use_placeholder:
      lin_op_reflection_axis = array_ops.placeholder_with_default(
          reflection_axis, shape=None)

    operator = householder.LinearOperatorHouseholder(lin_op_reflection_axis)

    mat = reflection_axis[..., array_ops.newaxis]
    matrix = -2 * math_ops.matmul(mat, mat, adjoint_b=True)
    matrix = array_ops.matrix_set_diag(
        matrix, 1. + array_ops.matrix_diag_part(matrix))

    return operator, matrix

  def test_scalar_reflection_axis_raises(self):
    # [FUNCTION DOC]
    # Constructs the operator from a scalar reflection axis and asserts a value error saying the
    # axis must have at least one dimension.
    with self.assertRaisesRegex(ValueError, "must have at least 1 dimension"):
      householder.LinearOperatorHouseholder(1.)

  def test_householder_adjoint_type(self):
    # [FUNCTION DOC]
    # Asserts the adjoint of a Householder operator (built from a four element axis) is again a
    # Householder operator.
    reflection_axis = [1., 3., 5., 8.]
    operator = householder.LinearOperatorHouseholder(reflection_axis)
    self.assertIsInstance(
        operator.adjoint(), householder.LinearOperatorHouseholder)

  def test_householder_inverse_type(self):
    # [FUNCTION DOC]
    # Asserts the inverse of a Householder operator (built from a four element axis) is again a
    # Householder operator.
    reflection_axis = [1., 3., 5., 8.]
    operator = householder.LinearOperatorHouseholder(reflection_axis)
    self.assertIsInstance(
        operator.inverse(), householder.LinearOperatorHouseholder)

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Builds a Householder operator over a variable reflection axis and runs the shared
    # tape-safety check, skipping determinant, log absolute determinant and trace because those
    # are hard-coded constants that do not depend on the variable.
    reflection_axis = variables_module.Variable([1., 3., 5., 8.])
    operator = householder.LinearOperatorHouseholder(reflection_axis)
    self.check_tape_safe(
        operator,
        skip_options=[
            # Determinant hard-coded as 1.
            CheckTapeSafeSkipOptions.DETERMINANT,
            CheckTapeSafeSkipOptions.LOG_ABS_DETERMINANT,
            # Trace hard-coded.
            CheckTapeSafeSkipOptions.TRACE,
        ])

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Builds a Householder operator over a variable reflection axis, initializes the variable in a
    # cached session and runs the shared check that converting the operator's variables to tensors
    # works.
    reflection_axis = variables_module.Variable([1., 3., 5., 8.])
    operator = householder.LinearOperatorHouseholder(reflection_axis)
    with self.cached_session() as sess:
      sess.run([reflection_axis.initializer])
      self.check_convert_variables_to_tensors(operator)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorHouseholderTest)
  test.main()
