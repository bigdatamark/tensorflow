# [MODULE DOC]
# Tests for the linear operator composition (the product of several operators). Contains a
# derived-class test case for square compositions and one for non-square compositions, both
# plugging into the shared linear operator test utility (which checks matmul, solve, etc. against
# a dense reference), plus direct tests for flag auto-setting, naming, validation errors and
# static and dynamic shapes. Import-time side effect: creates a seeded module-level random number
# generator. When run as a script it registers the generated tests and starts the test runner.
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

import numpy as np

from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(0)


class SquareLinearOperatorCompositionTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for compositions of square operators (one or two random positive-definite full
  # matrices), derived from the shared square operator test base class. Disables TF32 execution
  # and loosens float32/complex64 tolerances, skips the cholesky test, supplies the operator
  # factory, and adds tests for flag handling, naming and constructor errors.
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
    # Test fixture setup. Saves the current TF32 execution flag, disables TF32, and raises the
    # float32 tolerances to 2e-4 and the complex64 tolerances to 1e-4 (both absolute and
    # relative) from the default 1e-6. Returns nothing; mutates the global TF32 flag and the
    # per-dtype tolerance mappings.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Increase from 1e-6 to 1e-4 and 2e-4.
    self._atol[dtypes.float32] = 2e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 2e-4
    self._rtol[dtypes.complex64] = 1e-4

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Returns the generated test names to skip: only
    # cholesky, because it is not implemented for composition operators.
    # Cholesky not implemented.
    return ["cholesky"]

  def operator_and_matrix(self, build_info, dtype, use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory for the base test utility. Inputs: build_info (record with the operator shape),
    # dtype, use_placeholder (wrap matrices in placeholders with unknown shape) and
    # ensure_self_adjoint_and_pd. Draws randomly one or two factors. If self-adjoint and positive
    # definite is requested, the same random positive-definite matrix is repeated (so the product
    # of powers stays symmetric positive definite); otherwise independent random well-conditioned
    # positive-definite matrices are used. Builds the composition operator from full-matrix
    # operators (flagging positive definite and self-adjoint when requested, and square), and
    # computes the reference dense product in the order the composition applies them. Returns the
    # pair (operator, dense matrix). Uses the module-level random generator.
    shape = list(build_info.shape)

    # Either 1 or 2 matrices, depending.
    num_operators = rng.randint(low=1, high=3)
    if ensure_self_adjoint_and_pd:
      # The random PD matrices are also symmetric. Here we are computing
      # A @ A ... @ A. Since A is symmetric and PD, so are any powers of it.
      matrices = [
          linear_operator_test_util.random_positive_definite_matrix(
              shape, dtype, force_well_conditioned=True)] * num_operators
    else:
      matrices = [
          linear_operator_test_util.random_positive_definite_matrix(
              shape, dtype, force_well_conditioned=True)
          for _ in range(num_operators)
      ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          array_ops.placeholder_with_default(
              matrix, shape=None) for matrix in matrices]

    operator = linalg.LinearOperatorComposition(
        [linalg.LinearOperatorFullMatrix(l) for l in lin_op_matrices],
        is_positive_definite=True if ensure_self_adjoint_and_pd else None,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_square=True)

    matmul_order_list = list(reversed(matrices))
    mat = matmul_order_list[0]
    for other_mat in matmul_order_list[1:]:
      mat = math_ops.matmul(other_mat, mat)

    return operator, mat

  def test_is_x_flags(self):
    # [FUNCTION DOC]
    # Builds a composition of a single full-matrix operator with explicit positive-definite,
    # non-singular and not-self-adjoint flags and asserts they are reported back unchanged.
    # Matrix with two positive eigenvalues, 1, and 1.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorComposition(
        [linalg.LinearOperatorFullMatrix(matrix)],
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_is_non_singular_auto_set(self):
    # [FUNCTION DOC]
    # Composes two non-singular operators without saying whether the result is non-singular and
    # with positive definite explicitly false; asserts the composition reports non-singular and
    # not positive definite. Then asserts a value error ("always non-singular") when the
    # composition is forced to be singular.
    # Matrix with two positive eigenvalues, 11 and 8.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)

    operator = linalg.LinearOperatorComposition(
        [operator_1, operator_2],
        is_positive_definite=False,  # No reason it HAS to be False...
        is_non_singular=None)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)

    with self.assertRaisesRegex(ValueError, "always non-singular"):
      linalg.LinearOperatorComposition(
          [operator_1, operator_2], is_non_singular=False)

  def test_is_spd_is_auto_set(self):
    # [FUNCTION DOC]
    # Composes the sequence x, y, y-adjoint, x-adjoint of non-singular operators (a palindromic
    # product with adjoints) and asserts the composition automatically reports self-adjoint,
    # positive definite and non-singular. Also asserts value errors when composing x with its
    # adjoint while claiming it is not self-adjoint ("self-adjoint") or singular ("non-singular").
    matrix = [[11., 0.], [1., 8.]]
    x = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    y = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)

    operator = linalg.LinearOperatorComposition(
        [x, y, y.H, x.H], is_non_singular=None)

    self.assertTrue(operator.is_self_adjoint)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)

    with self.assertRaisesRegex(ValueError, "self-adjoint"):
      linalg.LinearOperatorComposition([x, x.H], is_self_adjoint=False)

    with self.assertRaisesRegex(ValueError, "non-singular"):
      linalg.LinearOperatorComposition([x, x.H], is_non_singular=False)

  def test_name(self):
    # [FUNCTION DOC]
    # Composes two operators named left and right and asserts the composition's default name is
    # the two names joined by an "o" separator ("left_o_right").
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, name="left")
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, name="right")

    operator = linalg.LinearOperatorComposition([operator_1, operator_2])

    self.assertEqual("left_o_right", operator.name)

  def test_different_dtypes_raises(self):
    # [FUNCTION DOC]
    # Composes operators of mixed dtypes (float64 and float32) and asserts a type error mentioning
    # "same dtype".
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3).astype(np.float32))
    ]
    with self.assertRaisesRegex(TypeError, "same dtype"):
      linalg.LinearOperatorComposition(operators)

  def test_empty_operators_raises(self):
    # [FUNCTION DOC]
    # Composes an empty list of operators and asserts a value error saying the list must be
    # non-empty.
    with self.assertRaisesRegex(ValueError, "non-empty"):
      linalg.LinearOperatorComposition([])


class NonSquareLinearOperatorCompositionTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for compositions of two non-square operators (product of an m by 2 and a 2 by n
  # well-conditioned matrix), derived from the shared non-square operator test base class.
  # Disables TF32 execution, loosens float32/complex64 tolerances to 1e-4, skips the condition
  # number test, supplies the operator factory, and adds graph-mode shape and squareness tests.
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
    # Test fixture setup. Saves the current TF32 execution flag, disables TF32, and raises the
    # float32 and complex64 absolute and relative tolerances to 1e-4 from the default 1e-6.
    # Returns nothing; mutates the global TF32 flag and the per-dtype tolerance mappings.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Increase from 1e-6 to 1e-4
    self._atol[dtypes.float32] = 1e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._rtol[dtypes.complex64] = 1e-4

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Static hook for the base test utility. Returns the base class's skip list plus the condition
    # number test, which is numerically unreliable with XLA and cuBLASLt matmul algorithms.
    # Testing the condition number fails when using XLA with cuBLASLt
    # A slight numerical difference between different matmul algorithms
    # leads to large precision issues
    return linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest.skip_these_tests(
    ) + ["cond"]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory for the base test utility. Inputs: build_info (record with the non-square operator
    # shape), dtype, use_placeholder (wrap matrices in placeholders with unknown shape) and
    # ensure_self_adjoint_and_pd (ignored; deleted since non-square operators cannot be
    # self-adjoint). Splits the shape into two factors through an inner dimension of two, builds
    # each as a random matrix with singular values near one (so the product stays
    # well-conditioned), composes full-matrix operators, and computes the dense product. Returns
    # the pair (operator, dense matrix).
    del ensure_self_adjoint_and_pd
    shape = list(build_info.shape)

    # Create 2 matrices/operators, A1, A2, which becomes A = A1 A2.
    # Use inner dimension of 2.
    k = 2
    batch_shape = shape[:-2]
    shape_1 = batch_shape + [shape[-2], k]
    shape_2 = batch_shape + [k, shape[-1]]

    # Ensure that the matrices are well-conditioned by generating
    # random matrices whose singular values are close to 1.
    # The reason to do this is because cond(AB) <= cond(A) * cond(B).
    # By ensuring that each factor has condition number close to 1, we ensure
    # that the condition number of the product isn't too far away from 1.
    def generate_well_conditioned(shape, dtype):
      # [FUNCTION DOC]
      # HELPER: local helper for the non-square factory. Takes a matrix (batch) shape and dtype
      # and returns a random matrix of that shape with singular values near one: draws diagonal
      # values with mean one and standard deviation 0.1, places them on a rectangular zero matrix,
      # and multiplies by random orthogonal factors obtained from QR decompositions on both sides.
      m, n = shape[-2], shape[-1]
      min_dim = min(m, n)
      # Generate singular values that are close to 1.
      d = linear_operator_test_util.random_normal(
          shape[:-2] + [min_dim],
          mean=1.,
          stddev=0.1,
          dtype=dtype)
      zeros = array_ops.zeros(shape=shape[:-2] + [m, n], dtype=dtype)
      d = linalg_lib.set_diag(zeros, d)
      u, _ = linalg_lib.qr(linear_operator_test_util.random_normal(
          shape[:-2] + [m, m], dtype=dtype))

      v, _ = linalg_lib.qr(linear_operator_test_util.random_normal(
          shape[:-2] + [n, n], dtype=dtype))
      return math_ops.matmul(u, math_ops.matmul(d, v))

    matrices = [
        generate_well_conditioned(shape_1, dtype=dtype),
        generate_well_conditioned(shape_2, dtype=dtype),
    ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          array_ops.placeholder_with_default(
              matrix, shape=None) for matrix in matrices]

    operator = linalg.LinearOperatorComposition(
        [linalg.LinearOperatorFullMatrix(l) for l in lin_op_matrices])

    matmul_order_list = list(reversed(matrices))
    mat = matmul_order_list[0]
    for other_mat in matmul_order_list[1:]:
      mat = math_ops.matmul(other_mat, mat)

    return operator, mat

  @test_util.run_deprecated_v1
  def test_static_shapes(self):
    # [FUNCTION DOC]
    # Graph-mode only (deprecated v1 decorator). Composes a batch of 3 by 4 and 4 by 5 operators
    # and asserts the static shape of the composition is the batch of 3 by 5.
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 4)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 4, 5))
    ]
    operator = linalg.LinearOperatorComposition(operators)
    self.assertAllEqual((2, 3, 5), operator.shape)

  @test_util.run_deprecated_v1
  def test_shape_tensors_when_statically_available(self):
    # [FUNCTION DOC]
    # Graph-mode only. Composes a batch of 3 by 4 and 4 by 5 operators with statically known
    # shapes and asserts the shape tensor evaluates to the batch of 3 by 5.
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 4)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 4, 5))
    ]
    operator = linalg.LinearOperatorComposition(operators)
    with self.cached_session():
      self.assertAllEqual((2, 3, 5), operator.shape_tensor())

  @test_util.run_deprecated_v1
  def test_shape_tensors_when_only_dynamically_available(self):
    # [FUNCTION DOC]
    # Graph-mode only. Builds the operators from placeholders without static shapes, composes them
    # and evaluates the shape tensor with values fed for the placeholders (shapes 1,2,3,4 and
    # 1,2,4,5); asserts the result is 1,2,3,5.
    mat_1 = rng.rand(1, 2, 3, 4)
    mat_2 = rng.rand(1, 2, 4, 5)
    mat_ph_1 = array_ops.placeholder(dtypes.float64)
    mat_ph_2 = array_ops.placeholder(dtypes.float64)
    feed_dict = {mat_ph_1: mat_1, mat_ph_2: mat_2}

    operators = [
        linalg.LinearOperatorFullMatrix(mat_ph_1),
        linalg.LinearOperatorFullMatrix(mat_ph_2)
    ]
    operator = linalg.LinearOperatorComposition(operators)
    with self.cached_session():
      self.assertAllEqual(
          (1, 2, 3, 5), operator.shape_tensor().eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def test_is_square_set_for_aat_form(self):
    # [FUNCTION DOC]
    # Graph-mode only. Composes an operator (of unknown shape, marked non-square) with its own
    # adjoint (the A times A-adjoint form) and asserts the composition automatically reports
    # itself square.
    mat_ph = array_ops.placeholder(dtypes.float64)  # No shape set at all.
    x = linalg.LinearOperatorFullMatrix(mat_ph, is_square=False)

    operator = linalg.LinearOperatorComposition([x, x.H])
    self.assertTrue(operator.is_square)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(SquareLinearOperatorCompositionTest)
  linear_operator_test_util.add_tests(NonSquareLinearOperatorCompositionTest)
  test.main()
