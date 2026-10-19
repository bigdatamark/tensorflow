# [MODULE DOC]
# Test module for the Kronecker-product linear operator in the linalg package. Contains a local
# helper that densifies a list of factor matrices into their Kronecker product (used as the
# reference value), a small test class verifying that helper against hand-computed products, and
# the main test class derived from the shared square linear-operator test base class, which
# builds a Kronecker operator from positive-definite factor operators and compares it with the
# dense reference. Also tests flag auto-setting, naming, dtype and operator-count validation,
# closure of adjoint, Cholesky and inverse (results remain Kronecker operators), tape safety,
# variable conversion and composite gradients. When run as a script it registers the generated
# base-class tests and starts the test runner. Creates a module-level seeded random generator.
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

from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_kronecker as kronecker
from tensorflow.python.ops.linalg import linear_operator_lower_triangular as lower_triangular
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(0)


def _kronecker_dense(factors):
  # [FUNCTION DOC]
  # HELPER: reference implementation used by the Kronecker operator tests to build the expected
  # dense matrix.
  # Computes the dense Kronecker product of a list of factor matrices (tensors, possibly with
  # leading batch dimensions that are assumed compatible). Starts from the first factor and, for
  # each further factor, multiplies an expanded view of the running product with an expanded view
  # of the factor (outer-product style via broadcasting), then reshapes so the row and column
  # counts are the products of the respective sizes, keeping batch dimensions. Input: list of at
  # least one tensor. Output: a single tensor holding the product. With a single factor it is
  # returned unchanged. No side effects; an empty list fails with an index error.
  """Convert a list of factors, into a dense Kronecker product."""
  product = factors[0]
  for factor in factors[1:]:
    product = product[..., array_ops.newaxis, :, array_ops.newaxis]
    factor_to_mul = factor[..., array_ops.newaxis, :, array_ops.newaxis, :]
    product *= factor_to_mul
    product = array_ops.reshape(
        product,
        shape=array_ops.concat(
            [array_ops.shape(product)[:-4],
             [array_ops.shape(product)[-4] * array_ops.shape(product)[-3],
              array_ops.shape(product)[-2] * array_ops.shape(product)[-1]]
            ], axis=0))

  return product


class KroneckerDenseTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the private dense Kronecker helper in this module (not for the operator
  # itself). Contains one test comparing the helper with explicitly written-out products.
  """Test of `_kronecker_dense` function."""

  def test_kronecker_dense_matrix(self):
    # [FUNCTION DOC]
    # Test: builds two 2 by 2 float32 matrices x and y and the hand-written 4 by 4 Kronecker
    # products of (x, y) and of (y, x); asserts the helper's result for each ordering is close to
    # the matching hand-written matrix, which also verifies the order of the factors matters.
    x = ops.convert_to_tensor([[2., 3.], [1., 2.]], dtype=dtypes.float32)
    y = ops.convert_to_tensor([[1., 2.], [5., -1.]], dtype=dtypes.float32)
    # From explicitly writing out the kronecker product of x and y.
    z = ops.convert_to_tensor([
        [2., 4., 3., 6.],
        [10., -2., 15., -3.],
        [1., 2., 2., 4.],
        [5., -1., 10., -2.]], dtype=dtypes.float32)
    # From explicitly writing out the kronecker product of y and x.
    w = ops.convert_to_tensor([
        [2., 3., 4., 6.],
        [1., 2., 2., 4.],
        [10., 15., -2., -3.],
        [5., 10., -1., -2.]], dtype=dtypes.float32)

    self.assertAllClose(
        self.evaluate(_kronecker_dense([x, y])), self.evaluate(z))
    self.assertAllClose(
        self.evaluate(_kronecker_dense([y, x])), self.evaluate(w))


@test_util.run_all_in_graph_and_eager_modes
class SquareLinearOperatorKroneckerTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the Kronecker-product linear operator with square factors. Inherits generic
  # square-operator checks from the shared base class, provides the shapes to test (each with
  # its list of factor shapes), the operator/matrix factory (reference is the dense Kronecker
  # product of random positive-definite factors) and relaxed float32/complex64 tolerances. Adds
  # operator-specific tests for flags, naming, validation, closure of derived operators, tape
  # safety, variable conversion and composite gradients. Runs in graph and eager modes;
  # TensorFloat-32 is disabled per test and restored afterwards.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: restores the global TensorFloat-32 execution setting saved in
    # setUp. Returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: saves the current TensorFloat-32 setting on the instance, disables it
    # globally, and relaxes the absolute and relative comparison tolerances for float32 and
    # complex64 from 1e-6 to 1e-4 (tolerance tables come from the base class). Returns nothing.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Increase from 1e-6 to 1e-4
    self._atol[dtypes.float32] = 1e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._rtol[dtypes.complex64] = 1e-4

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the list of operator shape descriptions to test. Each
    # entry carries the overall operator shape plus an extra field listing the factor shapes
    # whose Kronecker product has that shape, covering trivial 1 by 1, equal factors, mixed
    # factor sizes, and batched factors with broadcasting batch dimensions.
    shape_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shape_info((1, 1), factors=[(1, 1), (1, 1)]),
        shape_info((8, 8), factors=[(2, 2), (2, 2), (2, 2)]),
        shape_info((12, 12), factors=[(2, 2), (3, 3), (2, 2)]),
        shape_info((1, 3, 3), factors=[(1, 1), (1, 3, 3)]),
        shape_info((3, 6, 6), factors=[(3, 1, 1), (1, 2, 2), (1, 3, 3)]),
    ]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory hook used by the base class. Inputs: build_info (overall shape plus the list of
    # factor shapes), dtype, use_placeholder (feed factor matrices through placeholders of
    # unknown shape) and ensure_self_adjoint_and_pd (ignored and deleted, since the factors are
    # always self-adjoint positive definite, which makes the product so as well). Draws a well
    # conditioned random positive-definite matrix per factor, wraps each in a full-matrix
    # operator flagged square, self-adjoint and positive definite, and builds the Kronecker
    # operator. The dense reference is the Kronecker product of the matrices after broadcasting
    # their batch dimensions to a common shape; when no placeholders are used its static shape is
    # set to the requested one. Returns (operator, dense reference matrix).
    # Kronecker products constructed below will be from symmetric
    # positive-definite matrices.
    del ensure_self_adjoint_and_pd
    shape = list(build_info.shape)
    expected_factors = build_info.__dict__["factors"]
    matrices = [
        linear_operator_test_util.random_positive_definite_matrix(
            block_shape, dtype, force_well_conditioned=True)
        for block_shape in expected_factors
    ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          array_ops.placeholder_with_default(m, shape=None) for m in matrices]

    operator = kronecker.LinearOperatorKronecker(
        [linalg.LinearOperatorFullMatrix(
            l,
            is_square=True,
            is_self_adjoint=True,
            is_positive_definite=True)
         for l in lin_op_matrices])

    matrices = linear_operator_util.broadcast_matrix_batch_dims(matrices)

    kronecker_dense = _kronecker_dense(matrices)

    if not use_placeholder:
      kronecker_dense.set_shape(shape)

    return operator, kronecker_dense

  def test_is_x_flags(self):
    # [FUNCTION DOC]
    # Test: explicit flags passed to the Kronecker operator (positive definite, non-singular, not
    # self-adjoint) over two non-symmetric lower triangular factors are reported unchanged by
    # the operator's properties.
    # Matrix with two positive eigenvalues, 1, and 1.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = kronecker.LinearOperatorKronecker(
        [linalg.LinearOperatorFullMatrix(matrix),
         linalg.LinearOperatorFullMatrix(matrix)],
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_is_non_singular_auto_set(self):
    # [FUNCTION DOC]
    # Test: when all factor operators are flagged non-singular, the Kronecker operator
    # automatically becomes non-singular even if the flag is left unspecified, while an explicit
    # not-positive-definite flag is kept. Explicitly requesting non-singular false raises a value
    # error saying the product is always non-singular.
    # Matrix with two positive eigenvalues, 11 and 8.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)

    operator = kronecker.LinearOperatorKronecker(
        [operator_1, operator_2],
        is_positive_definite=False,  # No reason it HAS to be False...
        is_non_singular=None)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)

    with self.assertRaisesRegex(ValueError, "always non-singular"):
      kronecker.LinearOperatorKronecker(
          [operator_1, operator_2], is_non_singular=False)

  def test_name(self):
    # [FUNCTION DOC]
    # Test: the default name of a Kronecker operator joins the factor names with a product
    # marker; factors named "left" and "right" produce the name "left_x_right".
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, name="left")
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, name="right")

    operator = kronecker.LinearOperatorKronecker([operator_1, operator_2])

    self.assertEqual("left_x_right", operator.name)

  def test_different_dtypes_raises(self):
    # [FUNCTION DOC]
    # Test: building a Kronecker operator from one float64 factor and one float32 factor raises a
    # type error saying the operators must have the same dtype.
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3).astype(np.float32))
    ]
    with self.assertRaisesRegex(TypeError, "same dtype"):
      kronecker.LinearOperatorKronecker(operators)

  def test_empty_or_one_operators_raises(self):
    # [FUNCTION DOC]
    # Test: building a Kronecker operator from an empty list raises a value error mentioning that
    # at least one operator is required. KNOWN ISSUE: the name says "empty or one" but only the
    # empty list is exercised; the single-operator case is not tested.
    with self.assertRaisesRegex(ValueError, ">=1 operators"):
      kronecker.LinearOperatorKronecker([])

  def test_kronecker_adjoint_type(self):
    # [FUNCTION DOC]
    # Test: the adjoint of a Kronecker operator made of two non-singular identity-matrix factors
    # is again a Kronecker operator and still holds exactly two factor operators.
    matrix = [[1., 0.], [0., 1.]]
    operator = kronecker.LinearOperatorKronecker(
        [
            linalg.LinearOperatorFullMatrix(
                matrix, is_non_singular=True),
            linalg.LinearOperatorFullMatrix(
                matrix, is_non_singular=True),
        ],
        is_non_singular=True,
    )
    adjoint = operator.adjoint()
    self.assertIsInstance(
        adjoint,
        kronecker.LinearOperatorKronecker)
    self.assertEqual(2, len(adjoint.operators))

  def test_kronecker_cholesky_type(self):
    # [FUNCTION DOC]
    # Test: for a Kronecker operator of two factors flagged positive definite and self-adjoint,
    # the Cholesky factor is a Kronecker operator with two factors, each of which is a lower
    # triangular operator (the factorwise Cholesky factors).
    matrix = [[1., 0.], [0., 1.]]
    operator = kronecker.LinearOperatorKronecker(
        [
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_positive_definite=True,
                is_self_adjoint=True,
            ),
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_positive_definite=True,
                is_self_adjoint=True,
            ),
        ],
        is_positive_definite=True,
        is_self_adjoint=True,
    )
    cholesky_factor = operator.cholesky()
    self.assertIsInstance(
        cholesky_factor,
        kronecker.LinearOperatorKronecker)
    self.assertEqual(2, len(cholesky_factor.operators))
    self.assertIsInstance(
        cholesky_factor.operators[0],
        lower_triangular.LinearOperatorLowerTriangular)
    self.assertIsInstance(
        cholesky_factor.operators[1],
        lower_triangular.LinearOperatorLowerTriangular)

  def test_kronecker_inverse_type(self):
    # [FUNCTION DOC]
    # Test: the inverse of a Kronecker operator made of two non-singular factors is again a
    # Kronecker operator with two factors (the factorwise inverses).
    matrix = [[1., 0.], [0., 1.]]
    operator = kronecker.LinearOperatorKronecker(
        [
            linalg.LinearOperatorFullMatrix(
                matrix, is_non_singular=True),
            linalg.LinearOperatorFullMatrix(
                matrix, is_non_singular=True),
        ],
        is_non_singular=True,
    )
    inverse = operator.inverse()
    self.assertIsInstance(
        inverse,
        kronecker.LinearOperatorKronecker)
    self.assertEqual(2, len(inverse.operators))

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Test: builds a Kronecker operator whose two factors are full-matrix operators over
    # trainable variables and runs the shared gradient-tape safety check on it.
    matrix_1 = variables_module.Variable([[1., 0.], [0., 1.]])
    matrix_2 = variables_module.Variable([[2., 0.], [0., 2.]])
    operator = kronecker.LinearOperatorKronecker(
        [
            linalg.LinearOperatorFullMatrix(
                matrix_1, is_non_singular=True),
            linalg.LinearOperatorFullMatrix(
                matrix_2, is_non_singular=True),
        ],
        is_non_singular=True,
    )
    self.check_tape_safe(operator)

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Test: builds a Kronecker operator over two variable-backed factors, initializes all the
    # operator's variables in a session and runs the shared check that converting the operator's
    # variables to tensors yields an equivalent operator.
    matrix_1 = variables_module.Variable([[1., 0.], [0., 1.]])
    matrix_2 = variables_module.Variable([[2., 0.], [0., 2.]])
    operator = kronecker.LinearOperatorKronecker(
        [
            linalg.LinearOperatorFullMatrix(
                matrix_1, is_non_singular=True),
            linalg.LinearOperatorFullMatrix(
                matrix_2, is_non_singular=True),
        ],
        is_non_singular=True,
    )
    with self.cached_session() as sess:
      sess.run([x.initializer for x in operator.variables])
      self.check_convert_variables_to_tensors(operator)

  def test_composite_gradients(self):
    # [FUNCTION DOC]
    # Test: gradients with respect to composite (operator) objects. Inside a gradient tape,
    # watches two full-matrix operators (identity and twice identity) and the Kronecker operator
    # built from them, computes the first operator's matmul with a test input, and takes the
    # gradient of that output with respect to both factors and the Kronecker operator. Asserts
    # the first factor's gradient matches the first component of the Kronecker operator's
    # gradient, that the second component of the latter is all zeros, and that the gradient for
    # the unused second factor is nothing (disconnected).
    with backprop.GradientTape() as tape:
      op1 = linalg.LinearOperatorFullMatrix(
          [[1., 0.], [0., 1.]], is_non_singular=True)
      op2 = linalg.LinearOperatorFullMatrix(
          [[2., 0.], [0., 2.]], is_non_singular=True)
      tape.watch([op1, op2])
      operator = kronecker.LinearOperatorKronecker(
          [op1, op2], is_non_singular=True)

      x = self.make_x(op1, adjoint=False)
      y = op1.matmul(x)
      connected_grad, disconnected_grad, composite_grad = tape.gradient(
          y, [op1, op2, operator])

    disconnected_component_grad = composite_grad.operators[1].to_dense()
    self.assertAllClose(connected_grad.to_dense(),
                        composite_grad.operators[0].to_dense())
    self.assertAllClose(disconnected_component_grad,
                        array_ops.zeros_like(disconnected_component_grad))
    self.assertIsNone(disconnected_grad)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(SquareLinearOperatorKroneckerTest)
  test.main()
