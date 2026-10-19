# [MODULE DOC]
# Unit tests for the helper functions in the linear operator utility module: zero-imaginary-part
# and nonzero-modulus assertions, batch-dimension broadcasting of matrices, matrix solve with
# broadcasting, compatibility check of operator domain dimension with a right-hand side, adjoint
# pair and A-times-A-transpose form detection, combining operator and provided hint values, and
# detecting whether an argument is a blockwise (list-of-blocks) structure. Defines a module-level
# seeded numpy random generator (fixed seed 0) used to create test data, and small stub operator
# classes. Runs the tests through the TensorFlow test main when executed as a script.
#
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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test

rng = np.random.RandomState(0)


class AssertZeroImagPartTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility that asserts a tensor has a zero imaginary part. Covers real tensors,
  # complex tensors with zero imaginary part (both pass) and nonzero imaginary part (fails).

  def test_real_tensor_doesnt_raise(self):
    # [FUNCTION DOC]
    # Test: a real-valued tensor passed to the zero-imaginary-part assertion produces an op that
    # evaluates without raising. Asserts nothing beyond successful evaluation.
    x = ops.convert_to_tensor([0., 2, 3])
    # Should not raise.
    self.evaluate(
        linear_operator_util.assert_zero_imag_part(x, message="ABC123"))

  def test_complex_tensor_with_imag_zero_doesnt_raise(self):
    # [FUNCTION DOC]
    # Test: a complex tensor built from real parts and an all-zero imaginary part passes the
    # zero-imaginary-part assertion; evaluation must not raise.
    x = ops.convert_to_tensor([1., 0, 3])
    y = ops.convert_to_tensor([0., 0, 0])
    z = math_ops.complex(x, y)
    # Should not raise.
    self.evaluate(
        linear_operator_util.assert_zero_imag_part(z, message="ABC123"))

  def test_complex_tensor_with_nonzero_imag_raises(self):
    # [FUNCTION DOC]
    # Test: a complex tensor with nonzero imaginary parts makes the assertion op fail at
    # evaluation with an op error whose text contains the supplied message.
    x = ops.convert_to_tensor([1., 2, 0])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    with self.assertRaisesOpError("ABC123"):
      self.evaluate(
          linear_operator_util.assert_zero_imag_part(z, message="ABC123"))


class AssertNoEntriesWithModulusZeroTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility asserting that no entry of a (real or complex) tensor has zero modulus.
  # Nonzero tensors pass; tensors with a zero entry cause an op error carrying the message.

  def test_nonzero_real_tensor_doesnt_raise(self):
    # [FUNCTION DOC]
    # Test: a real tensor with all nonzero entries passes the assertion without raising.
    x = ops.convert_to_tensor([1., 2, 3])
    # Should not raise.
    self.evaluate(
        linear_operator_util.assert_no_entries_with_modulus_zero(
            x, message="ABC123"))

  def test_nonzero_complex_tensor_doesnt_raise(self):
    # [FUNCTION DOC]
    # Test: a complex tensor whose entries each have a nonzero real or imaginary part (so nonzero
    # modulus) passes the assertion without raising.
    x = ops.convert_to_tensor([1., 0, 3])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    # Should not raise.
    self.evaluate(
        linear_operator_util.assert_no_entries_with_modulus_zero(
            z, message="ABC123"))

  def test_zero_real_tensor_raises(self):
    # [FUNCTION DOC]
    # Test: a real tensor containing a zero entry makes the assertion fail at evaluation with an
    # op error containing the supplied message.
    x = ops.convert_to_tensor([1., 0, 3])
    with self.assertRaisesOpError("ABC123"):
      self.evaluate(
          linear_operator_util.assert_no_entries_with_modulus_zero(
              x, message="ABC123"))

  def test_zero_complex_tensor_raises(self):
    # [FUNCTION DOC]
    # Test: a complex tensor with an entry whose real and imaginary parts are both zero makes the
    # assertion fail with an op error containing the supplied message.
    x = ops.convert_to_tensor([1., 2, 0])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    with self.assertRaisesOpError("ABC123"):
      self.evaluate(
          linear_operator_util.assert_no_entries_with_modulus_zero(
              z, message="ABC123"))


class BroadcastMatrixBatchDimsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for broadcasting the batch dimensions of a list of matrices to a common batch shape,
  # with static and dynamic (placeholder) shapes, differing ranks, and the error case of inputs
  # with fewer than two dimensions.

  def test_zero_batch_matrices_returned_as_empty_list(self):
    # [FUNCTION DOC]
    # Test: broadcasting an empty list of matrices returns an empty result.
    self.assertAllEqual([],
                        linear_operator_util.broadcast_matrix_batch_dims([]))

  def test_one_batch_matrix_returned_after_tensor_conversion(self):
    # [FUNCTION DOC]
    # Test: a single numpy array passed in a list is returned as a tensor object holding the same
    # values (checks type is tensor and numeric equality).
    arr = rng.rand(2, 3, 4)
    tensor, = linear_operator_util.broadcast_matrix_batch_dims([arr])
    self.assertTrue(isinstance(tensor, tensor_lib.Tensor))

    self.assertAllClose(arr, self.evaluate(tensor))

  def test_static_dims_broadcast(self):
    # [FUNCTION DOC]
    # Test: two numpy arrays with different batch shapes (3,1,2 and 4,1) and different matrix
    # shapes are broadcast to the common batch shape (3,4,2); asserts the static result shapes and
    # evaluated values equal the inputs added to a zero array of the broadcast batch shape.
    # x.batch_shape = [3, 1, 2]
    # y.batch_shape = [4, 1]
    # broadcast batch shape = [3, 4, 2]
    x = rng.rand(3, 1, 2, 1, 5)
    y = rng.rand(4, 1, 3, 7)
    batch_of_zeros = np.zeros((3, 4, 2, 1, 1))
    x_bc_expected = x + batch_of_zeros
    y_bc_expected = y + batch_of_zeros

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x, y])

    self.assertAllEqual(x_bc_expected.shape, x_bc.shape)
    self.assertAllEqual(y_bc_expected.shape, y_bc.shape)
    x_bc_, y_bc_ = self.evaluate([x_bc, y_bc])
    self.assertAllClose(x_bc_expected, x_bc_)
    self.assertAllClose(y_bc_expected, y_bc_)

  def test_static_dims_broadcast_second_arg_higher_rank(self):
    # [FUNCTION DOC]
    # Test: like the static broadcast test but the second input has a higher batch rank (batch
    # shapes 1,2 and 1,3,1 giving 1,3,2); asserts shapes and values match the zero-added expected
    # arrays.
    # x.batch_shape =    [1, 2]
    # y.batch_shape = [1, 3, 1]
    # broadcast batch shape = [1, 3, 2]
    x = rng.rand(1, 2, 1, 5)
    y = rng.rand(1, 3, 2, 3, 7)
    batch_of_zeros = np.zeros((1, 3, 2, 1, 1))
    x_bc_expected = x + batch_of_zeros
    y_bc_expected = y + batch_of_zeros

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x, y])

    self.assertAllEqual(x_bc_expected.shape, x_bc.shape)
    self.assertAllEqual(y_bc_expected.shape, y_bc.shape)
    x_bc_, y_bc_ = self.evaluate([x_bc, y_bc])
    self.assertAllClose(x_bc_expected, x_bc_)
    self.assertAllClose(y_bc_expected, y_bc_)

  def test_dynamic_dims_broadcast_32bit(self):
    # [FUNCTION DOC]
    # Test: float32 inputs wrapped as placeholders with fully unknown static shape are broadcast
    # (batch shapes 3,1,2 and 4,1 to 3,4,2); asserts evaluated values match the expected
    # broadcast arrays. Exercises the dynamic path with 32-bit shape arithmetic.
    # x.batch_shape = [3, 1, 2]
    # y.batch_shape = [4, 1]
    # broadcast batch shape = [3, 4, 2]
    x = rng.rand(3, 1, 2, 1, 5).astype(np.float32)
    y = rng.rand(4, 1, 3, 7).astype(np.float32)
    batch_of_zeros = np.zeros((3, 4, 2, 1, 1)).astype(np.float32)
    x_bc_expected = x + batch_of_zeros
    y_bc_expected = y + batch_of_zeros

    x_ph = array_ops.placeholder_with_default(x, shape=None)
    y_ph = array_ops.placeholder_with_default(y, shape=None)

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x_ph, y_ph])

    x_bc_, y_bc_ = self.evaluate([x_bc, y_bc])
    self.assertAllClose(x_bc_expected, x_bc_)
    self.assertAllClose(y_bc_expected, y_bc_)

  def test_dynamic_dims_broadcast_32bit_second_arg_higher_rank(self):
    # [FUNCTION DOC]
    # Test: dynamic-shape float32 placeholders where the second input has higher batch rank (batch
    # shapes 1,2 and 3,4,1 giving 3,4,2); asserts evaluated values match the expected broadcast.
    # x.batch_shape =    [1, 2]
    # y.batch_shape = [3, 4, 1]
    # broadcast batch shape = [3, 4, 2]
    x = rng.rand(1, 2, 1, 5).astype(np.float32)
    y = rng.rand(3, 4, 1, 3, 7).astype(np.float32)
    batch_of_zeros = np.zeros((3, 4, 2, 1, 1)).astype(np.float32)
    x_bc_expected = x + batch_of_zeros
    y_bc_expected = y + batch_of_zeros

    x_ph = array_ops.placeholder_with_default(x, shape=None)
    y_ph = array_ops.placeholder_with_default(y, shape=None)

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x_ph, y_ph])

    x_bc_, y_bc_ = self.evaluate([x_bc, y_bc])
    self.assertAllClose(x_bc_expected, x_bc_)
    self.assertAllClose(y_bc_expected, y_bc_)

  def test_less_than_two_dims_raises_static(self):
    # [FUNCTION DOC]
    # Test: if either input has statically fewer than two dimensions (a vector paired with a
    # matrix, in either order) broadcasting raises a value error mentioning "at least two
    # dimensions".
    x = rng.rand(3)
    y = rng.rand(1, 1)

    with self.assertRaisesRegex(ValueError, "at least two dimensions"):
      linear_operator_util.broadcast_matrix_batch_dims([x, y])

    with self.assertRaisesRegex(ValueError, "at least two dimensions"):
      linear_operator_util.broadcast_matrix_batch_dims([y, x])


class MatrixSolveWithBroadcastTest(test.TestCase):
  # [CLASS DOC]
  # Tests for solving linear systems where the matrix and right-hand side may have different
  # batch dimensions; results are compared against the plain matrix-solve op applied to
  # explicitly broadcast inputs.

  def test_static_dims_broadcast_matrix_has_extra_dims(self):
    # [FUNCTION DOC]
    # Test: matrix has a batch dimension (2) that the right-hand side lacks; asserts the result
    # shape is 2,3,7 and values equal the plain solve with the right-hand side explicitly tiled.
    # batch_shape = [2]
    matrix = rng.rand(2, 3, 3)
    rhs = rng.rand(3, 7)
    rhs_broadcast = rhs + np.zeros((2, 1, 1))

    result = linear_operator_util.matrix_solve_with_broadcast(matrix, rhs)
    self.assertAllEqual((2, 3, 7), result.shape)
    expected = linalg_ops.matrix_solve(matrix, rhs_broadcast)
    self.assertAllClose(*self.evaluate([expected, result]))

  def test_static_dims_broadcast_rhs_has_extra_dims(self):
    # [FUNCTION DOC]
    # Test: right-hand side has a batch dimension (2) the single 3x3 matrix lacks; asserts result
    # shape 2,3,2 and values equal the plain solve with the matrix explicitly broadcast. The
    # comment says the optimization of flipping batch dims was verified manually with a debugger.
    # Since the second arg has extra dims, and the domain dim of the first arg
    # is larger than the number of linear equations, code will "flip" the extra
    # dims of the first arg to the far right, making extra linear equations
    # (then call the matrix function, then flip back).
    # We have verified that this optimization indeed happens.  How? We stepped
    # through with a debugger.
    # batch_shape = [2]
    matrix = rng.rand(3, 3)
    rhs = rng.rand(2, 3, 2)
    matrix_broadcast = matrix + np.zeros((2, 1, 1))

    result = linear_operator_util.matrix_solve_with_broadcast(matrix, rhs)
    self.assertAllEqual((2, 3, 2), result.shape)
    expected = linalg_ops.matrix_solve(matrix_broadcast, rhs)
    self.assertAllClose(*self.evaluate([expected, result]))

  def test_static_dims_broadcast_rhs_has_extra_dims_dynamic(self):
    # [FUNCTION DOC]
    # Test: same scenario as the static rhs-extra-dims test but both inputs are placeholders with
    # known rank and unknown dimension sizes; asserts the result rank is 3 and values equal the
    # plain solve with the matrix explicitly broadcast.
    # Since the second arg has extra dims, and the domain dim of the first arg
    # is larger than the number of linear equations, code will "flip" the extra
    # dims of the first arg to the far right, making extra linear equations
    # (then call the matrix function, then flip back).
    # We have verified that this optimization indeed happens.  How? We stepped
    # through with a debugger.
    # batch_shape = [2]
    matrix = rng.rand(3, 3)
    rhs = rng.rand(2, 3, 2)
    matrix_broadcast = matrix + np.zeros((2, 1, 1))

    matrix_ph = array_ops.placeholder_with_default(matrix, shape=[None, None])
    rhs_ph = array_ops.placeholder_with_default(rhs, shape=[None, None, None])

    result = linear_operator_util.matrix_solve_with_broadcast(matrix_ph, rhs_ph)
    self.assertAllEqual(3, result.shape.ndims)
    expected = linalg_ops.matrix_solve(matrix_broadcast, rhs)
    self.assertAllClose(*self.evaluate([expected, result]))

  def test_static_dims_broadcast_rhs_has_extra_dims_and_adjoint(self):
    # [FUNCTION DOC]
    # Test: right-hand side has extra batch dims and the adjoint option is true; asserts the result
    # shape is 2,3,2 and values equal the plain adjoint solve with the matrix broadcast.
    # Since the second arg has extra dims, and the domain dim of the first arg
    # is larger than the number of linear equations, code will "flip" the extra
    # dims of the first arg to the far right, making extra linear equations
    # (then call the matrix function, then flip back).
    # We have verified that this optimization indeed happens.  How? We stepped
    # through with a debugger.
    # batch_shape = [2]
    matrix = rng.rand(3, 3)
    rhs = rng.rand(2, 3, 2)
    matrix_broadcast = matrix + np.zeros((2, 1, 1))

    result = linear_operator_util.matrix_solve_with_broadcast(
        matrix, rhs, adjoint=True)
    self.assertAllEqual((2, 3, 2), result.shape)
    expected = linalg_ops.matrix_solve(matrix_broadcast, rhs, adjoint=True)
    self.assertAllClose(*self.evaluate([expected, result]))

  def test_dynamic_dims_broadcast_64bit(self):
    # [FUNCTION DOC]
    # Test: float64 matrix (batch 2) and right-hand side (batch 2,1) given as placeholders with
    # fully unknown shape; asserts the broadcast solve result equals the plain solve applied to
    # explicitly broadcast inputs (batch 2,2).
    # batch_shape = [2, 2]
    matrix = rng.rand(2, 3, 3)
    rhs = rng.rand(2, 1, 3, 7)
    matrix_broadcast = matrix + np.zeros((2, 2, 1, 1))
    rhs_broadcast = rhs + np.zeros((2, 2, 1, 1))

    matrix_ph = array_ops.placeholder_with_default(matrix, shape=None)
    rhs_ph = array_ops.placeholder_with_default(rhs, shape=None)

    result, expected = self.evaluate([
        linear_operator_util.matrix_solve_with_broadcast(matrix_ph, rhs_ph),
        linalg_ops.matrix_solve(matrix_broadcast, rhs_broadcast)
    ])
    self.assertAllClose(expected, result)


class DomainDimensionStubOperator(object):
  # [CLASS DOC]
  # HELPER: minimal stand-in operator for the compatible-dimensions tests.
  # Exposes only a method returning its domain dimension as a tensor; holds that value in a
  # private attribute set at construction.

  def __init__(self, domain_dimension):
    # [FUNCTION DOC]
    # HELPER: constructor of the stub operator. Takes a domain dimension (an integer or array-like)
    # and stores it converted to a tensor. Returns nothing.
    self._domain_dimension = ops.convert_to_tensor(domain_dimension)

  def domain_dimension_tensor(self):
    # [FUNCTION DOC]
    # HELPER: returns the stored domain dimension tensor; no side effects.
    return self._domain_dimension


class AssertCompatibleMatrixDimensionsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the assertion that an operator's domain dimension matches the second-to-last
  # dimension of a tensor, using a stub operator that reports a fixed domain dimension.

  def test_compatible_dimensions_do_not_raise(self):
    # [FUNCTION DOC]
    # Test: a tensor of shape 2,3,4 and a stub operator with domain dimension 3 are compatible;
    # the assertion op evaluates without raising.
    x = ops.convert_to_tensor(rng.rand(2, 3, 4))
    operator = DomainDimensionStubOperator(3)
    # Should not raise
    self.evaluate(
        linear_operator_util.assert_compatible_matrix_dimensions(operator, x))

  def test_incompatible_dimensions_raise(self):
    # [FUNCTION DOC]
    # Test: a tensor of shape 2,4,4 with a stub operator of domain dimension 3 is incompatible;
    # evaluation raises an op error containing "Dimensions are not compatible".
    x = ops.convert_to_tensor(rng.rand(2, 4, 4))
    operator = DomainDimensionStubOperator(3)
    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesOpError("Dimensions are not compatible"):
      self.evaluate(
          linear_operator_util.assert_compatible_matrix_dimensions(operator, x))
    # pylint: enable=g-error-prone-assert-raises


class IsAdjointPairTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the predicate that decides whether two operators are known adjoints of each other,
  # using full-matrix operators with various self-adjoint hints.

  def test_one_is_explicitly_adjoint_of_other_returns_true(self):
    # [FUNCTION DOC]
    # Test: an operator and its adjoint view form an adjoint pair, in both argument orders.
    x = linalg_lib.LinearOperatorFullMatrix(
        [[1., 2.], [3., 4.]], is_self_adjoint=False)
    self.assertTrue(linear_operator_util.is_adjoint_pair(x, x.H))
    self.assertTrue(linear_operator_util.is_adjoint_pair(x.H, x))

  def test_repeated_non_self_adjoint_operator_returns_false(self):
    # [FUNCTION DOC]
    # Test: pairing an operator flagged not self-adjoint with itself is not an adjoint pair.
    x = linalg_lib.LinearOperatorFullMatrix(
        [[1., 2.], [3., 4.]], is_self_adjoint=False)
    self.assertFalse(linear_operator_util.is_adjoint_pair(x, x))

  def test_repeated_self_adjoint_operator_returns_true(self):
    # [FUNCTION DOC]
    # Test: pairing an operator flagged self-adjoint with itself is an adjoint pair.
    x = linalg_lib.LinearOperatorFullMatrix(
        [[1., 2.], [2., 1.]], is_self_adjoint=True)
    self.assertTrue(linear_operator_util.is_adjoint_pair(x, x))

  def test_pair_of_non_self_adjoint_operator_returns_false(self):
    # [FUNCTION DOC]
    # Test: two different non-self-adjoint operators are not an adjoint pair.
    x = linalg_lib.LinearOperatorFullMatrix(
        [[1., 2.], [3., 4.]], is_self_adjoint=False)
    y = linalg_lib.LinearOperatorFullMatrix(
        [[10., 20.], [3., 4.]], is_self_adjoint=False)
    self.assertFalse(linear_operator_util.is_adjoint_pair(x, y))


class IsAATFormTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the predicate that checks whether a list of operators has the form A, A-adjoint
  # (or longer palindromic adjoint sequences). Avoids diagonal operators in the tests because
  # their adjoint equals themselves for real data.
  # Careful when writing tests to avoid LinearOperatorDiag, since D.H is D for
  # real D and this will be confusing.

  def test_empty_operators_raises(self):
    # [FUNCTION DOC]
    # Test: passing an empty list of operators raises a value error mentioning "empty operators".
    with self.assertRaisesRegex(ValueError, "empty operators"):
      linear_operator_util.is_aat_form(operators=[])

  def test_odd_length_returns_false(self):
    # [FUNCTION DOC]
    # Test: lists with an odd number of operators (one and three) are never in the required form.
    x = linalg_lib.LinearOperatorFullMatrix([[1., 2.], [2., 1]],
                                            is_self_adjoint=True)
    self.assertFalse(linear_operator_util.is_aat_form([x]))
    self.assertFalse(linear_operator_util.is_aat_form([x, x, x.H]))

  def test_length_2_aat_form_with_sa_x(self):
    # [FUNCTION DOC]
    # Test: a self-adjoint operator followed by its adjoint is recognized as the form.
    x = linalg_lib.LinearOperatorFullMatrix([[1., 2.], [2., 1]],
                                            is_self_adjoint=True)
    self.assertTrue(linear_operator_util.is_aat_form([x, x.H]))

  def test_length_2_aat_form_with_non_sa_x(self):
    # [FUNCTION DOC]
    # Test: a non-self-adjoint operator followed by its adjoint is recognized as the form.
    x = linalg_lib.LinearOperatorFullMatrix([[1., 5.], [2., 1]],
                                            is_self_adjoint=False)
    self.assertTrue(linear_operator_util.is_aat_form([x, x.H]))

  def test_length_4_aat_form(self):
    # [FUNCTION DOC]
    # Test: the sequence x, y, adjoint of y, adjoint of x is recognized as the form.
    x = linalg_lib.LinearOperatorFullMatrix([[1., 2.], [5., 1]],
                                            is_self_adjoint=False)
    y = linalg_lib.LinearOperatorFullMatrix([[10., 2.], [3., 10]],
                                            is_self_adjoint=False)
    self.assertTrue(linear_operator_util.is_aat_form([x, y, y.H, x.H]))


class DummyOperatorWithHint(object):
  # [CLASS DOC]
  # HELPER: stub operator whose attributes are set from keyword arguments, used to supply a hint
  # attribute for the hint-combining tests.

  def __init__(self, **kwargs):
    # [FUNCTION DOC]
    # HELPER: constructor; copies every keyword argument into the instance's attributes.
    # Returns nothing.
    self.__dict__.update(kwargs)


class UseOperatorOrProvidedHintUnlessContradictingTest(test.TestCase,
                                                       parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for the helper that merges an operator's hint attribute with a provided
  # hint value, raising when the two contradict.

  @parameterized.named_parameters(
      ("none_none", None, None, None),
      ("none_true", None, True, True),
      ("true_none", True, None, True),
      ("true_true", True, True, True),
      ("none_false", None, False, False),
      ("false_none", False, None, False),
      ("false_false", False, False, False),
  )
  def test_computes_an_or_if_non_contradicting(self, operator_hint_value,
                                               provided_hint_value,
                                               expected_result):
    # [FUNCTION DOC]
    # Parameterized test: for each combination of operator hint and provided hint (none, true,
    # false) that does not contradict, the merged hint equals the non-null one, or null when both
    # are null. Inputs: operator hint value, provided hint value, expected merged result.
    self.assertEqual(
        expected_result,
        linear_operator_util.use_operator_or_provided_hint_unless_contradicting(
            operator=DummyOperatorWithHint(my_hint=operator_hint_value),
            hint_attr_name="my_hint",
            provided_hint_value=provided_hint_value,
            message="should not be needed here"))

  @parameterized.named_parameters(
      ("true_false", True, False),
      ("false_true", False, True),
  )
  def test_raises_if_contradicting(self, operator_hint_value,
                                   provided_hint_value):
    # [FUNCTION DOC]
    # Parameterized test: when the operator hint and provided hint are opposite booleans, a value
    # error containing the supplied message is raised.
    with self.assertRaisesRegex(ValueError, "my error message"):
      linear_operator_util.use_operator_or_provided_hint_unless_contradicting(
          operator=DummyOperatorWithHint(my_hint=operator_hint_value),
          hint_attr_name="my_hint",
          provided_hint_value=provided_hint_value,
          message="my error message")


class BlockwiseTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the helper that decides whether an argument should be interpreted as a list of
  # blocks (one per operator) or as a single non-blocked structure, given operator dimensions
  # (known or unknown) and the split dimension.

  @parameterized.named_parameters(
      ("split_dim_1", [3, 3, 4], -1),
      ("split_dim_2", [2, 5], -2),
      )
  def test_blockwise_input(self, op_dimension_values, split_dim):
    # [FUNCTION DOC]
    # Parameterized test over two cases (split along last or second-to-last dimension). Inputs:
    # list of per-block operator dimension sizes and the split dimension. Builds blocks as arrays,
    # nested lists, tensors and placeholders and asserts the helper reports them as blockwise for
    # both known and unknown operator dimensions.

    op_dimensions = [
        tensor_shape.Dimension(v) for v in op_dimension_values]
    unknown_op_dimensions = [
        tensor_shape.Dimension(None) for _ in op_dimension_values]

    batch_shape = [2, 1]
    arg_dim = 5
    if split_dim == -1:
      blockwise_arrays = [np.zeros(batch_shape + [arg_dim, d])
                          for d in op_dimension_values]
    else:
      blockwise_arrays = [np.zeros(batch_shape + [d, arg_dim])
                          for d in op_dimension_values]

    blockwise_list = [block.tolist() for block in blockwise_arrays]
    blockwise_tensors = [ops.convert_to_tensor(block)
                         for block in blockwise_arrays]
    blockwise_placeholders = [
        array_ops.placeholder_with_default(block, shape=None)
        for block in blockwise_arrays]

    # Iterables of non-nested structures are always interpreted as blockwise.
    # The list of lists is interpreted as blockwise as well, regardless of
    # whether the operator dimensions are known, since the sizes of its elements
    # along `split_dim` are non-identical.
    for op_dims in [op_dimensions, unknown_op_dimensions]:
      for blockwise_inputs in [
          blockwise_arrays, blockwise_list,
          blockwise_tensors, blockwise_placeholders]:
        self.assertTrue(linear_operator_util.arg_is_blockwise(
            op_dims, blockwise_inputs, split_dim))

  def test_non_blockwise_input(self):
    # [FUNCTION DOC]
    # Test: a single 4-D array given as array, tensor, placeholder and nested list is reported as
    # not blockwise for both known operator dimensions and unknown operator dimensions.
    x = np.zeros((2, 3, 4, 6))
    x_tensor = ops.convert_to_tensor(x)
    x_placeholder = array_ops.placeholder_with_default(x, shape=None)
    x_list = x.tolist()

    # For known and matching operator dimensions, interpret all as non-blockwise
    op_dimension_values = [2, 1, 3]
    op_dimensions = [tensor_shape.Dimension(d) for d in op_dimension_values]
    for inputs in [x, x_tensor, x_placeholder, x_list]:
      self.assertFalse(linear_operator_util.arg_is_blockwise(
          op_dimensions, inputs, -1))

    # The input is still interpreted as non-blockwise for unknown operator
    # dimensions (`x_list` has an outermost dimension that does not matcn the
    # number of blocks, and the other inputs are not iterables).
    unknown_op_dimensions = [
        tensor_shape.Dimension(None) for _ in op_dimension_values]
    for inputs in [x, x_tensor, x_placeholder, x_list]:
      self.assertFalse(linear_operator_util.arg_is_blockwise(
          unknown_op_dimensions, inputs, -1))

  def test_ambiguous_input_raises(self):
    # [FUNCTION DOC]
    # Test: a nested list whose leftmost size equals the number of operators while operator
    # dimensions are unknown is ambiguous; the helper raises a value error mentioning that the
    # structure is ambiguous.
    x = np.zeros((3, 4, 2)).tolist()
    op_dimensions = [tensor_shape.Dimension(None) for _ in range(3)]

    # Since the leftmost dimension of `x` is equal to the number of blocks, and
    # the operators have unknown dimension, the input is ambiguous.
    with self.assertRaisesRegex(ValueError, "structure is ambiguous"):
      linear_operator_util.arg_is_blockwise(op_dimensions, x, -2)

  def test_mismatched_input_raises(self):
    # [FUNCTION DOC]
    # Test: operator dimensions summing to 7 against a two-element list whose dimensions sum to 12
    # (blockwise) or 6 (non-blockwise) match neither; the helper raises a value error mentioning
    # "dimension does not match".
    x = np.zeros((2, 3, 4, 6)).tolist()
    op_dimension_values = [4, 3]
    op_dimensions = [tensor_shape.Dimension(v) for v in op_dimension_values]

    # The dimensions of the two operator-blocks sum to 7. `x` is a
    # two-element list; if interpreted blockwise, its corresponding dimensions
    # sum to 12 (=6*2). If not interpreted blockwise, its corresponding
    # dimension is 6. This is a mismatch.
    with self.assertRaisesRegex(ValueError, "dimension does not match"):
      linear_operator_util.arg_is_blockwise(op_dimensions, x, -1)

if __name__ == "__main__":
  test.main()
