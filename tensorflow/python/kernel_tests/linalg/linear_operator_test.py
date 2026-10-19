# [MODULE DOC]
# Test module for the abstract linear operator base class in the linalg package. Defines two
# minimal concrete operators used as test doubles: one that only defines its shape (to test that
# all shape-related properties and methods derive from it) and one that wraps a dense matrix and
# implements shape, matmul and solve (to test generic behavior such as densification, matrix-
# vector products, square-ness hints, flag propagation through matmul and dispatch of the
# generic linear algebra functions to operators). The single test class runs each test in both
# graph and eager modes. Creates a module-level seeded random generator; has no import-time side
# effects otherwise. When run as a script it starts the test runner.
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

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(123)


class LinearOperatorShape(linalg.LinearOperator):
  # [CLASS DOC]
  # HELPER: test double for the linear operator base class tests in this module.
  # A minimal linear operator that only knows its shape: it stores a given shape and implements
  # the static and dynamic shape hooks from it, so the tests can check that all other shape
  # properties and methods of the base class derive from those two. Always has dtype 32-bit
  # float; accepts the optional property flags (non-singular, self-adjoint, positive definite,
  # square) and records all constructor arguments as the operator's parameters. Does not
  # implement real matrix multiplication.
  """LinearOperator that implements the methods ._shape and _shape_tensor."""

  def __init__(self,
               shape,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: shape (sequence of dimensions of the pretend operator, stored for the
    # shape hooks) and the optional flags is_non_singular, is_self_adjoint, is_positive_definite
    # and is_square (each true, false or nothing for unknown; default nothing). Collects all
    # arguments into a parameters mapping and calls the base class constructor with dtype
    # 32-bit float, the flags and the parameters. Side effect: stores the shape on the instance.
    parameters = dict(
        shape=shape,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square
    )

    self._stored_shape = shape
    super(LinearOperatorShape, self).__init__(
        dtype=dtypes.float32,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        parameters=parameters)

  def _shape(self):
    # [FUNCTION DOC]
    # Static shape hook of the operator interface: returns the stored shape converted to a
    # static tensor-shape object. Takes no inputs.
    return tensor_shape.TensorShape(self._stored_shape)

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # Dynamic shape hook of the operator interface: returns the stored shape as a constant
    # 32-bit integer tensor. Takes no inputs.
    return constant_op.constant(self._stored_shape, dtype=dtypes.int32)

  def _matmul(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the signature takes no arguments, unlike the operator interface's matmul hook
    # (which receives the right-hand side and the adjoint flags), so calling it through the
    # base class would fail with an argument-count error instead of the error below; it is
    # simply never called by these tests.
    # Placeholder: always raises a not-implemented error stating it is not needed for this test.
    raise NotImplementedError("Not needed for this test.")


class LinearOperatorMatmulSolve(linalg.LinearOperator):
  # [CLASS DOC]
  # HELPER: test double for the linear operator base class tests in this module.
  # A minimal linear operator wrapping a dense matrix (possibly batched), taking its dtype from
  # the matrix, and implementing the shape hooks, matmul and solve with ordinary matrix
  # multiplication and a linear solve. Accepts the optional property flags and records the
  # constructor arguments as parameters. Used to test generic base-class behavior such as
  # densification and flag propagation.
  """LinearOperator that wraps a [batch] matrix and implements matmul/solve."""

  def __init__(self,
               matrix,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: matrix (array-like or tensor, possibly batched, converted to a tensor
    # and stored) and the optional flags is_non_singular, is_self_adjoint, is_positive_definite
    # and is_square (each true, false or nothing for unknown; default nothing). Collects all
    # arguments (the original, unconverted matrix included) into a parameters mapping and calls
    # the base class constructor with the matrix's dtype, the flags and the parameters. Side
    # effect: stores the converted matrix on the instance.
    parameters = dict(
        matrix=matrix,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square
    )

    self._matrix = ops.convert_to_tensor(matrix, name="matrix")
    super(LinearOperatorMatmulSolve, self).__init__(
        dtype=self._matrix.dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        parameters=parameters)

  def _shape(self):
    # [FUNCTION DOC]
    # Static shape hook of the operator interface: returns the static shape of the stored matrix.
    return self._matrix.shape

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # Dynamic shape hook of the operator interface: returns the shape of the stored matrix as a
    # tensor computed at run time.
    return array_ops.shape(self._matrix)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Matrix multiplication hook of the operator interface. Inputs: x (right-hand side, converted
    # to a tensor), adjoint (use the conjugate transpose of the stored matrix; default false) and
    # adjoint_arg (use the conjugate transpose of x; default false). Returns the matrix product
    # of the stored matrix with x using the generic batched matrix multiply.
    x = ops.convert_to_tensor(x, name="x")
    return math_ops.matmul(
        self._matrix, x, adjoint_a=adjoint, adjoint_b=adjoint_arg)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Linear solve hook of the operator interface. Inputs: rhs (right-hand side, converted to a
    # tensor), adjoint (solve with the conjugate transpose of the stored matrix; default false)
    # and adjoint_arg (must be false). Returns the solution of the matrix system via the generic
    # matrix solve. Raises an assertion error saying it is not implemented when adjoint_arg is
    # true.
    rhs = ops.convert_to_tensor(rhs, name="rhs")
    assert not adjoint_arg, "Not implemented for this test class."
    return linalg_ops.matrix_solve(self._matrix, rhs, adjoint=adjoint)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the generic behavior of the abstract linear operator base class, using the two
  # test-double operators defined in this module and, for the dispatch tests, the full-matrix
  # operator. Covers shape-derived properties, flags, parameters, generic densification,
  # matrix-vector helpers, square-ness inference and validation, hint propagation through matmul,
  # dispatch of generic linear algebra functions, and vectorized mapping. Runs in graph and eager
  # modes. No shared fixtures.

  def test_all_shape_properties_defined_by_the_one_property_shape(self):
    # [FUNCTION DOC]
    # Test: an operator defined only by the shape (1, 2, 3, 4) reports tensor rank 4, batch shape
    # (1, 2), domain dimension 4 (last) and range dimension 3 (second to last), all derived from
    # the single shape property. Also asserts the parameters mapping holds the shape and all
    # flags as nothing.

    shape = (1, 2, 3, 4)
    operator = LinearOperatorShape(shape)

    self.assertAllEqual(shape, operator.shape)
    self.assertAllEqual(4, operator.tensor_rank)
    self.assertAllEqual((1, 2), operator.batch_shape)
    self.assertAllEqual(4, operator.domain_dimension)
    self.assertAllEqual(3, operator.range_dimension)
    expected_parameters = {
        "is_non_singular": None,
        "is_positive_definite": None,
        "is_self_adjoint": None,
        "is_square": None,
        "shape": (1, 2, 3, 4),
    }
    self.assertEqual(expected_parameters, operator.parameters)

  def test_all_shape_methods_defined_by_the_one_method_shape(self):
    # [FUNCTION DOC]
    # Test: for the same shape-only operator, the dynamic counterparts (shape tensor, tensor rank
    # tensor, batch shape tensor, domain and range dimension tensors) evaluate to the expected
    # values (1, 2, 3, 4), 4, (1, 2), 4 and 3, all derived from the one dynamic shape method.
    with self.cached_session():
      shape = (1, 2, 3, 4)
      operator = LinearOperatorShape(shape)

      self.assertAllEqual(shape, self.evaluate(operator.shape_tensor()))
      self.assertAllEqual(4, self.evaluate(operator.tensor_rank_tensor()))
      self.assertAllEqual((1, 2), self.evaluate(operator.batch_shape_tensor()))
      self.assertAllEqual(4, self.evaluate(operator.domain_dimension_tensor()))
      self.assertAllEqual(3, self.evaluate(operator.range_dimension_tensor()))

  def test_is_x_properties(self):
    # [FUNCTION DOC]
    # Test: flags given at construction (not non-singular, self-adjoint, not positive definite)
    # on a 2 by 2 shape-only operator are reported unchanged by the operator's properties.
    operator = LinearOperatorShape(
        shape=(2, 2),
        is_non_singular=False,
        is_self_adjoint=True,
        is_positive_definite=False)
    self.assertFalse(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)
    self.assertFalse(operator.is_positive_definite)

  def test_nontrivial_parameters(self):
    # [FUNCTION DOC]
    # Test: for a matrix-wrapping operator built from a placeholder tensor of unknown shape, the
    # parameters mapping holds the very placeholder (not a converted copy) under the matrix key
    # along with all flags as nothing.
    matrix = rng.randn(2, 3, 4)
    matrix_ph = array_ops.placeholder_with_default(input=matrix, shape=None)
    operator = LinearOperatorMatmulSolve(matrix_ph)
    expected_parameters = {
        "is_non_singular": None,
        "is_positive_definite": None,
        "is_self_adjoint": None,
        "is_square": None,
        "matrix": matrix_ph,
    }
    self.assertEqual(expected_parameters, operator.parameters)

  def test_generic_to_dense_method_non_square_matrix_static(self):
    # [FUNCTION DOC]
    # Test: the generic densification of a matrix-wrapping operator with a statically known
    # non-square batched shape (2, 3, 4) returns a tensor of that static shape whose values
    # match the original matrix.
    matrix = rng.randn(2, 3, 4)
    operator = LinearOperatorMatmulSolve(matrix)
    with self.cached_session():
      operator_dense = operator.to_dense()
      self.assertAllEqual((2, 3, 4), operator_dense.shape)
      self.assertAllClose(matrix, self.evaluate(operator_dense))

  def test_generic_to_dense_method_non_square_matrix_tensor(self):
    # [FUNCTION DOC]
    # Test: like the static densification test but the matrix is a placeholder of unknown shape;
    # the dense result must numerically match the original (2, 3, 4) matrix.
    matrix = rng.randn(2, 3, 4)
    matrix_ph = array_ops.placeholder_with_default(input=matrix, shape=None)
    operator = LinearOperatorMatmulSolve(matrix_ph)
    operator_dense = operator.to_dense()
    self.assertAllClose(matrix, self.evaluate(operator_dense))

  def test_matvec(self):
    # [FUNCTION DOC]
    # Test: matrix-vector product of the diagonal matrix with entries one and two with the vector
    # of ones gives a vector of shape (2) equal to one and two.
    matrix = [[1., 0], [0., 2.]]
    operator = LinearOperatorMatmulSolve(matrix)
    x = [1., 1.]
    with self.cached_session():
      y = operator.matvec(x)
      self.assertAllEqual((2,), y.shape)
      self.assertAllClose([1., 2.], self.evaluate(y))

  def test_solvevec(self):
    # [FUNCTION DOC]
    # Test: solving the diagonal matrix (entries one and two) against the vector of ones gives a
    # vector of shape (2) equal to one and one half.
    matrix = [[1., 0], [0., 2.]]
    operator = LinearOperatorMatmulSolve(matrix)
    y = [1., 1.]
    with self.cached_session():
      x = operator.solvevec(y)
      self.assertAllEqual((2,), x.shape)
      self.assertAllClose([1., 1 / 2.], self.evaluate(x))

  def test_is_square_set_to_true_for_square_static_shapes(self):
    # [FUNCTION DOC]
    # Test: a shape-only operator with static shape (2, 4, 4) automatically reports being square
    # when no square flag is given.
    operator = LinearOperatorShape(shape=(2, 4, 4))
    self.assertTrue(operator.is_square)

  def test_is_square_set_to_false_for_square_static_shapes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says "false for square static shapes" but the shape used, (2, 3, 4),
    # is not square; the test actually checks that a non-square static shape automatically
    # reports not being square.
    # Test: a shape-only operator with static shape (2, 3, 4) and no square flag reports not
    # being square.
    operator = LinearOperatorShape(shape=(2, 3, 4))
    self.assertFalse(operator.is_square)

  def test_is_square_set_incorrectly_to_false_raises(self):
    # [FUNCTION DOC]
    # Test: declaring the square flag false for a square static shape (2, 4, 4) makes reading the
    # square property raise a value error saying the flag was set but the operator was square.
    with self.assertRaisesRegex(ValueError, "but.*was square"):
      _ = LinearOperatorShape(shape=(2, 4, 4), is_square=False).is_square

  def test_is_square_set_inconsistent_with_other_hints_raises(self):
    # [FUNCTION DOC]
    # Test: declaring the operator non-square while also declaring it non-singular, or while
    # declaring it positive definite (properties that imply a square operator), makes the
    # constructor raise a value error saying the operator is always square. The matrix is a
    # placeholder holding an empty value with unknown shape.
    with self.assertRaisesRegex(ValueError, "is always square"):
      matrix = array_ops.placeholder_with_default(input=(), shape=None)
      LinearOperatorMatmulSolve(matrix, is_non_singular=True, is_square=False)

    with self.assertRaisesRegex(ValueError, "is always square"):
      matrix = array_ops.placeholder_with_default(input=(), shape=None)
      LinearOperatorMatmulSolve(
          matrix, is_positive_definite=True, is_square=False)

  def test_non_square_operators_raise_on_determinant_and_solve(self):
    # [FUNCTION DOC]
    # Test: for a non-square 2 by 3 shape-only operator, asking for the determinant, the log
    # absolute determinant or a solve raises a not-implemented error saying the operator may not
    # be square. The last block, declaring a positive-definite operator non-square, repeats the
    # inconsistent-hints constructor check from another test (value error saying it is always
    # square) rather than anything about determinants or solves.
    operator = LinearOperatorShape((2, 3))
    with self.assertRaisesRegex(NotImplementedError, "not be square"):
      operator.determinant()
    with self.assertRaisesRegex(NotImplementedError, "not be square"):
      operator.log_abs_determinant()
    with self.assertRaisesRegex(NotImplementedError, "not be square"):
      operator.solve(rng.rand(2, 2))

    with self.assertRaisesRegex(ValueError, "is always square"):
      matrix = array_ops.placeholder_with_default(input=(), shape=None)
      LinearOperatorMatmulSolve(
          matrix, is_positive_definite=True, is_square=False)

  def test_is_square_manual_set_works(self):
    # [FUNCTION DOC]
    # Test: with a placeholder matrix of unknown static shape, the square property is unknown
    # (nothing) in graph mode, because it must rely on the hint, while eager mode can read the
    # value and discover it is square (not asserted there). Passing an explicit square flag of
    # true makes the property true in both modes.
    matrix = array_ops.placeholder_with_default(
        input=np.ones((2, 2)), shape=None)
    operator = LinearOperatorMatmulSolve(matrix)
    if not context.executing_eagerly():
      # Eager mode will read in the default value, and discover the answer is
      # True.  Graph mode must rely on the hint, since the placeholder has
      # shape=None...the hint is, by default, None.
      self.assertEqual(None, operator.is_square)

    # Set to True
    operator = LinearOperatorMatmulSolve(matrix, is_square=True)
    self.assertTrue(operator.is_square)

  def test_linear_operator_matmul_hints_closed(self):
    # [FUNCTION DOC]
    # Test: property flags of the lazy product of two operators. For an operator with no flags,
    # multiplying it with itself leaves non-singular, self-adjoint and positive-definite unknown
    # (square is also unknown in graph mode; eager mode can see the data). For an operator
    # flagged square, non-singular, self-adjoint and positive definite, multiplying it with
    # itself is square, non-singular, self-adjoint (a power of a self-adjoint operator) and
    # positive definite (non-singular and equal to its own adjoint product).
    matrix = array_ops.placeholder_with_default(input=np.ones((2, 2)),
                                                shape=None)
    operator1 = LinearOperatorMatmulSolve(matrix)

    operator_matmul = operator1.matmul(operator1)

    if not context.executing_eagerly():
      # Eager mode will read in the input and discover matrix is square.
      self.assertEqual(None, operator_matmul.is_square)
    self.assertEqual(None, operator_matmul.is_non_singular)
    self.assertEqual(None, operator_matmul.is_self_adjoint)
    self.assertEqual(None, operator_matmul.is_positive_definite)

    operator2 = LinearOperatorMatmulSolve(
        matrix,
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=True,
        is_square=True,
    )

    operator_matmul = operator2.matmul(operator2)

    self.assertTrue(operator_matmul.is_square)
    self.assertTrue(operator_matmul.is_non_singular)

    # A @ A is SA since A is.
    self.assertEqual(True, operator_matmul.is_self_adjoint)

    # A @ A is non-singular (since A is) and A @ A = A @ A.H is semi-def so...
    self.assertEqual(True, operator_matmul.is_positive_definite)

  def test_linear_operator_matmul_hints_false(self):
    # [FUNCTION DOC]
    # Test: flag propagation when operands are flagged false. A square operator flagged
    # singular, not self-adjoint and not positive definite multiplied with itself is square and
    # singular, with self-adjointness and positive-definiteness unknown. A non-square 2 by 3
    # operator (all flags false) multiplied with its own adjoint is recognized as the form A times
    # A-adjoint, so the product is square and self-adjoint; its non-singular flag is false in
    # eager mode but unknown in graph mode (see the to-do about this inconsistency), and positive
    # definiteness cannot be deduced in either mode.
    matrix1 = array_ops.placeholder_with_default(
        input=rng.rand(2, 2), shape=None)
    operator1 = LinearOperatorMatmulSolve(
        matrix1,
        is_non_singular=False,
        is_self_adjoint=False,
        is_positive_definite=False,
        is_square=True,
    )

    operator_matmul = operator1.matmul(operator1)

    self.assertTrue(operator_matmul.is_square)
    self.assertFalse(operator_matmul.is_non_singular)
    self.assertEqual(None, operator_matmul.is_self_adjoint)
    self.assertEqual(None, operator_matmul.is_positive_definite)

    matrix2 = array_ops.placeholder_with_default(
        input=rng.rand(2, 3), shape=None)
    operator2 = LinearOperatorMatmulSolve(
        matrix2,
        is_non_singular=False,
        is_self_adjoint=False,
        is_positive_definite=False,
        is_square=False,
    )

    operator_matmul = operator2.matmul(operator2, adjoint_arg=True)

    # Composition recognizes this as the form A @ A.H, which is square, SA.
    self.assertTrue(operator_matmul.is_square)
    self.assertTrue(operator_matmul.is_self_adjoint)

    if context.executing_eagerly():
      # False since we specified is_non_singular=False.
      self.assertFalse(operator_matmul.is_non_singular)
    else:
      # May be non-singular, since it's the composition of two non-square.
      # TODO(b/136162840) This is a bit inconsistent, and should probably be
      # False since we specified operator2.is_non_singular == False.
      self.assertIsNone(operator_matmul.is_non_singular)

    # No way to deduce these, even in Eager mode.
    self.assertIsNone(operator_matmul.is_positive_definite)

  def test_linear_operator_matmul_hint_infer_square(self):
    # [FUNCTION DOC]
    # Test: square-ness inference for products of non-square operators with known static shapes:
    # operators of shape 2 by 3 and 3 by 2 multiplied in either order give a square operator,
    # while 2 by 3 times 3 by 4 does not.
    matrix1 = array_ops.placeholder_with_default(
        input=rng.rand(2, 3), shape=(2, 3))
    matrix2 = array_ops.placeholder_with_default(
        input=rng.rand(3, 2), shape=(3, 2))
    matrix3 = array_ops.placeholder_with_default(
        input=rng.rand(3, 4), shape=(3, 4))

    operator1 = LinearOperatorMatmulSolve(matrix1, is_square=False)
    operator2 = LinearOperatorMatmulSolve(matrix2, is_square=False)
    operator3 = LinearOperatorMatmulSolve(matrix3, is_square=False)

    self.assertTrue(operator1.matmul(operator2).is_square)
    self.assertTrue(operator2.matmul(operator1).is_square)
    self.assertFalse(operator1.matmul(operator3).is_square)

  def testDispatchedMethods(self):
    # [FUNCTION DOC]
    # Test: the generic linear algebra functions (trace, diagonal part, log absolute determinant
    # via the log-determinant function, determinant) applied to a 2 by 2 full-matrix operator
    # give the same values as the corresponding operator methods. Also asserts that the generic
    # adjoint, Cholesky and inverse functions return linear operator objects. Solve and matmul
    # are covered in other tests (per the comment).
    operator = linalg.LinearOperatorFullMatrix(
        [[1., 0.5], [0.5, 1.]],
        is_square=True,
        is_self_adjoint=True,
        is_non_singular=True,
        is_positive_definite=True)
    methods = {
        "trace": linalg.trace,
        "diag_part": linalg.diag_part,
        "log_abs_determinant": linalg.logdet,
        "determinant": linalg.det
    }
    for method in methods:
      op_val = getattr(operator, method)()
      linalg_val = methods[method](operator)
      self.assertAllClose(
          self.evaluate(op_val),
          self.evaluate(linalg_val))
    # Solve and Matmul go here.

    adjoint = linalg.adjoint(operator)
    self.assertIsInstance(adjoint, linalg.LinearOperator)
    cholesky = linalg.cholesky(operator)
    self.assertIsInstance(cholesky, linalg.LinearOperator)
    inverse = linalg.inv(operator)
    self.assertIsInstance(inverse, linalg.LinearOperator)

  def testDispatchMatmulSolve(self):
    # [FUNCTION DOC]
    # Test: for a 2 by 2 float64 full-matrix operator and a random batch of three 2 by 2
    # right-hand sides, the generic matrix multiply function with the operator as left operand
    # matches the operator's own matmul, for all four combinations of adjoint and adjoint-argument
    # flags; and the generic solve function matches the operator's solve for both adjoint
    # settings.
    operator = linalg.LinearOperatorFullMatrix(
        np.float64([[1., 0.5], [0.5, 1.]]),
        is_square=True,
        is_self_adjoint=True,
        is_non_singular=True,
        is_positive_definite=True)
    rhs = np.random.uniform(-1., 1., size=[3, 2, 2])
    for adjoint in [False, True]:
      for adjoint_arg in [False, True]:
        op_val = operator.matmul(
            rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
        matmul_val = math_ops.matmul(
            operator, rhs, adjoint_a=adjoint, adjoint_b=adjoint_arg)
        self.assertAllClose(
            self.evaluate(op_val), self.evaluate(matmul_val))

      op_val = operator.solve(rhs, adjoint=adjoint)
      solve_val = linalg.solve(operator, rhs, adjoint=adjoint)
      self.assertAllClose(
          self.evaluate(op_val), self.evaluate(solve_val))

  def testDispatchMatmulLeftOperatorIsTensor(self):
    # [FUNCTION DOC]
    # Test: the generic matrix multiply function with a plain tensor as the left operand and a
    # full-matrix operator as the right operand gives the same values as multiplying by the
    # plain dense matrix, for all four combinations of adjoint flags on the left and right
    # arguments (random batch of three 2 by 2 matrices on the left).
    mat = np.float64([[1., 0.5], [0.5, 1.]])
    right_operator = linalg.LinearOperatorFullMatrix(
        mat,
        is_square=True,
        is_self_adjoint=True,
        is_non_singular=True,
        is_positive_definite=True)
    lhs = np.random.uniform(-1., 1., size=[3, 2, 2])

    for adjoint in [False, True]:
      for adjoint_arg in [False, True]:
        op_val = math_ops.matmul(
            lhs, mat, adjoint_a=adjoint, adjoint_b=adjoint_arg)
        matmul_val = math_ops.matmul(
            lhs, right_operator, adjoint_a=adjoint, adjoint_b=adjoint_arg)
        self.assertAllClose(
            self.evaluate(op_val), self.evaluate(matmul_val))

  def testVectorizedMap(self):
    # [FUNCTION DOC]
    # Test: vectorized mapping over a function that returns a linear operator. The mapped
    # function builds a full-matrix operator from its input scaled by a constant length-2 vector,
    # so each call yields a batch of 2 matrices. Mapping over a random 3 by 5 by 5 float32 input
    # must return a single linear operator object whose batch shape is (3, 2).

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: per-element function for the vectorized mapping test.
      # Input: x, a square matrix tensor. Scales it by the constant vector (3, 4) along a new
      # leading batch dimension (giving shape 2 by N by N) and returns a full-matrix operator
      # flagged square built from the result. No side effects.
      y = constant_op.constant([3., 4.])
      # Make a [2, N, N] shaped operator.
      x = x * y[..., array_ops.newaxis, array_ops.newaxis]
      operator = linalg.LinearOperatorFullMatrix(
          x, is_square=True)
      return operator

    x = np.random.uniform(-1., 1., size=[3, 5, 5]).astype(np.float32)
    batched_operator = control_flow_ops.vectorized_map(
        fn, ops.convert_to_tensor(x))
    self.assertIsInstance(batched_operator, linalg.LinearOperator)
    self.assertAllEqual(batched_operator.batch_shape, [3, 2])


if __name__ == "__main__":
  test.main()
