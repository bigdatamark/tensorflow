# [MODULE DOC]
# Defines LinearOperatorLowerTriangular, a linear operator backed by a dense [batch] square matrix
# whose strictly upper triangle is ignored. Exposes only that class (exported publicly). Provides
# cheap determinant (product of the diagonal), triangular solves, eigenvalues and a fused product
# with diagonal operators. No import-time side effects other than export registration and
# composite tensor wrapping.
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
"""`LinearOperator` acting like a lower triangular matrix."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import property_hint_util
from tensorflow.python.util.tf_export import tf_export

__all__ = [
    "LinearOperatorLowerTriangular",
]


@tf_export("linalg.LinearOperatorLowerTriangular")
@linear_operator.make_composite_tensor
class LinearOperatorLowerTriangular(linear_operator.LinearOperator):
  # [CLASS DOC]
  # Operator acting like a [batch] N x N lower triangular matrix defined by a dense tensor; the
  # upper triangle of the input is ignored (masked out when used). Always square. Holds the
  # stored tensor in a private attribute. Matmul uses the masked matrix, solves use back
  # substitution, determinant is the product of the diagonal, eigenvalues are the diagonal.
  """`LinearOperator` acting like a [batch] square lower triangular matrix.

  This operator acts like a [batch] lower triangular matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.

  `LinearOperatorLowerTriangular` is initialized with a `Tensor` having
  dimensions `[B1,...,Bb, N, N]`. The upper triangle of the last two
  dimensions is ignored.

  ```python
  # Create a 2 x 2 lower-triangular linear operator.
  tril = [[1., 2.], [3., 4.]]
  operator = LinearOperatorLowerTriangular(tril)

  # The upper triangle is ignored.
  operator.to_dense()
  ==> [[1., 0.]
       [3., 4.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 4 linear operators.
  tril = tf.random.normal(shape=[2, 3, 4, 4])
  operator = LinearOperatorLowerTriangular(tril)
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorLowerTriangular` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` involves `N^2 * R` multiplications.
  * `operator.solve(x)` involves `N * R` size `N` back-substitutions.
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

  def __init__(self,
               tril,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorLowerTriangular"):
    # [FUNCTION DOC]
    # Constructor. Input: tril (tensor of shape batch + [N, N], at least 2 dimensions), property
    # hints is_non_singular, is_self_adjoint, is_positive_definite, is_square (booleans or
    # nothing) and a name. Converts tril to a tensor inside a name scope, checks its rank, and
    # calls the parent constructor with the tensor dtype and is_square forced to true.
    # Raises a value error if is_square is explicitly false or if tril has fewer than 2 dims.
    r"""Initialize a `LinearOperatorLowerTriangular`.

    Args:
      tril:  Shape `[B1,...,Bb, N, N]` with `b >= 0`, `N >= 0`.
        The lower triangular part of `tril` defines this operator.  The strictly
        upper triangle is ignored.
      is_non_singular:  Expect that this operator is non-singular.
        This operator is non-singular if and only if its diagonal elements are
        all non-zero.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  This operator is self-adjoint only if it is diagonal with
        real-valued diagonal entries.  In this case it is advised to use
        `LinearOperatorDiag`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      ValueError:  If `is_square` is `False`.
    """
    parameters = dict(
        tril=tril,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    if is_square is False:
      raise ValueError(
          "Only square lower triangular operators supported at this time.")
    is_square = True

    with ops.name_scope(name, values=[tril]):
      self._tril = linear_operator_util.convert_nonref_to_tensor(tril,
                                                                 name="tril")
      self._check_tril(self._tril)

      super(LinearOperatorLowerTriangular, self).__init__(
          dtype=self._tril.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  @property
  def tril(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored (unmasked) tensor that defines the operator.
    """The lower triangular matrix defining this operator."""
    return self._tril

  def _check_tril(self, tril):
    # [FUNCTION DOC]
    # HELPER: Argument validation used by the constructor.
    # Statically checks that tril has at least two dimensions when its rank is known. Returns
    # nothing; raises a value error otherwise.
    """Static check of the `tril` argument."""

    if tril.shape.ndims is not None and tril.shape.ndims < 2:
      raise ValueError(
          "Argument tril must have at least 2 dimensions.  Found: %s"
          % tril)

  def _get_tril(self):
    # [FUNCTION DOC]
    # HELPER: Returns the stored matrix with everything above the diagonal set to zero (lower
    # band only), used by matmul, solve and to-dense.
    """Gets the `tril` kwarg, with upper part zero-d out."""
    return array_ops.matrix_band_part(self._tril, -1, 0)

  def _get_diag(self):
    # [FUNCTION DOC]
    # HELPER: Returns the main diagonal of the stored tensor, shape batch + [N]; used by
    # determinant, eigenvalue and singularity-assertion methods.
    """Gets the diagonal part of `tril` kwarg."""
    return array_ops.matrix_diag_part(self._tril)

  def _shape(self):
    # [FUNCTION DOC]
    # Returns the static shape of the operator, which is the static shape of the stored tensor.
    return self._tril.shape

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # Returns the dynamic shape of the operator as an integer tensor (shape of stored tensor).
    return array_ops.shape(self._tril)

  def _assert_non_singular(self):
    # [FUNCTION DOC]
    # Builds an assertion that the operator is non-singular, i.e. that no diagonal entry has zero
    # modulus. Returns an assertion op (nothing in eager mode); fails at run time with a message
    # about a singular operator if any diagonal entry is zero.
    return linear_operator_util.assert_no_entries_with_modulus_zero(
        self._get_diag(),
        message="Singular operator:  Diagonal contained zero values.")

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Multiplies the lower triangular part of the matrix (or its adjoint) with x (or its adjoint
    # when adjoint_arg is true). Inputs: x tensor, adjoint and adjoint_arg booleans (default
    # False). Returns the dense product tensor.
    return math_ops.matmul(
        self._get_tril(), x, adjoint_a=adjoint, adjoint_b=adjoint_arg)

  def _linop_matmul(
      self,
      left_operator: "LinearOperatorLowerTriangular",
      right_operator: linear_operator.LinearOperator,
    ) -> linear_operator.LinearOperator:
    # [FUNCTION DOC]
    # Hook for composing this operator with another operator by matrix product. If the right
    # operator looks like a diagonal operator (has a diagonal check method), returns a new lower
    # triangular operator whose matrix is the dense left matrix with columns scaled by the right
    # diagonal, with non-singular and self-adjoint hints combined from both and positive-definite
    # unknown. Otherwise defers to the parent implementation. Inputs: left_operator (a lower
    # triangular operator), right_operator (any linear operator). Returns a linear operator.
    # instance check of linear_operator_diag.LinearOperatorDiag
    if hasattr(right_operator, "_check_diag"):
      return LinearOperatorLowerTriangular(
          tril=left_operator.to_dense() * right_operator.diag,
          is_non_singular=property_hint_util.combined_non_singular_hint(
              right_operator, left_operator),
          # This is safe to do since the Triangular matrix is only self-adjoint
          # when it is a diagonal matrix, and hence commutes.
          is_self_adjoint=property_hint_util.combined_commuting_self_adjoint_hint(
              right_operator, left_operator),
          is_positive_definite=None,
          is_square=True)
    return super()._linop_matmul(left_operator, right_operator)

  def _determinant(self):
    # [FUNCTION DOC]
    # Returns the determinant: the product of the diagonal entries, reduced over the last axis.
    return math_ops.reduce_prod(self._get_diag(), axis=[-1])

  def _log_abs_determinant(self):
    # [FUNCTION DOC]
    # Returns the log of the absolute determinant: sum over the diagonal of the log of the
    # absolute value of each entry.
    return math_ops.reduce_sum(
        math_ops.log(math_ops.abs(self._get_diag())), axis=[-1])

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Solves the triangular system with the masked lower matrix by back substitution. Inputs: rhs
    # tensor, adjoint (solve with the adjoint matrix) and adjoint_arg (use the adjoint of rhs),
    # both default False. Returns the solution tensor.
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    return linalg.triangular_solve(
        self._get_tril(), rhs, lower=True, adjoint=adjoint)

  def _to_dense(self):
    # [FUNCTION DOC]
    # Returns the dense matrix of the operator: the stored tensor with the upper part zeroed.
    return self._get_tril()

  def _eigvals(self):
    # [FUNCTION DOC]
    # Returns the eigenvalues of the operator, which for a triangular matrix are its diagonal.
    return self._get_diag()

  @property
  def _composite_tensor_fields(self):
    # [FUNCTION DOC]
    # Property giving the tuple of constructor argument names stored for composite tensor
    # conversion; here only the stored matrix.
    return ("tril",)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    # [FUNCTION DOC]
    # Property mapping the parameter name of the stored matrix to its number of matrix
    # dimensions (two), used for batch slicing in composite tensor handling.
    return {"tril": 2}
