# [MODULE DOC]
# Defines LinearOperatorDiag, a LinearOperator that represents a (batch of) square diagonal
# matrices stored only as the vector of diagonal entries. It exposes the public operator class
# (exported as the linalg diagonal operator) with matmul, solve, determinant, inverse, adjoint,
# Cholesky, eigenvalues, condition number, dense conversion and add-to-tensor implemented
# directly on the diagonal, plus fast paths for combining with other diagonal, scaled-identity and
# lower-triangular operators. Import-time side effects: the class is registered for public export
# and made into a composite-tensor type. Not a helper module.
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
"""`LinearOperator` acting like a diagonal matrix."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import property_hint_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorDiag",]


@tf_export("linalg.LinearOperatorDiag")
@linear_operator.make_composite_tensor
class LinearOperatorDiag(linear_operator.LinearOperator):
  # [CLASS DOC]
  # Linear operator acting like a (batch of) square diagonal matrix. Holds a single tensor
  # attribute, the diagonal, with shape batch dimensions followed by N; the operator shape is the
  # batch shape followed by N and N. The dense matrix is never materialized except by to-dense.
  # Subclasses the generic LinearOperator and overrides the shape, matmul, solve, determinant,
  # assertion, adjoint, inverse, Cholesky, eigenvalue and condition-number hooks. Created by
  # passing the diagonal tensor plus optional property hints; decorated so that instances are
  # composite tensors and exported publicly.
  """`LinearOperator` acting like a [batch] square diagonal matrix.

  This operator acts like a [batch] diagonal matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorDiag` is initialized with a (batch) vector.

  ```python
  # Create a 2 x 2 diagonal linear operator.
  diag = [1., -1.]
  operator = LinearOperatorDiag(diag)

  operator.to_dense()
  ==> [[1.,  0.]
       [0., -1.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 4 linear operators.
  diag = tf.random.normal(shape=[2, 3, 4])
  operator = LinearOperatorDiag(diag)

  # Create a shape [2, 1, 4, 2] vector.  Note that this shape is compatible
  # since the batch dimensions, [2, 1], are broadcast to
  # operator.batch_shape = [2, 3].
  y = tf.random.normal(shape=[2, 1, 4, 2])
  x = operator.solve(y)
  ==> operator.matmul(x) = y
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorDiag` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` involves `N * R` multiplications.
  * `operator.solve(x)` involves `N` divisions and `N * R` multiplications.
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
               diag,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorDiag"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a value error is raised if the diagonal is real and
    # is_self_adjoint is not true; the code raises only when is_self_adjoint is explicitly false
    # (null/unspecified is silently set to true). The docstring's type error for a disallowed
    # dtype is not raised here; no dtype check is performed in this constructor.
    # Builds the operator. Inputs: diag, a tensor (or tensor-like) of shape batch dims plus N;
    # is_non_singular, is_self_adjoint, is_positive_definite, is_square: optional boolean hints
    # (default null meaning unknown); name, a string name scope. The diagonal is converted to a
    # tensor unless it is a variable-like reference, and statically checked to have rank >= 1.
    # A real diagonal forces is_self_adjoint to true. is_square must not be false and is forced
    # to true. Raises a value error if the diagonal rank is 0, if a real diagonal is declared
    # not self-adjoint, or if is_square is false. Stores the diagonal and the constructor
    # parameters, then calls the base constructor. Returns nothing.
    r"""Initialize a `LinearOperatorDiag`.

    Args:
      diag:  Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.
        The diagonal of the operator.  Allowed dtypes: `float16`, `float32`,
          `float64`, `complex64`, `complex128`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `diag.dtype` is real, this is auto-set to `True`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If `diag.dtype` is not an allowed type.
      ValueError:  If `diag.dtype` is real, and `is_self_adjoint` is not `True`.
    """
    parameters = dict(
        diag=diag,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    with ops.name_scope(name, values=[diag]):
      self._diag = linear_operator_util.convert_nonref_to_tensor(
          diag, name="diag")
      self._check_diag(self._diag)

      # Check and auto-set hints.
      if not self._diag.dtype.is_complex:
        if is_self_adjoint is False:
          raise ValueError("A real diagonal operator is always self adjoint.")
        else:
          is_self_adjoint = True

      if is_square is False:
        raise ValueError("Only square diagonal operators currently supported.")
      is_square = True

      super(LinearOperatorDiag, self).__init__(
          dtype=self._diag.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _check_diag(self, diag):
    # [FUNCTION DOC]
    # HELPER: validation helper for the constructor.
    # Statically checks the diagonal tensor: if its rank is known and less than 1, raises a value
    # error saying it needs at least one dimension. Unknown rank passes. Returns nothing.
    """Static check of diag."""
    if diag.shape.ndims is not None and diag.shape.ndims < 1:
      raise ValueError("Argument diag must have at least 1 dimension.  "
                       "Found: %s" % diag)

  def _shape(self):
    # [FUNCTION DOC]
    # Returns the static operator shape: the diagonal's static shape with its last dimension
    # appended again (batch dims, N, N). No inputs, no side effects.
    # If d_shape = [5, 3], we return [5, 3, 3].
    d_shape = self._diag.shape
    return d_shape.concatenate(d_shape[-1:])

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # Returns the dynamic operator shape as a one-dimensional integer tensor: the runtime shape of
    # the diagonal with its last entry appended once more. Adds graph ops, no other side effects.
    d_shape = array_ops.shape(self._diag)
    k = d_shape[-1]
    return array_ops.concat((d_shape, [k]), 0)

  @property
  def diag(self):
    # [FUNCTION DOC]
    # Property returning the stored diagonal tensor (batch dims followed by N), unchanged.
    return self._diag

  def _linop_inverse(self) -> "LinearOperatorDiag":
    # [FUNCTION DOC]
    # Returns a new diagonal operator whose diagonal is the elementwise reciprocal of this one,
    # i.e. the inverse matrix. The non-singular, self-adjoint and positive-definite hints are
    # copied and is_square is true. Division by zero is not checked here.
    return LinearOperatorDiag(
        1. / self.diag,
        is_non_singular=self.is_non_singular,
        is_self_adjoint=self.is_self_adjoint,
        is_positive_definite=self.is_positive_definite,
        is_square=True)

  def _linop_matmul(
      self,
      left_operator: "LinearOperatorDiag",
      right_operator: linear_operator.LinearOperator,
    ) -> linear_operator.LinearOperator:
    # [FUNCTION DOC]
    # Computes the product of this diagonal operator (left_operator) with right_operator as an
    # operator, using a fast path when possible. Combined hints (non-singular, self-adjoint,
    # positive-definite) come from the property-hint utilities, assuming the factors commute.
    # If the right operator is diagonal, returns a diagonal operator with the elementwise product
    # of diagonals. If it looks like a scaled identity (has a ones-diagonal and a multiplier),
    # returns a diagonal operator with the diagonal scaled by the multiplier. If it is lower
    # triangular, returns a lower-triangular operator whose matrix is the diagonal scaling of the
    # rows of the dense right operator (positive-definite hint dropped). Otherwise defers to the
    # base class implementation. Returns a LinearOperator; no side effects.
    is_non_singular = property_hint_util.combined_non_singular_hint(
        left_operator, right_operator)
    is_self_adjoint = property_hint_util.combined_commuting_self_adjoint_hint(
        left_operator, right_operator)
    is_positive_definite = (
        property_hint_util.combined_commuting_positive_definite_hint(
            left_operator, right_operator))
    if isinstance(right_operator, LinearOperatorDiag):
      return LinearOperatorDiag(
          diag=left_operator.diag * right_operator.diag,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=True,
      )
    # instance of linear_operator_identity.LinearOperatorScaledIdentity
    elif hasattr(right_operator, "_ones_diag") and hasattr(
        right_operator, "multiplier"
    ):
      return LinearOperatorDiag(
          diag=left_operator.diag * right_operator.multiplier,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=True)
    elif isinstance(
        right_operator,
        linear_operator_lower_triangular.LinearOperatorLowerTriangular,
    ):
      return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
          tril=left_operator.diag[..., None] * right_operator.to_dense(),
          is_non_singular=is_non_singular,
          # This is safe to do since the Triangular matrix is only self-adjoint
          # when it is a diagonal matrix, and hence commutes.
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=None,
          is_square=True)
    else:
      return super()._linop_matmul(left_operator, right_operator)

  def _linop_solve(
      self,
      left_operator: "LinearOperatorDiag",
      right_operator: linear_operator.LinearOperator,
  ) -> linear_operator.LinearOperator:
    # [FUNCTION DOC]
    # Computes the operator that solves this diagonal operator (left_operator) against
    # right_operator, i.e. the inverse of the left times the right, with fast paths. Hints are
    # combined as in the matmul case. Diagonal right operator: diagonal operator with the right
    # diagonal divided elementwise by the left diagonal. Scaled identity: diagonal operator of the
    # multiplier divided by the left diagonal. Lower triangular: lower-triangular operator of the
    # dense right matrix with rows divided by the left diagonal (positive-definite hint dropped).
    # Otherwise defers to the base class. Returns a LinearOperator; no side effects.
    is_non_singular = property_hint_util.combined_non_singular_hint(
        left_operator, right_operator)
    is_self_adjoint = property_hint_util.combined_commuting_self_adjoint_hint(
        left_operator, right_operator)
    is_positive_definite = (
        property_hint_util.combined_commuting_positive_definite_hint(
            left_operator, right_operator))
    if isinstance(right_operator, LinearOperatorDiag):
      return LinearOperatorDiag(
          diag=right_operator.diag / left_operator.diag,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=True)
    # instance of linear_operator_identity.LinearOperatorScaledIdentity
    elif (hasattr(right_operator, "_ones_diag")
          and hasattr(right_operator, "multiplier")):
      return LinearOperatorDiag(
          diag=right_operator.multiplier / left_operator.diag,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=True)
    elif isinstance(
        right_operator,
        linear_operator_lower_triangular.LinearOperatorLowerTriangular):
      return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
          tril=right_operator.to_dense() / left_operator.diag[..., None],
          is_non_singular=is_non_singular,
          # This is safe to do since the Triangular matrix is only self-adjoint
          # when it is a diagonal matrix, and hence commutes.
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=None,
          is_square=True)
    else:
      return super()._linop_solve(left_operator, right_operator)

  def _assert_non_singular(self):
    # [FUNCTION DOC]
    # Returns an assertion op (or no-op when statically known) that fails with a "singular
    # operator" message if any diagonal entry has modulus zero.
    return linear_operator_util.assert_no_entries_with_modulus_zero(
        self._diag,
        message="Singular operator:  Diagonal contained zero values.")

  def _assert_positive_definite(self):
    # [FUNCTION DOC]
    # Returns an assertion op checking that the real part of every diagonal entry is strictly
    # positive; the failure message differs for complex versus real dtypes. Graph-side effect only.
    if self.dtype.is_complex:
      message = (
          "Diagonal operator had diagonal entries with non-positive real part, "
          "thus was not positive definite.")
    else:
      message = (
          "Real diagonal operator had non-positive diagonal entries, "
          "thus was not positive definite.")

    return check_ops.assert_positive(
        math_ops.real(self._diag),
        message=message)

  def _assert_self_adjoint(self):
    # [FUNCTION DOC]
    # Returns an assertion op that fails if any diagonal entry has a non-zero imaginary part
    # (a diagonal matrix is self-adjoint exactly when its diagonal is real).
    return linear_operator_util.assert_zero_imag_part(
        self._diag,
        message=(
            "This diagonal operator contained non-zero imaginary values.  "
            " Thus it was not self-adjoint."))

  def _linop_adjoint(self) -> "LinearOperatorDiag":
    # [FUNCTION DOC]
    # Returns the adjoint as a new diagonal operator: the diagonal is complex-conjugated when the
    # dtype is complex, otherwise unchanged. Property hints are copied from this operator.
    diag = self.diag
    if diag.dtype.is_complex:
      diag = math_ops.conj(diag)

    return LinearOperatorDiag(
        diag=diag,
        is_non_singular=self.is_non_singular,
        is_self_adjoint=self.is_self_adjoint,
        is_positive_definite=self.is_positive_definite,
        is_square=True)

  def _linop_cholesky(self) -> "LinearOperatorDiag":
    # [FUNCTION DOC]
    # Returns the Cholesky factor as a diagonal operator whose diagonal is the elementwise square
    # root of this diagonal; marked non-singular, self-adjoint, positive-definite and square.
    # Assumes this operator is positive definite; no check is made.
    return LinearOperatorDiag(
        math_ops.sqrt(self.diag),
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=True,
        is_square=True)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Multiplies this operator (or its adjoint) by a batch matrix x. Inputs: x, tensor of shape
    # batch dims, N, R; adjoint, if true use the conjugated diagonal; adjoint_arg, if true take
    # the conjugate transpose of x first. Computes the diagonal (expanded with a trailing axis)
    # times x by broadcasting. Returns the product tensor; no side effects.
    diag_term = math_ops.conj(self._diag) if adjoint else self._diag
    x = linalg.adjoint(x) if adjoint_arg else x
    diag_mat = array_ops.expand_dims(diag_term, -1)
    return diag_mat * x

  def _matvec(self, x, adjoint=False):
    # [FUNCTION DOC]
    # Applies the operator (or its adjoint when adjoint is true, using the conjugated diagonal) to
    # a batch of vectors x by elementwise multiplication with broadcasting. Returns a tensor.
    diag_term = math_ops.conj(self._diag) if adjoint else self._diag
    return diag_term * x

  def _determinant(self):
    # [FUNCTION DOC]
    # Returns the determinant per batch member: the product of the diagonal entries over the last
    # axis.
    return math_ops.reduce_prod(self._diag, axis=[-1])

  def _log_abs_determinant(self):
    # [FUNCTION DOC]
    # Returns the log of the absolute value of the determinant per batch member: the sum over the
    # last axis of the log of the absolute diagonal entries. For complex dtypes the (real) result
    # is cast to the operator's complex dtype.
    log_det = math_ops.reduce_sum(
        math_ops.log(math_ops.abs(self._diag)), axis=[-1])
    if self.dtype.is_complex:
      log_det = math_ops.cast(log_det, dtype=self.dtype)
    return log_det

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Solves the system with this operator (or its adjoint) for the right-hand side rhs, a batch
    # matrix. adjoint selects the conjugated diagonal; adjoint_arg takes the conjugate transpose
    # of rhs first. Computes rhs multiplied by the reciprocal diagonal (expanded with a trailing
    # axis); zero diagonal entries give infinities/NaNs rather than an error. Returns a tensor.
    diag_term = math_ops.conj(self._diag) if adjoint else self._diag
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    inv_diag_mat = array_ops.expand_dims(1. / diag_term, -1)
    return rhs * inv_diag_mat

  def _to_dense(self):
    # [FUNCTION DOC]
    # Materializes the dense batch of matrices with the diagonal on the main diagonal and zeros
    # elsewhere.
    return array_ops.matrix_diag(self._diag)

  def _diag_part(self):
    # [FUNCTION DOC]
    # Returns the diagonal of the operator, which is simply the stored diagonal tensor.
    return self.diag

  def _add_to_tensor(self, x):
    # [FUNCTION DOC]
    # Adds this operator to a dense batch matrix x: returns a copy of x whose diagonal has been
    # incremented by this operator's diagonal. x must be broadcast-compatible; no mutation of x.
    x_diag = array_ops.matrix_diag_part(x)
    new_diag = self._diag + x_diag
    return array_ops.matrix_set_diag(x, new_diag)

  def _eigvals(self):
    # [FUNCTION DOC]
    # Returns the eigenvalues, which are the diagonal entries, converted to a tensor.
    return tensor_conversion.convert_to_tensor_v2_with_dispatch(self.diag)

  def _cond(self):
    # [FUNCTION DOC]
    # Returns the condition number per batch member: largest absolute diagonal entry divided by
    # the smallest. A zero smallest entry yields infinity/NaN rather than an error.
    abs_diag = math_ops.abs(self.diag)
    return (math_ops.reduce_max(abs_diag, axis=-1) /
            math_ops.reduce_min(abs_diag, axis=-1))

  @property
  def _composite_tensor_fields(self):
    # [FUNCTION DOC]
    # Property naming the constructor arguments that are tensor components of the composite
    # tensor: just the diagonal.
    return ("diag",)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    # [FUNCTION DOC]
    # Property mapping each tensor parameter to the number of its trailing dimensions that belong
    # to the matrix rather than the batch: the diagonal has one.
    return {"diag": 1}
