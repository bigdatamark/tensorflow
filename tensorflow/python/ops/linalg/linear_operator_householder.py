# [MODULE DOC]
# Defines LinearOperatorHouseholder, an operator acting like a (batch of) Householder
# reflection defined by a single vector. Exposes only that class (exported publicly via the
# linalg namespace and made a composite tensor at import time). Builds on the base operator
# and utility modules of this package.
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
"""`LinearOperator` acting like a Householder transformation."""
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorHouseholder",]


@tf_export("linalg.LinearOperatorHouseholder")
@linear_operator.make_composite_tensor
class LinearOperatorHouseholder(linear_operator.LinearOperator):
  # [CLASS DOC]
  # Operator acting like a (batch of) Householder reflection matrices of shape batch + [N, N],
  # defined only by a reflection-axis vector and never materialized unless to_dense is called.
  # Holds the axis in a private attribute. It is always self-adjoint, square, not positive
  # definite, has determinant minus one and condition number one; matmul and solve are
  # computed from the axis directly. Created with the axis and optional hints; registered as a
  # composite tensor and exported publicly.
  """`LinearOperator` acting like a [batch] of Householder transformations.

  This operator acts like a [batch] of householder reflections with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorHouseholder` is initialized with a (batch) vector.

  A Householder reflection, defined via a vector `v`, which reflects points
  in `R^n` about the hyperplane orthogonal to `v` and through the origin.

  ```python
  # Create a 2 x 2 householder transform.
  vec = [1 / np.sqrt(2), 1. / np.sqrt(2)]
  operator = LinearOperatorHouseholder(vec)

  operator.to_dense()
  ==> [[0.,  -1.]
       [-1., -0.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

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
               reflection_axis,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorHouseholder"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says is_self_adjoint must be True and is_positive_definite and
    # is_square are autoset, but the code only raises when is_self_adjoint is explicitly false,
    # is_positive_definite explicitly true or is_square explicitly false; other values are
    # silently overwritten. The reflection axis dtype is also not validated despite the docstring.
    # Constructor. Takes reflection_axis (vector or batch of vectors, float or complex), the
    # optional is_* hints and a name. Converts the axis to a tensor (non-reference values only)
    # in a name scope, validates its rank, then forces self-adjoint true, positive-definite
    # false and square true. Raises a value error for contradicting hints. Calls the base class
    # constructor with the axis dtype and the parameter mapping, and stores the axis privately.
    r"""Initialize a `LinearOperatorHouseholder`.

    Args:
      reflection_axis:  Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.
        The vector defining the hyperplane to reflect about.
        Allowed dtypes: `float16`, `float32`, `float64`, `complex64`,
        `complex128`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  This is autoset to true
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
        This is autoset to false.
      is_square:  Expect that this operator acts like square [batch] matrices.
        This is autoset to true.
      name: A name for this `LinearOperator`.

    Raises:
      ValueError:  `is_self_adjoint` is not `True`, `is_positive_definite` is
        not `False` or `is_square` is not `True`.
    """
    parameters = dict(
        reflection_axis=reflection_axis,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    with ops.name_scope(name, values=[reflection_axis]):
      self._reflection_axis = linear_operator_util.convert_nonref_to_tensor(
          reflection_axis, name="reflection_axis")
      self._check_reflection_axis(self._reflection_axis)

      # Check and auto-set hints.
      if is_self_adjoint is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError("A Householder operator is always self adjoint.")
      else:
        is_self_adjoint = True

      if is_positive_definite is True:  # pylint:disable=g-bool-id-comparison
        raise ValueError(
            "A Householder operator is always non-positive definite.")
      else:
        is_positive_definite = False

      if is_square is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError("A Householder operator is always square.")
      is_square = True

      super(LinearOperatorHouseholder, self).__init__(
          dtype=self._reflection_axis.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _check_reflection_axis(self, reflection_axis):
    # [FUNCTION DOC]
    # HELPER: argument validation for the constructor of LinearOperatorHouseholder.
    # Raises a value error if the static rank of the reflection axis is known and below 1.
    # Returns nothing.
    """Static check of reflection_axis."""
    if (reflection_axis.shape.ndims is not None and
        reflection_axis.shape.ndims < 1):
      raise ValueError(
          "Argument reflection_axis must have at least 1 dimension.  "
          "Found: %s" % reflection_axis)

  def _shape(self):
    # [FUNCTION DOC]
    # HELPER: implements the static-shape hook of the LinearOperator base class.
    # Returns the static shape of the reflection axis with its last dimension repeated, giving
    # batch dimensions followed by [N, N].
    # If d_shape = [5, 3], we return [5, 3, 3].
    d_shape = self._reflection_axis.shape
    return d_shape.concatenate(d_shape[-1:])

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: implements the dynamic-shape hook of the LinearOperator base class.
    # Returns an integer tensor: the runtime shape of the reflection axis with its last
    # dimension appended again, i.e. batch dimensions followed by [N, N].
    d_shape = array_ops.shape(self._reflection_axis)
    k = d_shape[-1]
    return array_ops.concat((d_shape, [k]), 0)

  def _assert_non_singular(self):
    # [FUNCTION DOC]
    # HELPER: implements the non-singular assertion hook of the LinearOperator base class.
    # A reflection is always invertible, so returns a no-op graph operation named for the
    # assertion; nothing is checked at runtime.
    return control_flow_ops.no_op("assert_non_singular")

  def _assert_positive_definite(self):
    # [FUNCTION DOC]
    # HELPER: implements the positive-definite assertion hook of the LinearOperator base class.
    # Always raises an invalid-argument error stating that Householder operators are never
    # positive definite (they have a negative eigenvalue). Takes no inputs and returns nothing.
    raise errors.InvalidArgumentError(
        node_def=None, op=None, message="Householder operators are always "
        "non-positive definite.")

  def _assert_self_adjoint(self):
    # [FUNCTION DOC]
    # HELPER: implements the self-adjoint assertion hook of the LinearOperator base class.
    # Always true for a reflection, so returns a no-op graph operation named for the assertion.
    return control_flow_ops.no_op("assert_self_adjoint")

  def _linop_adjoint(self) -> "LinearOperatorHouseholder":
    # [FUNCTION DOC]
    # HELPER: implements the adjoint-operator hook of the LinearOperator base class.
    # A reflection is self-adjoint, so returns this same operator.
    return self

  def _linop_inverse(self) -> "LinearOperatorHouseholder":
    # [FUNCTION DOC]
    # HELPER: implements the inverse-operator hook of the LinearOperator base class.
    # A reflection is its own inverse, so returns this same operator.
    return self

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # HELPER: implements the matmul hook of the LinearOperator base class.
    # Applies the reflection to x without materializing the matrix: x minus twice the projection
    # of x onto the unit-normalized reflection axis. If adjoint_arg is true, x is conjugate
    # transposed first. The adjoint flag is ignored because the operator is self-adjoint.
    # Returns a tensor shaped like the product.
    # Given a vector `v`, we would like to reflect `x` about the hyperplane
    # orthogonal to `v` going through the origin.  We first project `x` to `v`
    # to get v * dot(v, x) / dot(v, v).  After we project, we can reflect the
    # projection about the hyperplane by flipping sign to get
    # -v * dot(v, x) / dot(v, v).  Finally, we can add back the component
    # that is orthogonal to v. This is invariant under reflection, since the
    # whole hyperplane is invariant. This component is equal to x - v * dot(v,
    # x) / dot(v, v), giving the formula x - 2 * v * dot(v, x) / dot(v, v)
    # for the reflection.

    # Note that because this is a reflection, it lies in O(n) (for real vector
    # spaces) or U(n) (for complex vector spaces), and thus is its own adjoint.
    reflection_axis = tensor_conversion.convert_to_tensor_v2_with_dispatch(
        self.reflection_axis
    )
    x = linalg.adjoint(x) if adjoint_arg else x
    normalized_axis = nn.l2_normalize(reflection_axis, axis=-1)
    mat = normalized_axis[..., array_ops.newaxis]
    x_dot_normalized_v = math_ops.matmul(mat, x, adjoint_a=True)

    return x - 2 * mat * x_dot_normalized_v

  def _trace(self):
    # [FUNCTION DOC]
    # HELPER: implements the trace hook of the LinearOperator base class.
    # Returns, for every batch member, N - 2 (N-1 eigenvalues of one plus one eigenvalue of minus
    # one) as a tensor of batch shape in the operator dtype.
    # We have (n - 1) +1 eigenvalues and a single -1 eigenvalue.
    shape = self.shape_tensor()
    return math_ops.cast(
        self._domain_dimension_tensor(shape=shape) - 2,
        self.dtype) * array_ops.ones(
            shape=self._batch_shape_tensor(shape=shape), dtype=self.dtype)

  def _determinant(self):
    # [FUNCTION DOC]
    # HELPER: implements the determinant hook of the LinearOperator base class.
    # Returns a tensor of minus ones with batch shape and operator dtype, because every
    # Householder reflection has determinant minus one.
    # For householder transformations, the determinant is -1.
    return -array_ops.ones(shape=self.batch_shape_tensor(), dtype=self.dtype)  # pylint: disable=invalid-unary-operand-type

  def _log_abs_determinant(self):
    # [FUNCTION DOC]
    # HELPER: implements the log-determinant hook of the LinearOperator base class.
    # Returns zeros of batch shape and operator dtype, since the matrix is orthogonal/unitary.
    # Orthogonal matrix -> log|Q| = 0.
    return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # HELPER: implements the solve hook of the LinearOperator base class.
    # Because a reflection is its own inverse and self-adjoint, solving is the same as applying
    # the operator: forwards rhs, adjoint and adjoint_arg to the matmul implementation and
    # returns that result.
    # A householder reflection is a reflection, hence is idempotent. Thus we
    # can just apply a matmul.
    return self._matmul(rhs, adjoint, adjoint_arg)

  def _to_dense(self):
    # [FUNCTION DOC]
    # HELPER: implements the dense-conversion hook of the LinearOperator base class.
    # Materializes the reflection matrix: the identity minus twice the outer product of the
    # unit-normalized reflection axis with its conjugate. Returns a tensor of shape batch + [N, N].
    reflection_axis = tensor_conversion.convert_to_tensor_v2_with_dispatch(
        self.reflection_axis
    )
    normalized_axis = nn.l2_normalize(reflection_axis, axis=-1)
    mat = normalized_axis[..., array_ops.newaxis]
    matrix = -2 * math_ops.matmul(mat, mat, adjoint_b=True)
    return array_ops.matrix_set_diag(
        matrix, 1. + array_ops.matrix_diag_part(matrix))

  def _diag_part(self):
    # [FUNCTION DOC]
    # HELPER: implements the diagonal hook of the LinearOperator base class.
    # Normalizes the reflection axis to unit length and returns one minus twice the squared
    # magnitude of each component, which is the diagonal of the reflection matrix (batched).
    reflection_axis = tensor_conversion.convert_to_tensor_v2_with_dispatch(
        self.reflection_axis
    )
    normalized_axis = nn.l2_normalize(reflection_axis, axis=-1)
    return 1. - 2 * normalized_axis * math_ops.conj(normalized_axis)

  def _eigvals(self):
    # [FUNCTION DOC]
    # HELPER: implements the eigenvalue hook of the LinearOperator base class.
    # Returns a tensor with shape equal to the reflection axis shape: a single eigenvalue of
    # minus one followed by (N - 1) eigenvalues of plus one, for every batch member, in the
    # operator dtype.
    # We have (n - 1) +1 eigenvalues and a single -1 eigenvalue.
    result_shape = array_ops.shape(self.reflection_axis)
    n = result_shape[-1]
    ones_shape = array_ops.concat([result_shape[:-1], [n - 1]], axis=-1)
    neg_shape = array_ops.concat([result_shape[:-1], [1]], axis=-1)
    eigvals = array_ops.ones(shape=ones_shape, dtype=self.dtype)
    eigvals = array_ops.concat(
        [-array_ops.ones(shape=neg_shape, dtype=self.dtype), eigvals], axis=-1)  # pylint: disable=invalid-unary-operand-type
    return eigvals

  def _cond(self):
    # [FUNCTION DOC]
    # HELPER: implements the condition-number hook of the LinearOperator base class.
    # Returns a tensor of ones with the batch shape and operator dtype, since a Householder
    # reflection is orthogonal/unitary and has condition number one.
    # Householder matrices are rotations which have condition number 1.
    return array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)

  @property
  def reflection_axis(self):
    # [FUNCTION DOC]
    # Read-only property returning the vector (possibly batched) that defines the hyperplane
    # of reflection, as stored at construction (converted to a tensor unless it is a reference).
    return self._reflection_axis

  @property
  def _composite_tensor_fields(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor support for LinearOperatorHouseholder.
    # Property returning the tuple of parameter names treated as tensor components: only the
    # reflection axis.
    return ("reflection_axis",)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor / batching support for LinearOperatorHouseholder.
    # Property mapping the reflection_axis constructor parameter to 1, the number of trailing
    # dimensions of that parameter that belong to the matrix part (the vector length).
    return {"reflection_axis": 1}
