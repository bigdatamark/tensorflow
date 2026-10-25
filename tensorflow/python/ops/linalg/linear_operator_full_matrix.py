# [MODULE DOC]
# Defines LinearOperatorFullMatrix, a LinearOperator wrapping a dense [batch] matrix tensor.
# Exposes only that class (exported publicly via the linalg namespace and made a composite
# tensor at import time). Builds on the base operator and utility modules of this package.
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
"""`LinearOperator` that wraps a [batch] matrix."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorFullMatrix"]


@tf_export("linalg.LinearOperatorFullMatrix")
@linear_operator.make_composite_tensor
class LinearOperatorFullMatrix(linear_operator.LinearOperator):
  # [CLASS DOC]
  # Linear operator backed by an explicit dense matrix tensor of shape batch dims plus [M, N].
  # Holds the matrix in a private attribute; matmul, solve and to_dense delegate to standard
  # TensorFlow matrix ops. Created with a matrix (any tensor-like of allowed float/complex
  # dtype) and optional is_* hints. Subclasses LinearOperator and is registered as a composite
  # tensor; exported as the public linear algebra full-matrix operator.
  """`LinearOperator` that wraps a [batch] matrix.

  This operator wraps a [batch] matrix `A` (which is a `Tensor`) with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `M x N` matrix.

  ```python
  # Create a 2 x 2 linear operator.
  matrix = [[1., 2.], [3., 4.]]
  operator = LinearOperatorFullMatrix(matrix)

  operator.to_dense()
  ==> [[1., 2.]
       [3., 4.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 4 linear operators.
  matrix = tf.random.normal(shape=[2, 3, 4, 4])
  operator = LinearOperatorFullMatrix(matrix)
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  with b >= 0
  x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
  ```

  #### Performance

  `LinearOperatorFullMatrix` has exactly the same performance as would be
  achieved by using standard `TensorFlow` matrix ops.  Intelligent choices are
  made based on the following initialization hints.

  * If `dtype` is real, and `is_self_adjoint` and `is_positive_definite`, a
    Cholesky factorization is used for the determinant and solve.

  In all cases, suppose `operator` is a `LinearOperatorFullMatrix` of shape
  `[M, N]`, and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` is `O(M * N * R)`.
  * If `M=N`, `operator.solve(x)` is `O(N^3 * R)`.
  * If `M=N`, `operator.determinant()` is `O(N^3)`.

  If instead `operator` and `x` have shape `[B1,...,Bb, M, N]` and
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
               matrix,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorFullMatrix"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the error mentions `diag.dtype`; the checked argument is
    # actually the matrix argument (a ValueError for rank below 2 is also possible, undocumented).
    # Constructor. Records the constructor arguments in a parameters mapping, converts matrix to a
    # tensor (non-reference values only) inside a name scope, validates it with the matrix check
    # (raising a type error for a disallowed dtype or a value error for static rank below 2), and
    # calls the base class constructor with the matrix dtype, the is_* hints (each true, false or
    # nothing), the parameters and the name. Stores the tensor as the private matrix attribute.
    r"""Initialize a `LinearOperatorFullMatrix`.

    Args:
      matrix:  Shape `[B1,...,Bb, M, N]` with `b >= 0`, `M, N >= 0`.
        Allowed dtypes: `float16`, `float32`, `float64`, `complex64`,
        `complex128`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If `diag.dtype` is not an allowed type.
    """
    parameters = dict(
        matrix=matrix,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    with ops.name_scope(name, values=[matrix]):
      self._matrix = linear_operator_util.convert_nonref_to_tensor(
          matrix, name="matrix")
      self._check_matrix(self._matrix)

      super(LinearOperatorFullMatrix, self).__init__(
          dtype=self._matrix.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _check_matrix(self, matrix):
    # [FUNCTION DOC]
    # HELPER: argument validation for the constructor of LinearOperatorFullMatrix.
    # Converts the matrix argument to a tensor (local only) and checks its dtype and rank.
    # Raises a type error if the dtype is not one of float16, float32, float64, complex64 or
    # complex128, and a value error if the static rank is known and below 2. Returns nothing.
    """Static check of the `matrix` argument."""
    allowed_dtypes = [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.complex64,
        dtypes.complex128,
    ]

    matrix = tensor_conversion.convert_to_tensor_v2_with_dispatch(
        matrix, name="matrix"
    )

    dtype = matrix.dtype
    if dtype not in allowed_dtypes:
      raise TypeError(f"Argument `matrix` must have dtype in {allowed_dtypes}. "
                      f"Received: {dtype}.")

    if matrix.shape.ndims is not None and matrix.shape.ndims < 2:
      raise ValueError(f"Argument `matrix` must have at least 2 dimensions. "
                       f"Received: {matrix}.")

  @property
  def matrix(self):
    # [FUNCTION DOC]
    # Read-only property returning the matrix tensor (possibly batched) that defines this
    # operator, as stored at construction time after conversion to a tensor.
    """The matrix defining this operator."""
    return self._matrix

  def _shape(self):
    # [FUNCTION DOC]
    # HELPER: implements the static-shape hook of the LinearOperator base class.
    # Returns the static (possibly partially unknown) shape of the wrapped matrix.
    return self._matrix.shape

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: implements the dynamic-shape hook of the LinearOperator base class.
    # Returns the runtime shape of the wrapped matrix as an integer tensor.
    return array_ops.shape(self._matrix)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # HELPER: implements the matmul hook of the LinearOperator base class.
    # Multiplies the wrapped matrix by the tensor x using the standard matrix multiply op.
    # adjoint takes the conjugate transpose of the matrix first; adjoint_arg does so for x.
    # Returns the product tensor.
    return math_ops.matmul(
        self._matrix, x, adjoint_a=adjoint, adjoint_b=adjoint_arg)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # HELPER: implements the solve hook of the LinearOperator base class.
    # Solves the linear system for the right-hand side rhs by delegating to the base class dense
    # solver (Cholesky when hints allow, otherwise a general solve). adjoint solves with the
    # adjoint of the operator; adjoint_arg uses the adjoint of rhs. Returns the solution tensor.
    return self._dense_solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _to_dense(self):
    # [FUNCTION DOC]
    # HELPER: implements the dense-conversion hook of the LinearOperator base class.
    # Returns the wrapped matrix tensor itself, with no copy or computation.
    return self._matrix

  @property
  def _composite_tensor_fields(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor support for LinearOperatorFullMatrix.
    # Property returning the tuple of constructor parameter names that are treated as tensor
    # components when this operator is converted to a composite tensor: just the matrix.
    return ("matrix",)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor / batching support for LinearOperatorFullMatrix.
    # Property returning a mapping from constructor parameter names to the number of trailing
    # dimensions that form the matrix part of that parameter. Here the only entry is the
    # matrix parameter, which has two matrix dimensions. Used to compute batch shapes.
    return {"matrix": 2}
