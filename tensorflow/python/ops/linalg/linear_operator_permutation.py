# [MODULE DOC]
# Defines LinearOperatorPermutation, a linear operator that permutes the rows of its input
# according to an integer vector (or batch of vectors), without materializing the matrix.
# Exposes only that class (exported publicly). Uses gather and argsort for products and solves.
# No import-time side effects besides export registration and composite tensor wrapping.
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
"""`LinearOperator` acting like a permutation matrix."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorPermutation",]


@tf_export("linalg.LinearOperatorPermutation")
@linear_operator.make_composite_tensor
class LinearOperatorPermutation(linear_operator.LinearOperator):
  # [CLASS DOC]
  # Operator acting like a [batch] of N x N permutation matrices defined by an integer vector perm
  # of unique values from zero to N-1; applying it moves row perm[i] of the input to row i.
  # Holds the permutation tensor and an explicit dtype. Always square and non-singular; the
  # determinant magnitude is one (log-abs-determinant zero), condition number one, solves are
  # products with the inverse (transposed) permutation.
  """`LinearOperator` acting like a [batch] of permutation matrices.

  This operator acts like a [batch] of permutations with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorPermutation` is initialized with a (batch) vector.

  A permutation, is defined by an integer vector `v` whose values are unique
  and are in the range `[0, ... n]`. Applying the permutation on an input
  matrix has the folllowing meaning: the value of `v` at index `i`
  says to move the `v[i]`-th row of the input matrix to the `i`-th row.
  Because all values are unique, this will result in a permutation of the
  rows the input matrix. Note, that the permutation vector `v` has the same
  semantics as `tf.transpose`.

  ```python
  # Create a 3 x 3 permutation matrix that swaps the last two columns.
  vec = [0, 2, 1]
  operator = LinearOperatorPermutation(vec)

  operator.to_dense()
  ==> [[1., 0., 0.]
       [0., 0., 1.]
       [0., 1., 0.]]

  operator.shape
  ==> [3, 3]

  # This will be zero.
  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [3, 4] Tensor
  operator.matmul(x)
  ==> Shape [3, 4] Tensor
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
               perm,
               dtype=dtypes.float32,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorPermutation"):
    # [FUNCTION DOC]
    # Constructor. Inputs: perm (integer tensor of shape batch + [N] with unique entries), dtype
    # (floating or complex dtype of the operator, default 32-bit float), property hints and a
    # name. Converts perm to a tensor in a name scope, validates it, rejects explicit non-singular
    # or non-square false hints, forces is_square to true, and calls the parent constructor.
    # Raises a value error if is_non_singular or is_square is explicitly false, plus the errors of
    # the perm check.
    # KNOWN ISSUE: the docstring says is_self_adjoint is auto-set to true and is_positive_definite
    # to false and lists value errors for wrong values, but the code does neither: both hints are
    # passed through unchanged and no error is raised for them. Also the unqualified use of the
    # self-adjoint hint in matmul means a wrongly claimed self-adjoint hint gives wrong adjoints.
    r"""Initialize a `LinearOperatorPermutation`.

    Args:
      perm:  Shape `[B1,...,Bb, N]` Integer `Tensor` with `b >= 0`
        `N >= 0`. An integer vector that represents the permutation to apply.
        Note that this argument is same as `tf.transpose`. However, this
        permutation is applied on the rows, while the permutation in
        `tf.transpose` is applied on the dimensions of the `Tensor`. `perm`
        is required to have unique entries from `{0, 1, ... N-1}`.
      dtype: The `dtype` of arguments to this operator. Default: `float32`.
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
        perm=perm,
        dtype=dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    with ops.name_scope(name, values=[perm]):
      self._perm = linear_operator_util.convert_nonref_to_tensor(
          perm, name="perm")
      self._check_perm(self._perm)

      # Check and auto-set hints.
      if is_non_singular is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError(f"A Permutation operator is always non-singular. "
                         f"Expected argument `is_non_singular` to be True. "
                         f"Received: {is_non_singular}.")

      if is_square is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError(f"A Permutation operator is always square. "
                         f"Expected argument `is_square` to be True. "
                         f"Received: {is_square}.")
      is_square = True

      super(LinearOperatorPermutation, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  def _check_perm(self, perm):
    # [FUNCTION DOC]
    # HELPER: Argument validation used by the constructor.
    # Statically checks perm: raises a value error if its rank is known and below one; raises a
    # type error if its dtype is not integer; and, when the values are statically known, raises a
    # value error unless the sorted entries equal zero to N-1 (unique, in range). Dynamic values
    # are not checked. Returns nothing.
    """Static check of perm."""
    if (perm.shape.ndims is not None and perm.shape.ndims < 1):
      raise ValueError(f"Argument `perm` must have at least 1 dimension. "
                       f"Received: {perm}.")
    if not perm.dtype.is_integer:
      raise TypeError(f"Argument `perm` must be integer dtype. "
                      f"Received: {perm}.")
    # Check that the permutation satisfies the uniqueness constraint.
    static_perm = tensor_util.constant_value(perm)
    if static_perm is not None:
      sorted_perm = np.sort(static_perm, axis=-1)
      if np.any(sorted_perm != np.arange(0, static_perm.shape[-1])):
        raise ValueError(
            f"Argument `perm` must be a vector of unique integers from "
            f"0 to {static_perm.shape[-1] - 1}.")

  def _shape(self):
    # [FUNCTION DOC]
    # Returns the static shape: perm's shape with its last dimension repeated, i.e. batch + [N, N].
    perm_shape = self._perm.shape
    return perm_shape.concatenate(perm_shape[-1:])

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # Returns the dynamic shape as an integer tensor: perm's shape with its last dimension
    # appended again.
    perm_shape = array_ops.shape(self._perm)
    k = perm_shape[-1]
    return array_ops.concat((perm_shape, [k]), 0)

  def _assert_non_singular(self):
    # [FUNCTION DOC]
    # Assertion that the operator is non-singular. Permutations are always invertible, so this
    # returns a no-op and never fails.
    return control_flow_ops.no_op("assert_non_singular")

  def _domain_dimension_tensor(self, perm=None):
    # [FUNCTION DOC]
    # HELPER: Computes the number of columns/rows N as a dynamic scalar tensor, the last
    # dimension of the given permutation, or of the stored one if none is passed.
    # Input: perm (optional tensor). Returns an integer scalar tensor.
    perm = perm if perm is not None else self.perm
    return array_ops.shape(perm)[-1]

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Applies the permutation (or its inverse when adjoint is requested) to x (or to the adjoint
    # of x when adjoint_arg is true). For the adjoint the permutation is inverted with an argsort
    # unless the operator is flagged self-adjoint. Broadcasts x and perm to a common batch shape
    # because gather does not broadcast, flattens batches, gathers rows per batch, and reshapes
    # back. Inputs: x tensor, adjoint and adjoint_arg booleans (default False). Returns a tensor
    # with the broadcast batch shape.
    perm = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.perm)
    if adjoint and not self.is_self_adjoint:
      # TODO(srvasude): invert_permutation doesn't work on batches so we use
      # argsort.
      perm = sort_ops.argsort(perm, axis=-1)
    x = linalg.adjoint(x) if adjoint_arg else x

    # We need to broadcast x and the permutation since tf.gather doesn't
    # broadcast.
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(x)[:-1], array_ops.shape(perm))
    k = array_ops.shape(x)[-1]
    broadcast_x_shape = array_ops.concat([broadcast_shape, [k]], axis=-1)
    x = array_ops.broadcast_to(x, broadcast_x_shape)
    perm = array_ops.broadcast_to(perm, broadcast_shape)

    m = array_ops.shape(x)[-2]
    x = array_ops.reshape(x, [-1, m, k])
    perm = array_ops.reshape(perm, [-1, m])

    y = array_ops.gather(x, perm, axis=-2, batch_dims=1)
    return array_ops.reshape(y, broadcast_x_shape)

  # TODO(srvasude): Permutation parity is equivalent to the determinant.

  def _log_abs_determinant(self):
    # [FUNCTION DOC]
    # Returns zeros of the batch shape and operator dtype, since permutation determinants are
    # plus or minus one so the log of the absolute value is zero.
    # Permutation matrices have determinant +/- 1.
    return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Solves against a permutation: the inverse equals the transpose, so it delegates to the
    # matmul with the adjoint flag flipped. Inputs: rhs, adjoint and adjoint_arg (default False).
    # Returns the solution tensor.
    # The inverse of a permutation matrix is the transpose matrix.
    # Apply a matmul and flip the adjoint bit.
    return self._matmul(rhs, adjoint=(not adjoint), adjoint_arg=adjoint_arg)

  def _to_dense(self):
    # [FUNCTION DOC]
    # Builds the dense matrix by comparing a range of column indices with each perm entry (one
    # hot encoding of perm) and casting to the operator dtype. Returns a tensor of shape
    # batch + [N, N].
    perm = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.perm)
    return math_ops.cast(math_ops.equal(
        math_ops.range(0, self._domain_dimension_tensor(perm)),
        perm[..., array_ops.newaxis]), self.dtype)

  def _diag_part(self):
    # [FUNCTION DOC]
    # Returns the main diagonal: one where the perm entry equals its own index (fixed points),
    # zero elsewhere, cast to the operator dtype. Shape is batch + [N].
    perm = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.perm)
    return math_ops.cast(math_ops.equal(
        math_ops.range(0, self._domain_dimension_tensor(perm)),
        perm), self.dtype)

  def _cond(self):
    # [FUNCTION DOC]
    # Returns the condition number, always one (ones of batch shape in operator dtype).
    # Permutation matrices are rotations which have condition number 1.
    return array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)

  @property
  def perm(self):
    # [FUNCTION DOC]
    # Read-only property returning the permutation tensor.
    return self._perm

  @property
  def _composite_tensor_fields(self):
    # [FUNCTION DOC]
    # Property giving the argument names stored for composite tensor conversion: perm and dtype.
    return ("perm", "dtype")

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    # [FUNCTION DOC]
    # Property mapping perm to one matrix dimension for batch slicing in composite tensor handling.
    return {"perm": 1}
