# [MODULE DOC]
# Defines LinearOperatorZeros, a linear operator acting like a (batch of) zero matrices of a given
# number of rows and columns, without materialising them. Products, determinants, traces,
# diagonals and eigenvalues are produced as freshly built zero tensors, possibly broadcast to the
# operator batch shape; solves, inverses and positive-definiteness assertions are unsupported.
# Exported publicly and registered as a composite tensor. No other import-time side effects.
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
"""`LinearOperator` acting like a zero matrix."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = [
    "LinearOperatorZeros",
]


@tf_export("linalg.LinearOperatorZeros")
@linear_operator.make_composite_tensor
class LinearOperatorZeros(linear_operator.LinearOperator):
  # [CLASS DOC]
  # Linear operator representing a (possibly batched, possibly non-square) all-zero matrix.
  # Holds the number of rows and columns (tensors plus statically known values), an optional
  # batch shape tensor, and a flag that adds run-time shape assertions. The operator is always
  # singular, self-adjoint when square and never positive definite; the constructor rejects hints
  # that contradict this. Matmul returns zeros of the right shape, broadcasting batch dimensions
  # only when needed. Decorated for public export and composite-tensor support.
  """`LinearOperator` acting like a [batch] zero matrix.

  This operator acts like a [batch] zero matrix `A` with shape
  `[B1,...,Bb, N, M]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x M` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorZeros` is initialized with `num_rows`, and optionally
  `num_columns, `batch_shape`, and `dtype` arguments.  If `num_columns` is
  `None`, then this operator will be initialized as a square matrix. If
  `batch_shape` is `None`, this operator efficiently passes through all
  arguments.  If `batch_shape` is provided, broadcasting may occur, which will
  require making copies.

  ```python
  # Create a 2 x 2 zero matrix.
  operator = LinearOperatorZero(num_rows=2, dtype=tf.float32)

  operator.to_dense()
  ==> [[0., 0.]
       [0., 0.]]

  operator.shape
  ==> [2, 2]

  operator.determinant()
  ==> 0.

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor, same as x.

  # Create a 2-batch of 2x2 zero matrices
  operator = LinearOperatorZeros(num_rows=2, batch_shape=[2])
  operator.to_dense()
  ==> [[[0., 0.]
        [0., 0.]],
       [[0., 0.]
        [0., 0.]]]

  # Here, even though the operator has a batch shape, the input is the same as
  # the output, so x can be passed through without a copy.  The operator is able
  # to detect that no broadcast is necessary because both x and the operator
  # have statically defined shape.
  x = ... Shape [2, 2, 3]
  operator.matmul(x)
  ==> Shape [2, 2, 3] Tensor, same as tf.zeros_like(x)

  # Here the operator and x have different batch_shape, and are broadcast.
  # This requires a copy, since the output is different size than the input.
  x = ... Shape [1, 2, 3]
  operator.matmul(x)
  ==> Shape [2, 2, 3] Tensor, equal to tf.zeros_like([x, x])
  ```

  ### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, M],  with b >= 0
  x.shape =   [C1,...,Cc] + [M, R],
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
               num_rows,
               num_columns=None,
               batch_shape=None,
               dtype=None,
               is_non_singular=False,
               is_self_adjoint=True,
               is_positive_definite=False,
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorZeros"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a value error is raised if any of the self-adjoint,
    # non-singular and positive-definite hints is not true; actually the error is raised when
    # is_non_singular or is_positive_definite is true (the zero matrix is singular and not
    # positive definite), or when is_self_adjoint is false while is_square is true.
    # Constructor. num_rows is a scalar non-negative integer; num_columns defaults to num_rows;
    # batch_shape is an optional one-dimensional integer tensor of leading dimensions (none by
    # default); dtype defaults to 32-bit float. The hint arguments default to the only valid
    # values for a zero matrix. assert_proper_shapes, when true, adds graph assertions when static
    # checks are inconclusive. name scopes the ops. Rejects reference-type (variable) shape
    # arguments with a type error, stores the shape tensors and their static values, validates
    # them, and raises a value error if square but the static row and column counts differ. Also
    # raises value or type errors from the shape checks (non-scalar, negative, non-integer).
    r"""Initialize a `LinearOperatorZeros`.

    The `LinearOperatorZeros` is initialized with arguments defining `dtype`
    and shape.

    This operator is able to broadcast the leading (batch) dimensions, which
    sometimes requires copying data.  If `batch_shape` is `None`, the operator
    can take arguments of any batch shape without copying.  See examples.

    Args:
      num_rows:  Scalar non-negative integer `Tensor`.  Number of rows in the
        corresponding zero matrix.
      num_columns:  Scalar non-negative integer `Tensor`.  Number of columns in
        the corresponding zero matrix. If `None`, defaults to the value of
        `num_rows`.
      batch_shape:  Optional `1-D` integer `Tensor`.  The shape of the leading
        dimensions.  If `None`, this operator has no leading dimensions.
      dtype:  Data type of the matrix that this operator represents.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      assert_proper_shapes:  Python `bool`.  If `False`, only perform static
        checks that initialization and method arguments have proper shape.
        If `True`, and static checks are inconclusive, add asserts to the graph.
      name: A name for this `LinearOperator`

    Raises:
      ValueError:  If `num_rows` is determined statically to be non-scalar, or
        negative.
      ValueError:  If `num_columns` is determined statically to be non-scalar,
        or negative.
      ValueError:  If `batch_shape` is determined statically to not be 1-D, or
        negative.
      ValueError:  If any of the following is not `True`:
        `{is_self_adjoint, is_non_singular, is_positive_definite}`.
    """
    parameters = dict(
        num_rows=num_rows,
        num_columns=num_columns,
        batch_shape=batch_shape,
        dtype=dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        assert_proper_shapes=assert_proper_shapes,
        name=name
    )

    dtype = dtype or dtypes.float32
    self._assert_proper_shapes = assert_proper_shapes

    with ops.name_scope(name):
      dtype = dtypes.as_dtype(dtype)
      if not is_self_adjoint and is_square:
        raise ValueError("A zero operator is always self adjoint.")
      if is_non_singular:
        raise ValueError("A zero operator is always singular.")
      if is_positive_definite:
        raise ValueError("A zero operator is always not positive-definite.")

      super(LinearOperatorZeros, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

      linear_operator_util.assert_not_ref_type(num_rows, "num_rows")
      linear_operator_util.assert_not_ref_type(num_columns, "num_columns")
      linear_operator_util.assert_not_ref_type(batch_shape, "batch_shape")

      self._num_rows = linear_operator_util.shape_tensor(
          num_rows, name="num_rows")
      self._num_rows_static = tensor_util.constant_value(self._num_rows)

      if num_columns is None:
        num_columns = num_rows

      self._num_columns = linear_operator_util.shape_tensor(
          num_columns, name="num_columns")
      self._num_columns_static = tensor_util.constant_value(self._num_columns)

      self._check_domain_range_possibly_add_asserts()

      if (self._num_rows_static is not None and
          self._num_columns_static is not None):
        if is_square and self._num_rows_static != self._num_columns_static:
          raise ValueError(
              "LinearOperatorZeros initialized as is_square=True, but got "
              "num_rows({}) != num_columns({})".format(
                  self._num_rows_static,
                  self._num_columns_static))

      if batch_shape is None:
        self._batch_shape_arg = None
      else:
        self._batch_shape_arg = linear_operator_util.shape_tensor(
            batch_shape, name="batch_shape_arg")
        self._batch_shape_static = tensor_util.constant_value(
            self._batch_shape_arg)
        self._check_batch_shape_possibly_add_asserts()

  def _shape(self):
    # [FUNCTION DOC]
    # Returns the static shape: the batch shape (when one was given) followed by the static row and
    # column counts, with unknown entries where not statically known.
    matrix_shape = tensor_shape.TensorShape((self._num_rows_static,
                                             self._num_columns_static))
    if self._batch_shape_arg is None:
      return matrix_shape

    batch_shape = tensor_shape.TensorShape(self._batch_shape_static)
    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # Returns the dynamic shape as a one-dimensional integer tensor: the batch shape (if given)
    # concatenated with the row and column counts.
    matrix_shape = array_ops_stack.stack(
        (self._num_rows, self._num_columns), axis=0)
    if self._batch_shape_arg is None:
      return matrix_shape

    return array_ops.concat((self._batch_shape_arg, matrix_shape), 0)

  def _assert_non_singular(self):
    # [FUNCTION DOC]
    # Always raises an invalid-argument error stating that zero operators are non-invertible,
    # since the assertion can never hold.
    raise errors.InvalidArgumentError(
        node_def=None, op=None, message="Zero operators are always "
        "non-invertible.")

  def _assert_positive_definite(self):
    # [FUNCTION DOC]
    # Always raises an invalid-argument error stating that zero operators are not positive
    # definite, since the assertion can never hold.
    raise errors.InvalidArgumentError(
        node_def=None, op=None, message="Zero operators are always "
        "non-positive definite.")

  def _assert_self_adjoint(self):
    # [FUNCTION DOC]
    # Returns a no-op: a zero matrix is trivially self-adjoint, so there is nothing to check.
    return control_flow_ops.no_op("assert_self_adjoint")

  def _possibly_broadcast_batch_shape(self, x):
    # [FUNCTION DOC]
    # HELPER: batch broadcasting for matmul and add_to_tensor.
    # Returns x unchanged if the operator has no batch shape, or if static shapes show that
    # broadcasting with the operator batch shape would not change x. Otherwise adds a tensor of
    # zeros (of operator dtype and shape batch plus two singleton dims, static or dynamic) to x so
    # that ordinary broadcasting produces the full result; the dynamic path always adds to avoid
    # a data-dependent branch. Returns the possibly broadcast tensor.
    """Return 'x', possibly after broadcasting the leading dimensions."""
    # If we have no batch shape, our batch shape broadcasts with everything!
    if self._batch_shape_arg is None:
      return x

    # Static attempt:
    #   If we determine that no broadcast is necessary, pass x through
    #   If we need a broadcast, add to an array of zeros.
    #
    # special_shape is the shape that, when broadcast with x's shape, will give
    # the correct broadcast_shape.  Note that
    #   We have already verified the second to last dimension of self.shape
    #   matches x's shape in assert_compatible_matrix_dimensions.
    #   Also, the final dimension of 'x' can have any shape.
    #   Therefore, the final two dimensions of special_shape are 1's.
    special_shape = self.batch_shape.concatenate([1, 1])
    bshape = array_ops.broadcast_static_shape(x.shape, special_shape)
    if special_shape.is_fully_defined():
      # bshape.is_fully_defined iff special_shape.is_fully_defined.
      if bshape == x.shape:
        return x
      # Use the built in broadcasting of addition.
      zeros = array_ops.zeros(shape=special_shape, dtype=self.dtype)
      return x + zeros

    # Dynamic broadcast:
    #   Always add to an array of zeros, rather than using a "cond", since a
    #   cond would require copying data from GPU --> CPU.
    special_shape = array_ops.concat((self.batch_shape_tensor(), [1, 1]), 0)
    zeros = array_ops.zeros(shape=special_shape, dtype=self.dtype)
    return x + zeros

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when assert_proper_shapes is true and adjoint_arg is true, x is replaced by
    # its adjoint for the shape assertion and then the output shape is computed as if x were
    # still un-adjointed, so the swap of the last two dimensions is applied twice.
    # Returns the product with x (adjoint flags say whether the operator and x are adjointed): a
    # zero tensor of the product's shape and x's dtype, broadcast to the operator batch shape.
    # Square case: output shape equals x's shape (last two dims swapped when adjoint_arg is
    # set). Non-square case: rows come from the column count when adjoint is set, else from the
    # row count; columns from x. Optionally adds a run-time dimension compatibility assertion.
    if self._assert_proper_shapes:
      x = linalg.adjoint(x) if adjoint_arg else x
      aps = linear_operator_util.assert_compatible_matrix_dimensions(self, x)
      x = control_flow_ops.with_dependencies([aps], x)
    if self.is_square:
      # Note that adjoint has no effect since this matrix is self-adjoint.
      if adjoint_arg:
        output_shape = array_ops.concat([
            array_ops.shape(x)[:-2],
            [array_ops.shape(x)[-1], array_ops.shape(x)[-2]]], axis=0)
      else:
        output_shape = array_ops.shape(x)

      return self._possibly_broadcast_batch_shape(
          array_ops.zeros(shape=output_shape, dtype=x.dtype))

    x_shape = array_ops.shape(x)
    n = self._num_columns if adjoint else self._num_rows
    m = x_shape[-2] if adjoint_arg else x_shape[-1]

    output_shape = array_ops.concat([x_shape[:-2], [n, m]], axis=0)

    zeros = array_ops.zeros(shape=output_shape, dtype=x.dtype)
    return self._possibly_broadcast_batch_shape(zeros)

  def _linop_matmul(
      self,
      left_operator: "LinearOperatorZeros",
      right_operator: linear_operator.LinearOperator
    ) -> linear_operator.LinearOperator:
    # [FUNCTION DOC]
    # Operator-by-operator product hook. Takes the left operator (this zeros operator) and the
    # right operator; if either is not square raises a value error (unsupported). Otherwise the
    # product with a zero matrix is again zero, so returns the left operator itself.
    if not left_operator.is_square or not right_operator.is_square:
      raise ValueError("Matmul with non-square `LinearOperator`s or non-square "
                       "`LinearOperatorZeros` not supported at this time.")
    return left_operator

  def _determinant(self):
    # [FUNCTION DOC]
    # Returns zeros of the batch shape (static if fully defined, else dynamic) and operator dtype.
    if self.batch_shape.is_fully_defined():
      return array_ops.zeros(shape=self.batch_shape, dtype=self.dtype)
    else:
      return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _trace(self):
    # [FUNCTION DOC]
    # Returns zeros of the batch shape (static if fully defined, else dynamic) and operator dtype.
    # Get Tensor of all zeros of same shape as self.batch_shape.
    if self.batch_shape.is_fully_defined():
      return array_ops.zeros(shape=self.batch_shape, dtype=self.dtype)
    else:
      return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

  def _diag_part(self):
    # [FUNCTION DOC]
    # Returns the main diagonal, which is all zeros (see the zero-diagonal helper).
    return self._zeros_diag()

  def add_to_tensor(self, mat, name="add_to_tensor"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says this is equivalent to adding the identity (I + mat), but
    # the operator is a zero matrix, so the result is just mat (broadcast to the batch shape).
    # Adds this (zero) operator to tensor mat: returns mat, broadcast over the operator batch
    # shape when needed. The name argument is accepted but not used.
    """Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.

    Args:
      mat:  `Tensor` with same `dtype` and shape broadcastable to `self`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    return self._possibly_broadcast_batch_shape(mat)

  def _check_domain_range_possibly_add_asserts(self):
    # [FUNCTION DOC]
    # HELPER: constructor validation of the row and column counts.
    # If run-time assertions are enabled, wraps the stored counts so that they must be rank-zero
    # and non-negative when used. Always performs static checks: raises a type error if either is
    # not an integer type, and a value error if a statically known value is not a scalar or is
    # negative. Mutates the stored row and column tensors when assertions are enabled.
    """Static check of init arg `num_rows`, possibly add asserts."""
    # Possibly add asserts.
    if self._assert_proper_shapes:
      self._num_rows = control_flow_ops.with_dependencies([
          check_ops.assert_rank(
              self._num_rows,
              0,
              message="Argument num_rows must be a 0-D Tensor."),
          check_ops.assert_non_negative(
              self._num_rows,
              message="Argument num_rows must be non-negative."),
      ], self._num_rows)
      self._num_columns = control_flow_ops.with_dependencies([
          check_ops.assert_rank(
              self._num_columns,
              0,
              message="Argument num_columns must be a 0-D Tensor."),
          check_ops.assert_non_negative(
              self._num_columns,
              message="Argument num_columns must be non-negative."),
      ], self._num_columns)

    # Static checks.
    if not self._num_rows.dtype.is_integer:
      raise TypeError("Argument num_rows must be integer type.  Found:"
                      " %s" % self._num_rows)

    if not self._num_columns.dtype.is_integer:
      raise TypeError("Argument num_columns must be integer type.  Found:"
                      " %s" % self._num_columns)

    num_rows_static = self._num_rows_static
    num_columns_static = self._num_columns_static

    if num_rows_static is not None:
      if num_rows_static.ndim != 0:
        raise ValueError("Argument num_rows must be a 0-D Tensor.  Found:"
                         " %s" % num_rows_static)

      if num_rows_static < 0:
        raise ValueError("Argument num_rows must be non-negative.  Found:"
                         " %s" % num_rows_static)
    if num_columns_static is not None:
      if num_columns_static.ndim != 0:
        raise ValueError("Argument num_columns must be a 0-D Tensor.  Found:"
                         " %s" % num_columns_static)

      if num_columns_static < 0:
        raise ValueError("Argument num_columns must be non-negative.  Found:"
                         " %s" % num_columns_static)

  def _check_batch_shape_possibly_add_asserts(self):
    # [FUNCTION DOC]
    # HELPER: constructor validation of the batch shape.
    # Does nothing when there is no batch shape. If run-time assertions are enabled, wraps the
    # stored batch shape so it must be one-dimensional and non-negative when used. Static checks
    # raise a type error for a non-integer dtype and a value error if the static value is not
    # one-dimensional or has negative entries. Mutates the stored batch shape tensor.
    """Static check of init arg `batch_shape`, possibly add asserts."""
    if self._batch_shape_arg is None:
      return

    # Possibly add asserts
    if self._assert_proper_shapes:
      self._batch_shape_arg = control_flow_ops.with_dependencies([
          check_ops.assert_rank(
              self._batch_shape_arg,
              1,
              message="Argument batch_shape must be a 1-D Tensor."),
          check_ops.assert_non_negative(
              self._batch_shape_arg,
              message="Argument batch_shape must be non-negative."),
      ], self._batch_shape_arg)

    # Static checks
    if not self._batch_shape_arg.dtype.is_integer:
      raise TypeError("Argument batch_shape must be integer type.  Found:"
                      " %s" % self._batch_shape_arg)

    if self._batch_shape_static is None:
      return  # Cannot do any other static checks.

    if self._batch_shape_static.ndim != 1:
      raise ValueError("Argument batch_shape must be a 1-D Tensor.  Found:"
                       " %s" % self._batch_shape_static)

    if np.any(self._batch_shape_static < 0):
      raise ValueError("Argument batch_shape must be non-negative.  Found:"
                       "%s" % self._batch_shape_static)

  def _min_matrix_dim(self):
    # [FUNCTION DOC]
    # HELPER: diagonal length computation. Returns the smaller of the static domain and range
    # dimensions, or null if either is unknown.
    """Minimum of domain/range dimension, if statically available, else None."""
    domain_dim = self.domain_dimension.value
    range_dim = self.range_dimension.value
    if domain_dim is None or range_dim is None:
      return None
    return min(domain_dim, range_dim)

  def _min_matrix_dim_tensor(self):
    # [FUNCTION DOC]
    # HELPER: diagonal length computation. Returns, as a scalar tensor, the smaller of the last two
    # entries of the dynamic shape.
    """Minimum of domain/range dimension, as a tensor."""
    return math_ops.reduce_min(self.shape_tensor()[-2:])

  def _zeros_diag(self):
    # [FUNCTION DOC]
    # HELPER: shared by diagonal and eigenvalue methods.
    # Returns a zero tensor of the operator dtype whose shape is the batch shape followed by the
    # smaller matrix dimension (static when the full shape is known, otherwise dynamic).
    """Returns the diagonal of this operator as all zeros."""
    if self.shape.is_fully_defined():
      d_shape = self.batch_shape.concatenate([self._min_matrix_dim()])
    else:
      d_shape = array_ops.concat(
          [self.batch_shape_tensor(),
           [self._min_matrix_dim_tensor()]], axis=0)

    return array_ops.zeros(shape=d_shape, dtype=self.dtype)

  def _eigvals(self):
    # [FUNCTION DOC]
    # Returns the eigenvalues, all zeros, in the same form as the zero diagonal.
    return self._zeros_diag()

  @property
  def _composite_tensor_prefer_static_fields(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor support hook.
    # Property naming the constructor arguments (rows, columns, batch shape) whose static values
    # are preferred over tensors when flattening the operator.
    return ("num_rows", "num_columns", "batch_shape")

  @property
  def _composite_tensor_fields(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor support hook.
    # Property naming the constructor arguments (rows, columns, batch shape, dtype and the
    # assertion flag) that make up the composite tensor representation.
    return ("num_rows", "num_columns", "batch_shape", "dtype",
            "assert_proper_shapes")

  def __getitem__(self, slices):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment says a new identity operator is returned, but a new zeros
    # operator is built. Also, if the operator has no batch shape the stored batch shape is null,
    # so slicing fails when building the proxy tensor.
    # Slices the batch dimensions. Builds a ones tensor of the batch shape, slices it with the
    # given slices, takes the resulting shape as the new batch shape, and returns a new zeros
    # operator constructed from the original parameters with that batch shape. Does not modify
    # this operator.
    # Slice the batch shape and return a new LinearOperatorIdentity.
    # Use a proxy shape and slice it. Use this as the new batch shape
    new_batch_shape = array_ops.shape(
        array_ops.ones(self._batch_shape_arg)[slices])
    parameters = dict(self.parameters, batch_shape=new_batch_shape)
    return LinearOperatorZeros(**parameters)

