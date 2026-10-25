# [MODULE DOC]
# Defines LinearOperatorBlockDiag, a linear operator whose matrix is the direct sum (block
# diagonal arrangement) of one or more component linear operators, with zeros off the blocks.
# Exposes only that class (exported publicly as the linalg block-diagonal operator) and is
# listed in the module's export list. Most operations (matmul, solve, determinant, trace,
# diagonal part, eigenvalues, dense conversion, assertions) are delegated to each block and the
# per-block results are combined, so cost is the sum of per-block costs. Arguments to the
# matrix-product and solve methods may be plain tensors or lists of per-block tensors.
# Import-time side effects: the class is registered with the public API export decorator and
# is wrapped to become a composite-tensor type. Not a helper module; it is one of the
# concrete operators in the linear operator family next to the block lower triangular one.
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
"""Create a Block Diagonal operator from one or more `LinearOperators`."""

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import property_hint_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorBlockDiag"]


@tf_export("linalg.LinearOperatorBlockDiag")
@linear_operator.make_composite_tensor
class LinearOperatorBlockDiag(linear_operator.LinearOperator):
  # [CLASS DOC]
  # Linear operator representing a block diagonal (direct sum) matrix built from a non-empty
  # list of component operators that share one dtype. State: the list of operators (also kept
  # as the list of diagonal operators used by shared blockwise helpers). The combined shape is
  # the broadcast batch shape of the blocks followed by summed range and domain dimensions.
  # Hints for non-singular, self-adjoint and positive-definite are inferred as true when all
  # blocks have them. Overrides adjoint, cholesky, inverse, matmul and solve to stay
  # block diagonal when possible, and overrides the matrix-vector and vector-solve methods and
  # many private hooks (shape, determinant, trace, diagonal, dense form, eigenvalues,
  # assertions). Instances are created from a list of operators plus optional hint flags.
  """Combines one or more `LinearOperators` in to a Block Diagonal matrix.

  This operator combines one or more linear operators `[op1,...,opJ]`,
  building a new `LinearOperator`, whose underlying matrix representation
  has each operator `opi` on the main diagonal, and zero's elsewhere.

  #### Shape compatibility

  If `opj` acts like a [batch] matrix `Aj`, then `op_combined` acts like
  the [batch] matrix formed by having each matrix `Aj` on the main
  diagonal.

  Each `opj` is required to represent a matrix, and hence will have
  shape `batch_shape_j + [M_j, N_j]`.

  If `opj` has shape `batch_shape_j + [M_j, N_j]`, then the combined operator
  has shape `broadcast_batch_shape + [sum M_j, sum N_j]`, where
  `broadcast_batch_shape` is the mutual broadcast of `batch_shape_j`,
  `j = 1,...,J`, assuming the intermediate batch shapes broadcast.

  Arguments to `matmul`, `matvec`, `solve`, and `solvevec` may either be single
  `Tensor`s or lists of `Tensor`s that are interpreted as blocks. The `j`th
  element of a blockwise list of `Tensor`s must have dimensions that match
  `opj` for the given method. If a list of blocks is input, then a list of
  blocks is returned as well.

  When the `opj` are not guaranteed to be square, this operator's methods might
  fail due to the combined operator not being square and/or lack of efficient
  methods.

  ```python
  # Create a 4 x 4 linear operator combined of two 2 x 2 operators.
  operator_1 = LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
  operator_2 = LinearOperatorFullMatrix([[1., 0.], [0., 1.]])
  operator = LinearOperatorBlockDiag([operator_1, operator_2])

  operator.to_dense()
  ==> [[1., 2., 0., 0.],
       [3., 4., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]]

  operator.shape
  ==> [4, 4]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x1 = ... # Shape [2, 2] Tensor
  x2 = ... # Shape [2, 2] Tensor
  x = tf.concat([x1, x2], 0)  # Shape [2, 4] Tensor
  operator.matmul(x)
  ==> tf.concat([operator_1.matmul(x1), operator_2.matmul(x2)])

  # Create a 5 x 4 linear operator combining three blocks.
  operator_1 = LinearOperatorFullMatrix([[1.], [3.]])
  operator_2 = LinearOperatorFullMatrix([[1., 6.]])
  operator_3 = LinearOperatorFullMatrix([[2.], [7.]])
  operator = LinearOperatorBlockDiag([operator_1, operator_2, operator_3])

  operator.to_dense()
  ==> [[1., 0., 0., 0.],
       [3., 0., 0., 0.],
       [0., 1., 6., 0.],
       [0., 0., 0., 2.]]
       [0., 0., 0., 7.]]

  operator.shape
  ==> [5, 4]


  # Create a [2, 3] batch of 4 x 4 linear operators.
  matrix_44 = tf.random.normal(shape=[2, 3, 4, 4])
  operator_44 = LinearOperatorFullMatrix(matrix)

  # Create a [1, 3] batch of 5 x 5 linear operators.
  matrix_55 = tf.random.normal(shape=[1, 3, 5, 5])
  operator_55 = LinearOperatorFullMatrix(matrix_55)

  # Combine to create a [2, 3] batch of 9 x 9 operators.
  operator_99 = LinearOperatorBlockDiag([operator_44, operator_55])

  # Create a shape [2, 3, 9] vector.
  x = tf.random.normal(shape=[2, 3, 9])
  operator_99.matmul(x)
  ==> Shape [2, 3, 9] Tensor

  # Create a blockwise list of vectors.
  x = [tf.random.normal(shape=[2, 3, 4]), tf.random.normal(shape=[2, 3, 5])]
  operator_99.matmul(x)
  ==> [Shape [2, 3, 4] Tensor, Shape [2, 3, 5] Tensor]
  ```

  #### Performance

  The performance of `LinearOperatorBlockDiag` on any operation is equal to
  the sum of the individual operators' operations.


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
               operators,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a false is_square raises a ValueError and that non-square
    # operators raise ValueError, but the code never checks squareness of blocks or the flag;
    # it only raises for an empty operator list (ValueError) or mismatched dtypes (TypeError).
    # Constructs the operator from an iterable of linear operators (must be a proper iterable,
    # non-empty, all the same dtype). Optional hint flags (non-singular, self-adjoint,
    # positive-definite, default none) are checked against hints implied by the blocks: if every
    # block has the property and the caller passed false, a ValueError is raised; otherwise the
    # hint is set to true. is_square defaults to true. name defaults to the block names joined
    # by a direct-sum separator. Stores the operators list and the diagonal-operators list, and
    # calls the base constructor inside a name scope, recording the constructor parameters.
    r"""Initialize a `LinearOperatorBlockDiag`.

    `LinearOperatorBlockDiag` is initialized with a list of operators
    `[op_1,...,op_J]`.

    Args:
      operators:  Iterable of `LinearOperator` objects, each with
        the same `dtype` and composable shape.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
        This is true by default, and will raise a `ValueError` otherwise.
      name: A name for this `LinearOperator`.  Default is the individual
        operators names joined with `_o_`.

    Raises:
      TypeError:  If all operators do not have the same `dtype`.
      ValueError:  If `operators` is empty or are non-square.
    """
    parameters = dict(
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )

    # Validate operators.
    check_ops.assert_proper_iterable(operators)
    operators = list(operators)
    if not operators:
      raise ValueError(
          "Expected a non-empty list of operators. Found: %s" % operators)
    self._operators = operators

    # Define diagonal operators, for functions that are shared across blockwise
    # `LinearOperator` types.
    self._diagonal_operators = operators

    # Validate dtype.
    dtype = operators[0].dtype
    for operator in operators:
      if operator.dtype != dtype:
        name_type = (str((o.name, o.dtype)) for o in operators)
        raise TypeError(
            "Expected all operators to have the same dtype.  Found %s"
            % "   ".join(name_type))

    # Auto-set and check hints.
    if all(operator.is_non_singular for operator in operators):
      if is_non_singular is False:
        raise ValueError(
            "The direct sum of non-singular operators is always non-singular.")
      is_non_singular = True

    if all(operator.is_self_adjoint for operator in operators):
      if is_self_adjoint is False:
        raise ValueError(
            "The direct sum of self-adjoint operators is always self-adjoint.")
      is_self_adjoint = True

    if all(operator.is_positive_definite for operator in operators):
      if is_positive_definite is False:
        raise ValueError(
            "The direct sum of positive definite operators is always "
            "positive definite.")
      is_positive_definite = True

    if name is None:
      # Using ds to mean direct sum.
      name = "_ds_".join(operator.name for operator in operators)
    with ops.name_scope(name):
      super(LinearOperatorBlockDiag, self).__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  @property
  def operators(self):
    # [FUNCTION DOC]
    # Property returning the list of component operators (the diagonal blocks) in order.
    return self._operators

  def _block_range_dimensions(self):
    # [FUNCTION DOC]
    # HELPER: supplies per-block sizes to the blockwise matmul/solve logic.
    # Returns a list with the static range dimension (rows) of each diagonal block.
    return [op.range_dimension for op in self._diagonal_operators]

  def _block_domain_dimensions(self):
    # [FUNCTION DOC]
    # HELPER: supplies per-block sizes to the blockwise matmul/solve logic.
    # Returns a list with the static domain dimension (columns) of each diagonal block.
    return [op.domain_dimension for op in self._diagonal_operators]

  def _block_range_dimension_tensors(self):
    # [FUNCTION DOC]
    # HELPER: supplies dynamic per-block sizes for splitting arguments into blocks.
    # Returns a list of scalar tensors, the range dimension (rows) of each diagonal block.
    return [op.range_dimension_tensor() for op in self._diagonal_operators]

  def _block_domain_dimension_tensors(self):
    # [FUNCTION DOC]
    # HELPER: supplies dynamic per-block sizes for splitting arguments into blocks.
    # Returns a list of scalar tensors, the domain dimension (columns) of each diagonal block.
    return [op.domain_dimension_tensor() for op in self._diagonal_operators]

  def _shape(self):
    # [FUNCTION DOC]
    # Computes the static shape of the combined operator: the batch shapes of all blocks are
    # broadcast together, then the matrix shape is the sum of block range dimensions by the sum
    # of block domain dimensions. Returns a tensor shape. Raises a ValueError (from the broadcast
    # helper) if batch shapes are incompatible. No side effects.
    # Get final matrix shape.
    domain_dimension = sum(self._block_domain_dimensions())
    range_dimension = sum(self._block_range_dimensions())
    matrix_shape = tensor_shape.TensorShape([range_dimension, domain_dimension])

    # Get broadcast batch shape.
    # broadcast_shape checks for compatibility.
    batch_shape = self.operators[0].batch_shape
    for operator in self.operators[1:]:
      batch_shape = common_shapes.broadcast_shape(
          batch_shape, operator.batch_shape)

    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # Returns the shape of the combined operator as a 1-D int32 tensor. If the static shape is
    # fully known it is converted to a constant. Otherwise the matrix part is built from summed
    # dynamic block dimensions and the batch part by adding dummy zero tensors of each block's
    # batch shape (to get broadcasting) and taking the shape of the result; the two are
    # concatenated. Adds graph ops; no other side effects.
    # Avoid messy broadcasting if possible.
    if self.shape.is_fully_defined():
      return tensor_conversion.convert_to_tensor_v2_with_dispatch(
          self.shape.as_list(), dtype=dtypes.int32, name="shape"
      )

    domain_dimension = sum(self._block_domain_dimension_tensors())
    range_dimension = sum(self._block_range_dimension_tensors())
    matrix_shape = array_ops_stack.stack([range_dimension, domain_dimension])

    # Dummy Tensor of zeros.  Will never be materialized.
    zeros = array_ops.zeros(shape=self.operators[0].batch_shape_tensor())
    for operator in self.operators[1:]:
      zeros += array_ops.zeros(shape=operator.batch_shape_tensor())
    batch_shape = array_ops.shape(zeros)

    return array_ops.concat((batch_shape, matrix_shape), 0)

  def _linop_adjoint(self) -> "LinearOperatorBlockDiag":
    # [FUNCTION DOC]
    # Returns a new block diagonal operator whose blocks are the adjoints of this operator's
    # blocks, preserving the non-singular, self-adjoint and positive-definite hints; square is
    # set true. Takes no arguments, raises nothing itself, no side effects.
    # We take the adjoint of each block on the diagonal.
    return LinearOperatorBlockDiag(
        operators=[operator.adjoint() for operator in self.operators],
        is_non_singular=self.is_non_singular,
        is_self_adjoint=self.is_self_adjoint,
        is_positive_definite=self.is_positive_definite,
        is_square=True)

  def _linop_cholesky(self) -> "LinearOperatorBlockDiag":
    # [FUNCTION DOC]
    # Returns a new block diagonal operator whose blocks are the Cholesky factors of this
    # operator's blocks. The result is flagged non-singular and square; the self-adjoint hint is
    # left unspecified so the blocks decide. Takes no arguments; no side effects. Failure modes
    # come from each block's own Cholesky.
    # We take the cholesky of each block on the diagonal.
    return LinearOperatorBlockDiag(
        operators=[operator.cholesky() for operator in self.operators],
        is_non_singular=True,
        is_self_adjoint=None,  # Let the operators passed in decide.
        is_square=True)

  def _linop_inverse(self) -> "LinearOperatorBlockDiag":
    # [FUNCTION DOC]
    # Returns a new block diagonal operator whose blocks are the inverses of this operator's
    # blocks, carrying over the non-singular, self-adjoint and positive-definite hints; square
    # is set true. Takes no arguments; no side effects.
    # We take the inverse of each block on the diagonal.
    return LinearOperatorBlockDiag(
        operators=[
            operator.inverse() for operator in self.operators],
        is_non_singular=self.is_non_singular,
        is_self_adjoint=self.is_self_adjoint,
        is_positive_definite=self.is_positive_definite,
        is_square=True)

  def _linop_matmul(
      self,
      left_operator: "LinearOperatorBlockDiag",
      right_operator: linear_operator.LinearOperator,
    ) -> linear_operator.LinearOperator:
    # [FUNCTION DOC]
    # Operator-by-operator product hook. Inputs: the left operator (block diagonal) and the right
    # operator (any linear operator). If the right operator is also block diagonal, returns a new
    # block diagonal operator whose blocks are the pairwise products of corresponding blocks
    # (non-singular hint combined from both operands; self-adjoint and positive-definite
    # unspecified; square true). Note the zip silently truncates if block counts differ, though
    # the public matmul checks counts beforehand. Otherwise defers to the base class.
    if isinstance(right_operator, LinearOperatorBlockDiag):
      return LinearOperatorBlockDiag(
          operators=[
              o1.matmul(o2) for o1, o2 in zip(
                  left_operator.operators, right_operator.operators)],
          is_non_singular=property_hint_util.combined_non_singular_hint(
              left_operator, right_operator),
          # In general, a product of self-adjoint positive-definite
          # block diagonal matrices is not self-adjoint.
          is_self_adjoint=None,
          # In general, a product of positive-definite block diagonal
          # matrices is not positive-definite.
          is_positive_definite=None,
          is_square=True)
    return super()._linop_matmul(left_operator, right_operator)

  def _linop_solve(
      self,
      left_operator: "LinearOperatorBlockDiag",
      right_operator: linear_operator.LinearOperator,
  ) -> linear_operator.LinearOperator:
    # [FUNCTION DOC]
    # Operator-by-operator solve hook. Inputs: left operator (block diagonal) and right operator.
    # If the right operator is also block diagonal, returns a new block diagonal operator whose
    # blocks are the blockwise solves of corresponding blocks (non-singular hint combined;
    # self-adjoint and positive-definite unspecified; square true). The zip silently truncates
    # on differing block counts, though the public solve checks counts first. Otherwise defers
    # to the base class implementation.
    if isinstance(right_operator, LinearOperatorBlockDiag):
      return LinearOperatorBlockDiag(
          operators=[
              o1.solve(o2) for o1, o2 in zip(
                  left_operator.operators, right_operator.operators)],
          is_non_singular=property_hint_util.combined_non_singular_hint(
              left_operator, right_operator),
          # In general, a solve of self-adjoint positive-definite block diagonal
          # matrices is not self-=adjoint.
          is_self_adjoint=None,
          # In general, a solve of positive-definite block diagonal matrices is
          # not positive-definite.
          is_positive_definite=None,
          is_square=True)
    return super()._linop_solve(left_operator, right_operator)

  # TODO(b/188080761): Add a more efficient implementation of `cond` that
  # constructs the condition number from the blockwise singular values.

  def matmul(self, x, adjoint=False, adjoint_arg=False, name="matmul"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when x is a blockwise list, the code assigns converted tensors back into the
    # caller's list, mutating the argument in place, which the docstring does not mention.
    # Multiplies this operator (or its adjoint) by x. x may be a linear operator (result is a
    # linear operator; for two block diagonal operators the block counts must match and each
    # block pair must be dimension-compatible), a single tensor (split into row blocks, each
    # multiplied by its block, results broadcast over batch dims and concatenated), or a list of
    # per-block tensors/operators (a list of per-block results is returned). adjoint multiplies by
    # the adjoint; adjoint_arg uses the hermitian transpose of x; name is the name scope.
    # Raises a ValueError for incompatible operator dimensions or differing block counts, and
    # type/shape errors for dtype or shape mismatches of x.
    """Transform [batch] matrix `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    X = ... # shape [..., N, R], batch matrix, R > 0.

    Y = operator.matmul(X)
    Y.shape
    ==> [..., M, R]

    Y[..., :, r] = sum_j A[..., :, j] X[j, r]
    ```

    Args:
      x: `LinearOperator`, `Tensor` with compatible shape and same `dtype` as
        `self`, or a blockwise iterable of `LinearOperator`s or `Tensor`s. See
        class docstring for definition of shape compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      adjoint_arg:  Python `bool`.  If `True`, compute `A x^H` where `x^H` is
        the hermitian transpose (transposition and complex conjugation).
      name:  A name for this `Op`.

    Returns:
      A `LinearOperator` or `Tensor` with shape `[..., M, R]` and same `dtype`
        as `self`, or if `x` is blockwise, a list of `Tensor`s with shapes that
        concatenate to `[..., M, R]`.
    """
    def _check_operators_agree(r, l, message):
      # [FUNCTION DOC]
      # HELPER: dimension validation inside matmul.
      # Raises a ValueError with the given message if both the right operator's range dimension
      # and the left operator's domain dimension are statically known and differ. Returns nothing.
      if (r.range_dimension is not None and
          l.domain_dimension is not None and
          r.range_dimension != l.domain_dimension):
        raise ValueError(message)

    if isinstance(x, linear_operator.LinearOperator):
      left_operator = self.adjoint() if adjoint else self
      right_operator = x.adjoint() if adjoint_arg else x

      _check_operators_agree(
          right_operator, left_operator,
          "Operators are incompatible. Expected `x` to have dimension"
          " {} but got {}.".format(
              left_operator.domain_dimension, right_operator.range_dimension))

      # We can efficiently multiply BlockDiag LinearOperators if the number of
      # blocks agree.
      if isinstance(x, LinearOperatorBlockDiag):
        if len(left_operator.operators) != len(right_operator.operators):
          raise ValueError(
              "Can not efficiently multiply two `LinearOperatorBlockDiag`s "
              "together when number of blocks differ.")

        for o1, o2 in zip(left_operator.operators, right_operator.operators):
          _check_operators_agree(
              o2, o1,
              "Blocks are incompatible. Expected `x` to have dimension"
              " {} but got {}.".format(
                  o1.domain_dimension, o2.range_dimension))

      with self._name_scope(name):  # pylint: disable=not-callable
        return self._linop_matmul(left_operator, right_operator)

    with self._name_scope(name):  # pylint: disable=not-callable
      arg_dim = -1 if adjoint_arg else -2
      block_dimensions = (self._block_range_dimensions() if adjoint
                          else self._block_domain_dimensions())
      if linear_operator_util.arg_is_blockwise(block_dimensions, x, arg_dim):
        for i, block in enumerate(x):
          if not isinstance(block, linear_operator.LinearOperator):
            block = tensor_conversion.convert_to_tensor_v2_with_dispatch(block)
            self._check_input_dtype(block)
            block_dimensions[i].assert_is_compatible_with(block.shape[arg_dim])
            x[i] = block
      else:
        x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name="x")
        self._check_input_dtype(x)
        op_dimension = (self.range_dimension if adjoint
                        else self.domain_dimension)
        op_dimension.assert_is_compatible_with(x.shape[arg_dim])
      return self._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Core product with already validated x. If x is a blockwise list it is used as is; otherwise
    # x is split along rows (columns when adjoint_arg) according to the block sizes (range sizes
    # if adjoint, else domain sizes). Each block operator multiplies its piece (honoring adjoint
    # and adjoint_arg). Returns the list of per-block results for blockwise input, or the results
    # broadcast to a common batch shape and concatenated along the row axis for a single tensor.
    arg_dim = -1 if adjoint_arg else -2
    block_dimensions = (self._block_range_dimensions() if adjoint
                        else self._block_domain_dimensions())
    block_dimensions_fn = (
        self._block_range_dimension_tensors if adjoint
        else self._block_domain_dimension_tensors)
    blockwise_arg = linear_operator_util.arg_is_blockwise(
        block_dimensions, x, arg_dim)
    if blockwise_arg:
      split_x = x

    else:
      split_dim = -1 if adjoint_arg else -2
      # Split input by rows normally, and otherwise columns.
      split_x = linear_operator_util.split_arg_into_blocks(
          block_dimensions, block_dimensions_fn, x, axis=split_dim)

    result_list = []
    for index, operator in enumerate(self.operators):
      result_list += [operator.matmul(
          split_x[index], adjoint=adjoint, adjoint_arg=adjoint_arg)]

    if blockwise_arg:
      return result_list

    result_list = linear_operator_util.broadcast_matrix_batch_dims(
        result_list)
    return array_ops.concat(result_list, axis=-2)

  def matvec(self, x, adjoint=False, name="matvec"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: for a blockwise list input, converted tensors are written back into the
    # caller's list (argument mutated), and the blockwise path also accepts operator entries
    # without converting them, which the docstring does not describe.
    # Multiplies this operator (or its adjoint when adjoint is true) by a vector or list of
    # per-block vectors. Vectors are turned into one-column matrices, passed through matmul, and
    # the trailing axis is squeezed away. Returns a tensor of shape batch plus range size, or a
    # list of per-block tensors for blockwise input. name is the name scope. Raises errors if
    # dtype or last-dimension sizes are incompatible.
    """Transform [batch] vector `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matric A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)

    X = ... # shape [..., N], batch vector

    Y = operator.matvec(X)
    Y.shape
    ==> [..., M]

    Y[..., :] = sum_j A[..., :, j] X[..., j]
    ```

    Args:
      x: `Tensor` with compatible shape and same `dtype` as `self`, or an
        iterable of `Tensor`s (for blockwise operators). `Tensor`s are treated
        a [batch] vectors, meaning for every set of leading dimensions, the last
        dimension defines a vector.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      name:  A name for this `Op`.

    Returns:
      A `Tensor` with shape `[..., M]` and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      block_dimensions = (self._block_range_dimensions() if adjoint
                          else self._block_domain_dimensions())
      if linear_operator_util.arg_is_blockwise(block_dimensions, x, -1):
        for i, block in enumerate(x):
          if not isinstance(block, linear_operator.LinearOperator):
            block = tensor_conversion.convert_to_tensor_v2_with_dispatch(block)
            self._check_input_dtype(block)
            block_dimensions[i].assert_is_compatible_with(block.shape[-1])
            x[i] = block
        x_mat = [block[..., array_ops.newaxis] for block in x]
        y_mat = self.matmul(x_mat, adjoint=adjoint)
        return [array_ops.squeeze(y, axis=-1) for y in y_mat]

      x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name="x")
      self._check_input_dtype(x)
      op_dimension = (self.range_dimension if adjoint
                      else self.domain_dimension)
      op_dimension.assert_is_compatible_with(x.shape[-1])
      x_mat = x[..., array_ops.newaxis]
      y_mat = self.matmul(x_mat, adjoint=adjoint)
      return array_ops.squeeze(y_mat, axis=-1)

  def _determinant(self):
    # [FUNCTION DOC]
    # Returns the determinant as the product of the determinants of all blocks (batch
    # broadcasting applies). No side effects.
    result = self.operators[0].determinant()
    for operator in self.operators[1:]:
      result *= operator.determinant()
    return result

  def _log_abs_determinant(self):
    # [FUNCTION DOC]
    # Returns the log of the absolute determinant as the sum of each block's log absolute
    # determinant (batch broadcasting applies). No side effects.
    result = self.operators[0].log_abs_determinant()
    for operator in self.operators[1:]:
      result += operator.log_abs_determinant()
    return result

  def solve(self, rhs, adjoint=False, adjoint_arg=False, name="solve"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: for a single (non-blockwise) tensor rhs the code splits it using the block
    # domain dimensions, but the right-hand side lives in the range space (and for the adjoint
    # case the block_dimensions variable computed above is not used), so splitting is wrong for
    # non-square blocks. Also, blockwise rhs lists are mutated in place by writing converted
    # tensors back into the caller's list.
    # Solves the block system A X = rhs (or with the adjoint, or with the hermitian transpose of
    # rhs). rhs may be an operator (block diagonal with equal block counts gives a block diagonal
    # result; dimension mismatch raises a ValueError), a tensor (split into row blocks, each block
    # solved, results broadcast and concatenated) or a list of per-block tensors (list returned).
    # Raises NotImplementedError if the operator is known singular or known non-square.
    """Solve (exact or approx) `R` (batch) systems of equations: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve R > 0 linear systems for every member of the batch.
    RHS = ... # shape [..., M, R]

    X = operator.solve(RHS)
    # X[..., :, r] is the solution to the r'th linear system
    # sum_j A[..., :, j] X[..., j, r] = RHS[..., :, r]

    operator.matmul(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator and compatible shape,
        or a list of `Tensor`s (for blockwise operators). `Tensor`s are treated
        like a [batch] matrices meaning for every set of leading dimensions, the
        last two dimensions defines a matrix.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      adjoint_arg:  Python `bool`.  If `True`, solve `A X = rhs^H` where `rhs^H`
        is the hermitian transpose (transposition and complex conjugation).
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    if self.is_non_singular is False:
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "be singular.")
    if self.is_square is False:
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "not be square.")

    def _check_operators_agree(r, l, message):
      # [FUNCTION DOC]
      # HELPER: dimension validation inside solve.
      # Raises a ValueError with the given message if both the right operator's range dimension
      # and the left operator's domain dimension are statically known and differ. Returns nothing.
      if (r.range_dimension is not None and
          l.domain_dimension is not None and
          r.range_dimension != l.domain_dimension):
        raise ValueError(message)

    if isinstance(rhs, linear_operator.LinearOperator):
      left_operator = self.adjoint() if adjoint else self
      right_operator = rhs.adjoint() if adjoint_arg else rhs

      _check_operators_agree(
          right_operator, left_operator,
          "Operators are incompatible. Expected `x` to have dimension"
          " {} but got {}.".format(
              left_operator.domain_dimension, right_operator.range_dimension))

      # We can efficiently solve BlockDiag LinearOperators if the number of
      # blocks agree.
      if isinstance(right_operator, LinearOperatorBlockDiag):
        if len(left_operator.operators) != len(right_operator.operators):
          raise ValueError(
              "Can not efficiently solve `LinearOperatorBlockDiag` when "
              "number of blocks differ.")

        for o1, o2 in zip(left_operator.operators, right_operator.operators):
          _check_operators_agree(
              o2, o1,
              "Blocks are incompatible. Expected `x` to have dimension"
              " {} but got {}.".format(
                  o1.domain_dimension, o2.range_dimension))

      with self._name_scope(name):  # pylint: disable=not-callable
        return self._linop_solve(left_operator, right_operator)

    with self._name_scope(name):  # pylint: disable=not-callable
      block_dimensions = (self._block_domain_dimensions() if adjoint
                          else self._block_range_dimensions())
      arg_dim = -1 if adjoint_arg else -2
      blockwise_arg = linear_operator_util.arg_is_blockwise(
          block_dimensions, rhs, arg_dim)

      if blockwise_arg:
        split_rhs = rhs
        for i, block in enumerate(split_rhs):
          if not isinstance(block, linear_operator.LinearOperator):
            block = tensor_conversion.convert_to_tensor_v2_with_dispatch(block)
            self._check_input_dtype(block)
            block_dimensions[i].assert_is_compatible_with(block.shape[arg_dim])
            split_rhs[i] = block
      else:
        rhs = tensor_conversion.convert_to_tensor_v2_with_dispatch(
            rhs, name="rhs"
        )
        self._check_input_dtype(rhs)
        op_dimension = (self.domain_dimension if adjoint
                        else self.range_dimension)
        op_dimension.assert_is_compatible_with(rhs.shape[arg_dim])
        split_dim = -1 if adjoint_arg else -2
        # Split input by rows normally, and otherwise columns.
        split_rhs = linear_operator_util.split_arg_into_blocks(
            self._block_domain_dimensions(),
            self._block_domain_dimension_tensors,
            rhs, axis=split_dim)

      solution_list = []
      for index, operator in enumerate(self.operators):
        solution_list += [operator.solve(
            split_rhs[index], adjoint=adjoint, adjoint_arg=adjoint_arg)]

      if blockwise_arg:
        return solution_list

      solution_list = linear_operator_util.broadcast_matrix_batch_dims(
          solution_list)
      return array_ops.concat(solution_list, axis=-2)

  def solvevec(self, rhs, adjoint=False, name="solve"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the default name scope is "solve" rather than "solvevec", and the docstring
    # claims NotImplementedError is raised, but this method raises it only indirectly through
    # solve. Blockwise lists passed by the caller are also mutated in place.
    # Solves for a vector (or list of per-block vectors) by adding a trailing axis, calling solve,
    # and squeezing the axis away. adjoint solves with the adjoint operator. Returns a tensor
    # (or a list of tensors for blockwise input).
    """Solve single equation with best effort: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve one linear system for every member of the batch.
    RHS = ... # shape [..., M]

    X = operator.solvevec(RHS)
    # X is the solution to the linear system
    # sum_j A[..., :, j] X[..., j] = RHS[..., :]

    operator.matvec(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator, or list of `Tensor`s
        (for blockwise operators). `Tensor`s are treated as [batch] vectors,
        meaning for every set of leading dimensions, the last dimension defines
        a vector.  See class docstring for definition of compatibility regarding
        batch dimensions.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      block_dimensions = (self._block_domain_dimensions() if adjoint
                          else self._block_range_dimensions())
      if linear_operator_util.arg_is_blockwise(block_dimensions, rhs, -1):
        for i, block in enumerate(rhs):
          if not isinstance(block, linear_operator.LinearOperator):
            block = tensor_conversion.convert_to_tensor_v2_with_dispatch(block)
            self._check_input_dtype(block)
            block_dimensions[i].assert_is_compatible_with(block.shape[-1])
            rhs[i] = block
        rhs_mat = [array_ops.expand_dims(block, axis=-1) for block in rhs]
        solution_mat = self.solve(rhs_mat, adjoint=adjoint)
        return [array_ops.squeeze(x, axis=-1) for x in solution_mat]

      rhs = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          rhs, name="rhs"
      )
      self._check_input_dtype(rhs)
      op_dimension = (self.domain_dimension if adjoint
                      else self.range_dimension)
      op_dimension.assert_is_compatible_with(rhs.shape[-1])
      rhs_mat = array_ops.expand_dims(rhs, axis=-1)
      solution_mat = self.solve(rhs_mat, adjoint=adjoint)
      return array_ops.squeeze(solution_mat, axis=-1)

  def _diag_part(self):
    # [FUNCTION DOC]
    # Returns the main diagonal of the combined operator by concatenating each block's diagonal
    # (after batch broadcasting). Raises NotImplementedError if any block is not square.
    if not all(operator.is_square for operator in self.operators):
      raise NotImplementedError(
          "`diag_part` not implemented for an operator whose blocks are not "
          "square.")
    diag_list = []
    for operator in self.operators:
      # Extend the axis for broadcasting.
      diag_list += [operator.diag_part()[..., array_ops.newaxis]]
    diag_list = linear_operator_util.broadcast_matrix_batch_dims(diag_list)
    diagonal = array_ops.concat(diag_list, axis=-2)
    return array_ops.squeeze(diagonal, axis=-1)

  def _trace(self):
    # [FUNCTION DOC]
    # Returns the trace as the sum of the traces of all blocks. Raises NotImplementedError if
    # any block is not square.
    if not all(operator.is_square for operator in self.operators):
      raise NotImplementedError(
          "`trace` not implemented for an operator whose blocks are not "
          "square.")
    result = self.operators[0].trace()
    for operator in self.operators[1:]:
      result += operator.trace()
    return result

  def _to_dense(self):
    # [FUNCTION DOC]
    # Builds the dense matrix: each block is densified and batch-broadcast, then padded with zero
    # columns before and after (according to running column offset and total domain size) to
    # form a full row strip; strips are concatenated along the row axis and the static shape is
    # set to this operator's shape. Returns a dense tensor. Adds graph ops only.
    num_cols = 0
    rows = []
    broadcasted_blocks = [operator.to_dense() for operator in self.operators]
    broadcasted_blocks = linear_operator_util.broadcast_matrix_batch_dims(
        broadcasted_blocks)
    for block in broadcasted_blocks:
      batch_row_shape = array_ops.shape(block)[:-1]

      zeros_to_pad_before_shape = array_ops.concat(
          [batch_row_shape, [num_cols]], axis=-1)
      zeros_to_pad_before = array_ops.zeros(
          shape=zeros_to_pad_before_shape, dtype=block.dtype)
      num_cols += array_ops.shape(block)[-1]
      zeros_to_pad_after_shape = array_ops.concat(
          [batch_row_shape,
           [self.domain_dimension_tensor() - num_cols]], axis=-1)
      zeros_to_pad_after = array_ops.zeros(
          shape=zeros_to_pad_after_shape, dtype=block.dtype)

      rows.append(array_ops.concat(
          [zeros_to_pad_before, block, zeros_to_pad_after], axis=-1))

    mat = array_ops.concat(rows, axis=-2)
    mat.set_shape(self.shape)
    return mat

  def _assert_non_singular(self):
    # [FUNCTION DOC]
    # Returns a grouped op that asserts every block is non-singular (the direct sum is
    # non-singular exactly when all blocks are).
    return control_flow_ops.group([
        operator.assert_non_singular() for operator in self.operators])

  def _assert_self_adjoint(self):
    # [FUNCTION DOC]
    # Returns a grouped op that asserts every block is self-adjoint.
    return control_flow_ops.group([
        operator.assert_self_adjoint() for operator in self.operators])

  def _assert_positive_definite(self):
    # [FUNCTION DOC]
    # Returns a grouped op that asserts every block is positive definite.
    return control_flow_ops.group([
        operator.assert_positive_definite() for operator in self.operators])

  def _eigvals(self):
    # [FUNCTION DOC]
    # Returns the eigenvalues of the combined operator by concatenating each block's eigenvalues
    # (after batch broadcasting). Raises NotImplementedError if any block is not square.
    if not all(operator.is_square for operator in self.operators):
      raise NotImplementedError(
          "`eigvals` not implemented for an operator whose blocks are not "
          "square.")
    eig_list = []
    for operator in self.operators:
      # Extend the axis for broadcasting.
      eig_list += [operator.eigvals()[..., array_ops.newaxis]]
    eig_list = linear_operator_util.broadcast_matrix_batch_dims(eig_list)
    eigs = array_ops.concat(eig_list, axis=-2)
    return array_ops.squeeze(eigs, axis=-1)

  @property
  def _composite_tensor_fields(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor support. Returns the tuple of constructor argument names that are
    # tensor-like parts of this operator: just the operators list.
    return ("operators",)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor/batching support. Returns a mapping from the operators parameter
    # to a list of zeros, one per block, meaning each block operator contributes zero extra
    # dimensions beyond its own matrix dimensions.
    return {"operators": [0] * len(self.operators)}
