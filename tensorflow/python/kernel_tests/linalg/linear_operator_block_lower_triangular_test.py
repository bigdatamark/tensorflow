# [MODULE DOC]
# Tests for the block lower-triangular linear operator. Defines a derived-class test case that
# plugs into the shared linear operator test utility (which generates matmul, solve, determinant,
# etc. checks against a dense reference matrix), plus extra tests for flag auto-setting, inverse
# type, argument validation errors, gradient (tape) safety, variable conversion and composite
# gradients. Also defines a helper that builds the dense reference matrix from a list of block
# rows. Import-time side effect: creates a seeded module-level random number generator. When run
# as a script it registers the generated tests and starts the test runner.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_block_lower_triangular as block_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(0)


def _block_lower_triangular_dense(expected_shape, blocks):
  # [FUNCTION DOC]
  # HELPER: builds the dense reference matrix for the block lower-triangular operator tests.
  # Takes the expected full dense shape (batch dims plus rows and columns) and a list of block
  # rows, each a list of dense tensors with the same batch dims. For each row, it concatenates
  # the blocks along columns and right-pads with zeros so the row spans the full column count,
  # then stacks all rows along the row axis. Returns the dense tensor. Side effect: appends the
  # zero padding tensor to each inner row list, mutating the caller's lists.
  """Convert a list of blocks into a dense blockwise lower-triangular matrix."""
  rows = []
  num_cols = 0
  for row_blocks in blocks:

    # Get the batch shape for the block.
    batch_row_shape = array_ops.shape(row_blocks[0])[:-1]

    num_cols += array_ops.shape(row_blocks[-1])[-1]
    zeros_to_pad_after_shape = array_ops.concat(
        [batch_row_shape, [expected_shape[-2] - num_cols]], axis=-1)
    zeros_to_pad_after = array_ops.zeros(
        zeros_to_pad_after_shape, dtype=row_blocks[-1].dtype)

    row_blocks.append(zeros_to_pad_after)
    rows.append(array_ops.concat(row_blocks, axis=-1))

  return array_ops.concat(rows, axis=-2)


@test_util.run_all_in_graph_and_eager_modes
class SquareLinearOperatorBlockLowerTriangularTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the block lower-triangular linear operator, derived from the shared square
  # operator test base class so the generic operator tests run against it. Supplies the shapes to
  # test, how to build an operator with a matching dense matrix, and tests for flags, inverse
  # type, error cases and gradients. Runs in both graph and eager modes. Disables TF32 execution
  # during each test and loosens float32/complex64 tolerances to 1e-5.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the base class tearDown, so parent cleanup is skipped.
    # Restores the TensorFloat-32 execution setting saved in setUp. Takes no inputs, returns
    # nothing; side effect is changing the global TF32 flag back to its original value.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup. Saves the current TF32 execution flag, turns TF32 off, raises the
    # absolute and relative tolerances for float32 and complex64 from 1e-6 to 1e-5, then calls
    # the base class setUp. Takes no inputs and returns nothing; mutates the global TF32 flag
    # and the per-dtype tolerance mappings held on the instance.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Increase from 1e-6 to 1e-5
    self._atol[dtypes.float32] = 1e-5
    self._atol[dtypes.complex64] = 1e-5
    self._rtol[dtypes.float32] = 1e-5
    self._rtol[dtypes.complex64] = 1e-5
    super(SquareLinearOperatorBlockLowerTriangularTest, self).setUp()

  @staticmethod
  def use_blockwise_arg():
    # [FUNCTION DOC]
    # Static hook read by the base test utility. Returns true to tell it that this operator
    # accepts and returns blockwise (list of per-block) arguments in addition to dense ones.
    return True

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Static hook read by the base test utility. Returns the list of generated test names to
    # skip: cholesky and eigenvalue tests, because this operator is generally not self-adjoint.
    # Skipping since `LinearOperatorBlockLowerTriangular` is in general not
    # self-adjoint.
    return ["cholesky", "eigvalsh"]

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Static hook read by the base test utility. Returns a list of shape-info records describing
    # the operator shapes to test: empty, 1x1, batched 3x3, and several multi-block layouts (with
    # explicit per-block shapes, including batched blocks) up to a batch of shape 2 by 4 of 6x6.
    shape_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shape_info((0, 0)),
        shape_info((1, 1)),
        shape_info((1, 3, 3)),
        shape_info((5, 5), blocks=[[(2, 2)], [(3, 2), (3, 3)]]),
        shape_info((3, 7, 7),
                   blocks=[[(1, 2, 2)], [(1, 3, 2), (3, 3, 3)],
                           [(1, 2, 2), (1, 2, 3), (1, 2, 2)]]),
        shape_info((2, 4, 6, 6),
                   blocks=[[(2, 1, 2, 2)], [(1, 4, 2), (4, 4, 4)]]),
    ]

  def operator_and_matrix(
      self, shape_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Builds the operator under test and its dense reference matrix for the base test utility.
    # Inputs: shape_info (record with the full shape and optional per-block shapes), dtype,
    # use_placeholder (if true, wrap block matrices in placeholders with unknown static shape),
    # and ensure_self_adjoint_and_pd (if true, mark the block operators self-adjoint and positive
    # definite). Diagonal blocks are random well-conditioned positive-definite matrices, the
    # others random normal. Wraps each in a full-matrix operator, builds the block lower-triangular
    # operator, asserts it auto-detected squareness, broadcasts block batch dims, and assembles the
    # dense matrix (setting its static shape unless placeholders are used). Returns the pair
    # (operator, dense matrix).

    expected_blocks = (
        shape_info.__dict__["blocks"] if "blocks" in shape_info.__dict__
        else [[list(shape_info.shape)]])

    matrices = []
    for i, row_shapes in enumerate(expected_blocks):
      row = []
      for j, block_shape in enumerate(row_shapes):
        if i == j:  # operator is on the diagonal
          row.append(
              linear_operator_test_util.random_positive_definite_matrix(
                  block_shape, dtype, force_well_conditioned=True))
        else:
          row.append(
              linear_operator_test_util.random_normal(block_shape, dtype=dtype))
      matrices.append(row)

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [[
          array_ops.placeholder_with_default(
              matrix, shape=None) for matrix in row] for row in matrices]

    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[linalg.LinearOperatorFullMatrix(  # pylint:disable=g-complex-comprehension
            l,
            is_square=True,
            is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
            is_positive_definite=True if ensure_self_adjoint_and_pd else None)
          for l in row] for row in lin_op_matrices])

    # Should be auto-set.
    self.assertTrue(operator.is_square)

    # Broadcast the shapes.
    expected_shape = list(shape_info.shape)
    broadcasted_matrices = linear_operator_util.broadcast_matrix_batch_dims(
        [op for row in matrices for op in row])  # pylint: disable=g-complex-comprehension
    matrices = [broadcasted_matrices[i * (i + 1) // 2:(i + 1) * (i + 2) // 2]
                for i in range(len(matrices))]

    block_lower_triangular_dense = _block_lower_triangular_dense(
        expected_shape, matrices)

    if not use_placeholder:
      block_lower_triangular_dense.set_shape(expected_shape)

    return operator, block_lower_triangular_dense

  def test_is_x_flags(self):
    # [FUNCTION DOC]
    # Builds an operator from a single block with explicit positive-definite, non-singular and
    # not-self-adjoint flags and asserts those flags are reported back unchanged.
    # Matrix with two positive eigenvalues, 1, and 1.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[linalg.LinearOperatorFullMatrix(matrix)]],
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_block_lower_triangular_inverse_type(self):
    # [FUNCTION DOC]
    # Builds a two-row non-singular block lower-triangular operator from identity blocks, takes its
    # inverse and asserts the result is again a block lower-triangular operator with one block in
    # the first row and two in the second.
    matrix = [[1., 0.], [0., 1.]]
    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)],
         [linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True),
          linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)]],
        is_non_singular=True,
    )
    inverse = operator.inverse()
    self.assertIsInstance(
        inverse,
        block_lower_triangular.LinearOperatorBlockLowerTriangular)
    self.assertEqual(2, len(inverse.operators))
    self.assertEqual(1, len(inverse.operators[0]))
    self.assertEqual(2, len(inverse.operators[1]))

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Builds an operator whose blocks hold trainable variables, runs the shared tape-safety check
    # (skipping the diagonal-only methods), then for diag part, trace, determinant and log absolute
    # determinant computes gradients with respect to the diagonal blocks' variables and asserts
    # none of the gradients is null.
    operator_1 = linalg.LinearOperatorFullMatrix(
        variables_module.Variable([[1., 0.], [0., 1.]]),
        is_self_adjoint=True,
        is_positive_definite=True)
    operator_2 = linalg.LinearOperatorFullMatrix(
        variables_module.Variable([[2., 0.], [1., 0.]]))
    operator_3 = linalg.LinearOperatorFullMatrix(
        variables_module.Variable([[3., 1.], [1., 3.]]),
        is_self_adjoint=True,
        is_positive_definite=True)
    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[operator_1], [operator_2, operator_3]],
        is_self_adjoint=False,
        is_positive_definite=True)

    diagonal_grads_only = ["diag_part", "trace", "determinant",
                           "log_abs_determinant"]
    self.check_tape_safe(operator, skip_options=diagonal_grads_only)

    for y in diagonal_grads_only:
      for diag_block in [operator_1, operator_3]:
        with backprop.GradientTape() as tape:
          grads = tape.gradient(getattr(operator, y)(), diag_block.variables)
          for item in grads:
            self.assertIsNotNone(item)

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Builds an operator over variable-backed blocks, initializes the variables in a cached
    # session, and runs the shared check that converting the operator's variables to tensors
    # works.
    operator_1 = linalg.LinearOperatorFullMatrix(
        variables_module.Variable([[1., 0.], [0., 1.]]),
        is_self_adjoint=True,
        is_positive_definite=True)
    operator_2 = linalg.LinearOperatorFullMatrix(
        variables_module.Variable([[2., 0.], [1., 0.]]))
    operator_3 = linalg.LinearOperatorFullMatrix(
        variables_module.Variable([[3., 1.], [1., 3.]]),
        is_self_adjoint=True,
        is_positive_definite=True)
    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[operator_1], [operator_2, operator_3]],
        is_self_adjoint=False,
        is_positive_definite=True)
    with self.cached_session() as sess:
      sess.run([x.initializer for x in operator.variables])
      self.check_convert_variables_to_tensors(operator)

  def test_is_non_singular_auto_set(self):
    # [FUNCTION DOC]
    # Checks automatic non-singularity inference: with non-singular diagonal blocks the operator
    # reports non-singular even if not told so. Asserts a value error ("always non-singular") when
    # non-singular is forced false, that a singular off-diagonal block is accepted, and a value
    # error ("always singular") when a singular diagonal block is combined with a non-singular
    # claim.
    # Matrix with two positive eigenvalues, 11 and 8.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    operator_3 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)

    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[operator_1], [operator_2, operator_3]],
        is_positive_definite=False,  # No reason it HAS to be False...
        is_non_singular=None)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)

    with self.assertRaisesRegex(ValueError, "always non-singular"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular(
          [[operator_1], [operator_2, operator_3]], is_non_singular=False)

    operator_4 = linalg.LinearOperatorFullMatrix(
        [[1., 0.], [2., 0.]], is_non_singular=False)

    # A singular operator off of the main diagonal shouldn't raise
    block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[operator_1], [operator_4, operator_2]], is_non_singular=True)

    with self.assertRaisesRegex(ValueError, "always singular"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular(
          [[operator_1], [operator_2, operator_4]], is_non_singular=True)

  def test_different_dtypes_raises(self):
    # [FUNCTION DOC]
    # Constructs the operator from blocks of mixed dtypes (float64 and float32) and asserts a type
    # error mentioning "same dtype".
    operators = [
        [linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3))],
        [linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3)),
         linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3).astype(np.float32))]
    ]
    with self.assertRaisesRegex(TypeError, "same dtype"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular(operators)

  def test_non_square_operator_raises(self):
    # [FUNCTION DOC]
    # Constructs the operator with a non-square diagonal block and asserts a value error saying
    # the blocks must be square.
    operators = [
        [linalg.LinearOperatorFullMatrix(rng.rand(3, 4), is_square=False)],
        [linalg.LinearOperatorFullMatrix(rng.rand(4, 4)),
         linalg.LinearOperatorFullMatrix(rng.rand(4, 4))]
    ]
    with self.assertRaisesRegex(ValueError, "must be square"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular(operators)

  def test_empty_operators_raises(self):
    # [FUNCTION DOC]
    # Constructs the operator from an empty list and asserts a value error saying a list of at
    # least one row is required.
    with self.assertRaisesRegex(ValueError, "must be a list of >=1"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular([])

  def test_operators_wrong_length_raises(self):
    # [FUNCTION DOC]
    # Constructs the operator with a second row of three blocks instead of two and asserts a value
    # error saying the row must contain two blocks.
    with self.assertRaisesRegex(ValueError, "must contain `2` blocks"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular([
          [linalg.LinearOperatorFullMatrix(rng.rand(2, 2))],
          [linalg.LinearOperatorFullMatrix(rng.rand(2, 2))
           for _ in range(3)]])

  def test_operators_mismatched_dimension_raises(self):
    # [FUNCTION DOC]
    # Constructs the operator with an off-diagonal block whose column count differs from the
    # diagonal block above it and asserts a value error about dimensions needing to be the same.
    operators = [
        [linalg.LinearOperatorFullMatrix(rng.rand(3, 3))],
        [linalg.LinearOperatorFullMatrix(rng.rand(3, 4)),
         linalg.LinearOperatorFullMatrix(rng.rand(3, 3))]
    ]
    with self.assertRaisesRegex(ValueError, "must be the same as"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular(operators)

  def test_incompatible_input_blocks_raises(self):
    # [FUNCTION DOC]
    # Builds an operator from placeholder blocks of unknown static shape (4x4, 3x4, 3x3) and
    # multiplies it by an input whose column structure cannot be split across blocks. Asserts a
    # value error: "dimension does not match" in eager mode, "input structure is ambiguous" in
    # graph mode.
    matrix_1 = array_ops.placeholder_with_default(rng.rand(4, 4), shape=None)
    matrix_2 = array_ops.placeholder_with_default(rng.rand(3, 4), shape=None)
    matrix_3 = array_ops.placeholder_with_default(rng.rand(3, 3), shape=None)
    operators = [
        [linalg.LinearOperatorFullMatrix(matrix_1, is_square=True)],
        [linalg.LinearOperatorFullMatrix(matrix_2),
         linalg.LinearOperatorFullMatrix(matrix_3, is_square=True)]
    ]
    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        operators)
    x = np.random.rand(2, 4, 5).tolist()
    msg = ("dimension does not match" if context.executing_eagerly()
           else "input structure is ambiguous")
    with self.assertRaisesRegex(ValueError, msg):
      operator.matmul(x)

  def test_composite_gradients(self):
    # [FUNCTION DOC]
    # Under a gradient tape, builds a block operator from three watched full-matrix operators,
    # multiplies the first by a generated input and takes gradients of the result with respect to
    # the first operator, the last operator and the composite operator. Asserts the gradient of
    # the first matches the composite's first block, the composite's last block gradient is all
    # zeros, and the standalone gradient for the unused operator is null.
    with backprop.GradientTape() as tape:
      op1 = linalg.LinearOperatorFullMatrix(rng.rand(4, 4), is_square=True)
      op2 = linalg.LinearOperatorFullMatrix(rng.rand(3, 4))
      op3 = linalg.LinearOperatorFullMatrix(rng.rand(3, 3), is_square=True)
      tape.watch([op1, op2, op3])
      operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
          [[op1], [op2, op3]])

      x = self.make_x(op1, adjoint=False)
      y = op1.matmul(x)
      connected_grad, disconnected_grad, composite_grad = tape.gradient(
          y, [op1, op3, operator]
      )

    disconnected_component_grad = composite_grad.operators[1][1].to_dense()
    self.assertAllClose(connected_grad.to_dense(),
                        composite_grad.operators[0][0].to_dense())
    self.assertAllClose(disconnected_component_grad,
                        array_ops.zeros_like(disconnected_component_grad))
    self.assertIsNone(disconnected_grad)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(
      SquareLinearOperatorBlockLowerTriangularTest)
  test.main()
