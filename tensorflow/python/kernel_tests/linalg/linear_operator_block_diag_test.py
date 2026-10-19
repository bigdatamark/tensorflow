# [MODULE DOC]
# Tests for the block-diagonal linear operator. Defines a dense block-diagonal reference builder,
# a square-operator test class and a non-square test class derived from the shared linear
# operator test utilities (which provide the generic tests), plus extra tests for flags,
# parameters, adjoint/cholesky/inverse/matmul/solve results staying block diagonal, error cases,
# variable handling, composite gradients, naming and dtype checks. Creates a module-level seeded
# random generator; when run as a program registers the generated tests for both classes.
#
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

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_block_diag as block_diag
from tensorflow.python.ops.linalg import linear_operator_lower_triangular as lower_triangular
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(0)


def _block_diag_dense(expected_shape, blocks):
  # [FUNCTION DOC]
  # HELPER: builds the dense reference matrix for the block-diagonal tests. Inputs:
  # expected_shape is the full shape (batch dimensions then total rows and columns), blocks is a
  # list of dense tensors with already broadcast batch dimensions. For each block it creates
  # zero padding before (columns used so far) and after (remaining columns), concatenates
  # padding and block along the column axis into one row-strip, and finally stacks the strips
  # along the row axis. Returns the dense block-diagonal tensor.
  """Convert a list of blocks, into a dense block diagonal matrix."""
  rows = []
  num_cols = 0
  for block in blocks:
    # Get the batch shape for the block.
    batch_row_shape = array_ops.shape(block)[:-1]

    zeros_to_pad_before_shape = array_ops.concat(
        [batch_row_shape, [num_cols]], axis=-1)
    zeros_to_pad_before = array_ops.zeros(
        shape=zeros_to_pad_before_shape, dtype=block.dtype)
    num_cols += array_ops.shape(block)[-1]
    zeros_to_pad_after_shape = array_ops.concat(
        [batch_row_shape, [expected_shape[-1] - num_cols]], axis=-1)
    zeros_to_pad_after = array_ops.zeros(
        zeros_to_pad_after_shape, dtype=block.dtype)

    rows.append(array_ops.concat(
        [zeros_to_pad_before, block, zeros_to_pad_after], axis=-1))

  return array_ops.concat(rows, axis=-2)


@test_util.run_all_in_graph_and_eager_modes
class SquareLinearOperatorBlockDiagTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Tests the block-diagonal operator built from square blocks. Inherits the generic square
  # operator test suite from the shared utilities; operator_and_matrix supplies the operator and
  # dense reference. Disables TensorFloat-32 per test and relaxes float32/complex64 tolerances.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture teardown: restores the global TensorFloat-32 execution setting saved in setUp.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the non-square test class, this setUp does not call the base class
    # setUp, so any base-class fixture initialization is skipped.
    # Fixture setup: saves and then disables global TensorFloat-32 execution (restored in
    # tearDown) and raises absolute and relative tolerances for float32 and complex64 to 1e-4.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Increase from 1e-6 to 1e-4
    self._atol[dtypes.float32] = 1e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._rtol[dtypes.complex64] = 1e-4

  @staticmethod
  def optional_tests():
    # [FUNCTION DOC]
    # Hook for the shared test base class. Takes no inputs; returns a list of names of optional
    # generated tests to enable: matmul with the same operator type and solve with the same type.
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Hook for the shared test base class. Takes no inputs; returns a list of shape descriptions
    # for the operators to test: empty, one-by-one, batched, and several with explicit lists of
    # per-block shapes (including batch dimensions that must broadcast).
    shape_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shape_info((0, 0)),
        shape_info((1, 1)),
        shape_info((1, 3, 3)),
        shape_info((5, 5), blocks=[(2, 2), (3, 3)]),
        shape_info((3, 7, 7), blocks=[(1, 2, 2), (3, 2, 2), (1, 3, 3)]),
        shape_info((2, 1, 5, 5), blocks=[(2, 1, 2, 2), (1, 3, 3)]),
    ]

  @staticmethod
  def use_blockwise_arg():
    # [FUNCTION DOC]
    # Hook for the shared test base class. Takes no inputs and always returns true, telling the
    # shared tests to also exercise blockwise (list-of-blocks) arguments.
    return True

  def operator_and_matrix(
      self, shape_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Hook for the shared test base class. Inputs: shape_info (target shape, with optional list
    # of block shapes, default a single block), dtype, use_placeholder (hide static shapes), and
    # ensure_self_adjoint_and_pd (default false; when true blocks are flagged self-adjoint and
    # positive definite). Draws a well-conditioned random positive definite matrix per block,
    # builds a block-diagonal operator from full-matrix operators, asserts it was auto-flagged
    # square, broadcasts batch dimensions of the blocks and builds the dense reference with the
    # block-diagonal helper (setting its static shape when not using placeholders). Returns the
    # pair of operator and dense matrix.
    shape = list(shape_info.shape)
    expected_blocks = (
        shape_info.__dict__["blocks"] if "blocks" in shape_info.__dict__
        else [shape])
    matrices = [
        linear_operator_test_util.random_positive_definite_matrix(
            block_shape, dtype, force_well_conditioned=True)
        for block_shape in expected_blocks
    ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          array_ops.placeholder_with_default(
              matrix, shape=None) for matrix in matrices]

    operator = block_diag.LinearOperatorBlockDiag(
        [linalg.LinearOperatorFullMatrix(
            l,
            is_square=True,
            is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
            is_positive_definite=True if ensure_self_adjoint_and_pd else None)
         for l in lin_op_matrices])

    # Should be auto-set.
    self.assertTrue(operator.is_square)

    # Broadcast the shapes.
    expected_shape = list(shape_info.shape)

    matrices = linear_operator_util.broadcast_matrix_batch_dims(matrices)

    block_diag_dense = _block_diag_dense(expected_shape, matrices)

    if not use_placeholder:
      block_diag_dense.set_shape(
          expected_shape[:-2] + [expected_shape[-1], expected_shape[-1]])

    return operator, block_diag_dense

  def test_is_x_flags(self):
    # [FUNCTION DOC]
    # Test scenario: builds a block-diagonal operator from one full-matrix block, passing
    # positive-definite and non-singular true and self-adjoint false. Asserts the operator
    # reports these flags.
    # Matrix with two positive eigenvalues, 1, and 1.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = block_diag.LinearOperatorBlockDiag(
        [linalg.LinearOperatorFullMatrix(matrix)],
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_is_x_parameters(self):
    # [FUNCTION DOC]
    # Test scenario: builds a block-diagonal operator from one full-matrix block with explicit
    # flags. Asserts the operator's parameters mapping holds the name (nothing), square flag
    # (true, auto-set), the supplied flags and the list of sub-operators, and that the
    # sub-operator's parameters hold its own defaults, the matrix and its default class name.
    matrix = [[1., 0.], [1., 1.]]
    sub_operator = linalg.LinearOperatorFullMatrix(matrix)
    operator = block_diag.LinearOperatorBlockDiag(
        [sub_operator],
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertEqual(
        operator.parameters,
        {
            "name": None,
            "is_square": True,
            "is_positive_definite": True,
            "is_self_adjoint": False,
            "is_non_singular": True,
            "operators": [sub_operator],
        })
    self.assertEqual(
        sub_operator.parameters,
        {
            "is_non_singular": None,
            "is_positive_definite": None,
            "is_self_adjoint": None,
            "is_square": None,
            "matrix": matrix,
            "name": "LinearOperatorFullMatrix",
        })

  def test_block_diag_adjoint_type(self):
    # [FUNCTION DOC]
    # Test scenario: block-diagonal operator of two identity full-matrix blocks flagged
    # non-singular. Asserts its adjoint is again a block-diagonal operator with two blocks.
    matrix = [[1., 0.], [0., 1.]]
    operator = block_diag.LinearOperatorBlockDiag(
        [
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_non_singular=True,
            ),
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_non_singular=True,
            ),
        ],
        is_non_singular=True,
    )
    adjoint = operator.adjoint()
    self.assertIsInstance(
        adjoint,
        block_diag.LinearOperatorBlockDiag)
    self.assertEqual(2, len(adjoint.operators))

  def test_block_diag_cholesky_type(self):
    # [FUNCTION DOC]
    # Test scenario: block-diagonal operator of two identity blocks flagged positive definite and
    # self-adjoint. Asserts the Cholesky factor is a block-diagonal operator with two blocks,
    # each a lower-triangular operator.
    matrix = [[1., 0.], [0., 1.]]
    operator = block_diag.LinearOperatorBlockDiag(
        [
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_positive_definite=True,
                is_self_adjoint=True,
            ),
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_positive_definite=True,
                is_self_adjoint=True,
            ),
        ],
        is_positive_definite=True,
        is_self_adjoint=True,
    )
    cholesky_factor = operator.cholesky()
    self.assertIsInstance(
        cholesky_factor,
        block_diag.LinearOperatorBlockDiag)
    self.assertEqual(2, len(cholesky_factor.operators))
    self.assertIsInstance(
        cholesky_factor.operators[0],
        lower_triangular.LinearOperatorLowerTriangular)
    self.assertIsInstance(
        cholesky_factor.operators[1],
        lower_triangular.LinearOperatorLowerTriangular
    )

  def test_block_diag_inverse_type(self):
    # [FUNCTION DOC]
    # Test scenario: block-diagonal operator of two identity blocks flagged non-singular.
    # Asserts its inverse is a block-diagonal operator with two blocks.
    matrix = [[1., 0.], [0., 1.]]
    operator = block_diag.LinearOperatorBlockDiag(
        [
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_non_singular=True,
            ),
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_non_singular=True,
            ),
        ],
        is_non_singular=True,
    )
    inverse = operator.inverse()
    self.assertIsInstance(
        inverse,
        block_diag.LinearOperatorBlockDiag)
    self.assertEqual(2, len(inverse.operators))

  def test_block_diag_matmul_type(self):
    # [FUNCTION DOC]
    # Test scenario: two non-square block-diagonal operators with four blocks each, whose blocks
    # have compatible sizes (two-by-i and i-by-three for i from one to four). Multiplies them as
    # operators. Asserts the product is a block-diagonal operator and its dense form equals the
    # product of the dense forms. Uses random inputs.
    matrices1 = []
    matrices2 = []
    for i in range(1, 5):
      matrices1.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_normal(
              [2, i], dtype=dtypes.float32)))

      matrices2.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_normal(
              [i, 3], dtype=dtypes.float32)))

    operator1 = block_diag.LinearOperatorBlockDiag(matrices1, is_square=False)
    operator2 = block_diag.LinearOperatorBlockDiag(matrices2, is_square=False)

    expected_matrix = math_ops.matmul(
        operator1.to_dense(), operator2.to_dense())
    actual_operator = operator1.matmul(operator2)

    self.assertIsInstance(
        actual_operator, block_diag.LinearOperatorBlockDiag)
    actual_, expected_ = self.evaluate([
        actual_operator.to_dense(), expected_matrix])
    self.assertAllClose(actual_, expected_)

  def test_block_diag_matmul_raises(self):
    # [FUNCTION DOC]
    # Test scenario: multiplies a four-block block-diagonal operator by a plain full-matrix
    # operator whose row count (fifteen) does not match its column total. Asserts a value error
    # saying the operators are incompatible.
    matrices1 = []
    for i in range(1, 5):
      matrices1.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_normal(
              [2, i], dtype=dtypes.float32)))
    operator1 = block_diag.LinearOperatorBlockDiag(matrices1, is_square=False)
    operator2 = linalg.LinearOperatorFullMatrix(
        linear_operator_test_util.random_normal(
            [15, 3], dtype=dtypes.float32))

    with self.assertRaisesRegex(ValueError, "Operators are incompatible"):
      operator1.matmul(operator2)

  def test_block_diag_solve_type(self):
    # [FUNCTION DOC]
    # Test scenario: a block-diagonal operator of well-conditioned lower-triangular blocks (sizes
    # one to four) solved against a non-square block-diagonal operator with matching block
    # rows. Asserts the solve result is a block-diagonal operator and its dense form equals the
    # dense solve of the dense forms.
    matrices1 = []
    matrices2 = []
    for i in range(1, 5):
      matrices1.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_tril_matrix(
              [i, i],
              dtype=dtypes.float32,
              force_well_conditioned=True)))

      matrices2.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_normal(
              [i, 3], dtype=dtypes.float32)))

    operator1 = block_diag.LinearOperatorBlockDiag(matrices1)
    operator2 = block_diag.LinearOperatorBlockDiag(matrices2, is_square=False)

    expected_matrix = linalg.solve(
        operator1.to_dense(), operator2.to_dense())
    actual_operator = operator1.solve(operator2)

    self.assertIsInstance(
        actual_operator, block_diag.LinearOperatorBlockDiag)
    actual_, expected_ = self.evaluate([
        actual_operator.to_dense(), expected_matrix])
    self.assertAllClose(actual_, expected_)

  def test_block_diag_solve_raises(self):
    # [FUNCTION DOC]
    # Test scenario: solves a four-block square block-diagonal operator against a full-matrix
    # operator with fifteen rows (incompatible with the blocks). Asserts a value error saying
    # the operators are incompatible.
    matrices1 = []
    for i in range(1, 5):
      matrices1.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_normal(
              [i, i], dtype=dtypes.float32)))
    operator1 = block_diag.LinearOperatorBlockDiag(matrices1)
    operator2 = linalg.LinearOperatorFullMatrix(
        linear_operator_test_util.random_normal(
            [15, 3], dtype=dtypes.float32))

    with self.assertRaisesRegex(ValueError, "Operators are incompatible"):
      operator1.solve(operator2)

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Test scenario: block-diagonal operator whose four blocks wrap trainable variables holding
    # random positive definite matrices, flagged self-adjoint and positive definite. Runs the
    # shared gradient-tape safety check.
    matrices = []
    for _ in range(4):
      matrices.append(variables_module.Variable(
          linear_operator_test_util.random_positive_definite_matrix(
              [2, 2], dtype=dtypes.float32, force_well_conditioned=True)))

    operator = block_diag.LinearOperatorBlockDiag(
        [linalg.LinearOperatorFullMatrix(
            matrix, is_self_adjoint=True,
            is_positive_definite=True) for matrix in matrices],
        is_self_adjoint=True,
        is_positive_definite=True,
    )
    self.check_tape_safe(operator)

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Test scenario: block-diagonal operator built over three variable-backed positive definite
    # blocks. Initializes the operator's variables in a cached session, then runs the shared
    # check that converting the operator's variables to tensors preserves behavior.
    matrices = []
    for _ in range(3):
      matrices.append(variables_module.Variable(
          linear_operator_test_util.random_positive_definite_matrix(
              [3, 3], dtype=dtypes.float32, force_well_conditioned=True)))

    operator = block_diag.LinearOperatorBlockDiag(
        [linalg.LinearOperatorFullMatrix(
            matrix, is_self_adjoint=True,
            is_positive_definite=True) for matrix in matrices],
        is_self_adjoint=True,
        is_positive_definite=True,
    )
    with self.cached_session() as sess:
      sess.run([x.initializer for x in operator.variables])
      self.check_convert_variables_to_tensors(operator)

  def test_composite_gradients(self):
    # [FUNCTION DOC]
    # Test scenario: under a gradient tape, builds a block-diagonal operator from an identity
    # full-matrix operator and a diagonal operator, both watched, and differentiates the result
    # of multiplying only the first block by a test input with respect to both blocks and the
    # composite operator. Asserts the first block's gradient equals the composite gradient's
    # first component, the composite's second component is all zeros, and the separately
    # watched second block's gradient is nothing (disconnected).
    with backprop.GradientTape() as tape:
      op1 = linalg.LinearOperatorFullMatrix([[1., 0.], [0., 1.]])
      op2 = linalg.LinearOperatorDiag([1., 2., 3.])
      tape.watch([op1, op2])
      operator = block_diag.LinearOperatorBlockDiag([op1, op2])

      x = self.make_x(op1, adjoint=False)
      y = op1.matmul(x)
      connected_grad, disconnected_grad, composite_grad = tape.gradient(
          y, [op1, op2, operator]
      )

    disconnected_component_grad = composite_grad.operators[1].to_dense()
    self.assertAllClose(connected_grad.to_dense(),
                        composite_grad.operators[0].to_dense())
    self.assertAllClose(disconnected_component_grad,
                        array_ops.zeros_like(disconnected_component_grad))
    self.assertIsNone(disconnected_grad)

  def test_is_non_singular_auto_set(self):
    # [FUNCTION DOC]
    # Test scenario: block-diagonal operator of two non-singular blocks with positive-definite
    # false and non-singular unspecified. Asserts positive definite is false and non-singular is
    # auto-set to true. Then asserts that explicitly claiming non-singular false raises a value
    # error saying the operator is always non-singular.
    # Matrix with two positive eigenvalues, 11 and 8.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)

    operator = block_diag.LinearOperatorBlockDiag(
        [operator_1, operator_2],
        is_positive_definite=False,  # No reason it HAS to be False...
        is_non_singular=None)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)

    with self.assertRaisesRegex(ValueError, "always non-singular"):
      block_diag.LinearOperatorBlockDiag(
          [operator_1, operator_2], is_non_singular=False)

  def test_name(self):
    # [FUNCTION DOC]
    # Test scenario: two named blocks ("left" and "right") combined without an explicit name.
    # Asserts the operator's default name joins the block names with a separator.
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, name="left")
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, name="right")

    operator = block_diag.LinearOperatorBlockDiag([operator_1, operator_2])

    self.assertEqual("left_ds_right", operator.name)

  def test_different_dtypes_raises(self):
    # [FUNCTION DOC]
    # Test scenario: two full-matrix blocks, one double and one single precision. Asserts that
    # constructing the block-diagonal operator raises a type error about requiring the same dtype.
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3).astype(np.float32))
    ]
    with self.assertRaisesRegex(TypeError, "same dtype"):
      block_diag.LinearOperatorBlockDiag(operators)

  def test_empty_operators_raises(self):
    # [FUNCTION DOC]
    # Test scenario: constructs a block-diagonal operator from an empty list. Asserts a value
    # error saying the list must be non-empty.
    with self.assertRaisesRegex(ValueError, "non-empty"):
      block_diag.LinearOperatorBlockDiag([])

  def test_incompatible_input_blocks_raises(self):
    # [FUNCTION DOC]
    # Test scenario: block-diagonal operator of a four-by-four and a three-by-three block with
    # hidden static shapes, multiplied by a nested list input whose per-block structure is
    # wrong (width four over a batch of two with five columns). Asserts a value error: in eager
    # mode about a dimension not matching, in graph mode about an ambiguous input structure.
    matrix_1 = array_ops.placeholder_with_default(rng.rand(4, 4), shape=None)
    matrix_2 = array_ops.placeholder_with_default(rng.rand(3, 3), shape=None)
    operators = [
        linalg.LinearOperatorFullMatrix(matrix_1, is_square=True),
        linalg.LinearOperatorFullMatrix(matrix_2, is_square=True)
    ]
    operator = block_diag.LinearOperatorBlockDiag(operators)
    x = np.random.rand(2, 4, 5).tolist()
    msg = ("dimension does not match" if context.executing_eagerly()
           else "input structure is ambiguous")
    with self.assertRaisesRegex(ValueError, msg):
      operator.matmul(x)


@test_util.run_all_in_graph_and_eager_modes
class NonSquareLinearOperatorBlockDiagTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Tests the block-diagonal operator built from non-square blocks. Inherits the generic
  # non-square operator test suite from the shared utilities and skips tests that need square
  # operators. Disables TensorFloat-32 per test and relaxes float32/complex64 tolerances.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture teardown: restores the global TensorFloat-32 execution setting saved in setUp.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: saves and then disables global TensorFloat-32 execution (restored in
    # tearDown), raises float32 and complex64 absolute/relative tolerances to 1e-4, then calls
    # the base class setUp.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Increase from 1e-6 to 1e-4
    self._atol[dtypes.float32] = 1e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._rtol[dtypes.complex64] = 1e-4
    super(NonSquareLinearOperatorBlockDiagTest, self).setUp()

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Hook for the shared test base class. Takes no inputs; returns a list of non-square shape
    # descriptions, some with explicit lists of per-block shapes and batch dimensions.
    shape_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shape_info((1, 0)),
        shape_info((1, 2, 3)),
        shape_info((5, 3), blocks=[(2, 1), (3, 2)]),
        shape_info((3, 6, 5), blocks=[(1, 2, 1), (3, 1, 2), (1, 3, 2)]),
        shape_info((2, 1, 5, 2), blocks=[(2, 1, 2, 1), (1, 3, 1)]),
    ]

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Hook for the shared test base class. Takes no inputs; returns the list of generated test
    # names to skip because they only make sense for square operators (Cholesky, condition
    # number, determinant, diagonal part, eigenvalues, inverse, log determinant, solves, trace).
    return [
        "cholesky",
        "cond",
        "det",
        "diag_part",
        "eigvalsh",
        "inverse",
        "log_abs_det",
        "solve",
        "solve_with_broadcast",
        "trace"]

  @staticmethod
  def use_blockwise_arg():
    # [FUNCTION DOC]
    # Hook for the shared test base class. Takes no inputs and always returns true, enabling
    # blockwise (list-of-blocks) argument tests.
    return True

  def operator_and_matrix(
      self, shape_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Hook for the shared test base class. Inputs: shape_info (target shape with optional list
    # of block shapes, default one block), dtype, use_placeholder (hide static shapes), and
    # ensure_self_adjoint_and_pd, which is deleted and ignored since non-square operators cannot
    # be self-adjoint or positive definite. Draws random normal blocks, wraps each in a
    # full-matrix operator flagged non-square, not self-adjoint and not positive definite, and
    # builds the block-diagonal operator. Broadcasts block batch dimensions, builds the dense
    # reference with the block-diagonal helper and sets its static shape unless placeholders
    # are used. Returns the operator and dense matrix pair.
    del ensure_self_adjoint_and_pd
    shape = list(shape_info.shape)
    expected_blocks = (
        shape_info.__dict__["blocks"] if "blocks" in shape_info.__dict__
        else [shape])
    matrices = [
        linear_operator_test_util.random_normal(block_shape, dtype=dtype)
        for block_shape in expected_blocks
    ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          array_ops.placeholder_with_default(
              matrix, shape=None) for matrix in matrices]

    blocks = []
    for l in lin_op_matrices:
      blocks.append(
          linalg.LinearOperatorFullMatrix(
              l,
              is_square=False,
              is_self_adjoint=False,
              is_positive_definite=False))
    operator = block_diag.LinearOperatorBlockDiag(blocks)

    # Broadcast the shapes.
    expected_shape = list(shape_info.shape)

    matrices = linear_operator_util.broadcast_matrix_batch_dims(matrices)

    block_diag_dense = _block_diag_dense(expected_shape, matrices)

    if not use_placeholder:
      block_diag_dense.set_shape(expected_shape)

    return operator, block_diag_dense


if __name__ == "__main__":
  linear_operator_test_util.add_tests(SquareLinearOperatorBlockDiagTest)
  linear_operator_test_util.add_tests(NonSquareLinearOperatorBlockDiagTest)
  test.main()
