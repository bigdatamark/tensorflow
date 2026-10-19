# [MODULE DOC]
# Test module for the Toeplitz linear operator in the linalg package. Defines one test class
# derived from the shared square linear-operator test base class, which builds a Toeplitz
# operator from random first-column and first-row vectors and compares it with a dense matrix
# built by the scientific-computing library's Toeplitz constructor, with relaxed tolerances and
# some generic tests skipped (solve-related ones, Cholesky, condition number, inverse). Also
# tests argument validation, gradient tape safety (including explicit checks for the diagonal
# part and trace) and variable conversion. Defines a module-level alias for the private
# complex-conversion helper of the operator module (not used in this file). Requires the
# scientific-computing library at import time. When run as a script it registers the generated
# base-class tests and starts the test runner.
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

import contextlib

import numpy as np
import scipy.linalg

from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import linear_operator_toeplitz
from tensorflow.python.platform import test

linalg = linalg_lib

_to_complex = linear_operator_toeplitz._to_complex


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorToeplitzTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the Toeplitz linear operator. Inherits generic square-operator checks from the
  # shared base class, relaxes tolerances for all four float and complex dtypes, skips the
  # Cholesky, condition number, inverse and solve tests (solve is not specialized yet), supplies
  # the shapes to test and the operator/matrix factory, and adds tests for validation, tape
  # safety and variable conversion. Runs in graph and eager modes.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @contextlib.contextmanager
  def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
    # [FUNCTION DOC]
    # HELPER: device-selection context manager used by the test framework's session handling.
    # KNOWN ISSUE: the docstring claims the FFT operation mapping is overwritten for testing, but
    # the body only delegates to the parent test case's implementation with the same arguments
    # and yields its session; no mapping is overwritten.
    # Context manager (generator). Inputs: sess (session), use_gpu and force_gpu (booleans).
    # Enters the parent test-case context for constraining devices and setting the default
    # session, and yields the session it provides; the context is left when the caller's block
    # ends.
    """We overwrite the FFT operation mapping for testing."""
    with test.TestCase._constrain_devices_and_set_default(
        self, sess, use_gpu, force_gpu) as sess:
      yield sess

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: relaxes the absolute and relative comparison tolerances stored by the
    # base class: 1e-4 for float32 and complex64, 1e-9 for float64 and complex128, until
    # specialized solve and determinant routines exist (see the to-do). Does not touch the
    # TensorFloat-32 setting. Returns nothing.
    # TODO(srvasude): Lower these tolerances once specialized solve and
    # determinants are implemented.
    self._atol[dtypes.float32] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._atol[dtypes.float64] = 1e-9
    self._rtol[dtypes.float64] = 1e-9
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.complex64] = 1e-4
    self._atol[dtypes.complex128] = 1e-9
    self._rtol[dtypes.complex128] = 1e-9

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the names of generated tests to skip: Cholesky,
    # condition number, inverse, solve and solve with broadcasting. Solve is skipped because it
    # currently exercises the generic base-class implementation, which could be more stable (see
    # the to-do about enabling these once solve is specialized).
    # Skip solve tests, as these could have better stability
    # (currently exercises the base class).
    # TODO(srvasude): Enable these when solve is implemented.
    return ["cholesky", "cond", "inverse", "solve", "solve_with_broadcast"]

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the list of operator shape descriptions to test, all
    # square: 1 by 1, batched 1 by 6 by 6, batched 3 by 4 by 4 and 2 by 1 by 3 by 3.
    shape_info = linear_operator_test_util.OperatorShapesInfo
    # non-batch operators (n, n) and batch operators.
    return [
        shape_info((1, 1)),
        shape_info((1, 6, 6)),
        shape_info((3, 4, 4)),
        shape_info((2, 1, 3, 3))
    ]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the self-adjoint positive-definite branch the decreasing sequence from ten
    # to one assigned to the row is immediately overwritten by the random column (the row is set
    # equal to the column), so that sequence is dead code, and the resulting symmetric Toeplitz
    # matrix built from random positive values is not guaranteed to be positive definite despite
    # the comment citing the decreasing-sequence result.
    # Factory hook used by the base class. Inputs: build_info (its shape is the full
    # batch-plus-matrix shape), dtype, use_placeholder (feed row and column through placeholders
    # of unknown shape) and ensure_self_adjoint_and_pd (make row equal to column so the matrix is
    # symmetric, and flag the operator self-adjoint and positive definite). Draws random column
    # and row vectors with entries between 1 and 5 (using the global numpy random state, so
    # unseeded here), forces their first entries to match (required for a valid Toeplitz matrix),
    # casts them to the dtype, and builds the Toeplitz operator. The dense reference is built
    # by flattening batch dimensions, creating each matrix with the scientific library's Toeplitz
    # function from the column and row, and reshaping back and casting to the dtype. Returns
    # (operator, dense matrix).
    shape = list(build_info.shape)
    row = np.random.uniform(low=1., high=5., size=shape[:-1])
    col = np.random.uniform(low=1., high=5., size=shape[:-1])

    # Make sure first entry is the same
    row[..., 0] = col[..., 0]

    if ensure_self_adjoint_and_pd:
      # Note that a Toeplitz matrix generated from a linearly decreasing
      # non-negative sequence is positive definite. See
      # https://www.math.cinvestav.mx/~grudsky/Papers/118_29062012_Albrecht.pdf
      # for details.
      row = np.linspace(start=10., stop=1., num=shape[-1])

      # The entries for the first row and column should be the same to guarantee
      # symmetric.
      row = col

    lin_op_row = math_ops.cast(row, dtype=dtype)
    lin_op_col = math_ops.cast(col, dtype=dtype)

    if use_placeholder:
      lin_op_row = array_ops.placeholder_with_default(
          lin_op_row, shape=None)
      lin_op_col = array_ops.placeholder_with_default(
          lin_op_col, shape=None)

    operator = linear_operator_toeplitz.LinearOperatorToeplitz(
        row=lin_op_row,
        col=lin_op_col,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None)

    flattened_row = np.reshape(row, (-1, shape[-1]))
    flattened_col = np.reshape(col, (-1, shape[-1]))
    flattened_toeplitz = np.zeros(
        [flattened_row.shape[0], shape[-1], shape[-1]])
    for i in range(flattened_row.shape[0]):
      flattened_toeplitz[i] = scipy.linalg.toeplitz(
          flattened_col[i],
          flattened_row[i])
    matrix = np.reshape(flattened_toeplitz, shape)
    matrix = math_ops.cast(matrix, dtype=dtype)

    return operator, matrix

  def test_scalar_row_col_raises(self):
    # [FUNCTION DOC]
    # Test: constructing the Toeplitz operator with a scalar for the row, the column or both
    # raises a value error saying the argument must have at least 1 dimension. Three cases:
    # scalar column and row, vector column with scalar row, scalar column with vector row.
    with self.assertRaisesRegex(ValueError, "must have at least 1 dimension"):
      linear_operator_toeplitz.LinearOperatorToeplitz(1., 1.)

    with self.assertRaisesRegex(ValueError, "must have at least 1 dimension"):
      linear_operator_toeplitz.LinearOperatorToeplitz([1.], 1.)

    with self.assertRaisesRegex(ValueError, "must have at least 1 dimension"):
      linear_operator_toeplitz.LinearOperatorToeplitz(1., [1.])

  def test_tape_safe(self):
    # [FUNCTION DOC]
    # Test: builds a self-adjoint, positive-definite 1 by 1 Toeplitz operator from two trainable
    # variables (column and row) and runs the shared gradient-tape safety check, skipping the
    # diagonal part and trace checks because they depend only on the column. Those two are then
    # tested explicitly: under a gradient tape, the gradient of the diagonal part and of the
    # trace with respect to the column variable must not be nothing.
    col = variables_module.Variable([1.])
    row = variables_module.Variable([1.])
    operator = linear_operator_toeplitz.LinearOperatorToeplitz(
        col, row, is_self_adjoint=True, is_positive_definite=True)
    self.check_tape_safe(
        operator,
        skip_options=[
            # .diag_part, .trace depend only on `col`, so test explicitly below.
            linear_operator_test_util.CheckTapeSafeSkipOptions.DIAG_PART,
            linear_operator_test_util.CheckTapeSafeSkipOptions.TRACE,
        ])

    with backprop.GradientTape() as tape:
      self.assertIsNotNone(tape.gradient(operator.diag_part(), col))

    with backprop.GradientTape() as tape:
      self.assertIsNotNone(tape.gradient(operator.trace(), col))

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Test: builds a self-adjoint, positive-definite Toeplitz operator from two variables,
    # initializes all of the operator's variables in a session and runs the shared check that
    # converting the operator's variables to tensors yields an equivalent operator.
    col = variables_module.Variable([1.])
    row = variables_module.Variable([1.])
    operator = linear_operator_toeplitz.LinearOperatorToeplitz(
        col, row, is_self_adjoint=True, is_positive_definite=True)
    with self.cached_session() as sess:
      sess.run([x.initializer for x in operator.variables])
      self.check_convert_variables_to_tensors(operator)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorToeplitzTest)
  test.main()
