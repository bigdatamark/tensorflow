# [MODULE DOC]
# Tests for the tridiagonal matrix multiplication op compiled with XLA. Compares JIT-compiled
# results (all three diagonal formats, batched and N-dimensional inputs) with a non-JIT run, and
# checks gradients against a dense-matrix reference. Complex types are not covered. When run as
# a script it enables eager execution before starting the test runner.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tridiagonal_matmul_ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops as random
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import googletest


class TridiagonalMatMulOpsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case (XLA test base) for the tridiagonal matmul op. Holds two wrapped helpers, one
  # compiled with XLA and one not, plus format-sweeping and gradient helpers used by the tests.

  @def_function.function(jit_compile=True)
  def _jit_tridiagonal_matmul(self, diagonals, rhs, diagonals_format):
    # [FUNCTION DOC]
    # HELPER: runs the op under test with XLA compilation turned on.
    # Multiplies a tridiagonal matrix (given in diagonals, in the named diagonals_format) by rhs
    # and returns the product tensor.
    return linalg_impl.tridiagonal_matmul(
        diagonals, rhs, diagonals_format=diagonals_format)

  @def_function.function(jit_compile=False)
  def _tridiagonal_matmul(self, diagonals, rhs, diagonals_format):
    # [FUNCTION DOC]
    # HELPER: runs the same op without XLA compilation, producing the reference result.
    # Same inputs and output as the compiled variant.
    return linalg_impl.tridiagonal_matmul(
        diagonals, rhs, diagonals_format=diagonals_format)

  def _testAllFormats(self,
                      superdiag,
                      maindiag,
                      subdiag,
                      rhs,
                      dtype=dtypes.float64):
    # [FUNCTION DOC]
    # HELPER: shared check used by the small-matrix tests.
    # Inputs: superdiag, maindiag, subdiag as lists of numbers (super and sub have one fewer
    # element than main), rhs as a matrix, and dtype (default 64-bit float). Builds the diagonals
    # in sequence, compact and dense-matrix forms, also batched versions (the second batch entry
    # has the inputs doubled), runs the compiled op for every format, and computes the expected
    # value with the uncompiled op. Asserts every unbatched result is close to the expected value
    # and every batched result is close to the expected batch (second entry scaled by four).
    # Returns nothing.
    superdiag_extended = np.pad(superdiag, [0, 1], 'constant')
    subdiag_extended = np.pad(subdiag, [1, 0], 'constant')

    diags_compact = np.stack([superdiag_extended, maindiag, subdiag_extended])
    diags_matrix = np.diag(superdiag, 1) + np.diag(maindiag, 0) + np.diag(
        subdiag, -1)

    with self.test_scope():
      diags_sequence_op = (constant_op.constant(superdiag_extended, dtype),
                           constant_op.constant(maindiag, dtype),
                           constant_op.constant(subdiag_extended, dtype))
      diags_compact_op = constant_op.constant(diags_compact, dtype)
      diags_matrix_op = constant_op.constant(diags_matrix, dtype)
      rhs_op = constant_op.constant(rhs, dtype)

      rhs_batch_op = array_ops_stack.stack([rhs_op, 2 * rhs_op])
      diags_compact_batch_op = array_ops_stack.stack(
          [diags_compact_op, 2 * diags_compact_op])
      diags_matrix_batch_op = array_ops_stack.stack(
          [diags_matrix_op, 2 * diags_matrix_op])
      diags_sequence_batch_op = [
          array_ops_stack.stack([x, 2 * x]) for x in diags_sequence_op
      ]

      results = [
          self._jit_tridiagonal_matmul(
              diags_sequence_op, rhs_op, diagonals_format='sequence'),
          self._jit_tridiagonal_matmul(
              diags_compact_op, rhs_op, diagonals_format='compact'),
          self._jit_tridiagonal_matmul(
              diags_matrix_op, rhs_op, diagonals_format='matrix')
      ]

      results_batch = [
          self._jit_tridiagonal_matmul(
              diags_sequence_batch_op,
              rhs_batch_op,
              diagonals_format='sequence'),
          self._jit_tridiagonal_matmul(
              diags_compact_batch_op, rhs_batch_op, diagonals_format='compact'),
          self._jit_tridiagonal_matmul(
              diags_matrix_batch_op, rhs_batch_op, diagonals_format='matrix')
      ]

      expected = self._tridiagonal_matmul(
          diags_sequence_op, rhs_op, diagonals_format='sequence')
      expected_batch = np.stack([expected, 4 * expected])
      for result in results:
        self.assertAllClose(result, expected)
      for result in results_batch:
        self.assertAllClose(result, expected_batch)

  def _makeTridiagonalMatrix(self, superdiag, maindiag, subdiag):
    # [FUNCTION DOC]
    # HELPER: builds dense reference matrices for the gradient test.
    # Inputs are batched tensors of the super, main and sub diagonals. Returns a batch of dense
    # square matrices formed by placing each diagonal (padded appropriately) and summing them.
    super_pad = [[0, 0], [0, 1], [1, 0]]
    sub_pad = [[0, 0], [1, 0], [0, 1]]

    super_part = array_ops.pad(array_ops.matrix_diag(superdiag), super_pad)
    main_part = array_ops.matrix_diag(maindiag)
    sub_part = array_ops.pad(array_ops.matrix_diag(subdiag), sub_pad)
    return super_part + main_part + sub_part

  def _gradientTest(self, diags, rhs, dtype=dtypes.float64):
    # [FUNCTION DOC]
    # HELPER: shared gradient check. Inputs: diags (compact-format diagonals, nested lists),
    # rhs (right-hand side), dtype (default 64-bit float). Computes gradients of the op under
    # test and of a dense-matrix reference product, in the XLA test scope, and asserts that the
    # theoretical and numerical gradients agree and that the theoretical gradient matches the
    # reference. Returns nothing.

    def reference_matmul(diags, rhs):
      # [FUNCTION DOC]
      # HELPER: nested reference implementation. Builds a dense tridiagonal matrix from the
      # compact diagonals (first row super, second main, third sub) and returns its product
      # with rhs via ordinary matrix multiplication.
      matrix = self._makeTridiagonalMatrix(diags[..., 0, :-1], diags[..., 1, :],
                                           diags[..., 2, 1:])
      return math_ops.matmul(matrix, rhs)

    with self.session(), self.test_scope():
      diags = constant_op.constant(diags, dtype=dtype)
      rhs = constant_op.constant(rhs, dtype=dtype)
      grad_reference, _ = gradient_checker_v2.compute_gradient(
          reference_matmul, [diags, rhs])
      grad_theoretical, grad_numerical = gradient_checker_v2.compute_gradient(
          linalg_impl.tridiagonal_matmul, [diags, rhs])
    self.assertAllClose(grad_theoretical, grad_numerical)
    self.assertAllClose(grad_theoretical, grad_reference)

  def test2x2(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 tridiagonal matrix with a two-column right-hand side. Runs the all-formats
    # comparison with default precision; expects compiled results to match the uncompiled one.
    self._testAllFormats(
        superdiag=[1], maindiag=[2, 3], subdiag=[4], rhs=[[2, 1], [4, 3]])

  def test3x3(self):
    # [FUNCTION DOC]
    # Scenario: a 3x3 tridiagonal matrix with a two-column right-hand side, tried with 32-bit
    # and 64-bit floats. Expects compiled results in all formats to match the uncompiled one.
    for dtype in [dtypes.float32, dtypes.float64]:
      self._testAllFormats(
          superdiag=[1, 2],
          maindiag=[1, 2, 1],
          subdiag=[2, 1],
          rhs=[[1, 1], [2, 2], [3, 3]],
          dtype=dtype)

  def testNDimensional(self):
    # [FUNCTION DOC]
    # Scenario: high-rank batched input (several leading batch dimensions, 7 rows, 5 columns of
    # rhs). Diagonals and rhs are drawn from seeded stateless uniform random generators in the XLA
    # scope. Multiplies in sequence format with and without compilation and asserts the two
    # results are close.
    with self.test_scope():
      shape = [77, 10, 3, 5, 7]
      superdiag = random.stateless_random_uniform(
          shape=shape[:-1], dtype=dtypes.float32, seed=[5, 10])
      maindiag = random.stateless_random_uniform(
          shape=shape[:-1], dtype=dtypes.float32, seed=[5, 11])
      subdiag = random.stateless_random_uniform(
          shape=shape[:-1], dtype=dtypes.float32, seed=[5, 12])
      rhs = random.stateless_random_uniform(
          shape=shape, dtype=dtypes.float32, seed=[5, 13])

      expected = self._tridiagonal_matmul((superdiag, maindiag, subdiag),
                                          rhs,
                                          diagonals_format='sequence')

      real = self._jit_tridiagonal_matmul((superdiag, maindiag, subdiag),
                                          rhs,
                                          diagonals_format='sequence')

      self.assertAllClose(expected, real)

  def testGradientSmall(self):
    # [FUNCTION DOC]
    # Scenario: gradient check on a single 3x3 batch entry with a 3x2 right-hand side in 64-bit
    # floats. Delegates to the gradient helper, which compares theoretical, numerical and
    # dense-reference gradients.
    self._gradientTest([[[1, 2, 0], [1, 2, 3], [0, 1, 2]]],
                       [[[1, 2], [3, 4], [5, 6]]],
                       dtype=dtypes.float64)

  # testComplex is skipped as complex type is not yet supported.


if __name__ == '__main__':
  ops.enable_eager_execution()
  googletest.main()
