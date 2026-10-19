# [MODULE DOC]
# Unit tests and benchmark for linalg_impl.tridiagonal_matmul, which multiplies a tridiagonal
# matrix by a dense matrix. Tests cover the three diagonal storage formats (sequence of three
# diagonals, compact three-row form, full matrix), real and complex dtypes, batches, gradients
# compared to a reference dense multiplication, and shape error messages of the raw op in eager
# mode on GPU. A benchmark compares against a baseline implementation. No import-time side
# effects; running as a script starts the test runner.
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
"""Tests for tensorflow.ops.linalg.linalg_impl.tridiagonal_matmul."""

import itertools

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class TridiagonalMulOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for tridiagonal matrix multiplication. Helper methods build inputs in each storage
  # format and compare results; also (nested) holds a benchmark class.

  def _testAllFormats(self,
                      superdiag,
                      maindiag,
                      subdiag,
                      rhs,
                      expected,
                      dtype=dtypes.float64):
    # [FUNCTION DOC]
    # HELPER: shared checker for the value tests. Inputs: the super-, main- and sub-diagonals
    # (sequences of numbers; super and sub are one shorter than main), a right-hand side matrix,
    # the expected product, and a TensorFlow dtype (default float64). Builds the diagonals in
    # sequence, compact and full-matrix formats, then also a batch of two where the second entry
    # is the first scaled by two (so its expected product is four times larger). Multiplies in
    # all three formats for both the single and batched inputs, evaluates in a session and asserts
    # every result is close to the expected value. Returns nothing.
    superdiag_extended = np.pad(superdiag, [0, 1], 'constant')
    subdiag_extended = np.pad(subdiag, [1, 0], 'constant')
    diags_compact = np.stack([superdiag_extended, maindiag, subdiag_extended])
    diags_matrix = np.diag(superdiag, 1) + np.diag(maindiag, 0) + np.diag(
        subdiag, -1)

    diags_sequence = (constant_op.constant(superdiag_extended, dtype),
                      constant_op.constant(maindiag, dtype),
                      constant_op.constant(subdiag_extended, dtype))
    diags_compact = constant_op.constant(diags_compact, dtype)
    diags_matrix = constant_op.constant(diags_matrix, dtype)
    rhs = constant_op.constant(rhs, dtype)

    rhs_batch = array_ops_stack.stack(
        [rhs, 2 * rhs])
    diags_compact_batch = array_ops_stack.stack(
        [diags_compact, 2 * diags_compact])
    diags_matrix_batch = array_ops_stack.stack(
        [diags_matrix, 2 * diags_matrix])
    diags_sequence_batch = [array_ops_stack.stack(
        [x, 2 * x]) for x in diags_sequence]

    results = [
        linalg_impl.tridiagonal_matmul(
            diags_sequence, rhs, diagonals_format='sequence'),
        linalg_impl.tridiagonal_matmul(
            diags_compact, rhs, diagonals_format='compact'),
        linalg_impl.tridiagonal_matmul(
            diags_matrix, rhs, diagonals_format='matrix')
    ]
    results_batch = [
        linalg_impl.tridiagonal_matmul(
            diags_sequence_batch, rhs_batch, diagonals_format='sequence'),
        linalg_impl.tridiagonal_matmul(
            diags_compact_batch, rhs_batch, diagonals_format='compact'),
        linalg_impl.tridiagonal_matmul(
            diags_matrix_batch, rhs_batch, diagonals_format='matrix')
    ]

    with self.cached_session():
      results = self.evaluate(results)
      results_batch = self.evaluate(results_batch)

    expected = np.array(expected)
    expected_batch = np.stack([expected, 4 * expected])
    for result in results:
      self.assertAllClose(result, expected)
    for result in results_batch:
      self.assertAllClose(result, expected_batch)

  def _makeTridiagonalMatrix(self, superdiag, maindiag, subdiag):
    # [FUNCTION DOC]
    # HELPER: builds dense batched tridiagonal matrices from batched diagonals. Inputs: super-,
    # main- and sub-diagonal tensors (batch by length; super and sub one shorter). Places each
    # diagonal on its matrix offset (padding to square) and sums them. Returns the dense tensor.
    super_pad = [[0, 0], [0, 1], [1, 0]]
    sub_pad = [[0, 0], [1, 0], [0, 1]]

    super_part = array_ops.pad(array_ops.matrix_diag(superdiag), super_pad)
    main_part = array_ops.matrix_diag(maindiag)
    sub_part = array_ops.pad(array_ops.matrix_diag(subdiag), sub_pad)
    return super_part + main_part + sub_part

  def _randomComplexArray(self, shape):
    # [FUNCTION DOC]
    # HELPER: returns a complex numpy array of the given shape whose real and imaginary parts are
    # uniform in minus ten to ten. Seeds numpy's global random state with a fixed value on every
    # call, so repeated calls with the same shape return identical arrays.
    np.random.seed(43)
    return (np.random.uniform(-10, 10, shape) +
            np.random.uniform(-10, 10, shape) * 1j)

  def _gradientTest(self, diags, rhs, dtype=dtypes.float64):
    # [FUNCTION DOC]
    # HELPER: shared checker for gradient tests. Inputs: compact diagonals array (batch by three
    # by length), right-hand side array, and dtype (default float64). Computes the gradient of
    # the tridiagonal multiplication numerically and theoretically, and the numerical gradient of
    # a reference dense multiplication, then asserts the theoretical gradient matches both.
    # Returns nothing.

    def reference_matmul(diags, rhs):
      # [FUNCTION DOC]
      # HELPER: nested reference implementation. Builds the dense tridiagonal matrix from the
      # compact diagonals (first row without its last entry, middle row, last row without its
      # first entry) and returns its ordinary matrix product with rhs.
      matrix = self._makeTridiagonalMatrix(diags[..., 0, :-1], diags[..., 1, :],
                                           diags[..., 2, 1:])
      return math_ops.matmul(matrix, rhs)

    diags = constant_op.constant(diags, dtype=dtype)
    rhs = constant_op.constant(rhs, dtype=dtype)
    with self.cached_session():
      grad_reference, _ = gradient_checker_v2.compute_gradient(
          reference_matmul, [diags, rhs])
      grad_theoretical, grad_numerical = gradient_checker_v2.compute_gradient(
          linalg_impl.tridiagonal_matmul, [diags, rhs])
    self.assertAllClose(grad_theoretical, grad_numerical)
    self.assertAllClose(grad_theoretical, grad_reference)

  def test2x2(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 tridiagonal matrix times a 2x2 right-hand side, checked in every storage
    # format, single and batched. Expects the known product.
    self._testAllFormats([1], [2, 3], [4], [[2, 1], [4, 3]], [[8, 5], [20, 13]])

  def test3x3(self):
    # [FUNCTION DOC]
    # Scenario: a 3x3 tridiagonal matrix times a 3x2 right-hand side in float32 and float64, in
    # every storage format. Expects the known product.
    for dtype in [dtypes.float32, dtypes.float64]:
      self._testAllFormats([1, 2], [1, 2, 1], [2, 1], [[1, 1], [2, 2], [3, 3]],
                           [[3, 3], [12, 12], [5, 5]],
                           dtype=dtype)

  def testComplex(self):
    # [FUNCTION DOC]
    # Scenario: a 3x3 complex tridiagonal matrix times a complex 3x2 right-hand side in complex64
    # and complex128, in every storage format. Expects the known complex product.
    for dtype in [dtypes.complex64, dtypes.complex128]:
      self._testAllFormats([1j, 1j], [1, -1, 0], [1j, 1j],
                           np.array([[1, 1j], [1, 1j], [1, 1j]]),
                           [[1 + 1j, -1 + 1j], [-1 + 2j, -2 - 1j], [1j, -1]],
                           dtype=dtype)

  def testBatch(self):
    # [FUNCTION DOC]
    # Scenario: twenty random complex tridiagonal 10x10 matrices (full-matrix format) times random
    # 10x15 right-hand sides. Expects the product to match numpy's batched matrix multiplication.
    b = 20
    m = 10
    n = 15
    superdiag = self._randomComplexArray((b, m - 1))
    maindiag = self._randomComplexArray((b, m))
    subdiag = self._randomComplexArray((b, m - 1))
    rhs = self._randomComplexArray((b, m, n))
    matrix = np.stack([np.diag(superdiag[i], 1) + \
                       np.diag(maindiag[i], 0) + \
                       np.diag(subdiag[i], -1) for i in range(b)])
    expected_result = np.matmul(matrix, rhs)
    result = linalg_impl.tridiagonal_matmul(
        constant_op.constant(matrix, dtype=dtypes.complex128),
        constant_op.constant(rhs, dtype=dtypes.complex128),
        diagonals_format='matrix')

    with self.cached_session():
      result = self.evaluate(result)

    self.assertAllClose(result, expected_result)

  def testGradientSmall(self):
    # [FUNCTION DOC]
    # Scenario: gradient check on a small real batch of one with a 3x3 matrix and 3x2 right-hand
    # side in float64. Expects theoretical gradients to match numerical and reference ones.
    self._gradientTest([[[1, 2, 0], [1, 2, 3], [0, 1, 2]]],
                       [[[1, 2], [3, 4], [5, 6]]],
                       dtype=dtypes.float64)

  def testGradientComplexSmall(self):
    # [FUNCTION DOC]
    # Scenario: gradient check on a small complex128 batch of one with a 3x3 matrix and a 3x2
    # right-hand side. Expects theoretical gradients to match numerical and reference ones.
    self._gradientTest(
        np.array([[[1 + 1j, 2j, 0], [1 + 2j, 2j, 3 + 0j], [0, 1j, 2 + 0j]]]),
        np.array([[[1j, 2 + 0j], [3 + 1j, 4j], [5j, 6 + 3j]]]),
        dtype=dtypes.complex128)

  def testGradientComplexWithBatches(self):
    # [FUNCTION DOC]
    # Scenario: gradient check on a batch of five random complex128 problems with matrix size ten
    # and fifteen right-hand side columns. Expects theoretical gradients to match numerical and
    # reference ones.
    b = 5
    m = 10
    n = 15
    diags = self._randomComplexArray((b, 3, m))
    rhs = self._randomComplexArray((b, m, n))
    self._gradientTest(diags, rhs, dtype=dtypes.complex128)

  def _testErrorWithShapesEager(self, exception_regex, superdiag_shape,
                                maindiag_shape, subdiag_shape, rhs_shape):
    # [FUNCTION DOC]
    # HELPER: error checker for the raw op in eager mode. Inputs: a regular expression for the
    # expected message and shapes for the super-, main-, sub-diagonals and right-hand side. Creates
    # tensors of ones with those shapes and calls the raw tridiagonal multiplication op, expecting
    # an invalid-argument error whose message matches the expression. Returns nothing.
    with context.eager_mode():
      superdiag = array_ops.ones(superdiag_shape)
      maindiag = array_ops.ones(maindiag_shape)
      subdiag = array_ops.ones(subdiag_shape)
      rhs = array_ops.ones(rhs_shape)
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  exception_regex):
        linalg_ops.tridiagonal_mat_mul(superdiag, maindiag, subdiag, rhs)

  def testInvalidShapesEagerGpu(self):
    # [FUNCTION DOC]
    # Scenario: the raw op is called in eager mode with five kinds of malformed shapes (rank below
    # two, mismatched rank, mismatched outer dimensions, wrong second-to-last dimension of the
    # sub-diagonal, wrong last dimension of the sub-diagonal). Expects the corresponding
    # invalid-argument messages. Skipped on ROCm (message differs) and when no GPU is available.
    if test.is_built_with_rocm():
      self.skipTest('Incorrect Regex on rocm')
    if not test.is_gpu_available():
      self.skipTest('Test requires GPU')
    self._testErrorWithShapesEager('Input must have rank >= 2, but got ',
                                   [2], [2], [2], [2])
    self._testErrorWithShapesEager(
        'superdiag must have same rank as rhs, but got 3 and 2',
        [2, 1, 2], [2, 1], [2, 1], [2, 2])
    self._testErrorWithShapesEager(
        'maindiag must have same outer dimensions as rhs, but for index 0, got '
        '3 and 2',
        [2, 1, 2], [3, 1, 2], [2, 1, 2], [2, 2, 2])
    self._testErrorWithShapesEager(
        "subdiag's second-to-last dimension must be 1, but got 3",
        [2, 1, 2], [2, 1, 2], [2, 3, 2], [2, 2, 2])
    self._testErrorWithShapesEager(
        "subdiag's last dimension size must be rhs's second-to-last dimension "
        "size, but got 3 and 2",
        [2, 1, 2], [2, 1, 2], [2, 1, 3], [2, 2, 2])

  # Benchmark
  class TridiagonalMatMulBenchmark(test.Benchmark):
    # [CLASS DOC]
    # KNOWN ISSUE: this benchmark class is nested inside the test case class, so benchmark
    # discovery by the usual module-level scan is unlikely to find or run it.
    # Benchmark comparing tridiagonal multiplication against a baseline built from elementwise
    # operations. The class attribute named sizes lists triples of matrix size, batch size and
    # number of right-hand side columns.
    sizes = [(100000, 1, 1), (1000000, 1, 1), (10000000, 1, 1), (100000, 10, 1),
             (100000, 100, 1), (10000, 1, 100), (10000, 1, 1000),
             (10000, 1, 10000)]

    def baseline(self, upper, diag, lower, vec):
      # [FUNCTION DOC]
      # HELPER: baseline implementation for the benchmark. Inputs: batched upper, main and lower
      # diagonals (batch by length) and a batched dense right-hand side. Multiplies each part
      # elementwise with the matching shifted rows of vec, padding to align, and returns the sum
      # of the lower, main and upper contributions.
      diag_part = array_ops.expand_dims(diag, -1) * vec
      lower_part = array_ops.pad(
          array_ops.expand_dims(lower[:, 1:], -1) * vec[:, :-1, :],
          [[0, 0], [1, 0], [0, 0]])
      upper_part = array_ops.pad(
          array_ops.expand_dims(upper[:, :-1], -1) * vec[:, 1:, :],
          [[0, 0], [0, 1], [0, 0]])
      return lower_part + diag_part + upper_part

    def _generateData(self, batch_size, m, n, seed=42):
      # [FUNCTION DOC]
      # HELPER: creates benchmark inputs. Inputs: batch size, matrix size m, number of right-hand
      # side columns n, and random seed (default 42, applied to numpy's global state). Draws
      # normal random data of shape batch, m, three plus n, and returns four float64 variables:
      # the upper, main and lower diagonals and the dense right-hand side.
      np.random.seed(seed)
      data = np.random.normal(size=(batch_size, m, 3 + n))
      return (variables.Variable(data[:, :, 0], dtype=dtypes.float64),
              variables.Variable(data[:, :, 1], dtype=dtypes.float64),
              variables.Variable(data[:, :, 2], dtype=dtypes.float64),
              variables.Variable(data[:, :, 3:], dtype=dtypes.float64))

    def benchmarkTridiagonalMulOp(self):
      # [FUNCTION DOC]
      # For each device (CPU, plus GPU if a CUDA GPU is available) and each configured size, builds
      # a graph with generated variable inputs, the baseline result and the tridiagonal multiply
      # op in sequence format, initializes variables, and reports two benchmark runs (baseline and
      # op, at least ten iterations, memory usage not stored) named by device and sizes. Side
      # effects: creates graphs and sessions and emits benchmark results. Returns nothing.
      devices = [('/cpu:0', 'cpu')]
      if test.is_gpu_available(cuda_only=True):
        devices += [('/gpu:0', 'gpu')]

      for device_option, size_option in itertools.product(devices, self.sizes):
        device_id, device_name = device_option
        m, batch_size, n = size_option

        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device(device_id):
          upper, diag, lower, vec = self._generateData(batch_size, m, n)
          x1 = self.baseline(upper, diag, lower, vec)
          x2 = linalg_impl.tridiagonal_matmul((upper, diag, lower),
                                              vec,
                                              diagonals_format='sequence')

          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(x1),
              min_iters=10,
              store_memory_usage=False,
              name=('tridiagonal_matmul_baseline_%s'
                    '_batch_size_%d_m_%d_n_%d' %
                    (device_name, batch_size, m, n)))

          self.run_op_benchmark(
              sess,
              control_flow_ops.group(x2),
              min_iters=10,
              store_memory_usage=False,
              name=('tridiagonal_matmul_%s_batch_size_%d_m_%d_n_%d' %
                    (device_name, batch_size, m, n)))


if __name__ == '__main__':
  test.main()
