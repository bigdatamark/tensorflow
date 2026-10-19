# [MODULE DOC]
# Tests and a benchmark for the LU decomposition op (linalg_ops.lu). The test class verifies that
# the permutation applied to the product of the unit-lower and upper factors reproduces the input,
# for real and complex dtypes, 32- and 64-bit permutation index types, batches, large and empty
# inputs, pivoting behavior, singular-input errors and repeated execution with identical inputs.
# The benchmark times the op on CPU (and GPU when available) over a range of shapes.
# No import-time side effects other than imports.
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
"""Tests for tensorflow.ops.tf.Lu."""

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
class LuOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the LU decomposition op. Provides verification helpers that rebuild the factors
  # from the packed result and check them, plus tests for basic, pivoting, singular, batched,
  # large, empty and concurrent cases.

  @property
  def float_types(self):
    # [FUNCTION DOC]
    # Property returning the set of element types under test: 64- and 32-bit floats and 64- and
    # 128-bit complex numbers. No side effects.
    return set((np.float64, np.float32, np.complex64, np.complex128))

  def _verifyLuBase(self, x, lower, upper, perm, verification,
                    output_idx_type):
    # [FUNCTION DOC]
    # HELPER: shared checks for the LU tests. Evaluates the lower factor, upper factor, permutation
    # and reconstructed matrix, then asserts: the reconstruction is close to the original input x;
    # factor shapes match x; the permutation shape equals the input shape without its last
    # dimension; factor dtypes equal the input dtype; permutation dtype equals the requested index
    # type; and each permutation vector is a valid permutation of the range of its length.
    # Inputs: x (original array), lower, upper, perm, verification (tensors), output_idx_type
    # (expected index dtype). Returns nothing.
    lower_np, upper_np, perm_np, verification_np = self.evaluate(
        [lower, upper, perm, verification])

    self.assertAllClose(x, verification_np)
    self.assertShapeEqual(x, lower)
    self.assertShapeEqual(x, upper)

    self.assertAllEqual(x.shape[:-1], perm.shape.as_list())

    # Check dtypes are as expected.
    self.assertEqual(x.dtype, lower_np.dtype)
    self.assertEqual(x.dtype, upper_np.dtype)
    self.assertEqual(output_idx_type.as_numpy_dtype, perm_np.dtype)

    # Check that the permutation is valid.
    if perm_np.shape[-1] > 0:
      perm_reshaped = np.reshape(perm_np, (-1, perm_np.shape[-1]))
      for perm_vector in perm_reshaped:
        self.assertAllClose(np.arange(len(perm_vector)), np.sort(perm_vector))

  def _verifyLu(self, x, output_idx_type=dtypes.int64):
    # [FUNCTION DOC]
    # HELPER: runs the LU op on x and verifies that row-permuted lower times upper reproduces x.
    # Inputs: x (array of matrices, possibly batched, possibly non-square or empty) and
    # output_idx_type (permutation index dtype, default 64-bit integer). Builds the unit lower
    # factor (padding or trimming it for non-square input and setting a unit diagonal) and the
    # upper factor from the packed result, multiplies them without TF32, undoes the permutation
    # per batch by inverting it and gathering rows, then delegates to the base verification.
    # Returns nothing; assertion failures signal errors.
    # Verify that Px = LU.
    lu, perm = linalg_ops.lu(x, output_idx_type=output_idx_type)

    # Prepare the lower factor of shape num_rows x num_rows
    lu_shape = np.array(lu.shape.as_list())
    batch_shape = lu_shape[:-2]
    num_rows = lu_shape[-2]
    num_cols = lu_shape[-1]

    lower = array_ops.matrix_band_part(lu, -1, 0)

    if num_rows > num_cols:
      eye = linalg_ops.eye(
          num_rows, batch_shape=batch_shape, dtype=lower.dtype)
      lower = array_ops.concat([lower, eye[..., num_cols:]], axis=-1)
    elif num_rows < num_cols:
      lower = lower[..., :num_rows]

    # Fill the diagonal with ones.
    ones_diag = array_ops.ones(
        np.append(batch_shape, num_rows), dtype=lower.dtype)
    lower = array_ops.matrix_set_diag(lower, ones_diag)

    # Prepare the upper factor.
    upper = array_ops.matrix_band_part(lu, 0, -1)

    verification = test_util.matmul_without_tf32(lower, upper)

    # Permute the rows of product of the Cholesky factors.
    if num_rows > 0:
      # Reshape the product of the triangular factors and permutation indices
      # to a single batch dimension. This makes it easy to apply
      # invert_permutation and gather_nd ops.
      perm_reshaped = array_ops.reshape(perm, [-1, num_rows])
      verification_reshaped = array_ops.reshape(verification,
                                                [-1, num_rows, num_cols])
      # Invert the permutation in each batch.
      inv_perm_reshaped = map_fn.map_fn(array_ops.invert_permutation,
                                        perm_reshaped)
      batch_size = perm_reshaped.shape.as_list()[0]
      # Prepare the batch indices with the same shape as the permutation.
      # The corresponding batch index is paired with each of the `num_rows`
      # permutation indices.
      batch_indices = math_ops.cast(
          array_ops.broadcast_to(
              math_ops.range(batch_size)[:, None], perm_reshaped.shape),
          dtype=output_idx_type)
      if inv_perm_reshaped.shape == [0]:
        inv_perm_reshaped = array_ops.zeros_like(batch_indices)
      permuted_verification_reshaped = array_ops.gather_nd(
          verification_reshaped,
          array_ops_stack.stack([batch_indices, inv_perm_reshaped], axis=-1))

      # Reshape the verification matrix back to the original shape.
      verification = array_ops.reshape(permuted_verification_reshaped,
                                       lu_shape)

    self._verifyLuBase(x, lower, upper, perm, verification,
                       output_idx_type)

  def testBasic(self):
    # [FUNCTION DOC]
    # Test: a fixed 3x3 matrix is factored and verified for float32/float64 and for complex64/
    # complex128 (built as a Hermitian-style complex variant of the same data), each with 32- and
    # 64-bit permutation index types, using sub-tests.
    data = np.array([[4., -1., 2.], [-1., 6., 0], [10., 0., 5.]])

    for dtype in (np.float32, np.float64):
      for output_idx_type in (dtypes.int32, dtypes.int64):
        with self.subTest(dtype=dtype, output_idx_type=output_idx_type):
          self._verifyLu(data.astype(dtype), output_idx_type=output_idx_type)

    for dtype in (np.complex64, np.complex128):
      for output_idx_type in (dtypes.int32, dtypes.int64):
        with self.subTest(dtype=dtype, output_idx_type=output_idx_type):
          complex_data = np.tril(1j * data, -1).astype(dtype)
          complex_data += np.triu(-1j * data, 1).astype(dtype)
          complex_data += data
          self._verifyLu(complex_data, output_idx_type=output_idx_type)

  def testPivoting(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the complex-dtype loop the permutation check calls the LU op on the real
    # matrix (data) rather than the complex matrix (complex_data), so it does not test complex
    # pivoting; the dtype loop variable is not used for that call.
    # Test: a matrix with a tiny first pivot forces partial pivoting. Verifies the factorization
    # for float32 and float64 (and complex variants) and asserts that the returned permutation
    # is not the identity.
    # This matrix triggers partial pivoting because the first diagonal entry
    # is small.
    data = np.array([[1e-9, 1., 0.], [1., 0., 0], [0., 1., 5]])
    self._verifyLu(data.astype(np.float32))

    for dtype in (np.float32, np.float64):
      with self.subTest(dtype=dtype):
        self._verifyLu(data.astype(dtype))
        _, p = linalg_ops.lu(data)
        p_val = self.evaluate([p])
        # Make sure p_val is not the identity permutation.
        self.assertNotAllClose(np.arange(3), p_val)

    for dtype in (np.complex64, np.complex128):
      with self.subTest(dtype=dtype):
        complex_data = np.tril(1j * data, -1).astype(dtype)
        complex_data += np.triu(-1j * data, 1).astype(dtype)
        complex_data += data
        self._verifyLu(complex_data)
        _, p = linalg_ops.lu(data)
        p_val = self.evaluate([p])
        # Make sure p_val is not the identity permutation.
        self.assertNotAllClose(np.arange(3), p_val)

  def testInvalidMatrix(self):
    # [FUNCTION DOC]
    # Test: for every supported dtype, factoring a singular 3x3 matrix, and a batch containing a
    # singular matrix, raises an invalid-argument error.
    # LU factorization gives an error when the input is singular.
    # Note: A singular matrix may return without error but it won't be a valid
    # factorization.
    for dtype in self.float_types:
      with self.subTest(dtype=dtype):
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(
              linalg_ops.lu(
                  np.array([[1., 2., 3.], [2., 4., 6.], [2., 3., 4.]],
                           dtype=dtype)))
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(
              linalg_ops.lu(
                  np.array([[[1., 2., 3.], [2., 4., 6.], [1., 2., 3.]],
                            [[1., 2., 3.], [3., 4., 5.], [5., 6., 7.]]],
                           dtype=dtype)))

  def testBatch(self):
    # [FUNCTION DOC]
    # Test: verifies the factorization for small batches of 2x2 and 3x3 matrices and for 200
    # random 5x5 real and complex matrices (random generator seeded with fixed values, which
    # changes global numpy random state).
    simple_array = np.array([[[1., -1.], [2., 5.]]])  # shape (1, 2, 2)
    self._verifyLu(simple_array)
    self._verifyLu(np.vstack((simple_array, simple_array)))
    odd_sized_array = np.array([[[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]])
    self._verifyLu(np.vstack((odd_sized_array, odd_sized_array)))

    batch_size = 200

    # Generate random matrices.
    np.random.seed(42)
    matrices = np.random.rand(batch_size, 5, 5)
    self._verifyLu(matrices)

    # Generate random complex valued matrices.
    np.random.seed(52)
    matrices = np.random.rand(batch_size, 5,
                              5) + 1j * np.random.rand(batch_size, 5, 5)
    self._verifyLu(matrices)

  def testLargeMatrix(self):
    # [FUNCTION DOC]
    # Test: verifies the factorization for random 500x500 real and complex matrices (fixed
    # seeds set on the global numpy random state).
    # Generate random matrices.
    n = 500
    np.random.seed(64)
    data = np.random.rand(n, n)
    self._verifyLu(data)

    # Generate random complex valued matrices.
    np.random.seed(129)
    data = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    self._verifyLu(data)

  @test_util.disable_xla("b/206106619")
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testEmpty(self):
    # [FUNCTION DOC]
    # Test: factorization of empty inputs (zero-size batch and zero-by-zero matrices) works and
    # verifies. Disabled under XLA; runs in graph and eager modes, allowing GPU.
    self._verifyLu(np.empty([0, 2, 2]))
    self._verifyLu(np.empty([2, 0, 0]))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testConcurrentExecutesWithoutError(self):
    # [FUNCTION DOC]
    # Test: two identical random 5x5 matrices (same stateless seed) are factored in the same
    # evaluation; asserts the inputs are equal and that both LU results and permutations are
    # equal, i.e. concurrent execution gives deterministic, error-free results.
    matrix_shape = [5, 5]
    seed = [42, 24]
    matrix1 = stateless_random_ops.stateless_random_normal(
        shape=matrix_shape, seed=seed)
    matrix2 = stateless_random_ops.stateless_random_normal(
        shape=matrix_shape, seed=seed)
    self.assertAllEqual(matrix1, matrix2)
    lu1, p1 = linalg_ops.lu(matrix1)
    lu2, p2 = linalg_ops.lu(matrix2)
    lu1_val, p1_val, lu2_val, p2_val = self.evaluate([lu1, p1, lu2, p2])
    self.assertAllEqual(lu1_val, lu2_val)
    self.assertAllEqual(p1_val, p2_val)


class LuBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark for the LU op. Class attribute shapes lists the matrix shapes (square, with optional
  # leading batch dimensions) to time.
  shapes = [
      (4, 4),
      (10, 10),
      (16, 16),
      (101, 101),
      (256, 256),
      (1000, 1000),
      (1024, 1024),
      (2048, 2048),
      (4096, 4096),
      (513, 2, 2),
      (513, 8, 8),
      (513, 256, 256),
      (4, 513, 2, 2),
  ]

  def _GenerateMatrix(self, shape):
    # [FUNCTION DOC]
    # HELPER: builds benchmark input. Input: shape tuple whose last two entries must be equal
    # (square matrices) with optional leading batch entries. Returns a float32 array: each matrix
    # is a constant small off-diagonal value (one over twice the size) plus an identity matrix
    # (well conditioned), tiled across the batch. Raises an assertion error for non-square shapes.
    batch_shape = shape[:-2]
    shape = shape[-2:]
    assert shape[0] == shape[1]
    n = shape[0]
    matrix = np.ones(shape).astype(np.float32) / (2.0 * n) + np.diag(
        np.ones(n).astype(np.float32))
    return np.tile(matrix, batch_shape + (1, 1))

  def benchmarkLuOp(self):
    # [FUNCTION DOC]
    # Benchmark: for each shape, in a fresh graph and session pinned to CPU, creates a variable
    # holding a generated matrix, runs the LU op and reports timing of the grouped outputs
    # (at least 25 iterations, named by device and shape). If a GPU is available, repeats on the
    # first GPU. Side effects: creates graphs and sessions and emits benchmark reports.
    for shape in self.shapes:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix = variables.Variable(self._GenerateMatrix(shape))
        lu, p = linalg_ops.lu(matrix)
        self.evaluate(variables.global_variables_initializer())
        self.run_op_benchmark(
            sess,
            control_flow_ops.group(lu, p),
            min_iters=25,
            name="lu_cpu_{shape}".format(shape=shape))

      if test.is_gpu_available(True):
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/device:GPU:0"):
          matrix = variables.Variable(self._GenerateMatrix(shape))
          lu, p = linalg_ops.lu(matrix)
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(lu, p),
              min_iters=25,
              name="lu_gpu_{shape}".format(shape=shape))


if __name__ == "__main__":
  test.main()
