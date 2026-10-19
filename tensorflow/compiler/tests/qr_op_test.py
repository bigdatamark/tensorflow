# [MODULE DOC]
# XLA compiler test for the QR decomposition op. Runs QR on random real and complex matrices of
# many shapes and batch layouts on the XLA device and checks that Q is unitary, that Q times R
# reconstructs the input, and that Q matches the numpy result up to sign/phase. No import-time
# side effects.
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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""

import itertools
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@test_util.run_all_without_tensor_float_32(
    "XLA QR op calls matmul. Also, matmul used for verification. Also with "
    'TensorFloat-32, mysterious "Unable to launch cuBLAS gemm" error '
    "occasionally occurs")
# TODO(b/165435566): Fix "Unable to launch cuBLAS gemm" error
class QrOpTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized XLA test case for the QR op. Class attributes SIZES (matrix dimensions to try),
  # DTYPES (single-precision real and complex) and PARAMS (cross product of row sizes, column
  # sizes and dtypes) drive the parameterized test. The decorator above the class disables
  # TensorFloat-32 because the op and the verification both use matrix multiplication (and a
  # sporadic GPU gemm launch failure was seen).

  def AdjustedNorm(self, x):
    # [FUNCTION DOC]
    # HELPER: error metric for the QR checks.
    # Takes an array of matrices (last two axes are matrix dims) and returns the matrix norm of
    # each, divided by the larger matrix dimension times the machine epsilon of the dtype.
    """Computes the norm of matrices in 'x', adjusted for dimension and type."""
    norm = np.linalg.norm(x, axis=(-2, -1))
    return norm / (max(x.shape[-2:]) * np.finfo(x.dtype).eps)

  def CompareOrthogonal(self, x, y, rank):
    # [FUNCTION DOC]
    # HELPER: compares two Q factors. Only the first rank columns are compared, since the rest
    # are an arbitrary basis. Each matrix's columns are rescaled by the sign (or complex phase)
    # estimated from the elementwise ratio of y to x, and the function asserts the adjusted norm
    # of the difference is below 30. NOTE: modifies x in place (phase scaling is applied to the
    # sliced view of the first argument). Returns nothing.
    # We only compare the first 'rank' orthogonal vectors since the
    # remainder form an arbitrary orthonormal basis for the
    # (row- or column-) null space, whose exact value depends on
    # implementation details. Notice that since we check that the
    # matrices of singular vectors are unitary elsewhere, we do
    # implicitly test that the trailing vectors of x and y span the
    # same space.
    x = x[..., 0:rank]
    y = y[..., 0:rank]
    # Q is only unique up to sign (complex phase factor for complex matrices),
    # so we normalize the sign first.
    sum_of_ratios = np.sum(np.divide(y, x), -2, keepdims=True)
    phases = np.divide(sum_of_ratios, np.abs(sum_of_ratios))
    x *= phases
    self.assertTrue(np.all(self.AdjustedNorm(x - y) < 30.0))

  def CheckApproximation(self, a, q, r):
    # [FUNCTION DOC]
    # HELPER: asserts the product of q and r reproduces the original matrices a, i.e. the
    # adjusted norm of the difference is below 11 for every matrix in the batch. Returns nothing.
    # Tests that a ~= q*r.
    precision = self.AdjustedNorm(a - np.matmul(q, r))
    self.assertTrue(np.all(precision < 11.0))

  def CheckUnitary(self, x):
    # [FUNCTION DOC]
    # HELPER: asserts that the conjugate transpose of x times x is close to the identity, using
    # a tolerance of 100 times the machine epsilon of the dtype. The identity is built by keeping
    # only the diagonal of an all-ones matrix. Returns nothing.
    # Tests that x[...,:,:]^H * x[...,:,:] is close to the identity.
    xx = math_ops.matmul(x, x, adjoint_a=True)
    identity = array_ops.matrix_band_part(array_ops.ones_like(xx), 0, 0)
    tol = 100 * np.finfo(x.dtype).eps
    self.assertAllClose(xx, identity, atol=tol)

  def _random_matrix(self, dtype, shape):
    # [FUNCTION DOC]
    # HELPER: builds a deterministic random test matrix. Seeds the global numpy random generator
    # with 1 (side effect), draws uniform values in [-1, 1) of the given shape cast to dtype,
    # and for complex dtypes adds an independent random imaginary part. Returns the array.
    np.random.seed(1)

    def rng():
      # [FUNCTION DOC]
      # HELPER: nested generator of one uniform random array in [-1, 1) with the enclosing shape
      # and dtype. Returns the array.
      return np.random.uniform(
          low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype)

    x_np = rng()
    if np.issubdtype(dtype, np.complexfloating):
      x_np += rng() * dtype(1j)
    return x_np

  def _test(self, x_np, full_matrices, full_rank=True):
    # [FUNCTION DOC]
    # HELPER: end-to-end QR check. Runs the QR op on the XLA device for the input array x_np
    # (optionally batched) with the full_matrices flag (complete versus reduced factors).
    # Computes the reference Q per batch element with numpy in the matching mode. If full_rank
    # (default true) is set, compares Q to the reference up to sign/phase on min(rows, cols)
    # columns; always checks Q times R approximates the input and that Q is unitary.
    # Returns nothing.
    dtype = x_np.dtype
    shape = x_np.shape
    with self.session() as sess:
      x_tf = array_ops.placeholder(dtype)
      with self.device_scope():
        q_tf, r_tf = linalg_ops.qr(x_tf, full_matrices=full_matrices)
      q_tf_val, r_tf_val = sess.run([q_tf, r_tf], feed_dict={x_tf: x_np})

      q_dims = q_tf_val.shape
      np_q = np.ndarray(q_dims, dtype)
      np_q_reshape = np.reshape(np_q, (-1, q_dims[-2], q_dims[-1]))
      new_first_dim = np_q_reshape.shape[0]

      x_reshape = np.reshape(x_np, (-1, x_np.shape[-2], x_np.shape[-1]))
      for i in range(new_first_dim):
        if full_matrices:
          np_q_reshape[i, :, :], _ = np.linalg.qr(
              x_reshape[i, :, :], mode="complete")
        else:
          np_q_reshape[i, :, :], _ = np.linalg.qr(
              x_reshape[i, :, :], mode="reduced")
      np_q = np.reshape(np_q_reshape, q_dims)
      if full_rank:
        # Q is unique up to sign/phase if the matrix is full-rank.
        self.CompareOrthogonal(np_q, q_tf_val, min(shape[-2:]))
      self.CheckApproximation(x_np, q_tf_val, r_tf_val)
      self.CheckUnitary(q_tf_val)

  SIZES = [1, 2, 5, 10, 32, 100, 300, 603]
  DTYPES = [np.float32, np.complex64]
  PARAMS = itertools.product(SIZES, SIZES, DTYPES)

  @parameterized.parameters(*PARAMS)
  def testQR(self, rows, cols, dtype):
    # [FUNCTION DOC]
    # Parameterized over every combination of row count, column count and dtype. For both
    # complete and reduced factorization, runs the QR check on an unbatched matrix, on a batch
    # of 3, and (only when both dimensions are under 10) on a 3x2 batch.
    for full_matrices in [True, False]:
      # Only tests the (3, 2) case for small numbers of rows/columns.
      for batch_dims in [(), (3,)] + [(3, 2)] * (max(rows, cols) < 10):
        x_np = self._random_matrix(dtype, batch_dims + (rows, cols))
        self._test(x_np, full_matrices)

  def testLarge2000x2000(self):
    # [FUNCTION DOC]
    # QR check with complete factors on a single 2000x2000 single-precision real matrix.
    x_np = self._random_matrix(np.float32, (2000, 2000))
    self._test(x_np, full_matrices=True)

  @unittest.skip("Test times out on CI")
  def testLarge17500x128(self):
    # [FUNCTION DOC]
    # QR check with complete factors on a tall 17500x128 real matrix. Skipped (decorator)
    # because it times out on continuous integration.
    x_np = self._random_matrix(np.float32, (17500, 128))
    self._test(x_np, full_matrices=True)

  @parameterized.parameters((23, 25), (513, 23))
  def testZeroColumn(self, rows, cols):
    # [FUNCTION DOC]
    # Parameterized over shapes (23x25) and (513x23). Makes a random complex matrix, zeroes
    # column 7 and runs the QR check with complete factors (full-rank comparison still applied).
    x_np = self._random_matrix(np.complex64, (rows, cols))
    x_np[:, 7] = 0.
    self._test(x_np, full_matrices=True)

  @parameterized.parameters((4, 4), (514, 20))
  def testRepeatedColumn(self, rows, cols):
    # [FUNCTION DOC]
    # Parameterized over shapes (4x4) and (514x20). Makes a random complex matrix with column 1
    # copied from column 2 (rank deficient) and runs the QR check with complete factors, skipping
    # the comparison against numpy's Q since Q is not unique.
    x_np = self._random_matrix(np.complex64, (rows, cols))
    x_np[:, 1] = x_np[:, 2]
    self._test(x_np, full_matrices=True, full_rank=False)


if __name__ == "__main__":
  test.main()
