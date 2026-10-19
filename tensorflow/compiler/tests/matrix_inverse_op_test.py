# [MODULE DOC]
# XLA compiler test module for the matrix inverse op: verifies inverse times the matrix is the
# identity on the XLA test device for plain, adjoint, batched and empty inputs. Runs the test
# runner when executed as a script. No import-time side effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class InverseOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the matrix inverse op on the XLA test device: checks that the product of
  # the inverse and the original is the identity, for single matrices and batches, with and
  # without the adjoint option, and for empty inputs.

  def _verifyInverse(self, x, np_type):
    # [FUNCTION DOC]
    # HELPER: shared check for the inverse tests. Inputs: x (numpy array of one or more square
    # matrices) and np_type (numpy element type). For both adjoint settings, casts x, feeds it
    # through a placeholder, computes the inverse in the test scope, multiplies it with the
    # input (adjoint applied to the second operand when requested) and asserts the result is a
    # tiled identity within tolerance 0.001, and that the shape equals the input shape.
    for adjoint in False, True:
      y = x.astype(np_type)
      with self.session() as sess:
        # Verify that x^{-1} * x == Identity matrix.
        p = array_ops.placeholder(dtypes.as_dtype(y.dtype), y.shape, name="x")
        with self.test_scope():
          inv = linalg_ops.matrix_inverse(p, adjoint=adjoint)
          tf_ans = math_ops.matmul(inv, p, adjoint_b=adjoint)
          np_ans = np.identity(y.shape[-1])
          if x.ndim > 2:
            tiling = list(y.shape)
            tiling[-2:] = [1, 1]
            np_ans = np.tile(np_ans, tiling)
        out = sess.run(tf_ans, feed_dict={p: y})
        self.assertAllClose(np_ans, out, rtol=1e-3, atol=1e-3)
        self.assertShapeEqual(y, tf_ans)

  def _verifyInverseReal(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the inverse check for x with each supported floating type that is both
    # float32 or float64 and available on the test device.
    for np_type in self.float_types & {np.float64, np.float32}:
      self._verifyInverse(x, np_type)

  def _makeBatch(self, matrix1, matrix2):
    # [FUNCTION DOC]
    # HELPER: stacks the two given matrices into a pair and tiles it to a four-dimensional
    # batch (two by three copies of the pair). Returns the numpy batch array.
    matrix_batch = np.concatenate(
        [np.expand_dims(matrix1, 0),
         np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    return matrix_batch

  def testNonsymmetric(self):
    # [FUNCTION DOC]
    # Scenario: inverse of two nonsymmetric two-by-two matrices, singly and as a
    # multi-dimensional batch. Expects inverse times original to be the identity.
    # 2x2 matrices
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    self._verifyInverseReal(matrix1)
    self._verifyInverseReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyInverseReal(self._makeBatch(matrix1, matrix2))

  def testSymmetricPositiveDefinite(self):
    # [FUNCTION DOC]
    # Scenario: inverse of two symmetric positive definite two-by-two matrices, singly and as
    # a multi-dimensional batch. Expects inverse times original to be the identity.
    # 2x2 matrices
    matrix1 = np.array([[2., 1.], [1., 2.]])
    matrix2 = np.array([[3., -1.], [-1., 3.]])
    self._verifyInverseReal(matrix1)
    self._verifyInverseReal(matrix2)
    # A multidimensional batch of 2x2 matrices
    self._verifyInverseReal(self._makeBatch(matrix1, matrix2))

  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: empty inputs, a batch of zero matrices and a batch of two zero-sized matrices.
    # Expects the inverse check to pass (shapes preserved, nothing to compare).
    self._verifyInverseReal(np.empty([0, 2, 2]))
    self._verifyInverseReal(np.empty([2, 0, 0]))


if __name__ == "__main__":
  googletest.main()
