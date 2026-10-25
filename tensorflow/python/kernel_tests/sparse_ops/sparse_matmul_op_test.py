# [MODULE DOC]
# Tests for matrix multiplication with the sparse-hint flags (a_is_sparse / b_is_sparse) of the
# matmul op, which dispatches to the sparse matmul kernel for float32 and bfloat16 operands.
# Forward results are compared against numpy products for basic, zero-sized, empty, very tall or
# wide and random shapes with all transpose/sparsity/dtype combinations; gradients are checked
# numerically. Defines a random-matrix helper and two test classes; no import-time side effects.
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
"""Tests for tensorflow.ops.tf.matmul."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def RandMatrix(rows, cols, tr, round_bfloat=False):
  # [FUNCTION DOC]
  # HELPER: random test-data generator for the sparse matmul tests.
  # Builds a float32 matrix of random values in the range minus half to plus half, clipped to
  # plus or minus 64/128 and scaled by 1/128. Inputs: rows and cols, the logical dimensions;
  # tr, whether the matrix will be used transposed (if true the two dimensions are swapped so the
  # stored shape is cols by rows); round_bfloat (default false), when true draws integers instead
  # of uniform reals so values are exactly representable in bfloat16. Returns the numpy array.
  # Uses the global numpy random state (not seeded here).
  if tr:
    rows, cols = cols, rows
  rand_func = np.random.randint if round_bfloat else np.random.uniform
  return (np.clip(
      rand_func(
          low=-256.0, high=256.0, size=rows * cols), -64,
      64) / 128.0).reshape([rows, cols]).astype(np.float32)


class SparseMatMulTest(test.TestCase):
  # [CLASS DOC]
  # Forward-pass tests of sparse-flagged matmul on CPU. All tests call one shared helper with
  # different shapes and flags and compare against numpy. No shared state.

  def _testCpuMatmul(self,
                     x,
                     y,
                     tr_a=False,
                     tr_b=False,
                     sp_a=True,
                     sp_b=False,
                     x_dtype=dtypes.float32,
                     y_dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: shared check used by the forward matmul tests.
    # Casts numpy arrays x and y to x_dtype / y_dtype (default float32; may be bfloat16),
    # multiplies them with the matmul op on CPU using the transpose flags tr_a / tr_b (default
    # false) and sparse hints sp_a (default true) / sp_b (default false). Then computes the
    # reference product in numpy from the (possibly bfloat16-rounded) inputs, applying the same
    # transposes. Asserts the result shape matches and values are close within a 1e-4 tolerance.
    # Returns nothing; fails the test on mismatch.
    with self.cached_session(use_gpu=False):
      tf_x = math_ops.cast(x, x_dtype)
      tf_y = math_ops.cast(y, y_dtype)
      tf_ans = math_ops.matmul(
          tf_x,
          tf_y,
          transpose_a=tr_a,
          transpose_b=tr_b,
          a_is_sparse=sp_a,
          b_is_sparse=sp_b)
      out = self.evaluate(tf_ans)
      np_x = math_ops.cast(tf_x, dtypes.float32).eval()
      np_y = math_ops.cast(tf_y, dtypes.float32).eval()

    if tr_a:
      np_x = np.transpose(np_x)
    if tr_b:
      np_y = np.transpose(np_y)

    np_ans = np.matrix(np_x) * np.matrix(np_y)
    self.assertShapeEqual(np_ans, tf_ans)
    self.assertAllCloseAccordingToType(np_ans, out, rtol=1e-4, atol=1e-4)

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Multiplies a 4 by 1 matrix of increasing values with a 1 by 2 matrix, for every float32 /
    # bfloat16 combination of operand dtypes, and expects agreement with numpy.
    x = np.arange(0., 4.).reshape([4, 1]).astype(np.float32)
    y = np.arange(-1., 1.).reshape([1, 2]).astype(np.float32)
    for x_dtype in (dtypes.float32, dtypes.bfloat16):
      for y_dtype in (dtypes.float32, dtypes.bfloat16):
        self._testCpuMatmul(x, y, x_dtype=x_dtype, y_dtype=y_dtype)

  @test_util.run_deprecated_v1
  def testZeroDim(self):
    # [FUNCTION DOC]
    # Multiplies a 4 by 0 matrix with a 0 by 3 matrix (zero inner dimension) for all dtype
    # combinations. Expects a 4 by 3 result matching numpy.
    x = np.ones((4, 0)).astype(np.float32)
    y = np.ones((0, 3)).astype(np.float32)
    for x_dtype in (dtypes.float32, dtypes.bfloat16):
      for y_dtype in (dtypes.float32, dtypes.bfloat16):
        self._testCpuMatmul(x, y, x_dtype=x_dtype, y_dtype=y_dtype)

  @test_util.run_deprecated_v1
  def testEmpty(self):
    # [FUNCTION DOC]
    # Multiplies two 0 by 0 matrices for all dtype combinations. Expects an empty result
    # matching numpy.
    x = np.ones((0, 0)).astype(np.float32)
    y = np.ones((0, 0)).astype(np.float32)
    for x_dtype in (dtypes.float32, dtypes.bfloat16):
      for y_dtype in (dtypes.float32, dtypes.bfloat16):
        self._testCpuMatmul(x, y, x_dtype=x_dtype, y_dtype=y_dtype)

  # Tests setting one dimension to be a high value.
  @test_util.run_deprecated_v1
  def testLarge(self):
    # [FUNCTION DOC]
    # Picks one random dimension between 6000 and 20000 and two small random dimensions, then
    # for each placement of the large dimension (rows, inner, columns) and all dtype
    # combinations multiplies random matrices and checks against numpy. Uses unseeded
    # randomness.
    r1 = np.random.randint(6000, 20000)
    r2 = np.random.randint(1, 10)
    r3 = np.random.randint(1, 10)
    for m, k, n in [(r1, r2, r3), (r2, r1, r3), (r2, r3, r1)]:
      for x_dtype in (dtypes.float32, dtypes.bfloat16):
        for y_dtype in (dtypes.float32, dtypes.bfloat16):
          x = RandMatrix(m, k, False)
          y = RandMatrix(k, n, False)
          self._testCpuMatmul(x, y, x_dtype=x_dtype, y_dtype=y_dtype)

  # Tests random sized matrices.
  @test_util.run_deprecated_v1
  def testRandom(self):
    # [FUNCTION DOC]
    # Exhaustively loops over both transpose flags, both sparse hints and all dtype combinations,
    # each time using random dimensions below 100 and random matrices, and checks the product
    # against numpy. Uses unseeded randomness.
    for tr_a in [True, False]:
      for tr_b in [True, False]:
        for sp_a in [True, False]:
          for sp_b in [True, False]:
            for x_dtype in (dtypes.float32, dtypes.bfloat16):
              for y_dtype in (dtypes.float32, dtypes.bfloat16):
                n, k, m = np.random.randint(1, 100, size=3)
                x = RandMatrix(n, k, tr_a)
                y = RandMatrix(k, m, tr_b)
                self._testCpuMatmul(
                    x,
                    y,
                    tr_a,
                    tr_b,
                    sp_a,
                    sp_b,
                    x_dtype=x_dtype,
                    y_dtype=y_dtype)


class MatMulGradientTest(test.TestCase):
  # [CLASS DOC]
  # Numerical gradient checks for sparse-flagged matmul over all transpose, sparsity and dtype
  # combinations. One helper performs a single check; one test sweeps the combinations.

  def _testGradients(self, tr_a, tr_b, sp_a, sp_b, a_dtype, b_dtype, delta,
                     name):
    # [FUNCTION DOC]
    # HELPER: shared gradient check for the gradient test.
    # Builds a random 3 by 2 operand and a 2 by 4 operand (bfloat16-exact values, transposed
    # storage per tr_a / tr_b), casts them to a_dtype / b_dtype when not float32, multiplies them
    # with sparse hints sp_a / sp_b under the given op name, and measures the numerical-versus-
    # analytic gradient error with respect to both operands using step size delta. Asserts the
    # summed error is at most half of delta. Returns nothing.
    with self.cached_session():
      a = constant_op.constant(
          RandMatrix(
              3, 2, tr_a, round_bfloat=True), dtype=dtypes.float32)
      b = constant_op.constant(
          RandMatrix(
              2, 4, tr_b, round_bfloat=True), dtype=dtypes.float32)
      tf_a = math_ops.cast(a, a_dtype) if a_dtype != dtypes.float32 else a
      tf_b = math_ops.cast(b, b_dtype) if b_dtype != dtypes.float32 else b

      m = math_ops.matmul(
          tf_a,
          tf_b,
          name=name,
          transpose_a=tr_a,
          transpose_b=tr_b,
          a_is_sparse=sp_a,
          b_is_sparse=sp_b)
      err = (
          gradient_checker.compute_gradient_error(
              a, [2, 3] if tr_a else [3, 2],
              m, [3, 4],
              x_init_value=self.evaluate(a),
              delta=delta) + gradient_checker.compute_gradient_error(
                  b, [4, 2] if tr_b else [2, 4],
                  m, [3, 4],
                  x_init_value=self.evaluate(b),
                  delta=delta))
    self.assertLessEqual(err, delta / 2.)

  @test_util.run_deprecated_v1
  def testGradientInput(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment says float32 has 10 mantissa bits; it really has 23 (10 is
    # the half-precision figure), though the tolerance logic is unaffected.
    # Sweeps both transpose flags, both sparse hints and all operand dtype combinations; uses a
    # looser step (one sixteenth) when both operands are bfloat16 and a tighter one (one
    # sixty-fourth) otherwise, builds a descriptive op name, and runs the gradient check helper.
    for tr_a in [True, False]:
      for tr_b in [True, False]:
        for sp_a in [True, False]:
          for sp_b in [True, False]:
            for a_dtype in (dtypes.float32, dtypes.bfloat16):
              for b_dtype in (dtypes.float32, dtypes.bfloat16):
                # Note: bfloat16 only has 7 mantissa bits, versus float32 with
                # 10. Hence, we shift by 2 bits to pass the test.
                if a_dtype == dtypes.bfloat16 and b_dtype == dtypes.bfloat16:
                  delta = 1 / 16.
                else:
                  delta = 1 / 64.
                name = "sparse_matmul_%s_%s_%s_%s" % (tr_a, tr_b, sp_a, sp_b)
                self._testGradients(tr_a, tr_b, sp_a, sp_b, a_dtype, b_dtype,
                                    delta, name)


if __name__ == "__main__":
  test.main()
