# [MODULE DOC]
# Tests for local response normalization (LRN) and its gradient on the XLA test device.
# The forward tests were copied from the core kernel tests; the gradient test compares the XLA
# result with the CPU kernel. Defines the constant naming the CPU device used for reference.
# Running the file directly starts the test runner.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Local Response Normalization ops."""

import copy

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import googletest

CPU_DEVICE = "/job:localhost/replica:0/task:0/cpu:0"


# Local response normalization tests. The forward tests are copied from
# tensorflow/python/kernel_tests/lrn_op_test.py
class LRNTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests LRN forward computation against a NumPy reference and the LRN gradient against CPU.

  def _LRN(self, input_image, lrn_depth_radius=5, bias=1.0, alpha=1.0,
           beta=0.5):
    # [FUNCTION DOC]
    # HELPER: NumPy reference implementation of local response normalization.
    # Inputs: input_image, a 4-D array (batch, rows, columns, depth); lrn_depth_radius
    # (default 5), bias (1.0), alpha (1.0), beta (0.5). For every position divides the value by
    # bias plus alpha times the sum of squares over the window of depth neighbors, raised to
    # beta. Returns a new array; the input is not modified.
    """Compute expected result."""
    output = copy.deepcopy(input_image)
    batch_size = input_image.shape[0]
    rows = input_image.shape[1]
    cols = input_image.shape[2]
    depth = input_image.shape[3]
    for b in range(batch_size):
      for r in range(rows):
        for c in range(cols):
          for d in range(depth):
            begin = max(0, d - lrn_depth_radius)
            end = min(depth, d + lrn_depth_radius + 1)
            patch = input_image[b, r, c, begin:end]
            output[b, r, c, d] /= (
                np.power(bias + alpha * np.sum(patch * patch), beta))
    return output

  def _RunAndVerify(self, dtype):
    # [FUNCTION DOC]
    # HELPER: one randomized forward check for the given dtype.
    # Picks a random 4-D shape (depth at least 2), random radius, bias, alpha and beta, runs the
    # op on the test device on random float32 data and compares with the reference: max error
    # below 1e-4 for float32, otherwise below 1e-2. Prints the error and checks shape equality.
    with self.session():
      # random shape
      shape = np.random.randint(1, 16, size=4)
      # Make depth at least 2 to make it meaningful
      shape[3] += 1
      p = array_ops.placeholder(dtype, shape=shape)
      # random depth_radius, bias, alpha, beta
      lrn_depth_radius = np.random.randint(1, shape[3])
      bias = 1.0 + np.random.rand()
      alpha = 2.0 * np.random.rand()
      beta = 2.0 * np.random.rand()
      with self.test_scope():
        lrn_t = nn.local_response_normalization(
            p,
            name="lrn",
            depth_radius=lrn_depth_radius,
            bias=bias,
            alpha=alpha,
            beta=beta)
      params = {p: np.random.rand(*shape).astype("f")}
      result = lrn_t.eval(feed_dict=params)
    expected = self._LRN(
        params[p],
        lrn_depth_radius=lrn_depth_radius,
        bias=bias,
        alpha=alpha,
        beta=beta)
    err = np.amax(np.abs(result - expected))
    print("LRN error for bias ", bias, "alpha ", alpha, " beta ", beta, " is ",
          err)
    if dtype == dtypes.float32:
      self.assertTrue(err < 1e-4)
    else:
      self.assertTrue(err < 1e-2)
    self.assertShapeEqual(expected, lrn_t)

  def testCompute(self):
    # [FUNCTION DOC]
    # Runs the randomized forward check twice with float32.
    for _ in range(2):
      self._RunAndVerify(dtypes.float32)

  def testLrnGrad(self):
    # [FUNCTION DOC]
    # Runs the LRN gradient op on the CPU and on the test device with identical ramp-valued
    # inputs and random parameters; asserts the results match with relative tolerance 1e-3.
    # Test for LRNGrad that compares against the CPU implementation.
    shape = [1, 2, 3, 4]
    total_size = np.prod(shape)
    in_image_vals = np.arange(1, total_size + 1, dtype=np.float32)
    out_image_vals = np.arange(1, total_size + 1, dtype=np.float32)
    out_grads_vals = np.arange(1, total_size + 1, dtype=np.float32)
    depth_radius = np.random.randint(1, shape[3])
    bias = 1.0 + np.random.rand()
    alpha = 1.0 * np.random.rand()
    beta = 1.0 * np.random.rand()

    with self.session():
      in_image = constant_op.constant(in_image_vals, shape=shape)
      out_image = constant_op.constant(out_image_vals, shape=shape)
      out_grads = constant_op.constant(out_grads_vals, shape=shape)
      with ops.device(CPU_DEVICE):
        expected = gen_nn_ops.lrn_grad(out_grads, in_image, out_image,
                                       depth_radius, bias, alpha, beta)
      with self.test_scope():
        actual = gen_nn_ops.lrn_grad(out_grads, in_image, out_image,
                                     depth_radius, bias, alpha, beta)
      expected_val = self.evaluate(expected)
      actual_val = self.evaluate(actual)
    self.assertAllClose(actual_val, expected_val, rtol=1e-3)


if __name__ == "__main__":
  googletest.main()
