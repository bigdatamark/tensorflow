# [MODULE DOC]
# XLA compiler test for 3-D pooling (average and max) and their gradients. Compares results of
# the forward ops against hand-computed values and the gradient ops (including max-pool second
# order gradients) against the TensorFlow CPU kernels. Defines one module-level helper wrapper
# and one test class; no import-time side effects. Runs via the test main entry point.
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
"""Functional tests for 3d pooling operations."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


# Wrapper around AvgPoolGrad that ignores extra arguments needed by
# MaxPoolGrad.
def _AvgPoolGrad(inputs, outputs, output_gradients, ksize, strides, padding):
  # [FUNCTION DOC]
  # HELPER: adapts the average-pool 3-D gradient op to the signature used for max-pool gradients.
  # Takes forward inputs (only their static shape is used), forward outputs (ignored), the
  # gradient w.r.t. outputs, and kernel size, strides and padding. Returns the gradient tensor
  # with respect to the inputs.
  del outputs  # Unused by average-pooling gradients.
  return gen_nn_ops.avg_pool3d_grad(
      inputs.get_shape().as_list(),
      output_gradients,
      ksize=ksize,
      strides=strides,
      padding=padding)


class Pooling3DTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for 3-D average/max pooling and their gradients on the XLA test device. Provides
  # helpers that run a pooling op on the XLA device and compare to expected values, and helpers
  # that compare XLA gradient results with the CPU kernels. No shared fixtures.

  def _VerifyValues(self, pool_func, input_sizes, window, strides, padding,
                    expected):
    # [FUNCTION DOC]
    # HELPER: shared check for the forward pooling tests.
    # Builds an input tensor of the given shape filled with 1, 2, 3, ... (float32), runs the
    # pooling function on the XLA test device with unit batch/channel windows and strides, and
    # asserts the flattened result is close to expected. Parameters: pool_func (pooling op),
    # input_sizes (list of 5 dims), window and strides (3 values each: planes, rows, cols),
    # padding (SAME or VALID string), expected (flat list/array). Returns nothing.
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called: co.MaxPool, co.AvgPool.
      input_sizes: Input tensor dimensions.
      window: Tuple of kernel dims: planes, rows, cols.
      strides: Tuple of strides for dims: planes, rows, cols.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
    """
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = np.arange(1.0, total_size + 1, dtype=np.float32)
    x = x.reshape(input_sizes)
    with self.session() as sess, self.test_scope():
      inputs = array_ops.placeholder(dtypes.float32)
      t = pool_func(
          inputs,
          ksize=[1] + window + [1],
          strides=[1] + strides + [1],
          padding=padding)
      vals = sess.run(t, {inputs: x})
    # Verifies values.
    actual = vals.flatten()
    self.assertAllClose(expected, actual)

  def testAvgPool3dValidPadding(self):
    # [FUNCTION DOC]
    # Average pool 3-D, 2x2x2 window and stride, VALID padding on a 1x3x3x3x3 input; expects the
    # three averaged channel values 20.5, 21.5, 22.5.
    expected_output = [20.5, 21.5, 22.5]
    self._VerifyValues(
        nn_ops.avg_pool3d,
        input_sizes=[1, 3, 3, 3, 3],
        window=[2, 2, 2],
        strides=[2, 2, 2],
        padding="VALID",
        expected=expected_output)

  def testAvgPool3dSamePadding(self):
    # [FUNCTION DOC]
    # Average pool 3-D, 2x2x2 window and stride, SAME padding on a 1x2x2x4x3 input; expects six
    # averaged values (two output columns by three channels).
    expected_output = [20.5, 21.5, 22.5, 26.5, 27.5, 28.5]
    self._VerifyValues(
        nn_ops.avg_pool3d,
        input_sizes=[1, 2, 2, 4, 3],
        window=[2, 2, 2],
        strides=[2, 2, 2],
        padding="SAME",
        expected=expected_output)

  def testAvgPool3dSamePaddingDifferentStrides(self):
    # [FUNCTION DOC]
    # Average pool 3-D with window 1x2x3, strides 2x3x1, SAME padding on a 1x5x8x1x1 input; the
    # windows differ per dimension. Expects nine averaged values.
    expected_output = [1.5, 4.5, 7.5, 17.5, 20.5, 23.5, 33.5, 36.5, 39.5]
    self._VerifyValues(
        nn_ops.avg_pool3d,
        input_sizes=[1, 5, 8, 1, 1],
        window=[1, 2, 3],
        strides=[2, 3, 1],
        padding="SAME",
        expected=expected_output)

  def testMaxPool3dValidPadding(self):
    # [FUNCTION DOC]
    # Max pool 3-D, 2x2x2 window and stride, VALID padding on a 1x3x3x3x3 input; expects the
    # maxima 40, 41, 42.
    expected_output = [40.0, 41.0, 42.0]
    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 3, 3, 3],
        window=[2, 2, 2],
        strides=[2, 2, 2],
        padding="VALID",
        expected=expected_output)

  def testMaxPool3dSamePadding(self):
    # [FUNCTION DOC]
    # Max pool 3-D, 2x2x2 window and stride, SAME padding on a 1x2x2x3x3 input; expects the
    # maxima 31 through 36.
    expected_output = [31., 32., 33., 34., 35., 36.]
    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 2, 2, 3, 3],
        window=[2, 2, 2],
        strides=[2, 2, 2],
        padding="SAME",
        expected=expected_output)

  def testMaxPool3dSamePaddingDifferentStrides(self):
    # [FUNCTION DOC]
    # Max pool 3-D with window 1x2x3, strides 2x3x1, SAME padding on a 1x5x8x1x1 input, expecting
    # nine maxima. Then repeats on a larger 1x5x27x27x64 input with window and strides 1x2x2,
    # where the expected output is simulated with numpy by zero-padding one row/column and
    # taking every second element, patching the last row/column from the second-to-last one.
    expected_output = [2., 5., 8., 18., 21., 24., 34., 37., 40.]
    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 5, 8, 1, 1],
        window=[1, 2, 3],
        strides=[2, 3, 1],
        padding="SAME",
        expected=expected_output)

    # Test pooling on a larger input, with different stride and kernel
    # size for the 'z' dimension.

    # Simulate max pooling in numpy to get the expected output.
    input_data = np.arange(1, 5 * 27 * 27 * 64 + 1).reshape((5, 27, 27, 64))
    input_data = np.pad(input_data, [[0, 0], [0, 1], [0, 1], [0, 0]],
                        mode="constant")
    expected_output = input_data[:, 1::2, 1::2, :]
    expected_output[:, -1, :, :] = input_data[:, -2, 1::2, :]
    expected_output[:, :, -1, :] = input_data[:, 1::2, -2, :]
    expected_output[:, -1, -1, :] = input_data[:, -2, -2, :]

    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 5, 27, 27, 64],
        window=[1, 2, 2],
        strides=[1, 2, 2],
        padding="SAME",
        expected=expected_output.flatten())

  def testKernelSmallerThanStride(self):
    # [FUNCTION DOC]
    # Checks pooling where the stride exceeds the window, for both max and average pooling: 1x1x1
    # window with stride 2 and SAME padding on a 3x3x3 input (picks corners and sampled
    # elements), and 2x2x2 window with stride 3 and VALID padding on a 7x7x7 input. Expected
    # values are fixed lists.
    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 3, 3, 1],
        window=[1, 1, 1],
        strides=[2, 2, 2],
        padding="SAME",
        expected=[1, 3, 7, 9, 19, 21, 25, 27])

    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 7, 7, 7, 1],
        window=[2, 2, 2],
        strides=[3, 3, 3],
        padding="VALID",
        expected=[58, 61, 79, 82, 205, 208, 226, 229])

    self._VerifyValues(
        nn_ops.avg_pool3d,
        input_sizes=[1, 3, 3, 3, 1],
        window=[1, 1, 1],
        strides=[2, 2, 2],
        padding="SAME",
        expected=[1, 3, 7, 9, 19, 21, 25, 27])

    self._VerifyValues(
        nn_ops.avg_pool3d,
        input_sizes=[1, 7, 7, 7, 1],
        window=[2, 2, 2],
        strides=[3, 3, 3],
        padding="VALID",
        expected=[29.5, 32.5, 50.5, 53.5, 176.5, 179.5, 197.5, 200.5])

  def _VerifyGradient(self,
                      pool_func,
                      pool_grad_func,
                      input_sizes,
                      ksize,
                      strides,
                      padding,
                      pool_grad_grad_func=None):
    # [FUNCTION DOC]
    # HELPER: shared check for the gradient tests.
    # Runs the forward pooling op on the CPU to obtain outputs, then computes reference input
    # gradients (and, if a second-order gradient function is given, reference grad-of-grad
    # values) with CPU kernels using ramp-valued gradient inputs. Reruns the gradient (and
    # second-order) op on the XLA test device and asserts closeness (rtol 1e-5/atol 1e-6 for the
    # first order; rtol 1e-4/atol 1e-6 for the second order) and matching shapes.
    # Parameters: pool_func, pool_grad_func, input_sizes (5 dims), ksize and strides (3 values
    # each, padded with 1 for batch/channel), padding string, optional pool_grad_grad_func
    # (default none: skip second-order check). Returns nothing.
    """Verifies the output values of the pooling gradient function.

    Args:
      pool_func: Forward pooling function
      pool_grad_func: Pooling gradient function for pool_grad_func
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      pool_grad_grad_func: Second-order gradient function, if available.
    """
    ksize = [1] + ksize + [1]
    strides = [1] + strides + [1]
    total_size = np.prod(input_sizes)
    x = np.arange(1, total_size + 1, dtype=np.float32).reshape(input_sizes)
    with self.session() as sess:
      # Use the forward pool function to compute some corresponding outputs
      # (needed for the CPU device, and we need the shape in both cases).
      with ops.device("CPU"):
        inputs = array_ops.placeholder(dtypes.float32, shape=input_sizes)
        outputs = pool_func(
            inputs,
            ksize=ksize,
            strides=strides,
            padding=padding)

      output_vals = np.array(sess.run(outputs, {inputs: x}))
      output_gradient_vals = np.arange(
          1, output_vals.size + 1, dtype=np.float32)
      output_gradient_vals = output_gradient_vals.reshape(output_vals.shape)
      output_grad_grad_vals = np.arange(1, x.size + 1, dtype=np.float32)
      output_grad_grad_vals = output_grad_grad_vals.reshape(x.shape)

      # Use the Tensorflow CPU pooling gradient to compute the expected input
      # gradients.
      with ops.device("CPU"):
        output_gradients = array_ops.placeholder(
            dtypes.float32, shape=output_vals.shape)
        expected_input_gradients = pool_grad_func(
            inputs,
            outputs,
            output_gradients,
            ksize=ksize,
            strides=strides,
            padding=padding)
        expected_input_gradient_vals = sess.run(
            expected_input_gradients,
            {inputs: x,
             output_gradients: output_gradient_vals})

        output_grad_gradients = array_ops.placeholder(
            dtypes.float32, shape=expected_input_gradient_vals.shape)
        if pool_grad_grad_func is not None:
          expected_grad_gradients = pool_grad_grad_func(
              inputs,
              outputs,
              output_grad_gradients,
              ksize=ksize,
              strides=strides,
              padding=padding,
              data_format="NDHWC")
          expected_grad_gradients_vals = sess.run(expected_grad_gradients, {
              inputs: x,
              output_grad_gradients: output_grad_grad_vals
          })

      # Run the gradient op on the XLA device
      with self.test_scope():
        outputs = array_ops.placeholder(dtypes.float32, shape=output_vals.shape)
        actual_input_gradients = pool_grad_func(
            inputs,
            outputs,
            output_gradients,
            ksize=ksize,
            strides=strides,
            padding=padding)
        if pool_grad_grad_func is not None:
          actual_grad_gradients = pool_grad_grad_func(
              inputs,
              outputs,
              output_grad_gradients,
              ksize=ksize,
              strides=strides,
              padding=padding,
              data_format="NDHWC")

      actual = sess.run(actual_input_gradients, {
          inputs: x,
          outputs: output_vals,
          output_gradients: output_gradient_vals
      })

      # Compare the Tensorflow and XLA results.
      self.assertAllClose(
          expected_input_gradient_vals.flatten(),
          actual.flatten(),
          rtol=1e-5,
          atol=1e-6)
      self.assertShapeEqual(actual, inputs)

      if pool_grad_grad_func is not None:
        actual_grad_gradients_vals = sess.run(
            actual_grad_gradients, {
                inputs: x,
                outputs: output_vals,
                output_grad_gradients: output_grad_grad_vals
            })

        # Compare the Tensorflow and XLA results.
        self.assertAllClose(
            expected_grad_gradients_vals,
            actual_grad_gradients_vals,
            rtol=1e-4,
            atol=1e-6)
        self.assertShapeEqual(actual_grad_gradients_vals, outputs)

  def testMaxPoolGradValidPadding1_1_3d(self):
    # [FUNCTION DOC]
    # Max-pool gradient and second-order gradient vs CPU: window 1, stride 1, VALID, input
    # 1x3x3x3x1.
    self._VerifyGradient(
        nn_ops.max_pool3d,
        gen_nn_ops.max_pool3d_grad,
        input_sizes=[1, 3, 3, 3, 1],
        ksize=[1, 1, 1],
        strides=[1, 1, 1],
        padding="VALID",
        pool_grad_grad_func=gen_nn_ops.max_pool3d_grad_grad)

  def testMaxPoolGradValidPadding2_1_6_3d(self):
    # [FUNCTION DOC]
    # Max-pool gradient and second-order gradient vs CPU: window 2, stride 1, VALID, input
    # 2x3x3x6x3.
    self._VerifyGradient(
        nn_ops.max_pool3d,
        gen_nn_ops.max_pool3d_grad,
        input_sizes=[2, 3, 3, 6, 3],
        ksize=[2, 2, 2],
        strides=[1, 1, 1],
        padding="VALID",
        pool_grad_grad_func=gen_nn_ops.max_pool3d_grad_grad)

  def testMaxPoolGradValidPadding2_1_7_3d(self):
    # [FUNCTION DOC]
    # Max-pool first-order gradient vs CPU: window 2, stride 1, VALID, input 2x3x5x7x3. The
    # second-order gradient is deliberately not checked because the bfloat16 implementation lacks
    # enough precision (see the TODO inside).
    # TODO(b/73062247): the bfloat16 implementation of MaxPool3DGradGrad does
    # not have enough precision for this test case to pass if
    # pool_grad_grad_func is passed.
    self._VerifyGradient(
        nn_ops.max_pool3d,
        gen_nn_ops.max_pool3d_grad,
        input_sizes=[2, 3, 5, 7, 3],
        ksize=[2, 2, 2],
        strides=[1, 1, 1],
        padding="VALID")

  def testMaxPoolGradValidPadding2_2_3d(self):
    # [FUNCTION DOC]
    # Max-pool gradient and second-order gradient vs CPU: window 2, stride 2, VALID, input
    # 2x2x2x2x3.
    self._VerifyGradient(
        nn_ops.max_pool3d,
        gen_nn_ops.max_pool3d_grad,
        input_sizes=[2, 2, 2, 2, 3],
        ksize=[2, 2, 2],
        strides=[2, 2, 2],
        padding="VALID",
        pool_grad_grad_func=gen_nn_ops.max_pool3d_grad_grad)

  def testMaxPoolGradSamePadding1_1_3d(self):
    # [FUNCTION DOC]
    # Max-pool gradient and second-order gradient vs CPU: window 1, stride 1, SAME, input
    # 2x3x2x4x1.
    self._VerifyGradient(
        nn_ops.max_pool3d,
        gen_nn_ops.max_pool3d_grad,
        input_sizes=[2, 3, 2, 4, 1],
        ksize=[1, 1, 1],
        strides=[1, 1, 1],
        padding="SAME",
        pool_grad_grad_func=gen_nn_ops.max_pool3d_grad_grad)

  def testMaxPoolGradSamePadding2_1_3d(self):
    # [FUNCTION DOC]
    # Max-pool gradient and second-order gradient vs CPU: window 2, stride 1, SAME, input
    # 2x3x2x4x1.
    self._VerifyGradient(
        nn_ops.max_pool3d,
        gen_nn_ops.max_pool3d_grad,
        input_sizes=[2, 3, 2, 4, 1],
        ksize=[2, 2, 2],
        strides=[1, 1, 1],
        padding="SAME",
        pool_grad_grad_func=gen_nn_ops.max_pool3d_grad_grad)

  def testMaxPoolGradSamePadding2_2_3d(self):
    # [FUNCTION DOC]
    # Max-pool gradient and second-order gradient vs CPU: window 2, stride 2, SAME, input
    # 2x5x2x4x3.
    self._VerifyGradient(
        nn_ops.max_pool3d,
        gen_nn_ops.max_pool3d_grad,
        input_sizes=[2, 5, 2, 4, 3],
        ksize=[2, 2, 2],
        strides=[2, 2, 2],
        padding="SAME",
        pool_grad_grad_func=gen_nn_ops.max_pool3d_grad_grad)

  def testMaxPoolGradSamePadding3_1_3d(self):
    # [FUNCTION DOC]
    # Max-pool gradient and second-order gradient vs CPU: window 3, stride 1, SAME, input
    # 1x3x3x7x1.
    self._VerifyGradient(
        nn_ops.max_pool3d,
        gen_nn_ops.max_pool3d_grad,
        input_sizes=[1, 3, 3, 7, 1],
        ksize=[3, 3, 3],
        strides=[1, 1, 1],
        padding="SAME",
        pool_grad_grad_func=gen_nn_ops.max_pool3d_grad_grad)

  def testAvgPoolGradValidPadding1_1_3d(self):
    # [FUNCTION DOC]
    # Average-pool gradient vs CPU: window 1, stride 1, VALID, input 2x3x3x3x3.
    self._VerifyGradient(
        nn_ops.avg_pool3d,
        _AvgPoolGrad,
        input_sizes=[2, 3, 3, 3, 3],
        ksize=[1, 1, 1],
        strides=[1, 1, 1],
        padding="VALID")

  def testAvgPoolGradValidPadding2_1_3d(self):
    # [FUNCTION DOC]
    # Average-pool gradient vs CPU: window 2, stride 1, VALID, input 2x3x3x3x3.
    self._VerifyGradient(
        nn_ops.avg_pool3d,
        _AvgPoolGrad,
        input_sizes=[2, 3, 3, 3, 3],
        ksize=[2, 2, 2],
        strides=[1, 1, 1],
        padding="VALID")

  def testAvgPoolGradValidPadding2_2_3d(self):
    # [FUNCTION DOC]
    # Average-pool gradient vs CPU: window 2, stride 2, VALID, input 2x2x2x2x3.
    self._VerifyGradient(
        nn_ops.avg_pool3d,
        _AvgPoolGrad,
        input_sizes=[2, 2, 2, 2, 3],
        ksize=[2, 2, 2],
        strides=[2, 2, 2],
        padding="VALID")

  def testAvgPoolGradSamePadding1_1_3d(self):
    # [FUNCTION DOC]
    # Average-pool gradient vs CPU: window 1, stride 1, SAME, input 2x3x2x4x3.
    self._VerifyGradient(
        nn_ops.avg_pool3d,
        _AvgPoolGrad,
        input_sizes=[2, 3, 2, 4, 3],
        ksize=[1, 1, 1],
        strides=[1, 1, 1],
        padding="SAME")

  def testAvgPoolGradSamePadding2_1_3d(self):
    # [FUNCTION DOC]
    # Average-pool gradient vs CPU: window 2, stride 1, SAME, input 1x2x2x2x1.
    self._VerifyGradient(
        nn_ops.avg_pool3d,
        _AvgPoolGrad,
        input_sizes=[1, 2, 2, 2, 1],
        ksize=[2, 2, 2],
        strides=[1, 1, 1],
        padding="SAME")

  def testAvgPoolGradSamePadding2_2_3d(self):
    # [FUNCTION DOC]
    # Average-pool gradient vs CPU: window 2, stride 2, SAME, input 2x5x2x4x3.
    self._VerifyGradient(
        nn_ops.avg_pool3d,
        _AvgPoolGrad,
        input_sizes=[2, 5, 2, 4, 3],
        ksize=[2, 2, 2],
        strides=[2, 2, 2],
        padding="SAME")

  def testAvgPoolGradSamePadding3_1_3d(self):
    # [FUNCTION DOC]
    # Average-pool gradient vs CPU: window 3, stride 1, SAME, input 1x3x6x7x1.
    self._VerifyGradient(
        nn_ops.avg_pool3d,
        _AvgPoolGrad,
        input_sizes=[1, 3, 6, 7, 1],
        ksize=[3, 3, 3],
        strides=[1, 1, 1],
        padding="SAME")


if __name__ == "__main__":
  test.main()
