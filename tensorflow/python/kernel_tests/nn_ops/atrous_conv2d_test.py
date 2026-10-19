# [MODULE DOC]
# Kernel tests for atrous (dilated) 2-D convolution in the nn ops: atrous_conv2d,
# atrous_conv2d_transpose and the dilated depthwise convolution. Each forward test compares the
# atrous op against a regular convolution with a zero-upsampled filter, one test checks the
# space-to-batch equivalence of a sequence of atrous convolutions, one checks numeric gradients,
# and two check that absurdly large rates are rejected. Contains a module-level helper that
# upsamples filters. No import-time side effects beyond importing the nn gradient registrations.
# Cases run in deprecated graph mode (v1) mostly, with a session.
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
"""Tests for convolution related functionality in tensorflow.ops.nn."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def _upsample_filters(filters, rate):
  # [FUNCTION DOC]
  # HELPER: builds the equivalent dense filter used as the reference in the atrous convolution tests.
  # Takes a numpy filter array laid out as height, width, input depth, output depth and an integer
  # rate. With rate one it returns the same array unchanged. Otherwise it inserts rate minus one
  # zeros between consecutive filter taps along height and width (via a Kronecker product with a
  # single-one kernel, trimming the trailing zeros) and returns the new array with shape
  # height + (height-1)*(rate-1) by the analogous width. No side effects, no exceptions of its own.
  """Upsamples the filters by a factor of rate along the spatial dimensions.

  Args:
    filters: [h, w, in_depth, out_depth]. Original filters.
    rate: An int, specifying the upsampling rate.

  Returns:
    filters_up: [h_up, w_up, in_depth, out_depth]. Upsampled filters with
      h_up = h + (h - 1) * (rate - 1)
      w_up = w + (w - 1) * (rate - 1)
      containing (rate - 1) zeros between consecutive filter values along
      the filters' spatial dimensions.
  """
  if rate == 1:
    return filters
  # [h, w, in_depth, out_depth] -> [in_depth, out_depth, h, w]
  filters_up = np.transpose(filters, [2, 3, 0, 1])
  ker = np.zeros([rate, rate], dtype=np.float32)
  ker[0, 0] = 1
  filters_up = np.kron(filters_up, ker)[:, :, :-(rate - 1), :-(rate - 1)]
  # [in_depth, out_depth, h_up, w_up] -> [h_up, w_up, in_depth, out_depth]
  filters_up = np.transpose(filters_up, [2, 3, 0, 1])
  return filters_up


class AtrousConv2DTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the atrous_conv2d op: forward equivalence with a conv on an upsampled
  # filter, equivalence with the space-to-batch formulation, gradient accuracy and invalid-rate
  # rejection. No shared fixtures; each test opens its own session.

  @test_util.run_deprecated_v1
  def testAtrousConv2DForward(self):
    # [FUNCTION DOC]
    # Scenario: for odd and even widths, kernel sizes one to three and rates one to three with
    # SAME and VALID padding, an arange-filled input and filter are used. Runs atrous_conv2d and
    # a regular conv2d with unit strides on the zero-upsampled filter. Asserts the two outputs
    # are close (tolerance 1e-3). Returns nothing; runs in a v1 graph session.
    with self.session():
      # Input: [batch, height, width, input_depth]
      height = 9
      for width in [9, 10]:  # Test both odd and even width.
        x_shape = [2, height, width, 2]
        x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)

        # Filter: [kernel_height, kernel_width, input_depth, output_depth]
        for kernel_height in range(1, 4):
          for kernel_width in range(1, 4):
            f_shape = [kernel_height, kernel_width, 2, 2]
            f = np.arange(np.prod(f_shape), dtype=np.float32).reshape(f_shape)

            for rate in range(1, 4):
              f_up = _upsample_filters(f, rate)

              for padding in ["SAME", "VALID"]:
                y1 = nn_ops.atrous_conv2d(x, f, rate, padding=padding)
                y2 = nn_ops.conv2d(
                    x, f_up, strides=[1, 1, 1, 1], padding=padding)
                self.assertAllClose(y1, y2, rtol=1e-3, atol=1e-3)

  @test_util.run_deprecated_v1
  def testAtrousSequence(self):
    # [FUNCTION DOC]
    # Scenario: seeds numpy for reproducibility, then for several input sizes, odd kernel sizes
    # and rates two and three applies three consecutive SAME-padded atrous convolutions with the
    # same random filter. Computes the reference by padding so dims divide by the rate, doing
    # space-to-batch, three ordinary SAME convolutions, then batch-to-space with matching crops.
    # Asserts the two results are close (tolerance 1e-2). Side effect: sets the numpy random seed.
    """Tests optimization of sequence of atrous convolutions.

    Verifies that a sequence of `atrous_conv2d` operations with identical `rate`
    parameters, 'SAME' `padding`, and `filters` with odd heights/ widths:

        net = atrous_conv2d(net, filters1, rate, padding="SAME")
        net = atrous_conv2d(net, filters2, rate, padding="SAME")
        ...
        net = atrous_conv2d(net, filtersK, rate, padding="SAME")

    is equivalent to:

        pad = ...  # padding so that the input dims are multiples of rate
        net = space_to_batch(net, paddings=pad, block_size=rate)
        net = conv2d(net, filters1, strides=[1, 1, 1, 1], padding="SAME")
        net = conv2d(net, filters2, strides=[1, 1, 1, 1], padding="SAME")
        ...
        net = conv2d(net, filtersK, strides=[1, 1, 1, 1], padding="SAME")
        net = batch_to_space(net, crops=pad, block_size=rate)
    """
    padding = "SAME"  # The padding needs to be "SAME"
    np.random.seed(1)  # Make it reproducible.

    with self.session():
      # Input: [batch, height, width, input_depth]
      for height in range(15, 17):
        for width in range(15, 17):
          x_shape = [3, height, width, 2]
          x = np.random.random_sample(x_shape).astype(np.float32)

          for kernel in [1, 3, 5]:  # The kernel size needs to be odd.
            # Filter: [kernel_height, kernel_width, input_depth, output_depth]
            f_shape = [kernel, kernel, 2, 2]
            f = 1e-2 * np.random.random_sample(f_shape).astype(np.float32)

            for rate in range(2, 4):
              # y1: three atrous_conv2d in a row.
              y1 = nn_ops.atrous_conv2d(x, f, rate, padding=padding)
              y1 = nn_ops.atrous_conv2d(y1, f, rate, padding=padding)
              y1 = nn_ops.atrous_conv2d(y1, f, rate, padding=padding)
              # y2: space_to_batch, three conv2d in a row, batch_to_space
              pad_bottom = 0 if height % rate == 0 else rate - height % rate
              pad_right = 0 if width % rate == 0 else rate - width % rate
              pad = [[0, pad_bottom], [0, pad_right]]
              y2 = array_ops.space_to_batch(x, paddings=pad, block_size=rate)
              y2 = nn_ops.conv2d(y2, f, strides=[1, 1, 1, 1], padding=padding)
              y2 = nn_ops.conv2d(y2, f, strides=[1, 1, 1, 1], padding=padding)
              y2 = nn_ops.conv2d(y2, f, strides=[1, 1, 1, 1], padding=padding)
              y2 = array_ops.batch_to_space(y2, crops=pad, block_size=rate)
              self.assertAllClose(y1, y2, rtol=1e-2, atol=1e-2)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario: random float32 input and filter constants (fixed numpy seed), atrous_conv2d with
    # SAME padding for rates one to three. Computes the numerical-vs-analytic gradient error with
    # respect to both input and filter, prints it, and asserts it is below 1e-3 (4e-3 when XLA is
    # enabled). Side effects: seeds numpy and prints to standard output.
    with self.session():
      # Input: [batch, height, width, input_depth]
      x_shape = [2, 5, 6, 2]
      # Filter: [kernel_height, kernel_width, input_depth, output_depth]
      f_shape = [3, 3, 2, 2]
      # Output: [batch, height, width, output_depth]
      y_shape = [2, 5, 6, 2]

      np.random.seed(1)  # Make it reproducible.
      x_val = np.random.random_sample(x_shape).astype(np.float32)
      f_val = np.random.random_sample(f_shape).astype(np.float32)
      x = constant_op.constant(x_val, name="x", dtype=dtypes.float32)
      f = constant_op.constant(f_val, name="f", dtype=dtypes.float32)

      for rate in range(1, 4):
        output = nn_ops.atrous_conv2d(x, f, rate=rate, padding="SAME")
        err = gradient_checker.compute_gradient_error([x, f],
                                                      [x_shape, f_shape],
                                                      output, y_shape)
        print("atrous_conv2d gradient err = %g " % err)
        err_tolerance = 4e-3 if test_util.is_xla_enabled() else 1e-3
        self.assertLess(err, err_tolerance)

  @test_util.run_deprecated_v1
  def testAtrousConv2DInvalid(self):
    # [FUNCTION DOC]
    # Scenario: calls atrous_conv2d on a tiny all-ones input and filter with an enormous rate
    # (the maximum 32-bit integer) and SAME padding, then evaluates the result. Asserts that
    # either an invalid-argument error or a value error is raised (no crash or overflow).
    with self.session():
      with self.assertRaises((errors.InvalidArgumentError, ValueError)):
        op = nn_ops.atrous_conv2d(
            value=np.ones((1, 1, 1, 5)),
            filters=np.ones((1, 1, 5, 1)),
            rate=2147483647,
            padding="SAME")
        self.evaluate(op)


class AtrousConv2DTransposeTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering atrous_conv2d_transpose: forward equivalence with conv2d_transpose on an
  # upsampled filter, and rejection of an absurdly large rate. No shared fixtures.

  @test_util.run_deprecated_v1
  def testAtrousConv2DTransposeForward(self):
    # [FUNCTION DOC]
    # Scenario: over odd and even widths, kernel sizes one to three, rates one to three and SAME or
    # VALID padding, computes the expected output shape (input size for SAME, input plus the
    # dilated kernel size minus one for VALID). Runs atrous_conv2d_transpose and a regular
    # conv2d_transpose with unit strides on the upsampled filter and asserts the outputs are close
    # (tolerance 1e-3). Returns nothing; v1 graph-mode session.
    with self.session():
      # Input: [batch, height, width, input_depth]
      height = 9
      for width in [9, 10]:  # Test both odd and even width.
        x_shape = [2, height, width, 2]
        x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)

        # Filter: [kernel_height, kernel_width, input_depth, output_depth]
        for kernel_height in range(1, 4):
          for kernel_width in range(1, 4):
            f_shape = [kernel_height, kernel_width, 2, 2]
            f = np.arange(np.prod(f_shape), dtype=np.float32).reshape(f_shape)

            for rate in range(1, 4):
              f_up = _upsample_filters(f, rate)
              kernel_height_up = (kernel_height + (kernel_height - 1) *
                                  (rate - 1))
              kernel_width_up = kernel_width + (kernel_width - 1) * (rate - 1)

              for padding in ["SAME", "VALID"]:
                if padding == "SAME":
                  y_shape = [2, height, width, 2]
                else:
                  y_shape = [
                      2, height + kernel_height_up - 1,
                      width + kernel_width_up - 1, 2
                  ]

                y1 = nn_ops.atrous_conv2d_transpose(x, f, y_shape, rate,
                                                    padding)
                y2 = nn_ops.conv2d_transpose(
                    x, f_up, y_shape, strides=[1, 1, 1, 1], padding=padding)
                self.assertAllClose(y1, y2, rtol=1e-3, atol=1e-3)

  def testAtrousConv2DTransposeInvalid(self):
    # [FUNCTION DOC]
    # Scenario: calls atrous_conv2d_transpose with a tiny input, a huge rate and a mismatching
    # output shape, then evaluates it. Asserts an invalid-argument or value error is raised.
    # Unlike its siblings it is not decorated for deprecated v1 mode.
    with self.session():
      with self.assertRaises((errors.InvalidArgumentError, ValueError)):
        op = nn_ops.atrous_conv2d_transpose(
            value=np.ones((10, 1, 1, 1)),
            filters=np.ones((1, 1, 1, 1)),
            rate=1356819205,
            padding="SAME",
            output_shape=[1, 1, 1, 1])
        self.evaluate(op)


class AtrousDepthwiseConv2DTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the dilated depthwise_conv2d of nn_impl; a single forward-equivalence test.

  @test_util.run_deprecated_v1
  def testAtrousDepthwiseConv2DForward(self):
    # [FUNCTION DOC]
    # Scenario: for odd and even widths, kernel sizes one to three, rates one to three and SAME or
    # VALID padding, runs depthwise_conv2d with a dilation rate on the original filter and a
    # plain depthwise_conv2d on the zero-upsampled filter (unit strides). Asserts the outputs are
    # close (tolerance 1e-3). Returns nothing; v1 graph-mode session.
    strides = [1, 1, 1, 1]
    with self.session():
      # Input: [batch, height, width, input_depth]
      height = 9
      for width in [9, 10]:  # Test both odd and even width.
        x_shape = [2, height, width, 2]
        x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)

        # Filter: [kernel_height, kernel_width, input_depth, output_depth]
        for kernel_height in range(1, 4):
          for kernel_width in range(1, 4):
            f_shape = [kernel_height, kernel_width, 2, 2]
            f = np.arange(np.prod(f_shape), dtype=np.float32).reshape(f_shape)

            for rate in range(1, 4):
              f_up = _upsample_filters(f, rate)

              for padding in ["SAME", "VALID"]:
                y1 = nn_impl.depthwise_conv2d(
                    x, f, strides, padding, rate=[rate, rate])
                y2 = nn_impl.depthwise_conv2d(x, f_up, strides, padding)
                self.assertAllClose(y1, y2, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  test.main()
