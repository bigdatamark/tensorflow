# [MODULE DOC]
# HELPER: shared test-support module for the depthwise convolution test files.
# Provides a NumPy reference implementation of depthwise 2-D convolution (basic, plus a version
# handling padding, dilation and data format), several functions returning lists of test
# configurations (input, filter and output shapes, stride, padding, dilations) for forward and
# gradient checks with implicit and explicit padding, and a base test case class
# (DepthwiseConv2DBase) with value tests against the reference and hand-computed results, numeric
# gradient checks for the input and filter, and CPU-versus-GPU comparisons.
# It does not define a test runner entry point. The non-deterministic test module aliases the base
# class and the determinism test module subclasses it.
# Import-time side effect: importing the neural network gradient module so gradients are registered.
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
"""Functional tests for depthwise convolutional operations."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _DepthwiseConv2dNumpyBasic(x1, x2, strides):
  # [FUNCTION DOC]
  # HELPER: reference (slow, loop-based) depthwise convolution used to check TensorFlow results.
  # Takes an NHWC input array x1, a filter array x2 shaped height, width, input channels, channel
  # multiplier, and a strides list of four entries (only the two spatial strides are used). Applies
  # no padding and no dilation: output size per spatial dimension is (input - filter + stride)
  # divided by stride using integer division. Each input channel is convolved only with its own
  # filter slices, producing channel-multiplier output channels, laid out channel-major.
  # Returns a floating point array of shape batch, out rows, out columns, channels times multiplier.
  # Asserts (raising an assertion error) that the input channel count equals the filter's.
  """Compute depthwise_conv2d using Numpy.

  This allows use to test TensorFlow's depthwise_conv2d by comparing to the
  Numpy version.

  Args:
    x1: The input Numpy array, in NHWC format.
    x2: The filter Numpy array.
    strides: A Python list of 4 elements representing the strides.

  Returns:
    The depthwise conv2d output as a Numpy array.
  """
  n, h, w, c = x1.shape
  fh, fw, c2, o = x2.shape
  assert c == c2
  _, sh, sw, _ = strides
  out_rows = (h - fh + sh) // sh
  out_cols = (w - fw + sw) // sw
  out = np.zeros([n, out_rows, out_cols, c * o])
  for i in range(out_rows):
    for j in range(out_cols):
      for k in range(c):
        start_height = i * sh
        end_height = start_height + fh
        start_width = j * sw
        end_width = start_width + fw
        # multiplied_slice.shape: (b, fh, fw, o)
        multiplied_slice = (
            x1[:, start_height:end_height, start_width:end_width, k, np.newaxis]
            * x2[:, :, k, :])
        # Set a slice of b * o elements of 'out'.
        out[:, i, j, k * o:(k + 1) * o] = np.sum(multiplied_slice, axis=(1, 2))
  return out


def _DepthwiseConv2dNumpy(x1, x2, strides, padding, data_format, dilations):
  # [FUNCTION DOC]
  # HELPER: reference depthwise convolution supporting padding, dilation and data format.
  # KNOWN ISSUE: the data-format handling looks inverted. For the layout name NCHW it transposes
  # the input with the NHWC-to-NCHW permutation (comment claims conversion to NHWC) and transposes
  # the result back with the opposite permutation. It only works for NHWC, which is the only
  # layout callers in this module pass.
  # Inputs: x1 input array, x2 filter array, strides list of four, padding either the name SAME,
  # the name VALID, or a list of four (before, after) pairs, data_format name, dilations list of
  # two or nothing. Dilation is applied by spreading the filter with zeros; SAME padding is
  # computed per dimension with the extra element placed after; VALID uses no padding. Pads the
  # input with zeros and delegates to the basic reference. Returns the convolution output array.
  """Compute depthwise_conv2d using Numpy.

  This allows use to test TensorFlow's depthwise_conv2d by comparing to the
  Numpy version.

  Unlike `_DepthwiseConv2dNumpyBasic`, this supports more advanced features
  like padding.

  Args:
    x1: The input Numpy array.
    x2: The filter Numpy array.
    strides: A Python list of 4 elements representing the strides.
    padding: The padding. "SAME", "VALID", or a list of explicit paddings.
    data_format: "NHWC" or "NCHW".
    dilations: A list of 2 elements, representing the dilations.

  Returns:
    The depthwise conv2d as a Numpy array.
  """
  if data_format == "NCHW":
    # Transpose arguments to NHWC format.
    x1 = np.transpose(x1, (0, 3, 1, 2))
    strides = [strides[0], strides[3], strides[1], strides[2]]
    if dilations:
      dilations = [dilations[0], dilations[3], dilations[1], dilations[2]]

  if dilations:
    # Dilate the filter so _DepthwiseConv2dNumpyBasic doesn't have to deal with
    # dilations.
    fh, fw, c, o = x2.shape
    new_fh = (fh - 1) * dilations[0] + 1
    new_fw = (fw - 1) * dilations[1] + 1
    new_x2 = np.zeros((new_fh, new_fw, c, o))
    for i in range(fh):
      for j in range(fw):
        new_x2[i * dilations[0], j * dilations[1], ::] = x2[i, j, :, :]
    x2 = new_x2

  # Pad input so _DepthwiseConv2dNumpyBasic doesn't have to deal with padding.
  if padding == "SAME":

    def PaddingsForDim(input_dim, filter_dim, stride):
      # [FUNCTION DOC]
      # HELPER: computes SAME padding amounts for one spatial dimension.
      # Takes the input size, the (possibly dilated) filter size and the stride, all integers.
      # Total padding is the filter size minus the stride if the input divides evenly by the
      # stride, otherwise filter size minus the remainder, never below zero. Returns a pair of
      # padding before (half, rounded down) and padding after (the rest).
      """Computes paddings for a single dimension."""
      if input_dim % stride == 0:
        total_padding = max(filter_dim - stride, 0)
      else:
        total_padding = max(filter_dim - (input_dim % stride), 0)
      pad_before = total_padding // 2
      pad_after = total_padding - pad_before
      return pad_before, pad_after

    padding = [(0, 0),
               PaddingsForDim(x1.shape[1], x2.shape[0], strides[1]),
               PaddingsForDim(x1.shape[2], x2.shape[1], strides[2]), (0, 0)]
  elif padding == "VALID":
    padding = [(0, 0)] * 4
  x1 = np.pad(x1, padding, "constant")

  y = _DepthwiseConv2dNumpyBasic(x1, x2, strides)

  if data_format == "NCHW":
    # Transpose back to NCHW format.
    y = np.transpose(y, (0, 2, 3, 1))

  return y


def ConfigsToTest():
  # [FUNCTION DOC]
  # HELPER: supplies the list of test configurations for the forward depthwise convolution tests.
  # KNOWN ISSUE: the docstring calls this an iterator but it builds and returns a plain list.
  # Takes no inputs and has no side effects. Returns a list of tuples of input shape, filter
  # shape, expected output shape, stride, padding name (SAME or VALID) and dilations (or nothing).
  # Covers 1x1, 3x3 and 5x5 filters, stride 1 to 3, valid padding and dilated cases.
  """Iterator for different convolution shapes, strides and paddings.

  Returns:
    List of tuples (input_size, filter_size, out_size, stride, padding,
    dilations), the depthwise convolution parameters.
  """

  def Config(input_size,
             filter_size,
             out_size,
             stride=1,
             padding="SAME",
             dilations=None):
    # [FUNCTION DOC]
    # HELPER: packs one test configuration into a tuple.
    # Takes input shape, filter shape, output shape, stride (default 1), padding name (default
    # SAME) and dilations (default nothing) and returns them unchanged as a six-element tuple.
    return input_size, filter_size, out_size, stride, padding, dilations

  return [
      Config([4, 5, 5, 48], [1, 1, 48, 2], [4, 5, 5, 96]),
      Config([4, 8, 8, 84], [1, 3, 84, 1], [4, 8, 8, 84]),
      Config([4, 17, 17, 48], [3, 1, 48, 4], [4, 17, 17, 192]),
      Config([4, 9, 27, 8], [3, 3, 8, 1], [4, 9, 27, 8]),
      Config([4, 31, 31, 7], [3, 3, 7, 1], [4, 31, 31, 7]),
      Config([4, 35, 35, 2], [5, 5, 2, 1], [4, 35, 35, 2]),
      Config([4, 147, 147, 2], [3, 3, 2, 8], [4, 49, 49, 16],
             3,
             padding="VALID"),
      Config([3, 299, 299, 3], [3, 2, 3, 8], [3, 150, 150, 24], 2),
      Config([5, 183, 183, 1], [5, 5, 1, 2], [5, 92, 92, 2], 2),
      Config([5, 183, 183, 1], [5, 5, 1, 2], [5, 183, 183, 2], dilations=[2,
                                                                          2]),
      Config([5, 41, 35, 2], [4, 7, 2, 2], [5, 32, 23, 4],
             padding="VALID",
             dilations=[3, 2]),
  ]


def ConfigsToTestExplicit():
  # [FUNCTION DOC]
  # HELPER: supplies test configurations using explicit padding for the forward tests.
  # KNOWN ISSUE: the docstring calls this an iterator but it returns a plain list.
  # Takes no inputs, no side effects. Returns a list of tuples of input shape, filter shape,
  # expected output shape, stride, explicit padding (a pair of before/after pairs for height and
  # width) and dilations (or nothing). Includes small-input cases that exercise the GPU kernels
  # used for small inputs.
  """Iterator for different convolution shapes, strides and explicit paddings.

  Returns:
    List of tuples (input_size, filter_size, out_size, stride, padding,
    dilations), the depthwise convolution parameters.
  """

  def Config(input_size,
             filter_size,
             out_size,
             stride=1,
             padding=None,
             dilations=None):
    # [FUNCTION DOC]
    # HELPER: packs one explicit-padding test configuration into a tuple.
    # Takes input shape, filter shape, output shape, stride (default 1), padding (default
    # nothing) and dilations (default nothing) and returns them unchanged as a six-element tuple.
    return input_size, filter_size, out_size, stride, padding, dilations

  return [
      Config([4, 5, 5, 48], [1, 1, 48, 2], [4, 8, 12, 96],
             padding=[[1, 2], [3, 4]]),
      Config([4, 1, 1, 3], [3, 3, 3, 2], [4, 29, 39, 6],
             padding=[[10, 20], [15, 25]]),
      Config([4, 9, 27, 8], [3, 3, 8, 1], [4, 14, 31, 8],
             padding=[[3, 4], [4, 2]]),
      Config([4, 31, 31, 7], [3, 3, 7, 1], [4, 29, 29, 7],
             padding=[[0, 0], [0, 0]]),
      Config([3, 299, 299, 3], [3, 2, 3, 8], [3, 150, 153, 24],
             2,
             padding=[[1, 2], [3, 5]]),
      Config([5, 183, 183, 1], [5, 5, 1, 2], [5, 62, 60, 2],
             3,
             padding=[[3, 2], [1, 0]]),
      Config([5, 29, 31, 1], [5, 4, 1, 2], [5, 26, 23, 2],
             padding=[[3, 2], [1, 0]],
             dilations=[2, 3]),
      # These cases test the kernels in depthwise_conv_op_gpu.h which are used
      # if the input size is small.
      Config([4, 5, 5, 48], [3, 3, 48, 1], [4, 5, 5, 48],
             padding=[[0, 2], [0, 2]]),
      Config([1, 8, 7, 2], [8, 7, 2, 1], [1, 8, 7, 2], padding=[[0, 7], [3,
                                                                         3]]),
      Config([2, 4, 3, 2], [3, 2, 2, 1], [2, 4, 3, 2], padding=[[2, 0], [1,
                                                                         0]]),
  ]


def CheckGradConfigsToTest():
  # [FUNCTION DOC]
  # HELPER: supplies small test configurations for the numeric gradient tests, which are costly.
  # KNOWN ISSUE: the docstring calls this an iterator but it returns a plain list.
  # Takes no inputs, no side effects. Returns a list of tuples of input shape, filter shape,
  # output shape, stride, padding name (SAME or VALID) and dilations (or nothing).
  """Iterator for different convolution shapes, strides and paddings.

  compute_gradient_error() is very expensive. So the configs should be
  relatively small.

  Returns:
    List of tuples (input_size, filter_size, out_size, stride, padding,
    dilations), the depthwise convolution parameters.
  """

  def Config(input_size,
             filter_size,
             out_size,
             stride=1,
             padding="SAME",
             dilations=None):
    # [FUNCTION DOC]
    # HELPER: packs one gradient-test configuration into a tuple.
    # Takes input shape, filter shape, output shape, stride (default 1), padding name (default
    # SAME) and dilations (default nothing) and returns them unchanged as a six-element tuple.
    return input_size, filter_size, out_size, stride, padding, dilations

  return [
      Config([2, 5, 8, 1], [4, 4, 1, 2], [2, 5, 8, 2]),
      Config([4, 5, 5, 1], [2, 2, 1, 2], [4, 2, 2, 2], 2, padding="VALID"),
      Config([2, 4, 4, 2], [3, 1, 2, 2], [2, 4, 4, 4]),
      Config([1, 15, 15, 2], [1, 3, 2, 1], [1, 15, 15, 2]),
      Config([2, 15, 16, 1], [3, 3, 1, 2], [2, 5, 5, 2], 3, padding="VALID"),
      Config([2, 5, 8, 1], [4, 3, 1, 2], [2, 5, 8, 2], dilations=[1, 2]),
      # These cases test the kernels in depthwise_conv_op_gpu.h which are used
      # if the input size is small.
      Config([1, 3, 1, 2], [2, 1, 2, 1], [1, 3, 1, 2]),
      Config([2, 2, 3, 2], [2, 1, 2, 1], [2, 2, 3, 2]),
      Config([2, 2, 3, 1], [2, 2, 1, 1], [2, 2, 3, 1]),
  ]


def CheckGradConfigsToTestExplicit():
  # [FUNCTION DOC]
  # HELPER: supplies small explicit-padding configurations for the numeric gradient tests.
  # KNOWN ISSUE: the docstring calls this an iterator but it returns a plain list.
  # Takes no inputs, no side effects. Returns a list of tuples of input shape, filter shape,
  # output shape, stride, explicit padding (before/after pairs for height and width) and
  # dilations (or nothing).
  """Iterator for different convolution shapes, strides and explicit paddings.

  compute_gradient_error() is very expensive. So the configs should be
  relatively small.

  Returns:
    List of tuples (input_size, filter_size, out_size, stride, padding,
    dilations), the depthwise convolution parameters.
  """

  def Config(input_size,
             filter_size,
             out_size,
             stride=1,
             padding=None,
             dilations=None):
    # [FUNCTION DOC]
    # HELPER: packs one explicit-padding gradient-test configuration into a tuple.
    # Takes input shape, filter shape, output shape, stride (default 1), padding (default
    # nothing) and dilations (default nothing) and returns them unchanged as a six-element tuple.
    return input_size, filter_size, out_size, stride, padding, dilations

  return [
      Config([2, 5, 8, 1], [4, 4, 1, 2], [2, 3, 10, 2],
             padding=[[0, 1], [2, 3]]),
      Config([4, 5, 5, 1], [2, 2, 1, 2], [4, 4, 5, 2],
             2,
             padding=[[3, 1], [5, 0]]),
      Config([2, 4, 4, 2], [3, 1, 2, 2], [2, 7, 11, 4],
             padding=[[4, 1], [3, 4]]),
      Config([1, 15, 15, 2], [1, 3, 2, 1], [1, 18, 23, 2],
             padding=[[3, 0], [2, 8]]),
      Config([2, 15, 16, 1], [3, 3, 1, 2], [2, 5, 8, 2],
             3,
             padding=[[0, 0], [10, 0]]),
      Config([2, 5, 8, 1], [3, 4, 1, 2], [2, 5, 10, 2],
             padding=[[3, 1], [2, 3]],
             dilations=[2, 1]),
      # These cases test the kernels in depthwise_conv_op_gpu.h which are used
      # if the input size is small.
      Config([2, 4, 3, 2], [3, 2, 2, 1], [2, 4, 3, 2], padding=[[2, 0], [1,
                                                                         0]]),
  ]


class DepthwiseConv2DBase(test.TestCase):
  # [CLASS DOC]
  # Base test case for depthwise 2-D convolution. Holds no instance state. Provides helper
  # methods that verify forward values against a NumPy reference or hand-computed values, run
  # numeric gradient checks for input and filter, and compare CPU against GPU results for the
  # forward op and both backprop ops, plus the public tests that drive them over the shared
  # configuration lists (implicit and explicit padding, NHWC and NCHW, several dtypes).
  # Meant to be aliased or subclassed by concrete test modules.
  """Base test class for depthwise Conv2D tests."""

  # This tests depthwise_conv2d and depthwise_conv2d_native
  def _VerifyValues(self,
                    tensor_in_sizes,
                    filter_in_sizes,
                    stride,
                    padding,
                    data_type,
                    use_gpu,
                    grouped_conv=False,
                    data_format="NHWC",
                    dilations=None,
                    tolerance=None):
    # [FUNCTION DOC]
    # HELPER: checks TensorFlow depthwise convolution output against the NumPy reference.
    # Inputs: tensor_in_sizes and filter_in_sizes (lists of four dimensions), stride (integer for
    # both spatial dims), padding (name or explicit list of two before/after pairs), data_type
    # (TensorFlow dtype; must be one of half, single, double or bfloat16 or a key error occurs),
    # use_gpu (bool), grouped_conv (bool, default false, routes the op to the cuDNN grouped kernel
    # label), data_format (NHWC default or NCHW), dilations (optional list of two) and tolerance
    # (optional; defaults per dtype). Fills input and filter with evenly spaced values up to 1,
    # computes the reference, resets the default graph (side effect), then builds the native op
    # (skipped when dilations are given) and the public op, and asserts both are close to the
    # reference. If the grouped kernel is not registered it logs a warning and returns early;
    # other invalid-argument errors are re-raised. Returns nothing.
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [filter_rows, filter_cols,
        input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      data_type: The data type to use.
      use_gpu: Whether to use GPU.
      grouped_conv: Whether to use cuDNN 7's grouped convolution.
      data_format: The data_format of the input. "NHWC" or "NCHW".
      dilations: A list of 2 elements, representing the dilations.
      tolerance: The absolute and relative tolarance when verifying the output.
    """
    input_size = 1
    filter_size = 1
    for s in tensor_in_sizes:
      input_size *= s
    for s in filter_in_sizes:
      filter_size *= s
    # Initializes the input and filter tensor with numbers incrementing to 1.0.
    x1 = [f * 1.0 / input_size for f in range(1, input_size + 1)]
    x1 = np.array(x1).reshape(tensor_in_sizes)
    x2 = [f * 1.0 / filter_size for f in range(1, filter_size + 1)]
    x2 = np.array(x2).reshape(filter_in_sizes)
    # Compute reference result
    strides = [1, stride, stride, 1]
    if isinstance(padding, list):
      padding = [(0, 0)] + padding + [(0, 0)]
    np_result = _DepthwiseConv2dNumpy(x1, x2, strides, padding, "NHWC",
                                      dilations)

    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with self.session(graph=graph, use_gpu=use_gpu) as sess:
      tolerance = tolerance or {
          dtypes.float16: 4e-2,
          dtypes.float32: 1e-5,
          dtypes.float64: 1e-12,
          dtypes.bfloat16: 1e-2,
      }[data_type]

      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=data_type)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=data_type)

      if data_format == "NCHW":
        # Transpose from NHWC input to NCHW
        # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
        t1 = array_ops.transpose(t1, [0, 3, 1, 2])
        strides = [1, 1, stride, stride]
        if isinstance(padding, list):
          padding = [padding[0], padding[3], padding[1], padding[2]]

      # depthwise_conv2d_native does not support dilations except on TPUs.
      if dilations is None:
        with sess.graph._kernel_label_map(  # pylint: disable=protected-access
            {"DepthwiseConv2dNative": "cudnn_grouped_convolution"}
            if grouped_conv else {}):
          conv_native = nn_ops.depthwise_conv2d_native(
              t1, t2, strides=strides, data_format=data_format, padding=padding)

        if data_format == "NCHW":
          # Transpose back from NCHW to NHWC
          conv_native = array_ops.transpose(conv_native, [0, 2, 3, 1])

        try:
          # The Numpy array from calling depthwise_conv2d_native
          native_result = self.evaluate(conv_native)
        except errors.InvalidArgumentError as e:
          # Grouped convolution kernel is only registered for cuDNN 7. Silently
          # return when we are running on an earlier version or without GPU.
          if ("No OpKernel was registered to support Op "
              "'DepthwiseConv2dNative'") in e.message:
            tf_logging.warn("Skipping grouped convolution test")
            return
          raise e

      conv_interface = nn_impl.depthwise_conv2d(
          t1,
          t2,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)
      if data_format == "NCHW":
        # Transpose back from NCHW to NHWC
        conv_interface = array_ops.transpose(conv_interface, [0, 2, 3, 1])

      # The Numpy array from calling depthwise_conv2d
      interface_result = self.evaluate(conv_interface)

    if dilations is None:
      self.assertAllClose(
          native_result, np_result, atol=tolerance, rtol=tolerance)
    self.assertAllClose(
        interface_result, np_result, atol=tolerance, rtol=tolerance)

  @test_util.run_v1_only("b/120545219")
  @test_util.run_gpu_only
  def testDepthwiseConv2DCudnn(self):
    # [FUNCTION DOC]
    # GPU-only, graph-mode-only test. For every forward configuration it verifies values with half
    # and bfloat16 data in NCHW layout on GPU (the configuration where cuDNN depthwise convolution
    # is used), logging each configuration. Asserts agreement with the reference.
    for index, (input_size, filter_size, _, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      # The CuDNN depthwise conv is turned on only when input/output is NCHW and
      # float16(half). See cudnn release note 7.6.3.
      tf_logging.info(
          "Testing DepthwiseConv2DCudnn, %dth config: %r * %r, stride: %d, "
          "padding: %s", index, input_size, filter_size, stride, padding)
      data_types = [dtypes.float16, dtypes.bfloat16]
      for data_type in data_types:
        self._VerifyValues(
            input_size,
            filter_size,
            stride,
            padding,
            data_type,
            use_gpu=True,
            data_format="NCHW",
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2D(self):
    # [FUNCTION DOC]
    # Graph-mode-only test of the forward op over all standard configurations. Uses single
    # precision (plus double unless built for ROCm) with NHWC layout, running each configuration
    # once normally and once with the cuDNN grouped convolution kernel label, with tight
    # per-dtype tolerances. Asserts agreement with the NumPy reference.
    for index, (input_size, filter_size, _, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2D, %dth config: %r * %r, stride: %d, padding: "
          "%s", index, input_size, filter_size, stride, padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float32] + optional_float64):
        tf_logging.info("Testing without grouped_conv")
        tolerance = 1e-4 if data_type == dtypes.float32 else 1e-12
        self._VerifyValues(
            input_size,
            filter_size,
            stride,
            padding,
            data_type,
            use_gpu=True,
            dilations=dilations,
            tolerance=tolerance)
        tf_logging.info("Testing with grouped_conv")
        self._VerifyValues(
            input_size,
            filter_size,
            stride,
            padding,
            data_type,
            use_gpu=True,
            grouped_conv=True,
            dilations=dilations,
            tolerance=tolerance)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DWithUnknownShape(self):
    # [FUNCTION DOC]
    # Regression test (GitHub issue 22110) for the depthwise convolution with an input placeholder
    # of unknown shape, NCHW layout, valid padding and a dilation rate of two by one on a one by
    # one filter. Silently passes (returns) when no GPU is available. Feeds an all-ones
    # 1x1x1x1 input and asserts the output equals all ones of the same shape.
    # GitHub issue 22110.
    if not test.is_gpu_available():
      return
    with self.session():
      x = array_ops.placeholder(dtypes.float32)
      f = np.ones([1, 1, 1, 1], np.float32)
      v = nn_impl.depthwise_conv2d(
          x, f, [1, 1, 1, 1], "VALID", rate=[2, 1], data_format="NCHW")
      self.assertAllEqual(
          np.ones([1, 1, 1, 1], np.float32),
          v.eval(feed_dict={x: np.ones([1, 1, 1, 1], np.float32)}))

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DFormat(self):
    # [FUNCTION DOC]
    # Graph-mode-only test of the forward op in NCHW layout on GPU. Returns immediately if no GPU
    # is available. Runs all standard configurations with single precision (and double unless
    # built for ROCm) and asserts agreement with the reference.
    if not test.is_gpu_available():
      return

    for index, (input_size, filter_size, _, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DFormat, %dth config: %r * %r, stride: %d, "
          "padding: %s", index, input_size, filter_size, stride, padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float32] + optional_float64):
        tolerance = 1e-4 if data_type == dtypes.float32 else 1e-12
        self._VerifyValues(
            input_size,
            filter_size,
            stride,
            padding,
            data_type,
            use_gpu=True,
            data_format="NCHW",
            dilations=dilations,
            tolerance=tolerance)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DExplicit(self):
    # [FUNCTION DOC]
    # Graph-mode-only test of the forward op with explicit padding. For all explicit-padding
    # configurations, tries half, single, bfloat16 (and double unless ROCm) in NHWC and, if a GPU
    # exists, NCHW, using a looser tolerance for bfloat16. Asserts agreement with the reference.
    for index, (input_size, filter_size, _, stride, padding,
                dilations) in enumerate(ConfigsToTestExplicit()):
      tf_logging.info(
          "Testing DepthwiseConv2D, %dth config: %r * %r, stride: %d, padding: "
          "%s", index, input_size, filter_size, stride, padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      data_types = [dtypes.float16, dtypes.float32, dtypes.bfloat16]
      if not test.is_built_with_rocm():
        data_types.extend([dtypes.float64])
      data_formats = ["NHWC", "NCHW"] if test.is_gpu_available() else ["NHWC"]
      for data_type in data_types:
        for data_format in data_formats:
          tolerance = 2e-2 if data_type == dtypes.bfloat16 else None
          self._VerifyValues(
              input_size,
              filter_size,
              stride,
              padding,
              data_type,
              use_gpu=True,
              data_format=data_format,
              dilations=dilations,
              tolerance=tolerance)

  # This is testing against hand calculated results.

  def _VerifyHandValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                        expected, use_gpu):
    # [FUNCTION DOC]
    # HELPER: checks the native depthwise convolution op against hand-calculated values.
    # Inputs: tensor_in_sizes and filter_in_sizes (lists of four dims), stride (integer), padding
    # name, expected (flat list of expected output numbers) and use_gpu (bool). Fills input and
    # filter with counting numbers from one, evaluates the native op in a cached session, logs the
    # value, asserts the flattened result is within 1e-5 of expected and that the result shape
    # matches the op's static shape. Returns nothing.
    """Verifies the output values of the depthwise convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [filter_rows, filter_cols,
        input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether to use GPU.
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    with self.cached_session(use_gpu=use_gpu) as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t1.set_shape(tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      conv = nn_ops.depthwise_conv2d_native(
          t1, t2, strides=[1, stride, stride, 1], padding=padding)
      value = self.evaluate(conv)
    tf_logging.info("value = %r", value)
    self.assertArrayNear(expected, np.ravel(value), 1e-5)
    self.assertShapeEqual(value, conv)

  def testConv2D2x2Filter(self):
    # [FUNCTION DOC]
    # Hand-computed example: a one by two by three by two input with a two by two filter of depth
    # two and multiplier two, stride one, valid padding. The eight expected outputs are worked out
    # in the comments below. Runs the check once on CPU and once on GPU.
    # The inputs look like this (it's a 3 x 2 matrix, each of depth 2):
    #
    # [ (1.0, 2.0), (3.0,  4.0), ( 5.0,  6.0) ]
    # [ (7.0, 8.0), (9.0, 10.0), (11.0, 12.0) ]
    #  We can view this as two inputs
    #
    #  input depth 0:
    #
    #  [ 1.0,  3.0,  5.0 ]
    #  [ 7.0,  9.0, 11.0 ]
    #
    #  input depth 1:
    #
    #  [ 2.0,  4.0,  6.0 ]
    #  [ 8.0, 10.0, 12.0 ]
    #
    # The filter looks like this (it has two 2 x 2 patches, each generating 2
    # depths):
    #
    #  filter #0:
    #
    #  [ (1.0,  3.0), ( 5.0,  7.0)]
    #  [ (9.0, 11.0), (13.0, 15.0)]
    #
    #  filter #1:
    #
    #  [ ( 2.0,  4.0), ( 6.0,  8.0)]
    #  [ (10.0, 12.0), (14.0, 16.0)]
    #
    # So the outputs are:
    #
    # (position 0, 0: in_depth 0, output_depth 0 -- using filter #0)
    #  1.0 * 1.0 + 7.0 * 9.0 + 3.0 * 5.0 + 9.0 * 13.0 = 196
    # (position 0, 0: in_depth 0, output_depth 1 -- using filter #1)
    #  1.0 * 2.0 + 7.0 * 10.0 + 3.0 * 6.0 + 9.0 * 14.0 = 216
    # (position 0, 0: in_depth 1, output_depth 2 -- using filter #0)
    #  2.0 * 3.0 + 8.0 * 11.0 + 4.0 * 7.0 + 10.0 * 15.0 = 272
    # (position 0, 0: in_depth 1, output_depth 3 -- using filter #1)
    #  2.0 * 4.0 + 8.0 * 12.0 + 4.0 * 8.0 + 10.0 * 16.0 = 296
    #
    # (position 1, 0: in_depth 0, output_depth 0 -- using filter #0)
    #  3.0 * 1.0 + 9.0 * 9.0 + 5.0 * 5.0 + 11.0 * 13.0 = 252
    # (position 1, 0: in_depth 0, output_depth 1 -- using filter #1)
    #  3.0 * 2.0 + 9.0 * 10.0 + 5.0 * 6.0 + 11.0 * 14.0 = 280
    # (position 1, 0: in_depth 1, output_depth 2 -- using filter #0)
    #  4.0 * 3.0 + 10.0 * 11.0 + 6.0 * 7.0 + 12.0 * 15.0 = 344
    # (position 1, 0: in_depth 1, output_depth 3 -- using filter #1)
    #  4.0 * 4.0 + 10.0 * 12.0 + 6.0 * 8.0 + 12.0 * 16.0 = 376
    expected_output = [196, 216, 272, 296, 252, 280, 344, 376]
    self._VerifyHandValues(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[2, 2, 2, 2],
        stride=1,
        padding="VALID",
        expected=expected_output,
        use_gpu=False)

    self._VerifyHandValues(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[2, 2, 2, 2],
        stride=1,
        padding="VALID",
        expected=expected_output,
        use_gpu=True)

  # Gradient checkers. This tests depthwise gradient computations for both
  # BackpropFilter and BackpropInput by comparing gradients computed by the
  # depthwise gradient ops with the gradients computed numerically (details can
  # be found in the compute_gradient_error().
  # Note this check is very expensive so the input should not be too big.
  def _ConstructAndTestGradient(self,
                                input_shape,
                                filter_shape,
                                output_shape,
                                stride,
                                padding,
                                data_type,
                                test_input,
                                use_gpu,
                                grouped_conv=False,
                                data_format="NHWC",
                                dilations=None):
    # [FUNCTION DOC]
    # HELPER: numeric gradient check for the public depthwise convolution.
    # KNOWN ISSUE: the float16 tolerance is 4.0 and bfloat16 is 1.0, so the check is very weak
    # for those types. Also catches grouped-convolution registration failures by matching text of
    # the error message, which is brittle (acknowledged in a TODO).
    # Inputs: input_shape, filter_shape, output_shape (lists of four dims, NHWC), stride (integer),
    # padding (name or explicit list), data_type, test_input (true to check the gradient with
    # respect to the input, false for the filter), use_gpu, grouped_conv (default false; uses the
    # cuDNN grouped kernel label for the forward and backprop ops), data_format (NHWC default or
    # NCHW, transposing input, shapes, strides and padding) and dilations (optional).
    # Resets the default graph, builds the op, asserts the static output shape, computes the
    # maximum error between analytic and numeric gradients and asserts it is below a per-dtype
    # tolerance. If the grouped kernel is not registered it logs a warning and returns; other
    # invalid-argument errors are re-raised. Returns nothing.
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    input_np = np.array(input_data).reshape(input_shape)
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
    filter_np = np.array(filter_data).reshape(filter_shape)
    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with self.session(graph=graph, use_gpu=use_gpu) as sess:
      tolerance = {
          dtypes.float16: 4e-0,
          dtypes.float32: 8e-4,
          dtypes.float64: 1e-12,
          dtypes.bfloat16: 1e-0,
      }[data_type]

      input_tensor = constant_op.constant(
          input_np, shape=input_shape, dtype=data_type, name="input")
      filter_tensor = constant_op.constant(
          filter_np, shape=filter_shape, dtype=data_type, name="filter")

      native_input = input_tensor
      strides = [1, stride, stride, 1]
      if isinstance(padding, list):
        padding = [(0, 0)] + padding + [(0, 0)]
      if data_format == "NCHW":
        # Transpose from NHWC input to NCHW
        # Ex. [4, 5, 5, 48] to [4, 48, 5, 5]
        native_input = array_ops.transpose(input_tensor, [0, 3, 1, 2])
        input_shape = [
            input_shape[0], input_shape[3], input_shape[1], input_shape[2]
        ]
        output_shape = [
            output_shape[0], output_shape[3], output_shape[1], output_shape[2]
        ]
        strides = [1, 1, stride, stride]
        if isinstance(padding, list):
          padding = [padding[0], padding[3], padding[1], padding[2]]

      with sess.graph._kernel_label_map({  # pylint: disable=protected-access,g-long-ternary
          "DepthwiseConv2dNative": "cudnn_grouped_convolution",
          "DepthwiseConv2dNativeBackpropInput": "cudnn_grouped_convolution",
          "DepthwiseConv2dNativeBackpropFilter": "cudnn_grouped_convolution",
      } if grouped_conv else {}):
        depthwise_conv2d = nn_impl.depthwise_conv2d(
            native_input,
            filter_tensor,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            name="depthwise_conv2d")

      self.assertEqual(output_shape, depthwise_conv2d.get_shape())

      try:
        if test_input:
          err = gradient_checker.compute_gradient_error(native_input,
                                                        input_shape,
                                                        depthwise_conv2d,
                                                        output_shape)
        else:
          err = gradient_checker.compute_gradient_error(filter_tensor,
                                                        filter_shape,
                                                        depthwise_conv2d,
                                                        output_shape)
      except errors.InvalidArgumentError as e:
        # TODO(xjun): Tests depend on error messages could be brittle.
        # Grouped convolution kernel is only registered for cuDNN 7. Silently
        # return when we are running on an earlier version or without GPU.
        if grouped_conv and ("No OpKernel was registered to support Op "
                             "'DepthwiseConv2dNative'") in e.message:
          tf_logging.warn("Skipping grouped convolution test")
          return
        raise e

      tf_logging.info(
          "data_type: %r, use_gpu: %r, grouped_conv: %r, error = %f", data_type,
          use_gpu, grouped_conv, err)
      self.assertLess(err, tolerance)

  @test_util.run_v1_only("b/120545219")
  @test_util.run_gpu_only
  def testDepthwiseConv2DInputGradCudnn(self):
    # [FUNCTION DOC]
    # GPU-only, graph-mode-only test of the input gradient using half and bfloat16 in NCHW layout
    # (the combination where cuDNN is used). Skips configurations whose stride is not one. Asserts
    # the numeric gradient error is within tolerance.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      # The CuDNN depthwise conv (input gradient) is turned on only when
      # stride = 1, input/output is NCHW and float16(half). See cudnn release
      # note 7.6.3.
      if stride != 1:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradCudnn, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      data_types = [dtypes.float16, dtypes.bfloat16]
      for data_type in data_types:
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=True,
            use_gpu=True,
            data_format="NCHW",
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DInputGrad(self):
    # [FUNCTION DOC]
    # Graph-mode-only test of the input gradient in NHWC layout over the gradient configurations,
    # with single precision (and double unless ROCm), each run normally and with the cuDNN grouped
    # kernel label. Asserts the numeric gradient error is within tolerance.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DInputGrad, %dth config: %r * %r, stride: %d, "
          "padding: %s", index, input_size, filter_size, stride, padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float32] + optional_float64):
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=True,
            use_gpu=True,
            dilations=dilations)
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=True,
            use_gpu=True,
            grouped_conv=True,
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DInputGradFormat(self):
    # [FUNCTION DOC]
    # Graph-mode-only test of the input gradient in NCHW layout. Returns immediately without a
    # GPU. Uses single precision (and double unless ROCm) over the gradient configurations and
    # asserts the numeric gradient error is within tolerance.
    if not test.is_gpu_available():
      return

    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradFormat, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float32] + optional_float64):
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=True,
            use_gpu=True,
            data_format="NCHW",
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DInputGradExplicit(self):
    # [FUNCTION DOC]
    # Graph-mode-only test of the input gradient with explicit padding over the explicit gradient
    # configurations, for half, single, bfloat16 (and double unless ROCm) in NHWC and, with a GPU,
    # NCHW. Asserts the numeric gradient error is within tolerance.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTestExplicit()):
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradExplicit, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      data_types = [dtypes.float16, dtypes.float32, dtypes.bfloat16]
      if not test.is_built_with_rocm():
        data_types.extend([dtypes.float64])
      data_formats = ["NHWC", "NCHW"] if test.is_gpu_available() else ["NHWC"]
      for data_type in data_types:
        for data_format in data_formats:
          self._ConstructAndTestGradient(
              input_size,
              filter_size,
              output_size,
              stride,
              padding,
              data_type,
              test_input=True,
              use_gpu=True,
              data_format=data_format,
              dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  @test_util.run_gpu_only
  def testDepthwiseConv2DFilterGradCudnn(self):
    # [FUNCTION DOC]
    # GPU-only, graph-mode-only test of the filter gradient using half and bfloat16, each in NCHW
    # and NHWC layout (cuDNN is used for half precision). Asserts the numeric gradient error is
    # within tolerance.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      # The CuDNN depthwise conv (filter gradient) is turned on only when
      # input/output is float16(half). See cudnn release note 7.6.3.
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradCudnn, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      data_types = [dtypes.float16, dtypes.bfloat16]
      for data_type in data_types:
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=False,
            use_gpu=True,
            data_format="NCHW",
            dilations=dilations)
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=False,
            use_gpu=True,
            data_format="NHWC",
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DFilterGrad(self):
    # [FUNCTION DOC]
    # Graph-mode-only test of the filter gradient in NHWC layout over the gradient configurations
    # with half, single (and double unless ROCm) precision. Asserts the numeric gradient error is
    # within tolerance.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGrad, %dth config: %r * %r, stride: "
          "%d, padding: %s", index, input_size, filter_size, stride, padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float16, dtypes.float32] + optional_float64):
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=False,
            use_gpu=True,
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DFilterGradFormat(self):
    # [FUNCTION DOC]
    # Graph-mode-only test of the filter gradient in NCHW layout. Returns immediately without a
    # GPU. Uses single precision (and double unless ROCm) and asserts the numeric gradient error
    # is within tolerance.
    if not test.is_gpu_available():
      return

    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTest()):
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradFormat, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
      for data_type in ([dtypes.float32] + optional_float64):
        self._ConstructAndTestGradient(
            input_size,
            filter_size,
            output_size,
            stride,
            padding,
            data_type,
            test_input=False,
            use_gpu=True,
            data_format="NCHW",
            dilations=dilations)

  @test_util.run_v1_only("b/120545219")
  def testDepthwiseConv2DFilterGradExplicit(self):
    # [FUNCTION DOC]
    # Graph-mode-only test of the filter gradient with explicit padding, for half, single,
    # bfloat16 (and double unless ROCm) in NHWC and, with a GPU, NCHW. Asserts the numeric
    # gradient error is within tolerance.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(CheckGradConfigsToTestExplicit()):
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradExplicit, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      # double datatype is currently not supported for convolution ops
      # on the ROCm platform
      data_types = [dtypes.float16, dtypes.float32, dtypes.bfloat16]
      if not test.is_built_with_rocm():
        data_types.extend([dtypes.float64])
      data_formats = ["NHWC", "NCHW"] if test.is_gpu_available() else ["NHWC"]
      for data_type in data_types:
        for data_format in data_formats:
          self._ConstructAndTestGradient(
              input_size,
              filter_size,
              output_size,
              stride,
              padding,
              data_type,
              test_input=False,
              use_gpu=True,
              data_format=data_format,
              dilations=dilations)

  def _CompareBackpropInput(self, input_sizes, filter_sizes, output_sizes,
                            stride, padding, dtype):
    # [FUNCTION DOC]
    # HELPER: compares CPU and GPU results of the input-backprop op.
    # Inputs: input_sizes, filter_sizes, output_sizes (lists of four dims), stride (integer),
    # padding (name or explicit list) and dtype (name such as float32, bfloat16 or float64).
    # Uses random filter and output-gradient values (unseeded, so values vary between runs),
    # evaluates the op on GPU and CPU and asserts the results are close (loose tolerance 0.1 for
    # bfloat16, 1e-4 otherwise). Returns nothing.
    x1 = np.random.rand(*filter_sizes)
    x2 = np.random.rand(*output_sizes)
    if isinstance(padding, list):
      padding = [(0, 0)] + padding + [(0, 0)]

    def _GetVal(use_gpu, dtype):
      # [FUNCTION DOC]
      # HELPER: runs the input-backprop op on the chosen device. Takes use_gpu (bool) and dtype,
      # evaluates the op with the captured random data in a cached session, asserts the result
      # shape matches the op's static shape, and returns the result array.
      with self.cached_session(use_gpu=use_gpu):
        t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
        t1 = constant_op.constant(x1, shape=filter_sizes, dtype=dtype)
        t2 = constant_op.constant(x2, shape=output_sizes, dtype=dtype)
        backprop = nn_ops.depthwise_conv2d_native_backprop_input(
            t0, t1, t2, strides=[1, stride, stride, 1], padding=padding)
        ret = self.evaluate(backprop)
        self.assertShapeEqual(ret, backprop)
        return ret

    rtol, atol = (1e-1, 1e-1) if dtype == "bfloat16" else (1e-4, 1e-4)
    gpu_value = _GetVal(use_gpu=True, dtype=dtype)
    cpu_value = _GetVal(use_gpu=False, dtype=dtype)
    self.assertAllClose(cpu_value, gpu_value, rtol=rtol, atol=atol)

  @test_util.run_gpu_only
  def testDepthwiseConv2DInputGradCompare(self):
    # [FUNCTION DOC]
    # GPU-only test comparing CPU and GPU input-backprop results over the standard configurations
    # without dilations (dilated ones are skipped), for single and bfloat16 precision and double
    # unless built for ROCm. Asserts closeness of CPU and GPU outputs.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      if dilations:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      self._CompareBackpropInput(input_size, filter_size, output_size, stride,
                                 padding, "float32")
      self._CompareBackpropInput(input_size, filter_size, output_size, stride,
                                 padding, "bfloat16")
      # Convolutions on the ROCm platform don't support double dtype.
      # So, we skip these tests.
      if not test.is_built_with_rocm():
        self._CompareBackpropInput(
            input_size, filter_size, output_size, stride, padding, "float64"
        )

  @test_util.run_gpu_only
  def testDepthwiseConv2DInputGradExplicitCompare(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the log message says "DepthwiseConv2DInputGradCompare" (copied from the
    # non-explicit test), so logs do not distinguish this test.
    # GPU-only test comparing CPU and GPU input-backprop results over the explicit-padding
    # configurations without dilations, for single, bfloat16 and (unless ROCm) double precision.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(ConfigsToTestExplicit()):
      if dilations:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DInputGradCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      self._CompareBackpropInput(input_size, filter_size, output_size, stride,
                                 padding, "float32")
      self._CompareBackpropInput(input_size, filter_size, output_size, stride,
                                 padding, "bfloat16")
      # Convolutions on the ROCm platform don't support double dtype.
      if not test.is_built_with_rocm():
        self._CompareBackpropInput(
            input_size, filter_size, output_size, stride, padding, "float64"
        )

  def _CompareBackpropFilter(self, input_sizes, filter_sizes, output_sizes,
                             stride, padding, dtype):
    # [FUNCTION DOC]
    # HELPER: compares CPU and GPU results of the filter-backprop op.
    # Inputs: input_sizes, filter_sizes, output_sizes (lists of four dims), stride (integer),
    # padding (name or explicit list) and dtype (name). Uses random unseeded input and
    # output-gradient values. Computes the CPU result in NHWC, then GPU results in both NHWC and
    # NCHW (transposing data, strides and explicit padding) and asserts each is close to the CPU
    # result with a dtype-aware tolerance (loose for bfloat16). Returns nothing.
    x0 = np.random.rand(*input_sizes)
    x2 = np.random.rand(*output_sizes)
    padding_nhwc = padding
    padding_nchw = padding
    if isinstance(padding, list):
      padding_nhwc = [(0, 0)] + padding + [(0, 0)]
      padding_nchw = [(0, 0)] + [(0, 0)] + padding

    def _GetVal(use_gpu, dtype, data_format="NHWC"):
      # [FUNCTION DOC]
      # HELPER: runs the filter-backprop op on the chosen device and layout. Takes use_gpu (bool),
      # dtype and data_format (NHWC default or NCHW; NCHW transposes data and uses NCHW strides and
      # padding), asserts the result shape matches the static shape and returns the result array.
      with self.cached_session(use_gpu=use_gpu):
        t0 = constant_op.constant(x0, shape=input_sizes, dtype=dtype)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes, dtype=dtype)
        strides = [1, stride, stride, 1]
        padding = padding_nhwc
        if data_format == "NCHW":
          t0 = array_ops.transpose(t0, [0, 3, 1, 2])
          t2 = array_ops.transpose(t2, [0, 3, 1, 2])
          strides = [1, 1, stride, stride]
          padding = padding_nchw
        backprop = nn_ops.depthwise_conv2d_native_backprop_filter(
            t0,
            t1,
            t2,
            strides=strides,
            padding=padding,
            data_format=data_format)
        ret = self.evaluate(backprop)
        self.assertShapeEqual(ret, backprop)
        return ret

    cpu_value = _GetVal(use_gpu=False, dtype=dtype)
    for data_format in ["NHWC", "NCHW"]:
      gpu_value = _GetVal(use_gpu=True, dtype=dtype, data_format=data_format)
      self.assertAllCloseAccordingToType(
          cpu_value, gpu_value, rtol=1e-4, atol=1e-4, bfloat16_rtol=1e-0)

  @test_util.run_gpu_only
  def testDepthwiseConv2DFilterGradCompare(self):
    # [FUNCTION DOC]
    # GPU-only test comparing CPU and GPU filter-backprop results over the standard configurations
    # without dilations, for single and bfloat16 precision and double unless ROCm.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      if dilations:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      self._CompareBackpropFilter(input_size, filter_size, output_size, stride,
                                  padding, "float32")
      self._CompareBackpropFilter(input_size, filter_size, output_size, stride,
                                  padding, "bfloat16")
      # Convolutions on the ROCm platform don't support double dtype.
      if not test.is_built_with_rocm():
        self._CompareBackpropFilter(
            input_size, filter_size, output_size, stride, padding, "float64"
        )

  @test_util.run_gpu_only
  def testDepthwiseConv2DFilterGradExplicitCompare(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the log message says "DepthwiseConv2DFilterGradCompare" (copied from the
    # non-explicit test), so logs do not distinguish this test.
    # GPU-only test comparing CPU and GPU filter-backprop results over the explicit-padding
    # configurations without dilations, for single, bfloat16 and (unless ROCm) double precision.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(ConfigsToTestExplicit()):
      if dilations:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DFilterGradCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      self._CompareBackpropFilter(input_size, filter_size, output_size, stride,
                                  padding, "float32")
      self._CompareBackpropFilter(input_size, filter_size, output_size, stride,
                                  padding, "bfloat16")
      # Convolutions on the ROCm platform don't support double dtype.
      if not test.is_built_with_rocm():
        self._CompareBackpropFilter(
            input_size, filter_size, output_size, stride, padding, "float64"
        )

  def _CompareForward(self, input_sizes, filter_sizes, output_sizes, stride,
                      padding, dtype):
    # [FUNCTION DOC]
    # HELPER: compares CPU and GPU results of the forward native depthwise op.
    # KNOWN ISSUE: the output_sizes parameter is accepted but never used.
    # Inputs: input_sizes, filter_sizes, output_sizes (lists of four dims), stride (integer),
    # padding (name or explicit list) and dtype (name). Uses random unseeded input and filter,
    # evaluates on GPU and CPU and asserts closeness with a dtype-aware tolerance. Returns nothing.
    x1 = np.random.rand(*input_sizes)
    x2 = np.random.rand(*filter_sizes)
    if isinstance(padding, list):
      padding = [(0, 0)] + padding + [(0, 0)]

    def _GetVal(use_gpu, dtype):
      # [FUNCTION DOC]
      # HELPER: runs the forward native depthwise op on the chosen device. Takes use_gpu (bool) and
      # dtype, evaluates in a cached session, asserts the result shape matches the static shape
      # and returns the result array.
      with self.cached_session(use_gpu=use_gpu):
        t1 = constant_op.constant(x1, shape=input_sizes, dtype=dtype)
        t2 = constant_op.constant(x2, shape=filter_sizes, dtype=dtype)
        output = nn_ops.depthwise_conv2d_native(
            t1, t2, strides=[1, stride, stride, 1], padding=padding)
        ret = self.evaluate(output)
        self.assertShapeEqual(ret, output)
        return ret

    gpu_value = _GetVal(use_gpu=True, dtype=dtype)
    cpu_value = _GetVal(use_gpu=False, dtype=dtype)
    self.assertAllCloseAccordingToType(
        cpu_value, gpu_value, rtol=1e-4, atol=1e-4, bfloat16_rtol=1e-1)

  @test_util.run_gpu_only
  def testDepthwiseConv2DForwardCompare(self):
    # [FUNCTION DOC]
    # GPU-only test comparing CPU and GPU forward results over the standard configurations
    # without dilations, for single and bfloat16 precision and double unless ROCm.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(ConfigsToTest()):
      if dilations:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DForwardCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)
      self._CompareForward(input_size, filter_size, output_size, stride,
                           padding, "float32")
      self._CompareForward(input_size, filter_size, output_size, stride,
                           padding, "bfloat16")

      # Convolutions on the ROCm platform don't support double dtype.
      if not test.is_built_with_rocm():
        self._CompareForward(
            input_size, filter_size, output_size, stride, padding, "float64"
        )

  @test_util.run_gpu_only
  def testDepthwiseConv2DForwardExplicitCompare(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the log message says "DepthwiseConv2DForwardCompare" (copied from the
    # non-explicit test), so logs do not distinguish this test.
    # GPU-only test comparing CPU and GPU forward results over the explicit-padding configurations
    # without dilations, for single, bfloat16 and (unless ROCm) double precision.
    for index, (input_size, filter_size, output_size, stride, padding,
                dilations) in enumerate(ConfigsToTestExplicit()):
      if dilations:
        continue
      tf_logging.info(
          "Testing DepthwiseConv2DForwardCompare, %dth config: %r * %r, "
          "stride: %d, padding: %s", index, input_size, filter_size, stride,
          padding)

      self._CompareForward(input_size, filter_size, output_size, stride,
                           padding, "float32")
      self._CompareForward(input_size, filter_size, output_size, stride,
                           padding, "bfloat16")

      # Convolutions on the ROCm platform don't support double dtype.
      if not test.is_built_with_rocm():
        self._CompareForward(
            input_size, filter_size, output_size, stride, padding, "float64"
        )
