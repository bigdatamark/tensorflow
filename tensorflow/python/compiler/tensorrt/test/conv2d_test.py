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
# [MODULE DOC]
# Integration tests for TF-TensorRT conversion of 2D convolutions: NCHW and NHWC layouts with
# several kernel sizes and dilations, strided convolutions, and transposed convolution. Also
# defines helper functions that build random-weight convolution graphs. Built on
# TfTrtIntegrationTestBase; runs via the standard test main.
"""Model script to test TF-TensorRT integration."""

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


def conv2d_layer(inputs,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 data_format="channels_last",
                 dilation_rate=(1, 1)):
  # [FUNCTION DOC]
  # HELPER: builds convolution layers for the conv2d graphs.
  # Applies a 2D convolution to the inputs tensor with a random normally distributed constant
  # weight tensor of shape kernel height, kernel width, input channels, filters. Parameters:
  # filters (number of output channels), kernel_size (pair), strides (pair, default one each),
  # padding (case-insensitive name, default valid), data_format (channels last or channels first,
  # default channels last), dilation_rate (pair, default one each). Returns the output tensor.
  # Uses unseeded random numbers unless the caller seeds them.
  dtype = inputs.dtype
  c_axis = -1 if data_format == "channels_last" else 1
  nchan = inputs.shape[c_axis]
  weights_shape = (kernel_size[0], kernel_size[1], nchan, filters)
  weights = constant_op.constant(np.random.randn(*weights_shape), dtype=dtype)
  padding = padding.upper()
  if data_format == "channels_last":
    strides = [1] + list(strides) + [1]
    dilations = [1] + list(dilation_rate) + [1]
    data_format = "NHWC"
  else:
    strides = [1, 1] + list(strides)
    dilations = [1, 1] + list(dilation_rate)
    data_format = "NCHW"
  return gen_nn_ops.conv2d(
      inputs,
      weights,
      strides=strides,
      padding=padding,
      dilations=dilations,
      data_format=data_format)


def div_round_up(n, d):
  # [FUNCTION DOC]
  # HELPER: integer arithmetic helper (currently unused in this file).
  # Returns the ceiling of n divided by d for positive integers, computed with integer division.
  return (n - 1) // d + 1


def build_graph(inp,
                dtype,
                num_filters,
                data_format,
                kernel_sizes,
                dilation_rates,
                padding="same"):
  # [FUNCTION DOC]
  # HELPER: builds the multi-convolution graph shared by the NCHW and NHWC tests.
  # KNOWN ISSUE: the dtype parameter is accepted but never used.
  # Applies one convolution for every combination of kernel size and dilation rate to the input
  # (stride one, given padding, default same, given data format and number of filters), sums all
  # results elementwise, and returns an identity of the sum named output_0.
  results = []
  for kernel_size in kernel_sizes:
    for dilation_rate in dilation_rates:
      result = conv2d_layer(inp, num_filters, kernel_size, (1, 1), padding,
                            data_format, dilation_rate)
      results.append(result)
  output = sum(results)
  return array_ops.identity(output, name="output_0")


class Conv2DNCHWTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test that four channels-first convolutions (two kernel sizes by two dilation rates) summed
  # together are converted into a single engine.
  """Testing conversion of Conv2D (data_format=NCHW) in TF-TRT conversion."""

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Seeds the random generator (fixed seed) for reproducible weights, then builds the
    # multi-convolution graph with 5 filters, channels-first layout, kernel sizes 3x3 and 3x2, and
    # dilation rates 1x1 and 2x3. Returns the output tensor.
    np.random.seed(1234)
    return build_graph(
        inp=inp,
        dtype=dtypes.float32,
        num_filters=5,
        data_format="channels_first",
        kernel_sizes=[(3, 3), (3, 2)],
        dilation_rates=[(1, 1), (2, 3)])

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, input of shape 13x3x7x11, output of shape 13x5x7x11.
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 3, 7, 11]],
                            [[13, 5, 7, 11]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects a single engine. Returns a list with its name.
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ExpectedAbsoluteTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns 4e-2 when quantization with calibration is used; otherwise the parent default.
    """The absolute tolerance to compare floating point results."""
    if trt_test.IsQuantizationWithCalibration(run_params):
      return 4e-02
    return super(Conv2DNCHWTest, self).ExpectedAbsoluteTolerance(run_params)

  def ExpectedRelativeTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns 4e-2 when quantization with calibration is used; otherwise the parent default.
    """The relative tolerance to compare floating point results."""
    if trt_test.IsQuantizationWithCalibration(run_params):
      return 4e-02
    return super(Conv2DNCHWTest, self).ExpectedRelativeTolerance(run_params)


class Conv2DNHWCTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says NCHW, but this test uses the channels-last (NHWC) layout.
  # Test that four channels-last convolutions (two kernel sizes by two dilation rates) summed
  # together are converted into a single engine.
  """Testing conversion of Conv2D (data_format=NCHW) in TF-TRT conversion."""

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Seeds the random generator (fixed seed), then builds the multi-convolution graph with 5
    # filters, channels-last layout, kernel sizes 3x3 and 3x2, and dilation rates 1x1 and 2x3.
    # Returns the output tensor.
    np.random.seed(1234)
    return build_graph(
        inp=inp,
        dtype=dtypes.float32,
        num_filters=5,
        data_format="channels_last",
        kernel_sizes=[(3, 3), (3, 2)],
        dilation_rates=[(1, 1), (2, 3)])

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, input of shape 13x7x11x3, output of shape 13x7x11x5.
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 7, 11, 3]],
                            [[13, 7, 11, 5]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects a single engine. Returns a list with its name.
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]


class Conv2DStridedNCHWTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test that two stacked strided channels-first convolutions (the second dilated) are converted
  # into a single engine, with looser tolerances for INT8.
  """Testing conversion of strided Conv2D (data_format=NCHW)."""

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Seeds the random generator (fixed seed), then applies two same-padded channels-first
    # convolutions with 5 filters and stride 2: kernel 3x2 and then kernel 3x3 with dilation 2x3.
    # Returns an identity named output_0.
    np.random.seed(1234)
    num_filters = 5
    output = inp
    output = conv2d_layer(
        output,
        num_filters, (3, 2),
        strides=(2, 2),
        padding="same",
        data_format="channels_first")
    output = conv2d_layer(
        output,
        num_filters, (3, 3),
        strides=(2, 2),
        dilation_rate=(2, 3),
        padding="same",
        data_format="channels_first")
    return array_ops.identity(output, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, input of shape 13x3x7x11, output of shape 13x5x2x3.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 3, 7, 11]],
                            [[13, 5, 2, 3]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects a single engine. Returns a list with its name.
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ExpectedAbsoluteTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the absolute tolerance: 0.5 for INT8, 1e-2 otherwise.
    """The absolute tolerance to compare floating point results."""
    return 5.e-01 if run_params.precision_mode == "INT8" else 1.e-02

  def ExpectedRelativeTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the relative tolerance: 5.0 (a very loose value) for INT8, 1e-2 otherwise.
    """The relative tolerance to compare floating point results."""
    return 5.e-00 if run_params.precision_mode == "INT8" else 1.e-02


class Conv2DTranposeTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test that a channels-first transposed convolution (backprop-input form) with stride 2 is
  # converted into a single engine.
  """Testing conversion of conv2d_transpose (AKA Conv2DBackpropInput)"""

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Seeds the random generator (fixed seed), builds random weights of shape 2x2x8x3 and an
    # output-shape constant (13, 8, 14, 22), applies a same-padded NCHW transposed convolution
    # with stride 2 on the spatial axes, and returns an identity named output_0.
    np.random.seed(1234)
    dtype = inp.dtype
    n, c, h, w = 13, 3, 7, 11
    num_filters = 8
    weights_shape = [2, 2, num_filters, c]
    weights = constant_op.constant(np.random.randn(*weights_shape), dtype=dtype)
    output_shape = constant_op.constant([n, num_filters, h * 2, w * 2],
                                        dtype=dtypes.int32)
    output = nn_ops.conv2d_transpose(
        inp,
        weights,
        output_shape,
        strides=[1, 1, 2, 2],
        padding="SAME",
        data_format="NCHW")
    return array_ops.identity(output, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, input of shape 13x3x7x11, output of shape 13x8x14x22.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[13, 3, 7, 11]],
                            [[13, 8, 14, 22]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects a single engine. Returns a list with its name.
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]


if __name__ == "__main__":
  test.main()
