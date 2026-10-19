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
# Integration test for TF-TensorRT conversion of a small two-convolution network run with many
# different dynamic input shapes (varying batch size and spatial dimensions). The module
# docstring mentions INT8 and Mnist, but the test excludes quantization modes and the network is
# a generic convolution/bias/relu stack. Built on TfTrtIntegrationTestBase.
"""Script to test TF-TRT INT8 conversion without calibration on Mnist model."""

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class DynamicInputShapesTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test that one dynamic engine handles inputs with fully unknown batch and spatial dimensions,
  # running a sequence of eight different input shapes and expecting identical output shapes.

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph: convolution with all-ones 3x3x1x8 weights, bias add with a random bias,
    # relu, convolution with all-ones 3x3x8x1 weights, bias add with a random bias, and an identity
    # named output. Returns the output tensor.
    conv_filter1 = constant_op.constant(
        np.ones([3, 3, 1, 8]), name="weights1", dtype=dtypes.float32)
    bias1 = constant_op.constant(np.random.randn(8), dtype=dtypes.float32)
    x = nn.conv2d(
        input=x,
        filter=conv_filter1,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name="conv")
    x = nn.bias_add(x, bias1)
    x = nn.relu(x)
    conv_filter2 = constant_op.constant(
        np.ones([3, 3, 8, 1]), name="weights2", dtype=dtypes.float32)
    bias2 = constant_op.constant(np.random.randn(1), dtype=dtypes.float32)
    x = nn.conv2d(
        input=x,
        filter=conv_filter2,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name="conv")
    x = nn.bias_add(x, bias2)
    return array_ops.identity(x, name="output")

  def GetParams(self):
    # [FUNCTION DOC]
    # Builds the test parameters directly: input and output tensor specs with unknown batch, height
    # and width and one channel, a list of eight input shapes (including batch size changes and
    # spatial size changes) and expected output shapes identical to the input shapes.
    # TODO(laigd): we should test the following cases:
    # - batch size is not changed, other dims are changing
    # - batch size is decreasing, other dims are identical
    # - batch size is decreasing, other dims are changing
    # - batch size is increasing, other dims are identical
    # - batch size is increasing, other dims are changing
    input_dims = [[[1, 5, 5, 1]], [[10, 5, 5, 1]], [[3, 5, 5, 1]],
                  [[1, 5, 5, 1]], [[1, 3, 1, 1]], [[2, 9, 9, 1]],
                  [[1, 224, 224, 1]], [[1, 128, 224, 1]]]
    expected_output_dims = input_dims

    return trt_test.TfTrtIntegrationTestParams(
        graph_fn=self.GraphFn,
        input_specs=[
            tensor_spec.TensorSpec([None, None, None, 1], dtypes.float32,
                                   "input")
        ],
        output_specs=[
            tensor_spec.TensorSpec([None, None, None, 1], dtypes.float32,
                                   "output")
        ],
        input_dims=input_dims,
        expected_output_dims=expected_output_dims)

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, then disables non-TRT graph optimizers (the layout optimizer would
    # convert bias add from NHWC to NCHW for four-dimensional input).
    super().setUp()
    # Disable layout optimizer, since it will convert BiasAdd with NHWC
    # format to NCHW format under four dimensional input.
    self.DisableNonTrtOptimizers()

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects a single engine. Returns a list with its name.
    return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Runs only for dynamic engines and non-quantization precision modes. Returns a pair of
    # boolean and reason string.
    return (run_params.dynamic_engine and not trt_test.IsQuantizationMode(
        run_params.precision_mode)), "test dynamic engine and non-INT8"

  def ExpectedAbsoluteTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the absolute tolerance: 1e-3 for FP32, 0.1 otherwise.
    """The absolute tolerance to compare floating point results."""
    return 1.e-03 if run_params.precision_mode == "FP32" else 1.e-01

  def ExpectedRelativeTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the relative tolerance: 1e-3 for FP32, 0.1 otherwise.
    """The relative tolerance to compare floating point results."""
    return 1.e-03 if run_params.precision_mode == "FP32" else 1.e-01


if __name__ == "__main__":
  test.main()
