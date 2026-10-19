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
# Integration test for TF-TensorRT conversion of convolutions with constant filters that need
# broadcasting. Built on TfTrtIntegrationTestBase; runs via the standard test main.
"""Script to test TF-TensorRT integration."""

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class ConstBroadcastTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test that three stacked convolutions with constant-valued filters, each followed by relu, are
  # converted into a single engine.
  """Test for Constant broadcasting in TF-TRT."""

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph: three same-padded stride-one convolutions with constant filters of value
    # 0.3 (shapes 3x3x2x1, 3x3x1x1, 3x3x1x1), each followed by relu; the last relu is named
    # output_0. Returns the output tensor.
    """Return the expected graph to convert."""
    dtype = x.dtype
    filt1 = constant_op.constant(
        0.3, shape=(3, 3, 2, 1), dtype=dtype, name='filt1')
    y1 = nn.conv2d(x, filt1, strides=[1, 1, 1, 1], padding='SAME', name='y1')
    z1 = nn.relu(y1, name='z1')
    filt2 = constant_op.constant(
        0.3, shape=(3, 3, 1, 1), dtype=dtype, name='filt2')
    y2 = nn.conv2d(z1, filt2, strides=[1, 1, 1, 1], padding='SAME', name='y2')
    z2 = nn.relu(y2, name='z')
    filt3 = constant_op.constant(
        0.3, shape=(3, 3, 1, 1), dtype=dtype, name='filt3')
    y3 = nn.conv2d(z2, filt3, strides=[1, 1, 1, 1], padding='SAME', name='y3')
    return nn.relu(y3, name='output_0')

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, input of shape 5x12x12x2, output of shape 5x12x12x1.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[5, 12, 12, 2]],
                            [[5, 12, 12, 1]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects a single engine. Returns a list with its name.
    """Return the expected engines to build."""
    return ['TRTEngineOp_000']

  def ExpectedAbsoluteTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the absolute tolerance for comparing results: 1e-4 for FP32, 1e-2 otherwise.
    """The absolute tolerance to compare floating point results."""
    return 1.e-04 if run_params.precision_mode == 'FP32' else 1.e-02

  def ExpectedRelativeTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the relative tolerance for comparing results: 1e-4 for FP32, 1e-2 otherwise.
    """The relative tolerance to compare floating point results."""
    return 1.e-04 if run_params.precision_mode == 'FP32' else 1.e-02


if __name__ == '__main__':
  test.main()
