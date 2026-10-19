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
# Integration test for TF-TensorRT conversion of two chained 1x1 convolutions with odd-sized
# weights, checking that weight memory alignment is handled. TF32 is disabled for the whole test
# class. Built on TfTrtIntegrationTestBase; runs via the standard test main.
"""Model script to test TF-TensorRT integration."""

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


@test_util.run_all_without_tensor_float_32("Uses matmul")
class MemoryAlignmentTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says BatchMatMul, but the test uses two convolutions.
  # Test that two valid-padded 1x1 convolutions with random weights of shapes 1x1x3x5 and 1x1x5x10
  # convert into a single engine. The decorator disables TF32 for all test methods.
  """Testing conversion of BatchMatMul in TF-TRT conversion."""

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Builds the graph: two stride-one valid-padded convolutions with random constant kernels
    # named kernel_1 and kernel_2, then a squeeze named output_0. Returns the output tensor.
    dtype = inp.dtype
    e1 = constant_op.constant(
        np.random.randn(1, 1, 3, 5), name="kernel_1", dtype=dtype)
    e2 = constant_op.constant(
        np.random.randn(1, 1, 5, 10), name="kernel_2", dtype=dtype)
    conv = nn.conv2d(
        input=inp,
        filter=e1,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    out = nn.conv2d(
        input=conv,
        filter=e2,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv_2")
    return array_ops.squeeze(out, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, input of shape 2x15x15x3, output of shape 2x15x15x10.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[2, 15, 15, 3]],
                            [[2, 15, 15, 10]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects a single engine. Returns a list with its name.
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ExpectedAbsoluteTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the absolute tolerance: 1e-6 for FP32, 1e-2 otherwise.
    """The absolute tolerance to compare floating point results."""
    return 1.e-06 if run_params.precision_mode == "FP32" else 1.e-02

  def ExpectedRelativeTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the relative tolerance: always 0.1.
    """The relative tolerance to compare floating point results."""
    return 0.1


if __name__ == "__main__":
  test.main()
