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
# TF-TensorRT integration test of a single VGG-style layer (batch norm, convolution, bias, relu,
# max pool) in the default channels-last (NHWC) layout. One test class on the shared TF-TRT base.
"""Model script to test TF-TensorRT integration."""

import os

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class VGGBlockTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test case for one VGG layer in channels-last layout; expects one engine and enables native
  # segment execution through an environment variable during setup.
  """Single vgg layer test in TF-TRT conversion."""

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from one input tensor: inference-mode fused batch norm (two channels),
    # 1x1x2x6 random convolution with stride 2, random bias add, relu, identity, 2x2 max pool, then
    # a squeeze named output_0. Returns the final tensor. Uses random numbers for weights and bias.
    dtype = x.dtype
    x, _, _ = nn_impl.fused_batch_norm(
        x, [1.0, 1.0], [0.0, 0.0],
        mean=[0.5, 0.5],
        variance=[1.0, 1.0],
        is_training=False)
    e = constant_op.constant(
        np.random.randn(1, 1, 2, 6), name="weights", dtype=dtype)
    conv = nn.conv2d(
        input=x, filter=e, strides=[1, 2, 2, 1], padding="SAME", name="conv")
    b = constant_op.constant(np.random.randn(6), name="bias", dtype=dtype)
    t = nn.bias_add(conv, b, name="biasAdd")
    relu = nn.relu(t, "relu")
    idty = array_ops.identity(relu, "ID")
    v = nn_ops.max_pool(
        idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
    return array_ops.squeeze(v, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns parameters: float32 input of shape 5x8x8x2 and expected output of shape 5x2x2x6.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[5, 8, 8, 2]],
                            [[5, 2, 2, 6]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns the expected engine names: a single engine, TRTEngineOp_000.
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]

  # TODO(b/159459919): remove this routine to disallow native segment execution.
  def setUp(self):
    # [FUNCTION DOC]
    # Per-test setup. Runs the parent setup, then sets the process environment variable that allows
    # engine native segment execution to true (a temporary workaround; the value is never reset).
    super().setUp()
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"


if __name__ == "__main__":
  test.main()
