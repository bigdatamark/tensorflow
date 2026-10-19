# [MODULE DOC]
# Integration test for TF-TensorRT conversion where a convolution feeds both a TensorRT-compatible
# path and a TensorRT-incompatible op whose result rejoins the compatible path. Defines one test
# class on the shared integration test base and checks which nodes end up in which engine.
# Running the file as a script starts the test runner. No import-time side effects.
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
"""Model script to test TF-TensorRT integration."""

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class NeighboringEngineTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test (derived from the TF-TRT integration test base) for wiring of neighboring
  # nodes around a TensorRT-incompatible op. Supplies the graph, shapes and expected engines.
  """Neighboring node wiring tests in TF-TRT conversion."""

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from input tensor x (float tensor of shape 2x3x7x5): a random-weight NCHW
    # convolution; its output is multiplied by a random bias (node "mul"), and also passed through
    # an incompatible op ("incompatible"); the two are subtracted ("sub"). Returns the squeezed
    # result named output_0. Uses random numbers for weights and bias.
    dtype = x.dtype
    e = constant_op.constant(
        np.random.normal(.3, 0.05, [3, 2, 3, 4]), name="weights", dtype=dtype)
    conv = nn.conv2d(
        input=x,
        filter=e,
        data_format="NCHW",
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    b = constant_op.constant(
        np.random.normal(1.0, 1.0, [1, 4, 1, 1]), name="bias", dtype=dtype)
    t = math_ops.mul(conv, b, name="mul")
    e = self.trt_incompatible_op(conv, name="incompatible")
    t = math_ops.sub(t, e, name="sub")
    return array_ops.squeeze(t, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns the test parameters built by the base class: the graph function, float32 dtype, one
    # input of shape 2x3x7x5 and one expected output of shape 2x4x5x4. Takes no arguments.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[2, 3, 7, 5]],
                            [[2, 4, 5, 4]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns a mapping from engine name to the node names expected inside it: engine 000 holds the
    # bias, mul and sub nodes; engine 001 holds the weights and conv nodes. run_params is unused.
    """Return the expected engines to build."""
    return {
        "TRTEngineOp_000": ["bias", "mul", "sub"],
        "TRTEngineOp_001": ["weights", "conv"]
    }


if __name__ == "__main__":
  test.main()
