# [MODULE DOC]
# Integration tests for TF-TensorRT conversion of the top-k operation. Two test classes on the
# shared integration test base: plain top-k with k equal to 5, and top-k whose index output goes
# through a reshape to check the engine output type is set to int32. Running as a script starts
# the test runner. No import-time side effects.
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

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class TopKTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test checking that a top-k op with a constant k converts into a single engine.
  """Testing Top-K in TF-TRT conversion."""

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from input x (float, 100x100): top-5 along the last axis with k as an int32
    # constant. Returns two identity tensors, output_0 (values) and output_1 (indices).
    k = 5
    k_tensor = constant_op.constant(k, dtype=dtypes.int32, name="Const")
    values, indices = nn_ops.top_k(x, k_tensor, name="TopK")
    values = array_ops.identity(values, name="output_0")
    indices = array_ops.identity(indices, name="output_1")
    return values, indices

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns base-class test parameters: float32, input 100x100, two outputs each 100x5.
    k = 5
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 100]],
                            [[100, k], [100, k]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns a mapping with one engine (000) expected to hold the constant and top-k nodes.
    # run_params is unused.
    """Return the expected engines to build."""
    return {"TRTEngineOp_000": ["Const", "TopK"]}


class TopKOutputTypeTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test checking that the engine output type is set correctly (int32) for top-k
  # indices that pass through an intermediate reshape before leaving the engine.
  """Testing that output type of engine using Top-K is set correctly."""

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from input x (float, 100x100): top-5 with constant k, reshapes the indices
    # to 100x1x5, and returns two identity tensors, output_0 (values) and output_1 (indices).
    k = 5
    k_tensor = constant_op.constant(k, dtype=dtypes.int32, name="Const")
    values, indices = nn_ops.top_k(x, k_tensor, name="TopK")
    # Reshape will act as a layer between the TopK output and the engine
    # output, requiring the output tensor of reshape to be set explicitly to
    # int32.
    indices = array_ops.reshape(indices, [100, 1, 5], name="Reshape")
    values = array_ops.identity(values, name="output_0")
    indices = array_ops.identity(indices, name="output_1")
    return values, indices

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns base-class test parameters: float32, input 100x100, outputs 100x5 and 100x1x5.
    k = 5
    return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 100]],
                            [[100, k], [100, 1, k]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns a mapping with one engine (000) expected to hold the constant, top-k, reshape and
    # reshape-shape nodes. run_params is unused.
    """Return the expected engines to build."""
    return {"TRTEngineOp_000": ["Const", "TopK", "Reshape", "Reshape/shape"]}


if __name__ == "__main__":
  test.main()
