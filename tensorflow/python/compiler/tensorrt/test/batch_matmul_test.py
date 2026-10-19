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
# Integration tests for TF-TensorRT conversion of batched matrix multiplication: both operands
# as tensors, and one operand a constant weight that must be broadcast (same rank, and lower
# rank). Uses static batch sizes because shape inference of the batched multiply does not work
# otherwise. Built on TfTrtIntegrationTestBase; runs via the standard test main.
"""Model script to test TF-TensorRT integration."""

import unittest

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class BatchMatMultTestBase(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # HELPER: shared base for the batch matmul tests.
  # Overrides parameter building to use fully static shapes and skips itself when run directly.
  """Base class for BatchMatMult tests."""

  # Shape inference of BatchMatMultV2 doesn't work. Use static batch size.
  def BuildParams(self, graph_fn, dtype, input_shapes, output_shapes):
    # [FUNCTION DOC]
    # HELPER: parameter builder for the batch matmul tests.
    # Builds test parameters with a mask marking every dimension of every input and output as
    # static, with no extra inputs or outputs. Takes the graph function, dtype and the lists of
    # input and output shapes; returns the parameter structure.
    return self.BuildParamsWithMask(
        graph_fn=graph_fn,
        dtype=dtype,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        input_mask=[[True] * len(s) for s in input_shapes],
        output_mask=[[True] * len(s) for s in output_shapes],
        extra_inputs=[],
        extra_outputs=[])

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Skips the test class when it is the abstract base itself (raises a skip signal); otherwise
    # delegates to the parent class setup.
    if cls is BatchMatMultTestBase:
      raise unittest.SkipTest(
          "BatchMatMultTestBase defines base class for other test.")
    super(BatchMatMultTestBase, cls).setUpClass()


class BatchMatMulTwoTensorTest(BatchMatMultTestBase):
  # [CLASS DOC]
  # Test conversion of batched matmul of two input tensors followed by relu into one engine.
  """Testing conversion of BatchMatMul where both inputs are tensors."""

  def GraphFn(self, inp, inp1):
    # [FUNCTION DOC]
    # Builds the graph: matmul of the two inputs, relu (to reach the minimum segment size), and an
    # identity named output_0. Returns the output tensor.
    x1 = math_ops.matmul(inp, inp1, name="matmul")
    # Relu to reach minimum segment size.
    x1 = nn.relu(x1, name="relu")
    return array_ops.identity(x1, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, inputs of shape 12x5x8x12 and 12x5x12x7, output 12x5x8x7.
    return self.BuildParams(self.GraphFn, dtypes.float32,
                            [[12, 5, 8, 12], [12, 5, 12, 7]], [[12, 5, 8, 7]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects one engine containing the matmul and relu nodes. Returns a mapping from engine name
    # to node name list.
    """Return the expected engines to build."""
    return {"TRTEngineOp_000": ["matmul", "relu"]}


class BatchMatMulWeightBroadcastTest(BatchMatMultTestBase):
  # [CLASS DOC]
  # Test conversion of batched matmul where the second operand is a random constant weight of the
  # same rank as the input but with batch dimension one (so it must be broadcast).
  """Testing BatchMatMulV2: one operand is weight and both have same rank."""

  def ShouldAllowTF32Computation(self):
    # [FUNCTION DOC]
    # Disallows TF32 computation for this test; always returns false.
    return False

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Builds the graph: a random constant kernel of shape 1x5x7 (same dtype as the input), matmul
    # of the input with it, and an identity named output_0. Returns the output tensor.
    dtype = inp.dtype
    b = constant_op.constant(
        np.random.randn(1, 5, 7), dtype=dtype, name="kernel")
    x1 = math_ops.matmul(inp, b, name="matmul")
    return array_ops.identity(x1, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, input of shape 12x9x5, output 12x9x7.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[12, 9, 5]],
                            [[12, 9, 7]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects one engine containing the matmul and kernel nodes. Returns a mapping from engine name
    # to node name list.
    """Return the expected engines to build."""
    return {"TRTEngineOp_000": ["matmul", "kernel"]}


class BatchMatMulWeightBroadcastDims2Test(BatchMatMultTestBase):
  # [CLASS DOC]
  # Test conversion of batched matmul where the second operand is a rank-2 random constant weight
  # that must be broadcast across the batch dimension of the rank-3 input.
  """Testing BatchMatMulV2: weight operand must be broadcasted."""

  def ShouldAllowTF32Computation(self):
    # [FUNCTION DOC]
    # Disallows TF32 computation for this test; always returns false.
    return False

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Builds the graph: a random constant kernel of shape 5x7, matmul of the input with it, and an
    # identity named output_0. Returns the output tensor.
    dtype = inp.dtype
    b = constant_op.constant(np.random.randn(5, 7), dtype=dtype, name="kernel")
    x1 = math_ops.matmul(inp, b, name="matmul")
    return array_ops.identity(x1, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, input of shape 12x9x5, output 12x9x7.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[12, 9, 5]],
                            [[12, 9, 7]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects one engine containing the matmul and kernel nodes. Returns a mapping from engine name
    # to node name list.
    """Return the expected engines to build."""
    return {"TRTEngineOp_000": ["matmul", "kernel"]}


if __name__ == "__main__":
  test.main()
