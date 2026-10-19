# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# TF-TensorRT integration tests for graphs that squeeze the batch dimension. Defines an abstract
# base test (skipped when run directly) plus two concrete variants: one with static input/output
# shapes and one with dynamic (unknown) shapes. Built on the shared TF-TRT integration test base.
"""Model script to test TF-TensorRT integration."""

from unittest import SkipTest  # pylint: disable=g-importing-member

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class TrtModeTestBase(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # HELPER: abstract base shared by StaticInputTest and DynamicInputTest.
  # Defines the common graph (unary math ops with a squeeze of the batch dimension), run filtering,
  # batch size and expected engine logic. Its class setup raises a skip signal when the class
  # itself is the one being run, so only subclasses execute.
  """Test squeeze on batch dim and some unary operations in TF-TRT."""

  def GraphFn(self, x1):
    # [FUNCTION DOC]
    # Builds the test graph from one input tensor: abs, add 1, multiply by 3, squeeze the first
    # (batch) axis, abs, add 5, and return the result as an identity named output_0.
    q = math_ops.abs(x1)
    q = q + 1.0
    q = q * 3.0
    q = array_ops.squeeze(q, 0)
    q = math_ops.abs(q)
    q = q + 5.0
    return array_ops.identity(q, name="output_0")

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Decides whether a parameter combination runs. Returns a pair of a boolean and a reason
    # string; the boolean is true only for version 2, dynamic-engine, non-calibration runs because
    # squeeze yields dynamic shapes that a static engine could not handle without native fallback.
    # Squeeze op produces dynamic shaped values. Therefore, we don't run the
    # test with static engine to avoid native segment execution.
    return (run_params.dynamic_engine and run_params.is_v2 and
            not run_params.use_calibration, "test v2 dynamic engine and "
            "non-calibration")

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns the test parameters: float32 input of shape 1x12x5 and expected output of shape
    # 12x5 (the leading dimension removed by the squeeze).
    """The input has 1 as a first dimension, which is removed by the squeeze.

    op in the graph.

    In explicit batch mode, TensorRT can convert the whole graph. In this mode
    it is possible to manipulate the batch dimension using the squeeze op.

    In implicit batch mode TensorRT cannot convert the whole graph. We are not
    allowed to manipulate (squeeze) the first dimension in implicit batch mode.
    Therefore the graph will be converted using multiple segments.
    """
    return self.BuildParams(self.GraphFn, dtypes.float32, [[1, 12, 5]],
                            [[12, 5]])

  def GetMaxBatchSize(self, run_params):
    # [FUNCTION DOC]
    # Returns the maximum batch size for the engine: nothing/null for dynamic engines, otherwise 12
    # (the batch size of the ops that follow the squeeze).
    if run_params.dynamic_engine:
      return None

    # The first dimension of the input is squeezed and the batch size for the
    # rest OPs is 12.
    return 12

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level setup. Raises a skip signal when invoked for the abstract base class itself;
    # otherwise delegates to the parent class setup.
    if cls is TrtModeTestBase:
      raise SkipTest("TrtModeTestBase defines base class for other test.")
    super(TrtModeTestBase, cls).setUpClass()

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns the expected engine names: a single engine when running in dynamic shape (explicit
    # batch) mode, otherwise two engines (before and after the squeeze).
    """Check that the expected engine is built.

    Args:
      run_params: the run parameters.

    Returns:
      the expected engines to build.

    The squeeze op is not converted by TensorRT in implicit batch mode.
    Because of this we have two TRTEngineOp in the graphs: one for the
    subgraph before 'squeeze(q,0)', and another one for the rest of the ops
    after the 'squeeze(q,0)'.

    In explicit batch mode the whole graph is converted using a single engine.
    """
    if run_params.dynamic_shape:
      return ["TRTEngineOp_000"]
    else:
      return ["TRTEngineOp_000", "TRTEngineOp_001"]


class StaticInputTest(TrtModeTestBase):
  # [CLASS DOC]
  # Concrete test using masks that keep input and output shapes fully static (known).

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns parameters built with masks marking every dimension as known: float32 input 1x12x5,
    # expected output 12x5, no extra inputs or outputs.
    """We specify input/output masks with static (known) shapes."""
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[1, 12, 5]], [[12, 5]],
        input_mask=[[True, True, True]],
        output_mask=[[True, True]],
        extra_inputs=[],
        extra_outputs=[])


class DynamicInputTest(TrtModeTestBase):
  # [CLASS DOC]
  # Concrete test using masks that make input and output shapes unknown, with extra input shapes.
  """Test with dynamic input shapes.

  The difference to the previous test is that we use input and output masks to
  change the input and output shapes to unknown shapes.
  """

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns parameters built with masks marking every dimension unknown, plus two extra input
    # shapes (1x2x3 and 1x4x6) and their expected outputs (2x3 and 4x6), so one engine must serve
    # several optimization profiles.
    """We specify input/output mask with dynamic (unknown) shapes.

    In dynamic shape mode, single engine with three optimization profiles can
    handle the three different input shapes.
    """
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[1, 12, 5]], [[12, 5]],
        extra_inputs=[[[1, 2, 3]], [[1, 4, 6]]],
        extra_outputs=[[[2, 3]], [[4, 6]]],
        input_mask=[[False, False, False]],
        output_mask=[[False, False]])

if __name__ == "__main__":
  test.main()
