# [MODULE DOC]
# Integration tests for TF-TensorRT handling of shape-valued tensors: engines whose outputs are
# shape tensors, pruned inputs, shape value masks, and input optimization profiles. Defines
# several test classes on the shared integration test base, most using dynamic shape mode only and
# disabling non-TRT graph optimizers. Running as a script starts the test runner. No import-time
# side effects.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ShapeOutputTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test where the first engine outputs the shape of its input as a float tensor, and
  # a second segment follows a TRT-incompatible op. Uses two input shape profiles. Runs only in
  # dynamic shape mode and not in INT8.
  """Test shape value output with TF-TRT."""

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: runs the parent setup, then disables the non-TRT graph optimizers.
    super().setUp()
    self.DisableNonTrtOptimizers()

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from x (float, 2x2x5x3): computes the shape of 2*x+1, casts it to float,
    # passes it through an incompatible op, then computes q*2+q*q and returns an identity named
    # output_0 (a 4-element vector).
    # The first engine returns the shape of q, which equals the shape of x. The
    # values of x are actually not needed for the TRT engine. This way x is
    # neither a shape tensor nor an execution tensor, we still need to set its
    # shape (binding dimensions). We confirm with this test that the binding
    # dimensions of x are correctly set before we execute the engine.
    q = 2 * x + 1
    q = array_ops.shape(q)
    q = math_ops.cast(q, dtypes.float32)
    q = self.trt_incompatible_op(q)
    q = q * 2 + q * q
    return array_ops.identity(q, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns masked test parameters: float32, input 2x2x5x3, output of 4 elements, one extra input
    # profile with a different batch of 8 and an extra output of 4 elements; the input mask marks
    # the batch dimension as not variable-batch and the output mask has one true entry.
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[2, 2, 5, 3]], [[4]],
        extra_inputs=[[[8, 2, 5, 3]]],
        extra_outputs=[[[4]]],
        input_mask=[[False, True, True, True]],
        output_mask=[[True]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns the expected engine names: two engines in dynamic shape mode, only the first
    # otherwise (the second segment has one-dimensional tensors, unsupported in implicit batch).
    """Returns the expected engines to build."""
    if run_params.dynamic_shape:
      return ["TRTEngineOp_000", "TRTEngineOp_001"]
    else:
      # Second segment not converted in implicit batch mode, because its
      # tensors have only one dimensions
      return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Returns a pair: true only in dynamic shape mode and for a precision mode other than INT8
    # (calibration needs a built engine), plus a reason string.
    # We cannot calibrate without building the engine, we turn of INT8 test.
    return (run_params.dynamic_shape and
            run_params.precision_mode != "INT8", "no calibration dynamic shape")


class ShapeOutputWithSingleInputProfile(ShapeOutputTest):
  # [CLASS DOC]
  # Variant of the shape output test using a single input profile (no extra inputs or outputs).
  # Inherits the graph and expectations from the parent.
  """Same as the previous test, but with a single input profile."""

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: runs the parent setup, then disables the non-TRT graph optimizers again.
    super().setUp()
    self.DisableNonTrtOptimizers()

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns masked test parameters like the parent, but with no extra input or output profiles:
    # float32, input 2x2x5x3, output of 4 elements.
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[2, 2, 5, 3]], [[4]],
        extra_inputs=[],
        extra_outputs=[],
        input_mask=[[False, True, True, True]],
        output_mask=[[True]])


class ShapeOutputWithSingleInputAndReshape(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test where the shape of the input is reshaped to a 2x2 matrix before the
  # incompatible op, so the engine output cannot be treated as a shape tensor. Expects two engines.
  """Similar to the previous test, but the ShapeOp output is reshaped to 2D.

  This makes the output tensor not compatible with shape tensor.
  """

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: runs the parent setup, then disables the non-TRT graph optimizers.
    super().setUp()
    self.DisableNonTrtOptimizers()

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from x (float, 2x2x5x3): shape of 2*x+1, reshaped to 2x2, cast to float,
    # through an incompatible op, then q*2+q*q; returns identity named output_0.
    q = 2 * x + 1
    q = array_ops.shape(q)
    q = gen_array_ops.reshape(q, [2, 2])
    q = math_ops.cast(q, dtypes.float32)
    q = self.trt_incompatible_op(q)
    q = q * 2 + q * q
    return array_ops.identity(q, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns masked test parameters: float32, input 2x2x5x3, output 2x2, no extra profiles; the
    # input mask marks the batch dimension and the output mask has two true entries.
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[2, 2, 5, 3]], [[2, 2]],
        extra_inputs=[],
        extra_outputs=[],
        input_mask=[[False, True, True, True]],
        output_mask=[[True, True]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns two expected engine names (000 and 001); run_params is unused.
    """Returns the expected engines to build."""
    return ["TRTEngineOp_000", "TRTEngineOp_001"]


class PrunedInputTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test for an input tensor pruned from the TRT network because only its shape is
  # used and the shape is fixed by the optimization profile, leaving the engine with no input
  # bindings. Runs only for version 2 in dynamic shape mode.
  """In TRT 7, an input tensor can be pruned if it is not used by the network.

  This happens if only its shape is used, but the shape is already defined by
  the optimization profile by setting min=max. (nvbugs/3153064)

  After pruning, the TRT network has no input bindings.
  """

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: runs the parent setup, then disables the non-TRT graph optimizers.
    super().setUp()
    self.DisableNonTrtOptimizers()

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from x (float, 1x2x5x3): takes only its shape, computes q*2+q*q and returns
    # identity named output_0 (4 elements).
    q = array_ops.shape(x)
    q = q * 2 + q * q
    return array_ops.identity(q, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns masked test parameters: float32, input 1x2x5x3, output of 4 elements, no extra
    # profiles.
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[1, 2, 5, 3]], [[4]],
        extra_inputs=[],
        extra_outputs=[],
        input_mask=[[False, True, True, True]],
        output_mask=[[True]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns one expected engine name (000); run_params is unused.
    """Returns the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Returns a pair: true only for dynamic shape mode with the version 2 converter, plus a
    # reason string.
    # Shape op is only converted in dynamic shape mode.
    return (run_params.dynamic_shape and
            run_params.is_v2, "test v2 dynamic shape")


class PrunedInputTest2(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test with two inputs where one (used only for its shape) is pruned; the other is
  # reshaped to that shape. Runs only for version 2 in dynamic shape mode.
  """Two inputs, one of the is pruned."""

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: runs the parent setup, then disables the non-TRT graph optimizers.
    super().setUp()
    self.DisableNonTrtOptimizers()

  def GraphFn(self, x, y):
    # [FUNCTION DOC]
    # Builds the graph from x (float, 1x2x5x3, used only for its shape) and y (float, 2x15):
    # computes y*y+y and reshapes it to the shape of x. Returns a pair of identity tensors:
    # output_0 (the shape vector) and output_1 (the reshaped result).
    q = array_ops.shape(x)
    z = y * y + y
    z = gen_array_ops.reshape(z, q)
    out_0 = array_ops.identity(q, name="output_0")
    out_1 = array_ops.identity(z, name="output_1")
    return (out_0, out_1)

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns masked test parameters: float32, inputs 1x2x5x3 and 2x15, outputs of 4 elements and
    # 1x2x5x3, no extra profiles, with per-input and per-output masks.
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float32, [[1, 2, 5, 3], [2, 15]], [[4], [1, 2, 5, 3]],
        extra_inputs=[],
        extra_outputs=[],
        input_mask=[[False, True, True, True], [False, True]],
        output_mask=[[True], [False, True, True, True]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns one expected engine name (000); run_params is unused.
    """Returns the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Returns a pair: true only for dynamic shape mode with the version 2 converter, plus a
    # reason string.
    # Shape op is only converted in dynamic shape mode.
    return (run_params.dynamic_shape and
            run_params.is_v2, "test v2 dynamic shape")


class ShapeValueMaskTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test confirming that zero- and one-dimensional non-integer tensors are not treated
  # as shape tensors, even when engine construction fails. Sets environment flags that force the
  # engine build to abort and allow native segment execution, restored in tearDown.
  """Confirm that 0D and 1D non int tensors are not treated as shape tensors."""

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: runs the parent setup, then sets two environment variables (abort CUDA engine
    # build, allow native segment execution) to true so the shape value mask is exercised when
    # engine construction fails.
    super().setUp()
    # This is to test whether shape value mask is correctly set in case engine
    # construction has failed.
    os.environ["TF_TRT_ABORT_CUDA_ENGINE_BUILD"] = "True"
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture: runs the parent teardown, then resets the two environment variables set in
    # setUp to false (they are set to false, not removed).
    super().tearDown()
    os.environ["TF_TRT_ABORT_CUDA_ENGINE_BUILD"] = "False"
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "False"

  def GraphFn(self, x, y):
    # [FUNCTION DOC]
    # Builds the graph from x (half-precision vector of 3) and y (half-precision scalar): computes
    # 2*x+y and returns identity named output_0.
    q = 2 * x + y
    return array_ops.identity(q, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns masked test parameters: float16, inputs of shape 3 and scalar, output of shape 3, no
    # extra profiles.
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.float16, [[3], []], [[3]],
        extra_inputs=[],
        extra_outputs=[],
        input_mask=[[True], []],
        output_mask=[[True]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns one expected engine name (000) in dynamic shape mode, otherwise an empty list.
    """Returns the expected engines to build."""
    if run_params.dynamic_shape:
      return ["TRTEngineOp_000"]
    else:
      return []

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Returns a pair: true only in dynamic shape mode for a precision mode other than INT8, plus
    # a reason string.
    # We cannot calibrate without building the engine, we turn of INT8 test.
    return (run_params.dynamic_shape and
            run_params.precision_mode != "INT8", "no calibration dynamic shape")


class InputProfile(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test of shape profile management with an int32 input that is not a shape tensor;
  # extra profile inputs with large values must not be misread as shape values. Runs only for
  # version 2, dynamic shape, non-INT8.
  """The shape profiles has to fit values of shape tensors, but for regular

  tensors the values do not matter. Here we test shape profile management with
  an INT32 input tensor that is not a shape tensor. The extra inputs with
  dim=10 would trigger an error if we mistakenly treat it as shape tensors.
  """

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: runs the parent setup, then disables the non-TRT graph optimizers.
    super().setUp()

    self.DisableNonTrtOptimizers()

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from x (int32 vector of 4): computes x*x+x+1 and returns identity named
    # output_0.
    z = x * x + x + 1
    z = array_ops.identity(z, name="output_0")
    return z

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns masked test parameters: int32, input and output of 4 elements, with two extra
    # profiles of sizes 5 and 10 for both inputs and outputs, and masks marking no shape values.
    return self.BuildParamsWithMask(
        self.GraphFn,
        dtypes.int32,
        [[4]],
        [[4]],
        extra_inputs=[[[5]], [[10]]],
        extra_outputs=[[[5]], [[10]]],
        input_mask=[[False]],
        output_mask=[[False]],
    )

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns one expected engine name (000); run_params is unused.
    """Returns the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Returns a pair: true only for dynamic shape mode, version 2 converter and a non-quantization
    # precision mode, plus a reason string.
    # Shape op is only converted in dynamic shape mode.
    return (
        run_params.dynamic_shape
        and run_params.is_v2
        and not trt_test.IsQuantizationMode(run_params.precision_mode),
        "Test v2 dynamic_shapes without INT8",
    )


if __name__ == "__main__":
  test.main()
