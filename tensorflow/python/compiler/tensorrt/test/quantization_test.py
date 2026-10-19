# [MODULE DOC]
# Integration tests for TF-TensorRT conversion of quantized (fake-quantization) graphs. Defines two
# private module-level helpers that build a shared small graph and its test parameters, and three
# test classes built on the shared integration test base: one without quantization ranges (INT8,
# static engine only), one with ranges (INT8, offline conversion), and one checking that the fake
# quantization ops split the graph in FP32/FP16 modes. Running as a script starts the test runner.
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
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _GraphFn(x, add_quantization_nodes):
  # [FUNCTION DOC]
  # HELPER: shared graph builder for the quantization test classes in this module.
  # Builds a graph from input tensor x (float tensor of shape 8x8): add 5, subtract 5, multiply by
  # 0.1, matrix-multiply by an 8x1 all-ones constant, with a quantization step between stages.
  # add_quantization_nodes (boolean) decides whether the quantization steps insert fake-quant ops
  # (symmetric ranges 10, 15, 10, 1 and 10) or leave values untouched. Returns the identity
  # tensor named output_0 (shape 8x1).

  def _Quantize(x, r):
    # [FUNCTION DOC]
    # HELPER: optional fake-quantization step for the enclosing graph builder.
    # When the enclosing flag is true, applies a fake quantize op with minimum -r and maximum r to
    # tensor x and returns the result; otherwise returns x unchanged. r is a float range bound.
    if add_quantization_nodes:
      x = gen_array_ops.fake_quant_with_min_max_vars(x, -r, r)
    return x

  x = _Quantize(x, 10.0)
  x = x + 5
  x = _Quantize(x, 15.0)
  x = x - 5
  x = _Quantize(x, 10.0)
  x = x * 0.1
  x = _Quantize(x, 1.0)
  w = constant_op.constant(np.ones((8, 1)), dtype=dtypes.float32)
  x = math_ops.matmul(x, w)
  x = _Quantize(x, 10.0)
  return array_ops.identity(x, name="output_0")


def _GetParams(self):
  # [FUNCTION DOC]
  # HELPER: shared GetParams implementation for the test classes in this module.
  # Takes a test instance (self) and returns the parameters built by its base class for its graph
  # function: float32, one input of shape 8x8 and one output of shape 8x1.
  return self.BuildParams(self.GraphFn, dtypes.float32, [[8, 8]], [[8, 1]])


class QuantizationMissingAllRangesTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test: graph without any quantization nodes, run in INT8 static-engine, offline
  # conversion mode only (with or without calibration). Uses the shared graph builder.
  """Create a graph containing single segment with no quantization ranges."""

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from input x using the shared builder without quantization nodes.
    return _GraphFn(x, add_quantization_nodes=False)

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns the standard test parameters via the shared helper (8x8 input, 8x1 output, float32).
    return _GetParams(self)

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Decides whether to run for the given run_params. Returns a pair: a boolean that is true only
    # for a quantization (INT8) precision mode with offline conversion and a static engine, and a
    # reason string.
    # Only test static engine mode, with or without calibration.
    return (trt_test.IsQuantizationMode(run_params.precision_mode) and
            not run_params.convert_online and not run_params.dynamic_engine
           ), "test static engine, offline conversion and INT8"

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns a list with one expected engine name (000); run_params is unused.
    """Return the expected engines to build."""
    # In static engine mode with calibration, it should build a calibration
    # engine.
    # In static engine mode without calibration, the engine building will
    # succeed but fall back to non-quantized ops.
    return ["TRTEngineOp_000"]


class QuantizationWithRangesTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring is copied from the previous class; this graph does contain
  # quantization nodes (ranges are provided).
  # Integration test: graph with fake-quantization ranges, run in INT8 modes with offline
  # conversion (static or dynamic engine, with or without calibration).
  """Create a graph containing single segment with no quantization ranges."""

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from input x using the shared builder with fake-quantization nodes.
    return _GraphFn(x, add_quantization_nodes=True)

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns the standard test parameters via the shared helper (8x8 input, 8x1 output, float32).
    return _GetParams(self)

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Returns a pair: true only when the precision mode is a quantization (INT8) mode and the
    # conversion is offline (not online), plus a reason string.
    # Test static/dynamic engine with/without calibration.
    return (trt_test.IsQuantizationMode(run_params.precision_mode) and
            not run_params.convert_online), "test offline conversion and INT8"

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns a list with one expected engine name (000); run_params is unused.
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]

  def ExpectedAbsoluteTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the absolute comparison tolerance: very tight for FP32, loose (0.1) otherwise.
    """The absolute tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01

  def ExpectedRelativeTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the relative comparison tolerance: very tight for FP32, loose (0.1) otherwise.
    """The relative tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01


class NonQuantizedPrecisionsWithRangesTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring is copied from another class; this graph contains quantization
  # nodes and is run only in non-INT8 modes.
  # Integration test: graph with fake-quantization nodes in FP32/FP16 modes, where those ops are
  # unsupported and split the graph into two engines.
  """Create a graph containing single segment with no quantization ranges."""

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph from input x using the shared builder with fake-quantization nodes.
    return _GraphFn(x, add_quantization_nodes=True)

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns the standard test parameters via the shared helper (8x8 input, 8x1 output, float32).
    return _GetParams(self)

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Returns a pair: true only when the precision mode is not a quantization mode, plus a reason.
    # Only test FP32/FP16 mode.
    return not trt_test.IsQuantizationMode(
        run_params.precision_mode), "test non-INT8"

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns a list of two expected engine names (000 and 001); run_params is unused.
    """Return the expected engines to build."""
    # The fake quant ops are not supported in FP32/FP16 mode, and will split the
    # graph into two TRT segments.
    return ["TRTEngineOp_000", "TRTEngineOp_001"]

  def ExpectedAbsoluteTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the absolute comparison tolerance: very tight for FP32, loose (0.1) otherwise.
    """The absolute tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01

  def ExpectedRelativeTolerance(self, run_params):
    # [FUNCTION DOC]
    # Returns the relative comparison tolerance: very tight for FP32, loose (0.1) otherwise.
    """The relative tolerance to compare floating point results."""
    return 1.e-05 if run_params.precision_mode == "FP32" else 1.e-01


if __name__ == "__main__":
  test.main()
