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
# Integration tests for TF-TensorRT conversion of graphs using int32 tensors: one checking that
# ops unsupported with int32 (matmul, bias add) are excluded from conversion, and one checking
# calibration works with an int32 input. Built on TfTrtIntegrationTestBase.
"""Test conversion of graphs involving INT32 tensors and operations."""

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class ExcludeUnsupportedInt32Test(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test that a matmul and bias add on int32 tensors are not converted, so that no engines are
  # built.
  """Test exclusion of ops which are not supported in INT32 mode by TF-TRT"""

  def _ConstOp(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: creates random constants for the graph.
    # Returns a constant tensor of the given shape and dtype built from normally distributed
    # random values (cast to the dtype).
    return constant_op.constant(np.random.randn(*shape), dtype=dtype)

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph: matmul of the input with a random 4x10 constant, bias add with a random
    # constant of length 10, and an identity named output_0. Returns the output tensor.
    dtype = x.dtype
    b = self._ConstOp((4, 10), dtype)
    x = math_ops.matmul(x, b)
    b = self._ConstOp((10,), dtype)
    x = nn.bias_add(x, b)
    return array_ops.identity(x, name='output_0')

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: int32, input of shape 100x4, output of shape 100x10.
    return self.BuildParams(self.GraphFn, dtypes.int32, [[100, 4]], [[100, 10]])

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, then disables non-TRT graph optimizers (the layout optimizer would
    # convert bias add from NHWC to NCHW for four-dimensional input).
    super().setUp()
    # Disable layout optimizer, since it will convert BiasAdd with NHWC
    # format to NCHW format under four dimensional input.
    self.DisableNonTrtOptimizers()

  def GetMaxBatchSize(self, run_params):
    # [FUNCTION DOC]
    # Returns the converter max batch size: nothing for dynamic engines, otherwise 100.
    """Returns the max_batch_size that the converter should use for tests."""
    if run_params.dynamic_engine:
      return None
    return 100

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects no engines. Returns an empty list.
    """Return the expected engines to build."""
    return []


class CalibrationInt32Support(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test that INT8 calibration runs correctly when the graph input is int32, using a transpose
  # that converts to TensorRT; runs only for quantization with calibration.
  """Test execution of calibration with int32 input"""

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Builds the graph: transpose of the input with permutation (0, 3, 2, 1), and an identity named
    # output_0. Returns the output tensor.
    # Can use any op that is converted to TRT with int32 inputs
    inp_transposed = array_ops.transpose(inp, [0, 3, 2, 1], name='transpose_0')
    return array_ops.identity(inp_transposed, name='output_0')

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: int32, input of shape 3x4x5x6, output of shape 3x6x5x4.
    return self.BuildParams(self.GraphFn, dtypes.int32, [[3, 4, 5, 6]],
                            [[3, 6, 5, 4]])

  def ShouldRunTest(self, run_params):
    # [FUNCTION DOC]
    # Runs only when quantization with calibration is used. Returns a pair of boolean and reason
    # string.
    # Although test passes with all configurations but only
    # execute INT8 with use_calibration=True because
    # that is the purpose of the test.
    return trt_test.IsQuantizationWithCalibration(
        run_params), 'test calibration and INT8'

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects a single engine. Returns a list with its name.
    return ['TRTEngineOp_000']


if __name__ == '__main__':
  test.main()
