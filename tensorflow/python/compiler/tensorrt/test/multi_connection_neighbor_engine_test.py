# [MODULE DOC]
# Integration test for TF-TensorRT conversion of a graph where one convolution output feeds several
# neighboring nodes, some of which are TensorRT-incompatible. Defines one test class built on the
# shared integration test base; it checks that the converter splits the graph into the expected
# number of engines. Running the file as a script starts the test runner. No import-time side effects.
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
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class MultiConnectionNeighborEngineTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Integration test (derived from the TF-TRT integration test base) covering wiring of several
  # neighboring nodes that all connect to one convolution, with TensorRT-incompatible ops
  # interleaved. Fixtures come from the base class; this class supplies the graph, input/output
  # shapes and the expected engine names.
  """Test for multi connection neighboring nodes wiring tests in TF-TRT."""

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph under test from input tensor x (float tensor of shape 2x3x7x5). Applies a
    # random-weight NCHW convolution, then three branches: adding a bias, subtracting a bias
    # (followed by an incompatible op), and adding a bias to the convolution (followed by another
    # incompatible op), plus a third incompatible op applied directly to the convolution output.
    # The branches are recombined by adds and subtracts. Returns the squeezed result tensor named
    # output_0. Uses random numbers for weights and biases.
    dtype = x.dtype
    e = constant_op.constant(
        np.random.normal(.05, .005, [3, 2, 3, 4]), name="weights", dtype=dtype)
    conv = nn.conv2d(
        input=x,
        filter=e,
        data_format="NCHW",
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    b = constant_op.constant(
        np.random.normal(2.0, 1.0, [1, 4, 1, 1]), name="bias", dtype=dtype)
    t = conv + b

    b = constant_op.constant(
        np.random.normal(5.0, 1.0, [1, 4, 1, 1]), name="bias", dtype=dtype)
    q = conv - b
    edge = self.trt_incompatible_op(q)

    b = constant_op.constant(
        np.random.normal(5.0, 1.0, [1, 4, 1, 1]), name="bias", dtype=dtype)
    d = b + conv
    edge3 = self.trt_incompatible_op(d)

    edge1 = self.trt_incompatible_op(conv)
    t = t - edge1
    q = q + edge
    t = t + q
    t = t + d
    t = t - edge3
    return array_ops.squeeze(t, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns the test parameters built by the base class: the graph function, float32 dtype, one
    # input of shape 2x3x7x5 and one expected output of shape 2x4x5x4. Takes no arguments.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[2, 3, 7, 5]],
                            [[2, 4, 5, 4]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Tells the base test which engines must be built: a list of two engine names (000 and 001),
    # regardless of run_params (the run configuration, unused here).
    """Return the expected engines to build."""
    return ["TRTEngineOp_000", "TRTEngineOp_001"]


if __name__ == "__main__":
  test.main()
