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
# TF-TensorRT integration test exercising chains of unary math operations and squeezes across
# several branches of one graph. Defines a single test class on the shared TF-TRT integration test
# base; run as a script it launches the test runner.
"""Model script to test TF-TensorRT integration."""

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class UnaryTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test case converting a graph made of unary operations (abs, exp, log, sqrt, rsqrt, negative,
  # reciprocal) and squeezes; expected engine count depends on whether dynamic shape mode is used.
  """Test for unary operations in TF-TRT."""

  def GraphFn(self, x1, x2):
    # [FUNCTION DOC]
    # Builds the graph from two input tensors. Branch a processes the first input, branch b
    # processes a random constant of shape 5x8x12 (a different batch size), branch c processes the
    # second input through squeezes and a reshape. The result is a times b divided by c, squeezed
    # and named output_0, which is returned. Uses random numbers when creating the constant.
    x = x1
    q = math_ops.abs(x)
    q = q + 1.0
    q = gen_math_ops.exp(q)
    q = gen_math_ops.log(q)
    q = array_ops.squeeze(q, axis=-2)
    q = math_ops.abs(q)
    q = q + 2.2
    q = gen_math_ops.sqrt(q)
    q = gen_math_ops.rsqrt(q)
    q = math_ops.negative(q)
    q = array_ops.squeeze(q, axis=3)
    q = math_ops.abs(q)
    q = q + 3.0
    a = gen_math_ops.reciprocal(q)

    # this chain of operations has a batch size of 5, which is different from
    # the batch size for the other operations.
    x = constant_op.constant(np.random.randn(5, 8, 12), dtype=x.dtype)
    q = math_ops.abs(x)
    q = q + 2.0
    q = gen_math_ops.exp(q)
    q = gen_math_ops.log(q)
    q = math_ops.abs(q)
    q = q + 2.1
    q = gen_math_ops.sqrt(q)
    q = gen_math_ops.rsqrt(q)
    q = math_ops.negative(q)
    q = math_ops.abs(q)
    q = q + 4.0
    b = gen_math_ops.reciprocal(q)

    # TODO(jie): this one will break, broadcasting on batch.
    x = x2
    q = math_ops.abs(x)
    q = q + 5.0
    q = gen_math_ops.exp(q)
    q = array_ops.squeeze(q, axis=[-1, -2, 3])
    q = gen_math_ops.log(q)
    q = math_ops.abs(q)
    q = q + 5.1
    q = gen_array_ops.reshape(q, [12, 5, 1, 1, 8, 1, 12])
    q = array_ops.squeeze(q, axis=[5, 2, 3])
    q = gen_math_ops.sqrt(q)
    q = math_ops.abs(q)
    q = q + 5.2
    q = gen_math_ops.rsqrt(q)
    q = math_ops.negative(q)
    q = math_ops.abs(q)
    q = q + 5.3
    c = gen_math_ops.reciprocal(q)

    q = a * b
    q = q / c
    return array_ops.squeeze(q, name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns parameters: float32 inputs of shapes 12x5x8x1x1x12 and 12x5x8x1x12x1x1, expected
    # output shape 12x5x8x12.
    return self.BuildParams(self.GraphFn, dtypes.float32,
                            [[12, 5, 8, 1, 1, 12], [12, 5, 8, 1, 12, 1, 1]],
                            [[12, 5, 8, 12]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns expected engine names: one engine in dynamic shape mode, otherwise two (the two
    # input branches become separate engines).
    """Return the expected engines to build."""
    if run_params.dynamic_shape:
      # All the ops are converted into a single TRTEngineOp.
      return ["TRTEngineOp_000"]
    else:
      # Final squeeze and div ops are not converted. The x1 and x2 branches
      # are segmented into separate TRTEngineOp.
      return ["TRTEngineOp_000", "TRTEngineOp_001"]


if __name__ == "__main__":
  test.main()
