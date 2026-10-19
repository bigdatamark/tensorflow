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
# Integration test for TF-TensorRT conversion of binary elementwise operations between a tensor
# and constant weights of many shapes that map to scale or elementwise layers. Built on
# TfTrtIntegrationTestBase; runs via the standard test main.
"""Model script to test TF-TensorRT integration."""

import os
import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class BinaryTensorWeightBroadcastTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test that adding random constant weights of eight different shapes, on both sides of the
  # addition, to a tensor separated by TRT-incompatible ops yields one engine per addition (16,
  # plus one for the final reshape in dynamic shape mode).
  """Tests for scale & elementwise layers in TF-TRT."""

  def _ConstOp(self, shape):
    # [FUNCTION DOC]
    # HELPER: creates random constants for the graph.
    # Returns a float32 constant tensor of the given shape with normally distributed random values.
    return constant_op.constant(np.random.randn(*shape), dtype=dtypes.float32)

  def GraphFn(self, x):
    # [FUNCTION DOC]
    # Builds the graph: for each of eight weight shapes (scale-like and elementwise-like), adds a
    # random constant to the running tensor then passes it through a TRT-incompatible op, and does
    # it again with the constant on the left. Finally reshapes to 5 rows and names it output_0.
    # Returns the output tensor.
    for weights_shape in [
        (1,),  # scale
        (24, 1, 1),  # scale
        (24, 24, 20),  # scale
        (20,),  # elementwise
        (1, 24, 1, 1),  # elementwise
        (1, 24, 24, 1),  # elementwise
        (1, 24, 24, 20),  # elementwise
        (24, 20),  # elementwise
    ]:
      a = self._ConstOp(weights_shape)
      f = x + a
      x = self.trt_incompatible_op(f)
      a = self._ConstOp(weights_shape)
      f = a + x
      x = self.trt_incompatible_op(f)
    return gen_array_ops.reshape(x, [5, -1], name="output_0")

  def GetParams(self):
    # [FUNCTION DOC]
    # Test parameters: float32, input of shape 10x24x24x20, output of shape 5x23040.
    # TODO(aaroey): test graph with different dtypes.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[10, 24, 24, 20]],
                            [[5, 23040]])

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Expects 17 engines in dynamic shape mode (the final reshape gets its own engine) and 16
    # otherwise. Returns the list of zero-padded engine names.
    """Return the expected engines to build."""
    # The final reshape op is converted only in dynamic shape mode. This op is
    # placed into a new engine due to the preceding trt_incompatible_ops.
    num_engines = 17 if run_params.dynamic_shape else 16
    return [f"TRTEngineOp_{i:03d}" for i in range(num_engines)]

  # TODO(b/176540862): remove this routine to disallow native segment execution
  # for TensorRT 7+.
  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, then sets an environment variable that allows native segment
    # execution of engines (process-wide side effect, never reset here), logs the physical GPUs and
    # enables memory growth on each.
    super().setUp()
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"
    gpus = config.list_physical_devices("GPU")

    logging.info("Found the following GPUs:")
    for gpu in gpus:
      logging.info(f"\t- {gpu}")
      config.set_memory_growth(gpu, True)


if __name__ == "__main__":
  test.main()
