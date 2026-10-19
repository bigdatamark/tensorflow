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
# [MODULE DOC]
# Integration test checking that a TF-TensorRT converted model keeps the expected input and output
# shapes and dtypes on the TRTEngineOp. It defines one test class built on the shared TF-TRT
# integration test base; run as a script it launches the test runner. No import-time side effects.
"""This script test input and output shapes and dtype of the TRTEngineOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


class TRTEngineOpInputOutputShapeTest(trt_test.TfTrtIntegrationTestBase):
  # [CLASS DOC]
  # Test case verifying that the converted saved model exposes a batch-unknown input shape and the
  # right output shapes and float32 dtypes. Overrides the inference-graph hook of the base class to
  # inspect the converted model, and declares a single expected engine.
  """Testing the output shape of a TRTEngine."""

  def GraphFn(self, inp):
    # [FUNCTION DOC]
    # Builds the test graph. Takes one input tensor, squeezes axis 2, applies relu, doubles the
    # result and also reduces it to a scalar sum. Returns two named outputs: the doubled tensor
    # (output_0) and the scalar sum (output_1).
    b = array_ops.squeeze(inp, axis=[2])
    c = nn.relu(b)
    d1 = c + c
    d2 = math_ops.reduce_sum(d1)

    d1 = array_ops.identity(d1, name="output_0")
    d2 = array_ops.identity(d2, name="output_1")
    return d1, d2

  def GetParams(self):
    # [FUNCTION DOC]
    # Returns the test parameters from the base-class builder: float32 input of shape 1x2x1x4 and
    # expected output shapes 1x2x4 and scalar.
    return self.BuildParams(self.GraphFn, dtypes.float32, [[1, 2, 1, 4]],
                            [[1, 2, 4], []])

  def _GetInferGraph(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: hook into the base-class conversion flow to validate the converted model.
    # Calls the parent implementation (all arguments forwarded) to get the directory of the
    # converted saved model, loads it with the serving tag and checks the serving signature: the
    # input dtype and the two output dtypes must be float32, the input shape must have an unknown
    # batch dimension (2x1x4 otherwise), the first output unknown batch by 2 by 4 and the second
    # output scalar. Failed assertions fail the test. Returns the saved model directory.
    trt_saved_model_dir = super(TRTEngineOpInputOutputShapeTest,
                                self)._GetInferGraph(*args, **kwargs)

    def get_func_from_saved_model(saved_model_dir):
      # [FUNCTION DOC]
      # HELPER: loads a saved model for _GetInferGraph.
      # Loads the saved model in the given directory with the serving tag (falling back when the
      # load function is a module) and returns a pair: the default serving signature function and
      # the loaded model object.
      try:  # Necessary for `bazel run ...`
        saved_model_load_fn = load.load
      except AttributeError:  # All the other cases
        saved_model_load_fn = load
      saved_model_loaded = saved_model_load_fn(
          saved_model_dir, tags=[tag_constants.SERVING])
      graph_func = saved_model_loaded.signatures[
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
      return graph_func, saved_model_loaded

    func, _ = get_func_from_saved_model(trt_saved_model_dir)

    input_shape = func.inputs[0].shape
    if isinstance(input_shape, tensor_shape.TensorShape):
      input_shape = input_shape.as_list()

    output_shapes = [
        out_shape.shape.as_list() if isinstance(
            out_shape.shape, tensor_shape.TensorShape) else out_shape.shape
        for out_shape in func.outputs
    ]

    self.assertEqual(func.inputs[0].dtype, dtypes.float32)
    self.assertEqual(func.outputs[0].dtype, dtypes.float32)
    self.assertEqual(func.outputs[1].dtype, dtypes.float32)

    self.assertEqual(input_shape, [None, 2, 1, 4])
    self.assertEqual(output_shapes[0], [None, 2, 4])
    self.assertEqual(output_shapes[1], [])

    return trt_saved_model_dir

  def ExpectedEnginesToBuild(self, run_params):
    # [FUNCTION DOC]
    # Returns the list of engine names expected after conversion: exactly one engine, named
    # TRTEngineOp_000, regardless of the run parameters.
    """Return the expected engines to build."""
    return ["TRTEngineOp_000"]


if __name__ == "__main__":
  test.main()
