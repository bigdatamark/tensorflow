# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
# [MODULE DOC]
# HELPER: standalone generator script producing test data for the TF-TRT tests.
# Creates a small model using either the version 2 or version 1 API, saves it, converts it with
# the TF-TensorRT converter and saves the converted result. When run as a script it generates the
# version 2 model into tf_saved_model and tftrt_saved_model in the working directory.
# KNOWN ISSUE: the operating-system module is used in the version 2 generator but never imported,
# so that function fails with a name error at the terminal-size lookup.
"""Saves a SavedModel after TensorRT conversion.

   The saved model is loaded and executed by tests to ensure backward
   compatibility across TF versions.
   The script may not work in TF1.x.

   Instructions on how to use this script:
   - Execute the script as follows:
       python gen_tftrt_model
   - Rename tftrt_saved_model to what makes sense for your test.
   - Delete directory tf_saved_model unless you want to use it.
"""

from tensorflow.python import Session
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.trackable import autotrackable


def GetGraph(input1, input2, var):
  # [FUNCTION DOC]
  # HELPER: shared graph definition for both generators. Given two input tensors and a variable,
  # computes a chain of adds and a multiply of input1 with (input1 plus var), and returns the
  # result as an identity tensor named output.
  """Define graph."""
  add = input1 + var
  mul = input1 * add
  add = mul + add
  add = add + input2
  out = array_ops.identity(add, name="output")
  return out


def GenerateModelV2(tf_saved_model_dir, tftrt_saved_model_dir):
  # [FUNCTION DOC]
  # Defines a trackable model with one traced function, saves it to tf_saved_model_dir (string
  # path), converts it with the version 2 TF-TensorRT converter, prints a detailed summary (line
  # width at least 160 or the terminal width) and saves the converted model to tftrt_saved_model_dir
  # (string path). Writes files to disk; returns nothing.
  # KNOWN ISSUE: uses the os module without importing it; a name error is raised, not caught by
  # the handler that only expects an operating-system error.
  """Generate and convert a model using TFv2 API."""

  class SimpleModel(autotrackable.AutoTrackable):
    # [CLASS DOC]
    # Trackable model with a variable created lazily on the first trace of its run function.
    """Define model with a TF function."""

    def __init__(self):
      # [FUNCTION DOC]
      # Constructor: initializes the variable attribute to nothing/null.
      self.v = None

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32)
    ])
    def run(self, input1, input2):
      # [FUNCTION DOC]
      # Traced function for two float32 inputs of shape unknown-by-1-by-1. Creates the variable on
      # first use, then returns the shared graph result for the inputs and the variable.
      if self.v is None:
        self.v = variables.Variable([[[1.0]]], dtype=dtypes.float32)
      return GetGraph(input1, input2, self.v)

  root = SimpleModel()

  # Saved TF model
  # pylint: disable=not-callable
  save(
      root,
      tf_saved_model_dir,
      {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: root.run})

  # Convert TF model to TensorRT
  converter = trt_convert.TrtGraphConverterV2(
      input_saved_model_dir=tf_saved_model_dir)
  converter.convert()
  try:
    line_length = max(160, os.get_terminal_size().columns)
  except OSError:
    line_length = 160
  converter.summary(line_length=line_length, detailed=True)
  converter.save(tftrt_saved_model_dir)


def GenerateModelV1(tf_saved_model_dir, tftrt_saved_model_dir):
  # [FUNCTION DOC]
  # Builds a graph-mode model with two placeholders and a variable, saves it as a saved model
  # with a serving signature to tf_saved_model_dir (string path), then converts it with the
  # version 1 converter (dynamic ops enabled) and saves the result to tftrt_saved_model_dir
  # (string path). Writes files to disk; returns nothing.
  """Generate and convert a model using TFv1 API."""

  def SimpleModel():
    # [FUNCTION DOC]
    # HELPER: builds the graph inside a fresh graph object. Returns the graph, the variable, both
    # input placeholders and the output tensor.
    """Define model with a TF graph."""

    def GraphFn():
      # [FUNCTION DOC]
      # HELPER: creates placeholders, the variable and the shared computation in the default
      # graph. Returns the enclosing graph (assigned in the outer scope before this is called),
      # the variable, the two inputs and the output.
      input1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input1")
      input2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input2")
      var = variables.Variable([[[1.0]]], dtype=dtypes.float32, name="v1")
      out = GetGraph(input1, input2, var)
      return g, var, input1, input2, out

    g = ops.Graph()
    with g.as_default():
      return GraphFn()

  g, var, input1, input2, out = SimpleModel()
  signature_def = signature_def_utils.build_signature_def(
      inputs={
          "input1": utils.build_tensor_info(input1),
          "input2": utils.build_tensor_info(input2)
      },
      outputs={"output": utils.build_tensor_info(out)},
      method_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
  saved_model_builder = builder.SavedModelBuilder(tf_saved_model_dir)
  with Session(graph=g) as sess:
    sess.run(var.initializer)
    saved_model_builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        })
  saved_model_builder.save()

  # Convert TF model to TensorRT
  converter = trt_convert.TrtGraphConverter(
      input_saved_model_dir=tf_saved_model_dir, is_dynamic_op=True)
  converter.convert()
  converter.save(tftrt_saved_model_dir)


if __name__ == "__main__":
  GenerateModelV2(
      tf_saved_model_dir="tf_saved_model",
      tftrt_saved_model_dir="tftrt_saved_model")
