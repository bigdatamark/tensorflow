# [MODULE DOC]
# Tests for the TF-TensorRT conversion API (TF1 graph converter and TF2 graph converter):
# conversion, calibration, building and saving engines, engine cache lifetime, device placement,
# signature retention, runtime-build options and variable handling. Defines one test class with
# helper graph/model builders. Most tests require TensorRT and a GPU. The main block runs the
# tests only when TensorRT is enabled. Nested functions inside traced model methods are
# documented with comments only; no test inspects source line numbers.
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
"""Utilities to test TF-TensorRT integration."""

import gc
import os
import re
import tempfile
from unittest import mock

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
from tensorflow.compiler.tf2tensorrt.utils.trt_engine_instance_pb2 import TRTEngineInstance  # pylint: disable=g-importing-member
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.compiler.tensorrt.test import test_utils
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util.lazy_loader import LazyLoader

_SAVED_MODEL_SIGNATURE_KEY = "mypredict"

gen_trt_ops = LazyLoader(
    "gen_trt_ops", globals(),
    "tensorflow.compiler.tf2tensorrt.ops.gen_trt_ops")


class TrtConvertTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the TF-TRT conversion API: the TF1 graph converter (graph definitions and
  # saved models, offline/online, dynamic/static engines, calibration) and the TF2 converter
  # (conversion, build, save, INT8 calibration, engine cache lifetime, device placement, signature
  # retention, runtime-build options, variables). Provides helper builders for graphs and saved
  # models. Many tests need a GPU and TensorRT to be meaningful.
  """Class to test Tensorflow-TensorRT integration python API."""

  # Use a small max_workspace_size for tests so they don't consume too much GPU
  # memory.
  _TRT_MAX_WORKSPACE_SIZE_BYTES = (
      trt_convert.DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES)

  def mkdtemp(self):
    # [FUNCTION DOC]
    # HELPER: creates and returns a new unique directory inside the test's temporary directory.
    return tempfile.mkdtemp(dir=self.get_temp_dir())

  def testTRTEngineInstanceAvailable(self):
    # [FUNCTION DOC]
    # Test: the engine instance protocol buffer can be imported and created, and exposes a
    # serialized engine field (asserted via attribute presence).
    # test if we can access the TRTEngineInstance protobuf
    assert hasattr(TRTEngineInstance(), "serialized_engine")

  def _GetConfigProto(self, rewriter_config=None):
    # [FUNCTION DOC]
    # HELPER: builds a session configuration with GPU memory growth enabled. Input:
    # rewriter_config, optional Grappler rewriter configuration copied into the graph options.
    # Output: the configuration protocol buffer.
    """Get ConfigProto for session creation."""
    config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(allow_growth=True))
    if rewriter_config:
      config.graph_options.rewrite_options.CopyFrom(rewriter_config)
    return config

  @classmethod
  def _GetGraph(cls, inp1, inp2, var):
    # [FUNCTION DOC]
    # HELPER: builds the small test computation (square of the first input plus first input times
    # variable, plus first input, second input and variable) and names the final identity op
    # "output". Inputs: inp1, inp2 tensors; var, a tensor or variable. Output: the output tensor.
    """Get the graph for testing."""
    # The graph computes: inp1^2 + inp1*var + inp1 + inp2 + var
    add = inp1 + var
    mul = inp1 * add
    add = mul + add
    add = add + inp2
    out = array_ops.identity(add, name="output")
    return out

  def _GetShapeOpModel(self):
    # [FUNCTION DOC]
    # HELPER: returns an instance of a small trackable model whose function takes a rank-2
    # float input of unknown shape, adds one, takes the shape, applies an unsupported cumulative
    # sum so TF-TRT builds two engines (int32 outputs/inputs), and doubles it.

    class ShapeOpModel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: trackable test model with a shape-op based function (see the builder).

      def __init__(self):
        # [FUNCTION DOC]
        # Initializes the model with an unset variable placeholder attribute.
        self.v = None

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, None], dtype=dtypes.float32)
      ])
      def run(self, x):
        # [FUNCTION DOC]
        # Graph function: input x (float, any 2-D shape). Computes shape of x+1, cumulative sum of
        # that shape, doubles it and returns it as an int32 vector named output.
        q = x + 1
        q_shape = array_ops.shape(q)
        # Add an OP that is not supported by TF-TRT. This allows TF-TRT to build
        # two engines. The first engine produces an int32 output and the second
        # engines has an int32 input and an int32 output.
        q = math_ops.cumsum(q_shape)
        q = q * 2
        return array_ops.identity(q, name="output")

    return ShapeOpModel()

  def _GetModelForV2(self):
    # [FUNCTION DOC]
    # HELPER: returns an instance of a simple trackable model for TF2 conversion tests, whose
    # function takes two float inputs of shape unknown-by-1-by-1 and applies the shared test graph
    # with a lazily created variable holding 1.0.

    class SimpleModel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: trackable test model with one two-input function and a lazily created variable.

      def __init__(self):
        # [FUNCTION DOC]
        # Initializes the model with the variable unset; it is created on first trace.
        self.v = None

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32),
          tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32)
      ])
      def run(self, inp1, inp2):
        # [FUNCTION DOC]
        # Graph function over two float tensors: creates the variable on first trace, then returns
        # the shared test graph output computed from the inputs and the variable.
        if self.v is None:
          self.v = variables.Variable([[[1.0]]], dtype=dtypes.float32)
        return TrtConvertTest._GetGraph(inp1, inp2, self.v)

    return SimpleModel()

  def _GetGraphForV1(self, device):
    # [FUNCTION DOC]
    # HELPER: builds a TF1 graph with two placeholders, a variable and the shared test graph.
    # Input: device, optional device string under which all ops are created (nothing for none).
    # Output: tuple of graph, variable, the two inputs and the output tensor.

    def _GraphFn():
      # [FUNCTION DOC]
      # HELPER: creates the placeholders, variable and test graph in the current default graph and
      # returns the graph, variable, inputs and output (uses the enclosing graph).
      inp1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input1")
      inp2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input2")
      var = variables.Variable([[[1.0]]], dtype=dtypes.float32, name="v1")
      out = TrtConvertTest._GetGraph(inp1, inp2, var)
      return g, var, inp1, inp2, out

    g = ops.Graph()
    with g.as_default():
      if device:
        with g.device(device):
          return _GraphFn()
      return _GraphFn()

  def _GetGraphDefForV1(self, device):
    # [FUNCTION DOC]
    # HELPER: builds the TF1 graph, initializes the variable in a session, freezes variables into
    # constants for the output node and asserts the resulting node-name to op-type mapping.
    # Input: device, optional device string. Output: the frozen graph definition.
    """Get the graph def for testing."""
    g, var, _, _, _ = self._GetGraphForV1(device)
    with self.session(graph=g, config=self._GetConfigProto()) as sess:
      sess.run(var.initializer)
      graph_def = convert_to_constants.convert_variables_to_constants(
          sess, g.as_graph_def(add_shapes=True), ["output"])
    node_name_to_op = {node.name: node.op for node in graph_def.node}
    self.assertEqual(
        {
            "v1": "Const",
            "add/ReadVariableOp": "Identity",
            "input1": "Placeholder",
            "input2": "Placeholder",
            "add": "AddV2",
            "mul": "Mul",
            "add_1": "AddV2",
            "add_2": "AddV2",
            "output": "Identity"
        }, node_name_to_op)
    return graph_def

  def _WriteInputSavedModelForV1(self, input_saved_model_dir, device):
    # [FUNCTION DOC]
    # HELPER: writes the TF1 test graph as a saved model with a serving tag and a predict
    # signature (inputs myinput1/myinput2, output myoutput) under the test signature key.
    # Inputs: input_saved_model_dir, destination; device, optional device string. Returns nothing.
    """Write the saved model as an input for testing."""
    g, var, inp1, inp2, out = self._GetGraphForV1(device)
    signature_def = signature_def_utils.build_signature_def(
        inputs={
            "myinput1": utils.build_tensor_info(inp1),
            "myinput2": utils.build_tensor_info(inp2)
        },
        outputs={"myoutput": utils.build_tensor_info(out)},
        method_name=signature_constants.PREDICT_METHOD_NAME)
    saved_model_builder = builder.SavedModelBuilder(input_saved_model_dir)
    with self.session(graph=g, config=self._GetConfigProto()) as sess:
      sess.run(var.initializer)
      saved_model_builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          signature_def_map={_SAVED_MODEL_SIGNATURE_KEY: signature_def})
    saved_model_builder.save()

  def _ConvertGraphV1(self,
                      output_saved_model_dir=None,
                      need_calibration=False,
                      max_batch_size=1,
                      minimum_segment_size=3,
                      is_dynamic_op=False,
                      maximum_cached_engines=1,
                      device=None):
    # [FUNCTION DOC]
    # HELPER: converts the TF1 test graph with the TF1 graph converter. Inputs:
    # output_saved_model_dir (if given, a saved model is written and used as input and the result
    # is saved there; otherwise a frozen graph definition is used); need_calibration (INT8
    # precision plus calibration over 10 runs; forces dynamic ops); max_batch_size (default 1,
    # ignored/none for dynamic ops); minimum_segment_size (default 3); is_dynamic_op;
    # maximum_cached_engines (default 1); device (optional). Output: converted (or calibrated)
    # graph definition.
    """Helper method to convert a GraphDef or SavedModel using TF-TRT."""
    input_saved_model_dir = None
    if output_saved_model_dir:
      input_saved_model_dir = self.mkdtemp()
      self._WriteInputSavedModelForV1(input_saved_model_dir, device)

    # Calibration requires dynamic_op.
    if need_calibration:
      is_dynamic_op = True

    # For dynamic_op, the converter requires the unused max_batch_size=None.
    if is_dynamic_op:
      max_batch_size = None

    converter = trt_convert.TrtGraphConverter(
        input_saved_model_dir=input_saved_model_dir,
        input_saved_model_signature_key=_SAVED_MODEL_SIGNATURE_KEY,
        input_graph_def=None
        if input_saved_model_dir else self._GetGraphDefForV1(device),
        nodes_denylist=None if input_saved_model_dir else ["output"],
        max_batch_size=max_batch_size,
        max_workspace_size_bytes=TrtConvertTest._TRT_MAX_WORKSPACE_SIZE_BYTES,
        precision_mode=(trt_convert.TrtPrecisionMode.INT8 if need_calibration
                        else trt_convert.TrtPrecisionMode.FP32),
        minimum_segment_size=minimum_segment_size,
        is_dynamic_op=is_dynamic_op,
        maximum_cached_engines=maximum_cached_engines)
    output_graph_def = converter.convert()

    if need_calibration:

      class CalibrationData(object):
        # [CLASS DOC]
        # HELPER: feed provider for calibration; each call yields incrementing constant inputs.

        def __init__(self):
          # [FUNCTION DOC]
          # Starts the counter at zero.
          self._data = 0

        def next(self):
          # [FUNCTION DOC]
          # Increments the counter and returns a feed mapping giving both input tensors a single
          # element equal to the counter.
          self._data += 1
          return {"input1:0": [[[self._data]]], "input2:0": [[[self._data]]]}

      output_graph_def = converter.calibrate(
          fetch_names=["output:0"],
          num_runs=10,
          feed_dict_fn=CalibrationData().next)

    if output_saved_model_dir is not None:
      converter.save(output_saved_model_dir=output_saved_model_dir)
    return output_graph_def

  # Remove the graph sequence number prefix from the name only if the name has
  # a prefix TRTEngineOp_n_.
  def _MayRemoveGraphSequenceNumber(self, name):
    # [FUNCTION DOC]
    # HELPER: normalizes engine op names by removing the numeric graph sequence part after the
    # engine op prefix (three or more digits), leaving other names unchanged. Input: name string.
    prefix = re.search(r"TRTEngineOp_\d{3,}_", name)
    if prefix and name.startswith(prefix.group(0)):
      parts = name.split("_", maxsplit=2)
      assert len(parts) == 3
      return parts[0] + "_" + parts[2]
    return name

  # Return the unique TRTEngineOp in the given graph def.
  def _GetUniqueTRTEngineOp(self, graph_def):
    # [FUNCTION DOC]
    # HELPER: returns the single engine op node of graph_def; asserts that exactly one exists.
    trt_engine_nodes = [
        node for node in graph_def.node if node.op == "TRTEngineOp"
    ]
    assert len(trt_engine_nodes) == 1
    return trt_engine_nodes[0]

  def _TestTrtGraphConverter(self,
                             device,
                             output_saved_model_dir=None,
                             need_calibration=False,
                             is_dynamic_op=False):
    # [FUNCTION DOC]
    # HELPER: shared test body for the TF1 converter. Converts the test graph (optionally also
    # via a saved model and with calibration), then for the returned graph and the saved model
    # graph asserts the node-to-op mapping: on a CPU device nothing is converted, otherwise one
    # engine op replaces the compute nodes. With calibration it checks engines carry calibration
    # data (none on CPU) and runs the calibrated graph for ten inputs, comparing to expected values.
    # Inputs: device, output_saved_model_dir, need_calibration, is_dynamic_op.
    """General method to test trt_convert.TrtGraphConverter()."""
    output_graph_def = self._ConvertGraphV1(
        output_saved_model_dir=output_saved_model_dir,
        need_calibration=need_calibration,
        is_dynamic_op=is_dynamic_op,
        device=device)
    graph_defs_to_verify = [output_graph_def]

    if output_saved_model_dir:
      saved_model_graph_def = saved_model_utils.get_meta_graph_def(
          output_saved_model_dir, tag_constants.SERVING).graph_def
      self.assertIsInstance(saved_model_graph_def, graph_pb2.GraphDef)
      graph_defs_to_verify.append(saved_model_graph_def)

    for graph_def in graph_defs_to_verify:
      node_name_to_op = {
          self._MayRemoveGraphSequenceNumber(node.name): node.op
          for node in graph_def.node
      }
      if device is not None and device.startswith("/CPU:"):
        self.assertEqual(
            {
                "add": "AddV2",
                "v1": "Const",
                "add_1": "AddV2",
                "add_2": "AddV2",
                "input1": "Placeholder",
                "input2": "Placeholder",
                "mul": "Mul",
                "output": "Identity"
            }, node_name_to_op)
      else:
        self.assertEqual(
            {
                "input1": "Placeholder",
                "input2": "Placeholder",
                "TRTEngineOp_000": "TRTEngineOp",
                "output": "Identity"
            }, node_name_to_op)

      if need_calibration:
        trt_engine_nodes = [
            node for node in graph_def.node if node.op == "TRTEngineOp"
        ]
        if device is not None and device.startswith("/CPU:"):
          self.assertEmpty(trt_engine_nodes)
          return

        self.assertNotEmpty(trt_engine_nodes)
        for node in trt_engine_nodes:
          self.assertTrue(len(node.attr["calibration_data"].s))
        # Run the calibrated graph.
        # TODO(laigd): consider having some input where the answer is different.
        with ops.Graph().as_default():
          importer.import_graph_def(graph_def, name="")
          with self.session(config=self._GetConfigProto()) as sess:
            for test_data in range(10):
              self.assertEqual((test_data + 1.0)**2 + test_data,
                               sess.run(
                                   "output:0",
                                   feed_dict={
                                       "input1:0": [[[test_data]]],
                                       "input2:0": [[[test_data]]]
                                   }))

  @parameterized.named_parameters([
      ("NoDeviceAssignment", None),
      ("GPU", "/GPU:0"),
      ("CPU", "/CPU:0"),
  ])
  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_OfflineConversion(self, device):
    # [FUNCTION DOC]
    # Test (graph mode, parameterized over no device, GPU, CPU): runs the shared converter test
    # on a graph definition and on a saved model, with and without calibration, expecting the
    # per-device conversion results checked by the helper.
    """Test case for trt_convert.TrtGraphConverter()."""

    for need_calibration in [False, True]:
      # Use GraphDef as input.
      self._TestTrtGraphConverter(device)

      # Use SavedModel as input.
      self._TestTrtGraphConverter(
          device,
          output_saved_model_dir=self.mkdtemp(),
          need_calibration=need_calibration)

  @parameterized.named_parameters([
      ("NoDeviceAssignment", None),
      ("GPU", "/device:GPU:0"),
      ("CPU", "/device:CPU:0"),
  ])
  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_OnlineConversion(self, device):
    # [FUNCTION DOC]
    # Test (graph mode, parameterized over no device, GPU, CPU): enables conversion through the
    # Grappler rewriter configuration in a session (FP32, static ops, batch 1, TF1 style), builds
    # the test graph with the first input reused as the variable, optionally under the device, and
    # runs it with batch size 1 expecting the correct output.
    """Test case for TF-TRT conversion using Grappler directly."""

    conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt_convert.TrtPrecisionMode.FP32)
    config = self._GetConfigProto(
        rewriter_config=trt_convert.get_tensorrt_rewriter_config(
            conversion_params,
            is_dynamic_op=False,
            max_batch_size=1,
            is_v2=False))

    with ops.Graph().as_default():
      # Online conversion requires a frozen graph, so we reuse inp1 as the var
      # argument.
      inp1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input1")
      inp2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input2")
      if device:
        with ops.device(device):
          TrtConvertTest._GetGraph(inp1, inp2, inp1)
      else:
        TrtConvertTest._GetGraph(inp1, inp2, inp1)
      with self.session(config=config) as sess:
        self._TestRun(sess, batch_size=1)

  def _CreateConverterV2(
      self,
      input_saved_model_dir,
      input_saved_model_signature_key=_SAVED_MODEL_SIGNATURE_KEY,
      max_workspace_size_bytes=10 << 20,  # Use a smaller workspace.
      precision_mode=trt_convert.TrtPrecisionMode.FP32,
      maximum_cached_engines=2,
      allow_build_at_runtime=True):
    # [FUNCTION DOC]
    # HELPER: creates a TF2 graph converter. Inputs: input_saved_model_dir (required);
    # input_saved_model_signature_key (default the test key); max_workspace_size_bytes (default
    # 10 MiB); precision_mode (default FP32); maximum_cached_engines (default 2);
    # allow_build_at_runtime (default true). Output: the converter.
    return trt_convert.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        input_saved_model_signature_key=input_saved_model_signature_key,
        max_workspace_size_bytes=max_workspace_size_bytes,
        precision_mode=precision_mode,
        maximum_cached_engines=maximum_cached_engines,
        allow_build_at_runtime=allow_build_at_runtime)

  def _CheckTrtOps(self, concrete_func, check_fn=None, num_engines=1):
    # [FUNCTION DOC]
    # HELPER: counts engine ops in a concrete function's graph and its function library,
    # optionally applying check_fn (callback taking the node) to each, and asserts the count equals
    # num_engines (default 1).
    graph_def = concrete_func.graph.as_graph_def()
    trt_op_names = []
    for node in graph_def.node:
      if node.op == "TRTEngineOp":
        trt_op_names.append(self._MayRemoveGraphSequenceNumber(node.name))
        if check_fn:
          check_fn(node)
    for func in graph_def.library.function:
      for node in func.node_def:
        if node.op == "TRTEngineOp":
          trt_op_names.append(self._MayRemoveGraphSequenceNumber(node.name))
          if check_fn:
            check_fn(node)
    self.assertLen(trt_op_names, num_engines)

  def _RandomInput(self, shape, dtype=np.float32):
    # [FUNCTION DOC]
    # HELPER: returns two independent random arrays of the given shape and dtype (default
    # float32), sampled uniformly in [0, 1).
    inp1 = np.random.random_sample(shape).astype(dtype)
    inp2 = np.random.random_sample(shape).astype(dtype)
    return inp1, inp2

  def _GetAssetFile(self, output_saved_model_dir, trt_engine_name):
    # [FUNCTION DOC]
    # HELPER: returns the path of the serialized engine asset for an engine inside a saved model
    # directory.
    asset_file = os.path.join(output_saved_model_dir,
                              "assets/trt-serialized-engine." + trt_engine_name)
    return asset_file

  def _BuildGraphWithInputGenerator(self, InputFunc, np_input=None):
    # [FUNCTION DOC]
    # HELPER: saves the shape-op model, converts it with the TF2 converter (dynamic shape,
    # minimum segment size 2), builds engines using InputFunc (a generator function of inputs)
    # and saves the result. np_input optional: if given, the expected output is computed from the
    # original model. Output: pair of (converted saved model directory, expected output or nothing).
    # Create the SavedModel.
    root = self._GetShapeOpModel()
    expected_output = None if np_input is None else root.run(np_input)
    input_saved_model_dir = self.mkdtemp()
    save.save(root, input_saved_model_dir, signatures=root.run)

    # Convert the graph to TF-TRT.
    conv_params = trt_convert.TrtConversionParams(minimum_segment_size=2)
    converter = trt_convert.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        use_dynamic_shape=True,
        **conv_params._asdict())
    converter.convert()

    # Build the graph with the input generator. This runs the TRTEngineOp native
    # segment.
    converter.build(InputFunc)
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    del converter
    return output_saved_model_dir, expected_output

  def _BuildGraphWithInputGeneratorTwoInputs(self, InputFunc, np_input=None):
    # [FUNCTION DOC]
    # HELPER: like the one-input builder but with the two-input model. Saves it, converts it,
    # asserts one engine op and that saving before building writes no engine asset, then builds
    # with InputFunc, saves again and asserts a non-empty engine asset exists. np_input optional
    # pair of arrays for computing the expected output. Output: pair of (converted saved model
    # directory, expected output or nothing).
    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    if np_input is None:
      expected_output = None
    else:
      expected_output = root.run(np_input[0], np_input[1])

    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Run TRT conversion.
    converter = self._CreateConverterV2(input_saved_model_dir)
    converter.convert()

    # Verify the converted GraphDef and ConcreteFunction.
    self._CheckTrtOps(converter._converted_func)  # pylint: disable=protected-access

    trt_engine_name = self._GetUniqueTRTEngineOp(
        converter._converted_graph_def).name

    # Save the converted model without any TRT engine cache.
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    unexpected_asset_file = \
        self._GetAssetFile(output_saved_model_dir, trt_engine_name)
    self.assertFalse(os.path.exists(unexpected_asset_file))

    # Run the converted function to populate the engine cache.
    converter.build(input_fn=InputFunc)

    # Save the converted model again with serialized engine cache.
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    expected_asset_file = \
        self._GetAssetFile(output_saved_model_dir, trt_engine_name)
    self.assertTrue(os.path.exists(expected_asset_file))
    self.assertTrue(os.path.getsize(expected_asset_file))

    del converter
    return output_saved_model_dir, expected_output

  @test_util.run_v2_only
  def testTrtGraphBuild(self):
    # [FUNCTION DOC]
    # Test (TF2 only): checks that engine building accepts input generators yielding inputs in
    # many forms. One-input model: tuple, list, bare array, dictionary, mixed forms and varying
    # batch shapes. Two-input model: tuple, list, dictionary, mixed and varying shapes. Each is
    # passed to the build helpers, which assert the conversion/engine-asset behavior.
    """Testing the construction of a graph with an input data generator

       that takes one or two input parameters passed in different formats.
    """
    # One input parameter:
    np_input = np.random.random_sample([5, 3]).astype(np.float32)

    def _Func_1():
      # [FUNCTION DOC]
      # HELPER: generator yielding the input wrapped in a one-element tuple.
      yield (np_input,)  # tuple with one element

    def _Func_2():
      # [FUNCTION DOC]
      # HELPER: generator yielding the input wrapped in a one-element list.
      yield [np_input]  # list with one element

    def _Func_3():
      # [FUNCTION DOC]
      # HELPER: generator yielding the bare input array.
      yield np_input  # array

    def _Func_4():
      # [FUNCTION DOC]
      # HELPER: generator yielding a dictionary mapping the argument name to the input.
      yield {"x": np_input}  # dictionary

    def _Func_5():
      # [FUNCTION DOC]
      # HELPER: generator yielding the input in four forms (plain, list, array, dictionary).
      # multiple yields: different types
      yield (np_input)  # tuple with one element
      yield [np_input]  # list with one element
      yield np_input  # array
      yield {"x": np_input}  # dictionary

    def _Func_6():
      # [FUNCTION DOC]
      # HELPER: generator yielding random arrays for three different batch sizes.
      # multiple yields: all arrays
      for shape in [(1, 128), (16, 128), (256, 128)]:
        yield np.random.random_sample(shape).astype(np.float32)

    for input_fn in [_Func_1, _Func_2, _Func_3, _Func_4, _Func_5, _Func_6]:
      self._BuildGraphWithInputGenerator(input_fn)

    # Two input parameters:
    np_input1, np_input2 = self._RandomInput([4, 1, 1])

    def _Func_A():
      # [FUNCTION DOC]
      # HELPER: generator yielding the two inputs as a tuple.
      yield np_input1, np_input2  # tuple

    def _Func_B():
      # [FUNCTION DOC]
      # HELPER: generator yielding the two inputs as a list.
      yield [np_input1, np_input2]  # list

    def _Func_C():
      # [FUNCTION DOC]
      # HELPER: generator yielding the two inputs as a dictionary keyed by input names.
      yield {"inp1": np_input1, "inp2": np_input2}  # dictionary

    def _Func_D():
      # [FUNCTION DOC]
      # HELPER: generator yielding the two inputs as a tuple, a list, then a dictionary.
      # multiple yields: different types
      yield np_input1, np_input2  # tuple
      yield [np_input1, np_input2]  # list
      yield {"inp1": np_input1, "inp2": np_input2}  # dictionary

    def _Func_E():
      # [FUNCTION DOC]
      # HELPER: generator yielding pairs of random inputs for three different shapes.
      # multiple yields: tuples
      for shape in [[4, 1, 1], [4, 2, 1], [4, 4, 1]]:
        yield self._RandomInput(shape)

    for input_fn in [_Func_A, _Func_B, _Func_C, _Func_D, _Func_E]:
      self._BuildGraphWithInputGeneratorTwoInputs(input_fn)

  @test_util.run_v2_only
  def testTrtGraphConverter_DynamicConversion_v2(self):
    # [FUNCTION DOC]
    # Test (TF2 only): converts and builds the two-input model, saves it, forces garbage
    # collection, reloads the saved model, asserts the signature has one engine op, runs it and
    # compares to the original model output within tight tolerance.
    """Test case for trt_convert.TrtGraphConverter()."""

    np_input1, np_input2 = self._RandomInput([4, 1, 1])

    def _InputFn():
      # [FUNCTION DOC]
      # HELPER: generator yielding the single pair of test inputs.
      yield np_input1, np_input2

    np_inputs = [np_input1, np_input2]
    output_saved_model_dir, expected_output = \
        self._BuildGraphWithInputGeneratorTwoInputs(_InputFn, np_inputs)
    gc.collect()  # Force GC to destroy the TRT engine cache.

    # Load and verify the converted model.
    #
    # TODO(laigd): the name of the new input_signature of the
    # `root_with_trt.run` function is empty string (originally was None),
    # investigate why.
    root_with_trt = load.load(output_saved_model_dir)
    # TODO(laigd): `root_with_trt.run` is still using the original graph without
    # trt. Consider changing that.
    # self._CheckTrtOps(root_with_trt.run.get_concrete_function())
    converted_signature = root_with_trt.signatures[_SAVED_MODEL_SIGNATURE_KEY]
    self._CheckTrtOps(converted_signature)
    output_with_trt = converted_signature(
        inp1=ops.convert_to_tensor(np_input1),
        inp2=ops.convert_to_tensor(np_input2))
    # The output of running the converted signature is a dict due to
    # compatibility reasons with V1 SavedModel signature mechanism.
    self.assertAllClose(
        expected_output,
        list(output_with_trt.values())[0],
        atol=1e-6,
        rtol=1e-6)

    del root_with_trt
    gc.collect()  # Force GC to destroy the TRT engine cache.

  @test_util.run_v2_only
  def testTrtGraphConverter_ShapeOp_Int32InputOutput_v2(self):
    # [FUNCTION DOC]
    # Test (TF2 only): converts the shape-op model, reloads it, asserts two engine ops, runs it on
    # a random 5 by 3 input and compares with the original model's output.
    """Testing ShapeOp and int32 values as engine input and output."""
    np_input = np.random.random_sample([5, 3]).astype(np.float32)

    def _InputFunc():
      # [FUNCTION DOC]
      # HELPER: generator yielding the input as a one-element tuple.
      # Passing single input parameter as a tuple with one element
      yield (np_input,)

    output_saved_model_dir, expected_output = \
        self._BuildGraphWithInputGenerator(_InputFunc, np_input)

    root_with_trt = load.load(output_saved_model_dir)
    converted_signature = root_with_trt.signatures["serving_default"]
    # Check that the graph is converted to two TRTEngineOps.
    self._CheckTrtOps(converted_signature, num_engines=2)
    # Run the graph.
    output_with_trt = converted_signature(x=ops.convert_to_tensor(np_input))
    # Check the result of the run.
    self.assertAllClose(expected_output, list(output_with_trt.values())[0])

  @test_util.run_v2_only
  def testTrtGraphConverter_Int8Conversion_v2(self):
    # [FUNCTION DOC]
    # Test (TF2 only): converts the two-input model with INT8 precision and calibration data,
    # asserts engines carry calibration data, builds another engine for a different batch size,
    # saves and checks a non-empty engine asset, reloads, verifies the output against the original
    # model and runs a third batch size (building a new engine from the calibration table).

    np_input1, np_input2 = self._RandomInput([4, 1, 1])

    # Create a model and save it.
    input_saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    root = self._GetModelForV2()
    expected_output = root.run(np_input1, np_input2)
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Run TRT conversion.
    converter = self._CreateConverterV2(
        input_saved_model_dir,
        precision_mode=trt_convert.TrtPrecisionMode.INT8,
        maximum_cached_engines=3)

    # Convert and perform INT8 calibration
    def _CalibrationInputFn():
      # [FUNCTION DOC]
      # HELPER: calibration generator yielding the single input pair.
      yield np_input1, np_input2

    converter.convert(calibration_input_fn=_CalibrationInputFn)

    trt_engine_name = self._GetUniqueTRTEngineOp(
        converter._converted_graph_def).name

    def _CheckFn(node):
      # [FUNCTION DOC]
      # HELPER: callback asserting the engine node has non-empty calibration data (the node name
      # is the failure message).
      self.assertTrue(len(node.attr["calibration_data"].s), node.name)

    # Verify the converted GraphDef.
    self._CheckTrtOps(converter._converted_func, _CheckFn)  # pylint: disable=protected-access

    # Build another engine with different batch size.
    def _InputFn():
      # [FUNCTION DOC]
      # HELPER: generator yielding random inputs with batch size 5.
      yield self._RandomInput([5, 1, 1])

    converter.build(input_fn=_InputFn)

    # Save the converted model.
    # TODO(laigd): check that it should contain two engines.
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    expected_asset_file = \
        self._GetAssetFile(output_saved_model_dir, trt_engine_name)
    self.assertTrue(os.path.exists(expected_asset_file))
    self.assertTrue(os.path.getsize(expected_asset_file))

    del converter
    gc.collect()  # Force GC to destroy the TRT engine cache.

    # Load and verify the converted model.
    root_with_trt = load.load(output_saved_model_dir)
    converted_signature = root_with_trt.signatures[_SAVED_MODEL_SIGNATURE_KEY]
    self._CheckTrtOps(converted_signature, _CheckFn)
    output_with_trt = converted_signature(
        inp1=ops.convert_to_tensor(np_input1),
        inp2=ops.convert_to_tensor(np_input2))
    self.assertEqual(1, len(output_with_trt))
    # The output of running the converted signature is a dict due to
    # compatibility reasons with V1 SavedModel signature mechanism.
    self.assertAllClose(
        expected_output,
        list(output_with_trt.values())[0],
        atol=1e-6,
        rtol=1e-6)

    # Run with an input of different batch size. It should build a new engine
    # using calibration table.
    # TODO(laigd): check that it should contain three engines.
    np_input1, np_input2 = self._RandomInput([6, 1, 1])
    converted_signature(
        inp1=ops.convert_to_tensor(np_input1),
        inp2=ops.convert_to_tensor(np_input2))

    del root_with_trt
    gc.collect()  # Force GC to destroy the TRT engine cache.

  @test_util.run_v2_only
  def testTrtGraphConverter_RemoveNativeSegments(self):
    # [FUNCTION DOC]
    # Test (TF2 only): converts the shape-op model with runtime building disallowed and minimum
    # segment size 1, builds engines, saves with the experimental feature for removing native
    # segments active, reloads, runs and compares output with the original model.
    """Test case for trt_convert._remove_native_segment()."""
    np_input = np.random.random_sample([5, 3]).astype(np.float32)
    # Create a model and save it.
    input_saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    root = self._GetShapeOpModel()
    expected_output = root.run(np_input)
    save.save(root, input_saved_model_dir, signatures=root.run)

    # Run TRT conversion.
    converter = trt_convert.TrtGraphConverterV2(
        input_saved_model_dir,
        precision_mode=trt_convert.TrtPrecisionMode.FP32,
        allow_build_at_runtime=False,
        minimum_segment_size=1,
    )

    def _input_fn():
      # [FUNCTION DOC]
      # HELPER: generator yielding the input as a one-element tuple.
      yield (np_input,)

    graph_func = converter.convert()
    converter.build(_input_fn)
    # Load and verify the reduced converted model.
    output_saved_model_dir2 = self.mkdtemp()
    with test_utils.experimental_feature_scope("remove_native_segments"):
      converter.save(output_saved_model_dir2)
    saved_model_loaded = load.load(output_saved_model_dir2)
    graph_func_after = saved_model_loaded.signatures["serving_default"]
    actual_output = graph_func_after(x=np_input)["output_0"]
    self.assertAllClose(expected_output, actual_output, atol=1e-6, rtol=1e-6)
    del graph_func
    del root
    gc.collect()  # Force GC to destroy the TRT engine cache.

  @test_util.run_v2_only
  def testTrtGraphConverter_DestroyEngineCache(self):
    # [FUNCTION DOC]
    # Test (TF2 only): converts, builds and saves the two-input model, asserts destroying the
    # engine resource fails (not found) before loading, then after loading the model the resource
    # exists and can be destroyed once, and after another load plus deletion and garbage
    # collection the resource is gone again (cache destroyed with the model).
    """Test case for trt_convert.TrtGraphConverter()."""

    np_input1, np_input2 = self._RandomInput([4, 1, 1])

    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Run TRT conversion.
    converter = self._CreateConverterV2(input_saved_model_dir)
    converter.convert()

    trt_engine_name = self._GetUniqueTRTEngineOp(
        converter._converted_graph_def).name

    def _InputFn():
      # [FUNCTION DOC]
      # HELPER: generator yielding the single input pair.
      yield np_input1, np_input2

    converter.build(input_fn=_InputFn)  # Populate the TRT engine cache.
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)

    def _DestroyCache():
      # [FUNCTION DOC]
      # HELPER: on the first GPU, creates a handle to the engine resource and destroys it,
      # failing with a not-found error if the resource does not exist.
      with ops.device("GPU:0"):
        handle = gen_trt_ops.create_trt_resource_handle(
            resource_name=trt_engine_name)
        gen_resource_variable_ops.destroy_resource_op(
            handle, ignore_lookup_error=False)

    with self.assertRaisesRegex(errors.NotFoundError,
                                r"Resource .* does not exist."):
      _DestroyCache()

    # Load the converted model and make sure the engine cache is populated by
    # default.
    root = load.load(output_saved_model_dir)
    _DestroyCache()
    with self.assertRaisesRegex(errors.NotFoundError,
                                r"Resource .* does not exist."):
      _DestroyCache()

    # Load the converted model again and make sure the engine cache is destroyed
    # when the model goes out of scope.
    root = load.load(output_saved_model_dir)
    del root
    gc.collect()  # Force GC to destroy the TRT engine cache.
    with self.assertRaisesRegex(errors.NotFoundError,
                                r"Resource .* does not exist."):
      _DestroyCache()

  def _CompareSavedModel(self, model_class):
    # [FUNCTION DOC]
    # HELPER: saves an instance of model_class, converts it, saves the conversion, and asserts
    # the serving signature is retained: same method name, same input and output keys, names,
    # dtypes and shapes, and the same structured outputs after loading.
    signature_key = "serving_default"

    def _GetModelPaths(model_class):
      # [FUNCTION DOC]
      # HELPER: saves the model and its TF-TRT conversion; returns the pair of directories.
      input_saved_model_dir = self.mkdtemp()
      root = model_class()
      save.save(root, input_saved_model_dir)

      converter = self._CreateConverterV2(
          input_saved_model_dir, input_saved_model_signature_key=signature_key)
      converter.convert()
      output_saved_model_dir = self.mkdtemp()
      converter.save(output_saved_model_dir)
      return input_saved_model_dir, output_saved_model_dir

    def _GetSignatureDef(export_dir):
      # [FUNCTION DOC]
      # HELPER: parses the saved model in export_dir, asserts exactly one meta graph containing
      # the serving signature, and returns that signature definition.
      saved_model_proto = loader_impl.parse_saved_model(export_dir)
      self.assertEqual(1, len(saved_model_proto.meta_graphs))
      meta_graph = saved_model_proto.meta_graphs[0]
      self.assertIn(signature_key, meta_graph.signature_def)
      return meta_graph.signature_def[signature_key]

    def _CompareSignatureDef(original_def, converted_def, is_input):
      # [FUNCTION DOC]
      # HELPER: compares inputs (if is_input) or outputs of two signature definitions, asserting
      # equal key sets and, per key, equal tensor name, dtype and shape.
      endpoints = original_def.inputs if is_input else original_def.outputs
      converted_endpoints = (
          converted_def.inputs if is_input else converted_def.outputs)
      self.assertEqual(set(endpoints.keys()), set(converted_endpoints.keys()))
      for key in endpoints:
        original_input = endpoints[key]
        converted_input = converted_endpoints[key]
        self.assertEqual(original_input.name, converted_input.name)
        self.assertEqual(original_input.dtype, converted_input.dtype)
        self.assertEqual(
            tensor_shape.TensorShape(original_input.tensor_shape).as_list(),
            tensor_shape.TensorShape(converted_input.tensor_shape).as_list())

    def _GetStructuredOutputs(export_dir):
      # [FUNCTION DOC]
      # HELPER: loads the saved model and returns the structured outputs of its serving signature.
      root = load.load(export_dir)
      return root.signatures[signature_key].structured_outputs

    saved_model_path, converted_saved_model_path = _GetModelPaths(model_class)
    original_def = _GetSignatureDef(saved_model_path)
    converted_def = _GetSignatureDef(converted_saved_model_path)
    self.assertEqual(original_def.method_name, converted_def.method_name)
    _CompareSignatureDef(original_def, converted_def, True)
    _CompareSignatureDef(original_def, converted_def, False)

    self.assertEqual(
        _GetStructuredOutputs(saved_model_path),
        _GetStructuredOutputs(converted_saved_model_path))

  @test_util.run_v2_only
  def testRetainSignatureInfo_NoInputs(self):
    # [FUNCTION DOC]
    # Test (TF2 only): a model whose function has no inputs and returns a constant keeps its
    # signature information (inputs, outputs, structured outputs) after conversion.

    class _Model(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: trackable model with a no-input function.

      @def_function.function(input_signature=[])
      def run(self):
        # [FUNCTION DOC]
        # Graph function with no inputs returning the constant 1.0.
        return array_ops.constant(1.0)

    self._CompareSavedModel(_Model)

  @test_util.run_v2_only
  def testRetainSignatureInfo_OneInput(self):
    # [FUNCTION DOC]
    # Test (TF2 only): a model with one float input keeps its signature information after
    # conversion.

    class _Model(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: trackable model with a one-input function.

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32)
      ])
      def run(self, inp):
        # [FUNCTION DOC]
        # Graph function: returns inp plus inp squared.
        return inp + inp * inp

    self._CompareSavedModel(_Model)

  @test_util.run_v2_only
  def testRetainSignatureInfo_TwoInputs(self):
    # [FUNCTION DOC]
    # Test (TF2 only): a model with two float inputs of different shapes keeps its signature
    # information after conversion.

    class _Model(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: trackable model with a two-input function.

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32),
          tensor_spec.TensorSpec(shape=[None, 2], dtype=dtypes.float32)
      ])
      def run(self, inp1, inp2):
        # [FUNCTION DOC]
        # Graph function: returns inp1 plus inp2 squared.
        return inp1 + inp2 * inp2

    self._CompareSavedModel(_Model)

  @test_util.run_v2_only
  def testRetainSignatureInfo_OneOutputSignatureKey(self):
    # [FUNCTION DOC]
    # Test (TF2 only): a model returning a dictionary with one named output keeps its output key
    # and signature information after conversion.

    class _Model(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: trackable model with a no-input function returning a one-entry dictionary.

      @def_function.function(input_signature=[])
      def run(self):
        # [FUNCTION DOC]
        # Graph function returning a dictionary with one entry, my_output, holding 1.0.
        return {"my_output": array_ops.constant(1.0)}

    self._CompareSavedModel(_Model)

  @test_util.run_v2_only
  def testRetainSignatureInfo_TwoOutputSignatureKeys(self):
    # [FUNCTION DOC]
    # Test (TF2 only): a model returning a dictionary with two outputs whose keys are not in
    # lexicographic order keeps its output keys and signature information after conversion.

    class _Model(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: trackable model with a one-input function returning a two-entry dictionary.

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32)
      ])
      def run(self, inp):
        # [FUNCTION DOC]
        # Graph function: returns a dictionary with output_b holding 1.0 and output_a holding inp
        # plus inp squared.
        # Here the keys are not ordered lexicographically on purpose.
        return {
            "output_b": array_ops.constant(1.0),
            "output_a": inp + inp * inp
        }

    self._CompareSavedModel(_Model)

  def _TestRun(self, sess, batch_size):
    # [FUNCTION DOC]
    # HELPER: runs the output tensor in session sess feeding ones of the given batch size to both
    # inputs and asserts the result equals 5.0 for every batch element.
    result = sess.run(
        "output:0",
        feed_dict={
            "input1:0": [[[1.0]]] * batch_size,
            "input2:0": [[[1.0]]] * batch_size
        })
    self.assertAllEqual([[[5.0]]] * batch_size, result)

  @parameterized.named_parameters([
      ("LargeSegmentSize", 7),
      ("NoMainGraphConversionSegmentSize", -1),
  ])
  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_MinimumSegmentSize(self, minimum_segment_size):
    # [FUNCTION DOC]
    # Test (graph mode, parameterized with a too-large segment size 7 and -1 meaning no main graph
    # conversion): converts the test graph and asserts no engine op is created; the node-to-op
    # mapping stays that of the original frozen graph.
    output_graph_def = self._ConvertGraphV1(
        minimum_segment_size=minimum_segment_size)
    node_name_to_op = {node.name: node.op for node in output_graph_def.node}
    self.assertEqual(
        {
            "v1": "Const",
            "input1": "Placeholder",
            "input2": "Placeholder",
            "add": "AddV2",
            "mul": "Mul",
            "add_1": "AddV2",
            "add_2": "AddV2",
            "output": "Identity"
        }, node_name_to_op)

  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_DynamicOp(self):
    # [FUNCTION DOC]
    # Test (graph mode): converts with dynamic ops and at most two cached engines, then runs the
    # output graph definition and the loaded saved model with batch sizes 1, 2 and 3 (the third
    # evicts an old engine), asserting correct results each time.

    output_saved_model_dir = self.mkdtemp()
    output_graph_def = self._ConvertGraphV1(
        output_saved_model_dir=output_saved_model_dir,
        is_dynamic_op=True,
        maximum_cached_engines=2)

    # Test the output GraphDef.
    with ops.Graph().as_default():
      importer.import_graph_def(output_graph_def, name="")
      with self.session(config=self._GetConfigProto()) as sess:
        # Run with batch size 1, a new engine is created and cached.
        self._TestRun(sess, 1)
        # Run with batch size 2, a new engine is created and cached.
        self._TestRun(sess, 2)
        # Run with batch size 3, since the number of cached engines has reached
        # the max, it should evict an old engine and create a new one.
        self._TestRun(sess, 3)

    # Test the output SavedModel
    with ops.Graph().as_default():
      with self.session(config=self._GetConfigProto()) as sess:
        loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
        # Run with batch size 1, a new engine is created and cached.
        self._TestRun(sess, 1)
        # Run with batch size 2, a new engine is created and cached.
        self._TestRun(sess, 2)
        # Run with batch size 3, since the number of cached engines has reached
        # the max, it should evict an old engine and create a new one.
        self._TestRun(sess, 3)

  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_StaticOp(self):
    # [FUNCTION DOC]
    # Test (graph mode): converts with static ops (one cached engine), then runs the output graph
    # definition and loaded saved model with batch size 1 (embedded engine) and 2 (exceeds the
    # maximum batch size, falling back to the native function), asserting correct results.

    output_saved_model_dir = self.mkdtemp()
    output_graph_def = self._ConvertGraphV1(
        output_saved_model_dir=output_saved_model_dir, maximum_cached_engines=1)

    # Test the output GraphDef.
    with ops.Graph().as_default():
      importer.import_graph_def(output_graph_def, name="")
      with self.session(config=self._GetConfigProto()) as sess:
        # Run with batch size 1, the default engine embedded in the graphdef
        # will be used.
        self._TestRun(sess, 1)
        # Run with batch size 2, which exceed the max_batch_size, it should try
        # to fall back to TF function.
        self._TestRun(sess, 2)

    # Test the output SavedModel
    with ops.Graph().as_default():
      with self.session(config=self._GetConfigProto()) as sess:
        loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
        # Run with batch size 1, the default engine embedded in the graphdef
        # will be used.
        self._TestRun(sess, 1)
        # Run with batch size 2, which exceed the max_batch_size, it should try
        # to fall back to TF function.
        self._TestRun(sess, 2)

  @test_util.run_v2_only
  def testTrtGraphConverter_AllowEngineNativeSegmentExecution(self):
    # [FUNCTION DOC]
    # Test (TF2 only): converts the two-input model, sets environment variables that disallow
    # native segment execution and force aborting engine build, and asserts building raises an
    # aborted error mentioning the disallowed execution; the variables are always reset in a
    # finally step (side effect on the process environment). Then building succeeds normally.
    np_input1, np_input2 = self._RandomInput([4, 1, 1])

    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    def _InputFn():
      # [FUNCTION DOC]
      # HELPER: generator yielding the single input pair.
      yield np_input1, np_input2

    # Run TRT conversion
    converter = self._CreateConverterV2(
        input_saved_model_dir, max_workspace_size_bytes=1 << 20)
    converter.convert()

    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "False"
    os.environ["TF_TRT_ABORT_CUDA_ENGINE_BUILD"] = "True"
    with self.assertRaisesRegex(
        errors.AbortedError,
        r"User disallowed engine native segment execution"):
      try:
        converter.build(input_fn=_InputFn)
      finally:
        # Always reset the environment variable.
        os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"
        os.environ["TF_TRT_ABORT_CUDA_ENGINE_BUILD"] = "False"

    converter.build(input_fn=_InputFn)

  @parameterized.parameters((True, True), (True, False), (False, True),
                            (False, False))
  @test_util.run_v2_only
  def testTrtGraphConverter_AllowBuildAtRuntime(self, build_offline,
                                                allow_build_at_runtime):
    # [FUNCTION DOC]
    # Test (TF2 only, parameterized over build_offline and allow_build_at_runtime): does nothing
    # if TensorRT is not enabled. Converts the two-input model, optionally builds engines offline,
    # saves and reloads it, asserts every engine op carries the allow-build-at-runtime attribute
    # equal to the setting. If not built offline and runtime build is disallowed, running (with
    # native segment execution disallowed through the environment, reset afterwards) must raise
    # an aborted error; otherwise the output must have shape 4 by 1 by 1 with all values 5.
    if not is_tensorrt_enabled():
      return

    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))

    def _InputFn():
      # [FUNCTION DOC]
      # HELPER: generator yielding the single input pair of tensors.
      yield np_input1, np_input2

    # Run TRT conversion and request an unreasonably large workspace.
    converter = self._CreateConverterV2(
        input_saved_model_dir, allow_build_at_runtime=allow_build_at_runtime)
    converter.convert()
    if build_offline:
      converter.build(input_fn=_InputFn)
    # Output saved model dir.
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)

    saved_model_loaded = load.load(
        output_saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[_SAVED_MODEL_SIGNATURE_KEY]

    # Checks the TrtEngineOp(s) have the correct attribute(s).
    def _CheckFn(node):
      # [FUNCTION DOC]
      # HELPER: callback asserting the engine node's allow-build-at-runtime attribute equals the
      # parameterized setting.
      self.assertEqual(node.attr["_allow_build_at_runtime"].b,
                       allow_build_at_runtime)

    self._CheckTrtOps(graph_func, _CheckFn)
    # If the engine was not build offline and the user set not to build at
    # runtime and not to run native segments. Then, it will report an error.
    if not build_offline and not allow_build_at_runtime:
      with self.assertRaisesRegex(
          errors.AbortedError,
          r"User disallowed engine native segment execution"):
        try:
          os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "False"
          graph_func(inp1=np_input1, inp2=np_input2)
        finally:
          os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"
    else:
      output = graph_func(inp1=np_input1, inp2=np_input2)["output_0"]
      self.assertEqual(output.shape, (4, 1, 1))
      self.assertAllClose(
          np.asarray([5.0, 5.0, 5.0, 5.0]).reshape([4, 1, 1]), output)

  @test_util.run_v2_only
  def testBackwardCompatibility(self):
    # [FUNCTION DOC]
    # Test (TF2 only): loads a checked-in TF-TRT model saved with TF 2.0, runs its serving
    # signature on ones of shape 4 by 1 by 1 and asserts the output has that shape with all
    # values 5.
    """Load and execute a model that was saved in TF2.0."""

    model_dir = test.test_src_dir_path(
        "python/compiler/tensorrt/test/testdata/tftrt_2.0_saved_model")
    saved_model_loaded = load.load(model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    output = graph_func(input1=np_input1, input2=np_input2)["output_0"]

    self.assertEqual(output.shape, (4, 1, 1))
    self.assertAllClose(
        np.asarray([5.0, 5.0, 5.0, 5.0]).reshape([4, 1, 1]), output)

  @parameterized.named_parameters([
      ("SaveGPUSpecificEngine", True),
      ("WithoutSaveGPUSpecificEngine", False),
  ])
  @test_util.run_v2_only
  def testTrtGraphConverter_SaveGPUSpecificEngine(self, save_engine_flag):
    # [FUNCTION DOC]
    # Test (TF2 only, parameterized by save_engine_flag): INT8-converts the two-input model with
    # calibration, saves it with GPU-specific engines saved or not per the flag, and asserts the
    # engine asset file exists, being non-empty if saved and empty otherwise.
    """Test case for trt_convert.TrtGraphConverter()."""

    np_input1, np_input2 = self._RandomInput([4, 1, 1])

    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Run TRT conversion.
    converter = self._CreateConverterV2(
        input_saved_model_dir, precision_mode=trt_convert.TrtPrecisionMode.INT8)

    # Run the converted function to populate the engine cache.
    def CalibrationFn():
      # [FUNCTION DOC]
      # HELPER: calibration generator yielding the single input pair.
      yield np_input1, np_input2

    converter.convert(calibration_input_fn=CalibrationFn)

    # Verify the converted GraphDef and ConcreteFunction.
    self._CheckTrtOps(converter._converted_func)

    trt_engine_name = self._GetUniqueTRTEngineOp(
        converter._converted_graph_def).name

    # Save the converted model with or without any TRT engine cache
    # based on the value of save_engine_flag.
    output_saved_model_dir = self.mkdtemp()

    converter.save(
        output_saved_model_dir, save_gpu_specific_engines=save_engine_flag)

    expected_asset_file = \
        self._GetAssetFile(output_saved_model_dir, trt_engine_name)

    self.assertTrue(os.path.exists(expected_asset_file))
    if save_engine_flag:
      # engine is saved so we expect engine data
      self.assertTrue(os.path.getsize(expected_asset_file))
    else:
      # engine is not saved so files should be empty
      self.assertFalse(os.path.getsize(expected_asset_file))

    del converter
    gc.collect()  # Force GC to destroy the TRT engine cache.

  @test_util.run_v2_only
  def testTrtGraphConverterV2_SaveWithOptions(self):
    # [FUNCTION DOC]
    # Test (TF2 only): converts the two-input model, replaces the saved-model save function in the
    # conversion module with a mock, saves the converter output with explicit save options and
    # asserts the mock was called exactly once with the given options as keyword argument.
    """Test to make sure that save method respects options kwarg."""

    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Run TRT conversion.
    converter = self._CreateConverterV2(input_saved_model_dir)
    converter.convert()

    # Patch save function with mock.
    with mock.patch.object(trt_convert, "save") as mock_save:
      mock_save.save = mock.MagicMock()
      # Save converted model with options.
      output_saved_model_dir = self.mkdtemp()
      options = save_options.SaveOptions(save_debug_info=True)
      converter.save(output_saved_model_dir, options=options)

      # Assert that the saved_model.save function was called with the given
      # save_options by TrtGraphConverterV2.save method.
      mock_save.save.assert_called_once_with(
          mock.ANY, mock.ANY, mock.ANY, options=options)

  @parameterized.named_parameters([
      ("NoDeviceAssignment", None),
      ("GPU1", "GPU:1"),
  ])
  @test_util.run_v2_only
  def testTrtGraphConverter_DevicePlacement(self, device_id):
    # [FUNCTION DOC]
    # Test (TF2 only, parameterized: no device or the second GPU): skipped if fewer than two GPUs
    # exist. Converts the model inside a device scope and asserts the single engine op is placed
    # on the requested device (default first GPU), compared case-insensitively.
    """Test case for trt_convert.TrtGraphConverter()."""

    gpus = config.list_physical_devices("GPU")
    if len(gpus) < 2:
      self.skipTest("Expected at least 2 GPUs but found {} GPUs".format(
          len(gpus)))

    np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))

    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    converter = self._CreateConverterV2(
        input_saved_model_dir, precision_mode=trt_convert.TrtPrecisionMode.FP32)

    converted_model = None
    # Specify device on which converted model should be placed
    with ops.device(device_id):
      converted_model = converter.convert()

    # Verify that TRT engine op has the correct device.
    self._CheckTrtOps(converter._converted_func)

    actual_device_id = self._GetUniqueTRTEngineOp(
        converter._converted_graph_def).device

    expected_device_id = None
    if device_id is not None:
      expected_device_id = device_id
    else:
      expected_device_id = "GPU:0"

    self.assertTrue(expected_device_id.lower() in actual_device_id.lower())

    del converter
    gc.collect()  # Force GC to destroy the TRT engine cache.

  @test_util.run_v2_only
  def testTrtGraphConverter_DevicePlacementOnCPU(self):
    # [FUNCTION DOC]
    # Test (TF2 only): converting inside a CPU device scope must raise a value error stating the
    # specified device is not a GPU.
    """Test case for trt_convert.TrtGraphConverter()."""

    np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))

    # Create a model and save it.
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})

    # Run TRT conversion.
    converter = self._CreateConverterV2(
        input_saved_model_dir, precision_mode=trt_convert.TrtPrecisionMode.FP32)

    converted_model = None
    # Specify device on which converted model should be placed
    with self.assertRaisesRegex(ValueError, r"Specified device is not a GPU"):
      with ops.device("CPU"):
        converted_model = converter.convert()

    del converter
    gc.collect()  # Force GC to destroy the TRT engine cache.

  def _TestVariableHelper(self, variable_op, tf_model_name, tftrt_model_name,
                          output_name):
    # [FUNCTION DOC]
    # HELPER: shared body for variable conversion tests. Loads a checked-in TF model
    # (tf_model_name), converts it with FP16, dynamic shapes and graph freezing disabled through
    # an experimental feature scope, builds with one pair of 4 by 1 by 1 inputs, saves under
    # tftrt_model_name, reloads and asserts there is exactly one native segment containing two
    # nodes of type variable_op; runs with inputs 1 and 2 and asserts the output named output_name
    # is 42 everywhere with shape 4 by 1 by 1.
    """Helper with the common code of variable converter tests."""

    model_dir = test.test_src_dir_path(
        "python/compiler/tensorrt/test/testdata/" + tf_model_name)
    trt_model_dir = os.path.join(self.mkdtemp(), tftrt_model_name)

    # Load and convert the TF model.
    conv_params = trt_convert.TrtConversionParams(
        precision_mode="FP16",
        minimum_segment_size=3,
        max_workspace_size_bytes=10 << 20,
        maximum_cached_engines=1)
    with test_utils.experimental_feature_scope("disable_graph_freezing"):
      converter = trt_convert.TrtGraphConverterV2(
          input_saved_model_dir=model_dir,
          conversion_params=conv_params,
          use_dynamic_shape=True,
          dynamic_shape_profile_strategy="Optimal")
    converter.convert()

    # Build and save the converted model.
    input_shapes = [[(4, 1, 1), (4, 1, 1)]]

    def _InputFn():
      # [FUNCTION DOC]
      # HELPER: generator yielding, for each configured pair of shapes, a list of ones-filled
      # float arrays of those shapes.
      for shapes in input_shapes:
        # return a list of input tensors
        yield [np.ones(shape=shape).astype(np.float32) for shape in shapes]

    converter.build(_InputFn)
    converter.save(trt_model_dir)

    # Load the converted model.
    saved_model_loaded = load.load(trt_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    # Check that there is one segment and that the 2 variables are in it.
    graph_def = graph_func.graph.as_graph_def()
    engines = []
    for lib_function in graph_def.library.function:
      if re.search(r"TRTEngineOp_\d+_\d+_native_segment",
                   lib_function.signature.name):
        node_ops = [node.op for node in lib_function.node_def]
        engines.append(node_ops)
    self.assertLen(engines, 1)
    self.assertEqual(engines[0].count(variable_op), 2)

    # Run the function and check the output.
    np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    np_input2 = ops.convert_to_tensor(2. *
                                      np.ones([4, 1, 1]).astype(np.float32))
    output = graph_func(input1=np_input1, input2=np_input2)[output_name]
    self.assertEqual(output.shape, (4, 1, 1))
    self.assertAllClose(
        np.asarray([42., 42., 42., 42.]).reshape([4, 1, 1]), output)

  @test_util.run_v2_only
  def testVariableV2(self):
    # [FUNCTION DOC]
    # Test (TF2 only): conversion of a model using legacy variable nodes via the shared variable
    # helper; expects output named output.
    """Test conversion of VariableV2 nodes."""

    self._TestVariableHelper("VariableV2", "tf_variablev2_saved_model",
                             "tftrt_variablev2_saved_model", "output")

  @test_util.run_v2_only
  def testReadVariableOp(self):
    # [FUNCTION DOC]
    # Test (TF2 only): conversion of a model using resource-variable read nodes via the shared
    # variable helper; expects output named output_0.
    """Test conversion of ReadVariableOp nodes."""

    self._TestVariableHelper("ReadVariableOp", "tf_readvariableop_saved_model",
                             "tftrt_readvariableop_saved_model", "output_0")

if __name__ == "__main__" and is_tensorrt_enabled():
  test.main()
