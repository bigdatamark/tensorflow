# [MODULE DOC]
# Tests for the tflite_convert command line tool and its argument parser.
# Defines a shared base test class with helpers that run the converter binary as a subprocess, plus
# V1 and V2 converter test classes (frozen graphs, saved models, Keras files, object detection
# models, custom op definitions, flex ops) and a parser test class checking the experimental
# new-converter and new-quantizer flags. Has no import-time side effects; runs the test main when
# executed directly.
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
# ==============================================================================
"""Tests for tflite_convert.py."""

import os

from absl.testing import parameterized
import numpy as np
from tensorflow import keras

from tensorflow.core.framework import graph_pb2
from tensorflow.lite.python import test_util as tflite_test_util
from tensorflow.lite.python import tflite_convert
from tensorflow.lite.python.convert import register_custom_opdefs
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.saved_model.save import save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training.training_util import write_graph


class TestModels(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the converter tests below.
  # Not a test itself (it defines no test methods); provides helpers to build temp file paths, run
  # the tflite_convert binary with a flag string, and create Keras model files.

  def _getFilepath(self, filename):
    # [FUNCTION DOC]
    # HELPER: builds paths for the test helpers.
    # Returns the given file name joined to the test's temporary directory.
    return os.path.join(self.get_temp_dir(), filename)

  def _run(self,
           flags_str,
           should_succeed,
           expected_ops_in_converted_model=None,
           expected_output_shapes=None):
    # [FUNCTION DOC]
    # HELPER: runs the converter binary for the conversion tests.
    # Inputs: flags_str is the command line flags text; should_succeed is whether the conversion
    # is expected to work; expected_ops_in_converted_model is an optional list of op names that
    # must appear in the output model; expected_output_shapes is an optional list of shapes the
    # output model must have. Returns nothing.
    # Runs the tflite_convert binary through the shell, writing to a model file in the temp dir.
    # If the exit code is zero, reads the file, asserts that content exists equals should_succeed,
    # checks the ops and output shapes if given, then deletes the file. If the exit code is
    # nonzero, asserts that should_succeed is false.
    # Side effects: spawns a process and creates/removes a file.
    output_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    tflite_bin = resource_loader.get_path_to_datafile('tflite_convert.par')
    cmdline = '{0} --output_file={1} {2}'.format(tflite_bin, output_file,
                                                 flags_str)

    exitcode = os.system(cmdline)
    if exitcode == 0:
      with gfile.Open(output_file, 'rb') as model_file:
        content = model_file.read()
      self.assertEqual(content is not None, should_succeed)
      if expected_ops_in_converted_model:
        op_set = tflite_test_util.get_ops_list(content)
        for opname in expected_ops_in_converted_model:
          self.assertIn(opname, op_set)
      if expected_output_shapes:
        output_shapes = tflite_test_util.get_output_shapes(content)
        self.assertEqual(output_shapes, expected_output_shapes)
      os.remove(output_file)
    else:
      self.assertFalse(should_succeed)

  def _getKerasModelFile(self):
    # [FUNCTION DOC]
    # HELPER: creates the Keras input file for the Keras conversion tests.
    # Builds a small sequential model (dropout then one dense unit), compiles it with sgd and mean
    # squared error, trains one epoch on two points, saves it as an h5 file in the temp dir and
    # returns the file path. Side effects: training, file creation.
    x = np.array([[1.], [2.]])
    y = np.array([[2.], [4.]])

    model = keras.models.Sequential([
        keras.layers.Dropout(0.2, input_shape=(1,)),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)

    keras_file = self._getFilepath('model.h5')
    keras.models.save_model(model, keras_file)
    return keras_file

  def _getKerasFunctionalModelFile(self):
    # [FUNCTION DOC]
    # HELPER: creates the functional Keras input file for the functional model tests.
    # Builds an untrained functional model with one input and two dense outputs (named b with 1
    # unit and a with 2 units), saves it as an h5 file in the temp dir and returns the path.
    """Returns a functional Keras model with output shapes [[1, 1], [1, 2]]."""
    input_tensor = keras.layers.Input(shape=(1,))
    output1 = keras.layers.Dense(1, name='b')(input_tensor)
    output2 = keras.layers.Dense(2, name='a')(input_tensor)
    model = keras.models.Model(inputs=input_tensor, outputs=[output1, output2])

    keras_file = self._getFilepath('functional_model.h5')
    keras.models.save_model(model, keras_file)
    return keras_file


class TfLiteConvertV1Test(TestModels):
  # [CLASS DOC]
  # Tests the command line converter with V1 style inputs: frozen graph definitions, QAT graphs,
  # saved models (including custom op definitions and flex ops), Keras files and an object
  # detection model. Extends the base class and overrides the run helper to add the V1 converter
  # flag when TF2 behavior is enabled.

  def _run(self,
           flags_str,
           should_succeed,
           expected_ops_in_converted_model=None):
    # [FUNCTION DOC]
    # HELPER: V1 override of the base run helper.
    # If TF2 behavior is enabled, appends the enable-V1-converter flag to the flags text, then
    # delegates to the base helper (without passing expected output shapes). Returns nothing.
    if tf2.enabled():
      flags_str += ' --enable_v1_converter'
    super(TfLiteConvertV1Test, self)._run(flags_str, should_succeed,
                                          expected_ops_in_converted_model)

  def testFrozenGraphDef(self):
    # [FUNCTION DOC]
    # Scenario: a graph with a float placeholder added to itself is written to a frozen graph file.
    # Action: converts it with input array Placeholder and output array add.
    # Expected: the conversion succeeds. The graph file is removed afterwards.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = self._getFilepath('model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    flags_str = ('--graph_def_file={0} --input_arrays={1} '
                 '--output_arrays={2}'.format(graph_def_file, 'Placeholder',
                                              'add'))
    self._run(flags_str, should_succeed=True)
    os.remove(graph_def_file)

  # Run `tflite_convert` explicitly with the legacy converter.
  # Before the new converter is enabled by default, this flag has no real
  # effects.
  def testFrozenGraphDefWithLegacyConverter(self):
    # [FUNCTION DOC]
    # Same scenario as the basic frozen graph test (placeholder added to itself, saved to file),
    # but the flags also disable the experimental new converter to force the legacy converter.
    # Expected: the conversion succeeds.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = self._getFilepath('model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    flags_str = (
        '--graph_def_file={0} --input_arrays={1} '
        '--output_arrays={2} --experimental_new_converter=false'.format(
            graph_def_file, 'Placeholder', 'add'))
    self._run(flags_str, should_succeed=True)
    os.remove(graph_def_file)

  def testFrozenGraphDefNonPlaceholder(self):
    # [FUNCTION DOC]
    # Scenario: the graph input is a random normal op (not a placeholder), doubled by an add.
    # Action: converts the frozen graph with the random op as input array and add as output.
    # Expected: the conversion succeeds.
    with ops.Graph().as_default():
      in_tensor = random_ops.random_normal(shape=[1, 16, 16, 3], name='random')
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = self._getFilepath('model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    flags_str = ('--graph_def_file={0} --input_arrays={1} '
                 '--output_arrays={2}'.format(graph_def_file, 'random', 'add'))
    self._run(flags_str, should_succeed=True)
    os.remove(graph_def_file)

  def testQATFrozenGraphDefInt8(self):
    # [FUNCTION DOC]
    # Scenario: two float placeholders are summed and passed through a 16-bit fake quantization op
    # with range 0 to 1, saved as a frozen graph.
    # Action: converts with INT8 inference type, mean and standard deviation values of 128 for both
    # inputs. Expected: the conversion succeeds.
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
      _ = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0., max=1., name='output',
          num_bits=16)  # INT8 inference type works for 16 bits fake quant.
      sess = session.Session()

    # Write graph to file.
    graph_def_file = self._getFilepath('model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    flags_str = ('--inference_type=INT8 --std_dev_values=128,128 '
                 '--mean_values=128,128 '
                 '--graph_def_file={0} --input_arrays={1},{2} '
                 '--output_arrays={3}'.format(graph_def_file, 'inputA',
                                              'inputB', 'output'))
    self._run(flags_str, should_succeed=True)
    os.remove(graph_def_file)

  def testQATFrozenGraphDefUInt8(self):
    # [FUNCTION DOC]
    # Scenario: two placeholders summed and passed through a default fake quantization op, saved as
    # a frozen graph. Action: converts with UINT8 inference type (default float-agnostic input
    # type), then again with the inference input type set to FLOAT. Expected: both succeed.
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
      _ = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0., max=1., name='output')
      sess = session.Session()

    # Write graph to file.
    graph_def_file = self._getFilepath('model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Define converter flags
    flags_str = ('--std_dev_values=128,128 --mean_values=128,128 '
                 '--graph_def_file={0} --input_arrays={1} '
                 '--output_arrays={2}'.format(graph_def_file, 'inputA,inputB',
                                              'output'))

    # Set inference_type UINT8 and (default) inference_input_type UINT8
    flags_str_1 = flags_str + ' --inference_type=UINT8'
    self._run(flags_str_1, should_succeed=True)

    # Set inference_type UINT8 and inference_input_type FLOAT
    flags_str_2 = flags_str_1 + ' --inference_input_type=FLOAT'
    self._run(flags_str_2, should_succeed=True)

    os.remove(graph_def_file)

  def testSavedModel(self):
    # [FUNCTION DOC]
    # Scenario: a graph-mode saved model is written with a simple save of a placeholder doubled by
    # addition. Action: converts it via the saved model directory flag. Expected: success.
    saved_model_dir = self._getFilepath('model')
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor = array_ops.placeholder(
            shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
        out_tensor = in_tensor + in_tensor
        inputs = {'x': in_tensor}
        outputs = {'z': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

    flags_str = '--saved_model_dir={}'.format(saved_model_dir)
    self._run(flags_str, should_succeed=True)

  def _createSavedModelWithCustomOp(self, opname='CustomAdd'):
    # [FUNCTION DOC]
    # HELPER: builds the saved model for the custom op definition tests.
    # Input: opname, the name for the custom op (default CustomAdd). Builds a text op definition
    # with two float inputs and one float output, makes a graph with one add, renames the add node
    # op to opname and removes its type attribute, registers the op definition globally, imports
    # the modified graph and saves it as a saved model in the temp dir.
    # Returns a pair: saved model directory and the op definition text.
    # Side effects: registers custom op definitions, writes files.
    custom_opdefs_str = (
        'name: \'' + opname + '\' input_arg: {name: \'Input1\' type: DT_FLOAT} '
        'input_arg: {name: \'Input2\' type: DT_FLOAT} output_arg: {name: '
        '\'Output\' type: DT_FLOAT}')

    # Create a graph that has one add op.
    new_graph = graph_pb2.GraphDef()
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor = array_ops.placeholder(
            shape=[1, 16, 16, 3], dtype=dtypes.float32, name='input')
        out_tensor = in_tensor + in_tensor
        inputs = {'x': in_tensor}
        outputs = {'z': out_tensor}

        new_graph.CopyFrom(sess.graph_def)

    # Rename Add op name to opname.
    for node in new_graph.node:
      if node.op.startswith('Add'):
        node.op = opname
        del node.attr['T']

    # Register custom op defs to import modified graph def.
    register_custom_opdefs([custom_opdefs_str])

    # Store saved model.
    saved_model_dir = self._getFilepath('model')
    with ops.Graph().as_default():
      with session.Session() as sess:
        import_graph_def(new_graph, name='')
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return (saved_model_dir, custom_opdefs_str)

  def testEnsureCustomOpdefsFlag(self):
    # [FUNCTION DOC]
    # Scenario: saved model containing a custom op. Action: converts with allow-custom-ops and the
    # new converter but without supplying the custom op definitions. Expected: conversion fails.
    saved_model_dir, _ = self._createSavedModelWithCustomOp()

    # Ensure --custom_opdefs.
    flags_str = ('--saved_model_dir={0} --allow_custom_ops '
                 '--experimental_new_converter'.format(saved_model_dir))
    self._run(flags_str, should_succeed=False)

  def testSavedModelWithCustomOpdefsFlag(self):
    # [FUNCTION DOC]
    # Scenario: saved model with a custom add op. Action: converts with the custom op definitions
    # flag, allow-custom-ops and the new converter. Expected: success and the output model
    # contains the CustomAdd op.
    saved_model_dir, custom_opdefs_str = self._createSavedModelWithCustomOp()

    # Valid conversion.
    flags_str = (
        '--saved_model_dir={0} --custom_opdefs="{1}" --allow_custom_ops '
        '--experimental_new_converter'.format(saved_model_dir,
                                              custom_opdefs_str))
    self._run(
        flags_str,
        should_succeed=True,
        expected_ops_in_converted_model=['CustomAdd'])

  def testSavedModelWithFlex(self):
    # [FUNCTION DOC]
    # Scenario: saved model with a custom op named CustomAdd2 (already registered). Action:
    # converts with custom op definitions, the new converter, CustomAdd2 as a user select-TF op and
    # target ops of builtins plus select TF ops. Expected: success and the output model contains
    # the FlexCustomAdd2 op.
    saved_model_dir, custom_opdefs_str = self._createSavedModelWithCustomOp(
        opname='CustomAdd2')

    # Valid conversion. OpDef already registered.
    flags_str = ('--saved_model_dir={0} --allow_custom_ops '
                 '--custom_opdefs="{1}" '
                 '--experimental_new_converter '
                 '--experimental_select_user_tf_ops=CustomAdd2 '
                 '--target_ops=TFLITE_BUILTINS,SELECT_TF_OPS'.format(
                     saved_model_dir, custom_opdefs_str))
    self._run(
        flags_str,
        should_succeed=True,
        expected_ops_in_converted_model=['FlexCustomAdd2'])

  def testSavedModelWithInvalidCustomOpdefsFlag(self):
    # [FUNCTION DOC]
    # Scenario: saved model with a custom add op, but the supplied op definition lacks the second
    # input argument and so does not match. Action: converts with that definition. Expected:
    # conversion fails. (A code comment says "valid conversion", which is misleading.)
    saved_model_dir, _ = self._createSavedModelWithCustomOp()

    invalid_custom_opdefs_str = (
        'name: \'CustomAdd\' input_arg: {name: \'Input1\' type: DT_FLOAT} '
        'output_arg: {name: \'Output\' type: DT_FLOAT}')

    # Valid conversion.
    flags_str = (
        '--saved_model_dir={0} --custom_opdefs="{1}" --allow_custom_ops '
        '--experimental_new_converter'.format(saved_model_dir,
                                              invalid_custom_opdefs_str))
    self._run(flags_str, should_succeed=False)

  def testKerasFile(self):
    # [FUNCTION DOC]
    # Scenario: a saved sequential Keras h5 model. Action: converts via the Keras model file flag.
    # Expected: success; the model file is removed afterwards.
    keras_file = self._getKerasModelFile()

    flags_str = '--keras_model_file={}'.format(keras_file)
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)

  def testKerasFileMLIR(self):
    # [FUNCTION DOC]
    # Same as the Keras file test but also passes the experimental new converter flag.
    # Expected: success.
    keras_file = self._getKerasModelFile()

    flags_str = (
        '--keras_model_file={} --experimental_new_converter'.format(keras_file))
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)

  def _initObjectDetectionArgs(self):
    # [FUNCTION DOC]
    # HELPER: prepares inputs for the object detection tests.
    # Locates the test graph file (first the bundled test data, else a sibling mobilenet ssd
    # directory); raises an IO error if neither exists. Stores on the instance the graph file path,
    # the input array name, the comma separated four output array names and the input shape text.
    # Initializes the arguments required for the object detection model.
    # Looks for the model file which is saved in a different location internally
    # and externally.
    filename = resource_loader.get_path_to_datafile('testdata/tflite_graph.pb')
    if not os.path.exists(filename):
      filename = os.path.join(
          resource_loader.get_root_dir_with_all_resources(),
          '../tflite_mobilenet_ssd_quant_protobuf/tflite_graph.pb')
      if not os.path.exists(filename):
        raise IOError("File '{0}' does not exist.".format(filename))

    self._graph_def_file = filename
    self._input_arrays = 'normalized_input_image_tensor'
    self._output_arrays = (
        'TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,'
        'TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3')
    self._input_shapes = '1,300,300,3'

  def testObjectDetection(self):
    # [FUNCTION DOC]
    # Converts the object detection frozen graph with input shapes and allow-custom-ops (legacy
    # TOCO path per its docstring). Expected: success.
    """Tests object detection model through TOCO."""
    self._initObjectDetectionArgs()
    flags_str = ('--graph_def_file={0} --input_arrays={1} '
                 '--output_arrays={2} --input_shapes={3} '
                 '--allow_custom_ops'.format(self._graph_def_file,
                                             self._input_arrays,
                                             self._output_arrays,
                                             self._input_shapes))
    self._run(flags_str, should_succeed=True)

  def testObjectDetectionMLIR(self):
    # [FUNCTION DOC]
    # Converts the object detection graph supplying custom op definitions for the detection
    # post-process op and allow-custom-ops. Expected: success and the output contains the
    # TFLite_Detection_PostProcess op.
    """Tests object detection model through MLIR converter."""
    self._initObjectDetectionArgs()
    custom_opdefs_str = (
        'name: \'TFLite_Detection_PostProcess\' '
        'input_arg: { name: \'raw_outputs/box_encodings\' type: DT_FLOAT } '
        'input_arg: { name: \'raw_outputs/class_predictions\' type: DT_FLOAT } '
        'input_arg: { name: \'anchors\' type: DT_FLOAT } '
        'output_arg: { name: \'TFLite_Detection_PostProcess\' type: DT_FLOAT } '
        'output_arg: { name: \'TFLite_Detection_PostProcess:1\' '
        'type: DT_FLOAT } '
        'output_arg: { name: \'TFLite_Detection_PostProcess:2\' '
        'type: DT_FLOAT } '
        'output_arg: { name: \'TFLite_Detection_PostProcess:3\' '
        'type: DT_FLOAT } '
        'attr : { name: \'h_scale\' type: \'float\'} '
        'attr : { name: \'max_classes_per_detection\' type: \'int\'} '
        'attr : { name: \'max_detections\' type: \'int\'} '
        'attr : { name: \'nms_iou_threshold\' type: \'float\'} '
        'attr : { name: \'nms_score_threshold\' type: \'float\'} '
        'attr : { name: \'num_classes\' type: \'int\'} '
        'attr : { name: \'w_scale\' type: \'float\'} '
        'attr : { name: \'x_scale\' type: \'float\'} '
        'attr : { name: \'y_scale\' type: \'float\'}')

    flags_str = ('--graph_def_file={0} --input_arrays={1} '
                 '--output_arrays={2} --input_shapes={3} '
                 '--custom_opdefs="{4}"'.format(self._graph_def_file,
                                                self._input_arrays,
                                                self._output_arrays,
                                                self._input_shapes,
                                                custom_opdefs_str))

    # Ensure --allow_custom_ops.
    flags_str_final = ('{} --allow_custom_ops').format(flags_str)

    self._run(
        flags_str_final,
        should_succeed=True,
        expected_ops_in_converted_model=['TFLite_Detection_PostProcess'])

  def testObjectDetectionMLIRWithFlex(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is copied from the previous test; this test uses flex (select TF
    # ops for the detection post-process op) rather than custom op definitions.
    # Converts the object detection graph with allow-custom-ops, the new converter, the user
    # select-TF-op and target ops of builtins plus select TF ops. Expected: success and the output
    # contains the FlexTFLite_Detection_PostProcess op.
    """Tests object detection model through MLIR converter."""
    self._initObjectDetectionArgs()

    flags_str = ('--graph_def_file={0} --input_arrays={1} '
                 '--output_arrays={2} --input_shapes={3}'.format(
                     self._graph_def_file, self._input_arrays,
                     self._output_arrays, self._input_shapes))

    # Valid conversion.
    flags_str_final = (
        '{} --allow_custom_ops '
        '--experimental_new_converter '
        '--experimental_select_user_tf_ops=TFLite_Detection_PostProcess '
        '--target_ops=TFLITE_BUILTINS,SELECT_TF_OPS').format(flags_str)
    self._run(
        flags_str_final,
        should_succeed=True,
        expected_ops_in_converted_model=['FlexTFLite_Detection_PostProcess'])


class TfLiteConvertV2Test(TestModels):
  # [CLASS DOC]
  # Tests the command line converter with V2 style inputs (concrete function saved models, Keras
  # files, functional Keras models with output shape checks) and argument validation errors.
  # Most tests run only when TF2 behavior is enabled.

  @test_util.run_v2_only
  def testSavedModel(self):
    # [FUNCTION DOC]
    # Scenario: an trackable object with a function doubling its input, saved with a concrete
    # function. Action: converts via the saved model directory flag. Expected: success.
    input_data = constant_op.constant(1., shape=[1])
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    to_save = root.f.get_concrete_function(input_data)

    saved_model_dir = self._getFilepath('model')
    save(root, saved_model_dir, to_save)

    flags_str = '--saved_model_dir={}'.format(saved_model_dir)
    self._run(flags_str, should_succeed=True)

  @test_util.run_v2_only
  def testKerasFile(self):
    # [FUNCTION DOC]
    # Converts a saved sequential Keras file via the Keras model file flag. Expected: success.
    keras_file = self._getKerasModelFile()

    flags_str = '--keras_model_file={}'.format(keras_file)
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)

  @test_util.run_v2_only
  def testKerasFileMLIR(self):
    # [FUNCTION DOC]
    # Converts a saved sequential Keras file with the experimental new converter flag.
    # Expected: success.
    keras_file = self._getKerasModelFile()

    flags_str = (
        '--keras_model_file={} --experimental_new_converter'.format(keras_file))
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)

  @test_util.run_v2_only
  def testFunctionalKerasModel(self):
    # [FUNCTION DOC]
    # Converts a functional Keras model file with two outputs. Expected: success and the output
    # model has output shapes [1,1] and [1,2].
    keras_file = self._getKerasFunctionalModelFile()

    flags_str = '--keras_model_file={}'.format(keras_file)
    self._run(flags_str, should_succeed=True,
              expected_output_shapes=[[1, 1], [1, 2]])
    os.remove(keras_file)

  @test_util.run_v2_only
  def testFunctionalKerasModelMLIR(self):
    # [FUNCTION DOC]
    # Same as the functional Keras model test with the experimental new converter flag.
    # Expected: success and output shapes [1,1] and [1,2].
    keras_file = self._getKerasFunctionalModelFile()

    flags_str = (
        '--keras_model_file={} --experimental_new_converter'.format(keras_file))
    self._run(flags_str, should_succeed=True,
              expected_output_shapes=[[1, 1], [1, 2]])
    os.remove(keras_file)

  def testMissingRequired(self):
    # [FUNCTION DOC]
    # Runs the converter with only an unknown flag (no required input). Expected: it fails.
    self._run('--invalid_args', should_succeed=False)

  def testMutuallyExclusive(self):
    # [FUNCTION DOC]
    # Runs the converter with both a Keras model file and a saved model directory, which are
    # mutually exclusive. Expected: it fails.
    self._run(
        '--keras_model_file=model.h5 --saved_model_dir=/tmp/',
        should_succeed=False)


class ArgParserTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests of the tflite_convert argument parser for both V1 and V2 converter modes,
  # covering the experimental new converter and new quantizer flags. No external fixtures.

  @parameterized.named_parameters(('v1', False), ('v2', True))
  def test_without_experimental_new_converter(self, use_v2_converter):
    # [FUNCTION DOC]
    # Parses arguments without the new converter flag, for V1 and V2 parsers. Expected: the
    # new-converter value is true (default) and the new-quantizer value is unset (none).
    args = [
        '--saved_model_dir=/tmp/saved_model/',
        '--output_file=/tmp/output.tflite',
    ]

    # Note that when the flag parses to None, the converter uses the default
    # value, which is True.

    parser = tflite_convert._get_parser(use_v2_converter=use_v2_converter)
    parsed_args = parser.parse_args(args)
    self.assertTrue(parsed_args.experimental_new_converter)
    self.assertIsNone(parsed_args.experimental_new_quantizer)

  @parameterized.named_parameters(('v1', False), ('v2', True))
  def test_experimental_new_converter_none(self, use_v2_converter):
    # [FUNCTION DOC]
    # Parses arguments with the bare new converter flag (no value), for V1 and V2 parsers.
    # Expected: the new-converter value is true.
    args = [
        '--saved_model_dir=/tmp/saved_model/',
        '--output_file=/tmp/output.tflite',
        '--experimental_new_converter',
    ]

    parser = tflite_convert._get_parser(use_v2_converter=use_v2_converter)
    parsed_args = parser.parse_args(args)
    self.assertTrue(parsed_args.experimental_new_converter)

  @parameterized.named_parameters(
      ('v1_true', False, True),
      ('v1_false', False, False),
      ('v2_true', True, True),
      ('v2_false', True, False),
  )
  def test_experimental_new_converter(self, use_v2_converter, new_converter):
    # [FUNCTION DOC]
    # Parses arguments with an explicit true or false new converter value, for V1 and V2 parsers.
    # Expected: the parsed value equals the given value.
    args = [
        '--saved_model_dir=/tmp/saved_model/',
        '--output_file=/tmp/output.tflite',
        '--experimental_new_converter={}'.format(new_converter),
    ]

    parser = tflite_convert._get_parser(use_v2_converter=use_v2_converter)
    parsed_args = parser.parse_args(args)
    self.assertEqual(parsed_args.experimental_new_converter, new_converter)

  @parameterized.named_parameters(('v1', False), ('v2', True))
  def test_experimental_new_quantizer_none(self, use_v2_converter):
    # [FUNCTION DOC]
    # Parses arguments with the bare new quantizer flag (no value), for V1 and V2 parsers.
    # Expected: the new-quantizer value is true.
    args = [
        '--saved_model_dir=/tmp/saved_model/',
        '--output_file=/tmp/output.tflite',
        '--experimental_new_quantizer',
    ]

    parser = tflite_convert._get_parser(use_v2_converter=use_v2_converter)
    parsed_args = parser.parse_args(args)
    self.assertTrue(parsed_args.experimental_new_quantizer)

  @parameterized.named_parameters(
      ('v1_true', False, True),
      ('v1_false', False, False),
      ('v2_true', True, True),
      ('v2_false', True, False),
  )
  def test_experimental_new_quantizer(self, use_v2_converter, new_quantizer):
    # [FUNCTION DOC]
    # Parses arguments with an explicit true or false new quantizer value, for V1 and V2 parsers.
    # Expected: the parsed value equals the given value.
    args = [
        '--saved_model_dir=/tmp/saved_model/',
        '--output_file=/tmp/output.tflite',
        '--experimental_new_quantizer={}'.format(new_quantizer),
    ]

    parser = tflite_convert._get_parser(use_v2_converter=use_v2_converter)
    parsed_args = parser.parse_args(args)
    self.assertEqual(parsed_args.experimental_new_quantizer, new_quantizer)


if __name__ == '__main__':
  test.main()
