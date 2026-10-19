# [MODULE DOC]
# Test module for the TFLite converter's select TF op (flex) support: session, concrete function,
# saved model, custom op and TF quantization mode conversions, each followed by running the model
# in the TFLite interpreter. Runs under the standard TensorFlow test main entry point.
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
"""Tests for lite.py functionality related to select TF op usage."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.lite.python import lite
from tensorflow.lite.python import test_util as tflite_test_util
from tensorflow.lite.python.convert import register_custom_opdefs
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.testdata import double_op
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.trackable import autotrackable


class FromSessionTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for converting session graphs that use select TF (flex) ops.

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  def testFlexMode(self, enable_mlir):
    # [FUNCTION DOC]
    # Test (run with and without the MLIR converter): converts a session graph adding a 1x4
    # placeholder to itself, with select TF ops as the supported ops. Asserts conversion yields
    # a model, runs it on one to four and asserts the output is two, four, six, eight.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[1, 4], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
    converter.experimental_new_converter = enable_mlir
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check the model works with TensorFlow ops.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testFlexWithAutomaticPassThrough(self):
    # [FUNCTION DOC]
    # Test: converts a session graph with a single L2 loss op with select TF ops and the private
    # allow-all-select-ops option enabled, and asserts the flex L2 loss op appears in the model.
    # Create a graph that has one L2Loss op.
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor = array_ops.placeholder(
            shape=[4], dtype=dtypes.float32, name='input')
        out_tensor = nn_ops.l2_loss(in_tensor)
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                      [out_tensor])
        converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
        converter._experimental_allow_all_select_tf_ops = True
        tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    self.assertIn('FlexL2Loss', tflite_test_util.get_ops_list(tflite_model))

  def testDeprecatedFlags(self):
    # [FUNCTION DOC]
    # Test: converts a session graph adding a tensor to itself after setting the deprecated
    # target ops property to select TF ops. Asserts the property and the supported ops of the
    # target spec both equal that set, converts, runs on one to four and asserts the output is
    # two, four, six, eight.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[1, 4], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.target_ops = set([lite.OpsSet.SELECT_TF_OPS])

    # Ensure `target_ops` is set to the correct value after flag deprecation.
    self.assertEqual(converter.target_ops, set([lite.OpsSet.SELECT_TF_OPS]))
    self.assertEqual(converter.target_spec.supported_ops,
                     set([lite.OpsSet.SELECT_TF_OPS]))

    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Check the model works with TensorFlow ops.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())


class FromConcreteFunctionTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):
  # [CLASS DOC]
  # Tests for converting concrete functions that use select TF ops.

  @parameterized.named_parameters(
      ('EnableMlirConverter', True),  # enable mlir
      ('DisableMlirConverter', False))  # disable mlir
  @test_util.run_v2_only
  def testFloat(self, enable_mlir):
    # [FUNCTION DOC]
    # Test (v2 only, run with and without the MLIR converter): converts a concrete function
    # multiplying two variables (3 and 2) by the input, with select TF ops, then runs it on 4
    # and asserts the output is 24.
    input_data = constant_op.constant(1., shape=[1])
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func],
                                                               root)
    converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
    converter.experimental_new_converter = enable_mlir
    tflite_model = converter.convert()

    # Check the model works with TensorFlow ops.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([4.0], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([24.0], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  # Ensure that input TFLite buffer is not reused for ops such as
  # `TensorListSetItem`. The example model has a while loop, and the while body
  # has a `TensorListSetItem` op which takes the output from a `Where` op.
  @test_util.run_v2_only
  def testDisableFlexTensorMemoryReusing(self):
    # [FUNCTION DOC]
    # Test: converts a function with a loop whose body stores the nonzero-element coordinates
    # into a tensor list, using built-in plus select TF ops. Runs it on a 2x3 input and asserts
    # the flattened output equals zero, one, one, two (so the input buffer is not wrongly
    # reused by the flex list-set-item op).

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[2, 3], dtype=tf.float32, name='x')
    ])
    def model(x):
      # [FUNCTION DOC]
      # Loops twice; each iteration finds the nonzero positions in row i of x and stores them
      # in a reserved int64 tensor list. Returns the stacked list.
      l = list_ops.tensor_list_reserve(
          element_dtype=tf.int64, element_shape=[None, 1], num_elements=2)
      init_state = (0, x, l)
      condition = lambda i, x, l: i < 2

      def body(i, x, l):
        # [FUNCTION DOC]
        # Loop body: stores the nonzero positions of row i of x at index i of the list l and
        # returns the incremented counter, x and the updated list.
        element = tf.where(x[i])
        l = list_ops.tensor_list_set_item(l, i, element)
        return i + 1, x, l

      _, _, l_final = tf.while_loop(condition, body, init_state)
      return list_ops.tensor_list_stack(l_final, element_dtype=tf.int64)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [model.get_concrete_function()])
    converter.target_spec.supported_ops = set(
        [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS])
    tflite_model = converter.convert()

    # Check the model produces correct result.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 0.0], [0.0, 5.0, 6.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([0, 1, 1, 2], dtype=np.int64)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == np.ndarray.flatten(output_data)).all())


class WithCustomOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for converting models containing custom or externally defined ops via select TF ops.

  def _createGraphWithCustomOp(self, opname='CustomAdd'):
    # [FUNCTION DOC]
    # HELPER: builds the custom-op graph for the custom op test.
    # Builds a graph with an input placeholder (1x16x16x3 float) added to itself, copies its
    # graph definition, renames add nodes to the given op name (default CustomAdd) and removes
    # their type attribute, and registers a matching custom op definition (two float inputs,
    # one float output) globally (side effect). Returns the modified graph definition, a mapping
    # of input name to tensor and a mapping of output name to tensor.
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

    return (new_graph, inputs, outputs)

  def testFlexWithCustomOp(self):
    # [FUNCTION DOC]
    # Test: builds a graph with a custom add op, saves it as a saved model, converts it with
    # select TF ops and the custom op listed as a user op, and asserts the flex custom add op
    # appears in the converted model's op list.
    new_graph, inputs, outputs = self._createGraphWithCustomOp(
        opname='CustomAdd4')

    # Import to load the custom opdef.
    saved_model_dir = os.path.join(self.get_temp_dir(), 'model')
    with ops.Graph().as_default():
      with session.Session() as sess:
        import_graph_def(new_graph, name='')
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
    converter.target_spec.experimental_select_user_tf_ops = ['CustomAdd4']
    tflite_model = converter.convert()

    self.assertIn('FlexCustomAdd4', tflite_test_util.get_ops_list(tflite_model))

  def testFlexWithDoubleOp(self):
    # [FUNCTION DOC]
    # Test: saves a model using the externally loaded double op on an int32 placeholder,
    # converts it with select TF ops and the double op named as a user op, asserts the flex
    # double op appears in the model, then runs it on one to four and asserts the output is
    # two, four, six, eight.
    # Create a graph that has one double op.
    saved_model_dir = os.path.join(self.get_temp_dir(), 'model2')
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor = array_ops.placeholder(
            shape=[1, 4], dtype=dtypes.int32, name='input')
        out_tensor = double_op.double(in_tensor)
        inputs = {'x': in_tensor}
        outputs = {'z': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = set([lite.OpsSet.SELECT_TF_OPS])
    converter.target_spec.experimental_select_user_tf_ops = ['Double']
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    self.assertIn('FlexDouble', tflite_test_util.get_ops_list(tflite_model))

    # Check the model works with TensorFlow ops.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.int32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.int32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())


class FromSavedModelTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for converting saved models that need select TF ops (resource variables).

  @test_util.run_v2_only
  def testFlexResourceVariables(self):
    # [FUNCTION DOC]
    # Test: saves a module with a variable and a function that conditionally adds ones to it,
    # converts it with built-in plus select TF ops and resource variables enabled, and runs the
    # signature twice on one to four.
    # KNOWN ISSUE: the assertions check the bound method of the comparison result without calling
    # it, so they always pass and verify nothing about the outputs.

    class Model(tf.Module):
      # [CLASS DOC]
      # Helper module with one variable and a traced function using control flow.

      def __init__(self):
        # [FUNCTION DOC]
        # Creates the variable v as a 1x4 tensor of zeros.
        self.v = tf.Variable([[0.0, 0.0, 0.0, 0.0]])

      @tf.function(
          input_signature=[tf.TensorSpec(shape=[1, 4], dtype=tf.float32)])
      def eval(self, x):
        # [FUNCTION DOC]
        # If the mean of x exceeds one, adds ones to the variable (side effect); returns the
        # variable plus x. Input is a 1x4 float tensor.
        # Control flow is needed to generate "FlexReadVariableOp".
        if tf.reduce_mean(x) > 1.0:
          self.v.assign_add([[1.0, 1.0, 1.0, 1.0]])
        return self.v + x

    m = Model()
    to_save = m.eval.get_concrete_function()
    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    tf.saved_model.save(m, save_dir, to_save)
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)

    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    # Check the model works with TensorFlow ops.
    interpreter = Interpreter(model_content=tflite_model)
    signature_runner = interpreter.get_signature_runner()
    outputs = signature_runner(
        x=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    expected_output = np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    self.assertTrue((expected_output == list(outputs.values())[0]).all)

    # Second run.
    outputs = signature_runner(
        x=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    expected_output = np.array([[3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    self.assertTrue((expected_output == list(outputs.values())[0]).all)


class TFQuantizationTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the experimental TF quantization mode of the V2 converter: which ops stay built-in
  # versus become flex ops, and that converted models still run correctly.

  @parameterized.named_parameters(('DefaultMode', 'DEFAULT'),
                                  ('LegacyIntegerMode', 'LEGACY_INTEGER'))
  def testAddOp(self, tf_quantization_mode):
    # [FUNCTION DOC]
    # Test: converts an add-to-itself function under each quantization mode. Asserts the model
    # holds a built-in add op in legacy integer mode and a flex add op in default mode, then
    # runs it on one to four and asserts the output is two, four, six, eight.
    root = autotrackable.AutoTrackable()
    root.add_func = def_function.function(lambda x: x + x)
    input_data = tf.reshape(tf.range(4, dtype=tf.float32), [1, 4])
    concrete_func = root.add_func.get_concrete_function(input_data)

    # Convert model and check if the op is not flex.
    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func],
                                                               root)
    converter._experimental_tf_quantization_mode = tf_quantization_mode
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    if tf_quantization_mode == 'LEGACY_INTEGER':
      self.assertIn('ADD', tflite_test_util.get_ops_list(tflite_model))
    else:
      self.assertIn('FlexAddV2', tflite_test_util.get_ops_list(tflite_model))

    # Check the model works.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([[2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  @parameterized.named_parameters(('DefaultMode', 'DEFAULT'),
                                  ('LegacyIntegerMode', 'LEGACY_INTEGER'))
  def testL2LossOp(self, tf_quantization_mode):
    # [FUNCTION DOC]
    # Test: converts a function computing the L2 loss under each quantization mode and asserts
    # the model contains the flex L2 loss op in both modes, then runs it on four values one to
    # four and asserts the output is 15.
    root = autotrackable.AutoTrackable()
    root.l2_loss_func = def_function.function(lambda x: nn_ops.l2_loss(x))  # pylint: disable=unnecessary-lambda
    input_data = tf.range(4, dtype=tf.float32)
    concrete_func = root.l2_loss_func.get_concrete_function(input_data)

    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func],
                                                               root)
    converter._experimental_tf_quantization_mode = tf_quantization_mode
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    self.assertIn('FlexL2Loss', tflite_test_util.get_ops_list(tflite_model))

    # Check the model works.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([15.0], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  @parameterized.named_parameters(('DefaultMode', 'DEFAULT'),
                                  ('LegacyIntegerMode', 'LEGACY_INTEGER'))
  def testConvOpWithBias(self, tf_quantization_mode):
    # [FUNCTION DOC]
    # Test: converts a function doing a 2D convolution, bias add and relu under each quantization
    # mode (default, legacy integer) and checks the resulting model contains only convolution and
    # reshape built-in ops (no flex ops), then runs it on a 1x2x2x1 input with a 1x2x1x1 filter
    # and asserts the output equals 4 and 6.

    class ConvModel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # Helper trackable model holding one traced function for the convolution test.

      @def_function.function
      def conv_func(self, in_tensor, filter_tensor):
        # [FUNCTION DOC]
        # Applies a valid-padding NHWC convolution with unit strides and dilations to the input
        # and filter tensors, adds a constant bias of 3, and returns the relu of the result.
        bias = constant_op.constant(3., shape=[1])
        conv_tensor = tf.nn.conv2d(
            in_tensor,
            filter_tensor,
            strides=[1, 1, 1, 1],
            dilations=[1, 1, 1, 1],
            padding='VALID',
            data_format='NHWC')
        conv_tensor = conv_tensor + bias
        return tf.nn.relu(conv_tensor)

    root = ConvModel()
    input_data = tf.reshape(tf.range(4, dtype=tf.float32), [1, 2, 2, 1])
    filter_data = tf.reshape(tf.range(2, dtype=tf.float32), [1, 2, 1, 1])
    concrete_func = root.conv_func.get_concrete_function(
        input_data, filter_data)

    converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func],
                                                               root)
    converter._experimental_tf_quantization_mode = tf_quantization_mode
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    self.assertCountEqual(['CONV_2D', 'RESHAPE'],
                          tflite_test_util.get_ops_list(tflite_model))

    # Check the model works.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    test_input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(
        (1, 2, 2, 1))
    interpreter.set_tensor(input_details[0]['index'], test_input)
    test_filter = np.array([1.0, 0.0], dtype=np.float32).reshape((1, 2, 1, 1))
    interpreter.set_tensor(input_details[1]['index'], test_filter)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    expected_output = np.array([[[[4.]], [[6.]]]], dtype=np.float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())


if __name__ == '__main__':
  test.main()
