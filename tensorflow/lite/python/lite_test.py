# [MODULE DOC]
# Unit tests for the TensorFlow Lite converter front end defined in the lite module (the
# TFLiteConverter class, its deprecated TocoConverter alias, and the QuantizationMode helper).
# Each test builds a small TensorFlow model (from a session graph, a frozen graph file, a
# SavedModel directory or a saved Keras HDF5 file), converts it to a TFLite flatbuffer, loads
# the result in the TFLite Interpreter and checks tensor names, dtypes, shapes, quantization
# parameters, inference results, conversion metadata and error messages.
# Test groups: constructor validation, conversion from sessions (float, quantized, dynamic
# range, int8/int16/float16, shape handling, graphviz dumps), frozen graph files (including an
# object detection model), SavedModels, Keras files, sparsity, grappler constant folding,
# default converter attributes, control flow V1 errors and deprecated optimization warnings.
# Import-time side effects: none beyond imports; running the file as a script starts the
# TensorFlow test runner. Not a helper module; it is a test module.
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
"""Tests for lite.py."""

import io
import logging
import os
import tempfile

from absl.testing import parameterized
import numpy as np
from tensorflow import keras

from tensorflow.lite.python import conversion_metadata_schema_py_generated as metadata_fb
from tensorflow.lite.python import lite
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import util
from tensorflow.lite.python.convert import ConverterError
from tensorflow.lite.python.convert import mlir_quantize
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
from tensorflow.lite.python.util import get_conversion_metadata
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.variables import global_variables_initializer as _global_variables_initializer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.training.training_util import write_graph


class LiteTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # HELPER: common base class for every test case in this module.
  # Empty subclass of the standard TensorFlow test case; it adds no state or behavior and only
  # gives all tests here a shared parent.
  """Base class of all the tests in this module."""


class TestModels(LiteTest):
  # [CLASS DOC]
  # HELPER: shared base for the model conversion test classes, providing a debug info check.
  # Holds no state; it only adds the debug info validation method used by several subclasses.

  def assertValidDebugInfo(self, debug_info):
    # [FUNCTION DOC]
    # HELPER: custom assertion used by the graph debug info tests.
    # Input: debug_info, a graph debug info message with a list of source file paths.
    # Collects the base names of all files in the message and asserts that this test file's name
    # is present and that the name of the V2 test file is absent. Returns nothing; fails the test
    # through assertion errors.
    """Verify the DebugInfo is valid."""
    file_names = set()
    for file_path in debug_info.files:
      file_names.add(os.path.basename(file_path))
    # To make the test independent on how the nodes are created, we only assert
    # the name of this test file.
    self.assertIn('lite_test.py', file_names)
    self.assertNotIn('lite_v2_test.py', file_names)


class FromConstructor(TestModels):
  # [CLASS DOC]
  # Tests for building a TFLiteConverter directly through its constructor (rather than the
  # from_session style factories): argument validation, redundant argument warnings and shape
  # overriding of a frozen graph definition. Builds graphs inside each test; no shared fixtures.

  # Tests invalid constructors using a dummy value for the GraphDef.
  def testInvalidConstructor(self):
    # [FUNCTION DOC]
    # Scenario: construct the converter with no tensors and only one of the input shapes or
    # output arrays given (graph definition is a dummy null value).
    # Expects converting to raise a value error carrying the message that both input arrays with
    # shape and output or control output arrays must be defined, for both missing-argument cases.
    message = (
        'If input_tensors and output_tensors are None, both '
        'input_arrays_with_shape and output_arrays|control_output_arrays must '
        'be defined.')

    # `output_arrays` is not defined.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter(
          None, None, [], input_arrays_with_shape=[('input', [3,
                                                              9])]).convert()
    self.assertEqual(message, str(error.exception))

    # `input_arrays_with_shape` is not defined.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter(None, [], None, output_arrays=['output']).convert()
    self.assertEqual(message, str(error.exception))

  # Tests valid constructors using a dummy value for the GraphDef.
  def testValidConstructor(self):
    # [FUNCTION DOC]
    # Scenario: construct the converter with a null graph definition plus input shapes and
    # output arrays, then with explicit tensor name lists.
    # Expects: no valid tensors are reported for the first, its input array names are listed,
    # setting a batch size raises a value error saying to use the input shapes parameter, and
    # the second converter reports valid tensors.
    converter = lite.TFLiteConverter(
        None,
        None,
        None,
        input_arrays_with_shape=[('input', [3, 9])],
        output_arrays=['output'])
    self.assertFalse(converter._has_valid_tensors())
    self.assertEqual(converter.get_input_arrays(), ['input'])

    with self.assertRaises(ValueError) as error:
      converter._set_batch_size(1)
    self.assertEqual(
        'The batch size cannot be set for this model. Please use '
        'input_shapes parameter.', str(error.exception))

    converter = lite.TFLiteConverter(None, ['input_tensor'], ['output_tensor'])
    self.assertTrue(converter._has_valid_tensors())

  def testRedundantArgumentsWarning(self):
    # [FUNCTION DOC]
    # Scenario: build and freeze a graph with a placeholder and an add, then construct the
    # converter giving both tensors and also input shapes and output arrays.
    # Side effect: temporarily attaches a log handler writing to an in-memory buffer to the
    # root logger and removes it at the end.
    # Expects conversion to succeed and the log to contain warnings that the input arrays with
    # shape and the output arrays arguments are ignored.
    """Test if the warning message when there are redundant arguments."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
      out_tensor = math_ops.add(in_tensor, in_tensor, name='add')
      sess = session.Session()

    frozen_graph_def = (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, sess.graph_def, ['add']))

    # Convert model and ensure model is not None.
    log = io.StringIO()
    handler = logging.StreamHandler(log)
    logging.root.addHandler(handler)
    converter = lite.TFLiteConverter(frozen_graph_def, [in_tensor],
                                     [out_tensor],
                                     [('in_tensor', [2, 16, 16, 3])], ['add'])

    input_warning_message = 'input_arrays_with_shape will be ignored'
    output_warning_message = 'output_arrays will be ignored'

    # Convert model and ensure model is not None.
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)
    self.assertIn(input_warning_message, log.getvalue())
    self.assertIn(output_warning_message, log.getvalue())
    logging.root.removeHandler(handler)

  def testShapeOverriding(self):
    # [FUNCTION DOC]
    # Scenario: freeze a graph whose input has an unknown batch dimension and convert it
    # through the constructor with an input shape override to a fixed shape.
    # Expects the interpreter to show one float input named in_tensor with the overridden shape,
    # one float output named add with the same shape, and zero quantization parameters.
    """Test a shape overriding case via the constructor."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
      math_ops.add(in_tensor, in_tensor, name='add')
      sess = session.Session()

    frozen_graph_def = (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, sess.graph_def, ['add']))

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter(frozen_graph_def, None, None,
                                     [('in_tensor', [2, 16, 16, 3])], ['add'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('in_tensor', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([2, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testPartialShapeOverriding(self):
    # [FUNCTION DOC]
    # Scenario: freeze a graph with two placeholders, then override the shape of only one of
    # them via the constructor.
    # Expects conversion to raise a converter error because the other placeholder stays
    # unhandled.
    """Test a partial shape overriding case via the constructor."""
    with ops.Graph().as_default():
      in_tensor_a = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor_a')
      in_tensor_b = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor_b')
      math_ops.add(in_tensor_a, in_tensor_b, name='add')
      sess = session.Session()

    frozen_graph_def = (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, sess.graph_def, ['add']))

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter(frozen_graph_def, None, None,
                                     [('in_tensor_a', [2, 16, 16, 3])], ['add'])
    # There is an unhandled Placeholder op.
    with self.assertRaises(ConverterError):
      converter.convert()

  def testInvalidShapeOverriding(self):
    # [FUNCTION DOC]
    # Scenario: freeze a graph and override the shape of an input name that does not exist.
    # Expects conversion to raise a converter error.
    """Test an invalid shape overriding case via the constructor."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
      math_ops.add(in_tensor, in_tensor, name='add')
      sess = session.Session()

    frozen_graph_def = (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, sess.graph_def, ['add']))

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter(frozen_graph_def, None, None,
                                     [('wrong_tensor', [2, 16, 16, 3])],
                                     ['add'])
    with self.assertRaises(ConverterError):
      converter.convert()


class FromSessionTest(TestModels, parameterized.TestCase):
  # [CLASS DOC]
  # Largest test group: covers converting models from a live TensorFlow session via the from
  # session factory. Includes float, uint8, int8, int16 and float16 quantization, dynamic range
  # quantization, per-channel options, strings, shapes with unknown dimensions, variable
  # freezing, graphviz output, conversion metadata, functions and debug info. Mixes in the
  # parameterized test case so some tests run once per named parameter set.

  def testFloatModel(self):
    # [FUNCTION DOC]
    # Scenario: a float placeholder of shape one by sixteen by sixteen by three added to itself.
    # Converts from the session and loads the result in the interpreter.
    # Expects one float input named Placeholder and one float output named add with the same
    # shape and no quantization parameters.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testFloatModelQuantizedInput(self):
    # [FUNCTION DOC]
    # Scenario: the float add model converted with an unsigned 8-bit inference input type, float
    # inference type, and quantized input stats (mean zero, standard deviation one) for the input.
    # Expects the input tensor to be unsigned 8-bit with quantization scale one and zero point
    # zero, while the output stays float with no quantization.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.inference_input_type = dtypes.uint8
    converter.inference_type = dtypes.float32
    converter.quantized_input_stats = {'Placeholder': (0., 1.)}  # mean, std_dev
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])  # float

  def testForgottenCallToAllocateTensors(self):
    # [FUNCTION DOC]
    # Scenario: convert the float add model, create an interpreter but never allocate tensors,
    # then try to set an input tensor.
    # Expects a value error from the interpreter.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()
    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    input_index = interpreter.get_input_details()[0]['index']
    dummy_tensor = np.ones(shape=[1, 16, 16, 3], dtype=np.float32)
    with self.assertRaises(ValueError):
      interpreter.set_tensor(input_index, dummy_tensor)

  @parameterized.named_parameters(
      ('_INT8InputOutput', False, False, dtypes.int8),
      ('_UINT8InputOutput', False, False, dtypes.uint8),
      ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16),
      ('_IntOnly_INT8InputOutput', True, False, dtypes.int8),
      ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8),
      ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16),
      ('_IntOnly_INT8InputOutputMlirQuant', True, False, dtypes.int8, True),
      ('_IntOnly_UINT8InputOutputMlirQuant', True, False, dtypes.uint8, True))
  def testIntegerQuantizationWithUnsupportedOps(self,
                                                is_int_only,
                                                is_int16_quantize,
                                                inference_input_output_type,
                                                enable_mlir_quantizer=False):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: is_int_only (restrict supported ops to integer builtins plus
    # float builtins), is_int16_quantize (use 16-bit activations with 8-bit weights),
    # inference_input_output_type (the integer type requested for model inputs and outputs),
    # enable_mlir_quantizer (optional, default false; use the new quantizer).
    # Scenario: a graph with two inputs and two outputs where a ceiling op, which has no integer
    # kernel, sits on one path. A small random representative dataset generator is supplied and
    # the default optimization is enabled.
    # Expects conversion to succeed and, in the interpreter, the inputs and outputs of the
    # path with the tanh to have the requested integer type, while those of the ceiling path
    # stay float unless the new quantizer is used (then they have the integer type too).
    with ops.Graph().as_default():
      in_tensor_a = array_ops.placeholder(shape=[3], dtype=dtypes.float32)
      in_tensor_b = array_ops.placeholder(shape=[3], dtype=dtypes.float32)
      # ceil kernel does not support int8 nor int16 types neither.
      left = math_ops.ceil(in_tensor_a)
      out_tensor_b = math_ops.tanh(in_tensor_b)
      add = math_ops.add(left, out_tensor_b)
      # ceil kernel does not support int8 nor int16 types neither.
      out_tensor_a = math_ops.ceil(add)
      sess = session.Session()

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: representative dataset generator for the unsupported-ops quantization test.
      # Yields five times a list of two random float vectors of length three, one per model input.
      for _ in range(5):
        yield [
            np.random.uniform(-1, 1, size=(3)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3)).astype(np.float32)
        ]

    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_a, in_tensor_b], [out_tensor_a, out_tensor_b])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    if is_int_only:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet
            .EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.TFLITE_BUILTINS
        ]
      else:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.TFLITE_BUILTINS_INT8, lite.OpsSet.TFLITE_BUILTINS
        ]
    else:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet
            .EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.TFLITE_BUILTINS
        ]
      else:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.TFLITE_BUILTINS
        ]

    quantized_converter.inference_input_type = inference_input_output_type
    quantized_converter.inference_output_type = inference_input_output_type
    quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    expected_dtype = inference_input_output_type.as_numpy_dtype
    # Allow float32 for fallback on non-quantizable op.
    expected_ceil_dtype = (
        expected_dtype if enable_mlir_quantizer else dtypes.float32)

    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual(input_details[0]['dtype'], expected_ceil_dtype)
    self.assertEqual(input_details[1]['dtype'], expected_dtype)
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 2)
    self.assertEqual(output_details[0]['dtype'], expected_ceil_dtype)
    self.assertEqual(output_details[1]['dtype'], expected_dtype)

  @parameterized.named_parameters(
      ('_PerChannelQuant', False, False),
      ('_PerChannelMlirQuant', False, True),
      ('_PerTensorQuant', True, False),
      ('_PerTensorMlirQuant', True, True),
      ('_PerChannelMlirDynamicRangeQuant', False, False, False),
      ('_PerTensorMlirDynamicRangeQuant', True, False, False))
  def testDisablePerChannelQuantization(self,
                                        disable_per_channel=False,
                                        enable_mlir_quantizer=False,
                                        representative_dataset=True):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: disable_per_channel (use per-tensor quantization of the
    # convolution weights, default false), enable_mlir_quantizer (use the new quantizer, default
    # false), representative_dataset (supply calibration data; default true, false means dynamic
    # range quantization).
    # Scenario: quantize a convolution model with 38 filters (enough elements for dynamic range
    # quantization), optionally setting the private disable-per-channel flag. The interpreter is
    # built without default delegates because a delegate could turn per-tensor into per-channel.
    # Expects the convolution weight tensor to have one scale when per-channel is disabled and
    # one scale per filter otherwise; zero points must also match unless there is only one.
    # With the new quantizer and per-tensor mode the tensor name differs from the default.
    if enable_mlir_quantizer:
      if disable_per_channel:
        k_conv_name = 'output1'
      else:
        k_conv_name = 'Conv2D1'
    else:
      k_conv_name = 'Conv2D1'

    # Dynamic range quant requires total num elements of filters > 1024.
    k_num_filters = 38
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel(
          k_num_filters)
      sess = session.Session()

    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    if representative_dataset:
      quantized_converter.representative_dataset = calibration_gen
    quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
    if disable_per_channel:
      quantized_converter._experimental_disable_per_channel = (
          disable_per_channel)
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # Do not apply delegates as XNNPack converts per tensor to per channel.
    interp = Interpreter(
        model_content=quantized_tflite_model,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    detail = next(
        (d for d in interp.get_tensor_details() if d['name'] == k_conv_name)
    )
    quant_params = detail['quantization_parameters']
    expected_num_params = 1 if disable_per_channel else k_num_filters
    self.assertLen(quant_params['scales'], expected_num_params)
    if len(quant_params['zero_points']) != 1:
      self.assertLen(quant_params['zero_points'], expected_num_params)

  def testString(self):
    # [FUNCTION DOC]
    # Scenario: a string placeholder of four elements reshaped to two by two.
    # Expects byte-string typed input named Placeholder with shape four and output named
    # Reshape with shape two by two. Reading and writing string data is not tested (open TODO).
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[4], dtype=dtypes.string)
      out_tensor = array_ops.reshape(in_tensor, shape=[2, 2])
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.bytes_, input_details[0]['dtype'])
    self.assertAllEqual([4], input_details[0]['shape'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('Reshape', output_details[0]['name'])
    self.assertEqual(np.bytes_, output_details[0]['dtype'])
    self.assertAllEqual([2, 2], output_details[0]['shape'])
    # TODO(b/122659643): Test setting/getting string data via the python
    # interpreter API after support has been added.

  def testIntermediateInputArray(self):
    # [FUNCTION DOC]
    # Scenario: a chain of two adds where the intermediate tensor of the first add is chosen as
    # the converter input instead of the placeholder.
    # Expects a single float input named add and a single float output named add_1.
    """Convert a model from an intermediate input array."""
    with ops.Graph().as_default():
      in_tensor_init = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      in_tensor_final = in_tensor_init + in_tensor_init
      out_tensor = in_tensor_final + in_tensor_final
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor_final],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('add', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add_1', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testSizeNoneInvalid(self):
    # [FUNCTION DOC]
    # Scenario: a placeholder with completely unknown shape, converted with the new converter
    # disabled so the old shape checking code runs.
    # Expects a value error with the message asking to provide an input shape for the array
    # named Placeholder.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test None as shape when dynamic shapes are disabled. Run with TOCO in
    # order to invoke shape checking code.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_new_converter = False
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual('Provide an input shape for input array \'Placeholder\'.',
                     str(error.exception))

  def testScalarValid(self):
    # [FUNCTION DOC]
    # Scenario: a scalar (empty shape) float placeholder added to itself.
    # Expects empty shapes in the tensor details and, after setting the scalar four and invoking,
    # the output to equal eight.
    # KNOWN ISSUE: the output shape check inspects the input details again instead of the output
    # details, so the output shape is never actually verified.
    # Construct a graph using a scalar (empty shape) input.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[])
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test conversion with the scalar input shape.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertEmpty(input_details[0]['shape'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertEmpty(input_details[0]['shape'])

    # Validate inference using the scalar inputs/outputs.
    test_input = np.array(4.0, dtype=np.float32)
    expected_output = np.array(8.0, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertEqual(expected_output, output_data)

  def testSizeInvalid(self):
    # [FUNCTION DOC]
    # Scenario: a placeholder with an unknown dimension in a position other than the first,
    # converted with the new converter disabled so the shape checking code runs.
    # Expects a value error saying that an unknown dimension is only supported in the first
    # dimension and naming the offending tensor and its shape.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, None, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test invalid shape. None after 1st dimension. Run with TOCO in order to
    # invoke shape checking code.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.experimental_new_converter = False
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual(
        'None is only supported in the 1st dimension. Tensor '
        '\'Placeholder\' has invalid shape \'[1, None, 16, 3]\'.',
        str(error.exception))

  def testSizeNone(self):
    # [FUNCTION DOC]
    # Scenario: a placeholder with an unknown second dimension converted with the default
    # converter.
    # Expects the interpreter shape to use one for the unknown dimension and the shape signature
    # to keep minus one. Strict resizing to a different known dimension must raise a runtime
    # error mentioning that only unknown dimensions may be changed; strict resizing of the
    # unknown dimension then succeeds, inference runs and the signatures remain unchanged.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, None, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Test None after 1st dimension.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 1, 16, 3], input_details[0]['shape'])
    self.assertAllEqual([1, -1, 16, 3], input_details[0]['shape_signature'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    # Resize tensor with strict checking.
    with self.assertRaises(RuntimeError) as error:
      interpreter.resize_tensor_input(0, [3, 16, 16, 3], strict=True)
    self.assertIn(
        'ResizeInputTensorStrict only allows mutating unknown dimensions '
        'identified by -1.', str(error.exception))

    # Resize tensor and invoke.
    interpreter.resize_tensor_input(0, [1, 16, 16, 3], strict=True)
    interpreter.allocate_tensors()

    test_input = np.full([1, 16, 16, 3], 1.0, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertAllEqual([1, -1, 16, 3], input_details[0]['shape_signature'])

    output_details = interpreter.get_output_details()
    self.assertAllEqual([1, -1, 16, 3], output_details[0]['shape_signature'])

  def testResizeTensorInputStrict(self):
    # [FUNCTION DOC]
    # Scenario: a fully known shape float add model; the interpreter is asked to resize the input
    # in strict mode, first to an incompatible shape then to the same shape.
    # Expects a runtime error for the incompatible resize, and tensor allocation to succeed after
    # the compatible one.
    # Ensures that resize_tensor_input(strict=True) works as expected.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)

    # Resize incorrect value.
    with self.assertRaises(RuntimeError) as error:
      interpreter.resize_tensor_input(0, [3, 16, 16, 3], strict=True)
    self.assertIn(
        'ResizeInputTensorStrict only allows mutating unknown dimensions '
        'identified by -1.', str(error.exception))

    # Resize correct value.
    interpreter.resize_tensor_input(0, [1, 16, 16, 3], strict=True)
    interpreter.allocate_tensors()

  def testBatchSizeValid(self):
    # [FUNCTION DOC]
    # Scenario: a placeholder whose first (batch) dimension is unknown.
    # Expects the converted input and output shapes to use batch size one, names Placeholder and
    # add, float type and no quantization.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testBatchSizeNonZero(self):
    # [FUNCTION DOC]
    # Scenario: a matrix multiplication of an input with unknown batch dimension and a second
    # fully known input.
    # Expects two inputs; the first gets batch size one (shape one by four) and the second keeps
    # its shape of four by ten.
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[None, 4], dtype=dtypes.float32, name='input1')
      in_tensor_2 = array_ops.placeholder(
          shape=[4, 10], dtype=dtypes.float32, name='input2')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2)
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess,
                                                  [in_tensor_1, in_tensor_2],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual('input1', input_details[0]['name'])
    self.assertAllEqual([1, 4], input_details[0]['shape'])
    self.assertEqual('input2', input_details[1]['name'])
    self.assertAllEqual([4, 10], input_details[1]['shape'])

  def testFreezeGraph(self):
    # [FUNCTION DOC]
    # Scenario: a graph adding a placeholder to a variable and taking the top-k, using the second
    # output (indices) as the converter output. The variables are initialized in the session.
    # Expects the converter to freeze the variable correctly (handling tensor names with an output
    # index suffix) and the interpreter to show a float input named Placeholder and an int32
    # output named top_k:1 with shape one by sixteen by sixteen by one.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      var = variable_scope.get_variable(
          'weights', shape=[1, 16, 16, 3], dtype=dtypes.float32)
      # Get the second output to ensure freezing properly processes tensor names
      # like 'X:1'.
      out_tensor = nn_ops.top_k(in_tensor + var, name='top_k')[1]
      sess = session.Session()
      sess.run(_global_variables_initializer())

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('top_k:1', output_details[0]['name'])
    self.assertEqual(np.int32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 1], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testGraphviz(self):
    # [FUNCTION DOC]
    # Scenario: the float add model converted with the output format set to the graphviz dot
    # format.
    # Expects the conversion result to be not null.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.output_format = lite_constants.GRAPHVIZ_DOT
    graphviz_output = converter.convert()
    self.assertIsNotNone(graphviz_output)

  def testDumpGraphviz(self):
    # [FUNCTION DOC]
    # Scenario: the float add model converted with a graphviz dump directory set to the test
    # temporary directory. Side effect: dot files may be written into that directory.
    # Expects conversion and tensor allocation to succeed.
    # KNOWN ISSUE: the final assertions test only that a count and two boolean existence results
    # are not null, which always holds; the dump files are not actually verified.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    graphviz_dir = self.get_temp_dir()
    converter.dump_graphviz_dir = graphviz_dir
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Ensure interpreter is able to allocate and check graphviz data.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    num_items_graphviz = len(os.listdir(graphviz_dir))
    self.assertIsNotNone(num_items_graphviz)
    self.assertIsNotNone(
        os.path.exists(os.path.join(graphviz_dir, 'toco_AT_IMPORT.dot')))
    self.assertIsNotNone(
        os.path.exists(
            os.path.join(graphviz_dir, 'toco_AFTER_TRANSFORMATIONS.dot')))

  def testQuantizeDynamicRange(self):
    # [FUNCTION DOC]
    # Scenario: matrix multiplication with a 33 by 33 random constant (more than 1024 elements so
    # weight quantization applies). Converts once as float and once with the default
    # optimization.
    # Expects both conversions to succeed and the quantized model to be smaller in bytes.
    # Seeds the global random generator with zero.
    np.random.seed(0)
    with ops.Graph().as_default():
      # We need the tensor to have more than 1024 elements for quantize_weights
      # to kick in. Thus, the [33, 33] shape.
      in_tensor_1 = array_ops.placeholder(
          shape=[33, 33], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = constant_op.constant(
          np.random.uniform(low=-10., high=10., size=(33, 33)),
          shape=[33, 33],
          dtype=dtypes.float32,
          name='inputB')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [in_tensor_1],
                                                        [out_tensor])
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    # Convert quantized weights model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_1], [out_tensor])

    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  def testQuantizeDynamicRangeDeprecatedPostTrainingQuantizeAttribute(
      self):
    # [FUNCTION DOC]
    # Scenario: the large matrix multiplication model; the deprecated post training quantize
    # attribute is read, then set to true.
    # Expects it to be false initially, true after setting, and the optimizations list to become
    # the default optimization only; conversion then succeeds.
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[33, 33], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = constant_op.constant(
          np.random.uniform(low=-10., high=10., size=(33, 33)),
          shape=[33, 33],
          dtype=dtypes.float32,
          name='inputB')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
      sess = session.Session()

    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_1], [out_tensor])
    self.assertFalse(quantized_converter.post_training_quantize)

    quantized_converter.post_training_quantize = True
    self.assertTrue(quantized_converter.post_training_quantize)
    self.assertEqual(quantized_converter.optimizations, [lite.Optimize.DEFAULT])

    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

  def _getIntegerQuantizeModel(self, num_filters=16):
    # [FUNCTION DOC]
    # HELPER: builds the convolution model shared by the quantization tests.
    # Input: num_filters, the number of convolution filters (default sixteen).
    # Seeds the global random generator with zero, creates (in the current default graph) a float
    # placeholder of shape one by five by five by three, a same-padded convolution with an
    # all-ones filter and a relu named output.
    # Returns a tuple of the input tensor, the output tensor, and a calibration generator
    # function that yields five lists each holding one random uniform float input batch.
    np.random.seed(0)
    inp = array_ops.placeholder(
        dtype=dtypes.float32, shape=(1, 5, 5, 3), name='input')
    conv = nn_ops.conv2d(
        inp,
        filter=array_ops.ones([3, 3, 3, num_filters]),
        strides=[1, 1, 1, 1],
        padding='SAME')
    output = nn_ops.relu(conv, name='output')

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: representative dataset generator for the integer quantization test model.
      # Yields five times a list with one random float array of shape one by five by five by
      # three drawn uniformly between minus one and one.
      for _ in range(5):
        yield [np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)]

    return (inp, output, calibration_gen)

  def testQuantizeInt8AllowFloat(self):
    # [FUNCTION DOC]
    # Scenario: the integer quantization test model converted as float and then with the default
    # optimization plus a representative dataset.
    # Expects conversion metadata in both models: for the float model, the TensorFlow version, API
    # version one, the TF session model type, no custom, select-ops or forced select-ops options
    # and no optimization modes; for the quantized model, the full integer quantization mode.
    # The quantized model keeps float input and output types and is smaller than the float one.
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)
    # Check the conversion metadata.
    metadata = get_conversion_metadata(float_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(
        metadata.environment.tensorflowVersion.decode('utf-8'),
        versions.__version__)
    self.assertEqual(metadata.environment.apiVersion, 1)
    self.assertEqual(metadata.environment.modelType,
                     metadata_fb.ModelType.TF_SESSION)
    self.assertEqual(metadata.options.allowCustomOps, False)
    self.assertEqual(metadata.options.enableSelectTfOps, False)
    self.assertEqual(metadata.options.forceSelectTfOps, False)
    self.assertAllEqual([], metadata.options.modelOptimizationModes)

    # Convert quantized model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)
    # Check the conversion metadata.
    metadata = get_conversion_metadata(quantized_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertAllEqual([metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER],
                        metadata.options.modelOptimizationModes)

    # The default input and output types should be float.
    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.float32, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  @parameterized.named_parameters(
      # Quantize model to Int8
      ('UseTfliteBuiltinsInt', [lite.OpsSet.TFLITE_BUILTINS_INT8],
       [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER]),
      ('UseTfliteBuiltinsInt16', [
          lite.OpsSet
          .EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
      ], [metadata_fb.ModelOptimizationMode.PTQ_INT16]))
  def testQuantizeInt8And16x8(self, supported_ops, expected_opt_modes):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: supported_ops, a list of op sets for the target specification
    # (integer builtins, or 16-bit activation with 8-bit weight builtins); expected_opt_modes,
    # the list of optimization modes expected in the metadata (full integer, or 16-bit integer).
    # Scenario: convert the integer quantization model with the default optimization, the given
    # supported ops and a representative dataset.
    # Expects the same environment and option metadata as a basic session conversion, the
    # expected optimization modes, float input and output types, and a smaller model than float.
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    # Convert model by specifying target spec (instead of optimizations), since
    # when targeting an integer only backend, quantization is mandatory.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.target_spec.supported_ops = supported_ops
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)
    # Check the conversion metadata.
    metadata = get_conversion_metadata(quantized_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(
        metadata.environment.tensorflowVersion.decode('utf-8'),
        versions.__version__)
    self.assertEqual(metadata.environment.apiVersion, 1)
    self.assertEqual(metadata.environment.modelType,
                     metadata_fb.ModelType.TF_SESSION)
    self.assertEqual(metadata.options.allowCustomOps, False)
    self.assertEqual(metadata.options.enableSelectTfOps, False)
    self.assertEqual(metadata.options.forceSelectTfOps, False)
    self.assertAllEqual(expected_opt_modes,
                        metadata.options.modelOptimizationModes)

    # The default input and output types should be float.
    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.float32, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  def testQuantizeInt8InputOutput(self):
    # [FUNCTION DOC]
    # Scenario: the integer quantization model converted with int8 inference input and output
    # types, default optimization and a representative dataset.
    # Expects int8 input and output tensors and a model smaller than the float conversion.
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    # Convert quantized weights model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.inference_input_type = dtypes.int8
    quantized_converter.inference_output_type = dtypes.int8
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # The input and output types should be int8.
    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.int8, input_details[0]['dtype'])
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.int8, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  def testInvalidQuantizeInt8(self):
    # [FUNCTION DOC]
    # Scenario: the large matrix multiplication model with default optimization and supported
    # types restricted to int8, but no representative dataset.
    # Expects a value error saying a representative dataset must be specified for full integer
    # quantization. Seeds the global random generator with zero.
    np.random.seed(0)
    with ops.Graph().as_default():
      # We need the tensor to have more than 1024 elements for quantize_weights
      # to kick in. Thus, the [33, 33] shape.
      in_tensor_1 = array_ops.placeholder(
          shape=[33, 33], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = constant_op.constant(
          np.random.uniform(low=-10., high=10., size=(33, 33)),
          shape=[33, 33],
          dtype=dtypes.float32,
          name='inputB')
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
      sess = session.Session()

    # Attempt to convert to quantized weights model.
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor_1], [out_tensor])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    # Restricting to int8 type only
    quantized_converter.target_spec.supported_types = [dtypes.int8]
    # A representative dataset is required for full fixed point quantization.
    with self.assertRaises(ValueError) as error:
      quantized_converter.convert()
    self.assertEqual(
        'For full integer quantization, a `representative_dataset` '
        'must be specified.', str(error.exception))

  def testQuantizeUInt8(self):
    # [FUNCTION DOC]
    # Scenario: two inputs added then passed through a fake-quantization op (range zero to one),
    # converted with unsigned 8-bit inference type and input stats for both inputs.
    # Expects both inputs to be unsigned 8-bit with scale one and zero point zero, and the output
    # unsigned 8-bit with a positive scale.
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess,
                                                  [in_tensor_1, in_tensor_2],
                                                  [out_tensor])
    converter.inference_type = dtypes.uint8
    converter.quantized_input_stats = {
        'inputA': (0., 1.),
        'inputB': (0., 1.)
    }  # mean, std_dev
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual('inputA', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    self.assertEqual('inputB', input_details[1]['name'])
    self.assertEqual(np.uint8, input_details[1]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
    self.assertEqual((1., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertGreater(output_details[0]['quantization'][0], 0)  # scale

  def testQuantizeUInt8UsingDefaultRangeStats(self):
    # [FUNCTION DOC]
    # Scenario: the float add model converted with unsigned 8-bit inference type, input stats and
    # default range statistics (minimum zero, maximum six) instead of fake-quantization ops.
    # Expects unsigned 8-bit input (scale one, zero point zero) and output (positive scale).
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.inference_type = dtypes.uint8
    converter.quantized_input_stats = {'Placeholder': (0., 1.)}  # mean, std_dev
    converter.default_ranges_stats = (0, 6)  # min, max
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertGreater(output_details[0]['quantization'][0], 0)  # scale

  @parameterized.named_parameters(
      # Quantize to Float16 even if rep data provided.
      ('UseRepresentativeData', True, False, True, False, False, False,
       [metadata_fb.ModelOptimizationMode.PTQ_FLOAT16]),
      # Quantize to Float16 if no rep data provided.
      ('NoRepresentativeData', False, False, True, False, False, False,
       [metadata_fb.ModelOptimizationMode.PTQ_FLOAT16]),
      # Post training quantization if both rep data and int8 included.
      ('SampleDataIncludeInt8', True, True, False, False, True, False,
       [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER]),
      # Same as above, but using MLIR quantizer
      ('SampleDataIncludeInt8Quant', True, True, False, False, True, True,
       [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER]))
  def testQuantizeFloat16(self, use_rep_data, include_int8,
                          is_float16_quantized, is_float16_accumulation,
                          is_post_training_quantized, enable_mlir_quantizer,
                          expected_opt_modes):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: use_rep_data (supply a representative dataset), include_int8
    # (add int8 to the supported types beside float16), is_float16_quantized (expect the bias to be
    # float16), is_float16_accumulation (set the float16 accumulation type), is_post_training_
    # quantized (expect the bias to be int32), enable_mlir_quantizer (use the new quantizer),
    # expected_opt_modes (expected optimization modes in the metadata).
    # Scenario: first converts the integer quantization model as float and checks the bias tensor
    # is float32; then converts with float16 supported types and checks the metadata modes and the
    # type of the convolution bias tensor (float16, or int32, otherwise a value error is raised).
    # KNOWN ISSUE: the checks of the quantized model sit in the else branch of the float16
    # accumulation check, so when accumulation is requested nothing is converted or asserted.
    # Also a stale comment mentions a different bias index although the bias is looked up by name.
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel()
      sess = session.Session()

    bias_idx = 1
    bias_name = 'Conv2D'

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)
    interpreter = Interpreter(model_content=float_tflite_model)
    interpreter.allocate_tensors()
    self.assertEqual(interpreter.get_tensor_details()[bias_idx]['name'],
                     bias_name)
    self.assertEqual(interpreter.get_tensor_details()[bias_idx]['dtype'],
                     dtypes.float32)

    # Convert model to quantized version
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.target_spec.supported_types = [dtypes.float16]
    if include_int8:
      quantized_converter.target_spec.supported_types.append(dtypes.int8)
    if use_rep_data:
      quantized_converter.representative_dataset = calibration_gen
    if is_float16_accumulation:
      quantized_converter.target_spec.experimental_supported_accumulation_type = dtypes.float16  # pylint: disable=line-too-long

    else:
      quantized_tflite_model = quantized_converter.convert()
      self.assertIsNotNone(quantized_tflite_model)
      metadata = get_conversion_metadata(quantized_tflite_model)
      self.assertIsNotNone(metadata)
      self.assertAllEqual(expected_opt_modes,
                          metadata.options.modelOptimizationModes)
      interpreter = Interpreter(model_content=quantized_tflite_model)
      interpreter.allocate_tensors()

      # MLIR quantizer has different bias index.
      bias_name = 'Conv2D'
      bias_tensor = [
          tensor for tensor in interpreter.get_tensor_details()
          if tensor['name'] == bias_name
      ]
      self.assertLen(bias_tensor, 1)

      if is_float16_quantized:
        # Verify that bias constant is float16 type.
        self.assertEqual(bias_tensor[0]['dtype'], dtypes.float16)
      elif is_post_training_quantized:
        # Verify that bias constants is int32 type.
        self.assertEqual(bias_tensor[0]['dtype'], dtypes.int32)
      else:
        raise ValueError('Invalid test options.')

  def testInvalidQuantizeFloat16(self):
    # [FUNCTION DOC]
    # Scenario: float16 supported types combined with supported ops restricted to integer
    # builtins.
    # Expects a value error explaining that full integer quantization was enabled via the
    # supported ops so the supported types must be left unset or be int8 only.
    with ops.Graph().as_default():
      inp, output, _ = self._getIntegerQuantizeModel()
      sess = session.Session()

    # Specify float16 quantization
    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [inp], [output])
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.target_spec.supported_types = [dtypes.float16]
    # Specify only int8 builtin ops
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    with self.assertRaises(ValueError) as error:
      quantized_converter.convert()
    self.assertEqual(
        'As full integer quantization has been enabled by setting '
        '`target_spec.supported_ops`={tf.lite.OpsSet.TFLITE_BUILTINS_INT8}, '
        'thus `target_spec.supported_types` should be left uninitizalized '
        'or set to {tf.int8}.', str(error.exception))

  @parameterized.named_parameters(('InferenceType_INT8', dtypes.int8),
                                  ('InferenceType_UINT8', dtypes.uint8))
  def testInvalidQuantizeQATModelRequiresInputStats(self, quantized_type):
    # [FUNCTION DOC]
    # Parameterized test. Input: quantized_type, either int8 or unsigned 8-bit.
    # Scenario: a model with a fake-quantization op converted with the inference type set to the
    # quantized type and no input stats, then with float inference type but the quantized
    # inference input type and no stats.
    # Expects, for both, a value error telling that the quantized input stats must be defined and
    # showing the current inference type and input type. Finally, with both types set and stats
    # provided for the first input, conversion runs without error.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor + in_tensor, min=0., max=1.)
      sess = session.Session()

    quantized_converter = lite.TFLiteConverter.from_session(
        sess, [in_tensor], [out_tensor])

    with self.assertRaises(ValueError) as error:
      quantized_converter.inference_type = quantized_type
      quantized_converter.convert()
    self.assertEqual(
        'The `quantized_input_stats` flag must be defined when either '
        '`inference_type` flag or `inference_input_type` flag is set to '
        'tf.int8 or tf.uint8. Currently, `inference_type=tf.{}` and '
        '`inference_input_type=None`.'.format(quantized_type.name),
        str(error.exception))

    with self.assertRaises(ValueError) as error:
      quantized_converter.inference_type = dtypes.float32
      quantized_converter.inference_input_type = quantized_type
      quantized_converter.convert()
    self.assertEqual(
        'The `quantized_input_stats` flag must be defined when either '
        '`inference_type` flag or `inference_input_type` flag is set to '
        'tf.int8 or tf.uint8. Currently, `inference_type=tf.float32` and '
        '`inference_input_type=tf.{}`.'.format(quantized_type.name),
        str(error.exception))

    quantized_converter.inference_type = quantized_type
    quantized_converter.inference_input_type = quantized_type

    input_arrays = quantized_converter.get_input_arrays()
    quantized_converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}
    quantized_converter.convert()

  def testInvalidQuantizeQATModelMissingInputStats(self):
    # [FUNCTION DOC]
    # Scenario: a two-input fake-quantized model converted with unsigned 8-bit inference type but
    # input stats given for only one input.
    # Expects a value error saying the stats are not available for the other input tensor.
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess,
                                                  [in_tensor_1, in_tensor_2],
                                                  [out_tensor])
    converter.inference_type = dtypes.uint8
    converter.quantized_input_stats = {'inputA': (0., 1.)}  # mean, std_dev
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual(
        'Quantization input stats are not available for input tensors '
        '\'inputB\'.', str(error.exception))

  def testTrainingTimeAndPostTrainingCalibrateAndQuantize(self):
    # [FUNCTION DOC]
    # Scenario: the integer quantization model converted with training-time quantization flags
    # (int8 inference type, float input and output types, input stats) combined with post-training
    # quantization (default optimization, representative dataset, new quantizer).
    # Expects the result to be smaller than the float model. Then it sets the private calibrate
    # only flag, converts, and quantizes the calibrated model with the MLIR quantize function in
    # full quantization mode; the interpreter must show an int8 input with scale one and zero
    # point zero and an int8 output.
    with ops.Graph().as_default():
      inp, output, calibration_gen = self._getIntegerQuantizeModel()
      sess = session.Session()

    # Convert float model.
    float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    converter = lite.TFLiteConverter.from_session(sess, [inp], [output])

    # extra flags to trigger training time quantization conversion
    converter.inference_type = dtypes.int8
    converter.inference_input_type = dtypes.float32
    converter.inference_output_type = dtypes.float32
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}
    # trigger post-training quantization
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_gen
    converter.experimental_new_quantizer = True
    quantized_tflite_model = converter.convert()
    self.assertIsNotNone(quantized_tflite_model)
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

    # calibration only api
    converter._experimental_calibrate_only = True
    calibrated_tflite = converter.convert()
    quantized_tflite_model = mlir_quantize(
        calibrated_tflite, fully_quantize=True)
    interpreter = Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    self.assertEqual(np.int8, input_details[0]['dtype'])
    self.assertEqual((1., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertEqual(np.int8, output_details[0]['dtype'])

  def testFloatTocoConverter(self):
    # [FUNCTION DOC]
    # Scenario: convert the float add model through the deprecated TocoConverter alias.
    # Expects a non-null result that the interpreter can load and allocate.
    """Tests deprecated test TocoConverter."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Ensure the interpreter is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  def testMultipleOutputNodeNames(self):
    # [FUNCTION DOC]
    # Scenario: a vector of four split into four one-element outputs of one op, all used as
    # converter outputs.
    # Expects four output tensors; after setting the input to one through four and invoking, the
    # outputs equal those values in order.
    """Tests converting a graph with an op that have multiple outputs."""
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(shape=[4], dtype=dtypes.float32)
      out0, out1, out2, out3 = array_ops.split(
          input_tensor, [1, 1, 1, 1], axis=0)
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [input_tensor],
                                                  [out0, out1, out2, out3])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    interpreter.set_tensor(input_details[0]['index'],
                           np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 4)
    self.assertEqual(1.0, interpreter.get_tensor(output_details[0]['index']))
    self.assertEqual(2.0, interpreter.get_tensor(output_details[1]['index']))
    self.assertEqual(3.0, interpreter.get_tensor(output_details[2]['index']))
    self.assertEqual(4.0, interpreter.get_tensor(output_details[3]['index']))

  @test_util.run_in_graph_and_eager_modes
  def testFunctions(self):
    # [FUNCTION DOC]
    # Decorated to run in both graph and eager modes.
    # Scenario: a graph function that adds a variable and a placeholder, followed by a multiply by
    # two named output_node; variables are initialized in a session.
    # Expects conversion to inline the function, and the interpreter to show a float input named
    # input and a float output named output_node, both of shape one, without quantization.
    """Tests tf.function in 1.X."""

    @def_function.function
    def plus_placeholder(x, placeholder):
      # [FUNCTION DOC]
      # HELPER: small traced function used as the model body in the test.
      # Inputs: x and placeholder, tensors. Returns their sum.
      return x + placeholder

    with ops.Graph().as_default():
      placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[1], name='input')
      variable_node = variables.Variable(1.0, name='variable_node')
      defun_node = plus_placeholder(variable_node, placeholder)
      output_node = math_ops.multiply(defun_node, 2.0, name='output_node')

      # Initialize variables in the model.
      sess = session.Session()
      sess.run(variables.variables_initializer([variable_node]))

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [placeholder],
                                                  [output_node])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('output_node', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testInferenceInputOutputTypeFloatDefault(self):
    # [FUNCTION DOC]
    # Scenario: the float add model converted with defaults.
    # Expects float32 input named Placeholder and float32 output named add with the original
    # shape.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])

  def testInferenceInputOutputTypeQuantizedUint8Default(self):
    # [FUNCTION DOC]
    # Scenario: a fake-quantized add model converted with unsigned 8-bit inference type and input
    # stats for the placeholder, with no explicit input or output type overrides.
    # Expects the input named Placeholder and the output named output both to be unsigned 8-bit.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor + in_tensor, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    converter.inference_type = dtypes.uint8
    converter.quantized_input_stats = {'Placeholder': (0., 1.)}  # mean, std_dev
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])

  def testReusingConverterWithDifferentPostTrainingQuantization(self):
    # [FUNCTION DOC]
    # Scenario: one converter instance is converted twice, first with the deprecated post
    # training quantize flag set to true then set to false.
    # Expects both conversions to return non-null models.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor + in_tensor, min=0., max=1., name='output')
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])

    converter.post_training_quantize = True
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    converter.post_training_quantize = False
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

  def testResizeWithShape(self):
    # [FUNCTION DOC]
    # Scenario: a constant two by two list reshaped using the dynamic shape of a placeholder with
    # two unknown dimensions.
    # Expects shape one by one and signature minus one by minus one initially; after resizing the
    # input to a vector of four, allocating and invoking, the int32 output has shape four and
    # holds one through four.
    with ops.Graph().as_default():
      # Construct a graph with a dynamically shapped input and an internal node
      # that relies on the output of that input's shape.
      in_tensor = array_ops.placeholder(
          shape=[None, None], dtype=dtypes.float32)
      in_tensor2 = [[1, 2], [3, 4]]
      out_tensor = array_ops.reshape(in_tensor2, array_ops.shape(in_tensor))
      sess = session.Session()

    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertAllEqual([1, 1], input_details[0]['shape'])
    self.assertAllEqual([-1, -1], input_details[0]['shape_signature'])

    # Resize tensor and invoke.
    interpreter.resize_tensor_input(0, [4])
    interpreter.allocate_tensors()
    interpreter.invoke()

    # The output should be reshaped properly according to the resized input.
    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.int32, output_details[0]['dtype'])
    self.assertAllEqual([4], output_details[0]['shape'])
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertAllEqual([1, 2, 3, 4], output_data)

  def testResizingIntermediateDynamicTensor(self):
    # [FUNCTION DOC]
    # Regression test (linked to a public issue): the output shape of an intermediate dynamic
    # tensor changes between invocations.
    # Scenario: a padding op whose padding amounts are an input, followed by an add with a
    # negated second input. The test invokes with padding of one, then with padding of two.
    # Expects both invocations to succeed without error.
    # This is a regression test for the case where shape of dynamic output
    # tensors changes between invocations.
    # See also https://github.com/tensorflow/tensorflow/issues/26549
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(shape=[1, 1], dtype=dtypes.float32)
      input2_tensor = array_ops.placeholder(shape=[1], dtype=dtypes.float32)

      # The bug is triggered only when dynamic tensor is intermediate. Putting
      # some other ops around it.
      neg = math_ops.negative(input2_tensor)
      padding = array_ops.placeholder(shape=[2, 2], dtype=dtypes.int32)
      output_tensor = array_ops.pad(input_tensor, padding) + neg

      sess = session.Session()

    converter = lite.TFLiteConverter.from_session(
        sess, [input_tensor, padding, input2_tensor], [output_tensor])
    tflite_model = converter.convert()

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[1]['index'],
                           np.array([[1, 1], [1, 1]], dtype=np.int32))
    interpreter.invoke()

    # Without the fix, invocation will fail when changing the shape of
    # intermediate dynamic tensors.
    interpreter.set_tensor(input_details[1]['index'],
                           np.array([[2, 2], [2, 2]], dtype=np.int32))
    interpreter.invoke()

  def testGraphDebugInfo(self):
    # [FUNCTION DOC]
    # Scenario: same graph function model as the functions test, converted from a session.
    # Expects the converter's debug info to be valid (it mentions this test file and not the V2
    # test file) and to include the add node inside the inlined function, named with the
    # function's signature name.
    """Test a session has debug info captured."""

    @def_function.function
    def plus_placeholder(x, placeholder):
      # [FUNCTION DOC]
      # HELPER: small traced function used as the model body in the test.
      # Inputs: x and placeholder, tensors. Returns their sum.
      return x + placeholder

    with ops.Graph().as_default():
      placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[1], name='input')
      variable_node = variables.Variable(1.0, name='variable_node')
      defun_node = plus_placeholder(variable_node, placeholder)
      output_node = math_ops.multiply(defun_node, 2.0, name='output_node')

      # Initialize variables in the model.
      sess = session.Session()
      sess.run(variables.variables_initializer([variable_node]))

    converter = lite.TFLiteConverter.from_session(sess, [placeholder],
                                                  [output_node])
    converter.convert()
    self.assertValidDebugInfo(converter._debug_info)

    # Check the add node in the inlined function is included.
    func = sess.graph.as_graph_def().library.function[0].signature.name
    self.assertIn(('add@' + func), repr(converter._debug_info))

  def testOutputOnlyModel(self):
    # [FUNCTION DOC]
    # Scenario: a model that has no inputs and only a random-normal output, with builtin and
    # select TensorFlow ops allowed.
    # Expects an empty input list to be considered valid and conversion to succeed.
    with ops.Graph().as_default():
      out_tensor = random_ops.random_normal(shape=[3])
      sess = session.Session()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_session(sess, [], [out_tensor])
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]

    # Empty input array is a valid input.
    self.assertTrue(converter._has_valid_tensors())

    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)


class FromFrozenGraphFile(LiteTest):
  # [CLASS DOC]
  # Tests for converting from a frozen graph definition file (binary or text format) through the
  # from frozen graph factory, including shape overrides, invalid files, deprecated alias and
  # conversion metadata exclusion. Each test writes a model file into the test temporary
  # directory.

  def testFloat(self):
    # [FUNCTION DOC]
    # Scenario: write a float add graph to a binary graph file and convert it from that file with
    # input Placeholder and output add.
    # Expects float input and output of shape one by sixteen by sixteen by three, names
    # Placeholder and add, and no quantization.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                       ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testFloatWithShapesArray(self):
    # [FUNCTION DOC]
    # Scenario: frozen graph file whose input has an unknown batch dimension, converted with an
    # input shapes mapping that sets the shape to two by sixteen by sixteen by three.
    # Expects the interpreter input shape to equal the override.
    """Test a shape overriding case."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, ['Placeholder'], ['add'],
        input_shapes={'Placeholder': [2, 16, 16, 3]})
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])

  def testInvalidShapesArray(self):
    # [FUNCTION DOC]
    # Scenario: input shapes mapping names a tensor that is not an input of the frozen graph.
    # Expects the factory to raise a value error.
    """Test an invalid shape overriding case, which has a wrong input name."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    with self.assertRaises(ValueError):
      lite.TFLiteConverter.from_frozen_graph(
          graph_def_file, ['Placeholder'], ['add'],
          input_shapes={'wrong_input': [2, 16, 16, 3]})

  def testPartialShapesArray(self):
    # [FUNCTION DOC]
    # Scenario: a frozen graph with two inputs a and b, and a shape override given only for a.
    # Expects input a to take the override shape and input b to take batch size one with its
    # other dimensions unchanged.
    """Test a shape overriding case, with the only one input among two."""
    with ops.Graph().as_default():
      a = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='a')
      b = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='b')
      _ = math_ops.add(a, b, name='add')
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, ['a', 'b'], ['add'], input_shapes={'a': [2, 16, 16, 3]})
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])
    self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])

  def testFreezeGraph(self):
    # [FUNCTION DOC]
    # Scenario: a graph file that still contains an unfrozen variable.
    # Expects the factory to raise a value error telling to freeze the graph with the freeze
    # graph tool.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      var = variable_scope.get_variable(
          'weights', shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + var
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Ensure the graph with variables cannot be converted.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'],
                                             ['add'])
    self.assertEqual('Please freeze the graph using freeze_graph.py.',
                     str(error.exception))

  def testPbtxt(self):
    # [FUNCTION DOC]
    # Scenario: the float add graph written in text format to a file with the text extension and
    # converted from it.
    # Expects the same tensor details as the binary format test.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pbtxt')
    write_graph(sess.graph_def, '', graph_def_file, True)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                       ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('add', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testInvalidFileNotFound(self):
    # [FUNCTION DOC]
    # Scenario: convert from a file path that does not exist.
    # Expects an I/O error with the message that the file does not exist.
    with self.assertRaises(IOError) as error:
      lite.TFLiteConverter.from_frozen_graph('invalid_file', ['Placeholder'],
                                             ['add'])
    self.assertEqual('File \'invalid_file\' does not exist.',
                     str(error.exception))

  def testInvalidFileBadData(self):
    # [FUNCTION DOC]
    # Scenario: a file in the test temporary directory containing garbage text is used as the
    # graph file. Side effect: creates that file.
    # Expects an I/O error saying the input file could not be parsed.
    # KNOWN ISSUE: the file is opened in binary mode but a text string is written, which would
    # fail under Python 3 before the converter is reached; the intended scenario relies on older
    # behavior.
    graph_def_file = os.path.join(self.get_temp_dir(), 'invalid_file')
    with gfile.Open(graph_def_file, 'wb') as temp_file:
      temp_file.write('bad data')
      temp_file.flush()

    # Attempts to convert the invalid model.
    with self.assertRaises(IOError) as error:
      lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'],
                                             ['add'])
    self.assertEqual(
        'Unable to parse input file \'{}\'.'.format(graph_def_file),
        str(error.exception))

  def testFloatTocoConverter(self):
    # [FUNCTION DOC]
    # Scenario: convert the float add graph file through the deprecated TocoConverter alias.
    # Expects a non-null result that the interpreter can load and allocate.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_frozen_graph(graph_def_file,
                                                     ['Placeholder'], ['add'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Ensure the model is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  def testGraphDebugInfo(self):
    # [FUNCTION DOC]
    # Scenario: convert a frozen graph file with the deprecated alias.
    # Expects the converter's debug info to be empty (false-like) since frozen graphs carry none.
    """Test a frozen graph doesn't have debug info captured."""
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_frozen_graph(graph_def_file,
                                                     ['Placeholder'], ['add'])
    converter.convert()
    # GraphDebugInfo should be none for frozen graph.
    self.assertFalse(converter._debug_info)

  def testExcludeConversionMetadata(self):
    # [FUNCTION DOC]
    # Scenario: convert a frozen graph file after setting the exclude conversion metadata flag.
    # Expects a non-null model that contains no conversion metadata.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                       ['Placeholder'], ['add'])
    converter.exclude_conversion_metadata = True
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)
    # Check the conversion metadata.
    metadata = get_conversion_metadata(tflite_model)
    self.assertIsNone(metadata)


class FromFrozenGraphObjectDetection(LiteTest):
  # [CLASS DOC]
  # Tests that use a prebuilt object detection (SSD MobileNet) frozen graph test data file,
  # which cannot be loaded in TensorFlow directly because of the custom detection post-process
  # op, plus a test with a control output on a print op. Instance state is set by the
  # initialization helper: graph file path, input arrays, output arrays and input shapes.

  def _initObjectDetectionArgs(self):
    # [FUNCTION DOC]
    # HELPER: sets up the shared arguments for the object detection model tests.
    # Looks for the test graph file in the test data directory, falling back to a sibling
    # directory of the resource root; raises an I/O error if neither exists.
    # Sets instance attributes: graph definition file path, input array names (the normalized
    # input image tensor), four output array names (the detection post-process op and its
    # outputs one to three) and an input shapes mapping of one by 300 by 300 by three.
    # Returns nothing.
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
    self._input_arrays = ['normalized_input_image_tensor']
    self._output_arrays = [
        'TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1',
        'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3'
    ]
    self._input_shapes = {'normalized_input_image_tensor': [1, 300, 300, 3]}

  def testTFLiteGraphDef(self):
    # [FUNCTION DOC]
    # Scenario: convert the object detection frozen graph with custom ops allowed.
    # Expects one float input of shape one by 300 by 300 by three and four outputs with the
    # detection post-process names and shapes (one by ten by four, one by ten, one by ten, one).
    # Tests the object detection model that cannot be loaded in TensorFlow.
    self._initObjectDetectionArgs()

    converter = lite.TFLiteConverter.from_frozen_graph(self._graph_def_file,
                                                       self._input_arrays,
                                                       self._output_arrays,
                                                       self._input_shapes)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('normalized_input_image_tensor', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 300, 300, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 4)
    self.assertEqual('TFLite_Detection_PostProcess', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 10, 4], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual('TFLite_Detection_PostProcess:1',
                     output_details[1]['name'])
    self.assertAllEqual([1, 10], output_details[1]['shape'])
    self.assertEqual('TFLite_Detection_PostProcess:2',
                     output_details[2]['name'])
    self.assertAllEqual([1, 10], output_details[2]['shape'])
    self.assertEqual('TFLite_Detection_PostProcess:3',
                     output_details[3]['name'])
    self.assertAllEqual([1], output_details[3]['shape'])

  def testTFLiteGraphDefWithControlOutput(self):
    # [FUNCTION DOC]
    # Scenario: an add whose result feeds a print op; the converter is built directly from the
    # graph definition with input shapes and the print op set as a private control output, with
    # builtin and select TensorFlow ops allowed.
    # Expects the converted model operator codes to be add, then custom string-format and
    # print-v2 flex ops; the interpreter shows one float input of shape five by five and no
    # outputs.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[5, 5], dtype=dtypes.float32, name='input')
      out_tensor = in_tensor + in_tensor
      logging_ops.print_v2(out_tensor)
      sess = session.Session()

    converter = lite.TFLiteConverter(
        sess.graph_def,
        input_tensors=None,
        output_tensors=None,
        input_arrays_with_shape=[('input', [5, 5])],
        output_arrays=None,
        experimental_debug_info_func=None)
    converter._control_output_arrays = ['PrintV2']
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    model = util._convert_model_from_bytearray_to_object(tflite_model)
    self.assertEqual(model.operatorCodes[0].builtinCode,
                     schema_fb.BuiltinOperator.ADD)
    self.assertEqual(model.operatorCodes[1].builtinCode,
                     schema_fb.BuiltinOperator.CUSTOM)
    self.assertEqual(model.operatorCodes[1].customCode, b'FlexStringFormat')
    self.assertEqual(model.operatorCodes[2].builtinCode,
                     schema_fb.BuiltinOperator.CUSTOM)
    self.assertEqual(model.operatorCodes[2].customCode, b'FlexPrintV2')

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([5, 5], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 0)

  def testModifyIOToUint8(self):
    # [FUNCTION DOC]
    # Scenario: quantize the object detection model with integer builtin ops, int8 inference type,
    # unsigned 8-bit input and output types, the new quantizer, input stats and custom ops
    # allowed, using a small random representative dataset.
    # Checks the converted model's quantize ops: any whose output is a graph output must take a
    # float32 input. Conversion must be non-null.
    # KNOWN ISSUE: the loop body only asserts for quantize ops that produce graph outputs; if
    # none exist the test asserts nothing about types.
    # Tests the object detection model that cannot be loaded in TensorFlow.
    self._initObjectDetectionArgs()

    def representative_dataset_gen():
      # [FUNCTION DOC]
      # HELPER: representative dataset generator for the object detection quantization test.
      # Yields twice a list holding one random float array of shape one by 300 by 300 by three
      # with values between zero and one.
      for _ in range(2):
        yield [
            np.random.uniform(low=0, high=1,
                              size=(1, 300, 300, 3)).astype(np.float32)
        ]

    converter = lite.TFLiteConverter.from_frozen_graph(self._graph_def_file,
                                                       self._input_arrays,
                                                       self._output_arrays,
                                                       self._input_shapes)
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = {lite.OpsSet.TFLITE_BUILTINS_INT8}
    converter.inference_type = dtypes.int8
    converter.inference_input_type = dtypes.uint8
    converter.inference_output_type = dtypes.uint8
    converter.experimental_new_quantizer = True
    converter.quantized_input_stats = {
        'normalized_input_image_tensor': (0., 1.)
    }  # mean, std_dev
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    self.assertIsNotNone(tflite_model)

    model = util._convert_model_from_bytearray_to_object(tflite_model)
    quant_opcode_idxs = util.get_quantize_opcode_idx(model)

    subgraph = model.subgraphs[0]
    tensors = subgraph.tensors
    operators = subgraph.operators
    for op in operators:
      if op.opcodeIndex in quant_opcode_idxs:
        input_type = util._convert_tflite_enum_type_to_tf_type(
            tensors[op.inputs[0]].type)
        if op.outputs[0] in subgraph.outputs:
          self.assertEqual(input_type, dtypes.float32)


class FromSavedModelTest(TestModels):
  # [CLASS DOC]
  # Tests for converting a SavedModel directory through the from saved model factory: default
  # conversion, unknown batch size, input array ordering, shape overrides, invalid and partial
  # input specifications, the deprecated alias and a skipped debug info test. Models are created
  # by the helper that saves a two-input add model.

  def _createSavedModel(self, shape):
    # [FUNCTION DOC]
    # HELPER: builds and saves the simple model used by the SavedModel tests.
    # Input: shape, the shape of both float placeholders (named inputB and inputA).
    # Side effect: writes a saved model with inputs x and y and output z (the sum) to a
    # directory in the test temporary directory.
    # Returns the path of that directory.
    """Create a simple SavedModel."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor_1 = array_ops.placeholder(
            shape=shape, dtype=dtypes.float32, name='inputB')
        in_tensor_2 = array_ops.placeholder(
            shape=shape, dtype=dtypes.float32, name='inputA')
        out_tensor = in_tensor_1 + in_tensor_2
        inputs = {'x': in_tensor_1, 'y': in_tensor_2}
        outputs = {'z': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return saved_model_dir

  def testSimpleModel(self):
    # [FUNCTION DOC]
    # Scenario: convert the simple saved two-input add model.
    # Expects two float inputs (names starting with inputA then inputB, i.e. sorted) and one float
    # output whose name starts with add, all of shape one by sixteen by sixteen by three, with no
    # quantization.
    """Test a SavedModel."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertStartsWith(input_details[0]['name'], 'inputA')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertStartsWith(input_details[1]['name'], 'inputB')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertStartsWith(output_details[0]['name'], 'add')
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testNoneBatchSize(self):
    # [FUNCTION DOC]
    # Scenario: saved model whose inputs have an unknown batch dimension.
    # Expects the converted tensors to have batch size one and the same details as the simple
    # model test.
    """Test a SavedModel, with None in input tensor's shape."""
    saved_model_dir = self._createSavedModel(shape=[None, 16, 16, 3])

    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertStartsWith(input_details[0]['name'], 'inputA')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertStartsWith(input_details[1]['name'], 'inputB')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertStartsWith(output_details[0]['name'], 'add')
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testOrderInputArrays(self):
    # [FUNCTION DOC]
    # Scenario: convert the saved model giving the input arrays in the order inputB, inputA.
    # Expects the interpreter inputs to still be listed as inputA then inputB, float, with the
    # original shape.
    # KNOWN ISSUE: the name suggests the given order is applied, but the assertions show the
    # inputs come out sorted alphabetically regardless.
    """Test a SavedModel ordering of input arrays."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir, input_arrays=['inputB', 'inputA'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertStartsWith(input_details[0]['name'], 'inputA')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertStartsWith(input_details[1]['name'], 'inputB')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertStartsWith(output_details[0]['name'], 'add')
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testShapeOverriding(self):
    # [FUNCTION DOC]
    # Scenario: saved model with unknown batch size converted with an input shapes mapping that
    # sets both inputs to two by sixteen by sixteen by three.
    # Expects both inputs and the output to have the overridden shape.
    """Test a SavedModel with the input_shapes arugment."""
    saved_model_dir = self._createSavedModel(shape=[None, 16, 16, 3])

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir,
        input_shapes={
            'inputA': [2, 16, 16, 3],
            'inputB': [2, 16, 16, 3]
        })
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertStartsWith(input_details[0]['name'], 'inputA')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertStartsWith(input_details[1]['name'], 'inputB')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([2, 16, 16, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertStartsWith(output_details[0]['name'], 'add')
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([2, 16, 16, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

  def testWrongInputShapes(self):
    # [FUNCTION DOC]
    # Scenario: input shapes mapping names a tensor that is not among the chosen input array.
    # Expects the factory to raise a value error.
    """Test a SavedModel with a wrong name in the input_shapes argument."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Check case where input shape is given.
    with self.assertRaises(ValueError):
      lite.TFLiteConverter.from_saved_model(
          saved_model_dir,
          input_arrays=['inputA'],
          input_shapes={'wrong_input': [1, 16, 16, 3]})

  def testSubsetInputShaapes(self):
    # [FUNCTION DOC]
    # Scenario: only one of the two inputs is selected, once with a shape given and once with a
    # null shape.
    # Expects conversion to raise a converter error in both cases, since the input is only
    # partially specified.
    # The method name contains a typo (Shaapes) but is the real test name.
    """Test a SavedModel with a subset of the input array names of the model."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Check case where input shape is given.
    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir,
        input_arrays=['inputA'],
        input_shapes={'inputA': [1, 16, 16, 3]})

    # Since we only partially specify the input, this is not allowed.
    with self.assertRaises(ConverterError):
      _ = converter.convert()

    # Check case where input shape is None.
    converter = lite.TFLiteConverter.from_saved_model(
        saved_model_dir, input_arrays=['inputA'], input_shapes={'inputA': None})

    # Since we only partially specify the input, this is not allowed.
    with self.assertRaises(ConverterError):
      _ = converter.convert()

  def testSimpleModelTocoConverter(self):
    # [FUNCTION DOC]
    # Scenario: convert the saved model through the deprecated TocoConverter alias.
    # Expects a non-null result that the interpreter can load and allocate.
    """Test a SavedModel with deprecated TocoConverter."""
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])

    # Convert model and ensure model is not None.
    converter = lite.TocoConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Ensure the model is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  def testGraphDebugInfo(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test skips itself immediately (the debug info comes from the saved model
    # loader, not the created model, per a tracked bug), so the code below never runs.
    # Intended scenario: convert the saved model and validate the converter's debug info.
    """Test a SavedModel has debug info captured."""
    self.skipTest(
        'b/221093690: The debug info is not from self._createSavedModel(), '
        'but from saved_model.loader_impl().')
    saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.convert()
    self.assertValidDebugInfo(converter._debug_info)


class MyAddLayer(keras.layers.Layer):
  # [CLASS DOC]
  # HELPER: custom Keras layer used by the Keras file tests to check that custom objects are
  # honored when loading a saved model file.
  # Adds a fixed increment to its inputs. State: the private increment value. Created with the
  # increment and passed as a custom object by name when converting.

  def __init__(self, increment, **kwargs):
    # [FUNCTION DOC]
    # Inputs: increment, the number added to inputs; extra keyword arguments are passed to the
    # parent layer constructor. Stores the increment on the instance.
    super(MyAddLayer, self).__init__(**kwargs)
    self._increment = increment

  def call(self, inputs):
    # [FUNCTION DOC]
    # Input: inputs, a tensor. Returns the tensor with the stored increment added to it.
    return inputs + self._increment

  def get_config(self):
    # [FUNCTION DOC]
    # Returns the parent layer configuration mapping extended with an entry named increment
    # holding the stored increment, so the layer can be reloaded from a saved file.
    config = super(MyAddLayer, self).get_config()
    config['increment'] = self._increment
    return config


class FromKerasFile(TestModels, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for converting a Keras HDF5 model file through the from Keras model file factory:
  # sequential, functional, custom layer, multiple inputs, input and output array selection,
  # shape overrides, the deprecated alias and a skipped debug info test.
  # Instance state: the path of the saved Keras file (removed at tear down) and the custom
  # objects mapping. Several tests run in both graph and eager mode via named parameters.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, resets the Keras file path and custom objects to nothing, and when
    # not executing eagerly clears the Keras backend session (global Keras state).
    super(FromKerasFile, self).setUp()
    self._keras_file = None
    self._custom_objects = None
    if not context.executing_eagerly():
      keras.backend.clear_session()

  def tearDown(self):
    # [FUNCTION DOC]
    # Deletes the saved Keras file from disk if one was created, then calls the parent tear down.
    if self._keras_file:
      os.remove(self._keras_file)
    super(FromKerasFile, self).tearDown()

  def _getSequentialModel(self, include_custom_layer=False):
    # [FUNCTION DOC]
    # HELPER: creates and saves the sequential Keras model used by many tests.
    # Input: include_custom_layer (default false); when true the custom add layer is inserted.
    # Builds a model of a dense layer with two units on a three-feature input, an optional custom
    # layer, a repeat-vector layer and a time-distributed dense layer; compiles, trains on one
    # random batch and predicts once. Saves it to a new temporary file with the HDF5 extension and
    # stores the path on the instance; the file descriptor from creating it is always closed.
    # When the custom layer is included, stores the custom objects mapping on the instance.
    # Returns nothing.
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_shape=(3,)))
    if include_custom_layer:
      model.add(MyAddLayer(1.0))
    model.add(keras.layers.RepeatVector(3))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
    model.compile(
        loss=keras.losses.MSE,
        optimizer='sgd',
        metrics=[keras.metrics.categorical_accuracy],
        sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    model.predict(x)

    try:
      fd, self._keras_file = tempfile.mkstemp('.h5')
      keras.models.save_model(model, self._keras_file)
    finally:
      os.close(fd)

    if include_custom_layer:
      self._custom_objects = {'MyAddLayer': MyAddLayer}

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testSequentialModel(self, test_context):
    # [FUNCTION DOC]
    # Parameterized test. Input: test_context, a context manager factory for graph mode or eager
    # mode.
    # Scenario: inside that mode, build, save and convert the sequential Keras model.
    # Expects one float input whose name ends with dense_input (shape one by three) and one float
    # output (shape one by three by three), and the interpreter result for a sample input to match
    # the reloaded Keras model prediction to five decimal places.
    """Test a Sequential tf.keras model with default inputs."""
    with test_context():
      self._getSequentialModel()

      converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
      tflite_model = converter.convert()
      self.assertIsNotNone(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEndsWith(input_details[0]['name'], 'dense_input')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 3, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(self._keras_file)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testCustomLayer(self, test_context):
    # [FUNCTION DOC]
    # Parameterized test. Input: test_context, a context manager factory for graph mode or eager
    # mode.
    # Scenario: sequential model containing the custom add layer, converted with the custom
    # objects mapping passed to the factory.
    # Expects the interpreter output for a sample input to match the reloaded Keras model
    # (loaded with the same custom objects) to five decimal places.
    # The docstring is copied from the sequential test and does not mention the custom layer.
    """Test a Sequential tf.keras model with default inputs."""
    with test_context():
      self._getSequentialModel(include_custom_layer=True)

      converter = lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, custom_objects=self._custom_objects)
      tflite_model = converter.convert()
      self.assertIsNotNone(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(
        self._keras_file, custom_objects=self._custom_objects)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  def testSequentialModelInputArray(self):
    # [FUNCTION DOC]
    # Side effect: disables eager execution globally for the process.
    # Scenario: convert the saved sequential model choosing input arrays.
    # Expects an invalid input name to raise a value error saying invalid tensors were found, and
    # the valid name dense_input to convert successfully.
    """Test a Sequential tf.keras model testing input arrays argument."""
    ops.disable_eager_execution()
    self._getSequentialModel()

    # Invalid input array raises error.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, input_arrays=['invalid-input'])
    self.assertEqual("Invalid tensors 'invalid-input' were found.",
                     str(error.exception))

    # Valid input array.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, input_arrays=['dense_input'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

  def testSequentialModelInputShape(self):
    # [FUNCTION DOC]
    # Scenario: convert the saved sequential model with an input shapes mapping.
    # Expects an unknown input name to raise a value error saying the tensor was found in the
    # shapes map but is invalid; a mapping for dense_input of two by three converts and the
    # interpreter input has that shape and a name ending with dense_input.
    """Test a Sequential tf.keras model testing input shapes argument."""
    self._getSequentialModel()

    # Passing in shape of invalid input array raises error.
    with self.assertRaises(ValueError) as error:
      converter = lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, input_shapes={'invalid-input': [2, 3]})
    self.assertEqual(
        "Invalid tensor 'invalid-input' found in tensor shapes map.",
        str(error.exception))

    # Passing in shape of valid input array.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, input_shapes={'dense_input': [2, 3]})
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check input shape from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEndsWith(input_details[0]['name'], 'dense_input')
    self.assertAllEqual([2, 3], input_details[0]['shape'])

  def testSequentialModelOutputArray(self):
    # [FUNCTION DOC]
    # Side effect: disables eager execution globally for the process.
    # Scenario: convert the saved sequential model choosing output arrays.
    # Expects an invalid output name to raise a value error saying invalid tensors were found, and
    # the valid reshape output of the time-distributed layer to convert successfully.
    """Test a Sequential tf.keras model testing output arrays argument."""
    ops.disable_eager_execution()
    self._getSequentialModel()

    # Invalid output array raises error.
    with self.assertRaises(ValueError) as error:
      lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, output_arrays=['invalid-output'])
    self.assertEqual("Invalid tensors 'invalid-output' were found.",
                     str(error.exception))

    # Valid output array.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, output_arrays=['time_distributed/Reshape_1'])
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testFunctionalModel(self, test_context):
    # [FUNCTION DOC]
    # Parameterized test. Input: test_context, a context manager factory for graph mode or eager
    # mode.
    # Scenario: inside that mode, build a functional model of two dense layers on an input named
    # input, train on one batch, save to a temporary HDF5 file (stored on the instance for clean
    # up) and convert.
    # Expects a float input named input (shape one by three), a float output (shape one by three)
    # and numerical agreement with the reloaded Keras model to five decimal places.
    """Test a Functional tf.keras model with default inputs."""
    with test_context():
      inputs = keras.layers.Input(shape=(3,), name='input')
      x = keras.layers.Dense(2)(inputs)
      output = keras.layers.Dense(3)(x)

      model = keras.models.Model(inputs, output)
      model.compile(
          loss=keras.losses.MSE,
          optimizer='sgd',
          metrics=[keras.metrics.categorical_accuracy])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      model.predict(x)
      fd, self._keras_file = tempfile.mkstemp('.h5')
      try:
        keras.models.save_model(model, self._keras_file)
      finally:
        os.close(fd)

      # Convert to TFLite model.
      converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
      tflite_model = converter.convert()
      self.assertIsNotNone(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(self._keras_file)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  def _getFunctionalModelMultipleInputs(self):
    # [FUNCTION DOC]
    # HELPER: creates and saves a functional Keras model with two inputs and two outputs.
    # Inputs a and b (three features each) share one dense layer with four units; the first
    # output is the dense result of b and the second is dropout applied to the dense result of a.
    # Compiles, trains on one random batch of ten, predicts, and saves to a new temporary HDF5
    # file whose path is stored on the instance (file descriptor is always closed).
    # Returns nothing.
    a = keras.layers.Input(shape=(3,), name='input_a')
    b = keras.layers.Input(shape=(3,), name='input_b')
    dense = keras.layers.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = keras.layers.Dropout(0.5, name='dropout')(c)

    model = keras.models.Model([a, b], [d, e])
    model.compile(
        loss=keras.losses.MSE,
        optimizer='sgd',
        metrics=[keras.metrics.mae],
        loss_weights=[1., 0.5])

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))
    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))
    model.train_on_batch([input_a_np, input_b_np], [output_d_np, output_e_np])

    model.predict([input_a_np, input_b_np], batch_size=5)
    fd, self._keras_file = tempfile.mkstemp('.h5')
    try:
      keras.models.save_model(model, self._keras_file)
    finally:
      os.close(fd)

  def testFunctionalModelMultipleInputs(self):
    # [FUNCTION DOC]
    # Scenario: convert the saved two-input two-output functional model.
    # Expects two float inputs with names ending in input_a and input_b (shape one by three) and
    # two float outputs (shape one by four), with no quantization.
    """Test a Functional tf.keras model with multiple inputs and outputs."""
    self._getFunctionalModelMultipleInputs()

    # Convert to TFLite model.
    converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEndsWith(input_details[0]['name'], 'input_a')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEndsWith(input_details[1]['name'], 'input_b')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([1, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 2)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 4], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual(np.float32, output_details[1]['dtype'])
    self.assertAllEqual([1, 4], output_details[1]['shape'])
    self.assertEqual((0., 0.), output_details[1]['quantization'])

  def testShapeOverriding(self):
    # [FUNCTION DOC]
    # Scenario: convert the two-input model with shape overrides for both inputs.
    # Expects both inputs to have shape two by three and both outputs shape two by four.
    # The overrides are given as sets of numbers rather than lists, which the converter
    # nonetheless accepts.
    """Test a Functional tf.keras model with input shape overriding."""
    self._getFunctionalModelMultipleInputs()

    # Convert to TFLite model.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, input_shapes={
            'input_a': {2, 3},
            'input_b': {2, 3}
        })
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEndsWith(input_details[0]['name'], 'input_a')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([2, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEndsWith(input_details[1]['name'], 'input_b')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([2, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 2)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([2, 4], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual(np.float32, output_details[1]['dtype'])
    self.assertAllEqual([2, 4], output_details[1]['shape'])
    self.assertEqual((0., 0.), output_details[1]['quantization'])

  def testPartialShapeOverriding(self):
    # [FUNCTION DOC]
    # Scenario: convert the two-input model overriding only the first input's shape.
    # Expects the first input to be two by three and the second to remain one by three; the
    # first output has shape one by four and the second two by four.
    """Test a Functional tf.keras model with partial input shape overriding."""
    self._getFunctionalModelMultipleInputs()

    # Convert to TFLite model.
    converter = lite.TFLiteConverter.from_keras_model_file(
        self._keras_file, input_shapes={'input_a': {2, 3}})
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEndsWith(input_details[0]['name'], 'input_a')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([2, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    self.assertEndsWith(input_details[1]['name'], 'input_b')
    self.assertEqual(np.float32, input_details[1]['dtype'])
    self.assertAllEqual([1, 3], input_details[1]['shape'])
    self.assertEqual((0., 0.), input_details[1]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 2)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 4], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    self.assertEqual(np.float32, output_details[1]['dtype'])
    self.assertAllEqual([2, 4], output_details[1]['shape'])
    self.assertEqual((0., 0.), output_details[1]['quantization'])

  def testWrongShapeOverriding(self):
    # [FUNCTION DOC]
    # Scenario: shape override given for an input name that does not exist.
    # Expects the factory to raise a value error.
    """Test a Functional tf.keras model with wrong input shape overriding."""
    self._getFunctionalModelMultipleInputs()

    # Convert to TFLite model.
    with self.assertRaises(ValueError):
      lite.TFLiteConverter.from_keras_model_file(
          self._keras_file, input_shapes={'wrong_input': {2, 3}})

  def testFunctionalSequentialModel(self):
    # [FUNCTION DOC]
    # Scenario: a sequential model rewrapped as a functional model using its input and output,
    # trained, saved to a temporary HDF5 file (path stored on the instance) and converted.
    # Expects a float input with name ending in dense_input (shape one by three), a float output
    # (shape one by three by three) and agreement with the reloaded Keras model to five decimals.
    """Test a Functional tf.keras model containing a Sequential model."""
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_shape=(3,)))
    model.add(keras.layers.RepeatVector(3))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
    model = keras.models.Model(model.input, model.output)

    model.compile(
        loss=keras.losses.MSE,
        optimizer='sgd',
        metrics=[keras.metrics.categorical_accuracy],
        sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    model.predict(x)

    model.predict(x)
    fd, self._keras_file = tempfile.mkstemp('.h5')
    try:
      keras.models.save_model(model, self._keras_file)
    finally:
      os.close(fd)

    # Convert to TFLite model.
    converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check tensor details of converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEndsWith(input_details[0]['name'], 'dense_input')
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1, 3], input_details[0]['shape'])
    self.assertEqual((0., 0.), input_details[0]['quantization'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1, 3, 3], output_details[0]['shape'])
    self.assertEqual((0., 0.), output_details[0]['quantization'])

    # Check inference of converted model.
    input_data = np.array([[1, 2, 3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    keras_model = keras.models.load_model(self._keras_file)
    keras_result = keras_model.predict(input_data)

    np.testing.assert_almost_equal(tflite_result, keras_result, 5)

  def testSequentialModelTocoConverter(self):
    # [FUNCTION DOC]
    # Scenario: convert the saved sequential model through the deprecated TocoConverter alias.
    # Expects a non-null result that the interpreter can load and allocate.
    """Test a Sequential tf.keras model with deprecated TocoConverter."""
    self._getSequentialModel()

    converter = lite.TocoConverter.from_keras_model_file(self._keras_file)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Ensure the model is able to load.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

  @parameterized.named_parameters(('_graph', context.graph_mode),
                                  ('_eager', context.eager_mode))
  def testGraphDebugInfo(self, test_context):
    # [FUNCTION DOC]
    # Parameterized test. Input: test_context, graph or eager mode context factory.
    # KNOWN ISSUE: the test skips itself immediately (a bug that cannot be fixed in the open
    # source build), so the remaining code never runs.
    # Intended scenario: convert the sequential Keras model and validate the debug info.
    """Test a Sequential tf.keras model has debug info captured."""
    self.skipTest('TODO(b/291005679): will not be able to fix on OSS')
    with test_context():
      self._getSequentialModel()
      converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
      converter.convert()
      self.assertValidDebugInfo(converter._debug_info)


class SparsityTest(TestModels):
  # [CLASS DOC]
  # Tests for the experimental sparsity optimization: converts a small matrix multiplication
  # model whose constant matrix has chosen values and checks the optimization modes recorded in
  # the conversion metadata (random sparsity, block sparsity, with or without dynamic range
  # quantization).

  def _getSparsificableModel(self, matrix_b_values):
    # [FUNCTION DOC]
    # HELPER: builds the model for the sparsity tests.
    # Input: matrix_b_values, 32 numbers forming the constant four by eight matrix.
    # Creates a session graph multiplying a sixteen by four float placeholder named input1 by that
    # constant. Returns a tuple of the session, a list with the input tensor and a list with the
    # output tensor.
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[16, 4], dtype=dtypes.float32, name='input1')
      in_tensor_2 = constant_op.constant(
          matrix_b_values, shape=[4, 8], dtype=dtypes.float32)
      out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2)
      sess = session.Session()

    return (sess, [in_tensor_1], [out_tensor])

  def testRandomSparsity(self):
    # [FUNCTION DOC]
    # Scenario: a scattered pattern of zeros and ones in the constant matrix, converted with the
    # experimental sparsity optimization.
    # Expects a non-null model whose metadata lists exactly the random sparsity mode.
    matrix_b_values = [
        0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1
    ]
    sess, inputs, outputs = self._getSparsificableModel(matrix_b_values)
    float_converter = lite.TFLiteConverter.from_session(sess, inputs, outputs)
    float_converter.optimizations = [lite.Optimize.EXPERIMENTAL_SPARSITY]
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)
    # Check the conversion metadata.
    metadata = get_conversion_metadata(float_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertAllEqual([metadata_fb.ModelOptimizationMode.RANDOM_SPARSITY],
                        metadata.options.modelOptimizationModes)

  def testSparsifyModel(self):
    # [FUNCTION DOC]
    # Scenario: a constant matrix with a single nonzero value, converted with only the
    # experimental sparsity optimization (given as a set).
    # Expects a truthy model whose metadata lists exactly the block sparsity mode.
    matrix_b_values = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0
    ]
    sess, inputs, outputs = self._getSparsificableModel(matrix_b_values)
    converter = lite.TFLiteConverter.from_session(sess, inputs, outputs)
    converter.optimizations = {lite.Optimize.EXPERIMENTAL_SPARSITY}
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    # Check the conversion metadata.
    metadata = get_conversion_metadata(tflite_model)
    self.assertIsNotNone(metadata)
    self.assertAllEqual([
        metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY,
    ], metadata.options.modelOptimizationModes)

  def testSparsifyQuantizedModel(self):
    # [FUNCTION DOC]
    # Scenario: same nearly-zero matrix, converted with both the default (dynamic range) and the
    # experimental sparsity optimizations.
    # Expects the metadata to list the dynamic range quantization mode then the block sparsity
    # mode.
    matrix_b_values = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0
    ]
    sess, inputs, outputs = self._getSparsificableModel(matrix_b_values)
    converter = lite.TFLiteConverter.from_session(sess, inputs, outputs)
    converter.optimizations = {
        lite.Optimize.DEFAULT, lite.Optimize.EXPERIMENTAL_SPARSITY
    }
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)
    # Check the conversion metadata.
    metadata = get_conversion_metadata(tflite_model)
    self.assertIsNotNone(metadata)
    self.assertAllEqual([
        metadata_fb.ModelOptimizationMode.PTQ_DYNAMIC_RANGE,
        metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY,
    ], metadata.options.modelOptimizationModes)


class GrapplerTest(TestModels, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that the graph optimizer (grappler) constant folding runs during session conversion so
  # ops unsupported by TFLite, such as broadcast, are folded away, and that explicitly chosen
  # input nodes are not folded. Mixes in the parameterized test case but defines no parameters.

  def testConstantFolding(self):
    # [FUNCTION DOC]
    # Side effect: disables eager execution globally for the process.
    # Scenario: a matrix multiplication whose second operand is a constant vector broadcast to
    # three by three.
    # Expects conversion to succeed (broadcast folded) with one float input named Placeholder and
    # one float output named output, both of shape three by three.
    ops.disable_eager_execution()
    # Constant folding handles the tf.broadcast_to operation which was not
    # supported by the TFLite at the time this test was added.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[3, 3], dtype=dtypes.float32)
      y_const = constant_op.constant([1., 2., 3.])
      y_broadcast = array_ops.broadcast_to(y_const, [3, 3])
      out_tensor = math_ops.matmul(in_tensor, y_broadcast, name='output')
      sess = session.Session()

    # Convert model.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([3, 3], input_details[0]['shape'])

    output_details = interpreter.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([3, 3], output_details[0]['shape'])

  def testInputNodeIsNotFolded(self):
    # [FUNCTION DOC]
    # Side effect: disables eager execution globally for the process.
    # Scenario: a multiply of a placeholder by the sum of a constant with itself, where the
    # constant is also listed as a converter input.
    # Expects two inputs named Placeholder and Const, showing the constant was kept as an input
    # instead of being folded away.
    # The leading comment about broadcast is copied from the constant folding test and does not
    # describe this scenario.
    ops.disable_eager_execution()
    # Constant folding handles the tf.broadcast_to operation which was not
    # supported by the TFLite at the time this test was added.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[3], dtype=dtypes.float32)
      y_const = constant_op.constant([1., 2., 3.])
      y_add = y_const + y_const
      out_tensor = in_tensor * y_add
      sess = session.Session()

    # Convert model.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor, y_const],
                                                  [out_tensor])
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual('Placeholder', input_details[0]['name'])
    self.assertEqual('Const', input_details[1]['name'])

  def testGrapplerConstFolding(self):
    # [FUNCTION DOC]
    # Scenario: a traced function adds a zeros tensor of shape two by two by two to a reshaped
    # placeholder, which constant folding turns into a broadcast.
    # Expects conversion to succeed and a single input named Placeholder.
    # Constant folding converts the following add operation to tf.broadcast_to
    # operation which was not supported by the TFLite at the time this test was
    # added.
    @def_function.function
    def plus_placeholder(x, placeholder):
      # [FUNCTION DOC]
      # HELPER: small traced function used as the model body in the test.
      # Inputs: x and placeholder, tensors. Returns their sum.
      return x + placeholder

    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[2, 2], dtype=dtypes.float32)
      out_tensor = plus_placeholder(
          array_ops.zeros([2, 2, 2]),
          array_ops.reshape(in_tensor, shape=[2, 2]))
      sess = session.Session()

    # Convert model.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])
    tflite_model = converter.convert()

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('Placeholder', input_details[0]['name'])


class DefaultConverterAttrsTest(LiteTest):
  # [CLASS DOC]
  # Test case checking the default attribute values of a freshly created session converter.

  def testAttrs(self):
    # [FUNCTION DOC]
    # Scenario: create a converter from a small add model and inspect its attributes without
    # converting.
    # Expects: TFLite output format; float32 inference type; no inference input or output type
    # overrides; empty quantized input stats; no default range stats; reorder across fake quant
    # and change concat input ranges both false; drop control dependency not null; and no
    # graphviz dump directory, video dump off and no conversion summary directory.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(shape=[2, 2], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Convert model.
    converter = lite.TFLiteConverter.from_session(sess, [in_tensor],
                                                  [out_tensor])

    # Assert output format.
    self.assertEqual(converter.output_format, lite_constants.TFLITE)

    # Assert the default inference type is float.
    self.assertEqual(converter.inference_type, dtypes.float32)

    # Assert the default inference type overrides are None.
    self.assertIsNone(converter.inference_input_type)
    self.assertIsNone(converter.inference_output_type)

    # Assert the default quantization options are not set.
    self.assertEqual(converter.quantized_input_stats, {})
    self.assertIsNone(converter.default_ranges_stats)
    self.assertFalse(converter.reorder_across_fake_quant)
    self.assertFalse(converter.change_concat_input_ranges)

    # Assert dropping control dependency is enabled by default.
    self.assertIsNotNone(converter.drop_control_dependency)

    # Assert dumping extra information is disabled by default.
    self.assertIsNone(converter.dump_graphviz_dir)
    self.assertFalse(converter.dump_graphviz_video)
    self.assertIsNone(converter.conversion_summary_dir)


class ControlFlowV1OpsTest(LiteTest):
  # [CLASS DOC]
  # Test case for the error reported when a model contains legacy (version 1) control flow ops.

  def testConverterErrorOnControlFlowV1Ops(self):
    # [FUNCTION DOC]
    # Scenario: convert a text format frozen graph from the test data folder that uses legacy
    # control flow ops, with inputs a, b, c, d and output Merge.
    # Expects a converter error whose message says it failed to functionalize control flow V1 ops
    # and suggests control flow V2 with a documentation link.
    graph_def_file = resource_loader.get_path_to_datafile(
        'testdata/control_flow_v1.pbtxt')
    input_arrays = ['a', 'b', 'c', 'd']
    output_arrays = ['Merge']

    converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                                       input_arrays,
                                                       output_arrays)
    with self.assertRaises(ConverterError) as error:
      converter.convert()
    self.assertIn(
        'Failed to functionalize Control Flow V1 ops. Consider using Control '
        'Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/'
        'tf/compat/v1/enable_control_flow_v2.', str(error.exception))


class QuantizationModeTest(LiteTest, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for the QuantizationMode helper of the converter module, covering
  # warnings about deprecated optimization options.

  @parameterized.named_parameters(
      ('size', lite.Optimize.OPTIMIZE_FOR_SIZE),
      ('latency', lite.Optimize.OPTIMIZE_FOR_LATENCY))
  def testDeprecatedOptionWarning(self, optimization):
    # [FUNCTION DOC]
    # Parameterized test. Input: optimization, the deprecated size or latency optimization option.
    # Side effect: temporarily attaches an in-memory log handler to the root logger and removes
    # it at the end.
    # Scenario: construct a quantization mode with that single optimization, a default target
    # specification and no other arguments.
    # Expects the log to contain the warning to use the default optimization instead.
    """Test if the warning message when using TOCO is logged."""
    log = io.StringIO()
    handler = logging.StreamHandler(log)
    logging.root.addHandler(handler)
    warning_message = 'please use optimizations=[Optimize.DEFAULT] instead.'
    lite.QuantizationMode([optimization], lite.TargetSpec(), None, None)
    self.assertIn(warning_message, log.getvalue())
    logging.root.removeHandler(handler)


if __name__ == '__main__':
  test.main()
