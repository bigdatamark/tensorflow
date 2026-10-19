# [MODULE DOC]
# Tests for the non-portable TFLite metrics: argument validation of the metrics class, counter and
# gauge updates, metrics reported by converters (via mock metrics objects) for successful and failed
# conversions, export timing, and the conversion error gauge including error location info for
# several converter input kinds. Requires the full TensorFlow package and converter.
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
"""TensorFlow Lite Python metrics helper TFLiteMetrics check."""
import gc
import os
import tempfile
import time
from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.compiler.mlir.lite.metrics import converter_error_data_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.lite.python import lite
from tensorflow.lite.python.convert import ConverterError
from tensorflow.lite.python.convert import register_custom_opdefs
from tensorflow.lite.python.metrics import metrics
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.trackable import autotrackable


class MetricsNonportableTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the metrics helper class: constructor validation and that each reporting method
  # updates the matching global monitoring counter or gauge. Tests read the module-level metric
  # objects directly, so expected counts assume a fresh process per test or no earlier increments.

  def test_TFLiteMetrics_creation_no_arg_success(self):
    # [FUNCTION DOC]
    # Creates the metrics object with no arguments; expects no error.
    metrics.TFLiteMetrics()

  def test_TFLiteMetrics_creation_arg_success(self):
    # [FUNCTION DOC]
    # Creates the metrics object with both a hash and a path; expects no error.
    metrics.TFLiteMetrics('hash', '/path/to/model')

  def test_TFLiteMetrics_creation_fails_with_only_hash(self):
    # [FUNCTION DOC]
    # Creating with only a model hash must raise a value error.
    with self.assertRaises(ValueError):
      metrics.TFLiteMetrics(model_hash='hash')

  def test_TFLiteMetrics_creation_fail2_with_only_model_path(self):
    # [FUNCTION DOC]
    # Creating with only a model path must raise a value error.
    with self.assertRaises(ValueError):
      metrics.TFLiteMetrics(model_path='/path/to/model')

  def test_debugger_creation_counter_increase_multiple_same_topic_success(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the bare except converts any failure, including failed assertions, into a
    # generic exception whose message says no exception should be raised, hiding the real cause.
    # Increments the debugger creation counter through two metrics objects (one then deleted and
    # garbage collected) and checks the shared counter reads one, two, then three.
    try:
      stub = metrics.TFLiteMetrics()
      stub.increase_counter_debugger_creation()
      self.assertEqual(metrics._counter_debugger_creation.get_cell().value(), 1)
      stub2 = metrics.TFLiteMetrics()
      stub2.increase_counter_debugger_creation()
      self.assertEqual(metrics._counter_debugger_creation.get_cell().value(), 2)
      del stub
      gc.collect()
      stub2.increase_counter_debugger_creation()
      self.assertEqual(metrics._counter_debugger_creation.get_cell().value(), 3)
    except:
      raise Exception('No exception should be raised.')

  def test_interpreter_creation_counter_increase_success(self):
    # [FUNCTION DOC]
    # Increments the interpreter creation counter once and asserts the python-labeled cell is one.
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_interpreter_creation()
    self.assertEqual(
        metrics._counter_interpreter_creation.get_cell('python').value(), 1)

  def test_converter_attempt_counter_increase_success(self):
    # [FUNCTION DOC]
    # Increments the conversion attempt counter once and asserts it reads one.
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_converter_attempt()
    self.assertEqual(metrics._counter_conversion_attempt.get_cell().value(), 1)

  def test_converter_success_counter_increase_success(self):
    # [FUNCTION DOC]
    # Increments the conversion success counter once and asserts it reads one.
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_converter_success()
    self.assertEqual(metrics._counter_conversion_success.get_cell().value(), 1)

  def test_converter_params_set_success(self):
    # [FUNCTION DOC]
    # Sets one converter parameter and asserts the gauge cell for that name holds the value.
    stub = metrics.TFLiteMetrics()
    stub.set_converter_param('name', 'value')
    self.assertEqual(
        metrics._gauge_conversion_params.get_cell('name').value(), 'value')

  def test_converter_params_multiple_set_success(self):
    # [FUNCTION DOC]
    # Sets the same parameter twice and asserts the gauge holds the second value.
    stub = metrics.TFLiteMetrics()
    stub.set_converter_param('name', 'value')
    stub.set_converter_param('name', 'value1')
    self.assertEqual(
        metrics._gauge_conversion_params.get_cell('name').value(), 'value1')

  def test_converter_params_multiple_label_success(self):
    # [FUNCTION DOC]
    # Sets two differently named parameters and asserts each gauge cell holds its own value.
    stub = metrics.TFLiteMetrics()
    stub.set_converter_param('name1', 'value1')
    stub.set_converter_param('name2', 'value2')
    self.assertEqual(
        metrics._gauge_conversion_params.get_cell('name1').value(), 'value1')
    self.assertEqual(
        metrics._gauge_conversion_params.get_cell('name2').value(), 'value2')

  def test_converter_params_set_latency(self):
    # [FUNCTION DOC]
    # Sets the conversion latency to 34566 and asserts the latency gauge holds that value.
    stub = metrics.TFLiteMetrics()
    stub.set_converter_latency(34566)
    self.assertEqual(metrics._gauge_conversion_latency.get_cell().value(),
                     34566)


class ConverterMetricsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that the converters report the expected metrics calls to a mock metrics object
  # installed on the converter, for success and failure, several converter entry points, and
  # metrics export timing.
  """Testing conversion metrics."""

  def _constructGraphDef(self):
    # [FUNCTION DOC]
    # HELPER: builds a frozen graph definition of a placeholder added to itself (output named
    # add) in a fresh graph and returns it with variables converted to constants.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
      math_ops.add(in_tensor, in_tensor, name='add')
      sess = session.Session()

    return (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, sess.graph_def, ['add']))

  def test_conversion_from_constructor_success(self):
    # [FUNCTION DOC]
    # Converts a frozen graph with a legacy converter whose metrics are replaced by a mock; asserts
    # a model is produced and the mock saw the attempt, success, export and several parameter
    # reports (in any order).
    frozen_graph_def = self._constructGraphDef()

    # Check metrics when conversion successed.
    converter = lite.TFLiteConverter(frozen_graph_def, None, None,
                                     [('in_tensor', [2, 16, 16, 3])], ['add'])
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    converter._tflite_metrics = mock_metrics
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.increase_counter_converter_success(),
        mock.call.export_metrics(),
        mock.call.set_converter_param('input_format', '1'),
        mock.call.set_converter_param('allow_custom_ops', 'False'),
        mock.call.set_converter_param('api_version', '1'),
    ], any_order=True)  # pyformat: disable

  def test_conversion_from_constructor_fail(self):
    # [FUNCTION DOC]
    # Converts with a wrong input tensor name so conversion fails; asserts a converter error is
    # raised, the mock saw the attempt and some parameter reports, and no success was reported.
    frozen_graph_def = self._constructGraphDef()

    # Check metrics when conversion failed.
    converter = lite.TFLiteConverter(frozen_graph_def, None, None,
                                     [('wrong_tensor', [2, 16, 16, 3])],
                                     ['add'])
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    converter._tflite_metrics = mock_metrics
    with self.assertRaises(ConverterError):
      converter.convert()
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.set_converter_param('output_format', '2'),
        mock.call.set_converter_param('select_user_tf_ops', 'None'),
        mock.call.set_converter_param('post_training_quantize', 'False'),
    ], any_order=True)  # pyformat: disable
    mock_metrics.increase_counter_converter_success.assert_not_called()

  def _getIntegerQuantizeModel(self):
    # [FUNCTION DOC]
    # HELPER: builds a trackable object holding a convolution plus relu function, seeds the random
    # generator, and returns a triple of the object, the function's concrete function and a
    # representative dataset generator callable.
    np.random.seed(0)

    root = autotrackable.AutoTrackable()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 5, 5, 3], dtype=tf.float32)])
    def func(inp):
      # [FUNCTION DOC]
      # Test helper: 3x3 convolution with all-ones filter (16 output channels, same padding)
      # followed by relu named output; returns the result.
      conv = tf.nn.conv2d(
          inp, tf.ones([3, 3, 3, 16]), strides=[1, 1, 1, 1], padding='SAME')
      output = tf.nn.relu(conv, name='output')
      return output

    def calibration_gen():
      # [FUNCTION DOC]
      # Test helper generator: yields five samples, each a list with one random float32 array in
      # the range minus one to one of shape one by five by five by three.
      for _ in range(5):
        yield [np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)]

    root.f = func
    to_save = root.f.get_concrete_function()
    return (root, to_save, calibration_gen)

  def test_conversion_from_frozen_graph_v2(self):
    # [FUNCTION DOC]
    # Converts the concrete function with the V2 converter using default optimizations and a
    # representative dataset, with mock metrics; asserts a model is produced and the mock saw
    # attempt, success and quantization related parameter reports (in any order).
    model, func, calibration_gen = self._getIntegerQuantizeModel()

    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func],
                                                                         model)
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    quantized_converter._tflite_metrics = mock_metrics
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.increase_counter_converter_success(),
        mock.call.set_converter_param(
            'optimization_post_training_integer_quantize', 'True'),
        mock.call.set_converter_param('inference_type', 'tf.int8'),
        mock.call.set_converter_param('select_user_tf_ops', 'None'),
        mock.call.set_converter_param('activations_type', 'tf.int8'),
    ], any_order=True)  # pyformat: disable

  def test_conversion_from_keras_v2(self):
    # [FUNCTION DOC]
    # Trains a one-unit dense Keras model briefly, converts it with the V2 converter using mock
    # metrics, and asserts the mock saw attempt, success, export and parameter reports.
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    converter._tflite_metrics = mock_metrics
    converter.convert()
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.increase_counter_converter_success(),
        mock.call.export_metrics(),
        mock.call.set_converter_param('inference_type', 'tf.float32'),
        mock.call.set_converter_param('target_ops', 'TFLITE_BUILTINS'),
        mock.call.set_converter_param('optimization_default', 'False'),
    ], any_order=True)  # pyformat: disable

  def _createV1SavedModel(self, shape):
    # [FUNCTION DOC]
    # HELPER: writes a simple saved model to a temporary directory and returns its path. The model
    # has two float placeholders (given shape) and a variable computing first plus second times the
    # variable. Created in a fresh graph and session; reuses the same directory name each call.
    """Create a simple SavedModel."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        in_tensor_1 = tf.compat.v1.placeholder(
            shape=shape, dtype=tf.float32, name='inputB')
        in_tensor_2 = tf.compat.v1.placeholder(
            shape=shape, dtype=tf.float32, name='inputA')
        variable_node = tf.Variable(1.0, name='variable_node')
        out_tensor = in_tensor_1 + in_tensor_2 * variable_node
        inputs = {'x': in_tensor_1, 'y': in_tensor_2}
        outputs = {'z': out_tensor}
        sess.run(tf.compat.v1.variables_initializer([variable_node]))
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return saved_model_dir

  def test_conversion_from_saved_model(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: replaces the global process-time function with a mock and never restores it,
    # leaking into later tests.
    # Converts a saved model with a mock metrics object and a mocked clock; asserts the mock saw
    # attempt, success, a latency of 2000 and export (in any order).
    saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])
    converter = lite.TFLiteSavedModelConverter(saved_model_dir, set(['serve']),
                                               ['serving_default'])
    converter.experimental_new_converter = True
    mock_metrics = mock.create_autospec(
        metrics.TFLiteConverterMetrics, instance=True)
    converter._tflite_metrics = mock_metrics
    time.process_time = mock.Mock(side_effect=np.arange(1, 1000, 2).tolist())
    converter.convert()
    mock_metrics.assert_has_calls([
        mock.call.increase_counter_converter_attempt(),
        mock.call.increase_counter_converter_success(),
        mock.call.set_converter_latency(2000),
        mock.call.export_metrics(),
    ], any_order=True)  # pyformat: disable

  def disable_converter_counter_metrics(self, tflite_metrics):
    # [FUNCTION DOC]
    # HELPER: replaces the attempt and success counter methods of the given metrics object with
    # do-nothing functions so tests do not touch the global counters.

    def empty_func():
      # [FUNCTION DOC]
      # HELPER: do-nothing replacement for counter methods.
      pass

    tflite_metrics.increase_counter_converter_attempt = empty_func
    tflite_metrics.increase_counter_converter_success = empty_func

  def test_export_at_conversion_done(self):
    # [FUNCTION DOC]
    # Replaces the converter's exporter with a mock, converts a saved model, and asserts metrics
    # were exported exactly once after conversion and that destroying the metrics object does not
    # export again.
    saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    tflite_metrics = converter._tflite_metrics
    mock_exporter = mock.MagicMock()
    tflite_metrics._metrics_exporter = mock_exporter
    self.disable_converter_counter_metrics(tflite_metrics)
    mock_exporter.ExportMetrics.assert_not_called()
    converter.convert()
    mock_exporter.ExportMetrics.assert_called_once()
    tflite_metrics.__del__()
    mock_exporter.ExportMetrics.assert_called_once()

  def test_export_at_exit(self):
    # [FUNCTION DOC]
    # Replaces the exporter with a mock without converting; asserts nothing is exported until the
    # metrics object destructor is invoked, then exactly one export happens.
    saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    tflite_metrics = converter._tflite_metrics
    mock_exporter = mock.MagicMock()
    tflite_metrics._metrics_exporter = mock_exporter
    self.disable_converter_counter_metrics(tflite_metrics)
    mock_exporter.ExportMetrics.assert_not_called()
    tflite_metrics.__del__()
    mock_exporter.ExportMetrics.assert_called_once()


def mock_ngrams(data, width, axis=-1, string_separator=' ', name=None):
  # [FUNCTION DOC]
  # HELPER: stand-in for a text n-grams operation, deliberately built without the width attribute
  # in its implements annotation so that conversion fails. Parameters: data (string tensor or
  # ragged tensor), width (integer window size), axis (default last), string_separator (default
  # one space), name (optional scope name). Builds a function annotated as implementing the
  # n-grams composite, which slices the data into shifted windows, stacks them and joins strings
  # with the separator, then calls it and returns the result.
  """This mock Ngrams lack the width attr, causing conversion to fail."""

  experimental_implements = [
      'name: "tftext:Ngrams"',
      'attr { key: "axis" value { i: %d } }' % axis,
      'attr { key: "reduction_type" value { s: "STRING_JOIN" } }',
      'attr { key: "string_separator" value { s: "%s" } }' % string_separator,
  ]
  experimental_implements = ' '.join(experimental_implements)

  @tf.function(experimental_implements=experimental_implements)
  def func(data):
    # [FUNCTION DOC]
    # HELPER: the annotated body of the mock n-grams: converts the input to a tensor, builds the
    # shifted slices along the axis, stacks them and joins along the axis with the separator;
    # returns the joined strings.
    with ops.name_scope(name, 'NGrams', [data, width]):
      data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name='data')
      slices = []
      for start in range(width):
        stop = None if start - width + 1 == 0 else start - width + 1
        if axis >= 0:
          idx = [slice(None)] * axis + [slice(start, stop)]
        else:
          idx = [Ellipsis, slice(start, stop)] + [slice(None)] * (-axis - 1)
        slices.append(data[idx])

      # Stack the slices.
      stack_axis = axis + 1 if axis >= 0 else axis
      windowed_data = array_ops_stack.stack(slices, stack_axis)

      return string_ops.reduce_join(
          windowed_data, axis=axis, separator=string_separator)

  return func(data)


class ConverterErrorMetricTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):
  # [CLASS DOC]
  # Tests that failed conversions populate the conversion error gauge and that the converter
  # error carries the expected location type and sources. Fixture: setUp swaps the attempt,
  # success and parameter metrics with mocks (keeping the real errors gauge) and tearDown
  # restores them.
  """Testing conversion error metric."""

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp, then saves the module-level attempt counter, success counter and
    # parameter gauge and replaces them with auto-specified mocks so other tests are unaffected.
    super(ConverterErrorMetricTest, self).setUp()

    # Mock metrics instance except errors so other test cases are not affected.
    mock_attempt = mock.create_autospec(monitoring.Counter, instance=True)
    self._counter_conversion_attempt = metrics._counter_conversion_attempt
    metrics._counter_conversion_attempt = mock_attempt

    mock_success = mock.create_autospec(monitoring.Counter, instance=True)
    self._counter_conversion_success = metrics._counter_conversion_success
    metrics._counter_conversion_success = mock_success

    mock_params = mock.create_autospec(monitoring.StringGauge, instance=True)
    self._gauge_conversion_params = metrics._gauge_conversion_params
    metrics._gauge_conversion_params = mock_params

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent tearDown, then restores the metric objects saved by setUp.
    super(ConverterErrorMetricTest, self).tearDown()
    # # Restore metrics instances.
    metrics._counter_conversion_attempt = self._counter_conversion_attempt
    metrics._counter_conversion_success = self._counter_conversion_success
    metrics._gauge_conversion_params = self._gauge_conversion_params

  def convert_and_check_location_info(self,
                                      converter,
                                      expected_type,
                                      expected_sources=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if conversion unexpectedly succeeds the model is asserted to be null, which
    # would fail confusingly; and when expected_sources is a plain string (as one caller passes) it
    # is iterated per character.
    # HELPER: runs the converter and checks the error it raises. Parameters: converter (object to
    # convert), expected_type (expected location type of the single reported error),
    # expected_sources (optional collection of strings that must all appear in the location text).
    # Asserts exactly one error with the expected location type.
    # The custom attribute of ConverterError can't be accessed with
    # assertRaises so use try-catch block instead.
    try:
      tflite_model = converter.convert()
      self.assertIsNone(tflite_model)
    except ConverterError as converter_error:
      # pylint: disable=g-assert-in-except
      self.assertLen(converter_error.errors, 1)
      location = converter_error.errors[0].location
      self.assertEqual(location.type, expected_type)

      if expected_sources:
        debug_string = str(location)
        for source in expected_sources:
          self.assertIn(source, debug_string)
      # pylint: enable=g-assert-in-except

  def test_failure_at_PrepareCompositeFunctionsPass(self):
    # [FUNCTION DOC]
    # Skipped when TFRT is enabled. Registers a fake tokenizer op definition, builds a Keras model
    # with the width-less mock n-grams layer, converts it allowing custom ops, and expects a
    # converter error with unknown location type. Asserts the error gauge cell for the composite
    # functions pass holds the message that the width attribute is not set or not an integer.
    if context.is_tfrt_enabled():
      self.skipTest('This test crashed with TFRT.')

    class NgramsLayer(tf.keras.layers.Layer):
      # [CLASS DOC]
      # Test helper Keras layer wrapping the mock n-grams function.

      def call(self, input_tensor, **kwargs):
        # [FUNCTION DOC]
        # Applies the mock n-grams with window width two on the last axis and a space separator;
        # extra keyword arguments are ignored. Returns the result.
        return mock_ngrams(input_tensor, width=2, axis=-1, string_separator=' ')

    # Registers a fake WhitespaceTokenizeWithOffsets so the TFText fusing logic
    # is enable in MLIR side.
    custom_opdefs_str = (
        'name: \'WhitespaceTokenizeWithOffsets\' input_arg: {name: \'Input1\' '
        'type: DT_FLOAT} input_arg: {name: \'Input2\' type: DT_FLOAT} '
        'output_arg: {name: \'Output\' type: DT_FLOAT}')
    register_custom_opdefs([custom_opdefs_str])

    model = tf.keras.models.Sequential([NgramsLayer()])
    model.predict(tf.constant(['test']))
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.allow_custom_ops = True
    self.convert_and_check_location_info(
        converter, converter_error_data_pb2.ConverterErrorData.UNKNOWNLOC)
    exported_error = metrics._gauge_conversion_errors.get_cell(
        'CONVERT_TF_TO_TFLITE_MODEL',
        'PrepareCompositeFunctionsPass',
        'tf.Const',
        'UNKNOWN',
    ).value()
    self.assertEqual(exported_error,
                     "\'width\' attribute is not set or not an integer")

  def test_need_flex_ops(self):
    # [FUNCTION DOC]
    # Builds a graph whose add op is renamed to an unregistered custom op, saves it as a saved
    # model, converts it, and expects a converter error with a name location pointing at add.
    # Asserts the error gauge holds a message saying the custom add op is neither a custom nor a
    # flex op, with the needs-custom-ops error code.

    def create_graph_with_custom_add(opname='CustomAdd'):
      # [FUNCTION DOC]
      # HELPER: builds a graph definition of a placeholder added to itself, renames the add node
      # to opname (default CustomAdd) removing its type attribute, registers a matching op
      # definition, and returns the graph definition with input and output mappings.
      custom_opdefs_str = (
          'name: \'' + opname +
          '\' input_arg: {name: \'Input1\' type: DT_FLOAT} '
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

    new_graph, inputs, outputs = create_graph_with_custom_add()

    # Import to load the custom opdef.
    saved_model_dir = os.path.join(self.get_temp_dir(), 'model')
    with ops.Graph().as_default():
      with session.Session() as sess:
        import_graph_def(new_graph, name='')
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    self.convert_and_check_location_info(
        converter,
        converter_error_data_pb2.ConverterErrorData.NAMELOC,
        expected_sources='add')
    exported_error = metrics._gauge_conversion_errors.get_cell(
        'CONVERT_TF_TO_TFLITE_MODEL', 'CONVERT_SAVED_MODEL', 'tf.CustomAdd',
        'ERROR_NEEDS_CUSTOM_OPS').value()
    self.assertIn(
        "'tf.CustomAdd' op is neither a custom op nor a flex op\n",
        exported_error,
    )
    self.assertIn('Error code: ERROR_NEEDS_CUSTOM_OPS', exported_error)

  def test_unsupported_control_flow_v1(self):
    # [FUNCTION DOC]
    # Converts a saved model from test data that uses legacy control flow; expects a converter
    # error with unknown location and asserts the error gauge cell for the unsupported control
    # flow error code holds the merge-node message.
    filename = resource_loader.get_path_to_datafile(
        '../testdata/control_flow_v1_saved_model')
    converter = lite.TFLiteConverterV2.from_saved_model(filename)
    self.convert_and_check_location_info(
        converter, converter_error_data_pb2.ConverterErrorData.UNKNOWNLOC)
    exported_error = metrics._gauge_conversion_errors.get_cell(
        'CONVERT_TF_TO_TFLITE_MODEL', 'CONVERT_SAVED_MODEL', '',
        'ERROR_UNSUPPORTED_CONTROL_FLOW_V1').value()
    self.assertEqual(
        exported_error,
        'Merge only has 4 inputs, while only merge nodes with two inputs '
        'supported.\n\tFailed to functionalize Control Flow V1 ops. Consider '
        'using Control Flow V2 ops instead. See https://www.tensorflow.org/'
        'api_docs/python/tf/compat/v1/enable_control_flow_v2.')

  def test_location_from_concrete_functions(self):
    # [FUNCTION DOC]
    # Converts a function adding two complex tensors with mismatched shapes from concrete
    # functions; expects a converter error with call-site location type and this test file named
    # in the location.

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 2, 3, 3], dtype=tf.complex64),
        tf.TensorSpec(shape=[None, None, 1, 3, 3], dtype=tf.complex64),
    ])
    def model(a, b):
      # [FUNCTION DOC]
      # Test helper: adds the two inputs with the op named add.
      return tf.add(a, b, name='add')

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [model.get_concrete_function()], model)
    self.convert_and_check_location_info(
        converter,
        converter_error_data_pb2.ConverterErrorData.CALLSITELOC,
        expected_sources=[
            'tensorflow/lite/python/metrics/metrics_nonportable_test.py',
        ])

  def test_location_from_saved_model(self):
    # [FUNCTION DOC]
    # Saves a module adding two complex tensors (with debug info) into a temporary directory,
    # converts the saved model, and expects a converter error with call-site location type naming
    # this test file.

    with tempfile.TemporaryDirectory() as tmp_dir:

      class Adder(tf.Module):
        # [CLASS DOC]
        # Test helper module with one serving function adding two complex tensors.

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, None, 2, 3, 3], dtype=tf.complex64),
            tf.TensorSpec(shape=[None, None, 1, 3, 3], dtype=tf.complex64),
        ])
        def serving_default(self, a, b):
          # [FUNCTION DOC]
          # Test helper: adds the two inputs with the op named add.
          return tf.add(a, b, name='add')

      tf.saved_model.save(
          Adder(),
          tmp_dir,
          options=tf.saved_model.SaveOptions(save_debug_info=True))

      converter = lite.TFLiteConverterV2.from_saved_model(tmp_dir)
      self.convert_and_check_location_info(
          converter,
          converter_error_data_pb2.ConverterErrorData.CALLSITELOC,
          expected_sources=[
              'tensorflow/lite/python/metrics/metrics_nonportable_test.py',
          ])

  @parameterized.named_parameters(
      ('_WithoutLoweringToSavedModel', False, None),
      ('_WithLoweringToSavedModel', True,
       'tensorflow/lite/python/metrics/metrics_nonportable_test.py'))
  def test_location_from_keras_model(self, lower_to_saved_model,
                                     expected_source):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the two Keras inputs are declared with the same shape although the scenario
    # elsewhere relies on mismatched shapes; the failure is presumably from the complex type.
    # Parameterized by a flag for lowering to a saved model and the expected source file text (or
    # nothing). Builds a Keras model adding two complex inputs, converts it with the flag applied,
    # and expects a converter error with call-site location type; when an expected source is given
    # it must appear in the location text.
    input_tensor1 = tf.keras.layers.Input(
        shape=[None, None, 2, 3, 3], dtype=tf.complex64)
    input_tensor2 = tf.keras.layers.Input(
        shape=[None, None, 2, 3, 3], dtype=tf.complex64)
    output = tf.keras.layers.Add()([input_tensor1, input_tensor2])
    model = tf.keras.Model(
        inputs=[input_tensor1, input_tensor2], outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.experimental_lower_to_saved_model = lower_to_saved_model
    # The location does not contain callsite to the current file.
    self.convert_and_check_location_info(
        converter,
        converter_error_data_pb2.ConverterErrorData.CALLSITELOC,
        expected_sources=[expected_source] if expected_source else None)


if __name__ == '__main__':
  test.main()
