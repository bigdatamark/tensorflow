# [MODULE DOC]
# Unit tests for the version 2 TensorFlow Lite converter (the converter class built from concrete
# functions, saved models, Keras models and Jax functions). Each test builds a small model, converts
# it, then loads the result in the TFLite interpreter and checks values, tensor dtypes, quantization
# parameters, signatures and conversion metadata.
# Test classes: concrete-function conversion, saved-model conversion, Keras conversion, Jax conversion,
# control flow and RNN models, strided slice, graph optimizer behavior, unknown shapes, resource and
# variant types, custom op registration, intermediate tensor preservation, dataset ops, sparsity,
# buffer offset mode and boundary (infinity) values.
# Import-time side effects: when the process supports it, the dynamic library load flags are changed so
# shared-object symbols are globally visible (needed so the interpreter extension can see the test
# registerer extension); Jax is imported if available, otherwise a module-level flag disables the Jax
# tests. A module-level alias names the preset quantization method enum of the stable-HLO options.
# All test classes derive from the shared model test base in the sibling test utility module. The
# module runs the TensorFlow test main when executed directly.
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
"""Tests for lite.py functionality related to TensorFlow 2.0."""

import ctypes
import functools
import itertools
import os
import sys

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# Force loaded shared object symbols to be globally visible. This is needed so
# that the interpreter_wrapper, in one .so file, can see the test_registerer,
# in a different .so file. Note that this may already be set by default.
# pylint: disable=g-import-not-at-top
if hasattr(sys, 'setdlopenflags') and hasattr(sys, 'getdlopenflags'):
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.compiler.mlir.quantization.stablehlo import quantization_options_pb2 as quant_opts_pb2
from tensorflow.lite.python import conversion_metadata_schema_py_generated as metadata_fb
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter
from tensorflow.lite.python import lite
from tensorflow.lite.python import lite_v2_test_util
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import test_util as tflite_test_util
from tensorflow.lite.python import util
from tensorflow.lite.python.testdata import _pywrap_test_registerer as test_registerer
from tensorflow.lite.python.testdata import double_op
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import map_ops
from tensorflow.python.ops import rnn
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import saved_model
from tensorflow.python.trackable import autotrackable

# Type alias for preset quantization method protobuf enums.
_PresetQuantizationMethod = quant_opts_pb2.PresetQuantizationMethod.PresetMethod

# Only run jax related tests when we can import jax.
DISABLE_JAX_TEST = False
try:
  import jax
  from jax import numpy as jnp
except ImportError:
  DISABLE_JAX_TEST = True
# pylint: enable=g-import-not-at-top


class FromConcreteFunctionTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case covering the version 2 TFLite converter when it is built from concrete functions.
  # Inherits the shared model-building and evaluation helpers from the common test base class
  # (simple variable model, multi-function model, ceil model, TFLite evaluation helpers).
  # Scenarios: invalid construction, float and scalar and string inputs, multiple functions,
  # signature definitions, post-training dynamic-range / integer / int16 quantization, quantization-aware
  # training models, flex ops, control flow, per-channel options, new quantizer debug modes, op versions,
  # and conversion metadata. No state is kept on the instance.

  @test_util.run_v2_only
  def testTypeInvalid(self):
    # [FUNCTION DOC]
    # Scenario: building the converter from a plain (not yet traced) function instead of a concrete
    # function. Setup: simple variable model; passes the raw function plus the model as owner.
    # Expected: a value error whose message tells the user to call the function-tracing method first.
    root = self._getSimpleVariableModel()
    with self.assertRaises(ValueError) as error:
      _ = lite.TFLiteConverterV2.from_concrete_functions([root.f], root)
    self.assertIn('call get_concrete_function', str(error.exception))

  @test_util.run_v2_only
  def testFloat(self):
    # [FUNCTION DOC]
    # Scenario: basic float conversion of a simple variable model through the concrete-function path.
    # Traces the function with a one-element float input, converts, runs the TFLite model and asserts the
    # output equals the original TensorFlow result.
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    tflite_model = converter.convert()

    # Check output value from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @parameterized.named_parameters(
      ('_INT8InputOutput', dtypes.int8),
      ('_UINT8InputOutput', dtypes.uint8),
      ('_INT16InputOutput', dtypes.int16),
  )
  @test_util.run_v2_only
  def testInvalidFloat(self, inference_input_output_type):
    # [FUNCTION DOC]
    # Parameterized over int8, uint8 and int16 as the requested inference input/output type.
    # Converts a float model (no quantization optimizations) while asking for an integer inference type.
    # Expected: a value error saying inference input and output types must be float32.
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    with self.assertRaises(ValueError) as error:
      converter.inference_input_type = inference_input_output_type
      converter.inference_output_type = inference_input_output_type
      converter.convert()
    self.assertEqual(
        'The inference_input_type and inference_output_type '
        'must be tf.float32.',
        str(error.exception),
    )

  @test_util.run_v2_only
  def testScalarInput(self):
    # [FUNCTION DOC]
    # Same as the float test but the traced input is a scalar (rank 0). Asserts the converted model gives
    # the same value as TensorFlow.
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testStringInput(self):
    # [FUNCTION DOC]
    # Scenario: a module whose call method returns its string-tensor argument unchanged. Converts it from
    # a concrete function, then calls the signature runner with a non-numpy (TensorFlow) string tensor.
    # Expected: a value error saying the passed value is not a numpy array.
    class Model(tf.Module):
      # [CLASS DOC]
      # HELPER: tiny module used only by the string-input test.
      # Module with one traced call method that returns its input unchanged; used to produce a string-typed
      # concrete function.

      @tf.function
      def __call__(self, x):
        # [FUNCTION DOC]
        # HELPER: identity function body for the string-input test model.
        # Traced function that returns its input unchanged.
        return x

    root = Model()
    concrete_func = root.__call__.get_concrete_function(
        tf.constant([str(x) for x in range(11)])
    )
    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    tflite_model = converter.convert()
    input_data = tf.constant(
        [str(x) for x in range(11)], shape=(11,), dtype=tf.dtypes.string
    )
    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    my_signature = interp.get_signature_runner()

    with self.assertRaises(ValueError) as error:
      _ = my_signature(x=input_data)
    self.assertIn(
        'Passed in value type is not a numpy array, got type ',
        str(error.exception),
    )

  @test_util.run_v2_only
  def testModelWithoutInputs(self):
    # [FUNCTION DOC]
    # Scenario: a function with an empty input signature that returns a random uniform tensor.
    # Builds the owner object and concrete function with a nested helper, converts, and only asserts that a
    # non-null model is produced.
    def _get_random_number_gen():
      # [FUNCTION DOC]
      # HELPER: builds the zero-input model for the model-without-inputs test.
      # Takes no arguments. Creates an auto-tracked owner object with a traced no-input function returning a
      # random float tensor of one element; returns a pair of the owner and its concrete function.
      root = autotrackable.AutoTrackable()

      @tf.function(input_signature=[])
      def func():
        # [FUNCTION DOC]
        # HELPER: the zero-input traced function (returns one random uniform float) of the nested model builder.
        # Takes no arguments, returns a one-element float tensor.
        return tf.random.uniform(shape=[1], dtype=tf.float32)

      root.f = func
      to_save = root.f.get_concrete_function()
      return (root, to_save)

    # Model with no input
    root, concrete_func = _get_random_number_gen()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

  @test_util.run_v2_only
  def testMultiFunctionModel(self):
    # [FUNCTION DOC]
    # Scenario: converting only one function (add) out of a model that has several functions.
    # Asserts conversion succeeds and the evaluated result matches TensorFlow for the add function.
    """Convert a single model in a multi-functional model."""
    root = self._getMultiFunctionModel()
    input_data = tf.constant(1.0, shape=[1])
    concrete_func = root.add.get_concrete_function(input_data)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.add(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testConvertMultipleFunctions(self):
    # [FUNCTION DOC]
    # Scenario: converting two functions (add and sub) at once from a multi-function model.
    # Asserts: two signatures named add and sub with inputs x and outputs output_0; signature runner results
    # (add gives 3, sub gives -2 for input 1.0); input/output names, float32 dtype, shape [1] and zero
    # quantization; and the conversion metadata (API version 2, model type concrete functions, no
    # optimization modes).
    """Convert multiple functions in a multi-functional model."""
    root = self._getMultiFunctionModel()
    input_data = tf.constant(1.0, shape=[1])
    add_func = root.add.get_concrete_function(input_data)
    sub_func = root.sub.get_concrete_function(input_data)

    # Try converting multiple functions.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [add_func, sub_func], root
    )
    tflite_model = converter.convert()

    # Check signatures are valid from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()

    # Verify the SignatureDef structure returned is as expected.
    self.assertLen(signature_defs, 2)
    self.assertEqual(list(signature_defs.keys()), ['add', 'sub'])
    self.assertLen(signature_defs.values(), 2)
    self.assertEqual(list(signature_defs['add'].keys()), ['inputs', 'outputs'])
    self.assertCountEqual(signature_defs['add']['inputs'], ['x'])
    self.assertEqual(list(signature_defs['add']['outputs']), ['output_0'])
    self.assertEqual(list(signature_defs['sub'].keys()), ['inputs', 'outputs'])
    self.assertCountEqual(signature_defs['sub']['inputs'], ['x'])
    self.assertEqual(list(signature_defs['sub']['outputs']), ['output_0'])

    # Verify the Signature runner executions.
    add_signature_runner = interp.get_signature_runner('add')
    add_output = add_signature_runner(x=input_data)
    self.assertEqual(add_output['output_0'], 3)
    input_details = add_signature_runner.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('add_x:0', input_details['x']['name'])
    self.assertEqual(np.float32, input_details['x']['dtype'])
    self.assertTrue(([1] == input_details['x']['shape']).all())
    self.assertEqual((0.0, 0), input_details['x']['quantization'])

    sub_signature_runner = interp.get_signature_runner('sub')
    sub_output = sub_signature_runner(x=input_data)
    self.assertEqual(sub_output['output_0'], -2)
    output_details = sub_signature_runner.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(
        'StatefulPartitionedCall_1:0', output_details['output_0']['name']
    )
    self.assertEqual(np.float32, output_details['output_0']['dtype'])
    self.assertTrue(([1] == output_details['output_0']['shape']).all())
    self.assertEqual((0.0, 0), output_details['output_0']['quantization'])

    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(metadata.environment.apiVersion, 2)
    self.assertEqual(
        metadata.environment.modelType,
        metadata_fb.ModelType.TF_CONCRETE_FUNCTIONS,
    )
    self.assertAllEqual([], metadata.options.modelOptimizationModes)

  def _getIntegerQuantizeModel(self, num_filters=16):
    # [FUNCTION DOC]
    # HELPER: builds a small convolution model plus calibration data for quantization tests.
    # Parameter num_filters (integer, default 16) is the number of convolution filters. Seeds numpy with 0,
    # creates an auto-tracked owner with a traced function (input 1x5x5x3 float; same-padded 3x3 convolution
    # with ones weights, then relu named output). Returns a triple: owner, concrete function and a calibration
    # generator function.
    np.random.seed(0)

    root = autotrackable.AutoTrackable()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 5, 5, 3], dtype=tf.float32)]
    )
    def func(inp):
      # [FUNCTION DOC]
      # HELPER: the traced convolution + relu function of the quantization test model builder.
      # Takes a float input tensor of shape 1x5x5x3 and returns the relu of a convolution with all-ones
      # filters.
      conv = tf.nn.conv2d(
          inp,
          tf.ones([3, 3, 3, num_filters]),
          strides=[1, 1, 1, 1],
          padding='SAME',
      )
      output = tf.nn.relu(conv, name='output')
      return output

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: representative-dataset generator for the quantization test model.
      # Yields five single-input lists, each a random float32 array of shape 1x5x5x3 in [-1, 1].
      for _ in range(5):
        yield [np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)]

    root.f = func
    to_save = root.f.get_concrete_function()
    return (root, to_save, calibration_gen)

  @parameterized.named_parameters(
      ('EnableMlirQuantizer', True),  # enable mlir quantizer
      ('DisableMlirQuantizer', False),
  )  # disable mlir quantizer
  def testPostTrainingCalibrateAndQuantize(self, mlir_quantizer):
    # [FUNCTION DOC]
    # Parameterized: new MLIR quantizer enabled or disabled (argument mlir_quantizer).
    # Converts the convolution model once as float and once with default optimizations and a representative
    # dataset. Asserts conversion metadata (TensorFlow version, API version 2, concrete-function model type,
    # no custom or select-TF ops, optimization mode full-integer PTQ), that default input/output stay
    # float32, and that the quantized model is smaller than the float model.
    root, func, calibration_gen = self._getIntegerQuantizeModel()

    # Convert float model.
    float_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    # Convert quantized model.
    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_converter.experimental_new_quantizer = mlir_quantizer
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(quantized_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(
        metadata.environment.tensorflowVersion.decode('utf-8'),
        versions.__version__,
    )
    self.assertEqual(metadata.environment.apiVersion, 2)
    self.assertEqual(
        metadata.environment.modelType,
        metadata_fb.ModelType.TF_CONCRETE_FUNCTIONS,
    )
    self.assertEqual(metadata.options.allowCustomOps, False)
    self.assertEqual(metadata.options.enableSelectTfOps, False)
    self.assertEqual(metadata.options.forceSelectTfOps, False)
    self.assertAllEqual(
        [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER],
        metadata.options.modelOptimizationModes,
    )

    # The default input and output types should be float.
    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.float32, input_details[0]['dtype'])
    output_details = interp.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  @parameterized.named_parameters(
      ('_INT8InputOutput', dtypes.int8),
      ('_UINT8InputOutput', dtypes.uint8),
      ('_INT16InputOutput', dtypes.int16),
  )
  @test_util.run_v2_only
  def testInvalidPostTrainingDynamicRangeQuantization(
      self, inference_input_output_type
  ):
    # [FUNCTION DOC]
    # Parameterized over int8, uint8, int16 inference types.
    # Dynamic-range quantization (default optimizations, no representative data) with a non-float inference
    # type. Expected: a value error saying inference input/output types must be float32.
    root, func, _ = self._getIntegerQuantizeModel()

    # Convert float model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Convert quantized model.
    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    with self.assertRaises(ValueError) as error:
      quantized_converter.inference_input_type = inference_input_output_type
      quantized_converter.inference_output_type = inference_input_output_type
      quantized_converter.convert()
    self.assertEqual(
        'The inference_input_type and inference_output_type '
        'must be tf.float32.',
        str(error.exception),
    )

  def _createV2QATSavedModelWithFloatOpsAtEnd(self):
    # [FUNCTION DOC]
    # HELPER: creates a saved Keras quantization-aware-training model with float ops at the end.
    # Takes no arguments. Builds a Keras model: fake-quant layer, 3x3 convolution, fake-quant layer, then a
    # sigmoid dense layer with no fake-quant after it. Saves it under a directory in the test temp dir and
    # returns the directory path. Side effect: writes files.
    """Create a simple QAT SavedModel that includes float ops at the end."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'qat_float_ops_at_end')
    input_tensor = tf.keras.layers.Input((32, 32, 128))

    class _FakeQuantArgsLayer(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: Keras layer wrapper required so the fake-quant op can be used inside Keras 3 models.
      # Applies fake quantization with fixed range -3 to 3 to its input.
      """A fake quantization layer with fake_quant_with_min_max_args.

      Keras 3 requires wrapping the tf function inside Keras layer.
      """

      def call(self, x):
        # [FUNCTION DOC]
        # HELPER: layer forward pass.
        # Takes input tensor x and returns it fake-quantized with min -3 and max 3.
        return tf.quantization.fake_quant_with_min_max_args(x, -3.0, 3.0)

    x = _FakeQuantArgsLayer()(input_tensor)
    x = tf.keras.layers.Conv2D(1, (3, 3), bias_initializer='ones')(x)
    x = _FakeQuantArgsLayer()(x)
    # Exclude the quantization of the following Dense layer by not putting
    # fake quant layer after the dense layer.
    output_tensor = tf.keras.layers.Dense(
        1, activation='sigmoid', bias_initializer='ones'
    )(x)
    model = tf.keras.Model(input_tensor, output_tensor)
    model.save(saved_model_dir)
    return saved_model_dir

  def testQuantizationRemovesQDQsForFloatIOInQAT(self):
    # [FUNCTION DOC]
    # Scenario: quantization-aware-trained saved model whose final dense layer is not fake-quantized.
    # Converts with default optimizations and loads with default delegates disabled.
    # Asserts the last operator in the model is the logistic op (not a dequantize op).
    saved_model_dir = self._createV2QATSavedModelWithFloatOpsAtEnd()
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    # Because assertions on the model later, we opt out applying default TFLite
    # delegates (i.e. the XNNPACK delegate).
    interp = interpreter.Interpreter(
        model_content=quantized_model,
        experimental_op_resolver_type=interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    # The model should have LOGISTIC op, instead of DEQUANTIZE op.
    op_details = interp._get_ops_details()
    self.assertEqual(op_details[len(op_details) - 1]['op_name'], 'LOGISTIC')

  @parameterized.named_parameters(
      ('EnableMlirQuantizer', True),  # enable mlir quantizer
      ('DisableMlirQuantizer', False),
  )  # disable mlir quantizer
  def testQuantizationRemovesQDQsForFloatIO(self, mlir_quantizer):
    # [FUNCTION DOC]
    # Parameterized: MLIR quantizer enabled or disabled.
    # Converts the ceil model with default optimizations and representative data. With default delegates
    # disabled, asserts the model has exactly one operator and it is the ceil op (quantize/dequantize pairs
    # removed for float input/output). Note: a code comment mentions a sqrt op but the check is for ceil.
    func, calibration_gen = self._getCeilModel()
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func.get_concrete_function()]
    )
    converter.representative_dataset = calibration_gen
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.experimental_new_quantizer = mlir_quantizer
    quantized_model = converter.convert()

    # Because assertions on the model later, we opt out applying default TFLite
    # delegates (i.e. the XNNPACK delegate).
    interp = interpreter.Interpreter(
        model_content=quantized_model,
        experimental_op_resolver_type=interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    # The model should have only one sqrt op.
    op_details = interp._get_ops_details()
    self.assertLen(op_details, 1)
    self.assertEqual(op_details[0]['op_name'], 'CEIL')

  @parameterized.named_parameters(
      ('_Default', False, False, dtypes.float32),
      ('_INT8InputOutput', False, False, dtypes.int8),
      ('_UINT8InputOutput', False, False, dtypes.uint8),
      ('_INT16Quantize', False, True, dtypes.float32),
      ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16),
      ('_IntOnly', True, False, dtypes.float32),
      ('_IntOnly_INT8InputOutput', True, False, dtypes.int8),
      ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8),
      ('_IntOnly_INT16Quantize', True, True, dtypes.float32),
      ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16),
  )
  def testIntegerQuantization(
      self, is_int_only, is_int16_quantize, inference_input_output_type
  ):
    # [FUNCTION DOC]
    # Parameterized over: integer-only mode, int16-activation mode and inference input/output dtype.
    # Converts the convolution model as float, then quantized with a representative dataset and the
    # supported-ops set chosen from the flags (int8 builtins, int16 activations with int8 weights, or both
    # with regular builtins). Sets the requested inference types, then asserts: metadata optimization mode
    # (full-integer PTQ, or int16 PTQ when int16 requested), interpreter input/output dtypes equal the
    # requested type, and quantized model smaller than float.
    root, func, calibration_gen = self._getIntegerQuantizeModel()

    # Convert float model.
    converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Convert quantized model.
    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    if is_int_only:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
      else:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
    else:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.TFLITE_BUILTINS,
        ]
    quantized_converter.inference_input_type = inference_input_output_type
    quantized_converter.inference_output_type = inference_input_output_type
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(quantized_tflite_model)
    self.assertIsNotNone(metadata)
    expected_opt_options = [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER]
    if is_int16_quantize:
      expected_opt_options = [metadata_fb.ModelOptimizationMode.PTQ_INT16]
    self.assertAllEqual(
        expected_opt_options, metadata.options.modelOptimizationModes
    )

    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(
        inference_input_output_type.as_numpy_dtype, input_details[0]['dtype']
    )
    output_details = interp.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(
        inference_input_output_type.as_numpy_dtype, output_details[0]['dtype']
    )

    # Ensure that the quantized tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(tflite_model))

  @parameterized.named_parameters(('_INT16Quantize_INT8InputOutput', True))
  def testInvalidIntegerQuantization(self, is_int16_quantize):
    # [FUNCTION DOC]
    # Parameterized with int16 quantization on. Requests int8 inference input/output with the int16 activation
    # op set. Expected: value error saying types must be one of float32 or int16.
    root, func, calibration_gen = self._getIntegerQuantizeModel()

    # Convert quantized model.
    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    if is_int16_quantize:
      quantized_converter.target_spec.supported_ops = [
          lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
          lite.OpsSet.TFLITE_BUILTINS,
      ]
    with self.assertRaises(ValueError) as error:
      quantized_converter.inference_input_type = dtypes.int8
      quantized_converter.inference_output_type = dtypes.int8
      quantized_converter.convert()
    self.assertEqual(
        'The inference_input_type and inference_output_type '
        "must be in ['tf.float32', 'tf.int16'].",
        str(error.exception),
    )

  def testCalibrateAndQuantizeBuiltinInt16(self):
    # [FUNCTION DOC]
    # Converts the convolution model with the int16-activation/int8-weight op set and calibration data.
    # Asserts float32 default input/output, tensor dtypes (weights int8, bias int64, activations int16) and
    # that the quantized model is smaller than the float one.
    root, func, calibration_gen = self._getIntegerQuantizeModel()

    # Convert float model.
    float_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    converter.representative_dataset = calibration_gen
    quantized_tflite_model = converter.convert()

    self.assertIsNotNone(quantized_tflite_model)

    # The default input and output types should be float.
    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.float32, input_details[0]['dtype'])
    output_details = interp.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(np.float32, output_details[0]['dtype'])

    # The weights tensor should be quantized to 8 bits,
    # the bias tensor should be 32 bits to utilize optimized kernels,
    # and the activations should be 16 bits.
    tensor_details = interp.get_tensor_details()
    self.assertEqual(np.int8, tensor_details[2]['dtype'])
    self.assertEqual(np.int64, tensor_details[1]['dtype'])
    self.assertEqual(np.int16, tensor_details[0]['dtype'])
    self.assertEqual(np.int16, tensor_details[3]['dtype'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  @test_util.run_v2_only
  def testSignatureDefs(self):
    # [FUNCTION DOC]
    # Builds the converter directly (constructor, not factory) from the add function and the owner as
    # trackable object. Asserts a single default serving signature with input x and one output starting
    # with the prefix output, and that evaluating via the signature gives the TensorFlow value.
    """Test converting SignatureDef is correct and uses SignatureDef API."""
    root = self._getMultiFunctionModel()
    input_data = tf.constant(1.0, shape=[1])
    add_func = root.add.get_concrete_function(input_data)

    converter = lite.TFLiteConverterV2([add_func], trackable_obj=root)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = add_func(input_data)
    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()
    results = self._evaluateTFLiteModelUsingSignatureDef(
        tflite_model, 'serving_default', {'x': input_data}
    )
    self.assertLen(list(results.keys()), 1)
    self.assertStartsWith(list(results.keys())[0], 'output')
    self.assertAllClose(
        expected_value.numpy(),
        results[signature_defs['serving_default']['outputs'][0]],
    )

    # Verify the SignatureDef structure returned is as expected.
    self.assertLen(signature_defs, 1)
    self.assertEqual(list(signature_defs.keys()), ['serving_default'])
    self.assertLen(signature_defs.values(), 1)
    self.assertEqual(
        list(signature_defs['serving_default'].keys()), ['inputs', 'outputs']
    )
    self.assertCountEqual(signature_defs['serving_default']['inputs'], ['x'])
    self.assertLen(list(signature_defs['serving_default']['outputs']), 1)
    self.assertStartsWith(
        list(signature_defs['serving_default']['outputs'])[0], 'output'
    )

  @test_util.run_v2_only
  def testNoSignatureDefsWhenTrackingObjIsNone(self):
    # [FUNCTION DOC]
    # Converts a concrete function with the trackable object set to nothing. Asserts the converted model has
    # no signature definitions.
    """Test converting SignatureDef is correct and uses SignatureDef API."""
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], None
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()
    # Verify that there is no SignatureDef structure found.
    self.assertEmpty(signature_defs)

  @test_util.run_v2_only
  def testNoSignatureDefsWhenInvalidTrackingObjIsGiven(self):
    # [FUNCTION DOC]
    # Converts a concrete function with an unrelated empty tracked object. Asserts the converted model has
    # no signature definitions.
    """Test converting SignatureDef is correct and uses SignatureDef API."""
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], trackable_obj=autotrackable.AutoTrackable()
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()
    # Verify that there is no SignatureDef structure found.
    self.assertEmpty(signature_defs)

  @test_util.run_v2_only
  def testTrackbleObject(self):
    # [FUNCTION DOC]
    # Converts the add function with the owner passed by keyword as the trackable object, and checks the
    # result against TensorFlow. (Name has a typo: Trackble.)
    """Test converting with trackable objects."""
    root = self._getMultiFunctionModel()
    input_data = tf.constant(1.0, shape=[1])
    add_func = root.add.get_concrete_function(input_data)

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [add_func], trackable_obj=root
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = add_func(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  def _getTrainingTimeQuantizedModel(self):
    # [FUNCTION DOC]
    # HELPER: builds a Keras model with fake-quant ops for quantization-aware-training tests.
    # Defines a custom layer inside, then returns a sequential model containing one such layer (3 units,
    # input size 2).
    class QLinear(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: custom Keras dense-like layer with fake quantization of inputs, weights and outputs.
      # Attributes: units, weight matrix, and non-trainable min and max variables (-6 and 6).

      def __init__(self, units=3, **kwargs):
        # [FUNCTION DOC]
        # Constructor. Parameters: units (integer, default 3) number of outputs; extra keyword arguments are
        # passed to the base layer. Stores units.
        super().__init__(**kwargs)
        self.units = units

      def build(self, input_shape):
        # [FUNCTION DOC]
        # Creates the weights: a trainable matrix of shape (last input dim, units) with random normal init and
        # two non-trainable scalars min (-6) and max (6). Parameter input_shape is the input shape.
        self.w = self.add_weight(
            'weight',
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
        )
        self.min_var = self.add_weight(
            'min',
            initializer=tf.keras.initializers.Constant(-6.0),
            trainable=False,
        )
        self.max_var = self.add_weight(
            'max',
            initializer=tf.keras.initializers.Constant(6.0),
            trainable=False,
        )

      def call(self, inputs):
        # [FUNCTION DOC]
        # Forward pass: fake-quantizes the input and the weights using the min/max variables, multiplies them
        # and fake-quantizes the product. Returns the result tensor.
        x = tf.quantization.fake_quant_with_min_max_vars(
            inputs, self.min_var, self.max_var
        )

        w_fq = tf.quantization.fake_quant_with_min_max_vars(
            self.w, self.min_var, self.max_var
        )
        x = tf.matmul(x, w_fq)

        x = tf.quantization.fake_quant_with_min_max_vars(
            x, self.min_var, self.max_var
        )

        return x

    return tf.keras.Sequential(QLinear(3, input_shape=(2,)))

  @parameterized.named_parameters(
      ('_DefaultFLOAT32InputOutput', dtypes.float32),
      ('_INT8InputOutput', dtypes.int8),
      ('_UINT8InputOutput', dtypes.uint8),
  )
  @test_util.run_v2_only
  def testTrainingTimeQuantization(self, inference_input_output_type):
    # [FUNCTION DOC]
    # Parameterized over float32, int8, uint8 inference types.
    # Converts the quantization-aware-training Keras model as float and with default optimizations plus the
    # inference types. Asserts metadata mode is quantization-aware training, interpreter I/O dtypes equal the
    # requested type, and the quantized model is smaller than the float model.
    model = self._getTrainingTimeQuantizedModel()

    float_converter = lite.TFLiteConverterV2.from_keras_model(model)
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)

    quantized_converter = lite.TFLiteConverterV2.from_keras_model(model)
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.inference_input_type = inference_input_output_type
    quantized_converter.inference_output_type = inference_input_output_type
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(quantized_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertAllEqual(
        [metadata_fb.ModelOptimizationMode.QUANTIZATION_AWARE_TRAINING],
        metadata.options.modelOptimizationModes,
    )

    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(
        inference_input_output_type.as_numpy_dtype, input_details[0]['dtype']
    )
    output_details = interp.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(
        inference_input_output_type.as_numpy_dtype, output_details[0]['dtype']
    )

    # Ensure that the quantized tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  @test_util.run_v2_only
  def testNewQuantizer(self):
    # [FUNCTION DOC]
    # Converts the convolution model with int8 builtins once with the old quantizer and once with the new
    # MLIR quantizer; for five random inputs asserts the two models agree within tolerance 0.1.
    """Test the model quantized by the new converter."""
    root, func, calibration_gen = self._getIntegerQuantizeModel()

    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    quantized_converter.representative_dataset = calibration_gen

    # default quantizer
    quantized_converter.experimental_new_quantizer = False
    old_tflite = quantized_converter.convert()

    # new quantizer
    quantized_converter.experimental_new_quantizer = True
    new_tflite = quantized_converter.convert()

    for _ in range(5):
      input_data = tf.constant(
          np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)
      )
      old_value = self._evaluateTFLiteModel(old_tflite, [input_data])
      new_value = self._evaluateTFLiteModel(new_tflite, [input_data])
      self.assertAllClose(old_value, new_value, atol=1e-01)

  @test_util.run_v2_only
  def testGatherNDQI8(self):
    # [FUNCTION DOC]
    # Scenario: a Keras model doing gather_nd between two fake-quant ops, converted with the private QDQ
    # conversion mode set to static. Runs with int8 input data and asserts the gathered values
    # [1, 4] (within tolerance). Defines a nested model class.
    """Test gather_nd with quantized i8 parameters."""

    class GatherNDQI8QDQ(tf.keras.Model):
      # [CLASS DOC]
      # HELPER: Keras model used only to build the quantize-dequantize gather_nd graph.
      # Has one traced method with a 2x2 float input.

      @tf.function(
          input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)]
      )
      def func(self, input_tensor):
        # [FUNCTION DOC]
        # HELPER: fake-quantizes the input, gathers the diagonal elements and fake-quantizes the result.
        # Parameter input_tensor is a 2x2 float tensor; returns a 2-element tensor.
        x = tf.quantization.fake_quant_with_min_max_args(
            input_tensor, -3.0, 3.0
        )
        x = tf.gather_nd(x, [[0, 0], [1, 1]])
        return tf.quantization.fake_quant_with_min_max_args(x, -3.0, 3.0)

    # Build a QDQ model so that tfl.gather_nd will be converted to a QI8 version
    # with the `_experimental_qdq_conversion_mode`` flag
    root = GatherNDQI8QDQ()
    concrete_func = root.func.get_concrete_function()
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    converter._experimental_qdq_conversion_mode = 'STATIC'
    tflite_model = converter.convert()

    np_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_tensor = tf.constant(np_data, dtype=tf.int8)
    expected_value = [1, 4]
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_tensor])
    self.assertAllClose(expected_value, actual_value[0], atol=1e-05)

  @test_util.run_v2_only
  def testEmbeddings(self):
    # [FUNCTION DOC]
    # Scenario: model that gathers rows from a 2000x300 shared weight by int32 indices (20 of them).
    # Converts and asserts output matches TensorFlow within 1e-5. Note: the random indices are random floats in
    # [0, 1) cast to integer, so they are all zero.
    """Test model with embeddings."""
    input_data = tf.constant(
        np.array(np.random.random_sample((20)), dtype=np.int32)
    )

    class EmbeddingModel(tf.keras.Model):
      # [CLASS DOC]
      # HELPER: Keras model holding a shared embedding weight; used only by the embedding test.
      # Attribute shared_weights is a 2000x300 float weight with random normal init.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor with no arguments; creates the shared embedding weight variable.
        super().__init__()
        self.shared_weights = self.add_weight(
            'weights',
            shape=(2000, 300),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                mean=0.0, stddev=300 ** (-0.5)
            ),
        )

      @tf.function(input_signature=[tf.TensorSpec(shape=(20), dtype=tf.int32)])
      def func(self, x):
        # [FUNCTION DOC]
        # HELPER: traced method taking 20 int32 indices; returns the gathered embedding rows.
        return tf.gather(self.shared_weights, x)

    # Building the model.
    root = EmbeddingModel()
    concrete_func = root.func.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.func(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertAllClose(expected_value.numpy(), actual_value[0], atol=1e-05)

  @test_util.run_v2_only
  def testGraphDebugInfo(self):
    # [FUNCTION DOC]
    # Converts a concrete function that uses a variable multiplication and asserts the converter recorded
    # valid debug information (checked by a shared helper from the base class).
    """Test a concrete function has debug info captured."""
    root = autotrackable.AutoTrackable()
    root.v1 = tf.Variable(3.0)
    root.f = tf.function(lambda x: root.v1 * x)
    input_data = tf.constant(1.0, shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    converter.convert()
    self._assertValidDebugInfo(converter._debug_info)

  def _getIntegerQuantizationModelWithFlexOp(self):
    # [FUNCTION DOC]
    # HELPER: model with ops needing the flex (select TF) delegate, plus calibration data.
    # Seeds numpy. Traced function on a 3x3x3x3x3 float input: tanh, 3D convolution with ones, erf, tanh.
    # Returns owner, concrete function and calibration generator.
    np.random.seed(0)

    root = autotrackable.AutoTrackable()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[3, 3, 3, 3, 3], dtype=tf.float32)]
    )
    def func(inp):
      # [FUNCTION DOC]
      # HELPER: the traced tanh/conv3d/erf/tanh function of the flex-op test model.
      tanh = tf.math.tanh(inp)
      # Flex delegate will merge the consecutive conv3d and erf ops into one
      # Delegate node.
      conv3d = tf.nn.conv3d(
          tanh,
          tf.ones([3, 3, 3, 3, 3]),
          strides=[1, 1, 1, 1, 1],
          padding='SAME',
      )
      erf = tf.math.erf(conv3d)
      output = tf.math.tanh(erf)
      return output

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields five single-input lists of random float32 arrays of shape 3x3x3x3x3 in [-1, 1].
      for _ in range(5):
        yield [
            np.random.uniform(-1, 1, size=(3, 3, 3, 3, 3)).astype(np.float32)
        ]

    root.f = func
    return (root, root.f.get_concrete_function(), calibration_gen)

  @parameterized.named_parameters(
      ('_Default', False, False, dtypes.float32),
      ('_INT8InputOutput', False, False, dtypes.int8),
      ('_UINT8InputOutput', False, False, dtypes.uint8),
      ('_INT16Quantize', False, True, dtypes.float32),
      ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16),
      ('_IntOnly', True, False, dtypes.float32),
      ('_IntOnly_INT8InputOutput', True, False, dtypes.int8),
      ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8),
      ('_IntOnly_INT16Quantize', True, True, dtypes.float32),
      ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16),
  )
  @test_util.run_v2_only
  def testIntegerQuantizationWithFlexOp(
      self, is_int_only, is_int16_quantize, inference_input_output_type
  ):
    # [FUNCTION DOC]
    # Parameterized over integer-only, int16 mode and inference dtype.
    # Quantizes the flex-op model with supported ops chosen per flags and always including select TF ops.
    # Asserts metadata marks select TF ops as enabled, the expected optimization mode (full-integer or int16
    # PTQ) and that interpreter input/output dtypes equal the requested type.
    root, func, calibration_gen = self._getIntegerQuantizationModelWithFlexOp()

    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    if is_int_only:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.SELECT_TF_OPS,
        ]
      else:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.TFLITE_BUILTINS_INT8,
            lite.OpsSet.SELECT_TF_OPS,
        ]
    else:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.TFLITE_BUILTINS,
            lite.OpsSet.SELECT_TF_OPS,
        ]
      else:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.TFLITE_BUILTINS,
            lite.OpsSet.SELECT_TF_OPS,
        ]

    quantized_converter.inference_input_type = inference_input_output_type
    quantized_converter.inference_output_type = inference_input_output_type
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(quantized_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(metadata.options.enableSelectTfOps, True)
    expected_opt_options = [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER]
    if is_int16_quantize:
      expected_opt_options = [metadata_fb.ModelOptimizationMode.PTQ_INT16]
    self.assertAllEqual(
        expected_opt_options, metadata.options.modelOptimizationModes
    )

    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(
        inference_input_output_type.as_numpy_dtype, input_details[0]['dtype']
    )
    output_details = interp.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(
        inference_input_output_type.as_numpy_dtype, output_details[0]['dtype']
    )

  def _getIntegerQuantizationModelWithUnsupportedOps(self):
    # [FUNCTION DOC]
    # HELPER: model with ceil ops (which have no int8/int16 kernel) mixed with tanh, plus calibration data.
    # Seeds numpy. Traced function with two 3-element float inputs; returns a pair (ceil of add, tanh).
    # Returns owner, concrete function and calibration generator.
    np.random.seed(0)

    root = autotrackable.AutoTrackable()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[3], dtype=tf.float32),
            tf.TensorSpec(shape=[3], dtype=tf.float32),
        ]
    )
    def func(a, b):
      # [FUNCTION DOC]
      # HELPER: computes ceil(a) + tanh(b), then ceil again; returns that and the tanh output.
      # ceil kernel does not support int8 nor int16 types neither.
      left = tf.math.ceil(a)
      right = tf.nn.tanh(b)
      add = tf.math.add(left, right)
      # ceil kernel does not support int8 nor int16 types neither.
      output = tf.math.ceil(add)
      return (output, right)

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields five pairs of random float32 arrays of length 3 in [-1, 1].
      for _ in range(5):
        yield [
            np.random.uniform(-1, 1, size=(3)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3)).astype(np.float32),
        ]

    root.f = func
    return (root, root.f.get_concrete_function(), calibration_gen)

  @parameterized.named_parameters(
      ('_INT8InputOutput', False, False, dtypes.int8),
      ('_UINT8InputOutput', False, False, dtypes.uint8),
      ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16),
      ('_IntOnly_INT8InputOutput', True, False, dtypes.int8),
      ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8),
      ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16),
      ('_IntOnly_INT8InputOutputMlirQuant', True, False, dtypes.int8, True),
      ('_IntOnly_UINT8InputOutputMlirQuant', True, False, dtypes.uint8, True),
  )
  @test_util.run_v2_only
  def testIntegerQuantizationWithUnsupportedOps(
      self,
      is_int_only,
      is_int16_quantize,
      inference_input_output_type,
      enable_mlir_quantizer=False,
  ):
    # [FUNCTION DOC]
    # Parameterized over integer-only, int16, inference dtype and an optional flag (default off) that enables
    # the MLIR quantizer. Quantizes the model with ceil ops and checks dtypes: first input and output use the
    # requested type; the second input and output use the requested type only with the MLIR quantizer,
    # otherwise they fall back to float32 because ceil cannot be quantized.
    root, func, calib_gen = (
        self._getIntegerQuantizationModelWithUnsupportedOps()
    )

    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calib_gen
    if is_int_only:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.TFLITE_BUILTINS,
        ]
      else:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.TFLITE_BUILTINS_INT8,
            lite.OpsSet.TFLITE_BUILTINS,
        ]
    else:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.TFLITE_BUILTINS,
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
        expected_dtype if enable_mlir_quantizer else dtypes.float32
    )

    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual(input_details[0]['dtype'], expected_dtype)
    self.assertEqual(input_details[1]['dtype'], expected_ceil_dtype)
    output_details = interp.get_output_details()
    self.assertLen(output_details, 2)
    self.assertEqual(output_details[0]['dtype'], expected_dtype)
    self.assertEqual(output_details[1]['dtype'], expected_ceil_dtype)

  def _getIntegerQuantizationModelWithControlFlow(self):
    # [FUNCTION DOC]
    # HELPER: model with a conditional, plus calibration data covering both branches.
    # Returns the traced function, its concrete function, and a calibration generator.
    def true_fn(x):
      # [FUNCTION DOC]
      # HELPER: identity branch function used by the conditional; returns its input x.
      return x

    def false_fn(x):
      # [FUNCTION DOC]
      # HELPER: identity branch function used by the conditional; returns its input x.
      return x

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, 2], dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        ]
    )
    def model(x, b):
      # [FUNCTION DOC]
      # HELPER: traced function with a 1x2 float input and a boolean scalar; doubles x, applies a conditional
      # with identity branches and doubles again.
      x = x + x
      x = tf.cond(b, true_fn=lambda: true_fn(x), false_fn=lambda: false_fn(x))
      return x + x

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields five samples with the condition true and five with it false, each with a random 1x2
      # float32 array.
      for _ in range(5):
        yield [
            np.random.uniform(
                -1,
                1,
                size=(
                    1,
                    2,
                ),
            ).astype(np.float32),
            tf.constant(True),
        ]
      for _ in range(5):
        yield [
            np.random.uniform(
                -1,
                1,
                size=(
                    1,
                    2,
                ),
            ).astype(np.float32),
            tf.constant(False),
        ]

    return (model, model.get_concrete_function(), calibration_gen)

  @parameterized.named_parameters(
      ('_INT8InputOutput', False, False, dtypes.int8),
      ('_UINT8InputOutput', False, False, dtypes.uint8),
      ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16),
      ('_IntOnly_INT8InputOutput', True, False, dtypes.int8),
      ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8),
      ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16),
  )
  @test_util.run_v2_only
  def testIntegerQuantizationWithControlFlow(
      self,
      is_int_only,
      is_int16_quantize,
      inference_input_output_type,
      enable_mlir_quantizer=False,
  ):
    # [FUNCTION DOC]
    # Parameterized over integer-only, int16 mode, inference dtype (extra optional MLIR flag defaults off and is
    # never supplied by the parameter sets). Quantizes the conditional model; asserts two inputs (first of the
    # requested dtype, second boolean) and one output of the requested dtype.
    root, func, calib_gen = self._getIntegerQuantizationModelWithControlFlow()

    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calib_gen
    if is_int_only:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.TFLITE_BUILTINS,
        ]
      else:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.TFLITE_BUILTINS_INT8,
            lite.OpsSet.TFLITE_BUILTINS,
        ]
    else:
      if is_int16_quantize:
        quantized_converter.target_spec.supported_ops = [
            lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            lite.OpsSet.TFLITE_BUILTINS,
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

    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    self.assertLen(input_details, 2)
    self.assertEqual(input_details[0]['dtype'], expected_dtype)
    self.assertEqual(input_details[1]['dtype'], dtypes.bool)
    output_details = interp.get_output_details()
    self.assertLen(output_details, 1)
    self.assertEqual(output_details[0]['dtype'], expected_dtype)

  @parameterized.named_parameters(
      ('_BlocklistedNoneWithLowering', None, None, True),
      ('_BlocklistedNoneWithoutLowering', None, None, False),
      ('_BlocklistedOpsWithLowering', {'CONV_2D'}, None, True),
      ('_BlocklistedOpsWithoutLowering', {'CONV_2D'}, None, False),
      ('_BlocklistedNodesWithLowering', None, {'PartitionedCall:0'}, True),
      ('_BlocklistedNodesWithoutLowering', None, {'Identity'}, False),
  )
  @test_util.run_v2_only
  def testNewQuantizerBlocklistingArgs(
      self, denylisted_ops, denylisted_nodes, lower_to_saved_model
  ):
    # [FUNCTION DOC]
    # Parameterized over denylisted op names, denylisted node names and whether the graph is lowered to a saved
    # model. Runs a calibrate-only conversion, then the MLIR quantize step with the denylists. Asserts that
    # when any denylist is given no tensor is quantized, otherwise exactly four tensors carry quantization
    # scales (quantize, filter, bias, dequantize).
    """Test the model quantized by the new converter and denylisted options."""
    root, func, calibration_gen = self._getIntegerQuantizeModel()
    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    quantized_converter.representative_dataset = calibration_gen
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.experimental_new_quantizer = True
    quantized_converter._experimental_calibrate_only = True
    quantized_converter.experimental_lower_to_saved_model = lower_to_saved_model
    calibrated = quantized_converter.convert()
    quantized_tflite_model = convert.mlir_quantize(
        calibrated,
        denylisted_ops=denylisted_ops,
        denylisted_nodes=denylisted_nodes,
    )
    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    details = interp.get_tensor_details()
    num_quantized_tensors = sum([
        1
        for detail in details
        if len(detail['quantization_parameters']['scales'])
    ])
    if denylisted_nodes or denylisted_ops:
      self.assertEqual(num_quantized_tensors, 0)
      return
    self.assertEqual(num_quantized_tensors, 4)  # quant, filter, bias, dequant

  @parameterized.named_parameters(
      ('_SingleLayer', False),
      ('_WholeModel', True),
  )
  @test_util.run_v2_only
  def testNewQuantizerNumericVerificationDebugMode(self, whole_model_verify):
    # [FUNCTION DOC]
    # Parameterized: single-layer or whole-model verification.
    # Creates a production quantized model and a debug model with numeric-verify ops. Asserts the flatbuffers
    # differ, both have exactly one quantize op, the debug model has exactly one numeric-verify tensor whose
    # name encodes the original tensor name and id, and the counts match.
    """Test the model quantized by the new converter with numeric verify ops."""
    root, func, calibration_gen = self._getIntegerQuantizeModel()

    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    quantized_converter.representative_dataset = calibration_gen

    # Create a TFLite model with new quantizer.
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.experimental_new_quantizer = True
    production_tflite = quantized_converter.convert()
    # Create a TFLite model with new quantizer and numeric verify ops.
    quantized_converter._experimental_calibrate_only = True
    calibrated = quantized_converter.convert()
    debug_mode_tflite = convert.mlir_quantize(
        calibrated,
        enable_numeric_verify=True,
        enable_whole_model_verify=whole_model_verify,
    )

    # Check if adding debug mode should output a different flatbuffer.
    self.assertNotEqual(production_tflite, debug_mode_tflite)

    # Check if newly added ops are numeric verify ops.
    input_data = tf.constant(
        np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)
    )

    def examine_tflite_model(tflite_content, input_data):
      # [FUNCTION DOC]
      # HELPER: runs a TFLite model once on given input and collects all tensors.
      # Parameters: tflite_content (model bytes) and input_data (tensor). Returns a pair: a mapping from
      # tensor name to its value after invoke, and the list of tensor details.
      interp = interpreter.Interpreter(
          model_content=tflite_content,
          experimental_op_resolver_type=interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
      )
      interp.allocate_tensors()
      input_details = interp.get_input_details()
      interp.set_tensor(input_details[0]['index'], input_data.numpy())
      interp.invoke()
      tensor_details = interp.get_tensor_details()
      return {
          details['name']: interp.get_tensor(details['index'])
          for details in interp.get_tensor_details()
      }, tensor_details

    tflite_result, _ = examine_tflite_model(production_tflite, input_data)
    debug_mode_tflite_result, debug_tensor_details = examine_tflite_model(
        debug_mode_tflite, input_data
    )

    # MLIR-based quantizer should output flatbuffer model with `tfl.quantize`.
    num_production_quantize_ops = len([
        None
        for output_tensor_name in tflite_result
        if 'tfl.quantize' in output_tensor_name
    ])
    self.assertEqual(num_production_quantize_ops, 1)
    # MLIR-based quantizer should output flatbuffer model with `tfl.quantize`.
    num_debug_quantize_ops = len([
        None
        for output_tensor_name in debug_mode_tflite_result
        if 'tfl.quantize' in output_tensor_name
    ])
    # Two numbers should be equal.
    self.assertEqual(num_production_quantize_ops, num_debug_quantize_ops)
    # DebugMode TFLite flatbuffer should have NumericVerifyOps more than zero.
    # The name has the prefix "NumericVerify/{name}:{id}
    # where {name} is the tensor name of the original quantized op's activation,
    # and {id} is its tensor id.
    num_debug_ops = 0
    for output_tensor_name in debug_mode_tflite_result:
      if 'NumericVerify' in output_tensor_name:
        pos_end_prefix = len('NumericVerify/')
        pos_colon = output_tensor_name.rfind(':')
        self.assertEqual('NumericVerify/', output_tensor_name[:pos_end_prefix])
        tensor_id = int(output_tensor_name[pos_colon + 1 :])
        original_tensor_name = output_tensor_name[pos_end_prefix:pos_colon]
        self.assertEqual(
            original_tensor_name, debug_tensor_details[tensor_id]['name']
        )
        num_debug_ops += 1
    self.assertEqual(num_debug_ops, 1)
    # The number of debug ops should be equal to that of quantized ops.
    self.assertEqual(num_debug_ops, num_debug_quantize_ops)

  @parameterized.named_parameters(
      ('_PerChannelQuant', False, False),
      ('_PerChannelMlirQuant', False, True),
      ('_PerTensorQuant', True, False),
      ('_PerTensorMlirQuant', True, True),
      ('_PerChannelDynamicRange', False, False),
      ('_PerTensorDynamicRange', True, False),
  )
  @test_util.run_v2_only
  def testDisablePerChannelQuantization(
      self,
      disable_per_channel=False,
      enable_mlir_quantizer=False,
  ):
    # [FUNCTION DOC]
    # Parameterized (disable per-channel flag, MLIR quantizer flag, and for dynamic-range cases no calibration
    # data - note both dynamic-range cases pass false for the MLIR flag). Converts the 38-filter convolution
    # model with the private disable-per-channel option. Finds the convolution tensor and asserts the number of
    # scales (and zero points, if more than one) is one when disabled, else 38.
    k_conv_name = 'Conv2D'
    # Dynamic range quant requires total num elements of filters > 1024.
    k_num_filters = 38
    root, func, calib_gen = self._getIntegerQuantizeModel(k_num_filters)
    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calib_gen
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS
    ]
    quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
    if disable_per_channel:
      quantized_converter._experimental_disable_per_channel = (
          disable_per_channel
      )
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    detail = next((
        d
        for d in interp.get_tensor_details()
        if d['name'].startswith(k_conv_name)
    ))
    quant_params = detail['quantization_parameters']
    expected_num_params = 1 if disable_per_channel else k_num_filters
    self.assertLen(quant_params['scales'], expected_num_params)
    if len(quant_params['zero_points']) != 1:
      self.assertLen(quant_params['zero_points'], expected_num_params)

  def _getIntegerQuantizeDenseModel(self, num_filters=32):
    # [FUNCTION DOC]
    # HELPER: builds a matmul+relu dense model plus calibration data.
    # Parameter num_filters (default 32) is the output size. Input 1x16 float; multiplies by an all-ones 16 x
    # num_filters matrix and applies relu. Returns owner, concrete function and calibration generator.
    np.random.seed(0)

    root = autotrackable.AutoTrackable()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 16], dtype=tf.float32)]
    )
    def func(inp):
      # [FUNCTION DOC]
      # HELPER: dense (matmul with ones) followed by relu for a 1x16 float input.
      dense = tf.matmul(a=inp, b=tf.ones([16, num_filters]))
      output = tf.nn.relu(dense, name='output')
      return output

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields five single-input lists of random 1x16 float32 arrays in [-1, 1].
      for _ in range(5):
        yield [np.random.uniform(-1, 1, size=(1, 16)).astype(np.float32)]

    root.f = func
    to_save = root.f.get_concrete_function()
    return (root, to_save, calibration_gen)

  @parameterized.named_parameters(
      ('_PerChannelQuant', False, False),
      ('_PerChannelMlirQuant', False, True),
      ('_PerTensorQuant', True, False),
      ('_PerTensorMlirQuant', True, True),
      ('_PerChannelDynamicRange', False, True, True),
      ('_PerTensorDynamicRange', True, True, True),
  )
  @test_util.run_v2_only
  def testDisablePerChannelQuantizationForDenseLayers(
      self,
      disable_per_channel_for_dense=False,
      enable_mlir_quantizer=False,
      representative_dataset=False,
  ):
    # [FUNCTION DOC]
    # Parameterized (disable per-channel for dense, MLIR quantizer, use representative data).
    # Converts the 64-unit dense model with the private option for disabling per-channel quantization of dense
    # layers; with default delegates off finds the matmul tensor and asserts one scale if disabled else 64.
    k_dense_name = 'MatMul'
    # Dynamic range quant requires total num elements of filters > 1024.
    k_num_filters = 64
    root, func, calib_gen = self._getIntegerQuantizeDenseModel(k_num_filters)
    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    if representative_dataset:
      quantized_converter.representative_dataset = calib_gen
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS
    ]
    quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
    if disable_per_channel_for_dense:
      quantized_converter._experimental_disable_per_channel_quantization_for_dense_layers = (
          disable_per_channel_for_dense
      )
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # Do not apply delegates as XNNPack converts per tensor to per channel.
    interp = interpreter.Interpreter(
        model_content=quantized_tflite_model,
        experimental_op_resolver_type=interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    detail = next((
        d
        for d in interp.get_tensor_details()
        if d['name'].startswith(k_dense_name)
    ))
    quant_params = detail['quantization_parameters']
    expected_num_params = 1 if disable_per_channel_for_dense else k_num_filters
    self.assertLen(quant_params['scales'], expected_num_params)
    if len(quant_params['zero_points']) != 1:
      self.assertLen(quant_params['zero_points'], expected_num_params)

  @parameterized.named_parameters(
      ('MlirQuantize', True), ('TocoQuantize', False)
  )
  @test_util.run_v2_only
  def testQuantizeBiasOverflow(self, enable_mlir_quantizer):
    # [FUNCTION DOC]
    # Parameterized: MLIR or old quantizer. Builds a matmul model with tiny weights and large biases,
    # quantizes with a one-sample calibration set, runs it on a tiny input and asserts the output equals the
    # bias, i.e. the quantizer adjusted scales to avoid bias overflow.
    """Tests if the quantizer handles bias overflow by adjusting scales."""
    input_data = np.array([[-1e-3, 1e-3]], dtype=np.float32)

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields one mapping from name x to the tiny 1x2 input array.
      yield {'x': input_data}

    root = self._getMatMulModelWithSmallWeights()
    input_data = tf.constant([-1e-3, 1e-3], shape=(1, 2))
    concrete_func = root.matmul.get_concrete_function(input_data)
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_gen
    converter.experimental_new_quantizer = enable_mlir_quantizer
    quantized_model = converter.convert()

    interp = interpreter.Interpreter(model_content=quantized_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    interp.set_tensor(input_details[0]['index'], input_data)
    interp.invoke()
    output_details = interp.get_output_details()
    output = interp.get_tensor(output_details[0]['index'])
    # the inputs and weights are far smaller than the biases, so the final
    # result should be equal to the biases.
    self.assertAllClose(root.bias, output.flatten())

  @test_util.run_v2_only
  def testOpVersion(self):
    # [FUNCTION DOC]
    # Scenario: two bilinear resize ops, one with half-pixel centers (needs version 3). Converts, parses the
    # flatbuffer and asserts the resize-bilinear operator code has version 3.
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[5, 5], dtype=tf.float32)]
    )
    def custom_resize(image):
      # [FUNCTION DOC]
      # HELPER: traced function on a 5x5 float input that adds batch/channel dimensions and sums two bilinear
      # resizes to 2x2 (one with half-pixel centers, one without).
      # Add "batch" and "channels" dimensions
      image = image[tf.newaxis, ..., tf.newaxis]
      # ResizeBilinear version 3.
      resize1 = tf.compat.v1.image.resize_bilinear(
          image, [2, 2], half_pixel_centers=True
      )
      # ResizeBilinear version 1.
      resize2 = tf.compat.v1.image.resize_bilinear(image, [2, 2])
      return resize1 + resize2

    concrete_func = custom_resize.get_concrete_function()
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], custom_resize
    )
    tflite_model = converter.convert()
    model_object = schema_fb.Model.GetRootAsModel(tflite_model, 0)
    model = schema_fb.ModelT.InitFromObj(model_object)

    for operator in model.operatorCodes:
      if operator.builtinCode == schema_fb.BuiltinOperator.RESIZE_BILINEAR:
        # half_pixel_centers is supported by ResizeBilinear version 3.
        self.assertEqual(operator.version, 3)
        break

  @test_util.run_v2_only
  def testForceSelectTFOps(self):
    # [FUNCTION DOC]
    # Converts the simple variable model with only select-TF-ops as supported ops. Asserts the metadata flag
    # for forced select TF ops is true and the result matches TensorFlow.
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    converter.target_spec.supported_ops = [lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(metadata.options.forceSelectTfOps, True)

    # Check output value from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  def testExcludeConversionMetadata(self):
    # [FUNCTION DOC]
    # Converts with the option to exclude conversion metadata set; asserts that no metadata can be read.
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    converter.exclude_conversion_metadata = True
    tflite_model = converter.convert()
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(tflite_model)
    self.assertIsNone(metadata)

  def testConversionMetadataForDynamicRange(self):
    # [FUNCTION DOC]
    # Converts the ceil model with default optimizations (no calibration data); asserts the metadata
    # optimization mode is dynamic-range PTQ.
    func, _ = self._getCeilModel()
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func.get_concrete_function()]
    )
    converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_model = converter.convert()
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(quantized_model)
    self.assertIsNotNone(metadata)
    self.assertAllEqual(
        [metadata_fb.ModelOptimizationMode.PTQ_DYNAMIC_RANGE],
        metadata.options.modelOptimizationModes,
    )

  def testConversionMetadataForFloat16(self):
    # [FUNCTION DOC]
    # Converts the convolution model with default optimizations, calibration data and float16 supported
    # types; asserts the metadata optimization mode is float16 PTQ.
    root, func, calibration_gen = self._getIntegerQuantizeModel()
    converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_gen
    converter.target_spec.supported_types = [dtypes.float16]
    quantized_model = converter.convert()
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(quantized_model)
    self.assertIsNotNone(metadata)
    self.assertAllEqual(
        [metadata_fb.ModelOptimizationMode.PTQ_FLOAT16],
        metadata.options.modelOptimizationModes,
    )

  def testSerializeDebugMetadata(self):
    # [FUNCTION DOC]
    # Converts with the serialize-debug-metadata option on, reads the model as an object and asserts one of
    # the metadata entries is named debug_metadata.
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    converter.serialize_debug_metadata = True
    tflite_model = flatbuffer_utils.convert_bytearray_to_object(
        converter.convert()
    )

    # Check the debug metadata.
    metadata_names = [m.name for m in tflite_model.metadata]
    self.assertIn(b'debug_metadata', metadata_names)


class FromSavedModelTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case covering the version 2 converter when built from a saved model directory.
  # Inherits helper methods from the common test base. Scenarios: stable-HLO quantizer behaviour, TF1 saved
  # models, hash tables, variables, signatures, multi-function and shared-weight models, Keras models,
  # ConvLSTM, input shapes, quantization options (per-channel, bias type, dynamic range) and QDQ modes.

  def _createV1SavedModel(self, shape):
    # [FUNCTION DOC]
    # HELPER: creates a simple TF1-style saved model.
    # Parameter shape: input shape. Builds a graph with two placeholders (inputB, inputA) and a variable;
    # output = inputB + inputA * variable. Saves it with the simple-save helper in a temp subdirectory and
    # returns its path. Side effects: files written, uses a session.
    """Create a simple SavedModel."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        in_tensor_1 = tf.compat.v1.placeholder(
            shape=shape, dtype=tf.float32, name='inputB'
        )
        in_tensor_2 = tf.compat.v1.placeholder(
            shape=shape, dtype=tf.float32, name='inputA'
        )
        variable_node = tf.Variable(1.0, name='variable_node')
        out_tensor = in_tensor_1 + in_tensor_2 * variable_node
        inputs = {'x': in_tensor_1, 'y': in_tensor_2}
        outputs = {'z': out_tensor}
        sess.run(tf.compat.v1.variables_initializer([variable_node]))
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return saved_model_dir

  def _createV2QATSavedModel(self, shape):
    # [FUNCTION DOC]
    # HELPER: creates a TF2 quantization-aware-training saved model.
    # Parameter shape (unused by the body; the input shape is hard-coded to 32x32x128). Builds fake-quant,
    # convolution, fake-quant, reshape. Saves it and returns a triple: directory path, input name and output
    # name. The shape argument is ignored.
    """Create a simple QAT SavedModel in TF 2."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    input_name = 'input'
    output_name = 'scores'

    class _FakeQuantArgsLayer(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: Keras layer wrapper applying fake quantization with fixed range -3 to 3.
      """A fake quantization layer with fake_quant_with_min_max_args.

      Keras 3 requires wrapping the tf function inside Keras layer.
      """

      def call(self, x):
        # [FUNCTION DOC]
        # HELPER: returns input x fake-quantized with min -3 and max 3.
        return tf.quantization.fake_quant_with_min_max_args(x, -3.0, 3.0)

    input_tensor = tf.keras.layers.Input((32, 32, 128), name=input_name)
    x = _FakeQuantArgsLayer()(input_tensor)
    x = tf.keras.layers.Conv2D(1, (3, 3))(x)
    x = _FakeQuantArgsLayer()(x)
    scores = tf.keras.layers.Reshape((-1,), name=output_name)(x)
    model = tf.keras.Model(input_tensor, scores)
    model.save(saved_model_dir)
    return saved_model_dir, input_name, output_name

  @test_util.run_v2_only
  def testStableHloQuantizerSupportsOnlyStaticRangePtq(self):
    # [FUNCTION DOC]
    # Saves a trivial doubling model, converts it with the stable-HLO quantizer flag set and no calibration.
    # Expected: value error stating only static-range and weight-only PTQ are supported.
    """Tests that StableHLO Quantizer supports only static-range PTQ."""
    input_data = tf.constant(1.0, shape=[1])
    root = autotrackable.AutoTrackable()
    root.f = tf.function(lambda x: 2.0 * x)
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, to_save)

    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    converter.experimental_use_stablehlo_quantizer = True
    with self.assertRaisesRegex(
        ValueError, 'only supports static-range and weight-only PTQ'
    ):
      converter.convert()

  @test_util.run_v2_only
  def testStableHloQuantizerNoOpForTfSavedModel(self):
    # [FUNCTION DOC]
    # Saves a trivial doubling model, enables the stable-HLO quantizer with default optimizations and a
    # representative dataset. Asserts conversion works and every tensor remains float32 (quantizer is a no-op
    # for a TF saved model).
    """Tests that StableHLO Quantizer does not run for TF SavedModel."""
    input_data = tf.constant(1.0, shape=[1])
    root = autotrackable.AutoTrackable()
    root.f = tf.function(lambda x: 2.0 * x)
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, to_save)

    def _representative_data_gen():
      # [FUNCTION DOC]
      # HELPER: returns a list holding one mapping from x to a ones array of shape [1] (float32).
      return [{'x': np.ones(shape=(1,), dtype=np.float32)}]

    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    # Set the flags to enable StableHLO Quantizer.
    converter.experimental_use_stablehlo_quantizer = True
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_data_gen

    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Test that no tensor is quantized.
    interp = interpreter.Interpreter(model_content=tflite_model)
    all_tensor_details = interp.get_tensor_details()
    for tensor_detail in all_tensor_details:
      self.assertIn('dtype', tensor_detail)
      self.assertEqual(tensor_detail['dtype'], np.float32)

  @test_util.run_v2_only
  def testV1SimpleModel(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: several shape checks use a truthiness assertion with the shape as the second (message)
    # argument, so they pass for any shape and verify nothing.
    # Converts a TF1 saved model with input shape 1x16x16x3. Asserts two inputs named starting with inputA and
    # inputB, one output starting with add, float32 dtypes, zero quantization.
    """Test a SavedModel."""
    with tf.Graph().as_default():
      saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])

      # Convert model and ensure model is not None.
      converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
      tflite_model = converter.convert()
      self.assertTrue(tflite_model)

      interp = interpreter.Interpreter(model_content=tflite_model)
      interp.allocate_tensors()

      input_details = interp.get_input_details()
      self.assertLen(input_details, 2)
      self.assertStartsWith(input_details[0]['name'], 'inputA')
      self.assertEqual(np.float32, input_details[0]['dtype'])
      self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
      self.assertEqual((0.0, 0.0), input_details[0]['quantization'])

      self.assertStartsWith(
          input_details[1]['name'],
          'inputB',
      )
      self.assertEqual(np.float32, input_details[1]['dtype'])
      self.assertTrue([1, 16, 16, 3], input_details[1]['shape'])
      self.assertEqual((0.0, 0.0), input_details[1]['quantization'])

      output_details = interp.get_output_details()
      self.assertLen(output_details, 1)
      self.assertStartsWith(output_details[0]['name'], 'add')
      self.assertEqual(np.float32, output_details[0]['dtype'])
      self.assertTrue([1, 16, 16, 3], output_details[0]['shape'])
      self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

  @parameterized.named_parameters(
      ('Default', False),
      ('UnfoldLargeConstant', True),
  )
  @test_util.run_v2_only
  def testUnfoldLargeConstant(self, unfold_large_constant):
    # [FUNCTION DOC]
    # Parameterized: unfold large splat constant off or on. Saves a model adding a 1000x1000 constant of ones
    # to the input, converts with the private option, and asserts operator order (fill then add when
    # unfolded; only add otherwise), tensor names, dtype, shape, and that running with ones gives all twos.
    """Test unfolding large splat constant in a TF Lite model."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        in_tensor = tf.compat.v1.placeholder(
            shape=[1000, 1000], dtype=tf.float32, name='input'
        )
        constant = tf.constant(value=1, dtype=tf.float32, shape=[1000, 1000])
        out_tensor = in_tensor + constant
        inputs = {'x': in_tensor}
        outputs = {'y': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter._experimental_unfold_large_splat_constant = unfold_large_constant
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    model = util._convert_model_from_bytearray_to_object(tflite_model)
    if unfold_large_constant:
      self.assertEqual(
          model.operatorCodes[0].builtinCode, schema_fb.BuiltinOperator.FILL
      )
      self.assertEqual(
          model.operatorCodes[1].builtinCode, schema_fb.BuiltinOperator.ADD
      )
    else:
      self.assertEqual(
          model.operatorCodes[0].builtinCode, schema_fb.BuiltinOperator.ADD
      )

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()

    input_details = interp.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual('input:0', input_details[0]['name'])
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([1000, 1000], input_details[0]['shape'])
    self.assertEqual((0.0, 0.0), input_details[0]['quantization'])

    output_details = interp.get_output_details()
    self.assertEqual('add:0', output_details[0]['name'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    self.assertAllEqual([1000, 1000], output_details[0]['shape'])
    self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    interp.set_tensor(
        input_details[0]['index'], np.ones(shape=[1000, 1000], dtype=np.float32)
    )
    interp.invoke()
    self.assertAllEqual(
        np.full(shape=[1000, 1000], fill_value=2.0, dtype=np.float32),
        interp.get_tensor(output_details[0]['index']),
    )

  @test_util.run_v2_only
  def testPreserveAssert(self):
    # [FUNCTION DOC]
    # Saves a model with an assert op guarding an add, converts with builtin and select-TF ops and the private
    # preserve-assert option, then asserts the model has a flex custom operator code for Assert.
    """Test preserving AssertOp in a TF Lite model."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        in_tensor = tf.compat.v1.placeholder(
            shape=[10, 10], dtype=tf.float32, name='input'
        )
        constant = tf.constant(value=1, dtype=tf.float32, shape=[10, 10])
        assert_op = tf.Assert(tf.less_equal(in_tensor, constant), [in_tensor])
        with tf.control_dependencies([assert_op]):
          out_tensor = in_tensor + constant
        inputs = {'x': in_tensor}
        outputs = {'y': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_preserve_assert_op = True
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    model = util._convert_model_from_bytearray_to_object(tflite_model)
    has_assert = False
    for op_code in model.operatorCodes:
      if op_code.customCode == b'FlexAssert':
        has_assert = True
        break
    self.assertTrue(has_assert)

  @test_util.run_v2_only
  def testTF1HubFormattedModel(self):
    # [FUNCTION DOC]
    # Creates a TF1 saved model, rewrites its saved-model proto with the schema version set to zero (mimics a
    # TF1 hub model), writes it back and asserts conversion still succeeds. Side effect: modifies the file.
    """Test a TF1 hub formatted model."""
    saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])

    # TF1 hub model is based on V1 saved model and they omit the saved model
    # schema version setting.
    saved_model_proto = loader_impl.parse_saved_model(saved_model_dir)
    saved_model_proto.saved_model_schema_version = 0

    saved_model_pb_file_path = os.path.join(saved_model_dir, 'saved_model.pb')
    with file_io.FileIO(saved_model_pb_file_path, 'wb') as writer:
      writer.write(saved_model_proto.SerializeToString())

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

  def _createV1ModelWithHashTableInitializer(self):
    # [FUNCTION DOC]
    # HELPER: builds a TF1 saved model with a static hash table and its table initializer.
    # Disables eager execution, builds a string-to-int64 table (keys a to d, values 1 to 4, default -1) and
    # a lookup on a string placeholder, saves with the SavedModel builder using the tables initializer as main
    # op, then resets the graph and re-enables eager execution. Returns the saved model path.
    # Create a v1 saved model with hash table initializers.
    tf.compat.v1.disable_eager_execution()
    saved_model_dir = os.path.join(
        self.get_temp_dir(), 'savedmodel_with_hashtable'
    )

    table_initializer = tf.lookup.KeyValueTensorInitializer(
        keys=['a', 'b', 'c', 'd'],
        values=[1, 2, 3, 4],
        key_dtype=tf.string,
        value_dtype=tf.int64,
    )
    table = tf.lookup.StaticHashTable(
        table_initializer, default_value=tf.constant(-1, dtype=tf.int64)
    )

    x = tf.compat.v1.placeholder(tf.string, shape=(), name='input')
    y = table.lookup(x)

    tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(y)

    signature_def_map, init_op, assets_collection = (
        {
            'serving_default': tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs={'x': tensor_info_x},
                outputs={'y': tensor_info_y},
                method_name='some_function',
            )
        },
        tf.compat.v1.tables_initializer(),
        None,
    )

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.initializers.global_variables())

    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(
        saved_model_dir
    )
    builder.add_meta_graph_and_variables(
        sess,
        [tf.compat.v1.saved_model.tag_constants.SERVING],
        signature_def_map,
        main_op=init_op,
        assets_collection=assets_collection,
        strip_default_attrs=True,
    )
    builder.save()

    # Restore TF v2 behavior.
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.enable_eager_execution()
    return saved_model_dir

  @test_util.run_v2_only
  def testModelWithHashTableInitializer(self):
    # [FUNCTION DOC]
    # Converts the hash table model, resizes the input to 4 strings (a, b, c, z), invokes three times and each
    # time asserts [1, 2, 3, -1], verifying the initializer graph does not break repeated invocation.
    """Test a model with saved_model's session initializer for hash tables."""
    saved_model_dir = self._createV1ModelWithHashTableInitializer()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    input_data = np.array(['a', 'b', 'c', 'z'], dtype=np.bytes_)
    interp.resize_tensor_input(input_details[0]['index'], [4], strict=False)
    interp.allocate_tensors()

    interp.set_tensor(input_details[0]['index'], input_data)

    # Invoke multiple times to ensure the initializer graph runs only once.
    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual([1, 2, 3, -1], list(actual_value))

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual([1, 2, 3, -1], list(actual_value))

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual([1, 2, 3, -1], list(actual_value))

  def _createV1ModelWithMutableHashTable(self):
    # [FUNCTION DOC]
    # HELPER: builds a TF1 saved model with a mutable hash table that inserts keys a and b (values 1 and 5)
    # before a lookup of the input. Same save/reset procedure as the static hash table helper. Returns the
    # saved model path.
    # Create a v1 saved model with mutable hash table.
    tf.compat.v1.disable_eager_execution()
    saved_model_dir = os.path.join(
        self.get_temp_dir(), 'savedmodel_with_mutable_hashtable'
    )

    table = tf.raw_ops.MutableHashTableV2(
        key_dtype=tf.string, value_dtype=tf.int64
    )
    x = tf.compat.v1.placeholder(tf.string, shape=(), name='input')
    keys = tf.constant(['a', 'b'], tf.string)
    values = tf.constant([1, 5], tf.int64)
    default_value = tf.constant(-1, tf.int64)
    insert_call = tf.raw_ops.LookupTableInsertV2(
        table_handle=table, keys=keys, values=values
    )
    with tf.control_dependencies([insert_call]):
      y = tf.raw_ops.LookupTableFindV2(
          table_handle=table, keys=x, default_value=default_value
      )

    tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(y)

    signature_def_map, init_op, assets_collection = (
        {
            'serving_default': tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs={'x': tensor_info_x},
                outputs={'y': tensor_info_y},
                method_name='some_function',
            )
        },
        tf.compat.v1.tables_initializer(),
        None,
    )

    sess = tf.compat.v1.Session()

    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(
        saved_model_dir
    )
    builder.add_meta_graph_and_variables(
        sess,
        [tf.compat.v1.saved_model.tag_constants.SERVING],
        signature_def_map,
        main_op=init_op,
        assets_collection=assets_collection,
        strip_default_attrs=True,
    )
    builder.save()

    # Restore TF v2 behavior.
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.enable_eager_execution()
    return saved_model_dir

  @test_util.run_v2_only
  def testModelWithMutableHashTable(self):
    # [FUNCTION DOC]
    # Converts the mutable hash table model with builtin and select-TF ops, feeds a, b, c (resized input) and
    # asserts the lookup result [1, 5, -1].
    """Test a model with saved_model's session initializer for hash tables."""
    saved_model_dir = self._createV1ModelWithMutableHashTable()

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    input_data = np.array(['a', 'b', 'c'], dtype=np.bytes_)
    interp.resize_tensor_input(input_details[0]['index'], [3], strict=False)
    interp.allocate_tensors()

    interp.set_tensor(input_details[0]['index'], input_data)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual([1, 5, -1], list(actual_value))

  @test_util.run_v2_only
  def testReduceSumWithInt16Quant(self):
    # [FUNCTION DOC]
    # Builds a small Keras model with a reduce-sum over the last axis, converts with the int16-activation op
    # set, int16 inference types, default optimizations and a random representative dataset. Runs the
    # signature runner on int16 data and asserts the sums [3, 6, 9].
    """Test a model with quantized int16 reduce sum op."""
    inp = tf.keras.Input([3, 3], 3, name='x')
    m = tf.keras.Model(inp, tf.reduce_sum(inp, axis=-1))

    converter = lite.TFLiteConverterV2.from_keras_model(m)
    converter.target_spec.supported_ops = [
        lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16
    converter.optimizations = [lite.Optimize.DEFAULT]
    inputs = {
        i.name: np.random.normal(size=i.shape).astype(np.float32)
        for i in m.inputs
    }
    converter.representative_dataset = lambda: [inputs]
    content = converter.convert()

    interp = interpreter.Interpreter(model_content=content)
    runner = interp.get_signature_runner('serving_default')
    y = runner(x=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).astype(np.int16))
    self.assertEqual([3, 6, 9], list(list(y.values())[0]))

  @test_util.run_v2_only
  def testConstModel(self):
    # [FUNCTION DOC]
    # Saves a model that doubles its input and converts from the saved model. Asserts the value matches
    # TensorFlow (functions are inlined).
    """Test a basic model with functions to make sure functions are inlined."""
    input_data = tf.constant(1.0, shape=[1])
    root = autotrackable.AutoTrackable()
    root.f = tf.function(lambda x: 2.0 * x)
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, to_save)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testVariableModel(self):
    # [FUNCTION DOC]
    # Saves the simple variable model, converts, asserts metadata model type is saved model and the result
    # matches TensorFlow.
    """Test a basic model with Variables with saving/loading the SavedModel."""
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[1])
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, to_save)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    tflite_model = converter.convert()
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(
        metadata.environment.modelType, metadata_fb.ModelType.TF_SAVED_MODEL
    )

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @parameterized.named_parameters(
      ('EnableResourceVariables', True), ('DisableResourceVariables', False)
  )
  @test_util.run_v2_only
  def testNativeVariablesModel(self, enable_resource_variables):
    # [FUNCTION DOC]
    # KNOWN ISSUE: both parameterizations execute the same path; the disabled-case error assertion is
    # commented out, so the flag value is not distinguished.
    # Parameterized: resource variables enabled or disabled. Saves a model with variables, converts with the
    # resource-variables option, and compares results (the failing-conversion check for the disabled case is
    # commented out).
    """Test a basic model with Variables with saving/loading the SavedModel."""
    root = self._getSimpleModelWithVariables()
    input_data = tf.constant(1.0, shape=[1, 10])
    to_save = root.assign_add.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, to_save)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    converter.experimental_enable_resource_variables = enable_resource_variables

    # TODO(b/355497070): Remove this check once the
    # CreateFreezeGlobalTensorsPass is migrated to the new TFL::Pass
    # in the converter.

    # if not enable_resource_variables:
    #   with self.assertRaises(convert.ConverterError) as error:
    #     tflite_model = converter.convert()
    #   self.assertIn(
    #       'is not immutable, try removing mutable variables in your model
    #       ' since mutable variables are currently not supported through this'
    #       ' converter',
    #       str(error.exception),
    #   )
    #   return

    # Enable resource variables.
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.assign_add(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    for tf_result, tflite_result in zip(expected_value, actual_value[0]):
      self.assertAllClose(tf_result, tflite_result, atol=1e-05)

  @test_util.run_v2_only
  def testSignatures(self):
    # [FUNCTION DOC]
    # Asserts that converting from a saved model with an invalid signature key raises a value error mentioning
    # the invalid key, and that an empty signature key list still converts and gives the right value.
    """Test values for `signature_keys` argument."""
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[1])
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, to_save)

    # Convert model with invalid `signature_keys`.
    with self.assertRaises(ValueError) as error:
      _ = lite.TFLiteConverterV2.from_saved_model(
          save_dir, signature_keys=['INVALID']
      )
    self.assertIn("Invalid signature key 'INVALID'", str(error.exception))

    # Convert model with empty `signature_keys`.
    converter = lite.TFLiteConverterV2.from_saved_model(
        save_dir, signature_keys=[]
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testSignatureDefsWithFullIntegerQuantization(self):
    # [FUNCTION DOC]
    # Compares a float TFLite model and an int8-I/O quantized TFLite model (no representative data) built from
    # the same QAT saved model. Checks signature tensor indices exist in the model, runs both with signature
    # evaluation (quantizing inputs, dequantizing outputs when scales exist) and asserts root mean square
    # difference below 1.0.
    # SETUP
    # 1. Define input shapes
    tf_input_shape = (32, 32, 128)
    tflite_input_shape = (1,) + tf_input_shape
    # 2. Define model
    tf_saved_model_dir, input_name, output_name = self._createV2QATSavedModel(
        tf_input_shape
    )

    # MODEL 1: TFLite (float) model
    # 1. Create TFLite model
    converter = lite.TFLiteConverterV2.from_saved_model(tf_saved_model_dir)
    converter.optimizations = [lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    # 2. Initialize the Interpreter
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()[0]
    interp.resize_tensor_input(input_details['index'], tflite_input_shape)
    interp.allocate_tensors()
    # 3. (Skip) Verify that signature def input/output tensors are in the model.
    # 4. Evaluate the model
    input_data = np.random.random(tflite_input_shape).astype(np.float32)
    result = self._evaluateTFLiteModelUsingSignatureDef(
        tflite_model, 'serving_default', {input_name: input_data}
    )[output_name]

    # MODEL 2: TFLite (full integer quantized) model
    # 1. Create TFLite model
    converter = lite.TFLiteConverterV2.from_saved_model(tf_saved_model_dir)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model_quant = converter.convert()
    # 2. Initialize the Interpreter
    interp = interpreter.Interpreter(model_content=tflite_model_quant)
    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]
    interp.resize_tensor_input(input_details['index'], tflite_input_shape)
    interp.allocate_tensors()
    # 3. Verify that signature def input/output tensors are in the model.
    all_indices = {item['index'] for item in interp.get_tensor_details()}
    signature_list = interp._get_full_signature_list()['serving_default']
    input_tensor_indices = set(signature_list['inputs'].values())
    assert input_tensor_indices.issubset(all_indices)
    output_tensor_indices = set(signature_list['outputs'].values())
    assert output_tensor_indices.issubset(all_indices)

    # 4. Evaluate the model
    input_data = np.random.random(tflite_input_shape)
    input_scale, input_zero_point = input_details['quantization']
    if (input_scale, input_zero_point) != (0.0, 0):
      input_data = input_data / input_scale + input_zero_point
      input_data = input_data.astype(input_details['dtype'])
    result_quant = self._evaluateTFLiteModelUsingSignatureDef(
        tflite_model_quant, 'serving_default', {input_name: input_data}
    )[output_name]
    output_scale, output_zero_point = output_details['quantization']
    if (output_scale, output_zero_point) != (0.0, 0):
      result_quant = result_quant.astype(np.float32)
      result_quant = (result_quant - output_zero_point) * output_scale

    # COMPARE: Validate that results from both models are approx. the same.
    root_mean_squared = np.sqrt(np.mean((result - result_quant) ** 2))
    assert root_mean_squared < 1.0

  @test_util.run_v2_only
  def testSignatureDefs(self):
    # [FUNCTION DOC]
    # Saves a multi-function model exposing only mul_add, converts with that signature key, and asserts one
    # signature with inputs x and y and output output_0, with the right value from signature evaluation.
    """Test converting SignatureDef is correct and uses SignatureDef API."""
    root = self._getMultiFunctionModel()
    input_data_0 = tf.constant(1.0, shape=[1])
    input_data_1 = tf.constant(3.0, shape=[1])
    mul_add_func = root.mul_add.get_concrete_function(
        input_data_1, input_data_0
    )

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, {'mul_add': mul_add_func})

    converter = lite.TFLiteConverterV2.from_saved_model(
        save_dir, signature_keys=['mul_add']
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.mul_add(input_data_1, input_data_0)
    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()
    results = self._evaluateTFLiteModelUsingSignatureDef(
        tflite_model, 'mul_add', {'y': input_data_0, 'x': input_data_1}
    )
    self.assertEqual(list(results.keys()), ['output_0'])
    self.assertEqual(expected_value.numpy(), results['output_0'])

    # Verify the SignatureDef structure returned is as expected.
    self.assertLen(signature_defs, 1)
    self.assertEqual(list(signature_defs.keys()), ['mul_add'])
    self.assertLen(signature_defs.values(), 1)
    self.assertEqual(
        list(signature_defs['mul_add'].keys()), ['inputs', 'outputs']
    )
    self.assertCountEqual(signature_defs['mul_add']['inputs'], ['x', 'y'])
    self.assertEqual(list(signature_defs['mul_add']['outputs']), ['output_0'])

  @test_util.run_v2_only
  def testSignatureDefsWithDefaultValue(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the test uses None as the signature key for the conversion, but the
    # converter is still created with the explicit key; only the evaluation uses the default key.
    # Same as the previous test, but evaluates the model with the signature key given as nothing (default).
    """Test converting SignatureDef is correct and uses SignatureDef API.

    This test uses None as signature_key to test default behavior.
    """
    root = self._getMultiFunctionModel()
    input_data_0 = tf.constant(1.0, shape=[1])
    input_data_1 = tf.constant(3.0, shape=[1])
    mul_add_func = root.mul_add.get_concrete_function(
        input_data_1, input_data_0
    )

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, {'mul_add': mul_add_func})

    converter = lite.TFLiteConverterV2.from_saved_model(
        save_dir, signature_keys=['mul_add']
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.mul_add(input_data_1, input_data_0)
    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()
    results = self._evaluateTFLiteModelUsingSignatureDef(
        tflite_model, None, {'y': input_data_0, 'x': input_data_1}
    )
    self.assertEqual(list(results.keys()), ['output_0'])
    self.assertEqual(expected_value.numpy(), results['output_0'])

    # Verify the SignatureDef structure returned is as expected.
    self.assertLen(signature_defs, 1)
    self.assertEqual(list(signature_defs.keys()), ['mul_add'])
    self.assertLen(signature_defs.values(), 1)
    self.assertEqual(
        list(signature_defs['mul_add'].keys()), ['inputs', 'outputs']
    )
    self.assertCountEqual(signature_defs['mul_add']['inputs'], ['x', 'y'])
    self.assertEqual(list(signature_defs['mul_add']['outputs']), ['output_0'])

  @test_util.run_v2_only
  def testSignatureDefsQuantizedModel(self):
    # [FUNCTION DOC]
    # Converts the mul_add signature with default optimizations, a representative dataset of named inputs x
    # and y, and int8 builtins; asserts the signature structure (inputs x and y, output output_0).
    """Test converting SignatureDef on quantized model."""
    root = self._getMultiFunctionModel()
    input_data_0 = tf.constant(1.0, shape=[1])
    input_data_1 = tf.constant(3.0, shape=[1])
    mul_add_func = root.mul_add.get_concrete_function(
        input_data_1, input_data_0
    )

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, {'mul_add': mul_add_func})

    converter = lite.TFLiteConverterV2.from_saved_model(
        save_dir, signature_keys=['mul_add']
    )

    def representative_dataset_gen():
      # [FUNCTION DOC]
      # HELPER: yields two mappings from x and y to random 1x1 float32 arrays.
      for _ in range(2):
        yield {
            'x': np.random.uniform(low=0, high=1, size=(1, 1)).astype(
                np.float32
            ),
            'y': np.random.uniform(low=0, high=1, size=(1, 1)).astype(
                np.float32
            ),
        }

    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()

    # Check signatures are valid from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()

    # Verify the SignatureDef structure returned is as expected.
    self.assertLen(signature_defs, 1)
    self.assertEqual(list(signature_defs.keys()), ['mul_add'])
    self.assertLen(signature_defs.values(), 1)
    self.assertEqual(
        list(signature_defs['mul_add'].keys()), ['inputs', 'outputs']
    )
    self.assertCountEqual(signature_defs['mul_add']['inputs'], ['x', 'y'])
    self.assertEqual(list(signature_defs['mul_add']['outputs']), ['output_0'])

  @test_util.run_v2_only
  def testMultipleFunctionModel(self):
    # [FUNCTION DOC]
    # Saves add and sub functions as signatures, converts, and asserts two signatures with correct inputs and
    # outputs and runner results (add gives 3, sub gives -2).
    """Convert multiple functions in a multi-functional model."""
    root = self._getMultiFunctionModel()
    input_data = tf.constant(1.0, shape=[1])
    add_func = root.add.get_concrete_function(input_data)
    sub_func = root.sub.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, {'add': add_func, 'sub': sub_func})

    # Try converting multiple functions.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()

    # Verify the SignatureDef structure returned is as expected.
    self.assertLen(signature_defs, 2)
    self.assertEqual(list(signature_defs.keys()), ['add', 'sub'])
    self.assertLen(signature_defs.values(), 2)
    self.assertEqual(list(signature_defs['add'].keys()), ['inputs', 'outputs'])
    self.assertCountEqual(signature_defs['add']['inputs'], ['x'])
    self.assertEqual(list(signature_defs['add']['outputs']), ['output_0'])
    self.assertEqual(list(signature_defs['sub'].keys()), ['inputs', 'outputs'])
    self.assertCountEqual(signature_defs['sub']['inputs'], ['x'])
    self.assertEqual(list(signature_defs['sub']['outputs']), ['output_0'])

    # Verify the Signature runner executions.
    add_signature_runner = interp.get_signature_runner('add')
    add_output = add_signature_runner(x=input_data)
    self.assertEqual(add_output['output_0'], 3)

    sub_signature_runner = interp.get_signature_runner('sub')
    sub_output = sub_signature_runner(x=input_data)
    self.assertEqual(sub_output['output_0'], -2)

  @parameterized.named_parameters(
      ('_Default', False, False, dtypes.float32, False),
      ('_DefaultMlirQuant', False, False, dtypes.float32, True),
      ('_INT8InputOutput', False, False, dtypes.int8),
      ('_UINT8InputOutput', False, False, dtypes.uint8),
      ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16),
      ('_IntOnly_INT8InputOutput', True, False, dtypes.int8),
      ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8),
      ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16),
      ('_IntOnly_INT8InputOutputMlirQuant', True, False, dtypes.int8, True),
      ('_IntOnly_UINT8InputOutputMlirQuant', True, False, dtypes.uint8, True),
  )
  @test_util.run_v2_only
  def testMultipleFunctionQuantizedModel(
      self,
      is_int_only,
      is_int16_quantize,
      inference_input_output_type,
      enable_mlir_quantizer=False,
  ):
    # [FUNCTION DOC]
    # Parameterized over integer-only mode, int16 mode, inference dtype and an optional MLIR flag. Converts
    # the add/sub saved model with a representative dataset yielding the signature name together with inputs.
    # Asserts signature structure, runner outputs not null, input/output names, dtype equal to the requested
    # type, shape [1], and zero quantization when float32.
    """Convert multiple functions in a multi-functional model."""
    root = self._getMultiFunctionModel()
    input_data = tf.constant(1.0, shape=[1])
    add_func = root.add.get_concrete_function(input_data)
    sub_func = root.sub.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, {'add': add_func, 'sub': sub_func})

    # Try converting multiple functions.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)

    def representative_dataset_gen():
      # [FUNCTION DOC]
      # HELPER: yields pairs of (signature name, input mapping); two for add and two for sub, with random
      # float32 x of length 1.
      for _ in range(2):
        yield (
            'add',
            {
                'x': np.random.uniform(low=0, high=1, size=(1,)).astype(
                    np.float32
                ),
            },
        )
      for _ in range(2):
        yield (
            'sub',
            {
                'x': np.random.uniform(low=0, high=1, size=(1,)).astype(
                    np.float32
                ),
            },
        )

    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    if is_int_only:
      if is_int16_quantize:
        converter.target_spec.supported_ops = [
            lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
      else:
        converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
    else:
      if is_int16_quantize:
        converter.target_spec.supported_ops = [
            lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
      else:
        converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = inference_input_output_type
    converter.inference_output_type = inference_input_output_type
    converter.experimental_new_quantizer = enable_mlir_quantizer
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()

    # Verify the SignatureDef structure returned is as expected.
    self.assertLen(signature_defs, 2)
    self.assertEqual(list(signature_defs.keys()), ['add', 'sub'])
    self.assertLen(signature_defs.values(), 2)
    self.assertEqual(list(signature_defs['add'].keys()), ['inputs', 'outputs'])
    self.assertCountEqual(signature_defs['add']['inputs'], ['x'])
    self.assertEqual(list(signature_defs['add']['outputs']), ['output_0'])
    self.assertEqual(list(signature_defs['sub'].keys()), ['inputs', 'outputs'])
    self.assertCountEqual(signature_defs['sub']['inputs'], ['x'])
    self.assertEqual(list(signature_defs['sub']['outputs']), ['output_0'])

    # Verify the Signature runner executions.
    input_data = tf.constant(
        np.random.uniform(-1, 1, size=(1,)).astype(
            inference_input_output_type.as_numpy_dtype
        )
    )
    add_signature_runner = interp.get_signature_runner('add')
    add_output = add_signature_runner(x=input_data)
    self.assertIsNotNone(add_output['output_0'])
    input_details = add_signature_runner.get_input_details()
    self.assertLen(input_details, 1)
    self.assertStartsWith(input_details['x']['name'], 'add_x:0')
    self.assertEqual(
        inference_input_output_type.as_numpy_dtype, input_details['x']['dtype']
    )
    self.assertTrue(([1] == input_details['x']['shape']).all())
    if inference_input_output_type == dtypes.float32:
      self.assertEqual((0.0, 0), input_details['x']['quantization'])

    sub_signature_runner = interp.get_signature_runner('sub')
    sub_output = sub_signature_runner(x=input_data)
    self.assertIsNotNone(sub_output['output_0'])
    output_details = sub_signature_runner.get_output_details()
    self.assertLen(output_details, 1)
    self.assertStartsWith(
        output_details['output_0']['name'], 'StatefulPartitionedCall_1:0'
    )
    self.assertEqual(
        inference_input_output_type.as_numpy_dtype,
        output_details['output_0']['dtype'],
    )
    self.assertTrue(([1] == output_details['output_0']['shape']).all())
    if inference_input_output_type == dtypes.float32:
      self.assertEqual((0.0, 0), output_details['output_0']['quantization'])

  @test_util.run_v2_only
  def testMultipleFunctionModelWithSharedWeight(self):
    # [FUNCTION DOC]
    # Saves add, sub and mul functions sharing weights, converts, asserts the flatbuffer is smaller than
    # 1.1 million bytes (weights shared) and that three signatures with runners exist.
    """Convert multiple functions with the shared weight."""
    root = self._getMultiFunctionModelWithSharedWeight()
    input_data = tf.constant(1.0, shape=[1])
    add_func = root.add.get_concrete_function(input_data)
    sub_func = root.sub.get_concrete_function(input_data)
    mul_func = root.mul.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(
        root, save_dir, {'add': add_func, 'sub': sub_func, 'mul': mul_func}
    )

    # Try converting multiple functions.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Make sure that the weight tensors are shared.
    self.assertLess(len(tflite_model), 1100000)

    interp = lite.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()
    self.assertLen(signature_defs, 3)
    add_signature_runner = interp.get_signature_runner('add')
    sub_signature_runner = interp.get_signature_runner('sub')
    mul_signature_runner = interp.get_signature_runner('mul')
    self.assertIsNotNone(add_signature_runner)
    self.assertIsNotNone(sub_signature_runner)
    self.assertIsNotNone(mul_signature_runner)

  @test_util.run_v2_only
  def testNoConcreteFunctionModel(self):
    # [FUNCTION DOC]
    # Saves a model without signatures; converting must raise a value error saying at least one signature key
    # is required.
    root = self._getMultiFunctionModel()

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir)

    with self.assertRaises(ValueError) as error:
      _ = lite.TFLiteConverterV2.from_saved_model(save_dir)
    self.assertIn(
        'Only support at least one signature key.', str(error.exception)
    )

  @test_util.run_v2_only
  def testKerasSequentialModel(self):
    # [FUNCTION DOC]
    # Trains a tiny dropout+dense Keras model, saves it as a saved model, converts and asserts that the
    # output equals the Keras prediction.
    """Test a simple sequential tf.Keras model."""
    input_data = tf.constant(1.0, shape=[1, 1])

    x = np.array([[1.0], [2.0]])
    y = np.array([[2.0], [4.0]])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(model, save_dir)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = model.predict(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value, actual_value)

  @test_util.run_v2_only
  def testKerasSequentialModelExport(self):
    # [FUNCTION DOC]
    # Same model, but exported with the export method. Converts and asserts a single serving_default signature
    # and equal prediction results.
    """Test a simple sequential tf.Keras model with `model.export` usage."""
    input_data = tf.constant(1.0, shape=[1, 1])

    x = np.array([[1.0], [2.0]])
    y = np.array([[2.0], [4.0]])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)

    export_dir = os.path.join(self.get_temp_dir(), 'exported_model')
    model.export(export_dir)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(export_dir)
    tflite_model = converter.convert()

    # Validate endpoints following `.export` to TFLite conversion.
    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()
    self.assertLen(signature_defs, 1)
    self.assertEqual(next(iter(signature_defs)), 'serving_default')

    # Check values from converted model.
    expected_value = model.predict(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value, actual_value)

  @test_util.run_v2_only
  def testGraphDebugInfo(self):
    # [FUNCTION DOC]
    # Saves a model with save-debug-info option on, converts and asserts valid debug info was captured.
    """Test a SavedModel has debug info captured."""
    input_data = tf.constant(1.0, shape=[1])
    root = autotrackable.AutoTrackable()
    root.f = tf.function(lambda x: 2.0 * x)
    to_save = root.f.get_concrete_function(input_data)
    options = save_options.SaveOptions(save_debug_info=True)
    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, to_save, options)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
    converter.convert()
    self._assertValidDebugInfo(converter._debug_info)

  @test_util.run_v2_only
  def testNonStatefulConvLSTM2D(self):
    # [FUNCTION DOC]
    # Saves a non-stateful ConvLSTM2D Keras model (batch input shape fixed) and converts with builtin and
    # select-TF ops; only asserts a non-empty model.
    """Test saved model with non stateful ConvLSTM2D keras layer."""
    # Create keras model
    model = tf.keras.Sequential([
        tf.keras.layers.ConvLSTM2D(
            32,
            (3, 3),
            padding='same',
            return_sequences=True,
            stateful=False,
            batch_input_shape=(1, 1, 10, 10, 1),
        )
    ])
    model.compile()

    # Export the keras model to saved model.
    saved_model_dir = os.path.join(self.get_temp_dir(), 'conv_lstm_2d')
    model.save(saved_model_dir, save_format='tf', include_optimizer=False)

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

  @test_util.run_v2_only
  def testKerasConvLSTM2DWithMoreThanOneDilationRate(self):
    # [FUNCTION DOC]
    # Saves a ConvLSTM2D model with dilation rate 2 and converts with builtin and select-TF ops; asserts a
    # non-empty model.
    input_tensor = tf.keras.layers.Input(
        batch_size=8,
        shape=[9, 10, 11, 12],
        name='input_tensor',
        dtype=tf.float32,
    )

    output = tf.keras.layers.ConvLSTM2D(
        filters=3,
        kernel_size=3,
        strides=1,
        padding='VALID',
        dilation_rate=2,
        use_bias=False,
        bias_initializer='ones',
        data_format='channels_last',
    )(input_tensor)

    model = tf.keras.Model(inputs=[input_tensor], outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Export the keras model to saved model.
    saved_model_dir = os.path.join(
        self.get_temp_dir(), 'conv_lstm_2d_with_dilation_rate'
    )
    model.save(saved_model_dir, save_format='tf', include_optimizer=False)

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

  @test_util.run_v2_only
  def testKerasFullyConnectedOutputShape3D(self):
    # [FUNCTION DOC]
    # Builds a fake-quant, dense and fake-quant Keras model with 3D output, saves, converts with default
    # optimizations, runs on a 1x3x3 input and asserts a rank-3 shape signature, results within 0.1 of Keras
    # and shape signature equal to the Keras layer output shape.
    """Create a simple FullyConnected Model with an output of three dimensions."""
    input_tensor = tf.keras.layers.Input(
        batch_size=1, shape=[3, 3], name='input_tensor', dtype=tf.float32
    )

    class _FakeQuantArgsLayer(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: Keras layer wrapper applying fake quantization with fixed range -3 to 3.
      """A fake quantization layer with fake_quant_with_min_max_args.

      Keras 3 requires wrapping the tf function inside Keras layer.
      """

      def call(self, x):
        # [FUNCTION DOC]
        # HELPER: returns input x fake-quantized with min -3 and max 3.
        return tf.quantization.fake_quant_with_min_max_args(x, -3.0, 3.0)

    x = _FakeQuantArgsLayer()(input_tensor)
    x = tf.keras.layers.Dense(3)(x)
    x = _FakeQuantArgsLayer()(x)
    model = tf.keras.Model(input_tensor, x)

    model.compile(
        optimizer='adam', loss='mean_squared_error', metrics=['accuracy']
    )

    # Export the keras model to saved model.
    saved_model_dir = os.path.join(
        self.get_temp_dir(), 'fully_connected_output_3d'
    )
    model.save(saved_model_dir, save_format='tf', include_optimizer=False)
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.optimizations = [lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    interp = interpreter.Interpreter(model_content=tflite_model)
    output_details = interp.get_output_details()
    input_details = interp.get_input_details()
    interp.allocate_tensors()

    input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], np.float32)
    interp.set_tensor(input_details[0]['index'], input_data)
    interp.invoke()

    actual_value = interp.get_tensor(output_details[0]['index'])
    expected_value = model.predict(input_data)

    self.assertLen(output_details[0]['shape_signature'], 3)
    self.assertAllClose(expected_value, actual_value, atol=1e-1)
    self.assertEqual(
        list(output_details[0]['shape_signature']),
        list(model.layers[-1].output_shape),
    )

  @test_util.run_v2_only
  def testKerasConv2DTransposedWithMismatchQuantizedAxes(self):
    # [FUNCTION DOC]
    # Scenario: transposed convolution whose weights are per-channel fake-quantized along a different axis
    # than the output. Saves a Keras model and converts with default optimizations. Expected: a converter
    # error whose message mentions mismatched quantized axes of input and output.
    class QuantConv2DTransposed(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: custom Keras layer used only in this test: transposed convolution with per-channel fake-quantized
      # kernel (24 output channels).

      def build(self, input_shape):
        # [FUNCTION DOC]
        # Creates the kernel weight with shape 3x3x(input channels)x24. Parameter input_shape is the input shape.
        self.kernel = self.add_weight('kernel', [3, 3, input_shape[-1], 24])

      def call(self, inputs):
        # [FUNCTION DOC]
        # Fake-quantizes the kernel per channel, transposes its last two axes and applies a transposed convolution
        # to the inputs (output with 24 channels, stride 1). Returns the result.
        filters = tf.quantization.fake_quant_with_min_max_vars_per_channel(
            self.kernel,
            -3.0 * tf.ones([24]),
            3.0 * tf.ones([24]),
            narrow_range=True,
        )
        filters = tf.transpose(filters, (0, 1, 3, 2))
        return tf.nn.conv2d_transpose(
            inputs, filters, [*inputs.shape[:-1], 24], 1
        )

    class _FakeQuantVarsLayer(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: Keras layer wrapper applying fake quantization (variable-range op) with range -3 to 3, narrow
      # range, needed because Keras 3 requires a layer wrapper.
      """A fake quantization layer with fake_quant_with_min_max_vars.

      Keras 3 requires wrapping the tf function inside Keras layer.
      """

      def call(self, x):
        # [FUNCTION DOC]
        # HELPER: returns x fake-quantized with min -3, max 3 and narrow range.
        return tf.quantization.fake_quant_with_min_max_vars(
            x, -3.0, 3.0, narrow_range=True)

    inp = tf.keras.Input(shape=(6, 8, 48), batch_size=1)
    x = _FakeQuantVarsLayer()(inp)
    x = QuantConv2DTransposed()(x)
    x = _FakeQuantVarsLayer()(x)

    model = tf.keras.Model(inp, x)

    saved_model_dir = os.path.join(
        self.get_temp_dir(), 'keras_conv2d_transpose'
    )
    model.save(saved_model_dir)
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.optimizations = [lite.Optimize.DEFAULT]

    with self.assertRaises(convert.ConverterError) as error:
      _ = converter.convert()
    self.assertIn(
        'mismatched quantized axes of input and output', str(error.exception)
    )

  def _createModelWithInputShape(self, shape):
    # [FUNCTION DOC]
    # HELPER: creates a saved model that doubles an input placeholder of a given shape.
    # Parameter shape: shape for the placeholder (nothing means unknown rank, empty list means scalar).
    # Returns the saved model directory path.
    """Create a simple SavedModel with a certain shape."""
    saved_model_dir = os.path.join(self.get_temp_dir(), 'input_shape_model')
    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        unknown_shape = tf.TensorShape(shape)
        in_tensor = tf.compat.v1.placeholder(
            shape=unknown_shape, dtype=tf.float32, name='input'
        )
        out_tensor = in_tensor + in_tensor
        inputs = {'input': in_tensor}
        outputs = {'output': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return saved_model_dir

  @test_util.run_v2_only
  def testUnknownInputShapeModel(self):
    # [FUNCTION DOC]
    # Converts a model with unknown-rank input. Asserts every tensor in the first subgraph has no rank and an
    # empty shape; resizes input to 3 and asserts the doubled values [2, 4, 6].
    """Test a SavedModel with an unknown input shape."""
    saved_model_dir = self._createModelWithInputShape(None)

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Validate that tensors with unknown shape have unknown rank.
    tflite_model_obj = flatbuffer_utils.convert_bytearray_to_object(
        tflite_model
    )
    for tensor in tflite_model_obj.subgraphs[0].tensors:
      self.assertEqual(False, tensor.hasRank)
      self.assertEqual([], tensor.shape.tolist())

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    interp.resize_tensor_input(input_details[0]['index'], [3], strict=False)
    interp.allocate_tensors()

    interp.set_tensor(input_details[0]['index'], input_data)
    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual([2.0, 4.0, 6.0], list(actual_value))

  @test_util.run_v2_only
  def testScalarInputShapeModel(self):
    # [FUNCTION DOC]
    # Converts a scalar-input model and asserts all tensors have a rank and an empty shape.
    """Test a SavedModel with a scalar input."""
    saved_model_dir = self._createModelWithInputShape([])

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Validate that scalar tensors have a rank = 0.
    tflite_model_obj = flatbuffer_utils.convert_bytearray_to_object(
        tflite_model
    )
    for tensor in tflite_model_obj.subgraphs[0].tensors:
      self.assertEqual(True, tensor.hasRank)
      self.assertEqual([], tensor.shape.tolist())

  @test_util.run_v2_only
  def testMatrixInputShapeModel(self):
    # [FUNCTION DOC]
    # Converts a 2x3-input model and asserts all tensors have rank and shape [2, 3].
    """Test a SavedModel with a matrix input."""
    saved_model_dir = self._createModelWithInputShape([2, 3])

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

    # Validate that matrix tensors have a rank = 2.
    tflite_model_obj = flatbuffer_utils.convert_bytearray_to_object(
        tflite_model
    )
    for tensor in tflite_model_obj.subgraphs[0].tensors:
      self.assertEqual(True, tensor.hasRank)
      self.assertEqual([2, 3], tensor.shape.tolist())

  @parameterized.named_parameters(
      ('_PerChannelQuant', False, False),
      ('_PerChannelMlirQuant', False, True),
      ('_PerTensorQuant', True, False),
      ('_PerTensorMlirQuant', True, True),
      ('_PerChannelDynamicRange', False, False, True),
      ('_PerTensorDynamicRange', True, False, True),
  )
  @test_util.run_v2_only
  def testDisablePerChannelQuantization(
      self,
      disable_per_channel=False,
      enable_mlir_quantizer=False,
      representative_dataset=True,
  ):
    # [FUNCTION DOC]
    # Parameterized (disable per-channel, MLIR quantizer flag, use representative data).
    # Saves a 38-filter convolution Keras model, converts with default optimizations and the private option,
    # finds the convolution tensor and asserts one scale if disabled, else 38.
    # Dynamic range quant requires total num elements of filters > 1024.
    k_num_filters = 38
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Conv2D(k_num_filters, (3, 3), activation='relu')]
    )
    model.build(input_shape=(1, 5, 5, 3))
    saved_model_dir = os.path.join(self.get_temp_dir(), 'conv_saved_model')
    save.save(model, saved_model_dir)
    k_conv_name = 'sequential/conv2d/Conv2D'
    quantized_converter = lite.TFLiteConverterV2.from_saved_model(
        saved_model_dir
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    if representative_dataset:

      def calib_gen():
        # [FUNCTION DOC]
        # HELPER: yields five single-input lists of random 1x5x5x3 float32 arrays in [-1, 1].
        for _ in range(5):
          yield [np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)]

      quantized_converter.representative_dataset = calib_gen
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS
    ]
    quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
    if disable_per_channel:
      quantized_converter._experimental_disable_per_channel = (
          disable_per_channel
      )
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    detail = next((
        d
        for d in interp.get_tensor_details()
        if d['name'].startswith(k_conv_name)
    ))
    quant_params = detail['quantization_parameters']
    expected_num_params = k_num_filters
    if disable_per_channel:
      expected_num_params = 1
    self.assertLen(quant_params['scales'], expected_num_params)
    if len(quant_params['zero_points']) != 1:
      self.assertLen(quant_params['zero_points'], expected_num_params)

  @parameterized.named_parameters(
      ('_INT8Quant_INT32Bias', False, False, dtypes.int32, True),
      ('_INT16Quant_INT64Bias', True, False, dtypes.int64, True),
      ('_INT8Quant_INT32Bias_Set', False, True, dtypes.int32, True),
      ('_INT8Quant_INT64Bias_Set', False, True, dtypes.int64, False),
      ('_INT16Quant_INT32Bias_Set', True, True, dtypes.int32, True),
      ('_INT16Quant_INT64Bias_Set', True, True, dtypes.int64, True),
      ('_INT16Quant_FLOAT32Bias_Set', True, True, dtypes.float32, False),
  )
  @test_util.run_v2_only
  def testBiasQuantization(
      self,
      is_int16_quantize,
      explicitly_set_bias,
      bias_type,
      is_valid_bias_type,
  ):
    # [FUNCTION DOC]
    # Parameterized over int16 mode, explicit bias type setting, bias dtype, and validity of that type.
    # Quantizes a 1024-unit dense model. When the bias type is invalid (int64 with int8, or float32) expects a
    # value error mentioning an expected bias type; otherwise asserts the bias tensor dtype equals the
    # requested type (int32 default for int8, int64 default for int16).
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            1024, input_shape=[1024], activation=None, bias_initializer='ones'
        )
    ])
    saved_model_dir = os.path.join(self.get_temp_dir(), 'dense_saved_model')
    save.save(model, saved_model_dir)
    k_dense_bias_name = 'sequential/dense/BiasAdd'
    quantized_converter = lite.TFLiteConverterV2.from_saved_model(
        saved_model_dir
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]

    if explicitly_set_bias:
      quantized_converter._experimental_full_integer_quantization_bias_type = (
          bias_type
      )

    if is_int16_quantize:
      quantized_converter.target_spec.supported_ops = [
          lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
      ]
    else:
      quantized_converter.target_spec.supported_ops = [
          lite.OpsSet.TFLITE_BUILTINS_INT8
      ]

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields five single-input lists of random normal float32 arrays of shape 1x1024.
      for _ in range(5):
        yield [np.random.randn(1, 1024).astype(np.float32)]

    quantized_converter.representative_dataset = calibration_gen

    if not is_valid_bias_type:
      with self.assertRaisesRegex(ValueError, 'Expected bias type to be'):
        quantized_converter.convert()
      return

    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    dense_bias = next((
        d
        for d in interp.get_tensor_details()
        if d['name'].startswith(k_dense_bias_name)
    ))
    self.assertEqual(bias_type, dense_bias['dtype'])

  @parameterized.named_parameters(
      ('_Int8PerChannelMlirDynamicRangeQuant', True, False, False),
      ('_Int8PerChannelTocoDynamicRangeQuant', False, False, False),
      ('_Int8PerTensorMlirDynamicRangeQuant', True, True, False),
      ('_Int8PerTensorTocoDynamicRangeQuant', False, True, False),
      ('_Float16DynamicRangeQuant', True, False, True),
  )
  @test_util.run_v2_only
  def testMlirDynamicRangeQuantization(
      self,
      enable_new_dynamic_range_quantizer,
      disable_per_channel,
      enable_float16_quant,
  ):
    # [FUNCTION DOC]
    # Parameterized over the new dynamic range quantizer, disable per-channel, and float16 modes.
    # Saves a 1024-filter convolution Keras model, converts with default optimizations. Finds the convolution
    # weight tensor (name with or without the suffix 1), asserts the number of scales (0 for float16, 1 if
    # per-tensor, else 1024), float32 input/output, and weight dtype float16 or int8.
    num_filters = 1024
    conv_name = 'sequential/conv2d/Conv2D'
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            num_filters, (3, 3), activation='relu', bias_initializer='ones'
        )
    ])
    model.build(input_shape=(1, 32, 32, 3))
    saved_model_dir = self.create_tempdir()
    save.save(model, saved_model_dir.full_path)

    converter = lite.TFLiteConverterV2.from_saved_model(
        saved_model_dir.full_path
    )
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.experimental_new_dynamic_range_quantizer = (
        enable_new_dynamic_range_quantizer
    )
    converter._experimental_disable_per_channel = disable_per_channel
    if enable_float16_quant:
      converter.target_spec.supported_types = [tf.float16]
    quantized_tflite_model = converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # Do not apply delegates as XNNPack converts per tensor to per channel.
    interp = interpreter.Interpreter(
        model_content=quantized_tflite_model,
        experimental_op_resolver_type=interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    quantized_weight = None
    quantized_weight_with_one_postfix = None
    quantized_weight_without_one_postfix = None
    for d in interp.get_tensor_details():
      if d['name'] == conv_name + '1':
        quantized_weight = d
        quantized_weight_with_one_postfix = d
        break
    for d in interp.get_tensor_details():
      if d['name'].startswith(conv_name):
        if quantized_weight is None:
          quantized_weight = d
        quantized_weight_without_one_postfix = d
        break

    self.assertIsNotNone(quantized_weight)
    quant_params = quantized_weight['quantization_parameters']

    if enable_float16_quant:
      expected_num_params = 0
    else:
      expected_num_params = 1 if disable_per_channel else num_filters
    self.assertLen(quant_params['scales'], expected_num_params)
    if len(quant_params['zero_points']) != 1:
      self.assertLen(quant_params['zero_points'], expected_num_params)

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    if enable_float16_quant:
      self.assertTrue(
          (
              quantized_weight_with_one_postfix is not None
              and np.float16 == quantized_weight_with_one_postfix['dtype']
          )
          or (
              quantized_weight_without_one_postfix is not None
              and np.float16 == quantized_weight_without_one_postfix['dtype']
          )
      )
    else:
      self.assertEqual(np.int8, quantized_weight['dtype'])

  @parameterized.named_parameters(
      ('_NONE', 'NONE'),
      ('_STATIC', 'STATIC'),
      ('_DYNAMIC', 'DYNAMIC'),
      ('_UNKNOWN', 'UNKNOWN'),
  )
  def testQDQConversionMode(self, mode):
    # [FUNCTION DOC]
    # Parameterized over QDQ conversion mode strings NONE, STATIC, DYNAMIC, UNKNOWN. Sets the private mode.
    # UNKNOWN must raise a converter error mentioning an unknown mode; others must produce a model.
    # (No run-v2 decorator, unlike its neighbours.)
    num_filters = 1024
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu')]
    )
    model.build(input_shape=(1, 32, 32, 3))
    saved_model_dir = self.create_tempdir()
    save.save(model, saved_model_dir.full_path)
    converter = lite.TFLiteConverterV2.from_saved_model(
        saved_model_dir.full_path
    )
    converter._experimental_qdq_conversion_mode = mode

    if mode == 'UNKNOWN':
      with self.assertRaises(convert.ConverterError) as error:
        converter.convert()
      self.assertIn('Unknown QDQ conversion mode:', str(error.exception))
    else:
      model = converter.convert()
      self.assertIsNotNone(model)


class FromKerasModelTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case covering the version 2 converter when built from Keras models. Scenarios: variable
  # quantization, float16, sequential and multi-input/output models, debug info, the fallback path,
  # signatures, dynamic range quantization, low-bit quantization-aware training, transposed convolution and
  # per-channel disabling for dense layers.

  @parameterized.named_parameters(
      ('EnableMlirVariableQuantizationNumState1', True, 1),
      ('DisablMlirVariableQuantizationNumState1', False, 1),
      ('EnableMlirVariableQuantizationNumState2', True, 2),
      ('DisablMlirVariableQuantizationNumState2', False, 2),
  )
  @test_util.run_v2_only
  def testVariableQuantization(self, variable_quantization, number_of_states):
    # [FUNCTION DOC]
    # Parameterized over variable quantization flag and number of states (1, 2). Converts a read/assign Keras
    # model with int8 builtins, int8 I/O and calibration data. Asserts the read-variable tensor has one scale
    # and zero point when enabled, zero otherwise.
    k_readvariable_name = 'model/read_assign/concat/ReadVariableOp'
    model, calibration_gen = self._createReadAssignModel(number_of_states)

    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_gen
    converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    converter._experimental_variable_quantization = variable_quantization

    quantized_tflite_model = converter.convert()

    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()

    detail = next((
        d
        for d in interp.get_tensor_details()
        if d['name'].startswith(k_readvariable_name)
    ))
    quant_params = detail['quantization_parameters']
    if variable_quantization:
      expected_num_params = 1
    else:
      # This number is not a spec. Since It's the unintended number, it can be
      # changed later by the other features of the quantizer.
      expected_num_params = 0

    self.assertLen(quant_params['scales'], expected_num_params)
    self.assertLen(quant_params['zero_points'], expected_num_params)

  @parameterized.named_parameters(
      ('EnableMlirVariableQuantizationNumState1', True, 1),
      ('DisablMlirVariableQuantizationNumState1', False, 1),
      ('EnableMlirVariableQuantizationNumState2', True, 2),
      ('DisablMlirVariableQuantizationNumState2', False, 2),
  )
  def testVariableQuantizationInFloat16(
      self, variable_quantization, number_of_states
  ):
    # [FUNCTION DOC]
    # Parameterized as above. With float16 support and the variable quantization flag on, expects a value error
    # saying the option is only supported for full integer quantization; with it off, conversion succeeds.
    model, _ = self._createReadAssignModel(number_of_states)
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter._experimental_variable_quantization = variable_quantization

    if variable_quantization:
      with self.assertRaises(ValueError) as error:
        converter.convert()
      self.assertIn(
          '`_experimental_variable_quantization` is only supported for full',
          str(error.exception),
      )
    else:
      converter.convert()

  @test_util.run_v2_only
  def testSequentialModel(self):
    # [FUNCTION DOC]
    # Trains a dropout+dense Keras model, converts, asserts metadata model type Keras and equal results to the
    # Keras prediction.
    """Test a simple sequential tf.Keras model."""
    input_data = tf.constant(1.0, shape=[1, 1])

    # Create a simple Keras model.
    x = np.array([[1.0], [2.0]])
    y = np.array([[2.0], [4.0]])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1, input_shape=[1]),
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    tflite_model = converter.convert()
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(
        metadata.environment.modelType, metadata_fb.ModelType.KERAS_MODEL
    )

    # Check values from converted model.
    expected_value = model.predict(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value, actual_value)

  @test_util.run_v2_only
  def testSequentialMultiInputOutputModel(self):
    # [FUNCTION DOC]
    # Trains a two-input, two-output Keras model with shared dense, concatenation and two softmax heads;
    # converts and asserts every output equals Keras within 1e-5.
    """Test a tf.Keras model with multiple inputs and outputs."""
    left_input_data = tf.constant(1.0, shape=[1, 3])
    right_input_data = tf.constant(1.0, shape=[1, 3])

    # Create a simple Keras model.
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))
    output_c_np = np.random.random((10, 3))
    output_d_np = np.random.random((10, 2))

    input_a = tf.keras.layers.Input(shape=(3,), name='input_a')
    input_b = tf.keras.layers.Input(shape=(3,), name='input_b')

    dense = tf.keras.layers.Dense(8, name='dense_1')
    interm_a = dense(input_a)
    interm_b = dense(input_b)
    merged = tf.keras.layers.concatenate([interm_a, interm_b], name='merge')

    output_c = tf.keras.layers.Dense(3, activation='softmax', name='dense_2')(
        merged
    )
    output_d = tf.keras.layers.Dense(2, activation='softmax', name='dense_3')(
        merged
    )

    model = tf.keras.models.Model(
        inputs=[input_a, input_b], outputs=[output_c, output_d]
    )
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit([input_a_np, input_b_np], [output_c_np, output_d_np], epochs=1)

    # Convert model and ensure model is not None.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    tflite_model = converter.convert()

    # Check values from converted model.
    input_data = [left_input_data, right_input_data]
    expected_value = model.predict(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, input_data)
    for tf_result, tflite_result in zip(expected_value, actual_value):
      self.assertAllClose(tf_result, tflite_result, atol=1e-05)

  @test_util.run_v2_only
  def testGraphDebugInfo(self):
    # [FUNCTION DOC]
    # Trains a tiny Keras model, converts, asserts valid debug info.
    """Test a tf.Keras model has debug info captured."""
    # Create a simple Keras model.
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(units=1, input_shape=[1])]
    )
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.convert()
    self._assertValidDebugInfo(converter._debug_info)

  @test_util.run_v2_only
  def testKerasFallbackPath(self):
    # [FUNCTION DOC]
    # Scenario: Keras model that fails saved-model export (weight created with no name), so the converter has
    # to use its fallback path. Trains briefly, converts and asserts a non-empty model.
    """Test keras model which failed when exporting to the saved model."""
    input_data = tf.constant(
        np.array(np.random.random_sample((20)), dtype=np.float32)
    )

    class Model(tf.keras.Model):
      # [CLASS DOC]
      # HELPER: Keras model used only by the fallback test; holds an unnamed 20x1 weight and adds it to the input.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor without arguments; creates the weight variable with name nothing, which is the cause of
        # the saved-model export failure.
        super().__init__()
        # A None name will cause a failure in exporting to a saved model.
        self.shared_weights = self.add_weight(
            name=None,
            shape=(20, 1),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                mean=0.0, stddev=300 ** (-0.5)
            ),
        )

      def call(self, x):
        # [FUNCTION DOC]
        # Returns the sum of the weight and input x.
        return tf.add(self.shared_weights, x)

    # Building the model.
    model = Model()
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(input_data, input_data, epochs=1)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)

  @test_util.run_v2_only
  def testSignatureDefs(self):
    # [FUNCTION DOC]
    # Converts a Conv2D plus dense Keras model and asserts a single serving_default signature with input
    # tensor_input and output output_tensor, plus value equality with Keras.
    """Test converting SignatureDef is correct and uses SignatureDef API."""
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32,
            kernel_size=3,
            padding='same',
            activation='relu',
            input_shape=(32, 32, 3),
            name='tensor',
        ),
        tf.keras.layers.Dense(10, name='output_tensor'),
    ])

    converter = lite.TFLiteConverterV2.from_keras_model(keras_model)
    tflite_model = converter.convert()

    # Check values from converted model.
    input_data = tf.constant(
        np.random.uniform(-1, 1, size=(1, 32, 32, 3)).astype(np.float32)
    )
    expected_value = keras_model(input_data)
    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()
    results = self._evaluateTFLiteModelUsingSignatureDef(
        tflite_model, 'serving_default', {'tensor_input': input_data}
    )
    self.assertEqual(list(results.keys()), ['output_tensor'])
    self.assertAllClose(expected_value.numpy(), results['output_tensor'])

    # Verify the SignatureDef structure returned is as expected.
    self.assertLen(signature_defs, 1)
    self.assertEqual(list(signature_defs.keys()), ['serving_default'])
    self.assertLen(signature_defs.values(), 1)
    self.assertEqual(
        list(signature_defs['serving_default'].keys()), ['inputs', 'outputs']
    )
    self.assertCountEqual(
        signature_defs['serving_default']['inputs'], ['tensor_input']
    )
    self.assertEqual(
        list(signature_defs['serving_default']['outputs']), ['output_tensor']
    )

  @parameterized.named_parameters(
      ('_PerChannelMlirDynamicRangeQuant', True, False, False),
      ('_PerChannelTocoDynamicRangeQuant', False, False, False),
      ('_PerTensorMlirDynamicRangeQuant', True, True, False),
      ('_PerTensorTocoDynamicRangeQuant', False, True, False),
      ('_Float16DynamicRangeQuant', True, False, True),
  )
  @test_util.run_v2_only
  def testMlirDynamicRangeQuantization(
      self,
      enable_new_dynamic_range_quantizer,
      disable_per_channel,
      enable_float16_quant,
  ):
    # [FUNCTION DOC]
    # Same checks as the saved-model version of this test, but starting from a Keras sequential model
    # (input layer, 1024-filter convolution). Parameterized over new quantizer, per-channel and float16.
    num_filters = 1024
    conv_name = 'sequential/conv2d/Conv2D'
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(
            num_filters, (3, 3), activation='relu', bias_initializer='ones'
        ),
    ])
    model.build()

    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.experimental_new_dynamic_range_quantizer = (
        enable_new_dynamic_range_quantizer
    )
    converter._experimental_disable_per_channel = disable_per_channel
    if enable_float16_quant:
      converter.target_spec.supported_types = [tf.float16]
    quantized_tflite_model = converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # Do not apply delegates as XNNPack converts per tensor to per channel.
    interp = interpreter.Interpreter(
        model_content=quantized_tflite_model,
        experimental_op_resolver_type=interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    quantized_weight = None
    quantized_weight_with_one_postfix = None
    quantized_weight_without_one_postfix = None
    for d in interp.get_tensor_details():
      if d['name'] == conv_name + '1':
        quantized_weight = d
        quantized_weight_with_one_postfix = d
        break
    for d in interp.get_tensor_details():
      if d['name'].startswith(conv_name):
        if quantized_weight is None:
          quantized_weight = d
        quantized_weight_without_one_postfix = d
        break

    self.assertIsNotNone(quantized_weight)
    quant_params = quantized_weight['quantization_parameters']

    if enable_float16_quant:
      expected_num_params = 0
    else:
      expected_num_params = 1 if disable_per_channel else num_filters
    self.assertLen(quant_params['scales'], expected_num_params)
    if len(quant_params['zero_points']) != 1:
      self.assertLen(quant_params['zero_points'], expected_num_params)

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertEqual(np.float32, output_details[0]['dtype'])
    if enable_float16_quant:
      self.assertTrue(
          (
              quantized_weight_with_one_postfix is not None
              and np.float16 == quantized_weight_with_one_postfix['dtype']
          )
          or (
              quantized_weight_without_one_postfix is not None
              and np.float16 == quantized_weight_without_one_postfix['dtype']
          )
      )
    else:
      self.assertEqual(np.int8, quantized_weight['dtype'])

  # TODO(b/242081598): The num_bits parameter should be restored to (2, 4, 6)
  # once a 4-bit conv kernel is available.
  @parameterized.named_parameters([
      (
          '{}BitWeightOnly={}LowBit={}'.format(num_bits, weight_only, low_bit),
          num_bits,
          weight_only,
          low_bit,
      )
      for num_bits, weight_only, low_bit in itertools.product(
          (5, 7, 6), (True, False), (True, False)
      )
  ])
  @test_util.run_v2_only
  def testQATLowBitKerasModel(self, num_bits, weight_only, low_bit):
    # [FUNCTION DOC]
    # Parameterized over bit widths (5, 7, 6), weight-only mode and low-bit option. Builds a low-bit QAT Keras
    # model, converts with default optimizations and the private low-bit QAT option when requested. Asserts
    # result equality with Keras, int8 weights stay inside the bit range when low-bit, scales are 1 for low-bit
    # weights and 6/255 for activations, and counts of int8 weights/activations.
    bit_max = (1 << (num_bits - 1)) - 1
    bit_min = -bit_max
    tf_input_shape = (5, 5, 3)
    tflite_input_shape = (1,) + tf_input_shape
    model, input_name, output_name = self._createV2QATLowBitKerasModel(
        tf_input_shape, weight_only, num_bits, bit_min, bit_max
    )
    input_data = np.linspace(0, 6, np.prod(tflite_input_shape)).reshape(
        tflite_input_shape
    )
    tf_result = model(input_data)

    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.optimizations = [lite.Optimize.DEFAULT]
    if low_bit:
      converter._experimental_low_bit_qat = True
    tflite_model = converter.convert()
    result = self._evaluateTFLiteModelUsingSignatureDef(
        tflite_model,
        'serving_default',
        {input_name: input_data.astype(np.float32)},
    )[output_name]
    self.assertAllClose(
        [np.linalg.norm(result - tf_result.numpy().astype(np.float32))], [0.0]
    )
    interp = interpreter.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    num_8bit_activations = 0
    num_8bit_weights = 0
    kernel_name = (
        'model/conv_wrapper/Conv2D;model/conv_wrapper/'
        'FakeQuantWithMinMaxVarsPerChannel'
    )

    for detail in interp.get_tensor_details():
      if (
          detail['dtype'] == np.int8
          and detail['name']
          and detail['name'] == kernel_name
      ):
        num_8bit_weights += 1
        weights = interp.get_tensor(detail['index'])
        if low_bit:
          self.assertFalse(
              (bit_min > weights).any() or (weights > bit_max).any()
          )
        else:
          self.assertTrue(
              (bit_min > weights).any() or (weights > bit_max).any()
          )
        self.assertIn('scales', detail['quantization_parameters'])
        if low_bit and detail['quantization_parameters']['scales']:
          self.assertAllClose(
              detail['quantization_parameters']['scales'], [1.0]
          )
      elif detail['dtype'] == np.int8 and detail['name']:
        self.assertFalse(weight_only)
        self.assertIn('scales', detail['quantization_parameters'])
        if detail['quantization_parameters']['scales']:
          self.assertAllClose(
              detail['quantization_parameters']['scales'], [6 / 255]
          )
        num_8bit_activations += 1

    self.assertEqual(num_8bit_weights, 0 if weight_only and not low_bit else 1)
    # 3 activations with full integer: conv_input, conv_output, reshape_output
    self.assertEqual(num_8bit_activations, 0 if weight_only else 3)

  @test_util.run_v2_only
  def testKerasConv2DTransposedWithBiasAndActivation(self):
    # [FUNCTION DOC]
    # Builds a fake-quantized transposed convolution with bias and relu, converts with default optimizations,
    # analyzes the model, compares the output with Keras (norm of the difference near zero) and asserts only two
    # float32 tensors remain (input and output).
    class QuantConv2DTransposedWithBiasAndActivation(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: custom Keras layer for this test: fake-quantized kernel transposed convolution with bias, relu
      # and output fake quantization.

      def build(self, input_shape):
        # [FUNCTION DOC]
        # Creates a 3x3x(input channels)x3 kernel and a bias of 3 values. Parameter input_shape is the input shape.
        self.kernel = self.add_weight('kernel', (3, 3, input_shape[-1], 3))
        self.bias = self.add_weight('bias', (3,))

      def call(self, inputs):
        # [FUNCTION DOC]
        # Applies fake-quantized kernel transposed convolution, bias add, relu, and fake-quantizes the result.
        filters = tf.quantization.fake_quant_with_min_max_vars(
            self.kernel, -3.0, 3.0, narrow_range=True
        )
        filters = tf.transpose(filters, (0, 1, 3, 2))
        result = tf.nn.conv2d_transpose(
            inputs, filters, [*inputs.shape[:-1], 3], 1
        )
        result = tf.nn.bias_add(result, self.bias)
        result = tf.nn.relu(result)

        return tf.quantization.fake_quant_with_min_max_vars(
            result, -3.0, 3.0, narrow_range=True
        )

    class _FakeQuantVarsLayer(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: Keras layer wrapper applying narrow-range fake quantization with range -3 to 3.
      """A fake quantization layer with fake_quant_with_min_max_vars.

      Keras 3 requires wrapping the tf function inside Keras layer.
      """

      def call(self, x):
        # [FUNCTION DOC]
        # HELPER: returns x fake-quantized with min -3, max 3, narrow range.
        return tf.quantization.fake_quant_with_min_max_vars(
            x, -3.0, 3.0, narrow_range=True)

    inp = tf.keras.Input(shape=(6, 8, 6), batch_size=1)
    x = _FakeQuantVarsLayer()(inp)
    x = QuantConv2DTransposedWithBiasAndActivation()(x)

    model = tf.keras.Model(inp, x)

    tf_input_shape = (1, 6, 8, 6)
    input_data = np.linspace(0, 6, np.prod(tf_input_shape)).reshape(
        tf_input_shape
    )
    tf_result = model(input_data)

    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.optimizations = [lite.Optimize.DEFAULT]

    converted_model = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=converted_model)

    interp = interpreter.Interpreter(model_content=converted_model)
    interp.allocate_tensors()

    input_index = interp.get_input_details()[0]['index']
    output_index = interp.get_output_details()[0]['index']

    interp.set_tensor(input_index, input_data.astype(np.float32))
    interp.invoke()
    tflite_result = interp.tensor(output_index)()

    self.assertAllClose(
        [np.linalg.norm(tflite_result - tf_result.numpy().astype(np.float32))],
        [0.0],
    )

    num_float32_tensor = 0
    for detail in interp.get_tensor_details():
      if detail['dtype'] == np.float32:
        num_float32_tensor += 1

    # There should be only 2 float tensors, input and output.
    self.assertEqual(num_float32_tensor, 2)

  @parameterized.named_parameters(
      ('_PerChannelQuant', False, False),
      ('_PerChannelMlirQuant', False, True),
      ('_PerTensorQuant', True, False),
      ('_PerTensorMlirQuant', True, True),
      ('_PerChannelDynamicRange', False, True, True),
      ('_PerTensorDynamicRange', True, True, True),
  )
  @test_util.run_v2_only
  def testDisablePerChannelQuantizationForDenseLayers(
      self,
      disable_per_channel_for_dense=False,
      enable_mlir_quantizer=False,
      representative_dataset=False,
  ):
    # [FUNCTION DOC]
    # Parameterized as the saved model variants. Converts a 64-unit dense Keras model with the private option
    # for disabling per-channel quantization on dense layers, and asserts one scale if disabled, else 64.
    k_dense_name = 'MatMul'
    # Dynamic range quant requires total num elements of filters > 1024.
    k_num_filters = 64
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(16,)),
        tf.keras.layers.Dense(k_num_filters, activation='relu'),
    ])
    model.build()

    quantized_converter = lite.TFLiteConverterV2.from_keras_model(model)
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    if representative_dataset:

      def calibration_gen():
        # [FUNCTION DOC]
        # HELPER: yields five single-input lists of random 1x16 float32 arrays in [-1, 1].
        for _ in range(5):
          yield [np.random.uniform(-1, 1, size=(1, 16)).astype(np.float32)]

      quantized_converter.representative_dataset = calibration_gen
    quantized_converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS
    ]
    quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
    if disable_per_channel_for_dense:
      quantized_converter._experimental_disable_per_channel_quantization_for_dense_layers = (
          disable_per_channel_for_dense
      )
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # Do not apply delegates as XNNPack converts per tensor to per channel.
    interp = interpreter.Interpreter(
        model_content=quantized_tflite_model,
        experimental_op_resolver_type=interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    detail = next(
        (d for d in interp.get_tensor_details() if k_dense_name in d['name'])
    )
    quant_params = detail['quantization_parameters']
    expected_num_params = 1 if disable_per_channel_for_dense else k_num_filters
    self.assertLen(quant_params['scales'], expected_num_params)
    if len(quant_params['zero_points']) != 1:
      self.assertLen(quant_params['zero_points'], expected_num_params)


class FromJaxModelTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for the experimental conversion from Jax functions. Every test returns immediately when Jax
  # could not be imported (module flag). Scenarios: invalid argument combinations, single and multiple inputs,
  # constant parameters and a while loop.

  @test_util.run_v2_only
  def testInvalidInputsModel(self):
    # [FUNCTION DOC]
    # Skips if Jax is unavailable. Asserts value errors for: no serving function; missing or empty inputs;
    # inputs not wrapped in a list; partial inputs; serving function count not matching input mapping count;
    # multiple serving functions (only a single one is supported).
    if DISABLE_JAX_TEST:
      return

    def simple_model(input1, input2):
      # [FUNCTION DOC]
      # HELPER: sum of sine of input1 and cosine of input2.
      return jnp.sin(input1) + jnp.cos(input2)

    input_tensor = jnp.zeros([10, 10])
    # Invalid case: not specify serving_func
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        None, [{'input1': input_tensor}]
    )
    with self.assertRaisesRegex(ValueError, 'No serving func is specified.'):
      converter.convert()

    # Invalid case: not specify input
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        [simple_model], None
    )
    with self.assertRaisesRegex(ValueError, 'Input tensors are not specified.'):
      converter.convert()

    converter = lite.TFLiteConverterV2.experimental_from_jax([simple_model], [])
    with self.assertRaisesRegex(ValueError, 'Input tensors are not specified.'):
      converter.convert()

    # Invalid case: not wrap input_tensor in a list.
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        [simple_model], input_tensor
    )
    with self.assertRaisesRegex(
        ValueError,
        'The truth value of an array with more than one element is ambiguous.',
    ):
      converter.convert()

    # Invalid case: only partial inputs are provided.
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        [simple_model], [[('input1', input_tensor)]]
    )
    with self.assertRaisesRegex(
        ValueError, 'Failed to convert the given Jax function to hlo.'
    ):
      converter.convert()

    # Invalid case: serving functions length does not match input mapping.
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        [simple_model, simple_model],
        [[
            ('input1', input_tensor),
            ('input2', input_tensor),
        ]],
    )
    with self.assertRaisesRegex(
        ValueError,
        'Input tensor mapping len 1 does not match serving func len 2.',
    ):
      converter.convert()

    # Invalid case: multiple serving function is provided.
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        [simple_model, simple_model],
        [
            [
                ('input1', input_tensor),
                ('input2', input_tensor),
            ],
            [
                ('input1', input_tensor),
                ('input2', input_tensor),
            ],
        ],
    )
    with self.assertRaisesRegex(
        ValueError, 'Currently only support single serving function.'
    ):
      converter.convert()

  @test_util.run_v2_only
  def testSingleInputModel(self):
    # [FUNCTION DOC]
    # Converts a one-input Jax sine function with a 10x10 zeros example; asserts metadata model type Jax and
    # numeric closeness (1e-5) to the Jax result.
    if DISABLE_JAX_TEST:
      return

    def single_input(input_tensor):
      # [FUNCTION DOC]
      # HELPER: returns the sine of the input tensor.
      return jnp.sin(input_tensor)

    # Convert model.
    input_tensor = jnp.zeros([10, 10])
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        [single_input], [[('input_tensor', input_tensor)]]
    )
    tflite_model = converter.convert()
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(metadata.environment.modelType, metadata_fb.ModelType.JAX)

    # Check values from converted_model
    input_data = np.random.random_sample((10, 10))
    tf_input_data = tf.constant(input_data, dtype=np.float32)
    actual_value = self._evaluateTFLiteModel(tflite_model, [tf_input_data])[0]
    expected_value = single_input(input_data)
    self.assertAllClose(expected_value, actual_value, atol=1e-05)

  @test_util.run_v2_only
  def testMultipleInputsModel(self):
    # [FUNCTION DOC]
    # Converts a two-input add function (10x10 and 10x1, broadcast) from Jax and compares with the Jax result.
    if DISABLE_JAX_TEST:
      return

    def multiple_inputs(input1, input2):
      # [FUNCTION DOC]
      # HELPER: returns input1 plus input2.
      return input1 + input2

    # Convert model.
    input1 = jnp.zeros([10, 10])
    input2 = jnp.zeros([10, 1])
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        [multiple_inputs], [[('input1', input1), ('input2', input2)]]
    )
    tflite_model = converter.convert()

    # Check values from converted_model
    input1_data = np.random.random_sample((10, 10))
    tf_input1_data = tf.constant(input1_data, dtype=np.float32)
    input2_data = np.random.random_sample((10, 1))
    tf_input2_data = tf.constant(input2_data, dtype=np.float32)
    actual_value = self._evaluateTFLiteModel(
        tflite_model, [tf_input1_data, tf_input2_data]
    )[0]
    expected_value = multiple_inputs(input1_data, input2_data)
    self.assertAllClose(expected_value, actual_value, atol=1e-05)

  @test_util.run_v2_only
  def testInputSignaturesModel(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test body is identical to the multiple-inputs test, so nothing specific to input
    # signatures is checked.
    # Converts a two-input add function from Jax and compares with the Jax result.
    if DISABLE_JAX_TEST:
      return

    def multiple_inputs(input1, input2):
      # [FUNCTION DOC]
      # HELPER: returns input1 plus input2.
      return input1 + input2

    # Convert model.
    input1 = jnp.zeros([10, 10])
    input2 = jnp.zeros([10, 1])
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        [multiple_inputs], [[('input1', input1), ('input2', input2)]]
    )
    tflite_model = converter.convert()

    # Check values from converted_model
    input1_data = np.random.random_sample((10, 10))
    tf_input1_data = tf.constant(input1_data, dtype=np.float32)
    input2_data = np.random.random_sample((10, 1))
    tf_input2_data = tf.constant(input2_data, dtype=np.float32)
    actual_value = self._evaluateTFLiteModel(
        tflite_model, [tf_input1_data, tf_input2_data]
    )[0]
    expected_value = multiple_inputs(input1_data, input2_data)
    self.assertAllClose(expected_value, actual_value, atol=1e-05)

  @test_util.run_v2_only
  def testModelWithParams(self):
    # [FUNCTION DOC]
    # Converts a Jax matmul whose weights are bound by partial application; checks result closeness to Jax.
    if DISABLE_JAX_TEST:
      return

    def model(inputs, weights):
      # [FUNCTION DOC]
      # HELPER: returns the matrix product of weights and inputs.
      return jnp.matmul(weights, inputs)

    weights = np.random.random_sample((10, 10))
    serving_func = functools.partial(model, weights=weights)

    # Convert model
    input_tensor = jnp.zeros([10, 10])
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        [serving_func], [[('inputs', input_tensor)]]
    )
    tflite_model = converter.convert()

    # Check values from converted_model
    input_data = np.random.random_sample((10, 10))
    tf_input_data = tf.constant(input_data, dtype=np.float32)
    actual_value = self._evaluateTFLiteModel(tflite_model, [tf_input_data])[0]
    expected_value = serving_func(input_data)
    self.assertAllClose(expected_value, actual_value, atol=1e-05)

  @test_util.run_v2_only
  def testWhileLoop(self):
    # [FUNCTION DOC]
    # Converts a Jax while loop (add 2 until sum reaches 100) and returns the first row; checks closeness.
    if DISABLE_JAX_TEST:
      return

    def condition(x):
      # [FUNCTION DOC]
      # HELPER: loop condition; true while the sum of x is below 100.
      return jnp.sum(x, keepdims=False) < 100

    def body(x):
      # [FUNCTION DOC]
      # HELPER: loop body; adds 2.0 to x.
      return jnp.add(x, 2.0)

    def model(x):
      # [FUNCTION DOC]
      # HELPER: runs the Jax while loop and returns the first row of the result.
      result = jax.lax.while_loop(condition, body, x)
      return result[0]

    # Convert model.
    input_tensor = jnp.zeros([3, 3])
    converter = lite.TFLiteConverterV2.experimental_from_jax(
        [model], [[('x', input_tensor)]]
    )
    tflite_model = converter.convert()

    # Check values from converted_model
    input_data = np.random.random_sample((3, 3))
    tf_input_data = tf.constant(input_data, dtype=np.float32)
    actual_value = self._evaluateTFLiteModel(tflite_model, [tf_input_data])[0]
    expected_value = model(input_data)
    self.assertAllClose(expected_value, actual_value, atol=1e-05)


class ControlFlowTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for conversion of conditionals, while loops and RNN models, including quantization with
  # control flow and error on control-flow V1 ops.

  @test_util.run_v2_only
  def testCond(self):
    # [FUNCTION DOC]
    # Converts a traced conditional (matmul with a variable versus add) with a boolean input; asserts results
    # match TensorFlow for the true branch.
    input_data = {
        'x': tf.constant([1.0, 2.0], shape=[1, 2]),
        'b': tf.constant(True),
    }

    weights = tf.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)

    def true_fn(x):
      # [FUNCTION DOC]
      # HELPER: branch computing matrix product of x with the weights.
      return tf.matmul(x, weights)

    def false_fn(x):
      # [FUNCTION DOC]
      # HELPER: branch computing x plus the weights.
      return tf.add(x, weights)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, 2], dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        ]
    )
    def model(x, b):
      # [FUNCTION DOC]
      # HELPER: traced function with a 1x2 float input and boolean scalar selecting true_fn or false_fn.
      return tf.cond(
          b, true_fn=lambda: true_fn(x), false_fn=lambda: false_fn(x)
      )

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(**input_data)
    actual_value = self._evaluateTFLiteModel(
        tflite_model, [input_data['x'], input_data['b']]
    )[0]
    self.assertAllClose(expected_value, actual_value)

  @test_util.run_v2_only
  def testCondWithFullIntegerQuantization(self):
    # [FUNCTION DOC]
    # Converts the conditional model with default optimizations and a representative dataset covering both
    # branches; asserts only that a model is produced.
    weights = tf.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)

    def true_fn(x):
      # [FUNCTION DOC]
      # HELPER: branch computing matrix product of x with the weights.
      return tf.matmul(x, weights)

    def false_fn(x):
      # [FUNCTION DOC]
      # HELPER: branch computing x plus the weights.
      return tf.add(x, weights)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, 2], dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.bool),
        ]
    )
    def model(x, b):
      # [FUNCTION DOC]
      # HELPER: traced conditional on a boolean selecting true_fn or false_fn.
      return tf.cond(
          b, true_fn=lambda: true_fn(x), false_fn=lambda: false_fn(x)
      )

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields five samples with true and five with false condition, random 1x2 inputs.
      for _ in range(5):
        yield [
            np.random.uniform(-1, 1, size=(1, 2)).astype(np.float32),
            tf.constant(True),
        ]
      for _ in range(5):
        yield [
            np.random.uniform(-1, 1, size=(1, 2)).astype(np.float32),
            tf.constant(False),
        ]

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_gen
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

  @test_util.run_v2_only
  def testConverterErrorOnControlFlowV1Ops(self):
    # [FUNCTION DOC]
    # Loads a test data saved model using control flow V1 and asserts conversion raises a converter error with
    # the message about failure to functionalize control flow V1 ops.
    filename = resource_loader.get_path_to_datafile(
        'testdata/control_flow_v1_saved_model'
    )
    converter = lite.TFLiteConverterV2.from_saved_model(filename)
    with self.assertRaises(convert.ConverterError) as error:
      converter.convert()
    self.assertIn(
        'Failed to functionalize Control Flow V1 ops. Consider using Control '
        'Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/'
        'tf/compat/v1/enable_control_flow_v2.',
        str(error.exception),
    )

  @test_util.run_v2_only
  def testStaticRnn(self):
    # [FUNCTION DOC]
    # Converts a traced static RNN over an LSTM cell on a 3x10 input split in three steps; compares each
    # output with TensorFlow.
    input_data = tf.constant(
        np.array(np.random.random_sample((3, 10)), dtype=np.float32)
    )

    cell = tf.keras.layers.LSTMCell(10)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[3, 10], dtype=tf.float32)]
    )
    def model(x):
      # [FUNCTION DOC]
      # HELPER: splits the input into three and applies the static RNN with a sequence length of one.
      seq = tf.split(x, 3, 0)
      return rnn.static_rnn(cell, seq, dtype=tf.float32, sequence_length=[1])

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)[0]
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    for expected, actual in zip(expected_value, actual_value):
      self.assertAllClose(expected, actual)

  @test_util.run_v2_only
  def testWhileLoop(self):
    # [FUNCTION DOC]
    # Converts a traced while loop adding variable weights until the sum is at least 100; compares result.
    input_data = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2])

    weights = tf.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)

    def condition(x):
      # [FUNCTION DOC]
      # HELPER: true while the sum of x is below 100.
      return tf.reduce_sum(x) < 100

    def body(x):
      # [FUNCTION DOC]
      # HELPER: adds the weights to x.
      return tf.add(x, weights)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[2, 2], dtype=tf.float32)]
    )
    def model(x):
      # [FUNCTION DOC]
      # HELPER: runs the while loop starting from the input.
      return tf.while_loop(condition, body, [x])

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)[0]
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
    self.assertAllClose(expected_value, actual_value)

  @test_util.run_v2_only
  def testDynamicRnn(self):
    # [FUNCTION DOC]
    # Converts a traced Keras RNN with an LSTM cell on a 3x10x10 input; asserts one output and closeness to
    # TensorFlow.
    input_data = tf.constant(
        np.array(np.random.random_sample((3, 10, 10)), dtype=np.float32)
    )

    cell = tf.keras.layers.LSTMCell(10)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[3, 10, 10], dtype=tf.float32)]
    )
    def model(x):
      # [FUNCTION DOC]
      # HELPER: applies a Keras RNN layer returning sequences.
      rnn_layer = tf.keras.layers.RNN([cell], return_sequences=True)
      return rnn_layer(x)

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)
    lite_outputs = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertLen(lite_outputs, 1)
    actual_value = lite_outputs[0]
    for expected, actual in zip(expected_value, actual_value):
      self.assertAllClose(expected, actual)

  @parameterized.named_parameters(
      ('LSTMBatchSizeOne', tf.keras.layers.LSTM, True),
      ('LSTM', tf.keras.layers.LSTM, False),
      ('SimpleRNNBatchSizeOne', tf.keras.layers.SimpleRNN, True),
      ('SimpleRNN', tf.keras.layers.SimpleRNN, False),
      ('GRUBatchSizeOne', tf.keras.layers.GRU, True),
      ('GRU', tf.keras.layers.GRU, False),
  )
  @test_util.run_v2_only
  def testKerasRNN(self, rnn_layer, default_to_single_batch):
    # [FUNCTION DOC]
    # Parameterized over LSTM, SimpleRNN, GRU and a flag to default to single batch in tensor list ops.
    # Converts a Keras model; when the flag is off enables select TF ops. Asserts closeness (1e-5) to Keras.
    input_data = tf.constant(
        np.array(np.random.random_sample((1, 10, 10)), dtype=np.float32)
    )
    rnn_obj = rnn_layer(units=10, input_shape=(10, 10))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(10, 10), name='input'),
        rnn_obj,
    ])

    # Convert model.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter._experimental_default_to_single_batch_in_tensor_list_ops = (
        default_to_single_batch
    )
    if not default_to_single_batch:
      converter.target_spec.supported_ops = [
          lite.OpsSet.TFLITE_BUILTINS,
          lite.OpsSet.SELECT_TF_OPS,
      ]
    tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]

    # Check values from converted model.
    expected_value = model.predict(input_data)
    self.assertAllClose(expected_value, actual_value, atol=1e-05)

  @parameterized.named_parameters(
      ('LSTM', tf.keras.layers.LSTM),
      ('SimpleRNN', tf.keras.layers.SimpleRNN),
      ('GRU', tf.keras.layers.GRU),
  )
  @test_util.run_v2_only
  def testKerasRNNMultiBatches(self, rnn_layer):
    # [FUNCTION DOC]
    # Parameterized over LSTM, SimpleRNN, GRU. Fixed batch size 4; converts and compares with Keras.
    input_data = tf.constant(
        np.array(np.random.random_sample((4, 10, 10)), dtype=np.float32)
    )
    # Specify a fixed batch size(4) for the test model.
    x = tf.keras.layers.Input(batch_shape=(4, 10, 10))
    y = rnn_layer(units=10, input_shape=(10, 10))(x)
    model = tf.keras.Model(inputs=[x], outputs=[y])

    # Convert model.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]

    # Check values from converted model.
    expected_value = model.predict(input_data)
    self.assertAllClose(expected_value, actual_value, atol=1e-05)

  @test_util.run_v2_only
  def testKerasRNNLSTMFloat16Quant(self):
    # [FUNCTION DOC]
    # Converts an LSTM model with float16 quantization and compares to Keras within 1e-3.
    input_data = tf.constant(
        np.array(np.random.random_sample((4, 10, 10)), dtype=np.float32)
    )
    # Specify a fixed batch size(4) for the test model.
    x = tf.keras.layers.Input(batch_shape=(4, 10, 10))
    y = tf.keras.layers.LSTM(units=10, input_shape=(10, 10))(x)
    model = tf.keras.Model(inputs=[x], outputs=[y])

    # Convert model.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]

    # Check values from converted model.
    expected_value = model.predict(input_data)
    self.assertAllClose(expected_value, actual_value, atol=1e-03)

  @parameterized.named_parameters(
      ('ForceToUseBatchSizeOne', True), ('DontForceToUseBatchSizeOne', False)
  )
  @test_util.run_v2_only
  def testKerasBidirectionalRNNReturnSequence(self, default_to_single_batch):
    # [FUNCTION DOC]
    # Parameterized by the single batch flag. Converts a bidirectional LSTM returning sequences followed by
    # flatten, dense and softmax; compares to Keras within 1e-5.
    input_data = tf.constant(
        np.array(np.random.random_sample((1, 10, 10)), dtype=np.float32)
    )
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(10, 10), name='input'))
    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=10, return_sequences=True),
            input_shape=(10, 10),
        )
    )
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(5))
    model.add(tf.keras.layers.Activation('softmax'))

    # Convert model.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter._experimental_default_to_single_batch_in_tensor_list_ops = (
        default_to_single_batch
    )
    if not default_to_single_batch:
      converter.target_spec.supported_ops = [
          lite.OpsSet.TFLITE_BUILTINS,
          lite.OpsSet.SELECT_TF_OPS,
      ]
    tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]

    # Check values from converted model.
    expected_value = model.predict(input_data)
    self.assertAllClose(expected_value, actual_value, atol=1e-05)

  @parameterized.named_parameters(
      ('ForceToUseBatchSizeOne', True), ('DontForceToUseBatchSizeOne', False)
  )
  @test_util.run_v2_only
  def testKerasBidirectionalRNN(self, default_to_single_batch):
    # [FUNCTION DOC]
    # Parameterized by the single batch flag. Converts a bidirectional LSTM with dense and softmax layers;
    # compares with Keras within 1e-5.
    input_data = tf.constant(
        np.array(np.random.random_sample((1, 10, 10)), dtype=np.float32)
    )
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(10, 10), name='input'))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=10)))
    model.add(tf.keras.layers.Dense(5))
    model.add(tf.keras.layers.Activation('softmax'))

    # Convert model.
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter._experimental_default_to_single_batch_in_tensor_list_ops = (
        default_to_single_batch
    )
    if not default_to_single_batch:
      converter.target_spec.supported_ops = [
          lite.OpsSet.TFLITE_BUILTINS,
          lite.OpsSet.SELECT_TF_OPS,
      ]
    tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]

    # Check values from converted model.
    expected_value = model.predict(input_data)
    self.assertAllClose(expected_value, actual_value, atol=1e-05)


class StridedSliceTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for the strided slice operation with variable begin index.

  @test_util.run_v2_only
  def testStridedSlice(self):
    # [FUNCTION DOC]
    # Converts a traced strided slice whose begin index is an input; runs with begin 1 over six values and
    # compares with TensorFlow.
    input_data = tf.constant(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6,
        ],
        shape=[6],
        dtype=tf.float32,
    )
    begin = tf.Variable([1], dtype=tf.int32)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[6], dtype=tf.float32),
            tf.TensorSpec(shape=[1], dtype=tf.int32),
        ]
    )
    def model(a, begin):
      # [FUNCTION DOC]
      # HELPER: strided slice of a from begin to begin plus three.
      return tf.strided_slice(a, begin, begin + 3)

    concrete_func = model.get_concrete_function()

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data, begin)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data, begin])[
        0
    ]
    self.assertAllClose(expected_value, actual_value)


class GrapplerTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for graph optimizations (Grappler) such as constant folding.

  @test_util.run_v2_only
  def testConstantFolding(self):
    # [FUNCTION DOC]
    # Converts a function that broadcasts a constant and multiplies; folding removes the unsupported
    # broadcast. Compares results, then repeats with default optimizations and compares again.
    # Constant folding handles the tf.broadcast_to operation which was not
    # supported by the TFLite at the time this test was added.
    input_data = tf.constant(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], shape=[3, 3]
    )

    @tf.function
    def func(x):
      # [FUNCTION DOC]
      # HELPER: matrix product of x with a broadcast 3x3 constant.
      y_const = tf.constant([1.0, 2.0, 3.0])
      y_broadcast = tf.broadcast_to(y_const, [3, 3])
      return tf.matmul(x, y_broadcast)

    root = autotrackable.AutoTrackable()
    root.f = func
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
    self.assertAllClose(expected_value, actual_value)

    # Enable hybrid quantization, same result
    converter.optimizations = [lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
    self.assertAllClose(expected_value, actual_value)


class UnknownShapes(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for models with unknown (None) dimensions: matmul, quantization and batch matmul variants.

  @test_util.run_v2_only
  def testMatMul(self):
    # [FUNCTION DOC]
    # Converts a function on an input with unknown batch size; runs with resized input and compares with
    # tolerance 1e-6.
    input_data = tf.constant(
        np.array(np.random.random_sample((10, 4)), dtype=np.float32)
    )

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)]
    )
    def model(in_tensor):
      # [FUNCTION DOC]
      # HELPER: multiplies the transposed ones-matrix (shape from input) with the input.
      shape = tf.shape(in_tensor)
      fill = tf.transpose(tf.fill(shape, 1.0))
      return tf.matmul(fill, in_tensor)

    concrete_func = model.get_concrete_function()

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)
    actual_value = self._evaluateTFLiteModel(
        tflite_model, [input_data], input_shapes=[([-1, 4], [10, 4])]
    )[0]
    self.assertAllClose(expected_value, actual_value, atol=1e-06)

  def _getIntegerQuantizeModelWithUnknownShapes(self):
    # [FUNCTION DOC]
    # HELPER: model with unknown batch size and a constant 33x33 matrix (more than 1024 elements for weight
    # quantization), plus calibration generator. Returns owner, concrete function and generator.
    np.random.seed(0)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 33], dtype=tf.float32)]
    )
    def model(input_tensor):
      # [FUNCTION DOC]
      # HELPER: two chained matmuls, the second with a random constant matrix.
      """Define a model with tf.MatMul and unknown shapes."""
      # We need the tensor to have more than 1024 elements for quantize_weights
      # to kick in. Thus, the [33, 33] shape.
      const_tensor = tf.constant(
          np.random.uniform(low=-10.0, high=10.0, size=[33, 33]),
          shape=[33, 33],
          dtype=tf.float32,
          name='inputB',
      )

      shape = tf.shape(input_tensor)
      fill = tf.transpose(tf.fill(shape, 1.0))
      mult = tf.matmul(fill, input_tensor)
      return tf.matmul(mult, const_tensor)

    root = autotrackable.AutoTrackable()
    root.f = model
    concrete_func = root.f.get_concrete_function()

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields 15 samples with batch sizes 5, 10, 15 of random 33-wide float32 arrays.
      for batch in range(5, 20, 5):
        for _ in range(5):
          yield [np.random.uniform(-1, 1, size=(batch, 33)).astype(np.float32)]

    return root, concrete_func, calibration_gen

  @test_util.run_v2_only
  def testMatMulQuantize(self):
    # [FUNCTION DOC]
    # Dynamic-range quantizes the unknown-shape model (no calibration); asserts float32 input, shape
    # signature [-1, 33] and smaller size than the float model.
    root, concrete_func, _ = self._getIntegerQuantizeModelWithUnknownShapes()
    float_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    float_tflite_model = float_converter.convert()

    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_tflite_model = quantized_converter.convert()

    # The default input and output types should be float.
    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([-1, 33], input_details[0]['shape_signature'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  @test_util.run_v2_only
  def testMatMulCalibrateAndQuantize(self):
    # [FUNCTION DOC]
    # Same as above but with a representative dataset.
    root, concrete_func, calibration_gen = (
        self._getIntegerQuantizeModelWithUnknownShapes()
    )
    float_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    float_tflite_model = float_converter.convert()

    quantized_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    quantized_converter.optimizations = [lite.Optimize.DEFAULT]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite_model = quantized_converter.convert()

    # The default input and output types should be float.
    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    self.assertLen(input_details, 1)
    self.assertEqual(np.float32, input_details[0]['dtype'])
    self.assertAllEqual([-1, 33], input_details[0]['shape_signature'])

    # Ensure that the quantized weights tflite model is smaller.
    self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

  def testBatchMatMul(self):
    # [FUNCTION DOC]
    # Converts batch matmul with unknown batch size on two 256x256 inputs; compares with tolerance 4.
    input_data_1 = tf.constant(
        np.array(np.random.random_sample((1, 256, 256)), dtype=np.float32)
    )
    input_data_2 = tf.constant(
        np.array(np.random.random_sample((1, 256, 256)), dtype=np.float32)
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 256, 256], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 256, 256], dtype=tf.float32),
        ]
    )
    def model(in_tensor_1, in_tensor_2):
      # [FUNCTION DOC]
      # HELPER: batch matmul of two inputs.
      return tf.matmul(in_tensor_1, in_tensor_2)

    concrete_func = model.get_concrete_function()

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data_1, input_data_2)
    actual_value = self._evaluateTFLiteModel(
        tflite_model,
        [input_data_1, input_data_2],
        input_shapes=[([-1, 256, 256], [1, 256, 256])],
    )[0]
    self.assertAllClose(expected_value, actual_value, atol=4)

  def testBatchMatMulInputInt8Int8OutputInt32(self):
    # [FUNCTION DOC]
    # Batch matmul with int8 inputs and int32 output type; asserts exact equality with TensorFlow.
    # Uses a deprecated numpy random integer function for input data.
    input_data_1 = tf.constant(
        np.array(
            np.random.random_integers(-128, high=127, size=(1, 20, 30)),
            dtype=np.int8,
        )
    )
    input_data_2 = tf.constant(
        np.array(
            np.random.random_integers(-128, high=127, size=(1, 30, 10)),
            dtype=np.int8,
        )
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 20, 30], dtype=tf.int8),
            tf.TensorSpec(shape=[None, 30, 10], dtype=tf.int8),
        ]
    )
    def model(in_tensor_1, in_tensor_2):
      # [FUNCTION DOC]
      # HELPER: batch matmul of int8 inputs with int32 output type.
      return tf.matmul(in_tensor_1, in_tensor_2, output_type=tf.int32)

    concrete_func = model.get_concrete_function()

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data_1, input_data_2)
    actual_value = self._evaluateTFLiteModel(
        tflite_model,
        [input_data_1, input_data_2],
        input_shapes=[([-1, 20, 30], [1, 20, 30]), ([-1, 30, 10], [1, 30, 10])],
    )[0]
    self.assertAllEqual(expected_value, actual_value)

  def testBatchMatMulHybrid(self):
    # [FUNCTION DOC]
    # Dynamic-range quantizes a batch matmul with a constant right-hand side (128x256) so a hybrid kernel with
    # float left and int8 right is used. Compares with TensorFlow within tolerance 4.
    # Test model that does batch matmul of:
    # lhs input (1, 256, 128), rhs const (1, 128, 256).
    # For dynamic range quantization situation, this will result in hybrid batch
    # matmul, where lhs type is float32 and rhs type is int8.

    # Intentionally set lhs, rhs sizes to satisfy following conditions:
    # 1. rhs const num_elements >= 1024, since dynamic range quantization
    # requires const tensor num_elements to be larger than
    # min_elements_for_weights (which defaults to 1024).
    # (https://github.com/tensorflow/tensorflow/blob/25e649ac3688655547da998eba2715cf70b3e5c9/tensorflow/compiler/mlir/lite/transforms/prepare_quantize_dynamic_range.cc#L262)
    # 2. batch_size (256) > accum_dim_size (128) and
    # num_units (256) > accum_dim_size (128), to test if the sizes are set
    # correctly according to dimensions. See HybridAsymmetricBatchMatMulOpTest
    # tests in
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/batch_matmul_test.cc.
    input_data = tf.constant(
        np.array(np.random.random_sample((1, 256, 128)), dtype=np.float32)
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 256, 128], dtype=tf.float32)
        ]
    )
    def model(in_tensor):
      # [FUNCTION DOC]
      # HELPER: batch matmul of the input with a random constant 1x128x256 matrix.
      rhs = tf.constant(
          np.array(np.random.random_sample((1, 128, 256)), dtype=np.float32)
      )
      return tf.matmul(in_tensor, rhs)

    concrete_func = model.get_concrete_function()

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    converter.optimizations = [lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = concrete_func(input_data)
    actual_value = self._evaluateTFLiteModel(
        tflite_model,
        [input_data],
        input_shapes=[([-1, 256, 128], [1, 256, 128])],
    )[0]
    self.assertAllClose(expected_value, actual_value, atol=4)

  def testSizeInvalid(self):
    # [FUNCTION DOC]
    # Converts with the old converter a model whose input has an unknown dimension after the first. Expected: a
    # value error that None is only supported in the first dimension, naming the tensor and shape.
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, None, 16, 3], dtype=tf.float32)
        ]
    )
    def model(in_tensor):
      # [FUNCTION DOC]
      # HELPER: adds the input to itself.
      return in_tensor + in_tensor

    concrete_func = model.get_concrete_function()

    # Test invalid shape. None after 1st dimension. Run with TOCO in order to
    # invoke shape checking code.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    converter.experimental_new_converter = False
    with self.assertRaises(ValueError) as error:
      converter.convert()
    self.assertEqual(
        'None is only supported in the 1st dimension. Tensor '
        "'in_tensor' has invalid shape '[1, None, 16, 3]'.",
        str(error.exception),
    )


class ResourceAndVariantTypes(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for resource and variant tensor types (tensor maps, stacks, tensor lists) through select TF
  # ops, including control flow.

  @test_util.run_v2_only
  def testVariants(self):
    # [FUNCTION DOC]
    # Converts a function using a tensor map insert and size with builtin and select-TF ops; invokes three
    # times and asserts result 1 each time.
    @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.float32)])
    def model(v):
      # [FUNCTION DOC]
      # HELPER: inserts a key/value pair into an empty tensor map and returns the map size.
      m = map_ops.empty_tensor_map()
      k = tf.constant(1.0)
      p = tf.add(k, v)
      with ops.control_dependencies([m]):
        m2 = map_ops.tensor_map_insert(m, p, v)
        with ops.control_dependencies([m2]):
          return map_ops.tensor_map_size(m2)

    concrete_func = model.get_concrete_function()

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.allocate_tensors()

    input_data = np.array([1.0], dtype=np.float32)
    interp.set_tensor(input_details[0]['index'], input_data)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(1, actual_value)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(1, actual_value)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(1, actual_value)

  @test_util.run_v2_only
  def testVariantsWithCond(self):
    # [FUNCTION DOC]
    # Builds a TF1 saved model with a conditional inserting into a tensor map; converts, invokes three times
    # and asserts value 1 for input 0.
    def create_v1_saved_model():
      # [FUNCTION DOC]
      # HELPER: builds and saves the TF1 graph with a conditional tensor map; returns the directory.
      saved_model_dir = os.path.join(self.get_temp_dir(), 'variants_with_cond')
      with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
          m = map_ops.empty_tensor_map()

          def body(i, m):
            # [FUNCTION DOC]
            # HELPER: inserts i as key and value into map m, returns i plus 1 and the map.
            m = map_ops.tensor_map_insert(m, i, i)
            return i + 1, m

          in_tensor = tf.compat.v1.placeholder(
              shape=[1], dtype=tf.int32, name='input'
          )
          _, result_m = tf.cond(
              in_tensor < 10,
              lambda: body(in_tensor, m),
              lambda: body(in_tensor + 1, m),
          )
          out_tensor = in_tensor + map_ops.tensor_map_size(result_m)

          inputs = {'x': in_tensor}
          outputs = {'z': out_tensor}
          saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
      return saved_model_dir

    saved_model_dir = create_v1_saved_model()

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.allocate_tensors()

    input_data = np.array([0], dtype=np.int32)
    interp.set_tensor(input_details[0]['index'], input_data)

    interp.invoke()
    expected_value = np.array([1], dtype=np.int32)
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(expected_value, actual_value)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(expected_value, actual_value)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(expected_value, actual_value)

  @test_util.run_v2_only
  def testVariantsWithWhile(self):
    # [FUNCTION DOC]
    # Builds a TF1 saved model with a while loop inserting ten entries into a tensor map; converts and asserts
    # output 10 across three invocations.
    def create_v1_saved_model():
      # [FUNCTION DOC]
      # HELPER: builds and saves the TF1 graph with a while loop over a tensor map; returns the directory.
      saved_model_dir = os.path.join(self.get_temp_dir(), 'variants_with_while')
      with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
          m = map_ops.empty_tensor_map()

          def cond(i, m):
            # [FUNCTION DOC]
            # HELPER: loop condition; true while i is below 10.
            del m
            return i < 10

          def body(i, m):
            # [FUNCTION DOC]
            # HELPER: inserts i into the map and increments i.
            m = map_ops.tensor_map_insert(m, i, i)
            return i + 1, m

          _, result_m = tf.while_loop(cond, body, [0, m])
          in_tensor = tf.compat.v1.placeholder(
              shape=[1], dtype=tf.int32, name='input'
          )
          out_tensor = in_tensor + map_ops.tensor_map_size(result_m)

          inputs = {'x': in_tensor}
          outputs = {'z': out_tensor}
          saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
      return saved_model_dir

    saved_model_dir = create_v1_saved_model()

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.allocate_tensors()

    input_data = np.array([0], dtype=np.int32)
    interp.set_tensor(input_details[0]['index'], input_data)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(10, actual_value)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(10, actual_value)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(10, actual_value)

  @test_util.run_v2_only
  def testResources(self):
    # [FUNCTION DOC]
    # Builds a TF1 saved model pushing to and popping from a stack resource; converts and asserts result 3.0
    # for input 1.0, three times.
    def create_v1_saved_model():
      # [FUNCTION DOC]
      # HELPER: builds and saves the TF1 graph with a stack push and pop; returns the directory.
      saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_resources')
      with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
          in_tensor = tf.compat.v1.placeholder(
              shape=[1], dtype=tf.float32, name='input'
          )

          stack = tf.raw_ops.StackV2(max_size=10, elem_type=tf.float32)
          w = tf.raw_ops.StackPushV2(handle=stack, elem=in_tensor)
          with ops.control_dependencies([w]):
            a = in_tensor + in_tensor
            with ops.control_dependencies([a]):
              out_tensor = a + tf.raw_ops.StackPopV2(
                  handle=stack, elem_type=tf.float32
              )

          inputs = {'x': in_tensor}
          outputs = {'z': out_tensor}
          saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
      return saved_model_dir

    saved_model_dir = create_v1_saved_model()

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.allocate_tensors()

    input_data = np.array([1.0], dtype=np.float32)
    interp.set_tensor(input_details[0]['index'], input_data)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(3.0, actual_value)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(3.0, actual_value)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(3.0, actual_value)

  @test_util.run_v2_only
  def testResourcesWithCond(self):
    # [FUNCTION DOC]
    # TF1 saved model pushing to a stack in both branches of a conditional, then popping; asserts 0.0 for input
    # 1.0.
    def create_v1_saved_model():
      # [FUNCTION DOC]
      # HELPER: builds and saves the TF1 graph with a conditional stack push; returns the directory.
      saved_model_dir = os.path.join(self.get_temp_dir(), 'resources_with_cond')
      with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
          in_tensor = tf.compat.v1.placeholder(
              shape=[1], dtype=tf.float32, name='input'
          )

          def body(i, arr):
            # [FUNCTION DOC]
            # HELPER: pushes the cast of i onto the stack and returns the push result and stack handle.
            n = tf.raw_ops.StackPushV2(
                handle=arr, elem=tf.cast(i, dtype=tf.float32)
            )
            return n, arr

          arr = tf.raw_ops.StackV2(max_size=10, elem_type=tf.float32)
          n, result_arr = tf.cond(
              in_tensor < 10, lambda: body(0, arr), lambda: body(1, arr)
          )

          with ops.control_dependencies([result_arr, n]):
            out_tensor = tf.raw_ops.StackPopV2(
                handle=result_arr, elem_type=tf.float32
            )

          inputs = {'x': in_tensor}
          outputs = {'a': out_tensor}
          saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
      return saved_model_dir

    saved_model_dir = create_v1_saved_model()

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.allocate_tensors()

    input_data = np.array([1.0], dtype=np.float32)
    interp.set_tensor(input_details[0]['index'], input_data)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(0.0, actual_value)

  @test_util.run_v2_only
  def testResourcesWithWhile(self):
    # [FUNCTION DOC]
    # TF1 saved model with a loop pushing ten values to a stack then popping; asserts 9.0.
    def create_v1_saved_model():
      # [FUNCTION DOC]
      # HELPER: builds and saves the TF1 graph with a while loop over a stack; returns the directory.
      saved_model_dir = os.path.join(
          self.get_temp_dir(), 'resources_with_while'
      )
      with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
          in_tensor = tf.compat.v1.placeholder(
              shape=[1], dtype=tf.float32, name='input'
          )

          def cond(i, arr, m):
            # [FUNCTION DOC]
            # HELPER: loop condition; true while i is below 10.
            del arr
            del m
            return i < 10

          def body(i, arr, m):
            # [FUNCTION DOC]
            # HELPER: pushes the cast of i and increments i.
            del m
            n = tf.raw_ops.StackPushV2(
                handle=arr, elem=tf.cast(i, dtype=tf.float32)
            )
            return i + 1, arr, n

          arr = tf.raw_ops.StackV2(max_size=10, elem_type=tf.float32)
          _, result_arr, n = tf.while_loop(cond, body, [0, arr, 0.0])

          with ops.control_dependencies([result_arr, n]):
            out_tensor = tf.raw_ops.StackPopV2(
                handle=result_arr, elem_type=tf.float32
            )

          inputs = {'x': in_tensor}
          outputs = {'a': out_tensor}
          saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
      return saved_model_dir

    saved_model_dir = create_v1_saved_model()

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.allocate_tensors()

    input_data = np.array([1.0], dtype=np.float32)
    interp.set_tensor(input_details[0]['index'], input_data)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(9.0, actual_value)

  @parameterized.named_parameters(
      ('EnableLoweringTensorListOps', True),
      ('DisableLoweringTensorListOps', False),
  )
  @test_util.run_v2_only
  def testTensorListWithStaticSize(self, lower_tensor_list_ops):
    # [FUNCTION DOC]
    # Parameterized: lowering of tensor list ops on or off. Converts a TF1 graph using a fixed-size tensor array
    # (10, 20, 30; reads first and last); when lowering is off also enables select TF ops. Asserts result 40.
    def create_v1_saved_model():
      # [FUNCTION DOC]
      # HELPER: builds and saves the TF1 graph with a static-size tensor array; returns the directory.
      saved_model_dir = os.path.join(
          self.get_temp_dir(), 'simple_mutable_variable'
      )
      with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
          in_tensor = tf.compat.v1.placeholder(
              shape=[1], dtype=tf.float32, name='input'
          )

          ta = tf.TensorArray(
              tf.float32, size=3, dynamic_size=False, clear_after_read=False
          )
          ta = ta.write(0, 10.0)
          ta = ta.write(1, 20.0)
          ta = ta.write(2, 30.0)

          out_tensor = ta.read(0) + ta.read(2)

          inputs = {'x': in_tensor}
          outputs = {'z': out_tensor}
          saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
      return saved_model_dir

    saved_model_dir = create_v1_saved_model()

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    if not lower_tensor_list_ops:
      converter.target_spec.supported_ops = [
          lite.OpsSet.TFLITE_BUILTINS,
          lite.OpsSet.SELECT_TF_OPS,
      ]
    converter._experimental_lower_tensor_list_ops = lower_tensor_list_ops
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.allocate_tensors()

    input_data = np.array([1.0], dtype=np.float32)
    interp.set_tensor(input_details[0]['index'], input_data)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(40.0, actual_value)

  @parameterized.named_parameters(
      ('EnableLoweringTensorListOps', True),
      ('DisableLoweringTensorListOps', False),
  )
  @test_util.run_v2_only
  def testTensorListWithDynamicSize(self, lower_tensor_list_ops):
    # [FUNCTION DOC]
    # Parameterized: lowering on or off. With lowering on, the first conversion must raise a converter error that
    # lowering failed and select TF ops should be used. Then converts with select TF ops enabled and asserts
    # the result 40.
    def create_v1_saved_model():
      # [FUNCTION DOC]
      # HELPER: builds and saves the TF1 graph with a dynamic-size tensor array; returns the directory.
      saved_model_dir = os.path.join(
          self.get_temp_dir(), 'simple_mutable_variable'
      )
      with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
          in_tensor = tf.compat.v1.placeholder(
              shape=[1], dtype=tf.float32, name='input'
          )

          ta = tf.TensorArray(
              tf.float32, size=0, dynamic_size=True, clear_after_read=False
          )
          ta = ta.write(0, 10.0)
          ta = ta.write(1, 20.0)
          ta = ta.write(2, 30.0)

          out_tensor = ta.read(0) + ta.read(2)

          inputs = {'x': in_tensor}
          outputs = {'z': out_tensor}
          saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
      return saved_model_dir

    saved_model_dir = create_v1_saved_model()

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    if lower_tensor_list_ops:
      with self.assertRaises(convert.ConverterError) as error:
        converter.convert()
      self.assertIn(
          'Lowering tensor list ops is failed. Please consider using Select '
          'TF ops and disabling `_experimental_lower_tensor_list_ops` flag in '
          'the TFLite converter object.',
          str(error.exception),
      )

    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.allocate_tensors()

    input_data = np.array([1.0], dtype=np.float32)
    interp.set_tensor(input_details[0]['index'], input_data)

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(40.0, actual_value)


class CalibrateAndQuantizeWithCustomOpTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for calibration and quantization of a model containing a custom op (double), where custom op
  # registerers are given by name or by function. Relies on the test registerer extension module.

  def _createGraphWithCustomOp(self):
    # [FUNCTION DOC]
    # HELPER: saves a TF1 model with one custom Double op on a 1x4 input and returns a pair: the directory and
    # a calibration generator yielding 100 random 1x4 samples. Seeds numpy.
    # Create a graph that has one double op.
    np.random.seed(0)

    saved_model_dir = os.path.join(self.get_temp_dir(), 'double_model')
    with ops.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        in_tensor = tf.compat.v1.placeholder(
            shape=[1, 4], dtype=dtypes.float32, name='input'
        )
        out_tensor = double_op.double(in_tensor)
        inputs = {'x': in_tensor}
        outputs = {'z': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields 100 single-input lists of random 1x4 float32 arrays in [-1, 1].
      for _ in range(100):
        yield [np.random.uniform(-1, 1, size=(1, 4)).astype(np.float32)]

    return (saved_model_dir, calibration_gen)

  def testCustomOpRegistererByName(self):
    # [FUNCTION DOC]
    # Converts with custom ops allowed and the registerer given by name. Asserts the registerer was called,
    # Double appears in the ops list, metadata allows custom ops, and the model runs correctly in an interpreter
    # with custom ops (input 0 to 0.3 doubled).
    """Test a calibration with custom op registered by name."""
    saved_model_dir, calibration_gen = self._createGraphWithCustomOp()

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_gen
    converter.allow_custom_ops = True
    converter.target_spec._experimental_custom_op_registerers = [
        'TF_TestRegisterer'
    ]
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    self.assertGreater(test_registerer.get_num_test_registerer_calls(), 0)
    self.assertIn('Double', tflite_test_util.get_ops_list(tflite_model))
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(metadata.options.allowCustomOps, True)

    # Check the model works with custom ops.
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model, custom_op_registerers=['TF_TestRegisterer']
    )
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    test_input = np.array([[0.0, 0.1, 0.2, 0.3]], dtype=np.float32)
    interp.set_tensor(input_details[0]['index'], test_input)
    interp.invoke()

    output_details = interp.get_output_details()
    expected_output = np.array([[0.0, 0.2, 0.4, 0.6]], dtype=np.float32)
    output_data = interp.get_tensor(output_details[0]['index'])
    self.assertArrayNear(expected_output[0], output_data[0], err=1e-2)

  def testCustomOpRegistererByFunc(self):
    # [FUNCTION DOC]
    # Same as the by-name test but passing the registerer as a function. No metadata check.
    """Test a calibration with custom op registered by function."""
    saved_model_dir, calibration_gen = self._createGraphWithCustomOp()

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_gen
    converter.allow_custom_ops = True
    converter.target_spec._experimental_custom_op_registerers = [
        test_registerer.TF_TestRegisterer
    ]
    tflite_model = converter.convert()
    self.assertTrue(tflite_model)
    self.assertGreater(test_registerer.get_num_test_registerer_calls(), 0)
    self.assertIn('Double', tflite_test_util.get_ops_list(tflite_model))

    # Check the model works with custom ops.
    interp = interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=[test_registerer.TF_TestRegisterer],
    )
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    test_input = np.array([[0.0, 0.1, 0.2, 0.3]], dtype=np.float32)
    interp.set_tensor(input_details[0]['index'], test_input)
    interp.invoke()

    output_details = interp.get_output_details()
    expected_output = np.array([[0.0, 0.2, 0.4, 0.6]], dtype=np.float32)
    output_data = interp.get_tensor(output_details[0]['index'])
    self.assertArrayNear(expected_output[0], output_data[0], err=1e-2)

  def testCustomOpRegistererFailure(self):
    # [FUNCTION DOC]
    # Using an unknown registerer name must raise a value error that the lookup of that symbol failed.
    """Test a calibration with wrong custom op registerer."""
    saved_model_dir, calibration_gen = self._createGraphWithCustomOp()

    bogus_name = 'CompletelyBogusRegistererName'

    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = calibration_gen
    converter.allow_custom_ops = True
    converter.target_spec._experimental_custom_op_registerers = [bogus_name]

    with self.assertRaisesRegex(
        ValueError, "Looking up symbol '" + bogus_name + "' failed"
    ):
      converter.convert()


class IntermediatesTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for the option to preserve all intermediate tensors in the interpreter.

  def _run(self, experimental_preserve_all_tensors):
    # [FUNCTION DOC]
    # HELPER: converts a chain of three adds and runs it with or without preserving intermediates.
    # Parameter experimental_preserve_all_tensors (boolean). Returns a pair: mapping from tensor names to values
    # (nothing where unavailable) and the output value.
    @tf.function
    def f(x):
      # [FUNCTION DOC]
      # HELPER: traced function computing y, z and w by repeated doubling.
      y = tf.add(x, x, name='y')
      z = tf.add(y, y, name='z')
      w = tf.add(z, z, name='w')
      return w

    # NOTE this is exactly representable as a float as are the intermediates of
    # f. So direct comparison is ok below.

    input_data = np.array(2.0, np.float32)
    concrete_func = f.get_concrete_function(input_data)
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], f
    )
    tflite_model = converter.convert()
    interp = interpreter.Interpreter(
        model_content=tflite_model,
        experimental_preserve_all_tensors=experimental_preserve_all_tensors,
    )
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]['index'], input_data)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]['index'])
    tensors = {}
    for t in interp.get_tensor_details():
      # With Tensorflow Lite default delegate applied to the model graph, the
      # access to original tensors of a delegated op could cause a ValueError
      # (i.e. 'Tensor data is null. Run allocate_tensors() first') to be thrown
      # out because the tensor memory isn't allocated at all.
      val = None
      try:
        val = interp.get_tensor(t['index'])
      except ValueError:
        pass
      tensors.update({t['name']: val})
    return (tensors, out)

  def testPreserve(self):
    # [FUNCTION DOC]
    # All intermediate tensors must have their true values (2, 4, 8) and result 16.
    tensors, result = self._run(experimental_preserve_all_tensors=True)
    # All intermediates should be true and result be true.
    self.assertAllClose(tensors['x'], 2.0)
    self.assertAllClose(tensors['y'], 4.0)
    self.assertAllClose(tensors['z'], 8.0)
    self.assertAllClose(result, 16.0)

  def testNoPreserve(self):
    # [FUNCTION DOC]
    # Without preserving, input must still be 2 and result 16, but at least one intermediate must differ.
    tensors, result = self._run(experimental_preserve_all_tensors=False)
    # One of them should be wrong if preserve is not true, but result should be
    # ok. Input should still be ok for repeated invocation.
    self.assertAllClose(tensors['x'], 2.0)
    self.assertTrue(tensors['y'] != 4.0 or tensors['z'] != 8.0)
    self.assertAllClose(result, 16.0)


class DatasetOpsTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for models using dataset operations through select TF ops.

  @test_util.run_v2_only
  def testReduceDataset(self):
    # [FUNCTION DOC]
    # Converts a function reducing a dataset of 1 to 4 and asserts output 10.
    @tf.function
    def model():
      # [FUNCTION DOC]
      # HELPER: builds a dataset from four values and sums it with reduce.
      dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
      output = dataset.reduce(np.int32(0), lambda x, y: x + y)
      return output

    concrete_func = model.get_concrete_function()
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], model
    )
    converter.target_spec.supported_ops = [
        lite.OpsSet.TFLITE_BUILTINS,
        lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    self.assertIsNotNone(tflite_model)

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    output_details = interp.get_output_details()

    interp.allocate_tensors()

    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertEqual(10, actual_value)


class SparsityTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for the sparsity optimization and its conversion metadata.

  def _getSparsificableModel(self, matrix_b_values):
    # [FUNCTION DOC]
    # HELPER: builds a model multiplying a 16x4 input by a constant 4x8 matrix with given values, then relu.
    # Parameter matrix_b_values: 32 numbers. Returns owner and concrete function.
    np.random.seed(0)
    root = autotrackable.AutoTrackable()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[16, 4], dtype=tf.float32)]
    )
    def func(inp):
      # [FUNCTION DOC]
      # HELPER: matmul with the constant matrix then relu.
      matrix_b = tf.constant(matrix_b_values, dtype=tf.float32)
      matrix_b = tf.reshape(matrix_b, [4, 8])
      matmul = tf.matmul(inp, matrix_b, transpose_a=False, transpose_b=False)
      output = tf.nn.relu(matmul, name='output')
      return output

    root.f = func
    to_save = root.f.get_concrete_function()
    return (root, to_save)

  def testRandomSparsity(self):
    # [FUNCTION DOC]
    # Converts with the experimental sparsity option using a random-pattern sparse matrix; asserts metadata
    # mode random sparsity.
    # pyformat: disable
    matrix_b_values = [
        0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1,
    ]
    # pyformat: enable
    root, func = self._getSparsificableModel(matrix_b_values)
    float_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    float_converter.optimizations = [lite.Optimize.EXPERIMENTAL_SPARSITY]
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(float_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertAllEqual(
        [metadata_fb.ModelOptimizationMode.RANDOM_SPARSITY],
        metadata.options.modelOptimizationModes,
    )

  def testBlockSparsity(self):
    # [FUNCTION DOC]
    # Converts with sparsity option using a matrix with one nonzero (block pattern); asserts metadata block
    # sparsity.
    # pyformat: disable
    matrix_b_values = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
    ]
    # pyformat: enable
    root, func = self._getSparsificableModel(matrix_b_values)
    float_converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func], root
    )
    float_converter.optimizations = [lite.Optimize.EXPERIMENTAL_SPARSITY]
    float_tflite_model = float_converter.convert()
    self.assertIsNotNone(float_tflite_model)
    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(float_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertAllEqual(
        [metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY],
        metadata.options.modelOptimizationModes,
    )

  @parameterized.named_parameters(
      ('_PerChannelQuantForDense', False),
      ('_PerTensorQuantForDense', True),
  )
  def testQuantizedBlockSparsity(
      self, disable_per_channel_quantization_for_dense_layers
  ):
    # [FUNCTION DOC]
    # Parameterized: per-channel versus per-tensor quantization for dense. Converts a dense Keras model with
    # specific sparse weights using sparsity plus default optimizations; asserts metadata modes full-integer
    # PTQ and block sparsity, and output near expected values.
    weight_values = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 7, 0, 0, 0, -6, -2, 0, 0, 0, 0, 0, -2, 0, 6],
    ])

    custom_init = tf.constant_initializer(weight_values.transpose())
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            8, kernel_initializer=custom_init, input_shape=[16]
        )
    ])

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields ten random 1x16 float32 arrays scaled by 16.
      for _ in range(10):
        yield [np.random.uniform(-1, 1, size=(1, 16)).astype(np.float32) * 16]

    quantized_converter = lite.TFLiteConverterV2.from_keras_model(model)
    quantized_converter.optimizations = [
        lite.Optimize.EXPERIMENTAL_SPARSITY,
        lite.Optimize.DEFAULT,
    ]
    quantized_converter.representative_dataset = calibration_gen
    quantized_converter._experimental_disable_per_channel_quantization_for_dense_layers = (
        disable_per_channel_quantization_for_dense_layers
    )
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(quantized_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(
        metadata.environment.tensorflowVersion.decode('utf-8'),
        versions.__version__,
    )
    self.assertEqual(metadata.environment.apiVersion, 2)
    self.assertAllEqual(
        [
            metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER,
            metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY,
        ],
        metadata.options.modelOptimizationModes,
    )

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    interp.allocate_tensors()
    input_data = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=np.float32,
    )
    interp.set_tensor(input_details[0]['index'], input_data)
    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertArrayNear(
        np.array([0, 87, 0, 0, 0, 0, 0, 34], dtype=np.float32),
        actual_value.flatten(),
        err=1,
    )

  def testQuantizedButNotEnoughBlockSparsity(self):
    # [FUNCTION DOC]
    # Weights are only 25 percent sparse so sparse conversion is not applied; asserts metadata only lists
    # full-integer PTQ and output near the expected values.
    # Sparsity level is 25%, which is not enough to apply the sparse conversion.
    weight_values = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 4, -3, 4, 4, 1, -2, -2, 1, 3, 4, 1, 1, 1, -4, -5],
        [1, 1, 5, -1, 3, -1, 1, -3, 4, -3, 2, -3, 3, -1, 3, -4],
        [0, -3, -2, 5, 4, 2, 1, 4, -4, 4, 1, -2, 3, -2, -2, -1],
    ])

    custom_init = tf.constant_initializer(weight_values.transpose())
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            4, kernel_initializer=custom_init, input_shape=[16]
        )
    ])

    def calibration_gen():
      # [FUNCTION DOC]
      # HELPER: yields ten random 1x16 float32 arrays scaled by 16.
      for _ in range(10):
        yield [np.random.uniform(-1, 1, size=(1, 16)).astype(np.float32) * 16]

    quantized_converter = lite.TFLiteConverterV2.from_keras_model(model)
    quantized_converter.optimizations = [
        lite.Optimize.EXPERIMENTAL_SPARSITY,
        lite.Optimize.DEFAULT,
    ]
    quantized_converter.representative_dataset = calibration_gen
    quantized_tflite_model = quantized_converter.convert()
    self.assertIsNotNone(quantized_tflite_model)

    # Check the conversion metadata.
    metadata = util.get_conversion_metadata(quantized_tflite_model)
    self.assertIsNotNone(metadata)
    self.assertEqual(
        metadata.environment.tensorflowVersion.decode('utf-8'),
        versions.__version__,
    )
    self.assertEqual(metadata.environment.apiVersion, 2)
    self.assertAllEqual(
        [
            metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER,
        ],
        metadata.options.modelOptimizationModes,
    )
    self.assertNotIn(
        metadata_fb.ModelOptimizationMode.RANDOM_SPARSITY,
        metadata.options.modelOptimizationModes,
    )
    self.assertNotIn(
        metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY,
        metadata.options.modelOptimizationModes,
    )

    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=quantized_tflite_model)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    interp.allocate_tensors()
    input_data = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=np.float32,
    )
    interp.set_tensor(input_details[0]['index'], input_data)
    interp.invoke()
    actual_value = interp.get_tensor(output_details[0]['index'])
    self.assertArrayNear(
        np.array([0, -3, 4, 35], dtype=np.float32),
        actual_value.flatten(),
        err=1,
    )


class BufferOffsetTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for the private buffer-offset flatbuffer option.

  @test_util.run_v2_only
  def testCOncreteFunctionFloat(self):
    # [FUNCTION DOC]
    # Converts with buffer offset on and compares with TensorFlow. (Name has a typo: COncrete.)
    root = self._getSimpleVariableModel()
    input_data = tf.constant(1.0, shape=[1])
    concrete_func = root.f.get_concrete_function(input_data)

    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    converter._experimental_use_buffer_offset = True
    tflite_model = converter.convert()

    # Check output value from converted model.
    expected_value = root.f(input_data)
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testConcreteFunctionStringInput(self):
    # [FUNCTION DOC]
    # Same as string input test with buffer offset: expects a value error about non-numpy input.
    class Model(tf.Module):
      # [CLASS DOC]
      # HELPER: tiny module returning its string input unchanged.

      @tf.function
      def __call__(self, x):
        # [FUNCTION DOC]
        # HELPER: identity traced method.
        return x

    root = Model()
    concrete_func = root.__call__.get_concrete_function(
        tf.constant([str(x) for x in range(11)])
    )
    # Convert model.
    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [concrete_func], root
    )
    converter._experimental_use_buffer_offset = True
    tflite_model = converter.convert()
    input_data = tf.constant(
        [str(x) for x in range(11)], shape=(11,), dtype=tf.dtypes.string
    )
    # Check values from converted model.
    interp = interpreter.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    my_signature = interp.get_signature_runner()

    with self.assertRaises(ValueError) as error:
      _ = my_signature(x=input_data)
    self.assertIn(
        'Passed in value type is not a numpy array, got type ',
        str(error.exception),
    )

  @test_util.run_v2_only
  def testSavedModelSignatureDefs(self):
    # [FUNCTION DOC]
    # Converts a saved model with the mul_add signature with buffer offset on; asserts signature structure and
    # value.
    """Test converting SignatureDef is correct and uses SignatureDef API."""
    root = self._getMultiFunctionModel()
    input_data_0 = tf.constant(1.0, shape=[1])
    input_data_1 = tf.constant(3.0, shape=[1])
    mul_add_func = root.mul_add.get_concrete_function(
        input_data_1, input_data_0
    )

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, {'mul_add': mul_add_func})

    converter = lite.TFLiteConverterV2.from_saved_model(
        save_dir, signature_keys=['mul_add']
    )
    converter._experimental_use_buffer_offset = True
    tflite_model = converter.convert()

    # Check values from converted model.
    expected_value = root.mul_add(input_data_1, input_data_0)
    interp = interpreter.Interpreter(model_content=tflite_model)
    signature_defs = interp.get_signature_list()
    results = self._evaluateTFLiteModelUsingSignatureDef(
        tflite_model, 'mul_add', {'y': input_data_0, 'x': input_data_1}
    )
    self.assertEqual(list(results.keys()), ['output_0'])
    self.assertEqual(expected_value.numpy(), results['output_0'])

    # Verify the SignatureDef structure returned is as expected.
    self.assertLen(signature_defs, 1)
    self.assertEqual(list(signature_defs.keys()), ['mul_add'])
    self.assertLen(signature_defs.values(), 1)
    self.assertEqual(
        list(signature_defs['mul_add'].keys()), ['inputs', 'outputs']
    )
    self.assertCountEqual(signature_defs['mul_add']['inputs'], ['x', 'y'])
    self.assertEqual(list(signature_defs['mul_add']['outputs']), ['output_0'])


class BoundaryValueTest(lite_v2_test_util.ModelTest):
  # [CLASS DOC]
  # Test case for infinity handling when converting.

  @parameterized.named_parameters(
      ('EnableCanonicalizeInfAsMaxMinFloatFromSavedModel', True, True),
      ('DisableCanonicalizeInfAsMaxMinFloatFromSavedModel', False, True),
      ('EnableCanonicalizeInfAsMaxMinFloatFromConcreteFunc', True, False),
      ('DisableCanonicalizeInfAsMaxMinFloatFromConcreteFunc', False, False),
  )
  @test_util.run_v2_only
  def testFloatBoundaryValue(self, is_canonicalized, is_from_saved_model):
    # [FUNCTION DOC]
    # Parameterized over canonicalization of infinity as maximum float and source (saved model or concrete
    # function). Asserts the output equals max float32 when canonicalized, else infinity.
    root = self._getInfFloatModel()
    input_data = None
    concrete_func = root.f.get_concrete_function(input_data)

    mdl = tf.Module()
    mdl.f = concrete_func

    def _get_converter() -> lite.TFLiteConverterV2:
      # [FUNCTION DOC]
      # HELPER: creates the converter from a saved model (saves the module first) or from the concrete function.
      # Returns the converter.
      if is_from_saved_model:
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        tf.saved_model.save(mdl, save_dir)
        return lite.TFLiteConverterV2.from_saved_model(save_dir)
      return lite.TFLiteConverterV2.from_concrete_functions(
          [concrete_func], root
      )

    converter = _get_converter()
    converter.canonicalizing_inf_as_min_max_float = is_canonicalized
    tflite_model = converter.convert()

    # Check output value from converted model.
    expected_value = [np.finfo(np.float32).max if is_canonicalized else np.inf]
    actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
    self.assertEqual(expected_value, actual_value)


if __name__ == '__main__':
  test.main()
