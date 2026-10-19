# [MODULE DOC]
# Tests for the calibrator: calibration with quantization for several activation types, single op
# quantization, string inputs, multiple inputs, invalid models and generators, plain calibration,
# adding intermediate tensors, and calibration of a model that uses offset buffers. Uses model
# files from the test data directories.
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
"""Tests for Calibrator."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.lite.python import lite
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python.optimize import calibrator as _calibrator
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


def _uses_buffer_offset(model: schema_fb.ModelT) -> bool:
  # [FUNCTION DOC]
  # HELPER: test utility telling whether a model object uses offset buffers.
  # Takes a TFLite model object; returns true if its metadata list contains an entry named buffer
  # location, false if there is no metadata or no such entry.
  """Determines whether the model is using an offset buffer.

  Args:
    model: A TFLite model.

  Returns:
    True iff the model is using offset buffers. Offset buffers are enabled by
    the flag `_experimental_use_buffer_offset`.
  """
  if not model.metadata:
    return False

  return any(
      map(
          lambda metadata: metadata.name.decode('utf-8') == 'buffer_location',
          model.metadata,
      )
  )


class CalibratorTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the calibrator. Most tests load a small float model from test data,
  # create a calibrator, feed generated samples and check results or errors. No shared fixtures.

  @parameterized.named_parameters(
      # Activation type Int8
      ('UseActivationTypeInt8', dtypes.int8),
      # Activation type Int16
      ('UseActivationTypeInt16', dtypes.int16),
  )
  def test_calibration_with_quantization(self, activations_type):
    # [FUNCTION DOC]
    # Parameterized by activation type (8-bit or 16-bit integer). Loads the mobilenet-like float
    # model, calibrates and quantizes with ten ones-filled samples, float fallback disallowed,
    # and asserts a model is returned.
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin'
    )
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator for the model.
    def input_gen():
      # [FUNCTION DOC]
      # Test helper generator: yields ten samples, each a list with one all-ones float32 array
      # of shape one by five by five by three.
      for _ in range(10):
        yield [np.ones(shape=(1, 5, 5, 3), dtype=np.float32)]

    quantized_model = quantizer.calibrate_and_quantize(
        input_gen, dtypes.float32, dtypes.float32, False, activations_type
    )
    self.assertIsNotNone(quantized_model)

  @parameterized.named_parameters(
      # Activation type Int8
      ('UseActivationTypeInt8', dtypes.int8),
      # Activation type Int16
      ('UseActivationTypeInt16', dtypes.int16),
  )
  def test_calibration_with_quantization_allow_float(self, activations_type):
    # [FUNCTION DOC]
    # Same as the basic quantization test but with float fallback allowed; asserts a model is
    # returned for both activation types.
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin'
    )
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator for the model.
    def input_gen():
      # [FUNCTION DOC]
      # Test helper generator: yields ten samples, each a list with one all-ones float32 array
      # of shape one by five by five by three.
      for _ in range(10):
        yield [np.ones(shape=(1, 5, 5, 3), dtype=np.float32)]

    quantized_model = quantizer.calibrate_and_quantize(
        input_gen, dtypes.float32, dtypes.float32, True, activations_type
    )
    self.assertIsNotNone(quantized_model)

  def test_calibration_with_quantization_single_op(self):
    # [FUNCTION DOC]
    # Quantizes only the op with a named bias-add output of the mobilenet-like model using ten
    # ones-filled samples; asserts a model is returned.
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin'
    )
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator for the model.
    def input_gen():
      # [FUNCTION DOC]
      # Test helper generator: yields ten samples, each a list with one all-ones float32 array
      # of shape one by five by five by three.
      for _ in range(10):
        yield [np.ones(shape=(1, 5, 5, 3), dtype=np.float32)]

    quantized_model = quantizer.calibrate_and_quantize_single(
        input_gen, dtypes.float32, dtypes.float32, True, 'conv2d_8/BiasAdd'
    )
    self.assertIsNotNone(quantized_model)

  def test_calibration_with_string_input(self):
    # [FUNCTION DOC]
    # Loads a model with a string input and Select TF ops, feeds ten string samples, quantizes
    # the single identity op, and asserts a model is returned.
    model_path = resource_loader.get_path_to_datafile(
        'test_data/string_input_flex_model.bin'
    )
    with open(model_path, 'rb') as fp:
      model_with_string_input = fp.read()
    quantizer = _calibrator.Calibrator(model_with_string_input)

    # Input generator for the model.
    def input_gen():
      # [FUNCTION DOC]
      # Test helper generator: yields ten samples, each a list with one string array holding the
      # text Test followed by the sample index.
      for i in range(10):
        yield [np.array('Test' + str(i))]

    quantized_model = quantizer.calibrate_and_quantize_single(
        input_gen, dtypes.float32, dtypes.float32, True, 'Identity'
    )
    self.assertIsNotNone(quantized_model)

  @parameterized.named_parameters(
      # Activation type Int8
      ('UseActivationTypeInt8 - EnableMlirQuantizer', dtypes.int8),
      # Activation type Int16
      ('UseActivationTypeInt16 - DisableEnableMlirQuantizer', dtypes.int16),
  )
  def test_calibration_with_quantization_multiple_inputs(
      self, activations_type
  ):
    # [FUNCTION DOC]
    # Parameterized by activation type. Loads the multi-add model with four inputs, feeds ten
    # samples of four all-ones arrays each, calibrates and quantizes, and asserts a model is
    # returned. (The parameter names mention a quantizer toggle that this test does not use.)
    # Load multi add model from test data.
    # This model has 4 inputs of size (1, 8, 8, 3).
    model_path = resource_loader.get_path_to_datafile(
        '../../testdata/multi_add.bin'
    )
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator for the model.
    def input_gen():
      # [FUNCTION DOC]
      # Test helper generator: yields ten samples, each a list of four all-ones float32 arrays
      # of shape one by eight by eight by three.
      for _ in range(10):
        yield [np.ones(shape=(1, 8, 8, 3), dtype=np.float32) for _ in range(4)]

    quantized_model = quantizer.calibrate_and_quantize(
        input_gen, dtypes.float32, dtypes.float32, False, activations_type
    )
    self.assertIsNotNone(quantized_model)

  def test_invalid_model_buffer(self):
    # [FUNCTION DOC]
    # Creates a calibrator from one hundred zero bytes; expects a value error whose message says
    # the model failed to parse.
    float_model = b'\0' * 100
    with self.assertRaisesRegex(ValueError, 'Failed to parse the model'):
      _calibrator.Calibrator(float_model)

  # TODO(fengliuai): enable mlir quantizer
  def test_empty_calibrator_gen(self):
    # [FUNCTION DOC]
    # Quantizes with a sample generator that yields nothing; expects a runtime error.
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin'
    )
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    def empty_input_gen():
      # [FUNCTION DOC]
      # Test helper generator that yields no samples.
      for i in ():
        yield i

    with self.assertRaises(RuntimeError):
      quantizer.calibrate_and_quantize(
          empty_input_gen, dtypes.float32, dtypes.float32, False
      )

  def test_invalid_shape_calibrator_gen(self):
    # [FUNCTION DOC]
    # Feeds samples of the wrong shape (two by two instead of five by five) with input resizing
    # turned off; expects a value error mentioning a size mismatch.
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin'
    )
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator with incorrect shape.
    def input_gen():
      # [FUNCTION DOC]
      # Test helper generator: yields ten samples with an all-ones float32 array of the wrong
      # shape one by two by two by three.
      for _ in range(10):
        yield [np.ones(shape=(1, 2, 2, 3), dtype=np.float32)]

    with self.assertRaisesRegex(ValueError, 'Size mismatch'):
      quantizer.calibrate_and_quantize(
          input_gen,
          dtypes.float32,
          dtypes.float32,
          False,
          activations_type=dtypes.int8,
          bias_type=dtypes.int32,
          resize_input=False,
      )

  def test_invalid_type_calibrator_gen(self):
    # [FUNCTION DOC]
    # Feeds samples of the wrong element type (32-bit integer instead of float); expects a value
    # error.
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin'
    )
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator with incorrect type.
    def input_gen():
      # [FUNCTION DOC]
      # Test helper generator: yields ten samples with an all-ones int32 array of the right shape.
      for _ in range(10):
        yield [np.ones(shape=(1, 5, 5, 3), dtype=np.int32)]

    with self.assertRaises(ValueError):
      quantizer.calibrate_and_quantize(
          input_gen, dtypes.float32, dtypes.float32, False, dtypes.int8
      )

  def test_calibration(self):
    # [FUNCTION DOC]
    # Calibrates (without quantizing) the mobilenet-like model with ten ones-filled samples and
    # asserts a model is returned.
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin'
    )
    float_model = open(model_path, 'rb').read()
    quantizer = _calibrator.Calibrator(float_model)

    # Input generator for the model.
    def input_gen():
      # [FUNCTION DOC]
      # Test helper generator: yields ten samples, each a list with one all-ones float32 array
      # of shape one by five by five by three.
      for _ in range(10):
        yield [np.ones(shape=(1, 5, 5, 3), dtype=np.float32)]

    quantized_model = quantizer.calibrate(input_gen)
    self.assertIsNotNone(quantized_model)

  def test_add_intermediate_tensors(self):
    # [FUNCTION DOC]
    # Runs the add-intermediate-tensors function on the mobilenet-like model bytes and asserts a
    # result is returned.
    model_path = resource_loader.get_path_to_datafile(
        'test_data/mobilenet_like_model.bin'
    )
    model = open(model_path, 'rb').read()
    added_model = _calibrator.add_intermediate_tensors(model)
    self.assertIsNotNone(added_model)

  def test_calibrate_model_with_offset_buffer(self):
    # [FUNCTION DOC]
    # Saves a small matrix multiply module, converts it with the offset buffer option on, checks
    # the model uses offset buffers, calibrates with two samples of ones, checks the calibrated
    # model still uses offset buffers, and asserts the recorded min and max are one for the
    # input and filter tensors and four for the output tensor.
    # Define a simple model to run calibration with.
    class MatMulModel(tf.Module):
      # [CLASS DOC]
      # Test helper module: holds a four by three ones filter and multiplies a one by four
      # input by it.

      def __init__(self):
        # [FUNCTION DOC]
        # Creates the all-ones float32 four by three filter used by the call method.
        # Use ones for predictable calibration results.
        self.filter = np.ones((4, 3)).astype(np.float32)

      @tf.function(
          input_signature=[tf.TensorSpec(shape=(1, 4), dtype=dtypes.float32)]
      )
      def __call__(self, input_tensor: tf.Tensor) -> tf.Tensor:
        # [FUNCTION DOC]
        # KNOWN ISSUE: annotated as returning a tensor but returns a mapping with one entry named
        # output.
        # Multiplies the input tensor by the filter and returns a mapping from the name output to
        # the product.
        output_tensor = tf.linalg.matmul(input_tensor, self.filter)
        return {'output': output_tensor}

    model = MatMulModel()
    saved_model_path = self.create_tempdir().full_path
    tf.saved_model.save(model, saved_model_path)

    converter = lite.TFLiteConverter.from_saved_model(saved_model_path)
    # Enable the use of buffer offsets.
    # pylint: disable=protected-access
    converter._experimental_use_buffer_offset = True
    # pylint: enable=protected-access
    converter.exclude_conversion_metadata = True

    model_serialized = converter.convert()

    model = flatbuffer_utils.convert_bytearray_to_object(model_serialized)
    self.assertTrue(_uses_buffer_offset(model))

    quantizer = _calibrator.Calibrator(model_serialized)

    # Input generator for the model.
    def input_gen():
      # [FUNCTION DOC]
      # Test helper generator: yields two samples, each a list with one float32 array of four ones.
      for _ in range(2):
        yield [np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)]

    calibrated_model_serialized = quantizer.calibrate(input_gen)
    self.assertIsNotNone(calibrated_model_serialized)

    calibrated_model = flatbuffer_utils.convert_bytearray_to_object(
        calibrated_model_serialized
    )
    self.assertTrue(_uses_buffer_offset(calibrated_model))

    # Confirm that the tensors are correctly calibrated.
    subgraph = calibrated_model.subgraphs[0]

    matmul_input_tensor = subgraph.tensors[0]
    self.assertAllClose(matmul_input_tensor.quantization.min, [1.0])
    self.assertAllClose(matmul_input_tensor.quantization.max, [1.0])

    matmul_filter_tensor = subgraph.tensors[1]
    self.assertAllClose(matmul_filter_tensor.quantization.min, [1.0])
    self.assertAllClose(matmul_filter_tensor.quantization.max, [1.0])

    # The matmul is performed with all ones so the output is expected to be 4s.
    matmul_output_tensor = subgraph.tensors[2]
    self.assertAllClose(matmul_output_tensor.quantization.min, [4.0])
    self.assertAllClose(matmul_output_tensor.quantization.max, [4.0])


if __name__ == '__main__':
  test.main()
