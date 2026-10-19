# [MODULE DOC]
# Unit tests for the TensorFlow Lite Python interpreter wrapper (the interpreter module).
# Covers custom-op registerers, op resolver types, thread counts, float/uint8/string models,
# quantization and sparsity parameter reporting, error propagation for bad models and indices,
# zero-copy tensor accessors, op/node details, delegate loading and lifetime, and multi-signature
# (multi-subgraph) models. Test models and the test delegate library come from the testdata folder.
# Import-time side effect: on platforms that support it, adds the global-symbol flag to the
# dynamic loader flags so that separately built shared objects (the test registerer) are visible
# to the interpreter wrapper. Running the file as a program starts the TensorFlow test runner.
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
"""TensorFlow Lite Python Interface: Sanity check."""
import ctypes
import io
import pathlib
import sys
from unittest import mock

import numpy as np
import tensorflow as tf

# Force loaded shared object symbols to be globally visible. This is needed so
# that the interpreter_wrapper, in one .so file, can see the test_registerer,
# in a different .so file. Note that this may already be set by default.
# pylint: disable=g-import-not-at-top
if hasattr(sys, 'setdlopenflags') and hasattr(sys, 'getdlopenflags'):
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.lite.python import interpreter as interpreter_wrapper
from tensorflow.lite.python import lite
from tensorflow.lite.python.metrics import metrics
from tensorflow.lite.python.testdata import _pywrap_test_registerer as test_registerer
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
# pylint: enable=g-import-not-at-top


class InterpreterCustomOpsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the interpreter variant that accepts custom op registerers, given either by symbol
  # name or as a function. Uses the permute float test model and a test registerer extension
  # module that counts how many times it was called. No shared fixtures.

  def testRegistererByName(self):
    # [FUNCTION DOC]
    # Scenario: build the custom-ops interpreter with the registerer given by its symbol name.
    # Expects the interpreter to report it is safe to run and the registerer call count to be one.
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'),
        custom_op_registerers=['TF_TestRegisterer'])
    self.assertTrue(interpreter._safe_to_run())
    self.assertEqual(test_registerer.get_num_test_registerer_calls(), 1)

  def testRegistererByFunc(self):
    # [FUNCTION DOC]
    # Scenario: build the custom-ops interpreter with the registerer given as a function object.
    # Expects the interpreter to be safe to run and the registerer call count to be one.
    # KNOWN ISSUE: the expected count of one assumes the by-name test did not already count a
    # call in the same process; the counter is not reset here.
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'),
        custom_op_registerers=[test_registerer.TF_TestRegisterer])
    self.assertTrue(interpreter._safe_to_run())
    self.assertEqual(test_registerer.get_num_test_registerer_calls(), 1)

  def testRegistererFailure(self):
    # [FUNCTION DOC]
    # Scenario: pass a registerer name that does not exist. Expects a value error whose message
    # says looking up the symbol with that name failed.
    bogus_name = 'CompletelyBogusRegistererName'
    with self.assertRaisesRegex(
        ValueError, 'Looking up symbol \'' + bogus_name + '\' failed'):
      interpreter_wrapper.InterpreterWithCustomOps(
          model_path=resource_loader.get_path_to_datafile(
              'testdata/permute_float.tflite'),
          custom_op_registerers=[bogus_name])

  # Register GenAI Ops is only supported when using LiteRT wheel.
  def testRegisterGenAIOpsFailure(self):
    # [FUNCTION DOC]
    # Scenario: request the GenAI ops registerer by name, which needs a shared library that is
    # only shipped with the LiteRT wheel. Expects a value error saying loading that shared
    # library failed because the file cannot be opened (not found).
    genai_ops_name = 'pywrap_genai_ops.GenAIOpsRegisterer'
    with self.assertRaisesRegex(
        ValueError,
        "Loading library 'pywrap_genai_ops.so' failed with error"
        " 'pywrap_genai_ops.so: cannot open shared object file: No such file or"
        " directory'",
    ):
      interpreter_wrapper.InterpreterWithCustomOps(
          model_path=resource_loader.get_path_to_datafile(
              'testdata/permute_float.tflite'
          ),
          custom_op_registerers=[genai_ops_name],
      )

  def testNoCustomOps(self):
    # [FUNCTION DOC]
    # Scenario: build the custom-ops interpreter without any registerers. Expects it to be safe
    # to run.
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    self.assertTrue(interpreter._safe_to_run())


class InterpreterTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Main tests for the interpreter: model loading from path, path-like object or bytes, thread
  # options, op resolver types, tensor and quantization details for float, uint8 and string
  # models, per-channel and sparse tensors, and the creation metrics counter. No shared fixtures.

  def assertQuantizationParamsEqual(self, scales, zero_points,
                                    quantized_dimension, params):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the quantization checks in the tests of this class.
    # Inputs: expected scales, expected zero points, expected quantized dimension (integer), and
    # params, a mapping holding the actual scales, zero points and quantized dimension.
    # Asserts element-wise equality of scales and zero points and equality of the dimension.
    # Returns nothing; fails the test on mismatch.
    self.assertAllEqual(scales, params['scales'])
    self.assertAllEqual(zero_points, params['zero_points'])
    self.assertEqual(quantized_dimension, params['quantized_dimension'])

  def testPathLikeModel(self):
    # [FUNCTION DOC]
    # Scenario: create the interpreter from a path-like object instead of a string and allocate
    # tensors. Expects no error.
    interpreter = interpreter_wrapper.Interpreter(
        model_path=pathlib.Path(
            resource_loader.get_path_to_datafile(
                'testdata/permute_float.tflite'
            )
        ),
    )
    interpreter.allocate_tensors()

  def testThreads_NegativeValue(self):
    # [FUNCTION DOC]
    # Scenario: create the interpreter with a thread count of minus one. Expects a value error
    # saying the thread count must be at least one.
    with self.assertRaisesRegex(ValueError, 'num_threads should >= 1'):
      interpreter_wrapper.Interpreter(
          model_path=resource_loader.get_path_to_datafile(
              'testdata/permute_float.tflite'),
          num_threads=-1)

  def testThreads_WrongType(self):
    # [FUNCTION DOC]
    # Scenario: create the interpreter with a floating-point thread count. Expects a value error
    # saying the thread count must be an integer.
    with self.assertRaisesRegex(ValueError,
                                'type of num_threads should be int'):
      interpreter_wrapper.Interpreter(
          model_path=resource_loader.get_path_to_datafile(
              'testdata/permute_float.tflite'),
          num_threads=4.2)

  def testNotSupportedOpResolverTypes(self):
    # [FUNCTION DOC]
    # Scenario: pass an unknown string as the experimental op resolver type. Expects a value
    # error saying the op resolver type is not recognized.
    with self.assertRaisesRegex(
        ValueError, 'Unrecognized passed in op resolver type: test'):
      interpreter_wrapper.Interpreter(
          model_path=resource_loader.get_path_to_datafile(
              'testdata/permute_float.tflite'),
          experimental_op_resolver_type='test')

  def testFloatWithDifferentOpResolverTypes(self):
    # [FUNCTION DOC]
    # Scenario: for each of three op resolver types (builtin, builtin reference kernels, builtin
    # without default delegates), load the float permute model and allocate tensors.
    # Asserts the single input and output details (name, float32 type, shape one by four, no
    # quantization, empty quantization parameters), then feeds one to four and expects the
    # reversed values four to one after invoking.
    op_resolver_types = [
        interpreter_wrapper.OpResolverType.BUILTIN,
        interpreter_wrapper.OpResolverType.BUILTIN_REF,
        interpreter_wrapper.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    ]

    for op_resolver_type in op_resolver_types:
      interpreter = interpreter_wrapper.Interpreter(
          model_path=resource_loader.get_path_to_datafile(
              'testdata/permute_float.tflite'),
          experimental_op_resolver_type=op_resolver_type)
      interpreter.allocate_tensors()

      input_details = interpreter.get_input_details()
      self.assertEqual(1, len(input_details))
      self.assertEqual('input', input_details[0]['name'])
      self.assertEqual(np.float32, input_details[0]['dtype'])
      self.assertTrue(([1, 4] == input_details[0]['shape']).all())
      self.assertEqual((0.0, 0), input_details[0]['quantization'])
      self.assertQuantizationParamsEqual(
          [], [], 0, input_details[0]['quantization_parameters'])

      output_details = interpreter.get_output_details()
      self.assertEqual(1, len(output_details))
      self.assertEqual('output', output_details[0]['name'])
      self.assertEqual(np.float32, output_details[0]['dtype'])
      self.assertTrue(([1, 4] == output_details[0]['shape']).all())
      self.assertEqual((0.0, 0), output_details[0]['quantization'])
      self.assertQuantizationParamsEqual(
          [], [], 0, output_details[0]['quantization_parameters'])

      test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
      expected_output = np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
      interpreter.set_tensor(input_details[0]['index'], test_input)
      interpreter.invoke()

      output_data = interpreter.get_tensor(output_details[0]['index'])
      self.assertTrue((expected_output == output_data).all())

  def testFloatWithTwoThreads(self):
    # [FUNCTION DOC]
    # Scenario: run the float permute model with two threads. Sets the input to one through four,
    # invokes, and expects the output to be the reversed values.
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'),
        num_threads=2)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    expected_output = np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testUint8(self):
    # [FUNCTION DOC]
    # Scenario: read the uint8 permute model file into bytes and create the interpreter from the
    # content. Asserts input and output details (name, uint8 type, shape one by four,
    # quantization scale one with zero point zero). Then resizes the input to its own shape,
    # reallocates, feeds one to four, invokes, and expects the reversed values.
    model_path = resource_loader.get_path_to_datafile(
        'testdata/permute_uint8.tflite')
    with io.open(model_path, 'rb') as model_file:
      data = model_file.read()

    interpreter = interpreter_wrapper.Interpreter(model_content=data)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.uint8, input_details[0]['dtype'])
    self.assertTrue(([1, 4] == input_details[0]['shape']).all())
    self.assertEqual((1.0, 0), input_details[0]['quantization'])
    self.assertQuantizationParamsEqual(
        [1.0], [0], 0, input_details[0]['quantization_parameters'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.uint8, output_details[0]['dtype'])
    self.assertTrue(([1, 4] == output_details[0]['shape']).all())
    self.assertEqual((1.0, 0), output_details[0]['quantization'])
    self.assertQuantizationParamsEqual(
        [1.0], [0], 0, output_details[0]['quantization_parameters'])

    test_input = np.array([[1, 2, 3, 4]], dtype=np.uint8)
    expected_output = np.array([[4, 3, 2, 1]], dtype=np.uint8)
    interpreter.resize_tensor_input(input_details[0]['index'], test_input.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testString(self):
    # [FUNCTION DOC]
    # Scenario: load the string gather model (inputs: ten strings and three int64 indices;
    # output: three strings). Asserts names, types, shapes and absent quantization for all
    # tensors. Feeds the letters a through j and indices one, two, three, invokes, and expects
    # the byte strings b, c, d.
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/gather_string.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual('input', input_details[0]['name'])
    self.assertEqual(np.bytes_, input_details[0]['dtype'])
    self.assertTrue(([10] == input_details[0]['shape']).all())
    self.assertEqual((0.0, 0), input_details[0]['quantization'])
    self.assertQuantizationParamsEqual(
        [], [], 0, input_details[0]['quantization_parameters'])
    self.assertEqual('indices', input_details[1]['name'])
    self.assertEqual(np.int64, input_details[1]['dtype'])
    self.assertTrue(([3] == input_details[1]['shape']).all())
    self.assertEqual((0.0, 0), input_details[1]['quantization'])
    self.assertQuantizationParamsEqual(
        [], [], 0, input_details[1]['quantization_parameters'])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual('output', output_details[0]['name'])
    self.assertEqual(np.bytes_, output_details[0]['dtype'])
    self.assertTrue(([3] == output_details[0]['shape']).all())
    self.assertEqual((0.0, 0), output_details[0]['quantization'])
    self.assertQuantizationParamsEqual(
        [], [], 0, output_details[0]['quantization_parameters'])

    test_input = np.array([1, 2, 3], dtype=np.int64)
    interpreter.set_tensor(input_details[1]['index'], test_input)

    test_input = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    expected_output = np.array([b'b', b'c', b'd'])
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    self.assertTrue((expected_output == output_data).all())

  def testStringZeroDim(self):
    # [FUNCTION DOC]
    # Scenario: load a model with a scalar (zero-dimensional) string input. Sets it to a byte
    # string containing text followed by sixteen zero bytes, reads it back, and expects the
    # stored item to have the same length as the original (embedded zero bytes preserved).
    data = b'abcd' + bytes(16)
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/gather_string_0d.tflite'))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], np.array(data))
    test_input_tensor = interpreter.get_tensor(input_details[0]['index'])
    self.assertEqual(len(data), len(test_input_tensor.item(0)))

  def testPerChannelParams(self):
    # [FUNCTION DOC]
    # Scenario: load the per-channel convolution model and read the details of tensor one (the
    # weights). Expects the quantization parameters to hold 128 per-channel scales.
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile('testdata/pc_conv.bin'))
    interpreter.allocate_tensors()

    # Tensor index 1 is the weight.
    weight_details = interpreter.get_tensor_details()[1]
    qparams = weight_details['quantization_parameters']
    # Ensure that we retrieve per channel quantization params correctly.
    self.assertEqual(len(qparams['scales']), 128)

  def testDenseTensorAccess(self):
    # [FUNCTION DOC]
    # Scenario: load the per-channel convolution model and read the weight tensor details.
    # Expects the sparsity parameters of a dense tensor to be an empty mapping.
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile('testdata/pc_conv.bin'))
    interpreter.allocate_tensors()
    weight_details = interpreter.get_tensor_details()[1]
    s_params = weight_details['sparsity_parameters']
    self.assertEqual(s_params, {})

  def testSparseTensorAccess(self):
    # [FUNCTION DOC]
    # Scenario: load the sparse tensor model with the test registerer and read tensor zero.
    # Expects the compressed buffer to have twelve values matching the known data, and the
    # sparsity parameters to have traversal order zero to three, block map zero and one, dense
    # dimensions of size two at positions zero, two and three, and a sparse dimension at position
    # one with the expected segment and index arrays.
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_path=resource_loader.get_path_to_datafile(
            '../testdata/sparse_tensor.bin'),
        custom_op_registerers=['TF_TestRegisterer'])
    interpreter.allocate_tensors()

    # Tensor at index 0 is sparse.
    compressed_buffer = interpreter.get_tensor(0)
    # Ensure that the buffer is of correct size and value.
    self.assertEqual(len(compressed_buffer), 12)
    sparse_value = [1, 0, 0, 4, 2, 3, 0, 0, 5, 0, 0, 6]
    self.assertAllEqual(compressed_buffer, sparse_value)

    tensor_details = interpreter.get_tensor_details()[0]
    s_params = tensor_details['sparsity_parameters']

    # Ensure sparsity parameter returned is correct
    self.assertAllEqual(s_params['traversal_order'], [0, 1, 2, 3])
    self.assertAllEqual(s_params['block_map'], [0, 1])
    dense_dim_metadata = {'format': 0, 'dense_size': 2}
    self.assertAllEqual(s_params['dim_metadata'][0], dense_dim_metadata)
    self.assertAllEqual(s_params['dim_metadata'][2], dense_dim_metadata)
    self.assertAllEqual(s_params['dim_metadata'][3], dense_dim_metadata)
    self.assertEqual(s_params['dim_metadata'][1]['format'], 1)
    self.assertAllEqual(s_params['dim_metadata'][1]['array_segments'],
                        [0, 2, 3])
    self.assertAllEqual(s_params['dim_metadata'][1]['array_indices'], [0, 1, 1])

  @mock.patch.object(metrics.TFLiteMetrics,
                     'increase_counter_interpreter_creation')
  def testCreationCounter(self, increase_call):
    # [FUNCTION DOC]
    # Scenario: with the metrics method that increments the interpreter-creation counter replaced
    # by a mock (passed in as increase_call), create an interpreter. Expects exactly one call.
    interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    increase_call.assert_called_once()


class InterpreterTestErrorPropagation(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that errors from the native layer surface as the right Python exceptions: invalid
  # model content or file, missing arguments, invoking too early, invalid indices, and that empty
  # input tensors work. No shared fixtures.

  # Model must have at least 7 bytes to hold model identifier
  def testTooShortModelContent(self):
    # [FUNCTION DOC]
    # Scenario: create an interpreter from a five-byte buffer. Expects a value error saying the
    # model is not a valid flat buffer.
    with self.assertRaisesRegex(ValueError,
                                'The model is not a valid Flatbuffer buffer'):
      interpreter_wrapper.Interpreter(model_content=b'short')

  def testInvalidModelContent(self):
    # [FUNCTION DOC]
    # Scenario: create an interpreter from a long buffer with a wrong identifier. Expects a value
    # error saying the model is not a valid flat buffer.
    with self.assertRaisesRegex(ValueError,
                                'The model is not a valid Flatbuffer buffer'):
      interpreter_wrapper.Interpreter(model_content=b'wrong_identifier')

  def testInvalidModelFile(self):
    # [FUNCTION DOC]
    # Scenario: create an interpreter from a nonexistent file path. Expects a value error saying
    # the file could not be opened.
    with self.assertRaisesRegex(ValueError,
                                'Could not open \'totally_invalid_file_name\''):
      interpreter_wrapper.Interpreter(model_path='totally_invalid_file_name')

  def testInvokeBeforeReady(self):
    # [FUNCTION DOC]
    # Scenario: invoke an interpreter before tensors are allocated. Expects a runtime error saying
    # invoke was called on a model that is not ready.
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    with self.assertRaisesRegex(RuntimeError,
                                'Invoke called on model that is not ready'):
      interpreter.invoke()

  def testInvalidModelFileContent(self):
    # [FUNCTION DOC]
    # Scenario: create an interpreter with neither a model path nor model content. Expects a value
    # error saying one of them must be specified.
    with self.assertRaisesRegex(
        ValueError, '`model_path` or `model_content` must be specified.'):
      interpreter_wrapper.Interpreter(model_path=None, model_content=None)

  def testInvalidIndex(self):
    # [FUNCTION DOC]
    # Scenario: with an allocated interpreter, ask the private tensor-details method for tensor
    # index four and the private op-details method for node index four. Expects value errors
    # about the tensor index exceeding the maximum of three and about an invalid node index.
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    interpreter.allocate_tensors()
    # Invalid tensor index passed.
    with self.assertRaisesRegex(
        ValueError, 'Invalid tensor index 4 exceeds max tensor index 3'
    ):
      interpreter._get_tensor_details(4, 0)
    with self.assertRaisesRegex(ValueError, 'Invalid node index'):
      interpreter._get_op_details(4)

  def testEmptyInputTensor(self):
    # [FUNCTION DOC]
    # Scenario: define a small model whose function sums a float vector of unknown length, convert
    # it to a lite model, and run it through the signature runner with an empty vector.
    # Expects no error (there is no explicit assertion on the result).

    class TestModel(tf.keras.models.Model):
      # [CLASS DOC]
      # Local keras model with one traced function that sums a one-dimensional float vector of
      # unknown length. Used only to produce a lite model for the empty-input test.

      @tf.function(
          input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
      def TestSum(self, x):
        # [FUNCTION DOC]
        # Returns the sum of input x (float vector of unknown length) over axis zero as a scalar.
        return tf.raw_ops.Sum(input=x, axis=[0])

    test_model = TestModel()
    converter = lite.TFLiteConverterV2.from_concrete_functions([
        test_model.TestSum.get_concrete_function(
            tf.TensorSpec([None], tf.float32))
    ], test_model)
    model = converter.convert()
    interpreter = lite.Interpreter(model_content=model)
    # Make sure that passing empty tensor doesn't cause any errors.
    interpreter.get_signature_runner()(x=tf.zeros([0], tf.float32))


class InterpreterTensorAccessorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the zero-copy tensor accessor versus the copying getter, and the safety checks that
  # block allocation or invocation while accessor buffers are alive. The fixture is an allocated
  # float permute interpreter, the index of its first input, and sample initial data.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates and allocates an interpreter on the float permute model, stores the first input
    # tensor index and a one by four float array of minus one to minus four. Returns nothing.
    super(InterpreterTensorAccessorTest, self).setUp()
    self.interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'))
    self.interpreter.allocate_tensors()
    self.input0 = self.interpreter.get_input_details()[0]['index']
    self.initial_data = np.array([[-1., -2., -3., -4.]], np.float32)

  def testTensorAccessor(self):
    # [FUNCTION DOC]
    # Scenario: get the accessor for the input tensor, copy data into the live buffer, and expect
    # both the buffer and a normal read of the tensor to show the data (it is a reference).
    """Check that tensor returns a reference."""
    array_ref = self.interpreter.tensor(self.input0)
    np.copyto(array_ref(), self.initial_data)
    self.assertAllEqual(array_ref(), self.initial_data)
    self.assertAllEqual(
        self.interpreter.get_tensor(self.input0), self.initial_data)

  def testGetTensorAccessor(self):
    # [FUNCTION DOC]
    # Scenario: set the input, read it with the getter, then set a changed value. Expects the
    # earlier read to keep the original data (a copy) while a new read shows the new value.
    """Check that get_tensor returns a copy."""
    self.interpreter.set_tensor(self.input0, self.initial_data)
    array_initial_copy = self.interpreter.get_tensor(self.input0)
    new_value = np.add(1., array_initial_copy)
    self.interpreter.set_tensor(self.input0, new_value)
    self.assertAllEqual(array_initial_copy, self.initial_data)
    self.assertAllEqual(self.interpreter.get_tensor(self.input0), new_value)

  def testBase(self):
    # [FUNCTION DOC]
    # Scenario: track the interpreter's safe-to-run status. It is safe before any buffer exists,
    # still safe after only creating the accessor, unsafe once a buffer is obtained from it
    # (also with two buffers), and safe again after both buffers are deleted.
    self.assertTrue(self.interpreter._safe_to_run())
    _ = self.interpreter.tensor(self.input0)
    self.assertTrue(self.interpreter._safe_to_run())
    in0 = self.interpreter.tensor(self.input0)()
    self.assertFalse(self.interpreter._safe_to_run())
    in0b = self.interpreter.tensor(self.input0)()
    self.assertFalse(self.interpreter._safe_to_run())
    # Now get rid of the buffers so that we can evaluate.
    del in0
    del in0b
    self.assertTrue(self.interpreter._safe_to_run())

  def testBaseProtectsFunctions(self):
    # [FUNCTION DOC]
    # Scenario: hold a live tensor buffer, then try to allocate tensors and to invoke. Expects
    # runtime errors saying at least one reference exists. After deleting the buffer, allocation
    # succeeds again.
    in0 = self.interpreter.tensor(self.input0)()
    # Make sure we get an exception if we try to run an unsafe operation
    with self.assertRaisesRegex(RuntimeError, 'There is at least 1 reference'):
      _ = self.interpreter.allocate_tensors()
    # Make sure we get an exception if we try to run an unsafe operation
    with self.assertRaisesRegex(RuntimeError, 'There is at least 1 reference'):
      _ = self.interpreter.invoke()  # pylint: disable=assignment-from-no-return
    # Now test that we can run
    del in0  # this is our only buffer reference, so now it is safe to change
    in0safe = self.interpreter.tensor(self.input0)
    _ = self.interpreter.allocate_tensors()
    del in0safe  # make sure in0Safe is held but lint doesn't complain


class InterpreterNodeAccessTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the private op (node) detail accessors of the interpreter. Fixture: an allocated float
  # permute interpreter plus the first input index and sample data.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates and allocates an interpreter on the float permute model and stores the first input
    # tensor index and a one by four float array. Returns nothing.
    super().setUp()
    self.interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'
        )
    )
    self.interpreter.allocate_tensors()
    self.input0 = self.interpreter.get_input_details()[0]['index']
    self.initial_data = np.array([[-1.0, -2.0, -3.0, -4.0]], np.float32)

  def testValidNode(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it checks that a tensor returns a reference, but the test
    # actually checks op details (copied text from the tensor accessor test).
    # Scenario: read all op details. Expects the first op to have index zero, name fully
    # connected, inputs zero, one and minus one, output two, two float32 operand types and one
    # float32 result type.
    """Check that tensor returns a reference."""
    ops_details = self.interpreter._get_ops_details()
    self.assertEqual(ops_details[0]['index'], 0)
    self.assertEqual(ops_details[0]['op_name'], 'FULLY_CONNECTED')
    self.assertAllEqual(ops_details[0]['inputs'], [0, 1, -1])
    self.assertAllEqual(ops_details[0]['outputs'], [2])
    self.assertAllEqual(
        ops_details[0]['operand_types'], [np.float32, np.float32]
    )
    self.assertAllEqual(ops_details[0]['result_types'], [np.float32])

  def testInvalidNode(self):
    # [FUNCTION DOC]
    # Scenario: ask for the op details of node index four on a one-op model. Expects a value
    # error about an invalid node index.
    with self.assertRaisesRegex(ValueError, 'Invalid node index'):
      self.interpreter._get_op_details(4)


class InterpreterDelegateTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests loading a test delegate shared library and its lifetime relative to interpreters.
  # The library exposes counters (delegates created, destroyed, invoked, options counter).
  # Fixture: paths of the delegate library and the float permute model.

  def setUp(self):
    # [FUNCTION DOC]
    # Stores the delegate library and model file paths, then loads the library and calls its
    # initialize-counters function to reset all counters. Returns nothing.
    super(InterpreterDelegateTest, self).setUp()
    self._delegate_file = resource_loader.get_path_to_datafile(
        'testdata/test_delegate.so')
    self._model_file = resource_loader.get_path_to_datafile(
        'testdata/permute_float.tflite')

    # Load the library to reset the counters.
    library = ctypes.pydll.LoadLibrary(self._delegate_file)
    library.initialize_counters()

  def _TestInterpreter(self, model_path, options=None):
    # [FUNCTION DOC]
    # HELPER: builds an interpreter with the test delegate for the delegate tests.
    # Inputs: model_path (string path to a model) and optional options (mapping of delegate
    # option names to values, default none). Loads the test delegate with those options and
    # returns an interpreter on the model using it. Propagates errors from delegate loading.
    """Test wrapper function that creates an interpreter with the delegate."""
    delegate = interpreter_wrapper.load_delegate(self._delegate_file, options)
    return interpreter_wrapper.Interpreter(
        model_path=model_path, experimental_delegates=[delegate])

  def testDelegate(self):
    # [FUNCTION DOC]
    # Scenario: create an interpreter with the delegate. Expects counts created one, destroyed
    # zero, invoked one; after deleting the interpreter, destroyed becomes one.
    """Tests the delegate creation and destruction."""
    interpreter = self._TestInterpreter(model_path=self._model_file)
    lib = interpreter._delegates[0]._library

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 1)

    del interpreter

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 1)
    self.assertEqual(lib.get_num_delegates_invoked(), 1)

  def testMultipleInterpreters(self):
    # [FUNCTION DOC]
    # Scenario: share one delegate between two interpreters. Checks counters after each step:
    # invoked rises to one then two while created stays one. Deleting the delegate handle and the
    # first interpreter does not destroy it; deleting the second interpreter does (destroyed one).
    delegate = interpreter_wrapper.load_delegate(self._delegate_file)
    lib = delegate._library

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)

    interpreter_a = interpreter_wrapper.Interpreter(
        model_path=self._model_file, experimental_delegates=[delegate])

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 1)

    interpreter_b = interpreter_wrapper.Interpreter(
        model_path=self._model_file, experimental_delegates=[delegate])

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 2)

    del delegate
    del interpreter_a

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 2)

    del interpreter_b

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 1)
    self.assertEqual(lib.get_num_delegates_invoked(), 2)

  def testDestructionOrder(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is disabled; it skips immediately (flakiness bug reference), so the
    # rest of the body is dead code and never runs.
    # Intended scenario: record destruction order via a callback and expect the interpreter to be
    # destroyed before the delegate.
    """Make sure internal _interpreter object is destroyed before delegate."""
    self.skipTest('TODO(b/142136355): fix flakiness and re-enable')
    # Track which order destructions were doned in
    destructions = []

    def register_destruction(x):
      # [FUNCTION DOC]
      # Callback given to the delegate library. Input x is a string or bytes name; it is decoded
      # to text if needed and appended to the destruction list of the enclosing test. Returns
      # integer zero.
      destructions.append(x if isinstance(x, str) else x.decode('utf-8'))
      return 0

    # Make a wrapper for the callback so we can send this to ctypes
    delegate = interpreter_wrapper.load_delegate(self._delegate_file)
    # Make an interpreter with the delegate
    interpreter = interpreter_wrapper.Interpreter(
        model_path=resource_loader.get_path_to_datafile(
            'testdata/permute_float.tflite'),
        experimental_delegates=[delegate])

    class InterpreterDestroyCallback:
      # [CLASS DOC]
      # Marker object attached to the internal interpreter; when it is destroyed it records the
      # word interpreter in the destruction list. Part of the disabled (dead) test.

      def __del__(self):
        # [FUNCTION DOC]
        # Finalizer: records that the interpreter was destroyed. Returns nothing.
        register_destruction('interpreter')

    interpreter._interpreter.stuff = InterpreterDestroyCallback()
    # Destroy both delegate and interpreter
    library = delegate._library
    prototype = ctypes.CFUNCTYPE(ctypes.c_int, (ctypes.c_char_p))
    library.set_destroy_callback(prototype(register_destruction))
    del delegate
    del interpreter
    library.set_destroy_callback(None)
    # check the interpreter was destroyed before the delegate
    self.assertEqual(destructions, ['interpreter', 'test_delegate'])

  def testOptions(self):
    # [FUNCTION DOC]
    # Scenario: load the delegate twice, the second time with options (an unused flag and an
    # options counter of two). Expects the options counter to be zero then two, created count two,
    # and after deleting both, destroyed count two and nothing invoked.
    delegate_a = interpreter_wrapper.load_delegate(self._delegate_file)
    lib = delegate_a._library

    self.assertEqual(lib.get_num_delegates_created(), 1)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)
    self.assertEqual(lib.get_options_counter(), 0)

    delegate_b = interpreter_wrapper.load_delegate(
        self._delegate_file, options={
            'unused': False,
            'options_counter': 2
        })
    lib = delegate_b._library

    self.assertEqual(lib.get_num_delegates_created(), 2)
    self.assertEqual(lib.get_num_delegates_destroyed(), 0)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)
    self.assertEqual(lib.get_options_counter(), 2)

    del delegate_a
    del delegate_b

    self.assertEqual(lib.get_num_delegates_created(), 2)
    self.assertEqual(lib.get_num_delegates_destroyed(), 2)
    self.assertEqual(lib.get_num_delegates_invoked(), 0)
    self.assertEqual(lib.get_options_counter(), 2)

  def testFail(self):
    # [FUNCTION DOC]
    # Scenario: load the delegate with an option that makes it fail. Expects a value error saying
    # loading the delegate failed.
    with self.assertRaisesRegex(
        # Due to exception chaining in PY3, we can't be more specific here and
        # check that the phrase 'Fail argument sent' is present.
        ValueError, 'Failed to load delegate from'):
      interpreter_wrapper.load_delegate(
          self._delegate_file, options={'fail': 'fail'})


class InterpreterMultiSignatureTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests subgraph counts and per-subgraph tensor details for a single-signature model and a
  # two-signature model. Fixture: paths of both model files.

  def setUp(self):
    # [FUNCTION DOC]
    # Stores the paths of the single-signature (float permute) and two-signature test models.
    # Returns nothing.
    super(InterpreterMultiSignatureTest, self).setUp()
    self._single_signature_file = resource_loader.get_path_to_datafile(
        'testdata/permute_float.tflite'
    )
    self._double_signature_file = resource_loader.get_path_to_datafile(
        'testdata/two_signatures.tflite'
    )

  def testNumSubgraphsSingleSignature(self):
    # [FUNCTION DOC]
    # Expects the single-signature model to report exactly one subgraph.
    single_signature_interpreter = interpreter_wrapper.Interpreter(
        model_path=self._single_signature_file
    )
    self.assertEqual(single_signature_interpreter.num_subgraphs(), 1)

  def testNumSubgraphsDoubleSignature(self):
    # [FUNCTION DOC]
    # Expects the two-signature model to report exactly two subgraphs.
    double_signature_interpreter = interpreter_wrapper.Interpreter(
        model_path=self._double_signature_file
    )
    self.assertEqual(double_signature_interpreter.num_subgraphs(), 2)

  def testGetTensorDetailsSingleSignature(self):
    # [FUNCTION DOC]
    # Expects three tensor details for the single-signature model with the first named input.
    # Requesting subgraph index one or minus one must raise a value error saying the subgraph
    # index is out of range.
    single_signature_interpreter = interpreter_wrapper.Interpreter(
        model_path=self._single_signature_file
    )
    tensor_details = single_signature_interpreter.get_tensor_details()
    self.assertLen(tensor_details, 3)
    self.assertEqual(tensor_details[0]['name'], 'input')

    with self.assertRaisesRegex(ValueError, 'subgraph_index is out of range'):
      single_signature_interpreter.get_tensor_details(subgraph_index=1)

    with self.assertRaisesRegex(ValueError, 'subgraph_index is out of range'):
      single_signature_interpreter.get_tensor_details(subgraph_index=-1)

  def testGetTensorDetailsDoubleSignature(self):
    # [FUNCTION DOC]
    # Expects three tensor details per subgraph of the two-signature model, with first tensor
    # names add_x:0 for subgraph zero and multiply_x:0 for subgraph one. Subgraph index three must
    # raise a value error saying the index is out of range.
    double_signature_interpreter = interpreter_wrapper.Interpreter(
        model_path=self._double_signature_file
    )
    subgraph0_tensor_details = double_signature_interpreter.get_tensor_details(
        subgraph_index=0
    )
    self.assertLen(subgraph0_tensor_details, 3)
    self.assertEqual(subgraph0_tensor_details[0]['name'], 'add_x:0')

    subgraph1_tensor_details = double_signature_interpreter.get_tensor_details(
        subgraph_index=1
    )
    self.assertLen(subgraph1_tensor_details, 3)
    self.assertEqual(subgraph1_tensor_details[0]['name'], 'multiply_x:0')

    with self.assertRaisesRegex(ValueError, 'subgraph_index is out of range'):
      double_signature_interpreter.get_tensor_details(subgraph_index=3)


if __name__ == '__main__':
  test.main()
