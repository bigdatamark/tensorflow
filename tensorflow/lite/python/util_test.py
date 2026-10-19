# [MODULE DOC]
# Tests for the TensorFlow Lite converter utility module (util.py).
# Covers enum-to-dtype conversion, tensor name and shape helpers, uint32 pass-through conversion,
# lower-using-switch-merge removal, C source generation from bytes, and modification of the
# input/output types of integer quantized models (including signature definitions and a concat
# residual model). Contains module-level helpers that build and quantize small Keras models.
# No import-time side effects; runs the test main when executed directly.
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
"""Tests for util.py."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.lite.python import lite
from tensorflow.lite.python import util
from tensorflow.lite.tools.flatbuffer_utils import read_model as _read_model
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


# TODO(nupurgarg): Add test for Grappler and frozen graph related functions.
class UtilTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # General tests for util.py: type code conversion, tensor naming, a uint32 conversion
  # round trip, switch-merge lowering removal and C source generation. No shared fixtures.

  def testConvertEnumToDtype(self):
    # [FUNCTION DOC]
    # Checks every supported schema type code (0 to 11 and 16) maps to the expected TensorFlow
    # dtype, and that an unsupported code 20 raises a value error with the exact message listing
    # the valid mapping.
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(0), dtypes.float32)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(1), dtypes.float16)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(2), dtypes.int32)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(3), dtypes.uint8)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(4), dtypes.int64)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(5), dtypes.string)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(6), dtypes.bool)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(7), dtypes.int16)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(8), dtypes.complex64)
    self.assertEqual(util._convert_tflite_enum_type_to_tf_type(9), dtypes.int8)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(10), dtypes.float64)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(11), dtypes.complex128)
    self.assertEqual(
        util._convert_tflite_enum_type_to_tf_type(16), dtypes.uint32)
    with self.assertRaises(ValueError) as error:
      util._convert_tflite_enum_type_to_tf_type(20)
    self.assertEqual(
        "Unsupported enum 20. The valid map of enum to tf types is : "
        "{0: tf.float32, 1: tf.float16, 2: tf.int32, 3: tf.uint8, 4: tf.int64, "
        "5: tf.string, 6: tf.bool, 7: tf.int16, 8: tf.complex64, 9: tf.int8, "
        "10: tf.float64, 11: tf.complex128, 16: tf.uint32}",
        str(error.exception))

  def testTensorName(self):
    # [FUNCTION DOC]
    # Splits a four element placeholder into four outputs and checks the tensor name helper drops
    # the ":0" suffix for the first and keeps ":1" to ":3" for the rest.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[4])
      out_tensors = array_ops.split(
          value=in_tensor, num_or_size_splits=[1, 1, 1, 1], axis=0)

    expect_names = ["split", "split:1", "split:2", "split:3"]
    for i in range(len(expect_names)):
      got_name = util.get_tensor_name(out_tensors[i])
      self.assertEqual(got_name, expect_names[i])

  def testUint32PassThrough(self):
    # [FUNCTION DOC]
    # Builds a Keras model that reshapes a uint32 vector of 4 into 2 by 2, converts it with the V2
    # converter and loads it in the interpreter. Expects input and output dtype uint32, and that
    # running with the maximum uint32 values returns the reshaped values unchanged.
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,), dtype=tf.uint32),
        tf.keras.layers.Reshape(target_shape=(2, 2))
    ])
    converter = lite.TFLiteConverterV2.from_keras_model(model)
    tflite_model = converter.convert()
    interpreter = lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    self.assertEqual(input_details["dtype"], np.uint32)
    self.assertEqual(output_details["dtype"], np.uint32)

    in_array = np.array([[1, 1, 1, 1]], dtype="uint32") * ((1 << 32) - 1)
    expected_out = np.reshape(in_array, (2, 2))

    interpreter.set_tensor(input_details["index"], in_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details["index"])[0]
    self.assertAllEqual(expected_out, output_data)

  @test_util.enable_control_flow_v2
  def testRemoveLowerUsingSwitchMerge(self):
    # [FUNCTION DOC]
    # With control flow v2, builds a while loop (counter less than 10, add 1) in a graph, then
    # applies the disable-lower-using-switch-merge conversion to its graph def. Expects at least
    # one While/StatelessWhile node whose lower-using-switch-merge attribute is false.
    with ops.Graph().as_default():
      i = array_ops.placeholder(dtype=dtypes.int32, shape=())
      c = lambda i: math_ops.less(i, 10)
      b = lambda i: math_ops.add(i, 1)
      while_loop.while_loop(c, b, [i])
      sess = session.Session()

    new_graph_def = convert_to_constants.disable_lower_using_switch_merge(
        sess.graph_def)
    lower_using_switch_merge_is_removed = False
    for node in new_graph_def.node:
      if node.op == "While" or node.op == "StatelessWhile":
        if not node.attr["_lower_using_switch_merge"].b:
          lower_using_switch_merge_is_removed = True
    self.assertTrue(lower_using_switch_merge_is_removed)

  def testConvertBytes(self):
    # [FUNCTION DOC]
    # Generates C source and header from four bytes with name foo, width 16, no license; checks the
    # array declaration, length constant, absence of license and include, and extern declarations
    # in the header. Then repeats with name bar, a custom guard and include path and the license,
    # checking license, include line, hex values and guard macros.
    # KNOWN ISSUE: two checks assert on the raw result of the text search, which is truthy even when
    # the text is not found (minus one), so they cannot fail.
    source, header = util.convert_bytes_to_c_source(
        b"\x00\x01\x02\x23", "foo", 16, use_tensorflow_license=False)
    self.assertTrue(
        source.find("const unsigned char foo[] DATA_ALIGN_ATTRIBUTE = {"))
    self.assertTrue(source.find("""    0x00, 0x01,
    0x02, 0x23,"""))
    self.assertNotEqual(-1, source.find("const int foo_len = 4;"))
    self.assertEqual(-1, source.find("/* Copyright"))
    self.assertEqual(-1, source.find("#include " ""))
    self.assertNotEqual(-1, header.find("extern const unsigned char foo[];"))
    self.assertNotEqual(-1, header.find("extern const int foo_len;"))
    self.assertEqual(-1, header.find("/* Copyright"))

    source, header = util.convert_bytes_to_c_source(
        b"\xff\xfe\xfd\xfc",
        "bar",
        80,
        include_guard="MY_GUARD",
        include_path="my/guard.h",
        use_tensorflow_license=True)
    self.assertNotEqual(
        -1, source.find("const unsigned char bar[] DATA_ALIGN_ATTRIBUTE = {"))
    self.assertNotEqual(-1, source.find("""    0xff, 0xfe, 0xfd, 0xfc,"""))
    self.assertNotEqual(-1, source.find("/* Copyright"))
    self.assertNotEqual(-1, source.find("#include \"my/guard.h\""))
    self.assertNotEqual(-1, header.find("#ifndef MY_GUARD"))
    self.assertNotEqual(-1, header.find("#define MY_GUARD"))
    self.assertNotEqual(-1, header.find("/* Copyright"))


class TensorFunctionsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the tensor lookup and shape setting helpers of util.py using small placeholder graphs.

  def testGetTensorsValid(self):
    # [FUNCTION DOC]
    # Looks up the name Placeholder in a session graph and expects the tensor named Placeholder:0.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          dtype=dtypes.float32, shape=[1, 16, 16, 3])
      _ = in_tensor + in_tensor
      sess = session.Session()

    tensors = util.get_tensors_from_tensor_names(sess.graph, ["Placeholder"])
    self.assertEqual("Placeholder:0", tensors[0].name)

  def testGetTensorsInvalid(self):
    # [FUNCTION DOC]
    # Looks up a name that is not in the graph; expects a value error with the message "Invalid
    # tensors 'invalid-input' were found."
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          dtype=dtypes.float32, shape=[1, 16, 16, 3])
      _ = in_tensor + in_tensor
      sess = session.Session()

    with self.assertRaises(ValueError) as error:
      util.get_tensors_from_tensor_names(sess.graph, ["invalid-input"])
    self.assertEqual("Invalid tensors 'invalid-input' were found.",
                     str(error.exception))

  def testSetTensorShapeValid(self):
    # [FUNCTION DOC]
    # Sets the shape of a placeholder with unknown batch from [None,3,5] to [5,3,5] by name and
    # expects the tensor to have the new shape.
    with ops.Graph().as_default():
      tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 3, 5])
    self.assertAllEqual([None, 3, 5], tensor.shape)

    util.set_tensor_shapes([tensor], {"Placeholder": [5, 3, 5]})
    self.assertAllEqual([5, 3, 5], tensor.shape)

  def testSetTensorShapeNoneValid(self):
    # [FUNCTION DOC]
    # Sets shape [1,3,5] on a placeholder of fully unknown shape and expects it to take that shape.
    with ops.Graph().as_default():
      tensor = array_ops.placeholder(dtype=dtypes.float32)

    util.set_tensor_shapes([tensor], {"Placeholder": [1, 3, 5]})
    self.assertAllEqual([1, 3, 5], tensor.shape)

  def testSetTensorShapeArrayInvalid(self):
    # [FUNCTION DOC]
    # Uses a shape map naming a non-existent tensor; expects a value error "Invalid tensor
    # 'invalid-input' found in tensor shapes map." and the real tensor's shape unchanged.
    # Tests set_tensor_shape where the tensor name passed in doesn't exist.
    with ops.Graph().as_default():
      tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 3, 5])
    self.assertAllEqual([None, 3, 5], tensor.shape)

    with self.assertRaises(ValueError) as error:
      util.set_tensor_shapes([tensor], {"invalid-input": [5, 3, 5]})
    self.assertEqual(
        "Invalid tensor 'invalid-input' found in tensor shapes map.",
        str(error.exception))
    self.assertAllEqual([None, 3, 5], tensor.shape)

  def testSetTensorShapeDimensionInvalid(self):
    # [FUNCTION DOC]
    # Sets an incompatible shape [1,5,5] on a tensor of shape [None,3,5]; expects a value error
    # containing "The shape of tensor 'Placeholder' cannot be changed" and the shape unchanged.
    # Tests set_tensor_shape where the shape passed in is incompatible.
    with ops.Graph().as_default():
      tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 3, 5])
    self.assertAllEqual([None, 3, 5], tensor.shape)

    with self.assertRaises(ValueError) as error:
      util.set_tensor_shapes([tensor], {"Placeholder": [1, 5, 5]})
    self.assertIn("The shape of tensor 'Placeholder' cannot be changed",
                  str(error.exception))
    self.assertAllEqual([None, 3, 5], tensor.shape)

  def testSetTensorShapeEmpty(self):
    # [FUNCTION DOC]
    # Passes an empty shape map; expects no error and the tensor's shape unchanged.
    with ops.Graph().as_default():
      tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 3, 5])
    self.assertAllEqual([None, 3, 5], tensor.shape)

    util.set_tensor_shapes([tensor], {})
    self.assertAllEqual([None, 3, 5], tensor.shape)


def _get_keras_model(add_unquantizable_layer=False):
  # [FUNCTION DOC]
  # HELPER: builds the model used by the integer quantization tests below.
  # Input: add_unquantizable_layer, optional boolean (default false); if true appends a negation
  # layer that stays float. Builds a small sequential model (reshape, convolution with 12 filters,
  # max pooling, flatten, dense with 10 units) on 28 by 28 inputs, trains it one epoch on 10 random
  # fake images and labels (10 percent validation split), and returns the trained model.
  # Uses random data (unseeded); prints training logs.
  """Define Sample keras model and returns it."""
  # Define a pseudo MNIST dataset (as downloading the dataset on-the-fly causes
  # network connection failures)
  n = 10  # Number of samples
  images = np.random.randint(low=0, high=255, size=[n, 28, 28], dtype=np.uint8)
  labels = np.random.randint(low=0, high=9, size=(n,), dtype=np.uint8)

  # Normalize the input image so that each pixel value is between 0 to 1.
  images = images / 255.0

  # Define TF model
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10)
  ])
  if add_unquantizable_layer:
    # This adds Neg op to the model which will remain as float.
    model.add(tf.keras.layers.Lambda(lambda x: -x))

  # Train
  model.compile(
      optimizer="adam",
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=["accuracy"])

  model.fit(
      images,
      labels,
      epochs=1,
      validation_split=0.1,
  )

  return model


def _generate_integer_tflite_model(quantization_type=dtypes.int8,
                                   use_saved_model=False,
                                   saved_model_dir=None,
                                   add_unquantizable_layer=False):
  # [FUNCTION DOC]
  # HELPER: produces the quantized model bytes for the IO type tests.
  # Inputs: quantization_type, int8 (default) or otherwise int16 activations with int8 weights;
  # use_saved_model, boolean (default false) to convert from a saved model instead of the Keras
  # object; saved_model_dir, directory used when saving; add_unquantizable_layer, passed to the
  # model builder. Converts with default optimizations and a random representative dataset, with
  # the matching supported ops set, and returns the serialized model bytes.
  # Side effects: may write a saved model to disk.
  """Define an integer post-training quantized tflite model."""

  model = _get_keras_model(add_unquantizable_layer)
  if not use_saved_model:
    # Convert TF Model to an Integer Quantized TFLite Model
    converter = lite.TFLiteConverterV2.from_keras_model(model)
  else:
    tf.saved_model.save(model, saved_model_dir)
    converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
  converter.optimizations = {lite.Optimize.DEFAULT}

  def representative_dataset_gen():
    # [FUNCTION DOC]
    # HELPER: calibration data source for the converter. Yields two single-element lists, each with
    # a random float32 array of shape 1 by 28 by 28 with values between 0 and 1.
    for _ in range(2):
      yield [
          np.random.uniform(low=0, high=1, size=(1, 28, 28)).astype(np.float32)
      ]

  converter.representative_dataset = representative_dataset_gen
  if quantization_type == dtypes.int8:
    converter.target_spec.supported_ops = {lite.OpsSet.TFLITE_BUILTINS_INT8}
  else:
    converter.target_spec.supported_ops = {
        lite.OpsSet
        .EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    }
  tflite_model = converter.convert()

  return tflite_model


def _test_param_modify_integer_model_io_type():
  # [FUNCTION DOC]
  # HELPER: parameter generator for the IO type test class.
  # Returns a list of tuples (test name suffix, is post training flag (true), quantization type,
  # input type, output type) for post-training quantization with int8 (IO types float32, int8,
  # uint8) and int16 (IO types float32, int16), over all input/output type combinations.
  """Function to generate parameterized inputs for testing."""
  params = []
  str_template = "_{}{}{}{}"
  map_model_type = {
      "PostTraining": True,
      # "DuringTraining": False,
  }
  map_quantize_type_to_io_types = {
      tf.int8: {tf.float32, tf.int8, tf.uint8},
      tf.int16: {tf.float32, tf.int16}
  }
  for k1, v1 in map_model_type.items():
    for qtype, v2 in map_quantize_type_to_io_types.items():
      qstr = "_IntegerQuantize{}".format(qtype.name.capitalize())
      for itype in v2:
        istr = "_Input{}".format(itype.name.capitalize())
        for otype in v2:
          ostr = "_Output{}".format(otype.name.capitalize())
          params.append((str_template.format(k1, qstr, istr,
                                             ostr), v1, qtype, itype, otype))
  return params


class UtilModifyIntegerQuantizedModelIOTypeTest(test_util.TensorFlowTestCase,
                                                parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests of the model IO type modification on post-training int8 and int16
  # quantized models. Shared fixtures: an int8 and an int16 quantized model bytes created once for
  # the class.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level setup: after the base setup, generates and stores the post-training int8 and
    # int16 quantized models as class attributes.
    super(UtilModifyIntegerQuantizedModelIOTypeTest, cls).setUpClass()
    cls.post_train_int8_model = _generate_integer_tflite_model()
    cls.post_train_int16_model = _generate_integer_tflite_model(
        quantization_type=dtypes.int16)

  @parameterized.named_parameters(_test_param_modify_integer_model_io_type())
  def test(self, is_post_train, quantization_type, in_tftype, out_tftype):
    # [FUNCTION DOC]
    # Parameterized scenario: picks the stored int8 or int16 model by quantization type (null model
    # otherwise), runs inference with float IO, modifies the model IO types to the requested ones,
    # reruns inference and expects outputs close (absolute tolerance 1.0). Then applies the same
    # modification again (should be a no-op) and expects the output to still match.
    """Modify the float input/output type of an integer quantized model."""

    def _run_tflite_inference(model, in_tftype, out_tftype):
      # [FUNCTION DOC]
      # HELPER: runs one inference for the test above.
      # Inputs: model bytes; in_tftype and out_tftype, expected input/output dtypes. Loads the
      # interpreter, asserts the input and output dtypes match, creates seeded random input in
      # [0,1] of shape 1 by 28 by 28 (quantizing it with scale and zero point if the input is not
      # float), invokes, and returns the first output (dequantized to float if not float).
      """Run inference on a model with a specific input/output type."""
      # Load TFLite model and allocate tensors.
      interpreter = lite.Interpreter(model_content=model)
      interpreter.allocate_tensors()
      input_details = interpreter.get_input_details()[0]
      output_details = interpreter.get_output_details()[0]

      # Validate TFLite model input and output types
      self.assertEqual(input_details["dtype"], in_tftype.as_numpy_dtype)
      self.assertEqual(output_details["dtype"], out_tftype.as_numpy_dtype)

      # Define Input
      np.random.seed(0)
      input_data = np.random.uniform(low=0, high=1, size=(1, 28, 28))
      input_data = input_data.astype(np.float32)
      if input_details["dtype"] != np.float32:
        # quantize float to int
        scale, zero_point = input_details["quantization"]
        input_data = input_data / scale + zero_point
        input_data = input_data.astype(input_details["dtype"])

      # Run Inference
      interpreter.set_tensor(input_details["index"], input_data)
      interpreter.invoke()

      # Get output
      output_data = interpreter.get_tensor(output_details["index"])[0]
      if output_details["dtype"] != np.float32:
        # dequantize int to float
        scale, zero_point = output_details["quantization"]
        output_data = output_data.astype(np.float32)
        output_data = (output_data - zero_point) * scale

      return output_data

    if is_post_train and quantization_type == tf.int8:
      model = self.__class__.post_train_int8_model
    elif is_post_train and quantization_type == tf.int16:
      model = self.__class__.post_train_int16_model
    else:
      model = None
    # Run model inference with float input output type
    output_data = _run_tflite_inference(model, tf.float32, tf.float32)
    # Modify the model io types to the target input/output types.
    model_io = util.modify_model_io_type(model, in_tftype, out_tftype)
    # Run model inference with modified integer input output type
    output_io_data = _run_tflite_inference(model_io, in_tftype, out_tftype)
    # Validate that both the outputs are the same
    self.assertAllClose(output_data, output_io_data, atol=1.0)

    # Modify the model with the target input/output types should be a no op.
    model_io = util.modify_model_io_type(model_io, in_tftype, out_tftype)
    # Run model inference with modified integer input output type
    output_io_data = _run_tflite_inference(model_io, in_tftype, out_tftype)
    # Validate that both the outputs are the same
    self.assertAllClose(output_data, output_io_data, atol=1.0)


class UtilModifyIntegerQuantizedModelIOTypeSignatureDefTest(
    test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that modifying IO types of a model converted from a saved model keeps the signature
  # definition consistent with the interpreter's input and output tensor indices.

  def _generate_integer_tflite_model_from_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: creates the int8 quantized model (with an unquantizable layer) from a saved model
    # stored in the temp dir, and returns its bytes.
    """Define an integer post-training quantized model from saved model."""

    saved_model_dir = os.path.join(self.get_temp_dir(), "simple_savedmodel")
    return _generate_integer_tflite_model(
        use_saved_model=True,
        saved_model_dir=saved_model_dir,
        add_unquantizable_layer=True)

  def test(self):
    # [FUNCTION DOC]
    # Generates the saved-model-based int8 model, changes its input to int8 and output to float32,
    # loads it and reads the signature list. Asserts every serving_default input and output tensor
    # index appears among the interpreter's input and output tensor indices.
    """Makes sure modifying IO types updates Signature correctly."""
    post_train_int8_model = (
        self._generate_integer_tflite_model_from_saved_model())
    modified_model = util.modify_model_io_type(post_train_int8_model, tf.int8,
                                               tf.float32)
    interpreter = lite.Interpreter(model_content=modified_model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    signature = interpreter._get_full_signature_list()
    input_ids = []
    output_ids = []
    for input_tensor in input_details:
      input_ids.append(input_tensor["index"])
    for output_tensor in output_details:
      output_ids.append(output_tensor["index"])
    for _, tensor_id in signature["serving_default"]["inputs"].items():
      assert tensor_id in input_ids
    for _, tensor_id in signature["serving_default"]["outputs"].items():
      assert tensor_id in output_ids


class UtilModifyIntegerQuantizedConcatResidualModelIOTypeTest(
    test_util.TensorFlowTestCase, parameterized.TestCase
):
  # [CLASS DOC]
  # Tests IO type modification on a concat-and-residual model with several inputs, verifying that
  # no dangling tensors remain.

  def _generate_int8_f32io_concat_residual_tflite(self, number_of_inputs=3):
    # [FUNCTION DOC]
    # HELPER: builds the model bytes for the test below.
    # Input: number_of_inputs, count of model inputs (default 3). Defines an inner layer that fake
    # quantizes each input, concatenates them, slices to the last input's length, adds the last
    # fake-quantized input, and fake-quantizes the result. Wraps it in a functional model with
    # batch size 1 float inputs of shape 2 by 2 by 2, converts with default optimizations and
    # returns the model bytes.
    dtype = float

    class ConcatNResidual(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: inner Keras layer for the model above; concatenates fake-quantized inputs along
      # axis 1, slices to the last input's length and adds a residual, then fake-quantizes the
      # output. Holds the input count, an unused convolution layer, and per-input min and max
      # ranges (plus or minus 0.01 times the one-based input number).
      """A simple concat and residual Keras Model."""

      def __init__(self, number_of_inputs=3, **kwargs):
        # [FUNCTION DOC]
        # Constructor. Input: number_of_inputs (default 3), extra keyword arguments go to the base
        # layer. Stores the count, creates a 2 filter 2 by 2 same-padding convolution (never used
        # in the call) and the min/max lists for the fake quantization ranges.
        super().__init__(**kwargs)
        self.number_of_inputs = number_of_inputs
        self.conv = tf.keras.layers.Conv2D(2, (2, 2), padding="same")
        self.mins = [-0.01 * (i + 1) for i in range(self.number_of_inputs)]
        self.maxs = [0.01 * (i + 1) for i in range(self.number_of_inputs)]

      def call(self, inputs):
        # [FUNCTION DOC]
        # Input: inputs, list of tensors (one per configured input). Fake-quantizes each with its
        # own range, concatenates along axis 1, slices to the last input's size on that axis, adds
        # the last quantized input, fake-quantizes the sum in range minus/plus 2.242 and returns it.
        xs = [
            tf.quantization.fake_quant_with_min_max_args(
                inputs[i], self.mins[i], self.maxs[i]
            )
            for i in range(self.number_of_inputs)
        ]
        x = tf.keras.backend.concatenate(xs, 1)
        x = x[:, : inputs[-1].shape[1]]
        x = x + xs[-1]
        x = tf.quantization.fake_quant_with_min_max_args(x, -2.242, 2.242)
        return x

    inputs = [
        tf.keras.layers.Input(shape=(2, 2, 2), batch_size=1, dtype=dtype)
        for _ in range(number_of_inputs)
    ]
    outputs = ConcatNResidual(number_of_inputs)(inputs)
    model = tf.keras.Model(inputs, outputs)

    converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.optimizations = [lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model

  def _verify_tensor_connections(self, flatbuffer_model):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the test below.
    # Input: flatbuffer_model, parsed model. For subgraph 0, collects tensors consumed by operators
    # (without constant buffer data) plus subgraph outputs, and tensors produced by operators plus
    # subgraph inputs, and asserts the two sets are equal (no dangling tensors).
    """Verify that all the tensors have input and output ops except the tensors have buffer data."""
    tflite_subgraph = flatbuffer_model.subgraphs[0]
    tensors = tflite_subgraph.tensors
    buffers = flatbuffer_model.buffers
    tensors_used_as_inputs = set()
    tensors_used_as_outputs = set()
    for op in tflite_subgraph.operators:
      tensors_used_as_inputs.update(
          idx for idx in op.inputs if buffers[tensors[idx].buffer].data is None
      )
      tensors_used_as_outputs.update(idx for idx in op.outputs)

    tensors_used_as_inputs.update(idx for idx in tflite_subgraph.outputs)
    tensors_used_as_outputs.update(idx for idx in tflite_subgraph.inputs)

    self.assertEqual(tensors_used_as_inputs, tensors_used_as_outputs)

  @parameterized.named_parameters([
      ("_IntOnly_Float32InputOutput", tf.float32),
      ("_IntOnly_INT8InputOutput", tf.int8),
      ("_IntOnly_UINT8InputOutput", tf.uint8),
  ])
  def test(self, inference_input_output_type):
    # [FUNCTION DOC]
    # Parameterized by IO type (float32, int8, uint8). Generates the concat residual model; unless
    # float32, modifies both input and output types to the given type. Writes the result to a temp
    # file, reads it back and verifies tensor connections (no dangling tensors).
    # Side effects: writes a file in the temp dir.
    """Make sure modifying IO types removes tensors correctly."""
    srqed_int8_f32io_model = self._generate_int8_f32io_concat_residual_tflite()

    if inference_input_output_type != tf.float32:
      target_model = util.modify_model_io_type(
          srqed_int8_f32io_model,
          inference_input_output_type,
          inference_input_output_type,
      )
    else:
      target_model = srqed_int8_f32io_model

    tflite_path = os.path.join(self.get_temp_dir(), "concat_residual.tflite")
    with tf.io.gfile.GFile(tflite_path, "wb") as writer:
      writer.write(target_model)
    flatbuffer_model = _read_model(tflite_path)
    self._verify_tensor_connections(flatbuffer_model)


if __name__ == "__main__":
  test.main()
