# [MODULE DOC]
# HELPER: shared base test class and model builders for the TF2 TFLite converter tests.
# Defines ModelTest, a base class for the TensorFlow 2.x TensorFlow Lite conversion tests (for
# example the lite_v2 tests). It offers helpers to run a converted model through the TFLite
# interpreter and many small factory methods that build trackable models, Keras models and
# calibration generators used by those tests. No tests of its own, no import-time side effects.
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

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable


class ModelTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # HELPER: base class for TFLite 2.x converter tests; subclasses inherit its helpers.
  # Combines the TensorFlow test case with parameterized-test support. Holds no state; supplies
  # methods to evaluate TFLite models with the interpreter, to build assorted test models, and to
  # validate debug info. Contains no test methods itself.
  """Base test class for TensorFlow Lite 2.x model tests."""

  def _evaluateTFLiteModel(self, tflite_model, input_data, input_shapes=None):
    # [FUNCTION DOC]
    # HELPER: runs a serialized TFLite model on given inputs for the converter tests.
    # Inputs: tflite_model, serialized model bytes; input_data, list of eager tensors, one per
    # model input, in input order; input_shapes, optional list of pairs (expected shape signature,
    # new concrete shape), one per input with dynamic dimensions, in input order.
    # Creates an interpreter; if input_shapes is given, asserts each input's shape signature equals
    # the expected one and resizes that input strictly to the new shape. Then allocates tensors,
    # sets each input from the tensors converted to arrays, invokes, and returns a list of output
    # arrays in output order. Fails the test on a signature mismatch. Only side effect is the
    # interpreter it creates.
    """Evaluates the model on the `input_data`.

    Args:
      tflite_model: TensorFlow Lite model.
      input_data: List of EagerTensor const ops containing the input data for
        each input tensor.
      input_shapes: List of tuples representing the `shape_signature` and the
        new shape of each input tensor that has unknown dimensions.

    Returns:
      [np.ndarray]
    """
    interpreter = Interpreter(model_content=tflite_model)
    input_details = interpreter.get_input_details()
    if input_shapes:
      for idx, (shape_signature, final_shape) in enumerate(input_shapes):
        self.assertTrue(
            (input_details[idx]['shape_signature'] == shape_signature).all())
        index = input_details[idx]['index']
        interpreter.resize_tensor_input(index, final_shape, strict=True)
    interpreter.allocate_tensors()

    output_details = interpreter.get_output_details()
    input_details = interpreter.get_input_details()

    for input_tensor, tensor_data in zip(input_details, input_data):
      interpreter.set_tensor(input_tensor['index'], tensor_data.numpy())
    interpreter.invoke()
    return [
        interpreter.get_tensor(details['index']) for details in output_details
    ]

  def _evaluateTFLiteModelUsingSignatureDef(self, tflite_model, signature_key,
                                            inputs):
    # [FUNCTION DOC]
    # HELPER: runs one signature of a serialized TFLite model for the converter tests.
    # Inputs: tflite_model, serialized model bytes; signature_key, name of the signature to run;
    # inputs, mapping from input names of that signature to array values. Builds an interpreter,
    # obtains the runner for the signature and calls it with the inputs as named arguments.
    # Returns the runner's mapping from output names to output values. Errors from an unknown
    # signature key or bad inputs propagate from the interpreter.
    """Evaluates the model on the `inputs`.

    Args:
      tflite_model: TensorFlow Lite model.
      signature_key: Signature key.
      inputs: Map from input tensor names in the SignatureDef to tensor value.

    Returns:
      Dictionary of outputs.
      Key is the output name in the SignatureDef 'signature_key'
      Value is the output value
    """
    interpreter = Interpreter(model_content=tflite_model)
    signature_runner = interpreter.get_signature_runner(signature_key)
    return signature_runner(**inputs)

  def _getSimpleVariableModel(self):
    # [FUNCTION DOC]
    # HELPER: model factory for tests. Builds a trackable root object with two float variables
    # (values 3 and 2) and a compiled function that multiplies its input by the product of both
    # variables. Takes no inputs; returns the root object.
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    return root

  def _getSimpleModelWithVariables(self):
    # [FUNCTION DOC]
    # HELPER: model factory for tests. Defines a local trackable class with one variable and an
    # add-in-place function, and returns a new instance of it. Takes no inputs.

    class SimpleModelWithOneVariable(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: local test model. Holds a single variable initialized to zeros of shape one by
      # ten, and exposes a compiled function that adds its input to the variable in place.
      """Basic model with 1 variable."""

      def __init__(self):
        # [FUNCTION DOC]
        # Calls the parent constructor and creates the variable attribute holding zeros of
        # shape one by ten.
        super(SimpleModelWithOneVariable, self).__init__()
        self.var = variables.Variable(array_ops.zeros((1, 10), name='var'))

      @def_function.function
      def assign_add(self, x):
        # [FUNCTION DOC]
        # Adds input x to the variable in place (a stateful update) and returns the variable.
        # Compiled as a graph function when traced.
        self.var.assign_add(x)
        return self.var

    return SimpleModelWithOneVariable()

  def _getMultiFunctionModel(self):
    # [FUNCTION DOC]
    # HELPER: model factory for tests. Defines a local trackable class with three compiled
    # functions (add, sub, mul_add) that lazily create variables, and returns a new instance.
    # Takes no inputs.

    class BasicModel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: local test model with several compiled functions (add, sub, mul_add). Holds two
      # attributes, y and z, initially empty and lazily set to variables (2 and 3) on first trace.
      """Basic model with multiple functions."""

      def __init__(self):
        # [FUNCTION DOC]
        # Sets both lazily-created variable attributes, y and z, to the null value.
        self.y = None
        self.z = None

      @def_function.function
      def add(self, x):
        # [FUNCTION DOC]
        # Creates variable y (value 2) if not yet created, then returns input x plus y.
        if self.y is None:
          self.y = variables.Variable(2.)
        return x + self.y

      @def_function.function
      def sub(self, x):
        # [FUNCTION DOC]
        # Creates variable z (value 3) if not yet created, then returns input x minus z.
        if self.z is None:
          self.z = variables.Variable(3.)
        return x - self.z

      @def_function.function
      def mul_add(self, x, y):
        # [FUNCTION DOC]
        # Creates variable z (value 3) if not yet created, then returns x times z plus the
        # second input y. Note the parameter y is distinct from the attribute y.
        if self.z is None:
          self.z = variables.Variable(3.)
        return x * self.z + y

    return BasicModel()

  def _getMultiFunctionModelWithSharedWeight(self):
    # [FUNCTION DOC]
    # HELPER: model factory for tests. Defines a local trackable class with add, sub and mul
    # functions sharing one constant weight, and returns a new instance. Takes no inputs.

    class BasicModelWithSharedWeight(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: local test model. Holds one constant float weight of shape one by 512 by 512 by
      # one (value 1.0) shared by three compiled functions: add, sub and mul.
      """Model with multiple functions and a shared weight."""

      def __init__(self):
        # [FUNCTION DOC]
        # Creates the shared constant weight attribute (float32, shape 1x512x512x1, value 1.0).
        self.weight = constant_op.constant([1.0],
                                           shape=(1, 512, 512, 1),
                                           dtype=dtypes.float32)

      @def_function.function
      def add(self, x):
        # [FUNCTION DOC]
        # Returns input x plus the shared weight.
        return x + self.weight

      @def_function.function
      def sub(self, x):
        # [FUNCTION DOC]
        # Returns input x minus the shared weight.
        return x - self.weight

      @def_function.function
      def mul(self, x):
        # [FUNCTION DOC]
        # Returns input x multiplied element-wise by the shared weight.
        return x * self.weight

    return BasicModelWithSharedWeight()

  def _getMatMulModelWithSmallWeights(self):
    # [FUNCTION DOC]
    # HELPER: model factory for tests of quantization with small weights. Defines a local
    # trackable class doing a matrix multiply plus bias, and returns a new instance. No inputs.

    class MatMulModelWithSmallWeights(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: local test model. Holds a two by two constant weight matrix with tiny values and a
      # bias vector of two relatively large values (1.28, 2.55); one compiled matmul function.
      """MatMul model with small weights and relatively large biases."""

      def __init__(self):
        # [FUNCTION DOC]
        # Creates the constant weight attribute (float32, 2x2, small values) and the constant
        # bias attribute (float32, length two).
        self.weight = constant_op.constant([[1e-3, -1e-3], [-2e-4, 2e-4]],
                                           shape=(2, 2),
                                           dtype=dtypes.float32)
        self.bias = constant_op.constant([1.28, 2.55],
                                         shape=(2,),
                                         dtype=dtypes.float32)

      @def_function.function
      def matmul(self, x):
        # [FUNCTION DOC]
        # Returns the matrix product of input x with the weight, plus the bias.
        return x @ self.weight + self.bias

    return MatMulModelWithSmallWeights()

  def _getCeilModel(self):
    # [FUNCTION DOC]
    # HELPER: model factory for tests of non-quantizable ops. Takes no inputs. Returns a pair:
    # a compiled function with a fixed float32 input signature of shape one by ten that applies
    # the ceiling op, and a calibration generator function (see the nested function).
    """Returns a model with only one ceil op, to test non-quantizable op."""

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=(1, 10), dtype=dtypes.float32)
    ])
    def ceil(x):
      # [FUNCTION DOC]
      # Returns the element-wise ceiling of input x.
      return math_ops.ceil(x)

    def calibration_gen():
      # [FUNCTION DOC]
      # Generator yielding five calibration samples; each is a list holding one random float32
      # array of shape one by ten drawn uniformly from zero to sixteen. Uses global random state.
      for _ in range(5):
        yield [np.random.uniform(0, 16, size=(1, 10)).astype(np.float32)]

    return ceil, calibration_gen

  def _assertValidDebugInfo(self, debug_info):
    # [FUNCTION DOC]
    # HELPER: assertion helper for converter debug-info tests.
    # Input: debug_info, an object with a list of file paths. Collects the base names of the
    # files and asserts that "lite_v2_test.py" is among them and "lite_test.py" is not.
    # KNOWN ISSUE: it hardcodes the name of the v2 test file, so it only works when called from
    # that file's tests.
    """Verify the DebugInfo is valid."""
    file_names = set()
    for file_path in debug_info.files:
      file_names.add(os.path.basename(file_path))
    # To make the test independent on how the nodes are created, we only assert
    # the name of this test file.
    self.assertIn('lite_v2_test.py', file_names)
    self.assertNotIn('lite_test.py', file_names)

  def _createV2QATLowBitKerasModel(self, shape, weight_only, num_bits, bit_min,
                                   bit_max):
    # [FUNCTION DOC]
    # HELPER: model factory for low-bit quantization-aware-training tests.
    # Inputs: shape, input shape without batch (last entry is channel count); weight_only, if
    # true only weights are fake-quantized, otherwise inputs and outputs are too (8 bits, range
    # zero to six); num_bits, bit width for the weight fake quantization; bit_min and bit_max,
    # lower and upper bound of the weight quantization range.
    # Builds a Keras model with a named input and a wrapped single-filter 3x3 convolution with
    # relu6 activation whose kernel is initialized to evenly spaced values from bit_min to bit_max
    # (so the range is exactly represented), followed by a named flatten layer. Returns a triple:
    # the model, the input layer name ("input") and the output layer name ("scores").
    """Creates a simple QAT num_bits-Weight Keras Model."""
    input_name = 'input'
    output_name = 'scores'

    class ConvWrapper(tf.keras.layers.Wrapper):
      # [CLASS DOC]
      # HELPER: local Keras wrapper layer that simulates quantization-aware training on a wrapped
      # convolution. Keeps the wrapped layer's original kernel as the quantized_weights attribute
      # and, on each call, replaces the layer's kernel by its per-channel fake-quantized version
      # using the enclosing function's bit settings.
      """A Wrapper for simulating QAT on Conv2D layers."""

      def build(self, input_shape):
        # [FUNCTION DOC]
        # Builds the wrapped layer if not yet built, and stores its kernel in the
        # quantized_weights attribute. Input: input_shape of the layer input.
        if not self.layer.built:
          self.layer.build(input_shape)
        self.quantized_weights = self.layer.kernel

      def call(self, inputs):
        # [FUNCTION DOC]
        # Forward pass. Sets the wrapped layer's kernel to the stored weights passed through
        # per-channel fake quantization (given bits, narrow range, bounds bit_min/bit_max).
        # If not weight_only, fake-quantizes inputs (8 bits, zero to six), runs the wrapped
        # layer, and fake-quantizes the outputs likewise; otherwise runs the wrapped layer on
        # the raw inputs. Returns the output tensor. Mutates the wrapped layer's kernel.
        self.layer.kernel = (
            tf.quantization.fake_quant_with_min_max_vars_per_channel(
                self.quantized_weights, min=[bit_min], max=[bit_max],
                num_bits=num_bits, narrow_range=True))
        if not weight_only:
          quant_inputs = tf.quantization.fake_quant_with_min_max_vars(
              inputs, min=0, max=6, num_bits=8)
          outputs = self.layer.call(quant_inputs)
          return tf.quantization.fake_quant_with_min_max_vars(
              outputs, min=0, max=6, num_bits=8)
        return self.layer.call(inputs)

    input_tensor = tf.keras.layers.Input(shape, name=input_name)
    kernel_shape = (shape[-1], 3, 3, 1)
    # Ensure constant weights contains the min and max.
    initial_weights = np.linspace(
        bit_min, bit_max, np.prod(kernel_shape)).reshape(kernel_shape)
    test_initializer = tf.constant_initializer(initial_weights)
    x = ConvWrapper(tf.keras.layers.Conv2D(
        1, (3, 3), kernel_initializer=test_initializer,
        activation='relu6'))(input_tensor)
    scores = tf.keras.layers.Flatten(name=output_name)(x)
    model = tf.keras.Model(input_tensor, scores)
    return model, input_name, output_name

  def _createReadAssignModel(self, number_of_states=2):
    # [FUNCTION DOC]
    # HELPER: model factory for variable-quantization tests.
    # Input: number_of_states, how many state variables the layer keeps (default two).
    # Builds a Keras model with a float input of shape two by three and batch size one, passed
    # through a stateful layer. Returns a pair: the model and a calibration generator function
    # yielding five samples (each a list with one random float32 array of shape 1x2x3 in [-1, 1]).
    dtype = float

    class ReadAssign(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: local Keras layer for the variable quantization test. Holds a list of
      # non-trainable zero-initialized state variables (shape 1x2x3), one per requested state. On
      # each call it concatenates every state with the inputs, writes the leading slice back into
      # the state, and returns the inputs unchanged.
      """ReadAssign model for the variable quantization test."""

      def __init__(self, number_of_states=2, **kwargs):
        # [FUNCTION DOC]
        # Calls the parent constructor with extra keyword arguments and records
        # number_of_states (default two), the count of state variables to create at build time.
        super().__init__(**kwargs)
        self.number_of_states = number_of_states

      def build(self, input_shape):
        # [FUNCTION DOC]
        # Calls the parent build, then creates number_of_states non-trainable weights named
        # states0, states1, ..., each of shape 1x2x3 zero-initialized, stored in the states list.
        super().build(input_shape)

        state_shape = (1, 2, 3)
        self.states = [None] * self.number_of_states
        for i in range(self.number_of_states):
          self.states[i] = self.add_weight(
              name=f'states{i}',
              shape=state_shape,
              trainable=False,
              initializer=tf.zeros_initializer,
              dtype=dtype,
          )

      def call(self, inputs):
        # [FUNCTION DOC]
        # For each state variable: concatenates the state and inputs along axis one, keeps the
        # first slice of that axis with the state's own length, and assigns it to the state
        # (a variable write). Returns the inputs unchanged.

        for state in self.states:
          memory = tf.keras.backend.concatenate([state, inputs], 1)
          new_state = memory[:, : state.shape[1], :]
          state.assign(new_state)

        return inputs

    def calibration_gen():
      # [FUNCTION DOC]
      # Generator yielding five calibration samples; each is a list with one random float32 array
      # of shape 1x2x3 drawn uniformly from minus one to one. Uses global random state.
      for _ in range(5):
        yield [np.random.uniform(-1, 1, size=(1, 2, 3)).astype(np.float32)]

    inputs = tf.keras.layers.Input(shape=(2, 3), batch_size=1, dtype=dtype)
    outputs = ReadAssign(number_of_states)(inputs)
    model = tf.keras.Model(inputs, outputs)
    return model, calibration_gen

  def _getInfFloatModel(self):
    # [FUNCTION DOC]
    # HELPER: model factory for tests with infinite constants. Builds a trackable root holding a
    # scalar float32 constant of positive infinity and a compiled function that ignores its input
    # and returns that constant. Takes no inputs; returns the root object.
    root = autotrackable.AutoTrackable()
    root.v = constant_op.constant([np.inf], shape=(), dtype=dtypes.float32)
    root.f = def_function.function(lambda x: root.v)
    return root
