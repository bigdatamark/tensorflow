# [MODULE DOC]
# HELPER: small shared test helper module for the TFLite signal kernel tests.
# Offers two functions: one converts a callable into a serialized TFLite model through the V2
# converter, the other runs such a model with the TFLite interpreter on numpy inputs. No
# import-time side effects beyond imports.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Test utilities for tf.signal."""

from tensorflow.lite.python import interpreter
from tensorflow.lite.python import lite
from tensorflow.python.eager import def_function


def tflite_convert(fn, input_templates):
  # [FUNCTION DOC]
  # HELPER: converts a callable to a TFLite model for the signal tests.
  # Parameters: fn is a callable taking tensors; input_templates is a list of tensors, arrays or
  # tensor specs describing the inputs (only shapes and types matter). Wraps fn as a graph function,
  # gets its concrete function for the templates, converts it with the V2 converter and returns the
  # serialized model bytes. Conversion errors propagate.
  """Converts the provided fn to tf.lite model.

  Args:
    fn: A callable that expects a list of inputs like input_templates that
      returns a tensor or structure of tensors.
    input_templates: A list of Tensors, ndarrays or TensorSpecs describing the
      inputs that fn expects. The actual values of the Tensors or ndarrays are
      unused.

  Returns:
    The serialized tf.lite model.
  """
  fn = def_function.function(fn)
  concrete_func = fn.get_concrete_function(*input_templates)
  converter = lite.TFLiteConverterV2([concrete_func])
  return converter.convert()


def evaluate_tflite_model(tflite_model, input_ndarrays):
  # [FUNCTION DOC]
  # HELPER: runs a serialized TFLite model once for the signal tests.
  # Parameters: tflite_model is the model as bytes; input_ndarrays is a list of numpy arrays, one
  # per model input, in input order. Creates an interpreter, allocates tensors, sets each input,
  # invokes it and returns a list of output arrays in output order. Raises a value error if the
  # number of inputs differs from the model's input count (message includes the details).
  """Evaluates the provided tf.lite model with the given input ndarrays.

  Args:
    tflite_model: bytes. The serialized tf.lite model.
    input_ndarrays: A list of NumPy arrays to feed as input to the model.

  Returns:
    A list of ndarrays produced by the model.

  Raises:
    ValueError: If the number of input arrays does not match the number of
      inputs the model expects.
  """
  the_interpreter = interpreter.Interpreter(model_content=tflite_model)
  the_interpreter.allocate_tensors()

  input_details = the_interpreter.get_input_details()
  output_details = the_interpreter.get_output_details()

  if len(input_details) != len(input_ndarrays):
    raise ValueError('Wrong number of inputs: provided=%s, '
                     'input_details=%s output_details=%s' % (
                         input_ndarrays, input_details, output_details))
  for input_tensor, data in zip(input_details, input_ndarrays):
    the_interpreter.set_tensor(input_tensor['index'], data)
  the_interpreter.invoke()
  return [the_interpreter.get_tensor(details['index'])
          for details in output_details]
