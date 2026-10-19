# [MODULE DOC]
# Example script showing the error stack trace reported by the MLIR-based TensorFlow Lite
# converter. Defines a tiny module whose model uses an operation the converter does not support
# (reciprocal), then tries to convert it both from a concrete function and from a saved model.
# Failures in the saved-model case are swallowed by a decorator unless the user removes it.
# Run as a script via the app runner; it rejects extra command line arguments.
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
"""CodeLab for displaying error stack trace w/ MLIR-based converter."""

import sys

from absl import app

import tensorflow as tf
from tensorflow.lite.python import lite


def suppress_exception(f):
  # [FUNCTION DOC]
  # HELPER: decorator used by this example to hide conversion failures.
  # Takes a function with no arguments and returns a wrapper that calls it and silently ignores
  # every exception (bare catch-all). The wrapper discards the original return value.
  def wrapped():
    # [FUNCTION DOC]
    # HELPER: wrapper produced by the exception-suppressing decorator. Takes no arguments, calls
    # the decorated function and swallows any exception. Returns nothing.
    try:
      f()
    except:  # pylint: disable=bare-except
      pass
  return wrapped


class TestModule(tf.Module):
  # [CLASS DOC]
  # Module holding one traced function with a fixed 3x3 float input signature whose body uses the
  # reciprocal operation, which the converter does not support. Used as the conversion target.
  """The test model has unsupported op."""

  @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
  def model(self, x):
    # [FUNCTION DOC]
    # Traced method. Input: a 3x3 float tensor. Computes its reciprocal (unsupported by the
    # converter) and returns the sum of that result with itself.
    y = tf.math.reciprocal(x)  # Not supported
    return y + y


# comment out the `@suppress_exception` to display the stack trace
@suppress_exception
def test_from_saved_model():
  # [FUNCTION DOC]
  # Demo (not a real test). Saves the test module as a saved model, with debug information, in a
  # fixed temporary path, then loads it into the lite converter and converts, which is expected
  # to fail on the unsupported operation. Decorated so the failure is hidden unless the decorator
  # is removed. Returns nothing. Side effect: writes files under the temporary directory.
  """displaying stack trace when converting saved model."""
  test_model = TestModule()
  saved_model_path = '/tmp/test.saved_model'
  save_options = tf.saved_model.SaveOptions(save_debug_info=True)
  tf.saved_model.save(test_model, saved_model_path, options=save_options)

  # load the model and convert
  converter = lite.TFLiteConverterV2.from_saved_model(saved_model_path)
  converter.convert()


# comment out the `@suppress_exception` to display the stack trace
# @suppress_exception
def test_from_concrete_function():
  # [FUNCTION DOC]
  # Demo (not a real test). Defines a traced function with a 3x3 float input that uses the
  # unsupported reciprocal operation, gets its concrete function and converts it with the lite
  # converter, which is expected to raise an error showing the stack trace. The decorator that
  # suppresses errors is commented out, so the error propagates. Returns nothing.
  """displaying stack trace when converting concrete function."""
  @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
  def model(x):
    # [FUNCTION DOC]
    # Nested traced function. Input: a 3x3 float tensor. Returns the sum of its reciprocal
    # (unsupported by the converter) with itself.
    y = tf.math.reciprocal(x)  # not supported
    return y + y

  func = model.get_concrete_function()
  converter = lite.TFLiteConverterV2.from_concrete_functions([func], model)
  converter.convert()


def main(argv):
  # [FUNCTION DOC]
  # Entry point. Input: the command line argument list; raises a usage error if more than one
  # entry (program name only expected) is present. Writes headers to standard output and runs the
  # concrete-function demo, then the saved-model demo. Returns nothing.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  sys.stdout.write('==== Testing from_concrete_functions ====\n')
  test_from_concrete_function()

  sys.stdout.write('==== Testing from_saved_model ====\n')
  test_from_saved_model()


if __name__ == '__main__':
  app.run(main)
