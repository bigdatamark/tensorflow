# [MODULE DOC]
# HELPER: Lit/FileCheck test driver for TFLite converter error reporting from saved models.
# Run as a program by the lit runner. It saves a module containing an unsupported operation as a
# saved model with debug info into a fixed temporary path, then converts it with the TFLite
# converter and prints the resulting error to standard output; the comment-embedded check lines
# verify the unsupported-operation message, the call-site notes and the failed-conversion line.
# Enables TF2 behavior at import time if available.
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
"""Test file to display the error message and verify it with FileCheck."""

# RUN: %p/saved_model_error | FileCheck %s

import sys

from absl import app

import tensorflow.compat.v2 as tf
if hasattr(tf, 'enable_v2_behavior'):
  tf.enable_v2_behavior()


class TestModule(tf.Module):
  # [CLASS DOC]
  # Module with one traced method that uses an operation unsupported by the TFLite converter;
  # saved and converted by the test scenario.
  """The test model has unsupported op."""

  @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
  def model(self, x):
    # [FUNCTION DOC]
    # Traced method. x: 3 by 3 float tensor. Applies the unsupported incomplete beta function
    # with constants 0.5 and 1.0 and returns the result added to itself.
    y = tf.math.betainc(x, 0.5, 1.0)  # Not supported
    return y + y


class TestGraphDebugInfo(object):
  # [CLASS DOC]
  # Plain (not framework test case) holder of one scenario checking that conversion errors from a
  # saved model carry stack trace debug info. Instantiated without arguments by the driver.
  """Test stack trace can be displayed."""

  def testSavedModelDebugInfo(self):
    # [FUNCTION DOC]
    # Scenario: save the test module (with debug info enabled) to a fixed path under the system
    # temporary directory (overwriting any previous contents), load it with the TFLite converter
    # from the saved model and convert. Expected: conversion fails with an exception whose text
    # is checked by the file's check lines. Nothing is asserted here. Returns nothing.
    """Save a saved model with unsupported ops, and then load and convert it."""
    # saved the model
    test_model = TestModule()
    saved_model_path = '/tmp/test.saved_model'
    save_options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(test_model, saved_model_path, options=save_options)

    # load the model and convert
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.convert()

# pylint: disable=line-too-long

# CHECK-LABEL: testSavedModelDebugInfo
# CHECK: error: 'tf.Betainc' op is neither a custom op nor a flex op
# CHECK:                                  attrs=attr_protos, op_def=op_def)
# CHECK:                                  ^
# CHECK: {{.*tensorflow/python/ops/gen_math_ops.py:[0-9]+:[0-9]+: note: called from}}
# CHECK:         "Betainc", a=a, b=b, x=x, name=name)
# CHECK:         ^
# CHECK: {{.*tensorflow/compiler/mlir/lite/tests/debuginfo/saved_model_error.py:[0-9]+:[0-9]+: note: called from}}
# CHECK:     y = tf.math.betainc(x, 0.5, 1.0)  # Not supported
# CHECK:     ^
# CHECK: <unknown>:0: error: failed while converting: 'main'

# pylint: enable=line-too-long


def main(argv):
  # [FUNCTION DOC]
  # Test driver. argv: command-line list; raises a usage error if it has more than the program
  # name. Runs the scenario; on any exception writes the test name and the exception text to
  # standard output. Nothing is printed if no exception occurs. Returns nothing.
  """test driver method writes the error message to stdout."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  try:
    TestGraphDebugInfo().testSavedModelDebugInfo()
  except Exception as e:  # pylint: disable=broad-except
    sys.stdout.write('testSavedModelDebugInfo')
    sys.stdout.write(str(e))


if __name__ == '__main__':
  app.run(main)
