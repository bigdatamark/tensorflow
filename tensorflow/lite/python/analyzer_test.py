# [MODULE DOC]
# Tests for the TFLite model analyzer (the analyzer module). Each test runs the analyzer with
# standard output replaced by an in-memory buffer and checks the captured text report or MLIR
# dump against expected fragments. Uses model files from the shared test data directory and
# models converted on the fly from TensorFlow functions or saved models.
# Runs the TensorFlow test main when executed as a program.
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
"""Tests for analyzer package."""

import io
import sys
import tempfile

import tensorflow as tf

from tensorflow.lite.python import analyzer
from tensorflow.lite.python import lite
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.trackable import autotrackable


class AnalyzerTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the model analyzer. Covers the text report and MLIR output for model files and
  # in-memory flatbuffers, signature definitions, models without inputs, and an einsum model.
  # No shared fixtures; each test captures standard output itself.

  def testTxt(self):
    # [FUNCTION DOC]
    # Analyzes the add test model file in text mode and asserts the subgraph header and the two
    # add ops with their tensor numbers appear, and that no GPU compatibility message appears.
    model_path = resource_loader.get_path_to_datafile('../testdata/add.bin')
    mock_stdout = io.StringIO()
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      analyzer.ModelAnalyzer.analyze(model_path=model_path)
    txt = mock_stdout.getvalue()
    self.assertIn('Subgraph#0(T#1) -> [T#2]', txt)
    self.assertIn('Op#0 ADD(T#1, T#1) -> [T#0]', txt)
    self.assertIn('Op#1 ADD(T#0, T#1) -> [T#2]', txt)
    self.assertNotIn('Your model looks compatible with GPU delegate', txt)

  def testMlir(self):
    # [FUNCTION DOC]
    # Analyzes the add test model file with the MLIR option and asserts the output contains the
    # main function signature with saved-model attributes, both add operations and the return.
    model_path = resource_loader.get_path_to_datafile('../testdata/add.bin')
    mock_stdout = io.StringIO()
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      analyzer.ModelAnalyzer.analyze(
          model_path=model_path, experimental_use_mlir=True)
    mlir = mock_stdout.getvalue()
    self.assertIn(
        'func @main(%arg0: tensor<1x8x8x3xf32> '
        '{tf_saved_model.index_path = ["a"]}) -> '
        '(tensor<1x8x8x3xf32> {tf_saved_model.index_path = ["x"]}) attributes '
        '{tf.entry_function = {inputs = "input", outputs = "output"}, '
        'tf_saved_model.exported_names = ["serving_default"]}', mlir)
    self.assertIn(
        '%0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : '
        'tensor<1x8x8x3xf32>', mlir)
    self.assertIn(
        '%1 = tfl.add %0, %arg0 {fused_activation_function = "NONE"} : '
        'tensor<1x8x8x3xf32>', mlir)
    self.assertIn('return %1 : tensor<1x8x8x3xf32>', mlir)

  def testMlirHugeConst(self):
    # [FUNCTION DOC]
    # Analyzes a convolution model with a very large constant using the MLIR option and asserts
    # the constant is printed as an elided dense resource rather than inline data.
    model_path = resource_loader.get_path_to_datafile(
        '../testdata/conv_huge_im2col.bin')
    mock_stdout = io.StringIO()
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      analyzer.ModelAnalyzer.analyze(
          model_path=model_path, experimental_use_mlir=True)
    mlir = mock_stdout.getvalue()
    self.assertIn(
        '%1 = "tfl.pseudo_const"() <{value = dense_resource<__elided__> : '
        'tensor<3x3x3x8xf32>}> : () -> tensor<3x3x3x8xf32>', mlir)

  def testTxtWithFlatBufferModel(self):
    # [FUNCTION DOC]
    # Converts a function computing x plus cosine of x (unknown-length float vector input) to a
    # flatbuffer, analyzes it from memory in text mode and asserts the subgraph header, the
    # cosine op and the add op are reported.

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def func(x):
      # [FUNCTION DOC]
      # Traced function being converted: returns the input plus its cosine.
      return x + tf.cos(x)

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func.get_concrete_function()], func)
    fb_model = converter.convert()
    mock_stdout = io.StringIO()
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      analyzer.ModelAnalyzer.analyze(model_content=fb_model)
    txt = mock_stdout.getvalue()
    self.assertIn('Subgraph#0 main(T#0) -> [T#2]', txt)
    self.assertIn('Op#0 COS(T#0) -> [T#1]', txt)
    self.assertIn('Op#1 ADD(T#0, T#1) -> [T#2]', txt)

  def testMlirWithFlatBufferModel(self):
    # [FUNCTION DOC]
    # Same model as the text flatbuffer test, but analyzed with the MLIR option; asserts the main
    # function signature with dynamic shapes, the cosine and add operations and the return.

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def func(x):
      # [FUNCTION DOC]
      # Traced function being converted: returns the input plus its cosine.
      return x + tf.cos(x)

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func.get_concrete_function()], func)
    fb_model = converter.convert()
    mock_stdout = io.StringIO()
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      analyzer.ModelAnalyzer.analyze(
          model_content=fb_model, experimental_use_mlir=True)
    mlir = mock_stdout.getvalue()
    self.assertIn('func @main(%arg0: tensor<?xf32>) -> tensor<?xf32>', mlir)
    self.assertIn('%0 = "tfl.cos"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>',
                  mlir)
    self.assertIn(
        '%1 = tfl.add %arg0, %0 {fused_activation_function = "NONE"} : '
        'tensor<?xf32>', mlir)
    self.assertIn('return %1 : tensor<?xf32', mlir)

  def testTxtSignatureDefs(self):
    # [FUNCTION DOC]
    # Saves a model with two signatures (add and subtract, each with two float inputs and one
    # named output) to a temporary directory, converts it to a flatbuffer, analyzes it in text
    # mode and asserts the report lists two signature definitions with their keys and the
    # tensor numbers of inputs and outputs.
    with tempfile.TemporaryDirectory() as tmp_dir:

      @tf.function(input_signature=[
          tf.TensorSpec(shape=None, dtype=tf.float32),
          tf.TensorSpec(shape=None, dtype=tf.float32)
      ])
      def add(a, b):
        # [FUNCTION DOC]
        # Signature function: returns a mapping with the name add_result holding the sum.
        return {'add_result': tf.add(a, b)}

      @tf.function(input_signature=[
          tf.TensorSpec(shape=None, dtype=tf.float32),
          tf.TensorSpec(shape=None, dtype=tf.float32)
      ])
      def sub(x, y):
        # [FUNCTION DOC]
        # Signature function: returns a mapping with the name sub_result holding the difference.
        return {'sub_result': tf.subtract(x, y)}

      root = autotrackable.AutoTrackable()
      root.f1 = add.get_concrete_function()
      root.f2 = sub.get_concrete_function()

      tf.saved_model.save(
          root, tmp_dir, signatures={
              'add': root.f1,
              'sub': root.f2
          })

      converter = lite.TFLiteConverterV2.from_saved_model(tmp_dir)
      fb_model = converter.convert()
      mock_stdout = io.StringIO()
      with test.mock.patch.object(sys, 'stdout', mock_stdout):
        analyzer.ModelAnalyzer.analyze(model_content=fb_model)
      txt = mock_stdout.getvalue()
      self.assertIn("Your TFLite model has '2' signature_def(s).", txt)
      self.assertIn("Signature#0 key: 'add'", txt)
      self.assertIn("  'a' : T#1", txt)
      self.assertIn("  'b' : T#0", txt)
      self.assertIn("  'add_result' : T#2", txt)
      self.assertIn("Signature#1 key: 'sub'", txt)
      self.assertIn("  'x' : T#1_1", txt)
      self.assertIn("  'y' : T#1_0", txt)
      self.assertIn("  'sub_result' : T#1_2", txt)

  def testTxtWithoutInput(self):
    # [FUNCTION DOC]
    # Converts a function with no inputs (cosine of a constant) and asserts the text report shows
    # a main subgraph with empty input list and one output tensor.

    @tf.function()
    def func():
      # [FUNCTION DOC]
      # Traced function being converted: returns the cosine of a constant.
      return tf.cos(1.0)

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func.get_concrete_function()], func)
    fb_model = converter.convert()
    mock_stdout = io.StringIO()
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      analyzer.ModelAnalyzer.analyze(model_content=fb_model)
    txt = mock_stdout.getvalue()
    self.assertIn('Subgraph#0 main() -> [T#0]', txt)

  def testTxtWithEinsum(self):
    # [FUNCTION DOC]
    # Converts an einsum of two fixed-shape float inputs with batch matmul unfolding enabled,
    # analyzes in text mode and asserts the expected reshape, transpose, fully connected and
    # reshape ops plus the constant tensors holding shape and permutation data.

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 100, 512], dtype=tf.float32),
        tf.TensorSpec(shape=[512, 8, 64], dtype=tf.float32)
    ])
    def func(lhs, rhs):
      # [FUNCTION DOC]
      # Traced function being converted: einsum contracting the last axis of the first input
      # with the first axis of the second.
      return tf.einsum('ABD,DNH->ABNH', lhs, rhs)

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func.get_concrete_function()], func)
    converter.unfold_batchmatmul = True
    fb_model = converter.convert()
    mock_stdout = io.StringIO()
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      analyzer.ModelAnalyzer.analyze(model_content=fb_model)
    txt = mock_stdout.getvalue()
    self.assertIn('Op#0 RESHAPE(T#1, T#4[512, 512]) -> [T#5]', txt)
    self.assertIn('Op#1 TRANSPOSE(T#5, T#3[1, 0]) -> [T#6]', txt)
    self.assertIn('Op#2 FULLY_CONNECTED(T#0, T#6, T#-1) -> [T#7]', txt)
    self.assertIn('Op#3 RESHAPE(T#7, T#2[1, 100, 8, 64]) -> [T#8]', txt)
    self.assertIn(
        'T#2(einsum/Einsum) shape:[4], type:INT32 RO 16 bytes, '
        'buffer: 3, data:[1, 100, 8, 64]', txt)
    self.assertIn(
        'T#3(einsum/Einsum2) shape:[2], type:INT32 RO 8 bytes, '
        'buffer: 4, data:[1, 0]', txt)
    self.assertIn(
        'T#4(einsum/Einsum3) shape:[2], type:INT32 RO 8 bytes, '
        'buffer: 5, data:[512, 512]', txt)


if __name__ == '__main__':
  test.main()
