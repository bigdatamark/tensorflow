# [MODULE DOC]
# Unit tests for the SavedModel freezing step of the TFLite converter (the freeze function in
# the saved-model conversion module). Builds small SavedModels on the fly (simple add graphs,
# a two-input graph, a model with two meta graphs) and checks the returned tensor names, shapes
# and error messages. Runs through the TensorFlow test main entry point; no import-time effects.
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
"""TFLite SavedModel conversion test cases.

  - Tests converting simple SavedModel graph to TFLite FlatBuffer.
  - Tests converting simple SavedModel graph to frozen graph.
  - Tests converting MNIST SavedModel to TFLite FlatBuffer.
"""

import os
from tensorflow.lite.python import convert_saved_model
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


class FreezeSavedModelTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering freezing of SavedModels into frozen graphs. Fixtures are helper methods
  # that write SavedModels into the test temp directory and call the freeze function with
  # defaults for tag set (serving) and signature key (default serving signature).

  def _createSimpleSavedModel(self, shape):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the tests in this class.
    # Creates a SavedModel in a temp subdirectory holding one float32 placeholder of the given
    # shape (list of dims, may contain nothing for unknown) whose output is the input added to
    # itself, saved with simple save under input name x and output name y. Returns the directory.
    """Create a simple SavedModel on the fly."""
    saved_model_dir = os.path.join(self.get_temp_dir(), "simple_savedmodel")
    with session.Session() as sess:
      in_tensor = array_ops.placeholder(shape=shape, dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      inputs = {"x": in_tensor}
      outputs = {"y": out_tensor}
      saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return saved_model_dir

  def _createSavedModelTwoInputArrays(self, shape):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the tests in this class.
    # Creates a SavedModel (same temp directory name as the simple one) with two float32
    # placeholders of the given shape named inputB and inputA, output is their sum. Signature
    # inputs are x (inputB) and y (inputA), output z. Returns the directory path.
    """Create a simple SavedModel."""
    saved_model_dir = os.path.join(self.get_temp_dir(), "simple_savedmodel")
    with session.Session() as sess:
      in_tensor_1 = array_ops.placeholder(
          shape=shape, dtype=dtypes.float32, name="inputB")
      in_tensor_2 = array_ops.placeholder(
          shape=shape, dtype=dtypes.float32, name="inputA")
      out_tensor = in_tensor_1 + in_tensor_2
      inputs = {"x": in_tensor_1, "y": in_tensor_2}
      outputs = {"z": out_tensor}
      saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
    return saved_model_dir

  def _getArrayNames(self, tensors):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Returns the list of names of the given tensors.
    return [tensor.name for tensor in tensors]

  def _getArrayShapes(self, tensors):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Returns, for each tensor, its shape as a list of dimension
    # sizes, unwrapping old-style dimension objects to plain values (unknown stays null).
    dims = []
    for tensor in tensors:
      dim_tensor = []
      for dim in tensor.shape:
        if isinstance(dim, tensor_shape.Dimension):
          dim_tensor.append(dim.value)
        else:
          dim_tensor.append(dim)
      dims.append(dim_tensor)
    return dims

  def _convertSavedModel(self,
                         saved_model_dir,
                         input_arrays=None,
                         input_shapes=None,
                         output_arrays=None,
                         tag_set=None,
                         signature_key=None):
    # [FUNCTION DOC]
    # HELPER: wrapper around the freeze function used by the tests.
    # Takes the model directory plus optional input arrays, input shapes, output arrays, tag set
    # and signature key. Defaults the tag set to the serving tag and the signature key to the
    # default serving signature. Returns a triple: frozen graph definition, input tensors and
    # output tensors (the graph object returned by the function is dropped).
    if tag_set is None:
      tag_set = set([tag_constants.SERVING])
    if signature_key is None:
      signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    graph_def, in_tensors, out_tensors, _ = (
        convert_saved_model.freeze_saved_model(
            saved_model_dir=saved_model_dir,
            input_arrays=input_arrays,
            input_shapes=input_shapes,
            output_arrays=output_arrays,
            tag_set=tag_set,
            signature_key=signature_key))
    return graph_def, in_tensors, out_tensors

  def testSimpleSavedModel(self):
    # [FUNCTION DOC]
    # Scenario: simple model with fixed shape 1x16x16x3, converted with defaults. Expects output
    # tensor add:0, input tensor Placeholder:0 and input shape equal to the original shape.
    """Test a SavedModel."""
    saved_model_dir = self._createSimpleSavedModel(shape=[1, 16, 16, 3])
    _, in_tensors, out_tensors = self._convertSavedModel(saved_model_dir)

    self.assertEqual(self._getArrayNames(out_tensors), ["add:0"])
    self.assertEqual(self._getArrayNames(in_tensors), ["Placeholder:0"])
    self.assertEqual(self._getArrayShapes(in_tensors), [[1, 16, 16, 3]])

  def testSimpleSavedModelWithNoneBatchSizeInShape(self):
    # [FUNCTION DOC]
    # Scenario: model whose batch dimension is unknown. Converted with defaults; expects the same
    # tensor names as the simple case and the input shape keeps the unknown batch dimension.
    """Test a SavedModel with None in input tensor's shape."""
    saved_model_dir = self._createSimpleSavedModel(shape=[None, 16, 16, 3])
    _, in_tensors, out_tensors = self._convertSavedModel(saved_model_dir)

    self.assertEqual(self._getArrayNames(out_tensors), ["add:0"])
    self.assertEqual(self._getArrayNames(in_tensors), ["Placeholder:0"])
    self.assertEqual(self._getArrayShapes(in_tensors), [[None, 16, 16, 3]])

  def testSimpleSavedModelWithInvalidSignatureKey(self):
    # [FUNCTION DOC]
    # Scenario: convert with a signature key that does not exist. Expects a value error whose
    # message names the bad key and lists the only valid key, the default serving signature.
    """Test a SavedModel that fails due to an invalid signature_key."""
    saved_model_dir = self._createSimpleSavedModel(shape=[1, 16, 16, 3])
    with self.assertRaises(ValueError) as error:
      self._convertSavedModel(saved_model_dir, signature_key="invalid-key")
    self.assertEqual(
        "No 'invalid-key' in the SavedModel's SignatureDefs. "
        "Possible values are 'serving_default'.", str(error.exception))

  def testSimpleSavedModelWithInvalidOutputArray(self):
    # [FUNCTION DOC]
    # Scenario: user supplies an output array name not in the graph. Expects a value error with
    # the message that invalid tensors with that name were found.
    """Test a SavedModel that fails due to invalid output arrays."""
    saved_model_dir = self._createSimpleSavedModel(shape=[1, 16, 16, 3])
    with self.assertRaises(ValueError) as error:
      self._convertSavedModel(saved_model_dir, output_arrays=["invalid-output"])
    self.assertEqual("Invalid tensors 'invalid-output' were found.",
                     str(error.exception))

  def testSimpleSavedModelWithWrongInputArrays(self):
    # [FUNCTION DOC]
    # Scenario: invalid input arrays, first only an unknown name, then a mix of the valid name
    # Placeholder and an unknown name. Both cases expect a value error naming only the invalid
    # tensor.
    """Test a SavedModel that fails due to invalid input arrays."""
    saved_model_dir = self._createSimpleSavedModel(shape=[1, 16, 16, 3])

    # Check invalid input_arrays.
    with self.assertRaises(ValueError) as error:
      self._convertSavedModel(saved_model_dir, input_arrays=["invalid-input"])
    self.assertEqual("Invalid tensors 'invalid-input' were found.",
                     str(error.exception))

    # Check valid and invalid input_arrays.
    with self.assertRaises(ValueError) as error:
      self._convertSavedModel(
          saved_model_dir, input_arrays=["Placeholder", "invalid-input"])
    self.assertEqual("Invalid tensors 'invalid-input' were found.",
                     str(error.exception))

  def testSimpleSavedModelWithCorrectArrays(self):
    # [FUNCTION DOC]
    # Scenario: unknown-batch model with explicit valid input array Placeholder and output array
    # add. Expects output add:0, input Placeholder:0 and the shape with unknown batch preserved.
    """Test a SavedModel with correct input_arrays and output_arrays."""
    saved_model_dir = self._createSimpleSavedModel(shape=[None, 16, 16, 3])
    _, in_tensors, out_tensors = self._convertSavedModel(
        saved_model_dir=saved_model_dir,
        input_arrays=["Placeholder"],
        output_arrays=["add"])

    self.assertEqual(self._getArrayNames(out_tensors), ["add:0"])
    self.assertEqual(self._getArrayNames(in_tensors), ["Placeholder:0"])
    self.assertEqual(self._getArrayShapes(in_tensors), [[None, 16, 16, 3]])

  def testSimpleSavedModelWithCorrectInputArrays(self):
    # [FUNCTION DOC]
    # Scenario: fixed-shape model with a valid input array and an explicit matching input shape
    # mapping. Expects the usual tensor names and the given shape.
    """Test a SavedModel with correct input_arrays and input_shapes."""
    saved_model_dir = self._createSimpleSavedModel(shape=[1, 16, 16, 3])
    _, in_tensors, out_tensors = self._convertSavedModel(
        saved_model_dir=saved_model_dir,
        input_arrays=["Placeholder"],
        input_shapes={"Placeholder": [1, 16, 16, 3]})

    self.assertEqual(self._getArrayNames(out_tensors), ["add:0"])
    self.assertEqual(self._getArrayNames(in_tensors), ["Placeholder:0"])
    self.assertEqual(self._getArrayShapes(in_tensors), [[1, 16, 16, 3]])

  def testTwoInputArrays(self):
    # [FUNCTION DOC]
    # Scenario: two-input model, user lists inputs as inputB then inputA. Expects a single output
    # add:0 and input tensors returned sorted by name (inputA then inputB), both with the
    # original shape.
    """Test a simple SavedModel."""
    saved_model_dir = self._createSavedModelTwoInputArrays(shape=[1, 16, 16, 3])

    _, in_tensors, out_tensors = self._convertSavedModel(
        saved_model_dir=saved_model_dir, input_arrays=["inputB", "inputA"])

    self.assertEqual(self._getArrayNames(out_tensors), ["add:0"])
    self.assertEqual(self._getArrayNames(in_tensors), ["inputA:0", "inputB:0"])
    self.assertEqual(
        self._getArrayShapes(in_tensors), [[1, 16, 16, 3], [1, 16, 16, 3]])

  def testSubsetInputArrays(self):
    # [FUNCTION DOC]
    # Scenario: two-input model but only inputA is requested, once with an explicit input shape
    # and once without. Both cases expect output add:0, a single input inputA:0 and shape
    # 1x16x16x3 (inferred from the graph when not given).
    """Test a SavedModel with a subset of the input array names of the model."""
    saved_model_dir = self._createSavedModelTwoInputArrays(shape=[1, 16, 16, 3])

    # Check case where input shape is given.
    _, in_tensors, out_tensors = self._convertSavedModel(
        saved_model_dir=saved_model_dir,
        input_arrays=["inputA"],
        input_shapes={"inputA": [1, 16, 16, 3]})

    self.assertEqual(self._getArrayNames(out_tensors), ["add:0"])
    self.assertEqual(self._getArrayNames(in_tensors), ["inputA:0"])
    self.assertEqual(self._getArrayShapes(in_tensors), [[1, 16, 16, 3]])

    # Check case where input shape is None.
    _, in_tensors, out_tensors = self._convertSavedModel(
        saved_model_dir=saved_model_dir, input_arrays=["inputA"])

    self.assertEqual(self._getArrayNames(out_tensors), ["add:0"])
    self.assertEqual(self._getArrayNames(in_tensors), ["inputA:0"])
    self.assertEqual(self._getArrayShapes(in_tensors), [[1, 16, 16, 3]])

  def testMultipleMetaGraphDef(self):
    # [FUNCTION DOC]
    # Scenario: a SavedModel saved with two meta graphs: one tagged serving plus an additional
    # tag containing the add graph with a predict signature, and a second tagged tflite. Converts
    # selecting the first tag set. Expects output add:0, input Placeholder:0, shape 1x28x28.
    """Test saved model with multiple MetaGraphDefs."""
    saved_model_dir = os.path.join(self.get_temp_dir(), "savedmodel_two_mgd")
    builder = saved_model.builder.SavedModelBuilder(saved_model_dir)
    with session.Session(graph=ops.Graph()) as sess:
      # MetaGraphDef 1
      in_tensor = array_ops.placeholder(shape=[1, 28, 28], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sig_input_tensor = saved_model.utils.build_tensor_info(in_tensor)
      sig_input_tensor_signature = {"x": sig_input_tensor}
      sig_output_tensor = saved_model.utils.build_tensor_info(out_tensor)
      sig_output_tensor_signature = {"y": sig_output_tensor}
      predict_signature_def = (
          saved_model.signature_def_utils.build_signature_def(
              sig_input_tensor_signature, sig_output_tensor_signature,
              saved_model.signature_constants.PREDICT_METHOD_NAME))
      signature_def_map = {
          saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              predict_signature_def
      }
      builder.add_meta_graph_and_variables(
          sess,
          tags=[saved_model.tag_constants.SERVING, "additional_test_tag"],
          signature_def_map=signature_def_map)

      # MetaGraphDef 2
      builder.add_meta_graph(tags=["tflite"])
      builder.save(True)

    # Convert to tflite
    _, in_tensors, out_tensors = self._convertSavedModel(
        saved_model_dir=saved_model_dir,
        tag_set=set([saved_model.tag_constants.SERVING, "additional_test_tag"]))

    self.assertEqual(self._getArrayNames(out_tensors), ["add:0"])
    self.assertEqual(self._getArrayNames(in_tensors), ["Placeholder:0"])
    self.assertEqual(self._getArrayShapes(in_tensors), [[1, 28, 28]])


if __name__ == "__main__":
  test.main()
