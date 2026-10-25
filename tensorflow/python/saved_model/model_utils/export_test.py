# [MODULE DOC]
# Unit tests for the export_utils module of the SavedModel model_utils package: building the full
# mapping of signature definitions from export outputs and receiver tensors (with and without
# alternative receivers, serving-only filtering), generating timestamped and temporary export
# directory names, and choosing export outputs for a given mode key.
# Importing it has no side effects; running it as a script starts the test runner.
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
"""Tests for export utils."""

import os
import tempfile
import time

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.model_utils import export_output
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.mode_keys import KerasModeKeys


class ExportTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the export utilities (signature-definition building, export directory naming
  # and mode-based export output selection). No shared fixtures; each test creates its own
  # graph, tensors and temporary directories.

  def test_build_all_signature_defs_without_receiver_alternatives(self):
    # [FUNCTION DOC]
    # Scenario: in an explicit graph, a single string placeholder receiver and three export outputs
    # (regression as the default serving key, classification, prediction). Calls the builder with no
    # alternatives and asserts the result equals a mapping with the matching regression,
    # classification and prediction signatures, where the prediction input is named input.
    # Force the test to run in graph mode.
    # This tests a deprecated v1 API that depends on graph-only functions such
    # as build_tensor_info.
    with ops.Graph().as_default():
      receiver_tensor = array_ops.placeholder(dtypes.string)
      output_1 = constant_op.constant([1.])
      output_2 = constant_op.constant(["2"])
      output_3 = constant_op.constant(["3"])
      export_outputs = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              export_output.RegressionOutput(value=output_1),
          "head-2":
              export_output.ClassificationOutput(classes=output_2),
          "head-3":
              export_output.PredictOutput(outputs={"some_output_3": output_3}),
      }

      signature_defs = export_utils.build_all_signature_defs(
          receiver_tensor, export_outputs)

      expected_signature_defs = {
          "serving_default":
              signature_def_utils.regression_signature_def(
                  receiver_tensor, output_1),
          "head-2":
              signature_def_utils.classification_signature_def(
                  receiver_tensor, output_2, None),
          "head-3":
              signature_def_utils.predict_signature_def(
                  {"input": receiver_tensor}, {"some_output_3": output_3})
      }

      self.assertDictEqual(expected_signature_defs, signature_defs)

  def test_build_all_signature_defs_with_dict_alternatives(self):
    # [FUNCTION DOC]
    # Scenario: in an explicit graph, one default receiver plus two named alternative receivers
    # given as mappings (one with int and sparse inputs, one with a boolean input that other
    # outputs depend on). Five export outputs are supplied. Asserts the result contains the default
    # signatures, the prediction signature for each alternative that can serve it, and omits the
    # regression/classification signatures for the multi-input alternative and heads that depend
    # on inputs not fed by the receiver.
    # Force the test to run in graph mode.
    # This tests a deprecated v1 API that depends on graph-only functions such
    # as build_tensor_info.
    with ops.Graph().as_default():
      receiver_tensor = array_ops.placeholder(dtypes.string)

      receiver_tensors_alternative_1 = {
          "foo": array_ops.placeholder(dtypes.int64),
          "bar": array_ops.sparse_placeholder(dtypes.float32)
      }

      unfed_input = array_ops.placeholder(dtypes.bool)
      receiver_tensors_alternative_2 = {"unfed": unfed_input}

      receiver_tensors_alternatives = {
          "other": receiver_tensors_alternative_1,
          "with_unfed_input": receiver_tensors_alternative_2
      }

      output_1 = constant_op.constant([1.])
      output_2 = constant_op.constant(["2"])
      output_3 = constant_op.constant(["3"])
      output_4 = unfed_input
      output_5 = math_ops.logical_not(unfed_input)
      export_outputs = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              export_output.RegressionOutput(value=output_1),
          "head-2":
              export_output.ClassificationOutput(classes=output_2),
          "head-3":
              export_output.PredictOutput(outputs={"some_output_3": output_3}),
          "head-4":
              export_output.PredictOutput(outputs={"some_output_4": output_4}),
          "head-5":
              export_output.PredictOutput(outputs={"some_output_5": output_5}),
      }

      signature_defs = export_utils.build_all_signature_defs(
          receiver_tensor, export_outputs, receiver_tensors_alternatives)

      expected_signature_defs = {
          "serving_default":
              signature_def_utils.regression_signature_def(
                  receiver_tensor, output_1),
          "head-2":
              signature_def_utils.classification_signature_def(
                  receiver_tensor, output_2, None),
          "head-3":
              signature_def_utils.predict_signature_def(
                  {"input": receiver_tensor}, {"some_output_3": output_3}),
          "other:head-3":
              signature_def_utils.predict_signature_def(
                  receiver_tensors_alternative_1, {"some_output_3": output_3}),

          # Note that the alternatives 'other:serving_default' and
          # 'other:head-2' are invalid, because regression and classification
          # signatures must take a single string input.  Here we verify that
          # these invalid signatures are not included in the export_utils.

          # Similarly, we verify that 'head-4' and 'head-5', which depend on an
          # input that is not being fed as a receiver tensor, are also omitted.

          # All the three heads are present when that input is fed, however:
          "with_unfed_input:head-3":
              signature_def_utils.predict_signature_def(
                  receiver_tensors_alternative_2, {"some_output_3": output_3}),
          "with_unfed_input:head-4":
              signature_def_utils.predict_signature_def(
                  receiver_tensors_alternative_2, {"some_output_4": output_4}),
          "with_unfed_input:head-5":
              signature_def_utils.predict_signature_def(
                  receiver_tensors_alternative_2, {"some_output_5": output_5})
      }

      self.assertDictEqual(expected_signature_defs, signature_defs)

  def test_build_all_signature_defs_with_single_alternatives(self):
    # [FUNCTION DOC]
    # Scenario: in an explicit graph, alternative receivers are given as single tensors (dense int
    # and sparse float) instead of mappings. Asserts they are wrapped under the default receiver
    # name input and yield prediction signatures named with the alternative name and head name,
    # while invalid regression/classification alternatives are excluded.
    # Force the test to run in graph mode.
    # This tests a deprecated v1 API that depends on graph-only functions such
    # as build_tensor_info.
    with ops.Graph().as_default():
      receiver_tensor = array_ops.placeholder(dtypes.string)
      receiver_tensors_alternative_1 = array_ops.placeholder(dtypes.int64)
      receiver_tensors_alternative_2 = array_ops.sparse_placeholder(
          dtypes.float32)
      # Note we are passing single Tensors as values of
      # receiver_tensors_alternatives, where normally that is a dict.
      # In this case a dict will be created using the default receiver tensor
      # name "input".
      receiver_tensors_alternatives = {
          "other1": receiver_tensors_alternative_1,
          "other2": receiver_tensors_alternative_2
      }
      output_1 = constant_op.constant([1.])
      output_2 = constant_op.constant(["2"])
      output_3 = constant_op.constant(["3"])
      export_outputs = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              export_output.RegressionOutput(value=output_1),
          "head-2":
              export_output.ClassificationOutput(classes=output_2),
          "head-3":
              export_output.PredictOutput(outputs={"some_output_3": output_3}),
      }

      signature_defs = export_utils.build_all_signature_defs(
          receiver_tensor, export_outputs, receiver_tensors_alternatives)

      expected_signature_defs = {
          "serving_default":
              signature_def_utils.regression_signature_def(
                  receiver_tensor, output_1),
          "head-2":
              signature_def_utils.classification_signature_def(
                  receiver_tensor, output_2, None),
          "head-3":
              signature_def_utils.predict_signature_def(
                  {"input": receiver_tensor}, {"some_output_3": output_3}),
          "other1:head-3":
              signature_def_utils.predict_signature_def(
                  {"input": receiver_tensors_alternative_1},
                  {"some_output_3": output_3}),
          "other2:head-3":
              signature_def_utils.predict_signature_def(
                  {"input": receiver_tensors_alternative_2},
                  {"some_output_3": output_3})

          # Note that the alternatives 'other:serving_default' and
          # 'other:head-2' are invalid, because regression and classification
          # signatures must take a single string input.  Here we verify that
          # these invalid signatures are not included in the export_utils.
      }

      self.assertDictEqual(expected_signature_defs, signature_defs)

  def test_build_all_signature_defs_export_outputs_required(self):
    # [FUNCTION DOC]
    # Scenario: call the builder with a constant receiver and a null export outputs argument.
    # Expects a value error whose message starts by saying export outputs must be a mapping.
    receiver_tensor = constant_op.constant(["11"])

    with self.assertRaises(ValueError) as e:
      export_utils.build_all_signature_defs(receiver_tensor, None)

    self.assertTrue(
        str(e.exception).startswith("`export_outputs` must be a dict"))

  def test_get_timestamped_export_dir(self):
    # [FUNCTION DOC]
    # Scenario: request three timestamped export directories under a fresh temporary base,
    # sleeping two seconds between requests (real sleeps, so the test is slow). Asserts each final
    # path component is a ten-digit seconds-since-epoch timestamp and that the timestamps are
    # strictly increasing. Side effect: creates a temporary base directory name on disk.
    export_dir_base = tempfile.mkdtemp() + "export/"
    export_dir_1 = export_utils.get_timestamped_export_dir(
        export_dir_base)
    time.sleep(2)
    export_dir_2 = export_utils.get_timestamped_export_dir(
        export_dir_base)
    time.sleep(2)
    export_dir_3 = export_utils.get_timestamped_export_dir(
        export_dir_base)

    # Export directories should be named using a timestamp that is seconds
    # since epoch.  Such a timestamp is 10 digits long.
    time_1 = os.path.basename(export_dir_1)
    self.assertEqual(10, len(time_1))
    time_2 = os.path.basename(export_dir_2)
    self.assertEqual(10, len(time_2))
    time_3 = os.path.basename(export_dir_3)
    self.assertEqual(10, len(time_3))

    self.assertLess(int(time_1), int(time_2))
    self.assertLess(int(time_2), int(time_3))

  def test_get_temp_export_dir(self):
    # [FUNCTION DOC]
    # Scenario: derive a temporary export directory from a timestamped directory given first as a
    # text path and then as a byte-string path. Asserts both give a byte-string sibling path whose
    # last component is the timestamp prefixed with temp-.
    export_dir = os.path.join("tmp", "export", "1576013284")
    tmp_export_dir = export_utils.get_temp_export_dir(export_dir)
    self.assertEqual(tmp_export_dir,
                     os.path.join(b"tmp", b"export", b"temp-1576013284"))

    export_dir = os.path.join(b"tmp", b"export", b"1576013284")
    tmp_export_dir = export_utils.get_temp_export_dir(export_dir)
    self.assertEqual(tmp_export_dir,
                     os.path.join(b"tmp", b"export", b"temp-1576013284"))

  def test_build_all_signature_defs_serving_only(self):
    # [FUNCTION DOC]
    # Scenario: in an explicit graph, a receiver mapping with one string placeholder and export
    # outputs holding a serving prediction output and a train output. With the default serving-only
    # setting, asserts only the prediction signature is returned. With serving-only disabled,
    # asserts the training signature is also included.
    # Force the test to run in graph mode.
    # This tests a deprecated v1 API that depends on graph-only functions such
    # as build_tensor_info.
    with ops.Graph().as_default():
      receiver_tensor = {"input": array_ops.placeholder(dtypes.string)}
      output_1 = constant_op.constant([1.])
      export_outputs = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              export_output.PredictOutput(outputs=output_1),
          "train":
              export_output.TrainOutput(loss=output_1),
      }

      signature_defs = export_utils.build_all_signature_defs(
          receiver_tensor, export_outputs)

      expected_signature_defs = {
          "serving_default":
              signature_def_utils.predict_signature_def(receiver_tensor,
                                                        {"output": output_1})
      }

      self.assertDictEqual(expected_signature_defs, signature_defs)

      signature_defs = export_utils.build_all_signature_defs(
          receiver_tensor, export_outputs, serving_only=False)

      expected_signature_defs.update({
          "train":
              signature_def_utils.supervised_train_signature_def(
                  receiver_tensor, loss={"loss": output_1})
      })

      self.assertDictEqual(expected_signature_defs, signature_defs)

  def test_export_outputs_for_mode(self):
    # [FUNCTION DOC]
    # Scenario: predictions, loss and metrics built from constants. Asserts the train mode gives a
    # train output under the default train key, test mode an eval output under the default eval
    # key (both carrying the original predictions, loss and flattened metrics), and predict mode a
    # prediction output under the default serving key. Also asserts that user-supplied serving
    # outputs are returned for predict mode.
    predictions = {"predictions": constant_op.constant([1.])}
    loss = {"loss": constant_op.constant([2.])}
    metrics = {
        "metrics": (constant_op.constant([3.]), constant_op.constant([4.]))}
    expected_metrics = {
        "metrics/value": metrics["metrics"][0],
        "metrics/update_op": metrics["metrics"][1]
    }

    def _build_export_output(mode):
      # [FUNCTION DOC]
      # HELPER: wrapper for the export-outputs-for-mode test that fixes the predictions, loss and
      # metrics arguments and passes no serving outputs.
      # Takes a mode key; returns the mapping of signature key to export output for that mode.
      return export_utils.export_outputs_for_mode(
          mode, None, predictions, loss, metrics)

    ret = _build_export_output(KerasModeKeys.TRAIN)
    self.assertIn(signature_constants.DEFAULT_TRAIN_SIGNATURE_DEF_KEY, ret)
    export_out = ret[signature_constants.DEFAULT_TRAIN_SIGNATURE_DEF_KEY]
    self.assertIsInstance(export_out, export_output.TrainOutput)
    self.assertEqual(export_out.predictions, predictions)
    self.assertEqual(export_out.loss, loss)
    self.assertEqual(export_out.metrics, expected_metrics)

    ret = _build_export_output(KerasModeKeys.TEST)
    self.assertIn(signature_constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY, ret)
    export_out = ret[signature_constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY]
    self.assertIsInstance(export_out, export_output.EvalOutput)
    self.assertEqual(export_out.predictions, predictions)
    self.assertEqual(export_out.loss, loss)
    self.assertEqual(export_out.metrics, expected_metrics)

    ret = _build_export_output(KerasModeKeys.PREDICT)
    self.assertIn(signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY, ret)
    export_out = ret[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    self.assertIsInstance(export_out, export_output.PredictOutput)
    self.assertEqual(export_out.outputs, predictions)

    classes = constant_op.constant(["class5"])
    ret = export_utils.export_outputs_for_mode(
        KerasModeKeys.PREDICT,
        {"classify": export_output.ClassificationOutput(
            classes=classes)})
    self.assertIn("classify", ret)
    export_out = ret["classify"]
    self.assertIsInstance(export_out, export_output.ClassificationOutput)
    self.assertEqual(export_out.classes, classes)


if __name__ == "__main__":
  test.main()
