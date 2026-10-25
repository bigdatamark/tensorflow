# [MODULE DOC]
# Unit tests for the export output classes (RegressionOutput, ClassificationOutput, PredictOutput,
# the supervised TrainOutput/EvalOutput family) found in the model_utils export_output module.
# The tests check argument validation (wrong dtypes, missing arguments, bad keys), the signature
# definitions produced from the outputs, and key-prefixing/wrapping behavior for supervised outputs.
# Importing this module has no side effects; running it as a script starts the test runner.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for export."""

from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import metrics as metrics_module
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model.model_utils import export_output as export_output_lib


class ExportOutputTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the regression, classification and prediction export output classes:
  # constructor validation and conversion into standardized signature definitions. No shared
  # fixtures; each test builds its own placeholders inside a graph-mode context.

  def test_regress_value_must_be_float(self):
    # [FUNCTION DOC]
    # Scenario: in graph mode, build a string-typed placeholder and pass it as the value of a
    # regression output. Expects a value error saying the regression value must be a float32 tensor.
    with context.graph_mode():
      value = array_ops.placeholder(dtypes.string, 1, name='output-tensor-1')
      with self.assertRaisesRegex(
          ValueError, 'Regression output value must be a float32 Tensor'):
        export_output_lib.RegressionOutput(value)

  def test_classify_classes_must_be_strings(self):
    # [FUNCTION DOC]
    # Scenario: in graph mode, pass a float32 placeholder as the classes of a classification
    # output. Expects a value error stating the classes must be a string tensor.
    with context.graph_mode():
      classes = array_ops.placeholder(dtypes.float32, 1, name='output-tensor-1')
      with self.assertRaisesRegex(
          ValueError, 'Classification classes must be a string Tensor'):
        export_output_lib.ClassificationOutput(classes=classes)

  def test_classify_scores_must_be_float(self):
    # [FUNCTION DOC]
    # Scenario: in graph mode, pass a string placeholder as the scores of a classification
    # output. Expects a value error stating the scores must be a float32 tensor.
    with context.graph_mode():
      scores = array_ops.placeholder(dtypes.string, 1, name='output-tensor-1')
      with self.assertRaisesRegex(
          ValueError, 'Classification scores must be a float32 Tensor'):
        export_output_lib.ClassificationOutput(scores=scores)

  def test_classify_requires_classes_or_scores(self):
    # [FUNCTION DOC]
    # Scenario: construct a classification output with neither classes nor scores. Expects a
    # value error saying a classification output cannot be created with empty arguments.
    with self.assertRaisesRegex(
        ValueError,
        'Cannot create a ClassificationOutput with empty arguments'):
      export_output_lib.ClassificationOutput()

  def test_build_standardized_signature_def_regression(self):
    # [FUNCTION DOC]
    # Scenario: a regression output wrapping a float32 placeholder is converted to a signature
    # definition using a one-entry mapping of string input placeholders. Asserts the result equals
    # a hand-built signature with the standard regress input/output keys, tensor names, dtypes,
    # shape of size one, and the regress method name.
    with context.graph_mode():
      input_tensors = {
          'input-1':
              array_ops.placeholder(
                  dtypes.string, 1, name='input-tensor-1')
      }
      value = array_ops.placeholder(dtypes.float32, 1, name='output-tensor-1')

      export_output = export_output_lib.RegressionOutput(value)
      actual_signature_def = export_output.as_signature_def(input_tensors)

      expected_signature_def = meta_graph_pb2.SignatureDef()
      shape = tensor_shape_pb2.TensorShapeProto(
          dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
      dtype_float = types_pb2.DataType.Value('DT_FLOAT')
      dtype_string = types_pb2.DataType.Value('DT_STRING')
      expected_signature_def.inputs[
          signature_constants.REGRESS_INPUTS].CopyFrom(
              meta_graph_pb2.TensorInfo(name='input-tensor-1:0',
                                        dtype=dtype_string,
                                        tensor_shape=shape))
      expected_signature_def.outputs[
          signature_constants.REGRESS_OUTPUTS].CopyFrom(
              meta_graph_pb2.TensorInfo(name='output-tensor-1:0',
                                        dtype=dtype_float,
                                        tensor_shape=shape))

      expected_signature_def.method_name = (
          signature_constants.REGRESS_METHOD_NAME)
      self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classify_classes_only(self):
    # [FUNCTION DOC]
    # Scenario: a classification output built with only a string classes tensor is converted to a
    # signature definition. Asserts it equals an expected signature holding the classify input and
    # only the classes output, with the classify method name.
    """Tests classification with one output tensor."""
    with context.graph_mode():
      input_tensors = {
          'input-1':
              array_ops.placeholder(
                  dtypes.string, 1, name='input-tensor-1')
      }
      classes = array_ops.placeholder(dtypes.string, 1, name='output-tensor-1')

      export_output = export_output_lib.ClassificationOutput(classes=classes)
      actual_signature_def = export_output.as_signature_def(input_tensors)

      expected_signature_def = meta_graph_pb2.SignatureDef()
      shape = tensor_shape_pb2.TensorShapeProto(
          dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
      dtype_string = types_pb2.DataType.Value('DT_STRING')
      expected_signature_def.inputs[
          signature_constants.CLASSIFY_INPUTS].CopyFrom(
              meta_graph_pb2.TensorInfo(name='input-tensor-1:0',
                                        dtype=dtype_string,
                                        tensor_shape=shape))
      expected_signature_def.outputs[
          signature_constants.CLASSIFY_OUTPUT_CLASSES].CopyFrom(
              meta_graph_pb2.TensorInfo(name='output-tensor-1:0',
                                        dtype=dtype_string,
                                        tensor_shape=shape))

      expected_signature_def.method_name = (
          signature_constants.CLASSIFY_METHOD_NAME)
      self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classify_both(self):
    # [FUNCTION DOC]
    # Scenario: a classification output built with both classes (string) and scores (float32) is
    # converted to a signature definition. Asserts the expected signature has the classify input,
    # both the classes and scores outputs, and the classify method name.
    """Tests multiple output tensors that include classes and scores."""
    with context.graph_mode():
      input_tensors = {
          'input-1':
              array_ops.placeholder(
                  dtypes.string, 1, name='input-tensor-1')
      }
      classes = array_ops.placeholder(dtypes.string, 1,
                                      name='output-tensor-classes')
      scores = array_ops.placeholder(dtypes.float32, 1,
                                     name='output-tensor-scores')

      export_output = export_output_lib.ClassificationOutput(
          scores=scores, classes=classes)
      actual_signature_def = export_output.as_signature_def(input_tensors)

      expected_signature_def = meta_graph_pb2.SignatureDef()
      shape = tensor_shape_pb2.TensorShapeProto(
          dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
      dtype_float = types_pb2.DataType.Value('DT_FLOAT')
      dtype_string = types_pb2.DataType.Value('DT_STRING')
      expected_signature_def.inputs[
          signature_constants.CLASSIFY_INPUTS].CopyFrom(
              meta_graph_pb2.TensorInfo(name='input-tensor-1:0',
                                        dtype=dtype_string,
                                        tensor_shape=shape))
      expected_signature_def.outputs[
          signature_constants.CLASSIFY_OUTPUT_CLASSES].CopyFrom(
              meta_graph_pb2.TensorInfo(name='output-tensor-classes:0',
                                        dtype=dtype_string,
                                        tensor_shape=shape))
      expected_signature_def.outputs[
          signature_constants.CLASSIFY_OUTPUT_SCORES].CopyFrom(
              meta_graph_pb2.TensorInfo(name='output-tensor-scores:0',
                                        dtype=dtype_float,
                                        tensor_shape=shape))

      expected_signature_def.method_name = (
          signature_constants.CLASSIFY_METHOD_NAME)
      self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classify_scores_only(self):
    # [FUNCTION DOC]
    # Scenario: a classification output built with only a float32 scores tensor is converted to a
    # signature definition. Asserts it equals an expected signature holding the classify input and
    # only the scores output, with the classify method name.
    """Tests classification without classes tensor."""
    with context.graph_mode():
      input_tensors = {
          'input-1':
              array_ops.placeholder(
                  dtypes.string, 1, name='input-tensor-1')
      }

      scores = array_ops.placeholder(dtypes.float32, 1,
                                     name='output-tensor-scores')

      export_output = export_output_lib.ClassificationOutput(
          scores=scores)
      actual_signature_def = export_output.as_signature_def(input_tensors)

      expected_signature_def = meta_graph_pb2.SignatureDef()
      shape = tensor_shape_pb2.TensorShapeProto(
          dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
      dtype_float = types_pb2.DataType.Value('DT_FLOAT')
      dtype_string = types_pb2.DataType.Value('DT_STRING')
      expected_signature_def.inputs[
          signature_constants.CLASSIFY_INPUTS].CopyFrom(
              meta_graph_pb2.TensorInfo(name='input-tensor-1:0',
                                        dtype=dtype_string,
                                        tensor_shape=shape))
      expected_signature_def.outputs[
          signature_constants.CLASSIFY_OUTPUT_SCORES].CopyFrom(
              meta_graph_pb2.TensorInfo(name='output-tensor-scores:0',
                                        dtype=dtype_float,
                                        tensor_shape=shape))

      expected_signature_def.method_name = (
          signature_constants.CLASSIFY_METHOD_NAME)
      self.assertEqual(actual_signature_def, expected_signature_def)

  def test_predict_outputs_valid(self):
    # [FUNCTION DOC]
    # Scenario: construct a prediction output from a mapping of string keys (including a unicode
    # key) to tensors, then from a single bare tensor. Expects both constructions to succeed
    # without raising; nothing else is asserted.
    """Tests that no errors are raised when provided outputs are valid."""
    outputs = {
        'output0': constant_op.constant([0]),
        u'output1': constant_op.constant(['foo']),
    }
    export_output_lib.PredictOutput(outputs)

    # Single Tensor is OK too
    export_output_lib.PredictOutput(constant_op.constant([0]))

  def test_predict_outputs_invalid(self):
    # [FUNCTION DOC]
    # Scenario: construct a prediction output with a non-string key, then with a sparse tensor as
    # a value. Expects value errors saying the key must be a string and the value must be a tensor.
    with self.assertRaisesRegex(ValueError,
                                'Prediction output key must be a string'):
      export_output_lib.PredictOutput({1: constant_op.constant([0])})

    with self.assertRaisesRegex(ValueError,
                                'Prediction output value must be a Tensor'):
      export_output_lib.PredictOutput({
          'prediction1': sparse_tensor.SparseTensor(
              indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
      })


class MockSupervisedOutput(export_output_lib._SupervisedOutput):
  # [CLASS DOC]
  # HELPER: minimal concrete subclass of the abstract supervised output so tests can exercise the
  # base-class constructor and key-normalization logic directly.
  # Its signature-definition factory does nothing and returns nothing.
  """So that we can test the abstract class methods directly."""

  def _get_signature_def_fn(self):
    # [FUNCTION DOC]
    # HELPER: stub implementing the abstract hook of the supervised output base class.
    # Does nothing and returns nothing (None); never used to build a signature in these tests.
    pass


class SupervisedOutputTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the supervised output family: the abstract base behavior via the mock subclass
  # (loss/predictions/metrics validation and key prefixing) plus train and eval signature
  # definitions and wrapping of non-tensor metric update ops. Tests build their own graph-mode data.

  def test_supervised_outputs_valid(self):
    # [FUNCTION DOC]
    # Scenario: build the mock supervised output from mappings of loss, predictions and metrics
    # (value/update-op pairs). Asserts keys are prefixed with loss/, predictions/ and metrics/ and
    # that the original tensors are preserved. Then passes single (non-mapping) values and asserts
    # the unprefixed-name keys loss and predictions are used and metrics are still prefixed.
    """Tests that no errors are raised when provided outputs are valid."""
    with context.graph_mode():
      loss = {'my_loss': constant_op.constant([0])}
      predictions = {u'output1': constant_op.constant(['foo'])}
      mean, update_op = metrics_module.mean_tensor(constant_op.constant([0]))
      metrics = {
          'metrics': (mean, update_op),
          'metrics2': (constant_op.constant([0]), constant_op.constant([10]))
      }

      outputter = MockSupervisedOutput(loss, predictions, metrics)
      self.assertEqual(outputter.loss['loss/my_loss'], loss['my_loss'])
      self.assertEqual(
          outputter.predictions['predictions/output1'], predictions['output1'])
      self.assertEqual(outputter.metrics['metrics/update_op'].name,
                       'mean/update_op:0')
      self.assertEqual(
          outputter.metrics['metrics2/update_op'], metrics['metrics2'][1])

      # Single Tensor is OK too
      outputter = MockSupervisedOutput(
          loss['my_loss'], predictions['output1'], metrics['metrics'])
      self.assertEqual(outputter.loss, {'loss': loss['my_loss']})
      self.assertEqual(
          outputter.predictions, {'predictions': predictions['output1']})
      self.assertEqual(outputter.metrics['metrics/update_op'].name,
                       'mean/update_op:0')

  def test_supervised_outputs_none(self):
    # [FUNCTION DOC]
    # Scenario: build the mock supervised output with only a loss tensor and nothing for
    # predictions and metrics. Asserts the loss mapping has one entry and the other two are null.
    outputter = MockSupervisedOutput(
        constant_op.constant([0]), None, None)
    self.assertLen(outputter.loss, 1)
    self.assertIsNone(outputter.predictions)
    self.assertIsNone(outputter.metrics)

  def test_supervised_outputs_invalid(self):
    # [FUNCTION DOC]
    # Scenario: build the mock supervised output with invalid pieces: a list as predictions, a
    # string as loss, a tuple of numbers as metrics, and a non-string loss key. Expects value
    # errors whose messages mention the predictions/loss/metrics value or key requirement.
    with self.assertRaisesRegex(ValueError, 'predictions output value must'):
      MockSupervisedOutput(constant_op.constant([0]), [3], None)
    with self.assertRaisesRegex(ValueError, 'loss output value must'):
      MockSupervisedOutput('str', None, None)
    with self.assertRaisesRegex(ValueError, 'metrics output value must'):
      MockSupervisedOutput(None, None, (15.3, 4))
    with self.assertRaisesRegex(ValueError, 'loss output key must'):
      MockSupervisedOutput({25: 'Tensor'}, None, None)

  def test_supervised_outputs_tuples(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is copy-pasted from the valid-outputs test; this test actually
    # checks tuple-valued keys being joined with slashes, not just absence of errors.
    # Scenario: loss, predictions and metrics use tuple keys. Asserts the resulting mapping keys are
    # the slash-joined, prefixed names, with each metric yielding both a value and an update-op key.
    """Tests that no errors are raised when provided outputs are valid."""
    with context.graph_mode():
      loss = {('my', 'loss'): constant_op.constant([0])}
      predictions = {(u'output1', '2'): constant_op.constant(['foo'])}
      mean, update_op = metrics_module.mean_tensor(constant_op.constant([0]))
      metrics = {
          ('metrics', '1'): (mean, update_op),
          ('metrics', '2'): (constant_op.constant([0]),
                             constant_op.constant([10]))
      }

      outputter = MockSupervisedOutput(loss, predictions, metrics)
      self.assertEqual(set(outputter.loss.keys()), set(['loss/my/loss']))
      self.assertEqual(set(outputter.predictions.keys()),
                       set(['predictions/output1/2']))
      self.assertEqual(
          set(outputter.metrics.keys()),
          set([
              'metrics/1/value', 'metrics/1/update_op', 'metrics/2/value',
              'metrics/2/update_op'
          ]))

  def test_supervised_outputs_no_prepend(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is copy-pasted from the valid-outputs test; this test actually
    # checks that keys already equal to loss or predictions are not prefixed a second time.
    # Scenario: single-key mappings named loss and predictions, plus metrics with ordinary names.
    # Asserts loss and predictions keep their names and metrics get value/update_op suffixes.
    """Tests that no errors are raised when provided outputs are valid."""
    with context.graph_mode():
      loss = {'loss': constant_op.constant([0])}
      predictions = {u'predictions': constant_op.constant(['foo'])}
      mean, update_op = metrics_module.mean_tensor(constant_op.constant([0]))
      metrics = {
          'metrics_1': (mean, update_op),
          'metrics_2': (constant_op.constant([0]), constant_op.constant([10]))
      }

      outputter = MockSupervisedOutput(loss, predictions, metrics)
      self.assertEqual(set(outputter.loss.keys()), set(['loss']))
      self.assertEqual(set(outputter.predictions.keys()), set(['predictions']))
      self.assertEqual(
          set(outputter.metrics.keys()),
          set([
              'metrics_1/value', 'metrics_1/update_op', 'metrics_2/update_op',
              'metrics_2/value'
          ]))

  def test_train_signature_def(self):
    # [FUNCTION DOC]
    # Scenario: a train output built from loss, predictions and metrics is converted to a
    # signature definition using a mapping of features and labels constants as receiver tensors.
    # Asserts outputs contain the loss, metric value and prediction keys and the inputs contain
    # the features key.
    with context.graph_mode():
      loss = {'my_loss': constant_op.constant([0])}
      predictions = {u'output1': constant_op.constant(['foo'])}
      mean, update_op = metrics_module.mean_tensor(constant_op.constant([0]))
      metrics = {
          'metrics_1': (mean, update_op),
          'metrics_2': (constant_op.constant([0]), constant_op.constant([10]))
      }

      outputter = export_output_lib.TrainOutput(loss, predictions, metrics)

      receiver = {u'features': constant_op.constant(100, shape=(100, 2)),
                  'labels': constant_op.constant(100, shape=(100, 1))}
      sig_def = outputter.as_signature_def(receiver)

      self.assertIn('loss/my_loss', sig_def.outputs)
      self.assertIn('metrics_1/value', sig_def.outputs)
      self.assertIn('metrics_2/value', sig_def.outputs)
      self.assertIn('predictions/output1', sig_def.outputs)
      self.assertIn('features', sig_def.inputs)

  def test_eval_signature_def(self):
    # [FUNCTION DOC]
    # Scenario: an eval output built with loss, predictions and no metrics is converted to a
    # signature definition from a features/labels receiver mapping. Asserts loss and prediction
    # outputs and the features input are present, and no metrics value output exists.
    with context.graph_mode():
      loss = {'my_loss': constant_op.constant([0])}
      predictions = {u'output1': constant_op.constant(['foo'])}

      outputter = export_output_lib.EvalOutput(loss, predictions, None)

      receiver = {u'features': constant_op.constant(100, shape=(100, 2)),
                  'labels': constant_op.constant(100, shape=(100, 1))}
      sig_def = outputter.as_signature_def(receiver)

      self.assertIn('loss/my_loss', sig_def.outputs)
      self.assertNotIn('metrics/value', sig_def.outputs)
      self.assertIn('predictions/output1', sig_def.outputs)
      self.assertIn('features', sig_def.inputs)

  def test_metric_op_is_tensor(self):
    # [FUNCTION DOC]
    # Scenario: metrics whose update part is a tensor, a no-op operation, or a variable. Builds
    # the mock supervised output and asserts that all update entries and values are tensors, that
    # an existing value tensor is kept as is, and that the no-op is wrapped in a tensor named with
    # a metric-op-wrapper prefix.
    """Tests that ops.Operation is wrapped by a tensor for metric_ops."""
    with context.graph_mode():
      loss = {'my_loss': constant_op.constant([0])}
      predictions = {u'output1': constant_op.constant(['foo'])}
      mean, update_op = metrics_module.mean_tensor(constant_op.constant([0]))
      metrics = {
          'metrics_1': (mean, update_op),
          'metrics_2': (constant_op.constant([0]), control_flow_ops.no_op()),
          # Keras metric's update_state() could return a Variable, rather than
          # an Operation or Tensor.
          'keras_1': (constant_op.constant([0.5]),
                      variables.Variable(1.0, name='AssignAddVariableOp_3'))
      }

      outputter = MockSupervisedOutput(loss, predictions, metrics)
      # If we get there, it means constructor succeeded; which is sufficient
      # for testing the constructor.

      self.assertTrue(outputter.metrics['metrics_1/update_op'].name.startswith(
          'mean/update_op'))
      self.assertIsInstance(
          outputter.metrics['metrics_1/update_op'], tensor.Tensor)
      self.assertIsInstance(outputter.metrics['metrics_1/value'], tensor.Tensor)

      self.assertEqual(outputter.metrics['metrics_2/value'],
                       metrics['metrics_2'][0])
      self.assertTrue(outputter.metrics['metrics_2/update_op'].name.startswith(
          'metric_op_wrapper'))
      self.assertIsInstance(
          outputter.metrics['metrics_2/update_op'], tensor.Tensor)


if __name__ == '__main__':
  test.main()
