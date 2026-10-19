# [MODULE DOC]
# Unit tests for the confusion matrix ops: the confusion matrix builder (basic, weighted,
# explicit class count, dtype, tensor inputs, out-of-range and negative labels/predictions,
# mismatched sizes) and the helper that removes squeezable trailing dimensions of labels and
# predictions (static and dynamic shapes, different expected rank differences).
# No import-time side effects; run as a script it starts the TensorFlow test runner.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for confusion_matrix_ops."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ConfusionMatrixTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the confusion matrix op. Holds a set of private helper methods that build the
  # matrix for given labels/predictions and compare with a truth matrix, plus public tests for
  # integer dtypes, weights, class counts and error conditions. No shared fixtures.

  @test_util.run_in_graph_and_eager_modes
  def testExample(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes) the example from the op documentation: labels one, two, four
    # versus predictions two, two, four give a 5x5 matrix with ones at (1,2), (2,2) and (4,4).
    """This is a test of the example provided in pydoc."""
    with self.cached_session():
      self.assertAllEqual([
          [0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1]
      ], self.evaluate(confusion_matrix.confusion_matrix(
          labels=[1, 2, 4], predictions=[2, 2, 4])))

  def _testConfMatrix(self, labels, predictions, truth, weights=None,
                      num_classes=None):
    # [FUNCTION DOC]
    # HELPER: core assertion routine of the confusion matrix tests. Inputs: labels and
    # predictions (numpy arrays; the dtype of predictions is used for the output), truth (expected
    # matrix), optional weights and optional num_classes. Builds the matrix in a cached session,
    # evaluates it via the default session, asserts closeness to truth (absolute tolerance 1e-10)
    # and that the result dtype equals the predictions dtype. Errors from the op propagate.
    with self.cached_session():
      dtype = predictions.dtype
      ans = confusion_matrix.confusion_matrix(
          labels, predictions, dtype=dtype, weights=weights,
          num_classes=num_classes).eval()
      self.assertAllClose(truth, ans, atol=1e-10)
      self.assertEqual(ans.dtype, dtype)

  def _testBasic(self, dtype):
    # [FUNCTION DOC]
    # HELPER: identical labels and predictions zero to four of the given numpy dtype must yield a
    # 5x5 identity matrix. Returns nothing.
    labels = np.arange(5, dtype=dtype)
    predictions = np.arange(5, dtype=dtype)

    truth = np.asarray(
        [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]],
        dtype=dtype)

    self._testConfMatrix(labels=labels, predictions=predictions, truth=truth)

  @test_util.run_deprecated_v1
  def testInt32Basic(self):
    # [FUNCTION DOC]
    # Graph-mode test of the basic identity case with 32-bit integers.
    self._testBasic(dtype=np.int32)

  @test_util.run_deprecated_v1
  def testInt64Basic(self):
    # [FUNCTION DOC]
    # Graph-mode test of the basic identity case with 64-bit integers.
    self._testBasic(dtype=np.int64)

  def _testConfMatrixOnTensors(self, tf_dtype, np_dtype):
    # [FUNCTION DOC]
    # HELPER: tests the confusion matrix on graph tensors produced by random ops. Inputs: tf_dtype
    # (tensor integer dtype) and np_dtype (matching numpy dtype). Draws 20 negative-class and 20
    # positive-class normal samples (means fed as zero and one, standard deviation one), rounds
    # and clamps them to zero or one as predictions, builds labels of twenty zeros then twenty
    # ones, computes a two-class matrix, then recomputes the truth by counting in python from the
    # same fetched values. Asserts the result dtype and values match. Uses random sampling but the
    # comparison uses the very same fetched samples.
    with self.cached_session() as sess:
      m_neg = array_ops.placeholder(dtype=dtypes.float32)
      m_pos = array_ops.placeholder(dtype=dtypes.float32)
      s = array_ops.placeholder(dtype=dtypes.float32)

      neg = random_ops.random_normal(
          [20], mean=m_neg, stddev=s, dtype=dtypes.float32)
      pos = random_ops.random_normal(
          [20], mean=m_pos, stddev=s, dtype=dtypes.float32)

      data = array_ops.concat([neg, pos], 0)
      data = math_ops.cast(math_ops.round(data), tf_dtype)
      data = math_ops.minimum(math_ops.maximum(data, 0), 1)
      lab = array_ops.concat(
          [
              array_ops.zeros(
                  [20], dtype=tf_dtype), array_ops.ones(
                      [20], dtype=tf_dtype)
          ],
          0)

      cm = confusion_matrix.confusion_matrix(
          lab, data, dtype=tf_dtype, num_classes=2)

      d, l, cm_out = sess.run([data, lab, cm], {m_neg: 0.0, m_pos: 1.0, s: 1.0})

      truth = np.zeros([2, 2], dtype=np_dtype)
      for i in range(len(d)):
        truth[l[i], d[i]] += 1

      self.assertEqual(cm_out.dtype, np_dtype)
      self.assertAllClose(cm_out, truth, atol=1e-10)

  @test_util.run_deprecated_v1
  def testOnTensors_int32(self):
    # [FUNCTION DOC]
    # Graph-mode test running the tensor-input helper with 32-bit integers.
    self._testConfMatrixOnTensors(dtypes.int32, np.int32)

  @test_util.run_deprecated_v1
  def testOnTensors_int64(self):
    # [FUNCTION DOC]
    # Graph-mode test running the tensor-input helper with 64-bit integers.
    self._testConfMatrixOnTensors(dtypes.int64, np.int64)

  def _testDifferentLabelsInPredictionAndTarget(self, dtype):
    # [FUNCTION DOC]
    # HELPER: labels (four, five, six) and predictions (one, two, three) share no classes. The
    # matrix must be 7x7 (size from the largest value) with ones at the label/prediction pairs.
    # Input: numpy dtype. Returns nothing.
    labels = np.asarray([4, 5, 6], dtype=dtype)
    predictions = np.asarray([1, 2, 3], dtype=dtype)

    truth = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0]],
        dtype=dtype)

    self._testConfMatrix(labels=labels, predictions=predictions, truth=truth)

  @test_util.run_deprecated_v1
  def testInt32DifferentLabels(self, dtype=np.int32):
    # [FUNCTION DOC]
    # Graph-mode test of disjoint label and prediction classes with 32-bit integers. The dtype
    # parameter is a default argument used only to pass the dtype to the helper.
    self._testDifferentLabelsInPredictionAndTarget(dtype)

  @test_util.run_deprecated_v1
  def testInt64DifferentLabels(self, dtype=np.int64):
    # [FUNCTION DOC]
    # Graph-mode test of disjoint label and prediction classes with 64-bit integers (dtype is a
    # default argument forwarded to the helper).
    self._testDifferentLabelsInPredictionAndTarget(dtype)

  def _testMultipleLabels(self, dtype):
    # [FUNCTION DOC]
    # HELPER: ten labels and predictions with repeated and mismatched classes must produce a 7x7
    # matrix with the hand-computed counts. Input: numpy dtype. Returns nothing.
    labels = np.asarray([1, 1, 2, 3, 5, 1, 3, 6, 3, 1], dtype=dtype)
    predictions = np.asarray([1, 1, 2, 3, 5, 6, 1, 2, 3, 4], dtype=dtype)

    truth = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 2, 0, 0, 1, 0, 1],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 2, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, 0]],
        dtype=dtype)

    self._testConfMatrix(labels=labels, predictions=predictions, truth=truth)

  @test_util.run_deprecated_v1
  def testInt32MultipleLabels(self, dtype=np.int32):
    # [FUNCTION DOC]
    # Graph-mode multiple-label test with 32-bit integers (dtype default forwarded to helper).
    self._testMultipleLabels(dtype)

  @test_util.run_deprecated_v1
  def testInt64MultipleLabels(self, dtype=np.int64):
    # [FUNCTION DOC]
    # Graph-mode multiple-label test with 64-bit integers (dtype default forwarded to helper).
    self._testMultipleLabels(dtype)

  @test_util.run_deprecated_v1
  def testWeighted(self):
    # [FUNCTION DOC]
    # Graph-mode test with identical labels/predictions zero to four and weights zero to four:
    # the matrix diagonal must equal the weights, other cells zero.
    labels = np.arange(5, dtype=np.int32)
    predictions = np.arange(5, dtype=np.int32)
    weights = np.arange(5, dtype=np.int32)

    truth = np.asarray(
        [[0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 2, 0, 0],
         [0, 0, 0, 3, 0],
         [0, 0, 0, 0, 4]],
        dtype=np.int32)

    self._testConfMatrix(
        labels=labels, predictions=predictions, weights=weights, truth=truth)

  @test_util.run_deprecated_v1
  def testLabelsTooLarge(self):
    # [FUNCTION DOC]
    # Graph-mode test that a label (five) not smaller than the explicit class count of three
    # raises an invalid-argument error whose message mentions labels being out of bound.
    labels = np.asarray([1, 1, 0, 3, 5], dtype=np.int32)
    predictions = np.asarray([2, 1, 0, 2, 2], dtype=np.int32)
    with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError,
                                             "`labels`.*out of bound"):
      self._testConfMatrix(
          labels=labels, predictions=predictions, num_classes=3, truth=None)

  def testLabelsNegative(self):
    # [FUNCTION DOC]
    # Test that negative labels with class count three raise an op error mentioning that labels
    # contain negative values.
    labels = np.asarray([1, 1, 0, -1, -1], dtype=np.int32)
    predictions = np.asarray([2, 1, 0, 2, 2], dtype=np.int32)
    with self.assertRaisesOpError("`labels`.*negative values"):
      self._testConfMatrix(
          labels=labels, predictions=predictions, num_classes=3, truth=None)

  @test_util.run_deprecated_v1
  def testPredictionsTooLarge(self):
    # [FUNCTION DOC]
    # Graph-mode test that a prediction (five) not smaller than the class count of three raises an
    # invalid-argument error mentioning that predictions are out of bound.
    labels = np.asarray([1, 1, 0, 2, 2], dtype=np.int32)
    predictions = np.asarray([2, 1, 0, 3, 5], dtype=np.int32)
    with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError,
                                             "`predictions`.*out of bound"):
      self._testConfMatrix(
          labels=labels, predictions=predictions, num_classes=3, truth=None)

  def testPredictionsNegative(self):
    # [FUNCTION DOC]
    # Test that negative predictions with class count three raise an op error mentioning that
    # predictions contain negative values.
    labels = np.asarray([1, 1, 0, 2, 2], dtype=np.int32)
    predictions = np.asarray([2, 1, 0, -1, -1], dtype=np.int32)
    with self.assertRaisesOpError("`predictions`.*negative values"):
      self._testConfMatrix(
          labels=labels, predictions=predictions, num_classes=3, truth=None)

  @test_util.run_deprecated_v1
  def testInputDifferentSize(self):
    # [FUNCTION DOC]
    # Graph-mode test that labels of length two and predictions of length three raise a value
    # error whose message says the sizes must be equal.
    labels = np.asarray([1, 2])
    predictions = np.asarray([1, 2, 3])
    self.assertRaisesRegex(ValueError, "must be equal",
                           confusion_matrix.confusion_matrix, predictions,
                           labels)

  def testOutputIsInt32(self):
    # [FUNCTION DOC]
    # Test that requesting a 32-bit integer dtype yields a result array of that dtype.
    labels = np.arange(2)
    predictions = np.arange(2)
    with self.cached_session():
      cm = confusion_matrix.confusion_matrix(
          labels, predictions, dtype=dtypes.int32)
      tf_cm = self.evaluate(cm)
    self.assertEqual(tf_cm.dtype, np.int32)

  def testOutputIsInt64(self):
    # [FUNCTION DOC]
    # Test that requesting a 64-bit integer dtype yields a result array of that dtype.
    labels = np.arange(2)
    predictions = np.arange(2)
    with self.cached_session():
      cm = confusion_matrix.confusion_matrix(
          labels, predictions, dtype=dtypes.int64)
      tf_cm = self.evaluate(cm)
    self.assertEqual(tf_cm.dtype, np.int64)


class RemoveSqueezableDimensionsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the function that squeezes a trailing dimension of size one from labels or
  # predictions when their ranks differ by the expected amount. Each test checks both the static
  # (known shape) path with plain values and the dynamic path with placeholders fed at run time.
  # All tests are graph-mode only. No shared fixtures.

  @test_util.run_deprecated_v1
  def testBothScalarShape(self):
    # [FUNCTION DOC]
    # Both inputs are scalars; neither should change. Checks static and placeholder paths.
    label_values = 1.0
    prediction_values = 0.0
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder))

    with self.cached_session():
      self.assertAllEqual(label_values, self.evaluate(static_labels))
      self.assertAllEqual(prediction_values, self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSameShape(self):
    # [FUNCTION DOC]
    # Labels and predictions both have shape 2x3x1 (same rank); neither is squeezed, in both
    # static and placeholder paths.
    label_values = np.ones(shape=(2, 3, 1))
    prediction_values = np.zeros_like(label_values)
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder))

    with self.cached_session():
      self.assertAllEqual(label_values, self.evaluate(static_labels))
      self.assertAllEqual(prediction_values, self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSameShapeExpectedRankDiff0(self):
    # [FUNCTION DOC]
    # Same as the same-shape test but with expected rank difference zero passed explicitly:
    # nothing is squeezed.
    label_values = np.ones(shape=(2, 3, 1))
    prediction_values = np.zeros_like(label_values)
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values, expected_rank_diff=0))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder, expected_rank_diff=0))

    with self.cached_session():
      self.assertAllEqual(label_values, self.evaluate(static_labels))
      self.assertAllEqual(prediction_values, self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSqueezableLabels(self):
    # [FUNCTION DOC]
    # Labels have shape 2x3x1 and predictions 2x3: with the default expected rank difference of
    # zero the labels lose the trailing dimension and predictions are unchanged (static and
    # dynamic paths).
    label_values = np.ones(shape=(2, 3, 1))
    prediction_values = np.zeros(shape=(2, 3))
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder))

    expected_label_values = np.reshape(label_values, (2, 3))
    with self.cached_session():
      self.assertAllEqual(expected_label_values, self.evaluate(static_labels))
      self.assertAllEqual(prediction_values, self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          expected_label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSqueezableLabelsExpectedRankDiffPlus1(self):
    # [FUNCTION DOC]
    # Labels 2x3x1 and predictions 2x3x5 with expected rank difference of one (predictions one
    # rank higher than the labels): labels are squeezed to 2x3, predictions are unchanged.
    label_values = np.ones(shape=(2, 3, 1))
    prediction_values = np.zeros(shape=(2, 3, 5))
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values, expected_rank_diff=1))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder, expected_rank_diff=1))

    expected_label_values = np.reshape(label_values, (2, 3))
    with self.cached_session():
      self.assertAllEqual(expected_label_values, self.evaluate(static_labels))
      self.assertAllEqual(prediction_values, self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          expected_label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSqueezablePredictions(self):
    # [FUNCTION DOC]
    # Labels 2x3 and predictions 2x3x1 with the default expected rank difference: predictions are
    # squeezed to 2x3, labels unchanged (static and dynamic paths).
    label_values = np.ones(shape=(2, 3))
    prediction_values = np.zeros(shape=(2, 3, 1))
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder))

    expected_prediction_values = np.reshape(prediction_values, (2, 3))
    with self.cached_session():
      self.assertAllEqual(label_values, self.evaluate(static_labels))
      self.assertAllEqual(expected_prediction_values,
                          self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          expected_prediction_values,
          dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testSqueezablePredictionsExpectedRankDiffMinus1(self):
    # [FUNCTION DOC]
    # Labels 2x3x5 and predictions 2x3x1 with expected rank difference of minus one: predictions
    # are squeezed to 2x3, labels unchanged.
    label_values = np.ones(shape=(2, 3, 5))
    prediction_values = np.zeros(shape=(2, 3, 1))
    static_labels, static_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            label_values, prediction_values, expected_rank_diff=-1))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(
            labels_placeholder, predictions_placeholder, expected_rank_diff=-1))

    expected_prediction_values = np.reshape(prediction_values, (2, 3))
    with self.cached_session():
      self.assertAllEqual(label_values, self.evaluate(static_labels))
      self.assertAllEqual(expected_prediction_values,
                          self.evaluate(static_predictions))
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))
      self.assertAllEqual(
          expected_prediction_values,
          dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testUnsqueezableLabels(self):
    # [FUNCTION DOC]
    # Labels 2x3x2 (extra dimension not of size one) and predictions 2x3, dynamic path only:
    # nothing can be squeezed, so predictions come back unchanged.
    label_values = np.ones(shape=(2, 3, 2))
    prediction_values = np.zeros(shape=(2, 3))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    _, dynamic_predictions = (
        confusion_matrix.remove_squeezable_dimensions(labels_placeholder,
                                                      predictions_placeholder))

    with self.cached_session():
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          prediction_values, dynamic_predictions.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testUnsqueezablePredictions(self):
    # [FUNCTION DOC]
    # Labels 2x3 and predictions 2x3x2, dynamic path only: nothing can be squeezed, so labels come
    # back unchanged.
    label_values = np.ones(shape=(2, 3))
    prediction_values = np.zeros(shape=(2, 3, 2))

    labels_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    predictions_placeholder = array_ops.placeholder(dtype=dtypes.int32)
    dynamic_labels, _ = (
        confusion_matrix.remove_squeezable_dimensions(labels_placeholder,
                                                      predictions_placeholder))

    with self.cached_session():
      feed_dict = {
          labels_placeholder: label_values,
          predictions_placeholder: prediction_values
      }
      self.assertAllEqual(
          label_values, dynamic_labels.eval(feed_dict=feed_dict))


if __name__ == "__main__":
  test.main()
