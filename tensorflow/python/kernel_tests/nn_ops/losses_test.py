# [MODULE DOC]
# Tests for the loss functions in the tensorflow ops losses package (the legacy graph-style
# losses API: absolute difference, softmax / sparse softmax / sigmoid cross entropy, log loss,
# hinge, Huber, mean squared error, mean pairwise squared error, cosine distance, plus the
# loss collection behavior and the weighted-loss reduction helper).
# Defines one test case class per loss function. Most tests build small constant or placeholder
# tensors, call the loss with different kinds of weights (none, python scalar, scalar tensor,
# per-batch-element, per-measurement, all-zero, invalid shapes), evaluate in a cached session
# and compare to values computed by hand or with numpy. Many classes and methods are decorated
# to run only in graph (deprecated v1) mode. No import-time side effects other than imports;
# tests run through the standard test main entry point when executed as a script.
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
"""Tests for losses."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.platform import test
from tensorflow.python.training import momentum as momentum_lib


@test_util.run_deprecated_v1
class AbsoluteDifferenceLossTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the absolute difference loss. The whole class runs in graph (deprecated v1) mode.
  # Shared fixture: two 2x3 integer constants for predictions and labels created in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the fixture: a 2x3 integer predictions constant and a 2x3 integer labels constant,
    # after running the parent setUp. Stores both on the instance; returns nothing.
    super(AbsoluteDifferenceLossTest, self).setUp()
    self._predictions = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    self._labels = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    # [FUNCTION DOC]
    # Test: calls the absolute difference loss function with weights explicitly set to nothing and expects a
    # value error to be raised.
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.absolute_difference(
            self._predictions, self._predictions, weights=None)

  def testAllCorrectNoLossWeight(self):
    # [FUNCTION DOC]
    # Test: uses identical values for predictions and labels with default weights and expects
    # the absolute difference loss to evaluate to zero (within three decimal places).
    loss = losses.absolute_difference(self._predictions, self._predictions)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  def testNonZeroLoss(self):
    # [FUNCTION DOC]
    # Test: computes the absolute difference loss on the fixture labels and predictions with default weights and
    # expects the hand-computed average value (within three decimal places).
    loss = losses.absolute_difference(self._labels, self._predictions)
    with self.cached_session():
      self.assertAlmostEqual(5.5, self.evaluate(loss), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    # [FUNCTION DOC]
    # Test: passes a plain python number (2.3) as weights; expects the absolute difference loss to equal the
    # unweighted value multiplied by that weight.
    weights = 2.3
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(5.5 * weights, self.evaluate(loss), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    # [FUNCTION DOC]
    # Test: passes the weight 2.3 wrapped in a scalar constant tensor; expects the absolute difference loss to equal
    # the unweighted value multiplied by that weight.
    weights = 2.3
    loss = losses.absolute_difference(self._labels, self._predictions,
                                      constant_op.constant(weights))
    with self.cached_session():
      self.assertAlmostEqual(5.5 * weights, self.evaluate(loss), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: passes one weight per batch element (a one-dimensional or column weight tensor) to
    # the absolute difference loss and expects the hand-computed weighted result.
    weights = constant_op.constant((1.2, 0.0), shape=(2, 1))
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(5.6, self.evaluate(loss), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: like the one-dimensional batch weights test but the weights are given as a column
    # tensor of shape two by one; expects the same weighted result of the absolute difference loss.
    weights = constant_op.constant([1.2, 0.0], shape=[2, 1])
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(5.6, self.evaluate(loss), 3)

  def testNonZeroLossWithSampleSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: passes a weight for every element of the 2x3 inputs; expects the hand-computed
    # weighted absolute difference loss (within three decimal places).
    weights = constant_op.constant([3, 6, 5, 0, 4, 2], shape=[2, 3])
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(16.6, self.evaluate(loss), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    # [FUNCTION DOC]
    # Test: passes element weights that are zero except for the last element; expects the absolute difference loss
    # to be determined by that single weighted element.
    weights = constant_op.constant([0, 0, 0, 0, 0, 2], shape=[2, 3])
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(6.0, self.evaluate(loss), 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    # [FUNCTION DOC]
    # Test: passes all-zero element weights; expects the absolute difference loss to evaluate to zero rather
    # than failing with a division by zero.
    weights = array_ops.zeros((2, 3))
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testEagerNoMemoryLeaked(self):
    # [FUNCTION DOC]
    # Test: builds fresh constants and calls the absolute difference loss once under a decorator
    # that asserts no new python objects remain after eager execution; verifies nothing is added
    # to a global collection. No explicit asserts in the body.
    # This is a somewhat convoluted way of testing that nothing gets added to
    # a global collection.
    predictions = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    labels = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
    losses.absolute_difference(labels, predictions)


class SoftmaxCrossEntropyLossTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the softmax cross entropy loss using 3x3 one-hot labels and logits with a
  # dominant value of 10, so the correct loss values are easy to derive by hand. Not decorated
  # at class level; individual methods opt into graph mode.

  def testNoneWeightRaisesValueError(self):
    # [FUNCTION DOC]
    # Test: calls the softmax cross entropy loss function with weights explicitly set to nothing and expects a
    # value error to be raised.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.softmax_cross_entropy(labels, logits, weights=None)

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    # [FUNCTION DOC]
    # Test (graph mode): logits strongly favor the true class of each row; expects the loss
    # operation to be named with the default scope and the value to be about zero.
    with self.cached_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      loss = losses.softmax_cross_entropy(labels, logits)
      self.assertEqual('softmax_cross_entropy_loss/value', loss.op.name)
      self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  @test_util.run_deprecated_v1
  def testAllWrong(self):
    # [FUNCTION DOC]
    # Test (graph mode): every row has its label on a class with low logit; expects the default
    # operation name and a loss of about ten.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 10.0, 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithPythonScalarWeight(self):
    # [FUNCTION DOC]
    # Test: passes a plain python number (2.3) as weights; expects the softmax cross entropy loss to equal the
    # unweighted value multiplied by that weight.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = 2.3
    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(weights * 10.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithScalarTensorWeight(self):
    # [FUNCTION DOC]
    # Test: passes the weight 2.3 wrapped in a scalar constant tensor; expects the softmax cross entropy loss to equal
    # the unweighted value multiplied by that weight.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = 2.3
    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits,
                                          constant_op.constant(weights))
      self.assertAlmostEqual(weights * 10.0, self.evaluate(loss), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: passes a vector with one weight per row (1.2, 3.4, 5.6) with all-wrong predictions; expects
    # the weighted sum times ten divided by the batch size.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant((1.2, 3.4, 5.6))
    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0,
                             self.evaluate(loss), 3)

  def testAllWrongAllWeightsMissing(self):
    # [FUNCTION DOC]
    # Test: all predictions wrong but every per-row weight is zero; expects a loss of zero.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant([0, 0, 0], shape=[3])
    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  def testSomeWeightsMissing(self):
    # [FUNCTION DOC]
    # Test: all predictions wrong, only the first row has non-zero weight (1.2); expects a loss of
    # twelve, since zero-weight rows are excluded from the averaging.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant([1.2, 0, 0], shape=[3])
    with self.cached_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(12.0, self.evaluate(loss), 3)

  def testSoftmaxWithMeasurementSpecificWeightsRaisesException(self):
    # [FUNCTION DOC]
    # Test: passes a full 3x3 per-measurement weight matrix, which the softmax loss does not
    # support (weights must be per example); expects a value error.
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      weights = constant_op.constant([[3, 4, 5], [2, 6, 0], [8, 0, 1]])

      with self.assertRaises(ValueError):
        losses.softmax_cross_entropy(labels, logits, weights=weights).eval()

  @test_util.run_deprecated_v1
  def testSoftmaxLabelSmoothing(self):
    # [FUNCTION DOC]
    # Test (graph mode): single row with a confident correct logit and label smoothing 0.1;
    # expects the loss to equal four hundred times the smoothing divided by the class count,
    # as derived in the inline comment, and the default operation name.
    with self.cached_session():
      # Softmax Cross Entropy Loss is:
      #   -\sum_i p_i \log q_i
      # where for a softmax activation
      # \log q_i = x_i - \log \sum_j \exp x_j
      #          = x_i - x_max - \log \sum_j \exp (x_j - x_max)
      # For our activations, [100, -100, -100] the log partition function
      # becomes \log ( exp(0) + exp(-200) + exp(-200) ) = 0
      # so our log softmaxes become: [0, -200, -200]
      # so our cross entropy loss is:
      # -(1 - L + L/n) * 0 + 400 * L/n = 400 L/n
      logits = constant_op.constant([[100.0, -100.0, -100.0]])
      labels = constant_op.constant([[1, 0, 0]])
      label_smoothing = 0.1
      loss = losses.softmax_cross_entropy(
          labels, logits, label_smoothing=label_smoothing)
      self.assertEqual(loss.op.name, 'softmax_cross_entropy_loss/value')
      expected_value = 400.0 * label_smoothing / 3.0
      self.assertAlmostEqual(self.evaluate(loss), expected_value, 3)


class SparseSoftmaxCrossEntropyLossTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the sparse softmax cross entropy loss, which takes integer class indices as
  # labels (column or flat). Covers int32 and int64 labels, various weight shapes and dtypes,
  # placeholders with unknown shapes, and the value errors for inconsistent shapes.

  def testNoneWeightRaisesValueError(self):
    # [FUNCTION DOC]
    # Test: calls the sparse softmax cross entropy loss function with weights explicitly set to nothing and expects a
    # value error to be raised.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0], [1], [2]])
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(labels, logits, weights=None)

  @test_util.run_deprecated_v1
  def testAllCorrectInt32Labels(self):
    # [FUNCTION DOC]
    # Test (graph mode): 3x3 logits with correct int32 column labels; expects the default
    # operation name and a loss of about zero.
    with self.cached_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([[0], [1], [2]], dtype=dtypes.int32)
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testEagerNoMemoryLeaked(self):
    # [FUNCTION DOC]
    # Test: calls the sparse softmax cross entropy loss once on constants under a decorator that
    # asserts no new python objects remain after eager execution. No explicit asserts.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0], [1], [2]], dtype=dtypes.int32)
    losses.sparse_softmax_cross_entropy(labels, logits)

  @test_util.run_deprecated_v1
  def testAllCorrectInt64Labels(self):
    # [FUNCTION DOC]
    # Test (graph mode): same as the int32 all-correct case but with int64 labels; expects about zero.
    with self.cached_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([[0], [1], [2]], dtype=dtypes.int64)
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  @test_util.run_deprecated_v1
  def testAllCorrectNonColumnLabels(self):
    # [FUNCTION DOC]
    # Test (graph mode): labels given as a flat vector of class indices instead of a column;
    # expects the default operation name and a loss of about zero.
    with self.cached_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([0, 1, 2])
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

  @test_util.run_deprecated_v1
  def testAllWrongInt32Labels(self):
    # [FUNCTION DOC]
    # Test (graph mode): int32 column labels all pointing at wrong classes; expects about ten.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]], dtype=dtypes.int32)

    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 10.0, 3)

  @test_util.run_deprecated_v1
  def testAllWrongInt64Labels(self):
    # [FUNCTION DOC]
    # Test (graph mode): int64 column labels all pointing at wrong classes; expects about ten.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]], dtype=dtypes.int64)

    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 10.0, 3)

  @test_util.run_deprecated_v1
  def testAllWrongNonColumnLabels(self):
    # [FUNCTION DOC]
    # Test (graph mode): flat label vector, all wrong; expects about ten.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([2, 0, 1])

    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEqual(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(self.evaluate(loss), 10.0, 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithPythonScalarWeight(self):
    # [FUNCTION DOC]
    # Test (graph mode): all-wrong labels with a python scalar weight of 2.3; expects ten times
    # the weight.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = 2.3
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(weights * 10.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithScalarTensorWeight(self):
    # [FUNCTION DOC]
    # Test (graph mode): all-wrong labels with the weight 2.3 as a scalar tensor; expects ten times
    # the weight.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = 2.3
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits,
                                                 constant_op.constant(weights))
      self.assertAlmostEqual(weights * 10.0, self.evaluate(loss), 3)

  def testNonZeroLossWith1DTensorWeight(self):
    # [FUNCTION DOC]
    # Test: passes a one-element vector tensor as weights with all-wrong labels; expects ten times
    # that weight, showing a length-one weight vector is accepted like a scalar.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = 2.3
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(
          labels, logits, constant_op.constant((weights,)))
      self.assertAlmostEqual(weights * 10.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithPlaceholderForWeights(self):
    # [FUNCTION DOC]
    # Test (graph mode): weights are a float placeholder with no shape, fed at run time with a
    # column of three values; expects the weighted average ten times mean of the weights.
    logits = constant_op.constant([[10.0, 0.0, 0.0],
                                   [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = array_ops.placeholder(dtypes.float32)
    with self.cached_session() as sess:
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      loss_val = sess.run(loss,
                          feed_dict={weights: ((1.2,), (3.4,), (5.6,))})
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0, loss_val, 3)

  @test_util.run_deprecated_v1
  def testUnknownShapePlaceholderForLogitsLabelsButScalarWeights(self):
    # [FUNCTION DOC]
    # Test (graph mode): logits and labels are placeholders of completely unknown shape and the
    # weight is the python value one; fed with concrete values at run time, expects about ten.
    logits = array_ops.placeholder(dtypes.float32)
    labels = array_ops.placeholder(dtypes.int32)
    weights = 1.0
    with self.cached_session() as sess:
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      loss_val = sess.run(loss,
                          feed_dict={
                              logits: [[10.0, 0.0, 0.0],
                                       [0.0, 10.0, 0.0],
                                       [0.0, 0.0, 10.0]],
                              labels: [[2], [0], [1]],
                          })
      self.assertAlmostEqual((1.0 + 1.0 + 1.0) * 10.0 / 3.0, loss_val, 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithPlaceholderForLogitsLabelsAndWeights(self):
    # [FUNCTION DOC]
    # Test (graph mode): logits, labels and weights are all placeholders with partially known
    # shapes, fed at run time; expects the weighted all-wrong loss.
    logits = array_ops.placeholder(dtypes.float32, shape=(None, 3))
    labels = array_ops.placeholder(dtypes.int32, shape=(None, 1))
    weights = array_ops.placeholder(dtypes.float32)
    with self.cached_session() as sess:
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      loss_val = sess.run(loss,
                          feed_dict={
                              logits: [[10.0, 0.0, 0.0],
                                       [0.0, 10.0, 0.0],
                                       [0.0, 0.0, 10.0]],
                              labels: [[2], [0], [1]],
                              weights: ((1.2,), (3.4,), (5.6,)),
                          })
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0, loss_val, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: weights are a column of per-example values (3 by 1) with all-wrong labels; expects
    # the sum of the weights times ten divided by three.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([1.2, 3.4, 5.6], shape=(3, 1))
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithColumnWeights(self):
    # [FUNCTION DOC]
    # Test: weights given as a nested column list tensor (3 by 1); expects the same weighted
    # result as the batch-specific weights test.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([[1.2], [3.4], [5.6]])
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0,
                             self.evaluate(loss), 3)

  def testAllWrongAllWeightsMissing(self):
    # [FUNCTION DOC]
    # Test: all-wrong labels but all per-example weights zero; expects a loss of zero.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([0, 0, 0], shape=(3, 1))
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  def testSomeWeightsMissing(self):
    # [FUNCTION DOC]
    # Test: all-wrong labels, only the first example weighted 1.2; expects a loss of twelve.
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([1.2, 0, 0], shape=(3, 1))
    with self.cached_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(12.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testMeasurementSpecificWeightsRaisesException(self):
    # [FUNCTION DOC]
    # Test (graph mode): passes a full 3x3 weight matrix which is not broadcastable to the
    # per-example loss; expects a value error.
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2]])
      weights = constant_op.constant([[3, 4, 5], [2, 6, 0], [8, 0, 1]])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  def testInconsistentWeightSizeRaisesException(self):
    # [FUNCTION DOC]
    # Test: weight vector has four elements while there are three examples; expects a value error.
    """The weight tensor has incorrect number of elements."""
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2]])
      weights = constant_op.constant([1.2, 3.4, 5.6, 7.8])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  def testInconsistentLabelSizeRaisesException(self):
    # [FUNCTION DOC]
    # Test: four labels are given for three logits rows; expects a value error.
    """The label tensor has incorrect number of elements."""
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2], [3]])
      weights = constant_op.constant([1.2, 3.4, 5.6])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  @test_util.run_deprecated_v1
  def testInconsistentWeightShapeRaisesException(self):
    # [FUNCTION DOC]
    # Test (graph mode): four examples but a 2x2 weight matrix; expects a value error.
    """The weight tensor has incorrect shape."""
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0, -100.0],
                                     [-100.0, -100.0, 100.0, -100.0],
                                     [-100.0, -100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2], [3]])
      weights = constant_op.constant([[1.2, 3.4], [5.6, 7.8]])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  @test_util.run_deprecated_v1
  def testInconsistentLabelShapeRaisesException(self):
    # [FUNCTION DOC]
    # Test (graph mode): labels of rank two for logits of rank two (labels must have rank one less);
    # expects a value error whose message says label rank must equal logits rank minus one.
    """The label tensor has incorrect shape."""
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0, -100.0],
                                     [-100.0, -100.0, 100.0, -100.0],
                                     [-100.0, -100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0, 1], [2, 3]])
      weights = constant_op.constant(1.2)

      with self.assertRaisesRegex(
          ValueError,
          '`labels.shape.rank` must equal `logits.shape.rank - 1`'):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()


class SigmoidCrossEntropyLossTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the sigmoid cross entropy loss, using logits of plus or minus 100 so results
  # are essentially exact. Also asserts the loss dtype matches the logits dtype, the default
  # operation name, float64 inputs, no-reduction mode and label smoothing.

  @test_util.run_deprecated_v1
  def testAllCorrectSigmoid(self):
    # [FUNCTION DOC]
    # Test (graph mode): logits agree with one-hot labels; expects dtype equal to logits, the
    # default operation name and a loss of about zero.
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      loss = losses.sigmoid_cross_entropy(labels, logits)
      self.assertEqual(logits.dtype, loss.dtype)
      self.assertEqual('sigmoid_cross_entropy_loss/value', loss.op.name)
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testLossWithSingleDimPlaceholderForLogitsAndWeights1(self):
    # [FUNCTION DOC]
    # Test (graph mode): placeholders of shape unknown-by-one for logits and labels, weights of ones,
    # fed with 32 ones each; expects the loss dtype to match and a value of about 0.313.
    logits = array_ops.placeholder(dtypes.float32, shape=(None, 1))
    labels = array_ops.placeholder(dtypes.float32, shape=(None, 1))
    weights = array_ops.ones_like(logits, dtype=dtypes.float32)

    loss = losses.sigmoid_cross_entropy(labels, logits, weights)
    self.assertEqual(logits.dtype, loss.dtype)

    with self.cached_session() as sess:
      loss = sess.run(loss,
                      feed_dict={
                          logits: np.ones((32, 1)),
                          labels: np.ones((32, 1)),
                      })
      self.assertAlmostEqual(0.313, loss, 3)

  @test_util.run_deprecated_v1
  def testLossWithSingleDimPlaceholderForLogitsAndWeights2(self):
    # [FUNCTION DOC]
    # Test (graph mode): same as the previous test but with two columns; expects the same value,
    # about 0.313, because the mean over elements is unchanged.
    logits = array_ops.placeholder(dtypes.float32, shape=(None, 2))
    labels = array_ops.placeholder(dtypes.float32, shape=(None, 2))
    weights = array_ops.ones_like(logits, dtype=dtypes.float32)

    loss = losses.sigmoid_cross_entropy(labels, logits, weights)
    self.assertEqual(logits.dtype, loss.dtype)

    with self.cached_session() as sess:
      loss = sess.run(loss,
                      feed_dict={
                          logits: np.ones((32, 2)),
                          labels: np.ones((32, 2)),
                      })
      self.assertAlmostEqual(0.313, loss, 3)

  @test_util.run_deprecated_v1
  def testAllWrongSigmoid(self):
    # [FUNCTION DOC]
    # Test (graph mode): labels opposite to every logit; expects dtype, operation name and a loss
    # of six hundred over nine.
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
      loss = losses.sigmoid_cross_entropy(labels, logits)
      self.assertEqual(logits.dtype, loss.dtype)
      self.assertEqual('sigmoid_cross_entropy_loss/value', loss.op.name)
      self.assertAlmostEqual(self.evaluate(loss), 600.0 / 9.0, 3)

  @test_util.run_deprecated_v1
  def testAllWrongSigmoidWithMeasurementSpecificWeights(self):
    # [FUNCTION DOC]
    # Test (graph mode): all-wrong predictions with a 3x3 per-element weight matrix; expects the
    # weighted average (seventeen hundred over seven), dtype and operation name.
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
      weights = constant_op.constant([[3, 4, 5], [2, 6, 0], [8, 0, 1]])
      loss = losses.sigmoid_cross_entropy(labels, logits, weights)
      self.assertEqual(logits.dtype, loss.dtype)
      self.assertEqual('sigmoid_cross_entropy_loss/value', loss.op.name)
      self.assertAlmostEqual(1700.0 / 7.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testMultiCorrectSigmoid(self):
    # [FUNCTION DOC]
    # Test (graph mode): multi-label targets where several classes are active and all logits are
    # correct; expects dtype, operation name and a loss of about zero.
    logits = constant_op.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0],
                                   [-100.0, 100.0, 100.0]])
    labels = constant_op.constant([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    loss = losses.sigmoid_cross_entropy(labels, logits)
    self.assertEqual(logits.dtype, loss.dtype)
    self.assertEqual('sigmoid_cross_entropy_loss/value', loss.op.name)

    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  def testSigmoidFloat64(self):
    # [FUNCTION DOC]
    # Test: float64 logits with int64 labels; expects the loss dtype to be float64 and a value of
    # about 44.444.
    logits = constant_op.constant((
        (100.0, -100.0, 100.0),
        (100.0, -100.0, 100.0),
        (100.0, 100.0, -100.0)
    ), dtype=dtypes.float64)
    labels = constant_op.constant((
        (1, 0, 1), (1, 1, 0), (0, 1, 1)
    ), dtype=dtypes.int64)
    loss = losses.sigmoid_cross_entropy(labels, logits)
    self.assertEqual(logits.dtype, loss.dtype)

    with self.cached_session():
      self.assertAlmostEqual(44.444, self.evaluate(loss), 3)

  def testSigmoidNoReduction(self):
    # [FUNCTION DOC]
    # Test: requests no reduction; expects the loss to keep the full 3x3 shape with each element
    # equal to its individual loss (zero or one hundred), and dtype equal to the logits.
    logits = constant_op.constant((
        (100.0, -100.0, 100.0),
        (100.0, -100.0, 100.0),
        (100.0, 100.0, -100.0)))
    labels = constant_op.constant(((1, 0, 1), (1, 1, 0), (0, 1, 1)))
    loss = losses.sigmoid_cross_entropy(
        labels, logits, reduction=losses.Reduction.NONE)
    self.assertEqual(logits.dtype, loss.dtype)

    with self.cached_session():
      self.assertAllClose(((0., 0., 0.), (0., 100., 100.), (100., 0., 100.)),
                          self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testSigmoidLabelSmoothingCorrect(self):
    # [FUNCTION DOC]
    # Test (graph mode): one row with label smoothing 0.1; expects the loss to equal a third of
    # one hundred plus fifty times the smoothing, as derived in the inline comment, plus the dtype
    # and operation name checks.
    with self.cached_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0]])
      labels = constant_op.constant([[1, 0, 1]])
      # Sigmoid cross entropy loss is:
      #   max(x,0) - x*z + log(1 + exp(-abs(x)))
      # The new labels are:
      #    z' = z * (1 - L) + 0.5 L
      #    1 -> 1 - 0.5 L
      #    0 -> 0.5 L
      # here we expect:
      # 1/3 * (100 - 100 * (1 - 0.5 L)  + 0
      #       + 0  + 100 * (0.5 L)      + 0
      #       + 0  + 100 * (1 - 0.5 L)  + 0)
      # = 1/3 * (100 + 50 L)
      label_smoothing = 0.1
      loss = losses.sigmoid_cross_entropy(
          labels, logits, label_smoothing=label_smoothing)
      self.assertEqual(logits.dtype, loss.dtype)
      self.assertEqual('sigmoid_cross_entropy_loss/value', loss.op.name)
      expected_value = (100.0 + 50.0 * label_smoothing) / 3.0
      self.assertAlmostEqual(self.evaluate(loss), expected_value, 3)

  @test_util.run_deprecated_v1
  def testSigmoidLabelSmoothingEqualsSoftmaxTwoLabel(self):
    # [FUNCTION DOC]
    # Test (graph mode): checks that sigmoid cross entropy with smoothing on three binary targets
    # equals softmax cross entropy with the same smoothing on an equivalent two-class encoding.
    with self.cached_session():
      label_smoothing = 0.1
      sigmoid_logits = constant_op.constant([[100.0, -100.0, -100.0]])
      sigmoid_labels = constant_op.constant([[1, 0, 1]])
      sigmoid_loss = losses.sigmoid_cross_entropy(
          sigmoid_labels, sigmoid_logits, label_smoothing=label_smoothing)
      self.assertEqual(sigmoid_logits.dtype, sigmoid_loss.dtype)

      softmax_logits = constant_op.constant(
          [[0.0, 100.0], [100.0, 0.0], [100.0, 0.0]])
      softmax_labels = constant_op.constant([[0, 1], [1, 0], [0, 1]])
      softmax_loss = losses.softmax_cross_entropy(
          softmax_labels, softmax_logits, label_smoothing=label_smoothing)
      self.assertAlmostEqual(
          self.evaluate(sigmoid_loss), self.evaluate(softmax_loss), 3)


@test_util.run_deprecated_v1
class LogLossTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the log loss (negative log likelihood of binary labels). The class runs in graph
  # (deprecated v1) mode. The fixture precomputes the expected per-element losses with numpy using
  # the same small epsilon the library applies.

  def setUp(self):
    # [FUNCTION DOC]
    # Builds the fixture: numpy predictions and labels reshaped to 2x3, the expected per-element
    # log losses (with epsilon one e minus seven), and the matching constant tensors. Stores them on
    # the instance; returns nothing.
    super(LogLossTest, self).setUp()
    predictions = np.asarray([.9, .2, .2, .8, .4, .6]).reshape((2, 3))
    labels = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]).reshape((2, 3))

    self._np_predictions = predictions
    self._np_labels = labels

    epsilon = 1e-7
    self._expected_losses = np.multiply(
        labels, np.log(predictions + epsilon)) + np.multiply(
            1 - labels, np.log(1 - predictions + epsilon))

    self._predictions = constant_op.constant(predictions)
    self._labels = constant_op.constant(labels)

  def testValueErrorThrownWhenWeightIsNone(self):
    # [FUNCTION DOC]
    # Test: calls the log loss function with weights explicitly set to nothing and expects a
    # value error to be raised.
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.log_loss(self._labels, self._labels, weights=None)

  def testAllCorrectNoLossWeight(self):
    # [FUNCTION DOC]
    # Test: uses identical values for predictions and labels with default weights and expects
    # the log loss to evaluate to zero (within three decimal places).
    loss = losses.log_loss(self._labels, self._labels)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  def testAllCorrectNoLossWeightWithPlaceholder(self):
    # [FUNCTION DOC]
    # Test: passes the labels as predictions through a float placeholder fed at run time; expects
    # a loss of about zero.
    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._np_labels.shape)
    loss = losses.log_loss(self._labels, tf_predictions)
    with self.cached_session():
      self.assertAlmostEqual(
          0.0, loss.eval(feed_dict={tf_predictions: self._np_labels}), 3)

  def testNonZeroLoss(self):
    # [FUNCTION DOC]
    # Test: computes the log loss on the fixture labels and predictions with default weights and
    # expects the hand-computed average value (within three decimal places).
    loss = losses.log_loss(self._labels, self._predictions)
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(self._expected_losses) / 6.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    # [FUNCTION DOC]
    # Test: passes a plain python number (2.3) as weights; expects the log loss to equal the
    # unweighted value multiplied by that weight.
    weights = 2.3
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    # [FUNCTION DOC]
    # Test: passes the weight 2.3 wrapped in a scalar constant tensor; expects the log loss to equal
    # the unweighted value multiplied by that weight.
    weights = 2.3
    loss = losses.log_loss(self._labels, self._predictions,
                           constant_op.constant(weights))
    with self.cached_session():
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithScalarTensorWeightAndPlaceholder(self):
    # [FUNCTION DOC]
    # Test: scalar tensor weight 2.3 with predictions supplied via a fully shaped placeholder; expects
    # the weight times the mean expected loss.
    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._np_predictions.shape)
    weights = 2.3
    loss = losses.log_loss(self._labels, tf_predictions,
                           constant_op.constant(weights))
    with self.cached_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             loss, 3)

  def testNonZeroLossWithScalarTensorWeightAndPlaceholderWithRankOnly(self):
    # [FUNCTION DOC]
    # Test: same as the placeholder test, but the placeholder only has a known rank (unknown sizes);
    # expects the same weighted loss.
    tf_predictions = array_ops.placeholder(dtypes.float32, shape=[None, None])
    weights = 2.3
    loss = losses.log_loss(self._labels, tf_predictions,
                           constant_op.constant(weights))
    with self.cached_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             loss, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: weights (1.2, 3.4) of shape two by one; expects the weighted sum of expected losses
    # divided by six.
    weights = constant_op.constant((1.2, 3.4), shape=(2, 1))
    expected_losses = np.multiply(
        self._expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 6.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeightsSomeZero(self):
    # [FUNCTION DOC]
    # Test: weights (1.2, 0) of shape two by one, so the second row is ignored; expects the weighted
    # sum divided by three, the number of elements with non-zero weight.
    weights = constant_op.constant((1.2, 0), shape=(2, 1))
    expected_losses = np.multiply(self._expected_losses,
                                  np.asarray([1.2, 1.2, 1.2, 0, 0, 0]).reshape(
                                      (2, 3)))
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 3.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeightsSomeZero(self):
    # [FUNCTION DOC]
    # Test: same as the previous test but weights constructed from a list with shape two by one;
    # expects the same result.
    weights = constant_op.constant([1.2, 0], shape=[2, 1])
    expected_losses = np.multiply(self._expected_losses,
                                  np.asarray([1.2, 1.2, 1.2, 0, 0, 0]).reshape(
                                      (2, 3)))
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 3.0,
                             self.evaluate(loss), 3)

  def testWeightsWithSameNumDimsButWrongShapeThrowsException(self):
    # [FUNCTION DOC]
    # Test: weights of shape two by four (right rank, wrong size) for 2x3 inputs; expects a value error.
    weights = constant_op.constant(np.random.normal(size=(2, 4)), shape=[2, 4])
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.log_loss(self._labels, self._predictions, weights)

  def testNonZeroLossWithMeasurementSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: per-element weights with one zero; expects the weighted sum divided by five, the count
    # of non-zero weights.
    weights = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    loss = losses.log_loss(
        self._labels,
        self._predictions,
        constant_op.constant(
            weights, shape=(2, 3)))
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 5.0,
                             self.evaluate(loss), 3)

  def testNonZeroLossWithMeasurementSpecificWeightsWithPlaceholder(self):
    # [FUNCTION DOC]
    # Test: as the previous test but predictions are a fully shaped placeholder fed at run time.
    weights = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    tf_predictions = array_ops.placeholder(dtypes.float32, shape=[2, 3])
    loss = losses.log_loss(
        self._labels,
        tf_predictions,
        constant_op.constant(
            weights, shape=(2, 3)))

    with self.cached_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(-np.sum(expected_losses) / 5.0, loss, 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    # [FUNCTION DOC]
    # Test: only the last element has non-zero weight (2); expects minus the sum of the weighted
    # expected losses.
    weights = np.array([0, 0, 0, 0, 0, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    loss = losses.log_loss(
        self._labels,
        self._predictions,
        constant_op.constant(
            weights, shape=(2, 3)))
    with self.cached_session():
      self.assertAlmostEqual(-np.sum(expected_losses), self.evaluate(loss), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZeroWithPlaceholder(self):
    # [FUNCTION DOC]
    # Test: as the most-zero weights test but predictions come from a placeholder fed at run time.
    weights = np.array([0, 0, 0, 0, 0, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    tf_predictions = array_ops.placeholder(dtypes.float32, shape=[2, 3])
    tf_weights = constant_op.constant(weights, shape=(2, 3))
    loss = losses.log_loss(self._labels, tf_predictions, tf_weights)

    with self.cached_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(-np.sum(expected_losses), loss, 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    # [FUNCTION DOC]
    # Test: passes all-zero element weights; expects the log loss to evaluate to zero rather
    # than failing with a division by zero.
    tf_weights = array_ops.zeros(shape=(2, 3))
    loss = losses.log_loss(self._labels, self._predictions, tf_weights)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)


class HingeLossTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the hinge loss. Uses small logits and labels in zero or one form and checks
  # outside-margin, inside-margin and misclassified examples. Methods opt into graph mode.

  def testIncompatibleShapes(self):
    # [FUNCTION DOC]
    # Test: passes logits or predictions of shape two by one together with labels of shape
    # two (rank mismatch) to the hinge loss and expects a value error when evaluated.
    with self.cached_session():
      logits = constant_op.constant([[-1.0], [2.1]])
      labels = constant_op.constant([0.0, 1.0])
      with self.assertRaises(ValueError):
        _ = losses.hinge_loss(labels, logits).eval()

  @test_util.run_deprecated_v1
  def testAllOutsideMargin(self):
    # [FUNCTION DOC]
    # Test (graph mode): every example is correctly classified beyond the margin; expects loss zero.
    with self.cached_session():
      logits = constant_op.constant([1.2, -1.4, -1.0, 2.1])
      labels = constant_op.constant([1.0, 0.0, 0.0, 1.0])
      loss = losses.hinge_loss(labels, logits)
      self.assertAllClose(loss, 0.0, atol=1e-3)

  @test_util.run_deprecated_v1
  def testSomeInsideMargin(self):
    # [FUNCTION DOC]
    # Test (graph mode): two examples are on the correct side but inside the margin; expects a
    # small positive average loss of 0.175.
    with self.cached_session():
      logits = constant_op.constant([[-0.7], [-1.4], [1.4], [0.6]])
      labels = constant_op.constant([[0.0], [0.0], [1.0], [1.0]])
      loss = losses.hinge_loss(labels, logits)
      # Examples 1 and 4 are on the correct side of the hyperplane but within
      # the margin so they incur some (small) loss.
      self.assertAllClose(loss, 0.175, atol=1e-3)

  @test_util.run_deprecated_v1
  def testSomeMisclassified(self):
    # [FUNCTION DOC]
    # Test (graph mode): three-dimensional inputs where two examples are on the wrong side; expects
    # an average loss of 0.875.
    with self.cached_session():
      logits = constant_op.constant([[[1.2], [0.4], [-1.0], [-1.1]]])
      labels = constant_op.constant([[[1.0], [0.0], [0.0], [1.0]]])
      loss = losses.hinge_loss(labels, logits)
      # Examples 2 and 4 are on the wrong side of the hyperplane so they incur
      # some (fairly large) loss.
      self.assertAllClose(loss, 0.875, atol=1e-3)


class HuberLossTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Huber loss (quadratic near zero, linear for large errors, with a delta
  # threshold defaulting to one). Covers all-quadratic, all-linear, mixed, and custom delta cases.

  def testIncompatibleShapes(self):
    # [FUNCTION DOC]
    # Test: passes logits or predictions of shape two by one together with labels of shape
    # two (rank mismatch) to the Huber loss and expects a value error when evaluated.
    with self.cached_session():
      predictions = constant_op.constant([[-1.0], [2.1]])
      labels = constant_op.constant([0.0, 1.0])
      with self.assertRaises(ValueError):
        _ = losses.huber_loss(labels, predictions).eval()

  @test_util.run_deprecated_v1
  def testAllQuadratic(self):
    # [FUNCTION DOC]
    # Test (graph mode): all errors are below the threshold; expects half the mean squared error.
    with self.cached_session():
      predictions = constant_op.constant([1.5, -1.4, -1.0, 0.0])
      labels = constant_op.constant([1.0, -1.0, 0.0, 0.5])
      loss = losses.huber_loss(labels, predictions)
      self.assertAllClose(
          loss, 0.5 * (0.25 + 0.16 + 1.0 + 0.25) / 4., atol=1e-5)

  @test_util.run_deprecated_v1
  def testAllLinear(self):
    # [FUNCTION DOC]
    # Test (graph mode): all errors exceed the threshold; expects the mean absolute error minus 0.5.
    with self.cached_session():
      predictions = constant_op.constant([1.5, -1.4, -1.0, 0.0])
      labels = constant_op.constant([0.0, 1.0, 0.0, 1.5])
      loss = losses.huber_loss(labels, predictions)
      self.assertAllClose(loss, (1.5 + 2.4 + 1.0 + 1.5) / 4. - 0.5, atol=1e-5)

  @test_util.run_deprecated_v1
  def testMixedQuadraticLinear(self):
    # [FUNCTION DOC]
    # Test (graph mode): two rows, one all-quadratic and one all-linear; expects the average of
    # the two row losses.
    with self.cached_session():
      predictions = constant_op.constant([[1.5, -1.4, -1.0, 0.0],
                                          [1.5, -1.4, -1.0, 0.0]])
      labels = constant_op.constant([[1.0, -1.0, 0.0, 0.5],
                                     [0.0, 1.0, 0.0, 1.5]])
      loss = losses.huber_loss(labels, predictions)
      quadratic = 0.5 * (0.25 + 0.16 + 1.0 + 0.25) / 4.
      linear = (1.5 + 2.4 + 1.0 + 1.5) / 4. - 0.5
      expected_loss = (quadratic + linear) / 2.
      self.assertAllClose(loss, expected_loss, atol=1e-5)

  def testAllQuadraticDelta(self):
    # [FUNCTION DOC]
    # Test: delta of 0.5 with all errors within it; expects half the mean of squared errors.
    with self.cached_session():
      delta = 0.5
      predictions = constant_op.constant([1.5, -1.4, -0.5, 0.0])
      labels = constant_op.constant([1.0, -1.0, 0.0, 0.5])
      expected = 0.5 * np.array([0.5**2, 0.4**2, 0.5**2, 0.5**2]).mean()
      loss = losses.huber_loss(labels, predictions, delta=delta)
      self.assertAllClose(expected, self.evaluate(loss), atol=1e-5)

  def testAllLinearDelta(self):
    # [FUNCTION DOC]
    # Test: delta of 0.5 with all errors beyond it; expects delta times mean absolute error minus
    # half delta squared.
    delta = 0.5
    predictions = constant_op.constant([1.5, -1.4, -1.0, 0.0])
    labels = constant_op.constant([0.0, 1.0, 0.0, 1.5])
    expected = delta * np.array([1.5, 2.4, 1.0, 1.5]).mean()
    expected -= 0.5 * delta**2
    loss = losses.huber_loss(labels, predictions, delta=delta)
    with self.cached_session():
      self.assertAllClose(expected, self.evaluate(loss), atol=1e-5)


@test_util.run_deprecated_v1
class MeanSquaredErrorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the mean squared error loss. Runs in graph (deprecated v1) mode; same 2x3
  # integer fixture as the absolute difference tests, with an expected unweighted loss of 49.5.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the fixture: a 2x3 integer predictions constant and a 2x3 integer labels constant,
    # after running the parent setUp. Stores both on the instance; returns nothing.
    super(MeanSquaredErrorTest, self).setUp()
    self._predictions = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    self._labels = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    # [FUNCTION DOC]
    # Test: calls the mean squared error loss function with weights explicitly set to nothing and expects a
    # value error to be raised.
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.mean_squared_error(
            self._predictions, self._predictions, weights=None)

  @test_util.run_deprecated_v1
  def testScalar(self):
    # [FUNCTION DOC]
    # Test (graph mode): scalar zero predictions and labels; expects exactly zero loss, showing
    # rank-zero inputs are supported.
    with self.cached_session():
      self.assertEqual(
          0.0,
          losses.mean_squared_error(predictions=constant_op.constant(0),
                                    labels=constant_op.constant(0)).eval())

  @test_util.run_deprecated_v1
  def testAllCorrectNoLossWeight(self):
    # [FUNCTION DOC]
    # Test: uses identical values for predictions and labels with default weights and expects
    # the mean squared error loss to evaluate to zero (within three decimal places).
    loss = losses.mean_squared_error(self._predictions, self._predictions)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLoss(self):
    # [FUNCTION DOC]
    # Test: computes the mean squared error loss on the fixture labels and predictions with default weights and
    # expects the hand-computed average value (within three decimal places).
    loss = losses.mean_squared_error(self._labels, self._predictions)
    with self.cached_session():
      self.assertAlmostEqual(49.5, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithPythonScalarWeight(self):
    # [FUNCTION DOC]
    # Test: passes a plain python number (2.3) as weights; expects the mean squared error loss to equal the
    # unweighted value multiplied by that weight.
    weights = 2.3
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(49.5 * weights, self.evaluate(loss), 3)

  @test_util.run_deprecated_v1
  def testNonZeroLossWithScalarTensorWeight(self):
    # [FUNCTION DOC]
    # Test: passes the weight 2.3 wrapped in a scalar constant tensor; expects the mean squared error loss to equal
    # the unweighted value multiplied by that weight.
    weights = 2.3
    loss = losses.mean_squared_error(self._labels, self._predictions,
                                     constant_op.constant(weights))
    with self.cached_session():
      self.assertAlmostEqual(49.5 * weights, self.evaluate(loss), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: passes one weight per batch element (a one-dimensional or column weight tensor) to
    # the mean squared error loss and expects the hand-computed weighted result.
    weights = constant_op.constant([1.2, 3.4], shape=(2, 1))
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(767.8 / 6.0, self.evaluate(loss), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: like the one-dimensional batch weights test but the weights are given as a column
    # tensor of shape two by one; expects the same weighted result of the mean squared error loss.
    weights = constant_op.constant([1.2, 3.4], shape=[2, 1])
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(767.8 / 6.0, self.evaluate(loss), 3)

  def testNonZeroLossWithSampleSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: passes a weight for every element of the 2x3 inputs; expects the hand-computed
    # weighted mean squared error loss (within three decimal places).
    weights = constant_op.constant([3, 6, 5, 0, 4, 2], shape=[2, 3])
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(587 / 5.0, self.evaluate(loss), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    # [FUNCTION DOC]
    # Test: passes element weights that are zero except for the last element; expects the mean squared error loss
    # to be determined by that single weighted element.
    weights = constant_op.constant([0, 0, 0, 0, 0, 2], shape=[2, 3])
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(18.0, self.evaluate(loss), 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    # [FUNCTION DOC]
    # Test: passes all-zero element weights; expects the mean squared error loss to evaluate to zero rather
    # than failing with a division by zero.
    weights = array_ops.zeros((2, 3))
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.cached_session():
      self.assertAlmostEqual(0.0, self.evaluate(loss), 3)


@test_util.run_deprecated_v1
class MeanPairwiseSquaredErrorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the mean pairwise squared error loss, which compares differences between
  # pairs of elements within each example. Runs in graph (deprecated v1) mode. The fixture
  # computes expected per-example losses by brute force; helper methods check valid and
  # invalid weights with both static and placeholder inputs.

  def setUp(self):
    # [FUNCTION DOC]
    # Builds the fixture: 2x3 integer predictions and labels, then loops over every pair of
    # elements in each example to accumulate squared differences of differences, dividing by three
    # to store the expected per-example losses. Returns nothing.
    super(MeanPairwiseSquaredErrorTest, self).setUp()
    self._predictions = np.array([[4, 8, 12], [8, 1, 3]])
    self._labels = np.array([[1, 9, 2], [-5, -5, 7]])

    batch_size, dims = self._labels.shape  # pylint: disable=unpacking-non-sequence

    # Compute the expected loss 'manually'.
    total = np.zeros((batch_size,))
    for b in range(batch_size):
      for i in range(dims - 1):
        for j in range(i + 1, dims):
          x = self._predictions[b, i].item() - self._predictions[b, j].item()
          y = self._labels[b, i].item() - self._labels[b, j].item()
          diff = (x - y)
          total[b] += (diff * diff)

    self._expected_losses = np.divide(total, 3.0)

  def testValueErrorThrownWhenWeightIsNone(self):
    # [FUNCTION DOC]
    # Test: calls the mean pairwise squared error loss function with weights explicitly set to nothing and expects a
    # value error to be raised.
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.mean_pairwise_squared_error(
            predictions=constant_op.constant(self._labels),
            labels=constant_op.constant(self._labels),
            weights=None)

  def _test_valid_weights(
      self, labels, predictions, expected_loss, weights=1.0):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine for the valid-weight tests in this class.
    # Takes labels, predictions, the expected loss and optional weights (default one). Evaluates the
    # loss on the static inputs, then rebuilds it with placeholders of matching shapes and feeds
    # the values at run time; asserts both equal the expected loss to three places. Returns nothing.
    with self.cached_session():
      static_inputs_op = losses.mean_pairwise_squared_error(
          predictions=predictions, labels=labels, weights=weights)
      self.assertAlmostEqual(
          expected_loss, self.evaluate(static_inputs_op), places=3)

      predictions_placeholder = array_ops.placeholder(
          dtypes.float32, shape=np.asarray(predictions.shape))
      labels_placeholder = array_ops.placeholder(
          dtypes.int32, shape=np.asarray(labels.shape))
      weights_placeholder = array_ops.placeholder(
          dtypes.float32, shape=np.asarray(weights).shape)
      dynamic_inputs_op = losses.mean_pairwise_squared_error(
          predictions=predictions_placeholder,
          labels=labels_placeholder,
          weights=weights_placeholder)
      feed_dict = {
          predictions_placeholder: predictions,
          labels_placeholder: labels,
          weights_placeholder: weights,
      }
      self.assertAlmostEqual(
          expected_loss, dynamic_inputs_op.eval(feed_dict=feed_dict), places=3)

  def testAllCorrectNoLossWeight(self):
    # [FUNCTION DOC]
    # Test: labels used as predictions with default weights; expects zero via the helper.
    self._test_valid_weights(
        self._labels, self._labels, expected_loss=0.0)

  def testNonZeroLoss(self):
    # [FUNCTION DOC]
    # Test: fixture labels and predictions with default weights; expects the sum of the brute
    # force per-example losses, checked by the helper for static and placeholder inputs.
    self._test_valid_weights(
        self._labels, self._predictions,
        expected_loss=np.sum(self._expected_losses))

  def testGradientWithZeroWeight(self):
    # [FUNCTION DOC]
    # Test: in a fresh graph with a fixed random seed, multiplies ones by a trainable weight matrix
    # to get predictions, uses the same predictions as labels with weight zero, computes optimizer
    # gradients and evaluates them; asserts none contain not-a-number values.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)

      inputs = array_ops.ones((2, 3))
      weights = variable_scope.get_variable(
          'weights',
          shape=[3, 4],
          initializer=init_ops.truncated_normal_initializer())
      predictions = math_ops.matmul(inputs, weights)

      optimizer = momentum_lib.MomentumOptimizer(
          learning_rate=0.001, momentum=0.9)
      loss = losses.mean_pairwise_squared_error(predictions, predictions, 0)

      gradients_to_variables = optimizer.compute_gradients(loss)

      init_op = variables.global_variables_initializer()

      with self.cached_session() as sess:
        self.evaluate(init_op)
        for grad, _ in gradients_to_variables:
          np_grad = self.evaluate(grad)
          self.assertFalse(np.isnan(np_grad).any())

  def testNonZeroLossWithPythonScalarWeight(self):
    # [FUNCTION DOC]
    # Test: passes a plain python number (2.3) as weights; expects the mean pairwise squared error loss to equal the
    # unweighted value multiplied by that weight.
    weight = 2.3
    self._test_valid_weights(
        self._labels, self._predictions,
        expected_loss=weight * np.sum(self._expected_losses),
        weights=weight)

  def testNonZeroLossWithScalarTensorWeight(self):
    # [FUNCTION DOC]
    # Test: passes the weight 2.3 wrapped in a scalar constant tensor; expects the mean pairwise squared error loss to equal
    # the unweighted value multiplied by that weight.
    weights = 2.3
    loss = losses.mean_pairwise_squared_error(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        weights=constant_op.constant(weights))
    with self.cached_session():
      self.assertAlmostEqual(weights * np.sum(self._expected_losses),
                             self.evaluate(loss), 3)

  def testNonZeroLossWithScalarZeroWeight(self):
    # [FUNCTION DOC]
    # Test: scalar weight of zero; expects zero loss via the helper.
    self._test_valid_weights(
        self._labels, self._predictions, expected_loss=0.0, weights=0.0)

  def test3d(self):
    # [FUNCTION DOC]
    # Test: three-dimensional (2x3x3) labels and predictions with default weights; expects 137.5.
    labels = np.array([
        [[1, 9, 2], [12, 11, 10], [9, 8, 7]],
        [[-5, -5, 7], [6, 5, 4], [3, 2, 1]],
    ])
    predictions = np.array([
        [[4, 8, 12], [1, 2, 3], [4, 5, 6]],
        [[8, 1, 3], [7, 8, 9], [10, 11, 12]],
    ])
    self._test_valid_weights(labels, predictions, expected_loss=137.5)

  def test3dWeightedScalar(self):
    # [FUNCTION DOC]
    # Test: same 3-D data with scalar weight 3; expects three times 137.5.
    labels = np.array([
        [[1, 9, 2], [12, 11, 10], [9, 8, 7]],
        [[-5, -5, 7], [6, 5, 4], [3, 2, 1]],
    ])
    predictions = np.array([
        [[4, 8, 12], [1, 2, 3], [4, 5, 6]],
        [[8, 1, 3], [7, 8, 9], [10, 11, 12]],
    ])
    weight = 3.0
    self._test_valid_weights(
        labels, predictions, expected_loss=weight * 137.5, weights=weight)

  def _test_invalid_weights(
      self, labels, predictions, weights=1.0):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine for invalid-weight tests in this class.
    # Takes labels, predictions and weights (default one). First asserts a value error whose message
    # says weights cannot be broadcast, using static inputs. Then builds the loss with unshaped
    # placeholders and asserts evaluating it with the same values raises an operation error with the
    # same message. Returns nothing.
    expected_error_msg = 'weights can not be broadcast to values'

    # Static check.
    with self.assertRaisesRegex(ValueError, expected_error_msg):
      losses.mean_pairwise_squared_error(
          predictions=predictions, labels=labels, weights=weights)

    # Dynamic check.
    predictions_placeholder = array_ops.placeholder(dtypes.float32)
    labels_placeholder = array_ops.placeholder(dtypes.int32)
    weights_placeholder = array_ops.placeholder(dtypes.float32)
    dynamic_inputs_op = losses.mean_pairwise_squared_error(
        predictions=predictions_placeholder,
        labels=labels_placeholder,
        weights=weights_placeholder)
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.OpError, expected_error_msg):
        dynamic_inputs_op.eval(feed_dict={
            predictions_placeholder: predictions,
            labels_placeholder: labels,
            weights_placeholder: weights,
        })

  def testInvalid3dWeighted2x0(self):
    # [FUNCTION DOC]
    # Test: 3-D inputs with a weight vector of length two (not broadcastable); expects the static and
    # dynamic broadcast errors from the helper.
    labels = np.array([
        [[1, 9, 2], [12, 11, 10], [9, 8, 7]],
        [[-5, -5, 7], [6, 5, 4], [3, 2, 1]],
    ])
    predictions = np.array([
        [[4, 8, 12], [1, 2, 3], [4, 5, 6]],
        [[8, 1, 3], [7, 8, 9], [10, 11, 12]],
    ])
    self._test_invalid_weights(
        labels, predictions, weights=np.asarray((1.2, 3.4)))

  def test3dWeighted2x3x3(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a TODO in the body says the expected value (nine times 137.5 for all-ones weights of the
    # full input shape) "doesn't look right"; the test pins current behavior rather than a verified spec.
    # Test: 3-D inputs with weights of ones in the full 2x3x3 shape; checks the loss equals nine times 137.5.
    labels = np.array([
        [[1, 9, 2], [12, 11, 10], [9, 8, 7]],
        [[-5, -5, 7], [6, 5, 4], [3, 2, 1]],
    ])
    predictions = np.array([
        [[4, 8, 12], [1, 2, 3], [4, 5, 6]],
        [[8, 1, 3], [7, 8, 9], [10, 11, 12]],
    ])
    self._test_valid_weights(
        # TODO(ptucker): This doesn't look right.
        labels,
        predictions,
        expected_loss=9 * 137.5,
        weights=np.ones((2, 3, 3)))

  def testLossWithAllZeroBatchSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: weights of zeros with shape two by one; expects zero loss via the helper.
    self._test_valid_weights(
        self._labels, self._predictions, expected_loss=0.0,
        weights=np.zeros((2, 1)))

  def testLossIsAssociativeAcrossBatchElements(self):
    # [FUNCTION DOC]
    # Test: in a fresh graph with a fixed seed, makes two random single-image label and prediction pairs
    # (shape 1x3x4x1) and computes the loss on each and on their concatenation along the batch axis. Asserts
    # both single losses are positive and the combined loss equals their sum to five places.
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)

      height = 3
      width = 4
      shape = (1, height, width, 1)

      labels0 = random_ops.random_uniform(
          shape, minval=0, maxval=1, dtype=dtypes.float32)
      predictions0 = random_ops.random_uniform(
          shape, minval=0, maxval=1, dtype=dtypes.float32)

      labels1 = random_ops.random_uniform(
          shape, minval=0, maxval=1, dtype=dtypes.float32)
      predictions1 = random_ops.random_uniform(
          shape, minval=0, maxval=1, dtype=dtypes.float32)

      loss0 = losses.mean_pairwise_squared_error(
          labels=labels0,
          predictions=predictions0)
      loss1 = losses.mean_pairwise_squared_error(
          labels=labels1,
          predictions=predictions1)
      loss0_1 = losses.mean_pairwise_squared_error(
          labels=array_ops.concat([labels0, labels1], 0),
          predictions=array_ops.concat([predictions0, predictions1], 0))

      with self.cached_session() as session:
        loss0, loss1, loss0_1 = session.run([loss0, loss1, loss0_1])

        self.assertTrue(loss0 > 0)
        self.assertTrue(loss1 > 0)
        self.assertAlmostEqual(loss0 + loss1, loss0_1, 5)


@test_util.run_deprecated_v1
class CosineDistanceLossTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the cosine distance loss, computed along dimension two of 3x2x3 inputs. Runs in graph
  # (deprecated v1) mode. The fixture holds integer predictions and labels for three batches of two
  # unit vectors each.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the fixture: numpy predictions and labels, each six unit vectors reshaped to 3x2x3,
    # after running the parent setUp. Stores them on the instance; returns nothing.
    super(CosineDistanceLossTest, self).setUp()
    self._predictions = np.asarray([
        [1, 0, 0],  # Batch 1
        [0, 0, -1],
        [1, 0, 0],  # Batch 2
        [1, 0, 0],
        [0, 0, -1],  # Batch 3
        [1, 0, 0]
    ]).reshape((3, 2, 3))

    self._labels = np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                               [0, 0, 1], [0, 1, 0]]).reshape((3, 2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    # [FUNCTION DOC]
    # Test: calls the cosine distance loss function with weights explicitly set to nothing and expects a
    # value error to be raised.
    with self.cached_session():
      with self.assertRaises(ValueError):
        losses.cosine_distance(
            predictions=constant_op.constant(self._labels),
            labels=constant_op.constant(self._labels),
            dim=2,
            weights=None)

  def testAllCorrectNoWeights(self):
    # [FUNCTION DOC]
    # Test: labels used as predictions; expects a distance of about zero.
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._labels),
        labels=constant_op.constant(self._labels),
        dim=2)
    with self.cached_session():
      self.assertAlmostEqual(0, self.evaluate(loss), 5)

  def testPartiallyCorrectWithIntegerValues(self):
    # [FUNCTION DOC]
    # Test: fixture predictions against labels, where half the vectors match; expects the average
    # distance of one.
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2)
    with self.cached_session():
      self.assertAlmostEqual(1, self.evaluate(loss), 5)

  def testPartiallyCorrectFloatingPointValues(self):
    # [FUNCTION DOC]
    # Test: normalized floating point vectors where the first matches and the others are opposite or
    # orthogonal pairs; expects an average distance of 1.0 to five places.
    predictions = np.matrix(
        ('0.819031913261206 0.567041924552012 0.087465312324590;'
         '-0.665139432070255 -0.739487441769973 -0.103671883216994;'
         '0.707106781186548 -0.707106781186548 0'))
    labels = np.matrix(('0.819031913261206 0.567041924552012 0.087465312324590;'
                        '0.665139432070255 0.739487441769973 0.103671883216994;'
                        '0.707106781186548 0.707106781186548 0'))

    tf_preds = constant_op.constant(
        predictions, shape=(3, 1, 3), dtype=dtypes.float32)
    tf_labels = constant_op.constant(
        labels, shape=(3, 1, 3), dtype=dtypes.float32)
    loss = losses.cosine_distance(tf_labels, tf_preds, dim=2)

    with self.cached_session():
      self.assertAlmostEqual(1.0, self.evaluate(loss), 5)

  def testSampleSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: weights of shape 3x1x1 selecting only the first batch; expects a loss of exactly one.
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=np.asarray((1, 0, 0)).reshape((3, 1, 1)))
    with self.cached_session():
      self.assertEqual(1.0, self.evaluate(loss))

  def testMeasurementSpecificWeights(self):
    # [FUNCTION DOC]
    # Test: weights of shape 3x2x1 with one zero; expects exactly three quarters.
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=constant_op.constant(
            [1, 0, 0, 1, 1, 1], shape=(3, 2, 1)))
    with self.cached_session():
      self.assertEqual(3.0 / 4.0, self.evaluate(loss))

  def testMeasurementSpecificWeightsWithPlaceholderWithShape(self):
    # [FUNCTION DOC]
    # Test: same measurement weights but predictions come from a fully shaped placeholder fed at run
    # time; expects three quarters.
    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._labels.shape)
    loss = losses.cosine_distance(
        predictions=tf_predictions,
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=constant_op.constant(
            [1, 0, 0, 1, 1, 1], shape=(3, 2, 1)))
    with self.cached_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._predictions})
      self.assertEqual(3.0 / 4.0, loss)

  def testZeroLossWhenAllSampleSpecificWeightsAreZero(self):
    # [FUNCTION DOC]
    # Test: all-zero weights of shape 3x1x1; expects exactly zero.
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=array_ops.zeros((3, 1, 1)))
    with self.cached_session():
      self.assertEqual(0, self.evaluate(loss))

  def testZeroLossWhenAllMeasurementSpecificWeightsAreZero(self):
    # [FUNCTION DOC]
    # Test: all-zero weights of shape 3x2x1; expects exactly zero.
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=array_ops.zeros((3, 2, 1)))
    with self.cached_session():
      self.assertEqual(0, self.evaluate(loss))


class AddLossTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the loss-collection behavior of the losses API (whether losses get registered
  # in the global loss collection).

  def testNoCollectLossesBatch2(self):
    # [FUNCTION DOC]
    # Test: with two-example logits and labels, first asserts the global loss collection is empty,
    # then calls five loss functions with the collection disabled (absolute difference, log loss, mean
    # squared error, sigmoid and softmax cross entropy) and asserts the collection is still empty.
    logits = constant_op.constant([[1.2, 0.4, -1.0, -1.1]] * 2)
    labels = constant_op.constant([[1.0, 0.0, 0.0, 1.0]] * 2)
    self.assertFalse(util.get_losses())
    losses.absolute_difference(logits, labels, loss_collection=None)
    losses.log_loss(logits, labels, loss_collection=None)
    losses.mean_squared_error(logits, labels, loss_collection=None)
    losses.sigmoid_cross_entropy(logits, labels, loss_collection=None)
    losses.softmax_cross_entropy(logits, labels, loss_collection=None)
    self.assertFalse(util.get_losses())


class ComputeWeightedLossTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the weighted loss computation function across all reduction types. The fixture is a
  # read-only 3x2x4 array of consecutive numbers. Helper methods run valid and invalid weight cases
  # and verify results per reduction and the number of losses registered in the global collection.

  def setUp(self):
    # [FUNCTION DOC]
    # Builds the fixture: a 3x2x4 numpy array filled with consecutive values from zero, marked
    # read-only and stored on the instance together with its shape. Returns nothing.
    super(ComputeWeightedLossTest, self).setUp()
    self._shape = (3, 2, 4)
    raw_losses = np.zeros(self._shape)
    next_loss = 0.0
    for i in range(self._shape[0]):
      for j in range(self._shape[1]):
        for k in range(self._shape[2]):
          raw_losses[i][j][k] = next_loss
          next_loss += 1.0
    raw_losses.setflags(write=False)
    self._raw_losses = raw_losses

  def testUnweighted(self):
    # [FUNCTION DOC]
    # Test: for each reduction type, in a fresh graph, calls the function with no weights and with
    # weights of ones in eight different broadcast shapes. Asserts nine losses are registered and every
    # result equals the raw losses (no reduction), their sum (sum), or their mean (other reductions).
    for reduction in losses.Reduction.all():
      with ops.Graph().as_default() as g:
        self.assertEqual(0, len(util.get_losses()))
        raw_losses = self._raw_losses
        unweighted_losses = (
            losses.compute_weighted_loss(raw_losses, reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 1, 1)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 1, 4)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 2, 1)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 2, 4)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((3, 1, 1)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((3, 1, 4)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((3, 2, 1)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones(self._shape), reduction=reduction)
        )
        self.assertEqual(9, len(util.get_losses()))
        with self.session(g):
          for unweighted_loss in unweighted_losses:
            if reduction == losses.Reduction.NONE:
              self.assertAllClose(self._raw_losses,
                                  self.evaluate(unweighted_loss))
            elif reduction == losses.Reduction.SUM:
              self.assertAllClose(
                  np.sum(self._raw_losses), self.evaluate(unweighted_loss))
            else:
              # reduction one of MEAN, SUM_OVER_NONZERO_WEIGHTS,
              # SUM_BY_NONZERO_WEIGHTS or SUM_OVER_BATCH_SIZE.
              self.assertAllClose(
                  np.mean(self._raw_losses), self.evaluate(unweighted_loss))

  def testUnweightedFromPlaceholder(self):
    # [FUNCTION DOC]
    # Test: like the unweighted test but the raw losses come from an unshaped placeholder fed at run
    # time and only three weight variants are used; asserts three losses are registered and the same values.
    for reduction in losses.Reduction.all():
      with ops.Graph().as_default() as g:
        self.assertEqual(0, len(util.get_losses()))
        raw_losses = array_ops.placeholder(dtype=dtypes.float32)
        feed_dict = {raw_losses: self._raw_losses}
        unweighted_losses = (
            losses.compute_weighted_loss(raw_losses, reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 1, 1)), reduction=reduction),
            losses.compute_weighted_loss(
                raw_losses, weights=np.ones((1, 1, 4)), reduction=reduction),
        )
        self.assertEqual(3, len(util.get_losses()))
        with self.session(g):
          for unweighted_loss in unweighted_losses:
            if reduction == losses.Reduction.NONE:
              self.assertAllClose(
                  self._raw_losses, unweighted_loss.eval(feed_dict))
            elif reduction == losses.Reduction.SUM:
              self.assertAllClose(
                  np.sum(self._raw_losses), unweighted_loss.eval(feed_dict))
            else:
              # reduction one of MEAN, SUM_OVER_NONZERO_WEIGHTS,
              # SUM_BY_NONZERO_WEIGHTS or SUM_OVER_BATCH_SIZE.
              self.assertAllClose(
                  np.mean(self._raw_losses), unweighted_loss.eval(feed_dict))

  def testScalarWeight(self):
    # [FUNCTION DOC]
    # Test: in a fresh graph, applies scalar weight 17 with default reduction; asserts exactly one loss
    # is registered and the result equals the mean of the weighted raw losses.
    with ops.Graph().as_default():
      self.assertEqual(0, len(util.get_losses()))
      weight = 17.0
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weight)
      self.assertEqual(1, len(util.get_losses()))
      with self.cached_session():
        self.assertAllClose(
            np.mean(weight * self._raw_losses), self.evaluate(weighted_loss))

  def _test_invalid_weights(self, weights):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine for the invalid-weight tests in this class.
    # Takes a weights value. In a fresh graph it asserts no losses are registered, that calling the
    # function with those weights raises a value error saying weights cannot be broadcast, then builds it
    # with a placeholder for weights and asserts evaluating with the same values raises an operation error
    # with the same message. Returns nothing.
    with ops.Graph().as_default():
      self.assertEqual(0, len(util.get_losses()))
      expected_error_msg = 'weights can not be broadcast to values'

      # Static check.
      with self.assertRaisesRegex(ValueError, expected_error_msg):
        losses.compute_weighted_loss(self._raw_losses, weights=weights)

      # Dynamic check.
      weights_placeholder = array_ops.placeholder(dtypes.float32)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights_placeholder)
      self.assertEqual(1, len(util.get_losses()))
      with self.cached_session():
        with self.assertRaisesRegex(errors_impl.OpError, expected_error_msg):
          weighted_loss.eval(feed_dict={weights_placeholder: weights})

  def testInvalidWeightTooManyDims(self):
    # [FUNCTION DOC]
    # Test: weights of rank four for rank-three losses; expects both the static and dynamic broadcast errors.
    self._test_invalid_weights(np.zeros(shape=(2, 2, 2, 2)))

  def testInvalidWeightMismatchedDim(self):
    # [FUNCTION DOC]
    # Test: losses reshaped to a trailing dimension of one and weights with a trailing dimension of two
    # (not broadcastable); performs the same static and dynamic error checks inline in a fresh graph.
    with ops.Graph().as_default():
      raw_losses = array_ops.reshape(self._raw_losses, shape=(3, 2, 4, 1))
      weights = np.ones(shape=(3, 2, 4, 2))
      expected_error_msg = 'weights can not be broadcast to values'
      self.assertEqual(0, len(util.get_losses()))

      # Static check.
      with self.assertRaisesRegex(ValueError, expected_error_msg):
        losses.compute_weighted_loss(raw_losses, weights=weights)

      # Dynamic check.
      weights_placeholder = array_ops.placeholder(dtypes.float32)
      weighted_loss = losses.compute_weighted_loss(
          raw_losses, weights=weights_placeholder)
      self.assertEqual(1, len(util.get_losses()))
      with self.cached_session():
        with self.assertRaisesRegex(errors_impl.OpError, expected_error_msg):
          weighted_loss.eval(feed_dict={weights_placeholder: weights})

  def testInvalid3Weight(self):
    # [FUNCTION DOC]
    # Test: a three-element weight tuple; expects broadcast errors from the invalid-weights helper.
    self._test_invalid_weights((17.0, 5.0, 2.0))

  def testInvalid3x1Weight(self):
    # [FUNCTION DOC]
    # Test: a 3x1 weight tuple; expects broadcast errors from the invalid-weights helper.
    self._test_invalid_weights(((17.0,), (5.0,), (2.0,),))

  def testInvalid3x2Weight(self):
    # [FUNCTION DOC]
    # Test: a 3x2 weight tuple; expects broadcast errors from the invalid-weights helper.
    self._test_invalid_weights((
        (17.0, 3.0),
        (5.0, 31.0),
        (2.0, 7.0),))

  def testInvalid1x2Weight(self):
    # [FUNCTION DOC]
    # Test: a 1x2 weight tuple; expects broadcast errors from the invalid-weights helper.
    self._test_invalid_weights((17.0, 3.0,),)

  def testInvalidScalar1DWeight(self):
    # [FUNCTION DOC]
    # Test: a one-element vector weight; expects broadcast errors from the invalid-weights helper.
    self._test_invalid_weights((17.0,),)

  def _test_valid_weights(self, weights):
    # [FUNCTION DOC]
    # HELPER: shared assertion routine for the valid-weight tests in this class.
    # KNOWN ISSUE: the chain of reduction checks has no final else branch, so a reduction type not
    # listed there would pass without asserting anything.
    # Takes a weights value. For each reduction type, in a fresh graph, asserts no losses are registered
    # beforehand and exactly one afterwards, and that the result equals: the weighted losses (no reduction),
    # their sum (sum), the sum over the sum of broadcast weights (mean), the sum over the count of non-zero
    # broadcast weights (the two non-zero-weights reductions), or the sum over element count (batch size).
    for reduction in losses.Reduction.all():
      with ops.Graph().as_default() as g:
        self.assertEqual(0, len(util.get_losses()))
        weighted_loss = losses.compute_weighted_loss(
            self._raw_losses, weights=weights, reduction=reduction)
        self.assertEqual(1, len(util.get_losses()))
        with self.session(g):
          weighted_losses = weights * self._raw_losses
          weighted_sum = np.sum(weighted_losses)
          if reduction == losses.Reduction.NONE:
            self.assertAllClose(weighted_losses, self.evaluate(weighted_loss))
          elif reduction == losses.Reduction.SUM:
            self.assertAllClose(weighted_sum, self.evaluate(weighted_loss))
          else:
            broadcast_weights = weights * np.ones_like(self._raw_losses)
            if reduction == losses.Reduction.MEAN:
              self.assertAllClose(weighted_sum / np.sum(broadcast_weights),
                                  self.evaluate(weighted_loss))
            elif (reduction == losses.Reduction.SUM_OVER_NONZERO_WEIGHTS or
                  reduction == losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
              self.assertAllClose(
                  weighted_sum / np.count_nonzero(broadcast_weights),
                  self.evaluate(weighted_loss))
            elif reduction == losses.Reduction.SUM_OVER_BATCH_SIZE:
              self.assertAllClose(weighted_sum / self._raw_losses.size,
                                  self.evaluate(weighted_loss))

  def test1x1x1Weight(self):
    # [FUNCTION DOC]
    # Test: weights of shape 1x1x1 checked by the valid-weights helper for all reductions.
    self._test_valid_weights((((17.0,),),))

  def test1x2x1Weight(self):
    # [FUNCTION DOC]
    # Test: weights of shape 1x2x1 checked by the valid-weights helper for all reductions.
    self._test_valid_weights((((17.0,), (3.0,),),))

  def test1x1x4Weight(self):
    # [FUNCTION DOC]
    # Test: weights of shape 1x1x4 (including a zero) checked by the valid-weights helper.
    self._test_valid_weights((((17.0, 0.0, 2.0, 5.0),),))

  def test3x1x1Weight(self):
    # [FUNCTION DOC]
    # Test: weights of shape 3x1x1 checked by the valid-weights helper for all reductions.
    self._test_valid_weights((((17.0,),), ((5.0,),), ((2.0,),),))

  def test3x2x1Weight(self):
    # [FUNCTION DOC]
    # Test: weights of shape 3x2x1 checked by the valid-weights helper for all reductions.
    self._test_valid_weights((
        ((17.0,), (3.0,)),
        ((5.0,), (31.0,)),
        ((2.0,), (7.0,)),
    ))

  def test3x1x4Weight(self):
    # [FUNCTION DOC]
    # Test: weights of shape 3x1x4 (including a zero) checked by the valid-weights helper.
    self._test_valid_weights((
        ((17.0, 0.0, 2.0, 5.0),),
        ((5.0, 31.0, 17.0, 5.0),),
        ((7.0, 3.0, 11.0, 5.0),),
    ))

  def test1x2x4Weight(self):
    # [FUNCTION DOC]
    # Test: weights of shape 1x2x4 (including a zero) checked by the valid-weights helper.
    self._test_valid_weights(((
        (17.0, 0.0, 2.0, 5.0),
        (3.0, 13.0, 11.0, 2.0),
    ),))

  def test3x2x4Weight(self):
    # [FUNCTION DOC]
    # Test: weights of the full 3x2x4 shape (including zeros) checked by the valid-weights helper.
    self._test_valid_weights((
        ((17.0, 0.0, 2.0, 5.0), (3.0, 13.0, 11.0, 2.0),),
        ((5.0, 31.0, 17.0, 5.0), (13.0, 3.0, 0.0, 11.0),),
        ((0.0, 3.0, 11.0, 5.0), (13.0, 11.0, 1.0, 7.0),),
    ))


if __name__ == '__main__':
  test.main()
