# [MODULE DOC]
# Unit tests for the streaming metrics library (the module of graph-mode, variable-backed metrics).
# Each metric is built as a pair of a value tensor and an update operation; the tests check the
# variables and collections a metric creates, idempotent reads, results for weighted and unweighted
# data, shape-compatibility errors, and numerically exact expected values. The file starts with small
# shared helpers (queue enqueueing, conversion of dense indicator labels to sparse values, nan and
# variable assertions, test-data generators and wrappers around the top-k metrics) followed by one
# test class per metric: mean, mean tensor, accuracy, precision, recall, AUC, specificity at
# sensitivity, sensitivity at specificity, precision and recall at thresholds, precision/recall/average
# precision at k, mean absolute/relative/squared/root-squared error, mean cosine distance, percentage
# below threshold, mean IOU, mean per-class accuracy, and the confusion-count metrics (true/false
# positives/negatives, with and without thresholds).
# Most tests are decorated to run only as deprecated graph-mode tests and reset the default graph in
# their setup. Importing the module defines a module-level not-a-number constant and registers two
# gradient modules via import side effects. Running the file as a script starts the test runner.
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
"""Tests for metrics."""

import functools
import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.data_flow_grad  # pylint: disable=unused-import
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test

NAN = float('nan')


def _enqueue_vector(sess, queue, values, shape=None):
  # [FUNCTION DOC]
  # HELPER: small shared utility for the queue-based tests in this module.
  # Enqueues one element built from a list of values into a queue, using the supplied session.
  # Inputs: sess is the session used to run the enqueue; queue is a queue whose first dtype is used as
  # the element dtype; values is a (possibly nested) list of numbers; shape is an optional target
  # shape and when it is missing or empty it defaults to one row with as many columns as values has
  # entries. Builds a constant of that dtype and shape and runs the queue's enqueue operation.
  # Returns nothing; the side effect is one more element in the queue.
  if not shape:
    shape = (1, len(values))
  dtype = queue.dtypes[0]
  sess.run(
      queue.enqueue(constant_op.constant(
          values, dtype=dtype, shape=shape)))


def _binary_2d_label_to_2d_sparse_value(labels):
  # [FUNCTION DOC]
  # HELPER: converts dense indicator labels into the sparse label form that the at-k metric tests feed in.
  # KNOWN ISSUE: the docstring says the result has shape batch size by "the number of 1 values in each
  # row", but the code sets the dense shape to the number of rows by the full number of columns of the
  # first row (the number of classes), regardless of how many ones are present.
  # Converts a dense two-dimensional binary indicator (list of rows of zeros and ones) into a sparse
  # tensor value. For every row, each position holding a one produces one entry whose index is the
  # row number and the running count of ones seen so far in that row, and whose value is the column
  # position (class id) of that one. Raises an assertion error if any entry is neither zero nor one.
  # Input: labels, a list of equal-length rows. Output: a sparse tensor value with int64 indices,
  # values and dense shape.
  """Convert dense 2D binary indicator to sparse ID.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator, shape [batch_size, num_classes].

  Returns:
    `SparseTensorValue` of shape [batch_size, num_classes], where num_classes
    is the number of `1` values in each row of `labels`. Values are indices
    of `1` values along the last dimension of `labels`.
  """
  indices = []
  values = []
  batch = 0
  for row in labels:
    label = 0
    xi = 0
    for x in row:
      if x == 1:
        indices.append([batch, xi])
        values.append(label)
        xi += 1
      else:
        assert x == 0
      label += 1
    batch += 1
  shape = [len(labels), len(labels[0])]
  return sparse_tensor.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64), np.array(shape, np.int64))


def _binary_2d_label_to_1d_sparse_value(labels):
  # [FUNCTION DOC]
  # HELPER: converts dense indicator labels into the one-label-per-example sparse form for the at-k tests.
  # Converts a dense two-dimensional binary indicator whose every row has exactly one 1 into a sparse
  # tensor value of rank one. Each example contributes one entry whose index is its row number and
  # whose value is the class id (column position) of the single one. Raises an assertion error when an
  # entry is neither zero nor one, and raises a value error when the collected indices are not exactly
  # one per row (a row with no ones or with several ones).
  # Input: labels, a list of equal-length rows. Output: a sparse tensor value with int64 indices,
  # values and dense shape equal to the number of rows.
  """Convert dense 2D binary indicator to sparse ID.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator, shape [batch_size, num_classes]. Each
    row must contain exactly 1 `1` value.

  Returns:
    `SparseTensorValue` of shape [batch_size]. Values are indices of `1` values
    along the last dimension of `labels`.

  Raises:
    ValueError: if there is not exactly 1 `1` value per row of `labels`.
  """
  indices = []
  values = []
  batch = 0
  for row in labels:
    label = 0
    xi = 0
    for x in row:
      if x == 1:
        indices.append([batch])
        values.append(label)
        xi += 1
      else:
        assert x == 0
      label += 1
    batch += 1
  if indices != [[i] for i in range(len(labels))]:
    raise ValueError('Expected 1 label/example, got %s.' % indices)
  shape = [len(labels)]
  return sparse_tensor.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64), np.array(shape, np.int64))


def _binary_3d_label_to_sparse_value(labels):
  # [FUNCTION DOC]
  # HELPER: converts dense indicator labels into sparse labels for the three-dimensional at-k tests.
  # KNOWN ISSUE: the docstring describes the input as a "Dense 2D binary indicator tensor" but the
  # code iterates three levels deep and builds three-dimensional indices, so the input must be a
  # three-dimensional (batch, example, class) nested list.
  # Converts a nested list of binary indicators into a sparse tensor value. For each first-level and
  # second-level position, every class position holding a one yields an entry whose index is the two
  # outer positions plus the running count of ones in that innermost row, and whose value is the class
  # id. Raises an assertion error when an entry is neither zero nor one. The dense shape is taken from
  # the lengths of the outer, middle and innermost lists of the first element at each level.
  # Output: a sparse tensor value with int64 indices, values and dense shape.
  """Convert dense 3D binary indicator tensor to sparse tensor.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator tensor.

  Returns:
    `SparseTensorValue` whose values are indices along the last dimension of
    `labels`.
  """
  indices = []
  values = []
  for d0, labels_d0 in enumerate(labels):
    for d1, labels_d1 in enumerate(labels_d0):
      d2 = 0
      for class_id, label in enumerate(labels_d1):
        if label == 1:
          values.append(class_id)
          indices.append([d0, d1, d2])
          d2 += 1
        else:
          assert label == 0
  shape = [len(labels), len(labels[0]), len(labels[0][0])]
  return sparse_tensor.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64), np.array(shape, np.int64))


def _assert_nan(test_case, actual):
  # [FUNCTION DOC]
  # HELPER: shared assertion used by the at-k metric test wrappers.
  # Asserts, on the given test case, that the actual number is not-a-number. The failure message
  # reports the offending value. Inputs: test_case, the running test case; actual, a floating-point
  # number. Returns nothing; raises an assertion failure if the value is a regular number.
  test_case.assertTrue(math.isnan(actual), 'Expected NAN, got %s.' % actual)


def _assert_metric_variables(test_case, expected):
  # [FUNCTION DOC]
  # HELPER: shared assertion used by the metric variable-creation tests.
  # Asserts that the names of all local variables in the current graph, and separately the names of
  # all variables recorded in the metric-variables graph collection, are each exactly equal (as sets,
  # ignoring order) to the expected collection of names. Inputs: test_case, the running test case;
  # expected, an iterable of full variable names. Returns nothing; raises an assertion failure on any
  # difference in either set.
  test_case.assertEqual(
      set(expected), set(v.name for v in variables.local_variables()))
  test_case.assertEqual(
      set(expected),
      set(v.name for v in ops.get_collection(ops.GraphKeys.METRIC_VARIABLES)))


def _test_values(shape):
  # [FUNCTION DOC]
  # HELPER: test-data generator for the mean tests.
  # Builds a numpy array of the requested shape holding the consecutive numbers one, two, three and so
  # on in row-major order (a cumulative sum of ones, reshaped). Input: shape, a tuple of dimension
  # sizes. Output: a floating-point array with that shape.
  return np.reshape(np.cumsum(np.ones(shape)), shape)


class MeanTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the streaming mean metric. Covers variable names, the metrics and updates collections,
  # reading the running value, behaviour with many broadcastable weight shapes (including weights of
  # rank higher than the values), numerically weighted results for three-dimensional values, and
  # errors raised for weights that cannot be broadcast. Fixture: every test resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Takes no arguments and returns nothing.
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the streaming mean metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the count accumulator and the total accumulator under the mean scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.mean(array_ops.ones([4, 3]))
    _assert_metric_variables(self, ('mean/count:0', 'mean/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the streaming mean metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean(
        array_ops.ones([4, 3]), metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the streaming mean metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.mean(
        array_ops.ones([4, 3]), updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: four batches of two values are pushed through a first-in-first-out queue and averaged.
    # Setup: a cached session, a queue of capacity four with float elements of shape one by two, four
    # enqueued rows, and the dequeued tensor passed to the mean metric. Action: run the update
    # operation four times after initializing local variables. Expectation: the final mean equals 1.65
    # to five places.
    with self.cached_session() as sess:
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.mean(values)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAlmostEqual(1.65, self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testUpdateOpsReturnsCurrentValue(self):
    # [FUNCTION DOC]
    # Scenario: the update operation of the mean metric returns the running mean after each batch.
    # Setup: the same four-row queue as the basic test. Action: run the update operation four times and
    # then read the value. Expectation: successive update results are 0.5, 1.475, about 2.067 and 1.65,
    # and the final value tensor is 1.65, each to five places.
    with self.cached_session() as sess:
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.mean(values)

      self.evaluate(variables.local_variables_initializer())

      self.assertAlmostEqual(0.5, self.evaluate(update_op), 5)
      self.assertAlmostEqual(1.475, self.evaluate(update_op), 5)
      self.assertAlmostEqual(12.4 / 6.0, self.evaluate(update_op), 5)
      self.assertAlmostEqual(1.65, self.evaluate(update_op), 5)

      self.assertAlmostEqual(1.65, self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    # [FUNCTION DOC]
    # Scenario: weights consisting only of ones, in many different broadcastable shapes, must not change
    # the plain mean. Setup: values of shape three by two by four by one holding one through twenty four,
    # and twenty mean metrics: one without weights, one with a scalar weight of one, and eighteen with all-ones
    # weights whose shapes are lower rank, equal rank or higher rank and broadcast against the values.
    # Action: in a cached session, initialize local variables, run each update operation and read each
    # value. Expectation: every update result and every value equals the plain mean of the values.
    values = _test_values((3, 2, 4, 1))
    mean_results = (
        metrics.mean(values),
        metrics.mean(values, weights=1.0),
        metrics.mean(values, weights=np.ones((1, 1, 1))),
        metrics.mean(values, weights=np.ones((1, 1, 1, 1))),
        metrics.mean(values, weights=np.ones((1, 1, 1, 1, 1))),
        metrics.mean(values, weights=np.ones((1, 1, 4))),
        metrics.mean(values, weights=np.ones((1, 1, 4, 1))),
        metrics.mean(values, weights=np.ones((1, 2, 1))),
        metrics.mean(values, weights=np.ones((1, 2, 1, 1))),
        metrics.mean(values, weights=np.ones((1, 2, 4))),
        metrics.mean(values, weights=np.ones((1, 2, 4, 1))),
        metrics.mean(values, weights=np.ones((3, 1, 1))),
        metrics.mean(values, weights=np.ones((3, 1, 1, 1))),
        metrics.mean(values, weights=np.ones((3, 1, 4))),
        metrics.mean(values, weights=np.ones((3, 1, 4, 1))),
        metrics.mean(values, weights=np.ones((3, 2, 1))),
        metrics.mean(values, weights=np.ones((3, 2, 1, 1))),
        metrics.mean(values, weights=np.ones((3, 2, 4))),
        metrics.mean(values, weights=np.ones((3, 2, 4, 1))),
        metrics.mean(values, weights=np.ones((3, 2, 4, 1, 1))),)
    expected = np.mean(values)
    with self.cached_session():
      variables.local_variables_initializer().run()
      for mean_result in mean_results:
        mean, update_op = mean_result
        self.assertAlmostEqual(expected, self.evaluate(update_op))
        self.assertAlmostEqual(expected, self.evaluate(mean))

  def _test_3d_weighted(self, values, weights):
    # [FUNCTION DOC]
    # HELPER: shared body of the weighted three-dimensional mean tests in this class.
    # Computes the expected weighted mean in numpy as the sum of weights times values divided by the sum of
    # weights broadcast to the values, builds the mean metric with those weights, and in a cached
    # session runs the update and checks both the update result and the value tensor against the
    # expectation to five places. Inputs: values, a three-dimensional numpy array; weights, an array
    # broadcastable to it. Returns nothing; fails an assertion on a mismatch. Not collected itself
    # because its name does not start with test.
    expected = (
        np.sum(np.multiply(weights, values)) /
        np.sum(np.multiply(weights, np.ones_like(values)))
    )
    mean, update_op = metrics.mean(values, weights=weights)
    with self.cached_session():
      variables.local_variables_initializer().run()
      self.assertAlmostEqual(expected, self.evaluate(update_op), places=5)
      self.assertAlmostEqual(expected, self.evaluate(mean), places=5)

  @test_util.run_deprecated_v1
  def test1x1x1Weighted(self):
    # [FUNCTION DOC]
    # Scenario: a single scalar-like weight of shape one by one by one applied to three by two by four values.
    # Delegates to the shared weighted-mean check with the weight five. Returns nothing.
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((5,)).reshape((1, 1, 1)))

  @test_util.run_deprecated_v1
  def test1x1xNWeighted(self):
    # [FUNCTION DOC]
    # Scenario: weights varying only along the last axis, shape one by one by four, applied to three by
    # two by four values. Delegates to the shared weighted-mean check with four distinct weights.
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((5, 7, 11, 3)).reshape((1, 1, 4)))

  @test_util.run_deprecated_v1
  def test1xNx1Weighted(self):
    # [FUNCTION DOC]
    # Scenario: weights varying only along the middle axis, shape one by two by one. Delegates to the shared
    # weighted-mean check with two distinct weights.
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((5, 11)).reshape((1, 2, 1)))

  @test_util.run_deprecated_v1
  def test1xNxNWeighted(self):
    # [FUNCTION DOC]
    # Scenario: weights varying along the middle and last axes, shape one by two by four. Delegates to the
    # shared weighted-mean check with eight distinct weights.
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((5, 7, 11, 3, 2, 13, 7, 5)).reshape((1, 2, 4)))

  @test_util.run_deprecated_v1
  def testNx1x1Weighted(self):
    # [FUNCTION DOC]
    # Scenario: weights varying only along the first axis, shape three by one by one. Delegates to the
    # shared weighted-mean check with three distinct weights.
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((5, 7, 11)).reshape((3, 1, 1)))

  @test_util.run_deprecated_v1
  def testNx1xNWeighted(self):
    # [FUNCTION DOC]
    # Scenario: weights varying along the first and last axes, shape three by one by four. Delegates to the
    # shared weighted-mean check with twelve distinct weights.
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3)).reshape((3, 1, 4)))

  @test_util.run_deprecated_v1
  def testNxNxNWeighted(self):
    # [FUNCTION DOC]
    # Scenario: a full-shape weight tensor, three by two by four, so every value has its own weight.
    # Delegates to the shared weighted-mean check with twenty four distinct weights.
    self._test_3d_weighted(
        _test_values((3, 2, 4)),
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidWeights(self):
    # [FUNCTION DOC]
    # Scenario: weights whose shape cannot be broadcast to the values must be rejected, both when shapes
    # are known statically and when they are known only at run time.
    # Setup: values of shape three by two by four by one, a float placeholder, and nine invalid weight
    # shapes (wrong rank or a mismatching dimension). Action and expectation: for each, building the
    # metric with the constant values raises a value error whose message says the weights cannot be
    # broadcast to the values; building it with the placeholder, initializing and running the update with
    # the values fed raises a lower-level op error with the same message. Returns nothing.
    values_placeholder = array_ops.placeholder(dtype=dtypes_lib.float32)
    values = _test_values((3, 2, 4, 1))
    invalid_weights = (
        (1,),
        (1, 1),
        (3, 2),
        (2, 4, 1),
        (4, 2, 4, 1),
        (3, 3, 4, 1),
        (3, 2, 5, 1),
        (3, 2, 4, 2),
        (1, 1, 1, 1, 1))
    expected_error_msg = 'weights can not be broadcast to values'
    for invalid_weight in invalid_weights:
      # Static shapes.
      with self.assertRaisesRegex(ValueError, expected_error_msg):
        metrics.mean(values, invalid_weight)

      # Dynamic shapes.
      with self.assertRaisesRegex(errors_impl.OpError, expected_error_msg):
        with self.cached_session():
          _, update_op = metrics.mean(values_placeholder, invalid_weight)
          variables.local_variables_initializer().run()
          update_op.eval(feed_dict={values_placeholder: values})


class MeanTensorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the element-wise mean tensor metric (a mean kept separately for every position of the input).
  # Covers variables, collections, basic and multi-dimensional running means, the value returned by
  # the update operation, and weighting by one-dimensional and two-dimensional weights fed from
  # queues. Fixture: every test resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Takes no arguments and returns nothing.
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the element-wise mean tensor metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the total-tensor accumulator and the count-tensor accumulator under the mean scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.mean_tensor(array_ops.ones([4, 3]))
    _assert_metric_variables(self,
                             ('mean/total_tensor:0', 'mean/count_tensor:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the element-wise mean tensor metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean_tensor(
        array_ops.ones([4, 3]), metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the element-wise mean tensor metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_tensor(
        array_ops.ones([4, 3]), updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: four rows of two values are averaged position by position. Setup: a float queue of
    # shape one by two with four rows. Action: run the update four times, then read the value.
    # Expectation: the mean tensor equals one by two result of the column means, about -0.225 and 3.525.
    with self.cached_session() as sess:
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAllClose([[-0.9 / 4., 3.525]], self.evaluate(mean))

  @test_util.run_deprecated_v1
  def testMultiDimensional(self):
    # [FUNCTION DOC]
    # Scenario: the mean tensor metric works on values of rank three. Setup: a queue with two elements,
    # each of shape two by two by two. Action: run the update twice then read. Expectation: the result is
    # the element-wise average of the two enqueued tensors.
    with self.cached_session() as sess:
      values_queue = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(2, 2, 2))
      _enqueue_vector(
          sess,
          values_queue, [[[1, 2], [1, 2]], [[1, 2], [1, 2]]],
          shape=(2, 2, 2))
      _enqueue_vector(
          sess,
          values_queue, [[[1, 2], [1, 2]], [[3, 4], [9, 10]]],
          shape=(2, 2, 2))
      values = values_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(2):
        self.evaluate(update_op)
      self.assertAllClose([[[1, 2], [1, 2]], [[2, 3], [5, 6]]],
                          self.evaluate(mean))

  @test_util.run_deprecated_v1
  def testUpdateOpsReturnsCurrentValue(self):
    # [FUNCTION DOC]
    # Scenario: each update returns the running element-wise mean so far. Setup: the same four-row
    # queue as the basic test. Action: run the update four times, then read. Expectation: the successive
    # results are the running means after one to four rows, and the final value matches the last one.
    with self.cached_session() as sess:
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values)

      self.evaluate(variables.local_variables_initializer())

      self.assertAllClose([[0, 1]], self.evaluate(update_op), 5)
      self.assertAllClose([[-2.1, 5.05]], self.evaluate(update_op), 5)
      self.assertAllClose([[2.3 / 3., 10.1 / 3.]], self.evaluate(update_op), 5)
      self.assertAllClose([[-0.9 / 4., 3.525]], self.evaluate(update_op), 5)

      self.assertAllClose([[-0.9 / 4., 3.525]], self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testBinaryWeighted1d(self):
    # [FUNCTION DOC]
    # Scenario: per-example weights of zero or one (a one-by-one weight tensor per step). Setup: values
    # queue plus a weights queue with one, zero, one, zero. Action: four updates, then read.
    # Expectation: only the first and third rows count, giving 3.25 and 0.5.
    with self.cached_session() as sess:
      # Create the queue that populates the values.
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [[1]])
      _enqueue_vector(sess, weights_queue, [[0]])
      _enqueue_vector(sess, weights_queue, [[1]])
      _enqueue_vector(sess, weights_queue, [[0]])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values, weights)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAllClose([[3.25, 0.5]], self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testWeighted1d(self):
    # [FUNCTION DOC]
    # Scenario: small fractional per-example weights. Setup: values queue plus a weights queue of shape
    # one by one holding 0.0025, 0.005, 0.01 and 0.0075. Action: four updates, then read. Expectation:
    # the weighted per-position means are 0.8 and 3.52.
    with self.cached_session() as sess:
      # Create the queue that populates the values.
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [[0.0025]])
      _enqueue_vector(sess, weights_queue, [[0.005]])
      _enqueue_vector(sess, weights_queue, [[0.01]])
      _enqueue_vector(sess, weights_queue, [[0.0075]])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values, weights)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAllClose([[0.8, 3.52]], self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testWeighted2d_1(self):
    # [FUNCTION DOC]
    # Scenario: per-position weights of shape one by two. Setup: weights rows are (1,1), (1,0), (0,1),
    # (0,0). Action: four updates, then read. Expectation: the result is -2.1 and 0.5.
    with self.cached_session() as sess:
      # Create the queue that populates the values.
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, weights_queue, [1, 1])
      _enqueue_vector(sess, weights_queue, [1, 0])
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values, weights)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAllClose([[-2.1, 0.5]], self.evaluate(mean), 5)

  @test_util.run_deprecated_v1
  def testWeighted2d_2(self):
    # [FUNCTION DOC]
    # Scenario: per-position weights of shape one by two where the first position is never counted.
    # Setup: weights rows are (0,1), (0,0), (0,1), (0,0). Action: four updates, then read. Expectation:
    # the result is 0 for the first position and 0.5 for the second.
    with self.cached_session() as sess:
      # Create the queue that populates the values.
      values_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 2))
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.mean_tensor(values, weights)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(4):
        self.evaluate(update_op)
      self.assertAllClose([[0, 0.5]], self.evaluate(mean), 5)


class AccuracyTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the accuracy metric (fraction of matching labels and predictions). Covers variables,
  # collections, shape-mismatch errors, idempotent reads, running accuracy over queued batches,
  # and the special handling of weights and labels whose shapes differ only by a trailing singleton
  # dimension. Fixture: every test resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Takes no arguments and returns nothing.
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the accuracy metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the count and total accumulators under the custom metric name given to the metric.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.accuracy(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        name='my_accuracy')
    _assert_metric_variables(self,
                             ('my_accuracy/count:0', 'my_accuracy/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the accuracy metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.accuracy(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the accuracy metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.accuracy(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testPredictionsAndLabelsOfDifferentSizeRaisesValueError(self):
    # [FUNCTION DOC]
    # Scenario: predictions of shape ten by three and labels of shape ten by four. Action: build the accuracy
    # metric. Expectation: a value error is raised because the shapes are not compatible.
    predictions = array_ops.ones((10, 3))
    labels = array_ops.ones((10, 4))
    with self.assertRaises(ValueError):
      metrics.accuracy(labels, predictions)

  @test_util.run_deprecated_v1
  def testPredictionsAndWeightsOfDifferentSizeRaisesValueError(self):
    # [FUNCTION DOC]
    # Scenario: predictions and labels of shape ten by three but weights of shape nine by three. Action:
    # build the accuracy metric with those weights. Expectation: a value error is raised.
    predictions = array_ops.ones((10, 3))
    labels = array_ops.ones((10, 3))
    weights = array_ops.ones((9, 3))
    with self.assertRaises(ValueError):
      metrics.accuracy(labels, predictions, weights)

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the accuracy metric repeatedly does not change it.
    # Inputs: integer predictions and labels drawn from the same random generator seed (three classes, shape ten by three), so they are identical.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_uniform(
        (10, 3), maxval=3, dtype=dtypes_lib.int64, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=3, dtype=dtypes_lib.int64, seed=1)
    accuracy, update_op = metrics.accuracy(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_accuracy = self.evaluate(accuracy)
      for _ in range(10):
        self.assertEqual(initial_accuracy, self.evaluate(accuracy))

  @test_util.run_deprecated_v1
  def testMultipleUpdates(self):
    # [FUNCTION DOC]
    # Scenario: four batches of one prediction and one label each, accumulated. Setup: predictions
    # 0, 1, 2, 1 and labels 0, 1, 1, 2 from queues. Action: three updates, then a fourth. Expectation: the
    # fourth update result and the value are both 0.5 (two of four match), compared exactly.
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      labels = labels_queue.dequeue()

      accuracy, update_op = metrics.accuracy(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(3):
        self.evaluate(update_op)
      self.assertEqual(0.5, self.evaluate(update_op))
      self.assertEqual(0.5, self.evaluate(accuracy))

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizes(self):
    # [FUNCTION DOC]
    # Scenario: predictions of shape forty by one and labels of shape forty, which are equivalent apart from
    # a trailing singleton dimension. Setup: both all ones. Action: update and read. Expectation: the
    # accuracy is exactly one, showing the metric squeezes the extra dimension.
    predictions = array_ops.ones((40, 1))
    labels = array_ops.ones((40,))
    with self.cached_session():
      accuracy, update_op = metrics.accuracy(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1.0, self.evaluate(update_op))
      self.assertEqual(1.0, self.evaluate(accuracy))

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizesWithScalarWeight(self):
    # [FUNCTION DOC]
    # Scenario: same equivalent-size inputs as above with a scalar weight of two. Expectation: accuracy
    # stays exactly one for both the update result and the value.
    predictions = array_ops.ones((40, 1))
    labels = array_ops.ones((40,))
    with self.cached_session():
      accuracy, update_op = metrics.accuracy(labels, predictions, weights=2.0)

      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1.0, self.evaluate(update_op))
      self.assertEqual(1.0, self.evaluate(accuracy))

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizesWithStaticShapedWeight(self):
    # [FUNCTION DOC]
    # Scenario: predictions of shape three, labels and weights of shape three by one, where weights are
    # 100, 1, 1 and only the first label matches. Action: update and read. Expectation: accuracy is above
    # 0.95, which proves the weight is flattened to match rather than broadcast (broadcasting would give
    # about one third).
    predictions = ops.convert_to_tensor([1, 1, 1])  # shape 3,
    labels = array_ops.expand_dims(ops.convert_to_tensor([1, 0, 0]),
                                   1)  # shape 3, 1
    weights = array_ops.expand_dims(ops.convert_to_tensor([100, 1, 1]),
                                    1)  # shape 3, 1

    with self.cached_session():
      accuracy, update_op = metrics.accuracy(labels, predictions, weights)

      self.evaluate(variables.local_variables_initializer())
      # if streaming_accuracy does not flatten the weight, accuracy would be
      # 0.33333334 due to an intended broadcast of weight. Due to flattening,
      # it will be higher than .95
      self.assertGreater(self.evaluate(update_op), .95)
      self.assertGreater(self.evaluate(accuracy), .95)

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizesWithDynamicallyShapedWeight(self):
    # [FUNCTION DOC]
    # Scenario: same data as the static-shape variant but the weight comes from an integer placeholder
    # whose shape is unknown until run time, fed through a feed mapping. Expectation: update result and
    # value are both above 0.95, confirming flattening also happens with dynamic shapes.
    predictions = ops.convert_to_tensor([1, 1, 1])  # shape 3,
    labels = array_ops.expand_dims(ops.convert_to_tensor([1, 0, 0]),
                                   1)  # shape 3, 1

    weights = [[100], [1], [1]]  # shape 3, 1
    weights_placeholder = array_ops.placeholder(
        dtype=dtypes_lib.int32, name='weights')
    feed_dict = {weights_placeholder: weights}

    with self.cached_session():
      accuracy, update_op = metrics.accuracy(labels, predictions,
                                             weights_placeholder)

      self.evaluate(variables.local_variables_initializer())
      # if streaming_accuracy does not flatten the weight, accuracy would be
      # 0.33333334 due to an intended broadcast of weight. Due to flattening,
      # it will be higher than .95
      self.assertGreater(update_op.eval(feed_dict=feed_dict), .95)
      self.assertGreater(accuracy.eval(feed_dict=feed_dict), .95)

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithWeightedValues(self):
    # [FUNCTION DOC]
    # Scenario: four batches with integer weights one, one, zero, zero, so only the first two
    # examples count. Setup: predictions 0, 1, 2, 1, labels 0, 1, 1, 2, weights from a queue of integers.
    # Action: three updates, then a fourth. Expectation: accuracy is exactly one for both the fourth
    # update and the value.
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      labels = labels_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.int64, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [0])
      _enqueue_vector(sess, weights_queue, [0])
      weights = weights_queue.dequeue()

      accuracy, update_op = metrics.accuracy(labels, predictions, weights)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(3):
        self.evaluate(update_op)
      self.assertEqual(1.0, self.evaluate(update_op))
      self.assertEqual(1.0, self.evaluate(accuracy))


class PrecisionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the precision metric (true positives over predicted positives). Covers variables,
  # collections, idempotent reads, perfect and fully wrong predictions, mixed input dtypes, one- and
  # two-dimensional weights (also with placeholders), and the zero-denominator case.
  # Fixture: every test seeds numpy and resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the precision metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the false-positive count and the true-positive count under the precision scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.precision(
        predictions=array_ops.ones((10, 1)), labels=array_ops.ones((10, 1)))
    _assert_metric_variables(self, ('precision/false_positives/count:0',
                                    'precision/true_positives/count:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the precision metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.precision(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the precision metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.precision(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the precision metric repeatedly does not change it.
    # Inputs: integer predictions and labels with an upper bound of one (so effectively all zeros), generated with identical seeds.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    precision, update_op = metrics.precision(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_precision = self.evaluate(precision)
      for _ in range(10):
        self.assertEqual(initial_precision, self.evaluate(precision))

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    # [FUNCTION DOC]
    # Scenario: random binary inputs of shape one hundred by one used as both predictions and labels.
    # Expectation: precision is one for both the update result and the value, to six places.
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(inputs)
    labels = constant_op.constant(inputs)
    precision, update_op = metrics.precision(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1.0, self.evaluate(update_op), 6)
      self.assertAlmostEqual(1.0, self.evaluate(precision), 6)

  @test_util.run_deprecated_v1
  def testSomeCorrect_multipleInputDtypes(self):
    # [FUNCTION DOC]
    # Scenario: predictions 1,0,1,0 and labels 0,1,1,0, cast to boolean, int32 and float32 in turn.
    # Action: for each dtype build the metric, update and read. Expectation: precision is 0.5 for every
    # dtype for both update result and value.
    for dtype in (dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
      predictions = math_ops.cast(
          constant_op.constant([1, 0, 1, 0], shape=(1, 4)), dtype=dtype)
      labels = math_ops.cast(
          constant_op.constant([0, 1, 1, 0], shape=(1, 4)), dtype=dtype)
      precision, update_op = metrics.precision(labels, predictions)

      with self.cached_session():
        self.evaluate(variables.local_variables_initializer())
        self.assertAlmostEqual(0.5, self.evaluate(update_op))
        self.assertAlmostEqual(0.5, self.evaluate(precision))

  @test_util.run_deprecated_v1
  def testWeighted1d(self):
    # [FUNCTION DOC]
    # Scenario: per-row weights of shape two by one (weights 2 and 5) over two rows of four predictions.
    # Expectation: precision equals the weighted true positives over weighted predicted positives, that
    # is seven over fourteen, for update result and value.
    predictions = constant_op.constant([[1, 0, 1, 0], [1, 0, 1, 0]])
    labels = constant_op.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    precision, update_op = metrics.precision(
        labels, predictions, weights=constant_op.constant([[2], [5]]))

    with self.cached_session():
      variables.local_variables_initializer().run()
      weighted_tp = 2.0 + 5.0
      weighted_positives = (2.0 + 2.0) + (5.0 + 5.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(expected_precision, self.evaluate(update_op))
      self.assertAlmostEqual(expected_precision, self.evaluate(precision))

  @test_util.run_deprecated_v1
  def testWeightedScalar_placeholders(self):
    # [FUNCTION DOC]
    # Scenario: float placeholders for predictions and labels, scalar weight two, data fed at run time.
    # Expectation: precision is the weighted true-positive count over the weighted predicted-positive
    # count (four over eight) for both update result and value.
    predictions = array_ops.placeholder(dtype=dtypes_lib.float32)
    labels = array_ops.placeholder(dtype=dtypes_lib.float32)
    feed_dict = {
        predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
        labels: ((0, 1, 1, 0), (1, 0, 0, 1))
    }
    precision, update_op = metrics.precision(labels, predictions, weights=2)

    with self.cached_session():
      variables.local_variables_initializer().run()
      weighted_tp = 2.0 + 2.0
      weighted_positives = (2.0 + 2.0) + (2.0 + 2.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(
          expected_precision, update_op.eval(feed_dict=feed_dict))
      self.assertAlmostEqual(
          expected_precision, precision.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testWeighted1d_placeholders(self):
    # [FUNCTION DOC]
    # Scenario: same as the one-dimensional weighted test but predictions and labels are placeholders fed
    # at run time. Expectation: precision is seven over fourteen for update result and value.
    predictions = array_ops.placeholder(dtype=dtypes_lib.float32)
    labels = array_ops.placeholder(dtype=dtypes_lib.float32)
    feed_dict = {
        predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
        labels: ((0, 1, 1, 0), (1, 0, 0, 1))
    }
    precision, update_op = metrics.precision(
        labels, predictions, weights=constant_op.constant([[2], [5]]))

    with self.cached_session():
      variables.local_variables_initializer().run()
      weighted_tp = 2.0 + 5.0
      weighted_positives = (2.0 + 2.0) + (5.0 + 5.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(
          expected_precision, update_op.eval(feed_dict=feed_dict))
      self.assertAlmostEqual(
          expected_precision, precision.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testWeighted2d(self):
    # [FUNCTION DOC]
    # Scenario: element-wise weights of shape two by four over two rows of predictions and labels.
    # Expectation: precision equals the weighted true positives (seven) over weighted predicted positives
    # (ten), for update result and value.
    predictions = constant_op.constant([[1, 0, 1, 0], [1, 0, 1, 0]])
    labels = constant_op.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    precision, update_op = metrics.precision(
        labels,
        predictions,
        weights=constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))

    with self.cached_session():
      variables.local_variables_initializer().run()
      weighted_tp = 3.0 + 4.0
      weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(expected_precision, self.evaluate(update_op))
      self.assertAlmostEqual(expected_precision, self.evaluate(precision))

  @test_util.run_deprecated_v1
  def testWeighted2d_placeholders(self):
    # [FUNCTION DOC]
    # Scenario: element-wise two-dimensional weights with placeholder predictions and labels fed at run
    # time. Expectation: same expected precision as the constant-input two-dimensional test.
    predictions = array_ops.placeholder(dtype=dtypes_lib.float32)
    labels = array_ops.placeholder(dtype=dtypes_lib.float32)
    feed_dict = {
        predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
        labels: ((0, 1, 1, 0), (1, 0, 0, 1))
    }
    precision, update_op = metrics.precision(
        labels,
        predictions,
        weights=constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))

    with self.cached_session():
      variables.local_variables_initializer().run()
      weighted_tp = 3.0 + 4.0
      weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(
          expected_precision, update_op.eval(feed_dict=feed_dict))
      self.assertAlmostEqual(
          expected_precision, precision.eval(feed_dict=feed_dict))

  @test_util.run_deprecated_v1
  def testAllIncorrect(self):
    # [FUNCTION DOC]
    # Scenario: labels are the complement of the predictions, random binary data of shape one hundred
    # by one. Action: update, then read. Expectation: precision is zero.
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(inputs)
    labels = constant_op.constant(1 - inputs)
    precision, update_op = metrics.precision(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertAlmostEqual(0, self.evaluate(precision))

  @test_util.run_deprecated_v1
  def testZeroTrueAndFalsePositivesGivesZeroPrecision(self):
    # [FUNCTION DOC]
    # Scenario: predictions and labels all zeros, so no positive is predicted. Expectation: the metric
    # defines precision as exactly zero rather than not-a-number when there are no positives predicted.
    predictions = constant_op.constant([0, 0, 0, 0])
    labels = constant_op.constant([0, 0, 0, 0])
    precision, update_op = metrics.precision(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertEqual(0.0, self.evaluate(precision))


class RecallTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the recall metric (true positives over actual positives). Covers variables, collections,
  # idempotent reads, perfect and fully wrong predictions, mixed input dtypes, weighted cases and the
  # zero-denominator case. Fixture: every test seeds numpy and resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the recall metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the false-negative count and the true-positive count under the recall scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.recall(
        predictions=array_ops.ones((10, 1)), labels=array_ops.ones((10, 1)))
    _assert_metric_variables(
        self,
        ('recall/false_negatives/count:0', 'recall/true_positives/count:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the recall metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.recall(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the recall metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.recall(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the recall metric repeatedly does not change it.
    # Inputs: integer predictions and labels with an upper bound of one (so effectively all zeros), generated with identical seeds.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    recall, update_op = metrics.recall(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_recall = self.evaluate(recall)
      for _ in range(10):
        self.assertEqual(initial_recall, self.evaluate(recall))

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    # [FUNCTION DOC]
    # Scenario: random binary inputs used as both predictions and labels. Expectation: recall is one to
    # six places after the update.
    np_inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(np_inputs)
    labels = constant_op.constant(np_inputs)
    recall, update_op = metrics.recall(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertAlmostEqual(1.0, self.evaluate(recall), 6)

  @test_util.run_deprecated_v1
  def testSomeCorrect_multipleInputDtypes(self):
    # [FUNCTION DOC]
    # Scenario: predictions 1,0,1,0 and labels 0,1,1,0 cast to boolean, int32 and float32 in turn.
    # Expectation: recall is 0.5 for each dtype for update result and value.
    for dtype in (dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
      predictions = math_ops.cast(
          constant_op.constant([1, 0, 1, 0], shape=(1, 4)), dtype=dtype)
      labels = math_ops.cast(
          constant_op.constant([0, 1, 1, 0], shape=(1, 4)), dtype=dtype)
      recall, update_op = metrics.recall(labels, predictions)

      with self.cached_session():
        self.evaluate(variables.local_variables_initializer())
        self.assertAlmostEqual(0.5, self.evaluate(update_op))
        self.assertAlmostEqual(0.5, self.evaluate(recall))

  @test_util.run_deprecated_v1
  def testWeighted1d(self):
    # [FUNCTION DOC]
    # Scenario: two rows with per-row weights 2 and 5. Expectation: recall equals weighted true
    # positives (seven) over weighted actual positives (fourteen).
    predictions = constant_op.constant([[1, 0, 1, 0], [0, 1, 0, 1]])
    labels = constant_op.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    weights = constant_op.constant([[2], [5]])
    recall, update_op = metrics.recall(labels, predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      weighted_tp = 2.0 + 5.0
      weighted_t = (2.0 + 2.0) + (5.0 + 5.0)
      expected_precision = weighted_tp / weighted_t
      self.assertAlmostEqual(expected_precision, self.evaluate(update_op))
      self.assertAlmostEqual(expected_precision, self.evaluate(recall))

  @test_util.run_deprecated_v1
  def testWeighted2d(self):
    # [FUNCTION DOC]
    # Scenario: element-wise weights of shape two by four. Expectation: recall equals weighted true
    # positives (four) over weighted actual positives (ten).
    predictions = constant_op.constant([[1, 0, 1, 0], [0, 1, 0, 1]])
    labels = constant_op.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    weights = constant_op.constant([[1, 2, 3, 4], [4, 3, 2, 1]])
    recall, update_op = metrics.recall(labels, predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      weighted_tp = 3.0 + 1.0
      weighted_t = (2.0 + 3.0) + (4.0 + 1.0)
      expected_precision = weighted_tp / weighted_t
      self.assertAlmostEqual(expected_precision, self.evaluate(update_op))
      self.assertAlmostEqual(expected_precision, self.evaluate(recall))

  @test_util.run_deprecated_v1
  def testAllIncorrect(self):
    # [FUNCTION DOC]
    # Scenario: labels are the complement of predictions. Expectation: recall is exactly zero.
    np_inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(np_inputs)
    labels = constant_op.constant(1 - np_inputs)
    recall, update_op = metrics.recall(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertEqual(0, self.evaluate(recall))

  @test_util.run_deprecated_v1
  def testZeroTruePositivesAndFalseNegativesGivesZeroRecall(self):
    # [FUNCTION DOC]
    # Scenario: predictions and labels all zeros, so there are no actual positives. Expectation: recall is
    # defined as exactly zero rather than not-a-number.
    predictions = array_ops.zeros((1, 4))
    labels = array_ops.zeros((1, 4))
    recall, update_op = metrics.recall(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertEqual(0, self.evaluate(recall))


class AUCTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the area-under-curve metric computed from thresholded confusion counters. Covers
  # variables, collections, idempotent reads, ROC and PR curves, summation methods (trapezoidal and
  # careful interpolation), weights, manually supplied thresholds, degenerate inputs, and a streaming
  # comparison against a numpy reference implementation. Fixture: every test seeds numpy and resets the
  # default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the area-under-curve (AUC) metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the four confusion counters true positives, false negatives, false positives and true negatives under the auc scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.auc(predictions=array_ops.ones((10, 1)),
                labels=array_ops.ones((10, 1)))
    _assert_metric_variables(self,
                             ('auc/true_positives:0', 'auc/false_negatives:0',
                              'auc/false_positives:0', 'auc/true_negatives:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the area-under-curve (AUC) metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.auc(predictions=array_ops.ones((10, 1)),
                          labels=array_ops.ones((10, 1)),
                          metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the area-under-curve (AUC) metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.auc(predictions=array_ops.ones((10, 1)),
                               labels=array_ops.ones((10, 1)),
                               updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the area-under-curve (AUC) metric repeatedly does not change it.
    # Inputs: random floating-point predictions and random integer labels of shape ten by three, both with a fixed seed.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.float32, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    auc, update_op = metrics.auc(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_auc = self.evaluate(auc)
      for _ in range(10):
        self.assertAlmostEqual(initial_auc, self.evaluate(auc), 5)

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    # [FUNCTION DOC]
    # Scenario: perfect predictions evaluated on the ROC curve. Delegates to the shared helper that
    # builds random binary labels and matching float predictions and expects an AUC of exactly one.
    self.allCorrectAsExpected('ROC')

  def allCorrectAsExpected(self, curve):
    # [FUNCTION DOC]
    # HELPER: shared body for the perfect-prediction test, parameterized by curve type.
    # KNOWN ISSUE: only the ROC curve is ever passed in (by the all-correct test), so the curve
    # parameter is never exercised with the precision-recall curve even though it is parameterized.
    # Inputs: curve, the curve name accepted by the AUC metric. Builds random binary data of shape one
    # hundred by one used as labels and as float predictions, builds the AUC metric for that curve, runs
    # the update and checks that both the update result and the value are exactly one. Not collected as
    # a test itself because its name does not start with test.
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.cached_session():
      predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
      labels = constant_op.constant(inputs)
      auc, update_op = metrics.auc(labels, predictions, curve=curve)

      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1, self.evaluate(update_op))

      self.assertEqual(1, self.evaluate(auc))

  @test_util.run_deprecated_v1
  def testSomeCorrect_multipleLabelDtypes(self):
    # [FUNCTION DOC]
    # Scenario: predictions 1,0,1,0 with labels 0,1,1,0 cast to boolean, int32 and float32. Expectation:
    # the AUC is 0.5 for update result and value for each dtype.
    with self.cached_session():
      for label_dtype in (
          dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
        predictions = constant_op.constant(
            [1, 0, 1, 0], shape=(1, 4), dtype=dtypes_lib.float32)
        labels = math_ops.cast(
            constant_op.constant([0, 1, 1, 0], shape=(1, 4)), dtype=label_dtype)
        auc, update_op = metrics.auc(labels, predictions)

        self.evaluate(variables.local_variables_initializer())
        self.assertAlmostEqual(0.5, self.evaluate(update_op))

        self.assertAlmostEqual(0.5, self.evaluate(auc))

  @test_util.run_deprecated_v1
  def testWeighted1d(self):
    # [FUNCTION DOC]
    # Scenario: one scalar-like weight of two (shape one by one) over four examples. Expectation: AUC is
    # 0.5 to five places, showing a uniform weight leaves the result unchanged.
    with self.cached_session():
      predictions = constant_op.constant(
          [1, 0, 1, 0], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 1, 1, 0], shape=(1, 4))
      weights = constant_op.constant([2], shape=(1, 1))
      auc, update_op = metrics.auc(labels, predictions, weights=weights)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.5, self.evaluate(update_op), 5)

      self.assertAlmostEqual(0.5, self.evaluate(auc), 5)

  @test_util.run_deprecated_v1
  def testWeighted2d(self):
    # [FUNCTION DOC]
    # Scenario: per-example weights 1,2,3,4 on the same four examples. Expectation: AUC is 0.7 to five
    # places for update and value.
    with self.cached_session():
      predictions = constant_op.constant(
          [1, 0, 1, 0], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 1, 1, 0], shape=(1, 4))
      weights = constant_op.constant([1, 2, 3, 4], shape=(1, 4))
      auc, update_op = metrics.auc(labels, predictions, weights=weights)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.7, self.evaluate(update_op), 5)

      self.assertAlmostEqual(0.7, self.evaluate(auc), 5)

  @test_util.run_deprecated_v1
  def testManualThresholds(self):
    # [FUNCTION DOC]
    # Scenario: default and manual thresholds give different AUCs. Setup: predictions 0.12, 0.3001, 0.3003,
    # 0.72, labels 0,1,0,1, unit weights, manual thresholds 0.0, 0.2, 0.3002, 0.6, 1.0 which split the
    # two middle predictions that the default thresholds do not. Expectation: default AUC is 0.875 and the
    # manual-threshold AUC is 0.75, each to three places, for update result and value.
    with self.cached_session():
      # Verifies that thresholds passed in to the `thresholds` parameter are
      # used correctly.
      # The default thresholds do not split the second and third predictions.
      # Thus, when we provide manual thresholds which correctly split it, we get
      # an accurate AUC value.
      predictions = constant_op.constant(
          [0.12, 0.3001, 0.3003, 0.72], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 1, 0, 1], shape=(1, 4))
      weights = constant_op.constant([1, 1, 1, 1], shape=(1, 4))
      thresholds = [0.0, 0.2, 0.3002, 0.6, 1.0]
      default_auc, default_update_op = metrics.auc(labels,
                                                   predictions,
                                                   weights=weights)
      manual_auc, manual_update_op = metrics.auc(labels,
                                                 predictions,
                                                 weights=weights,
                                                 thresholds=thresholds)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.875, self.evaluate(default_update_op), 3)
      self.assertAlmostEqual(0.875, self.evaluate(default_auc), 3)

      self.assertAlmostEqual(0.75, self.evaluate(manual_update_op), 3)
      self.assertAlmostEqual(0.75, self.evaluate(manual_auc), 3)

  # Regarding the AUC-PR tests: note that the preferred method when
  # calculating AUC-PR is summation_method='careful_interpolation'.
  @test_util.run_deprecated_v1
  def testCorrectAUCPRSpecialCase(self):
    # [FUNCTION DOC]
    # Scenario: precision-recall AUC with careful interpolation on four predictions (0.1, 0.4, 0.35,
    # 0.8) and labels 0,0,1,1. Expectation: AUC is about 0.797, computed from an analytic expression
    # involving a natural logarithm, within 0.001.
    with self.cached_session():
      predictions = constant_op.constant(
          [0.1, 0.4, 0.35, 0.8], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 1, 1], shape=(1, 4))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='careful_interpolation')

      self.evaluate(variables.local_variables_initializer())
      # expected ~= 0.79726744594
      expected = 1 - math.log(1.5) / 2
      self.assertAlmostEqual(expected, self.evaluate(update_op), delta=1e-3)
      self.assertAlmostEqual(expected, self.evaluate(auc), delta=1e-3)

  @test_util.run_deprecated_v1
  def testCorrectAnotherAUCPRSpecialCase(self):
    # [FUNCTION DOC]
    # Scenario: precision-recall AUC with careful interpolation on seven examples. Expectation: AUC
    # is about 0.6135 within 0.001, from an analytic expression involving logarithms.
    with self.cached_session():
      predictions = constant_op.constant(
          [0.1, 0.4, 0.35, 0.8, 0.1, 0.135, 0.81],
          shape=(1, 7),
          dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 1, 0, 1, 0, 1], shape=(1, 7))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='careful_interpolation')

      self.evaluate(variables.local_variables_initializer())
      # expected ~= 0.61350593198
      expected = (2.5 - 2 * math.log(4./3) - 0.25 * math.log(7./5)) / 3
      self.assertAlmostEqual(expected, self.evaluate(update_op), delta=1e-3)
      self.assertAlmostEqual(expected, self.evaluate(auc), delta=1e-3)

  @test_util.run_deprecated_v1
  def testThirdCorrectAUCPRSpecialCase(self):
    # [FUNCTION DOC]
    # Scenario: precision-recall AUC with careful interpolation on seven sorted predictions.
    # Expectation: AUC is about 0.904 within 0.001, from an analytic expression with a logarithm.
    with self.cached_session():
      predictions = constant_op.constant(
          [0.0, 0.1, 0.2, 0.33, 0.3, 0.4, 0.5],
          shape=(1, 7),
          dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 0, 0, 1, 1, 1], shape=(1, 7))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='careful_interpolation')

      self.evaluate(variables.local_variables_initializer())
      # expected ~= 0.90410597584
      expected = 1 - math.log(4./3) / 3
      self.assertAlmostEqual(expected, self.evaluate(update_op), delta=1e-3)
      self.assertAlmostEqual(expected, self.evaluate(auc), delta=1e-3)

  @test_util.run_deprecated_v1
  def testIncorrectAUCPRSpecialCase(self):
    # [FUNCTION DOC]
    # Scenario: same data as the first special case but with trapezoidal summation, which is known to be
    # a poorer estimator for the PR curve. Expectation: the trapezoidal result 0.79166 within 0.001.
    with self.cached_session():
      predictions = constant_op.constant(
          [0.1, 0.4, 0.35, 0.8], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 1, 1], shape=(1, 4))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='trapezoidal')

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.79166, self.evaluate(update_op), delta=1e-3)

      self.assertAlmostEqual(0.79166, self.evaluate(auc), delta=1e-3)

  @test_util.run_deprecated_v1
  def testAnotherIncorrectAUCPRSpecialCase(self):
    # [FUNCTION DOC]
    # Scenario: the seven-example data with trapezoidal summation. Expectation: AUC 0.610317 within
    # 0.001 (differs from the careful-interpolation answer).
    with self.cached_session():
      predictions = constant_op.constant(
          [0.1, 0.4, 0.35, 0.8, 0.1, 0.135, 0.81],
          shape=(1, 7),
          dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 1, 0, 1, 0, 1], shape=(1, 7))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='trapezoidal')

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.610317, self.evaluate(update_op), delta=1e-3)

      self.assertAlmostEqual(0.610317, self.evaluate(auc), delta=1e-3)

  @test_util.run_deprecated_v1
  def testThirdIncorrectAUCPRSpecialCase(self):
    # [FUNCTION DOC]
    # Scenario: the sorted seven-example data with trapezoidal summation. Expectation: AUC 0.90277
    # within 0.001.
    with self.cached_session():
      predictions = constant_op.constant(
          [0.0, 0.1, 0.2, 0.33, 0.3, 0.4, 0.5],
          shape=(1, 7),
          dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 0, 0, 0, 1, 1, 1], shape=(1, 7))
      auc, update_op = metrics.auc(labels, predictions, curve='PR',
                                   summation_method='trapezoidal')

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.90277, self.evaluate(update_op), delta=1e-3)

      self.assertAlmostEqual(0.90277, self.evaluate(auc), delta=1e-3)

  @test_util.run_deprecated_v1
  def testAllIncorrect(self):
    # [FUNCTION DOC]
    # Scenario: labels are the complement of the float predictions on random binary data. Expectation:
    # the AUC is zero for update result and value.
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.cached_session():
      predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
      labels = constant_op.constant(1 - inputs, dtype=dtypes_lib.float32)
      auc, update_op = metrics.auc(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0, self.evaluate(update_op))

      self.assertAlmostEqual(0, self.evaluate(auc))

  @test_util.run_deprecated_v1
  def testZeroTruePositivesAndFalseNegativesGivesOneAUC(self):
    # [FUNCTION DOC]
    # Scenario: all-zero predictions and labels (no positives at all). Expectation: the ROC AUC is
    # defined as one to six places.
    with self.cached_session():
      predictions = array_ops.zeros([4], dtype=dtypes_lib.float32)
      labels = array_ops.zeros([4])
      auc, update_op = metrics.auc(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1, self.evaluate(update_op), 6)

      self.assertAlmostEqual(1, self.evaluate(auc), 6)

  @test_util.run_deprecated_v1
  def testRecallOneAndPrecisionOneGivesOnePRAUC(self):
    # [FUNCTION DOC]
    # Scenario: all-one predictions and labels with the precision-recall curve. Expectation: the AUC is
    # one to six places.
    with self.cached_session():
      predictions = array_ops.ones([4], dtype=dtypes_lib.float32)
      labels = array_ops.ones([4])
      auc, update_op = metrics.auc(labels, predictions, curve='PR')

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1, self.evaluate(update_op), 6)

      self.assertAlmostEqual(1, self.evaluate(auc), 6)

  def np_auc(self, predictions, labels, weights):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation of ROC AUC used by the streaming AUC test.
    # KNOWN ISSUE: the docstring says weights is an array of shape N, but the code also accepts the
    # nothing value and then substitutes unit weights.
    # Inputs: predictions, labels, weights, one-dimensional arrays of equal length (weights may be
    # nothing). Positives are labels greater than zero. Sorts examples by descending prediction, computes
    # the cumulative weighted true-positive rate and sums it weighted over the negatives, normalized by
    # the total negative weight. Output: the area under the ROC curve as a number. No side effects.
    """Computes the AUC explicitly using Numpy.

    Args:
      predictions: an ndarray with shape [N].
      labels: an ndarray with shape [N].
      weights: an ndarray with shape [N].

    Returns:
      the area under the ROC curve.
    """
    if weights is None:
      weights = np.ones(np.size(predictions))
    is_positive = labels > 0
    num_positives = np.sum(weights[is_positive])
    num_negatives = np.sum(weights[~is_positive])

    # Sort descending:
    inds = np.argsort(-predictions)

    sorted_labels = labels[inds]
    sorted_weights = weights[inds]
    is_positive = sorted_labels > 0

    tp = np.cumsum(sorted_weights * is_positive) / num_positives
    return np.sum((sorted_weights * tp)[~is_positive]) / num_negatives

  @test_util.run_deprecated_v1
  def testWithMultipleUpdates(self):
    # [FUNCTION DOC]
    # Scenario: AUC accumulated over one hundred batches of ten examples, compared with a numpy reference.
    # Setup: one thousand noisy samples correlated with random binary labels, clipped to the range zero
    # to one, three weight choices (none, all ones, random exponential), each fed in batches through
    # queues. Action: for each weight choice, enqueue all batches, build an ROC AUC metric with five
    # hundred thresholds, run the update once per batch. Expectation: the streamed AUC matches the
    # reference to two decimal places (it is only approximate due to threshold discretization).
    num_samples = 1000
    batch_size = 10
    num_batches = int(num_samples / batch_size)

    # Create the labels and data.
    labels = np.random.randint(0, 2, size=num_samples)
    noise = np.random.normal(0.0, scale=0.2, size=num_samples)
    predictions = 0.4 + 0.2 * labels + noise
    predictions[predictions > 1] = 1
    predictions[predictions < 0] = 0

    def _enqueue_as_batches(x, enqueue_ops):
      # [FUNCTION DOC]
      # HELPER: local utility of the multiple-update AUC test for feeding data through a queue.
      # Splits a one-dimensional numpy array into equal batches (float32), creates a float queue holding
      # them, appends one enqueue operation per batch to the matching per-batch list of operations, and
      # returns the dequeue tensor. Inputs: x, the data array; enqueue_ops, a list with one list per batch
      # that receives the new enqueue operations (mutated). Output: a tensor that dequeues one batch.
      # Depends on the enclosing test's batch count and batch size.
      x_batches = x.astype(np.float32).reshape((num_batches, batch_size))
      x_queue = data_flow_ops.FIFOQueue(
          num_batches, dtypes=dtypes_lib.float32, shapes=(batch_size,))
      for i in range(num_batches):
        enqueue_ops[i].append(x_queue.enqueue(x_batches[i, :]))
      return x_queue.dequeue()

    for weights in (None, np.ones(num_samples), np.random.exponential(
        scale=1.0, size=num_samples)):
      expected_auc = self.np_auc(predictions, labels, weights)

      with self.cached_session() as sess:
        enqueue_ops = [[] for i in range(num_batches)]
        tf_predictions = _enqueue_as_batches(predictions, enqueue_ops)
        tf_labels = _enqueue_as_batches(labels, enqueue_ops)
        tf_weights = (_enqueue_as_batches(weights, enqueue_ops) if
                      weights is not None else None)

        for i in range(num_batches):
          sess.run(enqueue_ops[i])

        auc, update_op = metrics.auc(tf_labels,
                                     tf_predictions,
                                     curve='ROC',
                                     num_thresholds=500,
                                     weights=tf_weights)

        self.evaluate(variables.local_variables_initializer())
        for i in range(num_batches):
          self.evaluate(update_op)

        # Since this is only approximate, we can't expect a 6 digits match.
        # Although with higher number of samples/thresholds we should see the
        # accuracy improving
        self.assertAlmostEqual(expected_auc, self.evaluate(auc), 2)


class SpecificityAtSensitivityTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the specificity-at-sensitivity metric (the specificity achieved at the threshold
  # reaching a target sensitivity). Covers variables, collections, idempotent reads, perfect
  # predictions, high and low sensitivity targets, and weights. Fixture: every test seeds numpy and
  # resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the specificity-at-sensitivity metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the four confusion counters true positives, false negatives, false positives and true negatives under the specificity-at-sensitivity scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.specificity_at_sensitivity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        sensitivity=0.7)
    _assert_metric_variables(self,
                             ('specificity_at_sensitivity/true_positives:0',
                              'specificity_at_sensitivity/false_negatives:0',
                              'specificity_at_sensitivity/false_positives:0',
                              'specificity_at_sensitivity/true_negatives:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the specificity-at-sensitivity metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.specificity_at_sensitivity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        sensitivity=0.7,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the specificity-at-sensitivity metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.specificity_at_sensitivity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        sensitivity=0.7,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the specificity-at-sensitivity metric repeatedly does not change it.
    # Inputs: random floating-point predictions and random integer labels of shape ten by three with a fixed seed, using a target sensitivity of 0.7.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.float32, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=2, dtype=dtypes_lib.int64, seed=1)
    specificity, update_op = metrics.specificity_at_sensitivity(
        labels, predictions, sensitivity=0.7)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_specificity = self.evaluate(specificity)
      for _ in range(10):
        self.assertAlmostEqual(initial_specificity, self.evaluate(specificity),
                               5)

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    # [FUNCTION DOC]
    # Scenario: random binary inputs used as labels and float predictions, target sensitivity 0.7.
    # Expectation: specificity is exactly one for update result and value.
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
    labels = constant_op.constant(inputs)
    specificity, update_op = metrics.specificity_at_sensitivity(
        labels, predictions, sensitivity=0.7)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1, self.evaluate(update_op))
      self.assertEqual(1, self.evaluate(specificity))

  @test_util.run_deprecated_v1
  def testSomeCorrectHighSensitivity(self):
    # [FUNCTION DOC]
    # Scenario: ten examples, five negatives and five positives, target sensitivity 0.8. Expectation:
    # specificity is 1.0 for update result and value.
    predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0, 0.1, 0.45, 0.5, 0.8, 0.9]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = constant_op.constant(
        predictions_values, dtype=dtypes_lib.float32)
    labels = constant_op.constant(labels_values)
    specificity, update_op = metrics.specificity_at_sensitivity(
        labels, predictions, sensitivity=0.8)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1.0, self.evaluate(update_op))
      self.assertAlmostEqual(1.0, self.evaluate(specificity))

  @test_util.run_deprecated_v1
  def testSomeCorrectLowSensitivity(self):
    # [FUNCTION DOC]
    # Scenario: ten examples with overlapping scores, target sensitivity 0.4. Expectation: specificity is
    # 0.6 for update result and value.
    predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0, 0.1, 0.2, 0.2, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = constant_op.constant(
        predictions_values, dtype=dtypes_lib.float32)
    labels = constant_op.constant(labels_values)
    specificity, update_op = metrics.specificity_at_sensitivity(
        labels, predictions, sensitivity=0.4)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      self.assertAlmostEqual(0.6, self.evaluate(update_op))
      self.assertAlmostEqual(0.6, self.evaluate(specificity))

  @test_util.run_deprecated_v1
  def testWeighted1d_multipleLabelDtypes(self):
    # [FUNCTION DOC]
    # Scenario: the low-sensitivity data with a single scalar-like weight of three, labels cast to boolean,
    # int32 and float32 in turn. Expectation: specificity stays 0.6 for each dtype, showing a
    # uniform weight changes nothing.
    for label_dtype in (dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
      predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0, 0.1, 0.2, 0.2, 0.26, 0.26]
      labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
      weights_values = [3]

      predictions = constant_op.constant(
          predictions_values, dtype=dtypes_lib.float32)
      labels = math_ops.cast(labels_values, dtype=label_dtype)
      weights = constant_op.constant(weights_values)
      specificity, update_op = metrics.specificity_at_sensitivity(
          labels, predictions, weights=weights, sensitivity=0.4)

      with self.cached_session():
        self.evaluate(variables.local_variables_initializer())

        self.assertAlmostEqual(0.6, self.evaluate(update_op))
        self.assertAlmostEqual(0.6, self.evaluate(specificity))

  @test_util.run_deprecated_v1
  def testWeighted2d(self):
    # [FUNCTION DOC]
    # Scenario: the low-sensitivity data with per-example weights one through ten. Expectation: the
    # weighted specificity is eight fifteenths.
    predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0, 0.1, 0.2, 0.2, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    weights_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    predictions = constant_op.constant(
        predictions_values, dtype=dtypes_lib.float32)
    labels = constant_op.constant(labels_values)
    weights = constant_op.constant(weights_values)
    specificity, update_op = metrics.specificity_at_sensitivity(
        labels, predictions, weights=weights, sensitivity=0.4)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      self.assertAlmostEqual(8.0 / 15.0, self.evaluate(update_op))
      self.assertAlmostEqual(8.0 / 15.0, self.evaluate(specificity))


class SensitivityAtSpecificityTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the sensitivity-at-specificity metric (the sensitivity achieved at the threshold
  # reaching a target specificity). Covers variables, collections, idempotent reads, perfect
  # predictions, high and low specificity targets, and weights. Fixture: every test seeds numpy and
  # resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the sensitivity-at-specificity metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the four confusion counters true positives, false negatives, false positives and true negatives under the sensitivity-at-specificity scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.sensitivity_at_specificity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        specificity=0.7)
    _assert_metric_variables(self,
                             ('sensitivity_at_specificity/true_positives:0',
                              'sensitivity_at_specificity/false_negatives:0',
                              'sensitivity_at_specificity/false_positives:0',
                              'sensitivity_at_specificity/true_negatives:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the sensitivity-at-specificity metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.sensitivity_at_specificity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        specificity=0.7,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the sensitivity-at-specificity metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.sensitivity_at_specificity(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        specificity=0.7,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the sensitivity-at-specificity metric repeatedly does not change it.
    # Inputs: random floating-point predictions and random integer labels of shape ten by three with a fixed seed, using a target specificity of 0.7.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.float32, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=2, dtype=dtypes_lib.int64, seed=1)
    sensitivity, update_op = metrics.sensitivity_at_specificity(
        labels, predictions, specificity=0.7)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_sensitivity = self.evaluate(sensitivity)
      for _ in range(10):
        self.assertAlmostEqual(initial_sensitivity, self.evaluate(sensitivity),
                               5)

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    # [FUNCTION DOC]
    # Scenario: random binary inputs used as labels and float predictions, target specificity 0.7.
    # Expectation: sensitivity is one to six places. (The local variable is misleadingly named for
    # specificity but holds the sensitivity metric.)
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
    labels = constant_op.constant(inputs)
    specificity, update_op = metrics.sensitivity_at_specificity(
        labels, predictions, specificity=0.7)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1.0, self.evaluate(update_op), 6)
      self.assertAlmostEqual(1.0, self.evaluate(specificity), 6)

  @test_util.run_deprecated_v1
  def testSomeCorrectHighSpecificity(self):
    # [FUNCTION DOC]
    # Scenario: ten examples with target specificity 0.8. Expectation: sensitivity is 0.8.
    predictions_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.1, 0.45, 0.5, 0.8, 0.9]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = constant_op.constant(
        predictions_values, dtype=dtypes_lib.float32)
    labels = constant_op.constant(labels_values)
    specificity, update_op = metrics.sensitivity_at_specificity(
        labels, predictions, specificity=0.8)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.8, self.evaluate(update_op))
      self.assertAlmostEqual(0.8, self.evaluate(specificity))

  @test_util.run_deprecated_v1
  def testSomeCorrectLowSpecificity(self):
    # [FUNCTION DOC]
    # Scenario: ten examples with target specificity 0.4. Expectation: sensitivity is 0.6.
    predictions_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = constant_op.constant(
        predictions_values, dtype=dtypes_lib.float32)
    labels = constant_op.constant(labels_values)
    specificity, update_op = metrics.sensitivity_at_specificity(
        labels, predictions, specificity=0.4)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(0.6, self.evaluate(update_op))
      self.assertAlmostEqual(0.6, self.evaluate(specificity))

  @test_util.run_deprecated_v1
  def testWeighted_multipleLabelDtypes(self):
    # [FUNCTION DOC]
    # Scenario: per-example weights one through ten and target specificity 0.4, with labels cast to
    # boolean, int32 and float32 in turn. Expectation: sensitivity is 0.675 for each dtype.
    for label_dtype in (dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
      predictions_values = [
          0.0, 0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.25, 0.26, 0.26]
      labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
      weights_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

      predictions = constant_op.constant(
          predictions_values, dtype=dtypes_lib.float32)
      labels = math_ops.cast(labels_values, dtype=label_dtype)
      weights = constant_op.constant(weights_values)
      specificity, update_op = metrics.sensitivity_at_specificity(
          labels, predictions, weights=weights, specificity=0.4)

      with self.cached_session():
        self.evaluate(variables.local_variables_initializer())
        self.assertAlmostEqual(0.675, self.evaluate(update_op))
        self.assertAlmostEqual(0.675, self.evaluate(specificity))


# TODO(nsilberman): Break this up into two sets of tests.
class PrecisionRecallThresholdsTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: a to-do comment above the class says this class should be split into two sets of
  # tests (it exercises both precision at thresholds and recall at thresholds together).
  # Tests for the precision-at-thresholds and recall-at-thresholds metrics, which return one value per
  # threshold. Covers variables, collections, idempotent reads, perfect and fully wrong predictions,
  # mixed label dtypes, weights, extreme thresholds, all-zero data and streaming over many batches.
  # Fixture: every test seeds numpy and resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the precision-at-thresholds metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the true-positive and false-positive accumulators under the precision-at-thresholds scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.precision_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0, 0.5, 1.0])
    _assert_metric_variables(self, (
        'precision_at_thresholds/true_positives:0',
        'precision_at_thresholds/false_positives:0',
    ))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument for both metrics. Builds precision-at-thresholds and then
    # recall-at-thresholds with the same custom collection name and asserts that the collection holds
    # the precision value followed by the recall value, in that order. Returns nothing.
    my_collection_name = '__metrics__'
    prec, _ = metrics.precision_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        metrics_collections=[my_collection_name])
    rec, _ = metrics.recall_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [prec, rec])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument for both metrics. Builds precision-at-thresholds and then
    # recall-at-thresholds with the same custom collection name and asserts that the collection holds
    # the precision update operation followed by the recall update operation. Returns nothing.
    my_collection_name = '__updates__'
    _, precision_op = metrics.precision_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        updates_collections=[my_collection_name])
    _, recall_op = metrics.recall_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        updates_collections=[my_collection_name])
    self.assertListEqual(
        ops.get_collection(my_collection_name), [precision_op, recall_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that repeated reads of the precision and recall values do not change them. Setup: random
    # float predictions and integer labels of shape ten by three, thresholds 0, 0.5 and 1.0. Action: run
    # both update operations once, record both values, then ten more times run the updates and assert the
    # values stay close to the first reads. Returns nothing.
    predictions = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.float32, seed=1)
    labels = random_ops.random_uniform(
        (10, 3), maxval=1, dtype=dtypes_lib.int64, seed=1)
    thresholds = [0, 0.5, 1.0]
    prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                    thresholds)
    rec, rec_op = metrics.recall_at_thresholds(labels, predictions, thresholds)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates, then verify idempotency.
      self.evaluate([prec_op, rec_op])
      initial_prec = self.evaluate(prec)
      initial_rec = self.evaluate(rec)
      for _ in range(10):
        self.evaluate([prec_op, rec_op])
        self.assertAllClose(initial_prec, prec)
        self.assertAllClose(initial_rec, rec)

  # TODO(nsilberman): fix tests (passing but incorrect).
  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a to-do comment above this test says the tests are "passing but incorrect"
    # (they pass but may not verify what is intended).
    # Scenario: random binary labels and identical float predictions, one threshold of 0.5. Expectation:
    # precision and recall are both exactly one.
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.cached_session():
      predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
      labels = constant_op.constant(inputs)
      thresholds = [0.5]
      prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                      thresholds)
      rec, rec_op = metrics.recall_at_thresholds(labels, predictions,
                                                 thresholds)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertEqual(1, self.evaluate(prec))
      self.assertEqual(1, self.evaluate(rec))

  @test_util.run_deprecated_v1
  def testSomeCorrect_multipleLabelDtypes(self):
    # [FUNCTION DOC]
    # Scenario: predictions 1,0,1,0 and labels 0,1,1,0 cast to boolean, int32 and float32, threshold 0.5.
    # Expectation: precision and recall are both 0.5 for every dtype.
    with self.cached_session():
      for label_dtype in (
          dtypes_lib.bool, dtypes_lib.int32, dtypes_lib.float32):
        predictions = constant_op.constant(
            [1, 0, 1, 0], shape=(1, 4), dtype=dtypes_lib.float32)
        labels = math_ops.cast(
            constant_op.constant([0, 1, 1, 0], shape=(1, 4)), dtype=label_dtype)
        thresholds = [0.5]
        prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                        thresholds)
        rec, rec_op = metrics.recall_at_thresholds(labels, predictions,
                                                   thresholds)

        self.evaluate(variables.local_variables_initializer())
        self.evaluate([prec_op, rec_op])

        self.assertAlmostEqual(0.5, self.evaluate(prec))
        self.assertAlmostEqual(0.5, self.evaluate(rec))

  @test_util.run_deprecated_v1
  def testAllIncorrect(self):
    # [FUNCTION DOC]
    # Scenario: labels are the complement of the predictions, threshold 0.5. Expectation: precision and
    # recall are both zero.
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.cached_session():
      predictions = constant_op.constant(inputs, dtype=dtypes_lib.float32)
      labels = constant_op.constant(1 - inputs, dtype=dtypes_lib.float32)
      thresholds = [0.5]
      prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                      thresholds)
      rec, rec_op = metrics.recall_at_thresholds(labels, predictions,
                                                 thresholds)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertAlmostEqual(0, self.evaluate(prec))
      self.assertAlmostEqual(0, self.evaluate(rec))

  @test_util.run_deprecated_v1
  def testWeights1d(self):
    # [FUNCTION DOC]
    # Scenario: two thresholds (0.5 and 1.1) and per-row weights 0 and 1 on a two by two example set.
    # Expectation: the low-threshold precision and recall are 1.0, the high-threshold ones are 0.0
    # (nothing is predicted positive), each to five places, after splitting the per-threshold results.
    with self.cached_session():
      predictions = constant_op.constant(
          [[1, 0], [1, 0]], shape=(2, 2), dtype=dtypes_lib.float32)
      labels = constant_op.constant([[0, 1], [1, 0]], shape=(2, 2))
      weights = constant_op.constant(
          [[0], [1]], shape=(2, 1), dtype=dtypes_lib.float32)
      thresholds = [0.5, 1.1]
      prec, prec_op = metrics.precision_at_thresholds(
          labels, predictions, thresholds, weights=weights)
      rec, rec_op = metrics.recall_at_thresholds(
          labels, predictions, thresholds, weights=weights)

      [prec_low, prec_high] = array_ops.split(
          value=prec, num_or_size_splits=2, axis=0)
      prec_low = array_ops.reshape(prec_low, shape=())
      prec_high = array_ops.reshape(prec_high, shape=())
      [rec_low, rec_high] = array_ops.split(
          value=rec, num_or_size_splits=2, axis=0)
      rec_low = array_ops.reshape(rec_low, shape=())
      rec_high = array_ops.reshape(rec_high, shape=())

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertAlmostEqual(1.0, self.evaluate(prec_low), places=5)
      self.assertAlmostEqual(0.0, self.evaluate(prec_high), places=5)
      self.assertAlmostEqual(1.0, self.evaluate(rec_low), places=5)
      self.assertAlmostEqual(0.0, self.evaluate(rec_high), places=5)

  @test_util.run_deprecated_v1
  def testWeights2d(self):
    # [FUNCTION DOC]
    # Scenario: same two thresholds with element-wise weights of shape two by two (zero for the first
    # row, one for the second). Expectation: low-threshold precision and recall 1.0, high-threshold 0.0.
    with self.cached_session():
      predictions = constant_op.constant(
          [[1, 0], [1, 0]], shape=(2, 2), dtype=dtypes_lib.float32)
      labels = constant_op.constant([[0, 1], [1, 0]], shape=(2, 2))
      weights = constant_op.constant(
          [[0, 0], [1, 1]], shape=(2, 2), dtype=dtypes_lib.float32)
      thresholds = [0.5, 1.1]
      prec, prec_op = metrics.precision_at_thresholds(
          labels, predictions, thresholds, weights=weights)
      rec, rec_op = metrics.recall_at_thresholds(
          labels, predictions, thresholds, weights=weights)

      [prec_low, prec_high] = array_ops.split(
          value=prec, num_or_size_splits=2, axis=0)
      prec_low = array_ops.reshape(prec_low, shape=())
      prec_high = array_ops.reshape(prec_high, shape=())
      [rec_low, rec_high] = array_ops.split(
          value=rec, num_or_size_splits=2, axis=0)
      rec_low = array_ops.reshape(rec_low, shape=())
      rec_high = array_ops.reshape(rec_high, shape=())

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertAlmostEqual(1.0, self.evaluate(prec_low), places=5)
      self.assertAlmostEqual(0.0, self.evaluate(prec_high), places=5)
      self.assertAlmostEqual(1.0, self.evaluate(rec_low), places=5)
      self.assertAlmostEqual(0.0, self.evaluate(rec_high), places=5)

  @test_util.run_deprecated_v1
  def testExtremeThresholds(self):
    # [FUNCTION DOC]
    # Scenario: thresholds -1.0 and 2.0, below and above every prediction. Setup: predictions 1,0,1,0 and
    # labels 0,1,1,1. Expectation: at the low threshold everything is predicted positive giving precision
    # 0.75 and recall 1.0; at the high threshold nothing is predicted positive giving 0.0 for both.
    with self.cached_session():
      predictions = constant_op.constant(
          [1, 0, 1, 0], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant([0, 1, 1, 1], shape=(1, 4))
      thresholds = [-1.0, 2.0]  # lower/higher than any values
      prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                      thresholds)
      rec, rec_op = metrics.recall_at_thresholds(labels, predictions,
                                                 thresholds)

      [prec_low, prec_high] = array_ops.split(
          value=prec, num_or_size_splits=2, axis=0)
      [rec_low, rec_high] = array_ops.split(
          value=rec, num_or_size_splits=2, axis=0)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertAlmostEqual(0.75, self.evaluate(prec_low))
      self.assertAlmostEqual(0.0, self.evaluate(prec_high))
      self.assertAlmostEqual(1.0, self.evaluate(rec_low))
      self.assertAlmostEqual(0.0, self.evaluate(rec_high))

  @test_util.run_deprecated_v1
  def testZeroLabelsPredictions(self):
    # [FUNCTION DOC]
    # Scenario: all-zero predictions and labels, threshold 0.5. Expectation: precision and recall are
    # defined as zero, to six places.
    with self.cached_session():
      predictions = array_ops.zeros([4], dtype=dtypes_lib.float32)
      labels = array_ops.zeros([4])
      thresholds = [0.5]
      prec, prec_op = metrics.precision_at_thresholds(labels, predictions,
                                                      thresholds)
      rec, rec_op = metrics.recall_at_thresholds(labels, predictions,
                                                 thresholds)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([prec_op, rec_op])

      self.assertAlmostEqual(0, self.evaluate(prec), 6)
      self.assertAlmostEqual(0, self.evaluate(rec), 6)

  @test_util.run_deprecated_v1
  def testWithMultipleUpdates(self):
    # [FUNCTION DOC]
    # Scenario: precision and recall at a threshold of 0.3 accumulated over one hundred batches of ten
    # noisy examples. Setup: one thousand random binary labels, noisy predictions clipped to the range
    # zero to one, expected values counted with plain loops (with a small epsilon in the denominators),
    # data fed through queues. Action: run both updates once per batch. Expectation: streamed precision
    # and recall match the counted values to two decimal places (approximate).
    num_samples = 1000
    batch_size = 10
    num_batches = int(num_samples / batch_size)

    # Create the labels and data.
    labels = np.random.randint(0, 2, size=(num_samples, 1))
    noise = np.random.normal(0.0, scale=0.2, size=(num_samples, 1))
    predictions = 0.4 + 0.2 * labels + noise
    predictions[predictions > 1] = 1
    predictions[predictions < 0] = 0
    thresholds = [0.3]

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(num_samples):
      if predictions[i] > thresholds[0]:
        if labels[i] == 1:
          tp += 1
        else:
          fp += 1
      else:
        if labels[i] == 1:
          fn += 1
        else:
          tn += 1
    epsilon = 1e-7
    expected_prec = tp / (epsilon + tp + fp)
    expected_rec = tp / (epsilon + tp + fn)

    labels = labels.astype(np.float32)
    predictions = predictions.astype(np.float32)

    with self.cached_session() as sess:
      # Reshape the data so its easy to queue up:
      predictions_batches = predictions.reshape((batch_size, num_batches))
      labels_batches = labels.reshape((batch_size, num_batches))

      # Enqueue the data:
      predictions_queue = data_flow_ops.FIFOQueue(
          num_batches, dtypes=dtypes_lib.float32, shapes=(batch_size,))
      labels_queue = data_flow_ops.FIFOQueue(
          num_batches, dtypes=dtypes_lib.float32, shapes=(batch_size,))

      for i in range(int(num_batches)):
        tf_prediction = constant_op.constant(predictions_batches[:, i])
        tf_label = constant_op.constant(labels_batches[:, i])
        sess.run([
            predictions_queue.enqueue(tf_prediction),
            labels_queue.enqueue(tf_label)
        ])

      tf_predictions = predictions_queue.dequeue()
      tf_labels = labels_queue.dequeue()

      prec, prec_op = metrics.precision_at_thresholds(tf_labels, tf_predictions,
                                                      thresholds)
      rec, rec_op = metrics.recall_at_thresholds(tf_labels, tf_predictions,
                                                 thresholds)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(int(num_samples / batch_size)):
        self.evaluate([prec_op, rec_op])
      # Since this is only approximate, we can't expect a 6 digits match.
      # Although with higher number of samples/thresholds we should see the
      # accuracy improving
      self.assertAlmostEqual(expected_prec, self.evaluate(prec), 2)
      self.assertAlmostEqual(expected_rec, self.evaluate(rec), 2)


def _test_precision_at_k(predictions,
                         labels,
                         k,
                         expected,
                         class_id=None,
                         weights=None,
                         test_case=None):
  # [FUNCTION DOC]
  # HELPER: shared driver for the precision-at-k tests (wraps the metric over dense scores).
  # Builds precision at k in a fresh graph and session of the given test case, and checks its behaviour.
  # Inputs: predictions, a nested list of float scores; labels, a dense or sparse label value; k, the
  # cutoff; expected, the number the metric should give (not-a-number is allowed); class_id, optional
  # class to restrict to; weights, optional weights (converted to float constants); test_case, the
  # running test case (required in practice, its default is nothing). Steps: first asserts that reading
  # the value or running the update before initializing the local variables raises an op error; then
  # initializes local variables, runs the update and reads the value, comparing both to the expected
  # value (a not-a-number check when expected is not-a-number, exact equality otherwise).
  # Returns nothing; failures are assertion errors.
  with ops.Graph().as_default() as g, test_case.test_session(g):
    if weights is not None:
      weights = constant_op.constant(weights, dtypes_lib.float32)
    metric, update = metrics.precision_at_k(
        predictions=constant_op.constant(predictions, dtypes_lib.float32),
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)

    # Fails without initialized vars.
    test_case.assertRaises(errors_impl.OpError, metric.eval)
    test_case.assertRaises(errors_impl.OpError, update.eval)
    variables.variables_initializer(variables.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      _assert_nan(test_case, update.eval())
      _assert_nan(test_case, metric.eval())
    else:
      test_case.assertEqual(expected, update.eval())
      test_case.assertEqual(expected, metric.eval())


def _test_precision_at_top_k(
    predictions_idx,
    labels,
    expected,
    k=None,
    class_id=None,
    weights=None,
    test_case=None):
  # [FUNCTION DOC]
  # HELPER: shared driver for the precision-at-top-k tests (takes precomputed top-k class indices).
  # Same procedure as the precision-at-k driver but the metric is built from predictions_idx, a nested
  # list of integer class indices (converted to int32), plus labels, k (optional), class_id (optional),
  # weights (optional, converted to float) and test_case. Asserts that evaluation before variable
  # initialization raises an op error, then that update result and value both equal expected, or are
  # both not-a-number if expected is. Returns nothing; failures are assertion errors.
  with ops.Graph().as_default() as g, test_case.test_session(g):
    if weights is not None:
      weights = constant_op.constant(weights, dtypes_lib.float32)
    metric, update = metrics.precision_at_top_k(
        predictions_idx=constant_op.constant(predictions_idx, dtypes_lib.int32),
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)

    # Fails without initialized vars.
    test_case.assertRaises(errors_impl.OpError, metric.eval)
    test_case.assertRaises(errors_impl.OpError, update.eval)
    variables.variables_initializer(variables.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      test_case.assertTrue(math.isnan(update.eval()))
      test_case.assertTrue(math.isnan(metric.eval()))
    else:
      test_case.assertEqual(expected, update.eval())
      test_case.assertEqual(expected, metric.eval())


def _test_average_precision_at_k(predictions,
                                 labels,
                                 k,
                                 expected,
                                 weights=None,
                                 test_case=None):
  # [FUNCTION DOC]
  # HELPER: shared driver for the average-precision-at-k tests.
  # Inputs: predictions, nested list of float scores; labels; k; expected number (not-a-number allowed);
  # weights, optional per-example weights; test_case. Builds the average-precision-at-k metric in a
  # fresh graph and session, asserts that evaluation before variable initialization raises an op error,
  # then initializes local variables and checks the update result and the value against the expected
  # number (approximate equality, or a not-a-number check). Returns nothing.
  with ops.Graph().as_default() as g, test_case.test_session(g):
    if weights is not None:
      weights = constant_op.constant(weights, dtypes_lib.float32)
    predictions = constant_op.constant(predictions, dtypes_lib.float32)
    metric, update = metrics.average_precision_at_k(
        labels, predictions, k, weights=weights)

    # Fails without initialized vars.
    test_case.assertRaises(errors_impl.OpError, metric.eval)
    test_case.assertRaises(errors_impl.OpError, update.eval)
    variables.variables_initializer(variables.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      _assert_nan(test_case, update.eval())
      _assert_nan(test_case, metric.eval())
    else:
      test_case.assertAlmostEqual(expected, update.eval())
      test_case.assertAlmostEqual(expected, metric.eval())


class SingleLabelPrecisionAtKTest(test.TestCase):
  # [CLASS DOC]
  # Tests precision at k and average precision at k when every example has exactly one label. The
  # labels are provided in four equivalent forms (sparse of rank one, sparse of rank two, dense one
  # dimensional and dense column) that must all give identical results.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Creates two examples with four class scores each, precomputed top-one class indices, and
    # the four equivalent label forms (one-dimensional sparse, two-dimensional sparse, dense vector and
    # dense column), built from the indicator rows and class ids. Also stores partially applied versions
    # of the three driver helpers (precision at k, precision at top k, average precision at k) bound to
    # this test case. Does not reset the graph. Returns nothing.
    self._predictions = ((0.1, 0.3, 0.2, 0.4), (0.1, 0.2, 0.3, 0.4))
    self._predictions_idx = [[3], [3]]
    indicator_labels = ((0, 0, 0, 1), (0, 0, 1, 0))
    class_labels = (3, 2)
    # Sparse vs dense, and 1d vs 2d labels should all be handled the same.
    self._labels = (
        _binary_2d_label_to_1d_sparse_value(indicator_labels),
        _binary_2d_label_to_2d_sparse_value(indicator_labels), np.array(
            class_labels, dtype=np.int64), np.array(
                [[class_id] for class_id in class_labels], dtype=np.int64))
    self._test_precision_at_k = functools.partial(
        _test_precision_at_k, test_case=self)
    self._test_precision_at_top_k = functools.partial(
        _test_precision_at_top_k, test_case=self)
    self._test_average_precision_at_k = functools.partial(
        _test_average_precision_at_k, test_case=self)

  @test_util.run_deprecated_v1
  def test_at_k1_nan(self):
    # [FUNCTION DOC]
    # Scenario: classes with no predictions at k equals one, namely 0, 1 and 2, plus the out-of-range
    # classes -1 and 4. Action: for each label form and each of those class ids run the precision-at-k
    # and precision-at-top-k drivers. Expectation: every result is not-a-number.
    for labels in self._labels:
      # Classes 0,1,2 have 0 predictions, classes -1 and 4 are out of range.
      for class_id in (-1, 0, 1, 2, 4):
        self._test_precision_at_k(
            self._predictions, labels, k=1, expected=NAN, class_id=class_id)
        self._test_precision_at_top_k(
            self._predictions_idx, labels, k=1, expected=NAN, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_at_k1(self):
    # [FUNCTION DOC]
    # Scenario: k equals one with the four label forms. Expectations: for class three (one label, two
    # predictions, one correct) precision is one half for both precision-at-k and precision-at-top-k;
    # for all classes it is one half for those two metrics and for average precision at k as well.
    for labels in self._labels:
      # Class 3: 1 label, 2 predictions, 1 correct.
      self._test_precision_at_k(
          self._predictions, labels, k=1, expected=1.0 / 2, class_id=3)
      self._test_precision_at_top_k(
          self._predictions_idx, labels, k=1, expected=1.0 / 2, class_id=3)

      # All classes: 2 labels, 2 predictions, 1 correct.
      self._test_precision_at_k(
          self._predictions, labels, k=1, expected=1.0 / 2)
      self._test_precision_at_top_k(
          self._predictions_idx, labels, k=1, expected=1.0 / 2)
      self._test_average_precision_at_k(
          self._predictions, labels, k=1, expected=1.0 / 2)


class MultiLabelPrecisionAtKTest(test.TestCase):
  # [CLASS DOC]
  # Tests precision at k, precision at top k and average precision at k with several labels per example,
  # sparse and dense labels, labels out of the valid range, different label counts per example,
  # per-class results, three-dimensional inputs and weights.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Stores partially applied versions of the three driver helpers (precision at k, precision
    # at top k, average precision at k) bound to this test case. Returns nothing.
    self._test_precision_at_k = functools.partial(
        _test_precision_at_k, test_case=self)
    self._test_precision_at_top_k = functools.partial(
        _test_precision_at_top_k, test_case=self)
    self._test_average_precision_at_k = functools.partial(
        _test_average_precision_at_k, test_case=self)

  @test_util.run_deprecated_v1
  def test_average_precision(self):
    # [FUNCTION DOC]
    # Scenario: two worked examples of average precision (the first taken from a public article on mean
    # average precision). Action: for k from one to four, check precision at k, precision at top k and
    # average precision at k for each example alone, then for both examples together (expected is the
    # mean of the two), then for the average precision weighted by 0.3 and 0.6 (expected is the weighted
    # mean). Returns nothing.
    # Example 1.
    # Matches example here:
    # fastml.com/what-you-wanted-to-know-about-mean-average-precision
    labels_ex1 = (0, 1, 2, 3, 4)
    labels = np.array([labels_ex1], dtype=np.int64)
    predictions_ex1 = (0.2, 0.1, 0.0, 0.4, 0.0, 0.5, 0.3)
    predictions = (predictions_ex1,)
    predictions_idx_ex1 = (5, 3, 6, 0, 1)
    precision_ex1 = (0.0 / 1, 1.0 / 2, 1.0 / 3, 2.0 / 4)
    avg_precision_ex1 = (0.0 / 1, precision_ex1[1] / 2, precision_ex1[1] / 3,
                         (precision_ex1[1] + precision_ex1[3]) / 4)
    for i in range(4):
      k = i + 1
      self._test_precision_at_k(
          predictions, labels, k, expected=precision_ex1[i])
      self._test_precision_at_top_k(
          (predictions_idx_ex1[:k],), labels, k=k, expected=precision_ex1[i])
      self._test_average_precision_at_k(
          predictions, labels, k, expected=avg_precision_ex1[i])

    # Example 2.
    labels_ex2 = (0, 2, 4, 5, 6)
    labels = np.array([labels_ex2], dtype=np.int64)
    predictions_ex2 = (0.3, 0.5, 0.0, 0.4, 0.0, 0.1, 0.2)
    predictions = (predictions_ex2,)
    predictions_idx_ex2 = (1, 3, 0, 6, 5)
    precision_ex2 = (0.0 / 1, 0.0 / 2, 1.0 / 3, 2.0 / 4)
    avg_precision_ex2 = (0.0 / 1, 0.0 / 2, precision_ex2[2] / 3,
                         (precision_ex2[2] + precision_ex2[3]) / 4)
    for i in range(4):
      k = i + 1
      self._test_precision_at_k(
          predictions, labels, k, expected=precision_ex2[i])
      self._test_precision_at_top_k(
          (predictions_idx_ex2[:k],), labels, k=k, expected=precision_ex2[i])
      self._test_average_precision_at_k(
          predictions, labels, k, expected=avg_precision_ex2[i])

    # Both examples, we expect both precision and average precision to be the
    # average of the 2 examples.
    labels = np.array([labels_ex1, labels_ex2], dtype=np.int64)
    predictions = (predictions_ex1, predictions_ex2)
    streaming_precision = [(ex1 + ex2) / 2
                           for ex1, ex2 in zip(precision_ex1, precision_ex2)]
    streaming_average_precision = [
        (ex1 + ex2) / 2
        for ex1, ex2 in zip(avg_precision_ex1, avg_precision_ex2)
    ]
    for i in range(4):
      k = i + 1
      predictions_idx = (predictions_idx_ex1[:k], predictions_idx_ex2[:k])
      self._test_precision_at_k(
          predictions, labels, k, expected=streaming_precision[i])
      self._test_precision_at_top_k(
          predictions_idx, labels, k=k, expected=streaming_precision[i])
      self._test_average_precision_at_k(
          predictions, labels, k, expected=streaming_average_precision[i])

    # Weighted examples, we expect streaming average precision to be the
    # weighted average of the 2 examples.
    weights = (0.3, 0.6)
    streaming_average_precision = [
        (weights[0] * ex1 + weights[1] * ex2) / (weights[0] + weights[1])
        for ex1, ex2 in zip(avg_precision_ex1, avg_precision_ex2)
    ]
    for i in range(4):
      k = i + 1
      self._test_average_precision_at_k(
          predictions,
          labels,
          k,
          expected=streaming_average_precision[i],
          weights=weights)

  @test_util.run_deprecated_v1
  def test_average_precision_some_labels_out_of_range(self):
    # [FUNCTION DOC]
    # Scenario: the first worked example with extra labels, minus one and seven, outside the valid range.
    # Action: for k from one to four, check precision at k, at top k and average precision at k.
    # Expectation: the results equal those of the in-range example, i.e. out-of-range labels are ignored.
    """Tests that labels outside the [0, n_classes) range are ignored."""
    labels_ex1 = (-1, 0, 1, 2, 3, 4, 7)
    labels = np.array([labels_ex1], dtype=np.int64)
    predictions_ex1 = (0.2, 0.1, 0.0, 0.4, 0.0, 0.5, 0.3)
    predictions = (predictions_ex1,)
    predictions_idx_ex1 = (5, 3, 6, 0, 1)
    precision_ex1 = (0.0 / 1, 1.0 / 2, 1.0 / 3, 2.0 / 4)
    avg_precision_ex1 = (0.0 / 1, precision_ex1[1] / 2, precision_ex1[1] / 3,
                         (precision_ex1[1] + precision_ex1[3]) / 4)
    for i in range(4):
      k = i + 1
      self._test_precision_at_k(
          predictions, labels, k, expected=precision_ex1[i])
      self._test_precision_at_top_k(
          (predictions_idx_ex1[:k],), labels, k=k, expected=precision_ex1[i])
      self._test_average_precision_at_k(
          predictions, labels, k, expected=avg_precision_ex1[i])

  @test_util.run_deprecated_v1
  def test_average_precision_different_num_labels(self):
    # [FUNCTION DOC]
    # Scenario: two examples with different numbers of labels, given as sparse labels and as dense labels
    # padded with minus one. Action: for k from one to four, check the three metrics. Expectation: results
    # equal the means of the per-example values.
    """Tests the case where the numbers of labels differ across examples."""
    predictions = [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]
    sparse_labels = _binary_2d_label_to_2d_sparse_value(
        [[0, 0, 1, 1], [0, 0, 0, 1]])
    dense_labels = np.array([[2, 3], [3, -1]], dtype=np.int64)
    predictions_idx_ex1 = np.array(((0, 1, 2, 3), (3, 2, 1, 0)))
    precision_ex1 = ((0.0 / 1, 0.0 / 2, 1.0 / 3, 2.0 / 4),
                     (1.0 / 1, 1.0 / 2, 1.0 / 3, 1.0 / 4))
    mean_precision_ex1 = np.mean(precision_ex1, axis=0)
    avg_precision_ex1 = (
        (0.0 / 1, 0.0 / 2, 1.0 / 3 / 2, (1.0 / 3 + 2.0 / 4) / 2),
        (1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0 / 1))
    mean_avg_precision_ex1 = np.mean(avg_precision_ex1, axis=0)
    for labels in (sparse_labels, dense_labels):
      for i in range(4):
        k = i + 1
        self._test_precision_at_k(
            predictions, labels, k, expected=mean_precision_ex1[i])
        self._test_precision_at_top_k(
            predictions_idx_ex1[:, :k], labels, k=k,
            expected=mean_precision_ex1[i])
        self._test_average_precision_at_k(
            predictions, labels, k, expected=mean_avg_precision_ex1[i])

  @test_util.run_deprecated_v1
  def test_three_labels_at_k5_no_predictions(self):
    # [FUNCTION DOC]
    # Scenario: two examples over ten classes, k equals five. Action: for sparse and dense labels, and for
    # classes -1, 1, 3, 8 and 10 (classes that are never predicted or out of range), check precision.
    # Expectation: every result is not-a-number.
    predictions = [[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                   [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    predictions_idx = [[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]]
    sparse_labels = _binary_2d_label_to_2d_sparse_value(
        [[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Classes 1,3,8 have 0 predictions, classes -1 and 10 are out of range.
      for class_id in (-1, 1, 3, 8, 10):
        self._test_precision_at_k(
            predictions, labels, k=5, expected=NAN, class_id=class_id)
        self._test_precision_at_top_k(
            predictions_idx, labels, k=5, expected=NAN, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_three_labels_at_k5_no_labels(self):
    # [FUNCTION DOC]
    # Scenario: same data; classes 0, 4, 6 and 9 are predicted but never labeled. Expectation:
    # precision is exactly zero for those classes, for sparse and dense labels.
    predictions = [[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                   [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    predictions_idx = [[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]]
    sparse_labels = _binary_2d_label_to_2d_sparse_value(
        [[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Classes 0,4,6,9: 0 labels, >=1 prediction.
      for class_id in (0, 4, 6, 9):
        self._test_precision_at_k(
            predictions, labels, k=5, expected=0.0, class_id=class_id)
        self._test_precision_at_top_k(
            predictions_idx, labels, k=5, expected=0.0, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_three_labels_at_k5(self):
    # [FUNCTION DOC]
    # Scenario: same data, sparse and dense labels. Expectations: class two gives 2/2, class five gives
    # 1/1, class seven gives 0/1, and all classes together give 3/10, for both precision at k and
    # precision at top k.
    predictions = [[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                   [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    predictions_idx = [[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]]
    sparse_labels = _binary_2d_label_to_2d_sparse_value(
        [[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Class 2: 2 labels, 2 correct predictions.
      self._test_precision_at_k(
          predictions, labels, k=5, expected=2.0 / 2, class_id=2)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=2.0 / 2, class_id=2)

      # Class 5: 1 label, 1 correct prediction.
      self._test_precision_at_k(
          predictions, labels, k=5, expected=1.0 / 1, class_id=5)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=1.0 / 1, class_id=5)

      # Class 7: 1 label, 1 incorrect prediction.
      self._test_precision_at_k(
          predictions, labels, k=5, expected=0.0 / 1, class_id=7)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=0.0 / 1, class_id=7)

      # All classes: 10 predictions, 3 correct.
      self._test_precision_at_k(
          predictions, labels, k=5, expected=3.0 / 10)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=3.0 / 10)

  @test_util.run_deprecated_v1
  def test_three_labels_at_k5_some_out_of_range(self):
    # [FUNCTION DOC]
    # Scenario: sparse labels containing minus one and ten, which are outside the valid class range.
    # Expectation: they are ignored; class two 2/2, class five 1/1, class seven 0/1, all classes 3/10.
    """Tests that labels outside the [0, n_classes) range are ignored."""
    predictions = [[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                   [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    predictions_idx = [[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]]
    sp_labels = sparse_tensor.SparseTensorValue(
        indices=[[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2],
                 [1, 3]],
        # values -1 and 10 are outside the [0, n_classes) range and are ignored.
        values=np.array([2, 7, -1, 8, 1, 2, 5, 10], np.int64),
        dense_shape=[2, 4])

    # Class 2: 2 labels, 2 correct predictions.
    self._test_precision_at_k(
        predictions, sp_labels, k=5, expected=2.0 / 2, class_id=2)
    self._test_precision_at_top_k(
        predictions_idx, sp_labels, k=5, expected=2.0 / 2, class_id=2)

    # Class 5: 1 label, 1 correct prediction.
    self._test_precision_at_k(
        predictions, sp_labels, k=5, expected=1.0 / 1, class_id=5)
    self._test_precision_at_top_k(
        predictions_idx, sp_labels, k=5, expected=1.0 / 1, class_id=5)

    # Class 7: 1 label, 1 incorrect prediction.
    self._test_precision_at_k(
        predictions, sp_labels, k=5, expected=0.0 / 1, class_id=7)
    self._test_precision_at_top_k(
        predictions_idx, sp_labels, k=5, expected=0.0 / 1, class_id=7)

    # All classes: 10 predictions, 3 correct.
    self._test_precision_at_k(
        predictions, sp_labels, k=5, expected=3.0 / 10)
    self._test_precision_at_top_k(
        predictions_idx, sp_labels, k=5, expected=3.0 / 10)

  @test_util.run_deprecated_v1
  def test_3d_nan(self):
    # [FUNCTION DOC]
    # Scenario: three-dimensional predictions (two by two by ten) and sparse labels, k equals five.
    # Action: classes -1, 1, 3, 8 and 10, which have no predictions or are out of range. Expectation:
    # precision is not-a-number for each, for both precision at k and at top k.
    predictions = [[[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                    [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]],
                   [[0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
                    [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]]]
    predictions_idx = [[[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]],
                       [[5, 7, 2, 9, 6], [9, 4, 6, 2, 0]]]
    labels = _binary_3d_label_to_sparse_value(
        [[[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]],
         [[0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]])

    # Classes 1,3,8 have 0 predictions, classes -1 and 10 are out of range.
    for class_id in (-1, 1, 3, 8, 10):
      self._test_precision_at_k(
          predictions, labels, k=5, expected=NAN, class_id=class_id)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=NAN, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_3d_no_labels(self):
    # [FUNCTION DOC]
    # Scenario: same three-dimensional data; classes 0, 4, 6 and 9 are predicted but have no labels.
    # Expectation: precision is exactly zero.
    predictions = [[[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                    [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]],
                   [[0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
                    [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]]]
    predictions_idx = [[[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]],
                       [[5, 7, 2, 9, 6], [9, 4, 6, 2, 0]]]
    labels = _binary_3d_label_to_sparse_value(
        [[[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]],
         [[0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]])

    # Classes 0,4,6,9: 0 labels, >=1 prediction.
    for class_id in (0, 4, 6, 9):
      self._test_precision_at_k(
          predictions, labels, k=5, expected=0.0, class_id=class_id)
      self._test_precision_at_top_k(
          predictions_idx, labels, k=5, expected=0.0, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_3d(self):
    # [FUNCTION DOC]
    # Scenario: three-dimensional data. Expectations: class two 4/4, class five 2/2, class seven 1/2, all
    # classes 7/20, for both precision at k and precision at top k.
    predictions = [[[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                    [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]],
                   [[0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
                    [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]]]
    predictions_idx = [[[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]],
                       [[5, 7, 2, 9, 6], [9, 4, 6, 2, 0]]]
    labels = _binary_3d_label_to_sparse_value(
        [[[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]],
         [[0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]])

    # Class 2: 4 predictions, all correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=4.0 / 4, class_id=2)
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=4.0 / 4, class_id=2)

    # Class 5: 2 predictions, both correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=2.0 / 2, class_id=5)
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=2.0 / 2, class_id=5)

    # Class 7: 2 predictions, 1 correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=1.0 / 2, class_id=7)
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=1.0 / 2, class_id=7)

    # All classes: 20 predictions, 7 correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=7.0 / 20)
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=7.0 / 20)

  @test_util.run_deprecated_v1
  def test_3d_ignore_some(self):
    # [FUNCTION DOC]
    # Scenario: three-dimensional data with weights that zero out some examples or positions.
    # Expectations: for class two, weights selecting either example give 2/2; for class seven weights give
    # 0/1, 1/1, not-a-number (no predictions), or 1/2 depending on which positions are kept. Both the dense
    # score and top-index variants are checked.
    predictions = [[[0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
                    [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]],
                   [[0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
                    [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]]]
    predictions_idx = [[[9, 4, 6, 2, 0], [5, 7, 2, 9, 6]],
                       [[5, 7, 2, 9, 6], [9, 4, 6, 2, 0]]]
    labels = _binary_3d_label_to_sparse_value(
        [[[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]],
         [[0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]]])

    # Class 2: 2 predictions, both correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[1], [0]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[1], [0]])

    # Class 2: 2 predictions, both correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[0], [1]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[0], [1]])

    # Class 7: 1 incorrect prediction.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=0.0 / 1.0, class_id=7,
        weights=[[1], [0]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=0.0 / 1.0, class_id=7,
        weights=[[1], [0]])

    # Class 7: 1 correct prediction.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=1.0 / 1.0, class_id=7,
        weights=[[0], [1]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=1.0 / 1.0, class_id=7,
        weights=[[0], [1]])

    # Class 7: no predictions.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=NAN, class_id=7,
        weights=[[1, 0], [0, 1]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=NAN, class_id=7,
        weights=[[1, 0], [0, 1]])

    # Class 7: 2 predictions, 1 correct.
    self._test_precision_at_k(
        predictions, labels, k=5, expected=1.0 / 2.0, class_id=7,
        weights=[[0, 1], [1, 0]])
    self._test_precision_at_top_k(
        predictions_idx, labels, k=5, expected=1.0 / 2.0, class_id=7,
        weights=[[0, 1], [1, 0]])


def _test_recall_at_k(predictions,
                      labels,
                      k,
                      expected,
                      class_id=None,
                      weights=None,
                      test_case=None):
  # [FUNCTION DOC]
  # HELPER: shared driver for the recall-at-k tests (wraps the metric over dense scores).
  # Inputs: predictions, nested list of float scores; labels, dense or sparse; k; expected number
  # (not-a-number allowed); class_id, optional class; weights, optional (converted to float constants);
  # test_case, the running test case. Builds recall at k in a fresh graph and session, asserts that
  # reading the value or running the update before variable initialization raises an op error, then
  # initializes the local variables and compares update result and value to the expected number
  # (exactly, or a not-a-number check). Returns nothing; failures are assertion errors.
  with ops.Graph().as_default() as g, test_case.test_session(g):
    if weights is not None:
      weights = constant_op.constant(weights, dtypes_lib.float32)
    metric, update = metrics.recall_at_k(
        predictions=constant_op.constant(predictions, dtypes_lib.float32),
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)

    # Fails without initialized vars.
    test_case.assertRaises(errors_impl.OpError, metric.eval)
    test_case.assertRaises(errors_impl.OpError, update.eval)
    variables.variables_initializer(variables.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      _assert_nan(test_case, update.eval())
      _assert_nan(test_case, metric.eval())
    else:
      test_case.assertEqual(expected, update.eval())
      test_case.assertEqual(expected, metric.eval())


def _test_recall_at_top_k(
    predictions_idx,
    labels,
    expected,
    k=None,
    class_id=None,
    weights=None,
    test_case=None):
  # [FUNCTION DOC]
  # HELPER: shared driver for the recall-at-top-k tests (takes precomputed top-k class indices).
  # Same procedure as the recall-at-k driver but the metric is built from predictions_idx, a nested
  # list of integer class indices (converted to int32), plus labels, k (optional), class_id (optional),
  # weights (optional) and test_case. Asserts that evaluation before variable initialization raises
  # an op error, then that update result and value equal expected, or are both not-a-number when
  # expected is. Returns nothing.
  with ops.Graph().as_default() as g, test_case.test_session(g):
    if weights is not None:
      weights = constant_op.constant(weights, dtypes_lib.float32)
    metric, update = metrics.recall_at_top_k(
        predictions_idx=constant_op.constant(predictions_idx, dtypes_lib.int32),
        labels=labels,
        k=k,
        class_id=class_id,
        weights=weights)

    # Fails without initialized vars.
    test_case.assertRaises(errors_impl.OpError, metric.eval)
    test_case.assertRaises(errors_impl.OpError, update.eval)
    variables.variables_initializer(variables.local_variables()).run()

    # Run per-step op and assert expected values.
    if math.isnan(expected):
      _assert_nan(test_case, update.eval())
      _assert_nan(test_case, metric.eval())
    else:
      test_case.assertEqual(expected, update.eval())
      test_case.assertEqual(expected, metric.eval())


class SingleLabelRecallAtKTest(test.TestCase):
  # [CLASS DOC]
  # Tests recall at k and recall at top k with exactly one label per example, supplied in four equivalent
  # forms (sparse rank one, sparse rank two, dense vector, dense column) that must give identical results.
  # Covers nan cases, no-prediction cases and per-example weights.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Creates two examples with four class scores each, their precomputed top-one indices, and
    # the four equivalent label forms. Stores partially applied versions of the recall-at-k and
    # recall-at-top-k drivers bound to this test case. Does not reset the graph. Returns nothing.
    self._predictions = ((0.1, 0.3, 0.2, 0.4), (0.1, 0.2, 0.3, 0.4))
    self._predictions_idx = [[3], [3]]
    indicator_labels = ((0, 0, 0, 1), (0, 0, 1, 0))
    class_labels = (3, 2)
    # Sparse vs dense, and 1d vs 2d labels should all be handled the same.
    self._labels = (
        _binary_2d_label_to_1d_sparse_value(indicator_labels),
        _binary_2d_label_to_2d_sparse_value(indicator_labels), np.array(
            class_labels, dtype=np.int64), np.array(
                [[class_id] for class_id in class_labels], dtype=np.int64))
    self._test_recall_at_k = functools.partial(
        _test_recall_at_k, test_case=self)
    self._test_recall_at_top_k = functools.partial(
        _test_recall_at_top_k, test_case=self)

  @test_util.run_deprecated_v1
  def test_at_k1_nan(self):
    # [FUNCTION DOC]
    # Scenario: classes 0 and 1 have no labels and no predictions, classes -1 and 4 are out of range.
    # Expectation: for every label form and these class ids, recall at k and at top k are not-a-number.
    # Classes 0,1 have 0 labels, 0 predictions, classes -1 and 4 are out of
    # range.
    for labels in self._labels:
      for class_id in (-1, 0, 1, 4):
        self._test_recall_at_k(
            self._predictions, labels, k=1, expected=NAN, class_id=class_id)
        self._test_recall_at_top_k(
            self._predictions_idx, labels, k=1, expected=NAN, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_at_k1_no_predictions(self):
    # [FUNCTION DOC]
    # Scenario: class two has a label in the second example but is never predicted at k equals one.
    # Expectation: recall is exactly zero for every label form.
    for labels in self._labels:
      # Class 2: 0 predictions.
      self._test_recall_at_k(
          self._predictions, labels, k=1, expected=0.0, class_id=2)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=1, expected=0.0, class_id=2)

  @test_util.run_deprecated_v1
  def test_one_label_at_k1(self):
    # [FUNCTION DOC]
    # Scenario: one label per example, k equals one. Expectations: class three gives recall 1/1; all
    # classes give 1/2; for both recall at k and recall at top k, for every label form.
    for labels in self._labels:
      # Class 3: 1 label, 2 predictions, 1 correct.
      self._test_recall_at_k(
          self._predictions, labels, k=1, expected=1.0 / 1, class_id=3)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=1, expected=1.0 / 1, class_id=3)

      # All classes: 2 labels, 2 predictions, 1 correct.
      self._test_recall_at_k(self._predictions, labels, k=1, expected=1.0 / 2)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=1, expected=1.0 / 2)

  @test_util.run_deprecated_v1
  def test_one_label_at_k1_weighted_class_id3(self):
    # [FUNCTION DOC]
    # Scenario: class three with various weights, scalar or per example. Expectations: weight zero gives
    # not-a-number (no weighted labels), weights one or two give recall one, weights (0,1) give
    # not-a-number, (1,0) give one, and (2,3) give one. Checked for both metric variants and all label forms.
    predictions = self._predictions
    predictions_idx = self._predictions_idx
    for labels in self._labels:
      # Class 3: 1 label, 2 predictions, 1 correct.
      self._test_recall_at_k(
          predictions, labels, k=1, expected=NAN, class_id=3, weights=(0.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=NAN, class_id=3,
          weights=(0.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(2.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(2.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=NAN, class_id=3,
          weights=(0.0, 1.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=NAN, class_id=3,
          weights=(0.0, 1.0))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0, 0.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0, 0.0))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=2.0 / 2, class_id=3,
          weights=(2.0, 3.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=2.0 / 2, class_id=3,
          weights=(2.0, 3.0))

  @test_util.run_deprecated_v1
  def test_one_label_at_k1_weighted(self):
    # [FUNCTION DOC]
    # Scenario: all classes with various weights. Expectations: weight zero gives not-a-number; weights
    # one or two give 1/2; weights (1,0) give 1/1; (0,1) give 0/1; (2,3) give 2/5. Checked for both metric
    # variants and all label forms.
    predictions = self._predictions
    predictions_idx = self._predictions_idx
    for labels in self._labels:
      # All classes: 2 labels, 2 predictions, 1 correct.
      self._test_recall_at_k(
          predictions, labels, k=1, expected=NAN, weights=(0.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=NAN, weights=(0.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 2, weights=(1.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 2, weights=(1.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 2, weights=(2.0,))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 2, weights=(2.0,))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, weights=(1.0, 0.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=1.0 / 1, weights=(1.0, 0.0))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=0.0 / 1, weights=(0.0, 1.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=0.0 / 1, weights=(0.0, 1.0))
      self._test_recall_at_k(
          predictions, labels, k=1, expected=2.0 / 5, weights=(2.0, 3.0))
      self._test_recall_at_top_k(
          predictions_idx, labels, k=1, expected=2.0 / 5, weights=(2.0, 3.0))


class MultiLabel2dRecallAtKTest(test.TestCase):
  # [CLASS DOC]
  # Tests recall at k and recall at top k on two-dimensional multi-label data (two examples over ten
  # classes) with sparse and dense labels.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Builds two examples of ten scores, precomputed top-five class indices, and labels in two
    # equivalent forms (sparse from indicator rows, and dense class ids). Stores partially applied
    # versions of the recall-at-k and recall-at-top-k drivers bound to this test case. Does not reset
    # the graph. Returns nothing.
    self._predictions = ((0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9),
                         (0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6))
    self._predictions_idx = ((9, 4, 6, 2, 0), (5, 7, 2, 9, 6))
    indicator_labels = ((0, 0, 1, 0, 0, 0, 0, 1, 1, 0),
                        (0, 1, 1, 0, 0, 1, 0, 0, 0, 0))
    class_labels = ((2, 7, 8), (1, 2, 5))
    # Sparse vs dense labels should be handled the same.
    self._labels = (_binary_2d_label_to_2d_sparse_value(indicator_labels),
                    np.array(
                        class_labels, dtype=np.int64))
    self._test_recall_at_k = functools.partial(
        _test_recall_at_k, test_case=self)
    self._test_recall_at_top_k = functools.partial(
        _test_recall_at_top_k, test_case=self)

  @test_util.run_deprecated_v1
  def test_at_k5_nan(self):
    # [FUNCTION DOC]
    # Scenario: classes 0, 3, 4, 6, 9 have no labels and class ten is out of range, k equals five.
    # Expectation: recall is not-a-number for both metric variants and both label forms.
    for labels in self._labels:
      # Classes 0,3,4,6,9 have 0 labels, class 10 is out of range.
      for class_id in (0, 3, 4, 6, 9, 10):
        self._test_recall_at_k(
            self._predictions, labels, k=5, expected=NAN, class_id=class_id)
        self._test_recall_at_top_k(
            self._predictions_idx, labels, k=5, expected=NAN, class_id=class_id)

  @test_util.run_deprecated_v1
  def test_at_k5_no_predictions(self):
    # [FUNCTION DOC]
    # Scenario: class eight has one label but is never in the top five. Expectation: recall is 0/1, that is zero.
    for labels in self._labels:
      # Class 8: 1 label, no predictions.
      self._test_recall_at_k(
          self._predictions, labels, k=5, expected=0.0 / 1, class_id=8)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=5, expected=0.0 / 1, class_id=8)

  @test_util.run_deprecated_v1
  def test_at_k5(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the in-code comments for class five and class seven say "1 label, incorrect", but
    # the expected recall for class five is 1/1, meaning its label is predicted correctly.
    # Scenario: k equals five. Expectations: class two 2/2, class five 1/1, class seven 0/1, all classes
    # 3/6, for both metric variants and both label forms.
    for labels in self._labels:
      # Class 2: 2 labels, both correct.
      self._test_recall_at_k(
          self._predictions, labels, k=5, expected=2.0 / 2, class_id=2)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=5, expected=2.0 / 2, class_id=2)

      # Class 5: 1 label, incorrect.
      self._test_recall_at_k(
          self._predictions, labels, k=5, expected=1.0 / 1, class_id=5)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=5, expected=1.0 / 1, class_id=5)

      # Class 7: 1 label, incorrect.
      self._test_recall_at_k(
          self._predictions, labels, k=5, expected=0.0 / 1, class_id=7)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=5, expected=0.0 / 1, class_id=7)

      # All classes: 6 labels, 3 correct.
      self._test_recall_at_k(self._predictions, labels, k=5, expected=3.0 / 6)
      self._test_recall_at_top_k(
          self._predictions_idx, labels, k=5, expected=3.0 / 6)

  @test_util.run_deprecated_v1
  def test_at_k5_some_out_of_range(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment for class five says "1 label, incorrect" although the expected recall
    # is 1/1 (a correct prediction).
    # Scenario: sparse labels containing minus one and ten, outside the valid range. Expectation: they
    # still count in the denominator for the all-classes result: class two 2/2, class five 1/1, class
    # seven 0/1, all classes 3/8.
    """Tests that labels outside the [0, n_classes) count in denominator."""
    labels = sparse_tensor.SparseTensorValue(
        indices=[[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2],
                 [1, 3]],
        # values -1 and 10 are outside the [0, n_classes) range.
        values=np.array([2, 7, -1, 8, 1, 2, 5, 10], np.int64),
        dense_shape=[2, 4])

    # Class 2: 2 labels, both correct.
    self._test_recall_at_k(
        self._predictions, labels, k=5, expected=2.0 / 2, class_id=2)
    self._test_recall_at_top_k(
        self._predictions_idx, labels, k=5, expected=2.0 / 2, class_id=2)

    # Class 5: 1 label, incorrect.
    self._test_recall_at_k(
        self._predictions, labels, k=5, expected=1.0 / 1, class_id=5)
    self._test_recall_at_top_k(
        self._predictions_idx, labels, k=5, expected=1.0 / 1, class_id=5)

    # Class 7: 1 label, incorrect.
    self._test_recall_at_k(
        self._predictions, labels, k=5, expected=0.0 / 1, class_id=7)
    self._test_recall_at_top_k(
        self._predictions_idx, labels, k=5, expected=0.0 / 1, class_id=7)

    # All classes: 8 labels, 3 correct.
    self._test_recall_at_k(self._predictions, labels, k=5, expected=3.0 / 8)
    self._test_recall_at_top_k(
        self._predictions_idx, labels, k=5, expected=3.0 / 8)


class MultiLabel3dRecallAtKTest(test.TestCase):
  # [CLASS DOC]
  # Tests recall at k and recall at top k on three-dimensional multi-label data (two by two examples
  # over ten classes) with sparse labels only, including nan cases and weights that ignore examples.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Builds three-dimensional scores, precomputed top-five indices and sparse labels from a
    # three-dimensional indicator (dense labels are not used because examples have different label
    # counts). Stores partially applied recall-at-k and recall-at-top-k drivers bound to this test case.
    # Does not reset the graph. Returns nothing.
    self._predictions = (((0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9),
                          (0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6)),
                         ((0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6),
                          (0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9)))
    self._predictions_idx = (((9, 4, 6, 2, 0), (5, 7, 2, 9, 6)),
                             ((5, 7, 2, 9, 6), (9, 4, 6, 2, 0)))
    # Note: We don't test dense labels here, since examples have different
    # numbers of labels.
    self._labels = _binary_3d_label_to_sparse_value(((
        (0, 0, 1, 0, 0, 0, 0, 1, 1, 0), (0, 1, 1, 0, 0, 1, 0, 0, 0, 0)), (
            (0, 1, 1, 0, 0, 1, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0, 1, 0))))
    self._test_recall_at_k = functools.partial(
        _test_recall_at_k, test_case=self)
    self._test_recall_at_top_k = functools.partial(
        _test_recall_at_top_k, test_case=self)

  @test_util.run_deprecated_v1
  def test_3d_nan(self):
    # [FUNCTION DOC]
    # Scenario: classes 0, 3, 4, 6, 9 have no labels, class ten is out of range. Expectation: recall is
    # not-a-number for both metric variants.
    # Classes 0,3,4,6,9 have 0 labels, class 10 is out of range.
    for class_id in (0, 3, 4, 6, 9, 10):
      self._test_recall_at_k(
          self._predictions, self._labels, k=5, expected=NAN, class_id=class_id)
      self._test_recall_at_top_k(
          self._predictions_idx, self._labels, k=5, expected=NAN,
          class_id=class_id)

  @test_util.run_deprecated_v1
  def test_3d_no_predictions(self):
    # [FUNCTION DOC]
    # Scenario: classes one and eight have labels but are never in the top five. Expectation: recall is zero.
    # Classes 1,8 have 0 predictions, >=1 label.
    for class_id in (1, 8):
      self._test_recall_at_k(
          self._predictions, self._labels, k=5, expected=0.0, class_id=class_id)
      self._test_recall_at_top_k(
          self._predictions_idx, self._labels, k=5, expected=0.0,
          class_id=class_id)

  @test_util.run_deprecated_v1
  def test_3d(self):
    # [FUNCTION DOC]
    # Expectations: class two 4/4, class five 2/2, class seven 1/2, all classes 7/12, for both metric variants.
    # Class 2: 4 labels, all correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=4.0 / 4, class_id=2)
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=4.0 / 4,
        class_id=2)

    # Class 5: 2 labels, both correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=2.0 / 2, class_id=5)
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=2.0 / 2,
        class_id=5)

    # Class 7: 2 labels, 1 incorrect.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=1.0 / 2, class_id=7)
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=1.0 / 2,
        class_id=7)

    # All classes: 12 labels, 7 correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=7.0 / 12)
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=7.0 / 12)

  @test_util.run_deprecated_v1
  def test_3d_ignore_all(self):
    # [FUNCTION DOC]
    # Scenario: weights that are all zero (shapes two by one and two by two), for every class from zero to
    # nine and for the all-classes metric. Expectation: recall is not-a-number everywhere.
    for class_id in range(10):
      self._test_recall_at_k(
          self._predictions, self._labels, k=5, expected=NAN, class_id=class_id,
          weights=[[0], [0]])
      self._test_recall_at_top_k(
          self._predictions_idx, self._labels, k=5, expected=NAN,
          class_id=class_id, weights=[[0], [0]])
      self._test_recall_at_k(
          self._predictions, self._labels, k=5, expected=NAN, class_id=class_id,
          weights=[[0, 0], [0, 0]])
      self._test_recall_at_top_k(
          self._predictions_idx, self._labels, k=5, expected=NAN,
          class_id=class_id, weights=[[0, 0], [0, 0]])
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=NAN, weights=[[0], [0]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=NAN,
        weights=[[0], [0]])
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=NAN,
        weights=[[0, 0], [0, 0]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=NAN,
        weights=[[0, 0], [0, 0]])

  @test_util.run_deprecated_v1
  def test_3d_ignore_some(self):
    # [FUNCTION DOC]
    # Scenario: weights that zero out part of the data. Expectations: class two 2/2 with either example
    # kept; class seven 1/1, 0/1 or 1/2 depending on the kept positions, and not-a-number when none of its
    # labels are kept. Both metric variants are checked.
    # Class 2: 2 labels, both correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[1], [0]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=2.0 / 2.0,
        class_id=2, weights=[[1], [0]])

    # Class 2: 2 labels, both correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[0], [1]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=2.0 / 2.0,
        class_id=2, weights=[[0], [1]])

    # Class 7: 1 label, correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=1.0 / 1.0, class_id=7,
        weights=[[0], [1]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=1.0 / 1.0,
        class_id=7, weights=[[0], [1]])

    # Class 7: 1 label, incorrect.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=0.0 / 1.0, class_id=7,
        weights=[[1], [0]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=0.0 / 1.0,
        class_id=7, weights=[[1], [0]])

    # Class 7: 2 labels, 1 correct.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=1.0 / 2.0, class_id=7,
        weights=[[1, 0], [1, 0]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=1.0 / 2.0,
        class_id=7, weights=[[1, 0], [1, 0]])

    # Class 7: No labels.
    self._test_recall_at_k(
        self._predictions, self._labels, k=5, expected=NAN, class_id=7,
        weights=[[0, 1], [0, 1]])
    self._test_recall_at_top_k(
        self._predictions_idx, self._labels, k=5, expected=NAN, class_id=7,
        weights=[[0, 1], [0, 1]])


class MeanAbsoluteErrorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the mean absolute error metric: variables, collections, idempotent reads and a weighted
  # single update. Fixture: every test resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Takes no arguments and returns nothing.
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the mean absolute error metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the count and total accumulators under the mean-absolute-error scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.mean_absolute_error(
        predictions=array_ops.ones((10, 1)), labels=array_ops.ones((10, 1)))
    _assert_metric_variables(
        self, ('mean_absolute_error/count:0', 'mean_absolute_error/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the mean absolute error metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean_absolute_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the mean absolute error metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_absolute_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the mean absolute error metric repeatedly does not change it.
    # Inputs: random normal predictions and labels of shape ten by three, generated with two different fixed seeds.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_normal((10, 3), seed=1)
    labels = random_ops.random_normal((10, 3), seed=2)
    error, update_op = metrics.mean_absolute_error(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_error = self.evaluate(error)
      for _ in range(10):
        self.assertEqual(initial_error, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testSingleUpdateWithErrorAndWeights(self):
    # [FUNCTION DOC]
    # Scenario: predictions 2,4,6,8, labels 1,3,2,3 and weights 0,1,0,1 so only positions two and four
    # count. Expectation: the mean absolute error is exactly three for update and value.
    predictions = constant_op.constant(
        [2, 4, 6, 8], shape=(1, 4), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        [1, 3, 2, 3], shape=(1, 4), dtype=dtypes_lib.float32)
    weights = constant_op.constant([0, 1, 0, 1], shape=(1, 4))

    error, update_op = metrics.mean_absolute_error(labels, predictions, weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(3, self.evaluate(update_op))
      self.assertEqual(3, self.evaluate(error))


class MeanRelativeErrorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the mean relative error metric (absolute error divided by a normalizer): variables,
  # collections, idempotent reads and normalizations by the labels and by zeros. Fixture: every test
  # resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Takes no arguments and returns nothing.
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the mean relative error metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the count and total accumulators under the mean-relative-error scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.mean_relative_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        normalizer=array_ops.ones((10, 1)))
    _assert_metric_variables(
        self, ('mean_relative_error/count:0', 'mean_relative_error/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the mean relative error metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean_relative_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        normalizer=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the mean relative error metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_relative_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        normalizer=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the mean relative error metric repeatedly does not change it.
    # Inputs: random normal predictions, labels and normalizer of shape ten by three, generated with three different fixed seeds.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_normal((10, 3), seed=1)
    labels = random_ops.random_normal((10, 3), seed=2)
    normalizer = random_ops.random_normal((10, 3), seed=3)
    error, update_op = metrics.mean_relative_error(labels, predictions,
                                                   normalizer)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_error = self.evaluate(error)
      for _ in range(10):
        self.assertEqual(initial_error, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testSingleUpdateNormalizedByLabels(self):
    # [FUNCTION DOC]
    # Scenario: predictions 2,4,6,8 and labels 1,3,2,3 with the labels as the normalizer. Expectation:
    # the metric equals the numpy mean of absolute differences divided by the labels, exactly, for update
    # and value.
    np_predictions = np.asarray([2, 4, 6, 8], dtype=np.float32)
    np_labels = np.asarray([1, 3, 2, 3], dtype=np.float32)
    expected_error = np.mean(
        np.divide(np.absolute(np_predictions - np_labels), np_labels))

    predictions = constant_op.constant(
        np_predictions, shape=(1, 4), dtype=dtypes_lib.float32)
    labels = constant_op.constant(np_labels, shape=(1, 4))

    error, update_op = metrics.mean_relative_error(
        labels, predictions, normalizer=labels)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(expected_error, self.evaluate(update_op))
      self.assertEqual(expected_error, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testSingleUpdateNormalizedByZeros(self):
    # [FUNCTION DOC]
    # Scenario: the same predictions and labels but the normalizer is all zeros. Expectation: the metric
    # is exactly zero (division by a zero normalizer is treated as zero, not infinity).
    np_predictions = np.asarray([2, 4, 6, 8], dtype=np.float32)

    predictions = constant_op.constant(
        np_predictions, shape=(1, 4), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        [1, 3, 2, 3], shape=(1, 4), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_relative_error(
        labels, predictions, normalizer=array_ops.zeros_like(labels))

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(0.0, self.evaluate(update_op))
      self.assertEqual(0.0, self.evaluate(error))


class MeanSquaredErrorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the mean squared error metric: variables, collections, idempotent reads, zero and
  # non-zero errors, weights, multiple batches and several metrics running at once.
  # Fixture: every test resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Takes no arguments and returns nothing.
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the mean squared error metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the count and total accumulators under the mean-squared-error scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.mean_squared_error(
        predictions=array_ops.ones((10, 1)), labels=array_ops.ones((10, 1)))
    _assert_metric_variables(
        self, ('mean_squared_error/count:0', 'mean_squared_error/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the mean squared error metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean_squared_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the mean squared error metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_squared_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the mean squared error metric repeatedly does not change it.
    # Inputs: random normal predictions and labels of shape ten by three, generated with two different fixed seeds.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_normal((10, 3), seed=1)
    labels = random_ops.random_normal((10, 3), seed=2)
    error, update_op = metrics.mean_squared_error(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_error = self.evaluate(error)
      for _ in range(10):
        self.assertEqual(initial_error, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testSingleUpdateZeroError(self):
    # [FUNCTION DOC]
    # Scenario: all-zero predictions and labels of shape one by three. Expectation: error is exactly zero.
    predictions = array_ops.zeros((1, 3), dtype=dtypes_lib.float32)
    labels = array_ops.zeros((1, 3), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_squared_error(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(0, self.evaluate(update_op))
      self.assertEqual(0, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testSingleUpdateWithError(self):
    # [FUNCTION DOC]
    # Scenario: predictions 2,4,6 and labels 1,3,2. Expectation: error is six (squared differences 1, 1,
    # 16, mean 6) for update and value.
    predictions = constant_op.constant(
        [2, 4, 6], shape=(1, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        [1, 3, 2], shape=(1, 3), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_squared_error(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(6, self.evaluate(update_op))
      self.assertEqual(6, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testSingleUpdateWithErrorAndWeights(self):
    # [FUNCTION DOC]
    # Scenario: predictions 2,4,6,8, labels 1,3,2,3, weights 0,1,0,1. Expectation: error is thirteen
    # (only the squared errors 1 and 25 count).
    predictions = constant_op.constant(
        [2, 4, 6, 8], shape=(1, 4), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        [1, 3, 2, 3], shape=(1, 4), dtype=dtypes_lib.float32)
    weights = constant_op.constant([0, 1, 0, 1], shape=(1, 4))

    error, update_op = metrics.mean_squared_error(labels, predictions, weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(13, self.evaluate(update_op))
      self.assertEqual(13, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testMultipleBatchesOfSizeOne(self):
    # [FUNCTION DOC]
    # Scenario: two batches fed through queues, predictions (10,8,6) then (-4,3,-1), labels (1,3,2) then
    # (2,4,6). Action: run the update twice (the first run is separate, then one checked). Expectation:
    # the accumulated error is 208/6 to five places for update result and value.
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue, [10, 8, 6])
      _enqueue_vector(sess, preds_queue, [-4, 3, -1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue, [1, 3, 2])
      _enqueue_vector(sess, labels_queue, [2, 4, 6])
      labels = labels_queue.dequeue()

      error, update_op = metrics.mean_squared_error(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate(update_op)
      self.assertAlmostEqual(208.0 / 6, self.evaluate(update_op), 5)

      self.assertAlmostEqual(208.0 / 6, self.evaluate(error), 5)

  @test_util.run_deprecated_v1
  def testMetricsComputedConcurrently(self):
    # [FUNCTION DOC]
    # Scenario: two independent mean squared error metrics with different names built on different
    # prediction and label queues. Action: run both updates together twice, then read both values at once.
    # Expectation: the first is 208/6 and the second is 79/6, each to five places.
    with self.cached_session() as sess:
      # Create the queue that populates one set of predictions.
      preds_queue0 = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue0, [10, 8, 6])
      _enqueue_vector(sess, preds_queue0, [-4, 3, -1])
      predictions0 = preds_queue0.dequeue()

      # Create the queue that populates one set of predictions.
      preds_queue1 = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue1, [0, 1, 1])
      _enqueue_vector(sess, preds_queue1, [1, 1, 0])
      predictions1 = preds_queue1.dequeue()

      # Create the queue that populates one set of labels.
      labels_queue0 = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue0, [1, 3, 2])
      _enqueue_vector(sess, labels_queue0, [2, 4, 6])
      labels0 = labels_queue0.dequeue()

      # Create the queue that populates another set of labels.
      labels_queue1 = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue1, [-5, -3, -1])
      _enqueue_vector(sess, labels_queue1, [5, 4, 3])
      labels1 = labels_queue1.dequeue()

      mse0, update_op0 = metrics.mean_squared_error(
          labels0, predictions0, name='msd0')
      mse1, update_op1 = metrics.mean_squared_error(
          labels1, predictions1, name='msd1')

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([update_op0, update_op1])
      self.evaluate([update_op0, update_op1])

      mse0, mse1 = self.evaluate([mse0, mse1])
      self.assertAlmostEqual(208.0 / 6, mse0, 5)
      self.assertAlmostEqual(79.0 / 6, mse1, 5)

  @test_util.run_deprecated_v1
  def testMultipleMetricsOnMultipleBatchesOfSizeOne(self):
    # [FUNCTION DOC]
    # Scenario: mean absolute error and mean squared error computed on the same queued data. Action: run
    # both updates twice. Expectation: absolute error is 32/6 and squared error is 208/6, to five places.
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue, [10, 8, 6])
      _enqueue_vector(sess, preds_queue, [-4, 3, -1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          2, dtypes=dtypes_lib.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue, [1, 3, 2])
      _enqueue_vector(sess, labels_queue, [2, 4, 6])
      labels = labels_queue.dequeue()

      mae, ma_update_op = metrics.mean_absolute_error(labels, predictions)
      mse, ms_update_op = metrics.mean_squared_error(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([ma_update_op, ms_update_op])
      self.evaluate([ma_update_op, ms_update_op])

      self.assertAlmostEqual(32.0 / 6, self.evaluate(mae), 5)
      self.assertAlmostEqual(208.0 / 6, self.evaluate(mse), 5)


class RootMeanSquaredErrorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the root mean squared error metric: variables, collections, idempotent reads, zero and
  # non-zero errors, and weights. Fixture: every test resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Takes no arguments and returns nothing.
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the root mean squared error metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the count and total accumulators under the root-mean-squared-error scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.root_mean_squared_error(
        predictions=array_ops.ones((10, 1)), labels=array_ops.ones((10, 1)))
    _assert_metric_variables(
        self,
        ('root_mean_squared_error/count:0', 'root_mean_squared_error/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the root mean squared error metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.root_mean_squared_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the root mean squared error metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.root_mean_squared_error(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the root mean squared error metric repeatedly does not change it.
    # Inputs: random normal predictions and labels of shape ten by three, generated with two different fixed seeds.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_normal((10, 3), seed=1)
    labels = random_ops.random_normal((10, 3), seed=2)
    error, update_op = metrics.root_mean_squared_error(labels, predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_error = self.evaluate(error)
      for _ in range(10):
        self.assertEqual(initial_error, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testSingleUpdateZeroError(self):
    # [FUNCTION DOC]
    # Scenario: all-zero predictions and labels. Expectation: error is exactly zero for update and value.
    with self.cached_session():
      predictions = constant_op.constant(
          0.0, shape=(1, 3), dtype=dtypes_lib.float32)
      labels = constant_op.constant(0.0, shape=(1, 3), dtype=dtypes_lib.float32)

      rmse, update_op = metrics.root_mean_squared_error(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(0, self.evaluate(update_op))

      self.assertEqual(0, self.evaluate(rmse))

  @test_util.run_deprecated_v1
  def testSingleUpdateWithError(self):
    # [FUNCTION DOC]
    # Scenario: predictions 2,4,6 and labels 1,3,2. Expectation: error is the square root of six, to five
    # places, for update and value.
    with self.cached_session():
      predictions = constant_op.constant(
          [2, 4, 6], shape=(1, 3), dtype=dtypes_lib.float32)
      labels = constant_op.constant(
          [1, 3, 2], shape=(1, 3), dtype=dtypes_lib.float32)

      rmse, update_op = metrics.root_mean_squared_error(labels, predictions)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(math.sqrt(6), self.evaluate(update_op), 5)
      self.assertAlmostEqual(math.sqrt(6), self.evaluate(rmse), 5)

  @test_util.run_deprecated_v1
  def testSingleUpdateWithErrorAndWeights(self):
    # [FUNCTION DOC]
    # Scenario: predictions 2,4,6,8, labels 1,3,2,3 and weights 0,1,0,1. Expectation: error is the square
    # root of thirteen (the update result is compared with the default precision, the value to five places).
    with self.cached_session():
      predictions = constant_op.constant(
          [2, 4, 6, 8], shape=(1, 4), dtype=dtypes_lib.float32)
      labels = constant_op.constant(
          [1, 3, 2, 3], shape=(1, 4), dtype=dtypes_lib.float32)
      weights = constant_op.constant([0, 1, 0, 1], shape=(1, 4))

      rmse, update_op = metrics.root_mean_squared_error(labels, predictions,
                                                        weights)

      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(math.sqrt(13), self.evaluate(update_op))

      self.assertAlmostEqual(math.sqrt(13), self.evaluate(rmse), 5)


def _reweight(predictions, labels, weights):
  # [FUNCTION DOC]
  # HELPER: unused test-data utility.
  # KNOWN ISSUE: this function is never called anywhere in the module (dead code).
  # Intended to replicate each prediction and label as many times as its integer weight. Inputs:
  # predictions, labels, weights, equal-length sequences; weights are truncated to integers. Output:
  # a pair of concatenated numpy arrays, one of repeated predictions and one of repeated labels.
  return (np.concatenate([[p] * int(w) for p, w in zip(predictions, weights)]),
          np.concatenate([[l] * int(w) for l, w in zip(labels, weights)]))


class MeanCosineDistanceTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the mean cosine distance metric (one minus cosine similarity along a given dimension):
  # variables, collections, idempotent reads, zero error, errors with and without weights.
  # Fixture: every test resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Takes no arguments and returns nothing.
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the mean cosine distance metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the count and total accumulators under the mean-cosine-distance scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.mean_cosine_distance(
        predictions=array_ops.ones((10, 3)),
        labels=array_ops.ones((10, 3)),
        dim=1)
    _assert_metric_variables(self, (
        'mean_cosine_distance/count:0',
        'mean_cosine_distance/total:0',
    ))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the mean cosine distance metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.mean_cosine_distance(
        predictions=array_ops.ones((10, 3)),
        labels=array_ops.ones((10, 3)),
        dim=1,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the mean cosine distance metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_cosine_distance(
        predictions=array_ops.ones((10, 3)),
        labels=array_ops.ones((10, 3)),
        dim=1,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the mean cosine distance metric repeatedly does not change it.
    # Inputs: random normal predictions and labels of shape ten by three, generated with two different fixed seeds, using dimension one.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    predictions = random_ops.random_normal((10, 3), seed=1)
    labels = random_ops.random_normal((10, 3), seed=2)
    error, update_op = metrics.mean_cosine_distance(labels, predictions, dim=1)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_error = self.evaluate(error)
      for _ in range(10):
        self.assertEqual(initial_error, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testSingleUpdateZeroError(self):
    # [FUNCTION DOC]
    # Scenario: labels equal predictions (unit vectors from a three by three matrix reshaped to one by
    # three by three), dimension two. Expectation: distance is exactly zero.
    np_labels = np.matrix(('1 0 0;' '0 0 1;' '0 1 0'))

    predictions = constant_op.constant(
        np_labels, shape=(1, 3, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        np_labels, shape=(1, 3, 3), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_cosine_distance(labels, predictions, dim=2)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(0, self.evaluate(update_op))
      self.assertEqual(0, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testSingleUpdateWithError1(self):
    # [FUNCTION DOC]
    # Scenario: three unit-vector pairs reshaped as three by one by three where the first pair agrees, the
    # second is opposite and the third is orthogonal. Expectation: mean distance (0, 2, 1) is one, to five places.
    np_labels = np.matrix(('1 0 0;' '0 0 1;' '0 1 0'))
    np_predictions = np.matrix(('1 0 0;' '0 0 -1;' '1 0 0'))

    predictions = constant_op.constant(
        np_predictions, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        np_labels, shape=(3, 1, 3), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_cosine_distance(labels, predictions, dim=2)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1, self.evaluate(update_op), 5)
      self.assertAlmostEqual(1, self.evaluate(error), 5)

  @test_util.run_deprecated_v1
  def testSingleUpdateWithError2(self):
    # [FUNCTION DOC]
    # Scenario: normalized vectors where one pair agrees, one is opposite and one is orthogonal, using
    # non-trivial decimal components. Expectation: mean distance is 1.0 to five places.
    np_predictions = np.matrix(
        ('0.819031913261206 0.567041924552012 0.087465312324590;'
         '-0.665139432070255 -0.739487441769973 -0.103671883216994;'
         '0.707106781186548 -0.707106781186548 0'))
    np_labels = np.matrix(
        ('0.819031913261206 0.567041924552012 0.087465312324590;'
         '0.665139432070255 0.739487441769973 0.103671883216994;'
         '0.707106781186548 0.707106781186548 0'))

    predictions = constant_op.constant(
        np_predictions, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        np_labels, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    error, update_op = metrics.mean_cosine_distance(labels, predictions, dim=2)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAlmostEqual(1.0, self.evaluate(update_op), 5)
      self.assertAlmostEqual(1.0, self.evaluate(error), 5)

  @test_util.run_deprecated_v1
  def testSingleUpdateWithErrorAndWeights1(self):
    # [FUNCTION DOC]
    # Scenario: the three pairs of the first error test with weights one, zero, zero. Only the agreeing
    # pair counts. Expectation: distance is exactly zero.
    np_predictions = np.matrix(('1 0 0;' '0 0 -1;' '1 0 0'))
    np_labels = np.matrix(('1 0 0;' '0 0 1;' '0 1 0'))

    predictions = constant_op.constant(
        np_predictions, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        np_labels, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    weights = constant_op.constant(
        [1, 0, 0], shape=(3, 1, 1), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_cosine_distance(
        labels, predictions, dim=2, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(0, self.evaluate(update_op))
      self.assertEqual(0, self.evaluate(error))

  @test_util.run_deprecated_v1
  def testSingleUpdateWithErrorAndWeights2(self):
    # [FUNCTION DOC]
    # Scenario: same pairs with weights zero, one, one; only the opposite pair (distance two) and the
    # orthogonal pair (distance one) count. Expectation: distance is exactly 1.5.
    np_predictions = np.matrix(('1 0 0;' '0 0 -1;' '1 0 0'))
    np_labels = np.matrix(('1 0 0;' '0 0 1;' '0 1 0'))

    predictions = constant_op.constant(
        np_predictions, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    labels = constant_op.constant(
        np_labels, shape=(3, 1, 3), dtype=dtypes_lib.float32)
    weights = constant_op.constant(
        [0, 1, 1], shape=(3, 1, 1), dtype=dtypes_lib.float32)

    error, update_op = metrics.mean_cosine_distance(
        labels, predictions, dim=2, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1.5, self.evaluate(update_op))
      self.assertEqual(1.5, self.evaluate(error))


class PcntBelowThreshTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the percentage-below-threshold metric (fraction of values lower than a threshold).
  # Covers variables, collections and unweighted and weighted single updates. Fixture: every test
  # resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Takes no arguments and returns nothing.
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the percentage-below-threshold metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the count and total accumulators under the percentage-below-threshold scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.percentage_below(values=array_ops.ones((10,)), threshold=2)
    _assert_metric_variables(self, (
        'percentage_below_threshold/count:0',
        'percentage_below_threshold/total:0',
    ))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the percentage-below-threshold metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean, _ = metrics.percentage_below(
        values=array_ops.ones((10,)),
        threshold=2,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the percentage-below-threshold metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.percentage_below(
        values=array_ops.ones((10,)),
        threshold=2,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testOneUpdate(self):
    # [FUNCTION DOC]
    # Scenario: values 2,4,6,8 against thresholds 100, 7 and 1, using three differently named metrics.
    # Action: run all updates then read the values. Expectation: fractions are 1.0, 0.75 and 0.0.
    with self.cached_session():
      values = constant_op.constant(
          [2, 4, 6, 8], shape=(1, 4), dtype=dtypes_lib.float32)

      pcnt0, update_op0 = metrics.percentage_below(values, 100, name='high')
      pcnt1, update_op1 = metrics.percentage_below(values, 7, name='medium')
      pcnt2, update_op2 = metrics.percentage_below(values, 1, name='low')

      self.evaluate(variables.local_variables_initializer())
      self.evaluate([update_op0, update_op1, update_op2])

      pcnt0, pcnt1, pcnt2 = self.evaluate([pcnt0, pcnt1, pcnt2])
      self.assertAlmostEqual(1.0, pcnt0, 5)
      self.assertAlmostEqual(0.75, pcnt1, 5)
      self.assertAlmostEqual(0.0, pcnt2, 5)

  @test_util.run_deprecated_v1
  def testSomePresentOneUpdate(self):
    # [FUNCTION DOC]
    # Scenario: same values with weights 1,0,0,1 so only the first and last values count. Expectation:
    # update results are exactly 1.0, 0.5, 0.0 and the values are the same to five places.
    with self.cached_session():
      values = constant_op.constant(
          [2, 4, 6, 8], shape=(1, 4), dtype=dtypes_lib.float32)
      weights = constant_op.constant(
          [1, 0, 0, 1], shape=(1, 4), dtype=dtypes_lib.float32)

      pcnt0, update_op0 = metrics.percentage_below(
          values, 100, weights=weights, name='high')
      pcnt1, update_op1 = metrics.percentage_below(
          values, 7, weights=weights, name='medium')
      pcnt2, update_op2 = metrics.percentage_below(
          values, 1, weights=weights, name='low')

      self.evaluate(variables.local_variables_initializer())
      self.assertListEqual([1.0, 0.5, 0.0],
                           self.evaluate([update_op0, update_op1, update_op2]))

      pcnt0, pcnt1, pcnt2 = self.evaluate([pcnt0, pcnt1, pcnt2])
      self.assertAlmostEqual(1.0, pcnt0, 5)
      self.assertAlmostEqual(0.5, pcnt1, 5)
      self.assertAlmostEqual(0.0, pcnt2, 5)


class MeanIOUTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the mean intersection-over-union metric, which accumulates a confusion matrix.
  # Covers variables, collections, shape-mismatch errors, idempotent reads, streaming updates with and
  # without weights, missing classes, and the confusion matrix returned by the update operation.
  # Fixture: every test seeds numpy and resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the mean intersection-over-union (mean IOU) metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the single total-confusion-matrix variable under the mean-iou scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.mean_iou(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2)
    _assert_metric_variables(self, ('mean_iou/total_confusion_matrix:0',))

  @test_util.run_deprecated_v1
  def testMetricsCollections(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the mean intersection-over-union (mean IOU) metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean_iou, _ = metrics.mean_iou(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2,
        metrics_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [mean_iou])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the mean intersection-over-union (mean IOU) metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_iou(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testPredictionsAndLabelsOfDifferentSizeRaisesValueError(self):
    # [FUNCTION DOC]
    # Scenario: predictions of shape ten by three and labels of shape ten by four, two classes. Expectation:
    # building the metric raises a value error.
    predictions = array_ops.ones([10, 3])
    labels = array_ops.ones([10, 4])
    with self.assertRaises(ValueError):
      metrics.mean_iou(labels, predictions, num_classes=2)

  @test_util.run_deprecated_v1
  def testLabelsAndWeightsOfDifferentSizeRaisesValueError(self):
    # [FUNCTION DOC]
    # Scenario: predictions and labels of length ten and weights of length nine. Expectation: building the
    # metric raises a value error.
    predictions = array_ops.ones([10])
    labels = array_ops.ones([10])
    weights = array_ops.zeros([9])
    with self.assertRaises(ValueError):
      metrics.mean_iou(labels, predictions, num_classes=2, weights=weights)

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # Checks that reading the value tensor of the mean intersection-over-union (mean IOU) metric repeatedly does not change it.
    # Inputs: random integer predictions and labels of length ten over three classes, generated with identical seeds.
    # Initializes the local variables in a cached session, runs the update operation ten times, reads
    # the value once, then reads it ten more times and asserts each read equals that first read
    # (approximately, for the floating-point based metrics).
    # Returns nothing; fails an assertion otherwise.
    num_classes = 3
    predictions = random_ops.random_uniform(
        [10], maxval=num_classes, dtype=dtypes_lib.int64, seed=1)
    labels = random_ops.random_uniform(
        [10], maxval=num_classes, dtype=dtypes_lib.int64, seed=1)
    mean_iou, update_op = metrics.mean_iou(
        labels, predictions, num_classes=num_classes)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_mean_iou = self.evaluate(mean_iou)
      for _ in range(10):
        self.assertEqual(initial_mean_iou, self.evaluate(mean_iou))

  @test_util.run_deprecated_v1
  def testMultipleUpdates(self):
    # [FUNCTION DOC]
    # Scenario: five single-element batches with three classes, predictions 0,1,2,1,0 and labels 0,1,1,2,1.
    # Action: run the update five times. Expectation: mean IOU equals the mean of per-class IOUs one half, one
    # quarter and zero, compared exactly.
    num_classes = 3
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(5):
        self.evaluate(update_op)
      desired_output = np.mean([1.0 / 2.0, 1.0 / 4.0, 0.])
      self.assertEqual(desired_output, self.evaluate(miou))

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithWeights(self):
    # [FUNCTION DOC]
    # Scenario: six single-element batches with two classes and weights 1,1,1,0,1,0. Expectation: mean IOU
    # equals the mean of two thirds and one half.
    num_classes = 2
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.0])
      weights = weights_queue.dequeue()

      mean_iou, update_op = metrics.mean_iou(
          labels, predictions, num_classes, weights=weights)

      variables.local_variables_initializer().run()
      for _ in range(6):
        self.evaluate(update_op)
      desired_output = np.mean([2.0 / 3.0, 1.0 / 2.0])
      self.assertAlmostEqual(desired_output, self.evaluate(mean_iou))

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithMissingClass(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment in the setup says "There is label for class 2", but the enqueued labels
    # contain no class two (it should say there is no label for it).
    # Scenario: three classes where class two has no predictions and no labels, so its confusion matrix
    # row and column are all zeros. Expectation: mean IOU is the mean of one third and one half, i.e. the
    # empty class is excluded from the average.
    # Test the case where there are no predictions and labels for
    # one class, and thus there is one row and one column with
    # zero entries in the confusion matrix.
    num_classes = 3
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      # There is no prediction for class 2.
      preds_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      # There is label for class 2.
      labels_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(5):
        self.evaluate(update_op)
      desired_output = np.mean([1.0 / 3.0, 2.0 / 4.0])
      self.assertAlmostEqual(desired_output, self.evaluate(miou))

  @test_util.run_deprecated_v1
  def testUpdateOpEvalIsAccumulatedConfusionMatrix(self):
    # [FUNCTION DOC]
    # Scenario: predictions are five zeros then five ones, labels are three zeros then seven ones, two
    # classes. Action: evaluate the update operation. Expectation: it returns the accumulated confusion
    # matrix with rows (3, 0) and (2, 5); the mean IOU equals the mean of three fifths and five sevenths.
    predictions = array_ops.concat(
        [
            constant_op.constant(
                0, shape=[5]), constant_op.constant(
                    1, shape=[5])
        ],
        0)
    labels = array_ops.concat(
        [
            constant_op.constant(
                0, shape=[3]), constant_op.constant(
                    1, shape=[7])
        ],
        0)
    num_classes = 2
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      confusion_matrix = self.evaluate(update_op)
      self.assertAllEqual([[3, 0], [2, 5]], confusion_matrix)
      desired_miou = np.mean([3. / 5., 5. / 7.])
      self.assertAlmostEqual(desired_miou, self.evaluate(miou))

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    # [FUNCTION DOC]
    # Scenario: forty zeros as predictions and labels with one class. Expectation: the update returns a
    # confusion matrix whose first entry is forty and the mean IOU is exactly one.
    predictions = array_ops.zeros([40])
    labels = array_ops.zeros([40])
    num_classes = 1
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(40, self.evaluate(update_op)[0])
      self.assertEqual(1.0, self.evaluate(miou))

  @test_util.run_deprecated_v1
  def testAllWrong(self):
    # [FUNCTION DOC]
    # Scenario: predictions all zero, labels all one, two classes. Expectation: the confusion matrix is
    # rows (0, 0) and (40, 0), and the mean IOU is exactly zero.
    predictions = array_ops.zeros([40])
    labels = array_ops.ones([40])
    num_classes = 2
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([[0, 0], [40, 0]], update_op)
      self.assertEqual(0., self.evaluate(miou))

  @test_util.run_deprecated_v1
  def testResultsWithSomeMissing(self):
    # [FUNCTION DOC]
    # Scenario: ten predictions and labels as in the accumulation test, with weights that drop the first
    # and last example. Expectation: confusion matrix rows (2, 0) and (2, 4), and mean IOU is the mean of
    # one half and two thirds.
    predictions = array_ops.concat(
        [
            constant_op.constant(
                0, shape=[5]), constant_op.constant(
                    1, shape=[5])
        ],
        0)
    labels = array_ops.concat(
        [
            constant_op.constant(
                0, shape=[3]), constant_op.constant(
                    1, shape=[7])
        ],
        0)
    num_classes = 2
    weights = array_ops.concat(
        [
            constant_op.constant(
                0, shape=[1]), constant_op.constant(
                    1, shape=[8]), constant_op.constant(
                        0, shape=[1])
        ],
        0)
    with self.cached_session():
      miou, update_op = metrics.mean_iou(
          labels, predictions, num_classes, weights=weights)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([[2, 0], [2, 4]], update_op)
      desired_miou = np.mean([2. / 4., 4. / 6.])
      self.assertAlmostEqual(desired_miou, self.evaluate(miou))

  @test_util.run_deprecated_v1
  def testMissingClassInLabels(self):
    # [FUNCTION DOC]
    # Scenario: three classes where class two never appears in the labels but is predicted. Expectation:
    # confusion matrix with rows (7,4,3), (3,5,2) and (0,0,0); the mean IOU averages the three per-class
    # ratios, including one with a zero numerator, as a third of the sum.
    labels = constant_op.constant([
        [[0, 0, 1, 1, 0, 0],
         [1, 0, 0, 0, 0, 1]],
        [[1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0]]])
    predictions = constant_op.constant([
        [[0, 0, 2, 1, 1, 0],
         [0, 1, 2, 2, 0, 1]],
        [[0, 0, 2, 1, 1, 1],
         [1, 1, 2, 0, 0, 0]]])
    num_classes = 3
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([[7, 4, 3], [3, 5, 2], [0, 0, 0]], update_op)
      self.assertAlmostEqual(
          1 / 3 * (7 / (7 + 3 + 7) + 5 / (5 + 4 + 5) + 0 / (0 + 5 + 0)),
          self.evaluate(miou))

  @test_util.run_deprecated_v1
  def testMissingClassOverallSmall(self):
    # [FUNCTION DOC]
    # Scenario: one label and one prediction, both class zero, two classes (class one entirely absent).
    # Expectation: confusion matrix rows (1,0) and (0,0); mean IOU is one (the absent class is excluded).
    labels = constant_op.constant([0])
    predictions = constant_op.constant([0])
    num_classes = 2
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([[1, 0], [0, 0]], update_op)
      self.assertAlmostEqual(1, self.evaluate(miou))

  @test_util.run_deprecated_v1
  def testMissingClassOverallLarge(self):
    # [FUNCTION DOC]
    # Scenario: three classes where class two is absent from both labels and predictions, over a larger
    # two by two by six example. Expectation: confusion matrix rows (9,5,0), (3,7,0), (0,0,0); mean IOU
    # averages only the two present classes, a half of nine over seventeen plus seven over fifteen.
    labels = constant_op.constant([
        [[0, 0, 1, 1, 0, 0],
         [1, 0, 0, 0, 0, 1]],
        [[1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0]]])
    predictions = constant_op.constant([
        [[0, 0, 1, 1, 0, 0],
         [1, 1, 0, 0, 1, 1]],
        [[0, 0, 0, 1, 1, 1],
         [1, 1, 1, 0, 0, 0]]])
    num_classes = 3
    with self.cached_session():
      miou, update_op = metrics.mean_iou(labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([[9, 5, 0], [3, 7, 0], [0, 0, 0]], update_op)
      self.assertAlmostEqual(1 / 2 * (9 / (9 + 3 + 5) + 7 / (7 + 5 + 3)),
                             self.evaluate(miou))


class MeanPerClassAccuracyTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the mean per-class accuracy metric (average over classes of the fraction of each class
  # labels predicted correctly). Covers variables, collections, shape-mismatch errors, idempotent reads,
  # streaming updates with weights and missing classes. Fixture: every test seeds numpy and resets
  # the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the mean per-class accuracy metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the count and total accumulators under the mean-accuracy scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.mean_per_class_accuracy(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2)
    _assert_metric_variables(self, ('mean_accuracy/count:0',
                                    'mean_accuracy/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollections(self):
    # [FUNCTION DOC]
    # Checks the metrics-collections argument of the mean per-class accuracy metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the metric value tensor (not the update operation).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__metrics__'
    mean_accuracy, _ = metrics.mean_per_class_accuracy(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2,
        metrics_collections=[my_collection_name])
    self.assertListEqual(
        ops.get_collection(my_collection_name), [mean_accuracy])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    # [FUNCTION DOC]
    # Checks the updates-collections argument of the mean per-class accuracy metric.
    # Builds the metric with that argument set to a custom collection name, then asserts that the
    # collection holds exactly one item, which is the update operation (not the value tensor).
    # Returns nothing; fails an assertion otherwise.
    my_collection_name = '__updates__'
    _, update_op = metrics.mean_per_class_accuracy(
        predictions=array_ops.ones([10, 1]),
        labels=array_ops.ones([10, 1]),
        num_classes=2,
        updates_collections=[my_collection_name])
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testPredictionsAndLabelsOfDifferentSizeRaisesValueError(self):
    # [FUNCTION DOC]
    # Scenario: predictions of shape ten by three, labels ten by four, two classes. Expectation: building
    # the metric raises a value error.
    predictions = array_ops.ones([10, 3])
    labels = array_ops.ones([10, 4])
    with self.assertRaises(ValueError):
      metrics.mean_per_class_accuracy(labels, predictions, num_classes=2)

  @test_util.run_deprecated_v1
  def testLabelsAndWeightsOfDifferentSizeRaisesValueError(self):
    # [FUNCTION DOC]
    # Scenario: predictions and labels of length ten, weights of length nine. Expectation: building the
    # metric raises a value error.
    predictions = array_ops.ones([10])
    labels = array_ops.ones([10])
    weights = array_ops.zeros([9])
    with self.assertRaises(ValueError):
      metrics.mean_per_class_accuracy(
          labels, predictions, num_classes=2, weights=weights)

  @test_util.run_deprecated_v1
  def testValueTensorIsIdempotent(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this test is two tests merged. After the idempotency check, the method reuses the
    # variable for the number of classes and runs an unrelated multiple-update scenario with queues
    # (a copy of the mean IOU multiple-update test adapted for this metric) in the same graph; the second
    # part is not about idempotency.
    # Part one: random integer predictions and labels of length ten over three classes, identical seeds;
    # initialize, run the update ten times, read the value, then ten more reads must all equal the first.
    # Part two: five single-element batches over three classes from queues (predictions 0,1,2,1,0 and
    # labels 0,1,1,2,1); after five updates the value must equal the mean of one, one third and zero.
    num_classes = 3
    predictions = random_ops.random_uniform(
        [10], maxval=num_classes, dtype=dtypes_lib.int64, seed=1)
    labels = random_ops.random_uniform(
        [10], maxval=num_classes, dtype=dtypes_lib.int64, seed=1)
    mean_accuracy, update_op = metrics.mean_per_class_accuracy(
        labels, predictions, num_classes=num_classes)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        self.evaluate(update_op)

      # Then verify idempotency.
      initial_mean_accuracy = self.evaluate(mean_accuracy)
      for _ in range(10):
        self.assertEqual(initial_mean_accuracy, self.evaluate(mean_accuracy))

    num_classes = 3
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(5):
        self.evaluate(update_op)
      desired_output = np.mean([1.0, 1.0 / 3.0, 0.0])
      self.assertAlmostEqual(desired_output, self.evaluate(mean_accuracy))

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithWeights(self):
    # [FUNCTION DOC]
    # Scenario: six single-element batches, two classes, weights 1.0, 0.5, 1.0, 0.0, 1.0, 0.0.
    # Expectation: mean accuracy is the mean of 2/2 and 0.5/1.5.
    num_classes = 2
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          6, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.5])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.0])
      weights = weights_queue.dequeue()

      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes, weights=weights)

      variables.local_variables_initializer().run()
      for _ in range(6):
        self.evaluate(update_op)
      desired_output = np.mean([2.0 / 2.0, 0.5 / 1.5])
      self.assertAlmostEqual(desired_output, self.evaluate(mean_accuracy))

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithMissingClass(self):
    # [FUNCTION DOC]
    # Scenario: three classes where class two has no predictions or labels. Expectation: mean accuracy is
    # the mean of one half, two thirds and zero (the empty class counts as zero here).
    # Test the case where there are no predictions and labels for
    # one class, and thus there is one row and one column with
    # zero entries in the confusion matrix.
    num_classes = 3
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      # There is no prediction for class 2.
      preds_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      # There is label for class 2.
      labels_queue = data_flow_ops.FIFOQueue(
          5, dtypes=dtypes_lib.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(5):
        self.evaluate(update_op)
      desired_output = np.mean([1.0 / 2.0, 2.0 / 3.0, 0.])
      self.assertAlmostEqual(desired_output, self.evaluate(mean_accuracy))

  @test_util.run_deprecated_v1
  def testAllCorrect(self):
    # [FUNCTION DOC]
    # Scenario: forty zeros as predictions and labels with one class. Expectation: the update returns
    # a first entry of exactly one and the mean accuracy is exactly one.
    predictions = array_ops.zeros([40])
    labels = array_ops.zeros([40])
    num_classes = 1
    with self.cached_session():
      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1.0, self.evaluate(update_op)[0])
      self.assertEqual(1.0, self.evaluate(mean_accuracy))

  @test_util.run_deprecated_v1
  def testAllWrong(self):
    # [FUNCTION DOC]
    # Scenario: predictions all zero, labels all one, two classes. Expectation: the update returns per-class
    # accuracies zero and zero, and the mean accuracy is exactly zero.
    predictions = array_ops.zeros([40])
    labels = array_ops.ones([40])
    num_classes = 2
    with self.cached_session():
      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes)
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual([0.0, 0.0], update_op)
      self.assertEqual(0., self.evaluate(mean_accuracy))

  @test_util.run_deprecated_v1
  def testResultsWithSomeMissing(self):
    # [FUNCTION DOC]
    # Scenario: ten predictions and labels with weights dropping the first and last. Expectation: the update
    # returns per-class accuracies 2/2 and 4/6, and the mean accuracy is their mean.
    predictions = array_ops.concat([
        constant_op.constant(0, shape=[5]), constant_op.constant(1, shape=[5])
    ], 0)
    labels = array_ops.concat([
        constant_op.constant(0, shape=[3]), constant_op.constant(1, shape=[7])
    ], 0)
    num_classes = 2
    weights = array_ops.concat([
        constant_op.constant(0, shape=[1]), constant_op.constant(1, shape=[8]),
        constant_op.constant(0, shape=[1])
    ], 0)
    with self.cached_session():
      mean_accuracy, update_op = metrics.mean_per_class_accuracy(
          labels, predictions, num_classes, weights=weights)
      self.evaluate(variables.local_variables_initializer())
      desired_accuracy = np.array([2. / 2., 4. / 6.], dtype=np.float32)
      self.assertAllEqual(desired_accuracy, update_op)
      desired_mean_accuracy = np.mean(desired_accuracy)
      self.assertAlmostEqual(desired_mean_accuracy,
                             self.evaluate(mean_accuracy))


class FalseNegativesTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the false-negatives count metric (weighted count of positive labels predicted negative).
  # Covers the created variable and unweighted and weighted counts. Fixture: every test seeds numpy and
  # resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the false-negatives metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the single count variable under the false-negatives scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.false_negatives(
        labels=(0, 1, 0, 1),
        predictions=(0, 0, 1, 1))
    _assert_metric_variables(self, ('false_negatives/count:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variables are named for true negatives (a copy and paste leftover) though
    # the metric under test is false negatives.
    # Scenario: a four by five binary labels matrix and predictions matrix. Expectation: before the
    # update the value is zero, the update result is three, and the value then is three.
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    tn, tn_update_op = metrics.false_negatives(
        labels=labels, predictions=predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn)
      self.assertAllClose(3., tn_update_op)
      self.assertAllClose(3., tn)

  @test_util.run_deprecated_v1
  def testWeighted(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variables are named for true negatives (copy and paste leftover) though the
    # metric under test is false negatives.
    # Scenario: same data with per-row weights 1, 1.5, 2, 2.5. Expectation: value before update zero, and
    # after the update both update result and value equal five.
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    weights = constant_op.constant((1., 1.5, 2., 2.5))
    tn, tn_update_op = metrics.false_negatives(
        labels=labels, predictions=predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn)
      self.assertAllClose(5., tn_update_op)
      self.assertAllClose(5., tn)


class FalseNegativesAtThresholdsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the false-negatives-at-thresholds metric, which returns one count per threshold.
  # Covers the created variable and unweighted and per-row weighted counts, thresholds 0.15, 0.5, 0.85.
  # Fixture: every test seeds numpy and resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the false-negatives-at-thresholds metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the single false-negatives variable under the false-negatives scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.false_negatives_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0.15, 0.5, 0.85])
    _assert_metric_variables(self, ('false_negatives/false_negatives:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    # [FUNCTION DOC]
    # Scenario: three examples of four scores with binary labels. Expectation: before update the counts are
    # (0,0,0); update and value afterwards are (0,2,3).
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    fn, fn_update_op = metrics.false_negatives_at_thresholds(
        predictions=predictions, labels=labels, thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), fn)
      self.assertAllEqual((0, 2, 3), fn_update_op)
      self.assertAllEqual((0, 2, 3), fn)

  @test_util.run_deprecated_v1
  def testWeighted(self):
    # [FUNCTION DOC]
    # Scenario: same data with one weight per row (3, 5, 7). Expectation: before update (0,0,0); update
    # result and value afterwards are (0, 8, 11).
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    fn, fn_update_op = metrics.false_negatives_at_thresholds(
        predictions=predictions,
        labels=labels,
        weights=((3.0,), (5.0,), (7.0,)),
        thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), fn)
      self.assertAllEqual((0.0, 8.0, 11.0), fn_update_op)
      self.assertAllEqual((0.0, 8.0, 11.0), fn)


class FalsePositivesTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the false-positives count metric (weighted count of negative labels predicted positive).
  # Covers the created variable and unweighted and weighted counts. Fixture: every test seeds numpy and
  # resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the false-positives metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the single count variable under the false-positives scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.false_positives(
        labels=(0, 1, 0, 1),
        predictions=(0, 0, 1, 1))
    _assert_metric_variables(self, ('false_positives/count:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variables are named for true negatives (copy and paste leftover) though the
    # metric under test is false positives.
    # Scenario: a four by five binary labels matrix and predictions matrix. Expectation: value before update
    # zero; update result and value afterwards seven.
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    tn, tn_update_op = metrics.false_positives(
        labels=labels, predictions=predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn)
      self.assertAllClose(7., tn_update_op)
      self.assertAllClose(7., tn)

  @test_util.run_deprecated_v1
  def testWeighted(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variables are named for true negatives (copy and paste leftover) though the
    # metric under test is false positives.
    # Scenario: same data with per-row weights 1, 1.5, 2, 2.5. Expectation: value before update zero;
    # update result and value afterwards fourteen.
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    weights = constant_op.constant((1., 1.5, 2., 2.5))
    tn, tn_update_op = metrics.false_positives(
        labels=labels, predictions=predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn)
      self.assertAllClose(14., tn_update_op)
      self.assertAllClose(14., tn)


class FalsePositivesAtThresholdsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the false-positives-at-thresholds metric, which returns one count per threshold.
  # Covers the created variable and unweighted and element-wise weighted counts, thresholds 0.15, 0.5,
  # 0.85. Fixture: every test seeds numpy and resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the false-positives-at-thresholds metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the single false-positives variable under the false-positives scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.false_positives_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0.15, 0.5, 0.85])
    _assert_metric_variables(self, ('false_positives/false_positives:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    # [FUNCTION DOC]
    # Scenario: three examples of four scores with binary labels. Expectation: before update (0,0,0);
    # update and value afterwards (7,4,2).
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    fp, fp_update_op = metrics.false_positives_at_thresholds(
        predictions=predictions, labels=labels, thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), fp)
      self.assertAllEqual((7, 4, 2), fp_update_op)
      self.assertAllEqual((7, 4, 2), fp)

  @test_util.run_deprecated_v1
  def testWeighted(self):
    # [FUNCTION DOC]
    # Scenario: same data with element-wise weights (a three by four matrix of distinct
    # positive integers). Expectation: before update (0,0,0); afterwards (125, 42, 12).
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    fp, fp_update_op = metrics.false_positives_at_thresholds(
        predictions=predictions,
        labels=labels,
        weights=((1.0, 2.0, 3.0, 5.0),
                 (7.0, 11.0, 13.0, 17.0),
                 (19.0, 23.0, 29.0, 31.0)),
        thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), fp)
      self.assertAllEqual((125.0, 42.0, 12.0), fp_update_op)
      self.assertAllEqual((125.0, 42.0, 12.0), fp)


class TrueNegativesTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the true-negatives count metric (weighted count of negative labels predicted negative).
  # Covers the created variable and unweighted and weighted counts. Fixture: every test seeds numpy and
  # resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the true-negatives metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the single count variable under the true-negatives scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.true_negatives(
        labels=(0, 1, 0, 1),
        predictions=(0, 0, 1, 1))
    _assert_metric_variables(self, ('true_negatives/count:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    # [FUNCTION DOC]
    # Scenario: a four by five binary labels matrix and predictions matrix. Expectation: value before update
    # zero; update result and value afterwards three.
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    tn, tn_update_op = metrics.true_negatives(
        labels=labels, predictions=predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn)
      self.assertAllClose(3., tn_update_op)
      self.assertAllClose(3., tn)

  @test_util.run_deprecated_v1
  def testWeighted(self):
    # [FUNCTION DOC]
    # Scenario: same data with per-row weights 1, 1.5, 2, 2.5. Expectation: value before update zero; update
    # result and value afterwards four.
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    weights = constant_op.constant((1., 1.5, 2., 2.5))
    tn, tn_update_op = metrics.true_negatives(
        labels=labels, predictions=predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn)
      self.assertAllClose(4., tn_update_op)
      self.assertAllClose(4., tn)


class TrueNegativesAtThresholdsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the true-negatives-at-thresholds metric, which returns one count per threshold.
  # Covers the created variable and unweighted and weighted counts, thresholds 0.15, 0.5, 0.85.
  # Fixture: every test seeds numpy and resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the true-negatives-at-thresholds metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the single true-negatives variable under the true-negatives scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.true_negatives_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0.15, 0.5, 0.85])
    _assert_metric_variables(self, ('true_negatives/true_negatives:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    # [FUNCTION DOC]
    # Scenario: three examples of four scores with binary labels. Expectation: before update (0,0,0);
    # update and value afterwards (2,5,7).
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    tn, tn_update_op = metrics.true_negatives_at_thresholds(
        predictions=predictions, labels=labels, thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), tn)
      self.assertAllEqual((2, 5, 7), tn_update_op)
      self.assertAllEqual((2, 5, 7), tn)

  @test_util.run_deprecated_v1
  def testWeighted(self):
    # [FUNCTION DOC]
    # Scenario: same data with a single row of weights (0, 2, 3, 5) broadcast over the examples.
    # Expectation: before update (0,0,0); afterwards (5, 15, 23).
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    tn, tn_update_op = metrics.true_negatives_at_thresholds(
        predictions=predictions,
        labels=labels,
        weights=((0.0, 2.0, 3.0, 5.0),),
        thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), tn)
      self.assertAllEqual((5.0, 15.0, 23.0), tn_update_op)
      self.assertAllEqual((5.0, 15.0, 23.0), tn)


class TruePositivesTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the true-positives count metric (weighted count of positive labels predicted positive).
  # Covers the created variable and unweighted and weighted counts. Fixture: every test seeds numpy and
  # resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the true-positives metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the single count variable under the true-positives scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.true_positives(
        labels=(0, 1, 0, 1),
        predictions=(0, 0, 1, 1))
    _assert_metric_variables(self, ('true_positives/count:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variables are named for true negatives (copy and paste leftover) though the
    # metric under test is true positives.
    # Scenario: a four by five binary labels matrix and predictions matrix. Expectation: value before update
    # zero; update result and value afterwards seven.
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    tn, tn_update_op = metrics.true_positives(
        labels=labels, predictions=predictions)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn)
      self.assertAllClose(7., tn_update_op)
      self.assertAllClose(7., tn)

  @test_util.run_deprecated_v1
  def testWeighted(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variables are named for true negatives (copy and paste leftover) though the
    # metric under test is true positives.
    # Scenario: same data with per-row weights 1, 1.5, 2, 2.5. Expectation: value before update zero; update
    # result and value afterwards twelve.
    labels = constant_op.constant(((0, 1, 0, 1, 0),
                                   (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0),
                                   (0, 0, 0, 0, 1)))
    predictions = constant_op.constant(((0, 0, 1, 1, 0),
                                        (1, 1, 1, 1, 1),
                                        (0, 1, 0, 1, 0),
                                        (1, 1, 1, 1, 1)))
    weights = constant_op.constant((1., 1.5, 2., 2.5))
    tn, tn_update_op = metrics.true_positives(
        labels=labels, predictions=predictions, weights=weights)

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllClose(0., tn)
      self.assertAllClose(12., tn_update_op)
      self.assertAllClose(12., tn)


class TruePositivesAtThresholdsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the true-positives-at-thresholds metric, which returns one count per threshold.
  # Covers the created variable and unweighted and scalar-weighted counts, thresholds 0.15, 0.5, 0.85.
  # Fixture: every test seeds numpy and resets the default graph.

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture run before every test method of this class.
    # Resets the process-wide default graph so that metric variables, collections and ops created by
    # earlier tests do not leak into this one.
    # Also seeds the numpy random generator with a fixed value so randomly generated inputs are
    # reproducible.
    # Takes no arguments and returns nothing.
    np.random.seed(1)
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    # [FUNCTION DOC]
    # Checks which local variables the true-positives-at-thresholds metric creates.
    # Builds the metric once on small constant inputs, then asserts (through the shared metric-variable
    # check helper) that the set of local variables, and the set recorded in the metric-variables
    # collection, both equal exactly: the single true-positives variable under the true-positives scope.
    # Returns nothing; fails an assertion on any mismatch.
    metrics.true_positives_at_thresholds(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        thresholds=[0.15, 0.5, 0.85])
    _assert_metric_variables(self, ('true_positives/true_positives:0',))

  @test_util.run_deprecated_v1
  def testUnweighted(self):
    # [FUNCTION DOC]
    # Scenario: three examples of four scores with binary labels. Expectation: before update (0,0,0);
    # update and value afterwards (3,1,0).
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    tp, tp_update_op = metrics.true_positives_at_thresholds(
        predictions=predictions, labels=labels, thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), tp)
      self.assertAllEqual((3, 1, 0), tp_update_op)
      self.assertAllEqual((3, 1, 0), tp)

  @test_util.run_deprecated_v1
  def testWeighted(self):
    # [FUNCTION DOC]
    # Scenario: same data with a scalar weight of 37. Expectation: before update (0,0,0); update result
    # and value afterwards (111, 37, 0).
    predictions = constant_op.constant(((0.9, 0.2, 0.8, 0.1),
                                        (0.2, 0.9, 0.7, 0.6),
                                        (0.1, 0.2, 0.4, 0.3)))
    labels = constant_op.constant(((0, 1, 1, 0),
                                   (1, 0, 0, 0),
                                   (0, 0, 0, 0)))
    tp, tp_update_op = metrics.true_positives_at_thresholds(
        predictions=predictions, labels=labels, weights=37.0,
        thresholds=[0.15, 0.5, 0.85])

    with self.cached_session():
      self.evaluate(variables.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), tp)
      self.assertAllEqual((111.0, 37.0, 0.0), tp_update_op)
      self.assertAllEqual((111.0, 37.0, 0.0), tp)


if __name__ == '__main__':
  test.main()
