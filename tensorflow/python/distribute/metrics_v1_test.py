# [MODULE DOC]
# Tests for version-one streaming metrics (mean, accuracy, AUC, confusion counts, precision,
# recall, error metrics and others) when computed under distribution strategies in graph mode:
# default, one-device, mirrored and TPU. Defines small dataset factory functions yielding
# labels and predictions, combination helpers selecting strategies, and one test class whose
# shared driver runs a metric for several update steps and compares the value with expected
# numbers computed by hand. Runs the standard test main when executed.

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
"""Tests for V1 metrics."""
from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import variables


def _labeled_dataset_fn():
  # [FUNCTION DOC]
  # HELPER: dataset factory for classification metric tests. Returns a dataset of one thousand
  # elements mapped to mappings with integer labels (index modulo five) and predictions (index
  # modulo three), batched by four with the remainder dropped.
  # First four batches of x: labels, predictions -> (labels == predictions)
  #  0: 0, 0 -> True;   1: 1, 1 -> True;   2: 2, 2 -> True;   3: 3, 0 -> False
  #  4: 4, 1 -> False;  5: 0, 2 -> False;  6: 1, 0 -> False;  7: 2, 1 -> False
  #  8: 3, 2 -> False;  9: 4, 0 -> False; 10: 0, 1 -> False; 11: 1, 2 -> False
  # 12: 2, 0 -> False; 13: 3, 1 -> False; 14: 4, 2 -> False; 15: 0, 0 -> True
  return dataset_ops.Dataset.range(1000).map(
      lambda x: {"labels": x % 5, "predictions": x % 3}).batch(
          4, drop_remainder=True)


def _boolean_dataset_fn():
  # [FUNCTION DOC]
  # HELPER: dataset factory for confusion-count metric tests. Returns an endlessly repeating
  # dataset of mappings with four boolean labels and four boolean predictions, batched by three
  # with the remainder dropped, so batches cycle through true/false positives and negatives.
  # First four batches of labels, predictions: {TP, FP, TN, FN}
  # with a threshold of 0.5:
  #   T, T -> TP;  F, T -> FP;   T, F -> FN
  #   F, F -> TN;  T, T -> TP;   F, T -> FP
  #   T, F -> FN;  F, F -> TN;   T, T -> TP
  #   F, T -> FP;  T, F -> FN;   F, F -> TN
  return dataset_ops.Dataset.from_tensor_slices({
      "labels": [True, False, True, False],
      "predictions": [True, True, False, False]}).repeat().batch(
          3, drop_remainder=True)


def _threshold_dataset_fn():
  # [FUNCTION DOC]
  # HELPER: dataset factory for threshold-based metric tests. Returns an endlessly repeating
  # dataset of mappings with four boolean labels and four float predictions (one, three quarters,
  # one quarter, zero), batched by three with the remainder dropped.
  # First four batches of labels, predictions: {TP, FP, TN, FN}
  # with a threshold of 0.5:
  #   True, 1.0 -> TP;  False, .75 -> FP;   True, .25 -> FN
  #  False, 0.0 -> TN;   True, 1.0 -> TP;  False, .75 -> FP
  #   True, .25 -> FN;  False, 0.0 -> TN;   True, 1.0 -> TP
  #  False, .75 -> FP;   True, .25 -> FN;  False, 0.0 -> TN
  return dataset_ops.Dataset.from_tensor_slices({
      "labels": [True, False, True, False],
      "predictions": [1.0, 0.75, 0.25, 0.]}).repeat().batch(
          3, drop_remainder=True)


def _regression_dataset_fn():
  # [FUNCTION DOC]
  # HELPER: dataset factory for error metric tests. Returns an endlessly repeating, unbatched
  # dataset of mappings with four float labels and four float predictions.
  return dataset_ops.Dataset.from_tensor_slices({
      "labels": [1., .5, 1., 0.],
      "predictions": [1., .75, .25, 0.]}).repeat()


def all_combinations():
  # [FUNCTION DOC]
  # HELPER: test parameterization. Returns the combinations of graph mode with the default,
  # one-device, mirrored (GPU and CPU) and mirrored (two GPUs) strategies.
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.one_device_strategy,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],
      mode=["graph"])


def tpu_combinations():
  # [FUNCTION DOC]
  # HELPER: test parameterization. Returns the combinations of graph mode with the one-step TPU
  # strategy and the regular TPU strategy.
  return combinations.combine(
      distribution=[
          strategy_combinations.tpu_strategy_one_step,
          strategy_combinations.tpu_strategy
      ],
      mode=["graph"])


# TODO(josh11b): Test metrics.recall_at_top_k, metrics.average_precision_at_k,
# metrics.precision_at_k
class MetricsV1Test(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests version-one metrics under distribution strategies. Each test supplies a dataset
  # factory, a metric function and a function giving expected values; the shared driver
  # runs the metric and checks values. Parameterized by strategy through the distribute
  # combinations framework.

  def _test_metric(self, distribution, dataset_fn, metric_fn, expected_fn):
    # [FUNCTION DOC]
    # HELPER: shared driver for all metric tests. Parameters: the strategy; a dataset factory; a
    # metric function returning a (value, update) pair from a batch; and a function giving the
    # expected value after a number of consumed batches. In a fresh graph under the strategy scope
    # builds an input iterator. For TPU strategies runs multiple steps per run on the iterator and
    # captures the metric value as a non-tensor output; otherwise calls the metric per replica on
    # the next batch and groups the update. Initializes the iterator and local variables, then up
    # to four times evaluates the update and asserts the metric value is close (tolerance 0.001)
    # to the expected value for the number of batches consumed so far (replica count per update,
    # times steps per run on TPU), stopping once four batches have been consumed.
    with ops.Graph().as_default(), distribution.scope():
      iterator = distribution.make_input_fn_iterator(lambda _: dataset_fn())
      if strategy_test_lib.is_tpu_strategy(distribution):
        def step_fn(ctx, inputs):
          # [FUNCTION DOC]
          # Step function for TPU runs. Parameters: the step context and the batch inputs. Calls
          # the metric per replica, records the metric value as a non-tensor output named value
          # on the context, and returns the grouped update operation.
          value, update = distribution.extended.call_for_each_replica(
              metric_fn, args=(inputs,))
          ctx.set_non_tensor_output(name="value", output=value)
          return distribution.group(update)

        ctx = distribution.extended.experimental_run_steps_on_iterator(
            step_fn, iterator, iterations=distribution.extended.steps_per_run)
        update = ctx.run_op
        value = ctx.non_tensor_outputs["value"]
        # In each run, we run multiple steps, and each steps consumes as many
        # batches as number of replicas.
        batches_per_update = (
            distribution.num_replicas_in_sync *
            distribution.extended.steps_per_run)
      else:
        value, update = distribution.extended.call_for_each_replica(
            metric_fn, args=(iterator.get_next(),))
        update = distribution.group(update)
        # TODO(josh11b): Once we switch to using a global batch size for input,
        # replace "distribution.num_replicas_in_sync" with "1".
        batches_per_update = distribution.num_replicas_in_sync

      self.evaluate(iterator.initializer)
      self.evaluate(variables.local_variables_initializer())

      batches_consumed = 0
      for i in range(4):
        self.evaluate(update)
        batches_consumed += batches_per_update
        self.assertAllClose(expected_fn(batches_consumed),
                            self.evaluate(value),
                            0.001,
                            msg="After update #" + str(i+1))
        if batches_consumed >= 4:  # Consume 4 input batches in total.
          break

  @combinations.generate(all_combinations() + tpu_combinations())
  def testMean(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the mean metric over a range of one thousand floats batched by four. Expected
    # value after n batches is twice n minus one half.
    def _dataset_fn():
      # [FUNCTION DOC]
      # Returns a dataset of floats from zero to 999 batched by four, dropping the remainder.
      return dataset_ops.Dataset.range(1000).map(math_ops.to_float).batch(
          4, drop_remainder=True)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected mean after the given number of batches (twice the number minus one
      # half).
      # Mean(0..3) = 1.5, Mean(0..7) = 3.5, Mean(0..11) = 5.5, etc.
      return num_batches * 2 - 0.5

    self._test_metric(distribution, _dataset_fn, metrics.mean, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testAccuracy(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the accuracy metric on the labeled dataset; expected values come from a fixed
    # table of cumulative accuracies for one to four batches.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping, extracts labels and predictions, returns the accuracy metric pair.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.accuracy(labels, predictions)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected accuracy from a fixed table indexed by batches consumed (one to four).
      return [3./4, 3./8, 3./12, 4./16][num_batches - 1]

    self._test_metric(
        distribution, _labeled_dataset_fn, _metric_fn, _expected_fn)

  # TODO(priyag, jhseu): Enable TPU for this test once scatter_add is added
  # for TPUMirroredVariable.
  @combinations.generate(all_combinations())
  def testMeanPerClassAccuracy(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the mean per-class accuracy metric with five classes on the labeled dataset (not
    # run on TPU). Expected values are averages of per-class accuracies from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the mean per-class accuracy metric pair with five classes.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.mean_per_class_accuracy(
          labels, predictions, num_classes=5)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected metric from a fixed table (averages of per-class accuracies) indexed
      # by batches consumed (one to four).
      mean = lambda x: sum(x) / len(x)
      return [mean([1., 1., 1., 0., 0.]),
              mean([0.5, 0.5, 0.5, 0., 0.]),
              mean([1./3, 1./3, 0.5, 0., 0.]),
              mean([0.5, 1./3, 1./3, 0., 0.])][num_batches - 1]

    self._test_metric(
        distribution, _labeled_dataset_fn, _metric_fn, _expected_fn)

  # NOTE(priyag): This metric doesn't work on TPUs yet.
  @combinations.generate(all_combinations())
  def testMeanIOU(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the mean intersection-over-union metric with five classes on the labeled dataset
    # (not run on TPU). Expected values come from a fixed table of averages.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the mean intersection-over-union metric pair (five
      # classes).
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.mean_iou(
          labels, predictions, num_classes=5)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected metric from a fixed table indexed by batches consumed (one to four).
      # The first entry averages only four classes since class four is absent in the first batch.
      mean = lambda x: sum(x) / len(x)
      return [mean([1./2, 1./1, 1./1, 0.]),  # no class 4 in first batch
              mean([1./4, 1./4, 1./3, 0., 0.]),
              mean([1./6, 1./6, 1./5, 0., 0.]),
              mean([2./8, 1./7, 1./7, 0., 0.])][num_batches - 1]

    self._test_metric(
        distribution, _labeled_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testMeanTensor(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the element-wise mean-tensor metric over floats batched by four with a fixed
    # shape. Expected value is a four-element list whose entries grow by two per consumed batch.
    def _dataset_fn():
      # [FUNCTION DOC]
      # Returns a dataset of floats from zero to 999 batched by four, dropping the remainder to
      # keep a fixed, known shape.
      dataset = dataset_ops.Dataset.range(1000).map(math_ops.to_float)
      # Want to produce a fixed, known shape, so drop remainder when batching.
      dataset = dataset.batch(4, drop_remainder=True)
      return dataset

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected per-position means as a list of four values starting at twice the
      # batch count minus two and increasing by one per position.
      # Mean(0, 4, ..., 4 * num_batches - 4) == 2 * num_batches - 2
      # Mean(1, 5, ..., 4 * num_batches - 3) == 2 * num_batches - 1
      # Mean(2, 6, ..., 4 * num_batches - 2) == 2 * num_batches
      # Mean(3, 7, ..., 4 * num_batches - 1) == 2 * num_batches + 1
      first = 2. * num_batches - 2.
      return [first, first + 1., first + 2., first + 3.]

    self._test_metric(
        distribution, _dataset_fn, metrics.mean_tensor, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testAUCROC(self, distribution):
    # [FUNCTION DOC]
    # Scenario: area under the receiver operating characteristic curve with eight thresholds and
    # careful interpolation, on the threshold dataset; expected values from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the area-under-curve metric pair for the ROC curve.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.auc(labels, predictions, num_thresholds=8, curve="ROC",
                         summation_method="careful_interpolation")

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected area from a fixed table indexed by batches consumed (one to four).
      return [0.5, 7./9, 0.8, 0.75][num_batches - 1]

    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testAUCPR(self, distribution):
    # [FUNCTION DOC]
    # Scenario: area under the precision-recall curve with eight thresholds and careful
    # interpolation, on the threshold dataset; expected values from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the area-under-curve metric pair for the PR curve.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.auc(labels, predictions, num_thresholds=8, curve="PR",
                         summation_method="careful_interpolation")

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected area from a fixed table indexed by batches consumed (one to four).
      return [0.797267, 0.851238, 0.865411, 0.797267][num_batches - 1]

    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testFalseNegatives(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the cumulative false-negative count on the boolean dataset; expected values from a
    # fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the false-negatives metric pair.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.false_negatives(labels, predictions)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected count from a fixed table indexed by batches consumed (one to four).
      return [1., 1., 2., 3.][num_batches - 1]

    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testFalseNegativesAtThresholds(self, distribution):
    # [FUNCTION DOC]
    # Scenario: false negatives at a single threshold of one half on the threshold dataset;
    # expected values are one-element lists from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the false-negatives-at-thresholds metric pair for the
      # single threshold one half.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.false_negatives_at_thresholds(labels, predictions, [.5])

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected one-element list from a fixed table indexed by batches consumed.
      return [[1.], [1.], [2.], [3.]][num_batches - 1]

    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testTrueNegatives(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the cumulative true-negative count on the boolean dataset; expected values from a
    # fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the true-negatives metric pair.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.true_negatives(labels, predictions)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected count from a fixed table indexed by batches consumed (one to four).
      return [0., 1., 2., 3.][num_batches - 1]

    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testTrueNegativesAtThresholds(self, distribution):
    # [FUNCTION DOC]
    # Scenario: true negatives at a single threshold of one half on the threshold dataset;
    # expected values are one-element lists from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the true-negatives-at-thresholds metric pair for the
      # single threshold one half.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.true_negatives_at_thresholds(labels, predictions, [.5])

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected one-element list from a fixed table indexed by batches consumed.
      return [[0.], [1.], [2.], [3.]][num_batches - 1]

    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testFalsePositives(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the cumulative false-positive count on the boolean dataset; expected values from a
    # fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the false-positives metric pair.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.false_positives(labels, predictions)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected count from a fixed table indexed by batches consumed (one to four).
      return [1., 2., 2., 3.][num_batches - 1]

    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testFalsePositivesAtThresholds(self, distribution):
    # [FUNCTION DOC]
    # Scenario: false positives at a single threshold of one half on the threshold dataset;
    # expected values are one-element lists from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the false-positives-at-thresholds metric pair for the
      # single threshold one half.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.false_positives_at_thresholds(labels, predictions, [.5])

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected one-element list from a fixed table indexed by batches consumed.
      return [[1.], [2.], [2.], [3.]][num_batches - 1]

    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testTruePositives(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the cumulative true-positive count on the boolean dataset; expected values from a
    # fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the true-positives metric pair.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.true_positives(labels, predictions)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected count from a fixed table indexed by batches consumed (one to four).
      return [1., 2., 3., 3.][num_batches - 1]

    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testTruePositivesAtThresholds(self, distribution):
    # [FUNCTION DOC]
    # Scenario: true positives at a single threshold of one half on the threshold dataset;
    # expected values are one-element lists from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the true-positives-at-thresholds metric pair for the
      # single threshold one half.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.true_positives_at_thresholds(labels, predictions, [.5])

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected one-element list from a fixed table indexed by batches consumed.
      return [[1.], [2.], [3.], [3.]][num_batches - 1]

    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testPrecision(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the precision metric on the boolean dataset; expected values from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the precision metric pair.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.precision(labels, predictions)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected precision from a fixed table indexed by batches consumed (one to four).
      return [0.5, 0.5, 0.6, 0.5][num_batches - 1]

    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testPrecisionAtThreshold(self, distribution):
    # [FUNCTION DOC]
    # Scenario: precision at a single threshold of one half on the threshold dataset; expected
    # values are one-element lists from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the precision-at-thresholds metric pair (threshold one
      # half).
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.precision_at_thresholds(labels, predictions, [0.5])

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected one-element list from a fixed table indexed by batches consumed.
      return [[0.5], [0.5], [0.6], [0.5]][num_batches - 1]

    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testRecall(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the recall metric on the boolean dataset; expected values from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the recall metric pair.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.recall(labels, predictions)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected recall from a fixed table indexed by batches consumed (one to four).
      return [0.5, 2./3, 0.6, 0.5][num_batches - 1]

    self._test_metric(
        distribution, _boolean_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testRecallAtThreshold(self, distribution):
    # [FUNCTION DOC]
    # Scenario: recall at a single threshold of one half on the threshold dataset; expected values
    # are one-element lists from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the recall-at-thresholds metric pair (threshold one half).
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.recall_at_thresholds(labels, predictions, [0.5])

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected one-element list from a fixed table indexed by batches consumed.
      return [[0.5], [2./3], [0.6], [0.5]][num_batches - 1]

    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testMeanSquaredError(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the mean squared error metric on the regression dataset; expected values from a
    # fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the mean-squared-error metric pair.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.mean_squared_error(labels, predictions)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected error from a fixed table indexed by batches consumed (one to four).
      return [0., 1./32, 0.208333, 0.15625][num_batches - 1]

    self._test_metric(
        distribution, _regression_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations() + tpu_combinations())
  def testRootMeanSquaredError(self, distribution):
    # [FUNCTION DOC]
    # Scenario: the root mean squared error metric on the regression dataset; expected values from
    # a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the root-mean-squared-error metric pair.
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.root_mean_squared_error(labels, predictions)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected error from a fixed table indexed by batches consumed (one to four).
      return [0., 0.176777, 0.456435, 0.395285][num_batches - 1]

    self._test_metric(
        distribution, _regression_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations())
  def testSensitivityAtSpecificity(self, distribution):
    # [FUNCTION DOC]
    # Scenario: sensitivity at a specificity of 0.8 on the threshold dataset (not run on TPU);
    # expected values from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the sensitivity-at-specificity metric pair (0.8).
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.sensitivity_at_specificity(labels, predictions, 0.8)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected sensitivity from a fixed table indexed by batches consumed.
      return [0.5, 2./3, 0.6, 0.5][num_batches - 1]

    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)

  @combinations.generate(all_combinations())
  def testSpecificityAtSensitivity(self, distribution):
    # [FUNCTION DOC]
    # Scenario: specificity at a sensitivity of 0.95 on the threshold dataset (not run on TPU);
    # expected values from a fixed table.
    def _metric_fn(x):
      # [FUNCTION DOC]
      # Takes a batch mapping and returns the specificity-at-sensitivity metric pair (0.95).
      labels = x["labels"]
      predictions = x["predictions"]
      return metrics.specificity_at_sensitivity(labels, predictions, 0.95)

    def _expected_fn(num_batches):
      # [FUNCTION DOC]
      # Returns the expected specificity from a fixed table indexed by batches consumed.
      return [0., 1./3, 0.5, 0.5][num_batches - 1]

    self._test_metric(
        distribution, _threshold_dataset_fn, _metric_fn, _expected_fn)


if __name__ == "__main__":
  test.main()
