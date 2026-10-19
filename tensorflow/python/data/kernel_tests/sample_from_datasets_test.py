# [MODULE DOC]
# Tests for the dataset sample_from_datasets transformation, which interleaves several datasets by
# randomly choosing the source of each element according to weights (given as a list, a tensor or
# a dataset of weights). Covers statistical correctness (chi-squared), stop/skip behavior on empty
# datasets, zero weights, cardinality, nesting, per-iteration re-randomization, argument
# validation errors and checkpointing. Contains small module-level helpers for the weights types.
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
"""Tests for `tf.data.Dataset.sample_from_dataset()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import compat as tf_compat
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test


def _weights_type_combinations():
  # [FUNCTION DOC]
  # HELPER: shared by the sample_from_datasets tests to parameterize how weights are supplied.
  # Returns a test combination over the weights type: plain list, tensor or dataset.
  return combinations.combine(weights_type=["list", "tensor", "dataset"])


def _get_weights_of_type(weights_list, weights_type):
  # [FUNCTION DOC]
  # HELPER: converts a list/array of weights into the requested representation.
  # weights_type is "list" (returned unchanged), "tensor" (converted to a tensor named weights) or
  # anything else, treated as "dataset" (an endlessly repeating dataset of the weights).
  if weights_type == "list":
    return weights_list
  if weights_type == "tensor":
    return ops.convert_to_tensor(weights_list, name="weights")
  return dataset_ops.Dataset.from_tensors(weights_list).repeat()


class SampleFromDatasetsTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for sample_from_datasets: distribution accuracy, empty/zero-weight handling, cardinality,
  # nesting, re-randomization across epochs and argument errors.

  def _normalize(self, vec):
    # [FUNCTION DOC]
    # HELPER: scales a numpy vector so its entries sum to one and returns it.
    return vec / vec.sum()

  def _chi2(self, expected, actual):
    # [FUNCTION DOC]
    # HELPER: computes a chi-squared style statistic between expected and observed frequencies.
    # Both are converted to arrays; returns the sum over the first axis of squared difference
    # divided by the expected value.
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    diff = actual - expected
    chi2 = np.sum(diff * diff / expected, axis=0)
    return chi2

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _weights_type_combinations()))
  def testSampleFromDatasets(self, weights_type):
    # [FUNCTION DOC]
    # With a fixed random seed, samples five thousand elements from endless single-value datasets
    # for three probability vectors (fixed, random, single class) with weights supplied in the
    # parameterized form. Asserts the take ends with an out-of-range error and that the observed
    # frequencies pass a chi-squared check against the probabilities (threshold 0.01).
    random_seed.set_random_seed(1619)
    num_samples = 5000
    rand_probs = self._normalize(np.random.random_sample((5,)))

    # Use chi-squared test to assert that the observed distribution matches the
    # expected distribution. Based on the implementation in
    # "third_party/tensorflow/python/kernel_tests/multinomial_op_test.py".
    for probs in [[.85, .05, .1], rand_probs, [1.]]:
      weights = _get_weights_of_type(np.asarray(probs), weights_type)
      classes = len(probs)

      # Create a dataset that samples each integer in `[0, num_datasets)`
      # with probability given by `weights[i]`.
      dataset = dataset_ops.Dataset.sample_from_datasets([
          dataset_ops.Dataset.from_tensors(i).repeat() for i in range(classes)
      ], weights)
      dataset = dataset.take(num_samples)

      next_element = self.getNext(dataset, requires_initialization=True)
      freqs = np.zeros([classes])
      for _ in range(num_samples):
        freqs[self.evaluate(next_element())] += 1
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element())

      self.assertLess(self._chi2(probs, freqs / num_samples), 1e-2)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _weights_type_combinations()))
  def testSampleFromDatasetsStoppingOnEmptyDataset(self, weights_type):
    # [FUNCTION DOC]
    # Samples from a one-element dataset and two endless ones with stop-on-empty enabled.
    # Asserts the sentinel value from the first dataset appears exactly once (sampling ends when
    # it is exhausted).
    # Sampling stops when the first dataset is exhausted.
    weights = _get_weights_of_type(np.asarray([.5, .1, .4]), weights_type)
    datasets = [
        dataset_ops.Dataset.from_tensors(np.int64(-1)),
        dataset_ops.Dataset.from_tensors(np.int64(1)).repeat(),
        dataset_ops.Dataset.range(10).repeat()
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=True)

    samples_list = self.getIteratorOutput(self.getNext(
        sample_dataset, requires_initialization=True))
    self.assertEqual(samples_list.count(-1), 1)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _weights_type_combinations()))
  def testSampleFromDatasetsSkippingEmptyDataset(self, weights_type):
    # [FUNCTION DOC]
    # Same datasets as the stopping test but with stop-on-empty disabled and taking one hundred
    # elements. Asserts exactly one hundred outputs and that the sentinel appears once (the
    # exhausted dataset is skipped afterwards).
    # Sampling skips the first dataset after it becomes empty.
    weights = _get_weights_of_type(np.asarray([.5, .1, .4]), weights_type)
    datasets = [
        dataset_ops.Dataset.from_tensors(np.int64(-1)),
        dataset_ops.Dataset.from_tensors(np.int64(1)).repeat(),
        dataset_ops.Dataset.range(10).repeat()
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=False).take(100)

    samples_list = self.getIteratorOutput(self.getNext(
        sample_dataset, requires_initialization=True))
    self.assertLen(samples_list, 100)
    self.assertEqual(samples_list.count(-1), 1)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _weights_type_combinations()))
  def testSampleFromDatasetsWithZeroWeight(self, weights_type):
    # [FUNCTION DOC]
    # Two two-element datasets with weights zero and one and stop-on-empty. Asserts only the
    # second dataset's values are produced (two ones), ending when it is exhausted.
    # Sampling stops when the second dataset is exhausted.
    weights = _get_weights_of_type(np.asarray([0., 1.]), weights_type)
    datasets = [
        dataset_ops.Dataset.from_tensors(-1).repeat(2),
        dataset_ops.Dataset.from_tensors(1).repeat(2)
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=True)
    self.assertDatasetProduces(sample_dataset, [1, 1],
                               requires_initialization=True)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _weights_type_combinations()))
  def testSampleFromEmptyDataset(self, weights_type):
    # [FUNCTION DOC]
    # An empty dataset with weight one and an endless one with weight zero, stop-on-empty on.
    # Asserts the result is empty.
    weights = _get_weights_of_type(np.asarray([1., 0.]), weights_type)
    datasets = [
        dataset_ops.Dataset.range(0),
        dataset_ops.Dataset.range(1).repeat()
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=True)
    self.assertDatasetProduces(sample_dataset, [],
                               requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSampleFromDatasetsSkippingDatasetsWithZeroWeight(self):
    # [FUNCTION DOC]
    # An endless dataset with weight zero and a one-element dataset with weight one, stop-on-empty
    # off. Asserts the zero-weight dataset is never used and the output is the single element.
    # Sampling skips the first dataset.
    weights = np.asarray([0., 1.])
    datasets = [
        dataset_ops.Dataset.from_tensors(-1).repeat(),
        dataset_ops.Dataset.from_tensors(1)
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=False)
    self.assertDatasetProduces(sample_dataset, [1])

  @combinations.generate(test_base.default_test_combinations())
  def testSampleFromDatasetsAllWeightsAreZero(self):
    # [FUNCTION DOC]
    # Two endless datasets with both weights zero and stop-on-empty off. Asserts the output is
    # empty.
    # Sampling skips both datasets.
    weights = np.asarray([0., 0.])
    datasets = [
        dataset_ops.Dataset.from_tensors(-1).repeat(),
        dataset_ops.Dataset.from_tensors(1).repeat()
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=False)
    self.assertDatasetProduces(sample_dataset, [])

  @combinations.generate(test_base.default_test_combinations())
  def testSampleFromDatasetsCardinality(self):
    # [FUNCTION DOC]
    # Samples from two endless datasets with default weights. Asserts the cardinality is infinite.
    ds1 = dataset_ops.Dataset.from_tensors([1.0]).repeat()
    ds2 = dataset_ops.Dataset.from_tensors([2.0]).repeat()
    ds = dataset_ops.Dataset.sample_from_datasets([ds1, ds2])
    self.assertEqual(self.evaluate(ds.cardinality()), dataset_ops.INFINITE)

  @combinations.generate(test_base.default_test_combinations())
  def testSampleFromDatasetsNested(self):
    # [FUNCTION DOC]
    # Samples from two datasets of windows (nested datasets), then flattens with flat-map and
    # reads one element. Only asserts that this works without error.
    ds1 = dataset_ops.Dataset.range(10).window(2)
    ds2 = dataset_ops.Dataset.range(10, 20).window(2)
    ds = dataset_ops.Dataset.sample_from_datasets([ds1, ds2],
                                                  weights=[0.3, 0.7])
    ds = ds.flat_map(lambda x: x)
    next_element = self.getNext(ds, requires_initialization=True)
    self.evaluate(next_element())

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(rerandomize=[None, True, False])))
  def testSampleFromDatasetsRerandomizeEachIterationEpochs(self, rerandomize):
    # [FUNCTION DOC]
    # Eager only. Skipped if a rerandomize value is given but forward compatibility is not yet
    # reached. Samples two ranges with a fixed seed and the parameterized rerandomize flag and
    # reads two epochs from the same dataset. Asserts the epochs differ when rerandomizing is
    # true and are equal otherwise (false or unset).
    if rerandomize is not None and not tf_compat.forward_compatible(
        2022, 12, 17):
      self.skipTest(
          "target functionality not available due to forward compatibility")
    dataset1 = dataset_ops.Dataset.range(0, 10)
    dataset2 = dataset_ops.Dataset.range(100, 110)
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        [dataset1, dataset2],
        seed=42,
        weights=[0.5, 0.5],
        stop_on_empty_dataset=True,
        rerandomize_each_iteration=rerandomize)
    first_epoch = self.getDatasetOutput(sample_dataset)
    second_epoch = self.getDatasetOutput(sample_dataset)

    if rerandomize:
      self.assertNotEqual(first_epoch, second_epoch)
    else:
      self.assertEqual(first_epoch, second_epoch)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(rerandomize=[None, True, False])))
  def testSampleFromDatasetsRerandomizeRepeatEpochs(self, rerandomize):
    # [FUNCTION DOC]
    # Like the iteration-epochs test but the sampled dataset is repeated twice and the single
    # output is split in half into two epochs. Asserts halves differ when rerandomizing is true
    # and are equal otherwise. Skipped if forward compatibility is not reached.
    if rerandomize is not None and not tf_compat.forward_compatible(
        2022, 12, 17):
      self.skipTest(
          "target functionality not available due to forward compatibility")
    dataset1 = dataset_ops.Dataset.range(0, 10)
    dataset2 = dataset_ops.Dataset.range(100, 110)
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        [dataset1, dataset2],
        seed=42,
        weights=[0.5, 0.5],
        stop_on_empty_dataset=True,
        rerandomize_each_iteration=rerandomize)
    sample_dataset = sample_dataset.repeat(2)
    epochs = self.getDatasetOutput(sample_dataset, requires_initialization=True)
    first_epoch = epochs[:len(epochs) // 2]
    second_epoch = epochs[len(epochs) // 2:]

    if rerandomize:
      self.assertNotEqual(first_epoch, second_epoch)
    else:
      self.assertEqual(first_epoch, second_epoch)

  @combinations.generate(
      combinations.times(test_base.v2_eager_only_combinations(),
                         combinations.combine(rerandomize=[None, True, False])))
  def testSampleFromDatasetsRerandomizeInsideFunction(self, rerandomize):
    # [FUNCTION DOC]
    # Version-2 eager only. Builds the sampled dataset inside a traced function and reads two
    # epochs from the result. Asserts epochs differ when rerandomize is true and match otherwise.
    # Skipped if forward compatibility is not reached.
    if rerandomize is not None and not tf_compat.forward_compatible(
        2022, 12, 17):
      self.skipTest(
          "target functionality not available due to forward compatibility")
    @def_function.function
    def make_dataset():
      # [FUNCTION DOC]
      # HELPER: traced function returning a seeded equal-weight sample of two ranges with
      # stop-on-empty and the parameterized rerandomize flag.
      dataset1 = dataset_ops.Dataset.range(0, 10)
      dataset2 = dataset_ops.Dataset.range(100, 110)
      sample_dataset = dataset_ops.Dataset.sample_from_datasets(
          [dataset1, dataset2],
          seed=42,
          weights=[0.5, 0.5],
          stop_on_empty_dataset=True,
          rerandomize_each_iteration=rerandomize)
      return sample_dataset

    sample_dataset = make_dataset()
    first_epoch = self.getDatasetOutput(sample_dataset)
    second_epoch = self.getDatasetOutput(sample_dataset)

    if rerandomize:
      self.assertNotEqual(first_epoch, second_epoch)
    else:
      self.assertEqual(first_epoch, second_epoch)

  @combinations.generate(test_base.default_test_combinations())
  def testErrors(self):
    # [FUNCTION DOC]
    # Asserts argument validation: a weights list whose length differs from the number of
    # datasets, integer weights, datasets with incompatible element types, and an empty datasets
    # list each raise the expected value or type error with the matching message.
    with self.assertRaisesRegex(ValueError, r"should have the same length"):
      dataset_ops.Dataset.sample_from_datasets(
          [dataset_ops.Dataset.range(10),
           dataset_ops.Dataset.range(20)],
          weights=[0.25, 0.25, 0.25, 0.25])

    with self.assertRaisesRegex(TypeError, "`tf.float32` or `tf.float64`"):
      dataset_ops.Dataset.sample_from_datasets(
          [dataset_ops.Dataset.range(10),
           dataset_ops.Dataset.range(20)],
          weights=[1, 1])

    with self.assertRaisesRegex(TypeError, "must have compatible"):
      dataset_ops.Dataset.sample_from_datasets([
          dataset_ops.Dataset.from_tensors(0),
          dataset_ops.Dataset.from_tensors(0.0)
      ])

    with self.assertRaisesRegex(
        ValueError, r"Invalid `datasets`. `datasets` should not be empty."):
      dataset_ops.Dataset.sample_from_datasets(datasets=[], weights=[])


class SampleFromDatasetsCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                       parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for sample_from_datasets using the shared checkpoint base.

  def _build_dataset(self, probs, num_samples, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint test.
    # probs is a list of selection probabilities, one endless dataset per entry (each yielding its
    # index). Samples with a fixed seed, takes num_samples elements and, if options (optional)
    # is given, applies them. Returns the dataset.
    datasets = [
        dataset_ops.Dataset.from_tensors(i).repeat(None)
        for i in range(len(probs))
    ]
    dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, probs, seed=1813)
    dataset = dataset.take(num_samples)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Sets the symbolic-checkpoint option as parameterized and runs the verification callback on a
    # two-way equal-probability sample of one hundred elements, expecting one hundred outputs.
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self,
        lambda: self._build_dataset([0.5, 0.5], 100, options),
        num_outputs=100)


if __name__ == "__main__":
  test.main()
