# [MODULE DOC]
# Kernel tests for the random dataset source. Covers determinism with respect to global and
# per-dataset seeds (including after a graph round trip), the option to reshuffle on each
# iteration in graph, eager, repeat and in-function settings (including the version-one
# warning that the option is ignored), the name argument, and iterator checkpointing. No
# import-time side effects other than the test entry point.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.random()`."""
import warnings

from absl.testing import parameterized

from tensorflow.python import tf2
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test


class RandomTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the random dataset source: seeding, per-iteration re-randomization and naming.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(global_seed=[None, 10], local_seed=[None, 20])))
  def testDeterminism(self, global_seed, local_seed):
    # [FUNCTION DOC]
    # Scenario: a random dataset of ten elements is built with an optional global seed (set
    # process-wide) and optional local seed, read, graph-round-tripped, and read again.
    # Expected: both outputs are equal when at least one seed is given; otherwise they differ
    # (a tiny chance of equality is accepted). Side effect: sets the global random seed.
    expect_determinism = (global_seed is not None) or (local_seed is not None)

    random_seed.set_random_seed(global_seed)
    ds = dataset_ops.Dataset.random(seed=local_seed).take(10)

    output_1 = self.getDatasetOutput(ds, requires_initialization=True)
    ds = self.graphRoundTrip(ds)
    output_2 = self.getDatasetOutput(ds, requires_initialization=True)

    if expect_determinism:
      self.assertEqual(output_1, output_2)
    else:
      # Technically not guaranteed since the two randomly-chosen int64 seeds
      # could match, but that is sufficiently unlikely (1/2^128 with perfect
      # random number generation).
      self.assertNotEqual(output_1, output_2)

  @combinations.generate(
      combinations.times(test_base.graph_only_combinations(),
                         combinations.combine(rerandomize=[None, True, False])))
  def testRerandomizeEachIterationEpochsIgnored(self, rerandomize):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inner condition repeats the outer check on the re-randomize flag, so the
    # nested test is redundant (harmless).
    # Graph-only scenario: a seeded random dataset of ten elements with the re-randomize option
    # unset, true or false is read twice, capturing warnings during construction. Expected: the
    # two epochs are identical (the option is ignored in graph mode); when the option is true
    # and the second API version is not enabled, a warning saying the option is not supported in
    # version one is emitted.
    with warnings.catch_warnings(record=True) as w:
      dataset = dataset_ops.Dataset.random(
          seed=42,
          rerandomize_each_iteration=rerandomize,
          name="random").take(10)
    first_epoch = self.getDatasetOutput(dataset, requires_initialization=True)
    second_epoch = self.getDatasetOutput(dataset, requires_initialization=True)
    if rerandomize:
      if not tf2.enabled() and rerandomize:
        found_warning = False
        for warning in w:
          if ("In TF 1, the `rerandomize_each_iteration=True` option" in
              str(warning)):
            found_warning = True
            break
        self.assertTrue(found_warning)

    self.assertEqual(first_epoch, second_epoch)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(rerandomize=[None, True, False])))
  def testRerandomizeEachIterationEpochs(self, rerandomize):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected-equality expression is the negation of the flag or a null check,
    # which is always false whenever it is evaluated (flag truthy), so the check effectively
    # asserts that the two epochs differ; the written form is confusing.
    # Eager-only scenario: a seeded random dataset of ten elements is read in two epochs.
    # Expected: with re-randomization on, the epochs differ; otherwise they are identical.
    dataset = dataset_ops.Dataset.random(
        seed=42, rerandomize_each_iteration=rerandomize, name="random").take(10)
    first_epoch = self.getDatasetOutput(dataset)
    second_epoch = self.getDatasetOutput(dataset)

    if rerandomize:
      self.assertEqual(first_epoch == second_epoch,
                       not rerandomize or rerandomize is None)
    else:
      self.assertEqual(first_epoch, second_epoch)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(rerandomize=[None, True, False])))
  def testRerandomizeRepeatEpochs(self, rerandomize):
    # [FUNCTION DOC]
    # KNOWN ISSUE: same confusing expected-equality expression as the eager epochs test; it is
    # always false when evaluated, so the check asserts the epochs differ.
    # Scenario: a seeded random dataset of ten elements repeated twice is read as two epochs of
    # ten. Expected: with re-randomization on, the epochs differ; otherwise they are identical.
    dataset = dataset_ops.Dataset.random(
        seed=42, rerandomize_each_iteration=rerandomize, name="random").take(10)
    dataset = dataset.repeat(2)
    next_element = self.getNext(dataset, requires_initialization=True)
    first_epoch = []
    for _ in range(10):
      first_epoch.append(self.evaluate(next_element()))
    second_epoch = []
    for _ in range(10):
      second_epoch.append(self.evaluate(next_element()))

    if rerandomize:
      self.assertEqual(first_epoch == second_epoch,
                       not rerandomize or rerandomize is None)
    else:
      self.assertEqual(first_epoch, second_epoch)

  @combinations.generate(
      combinations.times(test_base.v2_eager_only_combinations(),
                         combinations.combine(rerandomize=[None, True, False])))
  def testRerandomizeInsideFunction(self, rerandomize):
    # [FUNCTION DOC]
    # KNOWN ISSUE: same confusing expected-equality expression as the eager epochs test; it is
    # always false when evaluated, so the check asserts the epochs differ.
    # Scenario (version-two eager only): the seeded random dataset is created inside a traced
    # function and read in two epochs. Expected: with re-randomization on, the epochs differ;
    # otherwise they are identical.

    @def_function.function
    def make_dataset():
      # [FUNCTION DOC]
      # HELPER: traced function; builds a seeded random dataset of ten elements with the
      # enclosing test's re-randomize setting and returns it.
      dataset = dataset_ops.Dataset.random(
          seed=42,
          rerandomize_each_iteration=rerandomize,
          name="random").take(10)
      return dataset

    dataset = make_dataset()
    first_epoch = self.getDatasetOutput(dataset)
    second_epoch = self.getDatasetOutput(dataset)

    if rerandomize:
      self.assertEqual(first_epoch == second_epoch,
                       not rerandomize or rerandomize is None)
    else:
      self.assertEqual(first_epoch, second_epoch)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Scenario: a seeded, named random dataset limited to one element and mapped to a constant.
    # Expected: the dataset produces 42 (requires iterator initialization in graph mode).
    dataset = dataset_ops.Dataset.random(
        seed=42, name="random").take(1).map(lambda _: 42)
    self.assertDatasetProduces(dataset, expected_output=[42],
                               requires_initialization=True)


class RandomCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint test case for the random dataset; limits the infinite source with take and uses
  # the shared checkpoint test base to verify save and restore of iterator state.

  def _build_random_dataset(
      self,
      num_elements=10,
      seed=None,
      rerandomize_each_iteration=None):
    # [FUNCTION DOC]
    # HELPER: builds the pipeline for the checkpoint test. Parameters: num_elements (how many
    # elements to keep, default ten), seed (optional integer seed) and
    # rerandomize_each_iteration (optional flag). The random source is infinite, so the result
    # is truncated to a finite dataset, which is returned.
    dataset = dataset_ops.Dataset.random(
        seed=seed, rerandomize_each_iteration=rerandomize_each_iteration)
    # Checkpoint tests need the test dataset to be finite whereas `random` is
    # infinite. Use `take` to limit the number of elements.
    return dataset.take(num_elements)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              rerandomize_each_iteration=[True, False])))

  def test(self, verify_fn, rerandomize_each_iteration):
    # [FUNCTION DOC]
    # Scenario: checkpoint verification of a seeded (55) random pipeline of ten elements with
    # re-randomization on or off. Parameters: verify_fn (checkpoint verification strategy from
    # the base class) and the re-randomize flag. Expected: verification passes for ten outputs.
    seed = 55
    num_elements = 10
    # pylint: disable=g-long-lambda
    verify_fn(
        self,
        lambda: self._build_random_dataset(
            seed=seed,
            num_elements=num_elements,
            rerandomize_each_iteration=rerandomize_each_iteration),
        num_elements)


if __name__ == "__main__":
  test.main()
