# [MODULE DOC]
# Tests for the private weighted flat-map dataset transformation, which interleaves elements from
# several input datasets in proportion to given weights. Covers basic output, validation errors
# (insufficient cardinality, infinite or unknown cardinality, incompatible element types, zero
# weight), interaction with global shuffling (currently skipped because of a batch dataset
# incompatibility), and checkpointing (also skipped). No import-time side effects.
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.weighted_flat_map()`."""

from typing import Callable
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import weighted_flat_map_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class WeightedFlatMapTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional and error-case tests for the weighted flat-map transformation using three range
  # datasets of ten elements each.

  @combinations.generate(test_base.default_test_combinations())
  def testWeightedFlatMap(self):
    # [FUNCTION DOC]
    # Combines ranges 0-9, 10-19 and 20-29 with weights 1, 1, 2 and asserts the output: the first
    # five of the first, the first five of the second, then all ten of the third.
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    dataset = weighted_flat_map_op._weighted_flat_map(
        [dataset1, dataset2, dataset3], np.asarray([1, 1, 2]))
    self.assertDatasetProduces(
        dataset, expected_output=list(range(5)) + list(range(10, 15)) +
        list(range(20, 30)))

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidCardinality(self):
    # [FUNCTION DOC]
    # With weights 1, 1, 100 the third input (10 elements) is too small; asserts an
    # invalid-argument error saying the input needs at least a minimum number of elements when the
    # output is consumed.
    dataset1 = dataset_ops.Dataset.range(100)
    dataset2 = dataset_ops.Dataset.range(100, 200)
    dataset3 = dataset_ops.Dataset.range(200, 210)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "Input.*needs to have at least."):
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3], np.asarray([1, 1, 100]))
      self.getDatasetOutput(dataset, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteCardinality(self):
    # [FUNCTION DOC]
    # One input is an infinitely repeated dataset and no weights are given; asserts an
    # invalid-argument error stating that input cardinalities must be known.
    dataset1 = dataset_ops.Dataset.range(10).repeat()
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Cardinalities of the inputs must be known."):
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3])
      self.getDatasetOutput(dataset, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyInputDatasets(self):
    # [FUNCTION DOC]
    # One input is built from an empty list (float element type) while others are integer ranges;
    # asserts a type error about incompatible dataset elements.
    dataset1 = dataset_ops.Dataset.from_tensor_slices([])
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    with self.assertRaisesRegex(
        TypeError,
        "Incompatible dataset elements"):
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3])
      self.getDatasetOutput(dataset, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testZeroWeight(self):
    # [FUNCTION DOC]
    # A weight of zero is invalid; asserts an invalid-argument error saying weights must be greater
    # than zero.
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "`weights` must be greater than 0.0"):
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3], [0, 1.0, 1.0])
      self.getDatasetOutput(dataset, requires_initialization=True)


@unittest.skip("TODO(b/325112575): Fix incompatibility with batch dataset")
class GlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the class is unconditionally skipped (decorated with a skip referencing an
  # incompatibility with batch datasets), so none of its tests run. Its docstring is also generic
  # and does not mention the weighted flat-map focus.
  # Tests combining weighted flat-map with global shuffling of inputs and/or output.
  """Tests for global shuffling of tf.data datasets."""

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledOutput(self):
    # [FUNCTION DOC]
    # Weighted flat-map of three prefetched ranges followed by global shuffle of the output;
    # asserts the output has the same elements as the unshuffled expectation (any order).
    dataset1 = dataset_ops.Dataset.range(10).prefetch(
        buffer_size=dataset_ops.AUTOTUNE)
    dataset2 = dataset_ops.Dataset.range(10, 20).prefetch(
        buffer_size=dataset_ops.AUTOTUNE)
    dataset3 = dataset_ops.Dataset.range(20, 30).prefetch(
        buffer_size=dataset_ops.AUTOTUNE)
    dataset = weighted_flat_map_op._weighted_flat_map(
        [dataset1, dataset2, dataset3], np.asarray([0.25, 0.25, 0.5]))
    dataset = global_shuffle_op._global_shuffle(dataset)

    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(
        output, list(range(5)) + list(range(10, 15)) + list(range(20, 30)))

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledInputs(self):
    # [FUNCTION DOC]
    # Globally shuffles each input (fixed seed) before the weighted flat-map; asserts the first
    # five outputs come from the first input in non-sorted order, the next five from the second
    # input similarly, and the last ten are a shuffled permutation of the third input.
    dataset1 = dataset_ops.Dataset.range(10).prefetch(
        buffer_size=dataset_ops.AUTOTUNE)
    dataset2 = dataset_ops.Dataset.range(10, 20).prefetch(
        buffer_size=dataset_ops.AUTOTUNE)
    dataset3 = dataset_ops.Dataset.range(20, 30).prefetch(
        buffer_size=dataset_ops.AUTOTUNE)
    dataset1 = global_shuffle_op._global_shuffle(dataset1, seed=42)
    dataset2 = global_shuffle_op._global_shuffle(dataset2, seed=42)
    dataset3 = global_shuffle_op._global_shuffle(dataset3, seed=42)
    dataset = weighted_flat_map_op._weighted_flat_map(
        [dataset1, dataset2, dataset3], np.asarray([0.25, 0.25, 0.5]))

    output = self.getDatasetOutput(dataset, requires_initialization=True)
    # Verifies that the first 5 elements are from `dataset1` in a random order.
    self.assertFalse(set(output[:5]).issubset(set(range(5))))
    self.assertTrue(set(output[:5]).issubset(set(range(10))))
    # Verifies that the second 5 elements are from `dataset2` in a random order.
    self.assertFalse(set(output[5:10]).issubset(set(range(10, 15))))
    self.assertTrue(set(output[5:10]).issubset(set(range(10, 20))))
    # Verifies that the last 10 elements are from `dataset3` in a random order.
    self.assertCountEqual(output[10:], range(20, 30))
    self.assertNotEqual(output[10:], range(20, 30))

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledInputsAndOutput(self):
    # [FUNCTION DOC]
    # Shuffles the inputs and then the combined output; asserts the output blocks are no longer
    # confined to the original source segments, while the sorted output still partitions into
    # five, five and ten elements from the three sources.
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    dataset1 = global_shuffle_op._global_shuffle(dataset1, seed=42)
    dataset2 = global_shuffle_op._global_shuffle(dataset2, seed=42)
    dataset3 = global_shuffle_op._global_shuffle(dataset3, seed=42)
    dataset = weighted_flat_map_op._weighted_flat_map(
        [dataset1, dataset2, dataset3], np.asarray([0.25, 0.25, 0.5]))
    dataset = global_shuffle_op._global_shuffle(dataset, seed=42)
    output = self.getDatasetOutput(dataset, requires_initialization=True)
    # Verifies that not all first 5 elements are from `dataset1`.
    self.assertFalse(set(output[:5]).issubset(set(range(10))))
    # Verifies that not all second 5 elements are from `dataset2`.
    self.assertFalse(set(output[5:10]).issubset(set(range(10, 20))))
    # Verifies that not all last 10 elements are from `dataset3`.
    self.assertFalse(set(output[10:]).issubset(set(range(20, 30))))

    sorted_output = sorted(output)
    # Verifies that there are 5 elements from dataset1
    self.assertTrue(set(sorted_output[:5]).issubset(set(range(10))))
    # Verifies that there are 5 elements from dataset2
    self.assertTrue(set(sorted_output[5:10]).issubset(set(range(10, 20))))
    # Verifies that there are 10 elements from dataset3
    self.assertTrue(set(sorted_output[10:]).issubset(set(range(20, 30))))

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledWithBatch(self):
    # [FUNCTION DOC]
    # Batches two inputs (size 2, dropping remainder), combines them with equal weights, shuffles
    # globally, and asserts the shuffled output has the same length as the unshuffled output.
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset1 = dataset1.batch(2, drop_remainder=True)
    dataset2 = dataset2.batch(2, drop_remainder=True)
    dataset = weighted_flat_map_op._weighted_flat_map(
        [dataset1, dataset2], np.asarray([0.5, 0.5])
    )
    shuffled_dataset = global_shuffle_op._global_shuffle(dataset, seed=42)
    shuffled_output = self.getDatasetOutput(
        shuffled_dataset, requires_initialization=True
    )
    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertLen(shuffled_output, len(output))


@unittest.skip("TODO(b/325112575): Fix incompatibility with batch dataset")
class WeightedFlatMapGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: unconditionally skipped (decorated with a skip referencing a batch dataset
  # incompatibility), so these tests never run.
  # Checkpoint tests of weighted flat-map, with and without global shuffle, for symbolic and
  # regular checkpoints.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[True, False])))
  def testWeightedFlatMap(
      self,
      verify_fn: Callable[..., None],
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Parameterized by the checkpoint verifier and whether symbolic checkpointing is enabled.
    # Verifies save/restore of the weighted flat-map dataset (three ranges, weights .25/.25/.5)
    # with default optimizations disabled, expecting 20 outputs.

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: builds the weighted flat-map dataset with default optimizations off and the
      # symbolic checkpoint option set from the enclosing parameter; returns the dataset.
      dataset1 = dataset_ops.Dataset.range(10)
      dataset2 = dataset_ops.Dataset.range(10, 20)
      dataset3 = dataset_ops.Dataset.range(20, 30)
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3], np.asarray([0.25, 0.25, 0.5]))
      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(self, _build_dataset, num_outputs=20)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testGlobalshuffle(
      self,
      verify_fn: Callable[..., None],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Parameterized by the checkpoint verifier, whether each iteration reshuffles, and whether
    # symbolic checkpointing is on. Verifies save/restore of the weighted flat-map dataset followed
    # by a seeded global shuffle, expecting 20 outputs (compared ignoring order when reshuffling).

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: builds the weighted flat-map plus global shuffle dataset (seed 42) with default
      # optimizations off and the symbolic checkpoint option set; returns the dataset.
      dataset1 = dataset_ops.Dataset.range(10)
      dataset2 = dataset_ops.Dataset.range(10, 20)
      dataset3 = dataset_ops.Dataset.range(20, 30)
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3], np.asarray([0.25, 0.25, 0.5]))
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=20,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
