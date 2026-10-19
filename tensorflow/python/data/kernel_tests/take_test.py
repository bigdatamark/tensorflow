# [MODULE DOC]
# Unit tests for the dataset take transformation, which keeps only the first N elements of a
# dataset (a negative count means keep everything). Covers basic behavior, naming, checkpointing
# (with and without symbolic checkpoints), random access by index, and global shuffling of a
# taken dataset. Not a helper module; it defines test classes only and runs the TensorFlow test
# main when executed directly. No import-time side effects beyond imports.
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
"""Tests for `tf.data.Dataset.take()`."""

from typing import Callable, Optional

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class TakeTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Basic behavior tests for the take transformation on small in-memory datasets.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(count=[-1, 0, 4, 10, 25])))
  def testBasic(self, count):
    # [FUNCTION DOC]
    # Parameterized over count values (negative, zero, smaller, equal, larger than the data size).
    # Builds a dataset from ten-element array slices and applies take with count. Asserts that the
    # legacy output shapes equal the per-element component shapes and that the dataset produces
    # the first min(count, 10) elements, or all ten when count is negative.
    components = (np.arange(10),)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).take(count)
    self.assertEqual(
        [c.shape[1:] for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    num_output = min(count, 10) if count != -1 else 10
    self.assertDatasetProduces(
        dataset, [tuple(components[0][i:i + 1]) for i in range(num_output)])

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Checks that take accepts a name argument: takes one element from a single-element dataset
    # holding 42 and asserts the dataset yields exactly that one value.
    dataset = dataset_ops.Dataset.from_tensors(42).take(1, name="take")
    self.assertDatasetProduces(dataset, [42])


class TakeDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for a range dataset followed by take, using the shared
  # checkpoint verification fixture from the checkpoint test base.

  def _build_take_dataset(self, count, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint test.
    # Returns a dataset of range 100 limited by take with the given count. If options (an optional
    # dataset options object) is given, it is applied to the dataset before returning.
    dataset = dataset_ops.Dataset.range(100).take(count)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True]),
          combinations.combine(count=[50], num_outputs=[50]) +
          combinations.combine(count=[200, 100, -1], num_outputs=[100]) +
          combinations.combine(count=[0], num_outputs=[0])))
  def test(self, verify_fn, symbolic_checkpoint, count, num_outputs):
    # [FUNCTION DOC]
    # Parameterized checkpoint test. Inputs: verify_fn (the checkpoint verification callback
    # supplied by the combinations), symbolic_checkpoint (whether symbolic checkpointing is
    # enabled in the options), count (take count, including larger than data, negative and
    # zero) and num_outputs (expected number of elements). Builds the dataset with the options
    # and asks verify_fn to check that saving and restoring yields num_outputs elements.
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(self, lambda: self._build_take_dataset(count, options),
              num_outputs)


class TakeRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests random access (reading an element by index) on datasets produced by take.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 3, 4])))
  def testInvalidIndex(self, index):
    # [FUNCTION DOC]
    # Parameterized over invalid indices (negative or at/after the taken length). Takes three
    # elements of a ten-element range and asserts that random access at the index raises an
    # out-of-range error.
    dataset = dataset_ops.Dataset.range(10).take(3)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-2, 0, 1])))
  def testEmptyDataset(self, index):
    # [FUNCTION DOC]
    # Parameterized over several indices. Takes five elements from an empty dataset and asserts
    # that random access at any index raises an out-of-range error.
    dataset = dataset_ops.Dataset.from_tensor_slices([]).take(5)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(count=[-1, 0, 4, 10, 25])))
  def testBasic(self, count):
    # [FUNCTION DOC]
    # Parameterized over take counts. For each valid output position, asserts random access
    # returns the element equal to its index; then asserts access at the first position past the
    # number of outputs (min of count and ten, or ten if count is negative) raises out-of-range.
    dataset = dataset_ops.Dataset.range(10).take(count)
    num_output = min(count, 10) if count != -1 else 10
    for i in range(num_output):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)), i)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=num_output))


class TakeGlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that a dataset using take (plus prefetch and optional repeat) can be globally shuffled.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[100],
              count=[20, 200],
              repetitions=[1, 2],
              seed=[None, 42],
              reshuffle_each_iteration=[True, False])))
  def test(
      self,
      dataset_range: int,
      count: int,
      repetitions: int,
      seed: Optional[int],
      reshuffle_each_iteration: bool):
    # [FUNCTION DOC]
    # Parameterized over range size, take count, repetitions, seed (optional) and whether to
    # reshuffle each iteration. Builds range, take, prefetch with autotuned buffer, optional
    # repeat, then applies the internal global shuffle. Asserts the output has the same elements
    # as the expected taken sequence repeated (ignoring order), that the order differs from the
    # unshuffled order, and that the output length equals the dataset cardinality.
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = dataset.take(count)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    if repetitions > 1:
      dataset = dataset.repeat(repetitions)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    expected = list(range(0, min(count, dataset_range))) * repetitions
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)
    self.assertLen(dataset_output, self.evaluate(dataset.cardinality()))


class TakeGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for a globally shuffled dataset built on take, prefetch and optional repeat.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              count=[2, 20],
              repetitions=[1, 2],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def test(
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      count: int,
      repetitions: int,
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Parameterized checkpoint test for a globally shuffled taken dataset. Inputs: verify_fn
    # (checkpoint verification callback), range size, take count, repetitions, whether to
    # reshuffle each iteration and whether symbolic checkpointing is on. Asks verify_fn to check
    # save/restore produces min(count, range) times repetitions elements, compared ignoring order
    # when reshuffling each iteration.
    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: dataset factory for the checkpoint verification.
      # Builds range, take, prefetch, optional repeat, global shuffle with a fixed seed, and
      # attaches options with the symbolic checkpoint setting. Returns the dataset.
      dataset = dataset_ops.Dataset.range(dataset_range)
      dataset = dataset.take(count)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      if repetitions > 1:
        dataset = dataset.repeat(repetitions)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=min(count, dataset_range) * repetitions,
        assert_items_equal=reshuffle_each_iteration,
    )


if __name__ == "__main__":
  test.main()
