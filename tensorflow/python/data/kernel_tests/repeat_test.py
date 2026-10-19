# [MODULE DOC]
# Tests for the dataset repeat transformation. Covers finite, infinite, empty and nested repeats,
# the name argument, checkpointing (with and without symbolic checkpoints), random access by
# index (including the failure for infinite cardinality) and interaction with the global shuffle
# op. Test classes only; no import-time side effects.
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
"""Tests for `tf.data.Dataset.repeat()`."""
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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class RepeatTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests for repeat: finite, infinite and composed repeats, and the name argument.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(count=[0, 3, 7])))
  def testFiniteRepeat(self, count):
    # [FUNCTION DOC]
    # Repeats a single three-component element count times (zero, three or seven). Asserts the
    # static component shapes are preserved and the output is the element repeated count times.
    """Test a dataset that repeats its input multiple times."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    dataset = dataset_ops.Dataset.from_tensors(components).repeat(count)
    self.assertEqual(
        [c.shape for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    self.assertDatasetProduces(dataset, [components] * count)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteRepeat(self):
    # [FUNCTION DOC]
    # Repeats an element with count minus one (infinite). Asserts shapes are preserved and the
    # first seventeen outputs equal the element. Cannot prove infiniteness.
    # NOTE(mrry): There's not a good way to test that the sequence is infinite.
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    dataset = dataset_ops.Dataset.from_tensors(components).repeat(-1)
    self.assertEqual(
        [c.shape for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    get_next = self.getNext(dataset)
    for _ in range(17):
      results = self.evaluate(get_next())
      for component, result_component in zip(components, results):
        self.assertAllEqual(component, result_component)

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatRepeat(self):
    # [FUNCTION DOC]
    # Applies repeat seven times then fourteen times to one element. Asserts shapes are preserved
    # and the output has ninety-eight copies of the element.
    """Test the composition of repeat datasets."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    inner_count, outer_count = 7, 14

    dataset = dataset_ops.Dataset.from_tensors(components).repeat(
        inner_count).repeat(outer_count)
    self.assertEqual(
        [c.shape for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    self.assertDatasetProduces(dataset,
                               [components] * (inner_count * outer_count))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Repeats a single-element dataset once with a custom name; asserts the output is just that
    # element.
    dataset = dataset_ops.Dataset.from_tensors(42).repeat(1, name="repeat")
    self.assertDatasetProduces(dataset, [42])


class RepeatDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                  parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for the repeat dataset using the shared checkpoint base.

  def _build_repeat_dataset(self,
                            num_elements,
                            num_epochs,
                            num_outputs=None,
                            options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint tests.
    # Creates a range of num_elements repeated num_epochs times (negative means forever). If
    # num_outputs (optional) is truthy, zips it with a range of that length to bound an infinite
    # dataset. If options (optional) is given, applies them. Returns the dataset.
    dataset = dataset_ops.Dataset.range(num_elements).repeat(num_epochs)
    if num_outputs:
      range_dataset = dataset_ops.Dataset.range(num_outputs)
      dataset = dataset_ops.Dataset.zip((dataset, range_dataset))
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def testFiniteRepeat(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Checkpoint test: ten elements repeated ten times with the symbolic-checkpoint option as
    # parameterized; the verification callback expects one hundred outputs.
    num_elements = 10
    num_epochs = 10
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self,
        lambda: self._build_repeat_dataset(
            num_elements, num_epochs, options=options),
        num_outputs=(num_elements * num_epochs))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def testEmptyRepeat(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Checkpoint test: repeat zero times; the verification callback expects zero outputs.
    num_elements = 10
    num_epochs = 0
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self,
        lambda: self._build_repeat_dataset(
            num_elements, num_epochs, options=options),
        num_outputs=0)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def testInfiniteRepeat(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Checkpoint test: ten elements repeated forever, bounded by zipping with a range of one
    # hundred; the verification callback expects one hundred outputs.
    num_elements = 10
    num_epochs = -1
    num_outputs = 100
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self,
        lambda: self._build_repeat_dataset(
            num_elements, num_epochs, num_outputs=num_outputs, options=options),
        num_outputs=num_outputs)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def testInfiniteEmptyRepeat(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Checkpoint test: repeating an empty dataset forever; the callback expects zero outputs.
    num_elements = 0
    num_epochs = -1
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self,
        lambda: self._build_repeat_dataset(
            num_elements, num_epochs, options=options),
        num_outputs=0)


class RepeatRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for index-based random access into repeat datasets via the random access helper.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 6, 7])))
  def testInvalidIndex(self, index):
    # [FUNCTION DOC]
    # A three-element dataset repeated twice (six elements): negative or too-large indices must
    # raise an out-of-range error.
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3]).repeat(2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 0])))
  def testEmptyDataset(self, index):
    # [FUNCTION DOC]
    # Random access into a repeated empty dataset at index minus one or zero must raise an
    # out-of-range error.
    dataset = dataset_ops.Dataset.from_tensor_slices([]).repeat(2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(elements=[0, 5, 10],
                                              count=[0, 3, 8])))
  def testFiniteRepeat(self, elements, count):
    # [FUNCTION DOC]
    # For a range of the given size repeated count times, random access at every index equals the
    # tiled expected sequence.
    dataset = dataset_ops.Dataset.range(elements).repeat(count)
    expected_dataset = np.tile(
        np.arange(
            start=0, stop=elements, step=1, dtype=dtypes.int64.as_numpy_dtype),
        count)
    for i in range(elements * count):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)),
          expected_dataset[i])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              elements=[0, 3, 5], count_1=[0, 1, 2], count_2=[3, 4, 5])))
  def testRepeatRepeat(self, elements, count_1, count_2):
    # [FUNCTION DOC]
    # For a range repeated count_1 times and then count_2 times, random access at every index
    # equals the tiled expected sequence of length elements times both counts.
    dataset = dataset_ops.Dataset.range(elements).repeat(count_1).repeat(
        count_2)
    expected_dataset = np.tile(
        np.arange(
            start=0, stop=elements, step=1, dtype=dtypes.int64.as_numpy_dtype),
        count_1 * count_2)
    for i in range(elements * count_1 * count_2):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)),
          expected_dataset[i])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(elements=[3, 5], count=[None, -1, -2])))
  def testInfiniteRepeat(self, elements, count):
    # [FUNCTION DOC]
    # Repeat with a null or negative count gives infinite cardinality; random access at index
    # zero must raise a failed-precondition error.
    dataset = dataset_ops.Dataset.range(elements).repeat(count=count)

    # Datasets with infinite cardinality do not support random access.
    with self.assertRaises(errors.FailedPreconditionError):
      self.evaluate(random_access.at(dataset, index=0))


class RepeatGlobalShuffleTest(
    test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for applying the global shuffle op to repeated datasets.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[41],
              repetitions=[1, 27],
              seed=[None, 19],
              reshuffle_each_iteration=[True, False])))
  def testRepeat(
      self,
      dataset_range: int,
      repetitions: int,
      seed: Optional[int],
      reshuffle_each_iteration: bool):
    # [FUNCTION DOC]
    # Repeats a range, prefetches, applies a global shuffle with the given seed (optional) and
    # reshuffle flag. Asserts the output has the same multiset of values as the unshuffled
    # repetition but a different order overall, that each repetition after the first is itself
    # shuffled (differs from sorted order and from the previous repetition).
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = dataset.repeat(repetitions)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    expected = list(range(dataset_range)) * repetitions
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)

    output_per_iteration = [
        dataset_output[i : i + dataset_range]
        for i in range(0, len(dataset_output), dataset_range)]
    # All sub-ranges should be shuffled.
    for i in range(1, repetitions):
      self.assertNotEqual(output_per_iteration[i], list(range(dataset_range)))
      self.assertNotEqual(output_per_iteration[i], output_per_iteration[i - 1])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(repetitions=[None, 0])))
  def testInvalidDataset(self, repetitions: Optional[int]):
    # [FUNCTION DOC]
    # Repeating with a null or zero count (infinite or unsupported) then global shuffling must
    # raise a failed-precondition error saying the repeat does not support random access.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.repeat(repetitions)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)

    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        r"`repeat.*` does not support random access of tf.data datasets."):
      dataset = global_shuffle_op._global_shuffle(dataset)
      self.getDatasetOutput(dataset, requires_initialization=True)


class RepeatGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for repeat followed by the global shuffle op.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[41],
              repetitions=[1, 27],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testRepeat(
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      repetitions: int,
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Runs the verification callback on a repeated, prefetched, globally shuffled range (fixed
    # seed, parameterized reshuffle and symbolic-checkpoint options), expecting range size times
    # repetitions outputs; items are compared order-insensitively when reshuffling each iteration.

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: builds the pipeline under test: range, repeat, prefetch with auto-tuned buffer,
      # global shuffle with seed forty-two, then applies the symbolic-checkpoint option. Returns
      # the dataset.
      dataset = dataset_ops.Dataset.range(dataset_range)
      dataset = dataset.repeat(repetitions)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=dataset_range * repetitions,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
