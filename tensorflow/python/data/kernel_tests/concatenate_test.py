# [MODULE DOC]
# Tests for the concatenate transformation of the tf.data API. Contains functional tests (output
# shapes, mismatched shapes/structures/keys/types, windows, naming), checkpoint tests, random
# access tests (indexing into concatenated datasets including empty inputs) and global-shuffle
# tests including checkpointing of nested and deep concatenations. No import-time side effects;
# running as a script starts the TensorFlow test runner.
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
"""Tests for `tf.data.Dataset.concatenate()."""
from typing import Callable, Tuple
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test


class ConcatenateTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests of concatenating two datasets. No shared fixtures.

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: concatenates a dataset of 4 elements with one of 5 elements, each made of three
    # components (two integer matrices and a float vector) of the same per-element shapes.
    # Asserts the legacy output shapes are [20], [15] and scalar, that the 9 elements come out in
    # order (first dataset then second), and that reading past the end raises an out-of-range
    # error. Returns nothing.
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0]))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0, 41.0]))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)
    concatenated = input_dataset.concatenate(dataset_to_concatenate)
    self.assertEqual(
        dataset_ops.get_legacy_output_shapes(concatenated),
        (tensor_shape.TensorShape([20]), tensor_shape.TensorShape([15]),
         tensor_shape.TensorShape([])))

    get_next = self.getNext(concatenated)

    for i in range(9):
      result = self.evaluate(get_next())
      if i < 4:
        for component, result_component in zip(input_components, result):
          self.assertAllEqual(component[i], result_component)
      else:
        for component, result_component in zip(to_concatenate_components,
                                               result):
          self.assertAllEqual(component[i - 4], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentShape(self):
    # [FUNCTION DOC]
    # Scenario: concatenates datasets whose second component has different per-element lengths
    # (4 versus 15). Asserts the combined output shapes are [20] and an unknown length, that all
    # 9 elements are produced in order with their original values, and that reading past the end
    # raises an out-of-range error. Returns nothing.
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)
    concatenated = input_dataset.concatenate(dataset_to_concatenate)
    self.assertEqual(
        [ts.as_list()
         for ts in nest.flatten(
             dataset_ops.get_legacy_output_shapes(concatenated))],
        [[20], [None]])
    get_next = self.getNext(concatenated)
    for i in range(9):
      result = self.evaluate(get_next())
      if i < 4:
        for component, result_component in zip(input_components, result):
          self.assertAllEqual(component[i], result_component)
      else:
        for component, result_component in zip(to_concatenate_components,
                                               result):
          self.assertAllEqual(component[i - 4], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentStructure(self):
    # [FUNCTION DOC]
    # Scenario: concatenates a dataset with two components onto one with three components.
    # Asserts a type error mentioning incompatible dataset elements. Returns nothing.
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0, 41.0]))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegex(TypeError, "Incompatible dataset elements"):
      input_dataset.concatenate(dataset_to_concatenate)

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentKeys(self):
    # [FUNCTION DOC]
    # Scenario: concatenates datasets whose elements are mappings with different key sets.
    # Asserts a type error mentioning incompatible dataset elements. Returns nothing.
    input_components = {
        "foo": np.array([[1], [2], [3], [4]]),
        "bar": np.array([[12], [13], [14], [15]])
    }
    to_concatenate_components = {
        "foo": np.array([[1], [2], [3], [4]]),
        "baz": np.array([[5], [6], [7], [8]])
    }

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegex(TypeError, "Incompatible dataset elements"):
      input_dataset.concatenate(dataset_to_concatenate)

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentType(self):
    # [FUNCTION DOC]
    # Scenario: concatenates datasets whose first component is integer in one and floating point
    # in the other. Asserts a type error mentioning incompatible dataset elements. Returns
    # nothing.
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1.0], [2.0], [3.0], [4.0]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 15))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegex(TypeError, "Incompatible dataset elements"):
      input_dataset.concatenate(dataset_to_concatenate)

  @combinations.generate(test_base.default_test_combinations())
  def testWindows(self):
    # [FUNCTION DOC]
    # Scenario: concatenates two datasets of window datasets (windows of size 1 over ranges 0-4
    # and 5-9), then flattens by identity. Asserts the numbers 0 through 9 are produced in order.
    # Returns nothing.
    a = dataset_ops.Dataset.range(5).window(1)
    b = dataset_ops.Dataset.range(5, 10).window(1)
    c = a.concatenate(b).flat_map(lambda x: x)
    self.assertDatasetProduces(c, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Scenario: concatenates ranges 0-4 and 5-9 with an explicit operation name; asserts the
    # numbers 0 through 9 are produced in order. Returns nothing.
    a = dataset_ops.Dataset.range(5)
    b = dataset_ops.Dataset.range(5, 10)
    c = a.concatenate(b, name="concatenate")
    self.assertDatasetProduces(c, list(range(10)))


class ConcatenateCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for the concatenate transformation, using the checkpoint test base class to
  # save, restore and verify iterator state.

  def _build_concatenate_dataset(self, var_array, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset for the checkpoint test. Concatenates a 4-element dataset of two
    # components with a dataset whose first component has 5 rows and whose second component is the
    # given array (var_array, 5 rows of 15 values in the test). If options (optional) is given,
    # applies it. Returns the dataset.
    input_components = (np.tile(np.array([[1], [2], [3], [4]]), 20),
                        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (np.tile(
        np.array([[5], [6], [7], [8], [9]]), 20), var_array)

    dataset = dataset_ops.Dataset.from_tensor_slices(
        input_components).concatenate(
            dataset_ops.Dataset.from_tensor_slices(to_concatenate_components))
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
    # Checkpoint scenario: sets the symbolic-checkpoint option per parameter and calls the
    # verification callback with a factory for the concatenated dataset and the expected 9
    # outputs. Returns nothing.
    num_outputs = 9
    array = np.tile(np.array([[16], [17], [18], [19], [20]]), 15)
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(self, lambda: self._build_concatenate_dataset(array, options),
              num_outputs)


class ConcatenateRandomAccessTest(test_base.DatasetTestBase,
                                  parameterized.TestCase):
  # [CLASS DOC]
  # Tests random access (fetching an element by index) on concatenated datasets, including empty
  # inputs and out-of-range indices.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 3, 4])))
  def testInvalidIndex(self, index):
    # [FUNCTION DOC]
    # Scenario: concatenates a one-element dataset with a two-element dataset (3 elements total)
    # and fetches by an invalid index (negative, 3 or 4). Asserts an out-of-range error. Input is
    # the index. Returns nothing.
    input_dataset = dataset_ops.Dataset.from_tensor_slices([-1])
    concatenate_dataset = dataset_ops.Dataset.from_tensor_slices([1, 2])
    concatenated = input_dataset.concatenate(concatenate_dataset)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(concatenated, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testConcatenateTwoEmptyDatasets(self):
    # [FUNCTION DOC]
    # Scenario: concatenates two empty datasets and fetches index 0. Asserts an out-of-range
    # error. Returns nothing.
    input_dataset = dataset_ops.Dataset.from_tensor_slices([])
    concatenate_dataset = dataset_ops.Dataset.from_tensor_slices([])
    concatenated = input_dataset.concatenate(concatenate_dataset)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(concatenated, index=0))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testConcatenateAnEmptyDataset(self):
    # [FUNCTION DOC]
    # Scenario: concatenates a one-element dataset (1.0) with an empty dataset. Asserts index 0
    # returns 1.0 and index 1 raises an out-of-range error. Returns nothing.
    input_dataset = dataset_ops.Dataset.from_tensor_slices([1.0])
    concatenate_dataset = dataset_ops.Dataset.from_tensor_slices([])
    concatenated = input_dataset.concatenate(concatenate_dataset)
    self.assertAllEqual(
        self.evaluate(random_access.at(concatenated, index=0)), 1.0)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(concatenated, index=1))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testConcatenateOntoEmptyDataset(self):
    # [FUNCTION DOC]
    # Scenario: concatenates an empty dataset with a two-element dataset (2.0, 3.0). Asserts
    # index 0 returns 2.0, index 1 returns 3.0, and index 2 raises an out-of-range error. Returns
    # nothing.
    input_dataset = dataset_ops.Dataset.from_tensor_slices([])
    concatenate_dataset = dataset_ops.Dataset.from_tensor_slices([2.0, 3.0])
    concatenated = input_dataset.concatenate(concatenate_dataset)
    self.assertAllEqual(
        self.evaluate(random_access.at(concatenated, index=0)), 2.0)
    self.assertAllEqual(
        self.evaluate(random_access.at(concatenated, index=1)), 3.0)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(concatenated, index=2))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testConcatenateTwoNonEmptyDatasets(self):
    # [FUNCTION DOC]
    # Scenario: concatenates datasets holding 0, 1, 2 and 3, 4. Asserts that indices 0 to 4
    # return the values equal to the index (the value is compared directly without an explicit
    # evaluation step), and that index 5 raises an out-of-range error. Returns nothing.
    input_dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2])
    concatenate_dataset = dataset_ops.Dataset.from_tensor_slices([3, 4])
    concatenated = input_dataset.concatenate(concatenate_dataset)
    for i in range(5):
      self.assertAllEqual(random_access.at(concatenated, index=i), i)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(concatenated, index=5))


class GlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that concatenated datasets can be globally shuffled and still produce every element.
  """Tests for global shuffling of tf.data datasets."""

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledOutput(self):
    # [FUNCTION DOC]
    # Scenario: concatenates ranges 0-4 and 5-16, applies a global shuffle, reads all output
    # (initialization required) and asserts the elements are exactly 0 through 16, in any order.
    # Returns nothing.
    dataset1 = dataset_ops.Dataset.range(0, 5)
    dataset2 = dataset_ops.Dataset.range(5, 17)

    dataset = dataset1.concatenate(dataset2)

    dataset = global_shuffle_op._global_shuffle(dataset)

    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(output, range(0, 17))

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledWithBatchOutput(self):
    # [FUNCTION DOC]
    # Scenario: concatenates ranges 0-3 and 4-9, batches by 3 dropping the remainder, and applies
    # a global shuffle (batching checks the shuffle map is stateless). Sorts the output batches
    # and asserts there are three batches equal to [0,1,2], [3,4,5] and [6,7,8]. Returns nothing.
    """Testing with `.batch()` ensures the global shuffle map is stateless."""
    dataset1 = dataset_ops.Dataset.range(0, 4)
    dataset2 = dataset_ops.Dataset.range(4, 10)

    dataset = dataset1.concatenate(dataset2)
    dataset = dataset.batch(3, drop_remainder=True)

    dataset = global_shuffle_op._global_shuffle(dataset)

    got = self.getDatasetOutput(dataset, requires_initialization=True)
    expected = [
        np.array([0, 1, 2], dtype=np.int32),
        np.array([3, 4, 5], dtype=np.int32),
        np.array([6, 7, 8], dtype=np.int32),
    ]

    self.assertIsInstance(got, list)
    # Converts to tuples for lexicographically sort
    got.sort(key=tuple)

    self.assertLen(got, len(expected))

    for element_got, element_expected in zip(got, expected):
      self.assertAllEqual(element_got, element_expected)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedConcatenateShuffledOutput(self):
    # [FUNCTION DOC]
    # Scenario: concatenates three ranges (0-2, 3-5, 6-8) in a nested way, applies a global
    # shuffle, and asserts the output elements are exactly 0 through 8 in any order. Returns
    # nothing.
    dataset1 = dataset_ops.Dataset.range(0, 3)
    dataset2 = dataset_ops.Dataset.range(3, 6)
    dataset3 = dataset_ops.Dataset.range(6, 9)

    dataset = dataset1.concatenate(dataset2)
    dataset = dataset.concatenate(dataset3)

    dataset = global_shuffle_op._global_shuffle(dataset)

    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(output, range(0, 9))


class ConcatenateGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase
):
  # [CLASS DOC]
  # Checkpoint tests for globally shuffled concatenations of ranges (two, three, four and six
  # pieces, plus a batched and prefetched variant), parameterized over the reshuffle and
  # symbolic-checkpoint flags.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(10, 8), (9, 5), (4, 7), (5, 8)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testConcatenate(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, int],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):
    # [FUNCTION DOC]
    # Checkpoint scenario: two consecutive ranges (sizes from dataset_ranges) concatenated and
    # globally shuffled with seed 10. Calls the verification callback with the dataset factory and
    # the total element count; order is ignored only when reshuffling each iteration. Returns
    # nothing.

    def _build_dataset():
      # [FUNCTION DOC]
      # HELPER: factory for the dataset under test; concatenates the two ranges, globally
      # shuffles with the configured reshuffle flag, disables default optimizations, sets the
      # symbolic-checkpoint option, and returns the dataset.
      first_dataset = dataset_ops.Dataset.range(dataset_ranges[0])
      second_dataset = dataset_ops.Dataset.range(
          dataset_ranges[0], dataset_ranges[0] + dataset_ranges[1]
      )
      dataset = first_dataset.concatenate(second_dataset)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=sum(dataset_ranges),
        assert_items_equal=reshuffle_each_iteration,
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(10, 8, 11), (9, 5, 3)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testNestedConcatenate(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, int],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):
    # [FUNCTION DOC]
    # Checkpoint scenario: three consecutive ranges concatenated in a nested (left-leaning) way
    # and globally shuffled with seed 10. Calls the verification callback with the dataset factory
    # and the total element count; order is ignored only when reshuffling each iteration. Returns
    # nothing.

    def _build_dataset():
      # [FUNCTION DOC]
      # HELPER: factory for the dataset under test; concatenates the three ranges, globally
      # shuffles, disables default optimizations, sets the symbolic-checkpoint option, and
      # returns the dataset.
      first_dataset = dataset_ops.Dataset.range(dataset_ranges[0])
      second_dataset = dataset_ops.Dataset.range(
          dataset_ranges[0], dataset_ranges[0] + dataset_ranges[1]
      )
      third_dataset = dataset_ops.Dataset.range(
          sum(dataset_ranges[:2]), sum(dataset_ranges[:3])
      )

      dataset = first_dataset.concatenate(second_dataset)
      dataset = dataset.concatenate(third_dataset)

      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=sum(dataset_ranges),
        assert_items_equal=reshuffle_each_iteration,
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(3, 4, 6, 5)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testFourNestedConcatenate(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, int],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):
    # [FUNCTION DOC]
    # Checkpoint scenario: four consecutive ranges concatenated as a balanced tree (first two,
    # last two, then the two halves) and globally shuffled with seed 10. Calls the verification
    # callback with the dataset factory and total element count; order is ignored only when
    # reshuffling each iteration. Returns nothing.
    def _build_dataset():
      # [FUNCTION DOC]
      # HELPER: factory for the dataset under test; builds the balanced concatenation, globally
      # shuffles, disables default optimizations, sets the symbolic-checkpoint option, and
      # returns the dataset.
      first_dataset = dataset_ops.Dataset.range(dataset_ranges[0])
      second_dataset = dataset_ops.Dataset.range(
          dataset_ranges[0], sum(dataset_ranges[:2])
      )
      third_dataset = dataset_ops.Dataset.range(
          sum(dataset_ranges[:2]), sum(dataset_ranges[:3])
      )
      fourth_dataset = dataset_ops.Dataset.range(
          sum(dataset_ranges[:3]), sum(dataset_ranges)
      )

      left = first_dataset.concatenate(second_dataset)
      right = third_dataset.concatenate(fourth_dataset)

      dataset = left.concatenate(right)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=sum(dataset_ranges),
        assert_items_equal=reshuffle_each_iteration,
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(1, 2, 3, 4, 5, 6)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testDeepConcatenate(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, ...],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):
    # [FUNCTION DOC]
    # Checkpoint scenario: six consecutive ranges (sizes 1 to 6) concatenated one after another
    # into a deep chain and globally shuffled with seed 10. Calls the verification callback with
    # the dataset factory and total element count; order is ignored only when reshuffling each
    # iteration. Returns nothing.
    def _build_dataset():
      # [FUNCTION DOC]
      # HELPER: factory for the dataset under test; computes running totals of the range sizes,
      # chains the ranges with concatenation, globally shuffles, disables default optimizations,
      # sets the symbolic-checkpoint option, and returns the dataset.
      prefix_sums = [0] * (len(dataset_ranges) + 1)
      for i, value in enumerate(dataset_ranges):
        prefix_sums[i + 1] = prefix_sums[i] + value

      dataset = dataset_ops.Dataset.range(prefix_sums[0], prefix_sums[1])
      for i in range(1, len(dataset_ranges)):
        to_concat = dataset_ops.Dataset.range(
            prefix_sums[i], prefix_sums[i + 1]
        )
        dataset = dataset.concatenate(to_concat)

      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=sum(dataset_ranges),
        assert_items_equal=reshuffle_each_iteration,
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(1, 2, 3, 4, 5, 6)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testDeepConcatenateWithBatchAndPrefetch(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, ...],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):
    # [FUNCTION DOC]
    # Checkpoint scenario: like the deep concatenation test, but the chain is batched by 2
    # (dropping the remainder), prefetched with automatic buffer size, globally shuffled and then
    # unbatched. Calls the verification callback with the dataset factory and an expected count
    # of the total rounded down to an even number; order is ignored only when reshuffling each
    # iteration. Returns nothing.
    def _build_dataset():
      # [FUNCTION DOC]
      # HELPER: factory for the dataset under test; chains the ranges, batches by 2, prefetches,
      # globally shuffles, unbatches, disables default optimizations, sets the symbolic-checkpoint
      # option, and returns the dataset.
      prefix_sums = [0] * (len(dataset_ranges) + 1)
      for i, value in enumerate(dataset_ranges):
        prefix_sums[i + 1] = prefix_sums[i] + value

      dataset = dataset_ops.Dataset.range(prefix_sums[0], prefix_sums[1])
      for i in range(1, len(dataset_ranges)):
        to_concat = dataset_ops.Dataset.range(
            prefix_sums[i], prefix_sums[i + 1]
        )
        dataset = dataset.concatenate(to_concat)

      dataset = dataset.batch(2, drop_remainder=True)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)

      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )
      dataset = dataset.unbatch()

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=(sum(dataset_ranges) // 2) * 2,
        assert_items_equal=reshuffle_each_iteration,
    )


if __name__ == "__main__":
  test.main()
