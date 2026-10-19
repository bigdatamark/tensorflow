# [MODULE DOC]
# Tests for the dataset shard transformation, which keeps every num_shards-th element starting at
# a given index. Covers simple and nested data, valid and invalid (out-of-range, negative, zero)
# shard arguments, small datasets, the name argument, checkpointing, random access by index and
# interaction with the global shuffle op. Test classes only.
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
"""Tests for `tf.data.Dataset.shard()`."""
from typing import Callable, Optional

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class ShardTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests for shard: selected elements and argument validation.

  @combinations.generate(test_base.default_test_combinations())
  def testSimpleCase(self):
    # [FUNCTION DOC]
    # Shards a range of ten into five shards and takes index two; asserts the output is 2 and 7.
    dataset = dataset_ops.Dataset.range(10).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[2, 7])

  @combinations.generate(test_base.default_test_combinations())
  def testNestedData(self):
    # [FUNCTION DOC]
    # Zips an ascending and a descending range, shards into five with index two; asserts the
    # output pairs (2, 8) and (7, 3).
    dataset_a = dataset_ops.Dataset.range(10)
    dataset_b = dataset_ops.Dataset.range(10, 0, -1)
    dataset = dataset_ops.Dataset.zip((dataset_a, dataset_b)).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[(2, 8), (7, 3)])

  @combinations.generate(test_base.default_test_combinations())
  def testOffsetZero(self):
    # [FUNCTION DOC]
    # Shards a range of ten into five with index zero; asserts the output is 0 and 5.
    dataset = dataset_ops.Dataset.range(10).shard(5, 0)
    self.assertDatasetProduces(dataset, expected_output=[0, 5])

  @combinations.generate(test_base.default_test_combinations())
  def testOffsetGreaterNumShards(self):
    # [FUNCTION DOC]
    # Shard with index seven but only five shards; fetching the first element must raise an
    # invalid-argument error.
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(5, 7)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testNegativeOffset(self):
    # [FUNCTION DOC]
    # Shard with a negative index; fetching the first element must raise an invalid-argument
    # error.
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(5, -3)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testNegativeNumShards(self):
    # [FUNCTION DOC]
    # Shard with a negative number of shards; fetching the first element must raise an
    # invalid-argument error.
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(-3, 1)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testZeroNumShards(self):
    # [FUNCTION DOC]
    # Shard with zero shards; fetching the first element must raise an invalid-argument error.
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(0, 1)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testIteratorEndsBeforeFirstElem(self):
    # [FUNCTION DOC]
    # A one-element range sharded into five with index two has no element at that index; asserts
    # the output is empty.
    dataset = dataset_ops.Dataset.range(1).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[])

  @combinations.generate(test_base.default_test_combinations())
  def testLargerWorkerPool(self):
    # [FUNCTION DOC]
    # Range of ten sharded into seven with index five; asserts only the element 5 is produced.
    dataset = dataset_ops.Dataset.range(10).shard(7, 5)
    self.assertDatasetProduces(dataset, expected_output=[5])

  @combinations.generate(test_base.default_test_combinations())
  def testIndexEqualsNumShards(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says the index equals the number of shards, but the test uses five
    # shards with index four (the last valid index, one less than the shard count).
    # Range of ten sharded into five with index four; asserts the output is 4 and 9.
    dataset = dataset_ops.Dataset.range(10).shard(5, 4)
    self.assertDatasetProduces(dataset, expected_output=[4, 9])

  @combinations.generate(test_base.default_test_combinations())
  def testIndexEqualsNumShards2(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says the index equals the number of shards, but the test uses four
    # shards with index three (the last valid index, one less than the shard count).
    # Range of ten sharded into four with index three; asserts the output is 3 and 7.
    dataset = dataset_ops.Dataset.range(10).shard(4, 3)
    self.assertDatasetProduces(dataset, expected_output=[3, 7])

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsLargerThanDataset(self):
    # [FUNCTION DOC]
    # Range of ten sharded into twenty with index five; asserts only the element 5 is produced.
    dataset = dataset_ops.Dataset.range(10).shard(20, 5)
    self.assertDatasetProduces(dataset, expected_output=[5])

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Shards a single-element dataset into one shard with a custom name; asserts the element is
    # kept.
    dataset = dataset_ops.Dataset.from_tensors(42).shard(1, 0, name="shard")
    self.assertDatasetProduces(dataset, [42])


class ShardCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                          parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for the shard dataset using the shared checkpoint base.

  def _build_dataset(self, num_elements, num_shards, index, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint test.
    # Creates a range of num_elements sharded by num_shards at index; applies options (optional)
    # if given. Returns the dataset.
    dataset = dataset_ops.Dataset.range(num_elements).shard(num_shards, index)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True]),
          combinations.combine(
              elems=[10, 100], num_shards=[2, 5], index=[0, 1])))
  def test(self, verify_fn, symbolic_checkpoint, elems, num_shards, index):
    # [FUNCTION DOC]
    # Sets the symbolic-checkpoint option and runs the verification callback on the sharded
    # dataset, expecting elems divided by num_shards outputs (integer division) and correct
    # checkpoint round-tripping.
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self,
        lambda: self._build_dataset(elems, num_shards, index, options),
        num_outputs=elems // num_shards)


class ShardRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for index-based random access into sharded datasets via the random access helper.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 2, 3, 4])))
  def testInvalidIndex(self, index):
    # [FUNCTION DOC]
    # A range of four sharded into two (two elements): negative or too-large access indices must
    # raise an out-of-range error.
    dataset = dataset_ops.Dataset.range(4).shard(num_shards=2, index=0)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDataset(self):
    # [FUNCTION DOC]
    # Random access at index zero into a sharded empty dataset must raise an out-of-range error.
    dataset = dataset_ops.Dataset.from_tensor_slices([]).shard(
        num_shards=2, index=1)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=0))

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsAndIndexLessThanNumElements(self):
    # [FUNCTION DOC]
    # Range of ten sharded into five with index zero: random access gives 0 then 5, and a third
    # access raises an out-of-range error.
    dataset = dataset_ops.Dataset.range(10).shard(5, 0)
    self.assertEqual(0, self.evaluate(random_access.at(dataset, 0)))
    self.assertEqual(5, self.evaluate(random_access.at(dataset, 1)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 2))

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsGreaterThanNumElementsIndexLess(self):
    # [FUNCTION DOC]
    # Range of seven sharded into eight with index three: the first access gives 3, the second
    # raises an out-of-range error.
    dataset = dataset_ops.Dataset.range(7).shard(8, 3)
    self.assertEqual(3, self.evaluate(random_access.at(dataset, 0)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 1))

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsAndIndexGreaterThanNumElements(self):
    # [FUNCTION DOC]
    # Range of thirteen sharded into twenty-three with index twenty-one (beyond the data) is
    # empty; random access at zero must raise an out-of-range error.
    dataset = dataset_ops.Dataset.range(13).shard(23, 21)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 0))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              elements=[0, 10, 50],
              num_shards=[5, 7, 10],
              index=[0, 1, 2, 3, 4],
          )))
  def testMultipleCombinations(self, elements, num_shards, index):
    # [FUNCTION DOC]
    # For many combinations of element count, shard count and index, checks random access at each
    # position of the sharded dataset equals the original element at index plus shards times the
    # position, and that access at the cardinality raises an out-of-range error.
    components = range(elements)
    dataset = dataset_ops.Dataset.range(elements).shard(
        num_shards=num_shards, index=index)
    len_dataset = self.evaluate(dataset.cardinality())
    for i in range(self.evaluate(dataset.cardinality())):
      self.assertAllEqual(components[index + (num_shards * i)],
                          self.evaluate(random_access.at(dataset, i)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=len_dataset))


class ShardGlobalShuffleTest(
    test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for applying the global shuffle op to sharded datasets.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[100],
              num_shards=[1, 3, 5],
              shard_index=[0, 1, 2, 4],
              seed=[None, 42],
              reshuffle_each_iteration=[True, False])))
  def testShard(
      self,
      dataset_range: int,
      num_shards: int,
      shard_index: int,
      seed: Optional[int],
      reshuffle_each_iteration: bool):
    # [FUNCTION DOC]
    # Returns immediately (does nothing) when the shard index is not below the shard count.
    # Otherwise shards a range, prefetches, applies a global shuffle with the given seed
    # (optional) and reshuffle flag. Asserts the output has the same multiset of values as the
    # unshuffled shard but in a different order.
    if shard_index >= num_shards:
      return

    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = dataset.shard(num_shards, shard_index)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    expected = list(range(shard_index, dataset_range, num_shards))
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)


class ShardGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for shard followed by the global shuffle op.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              num_shards=[1, 3],
              shard_index=[0, 1, 2],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testShard(
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      num_shards: int,
      shard_index: int,
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    # [FUNCTION DOC]
    # Returns immediately when the shard index is not below the shard count. Otherwise runs the
    # verification callback on a sharded, prefetched, globally shuffled range (fixed seed,
    # parameterized reshuffle and symbolic-checkpoint options), expecting as many outputs as the
    # shard holds; items are compared order-insensitively when reshuffling each iteration.

    if shard_index >= num_shards:
      return

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: builds the pipeline under test: range, shard, prefetch with auto-tuned buffer,
      # global shuffle with seed forty-two, then applies the symbolic-checkpoint option. Returns
      # the dataset.
      dataset = dataset_ops.Dataset.range(dataset_range)
      dataset = dataset.shard(num_shards, shard_index)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=len(range(shard_index, dataset_range, num_shards)),
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
