# [MODULE DOC]
# Test module for the dataset cache transformation. It covers the file-backed cache, the
# in-memory cache, checkpointing of cache iterators (in memory and on file), random access
# over cached datasets, and global shuffling over cached datasets. Importing it has no side
# effects other than importing test infrastructure; running it as a script starts the test
# runner.
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
"""Tests for `tf.data.Dataset.cache()`."""
import functools
import os
from os import path
import shutil
import tempfile
from typing import Callable, Optional

from absl.testing import parameterized
import numpy as np
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class FileCacheTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the file-backed cache of a dataset: passthrough of elements, concurrent writers
  # and readers of the same cache file, reading past the end, zipping several cached datasets
  # and cleanup of partial cache files. Each test uses a fresh temporary directory.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates a fresh temporary directory and records the cache file prefix inside it.
    # Takes no arguments and returns nothing. Side effect: creates a directory on disk.
    super(FileCacheTest, self).setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.cache_prefix = path.join(self.tmp_dir, "cache")

  def tearDown(self):
    # [FUNCTION DOC]
    # Removes the temporary directory created in setUp (ignoring errors) and then runs the
    # parent teardown. Takes no arguments and returns nothing.
    if self.tmp_dir:
      shutil.rmtree(self.tmp_dir, ignore_errors=True)
    super(FileCacheTest, self).tearDown()

  @combinations.generate(test_base.default_test_combinations())
  def testCacheDatasetPassthrough(self):
    # [FUNCTION DOC]
    # Scenario: a repeated dataset of three-component elements is read plain, then through a
    # file cache, then replayed from the cache file with an empty upstream. Expects the cached
    # run and the replay to produce exactly the same 20 elements as the plain run, followed by
    # an out-of-range error, and a missing cache file with an empty upstream to be immediately
    # out of range. Also checks the legacy output shapes of the dataset.
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))

    def dataset_fn(count=5, filename=None):
      # [FUNCTION DOC]
      # HELPER: builds the dataset used by the passthrough test.
      # Takes the repeat count (default 5) and an optional cache file name. Returns the
      # components dataset repeated count times, cached to the file if a name is given.
      repeat_dataset = (
          dataset_ops.Dataset.from_tensor_slices(components).repeat(count))
      if filename:
        return repeat_dataset.cache(filename)
      else:
        return repeat_dataset

    self.assertEqual(
        tuple([c.shape[1:] for c in components]),
        dataset_ops.get_legacy_output_shapes(dataset_fn()))

    get_next = self.getNext(dataset_fn())

    # First run without caching to collect the "ground truth".
    elements = []
    for _ in range(20):
      elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    # Assert that the cached dataset has the same elements as the
    # "ground truth".
    get_next = self.getNext(dataset_fn(filename=self.cache_prefix))
    cached_elements = []
    for _ in range(20):
      cached_elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertAllEqual(elements, cached_elements)

    # Re-initialize with an empty upstream (to throw errors.OutOfRangeError
    # if we didn't use the cache).
    get_next = self.getNext(dataset_fn(count=0, filename=self.cache_prefix))
    replayed_elements = []
    for _ in range(20):
      replayed_elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertEqual(cached_elements, replayed_elements)

    # Re-initialize with an empty upstream and a missing cache file (should
    # throw errors.OutOfRangeError immediately).
    get_next = self.getNext(
        dataset_fn(count=0, filename=self.cache_prefix + "nonsense"))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentWriters(self):
    # [FUNCTION DOC]
    # Scenario: two datasets cache to the same file prefix. After the first iterator starts
    # writing, the second iterator must fail with an already-exists error, while the first
    # keeps working.
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))

    cache_dataset1 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))
    cache_dataset2 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))

    get_next1 = self.getNext(cache_dataset1)
    get_next2 = self.getNext(cache_dataset2)

    self.evaluate(get_next1())  # this should succeed

    with self.assertRaises(errors.AlreadyExistsError):
      self.evaluate(get_next2())

    self.evaluate(get_next1())  # this should continue to succeed

  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentReaders(self):
    # [FUNCTION DOC]
    # Scenario: one iterator fully writes the file cache. Then two fresh iterators over the same
    # cache file read concurrently in interleaved order. Expects both to produce the same four
    # elements as the writer and then hit end of data.
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))

    cache_dataset1 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))
    cache_dataset2 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))

    get_next1 = self.getNext(cache_dataset1)
    get_next2 = self.getNext(cache_dataset2)

    elements = []
    for _ in range(4):
      elements.append(self.evaluate(get_next1()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next1())

    # Re-initialize
    get_next1 = self.getNext(cache_dataset1, requires_initialization=True)
    get_next2 = self.getNext(cache_dataset2, requires_initialization=True)

    # Reading concurrently should succeed.
    elements_itr1 = []
    elements_itr2 = []
    elements_itr2.append(self.evaluate(get_next2()))
    elements_itr1.append(self.evaluate(get_next1()))
    elements_itr2.append(self.evaluate(get_next2()))
    elements_itr1.append(self.evaluate(get_next1()))
    # Intentionally reversing the order
    elements_itr1.append(self.evaluate(get_next1()))
    elements_itr2.append(self.evaluate(get_next2()))
    elements_itr1.append(self.evaluate(get_next1()))
    elements_itr2.append(self.evaluate(get_next2()))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next2())

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next1())

    self.assertAllEqual(elements, elements_itr1)
    self.assertAllEqual(elements, elements_itr2)

  @combinations.generate(test_base.default_test_combinations())
  def testReadingPastEndOfSequence(self):
    # [FUNCTION DOC]
    # Scenario: range of 10 cached to a file, mapped, batched by 4 and repeated twice.
    # Expects the batches of 4, 4 and 2 elements twice, i.e. no error when the cache is read
    # past its end.
    dataset = dataset_ops.Dataset.range(10).cache(self.cache_prefix)
    dataset = dataset.map(lambda a: a).batch(4).repeat(2)
    expected_output = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]] * 2
    self.assertDatasetProduces(dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testCacheZipped(self):
    # [FUNCTION DOC]
    # Scenario: three shuffled range datasets, each cached to its own file, are zipped.
    # Reading the zipped dataset twice must give identical output, showing the second pass is
    # served from the caches.
    def make_dataset(i):
      # [FUNCTION DOC]
      # HELPER: builds one shuffled, file-cached range dataset for the zip test.
      # Takes an integer index used as the cache file name suffix; returns the dataset.
      cache_path = self.cache_prefix + "_" + str(i)
      return dataset_ops.Dataset.range(100).shuffle(100).cache(cache_path)

    datasets = [make_dataset(i) for i in range(3)]
    dataset = dataset_ops.Dataset.zip(tuple(datasets))
    first_order = self.getDatasetOutput(dataset)
    second_order = self.getDatasetOutput(dataset)
    self.assertEqual(first_order, second_order)

  @combinations.generate(test_base.default_test_combinations())
  def testCleaningUpCacheFiles(self):
    # [FUNCTION DOC]
    # Scenario (eager mode only, skipped otherwise): repeatedly builds an iterator over a cached
    # range and reads 0, 3, 10, 12 and 15 elements, stopping on end of data, so that partially
    # written iterators get destroyed. Asserts nothing explicitly; it checks that repeated use
    # of the same cache prefix does not fail, i.e. cache files are cleaned up.

    def do_test(i):
      # [FUNCTION DOC]
      # HELPER: builds a cached iterator and reads up to i elements, stopping at end of data.
      # Takes the number of elements to read; returns nothing.
      dataset = dataset_ops.Dataset.range(10).cache(self.cache_prefix)
      get_next = self.getNext(dataset)
      for _ in range(i):
        try:
          self.evaluate(get_next())
        except errors.OutOfRangeError:
          break

    if not context.executing_eagerly():
      self.skipTest(
          "Test requires eager mode for iterators to be deconstructed")

    for i in [0, 3, 10, 12, 15]:
      do_test(i)


class MemoryCacheTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the in-memory cache (no file name): passthrough, empty input, concurrent readers,
  # interaction with take and repeat, epoch semantics, resource capture in eager mode, known
  # cardinality, checkpointing of a finished cache and naming.

  @combinations.generate(test_base.default_test_combinations())
  def testCacheDatasetPassthrough(self):
    # [FUNCTION DOC]
    # Scenario: a flat-mapped dataset whose repeat count comes from a variable is read through a
    # cache and without one, twice each. After the variable is set to zero the uncached
    # iterator is empty while the cached one still replays all 30 elements from memory and then
    # ends. Prefetch injection is disabled to keep the behavior deterministic.
    with ops.device("cpu:0"):
      repeat_count = variables.Variable(constant_op.constant(10, dtypes.int64))
      dataset = dataset_ops.Dataset.range(3).flat_map(
          lambda x: dataset_ops.Dataset.from_tensors(x).repeat(repeat_count))

      options = options_lib.Options()
      options.experimental_optimization.inject_prefetch = False
      dataset = dataset.with_options(options)

      cached_dataset = dataset.cache().repeat(2)
      uncached_dataset = dataset.repeat(2)

      self.evaluate(repeat_count.initializer)
      # Needs to be initializable to capture the variable.
      cached_next = self.getNext(cached_dataset, requires_initialization=True)
      uncached_next = self.getNext(
          uncached_dataset, requires_initialization=True)
      for i in range(3):
        for _ in range(10):
          self.assertEqual(self.evaluate(cached_next()), i)
          self.assertEqual(self.evaluate(uncached_next()), i)

      self.evaluate(repeat_count.assign(0))

      # The uncached iterator should now be empty.
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(uncached_next())

      # The cached iterator replays from cache.
      for i in range(3):
        for _ in range(10):
          self.assertEqual(self.evaluate(cached_next()), i)

      # The cached iterator should now be empty.
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(cached_next())

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyCacheReading(self):
    # [FUNCTION DOC]
    # Scenario: a cache over an input repeated zero times. Expects the dataset to produce no
    # elements.
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))

    repeat_dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).repeat(0))
    cache_dataset = repeat_dataset.cache()

    self.assertDatasetProduces(cache_dataset, expected_output=[])

  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentReaders(self):
    # [FUNCTION DOC]
    # Scenario: two datasets each cache a range of five and add different offsets. Their
    # iterators are advanced in an interleaved order. Expects each to see the correct values
    # (shifted by 1 and 6) and then end of data, i.e. concurrent readers of memory caches do not
    # interfere.

    dataset_fn = lambda: dataset_ops.Dataset.range(5).cache()
    d1 = dataset_fn().map(lambda x: x + 1)
    d2 = dataset_fn().map(lambda x: x + 6)

    get_next1 = self.getNext(d1)

    self.assertEqual(1, self.evaluate(get_next1()))
    self.assertEqual(2, self.evaluate(get_next1()))
    self.assertEqual(3, self.evaluate(get_next1()))

    get_next2 = self.getNext(d2)

    self.assertEqual(6, self.evaluate(get_next2()))
    self.assertEqual(7, self.evaluate(get_next2()))
    self.assertEqual(4, self.evaluate(get_next1()))  # interleave execution
    self.assertEqual([8, 5],
                     [self.evaluate(get_next2()),
                      self.evaluate(get_next1())])
    self.assertEqual(9, self.evaluate(get_next2()))
    self.assertEqual(10, self.evaluate(get_next2()))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next2())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next1())

  @combinations.generate(test_base.default_test_combinations())
  def testCacheTakeRepeat(self):
    # [FUNCTION DOC]
    # Scenario: cache of a range of 10, then take 5, then repeat twice. Expects the first five
    # elements twice.
    dataset = dataset_ops.Dataset.range(10).cache().take(5).repeat(2)

    expected_output = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testCacheRepeatEpochs(self):
    # [FUNCTION DOC]
    # Scenario: a mapped range counts how often the map function runs, then is cached and
    # repeated twice. Expects the counter to advance with each element in the first epoch and to
    # stay at 10 during the second epoch (served from the cache), then end of data.
    counter = variables.Variable(0)
    self.evaluate(counter.initializer)

    def increment_fn(x):
      # [FUNCTION DOC]
      # HELPER: map function that increments the shared counter variable and returns its input.
      counter.assign_add(1)
      return x

    dataset = dataset_ops.Dataset.range(10).map(increment_fn).cache().repeat(2)
    options = options_lib.Options()
    options.experimental_optimization.inject_prefetch = False
    dataset = dataset.with_options(options)
    get_next = self.getNext(dataset, requires_initialization=True)

    # first epoch
    for i in range(10):
      self.assertEqual(i, self.evaluate(counter))
      self.assertEqual(i, self.evaluate(get_next()))
    # second epoch
    for i in range(10):
      self.assertEqual(10, self.evaluate(counter))
      self.assertEqual(i, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheIterationEpochs(self):
    # [FUNCTION DOC]
    # Scenario (eager, TF2 only): like the repeat-epochs test but iterates the cached dataset
    # twice with a for loop. The counter tracks each element during the first pass and stays at
    # 10 during the second pass, showing the second iteration reads from the cache.
    counter = variables.Variable(0)
    self.evaluate(counter.initializer)

    def increment_fn(x):
      # [FUNCTION DOC]
      # HELPER: map function that increments the shared counter variable and returns its input.
      counter.assign_add(1)
      return x

    dataset = dataset_ops.Dataset.range(10).map(increment_fn).cache()
    options = options_lib.Options()
    options.experimental_optimization.inject_prefetch = False
    dataset = dataset.with_options(options)

    # first epoch
    i = 0
    for elem in dataset:
      self.assertEqual(i, self.evaluate(elem))
      i += 1
      self.assertEqual(i, self.evaluate(counter))

    # second epoch
    i = 0
    for elem in dataset:
      self.assertEqual(10, self.evaluate(counter))
      self.assertEqual(i, self.evaluate(elem))
      i += 1

  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheV2ResourceCapture(self):
    # [FUNCTION DOC]
    # Scenario (eager, TF2 only): a cached dataset is captured (via a partially applied
    # function) inside an interleave function of another dataset. Expects iteration to yield
    # the numbers 0 to 9, i.e. the cache resource can be captured by a nested function.

    def make_dataset():
      # [FUNCTION DOC]
      # HELPER: builds the outer dataset whose interleave function returns the cached range.
      ids = dataset_ops.Dataset.range(10)
      ids = ids.cache()

      def interleave_fn(dataset, _):
        # [FUNCTION DOC]
        # HELPER: interleave function that ignores its element and returns the given dataset.
        return dataset

      dataset = dataset_ops.Dataset.range(1)
      dataset = dataset.interleave(functools.partial(interleave_fn, ids))
      return dataset

    results = []
    for elem in make_dataset():
      results.append(elem.numpy())

    self.assertAllEqual(results, range(10))

  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheV2ConcurrentIterators(self):
    # [FUNCTION DOC]
    # Scenario (eager, TF2 only): two iterators over the same cached range are advanced in
    # lockstep. Expects both to produce 0 through 9.

    dataset = dataset_ops.Dataset.range(10).cache()

    it1 = iter(dataset)
    it2 = iter(dataset)

    for i in range(10):
      self.assertEqual(next(it1), i)
      self.assertEqual(next(it2), i)

  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheKnownCardinality(self):
    # [FUNCTION DOC]
    # Scenario (eager, TF2 only): a reshuffled range of 10 is cached; an iterator reads all
    # 10 elements without reaching end of data. A second iterator must then replay exactly the
    # same order, showing the cache was completed once the known cardinality was reached.

    # Check that a dataset which produces random permutation of range(10) ends
    # up being cached when we read all of its element but do not reach EOF.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.shuffle(10, reshuffle_each_iteration=True).cache()

    it = iter(dataset)

    results = []
    for _ in range(10):
      results.append(next(it))

    it = iter(dataset)
    for i in range(10):
      self.assertEqual(next(it), results[i])

  @combinations.generate(test_base.eager_only_combinations())
  def testCheckpointFinishedCache(self):
    # [FUNCTION DOC]
    # Scenario (eager only): an iterator over a cached range is fully consumed, saved with a
    # checkpoint manager and restored. Expects the iterator to still be exhausted afterwards
    # (next raises stop-iteration). Writes a checkpoint into the test temporary directory.
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    ds = ds.cache()

    iterator = iter(ds)
    for i in range(num_elements):
      self.assertEqual(next(iterator).numpy(), i)
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=1)
    manager.save()
    manager.restore_or_initialize()
    with self.assertRaises(StopIteration):
      next(iterator)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Scenario: cache created with an explicit operation name over a single-element dataset.
    # Expects the dataset to produce that one element.
    dataset = dataset_ops.Dataset.from_tensors(42).cache(name="cache")
    self.assertDatasetProduces(dataset, [42])


class CacheCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                          parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for cache iterators, run both for the memory cache and the file cache.
  # Covers checkpoints before and after the first epoch is complete, saving earlier than the
  # point reached, an unused writer iterator, a checkpoint not restored, and ignoring a
  # checkpoint once the cache has been fully written.

  def setUp(self):
    # [FUNCTION DOC]
    # Initializes test parameters: range size 10, 3 repeats, total of 30 outputs, and the
    # cache file name prefix. Returns nothing.
    super(CacheCheckpointTest, self).setUp()
    self.range_size = 10
    self.num_repeats = 3
    self.num_outputs = self.range_size * self.num_repeats
    self.cache_file_prefix = "test"

  def make_dataset_fn(self, is_memory):
    # [FUNCTION DOC]
    # HELPER: builds the dataset factory used by the checkpoint tests.
    # Takes a boolean: true for an in-memory cache (empty file name), false for a file cache in
    # the temporary directory. Returns a no-argument function building a cached range
    # repeated num_repeats times.
    if is_memory:
      filename = ""
    else:
      filename = os.path.join(self.get_temp_dir(), self.cache_file_prefix)

    def ds_fn():
      # [FUNCTION DOC]
      # HELPER: returns the cached, repeated range dataset (closure over the settings).
      return dataset_ops.Dataset.range(self.range_size).cache(filename).repeat(
          self.num_repeats)

    return ds_fn

  def expected_outputs(self):
    # [FUNCTION DOC]
    # HELPER: returns the list of values the dataset should yield: the range repeated
    # num_repeats times.
    return list(range(self.range_size)) * self.num_repeats

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointBeforeOneEpoch(self, is_memory):
    # [FUNCTION DOC]
    # Scenario: read 5 elements, checkpoint, restore and read the rest. Expects the combined
    # output to equal the full expected sequence. Parameter is_memory selects memory or file
    # cache.
    ds_fn = self.make_dataset_fn(is_memory)

    # Generate 5 entries from iterator and save checkpoint.
    outputs = self.gen_outputs(ds_fn, [], 5, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(5))

    # Restore from checkpoint and produce the rest of the elements from the
    # iterator.
    outputs.extend(
        self.gen_outputs(
            ds_fn, [],
            self.num_outputs - 5,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, self.expected_outputs())

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointBeforeOneEpochThenRunFewSteps(self, is_memory):
    # [FUNCTION DOC]
    # Scenario: run 8 elements but checkpoint after 5 only, then restore and read the rest.
    # Expects the first 5 plus the restored remainder to equal the full expected sequence.
    ds_fn = self.make_dataset_fn(is_memory)

    # Generate 8 entries from iterator but save checkpoint after producing 5.
    outputs = self.gen_outputs(
        ds_fn, [5], 8, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, range(8))

    outputs = outputs[:5]
    outputs.extend(
        self.gen_outputs(
            ds_fn, [],
            self.num_outputs - 5,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, self.expected_outputs())

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointAfterOneEpoch(self, is_memory):
    # [FUNCTION DOC]
    # Scenario: read 15 elements (past the first epoch, so the cache is complete), checkpoint,
    # restore and read the remaining 15. Expects the full expected sequence.
    ds_fn = self.make_dataset_fn(is_memory)

    # Generate 15 entries from iterator and save checkpoint.
    outputs = self.gen_outputs(ds_fn, [], 15, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(5)))

    # Restore from checkpoint and produce the rest of the elements from the
    # iterator.
    outputs.extend(
        self.gen_outputs(
            ds_fn, [],
            self.num_outputs - 15,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, self.expected_outputs())

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointAfterOneEpochThenRunFewSteps(self, is_memory):
    # [FUNCTION DOC]
    # Scenario: run 18 elements but checkpoint after 15, then restore and read the remaining 15.
    # Expects the stitched sequence to equal three copies of the range.
    ds_fn = self.make_dataset_fn(is_memory)

    # Generate 18 entries from iterator but save checkpoint after producing 15.
    outputs = self.gen_outputs(
        ds_fn, [15], 18, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(8)))

    outputs = list(range(10)) + list(range(5)) + self.gen_outputs(
        ds_fn, [],
        self.num_outputs - 15,
        ckpt_saved=True,
        verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointBeforeOneEpochButRunCompleteEpoch(self, is_memory):
    # [FUNCTION DOC]
    # Scenario: run 13 elements (completing the cache) but checkpoint after 5, while the
    # iterator was still in cache-write mode. After restoring, the iterator must fall back to
    # read mode because the cache is complete, and produce the remaining 25 elements correctly.
    ds_fn = self.make_dataset_fn(is_memory)

    # Generate 13 entries from iterator but save checkpoint after producing 5.
    outputs = self.gen_outputs(
        ds_fn, [5], 13, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(3)))

    # Since we ran for more than one epoch, the cache was completely written.
    # The ckpt was saved when the iterator was in cache-write mode. Test that
    # the iterator falls back to read mode after restoring if the cache has
    # been completely written.

    outputs = list(range(5)) + self.gen_outputs(
        ds_fn, [],
        self.num_outputs - 5,
        ckpt_saved=True,
        verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointUnusedWriterIterator(self, is_memory):
    # [FUNCTION DOC]
    # Scenario: checkpoint before any element is requested (zero outputs), then restore and read
    # all 30 elements. Expects an empty first output and three copies of the range afterwards.
    ds_fn = self.make_dataset_fn(is_memory)

    # Checkpoint before get_next is called even once.
    outputs = self.gen_outputs(ds_fn, [], 0, verify_exhausted=False)
    self.assertSequenceEqual(outputs, [])

    outputs = self.gen_outputs(
        ds_fn, [], self.num_outputs, ckpt_saved=True, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testCheckpointUnusedMidwayWriterIterator(self, is_memory):
    # [FUNCTION DOC]
    # Scenario: read 5 elements and checkpoint; restore, read nothing and checkpoint again;
    # restore once more and read the rest. Expects no elements lost or duplicated overall.
    ds_fn = self.make_dataset_fn(is_memory)

    # Produce 5 elements and checkpoint.
    outputs = self.gen_outputs(ds_fn, [], 5, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(5))

    # Restore from checkpoint, then produce no elements and checkpoint.
    outputs.extend(
        self.gen_outputs(ds_fn, [], 0, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(5))

    # Restore from checkpoint and produce rest of the elements.
    outputs.extend(
        self.gen_outputs(
            ds_fn, [],
            self.num_outputs - 5,
            ckpt_saved=True,
            verify_exhausted=False))
    self.assertSequenceEqual(outputs, list(range(10)) * 3)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testUnusedCheckpointError(self, is_memory):
    # [FUNCTION DOC]
    # Scenario: read 5 elements and save a checkpoint, then build a new iterator without
    # restoring. For the memory cache this simply produces the full sequence; for the file
    # cache it must raise an already-exists error because a partial cache shard is present.
    ds_fn = self.make_dataset_fn(is_memory)

    # Produce 5 elements and save ckpt.
    outputs = self.gen_outputs(ds_fn, [], 5, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(5))

    if is_memory:
      outputs = self.gen_outputs(
          ds_fn, [], self.num_outputs, verify_exhausted=False)
      self.assertSequenceEqual(outputs, self.expected_outputs())
    else:
      # Since the complete cache has not been written, a new iterator which does
      # not restore the checkpoint will throw an error since there is a partial
      # cache shard.
      with self.assertRaises(errors.AlreadyExistsError):
        outputs = self.gen_outputs(
            ds_fn, [], self.num_outputs, verify_exhausted=False)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(is_memory=[True, False])))
  def testIgnoreCheckpointIfCacheWritten(self, is_memory):
    # [FUNCTION DOC]
    # Scenario: read 15 elements (completing the cache) and checkpoint, then build a new
    # iterator without restoring. Expects it to use the completed cache and yield three copies
    # of the range.
    ds_fn = self.make_dataset_fn(is_memory)

    # Produce 15 elements and save ckpt. This will write the complete cache.
    outputs = self.gen_outputs(ds_fn, [], 15, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) + list(range(5)))

    # Build the iterator again but do not restore from ckpt. Since the cache
    # has already been written we should be able to use it.
    outputs = self.gen_outputs(
        ds_fn, [], self.num_outputs, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(10)) * 3)


class CacheRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests random access (reading an element by index) on cached datasets: invalid indexes,
  # simple and multi-component elements, and inputs that are not randomly accessible, have
  # unknown cardinality or are infinite.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 3, 4])))
  def testInvalidIndex(self, index):
    # [FUNCTION DOC]
    # Scenario: cached dataset of three elements accessed at an out-of-bounds index (negative or
    # at/after the end). Expects an out-of-range error.
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3]).cache()
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index))

  @combinations.generate(test_base.default_test_combinations())
  def testCacheRangeDataset(self):
    # [FUNCTION DOC]
    # Scenario: random access over a cached range of 10; expects every element by index.
    dataset = dataset_ops.Dataset.range(10).cache()
    expected_elements = list(range(10))
    self.verifyRandomAccess(dataset, expected_elements)

  @combinations.generate(test_base.default_test_combinations())
  def testCacheOneDimensionalElements(self):
    # [FUNCTION DOC]
    # Scenario: random access over a cached dataset sliced from a flat list; expects each scalar.
    tensor = [1, 2, 3]
    dataset = dataset_ops.Dataset.from_tensor_slices(tensor).cache()
    self.verifyRandomAccess(dataset, tensor)

  @combinations.generate(test_base.default_test_combinations())
  def testCacheTwoDimensionalElements(self):
    # [FUNCTION DOC]
    # Scenario: random access over a cached dataset sliced from a nested list; expects each row.
    tensor = [[1, 2], [3, 4]]
    dataset = dataset_ops.Dataset.from_tensor_slices(tensor).cache()
    self.verifyRandomAccess(dataset, tensor)

  @combinations.generate(test_base.default_test_combinations())
  def testCacheThreeComponents(self):
    # [FUNCTION DOC]
    # Scenario: random access over a cached dataset with three components per element; expects
    # the two tuples of three values.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        ([1, 2], [3, 4], [5, 6])).cache()
    expected = [(1, 3, 5), (2, 4, 6)]
    self.verifyRandomAccess(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testCacheInputDatasetNotRandomlyAccessible(self):
    # [FUNCTION DOC]
    # Scenario: the cache input is a scan (running sum) dataset, which does not support random
    # access itself. Expects random access through the cache to return the running sums.
    dataset = dataset_ops.Dataset.range(10)
    initial_state = constant_op.constant(0, dtypes.int64)
    scan_func = lambda state, i: (state + i, state + i)
    dataset = dataset.scan(
        initial_state=initial_state, scan_func=scan_func).cache()
    expected = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
    self.verifyRandomAccess(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testCacheInputDatasetUnknownCardinality(self):
    # [FUNCTION DOC]
    # Scenario: the cache input is a filtered range (unknown cardinality). Expects random access
    # to return the even numbers below 20 by index.
    dataset = dataset_ops.Dataset.range(20).filter(
        lambda x: math_ops.equal(x % 2, 0)).cache()
    expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    self.verifyRandomAccess(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testCacheInputDatasetInfiniteCardinality(self):
    # [FUNCTION DOC]
    # Scenario: the cache input is an infinitely repeated filtered range. Random access caches
    # through the requested index; expects the first twelve elements (evens, wrapping around).
    dataset = dataset_ops.Dataset.range(20).filter(
        lambda x: math_ops.equal(x % 2, 0)).repeat(-1).cache()
    expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 2]
    # Since the dataset has infinite cardinality, random access with caching
    # will cache through the requested index. In this case, random access
    # with caching will cache through index 11.
    self.verifyRandomAccessInfiniteCardinality(dataset, expected)


class CacheGlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests global shuffling applied on top of a cached (and prefetched, optionally repeated)
  # dataset.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              repetitions=[1, 2],
              seed=[None, 42],
              reshuffle_each_iteration=[True, False])))
  def test(
      self,
      dataset_range: int,
      repetitions: int,
      seed: Optional[int],
      reshuffle_each_iteration: bool):
    # [FUNCTION DOC]
    # Scenario: a cached, prefetched range (repeated when repetitions exceeds one) is globally
    # shuffled with an optional seed and the reshuffle flag. Expects the output to contain the
    # same elements as the unshuffled sequence, in a different order, with length equal to the
    # dataset cardinality.
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    if repetitions > 1:
      dataset = dataset.repeat(repetitions)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    expected = list(range(0, dataset_range)) * repetitions
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)
    self.assertLen(dataset_output, self.evaluate(dataset.cardinality()))


class CacheGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for a globally shuffled dataset built on a cache, run with the standard
  # verification functions of the checkpoint test base.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              repetitions=[1, 2],
              reshuffle_each_iteration=[True, False])))
  def test(
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      repetitions: int,
      reshuffle_each_iteration: bool):
    # [FUNCTION DOC]
    # Scenario: builds a globally shuffled (seed 42) cached and prefetched range, optionally
    # repeated, and passes it to the given verification function with the expected number of
    # outputs. Order-insensitive comparison is used when reshuffling each iteration.

    def _build_dataset() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: builds the globally shuffled cached dataset checked by the verification function.
      # Takes no arguments and returns the dataset.
      dataset = dataset_ops.Dataset.range(dataset_range)
      dataset = dataset.cache()
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      if repetitions > 1:
        dataset = dataset.repeat(repetitions)
      return global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=dataset_range * repetitions,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
