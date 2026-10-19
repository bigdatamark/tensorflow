# [MODULE DOC]
# Unit tests for the dataset snapshot transformation, both the current implementation (shard
# based, via the dataset snapshot method) and the legacy implementation (via the experimental
# legacy snapshot op). Covers reading and writing snapshots, compression, shard functions, reader
# functions, fingerprints, run directories, modes, expiry, shuffle on read, and checkpointing.
# Also defines small module level helpers for filtering snapshot directory listings. Tests create
# and remove temporary directories and TFRecord files. No import time side effects; runs the
# test main when executed directly.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the `SnapshotDataset` transformation."""
import multiprocessing
import os
import shutil
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import snapshot
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


def is_graphdef_file(filename):
  # [FUNCTION DOC]
  # HELPER: used by the snapshot directory assertions to skip debugging graph files.
  # Returns true if the file name ends with the graph text-proto suffix, else false.
  return filename.endswith("-graph.pbtxt")


def is_temp_file(filename):
  # [FUNCTION DOC]
  # HELPER: used by the snapshot directory assertions to skip temporary files.
  # Returns true if the file name contains the temporary marker, else false.
  return "-tmp-" in filename


def listdir_and_filter(dirname, filter_fn):
  # [FUNCTION DOC]
  # HELPER: lists snapshot directories for the assertions.
  # Takes a directory name and a predicate on entry names. Returns the sorted names of the
  # directory entries for which the predicate is true. Raises an OS error if the directory is
  # missing.
  return [path for path in sorted(os.listdir(dirname)) if filter_fn(path)]


class SnapshotTest(tf_record_test_base.TFRecordTestBase,
                   parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the current (shard based) dataset snapshot method, built on the TFRecord test base
  # which can create TFRecord files. Each test gets a fresh "snapshot" directory under the test
  # temp dir, stored in the snapshot directory attribute, which is removed afterwards.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, then creates a fresh "snapshot" sub-directory in the temp dir and
    # remembers it in the snapshot directory attribute.
    super(SnapshotTest, self).setUp()
    tmpdir = self.get_temp_dir()
    tmpdir = os.path.join(tmpdir, "snapshot")
    os.mkdir(tmpdir)
    self._snapshot_dir = tmpdir

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent teardown, then recursively deletes the snapshot directory created in setup.
    super(SnapshotTest, self).tearDown()
    shutil.rmtree(self._snapshot_dir)

  def createTFRecords(self, num_files=10, num_records=100):
    # [FUNCTION DOC]
    # HELPER: fixture creating input TFRecord files.
    # Stores the number of files (default 10) and records per file (default 100) and creates the
    # files via the base class, saving their names in the file name list attribute.
    self._num_files = num_files
    self._num_records = num_records
    self._filenames = self._createFiles()

  def removeTFRecords(self):
    # [FUNCTION DOC]
    # HELPER: fixture deleting the input TFRecord files so that later reads can only be served
    # from the snapshot. Removes each file, then resets the file name list to empty and the file
    # and record counts to nothing.
    for filename in self._filenames:
      os.remove(filename)
    self._filenames = []
    self._num_files = None
    self._num_records = None

  def assertDatasetProducesSet(self, dataset, expected):
    # [FUNCTION DOC]
    # HELPER: custom assertion ignoring element order.
    # Reads exactly as many elements as the expected list holds, asserts they match the expected
    # elements as an unordered collection, then asserts the next fetch raises out-of-range.
    actual = []
    next_fn = self.getNext(dataset)
    for _ in range(len(expected)):
      elem = self.evaluate(next_fn())
      actual.append(elem)
    self.assertCountEqual(actual, expected)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_fn())

  def assertSnapshotDirectoryContains(self, directory, num_fingerprints,
                                      num_runs_per_fingerprint,
                                      num_snapshot_shards_per_run):
    # [FUNCTION DOC]
    # HELPER: custom assertion on the on-disk layout of a snapshot directory.
    # Ignoring graph-definition and temporary files, asserts the directory holds the given number
    # of fingerprint sub-directories; each holds the given number of run directories plus a
    # trailing metadata file; each run directory (polled for up to ten seconds, one second apart,
    # to allow slow writes) holds exactly the given number of shard files named by consecutive
    # eight-digit counters starting at zero with the shard suffix.

    # Ignore the graphdef pbtxts we write for debugging purposes and temporary
    # files that are an artifact of how TF writes files.
    dirlist = listdir_and_filter(
        directory, lambda p: not (is_graphdef_file(p) or is_temp_file(p)))
    self.assertLen(dirlist, num_fingerprints)

    for i in range(num_fingerprints):
      fingerprint_dir = os.path.join(directory, dirlist[i])
      fingerprint_dir_list = listdir_and_filter(fingerprint_dir,
                                                lambda p: not is_temp_file(p))
      self.assertLen(fingerprint_dir_list, num_runs_per_fingerprint + 1)
      self.assertEqual(fingerprint_dir_list[num_runs_per_fingerprint],
                       "snapshot.metadata")

      for j in range(num_runs_per_fingerprint):
        run_dir = os.path.join(fingerprint_dir, fingerprint_dir_list[j])
        run_dirlist = sorted(os.listdir(run_dir))
        # On a heavily loaded system all the snapshot shards may take a
        # little time to get written out to the file system so allow
        # up to 10s for this to happen while checking every 1s to see
        # if it is finished
        for k in range(10):
          if len(run_dirlist) == num_snapshot_shards_per_run:
            break
          time.sleep(1)
          run_dirlist = sorted(os.listdir(run_dir))
        self.assertLen(run_dirlist, num_snapshot_shards_per_run)

        file_counter = 0
        for filename in run_dirlist:
          self.assertEqual(filename, "%08d.shard" % file_counter)
          file_counter += 1

  @combinations.generate(test_base.default_test_combinations())
  def testCreateSnapshotDataset(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test makes no assertions and discards the result; it only checks that
    # applying the snapshot method to a one-element dataset does not raise.
    # Smoke test of constructing a snapshot dataset.
    dataset = dataset_ops.Dataset.from_tensors([1, 2, 3])
    dataset.snapshot(path=self._snapshot_dir)

  @combinations.generate(test_base.default_test_combinations())
  def testReadSnapshotDatasetDefault(self):
    # [FUNCTION DOC]
    # Writes a snapshot of ten TFRecord files of 100 records with default settings; asserts the
    # records are produced and the directory holds one fingerprint, one run and one shard per CPU.
    # Then deletes the input files and asserts a second pipeline reads the same records back from
    # the snapshot.
    self.createTFRecords()
    filenames = self._filenames
    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 100)
    ]

    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.snapshot(path=self._snapshot_dir)
    self.assertDatasetProduces(dataset, expected)
    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=multiprocessing.cpu_count())

    self.removeTFRecords()
    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.snapshot(path=self._snapshot_dir)
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testReadSnapshotDatasetAutoWriteSnappyRead(self):
    # [FUNCTION DOC]
    # Writes a snapshot with automatic compression, deletes the input files, then reads the
    # snapshot back with snappy compression specified; asserts the same records are produced both
    # times.
    self.createTFRecords()
    filenames = self._filenames
    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 100)
    ]

    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.snapshot(path=self._snapshot_dir, compression="AUTO")
    self.assertDatasetProduces(dataset, expected)

    self.removeTFRecords()
    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.snapshot(path=self._snapshot_dir, compression="SNAPPY")
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testReadSnapshotDatasetCustomShardFn(self):
    # [FUNCTION DOC]
    # Writes a snapshot with a shard function always returning zero (as a numpy integer); asserts
    # the records are produced and only one shard file exists. After deleting the inputs, reads
    # back with a shard function returning a plain zero and asserts the same records.
    self.createTFRecords()
    filenames = self._filenames
    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 100)
    ]

    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.snapshot(
        path=self._snapshot_dir, shard_func=lambda _: np.int64(0))
    self.assertDatasetProduces(dataset, expected)
    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=1)

    self.removeTFRecords()
    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.snapshot(
        path=self._snapshot_dir, shard_func=lambda _: 0)
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testReadSnapshotDatasetCustomReaderFn(self):
    # [FUNCTION DOC]
    # Writes a snapshot using a reader function that interleaves shards with cycle length and
    # parallelism of four; asserts the records in order and one shard per CPU. After deleting the
    # inputs, reads back with the same kind of reader function and asserts the records as an
    # unordered set.
    self.createTFRecords()
    filenames = self._filenames
    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 100)
    ]

    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.snapshot(
        path=self._snapshot_dir,
        reader_func=(
            lambda ds: ds.interleave(  # pylint:disable=g-long-lambda
                lambda x: x,
                cycle_length=4,
                num_parallel_calls=4)))
    self.assertDatasetProduces(dataset, expected)
    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=multiprocessing.cpu_count())

    self.removeTFRecords()
    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.snapshot(
        self._snapshot_dir,
        reader_func=(
            lambda ds: ds.interleave(  # pylint:disable=g-long-lambda
                lambda x: x,
                cycle_length=4,
                num_parallel_calls=4)))
    self.assertDatasetProducesSet(dataset2, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotDatasetInvalidShardFn(self):
    # [FUNCTION DOC]
    # A shard function returning a string instead of an integer must raise a type error when the
    # dataset is built or its first element is fetched.
    dataset = dataset_ops.Dataset.range(1000)
    with self.assertRaises(TypeError):
      dataset = dataset.snapshot(
          path=self._snapshot_dir, shard_func=lambda _: "invalid_fn")
      next_fn = self.getNext(dataset)
      self.evaluate(next_fn())

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotDatasetInvalidReaderFn(self):
    # [FUNCTION DOC]
    # A reader function that returns something that is not a dataset (adds one to its input) must
    # raise a type error when the dataset is built or its first element is fetched.
    dataset = dataset_ops.Dataset.range(1000)
    with self.assertRaises(TypeError):
      dataset = dataset.snapshot(
          path=self._snapshot_dir, reader_func=lambda x: x + 1)
      next_fn = self.getNext(dataset)
      self.evaluate(next_fn())

  @combinations.generate(test_base.default_test_combinations())
  def testRoundtripEmptySnapshot(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second dataset is first assigned an empty range but is then overwritten by
    # applying snapshot to the first dataset again, so the fresh range is unused.
    # Snapshots an empty dataset; asserts it produces nothing and the directory has one
    # fingerprint, one run and zero shards. Then re-reads the snapshot and asserts it is empty.
    dataset = dataset_ops.Dataset.range(0)
    dataset = dataset.snapshot(path=self._snapshot_dir)
    self.assertDatasetProduces(dataset, [])
    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=0)

    dataset2 = dataset_ops.Dataset.range(0)
    dataset2 = dataset.snapshot(path=self._snapshot_dir)
    self.assertDatasetProduces(dataset2, [])

  @combinations.generate(test_base.default_test_combinations())
  def testWriteSnapshotDatasetSimple(self):
    # [FUNCTION DOC]
    # Snapshots a range of 1000; asserts all elements are produced and the directory has one
    # fingerprint, one run and one shard per CPU.
    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.snapshot(path=self._snapshot_dir)
    self.assertDatasetProduces(dataset, list(range(1000)))
    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=multiprocessing.cpu_count())

  @combinations.generate(test_base.default_test_combinations())
  def testWriteSnapshotDatasetMultipleFingerprints(self):
    # [FUNCTION DOC]
    # Snapshots two different pipelines (ranges of 1000 and 2000) into one directory; asserts both
    # produce their elements and the directory has two fingerprints with one run each.
    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.snapshot(path=self._snapshot_dir)
    self.assertDatasetProduces(dataset1, list(range(1000)))

    dataset2 = dataset_ops.Dataset.range(2000)
    dataset2 = dataset2.snapshot(path=self._snapshot_dir)
    self.assertDatasetProduces(dataset2, list(range(2000)))

    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=2,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=multiprocessing.cpu_count())

  @combinations.generate(test_base.default_test_combinations())
  def testWriteSnapshotDatasetSameFingerprintMultipleCompleteRuns(self):
    # [FUNCTION DOC]
    # Runs the same snapshot pipeline twice to completion; asserts both produce all elements and
    # only one fingerprint with one run exists (the second run reads the first's snapshot).
    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.snapshot(path=self._snapshot_dir)
    self.assertDatasetProduces(dataset1, list(range(1000)))
    dataset2 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset2.snapshot(path=self._snapshot_dir)
    self.assertDatasetProduces(dataset2, list(range(1000)))

    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=multiprocessing.cpu_count())

  @combinations.generate(test_base.default_test_combinations())
  def testWriteSnapshotDatasetSameFingerprintIncompleteRunRestart(self):
    # [FUNCTION DOC]
    # Two iterators over the same snapshot pipeline are interleaved: each reads 500 elements and
    # then both read the rest, checking elements are in order. Asserts the directory ends with one
    # fingerprint and two runs (the second writer did not reuse the unfinished first run).
    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.snapshot(path=self._snapshot_dir)
    next1 = self.getNext(dataset1)
    for i in range(500):
      self.assertEqual(i, self.evaluate(next1()))

    dataset2 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset2.snapshot(path=self._snapshot_dir)
    next2 = self.getNext(dataset2)
    for i in range(500):
      self.assertEqual(i, self.evaluate(next2()))

    for i in range(500, 1000):
      self.assertEqual(i, self.evaluate(next1()))
      self.assertEqual(i, self.evaluate(next2()))

    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=2,
        num_snapshot_shards_per_run=multiprocessing.cpu_count())

  @combinations.generate(test_base.default_test_combinations())
  def testWriteSnapshotCustomShardFunction(self):
    # [FUNCTION DOC]
    # Enumerates a range of 1000, snapshots with a shard function returning index modulo two, then
    # drops the index with a map. Asserts all elements are produced and two shards exist.
    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.enumerate()
    dataset = dataset.snapshot(
        path=self._snapshot_dir, shard_func=lambda i, _: i % 2)
    dataset = dataset.map(lambda _, elem: elem)
    self.assertDatasetProduces(dataset, list(range(1000)))
    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=2)

  @combinations.generate(test_base.default_test_combinations())
  def testWriteSnapshotDatasetWithTuples(self):
    # [FUNCTION DOC]
    # Snapshots a zip of four ranges (tuple elements); asserts the tuples are produced and the
    # directory has one fingerprint, one run and one shard per CPU.
    dataset1 = dataset_ops.Dataset.range(0, 1000)
    dataset2 = dataset_ops.Dataset.range(1000, 2000)
    dataset3 = dataset_ops.Dataset.range(2000, 3000)
    dataset4 = dataset_ops.Dataset.range(3000, 4000)

    dataset = dataset_ops.Dataset.zip((dataset1, dataset2, dataset3, dataset4))
    dataset = dataset.snapshot(path=self._snapshot_dir)

    expected = list(
        zip(
            range(0, 1000), range(1000, 2000), range(2000, 3000),
            range(3000, 4000)))
    self.assertDatasetProduces(dataset, expected)
    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=multiprocessing.cpu_count())

  @combinations.generate(test_base.default_test_combinations())
  def testWriteSnapshotShuffleSameFingerprint(self):
    # [FUNCTION DOC]
    # Builds the same shuffled-then-snapshotted pipeline twice; asserts each produces all
    # elements (any order) and that only one fingerprint and run exist, i.e. shuffling does not
    # change the fingerprint.

    def make_dataset():
      # [FUNCTION DOC]
      # HELPER: returns a range of 1000, shuffled with a full-size buffer, then snapshotted into
      # the test's snapshot directory.
      dataset = dataset_ops.Dataset.range(1000)
      dataset = dataset.shuffle(1000)
      dataset = dataset.snapshot(path=self._snapshot_dir)
      return dataset

    dataset1 = make_dataset()
    self.assertDatasetProducesSet(dataset1, list(range(1000)))
    dataset2 = make_dataset()
    self.assertDatasetProducesSet(dataset2, list(range(1000)))
    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=multiprocessing.cpu_count())

  @combinations.generate(test_base.default_test_combinations())
  def testReadUsingFlatMap(self):
    # [FUNCTION DOC]
    # Writes a snapshot of a range of 1000, then wraps that dataset in a one-element dataset and
    # flattens it; asserts both produce all elements and only one snapshot run exists.
    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.snapshot(path=self._snapshot_dir)
    self.assertDatasetProduces(dataset, list(range(1000)))
    flat_map = dataset_ops.Dataset.from_tensors(dataset).flat_map(lambda x: x)
    self.assertDatasetProduces(flat_map, list(range(1000)))
    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=multiprocessing.cpu_count())

  @combinations.generate(test_base.default_test_combinations())
  def testReadOptimizableUsingFlatMap(self):
    # [FUNCTION DOC]
    # Like the flat-map read test, but the snapshotted pipeline is a shuffle followed by repeat
    # twice (which the optimizer fuses). Asserts both the direct and flattened reads produce two
    # copies of the range in any order and only one snapshot run exists.
    dataset = dataset_ops.Dataset.range(1000)
    # Will be optimized into ShuffleAndRepeat.
    dataset = dataset.shuffle(10)
    dataset = dataset.repeat(2)
    dataset = dataset.snapshot(path=self._snapshot_dir)
    self.assertDatasetProducesSet(dataset, 2 * list(range(1000)))
    flat_map = dataset_ops.Dataset.from_tensors(dataset).flat_map(lambda x: x)
    self.assertDatasetProducesSet(flat_map, 2 * list(range(1000)))
    self.assertSnapshotDirectoryContains(
        self._snapshot_dir,
        num_fingerprints=1,
        num_runs_per_fingerprint=1,
        num_snapshot_shards_per_run=multiprocessing.cpu_count())

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatAndPrefetch(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue: snapshot random data, then shuffle, batch, repeat
    # forever and prefetch; fetches 30 batches and asserts only that no error occurs.
    """This test reproduces github.com/tensorflow/tensorflow/issues/48903."""
    dataset = dataset_ops.Dataset.from_tensor_slices(np.random.rand(16, 32))
    dataset = dataset.snapshot(path=self._snapshot_dir)
    dataset = dataset.shuffle(buffer_size=16)
    dataset = dataset.batch(16)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    next_element = self.getNext(dataset)
    for _ in range(30):
      self.evaluate(next_element())

  def testName(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the neighboring tests this one has no test-combinations decorator.
    # Snapshots a one-element dataset with an explicit op name and asserts the element is
    # produced.
    dataset = dataset_ops.Dataset.from_tensors(42)
    dataset = dataset.snapshot(path=self._snapshot_dir, name="snapshot")
    self.assertDatasetProduces(dataset, [42])


class LegacySnapshotTest(tf_record_test_base.TFRecordTestBase,
                         parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the legacy snapshot op applied through the dataset apply method. Each test uses a
  # fresh "snapshot" directory (snapshot directory attribute) under the temp dir, removed in
  # teardown. Input TFRecord files are created on demand.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, removes any TFRecord files listed by the base fixture, then creates
    # a fresh "snapshot" sub-directory in the temp dir and remembers it.
    super(LegacySnapshotTest, self).setUp()
    self.removeTFRecords()
    tmpdir = self.get_temp_dir()
    tmpdir = os.path.join(tmpdir, "snapshot")
    os.mkdir(tmpdir)
    self.snapshot_dir = tmpdir

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent teardown, then recursively deletes the snapshot directory.
    super(LegacySnapshotTest, self).tearDown()
    shutil.rmtree(self.snapshot_dir)

  def removeTFRecords(self):
    # [FUNCTION DOC]
    # HELPER: fixture deleting the input TFRecord files so reads can only be served from the
    # snapshot. Removes each file in the file name list and resets the list to empty. Unlike the
    # other test class it does not reset the file and record counts.
    for filename in self._filenames:
      os.remove(filename)
    self._filenames = []

  def setUpTFRecord(self, num_files=10, num_records=10):
    # [FUNCTION DOC]
    # HELPER: fixture creating input TFRecord files. Stores the number of files (default 10) and
    # records per file (default 10) and creates the files via the base class, saving their names.
    self._num_files = num_files
    self._num_records = num_records
    self._filenames = self._createFiles()

  def makeSnapshotDirectory(self):
    # [FUNCTION DOC]
    # HELPER: returns the path of the snapshot directory created in setup (no side effects).
    return self.snapshot_dir

  def assertSnapshotDirectoryContains(self, directory, num_fingerprints,
                                      num_runs_per_fp, num_snapshot_files):
    # [FUNCTION DOC]
    # HELPER: custom assertion on the layout of a legacy snapshot directory.
    # Ignoring graph-definition and temporary files, asserts the given number of fingerprint
    # directories; each holds the given number of run directories plus a trailing metadata file;
    # each run directory holds exactly the given number of files named by consecutive
    # eight-digit counters from zero with the legacy snapshot suffix.
    # Ignore the graphdef pbtxts we write for debugging purposes and temporary
    # files that are an artifact of how TF writes files.
    dirlist = listdir_and_filter(
        directory, lambda p: not (is_graphdef_file(p) or is_temp_file(p)))
    self.assertLen(dirlist, num_fingerprints)

    for i in range(num_fingerprints):
      fingerprint_dir = os.path.join(directory, dirlist[i])
      fingerprint_dir_list = listdir_and_filter(fingerprint_dir,
                                                lambda p: not is_temp_file(p))
      self.assertLen(fingerprint_dir_list, num_runs_per_fp + 1)
      self.assertEqual(fingerprint_dir_list[num_runs_per_fp],
                       "snapshot.metadata")

      for j in range(num_runs_per_fp):
        run_dir = os.path.join(fingerprint_dir, fingerprint_dir_list[j])
        run_dirlist = sorted(os.listdir(run_dir))
        self.assertLen(run_dirlist, num_snapshot_files)

        file_counter = 0
        for filename in run_dirlist:
          self.assertEqual(filename, "%08d.snapshot" % file_counter)
          file_counter += 1

  @combinations.generate(test_base.default_test_combinations())
  def testWriteDifferentPipelinesInOneDirectory(self):
    # [FUNCTION DOC]
    # Legacy-snapshots ranges of 1000 and 1001 into one directory; asserts each produces its
    # elements and the directory has two fingerprints with one run and one file each.
    tmpdir = self.snapshot_dir

    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir))
    self.assertDatasetProduces(dataset, list(range(1000)))

    dataset = dataset_ops.Dataset.range(1001)
    dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir))
    self.assertDatasetProduces(dataset, list(range(1001)))

    self.assertSnapshotDirectoryContains(tmpdir, 2, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testWriteSnapshotMultipleSimultaneous(self):
    # [FUNCTION DOC]
    # Two iterators over the same legacy snapshot pipeline are read in lockstep for 1000
    # elements, checking values. Asserts only one copy of the metadata/run exists (the loser of
    # the race ran in passthrough mode).
    tmpdir = self.snapshot_dir

    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.apply(snapshot.legacy_snapshot(tmpdir))
    next1 = self.getNext(dataset1)

    dataset2 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir))
    next2 = self.getNext(dataset2)

    for i in range(0, 1000):
      self.assertEqual(i, self.evaluate(next1()))
      self.assertEqual(i, self.evaluate(next2()))

    # we check that only one copy of the metadata has been written, and the
    # one that lost the race would be in passthrough mode.
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testGetNextCreatesDir(self):
    # [FUNCTION DOC]
    # Creates two legacy snapshot iterators with different pipelines but fetches elements from
    # only one; asserts only one fingerprint directory is created (directories appear on first
    # fetch, not on iterator creation).
    tmpdir = self.snapshot_dir

    # We create two iterators but call getNext on only one.
    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.apply(snapshot.legacy_snapshot(tmpdir))
    next1 = self.getNext(dataset1)

    dataset2 = dataset_ops.Dataset.range(1001)
    dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir))
    _ = self.getNext(dataset2)

    for _ in range(1000):
      self.evaluate(next1())

    # We check that only one directory is created.
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[
              snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP,
              snapshot.COMPRESSION_SNAPPY
          ])))
  def testWriteSnapshotSimpleSuccessful(self, compression):
    # [FUNCTION DOC]
    # For each compression type (none, gzip, snappy), legacy-snapshots a range of 1000; asserts
    # all elements are produced and one fingerprint, one run and one file exist.
    tmpdir = self.snapshot_dir

    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(tmpdir, compression=compression))
    self.assertDatasetProduces(dataset, list(range(1000)))

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[
              snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP,
              snapshot.COMPRESSION_SNAPPY
          ])))
  def testWriteSnapshotRepeatAfterwards(self, compression):
    # [FUNCTION DOC]
    # For each compression type, legacy-snapshots a range of ten then repeats it ten times;
    # asserts ten copies are produced and a single fingerprint, run and file exist.
    tmpdir = self.snapshot_dir

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(tmpdir, compression=compression))
    dataset = dataset.repeat(10)
    self.assertDatasetProduces(dataset, list(range(10)) * 10)

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[
              snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP,
              snapshot.COMPRESSION_SNAPPY
          ])))
  def testWriteSnapshotMixTypes(self, compression):
    # [FUNCTION DOC]
    # For each compression type, maps a range of ten to tuples mixing integers and strings,
    # legacy-snapshots, and repeats ten times; asserts the expected tuples are produced ten times
    # and a single fingerprint, run and file exist.
    tmpdir = self.snapshot_dir

    dataset = dataset_ops.Dataset.range(10)

    def map_fn(x):
      # [FUNCTION DOC]
      # HELPER: maps an integer to a tuple of the integer, its string form, the string of twice
      # the integer, and twice the integer.
      return (x, string_ops.as_string(x), string_ops.as_string(2 * x), 2 * x)

    dataset = dataset.map(map_fn)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(tmpdir, compression=compression))
    dataset = dataset.repeat(10)

    expected = []
    for i in range(10):
      expected.append((i, str(i), str(2 * i), 2 * i))
    self.assertDatasetProduces(dataset, expected * 10)

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testSpecifySnapshotNameWriteAndRead(self):
    # [FUNCTION DOC]
    # Legacy-snapshots with a custom snapshot name and repeats ten times; asserts the elements,
    # one snapshot run, and that the custom-named directory and its inner custom entry exist.
    tmpdir = self.snapshot_dir

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(tmpdir, snapshot_name="my_custom_snapshot"))
    dataset = dataset.repeat(10)
    self.assertDatasetProduces(dataset, list(range(10)) * 10)

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)
    self.assertTrue(
        os.path.exists(os.path.join(tmpdir, "custom-my_custom_snapshot")))
    self.assertTrue(
        os.path.exists(
            os.path.join(tmpdir, "custom-my_custom_snapshot", "custom")))

  @combinations.generate(test_base.default_test_combinations())
  def testForcePassthroughMode(self):
    # [FUNCTION DOC]
    # Forces passthrough mode on a repeated range; asserts the data is produced and nothing is
    # written to the snapshot directory.
    tmpdir = self.snapshot_dir

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(tmpdir, mode="passthrough"))
    dataset = dataset.repeat(10)
    self.assertDatasetProduces(dataset, list(range(10)) * 10)

    self.assertSnapshotDirectoryContains(tmpdir, 0, 0, 0)

  @combinations.generate(test_base.default_test_combinations())
  def testForceWriteMode(self):
    # [FUNCTION DOC]
    # Forces write mode on a range of ten repeated ten times; asserts the data is produced and
    # one fingerprint with ten separate runs is written.
    tmpdir = self.snapshot_dir

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, mode="write"))
    dataset = dataset.repeat(10)
    self.assertDatasetProduces(dataset, list(range(10)) * 10)

    # We will end up writing 10 different runs.
    self.assertSnapshotDirectoryContains(tmpdir, 1, 10, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testForceReadMode(self):
    # [FUNCTION DOC]
    # Writes a named snapshot, renames its run directory, then reads with forced read mode naming
    # the renamed snapshot even though the metadata points to the old one. Asserts the data is
    # produced both times and one fingerprint, run and file remain.
    tmpdir = self.snapshot_dir

    # We write a copy of the snapshot first.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(
            tmpdir, mode="write", snapshot_name="my_custom_snapshot"))
    self.assertDatasetProduces(dataset, list(range(10)))

    # We move the run to a new name.
    shutil.move(
        os.path.join(tmpdir, "custom-my_custom_snapshot"),
        os.path.join(tmpdir, "custom-my_custom_snapshot_2"))

    # Even though the snapshot.metadata is pointing to the old run that no
    # longer exists after we moved, we force it to read from the run we specify.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(
            tmpdir, mode="read", snapshot_name="my_custom_snapshot_2"))
    self.assertDatasetProduces(dataset, list(range(10)))

    # We should still have one snapshot and one run.
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testForceReadNonexistentSnapshot(self):
    # [FUNCTION DOC]
    # Forced read mode with no snapshot present must raise a not-found error on the first fetch.
    tmpdir = self.snapshot_dir
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaises(errors.NotFoundError):
      dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, mode="read"))
      get_next = self.getNext(dataset)
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testForceReadNonexistentNamedSnapshot(self):
    # [FUNCTION DOC]
    # Forced read mode with a snapshot name that does not exist must raise a not-found error on
    # the first fetch.
    tmpdir = self.snapshot_dir
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaises(errors.NotFoundError):
      dataset = dataset.apply(
          snapshot.legacy_snapshot(
              tmpdir, mode="read", snapshot_name="my_nonexistent_snapshot"))
      get_next = self.getNext(dataset)
      self.evaluate(get_next())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[
              snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP,
              snapshot.COMPRESSION_SNAPPY
          ])))
  def testReadSnapshotBackAfterWrite(self, compression):
    # [FUNCTION DOC]
    # For each compression type, legacy-snapshots TFRecord input (10 files of 10 records), asserts
    # the records, deletes the input files, and asserts a second pipeline reads the same records
    # from the snapshot.
    self.setUpTFRecord()
    filenames = self._filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 10)
    ]

    tmpdir = self.snapshot_dir
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(tmpdir, compression=compression))
    self.assertDatasetProduces(dataset, expected)

    # remove the original files and try to read the data back only from snapshot
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(
        snapshot.legacy_snapshot(tmpdir, compression=compression))
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testReadShuffledSnapshotAfterWrite(self):
    # [FUNCTION DOC]
    # Writes a legacy snapshot with small shards from 10 files of 50 records, deletes the inputs,
    # then reads with shuffle on read. Asserts the output differs in order from the original but
    # has the same elements, and a third shuffled read again has all elements.
    self.setUpTFRecord(num_files=10, num_records=50)
    filenames = self._filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 50)
    ]

    tmpdir = self.snapshot_dir
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(tmpdir, shard_size_bytes=100))
    self.assertDatasetProduces(dataset, expected)

    # remove the original files and try to read the data back only from snapshot
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(
        snapshot.legacy_snapshot(
            tmpdir, shard_size_bytes=100, shuffle_on_read=True))
    shuffled_elements = self.getDatasetOutput(dataset2)
    # make sure that we don't read the file back in the same order.
    self.assertNotEqual(shuffled_elements, expected)
    self.assertCountEqual(shuffled_elements, expected)

    # make sure all the elements are still there
    dataset3 = core_readers._TFRecordDataset(filenames)
    dataset3 = dataset3.apply(
        snapshot.legacy_snapshot(
            tmpdir, shard_size_bytes=100, shuffle_on_read=True))
    self.assertDatasetProduces(dataset3, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testReadShuffledSnapshotWithSeedAfterWrite(self):
    # [FUNCTION DOC]
    # Writes a legacy snapshot with tiny shards, deletes the inputs, then creates two readers with
    # shuffle on read and the same seed. Asserts the first 500 elements are identical in both.
    self.setUpTFRecord(num_files=10, num_records=50)
    filenames = self._filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 50)
    ]

    tmpdir = self.snapshot_dir
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(tmpdir, shard_size_bytes=10))
    self.assertDatasetProduces(dataset, expected)

    # remove the original files and try to read the data back only from snapshot
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(
        snapshot.legacy_snapshot(
            tmpdir,
            shard_size_bytes=10,
            shuffle_on_read=True,
            shuffle_seed=123456))
    next2 = self.getNext(dataset2)

    dataset3 = core_readers._TFRecordDataset(filenames)
    dataset3 = dataset3.apply(
        snapshot.legacy_snapshot(
            tmpdir,
            shard_size_bytes=10,
            shuffle_on_read=True,
            shuffle_seed=123456))
    next3 = self.getNext(dataset3)

    # make sure that the items are read back in the same order for both datasets
    for _ in range(500):
      res2 = self.evaluate(next2())
      res3 = self.evaluate(next3())
      self.assertEqual(res2, res3)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[
              snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP,
              snapshot.COMPRESSION_SNAPPY
          ])))
  def testReadSnapshotParallelAfterWrite(self, compression):
    # [FUNCTION DOC]
    # For each compression type, writes a legacy snapshot of 5 files of 500 records with two
    # reader threads and a reader buffer, deletes inputs, and reads again with the same settings.
    # Asserts both reads give the expected records in any order.
    self.setUpTFRecord(5, 500)
    filenames = self._filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 5)
        for r in range(0, 500)
    ]

    tmpdir = self.snapshot_dir
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(
            tmpdir,
            shard_size_bytes=1024 * 1024,
            num_reader_threads=2,
            reader_buffer_size=10,
            compression=compression))
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

    # remove the original files and try to read the data back only from
    # snapshot.
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(
        snapshot.legacy_snapshot(
            tmpdir,
            shard_size_bytes=1024 * 1024,
            num_reader_threads=2,
            reader_buffer_size=10,
            compression=compression))
    self.assertDatasetProduces(dataset2, expected, assert_items_equal=True)

  # Not testing Snappy here because Snappy reads currently require a lot of
  # memory.
  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.times(
              combinations.combine(compression=[
                  snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP
              ]),
              combinations.combine(threads=2, size=[1, 2]) +
              combinations.combine(threads=8, size=[1, 4, 8]))))
  def testReadSnapshotBackAfterMultiThreadedWrite(self, compression, threads,
                                                  size):
    # [FUNCTION DOC]
    # Over gzip/no compression and several writer thread counts and writer buffer sizes (snappy
    # is excluded for memory reasons), writes a legacy snapshot of TFRecord input, deletes the
    # inputs and reads back. Asserts the first read in order and the second in any order.
    self.setUpTFRecord()
    filenames = self._filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 10)
    ]

    tmpdir = self.snapshot_dir
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(
            tmpdir,
            compression=compression,
            num_writer_threads=threads,
            writer_buffer_size=size))
    self.assertDatasetProduces(dataset, expected)

    # remove the original files and try to read the data back only from
    # snapshot
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(
        snapshot.legacy_snapshot(tmpdir, compression=compression))
    self.assertDatasetProduces(dataset2, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSameFingerprintWithDifferentInitializationOrder(self):
    # [FUNCTION DOC]
    # Builds the same concatenation of three ranges twice, creating the component datasets in a
    # different order; asserts both produce 0..299 and only one fingerprint, run and file exist.
    tmpdir = self.snapshot_dir

    dataset1 = dataset_ops.Dataset.range(0, 100)
    dataset2 = dataset_ops.Dataset.range(100, 200)
    dataset3 = dataset_ops.Dataset.range(200, 300)

    dataset = dataset1.concatenate(dataset2).concatenate(dataset3)
    dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir))
    self.assertDatasetProduces(dataset, list(range(300)))

    dataset4 = dataset_ops.Dataset.range(200, 300)
    dataset5 = dataset_ops.Dataset.range(100, 200)
    dataset6 = dataset_ops.Dataset.range(0, 100)

    dataset = dataset6.concatenate(dataset5).concatenate(dataset4)
    dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir))
    self.assertDatasetProduces(dataset, list(range(300)))

    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testExpiredSnapshotRewrite(self):
    # [FUNCTION DOC]
    # Starts a legacy snapshot with a one-second pending expiry and reads only half of it; after
    # sleeping two seconds, starts a second identical pipeline and reads half. Asserts one
    # fingerprint with one run after the first, then two runs after the second (the expired
    # unfinished run was not reused).
    tmpdir = self.snapshot_dir

    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.apply(
        snapshot.legacy_snapshot(tmpdir, pending_snapshot_expiry_seconds=1))
    next1 = self.getNext(dataset1)

    # Don't finish reading dataset1, so it is never finalized
    for _ in range(500):
      self.evaluate(next1())
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

    time.sleep(2)

    # Creating dataset2 after we run through dataset1 due to eager mode, where
    # the snapshot state is determined immediately upon dataset creation. We
    # only want to determine the snapshot state for dataset2 after the first
    # snapshot has expired.
    dataset2 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset2.apply(
        snapshot.legacy_snapshot(tmpdir, pending_snapshot_expiry_seconds=1))
    next2 = self.getNext(dataset2)

    for _ in range(500):
      self.evaluate(next2())
    self.assertSnapshotDirectoryContains(tmpdir, 1, 2, 1)

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotArgsCreateNewSnapshot(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second dataset is assigned a new range and then overwritten by applying
    # the second snapshot to the first dataset, so the new range is unused.
    # Reads a legacy snapshot with one shard size, then another with a different shard size;
    # asserts a second fingerprint appears (snapshot arguments affect the fingerprint).
    tmpdir = self.snapshot_dir

    dataset1 = dataset_ops.Dataset.range(1000)
    dataset1 = dataset1.apply(
        snapshot.legacy_snapshot(tmpdir, shard_size_bytes=10000))
    next1 = self.getNext(dataset1)

    for _ in range(1000):
      self.evaluate(next1())
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

    # Create second snapshot with a different shard_size_bytes
    dataset2 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset1.apply(
        snapshot.legacy_snapshot(tmpdir, shard_size_bytes=20000))
    next2 = self.getNext(dataset2)

    for _ in range(1000):
      self.evaluate(next2())
    self.assertSnapshotDirectoryContains(tmpdir, 2, 1, 1)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[
              snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP,
              snapshot.COMPRESSION_SNAPPY
          ])))
  def testSpecifyShardSize(self, compression):
    # [FUNCTION DOC]
    # For each compression type, snapshots ten large (1024 by 1024 float) elements with a 10MB
    # shard size and reads them. Asserts one fingerprint and run, with three files when
    # uncompressed and one file when compressed.
    tmpdir = self.snapshot_dir

    dataset = dataset_ops.Dataset.from_tensor_slices([1.0])
    dataset = dataset.map(lambda x: gen_array_ops.broadcast_to(x, [1024, 1024]))
    dataset = dataset.repeat(10)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(
            tmpdir, shard_size_bytes=10 * 1024 * 1024, compression=compression))
    next_fn = self.getNext(dataset)

    for _ in range(10):
      self.evaluate(next_fn())

    num_files = 1
    if compression == snapshot.COMPRESSION_NONE:
      num_files = 3
    self.assertSnapshotDirectoryContains(tmpdir, 1, 1, num_files)

  @combinations.generate(test_base.default_test_combinations())
  def testAdditionalOperationsAfterReadBack(self):
    # [FUNCTION DOC]
    # Writes a legacy snapshot of TFRecord input, deletes the inputs, reads it back, then reads
    # again with an extra map that strips the first two bytes of each record. Asserts the
    # original records and then the stripped records.
    self.setUpTFRecord()
    filenames = self._filenames

    expected = [
        b"Record %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 10)
    ]

    tmpdir = self.snapshot_dir
    dataset = core_readers._TFRecordDataset(filenames)
    dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir))
    self.assertDatasetProduces(dataset, expected)

    # remove the original files and try to read the data back only from snapshot
    self.removeTFRecords()

    dataset2 = core_readers._TFRecordDataset(filenames)
    dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir))
    self.assertDatasetProduces(dataset2, expected)

    expected_after = [
        b"cord %d of file %d" % (r, f)  # pylint:disable=g-complex-comprehension
        for f in range(0, 10)
        for r in range(0, 10)
    ]

    dataset3 = core_readers._TFRecordDataset(filenames)
    dataset3 = dataset3.apply(snapshot.legacy_snapshot(tmpdir))
    dataset3 = dataset3.map(lambda x: string_ops.substr_v2(x, 2, 1000))
    self.assertDatasetProduces(dataset3, expected_after)


class SnapshotCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                             parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for the current snapshot dataset, using the checkpoint test base's output
  # generation helpers to save and restore iterator state at various points.

  def _build_snapshot_dataset(self, repeat=False):
    # [FUNCTION DOC]
    # HELPER: returns a dataset factory function (not the dataset). The factory ensures a
    # "snapshot" directory exists under the temp dir (remembered in an attribute), builds a range
    # of 100 snapshotted into it, and repeats twice when the repeat flag (default false) is set.

    def ds_fn():
      # [FUNCTION DOC]
      # HELPER: the dataset factory returned by the enclosing builder; creates the snapshot
      # directory if needed and returns the snapshotted (optionally repeated) range of 100.
      self._snapshot_dir = os.path.join(self.get_temp_dir(), "snapshot")
      if not os.path.exists(self._snapshot_dir):
        os.mkdir(self._snapshot_dir)

      dataset = dataset_ops.Dataset.range(100)
      dataset = dataset.snapshot(path=self._snapshot_dir)
      if repeat:
        dataset = dataset.repeat(2)
      return dataset

    return ds_fn

  @combinations.generate(test_base.default_test_combinations())
  def testCheckpointBeforeEpochEndNoRepeat(self):
    # [FUNCTION DOC]
    # Reads 50 of 100 elements and checkpoints, then restores and reads the remaining 50 until
    # exhausted; asserts the concatenated output is the full range in order.
    ds_fn = self._build_snapshot_dataset(repeat=False)
    outputs = self.gen_outputs(ds_fn, [], 50, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(50))
    outputs.extend(
        self.gen_outputs(ds_fn, [], 50, ckpt_saved=True, verify_exhausted=True))
    self.assertSequenceEqual(outputs, range(100))

  @combinations.generate(test_base.default_test_combinations())
  def testCheckpointBeforeOneEpochWithReading(self):
    # [FUNCTION DOC]
    # With a two-epoch repeat, reads 50 elements and checkpoints, then restores and reads 150
    # more; asserts the full sequence is the first 100 elements followed by a second epoch.
    ds_fn = self._build_snapshot_dataset(repeat=True)

    # Generate 50 entries from iterator and save checkpoint.
    outputs = self.gen_outputs(ds_fn, [], 50, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(50)))

    # Restore from checkpoint and produce the rest of the elements from the
    # iterator.
    t = self.gen_outputs(
        ds_fn, [], 150, ckpt_saved=True, verify_exhausted=False)
    outputs.extend(t)
    self.assertSequenceEqual(
        outputs,
        list(range(50)) + list(range(50, 100)) + list(range(100)))

  @combinations.generate(test_base.default_test_combinations())
  def testCheckpointBeforeOneEpochThenRunAFewSteps(self):
    # [FUNCTION DOC]
    # Checkpoints at step 10 while reading 20 elements (not saving at the end); discards the
    # outputs after step 10, restores and reads the remaining 90. Asserts the combined output is
    # the full range in order.
    ds_fn = self._build_snapshot_dataset(repeat=False)
    outputs = self.gen_outputs(
        ds_fn, [10], 20, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, range(20))

    outputs = outputs[:10]
    outputs.extend(
        self.gen_outputs(ds_fn, [], 90, ckpt_saved=True, verify_exhausted=True))
    self.assertSequenceEqual(outputs, range(100))

  @combinations.generate(test_base.default_test_combinations())
  def testCheckpointAfterOneEpoch(self):
    # [FUNCTION DOC]
    # With a two-epoch repeat, reads 110 elements and checkpoints, then restores and reads the
    # remaining 90 until exhausted; asserts the sequence continues correctly into the second
    # epoch.
    ds_fn = self._build_snapshot_dataset(repeat=True)

    # Generate 110 entries from iterator and save checkpoint.
    outputs = self.gen_outputs(ds_fn, [], 110, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(100)) + list(range(10)))

    # Restore from checkpoint and produce the rest of the elements from the
    # iterator.
    t = self.gen_outputs(ds_fn, [], 90, ckpt_saved=True, verify_exhausted=True)
    outputs.extend(t)
    self.assertSequenceEqual(
        outputs,
        list(range(100)) + list(range(10)) + list(range(10, 100)))

  @combinations.generate(test_base.default_test_combinations())
  def testCheckpointAfterOneEpochRunFewSteps(self):
    # [FUNCTION DOC]
    # With a two-epoch repeat, reads 120 elements but checkpoints at 110; truncates the outputs to
    # 110, restores and reads 90 more until exhausted; asserts the continued sequence is correct.
    ds_fn = self._build_snapshot_dataset(repeat=True)

    # Generate 120 entries from iterator and save checkpoint at 110.
    outputs = self.gen_outputs(
        ds_fn, [110], 120, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, list(range(100)) + list(range(20)))

    # Restore from checkpoint and produce the rest of the elements from the
    # iterator.
    outputs = outputs[:110]
    t = self.gen_outputs(ds_fn, [], 90, ckpt_saved=True, verify_exhausted=True)
    outputs.extend(t)
    self.assertSequenceEqual(
        outputs,
        list(range(100)) + list(range(10)) + list(range(10, 100)))


class LegacySnapshotCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                   parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for the legacy snapshot op, including multi-threaded readers/writers, small
  # shards and pending-snapshot expiry, using the checkpoint test base helpers.

  def _build_snapshot_dataset(self,
                              num_threads=1,
                              repeat=False,
                              pending_snapshot_expiry_seconds=-1,
                              shard_size_bytes=None):
    # [FUNCTION DOC]
    # HELPER: returns a dataset factory function (not the dataset). Parameters: reader/writer
    # thread count (default 1; buffers are twice that), repeat flag (default false, repeats
    # twice), pending snapshot expiry in seconds (default -1) and shard size in bytes (optional).

    def ds_fn():
      # [FUNCTION DOC]
      # HELPER: factory returned by the enclosing builder. Ensures the snapshot directory exists,
      # legacy-snapshots a range of 1000 with the configured threads, buffers, expiry and shard
      # size, optionally repeats twice, and disables the prefetch-injection optimization so no
      # prefetched elements are saved in checkpoints. Returns the dataset.
      self.snapshot_dir = os.path.join(self.get_temp_dir(), "snapshot")
      if not os.path.exists(self.snapshot_dir):
        os.mkdir(self.snapshot_dir)
      dataset = dataset_ops.Dataset.range(1000)
      dataset = dataset.apply(
          snapshot.legacy_snapshot(
              self.snapshot_dir,
              num_writer_threads=num_threads,
              writer_buffer_size=2 * num_threads,
              num_reader_threads=num_threads,
              reader_buffer_size=2 * num_threads,
              pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds,
              shard_size_bytes=shard_size_bytes))
      if repeat:
        dataset = dataset.repeat(2)
      # Turn off `inject_prefetch` optimization. Otherwise, prefetched elements
      # are saved and restored in snapshots while tests assume that there is no
      # elements prefetched.
      options = options_lib.Options()
      options.experimental_optimization.inject_prefetch = False
      dataset = dataset.with_options(options)
      return dataset

    return ds_fn

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testSnapshotBeforeEpochEnd(self, pending_snapshot_expiry_seconds):
    # [FUNCTION DOC]
    # Reads 100 of 1000 elements and checkpoints, then restores and reads the remaining 900;
    # asserts the combined output is the full range in order.
    ds_fn = self._build_snapshot_dataset(
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)
    outputs = self.gen_outputs(ds_fn, [], 100, verify_exhausted=False)
    self.assertSequenceEqual(outputs, range(100))
    outputs.extend(
        self.gen_outputs(
            ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(1000))

  @combinations.generate(
      combinations.times(
          test_base.graph_only_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testCheckpointBeforeOneEpochThenRunFewStepsSmallShardMultiThread(
      self, pending_snapshot_expiry_seconds):
    # [FUNCTION DOC]
    # Graph-only. With 100-byte shards, runs a graph reading 100 elements, saves, reads 300 more,
    # and asserts the 400 outputs. Then restores and reads the remaining 900 and asserts the full
    # range. Finally asserts the snapshot directory has two fingerprint-level entries, each non-graph
    # one has two run entries, and each run directory has 258 files.
    ds_fn = self._build_snapshot_dataset(
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds,
        shard_size_bytes=100)

    outputs = []
    with ops.Graph().as_default() as g:
      init_op, get_next_op, saver = self._build_graph(ds_fn)
      with self.session(graph=g) as sess:
        self._initialize(init_op, sess)
        start = 0
        end = 100
        num_iters = end - start
        for _ in range(num_iters):
          outputs.append(sess.run(get_next_op))
        self._save(sess, saver)
        start = 100
        end = 400
        num_iters = end - start
        for _ in range(num_iters):
          outputs.append(sess.run(get_next_op))
    self.assertSequenceEqual(outputs, range(400))

    outputs = outputs[:100]
    outputs.extend(
        self.gen_outputs(
            ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(1000))
    fp_dir_list = os.listdir(self.snapshot_dir)
    self.assertLen(list(fp_dir_list), 2)
    for d in fp_dir_list:
      if not d.endswith("-graph.pbtxt"):
        fp_dir = os.path.join(self.snapshot_dir, d)
        run_dir_list = os.listdir(fp_dir)
        self.assertLen(list(run_dir_list), 2)
        for e in run_dir_list:
          if e != "snapshot.metadata":
            run_dir = os.path.join(fp_dir, e)
            self.assertLen(list(os.listdir(run_dir)), 258)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testCheckpointBeforeOneEpochThenRunFewSteps(
      self, pending_snapshot_expiry_seconds):
    # [FUNCTION DOC]
    # Reads 200 elements but checkpoints after 100; keeps the first 100, restores and reads 900
    # more; asserts the combined output is the full range of 1000 in order.
    ds_fn = self._build_snapshot_dataset(
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)

    # Generate 200 entries from iterator but save checkpoint after producing
    # 100.
    outputs = self.gen_outputs(
        ds_fn, [100], 200, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, range(200))

    outputs = outputs[:100]
    outputs.extend(
        self.gen_outputs(
            ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(1000))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testCheckpointBeforeOneEpochThenRunFewStepsMultipleThreads(
      self, pending_snapshot_expiry_seconds):
    # [FUNCTION DOC]
    # Same scenario as the single-thread version of this test but with two reader/writer threads.
    ds_fn = self._build_snapshot_dataset(
        num_threads=2,
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)

    # Generate 200 entries from iterator but save checkpoint after producing
    # 100.
    outputs = self.gen_outputs(
        ds_fn, [100], 200, verify_exhausted=False, save_checkpoint_at_end=False)
    self.assertSequenceEqual(outputs, range(200))

    outputs = outputs[:100]
    outputs.extend(
        self.gen_outputs(
            ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
    self.assertSequenceEqual(outputs, range(1000))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testCheckpointAfterOneEpoch(self, pending_snapshot_expiry_seconds):
    # [FUNCTION DOC]
    # With a two-epoch repeat, reads 1100 elements and checkpoints, then restores and reads 900
    # more; asserts the sequence continues correctly into the second epoch.
    ds_fn = self._build_snapshot_dataset(
        repeat=True,
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)

    # Generate 1100 entries from iterator and save checkpoint.
    outputs = self.gen_outputs(ds_fn, [], 1100, verify_exhausted=False)
    self.assertSequenceEqual(outputs, list(range(1000)) + list(range(100)))

    # Restore from checkpoint and produce the rest of the elements from the
    # iterator.
    t = self.gen_outputs(
        ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False)
    outputs.extend(t)
    self.assertSequenceEqual(
        outputs,
        list(range(1000)) + list(range(100)) + list(range(900)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
  def testCheckpointAfterOneEpochThenRunFewSteps(
      self, pending_snapshot_expiry_seconds):
    # [FUNCTION DOC]
    # With a two-epoch repeat, reads 1200 elements but checkpoints at 1100; truncates to 1100,
    # restores and reads 900 more; asserts the continued sequence is correct.
    ds_fn = self._build_snapshot_dataset(
        repeat=True,
        pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)

    # Generate 200 entries from iterator but save checkpoint after producing
    # 100.
    outputs = self.gen_outputs(
        ds_fn, [1100],
        1200,
        verify_exhausted=False,
        save_checkpoint_at_end=False)
    self.assertSequenceEqual(
        outputs,
        list(range(1000)) + list(range(100)) + list(range(100)))

    outputs = outputs[:1100]
    t = self.gen_outputs(
        ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False)
    outputs.extend(t)
    self.assertSequenceEqual(
        outputs, (list(range(1000)) + list(range(100)) + list(range(900))))


if __name__ == "__main__":
  test.main()
