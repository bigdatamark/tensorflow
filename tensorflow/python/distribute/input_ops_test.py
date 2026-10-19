# [MODULE DOC]
# Tests for the input pipeline utilities in the input_ops module: automatic sharding of datasets
# built from record, text-line and fixed-length-record files through many dataset transformations
# (flat map, interleave, list files, zip, concatenate, complex pipelines), and cloning of dataset
# graphs (single source, mapped, concatenated, zipped, and datasets producing several variant
# tensors). Includes a private test dataset class. Runs the standard test main when executed.

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
"""Tests for input pipeline modifications for distribution strategies."""

import os

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import structure
from tensorflow.python.distribute import input_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class AutoShardDatasetTest(test.TestCase):
  # [CLASS DOC]
  # Tests the auto-sharding function on datasets read from ten generated files of four records
  # each, with two shards and shard index zero. Fixture state: file count, records per file, shard
  # count, shard index and fixed record size; helper methods create the files and verify output.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the parent setup and initializes the number of files (ten), records per
    # file (four), shard count (two), shard index (zero) and fixed record byte length (ten).
    super(AutoShardDatasetTest, self).setUp()
    self._num_files = 10
    self._num_records = 4
    self._num_shards = 2
    self._shard_index = 0
    self._record_bytes = 10

  def _getNext(self, dataset):
    # [FUNCTION DOC]
    # HELPER: element fetching for the tests, working in both eager and graph modes.
    # Returns a callable that yields the next element of the given dataset: in eager mode the
    # iterator's internal next method; in graph mode a callable returning the get-next tensor of a
    # one-shot iterator (to be evaluated by the caller).
    if context.executing_eagerly():
      iterator = iter(dataset)
      return iterator._next_internal  # pylint: disable=protected-access
    else:
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      get_next = iterator.get_next()
      return lambda: get_next

  def _record(self, r, f):
    # [FUNCTION DOC]
    # HELPER: expected content generator. Returns the bytes of the record text for record number r
    # in file number f.
    return compat.as_bytes("Record %d of file %d" % (r, f))

  def _text_line(self, r, f):
    # [FUNCTION DOC]
    # HELPER: expected content generator. Returns the bytes of the text line for line number r in
    # file number f.
    return compat.as_bytes("Text line %d of file %d" % (r, f))

  def _fixed_length_record(self, r, f):
    # [FUNCTION DOC]
    # HELPER: expected content generator. Returns bytes made of the last decimal digit of the
    # product of record number r and file number f, repeated for the fixed record length.
    return compat.as_bytes(str((r * f) % 10) * self._record_bytes)

  def _createTFRecordFiles(self):
    # [FUNCTION DOC]
    # HELPER: fixture file creation. Writes one record file per file index in the temporary
    # directory, each holding the configured number of records produced by the record generator.
    # Returns the list of file names.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "tf_record.%d.txt" % i)
      filenames.append(fn)
      writer = python_io.TFRecordWriter(fn)
      for j in range(self._num_records):
        record = self._record(j, i)
        writer.write(record)
      writer.close()
    return filenames

  def _createTextFiles(self):
    # [FUNCTION DOC]
    # HELPER: fixture file creation. Writes one text file per file index in the temporary
    # directory. Each file holds the configured number of text lines separated by carriage return
    # plus newline; the final line terminator is omitted except for the first file. Returns the
    # list of file names.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "text_line.%d.txt" % i)
      filenames.append(fn)
      contents = []
      for j in range(self._num_records):
        contents.append(self._text_line(j, i))
        if j + 1 != self._num_records or i == 0:
          contents.append(b"\r\n")
      contents = b"".join(contents)

      with open(fn, "wb") as f:
        f.write(contents)
    return filenames

  def _createFixedLengthRecordFiles(self):
    # [FUNCTION DOC]
    # HELPER: fixture file creation. Writes one binary file per file index in the temporary
    # directory, each holding the configured number of fixed-length records back to back. Returns
    # the list of file names.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "fixed_length_record.%d.txt" % i)
      filenames.append(fn)
      with open(fn, "wb") as f:
        for j in range(self._num_records):
          f.write(self._fixed_length_record(j, i))
    return filenames

  def _verifySimpleShardingOutput(self, dataset, record_fn):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the sharding tests. Parameters: a sharded dataset and a function
    # mapping (record index, file index) to the expected element. Reads the dataset in a session
    # and asserts that elements arrive in order for every record of every file belonging to this
    # shard (files stepping by the shard count from the shard index), then that reading further
    # raises an out-of-range error. Returns nothing.
    next_element_fn = self._getNext(dataset)
    with self.cached_session():
      for f in range(self._shard_index, self._num_files, self._num_shards):
        for r in range(self._num_records):
          self.assertAllEqual(record_fn(r, f), self.evaluate(next_element_fn()))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element_fn())

  @test_util.run_in_graph_and_eager_modes
  def testTFRecordDataset(self):
    # [FUNCTION DOC]
    # Scenario: a record-file dataset over all files is auto-sharded. Expects only the records of
    # this shard's files, in order.
    dataset = readers.TFRecordDataset(self._createTFRecordFiles())
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._record)

  @test_util.run_in_graph_and_eager_modes
  def testFlatMap(self):
    # [FUNCTION DOC]
    # Scenario: a dataset of file names flat-mapped into record datasets is auto-sharded. Expects
    # only the records of this shard's files, in order.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        self._createTFRecordFiles())
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._record)

  @test_util.run_in_graph_and_eager_modes
  def testInterleave(self):
    # [FUNCTION DOC]
    # Scenario: file names interleaved into record datasets with cycle length four and block length
    # equal to the records per file, then auto-sharded. Because the block length equals the file
    # size, output stays in file order; expects this shard's records in order.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        self._createTFRecordFiles())
    dataset = dataset.interleave(
        readers.TFRecordDataset, cycle_length=4, block_length=self._num_records)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    # Since block_length == num records in each file, the output will still
    # contain records in order of files.
    self._verifySimpleShardingOutput(dataset, self._record)

  @test_util.run_in_graph_and_eager_modes
  def testListfiles(self):
    # [FUNCTION DOC]
    # Scenario: files are discovered by a file pattern (unshuffled), flat-mapped to record
    # datasets and auto-sharded. Collects all elements and asserts they equal this shard's
    # expected records in order, and that reading past the end raises an out-of-range error.
    filenames = self._createTFRecordFiles()
    file_pattern = filenames[0].rsplit(os.sep, 1)[0] + "/tf_record.*.txt"
    dataset = dataset_ops.Dataset.list_files(file_pattern, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    next_element_fn = self._getNext(dataset)
    actual, expected = [], []
    for f in range(self._shard_index, self._num_files, self._num_shards):
      for r in range(self._num_records):
        actual.append(self.evaluate(next_element_fn()))
        expected.append(self._record(r, f))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element_fn())
    self.assertAllEqual(expected, actual)

  @test_util.run_in_graph_and_eager_modes
  def testComplexPipeline(self):
    # [FUNCTION DOC]
    # Scenario: a pipeline of shuffled file names, flat-mapped records, prefetch, shuffle, five
    # epochs of repeat, identity map, batch of two and prefetch is auto-sharded. Reads the expected
    # number of batches, asserts out-of-range afterwards, and compares the sorted collected
    # records with the sorted expected records of this shard repeated for every epoch.
    # Setup a complex input pipeline.
    batch_size = 2
    num_epochs = 5
    dataset = dataset_ops.Dataset.from_tensor_slices(
        self._createTFRecordFiles())
    dataset = dataset.shuffle(buffer_size=self._num_files)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.shuffle(2 * self._num_files * self._num_records)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(lambda x: x)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=None)

    # Auto shard.
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    # Verify output.
    next_element_fn = self._getNext(dataset)
    actual = []
    num_iterations = (self._num_files * self._num_records * num_epochs) // (
        self._num_shards * batch_size)
    for _ in range(num_iterations):
      actual.extend(self.evaluate(next_element_fn()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element_fn())

    expected = []
    for f in range(0, self._num_files, self._num_shards):
      for r in range(self._num_records):
        expected.append(self._record(r, f))
    expected *= num_epochs

    self.assertAllEqual(sorted(expected), sorted(actual))

  @test_util.run_in_graph_and_eager_modes
  def testZip(self):
    # [FUNCTION DOC]
    # Scenario: a record dataset zipped with a text-line dataset is auto-sharded. Expects pairs of
    # (record, text line) for this shard's files, in order.
    dataset1 = readers.TFRecordDataset(self._createTFRecordFiles())
    dataset2 = readers.TextLineDataset(self._createTextFiles())

    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    record_fn = lambda r, f: (self._record(r, f), self._text_line(r, f))
    self._verifySimpleShardingOutput(dataset, record_fn)

  @test_util.run_in_graph_and_eager_modes
  def testConcat(self):
    # [FUNCTION DOC]
    # Scenario: a record dataset concatenated with a text-line dataset is auto-sharded. Expects
    # this shard's records first, then this shard's text lines, then an out-of-range error.
    dataset1 = readers.TFRecordDataset(self._createTFRecordFiles())
    dataset2 = readers.TextLineDataset(self._createTextFiles())

    dataset = dataset1.concatenate(dataset2)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    next_element_fn = self._getNext(dataset)
    for f in range(self._shard_index, self._num_files, self._num_shards):
      for r in range(self._num_records):
        self.assertAllEqual(
            self._record(r, f), self.evaluate(next_element_fn()))
    for f in range(self._shard_index, self._num_files, self._num_shards):
      for r in range(self._num_records):
        self.assertAllEqual(
            self._text_line(r, f), self.evaluate(next_element_fn()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element_fn())

  @test_util.run_in_graph_and_eager_modes
  def testTextLineReader(self):
    # [FUNCTION DOC]
    # Scenario: a text-line dataset over all text files is auto-sharded. Expects this shard's
    # lines in order.
    dataset = readers.TextLineDataset(self._createTextFiles())

    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._text_line)

  @test_util.run_in_graph_and_eager_modes
  def testTextLineReaderWithFlatMap(self):
    # [FUNCTION DOC]
    # Scenario: a text-line dataset over all text files is auto-sharded (despite the name, no flat
    # map is actually used). Expects this shard's lines in order.
    # KNOWN ISSUE: the name mentions a flat map but the body builds the same plain text-line
    # dataset as the plain text-line reader test.
    dataset = readers.TextLineDataset(self._createTextFiles())
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._text_line)

  @test_util.run_in_graph_and_eager_modes
  def testFixedLengthReaderWithFlatMap(self):
    # [FUNCTION DOC]
    # Scenario: a fixed-length-record dataset over all fixed-length files is auto-sharded (no flat
    # map is actually used despite the name). Expects this shard's records in order.
    # KNOWN ISSUE: the name mentions a flat map but none is used in the body.
    dataset = readers.FixedLengthRecordDataset(
        self._createFixedLengthRecordFiles(), self._record_bytes)
    dataset = input_ops.auto_shard_dataset(
        dataset, self._num_shards, self._shard_index)

    self._verifySimpleShardingOutput(dataset, self._fixed_length_record)


# A dataset that creates two variant tensors.
class _TestDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # HELPER: test dataset for the cloning tests. A dataset wrapping an input dataset through a
  # prefetch operation followed by a model operation, so its graph has two variant tensors in
  # sequence. Preserves the input's element structure.

  def __init__(self, input_dataset):
    # [FUNCTION DOC]
    # Constructor. Parameter: the input dataset. Builds a prefetch (buffer size one) variant
    # tensor from the input's variant tensor, then a model variant tensor from that, and
    # initializes the parent with the input dataset and the final variant tensor. Adds operations
    # to the current graph.
    self._input_dataset = input_dataset
    temp_variant_tensor = gen_dataset_ops.prefetch_dataset(
        input_dataset._variant_tensor,
        buffer_size=1,
        **self._flat_structure)
    variant_tensor = gen_dataset_ops.model_dataset(
        temp_variant_tensor, **self._flat_structure)
    super(_TestDataset, self).__init__(input_dataset, variant_tensor)


class CloneDatasetTest(test.TestCase):
  # [CLASS DOC]
  # Tests the private dataset cloning function of the input_ops module by cloning various
  # datasets and comparing structure and first values. Runs as deprecated version-one (graph) tests.

  def _assert_datasets_equal(self, ds1, ds2):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the cloning tests. Asserts the two datasets have compatible
    # element specs, creates initializable iterators on both, initializes them in a cached session,
    # fetches one element from each and asserts the two values are equal. Returns nothing.
    # First lets assert the structure is the same.
    self.assertTrue(
        structure.are_compatible(ds1.element_spec, ds2.element_spec))

    # Now create iterators on both and assert they produce the same values.
    it1 = dataset_ops.make_initializable_iterator(ds1)
    it2 = dataset_ops.make_initializable_iterator(ds2)

    get_next1 = it1.get_next()
    get_next2 = it2.get_next()

    with self.cached_session():
      self.evaluate([it1.initializer, it2.initializer])
      val1, val2 = self.evaluate([get_next1, get_next2])
      self.assertEqual(val1, val2)

  @test_util.run_deprecated_v1
  def testOnlySource(self):
    # [FUNCTION DOC]
    # Scenario: clone a plain range dataset (a single source operation); expects the clone to match
    # the original.
    ds = dataset_ops.Dataset.range(10)
    cloned_ds = input_ops._clone_dataset(ds)
    self._assert_datasets_equal(ds, cloned_ds)

  @test_util.run_deprecated_v1
  def testSimplePipeline(self):
    # [FUNCTION DOC]
    # Scenario: clone a range dataset mapped with the square function; expects the clone to match
    # the original.
    ds = dataset_ops.Dataset.range(10).map(math_ops.square)
    cloned_ds = input_ops._clone_dataset(ds)
    self._assert_datasets_equal(ds, cloned_ds)

  @test_util.run_deprecated_v1
  def testConcat(self):
    # [FUNCTION DOC]
    # Scenario: clone the concatenation of two range datasets; expects the clone to match the
    # original.
    ds1 = dataset_ops.Dataset.range(10)
    ds2 = dataset_ops.Dataset.range(10)
    ds = ds1.concatenate(ds2)
    cloned_ds = input_ops._clone_dataset(ds)
    self._assert_datasets_equal(ds, cloned_ds)

  @test_util.run_deprecated_v1
  def testZip(self):
    # [FUNCTION DOC]
    # Scenario: clone the zip of two range datasets; expects the clone to match the original.
    ds1 = dataset_ops.Dataset.range(10)
    ds2 = dataset_ops.Dataset.range(10)
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    cloned_ds = input_ops._clone_dataset(ds)
    self._assert_datasets_equal(ds, cloned_ds)

  @test_util.run_deprecated_v1
  def testMultipleVariantTensors(self):
    # [FUNCTION DOC]
    # Scenario: clone a range dataset wrapped by the test dataset that produces two chained
    # variant tensors; expects the clone to match the original.
    ds = dataset_ops.Dataset.range(10)
    ds = _TestDataset(ds)
    cloned_ds = input_ops._clone_dataset(ds)
    self._assert_datasets_equal(ds, cloned_ds)


if __name__ == "__main__":
  test.main()
