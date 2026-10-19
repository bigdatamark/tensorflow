# [MODULE DOC]
# Tests for the fixed-length-record file reader dataset of the tf.data API. Provides a shared
# test base class that writes small binary files (header, fixed-size records, footer; optionally
# gzip or zlib compressed) into the test temp directory, a functional test class (reading with and
# without compression, buffering, parallel reads, wrong record size error, path objects, naming)
# and a checkpoint test class. No import-time side effects; running as a script starts the
# TensorFlow test runner.
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
"""Tests for `tf.data.FixedLengthRecordDataset`."""
import gzip
import os
import pathlib
import zlib

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class FixedLengthRecordDatasetTestBase(test_base.DatasetTestBase):
  # [CLASS DOC]
  # HELPER: shared base class for the fixed-length-record tests in this module.
  # Holds the file layout settings (number of files, records per file, header, record and footer
  # byte sizes) and offers helpers to compute expected record contents and to create the files.
  """Base class for setting up and testing FixedLengthRecordDataset."""

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: calls the parent setup, then sets the layout: 2 files, 7 records per
    # file, a 5-byte header, 3-byte records and a 2-byte footer. Returns nothing.
    super(FixedLengthRecordDatasetTestBase, self).setUp()
    self._num_files = 2
    self._num_records = 7
    self._header_bytes = 5
    self._record_bytes = 3
    self._footer_bytes = 2

  def _record(self, f, r):
    # [FUNCTION DOC]
    # HELPER: computes the expected bytes of record r in file f. The record is the decimal text of
    # twice the file index plus the record index, repeated to fill the record size. Inputs are two
    # integers; returns a byte string.
    return compat.as_bytes(str(f * 2 + r) * self._record_bytes)

  def _createFiles(self, compression_type=None):
    # [FUNCTION DOC]
    # HELPER: writes the test input files into the test temp directory and returns their paths as
    # a list of strings. Each file holds the header filler, all records, then the footer filler.
    # compression_type (optional, default none) may be gzip or zlib to compress the contents;
    # any other value raises a value error. Overwrites existing files of the same names.
    filenames = []
    for i in range(self._num_files):
      fn = os.path.join(self.get_temp_dir(), "fixed_length_record.%d.txt" % i)
      filenames.append(fn)

      contents = []
      contents.append(b"H" * self._header_bytes)
      for j in range(self._num_records):
        contents.append(self._record(i, j))
      contents.append(b"F" * self._footer_bytes)
      contents = b"".join(contents)

      if not compression_type:
        with open(fn, "wb") as f:
          f.write(contents)
      elif compression_type == "GZIP":
        with gzip.GzipFile(fn, "wb") as f:
          f.write(contents)
      elif compression_type == "ZLIB":
        contents = zlib.compress(contents)
        with open(fn, "wb") as f:
          f.write(contents)
      else:
        raise ValueError("Unsupported compression_type", compression_type)

    return filenames


class FixedLengthRecordDatasetTest(FixedLengthRecordDatasetTestBase,
                                   parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests of the fixed-length-record dataset reader, using the files created by the
  # shared base class.

  def _test(self, compression_type=None):
    # [FUNCTION DOC]
    # HELPER: common scenario for the compression tests. Creates the files with the given
    # compression type (optional, default none) and checks reading: file 0 alone, file 1 alone,
    # both files, 10 repeated epochs through both files (then an out-of-range error), and 10
    # epochs batched by the number of records (one batch per file). Returns nothing.
    test_filenames = self._createFiles(compression_type=compression_type)

    def dataset_fn(filenames, num_epochs, batch_size=None):
      # [FUNCTION DOC]
      # HELPER: builds a reader dataset over the given file names with the configured layout and
      # compression, repeated num_epochs times, and batched by batch_size if given (optional).
      # Returns the dataset.
      repeat_dataset = readers.FixedLengthRecordDataset(
          filenames,
          self._record_bytes,
          self._header_bytes,
          self._footer_bytes,
          compression_type=compression_type).repeat(num_epochs)
      if batch_size:
        return repeat_dataset.batch(batch_size)
      return repeat_dataset

    # Basic test: read from file 0.
    self.assertDatasetProduces(
        dataset_fn([test_filenames[0]], 1),
        expected_output=[
            self._record(0, i) for i in range(self._num_records)
        ])

    # Basic test: read from file 1.
    self.assertDatasetProduces(
        dataset_fn([test_filenames[1]], 1),
        expected_output=[
            self._record(1, i) for i in range(self._num_records)
        ])

    # Basic test: read from both files.
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(
        dataset_fn(test_filenames, 1), expected_output=expected_output)

    # Test repeated iteration through both files.
    get_next = self.getNext(dataset_fn(test_filenames, 10))
    for _ in range(10):
      for j in range(self._num_files):
        for i in range(self._num_records):
          self.assertEqual(self._record(j, i), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    # Test batched and repeated iteration through both files.
    get_next = self.getNext(dataset_fn(test_filenames, 10, self._num_records))
    for _ in range(10):
      for j in range(self._num_files):
        self.assertAllEqual(
            [self._record(j, i) for i in range(self._num_records)],
            self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testNoCompression(self):
    # [FUNCTION DOC]
    # Scenario: runs the common read scenario on uncompressed files. Returns nothing.
    self._test()

  @combinations.generate(test_base.default_test_combinations())
  def testGzipCompression(self):
    # [FUNCTION DOC]
    # Scenario: runs the common read scenario on gzip-compressed files. Returns nothing.
    self._test(compression_type="GZIP")

  @combinations.generate(test_base.default_test_combinations())
  def testZlibCompression(self):
    # [FUNCTION DOC]
    # Scenario: runs the common read scenario on zlib-compressed files. Returns nothing.
    self._test(compression_type="ZLIB")

  @combinations.generate(test_base.default_test_combinations())
  def testBuffering(self):
    # [FUNCTION DOC]
    # Scenario: reads both uncompressed files with a small read buffer size of 10 bytes and
    # asserts all records of both files are produced in order. Returns nothing.
    test_filenames = self._createFiles()
    dataset = readers.FixedLengthRecordDataset(
        test_filenames,
        self._record_bytes,
        self._header_bytes,
        self._footer_bytes,
        buffer_size=10)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testParallelRead(self):
    # [FUNCTION DOC]
    # Scenario: reads both files with buffer size 10 and 4 parallel reads; asserts all records
    # are produced, ignoring order. Returns nothing.
    test_filenames = self._createFiles()
    dataset = readers.FixedLengthRecordDataset(
        test_filenames,
        self._record_bytes,
        self._header_bytes,
        self._footer_bytes,
        buffer_size=10,
        num_parallel_reads=4)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output,
                               assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testWrongSize(self):
    # [FUNCTION DOC]
    # Scenario: reads with a record length one byte larger than the real one. Asserts an
    # invalid-argument error whose message states that the body length (21 bytes, excluding the
    # 5-byte header and 2-byte footer) of the first file is not an exact multiple of the record
    # length (4 bytes). Returns nothing.
    test_filenames = self._createFiles()
    dataset = readers.FixedLengthRecordDataset(
        test_filenames,
        self._record_bytes + 1,  # Incorrect record length.
        self._header_bytes,
        self._footer_bytes,
        buffer_size=10)
    self.assertDatasetProduces(
        dataset,
        expected_error=(
            errors.InvalidArgumentError,
            r"Excluding the header \(5 bytes\) and footer \(2 bytes\), input "
            r"file \".*fixed_length_record.0.txt\" has body length 21 bytes, "
            r"which is not an exact multiple of the record length \(4 bytes\).")
        )

  @combinations.generate(test_base.default_test_combinations())
  def testPathlib(self):
    # [FUNCTION DOC]
    # Scenario: passes the file names as path objects instead of strings, with buffer size 10 and
    # 4 parallel reads; asserts all records are produced, ignoring order. Returns nothing.
    test_filenames = self._createFiles()
    test_filenames = [pathlib.Path(f) for f in test_filenames]
    dataset = readers.FixedLengthRecordDataset(
        test_filenames,
        self._record_bytes,
        self._header_bytes,
        self._footer_bytes,
        buffer_size=10,
        num_parallel_reads=4)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output,
                               assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Scenario: reads both files with an explicit dataset name; asserts all records are produced
    # in order. Returns nothing.
    test_filenames = self._createFiles()
    dataset = readers.FixedLengthRecordDataset(
        test_filenames,
        self._record_bytes,
        self._header_bytes,
        self._footer_bytes,
        name="fixed_length_record_dataset")
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output)


class FixedLengthRecordDatasetCheckpointTest(
    FixedLengthRecordDatasetTestBase, checkpoint_test_base.CheckpointTestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for the fixed-length-record dataset, combining the shared file-creating base
  # class with the checkpoint test base class.

  def _build_dataset(self, num_epochs, compression_type=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset for the checkpoint test.
    # KNOWN ISSUE: the compression_type parameter is accepted but ignored; files are always
    # created and read without compression.
    # Creates uncompressed files and returns a reader dataset over them repeated num_epochs times.
    filenames = self._createFiles()
    return readers.FixedLengthRecordDataset(
        filenames, self._record_bytes, self._header_bytes,
        self._footer_bytes).repeat(num_epochs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    # [FUNCTION DOC]
    # Checkpoint scenario: calls the verification callback with a factory for the dataset
    # repeated 5 epochs and the expected total outputs (epochs times files times records).
    # Returns nothing.
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    verify_fn(self, lambda: self._build_dataset(num_epochs), num_outputs)


if __name__ == "__main__":
  test.main()
