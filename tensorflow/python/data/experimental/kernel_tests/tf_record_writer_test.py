# [MODULE DOC]
# Tests for the experimental TFRecord writer, which writes a dataset of scalar strings to a
# record file. Covers plain, ZLIB and GZIP compression, type and shape validation errors, use
# inside a traced function (side effect retained), and sharded writing via grouping.
# No import-time side effects besides imports.
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
"""Tests for `tf.data.experimental.TFRecordWriter`."""
import os

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import writers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import python_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class TFRecordWriterTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the dataset-based TFRecord writer. Uses fixed input and output files in the temp
  # directory and eight generated records.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the base setup and sets the number of records to 8.
    super(TFRecordWriterTest, self).setUp()
    self._num_records = 8

  def writer_fn(self, filename, compression_type=""):
    # [FUNCTION DOC]
    # HELPER: builds the write operation under test. Reads the given record file (optional
    # compression type, default none) as a dataset and returns the op/result of writing it to the
    # output file with the same compression type.
    input_dataset = readers.TFRecordDataset([filename], compression_type)
    return writers.TFRecordWriter(self._outputFilename(),
                                  compression_type).write(input_dataset)

  def _record(self, i):
    # [FUNCTION DOC]
    # HELPER: returns the expected bytes for record number i (text "Record" plus the number).
    return compat.as_bytes("Record %d" % (i))

  def _createFile(self, options=None):
    # [FUNCTION DOC]
    # HELPER: writes the eight expected records to the input file using the low-level record writer
    # with optional record options (e.g. compression) and returns the input file name.
    filename = self._inputFilename()
    writer = python_io.TFRecordWriter(filename, options)
    for i in range(self._num_records):
      writer.write(self._record(i))
    writer.close()
    return filename

  def _inputFilename(self):
    # [FUNCTION DOC]
    # HELPER: returns the path of the input record file in the test temp directory.
    return os.path.join(self.get_temp_dir(), "tf_record.in.txt")

  def _outputFilename(self):
    # [FUNCTION DOC]
    # HELPER: returns the path of the output record file in the test temp directory.
    return os.path.join(self.get_temp_dir(), "tf_record.out.txt")

  @combinations.generate(test_base.default_test_combinations())
  def testWrite(self):
    # [FUNCTION DOC]
    # Writes an uncompressed input file through the writer and asserts the output file contains
    # the same records in order.
    self.evaluate(self.writer_fn(self._createFile()))
    for i, r in enumerate(tf_record.tf_record_iterator(self._outputFilename())):
      self.assertAllEqual(self._record(i), r)

  @combinations.generate(test_base.default_test_combinations())
  def testWriteZLIB(self):
    # [FUNCTION DOC]
    # Same as the plain write test with ZLIB compression for both input and output files.
    options = tf_record.TFRecordOptions(tf_record.TFRecordCompressionType.ZLIB)
    self.evaluate(
        self.writer_fn(self._createFile(options), compression_type="ZLIB"))
    for i, r in enumerate(
        tf_record.tf_record_iterator(self._outputFilename(), options=options)):
      self.assertAllEqual(self._record(i), r)

  @combinations.generate(test_base.default_test_combinations())
  def testWriteGZIP(self):
    # [FUNCTION DOC]
    # Same as the plain write test with GZIP compression for both input and output files.
    options = tf_record.TFRecordOptions(tf_record.TFRecordCompressionType.GZIP)
    self.evaluate(
        self.writer_fn(self._createFile(options), compression_type="GZIP"))
    for i, r in enumerate(
        tf_record.tf_record_iterator(self._outputFilename(), options=options)):
      self.assertAllEqual(self._record(i), r)

  @combinations.generate(test_base.default_test_combinations())
  def testFailDataset(self):
    # [FUNCTION DOC]
    # Passing a plain string instead of a dataset to the writer must raise a type error.
    with self.assertRaises(TypeError):
      writers.TFRecordWriter(self._outputFilename(), "").write("whoops")

  @combinations.generate(test_base.default_test_combinations())
  def testFailDType(self):
    # [FUNCTION DOC]
    # Writing a dataset of integers (not strings) must raise a type error.
    input_dataset = dataset_ops.Dataset.from_tensors(10)
    with self.assertRaises(TypeError):
      writers.TFRecordWriter(self._outputFilename(), "").write(input_dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testFailShape(self):
    # [FUNCTION DOC]
    # Writing a dataset whose elements are non-scalar string matrices must raise a type error.
    input_dataset = dataset_ops.Dataset.from_tensors([["hello"], ["world"]])
    with self.assertRaises(TypeError):
      writers.TFRecordWriter(self._outputFilename(), "").write(input_dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testSideEffect(self):
    # [FUNCTION DOC]
    # Defines a traced function that builds the write op but discards its result and returns a
    # constant string; asserts the constant is returned and that the output file still holds all
    # records, proving the write is kept as a side effect.
    def writer_fn():
      # [FUNCTION DOC]
      # HELPER: creates the input file, reads it as a dataset and returns the write op for the
      # output file (no compression).
      input_dataset = readers.TFRecordDataset(self._createFile())
      return writers.TFRecordWriter(self._outputFilename()).write(input_dataset)

    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: traced function that runs the writer, ignores its result and returns the bytes
      # of the string hello.
      _ = writer_fn()
      return "hello"

    self.assertEqual(self.evaluate(fn()), b"hello")
    for i, r in enumerate(tf_record.tf_record_iterator(self._outputFilename())):
      self.assertAllEqual(self._record(i), r)

  @combinations.generate(test_base.default_test_combinations())
  def testShard(self):
    # [FUNCTION DOC]
    # Reads the input records, enumerates them, groups by parity of the index into windows, and in
    # the window reduce callback writes each group to its own shard file named from the input name
    # and key. Asserts the dataset yields the two shard file names and that each shard holds the
    # records of matching parity in order.
    filename = self._createFile()
    dataset = readers.TFRecordDataset([filename])

    def reduce_func(key, dataset):
      # [FUNCTION DOC]
      # Window reduce callback: takes the group key and the window dataset of (index, record)
      # pairs, writes the records to a shard file named by joining the input name and key, and
      # returns a one-element dataset holding the shard file name.
      shard_filename = string_ops.string_join(
          [filename, string_ops.as_string(key)])
      writer = writers.TFRecordWriter(shard_filename)
      writer.write(dataset.map(lambda _, x: x))
      return dataset_ops.Dataset.from_tensors(shard_filename)

    dataset = dataset.enumerate()
    dataset = dataset.apply(
        grouping.group_by_window(lambda i, _: i % 2, reduce_func,
                                 dtypes.int64.max))

    get_next = self.getNext(dataset)
    for i in range(2):
      shard_filename = (filename + str(i)).encode()
      self.assertEqual(self.evaluate(get_next()), shard_filename)
      for j, r in enumerate(tf_record.tf_record_iterator(shard_filename)):
        self.assertAllEqual(self._record(i + 2*j), r)


if __name__ == "__main__":
  test.main()
