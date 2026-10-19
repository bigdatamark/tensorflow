# [MODULE DOC]
# Unit tests for the TFRecord dataset reader, which reads serialized records from record files.
# Uses the shared record-file fixture from the tf_record_test_base module. Covers constructor type
# validation, reading one or many epochs, batching, zlib and gzip compressed files, explicit
# buffer sizes, datasets of file names, parallel reads, path objects, names, and checkpoints
# (symbolic or not, buffer sizes, compression types). Not a helper module; no import-time effects.
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
"""Tests for `tf.data.TFRecordDataset`."""
import gzip
import os
import pathlib
import zlib

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class TFRecordDatasetTest(tf_record_test_base.TFRecordTestBase,
                          parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests for the TFRecord dataset. Relies on the record-file fixture, which provides
  # the file name list, the number of files and records per file, and the expected record content.

  def _dataset_factory(self,
                       filenames,
                       compression_type="",
                       num_epochs=1,
                       batch_size=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test. Reads filenames (a name, list or dataset of names)
    # with compression_type (default empty, meaning none), repeats it num_epochs times (default
    # one), and batches by batch_size if given (optional). Returns the dataset.

    repeat_dataset = readers.TFRecordDataset(
        filenames, compression_type).repeat(num_epochs)
    if batch_size:
      return repeat_dataset.batch(batch_size)
    return repeat_dataset

  @combinations.generate(test_base.default_test_combinations())
  def testConstructorErrorsTensorInput(self):
    # [FUNCTION DOC]
    # Asserts the constructor raises a type error (string elements required, got 32-bit integer
    # elements) for a list of integers and for an integer tensor, and raises some error for an
    # arbitrary object (the exact type differs between graph and eager mode).
    with self.assertRaisesRegex(
        TypeError,
        "The `filenames` argument must contain `tf.string` elements. Got "
        "`tf.int32` elements."):
      readers.TFRecordDataset([1, 2, 3])
    with self.assertRaisesRegex(
        TypeError,
        "The `filenames` argument must contain `tf.string` elements. Got "
        "`tf.int32` elements."):
      readers.TFRecordDataset(constant_op.constant([1, 2, 3]))
    # convert_to_tensor raises different errors in graph and eager
    with self.assertRaises(Exception):
      readers.TFRecordDataset(object())

  @combinations.generate(test_base.default_test_combinations())
  def testReadOneEpoch(self):
    # [FUNCTION DOC]
    # Reads one epoch from the first file alone, the second file alone, and then all files, and
    # asserts the records come out in file order and record order.
    # Basic test: read from file 0.
    dataset = self._dataset_factory(self._filenames[0])
    self.assertDatasetProduces(
        dataset,
        expected_output=[self._record(0, i) for i in range(self._num_records)])

    # Basic test: read from file 1.
    dataset = self._dataset_factory(self._filenames[1])
    self.assertDatasetProduces(
        dataset,
        expected_output=[self._record(1, i) for i in range(self._num_records)])

    # Basic test: read from both files.
    dataset = self._dataset_factory(self._filenames)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testReadTenEpochs(self):
    # [FUNCTION DOC]
    # Reads all files for ten epochs and asserts the full record sequence repeats ten times.
    dataset = self._dataset_factory(self._filenames, num_epochs=10)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output * 10)

  @combinations.generate(test_base.default_test_combinations())
  def testReadTenEpochsOfBatches(self):
    # [FUNCTION DOC]
    # Reads all files for ten epochs with batch size equal to the records per file, and asserts
    # one batch per file per epoch, repeated ten times.
    dataset = self._dataset_factory(
        self._filenames, num_epochs=10, batch_size=self._num_records)
    expected_output = []
    for j in range(self._num_files):
      expected_output.append(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output * 10)

  @combinations.generate(test_base.default_test_combinations())
  def testReadZlibFiles(self):
    # [FUNCTION DOC]
    # Compresses each fixture file with zlib into the temp directory (side effect: writes new
    # files), reads them with the zlib compression type and asserts all records are produced in
    # order.
    zlib_files = []
    for i, fn in enumerate(self._filenames):
      with open(fn, "rb") as f:
        cdata = zlib.compress(f.read())

        zfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.z" % i)
        with open(zfn, "wb") as f:
          f.write(cdata)
        zlib_files.append(zfn)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    dataset = self._dataset_factory(zlib_files, compression_type="ZLIB")
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testReadGzipFiles(self):
    # [FUNCTION DOC]
    # Compresses each fixture file with gzip into the temp directory (side effect: writes new
    # files), reads them with the gzip compression type and asserts all records are produced in
    # order.
    gzip_files = []
    for i, fn in enumerate(self._filenames):
      with open(fn, "rb") as f:
        gzfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.gz" % i)
        with gzip.GzipFile(gzfn, "wb") as gzf:
          gzf.write(f.read())
        gzip_files.append(gzfn)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    dataset = self._dataset_factory(gzip_files, compression_type="GZIP")
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testReadWithBuffer(self):
    # [FUNCTION DOC]
    # Reads all files with an explicit one mebibyte buffer and asserts all records are produced in
    # order.
    one_mebibyte = 2**20
    dataset = readers.TFRecordDataset(
        self._filenames, buffer_size=one_mebibyte)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testReadFromDatasetOfFiles(self):
    # [FUNCTION DOC]
    # Supplies the file names as a dataset of strings instead of a list and asserts all records are
    # produced in order.
    files = dataset_ops.Dataset.from_tensor_slices(self._filenames)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    dataset = readers.TFRecordDataset(files)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testReadTenEpochsFromDatasetOfFilesInParallel(self):
    # [FUNCTION DOC]
    # Repeats a dataset of file names ten times and reads with four parallel reads. Asserts the
    # records of all files appear ten times, ignoring order.
    files = dataset_ops.Dataset.from_tensor_slices(
        self._filenames).repeat(10)
    expected_output = []
    for j in range(self._num_files):
      expected_output.extend(
          [self._record(j, i) for i in range(self._num_records)])
    dataset = readers.TFRecordDataset(files, num_parallel_reads=4)
    self.assertDatasetProduces(
        dataset, expected_output=expected_output * 10, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testPathlib(self):
    # [FUNCTION DOC]
    # Supplies a path object instead of a string for the first file and asserts its records are
    # produced (ignoring order).
    files = [pathlib.Path(self._filenames[0])]

    expected_output = [self._record(0, i) for i in range(self._num_records)]
    ds = readers.TFRecordDataset(files)
    self.assertDatasetProduces(
        ds, expected_output=expected_output, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Checks that the constructor accepts a name argument and still yields the first file's records
    # (ignoring order).
    files = [self._filenames[0]]

    expected_output = [self._record(0, i) for i in range(self._num_records)]
    ds = readers.TFRecordDataset(files, name="tf_record_dataset")
    self.assertDatasetProduces(
        ds, expected_output=expected_output, assert_items_equal=True)


class TFRecordDatasetCheckpointTest(tf_record_test_base.TFRecordTestBase,
                                    checkpoint_test_base.CheckpointTestBase,
                                    parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for the TFRecord dataset, using the record-file fixture and the
  # shared checkpoint verification fixture.

  def make_dataset(self,
                   num_epochs,
                   compression_type=None,
                   buffer_size=None,
                   symbolic_checkpoint=False):
    # [FUNCTION DOC]
    # HELPER: dataset factory for checkpoint tests. Creates the fixture record files, optionally
    # re-writes them compressed with zlib or gzip (side effect: new files in the temp directory),
    # and returns a TFRecord dataset over them with the given compression type (default none),
    # optional buffer size, repeated num_epochs times. If symbolic_checkpoint is true (default
    # false) the symbolic checkpoint option is attached.
    # KNOWN ISSUE: the gzip branch compresses the fixture's stored file names rather than the
    # list returned by the file creation call used in the zlib branch; harmless only if both are
    # the same files.
    filenames = self._createFiles()
    if compression_type == "ZLIB":
      zlib_files = []
      for i, fn in enumerate(filenames):
        with open(fn, "rb") as f:
          cdata = zlib.compress(f.read())
          zfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.z" % i)
          with open(zfn, "wb") as f:
            f.write(cdata)
          zlib_files.append(zfn)
      filenames = zlib_files

    elif compression_type == "GZIP":
      gzip_files = []
      for i, fn in enumerate(self._filenames):
        with open(fn, "rb") as f:
          gzfn = os.path.join(self.get_temp_dir(), "tfrecord_%s.gz" % i)
          with gzip.GzipFile(gzfn, "wb") as gzf:
            gzf.write(f.read())
          gzip_files.append(gzfn)
      filenames = gzip_files

    dataset = readers.TFRecordDataset(
        filenames, compression_type,
        buffer_size=buffer_size).repeat(num_epochs)
    if symbolic_checkpoint:
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[True, False])))
  def test(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Parameterized checkpoint test over symbolic checkpoint on/off. Asks verify_fn (checkpoint
    # verification callback) to check save/restore of a five-epoch dataset that yields every
    # record of every file five times.
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    verify_fn(
        self,
        lambda: self.make_dataset(
            num_epochs, symbolic_checkpoint=symbolic_checkpoint
        ),
        num_outputs,
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(buffer_size=[0, 5])))
  def testBufferSize(self, verify_fn, buffer_size):
    # [FUNCTION DOC]
    # Parameterized checkpoint test over buffer sizes zero and five. Asks verify_fn to check
    # save/restore of the five-epoch dataset yields all records five times.
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    verify_fn(self,
              lambda: self.make_dataset(num_epochs, buffer_size=buffer_size),
              num_outputs)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(compression_type=[None, "GZIP", "ZLIB"])))
  def testCompressionTypes(self, verify_fn, compression_type):
    # [FUNCTION DOC]
    # Parameterized checkpoint test over no compression, gzip and zlib. Asks verify_fn to check
    # save/restore of the five-epoch dataset yields all records five times.
    num_epochs = 5
    num_outputs = num_epochs * self._num_files * self._num_records
    # pylint: disable=g-long-lambda
    verify_fn(
        self, lambda: self.make_dataset(
            num_epochs, compression_type=compression_type), num_outputs)


if __name__ == "__main__":
  test.main()
