# [MODULE DOC]
# Unit tests for the text line dataset reader, which reads files as one string element per line.
# Provides a shared base class that writes sample text files (optionally with Windows line endings
# and gzip or zlib compression), then tests basic reading, parallel reads, buffering, resource
# cleanup, path objects, names, empty and missing files, argument type validation, and checkpoints.
# The optional process-utility library is imported with a fallback flag at import time.
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
"""Tests for `tf.data.TextLineDataset`."""
import gzip
import os
import pathlib
import zlib

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.util import compat


try:
  import psutil  # pylint: disable=g-import-not-at-top
  psutil_import_succeeded = True
except ImportError:
  psutil_import_succeeded = False


class TextLineDatasetTestBase(test_base.DatasetTestBase):
  # [CLASS DOC]
  # HELPER: shared base class for text line dataset tests. Provides helpers to build the expected
  # line contents and to write sample files into the test temp directory. Holds no state.
  """Base class for setting up and testing TextLineDataset."""

  def _lineText(self, f, l):
    # [FUNCTION DOC]
    # HELPER: returns the expected bytes content of line l (integer) in file number f (integer),
    # formatted as the file number, a colon and the line number.
    return compat.as_bytes("%d: %d" % (f, l))

  def _createFiles(self,
                   num_files,
                   num_lines,
                   crlf=False,
                   compression_type=None):
    # [FUNCTION DOC]
    # HELPER: writes sample text files into the test temp directory (side effect: creates or
    # overwrites files named by index). Inputs: num_files and num_lines (integers), crlf (use
    # carriage return plus newline as line ending; default false), compression_type (nothing,
    # "GZIP" or "ZLIB"). Each line holds the file and line numbers. Returns the list of file paths.
    # Raises a value error for an unsupported compression type.
    # KNOWN ISSUE: the comment says a newline follows every record except at the end of a file, but
    # the code omits the final newline only for files other than the first (file zero always ends
    # with a newline), so the last-line terminator differs between files.
    filenames = []
    for i in range(num_files):
      fn = os.path.join(self.get_temp_dir(), "text_line.%d.txt" % i)
      filenames.append(fn)
      contents = []
      for j in range(num_lines):
        contents.append(self._lineText(i, j))
        # Always include a newline after the record unless it is
        # at the end of the file, in which case we include it
        if j + 1 != num_lines or i == 0:
          contents.append(b"\r\n" if crlf else b"\n")
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


class TextLineDatasetTest(TextLineDatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Functional tests for the text line dataset: reading, compression, parallelism, buffering,
  # resource cleanup, input validation. Uses files created by the base class helpers.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression_type=[None, "GZIP", "ZLIB"])))
  def testBasic(self, compression_type):
    # [FUNCTION DOC]
    # Parameterized over no compression, gzip and zlib. Creates two files of five lines with
    # CRLF endings and asserts the dataset yields: file 0 alone, file 1 alone, both files in
    # order, both files repeated ten times, and both files repeated ten times in batches of five
    # (one batch per file per epoch).
    test_filenames = self._createFiles(
        2, 5, crlf=True, compression_type=compression_type)

    def dataset_fn(filenames, num_epochs, batch_size=None):
      # [FUNCTION DOC]
      # HELPER: builds the dataset under test: a text line dataset over filenames using the
      # enclosing compression type, repeated num_epochs times, and batched by batch_size if
      # given (optional). Returns the dataset.
      repeat_dataset = readers.TextLineDataset(
          filenames, compression_type=compression_type).repeat(num_epochs)
      if batch_size:
        return repeat_dataset.batch(batch_size)
      return repeat_dataset

    # Basic test: read from file 0.
    expected_output = [self._lineText(0, i) for i in range(5)]
    self.assertDatasetProduces(
        dataset_fn([test_filenames[0]], 1), expected_output=expected_output)

    # Basic test: read from file 1.
    self.assertDatasetProduces(
        dataset_fn([test_filenames[1]], 1),
        expected_output=[self._lineText(1, i) for i in range(5)])

    # Basic test: read from both files.
    expected_output = [self._lineText(0, i) for i in range(5)]
    expected_output.extend(self._lineText(1, i) for i in range(5))
    self.assertDatasetProduces(
        dataset_fn(test_filenames, 1), expected_output=expected_output)

    # Test repeated iteration through both files.
    expected_output = [self._lineText(0, i) for i in range(5)]
    expected_output.extend(self._lineText(1, i) for i in range(5))
    self.assertDatasetProduces(
        dataset_fn(test_filenames, 10), expected_output=expected_output * 10)

    # Test batched and repeated iteration through both files.
    self.assertDatasetProduces(
        dataset_fn(test_filenames, 10, 5),
        expected_output=[[self._lineText(0, i) for i in range(5)],
                         [self._lineText(1, i) for i in range(5)]] * 10)

  @combinations.generate(test_base.default_test_combinations())
  def testParallelRead(self):
    # [FUNCTION DOC]
    # Creates ten files of ten lines, repeats the file name list ten times, and reads with four
    # parallel reads. Asserts the lines of all files appear ten times each, ignoring order.
    test_filenames = self._createFiles(10, 10)
    files = dataset_ops.Dataset.from_tensor_slices(test_filenames).repeat(10)
    expected_output = []
    for j in range(10):
      expected_output.extend(self._lineText(j, i) for i in range(10))
    dataset = readers.TextLineDataset(files, num_parallel_reads=4)
    self.assertDatasetProduces(
        dataset, expected_output=expected_output * 10, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testBuffering(self):
    # [FUNCTION DOC]
    # Reads two CRLF files with a small explicit buffer size of ten bytes and asserts all lines
    # are still produced in order.
    test_filenames = self._createFiles(2, 5, crlf=True)

    repeat_dataset = readers.TextLineDataset(test_filenames, buffer_size=10)
    expected_output = []
    for j in range(2):
      expected_output.extend([self._lineText(j, i) for i in range(5)])
    self.assertDatasetProduces(repeat_dataset, expected_output=expected_output)

  @combinations.generate(test_base.eager_only_combinations())
  def testIteratorResourceCleanup(self):
    # [FUNCTION DOC]
    # Eager-only. Writes a three-line file, opens two iterators on equal datasets and a third on
    # a differently shaped (repeat and batch of sixteen) dataset, reads one element from each
    # (asserting the first elements and the batch shape of sixteen), then deletes the iterators.
    # If the process-utility library is unavailable the test is skipped; otherwise asserts the
    # file is no longer among the process's open files, showing resources were released.
    filename = os.path.join(self.get_temp_dir(), "text.txt")
    with open(filename, "wt") as f:
      for i in range(3):
        f.write("%d\n" % (i,))
    first_iterator = iter(readers.TextLineDataset(filename))
    self.assertEqual(b"0", next(first_iterator).numpy())
    second_iterator = iter(readers.TextLineDataset(filename))
    self.assertEqual(b"0", next(second_iterator).numpy())
    # Eager kernel caching is based on op attributes, which includes the
    # Dataset's output shape. Create a different kernel to test that they
    # don't create resources with the same names.
    different_kernel_iterator = iter(
        readers.TextLineDataset(filename).repeat().batch(16))
    self.assertEqual([16], next(different_kernel_iterator).shape)
    # Remove our references to the Python Iterator objects, which (assuming no
    # reference cycles) is enough to trigger DestroyResourceOp and close the
    # partially-read files.
    del first_iterator
    del second_iterator
    del different_kernel_iterator
    if not psutil_import_succeeded:
      self.skipTest(
          "psutil is required to check that we've closed our files.")
    open_files = psutil.Process().open_files()
    self.assertNotIn(filename, [open_file.path for open_file in open_files])

  @combinations.generate(test_base.default_test_combinations())
  def testPathlib(self):
    # [FUNCTION DOC]
    # Passes path objects (not strings) as file names and asserts the lines are read correctly
    # (compared ignoring order).
    files = self._createFiles(1, 5)
    files = [pathlib.Path(f) for f in files]

    expected_output = [self._lineText(0, i) for i in range(5)]
    ds = readers.TextLineDataset(files)
    self.assertDatasetProduces(
        ds, expected_output=expected_output, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Checks that the dataset constructor accepts a name argument and still yields all five lines
    # of one file (compared ignoring order).
    files = self._createFiles(1, 5)
    expected_output = [self._lineText(0, i) for i in range(5)]
    ds = readers.TextLineDataset(files, name="text_line_dataset")
    self.assertDatasetProduces(
        ds, expected_output=expected_output, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyFileList(self):
    # [FUNCTION DOC]
    # An empty list of file names produces an empty dataset; asserts no elements.
    dataset = readers.TextLineDataset(filenames=[])
    self.assertDatasetProduces(dataset, [])

  @combinations.generate(test_base.default_test_combinations())
  def testFileDoesNotExist(self):
    # [FUNCTION DOC]
    # Reads from a nonexistent file name and asserts iteration raises a not-found error whose
    # message mentions no such file or directory.
    dataset = readers.TextLineDataset(filenames=["File not exist"])
    with self.assertRaisesRegex(errors.NotFoundError,
                                "No such file or directory"):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testFileNamesMustBeStrings(self):
    # [FUNCTION DOC]
    # Passing an integer as file names must raise a type error saying the argument must contain
    # string elements and got 32-bit integer elements.
    with self.assertRaisesRegex(
        TypeError,
        "The `filenames` argument must contain `tf.string` elements. Got "
        "`tf.int32` elements."):
      readers.TextLineDataset(filenames=0)

  @combinations.generate(test_base.default_test_combinations())
  def testFileNamesDatasetMustContainStrings(self):
    # [FUNCTION DOC]
    # Passing a dataset of integers as file names must raise a type error saying the dataset must
    # contain string elements.
    with self.assertRaisesRegex(
        TypeError,
        "The `filenames` argument must contain `tf.string` elements. Got a "
        "dataset of `tf.int32` elements."):
      filenames = dataset_ops.Dataset.from_tensors(0)
      readers.TextLineDataset(filenames)

  @combinations.generate(test_base.default_test_combinations())
  def testFileNamesMustBeScalars(self):
    # [FUNCTION DOC]
    # Passing a dataset whose single element is a two-by-two matrix of strings must raise a type
    # error saying file names must be scalars.
    with self.assertRaisesRegex(
        TypeError,
        "The `filenames` argument must contain `tf.string` elements of shape "
        r"\[\] \(i.e. scalars\)."):
      filenames = dataset_ops.Dataset.from_tensors([["File 1", "File 2"],
                                                    ["File 3", "File 4"]])
      readers.TextLineDataset(filenames)


class TextLineDatasetCheckpointTest(TextLineDatasetTestBase,
                                    checkpoint_test_base.CheckpointTestBase,
                                    parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for the text line dataset, built on the file-writing base class
  # and the shared checkpoint verification fixture.

  def _build_iterator_graph(
      self, test_filenames, symbolic_checkpoint, compression_type=None
  ):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for checkpoint verification. Reads the given file
    # names with the optional compression type and a ten-byte buffer, and attaches options with
    # the symbolic checkpoint flag set. Returns the dataset.
    dataset = readers.TextLineDataset(
        test_filenames, compression_type=compression_type, buffer_size=10
    )
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              symbolic_checkpoint=[False, True],
              compression_type=[None, "GZIP", "ZLIB"],
          ),
      )
  )
  def test(self, verify_fn, symbolic_checkpoint, compression_type):
    # [FUNCTION DOC]
    # Parameterized checkpoint test over symbolic checkpoint on/off and compression type. Writes
    # five CRLF files of five lines, then asks verify_fn (checkpoint verification callback) to
    # check that save and restore of the dataset yields all twenty-five lines.
    num_files = 5
    lines_per_file = 5
    num_outputs = num_files * lines_per_file
    test_filenames = self._createFiles(
        num_files, lines_per_file, crlf=True, compression_type=compression_type)
    verify_fn(
        self,
        lambda: self._build_iterator_graph(
            test_filenames, symbolic_checkpoint, compression_type
        ),
        num_outputs,
    )


if __name__ == "__main__":
  test.main()
