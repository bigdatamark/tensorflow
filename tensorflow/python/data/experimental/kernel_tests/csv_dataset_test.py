# [MODULE DOC]
# Kernel tests for the experimental CSV dataset reader. One test class compares the CSV dataset
# against a text-line dataset mapped through the CSV decoding op, or against expected values or
# error messages, covering types, empty fields, quoting, delimiters, NA values, column selection
# and exclusion, headers, line break styles, buffer sizes, compression and bad record defaults.
# A second class tests iterator checkpointing. Writes temporary CSV files in the test directory.
#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.CsvDataset`."""

import gzip
import os
import zlib

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class CsvDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the CSV dataset reader. Provides helpers that write temporary CSV files and
  # compare the reader's output against a reference (text lines mapped through CSV decoding),
  # against expected tuples, or against expected error messages.

  def _setup_files(self, inputs, linebreak='\n', compression_type=None):
    # [FUNCTION DOC]
    # HELPER: writes CSV fixture files for the tests.
    # inputs is a list (one per file) of lists of row strings; linebreak is the separator
    # joining rows (default newline); compression_type is nothing (plain), GZIP or ZLIB.
    # Each file is written to the test temp directory as a numbered CSV file, UTF-8 encoded.
    # Returns the list of file names. Raises a value error for any other compression type.
    filenames = []
    for i, file_rows in enumerate(inputs):
      fn = os.path.join(self.get_temp_dir(), 'temp_%d.csv' % i)
      contents = linebreak.join(file_rows).encode('utf-8')
      if compression_type is None:
        with open(fn, 'wb') as f:
          f.write(contents)
      elif compression_type == 'GZIP':
        with gzip.GzipFile(fn, 'wb') as f:
          f.write(contents)
      elif compression_type == 'ZLIB':
        contents = zlib.compress(contents)
        with open(fn, 'wb') as f:
          f.write(contents)
      else:
        raise ValueError('Unsupported compression_type', compression_type)
      filenames.append(fn)
    return filenames

  def _make_test_datasets(self, inputs, **kwargs):
    # [FUNCTION DOC]
    # HELPER: builds the pair of datasets compared by the tests.
    # Writes the input files, then returns the CSV dataset (actual) and a text-line dataset
    # mapped through the CSV decode op (expected); keyword arguments are forwarded to both.
    # Test by comparing its output to what we could get with map->decode_csv
    filenames = self._setup_files(inputs)
    dataset_expected = core_readers.TextLineDataset(filenames)
    dataset_expected = dataset_expected.map(
        lambda l: parsing_ops.decode_csv(l, **kwargs))
    dataset_actual = readers.CsvDataset(filenames, **kwargs)
    return (dataset_actual, dataset_expected)

  def _test_by_comparison(self, inputs, **kwargs):
    # [FUNCTION DOC]
    # HELPER: asserts the CSV dataset yields the same elements as the reference text-line plus
    # decode pipeline for the given input rows and keyword options. Returns nothing.
    """Checks that CsvDataset is equiv to TextLineDataset->map(decode_csv)."""
    dataset_actual, dataset_expected = self._make_test_datasets(
        inputs, **kwargs)
    self.assertDatasetsEqual(dataset_actual, dataset_expected)

  def _test_dataset(
      self,
      inputs,
      expected_output=None,
      expected_err_re=None,
      linebreak='\n',
      compression_type=None,  # Used for both setup and parsing
      **kwargs):
    # [FUNCTION DOC]
    # HELPER: core assertion routine for the CSV tests.
    # Inputs: file contents, optional expected output (list of rows), optional expected error
    # regular expression, linebreak and compression type (used both to write and to read), and
    # extra options for the CSV dataset. If an error pattern is given, building and reading the
    # dataset must raise a matching op error. Otherwise string expected values are encoded to
    # bytes and the dataset must produce exactly the expected tuples. Returns nothing.
    """Checks that elements produced by CsvDataset match expected output."""
    # Convert str type because py3 tf strings are bytestrings
    filenames = self._setup_files(inputs, linebreak, compression_type)
    kwargs['compression_type'] = compression_type
    if expected_err_re is not None:
      # Verify that OpError is produced as expected
      with self.assertRaisesOpError(expected_err_re):
        dataset = readers.CsvDataset(filenames, **kwargs)
        self.getDatasetOutput(dataset)
    else:
      dataset = readers.CsvDataset(filenames, **kwargs)
      expected_output = [
          tuple(v.encode('utf-8') if isinstance(v, str) else v
                for v in op)
          for op in expected_output
      ]
      self.assertDatasetProduces(dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testRequiredFields(self):
    # [FUNCTION DOC]
    # Test: four required (no default) fields; CSV dataset matches the reference pipeline.
    record_defaults = [[]] * 4
    inputs = [['1,2,3,4']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testInt(self):
    # [FUNCTION DOC]
    # Test: integer columns with integer defaults match the reference pipeline.
    record_defaults = [[0]] * 4
    inputs = [['1,2,3,4', '5,6,7,8']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testFloat(self):
    # [FUNCTION DOC]
    # Test: float columns with float defaults match the reference pipeline.
    record_defaults = [[0.0]] * 4
    inputs = [['1.0,2.1,3.2,4.3', '5.4,6.5,7.6,8.7']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testString(self):
    # [FUNCTION DOC]
    # Test: string columns with empty-string defaults match the reference pipeline.
    record_defaults = [['']] * 4
    inputs = [['1.0,2.1,hello,4.3', '5.4,6.5,goodbye,8.7']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithEmptyFields(self):
    # [FUNCTION DOC]
    # Test: empty fields are replaced by the integer default zero in the produced rows.
    record_defaults = [[0]] * 4
    inputs = [[',,,', '1,1,1,', ',2,2,2']]
    self._test_dataset(
        inputs, [[0, 0, 0, 0], [1, 1, 1, 0], [0, 2, 2, 2]],
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testErrWithUnquotedQuotes(self):
    # [FUNCTION DOC]
    # Test: a quote inside an unquoted field raises an error saying unquoted fields cannot
    # contain quotes.
    record_defaults = [['']] * 3
    inputs = [['1,2"3,4']]
    self._test_dataset(
        inputs,
        expected_err_re='Unquoted fields cannot have quotes inside',
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testErrWithUnescapedQuotes(self):
    # [FUNCTION DOC]
    # Test: a quote inside a quoted field that is not escaped by another quote raises an error.
    record_defaults = [['']] * 3
    inputs = [['"a"b","c","d"']]
    self._test_dataset(
        inputs,
        expected_err_re=
        'Quote inside a string has to be escaped by another quote',
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testIgnoreErrWithUnescapedQuotes(self):
    # [FUNCTION DOC]
    # Test: with the ignore-errors transformation applied, rows with unescaped quotes are
    # dropped and only the valid row (e, f, g) is produced.
    record_defaults = [['']] * 3
    inputs = [['1,"2"3",4', '1,"2"3",4",5,5', 'a,b,"c"d"', 'e,f,g']]
    filenames = self._setup_files(inputs)
    dataset = readers.CsvDataset(filenames, record_defaults=record_defaults)
    dataset = dataset.apply(error_ops.ignore_errors())
    self.assertDatasetProduces(dataset, [(b'e', b'f', b'g')])

  @combinations.generate(test_base.default_test_combinations())
  def testIgnoreErrWithUnquotedQuotes(self):
    # [FUNCTION DOC]
    # Test: with ignore-errors applied, rows having quotes in unquoted fields are dropped and
    # only the valid row (e, f, g) is produced.
    record_defaults = [['']] * 3
    inputs = [['1,2"3,4', 'a,b,c"d', '9,8"7,6,5', 'e,f,g']]
    filenames = self._setup_files(inputs)
    dataset = readers.CsvDataset(filenames, record_defaults=record_defaults)
    dataset = dataset.apply(error_ops.ignore_errors())
    self.assertDatasetProduces(dataset, [(b'e', b'f', b'g')])

  @combinations.generate(test_base.default_test_combinations())
  def testWithNoQuoteDelimAndUnquotedQuotes(self):
    # [FUNCTION DOC]
    # Test: with quote delimiting disabled, a stray quote is treated as ordinary text and the
    # output matches the reference pipeline.
    record_defaults = [['']] * 3
    inputs = [['1,2"3,4']]
    self._test_by_comparison(
        inputs, record_defaults=record_defaults, use_quote_delim=False)

  @combinations.generate(test_base.default_test_combinations())
  def testMixedTypes(self):
    # [FUNCTION DOC]
    # Test: columns of int32, float32, string and float64 declared through empty typed constants
    # (required fields) match the reference pipeline.
    record_defaults = [
        constant_op.constant([], dtype=dtypes.int32),
        constant_op.constant([], dtype=dtypes.float32),
        constant_op.constant([], dtype=dtypes.string),
        constant_op.constant([], dtype=dtypes.float64)
    ]
    inputs = [['1,2.1,3.2,4.3', '5,6.5,7.6,8.7']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithUseQuoteDelimFalse(self):
    # [FUNCTION DOC]
    # Test: with quote delimiting turned off, quoted commas split fields; output matches the
    # reference pipeline.
    record_defaults = [['']] * 4
    inputs = [['1,2,"3,4"', '"5,6",7,8']]
    self._test_by_comparison(
        inputs, record_defaults=record_defaults, use_quote_delim=False)

  @combinations.generate(test_base.default_test_combinations())
  def testWithFieldDelim(self):
    # [FUNCTION DOC]
    # Test: a colon field delimiter is honored; output matches the reference pipeline.
    record_defaults = [[0]] * 4
    inputs = [['1:2:3:4', '5:6:7:8']]
    self._test_by_comparison(
        inputs, record_defaults=record_defaults, field_delim=':')

  @combinations.generate(test_base.default_test_combinations())
  def testWithNaValue(self):
    # [FUNCTION DOC]
    # Test: fields equal to the configured NA marker are replaced by defaults; output matches
    # the reference pipeline.
    record_defaults = [[0]] * 4
    inputs = [['1,NA,3,4', 'NA,6,7,8']]
    self._test_by_comparison(
        inputs, record_defaults=record_defaults, na_value='NA')

  @combinations.generate(test_base.default_test_combinations())
  def testWithSelectCols(self):
    # [FUNCTION DOC]
    # Test: selecting columns one and two (quoted and unquoted values) matches the reference
    # pipeline.
    record_defaults = [['']] * 2
    inputs = [['1,2,3,4', '"5","6","7","8"']]
    self._test_by_comparison(
        inputs, record_defaults=record_defaults, select_cols=[1, 2])

  @combinations.generate(test_base.default_test_combinations())
  def testWithSelectColsTooHigh(self):
    # [FUNCTION DOC]
    # Test: selecting column indices beyond the row width raises an error that two fields were
    # expected but only one was found.
    record_defaults = [[0]] * 2
    inputs = [['1,2,3,4', '5,6,7,8']]
    self._test_dataset(
        inputs,
        expected_err_re='Expect 2 fields but have 1 in record',
        record_defaults=record_defaults,
        select_cols=[3, 4])

  @combinations.generate(test_base.default_test_combinations())
  def testWithOneCol(self):
    # [FUNCTION DOC]
    # Test: a single-column file with an empty line yields the default NA for the empty row.
    record_defaults = [['NA']]
    inputs = [['0', '', '2']]
    self._test_dataset(
        inputs, [['0'], ['NA'], ['2']], record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithMultipleFiles(self):
    # [FUNCTION DOC]
    # Test: two input files are read in sequence; output matches the reference pipeline.
    record_defaults = [[0]] * 4
    inputs = [['1,2,3,4', '5,6,7,8'], ['5,6,7,8']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithLeadingAndTrailingSpaces(self):
    # [FUNCTION DOC]
    # Test: spaces around numeric float fields are tolerated and parsed as 0, 1, 2, 3.
    record_defaults = [[0.0]] * 4
    inputs = [['0, 1, 2, 3']]
    expected = [[0.0, 1.0, 2.0, 3.0]]
    self._test_dataset(inputs, expected, record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testErrorWithMissingDefault(self):
    # [FUNCTION DOC]
    # Test: an empty field with no default (required) raises an error that field 1 is required.
    record_defaults = [[]] * 2
    inputs = [['0,']]
    self._test_dataset(
        inputs,
        expected_err_re='Field 1 is required but missing in record!',
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testErrorWithFewerDefaultsThanFields(self):
    # [FUNCTION DOC]
    # Test: two defaults but four fields in the record raises an error expecting two fields.
    record_defaults = [[0.0]] * 2
    inputs = [['0,1,2,3']]
    self._test_dataset(
        inputs,
        expected_err_re='Expect 2 fields but have more in record',
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testErrorWithMoreDefaultsThanFields(self):
    # [FUNCTION DOC]
    # Test: five defaults but four fields in the record raises an error expecting five fields.
    record_defaults = [[0.0]] * 5
    inputs = [['0,1,2,3']]
    self._test_dataset(
        inputs,
        expected_err_re='Expect 5 fields but have 4 in record',
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithHeader(self):
    # [FUNCTION DOC]
    # Test: with the header option on, the first line is skipped and only the data row is
    # produced.
    record_defaults = [[0]] * 2
    inputs = [['col1,col2', '1,2']]
    expected = [[1, 2]]
    self._test_dataset(
        inputs,
        expected,
        record_defaults=record_defaults,
        header=True,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithHeaderAndNoRecords(self):
    # [FUNCTION DOC]
    # Test: a file containing only a header produces an empty dataset.
    record_defaults = [[0]] * 2
    inputs = [['col1,col2']]
    expected = []
    self._test_dataset(
        inputs,
        expected,
        record_defaults=record_defaults,
        header=True,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testErrorWithHeaderEmptyFile(self):
    # [FUNCTION DOC]
    # Test: with the header option on, an empty file raises an error that the header cannot be
    # read.
    record_defaults = [[0]] * 2
    inputs = [[]]
    expected_err_re = "Can't read header of file"
    self._test_dataset(
        inputs,
        expected_err_re=expected_err_re,
        record_defaults=record_defaults,
        header=True,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithEmptyFile(self):
    # [FUNCTION DOC]
    # Test: an empty file yields no elements.
    record_defaults = [['']] * 2
    inputs = [['']]  # Empty file
    self._test_dataset(
        inputs, expected_output=[], record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testErrorWithEmptyRecord(self):
    # [FUNCTION DOC]
    # Test: an empty first record where two fields are expected raises an error (one field
    # found).
    record_defaults = [['']] * 2
    inputs = [['', '1,2']]  # First record is empty
    self._test_dataset(
        inputs,
        expected_err_re='Expect 2 fields but have 1 in record',
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithChainedOps(self):
    # [FUNCTION DOC]
    # Test: repeating five times and prefetching one element on both the CSV dataset and the
    # reference dataset gives equal outputs, showing one dataset can create several iterators.
    # Testing that one dataset can create multiple iterators fine.
    # `repeat` creates multiple iterators from the same C++ Dataset.
    record_defaults = [[0]] * 4
    inputs = [['1,,3,4', '5,6,,8']]
    ds_actual, ds_expected = self._make_test_datasets(
        inputs, record_defaults=record_defaults)
    self.assertDatasetsEqual(
        ds_actual.repeat(5).prefetch(1),
        ds_expected.repeat(5).prefetch(1))

  @combinations.generate(test_base.default_test_combinations())
  def testWithTypeDefaults(self):
    # [FUNCTION DOC]
    # Test: a bare data type as a record default marks a required field of that type; produced
    # floats match the expected values.
    # Testing using dtypes as record_defaults for required fields
    record_defaults = [dtypes.float32, [0.0]]
    inputs = [['1.0,2.0', '3.0,4.0']]
    self._test_dataset(
        inputs,
        [[1.0, 2.0], [3.0, 4.0]],
        record_defaults=record_defaults,
    )

## The following tests exercise parsing logic for quoted fields

  @combinations.generate(test_base.default_test_combinations())
  def testWithQuoted(self):
    # [FUNCTION DOC]
    # Test: fully quoted fields (including smileys with punctuation) match the reference
    # pipeline.
    record_defaults = [['']] * 4
    inputs = [['"a","b","c :)","d"', '"e","f","g :(","h"']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithOneColAndQuotes(self):
    # [FUNCTION DOC]
    # Test: a single quoted column yields the unquoted strings.
    record_defaults = [['']]
    inputs = [['"0"', '"1"', '"2"']]
    self._test_dataset(
        inputs, [['0'], ['1'], ['2']], record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithNewLine(self):
    # [FUNCTION DOC]
    # Test: a quoted field containing a newline stays inside one record; compared with explicit
    # expected values because the reference pipeline handles this case incorrectly.
    # In this case, we expect it to behave differently from
    # TextLineDataset->map(decode_csv) since that flow has bugs
    record_defaults = [['']] * 4
    inputs = [['a,b,"""c""\n0","d\ne"', 'f,g,h,i']]
    expected = [['a', 'b', '"c"\n0', 'd\ne'], ['f', 'g', 'h', 'i']]
    self._test_dataset(inputs, expected, record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithNewLineInUnselectedCol(self):
    # [FUNCTION DOC]
    # Test: a newline inside a quoted field of an unselected column is skipped correctly while
    # selecting only column zero.
    record_defaults = [['']]
    inputs = [['1,"2\n3",4', '5,6,7']]
    self._test_dataset(
        inputs,
        expected_output=[['1'], ['5']],
        record_defaults=record_defaults,
        select_cols=[0])

  @combinations.generate(test_base.v2_only_combinations())
  def testWithExcludeCol(self):
    # [FUNCTION DOC]
    # Test (v2 only): excluding columns one and two leaves column zero.
    record_defaults = [['']]
    inputs = [['1,2,3', '5,6,7']]
    self._test_dataset(
        inputs,
        expected_output=[['1'], ['5']],
        record_defaults=record_defaults,
        exclude_cols=[1, 2])

  @combinations.generate(test_base.v2_only_combinations())
  def testWithSelectandExcludeCol(self):
    # [FUNCTION DOC]
    # Test (v2 only): giving both selected and excluded columns raises an error that one of them
    # must be empty.
    record_defaults = [['']]
    inputs = [['1,2,3', '5,6,7']]
    self._test_dataset(
        inputs,
        expected_err_re='Either select_cols or exclude_cols should be empty',
        record_defaults=record_defaults,
        select_cols=[0],
        exclude_cols=[1, 2])

  @combinations.generate(test_base.v2_only_combinations())
  def testWithExcludeColandRecordDefaultsTooLow(self):
    # [FUNCTION DOC]
    # Test (v2 only): excluding one of three columns but supplying only one default raises an
    # error that more fields than defaults remain.
    record_defaults = [['']]
    inputs = [['1,2,3', '5,6,7']]
    self._test_dataset(
        inputs,
        expected_err_re='Expect 1 fields but have more in record',
        record_defaults=record_defaults,
        exclude_cols=[0])

  @combinations.generate(test_base.v2_only_combinations())
  def testWithExcludeColandRecordDefaultsTooHigh(self):
    # [FUNCTION DOC]
    # Test (v2 only): excluding one of three columns but supplying three defaults raises an
    # error that three fields were expected but two found.
    record_defaults = [['']] * 3
    inputs = [['1,2,3', '5,6,7']]
    self._test_dataset(
        inputs,
        expected_err_re='Expect 3 fields but have 2 in record',
        record_defaults=record_defaults,
        exclude_cols=[0])

  @combinations.generate(test_base.default_test_combinations())
  def testWithMultipleNewLines(self):
    # [FUNCTION DOC]
    # Test: quoted fields containing several newlines remain in one record; compared with
    # explicit expected values.
    # In this case, we expect it to behave differently from
    # TextLineDataset->map(decode_csv) since that flow has bugs
    record_defaults = [['']] * 4
    inputs = [['a,"b\n\nx","""c""\n \n0","d\ne"', 'f,g,h,i']]
    expected = [['a', 'b\n\nx', '"c"\n \n0', 'd\ne'], ['f', 'g', 'h', 'i']]
    self._test_dataset(inputs, expected, record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testErrorWithTerminateMidRecord(self):
    # [FUNCTION DOC]
    # Test: a file ending inside an unclosed quoted field raises an end-of-file error.
    record_defaults = [['']] * 4
    inputs = [['a,b,c,"a']]
    self._test_dataset(
        inputs,
        expected_err_re=
        'Reached end of file without closing quoted field in record',
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithEscapedQuotes(self):
    # [FUNCTION DOC]
    # Test: doubled quotes inside quoted fields are unescaped; matches the reference pipeline.
    record_defaults = [['']] * 4
    inputs = [['1.0,2.1,"she said: ""hello""",4.3', '5.4,6.5,goodbye,8.7']]
    self._test_by_comparison(inputs, record_defaults=record_defaults)


## Testing that parsing works with all buffer sizes, quoted/unquoted fields,
## and different types of line breaks

  @combinations.generate(test_base.default_test_combinations())
  def testWithInvalidBufferSize(self):
    # [FUNCTION DOC]
    # Test: a buffer size of zero raises an error that the buffer size must be positive.
    record_defaults = [['']] * 4
    inputs = [['a,b,c,d']]
    self._test_dataset(
        inputs,
        expected_err_re='buffer_size should be positive',
        record_defaults=record_defaults,
        buffer_size=0)

  def _test_dataset_on_buffer_sizes(self,
                                    inputs,
                                    expected,
                                    linebreak,
                                    record_defaults,
                                    compression_type=None,
                                    num_sizes_to_test=20):
    # [FUNCTION DOC]
    # HELPER: repeats the expected-output check for many buffer sizes.
    # Inputs: file contents, expected rows, linebreak, record defaults, optional compression
    # type, and how many sizes to test (default 20). Runs the check for buffer sizes one up to
    # that count, plus the default (unspecified) size. Returns nothing.
    # Testing reading with a range of buffer sizes that should all work.
    for i in list(range(1, 1 + num_sizes_to_test)) + [None]:
      self._test_dataset(
          inputs,
          expected,
          linebreak=linebreak,
          compression_type=compression_type,
          record_defaults=record_defaults,
          buffer_size=i)

  @combinations.generate(test_base.default_test_combinations())
  def testWithLF(self):
    # [FUNCTION DOC]
    # Test: newline line breaks parse correctly at all tested buffer sizes, with empty fields
    # replaced by NA.
    record_defaults = [['NA']] * 3
    inputs = [['abc,def,ghi', '0,1,2', ',,']]
    expected = [['abc', 'def', 'ghi'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\n', record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithCR(self):
    # [FUNCTION DOC]
    # Test: carriage-return line breaks parse correctly at all tested buffer sizes.
    # Test that when the line separator is '\r', parsing works with all buffer
    # sizes
    record_defaults = [['NA']] * 3
    inputs = [['abc,def,ghi', '0,1,2', ',,']]
    expected = [['abc', 'def', 'ghi'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\r', record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithCRLF(self):
    # [FUNCTION DOC]
    # Test: carriage-return plus newline line breaks parse correctly at all tested buffer sizes.
    # Test that when the line separator is '\r\n', parsing works with all buffer
    # sizes
    record_defaults = [['NA']] * 3
    inputs = [['abc,def,ghi', '0,1,2', ',,']]
    expected = [['abc', 'def', 'ghi'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\r\n', record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithBufferSizeAndQuoted(self):
    # [FUNCTION DOC]
    # Test: quoted fields containing newline and carriage-return characters parse correctly
    # across all tested buffer sizes with newline line breaks.
    record_defaults = [['NA']] * 3
    inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
    expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'],
                ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\n', record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithCRAndQuoted(self):
    # [FUNCTION DOC]
    # Test: quoted fields with embedded line-break characters and carriage-return row separators
    # parse correctly at all tested buffer sizes.
    # Test that when the line separator is '\r', parsing works with all buffer
    # sizes
    record_defaults = [['NA']] * 3
    inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
    expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'],
                ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\r', record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithCRLFAndQuoted(self):
    # [FUNCTION DOC]
    # Test: quoted fields with embedded line-break characters and CR-LF row separators parse
    # correctly at all tested buffer sizes.
    # Test that when the line separator is '\r\n', parsing works with all buffer
    # sizes
    record_defaults = [['NA']] * 3
    inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
    expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'],
                ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs, expected, linebreak='\r\n', record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithGzipCompressionType(self):
    # [FUNCTION DOC]
    # Test: GZIP-compressed CSV files with CR-LF separators and quoted fields are read correctly
    # at all tested buffer sizes.
    record_defaults = [['NA']] * 3
    inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
    expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'],
                ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs,
        expected,
        linebreak='\r\n',
        compression_type='GZIP',
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithZlibCompressionType(self):
    # [FUNCTION DOC]
    # Test: ZLIB-compressed CSV files with CR-LF separators and quoted fields are read correctly
    # at all tested buffer sizes.
    record_defaults = [['NA']] * 3
    inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
    expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'],
                ['NA', 'NA', 'NA']]
    self._test_dataset_on_buffer_sizes(
        inputs,
        expected,
        linebreak='\r\n',
        compression_type='ZLIB',
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWithScalarDefaults(self):
    # [FUNCTION DOC]
    # Test: scalar int64 tensors as record defaults fill empty fields with zero.
    record_defaults = [constant_op.constant(0, dtype=dtypes.int64)] * 4
    inputs = [[',,,', '1,1,1,', ',2,2,2']]
    self._test_dataset(
        inputs, [[0, 0, 0, 0], [1, 1, 1, 0], [0, 2, 2, 2]],
        record_defaults=record_defaults)

  @combinations.generate(test_base.default_test_combinations())
  def testWith2DDefaults(self):
    # [FUNCTION DOC]
    # Test: rank-2 record defaults are rejected: an invalid-argument error in eager mode
    # (record default must be at most rank 1) or a value error in graph mode.
    record_defaults = [constant_op.constant([[0]], dtype=dtypes.int64)] * 4
    inputs = [[',,,', '1,1,1,', ',2,2,2']]

    if context.executing_eagerly():
      err_spec = errors.InvalidArgumentError, (
          'Each record default should be at '
          'most rank 1')
    else:
      err_spec = ValueError, 'Shape must be at most rank 1 but is rank 2'

    with self.assertRaisesWithPredicateMatch(*err_spec):
      self._test_dataset(
          inputs, [[0, 0, 0, 0], [1, 1, 1, 0], [0, 2, 2, 2]],
          record_defaults=record_defaults)

  def testImmutableParams(self):
    # [FUNCTION DOC]
    # Test: creating a batched CSV dataset with a list of selected column names must not modify
    # that list; asserts the list is unchanged afterwards.
    inputs = [['a,b,c', '1,2,3', '4,5,6']]
    filenames = self._setup_files(inputs)
    select_cols = ['a', 'c']
    _ = readers.make_csv_dataset(
        filenames, batch_size=1, select_columns=select_cols)
    self.assertAllEqual(select_cols, ['a', 'c'])


class CsvDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                               parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint test for the CSV dataset: writes a plain and a GZIP-compressed CSV file of 10
  # rows by 7 columns in setUp and verifies iterator save/restore over 14 repeated epochs.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the parent setup, then writes a 10-row, 7-column CSV file with consecutive
    # integers and a GZIP copy in the test temp directory; records column, row, epoch and total
    # output counts (140 outputs) on the instance.
    super(CsvDatasetCheckpointTest, self).setUp()
    self._num_cols = 7
    self._num_rows = 10
    self._num_epochs = 14
    self._num_outputs = self._num_rows * self._num_epochs

    inputs = [
        ','.join(str(self._num_cols * j + i)
                 for i in range(self._num_cols))
        for j in range(self._num_rows)
    ]
    contents = '\n'.join(inputs).encode('utf-8')

    self._filename = os.path.join(self.get_temp_dir(), 'file.csv')
    self._compressed = os.path.join(self.get_temp_dir(),
                                    'comp.csv')  # GZip compressed

    with open(self._filename, 'wb') as f:
      f.write(contents)
    with gzip.GzipFile(self._compressed, 'wb') as f:
      f.write(contents)

  def ds_func(self, **kwargs):
    # [FUNCTION DOC]
    # HELPER: dataset factory for the checkpoint test.
    # Reads an optional compression type from the keyword arguments (nothing or GZIP) to pick
    # the compressed or plain fixture file, raising a value error for any other type. Returns the
    # CSV dataset (extra options forwarded) repeated for the configured number of epochs.
    compression_type = kwargs.get('compression_type', None)
    if compression_type == 'GZIP':
      filename = self._compressed
    elif compression_type is None:
      filename = self._filename
    else:
      raise ValueError('Invalid compression type:', compression_type)

    return readers.CsvDataset(filename, **kwargs).repeat(self._num_epochs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testCore(self, verify_fn):
    # [FUNCTION DOC]
    # Test: with integer defaults for all seven columns and a buffer size of two, calls the
    # verification callback with the dataset factory and the expected total output count.
    defs = [[0]] * self._num_cols
    verify_fn(self, lambda: self.ds_func(record_defaults=defs, buffer_size=2),
              self._num_outputs)


if __name__ == '__main__':
  test.main()
