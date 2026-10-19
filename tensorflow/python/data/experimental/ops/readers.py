# [MODULE DOC]
# Experimental reader datasets for tf.data. Provides helpers that read TFRecord and CSV files
# into batched datasets (with shuffling, repeating, prefetching and type/column inference for
# CSV), the CSV record dataset class, a SQL query dataset class, and the deprecated and
# versioned (v1 and v2) public wrappers, including legacy file-based dataset classes and the
# factory/wrapper functions that choose between them. Importing registers public export names for
# the experimental data API. Several private functions are helpers for CSV schema inference.
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
"""Python wrappers for reader Datasets."""
import collections
import csv
import functools
import gzip

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.experimental.ops import parsing_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import map_op
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util.tf_export import tf_export

_ACCEPTABLE_CSV_TYPES = (dtypes.float32, dtypes.float64, dtypes.int32,
                         dtypes.int64, dtypes.string)


def _is_valid_int32(str_val):
  # [FUNCTION DOC]
  # HELPER: type probe used by CSV type inference.
  # Returns true if the text converts to a 32-bit integer without overflow (checked by comparing
  # the 32-bit and 64-bit conversions); returns false on a conversion or overflow error.
  try:
    # Checks equality to prevent int32 overflow
    return dtypes.int32.as_numpy_dtype(str_val) == dtypes.int64.as_numpy_dtype(
        str_val)
  except (ValueError, OverflowError):
    return False


def _is_valid_int64(str_val):
  # [FUNCTION DOC]
  # HELPER: type probe used by CSV type inference.
  # Returns true if the text converts to a 64-bit integer, false on a conversion or overflow error.
  try:
    dtypes.int64.as_numpy_dtype(str_val)
    return True
  except (ValueError, OverflowError):
    return False


def _is_valid_float(str_val, float_dtype):
  # [FUNCTION DOC]
  # HELPER: type probe used by CSV type inference.
  # Returns true if the text converts to the given floating point type and the result is below
  # positive infinity (so values overflowing the type are rejected); false on a value error.
  try:
    return float_dtype.as_numpy_dtype(str_val) < np.inf
  except ValueError:
    return False


def _infer_type(str_val, na_value, prev_type):
  # [FUNCTION DOC]
  # HELPER: per-cell step of CSV column type inference.
  # Takes a cell text, the additional text meaning missing, and the type inferred so far for the
  # column (or nothing). Empty or missing cells return the previous type unchanged. Otherwise
  # tries 32-bit int, 64-bit int, 32-bit float, 64-bit float, then string, and returns the first
  # type that accepts the text and is not narrower than the previous type. Pure function.
  """Given a string, infers its tensor type.

  Infers the type of a value by picking the least 'permissive' type possible,
  while still allowing the previous type inference for this column to be valid.

  Args:
    str_val: String value to infer the type of.
    na_value: Additional string to recognize as a NA/NaN CSV value.
    prev_type: Type previously inferred based on values of this column that
      we've seen up till now.
  Returns:
    Inferred dtype.
  """
  if str_val in ("", na_value):
    # If the field is null, it gives no extra information about its type
    return prev_type

  type_list = [
      dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64, dtypes.string
  ]  # list of types to try, ordered from least permissive to most

  type_functions = [
      _is_valid_int32,
      _is_valid_int64,
      lambda str_val: _is_valid_float(str_val, dtypes.float32),
      lambda str_val: _is_valid_float(str_val, dtypes.float64),
      lambda str_val: True,
  ]  # Corresponding list of validation functions

  for i in range(len(type_list)):
    validation_fn = type_functions[i]
    if validation_fn(str_val) and (prev_type is None or
                                   prev_type in type_list[:i + 1]):
      return type_list[i]


def _next_csv_row(filenames, num_cols, field_delim, use_quote_delim, header,
                  file_io_fn):
  # [FUNCTION DOC]
  # HELPER: row reader used by CSV type inference.
  # Generator over the given files in order. filenames is a list of paths; num_cols the expected
  # number of fields per row; field_delim the delimiter; use_quote_delim whether quotes group
  # fields; header whether to skip the first row of each file; file_io_fn opens a path as a text
  # stream. Yields each row as a list of strings. Raises a value error if a row has a different
  # number of fields than num_cols.
  """Generator that yields rows of CSV file(s) in order."""
  for fn in filenames:
    with file_io_fn(fn) as f:
      rdr = csv.reader(
          f,
          delimiter=field_delim,
          quoting=csv.QUOTE_MINIMAL if use_quote_delim else csv.QUOTE_NONE)
      row_num = 1
      if header:
        next(rdr)  # Skip header lines
        row_num += 1

      for csv_row in rdr:
        if len(csv_row) != num_cols:
          raise ValueError(
              f"Problem inferring types: CSV row {row_num} has {len(csv_row)} "
              f"number of fields. Expected: {num_cols}.")
        row_num += 1
        yield csv_row


def _infer_column_defaults(filenames, num_cols, field_delim, use_quote_delim,
                           na_value, header, num_rows_for_inference,
                           select_columns, file_io_fn):
  # [FUNCTION DOC]
  # HELPER: default-value inference for make_csv_dataset when defaults are not supplied.
  # Reads up to num_rows_for_inference rows (all rows if null) from the files and infers a type
  # for each selected column (all columns if select_columns is null). Columns with no information
  # become strings. Returns a list of one-element constant tensors, each holding zero or the empty
  # string, typed by the inferred type, used as optional-column defaults. Raises a value error
  # when a row has the wrong number of fields.
  """Infers column types from the first N valid CSV records of files."""
  if select_columns is None:
    select_columns = range(num_cols)
  inferred_types = [None] * len(select_columns)

  for i, csv_row in enumerate(
      _next_csv_row(filenames, num_cols, field_delim, use_quote_delim, header,
                    file_io_fn)):
    if num_rows_for_inference is not None and i >= num_rows_for_inference:
      break

    for j, col_index in enumerate(select_columns):
      inferred_types[j] = _infer_type(csv_row[col_index], na_value,
                                      inferred_types[j])

  # Replace None's with a default type
  inferred_types = [t or dtypes.string for t in inferred_types]
  # Default to 0 or '' for null values
  return [
      constant_op.constant([0 if t is not dtypes.string else ""], dtype=t)
      for t in inferred_types
  ]


def _infer_column_names(filenames, field_delim, use_quote_delim, file_io_fn):
  # [FUNCTION DOC]
  # HELPER: header-reading step for make_csv_dataset.
  # Reads the first row of the first file as the column names and checks that the first row of
  # every other file is identical. Returns the list of names. Raises a value error if a file is
  # empty or if headers differ between files.
  """Infers column names from first rows of files."""
  csv_kwargs = {
      "delimiter": field_delim,
      "quoting": csv.QUOTE_MINIMAL if use_quote_delim else csv.QUOTE_NONE
  }
  with file_io_fn(filenames[0]) as f:
    try:
      column_names = next(csv.reader(f, **csv_kwargs))
    except StopIteration:
      raise ValueError("Failed when reading the header line of "
                       f"{filenames[0]}. Is it an empty file?")

  for name in filenames[1:]:
    with file_io_fn(name) as f:
      try:
        if next(csv.reader(f, **csv_kwargs)) != column_names:
          raise ValueError(
              "All input CSV files should have the same column names in the "
              f"header row. File {name} has different column names.")
      except StopIteration:
        raise ValueError("Failed when reading the header line of "
                         f"{name}. Is it an empty file?")
  return column_names


def _get_sorted_col_indices(select_columns, column_names):
  # [FUNCTION DOC]
  # HELPER: normalizes the column selection for make_csv_dataset.
  # KNOWN ISSUE: the range error message says indices must be greater than zero and at most the
  # column count, but the check accepts zero and rejects the count itself. Also, duplicates are
  # removed by a set before the comparison, so the duplicate list in the error message is always
  # empty.
  # Takes a list of integer indices or column names and the list of all names; returns a sorted
  # list of unique integer indices. Raises a value error for out-of-range indices, unknown names,
  # or duplicate selections.
  """Transforms select_columns argument into sorted column indices."""
  names_to_indices = {n: i for i, n in enumerate(column_names)}
  num_cols = len(column_names)

  results = []
  for v in select_columns:
    # If value is already an int, check if it's valid.
    if isinstance(v, int):
      if v < 0 or v >= num_cols:
        raise ValueError(
            f"Column index {v} specified in `select_columns` should be > 0 "
            f" and <= {num_cols}, which is the number of columns.")
      results.append(v)
    # Otherwise, check that it's a valid column name and convert to the
    # the relevant column index.
    elif v not in names_to_indices:
      raise ValueError(
          f"Column {v} specified in `select_columns` must be of one of the "
          f"columns: {names_to_indices.keys()}.")
    else:
      results.append(names_to_indices[v])

  # Sort and ensure there are no duplicates
  results = sorted(set(results))
  if len(results) != len(select_columns):
    sorted_names = sorted(results)
    duplicate_columns = set([a for a, b in zip(
        sorted_names[:-1], sorted_names[1:]) if a == b])
    raise ValueError("The `select_columns` argument contains duplicate "
                     f"columns: {duplicate_columns}.")
  return results


def _maybe_shuffle_and_repeat(
    dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed):
  # [FUNCTION DOC]
  # HELPER: shared by the TFRecord and CSV dataset builders.
  # Optionally shuffles the dataset with the given buffer size and seed, then repeats it for
  # num_epochs unless num_epochs is exactly one (null means repeat forever). Returns the dataset.
  """Optionally shuffle and repeat dataset, as requested."""
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, shuffle_seed)
  if num_epochs != 1:
    dataset = dataset.repeat(num_epochs)
  return dataset


def make_tf_record_dataset(file_pattern,
                           batch_size,
                           parser_fn=None,
                           num_epochs=None,
                           shuffle=True,
                           shuffle_buffer_size=None,
                           shuffle_seed=None,
                           prefetch_buffer_size=None,
                           num_parallel_reads=None,
                           num_parallel_parser_calls=None,
                           drop_final_batch=False):
  # [FUNCTION DOC]
  # Builds a batched dataset from TFRecord files. file_pattern is a list of paths or patterns;
  # batch_size the batch size; parser_fn an optional per-record parser (records stay raw strings
  # if absent). Optional arguments: num_epochs (null repeats forever), shuffle (default true)
  # with its buffer size (default 10000) and seed, prefetch size (default autotune, zero disables),
  # parallel reads (default 24, interleaved), parallel parser calls (default batch size), and
  # whether to drop a short final batch (forced on when repeating forever so batch size is
  # static). Lists and optionally shuffles files, reads records, shuffles/repeats, parses, batches
  # and prefetches. Returns the dataset.
  """Reads and optionally parses TFRecord files into a dataset.

  Provides common functionality such as batching, optional parsing, shuffling,
  and performant defaults.

  Args:
    file_pattern: List of files or patterns of TFRecord file paths.
      See `tf.io.gfile.glob` for pattern rules.
    batch_size: An int representing the number of records to combine
      in a single batch.
    parser_fn: (Optional.) A function accepting string input to parse
      and process the record contents. This function must map records
      to components of a fixed shape, so they may be batched. By
      default, uses the record contents unmodified.
    num_epochs: (Optional.) An int specifying the number of times this
      dataset is repeated.  If None (the default), cycles through the
      dataset forever.
    shuffle: (Optional.) A bool that indicates whether the input
      should be shuffled. Defaults to `True`.
    shuffle_buffer_size: (Optional.) Buffer size to use for
      shuffling. A large buffer size ensures better shuffling, but
      increases memory usage and startup time.
    shuffle_seed: (Optional.) Randomization seed to use for shuffling.
    prefetch_buffer_size: (Optional.) An int specifying the number of
      feature batches to prefetch for performance improvement.
      Defaults to auto-tune. Set to 0 to disable prefetching.
    num_parallel_reads: (Optional.) Number of threads used to read
      records from files. By default or if set to a value >1, the
      results will be interleaved. Defaults to `24`.
    num_parallel_parser_calls: (Optional.) Number of parallel
      records to parse in parallel. Defaults to `batch_size`.
    drop_final_batch: (Optional.) Whether the last batch should be
      dropped in case its size is smaller than `batch_size`; the
      default behavior is not to drop the smaller batch.

  Returns:
    A dataset, where each element matches the output of `parser_fn`
    except it will have an additional leading `batch-size` dimension,
    or a `batch_size`-length 1-D tensor of strings if `parser_fn` is
    unspecified.
  """
  if num_parallel_reads is None:
    # NOTE: We considered auto-tuning this value, but there is a concern
    # that this affects the mixing of records from different files, which
    # could affect training convergence/accuracy, so we are defaulting to
    # a constant for now.
    num_parallel_reads = 24

  if num_parallel_parser_calls is None:
    # TODO(josh11b): if num_parallel_parser_calls is None, use some function
    # of num cores instead of `batch_size`.
    num_parallel_parser_calls = batch_size

  if prefetch_buffer_size is None:
    prefetch_buffer_size = dataset_ops.AUTOTUNE

  files = dataset_ops.Dataset.list_files(
      file_pattern, shuffle=shuffle, seed=shuffle_seed)

  dataset = core_readers.TFRecordDataset(
      files, num_parallel_reads=num_parallel_reads)

  if shuffle_buffer_size is None:
    # TODO(josh11b): Auto-tune this value when not specified
    shuffle_buffer_size = 10000
  dataset = _maybe_shuffle_and_repeat(
      dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)

  # NOTE(mrry): We set `drop_final_batch=True` when `num_epochs is None` to
  # improve the shape inference, because it makes the batch dimension static.
  # It is safe to do this because in that case we are repeating the input
  # indefinitely, and all batches will be full-sized.
  drop_final_batch = drop_final_batch or num_epochs is None

  if parser_fn is None:
    dataset = dataset.batch(batch_size, drop_remainder=drop_final_batch)
  else:
    dataset = dataset.map(
        parser_fn, num_parallel_calls=num_parallel_parser_calls)
    dataset = dataset.batch(batch_size, drop_remainder=drop_final_batch)

  if prefetch_buffer_size == 0:
    return dataset
  else:
    return dataset.prefetch(buffer_size=prefetch_buffer_size)


@tf_export("data.experimental.make_csv_dataset", v1=[])
def make_csv_dataset_v2(
    file_pattern,
    batch_size,
    column_names=None,
    column_defaults=None,
    label_name=None,
    select_columns=None,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    header=True,
    num_epochs=None,  # TODO(aaudibert): Change default to 1 when graduating.
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=None,
    num_parallel_reads=None,
    sloppy=False,
    num_rows_for_inference=100,
    compression_type=None,
    ignore_errors=False,
    encoding="utf-8",
):
  # [FUNCTION DOC]
  # Version 2 public API for reading CSV files into a dataset of batched (features, label) pairs,
  # or just a features mapping when no label name is given. Inputs: file pattern(s); batch size;
  # optional column names (inferred from the header if absent), column defaults (types or
  # tensors; inferred from the first rows if absent), label column name, subset of columns to
  # select, delimiter, quoting flag, missing-value text, header flag, epochs (null repeats
  # forever), shuffle flag/buffer/seed, prefetch size, parallel reads (default one), sloppy
  # (non-deterministic) ordering flag, rows used for type inference, compression type, flag to
  # ignore parse errors, and text encoding. Resolves and validates the schema (duplicate names,
  # selection and defaults length mismatch, unknown label, unsupported compression for probing)
  # and raises a value error on problems. Builds a per-file CSV dataset, interleaves files
  # sequentially, in parallel or by autotune, shuffles/repeats, batches (dropping the remainder
  # when repeating forever), maps rows to a mapping of names to tensors, and prefetches.
  # Reads files at graph construction time to infer names and types.
  """Reads CSV files into a dataset.

  Reads CSV files into a dataset, where each element of the dataset is a
  (features, labels) tuple that corresponds to a batch of CSV rows. The features
  dictionary maps feature column names to `Tensor`s containing the corresponding
  feature data, and labels is a `Tensor` containing the batch's label data.

  By default, the first rows of the CSV files are expected to be headers listing
  the column names. If the first rows are not headers, set `header=False` and
  provide the column names with the `column_names` argument.

  By default, the dataset is repeated indefinitely, reshuffling the order each
  time. This behavior can be modified by setting the `num_epochs` and `shuffle`
  arguments.

  For example, suppose you have a CSV file containing

  | Feature_A | Feature_B |
  | --------- | --------- |
  | 1         | "a"       |
  | 2         | "b"       |
  | 3         | "c"       |
  | 4         | "d"       |

  ```
  # No label column specified
  dataset = tf.data.experimental.make_csv_dataset(filename, batch_size=2)
  iterator = dataset.as_numpy_iterator()
  print(dict(next(iterator)))
  # prints a dictionary of batched features:
  # OrderedDict([('Feature_A', array([1, 4], dtype=int32)),
  #              ('Feature_B', array([b'a', b'd'], dtype=object))])
  ```

  ```
  # Set Feature_B as label column
  dataset = tf.data.experimental.make_csv_dataset(
      filename, batch_size=2, label_name="Feature_B")
  iterator = dataset.as_numpy_iterator()
  print(next(iterator))
  # prints (features, labels) tuple:
  # (OrderedDict([('Feature_A', array([1, 2], dtype=int32))]),
  #  array([b'a', b'b'], dtype=object))
  ```

  See the
  [Load CSV data guide](https://www.tensorflow.org/tutorials/load_data/csv) for
  more examples of using `make_csv_dataset` to read CSV data.

  Args:
    file_pattern: List of files or patterns of file paths containing CSV
      records. See `tf.io.gfile.glob` for pattern rules.
    batch_size: An int representing the number of records to combine
      in a single batch.
    column_names: An optional list of strings that corresponds to the CSV
      columns, in order. One per column of the input record. If this is not
      provided, infers the column names from the first row of the records.
      These names will be the keys of the features dict of each dataset element.
    column_defaults: A optional list of default values for the CSV fields. One
      item per selected column of the input record. Each item in the list is
      either a valid CSV dtype (float32, float64, int32, int64, or string), or a
      `Tensor` with one of the aforementioned types. The tensor can either be
      a scalar default value (if the column is optional), or an empty tensor (if
      the column is required). If a dtype is provided instead of a tensor, the
      column is also treated as required. If this list is not provided, tries
      to infer types based on reading the first num_rows_for_inference rows of
      files specified, and assumes all columns are optional, defaulting to `0`
      for numeric values and `""` for string values. If both this and
      `select_columns` are specified, these must have the same lengths, and
      `column_defaults` is assumed to be sorted in order of increasing column
      index.
    label_name: A optional string corresponding to the label column. If
      provided, the data for this column is returned as a separate `Tensor` from
      the features dictionary.
    select_columns: An optional list of integer indices or string column
      names, that specifies a subset of columns of CSV data to select. If
      column names are provided, these must correspond to names provided in
      `column_names` or inferred from the file header lines. When this argument
      is specified, only a subset of CSV columns will be parsed and returned,
      corresponding to the columns specified. Using this results in faster
      parsing and lower memory usage. If both this and `column_defaults` are
      specified, these must have the same lengths, and `column_defaults` is
      assumed to be sorted in order of increasing column index.
    field_delim: An optional `string`. Defaults to `","`. Char delimiter to
      separate fields in a record.
    use_quote_delim: An optional bool. Defaults to `True`. If false, treats
      double quotation marks as regular characters inside of the string fields.
    na_value: Additional string to recognize as NA/NaN.
    header: A bool that indicates whether the first rows of provided CSV files
      correspond to header lines with column names, and should not be included
      in the data.
    num_epochs: An int specifying the number of times this dataset is repeated.
      If None, cycles through the dataset forever.
    shuffle: A bool that indicates whether the input should be shuffled.
    shuffle_buffer_size: Buffer size to use for shuffling. A large buffer size
      ensures better shuffling, but increases memory usage and startup time.
    shuffle_seed: Randomization seed to use for shuffling.
    prefetch_buffer_size: An int specifying the number of feature
      batches to prefetch for performance improvement. Recommended value is the
      number of batches consumed per training step. Defaults to auto-tune.
    num_parallel_reads: Number of threads used to read CSV records from files.
      If >1, the results will be interleaved. Defaults to `1`.
    sloppy: If `True`, reading performance will be improved at
      the cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order
      of elements after shuffling is deterministic). Defaults to `False`.
    num_rows_for_inference: Number of rows of a file to use for type inference
      if record_defaults is not provided. If None, reads all the rows of all
      the files. Defaults to 100.
    compression_type: (Optional.) A `tf.string` scalar evaluating to one of
      `""` (no compression), `"ZLIB"`, or `"GZIP"`. Defaults to no compression.
    ignore_errors: (Optional.) If `True`, ignores errors with CSV file parsing,
      such as malformed data or empty lines, and moves on to the next valid
      CSV record. Otherwise, the dataset raises an error and stops processing
      when encountering any invalid records. Defaults to `False`.
    encoding: Encoding to use when reading. Defaults to `UTF-8`.

  Returns:
    A dataset, where each element is a (features, labels) tuple that corresponds
    to a batch of `batch_size` CSV rows. The features dictionary maps feature
    column names to `Tensor`s containing the corresponding column data, and
    labels is a `Tensor` containing the column data for the label column
    specified by `label_name`.

  Raises:
    ValueError: If any of the arguments is malformed.
  """
  if num_parallel_reads is None:
    num_parallel_reads = 1

  if prefetch_buffer_size is None:
    prefetch_buffer_size = dataset_ops.AUTOTUNE

  # Create dataset of all matching filenames
  filenames = _get_file_names(file_pattern, False)
  dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
  if shuffle:
    dataset = dataset.shuffle(len(filenames), shuffle_seed)

  # Clean arguments; figure out column names and defaults
  if column_names is None or column_defaults is None:
    # Find out which io function to open the file
    file_io_fn = lambda filename: file_io.FileIO(  # pylint: disable=g-long-lambda
        filename, "r", encoding=encoding)
    if compression_type is not None:
      compression_type_value = tensor_util.constant_value(compression_type)
      if compression_type_value is None:
        raise ValueError(
            f"Received unknown `compression_type` {compression_type}. "
            "Expected: GZIP, ZLIB or "" (empty string).")
      if compression_type_value == "GZIP":
        file_io_fn = lambda filename: gzip.open(  # pylint: disable=g-long-lambda
            filename, "rt", encoding=encoding)
      elif compression_type_value == "ZLIB":
        raise ValueError(
            f"`compression_type` {compression_type} is not supported for "
            "probing columns.")
      elif compression_type_value != "":
        raise ValueError(
            f"Received unknown `compression_type` {compression_type}. "
            "Expected: GZIP, ZLIB or "
            " (empty string).")
  if column_names is None:
    if not header:
      raise ValueError("Expected `column_names` or `header` arguments. Neither "
                       "is provided.")
    # If column names are not provided, infer from the header lines
    column_names = _infer_column_names(filenames, field_delim, use_quote_delim,
                                       file_io_fn)
  if len(column_names) != len(set(column_names)):
    sorted_names = sorted(column_names)
    duplicate_columns = set([a for a, b in zip(
        sorted_names[:-1], sorted_names[1:]) if a == b])
    raise ValueError(
        "Either `column_names` argument or CSV header row contains duplicate "
        f"column names: {duplicate_columns}.")

  if select_columns is not None:
    select_columns = _get_sorted_col_indices(select_columns, column_names)

  if column_defaults is not None:
    column_defaults = [
        constant_op.constant([], dtype=x)
        if not tensor_util.is_tf_type(x) and x in _ACCEPTABLE_CSV_TYPES else x
        for x in column_defaults
    ]
  else:
    # If column defaults are not provided, infer from records at graph
    # construction time
    column_defaults = _infer_column_defaults(filenames, len(column_names),
                                             field_delim, use_quote_delim,
                                             na_value, header,
                                             num_rows_for_inference,
                                             select_columns, file_io_fn)

  if select_columns is not None and len(column_defaults) != len(select_columns):
    raise ValueError(
        "If specified, `column_defaults` and `select_columns` must have the "
        f"same length: `column_defaults` has length {len(column_defaults)}, "
        f"`select_columns` has length {len(select_columns)}.")
  if select_columns is not None and len(column_names) > len(select_columns):
    # Pick the relevant subset of column names
    column_names = [column_names[i] for i in select_columns]

  if label_name is not None and label_name not in column_names:
    raise ValueError("`label_name` provided must be one of the columns: "
                     f"{column_names}. Received: {label_name}.")

  def filename_to_dataset(filename):
    # [FUNCTION DOC]
    # HELPER: per-file reader for the CSV dataset builders.
    # Creates a CSV record dataset for one file using the resolved schema and options, optionally
    # wrapped to skip erroneous records, and returns it.
    dataset = CsvDataset(
        filename,
        record_defaults=column_defaults,
        field_delim=field_delim,
        use_quote_delim=use_quote_delim,
        na_value=na_value,
        select_cols=select_columns,
        header=header,
        compression_type=compression_type
    )
    if ignore_errors:
      dataset = dataset.apply(error_ops.ignore_errors())
    return dataset

  def map_fn(*columns):
    # [FUNCTION DOC]
    # HELPER: batch mapping step of the CSV dataset builders.
    # Takes the column tensors of one batch and returns an ordered mapping from column names to
    # tensors; if a label name was given, that entry is removed and returned as a second element
    # together with the remaining features.
    """Organizes columns into a features dictionary.

    Args:
      *columns: list of `Tensor`s corresponding to one csv record.
    Returns:
      An OrderedDict of feature names to values for that particular record. If
      label_name is provided, extracts the label feature to be returned as the
      second element of the tuple.
    """
    features = collections.OrderedDict(zip(column_names, columns))
    if label_name is not None:
      label = features.pop(label_name)
      return features, label
    return features

  if num_parallel_reads == dataset_ops.AUTOTUNE:
    dataset = dataset.interleave(
        filename_to_dataset, num_parallel_calls=num_parallel_reads)
    options = options_lib.Options()
    options.deterministic = not sloppy
    dataset = dataset.with_options(options)
  else:
    # Read files sequentially (if num_parallel_reads=1) or in parallel
    def apply_fn(dataset):
      # [FUNCTION DOC]
      # HELPER: wraps the file dataset in a parallel interleave over per-file CSV datasets using
      # the configured cycle length and sloppy flag; returns the interleaved dataset.
      return core_readers.ParallelInterleaveDataset(
          dataset,
          filename_to_dataset,
          cycle_length=num_parallel_reads,
          block_length=1,
          sloppy=sloppy,
          buffer_output_elements=None,
          prefetch_input_elements=None)

    dataset = dataset.apply(apply_fn)

  dataset = _maybe_shuffle_and_repeat(
      dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)

  # Apply batch before map for perf, because map has high overhead relative
  # to the size of the computation in each map.
  # NOTE(mrry): We set `drop_remainder=True` when `num_epochs is None` to
  # improve the shape inference, because it makes the batch dimension static.
  # It is safe to do this because in that case we are repeating the input
  # indefinitely, and all batches will be full-sized.
  dataset = dataset.batch(batch_size=batch_size,
                          drop_remainder=num_epochs is None)
  dataset = map_op._MapDataset(  # pylint: disable=protected-access
      dataset, map_fn, use_inter_op_parallelism=False)
  dataset = dataset.prefetch(prefetch_buffer_size)

  return dataset


@tf_export(v1=["data.experimental.make_csv_dataset"])
def make_csv_dataset_v1(
    file_pattern,
    batch_size,
    column_names=None,
    column_defaults=None,
    label_name=None,
    select_columns=None,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    header=True,
    num_epochs=None,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=None,
    num_parallel_reads=None,
    sloppy=False,
    num_rows_for_inference=100,
    compression_type=None,
    ignore_errors=False,
    encoding="utf-8",
):  # pylint: disable=missing-docstring
  # [FUNCTION DOC]
  # Version 1 public API for reading CSV files. Takes exactly the same arguments as the version 2
  # function, calls it, and wraps the resulting dataset in the version 1 adapter. Its documentation
  # text is copied from the version 2 function after definition. Returns the adapted dataset.
  return dataset_ops.DatasetV1Adapter(
      make_csv_dataset_v2(file_pattern, batch_size, column_names,
                          column_defaults, label_name, select_columns,
                          field_delim, use_quote_delim, na_value, header,
                          num_epochs, shuffle, shuffle_buffer_size,
                          shuffle_seed, prefetch_buffer_size,
                          num_parallel_reads, sloppy, num_rows_for_inference,
                          compression_type, ignore_errors, encoding))
make_csv_dataset_v1.__doc__ = make_csv_dataset_v2.__doc__


_DEFAULT_READER_BUFFER_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB


@tf_export("data.experimental.CsvDataset", v1=[])
class CsvDatasetV2(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # Version 2 public CSV dataset: a source dataset whose elements are tuples of scalar tensors,
  # one per selected column, parsed from rows of one or more CSV files by a native op. Holds the
  # settings as tensors (file names, compression, defaults, buffer size, header flag, delimiter,
  # quoting flag, missing-value text, selected and excluded columns) and the element spec derived
  # from the defaults' types. Registered under the public CsvDataset name for version 2.
  r"""A Dataset comprising lines from one or more CSV files.

  The `tf.data.experimental.CsvDataset` class provides a minimal CSV Dataset
  interface. There is also a richer `tf.data.experimental.make_csv_dataset`
  function which provides additional convenience features such as column header
  parsing, column type-inference, automatic shuffling, and file interleaving.

  The elements of this dataset correspond to records from the file(s).
  RFC 4180 format is expected for CSV files
  (https://tools.ietf.org/html/rfc4180)
  Note that we allow leading and trailing spaces for int or float fields.

  For example, suppose we have a file 'my_file0.csv' with four CSV columns of
  different data types:

  >>> with open('/tmp/my_file0.csv', 'w') as f:
  ...   f.write('abcdefg,4.28E10,5.55E6,12\n')
  ...   f.write('hijklmn,-5.3E14,,2\n')

  We can construct a CsvDataset from it as follows:

  >>> dataset = tf.data.experimental.CsvDataset(
  ...   "/tmp/my_file0.csv",
  ...   [tf.float32,  # Required field, use dtype or empty tensor
  ...    tf.constant([0.0], dtype=tf.float32),  # Optional field, default to 0.0
  ...    tf.int32,  # Required field, use dtype or empty tensor
  ...   ],
  ...   select_cols=[1,2,3]  # Only parse last three columns
  ... )

  The expected output of its iterations is:
  >>> for n0, n1, n2 in dataset.as_numpy_iterator():
  ...   print(n0, n1, n2)
  4.28e10 5.55e6 12
  -5.3e14 0.0 2

  See
  https://www.tensorflow.org/tutorials/load_data/csv#tfdataexperimentalcsvdataset
  for more in-depth example usage.
  """

  def __init__(
      self,
      filenames,
      record_defaults,
      compression_type=None,
      buffer_size=None,
      header=False,
      field_delim=",",
      use_quote_delim=True,
      na_value="",
      select_cols=None,
      exclude_cols=None,
  ):
    # [FUNCTION DOC]
    # Converts all settings to tensors and creates the native CSV dataset op. filenames: one or
    # more file names; record_defaults: list of dtypes (required column) or tensors (default value
    # for optional column; empty tensor means required), one per parsed column; compression_type
    # (default none); buffer_size in bytes (default 4 MB); header flag (default false);
    # field_delim (default comma); use_quote_delim (default true); na_value text meaning missing
    # (default empty); select_cols or exclude_cols: optional sorted column index lists, at most one
    # should be given. Bare dtypes in defaults become empty tensors. The element spec is a tuple of
    # scalar specs with the defaults' dtypes. At run time the op raises an invalid-argument error
    # if the column counts do not match. Returns nothing.
    """Creates a `CsvDataset` by reading and decoding CSV files.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      record_defaults: A list of default values for the CSV fields. Each item in
        the list is either a valid CSV `DType` (float32, float64, int32, int64,
        string), or a `Tensor` object with one of the above types. One per
        column of CSV data, with either a scalar `Tensor` default value for the
        column if it is optional, or `DType` or empty `Tensor` if required. If
        both this and `select_columns` are specified, these must have the same
        lengths, and `column_defaults` is assumed to be sorted in order of
        increasing column index. If both this and 'exclude_cols' are specified,
        the sum of lengths of record_defaults and exclude_cols should equal
        the total number of columns in the CSV file.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`. Defaults to no
        compression.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer while reading files. Defaults to 4MB.
      header: (Optional.) A `tf.bool` scalar indicating whether the CSV file(s)
        have header line(s) that should be skipped when parsing. Defaults to
        `False`.
      field_delim: (Optional.) A `tf.string` scalar containing the delimiter
        character that separates fields in a record. Defaults to `","`.
      use_quote_delim: (Optional.) A `tf.bool` scalar. If `False`, treats
        double quotation marks as regular characters inside of string fields
        (ignoring RFC 4180, Section 2, Bullet 5). Defaults to `True`.
      na_value: (Optional.) A `tf.string` scalar indicating a value that will
        be treated as NA/NaN.
      select_cols: (Optional.) A sorted list of column indices to select from
        the input data. If specified, only this subset of columns will be
        parsed. Defaults to parsing all columns. At most one of `select_cols`
        and `exclude_cols` can be specified.
      exclude_cols: (Optional.) A sorted list of column indices to exclude from
        the input data. If specified, only the complement of this set of column
        will be parsed. Defaults to parsing all columns. At most one of
        `select_cols` and `exclude_cols` can be specified.

    Raises:
       InvalidArgumentError: If exclude_cols is not None and
           len(exclude_cols) + len(record_defaults) does not match the total
           number of columns in the file(s)


    """
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    record_defaults = [
        constant_op.constant([], dtype=x)
        if not tensor_util.is_tf_type(x) and x in _ACCEPTABLE_CSV_TYPES else x
        for x in record_defaults
    ]
    self._record_defaults = ops.convert_n_to_tensor(
        record_defaults, name="record_defaults")
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size", buffer_size, _DEFAULT_READER_BUFFER_SIZE_BYTES)
    self._header = ops.convert_to_tensor(
        header, dtype=dtypes.bool, name="header")
    self._field_delim = ops.convert_to_tensor(
        field_delim, dtype=dtypes.string, name="field_delim")
    self._use_quote_delim = ops.convert_to_tensor(
        use_quote_delim, dtype=dtypes.bool, name="use_quote_delim")
    self._na_value = ops.convert_to_tensor(
        na_value, dtype=dtypes.string, name="na_value")
    self._select_cols = convert.optional_param_to_tensor(
        "select_cols",
        select_cols,
        argument_default=[],
        argument_dtype=dtypes.int64,
    )
    self._exclude_cols = convert.optional_param_to_tensor(
        "exclude_cols",
        exclude_cols,
        argument_default=[],
        argument_dtype=dtypes.int64,
    )
    self._element_spec = tuple(
        tensor_spec.TensorSpec([], d.dtype) for d in self._record_defaults)
    variant_tensor = gen_experimental_dataset_ops.csv_dataset_v2(
        filenames=self._filenames,
        record_defaults=self._record_defaults,
        buffer_size=self._buffer_size,
        header=self._header,
        output_shapes=self._flat_shapes,
        field_delim=self._field_delim,
        use_quote_delim=self._use_quote_delim,
        na_value=self._na_value,
        select_cols=self._select_cols,
        exclude_cols=self._exclude_cols,
        compression_type=self._compression_type)
    super(CsvDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the tuple of scalar tensor specs, one per parsed column.
    return self._element_spec


@tf_export(v1=["data.experimental.CsvDataset"])
class CsvDatasetV1(dataset_ops.DatasetV1Adapter):
  # [CLASS DOC]
  # Version 1 public CSV dataset: a thin adapter around the version 2 class that exposes the
  # version 1 dataset interface. Registered under the public CsvDataset name for version 1.
  """A Dataset comprising lines from one or more CSV files."""

  @functools.wraps(CsvDatasetV2.__init__, ("__module__", "__name__"))
  def __init__(self,
               filenames,
               record_defaults,
               compression_type=None,
               buffer_size=None,
               header=False,
               field_delim=",",
               use_quote_delim=True,
               na_value="",
               select_cols=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring mentions an exclude_cols argument, but this constructor has no
    # such parameter and never passes one to the version 2 class.
    # Creates the version 2 CSV dataset from the same arguments (file names, per-column defaults,
    # compression type, buffer size, header flag, delimiter, quoting flag, missing-value text,
    # selected columns) and hands it to the version 1 adapter. Returns nothing.
    """Creates a `CsvDataset` by reading and decoding CSV files.

    The elements of this dataset correspond to records from the file(s).
    RFC 4180 format is expected for CSV files
    (https://tools.ietf.org/html/rfc4180)
    Note that we allow leading and trailing spaces with int or float field.


    For example, suppose we have a file 'my_file0.csv' with four CSV columns of
    different data types:
    ```
    abcdefg,4.28E10,5.55E6,12
    hijklmn,-5.3E14,,2
    ```

    We can construct a CsvDataset from it as follows:

    ```python
     dataset = tf.data.experimental.CsvDataset(
        "my_file*.csv",
        [tf.float32,  # Required field, use dtype or empty tensor
         tf.constant([0.0], dtype=tf.float32),  # Optional field, default to 0.0
         tf.int32,  # Required field, use dtype or empty tensor
         ],
        select_cols=[1,2,3]  # Only parse last three columns
    )
    ```

    The expected output of its iterations is:

    ```python
    for element in dataset:
      print(element)

    >> (4.28e10, 5.55e6, 12)
    >> (-5.3e14, 0.0, 2)
    ```

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      record_defaults: A list of default values for the CSV fields. Each item in
        the list is either a valid CSV `DType` (float32, float64, int32, int64,
        string), or a `Tensor` object with one of the above types. One per
        column of CSV data, with either a scalar `Tensor` default value for the
        column if it is optional, or `DType` or empty `Tensor` if required. If
        both this and `select_columns` are specified, these must have the same
        lengths, and `column_defaults` is assumed to be sorted in order of
        increasing column index. If both this and 'exclude_cols' are specified,
        the sum of lengths of record_defaults and exclude_cols should equal the
        total number of columns in the CSV file.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`. Defaults to no
        compression.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer while reading files. Defaults to 4MB.
      header: (Optional.) A `tf.bool` scalar indicating whether the CSV file(s)
        have header line(s) that should be skipped when parsing. Defaults to
        `False`.
      field_delim: (Optional.) A `tf.string` scalar containing the delimiter
        character that separates fields in a record. Defaults to `","`.
      use_quote_delim: (Optional.) A `tf.bool` scalar. If `False`, treats double
        quotation marks as regular characters inside of string fields (ignoring
        RFC 4180, Section 2, Bullet 5). Defaults to `True`.
      na_value: (Optional.) A `tf.string` scalar indicating a value that will be
        treated as NA/NaN.
      select_cols: (Optional.) A sorted list of column indices to select from
        the input data. If specified, only this subset of columns will be
        parsed. Defaults to parsing all columns. At most one of `select_cols`
        and `exclude_cols` can be specified.
    """
    wrapped = CsvDatasetV2(filenames, record_defaults, compression_type,
                           buffer_size, header, field_delim, use_quote_delim,
                           na_value, select_cols)
    super(CsvDatasetV1, self).__init__(wrapped)


@tf_export("data.experimental.make_batched_features_dataset", v1=[])
def make_batched_features_dataset_v2(file_pattern,
                                     batch_size,
                                     features,
                                     reader=None,
                                     label_key=None,
                                     reader_args=None,
                                     num_epochs=None,
                                     shuffle=True,
                                     shuffle_buffer_size=10000,
                                     shuffle_seed=None,
                                     prefetch_buffer_size=None,
                                     reader_num_threads=None,
                                     parser_num_threads=None,
                                     sloppy_ordering=False,
                                     drop_final_batch=False):
  # [FUNCTION DOC]
  # Version 2 public API building a batched dataset of parsed Example records. Inputs: file
  # pattern(s), batch size, a mapping of feature keys to feature specs, reader (a callable that
  # maps a file name and extra reader arguments to a dataset; default TFRecord dataset), optional
  # label key, extra reader arguments, epochs (null repeats forever), shuffle flag/buffer/seed,
  # prefetch size (default autotune), reader threads (default 1, interleaved), parser threads
  # (default 2), sloppy ordering flag, and drop-final-batch flag. Lists files, reads records
  # through the reader (taking the value if elements are key/value pairs), shuffles/repeats,
  # batches (dropping the remainder when repeating forever), parses in parallel, optionally
  # splits off the label, and prefetches. Raises a type error if the reader is a legacy reader
  # class and a value error if the label key is not among the features. Returns the dataset.
  """Returns a `Dataset` of feature dictionaries from `Example` protos.

  If label_key argument is provided, returns a `Dataset` of tuple
  comprising of feature dictionaries and label.

  Example:

  ```
  serialized_examples = [
    features {
      feature { key: "age" value { int64_list { value: [ 0 ] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      feature { key: "kws" value { bytes_list { value: [ "code", "art" ] } } }
    },
    features {
      feature { key: "age" value { int64_list { value: [] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      feature { key: "kws" value { bytes_list { value: [ "sports" ] } } }
    }
  ]
  ```

  We can use arguments:

  ```
  features: {
    "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
    "gender": FixedLenFeature([], dtype=tf.string),
    "kws": VarLenFeature(dtype=tf.string),
  }
  ```

  And the expected output is:

  ```python
  {
    "age": [[0], [-1]],
    "gender": [["f"], ["f"]],
    "kws": SparseTensor(
      indices=[[0, 0], [0, 1], [1, 0]],
      values=["code", "art", "sports"]
      dense_shape=[2, 2]),
  }
  ```

  Args:
    file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
    batch_size: An int representing the number of records to combine
      in a single batch.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. See `tf.io.parse_example`.
    reader: A function or class that can be
      called with a `filenames` tensor and (optional) `reader_args` and returns
      a `Dataset` of `Example` tensors. Defaults to `tf.data.TFRecordDataset`.
    label_key: (Optional) A string corresponding to the key labels are stored in
      `tf.Examples`. If provided, it must be one of the `features` key,
      otherwise results in `ValueError`.
    reader_args: Additional arguments to pass to the reader class.
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever. Defaults to `None`.
    shuffle: A boolean, indicates whether the input should be shuffled. Defaults
      to `True`.
    shuffle_buffer_size: Buffer size of the ShuffleDataset. A large capacity
      ensures better shuffling but would increase memory usage and startup time.
    shuffle_seed: Randomization seed to use for shuffling.
    prefetch_buffer_size: Number of feature batches to prefetch in order to
      improve performance. Recommended value is the number of batches consumed
      per training step. Defaults to auto-tune.
    reader_num_threads: Number of threads used to read `Example` records. If >1,
      the results will be interleaved. Defaults to `1`.
    parser_num_threads: Number of threads to use for parsing `Example` tensors
      into a dictionary of `Feature` tensors. Defaults to `2`.
    sloppy_ordering: If `True`, reading performance will be improved at
      the cost of non-deterministic ordering. If `False`, the order of elements
      produced is deterministic prior to shuffling (elements are still
      randomized if `shuffle=True`. Note that if the seed is set, then order
      of elements after shuffling is deterministic). Defaults to `False`.
    drop_final_batch: If `True`, and the batch size does not evenly divide the
      input dataset size, the final smaller batch will be dropped. Defaults to
      `False`.

  Returns:
    A dataset of `dict` elements, (or a tuple of `dict` elements and label).
    Each `dict` maps feature keys to `Tensor` or `SparseTensor` objects.

  Raises:
    TypeError: If `reader` is of the wrong type.
    ValueError: If `label_key` is not one of the `features` keys.
  """
  if reader is None:
    reader = core_readers.TFRecordDataset

  if reader_num_threads is None:
    reader_num_threads = 1
  if parser_num_threads is None:
    parser_num_threads = 2
  if prefetch_buffer_size is None:
    prefetch_buffer_size = dataset_ops.AUTOTUNE

  # Create dataset of all matching filenames
  dataset = dataset_ops.Dataset.list_files(
      file_pattern, shuffle=shuffle, seed=shuffle_seed)

  if isinstance(reader, type) and issubclass(reader, io_ops.ReaderBase):
    raise TypeError("The `reader` argument must return a `Dataset` object. "
                    "`tf.ReaderBase` subclasses are not supported. For "
                    "example, pass `tf.data.TFRecordDataset` instead of "
                    "`tf.TFRecordReader`.")

  # Read `Example` records from files as tensor objects.
  if reader_args is None:
    reader_args = []

  if reader_num_threads == dataset_ops.AUTOTUNE:
    dataset = dataset.interleave(
        lambda filename: reader(filename, *reader_args),
        num_parallel_calls=reader_num_threads)
    options = options_lib.Options()
    options.deterministic = not sloppy_ordering
    dataset = dataset.with_options(options)
  else:
    # Read files sequentially (if reader_num_threads=1) or in parallel
    def apply_fn(dataset):
      # [FUNCTION DOC]
      # HELPER: wraps the file name dataset in a parallel interleave that reads each file with the
      # reader, using the configured thread count and sloppy flag; returns the interleaved dataset.
      return core_readers.ParallelInterleaveDataset(
          dataset,
          lambda filename: reader(filename, *reader_args),
          cycle_length=reader_num_threads,
          block_length=1,
          sloppy=sloppy_ordering,
          buffer_output_elements=None,
          prefetch_input_elements=None)

    dataset = dataset.apply(apply_fn)

  # Extract values if the `Example` tensors are stored as key-value tuples.
  if dataset_ops.get_legacy_output_types(dataset) == (
      dtypes.string, dtypes.string):
    dataset = map_op._MapDataset(  # pylint: disable=protected-access
        dataset, lambda _, v: v, use_inter_op_parallelism=False)

  # Apply dataset repeat and shuffle transformations.
  dataset = _maybe_shuffle_and_repeat(
      dataset, num_epochs, shuffle, shuffle_buffer_size, shuffle_seed)

  # NOTE(mrry): We set `drop_remainder=True` when `num_epochs is None` to
  # improve the shape inference, because it makes the batch dimension static.
  # It is safe to do this because in that case we are repeating the input
  # indefinitely, and all batches will be full-sized.
  dataset = dataset.batch(
      batch_size, drop_remainder=drop_final_batch or num_epochs is None)

  # Parse `Example` tensors to a dictionary of `Feature` tensors.
  dataset = dataset.apply(
      parsing_ops.parse_example_dataset(
          features, num_parallel_calls=parser_num_threads))

  if label_key:
    if label_key not in features:
      raise ValueError(
          f"The `label_key` provided ({label_key}) must be one of the "
          f"`features` keys: {features.keys()}.")
    dataset = dataset.map(lambda x: (x, x.pop(label_key)))

  dataset = dataset.prefetch(prefetch_buffer_size)
  return dataset


@tf_export(v1=["data.experimental.make_batched_features_dataset"])
def make_batched_features_dataset_v1(file_pattern,  # pylint: disable=missing-docstring
                                     batch_size,
                                     features,
                                     reader=None,
                                     label_key=None,
                                     reader_args=None,
                                     num_epochs=None,
                                     shuffle=True,
                                     shuffle_buffer_size=10000,
                                     shuffle_seed=None,
                                     prefetch_buffer_size=None,
                                     reader_num_threads=None,
                                     parser_num_threads=None,
                                     sloppy_ordering=False,
                                     drop_final_batch=False):
  # [FUNCTION DOC]
  # Version 1 public API; takes the same arguments as the version 2 function, calls it and wraps
  # the result in the version 1 dataset adapter. Its documentation is copied from version 2.
  return dataset_ops.DatasetV1Adapter(make_batched_features_dataset_v2(
      file_pattern, batch_size, features, reader, label_key, reader_args,
      num_epochs, shuffle, shuffle_buffer_size, shuffle_seed,
      prefetch_buffer_size, reader_num_threads, parser_num_threads,
      sloppy_ordering, drop_final_batch))
make_batched_features_dataset_v1.__doc__ = (
    make_batched_features_dataset_v2.__doc__)


def _get_file_names(file_pattern, shuffle):
  # [FUNCTION DOC]
  # HELPER: file discovery for the CSV dataset builder.
  # Expands a glob pattern or a list of patterns into file names. The result is sorted unless
  # shuffle is true (callers pass false when determinism is wanted). Raises a value error if a
  # list is empty or nothing matches.
  """Parse list of file names from pattern, optionally shuffled.

  Args:
    file_pattern: File glob pattern, or list of glob patterns.
    shuffle: Whether to shuffle the order of file names.

  Returns:
    List of file names matching `file_pattern`.

  Raises:
    ValueError: If `file_pattern` is empty, or pattern matches no files.
  """
  if isinstance(file_pattern, list):
    if not file_pattern:
      raise ValueError("Argument `file_pattern` should not be empty.")
    file_names = []
    for entry in file_pattern:
      file_names.extend(gfile.Glob(entry))
  else:
    file_names = list(gfile.Glob(file_pattern))

  if not file_names:
    raise ValueError(f"No files match `file_pattern` {file_pattern}.")

  # Sort files so it will be deterministic for unit tests.
  if not shuffle:
    file_names = sorted(file_names)
  return file_names


@tf_export("data.experimental.SqlDataset", v1=[])
class SqlDatasetV2(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # Version 2 public dataset yielding the rows of a SQL query result as tuples of scalar tensors.
  # Holds the driver name, connection string and query as string tensors and the element spec
  # built from the requested column types. Registered under the public SqlDataset name for v2.
  """A `Dataset` consisting of the results from a SQL query.

  `SqlDataset` allows a user to read data from the result set of a SQL query.
  For example:

  ```python
  dataset = tf.data.experimental.SqlDataset("sqlite", "/foo/bar.sqlite3",
                                            "SELECT name, age FROM people",
                                            (tf.string, tf.int32))
  # Prints the rows of the result set of the above query.
  for element in dataset:
    print(element)
  ```
  """

  def __init__(self, driver_name, data_source_name, query, output_types):
    # [FUNCTION DOC]
    # Converts the driver name (only sqlite is supported), connection string and query to string
    # tensors, builds the element spec of scalar tensors from output_types (a structure of
    # dtypes, one per result column) and creates the native SQL dataset op. Returns nothing.
    """Creates a `SqlDataset`.

    Args:
      driver_name: A 0-D `tf.string` tensor containing the database type.
        Currently, the only supported value is 'sqlite'.
      data_source_name: A 0-D `tf.string` tensor containing a connection string
        to connect to the database.
      query: A 0-D `tf.string` tensor containing the SQL query to execute.
      output_types: A tuple of `tf.DType` objects representing the types of the
        columns returned by `query`.
    """
    self._driver_name = ops.convert_to_tensor(
        driver_name, dtype=dtypes.string, name="driver_name")
    self._data_source_name = ops.convert_to_tensor(
        data_source_name, dtype=dtypes.string, name="data_source_name")
    self._query = ops.convert_to_tensor(
        query, dtype=dtypes.string, name="query")
    self._element_spec = nest.map_structure(
        lambda dtype: tensor_spec.TensorSpec([], dtype), output_types)
    variant_tensor = gen_experimental_dataset_ops.sql_dataset(
        self._driver_name, self._data_source_name, self._query,
        **self._flat_structure)
    super(SqlDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the structure of scalar tensor specs describing one result row.
    return self._element_spec


@tf_export(v1=["data.experimental.SqlDataset"])
class SqlDatasetV1(dataset_ops.DatasetV1Adapter):
  # [CLASS DOC]
  # Version 1 public SQL dataset: thin adapter around the version 2 class exposing the version 1
  # dataset interface. Registered under the public SqlDataset name for version 1.
  """A `Dataset` consisting of the results from a SQL query."""

  @functools.wraps(SqlDatasetV2.__init__)
  def __init__(self, driver_name, data_source_name, query, output_types):
    # [FUNCTION DOC]
    # Creates the version 2 SQL dataset from the same arguments (driver name, connection string,
    # query, column types) and passes it to the version 1 adapter. Returns nothing.
    wrapped = SqlDatasetV2(driver_name, data_source_name, query, output_types)
    super(SqlDatasetV1, self).__init__(wrapped)


if tf2.enabled():
  CsvDataset = CsvDatasetV2
  SqlDataset = SqlDatasetV2
  make_batched_features_dataset = make_batched_features_dataset_v2
  make_csv_dataset = make_csv_dataset_v2
else:
  CsvDataset = CsvDatasetV1
  SqlDataset = SqlDatasetV1
  make_batched_features_dataset = make_batched_features_dataset_v1
  make_csv_dataset = make_csv_dataset_v1


def _tf2_callback():
  # [FUNCTION DOC]
  # HELPER: version switching glue, registered as a callback with the v2 behavior toggle.
  # Rebinds the module-level public names for the CSV dataset, SQL dataset, batched features
  # builder and CSV builder to the version 2 implementations when version 2 behavior is enabled,
  # or to the version 1 implementations otherwise. Mutates module globals; returns nothing.
  global CsvDataset, SqlDataset, make_batched_features_dataset, make_csv_dataset
  if tf2.enabled():
    CsvDataset = CsvDatasetV2
    SqlDataset = SqlDatasetV2
    make_batched_features_dataset = make_batched_features_dataset_v2
    make_csv_dataset = make_csv_dataset_v2
  else:
    CsvDataset = CsvDatasetV1
    SqlDataset = SqlDatasetV1
    make_batched_features_dataset = make_batched_features_dataset_v1
    make_csv_dataset = make_csv_dataset_v1


v2_compat.register_data_v2_callback(_tf2_callback)
