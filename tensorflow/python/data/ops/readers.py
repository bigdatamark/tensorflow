# [MODULE DOC]
# Python wrappers for reader datasets: text line, TFRecord and fixed-length record datasets, in
# TF2 and legacy TF1 adapter variants, plus a legacy parallel interleave dataset and helpers that
# validate filenames and build per-file readers. Exported to the public data API through export
# decorators. Import-time side effects: chooses the default public class names depending on
# whether TF2 behavior is enabled and registers a callback so they are rebound when TF2 behavior
# is toggled later.
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
import os

from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import from_tensor_slices_op
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB
# The default TFRecordDataset buffer size is set to -1. The actual default is
# set in the kernel when this value is detected.
_DEFAULT_TF_RECORD_BUFFER_SIZE_BYTES = -1


def _normalise_fspath(path):
  # [FUNCTION DOC]
  # HELPER: path normalizer used when building filename datasets.
  # Input: path, any value. If it is a path-like object (supports the file system path protocol)
  # it is converted to a plain string; any other value is returned unchanged.
  """Convert pathlib-like objects to str (__fspath__ compatibility, PEP 519)."""
  return os.fspath(path) if isinstance(path, os.PathLike) else path


def _create_or_validate_filenames_dataset(filenames, name=None):
  # [FUNCTION DOC]
  # HELPER: shared by all reader dataset constructors to normalize their filenames argument.
  # Inputs: filenames (a dataset of string scalars, or a string, list or tensor convertible to
  # string tensor, possibly containing path-like objects), name (optional operation name).
  # If a dataset is given it is validated: its element type must be string and its element shape
  # must be compatible with a scalar; otherwise a type error is raised. Other inputs have path-like
  # objects converted to strings, are converted to a string tensor (type error if the dtype is
  # not string), flattened to one dimension and turned into a tensor-slice dataset flagged as
  # holding files. Returns the filenames dataset.
  """Creates (or validates) a dataset of filenames.

  Args:
    filenames: Either a list or dataset of filenames. If it is a list, it is
      convert to a dataset. If it is a dataset, its type and shape is validated.
    name: (Optional.) A name for the tf.data operation.

  Returns:
    A dataset of filenames.
  """
  if isinstance(filenames, data_types.DatasetV2):
    element_type = dataset_ops.get_legacy_output_types(filenames)
    if element_type != dtypes.string:
      raise TypeError(
          "The `filenames` argument must contain `tf.string` elements. Got a "
          f"dataset of `{element_type!r}` elements.")
    element_shape = dataset_ops.get_legacy_output_shapes(filenames)
    if not element_shape.is_compatible_with(tensor_shape.TensorShape([])):
      raise TypeError(
          "The `filenames` argument must contain `tf.string` elements of shape "
          "[] (i.e. scalars). Got a dataset of element shape "
          f"{element_shape!r}.")
  else:
    filenames = nest.map_structure(_normalise_fspath, filenames)
    filenames = ops.convert_to_tensor(filenames, dtype_hint=dtypes.string)
    if filenames.dtype != dtypes.string:
      raise TypeError(
          "The `filenames` argument must contain `tf.string` elements. Got "
          f"`{filenames.dtype!r}` elements.")
    filenames = array_ops.reshape(filenames, [-1], name="flat_filenames")
    filenames = from_tensor_slices_op._TensorSliceDataset(  # pylint: disable=protected-access
        filenames,
        is_files=True,
        name=name)
  return filenames


def _create_dataset_reader(dataset_creator,
                           filenames,
                           num_parallel_reads=None,
                           name=None):
  # [FUNCTION DOC]
  # HELPER: shared by the text, TFRecord and fixed-length record dataset constructors.
  # Inputs: dataset_creator (callback taking one filename tensor and returning a dataset),
  # filenames (dataset of filenames), num_parallel_reads (optional; nothing means sequential,
  # the autotune sentinel means autotuned parallelism, any other number is the cycle length),
  # name (optional operation name).
  # Returns a dataset reading all files: sequential flat-map when no parallelism is requested,
  # an autotuned interleave for the autotune sentinel, otherwise a legacy parallel interleave
  # dataset with block length one, non-sloppy ordering and default buffering.
  """Creates a dataset that reads the given files using the given reader.

  Args:
    dataset_creator: A function that takes in a single file name and returns a
      dataset.
    filenames: A `tf.data.Dataset` containing one or more filenames.
    num_parallel_reads: The number of parallel reads we should do.
    name: (Optional.) A name for the tf.data operation.

  Returns:
    A `Dataset` that reads data from `filenames`.
  """

  def read_one_file(filename):
    # [FUNCTION DOC]
    # HELPER: nested mapping callback. Converts the given filename to a string tensor and returns
    # the dataset produced by the enclosing dataset creator callback for that file.
    filename = ops.convert_to_tensor(filename, dtypes.string, name="filename")
    return dataset_creator(filename)

  if num_parallel_reads is None:
    return filenames.flat_map(read_one_file, name=name)
  elif num_parallel_reads == dataset_ops.AUTOTUNE:
    return filenames.interleave(
        read_one_file, num_parallel_calls=num_parallel_reads, name=name)
  else:
    return ParallelInterleaveDataset(
        filenames,
        read_one_file,
        cycle_length=num_parallel_reads,
        block_length=1,
        sloppy=False,
        buffer_output_elements=None,
        prefetch_input_elements=None,
        name=name)


def _get_type(value):
  # [FUNCTION DOC]
  # HELPER: produces readable type names for error messages.
  # Returns the value type of a type specification, or the ordinary type of any other value.
  """Returns the type of `value` if it is a TypeSpec."""

  if isinstance(value, type_spec.TypeSpec):
    return value.value_type()
  else:
    return type(value)


class _TextLineDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # HELPER: single-file-level text line source used by the public text line dataset classes.
  # Dataset of lines read from the given filenames tensor; each element is a scalar string.
  # Holds filenames, compression type and buffer size tensors.
  """A `Dataset` comprising records from one or more text files."""

  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               name=None):
    # [FUNCTION DOC]
    # Inputs: filenames (string tensor of one or more names), compression_type (optional string
    # scalar: empty for none, ZLIB or GZIP; default empty), buffer_size (optional integer byte
    # count; default 256 kilobytes), name (optional operation name).
    # Converts the optional parameters to tensors with defaults and creates the text line op, then
    # calls the parent constructor.
    """Creates a `TextLineDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer. A value of 0 results in the default buffering values chosen
        based on the compression type.
      name: (Optional.) A name for the tf.data operation.
    """
    self._filenames = filenames
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size",
        buffer_size,
        argument_default=_DEFAULT_READER_BUFFER_SIZE_BYTES)
    self._name = name

    variant_tensor = gen_dataset_ops.text_line_dataset(
        self._filenames,
        self._compression_type,
        self._buffer_size,
        metadata=self._metadata.SerializeToString())
    super(_TextLineDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec: a scalar string tensor spec (one line per element).
    return tensor_spec.TensorSpec([], dtypes.string)


@tf_export("data.TextLineDataset", v1=[])
class TextLineDatasetV2(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # Public dataset (exported under the TF2 data API) whose elements are the lines of one or more
  # text files, with newline characters stripped. Holds the filenames dataset, compression type,
  # buffer size and an internal implementation dataset that does the actual (optionally parallel)
  # reading.
  r"""Creates a `Dataset` comprising lines from one or more text files.

  The `tf.data.TextLineDataset` loads text from text files and creates a dataset
  where each line of the files becomes an element of the dataset.

  For example, suppose we have 2 files "text_lines0.txt" and "text_lines1.txt"
  with the following lines:

  >>> with open('/tmp/text_lines0.txt', 'w') as f:
  ...   f.write('the cow\n')
  ...   f.write('jumped over\n')
  ...   f.write('the moon\n')
  >>> with open('/tmp/text_lines1.txt', 'w') as f:
  ...   f.write('jack and jill\n')
  ...   f.write('went up\n')
  ...   f.write('the hill\n')

  We can construct a TextLineDataset from them as follows:

  >>> dataset = tf.data.TextLineDataset(['/tmp/text_lines0.txt',
  ...                                    '/tmp/text_lines1.txt'])

  The elements of the dataset are expected to be:

  >>> for element in dataset.as_numpy_iterator():
  ...   print(element)
  b'the cow'
  b'jumped over'
  b'the moon'
  b'jack and jill'
  b'went up'
  b'the hill'
  """

  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               num_parallel_reads=None,
               name=None):
    # [FUNCTION DOC]
    # Inputs: filenames (dataset of strings, string tensor, or list of strings/paths),
    # compression_type (optional: empty, ZLIB or GZIP), buffer_size (optional byte count),
    # num_parallel_reads (optional number of files read in parallel; nothing means sequential),
    # name (optional operation name).
    # Validates or builds the filenames dataset (type error for non-string filenames), builds the
    # internal reader dataset using a callback that creates the single-file text line dataset, and
    # calls the parent constructor with its variant tensor.
    r"""Creates a `TextLineDataset`.

    The elements of the dataset will be the lines of the input files, using
    the newline character '\n' to denote line splits. The newline characters
    will be stripped off of each element.

    Args:
      filenames: A `tf.data.Dataset` whose elements are `tf.string` scalars, a
        `tf.string` tensor, or a value that can be converted to a `tf.string`
        tensor (such as a list of Python strings).
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer. A value of 0 results in the default buffering values chosen
        based on the compression type.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. If greater than one, the records of
        files read in parallel are outputted in an interleaved order. If your
        input pipeline is I/O bottlenecked, consider setting this parameter to a
        value greater than one to parallelize the I/O. If `None`, files will be
        read sequentially.
      name: (Optional.) A name for the tf.data operation.
    """
    filenames = _create_or_validate_filenames_dataset(filenames, name=name)
    self._filenames = filenames
    self._compression_type = compression_type
    self._buffer_size = buffer_size

    def creator_fn(filename):
      # [FUNCTION DOC]
      # HELPER: nested callback. Returns a single-file text line dataset for the given filename
      # using the compression type, buffer size and name of the enclosing constructor.
      return _TextLineDataset(
          filename, compression_type, buffer_size, name=name)

    self._impl = _create_dataset_reader(
        creator_fn, filenames, num_parallel_reads, name=name)
    variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access

    super(TextLineDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec: a scalar string tensor spec.
    return tensor_spec.TensorSpec([], dtypes.string)


@tf_export(v1=["data.TextLineDataset"])
class TextLineDatasetV1(dataset_ops.DatasetV1Adapter):
  # [CLASS DOC]
  # Legacy (TF1 API) adapter that wraps the TF2 text line dataset and exposes it through the
  # version-1 dataset interface. Forwards the filenames attribute to the wrapped dataset.
  """A `Dataset` comprising lines from one or more text files."""

  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               num_parallel_reads=None,
               name=None):
    # [FUNCTION DOC]
    # Takes the same arguments as the TF2 text line dataset (filenames, compression type, buffer
    # size, number of parallel reads, name), builds that dataset and wraps it in the adapter.
    wrapped = TextLineDatasetV2(filenames, compression_type, buffer_size,
                                num_parallel_reads, name)
    super(TextLineDatasetV1, self).__init__(wrapped)

  __init__.__doc__ = TextLineDatasetV2.__init__.__doc__

  @property
  def _filenames(self):
    # [FUNCTION DOC]
    # HELPER: property forwarding to the filenames dataset of the wrapped TF2 dataset.
    return self._dataset._filenames  # pylint: disable=protected-access

  @_filenames.setter
  def _filenames(self, value):
    # [FUNCTION DOC]
    # HELPER: setter replacing the filenames dataset stored on the wrapped TF2 dataset.
    self._dataset._filenames = value  # pylint: disable=protected-access


class _TFRecordDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # HELPER: low-level source used by the public TFRecord dataset classes. Dataset of raw
  # serialized records (scalar strings) from the given filenames tensor. Holds filenames,
  # compression type and buffer size tensors.
  """A `Dataset` comprising records from one or more TFRecord files."""

  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               name=None):
    # [FUNCTION DOC]
    # Inputs: filenames (string tensor), compression_type (optional: empty, ZLIB or GZIP),
    # buffer_size (optional byte count; default is a sentinel meaning the kernel chooses),
    # name (optional operation name).
    # Converts optional parameters to tensors, creates the TFRecord op and calls the parent
    # constructor.
    """Creates a `TFRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes in the read buffer. 0 means no buffering.
      name: (Optional.) A name for the tf.data operation.
    """
    self._filenames = filenames
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size",
        buffer_size,
        argument_default=_DEFAULT_TF_RECORD_BUFFER_SIZE_BYTES)
    self._name = name

    variant_tensor = gen_dataset_ops.tf_record_dataset(
        self._filenames, self._compression_type, self._buffer_size,
        metadata=self._metadata.SerializeToString())
    super(_TFRecordDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec: a scalar string tensor spec holding one record.
    return tensor_spec.TensorSpec([], dtypes.string)


class ParallelInterleaveDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # Legacy parallel interleave dataset: maps a function (returning datasets) over the input
  # elements and interleaves the results with the given cycle length. Used here for reading
  # several files in parallel with a fixed number of readers. Holds the wrapped map function,
  # element spec, cycle and block lengths, buffer settings and the determinism setting.
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self,
               input_dataset,
               map_func,
               cycle_length,
               block_length,
               sloppy,
               buffer_output_elements,
               prefetch_input_elements,
               name=None):
    # [FUNCTION DOC]
    # Inputs: input_dataset, map_func (callback mapping an element to a dataset), cycle_length
    # (number of inputs processed concurrently), block_length (consecutive elements taken from
    # each), sloppy (true allows nondeterministic order, false forces determinism, nothing uses the
    # default), buffer_output_elements and prefetch_input_elements (optional; default to twice the
    # block length and twice the cycle length), name (optional).
    # Wraps the map function; raises a type error when it does not return a dataset. Creates the
    # legacy parallel interleave op and calls the parent constructor.
    """See `tf.data.experimental.parallel_interleave()` for details."""
    self._input_dataset = input_dataset
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):
      raise TypeError(
          "The `map_func` argument must return a `Dataset` object. Got "
          f"{_get_type(self._map_func.output_structure)!r}.")
    self._element_spec = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")
    self._buffer_output_elements = convert.optional_param_to_tensor(
        "buffer_output_elements",
        buffer_output_elements,
        argument_default=2 * block_length)
    self._prefetch_input_elements = convert.optional_param_to_tensor(
        "prefetch_input_elements",
        prefetch_input_elements,
        argument_default=2 * cycle_length)
    if sloppy is None:
      self._deterministic = "default"
    elif sloppy:
      self._deterministic = "false"
    else:
      self._deterministic = "true"
    self._name = name

    variant_tensor = ged_ops.legacy_parallel_interleave_dataset_v2(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        self._cycle_length,
        self._block_length,
        self._buffer_output_elements,
        self._prefetch_input_elements,
        f=self._map_func.function,
        deterministic=self._deterministic,
        **self._common_args)
    super(ParallelInterleaveDataset, self).__init__(input_dataset,
                                                    variant_tensor)

  def _functions(self):
    # [FUNCTION DOC]
    # Returns a list holding the wrapped map function, used for function tracking.
    return [self._map_func]

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec of the datasets produced by the map function.
    return self._element_spec

  def _transformation_name(self):
    # [FUNCTION DOC]
    # Returns the display name of this transformation used in error messages.
    return "tf.data.experimental.parallel_interleave()"


@tf_export("data.TFRecordDataset", v1=[])
class TFRecordDatasetV2(dataset_ops.DatasetV2):
  # [CLASS DOC]
  # Public dataset (TF2 data API) whose elements are the raw serialized records of one or more
  # TFRecord files. Holds the filenames dataset, compression type, buffer size, parallel read
  # count and an internal implementation dataset doing the reading.
  """A `Dataset` comprising records from one or more TFRecord files.

  This dataset loads TFRecords from the files as bytes, exactly as they were
  written.`TFRecordDataset` does not do any parsing or decoding on its own.
  Parsing and decoding can be done by applying `Dataset.map` transformations
  after the `TFRecordDataset`.

  A minimal example is given below:

  >>> import tempfile
  >>> example_path = os.path.join(tempfile.gettempdir(), "example.tfrecords")
  >>> np.random.seed(0)

  >>> # Write the records to a file.
  ... with tf.io.TFRecordWriter(example_path) as file_writer:
  ...   for _ in range(4):
  ...     x, y = np.random.random(), np.random.random()
  ...
  ...     record_bytes = tf.train.Example(features=tf.train.Features(feature={
  ...         "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
  ...         "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
  ...     })).SerializeToString()
  ...     file_writer.write(record_bytes)

  >>> # Read the data back out.
  >>> def decode_fn(record_bytes):
  ...   return tf.io.parse_single_example(
  ...       # Data
  ...       record_bytes,
  ...
  ...       # Schema
  ...       {"x": tf.io.FixedLenFeature([], dtype=tf.float32),
  ...        "y": tf.io.FixedLenFeature([], dtype=tf.float32)}
  ...   )

  >>> for batch in tf.data.TFRecordDataset([example_path]).map(decode_fn):
  ...   print("x = {x:.4f},  y = {y:.4f}".format(**batch))
  x = 0.5488,  y = 0.7152
  x = 0.6028,  y = 0.5449
  x = 0.4237,  y = 0.6459
  x = 0.4376,  y = 0.8918
  """

  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               num_parallel_reads=None,
               name=None):
    # [FUNCTION DOC]
    # Inputs: filenames (string tensor, list, or dataset of string scalars), compression_type
    # (optional: empty, ZLIB or GZIP), buffer_size (optional byte count), num_parallel_reads
    # (optional number of files read in parallel; nothing means sequential), name (optional).
    # Validates or builds the filenames dataset (type error for non-string filenames), builds the
    # internal reader dataset with a callback creating single-file TFRecord datasets, and calls
    # the parent constructor.
    """Creates a `TFRecordDataset` to read one or more TFRecord files.

    Each element of the dataset will contain a single TFRecord.

    Args:
      filenames: A `tf.string` tensor or `tf.data.Dataset` containing one or
        more filenames.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes in the read buffer. If your input pipeline is I/O bottlenecked,
        consider setting this parameter to a value 1-100 MBs. If `None`, a
        sensible default for both local and remote file systems is used.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. If greater than one, the records of
        files read in parallel are outputted in an interleaved order. If your
        input pipeline is I/O bottlenecked, consider setting this parameter to a
        value greater than one to parallelize the I/O. If `None`, files will be
        read sequentially.
      name: (Optional.) A name for the tf.data operation.

    Raises:
      TypeError: If any argument does not have the expected type.
      ValueError: If any argument does not have the expected shape.
    """
    filenames = _create_or_validate_filenames_dataset(filenames, name=name)

    self._filenames = filenames
    self._compression_type = compression_type
    self._buffer_size = buffer_size
    self._num_parallel_reads = num_parallel_reads

    def creator_fn(filename):
      # [FUNCTION DOC]
      # HELPER: nested callback returning a single-file TFRecord dataset for the given filename,
      # using the compression type, buffer size and name of the enclosing constructor.
      return _TFRecordDataset(
          filename, compression_type, buffer_size, name=name)

    self._impl = _create_dataset_reader(
        creator_fn, filenames, num_parallel_reads, name=name)
    variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
    super(TFRecordDatasetV2, self).__init__(variant_tensor)

  def _inputs(self):
    # [FUNCTION DOC]
    # Returns the input datasets of the internal implementation dataset (graph traversal helper).
    return self._impl._inputs()  # pylint: disable=protected-access

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec: a scalar string tensor spec holding one record.
    return tensor_spec.TensorSpec([], dtypes.string)


@tf_export(v1=["data.TFRecordDataset"])
class TFRecordDatasetV1(dataset_ops.DatasetV1Adapter):
  # [CLASS DOC]
  # Legacy (TF1 API) adapter wrapping the TF2 TFRecord dataset behind the version-1 dataset
  # interface. Forwards the filenames attribute to the wrapped dataset.
  """A `Dataset` comprising records from one or more TFRecord files."""

  def __init__(self,
               filenames,
               compression_type=None,
               buffer_size=None,
               num_parallel_reads=None,
               name=None):
    # [FUNCTION DOC]
    # Takes the same arguments as the TF2 TFRecord dataset, builds that dataset and wraps it in
    # the version-1 adapter.
    wrapped = TFRecordDatasetV2(
        filenames, compression_type, buffer_size, num_parallel_reads, name=name)
    super(TFRecordDatasetV1, self).__init__(wrapped)

  __init__.__doc__ = TFRecordDatasetV2.__init__.__doc__

  @property
  def _filenames(self):
    # [FUNCTION DOC]
    # HELPER: property forwarding to the filenames dataset of the wrapped TF2 dataset.
    return self._dataset._filenames  # pylint: disable=protected-access

  @_filenames.setter
  def _filenames(self, value):
    # [FUNCTION DOC]
    # HELPER: setter replacing the filenames dataset stored on the wrapped TF2 dataset.
    self._dataset._filenames = value  # pylint: disable=protected-access


class _FixedLengthRecordDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # HELPER: low-level source used by the public fixed-length record dataset classes. Dataset of
  # fixed-size binary records (scalar strings) from the given filenames tensor, skipping optional
  # header and footer bytes per file. Holds the corresponding tensors.
  """A `Dataset` of fixed-length records from one or more binary files."""

  def __init__(self,
               filenames,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               buffer_size=None,
               compression_type=None,
               name=None):
    # [FUNCTION DOC]
    # Inputs: filenames (string tensor), record_bytes (required record size in bytes),
    # header_bytes and footer_bytes (optional byte counts skipped at file start and end),
    # buffer_size (optional read buffer bytes; default 256 kilobytes), compression_type (optional:
    # empty, ZLIB or GZIP), name (optional).
    # Converts parameters to tensors, creates the fixed-length record op and calls the parent
    # constructor.
    """Creates a `FixedLengthRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      record_bytes: A `tf.int64` scalar representing the number of bytes in each
        record.
      header_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to skip at the start of a file.
      footer_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to ignore at the end of a file.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes to buffer when reading.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      name: (Optional.) A name for the tf.data operation.
    """
    self._filenames = filenames
    self._record_bytes = ops.convert_to_tensor(
        record_bytes, dtype=dtypes.int64, name="record_bytes")
    self._header_bytes = convert.optional_param_to_tensor(
        "header_bytes", header_bytes)
    self._footer_bytes = convert.optional_param_to_tensor(
        "footer_bytes", footer_bytes)
    self._buffer_size = convert.optional_param_to_tensor(
        "buffer_size", buffer_size, _DEFAULT_READER_BUFFER_SIZE_BYTES)
    self._compression_type = convert.optional_param_to_tensor(
        "compression_type",
        compression_type,
        argument_default="",
        argument_dtype=dtypes.string)
    self._name = name

    variant_tensor = gen_dataset_ops.fixed_length_record_dataset_v2(
        self._filenames,
        self._header_bytes,
        self._record_bytes,
        self._footer_bytes,
        self._buffer_size,
        self._compression_type,
        metadata=self._metadata.SerializeToString())
    super(_FixedLengthRecordDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec: a scalar string tensor spec holding one record.
    return tensor_spec.TensorSpec([], dtypes.string)


@tf_export("data.FixedLengthRecordDataset", v1=[])
class FixedLengthRecordDatasetV2(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # Public dataset (TF2 data API) of fixed-length binary records from one or more files, each
  # optionally having a fixed-size header and footer that are skipped. Holds the filenames
  # dataset, record, header and footer sizes, buffer size, compression type and an internal
  # implementation dataset doing the reading.
  """A `Dataset` of fixed-length records from one or more binary files.

  The `tf.data.FixedLengthRecordDataset` reads fixed length records from binary
  files and creates a dataset where each record becomes an element of the
  dataset. The binary files can have a fixed length header and a fixed length
  footer, which will both be skipped.

  For example, suppose we have 2 files "fixed_length0.bin" and
  "fixed_length1.bin" with the following content:

  >>> with open('/tmp/fixed_length0.bin', 'wb') as f:
  ...   f.write(b'HEADER012345FOOTER')
  >>> with open('/tmp/fixed_length1.bin', 'wb') as f:
  ...   f.write(b'HEADER6789abFOOTER')

  We can construct a `FixedLengthRecordDataset` from them as follows:

  >>> dataset1 = tf.data.FixedLengthRecordDataset(
  ...     filenames=['/tmp/fixed_length0.bin', '/tmp/fixed_length1.bin'],
  ...     record_bytes=2, header_bytes=6, footer_bytes=6)

  The elements of the dataset are:

  >>> for element in dataset1.as_numpy_iterator():
  ...   print(element)
  b'01'
  b'23'
  b'45'
  b'67'
  b'89'
  b'ab'
  """

  def __init__(self,
               filenames,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               buffer_size=None,
               compression_type=None,
               num_parallel_reads=None,
               name=None):
    # [FUNCTION DOC]
    # Inputs: filenames (string tensor, list, or dataset of string scalars), record_bytes
    # (required record size in bytes), header_bytes, footer_bytes (optional bytes skipped per
    # file), buffer_size (optional), compression_type (optional), num_parallel_reads (optional
    # number of files read in parallel; nothing means sequential), name (optional).
    # Validates or builds the filenames dataset (type error for non-string filenames), builds the
    # internal reader dataset using a callback creating single-file datasets, and calls the parent
    # constructor.
    """Creates a `FixedLengthRecordDataset`.

    Args:
      filenames: A `tf.string` tensor or `tf.data.Dataset` containing one or
        more filenames.
      record_bytes: A `tf.int64` scalar representing the number of bytes in each
        record.
      header_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to skip at the start of a file.
      footer_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to ignore at the end of a file.
      buffer_size: (Optional.) A `tf.int64` scalar representing the number of
        bytes to buffer when reading.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. If greater than one, the records of
        files read in parallel are outputted in an interleaved order. If your
        input pipeline is I/O bottlenecked, consider setting this parameter to a
        value greater than one to parallelize the I/O. If `None`, files will be
        read sequentially.
      name: (Optional.) A name for the tf.data operation.
    """
    filenames = _create_or_validate_filenames_dataset(filenames, name=name)

    self._filenames = filenames
    self._record_bytes = record_bytes
    self._header_bytes = header_bytes
    self._footer_bytes = footer_bytes
    self._buffer_size = buffer_size
    self._compression_type = compression_type

    def creator_fn(filename):
      # [FUNCTION DOC]
      # HELPER: nested callback returning a single-file fixed-length record dataset for the given
      # filename, using the record, header, footer, buffer, compression settings and name of the
      # enclosing constructor.
      return _FixedLengthRecordDataset(
          filename,
          record_bytes,
          header_bytes,
          footer_bytes,
          buffer_size,
          compression_type,
          name=name)

    self._impl = _create_dataset_reader(
        creator_fn, filenames, num_parallel_reads, name=name)
    variant_tensor = self._impl._variant_tensor  # pylint: disable=protected-access
    super(FixedLengthRecordDatasetV2, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec: a scalar string tensor spec holding one record.
    return tensor_spec.TensorSpec([], dtypes.string)


@tf_export(v1=["data.FixedLengthRecordDataset"])
class FixedLengthRecordDatasetV1(dataset_ops.DatasetV1Adapter):
  # [CLASS DOC]
  # Legacy (TF1 API) adapter wrapping the TF2 fixed-length record dataset behind the version-1
  # dataset interface. Forwards the filenames attribute to the wrapped dataset.
  """A `Dataset` of fixed-length records from one or more binary files."""

  def __init__(self,
               filenames,
               record_bytes,
               header_bytes=None,
               footer_bytes=None,
               buffer_size=None,
               compression_type=None,
               num_parallel_reads=None,
               name=None):
    # [FUNCTION DOC]
    # Takes the same arguments as the TF2 fixed-length record dataset, builds that dataset and
    # wraps it in the version-1 adapter.
    wrapped = FixedLengthRecordDatasetV2(
        filenames,
        record_bytes,
        header_bytes,
        footer_bytes,
        buffer_size,
        compression_type,
        num_parallel_reads,
        name=name)
    super(FixedLengthRecordDatasetV1, self).__init__(wrapped)

  __init__.__doc__ = FixedLengthRecordDatasetV2.__init__.__doc__

  @property
  def _filenames(self):
    # [FUNCTION DOC]
    # HELPER: property forwarding to the filenames dataset of the wrapped TF2 dataset.
    return self._dataset._filenames  # pylint: disable=protected-access

  @_filenames.setter
  def _filenames(self, value):
    # [FUNCTION DOC]
    # HELPER: setter replacing the filenames dataset stored on the wrapped TF2 dataset.
    self._dataset._filenames = value  # pylint: disable=protected-access


if tf2.enabled():
  FixedLengthRecordDataset = FixedLengthRecordDatasetV2
  TFRecordDataset = TFRecordDatasetV2
  TextLineDataset = TextLineDatasetV2
else:
  FixedLengthRecordDataset = FixedLengthRecordDatasetV1
  TFRecordDataset = TFRecordDatasetV1
  TextLineDataset = TextLineDatasetV1


def _tf2_callback():
  # [FUNCTION DOC]
  # HELPER: callback registered with the v2 compatibility layer. Rebinds the module-level public
  # names for the fixed-length record, TFRecord and text line datasets to the TF2 classes when
  # TF2 behavior is enabled, otherwise to the version-1 adapter classes. Mutates module globals.
  global FixedLengthRecordDataset, TFRecordDataset, TextLineDataset
  if tf2.enabled():
    FixedLengthRecordDataset = FixedLengthRecordDatasetV2
    TFRecordDataset = TFRecordDatasetV2
    TextLineDataset = TextLineDatasetV2
  else:
    FixedLengthRecordDataset = FixedLengthRecordDatasetV1
    TFRecordDataset = TFRecordDatasetV1
    TextLineDataset = TextLineDatasetV1


v2_compat.register_data_v2_callback(_tf2_callback)
