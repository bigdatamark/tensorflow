# [MODULE DOC]
# Cloud TPU data loading helpers. Because Cloud TPUs cannot read files local to the VM, this
# module offers a function that builds a dataset which reads files on a coordinator job and
# streams the elements to the worker job through remote function calls. Defines private reader
# builders for text lines and records, and a mapping from file type names to those builders.
# No import-time side effects beyond building that mapping.
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
# ======================================
"""Library of Cloud TPU helper functions for data loading."""

from typing import Callable, Optional, Text, Union

from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.types import data as data_types


def _TextLineDataset(filename: Text) -> dataset_ops.Dataset:
  # [FUNCTION DOC]
  # HELPER: file-type reader used by the streaming files dataset. Input: filename, a string.
  # Returns a text-line dataset for that file with an 8 MiB read buffer. No other side effects.
  buffer_size = 8 * 1024 * 1024  # 8 MiB per file
  dataset = readers.TextLineDataset(filename, buffer_size=buffer_size)
  return dataset


def _TFRecordDataset(filename: Text) -> dataset_ops.Dataset:
  # [FUNCTION DOC]
  # HELPER: file-type reader used by the streaming files dataset. Input: filename, a string.
  # Returns a record-file dataset for that file with an 8 MiB read buffer. No other side effects.
  buffer_size = 8 * 1024 * 1024  # 8 MiB per file
  dataset = readers.TFRecordDataset(filename, buffer_size=buffer_size)
  return dataset


_FILETYPE_MAP = {
    'tfrecord': _TFRecordDataset,
    'textline': _TextLineDataset,
    'text': _TextLineDataset,
}


def StreamingFilesDataset(
    files: Union[Text, dataset_ops.Dataset],
    filetype: Optional[Union[Text, Callable[[Text],
                                            dataset_ops.Dataset]]] = None,
    file_reader_job: Optional[Text] = None,
    worker_job: Optional[Text] = None,
    num_epochs: Optional[int] = None,
    filename_shuffle_buffer_size: Optional[Union[int, bool]] = None,
    num_parallel_reads: Optional[int] = None,
    batch_transfer_size: Optional[Union[int, bool]] = None,
    sloppy: bool = True) -> dataset_ops.Dataset:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says filetype is one of the record or text-line names, but the
  # accepted names also include a plain text alias. The output dataset is repeated forever on the
  # worker regardless of num_epochs (num_epochs only limits the inner file dataset), and the
  # docstring's mention of an out-of-range error when no files match is not enforced here.
  # Builds a dataset that reads files on the file reader job and streams elements to the worker
  # job. Inputs: files (glob string, or dataset of file names), filetype (name string, default
  # record files, or a function mapping a file name to a dataset), file_reader_job (default the
  # coordinator job, mapped to its first task device), worker_job (default worker),
  # num_epochs (repeat count of the inner dataset; nothing repeats forever),
  # filename_shuffle_buffer_size (default 4096; zero or false disables shuffling),
  # num_parallel_reads (default 8, also used for a falsy value), batch_transfer_size (default 256;
  # false disables batching), sloppy (default true; allow non-deterministic order).
  # Steps: on the reader device, list or accept files, optionally shuffle, interleave readers in
  # parallel, repeat, batch, prefetch, and expose a one-shot iterator handle. Then a legacy
  # function loads the next element from that iterator; a map function calls it remotely on the
  # reader job; on the worker device a range dataset is repeated, mapped (parallel if sloppy),
  # prefetched and unbatched if batching was used.
  # Returns the resulting dataset. Raises a value error for an unknown filetype name, a filetype
  # that is neither string nor callable, or files that are neither string nor dataset (and, when
  # run, for invalid source output types). Side effects: adds ops and a function to the graph.
  # Nested: LoadingFunc and MapFn.
  """StreamingFilesDataset constructs a dataset to stream from workers (GCE VM).

  Because Cloud TPUs are allocated over the network, a Cloud TPU cannot read
  files local to your GCE VM. In order to train using files stored on your local
  VM (e.g. on local SSD for extreme performance), use the StreamingFilesDataset
  helper to generate a dataset to feed your Cloud TPU with files from your GCE
  VM.

  The resulting dataset may return an OutOfRangeError if there are no files
  found as a result of the fileglob expansion.

  Note: StreamingFilesDataset assumes that the session is using a
  TPUClusterResolver and has therefore a worker and a coordinator job. File
  loading will be done on the coordinator job.

  Args:
    files: A string glob to match files, or a `tf.data.Dataset` generating file
      names.
    filetype: A string (one of 'tfrecord', or 'textline') or a single-argument
      TensorFlow function that when given a filename returns a dataset.
    file_reader_job: An optional string that corresponds to the job that should
      perform the file reads.
    worker_job: An optional string that corresponds to the job that should
      process the tensors (i.e. your GPU or TPU worker).
    num_epochs: The number of epochs through the training set that should be
      generated. By default, it will repeat infinitely.
    filename_shuffle_buffer_size: An optional integer whose value controls the
      shuffling of the file names. If you would like to read from the files in
      the same order, set to 0 or False.
    num_parallel_reads: An optional integer controlling the number of files to
      read from concurrently. (Set to 1 for no parallelism.)
    batch_transfer_size: An optional integer controlling the batching used to
      amortize the remote function invocation overhead. Set to a very large
      number to increase throughput. Set to a very small number to reduce memory
      consumption. Set to False to skip batching.
    sloppy: (Optional.) If `False`, read input data while maintaining a
      deterministic order. (This may have significant performance impacts.)
      sloppy defaults to: True.
  Returns:
    A `tf.data.Dataset` with an infinite stream of elements generated by a
    parallel interleaving of the set of files matched (or generated) by `files`
    with a type is the output of the dataset specified by `filetype`.

  Raises:
    ValueError: if any argument is not of the expected type.
  """
  if filetype is None:
    filetype = 'tfrecord'

  if isinstance(filetype, str):
    if filetype not in _FILETYPE_MAP:
      raise ValueError(
          f'Unexpected filetype. Received: {filetype}. Expected one of '
          f'{list(_FILETYPE_MAP.keys())}')
    reader_fn = _FILETYPE_MAP[filetype]
  elif callable(filetype):
    reader_fn = filetype
  else:
    raise ValueError(f'Argument `filetype` should be a string or a callable. '
                     f'Received: {filetype} of type {type(filetype)}.')

  file_reader_job = file_reader_job or 'coordinator'

  worker_job = worker_job or 'worker'

  if filename_shuffle_buffer_size is None:
    filename_shuffle_buffer_size = 4096

  num_parallel_reads = num_parallel_reads or 8

  if batch_transfer_size is None:
    batch_transfer_size = 256

  if file_reader_job == 'coordinator':
    file_reader_device = '/job:coordinator/task:0'
  else:
    file_reader_device = '/job:%s' % file_reader_job

  with ops.device(file_reader_device):
    if isinstance(files, str):
      source_dataset = dataset_ops.Dataset.list_files(files)
    elif isinstance(files, data_types.DatasetV2):
      source_dataset = files
    else:
      raise ValueError(
          'Argument `files` should be a string or a `tf.data.Dataset` '
          f'instance. Received: {files}')

    if filename_shuffle_buffer_size:
      source_dataset = source_dataset.shuffle(
          buffer_size=filename_shuffle_buffer_size)

    source_dataset = source_dataset.apply(
        interleave_ops.parallel_interleave(
            reader_fn, cycle_length=num_parallel_reads, sloppy=sloppy))

    source_dataset = source_dataset.repeat(num_epochs)

    if batch_transfer_size:
      source_dataset = source_dataset.batch(batch_transfer_size)

    source_dataset = source_dataset.prefetch(1)

    source_iterator = dataset_ops.make_one_shot_iterator(source_dataset)
    source_handle = source_iterator.string_handle()

  @function.Defun(dtypes.string)
  def LoadingFunc(h):
    # [FUNCTION DOC]
    # HELPER: legacy graph function (defined with a string input) executed on the reader job.
    # Input: h, a string iterator handle. Rebuilds the iterator from the handle using the source
    # dataset's types and shapes and returns the next element (or tuple of elements).
    remote_iterator = iterator_ops.Iterator.from_string_handle(
        h, dataset_ops.get_legacy_output_types(source_dataset),
        dataset_ops.get_legacy_output_shapes(source_dataset))
    return remote_iterator.get_next()

  def MapFn(unused_input):
    # [FUNCTION DOC]
    # HELPER: map function of the output dataset. Input: unused_input (ignored; it only drives
    # repetition). Determines the source dataset's output types (a single type becomes a list;
    # lists or tuples are used as is; anything else raises a value error), then remote-calls the
    # loading function on the reader job's CPU with the iterator handle. Returns the single
    # result if there is one output, otherwise the list of results.
    source_dataset_output_types = dataset_ops.get_legacy_output_types(
        source_dataset)
    if isinstance(source_dataset_output_types, dtypes.DType):
      output_types = [source_dataset_output_types]
    elif isinstance(source_dataset_output_types, (list, tuple)):
      output_types = source_dataset_output_types
    else:
      raise ValueError('Source dataset has invalid output types. Only '
                       'list/tuples or TensorFlow tensor types are accepted.')
    remote_calls = functional_ops.remote_call(
        args=[source_handle],
        Tout=output_types,
        f=LoadingFunc,
        target='/job:%s/replica:0/task:0/cpu:0' % file_reader_job)
    if len(remote_calls) == 1:
      return remote_calls[0]
    else:
      return remote_calls

  with ops.device('/job:%s' % worker_job):
    output_dataset = dataset_ops.Dataset.range(2).repeat().map(
        MapFn, num_parallel_calls=4 if sloppy else None)
    output_dataset = output_dataset.prefetch(1)

    if batch_transfer_size:
      # Undo the batching used during the transfer.
      output_dataset = output_dataset.unbatch().prefetch(1)

  return output_dataset
