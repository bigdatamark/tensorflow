# [MODULE DOC]
# HELPER: internal implementation module behind the public Dataset load operation in dataset_ops.
# Implements loading of a dataset previously saved as a tf.data snapshot. Handles both the
# regular (non-distributed) snapshot format, read through a LoadDataset op, and the distributed
# snapshot format, read by listing chunk files and reading each chunk through a snapshot-chunk
# op, then combining the per-chunk datasets with a user or default reader function.
# Exposes the private entry point _load plus dataset source classes for loading, chunk reading
# and chunk listing. Optionally waits and retries until the snapshot appears.
# No import-time side effects other than defining a module constant for the retry interval.
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
"""Implementation of LoadDataset in Python."""
import multiprocessing
import os
import time
from typing import Any, Callable, Optional, Union

from absl import logging

from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.data.experimental.service import _pywrap_snapshot_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
# TODO(b/238903802): Use TypeSpec serialization methods directly.
from tensorflow.python.saved_model import nested_structure_coder

# For distributed snapshot load V2, retries loading after this time, if the
# snapshot is not ready yet.
_RETRY_INTERVAL_SEC = 5


def _load(  # pylint: disable=unused-private-name
    path: str,
    element_spec: Any,
    compression: Optional[str],
    reader_func: Optional[Callable[[dataset_ops.Dataset], dataset_ops.Dataset]],
    wait: bool,
) -> dataset_ops.Dataset:
  # [FUNCTION DOC]
  # HELPER: backs the public Dataset load method.
  # Loads a dataset from a snapshot directory. Inputs: path of the snapshot, element_spec
  # (structure description of elements, may be nothing to infer it), compression name (may be
  # nothing), reader_func (callback mapping a dataset of datasets to a dataset, may be nothing)
  # and wait (boolean). If wait is true, delegates to the retrying loader and returns its result.
  # Otherwise a default reader function is used when none is given: interleave all inputs with
  # cycle length equal to the CPU count and automatic parallelism. If distributed snapshot
  # metadata is found, the snapshot is validated and loaded as a distributed snapshot. Else the
  # element spec is read from disk when not supplied and a regular load dataset is returned.
  # Raises errors from validation or from a missing element spec file. Returns a Dataset.
  """Loads dataset from tf.data snapshot."""

  if wait:
    return _load_with_retry(path, element_spec, compression, reader_func)

  if reader_func is None:
    reader_func = lambda datasets: datasets.interleave(  # pylint:disable=g-long-lambda
        lambda x: x,
        cycle_length=multiprocessing.cpu_count(),
        num_parallel_calls=dataset_ops.AUTOTUNE)

  distributed_snapshot_metadata = _load_distributed_snapshot_metadata(path)
  if distributed_snapshot_metadata:
    _validate_snapshot(
        path, distributed_snapshot_metadata, element_spec, compression)
    return _load_distributed_snapshot(
        path, distributed_snapshot_metadata, reader_func)

  if element_spec is None:
    element_spec = _load_element_spec(path)
  return _LoadDataset(path, element_spec, compression, reader_func)


def _load_with_retry(  # pylint: disable=unused-private-name
    path: str,
    element_spec: Any = None,
    compression: Optional[str] = None,
    reader_func: Optional[
        Callable[[dataset_ops.Dataset], dataset_ops.Dataset]] = None,
) -> dataset_ops.Dataset:
  # [FUNCTION DOC]
  # HELPER: implements the wait option of the snapshot load.
  # Repeatedly calls the public Dataset load with waiting disabled, passing along path,
  # element_spec, compression and reader_func (all but path optional). On success logs and
  # returns the dataset. On a not-found error or missing-file error it logs, sleeps for the
  # retry interval (5 seconds) and tries again. Loops forever until the snapshot appears; other
  # errors propagate.
  """Tries loading the snapshot. Retries if not found."""

  while True:
    try:
      dataset = dataset_ops.Dataset.load(
          path=path,
          element_spec=element_spec,
          compression=compression,
          reader_func=reader_func,
          wait=False)
      logging.info("Load tf.data snapshot at %s.", path)
      return dataset
    except (errors.NotFoundError, FileNotFoundError):
      logging.info(
          "Could not find tf.data snapshot at %s. Will wait and retry.", path)
      time.sleep(_RETRY_INTERVAL_SEC)


def _load_distributed_snapshot_metadata(
    path: str,
) -> Optional[snapshot_pb2.DistributedSnapshotMetadata]:
  # [FUNCTION DOC]
  # HELPER: detects whether a snapshot directory is a distributed snapshot.
  # Input: base path of the snapshot. Looks up the metadata file location for that path; if the
  # file does not exist returns nothing. Otherwise parses it as a text-format metadata message
  # and returns it. If the file is missing at read time or cannot be parsed or decoded, returns
  # nothing rather than raising.
  """Reads the distributed snapshot metadata.

  Args:
    path: Base path of the snapshot.

  Returns:
    DistributedSnapshotMetadata if the snapshot is a distributed snapshot.
    Returns None if it is a non-distributed snapshot.
  """
  metadata_file = _pywrap_snapshot_utils.TF_DATA_SnapshotMetadataFilePath(path)
  if not gfile.Exists(metadata_file):
    return None

  try:
    with gfile.GFile(metadata_file, "r") as f:
      return text_format.ParseLines(
          f, snapshot_pb2.DistributedSnapshotMetadata())
  except (
      errors.NotFoundError,
      text_format.ParseError,
      message.DecodeError,
      UnicodeDecodeError):
    return None


def _load_distributed_snapshot(
    path: str,
    metadata: snapshot_pb2.DistributedSnapshotMetadata,
    reader_func: Callable[[dataset_ops.Dataset], dataset_ops.Dataset],
) -> dataset_ops.Dataset:
  # [FUNCTION DOC]
  # HELPER: loads the distributed snapshot variant for the snapshot load.
  # Inputs: snapshot path, parsed distributed snapshot metadata, and reader_func callback.
  # Builds a dataset listing the chunk files, maps each chunk file to a chunk-reading dataset
  # using the element spec decoded from the metadata and the metadata's compression, and
  # returns the result of applying reader_func to that dataset of datasets.
  """Loads a distributed snapshot."""

  dataset = _ListSnapshotChunksDataset(path)
  dataset = dataset.map(
      lambda chunk_file: _SnapshotChunkDataset(  # pylint:disable=g-long-lambda
          chunk_file,
          element_spec=_parse_element_spec(metadata.element_spec),
          compression=metadata.compression))
  return reader_func(dataset)


def _load_element_spec(path: str) -> Any:
  # [FUNCTION DOC]
  # HELPER: reads the saved element structure for the snapshot load.
  # Input: base snapshot path. Reads the dataset spec file in that directory and decodes it into
  # an element spec, which is returned. Raises a not-found error if the file does not exist or
  # if its contents cannot be decoded as an encodable structure.
  """Loads the dataset element spec.

  Args:
    path: Base path of the snapshot.

  Returns:
    Dataset element_spec.

  Raises:
    NotFoundError if the element spec file does not exist or cannot be decoded.
  """
  dataset_spec_filename = os.path.join(path, dataset_ops.DATASET_SPEC_FILENAME)
  if not gfile.Exists(dataset_spec_filename):
    raise errors.NotFoundError(
        node_def=None, op=None,
        message="tf.data snapshot element_spec file not found: "
                f"{dataset_spec_filename}.")

  with gfile.GFile(dataset_spec_filename, "rb") as f:
    encoded_spec = f.read()
  try:
    return _parse_element_spec(encoded_spec)
  except nested_structure_coder.NotEncodableError as e:
    raise errors.NotFoundError(
        node_def=None, op=None,
        message="tf.data snapshot element_spec file not found or invalid: "
                f"{dataset_spec_filename}.") from e


def _parse_element_spec(encoded_element_spec: Union[bytes, str]) -> Any:
  # [FUNCTION DOC]
  # HELPER: decodes a serialized element spec for the snapshot load.
  # Input: serialized structured-value message as bytes or string. Parses it into a structured
  # value message and decodes that into the element spec structure, which is returned. Parse or
  # decode failures propagate.
  struct_pb = nested_structure_coder.struct_pb2.StructuredValue()
  struct_pb.ParseFromString(encoded_element_spec)
  return nested_structure_coder.decode_proto(struct_pb)


class _LoadDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # HELPER: dataset source used by the snapshot load for non-distributed snapshots.
  # Wraps the load-dataset op. Holds the path, element spec, compression and the traced
  # reader function. Elements have the given element spec.
  """A dataset that loads previously saved dataset."""

  def __init__(
      self,
      path: str,
      element_spec: Any,
      compression: str,
      reader_func: Callable[[dataset_ops.Dataset], dataset_ops.Dataset]):
    # [FUNCTION DOC]
    # Stores path, element spec and compression; traces reader_func into a graph function whose
    # input is a dataset of datasets of the element spec; creates the load-dataset op with the
    # path, the function, its captured inputs and compression; then initializes the base
    # dataset source with the resulting variant tensor. Adds graph ops.
    self._path = path
    self._element_spec = element_spec
    self._compression = compression
    self._reader_func = structured_function.StructuredFunctionWrapper(
        reader_func,
        "load()",
        # Dataset of datasets of input elements
        input_structure=dataset_ops.DatasetSpec(
            dataset_ops.DatasetSpec(self._element_spec)))

    variant_tensor = ged_ops.load_dataset(
        path,
        reader_func_other_args=self._reader_func.function.captured_inputs,
        compression=compression,
        reader_func=self._reader_func.function,
        **self._flat_structure)
    super().__init__(variant_tensor)

  @property
  def element_spec(self) -> Any:
    # [FUNCTION DOC]
    # Returns the element structure description given at construction.
    return self._element_spec


class _SnapshotChunkDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # HELPER: dataset source used by the distributed snapshot load.
  # Reads all elements of a single chunk file of a distributed snapshot, with a given element
  # spec and compression.
  """A dataset for one chunk file from a tf.data distributed snapshot."""

  def __init__(self, chunk_file: str, element_spec: Any, compression: str):
    # [FUNCTION DOC]
    # Inputs: chunk file path, element spec, compression name. Stores the chunk file and element
    # spec, creates the snapshot-chunk dataset op and initializes the base dataset source.
    self._chunk_file = chunk_file
    self._element_spec = element_spec
    variant_tensor = ged_ops.snapshot_chunk_dataset(
        chunk_file,
        compression=compression,
        **self._flat_structure)
    super().__init__(variant_tensor)

  @property
  def element_spec(self) -> Any:
    # [FUNCTION DOC]
    # Returns the element structure description given at construction.
    return self._element_spec


class _ListSnapshotChunksDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # HELPER: dataset source used by the distributed snapshot load.
  # Produces scalar strings, each the path of a chunk file currently available in the snapshot.
  """A dataset for listing snapshot chunk files.

  It supports listing partially written snapshots. When a snapshot is being
  written, it returns the currently available chunk files.
  """

  def __init__(self, snapshot_path: str):
    # [FUNCTION DOC]
    # Input: snapshot directory path. Stores it, creates the list-snapshot-chunks dataset op and
    # initializes the base dataset source with the resulting variant tensor.
    self._snapshot_path = snapshot_path
    variant_tensor = ged_ops.list_snapshot_chunks_dataset(
        snapshot_path, **self._flat_structure)
    super().__init__(variant_tensor)

  @property
  def element_spec(self) -> tensor_spec.TensorSpec:
    # [FUNCTION DOC]
    # Returns a fixed spec: a scalar string tensor, one per chunk file path.
    return tensor_spec.TensorSpec([], dtypes.string)


def _validate_snapshot(
    path: str,
    metadata: snapshot_pb2.DistributedSnapshotMetadata,
    element_spec: Any,
    compression: str) -> None:
  # [FUNCTION DOC]
  # HELPER: sanity check used before loading a distributed snapshot.
  # Inputs: snapshot path, parsed metadata, user-given element spec (may be nothing) and
  # compression (may be nothing). Raises a value error if an error file exists in the snapshot
  # (the save job failed, message includes the file content), if a given element spec differs
  # from the one in the metadata, or if a given compression differs from the metadata's.
  # Returns nothing.
  """Validates a tf.data distributed snapshot.

  Args:
    path: Root path of the distributed snapshot.
    metadata: The DistributedSnapshotMetadata of the snapshot.
    element_spec: Dataset element_spec.
    compression: Compression method used for saving.

  Raises:
    ValueError if the snapshot is invalid.
  """

  error_file = _pywrap_snapshot_utils.TF_DATA_SnapshotErrorFilePath(path)
  if gfile.Exists(error_file):
    with gfile.GFile(error_file, "r") as f:
      raise ValueError(
          f"Failed to load tf.data snapshot at {path}. The save job failed to "
          f"write it. Status: {f.read()}")

  snapshot_element_spec = _parse_element_spec(metadata.element_spec)
  if element_spec and element_spec != snapshot_element_spec:
    raise ValueError(
        f"Failed to load tf.data snapshot at {path}. User specified "
        f"element_spec {element_spec}, but the actual element_spec is "
        f"{snapshot_element_spec}.")

  if compression and compression != metadata.compression:
    raise ValueError(
        f"Failed to load tf.data snapshot at {path}. User specified "
        f"compression {compression}, but the actual compression is "
        f"{metadata.compression}.")
