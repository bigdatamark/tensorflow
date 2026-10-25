# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Checkpoint adapter that reshards TPU embedding V3 tables on restore. Provides parsing and
# formatting of shape-and-slice strings, routines that shard per-table tensors into the
# SparseCore stacked layout and unshard them back, two reshard callback classes (from
# unsharded checkpoints, and between different SparseCore layouts), and the adapter class that
# reads layouts from a checkpoint and picks callbacks. No import-time side effects beyond
# imports.
"""Checkpoint adapter for TPUEmbedding."""

import collections
import time
from typing import Mapping, Optional, Sequence

from absl import logging

from tensorflow.core.tpu.kernels import sparse_core_layout_pb2
from tensorflow.python.checkpoint import checkpoint_adapter
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_embedding_v3_utils
from tensorflow.python.trackable import base as trackable_base
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.util.protobuf import compare


def _parse_shard_info_str(
    spec: str,
) -> tuple[list[int], trackable_base.ShardInfo]:
  # [FUNCTION DOC]
  # HELPER: decodes checkpoint shape-and-slice strings for the resharding callbacks.
  # Input: spec, a text made of whitespace separated full-shape integers followed by a last
  # token of colon separated "offset,size" pairs, one per dimension.
  # Returns a pair: the list of full-shape integers, and a shard-info object holding the
  # per-dimension offsets and shard sizes. Raises a value error (from integer conversion) or
  # an index error if the text is malformed. No side effects.
  """Parses shape and shard_info string."""

  shape = [int(x) for x in spec.split()[:-1]]
  slices = spec.split()[-1].split(":")
  offset = [int(x.split(",")[0]) for x in slices]
  shard_shape = [int(x.split(",")[1]) for x in slices]
  return shape, trackable_base.ShardInfo(offset=offset, shape=shard_shape)


def _shard_info_str(shape, shard_info) -> str:
  # [FUNCTION DOC]
  # HELPER: inverse of the shape-and-slice parser, used to build restore slice specs.
  # Inputs: shape, a list of full-shape integers; shard_info, an object with per-dimension
  # offset and shape lists. Returns text with the full shape integers separated by spaces,
  # a space, then colon separated "offset,size" pairs. No side effects.
  """Created shape and shard_info string."""

  full_shape_str = " ".join("%d" % d for d in shape) + " "
  slice_spec = ":".join(
      "%d,%d" % (o, s) for o, s in zip(shard_info.offset, shard_info.shape)
  )
  return full_shape_str + slice_spec


def _shard_from_cpu_to_sc(
    feature_values: tensor.Tensor,
    shape_and_slice: str,
    to_shard_layout: Sequence[sparse_core_layout_pb2.SparseCoreTableLayout],
) -> tensor.Tensor:
  # [FUNCTION DOC]
  # HELPER: core conversion from unsharded per-table values to a SparseCore stacked shard.
  # KNOWN ISSUE: the offset validity check compares two lists with a lexicographic ordering
  # instead of comparing each dimension, so it only rejects some invalid offsets, and it
  # allows an offset equal to the full shape although the message says "less than".
  # Inputs: feature_values, indexable collection (one entry per table layout) of full
  # unsharded table tensors; shape_and_slice, a shape-and-slice text for the requested
  # slice of the stacked variable; to_shard_layout, the list of target table layouts that
  # make up the stacked table (all share the sparse core and partition counts).
  # Behavior: pads each table to its padded shape, then for every partition-sized row range
  # in the requested slice and every sparse core of that partition, takes each table's
  # strided rows (after applying the table's shard rotation) and concatenates them; the
  # per-partition results are concatenated along rows.
  # Returns the resulting tensor. Raises a value error if the offset is invalid or if the
  # slice row count is not a multiple of the rows in one partition. Logs at verbose level.
  """Shards the feature tables from CPU to SparseCore."""

  def pad_value(value, variable_shape, table_shape):
    # [FUNCTION DOC]
    # HELPER: pads a table with zero-style constant padding on the bottom and right.
    # Inputs: value, the table tensor; variable_shape, the padded target shape; table_shape,
    # the current shape. Returns the padded tensor (padding applied to rows and columns).
    return array_ops.pad(
        value,
        [
            [0, variable_shape[0] - table_shape[0]],
            [0, variable_shape[1] - table_shape[1]],
        ],
        "CONSTANT",
    )

  var_full_shape, shard_info = _parse_shard_info_str(shape_and_slice)
  if shard_info.offset > var_full_shape:
    raise ValueError(
        "Invalid shard offset: {}. Offset should be less than the full shape"
        " of the variable: {}".format(
            shard_info.offset,
            var_full_shape,
        )
    )
  num_sc_per_partition = (
      to_shard_layout[0].num_sparse_cores // to_shard_layout[0].num_partitions
  )

  total_rows_per_sc = to_shard_layout[0].total_rows_per_sparse_core_shard
  total_rows_per_partition = total_rows_per_sc * num_sc_per_partition
  full_values = {}
  if (shard_info.shape[0] % total_rows_per_partition) != 0:
    raise ValueError(
        "Invalid shard shape: {}. Number of rows in input shard slice should"
        " be multiple of number of rows in a partition({})".format(
            shard_info.shape,
            total_rows_per_partition,
        )
    )
  # From the shard info, get the row offsets corresponding to the slice
  # being looked up.
  required_shard_offsets = range(
      shard_info.offset[0],
      shard_info.offset[0] + shard_info.shape[0],
      total_rows_per_partition,
  )
  output_shards = []
  for required_shard_offset in required_shard_offsets:
    sharded_tensors = []
    for i in range(num_sc_per_partition):
      shard_idx = (required_shard_offset // total_rows_per_sc) + i
      for table_idx, layout in enumerate(to_shard_layout):
        if table_idx not in full_values:
          full_values[table_idx] = pad_value(
              feature_values[table_idx],
              layout.unsharded_padded_shape,
              layout.unsharded_shape,
          )

        table_value = full_values[table_idx]
        # Apply rotation to get this table's shard index
        table_shard_offset = (
            shard_idx
            + (layout.num_sparse_cores - layout.sparse_core_shard_rotation)
        ) % layout.num_sparse_cores
        sharded_tensors.append(
            table_value[
                table_shard_offset :: layout.num_sparse_cores,
                :,
            ]
        )
    output_shards.append(array_ops.concat(sharded_tensors, axis=0))
    logging.vlog(
        1,
        "_shard_from_cpu_to_sc: last output_shards.shape: %s",
        output_shards[-1].shape,
    )
  return array_ops.concat(output_shards, axis=0)


def _unshard_from_sc_to_cpu(
    stacked_table: tensor.Tensor,
    from_shard_layouts: Sequence[sparse_core_layout_pb2.SparseCoreTableLayout],
) -> Sequence[tensor.Tensor]:
  # [FUNCTION DOC]
  # HELPER: reverses the SparseCore stacking and sharding for restoring from a checkpoint.
  # Inputs: stacked_table, the full value of a stacked sharded table; from_shard_layouts,
  # the layouts of the constituent tables as stored in the checkpoint.
  # For each layout, undoes the sparse core shuffle (using core count, row offset, size per
  # shard and rotation) and strips the padding down to the table's unsharded shape.
  # Returns a list of unsharded table tensors in layout order. Logs at verbose level.
  """Undo the shard the feature tables into SparseCore stacked table.

  Args:
    stacked_table: The value of a SparseCore stacked and sharded table.
    from_shard_layouts: The target layouts for the target hardware.

  Returns:
    The unsharded feature tables.
  """
  logging.vlog(
      1,
      "To unshuffle_from_sc_to_cpu on stacked_table.shape: %s",
      stacked_table.shape,
  )
  ret_tensors = []

  for layout in from_shard_layouts:
    padded_table = tpu_embedding_v3_utils.unshuffle_from_sc_to_cpu(
        stacked_table,
        num_sparse_cores=layout.num_sparse_cores,
        offset_in_shard=layout.sparse_core_shard_row_offset,
        size_in_shard=layout.unsharded_padded_shape[0]
        // layout.num_sparse_cores,
        shard_rotation=layout.sparse_core_shard_rotation,
    )

    orig_table = tpu_embedding_v3_utils.remove_padding_from_sc(
        padded_table, layout.unsharded_shape
    )

    logging.vlog(
        1, "orig_tensors.shape[%s]: %s", layout.table_name, orig_table.shape
    )
    ret_tensors.append(orig_table)

  return ret_tensors


class EmbeddingUnshardToShardCallback(checkpoint_adapter.ReshardCallback):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring is identical to the one of EmbeddingReshardCallback and does
  # not say this class converts an unsharded (plain per-table) checkpoint into the sharded
  # SparseCore layout.
  # Reshard callback used when the checkpoint holds one plain full tensor per logical table
  # and the object being restored is a SparseCore stacked sharded variable. It remembers the
  # object's local name, the checkpoint names of each constituent table, and the target
  # layouts (the optional unshard layout is stored but unused). Created by the checkpoint
  # adapter; the restore machinery asks it which keys and slices to read and then calls
  # reshard on the read values.
  """Reshard callback for embeddings."""

  def __init__(
      self,
      object_local_name: str,
      checkpoint_local_names: Sequence[str],
      to_shard_layout: Optional[
          Sequence[sparse_core_layout_pb2.SparseCoreTableLayout]
      ] = None,
      to_unshard_layout: Optional[
          Sequence[sparse_core_layout_pb2.SparseCoreTableLayout]
      ] = None,
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: object_local_name, the name of the stacked variable being
    # restored; checkpoint_local_names, ordered names of the per-table checkpoint entries
    # (the first is treated as the main name); to_shard_layout, optional list of target table
    # layouts; to_unshard_layout, optional layouts of the checkpoint (stored, never used).
    # Stores all of them on the instance. Raises an index error if the name list is empty.
    # Returns nothing.
    """Initializes  Reshard callback.

    Args:
      object_local_name:  The local name of the object being restored.
      checkpoint_local_names: The local names of the checkpoint positions that
        need to be read.
      to_shard_layout: (Optional) Target layouts as specified in the embedding
        being restored.
      to_unshard_layout: (Optional) Layouts as stored in checkpoint being
        restored from.
    """
    self._object_local_name = object_local_name
    self._checkpoint_local_names = checkpoint_local_names
    self._to_shard_layout = to_shard_layout
    self._to_unshard_layout = to_unshard_layout
    self._main_checkpoint_name = checkpoint_local_names[0]

  def object_name(self) -> str:
    # [FUNCTION DOC]
    # Returns the local name of the object being restored, as given at construction.
    return self._object_local_name

  def update_restore_inputs(
      self, checkpoint_key: str, shape_and_slice_spec: str
  ) -> tuple[Sequence[str], Sequence[str]]:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shape and slice input is only logged, never used for the result, and
    # the target layout is iterated without a check, so it fails if no target layout was given.
    # Inputs: checkpoint_key, the key requested by the restore; shape_and_slice_spec, the
    # requested slice text. For each target table layout, builds a key by replacing the main
    # checkpoint name in the requested key with that table's checkpoint name, and a slice text
    # covering the table's whole unsharded shape (offset zero).
    # Returns a pair of lists (keys, slices) for the restore op. Logs at info level.
    """Updates checkpoint key and slice spec acorrding to the resharding plan.

    Args:
      checkpoint_key: The input checkpoint key to be read.
      shape_and_slice_spec: The shape and slice spec of the checkpoint key to be
        read.

    Returns:
      A tuple of (keys, slices) that should be passed to restore_v2 inorder to
      reshard according to the resharding plan. The restored tensors from
      restore_v2 op will usually be passed to reshard method of this class to
      get the final resharded value.
    """
    keys = []
    slices = []
    # TODO(b/398016624): Make this a vlog this log after bug is fixed.
    logging.info(
        "Updating restore v2 inputs for %s: %s",
        checkpoint_key,
        shape_and_slice_spec,
    )
    for i, layout in enumerate(self._to_shard_layout):
      sub_checkpoint_key = checkpoint_key.replace(
          self._main_checkpoint_name, self._checkpoint_local_names[i]
      )
      # For resharding later, we need to read the full value here.
      # TODO(b/398016624): Make this a vlog this log after bug is fixed.
      logging.info(
          "Will read sub key %s: %s",
          sub_checkpoint_key,
          layout.unsharded_shape,
      )
      keys.append(sub_checkpoint_key)
      slices.append(
          _shard_info_str(
              layout.unsharded_shape,
              trackable_base.ShardInfo(
                  offset=[0, 0], shape=layout.unsharded_shape
              ),
          )
      )
    return (keys, slices)

  def reshard(
      self, checkpoint_values: tensor.Tensor, shape_and_slice: str
  ) -> tensor.Tensor:
    # [FUNCTION DOC]
    # Inputs: checkpoint_values, the full per-table tensors read from the checkpoint (a
    # sequence, one per target layout, despite the single-tensor annotation); shape_and_slice,
    # the slice text of the stacked variable that is wanted. Delegates to the module's
    # CPU-to-SparseCore sharding routine using the stored target layouts and returns the
    # resulting tensor slice. Raises a value error for an invalid offset or row count.
    """Reshards the checkpoint values according to the resharding plan.

    Args:
      checkpoint_values: The checkpoint values to be resharded.
      shape_and_slice: The shape and slice spec to be returned after resharding.

    Returns:
      The resharded tensor slice.
    """
    return _shard_from_cpu_to_sc(
        checkpoint_values, shape_and_slice, self._to_shard_layout
    )


class EmbeddingReshardCallback(checkpoint_adapter.ReshardCallback):
  # [CLASS DOC]
  # Reshard callback used when the checkpoint already holds SparseCore stacked tables saved
  # with a different layout than the one being restored. It keeps the object's local name,
  # a mapping from checkpoint stacked table names to their constituent table layouts, and the
  # list of target layouts. On restore it asks for the full stacked tables, unshards them into
  # per-table tensors and reshards them into the target layout. Created by the checkpoint
  # adapter when layouts differ.
  """Reshard callback for embeddings."""

  def __init__(
      self,
      object_local_name: str,
      from_shard_layouts: Mapping[
          str, Sequence[sparse_core_layout_pb2.SparseCoreTableLayout]
      ],
      to_shard_layouts: Sequence[sparse_core_layout_pb2.SparseCoreTableLayout],
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: object_local_name, the stacked variable name being restored;
    # from_shard_layouts, mapping from checkpoint stacked table names to their ordered table
    # layouts; to_shard_layouts, list of target table layouts. Stores them on the instance and
    # logs at info level. Returns nothing.
    """Initializes  Reshard callback.

    Args:
      object_local_name:  The local name of the object being restored.
      from_shard_layouts: A dictionary in stacked table name to a list of its
        consituent table layouts.  The layouts are coming from the checkpoint
        being restored.
      to_shard_layouts: a list of target layouts that will be resharded to.
    """
    logging.info("Creating EmbeddingReshardCallback for %s", object_local_name)
    self._object_local_name = object_local_name
    self._from_shard_layouts = from_shard_layouts
    self._to_shard_layouts = to_shard_layouts

  def object_name(self) -> str:
    # [FUNCTION DOC]
    # Returns the local name of the object being restored, as given at construction.
    return self._object_local_name

  def update_restore_inputs(
      self, checkpoint_key: str, shape_and_slice_spec: str
  ) -> tuple[Sequence[str], Sequence[str]]:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns the full shape, but it returns a pair of
    # lists (keys, slices); the shape and slice input is only used in a log message.
    # Inputs: checkpoint_key, the requested key; shape_and_slice_spec, the requested slice
    # text. For every checkpoint stacked table, builds a key by replacing the object name in
    # the requested key with the stacked table name, and a slice text covering the whole
    # stacked table (rows from the first layout's rows per core times core count, columns from
    # its padded width, offset zero).
    # Returns a pair of lists (keys, slices). Logs at info level.
    """Return the full shape of the stacked that is passed into restore_v2.

    This shape information is required by the restore_v2 process to ensure it
    loads the complete tensor from the checkpoint. The full tensor is required
    to perform resharding operations.

    Args:
      checkpoint_key: The input checkpoint key to be read.
      shape_and_slice_spec: The shape and slice spec of the checkpoint key to be
        read.

    Returns:
      A tuple of (keys, slices) that should be passed to restore_v2 in order to
      reshard according to the resharding plan. The restored tensors from
      restore_v2 op will usually be passed to reshard method of this class to
      get the final resharded value.
    """
    keys = []
    slices = []
    for stacked_name, table_layouts in self._from_shard_layouts.items():
      key = checkpoint_key.replace(self._object_local_name, stacked_name)
      keys.append(key)

      # use the first layout get the full shape of the stacked table
      first_layout = table_layouts[0]
      full_vocab_size = (
          first_layout.total_rows_per_sparse_core_shard
          * first_layout.num_sparse_cores
      )
      stack_dim = first_layout.unsharded_padded_shape[1]
      full_shape = [full_vocab_size, stack_dim]
      slices.append(
          _shard_info_str(
              full_shape,
              trackable_base.ShardInfo(offset=[0, 0], shape=full_shape),
          )
      )

    logging.info(
        "Updating restore v2 inputs for %s[%s]:%s to stacked_tables: [%s],"
        " slices: [%s]",
        checkpoint_key,
        self._object_local_name,
        shape_and_slice_spec,
        ", ".join(keys),
        ", ".join(slices),
    )

    return (keys, slices)

  def reshard(
      self,
      checkpoint_values: Sequence[tensor.Tensor],
      shape_and_slice: str,
  ) -> tensor.Tensor:
    # [FUNCTION DOC]
    # Inputs: checkpoint_values, sequence of full stacked table tensors read from the
    # checkpoint, in the order of the source layout mapping; shape_and_slice, slice text of
    # the wanted part of the target stacked variable. Unshards each stacked table into its
    # per-table tensors, picks the tables needed by the target layouts by table name, then
    # shards them into the target layout for the requested slice.
    # Returns the resulting tensor. Raises a key error if a needed table is missing, and a
    # value error for an invalid slice. Logs progress and elapsed time at info level.
    # unshard
    stime = time.time()
    logging.info(
        "EmbeddingReshardCallback: starting to reshard [%s],"
        " from checkpoint_value with shapes: %s",
        self._object_local_name,
        ", ".join([str(t.shape) for t in checkpoint_values]),
    )

    unsharded_tables = dict()

    for stacked_table, layouts in zip(
        checkpoint_values,
        list(self._from_shard_layouts.values()),
    ):
      logging.info(
          "Unshard sc_to_cpu stacked_table: %s, shape: %s, no. of constituent"
          " tables: %d",
          layouts[0].stacked_table_name,
          stacked_table.shape,
          len(layouts),
      )

      unsharded_tensors = _unshard_from_sc_to_cpu(stacked_table, layouts)
      for unshared_tensor, layout in zip(unsharded_tensors, layouts):
        unsharded_tables[layout.table_name] = unshared_tensor

    required_tables = [
        unsharded_tables[layout.table_name] for layout in self._to_shard_layouts
    ]
    ret = _shard_from_cpu_to_sc(
        required_tables, shape_and_slice, self._to_shard_layouts
    )

    etime = time.time()
    logging.info(
        "EmbeddingReshardCallback: reshard [%s] took %s",
        self._object_local_name,
        etime - stime,
    )
    return ret


def _reorg_layouts(
    layouts: Sequence[sparse_core_layout_pb2.SparseCoreTableLayout],
) -> Mapping[str, Sequence[sparse_core_layout_pb2.SparseCoreTableLayout]]:
  # [FUNCTION DOC]
  # HELPER: groups table layouts by stacked table for the checkpoint adapter.
  # Input: layouts, a sequence of table layouts. Groups them by stacked table name and sorts
  # each group by the row offset within the shard. Returns a defaultdict-style mapping from
  # stacked table name to the sorted list of layouts. No side effects.
  """Reorg the layouts to be in the order of the logical table.

    Building a Dict[StackedTableName, SortedList[TableLayout]]

  Args:
    layouts: The layouts to be reorged.

  Returns:
    A dict of stacked table name to sorted list of table layouts.
  """
  stacked_name_to_table_names = collections.defaultdict(list)
  for layout in layouts:
    stacked_name_to_table_names[layout.stacked_table_name].append(layout)
  for stacked_name in stacked_name_to_table_names.keys():
    sorted_layouts = sorted(
        stacked_name_to_table_names[stacked_name],
        key=lambda layout: layout.sparse_core_shard_row_offset,
    )
    stacked_name_to_table_names[stacked_name] = sorted_layouts

  return stacked_name_to_table_names


class TpuEmbeddingV3CheckpointAdapter(
    checkpoint_adapter.AbstractCheckpointAdapter
):
  # [CLASS DOC]
  # Checkpoint adapter that lets TPU embedding V3 objects be restored when the checkpoint
  # layout differs from the live layout. State: a mapping from table name to the layouts found
  # in the checkpoint, and a mapping from checkpoint names to reshard callbacks, filled by
  # initialize_reshard_callbacks. It decides whether a trackable needs resharding
  # (is_applicable) and supplies the callback by name (get_reshard_callback). Subclass of the
  # abstract checkpoint adapter.
  """Adapter for TPU Embedding V3 to handle checkpoint resharding."""

  def __init__(
      self,
      layouts: Optional[sparse_core_layout_pb2.SparseCoreTableLayouts] = None,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says layouts are the "target layouts", but they are the
    # layouts stored in the checkpoint being restored.
    # Constructor. Input: layouts, optional collection of table layouts read from a
    # checkpoint (nothing or empty means the checkpoint has none). Builds a mapping from table
    # name to layout and an empty callback mapping. Returns nothing.
    """An adapter for TPUEmbeddingV3 checkpoints.

    Constructs an adapter for TPUEmbeddingV3 to handle layout changes. between
    checkpoint values and embedding object being restored.

    Args:
     layouts: The target layouts required.
    """
    self._checkpoint_layouts = {}
    self._checkpoint_to_reshard_callback = {}
    if layouts:
      for layout in layouts.tables:
        self._checkpoint_layouts[layout.table_name] = layout

  @classmethod
  def create_from_checkpoint(cls, save_path: str):
    # [FUNCTION DOC]
    # Factory. Input: save_path, path prefix of a checkpoint. Opens a checkpoint reader, looks
    # for the first variable whose name contains the SparseCore layouts key, and parses its
    # stored bytes as a layouts message. Returns a new adapter built from those layouts, or
    # one without layouts if the checkpoint has none. Logs the loaded layouts at info level.
    # Raises errors from the reader if the path is unreadable.
    reader = py_checkpoint_reader.NewCheckpointReader(save_path)
    sparsecore_layouts_str = None
    for name in reader.get_variable_to_dtype_map():
      if tpu_embedding_v3_utils.SPARSECORE_LAYOUTS_CHECKPOINT_KEY in name:
        sparsecore_layouts_str = reader.get_tensor(name)
        break
    if sparsecore_layouts_str is None:
      return cls(None)
    layouts = sparse_core_layout_pb2.SparseCoreTableLayouts()
    layouts.ParseFromString(sparsecore_layouts_str)
    logging.info("Loaded layouts from checkpoint: %s", layouts)
    return cls(layouts)

  def initialize_reshard_callbacks(
      self,
      embedding_layouts: Optional[
          Mapping[str, sparse_core_layout_pb2.SparseCoreTableLayout]
      ] = None,
  ):
    # [FUNCTION DOC]
    # Builds and stores the reshard callbacks for a restore. Input: embedding_layouts,
    # optional mapping from table name to the live (target) layout.
    # Cases: if the checkpoint has no layouts and the embedding has some, groups target layouts
    # by stacked table, sorts by row offset and registers an unshard-to-shard callback keyed
    # by the first table name. If the embedding has no layouts, raises a not-implemented error
    # (sharded to unsharded). Otherwise registers a reshard callback per target stacked table
    # that lists the checkpoint stacked tables containing its tables.
    # Side effects: fills the callback mapping; logs at info level. Returns nothing.
    if not self._checkpoint_layouts and embedding_layouts:
      # From Unsharded to Sharded
      stacked_name_to_table_names = collections.defaultdict(list)
      for layout in embedding_layouts.values():
        stacked_name_to_table_names[layout.stacked_table_name].append(layout)
      for stacked_name, layouts in stacked_name_to_table_names.items():
        # Make the first table name as the key for checkpoint position
        # The sorting here is by the position of the logical table in the shard
        sorted_layouts = sorted(
            layouts, key=lambda layout: layout.sparse_core_shard_row_offset
        )
        logging.info("Creating resharding plan for %s", stacked_name)
        self._checkpoint_to_reshard_callback[sorted_layouts[0].table_name] = (
            EmbeddingUnshardToShardCallback(
                stacked_name,
                [l.table_name for l in sorted_layouts],
                sorted_layouts,
                None,
            )
        )
      return
    if not embedding_layouts:
      # TODO(b/326644306): From sharded to unsharded
      raise NotImplementedError("Sharded to Unsharded is not implemented yet.")
    # Reshard to different SC Layout
    from_layouts = _reorg_layouts(list(self._checkpoint_layouts.values()))
    to_layouts = _reorg_layouts(list(embedding_layouts.values()))
    for stacked_name, table_layouts in to_layouts.items():
      # look for required stacked tables
      required_stacked_tables = dict()
      for table_layout in table_layouts:
        for from_stacked_name, from_table_layouts in from_layouts.items():
          if table_layout.table_name in {
              layout.table_name for layout in from_table_layouts
          }:
            required_stacked_tables[from_stacked_name] = from_table_layouts

      logging.info(
          "Creating resharding plan for %s, required stacked_tables: %s",
          stacked_name,
          ", ".join(required_stacked_tables.keys()),
      )
      self._checkpoint_to_reshard_callback[stacked_name] = (
          EmbeddingReshardCallback(
              object_local_name=stacked_name,
              from_shard_layouts=required_stacked_tables,
              to_shard_layouts=to_layouts[stacked_name],
          )
      )

  def is_layouts_same(self, embedding_layouts) -> bool:
    # [FUNCTION DOC]
    # Input: embedding_layouts, mapping from table name to the live layout. Raises a value
    # error if its keys differ from the checkpoint layout keys. Compares each checkpoint
    # layout with the live one by protocol-buffer equality; logs the first mismatch at info
    # level. Returns false on any mismatch, true if all match.
    """Returns True if the all the embedding and checkpoint layouts are the same.

    Args:
      embedding_layouts: dict of layouts for embedding tables.

    Raises: ValueError if the embedding layouts and checkpoint layouts do not
      have the same keys.
    Returns: Bool representing if the embedding layouts match the layouts in
      checkpoint.
    """
    if self._checkpoint_layouts.keys() != embedding_layouts.keys():
      raise ValueError(
          "Layouts in checkpoint and embedding must have the same keys. found"
          " {} and {}".format(
              self._checkpoint_layouts.keys(), embedding_layouts.keys()
          )
      )

    for key, layout in self._checkpoint_layouts.items():
      if not compare.ProtoEq(layout, embedding_layouts[key]):
        logging.info(
            "Layouts do not match for %s this will require resharding; %s"
            " vs %s",
            key,
            layout,
            embedding_layouts[key],
        )
        return False
    return True

  def is_applicable(self, trackable: trackable_base.Trackable) -> bool:
    # [FUNCTION DOC]
    # Decides whether this adapter must reshard the given trackable. Input: trackable, the
    # object being restored. Returns false if its type name does not look like a V2 or V2-plus
    # TPU embedding (checked by text to avoid a circular dependency), if neither checkpoint
    # nor object has layouts, or if both have layouts and they match. Otherwise initializes the
    # reshard callbacks from the object's layouts (side effect) and returns true. May raise
    # the errors of is_layouts_same or initialize_reshard_callbacks. Logs at info level.
    # issubclass(trackable, TPUEmbeddingBase) adds circular deps, hence using
    #  a workaround to select the applicable embedding implementations.
    allowed_class_names = [".TPUEmbeddingV2Plus", ".TPUEmbeddingV2"]
    if not any(x in str(type(trackable)) for x in allowed_class_names):
      return False
    embedding_layouts = None
    if hasattr(trackable, "embedding_layouts"):
      embedding_layouts = trackable.embedding_layouts
    # Neither checkpoint not target embedding has layout, no resharding needed.
    if not self._checkpoint_layouts and not embedding_layouts:
      logging.info("No resharding needed, no layouts")
      return False
    # Only if both checkpoint and embedding have layouts and they match,
    # no resharding needed.
    if (
        self._checkpoint_layouts
        and embedding_layouts
        and self.is_layouts_same(embedding_layouts)
    ):
      logging.info("No resharding needed; layouts match")
      return False
    # Else we need to reshard.
    self.initialize_reshard_callbacks(embedding_layouts)
    return True

  def get_reshard_callback(
      self, name: str
  ) -> Optional[checkpoint_adapter.ReshardCallback]:
    # [FUNCTION DOC]
    # Input: name, a checkpoint object name. Returns the registered reshard callback for the
    # name, or, if absent, for the part before the first slash (so slot variables of an
    # embedding share its callback), or nothing if neither is registered. No side effects.
    if name in self._checkpoint_to_reshard_callback:
      return self._checkpoint_to_reshard_callback[name]
    # Check if this is slot variable
    var_name = name.split("/")[0]
    if var_name in self._checkpoint_to_reshard_callback:
      return self._checkpoint_to_reshard_callback[var_name]
    return None
