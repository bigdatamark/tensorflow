# [MODULE DOC]
# Input-pipeline utilities for distribution strategies: automatic sharding of a dataset across
# workers (public-to-the-package function) plus private helpers that clone a dataset's graph of
# variant-producing operations into the current default graph. Used by the distributed input
# library. No import-time side effects beyond imports.

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
"""Input-pipeline utilities for Distribution strategies."""

from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.data.util import traverse
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.types import data as data_types
from tensorflow.python.types import distribute as distribute_types


# pylint: disable=protected-access
def auto_shard_dataset(dataset, num_shards, index, num_replicas_in_sync=None):
  # [FUNCTION DOC]
  # Shards a dataset for one worker. Parameters: the dataset; the number of shards (scalar integer
  # or tensor); this worker's shard index; and an optional total replica count across workers
  # (used when sharding by data; defaults to one when omitted). If the dataset is already a
  # distributed dataset, delegates to its own auto-shard method. Otherwise, if the dataset's
  # distribute option for auto-shard policy is not off, wraps it in the auto-shard dataset
  # transformation (the version-one variant for version-one datasets) and returns it. If the
  # policy is off, returns the input dataset unchanged. Also returns the input unchanged when no
  # good sharding is determined by the rewrite. Raises nothing itself; no global side effects.
  """Shard the input pipeline by sharding the underlying list of files.

  Args:
    dataset: A `tf.data.Dataset` instance, typically the result of a bunch of
      dataset transformations.
    num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel. Same usage as in `tf.data.Dataset.shard`.
    index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.
      Same usage as in `tf.data.Dataset.shard`.
    num_replicas_in_sync: An integer representing the total number of replicas
      across all workers. This is used in the rewrite when sharding by data.

  Returns:
    A modified `Dataset` obtained by updating the pipeline sharded by the
    files. The input dataset will be returned if we cannot automatically
    determine a good way to shard the input dataset.
  """
  if isinstance(dataset, distribute_types.DistributedDatasetInterface):
    return dataset.auto_shard(num_shards, index)
  if (dataset.options().experimental_distribute.auto_shard_policy !=
      AutoShardPolicy.OFF):
    if num_replicas_in_sync is None:
      num_replicas_in_sync = 1
    if isinstance(dataset, data_types.DatasetV1):
      return distribute._AutoShardDatasetV1(dataset, num_shards, index,
                                            num_replicas_in_sync)
    else:
      return distribute._AutoShardDataset(dataset, num_shards, index,
                                          num_replicas_in_sync)
  else:
    return dataset


def _clone_dataset(dataset):
  # [FUNCTION DOC]
  # HELPER: dataset graph cloning for the distributed input code.
  # Clones the chain of operations that produce the given dataset's variant tensor into the
  # current default graph, and returns a new dataset wrapping the cloned variant tensor with the
  # original element spec. Parameter: a dataset. Side effect: adds operations to the default graph.
  """Returns a cloned version of `dataset`."""
  variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(dataset)
  remap_dict = _clone_helper(dataset._variant_tensor.op, variant_tensor_ops)
  new_variant_tensor = remap_dict[dataset._variant_tensor.op].outputs[0]
  return dataset_ops._VariantDataset(new_variant_tensor, dataset.element_spec)


def _get_op_def(op):
  # [FUNCTION DOC]
  # HELPER: operation definition lookup for the cloning code.
  # Returns the operation's own op definition if it has one, otherwise looks it up in the global
  # op definition registry by the operation's type name (may return nothing if unregistered).
  return op.op_def or op_def_registry.get(op.type)


def _clone_helper(op_to_clone, variant_tensor_ops):
  # [FUNCTION DOC]
  # HELPER: recursive worker of dataset cloning.
  # Parameters: the operation to clone, and the collection of variant-producing operations that
  # must be cloned along the way. First recursively clones every input operation that is in that
  # collection and merges the results; then builds the input list for the new operation, using
  # the cloned output where the producer was cloned and the original tensor otherwise. Creates the
  # copy in the current default graph with the same type, output types, name, attributes and op
  # definition. Returns a mapping from original operations to new operations, including the
  # operation passed in. Side effect: adds operations to the default graph.
  """Helper method that recursively clones `op_to_clone`.

  Args:
    op_to_clone: The op we want to clone.
    variant_tensor_ops: A list of ops that we have to clone along the way.

  Returns:
    A dictionary mapping old_ops to new_ops created. Includes op_to_clone
    as a key.
  """
  remap_dict = {}
  for input_tensor in op_to_clone.inputs:
    input_tensor_op = input_tensor.op
    if input_tensor_op in variant_tensor_ops:
      recursive_map = _clone_helper(input_tensor_op, variant_tensor_ops)
      remap_dict.update(recursive_map)
  inputs_list = []
  for input_tensor in op_to_clone.inputs:
    input_tensor_op = input_tensor.op
    if input_tensor_op in remap_dict:
      remapped_input = remap_dict[input_tensor_op].outputs[0]
      inputs_list.append(remapped_input)
    else:
      inputs_list.append(input_tensor_op.outputs[input_tensor.value_index])
  g = ops.get_default_graph()
  new_op = g.create_op(
      op_to_clone.type,
      inputs_list, [o.dtype for o in op_to_clone.outputs],
      name=op_to_clone.name,
      attrs=op_to_clone.node_def.attr,
      op_def=_get_op_def(op_to_clone))
  remap_dict[op_to_clone] = new_op
  return remap_dict
