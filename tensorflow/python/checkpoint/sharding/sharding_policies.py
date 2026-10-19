# [MODULE DOC]
# Built-in checkpoint sharding policies, exported publicly as experimental train policies. A
# sharding policy is a callback taking a list of shardable tensors and returning a list of shards
# (each a mapping from checkpoint key to a mapping from slice spec to tensor). Provides a policy
# that groups everything in one shard and a policy that splits tensors, slicing them along an axis
# when needed, so shards stay under a maximum byte size. Registers the public exports at import.
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
"""Checkpoint policies that determine how tensors are split into shards."""

import math
import operator
from typing import MutableSequence, Sequence

from absl import logging

from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.eager import context
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.util import tf_export


@tf_export.tf_export("train.experimental.ShardByTaskPolicy")
class ShardByTaskPolicy(sharding_util.ShardingCallback):
  # [CLASS DOC]
  # KNOWN ISSUE: despite the docstring and description, no device or task information is used;
  # all tensors end up in a single shard.
  # Sharding callback with no state. Instances are called with the shardable tensors.
  """Policy that splits tensors into shards based on their device spec task."""

  @property
  def description(self) -> str:
    # [FUNCTION DOC]
    # Returns the fixed human-readable description string of this policy (which claims task
    # based splitting).
    return "Split tensors into shards based on their device spec task."

  def __call__(
      self,
      shardable_tensors: Sequence[sharding_util.ShardableTensor]
  ) -> Sequence[sharding_util.Shard]:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says tensors are split by device spec task, but the code never
    # reads the device; it puts every tensor into one shard, keyed by checkpoint key and slice
    # spec (the local named tensors_by_task is misleading).
    # Input: shardable_tensors, list of records with tensor, checkpoint key and slice spec.
    # Returns a list containing exactly one shard: mapping from checkpoint key to a mapping from
    # slice spec to tensor. No side effects.
    """Callback to split tensors into shards based on their device spec task.

    Args:
      shardable_tensors: A list of ShardableTensors.

    Returns:
      List of shard dicts containing tensors.
          [ {checkpoint key: {slice_spec: tensor} } ]
    """
    tensors_by_task = {}

    for shardable_tensor in shardable_tensors:
      tensor = shardable_tensor.tensor
      checkpoint_key = shardable_tensor.checkpoint_key
      slice_spec = shardable_tensor.slice_spec

      (tensors_by_task
       .setdefault(checkpoint_key, {})[slice_spec]) = tensor

    return [tensors_by_task]


_OffsetAndShape = tuple[Sequence[int], Sequence[int]]


@tf_export.tf_export("train.experimental.MaxShardSizePolicy")
class MaxShardSizePolicy(sharding_util.ShardingCallback):
  # [CLASS DOC]
  # Sharding callback that packs tensors into shards no larger than a maximum byte size, slicing
  # tensors along an axis when they do not fit. Holds max_shard_size, set at construction.
  # Scalars or minimal slices larger than the limit become their own oversized shards; the object
  # graph proto and variant values count as zero size. Each call uses a fresh nested partitioner.
  """Policy that splits tensors into shards with a max shard size.

  Shards may exceed the max shard size if they contain 1. a single scalar/string
  tensor that could not be sliced and exceeds the max shard size or 2. the
  checkpoint object graph, whose size cannot be calculated when saving.
  """

  class MaxShardSizePartitioner():
    # [CLASS DOC]
    # HELPER: worker object for MaxShardSizePolicy.
    # Mutable per-call state: the max shard size, the shards being filled, oversized scalar
    # shards, the remaining room in the current shard, and details of the tensor currently being
    # partitioned (key, dtype, device, root tensor, slice spec, shapes, offsets, element size,
    # and the not yet placed remainder's offset and shape). Created with no arguments and used
    # through its method that returns the shards.
    """Partition tensors into shards with a max shard size."""

    max_shard_size: int
    _large_scalars: MutableSequence[sharding_util.Shard]
    _tensors_by_shard: MutableSequence[sharding_util.Shard]
    _shard_size_remaining: int
    _checkpoint_key: str
    _dtype: dtypes.DType
    _device: device_lib.DeviceSpec
    _root_tensor: tensor_lib.Tensor
    _slice_spec: variables.Variable.SaveSliceInfo
    _full_shape: tensor_shape.TensorShape
    _root_shape: tensor_shape.TensorShape
    _root_offset: Sequence[int]
    _dtype_size: int
    _working_tensor_offset: MutableSequence[float]
    _working_tensor_shape: tensor_shape.TensorShape

    def _get_next_partition(self) -> tuple[int, float]:
      # [FUNCTION DOC]
      # HELPER: chooses how to cut the remaining tensor for the current shard.
      # Uses the remaining working shape, element size and room left in the shard. For unknown
      # rank or scalars returns axis zero and infinity. Otherwise picks the axis needing the
      # fewest partitions whose partition fits in the remaining room (axis zero is the default
      # even if it does not fit). Returns the axis and the number of slices along it per
      # partition; the size is zero when not even one slice fits the remaining room.
      """Gets tensor partition with size closest to shard_size_remaining.

      Returns:
        A tuple containing the axis and size of the next partition.
      """
      rank = self._working_tensor_shape.rank
      if rank is None or rank == 0:
        return 0, math.inf

      num_elems = self._working_tensor_shape.num_elements()

      def num_partitions(axis: int) -> float:
        # [FUNCTION DOC]
        # HELPER: for the given axis, computes how many bytes one slice along it takes and how
        # many slices fit in the remaining shard room; returns infinity if none fit, otherwise
        # the number of partitions needed to cover the axis length (rounded up).
        axis_len = self._working_tensor_shape.dims[axis].value
        slice_elems = num_elems // axis_len
        bytes_per_slice = slice_elems * self._dtype_size
        slices_per_shard = self._shard_size_remaining // bytes_per_slice
        if slices_per_shard == 0:
          return math.inf
        return math.ceil(axis_len / slices_per_shard)

      # Find axis with minimum partitions. (axis with maximum partition size)
      # (max partition size is as close as possible to the shard_size_remaining)
      min_parts = num_partitions(0)
      min_axis = 0
      for axis in range(1, rank):
        parts_along_axis = num_partitions(axis)
        part_size = num_elems * self._dtype_size / parts_along_axis
        if (parts_along_axis < min_parts and
            part_size <= self._shard_size_remaining):
          min_axis, min_parts = axis, int(parts_along_axis)
      return (min_axis,
              math.ceil(int(self._working_tensor_shape[min_axis]) / min_parts))

    def _add_partition(self, part_axis: int, part_size: float):
      # [FUNCTION DOC]
      # HELPER: slices off a partition of the working tensor and places it.
      # Inputs: part_axis (axis to cut), part_size (slice extent along that axis). Slices the root
      # tensor on its device and builds the slice spec. If the slice is larger than the maximum
      # shard size it is logged as a warning and stored as its own oversized shard; if it is
      # larger than the remaining room (but not the maximum) raises a runtime error; otherwise
      # adds it to the current shard, reducing remaining room and starting a new shard when the
      # room hits zero. Finally advances the working offset and shape past the placed part.
      # Mutates the partitioner state; returns nothing.
      """Adds the tensor partition to the shard, if possible.

      Args:
        part_axis: The axis of the partition.
        part_size: The size of the partition.

      Raises:
        RuntimeError: When the slice size is larger than the remaining shard
        size.
      """

      # Add what we can to the current shard.
      relative_offset = list(
          map(operator.sub, self._working_tensor_offset, self._root_offset))
      slice_shape = list(map(operator.sub, self._root_shape, relative_offset))
      slice_shape[part_axis] = part_size
      slice_size_in_bytes = int(math.prod(slice_shape)) * self._dtype_size
      with ops.device(self._device):
        tensor_slice = array_ops.slice(
            self._root_tensor, begin=relative_offset, size=slice_shape)
      slice_spec = variables.Variable.SaveSliceInfo(
          full_name=self._checkpoint_key,
          full_shape=self._full_shape,
          var_offset=self._working_tensor_offset,
          var_shape=slice_shape).spec.strip()
      if slice_size_in_bytes > self.max_shard_size:
        logging.warning("Tensor %s's minimum slice %s has size %s bytes and "
                        "cannot be partitioned into a shard of max shard size "
                        "%s bytes. It will be added as an individual shard "
                        "that exceeds the max shard size.",
                        self._checkpoint_key, slice_spec, slice_size_in_bytes,
                        self.max_shard_size)
        self._large_scalars.append(
            {self._checkpoint_key: {slice_spec: tensor_slice}})
      elif slice_size_in_bytes > self._shard_size_remaining:
        raise RuntimeError(
            f"Slice size ({slice_size_in_bytes} bytes) is larger than the "
            f"remaining shard size ({self._shard_size_remaining} bytes). This "
            "should have been caught in MaxShardSizePolicy._add_partition().")
      else:
        (self._tensors_by_shard[-1]
         .setdefault(self._checkpoint_key, {})[slice_spec]) = tensor_slice
        self._shard_size_remaining -= slice_size_in_bytes
        if self._shard_size_remaining == 0:
          self._tensors_by_shard.append({})
          self._shard_size_remaining = self.max_shard_size

      # Get remaining portion of tensor to add to the next shard(s).
      self._working_tensor_offset[part_axis] += part_size
      relative_offset[part_axis] += part_size
      self._working_tensor_shape = tensor_shape.TensorShape(list(
          map(operator.sub, self._root_shape, relative_offset)))

    def get_shards(
        self,
        max_shard_size: int,
        shardable_tensors: Sequence[sharding_util.ShardableTensor]
    ) -> Sequence[sharding_util.Shard]:
      # [FUNCTION DOC]
      # Packs tensors into shards. Inputs: max_shard_size (bytes) and shardable_tensors (list of
      # records with key, dtype, device, tensor, slice spec, shape). Resets state, then for each
      # tensor derives full and root shape and offset (from an existing slice spec if sliced),
      # and its byte size: zero for the object graph proto key and variants, and for strings in a
      # function without a session (warns once); otherwise string sizes are measured, evaluating
      # via the default session in graph mode. Scalars go into the current shard, a new one if
      # needed, or alone if over the limit (with a warning). Other tensors are repeatedly cut into
      # partitions that fit; a zero-size partition starts a fresh shard. The remainder is added
      # as a slice (or the whole tensor). Returns the full shards in order (leading empty shard
      # dropped) followed by oversized scalar shards.
      """Callback to split tensors into shards with a max shard size.


      Args:
        max_shard_size: The maximum size of a shard file in bytes.
        shardable_tensors: A list of ShardableTensors.

      Returns:
        List of shard dicts containing tensors.
            [ {checkpoint key: {slice_spec: tensor} } ]
      """
      self.max_shard_size = max_shard_size
      self._tensors_by_shard = [{}]
      self._large_scalars = []

      string_size_warning_printed = False
      self._shard_size_remaining = self.max_shard_size
      for shardable_tensor in shardable_tensors:
        self._checkpoint_key = shardable_tensor.checkpoint_key
        self._dtype = shardable_tensor.dtype
        self._device = shardable_tensor.device
        self._root_tensor = shardable_tensor.tensor
        self._slice_spec = shardable_tensor.slice_spec
        # If the tensor has already been sliced, make sure to keep track of its
        # parent tensor's shape & offset. These will be used when creating slice
        # specs later.
        if self._slice_spec:
          save_slice_info = variables.Variable.SaveSliceInfo.from_spec(
              self._slice_spec)
          self._full_shape = tensor_shape.TensorShape(
              save_slice_info.full_shape)
          self._root_shape = tensor_shape.TensorShape(save_slice_info.var_shape)
          self._root_offset = save_slice_info.var_offset
        else:
          self._full_shape = self._root_shape = shardable_tensor.shape
          self._root_offset = [0] * self._root_shape.rank

        self._dtype_size = dtypes.as_dtype(self._dtype).size
        total_size = self._root_shape.num_elements() * self._dtype_size  # bytes

        # Calculate string tensor sizes.
        if self._checkpoint_key == base.OBJECT_GRAPH_PROTO_KEY:
          # In graph mode, the object graph is populated using feed_additions
          # when the session is run. So, we can't calculate the size here.
          # Fortunately, the serialized object graph string will never be that
          # big, so we just place it in the current shard without worrying about
          # its size.
          total_size = self._dtype_size = 0
        elif self._dtype == dtypes.variant:
          # Can't determine a variant's type, so can't calculate its size.
          total_size = self._dtype_size = 0
        elif (self._dtype == dtypes.string
              and not context.executing_eagerly()
              and ops.get_default_session() is None):
          # TODO(b/326287351): Get string tensor size in tf.function.
          total_size = self._dtype_size = 0
          if not string_size_warning_printed:
            logging.warning("The checkpoint sharding policy is being executed "
                            "in a tf.function. The size of the string/variant "
                            "constant cannot be obtained.")
            string_size_warning_printed = True
        elif self._dtype == dtypes.string:
          with ops.device(self._device):
            if not context.executing_eagerly():
              self._root_tensor = ops.get_default_session().run(
                  self._root_tensor)

            if self._root_shape.rank is None or self._root_shape.rank == 0:
              sizes = [string_ops.string_length(self._root_tensor,
                                                unit="BYTE")]
            else:
              sizes = [string_ops.string_length(elem, unit="BYTE")
                       for elem in self._root_tensor]

            if context.executing_eagerly():
              sizes = [size.numpy() for size in sizes]
            else:
              sizes = ops.get_default_session().run(sizes)

          total_size = sum(sizes)
          self._dtype_size = max(sizes)

        if self._root_shape.rank is None or self._root_shape.rank == 0:
          if total_size > self.max_shard_size:
            logging.warning(
                "Tensor %s is a %s scalar of size %s bytes and cannot be "
                "partitioned into a shard of max shard size %s bytes. It will "
                "be added as an individual shard that exceeds the max shard "
                "size.", self._checkpoint_key, self._dtype, total_size,
                self.max_shard_size)
            self._large_scalars.append(
                {self._checkpoint_key: {self._slice_spec: self._root_tensor}})
          else:
            if total_size > self._shard_size_remaining:
              self._tensors_by_shard.append({})
              self._shard_size_remaining = self.max_shard_size
            (self._tensors_by_shard[-1]
             .setdefault(self._checkpoint_key, {})
             [self._slice_spec]) = self._root_tensor
            self._shard_size_remaining -= total_size
          continue

        # Partition tensor and add partitions to shards.
        self._working_tensor_offset = self._root_offset[:]
        self._working_tensor_shape = self._root_shape
        working_tensor_size = total_size
        while working_tensor_size > self._shard_size_remaining:
          (part_axis, part_size) = self._get_next_partition()

          if part_size == 0:
            # Tensor partition couldn't fit in remaining shard space. Try again
            # with the next full shard.
            self._tensors_by_shard.append({})
            self._shard_size_remaining = self.max_shard_size
          else:
            self._add_partition(part_axis, part_size)

            working_tensor_size = (
                int(math.prod(self._working_tensor_shape)) * self._dtype_size)

        if self._working_tensor_shape.num_elements() > 0:
          if self._working_tensor_offset and self._working_tensor_shape:
            with ops.device(self._device):
              working_tensor = array_ops.slice(
                  self._root_tensor,
                  begin=list(map(
                      operator.sub,
                      self._working_tensor_offset, self._root_offset)),
                  size=self._working_tensor_shape.as_list())
          else:
            working_tensor = self._root_tensor
          remaining_tensor_slice_spec = variables.Variable.SaveSliceInfo(
              full_name=self._checkpoint_key,
              full_shape=self._full_shape,
              var_offset=self._working_tensor_offset,
              var_shape=self._working_tensor_shape).spec.strip()
          (self._tensors_by_shard[-1]
           .setdefault(self._checkpoint_key, {})
           [remaining_tensor_slice_spec]) = working_tensor
        self._shard_size_remaining -= working_tensor_size

      shards = []
      if self._tensors_by_shard[0]:
        shards.extend(self._tensors_by_shard)
      shards.extend(self._large_scalars)

      return shards

  def __init__(self, max_shard_size: int):
    # [FUNCTION DOC]
    # Input: max_shard_size, integer maximum shard size in bytes. Stores it. Returns nothing.
    self.max_shard_size = max_shard_size

  @property
  def description(self) -> str:
    # [FUNCTION DOC]
    # Returns the fixed human-readable description string of this policy.
    return "Split tensors into shards with a max shard size."

  def __call__(
      self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
  ) -> Sequence[sharding_util.Shard]:
    # [FUNCTION DOC]
    # Input: shardable_tensors, list of shardable tensor records. Creates a new partitioner and
    # returns the list of shards it produces using this policy's maximum shard size.
    return self.MaxShardSizePartitioner().get_shards(
        self.max_shard_size, shardable_tensors)
