# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# Helpers that convert between ordinary numpy arrays / TensorFlow tensors and DTensor tensors.
# Exposes: to_numpy (gather a DTensor into one local array), unpacked_to_numpy (stitch per-device
# component arrays back together using a layout), unpack (slice a full value into the per-device
# pieces a layout requires), pack_numpy and pack_tf_tensor (slice then pack into a DTensor), and
# stateless_random_uniform (random tensor created with a given layout). Built on the dtensor api
# and layout modules. No import-time side effects other than wrapping one function as a graph
# function.
# KNOWN ISSUE: per the in-file note, these functions do not distinguish client-local data from the
# global view, so results are only correct in single-client settings.
"""Utilities to convert data buffers to/from DTensor tensors."""
from typing import List

import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.types.core import Tensor, TensorLike  # pylint: disable=g-multiple-import

# FIXME(b/262894693): Functions in this file are buggy.
# They do not distinguish between the client-local data and the global view.


def _split(value, splits, axis=0, split_fn=np.split, stack_fn=np.stack):
  # [FUNCTION DOC]
  # HELPER: recursive splitter used by unpack to shard a value along successive axes.
  # Splits value along the given axis into splits[0] equal pieces with split_fn; if more entries
  # remain in splits, recursively splits each piece along the next axis. Finally stacks the
  # pieces with stack_fn into one nested array whose leading dimensions equal the split counts.
  # Parameters: value (array or tensor), splits (list of integer split counts per axis), axis
  # (starting axis, default 0), split_fn and stack_fn (callables, default numpy split and stack).
  # KNOWN ISSUE: the recursive call does not forward split_fn or stack_fn, so nested levels always
  # use the numpy defaults even when tensor functions were requested.
  """Split `value` into a sharded nparray/tf tensor based on the number of splits.
  """
  children = split_fn(value, splits[0], axis=axis)
  if len(splits) > 1:
    splits = splits[1:]
    children = [_split(child, splits, axis + 1) for child in children]
  return stack_fn(children)


def to_numpy(tensor: TensorLike) -> np.ndarray:
  # [FUNCTION DOC]
  # Copies a DTensor into one local numpy array. Fetches the tensor's layout; if its mesh is
  # remote, returns an array holding a single null value. Otherwise unpacks the tensor into its
  # local components, converts each to numpy and stitches them with unpacked_to_numpy.
  # Parameter: tensor, a DTensor. Returns the assembled numpy array.
  """Copy `input` DTensor to an equivalent local numpy array."""
  layout = api.fetch_layout(tensor)
  if layout.mesh.is_remote():
    return np.array([None])

  unpacked = [tensor.numpy() for tensor in api.unpack(tensor)]
  return unpacked_to_numpy(unpacked, layout)


def unpacked_to_numpy(unpacked: List[TensorLike],
                      layout: layout_lib.Layout) -> np.ndarray:
  # [FUNCTION DOC]
  # Reassembles per-device component arrays into a single array. Places each component into a
  # grid (one dimension per layout rank, sized by the shard counts) at the shard location the
  # layout assigns to its device offset, concatenates the grid as nested blocks, then removes
  # leading size-one dimensions introduced by block concatenation until the rank matches the
  # components.
  # Parameters: unpacked (list of local components, one per local device), layout (the Layout).
  # Raises a value error if the number of components differs from the layout's shard offsets.
  """Heals local Tensor components to a numpy array."""
  if len(unpacked) != len(layout.offset_to_shard()):
    raise ValueError('Wrong number of component Tensors.')

  unravelled = np.ndarray([layout.num_shards(i) for i in range(layout.rank)],
                          dtype=object)

  for offset, loc in enumerate(layout.offset_to_shard()):
    unravelled[loc] = unpacked[offset]

  concat_tensor = np.block(unravelled.tolist())

  # np.block can introduce empty initial dimensions, peel these off until
  # the output matches the rank of the input tensors.
  while concat_tensor.ndim > unpacked[0].ndim:
    concat_tensor = np.squeeze(concat_tensor, axis=0)
  return concat_tensor


# TODO(feyu): rename to slice.
def unpack(t: TensorLike,
           layout: layout_lib.Layout,
           split_fn=np.split,
           stack_fn=np.stack) -> List[TensorLike]:
  # [FUNCTION DOC]
  # Slices a full value into the flat list of per-device pieces that the layout requires, in the
  # order suitable for packing. For a rank-zero layout, returns the value repeated once per mesh
  # device. Otherwise shards the value by the layout's per-dimension shard counts and picks, for
  # every device offset, the piece at that device's shard location.
  # Parameters: t (array or tensor), layout, split_fn and stack_fn (callables; defaults numpy).
  # Returns a list with one entry per mesh device.
  """Slice `t` into a flattened list of tensors suitable for `pack`."""
  if not layout.rank:
    return [t] * layout.mesh.size
  sharded_tensor = _split(
      t, [layout.num_shards(i) for i in range(layout.rank)],
      split_fn=split_fn,
      stack_fn=stack_fn)
  flattened = [np.ndarray([])] * layout.mesh.size
  for offset, shard in enumerate(layout.offset_to_shard()):
    flattened[offset] = sharded_tensor[tuple(shard)]
  return flattened


def pack_numpy(value: np.ndarray,
               layout: layout_lib.Layout,
               make_sparse: bool = False) -> Tensor:
  # [FUNCTION DOC]
  # Packs a numpy array into a DTensor with the given layout. Slices the value per device, then
  # optionally converts each slice to a sparse tensor (when make_sparse is true, default false)
  # before packing through the dtensor api.
  # Parameters: value (numpy array, must not be null), layout, make_sparse (boolean).
  # Raises an assertion error if value is null. Returns the packed DTensor.
  assert value is not None
  unpacked = unpack(value, layout)
  if make_sparse:
    return api.pack([sparse_ops.from_dense(t) for t in unpacked], layout)
  return api.pack(unpacked, layout)


def pack_tf_tensor(value: Tensor, layout: layout_lib.Layout) -> Tensor:
  # [FUNCTION DOC]
  # Packs a TensorFlow tensor into a DTensor with the given layout, slicing it per device with
  # TensorFlow split and stack operations (top-level only; see unpack/_split issue).
  # Parameters: value (tensor), layout. Raises a value error if value is null.
  # Returns the packed DTensor.
  if value is None:
    raise ValueError('pack requires values to be passed in')
  unpacked = unpack(
      value, layout, split_fn=array_ops.split, stack_fn=array_ops_stack.stack)
  return api.pack(unpacked, layout)


@polymorphic_function.function
def stateless_random_uniform(shape, seed, layout):
  # [FUNCTION DOC]
  # Graph function that creates a deterministic uniform random tensor of the given shape from a
  # seed pair, then relayouts it to the requested layout.
  # Parameters: shape (integer shape), seed (two-element seed), layout (target Layout).
  # Returns the DTensor with that layout.
  """Creates uniform random tensor with the given layout."""
  return api.relayout(
      stateless_random_ops.stateless_random_uniform(shape=shape, seed=seed),
      layout=layout,
  )
