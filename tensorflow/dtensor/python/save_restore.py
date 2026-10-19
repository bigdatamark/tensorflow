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
# Checkpoint support for DTensor. Exposes sharded_save (multi-client safe save with barriers and
# checkpoint merge), enable_save_as_bf16 (flag DTensor variables to save as bfloat16),
# name_based_restore and name_based_save (restore/save dictionaries of named DTensors on one mesh),
# all exported under the experimental dtensor API. Uses the dtensor api, mesh utilities (for
# barriers), generated dtensor ops and TF io ops. Import-time side effects: API registrations only.
"""Contains functionaility for Checkpoint/SavedModel in DTensor."""

import collections
from typing import Dict, List, Union

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.eager import context
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util.tf_export import tf_export


@tf_export('experimental.dtensor.sharded_save', v1=[])
def sharded_save(
    mesh: layout_lib.Mesh,
    file_prefix: Union[str, tensor_lib.Tensor],
    tensor_names: Union[List[str], tensor_lib.Tensor],
    shape_and_slices: Union[List[str], tensor_lib.Tensor],
    tensors: List[Union[tensor_lib.Tensor, tf_variables.Variable]],
):
  # [FUNCTION DOC]
  # Saves named tensors in a multi-client safe way. Runs the save op on the DTensor device, waits
  # at a barrier on the mesh's host mesh so every client finished writing, merges the sharded
  # checkpoint files under the host mesh (destination equals the prefix, old directories deleted),
  # waits at a second barrier, and returns the merge op.
  # Parameters: mesh, file_prefix (string or tensor), tensor_names, shape_and_slices (only empty
  # strings supported), tensors (aligned with names). Returns the merge operation.
  """Saves given named tensor slices in a sharded, multi-client safe fashion.

  The method makes sure the checkpoint directory state is correct in a sharded
  mutli-client saving. Namely, we place a barrier after SaveV2 to make sure
  every client has done writing the files. And another one after
  MergeV2Checkpoints to make sure all Metadata is properly merged.

  Upon existing, the checkpoint is completed and the all directory operations
  are done.

  Args:
    mesh: The Mesh that contains the Tensors to save.
    file_prefix: The prefix of checkpoint.
    tensor_names: a list of tensor names used in save op.
    shape_and_slices: a list of shape and slice specification used in save op.
      The only supported value is "" as we don't support distributed saving with
      slices yet.
    tensors: a list of tensors used in save op. The order should match
      tensor_names.

  Returns:
    A MergeV2Checkpoints op that merged all Metadata.
  """
  with ops.device(api.device_name()):
    io_ops.save_v2(file_prefix, tensor_names, shape_and_slices, tensors)

  # Make sure all clients have written the files
  mesh_util.barrier(mesh.host_mesh(), 'SaveV2')  # pylint: disable=protected-access

  with api.default_mesh(mesh.host_mesh()):
    merge_op = io_ops.MergeV2Checkpoints(
        checkpoint_prefixes=[file_prefix],
        destination_prefix=file_prefix,
        delete_old_dirs=True)

  # Make sure first device in first host has finished merge.
  mesh_util.barrier(mesh.host_mesh(), 'MergeV2Checkpoints')

  return merge_op


@tf_export('experimental.dtensor.enable_save_as_bf16', v1=[])
def enable_save_as_bf16(variables: List[tf_variables.Variable]):
  # [FUNCTION DOC]
  # Marks each DTensor variable in the list so it is saved and restored as bfloat16 by setting its
  # save-as-bfloat16 flag. Other variables are ignored. Mutates the variables; returns nothing.
  """Allows float32 DVariables to be checkpointed and restored as bfloat16.

  The method only affects the DVariable part inside the model and leaves
  non-DTensor Variables/Tensors untouched.

  Args:
    variables: A list of tf.Variable to be enabled with bfloat16 save/restore.
      Only has effect on DTensor Variables as they go through d_variables with
      DTensor Specific logis.
  """
  for v in variables:
    if isinstance(v, d_variable.DVariable):
      v.save_as_bf16 = True


@tf_export('experimental.dtensor.name_based_restore', v1=[])
def name_based_restore(
    mesh: layout_lib.Mesh,
    checkpoint_prefix: str,
    name_tensor_dict: Dict[
        str, Union[tensor_lib.Tensor, tf_variables.Variable]],
):
  # [FUNCTION DOC]
  # Restores named DTensors from a checkpoint. Must run eagerly (else value error). Converts the
  # input mapping to an ordered mapping, checks each tensor is a DTensor on a CPU mesh (value
  # error otherwise, naming the tensor), packs the prefix, names and empty slice specs onto the
  # host mesh, and runs the DTensor restore op with each tensor's shape, layout and dtype.
  # Parameters: mesh, checkpoint_prefix, name_tensor_dict (names to existing DTensors that supply
  # shape, dtype and layout). Returns an ordered mapping from names to restored DTensors.
  """Restores from checkpoint_prefix to name based DTensors.

  It is required to have already-initialized DTensor variables that have same
  shape/dtype for the tensors being restored.

  Also, we currently only support a named based restore on a single mesh.

  Args:
    mesh: The single mesh that all Tensors would be restored to.
    checkpoint_prefix : The prefix of checkpoint to be restored.
    name_tensor_dict: A ordered dictionary of tensor_names to a DTensor. The
      DTensor shape/dtype must match the tensors being saved/restored for now.

  Returns:
    A dictionary of name to its restored DTensor value.
  """
  if not context.executing_eagerly():
    raise ValueError('name based restore must run eagerly.')

  ordered_name_tensor_dict = name_tensor_dict
  if not isinstance(name_tensor_dict, collections.OrderedDict):
    ordered_name_tensor_dict = collections.OrderedDict(name_tensor_dict)

  # Make sure that all tensors are on CPU mesh for now.
  # This might not be a hard limitation in the future.
  for name, tensor in ordered_name_tensor_dict.items():
    try:
      if api.fetch_layout(tensor).mesh.device_type().upper() != 'CPU':
        raise ValueError(
            'Restoring a non CPU Tensor is not supported currently. Offending '
            'tensor name : {tensor_name}'.format(tensor_name=name))
    except errors_impl.OpError as op_error:
      raise ValueError(
          'Saving/Restoring tensor must be a DTensor') from op_error

  # Now that we have all tensors on CPU mesh, do a DTensorRestoreV2.
  checkpoint_prefix = api.pack(
      [checkpoint_prefix] * mesh.num_local_devices(),
      layout_lib.Layout.replicated(mesh.host_mesh(), rank=0))
  # Explicitly pack to mesh to avoid implicit small constant extraction, which
  # does not work larger restores that has lots of names.
  tensor_names = api.pack(
      [list(ordered_name_tensor_dict.keys())] * mesh.num_local_devices(),
      layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
  shape_and_slices = api.pack(
      [[''] * len(ordered_name_tensor_dict)] * mesh.num_local_devices(),
      layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
  # A list of TensorShape representing all shapes for the input tensors.
  input_shapes = [tensor.shape for tensor in ordered_name_tensor_dict.values()]
  input_layouts = [
      api.fetch_layout(tensor).to_string()
      for tensor in ordered_name_tensor_dict.values()
  ]

  with ops.device(api.device_name()):
    restored_cpu_tensors = gen_dtensor_ops.d_tensor_restore_v2(
        prefix=checkpoint_prefix,
        tensor_names=tensor_names,
        shape_and_slices=shape_and_slices,
        input_shapes=input_shapes,
        input_layouts=input_layouts,
        dtypes=[tensor.dtype for tensor in ordered_name_tensor_dict.values()],
    )

  return collections.OrderedDict(
      zip(ordered_name_tensor_dict.keys(), restored_cpu_tensors)
  )


@tf_export('experimental.dtensor.name_based_save', v1=[])
def name_based_save(
    mesh: layout_lib.Mesh,
    checkpoint_prefix: Union[str, tensor_lib.Tensor],
    name_tensor_dict: Dict[
        str, Union[tensor_lib.Tensor, tf_variables.Variable]],
):
  # [FUNCTION DOC]
  # Saves a mapping of names to DTensors into a checkpoint using the sharded save. Must run
  # eagerly (else value error). Converts to an ordered mapping, packs the prefix and the list of
  # names onto the host mesh, then calls the sharded save with empty slice specs and the tensors
  # in order. Parameters: mesh, checkpoint_prefix, name_tensor_dict. Returns nothing.
  # KNOWN ISSUE: the docstring describes mesh and prefix as restore targets, but this function
  # saves; the sharded save's merge op result is discarded.
  """Saves name based Tensor into a Checkpoint.

  The function prepares the input dictionary to the format of a `sharded_save`,
  so that it can take advantage of DTensor SPMD based distributed save.

  Same as restore, the function only supports saving on the single mesh.

  Args:
    mesh: The single mesh that all Tensors would be restored to.
    checkpoint_prefix : The prefix of checkpoint to be restored.
    name_tensor_dict: A ordered dictionary of tensor_names to a DTensor. The
      DTensor shape/dtype must match the tensors being saved/restored for now.
  """
  if not context.executing_eagerly():
    raise ValueError('name based save must run eagerly.')

  ordered_name_tensor_dict = name_tensor_dict
  if not isinstance(name_tensor_dict, collections.OrderedDict):
    ordered_name_tensor_dict = collections.OrderedDict(name_tensor_dict)

  # Current _dtensor_device() in api.py is the correct way of specifying
  # DTensor device singletons. The API itself will be eventually be moved to
  # a public API and provides global singleton in DTensor context.
  # For now, we just use the current `internal` API and aim at migrating in
  # one shot later.
  # TODO(hthu): Provide _dtensor_device() singleton as a public API.
  # pylint: disable=protected-access
  checkpoint_prefix = api.pack([checkpoint_prefix] * mesh.num_local_devices(),
                               layout_lib.Layout.replicated(
                                   mesh.host_mesh(), rank=0))
  tensor_names = api.pack(
      [list(ordered_name_tensor_dict.keys())] * mesh.num_local_devices(),
      layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))

  sharded_save(
      mesh,
      file_prefix=checkpoint_prefix,
      tensor_names=tensor_names,
      shape_and_slices=[''] * len(ordered_name_tensor_dict),
      tensors=list(ordered_name_tensor_dict.values()))
