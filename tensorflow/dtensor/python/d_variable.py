# [MODULE DOC]
# DTensor variable support. Defines DSaveSpec (a save specification that also records global shape
# and layout), a private saveable object describing how to checkpoint a DTensor variable, and the
# public DVariable class (exported under the experimental dtensor namespace), a resource variable
# whose placement follows its initial value and which can be saved as bfloat16. Used by the DTensor
# checkpoint module. Registers the export at import time; no other import-time side effects.

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
"""DTensor variable and saveable."""

import functools

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util.tf_export import tf_export


class DSaveSpec(saveable_object.SaveSpec):
  # [CLASS DOC]
  # Save specification for one DTensor tensor. Extends the standard save spec with two attributes:
  # global_shape (full unsharded shape) and layout (layout, as a string, of the value on the host
  # mesh). The DTensor saver and restorer inspect these to write and read sharded data.
  """DTensor SaveSpec that additionaly captures global_shape and layout."""

  def __init__(self,
               tensor,
               slice_spec,
               name,
               global_shape,
               layout,
               dtype=None,
               device=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: tensor (value or callable producing it); slice_spec (slice description);
    # name (checkpoint key); global_shape (unsharded shape); layout (layout string); dtype and device
    # (optional, default nothing). Passes tensor, slice spec, name, dtype and device to the parent
    # and stores global_shape and layout. Returns nothing.
    super().__init__(
        tensor=tensor,
        slice_spec=slice_spec,
        name=name,
        dtype=dtype,
        device=device)
    self.global_shape = global_shape
    self.layout = layout


class _DVariableSaveable(saveable_object.SaveableObject):
  # [CLASS DOC]
  # HELPER: internal saveable describing how a DVariable is saved to and restored from a
  # checkpoint. Holds the variable and its original layout. Saves the value copied to the host
  # mesh (optionally cast to bfloat16) with a DSaveSpec; restores by copying back to the original
  # layout and assigning. Created through the variable's checkpoint saveable hook.
  """Class for defining how to save/restore DTensor variable."""

  def __init__(self, dvariable, name):
    # [FUNCTION DOC]
    # Constructor. Inputs: dvariable (the DVariable to save); name (checkpoint key). Records the
    # variable's current layout and the variable, builds the host-mesh layout with the same sharding
    # specs, and calls the parent with a single DSaveSpec whose tensor is a callable reading the
    # variable on the host mesh, whose slice spec and name are replicated string DTensors on the
    # host mesh (one copy per local device), whose global shape is the variable shape, layout is the
    # host layout string, dtype is bfloat16 if saving as bfloat16 applies else the variable dtype,
    # and device is the variable device. Returns nothing.
    with ops.device(dvariable.device):
      original_layout = api.fetch_layout(dvariable)
    # Record original layout to allow restore.
    self._original_layout = original_layout
    self._dvariable = dvariable

    def pack(tensors, layout):
      # [FUNCTION DOC]
      # HELPER: packs component tensors into a DTensor with the given layout while placing the pack
      # on the variable's device. Inputs: tensors (list of per-device components); layout. Returns
      # the packed DTensor.
      with ops.device(dvariable.device):
        return api.pack(tensors, layout)

    host_layout = layout_lib.Layout(original_layout.sharding_specs,
                                    original_layout.mesh.host_mesh())

    def get_host_dtensor():
      # [FUNCTION DOC]
      # HELPER: callable used as the save tensor; reads the variable and returns it on the host
      # mesh. If the variable's mesh is not CPU, it is moved to the host layout (unpack then pack in
      # eager mode, copy-to-mesh in graph mode); otherwise read directly. Casts to bfloat16 when
      # the variable is set to save as bfloat16 and is float32. Takes no inputs.
      # Copy to host mesh if needed.
      if original_layout.mesh.device_type().upper() != 'CPU':
        # Prefer pack and unpack in eager mode because it supports sharded
        # layouts.
        if context.executing_eagerly():
          host_dtensor = api.pack(
              api.unpack(dvariable.read_value()), host_layout)
        else:
          host_dtensor = api.copy_to_mesh(dvariable.read_value(), host_layout)
      else:
        host_dtensor = dvariable.read_value()
      return (math_ops.cast(host_dtensor, dtypes.bfloat16)
              if self.should_cast(host_dtensor) else host_dtensor)

    num_local_devices = original_layout.mesh.num_local_devices()
    super(_DVariableSaveable, self).__init__(
        None,
        [
            DSaveSpec(
                tensor=get_host_dtensor,
                slice_spec=pack([''] * num_local_devices,
                                layout_lib.Layout.replicated(
                                    original_layout.mesh.host_mesh(), rank=0)),
                name=pack([name] * num_local_devices,
                          layout_lib.Layout.replicated(
                              original_layout.mesh.host_mesh(), rank=0)),
                global_shape=dvariable.shape,
                # Layout is attached as attribute, no need to put it as a
                # Tensor on DTensorDevice.
                layout=host_layout.to_string(),
                dtype=dtypes.bfloat16
                if self.should_cast(dvariable) else dvariable.dtype,
                device=dvariable.device)
        ],
        name)

  def should_cast(self, v):
    # [FUNCTION DOC]
    # Tells whether a value should be cast to bfloat16 when saving: true only if the owning
    # variable has bfloat16 saving enabled and the given value has 32-bit float dtype. Input: v, the
    # tensor or variable to test. Returns a boolean.
    """Returns True if v has float32 dtype and is intructed to save as bf16.

    Args:
      v : The variable that determines whether to cast.

    Returns:
      True if current savable DVariable is instructed to save as bfloat16 and
        the variable has dtype float32.
    """
    return self._dvariable.save_as_bf16 and v.dtype == dtypes.float32

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # Restores the variable from a loaded tensor. Inputs: restored_tensors (sequence holding exactly
    # one tensor, on CPU/host); restored_shapes (ignored). If the variable's mesh is not CPU, the
    # tensor is first copied to the original layout inside a traced function so the layout can be
    # tracked. Assigns the value to the variable, casting back to the variable dtype if it was saved
    # as bfloat16. Returns the assignment result.
    """Restores the same value into all variables."""
    tensor, = restored_tensors

    @def_function.function
    def _restore(t):
      # [FUNCTION DOC]
      # HELPER: traced function that copies a restored tensor to the variable's original layout on
      # the variable's device. Input: t, the restored tensor. Returns the resulting DTensor.
      with ops.device(self._dvariable.device):
        return api.copy_to_mesh(t, self._original_layout)

    # This assign establishes connections from restored tensor and tensors
    # being restored to -- so that restore in SPMD can backtrack the DVariable
    # and its layout, given that we're using tf.function style restore.
    # Note that the restored dvaraible is on CPU no matter what as the restoreV2
    # op must run on CPU.
    # TODO(b/159035705): Allow restore for Tensor objects as well?
    # Restore the dvariable back to original layout.
    if self._original_layout.mesh.device_type().upper() != 'CPU':
      tensor = _restore(tensor)
    return self._dvariable.assign(
        math_ops.cast(tensor, dtype=self._dvariable.dtype) if self._dvariable
        .save_as_bf16 else tensor)


@tf_export('experimental.dtensor.DVariable', v1=[])
class DVariable(resource_variable_ops.ResourceVariable):
  # [CLASS DOC]
  # Resource variable for DTensor. Unlike a plain variable, its placement follows the initial
  # value (the DTensor device when that is a DTensor) and it records a layout attribute (eager
  # mode). Supports optionally saving float32 values as bfloat16 and provides the DTensor-aware
  # checkpoint saveable. Falls back to ordinary behavior if the initial value is not a DTensor but
  # a layout is given (it is relaid out). Exported publicly under the experimental dtensor namespace.
  """A replacement for tf.Variable which follows initial value placement.

    The class also handles restore/save operations in DTensor. Note that,
    DVariable may fall back to normal tf.Variable at this moment if
    `initial_value` is not a DTensor.
  """

  def __init__(self, initial_value, *args, dtype=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: initial_value (tensor, or callable producing it, possibly wrapped in a
    # partial or a checkpoint initial value); extra positional arguments and keyword arguments passed
    # to the parent variable (dtype is a named optional argument). The optional keyword layout is
    # removed from the keyword arguments before calling the parent; shape is read from them.
    # If the initial value is a checkpoint-restore callable, shape and layout are required (else
    # a value error is raised) and the callable is invoked under the layout with shard info built
    # from the shape; any other callable is simply called. A wrapped checkpoint value is unwrapped,
    # then the value is converted to a tensor and the variable placed on its device.
    # Eager mode: if the value is a DTensor, its layout is used, raising an invalid-argument error if
    # a different layout was also given; otherwise if a layout is given the value is relaid out;
    # else an invalid-argument error is raised. Stores the layout attribute and builds the parent
    # variable under the default mesh of the layout. Graph mode: relays out if a layout is given and
    # builds the parent variable; the layout attribute is not recorded.
    # Sets the bfloat16-saving flag to false. Returns nothing.
    """Overrides tf.Variable to fix VarHandleOp placements."""
    # Variables by default use the current device scope for placement. This
    # wrapper has them follow the initial value's placement instead (which will
    # be the DTensor device if the initial value has a layout).

    # Pop layout from kwargs since keras make_variable may pass a 'layout'
    # keyword argument. We need to pop it because we are passing kwargs to
    # super class constructor.
    layout = kwargs.pop('layout', None)
    shape = kwargs.get('shape', None)

    if callable(initial_value):
      unwrapped = initial_value
      if issubclass(type(initial_value), functools.partial):
        unwrapped = initial_value.func

      # If wrapped is a CheckpointInitialValueCallable, this means that
      # we are creating a Variable during a checkpoint restore.
      # Thus the restore will happen now through this callable
      # and we will create the DVariable with the restored dtensor.
      if issubclass(type(unwrapped), trackable.CheckpointInitialValueCallable):
        if not shape or not layout:
          raise ValueError('Expected shape and layout to be not None.')

        # CheckpointInitialValueCallable will call an eager tf.RestoreV2,
        # which does not have any shape information or layout information
        # attached. Thus we will do two things to have them correctly specified:
        #
        # The default layout scope allows us to correctly specify the output
        # layout of the tf.RestoreV2 that will be called
        #
        # Passing shard_info with the correct shape allows the tf.RestoreV2
        # ShapeInference to extract the shape.
        initial_value = api.call_with_layout(
            initial_value,
            layout,
            shard_info=trackable.ShardInfo(
                shape=shape, offset=[0] * len(shape)))
      else:
        initial_value = initial_value()

    # When the initial value came from a Checkpoint restoration, fetch tensor.
    if isinstance(initial_value, trackable.CheckpointInitialValue):
      initial_value = initial_value.wrapped_value

    initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
    variable_device = initial_value.device
    self._save_as_bf16 = False
    # TODO(b/159035705): The following code enables variable creation inside
    # a tf.function. However, it requires a global dtensor device.
    # if not variable_device and not tf.executing_eagerly():
    #   try:
    #     initial_value.op.get_attr("_layout")
    #   except ValueError:
    #     pass
    #   else:
    #     # The initial value is a DTensor, but because the DTensor device is
    #     # only active during eager execution at the moment we need to
    #     # translate that into a placement for the eager VarHandleOp.
    #     variable_device = _dtensor_device().name
    with ops.device(variable_device):
      # If initial tensor assigned to DVariable is DTensor, record the layout of
      # the resource so that this can be queried.
      if context.executing_eagerly():
        if api.is_dtensor(initial_value):
          value_layout = api.fetch_layout(initial_value)
          if layout is not None and layout != value_layout:
            raise errors_impl.InvalidArgumentError(
                None,
                None,
                'Conflicting layout are provided for initial '
                f'value layout ({value_layout}) and variable ({layout}).',
            )
          layout = value_layout
        elif layout is not None:
          initial_value = api.relayout(initial_value, layout)
        else:
          raise errors_impl.InvalidArgumentError(
              None,
              None,
              'Neither layout nor DTensor initial value are provided.',
          )
        self.layout = layout
        with api.default_mesh(layout.mesh):
          super(DVariable, self).__init__(
              initial_value, *args, dtype=dtype, **kwargs
          )
      else:
        # FIXME(175928457): Record value layout in graph mode.
        if layout is not None:
          initial_value = api.relayout(initial_value, layout)
        super(DVariable, self).__init__(
            initial_value, *args, dtype=dtype, **kwargs)

  @property
  def save_as_bf16(self):
    # [FUNCTION DOC]
    # Property getter. Returns a boolean telling whether this variable is saved to checkpoints as
    # bfloat16 (false by default).
    return self._save_as_bf16

  @save_as_bf16.setter
  def save_as_bf16(self, save_as_bf16):
    # [FUNCTION DOC]
    # Property setter. Input: save_as_bf16, a boolean request. The stored flag becomes true only if
    # the request is true and the variable dtype is 32-bit float; for other dtypes it silently stays
    # false. Returns nothing.
    """Enables saving float32 as bfloat16."""
    self._save_as_bf16 = save_as_bf16 and self.dtype == dtypes.float32

  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # HELPER: checkpoint hook called by the trackable machinery. Returns a mapping from the standard
    # variable-value key to a factory (the variable-saveable class bound to this variable) that,
    # given a name, creates the DTensor-aware saveable for this variable.
    return {
        trackable.VARIABLE_VALUE_KEY:
            functools.partial(_DVariableSaveable, self)
    }
