# [MODULE DOC]
# Defines an experimental MirroredStrategy built on the low-level DTensor API, used to
# validate the DTensor design and expose feature gaps. Exposes one class, the strategy, which
# wraps a DTensor strategy extended object over a one-dimensional batch mesh. Related to the
# sibling DTensor strategy extended and DTensor utility modules. No import-time side effects.
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
"""Implement a MirroredStrategy based on the DTensor low level API.

This is an experiment to validate the viability of the DTensor API, and expose
any potential feature gaps between the current API and the need.
"""

from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute.experimental import dtensor_strategy_extended
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import device as tf_device


class MirroredStrategy(distribute_lib.Strategy):
  # [CLASS DOC]
  # Experimental single-machine synchronous data-parallel strategy built on the DTensor API. It
  # subclasses the generic distribution strategy and delegates all work to a DTensor strategy
  # extended object created from a one-dimensional mesh. Holds the mesh (mesh property) and the
  # device list given at construction (which stays null when a mesh or no devices were given).
  # Created directly by users with either a device list or a mesh, or neither.
  """Synchronous training across multiple replicas on one machine.

  This strategy is typically used for training on one machine with multiple
  accelerators (GPUs/TPUs).

  For example, a variable created under a `MirroredStrategy` is a distributed
  variable with layout replicated on each dimension. The variables will be
  placed on the `mesh` that is specified in the __init__.
  """

  def __init__(self, devices=None, cross_device_ops=None, *, mesh=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the cross_device_ops argument is accepted but never used (ignored, as the
    # docstring admits). Also, the stored devices attribute is the argument as passed, so it
    # stays null when devices were auto-detected or a mesh was supplied.
    # Builds the strategy. Parameters: devices, an optional list of device name strings;
    # cross_device_ops, ignored; mesh (keyword only), an optional one-dimensional DTensor mesh.
    # Validates that mesh and devices are not both given and that a mesh is one-dimensional
    # (raises a value error otherwise). If no mesh is given, one is built from the devices
    # (or from auto-detected accelerators, falling back to CPU). Then creates the DTensor
    # strategy extended object, calls the parent constructor with it and stores the mesh and
    # devices. Side effect: may initialize the accelerator system once for the device type.
    """Synchronous training across multiple replicas on one machine.

    Args:
      devices: a list of device strings, such as ['/gpu:0', '/gpu:1']. If both
        `mesh` and `devices` are None, all the available GPU/TPU will be used.
        If no accelerators are found, CPU is used.
      cross_device_ops: optional, a descendant of `CrossDeviceOps`. The value is
        ignored at the moment, and support will be added later.
      mesh: optional DTensor mesh for the computation. Note that either `mesh`
        or `devices` should be provided, and not both. The mesh should be 1D,
        and will be used to split the input data among that dimension.
    """
    self._validate_init_args(mesh, devices)
    if not mesh:
      mesh = self._build_mesh_from_device_list(devices)

    extended = dtensor_strategy_extended.DTensorStrategyExtended(
        container_strategy=self, mesh=mesh)
    super().__init__(extended)
    self._mesh = mesh
    self._devices = devices

  @classmethod
  def _validate_init_args(cls, mesh, devices):
    # [FUNCTION DOC]
    # HELPER: argument validation for the strategy constructor.
    # Class method taking an optional mesh and an optional device list. Raises a value error if
    # both are provided, or if a mesh is provided whose shape does not have exactly one
    # dimension (the single dimension is used to shard inputs). Returns nothing.
    if mesh and devices:
      raise ValueError('Mesh and devices can not be provided at the same time. '
                       f'received mesh = {mesh}, devices = {devices}')

    # For mirrored strategy, the mesh should be 1D, and only contains a batch
    # dimension, we will use that dimension to shard the inputs.
    if mesh and len(mesh.shape()) != 1:
      raise ValueError('The mesh for MirroredStrategy must be 1D, received: '
                       f'{len(mesh.shape())}D')

  @classmethod
  def _build_mesh_from_device_list(cls, devices):
    # [FUNCTION DOC]
    # HELPER: mesh construction for the strategy constructor.
    # Class method taking an optional list of device name strings. If devices are given, the
    # device type is parsed from the first one, the accelerator system is initialized once for
    # that type, and a one-dimensional batch mesh over those devices is created. Otherwise it
    # detects the preferred device type and its local devices, initializes the accelerator
    # system once and creates a batch mesh sized to the number of local devices. Returns the mesh.
    if devices:
      device_type = tf_device.DeviceSpec.from_string(devices[0]).device_type
      dtensor_util.initialize_accelerator_system_once(device_type)
      mesh = mesh_util.create_mesh(
          mesh_dims=[(dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, len(devices))],
          devices=devices)
    else:
      # Trying to detect if there is any GPU/TPUs attached.
      device_type = d_config.preferred_device_type()
      devices = d_config.local_devices(device_type)
      dtensor_util.initialize_accelerator_system_once(device_type)
      mesh = mesh_util.create_mesh(
          mesh_dims=[(dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, len(devices))],
          device_type=device_type)
    return mesh

  def reduce(self, reduce_op, value, axis):
    # [FUNCTION DOC]
    # Overrides the strategy reduce. Reduces a value across replicas using the DTensor reduce
    # helper. Parameters: reduce_op, the reduction kind (sum or mean); value, the distributed
    # value to reduce; axis, the dimension to also reduce over within each replica (may be
    # null). Returns the reduced result.
    return dtensor_util.dtensor_reduce(self, reduce_op, value, axis)

  @property
  def mesh(self):
    # [FUNCTION DOC]
    # Read-only property returning the DTensor mesh the strategy computes on.
    """Returns the mesh used by the strategy."""
    return self._mesh
