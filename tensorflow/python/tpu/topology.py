# [MODULE DOC]
# Defines the Topology class (exported publicly as the experimental TPU topology type) which
# describes a TPU fabric: the mesh shape (x, y, z, core) and the mapping between TensorFlow
# task/device pairs and physical mesh coordinates, with serialization to and from the topology
# protocol buffer. Also has two private helpers that build TPU and host CPU device name strings.
# Import-time side effect: registers the class with the public API exporter.
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
"""Defines the `Topology` class, that describes a TPU fabric topology."""

import numpy as np

from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util import numpy_compat
from tensorflow.python.util.tf_export import tf_export


def _tpu_device_name(job, task, device):
  # [FUNCTION DOC]
  # HELPER: device-name builder for the Topology class.
  # Inputs: job name (string or null), task number, TPU device number. Returns a device string
  # of the form job/task/TPU-device, omitting the job part when job is null. Pure function.
  """Returns the device name for the TPU `device` on `task` of `job`."""
  if job is None:
    return "/task:%d/device:TPU:%d" % (task, device)
  else:
    return "/job:%s/task:%d/device:TPU:%d" % (job, task, device)


def _tpu_host_device_name(job, task):
  # [FUNCTION DOC]
  # HELPER: device-name builder for the Topology class.
  # Inputs: job name (string or null) and task number. Returns the host CPU device string
  # (device zero) for that task, omitting the job part when job is null. Pure function.
  """Returns the device name for the CPU device on `task` of `job`."""
  if job is None:
    return "/task:%d/device:CPU:0" % task
  else:
    return "/job:%s/task:%d/device:CPU:0" % (job, task)


@tf_export("tpu.experimental.Topology")
class Topology(object):
  # [CLASS DOC]
  # Immutable description of a TPU slice. Holds the mesh shape (rank 1 int32 array of 4 positive
  # sizes), the device coordinates array (tasks by devices by axes), inverse lookup arrays that
  # map a mesh coordinate to the task and device ordinal (minus one where no device exists), the
  # coordinates of missing devices, and a cached serialized proto. Built either from a
  # serialized proto or from a mesh shape plus device coordinates. Provides coordinate-to-task,
  # device ordinal and device-name queries and serialization.
  """Describes a set of TPU devices.

  Represents both the shape of the physical mesh, and the mapping between
  TensorFlow TPU devices to physical mesh coordinates.
  """

  def __init__(self, serialized=None, mesh_shape=None, device_coordinates=None):
    # [FUNCTION DOC]
    # Inputs: optional serialized topology proto bytes, optional mesh shape (4 positive
    # integers) and optional device coordinates (rank 3 integer array). If serialized is truthy
    # it is parsed and the other arguments ignored; otherwise the others are converted to 32-bit
    # integer arrays and validated. Then builds the inverse task/device maps and finds missing
    # devices (positions with no assigned device). Raises a value error for a malformed proto, a
    # bad mesh shape, or device coordinates that are not rank 3 with minor dimension equal to the
    # mesh rank. Returns nothing.
    # KNOWN ISSUE: the docstring has a stray backtick typo, and an empty serialized value is
    # treated as absent (truthiness check) rather than only null.
    """Builds a Topology object.

    If `serialized` is not `None`, the topology is parsed from `serialized` and
    the other arguments are ignored. Otherwise, the topology is computed from
    `mesh_shape` and `device_coordinates`.

    Args:
      serialized: A serialized `TopologyProto`, or `None`. If not `None`, the
        serialized proto is parsed to discover the topology.
      mesh_shape: A sequence of 4 positive integers, or `None`. If not `None`,
        the shape of the TPU topology, in number of cores. Ignored if
        `serialized` is not `None`.
      device_coordinates: A rank 3 numpy array that describes the mapping from
        TensorFlow TPU devices to TPU fabric coordinates, or `None`. If
        specified, array is a rank 3 int32 array with shape
        `[tasks, devices, axis]`.  `tasks` is the number of tasks in the TPU
        cluster, `devices` is the number of TPU devices per task, and `axis` is
        the number of axes in the TPU cluster topology. Each entry gives the
        `axis`-th coordinate in the topology of a task/device pair. TPU
        topologies are 4-dimensional, with dimensions `(x, y, z, core number)`.
        This arg is ignored if `serialized is not `None`.

    Raises:
      ValueError: If `serialized` does not describe a well-formed topology.
      ValueError: If `serialized` is `None` and `mesh_shape` is not a sequence
        of 4 positive integers.
      ValueError: If `serialized` is `None` and `device_coordinates` is not a
        rank 3 numpy int32 array that describes a valid coordinate mapping.
    """

    self._serialized = serialized

    if serialized:
      self._parse_topology(serialized)
    else:
      self._mesh_shape = numpy_compat.np_asarray(mesh_shape, dtype=np.int32)
      self._device_coordinates = numpy_compat.np_asarray(device_coordinates,
                                                         dtype=np.int32)
      if len(self._mesh_shape) != 4 or any(self._mesh_shape < 1):
        raise ValueError("`mesh_shape` must be a sequence of 4 positive "
                         f"entries; got `mesh_shape={self._mesh_shape}`")

      if (len(self._device_coordinates.shape) != 3 or
          self._device_coordinates.shape[2] != len(self._mesh_shape)):
        raise ValueError(
            "`device_coordinates` must be a rank 3 int32 array "
            "with minor dimension equal to the `mesh_shape` rank"
            "got device_coordinates.shape={} len(device_coordinates.shape)={} device_coordinates.shape[2]={} mesh_shape={}, len(mesh_shape)={}"
            .format(self._device_coordinates.shape,
                    len(self._device_coordinates.shape),
                    self._device_coordinates.shape[2], self._mesh_shape,
                    len(self._mesh_shape)))

    self._topology_tasks, self._topology_devices = self._invert_topology()

    # Coordinates of devices that are missing
    self._missing_devices = np.argwhere(self._topology_tasks < 0)

  def _parse_topology(self, serialized):
    # [FUNCTION DOC]
    # HELPER: constructor helper for the serialized path.
    # Parses the proto bytes and sets the mesh shape and device coordinates (reshaped to tasks by
    # devices per task by mesh rank) on the instance. Raises a value error if the mesh shape is
    # not 4 positive entries, task or per-task device counts are negative, the coordinate count
    # does not equal tasks times devices times mesh rank, or any coordinate is negative.
    # KNOWN ISSUE: in the size-mismatch message the "got shape" slot receives the coordinate
    # count and the mesh-rank slot receives the whole mesh shape, so the message is misleading.
    """Parses a serialized `TopologyProto` into `self`."""
    proto = topology_pb2.TopologyProto()
    proto.ParseFromString(serialized)

    self._mesh_shape = np.array(proto.mesh_shape, dtype=np.int32)
    if len(self._mesh_shape) != 4 or any(self._mesh_shape < 1):
      raise ValueError("`mesh_shape` must be a vector of size 4 with positive "
                       "entries; got {}".format(self._mesh_shape))

    if proto.num_tasks < 0:
      raise ValueError("`num_tasks` must be >= 0; got {}".format(
          proto.num_tasks))
    if proto.num_tpu_devices_per_task < 0:
      raise ValueError("`num_tpu_devices_per_task` must be >= 0; got {}".format(
          proto.num_tpu_devices_per_task))

    expected_coordinates_size = (
        proto.num_tasks * proto.num_tpu_devices_per_task * len(
            proto.mesh_shape))
    if len(proto.device_coordinates) != expected_coordinates_size:
      raise ValueError("`device_coordinates` must have shape num_tasks ({}) * "
                       "num_tpu_devices_per_task ({}) * len(mesh_shape) ({}); "
                       "got shape {}".format(proto.num_tasks,
                                             proto.num_tpu_devices_per_task,
                                             proto.mesh_shape,
                                             len(proto.device_coordinates)))

    coords = np.array(proto.device_coordinates, dtype=np.int32)
    if any(coords < 0):
      raise ValueError(
          "All values in `device_coordinates` must be >= 0, got {}"
          .format(coords))
    coords = coords.reshape((proto.num_tasks, proto.num_tpu_devices_per_task,
                             len(proto.mesh_shape)))
    self._device_coordinates = coords

  def _invert_topology(self):
    # [FUNCTION DOC]
    # HELPER: constructor helper building the reverse lookup.
    # Creates two integer arrays of the mesh shape filled with minus one, then for each
    # task/device pair writes its task number and device number at its mesh coordinates.
    # Returns the pair (task array, device array). Unassigned positions stay minus one.
    """Inverts a [task,device,axis] topology to [x,y,z] -> task/device maps."""
    tasks = np.full(list(self.mesh_shape), -1, dtype=np.int32)
    devices = np.full(list(self.mesh_shape), -1, dtype=np.int32)
    for task in range(self.device_coordinates.shape[0]):
      for device in range(self.device_coordinates.shape[1]):
        x, y, z, core = self.device_coordinates[task, device, :]
        tasks[x, y, z, core] = task
        devices[x, y, z, core] = device
    return tasks, devices

  @property
  def mesh_shape(self):
    # [FUNCTION DOC]
    # Property returning the rank 1 int32 array with the sizes of the 4 mesh dimensions.
    """A rank 1 int32 array describing the shape of the TPU topology."""
    return self._mesh_shape

  @property
  def mesh_rank(self):
    # [FUNCTION DOC]
    # Property returning the number of mesh dimensions (length of the mesh shape).
    """Returns the number of dimensions in the mesh."""
    return len(self._mesh_shape)

  @property
  def device_coordinates(self):
    # [FUNCTION DOC]
    # Property returning the rank 3 int32 array (tasks, devices per task, axes) of mesh
    # coordinates for each task/device pair.
    """Describes the mapping from TPU devices to topology coordinates.

    Returns:
      A rank 3 int32 array with shape `[tasks, devices, axis]`.
      `tasks` is the number of tasks in the TPU cluster, `devices` is the number
      of TPU devices per task, and `axis` is the number of axes in the TPU
      cluster topology. Each entry gives the `axis`-th coordinate in the
      topology of a task/device pair. TPU topologies are 4-dimensional, with
      dimensions `(x, y, z, core number)`.
    """
    return self._device_coordinates

  @property
  def missing_devices(self):
    # [FUNCTION DOC]
    # Property returning an array of mesh coordinates (one row each) that have no device assigned.
    """Array of indices of missing devices."""
    return self._missing_devices

  def task_ordinal_at_coordinates(self, device_coordinates):
    # [FUNCTION DOC]
    # Input: integer sequence of mesh coordinates (x, y, z, core). Returns the task number owning
    # the device there, or minus one if none. Raises an index error for out-of-range coordinates.
    """Returns the TensorFlow task number attached to `device_coordinates`.

    Args:
      device_coordinates: An integer sequence describing a device's physical
        coordinates in the TPU fabric.

    Returns:
      Returns the TensorFlow task number that contains the TPU device with those
      physical coordinates.
    """
    return self._topology_tasks[tuple(device_coordinates)]

  def tpu_device_ordinal_at_coordinates(self, device_coordinates):
    # [FUNCTION DOC]
    # Input: integer sequence of mesh coordinates. Returns the TPU device number within its task
    # at those coordinates, or minus one if none. Raises an index error if out of range.
    """Returns the TensorFlow device number at `device_coordinates`.

    Args:
      device_coordinates: An integer sequence describing a device's physical
        coordinates in the TPU fabric.

    Returns:
      Returns the TensorFlow device number within the task corresponding to
      attached to the device with those physical coordinates.
    """
    return self._topology_devices[tuple(device_coordinates)]

  def cpu_device_name_at_coordinates(self, device_coordinates, job=None):
    # [FUNCTION DOC]
    # Inputs: mesh coordinates and optional job name. Returns the host CPU device name string of
    # the task that owns the device at those coordinates.
    # KNOWN ISSUE: if no device sits at the coordinates the task is minus one and a name for task
    # minus one is returned without any error.
    """Returns the CPU device attached to a logical core."""
    return _tpu_host_device_name(
        job, self._topology_tasks[tuple(device_coordinates)])

  def tpu_device_name_at_coordinates(self, device_coordinates, job=None):
    # [FUNCTION DOC]
    # Inputs: mesh coordinates and optional job name. Returns the TPU device name string built
    # from the task and device ordinal at those coordinates.
    # KNOWN ISSUE: missing devices yield minus one ordinals in the name instead of an error.
    """Returns the name of the TPU device assigned to a logical core."""
    return _tpu_device_name(job,
                            self._topology_tasks[tuple(device_coordinates)],
                            self._topology_devices[tuple(device_coordinates)])

  @property
  def num_tasks(self):
    # [FUNCTION DOC]
    # Property returning the number of tasks (first dimension of the device coordinates).
    """Returns the number of TensorFlow tasks in the TPU slice."""
    return self._device_coordinates.shape[0]

  @property
  def num_tpus_per_task(self):
    # [FUNCTION DOC]
    # Property returning the number of TPU devices per task (second dimension of the device
    # coordinates).
    """Returns the number of TPU devices per task in the TPU slice."""
    return self._device_coordinates.shape[1]

  def serialized(self):
    # [FUNCTION DOC]
    # Returns the topology as serialized proto bytes. If none is cached, builds the proto from
    # the mesh shape, task and device counts and flattened coordinates, serializes it and caches
    # it on the instance (side effect). Takes no other input.
    """Returns the serialized form of the topology."""
    if self._serialized is None:
      proto = topology_pb2.TopologyProto()
      proto.mesh_shape[:] = list(self._mesh_shape)
      proto.num_tasks = self._device_coordinates.shape[0]
      proto.num_tpu_devices_per_task = self._device_coordinates.shape[1]
      proto.device_coordinates.extend(list(self._device_coordinates.flatten()))
      self._serialized = proto.SerializeToString()

    return self._serialized
