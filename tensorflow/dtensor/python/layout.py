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
# Python definitions of the two core DTensor concepts. Mesh describes named dimensions over an
# arrangement of devices (global and local device ids and device specs); Layout describes how a
# tensor is partitioned over a mesh (one sharding spec per tensor axis, each naming a mesh
# dimension or the special unsharded or match value). Both subclass native (pybind) classes and
# mostly adapt argument types, with a few helper computations kept in Python. Also defines the
# constants for unsharded, match and the default XLA SPMD flag, the MeshDimension named pair, and
# the LayoutType alias. Import-time side effects: exports the unsharded and match constants and the
# Mesh and Layout classes into the public experimental dtensor API.
"""Python definitions for `Mesh` and `Layout`."""

import collections
import functools
import itertools
from typing import List, Dict, Optional, Union

import numpy as np

from tensorflow.dtensor.proto import layout_pb2
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.util.tf_export import tf_export

# UNSHARDED indicates a tensor dimension is not sharded over any mesh dimension.
UNSHARDED = 'unsharded'
MATCH = 'match'
USE_XLA_SPMD = False

tf_export(
    'experimental.dtensor.UNSHARDED',
    v1=[]).export_constant(__name__, 'UNSHARDED')
tf_export(
    'experimental.dtensor.MATCH', v1=[]).export_constant(__name__, 'MATCH')

MeshDimension = collections.namedtuple('MeshDimension', ['name', 'size'])


def _compute_mesh_strides(shape: List[int]) -> List[int]:
  # [FUNCTION DOC]
  # HELPER: stride computation for Mesh.strides.
  # Given a mesh shape (list of dimension sizes) returns the row-major strides: for each
  # dimension, the product of the sizes of all later dimensions (the last stride is one).
  strides = [1]
  for idx, dim_size in enumerate(reversed(shape[1:])):
    strides.append(strides[idx] * dim_size)
  strides.reverse()
  return strides


@tf_export('experimental.dtensor.Mesh', v1=[])
class Mesh(_pywrap_dtensor_device.Mesh):
  # [CLASS DOC]
  # A mesh of devices with named dimensions. Thin Python subclass of the native Mesh class that
  # validates and converts constructor arguments (device specs to strings, flattened ids and
  # shape) and adds convenience methods: global ids as an array, item access by dimension name,
  # hashing via the serialized proto, repr/pickle through the string form, coordinates, strides,
  # local device locations, index unravelling, and a cached host (CPU) mesh. Alternate
  # constructors build from a proto, a string, a single device or an existing native mesh.
  """Represents a Mesh configuration over a certain list of Mesh Dimensions.

  A mesh consists of named dimensions with sizes, which describe how a set of
  devices are arranged. Defining tensor layouts in terms of mesh dimensions
  allows us to efficiently determine the communication required when computing
  an operation with tensors of different layouts.

  A mesh provides information not only about the placement of the tensors but
  also the topology of the underlying devices. For example, we can group 8 TPUs
  as a 1-D array for data parallelism or a `2x4` grid for (2-way) data
  parallelism and (4-way) model parallelism.

  Refer to [DTensor Concepts](https://www.tensorflow.org/guide/dtensor_overview)
  for in depth discussion and examples.

  Note: the utilities `dtensor.create_mesh` and
  `dtensor.create_distributed_mesh` provide a simpler API to create meshes for
  single- or multi-client use cases.
  """

  def __init__(
      self,
      dim_names: List[str],
      global_device_ids: np.ndarray,
      local_device_ids: List[int],
      local_devices: List[Union[tf_device.DeviceSpec, str]],
      mesh_name: str = '',
      global_devices: Optional[List[Union[tf_device.DeviceSpec, str]]] = None,
      use_xla_spmd: bool = USE_XLA_SPMD,
  ):
    # [FUNCTION DOC]
    # Constructor. Validates inputs and then calls the native constructor. Parameters:
    # dim_names (list of strings, one per array dimension), global_device_ids (numpy array of
    # consecutive increasing integers, non-empty; its shape is the mesh shape), local_device_ids
    # (list of integers, same length as local_devices, not longer than the global ids),
    # local_devices (list of device specs or strings), mesh_name (default empty), global_devices
    # (optional list of specs or strings), use_xla_spmd (default from the module flag).
    # Raises value errors when: ids are not an array or empty or not consecutive; the number of
    # names differs from the array rank; local ids, local devices or global devices are not
    # lists; no devices at all; duplicate local devices (except a lone host CPU); size mismatch
    # between local ids and local devices; more local ids than global ids; missing or mixed
    # device types; or XLA SPMD requested for a non-TPU mesh.
    """Builds a Mesh.

    The `dim_names` and `global_device_ids` arguments describe the dimension
    names and shape for the mesh.

    For example,

    ```python
      dim_names = ('x', 'y'),
      global_device_ids = [[0, 1],
                           [2, 3],
                           [4, 5]]
    ```

    defines a 2D mesh of shape 3x2. A reduction over the 'x' dimension will
    reduce across columns (0, 2, 4) and (1, 3, 5), and a reduction over the 'y'
    dimension reduces across rows.

    Note: the utilities `dtensor.create_mesh` and
    `dtensor.create_distributed_mesh` provide a simpler API to create meshes for
    single- or multi-client use cases.

    Args:
      dim_names: A list of strings indicating dimension names.
      global_device_ids: An ndarray of global device IDs is used to compose
        DeviceSpecs describing the mesh. The shape of this array determines the
        size of each mesh dimension. Values in this array should increment
        sequentially from 0. This argument is the same for every DTensor client.
      local_device_ids: A list of local device IDs equal to a subset of values
        in global_device_ids. They indicate the position of local devices in the
        global mesh. Different DTensor clients must contain distinct
        local_device_ids contents. All local_device_ids from all DTensor clients
        must cover every element in global_device_ids.
      local_devices: The list of devices hosted locally. The elements correspond
        1:1 to those of local_device_ids.
      mesh_name: The name of the mesh. Currently, this is rarely used, and is
        mostly used to indicate whether it is a CPU, GPU, or TPU-based mesh.
      global_devices (optional): The list of global devices. Set when multiple
        device meshes are in use.
      use_xla_spmd (optional): Boolean when True, will use XLA SPMD instead of
        DTensor SPMD.
    """
    # Check if input args are valid.
    if not isinstance(global_device_ids, np.ndarray):
      raise ValueError('Variable global_device_ids must be an ndarray.')
    if global_device_ids.size == 0:
      raise ValueError('Variable global_device_ids must be non-empty.')
    flat_global_device_ids = global_device_ids.flatten()
    # global_device_ids are expected to be consecutive numbers.
    # LINT.IfChange
    distance = flat_global_device_ids[0]
    if any(
        (gid - i != distance) for i, gid in enumerate(flat_global_device_ids)):
      raise ValueError('global_device_ids must sequentially increase: %s' %
                       global_device_ids)
    # LINT.ThenChange(//tensorflow/dtensor/cc/dtensor_device.cc)

    # TODO(b/242201545): This class is only for args type transformation for
    # exported C++ Mesh class after the unification is complete. Any other
    # logics should reside in the C++ layer, including validation checks, shall
    # go to C++.

    if len(dim_names) != global_device_ids.ndim:
      raise ValueError(
          'Number of mesh dimensions does not match number of dimension names.')

    if not isinstance(local_device_ids, list):
      raise ValueError('Variable local_device_ids must be a list of integers.')

    if not isinstance(local_devices, list):
      raise ValueError('Variable local_devices must be a list of DeviceSpecs.')

    if global_devices and not isinstance(global_devices, list):
      raise ValueError('Variable global_devices must be a list of DeviceSpecs.')

    if not local_devices and not global_devices:
      raise ValueError('Empty list of devices not allowed.')

    # Transform args format for C++ Mesh constructor
    global_device_ids_flatten = global_device_ids.flatten()
    global_device_ids_shape = global_device_ids.shape

    def to_str(d) -> str:
      # [FUNCTION DOC]
      # HELPER: normalizes a device (spec object or string) to its string form.
      if isinstance(d, tf_device.DeviceSpec):
        return d.to_string()
      return d

    def to_spec(d) -> tf_device.DeviceSpec:
      # [FUNCTION DOC]
      # HELPER: normalizes a device (spec object or string) to a device spec object.
      if not isinstance(d, tf_device.DeviceSpec):
        return tf_device.DeviceSpec.from_string(d)
      return d

    local_devices_str = [to_str(d) for d in local_devices]
    local_devices_spec = [to_spec(d) for d in local_devices]
    if not global_devices:
      global_devices = []
    global_devices_str = [to_str(d) for d in global_devices]
    global_devices_spec = [to_spec(d) for d in global_devices]

    local_devices_set = set(local_devices_spec)
    local_device_only_contains_host_cpu = (
        len(local_devices_set) == 1 and
        list(local_devices_set)[0].device_type == 'CPU')
    if not local_device_only_contains_host_cpu and len(local_devices) != len(
        local_devices_set):
      raise ValueError('Duplicate devices found in mesh specification %s.' %
                       [d for d in local_devices if local_devices.count(d) > 1])

    if len(local_device_ids) != len(local_devices):
      raise ValueError(
          'Variable local_device_ids does not have same size as local_devices.')

    if len(local_device_ids) > len(np.ravel(global_device_ids)):
      raise ValueError('Cannot have more local than gobal device IDs.')

    device_types = set([device.device_type for device in local_devices_spec])
    if not device_types:
      device_types = set([device.device_type for device in global_devices_spec])
    if None in device_types:
      raise ValueError('device_type is required')
    if len(device_types) > 1:
      raise ValueError('Devices containing multiple device_types : %s' %
                       device_types)
    device_type = device_types.pop()
    if use_xla_spmd and device_type != 'TPU':
      raise ValueError('XLA SPMD is not currently not supported for %s mesh.' %
                       device_type)

    super().__init__(
        mesh_name,
        dim_names,
        global_device_ids_shape,
        global_device_ids_flatten,
        global_devices_str,
        local_device_ids,
        local_devices_str,
        use_xla_spmd,
    )

  @classmethod
  def _new_object(cls, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: factory shared by the alternate constructors of Mesh.
    # Creates an instance without running the Python constructor and initializes it directly
    # with the native constructor using the given positional and keyword arguments. Returns it.
    # Need to explicitly invoke the base class __init__ because
    # Mesh.__init__ overrode it with a different signature.
    self = _pywrap_dtensor_device.Mesh.__new__(cls)
    super().__init__(self, *args, **kwargs)
    return self

  def global_device_ids(self) -> np.ndarray:
    # [FUNCTION DOC]
    # Returns the global device ids as a 64-bit integer array reshaped to the mesh shape.
    """Returns a global device list as an array."""
    return np.array(super().global_device_ids(), dtype=np.int64).reshape(
        self.shape()
    )

  def __getitem__(self, dim_name: str) -> MeshDimension:
    # [FUNCTION DOC]
    # Returns a name and size pair for the named mesh dimension (size from the native side).
    return MeshDimension(name=dim_name, size=self.dim_size(dim_name))

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash of the deterministic serialization of the mesh proto, so equal meshes hash equally.
    return hash(self.as_proto().SerializeToString(deterministic=True))

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns text that names the string constructor applied to the mesh's string form.
    return f'Mesh.from_string({self.to_string()})'

  # TODO(panzf): change to pybind11 pickle implementation in the last step
  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickle support: reconstruct by calling the string constructor with the mesh's string form.
    return Mesh.from_string, (self.to_string(),)

  # TODO(b/242201545): implement this in Mesh C++ class
  def coords(self, device_idx: int) -> tensor.Tensor:
    # [FUNCTION DOC]
    # Converts a device index (integer or tensor) into a tensor of mesh coordinates: integer
    # division by the strides, modulo the mesh shape, both converted to tensors. Returns a tensor.
    """Converts the device index into a tensor of mesh coordinates."""
    strides = ops.convert_to_tensor(self.strides)
    shape = ops.convert_to_tensor(self.shape())
    return (device_idx // strides) % shape

  @classmethod
  def from_proto(cls, proto: layout_pb2.MeshProto) -> 'Mesh':
    # [FUNCTION DOC]
    # Alternate constructor: builds a Mesh from a mesh protocol buffer message.
    """Construct a mesh instance from input `proto`."""
    return cls._new_object(mesh_proto=proto)

  @classmethod
  def from_string(cls, mesh_str: str) -> 'Mesh':
    # [FUNCTION DOC]
    # Alternate constructor: builds a Mesh from its string form (as produced by to_string).
    return cls._new_object(mesh_str=mesh_str)

  @classmethod
  def from_device(cls, device: str) -> 'Mesh':
    # [FUNCTION DOC]
    # Alternate constructor: builds a single-device mesh from a device string.
    """Constructs a single device mesh from a device string."""
    return cls._new_object(single_device=device)

  @classmethod
  def _from_mesh(cls, mesh: _pywrap_dtensor_device.Mesh):
    # [FUNCTION DOC]
    # HELPER: wraps an existing native mesh object into a Python Mesh (used for host meshes and
    # layout meshes). Returns the new Mesh copy.
    """Creates a copy from an existing pywrap mesh object."""
    return cls._new_object(mesh=mesh)

  @functools.cached_property
  def _host_mesh(self) -> 'Mesh':
    # [FUNCTION DOC]
    # HELPER: cached property backing host_mesh. On first access wraps the native host mesh in a
    # Python Mesh; the result is cached on the instance.
    return Mesh._from_mesh(super().host_mesh())

  def host_mesh(self) -> 'Mesh':
    # [FUNCTION DOC]
    # Returns the (cached) host mesh corresponding to this mesh, as a Python Mesh.
    """Returns a host mesh."""
    # TODO(b/242201545): Find a way to get the super class to return correct
    # typed objects.
    return self._host_mesh

  # TODO(b/242201545): implement this in Mesh C++ class
  def local_device_locations(self) -> List[Dict[str, int]]:
    # [FUNCTION DOC]
    # Returns, for each local device id in order, a mapping from mesh dimension names to that
    # device's index along the dimension (looked up in the unravelled index mapping).
    """Returns a list of local device locations.

    A device location is a dictionary from dimension names to indices on those
    dimensions.
    """
    mapping = self.unravel_index()
    return [mapping[device_id] for device_id in self.local_device_ids()]

  # TODO(b/242201545): implement this in Mesh C++ class
  @property
  def strides(self) -> List[int]:
    # [FUNCTION DOC]
    # Property: row-major strides of the mesh shape, computed from the shape.
    # KNOWN ISSUE: the docstring says an integer tensor is returned, but a plain list of integers
    # is returned.
    """Returns the strides tensor array for this mesh.

    If the mesh shape is `[a, b, c, d]`, then the strides array can be computed
    as `[b*c*d, c*d, d, 1]`. This array can be useful in computing local device
    offsets given a device ID. Using the same example, the device coordinates of
    the mesh can be computed as:

    ```
    [(device_id / (b*c*d)) % a,
     (device_id / (c*d))   % b,
     (device_id / (d))     % c,
     (device_id)           % d]
    ```

    This is the same as `(device_id // mesh.strides) % mesh.shape`.

    Returns:
      The mesh strides as an integer tensor.
    """
    return _compute_mesh_strides(self.shape())

  # TODO(b/242201545): implement this in Mesh C++ class
  def unravel_index(self):
    # [FUNCTION DOC]
    # Returns a mapping from device id (counting in row-major order over the mesh) to a mapping
    # from each dimension name to the device's index along that dimension. Takes no arguments.
    """Returns a dictionary from device ID to {dim_name: dim_index}.

    For example, for a 3x2 mesh, return this:

    ```
      { 0: {'x': 0, 'y', 0},
        1: {'x': 0, 'y', 1},
        2: {'x': 1, 'y', 0},
        3: {'x': 1, 'y', 1},
        4: {'x': 2, 'y', 0},
        5: {'x': 2, 'y', 1} }
    ```
    """
    idx_ranges = [range(self.dim_size(dim_name)) for dim_name in self.dim_names]
    mesh_pos = itertools.product(*idx_ranges)
    mapping = {}
    for device_id, device_pos in enumerate(mesh_pos):
      device_loc = {}
      for dim_name, dim_index in zip(self.dim_names, device_pos):
        device_loc[dim_name] = dim_index
      mapping[device_id] = device_loc
    return mapping


LayoutType = _pywrap_dtensor_device.LayoutType


# TODO(hthu): Consider making this class immutable.
@tf_export('experimental.dtensor.Layout', v1=[])
class Layout(_pywrap_dtensor_device.Layout):
  # [CLASS DOC]
  # Describes how a DTensor is partitioned over a Mesh: an ordered list of sharding specs, one
  # per tensor axis, each naming a mesh dimension or the special unsharded or match value. Thin
  # Python subclass of the native Layout class with validation in the constructor, alternate
  # constructors (proto, string, batch/inner sharded, replicated, single device), hashing,
  # repr/pickle via string form, and helpers to map device offsets to shard locations. Meant to be
  # treated as immutable.
  """Represents the layout information of a DTensor.

  A layout describes how a distributed tensor is partitioned across a mesh (and
  thus across devices). For each axis of the tensor, the corresponding
  sharding spec indicates which dimension of the mesh it is sharded over. A
  special sharding spec `UNSHARDED` indicates that axis is replicated on
  all the devices of that mesh.

  Refer to [DTensor Concepts](https://www.tensorflow.org/guide/dtensor_overview)
  for in depth discussion and examples.

  For example, let's consider a 1-D mesh:

  ```
  Mesh(["TPU:0", "TPU:1", "TPU:2", "TPU:3", "TPU:4", "TPU:5"], [("x", 6)])
  ```

  This mesh arranges 6 TPU devices into a 1-D array. `Layout([UNSHARDED], mesh)`
  is a layout for rank-1 tensor which is replicated on the 6 devices.

  For another example, let's consider a 2-D mesh:

  ```
  Mesh(["TPU:0", "TPU:1", "TPU:2", "TPU:3", "TPU:4", "TPU:5"],
       [("x", 3), ("y", 2)])
  ```

  This mesh arranges 6 TPU devices into a `3x2` 2-D array.
  `Layout(["x", UNSHARDED], mesh)` is a layout for rank-2 tensor whose first
  axis is sharded on mesh dimension "x" and the second axis is replicated. If we
  place `np.arange(6).reshape((3, 2))` using this layout, the individual
  components tensors would look like:

  ```
  Device  |  Component
   TPU:0     [[0, 1]]
   TPU:1     [[0, 1]]
   TPU:2     [[2, 3]]
   TPU:3     [[2, 3]]
   TPU:4     [[4, 5]]
   TPU:5     [[4, 5]]
  ```
  """

  def __init__(self, sharding_specs: List[str], mesh: Mesh):
    # [FUNCTION DOC]
    # Constructor. Validates that mesh is a Mesh and that every sharding spec is the unsharded or
    # match value or a unique dimension of the mesh, then calls the native constructor with a
    # static layout type. Parameters: sharding_specs (list of strings, one per tensor axis), mesh.
    # Raises a value error if mesh is not a Mesh, a mesh dimension is repeated, or a spec is not
    # a mesh dimension nor a special value.
    # KNOWN ISSUE: the docstring has a Returns section, but a constructor returns nothing.
    """Builds a Layout from a list of dimension names and a Mesh.

    Args:
      sharding_specs: List of sharding specifications, each corresponding to a
        tensor axis. Each specification (dim_sharding) can either be a mesh
        dimension or the special value UNSHARDED.
      mesh: A mesh configuration for the Tensor.

    Returns:
      A valid Layout built with given layout & mesh.
    """
    # Validate mesh
    if not isinstance(mesh, Mesh):
      raise ValueError('mesh is not a valid Mesh object.')

    # Validate sharding spec
    for _, dim_sharding in enumerate(sharding_specs):
      # If special value no need to check for uniqueness, just skip.
      if dim_sharding == UNSHARDED or dim_sharding == MATCH:
        continue
      # Check dim_sharding is unique.
      if sharding_specs.count(dim_sharding) > 1:
        raise ValueError(
            ('Mesh dimension {mesh_dim} was repeated in sharding ' +
             'specification {sharding_specs}. Mesh dimensions must be unique ' +
             'in a layout.').format(
                 mesh_dim=dim_sharding, sharding_specs=sharding_specs))
      # Check dim_sharding is mesh dimension.
      if dim_sharding not in mesh:
        raise ValueError(
            ('{dim_sharding}: A dimension sharding must either be a ' +
             'valid mesh dimension or UNSHARDED.').format(
                 dim_sharding=dim_sharding))

    super().__init__(
        type=LayoutType.STATIC, sharding_specs=sharding_specs, mesh=mesh
    )

  @classmethod
  def _new_object(cls, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: factory shared by the alternate constructors of Layout.
    # Creates an instance without running the Python constructor and initializes it directly
    # with the native constructor using the given arguments. Returns it.
    # Need to explicitly invoke the base class __init__ because
    # Layout.__init__ overrode it with a different signature.
    self = _pywrap_dtensor_device.Layout.__new__(cls)
    super().__init__(self, *args, **kwargs)
    return self

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns text that names the string constructor applied to the layout's string form.
    return f'Layout.from_string({self.to_string()})'

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash of the deterministic serialization of the layout proto, so equal layouts hash equally.
    return hash(self.as_proto().SerializeToString(deterministic=True))

  # TODO(panzf): change to pybind11 pickle implementation in the last step
  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickle support: reconstruct by calling the string constructor with the layout's string form.
    return Layout.from_string, (self.to_string(),)

  @property
  def mesh(self):
    # [FUNCTION DOC]
    # Property: the layout's mesh, wrapped as a Python Mesh from the native mesh object.
    return Mesh._from_mesh(mesh=super().mesh)  # pylint: disable=protected-access

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property: the shape of the layout's mesh (list of mesh dimension sizes), not of the tensor.
    return self.mesh.shape()

  @classmethod
  def batch_sharded(
      cls, mesh: Mesh, batch_dim: str, rank: int, axis: int = 0
  ) -> 'Layout':
    # [FUNCTION DOC]
    # Alternate constructor: layout of the given tensor rank in which the axis at index axis
    # (default 0) is sharded over the mesh dimension batch_dim and all others are unsharded.
    # Parameters: mesh, batch_dim (mesh dimension name), rank (integer), axis (integer).
    """Returns a layout sharded on batch dimension."""
    return cls._new_object(
        # Watchout for the different ordering.
        mesh=mesh,
        rank=rank,
        batch_dim=batch_dim,
        axis=axis,
    )

  # TODO(b/242201545): Move this to C++ / find the corresponding function there.
  def delete(self, dims: List[int]) -> 'Layout':
    # [FUNCTION DOC]
    # Returns a new Layout on the same mesh with the sharding specs at the given axis indices
    # removed. A single integer is accepted and treated as a one-element list.
    """Returns the layout with the give dimensions deleted."""
    if not isinstance(dims, list):
      dims = [dims]
    new_specs = [
        spec for i, spec in enumerate(self.sharding_specs) if i not in dims
    ]
    return Layout(new_specs, self.mesh)

  @classmethod
  def from_proto(cls, layout_proto: layout_pb2.LayoutProto) -> 'Layout':
    # [FUNCTION DOC]
    # Alternate constructor: builds a Layout from a layout protocol buffer message.
    """Creates an instance from a LayoutProto."""
    return cls._new_object(layout_proto=layout_proto)

  @classmethod
  def from_string(cls, layout_str: str) -> 'Layout':
    # [FUNCTION DOC]
    # Alternate constructor: builds a Layout from its human-readable string form.
    """Creates an instance from a human-readable string."""
    return cls._new_object(layout_str=layout_str)

  def to_parted(self) -> 'Layout':
    # [FUNCTION DOC]
    # Returns a "parted" version of this static layout, obtained from the native conversion.
    # Semantics are still under investigation per the source note.
    """Returns a "parted" layout from a static layout.

    A parted layout contains axes that are treated as independent by most of
    SPMD expanders.

    FIXME(b/285905569): The exact semantics is still being investigated.
    """
    return Layout._new_object(layout=super().to_parted())

  @classmethod
  def inner_sharded(cls, mesh: Mesh, inner_dim: str, rank: int) -> 'Layout':
    # [FUNCTION DOC]
    # Alternate constructor: layout of the given rank with only the last axis sharded over the
    # mesh dimension inner_dim (implemented via the batch-sharded constructor).
    """Returns a layout sharded on inner dimension."""
    return cls.batch_sharded(mesh, inner_dim, rank, axis=rank - 1)

  @classmethod
  def from_single_device_mesh(cls, mesh: Mesh) -> 'Layout':
    # [FUNCTION DOC]
    # Alternate constructor: single-device layout on the given single-device mesh.
    """Constructs a single device layout from a single device mesh."""
    return cls._new_object(mesh=mesh)

  @classmethod
  def from_device(cls, device: str) -> 'Layout':
    # [FUNCTION DOC]
    # Alternate constructor: single-device layout for a device string, via a single-device mesh.
    """Constructs a single device layout from a single device mesh."""
    return cls.from_single_device_mesh(Mesh.from_device(device))

  # TODO(b/242201545): Move this to C++ / find the corresponding function there.
  def offset_to_shard(self):
    # [FUNCTION DOC]
    # Returns a list indexed by device offset (mesh device id) giving, as a tuple, the shard
    # index along each tensor axis: zero for unsharded axes, otherwise the device's index along
    # the mesh dimension named by that axis's spec.
    # KNOWN ISSUE: only the unsharded value is special-cased, so a layout containing the match
    # value would fail with a missing-key error.
    """Mapping from offset in a flattened list to shard index."""
    unravel_index = self.mesh.unravel_index()
    locations = [None] * self.mesh.size
    for offset, mesh_loc in unravel_index.items():
      loc = []
      for dim_sharding in self.sharding_specs:
        if dim_sharding == UNSHARDED:
          loc.append(0)
        else:
          loc.append(mesh_loc[dim_sharding])
      locations[offset] = tuple(loc)

    return locations

  # TODO(b/242201545): Move this to C++ / find the corresponding function there.
  def offset_tuple_to_global_index(self, offset_tuple):
    # [FUNCTION DOC]
    # Converts a tuple of per-axis shard offsets into one row-major linear index over the shard
    # grid, weighting each offset by the product of the shard counts of all later axes.
    # Parameter: offset_tuple, one integer per tensor axis. Returns an integer.
    """Mapping from offset to index in global tensor."""
    index = 0
    for i, o in enumerate(offset_tuple):
      m = 1
      for x in range(i + 1, self.rank):
        m = m * self.num_shards(x)
      index = index + m * o
    return index

  @classmethod
  def replicated(cls, mesh: Mesh, rank: int) -> 'Layout':
    # [FUNCTION DOC]
    # Alternate constructor: fully replicated layout (all axes unsharded) of the given rank.
    """Returns a replicated layout of rank `rank`."""
    return cls._new_object(mesh=mesh, rank=rank)
