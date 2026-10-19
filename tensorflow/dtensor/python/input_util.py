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
# Input pipeline support for DTensor. Public pieces: DTensorDataset (wraps an unbatched or
# per-replica-batched tf.data dataset and yields DTensors laid out on a mesh according to given
# layouts) and TFDataServiceConfig (settings for tf.data service distribution, currently
# rejected for multi-client use). Private pieces: an iterator class and its type spec that pack
# per-device iterator resources into a single DTensor and copy elements to the mesh, plus
# validation, shard-count and index-matrix helpers used to slice batch elements per device.
# Import-time side effects: registers DTensorDataset in the experimental dtensor API.
"""APIs to deal with input datasets efficiently in DTensor.

When using tf.data with DTensor, the `DTensorDataset` API can be used to
efficiently handle loading the input data and correctly packing it to the
corresponding devices. This API is intended to work with unbatched data and can
be used for both data and model parallel setups.

Example usage:

>>> # 1-D mesh with 4 devices
>>> mesh = dtensor.Mesh(dim_names=['batch'], ...)
>>> layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)
>>> dataset = tf.data.Dataset.range(256)
>>> d_dataset = dtensor.DTensorDataset(
...     dataset=dataset,
...     global_batch_size=16,
...     mesh=mesh,
...     layouts=layout,
...     batch_dim='batch')
>>> d_iter = iter(d_dataset)
>>> # Each batch is a length 16 tensor sharded across 4 devices
>>> batch_0_dtensor = next(d_iter)
>>> batch_0_dtensor
<tf.Tensor: shape=(16,),
            dtype=int64,
            value={"CPU:0": [ 0  1  2  4],
                   "CPU:1": [ 5  6  7  8],
                   "CPU:2": [ 9 10 11 12],
                   "CPU:3": [13 14 15 16]}>
>>> batch_1_dtensor = next(d_iter)
>>> batch_1_dtensor
<tf.Tensor: shape=(16,),
            dtype=int64,
            value={"CPU:0": [17 18 19 20],
                   "CPU:1": [21 22 23 24],
                   "CPU:2": [25 26 27 28],
                   "CPU:3": [29 30 31 32]}>

For multi-client setups, `DTensorDataset` interacts with tf.data service to
correctly distribute the dataset among the participating clients. DTensor works
with tf.data service in co-located mode where each worker is running alongside
the DTensor client (the Tensorflow Python process). The `TFDataServiceConfig`
dataclass can be filled with information about the tf.data service cluster, and
passed to `DTensorDataset` to enable distribution.
"""

import dataclasses
import operator

from typing import Any, List, Optional, Sequence, Tuple

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@dataclasses.dataclass
class TFDataServiceConfig:
  # [CLASS DOC]
  # Plain data holder (dataclass) with two string fields: the tf.data service dispatcher address
  # and the shared job name. Passed to DTensorDataset to request service-based distribution.
  """Specifies the tf.data service configuration to use.

  Attributes:
    dispatcher_address: a string specifying the address of the tf.data service
      dispatcher server.
    job_name: a non-empty string identifying the shared job that will be created
      on tf.data service to process this dataset.
  """
  dispatcher_address: str
  job_name: str


# TODO(b/223275517): Add support for get_next_as_optional().
class _DTensorIterator(iterator_ops.OwnedIterator):
  # [CLASS DOC]
  # HELPER: iterator type returned by DTensorDataset iteration.
  # Owned-iterator subclass over a single DTensor holding the per-device iterator resources.
  # Each step fetches the next host element from the underlying iterators and copies it onto the
  # device mesh using the stored layouts. Holds the iterator resource DTensor, the global element
  # spec, the layouts and their string forms (used for the type spec). Does not support the
  # optional-returning next operation yet.
  """An iterator for a tf.data.Dataset distributed using DTensor.

  DTensorIterator encapsulates multiple underlying dataset iterators. It handles
  retrieving the tensors to be placed on each underlying device and then uses
  the 'pack' operation to create and return a DTensor. Thus users need only
  interact with a single DTensorIterator to automatically distribute dataset
  tensors onto devices.
  """

  def __init__(
      self,
      dtensor_components: Tuple[tensor.Tensor],
      global_element_spec: tensor_spec.TensorSpec,
      layouts: Any):
    # [FUNCTION DOC]
    # HELPER: constructor of the private DTensor iterator.
    # Unpacks the single-element components tuple to keep the iterator resource DTensor, stores
    # the global element spec and layouts, precomputes the string form of every layout in the
    # layout structure, and initializes the base owned iterator with the components and spec.
    # Parameters: dtensor_components (single-element tuple), global_element_spec, layouts (one
    # layout or nested structure matching the elements).
    """Initializes a distributed iterator for DTensor datasets.

    This iterator encapsulates tf.data iterators for the underlying devices, and
    treats it as a packed DTensor of iterator resource tensors.

    Args:
      dtensor_components: a tuple containing the underlying iterator resources
        packed into a DTensor. This is expected to be a tuple with a single
        element.
      global_element_spec: the underlying dataset's element spec from a global
        view.
      layouts: a structure of DTensor layouts to be applied to the elements
        returned by the underlying iterators. This can be a single layout or
        (possibly nested) tuples or dictionaries of layouts, and the structure
        must match the structure of the iterator elements.
    """
    # dtensor_components is expected to be a single-element tuple.
    [self._iterator_resource_dtensor] = dtensor_components
    self._global_element_spec = global_element_spec
    self._layouts = layouts
    self._layouts_str = nest.map_structure(
        lambda layout: layout.to_string(), layouts)

    super().__init__(
        components=dtensor_components, element_spec=global_element_spec)

  def __next__(self):
    # [FUNCTION DOC]
    # Returns the next element as DTensors on the device mesh. Gets the host element from the
    # base iterator, waits for async ops, copies each component to the mesh according to its
    # layout, waits again and returns the structure. When the data is exhausted: in eager mode
    # raises the end-of-iteration signal (stop iteration); in graph mode re-raises the original
    # out-of-range error.
    try:
      # IteratorGetNext will return a DTensor on the host, so move it to the
      # device mesh. If the dataset layouts are on the host mesh itself, this
      # is handled by DTensor as a no-op.
      host_elem = self._next_internal()
      context.async_wait()
      device_elem = nest.map_structure(
          api.copy_to_mesh, host_elem, self._layouts)
      context.async_wait()
      return device_elem
    except errors.OutOfRangeError as e:
      # Match TF2 eager executor behavior by raising StopIteration when iterator
      # is out of range.
      if context.executing_eagerly():
        raise StopIteration from e
      else:
        raise e

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # HELPER: type specification support. Returns an iterator spec built from the global element
    # spec and the string forms of the layouts.
    return _DTensorIteratorSpec(self._global_element_spec, self._layouts_str)


class _DTensorIteratorSpec(iterator_ops.IteratorSpec):
  # [CLASS DOC]
  # HELPER: type specification for the private DTensor iterator, used when the iterator crosses
  # function boundaries. Stores the global element spec and the layouts in string form (so the
  # spec is serializable); the single component is a resource tensor.
  """Type specification for `_DTensorIterator`."""

  __slots__ = ['_global_element_spec', '_layouts_str']

  def __init__(
      self, global_element_spec: tensor_spec.TensorSpec, layouts_str: Any):
    # [FUNCTION DOC]
    # HELPER: constructor. Initializes the base spec with the global element spec and stores the
    # spec and the structure of layout strings.
    super().__init__(global_element_spec)
    self._global_element_spec = global_element_spec
    self._layouts_str = layouts_str

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # HELPER: property returning the iterator class this spec describes.
    return _DTensorIterator

  def _serialize(self):
    # [FUNCTION DOC]
    # HELPER: returns a pair of the global element spec and the layout strings, which identifies
    # the spec.
    return (self._global_element_spec, self._layouts_str)

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # HELPER: property returning a one-element tuple with a scalar resource tensor spec (the
    # packed iterator resource).
    return (tensor_spec.TensorSpec([], dtypes.resource),)

  def _to_components(self, value):
    # [FUNCTION DOC]
    # HELPER: returns a one-element tuple holding the iterator's resource DTensor.
    return (value._iterator_resource_dtensor,)  # pylint: disable=protected-access

  def _from_components(self, components):
    # [FUNCTION DOC]
    # HELPER: rebuilds an iterator from its components tuple: parses the stored layout strings
    # back into layouts and constructs a new private DTensor iterator with the stored global
    # element spec. Returns the iterator.
    layouts = nest.map_structure(
        layout_lib.Layout.from_string, self._layouts_str)
    return _DTensorIterator(
        dtensor_components=components,
        global_element_spec=self._global_element_spec,
        layouts=layouts)

  @classmethod
  def from_value(cls, value):
    # [FUNCTION DOC]
    # HELPER: class-level factory that builds a spec from an existing iterator, copying its
    # global element spec and layout strings.
    return cls(value._global_element_spec, value._layouts_str)  # pylint: disable=protected-access


def _validate_input(flattened_layouts: Sequence[layout_lib.Layout],
                    flattened_elem_spec: Sequence[tensor_spec.TensorSpec],
                    dataset_already_batched: bool):
  # [FUNCTION DOC]
  # HELPER: input validation for DTensorDataset construction.
  # Checks, for each pair of flattened layout and element spec: the element spec list is
  # non-empty; each element has a known rank; the layout rank equals the element rank (plus one
  # when the dataset is not yet batched); and, for already-batched data, the batch size of the
  # first element is known and every element has that same batch size. Raises a value error with
  # a descriptive message on the first violation. Returns nothing.
  """Checks that the dataset's layouts and element specs are compatible.

  Args:
    flattened_layouts: the flattened list of layouts used to distribute the
      dataset.
    flattened_elem_spec: the flattened list of element specs used in the
      dataset's components.
    dataset_already_batched: whether the dataset to be validated is already
      batched.

  Raises:
    ValueError: if the dataset's inputs are incompatible.
  """
  if not flattened_elem_spec:
    raise ValueError(
        'Expected input element spec of at least one element, was empty.')

  first_elem_shape = flattened_elem_spec[0].shape

  for layout, elem_spec in zip(flattened_layouts, flattened_elem_spec):
    if elem_spec.shape.rank is None:
      raise ValueError(
          'Dataset element shape must have a valid rank, got spec %s.' %
          elem_spec)

    # Check that layout's rank matches the element's rank. If dataset is not yet
    # batched, then the layout's rank must be one greater than the element's
    # rank.
    expected_rank = elem_spec.shape.rank
    if not dataset_already_batched:
      expected_rank += 1
    if layout.rank != expected_rank:
      raise ValueError(
          ('Expected layout with rank %d for element spec %s, got layout %s. '
           'Check that the dataset is not batched before passing to '
           'DTensorDataset.') %
          (expected_rank, elem_spec, layout.sharding_specs))

    if dataset_already_batched:
      # Check that the batch dimension size of all dataset elements match.
      batch_dim_size = first_elem_shape.as_list()[0]
      if batch_dim_size is None:
        raise ValueError(
            ('Size of batch dimension of element spec %s is None. Ensure '
             'drop_remainder=True when batching the dataset.') % elem_spec)

      if elem_spec.shape.as_list()[0] != batch_dim_size:
        raise ValueError(
            ('Size of batch dimension of element spec %s does not match '
             'expected size %d.') % (elem_spec, batch_dim_size))


def _shard_counts(layout: layout_lib.Layout,
                  batch_dim: Optional[str] = None) -> List[int]:
  # [FUNCTION DOC]
  # HELPER: computes local slicing factors for DTensorDataset.
  # Returns, for each axis of the layout, the number of local shards: one for axes that are
  # unsharded or sharded on the batch dimension (the batch axis is split across replicas
  # elsewhere), otherwise the size of the mesh dimension the axis is sharded on.
  # Parameters: layout, batch_dim (optional name of the batch mesh dimension).
  """Computes a list of the number of shards in each dimension of the layout.

  The shard counts are used to slice each dataset element. The batch dimension's
  count is overridden to 1 since we only consider how many shards to make
  locally (within each local replica). Sharding across clients is handled by
  either tf.data.Dataset's shard transformation (in the single-client case) or
  tf.data service's distribute function (in the multi-client case).

  Args:
    layout: the layout to compute the shard counts for.
    batch_dim: the name of the batch dimension of the layout, if present.

  Returns:
    A list of shard counts, one element per dimension of the layout.
  """
  shard_counts = []
  for spec in layout.sharding_specs:
    if spec in (batch_dim, layout_lib.UNSHARDED):
      shard_counts.append(1)
    else:
      shard_counts.append(layout.mesh.dim_size(spec))
  return shard_counts


def _index_matrix(layout: layout_lib.Layout,
                  elem_spec: tensor_spec.TensorSpec) -> tensor.Tensor:
  # [FUNCTION DOC]
  # HELPER: builds the matrix used by DTensorDataset to compute per-device slice offsets.
  # Returns a 32-bit integer tensor with one row per mesh dimension and one column per layout
  # axis. The first column (batch axis) is always zero. For each later axis sharded on a mesh
  # dimension the entry is the element's size along that axis divided by the mesh dimension size;
  # all other entries are zero. Parameters: layout, elem_spec.
  # KNOWN ISSUE: the element dimension is looked up with an index that skips the batch axis, which
  # matches unbatched element specs; the caller also passes already-batched specs, where the
  # offset would refer to the wrong dimension.
  """Computes a utility matrix to derive device-based slice offsets.

  This function builds a matrix of shape `[mesh.rank, layout.rank]` for each
  dataset element. This matrix can be used to slice the DTensor components
  returned by the iterator according to the local device that component is to be
  placed on. This can be done by multiplying the device offsets of shape
  `[1, mesh.rank]` with this index matrix to get a `[1, layout.rank]` shape
  tensor containing the slice offsets.

  Note: the index on the batch dim is always 0 since sharding on the batch
  dimension is handled by either tf.data.Dataset's shard transformation (in the
  single-client case) or tf.data service's distribute function (in the
  multi-client case). If there is no sharding on the batch dimension (or any
  other dimension), the slice index remains 0.

  Args:
    layout: the layout of the dataset element.
    elem_spec: the spec of the dataset element.

  Returns:
    The index matrix as a tensor.
  """
  matrix = []
  for dim in layout.mesh.dim_names:
    row = [0]
    for layout_idx, spec in enumerate(layout.sharding_specs[1:]):
      if spec == layout_lib.UNSHARDED or spec != dim:
        row.append(0)
      else:
        row.append(elem_spec.shape[layout_idx] // layout.mesh.dim_size(dim))
    matrix.append(row)

  return constant_op.constant(matrix, dtype=dtypes.int32)


def _pack_iterator_resource_dtensor(
    datasets: List[Tuple[int, data_types.DatasetV2]],
    layouts: Any,
    mesh: layout_lib.Mesh,
    num_local_devices_per_replica: int):
  # [FUNCTION DOC]
  # HELPER: builds the packed iterator resource for DTensorDataset iteration.
  # For every local replica dataset, creates one iterator per local device in that replica (each
  # on the matching host mesh device and reading its own shard of the dataset), then packs all
  # iterator resources into a replicated scalar DTensor on the host mesh and tells the DTensor
  # device the host-mesh layouts expected for the elements. Raises a value error if the number of
  # created iterators differs from the host mesh's local device count. Parameters: datasets (list
  # of pairs of replica id and dataset), layouts, mesh, num_local_devices_per_replica. Returns the
  # packed resource DTensor. Side effect: configures element layouts on the global device.
  """Creates a DTensor iterator resource for the per-replica datasets.

  Given a list of replica ID to tf.data.Dataset mappings, this function creates
  iterators for each device and then packs the underlying iterator resource
  tensors into a single DTensor. This resource tensor is used by the
  IteratorGetNext op to retrieve the next element in the dataset.

  Args:
    datasets: a list of tuples of each unique local replica ID to the dataset
      object whose elements will be placed on the devices corresponding to that
      replica.
    layouts: a structure of DTensor layouts to be applied to the elements
      returned by the underlying iterators. This can be a single layout or
      (possibly nested) tuples or dictionaries of layouts, and the structure
      must match the structure of the iterator elements.
    mesh: the DTensor mesh to place the iterator batches on.
    num_local_devices_per_replica: the number of devices in each data-parallel
      replica.

  Returns:
    A DTensor of the underlying iterator resource tensors.
  """
  host_mesh_devices = mesh.host_mesh().local_devices()
  device_idx = 0

  iterators = []
  for _, dataset in datasets:
    for idx in range(num_local_devices_per_replica):
      with ops.device_v2(host_mesh_devices[device_idx]):
        device_dataset = dataset.shard(
            num_shards=num_local_devices_per_replica, index=idx)
        iterators.append(iter(device_dataset))
      device_idx += 1

  if device_idx != len(host_mesh_devices):
    raise ValueError(
        'The `datasets` argument does not have the correct number of'
        f' underlying datasets, found {device_idx} but expected'
        f' {len(host_mesh_devices)}.')

  host_layouts = nest.map_structure(
      lambda l: layout_lib.Layout(l.sharding_specs, mesh.host_mesh()), layouts)

  # Pack the iterator resource tensors into a replicated 0-dimensional DTensor
  # and set the element layouts.
  iterator_resources = [it._iterator_resource for it in iterators]  # pylint: disable=protected-access
  d_iterator_resource = api.pack(
      iterator_resources,
      layout_lib.Layout.replicated(mesh=mesh.host_mesh(), rank=0))
  api._dtensor_device().set_iterator_element_layouts(  # pylint: disable=protected-access
      d_iterator_resource, nest.flatten(host_layouts))

  return d_iterator_resource


@tf_export('experimental.dtensor.DTensorDataset', v1=[])
class DTensorDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # Public dataset wrapper that turns a tf.data dataset into a source of DTensors. At
  # construction it validates layouts against the element specs, batches unbatched data to the
  # per-replica batch size, computes the global element spec, replica and device counts and
  # slicing helper data. Iterating it creates per-replica, per-device pipelines (sharding,
  # repeating batches per device, partitioning along non-batch sharded axes, prefetching), packs
  # their iterators and returns a private DTensor iterator. Supports only iteration and element
  # spec access, not the full dataset interface.
  """A dataset of DTensors.

  DTensorDataset encapsulates a `tf.data.Dataset` whose elements are
  automatically packed and returned as DTensors based on a given mesh and
  layouts.
  """

  def __init__(self,
               dataset: data_types.DatasetV2,
               *,
               mesh: layout_lib.Mesh,
               layouts: Any,
               global_batch_size: int,
               dataset_already_batched: bool = False,
               batch_dim: Optional[str] = None,
               prefetch: Optional[int] = None,
               tf_data_service_config: Optional[TFDataServiceConfig] = None):
    # [FUNCTION DOC]
    # Constructor. Parameters: dataset, mesh, layouts (layout or matching nested structure),
    # global_batch_size, dataset_already_batched (default false), batch_dim (optional mesh
    # dimension for the batch axis), prefetch (optional), tf_data_service_config (optional).
    # Raises not-implemented if a service config is given. Checks layouts match the dataset
    # structure, determines replicas (mesh size along the batch dimension, or one without it) and
    # this client's replica ids, requires every layout to be sharded on the batch dimension,
    # validates ranks and batch sizes, batches unbatched data with dropped remainder by the
    # per-replica size (or checks the existing batch size), builds the global element spec with
    # the global batch size, and computes device counts per replica, clients per replica,
    # the partition offset for this client, shard counts and index matrices.
    # Raises value errors for the mismatches described; structure mismatches raise type or value
    # errors from the structure checker.
    # KNOWN ISSUE: the docstring presents tf.data service as an option, but any non-null config
    # always raises a not-implemented error.
    """Creates a DTensorDataset.

    DTensorDataset automatically handles distribution of the dataset elements to
    each client's devices. It can be used to create an iterator that returns
    DTensors of the input data on each iteration.

    DTensorDataset works best with unbatched datasets. It takes the mesh and the
    provided layouts to automatically calculate how to batch the input locally
    for each replica.

    If the provided dataset is already batched according to the per-replica
    batch size, then `dataset_already_batched` must be set and DTensorDataset
    will check that the batch size is consistent with the intended
    `global_batch_size` using the layout information. Each replica receives a
    separate slice of the global batch, thus the per-replica batch size can be
    computed as the global batch size divided by the number of model replicas.
    For a DTensor mesh, the number of replicas is equal to the size of the
    mesh's batch dimension.

    Note: `tf.experimental.dtensor.DTensorDataset` instances do *not* implement
    the full interface of `tf.data.Dataset`. It only supports two usages we will
    mention below: iteration and `element_spec`. We don't support any other APIs
    to transform or inspect the dataset.

    TODO(b/223275517): add support for input datasets that are already batched
    to the global batch size.

    Args:
      dataset: a `tf.data.Dataset` object.
      mesh: the DTensor mesh to place the dataset batches on.
      layouts: a structure of DTensor layouts to be applied to the input dataset
        values. This can be a single layout or (possibly nested) tuples or
        dictionaries of layouts, and the structure must match the structure of
        the dataset. Either all or none of the layouts should be sharded on the
        batch dimension; having only a subset of layouts batch sharded will not
        work and raises a ValueError.
      global_batch_size: the desired global batch size.
      dataset_already_batched: must be set only if the dataset is already
        batched to the per-replica batch size. The batched dataset must have
        `drop_remainder=True` set since DTensor requires static shapes for
        slicing the input tensors.
      batch_dim: the mesh dimension on which the input's batch dimension is
        sharded. Set to None if the input layouts do not shard on the batch
        dimension.
      prefetch: number of batches to prefetch using Dataset.prefetch.
      tf_data_service_config: if operating in multi-client mode, this config
        specifies the tf.data service configuration to use.

    Raises:
      ValueError: on any of the following situations,
        1. if the structures and ranks of layouts and the dataset do not match.
        2. if the shapes in the dataset's spec are not fully defined.
        3. if batch_dim is specified and all layouts are not batch-sharded.
        4. if per_replica_batch_size is specified for an already batched Dataset
           but it does not match the expected per-replica size based on the
           provided mesh.
      TypeError: if type of structures of layouts and the dataset do not match.
    """
    super().__init__(dataset, dataset_ops.to_variant(dataset))

    # TODO(b/271162918): fix multi-client use case.
    if tf_data_service_config is not None:
      raise NotImplementedError(
          'Multi-client DTensorDataset is currently not supported.'
          ' Check b/271162918.')

    self._mesh = mesh
    self._layouts = layouts
    self._batch_dim = batch_dim
    self._prefetch = prefetch
    self._tf_data_service_config = tf_data_service_config

    nest.assert_same_structure(dataset.element_spec, layouts)
    flattened_layouts = nest.flatten(layouts)
    flattened_elem_spec = nest.flatten(dataset.element_spec)

    if batch_dim:
      self.num_global_replicas = mesh.dim_size(batch_dim)
      self._local_replica_ids = list(
          dict.fromkeys(
              [loc[batch_dim] for loc in mesh.local_device_locations()]))

      for layout in flattened_layouts:
        if batch_dim != layout.sharding_specs[0]:
          raise ValueError(
              ('batch_dim %s was specified but at least one layout did not '
               'contain it: %s') % (batch_dim, layout))
    else:
      # Only one replica since there is no sharding on the batch dimension.
      self.num_global_replicas = 1
      self._local_replica_ids = [0]

    # Validate layout and element spec compatibility, and raise ValueError if
    # invalid.
    _validate_input(
        flattened_layouts,
        flattened_elem_spec,
        dataset_already_batched=dataset_already_batched)

    expected_batch_size = global_batch_size // self.num_global_replicas
    if not dataset_already_batched:
      self._batched_dataset = dataset.batch(
          expected_batch_size, drop_remainder=True)
    else:
      per_replica_batch_size = flattened_elem_spec[0].shape.as_list()[0]
      if per_replica_batch_size != expected_batch_size:
        raise ValueError(
            ('per_replica_batch_size does not matched expected size based on '
             'the mesh, got %d but expected %d.') %
            (per_replica_batch_size, expected_batch_size))
      self._batched_dataset = dataset

    # Construct a global element spec of the dataset.
    flattened_global_elem_spec = []
    batch_tensor_shape = tensor_shape.as_shape([global_batch_size])
    for elem_spec in nest.flatten(self._batched_dataset.element_spec):
      new_elem_spec = tensor_spec.TensorSpec(
          shape=operator.concat(batch_tensor_shape, elem_spec.shape[1:]),
          dtype=elem_spec.dtype,
          name=elem_spec.name)
      flattened_global_elem_spec.append(new_elem_spec)
    self._global_element_spec = nest.pack_sequence_as(
        dataset.element_spec, flattened_global_elem_spec)

    num_global_devices_per_replica = config.num_global_devices(
        mesh.device_type()) // self.num_global_replicas
    self._num_local_replicas = len(self._local_replica_ids)
    self._num_local_devices_per_replica = mesh.num_local_devices(
    ) // self._num_local_replicas
    # The number of clients each replica is split over.
    self._num_clients_per_replica = (
        num_global_devices_per_replica // self._num_local_devices_per_replica)
    # In the case where a replica is split across multiple clients, an offset
    # needs to be added to the index used by the partitioning logic such that
    # the local devices on that client can be correctly matched to slices of the
    # input tensor(s). If replicas are wholly contained within a client, then
    # this offset is always 0.
    self._partition_offset = (config.client_id() % self._num_clients_per_replica
                             ) * self._num_local_devices_per_replica

    # Helper data structures used in partitioning the dataset tensors.
    self._all_shard_counts = [
        _shard_counts(layout, batch_dim) for layout in flattened_layouts
    ]
    self._index_matrices = [
        _index_matrix(layout, elem_spec)
        for layout, elem_spec in zip(flattened_layouts, flattened_elem_spec)
    ]

  def __iter__(self):
    # [FUNCTION DOC]
    # Creates and returns a new private DTensor iterator over the dataset. Picks a sharding policy
    # (data-based when a replica spans several clients, in which case batches are first repeated
    # per client; file-based when each replica belongs to one client; off without a batch
    # dimension) and applies tf.data service distribution when configured. For each local
    # replica: auto-shards the dataset, repeats each batch per local device, partitions along
    # non-batch axes, and applies prefetch scaled by the devices per replica. Then packs all
    # iterators into one resource DTensor and wraps it. Side effects: creates iterators and
    # configures the global DTensor device.
    datasets: List[Tuple[int, data_types.DatasetV2]] = []

    # Start with the batched the dataset.
    local_dataset = self._batched_dataset

    if self._batch_dim is not None:
      if self._num_clients_per_replica > 1:
        # If a replica is split over multiple clients then each batch needs to
        # be repeated before distribution as many times as there are clients
        # corresponding to that replica.
        local_dataset = self._repeat_batch(local_dataset,
                                           self._num_clients_per_replica)
        sharding_policy = data_service_ops.ShardingPolicy.DATA
      else:
        # Replicas are unique to each client, so FILE based sharding can be used
        # which is more performant since each worker does not need to read the
        # entire dataset.
        sharding_policy = data_service_ops.ShardingPolicy.FILE
    else:
      # No batch dimension sharding specified so disable dataset sharding during
      # the distribute step.
      sharding_policy = data_service_ops.ShardingPolicy.OFF

    # Apply distribution here (if specified) so all remaining transformations
    # are executed locally.
    if self._tf_data_service_config is not None:
      local_dataset = local_dataset.apply(
          data_service_ops.distribute(
              processing_mode=sharding_policy,
              service=self._tf_data_service_config.dispatcher_address,
              job_name=f'{self._tf_data_service_config.job_name}_{config.client_id()}',
              target_workers='LOCAL'))

    for local_replica_idx, replica_id in enumerate(self._local_replica_ids):
      # Select the shard for the corresponding replica.
      dataset = distribute._AutoShardDataset(
          local_dataset,
          num_workers=self._num_local_replicas,
          index=local_replica_idx,
          num_replicas=self.num_global_replicas)

      # Repeat each batch for each local device in the replica.
      dataset = self._repeat_batch(dataset, self._num_local_devices_per_replica)

      # Slice each shard further for all non-batch dim shards. If there is no
      # non-batch dim sharding, this slice is essentially a no-op.
      dataset = self._partition(dataset)

      # Apply prefetch as the last step. Since each batch is repeated, the
      # number of elements to prefetch has to be scaled by the same size.
      if self._prefetch is not None:
        dataset = dataset.prefetch(
            self._prefetch * self._num_local_devices_per_replica)

      datasets.append((replica_id, dataset))

    # Convert the datasets into iterators placed on the host.
    d_iterator_resource = _pack_iterator_resource_dtensor(
        datasets=datasets,
        layouts=self._layouts,
        mesh=self._mesh,
        num_local_devices_per_replica=self._num_local_devices_per_replica)

    return _DTensorIterator(
        dtensor_components=(d_iterator_resource,),
        global_element_spec=self._global_element_spec,
        layouts=self._layouts)

  def _repeat_batch(self, dataset, repeats):
    # [FUNCTION DOC]
    # HELPER: repeats every element of a dataset the given number of times in a row (so each
    # batch is delivered to every device of a replica or client). Returns the dataset unchanged
    # when repeats is one; otherwise returns a flat-mapped dataset.
    if repeats == 1:
      # Remove this shortcut if tf.data can optimize this away.
      return dataset

    def repeat(*x):
      # [FUNCTION DOC]
      # HELPER: flat-map callback; wraps the element components into a one-element dataset and
      # repeats it the captured number of times.
      return dataset_ops.DatasetV2.from_tensors(x).repeat(repeats)

    return dataset.flat_map(repeat)

  def _partition(self, dataset):
    # [FUNCTION DOC]
    # HELPER: slices each batch for the device that will receive it. Returns the dataset unchanged
    # when there is one local device per replica and no partition offset. Otherwise enumerates the
    # dataset and maps each element through a slicing function.
    """Slices each dataset element on any sharded non-batch dimension."""
    if self._num_local_devices_per_replica == 1 and self._partition_offset == 0:
      # Remove this shortcut if tf.data can optimize this away.
      return dataset

    # TODO(b/223275517): decouple from self and make testable.
    def slice_batch(index, batch):
      # [FUNCTION DOC]
      # HELPER: map callback for _partition. Inputs: the element's running index and the batch
      # structure. Computes the local device slot as the index modulo local devices per replica,
      # adds the partition offset, converts it to mesh coordinates, and for each component
      # multiplies the coordinates by that component's index matrix to get the start offsets; the
      # slice size is the component shape divided by its shard counts. Returns the batch structure
      # with every component replaced by its device's slice.
      flattened_batch = nest.flatten(batch)
      flattened_output = []

      norm_index = math_ops.cast(
          index % self._num_local_devices_per_replica, dtype=dtypes.int32)
      norm_index += self._partition_offset
      coords = self._mesh.coords(norm_index)
      coords = array_ops.reshape(coords, (1, -1))

      for element, shard_counts, idx_matrix in zip(flattened_batch,
                                                   self._all_shard_counts,
                                                   self._index_matrices):
        indexes = math_ops.matmul(coords, idx_matrix)
        start = array_ops.reshape(indexes, (-1,))
        size = array_ops.shape_v2(
            element, out_type=dtypes.int32) // shard_counts
        flattened_output.append(
            array_ops.slice(element, begin=start, size=size))

      return nest.pack_sequence_as(batch, flattened_output)

    enumerated_dataset = dataset.enumerate()
    partitioned_dataset = enumerated_dataset.map(slice_batch)
    return partitioned_dataset

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property: the global (whole-batch) element spec, with the batch dimension set to the global
    # batch size.
    return self._global_element_spec
