# [MODULE DOC]
# HELPER: shared utilities for distribution strategies backed by DTensor (used by the DTensor
# strategy extension in this package). Provides the default batch mesh dimension name, a
# distributed value class that wraps a DTensor, a replica context class that rejects local
# replica operations because DTensor runs in a global context, and functions to initialize the
# accelerator system, convert inputs and per-replica values to DTensors, check for distributed
# values, and implement reduction. Import time side effect: registers a tensor conversion
# function for the DTensor distributed value class.
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
"""Utilities for strategies that are backed by DTensor."""

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2


# Default dimension name used for the mesh created when user provide a list
# of devices. For mirrored strategy, it should be a 1D mesh with batch dim only.
DEFAULT_BATCH_MESH_DIM_NAME = "batch"


class DTensorDistributedValue(values_lib.DistributedValues):
  # [CLASS DOC]
  # Distributed value backed by one DTensor, so that the same object can be passed to APIs that
  # expect distributed values and to DTensor low level APIs. Holds the DTensor and (through the
  # parent) the list of its per-device components. In eager mode the components are unpacked from
  # the DTensor; in graph mode the DTensor is kept as a single global component.
  """DistributedValue backed by a DTensor instance.

  This class is useful to align the interface between DTensor and tf.distribute.
  Most of the tf.distribute API will accept/return DistributedValue, whereas
  DTensor low level API will only accept DTensor instance. In order to avoid
  the conversion back and forth between DistributedValue and DTensor, we
  introduce this class so that it can work with both side.
  """

  def __init__(self, dtensor):
    # [FUNCTION DOC]
    # Parameter dtensor: the DTensor instance to wrap. In eager mode raises a value error if the
    # argument is not a DTensor, and initializes the parent with the unpacked components. In graph
    # mode it cannot unpack, so the parent is initialized with a one element list holding the
    # DTensor. Stores the DTensor. Returns nothing.
    if context.executing_eagerly():
      if not d_api.is_dtensor(dtensor):
        raise ValueError("The DTensorDistributedValue can only be built with "
                         f"DTensor instance, got {type(dtensor)}")
      super().__init__(d_api.unpack(dtensor))
    else:
      # We can't unpack the dtensor instance for now due to graph context.
      # We will treat the dtensor instance as one global instance and let it
      # return as a global replica instance.
      # TODO(feyu): Support unpack in the graph context.
      super().__init__([dtensor,])
    self._dtensor = dtensor

  def get_dtensor(self):
    # [FUNCTION DOC]
    # Returns the wrapped DTensor.
    return self._dtensor

  @property
  def values(self):
    # [FUNCTION DOC]
    # Property returning the stored components, to match the per-replica interface (the public
    # per-replica type defines no methods).
    # Note that this method exists so that it match the interface for PerReplica
    # The public API in `tf.types.experimental.distributed.PerReplica` doesn't
    # define any methods.
    return self._values


def _dtensor_distributed_value_to_tensor(
    var, dtype=None, name=None, as_ref=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the registration comment says this gives an error message for wrong usage, but
  # for the normal case it actually returns the wrapped DTensor as the converted tensor; the
  # errors only cover incompatible dtype and reference requests. The messages also talk about a
  # variable and a per-replica value although the object is a DTensor distributed value.
  # HELPER: tensor conversion function registered for the DTensor distributed value class.
  # Parameters: var (the distributed value), dtype (optional requested type), name (ignored),
  # as_ref (whether a reference is requested). Raises a value error if dtype is given and not
  # compatible with the DTensor's type, and a not-implemented error if a reference is requested.
  # Returns the wrapped DTensor.
  del name
  dtensor = var.get_dtensor()
  if dtype is not None and not dtype.is_compatible_with(dtensor.dtype):
    raise ValueError(
        "Incompatible type conversion requested to type {!r} for variable "
        "of type {!r}".format(dtype.name, dtensor.dtype.name))
  if as_ref:
    raise NotImplementedError(
        "PerReplica doesn't support being used as a reference.")
  return dtensor


# Register a conversion function to provide a useful error message when users
# try to use PerReplica values in the wrong contexts
tensor_conversion_registry.register_tensor_conversion_function(
    DTensorDistributedValue, _dtensor_distributed_value_to_tensor)


class DTensorReplicaContext(distribute_lib.ReplicaContext):
  # [CLASS DOC]
  # Replica context for DTensor backed strategies. DTensor programs run in one global context, so
  # there is a single replica with a constant id of zero, and operations that need local values
  # (merge call, reduce, gather, local replica id, update) raise an explicit not-implemented
  # error with a class-level message. Created by the strategy extension around each function run.
  """ReplicaContext for strategy that is backed by DTensor.

  Since the DTensor is operated in the global context, most of the methods from
  existing strategy ReplicaContext is not applicable since they need to access
  local values. For now most of the methods in this class will raise explicit
  error to user, and we will add more support for local values in future.
  """
  _UNSUPPORTED_ERROR_MSG = (
      "Strategy that is backed by DTensor is run with a global context, and "
      "doesn't support operations for local context, like any call to merge/"
      "gather/reduce or local replica ID. Please use any strategy that is not "
      "backed by DTensor")

  def __init__(self, strategy):
    # [FUNCTION DOC]
    # Initializes the parent replica context with the given strategy and no local replica id in
    # the sync group (accessing it raises an error elsewhere). Returns nothing.
    # Since DTensor strategy only runs in a global context, and we can't have
    # a local replica ID in the sync group. For now we pass None to parent, and
    # raise an explicit error when it is accessed.
    super().__init__(strategy, replica_id_in_sync_group=None)

  def __enter__(self):
    # [FUNCTION DOC]
    # Enters the context. Copy of the parent behavior without the first-replica check: pushes this
    # context's per-thread mode, remembers the previous summary recording flag for distribution
    # strategies, and sets that flag to true. Side effects on thread and summary global state;
    # returns nothing (not the context object).
    # This is a copy of parent class, without any check about whether the
    # current replica is the first one (since DTensor only has one).
    distribute_lib._push_per_thread_mode(self._thread_context)  # # pylint: disable=protected-access

    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    self._summary_recording_distribution_strategy = (
        summary_state.is_recording_distribution_strategy)
    summary_state.is_recording_distribution_strategy = True

  @property
  def replica_id_in_sync_group(self):
    # [FUNCTION DOC]
    # Always returns zero, since there is one global context. Needed by random number generation
    # that derives seeds from the replica id.
    # Since there is only one global context for DTensor, we always return a
    # constant value here. This value is needed by the RNG which try to generate
    # different seed for different replica.
    return 0

  @property
  def _replica_id(self):
    # [FUNCTION DOC]
    # Always raises a not-implemented error with the unsupported-operation message.
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def merge_call(self, merge_fn, args=(), kwargs=None):
    # [FUNCTION DOC]
    # Not supported: always raises a not-implemented error with the unsupported-operation message.
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def all_reduce(self, reduce_op, value, options=None):
    # [FUNCTION DOC]
    # Not supported: always raises a not-implemented error with the unsupported-operation message.
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def all_gather(self, value, axis, options=None):
    # [FUNCTION DOC]
    # Not supported: always raises a not-implemented error with the unsupported-operation message.
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

  def _update(self, var, fn, args=(), kwargs=None, group=True):
    # [FUNCTION DOC]
    # Not supported: always raises a not-implemented error with the unsupported-operation message.
    raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)


def initialize_accelerator_system_once(device_type):
  # [FUNCTION DOC]
  # Initializes the accelerator system for the given device type string (for example GPU or TPU)
  # if it has not been initialized yet, resetting the eager context in the process. This also
  # creates the paired virtual host CPUs needed by datasets and checkpoints. Does nothing if
  # already initialized (even for a different device type). Returns nothing.
  # Initialize the GPU/TPU before creating the mesh.
  # Note that this method will also trigger the creation of the pairing
  # virtual host CPUs, which is needed by dataset and checkpoint.
  if not accelerator_util.is_initialized():
    # TODO(feyu): Add a method in accelerator_util to check the initialized
    # mesh device types.
    accelerator_util.initialize_accelerator_system(
        device_type,
        experimental_reset_context=True)


def convert_inputs_to_dtensor(inputs, mesh):
  # [FUNCTION DOC]
  # Converts an input to something DTensor code accepts. A DTensor distributed value gives its
  # DTensor; other distributed values are packed via the per-replica conversion with the given
  # mesh; DTensor iterators are returned as they are; tensors are returned as they are when they
  # are DTensors (eager) or always in graph mode (assumed pre-distributed); a non-DTensor tensor
  # in eager mode and any other type raise a value error through the error helper.
  """Convert any input types to DTensor instance."""
  if isinstance(inputs, DTensorDistributedValue):
    return inputs.get_dtensor()
  elif isinstance(inputs, values_lib.DistributedValues):
    return convert_per_replica_to_dtensor(inputs, mesh)
  elif isinstance(inputs, input_util._DTensorIterator):   # pylint: disable=protected-access
    return inputs
  elif tensor_util.is_tensor(inputs):
    if context.executing_eagerly():
      if d_api.is_dtensor(inputs):
        return inputs
      else:
        # For a non-dtensor input in eager context, we could choose to replica
        # them into per-replica and then pack them into dtensor. However, this
        # will cause an eager/graph discrepancy since we can't do this check in
        # the graph context. For now, we will ask user to provide a distributed
        # value for inputs.
        _raise_unsupported_input_type_error(inputs)
    else:
      # For graph context, since we can't check if they are dtensor or not. We
      # will assume the value is already distributed. This is a critical use
      # case for keras, where all the inputs are pre-distributed via strategy,
      # and the train function execute within graph context.
      return inputs
  else:
    # For any other types.
    _raise_unsupported_input_type_error(inputs)


def _raise_unsupported_input_type_error(inputs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the message mentions MirroredStrategy and suggests API names that are not the
  # public method names, even though it is used for any DTensor strategy.
  # HELPER: always raises a value error explaining that the input type is unsupported and showing
  # the received type.
  raise ValueError("Unsupported input types for MirroredStrategy. "
                   "Please use `strategy.distribute_dataset` or "
                   "`strategy.distribute_values_from_function` to "
                   f"distribute inputs. Received input type: {type(inputs)}")


def is_distributed_value(value):
  # [FUNCTION DOC]
  # Returns true if the value is a distributed values object or a DTensor, false otherwise.
  return isinstance(
      value, values_lib.DistributedValues) or d_api.is_dtensor(value)


def convert_per_replica_to_dtensor(per_replica_value, mesh):
  # [FUNCTION DOC]
  # Packs the per-replica components of a per-replica value into one DTensor with a layout
  # sharded over the batch mesh dimension. Parameters: per_replica_value (object with a list of
  # component values) and mesh. The rank is taken from the first component (zero for plain
  # numbers). Scalar components are expanded to one dimension first because packing needs equal
  # rank, and the rank is increased by one. Returns the packed DTensor. A TODO notes the result
  # may in fact be a partial value rather than batch sharded.
  """Convert a PerReplica result to a DTensor instance.

  Args:
    per_replica_value: A PerReplica instance whose value will be converted
      to DTensor.
    mesh: The mesh used for layout creation.

  Returns:
    A DTensor instance that packed from per_replica_value with batch sharded
      layout.
  """
  values = per_replica_value.values
  if isinstance(values[0], (float, int)):
    rank = 0
  else:
    rank = len(values[0].shape)

  if rank == 0:
    result = []
    # dtensor.pack requires each component to have same rank as the packed
    # result. When the individual value is scalar, it needs to be expanded into
    # 1D tensor.
    for v in values:
      result.append(array_ops.expand_dims_v2(v, axis=0))
    rank += 1
  else:
    result = list(values)   # dtensor.pack requires a list as input.

  # TODO(scottzhu): Note that the result tensor could be a partial value and
  # not always batch shard or fully replicaed. See
  # http://screenshot/6ERkXyX95KqftCw as an example.
  batch_layout = layout.Layout.batch_sharded(
      mesh, batch_dim=DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)

  return d_api.pack(result, batch_layout)


def dtensor_reduce(strategy, reduce_op, value, axis):
  # [FUNCTION DOC]
  # Implements the strategy reduce operation for DTensor strategies, overriding the default
  # because scalar DTensors lose batch sharding information. Parameters: strategy, reduce_op (a
  # reduce op or its string name, case insensitive), value (distributed or ordinary), axis
  # (optional integer). Requires cross-replica or default context. A non-distributed value with no
  # axis is reduced the way non-distributed values are, on the first destination device. Otherwise
  # the value is converted to a DTensor; the mean op uses mean and every other op uses sum. A
  # fully replicated DTensor is reduced only over the axis if one is given; otherwise it is
  # reshaped so a leading replica dimension is added, reduced over the axis (shifted by one) if
  # given, and then reduced over the replica dimension. Returns a DTensor.
  """Implement dtensor based strategy.reduce()."""
  # Due to the limitation of using scalar in DTensor (e.g. the rank 0 tensor
  # loss the batch shard information), we need to override the default
  # reduce in addition to the strategy.extend._reduce_to()
  # Most of the logic here is a mimic of the parent class, except for how
  # mean and sum are calculated in a global context.
  distribute_lib._require_cross_replica_or_default_context_extended(  # pylint: disable=protected-access
      strategy.extended)
  if isinstance(reduce_op, str):
    reduce_op = reduce_util.ReduceOp(reduce_op.upper())

  distributed_input = is_distributed_value(value)
  if not distributed_input and axis is None:
    # For any value that isn't distributed and doesn't need a reduction within
    # the replica.
    destinations = (device_util.current() or
                    strategy.extended._default_device or  # pylint: disable=protected-access
                    "/device:CPU:0")
    devices = cross_device_ops_lib.get_devices_from(destinations)
    with ops.device(devices[0]):
      return array_ops.identity(
          cross_device_ops_lib.reduce_non_distributed_value(
              reduce_op, value, destinations, strategy.num_replicas_in_sync))

  value = convert_inputs_to_dtensor(value, strategy._mesh)  # pylint: disable=protected-access
  # At this point, the value is a DTensor instance now.
  # There will be a final reduction step cross replica. In order to maintain
  # the shape of each local replica, we need to add a new dim to the front.
  # E.g. 2 replica with local shape as (4, 5, 6), the global tensor shape
  # should be (8, 5, 6), we will reshape into (2, 4, 5, 6) and then do a
  # reduction on axis 0.
  if reduce_op == reduce_util.ReduceOp.MEAN:
    reduce_op = math_ops.reduce_mean
  else:
    reduce_op = math_ops.reduce_sum

  # TODO(scottzhu): Make sure we handle dynamic/uneven shape in future.
  if d_api.fetch_layout(value).is_fully_replicated():
    # In case of fully mirrored dtensor, we only need to do one reduce, and
    # don't need to care about any per-replica logic.
    if axis is not None:
      value = reduce_op(value, axis=axis)
  else:
    new_shape = [strategy.num_replicas_in_sync, -1]
    if len(value.shape) > 1:
      new_shape.extend(array_ops.shape(value)[1:])
    value = array_ops.reshape(value, new_shape)
    if axis is not None:
      # we do a reduce_sum/mean within each of the replica when axis is not
      # None. Add 1 to the axis since there is a new dim added by reshape in
      # front.
      value = reduce_op(value, axis=axis + 1)
    value = reduce_op(value, axis=0)

  # Note that we return a DTensor instance here, which should have the same
  # value as the original MirroredStrategy, but with a different type. User
  # might want a tf.Tensor for the status quo.
  return value
