# [MODULE DOC]
# Cross-device communication algorithms for tf.distribute: reducing, gathering, batch
# reducing and broadcasting values among devices (and workers). Defines the public base
# class for such algorithms and several implementations: reduction to one device, NCCL
# all-reduce, hierarchical-copy all-reduce, a packed all-reduce wrapper, and a collective
# all-reduce implementation (single-worker and multi-worker, with batching, packing and
# optional background threads). Also provides helper functions for validating destinations,
# resolving devices, simple copy-based reductions/broadcasts, and for choosing a default
# implementation given the available devices.
# Used by the mirrored and collective strategies. Import-time side effects: registers the
# public classes in the public API namespace via the export decorator; no other global state.
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
"""Classes for different algorithms of reduction and broadcasting."""

import collections
import copy
import multiprocessing.dummy
import multiprocessing.pool
import threading

import numpy as np
import six

from tensorflow.python.client import device_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls


def check_destinations(destinations):
  # [FUNCTION DOC]
  # Tells whether a destinations argument is non-empty. Input: a distributed value,
  # variable, tensor or device string (or other container). For variables and tensors it
  # returns whether the device string is non-empty (truth-testing the variable itself is not
  # allowed); for anything else it returns the general truthiness. Returns a boolean.
  """Checks whether `destinations` is not empty.

  Args:
    destinations: a `DistributedValues`, variable, or string object.

  Returns:
    Boolean which is True if `destinations` is not empty.
  """
  # Calling bool() on a ResourceVariable is not allowed.
  if isinstance(destinations,
                (resource_variable_ops.BaseResourceVariable,
                 tensor_lib.Tensor)):
    return bool(destinations.device)
  return bool(destinations)


def validate_destinations(destinations):
  # [FUNCTION DOC]
  # Validates a destinations argument. Accepts distributed values, tensors, indexed slices,
  # aggregating variables, device strings, TPU mirrored variables, or any resource variable.
  # Raises a value error if the type is not one of these, or if the destinations are empty
  # (per the emptiness check). Returns nothing.
  """Validates the `destination` is one of expected types."""
  if not isinstance(
      destinations,
      (value_lib.DistributedValues, tensor_lib.Tensor,
       indexed_slices.IndexedSlices, ps_values.AggregatingVariable,
       six.string_types, tpu_values.TPUMirroredVariable
      )) and not resource_variable_ops.is_resource_variable(destinations):
    raise ValueError("destinations must be one of a `DistributedValues` object,"
                     " a tf.Variable object, or a device string.")

  if not check_destinations(destinations):
    raise ValueError("destinations can not be empty")


def reduce_non_distributed_value(reduce_op,
                                 value,
                                 destinations,
                                 num_replicas_in_graph,
                                 canonicalize_devices=True):
  # [FUNCTION DOC]
  # Reduces a value that is not a distributed value (a single value standing for the same
  # value on all replicas). Inputs: reduce_op (sum or mean), value, destinations,
  # num_replicas_in_graph (integer), canonicalize_devices (default true; whether device
  # names are fully resolved). Raises a value error if value is a distributed value, or if the
  # op is not mean and there is more than one replica in the graph (sum of identical values
  # is undefined). Behavior: a non-tensor value that is all zeros yields a zero array of the
  # same shape and type; for mean the value is returned unchanged; for a single replica the
  # destinations are validated and the value is broadcast by simple copying.
  """Reduce a non-DistributedValue `value` to `destinations`."""
  if isinstance(value, value_lib.DistributedValues):
    raise ValueError("You are passing a `DistributedValues` to "
                     "`reduce_non_distributed_value`, which is not allowed.")

  # If the same value is present on all replicas then the PerReplica value will
  # be a single value. We also handle the case when `value` is a single value
  # and equal to 0.
  # TODO(b/138823479): handle the tensor value properly.
  if not tensor_util.is_tf_type(value) and np.all(value == 0):
    return np.zeros(value.shape, dtype=value.dtype)
  # If there is only a single value and the reduce op is MEAN,
  # that value should be on all destinations.
  if reduce_op == reduce_util.ReduceOp.MEAN:
    return value
  elif num_replicas_in_graph != 1:
    # We do not support a reduce op of SUM if the value is the same across
    # all replicas. We call this as part of assign functions for
    # MirroredVariables and summing up identical values across replicas is not
    # clearly defined.
    raise ValueError("A non-DistributedValues value %s cannot be reduced with "
                     "the given reduce op %s." % (value, reduce_op))
  else:
    validate_destinations(destinations)
    return simple_broadcast(
        value, destinations, canonicalize_devices=canonicalize_devices)


def _make_tensor_into_per_replica(input_tensor):
  # [FUNCTION DOC]
  # HELPER: input normalization for reduce/gather. Returns distributed values unchanged;
  # otherwise converts a non-tensor to a tensor, and wraps a tensor that has a device
  # attribute into a single-component per-replica value. Raises a value error if the tensor
  # has no device attribute.
  """Converts a single tensor into a PerReplica object."""
  if isinstance(input_tensor, value_lib.DistributedValues):
    return input_tensor

  # If input is not a Tensor, convert it to a Tensor first.
  if not tensor_util.is_tensor(input_tensor):
    input_tensor = ops.convert_to_tensor(input_tensor)

  if hasattr(input_tensor, "device"):
    return value_lib.PerReplica((input_tensor,))

  raise ValueError("Cannot convert `input_tensor` to a `PerReplica` object "
                   "because it doesn't have device set.")


def _normalize_value_destination_pairs(value_destination_pairs):
  # [FUNCTION DOC]
  # HELPER: input normalization for batch reduce. Takes an iterable of (value, destinations)
  # pairs, copies it to a list, and returns a new list in which each value is converted to a
  # per-replica value. Raises a value error if an element is not a tuple or not of size two.
  # (The check that the input is a list or tuple can never fail since it was just converted.)
  """Converts each tensor into a PerReplica object in the input list."""
  result = []

  value_destination_pairs = list(value_destination_pairs)

  if not isinstance(value_destination_pairs, (list, tuple)):
    raise ValueError("`value_destination_pairs` should be a list or tuple")
  for pair in value_destination_pairs:
    if not isinstance(pair, tuple):
      raise ValueError(
          "Each element of `value_destination_pairs` should be a tuple.")
    if len(pair) != 2:
      raise ValueError("Each element of `value_destination_pairs` should be a "
                       "tuple of size 2.")

    per_replica = _make_tensor_into_per_replica(pair[0])
    result.append((per_replica, pair[1]))
  return result


def _validate_value_destination_pairs(value_destination_pairs):
  # [FUNCTION DOC]
  # HELPER: predicate for batch reduce. Returns true only if the input is a non-empty list or
  # tuple whose elements are all tuples whose first item is a per-replica value; otherwise
  # false (it does not raise).
  """Validates value_destination_pairs are valid."""
  # TODO(yuefengz): raise exceptions instead of returning False.
  if not value_destination_pairs: return False
  if not isinstance(value_destination_pairs, (list, tuple)): return False
  if not all(isinstance(pair, tuple) for pair in value_destination_pairs):
    return False
  if not all(isinstance(v[0], value_lib.PerReplica)
             for v in value_destination_pairs):
    return False
  return True


# TODO(yuefengz): consider calling this function in the caller of
# CrossDeviceOps.
def get_devices_from(destinations, canonicalize_devices=True):
  # [FUNCTION DOC]
  # HELPER: device lookup shared by the reduction code. Input: destinations, a distributed
  # value, a device string, or an object with a device attribute; canonicalize_devices
  # (default true). Output: a tuple of device names. For distributed values, its device list
  # is returned. Otherwise a single-element tuple: fully resolved name when canonicalizing,
  # or the name stripped of job and task parts when not (leaving the placer to resolve it).
  if isinstance(destinations, value_lib.DistributedValues):
    return destinations._devices  # pylint: disable=protected-access
  if canonicalize_devices:
    if isinstance(destinations, six.string_types):
      return (device_util.resolve(destinations),)
    return (device_util.resolve(destinations.device),)

  # Let placer canonicalize and resolve destination devices.
  if isinstance(destinations, six.string_types):
    return (device_util.canonicalize_without_job_and_task(destinations),)
  return (device_util.canonicalize_without_job_and_task(destinations.device),)


def _devices_match(left, right, canonicalize_devices=True):
  # [FUNCTION DOC]
  # HELPER: compares destinations. Returns true if both are the same object or their device
  # sets are equal (devices obtained with the given canonicalization setting).
  return left is right or set(get_devices_from(
      left, canonicalize_devices)) == set(
          get_devices_from(right, canonicalize_devices))


def _all_devices_match(value_destination_pairs, canonicalize_devices=True):
  # [FUNCTION DOC]
  # HELPER: batch-reduce shortcut check. Input: list of (value, destination) pairs. Returns
  # true if every value's devices match its own destination's devices and all values share
  # the devices of the first value; otherwise false.
  if not all(
      _devices_match(v, d, canonicalize_devices)
      for v, d in value_destination_pairs):
    return False
  if not all(
      _devices_match(v, value_destination_pairs[0][0], canonicalize_devices)
      for v, _ in value_destination_pairs[1:]):
    return False
  return True


def simple_broadcast(value,
                     destinations,
                     always_mirrored=False,
                     canonicalize_devices=True):
  # [FUNCTION DOC]
  # Broadcasts a value to destinations by plain copies. Inputs: value (tensor or indexed
  # slices); destinations (distributed value, variable/tensor-like or device string);
  # always_mirrored (default false); canonicalize_devices (default true). With a single
  # destination device and not always_mirrored, returns the value copied to that device;
  # otherwise copies to each device and returns a mirrored value grouping the copies.
  """Broadcast `value` to `destinations` using simple copies."""
  devices = get_devices_from(destinations, canonicalize_devices)
  if len(devices) == 1 and not always_mirrored:
    return cross_device_utils.copy_tensor_or_indexed_slices_to_device(
        value, devices[0])
  else:
    value_updates = []
    for d in devices:
      value_updates.append(
          cross_device_utils.copy_tensor_or_indexed_slices_to_device(value, d))
    return distribute_utils.regroup(value_updates,
                                    wrap_class=value_lib.Mirrored)


def _simple_reduce(per_replica_value, reduce_to_device, accumulation_fn,
                   reduce_op):
  # [FUNCTION DOC]
  # HELPER: reduction used by the reduce-to-one-device implementation. Inputs:
  # per_replica_value (distributed value), reduce_to_device (device name), accumulation_fn
  # (callable summing a list of tensors), reduce_op (sum or mean). Raises a value error if
  # there are no component values or if reduce_op is neither sum nor mean. Aggregates all
  # component values on the target device (silent implicit copies allowed), dividing by the
  # count for mean. Returns the reduced tensor or indexed slices.
  """Reduces the value by accumulation_fn and reduce_op."""
  all_values = per_replica_value.values
  if not all_values:
    raise ValueError("`per_replica_value` must be non-empty")
  count = len(all_values)

  with ops.device(reduce_to_device):
    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
      reduced = cross_device_utils.aggregate_tensors_or_indexed_slices(
          all_values, accumulation_fn)
      if reduce_op == reduce_util.ReduceOp.MEAN:
        reduced = cross_device_utils.divide_by_n_tensors_or_indexed_slices(
            reduced, count)
      elif reduce_op != reduce_util.ReduceOp.SUM:
        raise ValueError("`reduce_op` must be Reduce.SUM or Reduce.MEAN.")
  return reduced


def _simple_gather(per_replica_value, reduce_to_device, axis):
  # [FUNCTION DOC]
  # HELPER: gather used by the reduce-to-one-device implementation. Concatenates all component
  # values of the distributed value along the given axis on the target device (silent implicit
  # copies allowed) and returns the result. Raises a value error if there are no components.
  """Concatenate all values in the DistributedValues input and return."""
  all_values = per_replica_value.values
  if not all_values:
    raise ValueError("`per_replica_value` must be non-empty")

  with ops.device(reduce_to_device):
    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
      gathered = array_ops.concat(all_values, axis)
  return gathered


@tf_export("distribute.CrossDeviceOps")
class CrossDeviceOps(object):
  # [CLASS DOC]
  # Public abstract base for cross-device reduction, gather and broadcast algorithms, passed
  # to mirrored strategies to choose a communication implementation. Provides the public
  # entry points (reduce, batch reduce, broadcast; private gather and all-reduce) that
  # validate inputs, apply shortcuts for single-valued cases, then delegate to implementation
  # hooks that subclasses override. State: a flag controlling whether device names are fully
  # canonicalized (true by default).
  """Base class for cross-device reduction and broadcasting algorithms.

  The main purpose of this class is to be passed to
  `tf.distribute.MirroredStrategy` in order to choose among different cross
  device communication implementations. Prefer using the methods of
  `tf.distribute.Strategy` instead of the ones of this class.

  Implementations:
  * `tf.distribute.ReductionToOneDevice`
  * `tf.distribute.NcclAllReduce`
  * `tf.distribute.HierarchicalCopyAllReduce`
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Sets the device-canonicalization flag to true. Takes no inputs.
    self._canonicalize_devices = True
    pass

  @property
  def _num_between_graph_workers(self):
    # [FUNCTION DOC]
    # HELPER: hook for subclasses. Property giving the number of between-graph workers; the
    # base class always reports one.
    # Returns 1 by default, the value may be overridden by sub classes.
    return 1

  def reduce(self, reduce_op, per_replica_value, destinations, options=None):
    # [FUNCTION DOC]
    # Reduces a value across replicas to the destinations (cross-replica context only).
    # Inputs: reduce_op (sum or mean), per_replica_value (distributed value or tensor-like),
    # destinations (distributed value, variable, tensor-like or device string), options
    # (optional communication options; defaults created when absent). Behavior: converts the
    # value into a per-replica value, validates the destinations (value error if invalid or
    # empty), and if there is a single between-graph worker, a single component and matching
    # devices, returns an identity copy on that device as a mirrored value. Otherwise delegates
    # to the reduce implementation hook. Returns a tensor or distributed value.
    """Reduce `per_replica_value` to `destinations`.

    See `tf.distribute.StrategyExtended.reduce_to`. This can only be called in
    the cross-replica context.

    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      per_replica_value: a `tf.distribute.DistributedValues`, or a `tf.Tensor`
        like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to reduce to. To perform an all-reduce, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is reduced
        to the devices of that variable, and this method doesn't update the
        variable.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.

    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.

    Raises:
      ValueError: if per_replica_value can't be converted to a
        `tf.distribute.DistributedValues` or if destinations is not a string,
        `tf.Variable` or `tf.distribute.DistributedValues`.
    """
    if options is None:
      options = collective_util.Options()

    per_replica_value = _make_tensor_into_per_replica(per_replica_value)

    validate_destinations(destinations)

    # Shortcut if `per_replica_value` only contains one value.
    if self._num_between_graph_workers == 1 and len(
        per_replica_value.values) == 1 and _devices_match(
            per_replica_value, destinations, self._canonicalize_devices):
      with ops.device(per_replica_value.values[0].device):
        v = array_ops.identity(per_replica_value.values[0])
      return distribute_utils.regroup((v,), wrap_class=value_lib.Mirrored)

    if options is None:
      options = collective_util.Options()
    return self.reduce_implementation(reduce_op, per_replica_value,
                                      destinations, options)

  def _gather(self, per_replica_value, destinations, axis, options=None):
    # [FUNCTION DOC]
    # Gathers (concatenates) a value from all replicas to destinations. Inputs:
    # per_replica_value, destinations, axis (dimension to concatenate along), options
    # (optional; defaults created). Raises not-implemented for indexed slices, and a value
    # error for invalid or empty destinations. Same single-component identity shortcut as
    # reduce; otherwise delegates to the gather implementation hook. Returns a tensor or
    # distributed value.
    """Gather `per_replica_value` to `destinations`.

    Args:
      per_replica_value: a `tf.distribute.DistributedValues`, or a `tf.Tensor`
        like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to gather to. To perform an all-gather, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is gathered
        to the devices of that variable, and this method doesn't update the
        variable.
      axis: specifies the dimension to gather along within each replica's
        tensor.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.

    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`

    Raises:
      ValueError: if per_replica_value can't be converted to a
        `tf.distribute.DistributedValues` or if destinations is not a string,
        `tf.Variable` or `tf.distribute.DistributedValues`.
    """
    if isinstance(per_replica_value, indexed_slices.IndexedSlices):
      raise NotImplementedError("gather/all_gather does not support "
                                "IndexedSlices")
    if options is None:
      options = collective_util.Options()

    per_replica_value = _make_tensor_into_per_replica(per_replica_value)

    validate_destinations(destinations)

    # Shortcut if `per_replica_value` only contains one value.
    if self._num_between_graph_workers == 1 and len(
        per_replica_value.values) == 1 and _devices_match(
            per_replica_value, destinations, self._canonicalize_devices):
      with ops.device(per_replica_value.values[0].device):
        v = array_ops.identity(per_replica_value.values[0])
      return distribute_utils.regroup((v,), wrap_class=value_lib.Mirrored)

    return self._gather_implementation(per_replica_value, destinations, axis,
                                       options)

  def _gather_implementation(self, per_replica_value, destinations, axis,
                             options):
    # [FUNCTION DOC]
    # Subclass hook for gather. The base version always raises a not-implemented error.
    # Inputs as for the gather entry point.
    """Implementation of `gather` method of `tf.distribute.CrossDeviceOps`.

    Overriding this method is useful for subclass implementers.

    Args:
      per_replica_value: a `tf.distribute.DistributedValues`, or a `tf.Tensor`
        like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to gather to. To perform an all-gather, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is gathered
        to the devices of that variable, this method doesn't update the
        variable.
      axis: specifies the dimension to gather along within each replica's
        tensor.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.

    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.

    Raises:
      ValueError: if per_replica_value can't be converted to a
        `tf.distribute.DistributedValues` or if destinations is not a string,
        `tf.Variable` or `tf.distribute.DistributedValues`.
    """
    raise NotImplementedError(
        "_gather method must be implemented in descendants.")

  def batch_reduce(self, reduce_op, value_destination_pairs, options=None):
    # [FUNCTION DOC]
    # Reduces many values to their destinations at once (cross-replica context only).
    # Inputs: reduce_op, value_destination_pairs (sequence of value and destination pairs),
    # options (optional; defaults created). If the pairs are not already per-replica pairs,
    # tensors are normalized (value error for malformed pairs). Validates each destination.
    # If there is one between-graph worker, all devices match and the first value has a single
    # component, returns mirrored values built from the components without communication.
    # Otherwise delegates to the batch reduce hook. Returns a list with one result per pair.
    """Reduce values to destinations in batches.

    See `tf.distribute.StrategyExtended.batch_reduce_to`. This can only be
    called in the cross-replica context.

    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      value_destination_pairs: a sequence of (value, destinations) pairs. See
        `tf.distribute.CrossDeviceOps.reduce` for descriptions.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.

    Returns:
      A list of `tf.Tensor` or `tf.distribute.DistributedValues`, one per pair
      in `value_destination_pairs`.

    Raises:
      ValueError: if `value_destination_pairs` is not an iterable of
        tuples of `tf.distribute.DistributedValues` and destinations.
    """
    if options is None:
      options = collective_util.Options()
    # TODO(yuefengz): if destinations are different, split into several
    # `_batch_reduce` invocations.
    if not _validate_value_destination_pairs(value_destination_pairs):
      # If the first element of each pair is a tensor, we try to turn it into a
      # PerReplica object.
      value_destination_pairs = _normalize_value_destination_pairs(
          value_destination_pairs)

    for _, d in value_destination_pairs:
      validate_destinations(d)

    # Shortcut all PerReplica objects only contain one value.
    if self._num_between_graph_workers == 1 and _all_devices_match(
        value_destination_pairs, self._canonicalize_devices) and len(
            value_destination_pairs[0][0].values) == 1:
      return [
          distribute_utils.regroup(v.values, wrap_class=value_lib.Mirrored)
          for v, _ in value_destination_pairs
      ]

    if options is None:
      options = collective_util.Options()
    return self.batch_reduce_implementation(reduce_op, value_destination_pairs,
                                            options)

  def broadcast(self, tensor, destinations):
    # [FUNCTION DOC]
    # Broadcasts a tensor to destinations (cross-replica context only). Validates the
    # destinations (value error if invalid or empty) then delegates to the broadcast hook.
    # Returns a tensor or distributed value.
    """Broadcast `tensor` to `destinations`.

    This can only be called in the cross-replica context.

    Args:
      tensor: a `tf.Tensor` like object. The value to broadcast.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to broadcast to. Note that if it's a `tf.Variable`, the value is
        broadcasted to the devices of that variable, this method doesn't update
        the variable.

    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.
    """
    validate_destinations(destinations)
    return self.broadcast_implementation(tensor, destinations)

  @doc_controls.for_subclass_implementers
  def reduce_implementation(self, reduce_op, per_replica_value, destinations,
                            options):
    # [FUNCTION DOC]
    # Subclass hook for reduce. The base version always raises a not-implemented error.
    """Implementation of `reduce`.

    Overriding this method is useful for subclass implementers.

    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      per_replica_value: a `tf.distribute.DistributedValues`, or a `tf.Tensor`
        like object.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to reduce to. To perform an all-reduce, pass the same to `value` and
        `destinations`. Note that if it's a `tf.Variable`, the value is reduced
        to the devices of that variable, this method doesn't update the
        variable.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.

    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.

    Raises:
      ValueError: if per_replica_value can't be converted to a
        `tf.distribute.DistributedValues` or if destinations is not a string,
        `tf.Variable` or `tf.distribute.DistributedValues`.
    """
    raise NotImplementedError(
        "_reduce method must be implemented in descendants.")

  @doc_controls.for_subclass_implementers
  def batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                  options):
    # [FUNCTION DOC]
    # Subclass hook for batch reduce. The base version always raises a not-implemented error.
    """Implementation of `batch_reduce`.

    Overriding this method is useful for subclass implementers.

    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      value_destination_pairs: a sequence of (value, destinations) pairs. See
        `reduce` for descriptions.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.

    Returns:
      A list of `tf.Tensor` or `tf.distribute.DistributedValues`, one per pair
      in `value_destination_pairs`.

    Raises:
      ValueError: if `value_destination_pairs` is not an iterable of
        tuples of `tf.distribute.DistributedValues` and destinations.
    """
    raise NotImplementedError(
        "batch_reduce_implementation method must be implemented in descendants."
    )

  @doc_controls.for_subclass_implementers
  def broadcast_implementation(self, tensor, destinations):
    # [FUNCTION DOC]
    # Default broadcast hook: copies the tensor to every destination device (always producing a
    # mirrored value) using simple copies and this object's canonicalization setting.
    """Implementation of `broadcast`.

    Args:
      tensor: a `tf.Tensor` like object. The value to broadcast.
      destinations: a `tf.distribute.DistributedValues`, a `tf.Variable`, a
        `tf.Tensor` alike object, or a device string. It specifies the devices
        to broadcast to.
        `destinations`. Note that if it's a `tf.Variable`, the value is
        broadcasted to the devices of that variable, this method doesn't update
        the variable.

    Returns:
      A `tf.Tensor` or `tf.distribute.DistributedValues`.
    """
    return simple_broadcast(
        tensor,
        destinations,
        always_mirrored=True,
        canonicalize_devices=self._canonicalize_devices)

  # ========================== Collective APIs ================================
  #
  # Different than `reduce`, `batch_reduce` and `broadcast` which must be called
  # in cross-replcia context, collective APIs are to be called in replica
  # context.

  def _all_reduce(self, reduce_op, value, replica_id, options):
    # [FUNCTION DOC]
    # Replica-context hook for all-reduce of a tensor or nested structure of tensors and indexed
    # slices, given the local replica id and communication options. The base version always
    # raises a not-implemented error; collective implementations override it.
    """All-reduce the `value` across all replicas so that all get the result.

    `value` can be a nested structure of tensors or `IndexedSlices`. The
    implementation should generally batch the all-reduces when possible.
    `options` can be set to hint the batching behavior.

    This API must be called in a replica context.

    Args:
      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should
        be combined.
      value: Value to be reduced. A tensor or a nested structure of tensors or
        `IndexedSlices`.
      replica_id: An integer indicating the id of the replica where this
        all_reduce is called under. This is the local replica id that ranges
        from 0 to len(local_devices) - 1.
      options: A `tf.distribute.experimental.CommunicationOptions`.

    Returns:
      A tensor/IndexedSlices or a nested structure of tensors/IndexedSlices with
      the reduced values. The structure is the same as `value`.
    """
    raise NotImplementedError("_all_reduce must be implemented in descendants.")


@tf_export("distribute.ReductionToOneDevice")
class ReductionToOneDevice(CrossDeviceOps):
  # [CLASS DOC]
  # Simple cross-device implementation: copies all values to a single device, reduces them there
  # with an accumulation function, and broadcasts the result to the destinations. Holds the
  # chosen intermediate device (or nothing to use the first destination device) and the
  # accumulation function. No efficient batching: batch reduce just reduces each pair in turn.
  """A CrossDeviceOps implementation that copies values to one device to reduce.

  This implementation always copies values to one device to reduce them, then
  broadcast reduced values to the destinations. It doesn't support efficient
  batching.

  Here is how you can use `ReductionToOneDevice` in
  `tf.distribute.MirroredStrategy`:

  ```
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.ReductionToOneDevice())
  ```
  """

  def __init__(self, reduce_to_device=None, accumulation_fn=None):
    # [FUNCTION DOC]
    # Constructor. reduce_to_device: optional device name for the intermediate reduction
    # (nothing means the first destination device at reduce time). accumulation_fn: optional
    # callable summing a list of tensors (defaults to the add-n operation). Stores both and
    # calls the base constructor.
    """Initializes with a device to reduce to and a way to accumulate.

    Args:
      reduce_to_device: the intermediate device to reduce to. If None, reduce
        to the first device in `destinations` of the `reduce` method.
      accumulation_fn: a function that does accumulation.  If None,
        `tf.math.add_n` is used.
    """
    self.reduce_to_device = reduce_to_device
    self.accumulation_fn = accumulation_fn or math_ops.add_n
    super(ReductionToOneDevice, self).__init__()

  def reduce_implementation(self, reduce_op, per_replica_value, destinations,
                            options):
    # [FUNCTION DOC]
    # Reduce hook. Ignores options. Picks the target devices from the destinations when they are
    # non-empty, else from the value itself; the intermediate device is the configured one or the
    # first target device. Logs (limited to the first ten times) the plan, reduces the value on
    # that device with the accumulation function and reduce_op (sum or mean, else value error),
    # then broadcasts the result to the destinations and returns it.
    del options  # Unused.
    if check_destinations(destinations):
      devices = get_devices_from(destinations, self._canonicalize_devices)
    else:
      devices = get_devices_from(per_replica_value, self._canonicalize_devices)
    reduce_to_device = self.reduce_to_device or devices[0]
    logging.log_first_n(
        logging.INFO,
        "Reduce to %s then broadcast to %r." % (reduce_to_device, devices), 10)
    reduced = _simple_reduce(per_replica_value, reduce_to_device,
                             self.accumulation_fn, reduce_op)
    return self.broadcast(reduced, destinations)

  def _gather_implementation(self, per_replica_value, destinations, axis,
                             options):
    # [FUNCTION DOC]
    # Gather hook. Ignores options. Chooses devices as the reduce hook does, logs the plan
    # (first ten times only), concatenates all components along the axis on the intermediate
    # device, then broadcasts the gathered tensor to the destinations and returns it.
    del options  # Unused.
    if check_destinations(destinations):
      devices = get_devices_from(destinations, self._canonicalize_devices)
    else:
      devices = get_devices_from(per_replica_value, self._canonicalize_devices)
    reduce_to_device = self.reduce_to_device or devices[0]
    logging.log_first_n(
        logging.INFO,
        "Gather to %s then broadcast to %r." % (reduce_to_device, devices), 10)
    gathered = _simple_gather(per_replica_value, reduce_to_device, axis)
    return self.broadcast(gathered, destinations)

  def batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                  options):
    # [FUNCTION DOC]
    # Batch reduce hook with no batching: reduces each (value, destinations) pair separately
    # through the reduce hook and returns the list of results in order.
    return [
        self.reduce_implementation(
            reduce_op, t, destinations=v, options=options)
        for t, v in value_destination_pairs
    ]


def _group_value_by_device(per_replica_values):
  # [FUNCTION DOC]
  # HELPER: prepares input for the all-reduce libraries. Takes a non-empty list of per-replica
  # values sharing the same devices (asserted) and returns a list with one sublist per device,
  # each holding (component, nothing) pairs for that device in input order.
  """Group values into sublists by their devices.

  This grouping is needed to call the all-reduce library because it expects a
  list of the following form:
    [[(grad0_gpu0, v0_gpu0), (grad1_gpu0, v1_gpu0), (grad2_gpu0, v2_gpu0) ...],
     [(grad0_gpu1, v0_gpu1), (grad1_gpu1, v1_gpu1), (grad2_gpu1, v2_gpu1) ...],
     [(grad0_gpu2, v0_gpu2), (grad1_gpu0, v1_gpu2), (grad2_gpu0, v2_gpu2) ...],
     ...
    ]

  Args:
    per_replica_values: a list of PerReplica objects.

  Returns:
    a list of lists, each sublist has components for its corresponding device of
      PerReplica objects, paired with a None.
  """
  destinations = per_replica_values[0]._devices  # pylint: disable=protected-access
  grouped = [[] for _ in range(len(destinations))]
  for per_replica_value in per_replica_values:
    # pylint: disable=protected-access
    for i, v in enumerate(per_replica_value.values):
      assert per_replica_value._devices == destinations
      grouped[i].append((v, None))
  return grouped


def _ungroup_and_make_mirrored(grouped_reduced,
                               destinations,
                               reduce_op,
                               num_between_graph_workers=1):
  # [FUNCTION DOC]
  # HELPER: converts all-reduce library output back into mirrored values. Inputs:
  # grouped_reduced (list of per-device sublists of pairs), destinations (device holder, used to
  # count devices), reduce_op, num_between_graph_workers (default one). Output: list of mirrored
  # values, one per original value. See the rest of the function for the mean division.
  """Ungroup results from all-reduce and make Mirrored objects.

  Each all-reduce result will be divided by the number of destinations before
  Mirrored objects are created if reduce_op is "mean".

  Args:
    grouped_reduced: a list of lists, each sublist has components for each
      device, paired with a None. It is the result from
      cross_device_utils.aggregate_gradients_using*.
    destinations: a value to colocate the result with.
    reduce_op: Indicates how values will be aggregated. Accepted values
      are `tf.distribute.ReduceOp.SUM`, `tf.distribute.ReduceOp.MEAN`.
    num_between_graph_workers: number of workers in the between-graph
      replication.

  Returns:
    a list of Mirrored objects.
  """
  # (Mean handling: for mean, each reduced component is divided by the number of destination devices
  # times the number of between-graph workers, computed on that component's device.)
  num_replicas = len(get_devices_from(destinations)) * num_between_graph_workers
  index = [[] for _ in range(len(grouped_reduced[0]))]
  for per_replica_reduced in grouped_reduced:
    for i, (v, _) in enumerate(per_replica_reduced):
      if reduce_op == reduce_util.ReduceOp.MEAN:
        with ops.device(v.device):
          index[i].append(v / num_replicas)
      else:
        index[i].append(v)
  return [distribute_utils.regroup(
      v, wrap_class=value_lib.Mirrored) for v in index]


class _ConcatAndSplitPacker(object):
  # [CLASS DOC]
  # HELPER: tensor packing for all-reduce. Concatenates all per-device gradients into one flat
  # tensor and splits it into a configured number of packs for efficient communication, and can
  # reverse the process. State: the number of packs; after packing it also remembers the input
  # grouping and, per device, the original shapes and sizes needed to unpack. Not reusable across
  # unrelated pack calls without repacking, since pack overwrites that remembered state.
  """Concatenate and split tensors for reduction."""

  def __init__(self, num_packs=1):
    # [FUNCTION DOC]
    # Constructor. num_packs (default one) is the number of packs to split into. Raises a value
    # error if it is not greater than zero.
    """Initialize the _ConcatAndSplitPacker object.

    Args:
      num_packs: specifies the number of split packs that will be
        formed.

    Raises:
      ValueError: if num_packs is not greater than 0.
    """
    if num_packs <= 0:
      raise ValueError("num_packs must be greater than zero.")
    self.num_packs = num_packs

  def pack(self, grouped_grads_and_vars):
    # [FUNCTION DOC]
    # Packs gradients. Input: a list with one sublist per device of (gradient, variable) pairs.
    # Side effects: remembers the input and records per-device original shapes and sizes. For
    # each device (colocated with its first gradient) it flattens every gradient, concatenates
    # them, computes the total size (statically when all shapes are fully defined, otherwise
    # dynamically), splits into the configured number of packs (equal sizes, the last pack takes
    # the remainder), and pairs each pack with nothing. Returns a list with, per device, an
    # iterable of (pack, nothing) pairs.
    """Pack tensors."""
    self.grouped_grads_and_vars = grouped_grads_and_vars
    self.all_device_shapes = []
    self.all_device_sizes = []

    device_grad_packs = []
    for device_grads_and_vars in grouped_grads_and_vars:
      with ops.colocate_with(device_grads_and_vars[0][0]):
        # Flatten all the grads.
        flat_grads = [
            array_ops.reshape(g, [-1]) for g, _ in device_grads_and_vars
        ]
        # Remember the original shape of all the grads.
        device_shapes = [array_ops.shape(g) for g, _ in device_grads_and_vars]
        # Remember the original sizes of all the grads.
        device_sizes = [array_ops.size(g) for g, _ in device_grads_and_vars]
        # Concat all the flat grads into a big flat tensor.
        concat_grads = array_ops.concat(flat_grads, 0)

        # Split the big tensor into num_splits packs. In cases where the
        # total size is not divisible num_splits, the last pack gets
        # more elements.
        # TODO(zhengxq): it is also possible to optimize away all the concat
        # as well.
        num_splits = self.num_packs

        # The array_ops.size function will sometimes remove static shapes. So if
        # all gradient shapes are defined, we use another method to get the
        # total size.
        # TODO(yuefengz): move this logic to array_ops.size.
        if all(g.shape.is_fully_defined() for g, _ in device_grads_and_vars):
          total_grad_size = sum(
              [g.shape.num_elements() for g, _ in device_grads_and_vars])
        else:
          total_grad_size = array_ops.size(concat_grads)

        split_size = total_grad_size // num_splits
        split_size_last = total_grad_size - split_size * (num_splits - 1)
        split_sizes = [split_size] * (num_splits - 1) + [split_size_last]
        grad_packs = array_ops.split(concat_grads, split_sizes)

        # Ready to aggregate the repacked gradients, with fake variables.
        # TODO(zhengxq): It is hacky to have to use fake variables.
        # We should remove the need for variables in
        # aggregate_gradients_using*.
        device_grad_packs.append(zip(grad_packs, [None] * num_splits))
        self.all_device_shapes.append(device_shapes)
        self.all_device_sizes.append(device_sizes)

    return device_grad_packs

  def unpack(self, summed_device_grad_packs):
    # [FUNCTION DOC]
    # Reverses the packing. Input: per-device lists of (reduced pack, nothing) pairs, in the same
    # device order used by the earlier pack call (relies on state remembered by it). For each
    # device it concatenates the packs, splits by the remembered original sizes, reshapes to the
    # original shapes, and re-pairs each result with its original variable. Returns a list with
    # one list of (gradient, variable) pairs per device.
    """Reverse the pack."""
    aggregated_device_grads = []
    for (summed_device_grad_packs,
         device_grads_and_vars, device_shapes, device_sizes) in zip(
             summed_device_grad_packs, self.grouped_grads_and_vars,
             self.all_device_shapes, self.all_device_sizes):
      # pylint: enable=line-too-long
      # Reverse the packing operations in the previous steps. Form the
      # summed gradients back into their original shapes.
      with ops.colocate_with(summed_device_grad_packs[0][0]):
        # Form a list of the summed grad packs.
        device_grad_packs = [g for g, _ in summed_device_grad_packs]

        # Concat them back into a big flat tensor.
        device_grads_concat = array_ops.concat(device_grad_packs, 0)

        # Split the tensors back into their original sizes.
        grads_with_sizes = array_ops.split(device_grads_concat, device_sizes)

        # Reshape the tensors back into their original shapes.
        grads_with_shapes = [
            array_ops.reshape(grad, shape)
            for shape, grad in zip(device_shapes, grads_with_sizes)
        ]

        # Form the list with the original list of variables.
        summed_device_grads = [
            (g, v) for g, (_, v) in zip(grads_with_shapes,
                                        device_grads_and_vars)
        ]
        aggregated_device_grads.append(summed_device_grads)
    return aggregated_device_grads


def _pack_tensors(device_grads, num_packs=0):
  # [FUNCTION DOC]
  # HELPER: optional packing step for all-reduce. If num_packs is positive, creates a packer and
  # packs the grouped gradients; otherwise passes them through. Returns a pair of (packed or
  # original gradients, packer or nothing).
  """Pack tensors if specified."""
  if num_packs > 0:
    tensor_packer = _ConcatAndSplitPacker(num_packs)
    device_grad_packs = tensor_packer.pack(device_grads)
  else:
    tensor_packer = None
    device_grad_packs = device_grads
  return device_grad_packs, tensor_packer


def _unpack_tensors(reduced, tensor_packer=None):
  # [FUNCTION DOC]
  # HELPER: inverse of the optional packing step. If a packer is given, unpacks the reduced
  # values with it; otherwise returns the reduced values unchanged.
  """Unpack tensors if they are packed before all-reduce."""
  if tensor_packer:
    return tensor_packer.unpack(reduced)
  return reduced


class AllReduceCrossDeviceOps(CrossDeviceOps):
  # [CLASS DOC]
  # Cross-device implementation that performs real all-reduce (via NCCL or hierarchical copy)
  # when source and destination devices match and none are CPUs, with optional tensor packing;
  # all other cases (and gather) fall back to a reduce-to-one-device helper. State: algorithm
  # name, pack count, and the fallback implementation. Base class for the NCCL and
  # hierarchical-copy public classes.
  """All-reduce implementation of CrossDeviceOps.

  It performs all-reduce when applicable using NCCL or hierarchical copy. For
  the batch API, tensors will be repacked or aggregated for more efficient
  cross-device transportation.

  For reduces that are not all-reduce, it falls back to
  `tf.distribute.ReductionToOneDevice`.
  """

  def __init__(self, all_reduce_alg="nccl", num_packs=1):
    # [FUNCTION DOC]
    # Constructor. all_reduce_alg: algorithm name, "nccl" (default) or "hierarchical_copy"
    # (not validated here; any other name behaves as hierarchical copy). num_packs: non-negative
    # integer (default one; zero means no packing; not validated here). Creates the fallback
    # reduce-to-one-device helper and calls the base constructor.
    """Initializes the object.

    Args:
      all_reduce_alg: the all-reduce algorithm to use, currently only "nccl" or
        "hierarchical_copy" are supported.
      num_packs: a non-negative integer. The number of packs to split values
        into. If zero, no packing will be done.
    """
    self._all_reduce_alg = all_reduce_alg
    self._num_packs = num_packs
    self._simple_cross_replica_ops = ReductionToOneDevice()
    super(AllReduceCrossDeviceOps, self).__init__()

  def reduce_implementation(self, reduce_op, per_replica_value, destinations,
                            options):
    # [FUNCTION DOC]
    # Reduce hook. Ignores options. If the value's devices equal the destination devices and no
    # device is a CPU, performs a batched all-reduce on the single value and returns its result;
    # otherwise delegates to the fallback reduce-to-one-device helper.
    del options  # Unused.
    # To use NCCL or all-reduce, source and destination devices should match,
    # and none of the devices should be CPU.
    if (_devices_match(per_replica_value, destinations) and
        not any("cpu" in d.lower() for d in get_devices_from(destinations))):
      return self._batch_all_reduce(reduce_op, [per_replica_value])[0]
    else:
      return self._simple_cross_replica_ops.reduce(reduce_op, per_replica_value,
                                                   destinations)

  def batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                  options):
    # [FUNCTION DOC]
    # Batch reduce hook. If all values and destinations share the same devices, runs one batched
    # all-reduce over all the values; otherwise reduces each pair separately through the reduce
    # hook. Returns a list of results, one per pair.
    if _all_devices_match(value_destination_pairs):
      return self._batch_all_reduce(reduce_op,
                                    [v[0] for v in value_destination_pairs])
    else:
      return [
          self.reduce_implementation(reduce_op, value, dest, options)
          for value, dest in value_destination_pairs
      ]

  def _batch_all_reduce(self, reduce_op, per_replica_values):
    # [FUNCTION DOC]
    # HELPER: batched all-reduce driver. Splits the values into dense and sparse (indexed slices)
    # sets, runs the dense batch all-reduce and the sparse fallback batch as needed (skipping
    # empty sets), then stitches results back into the original order. Returns the list of
    # reduced values.
    """All-reduce algorithm in a batch."""
    dense_values, dense_indices, sparse_values, sparse_indices = (
        cross_device_utils.split_by_sparsity(per_replica_values))
    if dense_values:
      dense_results = self._do_batch_all_reduce(reduce_op, dense_values)
    else:
      dense_results = []
    if sparse_values:
      sparse_results = self._do_batch_all_reduce_sparse(reduce_op,
                                                        sparse_values)
    else:
      sparse_results = []
    return cross_device_utils.stitch_values(((dense_results, dense_indices),
                                             (sparse_results, sparse_indices)))

  def _do_batch_all_reduce(self, reduce_op, dense_values):
    # [FUNCTION DOC]
    # HELPER: all-reduce of dense per-replica values. Logs the plan (first ten times), groups the
    # values by device, optionally packs them, aggregates using NCCL when the algorithm is nccl
    # or using hierarchical copy otherwise, unpacks, and converts results back to mirrored values
    # (dividing by the replica count for mean). Returns a list of mirrored values.
    """Run batch all-reduces."""
    logging.log_first_n(
        logging.INFO,
        "batch_all_reduce: %d all-reduces with algorithm = %s, num_packs = %d" %
        (len(dense_values), self._all_reduce_alg, self._num_packs), 10)

    destinations = dense_values[0]._devices  # pylint: disable=protected-access
    grouped = _group_value_by_device(dense_values)

    # device_grad_packs:
    # [[(t0_gpu0, None), (t1_gpu0, None)], [(t0_gpu1, None), (t1_gpu1, None)]]
    device_grad_packs, tensor_packer = _pack_tensors(grouped, self._num_packs)

    # The actual aggregation of the repacked gradients. Note that they are
    # sharded among different aggregation trees. So it is important to strike
    # the balance on num_splits.
    if self._all_reduce_alg == "nccl":
      # TODO(yuefengz): merge this into the all-reduce library.
      reduced = cross_device_utils.aggregate_gradients_using_nccl(
          device_grad_packs)
    else:
      # TODO(yuefengz): check that gpu ids in `destinations` are in ascending
      # order.
      reduced = (
          cross_device_utils.aggregate_gradients_using_hierarchical_copy(
              destinations, device_grad_packs))

    reduced = _unpack_tensors(reduced, tensor_packer)
    return _ungroup_and_make_mirrored(reduced, dense_values[0], reduce_op)

  def _do_batch_all_reduce_sparse(self, reduce_op, sparse_values):
    # [FUNCTION DOC]
    # HELPER: fallback for indexed slices. Logs a warning (first ten times) that efficient
    # all-reduce is unsupported, then batch-reduces using each value as its own destination via
    # the reduce-to-one-device helper (effectively an inefficient all-gather). Returns the list
    # of results.
    """Run batch all-reduce for sparse values."""
    logging.log_first_n(
        logging.WARN,
        "Efficient allreduce is not supported for %d IndexedSlices" %
        len(sparse_values), 10)
    # Use `sparse_values` as destinations to do all-reduces. It is effectively
    # an allgather under the hood but not an efficient one.
    return self._simple_cross_replica_ops.batch_reduce(
        reduce_op, zip(sparse_values, sparse_values))

  def _gather_implementation(self, per_replica_value, destinations, axis,
                             options):
    # [FUNCTION DOC]
    # Gather hook. Logs a warning (first three times) that efficient gather is unsupported, then
    # gathers on one device and broadcasts using a new reduce-to-one-device helper, passing all
    # inputs through. Returns its result.
    logging.log_first_n(
        logging.WARN,
        "gather/all_gather with NCCL or HierarchicalCopy is not supported. "
        "Falling back to gather on one device and then broadcast. We're working"
        " on a more efficient implementation.", 3)
    return ReductionToOneDevice()._gather(per_replica_value, destinations, axis,  # pylint: disable=protected-access
                                          options)


# For compatibility with code using the old name of `AllReduceCrossDeviceOps`.
AllReduceCrossTowerOps = AllReduceCrossDeviceOps


AllReduceSpecTuple = collections.namedtuple("AllReduceSpecTuple",
                                            "alg shards limit")


@tf_export("distribute.NcclAllReduce")
class NcclAllReduce(AllReduceCrossDeviceOps):
  # [CLASS DOC]
  # Public all-reduce implementation using NVIDIA NCCL; a thin specialization of the generic
  # all-reduce class with the algorithm fixed to NCCL. Registered in the public API.
  """NCCL all-reduce implementation of CrossDeviceOps.

  It uses Nvidia NCCL for all-reduce. For the batch API, tensors will be
  repacked or aggregated for more efficient cross-device transportation.

  For reduces that are not all-reduce, it falls back to
  `tf.distribute.ReductionToOneDevice`.

  Here is how you can use `NcclAllReduce` in `tf.distribute.MirroredStrategy`:


  ```
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.NcclAllReduce())
  ```
  """

  def __init__(self, num_packs=1):
    # [FUNCTION DOC]
    # Constructor. num_packs: non-negative integer (default one; zero disables packing). Raises
    # a value error if negative. Calls the base constructor with the NCCL algorithm.
    """Initializes the object.

    Args:
      num_packs: a non-negative integer. The number of packs to split values
        into. If zero, no packing will be done.

    Raises:
      ValueError: if `num_packs` is negative.
    """
    if num_packs < 0:
      raise ValueError(
          "NCCL all-reduce requires num_packs >= 0, but {} is specified".format(
              num_packs))
    super(NcclAllReduce, self).__init__(
        all_reduce_alg="nccl", num_packs=num_packs)


@tf_export("distribute.HierarchicalCopyAllReduce")
class HierarchicalCopyAllReduce(AllReduceCrossDeviceOps):
  # [CLASS DOC]
  # Public all-reduce implementation that reduces along a fixed GPU hierarchy (designed for
  # DGX-1 style topology) and broadcasts back; a thin specialization of the generic all-reduce
  # class with the algorithm fixed to hierarchical copy. Registered in the public API.
  """Hierarchical copy all-reduce implementation of CrossDeviceOps.

  It reduces to one GPU along edges in some hierarchy and broadcasts back to
  each GPU along the same path. For the batch API, tensors will be repacked or
  aggregated for more efficient cross-device transportation.

  This is a reduction created for Nvidia DGX-1 which assumes GPUs connects like
  that on DGX-1 machine. If you have different GPU inter-connections, it is
  likely that it would be slower than `tf.distribute.ReductionToOneDevice`.

  For reduces that are not all-reduce, it falls back to
  `tf.distribute.ReductionToOneDevice`.

  Here is how you can use `HierarchicalCopyAllReduce` in
  `tf.distribute.MirroredStrategy`:

  ```
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
  ```
  """

  def __init__(self, num_packs=1):
    # [FUNCTION DOC]
    # Constructor. num_packs: non-negative integer (default one; zero disables packing). Raises
    # a value error if negative. Calls the base constructor with the hierarchical-copy algorithm.
    """Initializes the object.

    Args:
      num_packs: a non-negative integer. The number of packs to split values
        into. If zero, no packing will be done.

    Raises:
      ValueError if `num_packs` is negative.
    """
    if num_packs < 0:
      raise ValueError(
          "HierarchicalCopy requires num_packs >= 0, but {} is specified"
          .format(num_packs))
    super(HierarchicalCopyAllReduce, self).__init__(
        all_reduce_alg="hierarchical_copy",
        num_packs=num_packs)


# TODO(crccw): remove after migrating all callers.
CollectiveCommunication = collective_util.CommunicationImplementation
CommunicationImplementation = collective_util.CommunicationImplementation


# TODO(yuefengz): support in-graph collective all-reduce.
class CollectiveAllReduce(CrossDeviceOps):
  # [CLASS DOC]
  # Cross-device implementation built on collective ops, used for single- and multi-worker
  # training. Holds the group size, communication options, a collective key generator, a lock
  # serializing collective launches (to avoid deadlocks from interleaved launches across threads),
  # the canonicalized device tuple, one collective launcher per device, and a flag saying whether
  # NCCL launches cannot be ordered. Supports reduce, batch reduce, gather and the replica-context
  # all-reduce, and supports deep copying.
  """All-reduce cross device ops using collective ops.

  In the between-graph replicated training, it will still do all-reduces across
  all workers and then put results on the right destinations.
  """

  def __init__(self,
               devices,
               group_size,
               options,
               collective_keys=None,
               canonicalize_devices=True):
    # [FUNCTION DOC]
    # Constructor. devices: list of device strings to run collectives on. group_size: total
    # number of devices across all workers; must be divisible by the number of local devices (value
    # error otherwise). options: communication options. collective_keys: optional key generator
    # (a new one is created if absent). canonicalize_devices: default true; whether to fully
    # canonicalize device names or strip job and task. Side effects: creates the launch lock,
    # obtains a group key for the devices, creates one collective launcher per device, and marks
    # NCCL as limited if any launcher cannot order NCCL launches. Finally calls the base constructor
    # and then sets the canonicalization flag.
    """Initializes the object.

    Args:
      devices: a list of device strings to run collectives on.
      group_size: the global group size. For between-graph replicated training
        it's the total number of devices across all workers.
      options: a `tf.distribute.experimental.CommunicationOptions`.
      collective_keys: an optional CollectiveKey object.
      canonicalize_devices: Whether to canonicalize devices for workers or not.
    """
    if group_size % len(devices) > 0:
      raise ValueError("group_size must be divisible by the number of devices.")

    self._group_size = group_size
    self._options = options
    self._collective_keys = (collective_keys or
                             cross_device_utils.CollectiveKeys())
    # This lock guards all collective launches, i.e. calls to
    # cross_device_utils.build_collectve_*.
    #
    # In a multi threaded eager program we need to ensure different groups of
    # collectives don't interleave each other, otherwise there could be
    # deadlocks. E.g. if two user threads both are launching collectives:
    #   user-thread-0  device0                 device1
    #   user-thread-1          device0 device1
    # In eager mode, we use one thread per device to launch collective ops, so
    # the above launch sequences end up with the following queues:
    #   device-0  collective-0  collective-1
    #   device-1  collective-1  collective-0
    # This deadlocks since neither collective is able to finish.
    self._lock = threading.Lock()

    if canonicalize_devices:
      self._devices = tuple(device_util.canonicalize(d) for d in devices)
    else:
      self._devices = tuple(
          device_util.canonicalize_without_job_and_task(d) for d in devices)
    group_key = self._collective_keys.get_group_key(self._devices)
    self._launchers = []
    # Whether to only use NCCL for batched all-reduce when NCCL is requested.
    # This is because of the lack of mechanism to order NCCL operations
    # deterministically.
    self._limited_nccl = False
    for device in self._devices:
      launcher = cross_device_utils.CollectiveReplicaLauncher(
          group_key, group_size, self._collective_keys, device, options)
      self._launchers.append(launcher)
      if not launcher.can_order_nccl():
        self._limited_nccl = True

    super(CollectiveAllReduce, self).__init__()
    self._canonicalize_devices = canonicalize_devices

  @property
  def _num_between_graph_workers(self):
    # [FUNCTION DOC]
    # HELPER: property giving the number of workers as the group size divided by the number of
    # local devices (assumes equal device counts per worker; the result is a floating-point
    # number, not an integer).
    # Currently we only support equal number of devices on each worker.
    return self._group_size / len(self._devices)

  def _all_reduce(self, reduce_op, value, replica_id, options):
    # [FUNCTION DOC]
    # All-reduces a tensor or nested structure across all replicas (replica context). Inputs:
    # reduce_op (sum or mean), value, replica_id (local index selecting the launcher/device),
    # options (communication options). If NCCL launches cannot be ordered and NCCL is requested
    # for a single value, switches to the ring implementation. Splits flattened values into dense
    # and sparse sets. Dense values are reversed (to better overlap with computation), grouped
    # into packs by the configured bytes per pack, all-reduced in batch (logging in graph mode on
    # replica zero), divided by the group size for mean, and reversed back. Sparse values are
    # all-reduced one by one, with the values divided by the group size for mean. Results are
    # stitched back and repacked into the input structure, which is returned.
    """Implements CrossDeviceOps.all_reduce."""
    # TODO(b/122840926): reuse this method in _batch_all_reduce.
    flat_values = nest.flatten(value)

    # If NCCL launches can't be ordered (self._limited_nccl == True), we only
    # use NCCL when batch_size > 1, hoping that there's only one batched
    # all-reduce, which is the gradient aggregation in optimizer. For TF 2.x,
    # NCCL launches are always ordered.
    if (self._limited_nccl and options.implementation
        == collective_util.CommunicationImplementation.NCCL and
        len(flat_values) == 1):
      options = options.merge(
          collective_util.Options(
              implementation=collective_util.CommunicationImplementation.RING))

    launcher = self._launchers[replica_id]
    dense_values, dense_indices, sparse_values, sparse_indices = (
        cross_device_utils.split_by_sparsity(flat_values))
    dense_results = []
    sparse_results = []

    if dense_values:
      # Reverse the lists so that there's better chance that values follows
      # the order in which they are calculated (e.g. when they're gradients), so
      # as to overlap calculation with communication. However, this may not be
      # optimal for cases like gradients of complicated non-sequential models.
      #
      # Note that we reverse the list before packing so that the first pack
      # won't be too small, since it's more likely for first few packs to have
      # long queuing time due to concurrent intense computation.
      #
      # TODO(b/147393503): explore solutions for optimal ordering.
      dense_values.reverse()
      packs = cross_device_utils.group_by_size(dense_values,
                                               options.bytes_per_pack)

      if not context.executing_eagerly() and replica_id == 0:
        logging.info(
            "Collective all_reduce tensors: %d all_reduces, num_devices = %d, "
            "group_size = %d, implementation = %s, num_packs = %d",
            len(dense_values), len(self._launchers), self._group_size,
            options.implementation, len(packs))

      dense_results = launcher.batch_all_reduce(packs, options)
      if reduce_op == reduce_util.ReduceOp.MEAN:
        for i, v in enumerate(dense_results):
          with ops.device(self._devices[replica_id]):
            dense_results[i] = v / self._group_size
      dense_results.reverse()

    if sparse_values:
      if not context.executing_eagerly() and replica_id == 0:
        logging.info(
            "Collective all_reduce IndexedSlices: %d all_reduces, num_devices ="
            "%d, group_size = %d, implementation = %s", len(sparse_values),
            len(self._launchers), self._group_size, options.implementation)

      for indexed_slice in sparse_values:
        sparse_results.append(
            launcher.all_reduce_indexed_slices(indexed_slice, options))

      if reduce_op == reduce_util.ReduceOp.MEAN:
        for i, v in enumerate(sparse_results):
          with ops.device(self._devices[replica_id]):
            sparse_results[i] = indexed_slices.IndexedSlices(
                values=sparse_results[i].values / self._group_size,
                indices=sparse_results[i].indices,
                dense_shape=sparse_results[i].dense_shape)

    flat_results = cross_device_utils.stitch_values(
        ((dense_results, dense_indices), (sparse_results, sparse_indices)))
    return nest.pack_sequence_as(value, flat_results)

  def _all_reduce_per_replica_values(self, reduce_op, per_replica_values,
                                     options):
    # [FUNCTION DOC]
    # HELPER: all-reduces a list of per-replica values across local devices. Inputs: reduce_op,
    # per_replica_values (list of distributed values with one component per local device), options.
    # Gathers the components per device, then under the launch lock runs the replica all-reduce for
    # each device: in eager mode on a thread pool with one thread per device (each in eager mode),
    # otherwise sequentially. Returns a list of mirrored values, one per input value.
    """All reduce a list of per_replica_value."""
    values_by_device = [[] for _ in self._devices]
    num_devices = len(self._devices)
    for per_replica in per_replica_values:
      for i in range(num_devices):
        values_by_device[i].append(per_replica.values[i])

    if context.executing_eagerly():

      def thread_fn(device_id):
        # [FUNCTION DOC]
        # HELPER: worker for the thread pool. Runs the replica all-reduce for one device index
        # inside eager mode and returns its results.
        with context.eager_mode():
          return self._all_reduce(reduce_op, values_by_device[device_id],
                                  device_id, options)

      with self._lock:
        pool = multiprocessing.pool.ThreadPool(len(self._devices))
        outputs_by_device = pool.map(thread_fn, list(range(num_devices)))
        pool.close()
    else:
      outputs_by_device = []
      with self._lock:
        for i in range(num_devices):
          outputs_by_device.append(
              self._all_reduce(reduce_op, values_by_device[i], i, options))

    result = []
    for values in zip(*outputs_by_device):
      result.append(
          distribute_utils.regroup(values, wrap_class=value_lib.Mirrored))
    return result

  def reduce_implementation(self, reduce_op, per_replica_value, destinations,
                            options):
    # [FUNCTION DOC]
    # Reduce hook. Marks the current context as unsaveable (side effect), all-reduces the single
    # value across devices, and if the destinations' devices match the value's devices returns
    # that result. Otherwise wraps the result as a mirrored value if needed and builds, for each
    # destination device, an identity copy of the matching component (or of the primary component
    # when none matches) under control dependencies on all components (to avoid deadlock), and
    # returns them grouped as a mirrored value.
    values_util.mark_as_unsaveable()
    all_reduced = self._all_reduce_per_replica_values(reduce_op,
                                                      [per_replica_value],
                                                      options)[0]
    devices = get_devices_from(destinations, self._canonicalize_devices)

    if _devices_match(per_replica_value, destinations,
                      self._canonicalize_devices):
      return all_reduced

    # Convert `all_reduced` to a `Mirrored` object, as a simple and uniform
    # utility to access component for a particular device.
    if not isinstance(all_reduced, value_lib.Mirrored):
      all_reduced = value_lib.Mirrored([all_reduced])

    # If we got this far, the destination devices do not match the all-reduce
    # devices, so we must map from one to the other.
    index = []
    # We must add these control dependencies, otherwise we can get deadlock.
    with ops.control_dependencies(all_reduced.values):
      for d in devices:
        with ops.device(d):
          for v in all_reduced.values:
            if v.device == d:
              index.append(array_ops.identity(v))
              break
          else:
            # TODO(josh11b): Once we add support for model parallelism, get the
            # copy from the corresponding replica instead of the primary.
            index.append(array_ops.identity(all_reduced._primary))  # pylint: disable=protected-access
    return distribute_utils.regroup(index, wrap_class=value_lib.Mirrored)

  def batch_reduce_implementation(self, reduce_op, value_destination_pairs,
                                  options):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inner check that logs the warning repeats the already-known negation of the
    # outer condition, so it is always true there (redundant, harmless).
    # Batch reduce hook. Marks the context as unsaveable (side effect). If all devices match, runs
    # one batched all-reduce over all values and returns the list. Otherwise logs a warning (first
    # ten times) and reduces each pair separately via the reduce hook, returning the list.
    values_util.mark_as_unsaveable()
    all_devices_match = _all_devices_match(value_destination_pairs,
                                           self._canonicalize_devices)
    if all_devices_match:
      return self._all_reduce_per_replica_values(
          reduce_op, [v[0] for v in value_destination_pairs], options)
    else:
      if not all_devices_match:
        logging.log_first_n(
            logging.WARN, "Efficient batch_reduce is not supported if "
            "destinations are different.", 10)

      return [
          self.reduce_implementation(reduce_op, value, dest, options)
          for value, dest in value_destination_pairs
      ]

  def _gather_implementation(self, per_replica_value, destinations, axis,
                             options):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the loop mapping gathered components to destination devices, the else branch
    # belongs to the device comparison inside the loop (not to the loop itself, unlike the reduce
    # hook), so a copy of the primary component is appended for every non-matching component and the
    # loop does not fall back only when nothing matched. This can produce extra or wrong entries
    # when destinations differ from the gather devices.
    # Gather hook. Gathers the single value along axis across devices, marks the context as
    # unsaveable, and returns it directly if the destination devices match; otherwise maps the
    # result onto the destination devices (see the known issue) under control dependencies and
    # returns them grouped as a mirrored value.
    all_gathered = self._batch_all_gather([per_replica_value], axis, options)[0]
    values_util.mark_as_unsaveable()
    devices = get_devices_from(destinations, self._canonicalize_devices)

    if _devices_match(per_replica_value, destinations,
                      self._canonicalize_devices):
      return all_gathered

    # Convert `all_gathered` to a `Mirrored` object, as a simple and uniform
    # utility to access component for a particular device.
    if not isinstance(all_gathered, value_lib.Mirrored):
      all_gathered = value_lib.Mirrored([all_gathered])

    # If we got this far, the destination devices do not match the all-gather
    # devices, so we must map from one to the other.
    index = []
    # We must add these control dependencies, otherwise we can get deadlock.
    with ops.control_dependencies(all_gathered.values):
      for d in devices:
        with ops.device(d):
          for v in all_gathered.values:
            if v.device == d:
              index.append(array_ops.identity(v))
              break
            else:
              index.append(array_ops.identity(all_gathered._primary))  # pylint: disable=protected-access
    return distribute_utils.regroup(index, wrap_class=value_lib.Mirrored)

  def _batch_all_gather(self, per_replica_values, axis, options):
    # [FUNCTION DOC]
    # HELPER: all-gathers a list of per-replica values along axis. If NCCL launches cannot be
    # ordered and NCCL is requested for a batch of one, switches to the ring implementation. Logs
    # the plan (first ten times). Under the launch lock and a name scope, launches an all-gather
    # on every device launcher for each value; in eager mode this is wrapped in a compiled function.
    # Returns a list of mirrored values, one per input value.
    """all gather multiple per-replica-values."""
    batch_size = len(per_replica_values)
    # For now, we use NCCL only when batch_size > 1.
    # TODO(b/132575814): switch to NCCL for all collectives when implementation
    # is NCCL.
    if (self._limited_nccl and options.implementation
        == collective_util.CommunicationImplementation.NCCL and
        batch_size == 1):
      options = options.merge(
          collective_util.Options(
              implementation=collective_util.CommunicationImplementation.RING))

    logging.log_first_n(
        logging.INFO, "Collective batch_all_gather: %d all-gathers, "
        "num_devices = %d, group_size = %d, implementation = %s, " %
        (batch_size, len(
            self._devices), self._group_size, options.implementation), 10)

    def compute_gathered_values():
      # [FUNCTION DOC]
      # HELPER: launches the all-gathers. Holds the launch lock and a name scope while, for each
      # value, collecting one gathered output per device from the launchers. Returns a list of
      # per-value lists of outputs.
      gathered_values = []
      with self._lock, ops.name_scope("allgather"):
        for per_replica in per_replica_values:
          outputs = []
          for i in range(len(self._devices)):
            outputs.append(self._launchers[i].all_gather(
                per_replica.values[i], axis, options))
          gathered_values.append(outputs)
      return gathered_values

    if context.executing_eagerly():
      gathered_values = def_function.function(compute_gathered_values)()
    else:
      gathered_values = compute_gathered_values()

    mirrored = []
    for value in gathered_values:
      mirrored.append(
          distribute_utils.regroup(value, wrap_class=value_lib.Mirrored))
    return mirrored

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Supports deep copying (needed because the coordinator deep-copies strategies). Deep-copies
    # the collective keys using the shared memo and builds a new instance with the same devices,
    # group size, options and canonicalization setting. Returns the new instance.
    # distribute_coordinator deep-copies the strategy object, so
    # CollectiveAllReduce needs to support deep copy as well.
    collective_keys = copy.deepcopy(self._collective_keys, memo)
    return CollectiveAllReduce(self._devices, self._group_size, self._options,
                               collective_keys, self._canonicalize_devices)


def select_cross_device_ops(devices, session_config=None):
  # [FUNCTION DOC]
  # Chooses a default cross-device implementation. Inputs: devices (list of device names requested
  # by the strategy), session_config (optional session configuration, used only in graph mode to
  # list local devices). In eager mode, if the numbers of logical and physical GPUs differ (virtual
  # GPUs) it warns and returns a reduce-to-one-device instance. Lists machine devices, warns about
  # requested devices not visible, and returns reduce-to-one-device (with a warning) if any
  # requested device is not a GPU. If the NCCL all-reduce kernel is registered, returns an NCCL
  # all-reduce with one pack; otherwise warns and returns reduce-to-one-device.
  """Find the best `CrossDeviceOps` locally given a `tf.compat.v1.ConfigProto`.

  Args:
    devices: a list of devices passed to `tf.distribute.Strategy`.
    session_config: a `tf.compat.v1.ConfigProto` or `None`. If `None`, it will
      make decision based on all logical devices.

  Returns:
    A subclass of `CrossDeviceOps`.
  """
  requested_devices = set(device_util.canonicalize(d) for d in devices)
  if ops.executing_eagerly_outside_functions():
    logical_gpus = context.context().list_logical_devices(device_type="GPU")
    physical_gpus = context.context().list_physical_devices(device_type="GPU")
    if len(logical_gpus) != len(physical_gpus):
      logging.warning("NCCL is not supported when using virtual GPUs, falling"
                      "back to reduction to one device")
      return ReductionToOneDevice()

    machine_devices = context.context().list_logical_devices()
  else:
    machine_devices = device_lib.list_local_devices(
        session_config=session_config)
  using_devices = set()
  for d in machine_devices:
    if device_util.canonicalize(d.name) in requested_devices:
      using_devices.add(d.name)

  if len(using_devices) != len(requested_devices):
    logging.warning(
        "Some requested devices in `tf.distribute.Strategy` are not visible "
        "to TensorFlow: %s", ",".join(list(requested_devices - using_devices)))

  if any("gpu" not in d.lower() for d in requested_devices):
    logging.warning("There are non-GPU devices in `tf.distribute.Strategy`, "
                    "not using nccl allreduce.")
    return ReductionToOneDevice()

  if kernels.get_registered_kernels_for_op("NcclAllReduce"):
    return NcclAllReduce(num_packs=1)
  else:
    logging.warning("Nccl kernel is not found, not using nccl allreduce.")
    return ReductionToOneDevice()
