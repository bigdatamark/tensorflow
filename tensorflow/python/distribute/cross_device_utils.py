# [MODULE DOC]
# HELPER: internal helper module for the cross-device reduction code in cross_device_ops.
# Utilities for combining, moving and all-reducing gradients and tensors across devices.
# Exposes gradient aggregation helpers (NCCL based, hierarchical copy based, and plain copy
# based), a thread-safe manager of collective group and instance keys, a per-replica
# collective launcher (all-reduce, all-gather, batch all-reduce and sparse all-reduce), and
# small helpers for dealing with dense tensors versus sparse indexed slices (aggregation,
# division, copying to a device, splitting by sparsity, stitching back, grouping by size).
# Defines one module constant, the starting number for per-device instance keys.
# No registrations or flags are created at import time.
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
"""Utilities for cross_device_ops."""

import copy
import threading
from typing import Callable, List, Optional, Union

from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core

INSTANCE_KEY_START_NUMBER = 100


def aggregate_gradients_using_nccl(replica_grads):
  # [FUNCTION DOC]
  # HELPER: gradient aggregation helper for multi-GPU cross-device reductions.
  # Sums gradients across replicas with an NCCL all-sum. The input replica_grads is a list
  # (one entry per replica) of lists of (gradient, variable) pairs. For each position the
  # gradients of all replicas are summed with the NCCL op, and each replica's variable is
  # paired with its summed gradient. Returns a list (one entry per replica) of lists of
  # (summed gradient, variable) pairs, regrouped from the position-major layout used inside.
  # Raises nothing itself; adds NCCL ops to the current graph/device context.
  """Aggregate gradients using nccl allreduce."""
  agg_all_g_and_v = []
  for single_g_and_v in zip(*replica_grads):
    single_grads = [g for g, _ in single_g_and_v]
    agg_grads = nccl_ops.all_sum(single_grads)
    agg_all_g_and_v.append(
        [(g, v) for g, (_, v) in zip(agg_grads, single_g_and_v)])

  agg_all_g_and_v = list(zip(*agg_all_g_and_v))

  return agg_all_g_and_v


def aggregate_gradients_using_hierarchical_copy(avail_devices, replica_grads):
  # [FUNCTION DOC]
  # HELPER: gradient aggregation helper for multi-GPU cross-device reductions.
  # Sums gradients across replicas using a two-level copy scheme tuned for the DGX-1 eight-GPU
  # peer-to-peer topology. Inputs: avail_devices, the list of GPU device names (assumed to be
  # split into two equal halves); replica_grads, a list (per replica) of lists of
  # (gradient, variable) pairs. For each gradient position, each half is summed on a main
  # device, the two partial sums are added on the first group's main device, and the total is
  # broadcast back to each device via identity copies. The main device rotates with the
  # position index to spread load. Returns a list (per replica) of lists of
  # (aggregated gradient, variable) pairs. Only meaningful for the DGX-1 topology; with an
  # odd or mismatched device count the groups would be unequal. Creates ops under device scopes.
  """Aggregate gradients using hierarchical copies.

  Args:
    avail_devices: available GPU devices.
    replica_grads: List of lists of (gradient, variable) tuples. The outer list
      is over replicas. The inner list is over individual gradients.

  Returns:
    The list of (aggregated_gradient, variable), where the gradient has been
      summed across all replicas and the variable is chosen from the first
      replica.
  """
  # This only works for DGX-1 type of machine topology
  # Device peer to peer matrix
  # DMA: 0 1 2 3 4 5 6 7
  # 0:   Y Y Y Y Y N N N
  # 1:   Y Y Y Y N Y N N
  # 2:   Y Y Y Y N N Y N
  # 3:   Y Y Y Y N N N Y
  # 4:   Y N N N Y Y Y Y
  # 5:   N Y N N Y Y Y Y
  # 6:   N N Y N Y Y Y Y
  # 7:   N N N Y Y Y Y Y
  agg_grads = []
  num_devices = len(avail_devices)
  # In the special case of DGX-1 machine topology, the two groups have equal
  # size.
  group_size = num_devices // 2
  for i, single_grads in enumerate(zip(*replica_grads)):
    group_0_main_device = i % num_devices
    group_1_main_device = (group_0_main_device + group_size) % num_devices
    if group_0_main_device < group_size:
      group_0_begin = 0
      group_1_begin = group_size
    else:
      group_0_begin = group_size
      group_1_begin = 0

    # Aggregate the first group.
    group_0_device_grads = single_grads[group_0_begin:
                                        group_0_begin + group_size]
    with ops.device(avail_devices[group_0_main_device]):
      group_0_agg_grads, _ = aggregate_single_gradient_using_copy(
          group_0_device_grads, False, False)

    # Aggregate the second group.
    group_1_device_grads = single_grads[group_1_begin:
                                        group_1_begin + group_size]
    with ops.device(avail_devices[group_1_main_device]):
      group_1_agg_grads, _ = aggregate_single_gradient_using_copy(
          group_1_device_grads, False, False)

    # Aggregate between the groups.
    with ops.device(avail_devices[group_0_main_device]):
      (agg_total_grads, _), _ = aggregate_single_gradient_using_copy(
          [group_0_agg_grads, group_1_agg_grads], False, False)

    # Broadcast the result back into the root of each group.
    with ops.device(avail_devices[group_0_main_device]):
      group_0_agg_grads_bcast = array_ops.identity(agg_total_grads)
    with ops.device(avail_devices[group_1_main_device]):
      group_1_agg_grads_bcast = array_ops.identity(agg_total_grads)

    agg_grads_bcast = []
    for j in range(len(single_grads)):
      with ops.device(avail_devices[j]):
        # Broadcast the result back to each member in the group from the root.
        if (group_0_main_device < group_size) == (j < group_size):
          src_device_grad = group_0_agg_grads_bcast
        else:
          src_device_grad = group_1_agg_grads_bcast
        agg_grads_bcast.append(array_ops.identity(src_device_grad))

    agg_grads.append(
        [(g, v) for g, (_, v) in zip(agg_grads_bcast, single_grads)])

  agg_grads = list(zip(*agg_grads))

  return agg_grads


def aggregate_single_gradient_using_copy(grad_and_vars, use_mean,
                                         check_inf_nan):
  # [FUNCTION DOC]
  # HELPER: gradient aggregation helper used by the hierarchical copy aggregation and by
  # cross-device reduction code.
  # KNOWN ISSUE: the docstring says a list of (gradient, variable) pairs is returned, but the
  # code returns a single (gradient, variable) tuple, not a list.
  # Sums (or averages) the per-replica gradients of one shared variable. grad_and_vars is a
  # list or tuple of (gradient, variable) pairs, one per replica; use_mean chooses averaging
  # (only applied when more than one replica) versus summing; check_inf_nan chooses whether a
  # finiteness check is built. Returns a pair: first the tuple of aggregated gradient and the
  # variable of the first replica, second a boolean tensor telling whether any gradient had a
  # non-finite value, or nothing/null value when check_inf_nan is false.
  """Calculate the average gradient for a shared variable across all replicas.

  Note that this function provides a synchronization point across all replicas.

  Args:
    grad_and_vars: A list or tuple of (gradient, variable) tuples. Each
      (gradient, variable) pair within the outer list represents the gradient
      of the variable calculated for a single replica, and the number of pairs
      equals the number of replicas.
    use_mean: if True, mean is taken, else sum of gradients is taken.
    check_inf_nan: check grads for nans and infs.

  Returns:
    The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
      gradient has been averaged across all replicas. The variable is chosen
      from the first replica. The has_nan_or_inf indicates the grads has nan or
      inf.
  """
  grads = [g for g, _ in grad_and_vars]
  grad = math_ops.add_n(grads)

  if use_mean and len(grads) > 1:
    grad = array_ops.multiply(grad, 1.0 / len(grads))

  v = grad_and_vars[0][1]
  if check_inf_nan:
    has_nan_or_inf = array_ops.logical_not(
        array_ops.reduce_all(array_ops.is_finite(grads)))
    return (grad, v), has_nan_or_inf
  else:
    return (grad, v), None


# TODO(yuefengz): use random key starts to avoid reusing keys?
class CollectiveKeys(object):
  # [CLASS DOC]
  # HELPER: key bookkeeping helper for collective ops used by the collective launcher.
  # Hands out integer group keys (one per distinct set of devices) and per-device instance
  # keys that identify matching collective ops within a group. State: next group key, a
  # mapping from group key to a mapping of device to next instance key, a mapping from joined
  # device string to known group key, and a lock; all public operations take the lock, so the
  # class is thread safe. Supports deep copy (copies the counters and instance table only).
  """Class that manages collective keys.

  We need to manage three different keys for collective:

  *Group key*: an integer key to identify the set of cooperative devices.
  Collective ops work under the same set of devices must using the same group
  key.

  *Instance key*: an integer key to identify the set of same counterpart of
  tensors on different devices in a device group that need to be all-reduced.

  This class is thread safe.
  """

  def __init__(self, group_key_start=1):
    # [FUNCTION DOC]
    # Constructor. group_key_start (integer, default 1) is the first group key handed out.
    # Initializes the group key counter, an empty instance key table, an empty mapping of
    # known device groups and a lock. Returns nothing.
    """Initializes the object.

    Args:
      group_key_start: the starting integer of group key.
    """
    self._group_key = group_key_start
    self._instance_key_table = {}
    self._lock = threading.Lock()
    self._known_groups = {}

  def get_group_key(self, devices):
    # [FUNCTION DOC]
    # Returns the group key for a list of canonical local device strings. The devices are
    # joined (in the given order, so order matters) into one identifier; the first time an
    # identifier is seen a new key is allocated and remembered, afterwards the same key is
    # returned. Takes the lock, so thread safe. Mutates internal state on first sight of a
    # device list. Returns an integer.
    """Returns a group key for the list of local devices.

    The same group key is returned if the list of local devices is the same.

    Args:
      devices: a list of local canonical device strings in a collective group.

    Returns:
      a group key.
    """
    with self._lock:
      devices_key = ','.join(devices)
      if devices_key not in self._known_groups:
        self._known_groups[devices_key] = self._get_new_group_key(devices)
      return self._known_groups[devices_key]

  def _get_new_group_key(self, devices):
    # [FUNCTION DOC]
    # HELPER: allocation helper for get_group_key.
    # Allocates the next group key, increments the counter and creates an instance key table
    # for it in which every given device starts at the instance key start number. Takes a list
    # of canonical device strings. Returns the new integer key. Not thread safe by itself; the
    # caller must hold the lock.
    """Returns a new group key.

    The caller should store and reuse the same group key for the same set of
    devices. Calling this method always returns a new group key.

    This method is not thread-safe.

    Args:
      devices: a list of canonical device strings in a collective group.

    Returns:
      a new group key.
    """
    new_key = self._group_key
    self._group_key += 1
    self._instance_key_table[new_key] = {}
    for device in devices:
      self._instance_key_table[new_key][device] = INSTANCE_KEY_START_NUMBER
    return new_key

  def get_instance_key(self, group_key, device):
    # [FUNCTION DOC]
    # Returns the next unused instance key for a device within a group and advances that
    # device's counter. group_key must be a key previously produced by get_group_key and device
    # a canonical device string that belongs to that group. Takes the lock. Raises a value
    # error if the group key is unknown or the device is not part of the group.
    """Returns a new instance key for use in defining a collective op.

    You should call this once per each collective op of a collective instance.

    Args:
      group_key: the group key returned by get_group_key(). You should not
        assign the group key yourself.
      device: a canonical device string. It should be the device this collective
        op is on.

    Returns:
      a new instance key.

    Raises:
      ValueError: when the group key is invalid or the device is not in the
      group.
    """
    with self._lock:
      group = self._instance_key_table.get(group_key, None)
      if group is None:
        raise ValueError(f'Group {group_key} is not found.')
      if device not in group:
        raise ValueError(f'Device {device} is not present in group {group_key}')
      v = group[device]
      group[device] += 1
      return v

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Custom deep copy hook (needed because the object holds a lock, and the distribute
    # coordinator deep-copies strategy objects). memo is the standard deep-copy bookkeeping
    # mapping. Builds a new instance with default start, then copies the current group key
    # counter and a deep copy of the instance key table. The mapping of known device groups is
    # NOT copied, so the copy will allocate fresh group keys for device lists it is asked
    # about, which may differ from the original's keys. Returns the new instance.
    # distribute_coordinator deep-copies the strategy object, so
    # CollectiveKeys needs to support deep copy as well.
    copied = CollectiveKeys()
    copied._group_key = self._group_key
    copied._instance_key_table = copy.deepcopy(self._instance_key_table, memo)
    return copied


class CollectiveReplicaLauncher(object):
  # [CLASS DOC]
  # Launches collective operations (all-reduce, all-gather, batch all-reduce, sparse
  # all-reduce) on behalf of a single replica. Holds the group key, group size, shared key
  # manager, the replica's device, merged-in default communication options, and optionally an
  # ordering token (a scalar resource variable) used to serialize collectives. Two class-level
  # switches, prefer unique instance key and prefer ordering token, default to true and can be
  # toggled globally (for instance by tests) to change how keys and ordering are handled.
  """Launch collectives on one replica."""

  _prefer_unique_instance_key = True
  _prefer_ordering_token = True

  def __init__(self, group_key: int, group_size: int,
               collective_keys: CollectiveKeys, device: str,
               options: collective_util.Options):
    # [FUNCTION DOC]
    # Constructor. group_key: integer key of the device group; group_size: number of
    # participants; collective_keys: the shared key manager; device: device string this replica
    # runs on; options: default communication options. Stores them. If ordering tokens are in
    # use (eager-outside-functions mode and the class switch on), creates a scalar resource
    # variable initialized to zero on the device, inside the init scope, as the ordering token;
    # otherwise the token is null. Returns nothing.
    self._group_key = group_key
    self._group_size = group_size
    self._collective_keys = collective_keys
    self._device = device
    self._options = options
    if self._use_ordering_token():
      with ops.init_scope(), ops.device(device):
        self._ordering_token = resource_variable_ops.ResourceVariable(0.)
    else:
      self._ordering_token = None

  def _control_input(self, control_input: Union[core.TensorLike,
                                                ops.Operation]):
    # [FUNCTION DOC]
    # HELPER: ordering helper for all_reduce.
    # Returns a context manager that makes ops built inside it depend on the given control
    # input (a tensor or operation) when one is provided and ordering tokens are not in use.
    # Otherwise returns a no-op context manager (ordering is then done by the token).
    if control_input is not None and not self._use_ordering_token():
      return ops.control_dependencies([control_input])
    return ops.NullContextmanager()

  def _use_unique_instance_key(self):
    # [FUNCTION DOC]
    # HELPER: policy check for instance key assignment.
    # Returns false in TF1 graph mode (not executing eagerly outside functions); otherwise
    # returns the class-wide preference flag for unique instance keys.
    if not ops.executing_eagerly_outside_functions():
      return False
    return CollectiveReplicaLauncher._prefer_unique_instance_key

  def _use_ordering_token(self):
    # [FUNCTION DOC]
    # HELPER: policy check for collective ordering.
    # Returns false in TF1 graph mode (where automatic control dependencies are not used, so
    # ordering tokens cannot work); otherwise returns the class-wide preference flag for
    # ordering tokens.
    # We rely on auto control dep to insert control edges between NCCL calls,
    # but for tf1 graph mode auto control dep is not used.
    if not ops.executing_eagerly_outside_functions():
      return False
    return CollectiveReplicaLauncher._prefer_ordering_token

  def _next_instance_key(self):
    # [FUNCTION DOC]
    # HELPER: instance key supplier for the collective ops of this launcher.
    # Returns the instance key for the next collective op. With unique instance keys, when
    # building a function graph (skipping out of control-flow graphs to the enclosing function
    # graph) the key is a call-time placeholder captured from this same method, so it is
    # assigned each time the function runs, which avoids retracing mismatches across workers;
    # otherwise (eager) a key is drawn from the key manager and returned as an int32 tensor
    # on the CPU. Without unique keys, a plain integer key is drawn from the key manager.
    # Side effect: advances the key manager's counter when a key is drawn.
    """Returns the next instance key."""
    if self._use_unique_instance_key():
      # Assigning instance keys at function building time have issues since
      # different workers may retrace the function at different times. With
      # collective V2 we can use capture_call_time_value to use a placeholder as
      # the instance key and feed it at function call time. In this way we also
      # don't reuse instance keys, which allows for per-instance cancellation.
      graph = ops.get_default_graph()
      # Control flow ops don't work with capture_call_time_value, so we put the
      # capture in the function graph of that control flow op.
      while getattr(graph, 'is_control_flow_graph', False):
        graph = graph.outer_graph
      if not context.executing_eagerly() and graph.building_function:
        with graph.as_default():
          # Capture self._next_instance_key so that when building a function
          # that calls another tf.function, the instance key assignment is
          # further delayed until we actually call the function in eager. Note
          # that capture_call_time_value doesn't automatically propagate the
          # deferred capture to the outer function.
          return graph.capture_call_time_value(
              self._next_instance_key, tensor_spec.TensorSpec([], dtypes.int32))
      else:
        instance_key = self._collective_keys.get_instance_key(
            self._group_key, self._device)
        with ops.device('CPU:0'):
          return ops.convert_to_tensor(instance_key, dtype=dtypes.int32)
    else:
      return self._collective_keys.get_instance_key(self._group_key,
                                                    self._device)

  def _get_ordering_token(self):
    # [FUNCTION DOC]
    # HELPER: accessor for the ordering token used by collective ops.
    # Returns the handle of the ordering token variable when ordering tokens are in use,
    # otherwise returns nothing/null value (implicitly).
    if self._use_ordering_token():
      return self._ordering_token.handle  # pytype: disable=attribute-error

  def can_order_nccl(self):
    # [FUNCTION DOC]
    # Reports whether NCCL collectives from this launcher can be ordered, which is the case
    # exactly when ordering tokens are in use. Returns a boolean.
    """Whether this launcher can order NCCL operations."""
    return self._use_ordering_token()

  def all_reduce(
      self,
      input_tensor: core.TensorLike,
      control_input: Optional[Union[core.TensorLike, ops.Operation]] = None,
      options: Optional[collective_util.Options] = None) -> core.Tensor:
    # [FUNCTION DOC]
    # Sums a dense tensor across all replicas of the group with a version-2 all-reduce op.
    # input_tensor: dense tensor, same shape on all replicas. control_input: optional tensor or
    # operation the all-reduce must wait for (only honored when ordering tokens are not in
    # use). options: optional communication options merged over the launcher defaults (timeout
    # and implementation hint are taken from the merged result). Gets a fresh instance key,
    # builds the op on the launcher's device and passes the ordering token. Returns the reduced
    # tensor.
    """All-reduce a dense tensor.

    Args:
      input_tensor: a dense tensor. It must have the same shape on all replicas.
      control_input: if not None, add control edges between control_input and
        the all-reduce.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The reduced tensor.
    """
    instance_key = self._next_instance_key()
    options = self._options.merge(options)
    ordering_token = self._get_ordering_token()
    with ops.device(self._device), \
         self._control_input(control_input):
      return collective_ops.all_reduce_v2(
          input_tensor,
          self._group_size,
          self._group_key,
          instance_key,
          communication_hint=options.implementation.value,
          timeout=options.timeout_seconds,
          ordering_token=ordering_token)

  def _all_gather(self, input_tensor: core.TensorLike,
                  options: Optional[collective_util.Options]) -> core.Tensor:
    # [FUNCTION DOC]
    # HELPER: low-level gather used by all_gather and the sparse all-reduce.
    # KNOWN ISSUE: the docstring's Returns section says "The reduced tensor", but this
    # concatenates (gathers) tensors from all replicas along the first dimension; nothing is
    # reduced.
    # Concatenates the input tensor from every replica along the first dimension with a
    # version-2 all-gather op on the launcher's device. input_tensor: dense tensor with equal
    # shape on all replicas; options: optional communication options merged over the defaults
    # (required positional argument, may be null). Uses a fresh instance key and the ordering
    # token. Returns the gathered tensor.
    """All-gather a dense tensor.

    Args:
      input_tensor: a dense tensor. It must have the same shape on all replicas.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The reduced tensor.
    """
    instance_key = self._next_instance_key()
    options = self._options.merge(options)
    ordering_token = self._get_ordering_token()
    with ops.device(self._device):
      return collective_ops.all_gather_v2(
          input_tensor,
          self._group_size,
          self._group_key,
          instance_key,
          communication_hint=options.implementation.value,
          timeout=options.timeout_seconds,
          ordering_token=ordering_token)

  def batch_all_reduce(
      self,
      input_tensor_packs: List[List[core.TensorLike]],
      options: Optional[collective_util.Options] = None) -> core.Tensor:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as returning a single tensor, but actually returns a flat list of
    # tensors.
    # All-reduces several packs of dense tensors. input_tensor_packs: list of lists of dense
    # tensors; options: optional communication options merged over the defaults. In eager mode
    # each tensor is reduced individually (no batching). In graph mode each pack is flattened,
    # concatenated, reduced once, then split and reshaped back to the original shapes; when
    # the NCCL implementation is used, each pack's reduction gets a control dependency on the
    # previous output to order the NCCL calls. Returns a flat list of reduced tensors in input
    # order.
    """Batch all-reduce dense tensors.

    This takes a list of batches of tensors. Using multiple batches have the
    benefit that it doesn't need to wait for all inputs to be ready to start the
    all-reduce.

    Args:
      input_tensor_packs: a list of lists of dense tensors.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      A flat list of reduced tensors.
    """
    options = self._options.merge(options)
    outputs = []
    for pack in input_tensor_packs:
      if context.executing_eagerly():
        # We don't batch in eager as it sometimes makes the performance worse
        # due the concat/split ops.
        for input_tensor in pack:
          outputs.append(self.all_reduce(input_tensor, None, options))
      else:
        # TODO(b/169168846): inserts a parallel all_gather to verify packings
        # are the same on each replica.
        with ops.device(self._device):
          flat_tensors = [array_ops.reshape(t, [-1]) for t in pack]
          shapes = [array_ops.shape(t) for t in pack]
          if (options.implementation
              == collective_util.CommunicationImplementation.NCCL and outputs):
            control_input = outputs[-1]
          else:
            control_input = None
          reduced = self.all_reduce(
              array_ops.concat(flat_tensors, axis=0), control_input, options)
          num_elements = [math_ops.reduce_prod(s) for s in shapes]
          flat_outputs = array_ops.split(reduced, num_elements, axis=0)
          for shape, flat_output in zip(shapes, flat_outputs):
            outputs.append(array_ops.reshape(flat_output, shape))

    return outputs

  def all_gather(
      self,
      input_tensor: core.TensorLike,
      axis: core.TensorLike,
      options: Optional[collective_util.Options] = None) -> core.Tensor:
    # [FUNCTION DOC]
    # Gathers dense tensors from all replicas along an arbitrary axis, allowing the sizes along
    # that axis to differ between replicas. input_tensor: dense tensor with the same rank
    # everywhere and equal sizes in other dimensions; axis: scalar int32 tensor giving the
    # gather dimension; options: optional communication options. Steps: move the axis to the
    # front by transposing, gather every replica's first-dimension size, pad each input to the
    # largest size, gather the padded tensors, cut each replica's part back to its true size,
    # concatenate, and transpose back. Raises a runtime error when called in eager mode (must
    # be inside a function graph). Returns the gathered tensor.
    """All-gather a dense tensor.

    This method must be called inside a tf.function.

    Args:
      input_tensor: a dense tensor. It must have the same rank on all replicas,
        and dimensions other than `axis` need to be the same as well.
      axis: 0-D int32 Tensor. Dimension along which to gather. Must be in the
        range [0, rank(value)).
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The gathered Tensor.

    Raises:
      RuntimeError: if called in eager mode.
    """
    if context.executing_eagerly():
      raise RuntimeError('all_gather is not supported in eager mode.')

    with ops.device(self._device), \
         ops.control_dependencies([array_ops.identity(input_tensor)]):
      # 1. Transpose
      # E.g. Given an input_tensor with shape [2,2,5,1] and axis to gather is 3,
      # we use perm_pre=[3 0 1 2] to reshape it to [1,2,2,5], which
      # brings the 3rd dim first; afterwards we use perm_after=[1,2,3,0] to
      # place it back.
      perm_pre = array_ops.concat(
          ([axis], math_ops.range(axis),
           math_ops.range(axis + 1, array_ops.rank(input_tensor))),
          axis=0)
      input_tensor_t = array_ops.transpose(input_tensor, perm=perm_pre)
      # 2. Pad
      gathered_shape = self._all_gather(
          array_ops.expand_dims_v2(array_ops.shape_v2(input_tensor_t), axis=0),
          options)
      first_dims = gathered_shape[:, 0]
      full_axis_dim = math_ops.reduce_max(first_dims)
      padded_input_tensor = _pad_util(input_tensor_t, full_axis_dim)

      # 3. Gather
      gather_padded_out_tensor = self._all_gather(padded_input_tensor, options)
      # 4. Unpad
      split_tensors = []
      for i in range(self._group_size):
        start_pos = i * full_axis_dim
        split_tensors.append(gather_padded_out_tensor[start_pos:start_pos +
                                                      first_dims[i]])
      out_tensor_t = array_ops.concat(split_tensors, 0)

      # 5. Transpose back
      perm_after = array_ops.concat(
          (math_ops.range(1, axis + 1), [0],
           math_ops.range(axis + 1, array_ops.rank(input_tensor_t))),
          axis=0)
      return array_ops.transpose(out_tensor_t, perm=perm_after)

  def all_reduce_indexed_slices(
      self,
      input_slices: indexed_slices.IndexedSlices,
      options: Optional[collective_util.Options] = None
  ) -> indexed_slices.IndexedSlices:
    # [FUNCTION DOC]
    # Reduces sparse indexed slices across replicas by gathering (concatenating) every
    # replica's values and indices rather than summing. input_slices: the indexed slices of
    # this replica; options: optional communication options. Gathers all index lengths first;
    # a runtime conditional then picks either a direct gather when all lengths are equal, or a
    # gather that pads values and indices to the maximum length and trims each part back. With
    # the NCCL implementation the indices gather is ordered after the values gather via a
    # control dependency. The dense shape is carried over unchanged. Returns the combined
    # indexed slices.
    """All-reduce an IndexedSlices.

    This method can be called outside  tf.function.

    Args:
      input_slices: an IndexedSlices.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The reduced IndexedSlices.
    """

    # Current CollectiveAllGather implementations require input IndexedSlices to
    # have consistent length across the board, we handle the reduction of
    # IndexedSlices as follows:
    #   1. Gather the lengths of IndexedSlices from all participants.
    #   2. If they have consistent length, apply all_gather.
    #   3. Otherwise pad IndexedSlices to be the same length across all
    #      participants and apply_gather.
    options = self._options.merge(options)
    with ops.device(self._device):

      def all_gather_indexed_slices(
          all_gather_fn: Callable[
              [core.TensorLike, Optional[collective_util.Options]], core.Tensor]
      ) -> indexed_slices.IndexedSlices:
        # [FUNCTION DOC]
        # HELPER: nested helper of all_reduce_indexed_slices.
        # Gathers the values and the indices of the enclosing input slices with the given
        # gather function (which takes a tensor and options and returns a tensor), ordering the
        # indices gather after the values gather for NCCL, and returns new indexed slices with
        # the original dense shape.
        """Use all_gather_fn to aggregate `IndexedSlices`."""
        all_values = all_gather_fn(input_slices.values, options)
        # Add control dependency to order the all-gather.
        if (options.implementation ==
            collective_util.CommunicationImplementation.NCCL):
          control = [all_values]
        else:
          control = []
        with ops.control_dependencies(control):
          all_indices = all_gather_fn(input_slices.indices, options)
        return indexed_slices.IndexedSlices(
            values=all_values,
            indices=all_indices,
            dense_shape=input_slices.dense_shape)

      length = array_ops.shape(input_slices.indices)
      all_lengths = self._all_gather(length, options)

      def all_gather_with_padding(
          input_tensor: core.TensorLike,
          options: Optional[collective_util.Options]) -> core.Tensor:
        # [FUNCTION DOC]
        # HELPER: nested helper of all_reduce_indexed_slices.
        # Gathers a tensor whose first-dimension size may differ across replicas: pads it to the
        # maximum length found in the enclosing gathered lengths, gathers, then slices out each
        # replica's true-length part and concatenates them. Returns the concatenated tensor.
        """all_gather tensors of different sizes using padding."""
        max_length = math_ops.reduce_max(all_lengths)
        padded_tensor = _pad_util(input_tensor, max_length)
        all_padded_tensors = self._all_gather(padded_tensor, options)
        split_tensors = []
        for i in range(self._group_size):
          start_pos = i * max_length
          split_tensors.append(all_padded_tensors[start_pos:start_pos +
                                                  all_lengths[i]])
        return array_ops.concat(split_tensors, 0)

      return cond.cond(
          math_ops.equal(
              math_ops.reduce_max(all_lengths),
              math_ops.reduce_min(all_lengths)),
          lambda: all_gather_indexed_slices(self._all_gather),
          lambda: all_gather_indexed_slices(all_gather_with_padding))


def aggregate_tensors_or_indexed_slices(values, accumulation_fn=math_ops.add_n):
  # [FUNCTION DOC]
  # Aggregates a list of per-replica values. If any value is indexed slices, the values are
  # combined with the gradient indexed-slices aggregation helper (concatenation of slices);
  # otherwise the accumulation function (default element-wise add of a list) is applied to the
  # list. Returns the aggregated tensor or indexed slices.
  """Aggregate tensors using `accumulation_fn` and IndexedSlices via concat."""
  if any(isinstance(v, indexed_slices.IndexedSlices) for v in values):
    return backprop_util.AggregateIndexedSlicesGradients(values)
  else:
    return accumulation_fn(values)


def divide_by_n_tensors_or_indexed_slices(value, n):
  # [FUNCTION DOC]
  # Divides a tensor or indexed slices by n (a number). For indexed slices, nested slices are
  # first flattened and only the values are divided, indices and dense shape are kept. Returns
  # the divided tensor or new indexed slices.
  if isinstance(value, indexed_slices.IndexedSlices):
    value = backprop_util.FlattenNestedIndexedSlices(value)
    return indexed_slices.IndexedSlices(value.values / n, value.indices,
                                        value.dense_shape)
  else:
    return value / n


def copy_tensor_or_indexed_slices_to_device(value, device):
  # [FUNCTION DOC]
  # Copies a tensor or indexed slices onto a device by creating identity ops under that device
  # scope. For indexed slices the values, indices and (if present) dense shape are each copied.
  # value: tensor or indexed slices; device: device name string. Returns the copy.
  """Copies a tensor or IndexedSlices to a device."""
  with ops.device(device):
    if isinstance(value, indexed_slices.IndexedSlices):
      copied_values = array_ops.identity(value.values)
      copied_indices = array_ops.identity(value.indices)
      if value.dense_shape is not None:
        copied_shape = array_ops.identity(value.dense_shape)
      else:
        copied_shape = None
      result = indexed_slices.IndexedSlices(copied_values, copied_indices,
                                            copied_shape)
    else:
      result = array_ops.identity(value)
  return result


def is_indexed_slices(value):
  # [FUNCTION DOC]
  # Tells whether a value is sparse: true for indexed slices, or for a distributed values
  # object all of whose per-replica components are indexed slices. Otherwise false. Returns a
  # boolean.
  if isinstance(value, indexed_slices.IndexedSlices):
    return True
  if isinstance(value, value_lib.DistributedValues):
    return all(
        isinstance(v, indexed_slices.IndexedSlices) for v in value.values)
  return False


def split_by_sparsity(values):
  # [FUNCTION DOC]
  # Partitions a list of values into dense and sparse ones, remembering original positions.
  # Returns four lists in this order: dense values, their positions in the input, sparse
  # values, their positions. Inverse of stitch_values.
  """Split values into dense and sparse values.

  Args:
    values: a list of tensors or `PerReplica`s.

  Returns:
    Four lists:
      a list of dense values, a list of their indices in `values` and
      a list of sparse values, a list of their indices in `values`.
  """
  dense_values = []
  dense_indices = []
  sparse_values = []
  sparse_indices = []
  for i, v in enumerate(values):
    if is_indexed_slices(v):
      sparse_values.append(v)
      sparse_indices.append(i)
    else:
      dense_values.append(v)
      dense_indices.append(i)
  return dense_values, dense_indices, sparse_values, sparse_indices


def stitch_values(values_and_indices_list):
  # [FUNCTION DOC]
  # Rebuilds one list from several (values, positions) pairs, such as the dense and sparse
  # parts from split_by_sparsity. The result length is the total number of values; each value
  # is placed at its position. Empty pairs are skipped. Asserts that no position is filled
  # twice (assertion error otherwise); positions never filled stay as null values. Returns the
  # stitched list.
  """Stitch values together according to their indices.

  Args:
    values_and_indices_list: a list of tuples of values and indices indicating
      the values and positions in the returned list.

  Returns:
    a stitched list of values.
  """
  length = 0
  for values_and_indices in values_and_indices_list:
    length += len(values_and_indices[0])

  result = [None] * length
  for values_and_indices in values_and_indices_list:
    if values_and_indices and values_and_indices[0]:
      for v, i in zip(*values_and_indices):
        assert result[i] is None
        result[i] = v
  return result


def group_by_size(input_tensors, bytes_per_pack):
  # [FUNCTION DOC]
  # Splits a list of tensors into consecutive packs by byte size, keeping order. A new pack is
  # started when there is none or the current pack already exceeds bytes_per_pack, so packs
  # tend to be slightly larger than the target. If bytes_per_pack is zero, or any tensor has
  # an unknown element count (a warning is logged), everything is returned as a single pack.
  # Returns a list of lists of tensors.
  """Groups `input_tensors` into chunks of `bytes_per_pack`.

  The method preserves the original order of `input_tensors`. The grouping is
  best effort, each pack could have more or less bytes than `bytes_per_pack`.
  It only groups values with known shape.

  Args:
    input_tensors: a list of Tensor.
    bytes_per_pack: an integer.

  Returns:
    A list of packs of Tensor. All values are grouped into one pack if
    `bytes_per_pack` is zero or any of the value has unknown shape.
  """

  if bytes_per_pack == 0:
    return [input_tensors]
  packs = []
  last_pack_size = 0
  for value in input_tensors:
    num_elements = value.shape.num_elements()
    if num_elements is None:
      # Can't pack values with unknown shape.
      logging.warning(
          'not packing values due to the unknown or inconsistent shape of %s',
          value)
      return [input_tensors]
    size = num_elements * value.dtype.size
    # Try to keep each pack as close to bytes_per_pack as possible, while each
    # pack is at least bytes_per_pack large. I.E. we err on the side of having
    # few but large packs.
    if not packs or last_pack_size > bytes_per_pack:
      packs.append([])
      last_pack_size = 0
    packs[-1].append(value)
    last_pack_size += size
  return packs


def _pad_util(input_tensor, full_axis_dim):
  # [FUNCTION DOC]
  # HELPER: padding helper for the all-gather methods of the collective launcher.
  # Appends zeros at the end of the first dimension so its size becomes full_axis_dim (a
  # scalar integer tensor); other dimensions are unchanged. Returns the padded tensor.
  """Pad the `input_tensor`'s first dimension to be `full_axis_dim`."""
  missing_axis_dim = full_axis_dim - array_ops.shape_v2(input_tensor)[0]
  tensor_rank = array_ops.rank(input_tensor)
  paddings_axis = [[0, missing_axis_dim]]
  paddings = array_ops.concat([
      paddings_axis,
      array_ops.zeros(shape=(tensor_rank - 1, 2), dtype=dtypes.int32)
  ],
                              axis=0)
  padded_input_tensor = array_ops.pad(input_tensor, paddings)
  return padded_input_tensor
