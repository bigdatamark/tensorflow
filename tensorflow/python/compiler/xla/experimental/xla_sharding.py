# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================
# [MODULE DOC]
# Experimental helpers for annotating TensorFlow tensors, operations and resource variables with
# XLA sharding information (how a tensor is replicated, assigned to one core, tiled or split across
# devices). Exposes the Sharding class with factory constructors, convenience functions that build
# and apply a sharding in one call (replicate, assign_device, tile, split, partial_tile, mesh_split),
# functions that read sharding back from ops/tensors, copy it, and switch between automatic and
# manual SPMD partitioning. Sharding is stored as a serialized sharding message in an op attribute.
# Not a helper module; it is user-facing (experimental). The numerical library is imported under a
# private alias so it does not become part of the public API.
"""Experimental support for defining XLA shardings."""

import numpy as _np  # Avoids becoming a part of public Tensorflow API.

from tensorflow.compiler.tf2xla.python import xla as tf2xla
from xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.ops import resource_variable_ops


class Sharding(object):
  """A class to support adding sharding attributes to Ops.

  Use the factory constructors and then call apply_to_tensor:
    Sharding.replicate().apply_to_tensor(tensor)
  """
  # [CLASS DOC]
  # Wraps one XLA sharding message (describing replicated, manual, single-device, tiled or tuple
  # sharding) and applies it to tensors, operations or resource variables. Holds the message as its
  # only state. Instances are normally created by the class-level factory functions (replicate,
  # manual, assign_device, tile, subgroup_tile, partial_tile, split) rather than the constructor.

  def __init__(self, proto=None):
    # [FUNCTION DOC]
    # Constructor: stores the given sharding message (optional, default nothing/null). Intended to be
    # called by the factory functions, not by users.
    """Do not use this constructor; use the factory functions below."""
    self._proto = proto

  @classmethod
  def replicate(cls):
    # [FUNCTION DOC]
    # Factory: returns a Sharding of the replicated type, meaning the op is computed fully and
    # independently on every core. No inputs.
    """Returns a replicated sharding attribute.

    This causes an op to be computed in its entirety independently on all
    cores in the XLA device.
    """
    return Sharding(
        proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED))

  @classmethod
  def manual(cls):
    # [FUNCTION DOC]
    # Factory: returns a Sharding of the manual type, meaning the user partitions the op and XLA
    # leaves shapes unchanged. No inputs.
    """Returns a manuall sharding attribute.

    This means the op is manually partitioned by the user and XLA will not
    change the shapes.
    """
    return Sharding(
        proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.MANUAL))

  @classmethod
  def assign_device(cls, core):
    # [FUNCTION DOC]
    # Factory: returns a Sharding of the single-device (maximal) type that places the whole op on
    # the given core (integer core number). The tile assignment is one dimension of size one.
    """Returns an AssignDevice sharding attribute.

    This causes an op to be computed in its entirety only on one core in
    the XLA device.
    Args:
      core: The core to assign this Op to.
    """
    return Sharding(
        proto=xla_data_pb2.OpSharding(
            type=xla_data_pb2.OpSharding.MAXIMAL,
            tile_assignment_dimensions=[1],
            tile_assignment_devices=[core]))

  @classmethod
  def tile(cls, tile_assignment):
    # [FUNCTION DOC]
    # Factory: returns a tiled Sharding. Input: tile_assignment, a numeric n-dimensional array whose
    # shape gives the tiling topology and whose values give which device computes each tile
    # (flattened in row-major order). Raises a type error if it is not such an array.
    """Returns a Tiled sharding attribute.

    This causes an op to be partially computed on multiple cores in the
    XLA device.

    Args:
      tile_assignment: An np.ndarray describing the topology of the tiling and
        which device will compute which part of the topology.

    Raises:
      TypeError: tile_assignment was not of np.array type.

    TODO(jmolloy): This concept is nefarious and is not
    something we really want to expose to users (especially as the
    contract for tile_assignment is very strict).
    """
    if not isinstance(tile_assignment, _np.ndarray):
      raise TypeError('Tile assignment must be of type np.ndarray')
    dims = list(tile_assignment.shape)
    flattened_devices = tile_assignment.reshape(-1, order='C')
    return Sharding(
        proto=xla_data_pb2.OpSharding(
            type=xla_data_pb2.OpSharding.OTHER,
            tile_assignment_dimensions=dims,
            tile_assignment_devices=list(flattened_devices)))

  @classmethod
  def subgroup_tile(cls, tile_assignment, subgroup_modes):
    # [FUNCTION DOC]
    # Factory: returns a tiled Sharding with subgroup modes. Inputs: tile_assignment, an n-dimensional
    # array of device numbers; subgroup_modes, a list of sharding type constants (only replicated
    # or manual allowed) for the trailing extra dimensions. Raises a type error if the assignment is
    # not an array, the modes are not a list, the array rank is smaller than the number of modes, or
    # a mode is neither replicated nor manual.
    # KNOWN ISSUE: the docstring calls the result a "subgroup manual" sharding, but the modes may be
    # replicated or manual.
    """Returns a subgroup manual sharding attribute.

    This is similar to tile(), but tile_assignment has one or more dimension
    than the tensor, and subgroup_modes define the sharding types in the last
    dimensions of tile_assignment.

    Args:
      tile_assignment: An np.ndarray describing the topology of the tiling and
        which device will compute which part of the topology.
      subgroup_modes: sharding types for the dimension more than the tensor
        shape rank.

    Raises:
      TypeError: tile_assignment was not of np.array type or subgroup_modes
        has unsupported sharding type.
    """
    if not isinstance(tile_assignment, _np.ndarray):
      raise TypeError('SubgroupTile assignment must be of type np.ndarray')

    if not isinstance(subgroup_modes, list):
      raise TypeError('subgroup_modes in subgroup manual must be of type list')

    if len(tile_assignment.shape) < len(subgroup_modes):
      raise TypeError('SubgroupTile assignment must have rank larger than'
                      ' length of subgroup_modes')

    for sharding_type in subgroup_modes:
      if sharding_type not in [
          xla_data_pb2.OpSharding.REPLICATED, xla_data_pb2.OpSharding.MANUAL
      ]:
        raise TypeError(
            'Each sharding_type in subgroup_modes in subgroup manual must '
            'be of type xla_data_pb2.OpSharding.REPLICATED'
            ' or xla_data_pb2.OpSharding.MANUAL')
    dims = list(tile_assignment.shape)
    flattened_devices = tile_assignment.reshape(-1, order='C')
    return Sharding(
        proto=xla_data_pb2.OpSharding(
            type=xla_data_pb2.OpSharding.OTHER,
            tile_assignment_dimensions=dims,
            tile_assignment_devices=list(flattened_devices),
            last_tile_dims=list(subgroup_modes)))

  @classmethod
  def partial_tile(cls, tile_assignment):
    # [FUNCTION DOC]
    # Factory: returns a tiled Sharding whose last tile dimension is replicated. Input:
    # tile_assignment, an n-dimensional array of device numbers with one more dimension than the
    # tensor. Raises a type error if it is not an array.
    """Returns a partially tiled sharding attribute.

    This is similar to tile(), but tile_assignment has one more dimension than
    the tensor, and tiles in the last dimension of tile_assignment are
    replicated.

    Args:
      tile_assignment: An np.ndarray describing the topology of the tiling and
        which device will compute which part of the topology.

    Raises:
      TypeError: tile_assignment was not of np.array type.
    """
    if not isinstance(tile_assignment, _np.ndarray):
      raise TypeError('PartialTile assignment must be of type np.ndarray')
    dims = list(tile_assignment.shape)
    flattened_devices = tile_assignment.reshape(-1, order='C')
    return Sharding(
        proto=xla_data_pb2.OpSharding(
            type=xla_data_pb2.OpSharding.OTHER,
            tile_assignment_dimensions=dims,
            tile_assignment_devices=list(flattened_devices),
            replicate_on_last_tile_dim=True))

  @classmethod
  def split(cls, tensor, split_dimension, num_devices, input_shape=None):
    # [FUNCTION DOC]
    # Factory: returns a tiled Sharding that splits one dimension into num_devices pieces and keeps
    # the others whole. Inputs: tensor (used for its shape), split_dimension (integer axis),
    # num_devices (integer), input_shape (optional list; when given and non-empty it is used instead
    # of the tensor's shape). Raises a value error when the known size of the split dimension is
    # smaller than num_devices. Devices are assigned in order from zero.
    """Returns a Sharding that splits a tensor across a dimension.

    This creates a Tiled attribute, similar to tile(), but easier to use for the
    common case of tiling a tensor N ways in one dimension.

    Args:
      tensor: A tf.Tensor to split.
      split_dimension: The dimension number to split.
      num_devices: The number of cores to split `tensor` over.
      input_shape: The shape of the original tensor.

    Raises:
      ValueError: The tensor to split was smaller in the split dimension than
        the number of devices to split over.
    """
    if input_shape:
      shape = input_shape
    else:
      shape = tensor.shape.as_list()
    if (shape[split_dimension] is not None and
        shape[split_dimension] < num_devices):
      raise ValueError('Split dimension was smaller than the required number '
                       'of splits: shape=%r, dimension=%r, num_devices=%r' %
                       (shape, split_dimension, num_devices))

    tile_assignment_dims = [1] * len(shape)
    tile_assignment_dims[split_dimension] = num_devices

    return Sharding(
        proto=xla_data_pb2.OpSharding(
            type=xla_data_pb2.OpSharding.OTHER,
            tile_assignment_dimensions=tile_assignment_dims,
            tile_assignment_devices=range(num_devices)))

  def apply_to_tensor(
      self,
      tensor,
      assign_tuple_sharding=False,
      use_sharding_op=False,
      unspecified_dims=None,
      sharding_v2_proto=None,
  ):
    # [FUNCTION DOC]
    # Applies this sharding to a tensor or resource variable. Inputs: tensor; assign_tuple_sharding
    # (boolean, default false) to use a tuple sharding; use_sharding_op (boolean, default false) to
    # insert a separate sharding op; unspecified_dims (optional list of dimensions; requires the
    # sharding op and no tuple sharding, otherwise an assertion fails); sharding_v2_proto (optional
    # second-version sharding message). If the tensor is a resource variable and sharding for
    # resource variables is enabled, the sharding is stored on the variable and the variable is
    # returned (the other options except the tuple choice are ignored). Otherwise, with the sharding
    # op, wraps the tensor in a sharding op; with tuple sharding or a multi-output op, builds a
    # tuple sharding by replacing this tensor's entry in the op's existing (or default replicated)
    # tuple. Then sets the sharding attribute (and the v2 attribute if given) on the tensor's
    # op, mutating it. Returns the (possibly new) tensor.
    # KNOWN ISSUE: the unspecified_dims value is silently ignored when tuple sharding is requested
    # together with the sharding op (it is only passed on in the non-tuple case).
    """Applies this Sharding attribute to `tensor`.

    Args:
      tensor: A tf.Tensor to split.
      assign_tuple_sharding: If the sharding type should be a tuple.
      use_sharding_op: Whether to create a sharding op on `tensor`.
      unspecified_dims: An optional list of dimensions unspecified.
      sharding_v2_proto: The v2 sharding proto to use.

    Returns:
      The tensor with Sharding attribute.
    """
    if unspecified_dims:
      assert use_sharding_op and not assign_tuple_sharding
    proto = self._proto

    # If passed a tf.BaseResourceVariable instead of a tf.Tensor, simply store
    # the sharding proto on the tf.BaseResourceVariable object. An XlaShardingOp
    # will be created down the line whenever a ReadVariableOp is created by the
    # tf.BaseResourceVariable.
    if (
        isinstance(tensor, resource_variable_ops.BaseResourceVariable)
        and context.xla_sharding_for_resource_variables_enabled()
    ):
      if assign_tuple_sharding:
        proto = self._create_tuple_proto(num_outputs=1)
      # pylint: disable=protected-access
      tensor._set_xla_sharding(proto)
      return tensor

    if use_sharding_op:
      if assign_tuple_sharding:
        proto = self._create_tuple_proto(num_outputs=1)
        tensor = tf2xla.sharding(tensor, sharding=proto.SerializeToString())
      else:
        tensor = tf2xla.sharding(
            tensor,
            sharding=proto.SerializeToString(),
            unspecified_dims=unspecified_dims or [],
        )
    elif assign_tuple_sharding or len(tensor.op.outputs) > 1:
      proto = self._get_or_create_tuple_proto(tensor.op)
      # We can't mutate an element of old_proto.tuple_shardings, so create
      # a new proto.
      tuple_shardings = list(proto.tuple_shardings)
      tuple_shardings[tensor.value_index] = self._proto
      proto = xla_data_pb2.OpSharding(
          type=xla_data_pb2.OpSharding.TUPLE, tuple_shardings=tuple_shardings)

    # TODO(jmolloy): This need to be seriously revisited before declaring this
    # API available for public use.
    # pylint: disable=protected-access
    tensor.op._set_attr('_XlaSharding',
                        attr_value_pb2.AttrValue(s=proto.SerializeToString()))
    if sharding_v2_proto:
      tensor.op._set_attr(
          '_XlaShardingV2',
          attr_value_pb2.AttrValue(s=sharding_v2_proto.SerializeToString()),
      )
    return tensor

  def apply_to_operation(self, operation):
    # [FUNCTION DOC]
    # Sets the sharding attribute on the given operation using this sharding's serialized message.
    # Mutates the operation; returns nothing.
    """Applies this Sharding attribute to `operation`.

    Args:
      operation: A tf.Operation to add sharding annotation.
    """
    attr_value = attr_value_pb2.AttrValue(s=self._proto.SerializeToString())
    # pylint: disable=protected-access
    operation._set_attr('_XlaSharding', attr_value)

  @property
  def proto(self):
    # [FUNCTION DOC]
    # Property: returns the wrapped sharding message.
    """Return the sharding protobuf of type xla_data_pb2.OpSharding."""
    return self._proto

  def _get_or_create_tuple_proto(self, op):
    # [FUNCTION DOC]
    # HELPER: used by apply_to_tensor. Returns the sharding message already stored on the op's
    # sharding attribute (parsed), or, if the op has no such attribute (a value error on lookup),
    # a new tuple sharding with one replicated entry per op output.
    try:
      attr = op.get_attr('_XlaSharding')
      proto = xla_data_pb2.OpSharding()
      proto.ParseFromString(attr)
      return proto
    except ValueError:
      return self._create_tuple_proto(len(op.outputs))

  def _create_tuple_proto(self, num_outputs):
    # [FUNCTION DOC]
    # HELPER: builds a tuple sharding message with num_outputs (integer) replicated entries.
    shardings = [
        xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED)
    ] * num_outputs
    return xla_data_pb2.OpSharding(
        type=xla_data_pb2.OpSharding.TUPLE, tuple_shardings=shardings)


def copy_sharding(from_tensor, to_tensor, use_sharding_op=False):
  # [FUNCTION DOC]
  # Copies the sharding of one tensor onto another. Inputs: from_tensor (source), to_tensor
  # (destination, may be a resource variable), use_sharding_op (boolean, default false) to wrap the
  # destination in a sharding op. Reads the serialized sharding of the source; if there is none,
  # returns the destination unchanged. For a resource variable (with the feature enabled) stores the
  # parsed sharding on the variable and returns it. Otherwise optionally wraps in a sharding op and
  # sets the sharding attribute on the destination's op. Returns the destination tensor.
  """Copies the a tensor's sharding to another.

  Args:
    from_tensor: Source tensor. Must be the sole output of an op.
    to_tensor: the tensor the annotate with the copy.
    use_sharding_op: whether to create a sharding op on `to_tensor`.

  Returns:
    A tensor with sharding annotation copied from `from_tensor`.
  """
  sharding = get_tensor_sharding(from_tensor)
  if sharding is None:
    return to_tensor

  # If passed a tf.BaseResourceVariable instead of a tf.Tensor, simply store the
  # sharding proto on the tf.BaseResourceVariable object. An XlaShardingOp
  # will be created down the line whenever a ReadVariableOp is created by the
  # tf.BaseResourceVariable.
  if (
      isinstance(to_tensor, resource_variable_ops.BaseResourceVariable)
      and context.xla_sharding_for_resource_variables_enabled()
  ):
    proto = xla_data_pb2.OpSharding()
    proto.ParseFromString(sharding)
    # pylint: disable=protected-access
    to_tensor._set_xla_sharding(proto)
    return to_tensor

  if use_sharding_op:
    to_tensor = tf2xla.sharding(to_tensor, sharding=sharding)
  attr_value = attr_value_pb2.AttrValue(s=sharding)
  # pylint: disable=protected-access
  to_tensor.op._set_attr('_XlaSharding', attr_value)
  return to_tensor

# Helpers for the above factory functions that allow easy application of
# shardings, for example:
#   tensor = xla_sharding.replicate(tensor)


def replicate(tensor, assign_tuple_sharding=False, use_sharding_op=False):
  # [FUNCTION DOC]
  # Convenience wrapper: applies a replicated sharding to the tensor. Inputs: tensor, plus the
  # optional booleans assign_tuple_sharding and use_sharding_op (both default false), forwarded to
  # the apply step. Returns the annotated tensor.
  return Sharding.replicate().apply_to_tensor(
      tensor,
      assign_tuple_sharding=assign_tuple_sharding,
      use_sharding_op=use_sharding_op)


def assign_device(tensor,
                  device,
                  assign_tuple_sharding=False,
                  use_sharding_op=False):
  # [FUNCTION DOC]
  # Convenience wrapper: applies a single-device sharding that places the tensor on the given
  # device (integer core). Optional booleans assign_tuple_sharding and use_sharding_op (default
  # false) are forwarded. Returns the annotated tensor.
  """Returns a tensor that has AssignDevice sharding attribute."""
  return Sharding.assign_device(device).apply_to_tensor(
      tensor,
      assign_tuple_sharding=assign_tuple_sharding,
      use_sharding_op=use_sharding_op)


def tile(tensor,
         tile_assignment,
         assign_tuple_sharding=False,
         use_sharding_op=False,
         unspecified_dims=None):
  # [FUNCTION DOC]
  # Convenience wrapper: applies a tiled sharding built from tile_assignment (numeric array of
  # device numbers) to the tensor. Optional: assign_tuple_sharding and use_sharding_op (booleans,
  # default false), unspecified_dims (list, default empty). Returns the annotated tensor. Raises
  # a type error for a non-array assignment; non-empty unspecified_dims without the sharding op
  # option fails an assertion in the apply step.
  # KNOWN ISSUE: the docstring has no return description; it returns the annotated tensor.
  """Returns a tensor that has tiled sharding.

  Args:
    tensor: A tf.Tensor to shard.
    tile_assignment: An np.ndarray describing the topology of the tiling and
      which device will compute which part of the topology.
    assign_tuple_sharding: If the sharding type should be a tuple.
    use_sharding_op: If true, adds a sharding op to set the sharding.
    unspecified_dims: An optional list of dimensions unspecified.
  """
  return Sharding.tile(tile_assignment).apply_to_tensor(
      tensor,
      assign_tuple_sharding=assign_tuple_sharding,
      use_sharding_op=use_sharding_op,
      unspecified_dims=unspecified_dims or [])


def split(tensor,
          split_dimension,
          num_devices,
          assign_tuple_sharding=False,
          use_sharding_op=False,
          input_shape=None):
  # [FUNCTION DOC]
  # Convenience wrapper: splits the tensor along split_dimension across num_devices devices.
  # Optional: assign_tuple_sharding and use_sharding_op (booleans, default false), input_shape
  # (list overriding the tensor's shape). Raises a value error if the dimension is smaller than
  # the device count. Returns the annotated tensor.
  """Returns a tensor that is split along the given dimension.

  Args:
    tensor: A tf.Tensor to split.
    split_dimension: The dimension to split.
    num_devices: The number of devices to partition the dimension.
    assign_tuple_sharding: If the sharding type should be a tuple.
    use_sharding_op: If true, adds a sharding op to set the sharding.
    input_shape: The full shape of the input tensor.
  """
  return Sharding.split(tensor, split_dimension, num_devices,
                        input_shape).apply_to_tensor(
                            tensor,
                            assign_tuple_sharding=assign_tuple_sharding,
                            use_sharding_op=use_sharding_op)


def partial_tile(tensor,
                 tile_assignment,
                 use_sharding_op=False,
                 unspecified_dims=None):
  # [FUNCTION DOC]
  # Convenience wrapper: applies a partially tiled sharding (last assignment dimension replicated)
  # to the tensor. Optional: use_sharding_op (boolean, default false), unspecified_dims (list,
  # default empty). Returns the annotated tensor. Raises a type error for a non-array assignment.
  """Returns a tensor that has tiled sharding.

  Args:
    tensor: A tf.Tensor to shard.
    tile_assignment: An np.ndarray describing the topology of the tiling and
      which device will compute which part of the topology. It must have one
      more dimension than tensor, and the last dimension represents partially
      replicated tiles.
    use_sharding_op: If true, adds a sharding op to set the sharding.
    unspecified_dims: An optional list of dimensions unspecified.
  """
  return Sharding.partial_tile(tile_assignment).apply_to_tensor(
      tensor,
      use_sharding_op=use_sharding_op,
      unspecified_dims=unspecified_dims or [])


def get_op_sharding(op):
  # [FUNCTION DOC]
  # Reads the sharding attribute of an op. Returns the serialized sharding bytes, or nothing/null
  # if the attribute is absent or the object has no attribute-lookup capability (such as a
  # distributed variable op).
  """Returns sharding attribute of an op.

  Args:
    op: a TensorFlow op.

  Returns:
    The attribute representing XLA sharding on this op.
  """
  try:
    return op.get_attr('_XlaSharding')
  except ValueError:
    return None
  except AttributeError:
    # AttributeError: 'DistributedVarOp' object has no attribute 'get_attr'.
    return None


def get_tensor_sharding(tensor):
  # [FUNCTION DOC]
  # Reads the sharding of a tensor or resource variable. For a resource variable (with the feature
  # enabled) returns its stored sharding serialized, or nothing if none was set. Otherwise returns
  # the sharding attribute of the tensor's op, or nothing/null when absent or when the tensor has
  # no op (eager mode).
  """Returns sharding attribute of a Tensor.

  Args:
    tensor: a Tensor.

  Returns:
    The attribute representing XLA sharding on tensor's op.
  """
  # If passed a tf.BaseResourceVariable instead of a tf.Tensor, simply get the
  # sharding proto set on the _xla_sharding field of the tf.BaseResourceVariable
  # object.
  if (
      isinstance(tensor, resource_variable_ops.BaseResourceVariable)
      and context.xla_sharding_for_resource_variables_enabled()
  ):
    # pylint: disable=protected-access
    sharding = tensor._get_xla_sharding()
    if sharding is None:
      return None
    else:
      return sharding.SerializeToString()

  try:
    return get_op_sharding(tensor.op)
  except AttributeError:
    # AttributeError: Tensor.op is meaningless when eager execution is enabled.
    return None


def get_sharding_tile_shape(sharding):
  # [FUNCTION DOC]
  # Parses a serialized sharding message (or nothing/null) and returns its tile assignment
  # dimensions, i.e. the number of shards per tensor dimension. Returns nothing/null when the input
  # is null or has no tile dimensions.
  """Returns the tile assignment shape for a sharded Tensor.

  Args:
    sharding: a serialized OpSharding message describing the layout of a
      sharded Tensor.

  Returns:
    A list, for each dimension of the sharded Tensor, of the number of shards
      into which it has been split. Returns None if the input indicates no tile
      assignments.
  """
  if sharding is None:
    return None
  sharding_message = xla_data_pb2.OpSharding()
  sharding_message.ParseFromString(sharding)
  if sharding_message.tile_assignment_dimensions:
    return sharding_message.tile_assignment_dimensions
  else:
    return None


def auto_to_manual_spmd_partition(tensor,
                                  manual_sharding,
                                  single_dim=-1,
                                  unspecified_dims=None):
  # [FUNCTION DOC]
  # Converts a full-shaped tensor into a shard-shaped one for manually partitioned ops via the XLA
  # full-to-shard operation. Inputs: tensor; manual_sharding (serialized sharding string);
  # single_dim (integer, default -1, restricts conversion to one subgroup dimension when
  # non-negative); unspecified_dims (optional list, default empty). Returns the shard-shaped tensor.
  """Switches from automatic SPMD partitioning to manual partitioning.

  Converts a full-shaped tensor (to be automatically partitioned by SPMD
  partitioner) to a shard-shaped tensor to be consumed by manually partitioned
  ops.

  Args:
    tensor: A tf.Tensor in full shape.
    manual_sharding: A serialized string of OpSharding to be used in manual
      partitioning.
    single_dim: If >= 0, the conversion will happen only on this dim in
      subgroups.
    unspecified_dims: An optional list of dimensions unspecified.

  Returns:
    A shard-shaped tensor to be consumed by manually partitioned ops.
  """
  return tf2xla.spmd_full_to_shard_shape(
      tensor,
      manual_sharding=manual_sharding,
      dim=single_dim,
      unspecified_dims=unspecified_dims or [])


def manual_to_auto_spmd_partition(tensor,
                                  manual_sharding,
                                  full_shape,
                                  single_dim=-1,
                                  unspecified_dims=None):
  # [FUNCTION DOC]
  # Converts a shard-shaped tensor back to a full-shaped one for automatic SPMD partitioning via the
  # XLA shard-to-full operation. Inputs: tensor; manual_sharding (serialized sharding string);
  # full_shape (shape before partitioning); single_dim (integer, default -1); unspecified_dims
  # (optional list, default empty). Returns the full-shaped tensor.
  """Switches from manual partitioning to automatic SPMD partitioning.

  Converts a shard-shaped tensor (manually partitioned in SPMD-style) to a
  full-shaped tensor to be partitioned automatically by the SPMD partitioner.

  Args:
    tensor: A tf.Tensor in shard shape.
    manual_sharding: a serialized string of OpSharding to be used in manual
      partitioning.
    full_shape: the shape of tensor before partitioning.
    single_dim: If >= 0, the conversion will happen only on this dim in
      subgroups.
    unspecified_dims: An optional list of dimensions unspecified.

  Returns:
    A full-shaped tensor to be partitioned automatically by the SPMD
    partitioner.
  """
  return tf2xla.spmd_shard_to_full_shape(
      tensor,
      manual_sharding=manual_sharding,
      full_shape=full_shape,
      dim=single_dim,
      unspecified_dims=unspecified_dims or [])


def mesh_split_sharding(device_mesh,
                        tensor_split_dims_mapping,
                        manual_mesh_dims=None):
  # [FUNCTION DOC]
  # Builds a Sharding from a device mesh. Inputs: device_mesh (n-dimensional array of device ids);
  # tensor_split_dims_mapping (list with one integer per tensor axis giving the mesh axis it is
  # split along, or negative for unsplit); manual_mesh_dims (optional list of mesh axes handled
  # manually). Raises a value error if more mesh axes are used than the mesh has. Transposes the
  # mesh so used axes come first, derives tile shape, appends a replicated trailing dimension if
  # some mesh axes are unused (partial), and returns a subgroup tiled sharding when manual axes are
  # given, a partially tiled one when only partial, otherwise a tiled one.
  # KNOWN ISSUE: the docstring lacks a Returns section; the result is a Sharding object.
  """Returns a Sharding object representing sharding along multiple dimensions.

  Args:
    device_mesh: An np.ndarray describing the topology of the device mesh and
      each element is the ID of the device in the topology.
    tensor_split_dims_mapping: A list of integers that map each tensor axis to
      the device mesh axis along which it is sharded. Its length is the tensor
      rank, and tensor_split_dims_mapping[i] is device mesh axis for tensor
      dimension i. Use -1 for tensor dimensions that are not sharded.
    manual_mesh_dims: An optional list of mesh dims for manual subgroups.

  Raises:
    ValueError: The number of tensor split dimensions is larger than device mesh
      rank.
  """
  manual_mesh_dims = manual_mesh_dims or []
  permutation = [d for d in tensor_split_dims_mapping if d >= 0
                ] + manual_mesh_dims
  if len(permutation) > len(device_mesh.shape):
    raise ValueError(
        'Number of tensor split dimensions (%r) is larger than device mesh '
        'rank (%r). tensor_split_dims_mapping: %r, device_mesh.shape: %r' %
        (len(permutation), len(
            device_mesh.shape), tensor_split_dims_mapping, device_mesh.shape))
  # Append replicated dimensions to the end.
  transpose_permutation = permutation + [
      d for d in range(len(device_mesh.shape)) if d not in permutation
  ]
  tile_assignment = _np.transpose(device_mesh, transpose_permutation)
  tile_shape = [
      1 if d < 0 else device_mesh.shape[d]
      for d in (tensor_split_dims_mapping + manual_mesh_dims)
  ]
  subgroup_modes = [xla_data_pb2.OpSharding.MANUAL] * len(manual_mesh_dims)
  partial = len(permutation) < len(device_mesh.shape)
  if partial:
    tile_shape.append(_np.prod(device_mesh.shape) // _np.prod(tile_shape))
    subgroup_modes.append(xla_data_pb2.OpSharding.REPLICATED)
  tile_assignment = _np.reshape(tile_assignment, tile_shape)

  if manual_mesh_dims:
    return Sharding.subgroup_tile(tile_assignment, subgroup_modes)

  if partial:
    return Sharding.partial_tile(tile_assignment)
  return Sharding.tile(tile_assignment)


def mesh_split(tensor,
               device_mesh,
               tensor_split_dims_mapping,
               use_sharding_op=False,
               manual_mesh_dims=None,
               unspecified_dims=None):
  # [FUNCTION DOC]
  # Splits a tensor along multiple dimensions of a device mesh. Builds the sharding with the mesh
  # helper from device_mesh, tensor_split_dims_mapping and optional manual_mesh_dims, then applies
  # it with use_sharding_op (boolean, default false) and unspecified_dims (default empty list).
  # Raises a value error if too many mesh axes are used. Returns the annotated tensor.
  """Returns a tensor that is split along multiple dimensions in a device mesh.

  Args:
    tensor: A tf.Tensor to split.
    device_mesh: An np.ndarray describing the topology of the device mesh and
      each element is the ID of the device in the topology.
    tensor_split_dims_mapping: A list of integers that map each tensor axis to
      the device mesh axis along which it is sharded. Its length is the tensor
      rank, and tensor_split_dims_mapping[i] is device mesh axis for tensor
      dimension i. Use -1 for tensor dimensions that are not sharded.
    use_sharding_op: If true, adds a sharding op to set the sharding.
    manual_mesh_dims: An optional list of mesh dims for manual subgroups.
    unspecified_dims: An optional list of dimensions unspecified.

  Raises:
    ValueError: The number of tensor split dimensions is larger than device mesh
      rank.
  """
  sharding = mesh_split_sharding(device_mesh, tensor_split_dims_mapping,
                                 manual_mesh_dims)
  return sharding.apply_to_tensor(
      tensor,
      use_sharding_op=use_sharding_op,
      unspecified_dims=unspecified_dims or [])
