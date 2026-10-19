# [MODULE DOC]
# Defines partitioner classes (fixed shards, minimum shard size, maximum shard size) used to decide
# how a large variable is split along its first axis, and the ShardedVariable container that
# treats a list of smaller variables as shards of one logical variable. Also provides the type
# specification for sharded variables, a mixin holding most of the behavior (indexing, scatter and
# assign ops, sparse reads, checkpoint and SavedModel hooks), and dispatch overrides for embedding
# lookup so sharded variables are passed to lookup ops as a list of shard variables.
# Import-time side effects: registers a tensor conversion function for ShardedVariable (which
# concatenates the shards), overloads all tensor operators on ShardedVariable, and registers
# operator dispatch overrides for the embedding lookup and safe sparse embedding lookup APIs.
# Several classes are exported publicly under the distribute partitioners namespace.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""ShardedVariable class."""
import copy
import math
from typing import Sequence
import weakref

import numpy as np

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export('distribute.experimental.partitioners.Partitioner', v1=[])
class Partitioner(object):
  # [CLASS DOC]
  # Abstract base class for variable partitioners. A partitioner is a callable that, given a
  # shape, a dtype and an axis, returns the number of partitions along each axis of the shape.
  # Subclasses override the call method; the base implementation is not usable directly.
  """Partitioner base class: all partitiners inherit from this class.

  Partitioners should implement a `__call__` method with the following
  signature:

  ```python
  def __call__(self, shape, dtype, axis=0):
    # Partitions the given `shape` and returns the partition results.
    # See docstring of `__call__` method for the format of partition results.
  ```
  """

  def __call__(self, shape, dtype, axis=0):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring example calls a subclass with a malformed shape argument and
    # shows an output of two and zero, which does not match what real partitioners return.
    # Abstract partition method. Inputs: shape (tensor shape to partition), dtype (element type
    # of the variable) and axis (axis to split along, default 0). Always raises a
    # not-implemented error; subclasses return a list with one partition count per axis.
    """Partitions the given `shape` and returns the partition results.

    Examples of a partitioner that allocates a fixed number of shards:

    ```python
    partitioner = FixedShardsPartitioner(num_shards=2)
    partitions = partitioner(tf.TensorShape([10, 3], tf.float32), axis=0)
    print(partitions) # [2, 0]
    ```

    Args:
      shape: a `tf.TensorShape`, the shape to partition.
      dtype: a `tf.dtypes.Dtype` indicating the type of the partition value.
      axis: The axis to partition along.  Default: outermost axis.

    Returns:
      A list of integers representing the number of partitions on each axis,
      where i-th value corresponds to i-th axis.
    """
    raise NotImplementedError


@tf_export('distribute.experimental.partitioners.FixedShardsPartitioner', v1=[])
class FixedShardsPartitioner(Partitioner):
  # [CLASS DOC]
  # Partitioner that always requests a fixed number of shards along the chosen axis, capped by
  # the size of that axis. Holds the configured shard count. Typically passed as the variable
  # partitioner of a parameter server strategy.
  """Partitioner that allocates a fixed number of shards.

  Examples:

  >>> # standalone usage:
  >>> partitioner = FixedShardsPartitioner(num_shards=2)
  >>> partitions = partitioner(tf.TensorShape([10, 3]), tf.float32)
  >>> [2, 1]
  >>>
  >>> # use in ParameterServerStrategy
  >>> # strategy = tf.distribute.experimental.ParameterServerStrategy(
  >>> #   cluster_resolver=cluster_resolver, variable_partitioner=partitioner)
  """

  def __init__(self, num_shards):
    # [FUNCTION DOC]
    # Constructor. Input: num_shards, integer number of shards wanted. Stores it; no validation
    # is done and nothing is returned.
    """Creates a new `FixedShardsPartitioner`.

    Args:
      num_shards: `int`, number of shards to partition.
    """
    self._num_shards = num_shards

  def __call__(self, shape, dtype, axis=0):
    # [FUNCTION DOC]
    # Returns a list with one entry per dimension of shape, all ones except the entry at axis,
    # which is the smaller of the configured shard count and the size of that dimension. The
    # dtype is ignored. Shape must have known dimension sizes (a tensor shape object).
    del dtype
    result = [1] * len(shape)
    result[axis] = min(self._num_shards, shape.dims[axis].value)
    return result


@tf_export('distribute.experimental.partitioners.MinSizePartitioner', v1=[])
class MinSizePartitioner(Partitioner):
  # [CLASS DOC]
  # Partitioner that creates as many shards as possible while keeping every shard at least a
  # minimum number of bytes, up to a maximum shard count. Delegates the computation to the
  # min-max variable partitioner helper. Holds minimum shard bytes, maximum shards and an
  # estimated size per string element.
  """Partitioner that allocates a minimum size per shard.

  This partitioner ensures each shard has at least `min_shard_bytes`, and tries
  to allocate as many shards as possible, i.e., keeping shard size as small as
  possible. The maximum number of such shards (upper bound) is given by
  `max_shards`.

  Examples:

  >>> partitioner = MinSizePartitioner(min_shard_bytes=4, max_shards=2)
  >>> partitions = partitioner(tf.TensorShape([6, 1]), tf.float32)
  >>> [2, 1]
  >>> partitioner = MinSizePartitioner(min_shard_bytes=4, max_shards=10)
  >>> partitions = partitioner(tf.TensorShape([6, 1]), tf.float32)
  >>> [6, 1]
  >>>
  >>> # use in ParameterServerStrategy
  >>> # strategy = tf.distribute.experimental.ParameterServerStrategy(
  >>> #   cluster_resolver=cluster_resolver, variable_partitioner=partitioner)
  """

  def __init__(
      self, min_shard_bytes=256 << 10, max_shards=1, bytes_per_string=16
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: min_shard_bytes (minimum bytes per shard, default 256 KiB),
    # max_shards (upper bound on shard count, default 1) and bytes_per_string (estimated bytes
    # per string element, default 16). Each must be at least 1, otherwise raises a value error.
    # Stores the three values; returns nothing.
    """Creates a new `MinSizePartitioner`.

    Args:
      min_shard_bytes: Minimum bytes of each shard. Defaults to 256K.
      max_shards: Upper bound on the number of shards. Defaults to 1.
      bytes_per_string: If the partition value is of type string, this provides
        an estimate of how large each string is.
    """
    if min_shard_bytes < 1:
      raise ValueError(
          'Argument `min_shard_bytes` must be positive. '
          f'Received: {min_shard_bytes}'
      )
    if max_shards < 1:
      raise ValueError(
          f'Argument `max_shards` must be positive. Received: {max_shards}'
      )
    if bytes_per_string < 1:
      raise ValueError(
          'Argument `bytes_per_string` must be positive. '
          f'Received: {bytes_per_string}'
      )
    self._min_shard_bytes = min_shard_bytes
    self._max_shards = max_shards
    self._bytes_per_string = bytes_per_string

  def __call__(self, shape, dtype, axis=0):
    # [FUNCTION DOC]
    # Builds a min-max variable partitioner from the stored settings (maximum partitions,
    # minimum slice size, bytes per string element, and the given axis) and applies it to shape
    # and dtype. Returns the list of per-axis partition counts it produces.
    return partitioned_variables.min_max_variable_partitioner(
        max_partitions=self._max_shards,
        axis=axis,
        min_slice_size=self._min_shard_bytes,
        bytes_per_string_element=self._bytes_per_string,
    )(shape, dtype)


@tf_export('distribute.experimental.partitioners.MaxSizePartitioner', v1=[])
class MaxSizePartitioner(Partitioner):
  # [CLASS DOC]
  # Partitioner that uses as few shards as possible while keeping each shard at or below a
  # maximum byte size, optionally capped by a maximum shard count (which takes precedence).
  # Delegates to the axis-size variable partitioner helper. Holds maximum shard bytes, maximum
  # shards (may be nothing) and an estimated size per string element.
  """Partitioner that keeps shards below `max_shard_bytes`.

  This partitioner ensures each shard has at most `max_shard_bytes`, and tries
  to allocate as few shards as possible, i.e., keeping shard size as large
  as possible.

  If the partitioner hits the `max_shards` limit, then each shard may end up
  larger than `max_shard_bytes`. By default `max_shards` equals `None` and no
  limit on the number of shards is enforced.

  Examples:

  >>> partitioner = MaxSizePartitioner(max_shard_bytes=4)
  >>> partitions = partitioner(tf.TensorShape([6, 1]), tf.float32)
  >>> [6, 1]
  >>> partitioner = MaxSizePartitioner(max_shard_bytes=4, max_shards=2)
  >>> partitions = partitioner(tf.TensorShape([6, 1]), tf.float32)
  >>> [2, 1]
  >>> partitioner = MaxSizePartitioner(max_shard_bytes=1024)
  >>> partitions = partitioner(tf.TensorShape([6, 1]), tf.float32)
  >>> [1, 1]
  >>>
  >>> # use in ParameterServerStrategy
  >>> # strategy = tf.distribute.experimental.ParameterServerStrategy(
  >>> #   cluster_resolver=cluster_resolver, variable_partitioner=partitioner)
  """

  def __init__(self, max_shard_bytes, max_shards=None, bytes_per_string=16):
    # [FUNCTION DOC]
    # Constructor. Inputs: max_shard_bytes (required maximum bytes per shard), max_shards
    # (optional cap on shard count, default none meaning unlimited) and bytes_per_string
    # (estimated bytes per string element, default 16). Raises a value error if max_shard_bytes
    # or bytes_per_string is below 1, or if max_shards is given and below 1. Stores the values.
    """Creates a new `MaxSizePartitioner`.

    Args:
      max_shard_bytes: The maximum size any given shard is allowed to be.
      max_shards: The maximum number of shards in `int` created taking
        precedence over `max_shard_bytes`.
      bytes_per_string: If the partition value is of type string, this provides
        an estimate of how large each string is.
    """
    if max_shard_bytes < 1:
      raise ValueError(
          'Argument `max_shard_bytes` must be positive. '
          f'Received {max_shard_bytes}'
      )
    if max_shards and max_shards < 1:
      raise ValueError(
          f'Argument `max_shards` must be positive. Received {max_shards}'
      )
    if bytes_per_string < 1:
      raise ValueError(
          'Argument `bytes_per_string` must be positive. '
          f'Received: {bytes_per_string}'
      )

    self._max_shard_bytes = max_shard_bytes
    self._max_shards = max_shards
    self._bytes_per_string = bytes_per_string

  def __call__(self, shape, dtype, axis=0):
    # [FUNCTION DOC]
    # Builds an axis-size variable partitioner from the stored settings (maximum shard bytes,
    # maximum shards, bytes per string element and the given axis) and applies it to shape and
    # dtype. Returns the list of per-axis partition counts it produces.
    return partitioned_variables.variable_axis_size_partitioner(
        max_shard_bytes=self._max_shard_bytes,
        max_shards=self._max_shards,
        bytes_per_string_element=self._bytes_per_string,
        axis=axis,
    )(shape, dtype)


class ShardedVariableSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # Type specification for a sharded variable: holds one variable spec per shard. Used so a
  # sharded variable can act as a composite value, being split into and rebuilt from its shard
  # variables. The value type is the ShardedVariable class.
  """Type specification for a `ShardedVariable`."""

  __slots__ = ['_variable_specs']

  value_type = property(lambda self: ShardedVariable)

  def __init__(self, *variable_specs):
    # [FUNCTION DOC]
    # Constructor. Takes any number of per-shard variable specs and stores them as a tuple.
    self._variable_specs = tuple(variable_specs)

  def _serialize(self):
    # [FUNCTION DOC]
    # Returns the tuple of per-shard variable specs as the serialized form of this spec.
    return self._variable_specs

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Property returning the tuple of per-shard variable specs as the component specs.
    return self._variable_specs

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Splits a sharded variable value into a tuple of its shard variables.
    return tuple(value.variables)

  def _from_components(self, variables):
    # [FUNCTION DOC]
    # Rebuilds a sharded variable from the given sequence of shard variables.
    return ShardedVariable(variables)

  def _cast(self, value, _):
    # [FUNCTION DOC]
    # Cast hook: returns the value unchanged; the second argument (target spec) is ignored.
    return value


class ShardedVariableMixin(trackable.Trackable):
  # [CLASS DOC]
  # Mixin implementing the behavior of a sharded variable: a container over a list of shard
  # variables (split along the first axis) that exposes combined shape and dtype, slicing,
  # assign and scatter operations, sparse reads, and checkpoint and SavedModel integration.
  # State: the shard list, the container name, combined dtype and shape, per-shard offsets and an
  # uninitialized full-shape variable used when saving. Exists separately from ShardedVariable
  # because some embedding variable types cannot be composite tensors.
  """Mixin for ShardedVariable."""

  # TODO(b/170877138): Remove this mixin once fixed. This mixin is required
  # since TPUEmbeddingVariable can't be a CompositeTensor.

  def __init__(self, variables, name='ShardedVariable'):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says save slice info should not be set for all elements, but the
    # code raises when it is set on ANY element. The shard list and name are also stored before
    # the arguments are validated.
    # Constructor. Inputs: variables (non-empty sequence of variable objects, all with the same
    # dtype and the same shape except along axis 0) and name (string, default "ShardedVariable").
    # Raises a type error if variables is not a non-empty sequence of variables, a value error if
    # dtypes differ, trailing shapes differ, or any variable already has save slice info.
    # Computes combined dtype, combined shape (axis 0 summed), and the start offset of each
    # shard along axis 0 (zero elsewhere). Side effects: sets a weak reference to this container
    # on every shard, and creates an uninitialized full-shape variable (no synchronization or
    # aggregation) used as a stand-in when saving.
    """Treats `variables` as shards of a larger Variable.

    Example:

    ```
    variables = [
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
      tf.Variable(..., shape=(15, 100), dtype=tf.float32),
      tf.Variable(..., shape=(5, 100), dtype=tf.float32)
    ]
    sharded_variable = ShardedVariableMixin(variables)
    assert sharded_variable.shape.as_list() == [30, 100]
    ```

    Args:
      variables: A list of `ResourceVariable`s that comprise this sharded
        variable. Variables should not be shared between different
        `ShardedVariableMixin` objects.
      name: String. Name of this container. Defaults to "ShardedVariable".
    """
    super(ShardedVariableMixin, self).__init__()
    self._variables = variables
    self._name = name

    if (
        not isinstance(variables, Sequence)
        or not variables
        or any(not isinstance(v, variables_lib.Variable) for v in variables)
    ):
      raise TypeError(
          'Argument `variables` should be a non-empty list of '
          f'`variables.Variable`s. Received {variables}'
      )

    var_dtypes = {v.dtype for v in variables}
    if len(var_dtypes) > 1:
      raise ValueError(
          'All elements in argument `variables` must have the same dtype. '
          f'Received dtypes: {[v.dtype for v in variables]}'
      )

    first_var = variables[0]
    self._dtype = first_var.dtype

    # All variables must have the same shape for axes > 0.
    higher_dim_shapes = {tuple(v.shape.as_list()[1:]) for v in variables}
    if len(higher_dim_shapes) > 1:
      raise ValueError(
          'All elements in argument `variables` must have the same shapes '
          'except for the first axis. '
          f'Received shapes: {[v.shape for v in variables]}'
      )
    first_dim = sum(int(v.shape.as_list()[0]) for v in variables)
    self._shape = tensor_shape.TensorShape(
        [first_dim] + first_var.shape.as_list()[1:]
    )

    for v in variables:
      v._sharded_container = weakref.ref(self)

    self._var_offsets = [
        [0 for _ in range(len(first_var.shape))] for _ in range(len(variables))
    ]
    for i in range(1, len(variables)):
      # Always partition on the first axis. Offsets on other axes are 0.
      self._var_offsets[i][0] += (
          self._var_offsets[i - 1][0] + variables[i - 1].shape.as_list()[0]
      )

    save_slice_info = [v._get_save_slice_info() for v in variables]  # pylint: disable=protected-access
    if any(slice_info is not None for slice_info in save_slice_info):
      raise ValueError(
          '`SaveSliceInfo` should not be set for all elements in argument '
          '`variables`. `ShardedVariable` will infer `SaveSliceInfo` according '
          'to the order of the elements `variables`. '
          f'Received save slice info {save_slice_info}'
      )

    # We create an uninitialized saving_variable with the full shape, which can
    # be later captured in signatures so that the signatures can treat this
    # ShardedVariable as one single variable.
    self._saving_variable = resource_variable_ops.UninitializedVariable(
        shape=self._shape,
        dtype=self._dtype,
        name=self._name,
        trainable=self._variables[0].trainable,
        synchronization=variables_lib.VariableSynchronization.NONE,
        aggregation=variables_lib.VariableAggregation.NONE,
    )

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the shard variables (always the real shards, not the saving
    # stand-in).
    """Return an iterable for accessing the underlying sharded variables."""
    return iter(self._variables)

  def __getitem__(self, slice_spec):
    # [FUNCTION DOC]
    # Indexing and slicing over the whole logical variable. Input: slice_spec, a single index or
    # a list/tuple of indices (integers, slices, ellipsis, new-axis markers), or a boolean mask
    # (boolean value, boolean tensor or boolean array). Boolean masks convert the whole variable
    # to a tensor and apply a boolean mask. A slice on axis 0 is decomposed into per-shard slices,
    # the results are concatenated (reversed for negative steps; an empty tensor if nothing is
    # selected). Ellipsis and new-axis are applied to every shard then concatenated. An integer
    # index (negative allowed) picks the owning shard and indexes it locally.
    # Returns a tensor. Raises a type error if a tensor is used as an index, an index error if an
    # integer index is out of bounds. Assignment through slicing is not supported.
    """Extracts the specified region as a Tensor from the sharded variable.

    The API contract is identical to `Tensor.__getitem__`. Assignment to the
    sliced range is not yet supported.

    Args:
      slice_spec: The arguments to __getitem__, specifying the global slicing of
        the sharded variable.

    Returns:
      The appropriate slice of tensor based on `slice_spec`.

    Raises:
      IndexError: If a slice index is out of bound.
      TypeError: If `spec_spec` contains Tensor.
    """

    # TODO(b/177482728): Support tensor input.
    # TODO(b/177482728): Support slice assign, similar to variable slice assign.

    if (
        isinstance(slice_spec, bool)
        or (
            isinstance(slice_spec, tensor_lib.Tensor)
            and slice_spec.dtype == dtypes.bool
        )
        or (isinstance(slice_spec, np.ndarray) and slice_spec.dtype == bool)
    ):
      tensor = _var_to_tensor(self)
      return array_ops.boolean_mask(tensor=tensor, mask=slice_spec)

    if not isinstance(slice_spec, (list, tuple)):
      slice_spec = (slice_spec,)

    s = slice_spec[0]
    if isinstance(s, slice):
      first_dim_slice_specs = self._decompose_slice_spec(s)
      values = []
      for i, var in enumerate(self._variables):
        if first_dim_slice_specs[i] is not None:
          all_dim_slice_spec = (first_dim_slice_specs[i],) + slice_spec[1:]
          values.append(var[all_dim_slice_spec])
      if s.step is not None and s.step < 0:
        values.reverse()
      if not values:
        return constant_op.constant(
            [], dtype=self._dtype, shape=((0,) + self._shape[1:])
        )
      return array_ops.concat(values, axis=0)
    elif s is Ellipsis:
      return array_ops.concat(
          [var[slice_spec] for var in self._variables], axis=0
      )
    elif s is array_ops.newaxis:
      return array_ops.concat(
          [var[slice_spec[1:]] for var in self._variables], axis=0
      )[array_ops.newaxis]
    else:
      if isinstance(s, tensor_lib.Tensor):
        raise TypeError(
            'ShardedVariable: using Tensor for indexing is not allowed.'
        )
      if s < 0:
        s += self._shape[0]
      if s < 0 or s >= self._shape[0]:
        raise IndexError(
            f'ShardedVariable: slice index {s} of dimension 0 out of bounds.'
        )
      for i in range(len(self._variables)):
        if i == len(self._variables) - 1 or (
            s >= self._var_offsets[i][0] and s < self._var_offsets[i + 1][0]
        ):
          return self._variables[i][
              (s - self._var_offsets[i][0],) + slice_spec[1:]
          ]

  def _decompose_slice_spec(self, slice_spec):
    # [FUNCTION DOC]
    # HELPER: used by the indexing method to turn an axis-0 slice into per-shard slices.
    # Input: a slice object over the first axis (start, stop, step may be missing or negative,
    # but not tensors). Normalizes negative or missing start and stop using the total first
    # dimension, then walks the shards (forward for positive step, backward for negative step)
    # tracking a cursor, and produces for each shard either a local slice or nothing when the
    # shard contributes no elements. Returns a list with one entry per shard, in shard order.
    # Raises a type error if any component is a tensor, and a value error if the step is zero.
    """Decompose a global slice_spec into a list of per-variable slice_spec.

    `ShardedVariable` only supports first dimension partitioning, thus
    `slice_spec` must be for first dimension.

    Args:
      slice_spec: A python `slice` object that specifies the global slicing.

    Returns:
      A list of python `slice` objects or None specifying the local slicing for
      each component variable. None means no slicing.

    For example, given component variables:
      v0 = [0, 1, 2]
      v1 = [3, 4, 5]
      v2 = [6, 7, 8, 9]

    If `slice_spec` is slice(start=None, stop=None, step=None), we will have:
      v0[returned[0]] = [0, 1, 2]
      v1[returned[1]] = [3, 4, 5]
      v2[returned[2]] = [6, 7, 8, 9]
    If `slice_spec` is slice(start=2, stop=8, step=3), we will have:
      v0[returned[0]] = [2]
      v1[returned[1]] = [5]
      returned[2] == None
    If `slice_spec` is slice(start=9, stop=3, step=-2), we will have:
      returned[0] == None
      v1[returned[1]] = [5]
      v2[returned[2]] = [9, 7]
    """
    if (
        isinstance(slice_spec.start, tensor_lib.Tensor)
        or isinstance(slice_spec.stop, tensor_lib.Tensor)
        or isinstance(slice_spec.step, tensor_lib.Tensor)
    ):
      raise TypeError(
          'ShardedVariable: using Tensor in slice_spec is not allowed. Please '
          'file a feature request with the TensorFlow team.'
      )

    result = []
    # Normalize start, end and stop.
    slice_step = slice_spec.step if slice_spec.step is not None else 1
    if slice_step == 0:
      raise ValueError('slice step cannot be zero')
    slice_start = slice_spec.start
    if slice_start is None:
      slice_start = 0 if slice_step > 0 else self._shape[0] - 1
    elif slice_start < 0:
      slice_start += self._shape[0]
    slice_end = slice_spec.stop
    if slice_end is None:
      # After the normalization, we no longer interpret negative index, thus
      # "-1" conceptually refers to the element before the first one, which
      # doesn't exist. This is to ease the decomposition code.
      slice_end = self._shape[0] if slice_step > 0 else -1
    elif slice_end < 0:
      slice_end += self._shape[0]

    # To find the local slice_spec of each component variable, we start from
    # the start of the global slice, and iterate through each variable.
    # When iterating on a variable, we move the cursor (`cur`) to the first
    # index that falls into the variable's range, which becomes the start of
    # the variable's local slice_spec. The end of the local_spec is determined
    # by using whatever is smaller between global slice end and variable range
    # end.
    cur = slice_start
    if slice_step > 0:
      for i in range(len(self._var_offsets)):
        var_start = self._var_offsets[i][0]
        var_end = (
            self._var_offsets[i + 1][0]
            if i < len(self._var_offsets) - 1
            else self._shape[0]
        )
        if cur < var_start:
          cur += slice_step * int(math.ceil((var_start - cur) / slice_step))
        if cur >= var_end or cur >= slice_end:
          result.append(None)
        else:
          start = cur - var_start
          end = min(slice_end, var_end) - var_start
          result.append(slice(start, end, slice_step))
    else:  # slice_step < 0
      for i in range(len(self._var_offsets) - 1, -1, -1):
        var_start = self._var_offsets[i][0]
        var_end = (
            self._var_offsets[i + 1][0]
            if i < len(self._var_offsets) - 1
            else self._shape[0]
        )
        if cur >= var_end:
          cur += slice_step * int(math.ceil((var_end - cur - 1) / slice_step))
        if cur < var_start or cur <= slice_end:
          result.append(None)
        else:
          start = cur - var_start
          if slice_end >= var_start:
            end = slice_end - var_start
          else:
            end = None  # no explicit end: slice until hitting the boundary.
          result.append(slice(start, end, slice_step))

      result.reverse()

    return result

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Property returning a sharded-variable spec built from one variable spec (shape and dtype)
    # per shard.
    return ShardedVariableSpec(
        *(
            resource_variable_ops.VariableSpec(v.shape, v.dtype)
            for v in self._variables
        )
    )

  @property
  def variables(self):
    # [FUNCTION DOC]
    # Property returning the shard variables. When running inside a save context, returns a
    # one-element list holding the full-shape stand-in variable instead, so saved signatures see
    # a single variable.
    """The list of `Variable`s that make up the shards of this object."""
    if save_context.in_save_context():
      return [self._saving_variable]
    return self._variables

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the container name (used for checkpointing).
    """The name of this object. Used for checkpointing."""
    return self._name

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the common dtype of all shards.
    """The dtype of all `Variable`s in this object."""
    return self._dtype

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property returning the combined shape (shard sizes summed along axis 0).
    """The overall shape, combining all shards along axis `0`."""
    return self._shape

  def assign(self, value, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Assigns a full-shape value to the variable by slicing it per shard (using stored offsets)
    # and assigning each slice to its shard. The use_locking, name and read_value arguments are
    # accepted but ignored. Returns this container (not an updated value).
    for i, v in enumerate(self._variables):
      v.assign(array_ops.slice(value, self._var_offsets[i], v.shape.as_list()))
    return self

  def assign_add(self, delta, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Adds a full-shape delta by slicing it per shard and calling add-assign on each shard. The
    # use_locking, name and read_value arguments are ignored. Returns this container.
    for i, v in enumerate(self._variables):
      v.assign_add(
          array_ops.slice(delta, self._var_offsets[i], v.shape.as_list())
      )
    return self

  def assign_sub(self, delta, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Subtracts a full-shape delta by slicing it per shard and calling subtract-assign on each
    # shard. The use_locking, name and read_value arguments are ignored. Returns this container.
    for i, v in enumerate(self._variables):
      v.assign_sub(
          array_ops.slice(delta, self._var_offsets[i], v.shape.as_list())
      )
    return self

  def _decompose_indices(self, indices):
    # [FUNCTION DOC]
    # HELPER: used by the scatter and sparse read operations to route global row indices.
    # Input: a one-dimensional integer tensor of global indices. Requires the shards to follow
    # "div" sharding (sizes differ by at most one, larger shards first); raises a value error if
    # indices is not rank one and a not-implemented error if the sharding is not of that form.
    # Computes which shard each index belongs to and its local index, then groups local indices
    # per shard with a dynamic partition. Returns a pair: list of per-shard index tensors, and
    # the per-index shard assignment tensor (int32).
    """Decompose a global 1D indices into a list of per-variable indices."""
    if indices.shape.rank != 1:
      raise ValueError(
          'ShardedVariable: indices must be 1D Tensor for sparse operations. '
          f'Received shape: {indices.shape}'
      )

    base = self._shape[0] // len(self._variables)
    extra = self._shape[0] % len(self._variables)

    # Assert that sharding conforms to "div" sharding
    expect_first_dim = [base] * len(self._variables)
    for i in range(extra):
      expect_first_dim[i] = expect_first_dim[i] + 1
    actual_first_dim = [v.shape.as_list()[0] for v in self._variables]
    if expect_first_dim != actual_first_dim:
      raise NotImplementedError(
          'scatter_xxx ops are not supported in ShardedVariable that does not '
          'conform to "div" sharding'
      )

    # For index that falls into the partition that has extra 1, assignment is
    # `index // (base + 1)` (no less than `(indices - extra) // base`)
    # For index that falls into the partition that doesn't has extra 1,
    # assignment is `(indices - extra) // base` (no less than
    # `indices // (base + 1)`)
    #
    # Example:
    #   base = 10, extra = 2, partitions: [0, 11), [11, 22), [22, 32)
    #   index = 10 -> partition_assignment = 0
    #   index = 22 -> partition_assignment = 2
    partition_assignments = math_ops.maximum(
        indices // (base + 1), (indices - extra) // base
    )
    local_indices = array_ops.where(
        partition_assignments < extra,
        indices % (base + 1),
        (indices - extra) % base,
    )
    # For whatever reason `dynamic_partition` only supports int32
    partition_assignments = math_ops.cast(partition_assignments, dtypes.int32)
    per_var_indices = data_flow_ops.dynamic_partition(
        local_indices, partition_assignments, len(self._variables)
    )

    return per_var_indices, partition_assignments

  def _decompose_indexed_slices(self, indexed_slices):
    # [FUNCTION DOC]
    # HELPER: used by the scatter operations. Input: an indexed-slices object with global row
    # indices and values. Splits indices per shard, partitions the values the same way, and
    # returns a list of indexed-slices objects, one per shard, with local indices.
    """Decompose a global `IndexedSlices` into a list of per-variable ones."""
    per_var_indices, partition_assignments = self._decompose_indices(
        indexed_slices.indices
    )
    per_var_values = data_flow_ops.dynamic_partition(
        indexed_slices.values, partition_assignments, len(self._variables)
    )

    return [
        indexed_slices_lib.IndexedSlices(
            values=per_var_values[i], indices=per_var_indices[i]
        )
        for i in range(len(self._variables))
    ]

  # ==================== scatter ops implementations ======================== #

  def scatter_add(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse add of an indexed-slices delta (global row indices). Splits the delta per shard and
    # calls the shard's scatter add; when a name is given each shard op is named with the name
    # plus a part suffix. use_locking is ignored. Returns this container. May raise errors from
    # index decomposition (non rank-one indices, non-div sharding).
    """Implements tf.Variable.scatter_add."""
    per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
    for i, v in enumerate(self._variables):
      new_name = None
      if name is not None:
        new_name = '{}/part_{}'.format(name, i)
      v.scatter_add(per_var_sparse_delta[i], name=new_name)
    return self

  def scatter_div(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse divide by an indexed-slices delta (global row indices). Splits the delta per shard
    # and calls each shard's scatter divide, naming ops with a part suffix when a name is given.
    # use_locking is ignored. Returns this container.
    """Implements tf.Variable.scatter_div."""
    per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
    for i, v in enumerate(self._variables):
      new_name = None
      if name is not None:
        new_name = '{}/part_{}'.format(name, i)
      v.scatter_div(per_var_sparse_delta[i], name=new_name)
    return self

  def scatter_max(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse element-wise maximum with an indexed-slices delta (global row indices). Splits the
    # delta per shard and calls each shard's scatter max, naming ops with a part suffix when a
    # name is given. use_locking is ignored. Returns this container.
    """Implements tf.Variable.scatter_max."""
    per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
    for i, v in enumerate(self._variables):
      new_name = None
      if name is not None:
        new_name = '{}/part_{}'.format(name, i)
      v.scatter_max(per_var_sparse_delta[i], name=new_name)
    return self

  def scatter_min(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse element-wise minimum with an indexed-slices delta (global row indices). Splits the
    # delta per shard and calls each shard's scatter min, naming ops with a part suffix when a
    # name is given. use_locking is ignored. Returns this container.
    """Implements tf.Variable.scatter_min."""
    per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
    for i, v in enumerate(self._variables):
      new_name = None
      if name is not None:
        new_name = '{}/part_{}'.format(name, i)
      v.scatter_min(per_var_sparse_delta[i], name=new_name)
    return self

  def scatter_mul(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse multiply by an indexed-slices delta (global row indices). Splits the delta per
    # shard and calls each shard's scatter multiply, naming ops with a part suffix when a name
    # is given. use_locking is ignored. Returns this container.
    """Implements tf.Variable.scatter_mul."""
    per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
    for i, v in enumerate(self._variables):
      new_name = None
      if name is not None:
        new_name = '{}/part_{}'.format(name, i)
      v.scatter_mul(per_var_sparse_delta[i], name=new_name)
    return self

  def scatter_sub(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse subtract of an indexed-slices delta (global row indices). Splits the delta per
    # shard and calls each shard's scatter subtract, naming ops with a part suffix when a name
    # is given. use_locking is ignored. Returns this container.
    """Implements tf.Variable.scatter_sub."""
    per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
    for i, v in enumerate(self._variables):
      new_name = None
      if name is not None:
        new_name = '{}/part_{}'.format(name, i)
      v.scatter_sub(per_var_sparse_delta[i], name=new_name)
    return self

  def scatter_update(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Sparse overwrite of rows from an indexed-slices delta (global row indices). Splits the
    # delta per shard and calls each shard's scatter update, naming ops with a part suffix when a
    # name is given. use_locking is ignored. Returns this container.
    """Implements tf.Variable.scatter_update."""
    per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
    for i, v in enumerate(self._variables):
      new_name = None
      if name is not None:
        new_name = '{}/part_{}'.format(name, i)
      v.scatter_update(per_var_sparse_delta[i], name=new_name)
    return self

  def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
    # [FUNCTION DOC]
    # Batched sparse overwrite from an indexed-slices delta (global row indices). Splits the
    # delta per shard and calls each shard's batch scatter update, naming ops with a part suffix
    # when a name is given. use_locking is ignored. Returns this container.
    """Implements tf.Variable.batch_scatter_update."""
    per_var_sparse_delta = self._decompose_indexed_slices(sparse_delta)
    for i, v in enumerate(self._variables):
      new_name = None
      if name is not None:
        new_name = '{}/part_{}'.format(name, i)
      v.batch_scatter_update(per_var_sparse_delta[i], name=new_name)
    return self

  # ================== scatter ops implementations END ====================== #

  def sparse_read(self, indices, name=None):
    # [FUNCTION DOC]
    # Reads rows by global index. Input: one-dimensional integer indices and optional name
    # (shard ops get a part suffix). Splits indices per shard, reads each shard sparsely and
    # concatenates the results along axis 0 in shard order (note: not restored to the original
    # index order unless indices are grouped by shard). Returns a tensor.
    """Implements tf.Variable.sparse_read."""
    per_var_indices, _ = self._decompose_indices(indices)
    result = []
    for i, v in enumerate(self._variables):
      new_name = None
      if name is not None:
        new_name = '{}/part_{}'.format(name, i)
      result.append(v.sparse_read(per_var_indices[i], name=new_name))
    return array_ops.concat(result, axis=0)

  def _gather_saveables_for_checkpoint(self):
    # [FUNCTION DOC]
    # Checkpoint hook. Returns a mapping from the standard variable-value key to a factory that
    # creates the saveables. The mapping has a single entry.
    """Return a `Saveable` for each shard. See `Trackable`."""

    def _saveable_factory(name=self.name):
      # [FUNCTION DOC]
      # HELPER: factory for checkpoint saveables. Input: name (defaults to the container name).
      # For each shard builds save slice info (full name, full shape, running offset along axis
      # 0, shard shape) and wraps the shard in a resource-variable saveable with that slice
      # spec, so all shards are saved as slices of one larger variable. Returns the list.
      """Creates `SaveableObject`s for this `ShardedVariable`."""
      saveables = []
      dims = len(self._variables[0].shape)
      var_offset = [0 for _ in range(dims)]
      for v in self._variables:
        save_slice_info = variables_lib.Variable.SaveSliceInfo(
            full_name=self.name,
            full_shape=self.shape.as_list(),
            var_offset=copy.copy(var_offset),
            var_shape=v.shape.as_list(),
        )
        saveables.append(
            saveable_object_util.ResourceVariableSaveable(
                v, save_slice_info.spec, name
            )
        )
        var_offset[0] += int(v.shape[0])
      return saveables

    return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # Async checkpointing hook. Intentionally does nothing in the mixin (subclasses that need
    # CPU copies implement it). Input: the object map, which is left untouched.
    """For implementing `Trackable` async checkpointing."""
    # This is not implemented in the ShardedVariableMixin class because multiple
    # classes inherit from it. If your class contains values that should be
    # copied to CPU for async checkpointing, please implement this in the class
    # definition.

  def _export_to_saved_model_graph(
      self, object_map, tensor_map, options, **kwargs
  ):
    # [FUNCTION DOC]
    # SavedModel export hook. Exports every shard and the saving stand-in variable via their own
    # export hooks, collecting the resulting resources into one list. Side effect: records in the
    # object map a new single-shard sharded variable wrapping the exported stand-in, under the
    # same name, as the replacement for this object. Returns the combined resource list.
    """For implementing `Trackable` SavedModel export."""
    resource_list = []
    for v in self._variables + [self._saving_variable]:
      resource_list.extend(
          v._export_to_saved_model_graph(  # pylint:disable=protected-access
              object_map, tensor_map, options, **kwargs
          )
      )
    object_map[self] = ShardedVariable(
        [object_map[self._saving_variable]], name=self.name
    )
    return resource_list

  @property
  def _unique_id(self):
    # [FUNCTION DOC]
    # Property: unique id for checkpoint tracking, derived from the first variable's id with the
    # first-part marker replaced by a sharded marker.
    # String-replace to ensure uniqueness for checkpoint tracking
    return self.variables[0]._unique_id.replace('part_0', 'sharded')  # pylint: disable=protected-access

  @property
  def _distribute_strategy(self):
    # [FUNCTION DOC]
    # Property returning the distribution strategy of the first variable.
    return self.variables[0]._distribute_strategy  # pylint: disable=protected-access

  @property
  def _shared_name(self):
    # [FUNCTION DOC]
    # Property returning the container name as the shared name.
    return self._name

  @property
  def is_sharded_variable(self):
    # [FUNCTION DOC]
    # Property marker: always true, identifying this object as a sharded variable.
    return True

  def numpy(self):
    # [FUNCTION DOC]
    # Converts the whole variable to a single tensor (concatenating shards) and returns its
    # values as an array with the same shape and dtype. Requires eager execution.
    """Copies the values in this ShardedVariable to a NumPy array.

    First converts to a single Tensor using the registered conversion function,
    which concatenates the shards, then uses Tensor.numpy() to convert to
    a NumPy array.

    Returns:
      A NumPy array of the same shape and dtype.
    """
    return _var_to_tensor(self).numpy()


@tf_export('__internal__.distribute.ShardedVariable', v1=[])
class ShardedVariable(ShardedVariableMixin, composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # Public sharded variable container: the mixin behavior combined with composite tensor
  # support, so it can pass through tf.function boundaries as its shard variables. Adds the
  # spec, operator overloading (operators convert to a tensor by concatenation), resource
  # variable compatibility and object-proto writing, and async checkpoint copying. Created from
  # a list of shard variables and a name; exported as an internal distribute API.
  """A container for `Variables` that should be treated as shards.

  Variables that are too large to fit on a single device (e.g., large
  embeddings)
  may need to be sharded over multiple devices. This class maintains a list of
  smaller variables that can be independently stored on separate devices (eg,
  multiple parameter servers), and saves and restores those variables as if they
  were a single larger variable.

  Objects of this class can be saved with a given number of shards and then
  restored from a checkpoint into a different number of shards.

  Objects of this class can be saved to SavedModel format using
  `tf.saved_model.save`. The SavedModel can be used by programs like TF serving
  APIs. It is not yet supported to load the SavedModel with
  `tf.saved_model.load`.

  Since `ShardedVariable` can be saved and then restored to different number of
  shards depending on the restore environments, for example, TF serving APIs
  would restore to one shard for serving efficiency, when using
  `ShardedVariable` in a tf.function, one should generally not assume it has the
  same number of shards across save and load.

  Sharding is only supported along the first dimension.

  >>> class Model(tf.Module):
  ...   def __init__(self):
  ...     self.sharded_variable = ShardedVariable([
  ...       tf.Variable([3.0], dtype=tf.float32),
  ...       tf.Variable([2.0], dtype=tf.float32)
  ...     ])
  ...
  ...   @tf.function(input_signature=[tf.TensorSpec([], dtype=tf.int32)])
  ...   def fn(self, x):
  ...     return tf.nn.embedding_lookup(self.sharded_variable.variables, x)
  ...
  ...   @tf.function(input_signature=[tf.TensorSpec([], dtype=tf.int32)])
  ...   def serve_fn(self, x):
  ...     return tf.nn.embedding_lookup(self.sharded_variable.variables, x)
  >>>
  >>> model = Model()
  >>> model.fn(1).numpy()
  2.0
  >>> tf.saved_model.save(model, export_dir='/tmp/saved_model',
  ...   signatures=model.serve_fn)
  """

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Property returning a sharded-variable spec with one variable spec per shard. Duplicates the
    # mixin's identical property (same behavior, different formatting).
    return ShardedVariableSpec(
        *(resource_variable_ops.VariableSpec(v.shape, v.dtype)
          for v in self._variables))

  @classmethod
  def _overload_all_operators(cls):
    # [FUNCTION DOC]
    # HELPER: class-level setup run once at import. Installs a tensor-delegating overload on this
    # class for every overloadable tensor operator except item access (which the mixin
    # implements itself). Returns nothing.
    """Register overloads for all operators."""
    for operator in tensor_lib.Tensor.OVERLOADABLE_OPERATORS:
      if operator == '__getitem__':
        continue

      cls._overload_operator(operator)

  @classmethod
  def _overload_operator(cls, operator):
    # [FUNCTION DOC]
    # HELPER: for _overload_all_operators. Input: operator, the name of a tensor operator method.
    # Looks up the tensor class's implementation and sets on this class a method that converts
    # the sharded variable to a tensor and applies that implementation with any extra arguments.
    """Delegate an operator overload to `tensor_lib.Tensor`."""
    tensor_operator = getattr(tensor_lib.Tensor, operator)

    def _operator(v, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: the installed operator method. Converts v to a tensor by concatenating its shards
      # and applies the tensor operator with the remaining arguments; returns its result.
      return tensor_operator(_var_to_tensor(v), *args, **kwargs)

    setattr(cls, operator, _operator)

  def __tf_experimental_restore_capture__(
      self, concrete_function, internal_capture
  ):
    # [FUNCTION DOC]
    # Hook for restoring captures of loaded functions. Always returns nothing, deliberately
    # skipping restoration because the layer is recreated when a Keras model is loaded.
    # Loading SavedModels containing sharded variables is not yet supported.
    # Avoid restoring captures for functions that use ShardedVariable - the
    # layer will be recreated during Keras model loading
    # TODO(jmullenbach): support loading models with ShardedVariables using
    # tf.saved_model.load
    return None

  def _should_act_as_resource_variable(self):
    # [FUNCTION DOC]
    # Always returns true so resource-variable type checks accept this object.
    """Pass resource_variable_ops.is_resource_variable check."""
    return True

  def _write_object_proto(self, proto, options):
    # [FUNCTION DOC]
    # SavedModel hook. Fills the given object proto as for a resource variable, using the saving
    # stand-in variable, without enforcing naming. Inputs: proto (message to fill) and options
    # (save options). Returns nothing.
    resource_variable_ops.write_object_proto_for_resource_variable(
        self._saving_variable, proto, options, enforce_naming=False
    )

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # Async checkpointing hook. Input: object_map, mapping from original objects to their CPU
    # copies. If this object is already mapped, refreshes the values of every shard's copy.
    # Otherwise copies each shard (creating its CPU copy and copying its value), builds a new
    # sharded variable from the copies with the same name, and records it in the map.
    # Mutates the map; returns nothing.
    """For implementing `Trackable` async checkpointing."""
    if self in object_map:
      # If populated already, simply loop through sub-variables to copy values.
      for v in self._variables:
        v._copy_trackable_to_cpu(object_map)  # pylint: disable=protected-access
    else:
      # If not populated, populate first, then copy.
      copied_vars = []
      for v in self._variables:
        # This step will both instantiate `v`'s CPU copy and copy its value.
        v._copy_trackable_to_cpu(object_map)  # pylint: disable=protected-access
        copied_vars.append(object_map[v])
      new_var = ShardedVariable(copied_vars, name=self.name)
      object_map[self] = new_var


def _var_to_tensor(var, dtype=None, name=None, as_ref=False):
  # [FUNCTION DOC]
  # HELPER: tensor conversion function registered for sharded variables, also used by indexing,
  # numpy and operator overloads. Inputs: var (sharded variable), dtype (optional requested
  # dtype; must be compatible), name (ignored), as_ref (must be false). Returns the concatenation
  # of the shard variables along axis 0. Raises a value error for an incompatible dtype, a
  # not-implemented error if a reference is requested, and a type error if called inside an op
  # name scope containing the embedding lookup name (so lookup dispatch can take over; does not
  # work in eager mode).
  """Converts a `ShardedVariable` to a `Tensor`."""
  del name
  if dtype is not None and not dtype.is_compatible_with(var.dtype):
    raise ValueError(
        'Incompatible type conversion requested to type {!r} for variable '
        'of type {!r}'.format(dtype.name, var.dtype.name)
    )
  if as_ref:
    raise NotImplementedError(
        "ShardedVariable doesn't support being used as a reference."
    )
  # We use op dispatch mechanism to override embedding_lookup ops when called
  # with ShardedVariable. This requires embedding_lookup ops to raise TypeError
  # when called with ShardedVariable. However since ShardedVariable can be
  # converted to a tensor via concat, embedding_lookup ops would silently
  # do the conversion and never raise a TypeError. To be able to properly
  # raise a TypeError, namescope is used to detect if this method is called
  # within a embedding_lookup op.
  # NOTE: This doesn't work in eager mode since op namescope is always cleared
  # in eager. This also breaks if user sets the name of embedding_lookup op
  # with something that doesn't contain str "embedding_lookup".
  #
  # TODO(chenkai): Find a more robust way to do this, which should not rely
  # on namescope.
  if 'embedding_lookup' in ops.get_name_scope():
    raise TypeError(
        'Converting ShardedVariable to tensor in embedding lookup'
        ' ops is disallowed.'
    )
  return array_ops.concat(var.variables, axis=0)


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.
tensor_conversion_registry.register_tensor_conversion_function(
    ShardedVariable, _var_to_tensor
)

ShardedVariable._overload_all_operators()  # pylint: disable=protected-access


# Override the behavior of embedding_lookup(sharded_variable, ...)
@dispatch.dispatch_for_types(embedding_ops.embedding_lookup, ShardedVariable)
def embedding_lookup(
    params,
    ids,
    partition_strategy='mod',
    name=None,
    validate_indices=True,
    max_norm=None,
):
  # [FUNCTION DOC]
  # Dispatch override of the embedding lookup API for sharded variables. Inputs: params (a
  # sharded variable, or a list whose first element is used), ids, partition strategy (default
  # "mod"), name, validate_indices flag (default true) and max_norm (optional). Calls the
  # regular lookup with the list of shard variables and returns its result.
  if isinstance(params, list):
    params = params[0]
  return embedding_ops.embedding_lookup(
      params.variables,
      ids,
      partition_strategy,
      name,
      validate_indices,
      max_norm,
  )


# Separately override safe_embedding_lookup_sparse, to avoid conversion of
# ShardedVariable to tensor.
@dispatch.dispatch_for_api(embedding_ops.safe_embedding_lookup_sparse)
def safe_embedding_lookup_sparse(
    embedding_weights: ShardedVariable,
    sparse_ids,
    sparse_weights=None,
    combiner='mean',
    default_id=None,
    name=None,
    partition_strategy='div',
    max_norm=None,
    allow_fast_lookup=False,
):
  # [FUNCTION DOC]
  # Dispatch override of the safe sparse embedding lookup API, used when the weights argument is
  # a sharded variable. Passes the list of shard variables plus all other arguments (sparse ids,
  # optional sparse weights, combiner default "mean", default id, name, partition strategy
  # default "div", max_norm, allow_fast_lookup flag) unchanged to the regular implementation and
  # returns its result. Avoids converting the sharded variable to a single tensor.
  """Pass the individual shard variables as a list."""
  return embedding_ops.safe_embedding_lookup_sparse(
      embedding_weights.variables,
      sparse_ids,
      sparse_weights=sparse_weights,
      combiner=combiner,
      default_id=default_id,
      name=name,
      partition_strategy=partition_strategy,
      max_norm=max_norm,
      allow_fast_lookup=allow_fast_lookup,
  )
