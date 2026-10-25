# [MODULE DOC]
# Dynamic (tensor-valued) shapes for ragged tensors and broadcasting built on them. Provides the
# RaggedTensorDynamicShape class, broadcast_dynamic_shape for combining two such shapes, and
# broadcast_to for tiling a (possibly ragged) tensor to a ragged shape, plus private helpers.
# Used by ragged elementwise ops and ragged broadcasting. No import-time side effects.
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
"""Shapes & broadcasting for RaggedTensors."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util


class RaggedTensorDynamicShape:
  # [CLASS DOC]
  # Immutable description of the shape of a possibly ragged tensor, stored as tensors. Holds a
  # tuple of partitioned dimension sizes (scalar for uniform dimensions, vector of slice lengths
  # for ragged ones) and a vector of inner dimension sizes, all cast to one integer dtype
  # (int32 or int64). Created via the constructor, from_dim_sizes or from_tensor. Offers rank and
  # size queries and the broadcasting operations used by broadcast_dynamic_shape and broadcast_to.
  """A collection of tensors encoding the shape of a potentially ragged tensor.

  Each `RaggedTensorDynamicShape` consists of an ordered list of dimension
  sizes.  There are two dimension types:

    * "Uniform dimensions" are dimensions where all slices have the same
      length.  `RaggedTensorDynamicShape` records the size of each uniform
      dimension using a single scalar integer.

    * "Ragged dimensions" are dimensions whose slices may have different
      lengths.  `RaggedTensorDynamicShape` records the size of each ragged
      dimension using an integer vector containing the slice lengths for all
      the slices across that dimension.

  Furthermore, there are two ways a dimension might be encoded:

    * "Partitioned dimensions" are dimensions that are encoded using a
      `RaggedTensor`'s `nested_row_splits`.  The outermostmost partitioned
      dimension must be uniform, and the innermost partitioned dimension must
      be ragged.

    * "Inner dimensions" are dimensions that are encoded using a
      `RaggedTensor`'s `flat_values`.  Inner dimensions are always uniform.

  The sizes of partitioned dimensions are recorded using `partitioned_dim_sizes`
  and `inner_dim_sizes`:

    * `partitioned_dim_sizes` is a list of tensors (one for each partitioned
      dimension).

      * For uniform dimensions, the tensor is an integer scalar specifying the
        size of all slices across that dimension.
      * For ragged dimensions, the tensor is an integer vector specifying the
        size of each slice across that dimension.

    * `inner_dim_sizes` is a single integer vector, where each element
      specifies the size of a single inner dimension.

  Examples:

  Tensor                         | Ragged | Partitioned Dim Sizes  | Inner Dim
                                 : Rank   :                        : Sizes
  ------------------------------ | ------ | ---------------------- | ----------
  `[[1, 2, 3], [4, 5, 6]]`       |      0 |                        | `2, 3`
  `[[1, 2], [], [3, 4, 5]]`      |      1 | `3, (2, 0, 3)`         |
  `[[[1, 2], [3, 4]], [[5, 6]]]` |      1 | `2, (2, 1)`            | 2
  `[[[1, 2], [3]], [[4, 5]]]`    |      2 | `2, (2, 1), (2, 1, 2)` |
  """

  def __init__(self, partitioned_dim_sizes, inner_dim_sizes,
               dim_size_dtype=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: partitioned_dim_sizes (list or tuple of scalar or vector integer
    # tensors), inner_dim_sizes (1-D integer tensor), optional dim_size_dtype. Converts all to
    # tensors, validates ranks, and casts everything to one dtype: the given one, else the common
    # dtype of the ragged (vector) sizes, else int64. If the dtypes differ it casts to int64 only
    # when automatic partition dtype casting is enabled. Raises ValueError if a rank is unknown,
    # the outermost partitioned dimension is ragged, the innermost is uniform, inner sizes are
    # not rank 1, or dtypes mismatch without auto-cast. An assertion error occurs if the first
    # argument is not a list or tuple. Stores the results on the instance; no other side effects.
    """Creates a RaggedTensorDynamicShape.

    Args:
      partitioned_dim_sizes: A `list` of 0-D or 1-D integer `Tensor`, one for
        each partitioned dimension.  If dimension `d` is uniform, then
        `partitioned_dim_sizes[d]` must be an integer scalar, specifying the
        size of all slices across dimension `d`.  If dimension `d` is ragged,
        then `partitioned_dim_sizes[d]` must be an integer vector, specifying
        the size of each slice across dimension `d`.
      inner_dim_sizes: A 1-D integer `Tensor`, whose length is equal to the
        number of inner dimensions.  `inner_dim_sizes[n]` is the size of all
        slices across the `n`th inner dimension (which is the
        `(len(partitioned_dim_sizes)+n)`th dimension in the overall tensor.
      dim_size_dtype: dtype for dimension sizes.  If not specified, then it
        is chosen based on the dtypes of `partitioned_dim_sizes` and
        `inner_dim_sizes`.
    """
    assert isinstance(partitioned_dim_sizes, (list, tuple))

    with ops.name_scope(None, 'RaggedTensorDynamicShape',
                        (partitioned_dim_sizes, inner_dim_sizes)):
      partitioned_dim_sizes = tuple(
          ops.convert_to_tensor(size, name='partitioned_dimension_size_%d' % i)
          for (i, size) in enumerate(partitioned_dim_sizes))
      inner_dim_sizes = ops.convert_to_tensor(
          inner_dim_sizes, name='inner_dim_sizes')

      # Validate shapes.
      if partitioned_dim_sizes:
        for axis, dimension_size in enumerate(partitioned_dim_sizes):
          if dimension_size.shape.ndims is None:
            raise ValueError(
                'rank of partitioned_dim_sizes[%d] is unknown' % axis)
          dimension_size.shape.with_rank_at_most(1)
        if partitioned_dim_sizes[0].shape.ndims == 1:
          raise ValueError('outermost partitioned dimension must be uniform')
        if partitioned_dim_sizes[-1].shape.ndims == 0:
          raise ValueError('innermost partitioned dimension must be ragged')
      inner_dim_sizes.shape.assert_has_rank(1)

      # Convert dimension size tensors to a single dtype.
      if dim_size_dtype is None:
        dim_size_dtypes = set(
            p.dtype for p in partitioned_dim_sizes if p.shape.ndims == 1)
        if not dim_size_dtypes:
          dim_size_dtype = dtypes.int64
        elif len(dim_size_dtypes) == 1:
          dim_size_dtype = dim_size_dtypes.pop()
        else:
          if not ragged_config.auto_cast_partition_dtype():
            raise ValueError('partitioned_dim_sizes must have matching dtypes')
          dim_size_dtype = dtypes.int64
      partitioned_dim_sizes = tuple(math_ops.cast(p, dim_size_dtype)
                                    for p in partitioned_dim_sizes)
      inner_dim_sizes = math_ops.cast(inner_dim_sizes, dim_size_dtype)

      self._partitioned_dim_sizes = partitioned_dim_sizes
      self._inner_dim_sizes = inner_dim_sizes

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a debugging string naming the class with the partitioned and inner dimension sizes.
    return ('RaggedTensorDynamicShape'
            '(partitioned_dim_sizes=%r, inner_dim_sizes=%r)' %
            (self._partitioned_dim_sizes, self._inner_dim_sizes))

  @staticmethod
  def from_dim_sizes(dim_sizes):
    # [FUNCTION DOC]
    # Static factory. Input: a list of integer scalars (uniform dimensions) or vectors (ragged
    # dimensions), one per dimension; converted to tensors preferring int64. Dimensions up to and
    # including the last ragged one become partitioned sizes, the rest become inner sizes.
    # Returns a new shape object. Raises ValueError if any entry is neither a scalar nor a vector,
    # plus any constructor errors (e.g. ragged outermost dimension).
    """Constructs a ragged shape from a list of dimension sizes.

    This list contains a single tensor for each dimension, where the tensor
    is a scalar if the dimension is uniform, or a vector if the dimension is
    ragged.

    Args:
      dim_sizes: List of int32 or int64 scalars or vectors.

    Returns:
      A RaggedTensorDynamicShape.
    """
    with ops.name_scope(None, 'RaggedTensorDynamicShapeFromDimensionSizes',
                        [dim_sizes]):
      dim_sizes = tuple(
          ops.convert_to_tensor(size, preferred_dtype=dtypes.int64,
                                name='dim_sizes') for size in dim_sizes)
      # Split the dimensions into partitioned & inner dimensions.
      inner_split = 0
      for dim, dim_size in enumerate(dim_sizes):
        if dim_size.shape.ndims == 1:
          inner_split = dim + 1
        elif dim_size.shape.ndims != 0:
          raise ValueError('Each dim_size must be a scalar or a vector')
      return RaggedTensorDynamicShape(dim_sizes[:inner_split],
                                      dim_sizes[inner_split:])

  @classmethod
  def from_tensor(cls, rt_input, dim_size_dtype=None):
    # [FUNCTION DOC]
    # Class factory. Input: a dense or ragged tensor (converted if needed) and an optional dtype
    # for the sizes. For a dense tensor the result has no partitioned dimensions and its full
    # dynamic shape as inner sizes. For a ragged tensor the partitioned sizes are the row count
    # followed by each nested row-lengths vector, and the inner sizes are the shape of the
    # flat values without its first dimension. Returns the shape object; adds ops to the graph.
    """Constructs a ragged shape for a potentially ragged tensor."""
    with ops.name_scope(None, 'RaggedTensorDynamicShapeFromTensor', [rt_input]):
      rt_input = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt_input)
      if not ragged_tensor.is_ragged(rt_input):
        return cls([], array_ops.shape(rt_input), dim_size_dtype=dim_size_dtype)
      else:
        partitioned_dim_sizes = (
            (rt_input.nrows(),) + rt_input.nested_row_lengths())
        return RaggedTensorDynamicShape(
            partitioned_dim_sizes,
            array_ops.shape(rt_input.flat_values)[1:],
            dim_size_dtype=dim_size_dtype)

  def dimension_size(self, axis):
    # [FUNCTION DOC]
    # KNOWN ISSUE: negative or out-of-range axes are not validated here; negative values index
    # from the end of the inner sizes or fail with an indexing error.
    # Returns the size tensor for the given axis: an entry of the partitioned sizes (scalar or
    # vector) or of the inner sizes (scalar). Input axis must be a Python integer, otherwise a
    # type error is raised.
    """Returns the size of slices across the specified dimension."""
    if not isinstance(axis, int):
      raise TypeError('axis must be an integer')
    partitioned_ndims = len(self._partitioned_dim_sizes)
    if axis < partitioned_ndims:
      return self._partitioned_dim_sizes[axis]
    else:
      return self._inner_dim_sizes[axis - partitioned_ndims]

  def is_ragged(self, axis):
    # [FUNCTION DOC]
    # Returns a Python boolean: true when the axis is a partitioned dimension greater than zero
    # whose size is a vector. Raises a type error if axis is not an integer and a value error for
    # negative axes or axes at or beyond a known rank. Axis zero is never ragged.
    """Returns true if the indicated dimension is ragged."""
    if not isinstance(axis, int):
      raise TypeError('axis must be an integer')
    rank = self.rank
    if axis < 0:
      raise ValueError('Negative axis values are not supported')
    elif rank is not None and axis >= rank:
      raise ValueError('Expected axis=%s < rank=%s' % (axis, rank))
    else:
      return (axis > 0 and axis < len(self._partitioned_dim_sizes) and
              self._partitioned_dim_sizes[axis].shape.ndims == 1)

  @property
  def rank(self):
    # [FUNCTION DOC]
    # Property. Returns the number of partitioned dimensions plus the statically known number of
    # inner dimensions, or nothing (null) when the inner sizes vector has unknown length.
    """The number of dimensions in this shape, or None if unknown."""
    inner_ndims = tensor_shape.dimension_value(self._inner_dim_sizes.shape[0])
    if inner_ndims is None:
      return None
    else:
      return len(self._partitioned_dim_sizes) + inner_ndims

  @property
  def partitioned_dim_sizes(self):
    # [FUNCTION DOC]
    # Property returning the stored tuple of partitioned dimension size tensors.
    # KNOWN ISSUE: the docstring says a list, but the stored value is a tuple.
    """The partitioned dimension sizes for this shape.

    Returns:
      A `list` of 0-D or 1-D integer `Tensor`.
    """
    return self._partitioned_dim_sizes

  @property
  def inner_dim_sizes(self):
    # [FUNCTION DOC]
    # Property returning the stored 1-D integer tensor of inner dimension sizes.
    """The inner dimension sizes for this shape.

    Returns:
      A 1-D integer `Tensor`.
    """
    return self._inner_dim_sizes

  @property
  def num_partitioned_dimensions(self):
    # [FUNCTION DOC]
    # Property returning the number of partitioned dimensions as a Python integer.
    """The number of partitioned dimensions in this shape."""
    return len(self._partitioned_dim_sizes)

  @property
  def num_inner_dimensions(self):
    # [FUNCTION DOC]
    # Property returning the static length of the inner sizes vector, or null if not known.
    """The number of inner dimensions, or `None` if not statically known."""
    return tensor_shape.dimension_value(self._inner_dim_sizes.shape[0])

  @property
  def dim_size_dtype(self):
    # [FUNCTION DOC]
    # Property returning the integer dtype shared by all dimension size tensors (taken from the
    # inner sizes vector).
    """DType used by this shape for dimension sizes."""
    return self._inner_dim_sizes.dtype

  def broadcast_to_rank(self, rank):
    # [FUNCTION DOC]
    # Input: target rank (integer). Returns this shape itself if the rank already matches,
    # otherwise a new shape with leading size-one uniform dimensions added (as extra partitioned
    # sizes if any exist, else prepended to the inner sizes). Raises ValueError if the current
    # rank is unknown or larger than the requested rank.
    """Adds leading size-1 dimensions to broadcast `self` to the given rank.

    E.g., if `shape1` is `[3, (D2), 4]`, then `shape1.broadcast_to_rank(5)`
    is `[1, 1, 3, (D2), 4]`.

    Args:
      rank: The rank for the returned shape.

    Returns:
      A RaggedTensorDynamicShape with `rank` dimensions, whose inner dimensions
      have the same size as `self` and whose outer dimensions have size `1`.

    Raises:
      ValueError: If `self.rank` is unknown or greater than `rank`.
    """
    if self.rank is None:
      raise ValueError('Unable to broadcast: self.rank is unknown')
    dims_to_add = rank - self.rank
    if dims_to_add < 0:
      raise ValueError('Unable to broadcast: rank=%d must be greater than '
                       'self.rank=%d.' % (rank, self.rank))
    elif dims_to_add == 0:
      return self
    elif self._partitioned_dim_sizes:
      partitioned_dims = (1,) * dims_to_add + self._partitioned_dim_sizes
      return RaggedTensorDynamicShape(partitioned_dims, self.inner_dim_sizes,
                                      self.dim_size_dtype)
    else:
      inner_dims = array_ops.concat(
          [array_ops.ones([dims_to_add], self.dim_size_dtype),
           self.inner_dim_sizes],
          axis=0)
      return RaggedTensorDynamicShape([], inner_dims, self.dim_size_dtype)

  def broadcast_dimension(self, axis, lengths):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring describes tiling with a where-based repeat count; the code only
    # builds a runtime assertion and delegates to helper methods, and the "??" note is stale.
    # Inputs: axis (integer) and lengths (scalar or vector integer tensor, converted to this
    # shape's dtype). Adds an assertion op that the sizes are compatible (ragged axis: scalar 1
    # or vector equal to existing sizes; uniform axis: scalar 1 or equal, or existing size 1 for
    # vector lengths). Returns a new shape broadcast along the axis, with the check as a control
    # dependency. Raises ValueError for unknown or higher-than-1 rank lengths, and for a vector
    # of lengths on a non-partitioned axis zero. Failed compatibility is a runtime error.
    """Returns a shape that is broadcast-compatible with self & lengths.

    * If dimension[axis] is uniform and lengths is a scalar, the check
      that either lengths==1 or axis==1 or lengths==axis, and tile
      dimension[axis] with tf.where(lengths==axis, 1, axis) repeats.

    * If dimension[axis] is uniform and lengths is a vector, then check
      that dimension[axis]==1, and raggedly tile dimension[axis] with
      lengths repeats.  (we can skip tiling if we statically know that
      slice_lengths == 1??)

    * If dimension[axis] is ragged and lengths is a scalar, then check
      that lengths==1.

    * If dimension[axis] is ragged and lengths is a vector, then check
      that self.dimension_size(axis) == lengths.

    Args:
      axis: `int`.  The dimension to broadcast.
      lengths: 0-D or 1-D integer `Tensor`.

    Returns:
      A `RaggedTensorDynamicShape`.
    """
    lengths = ragged_util.convert_to_int_tensor(
        lengths, name='lengths', dtype=self.dim_size_dtype)
    # Check whether lengths is a scalar (for uniform dimensions) or
    # vector (for ragged dimensions).
    if lengths.shape.ndims is None:
      raise ValueError('lengths must have a known rank.')
    elif lengths.shape.ndims > 1:
      raise ValueError('lengths must be a scalar or vector')
    else:
      lengths_is_scalar = (lengths.shape.ndims == 0)

    # Verify that the shapes are compatible.
    if self.is_ragged(axis):
      if lengths_is_scalar:
        condition = math_ops.equal(lengths, 1)
      else:
        condition = math_ops.reduce_all(
            math_ops.equal(lengths, self.dimension_size(axis)))
    else:
      axis_dim_size = self.dimension_size(axis)
      if lengths_is_scalar:
        condition = (
            math_ops.equal(lengths, 1) | math_ops.equal(axis_dim_size, 1)
            | math_ops.equal(axis_dim_size, lengths))
      else:
        condition = math_ops.equal(axis_dim_size, 1)
    broadcast_err = [
        'Unable to broadcast: dimension size mismatch in dimension', axis,
        'lengths=', lengths, 'dim_size=',
        self.dimension_size(axis)
    ]
    broadcast_check = control_flow_assert.Assert(
        condition, data=broadcast_err, summarize=10)

    with ops.control_dependencies([broadcast_check]):
      # Partitioned dimensions:
      if axis < self.num_partitioned_dimensions:
        if self.is_ragged(axis):
          # Use an identity op to make sure the check actually gets run.
          return RaggedTensorDynamicShape(
              self._partitioned_dim_sizes,
              array_ops.identity(self.inner_dim_sizes), self.dim_size_dtype)
        else:
          return self._broadcast_uniform_partitioned_dimension(axis, lengths)

      # Inner dimensions:
      else:
        if lengths_is_scalar:
          return self._broadcast_inner_dimension_to_uniform(axis, lengths)
        else:
          if axis == 0:
            raise ValueError('Unable to broadcast: '
                             'outermost dimension must be uniform.')
          return self._broadcast_inner_dimension_to_ragged(axis, lengths)

  def num_slices_in_dimension(self, axis):
    # [FUNCTION DOC]
    # Input: axis (integer). Returns a scalar tensor with the total number of slices across the
    # axis: one for negative axes, the sum of the lengths for a ragged axis, and otherwise the
    # axis size times the slice count of the previous axis (computed recursively).
    """Returns the total number of slices across the indicated dimension."""
    if axis < 0:
      return constant_op.constant(1, dtype=self.dim_size_dtype)
    elif self.is_ragged(axis):
      return math_ops.reduce_sum(self._partitioned_dim_sizes[axis])
    else:
      return self.dimension_size(axis) * self.num_slices_in_dimension(axis - 1)

  def _broadcast_uniform_partitioned_dimension(self, axis, lengths):
    # [FUNCTION DOC]
    # HELPER: used by broadcast_dimension for uniform partitioned axes.
    # Inputs: axis and lengths (scalar or vector tensor). For scalar lengths the axis size is
    # replaced by lengths when it was one; for vector lengths the axis becomes the ragged lengths
    # vector. Every later partitioned dimension is rebuilt by repeating ranges accordingly (or
    # kept when uniform). Returns a new shape; inner sizes are unchanged.
    """Broadcasts the partitioned dimension `axis` to match `lengths`."""
    axis_dim_size = self.dimension_size(axis)
    partitioned_sizes = list(self._partitioned_dim_sizes[:axis])

    if lengths.shape.ndims == 0:
      lengths = array_ops.where(
          math_ops.equal(axis_dim_size, 1), lengths, axis_dim_size)
      repeats = array_ops.where(math_ops.equal(axis_dim_size, 1), lengths, 1)
      splits = array_ops_stack.stack([0, self.num_slices_in_dimension(axis)])
    else:
      splits = math_ops.range(
          array_ops.size(lengths, out_type=self.dim_size_dtype) + 1)
      repeats = lengths

    partitioned_sizes.append(lengths)

    for dim_size in self._partitioned_dim_sizes[axis + 1:]:
      if dim_size.shape.ndims == 0:
        partitioned_sizes.append(dim_size)
        splits *= dim_size
      else:
        partitioned_sizes.append(
            ragged_util.repeat_ranges(dim_size, splits, repeats))
        splits = array_ops.gather(
            ragged_util.lengths_to_splits(dim_size), splits)
    inner_sizes = self._inner_dim_sizes
    return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes,
                                    self.dim_size_dtype)

  def _broadcast_inner_dimension_to_uniform(self, axis, length):
    # [FUNCTION DOC]
    # HELPER: used by broadcast_dimension for inner axes with scalar lengths.
    # KNOWN ISSUE: docstring mentions lengths but the parameter is named length.
    # Replaces the inner size at the axis by the given length when the current size is one;
    # otherwise keeps it. Returns a new shape with the same partitioned sizes.
    """Broadcasts the inner dimension `axis` to match `lengths`."""
    dim_size = self.dimension_size(axis)
    axis_in_inner_dims = axis - self.num_partitioned_dimensions
    partitioned_sizes = self._partitioned_dim_sizes
    inner_sizes = array_ops.concat([
        self._inner_dim_sizes[:axis_in_inner_dims],
        [array_ops.where(math_ops.equal(dim_size, 1), length, dim_size)],
        self._inner_dim_sizes[axis_in_inner_dims + 1:]
    ],
                                   axis=0)
    return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes,
                                    self.dim_size_dtype)

  def _broadcast_inner_dimension_to_ragged(self, axis, lengths):
    # [FUNCTION DOC]
    # HELPER: used by broadcast_dimension for inner axes with vector lengths.
    # KNOWN ISSUE: no dtype is passed, so the constructor re-infers the size dtype instead of
    # reusing this shape's dtype.
    # Turns the inner dimensions up to and including the axis into partitioned dimensions (the
    # earlier inner sizes become uniform partitioned sizes, the axis becomes the lengths vector)
    # and keeps the remaining inner sizes. Returns a new shape.
    axis_in_inner_dims = axis - self.num_partitioned_dimensions
    partitioned_sizes = (
        self._partitioned_dim_sizes + tuple([
            self._inner_dim_sizes[i] for i in range(axis_in_inner_dims)
        ]) + (lengths,))
    inner_sizes = self._inner_dim_sizes[axis_in_inner_dims + 1:]
    return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes)

  def with_dim_size_dtype(self, dtype):
    # [FUNCTION DOC]
    # Input: dtype, which must be int32 or int64 (else ValueError). Returns this shape if it
    # already uses that dtype, otherwise a new shape with all sizes cast to it.
    if dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError('dtype must be int32 or int64')
    if self.dim_size_dtype == dtype:
      return self
    return RaggedTensorDynamicShape(
        [math_ops.cast(p, dtype) for p in self._partitioned_dim_sizes],
        math_ops.cast(self._inner_dim_sizes, dtype))


def broadcast_dynamic_shape(shape_x, shape_y):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring lists only ValueError, but a type error is raised when either
  # argument is not a RaggedTensorDynamicShape.
  # Inputs: two shape objects. Raises ValueError if either rank is unknown. Pads both to the
  # larger rank, then broadcasts one axis at a time, outermost first, each against the other.
  # Returns the broadcast shape (built from the first argument). Incompatible sizes surface as
  # runtime assertion failures.
  """Returns the shape formed by broadcasting two shapes to be compatible.

  Args:
    shape_x: A `RaggedTensorDynamicShape`
    shape_y: A `RaggedTensorDynamicShape`

  Returns:
    A `RaggedTensorDynamicShape`.
  Raises:
    ValueError: If `shape_x` and `shape_y` are not broadcast-compatible.
  """
  if not isinstance(shape_x, RaggedTensorDynamicShape):
    raise TypeError('shape_x must be a RaggedTensorDynamicShape')
  if not isinstance(shape_y, RaggedTensorDynamicShape):
    raise TypeError('shape_y must be a RaggedTensorDynamicShape')

  # Broadcast both shapes to have the same rank.
  if shape_x.rank is None or shape_y.rank is None:
    raise ValueError('Unable to broadcast: unknown rank')
  broadcast_rank = max(shape_x.rank, shape_y.rank)
  shape_x = shape_x.broadcast_to_rank(broadcast_rank)
  shape_y = shape_y.broadcast_to_rank(broadcast_rank)

  # Broadcast dimensions one at a time, starting from the outermost dimension.
  for axis in range(broadcast_rank):
    shape_x = shape_x.broadcast_dimension(axis, shape_y.dimension_size(axis))
    shape_y = shape_y.broadcast_dimension(axis, shape_x.dimension_size(axis))

  return shape_x


def broadcast_to(rt_input, shape, broadcast_inner_dimensions=True):
  # [FUNCTION DOC]
  # Inputs: a dense or ragged tensor, a RaggedTensorDynamicShape target, and a flag (default
  # true) controlling whether inner dimensions are tiled. Raises a type error if shape is not the
  # right class. Dispatches to the uniform-shape helper when the shape has no partitioned
  # dimensions, otherwise to the ragged helper. Returns the broadcast tensor.
  """Broadcasts a potentially ragged tensor to a ragged shape.

  Tiles `rt_input` as necessary to match the given shape.

  Behavior is undefined if `rt_input` is not broadcast-compatible with `shape`.

  Args:
    rt_input: The potentially ragged tensor to broadcast.
    shape: A `RaggedTensorDynamicShape`
    broadcast_inner_dimensions: If false, then inner dimensions will not be
      tiled.

  Returns:
    A potentially ragged tensor whose values are taken from
    `rt_input`, and whose shape matches `shape`.
  """
  if not isinstance(shape, RaggedTensorDynamicShape):
    raise TypeError('shape must be a RaggedTensorDynamicShape')
  rt_input = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt_input)

  # Broadcasting to a uniform shape.
  if shape.num_partitioned_dimensions == 0:
    return _broadcast_to_uniform_shape(rt_input, shape,
                                       broadcast_inner_dimensions)
  else:
    return _broadcast_to_ragged_shape(rt_input, shape,
                                      broadcast_inner_dimensions)


def _broadcast_to_uniform_shape(rt_input, shape, broadcast_inner_dimensions):
  # [FUNCTION DOC]
  # HELPER: broadcast_to for shapes without partitioned dimensions.
  # Raises ValueError if the input is ragged. Returns the input broadcast to the inner sizes, or
  # the input unchanged when inner broadcasting is disabled.
  """Broadcasts rt_input to the uniform shape `shape`."""
  if isinstance(rt_input, ragged_tensor.RaggedTensor):
    raise ValueError('Incompatible with shape: ragged rank mismatch')
  if broadcast_inner_dimensions:
    return array_ops.broadcast_to(rt_input, shape.inner_dim_sizes)
  else:
    return rt_input


def _broadcast_to_ragged_shape(rt_input, dst_shape, broadcast_inner_dimensions):
  # [FUNCTION DOC]
  # HELPER: broadcast_to for shapes with partitioned dimensions.
  # Inputs: tensor, destination shape, inner-broadcast flag. Aligns row-splits dtypes (casting
  # both to int64 only if auto-cast is enabled, else ValueError), validates rank and ragged rank
  # (ValueError on unknown rank or mismatch), then wraps the input in extra outer dimensions,
  # converts it to a ragged tensor of matching ragged rank, tiles dimensions that stay uniform,
  # broadcasts the flat values over inner dimensions, and finally tiles dimensions that become
  # ragged from outermost to innermost. Returns the resulting ragged tensor.
  """Broadcasts rt_input to the ragged shape `dst_shape`."""
  # Check that rt_input and dst_shape have the same row_splits dtype.
  if (isinstance(rt_input, ragged_tensor.RaggedTensor) and
      rt_input.row_splits.dtype != dst_shape.dim_size_dtype):
    if not ragged_config.auto_cast_partition_dtype():
      raise ValueError('rt_input and dst_shape have different row_split '
                       'dtypes; use RaggedTensor.with_row_splits_dtype() or '
                       'RaggedTensorDynamicShape.with_dim_size_dtype() to '
                       'convert to a compatible dtype.')
    rt_input = rt_input.with_row_splits_dtype(dtypes.int64)
    dst_shape = dst_shape.with_dim_size_dtype(dtypes.int64)

  # dst_shape's rank and ragged_rank must be greater than or equal to rt_input's
  if rt_input.shape.ndims is None or dst_shape.rank is None:
    raise ValueError('Unable to broadcast: unknown rank')
  if rt_input.shape.ndims > dst_shape.rank:
    raise ValueError('Incompatible with shape: rank mismatch')
  if (isinstance(rt_input, ragged_tensor.RaggedTensor) and
      rt_input.ragged_rank >= dst_shape.num_partitioned_dimensions):
    raise ValueError('Incompatible with shape: ragged rank mismatch')

  src_shape = RaggedTensorDynamicShape.from_tensor(rt_input)
  src_shape = src_shape.broadcast_to_rank(dst_shape.rank)

  # Add dimensions to rt_input so its rank and ragged_rank matches dst_shape.
  if dst_shape.rank > rt_input.shape.ndims:
    if rt_input.shape.ndims < dst_shape.num_inner_dimensions + 1:
      rt_input = array_ops.reshape(
          rt_input, array_ops.concat([[-1], dst_shape.inner_dim_sizes], axis=0))
    for _ in range(dst_shape.rank - rt_input.shape.ndims):
      if ragged_tensor.is_ragged(rt_input):
        nrows = rt_input.nrows()
      else:
        nrows = array_ops.shape(rt_input,
                                out_type=dst_shape.dim_size_dtype)[0]
      rt_input = ragged_tensor.RaggedTensor.from_row_lengths(rt_input, [nrows],
                                                             validate=False)

  # Add ragged dimensions to match dst_shape.
  if ragged_tensor.is_ragged(rt_input):
    inner_rank_diff = (
        rt_input.flat_values.shape.ndims - 1 - dst_shape.num_inner_dimensions)
    if inner_rank_diff > 0:
      rt_input = rt_input.with_flat_values(
          ragged_tensor.RaggedTensor.from_tensor(
              rt_input.flat_values, ragged_rank=inner_rank_diff,
              row_splits_dtype=dst_shape.dim_size_dtype))
  else:
    rt_input = ragged_tensor.RaggedTensor.from_tensor(
        rt_input, ragged_rank=dst_shape.num_partitioned_dimensions - 1,
        row_splits_dtype=dst_shape.dim_size_dtype)

  # Do broadcasting for any dimensions that will remain uniform.  We can do
  # these all at once, since they're independent of one another.
  multiples = [1] * dst_shape.rank
  for axis in range(dst_shape.num_partitioned_dimensions):
    if not src_shape.is_ragged(axis) and not dst_shape.is_ragged(axis):
      src_size = src_shape.dimension_size(axis)
      dst_size = dst_shape.dimension_size(axis)
      if ((tensor_util.constant_value(src_size) in (1, None)) and
          (tensor_util.constant_value(dst_size) != 1)):
        multiples[axis] = array_ops.where(
            math_ops.equal(src_size, 1), dst_size, 1)
  if not all(isinstance(v, int) and v == 1 for v in multiples):
    multiples = array_ops_stack.stack(multiples, axis=0)
    rt_input = ragged_array_ops.tile(rt_input, multiples)

  if broadcast_inner_dimensions:
    new_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(
            rt_input.flat_values, out_type=dst_shape.dim_size_dtype),
        array_ops.concat([[1], dst_shape.inner_dim_sizes], axis=0))
    rt_input = rt_input.with_flat_values(
        array_ops.broadcast_to(rt_input.flat_values, new_shape))

  # Do broadcasting for dimensions that become ragged.  We must do these from
  # outermost to innermost.
  for axis in range(dst_shape.num_partitioned_dimensions):
    if not src_shape.is_ragged(axis) and dst_shape.is_ragged(axis):
      dst_size = dst_shape.dimension_size(axis)
      rt_input = _ragged_tile_axis(rt_input, axis, dst_size,
                                   dst_shape.dim_size_dtype)

  return rt_input


def _ragged_tile_axis(rt_input, axis, repeats, row_splits_dtype):
  # [FUNCTION DOC]
  # HELPER: used by _broadcast_to_ragged_shape to make a uniform axis ragged.
  # Inputs: tensor, axis (must be greater than zero, checked by assertion), repeats (per-slice
  # repeat counts) and the row-splits dtype. Converts a dense input to ragged, recurses into the
  # values for deeper axes, and at axis one repeats each row's contents the given number of times
  # while rebuilding nested row lengths. Returns the new ragged tensor.
  """Tile a dimension of a RaggedTensor to match a ragged shape."""
  assert axis > 0  # Outermost dimension may not be ragged.

  if not ragged_tensor.is_ragged(rt_input):
    rt_input = ragged_tensor.RaggedTensor.from_tensor(
        rt_input, ragged_rank=1, row_splits_dtype=row_splits_dtype)

  if axis > 1:
    return rt_input.with_values(
        _ragged_tile_axis(rt_input.values, axis - 1, repeats,
                          row_splits_dtype))
  else:
    src_row_splits = rt_input.nested_row_splits
    src_row_lengths = rt_input.nested_row_lengths()
    splits = src_row_splits[0]

    dst_row_lengths = [repeats]
    for i in range(1, len(src_row_lengths)):
      dst_row_lengths.append(
          ragged_util.repeat_ranges(src_row_lengths[i], splits, repeats))
      splits = array_ops.gather(src_row_splits[i], splits)
    dst_values = ragged_util.repeat_ranges(rt_input.flat_values, splits,
                                           repeats)
    return ragged_tensor.RaggedTensor.from_nested_row_lengths(
        dst_values, dst_row_lengths, validate=False)
