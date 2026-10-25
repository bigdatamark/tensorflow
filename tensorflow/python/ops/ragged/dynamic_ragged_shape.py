# [MODULE DOC]
# Shapes and broadcasting for ragged and dense tensors. Defines DynamicRaggedShape, an extension
# type (exported publicly as the experimental DynamicRaggedShape) that stores a shape as an
# inner dense shape vector plus a list of row partitions, so that a shape can describe a ragged
# tensor, a dense tensor, or a tensor of unknown rank. Also contains the machinery used to
# broadcast two such shapes together and to broadcast ragged or dense tensors to a target shape:
# layer broadcasters (one gather index per axis), a whole-shape broadcaster, and the public
# functions for broadcasting shapes and tensors.
# Also registers, at import time, the handlers (via the dispatch mechanism) that implement
# binary elementwise operations and binary elementwise assert operations for ragged tensors.
# Most private helpers at the bottom of the file support the shape class (dtype detection,
# slicing index normalization, batching of specs, formatting of lengths for repr).
# Imports of other TensorFlow ops are made at import time; no flags are defined and no other
# global state is modified.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Shapes & broadcasting for RaggedTensors.

TODO(martinz): make this suitable for output for tf.shape
TODO(martinz): replace ragged_tensor_shape with this.
"""

import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


class _DynamicRaggedShapeBatchEncoder(extension_type.ExtensionTypeBatchEncoder):
  # [CLASS DOC]
  # HELPER: batch encoder plugged into DynamicRaggedShape so shapes can be used in batching APIs.
  # Batch encoder for DynamicRaggedShape, installed as the batch encoder of that extension type.
  # It tells the framework how the spec of a shape changes when many shapes are batched into one
  # or one batched shape is unbatched, and how a shape is encoded as an ordinary tensor (a
  # boolean tensor of ones with that shape) and decoded back. Stateless; one instance is created
  # at class-definition time of DynamicRaggedShape.
  """A batch encoder for DynamicRaggedShape below."""

  def batch(self, spec: "DynamicRaggedShape.Spec",
            batch_size) -> "DynamicRaggedShape.Spec":
    # [FUNCTION DOC]
    # HELPER: batching support for the DynamicRaggedShape batch encoder.
    # Computes the spec of a batch of shapes, given the spec of one shape and a batch size, which
    # may be unknown. Inputs: spec, a DynamicRaggedShape spec, and batch_size, an integer or
    # nothing. Returns a new spec. If the spec has row partitions, a new leading uniform partition
    # for the batch dimension is added and every existing partition is batched (rows and values
    # multiplied by the batch size when known), and the inner shape's first dimension is scaled.
    # If there are no row partitions and the batch size is unknown, a scalar spec becomes a vector
    # spec of unknown length, and otherwise a uniform partition with the old first dimension is
    # created. If there are no row partitions and the batch size is known, a dense spec with the
    # batch size prepended to the inner shape is returned. No exceptions are raised here; no side
    # effects.
    if spec.num_row_partitions:
      new_head = _batch_rp_spec_head(spec._row_partitions[0], batch_size)  # pylint:disable=protected-access
      new_tail = [_batch_rp_spec(rp, batch_size) for rp in spec._row_partitions]  # pylint:disable=protected-access
      new_rp = [new_head] + new_tail
      new_static_inner_shape = _batch_static_inner_shape(
          spec._static_inner_shape, batch_size)  # pylint:disable=protected-access

      return DynamicRaggedShape.Spec(
          row_partitions=new_rp,
          static_inner_shape=new_static_inner_shape,
          dtype=spec.dtype)
    elif batch_size is None:
      if spec.inner_rank == 0:
        return DynamicRaggedShape.Spec._from_tensor_shape(  # pylint:disable=protected-access
            [None],
            0,
            dtype=spec.dtype)
      else:
        # Might be None
        new_head = RowPartitionSpec(
            uniform_row_length=spec._dimension(0),  # pylint:disable=protected-access
            dtype=spec.dtype)
        new_static_inner_shape = _batch_static_inner_shape(
            spec._static_inner_shape, batch_size)  # pylint:disable=protected-access
        return DynamicRaggedShape.Spec(
            row_partitions=[new_head],
            static_inner_shape=new_static_inner_shape,
            dtype=spec.dtype)
    else:

      return DynamicRaggedShape.Spec(
          row_partitions=[],
          static_inner_shape=_batch_tensor_shape(
              spec._static_inner_shape,  # pylint:disable=protected-access
              batch_size),
          dtype=spec.dtype)

  def unbatch(self,
              spec: "DynamicRaggedShape.Spec") -> "DynamicRaggedShape.Spec":
    # [FUNCTION DOC]
    # HELPER: unbatching support for the DynamicRaggedShape batch encoder.
    # Computes the spec of a single element from the spec of a batched shape. Input: spec, a
    # DynamicRaggedShape spec. Returns a new spec with the outermost dimension removed. With row
    # partitions, the first partition is dropped; the remaining partitions have their row counts
    # and value counts divided by the number of rows of the first partition while the chain of
    # uniform partitions is unbroken (after the first non-uniform partition the counts become
    # unknown), and the static inner shape is unbatched accordingly. Without row partitions, the
    # first dimension of the static inner shape is simply removed. No side effects.
    if spec.num_row_partitions:
      result = []
      head = spec._row_partitions[0]  # pylint:disable=protected-access
      scale = None if head.uniform_row_length is None else head.nrows

      for rp in spec._row_partitions[1:]:  # pylint:disable=protected-access
        if scale is None:
          result.append(
              RowPartitionSpec(
                  nrows=None,
                  nvals=None,
                  uniform_row_length=rp.uniform_row_length,
                  dtype=spec.dtype))
        else:
          nrows = None if rp.nrows is None else rp.nrows // scale
          if rp.uniform_row_length is None:
            scale = None
            result.append(
                RowPartitionSpec(
                    nrows=nrows,
                    nvals=None,
                    uniform_row_length=None,
                    dtype=spec.dtype))
          else:
            result.append(
                RowPartitionSpec(
                    nrows=nrows,
                    nvals=rp.nvals // scale,
                    uniform_row_length=rp.uniform_row_length,
                    dtype=spec.dtype))
      return DynamicRaggedShape.Spec(
          row_partitions=result,
          static_inner_shape=_unbatch_static_inner_shape(
              spec._static_inner_shape, scale),  # pylint:disable=protected-access
          dtype=spec.dtype)
    else:  # spec.num_row_partitions == 0
      return DynamicRaggedShape.Spec(
          row_partitions=[],
          static_inner_shape=spec._static_inner_shape[1:],  # pylint:disable=protected-access
          dtype=spec.dtype)

  def decode(self, spec: "DynamicRaggedShape.Spec",
             encoding) -> "DynamicRaggedShape":
    # [FUNCTION DOC]
    # HELPER: decoding support for the DynamicRaggedShape batch encoder.
    # Turns an encoded value (a tensor or ragged tensor) back into a DynamicRaggedShape by taking
    # its shape, using the dtype from the given spec. Inputs: spec, the shape spec, and encoding,
    # the encoded tensor. Returns a DynamicRaggedShape. No side effects.
    return DynamicRaggedShape.from_tensor(encoding, dtype=spec.dtype)

  def encode(
      self,
      spec: "DynamicRaggedShape.Spec",
      value,
      minimum_rank=0) -> Union[ragged_tensor.RaggedTensor, tensor_lib.Tensor]:
    # [FUNCTION DOC]
    # HELPER: encoding support for the DynamicRaggedShape batch encoder.
    # KNOWN ISSUE: the spec and minimum_rank arguments are accepted but ignored.
    # Encodes a shape value as a tensor: returns a tensor or ragged tensor of boolean ones whose
    # shape equals the given shape value (built by the module level function that creates ones).
    # Inputs: spec (unused), value, the DynamicRaggedShape to encode, minimum_rank (unused,
    # default zero). No side effects.
    return ones(value, dtype=dtypes.bool)

  def encoding_specs(
      self, spec: "DynamicRaggedShape.Spec"
  ) -> Union[ragged_tensor.RaggedTensorSpec, tensor_lib.TensorSpec]:
    # [FUNCTION DOC]
    # HELPER: encoding support for the DynamicRaggedShape batch encoder.
    # Returns the type specification of the boolean tensor produced by encode, given a shape spec.
    # The result is a ragged tensor spec with the shape converted from the spec, boolean values,
    # and the spec's dtype as row splits dtype. The ragged rank equals the number of row
    # partitions, except for a scalar shape (rank not zero test fails) where it is set to minus one
    # because the encoding must be unbatched twice to get a ragged tensor. No side effects.
    if spec.rank != 0:
      ragged_rank = spec.num_row_partitions
    else:
      # special case: need to unbatch twice to get ragged tensor.
      ragged_rank = -1
    return ragged_tensor.RaggedTensorSpec(
        shape=spec._to_tensor_shape(),  # pylint:disable=protected-access
        dtype=dtypes.bool,
        ragged_rank=ragged_rank,
        row_splits_dtype=spec.dtype)


# TODO(martinz): allow inner_shape to be a fully defined TensorShape.
# A "fully defined TensorShape" means one where the rank and all dimensions are
# known.
# Allowing inner_shape might mean allowing inner_shape to be initialized by
# a fully defined TensorShape, or it might mean that you can actually store
# TensorShape in the inner_shape field. This could conceivably construct
# a DynamicRaggedShape that was dtype agnostic.
#
# TODO(martinz): unify the impl of the determination of index type across
#     RowPartition and DynamicRaggedShape.
@tf_export("experimental.DynamicRaggedShape")
class DynamicRaggedShape(extension_type.BatchableExtensionType):
  # [CLASS DOC]
  # Extension type representing the shape of a ragged or dense tensor, registered for public use
  # as the experimental DynamicRaggedShape. A shape is stored as an inner shape (an integer
  # vector tensor giving the shape of the flat values or of the whole dense tensor), a tuple of
  # row partitions describing the ragged or uniform splits of the outer dimensions, and a static
  # inner shape holding what is known at graph construction time. The dtype (int32 or int64) is
  # shared by the inner shape and all row partitions. Instances are created through the
  # constructor, from_lengths, from_row_partitions or from_tensor. Offers rank queries, indexing
  # and slicing, uniformity tests, re-factoring into a different number of row partitions,
  # merging, and dtype conversion. The nested Spec class is the type specification. Batching is
  # handled by the batch encoder class declared as the batch encoder attribute.
  """The shape of a ragged or dense tensor.

  Ragged shapes are encoded using two fields:

  * `inner_shape`: An integer vector giving the shape of a dense tensor.
  * `row_partitions`: A list of `RowPartition` objects, describing how
    that flat shape should be partitioned to add ragged axes.

  If a DynamicRaggedShape is the shape of a RaggedTensor rt, then:
  1. row_partitions = rt._nested_row_partitions
     (and thus len(row_partitions) > 0)
  2. inner_shape is the shape of rt.flat_values

  If a DynamicRaggedShape is the shape of a dense tensor t, then:
  1. row_partitions = []
  2. inner_shape is the shape of t.

  Examples:

  The following table gives a few examples (where `RP(lengths)` is short
  for `RowPartition.from_lengths(lengths)`):

  Row Partitions              | Inner Shape  | Example Tensor
  --------------------------- | ------------ | ----------------------------
  []                          | [2, 3]       | `[[1, 2, 3], [4, 5, 6]]`
  [RP([2, 0, 3])]             | [5]          | `[[1, 2], [], [3, 4, 5]]`
  [RP([2, 1])]                | [3, 2]       | `[[[1, 2], [3, 4]], [[5, 6]]]`
  [RP([2, 1]), RP([2, 1, 2])] | [5]          | `[[[1, 2], [3]], [[4, 5]]]`
  """
  _row_partitions: Tuple[RowPartition, ...]
  _inner_shape: tensor_lib.Tensor
  _static_inner_shape: tensor_shape.TensorShape
  __batch_encoder__ = _DynamicRaggedShapeBatchEncoder()
  __name__ = "tf.DynamicRaggedShape"

  def __init__(self,
               row_partitions: Sequence[RowPartition],
               inner_shape: core.TensorLike,
               dtype: Optional[dtypes.DType] = None,
               validate: bool = False,
               static_inner_shape: ... = None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the static_inner_shape annotation is a literal ellipsis placeholder, not a type.
    # KNOWN ISSUE: when dynamic checks are needed the row partitions are stored as a list, not a
    # tuple as in the class annotation, which may break later code that adds tuples.
    # Core constructor. Inputs: row_partitions, an iterable of RowPartition objects (empty for a
    # dense shape); inner_shape, a tensor-like integer vector; dtype, optional int32 or int64
    # (otherwise inferred from the partitions, the inner shape, a numpy int32 array, and finally
    # defaulting to int64); validate, if true adds dynamic assertions; static_inner_shape,
    # optional extra static information merged into the statically inferred one.
    # Behavior: converts the inner shape to a tensor and casts all parts to the chosen dtype;
    # checks statically (raising a value error) that consecutive partitions align and that the last
    # partition's value count matches the first inner dimension; if validate is true and static
    # values are unknown, adds assert ops and attaches them as control dependencies to the inner
    # shape and partitions. Raises a type error if row_partitions is not iterable or contains
    # non-RowPartition items, and a value error for static mismatches. The inner shape must be rank
    # one. Sets the instance attributes.
    """Core constructor for a DynamicRaggedShape.

    Create a DynamicRaggedShape. This can be used to construct a
    DynamicRaggedShape representing a ragged or dense shape. If row_partitions
    is an empty list, then this is equivalent to a dense shape.

    If row_partitions is specified, then the num_row_partitions will be equal
    to len(row_partitions). There are several checks made.
    Specifically:
    1. Consecutive row_partitions must have consistent nvals and nrows.
    2. The last row_partitions must have nvals equal to the first element of
       inner_shape.

    The inner_shape is converted to a tensor.
    All row_partitions and the inner_shape are converted to the same dtype
    (int64 or int32).

    Args:
      row_partitions: the row_partitions of the shape.
      inner_shape: if len(row_partitions) > 0, the shape of the flat_values.
        Otherwise, the shape of the tensor.
      dtype: tf.int64, tf.int32, or None representing the preferred dtype.
      validate: if true, dynamic validation is applied to the shape.
      static_inner_shape: if len(row_partitions) > 0, the static shape of the
        flat_values. Otherwise, the static shape of the tensor. Should be
        convertible to a TensorShape.
    """
    if not isinstance(row_partitions, Iterable):
      raise TypeError(
          "row_partitions should be a list of row partitions. Instead, got " +
          str(row_partitions))
    for x in row_partitions:
      if not isinstance(x, RowPartition):
        raise TypeError("row_partitions contains " + str(x) +
                        " which is not a RowPartition")
    dtype = _find_dtype_iterable(row_partitions, dtype)
    dtype = _find_dtype(inner_shape, dtype)
    if (isinstance(inner_shape, np.ndarray) and
        inner_shape.dtype == np.int32 and dtype is None):
      dtype = dtypes.int32
    dtype = _find_dtype(dtypes.int64, dtype)

    row_partitions = tuple([rp.with_dtype(dtype) for rp in row_partitions])
    self._row_partitions = row_partitions
    self._inner_shape = ops.convert_to_tensor(
        inner_shape, dtype_hint=dtype, name="inner_dim_sizes")
    if self._inner_shape.dtype != dtype:
      self._inner_shape = math_ops.cast(self._inner_shape, dtype)

    checks = []
    # Validate shapes.
    if self._row_partitions:
      for axis, rp in enumerate(self._row_partitions):
        if axis > 0:
          previous_row_partition = self._row_partitions[axis - 1]
          msg = ("RowPartitions in DynamicRaggedShape do not align "
                 f"between {axis - 1} and {axis}")
          static_nrows = rp.static_nrows
          static_nvals = previous_row_partition.static_nvals
          if (static_nrows is not None) and (static_nvals is not None):
            if static_nrows != static_nvals:
              raise ValueError(msg)
            else:
              continue
          if validate:
            checks.append(
                check_ops.assert_equal(
                    previous_row_partition.nvals(), rp.nrows(), message=msg))

    self._inner_shape.shape.assert_has_rank(1)

    self._static_inner_shape = tensor_util.constant_value_as_shape(
        self._inner_shape)
    if static_inner_shape is not None:
      self._static_inner_shape = self._static_inner_shape.merge_with(
          static_inner_shape)

    if row_partitions:
      last_row_partition = row_partitions[-1]
      static_nvals = last_row_partition.static_nvals
      static_inner_shape_nvals = tensor_shape.dimension_value(
          self._static_inner_shape[0])
      if static_nvals is not None and static_inner_shape_nvals is not None:
        if static_nvals != static_inner_shape_nvals:
          raise ValueError("Last row partition does not match inner_shape.")
      elif validate:
        checks.append(
            check_ops.assert_equal(
                last_row_partition.nvals(),
                self._inner_shape[0],
                message="Last row partition does not match inner_shape."))
    if checks:
      self._inner_shape = control_flow_ops.with_dependencies(
          checks, self._inner_shape, name="inner_shape_validated")
      self._row_partitions = [
          rp._with_dependencies(checks) for rp in self._row_partitions  # pylint: disable=protected-access
      ]

  @classmethod
  def from_lengths(cls,
                   lengths: Sequence[Union[Sequence[int], int]],
                   num_row_partitions=None,
                   dtype=dtypes.int64):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says each length may be an int or a list of ints, but the code
    # only accepts an int or a tuple of ints; a list element raises a value error.
    # KNOWN ISSUE: the dtype is applied only when building the final shape; the row partitions
    # built from the lengths are created without it and converted afterwards.
    # Class method that builds a shape from per-axis lengths. Inputs: lengths, a list in which each
    # element is a non-negative int (uniform axis) or a tuple of ints (ragged axis, one entry per
    # slice); num_row_partitions, optional int, defaulting to the smallest number that covers the
    # last ragged axis; dtype, int64 by default. Returns a DynamicRaggedShape whose leading
    # partitions are built from the lengths and whose inner shape holds the remaining dimensions.
    # Raises a value error if lengths is not a list, an element has a bad type, num_row_partitions
    # is not an int, it is positive for a scalar shape, or it is not smaller than the number of
    # lengths; a ragged length inconsistent with the size so far also raises a value error.
    """Creates a shape with the given lengths and num_row_partitions.

    The lengths can either be a nonnegative int or a list of nonnegative ints.

    If num_row_partitions is None, then the minimal num_row_partitions is used.

    For example, [2, (3, 2)] is the shape of [[0, 0, 0], [0, 0]], and
    [2, 2] is the shape of [[0, 0], [0, 0]]

    This chooses the minimal num_row_partitions required (including zero).

    The following table gives a few examples (where `RP(lengths)` is short
    for `RowPartition.from_lengths(lengths)`):

    For example:
    from_lengths           | row_partitions            | inner_shape
    ---------------------- | --------------------------| -------------
    []                     | []                        | []
    [2, (3, 2)]            | [RP([3, 2])]              | [5]
    [2, 2]                 | []                        | [2, 2]
    [2, (3, 2), 7]         | [RP([3, 2])]              | [5, 7]
    [2, (2, 2), 3]         | [RP([2, 2])]              | [4, 3]
    [2, 2, 3]              | []                        | [2, 2, 3]
    [2, (2, 1), (2, 0, 3)] | [RP(2, 1), RP([2, 0, 3])] | [5]

    If we want the row partitions to end with uniform row partitions, then
    we can set num_row_partitions.

    For example,
    below URP(3, 12) is RowPartition.from_uniform_row_length(3, 12)

    from_lengths   | num_row_partitions | row_partitions           | inner_shape
    ---------------| -------------------|--------------------------|------------
    [2, (3, 2), 2] | 2                  | [RP([3, 2]), URP(2, 10)] | [10]
    [2, 2]         | 1                  | [URP(2, 4)]              | [4]
    [2, 2, 3]      | 0                  | []                       | [2, 2, 3]
    [2, 2, 3]      | 1                  | [URP(2, 4)]              | [4, 3]
    [2, 2, 3]      | 2                  | [URP(2, 4), URP(3, 12)]  | [12]



    Representing the shapes from init():

    from_lengths             | Tensor Example
    ------------------------ | ------------------------------
    `[2, 3]`                 | `[[1, 2, 3], [4, 5, 6]]`
    `[3, (2, 0, 3)]`         | `[[1, 2], [], [3, 4, 5]]`
    `[2, (2, 1), 2]`         | `[[[1, 2], [3, 4]], [[5, 6]]]`
    `[2, (2, 1), (2, 1, 2)]` | `[[[1, 2], [3]], [[4, 5]]]`

    Args:
      lengths: the lengths of sublists along each axis.
      num_row_partitions: the num_row_partitions of the result or None
        indicating the minimum number of row_partitions.
      dtype: the dtype of the shape (tf.int32 or tf.int64).

    Returns:
      a new DynamicRaggedShape
    """
    if not isinstance(lengths, list):
      raise ValueError("lengths should be a list")
    for x in lengths:
      if not _is_int_or_tuple_of_ints(x):
        raise ValueError(
            "element of lengths should be int or tuple of ints: instead %r" %
            (x,))

    if num_row_partitions is None:
      # Calculate the minimal num_row_partitions.
      is_list = [not isinstance(x, int) for x in lengths]
      if any(is_list):
        # Last index when not a list.
        num_row_partitions = len(is_list) - is_list[-1::-1].index(True) - 1
      else:
        num_row_partitions = 0

    if not isinstance(num_row_partitions, int):
      raise ValueError("num_row_partitions should be an int or None")

    if not lengths:
      if num_row_partitions > 0:
        raise ValueError("num_row_partitions==0 for a scalar shape")
      return DynamicRaggedShape([], [], dtype=dtype)

    if not num_row_partitions < len(lengths):
      raise ValueError("num_row_partitions should be less than `len(lengths)` "
                       "if shape is not scalar.")

    if num_row_partitions > 0:
      (row_partitions, nvals) = _to_row_partitions_and_nvals_from_lengths(
          lengths[:num_row_partitions + 1])
      inner_shape = [nvals] + lengths[num_row_partitions + 1:]
      return DynamicRaggedShape(row_partitions, inner_shape, dtype=dtype)
    else:
      return DynamicRaggedShape([], lengths, dtype=dtype)

  @classmethod
  def from_row_partitions(cls, row_partitions, dtype=None):
    # [FUNCTION DOC]
    # Class method building a shape from a non-empty sequence of row partitions. The inner shape
    # is a vector holding the value count of the last partition, so the inner rank is one. Inputs:
    # row_partitions, a list of RowPartition; dtype, optional preferred dtype (default taken from
    # the partitions). Raises a value error if row_partitions is empty. Returns a new shape.
    """Create a shape from row_partitions.

    Args:
      row_partitions: a nonempty list of RowPartition objects.
      dtype: the dtype to use, or None to use the row_partitions dtype.

    Returns:
      a DynamicRaggedShape with inner_rank==1.
    """
    if not row_partitions:
      raise ValueError("row_partitions cannot be empty")
    inner_shape = [row_partitions[-1].nvals()]
    return DynamicRaggedShape(row_partitions, inner_shape, dtype=dtype)

  @classmethod
  def _from_inner_shape(cls, inner_shape, dtype=None):
    # [FUNCTION DOC]
    # HELPER: private constructor shortcut used by shape slicing, merging and broadcasting code.
    # Class method creating a dense shape (no row partitions) from an inner shape vector, with an
    # optional dtype. Returns a new DynamicRaggedShape.
    """Create a shape from inner_shape, where num_row_partitions == 0."""
    return DynamicRaggedShape([], inner_shape, dtype=dtype)

  # pylint: disable=protected-access
  @classmethod
  def from_tensor(cls, t, dtype=None):
    # [FUNCTION DOC]
    # Class method returning the shape of a tensor or ragged tensor. For a ragged tensor, uses its
    # nested row partitions and the shape of its flat values; for a dense tensor, uses its
    # dynamic shape as the inner shape with no partitions. dtype is optional. Returns a
    # DynamicRaggedShape. No other side effects.
    """Constructs a ragged shape for a potentially ragged tensor."""
    if ragged_tensor.is_ragged(t):
      return DynamicRaggedShape(
          t._nested_row_partitions, _flat_values_shape(t), dtype=dtype)
    else:
      return DynamicRaggedShape._from_inner_shape(
          array_ops.shape(t), dtype=dtype)

  @property
  def row_partitions(self):
    # [FUNCTION DOC]
    # Property returning the tuple of row partitions of this shape (empty for a dense shape).
    """The row_partitions of the shape."""
    return self._row_partitions

  @property
  def num_row_partitions(self):
    # [FUNCTION DOC]
    # Property returning the number of row partitions as an int.
    """The number of row_partitions of the shape."""
    return len(self._row_partitions)

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the dtype of the shape (int32 or int64), taken from the inner shape.
    """The dtype of the shape -- one of tf.int32 or tf.int64."""
    return self._inner_shape.dtype

  def _static_inner_shape_as_list(self, truncate_first):
    # [FUNCTION DOC]
    # HELPER: used by static_lengths to report the inner dimensions.
    # Returns the statically known inner dimensions as a list (unknown ones as nothing). If the
    # inner rank is unknown, returns a list holding only an ellipsis marker. If truncate_first is
    # true, the first inner dimension is dropped (it is represented by the last row partition).
    """Returns the lengths of the inner shape (if rank known), or [...]."""
    if self._static_inner_shape.rank is None:
      return [...]
    result = self._static_inner_shape.as_list()
    if truncate_first:
      return result[1:]
    return result

  def static_lengths(self, ragged_lengths=True):
    # [FUNCTION DOC]
    # Returns a list of statically known axis lengths. Input: ragged_lengths, default true; if
    # false, ragged axes are reported as unknown. For a dense shape returns the static inner shape
    # list. Otherwise the list starts with the static row count of the first partition, then for
    # each partition either the static uniform row length, a tuple of row lengths when ragged and
    # statically constant, or nothing when unknown, followed by the inner dimensions after the
    # first. If the inner rank is unknown the list ends with an ellipsis marker.
    """Returns a list of statically known axis lengths.

    This represents what values are known. For each row partition, it presents
    either the uniform row length (if statically known),
    the list of row lengths, or none if it is not statically known.
    For the inner shape, if the rank is known, then each dimension is reported
    if known, and None otherwise. If the rank of the inner shape is not known,
    then the returned list ends with an ellipsis.

    Args:
      ragged_lengths: If false, returns None for all ragged dimensions.

    Returns:
      A Sequence[Union[Sequence[int],int, None]] of lengths, with a possible
      Ellipsis at the end.
    """
    if self.num_row_partitions == 0:
      return self._static_inner_shape_as_list(False)
    first_dim = self.row_partitions[0].static_nrows
    if isinstance(first_dim, tensor_shape.Dimension):
      first_dim = first_dim.value
    rp_dims = [first_dim]
    for rp in self.row_partitions:
      if rp.is_uniform():
        rp_dims.append(rp.static_uniform_row_length)
      elif ragged_lengths:
        const_vals = tensor_util.constant_value(rp.row_lengths())
        if const_vals is None:
          rp_dims.append(None)
        else:
          rp_dims.append(tuple(const_vals.tolist()))
      else:
        rp_dims.append(None)

    return rp_dims + self._static_inner_shape_as_list(True)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a string with the static lengths and the number of row partitions, for debugging.
    lengths = _list_with_ellipsis_to_str(self.static_lengths())
    return ("<DynamicRaggedShape "
            "lengths=%s num_row_partitions=%r>" %
            (lengths, self.num_row_partitions))

  def _to_tensor_shape(self) -> tensor_shape.TensorShape:
    # [FUNCTION DOC]
    # HELPER: converts to a static TensorShape for use by the spec and encoder code.
    # Returns a TensorShape: scalar shape when there are no lengths, unknown shape if the rank is
    # unknown, otherwise the static lengths with ragged axes unknown.
    """Returns a TensorShape representation of the shape."""
    lengths = self.static_lengths(ragged_lengths=False)
    if not lengths:
      return tensor_shape.TensorShape(())
    if lengths[-1] == Ellipsis:
      return tensor_shape.TensorShape(None)
    return tensor_shape.TensorShape(lengths)

  def _slice_shape(self, start, stop):
    # [FUNCTION DOC]
    # HELPER: implements slicing for the indexing operator.
    # Returns the shape for dimensions from start up to (excluding) stop. Inputs: start and stop,
    # already normalized integers between zero and the rank. If stop is not after start, returns a
    # scalar shape. If start is zero, truncates to the leading dimensions, reusing row partitions
    # (and cutting the inner shape when stop exceeds the number of partitions; if the rank is
    # known and not larger than stop returns this shape itself). If start is positive, all
    # dimensions involved must be uniform (value error otherwise), the rank must be known (raises
    # not-implemented otherwise), and the result is a dense shape holding the corresponding tail.
    """Returns a shape self[start:stop].

    If start == 0, then this truncates dimensions after stop.
    If start != 0, then this will return a shape with num_row_partitions == 0.

    See __getitem__.

    Args:
      start: the first dimension. 0 <= start <= rank
      stop: the last dimension (exclusive). 0 <= stop <= rank
    """
    if stop <= start:
      return DynamicRaggedShape._from_inner_shape([])
    elif start == 0:
      if stop <= self.num_row_partitions:
        if stop == 1:
          return DynamicRaggedShape._from_inner_shape(
              [self.row_partitions[0].nrows()])
        new_row_partitions = self.row_partitions[:stop - 1]
        new_inner_shape = [new_row_partitions[-1].nvals()]
        return DynamicRaggedShape(new_row_partitions, new_inner_shape)
      else:
        if self.rank is None:
          new_inner_rank = stop - self.num_row_partitions
          new_inner_shape = self.inner_shape[:new_inner_rank]
          return DynamicRaggedShape(
              row_partitions=self.row_partitions,
              inner_shape=new_inner_shape,
              static_inner_shape=None,
              validate=False)

        elif self.rank <= stop:
          return self
        new_inner_rank = stop - self.num_row_partitions
        new_inner_shape = self.inner_shape[:new_inner_rank]
        return DynamicRaggedShape(
            row_partitions=self.row_partitions,
            inner_shape=new_inner_shape,
            static_inner_shape=tensor_shape.TensorShape([None] *
                                                        new_inner_rank),
            validate=False)
    else:
      if self.rank is None or stop < self.rank:
        partial = self._slice_shape(0, stop)
      else:
        partial = self

      for x in partial.row_partitions:
        if not x.is_uniform():
          raise ValueError("All relevant dimensions must be uniform")
      if partial.rank is None:
        # TODO(martinz): Implement _with_num_row_partitions(0) if rank is
        # unknown, and remove.
        raise NotImplementedError(
            "__getitem__[start:stop] where start > 0 not implemented")

      return DynamicRaggedShape._from_inner_shape(
          partial._with_num_row_partitions(0).inner_shape[start:])

  def _dimension(self, index):
    # [FUNCTION DOC]
    # HELPER: implements integer indexing for the indexing operator.
    # Returns the size of a single uniform dimension as a scalar tensor (constant when the size is
    # statically known). Input: index, a non-negative int. Raises a type error if index is not an
    # int, an index error if it is negative or not below a known rank, and a value error if the
    # dimension is ragged or if the rank is unknown and the index is large.
    """Return a dimension, if the dimension is not ragged (see __getitem__)."""
    rank = self.rank
    if not isinstance(index, int):
      raise TypeError("index should be an int")
    if (self.num_row_partitions == 0 or index > self.num_row_partitions + 1):
      # If num_row_partitions > 0 and index <= num_row_partitions + 1, then
      # we are safe.
      if rank is None:
        raise ValueError(
            "Rank must be known to use __getitem__ on a large index.")
      if index >= rank:
        raise IndexError("Index is too big: " + str(index) + ">=" + str(rank))
    if index < 0:
      raise IndexError("Index must be non-negative: " + str(index))
    elif not self.is_uniform(index):
      raise ValueError("Index " + str(index) + " is not uniform")
    elif index == 0 and self.num_row_partitions > 0:
      static_nrows = self.row_partitions[0].static_nrows
      if static_nrows is not None:
        return constant_op.constant(static_nrows, dtype=self.dtype)
      return self.row_partitions[0].nrows()
    elif self.num_row_partitions == 0:
      static_result = tensor_shape.dimension_value(
          self._static_inner_shape[index])
      if static_result is not None:
        return constant_op.constant(static_result, dtype=self.dtype)
      return self.inner_shape[index]
    elif index > self.num_row_partitions:
      static_result = tensor_shape.dimension_value(
          self._static_inner_shape[index - self.num_row_partitions])
      if static_result is not None:
        return constant_op.constant(static_result, dtype=self.dtype)

      return self.inner_shape[index - self.num_row_partitions]
    else:
      return self.row_partitions[index - 1].uniform_row_length()

  def __getitem__(self, index):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring does not mention that a type error is raised for an index that is
    # neither an int nor a slice, and that strided slices raise an index error.
    # Indexing operator. Input: index, an int (possibly negative when rank is known) or a slice with
    # no step or a step of one. An int returns the size of that uniform dimension as a scalar
    # tensor; a slice returns a new shape for the selected dimensions (a slice not starting at zero
    # gives a dense shape). Raises an index error for strided slices, a value error for negative
    # indices with unknown rank or ragged dimensions, and a type error otherwise.
    """Returns a dimension or a slice of the shape.

    Ragged shapes can have ragged dimensions that depend upon other dimensions.
    Therefore, if you ask for a dimension that is ragged, this function returns
    a ValueError. For similar reasons, if a slice is selected that includes
    a ragged dimension without including the zero dimension, then this fails.

    Any slice that does not start at zero will return a shape
    with num_row_partitions == 0.

    Args:
      index: the index: can be an int or a slice.

    Raises:
      IndexError: if the index is not in range.
      ValueError: if the rank is unknown, or a ragged rank is requested
      incorrectly.
    """
    rank = self.rank
    if isinstance(index, slice):

      if (index.step is not None) and (index.step != 1):
        raise IndexError("Cannot stride through a shape")
      start = index.start
      stop = index.stop
      if start is None:
        start = 0
      start = _fix_start_index(start, rank, self.num_row_partitions)
      stop = _fix_stop_index(stop, rank)
      return self._slice_shape(start, stop)
    elif isinstance(index, int):
      if index < 0:
        if rank is None:
          raise ValueError(
              "Rank must be known to use __getitem__ with a negative index.")
        return self._dimension(rank + index)
      return self._dimension(index)
    else:
      raise TypeError("Argument is not an int or a slice")

  def _num_elements(self):
    # [FUNCTION DOC]
    # HELPER: total element count of the flat values of this shape.
    # Returns a scalar tensor: the product of the inner shape entries.
    """Number of elements in a shape.

    Returns:
      The number of elements in the shape.

    """
    return math_ops.reduce_prod(self.inner_shape)

  def _num_slices_in_dimension(self, axis):
    # [FUNCTION DOC]
    # HELPER: used when building broadcasters and merging dimensions.
    # KNOWN ISSUE: the docstring example repeats the line for axis minus two with two different
    # results (the last one should be for axis minus three).
    # Returns the total number of slices up to and including an axis (like the number of values at
    # that axis). Input: axis, an int; negative values count from the end and need a known rank.
    # Raises a type error if axis is not an int and a value error for a negative axis with unknown
    # rank. Axis zero gives the first dimension, axes inside the row partitions give the value
    # count of the corresponding partition, and deeper axes give the product of the leading inner
    # shape entries.
    """The total size of a dimension (like nvals).

    Effectively, this is self[:axis+1]._num_elements()

    Example:
    shape = DynamicRaggedShape._from_inner_shape([2, 3, 4])
    shape._num_slices_in_dimension(0) = 2
    shape._num_slices_in_dimension(1) = 6
    shape._num_slices_in_dimension(2) = 24
    shape._num_slices_in_dimension(-1) = 24
    shape._num_slices_in_dimension(-2) = 6
    shape._num_slices_in_dimension(-2) = 2

    Args:
      axis: the last axis to include in the number of elements. If negative,
        then axis = axis + rank.

    Returns:
      The number of elements in the shape.
    """
    if not isinstance(axis, int):
      raise TypeError("axis must be an integer")
    if axis < 0:
      rank = self.rank
      if rank is None:
        raise ValueError(
            "You can't use negative values if the rank is undefined")
      axis = axis + rank
    if axis == 0:
      return self._dimension(0)
    if axis <= self.num_row_partitions:
      return self.row_partitions[axis - 1].nvals()
    # If self.num_row_partitions = 1, and
    # self.inner_shape=[3,5,6], and axis=2, then you want:
    # 15 = 3 * 5 = math_ops.reduce_prod(self.inner_shape[:2])
    # 2 = axis - (self.num_row_partitions - 1)
    # If num_row_partitions=0, and
    # self.inner_shape=[3,5,6] and axis=2, then you want:
    # 90 = 3 * 5 * 6 = math_ops.reduce_prod(self.inner_shape[:3])
    # 3 = axis - (self.num_row_partitions - 1)
    remainder = axis - (self.num_row_partitions - 1)
    return _reduce_prod_patch(self.inner_shape[:remainder])

  def is_uniform(self, axis):
    # [FUNCTION DOC]
    # Returns true if the given dimension is uniform (not ragged). Input: axis, a non-negative
    # int. Raises a type error if it is not an int, and an index error for a negative axis or an
    # axis at or beyond a known rank. Axis zero and axes inside the inner shape are always
    # uniform; other axes are answered by the corresponding row partition.
    """Returns true if the indicated dimension is uniform."""
    if not isinstance(axis, int):
      raise TypeError("axis must be an integer")
    rank = self.rank
    if axis < 0:
      raise IndexError("Negative axis values are not supported")
    elif rank is not None and axis >= rank:
      raise IndexError("Expected axis=%s < rank=%s" % (axis, rank))
    else:
      return ((axis == 0 or axis > len(self._row_partitions))  # pylint:disable=superfluous-parens
              or self._row_partitions[axis - 1].is_uniform())

  @property
  def rank(self):
    # [FUNCTION DOC]
    # Property returning the number of dimensions, the number of row partitions plus the inner
    # rank, or nothing if the inner rank is unknown.
    """The number of dimensions in this shape, or None if unknown."""
    inner_rank = self.inner_rank
    if inner_rank is None:
      return None
    else:
      return self.num_row_partitions + inner_rank

  @property
  def inner_shape(self):
    # [FUNCTION DOC]
    # Property returning the inner shape as a one-dimensional integer tensor.
    """The inner dimension sizes for this shape.

    Returns:
      A 1-D integer `Tensor`.
    """
    return self._inner_shape

  @property
  def inner_rank(self):
    # [FUNCTION DOC]
    # Property returning the rank of the inner shape (statically known length of the inner shape
    # vector), or nothing if unknown.
    """The rank of inner_shape."""
    return tensor_shape.dimension_value(self._static_inner_shape.rank)

  def _alt_inner_shape(self, new_inner_rank):
    # [FUNCTION DOC]
    # HELPER: used to re-factor a shape with a different number of row partitions.
    # Returns an inner shape tensor equivalent to this one but having the requested inner rank.
    # Input: new_inner_rank, a positive int. Raises a value error if the new or old inner rank is
    # zero, or if raising the inner rank would swallow a ragged partition. Lower ranks fold the
    # leading dimensions into one (statically when fully defined); higher ranks unfold using the
    # uniform row lengths of the trailing partitions.
    """Get an alternative inner shape with higher or lower rank.

    For the rank of the inner shape to be be higher, the last few ragged
    dimensions must have uniform_row_length.

    Args:
      new_inner_rank: the new rank of the inner_shape

    Returns:
       A new inner_shape of rank new_inner_rank.
    """
    if new_inner_rank == 0:
      raise ValueError("new_inner_rank cannot be zero")
    elif self.inner_rank == 0:
      raise ValueError("old inner_rank cannot be zero")
    elif new_inner_rank == self.inner_rank:
      return self.inner_shape
    elif new_inner_rank < self.inner_rank:
      if self._static_inner_shape.is_fully_defined():
        return _alt_inner_shape_from_tensor_shape(self._static_inner_shape,
                                                  self.dtype, new_inner_rank)
      first_dimension = self._num_slices_in_dimension(-new_inner_rank)
      if new_inner_rank == 1:
        return array_ops.expand_dims(first_dimension, 0)
      remaining_dimensions = self.inner_shape[1 - new_inner_rank:]
      return array_ops.concat(
          [array_ops.expand_dims(first_dimension, 0), remaining_dimensions],
          axis=0)
    else:
      assert new_inner_rank > self.inner_rank
      new_dimensions = new_inner_rank - self.inner_rank
      if any(
          [not x.is_uniform() for x in self.row_partitions[-new_dimensions:]]):
        raise ValueError("Cannot get an inner shape over a ragged dimension")
      first_dimension = self._num_slices_in_dimension(-new_inner_rank)
      new_dimensions = new_inner_rank - self.inner_rank
      new_dims = [first_dimension] + [
          x.uniform_row_length() for x in self.row_partitions[-new_dimensions:]
      ]
      return array_ops.concat(
          [array_ops_stack.stack(new_dims), self.inner_shape[1:]], axis=0)

  def _inner_shape_dim(self, dimension):
    # [FUNCTION DOC]
    # HELPER: used when converting inner dimensions into uniform row partitions.
    # Returns the inner shape entry at the given position as a plain int when statically known,
    # otherwise as a scalar tensor.
    """Returns an int or a tensor representing _inner_shape[dimension]."""
    result = tensor_shape.dimension_value(self._static_inner_shape[dimension])
    return self._inner_shape[dimension] if result is None else result

  def _with_inner_rank(self, inner_rank):
    # [FUNCTION DOC]
    # HELPER: expresses the inner-rank view in terms of the number of row partitions.
    # Returns the same shape with the given inner rank. Raises a value error if the rank is
    # unknown, or if the rank is below two and a different inner rank is requested. Otherwise
    # delegates to the re-factoring by number of row partitions.
    """Returns the same shape but a different inner_rank.

    All dimensions that are to be represented in the inner_shape must be dense.
    See inner_rank.

    Args:
      inner_rank: the new inner_rank of the shape.

    Returns:
      the same shape but a different inner_rank

    Raises:
      ValueError if the new dense rank is invalid, or the old rank is unknown.
    """
    rank = self.rank
    if rank is None:
      raise ValueError("Rank must be known to adjust inner_rank")
    elif rank < 2:
      if inner_rank == rank:
        return self
      raise ValueError("Cannot change inner_rank if rank < 2")
    else:
      # When self.rank is not None:
      # self.rank = self.inner_rank + self.num_row_partitions
      new_num_row_partitions = rank - inner_rank
      return self._with_num_row_partitions(new_num_row_partitions)

  def _with_num_row_partitions(self, num_row_partitions):
    # [FUNCTION DOC]
    # HELPER: central re-factoring routine for the shape and its broadcasting.
    # Returns an equivalent shape with exactly the requested number of row partitions. Input:
    # num_row_partitions, a non-negative int below the rank. Returns self when unchanged. To
    # increase it, converts leading inner dimensions into uniform row partitions; to decrease it,
    # folds trailing partitions back into the inner shape. Raises a value error if the rank is
    # unknown, the argument is not an int, negative, or not smaller than the rank (and a ragged
    # partition cannot be folded into the inner shape, via the inner shape helper).
    """Creates an identical shape with the given num_row_partitions.

    Note that the shape must be statically refactorable to this rank.
    In particular:
    * rank must be known.
    * num_row_partitions must be a nonnegative int.
    * num_row_partitions must be less than the rank of the shape
    * num_row_partitions must be greater or equal to the index of any ragged
    dimension.

    Note that if the num_row_partitions is the same, self is returned.

    Args:
      num_row_partitions: the target num_row_partitions (must be a nonnegative
        int).

    Returns:
      a shape with a (possibly) different num_row_partitions.

    Raises:
      ValueError: if the rank is unknown, the argument is not a nonnegative int,
        or there is a dimension that is nonuniform.
    """
    rank = self.rank
    if rank is None:
      raise ValueError("Rank must be known to adjust num_row_partitions")
    if not isinstance(num_row_partitions, int):
      raise ValueError("num_row_partitions must be an int")
    if num_row_partitions < 0:
      raise ValueError("num_row_partitions must be nonnegative")
    if num_row_partitions == self.num_row_partitions:
      return self
    if num_row_partitions >= rank:
      raise ValueError("num_row_partitions must be less than rank")
    if num_row_partitions > self.num_row_partitions:
      num_row_partitions_diff = num_row_partitions - self.num_row_partitions
      new_inner_rank = self.rank - num_row_partitions
      nvals = self._inner_shape_dim(0)
      more_rp = []
      for i in range(num_row_partitions_diff):
        nrows = nvals
        row_length = self._inner_shape_dim(i + 1)
        nvals = nrows * row_length
        rp = RowPartition.from_uniform_row_length(
            row_length, nrows=nrows, dtype=self.dtype)
        more_rp.append(rp)
      alt_inner = self._alt_inner_shape(new_inner_rank)
      return DynamicRaggedShape(list(self.row_partitions) + more_rp, alt_inner)
    else:
      assert num_row_partitions < self.num_row_partitions
      return DynamicRaggedShape(
          self.row_partitions[:num_row_partitions],
          self._alt_inner_shape(self.rank - num_row_partitions))

  def _merge_dims(self, outer_axis: int,
                  inner_axis: int) -> "DynamicRaggedShape":
    # [FUNCTION DOC]
    # HELPER: used by ragged tensor merge operations to compute the merged shape.
    # Returns a copy of the shape in which the dimensions from outer_axis to inner_axis (inclusive,
    # possibly negative with known rank) are flattened into one. Raises a value error if
    # outer_axis is greater than inner_axis, if axes are invalid for the rank, or if the rank of
    # the inner shape is unknown when merging across the boundary. Equal axes return self. Handles
    # four cases: dense shape, merge within row partitions, merge within inner shape, and merge
    # across both (using row splits composition and a product of inner dimensions).
    """Merges outer_axis...inner_axis into a single dimension.

    Returns a copy of this shape with the specified range of dimensions
    flattened into a single dimension, with elements in row-major order.

    #### Examples:

    >>> tf.experimental.DynamicRaggedShape.from_lengths([2, (2,1),
    ...     (1,2,3)])._merge_dims(0, 1)
    <DynamicRaggedShape lengths=[3, (1, 2, 3)] num_row_partitions=1>
    >>> tf.experimental.DynamicRaggedShape.from_lengths([2, (2,1),
    ...     (1,2,3)])._merge_dims(1, 2)
    <DynamicRaggedShape lengths=[2, (3, 3)] num_row_partitions=1>
    >>> tf.experimental.DynamicRaggedShape.from_lengths([2, (2,1),
    ...     (1,2,3)])._merge_dims(0, 2)
    <DynamicRaggedShape lengths=[6] num_row_partitions=0>

    To mimic the behavior of `np.flatten` (which flattens all dimensions), use
    `rt.merge_dims(0, -1).  To mimic the behavior of `tf.layers.Flatten` (which
    flattens all dimensions except the outermost batch dimension), use
    `rt.merge_dims(1, -1)`.

    Args:
      outer_axis: `int`: The first dimension in the range of dimensions to
        merge. May be negative if `self.shape.rank` is statically known.
      inner_axis: `int`: The last dimension in the range of dimensions to merge.
        May be negative if `self.shape.rank` is statically known.

    Returns:
      A copy of this shape, with the specified dimensions merged into a
      single dimension.  The returned shape will be
      `self.shape[:outer_axis] + [N] + self.shape[inner_axis + 1:]`, where `N`
      is the total number of slices in the merged dimensions.
    """
    outer_axis = array_ops.get_positive_axis(
        outer_axis, self.rank, axis_name="outer_axis", ndims_name="rank(self)")
    inner_axis = array_ops.get_positive_axis(
        inner_axis, self.rank, axis_name="inner_axis", ndims_name="rank(self)")
    if not outer_axis <= inner_axis:
      raise ValueError(f"Expected outer_axis ({outer_axis}) to be less than or "
                       f"equal to inner_axis ({inner_axis}).")
    if outer_axis == inner_axis:
      return self
    if self.num_row_partitions == 0:
      # A dense tensor.
      (new_inner_shape,
       new_static_inner_shape) = _merge_inner_shape(self._inner_shape,
                                                    self._static_inner_shape,
                                                    outer_axis, inner_axis)
      return DynamicRaggedShape([],
                                new_inner_shape,
                                dtype=self.dtype,
                                static_inner_shape=new_static_inner_shape)
    if inner_axis <= self.num_row_partitions:
      # Here, we are merging the row_partitions,
      # but the inner_shape is unchanged.
      if outer_axis == 0:
        # There is no need to merge axes before the first, just truncate them.
        return DynamicRaggedShape(
            self._row_partitions[inner_axis:],
            self.inner_shape,
            dtype=self.dtype,
            static_inner_shape=self._static_inner_shape)
      prefix_rp = self._row_partitions[:outer_axis - 1]
      suffix_rp = self._row_partitions[inner_axis:]
      internal_rp = self._row_partitions[outer_axis - 1:inner_axis]
      new_rp = prefix_rp + (_merge_row_partitions(internal_rp),) + suffix_rp

      return DynamicRaggedShape(
          new_rp,
          self.inner_shape,
          dtype=self.dtype,
          static_inner_shape=self._static_inner_shape)
    elif outer_axis > self.num_row_partitions:
      # In this scenario, only the inner_shape is changed.
      # Example #1:
      # if [2, (1, 2), 5, 3], num_row_partitions=1, outer_axis=2, inner_axis=3.
      # Result: [2, (1, 2), 15], num_row_partitions=1, outer_axis=2,
      #     inner_axis=3.
      (new_inner_shape, new_static_inner_shape) = _merge_inner_shape(
          self._inner_shape, self._static_inner_shape,
          outer_axis - self.num_row_partitions,
          inner_axis - self.num_row_partitions)
      return DynamicRaggedShape(
          self._row_partitions,
          new_inner_shape,
          dtype=self.dtype,
          static_inner_shape=new_static_inner_shape)
    else:
      # Here, both inner_shape and row_partitions are changed.
      rank = self.rank
      if rank is None:
        raise ValueError("Cannot merge_dims of the inner shape if the " +
                         "dimension of inner_shape is unknown")
      if outer_axis == 0:
        new_inner_shape = self._alt_inner_shape(rank - inner_axis)
        return DynamicRaggedShape._from_inner_shape(new_inner_shape)
      else:
        prefix = self._row_partitions[:outer_axis - 1]
        suffix = _merge_row_partitions(self._row_partitions[outer_axis - 1:])
        new_inner_shape = self._alt_inner_shape(rank - inner_axis)
        num_merged_inner = inner_axis - self.num_row_partitions
        prod = _reduce_prod_patch(self._inner_shape[1:num_merged_inner + 1])
        tail_suffix = RowPartition.from_row_splits(suffix.row_splits() * prod)
        return DynamicRaggedShape(prefix + (tail_suffix,), new_inner_shape)

  def with_dtype(self, dtype):
    # [FUNCTION DOC]
    # Returns this shape if the dtype already matches, otherwise a new shape with all row
    # partitions and the inner shape converted to the given dtype (int32 or int64).
    """Change the dtype of the shape."""
    if dtype == self.dtype:
      return self
    else:
      return DynamicRaggedShape(
          self.row_partitions, self.inner_shape, dtype=dtype)

  def _merge_with(self, other: "DynamicRaggedShape") -> "DynamicRaggedShape":
    # [FUNCTION DOC]
    # HELPER: used to combine two shapes known to be equal (e.g. in type merging).
    # Merges two shapes that are equal apart from the number of row partitions. Both are
    # re-factored to the larger number of partitions, the partitions are merged pairwise by their
    # precomputed encodings, and the static inner shapes are merged (raising a value error if
    # incompatible). Returns a new shape created with validation enabled.
    """Merge two shapes that are equal modulo num_row_partitions.

    The resulting num_row_partitions is the maximum of the two
    num_row_partitions.

    Args:
      other: a DynamicRaggedShape representing the same shape with a possibly
        different number of row partitions.

    Returns:
      A DynamicRaggedShape with the same shape and the maximum of the
      num_row_partitions of the two shapes.
    """
    max_num_row_partitions = max(self.num_row_partitions,
                                 other.num_row_partitions)
    a = self._with_num_row_partitions(max_num_row_partitions)
    b = other._with_num_row_partitions(max_num_row_partitions)
    new_row_partitions = [
        rp_a._merge_precomputed_encodings(rp_b)
        for (rp_a, rp_b) in zip(a._row_partitions, b._row_partitions)
    ]
    new_dtype = b.dtype if a.dtype == dtypes.int32 else dtypes.int64

    new_static_inner_shape = a._static_inner_shape.merge_with(
        b._static_inner_shape)
    new_inner_shape = a._inner_shape
    return DynamicRaggedShape(new_row_partitions, new_inner_shape, new_dtype,
                              True, new_static_inner_shape)

  def _merge_with_spec(
      self, other: "DynamicRaggedShape.Spec") -> "DynamicRaggedShape":
    # [FUNCTION DOC]
    # HELPER: used to combine a concrete shape with a shape spec.
    # Like the merge of two shapes but the other argument is a DynamicRaggedShape spec. Both are
    # re-factored to the larger number of row partitions, the partitions are merged with the spec
    # partitions, and static inner shapes are merged. Returns a new shape built with validation.
    # A TODO in the code notes there are no tests for dynamic inconsistencies.
    """Merge a spec with a DynamicRaggedShape."""
    # TODO(martinz): add tests for dynamic inconsistencies.
    max_num_row_partitions = max(self.num_row_partitions,
                                 other.num_row_partitions)
    a = self._with_num_row_partitions(max_num_row_partitions)
    b = other._with_num_row_partitions(max_num_row_partitions)
    new_row_partitions = [
        rp_a._merge_with_spec(rp_b)
        for (rp_a, rp_b) in zip(a._row_partitions, b._row_partitions)
    ]
    new_dtype = b.dtype if a.dtype == dtypes.int32 else dtypes.int64

    new_static_inner_shape = a._static_inner_shape.merge_with(
        b._static_inner_shape)
    new_inner_shape = a._inner_shape
    return DynamicRaggedShape(new_row_partitions, new_inner_shape, new_dtype,
                              True, new_static_inner_shape)

  def _as_row_partitions(self):
    # [FUNCTION DOC]
    # HELPER: used by broadcasting, which needs every dimension expressed as a partition.
    # Returns the row partitions of an equivalent shape with rank minus one partitions (every axis
    # but the innermost is a partition). Raises a value error if the rank is unknown or below one.
    """Returns row partitions representing this shape.

    In order to represent a shape as row partitions, the rank of the shape
    must be known, and the shape must have rank at least one.

    Returns:
      A list of RowPartition objects.
    Raises:
      ValueError, if the shape cannot be represented by RowPartitions.
    """
    rank = self.rank
    if rank is None:
      raise ValueError("rank must be known for _as_row_partitions")
    elif rank < 1:
      raise ValueError("rank must be >= 1 for _as_row_partitions")
    fully_ragged = self._with_num_row_partitions(rank - 1)
    return fully_ragged.row_partitions

  def _validate_flat_values_dynamically(self, flat_values):
    # [FUNCTION DOC]
    # HELPER: used by flat value validation when static checking is impossible.
    # If the shape has row partitions, returns the flat values wrapped with an assertion that its
    # first dimension equals the last partition's value count; otherwise returns it unchanged.
    """Test if flat_values have the right nvals dynamically."""
    if self.row_partitions:
      assert_op = check_ops.assert_equal(
          self.row_partitions[-1].nvals(),
          array_ops.shape(flat_values, out_type=self.dtype)[0],
          message="Last row partition does not match flat_values.")
      return control_flow_ops.with_dependencies([assert_op], flat_values)
    return flat_values

  def _validate_flat_values(self, flat_values):
    # [FUNCTION DOC]
    # HELPER: used by _add_row_partitions when validation is requested.
    # KNOWN ISSUE: the check for a shape of nothing is dead code because a tensor always has a
    # shape object; unknown rank is handled only by the later unknown first dimension test.
    # Checks that flat values match the last row partition's value count. Non-tensor inputs are
    # returned unchanged. If the sizes are statically known and differ, raises a value error;
    # if unknown, falls back to a dynamic assertion. Returns the (possibly guarded) flat values.
    """Test if flat_values have the right nvals."""
    if not isinstance(flat_values, tensor_lib.Tensor):
      return flat_values
    if self.row_partitions:
      last_row_partition = self.row_partitions[-1]
      flat_values_shape = flat_values.shape
      if flat_values_shape is None:
        return self._validate_flat_values_dynamically(flat_values)
      first_dim_flat_values = flat_values_shape[0]
      if isinstance(first_dim_flat_values, tensor_shape.Dimension):
        first_dim_flat_values = first_dim_flat_values.value
      if first_dim_flat_values is None:
        return self._validate_flat_values_dynamically(flat_values)
      static_nvals = last_row_partition.static_nvals
      if static_nvals is None:
        return self._validate_flat_values_dynamically(flat_values)
      if first_dim_flat_values != static_nvals:
        raise ValueError("Last row partition does not match flat_values.")
    return flat_values

  def _add_row_partitions(self, flat_values, validate=False):
    # [FUNCTION DOC]
    # HELPER: used by broadcasting and elementwise operations to rebuild ragged results.
    # Wraps flat values with this shape's row partitions to form a ragged tensor; with no
    # partitions returns the flat values unchanged. Input: flat_values, tensor; validate, default
    # false, whether to validate the first dimension of flat values. Returns a ragged or dense
    # tensor. Validation may raise a value error.
    """Add row partitions to flat_values, if necessary.

    If the shape is truly ragged, then this adds the row_partitions.

    The shape is dense, then this just returns flat_values.

    Args:
      flat_values: the flat_values of a ragged tensor with this shape, or a
        dense tensor with this shape.
      validate: validate the flat_values have the right first dimension.

    Returns:
      flat_values reshaped to have row_partitions.
    """
    if self.row_partitions:
      if validate:
        flat_values = self._validate_flat_values(flat_values)
      return ragged_tensor.RaggedTensor._from_nested_row_partitions(
          flat_values, self.row_partitions, validate=False)
    else:
      return flat_values

  class Spec:
    # [CLASS DOC]
    # Nested type specification for DynamicRaggedShape, similar to a static shape. Holds the
    # tuple of RowPartitionSpec objects, the static inner shape, and a vector spec for the
    # inner shape (whose length is the inner rank and whose dtype is the shape dtype). Used by
    # tf.function tracing and dataset code to describe shapes without values. Methods compute
    # dimensions, merge specs, change the number of partitions and truncate. It is created by the
    # constructor, by from_value, or by the private factory methods.
    """A Spec for DynamicRaggedShape: similar to a static shape."""

    def __init__(self, row_partitions: Tuple[RowPartitionSpec, ...],
                 static_inner_shape: tensor_shape.TensorShape,
                 dtype: dtypes.DType):
      # [FUNCTION DOC]
      # Constructor of the spec. Inputs: row_partitions, an iterable of RowPartitionSpec; static_inner_shape,
      # anything convertible to a TensorShape; dtype, int32 or int64. Validates and coerces the
      # arguments: raises a type error if the partitions are not an iterable of specs, a value error
      # if the dtype is not int32 or int64, if any partition dtype differs, or if partitions are given
      # with an inner rank of zero. When the inner rank is not zero, propagates known row counts,
      # value counts and uniform row lengths forward and backward through the partitions (merging
      # compatible information) and rebuilds the partitions and the static inner shape accordingly.
      # Sets the stored static inner shape, an inner shape tensor spec, and row partition sequence.
      """Create a Spec given row partitions, a static inner shape, and a dtype.

      Args:
        row_partitions: A sequence of `RowPartitionSpec`s describing how the
          ragged shape is partitioned.
        static_inner_shape: The static shape of the flat_values.
        dtype: The DType used to encode the shape (tf.int64 or tf.int32).
      """
      # Independent validation and coercion of each argument.
      if not isinstance(row_partitions, Iterable):
        raise TypeError("row_partitions should be an Iterable")

      row_partitions = tuple(row_partitions)

      static_inner_shape = tensor_shape.as_shape(static_inner_shape)

      dtype = dtypes.as_dtype(dtype)

      if not all(isinstance(rp, RowPartitionSpec) for rp in row_partitions):
        raise TypeError(
            "row_partitions should be an Iterable of RowPartitionSpecs")

      if dtype != dtypes.int32 and dtype != dtypes.int64:
        raise ValueError("dtype must be tf.int32 or tf.int64")

      # All fields are now typechecked and internally consistent.
      for spec in row_partitions:
        if spec.dtype != dtype:
          raise ValueError(
              f"dtype of {spec!r} is {spec.dtype!r}: expected {dtype!r}")

      row_partitions = tuple(row_partitions)

      inner_rank = static_inner_shape.rank

      if inner_rank == 0:
        if row_partitions:
          raise ValueError(
              "If row_partitions are provided, must have inner_rank > 0")
      else:
        num_slices_in_dimension = []  # type: Sequence[tensor_shape.Dimension]

        # We first attempt to calculate num_slices_in_dimension through a
        # forward pass, using nrows[k] = nrows[k-1] * uniform_row_length
        # and other tricks.
        for i in range(len(row_partitions)):
          rp = row_partitions[i]
          result = tensor_shape.Dimension(rp.nrows)
          if i > 0:
            previous_rp = row_partitions[i - 1]
            result = result.merge_with(previous_rp.nvals)
            result = result.merge_with(num_slices_in_dimension[-1] *
                                       previous_rp.uniform_row_length)
          num_slices_in_dimension.append(result)
        # In the last step of the forward pass,
        # we combine nvals and the first dimension in static_inner_shape.
        if row_partitions:
          last_rp = row_partitions[-1]
          result = (num_slices_in_dimension[-1] *
                    last_rp.uniform_row_length).merge_with(last_rp.nvals)
          if inner_rank is not None:
            result = result.merge_with(
                tensor_shape.dimension_at_index(static_inner_shape, 0))
            static_inner_shape = result + static_inner_shape[1:]
          num_slices_in_dimension.append(result)

        # Now, we start a backward pass.
        for i in range(len(num_slices_in_dimension) - 1, 0, -1):
          num_slices_in_dimension[i - 1] = num_slices_in_dimension[
              i - 1].merge_with(
                  _safe_floor_div(num_slices_in_dimension[i],
                                  row_partitions[i - 1].uniform_row_length))

        # Finally, we construct the partitions.
        row_partitions = [
            RowPartitionSpec(  # pylint: disable=g-complex-comprehension
                nrows=num_slices_in_dimension[i].value,
                uniform_row_length=rp.uniform_row_length,
                nvals=num_slices_in_dimension[i + 1].value,
                dtype=rp.dtype) for i, rp in enumerate(row_partitions)
        ]

      self._static_inner_shape = static_inner_shape
      self._inner_shape = tensor_lib.TensorSpec([inner_rank], dtype=dtype)
      self._row_partitions = row_partitions

    def __repr__(self):
      # [FUNCTION DOC]
      # Returns a string with the row partitions, static inner shape and dtype of the spec.
      return (
          f"DynamicRaggedShape.Spec(row_partitions={self._row_partitions!r}, " +
          f"static_inner_shape={self._static_inner_shape!r}, " +
          f"dtype={self.dtype!r})")

    @classmethod
    def from_value(cls, value: Any) -> "DynamicRaggedShape.Spec":
      # [FUNCTION DOC]
      # Class method creating a spec from a DynamicRaggedShape value. Calls the base class factory
      # to get unvalidated properties and then builds the spec through the validating constructor
      # using the row partitions, static inner shape and the dtype of the inner shape. Returns a spec.
      """Create a Spec from a DynamicRaggedShape."""
      # super().from_value(...) creates an object, but there is no validation.
      # No methods can be trusted on the object, just the properties.
      initial = super(DynamicRaggedShape.Spec, cls).from_value(value)

      # However, since value is a DynamicRaggedShape, we
      # can guarantee that initial._inner_shape.shape.rank == 1

      # Moreover, if inner_shape.shape[0] is not None, then
      # static_inner_shape.rank is not None.

      return DynamicRaggedShape.Spec(
          row_partitions=initial._row_partitions,
          static_inner_shape=initial._static_inner_shape,
          dtype=initial._inner_shape.dtype)

    # TODO(martinz): it is unclear what the default uniformity of RowPartitions
    # should be, so I am moving this to experimental until we figure it out.
    # Also, while I have specified this is meant to represent a shape of a
    # proper Tensor instead of a RaggedTensor, this is also subject to
    # interpretation.
    @classmethod
    def _from_tensor_shape(cls, shape: Any, num_row_partitions: int,
                           dtype: dtypes.DType) -> "DynamicRaggedShape.Spec":
      # [FUNCTION DOC]
      # HELPER: factory used by batching, truncation and re-factoring of specs.
      # Class method creating a spec from a TensorShape (coming from a tensor spec). Inputs: shape,
      # anything convertible to a TensorShape; num_row_partitions, int, how many leading dimensions
      # become uniform row partitions; dtype, int32 or int64. Raises a value error for another
      # dtype, if partitions are requested for a scalar or vector, or if the number of partitions is
      # not less than the rank. Unknown rank gives partitions of unknown content and unknown inner
      # shape. Otherwise each partition gets its row count, value count and uniform row length from
      # the shape dimensions. Returns a spec.
      """Creates a `DynamicRaggedShape.Spec` corresponding to a `tf.TensorShape`.

      It is assumed that this is a `tf.TensorShape` coming from a
      `tf.TensorSpec`, not from `RaggedTensor.shape`.

      In addition to the shape, we need to know the number of row partitions,
      and the dtype used in the shape (tf.int32 or tf.int64).

      Within the dimensions that are partitioned, all dimensions are assumed
      to be uniform.

      Args:
        shape: a TensorShape.
        num_row_partitions: the ragged rank of the RaggedShape.
        dtype: the dtype of the shape (not the tensor); tf.int64 or tf.int32.

      Returns:
        a DynamicRaggedShape.Spec representing a TensorShape.
      """
      if dtype != dtypes.int32 and dtype != dtypes.int64:
        raise ValueError("dtype must be tf.int32 or tf.int64")

      shape = tensor_shape.as_shape(shape)
      if shape.rank is None:
        row_partitions = [
            RowPartitionSpec(dtype=dtype) for _ in range(num_row_partitions)
        ]
        return DynamicRaggedShape.Spec(
            row_partitions=row_partitions,
            static_inner_shape=tensor_shape.TensorShape(None),
            dtype=dtype)

      if shape.rank <= 1:
        # Create a scalar or vector shape.
        if num_row_partitions:
          raise ValueError("num_row_partitions should be zero " +
                           "if shape is a scalar or vector.")
        return DynamicRaggedShape.Spec(
            row_partitions=[], static_inner_shape=shape, dtype=dtype)

      if shape.rank <= num_row_partitions:
        raise ValueError("num_row_partitions must be less than rank")

      num_elements_so_far = tensor_shape.dimension_value(shape[0])
      rp_specs = []
      for i in range(num_row_partitions):
        current_dim = tensor_shape.dimension_value(shape[i + 1])
        if current_dim is None or num_elements_so_far is None:
          nvals = None
        else:
          nvals = num_elements_so_far * current_dim
        rp_specs.append(
            RowPartitionSpec(
                nrows=num_elements_so_far,
                nvals=nvals,
                uniform_row_length=current_dim,
                dtype=dtype))
        num_elements_so_far = nvals

      static_inner_shape = tensor_shape.TensorShape(
          [num_elements_so_far]) + shape[num_row_partitions + 1:]
      return DynamicRaggedShape.Spec(
          row_partitions=rp_specs,
          static_inner_shape=static_inner_shape,
          dtype=dtype)

    @classmethod
    def _from_spec(
        cls,
        spec: Union["DynamicRaggedShape.Spec", ragged_tensor.RaggedTensorSpec,
                    tensor_lib.TensorSpec],
        dtype: dtypes.DType = dtypes.int64) -> "DynamicRaggedShape.Spec":
      # [FUNCTION DOC]
      # HELPER: derives the shape spec for a value described by a tensor or ragged tensor spec.
      # KNOWN ISSUE: the docstring calls the method from_spec, but it is named with a leading
      # underscore; also nothing is returned (None) and no error raised if the spec is of any type
      # other than the three handled, and the dtype argument is ignored for ragged tensor specs.
      # Class method. Input: spec, either a shape spec (returned unchanged), a ragged tensor spec
      # (uses its shape, ragged rank and row splits dtype), or a tensor spec (no partitions, uses
      # the dtype argument, default int64). Returns a shape spec.
      """Create a TypeSpec for the shape of an object with a given TypeSpec.

      I.e., if `x_spec = tf.type_spec_from_value(x)`, then
      `DynamicRaggedShape.from_spec(x_spec)` returns a TypeSpec compatible with
      `tf.type_spec_from_value(tf.shape(x))`.

      >>> rt = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
      >>> rt_spec = tf.type_spec_from_value(rt)
      >>> rt_shape = DynamicRaggedShape.from_tensor(rt)

      >>> shape_spec_1 = tf.type_spec_from_value(rt_shape)
      >>> shape_spec_2 = DynamicRaggedShape.Spec._from_spec(rt_spec)
      >>> assert shape_spec_1.is_compatible_with(shape_spec_2)

      Args:
        spec: a Spec of a Tensor or RaggedTensor.
        dtype: the default dtype (if necessary).

      Returns:
        A Spec of the shape of a Tensor or RaggedTensor.

      """
      # TODO(martinz): Add StructuredTensor.Spec when its easy.
      if isinstance(spec, DynamicRaggedShape.Spec):
        return spec
      elif isinstance(spec, ragged_tensor.RaggedTensorSpec):
        return cls._from_tensor_shape(spec.shape, spec.ragged_rank,
                                      spec.row_splits_dtype)
      elif isinstance(spec, tensor_lib.TensorSpec):
        return cls._from_tensor_shape(
            shape=spec.shape, num_row_partitions=0, dtype=dtype)

    @property
    def dtype(self) -> dtypes.DType:
      # [FUNCTION DOC]
      # Property returning the dtype of the shape described by this spec, taken from the inner shape
      # tensor spec.
      return self._inner_shape.dtype

    @property
    def inner_rank(self) -> Optional[int]:
      # [FUNCTION DOC]
      # Property returning the inner rank of the spec: the rank of the static inner shape if known,
      # else the static length of the inner shape vector spec, else nothing.
      if self._static_inner_shape.rank is not None:
        return self._static_inner_shape.rank
      if self._inner_shape.shape.rank is None:
        return None
      return tensor_shape.dimension_value(self._inner_shape.shape[0])

    @property
    def num_row_partitions(self) -> int:
      # [FUNCTION DOC]
      # Property returning the number of row partition specs.
      return len(self._row_partitions)

    @property
    def rank(self) -> Optional[int]:
      # [FUNCTION DOC]
      # Property returning the rank, number of partitions plus inner rank, or nothing if the inner
      # rank is unknown.
      inner_rank = self.inner_rank
      return None if inner_rank is None else inner_rank + self.num_row_partitions

    def _dimension(self, index: int) -> Optional[int]:
      # [FUNCTION DOC]
      # HELPER: used by spec code to query a statically known dimension.
      # Returns the statically known size of the dimension at the given index, or nothing if not
      # known. Dimension zero is the row count of the first partition, or the first inner dimension
      # when there are none; later dimensions inside the partitions are uniform row lengths, and the
      # rest come from the static inner shape. Raises a value error when the index is out of range
      # for a known inner rank.
      """Get the size of dimension index, if known statically."""
      if index == 0:
        if self._row_partitions:
          return self._row_partitions[0].nrows
        elif self.inner_rank is None:
          return None
        elif self.inner_rank == 0:
          raise ValueError("Index out of range: 0.")
        else:
          return tensor_shape.dimension_value(self._static_inner_shape[0])
      if index <= len(self._row_partitions):
        return self._row_partitions[index - 1].uniform_row_length

      relative_index = index - self.num_row_partitions

      if self.inner_rank is None:
        return None
      elif self.inner_rank <= relative_index:
        raise ValueError(f"Index out of range: {index}.")
      else:
        return tensor_shape.dimension_value(
            self._static_inner_shape[relative_index])

    def _num_slices_in_dimension(self, axis: int) -> Optional[int]:
      # [FUNCTION DOC]
      # HELPER: static counterpart of the shape's number-of-slices computation.
      # KNOWN ISSUE: the docstring example uses field names with leading underscores that are not
      # constructor arguments and omits a comma between the partition specs, so it is illustrative only.
      # Returns the statically known number of values up to and including an axis, or nothing if
      # unknown. Input: axis, an int (negative counts from the end, needs known rank). Raises a type
      # error if not an int. Axis zero gives the first dimension; axes within the partitions give the
      # partition value count; deeper axes give the element count of the leading static inner shape.
      """The total size of a dimension (like nvals).

      This is a static version of DynamicRaggedShape._num_slices_in_dimension()

      Example:

      ```
      shape = DynamicRaggedShape.Spec(
        _row_partitions=[
          RowPartitionSpec(nrows=3, nvals=14, dtype=tf.int32)
          RowPartitionSpec(nrows=14, nvals=25, dtype=tf.int32)

        ],
        _static_inner_shape=tf.TensorShape([25, 3, 4]),
        _inner_shape=tf.TensorSpec(tf.TensorShape([3]), dtype=tf.int32))
      shape._num_slices_in_dimension(0) = 3
      shape._num_slices_in_dimension(1) = 14
      shape._num_slices_in_dimension(2) = 25
      shape._num_slices_in_dimension(3) = 3
      shape._num_slices_in_dimension(4) = 4
      shape._num_slices_in_dimension(-2) = 3
      ```

      Args:
        axis: the last dimension to include.

      Returns:
        the number of values in a dimension.
      """
      if not isinstance(axis, int):
        raise TypeError("axis must be an integer")
      axis = array_ops.get_positive_axis(axis, self.rank, ndims_name="rank")

      if axis == 0:
        return self._dimension(0)
      if axis <= self.num_row_partitions:
        # TODO(martinz): use nvals OR nrows, whichever is defined.
        return self._row_partitions[axis - 1].nvals
      remainder = axis - (self.num_row_partitions - 1)
      head_inner_shape = self._static_inner_shape[:remainder]
      return head_inner_shape.num_elements()

    def with_dtype(self, dtype: dtypes.DType) -> "DynamicRaggedShape.Spec":
      # [FUNCTION DOC]
      # Returns a copy of the spec with all partition specs and the shape dtype converted to the
      # given dtype.
      """Return the same spec, but with a different DType."""
      new_rp_specs = [rp.with_dtype(dtype) for rp in self._row_partitions]
      return DynamicRaggedShape.Spec(
          row_partitions=new_rp_specs,
          static_inner_shape=self._static_inner_shape,
          dtype=dtype)

    def _merge_with(
        self, other: "DynamicRaggedShape.Spec") -> "DynamicRaggedShape.Spec":
      # [FUNCTION DOC]
      # HELPER: combines static information of two specs of the same shape.
      # Re-factors both specs to the larger number of row partitions, merges the partitions pairwise
      # and the static inner shapes (raising a value error on incompatible information), and returns
      # a new spec. The dtype is taken from the other spec if this one is int32, otherwise int64.
      # Specs of different rank fail through the re-factoring step.
      """Merges all information between two specs.

      Specs are expected to represent the same information modulo
      num_row_partitions.

      If the specs are of different ranks, then fail.

      Args:
        other: another Spec of the same rank.

      Returns:
        a Spec with the union of information.
      """
      max_num_row_partitions = max(self.num_row_partitions,
                                   other.num_row_partitions)
      a = self._with_num_row_partitions(max_num_row_partitions)
      b = other._with_num_row_partitions(max_num_row_partitions)

      new_rp = [
          a._merge_with(b)
          for (a, b) in zip(a._row_partitions, b._row_partitions)
      ]

      new_static_inner_shape = a._static_inner_shape.merge_with(
          b._static_inner_shape)

      dtype = b.dtype if (a.dtype == dtypes.int32) else dtypes.int64

      return DynamicRaggedShape.Spec(
          new_rp, new_static_inner_shape, dtype=dtype)

    def _with_num_row_partitions(
        self, new_num_row_partitions: int) -> "DynamicRaggedShape.Spec":
      # [FUNCTION DOC]
      # HELPER: re-factoring routine for specs, mirror of the shape method.
      # Returns an equivalent spec with the requested number of partitions. Raises a value error if
      # the rank is unknown, or the number is larger than the rank minus one or negative. Returns
      # self when unchanged. Adding partitions converts leading inner dimensions to uniform
      # partition specs; removing them folds the tail back into the static inner shape.
      """Change the number of row partitions in the spec."""
      rank = self.rank
      if rank is None:
        raise ValueError(
            "Changing num_row_partitions with unknown rank unsupported")
      if new_num_row_partitions > max(rank - 1, 0):
        raise ValueError("Number of row partitions too large")
      if new_num_row_partitions < 0:
        raise ValueError("Number of row partitions negative")
      if self.num_row_partitions == new_num_row_partitions:
        return self
      elif self.num_row_partitions < new_num_row_partitions:
        # TODO(martinz): Consider swapping.
        rp_delta = new_num_row_partitions - self.num_row_partitions
        tail_shape = DynamicRaggedShape.Spec._from_tensor_shape(
            self._static_inner_shape, rp_delta, self.dtype)
        return DynamicRaggedShape.Spec(
            row_partitions=self._row_partitions + tail_shape._row_partitions,
            static_inner_shape=tail_shape._static_inner_shape,
            dtype=self.dtype)
      else:
        assert self.num_row_partitions > new_num_row_partitions
        new_row_partitions = self._row_partitions[:new_num_row_partitions]
        last_row_partition = new_row_partitions[-1]
        old_row_partitions = self._row_partitions[new_num_row_partitions:]
        new_static_inner_shape = (
            tensor_shape.TensorShape(
                [last_row_partition.nvals] +
                [x.uniform_row_length for x in old_row_partitions]) +
            self._static_inner_shape[1:])
        return DynamicRaggedShape.Spec(new_row_partitions,
                                       new_static_inner_shape, self.dtype)

    def _set_rank_if_unknown(self, new_rank: int) -> "DynamicRaggedShape.Spec":
      # [FUNCTION DOC]
      # HELPER: used by truncation to give specs with unknown rank a concrete rank.
      # Ensures the spec has a known rank. Input: new_rank, an int. Raises a type error if nothing,
      # a value error if negative or if the current known rank is smaller. Returns self if the rank
      # is known; otherwise returns a spec whose static inner shape has unknown dimensions of enough
      # length (the first being the last partition's value count if partitions exist).
      """Ensures this has a known rank at least new_rank."""
      if new_rank is None:
        raise TypeError("new_rank is None, but expected int")
      if new_rank < 0:
        raise ValueError("Rank must be non-negative")
      current_rank = self.rank
      if current_rank is not None and current_rank < new_rank:
        raise ValueError(
            "Rank is {current_rank}, expected at least {new_rank}.".format(
                current_rank=current_rank, new_rank=new_rank))

      if current_rank is not None:
        return self

      if self._row_partitions:
        new_inner_rank = max(new_rank - self.num_row_partitions, 1)
        first_dim = self._row_partitions[-1].nvals
        static_inner_shape = tensor_shape.TensorShape([first_dim] + [None] *
                                                      (new_inner_rank - 1))
      else:
        static_inner_shape = tensor_shape.TensorShape([None] * new_rank)

      return DynamicRaggedShape.Spec(
          row_partitions=self._row_partitions,
          static_inner_shape=static_inner_shape,
          dtype=self.dtype)

    def _truncate(self, new_rank: int) -> "DynamicRaggedShape.Spec":
      # [FUNCTION DOC]
      # HELPER: used by code that needs to cut a spec to its first dimensions.
      # KNOWN ISSUE: the docstring refers to a function named truncate_dynamic_ragged_shape_spec,
      # which does not exist; this is the method that does it. No check that new_rank is not larger
      # than the rank.
      # Returns a spec for the first new_rank dimensions. Unknown rank is first fixed. Zero gives a
      # scalar spec, one gives a vector with the first dimension, a rank within the partitions
      # keeps the leading partitions, otherwise the inner shape is cut.
      """Truncate a ragged shape spec.

      For example, if the original spec s was for a shape:
      [3, [4, 1], 2, 7]

      Then truncate_dynamic_ragged_shape_spec(s, 3) is a spec for:
      [3, [4, 1], 2]

      Args:
        new_rank: the new rank

      Returns:
        A truncated DynamicRaggedShape.Spec.
      """
      if self.rank is None:
        return self._set_rank_if_unknown(new_rank)._truncate(new_rank)

      if new_rank == 0:
        return DynamicRaggedShape.Spec._from_tensor_shape([], 0, self.dtype)

      if new_rank == 1:
        vector_size = self._dimension(0)
        return DynamicRaggedShape.Spec._from_tensor_shape([vector_size], 0,
                                                          self.dtype)

      if new_rank < self.num_row_partitions + 1:
        new_row_partitions = self._row_partitions[:new_rank - 1]
        new_static_inner_shape = tensor_shape.TensorShape(
            [new_row_partitions[-1].nvals])
        return DynamicRaggedShape.Spec(
            row_partitions=new_row_partitions,
            static_inner_shape=new_static_inner_shape,
            dtype=self.dtype)
      else:
        remainder = new_rank - self.num_row_partitions
        new_static_inner_shape = self._static_inner_shape[:remainder]
        return DynamicRaggedShape.Spec(
            row_partitions=self._row_partitions,
            static_inner_shape=new_static_inner_shape,
            dtype=self.dtype)

    def _to_tensor_shape(self):
      # [FUNCTION DOC]
      # HELPER: used by encoders to get the static shape of the described tensor.
      # Returns a TensorShape: unknown if the inner rank is unknown; the inner shape itself for a
      # scalar; otherwise the first dimension, the uniform row lengths and the remaining inner
      # dimensions (ragged dimensions are unknown).
      """Get a tensor shape corresponding to this type."""
      alt = self
      if alt._static_inner_shape.rank is None:
        return tensor_shape.TensorShape(None)
      if alt._static_inner_shape.rank == 0:
        assert not alt._row_partitions
        return alt._static_inner_shape
      prefix = [alt._dimension(0)]
      prefix.extend([rp.uniform_row_length for rp in alt._row_partitions])
      suffix = alt._static_inner_shape[1:]
      return tensor_shape.TensorShape(prefix) + suffix


def broadcast_dynamic_shape(shape_x: DynamicRaggedShape,
                            shape_y: DynamicRaggedShape) -> DynamicRaggedShape:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring lists only a value error, but type errors are raised if either
  # argument is not a DynamicRaggedShape.
  # Public function returning the shape produced by broadcasting two shapes together. Inputs:
  # shape_x and shape_y, both DynamicRaggedShape. Returns a DynamicRaggedShape. Raises a type
  # error for wrong argument types, and a value error if dtypes conflict, ranks are unknown, or
  # the shapes are not compatible (compatibility errors may also appear at run time as
  # assertion failures). Delegates to the extended version and keeps only the shape.
  """Returns the shape formed by broadcasting two shapes to be compatible.

  1. If shape_x and shape_y both have row_partitions, then fail if their dtypes
     don't match.
  2. If neither has row_partitions and they have different dtypes,
     go with int64.
  3. If one has row_partitions, go with that dtype.

  Args:
    shape_x: A `DynamicRaggedShape`
    shape_y: A `DynamicRaggedShape`

  Returns:
    A `DynamicRaggedShape`.
  Raises:
    ValueError: If `shape_x` and `shape_y` are not broadcast-compatible.
  """
  if not isinstance(shape_x, DynamicRaggedShape):
    raise TypeError("shape_x must be a DynamicRaggedShape")
  if not isinstance(shape_y, DynamicRaggedShape):
    raise TypeError("shape_y must be a DynamicRaggedShape")

  return broadcast_dynamic_shape_extended(shape_x, shape_y)[0]


def broadcast_to(rt_input, shape: DynamicRaggedShape):
  # [FUNCTION DOC]
  # Public function broadcasting a tensor or ragged tensor to a target shape. Inputs: rt_input,
  # anything convertible to a tensor or ragged tensor; shape, a DynamicRaggedShape. Returns a
  # tensor or ragged tensor with the target shape. Raises a type error if shape is not a
  # DynamicRaggedShape and a value error if row split dtypes cannot be reconciled. Dtype rules:
  # for a ragged input with partitions in the target they must agree; otherwise the target is
  # cast. Behaviour for incompatible shapes is undefined (may fail at run time). Builds a
  # broadcaster from the input's shape to the target and applies it.
  """Broadcasts a potentially ragged tensor to a ragged shape.

  Tiles `rt_input` as necessary to match the given shape.

  Behavior is undefined if `rt_input` is not broadcast-compatible with `shape`.

  Args:
    rt_input: The potentially ragged tensor to broadcast.
    shape: A `DynamicRaggedShape`

  Returns:
    A potentially ragged tensor whose values are taken from
    `rt_input`, and whose shape matches `shape`.
  """
  if not isinstance(shape, DynamicRaggedShape):
    raise TypeError("shape must be a DynamicRaggedShape")
  rt_input = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt_input)
  origin_shape = None
  if ragged_tensor.is_ragged(rt_input):
    if shape.num_row_partitions != 0:
      if rt_input.row_splits.dtype != shape.dtype:
        raise ValueError("Cannot coerce row_splits.dtype")
    else:
      shape = shape.with_dtype(rt_input.row_splits.dtype)
    origin_shape = DynamicRaggedShape.from_tensor(rt_input)
  else:
    if shape.num_row_partitions != 0:
      origin_shape = DynamicRaggedShape.from_tensor(rt_input, dtype=shape.dtype)
    else:
      origin_shape = DynamicRaggedShape.from_tensor(
          rt_input, dtype=dtypes.int64)
      shape = shape.with_dtype(dtype=dtypes.int64)

  broadcaster = _get_broadcaster(origin_shape, shape)
  return broadcaster.broadcast(rt_input)


def broadcast_dynamic_shape_extended(
    a: DynamicRaggedShape, b: DynamicRaggedShape
):  #  -> Tuple[DynamicRaggedShape, _Broadcaster, _Broadcaster]
  # [FUNCTION DOC]
  # Computes the broadcast of two shapes together with the transformations from each input shape
  # to the result. Inputs: a and b, DynamicRaggedShape. Returns a tuple of the result shape and
  # two broadcaster objects (a to result and b to result). Reconciles dtypes (error if both have
  # partitions with different dtypes, otherwise follows the one with partitions or int64).
  # Raises a value error for unknown rank or mismatched dtypes. Handles scalar operands, two
  # vectors, and general case where the lower-rank shape is aligned to the trailing dimensions
  # of the other. Compatibility may also be enforced by run-time assertions.
  """Gets the smallest shape to which a and b can broadcast.

  In order to create the smallest shape, one must also do most of the
  work to figure out how to transform from the shapes given. Thus, in addition
  to returning the shape, it also creates transformations from the
  original shapes to the result.

  This is the equivalent of:

  c = broadcast_dynamic_shape(a, b)
  ac = get_broadcaster(a, c)
  bc = get_broadcaster(b, c)
  return (c, ac, bc)

  Args:
    a: a DynamicRaggedShape
    b: a DynamicRaggedShape

  Returns:
    A triple of a shape and two broadcasters.
  """
  if a.row_partitions and b.row_partitions:
    if a.dtype != b.dtype:
      raise ValueError("Dtypes don't match")
  elif a.dtype != b.dtype:
    if a.row_partitions:
      b = b.with_dtype(a.dtype)
    elif b.row_partitions:
      a = a.with_dtype(b.dtype)
    else:
      a = a.with_dtype(dtypes.int64)
      b = b.with_dtype(dtypes.int64)

  if (a.rank is None or b.rank is None):
    raise ValueError("Unable to broadcast: unknown rank")
  elif a.rank == 0:
    return (b, _Broadcaster(a, b, []), _get_identity_broadcaster(b))
  elif b.rank == 0:
    return (a, _get_identity_broadcaster(a), _Broadcaster(b, a, []))
  elif a.rank == 1 and b.rank == 1:
    [a_layer, b_layer,
     target] = _broadcast_dynamic_shape_one_layer(a.inner_shape, b.inner_shape)
    target_shape = DynamicRaggedShape._from_inner_shape(target)  # pylint: disable=protected-access
    return (target_shape, _Broadcaster(a, target_shape, [a_layer]),
            _Broadcaster(b, target_shape, [b_layer]))

  if a.rank > b.rank:
    (c, bc, ac) = _broadcast_dynamic_shape_extended_helper(b, a)  # pylint: disable=arguments-out-of-order

    return (c, ac, bc)

  return _broadcast_dynamic_shape_extended_helper(a, b)


def _row_partitions_identical(shape_a, shape_b):
  # [FUNCTION DOC]
  # HELPER: fast path check for elementwise operations on ragged tensors.
  # Returns true if both shapes have the same number of row partitions and each pair is the
  # same object (identity, not equality of values).
  """Returns True iff all row_partitions in shapes are identical."""
  return ((shape_a.num_row_partitions == shape_b.num_row_partitions) and all(
      a is b for a, b in zip(shape_a.row_partitions, shape_b.row_partitions)))


# TODO(martinz): Preserve shapes better (see CL/414806185)
@dispatch.dispatch_for_binary_elementwise_apis(ragged_tensor.RaggedOrDense,
                                               ragged_tensor.RaggedOrDense)
def ragged_binary_elementwise_op_impl(op, x, y):
  # [FUNCTION DOC]
  # Handler registered with the dispatch system for all binary elementwise operations when
  # arguments are ragged or dense. Inputs: op, the underlying elementwise function; x and y,
  # anything convertible to tensors or ragged tensors. Converts arguments (with preferred dtype
  # from the other), matches row split dtypes, and if either is ragged with compatible flat
  # rank: when the row partitions are identical applies op to the flat values and rewraps;
  # otherwise broadcasts the flat values of both to the common shape and applies op, wrapping the
  # result with the common shape (validated). In other cases applies op to the flat values and
  # reuses the ragged structure of whichever argument is ragged. A plain boolean result (e.g.
  # from tensor equality) is returned as is. Returns a ragged tensor or tensor.
  """Binary elementwise api handler for RaggedTensors."""
  x_is_ragged = ragged_tensor.is_ragged(x)
  y_is_ragged = ragged_tensor.is_ragged(y)

  # Convert args to tensors.
  x = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      x, preferred_dtype=(y.dtype if y_is_ragged else None))
  y = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      y, preferred_dtype=x.dtype)

  if x_is_ragged and y_is_ragged:
    x, y = ragged_tensor.match_row_splits_dtypes(x, y)

  if ((x_is_ragged and y_is_ragged) or
      (x_is_ragged and x.flat_values.shape.ndims <= y.shape.ndims) or
      (y_is_ragged and y.flat_values.shape.ndims <= x.shape.ndims)):
    shape_x = DynamicRaggedShape.from_tensor(x)
    shape_y = DynamicRaggedShape.from_tensor(y)
    if shape_x.dtype != shape_y.dtype:
      if not x_is_ragged:
        shape_x = shape_x.with_dtype(shape_y.dtype)
      elif not y_is_ragged:
        shape_y = shape_y.with_dtype(shape_x.dtype)

    if _row_partitions_identical(shape_x, shape_y):
      # At this point, both x and y must be ragged.
      return shape_x._add_row_partitions(  # pylint: disable=protected-access
          op(x.flat_values, y.flat_values),
          validate=False)

    (shape_z, bcast_xz,
     bcast_yz) = broadcast_dynamic_shape_extended(shape_x, shape_y)
    x_new_flat = bcast_xz.broadcast_flat_values(x, inner_dimensions=False)
    y_new_flat = bcast_yz.broadcast_flat_values(y, inner_dimensions=False)
    z_flat = op(x_new_flat, y_new_flat)
    return shape_z._add_row_partitions(z_flat, validate=True)  # pylint: disable=protected-access

  x_values = x.flat_values if ragged_tensor.is_ragged(x) else x
  y_values = y.flat_values if ragged_tensor.is_ragged(y) else y
  mapped_values = op(x_values, y_values)
  if isinstance(mapped_values, bool):
    return mapped_values  # Special case for tensor_equals.
  if ragged_tensor.is_ragged(x):
    return x.with_flat_values(mapped_values)
  else:
    return y.with_flat_values(mapped_values)


@dispatch.dispatch_for_binary_elementwise_assert_apis(
    ragged_tensor.RaggedOrDense, ragged_tensor.RaggedOrDense)
def ragged_binary_elementwise_assert_op_impl(op, x, y):
  # [FUNCTION DOC]
  # Handler registered with the dispatch system for binary assert operations on ragged tensors.
  # Same argument handling and broadcasting logic as the elementwise handler, but it returns
  # the result of applying op (an assert operation) to the aligned flat values and does not build
  # a ragged tensor. Inputs: op, x, y. Returns the assertion op result. May raise errors for
  # incompatible shapes or dtypes.
  """Binary elementwise assert api handler for RaggedTensors.

  This handles binary assert operations for ragged tensors. Compared with
  `ragged_binary_elementwise_op_impl`, this handler does not compute a ragged
  tensor as output. Instead, it applies the assert operation `op` to input
  tensors based on their ragged shapes and flat_values, and returns the result
  of the assertion operation.

  Args:
    op: a binary assert operation on Tensors.
    x: something that can be coerced to a Tensor or RaggedTensor.
    y: something that can be coerced to a Tensor or RaggedTensor.

  Returns:
    the result of the assertion operation.

  """
  x_is_ragged = ragged_tensor.is_ragged(x)
  y_is_ragged = ragged_tensor.is_ragged(y)

  # Convert args to tensors.
  x = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      x, preferred_dtype=(y.dtype if y_is_ragged else None))
  y = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      y, preferred_dtype=x.dtype)

  if x_is_ragged and y_is_ragged:
    x, y = ragged_tensor.match_row_splits_dtypes(x, y)

  if ((x_is_ragged and y_is_ragged) or
      (x_is_ragged and x.flat_values.shape.ndims <= y.shape.ndims) or
      (y_is_ragged and y.flat_values.shape.ndims <= x.shape.ndims)):
    shape_x = DynamicRaggedShape.from_tensor(x)
    shape_y = DynamicRaggedShape.from_tensor(y)
    if shape_x.dtype != shape_y.dtype:
      if not x_is_ragged:
        shape_x = shape_x.with_dtype(shape_y.dtype)
      elif not y_is_ragged:
        shape_y = shape_y.with_dtype(shape_x.dtype)

    if _row_partitions_identical(shape_x, shape_y):
      # At this point, both x and y must be ragged.
      return op(x.flat_values, y.flat_values)

    (_, bcast_xz, bcast_yz) = broadcast_dynamic_shape_extended(shape_x, shape_y)
    x_new_flat = bcast_xz.broadcast_flat_values(x, inner_dimensions=False)
    y_new_flat = bcast_yz.broadcast_flat_values(y, inner_dimensions=False)
    return op(x_new_flat, y_new_flat)

  x_values = x.flat_values if ragged_tensor.is_ragged(x) else x
  y_values = y.flat_values if ragged_tensor.is_ragged(y) else y
  return op(x_values, y_values)


def _find_dtype_helper(value, preferred):
  # [FUNCTION DOC]
  # HELPER: used by _find_dtype to read a dtype from various objects.
  # KNOWN ISSUE: the explicit branch for tensors repeats the final fallback, so it is redundant;
  # objects without a dtype attribute (e.g. strings) raise an attribute error.
  # Returns the preferred dtype if given; the dtype of a row partition, a dtype itself or a
  # tensor; nothing for ints, lists and tuples; otherwise the object's dtype attribute.
  """Helper for _find_dtype."""
  if preferred is not None:
    return preferred
  elif isinstance(value, RowPartition):
    return value.dtype
  elif isinstance(value, dtypes.DType):
    return value
  elif isinstance(value, int):
    return None
  elif isinstance(value, list):
    return None
  elif isinstance(value, tuple):
    return None
  elif isinstance(value, core.Tensor):
    return value.dtype
  return value.dtype


def _find_dtype(value, preferred):
  # [FUNCTION DOC]
  # HELPER: dtype inference for the shape constructors.
  # Returns preferred if not nothing, else the dtype of value (nothing for plain python
  # containers and ints). Raises a value error if the result is not int32, int64 or nothing.
  """Returns the preferred dtype of value or preferred if preferred != None.

  This is used as an operator to pass over multiple objects in decreasing order
  of priority until there is a preferred dtype for one. For example, if you were
  adding three tensor-ish things (some tensors, some lists), and needed a
  preferred dtype, you could use this as:

  def adding(a, b, c, dtype = None):
    dtype = _find_dtype(a, dtype)
    dtype = _find_dtype(b, dtype)
    dtype = _find_dtype(c, dtype)
    if dtype is None:
      dtype = tf.float32
    ...Code continues here...

  Args:
    value: a list, value, RowPartition, or tensor.
    preferred: a given dtype. If not None, this will be returned.

  Returns:
    an optional dtype.
  """
  result = _find_dtype_helper(value, preferred)
  if (result == dtypes.int64 or result == dtypes.int32 or result is None):
    return result
  raise ValueError("Illegal dtype: " + str(result))


def _find_dtype_iterable(
    iterable: Iterable[Any],
    dtype: Optional[dtypes.DType]) -> Optional[dtypes.DType]:
  # [FUNCTION DOC]
  # HELPER: dtype inference over a sequence of objects.
  # Returns the given dtype if not nothing; otherwise scans the iterable and returns the dtype
  # of the first item that has one (nothing if none). Errors from item inference propagate.
  """Find the preferred dtype of a list of objects.

  This will go over the iterable, and use the first object with a preferred
  dtype. The dtype passed has highest priority if it is not None.

  Args:
    iterable: an iterable with things that might have a dtype.
    dtype: an overriding dtype, or None.

  Returns:
    an optional dtype.
  """
  if dtype is not None:
    return dtype
  for x in iterable:
    dtype = _find_dtype(x, dtype)
  return dtype


class _LayerBroadcaster(abc.ABC):
  # [CLASS DOC]
  # HELPER: internal building block for shape broadcasting.
  # Abstract base for broadcasting a single layer (axis) of a shape: a mapping from each
  # destination index to the source index it copies from, called the gather index. Subclasses
  # must supply the gather index and dtype conversion and dependency handling; other methods
  # (broadcast of tensors and row partitions, next layer computation, factories) are implemented
  # on top of it. Class methods create instances of the concrete gather based subclass.
  """A broadcaster of a single layer.

  Although this class does not literally contain a gather_index, the reference
  implementation is defined through a gather_index. Thus, any subclasses should
  first define the gather_index property. Other functions can be overridden
  for optimization, but it should not change the behavior.
  """

  @property
  @abc.abstractmethod
  def gather_index(self):
    # [FUNCTION DOC]
    # Abstract property: a one-dimensional tensor whose size equals the destination size and whose
    # ith element is the source index for the ith destination element.
    """Returns a 1D tensor.

    The size of the 1D tensor is equal to the destination size.

    The ith element of the result is the index of the source of the ith element.
    """
    pass

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the dtype of the broadcast, taken from the gather index.
    """Returns the dtype of the broadcast."""
    return self.gather_index.dtype

  @abc.abstractmethod
  def with_dtype(self, dtype):
    # [FUNCTION DOC]
    # Abstract method: returns an identical layer broadcaster using the given dtype.
    """Returns an identical _LayerBroadcaster with a different dtype."""
    pass

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the string form of the gather index tensor.
    return str(self.gather_index)

  @classmethod
  def from_gather_index(cls, gather_index):
    # [FUNCTION DOC]
    # HELPER: factory for layer broadcasters.
    # Class method creating a gather based layer broadcaster from a gather index tensor.
    """Create a broadcaster from a gather_index."""
    return _GatherLayerBroadcaster(gather_index)

  @classmethod
  def first_layer(cls, nrows_source, nrows_target):
    # [FUNCTION DOC]
    # HELPER: factory for the outermost layer.
    # KNOWN ISSUE: the docstring is a copy of the one for the gather index factory; actually this
    # builds the gather index from row counts (zeros if the source has one row, range if the
    # counts are equal, with a run-time assertion otherwise) and wraps it.
    # Class method. Inputs: nrows_source and nrows_target, scalar integer tensors. Returns a layer
    # broadcaster. A run-time assertion fails with a cannot-broadcast message if neither rule fits.
    """Create a broadcaster from a gather_index."""
    gather_index = _first_layer_gather_index(nrows_source, nrows_target)
    return _LayerBroadcaster.from_gather_index(gather_index)

  @classmethod
  def get_singleton_broadcaster(cls, target_size):
    # [FUNCTION DOC]
    # HELPER: factory for broadcasting a single element.
    # Class method returning a layer broadcaster in which every one of target_size destination
    # elements maps to source element zero. Input: target_size, scalar integer tensor.
    """Broadcast from 1 element to target_size elements."""
    return _LayerBroadcaster.from_gather_index(
        array_ops.zeros(target_size, dtype=target_size.dtype))

  @abc.abstractmethod
  def with_dependencies(self, checks):
    # [FUNCTION DOC]
    # Abstract method: returns a copy of the broadcaster with control dependencies on the given
    # list of ops so they run before the tensors of the broadcaster are used.
    """Add dependencies to a _LayerBroadcaster.

    Args:
      checks: a list of ops that need to be run before any tensors from the
        Broadcaster are used.

    Returns:
      a copy of this _LayerBroadcaster with dependencies added.
    """
    pass

  @classmethod
  def get_identity_broadcaster(cls, nvals, dtype=None):
    # [FUNCTION DOC]
    # HELPER: factory for no-op broadcasting.
    # Class method creating a broadcaster mapping each index to itself. Inputs: nvals, the number
    # of values; dtype, optional (otherwise follows nvals). Returns a gather based broadcaster. A
    # TODO notes this could be implemented more efficiently.
    """Create an identity broadcaster.

    TODO(martinz): an identity broadcaster can be far more efficient than a
    generic broadcaster. Add an optimized implementation.
    Args:
      nvals: the number of values for the broadcaster.
      dtype: the dtype of the broadcaster, or None to use the dtype of nvals.

    Returns:
      an identity broadcaster from [0....nvals-1] to [0...nvals-1]
    """
    return _GatherLayerBroadcaster(math_ops.range(nvals, dtype=dtype))

  def broadcast_tensor(self, tensor):
    # [FUNCTION DOC]
    # Broadcasts a dense tensor along its first axis (indexed by the source) so that the first
    # axis is indexed by the destination, by gathering with the gather index. Input: tensor.
    # Returns a tensor.
    """Broadcast from a dense tensor.

    It is assumed that the first axis of the dense tensor is indexed by the
    source shape, and at the end, the first axis of the dense tensor is
    indexed by the destination shape.

    Args:
      tensor: a dense tensor.

    Returns:
      A dense tensor.
    """
    return array_ops.gather(tensor, self.gather_index)

  def dest_nrows(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it may return nothing when tiling, but it always returns the
    # size of the gather index cast to the broadcaster dtype.
    # Returns a scalar tensor with the number of destination rows (length of the gather index).
    """Return the number of rows in the resulting gather, or None if tiling."""
    return math_ops.cast(
        array_ops.shape(self.gather_index)[0], dtype=self.dtype)

  def broadcast_row_partition(self, rp):
    # [FUNCTION DOC]
    # Broadcasts a row partition through this layer. Input: rp, a row partition of the source.
    # Returns a row partition of the destination. A ragged partition has its row lengths gathered;
    # a uniform partition stays uniform with the same row length and the destination row count.
    """Return a new shape where the rows are broadcasted.

        *--self--->*
        |          |
        rp       result
        |          |
        V          V
        *--------->*

    This is equivalent to:
      return RowPartition.from_row_lengths(self.broadcast(rp.row_lengths()))

    However, if the shape has uniform row length, then that property is
    maintained.

    Args:
      rp: a row partition.

    Returns:
      a RowPartition representing a broadcast version of this row partition.
    """
    if not rp.is_uniform():
      return RowPartition.from_row_lengths(
          self.broadcast_tensor(rp.row_lengths()))
    else:
      return RowPartition.from_uniform_row_length(
          rp.uniform_row_length(),
          nvals=rp.uniform_row_length() * self.dest_nrows(),
          nrows=self.dest_nrows())

  def next_layer(self, original_rp, broadcast_rp):
    # [FUNCTION DOC]
    # Computes the broadcaster for the following axis. Inputs: original_rp, the source row
    # partition, and broadcast_rp, the target row partition. Returns a layer broadcaster whose
    # gather index is computed by the module helper (broadcast when the source row length is one,
    # otherwise a checked copy, with run-time assertions for incompatibility).
    r"""Create the next layer gather_index whether or not a broadcast happens.

       *---------self------->*
       |                     |
    original_rp           broadcast_rp
       |                     |
      \|/                   \|/
       *--next_broadcaster-->*
    Args:
      original_rp: the original row partition.
      broadcast_rp: the target row partition.

    Returns:
      the gather_index for next_broadcaster.

    """
    gather_index = _next_layer_gather_index(self, original_rp, broadcast_rp)
    return _LayerBroadcaster.from_gather_index(gather_index)


class _GatherLayerBroadcaster(_LayerBroadcaster):
  # [CLASS DOC]
  # HELPER: concrete layer broadcaster used throughout broadcasting.
  # Layer broadcaster backed by an explicit gather index tensor, for example a source with
  # slices of sizes one and two broadcast to four slices has a gather index of zero, one, two,
  # zero, one, two when the source has three elements. Holds one attribute with the gather index.
  """Implements _LayerBroadcaster with an explicit gather_index.

  For example, suppose that the source shape is:
  [*],[*,*]
  And the target shape is:
  [*],[*,*],[*],[*,*]
  Then, this can be represented with a map:
  [0,1,2,0,1,2]

  """

  def __init__(self, gather_index):
    # [FUNCTION DOC]
    # Converts the gather index to a tensor and stores it. Raises a value error if its dtype is
    # neither int32 nor int64.
    gather_index = ops.convert_to_tensor(gather_index)
    if (gather_index.dtype != dtypes.int64 and
        gather_index.dtype != dtypes.int32):
      raise ValueError("gather_index must be int64 or int32")
    self._gather_index = gather_index

  @property
  def gather_index(self):
    # [FUNCTION DOC]
    # Property returning the stored gather index tensor.
    return self._gather_index

  def with_dtype(self, dtype):
    # [FUNCTION DOC]
    # Returns a new broadcaster whose gather index is cast to the given dtype.
    return _GatherLayerBroadcaster(math_ops.cast(self._gather_index, dtype))

  def with_dependencies(self, checks):
    # [FUNCTION DOC]
    # Returns a new broadcaster whose gather index depends on the given control ops.
    new_gather_index = control_flow_ops.with_dependencies(
        checks, self._gather_index)
    return _GatherLayerBroadcaster(new_gather_index)


class _Broadcaster:
  # [CLASS DOC]
  # HELPER: internal object returned by the broadcast functions.
  # Represents a transformation from a source shape to a destination shape, holding the source
  # and target DynamicRaggedShape and one layer broadcaster per axis of the source. Can broadcast
  # the flat values of a tensor or the whole tensor. All parts share a dtype.
  """A _Broadcaster represents a transformation from one shape to another.

  It provides a transform for each axis of the source shape to the
  corresponding axis of the destination shape.

  """

  def __init__(self,
               source_shape,
               target_shape,
               layer_broadcasters,
               dtype=None):
    # [FUNCTION DOC]
    # Creates a broadcaster; the docstring says not to call it directly. Inputs: source_shape and
    # target_shape, DynamicRaggedShape; layer_broadcasters, a list of layer broadcasters;
    # dtype, optional preferred dtype (otherwise found from the inputs, falling back to int64).
    # Raises a type error if arguments have the wrong types. All components are converted to the
    # common dtype. The length of the layer list is not checked. Stores three attributes.
    """Create a broadcaster.

    Do not call directly.
    The source_shape, target_shape, and layer_broadcasters are converted
    to have the same dtype.

    Note: source_shape.rank and target_shape.rank must be known.
    Args:
      source_shape: the source DynamicRaggedShape
      target_shape: the target DynamicRaggedShape
      layer_broadcasters: List[_LayerBroadcaster] of length source_shape.rank.
      dtype: the preferred dtype of the broadcaster.

    Raises:
      TypeError: if the input types don't match.
    """
    if not isinstance(source_shape, DynamicRaggedShape):
      raise TypeError("source_shape is not a DynamicRaggedShape")
    if not isinstance(target_shape, DynamicRaggedShape):
      raise TypeError("target_shape is not a DynamicRaggedShape")
    if not isinstance(layer_broadcasters, list):
      raise TypeError("layer_broadcasters not a list: " +
                      str(layer_broadcasters))
    for bc in layer_broadcasters:
      if not isinstance(bc, _LayerBroadcaster):
        raise TypeError("Not a LayerBroadcaster: " + str(bc))

    dtype = _find_dtype(source_shape, dtype)
    dtype = _find_dtype(target_shape, dtype)
    dtype = _find_dtype_iterable(layer_broadcasters, dtype)
    dtype = _find_dtype(dtypes.int64, dtype)
    self._source_shape = source_shape.with_dtype(dtype)
    self._target_shape = target_shape.with_dtype(dtype)
    self._layer_broadcasters = [x.with_dtype(dtype) for x in layer_broadcasters]

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a string with the source shape, target shape and layer broadcasters.
    return ("{src_shape:" + str(self._source_shape) + ", target_shape:" +
            str(self._target_shape) + " layer_broadcasters: " +
            str(self._layer_broadcasters) + "}")

  def with_dtype(self, dtype):
    # [FUNCTION DOC]
    # Returns a copy of this broadcaster using the given dtype.
    """Return a copy of this Broadcaster with a different dtype."""
    return _Broadcaster(self._source_shape, self._target_shape,
                        self._layer_broadcasters, dtype)

  @property
  def source_shape(self):
    # [FUNCTION DOC]
    # Property returning the source shape.
    return self._source_shape

  @property
  def target_shape(self):
    # [FUNCTION DOC]
    # Property returning the target shape.
    return self._target_shape

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the dtype, taken from the source shape.
    return self._source_shape.dtype

  def _target_inner_shape_int32(self):
    # [FUNCTION DOC]
    # HELPER: used by broadcast_flat_values to broadcast dense data.
    # Returns the target inner shape, cast to 32-bit integers if it is 64-bit.
    new_inner_shape = self.target_shape.inner_shape
    if new_inner_shape.dtype == dtypes.int64:
      new_inner_shape = math_ops.cast(new_inner_shape, dtype=dtypes.int32)
    return new_inner_shape

  # pylint:disable=protected-access
  def broadcast_flat_values(self, rt, inner_dimensions=True):
    # [FUNCTION DOC]
    # Broadcasts the flat values of a tensor or ragged tensor to the target shape. Inputs: rt, a
    # dense or ragged tensor; inner_dimensions, default true; if true the result has the shape of
    # the target inner shape, otherwise the result may be smaller but satisfies the properties
    # listed in the docstring. Returns a dense tensor. A scalar target returns the flat values.
    # When the target inner rank exceeds the source rank, the source is reshaped to its dense form
    # and broadcast; otherwise the flat values are reshaped to the target inner rank, gathered
    # through the matching layer broadcaster, and broadcast to the inner shape.
    """flat_values of a ragged tensor broadcast to target_shape.

    If inner_dimensions==True, then the result is a dense tensor with shape
    target_shape.inner_shape, the flat values of the broadcasted shape.

    If you add target_shape.row_partitions, you will get the full broadcasted
    shape.

    If inner_dimensions==False, the result is a dense tensor that satisfies
    certain properties:
    1. broadcast_to(result, target_shape.inner_shape) will give the result
       if inner_dimensions==True.
    2. Either (a) (result.rank < target_shape.inner_rank)
       or (b) (result.shape[0] == target_shape.inner_shape[0]).
    3. result.rank = min(target_shape.inner_rank, rt.rank)
    4. For i < target_shape.inner_rank - 1, and i < rt.rank,
       and if rt.shape[-i]!=1, then result.shape[-i]=target_shape[-i].
    Args:
      rt: a ragged or dense tensor.
      inner_dimensions: if true, broadcast the inner dimensions as well.

    Returns:
      a dense tensor
    """
    if ragged_tensor.is_ragged(rt):
      rt = rt.flat_values
    # If rt was a regular tensor, it is its own flat_values.
    if self.target_shape.rank == 0:
      return rt
    inner_rank = self.target_shape.inner_rank
    if inner_rank > self._source_shape.rank:
      # The dense rank is larger than the whole shape. So, we make the shape
      # dense.
      if self.source_shape.num_row_partitions > 0:
        rt = array_ops.reshape(
            rt, self.source_shape._alt_inner_shape(self.source_shape.rank))
      # rt.rank == self._source_shape.rank < inner_rank
      # Here, property 2a holds.
      if inner_dimensions:
        return array_ops.broadcast_to(rt, self._target_inner_shape_int32())
      return rt
    else:
      if self._source_shape.inner_rank != inner_rank:
        rt = array_ops.reshape(rt,
                               self._source_shape._alt_inner_shape(inner_rank))  # pylint:disable=protected-access
      # After the reshape, rt is flat_values with inner_rank.
      flat_broadcaster = self._layer_broadcasters[-inner_rank]
      rt = flat_broadcaster.broadcast_tensor(rt)
      # Here, property 2b holds.
      if inner_dimensions:
        rt = array_ops.broadcast_to(rt, self._target_inner_shape_int32())
      return rt

  def broadcast(self, rt):
    # [FUNCTION DOC]
    # Broadcasts a tensor or ragged tensor of the source shape to the target shape: broadcasts the
    # flat values and attaches the target's row partitions. Returns a tensor or ragged tensor.
    """Broadcast a tensor of source_shape to target_shape."""
    flat_values = self.broadcast_flat_values(rt)
    return self.target_shape._add_row_partitions(flat_values)  # pylint:disable=protected-access


def _get_layer_broadcasters_from_rps(zero_broadcaster, source_rps, target_rps):
  # [FUNCTION DOC]
  # HELPER: builds per-axis broadcasters for a broadcaster object.
  # Recursively computes the layer broadcasters for each row partition pair, starting with the
  # broadcaster of the rows. Inputs: zero_broadcaster, a layer broadcaster (raises a type error
  # otherwise); source_rps and target_rps, lists of equal length. Returns a list whose first
  # element is zero_broadcaster followed by one for each pair.
  """Get LayerBroadcasters from RowPartitions.

           *--zero_broadcaster->*
           |                    |
         source_rps[0]     target_rps[0]
           |                    |
           V                    V
           *---result[1]------->*
           |                    |
         source_rps[1]     target_rps[1]
           |                    |
           V                    V
           *---result[2]------->*
                  .
                  .
                  .
           *---result[k-1]----->*
           |                    |
         source_rps[k]     target_rps[k]
           |                    |
           V                    V
           *---result[k]------->*

  Note: result[0] = zero_broadcaster

  Args:
    zero_broadcaster: a broadcaster between the source and target row
      partitions' rows, and equal to result[0].
    source_rps: source row partitions.
    target_rps: target row partitions (same length as source_rps).

  Returns:
    result: a list of LayerBroadcasters.
  """
  if not isinstance(zero_broadcaster, _LayerBroadcaster):
    raise TypeError("Not a _LayerBroadcaster: " + str(zero_broadcaster))
  assert len(source_rps) == len(target_rps)
  if not source_rps:
    return [zero_broadcaster]
  next_broadcaster = zero_broadcaster.next_layer(source_rps[0], target_rps[0])
  tail_broadcasters = _get_layer_broadcasters_from_rps(next_broadcaster,
                                                       source_rps[1:],
                                                       target_rps[1:])
  return [zero_broadcaster] + tail_broadcasters


def _get_broadcaster(source_shape, target_shape):
  # [FUNCTION DOC]
  # HELPER: used by broadcast_to.
  # Builds a _Broadcaster from a source shape to a target shape. Raises a value error if the
  # dtypes differ, ranks are unknown, or the source has a larger rank. Handles scalar sources,
  # vector targets, a source of lower rank (aligned to trailing target dimensions, using a
  # singleton broadcaster for the dimension before the source) and equal ranks.
  """Get a _Broadcaster from source_shape to target_shape."""
  if source_shape.dtype != target_shape.dtype:
    raise ValueError("The source and target row_split dtypes should be equal")

  if (source_shape.rank is None or target_shape.rank is None):
    raise ValueError("Rank of source and target must be statically known")
  elif source_shape.rank > target_shape.rank:
    raise ValueError("Cannot broadcast to a shape with smaller rank")
  elif source_shape.rank == 0:
    return _Broadcaster(source_shape, target_shape, [])
  elif target_shape.rank == 1:
    assert source_shape.rank == 1
    layer = _LayerBroadcaster.first_layer(source_shape.inner_shape[0],
                                          target_shape.inner_shape[0])
    return _Broadcaster(source_shape, target_shape, [layer])

  assert source_shape.rank <= target_shape.rank
  assert target_shape.rank >= 2
  assert source_shape.rank >= 1

  source_rps = source_shape._as_row_partitions()  # pylint: disable=protected-access

  target_rps = target_shape._as_row_partitions()  # pylint: disable=protected-access

  assert len(target_rps) >= 1
  assert len(source_rps) <= len(target_rps)
  source_nrows = source_shape[0]
  if len(source_rps) < len(target_rps):
    # Note: this includes the case where len(source_rps)==0.
    # Here we begin at -1, one dimension before source_rps[0].
    # neg_one_source_rp  | neg_one_target_rp=target_rps[-(len(source_rps)+1)]
    # source_rps[0]      | target_rps[-len(source_rps)]
    # source_rps[1]      | target_rps[1-len(source_rps)]
    # ...                | ...
    # source_rps[-1]     | target_rps[-1]
    neg_one_source_rp = RowPartition.from_uniform_row_length(
        uniform_row_length=source_nrows, nrows=1, nvals=source_nrows)
    neg_one_target_rp = target_rps[-(len(source_rps) + 1)]
    neg_one_broadcaster = _LayerBroadcaster.get_singleton_broadcaster(
        neg_one_target_rp.nrows())
    zeroth_broadcaster = neg_one_broadcaster.next_layer(neg_one_source_rp,
                                                        neg_one_target_rp)
    target_rps_tail = target_rps[-len(source_rps):] if len(
        source_rps) >= 1 else []

    layers = _get_layer_broadcasters_from_rps(zeroth_broadcaster, source_rps,
                                              target_rps_tail)
    return _Broadcaster(source_shape, target_shape, layers)
  else:
    assert len(target_rps) == len(source_rps)
    zeroth_broadcaster = _LayerBroadcaster.first_layer(source_rps[0].nrows(),
                                                       target_rps[0].nrows())
    layers = _get_layer_broadcasters_from_rps(zeroth_broadcaster, source_rps,
                                              target_rps)

    return _Broadcaster(source_shape, target_shape, layers)


def _get_identity_broadcaster(shape):
  # [FUNCTION DOC]
  # HELPER: used by broadcast_dynamic_shape_extended for scalars.
  # Returns a broadcaster from a shape to itself using identity layers for every axis. Raises
  # a value error if the rank is unknown.
  """Gets a Broadcaster for two identical shapes."""
  if shape.rank is None:
    raise ValueError("Shape must have a defined rank")
  layers = [
      _LayerBroadcaster.get_identity_broadcaster(
          shape._num_slices_in_dimension(i)) for i in range(shape.rank)  # pylint: disable=protected-access
  ]
  return _Broadcaster(shape, shape, layers)


def _broadcast_dynamic_shape_one_layer(a, b):
  # [FUNCTION DOC]
  # HELPER: broadcasting of two vector shapes.
  # Inputs: a and b, one-element shape vectors (row counts). Returns a list of two layer
  # broadcasters and the target shape. If either is statically one it is chosen directly;
  # otherwise uses conditionals with a run-time assertion (invalid-argument error at run
  # time) that the sizes are broadcast compatible.
  """Broadcast two vectors, given their shapes.

  Args:
    a: the number of rows in a.
    b: the number of rows in b.

  Returns:
    (layer_a, layer_b, target_shape)
    layer_a is a _LayerBroadcaster from a to the target_shape.
    layer_b is a _LayerBroadcaster from b to the target_shape.
    target_shape is the target_shape

  Raises:
    InvalidArgumentError if the shapes are not consistent.
  """
  a_0 = a[0]
  b_0 = b[0]

  def broadcast_from_a():
    # [FUNCTION DOC]
    # HELPER: branch of the one-layer vector broadcast.
    # Used when the first operand has size one: returns gather indices (zeros for a, range for b)
    # and the target b.
    # Assumes a_0 == 1
    a_layer = array_ops.zeros(b_0, dtype=b_0.dtype)
    b_layer = math_ops.range(b_0)
    target = b
    return [a_layer, b_layer, target]

  a_static = tensor_util.constant_value(a)
  if a_static is not None and a_static[0] == 1:
    [a_gi, b_gi, target] = broadcast_from_a()
    a_layer = _LayerBroadcaster.from_gather_index(a_gi)
    b_layer = _LayerBroadcaster.from_gather_index(b_gi)
    return [a_layer, b_layer, target]

  def broadcast_from_b():
    # [FUNCTION DOC]
    # HELPER: branch of the one-layer vector broadcast.
    # Used when the second operand has size one: returns gather indices (range for a, zeros for b)
    # and the target a.
    # Assumes b_0 == 1
    a_layer = math_ops.range(a_0)
    b_layer = array_ops.zeros(a_0, dtype=a_0.dtype)
    target = a
    return [a_layer, b_layer, target]

  b_static = tensor_util.constant_value(b)
  if b_static is not None and b_static[0] == 1:
    [a_gi, b_gi, target] = broadcast_from_b()
    a_layer = _LayerBroadcaster.from_gather_index(a_gi)
    b_layer = _LayerBroadcaster.from_gather_index(b_gi)
    return [a_layer, b_layer, target]

  def broadcast_noop():
    # [FUNCTION DOC]
    # HELPER: branch of the one-layer vector broadcast.
    # KNOWN ISSUE: the comment in the body says it assumes the first size is one, but this branch
    # is for equal sizes.
    # Returns identity gather indices for both operands and the target b.
    # Assumes a_0 == 1
    a_layer = math_ops.range(a_0)
    b_layer = math_ops.range(b_0)
    target = b
    return [a_layer, b_layer, target]

  can_broadcast_from_a = math_ops.equal(a_0, 1)
  can_broadcast_from_b = math_ops.equal(b_0, 1)

  def broadcast_not_from_a():
    # [FUNCTION DOC]
    # HELPER: branch of the one-layer vector broadcast.
    # Used when the first operand is not one: conditionally chooses broadcast from b or no-op.
    return cond.cond(
        can_broadcast_from_b, true_fn=broadcast_from_b, false_fn=broadcast_noop)

  nrows_equal = math_ops.equal(a_0, b_0)
  can_broadcast = math_ops.logical_or(
      can_broadcast_from_a,
      math_ops.logical_or(can_broadcast_from_b, nrows_equal))

  check_can_broadcast = check_ops.assert_equal(
      can_broadcast, True, message="Cannot broadcast")

  results = cond.cond(
      can_broadcast_from_a,
      true_fn=broadcast_from_a,
      false_fn=broadcast_not_from_a)

  results = [
      control_flow_ops.with_dependencies([check_can_broadcast], x)
      for x in results
  ]
  [a_gi, b_gi, target] = results
  a_layer = _LayerBroadcaster.from_gather_index(a_gi)
  b_layer = _LayerBroadcaster.from_gather_index(b_gi)
  return [a_layer, b_layer, target]


def _broadcast_dynamic_shape_first_layer(a_0, b_0):
  # [FUNCTION DOC]
  # HELPER: broadcasting of the first axis of two shapes.
  # KNOWN ISSUE: the docstring says it returns a triple starting with a flag for whether the
  # target equals a, but the code returns only the two layer broadcasters.
  # Inputs: a_0 and b_0, scalar row counts. Returns a list with the layer broadcaster of a and
  # of b. Statically known equal sizes or a size of one use identity or singleton broadcasters;
  # otherwise conditionals are used with a run-time assertion that sizes are compatible.
  """Broadcast the first layer of two dynamic shapes given the dimensions.

  Args:
    a_0: the number of rows in a.
    b_0: the number of rows in b.

  Returns:
    (use_a, layer_a, layer_b)
    where use_a is true if the target provably equals a, false otherwise.
    layer_a is a _LayerBroadcaster from a to the target.
    layer_b is a _LayerBroadcaster from b to the target.
  """

  def broadcast_from_a():
    # [FUNCTION DOC]
    # HELPER: branch of the first layer broadcast, for a first operand of size one. Returns gather
    # indices (zeros for a, range for b).
    # Assumes a_0 == 1
    a_layer = array_ops.zeros(b_0, dtype=b_0.dtype)
    b_layer = math_ops.range(b_0)
    return [a_layer, b_layer]

  static_a_0 = tensor_util.constant_value(a_0)
  static_b_0 = tensor_util.constant_value(b_0)
  if static_a_0 is not None:
    if static_a_0 == static_b_0:
      id_broadcaster = _LayerBroadcaster.get_identity_broadcaster(
          static_a_0, dtype=a_0.dtype)
      return [id_broadcaster, id_broadcaster]
    elif static_a_0 == 1:
      return [
          _LayerBroadcaster.get_singleton_broadcaster(b_0),
          _LayerBroadcaster.get_identity_broadcaster(b_0)
      ]

  if static_b_0 == 1:
    return [
        _LayerBroadcaster.get_identity_broadcaster(a_0),
        _LayerBroadcaster.get_singleton_broadcaster(a_0)
    ]

  def broadcast_from_b():
    # [FUNCTION DOC]
    # HELPER: branch of the first layer broadcast, for a second operand of size one. Returns gather
    # indices (range for a, zeros for b).
    # Assumes b_0 == 1
    a_layer = math_ops.range(a_0)
    b_layer = array_ops.zeros(a_0, dtype=a_0.dtype)
    return [a_layer, b_layer]

  def broadcast_noop():
    # [FUNCTION DOC]
    # HELPER: branch of the first layer broadcast, for equal sizes. Returns identity gather indices.
    # Assumes a_0 == b_0
    a_layer = math_ops.range(a_0)
    b_layer = math_ops.range(b_0)
    return [a_layer, b_layer]

  can_broadcast_from_a = math_ops.equal(a_0, constant_op.constant(1, a_0.dtype))
  can_broadcast_from_b = math_ops.equal(b_0, constant_op.constant(1, b_0.dtype))

  def broadcast_not_from_a():
    # [FUNCTION DOC]
    # HELPER: branch of the first layer broadcast, for a first operand not of size one: picks
    # between broadcast from b and no-op with a conditional.
    return cond.cond(
        can_broadcast_from_b, true_fn=broadcast_from_b, false_fn=broadcast_noop)

  # Ideally, this would only block control flow on broadcast_noop, but
  # the control flow doesn't seem to work.
  can_broadcast = math_ops.logical_or(
      math_ops.logical_or(can_broadcast_from_a, can_broadcast_from_b),
      math_ops.equal(a_0, b_0))

  result = cond.cond(
      can_broadcast_from_a,
      true_fn=broadcast_from_a,
      false_fn=broadcast_not_from_a)

  return [
      _LayerBroadcaster.from_gather_index(
          control_flow_ops.with_dependencies(
              [check_ops.assert_equal(can_broadcast, True)], x)) for x in result
  ]


def _broadcast_half(
    ac_0: _LayerBroadcaster,
    a_1: RowPartition) -> Tuple[_LayerBroadcaster, RowPartition]:
  # [FUNCTION DOC]
  # HELPER: half of broadcasting two shapes, the no-op broadcast of one side.
  # Inputs: ac_0, the previous layer broadcaster; a_1, the next row partition of the source.
  # Returns a list of the next layer broadcaster and the broadcast row partition. This cannot
  # fail.
  """Does a NOOP broadcast of a_1.

      *-ac_0-->*
      |        |
     a_1      c_1
      |        |
      V        V
      *-ac_1-->*

  Note that by definition this cannot fail: there is always a well-defined
  NOOP broadcast. This is usually intended as half of broadcasting two shapes
  together.
  Args:
    ac_0: previous LayerBroadcaster
    a_1: previous RowPartition

  Returns:
    [ac_1, c_1] where ac_1 is the next LayerBroadcaster, and c_1 is the
    broadcast RowPartition
  """
  c_1 = ac_0.broadcast_row_partition(a_1)
  old_value_rowids = array_ops.gather(ac_0.gather_index, c_1.value_rowids())
  old_row_starts = array_ops.gather(a_1.row_splits(), old_value_rowids)
  gather_index = old_row_starts + c_1.offsets_in_rows()
  return [_LayerBroadcaster.from_gather_index(gather_index), c_1]


def _broadcast_dynamic_shape_next_layer_half_ragged(
    ac_0: _LayerBroadcaster, bc_0: _LayerBroadcaster, a_1: RowPartition,
    b_1: RowPartition
) -> Tuple[RowPartition, _LayerBroadcaster, _LayerBroadcaster]:
  # [FUNCTION DOC]
  # HELPER: next layer broadcasting where a_1 is uniform and b_1 is ragged.
  # Inputs: ac_0 and bc_0, previous layer broadcasters; a_1 uniform and b_1 ragged row
  # partitions. Raises type errors for wrong types. Returns the target row partition and the
  # two next layer broadcasters. If a_1 has row length one it is expanded to the ragged side;
  # otherwise a conditional chooses between expanding a_1 and a no-op requiring equal row
  # splits, guarded by run-time assertions.
  r"""Broadcast target and next layer broadcaster of two dynamic shapes.

  a_1 is uniform, and b_1 is ragged.
     *--ac_0-->*<--bc_0--*
     |         |         |
    a_1       c_1       b_1
     |         |         |
     V         V         V
     *--ac_1-->*<--bc_1--*

  Args:
    ac_0: _LayerBroadcaster from a to c in the previous layer.
    bc_0: _LayerBroadcaster from b to c in the previous layer.
    a_1: a uniform RowPartition for the next layer of a.
    b_1: a ragged RowPartition for the next layer of b.

  Returns:
    (c_1, ac_1, bc_1)
    c_1: a RowPartition for the next layer of the dynamic shape.
    ac_1: _LayerBroadcaster from a to c in the next layer.
    bc_1: _LayerBroadcaster from b to c in the next layer.
  """
  if not isinstance(ac_0, _LayerBroadcaster):
    raise TypeError("ac_0 should be a _LayerBroadcaster")
  if not isinstance(bc_0, _LayerBroadcaster):
    raise TypeError("bc_0 should be a _LayerBroadcaster")
  if not isinstance(a_1, RowPartition):
    raise TypeError("a_1 should be a RowPartition")
  if not isinstance(b_1, RowPartition):
    raise TypeError("b_1 should be a RowPartition")

  assert a_1.is_uniform()
  assert not b_1.is_uniform()

  static_a_1 = tensor_util.constant_value(a_1.uniform_row_length())
  if static_a_1 == 1:
    [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
    ac_1_gather_index = array_ops.gather(ac_0.gather_index, c_1b.value_rowids())
    c_1 = RowPartition.from_row_splits(c_1b.row_splits())
    ac_1 = _LayerBroadcaster.from_gather_index(ac_1_gather_index)
    bc_1 = _LayerBroadcaster.from_gather_index(bc_1.gather_index)
    return [c_1, ac_1, bc_1]

  def broadcast_noop():
    # [FUNCTION DOC]
    # HELPER: branch of the half ragged broadcast, used when both sides must be equal. Returns the
    # row splits and gather indices, guarded by an assertion that the broadcast row splits agree.
    # The sides must be "equal".
    [ac_1, c_1a] = _broadcast_half(ac_0, a_1)
    [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
    checks = [check_ops.assert_equal(c_1a.row_splits(), c_1b.row_splits())]
    return [
        control_flow_ops.with_dependencies(checks, x)
        for x in [a_1.row_splits(), ac_1.gather_index, bc_1.gather_index]
    ]

  def broadcast_a():
    # [FUNCTION DOC]
    # HELPER: branch of the half ragged broadcast, used when the uniform side has row length one.
    # Returns the row splits of the ragged side and gather indices for both sides.
    [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
    ac_1_gather_index = array_ops.gather(ac_0.gather_index, c_1b.value_rowids())
    return [
        c_1b.row_splits(),
        ac_1_gather_index,
        bc_1.gather_index,
    ]

  can_broadcast_a = math_ops.equal(a_1.uniform_row_length(), 1)

  [c_1_row_splits, ac_1_gather_index,
   bc_1_gather_index] = cond.cond(
       can_broadcast_a, true_fn=broadcast_a, false_fn=broadcast_noop)

  c_1 = RowPartition.from_row_splits(c_1_row_splits)
  ac_1 = _LayerBroadcaster.from_gather_index(ac_1_gather_index)
  bc_1 = _LayerBroadcaster.from_gather_index(bc_1_gather_index)
  return [c_1, ac_1, bc_1]


def _broadcast_dynamic_shape_next_layer_both_uniform(
    ac_0: _LayerBroadcaster, bc_0: _LayerBroadcaster, a_1: RowPartition,
    b_1: RowPartition
) -> Tuple[RowPartition, _LayerBroadcaster, _LayerBroadcaster]:
  # [FUNCTION DOC]
  # HELPER: next layer broadcasting where both partitions are uniform.
  # Inputs: ac_0 and bc_0 layer broadcasters, a_1 and b_1 uniform row partitions. Raises type
  # errors for wrong types. Returns the target row partition and two layer broadcasters.
  # Statically known equal lengths, or a statically known one, are handled directly; otherwise
  # conditionals and run-time assertions check that the lengths are equal or one.
  r"""Broadcast target and next layer broadcaster of two uniform dynamic shapes.

     *--ac_0-->*<--bc_0--*
     |         |         |
    a_1       c_1       b_1
     |         |         |
     V         V         V
     *--ac_1-->*<--bc_1--*

  Args:
    ac_0: _LayerBroadcaster from a to c in the previous layer.
    bc_0: _LayerBroadcaster from b to c in the previous layer.
    a_1: a RowPartition for the next layer of a.
    b_1: a RowPartition for the next layer of b.

  Returns:
    (c_1, ac_1, bc_1)
    c_1: a RowPartition for the next layer of the dynamic shape.
    ac_1: _LayerBroadcaster from a to c in the next layer.
    bc_1: _LayerBroadcaster from b to c in the next layer.
  """
  if not isinstance(ac_0, _LayerBroadcaster):
    raise TypeError("ac_0 should be a _LayerBroadcaster")
  if not isinstance(bc_0, _LayerBroadcaster):
    raise TypeError("bc_0 should be a _LayerBroadcaster")
  if not isinstance(a_1, RowPartition):
    raise TypeError("a_1 should be a RowPartition")
  if not isinstance(b_1, RowPartition):
    raise TypeError("b_1 should be a RowPartition")
  assert a_1.is_uniform()
  assert b_1.is_uniform()

  static_a_1 = tensor_util.constant_value(a_1.uniform_row_length())
  static_b_1 = tensor_util.constant_value(b_1.uniform_row_length())

  if static_a_1 is not None:
    if static_a_1 == static_b_1:
      # Here, this dimension is the same, but we may have to broadcast previous
      # dimensions.
      [ac_1, _] = _broadcast_half(ac_0, a_1)
      [bc_1, _] = _broadcast_half(bc_0, b_1)
      c_1 = RowPartition.from_uniform_row_length(
          static_a_1, nrows=ac_0.dest_nrows())
      return [c_1, ac_1, bc_1]
    elif static_a_1 == 1:
      [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
      ac_1 = _LayerBroadcaster.from_gather_index(
          array_ops.gather(ac_0.gather_index, c_1b.value_rowids()))
      c_1 = RowPartition.from_uniform_row_length(
          b_1.uniform_row_length(), nrows=bc_0.dest_nrows())
      return [c_1, ac_1, bc_1]

  if static_b_1 == 1:
    [ac_1, c_1a] = _broadcast_half(ac_0, a_1)
    bc_1 = _LayerBroadcaster.from_gather_index(
        array_ops.gather(bc_0.gather_index, c_1a.value_rowids()))
    c_1 = RowPartition.from_uniform_row_length(
        a_1.uniform_row_length(), nrows=ac_0.dest_nrows())
    return [c_1, ac_1, bc_1]

  def broadcast_noop():
    # [FUNCTION DOC]
    # HELPER: branch for equal uniform row lengths. Returns the row length and both gather indices.
    # Assumes a_1.uniform_row_length() == b_1.uniform_row_length()
    # Both sides broadcast to a single shape.
    [ac_1, _] = _broadcast_half(ac_0, a_1)
    [bc_1, _] = _broadcast_half(bc_0, b_1)
    return [a_1.uniform_row_length(), ac_1.gather_index, bc_1.gather_index]

  def broadcast_a():
    # [FUNCTION DOC]
    # HELPER: branch where the first partition has row length one. Returns the second row length
    # and the gather indices.
    [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
    ac_1_gather_index = array_ops.gather(ac_0.gather_index, c_1b.value_rowids())
    return [
        b_1.uniform_row_length(),
        ac_1_gather_index,
        bc_1.gather_index,
    ]

  def broadcast_b():
    # [FUNCTION DOC]
    # HELPER: branch where the second partition has row length one. Returns the first row length
    # and the gather indices.
    [ac_1, c_1a] = _broadcast_half(ac_0, a_1)
    bc_1_gather_index = array_ops.gather(bc_0.gather_index, c_1a.value_rowids())
    return [a_1.uniform_row_length(), ac_1.gather_index, bc_1_gather_index]

  can_broadcast_b = math_ops.equal(b_1.uniform_row_length(), 1)

  def no_broadcast_a():
    # [FUNCTION DOC]
    # HELPER: branch where the first row length is not one: chooses between broadcast of b and
    # no-op with a conditional.
    return cond.cond(
        can_broadcast_b, true_fn=broadcast_b, false_fn=broadcast_noop)

  can_broadcast_a = math_ops.equal(a_1.uniform_row_length(), 1)

  broadcast_asserts = [
      check_ops.assert_equal(
          math_ops.logical_or(
              math_ops.logical_or(can_broadcast_a, can_broadcast_b),
              math_ops.equal(a_1.uniform_row_length(),
                             b_1.uniform_row_length())), True)
  ]

  result = cond.cond(
      can_broadcast_a, true_fn=broadcast_a, false_fn=no_broadcast_a)

  [c_1_uniform_row_length, ac_1_gather_index, bc_1_gather_index] = [
      control_flow_ops.with_dependencies(broadcast_asserts, x) for x in result
  ]

  c_1 = RowPartition.from_uniform_row_length(
      c_1_uniform_row_length,
      nvals=c_1_uniform_row_length * ac_0.dest_nrows(),
      nrows=ac_0.dest_nrows())
  ac_1 = _LayerBroadcaster.from_gather_index(ac_1_gather_index)
  bc_1 = _LayerBroadcaster.from_gather_index(bc_1_gather_index)
  return [c_1, ac_1, bc_1]


def _broadcast_dynamic_shape_next_layer(
    ac_0: _LayerBroadcaster, bc_0: _LayerBroadcaster, a_1: RowPartition,
    b_1: RowPartition
) -> Tuple[RowPartition, _LayerBroadcaster, _LayerBroadcaster]:
  # [FUNCTION DOC]
  # HELPER: dispatcher for the next layer of broadcasting two shapes.
  # Inputs: ac_0, bc_0, a_1, b_1 as in the specialised versions. Raises type errors for wrong
  # types. Returns the target row partition and the two next layer broadcasters. Chooses the
  # both-uniform or half-ragged versions (swapping the arguments when only b_1 is uniform and
  # swapping the results back); if both are ragged they must be equal, enforced by an
  # assertion on row splits.
  r"""Broadcast target and next layer broadcaster of two dynamic shapes.

     *--ac_0-->*<--bc_0--*
     |         |         |
    a_1       c_1       b_1
     |         |         |
     V         V         V
     *--ac_1-->*<--bc_1--*

  Args:
    ac_0: _LayerBroadcaster from a to c in the previous layer.
    bc_0: _LayerBroadcaster from b to c in the previous layer.
    a_1: a RowPartition for the next layer of a.
    b_1: a RowPartition for the next layer of b.

  Returns:
    (c_1, ac_1, bc_1)
    c_1: a RowPartition for the next layer of the dynamic shape.
    ac_1: _LayerBroadcaster from a to c in the next layer.
    bc_1: _LayerBroadcaster from b to c in the next layer.
  """
  if not isinstance(ac_0, _LayerBroadcaster):
    raise TypeError("ac_0 should be a _LayerBroadcaster")
  if not isinstance(bc_0, _LayerBroadcaster):
    raise TypeError("bc_0 should be a _LayerBroadcaster")
  if not isinstance(a_1, RowPartition):
    raise TypeError("a_1 should be a RowPartition")
  if not isinstance(b_1, RowPartition):
    raise TypeError("b_1 should be a RowPartition")

  if a_1.is_uniform():
    if b_1.is_uniform():
      return _broadcast_dynamic_shape_next_layer_both_uniform(
          ac_0, bc_0, a_1, b_1)
    else:
      return _broadcast_dynamic_shape_next_layer_half_ragged(
          ac_0, bc_0, a_1, b_1)
  else:
    if b_1.is_uniform():
      [c_1, bc_1, ac_1] = _broadcast_dynamic_shape_next_layer_half_ragged(  # pylint: disable=arguments-out-of-order
          bc_0, ac_0, b_1, a_1)
      return (c_1, ac_1, bc_1)
    else:
      # If neither shape is uniform, we cannot broadcast the dimension.
      [ac_1, c_1a] = _broadcast_half(ac_0, a_1)
      [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
      check_valid = [
          check_ops.assert_equal(c_1a.row_splits(), c_1b.row_splits())
      ]
      return (
          c_1a._with_dependencies(check_valid),  # pylint: disable=protected-access
          ac_1.with_dependencies(check_valid),
          bc_1.with_dependencies(check_valid))


def _broadcast_dynamic_shape_from_rps(
    a_zero: _LayerBroadcaster, b_zero: _LayerBroadcaster,
    a_rps: Sequence[RowPartition], b_rps: Sequence[RowPartition]
) -> Tuple[Sequence[RowPartition], Sequence[_LayerBroadcaster],
           Sequence[_LayerBroadcaster]]:
  # [FUNCTION DOC]
  # HELPER: recursive loop over row partitions for broadcasting two shapes.
  # Inputs: a_zero and b_zero, layer broadcasters of the first rows; a_rps and b_rps, equal
  # length sequences of row partitions. Returns the target partitions and the layer broadcaster
  # lists for both shapes (empty for empty inputs).
  """Create BroadcastLayers from two shapes to a target shape.


      *--a_zero->*<-b_zero-*
      |          |         |
   a_rps[0]    c_rps[0]  b_rps[0]
      |          |         |
      V          V         V
      *--ac[1]-->*<-bc[1]--*
      |          |         |
   a_rps[1]   c_rps[0]   b_rps[1]
      |          |         |
      V          V         V
      *--ac[2]-->*<-bc[2]--*

  Note: ac[0]=a_zero, and bc[0]=b_zero.
  Args:
    a_zero: broadcaster from rows of a_rps[0] to target shape.
    b_zero: broadcaster from rows of b_rps[0] to target shape.
    a_rps: RowPartitions of first shape.
    b_rps: RowPartitions of second shape, equal in length to a_rps.

  Returns:
    (c_rps, ac, bc) where:
    c_rps: RowPartitions of target shape.
    ac: layers broadcasting from the first shape.
    bc: layers broadcasting from the second shape.
  """
  assert len(a_rps) == len(b_rps)
  if a_rps:
    (c_1, ac_1,
     bc_1) = _broadcast_dynamic_shape_next_layer(a_zero, b_zero, a_rps[0],
                                                 b_rps[0])
    (c_suffix, a_layers,
     b_layers) = _broadcast_dynamic_shape_from_rps(ac_1, bc_1, a_rps[1:],
                                                   b_rps[1:])

    return ([c_1] + c_suffix, [ac_1] + a_layers, [bc_1] + b_layers)
  else:
    return ([], [], [])


def _get_broadcast_num_row_partitions(a: DynamicRaggedShape,
                                      b: DynamicRaggedShape):
  # [FUNCTION DOC]
  # HELPER: computes the number of row partitions of the broadcast result.
  # Inputs: a and b, shapes with known rank. Returns zero if both have none; if one has none,
  # the other's count plus the rank difference; otherwise the larger of the adjusted counts.
  """Returns broadcast_dynamic_shape(a, b).num_row_partitions."""
  # Assumes rank and num_row_partitions are not None.
  if (a.num_row_partitions == 0 and b.num_row_partitions == 0):
    return 0
  expanded_num_row_partitions_a = a.num_row_partitions + max(0, b.rank - a.rank)
  expanded_num_row_partitions_b = b.num_row_partitions + max(0, a.rank - b.rank)

  if a.num_row_partitions == 0:
    return expanded_num_row_partitions_b

  if b.num_row_partitions == 0:
    return expanded_num_row_partitions_a

  return max(expanded_num_row_partitions_a, expanded_num_row_partitions_b)


# pylint: disable=protected-access
def _broadcast_dynamic_shape_extended_complete(
    a: DynamicRaggedShape, b: DynamicRaggedShape, b_rps: Sequence[RowPartition],
    c_suffix: Sequence[RowPartition], ac: Sequence[_LayerBroadcaster],
    bc_suffix: Sequence[_LayerBroadcaster]
) -> Tuple[DynamicRaggedShape, _Broadcaster, _Broadcaster]:
  # [FUNCTION DOC]
  # HELPER: finishing step of broadcast_dynamic_shape_extended.
  # KNOWN ISSUE: it concatenates a slice of the row partitions of b with a tuple; if the
  # partitions are stored as a list (as happens when validation adds dependencies) the addition
  # of list and tuple would raise a type error.
  # Builds the result shape from the unchanged prefix partitions of b and the broadcast suffix,
  # re-factors it to the expected number of partitions, and returns the shape and the two
  # broadcaster objects (identity layers prepended for b's leading dimensions).
  """Helper for broadcast_dynamic_shape_extended."""
  c_prefix = b_rps[:-len(c_suffix)]
  bc_prefix_length = b.rank - len(bc_suffix)
  bc_prefix = [
      _LayerBroadcaster.get_identity_broadcaster(b._num_slices_in_dimension(i))
      for i in range(bc_prefix_length)
  ]
  c_num_row_partitions = _get_broadcast_num_row_partitions(a, b)

  c_raw = DynamicRaggedShape.from_row_partitions(c_prefix + tuple(c_suffix))
  c = c_raw._with_num_row_partitions(c_num_row_partitions)
  return (c, _Broadcaster(a, c, ac), _Broadcaster(b, c, bc_prefix + bc_suffix))


def _broadcast_dynamic_shape_extended_helper(
    a: DynamicRaggedShape, b: DynamicRaggedShape
) -> Tuple[DynamicRaggedShape, _Broadcaster, _Broadcaster]:
  # [FUNCTION DOC]
  # HELPER: main algorithm of broadcast_dynamic_shape_extended.
  # Requires (by assertions) rank of a not above rank of b, b rank at least two and a rank at
  # least one. Converts both to row partitions. If a has fewer partitions, an extra leading
  # partition aligns it with b; the broadcast is then computed layer by layer; otherwise computed
  # directly. Returns the shape and two broadcasters.
  """Helper for broadcast_dynamic_shape_extended.

  Here, we force:
    a.rank <= b.rank
    2 <= b.rank
    1 <= a.rank
  Args:
    a: a DynamicRaggedShape
    b: a DynamicRaggedShape

  Returns:
    A triple of a shape and two broadcasters.
  """
  assert a.rank <= b.rank
  assert 2 <= b.rank
  assert 1 <= a.rank
  a_rps = a._as_row_partitions()  # pylint: disable=protected-access
  b_rps = b._as_row_partitions()  # pylint: disable=protected-access

  if len(a_rps) < len(b_rps):
    # Note: this includes the case where len(a_rps)==0.
    # Here we begin at -1, one dimension before a_rps[0].
    # neg_one_a_rp  | b_rps[-(len(a_rps)+1)]
    # a_rps[0]      | b_rps[-len(a_rps)]
    # a_rps[1]      | b_rps[1-len(a_rps)]
    # ...           | ...
    # a_rps[-1]     | b_rps[-1]

    a_nrows = a[0]
    a_nrows_static = tensor_util.constant_value(a_nrows)
    if a_nrows_static is not None:
      a_nrows = a_nrows_static

    neg_one_a_rp = RowPartition.from_uniform_row_length(
        uniform_row_length=a_nrows, nrows=1, nvals=a_nrows)
    neg_one_b_rp = b_rps[-(len(a_rps) + 1)]
    (neg_one_ac, neg_one_bc) = _broadcast_dynamic_shape_first_layer(
        constant_op.constant(1, dtype=b_rps[0].dtype), neg_one_b_rp.nrows())

    # The first part of the solution.
    (c_zero, ac_zero,
     bc_zero) = _broadcast_dynamic_shape_next_layer(neg_one_ac, neg_one_bc,
                                                    neg_one_a_rp, neg_one_b_rp)
    b_rps_tail = b_rps[-len(a_rps):] if len(a_rps) >= 1 else []

    (c_suffix, ac_layers,
     bc_layers) = _broadcast_dynamic_shape_from_rps(ac_zero, bc_zero, a_rps,
                                                    b_rps_tail)

    return _broadcast_dynamic_shape_extended_complete(
        a=a,
        b=b,
        b_rps=b_rps,
        c_suffix=[c_zero] + c_suffix,
        ac=[ac_zero] + ac_layers,
        bc_suffix=[neg_one_bc, bc_zero] + bc_layers)

  else:
    assert len(a_rps) == len(b_rps)
    (ac_zero,
     bc_zero) = _broadcast_dynamic_shape_first_layer(a_rps[0].nrows(),
                                                     b_rps[0].nrows())

    (c_rps, a_layers,
     b_layers) = _broadcast_dynamic_shape_from_rps(ac_zero, bc_zero, a_rps,
                                                   b_rps)
    return _broadcast_dynamic_shape_extended_complete(
        a=a,
        b=b,
        b_rps=b_rps,
        c_suffix=c_rps,
        ac=[ac_zero] + a_layers,
        bc_suffix=[bc_zero] + b_layers)


def _fix_start_index(index, rank, num_row_partitions):
  # [FUNCTION DOC]
  # HELPER: normalizes the start of a slice for the indexing operator.
  # Negative indices count from the end (value error if rank unknown), are clipped to zero, and
  # large indices are clipped to the rank (value error if rank unknown and the index is large
  # without partitions). Returns an int.
  """Slice indexes are always silently truncated."""
  if index < 0:
    if rank is None:
      raise ValueError(
          "Rank must be known to use __getitem__ on a negative index.")
    index = rank + index
  if index < 0:
    index = 0
  if (num_row_partitions > 0 and index <= num_row_partitions + 1):
    # The rank is always >= num_row_partitions + 1 if num_row_partitions > 0.
    return index
  if index == 0:
    return index
  if rank is None:
    raise ValueError("Rank must be known to use __getitem__ on a large index.")
  if index >= rank:
    index = rank
  return index


def _fix_stop_index(index, rank):
  # [FUNCTION DOC]
  # HELPER: normalizes the stop of a slice for the indexing operator.
  # Nothing means the rank (value error if unknown); negative counts from the end; clipped to
  # zero and to the rank when known. Returns an int.
  """Slice indexes are always silently truncated."""
  if index is None:
    if rank is None:
      raise ValueError("Rank must be known to use __getitem__ without a stop.")
    index = rank
  if index < 0:
    if rank is None:
      raise ValueError(
          "Rank must be known to use __getitem__ on a negative index.")
    index = rank + index
  if index < 0:
    index = 0
  if rank is not None:
    index = min(rank, index)
  return index


def _first_layer_gather_index(nrows_source, nrows_target):
  # [FUNCTION DOC]
  # HELPER: used by the first layer broadcaster factory.
  # Returns a gather index tensor from source row count to target row count: zeros when the
  # source count is one, range otherwise, wrapped with a run-time assertion that the source is
  # one or the counts are equal.
  """Return the first layer gather_index.

  Args:
    nrows_source: the number of rows in the source.
    nrows_target: the number of rows in the target.

  Returns:
    A tensor, usable as a gather_index for a _LayerBroadcaster.
  """

  def gi_broadcast_first():
    # [FUNCTION DOC]
    # HELPER: branch returning zeros of target size, used when the source has one row.
    return array_ops.zeros(nrows_target, dtype=nrows_target.dtype)

  def gi_no_broadcast_first():
    # [FUNCTION DOC]
    # HELPER: branch returning a range of the target size, used when the counts are equal.
    gather_index = math_ops.range(nrows_target, dtype=nrows_target.dtype)
    return gather_index

  do_broadcast = math_ops.equal(nrows_source,
                                constant_op.constant(1, nrows_source.dtype))
  nrows_equal = math_ops.equal(nrows_source, nrows_target)
  can_broadcast = check_ops.assert_equal(
      math_ops.logical_or(do_broadcast, nrows_equal),
      True,
      message="Cannot broadcast")

  gather_index = cond.cond(
      do_broadcast, true_fn=gi_broadcast_first, false_fn=gi_no_broadcast_first)

  return control_flow_ops.with_dependencies([can_broadcast], gather_index)


def _next_layer_gather_index(bc, original_rp, broadcast_rp):
  # [FUNCTION DOC]
  # HELPER: used by the layer broadcaster to compute the next layer.
  # Inputs: bc, the old layer broadcaster; original_rp and broadcast_rp, the source and target
  # row partitions. Returns the gather index for the next layer. A ragged source is gathered
  # with a check that the row lengths match; a uniform source uses a conditional between simple
  # broadcast (row length one) and the checked path. Mismatch causes an invalid-argument error
  # at run time.
  r"""Create the next layer gather_index whether or not a broadcast happens.

     *----------bc-------->*
     |                     |
  original_rp           broadcast_rp
     |                     |
    \|/                   \|/
     *--next_broadcaster-->*

  Args:
    bc: the old broadcaster.
    original_rp: the original row partition.
    broadcast_rp: the target row partition.

  Returns:
    the gather_index for next_broadcaster.
  Raises:
    InvalidArgumentError if the shapes are incompatible.
  """
  old_value_rowids = array_ops.gather(bc.gather_index,
                                      broadcast_rp.value_rowids())

  def gi_no_broadcast():
    # [FUNCTION DOC]
    # HELPER: branch computing the gather index without broadcasting; checks that expected and
    # actual row lengths agree with an assertion and returns row starts plus offsets.
    # TODO(martinz): decide if row_splits or row_starts should be used here.
    old_row_starts = array_ops.gather(original_rp.row_splits(),
                                      old_value_rowids)
    expected_row_lengths = array_ops.gather(
        params=original_rp.row_lengths(), indices=bc.gather_index)
    actual_row_lengths = broadcast_rp.row_lengths()
    check_valid = check_ops.assert_equal(
        expected_row_lengths, actual_row_lengths, message="Cannot broadcast")
    gather_index = old_row_starts + broadcast_rp.offsets_in_rows()
    return control_flow_ops.with_dependencies([check_valid], gather_index)

  def gi_broadcast():
    # [FUNCTION DOC]
    # HELPER: branch computing the gather index when the source row length is one; returns the
    # old value row ids directly.
    # Several optimizations can occur here.
    # old_row_starts == old_value_rowids, because:
    #   if you are broadcasting, then the source has uniform row length of 1,
    #   implying original_rp.row_splits == tf.range(original_rp.nvals + 1)
    # When broadcasting, there is no need to add offsets to the
    # source, because the source has size 1.
    # Also, this is always valid, because we enforce source and destination
    # have uniform_row_length.
    return old_value_rowids

  if not original_rp.is_uniform():
    return gi_no_broadcast()

  do_broadcast = math_ops.equal(original_rp.uniform_row_length(),
                                constant_op.constant(1, original_rp.dtype))
  gather_index = cond.cond(
      do_broadcast, true_fn=gi_broadcast, false_fn=gi_no_broadcast)

  return gather_index


def _flat_values_shape(rt):
  # [FUNCTION DOC]
  # HELPER: used by from_tensor.
  # KNOWN ISSUE: for objects that are not ragged tensors it reads a flat values attribute that
  # dense tensors do not have, so that branch would fail; it is only reached with ragged input.
  # Returns the dynamic shape of a ragged tensor's flat values.
  if isinstance(rt, ragged_tensor.RaggedTensor):
    return array_ops.shape(rt.flat_values)
  return rt.flat_values.shape


def _to_row_partitions_and_nvals_from_lengths(
    lengths: Sequence[Union[int, Sequence[int]]],
    dtype=None) -> Tuple[Sequence[RowPartition], int]:
  # [FUNCTION DOC]
  # HELPER: used by from_lengths.
  # Converts a list of lengths (ints for uniform axes, sequences of ints for ragged ones) into
  # row partitions. Inputs: lengths; dtype, optional hint. Returns a pair of the partitions and
  # the number of values of the last one. Raises a value error if a ragged length count does not
  # match the number of values so far.
  """Allow ragged and uniform shapes to be specified.

  For example, [2, [2,1], 2] represents a shape like:
  [[[0, 0], [0, 0]], [[0, 0]]]

  Args:
    lengths: a list of integers and lists of integers.
    dtype: dtype of the shape (tf.int32 or tf.int64)

  Returns:
    a sequence of RowPartitions, and the number of values of the last partition.
  """
  size_so_far = lengths[0]
  result = []
  for current_lengths in lengths[1:]:
    if isinstance(current_lengths, int):
      nrows = size_so_far
      nvals = current_lengths * nrows
      size_so_far = nvals
      result.append(
          RowPartition.from_uniform_row_length(
              current_lengths, nvals, nrows=nrows, dtype_hint=dtype))
    else:
      if size_so_far != len(current_lengths):
        raise ValueError("Shape not consistent.")
      result.append(
          RowPartition.from_row_lengths(current_lengths, dtype_hint=dtype))
      size_so_far = sum(current_lengths)
  return (result, size_so_far)


def _element_to_string(x):
  # [FUNCTION DOC]
  # HELPER: formatting for repr. Returns three dots for an ellipsis, a quoted string for a
  # string, and the text form otherwise.
  """element to a string within a list."""
  if x is Ellipsis:
    return "..."
  if isinstance(x, str):
    return "'" + x + "'"
  return str(x)


def _list_tail_with_ellipsis(arr):
  # [FUNCTION DOC]
  # HELPER: formatting for repr. Recursively renders the tail of a list with leading commas and a
  # closing bracket; an empty tail gives the closing bracket.
  """Print the tail of a list where the list might have an ellipsis."""
  if not arr:
    return "]"
  else:
    return ", " + _element_to_string(arr[0]) + _list_tail_with_ellipsis(arr[1:])


def _list_with_ellipsis_to_str(arr):
  # [FUNCTION DOC]
  # HELPER: formatting for repr. Renders a list that may contain an ellipsis marker as a string;
  # empty gives an empty list text.
  """Print a list that might have ellipsis."""
  if not arr:
    return "[]"
  return "[" + _element_to_string(arr[0]) + _list_tail_with_ellipsis(arr[1:])


def _is_int_or_tuple_of_ints(x):
  # [FUNCTION DOC]
  # HELPER: validation for from_lengths. Returns true for an int or a tuple whose items are all
  # ints, false otherwise (a list is rejected; booleans count as ints).
  if isinstance(x, int):
    return True
  if not isinstance(x, tuple):
    return False
  for y in x:
    if not isinstance(y, int):
      return False
  return True


def _alt_inner_shape_from_tensor_shape(shape, dtype, new_inner_rank):
  # [FUNCTION DOC]
  # HELPER: used by _alt_inner_shape and _with_num_row_partitions for statically known shapes.
  # Inputs: shape, a fully defined TensorShape; dtype; new_inner_rank, a positive int. Returns a
  # constant tensor of the shape folded to the requested inner rank (leading dimensions
  # collapsed into one).
  """Helper for _alt_inner_shape, used directly in _with_num_row_partitions."""
  if new_inner_rank == 1:
    return constant_op.constant([shape.num_elements()], dtype=dtype)
  new_inner_rank_tail_length = new_inner_rank - 1
  inner_shape_tail = shape[-new_inner_rank_tail_length:].as_list()
  first_dim = shape[:-new_inner_rank_tail_length].num_elements()
  return constant_op.constant([first_dim] + inner_shape_tail, dtype=dtype)


def _safe_floor_div(dividend: tensor_shape.Dimension,
                    divisor: tensor_shape.Dimension) -> tensor_shape.Dimension:
  # [FUNCTION DOC]
  # HELPER: used by the spec constructor.
  # KNOWN ISSUE: annotated to return a Dimension but returns nothing when the divisor is zero.
  # Floor-divides one dimension by another; returns nothing when the divisor is zero.
  if tensor_shape.dimension_value(divisor) == 0:
    return None
  return dividend // divisor


# TODO(b/218932570)
def _reduce_prod_patch(x):
  # [FUNCTION DOC]
  # HELPER: workaround for a product reduction problem (see the bug id in the comment).
  # KNOWN ISSUE: 64-bit values are cast to 32-bit before multiplying, so products above the
  # 32-bit range overflow silently; the workaround is only valid for small shapes.
  # Returns the product of a vector as a scalar tensor in the input dtype.
  if x.dtype == dtypes.int64:
    return math_ops.cast(
        math_ops.reduce_prod(math_ops.cast(x, dtypes.int32)), dtypes.int64)
  return math_ops.reduce_prod(x)


# Type alias for shape encoded as a DynamicRaggedShape or a Tensor.
DenseOrRaggedShape = Union[DynamicRaggedShape, core.TensorLike]


def _merge_row_partitions(
    row_partitions: Sequence[RowPartition]) -> RowPartition:
  # [FUNCTION DOC]
  # HELPER: used by _merge_dims.
  # Merges consecutive row partitions into one by composing their row splits through gathering.
  # Input: a non-empty sequence of row partitions. Returns a new row partition. Does not handle
  # uniform splits specially.
  # TODO(martinz): handle uniform splits.
  # TODO(martinz): consider using value_row_ids if present.
  # Note: this probably won't be called with len(row_partitions)==1, so no
  # need to optimize.
  row_splits = row_partitions[0].row_splits()
  for rp in row_partitions[1:]:
    row_splits = array_ops.gather(rp.row_splits(), row_splits)
  return RowPartition.from_row_splits(row_splits)


def _merge_inner_shape(
    inner_shape: tensor_lib.Tensor,
    static_inner_shape: tensor_shape.TensorShape,
    outer_axis: int,
    inner_axis: int) -> Tuple[tensor_lib.Tensor, tensor_shape.TensorShape]:
  # [FUNCTION DOC]
  # HELPER: used by _merge_dims for the inner shape.
  # Collapses inner dimensions between outer_axis and inner_axis (inclusive) into their product,
  # both dynamically and statically. Returns a pair of the new inner shape tensor and the new
  # static inner shape.
  """Merge the inner shape of a DynamicRaggedShape."""
  prefix = inner_shape[:outer_axis]
  suffix = inner_shape[inner_axis + 1:]

  internal = inner_shape[outer_axis:inner_axis + 1]
  internal_value = [_reduce_prod_patch(internal)]
  new_internal = array_ops.concat([prefix, internal_value, suffix], axis=0)
  prefix_static = static_inner_shape[:outer_axis]
  suffix_static = static_inner_shape[inner_axis + 1:]
  internal_static = static_inner_shape[outer_axis:inner_axis + 1]
  internal_value_static = tensor_shape.TensorShape(
      [internal_static.num_elements()])
  new_internal_static = prefix_static + internal_value_static + suffix_static

  return (new_internal, new_internal_static)


def _batch_rp_spec(rp_spec: RowPartitionSpec,
                   batch_size: Optional[int]) -> RowPartitionSpec:
  # [FUNCTION DOC]
  # HELPER: used by the batch encoder.
  # Returns the spec of the concatenation of batch_size row partitions: with unknown batch size
  # only the uniform row length and dtype are kept; otherwise row and value counts are
  # multiplied by the batch size when known.
  """Batches a RowPartitionSpec.

  Given a RowPartitionSpec and a batch_size, create a RowPartitionSpec that
  will be the spec for the concatenation of batch_size RowPartitions.

  A RowPartition can be considered a transformation from a list of a given
  length to a list of lists. Assume rp_a is a map from list_a to nlist_a,
  And rp_b is a map from list_b to nlist_b. concat(rp_a, rp_b) is a
  transform of concat(list_a, list_b) to concat(nlist_a, nlist_b).

  If batch_size is None, then have the spec be able to handle an arbitrary
  number of RowPartitions.

  Args:
    rp_spec: a RowPartitionSpec for all the RowPartitions to be concatenated.
    batch_size: the number of rp_specs to be concatenated.

  Returns:
    a batched RowPartitionSpec.
  """
  if batch_size is None:
    return RowPartitionSpec(
        uniform_row_length=rp_spec.uniform_row_length, dtype=rp_spec.dtype)
  nrows = None if rp_spec.nrows is None else rp_spec.nrows * batch_size
  nvals = None if rp_spec.nvals is None else rp_spec.nvals * batch_size
  return RowPartitionSpec(
      nrows=nrows,
      nvals=nvals,
      uniform_row_length=rp_spec.uniform_row_length,
      dtype=rp_spec.dtype)


def _batch_rp_spec_head(old_head: RowPartitionSpec,
                        batch_size: Optional[int]) -> RowPartitionSpec:
  # [FUNCTION DOC]
  # HELPER: used by the batch encoder. Returns the spec of the new leading uniform partition
  # created by batching: its row count is the batch size, its row length the old head's row
  # count, and its value count the product when both are known.
  """Creates a RowPartitionSpec representing the new dimension created."""
  nvals = None if (old_head.nrows is None or
                   batch_size is None) else batch_size * old_head.nrows
  return RowPartitionSpec(
      nrows=batch_size,
      nvals=nvals,
      uniform_row_length=old_head.nrows,
      dtype=old_head.dtype)


def _batch_static_inner_shape(
    old_shape: tensor_shape.TensorShape,
    batch_size: Optional[int]) -> tensor_shape.TensorShape:
  # [FUNCTION DOC]
  # HELPER: used by the batch encoder. Returns a copy of the shape in which the first dimension
  # is multiplied by the batch size.
  """Returns a copy of old_shape with axis=0 multiplied by batch_size.

  Only use if this is the inner_shape of a DynamicRaggedShape.Spec with one
  or more row partitions.

  Args:
    old_shape: the original inner_shape.
    batch_size: the batch size.

  Returns:
    a new shape.
  """
  head_dim = tensor_shape.dimension_at_index(old_shape, 0) * batch_size
  return head_dim + old_shape[1:]


def _batch_tensor_shape(old_shape: tensor_shape.TensorShape,
                        batch_size: int) -> tensor_shape.TensorShape:
  # [FUNCTION DOC]
  # HELPER: used by the batch encoder. Returns the shape with the batch size prepended.
  return tensor_shape.TensorShape([batch_size]) + old_shape


def _unbatch_static_inner_shape(
    old_shape: tensor_shape.TensorShape,
    batch_size: Optional[int]) -> tensor_shape.TensorShape:
  # [FUNCTION DOC]
  # HELPER: used by the batch encoder. Returns a copy of the shape with the first dimension
  # divided by the batch size (unknown stays unknown).
  """Unbatch a static_inner_shape when num_row_partitions > 0."""
  head_dim = tensor_shape.dimension_at_index(old_shape, 0) // batch_size
  return head_dim + old_shape[1:]


# Copied from ragged_array_ops.py
def ones(shape: DynamicRaggedShape,
         dtype=dtypes.float32,
         name: Optional[str] = None) -> ragged_tensor.RaggedOrDense:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says ones shaped like x, but the argument is a DynamicRaggedShape,
  # not a tensor.
  # Public helper copied from ragged array ops. Inputs: shape, a DynamicRaggedShape; dtype,
  # default float32; name, optional op name. Returns a tensor (or ragged tensor if the shape has
  # partitions) full of ones with that shape.
  """Returns ones shaped like x."""
  flat_values = array_ops.ones(shape.inner_shape, dtype=dtype, name=name)
  return ragged_tensor.RaggedTensor._from_nested_row_partitions(  # pylint: disable=protected-access
      flat_values, shape.row_partitions)
