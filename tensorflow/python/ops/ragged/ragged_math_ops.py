# [MODULE DOC]
# Ragged-tensor math operations: ragged range, segment aggregations (sum, prod, min, max, mean,
# sqrt-n), axis reductions (sum, prod, min, max, mean, variance, std, all, any), matrix
# multiplication, softmax, add_n, dropout variants, equality and not-equality, and cumulative sum.
# Most public functions are registered as dispatch targets for the corresponding dense math
# functions, so calling the dense API with a ragged input is routed here. Only range and cumsum-like
# entry points are exported directly under the ragged namespace (range is exported explicitly).
# Import-time side effects: registers dispatch handlers, registers RaggedRange as having no
# gradient, and overwrites the documentation strings of the segment and reduce functions with text
# generated from shared templates.
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
"""Support for ragged tensors."""

import functools
import typing

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


# ===============================================================================
# ragged.range
# ===============================================================================
# pylint: disable=redefined-builtin
@tf_export('ragged.range')
@dispatch.add_dispatch_support
def range(starts,
          limits=None,
          deltas=1,
          dtype=None,
          name=None,
          row_splits_dtype=dtypes.int64):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says an empty row results when start is less than limit with a
  # positive delta (and the mirrored case); that is backwards. Rows are empty when the range is
  # empty, i.e. start is at or beyond limit for a positive delta.
  # Builds a ragged tensor whose row i is the arithmetic sequence from starts[i] up to (excluding)
  # limits[i] with step deltas[i]. Inputs: starts, limits, deltas are scalars or equal-length
  # vectors (scalars broadcast); if limits is None, starts is treated as the limits and starts
  # become 0; dtype optionally forces the element type, otherwise int32, int64, float32 or float64
  # is chosen as the widest of the inputs; name is an optional op name; row_splits_dtype (int32 or
  # int64, default int64) is the type of the result's row splits. Returns a ragged tensor with
  # ragged rank 1. Runs the ragged range kernel; the op has no gradient. Invalid dtype or shapes
  # surface as errors from the underlying conversion or kernel.
  """Returns a `RaggedTensor` containing the specified sequences of numbers.

  Each row of the returned `RaggedTensor` contains a single sequence:

  ```python
  ragged.range(starts, limits, deltas)[i] ==
      tf.range(starts[i], limits[i], deltas[i])
  ```

  If `start[i] < limits[i] and deltas[i] > 0`, then `output[i]` will be an
  empty list.  Similarly, if `start[i] > limits[i] and deltas[i] < 0`, then
  `output[i]` will be an empty list.  This behavior is consistent with the
  Python `range` function, but differs from the `tf.range` op, which returns
  an error for these cases.

  Examples:

  >>> tf.ragged.range([3, 5, 2]).to_list()
  [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1]]
  >>> tf.ragged.range([0, 5, 8], [3, 3, 12]).to_list()
  [[0, 1, 2], [], [8, 9, 10, 11]]
  >>> tf.ragged.range([0, 5, 8], [3, 3, 12], 2).to_list()
  [[0, 2], [], [8, 10]]

  The input tensors `starts`, `limits`, and `deltas` may be scalars or vectors.
  The vector inputs must all have the same size.  Scalar inputs are broadcast
  to match the size of the vector inputs.

  Args:
    starts: Vector or scalar `Tensor`.  Specifies the first entry for each range
      if `limits` is not `None`; otherwise, specifies the range limits, and the
      first entries default to `0`.
    limits: Vector or scalar `Tensor`.  Specifies the exclusive upper limits for
      each range.
    deltas: Vector or scalar `Tensor`.  Specifies the increment for each range.
      Defaults to `1`.
    dtype: The type of the elements of the resulting tensor.  If not specified,
      then a value is chosen based on the other args.
    name: A name for the operation.
    row_splits_dtype: `dtype` for the returned `RaggedTensor`'s `row_splits`
      tensor.  One of `tf.int32` or `tf.int64`.

  Returns:
    A `RaggedTensor` of type `dtype` with `ragged_rank=1`.
  """
  row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
  if limits is None:
    starts, limits = 0, starts

  with ops.name_scope(name, 'RaggedRange', [starts, limits, deltas]) as name:
    starts = ops.convert_to_tensor(starts, dtype=dtype, name='starts')
    limits = ops.convert_to_tensor(limits, dtype=dtype, name='limits')
    deltas = ops.convert_to_tensor(deltas, dtype=dtype, name='deltas')

    # infer dtype if not explicitly provided
    if dtype is None:
      starts, limits, deltas = _infer_matching_dtype(
          [starts, limits, deltas],
          [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64])

    result = gen_ragged_math_ops.ragged_range(
        starts, limits, deltas, Tsplits=row_splits_dtype, name=name)
    return ragged_tensor.RaggedTensor.from_row_splits(
        result.rt_dense_values, result.rt_nested_splits, validate=False)


def _infer_matching_dtype(tensors, dtype_hierarchy):
  # [FUNCTION DOC]
  # HELPER: dtype unification for the ragged range function.
  # Picks the dtype that appears latest in the given precedence list among the input tensors and
  # casts every tensor to it. Inputs: a list of tensors and an ordered list of allowed dtypes.
  # Returns the list of cast tensors. Raises an assertion error if any tensor has a dtype not in the
  # list.
  """Infers a matching dtype for tensors, and casts them to that dtype."""
  assert all(t.dtype in dtype_hierarchy for t in tensors)
  inferred_dtype = max([t.dtype for t in tensors], key=dtype_hierarchy.index)
  return [math_ops.cast(t, inferred_dtype) for t in tensors]


ops.no_gradient('RaggedRange')

# ===============================================================================
# ragged_segment_<AGGREGATE>
# ===============================================================================

# Docstring template used for the raggged_segment_<AGGREGATE> ops.
_RAGGED_SEGMENT_DOCSTRING = """\
Computes the %(combination)s along segments of a RaggedTensor.

  Returns a RaggedTensor `output` with `num_segments` rows, where the row
  `output[i]` is formed by taking the %(combination)s of all rows of `data`
  whose corresponding `segment_id` is `i`.

  The length of the row `output[i]` will be the maximum of the lengths of
  all rows of `data` whose corresponding `segment_id` is `i`.  If no `data`
  rows correspond to a given segment ID, then the output row for that segment
  ID will be empty.

  Args:
    data: A `RaggedTensor` containing the values to combine.
    segment_ids: A `Tensor` or `RaggedTensor`.  Must have type `int64` or
      `int32`.  `segment_ids.shape` must be a prefix of `data.shape`.
      Must be greater than or equal to zero, and less than `num_segments`.
      `segment_ids` is not required to be sorted.
    num_segments: An `int32` or `int64` scalar specifying the number of
      distinct segment ids.
    name: A name prefix for the returned tensor (optional).
  Returns:
    A `RaggedTensor` containing the %(combined)s values.  The returned tensor
    has the same dtype as `data`, and its shape is
    `[num_segments] + data.shape[segment_ids.rank:]`.
  Raises:
    ValueError: If `segment_ids.shape` is not a prefix of `data.shape`.
"""


def _ragged_segment_aggregate(unsorted_segment_op,
                              data,
                              segment_ids,
                              num_segments,
                              separator=None,
                              name=None):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the ragged segment_* functions and the reductions.
  # Combines rows of data that share a segment id using an unsorted-segment op. If neither data nor
  # segment_ids is ragged, simply calls the op (passing the separator when it is given, for the
  # string join op). Otherwise validates that segment_ids has int32 or int64 dtype, and that a
  # ragged segment_ids has row splits equal to data's (then recurses on the flat values). Output row
  # lengths are the maximum input row length per segment (at least zero); the function builds the
  # output splits, maps each input value to its output position and recursively aggregates the
  # values. Inputs: the op, data, segment_ids, num_segments, optional separator (string data only)
  # and optional name. Returns a ragged (or dense) tensor with num_segments rows. Raises a value
  # error for a bad segment_ids dtype or if segment_ids is ragged but data is not; a mismatch of row
  # splits raises an invalid-argument error when the assertion runs.
  """Aggregates along segments of a RaggedTensor using `unsorted_segment_op`.

  Returns a RaggedTensor `output` with `num_segments` rows, where the row
  `output[i]` is formed by combining all rows of `data` whose corresponding
  `segment_id` is `i`.  The values in each row are combined using
  `unsorted_segment_op`.

  The length of the row `output[i]` will be the maximum of the lengths of
  all rows of `data` whose corresponding `segment_id` is `i`.  If no `data`
  rows correspond to a given segment ID, then the output row for that segment
  ID will be empty.

  Args:
    unsorted_segment_op: The tensorflow `op` that should be used to combine
      values in each row.  Must have the same signature and basic behavior as
      `unsorted_segment_sum`, `unsorted_segment_max`, etc.
    data: A `RaggedTensor` containing the values to be combined.
    segment_ids: A `Tensor` or `RaggedTensor`.  Must have type `int64` or
      `int32`.  `segment_ids.shape` must be a prefix of `data.shape`.
      `segment_ids` is not required to be sorted.
    num_segments: An `int32` or `int64` scalar.
    separator: An optional string. Defaults to None. The separator to use when
      joining. Only used for string types.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` containing the aggregated values.  The returned tensor
    has the same dtype as `data`, and its shape is
    `[num_segments] + data.shape[segment_ids.rank:]`.
  Raises:
    ValueError: If segment_ids.shape is not a prefix of data.shape.
  """
  if not (ragged_tensor.is_ragged(data) or
          ragged_tensor.is_ragged(segment_ids)):
    if separator is not None:
      # It uses unsorted_segment_join.
      return unsorted_segment_op(data, segment_ids, num_segments, separator,
                                 name)
    else:
      return unsorted_segment_op(data, segment_ids, num_segments, name)

  with ops.name_scope(name, 'RaggedSegment',
                      [data, segment_ids, num_segments]) as name:
    data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name='data')
    segment_ids = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        segment_ids, name='segment_ids')
    data, segment_ids = ragged_tensor.match_row_splits_dtypes(data, segment_ids)
    if segment_ids.dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError('segment_ids must have dtype int32 or int64.')

    if ragged_tensor.is_ragged(segment_ids):
      if not ragged_tensor.is_ragged(data):
        raise ValueError('segment_ids.shape must be a prefix of data.shape, '
                         'but segment_ids is ragged and data is not.')
      check_splits = check_ops.assert_equal(
          segment_ids.row_splits,
          data.row_splits,
          message='segment_ids.shape must be a prefix of data.shape')
      with ops.control_dependencies([check_splits]):
        return _ragged_segment_aggregate(unsorted_segment_op, data.values,
                                         segment_ids.values, num_segments,
                                         separator)

    # Find the length of each row in data.  (shape=[data_nrows])
    data_row_lengths = data.row_splits[1:] - data.row_splits[:-1]

    # Find the length that each output row will have.  The length of the row
    # corresponding to segment `id` is `max(data_row_lengths[i])` where
    # `segment_ids[i]=id`.  (shape=[output_nrows])
    output_row_lengths = math_ops.maximum(
        math_ops.unsorted_segment_max(data_row_lengths, segment_ids,
                                      num_segments), 0)

    # Build the splits tensor for the output RaggedTensor.
    output_splits = array_ops.concat([
        array_ops.zeros([1], output_row_lengths.dtype),
        math_ops.cumsum(output_row_lengths)
    ],
                                     axis=0)

    # For each row in `data`, find the start & limit position where that row's
    # values will be aggregated in output.values.
    data_row_to_out_row_start = array_ops.gather(output_splits, segment_ids)
    data_row_to_out_row_limit = data_row_to_out_row_start + data_row_lengths

    # For each value in `data.values`, find the position where it will
    # aggregated in `output.values`.
    # Get the target output values index for each data values index.
    data_val_to_out_val_index = range(data_row_to_out_row_start,
                                      data_row_to_out_row_limit).values

    # Recursively aggregate the values.
    output_values = _ragged_segment_aggregate(unsorted_segment_op, data.values,
                                              data_val_to_out_val_index,
                                              output_splits[-1], separator)
    return ragged_tensor.RaggedTensor.from_row_splits(
        output_values, output_splits, validate=False)


@dispatch.dispatch_for_api(math_ops.unsorted_segment_sum)
def segment_sum(data: ragged_tensor.RaggedOrDense,
                segment_ids: ragged_tensor.RaggedOrDense,
                num_segments,
                name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for the unsorted segment sum. Sums the rows of data that share a
  # segment id (segment ids need not be sorted) into num_segments output rows; inputs are data,
  # segment_ids (ragged or dense, int32 or int64), num_segments and an optional name that defaults
  # to a ragged segment sum name. Returns a ragged tensor. Errors come from the shared aggregation
  # helper. The public docstring is generated from a template at import time.
  # For docs, see: _RAGGED_SEGMENT_DOCSTRING
  return _ragged_segment_aggregate(
      math_ops.unsorted_segment_sum,
      data=data,
      segment_ids=segment_ids,
      num_segments=num_segments,
      name=(name or 'RaggedSegmentSum'))


@dispatch.dispatch_for_api(math_ops.unsorted_segment_prod)
def segment_prod(data: ragged_tensor.RaggedOrDense,
                 segment_ids: ragged_tensor.RaggedOrDense,
                 num_segments,
                 name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for the unsorted segment product. Multiplies the rows of data that share
  # a segment id into num_segments output rows; inputs are data, segment_ids, num_segments and an
  # optional name. Returns a ragged tensor. Errors come from the shared aggregation helper.
  # For docs, see: _RAGGED_SEGMENT_DOCSTRING
  return _ragged_segment_aggregate(
      math_ops.unsorted_segment_prod,
      data=data,
      segment_ids=segment_ids,
      num_segments=num_segments,
      name=(name or 'RaggedSegmentProd'))


@dispatch.dispatch_for_api(math_ops.unsorted_segment_min)
def segment_min(data: ragged_tensor.RaggedOrDense,
                segment_ids: ragged_tensor.RaggedOrDense,
                num_segments,
                name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for the unsorted segment minimum. Takes the element-wise minimum of the
  # rows of data sharing a segment id; inputs are data, segment_ids, num_segments and an optional
  # name. Returns a ragged tensor with num_segments rows. Errors come from the shared helper.
  # For docs, see: _RAGGED_SEGMENT_DOCSTRING
  return _ragged_segment_aggregate(
      math_ops.unsorted_segment_min,
      data=data,
      segment_ids=segment_ids,
      num_segments=num_segments,
      name=(name or 'RaggedSegmentMin'))


@dispatch.dispatch_for_api(math_ops.unsorted_segment_max)
def segment_max(data: ragged_tensor.RaggedOrDense,
                segment_ids: ragged_tensor.RaggedOrDense,
                num_segments,
                name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for the unsorted segment maximum. Takes the element-wise maximum of the
  # rows of data sharing a segment id; inputs are data, segment_ids, num_segments and an optional
  # name. Returns a ragged tensor with num_segments rows. Errors come from the shared helper.
  # For docs, see: _RAGGED_SEGMENT_DOCSTRING
  return _ragged_segment_aggregate(
      math_ops.unsorted_segment_max,
      data=data,
      segment_ids=segment_ids,
      num_segments=num_segments,
      name=(name or 'RaggedSegmentMax'))


@dispatch.dispatch_for_api(math_ops.unsorted_segment_mean)
def segment_mean(data: ragged_tensor.RaggedOrDense,
                 segment_ids: ragged_tensor.RaggedOrDense,
                 num_segments,
                 name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: assumes data is ragged (reads its flat values and nested row splits), so a dense
  # data input fails with an attribute error although the dispatch signature allows dense input.
  # Ragged dispatch target for the segment mean: computes the segment sum of data and divides by
  # the segment sum of an all-ones tensor with the same row structure (the per-position counts).
  # Inputs: data, segment_ids, num_segments, optional name. Returns a ragged tensor (or a dense
  # one if the sum is dense) of means; positions with a zero count divide by zero.
  """For docs, see: _RAGGED_SEGMENT_DOCSTRING."""
  with ops.name_scope(name, 'RaggedSegmentMean',
                      [data, segment_ids, num_segments]):
    total = segment_sum(data, segment_ids, num_segments)
    ones = ragged_tensor.RaggedTensor.from_nested_row_splits(
        array_ops.ones_like(data.flat_values),
        data.nested_row_splits,
        validate=False)
    count = segment_sum(ones, segment_ids, num_segments)
    if ragged_tensor.is_ragged(total):
      return total.with_flat_values(total.flat_values / count.flat_values)
    else:
      return total / count


@dispatch.dispatch_for_api(math_ops.unsorted_segment_sqrt_n)
def segment_sqrt_n(data: ragged_tensor.RaggedOrDense,
                   segment_ids: ragged_tensor.RaggedOrDense,
                   num_segments,
                   name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: assumes data is ragged (reads its flat values and nested row splits), so a dense
  # data input fails although the dispatch signature allows dense input.
  # Ragged dispatch target for the segment sum divided by the square root of the count. Computes
  # the segment sum of data and the segment sum of an all-ones tensor of the same structure, then
  # divides the sums by the square root of the counts. Inputs: data, segment_ids, num_segments,
  # optional name. Returns a ragged (or dense) tensor.
  """For docs, see: _RAGGED_SEGMENT_DOCSTRING."""
  with ops.name_scope(name, 'RaggedSegmentSqrtN',
                      [data, segment_ids, num_segments]):
    total = segment_sum(data, segment_ids, num_segments)
    ones = ragged_tensor.RaggedTensor.from_nested_row_splits(
        array_ops.ones_like(data.flat_values),
        data.nested_row_splits,
        validate=False)
    count = segment_sum(ones, segment_ids, num_segments)
    if ragged_tensor.is_ragged(total):
      return total.with_flat_values(total.flat_values /
                                    math_ops.sqrt(count.flat_values))
    else:
      return total / math_ops.sqrt(count)


def _set_ragged_segment_docstring(func, combination, combined):
  # [FUNCTION DOC]
  # HELPER: documentation generator for the ragged segment functions.
  # Overwrites the documentation string of func with the segment template filled in with the
  # given verb-like word (combination) and its past participle (combined). Side effect: mutates
  # the function object's doc attribute. Returns nothing.
  func.__doc__ = _RAGGED_SEGMENT_DOCSTRING % dict(
      combination=combination, combined=combined)


_set_ragged_segment_docstring(segment_sum, 'sum', 'summed')
_set_ragged_segment_docstring(segment_prod, 'product', 'multiplied')
_set_ragged_segment_docstring(segment_min, 'minimum', 'minimized')
_set_ragged_segment_docstring(segment_max, 'maximum', 'maximized')
_set_ragged_segment_docstring(segment_mean, 'mean', 'averaged')
_set_ragged_segment_docstring(segment_sqrt_n, 'sum divided by sqrt(N)',
                              'summed')

# ===============================================================================
# ragged_reduce_<AGGREGATE>
# ===============================================================================

# Docstring template used for ragged_reduce_<AGGREGATE> ops.
_RAGGED_REDUCE_DOCSTRING = """\
Computes the %(combination)s of elements across dimensions of a `RaggedTensor`.

  Reduces `input_tensor` along the dimensions given in `axis` by taking the
  %(combination)s of values.  If a reduced dimension has no elements for
  some index, then the value for that index will be %(default)s.

  The rank of the tensor is reduced by `1` for each entry in `axis`.  If
  `axis` is not specified, then all dimensions are reduced, and a scalar
  value is returned.
  Args:
    input_tensor: A `RaggedTensor` containing the values to be %(combined)s.
    axis: The dimensions to reduce.  May be `None` (to reduce all axes), an
      `int` (to reduce a single axis), a `list` or `tuple` of `int` (to reduce
      a given set of axes), or a `Tensor` with a constant value.  Must be in
      the range `[0, input_tensor.rank]`.
    name: A name prefix for the returned tensor (optional).
  Returns:
    A `RaggedTensor` containing the %(combined)s values.  The returned tensor
    has the same dtype as `data`, and its shape is given by removing the
    dimensions specified in `axis` from `input_tensor.shape`.  The `ragged_rank`
    of the returned tensor is given by subtracting any ragged dimensions
    specified in `axis` from `input_tensor.ragged_rank`.
  Raises:
    ValueError: If `axis` contains a `Tensor` whose value is not constant.
  ####Example:
    %(example)s
"""
_RAGGED_REDUCE_SUM_EXAMPLE = """
    >>> rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9], [2, 6]])
    >>> tf.reduce_sum(rt, axis=0).numpy()  # = [3+1+9+2, 1+5+6, 4]
    array([15, 12, 4], dtype=int32)
    >>> tf.reduce_sum(rt, axis=1).numpy()  # = [3+1+4, 1+5, 9, 2+6]
    array([8, 6, 9, 8], dtype=int32)
"""
_RAGGED_REDUCE_PROD_EXAMPLE = """
    >>> rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9], [2, 6]])
    >>> tf.reduce_prod(rt, axis=0).numpy()  # = [3*1*9*2, 1*5*6, 4]
    array([54, 30, 4], dtype=int32)
    >>> tf.reduce_prod(rt, axis=1).numpy()  # = [3*1*4, 1*5, 9, 2*6]
    array([12, 5, 9, 12], dtype=int32)
"""
_RAGGED_REDUCE_MIN_EXAMPLE = """
    >>> rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9], [2, 6]])
    >>> tf.reduce_min(rt, axis=0).numpy()
    array([1, 1, 4], dtype=int32)
    >>> tf.reduce_min(rt, axis=1).numpy()
    array([1, 1, 9, 2], dtype=int32)
"""
_RAGGED_REDUCE_MAX_EXAMPLE = """
    >>> rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9], [2, 6]])
    >>> tf.reduce_max(rt, axis=0).numpy()
    array([9, 6, 4], dtype=int32)
    >>> tf.reduce_max(rt, axis=1).numpy()
    array([4, 5, 9, 6], dtype=int32)
"""
_RAGGED_REDUCE_MEAN_EXAMPLE = """
    >>> rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9], [2, 6]])
    >>> tf.reduce_mean(rt, axis=0).numpy()
    array([3.75, 4.  , 4. ])
    >>> tf.reduce_mean(rt, axis=1).numpy()
    array([2.66666667, 3.  , 9.  , 4.  ])
"""
_RAGGED_REDUCE_VARIANCE_EXAMPLE = """
    >>> rt = tf.ragged.constant([[1, 1, 4], [2, 1], [3], [4, 1]],
    ...                         dtype=tf.float64)
    >>> tf.math.reduce_variance(rt, axis=0).numpy()
    array([1.25, 0., 0.])
    >>> tf.math.reduce_variance(rt, axis=1).numpy()
    array([2., 0.25, 0., 2.25])
"""
_RAGGED_REDUCE_STD_EXAMPLE = """
    >>> rt = tf.ragged.constant([[1, 0], [2, 1], [3], [4, 1]],
    ...                         dtype=tf.float64)
    >>> tf.math.reduce_std(rt, axis=0).numpy()
    array([1.11803399, 0.47140452])
    >>> tf.math.reduce_std(rt, axis=1).numpy()
    array([0.5, 0.5, 0., 1.5])
"""
_RAGGED_REDUCE_ALL_EXAMPLE = """
    >>> rt = tf.ragged.constant([[True, True], [True, True, False, True], [False, True]])
    >>> tf.reduce_all(rt, axis=0).numpy()
    array([False,  True, False,  True])
    >>> tf.reduce_all(rt, axis=1).numpy()
    array([ True, False, False])
"""
_RAGGED_REDUCE_ANY_EXAMPLE = """
    >>> rt = tf.ragged.constant([[True, True], [True, True, False, True], [False, True]])
    >>> tf.reduce_any(rt, axis=0).numpy()
    array([ True,  True, False,  True])
    >>> tf.reduce_any(rt, axis=1).numpy()
    array([ True,  True,  True])
"""


def ragged_reduce_aggregate(reduce_op,
                            unsorted_segment_op,
                            rt_input,
                            axis,
                            keepdims,
                            separator=None,
                            name=None):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the ragged reduce_* functions.
  # Reduces rt_input over the given axis or axes using a dense reduce op for uniform dimensions and
  # an unsorted segment op for ragged ones. Non-ragged input is passed straight to reduce_op. A
  # tensor axis must have a statically known value (otherwise a value error is raised). Axis none
  # reduces the flat values (with keepdims it re-expands one leading dimension per extra input
  # dimension). An empty axis list returns the input unchanged; a single-element list is unwrapped;
  # several axes are made non-negative, sorted and reduced one at a time starting from the
  # innermost (so ops must be associative). Axis 0 aggregates by position across rows, axis 1
  # aggregates each row, higher axes recurse into the values with axis minus one. keepdims adds a
  # length-one dimension back; separator (string data) selects the join variants. Returns a ragged
  # or dense tensor.
  """Aggregates across axes of a RaggedTensor using the given `Tensor` ops.

  Reduces `rt_input` along the dimensions given in `axis`.  The rank of the
  tensor is reduced by 1 for each entry in `axis`.  If `axis` is not specified,
  then all dimensions are reduced, and a scalar value is returned.

  This op assumes that `reduce_op` and `unsorted_segment_op` are associative;
  if not, then reducing multiple axes will return incorrect results.  (In
  particular, reducing multiple axes is currently implemented by reducing the
  axes one at a time.)

  Args:
    reduce_op: The tensorflow `op` that should be used to reduce values in
      uniform dimensions.  Must have the same signature and basic behavior as
      `reduce_sum`, `reduce_max`, etc.
    unsorted_segment_op: The tensorflow `op` that should be used to combine
      values in ragged dimensions.  Must have the same signature and basic
      behavior as `unsorted_segment_sum`, `unsorted_segment_max`, etc.
    rt_input: A `Tensor` or `RaggedTensor` containing the values to be reduced.
    axis: The axis or axes to reduce.  May be `None` (to reduce all axes), an
      `int` (to reduce a single axis), a `list` or `tuple` of `int` (to reduce a
      given set of axes), or a `Tensor` with a constant value.  Must be in the
      range `[0, rt_input.rank)`.
    keepdims: If true, retains reduced dimensions with length 1.
    separator: An optional string. Defaults to None. The separator to use when
      joining. The separator must not be set for non-string data types. (i.e. if
      separator is not None then it uses string ops)
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` containing the reduced values.  The returned tensor
    has the same dtype as `data`, and its shape is given by removing the
    dimensions specified in `axis` from `rt_input.shape`.  The `ragged_rank`
    of the returned tensor is given by subtracting any ragged dimensions
    specified in `axis` from `rt_input.ragged_rank`.
  Raises:
    ValueError: If `axis` contains a `Tensor` whose value is not constant.
  """
  # When separator is not None, We infer that dtype is string and
  # reduce_join will be called.
  if separator is None:
    maybe_separator = {}
  else:
    maybe_separator = {'separator': separator}

  if not ragged_tensor.is_ragged(rt_input):
    return reduce_op(
        rt_input, axis, keepdims=keepdims, name=name, **maybe_separator)

  if isinstance(axis, tensor.Tensor):
    axis = tensor_util.constant_value(axis)
    if axis is None:
      raise ValueError('axis must be known at graph construction time.')
    if isinstance(axis, np.ndarray):
      axis = axis.tolist()

  # When reducing all axes, just ignore splits & reduce the inner values.
  if axis is None:
    result = reduce_op(rt_input.flat_values, None, keepdims=keepdims,
                       name=name, **maybe_separator)
    if keepdims:
      # Expand the result to the input number of dimensions.
      for _ in rt_input.shape[1:]:
        result = array_ops.expand_dims(result, axis=0)
    return result

  with ops.name_scope(name, 'RaggedReduce', [rt_input, axis]):
    if isinstance(axis, (tuple, list)):
      if not axis:
        return rt_input
      elif len(axis) == 1:
        axis = axis[0]
      else:
        # When reducing multiple axes, as we reduce one at a time (see below),
        # the negative axis has to be converted to positive at the first run
        # as the sort with negative axis will have different orders.
        # See GitHub issue 27497.
        axis = [
            array_ops.get_positive_axis(a, rt_input.shape.ndims, 'axis[%s]' % i,
                                        'rank(input_tensor)')
            for i, a in enumerate(axis)
        ]
        # When reducing multiple axes, just reduce one at a time.  This is less
        # efficient, and only works for associative ops.  (In particular, it
        # does not work for reduce_mean.)  However, reducing multiple axes at
        # once will probably require a nontrivial c++ op.
        axis = sorted(axis)
        inner_reduced = ragged_reduce_aggregate(reduce_op, unsorted_segment_op,
                                                rt_input, axis[-1], keepdims,
                                                separator)
        return ragged_reduce_aggregate(reduce_op, unsorted_segment_op,
                                       inner_reduced, axis[:-1], keepdims,
                                       separator)

    rt_input = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        rt_input, name='rt_input')

    axis = array_ops.get_positive_axis(
        axis, rt_input.shape.ndims, ndims_name='rank(input_tensor)')

    if axis == 0:
      # out[i_1, i_2, ..., i_N] = sum_{j} rt_input[j, i_1, i_2, ..., i_N]
      row_lengths = rt_input.row_splits[1:] - rt_input.row_splits[:-1]
      num_segments = math_ops.maximum(math_ops.reduce_max(row_lengths), 0)
      segment_ids = range(row_lengths).values
      result = _ragged_segment_aggregate(unsorted_segment_op, rt_input.values,
                                         segment_ids, num_segments, separator)
      if keepdims:
        result = array_ops.expand_dims(result, axis=0)
      return result
    elif axis == 1:
      # out[i_0, i_1, i_2, ..., i_N] = sum_{j} rt_input[i_0, j, i_2, ..., i_N]
      num_segments = array_ops.shape(rt_input.row_splits)[0] - 1
      segment_ids = segment_id_ops.row_splits_to_segment_ids(
          rt_input.row_splits)
      result = _ragged_segment_aggregate(unsorted_segment_op, rt_input.values,
                                         segment_ids, num_segments, separator)
      if keepdims:
        result = array_ops.expand_dims(result, axis=1)
      return result
    else:
      # out[i_0, ..., i_[axis-1], i_axis+1], ..., i_N] =
      #     sum_{j} rt_input [i_0, ..., i_[axis-1], j, i_axis+1], ..., i_N]
      return rt_input.with_values(
          ragged_reduce_aggregate(reduce_op, unsorted_segment_op,
                                  rt_input.values, axis - 1, keepdims,
                                  separator))


@dispatch.dispatch_for_api(math_ops.reduce_sum)
def reduce_sum(input_tensor: ragged_tensor.Ragged,
               axis=None,
               keepdims=None,
               name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for reduce_sum: sums over the given axis or axes (all axes if none)
  # using the shared reduction helper with the sum and unsorted-segment-sum ops. Inputs: ragged
  # input tensor, axis, keepdims flag (default none), optional name. Returns the reduced tensor.
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""

  return ragged_reduce_aggregate(
      reduce_op=math_ops.reduce_sum,
      unsorted_segment_op=math_ops.unsorted_segment_sum,
      rt_input=input_tensor,
      axis=axis,
      keepdims=keepdims,
      name=(name or 'RaggedReduceSum'))


@dispatch.dispatch_for_api(math_ops.reduce_prod)
def reduce_prod(input_tensor: ragged_tensor.Ragged,
                axis=None,
                keepdims=None,
                name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for reduce_prod: multiplies values over the given axis or axes via the
  # shared reduction helper with the product and unsorted-segment-product ops. Inputs: ragged
  # input tensor, axis, keepdims flag, optional name. Returns the reduced tensor.
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
  return ragged_reduce_aggregate(
      reduce_op=math_ops.reduce_prod,
      unsorted_segment_op=math_ops.unsorted_segment_prod,
      rt_input=input_tensor,
      axis=axis,
      keepdims=keepdims,
      name=(name or 'RaggedReduceProd'))


@dispatch.dispatch_for_api(math_ops.reduce_min)
def reduce_min(input_tensor: ragged_tensor.Ragged,
               axis=None,
               keepdims=None,
               name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for reduce_min: minimum over the given axis or axes via the shared
  # reduction helper with the min and unsorted-segment-min ops. Inputs: ragged input tensor, axis,
  # keepdims flag, optional name. Returns the reduced tensor.
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
  return ragged_reduce_aggregate(
      reduce_op=math_ops.reduce_min,
      unsorted_segment_op=math_ops.unsorted_segment_min,
      rt_input=input_tensor,
      axis=axis,
      keepdims=keepdims,
      name=(name or 'RaggedReduceMin'))


@dispatch.dispatch_for_api(math_ops.reduce_max)
def reduce_max(input_tensor: ragged_tensor.Ragged,
               axis=None,
               keepdims=None,
               name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for reduce_max: maximum over the given axis or axes via the shared
  # reduction helper with the max and unsorted-segment-max ops. Inputs: ragged input tensor, axis,
  # keepdims flag, optional name. Returns the reduced tensor.
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
  return ragged_reduce_aggregate(
      reduce_op=math_ops.reduce_max,
      unsorted_segment_op=math_ops.unsorted_segment_max,
      rt_input=input_tensor,
      axis=axis,
      keepdims=keepdims,
      name=(name or 'RaggedReduceMax'))


@dispatch.dispatch_for_api(math_ops.reduce_mean)
def reduce_mean(input_tensor: ragged_tensor.Ragged,
                axis=None,
                keepdims=None,
                name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for reduce_mean. Computes the reduce_sum over the axis divided by the
  # reduce_sum of an all-ones tensor of the same structure (the element counts), rebuilding the
  # ragged result from the divided flat values. Inputs: ragged input tensor, axis, keepdims flag,
  # optional name. Returns the mean tensor; empty reductions give not-a-number values.
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
  with ops.name_scope(name, 'RaggedReduceMean', [input_tensor, axis]):
    total = reduce_sum(input_tensor, axis, keepdims)
    if ragged_tensor.is_ragged(input_tensor):
      ones = ragged_tensor.RaggedTensor.from_nested_row_splits(
          array_ops.ones_like(input_tensor.flat_values),
          input_tensor.nested_row_splits,
          validate=False)
    else:
      ones = array_ops.ones_like(input_tensor)
    count = reduce_sum(ones, axis, keepdims)
    if ragged_tensor.is_ragged(total):
      return ragged_tensor.RaggedTensor.from_nested_row_splits(
          total.flat_values / count.flat_values,
          total.nested_row_splits,
          validate=False)
    else:
      return total / count


@dispatch.dispatch_for_api(math_ops.reduce_variance)
def reduce_variance(input_tensor: ragged_tensor.Ragged,
                    axis=None,
                    keepdims=False,
                    name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for reduce_variance. Computes mean of squares minus square of the mean
  # and clips the result at zero (the formula is numerically unstable and can go negative).
  # Inputs: ragged input tensor, axis, keepdims flag (default false), optional name. Returns the
  # variance tensor. Raises a value error for complex dtypes.
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
  with ops.name_scope(name, 'RaggedReduceVariance', [input_tensor, axis]):
    input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        input_tensor, name='input_tensor')
    if input_tensor.dtype.is_complex:
      raise ValueError(
          'reduce_variance is not supported for RaggedTensors with complex'
          ' dtypes.'
      )
    square_of_input = math_ops.square(input_tensor)
    mean_of_square = reduce_mean(square_of_input, axis=axis, keepdims=keepdims)
    mean = reduce_mean(input_tensor, axis=axis, keepdims=keepdims)
    square_of_mean = math_ops.square(mean)
    # Note: the above method of computing variance is not numerically stable,
    # and can result in negative variances.  Here we clip to >= 0.
    return math_ops.maximum(mean_of_square - square_of_mean, 0)


@dispatch.dispatch_for_api(math_ops.reduce_std)
def reduce_std(input_tensor: ragged_tensor.Ragged,
               axis=None,
               keepdims=False,
               name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for reduce_std: the square root of the ragged reduce_variance over the
  # given axis. Inputs: ragged input tensor, axis, keepdims flag (default false), optional name.
  # Returns the standard deviation tensor; complex input raises a value error via the variance.
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
  with ops.name_scope(name, 'RaggedReduceStd', [input_tensor, axis]):
    variance = reduce_variance(input_tensor, axis=axis, keepdims=keepdims)
    return math_ops.sqrt(variance)


def _cast(input_tensor, dtype):
  # [FUNCTION DOC]
  # HELPER: dtype conversion for the ragged all/any reductions.
  # Casts a possibly ragged tensor to the given dtype by mapping the cast over its flat values.
  # Returns a tensor with the same structure and the new dtype.
  return ragged_functional_ops.map_flat_values(math_ops.cast, input_tensor,
                                               dtype)


@dispatch.dispatch_for_api(math_ops.reduce_all)
def reduce_all(input_tensor: ragged_tensor.Ragged,
               axis=None,
               keepdims=None,
               name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for reduce_all (logical and). Casts the input to int32, takes the
  # ragged product over the axis, and casts the result back to boolean. Inputs: ragged input
  # tensor, axis, keepdims flag, optional name. Returns a boolean tensor.
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
  with ops.name_scope(name, 'RaggedReduceAll', [input_tensor, axis]):
    return _cast(
        reduce_prod(_cast(input_tensor, dtypes.int32), axis, keepdims),
        dtypes.bool)


@dispatch.dispatch_for_api(math_ops.reduce_any)
def reduce_any(input_tensor: ragged_tensor.Ragged,
               axis=None,
               keepdims=None,
               name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for reduce_any (logical or). Casts the input to int32, takes the ragged
  # sum over the axis, and casts the result back to boolean (nonzero becomes true). Inputs: ragged
  # input tensor, axis, keepdims flag, optional name. Returns a boolean tensor.
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
  with ops.name_scope(name, 'RaggedReduceAny', [input_tensor, axis]):
    return _cast(
        reduce_sum(_cast(input_tensor, dtypes.int32), axis, keepdims),
        dtypes.bool)


def _set_ragged_reduce_docstring(func, combination, combined, default, example):
  # [FUNCTION DOC]
  # HELPER: documentation generator for the ragged reduce functions.
  # Overwrites the documentation string of func with the reduce template filled in with the
  # operation name (combination), its past participle (combined), the text describing the value
  # for empty reductions (default) and an example text. Side effect: mutates the function object's
  # doc attribute. Returns nothing.
  func.__doc__ = _RAGGED_REDUCE_DOCSTRING % dict(
      combination=combination,
      combined=combined,
      default=default,
      example=example)


_set_ragged_reduce_docstring(reduce_sum, 'sum', 'summed', '0',
                             _RAGGED_REDUCE_SUM_EXAMPLE)
_set_ragged_reduce_docstring(reduce_prod, 'product', 'multiplied', '1',
                             _RAGGED_REDUCE_PROD_EXAMPLE)
_set_ragged_reduce_docstring(reduce_min, 'minimum', 'minimized',
                             '`input_tensor.dtype.min`',
                             _RAGGED_REDUCE_MIN_EXAMPLE)
_set_ragged_reduce_docstring(reduce_max, 'maximum', 'maximized',
                             '`input_tensor.dtype.max`',
                             _RAGGED_REDUCE_MAX_EXAMPLE)
_set_ragged_reduce_docstring(reduce_mean, 'mean', 'averaged', 'NaN',
                             _RAGGED_REDUCE_MEAN_EXAMPLE)
_set_ragged_reduce_docstring(reduce_variance, 'variance', 'averaged', 'NaN',
                             _RAGGED_REDUCE_VARIANCE_EXAMPLE)
_set_ragged_reduce_docstring(reduce_std, 'std', 'averaged', 'NaN',
                             _RAGGED_REDUCE_STD_EXAMPLE)
_set_ragged_reduce_docstring(reduce_all, 'logical and', 'and-ed', 'True',
                             _RAGGED_REDUCE_ALL_EXAMPLE)
_set_ragged_reduce_docstring(reduce_any, 'logical or', 'or-ed', 'False',
                             _RAGGED_REDUCE_ANY_EXAMPLE)


# ===============================================================================
# ragged.matmul
# ===============================================================================
@dispatch.dispatch_for_api(math_ops.matmul)
def matmul(
    a: ragged_tensor.RaggedOrDense,
    b: ragged_tensor.RaggedOrDense,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,
    grad_a=False,
    grad_b=False,
    name=None,
):
  # [FUNCTION DOC]
  # Ragged dispatch target for matmul. If neither input is ragged, defers to the dense matmul.
  # Otherwise requires matching dtypes and equal known ranks (at least one rank must be known and
  # both at least 2). Rank above 3: converts dense operands to ragged, asserts the outer row
  # splits match, and recursively multiplies the values. Rank 2: converts ragged operands to dense
  # (checking they are actually uniform) and multiplies. Rank 3: uses batch-dimension folding when
  # a has ragged rank 1 and b is dense and a is not transposed or adjointed; otherwise uses a map
  # over the batch. Inputs: a, b, transpose/adjoint/sparse flags, output_type (optional), grad_a
  # and grad_b (deleted and ignored), name. Returns a dense or ragged tensor. Raises a value error
  # if both transpose and adjoint are set for one operand, dtypes differ, ranks differ or are
  # unknown, or rank is below 2.
  """Multiplies matrix `a` by matrix `b`.

  If all transpose or adjoint attributes are `False` then:

  ```
  output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j.
  ```

  The inputs `a` and `b` must have `rank >= 2`, where the outermost `rank - 2`
  dimensions are batch dimensions.  The inputs must have the same dtype.  See
  `tf.matmul` for more information.

  Args:
    a: `tf.Tensor` or `RaggedTensor` with `rank > 1`.
    b: `tf.Tensor` or `RaggedTensor` with same type and rank as `a`.
    transpose_a: If `True`, `a` is transposed before multiplication.
    transpose_b: If `True`, `b` is transposed before multiplication.
    adjoint_a: If `True`, `a` is conjugated & transposed before multiplication.
    adjoint_b: If `True`, `b` is conjugated & transposed before multiplication.
    a_is_sparse: If `True`, optimize assuming `a` is mostly zero.
    b_is_sparse: If `True`, optimize assuming `b` is mostly zero.
    output_type: The output datatype (optional).
    grad_a: Unused.
    grad_b: Unused.
    name: Name for the operation (optional).

  Returns:
    A `Tensor` or `RaggedTensor` with the same rank and shape as `a`, where
    each inner-most matrix is the product of the corresponding matrices in `a`
    and `b`.
  """
  del grad_a
  del grad_b
  if transpose_a and adjoint_a:
    raise ValueError('Only one of transpose_a and adjoint_a can be True.')
  if transpose_b and adjoint_b:
    raise ValueError('Only one of transpose_b and adjoint_b can be True.')

  kwargs = dict(
      transpose_a=transpose_a,
      transpose_b=transpose_b,
      adjoint_a=adjoint_a,
      adjoint_b=adjoint_b,
      a_is_sparse=a_is_sparse,
      b_is_sparse=b_is_sparse,
      output_type=output_type)

  with ops.name_scope(name, 'RaggedMatMul', [a, b]) as name:
    a = ragged_tensor.convert_to_tensor_or_ragged_tensor(a, name='a')
    b = ragged_tensor.convert_to_tensor_or_ragged_tensor(b, name='b')

    a_is_ragged = isinstance(a, ragged_tensor.RaggedTensor)
    b_is_ragged = isinstance(b, ragged_tensor.RaggedTensor)
    if not (a_is_ragged or b_is_ragged):
      return math_ops.matmul(a, b, **kwargs)

    if a.dtype != b.dtype:
      raise ValueError('`a` and `b` must have the same dtype.')

    # TODO(edloper): Support broadcasting inputs.  (Broadcast support is not
    # documented by https://www.tensorflow.org/api_docs/python/tf/linalg/matmul,
    # but it is supported by the op.)

    # Find the rank of the input tensors.
    if a.shape.rank is None:
      if b.shape.rank is None:
        raise ValueError('matmul requires at least one input to have known '
                         'rank if either input is ragged.')
      rank = b.shape.rank
    else:
      if b.shape.rank is not None and a.shape.rank != b.shape.rank:
        raise ValueError('`a` and `b` must have the same rank.')
      rank = a.shape.rank

    # At least one of `a` and `b` is ragged; and ragged tensors always have
    # rank>=2.
    if rank < 2:
      # This can happen if e.g. `a` is a 1D dense tensor and `b` is a
      # ragged tensor with unknown rank.  Since ragged tensors always have
      # `rank>=2`, this implies that `a` and `b` have different ranks.
      raise ValueError('`a` and `b` must have the same rank.')

    # Rank>3: We have multiple batch dimensions.  Merge them into a single
    # batch dimension, recursively call `matmul`, and then restore the original
    # batch dimension (using a.row_splits).
    if rank > 3:
      shape_err = 'Batch dimensions of `a` and `b` do not have the same size.'
      if not a_is_ragged:
        a = ragged_tensor.RaggedTensor.from_tensor(a, ragged_rank=1)
      if not b_is_ragged:
        b = ragged_tensor.RaggedTensor.from_tensor(b, ragged_rank=1)
      with ops.control_dependencies([
          check_ops.assert_equal(a.row_splits, b.row_splits, message=shape_err)
      ]):
        flat_result = matmul(a.values, b.values, **kwargs)
        return a.with_values(flat_result)

    if rank == 2:
      return _matmul_2d(a, b, **kwargs)

    assert rank == 3  # I.e., we have a single batch dimension.

    a_ragged_rank = a.ragged_rank if a_is_ragged else 0
    if a_ragged_rank == 1 and not (b_is_ragged or transpose_a or adjoint_a):
      # If `a.shape=[B, (I), J]` and `b.shape=[B, J, K], then we can compute
      # the result with a single dense `matmul`.
      return _matmul_3d_with_batch_dim_folding(a, b, **kwargs)
    else:
      # Otherwie, fall back on using `map_fn`.
      return _matmul_3d_with_map_fn(a, b, **kwargs)


def _matmul_2d(a, b, **kwargs):
  # [FUNCTION DOC]
  # HELPER: rank-2 case of the ragged matmul.
  # Converts any ragged operand to a dense tensor, adding an assertion that no padding was needed
  # (the matrix must be uniform), then multiplies with the dense matmul passing extra keyword
  # arguments through. Returns a dense 2D tensor. The assertion fails with an invalid-argument
  # error at run time when a matrix is ragged in its inner dimension.
  """Multiplies potentially ragged 2D tensors.

  Args:
    a: A 2D Tensor or RaggedTensor with `shape=[I, J]`
    b: A 2D Tensor or RaggedTensor with `shape=[J, K]`
    **kwargs: Additional arguments for `tf.matmul` (e.g. transpose_a).

  Returns:
    A 2D Tensor with `shape=[I, K]`.
  """
  # multiplying `a` and `b` is only well-defined if `a` and `b` are
  # actually uniform (and just happened to be stored as ragged tensors).
  # Check that they're uniform, convert them to tf.Tensor.
  ragged_err = ('The matrices in `a` and `b` may not be '
                'ragged in their innermost dimension.')
  checks = []
  if isinstance(a, ragged_tensor.RaggedTensor):
    original_size = array_ops.size(a.flat_values)
    a = a.to_tensor()
    checks.append(
        check_ops.assert_equal(
            original_size, array_ops.size(a), message=ragged_err))
  if isinstance(b, ragged_tensor.RaggedTensor):
    original_size = array_ops.size(b.flat_values)
    b = b.to_tensor()
    checks.append(
        check_ops.assert_equal(
            original_size, array_ops.size(b), message=ragged_err))
  with ops.control_dependencies(checks):
    return math_ops.matmul(a, b, **kwargs)


def _matmul_3d_with_map_fn(a, b, **kwargs):
  # [FUNCTION DOC]
  # HELPER: rank-3 case of the ragged matmul (general fallback).
  # Multiplies each batch element pair using a map over the batch with the 2D helper. The result
  # has ragged rank 2 when b has ragged rank 2 or is transposed/adjointed, otherwise ragged rank 1.
  # The output element type is output_type from the keyword arguments or a's dtype. Afterwards the
  # statically known result shape is restored where possible. Returns a 3D ragged tensor. Errors
  # come from the 2D helper's uniformity check.
  # Nested function: multiplies one batch element (a pair of matrices) and converts the result to
  # ragged form when the output ragged rank is 2.
  """Multiplies batches of 2D matrices using map_fn.

  `output[n, i, k]` = sum_j (a[n, i, j] * b[n, j, k])` (for all `n`, `i`, `k`).

  Requires that `a[n, i].nrows()` == `b[n].nrows()` (for all `n` and `i`).

  Args:
    a: A 3D Tensor or RaggedTensor with `shape=[B, I, J]`, where dimensions `I`
      and `J` may be ragged.
    b: A 3D Tensor or RaggedTensor with `shape=[B, J, K]`, where dimensions `J`
      and `K` may be ragged.
    **kwargs: Additional arguments for `tf.matmul` (e.g. transpose_a).

  Returns:
    A 3D RaggedTensor with `shape=[B, (I), (K)]`.
  """
  # Determine the ragged rank of the result.  In the normal case, we have:
  #   [B, I, J] * [B, J, K] -> [B, I, K]
  # Or if we're using transpose_b, then we have:
  #   [B, I, J] * [B, K, J] -> [B, I, K]
  # In either case, output_ragged_rank=2 iff the K dimension is ragged.
  if (isinstance(b, ragged_tensor.RaggedTensor) and
      (b.ragged_rank == 2 or kwargs.get('transpose_b') or
       kwargs.get('adjoint_b'))):
    output_ragged_rank = 2
  else:
    output_ragged_rank = 1

  def single_batch_matmul(x):
    # [FUNCTION DOC]
    # HELPER: per-batch-element body for the map over the batch.
    # Takes a pair of matrices, multiplies them with the 2D helper and, when the output ragged
    # rank is 2, converts the dense product to a ragged tensor. Returns the product.
    out = _matmul_2d(x[0], x[1], **kwargs)
    if output_ragged_rank == 2:
      out = ragged_tensor.RaggedTensor.from_tensor(out)
    return out

  fn_out_shape = None  # Figure out proper shape.
  row_splits_dtype = (
      a.row_splits.dtype
      if isinstance(a, ragged_tensor.RaggedTensor) else b.row_splits.dtype)
  output_type = kwargs['output_type']
  if output_type is None:
    output_type = a.dtype
  spec = ragged_tensor.RaggedTensorSpec(
      shape=fn_out_shape,
      dtype=output_type,
      ragged_rank=output_ragged_rank - 1,
      row_splits_dtype=row_splits_dtype)
  result = map_fn.map_fn(
      single_batch_matmul, elems=(a, b), fn_output_signature=spec)

  # map_fn loses shape information; restore it, where possible.
  # pylint: disable=protected-access
  if kwargs.get('transpose_a') or kwargs.get('adjoint_a'):
    result._set_shape(a.shape[:-2] + a.shape[-1:] + [None])
  else:
    result._set_shape(a.shape[:-2] + a.shape[-2:-1] + [None])
  if kwargs.get('transpose_b') or kwargs.get('adjoint_b'):
    result._set_shape(b.shape[:-2] + [None] + b.shape[-2:-1])
  else:
    result._set_shape(b.shape[:-2] + [None] + b.shape[-1:])

  return result


def _matmul_3d_with_batch_dim_folding(a, b, **kwargs):
  # [FUNCTION DOC]
  # HELPER: fast rank-3 case of the ragged matmul.
  # Handles a with ragged rank 1 and dense b: repeats each batch's b matrix once per row of a,
  # multiplies all rows in a single dense matmul, and re-wraps the result with a's row structure.
  # Returns a ragged tensor. Transposing or adjointing a is not allowed (caller guarantees it).
  """Multiply batches of 2D matrices where only `a.shape[1]` is ragged.

  Args:
    a: A RaggedTensor with `shape=[B, (I), J]`.  (ragged_rank must be 1.)
    b: A Tensor with `shape=[B, J, K]`
    **kwargs: Additional arguments for `tf.matmul` (e.g. transpose_a).
      transpose_a and adjoint_a must not be true.

  Returns:
    A RaggedTensor with `shape=[B, (I), K].
  """
  # reshaped_a.shape = [sum(i_1, i_2, ..., i_B), 1, J]
  reshaped_a = array_ops.expand_dims(a.values, 1)
  # reshaped_b.shape = [sum(i_1, i_2, ..., i_B), J, K]
  reshaped_b = array_ops.repeat(b, a.row_lengths(), axis=0)
  # flat_result.shape = [sum(i_1, i_2, ..., i_B), 1, K]
  flat_result = math_ops.matmul(reshaped_a, reshaped_b, **kwargs)
  # result.shape = [B, (I), K]
  return a.with_values(array_ops.squeeze(flat_result, axis=1))


# ===============================================================================
# ragged.softmax
# ===============================================================================
@dispatch.dispatch_for_api(nn_ops.softmax_v2)
def softmax(logits: ragged_tensor.Ragged, axis=None, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring example and Raises section describe the dense softmax; this ragged
  # version does no emptiness or axis validation itself.
  # Ragged dispatch target for softmax. Subtracts the per-axis maximum for stability,
  # exponentiates, and divides by the per-axis sum, using ragged reductions. Inputs: ragged logits,
  # axis (default none meaning the last axis), optional name. Returns a ragged tensor of the same
  # shape. Errors come from the reductions.
  """Computes softmax activations.

  Used for multi-class predictions. The sum of all outputs generated by softmax
  is 1.

  This function performs the equivalent of

      softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)

  Example usage:

  >>> softmax = tf.nn.softmax([-1, 0., 1.])
  >>> softmax
  <tf.Tensor: shape=(3,), dtype=float32,
  numpy=array([0.09003057, 0.24472848, 0.66524094], dtype=float32)>
  >>> sum(softmax)
  <tf.Tensor: shape=(), dtype=float32, numpy=1.0>

  Args:
    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    axis: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type and shape as `logits`.

  Raises:
    InvalidArgumentError: if `logits` is empty or `axis` is beyond the last
      dimension of `logits`.
  """
  if axis is None:
    axis = -1

  with ops.name_scope(name, 'RaggedSoftmax', [logits]) as name:
    max_input = reduce_max(logits, axis=axis, keepdims=True)
    logits_exp = math_ops.exp(math_ops.subtract(logits, max_input))
    denominator = reduce_sum(logits_exp, axis=axis, keepdims=True)
    return math_ops.divide(logits_exp, denominator)


# ===============================================================================
# ragged.add_n
# ===============================================================================
@dispatch.dispatch_for_api(math_ops.add_n)
def add_n(inputs: typing.List[ragged_tensor.RaggedOrDense], name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the empty-input check tests whether the length is below zero, which is never
  # true, so an empty list is not rejected here despite the error message.
  # Ragged dispatch target for add_n: element-wise sum of a list of ragged or dense tensors,
  # computed by mapping the dense add_n over the flat values. Inputs: list of tensors, optional
  # name. Returns the summed tensor.
  """RaggedTensor implementation for tf.math.add_n."""
  if len(inputs) < 0:
    raise ValueError('tf.add_n: expected at least one input.')
  with ops.name_scope(name, 'RaggedAddN', inputs):
    return ragged_functional_ops.map_flat_values(math_ops.add_n, inputs)


# ===============================================================================
# Ragged version of nn_ops.dropout
# ===============================================================================
@dispatch.dispatch_for_api(nn_ops.dropout)
def dropout_v1(x: ragged_tensor.Ragged,
               keep_prob=None,
               noise_shape=None,
               seed=None,
               name=None,
               rate=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for the legacy dropout (keep probability or rate). Applies dropout to
  # the flat values and rebuilds the ragged tensor. Inputs: ragged x, keep_prob, noise_shape (must
  # be none), seed, name, rate. Returns a ragged tensor. Raises a value error if noise_shape is
  # given. Uses randomness.
  """Ragged dispatch target for tf.nn.dropout."""
  if noise_shape is not None:
    raise ValueError('noise_shape is not supported yet for RaggedTensor x')
  with ops.name_scope(name, 'RaggedNNDropout', [x, rate]):
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
    return x.with_flat_values(
        nn_ops.dropout(
            x.flat_values, keep_prob=keep_prob, seed=seed, rate=rate))


@dispatch.dispatch_for_api(nn_ops.dropout_v2)
def dropout_v2(x: ragged_tensor.Ragged,
               rate,
               noise_shape=None,
               seed=None,
               name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for dropout (version 2). Applies dropout with the given rate to the
  # flat values and rebuilds the ragged tensor. Inputs: ragged x, rate, noise_shape (must be
  # none), seed, name. Returns a ragged tensor. Raises a value error if noise_shape is given.
  # Uses randomness.
  """Ragged dispatch target for tf.nn.dropout."""
  if noise_shape is not None:
    raise ValueError('noise_shape is not supported yet for RaggedTensor x')
  with ops.name_scope(name, 'RaggedNNDropout', [x, rate]):
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
    return x.with_flat_values(
        nn_ops.dropout_v2(x.flat_values, rate=rate, seed=seed))


@dispatch.dispatch_for_api(nn_ops.stateless_dropout)
def stateless_dropout(x: ragged_tensor.Ragged,
                      rate,
                      seed,
                      rng_alg=None,
                      noise_shape=None,
                      name=None):
  # [FUNCTION DOC]
  # Ragged dispatch target for stateless dropout. Applies deterministic dropout, driven by seed
  # and an optional random algorithm, to the flat values. Inputs: ragged x, rate, seed, rng_alg,
  # noise_shape (must be none), name. Returns a ragged tensor. Raises a value error if noise_shape
  # is given.
  """Ragged dispatch target for tf.nn.experimental.stateless_dropout."""
  if noise_shape is not None:
    raise ValueError('noise_shape is not supported yet for RaggedTensor x')
  with ops.name_scope(name, 'RaggedNNStatelessDropout', [x, rate]):
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
    return x.with_flat_values(
        nn_ops.stateless_dropout(
            x.flat_values, rate=rate, seed=seed, rng_alg=rng_alg))


# ===============================================================================
# Ragged version of Tensor.__eq__ and Tensor.__ne__
# ===============================================================================
@dispatch.dispatch_for_api(math_ops.tensor_equals)
def tensor_equals(self: ragged_tensor.RaggedOrDense,
                  other: ragged_tensor.RaggedOrDense):
  # [FUNCTION DOC]
  # Ragged version of tensor equality. Returns false if other is nothing; in legacy mode returns
  # whether both are the same object; otherwise returns the element-wise equal result, or plain
  # false if the shapes cannot be broadcast (invalid-argument or value errors are swallowed).
  """Ragged version of the operation invoked by `Tensor.__eq__`."""
  if other is None:
    return False
  elif _use_legacy_mode_for_tensor_equality(self):
    return self is other
  else:
    try:
      return math_ops.equal(self, other)
    except (errors.InvalidArgumentError, ValueError):
      return False  # values are not broadcast-compatible.


@dispatch.dispatch_for_api(math_ops.tensor_not_equals)
def tensor_not_equals(self: ragged_tensor.RaggedOrDense,
                      other: ragged_tensor.RaggedOrDense):
  # [FUNCTION DOC]
  # KNOWN ISSUE: returns false when other is nothing, although a tensor is "not equal" to nothing;
  # this mirrors the equality function instead of inverting it.
  # Ragged version of tensor inequality. In legacy mode returns whether the objects differ;
  # otherwise the element-wise not-equal result, or plain true if the shapes cannot be broadcast.
  """Ragged version of the operation invoked by `Tensor.__ne__`."""
  if other is None:
    return False
  elif _use_legacy_mode_for_tensor_equality(self):
    return self is not other
  else:
    try:
      return math_ops.not_equal(self, other)
    except (errors.InvalidArgumentError, ValueError):
      return True  # values are not broadcast-compatible.


def _use_legacy_mode_for_tensor_equality(self):
  # [FUNCTION DOC]
  # HELPER: mode switch for the ragged equality and inequality functions.
  # Returns true unless element-wise tensor equality is enabled, eager execution is active outside
  # functions, and the tensor has no graph or its graph is a function-building graph.
  g = getattr(self, 'graph', None)
  return not (tensor.Tensor._USE_EQUALITY and  # pylint: disable=protected-access
              ops.executing_eagerly_outside_functions() and
              (g is None or g.building_function))


def _cumsum_flat_values_at_ragged_rank(last_rp, flat_values, exclusive=False,
                                       reverse=False):
  # [FUNCTION DOC]
  # HELPER: cumulative sum along the innermost ragged dimension.
  # Computes per-row cumulative sums over the flat values, where rows are defined by the last row
  # partition. The inclusive form is the exclusive form plus the values (recursion). The exclusive
  # form takes a global exclusive cumulative sum and subtracts the value at each row's first
  # (or, when reversed, one past its last) position so each row restarts at zero. Inputs: last row
  # partition, flat values, exclusive and reverse flags. Returns the new flat values.
  """Calculate flat_values for math_ops.cumsum when axis==ragged_rank."""
  if not exclusive:
    partial = _cumsum_flat_values_at_ragged_rank(
        last_rp, flat_values, exclusive=True, reverse=reverse)
    return partial + flat_values

  if reverse:
    youngest_sibling = array_ops.gather(
        params=last_rp.row_splits(), indices=last_rp.value_rowids() + 1) - 1
    new_flat_values = math_ops.cumsum(flat_values, exclusive=True, reverse=True)
    initial_values = array_ops.gather(params=new_flat_values,
                                      indices=youngest_sibling)

    return new_flat_values - initial_values
  else:
    eldest_sibling = array_ops.gather(
        params=last_rp.row_splits(), indices=last_rp.value_rowids())
    new_flat_values = math_ops.cumsum(flat_values, exclusive=True)
    initial_values = array_ops.gather(params=new_flat_values,
                                      indices=eldest_sibling)
    return new_flat_values - initial_values


@dispatch.dispatch_for_api(math_ops.cumsum)
def ragged_cumsum(x: ragged_tensor.Ragged,
                  axis: int = 0,
                  exclusive: bool = False,
                  reverse: bool = False,
                  name: typing.Optional[str] = None):
  # [FUNCTION DOC]
  # Ragged dispatch target for cumsum. Normalizes the axis to non-negative. If the axis is the
  # innermost ragged dimension, uses the row-wise flat-value helper; if it is deeper, maps a dense
  # cumsum over the flat values with the shifted axis; if it is shallower, densifies, applies the
  # dense cumsum and converts back using the original row lengths. Inputs: ragged x, axis
  # (default 0), exclusive flag, reverse flag, optional name. Returns a ragged tensor of the same
  # shape. Errors for an invalid axis come from axis normalization.
  """Calculate math_ops.cumsum for a RaggedTensor.

  Given a ragged tensor `x`, the `result` is a ragged tensor with the same
  shape. One can calculate the value of `result[i_1...i_k]` as follows:
  ```
  dense_result=tf.math.cumsum(rt.to_tensor(), axis=axis, exclusive=exclusive,
                              reverse=reverse)
  result[i_1...i_k]=dense_result[i_1...i_k]
  ```

  Args:
    x: the original ragged tensor to sum.
    axis: the axis along which to sum, can range -rank<=axis<rank.
    exclusive: is the sum exclusive or inclusive? If True, then result[0]=0.
        If False, then result[0]=x[0].
    reverse: If True, sum from back to front.
    name: the name of the op.
  Returns:
    the cumulative sum.
  """
  with ops.name_scope(name, 'RaggedCumSum', [x, axis, exclusive, reverse]):
    axis = array_ops.get_positive_axis(axis, x.shape.rank, ndims_name='rank')
    if axis == x.ragged_rank:
      last_rp = x._nested_row_partitions[-1]  # pylint: disable=protected-access
      return x.with_flat_values(
          _cumsum_flat_values_at_ragged_rank(last_rp, x.flat_values,
                                             exclusive=exclusive,
                                             reverse=reverse))
    elif axis > x.ragged_rank:
      new_axis = axis - x.ragged_rank
      cumsum_bound = functools.partial(
          math_ops.cumsum, axis=new_axis, exclusive=exclusive, reverse=reverse)
      return ragged_functional_ops.map_flat_values(cumsum_bound, x)
    else:
      dense_version = x.to_tensor()
      result = math_ops.cumsum(
          dense_version, axis, exclusive=exclusive, reverse=reverse, name=name)
      return ragged_tensor.RaggedTensor.from_tensor(
          result, lengths=x.nested_row_lengths())
