# [MODULE DOC]
# Concatenation and stacking of ragged (and mixed ragged/dense) tensors. Exposes the ragged
# handler for the generic concat API, and the public ragged stack function (also exported
# under the ragged namespace and registered as a handler for the generic stack API). Both
# delegate to one shared engine that unifies inputs to a common ragged rank and handles
# axis 0 by joining flat values and row splits, axis 1 by joining and then permuting rows,
# and higher axes by recursing on the values. Private helpers at the bottom adjust ragged
# rank, merge row splits and copy static shapes. Import-time side effects: dispatch and
# export registration for the two public functions.
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
"""Concat and stack operations for RaggedTensors."""

import typing

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@dispatch.dispatch_for_api(array_ops.concat)
def concat(values: typing.List[ragged_tensor.RaggedOrDense], axis, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says only a value error is raised, but a non-integer axis
  # raises a type error (see the tests), and a mismatch of ranks can also surface as an
  # invalid-argument error at run time. A single non-list input is accepted, undocumented.
  # Dispatch handler for the generic concat API when any input is ragged. Inputs: values (a
  # list or tuple of ragged or dense tensors of equal rank and dtype; a single non-list
  # value is wrapped into a one-element list), axis (a statically known integer, negative
  # allowed if rank is known) and name (optional scope name, default "RaggedConcat").
  # Delegates to the shared stack/concat engine in concatenate mode and returns the
  # concatenated tensor (a ragged tensor, or the input itself when there is just one, or a
  # dense tensor if all inputs are dense and the last axis is joined). Raises value errors
  # for an empty list, out-of-bounds axis, or rank mismatch. No side effects.
  """Concatenates potentially ragged tensors along one dimension.

  Given a list of tensors with the same rank `K` (`K >= axis`), returns a
  rank-`K` `RaggedTensor` `result` such that `result[i0...iaxis]` is the
  concatenation of `[rt[i0...iaxis] for rt in values]`.

  Args:
    values: A list of potentially ragged tensors.  May not be empty. All
      `values` must have the same rank and the same dtype; but unlike
      `tf.concat`, they can have arbitrary shapes.
    axis: A python integer, indicating the dimension along which to concatenate.
      (Note: Unlike `tf.concat`, the `axis` parameter must be statically known.)
        Negative values are supported only if the rank of at least one
        `values` value is statically known.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` with rank `K`.
    `result.ragged_rank=max(axis, max(rt.ragged_rank for rt in values]))`.

  Raises:
    ValueError: If `values` is empty, if `axis` is out of bounds or if
      the input tensors have different ranks.

  #### Example:

  >>> t1 = tf.ragged.constant([[1, 2], [3, 4, 5]])
  >>> t2 = tf.ragged.constant([[6], [7, 8, 9]])
  >>> tf.concat([t1, t2], axis=0)
  <tf.RaggedTensor [[1, 2], [3, 4, 5], [6], [7, 8, 9]]>
  >>> tf.concat([t1, t2], axis=1)
  <tf.RaggedTensor [[1, 2, 6], [3, 4, 5, 7, 8, 9]]>
  """
  if not isinstance(values, (list, tuple)):
    values = [values]
  with ops.name_scope(name, 'RaggedConcat', values):
    return _ragged_stack_concat_helper(values, axis, stack_values=False)


@tf_export('ragged.stack')
@dispatch.add_dispatch_support
@dispatch.dispatch_for_api(array_ops_stack.stack)
def stack(values: typing.List[ragged_tensor.RaggedOrDense],
          axis=0,
          name=None):
  # [FUNCTION DOC]
  # Public ragged stack (also registered as the handler of the generic stack API). Inputs:
  # values (list or tuple of dense or ragged tensors with equal rank and dtype; a single
  # non-list value is wrapped into a list), axis (statically known integer, default 0;
  # negative allowed if rank is known) and name (optional scope name, default
  # "RaggedConcat"). Delegates to the shared engine in stack mode, which adds a new
  # dimension at axis. Returns a ragged tensor of rank one higher than the inputs (a dense
  # vector result is possible for dense rank-0 inputs, as documented). Raises value errors
  # for an empty list, out-of-bounds axis or rank mismatch. No side effects.
  """Stacks a list of rank-`R` tensors into one rank-`(R+1)` `RaggedTensor`.

  Given a list of tensors or ragged tensors with the same rank `R`
  (`R >= axis`), returns a rank-`R+1` `RaggedTensor` `result` such that
  `result[i0...iaxis]` is `[value[i0...iaxis] for value in values]`.

  #### Examples:

  >>> # Stacking two ragged tensors.
  >>> t1 = tf.ragged.constant([[1, 2], [3, 4, 5]])
  >>> t2 = tf.ragged.constant([[6], [7, 8, 9]])
  >>> tf.ragged.stack([t1, t2], axis=0)
  <tf.RaggedTensor [[[1, 2], [3, 4, 5]], [[6], [7, 8, 9]]]>
  >>> tf.ragged.stack([t1, t2], axis=1)
  <tf.RaggedTensor [[[1, 2], [6]], [[3, 4, 5], [7, 8, 9]]]>

  >>> # Stacking two dense tensors with different sizes.
  >>> t3 = tf.constant([[1, 2, 3], [4, 5, 6]])
  >>> t4 = tf.constant([[5], [6], [7]])
  >>> tf.ragged.stack([t3, t4], axis=0)
  <tf.RaggedTensor [[[1, 2, 3], [4, 5, 6]], [[5], [6], [7]]]>

  Args:
    values: A list of `tf.Tensor` or `tf.RaggedTensor`.  May not be empty. All
      `values` must have the same rank and the same dtype; but unlike
      `tf.stack`, they can have arbitrary dimension sizes.
    axis: A python integer, indicating the dimension along which to stack.
      (Note: Unlike `tf.stack`, the `axis` parameter must be statically known.)
      Negative values are supported only if the rank of at least one
      `values` value is statically known.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` with rank `R+1` (if `R>0`).
    If `R==0`, then the result will be returned as a 1D `Tensor`, since
    `RaggedTensor` can only be used when `rank>1`.
    `result.ragged_rank=1+max(axis, max(rt.ragged_rank for rt in values]))`.

  Raises:
    ValueError: If `values` is empty, if `axis` is out of bounds or if
      the input tensors have different ranks.
  """
  if not isinstance(values, (list, tuple)):
    values = [values]
  with ops.name_scope(name, 'RaggedConcat', values):
    return _ragged_stack_concat_helper(values, axis, stack_values=True)


def _ragged_stack_concat_helper(rt_inputs, axis, stack_values):
  # [FUNCTION DOC]
  # HELPER: shared engine behind the public ragged concat and stack functions.
  # Inputs: rt_inputs (list of dense or ragged tensors), axis (integer, negative values are
  # resolved against the output rank) and stack_values (true to stack, false to
  # concatenate). Steps: reject an empty list with a value error; convert inputs to tensors
  # and match their row-splits dtypes; return the only input unchanged when concatenating
  # a single tensor; check all ranks agree; special case stacking vectors on axis 0 by
  # building a ragged tensor from row lengths; delegate to the dense stack/concat when all
  # inputs are dense and the last axis is used; otherwise turn dense inputs into ragged ones,
  # raise all to the same ragged rank, and dispatch to the axis-0 or axis-1 routine, or for
  # higher axes recurse on the inner values after asserting identical row splits.
  # Returns a ragged (or dense) tensor.
  """Helper function to concatenate or stack ragged tensors.

  Args:
    rt_inputs: A list of RaggedTensors or Tensors to combine.
    axis: The axis along which to concatenate or stack.
    stack_values: A boolean -- if true, then stack values; otherwise,
      concatenate them.

  Returns:
    A RaggedTensor.
  Raises:
    ValueError: If rt_inputs is empty, or if axis is out of range.
  """
  # Validate parameters.
  if not rt_inputs:
    raise ValueError('rt_inputs may not be empty.')

  # Convert input tensors.
  rt_inputs = [
      ragged_tensor.convert_to_tensor_or_ragged_tensor(
          rt_input, name='rt_input') for rt_input in rt_inputs
  ]
  row_splits_dtype, rt_inputs = ragged_tensor.match_row_splits_dtypes(
      *rt_inputs, return_dtype=True)
  rt_inputs = list(rt_inputs)

  # Special case: if there's only one input, then return it as-is.
  if len(rt_inputs) == 1 and not stack_values:
    return rt_inputs[0]

  # Check the rank (number of dimensions) of the input tensors.
  ndims = None
  for rt in rt_inputs:
    if ndims is None:
      ndims = rt.shape.ndims
    else:
      rt.shape.assert_has_rank(ndims)

  out_ndims = ndims if (ndims is None or not stack_values) else ndims + 1
  axis = array_ops.get_positive_axis(axis, out_ndims)

  if stack_values and ndims == 1 and axis == 0:
    return ragged_tensor.RaggedTensor.from_row_lengths(
        values=array_ops.concat(rt_inputs, axis=0),
        row_lengths=array_ops.concat([array_ops.shape(r) for r in rt_inputs],
                                     axis=0))

  # If all the inputs are Tensors, and we're combining the final dimension,
  # then we can delegate to the tf.stack/tf.concat operation, and return a
  # Tensor.
  if all(not ragged_tensor.is_ragged(rt) for rt in rt_inputs):
    if ndims is not None and (axis == out_ndims - 1 or axis == ndims - 1):
      if stack_values:
        return array_ops_stack.stack(rt_inputs, axis)
      else:
        return array_ops.concat(rt_inputs, axis)

  # Convert any Tensor inputs to RaggedTensors.  This makes it
  # possible to concatenate Tensors and RaggedTensors together.
  for i in range(len(rt_inputs)):
    if not ragged_tensor.is_ragged(rt_inputs[i]):
      rt_inputs[i] = ragged_tensor.RaggedTensor.from_tensor(
          rt_inputs[i], ragged_rank=1, row_splits_dtype=row_splits_dtype)

  # Convert the input tensors to all have the same ragged_rank.
  ragged_rank = max(max(rt.ragged_rank for rt in rt_inputs), 1)
  rt_inputs = [_increase_ragged_rank_to(rt, ragged_rank, row_splits_dtype)
               for rt in rt_inputs]

  if axis == 0:
    return _ragged_stack_concat_axis_0(rt_inputs, stack_values)
  elif axis == 1:
    return _ragged_stack_concat_axis_1(rt_inputs, stack_values)
  else:  # axis > 1: recurse.
    values = [rt.values for rt in rt_inputs]
    splits = [[rt_input.row_splits] for rt_input in rt_inputs]
    with ops.control_dependencies(ragged_util.assert_splits_match(splits)):
      return ragged_tensor.RaggedTensor.from_row_splits(
          _ragged_stack_concat_helper(values, axis - 1, stack_values),
          splits[0][0], validate=False)


def _ragged_stack_concat_axis_0(rt_inputs, stack_values):
  # [FUNCTION DOC]
  # HELPER: axis-0 case of the ragged stack/concat engine.
  # Inputs: rt_inputs (ragged tensors of equal rank and ragged rank) and stack_values (true
  # to stack, false to concatenate). Concatenates the flat values of all inputs, and
  # concatenates the row splits of each ragged dimension with offsets adjusted. When
  # stacking, prepends one more splits level derived from the row counts of the inputs.
  # Returns the resulting ragged tensor, built without validation.
  """Helper function to concatenate or stack ragged tensors along axis 0.

  Args:
    rt_inputs: A list of RaggedTensors, all with the same rank and ragged_rank.
    stack_values: Boolean.  If true, then stack values; otherwise, concatenate
      them.

  Returns:
    A RaggedTensor.
  """
  # Concatenate the inner values together.
  flat_values = [rt.flat_values for rt in rt_inputs]
  concatenated_flat_values = array_ops.concat(flat_values, axis=0)

  # Concatenate the splits together for each ragged dimension (adjusting
  # split offsets as necessary).
  nested_splits = [rt.nested_row_splits for rt in rt_inputs]
  ragged_rank = rt_inputs[0].ragged_rank
  concatenated_nested_splits = [
      _concat_ragged_splits([ns[dim]
                             for ns in nested_splits])
      for dim in range(ragged_rank)
  ]

  # If we are performing a stack operation, then add another splits.
  if stack_values:
    stack_lengths = array_ops_stack.stack([rt.nrows() for rt in rt_inputs])
    stack_splits = ragged_util.lengths_to_splits(stack_lengths)
    concatenated_nested_splits.insert(0, stack_splits)

  return ragged_tensor.RaggedTensor.from_nested_row_splits(
      concatenated_flat_values, concatenated_nested_splits, validate=False)


def _ragged_stack_concat_axis_1(rt_inputs, stack_values):
  # [FUNCTION DOC]
  # HELPER: axis-1 case of the ragged stack/concat engine.
  # Inputs: rt_inputs (ragged tensors of equal rank and ragged rank) and stack_values (true
  # to stack, false to concatenate). Asserts that every input has the same number of rows
  # as the first (a runtime invalid-argument error with a descriptive message otherwise),
  # concatenates the inputs along axis 0, then gathers rows in an interleaved order so that
  # row i of every input becomes adjacent. When stacking, new splits group one row per
  # input; when concatenating, interior splits are dropped to merge adjacent rows. Static
  # row shapes are copied onto the new splits. Returns the resulting ragged tensor.
  """Helper function to concatenate or stack ragged tensors along axis 1.

  Args:
    rt_inputs: A list of RaggedTensors, all with the same rank and ragged_rank.
    stack_values: Boolean.  If true, then stack values; otherwise, concatenate
      them.

  Returns:
    A RaggedTensor.
  """
  num_inputs = len(rt_inputs)

  nrows_checks = []
  rt_nrows = rt_inputs[0].nrows()
  for index, rt in enumerate(rt_inputs[1:]):
    nrows_checks.append(
        check_ops.assert_equal(
            rt_nrows,
            rt.nrows(),
            message=(
                f'Input tensors at index 0 (=x) and {index+1} (=y) have'
                ' incompatible shapes.'
            ),
        )
    )

  with ops.control_dependencies(nrows_checks):
    # Concatenate the inputs together to put them in a single ragged tensor.
    concatenated_rt = _ragged_stack_concat_axis_0(rt_inputs, stack_values=False)

    # Use ragged.gather to permute the rows of concatenated_rt.  In particular,
    #   permuted_rt = [rt_inputs[0][0], ..., rt_inputs[N][0],
    #                  rt_inputs[0][1], ..., rt_inputs[N][1],
    #                      ...,
    #                  rt_inputs[0][M], ..., rt_input[N][M]]
    # where `N=num_inputs-1` and `M=rt_nrows-1`.
    row_indices = math_ops.range(rt_nrows * num_inputs)
    row_index_matrix = array_ops.reshape(row_indices, [num_inputs, -1])
    transposed_row_index_matrix = array_ops.transpose(row_index_matrix)
    row_permutation = array_ops.reshape(transposed_row_index_matrix, [-1])
    permuted_rt = ragged_gather_ops.gather(concatenated_rt, row_permutation)

    if stack_values:
      # Add a new splits tensor to group together the values.
      stack_splits = math_ops.range(0, rt_nrows * num_inputs + 1, num_inputs)
      _copy_row_shape(rt_inputs, stack_splits)
      return ragged_tensor.RaggedTensor.from_row_splits(
          permuted_rt, stack_splits, validate=False)
    else:
      # Merge together adjacent rows by dropping the row-split indices that
      # separate them.
      concat_splits = permuted_rt.row_splits[::num_inputs]
      _copy_row_shape(rt_inputs, concat_splits)
      return ragged_tensor.RaggedTensor.from_row_splits(
          permuted_rt.values, concat_splits, validate=False)


def _copy_row_shape(rt_inputs, splits):
  # [FUNCTION DOC]
  # HELPER: static-shape propagation for the axis-1 stack/concat routine.
  # KNOWN ISSUE: the shape is set once per input inside a loop, using the row count of each
  # input plus one; but for stacking, the splits length is rows times inputs plus one, so
  # the set shape appears to describe the wrong length (it only works when the static
  # values are consistent or merged by the shape check).
  # Inputs: rt_inputs (list of ragged tensors) and splits (a row-splits tensor, mutated).
  # For each input with a statically known number of rows, sets the static shape of splits
  # to a one-dimensional shape with that row count plus one. Returns nothing.
  """Sets splits.shape to [rt[shape[0]+1] for each rt in rt_inputs."""
  for rt in rt_inputs:
    if rt.shape[0] is not None:
      splits.set_shape(tensor_shape.TensorShape(rt.shape[0] + 1))


def _increase_ragged_rank_to(rt_input, ragged_rank, row_splits_dtype):
  # [FUNCTION DOC]
  # HELPER: rank unification for the ragged stack/concat engine.
  # Inputs: rt_input (dense or ragged tensor), ragged_rank (target ragged rank, integer) and
  # row_splits_dtype (dtype for any newly created row splits). If the target is positive,
  # converts a dense input to a ragged one, and recursively converts its values until the
  # ragged rank reaches the target. Returns the (possibly new) ragged tensor; an already
  # sufficient input is returned unchanged.
  """Adds ragged dimensions to `rt_input` so it has the desired ragged rank."""
  if ragged_rank > 0:
    if not ragged_tensor.is_ragged(rt_input):
      rt_input = ragged_tensor.RaggedTensor.from_tensor(
          rt_input, row_splits_dtype=row_splits_dtype)
    if rt_input.ragged_rank < ragged_rank:
      rt_input = rt_input.with_values(
          _increase_ragged_rank_to(rt_input.values, ragged_rank - 1,
                                   row_splits_dtype))
  return rt_input


def _concat_ragged_splits(splits_list):
  # [FUNCTION DOC]
  # HELPER: joins row-splits vectors when concatenating ragged tensors along axis 0.
  # Input: splits_list (non-empty list of row-splits tensors, each starting at zero). Keeps
  # the first fully; for each later one drops its leading zero and shifts it by the running
  # total of the previous final values. Returns one concatenated row-splits tensor. No side
  # effects; an empty list would fail with an index error.
  """Concatenates a list of RaggedTensor splits to form a single splits."""
  pieces = [splits_list[0]]
  splits_offset = splits_list[0][-1]
  for splits in splits_list[1:]:
    pieces.append(splits[1:] + splits_offset)
    splits_offset += splits[-1]
  return array_ops.concat(pieces, axis=0)
