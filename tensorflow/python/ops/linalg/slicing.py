# [MODULE DOC]
# HELPER: helper module implementing batch slicing for LinearOperator classes.
# Exposes batch_slice, which builds a new operator of the same type whose parameters are sliced
# along the batch dimensions (the operator's item access delegates here). Private helpers
# broadcast parameters to full batch rank, sanitize slices so they do not overflow broadcast
# (size one) dimensions, and slice one parameter (recursing into nested operators). No import-time
# side effects.
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
"""Utilities for slicing in to a `LinearOperator`."""

import collections
import functools
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest


__all__ = ['batch_slice']


def _prefer_static_where(condition, x, y):
  # [FUNCTION DOC]
  # HELPER: element selection that stays static when possible.
  # If condition, x and y all have statically known values, returns the numpy selection of x where
  # condition holds else y; otherwise returns the graph selection op result. No side effects.
  args = [condition, x, y]
  constant_args = [tensor_util.constant_value(a) for a in args]
  # Do this statically.
  if all(arg is not None for arg in constant_args):
    condition_, x_, y_ = constant_args
    return np.where(condition_, x_, y_)
  return array_ops.where(condition, x, y)


def _broadcast_parameter_with_batch_shape(
    param, param_ndims_to_matrix_ndims, batch_shape):
  # [FUNCTION DOC]
  # HELPER: gives parameters the full batch rank before slicing.
  # If param is an operator (has a batch shape method), recursively broadcasts each of its own
  # parameters (as listed by its parameter-ndims mapping) and rebuilds an operator of the same
  # type from its parameters. Otherwise param is a tensor: broadcasts it against a shape made of
  # the batch shape followed by ones for the matrix dimensions (param_ndims_to_matrix_ndims says
  # how many trailing dims belong to the matrix part). Returns the broadcast tensor or new operator.
  """Broadcasts `param` with the given batch shape, recursively."""
  if hasattr(param, 'batch_shape_tensor'):
    # Recursively broadcast every parameter inside the operator.
    override_dict = {}
    for name, ndims in param._experimental_parameter_ndims_to_matrix_ndims.items():  # pylint:disable=protected-access,line-too-long
      sub_param = getattr(param, name)
      override_dict[name] = nest.map_structure_up_to(
          sub_param, functools.partial(
              _broadcast_parameter_with_batch_shape,
              batch_shape=batch_shape), sub_param, ndims)
    parameters = dict(param.parameters, **override_dict)
    return type(param)(**parameters)

  base_shape = array_ops.concat(
      [batch_shape, array_ops.ones(
          [param_ndims_to_matrix_ndims], dtype=dtypes.int32)], axis=0)
  return array_ops.broadcast_to(
      param,
      array_ops.broadcast_dynamic_shape(base_shape, array_ops.shape(param)))


def _sanitize_slices(slices, intended_shape, deficient_shape):
  # [FUNCTION DOC]
  # HELPER: adapts user slices to a parameter whose batch dims may be size one (broadcast).
  # Walks the slices: the ellipsis switches to negative indexing (a second ellipsis raises a value
  # error), new-axis entries are passed through, and for other entries it checks whether the
  # intended size exceeds the actual size at that position (a broadcast dimension); if so, indices
  # become zero and slice stop and step become one, with start zero, so the slice does not overflow.
  # Returns the list of sanitized slice objects. intended_shape and deficient_shape are integer
  # shape tensors of equal rank.
  """Restricts slices to avoid overflowing size-1 (broadcast) dimensions.

  Args:
    slices: iterable of slices received by `__getitem__`.
    intended_shape: int `Tensor` shape for which the slices were intended.
    deficient_shape: int `Tensor` shape to which the slices will be applied.
      Must have the same rank as `intended_shape`.
  Returns:
    sanitized_slices: Python `list` of slice objects.
  """
  sanitized_slices = []
  idx = 0
  for slc in slices:
    if slc is Ellipsis:  # Switch over to negative indexing.
      if idx < 0:
        raise ValueError('Found multiple `...` in slices {}'.format(slices))
      num_remaining_non_newaxis_slices = sum(
          s is not array_ops.newaxis for s in slices[
              slices.index(Ellipsis) + 1:])
      idx = -num_remaining_non_newaxis_slices
    elif slc is array_ops.newaxis:
      pass
    else:
      is_broadcast = intended_shape[idx] > deficient_shape[idx]
      if isinstance(slc, slice):
        # Slices are denoted by start:stop:step.
        start, stop, step = slc.start, slc.stop, slc.step
        if start is not None:
          start = _prefer_static_where(is_broadcast, 0, start)
        if stop is not None:
          stop = _prefer_static_where(is_broadcast, 1, stop)
        if step is not None:
          step = _prefer_static_where(is_broadcast, 1, step)
        slc = slice(start, stop, step)
      else:  # int, or int Tensor, e.g. d[d.batch_shape_tensor()[0] // 2]
        slc = _prefer_static_where(is_broadcast, 0, slc)
      idx += 1
    sanitized_slices.append(slc)
  return sanitized_slices


def _slice_single_param(
    param, param_ndims_to_matrix_ndims, slices, batch_shape):
  # [FUNCTION DOC]
  # HELPER: slices the batch dimensions of one operator parameter.
  # Broadcasts param to full batch rank, determines its batch shape (trimming the matrix dims),
  # and returns it unchanged if it is an atomic object with no batch rank while the operator has
  # one. Otherwise sanitizes the slices, appends an ellipsis and full slices for the trailing
  # matrix dimensions, and returns the parameter indexed with the final slices. Inputs: the
  # parameter (tensor or operator), the number of trailing matrix dims, the slices, and the
  # operator batch shape tensor.
  """Slices into the batch shape of a single parameter.

  Args:
    param: The original parameter to slice; either a `Tensor` or an object
      with batch shape (LinearOperator).
    param_ndims_to_matrix_ndims: `int` number of right-most dimensions used for
      inferring matrix shape of the `LinearOperator`. For non-Tensor
      parameters, this is the number of this param's batch dimensions used by
      the matrix shape of the parent object.
    slices: iterable of slices received by `__getitem__`.
    batch_shape: The parameterized object's batch shape `Tensor`.

  Returns:
    new_param: Instance of the same type as `param`, batch-sliced according to
      `slices`.
  """
  # Broadcast the parammeter to have full batch rank.
  param = _broadcast_parameter_with_batch_shape(
      param, param_ndims_to_matrix_ndims, array_ops.ones_like(batch_shape))

  if hasattr(param, 'batch_shape_tensor'):
    param_batch_shape = param.batch_shape_tensor()
  else:
    param_batch_shape = array_ops.shape(param)
  # Truncate by param_ndims_to_matrix_ndims
  param_batch_rank = array_ops.size(param_batch_shape)
  param_batch_shape = param_batch_shape[
      :(param_batch_rank - param_ndims_to_matrix_ndims)]

  # At this point the param should have full batch rank, *unless* it's an
  # atomic object like `tfb.Identity()` incapable of having any batch rank.
  if (tensor_util.constant_value(array_ops.size(batch_shape)) != 0 and
      tensor_util.constant_value(array_ops.size(param_batch_shape)) == 0):
    return param
  param_slices = _sanitize_slices(
      slices, intended_shape=batch_shape, deficient_shape=param_batch_shape)

  # Extend `param_slices` (which represents slicing into the
  # parameter's batch shape) with the parameter's event ndims. For example, if
  # `params_ndims == 1`, then `[i, ..., j]` would become `[i, ..., j, :]`.
  if param_ndims_to_matrix_ndims > 0:
    if Ellipsis not in [
        slc for slc in slices if not tensor_util.is_tensor(slc)]:
      param_slices.append(Ellipsis)
    param_slices += [slice(None)] * param_ndims_to_matrix_ndims
  return param.__getitem__(tuple(param_slices))


def batch_slice(linop, params_overrides, slices):
  # [FUNCTION DOC]
  # Slices a LinearOperator along its batch dimensions. Wraps a single non-sequence slice into a
  # tuple. If the slices are just an ellipsis no parameters are sliced; otherwise each non-null
  # parameter listed in the operator's parameter-ndims mapping is sliced with the single-parameter
  # helper. Entries in params_overrides (mapping from parameter names to values) then replace
  # those results. Returns a new operator of the same type built from the original parameters
  # updated by these overrides. Does not modify the input operator.
  """Slices `linop` along its batch dimensions.

  Args:
    linop: A `LinearOperator` instance.
    params_overrides: A `dict` of parameter overrides.
    slices: A `slice` or `int` or `int` `Tensor` or `tf.newaxis` or `tuple`
      thereof. (e.g. the argument of a `__getitem__` method).

  Returns:
    new_linop: A batch-sliced `LinearOperator`.
  """
  if not isinstance(slices, collections.abc.Sequence):
    slices = (slices,)
  if len(slices) == 1 and slices[0] is Ellipsis:
    override_dict = {}
  else:
    batch_shape = linop.batch_shape_tensor()
    override_dict = {}
    for param_name, param_ndims_to_matrix_ndims in linop._experimental_parameter_ndims_to_matrix_ndims.items():  # pylint:disable=protected-access,line-too-long
      param = getattr(linop, param_name)
      # These represent optional `Tensor` parameters.
      if param is not None:
        override_dict[param_name] = nest.map_structure_up_to(
            param, functools.partial(
                _slice_single_param, slices=slices, batch_shape=batch_shape),
            param, param_ndims_to_matrix_ndims)
  override_dict.update(params_overrides)
  parameters = dict(linop.parameters, **override_dict)
  return type(linop)(**parameters)
