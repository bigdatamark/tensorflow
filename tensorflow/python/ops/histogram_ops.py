# [MODULE DOC]
# Public histogram operations: histogram_fixed_width_bins (maps each value to the index of its
# equal-width bin) and histogram_fixed_width (counts values per bin). Both are exported to the
# public API with dispatch support. Importing has no side effects beyond that registration.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-short-docstring-punctuation
"""Histograms.
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export('histogram_fixed_width_bins')
@dispatch.add_dispatch_support
def histogram_fixed_width_bins(values,
                               value_range,
                               nbins=100,
                               dtype=dtypes.int32,
                               name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the dtype parameter is documented as the dtype of the result but is never used;
  # the returned indices are always cast to 32-bit integers. Also the docstring says the default
  # scope name is 'histogram_fixed_width' while the code uses 'histogram_fixed_width_bins'.
  # Computes, for every element of values, the index of the equal-width bin it falls in. Inputs:
  # values (numeric tensor of any shape), value_range (two-element tensor with low and high edge),
  # nbins (scalar, default 100, converted to int32), dtype (ignored), name (optional scope name).
  # Values are scaled into the unit interval using the range, multiplied by nbins and floored;
  # results are clipped to [0, nbins - 1] so outliers go to the first or last bin. An assertion
  # on nbins > 0 is attached as a control dependency, so a non-positive nbins fails at run time
  # with an invalid-argument error. Returns an int32 tensor with the same shape as values.
  # No state is mutated.
  """Bins the given values for use in a histogram.

  Given the tensor `values`, this operation returns a rank 1 `Tensor`
  representing the indices of a histogram into which each element
  of `values` would be binned. The bins are equal width and
  determined by the arguments `value_range` and `nbins`.

  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram_fixed_width').

  Returns:
    A `Tensor` holding the indices of the binned values whose shape matches
    `values`.

  Raises:
    TypeError: If any unsupported dtype is provided.
    tf.errors.InvalidArgumentError: If value_range does not
        satisfy value_range[0] < value_range[1].

  Examples:

  >>> # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
  ...
  >>> nbins = 5
  >>> value_range = [0.0, 5.0]
  >>> new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
  >>> indices = tf.histogram_fixed_width_bins(new_values, value_range, nbins=5)
  >>> indices.numpy()
  array([0, 0, 1, 2, 4, 4], dtype=int32)
  """
  with ops.name_scope(name, 'histogram_fixed_width_bins',
                      [values, value_range, nbins]):
    values = ops.convert_to_tensor(values, name='values')
    shape = array_ops.shape(values)

    values = array_ops.reshape(values, [-1])
    value_range = ops.convert_to_tensor(value_range, name='value_range')
    nbins = ops.convert_to_tensor(nbins, dtype=dtypes.int32, name='nbins')
    check = control_flow_assert.Assert(
        math_ops.greater(nbins, 0), ['nbins %s must > 0' % nbins])
    nbins = control_flow_ops.with_dependencies([check], nbins)
    nbins_float = math_ops.cast(nbins, values.dtype)

    # Map tensor values that fall within value_range to [0, 1].
    scaled_values = math_ops.truediv(
        values - value_range[0],
        value_range[1] - value_range[0],
        name='scaled_values')

    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
    indices = math_ops.floor(nbins_float * scaled_values, name='indices')

    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    indices = math_ops.cast(
        clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)
    return array_ops.reshape(indices, shape)


@tf_export('histogram_fixed_width')
@dispatch.add_dispatch_support
def histogram_fixed_width(values,
                          value_range,
                          nbins=100,
                          dtype=dtypes.int32,
                          name=None):
  # [FUNCTION DOC]
  # Counts how many entries of values fall into each of nbins equal-width bins spanning
  # value_range; values at or below the low edge count in the first bin and values at or above the
  # high edge count in the last. Inputs: values (numeric tensor), value_range (two-element tensor of
  # the same dtype), nbins (scalar, default 100), dtype (result dtype, default int32, passed to the
  # kernel), name (optional scope name). Delegates to the generated histogram kernel inside a name
  # scope. Returns a 1-D tensor of length nbins with the counts. Invalid ranges or dtypes surface
  # as errors from the kernel (invalid-argument or type errors). No state is mutated.
  """Return histogram of values.

  Given the tensor `values`, this operation returns a rank 1 histogram counting
  the number of entries in `values` that fell into every bin.  The bins are
  equal width and determined by the arguments `value_range` and `nbins`.

  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram_fixed_width').

  Returns:
    A 1-D `Tensor` holding histogram of values.

  Raises:
    TypeError: If any unsupported dtype is provided.
    tf.errors.InvalidArgumentError: If value_range does not
        satisfy value_range[0] < value_range[1].

  Examples:

  >>> # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
  ...
  >>> nbins = 5
  >>> value_range = [0.0, 5.0]
  >>> new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
  >>> hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
  >>> hist.numpy()
  array([2, 1, 1, 0, 2], dtype=int32)
  """
  with ops.name_scope(name, 'histogram_fixed_width',
                      [values, value_range, nbins]) as name:
    # pylint: disable=protected-access
    return gen_math_ops._histogram_fixed_width(
        values, value_range, nbins, dtype=dtype, name=name)
    # pylint: enable=protected-access
