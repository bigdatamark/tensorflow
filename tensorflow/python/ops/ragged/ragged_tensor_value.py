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
# [MODULE DOC]
# Defines RaggedTensorValue, the plain numpy-based container holding the concrete value of a
# ragged tensor (a values array plus a row_splits array) when evaluated in graph mode.
# Exported to the public API only under the v1 namespace for ragged tensor values. Depends on
# numpy, RowPartition (to build row partitions from splits) and the dispatch machinery, where the
# class is registered as a dispatchable type at import time. Has no other import side effects.
"""Value for RaggedTensor."""

import numpy as np

from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["ragged.RaggedTensorValue"])
@dispatch.register_dispatchable_type
class RaggedTensorValue:
  # [CLASS DOC]
  # Immutable-by-convention holder for the evaluated value of a ragged tensor, stored as a
  # values object (numpy array or another RaggedTensorValue for nested raggedness) and a 1-D
  # numpy row_splits array. Offers read-only properties (row_splits, values, dtype, flat_values,
  # nested_row_splits, ragged_rank, shape) and conversion to a nested list. Created by graph-mode
  # session runs or directly by callers. Decorated for v1 API export and registered as a
  # dispatchable type, which are import-time side effects.
  """Represents the value of a `RaggedTensor`.

  Warning: `RaggedTensorValue` should only be used in graph mode; in
  eager mode, the `tf.RaggedTensor` class contains its value directly.

  See `tf.RaggedTensor` for a description of ragged tensors.
  """

  def __init__(self, values, row_splits):
    # [FUNCTION DOC]
    # Constructor. Inputs: values, a numpy array or numpy scalar of any dtype/shape, or a
    # RaggedTensorValue; row_splits, a 1-D numpy array of int32 or int64 dtype. Stores both as
    # attributes and returns nothing. Raises a type error if row_splits is not a 1-D int32/int64
    # numpy array or if values is neither numpy data nor a RaggedTensorValue; raises a value error
    # if values is a RaggedTensorValue whose row_splits dtype differs from row_splits.
    # Does not check that the splits are consistent with the number of values.
    """Creates a `RaggedTensorValue`.

    Args:
      values: A numpy array of any type and shape; or a RaggedTensorValue.
      row_splits: A 1-D int32 or int64 numpy array.
    """
    if not (isinstance(row_splits, (np.ndarray, np.generic)) and
            row_splits.dtype in (np.int64, np.int32) and row_splits.ndim == 1):
      raise TypeError("row_splits must be a 1D int32 or int64 numpy array")
    if not isinstance(values, (np.ndarray, np.generic, RaggedTensorValue)):
      raise TypeError("values must be a numpy array or a RaggedTensorValue")
    if (isinstance(values, RaggedTensorValue) and
        row_splits.dtype != values.row_splits.dtype):
      raise ValueError("row_splits and values.row_splits must have "
                       "the same dtype")
    self._values = values
    self._row_splits = row_splits

  row_splits = property(
      lambda self: self._row_splits,
      doc="""The split indices for the ragged tensor value.""")
  values = property(
      lambda self: self._values,
      doc="""The concatenated values for all rows in this tensor.""")
  dtype = property(
      lambda self: self._values.dtype,
      doc="""The numpy dtype of values in this tensor.""")

  @property
  def flat_values(self):
    # [FUNCTION DOC]
    # Property. Repeatedly unwraps nested RaggedTensorValue values until reaching the innermost
    # numpy array, and returns that array. No arguments, no side effects, no exceptions.
    """The innermost `values` array for this ragged tensor value."""
    rt_values = self.values
    while isinstance(rt_values, RaggedTensorValue):
      rt_values = rt_values.values
    return rt_values

  @property
  def nested_row_splits(self):
    # [FUNCTION DOC]
    # Property. Walks down the chain of nested values and collects the row_splits array of every
    # ragged level, outermost first. Returns a tuple of numpy arrays whose length equals the
    # ragged rank. No side effects, no exceptions.
    """The row_splits for all ragged dimensions in this ragged tensor value."""
    rt_nested_splits = [self.row_splits]
    rt_values = self.values
    while isinstance(rt_values, RaggedTensorValue):
      rt_nested_splits.append(rt_values.row_splits)
      rt_values = rt_values.values
    return tuple(rt_nested_splits)

  @property
  def ragged_rank(self):
    # [FUNCTION DOC]
    # Property. Returns the number of ragged dimensions as an integer: one if values is a plain
    # array, otherwise one plus the ragged rank of the nested value. Recursive, no side effects.
    """The number of ragged dimensions in this ragged tensor value."""
    values_is_ragged = isinstance(self._values, RaggedTensorValue)
    return self._values.ragged_rank + 1 if values_is_ragged else 1

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property. Returns a tuple: number of rows (length of row_splits minus one), then an
    # unknown (null) entry for the ragged dimension, then the trailing dimensions of the values
    # shape after its first dimension. No side effects.
    # KNOWN ISSUE: for ragged rank above one the nested values' shape already contains the null
    # entry for its own ragged dimension, so only the outermost ragged dimension is explicit.
    """A tuple indicating the shape of this RaggedTensorValue."""
    return (self._row_splits.shape[0] - 1,) + (None,) + self._values.shape[1:]

  @property
  def _nested_row_partitions(self):
    # [FUNCTION DOC]
    # HELPER: internal accessor used by code that needs row partitions rather than raw splits.
    # Property. Builds one RowPartition from each nested row_splits array (outermost first) and
    # returns them as a list. No side effects.
    """The row_partitions representing this shape."""
    return [RowPartition.from_row_splits(rs) for rs in self.nested_row_splits]

  def __str__(self):
    # [FUNCTION DOC]
    # Returns a human-readable string of the form tensor-value-marker wrapping the nested list
    # produced by converting this value to a list. No side effects.
    return "<tf.RaggedTensorValue %s>" % self.to_list()

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a constructor-style string showing the repr of the stored values and row_splits.
    # No side effects.
    return "tf.RaggedTensorValue(values=%r, row_splits=%r)" % (self._values,
                                                               self._row_splits)

  def to_list(self):
    # [FUNCTION DOC]
    # Converts the value to a nested Python list: recursively converts nested values (or calls
    # the numpy conversion on a plain array), then slices it per row using consecutive row_splits
    # entries. Returns a list with one sublist per row. No side effects.
    """Returns this ragged tensor value as a nested Python list."""
    if isinstance(self._values, RaggedTensorValue):
      values_as_list = self._values.to_list()
    else:
      values_as_list = self._values.tolist()
    return [
        values_as_list[self._row_splits[i]:self._row_splits[i + 1]]
        for i in range(len(self._row_splits) - 1)
    ]
