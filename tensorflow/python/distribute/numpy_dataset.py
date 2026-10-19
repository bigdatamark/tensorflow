# [MODULE DOC]
# HELPER: support code for building a dataset from NumPy arrays under distribution strategies.
# Provides a function that fills a variable from a NumPy array (in bounded-size slices in graph
# mode), a function that creates a dataset on a given device from (nested) NumPy arrays by first
# copying them into non-trainable variables, and a tiny marker class naming a single device so
# variables are placed there and not mirrored. No import-time side effects.

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
"""Code for creating a dataset out of a NumPy array."""

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.util import nest


def init_var_from_numpy(input_var, numpy_input, session):
  # [FUNCTION DOC]
  # HELPER: loads a NumPy array into a variable. Inputs: the variable, the array, and a session
  # (only needed, and asserted non-null, in graph mode). Runs outside any function-building scope.
  # In eager mode simply assigns. In graph mode runs the variable's initializer, then copies the
  # array in slices along the first axis: each slice holds as many rows as fit in about 64 MB (at
  # least one row), fed through placeholders into a slice-assign operation, so no single transfer is
  # too large. Returns nothing. Mutates the variable.
  """Initialize `input_var` to `numpy_input` using `session` in graph mode."""
  with ops.init_scope():
    if context.executing_eagerly():
      input_var.assign(numpy_input)
      return

    assert session is not None
    session.run(input_var.initializer)

    start_placeholder = array_ops.placeholder(dtypes.int64, ())
    end_placeholder = array_ops.placeholder(dtypes.int64, ())
    slice_placeholder = array_ops.placeholder(input_var.dtype)
    assign_slice_op = input_var[start_placeholder:end_placeholder].assign(
        slice_placeholder)

    # If each batch element is > 64 MB, then we copy each batch element
    # individually. Otherwise, the slices will be < 128 MB. There might be
    # padding which might mean that the slices are 128 MB even if the size of
    # the tensor allocated is less than 128 MB.  This formula gives slices with
    # size: ceil(64 MB / byte size per batch element) bytes.  Using ceil()
    # guarantees we get a number >= 1.

    # Calculate the size of each batch element.
    byte_size_per_batch_element = (
        np.prod(numpy_input.shape[1:]) * input_var.dtype.size)

    # Calculate number of elements we want to copy per slice.
    batch_size_per_slice = int(
        np.ceil((64 << 20) / byte_size_per_batch_element))

    # Copy slices of the above size starting at 0, except the last slice will be
    # smaller.
    start = 0
    limit = numpy_input.shape[0]
    while start < limit:
      end = min(start + batch_size_per_slice, limit)
      session.run(assign_slice_op, feed_dict={
          start_placeholder: start,
          end_placeholder: end,
          slice_placeholder: numpy_input[start:end]})
      start = end


def one_host_numpy_dataset(numpy_input, colocate_with, session):
  # [FUNCTION DOC]
  # HELPER: creates a dataset from NumPy data placed on one device. Inputs: a NumPy array or nested
  # structure of arrays, the object or device to colocate the variables with, and a session (used in
  # graph mode). Creates a non-trainable zero-filled variable per array with a creator scope forcing
  # the colocation, fills each from its array, repacks to the input structure and returns a dataset
  # of slices along the first axis.
  """Create a dataset on `colocate_with` from `numpy_input`."""

  def create_colocated_variable(next_creator, **kwargs):
    # [FUNCTION DOC]
    # HELPER: variable creator hook. Sets the colocation argument to the enclosing device or object
    # and delegates to the next creator, returning the created variable.
    kwargs["colocate_with"] = colocate_with
    return next_creator(**kwargs)

  numpy_flat = nest.flatten(numpy_input)
  with variable_scope.variable_creator_scope(create_colocated_variable):
    vars_flat = tuple(variable_v1.VariableV1(array_ops.zeros(i.shape, i.dtype),
                                             trainable=False)
                      for i in numpy_flat)
  for v, i in zip(vars_flat, numpy_flat):
    init_var_from_numpy(v, i, session)
  vars_nested = nest.pack_sequence_as(numpy_input, vars_flat)
  return dataset_ops.Dataset.from_tensor_slices(vars_nested)


class SingleDevice(object):
  # [CLASS DOC]
  # HELPER: marker object holding one device name. Passed as the colocation target so that
  # variables are created on that device and not mirrored. Attribute: the device name.
  """Used with `colocate_with` to create a non-mirrored variable."""

  def __init__(self, device):
    # [FUNCTION DOC]
    # Constructor. Input: a device name string. Stores it as the device attribute.
    self.device = device
