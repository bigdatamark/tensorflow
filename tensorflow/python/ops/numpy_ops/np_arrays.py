# [MODULE DOC]
# Defines the ndarray type used by TF NumPy and a safer tensor-conversion wrapper.
# In this version the ndarray name is simply an alias of the regular tensor class, so TF NumPy
# functions return and accept ordinary tensors. The module also exposes convert_to_tensor, which
# works around a conversion bug for very large Python integers and picks the configured default
# float type for plain Python floats. Imports the dtype helper module np_dtypes. No import-time
# side effects other than creating the alias.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""ndarray class."""

# pylint: disable=g-direct-tensorflow-import

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops.numpy_ops import np_dtypes


def convert_to_tensor(value, dtype=None, dtype_hint=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the module docstring says "ndarray class" but no class is defined here; ndarray is just
  # KNOWN ISSUE: an alias of the tensor class defined at the bottom of the file.
  # Converts a value to a tensor, wrapping the generic tensor-conversion API with two special cases.
  # Inputs: value is anything convertible to a tensor; dtype is an optional explicit element type;
  # dtype_hint is an optional soft preference for the type. If no dtype is given and value is a Python
  # integer of at least 2 to the power 63, the dtype is forced to unsigned 64-bit integer. If neither
  # dtype nor dtype_hint is given and value is a Python float, the dtype becomes the default float type
  # from np_dtypes (float64 unless float32 is preferred or float64 is disallowed). Otherwise the
  # arguments pass through unchanged. Returns the resulting tensor. Errors from the underlying
  # conversion (for example incompatible values) propagate. No side effects.
  """Wrapper over `tf.convert_to_tensor`.

  Args:
    value: value to convert
    dtype: (optional) the type we would like it to be converted to.
    dtype_hint: (optional) soft preference for the type we would like it to be
      converted to. `tf.convert_to_tensor` will attempt to convert value to this
      type first, but will not fail if conversion is not possible falling back
      to inferring the type instead.

  Returns:
    Value converted to tf.Tensor.
  """
  # A safer version of `tf.convert_to_tensor` to work around b/149876037.
  # TODO(wangpeng): Remove this function once the bug is fixed.
  if (dtype is None and isinstance(value, int) and
      value >= 2**63):
    dtype = dtypes.uint64
  elif dtype is None and dtype_hint is None and isinstance(value, float):
    dtype = np_dtypes.default_float_type()
  return tensor_conversion.convert_to_tensor_v2_with_dispatch(
      value, dtype=dtype, dtype_hint=dtype_hint)


ndarray = tensor.Tensor
