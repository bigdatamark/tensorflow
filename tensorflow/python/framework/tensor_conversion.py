# [MODULE DOC]
# Public entry points for converting arbitrary values to tensors. Exposes the version 1 and
# version 2 flavours of the public convert-to-tensor API (exported to the public namespace under
# the same name, one for each API generation, each wrapped with dispatch support), plus two
# undecorated internal implementations. All of them ultimately delegate to the conversion
# registry module, which runs registered per-type conversion functions. Import-time side effects
# are limited to the public-API export registration done by the decorators.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tensor conversion functions."""
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_export


def convert_to_tensor_v1(
    value, dtype=None, name=None, preferred_dtype=None, dtype_hint=None
) -> tensor_lib.Tensor:
  # [FUNCTION DOC]
  # HELPER: undecorated implementation behind the version 1 public convert-to-tensor function.
  # Converts value to a tensor. Inputs: value (any object with a registered conversion), dtype
  # (optional target element type), name (optional op name), preferred_dtype and dtype_hint
  # (optional soft dtype preferences; the deprecated preferred_dtype name is merged with
  # dtype_hint via the deprecation lookup, which prefers dtype_hint and emits a deprecation
  # warning when the old name is used; it raises a value error if both are given and differ).
  # Returns the converted tensor by delegating to the version 2 implementation. Errors from
  # conversion propagate. No other side effects.
  """Converts the given `value` to a `Tensor` (with the TF1 API)."""
  preferred_dtype = deprecation.deprecated_argument_lookup(
      "dtype_hint", dtype_hint, "preferred_dtype", preferred_dtype
  )
  return convert_to_tensor_v2(value, dtype, preferred_dtype, name)


@tf_export.tf_export(v1=["convert_to_tensor"])
@dispatch.add_dispatch_support
def convert_to_tensor_v1_with_dispatch(
    value, dtype=None, name=None, preferred_dtype=None, dtype_hint=None
) -> tensor_lib.Tensor:
  # [FUNCTION DOC]
  # Public version 1 convert-to-tensor function, exported to the public API and given dispatch
  # support so that types can override it. Inputs: value (object to convert), dtype (optional
  # element type), name (optional name for a newly created tensor), preferred_dtype (deprecated
  # soft dtype preference) and dtype_hint (same meaning, takes precedence). Returns the tensor.
  # Raises a type error when no conversion is registered, a runtime error when a conversion
  # function returns something invalid, and a value error on dtype mismatch for an existing
  # tensor in graph mode. Simply forwards all arguments to the internal version 1 function.
  """Converts the given `value` to a `Tensor`.

  This function converts Python objects of various types to `Tensor`
  objects. It accepts `Tensor` objects, numpy arrays, Python lists,
  and Python scalars. For example:

  ```python
  import numpy as np

  def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg

  # The following calls are equivalent.
  value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
  value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
  value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
  ```

  This function can be useful when composing a new operation in Python
  (such as `my_func` in the example above). All standard Python op
  constructors apply this function to each of their Tensor-valued
  inputs, which allows those ops to accept numpy arrays, Python lists,
  and scalars in addition to `Tensor` objects.

  Note: This function diverges from default Numpy behavior for `float` and
    `string` types when `None` is present in a Python list or scalar. Rather
    than silently converting `None` values, an error will be thrown.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    preferred_dtype: Optional element type for the returned tensor, used when
      dtype is None. In some cases, a caller may not have a dtype in mind when
      converting to a tensor, so preferred_dtype can be used as a soft
      preference.  If the conversion to `preferred_dtype` is not possible, this
      argument has no effect.
    dtype_hint: same meaning as preferred_dtype, and overrides it.

  Returns:
    A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.
  """
  return convert_to_tensor_v1(
      value,
      dtype=dtype,
      name=name,
      preferred_dtype=preferred_dtype,
      dtype_hint=dtype_hint,
  )


@tf_export.tf_export("convert_to_tensor", v1=[])
@dispatch.add_dispatch_support
def convert_to_tensor_v2_with_dispatch(
    value, dtype=None, dtype_hint=None, name=None
) -> tensor_lib.Tensor:
  # [FUNCTION DOC]
  # Public version 2 convert-to-tensor function, exported to the public API (no version 1
  # export) with dispatch support. Inputs: value (object to convert), dtype (optional element
  # type), dtype_hint (optional soft preference used only when dtype is absent), name (optional
  # name for a new tensor). Returns the tensor. Raises a type error if no conversion function is
  # registered, a runtime error if a conversion function returns an invalid value, and a value
  # error for an existing tensor of a different dtype in graph mode. Forwards to the internal
  # version 2 function; no other side effects.
  """Converts the given `value` to a `Tensor`.

  This function converts Python objects of various types to `Tensor`
  objects. It accepts `Tensor` objects, numpy arrays, Python lists,
  and Python scalars.

  For example:

  >>> import numpy as np
  >>> def my_func(arg):
  ...   arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  ...   return arg

  >>> # The following calls are equivalent.
  ...
  >>> value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
  >>> print(value_1)
  tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float32)
  >>> value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
  >>> print(value_2)
  tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float32)
  >>> value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
  >>> print(value_3)
  tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float32)

  This function can be useful when composing a new operation in Python
  (such as `my_func` in the example above). All standard Python op
  constructors apply this function to each of their Tensor-valued
  inputs, which allows those ops to accept numpy arrays, Python lists,
  and scalars in addition to `Tensor` objects.

  Note: This function diverges from default Numpy behavior for `float` and
    `string` types when `None` is present in a Python list or scalar. Rather
    than silently converting `None` values, an error will be thrown.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    dtype_hint: Optional element type for the returned tensor, used when dtype
      is None. In some cases, a caller may not have a dtype in mind when
      converting to a tensor, so dtype_hint can be used as a soft preference. If
      the conversion to `dtype_hint` is not possible, this argument has no
      effect.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.
  """
  return convert_to_tensor_v2(
      value, dtype=dtype, dtype_hint=dtype_hint, name=name
  )


def convert_to_tensor_v2(
    value, dtype=None, dtype_hint=None, name=None
) -> tensor_lib.Tensor:
  # [FUNCTION DOC]
  # HELPER: undecorated core implementation shared by the public version 1 and version 2
  # convert-to-tensor functions.
  # Inputs: value (object to convert), dtype (optional element type), dtype_hint (optional soft
  # preferred type), name (optional name). Returns the tensor produced by the conversion
  # registry, passing dtype_hint as its preferred dtype. Exceptions from the registry (type,
  # runtime, value errors) propagate. No side effects of its own.
  """Converts the given `value` to a `Tensor`."""
  # preferred_dtype = preferred_dtype or dtype_hint
  return tensor_conversion_registry.convert(
      value, dtype, name, preferred_dtype=dtype_hint
  )
