# [MODULE DOC]
# HELPER: shared test utilities for the WeakTensor test files (weak_tensor_ops_test and others).
# Provides functions that build test inputs of several kinds (WeakTensor, Tensor, NumPy array,
# plain Python value) from one base value, a convenience constructor for weak tensors, and a
# context-manager class that temporarily switches the global dtype conversion (promotion) mode.
# No import-time side effects besides imports.
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
"""Utils for WeakTensor related tests."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.weak_tensor import WeakTensor


def get_test_input_for_op(val, dtype):
  # [FUNCTION DOC]
  # HELPER: builds the family of equivalent inputs used by WeakTensor op tests.
  # Takes a base value and a pair of (dtype, is_weak). For a weak dtype it returns a WeakTensor
  # made from the value; if the dtype is one of the three Python-default types (int32, float32,
  # complex128) it also returns the value multiplied by the matching Python scalar and a
  # Tensor built from that product with inferred dtype. For a non-weak dtype it returns a Tensor
  # and a NumPy array of that dtype. Returns a list; raises nothing itself (the conversion
  # helpers may raise on bad values). No side effects.
  """Returns a list containing all the possible inputs with a given dtype.

  Args:
    val: value to convert to test input.
    dtype: a tuple of format (tf.Dtype, bool) where the bool value represents
      whether the dtype is "weak" or not.

  Returns:
    A list of all possible inputs given a value and a dtype.
  """
  python_inferred_types = {
      (dtypes.int32, True): 1,
      (dtypes.float32, True): 1.0,
      (dtypes.complex128, True): 1.0j,
  }
  dtype, weak = dtype
  inputs = []
  if weak:
    # WeakTensor and Python input types.
    inputs.append(convert_to_input_type(val, "WeakTensor", dtype))
    if dtype in python_inferred_types:
      # There are only 3 possible Python default types : int, float, complex.
      val_in_dtype = val * python_inferred_types[dtype]
      inputs.append(val_in_dtype)
      inputs.append(convert_to_input_type(val_in_dtype, "Tensor", None))
  else:
    # Tensor and NumPy input types.
    inputs.append(convert_to_input_type(val, "Tensor", dtype))
    inputs.append(convert_to_input_type(val, "NumPy", dtype))
  return inputs


def convert_to_input_type(base_input, input_type, dtype=None):
  # [FUNCTION DOC]
  # HELPER: converts a base value into the requested kind of test input.
  # input_type is one of the strings WeakTensor, Tensor, NumPy or Python. WeakTensor and Tensor
  # build a constant with the optional dtype (WeakTensor wraps it as weak); NumPy builds an
  # array, converting a TensorFlow dtype to its NumPy equivalent first; Python returns the
  # value unchanged. Raises a value error for any other input_type. No side effects.
  if input_type == "WeakTensor":
    return WeakTensor.from_tensor(constant_op.constant(base_input, dtype=dtype))
  elif input_type == "Tensor":
    return constant_op.constant(base_input, dtype=dtype)
  elif input_type == "NumPy":
    dtype = dtype.as_numpy_dtype if isinstance(dtype, dtypes.DType) else dtype
    return np.array(base_input, dtype=dtype)
  elif input_type == "Python":
    return base_input
  else:
    raise ValueError(f"The provided input_type {input_type} is not supported.")


def get_weak_tensor(*args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: shortcut that creates a constant tensor from the given arguments and wraps it as a
  # WeakTensor. All positional and keyword arguments are forwarded to the constant creation
  # function. Returns the WeakTensor.
  return WeakTensor.from_tensor(constant_op.constant(*args, **kwargs))


class DtypeConversionTestEnv:
  # [CLASS DOC]
  # HELPER: context manager used by tests to run a block under a specific dtype conversion
  # (type promotion) mode and restore the previous mode afterwards. Holds the previous and the
  # requested mode. Entering sets the global mode as a side effect; exiting restores it, even
  # when the block raised (exceptions are not suppressed).
  """Test environment for different dtype conversion semantics."""

  def __init__(self, promo_mode):
    # [FUNCTION DOC]
    # Records the currently active conversion mode (converted from its enum to a string) and
    # remembers promo_mode, the mode string to activate on entry. Does not change global state.
    self._old_promo_mode = ops.promo_mode_enum_to_string(
        ops.get_dtype_conversion_mode()
    )
    self._new_promo_mode = promo_mode

  def __enter__(self):
    # [FUNCTION DOC]
    # Activates the requested dtype conversion mode globally and returns this environment object.
    ops.set_dtype_conversion_mode(self._new_promo_mode)
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    # [FUNCTION DOC]
    # Restores the dtype conversion mode that was active when the object was created. The
    # exception arguments are ignored and nothing is returned, so exceptions propagate.
    ops.set_dtype_conversion_mode(self._old_promo_mode)
