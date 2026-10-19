# [MODULE DOC]
# Implements TensorFlow's flexible (JAX-style) implicit dtype promotion semantics. The module
# builds a full two-way promotion table over boolean, unsigned and signed integer, bfloat16,
# float16, float32, float64, complex64 and complex128 types, plus "weak" variants of the Python
# style scalar types (int32, int64, float32, float64, complex128). Each table entry gives the
# result dtype, its weakness, and the promotion mode (SAFE or ALL) required to allow it.
# Strings only promote with strings. The public entry point is the result type function, which
# takes tensors, variables, NumPy values, Python numbers or dtypes and returns the promoted
# (dtype, weak) pair, honoring the global dtype conversion mode. The table is generated at import
# time by an initializer that fills module-level state. Used by the weak tensor and
# auto-conversion machinery elsewhere in the framework package.
#
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
# =============================================================================
"""Auto dtype conversion semantics for TF."""

import numpy as np

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import variables
from tensorflow.python.util import nest


# PromoMode Enum that denotes safe and all mode.
PromoMode = ops.PromoMode

# Namings are similar to third_party/py/jax/_src/dtypes.py.
_b8 = (dtypes.bool, False)
_u8 = (dtypes.uint8, False)
_u16 = (dtypes.uint16, False)
_u32 = (dtypes.uint32, False)
_u64 = (dtypes.uint64, False)
_i8 = (dtypes.int8, False)
_i16 = (dtypes.int16, False)
_i32 = (dtypes.int32, False)
_i64 = (dtypes.int64, False)
_bf16 = (dtypes.bfloat16, False)
_f16 = (dtypes.float16, False)
_f32 = (dtypes.float32, False)
_f64 = (dtypes.float64, False)
_c64 = (dtypes.complex64, False)
_c128 = (dtypes.complex128, False)
# Weak dtypes
_i32w = (dtypes.int32, True)
_i64w = (dtypes.int64, True)
_f32w = (dtypes.float32, True)
_f64w = (dtypes.float64, True)
_c128w = (dtypes.complex128, True)
# String
_str = (dtypes.string, False)

_all_dtypes = [
    _b8,
    _u8,
    _u16,
    _u32,
    _u64,
    _i8,
    _i16,
    _i32,
    _i64,
    _bf16,
    _f16,
    _f32,
    _f64,
    _c64,
    _c128,
    _i32w,
    _i64w,
    _f32w,
    _f64w,
    _c128w,
]
# Python numbers
_pi = int  # pylint: disable=invalid-name
_pf = float  # pylint: disable=invalid-name
_pc = complex  # pylint: disable=invalid-name

# Mappings between types_pb2.DataType values and numpy.dtypes.
_NP_TO_TF = dtypes._NP_TO_TF  # pylint: disable=protected-access

# OP(arg1, arg2) => (res, weak_type, safety level)
# If promotion mode is SAFE, results corresponding to ALL will be disallowed.
# This map only contains one-way dtype promotion results and is used to generate
# _BINARY_DTYPE_RES_FULL.
_BINARY_DTYPE_RES_HALF = {
    _b8: {
        _b8: (_b8, PromoMode.SAFE),
        _u8: (_u8, PromoMode.SAFE),
        _u16: (_u16, PromoMode.SAFE),
        _u32: (_u32, PromoMode.SAFE),
        _u64: (_u64, PromoMode.SAFE),
        _i8: (_i8, PromoMode.SAFE),
        _i16: (_i16, PromoMode.SAFE),
        _i32: (_i32, PromoMode.SAFE),
        _i64: (_i64, PromoMode.SAFE),
        _bf16: (_bf16, PromoMode.SAFE),
        _f16: (_f16, PromoMode.SAFE),
        _f32: (_f32, PromoMode.SAFE),
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c64, PromoMode.SAFE),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_i32w, PromoMode.SAFE),
        _i64w: (_i64w, PromoMode.SAFE),
        _f32w: (_f32w, PromoMode.SAFE),
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _u8: {
        _u8: (_u8, PromoMode.SAFE),
        _u16: (_u16, PromoMode.SAFE),
        _u32: (_u32, PromoMode.SAFE),
        _u64: (_u64, PromoMode.SAFE),
        _i8: (_i16, PromoMode.ALL),
        _i16: (_i16, PromoMode.SAFE),
        _i32: (_i32, PromoMode.SAFE),
        _i64: (_i64, PromoMode.SAFE),
        _bf16: (_bf16, PromoMode.SAFE),
        _f16: (_f16, PromoMode.SAFE),
        _f32: (_f32, PromoMode.SAFE),
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c64, PromoMode.SAFE),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_u8, PromoMode.SAFE),
        _i64w: (_u8, PromoMode.SAFE),
        _f32w: (_f64w, PromoMode.ALL),
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _u16: {
        _u16: (_u16, PromoMode.SAFE),
        _u32: (_u32, PromoMode.SAFE),
        _u64: (_u64, PromoMode.SAFE),
        _i8: (_i32, PromoMode.ALL),
        _i16: (_i32, PromoMode.ALL),
        _i32: (_i32, PromoMode.SAFE),
        _i64: (_i64, PromoMode.SAFE),
        _bf16: (_bf16, PromoMode.ALL),
        _f16: (_f16, PromoMode.ALL),
        _f32: (_f32, PromoMode.SAFE),
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c64, PromoMode.SAFE),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_u16, PromoMode.SAFE),
        _i64w: (_u16, PromoMode.SAFE),
        _f32w: (_f64w, PromoMode.ALL),
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _u32: {
        _u32: (_u32, PromoMode.SAFE),
        _u64: (_u64, PromoMode.SAFE),
        _i8: (_i64, PromoMode.ALL),
        _i16: (_i64, PromoMode.ALL),
        _i32: (_i64, PromoMode.ALL),
        _i64: (_i64, PromoMode.SAFE),
        _bf16: (_bf16, PromoMode.ALL),
        _f16: (_f16, PromoMode.ALL),
        _f32: (_f32, PromoMode.ALL),
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c64, PromoMode.ALL),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_u32, PromoMode.SAFE),
        _i64w: (_u32, PromoMode.SAFE),
        _f32w: (_f64w, PromoMode.ALL),
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _u64: {
        _u64: (_u64, PromoMode.SAFE),
        _i8: (_f64w, PromoMode.ALL),
        _i16: (_f64w, PromoMode.ALL),
        _i32: (_f64w, PromoMode.ALL),
        _i64: (_f64w, PromoMode.ALL),
        _bf16: (_bf16, PromoMode.ALL),
        _f16: (_f16, PromoMode.ALL),
        _f32: (_f32, PromoMode.ALL),
        _f64: (_f64, PromoMode.ALL),
        _c64: (_c64, PromoMode.ALL),
        _c128: (_c128, PromoMode.ALL),
        _i32w: (_u64, PromoMode.SAFE),
        _i64w: (_u64, PromoMode.SAFE),
        _f32w: (_f64w, PromoMode.ALL),
        _f64w: (_f64w, PromoMode.ALL),
        _c128w: (_c128w, PromoMode.ALL),
    },
    _i8: {
        _i8: (_i8, PromoMode.SAFE),
        _i16: (_i16, PromoMode.SAFE),
        _i32: (_i32, PromoMode.SAFE),
        _i64: (_i64, PromoMode.SAFE),
        _bf16: (_bf16, PromoMode.SAFE),
        _f16: (_f16, PromoMode.SAFE),
        _f32: (_f32, PromoMode.SAFE),
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c64, PromoMode.SAFE),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_i8, PromoMode.SAFE),
        _i64w: (_i8, PromoMode.SAFE),
        _f32w: (_f64w, PromoMode.ALL),
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _i16: {
        _i16: (_i16, PromoMode.SAFE),
        _i32: (_i32, PromoMode.SAFE),
        _i64: (_i64, PromoMode.SAFE),
        _bf16: (_bf16, PromoMode.ALL),
        _f16: (_f16, PromoMode.ALL),
        _f32: (_f32, PromoMode.SAFE),
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c64, PromoMode.SAFE),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_i16, PromoMode.SAFE),
        _i64w: (_i16, PromoMode.SAFE),
        _f32w: (_f64w, PromoMode.ALL),
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _i32: {
        _i32: (_i32, PromoMode.SAFE),
        _i64: (_i64, PromoMode.SAFE),
        _bf16: (_bf16, PromoMode.ALL),
        _f16: (_f16, PromoMode.ALL),
        _f32: (_f32, PromoMode.ALL),
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c64, PromoMode.ALL),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_i32, PromoMode.SAFE),
        _i64w: (_i32, PromoMode.SAFE),
        _f32w: (_f64w, PromoMode.ALL),
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _i64: {
        _i64: (_i64, PromoMode.SAFE),
        _bf16: (_bf16, PromoMode.ALL),
        _f16: (_f16, PromoMode.ALL),
        _f32: (_f32, PromoMode.ALL),
        _f64: (_f64, PromoMode.ALL),
        _c64: (_c64, PromoMode.ALL),
        _c128: (_c128, PromoMode.ALL),
        _i32w: (_i64, PromoMode.SAFE),
        _i64w: (_i64, PromoMode.SAFE),
        _f32w: (_f64w, PromoMode.ALL),
        _f64w: (_f64w, PromoMode.ALL),
        _c128w: (_c128w, PromoMode.ALL),
    },
    _bf16: {
        _bf16: (_bf16, PromoMode.SAFE),
        _f16: (_f32, PromoMode.ALL),
        _f32: (_f32, PromoMode.SAFE),
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c64, PromoMode.SAFE),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_bf16, PromoMode.SAFE),
        _i64w: (_bf16, PromoMode.SAFE),
        _f32w: (_bf16, PromoMode.SAFE),
        _f64w: (_bf16, PromoMode.SAFE),
        _c128w: (_c64, PromoMode.ALL),
    },
    _f16: {
        _f16: (_f16, PromoMode.SAFE),
        _f32: (_f32, PromoMode.SAFE),
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c64, PromoMode.SAFE),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_f16, PromoMode.SAFE),
        _i64w: (_f16, PromoMode.SAFE),
        _f32w: (_f16, PromoMode.SAFE),
        _f64w: (_f16, PromoMode.SAFE),
        _c128w: (_c64, PromoMode.ALL),
    },
    _f32: {
        _f32: (_f32, PromoMode.SAFE),
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c64, PromoMode.SAFE),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_f32, PromoMode.SAFE),
        _i64w: (_f32, PromoMode.SAFE),
        _f32w: (_f32, PromoMode.SAFE),
        _f64w: (_f32, PromoMode.SAFE),
        _c128w: (_c64, PromoMode.ALL),
    },
    _f64: {
        _f64: (_f64, PromoMode.SAFE),
        _c64: (_c128, PromoMode.ALL),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_f64, PromoMode.SAFE),
        _i64w: (_f64, PromoMode.SAFE),
        _f32w: (_f64, PromoMode.SAFE),
        _f64w: (_f64, PromoMode.SAFE),
        _c128w: (_c128, PromoMode.SAFE),
    },
    _c64: {
        _c64: (_c64, PromoMode.SAFE),
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_c64, PromoMode.SAFE),
        _i64w: (_c64, PromoMode.SAFE),
        _f32w: (_c64, PromoMode.SAFE),
        _f64w: (_c64, PromoMode.SAFE),
        _c128w: (_c64, PromoMode.SAFE),
    },
    _c128: {
        _c128: (_c128, PromoMode.SAFE),
        _i32w: (_c128, PromoMode.SAFE),
        _i64w: (_c128, PromoMode.SAFE),
        _f32w: (_c128, PromoMode.SAFE),
        _f64w: (_c128, PromoMode.SAFE),
        _c128w: (_c128, PromoMode.SAFE),
    },
    _i32w: {
        _i32w: (_i32w, PromoMode.SAFE),
        _i64w: (_i64w, PromoMode.SAFE),
        _f32w: (_f32w, PromoMode.SAFE),
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _i64w: {
        _i64w: (_i64w, PromoMode.SAFE),
        _f32w: (_f32w, PromoMode.SAFE),
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _f32w: {
        _f32w: (_f32w, PromoMode.SAFE),
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _f64w: {
        _f64w: (_f64w, PromoMode.SAFE),
        _c128w: (_c128w, PromoMode.SAFE),
    },
    _c128w: {
        _c128w: (_c128w, PromoMode.SAFE),
    },
}

# A full promotion table that contains two-way mappings.
# This table can be directly used for NumPy types as well, because
# e.g. `np.int32` == `tf.int32`.
_BINARY_DTYPE_RES_FULL = {}


def _initialize():
  # [FUNCTION DOC]
  # HELPER: builds the module-level promotion table used by the result type function.
  # Takes no inputs and returns nothing. For every ordered pair of known dtypes it looks up the
  # one-way table, falling back to the reversed pair when the key is missing, and stores the
  # result in the full table (mutating module-level state). Finally registers the string type as
  # promoting only with itself under the safe mode. Called once at import time.
  """Generate the rest of the promotion table from the one-way promotion table.

  Returns: None
  """
  for dtype1 in _all_dtypes:
    _BINARY_DTYPE_RES_FULL[dtype1] = {}
    for dtype2 in _all_dtypes:
      try:
        res = _BINARY_DTYPE_RES_HALF[dtype1][dtype2]
      except KeyError:
        res = _BINARY_DTYPE_RES_HALF[dtype2][dtype1]

      _BINARY_DTYPE_RES_FULL[dtype1][dtype2] = res

  # We do not support any conversions between string and others dtypes.
  _BINARY_DTYPE_RES_FULL[_str] = {_str: (_str, PromoMode.SAFE)}


_initialize()

# The name np.string_ and np.unicode_ were available as aliases to np.bytes_ and
# np.str_; however, they are removed in numpy 2.0. See
# https://numpy.org/devdocs/numpy_2_0_migration_guide.html#changes-to-namespaces.
_all_str_dtypes = (
    np.dtype('object_'),
    np.dtype('bytes_'),
    np.dtype('str_'),
    dtypes.string,
)


def _is_acceptable_input_type(x):
  # [FUNCTION DOC]
  # HELPER: input validation for the promotion logic.
  # Takes any object x and returns a boolean. True for indexed slices, weak tensors, variables,
  # and anything that is not a composite tensor; False for other composite tensors (for example
  # ragged or sparse tensors). No side effects.
  """Determines if x is an acceptable input type for auto dtype conversion semantics."""
  # List of composite types that are supported by the auto dtype conversion
  # semantics.
  supported_composite_types = (
      indexed_slices.IndexedSlices,
      weak_tensor.WeakTensor,
      variables.Variable,
  )
  return isinstance(x, supported_composite_types) or not isinstance(
      x, composite_tensor.CompositeTensor
  )


def _get_dtype_and_weakness(x):
  # [FUNCTION DOC]
  # HELPER: maps one input to its (dtype, weak flag) key for the promotion table.
  # Input x may be a weak tensor (weak), a TF dtype, any object with a TF or NumPy dtype
  # attribute, a NumPy array or scalar, bytes or text (string), a NumPy type class, a Python bool,
  # int, float or complex value or type (weak types; bool is checked before int), a tensor shape
  # (int32, not weak) or a NumPy dtype object. Returns a pair of TF dtype and boolean weakness.
  # Raises OverflowError for Python ints outside the int32 range, and NotImplementedError for
  # NumPy dtypes without a TF equivalent or for any other unsupported type. No side effects.
  """Returns a TF type and weak type information from x.

  Args:
    x: an input scalar, array or a NumPy/TF/Python dtype.

  Raises:
    OverflowError: if Python int x is too large to convert to int32.
    NotImplementedError: when x is an unsupported input type.

  Returns:
    TF type and weak type information inferred from x in the form of
    (dtype, bool).
  """
  if isinstance(x, weak_tensor.WeakTensor):
    return (x.dtype, True)
  if isinstance(x, dtypes.DType):
    return (x, False)
  # TODO(b/286585200): Add support for `AutoCastVariable` in Keras.
  tf_dtype = getattr(x, 'dtype', None)
  if isinstance(tf_dtype, dtypes.DType):
    return (tf_dtype, False)
  # `isinstance(tf_dtype, np.dtype)` handles classes that implement `dtype`
  # using `np.dtype` (e.g. `xla_extension.Array`).
  # This condition is put before e.g. python int/float because
  # `isinstance(np.float64(1), float)` returns True.
  if isinstance(x, (np.ndarray, np.generic)) or isinstance(tf_dtype, np.dtype):
    # Use `dtypes.as_dtype(x.dtype)` because in `as_dtype`, the input will be
    # compared against a list of types including TF Dtypes which are protobufs.
    infer_dtype = dtypes.as_dtype(tf_dtype)
    return (infer_dtype, False)
  if isinstance(x, (bytes, str)) or tf_dtype in _all_str_dtypes:
    return _str
  try:
    if x in _NP_TO_TF:
      return (_NP_TO_TF[x], False)
  except TypeError:
    pass
  # bool type check must happen before int type check because
  # isinstance(True, int) == True (https://peps.python.org/pep-0285/).
  if isinstance(x, bool) or x == bool:
    return _b8
  # TODO(b/286585058): Update implementation depending on whether Python
  # scalars are inferred to 32 bit or 64 bit.
  if isinstance(x, _pi):
    if x < np.iinfo(np.int32).min or x > np.iinfo(np.int32).max:
      raise OverflowError(f'Python int {x} too large to convert to np.int32')
    return _i32w
  if x == int:
    return _i32w
  if isinstance(x, _pf) or x == float:
    return _f32w
  if isinstance(x, _pc) or x == complex:
    return _c128w
  if isinstance(x, tensor_shape.TensorShape):
    # Since TensorShape is always integer value, return int32.
    return _i32
  # Only support NumPy dtype objects with corresponding TF types.
  if isinstance(x, np.dtype):
    try:
      np_dtype = dtypes.as_dtype(x)
      return (np_dtype, False)
    except TypeError as exc:
      raise NotImplementedError(
          f'Auto dtype conversion semantics does not support {x}. Try using a'
          ' NumPy built-in dtype objects or cast them explicitly.'
      ) from exc
  raise NotImplementedError(
      f'Auto dtype conversion semantics does not support {type(x)} type.'
  )


def _result_type_impl(*arrays_and_dtypes):
  # [FUNCTION DOC]
  # HELPER: core implementation behind the public result type function.
  # Takes any number of inputs (tensors, variables, NumPy values, Python numbers, dtypes, nested
  # structures of these); None entries are dropped. Flattens the inputs, computes each one's
  # (dtype, weak) pair, and folds them left to right through the full promotion table, using base
  # dtypes so reference types behave like plain ones. With no valid inputs it returns a weak
  # float32. Returns the final (dtype, weak) pair. Reads the global dtype conversion mode.
  # Raises NotImplementedError for unsupported composite inputs or when the table has no entry for
  # a pair, and TypeError when a promotion needs ALL mode but the current mode is SAFE.
  """Internal implementation of jnp_style_result_type.

  Args:
    *arrays_and_dtypes: A list of Tensors, Variables, NumPy arrays or python
      numbers.

  Returns:
    The result promotion type from all the inputs.

  Raises:
    TypeError: when the promotion between the input dtypes is disabled in the
    current mode

    NotImplementedError:
      (1) When arrays_and_dtypes contains an unsupported input type (e.g.
      RaggedTensor).
      (2) When there isn't a possible promotion for the input dtypes.
  """
  promo_safety_mode = ops.get_dtype_conversion_mode()
  # Drop None inputs and check if input type is supported.
  valid_arrays_and_dtypes = []
  for inp in arrays_and_dtypes:
    if inp is not None:
      if _is_acceptable_input_type(inp):
        valid_arrays_and_dtypes.append(inp)
      else:
        raise NotImplementedError(
            'Auto dtype conversion semantics does not support'
            f' {type(inp)} type.'
        )

  dtypes_and_is_weak = [
      _get_dtype_and_weakness(x) for x in nest.flatten(valid_arrays_and_dtypes)
  ]

  # If there are no valid inputs, return f32.
  if not dtypes_and_is_weak:
    # If dtypes_and_is_weak is an empty list, return weakly-typed f32.
    dtypes_and_is_weak = [(dtypes.float32, True)]

  res = dtypes_and_is_weak[0]
  for arg in dtypes_and_is_weak[1:]:
    # Use `base_dtype` in case of `ref` types (for e.g. ref_variables).
    res = (res[0].base_dtype, res[1])
    arg = (arg[0].base_dtype, arg[1])
    try:
      res_next, allowed_mode = _BINARY_DTYPE_RES_FULL[res][arg]
    except KeyError as exc:
      # Throw NotImplementedError When there isn't a possible promotion for the
      # input dtypes. We will proceed with the default system promotion if
      # NotImplementedError is thrown.
      raise NotImplementedError(
          f'Implicit Conversion between {res[0]} and {arg[0]} is '
          'not allowed. Please convert the input manually if you '
          'need to.'
      ) from exc
    if allowed_mode.value > promo_safety_mode.value:
      raise TypeError(
          f'In promotion mode {promo_safety_mode}, implicit dtype '
          f'promotion between ({res[0]}, weak={res[1]}) and '
          f'({arg[0]}, weak={arg[1]}) is disallowed. '
          'You need to explicitly specify the dtype in your op, '
          'or relax your dtype promotion rules (such as from SAFE '
          'mode to ALL mode).'
      )
    res = res_next

  return res


def result_type(*arrays_and_dtypes):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a promotion type, but it actually returns a pair
  # of a TF dtype and a boolean weakness flag, as produced by the internal implementation.
  # Public entry point: takes any number of tensors, variables, NumPy values, Python numbers or
  # dtypes and delegates to the internal implementation. Returns the promoted (dtype, weak) pair.
  # Raises NotImplementedError for unsupported inputs and TypeError when the promotion is
  # disallowed in the current mode. Callers should catch NotImplementedError and fall back.
  """Determine the result promotion dtype using the JNP-like promotion system.

  Args:
    *arrays_and_dtypes: A list of Tensors, Variables, NumPy arrays or python
      numbers.

  Returns:
    The result promotion type from all the inputs.
  """
  # Make sure to catch NotImplementedError when using this method to account for
  # inputs that are not supported yet.
  return _result_type_impl(*arrays_and_dtypes)
