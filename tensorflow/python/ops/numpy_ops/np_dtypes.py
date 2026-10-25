# [MODULE DOC]
# Dtype aliases and dtype utilities for TF NumPy. Defines module-level names (bool_, int32, float64,
# complex128, string_, and so on) that are NumPy scalar types rather than TensorFlow dtypes, so users
# can combine them with NumPy facilities, and exports each of them under the experimental numpy
# namespace. Also exports iinfo and issubdtype. Holds two global switches, prefer-float32 and
# allow-float64, with getters and setters, plus helpers to canonicalize dtypes, compute the result
# type of mixed arrays, and cache NumPy dtypes for TensorFlow dtypes. Import-time side effects are
# the public API export registrations. Used by np_arrays, np_config and the other numpy_ops modules.
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
"""Dtypes and dtype utilities."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.util import tf_export


# We use numpy's dtypes instead of TF's, because the user expects to use them
# with numpy facilities such as `np.dtype(np.int64)` and
# `if x.dtype.type is np.int64`.
bool_ = np.bool_
tf_export.tf_export('experimental.numpy.bool_', v1=[]).export_constant(
    __name__, 'bool_'
)
complex128 = np.complex128
tf_export.tf_export('experimental.numpy.complex128', v1=[]).export_constant(
    __name__, 'complex128'
)
complex64 = np.complex64
tf_export.tf_export('experimental.numpy.complex64', v1=[]).export_constant(
    __name__, 'complex64'
)
float16 = np.float16
tf_export.tf_export('experimental.numpy.float16', v1=[]).export_constant(
    __name__, 'float16'
)
float32 = np.float32
tf_export.tf_export('experimental.numpy.float32', v1=[]).export_constant(
    __name__, 'float32'
)
float64 = np.float64
tf_export.tf_export('experimental.numpy.float64', v1=[]).export_constant(
    __name__, 'float64'
)
inexact = np.inexact
tf_export.tf_export('experimental.numpy.inexact', v1=[]).export_constant(
    __name__, 'inexact'
)
int_ = np.int_
tf_export.tf_export('experimental.numpy.int_', v1=[]).export_constant(
    __name__, 'int_'
)
int16 = np.int16
tf_export.tf_export('experimental.numpy.int16', v1=[]).export_constant(
    __name__, 'int16'
)
int32 = np.int32
tf_export.tf_export('experimental.numpy.int32', v1=[]).export_constant(
    __name__, 'int32'
)
int64 = np.int64
tf_export.tf_export('experimental.numpy.int64', v1=[]).export_constant(
    __name__, 'int64'
)
int8 = np.int8
tf_export.tf_export('experimental.numpy.int8', v1=[]).export_constant(
    __name__, 'int8'
)
object_ = np.object_
tf_export.tf_export('experimental.numpy.object_', v1=[]).export_constant(
    __name__, 'object_'
)
# np.string_ is aliased to np.bytes_ and depercated in numpy 2.0.
string_ = np.bytes_
tf_export.tf_export('experimental.numpy.string_', v1=[]).export_constant(
    __name__, 'string_'
)
uint16 = np.uint16
tf_export.tf_export('experimental.numpy.uint16', v1=[]).export_constant(
    __name__, 'uint16'
)
uint32 = np.uint32
tf_export.tf_export('experimental.numpy.uint32', v1=[]).export_constant(
    __name__, 'uint32'
)
uint64 = np.uint64
tf_export.tf_export('experimental.numpy.uint64', v1=[]).export_constant(
    __name__, 'uint64'
)
uint8 = np.uint8
tf_export.tf_export('experimental.numpy.uint8', v1=[]).export_constant(
    __name__, 'uint8'
)
# np.unicode_ is aliased to np.str_ and depercated in numpy 2.0.
unicode_ = np.str_
tf_export.tf_export('experimental.numpy.unicode_', v1=[]).export_constant(
    __name__, 'unicode_'
)
if int(np.__version__.split('.')[0]) < 2:
  complex_ = np.complex_
  float_ = np.float_
else:
  # Aliases np.complex_ and np.float_ have been removed in Numpy 2.0. Use
  # np.complex128 and np.float64 instead.
  complex_ = np.complex128
  float_ = np.float64
tf_export.tf_export('experimental.numpy.complex_', v1=[]).export_constant(
    __name__, 'complex_'
)
tf_export.tf_export('experimental.numpy.float_', v1=[]).export_constant(
    __name__, 'float_'
)


iinfo = np.iinfo
tf_export.tf_export('experimental.numpy.iinfo', v1=[]).export_constant(
    __name__, 'iinfo'
)


issubdtype = tf_export.tf_export('experimental.numpy.issubdtype', v1=[])(
    np.issubdtype
)


_to_float32 = {
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}


_cached_np_dtypes = {}


# Difference between is_prefer_float32 and is_allow_float64: is_prefer_float32
# only decides which dtype to use for Python floats; is_allow_float64 decides
# whether float64 dtypes can ever appear in programs. The latter is more
# restrictive than the former.
_prefer_float32 = False


# TODO(b/178862061): Consider removing this knob
_allow_float64 = True


def is_prefer_float32():
  # [FUNCTION DOC]
  # Returns the current value of the global prefer-float32 switch. Takes no inputs. When true, Python
  # floats and complex values are treated as 32-bit when inferring result types. No side effects.
  return _prefer_float32


def set_prefer_float32(b):
  # [FUNCTION DOC]
  # Sets the global prefer-float32 switch. Input b is a boolean (any truthy or falsy value is stored
  # as given, not validated). Returns nothing. Side effect: replaces the module-level flag read by
  # is_prefer_float32 and default_float_type.
  global _prefer_float32
  _prefer_float32 = b


def is_allow_float64():
  # [FUNCTION DOC]
  # Returns the current value of the global allow-float64 switch. Takes no inputs. When false, 64-bit
  # float and complex dtypes are canonicalized down to 32-bit and 64-bit-complex respectively. No
  # side effects.
  return _allow_float64


def set_allow_float64(b):
  # [FUNCTION DOC]
  # Sets the global allow-float64 switch. Input b is a boolean stored as given without validation.
  # Returns nothing. Side effect: replaces the module-level flag read by is_allow_float64,
  # canonicalize_dtype and default_float_type.
  global _allow_float64
  _allow_float64 = b


def canonicalize_dtype(dtype):
  # [FUNCTION DOC]
  # Maps a dtype to its canonical form given the allow-float64 switch. Input dtype is a NumPy dtype.
  # If float64 is disallowed and the dtype is float64 or complex128, returns float32 or complex64
  # respectively; any other dtype, or any dtype when float64 is allowed, is returned unchanged.
  # Does not raise for unknown dtypes (a lookup failure is swallowed). No side effects.
  if not _allow_float64:
    try:
      return _to_float32[dtype]
    except KeyError:
      pass
  return dtype


def _result_type(*arrays_and_dtypes):
  # [FUNCTION DOC]
  # HELPER: computes result dtypes for the TF NumPy type-promotion logic.
  # Returns the TensorFlow dtype that NumPy promotion rules give for a variable number of arrays,
  # scalars or dtypes. Python floats and complex numbers are first narrowed to 32-bit/64-bit-complex
  # when the prefer-float32 switch is on; then NumPy computes the common type, the result is
  # canonicalized according to allow-float64, and converted to a TensorFlow dtype. Raises whatever
  # NumPy or the dtype conversion raises for incompatible inputs. No side effects.
  """Returns the resulting type given a set of arrays."""

  def preprocess_float(x):
    # [FUNCTION DOC]
    # HELPER: nested helper of _result_type that narrows Python floats.
    # Input x is any value. If prefer-float32 is on, a Python float becomes a 32-bit float and a
    # Python complex becomes a 64-bit complex; everything else, and all values when the switch is
    # off, is returned unchanged.
    if is_prefer_float32():
      if isinstance(x, float):
        return np.float32(x)
      elif isinstance(x, complex):
        return np.complex64(x)
    return x

  arrays_and_dtypes = [preprocess_float(x) for x in arrays_and_dtypes]
  dtype = np.result_type(*arrays_and_dtypes)
  return dtypes.as_dtype(canonicalize_dtype(dtype))


def _get_cached_dtype(dtype):
  # [FUNCTION DOC]
  # HELPER: caching utility for converting TensorFlow dtypes to NumPy dtype objects.
  # KNOWN ISSUE: the global statement inside is unnecessary since the cache is only mutated, never rebound.
  # Input dtype is a TensorFlow dtype. Returns the corresponding NumPy dtype object, building it from
  # the dtype's NumPy type on first use and storing it in the module-level cache for later calls.
  # Side effect: grows the module-level cache.
  """Returns an np.dtype for the TensorFlow DType."""
  global _cached_np_dtypes
  try:
    return _cached_np_dtypes[dtype]
  except KeyError:
    pass
  cached_dtype = np.dtype(dtype.as_numpy_dtype)
  _cached_np_dtypes[dtype] = cached_dtype
  return cached_dtype


def default_float_type():
  # [FUNCTION DOC]
  # Returns the default floating point type as a NumPy scalar type. Takes no inputs. Returns float64
  # when prefer-float32 is off and allow-float64 is on; otherwise returns float32. No side effects.
  """Gets the default float type.

  Returns:
    If `is_prefer_float32()` is false and `is_allow_float64()` is true, returns
    float64; otherwise returns float32.
  """
  if not is_prefer_float32() and is_allow_float64():
    return float64
  else:
    return float32
