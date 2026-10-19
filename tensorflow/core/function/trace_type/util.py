# [MODULE DOC]
# HELPER: Small shared utilities for the trace_type package.
# Provides type-detection helpers (named tuple, attrs instance, NumPy array) and a helper that
# casts a sequence of values with a sequence of TraceTypes while reporting whether anything
# changed. Has no import-time side effects beyond importing NumPy.
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
"""Utilities for the trace_type module."""

from typing import Any, List, Tuple

import numpy as np


# TODO(b/225045380): Depend on the abstracted `leaf` lib from 'nest'.
def is_namedtuple(obj):
  # [FUNCTION DOC]
  # HELPER: Type check used by the tracing/TraceType code.
  # Returns true if the object has a field-names attribute whose entries are all strings (the
  # duck-typed signature of a named tuple). Any object is accepted; nothing is raised.
  return hasattr(obj, "_fields") and all(
      isinstance(field, str) for field in obj._fields)


# TODO(b/225045380): Depend on the abstracted `leaf` lib from 'nest'.
def is_attrs(obj):
  # [FUNCTION DOC]
  # HELPER: Type check used by the tracing/TraceType code.
  # Returns true if the object's class carries attrs-library metadata (the attrs attribute list),
  # meaning it is an instance of an attrs-decorated class.
  return hasattr(type(obj), "__attrs_attrs__")


# TODO(b/225045380): Depend on the abstracted `leaf` lib from 'nest'.
def is_np_ndarray(value):
  # [FUNCTION DOC]
  # HELPER: Type check used by the tracing/TraceType code.
  # Returns true if the value can be converted to an array (has the array-conversion method) and
  # is neither a NumPy string scalar (deliberately not promoted) nor a class object (NumPy dtype
  # classes have unbound versions of that method). Name suggests only ndarrays, but any
  # array-like object (for example tensors-like or array scalars) passes.
  return hasattr(value, "__array__") and not (
      # For legacy reasons we do not automatically promote Numpy strings.
      isinstance(value, np.str_)
      # NumPy dtypes have __array__ as unbound methods.
      or isinstance(value, type))


def cast_and_return_whether_casted(
    trace_types, values, context
) -> Tuple[List[Any], bool]:
  # [FUNCTION DOC]
  # HELPER: Shared by the container TraceTypes (tuple, list, named tuple, attrs, dict) in their
  # cast methods.
  # Takes a sequence of TraceTypes, a sequence of values and a casting context. Pairs them in
  # order (extra items in the longer one are ignored), casts each value with its TraceType, and
  # returns a list of casted values plus a flag that is true if any casted value is not the same
  # object as the input. Propagates any error raised by a cast.
  did_cast = False
  casted_values = []
  for t, v in zip(trace_types, values):
    casted_v = t.cast(v, context)
    casted_values.append(casted_v)
    if casted_v is not v:
      did_cast = True
  return casted_values, did_cast
