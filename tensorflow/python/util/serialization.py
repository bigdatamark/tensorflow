# [MODULE DOC]
# HELPER: JSON serialization hook used when saving Keras and TensorFlow configurations.
# Exposes get_json_type, intended as the default-handler argument of a JSON encoder, converting
# configurable objects, NumPy values, callables, types, tensor shapes, dtypes, mappings and the
# ellipsis into JSON-friendly structures. Depends on NumPy, wrapt and framework dtype/shape
# classes. No import-time side effects.

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
"""Utilities for serializing Python objects."""

import numpy as np
import wrapt

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util.compat import collections_abc


def get_json_type(obj):
  # [FUNCTION DOC]
  # HELPER: fallback converter for JSON encoding of objects the encoder cannot handle itself.
  # KNOWN ISSUE: because every callable (including classes) is handled by the callable branch
  # first, the later branch for Python types is unreachable for classes; and a callable object
  # without a name attribute raises an attribute error rather than the documented type error. The
  # mapping and proxy branches return values that may themselves need further conversion.
  # Input: obj (any object). Checked in order: objects with a configuration method give a mapping
  # of class name and config; NumPy arrays give nested lists and NumPy scalars give Python
  # scalars; callables give their name; dimensions give their value (note: callable check runs
  # before); tensor shapes give a list; dtypes give their name; mappings give a plain dictionary;
  # the ellipsis gives a marker mapping; wrapt proxies give the wrapped object.
  # Raises a type error with guidance to add a configuration method if nothing matches.
  """Serializes any object to a JSON-serializable structure.

  Args:
      obj: the object to serialize

  Returns:
      JSON-serializable structure representing `obj`.

  Raises:
      TypeError: if `obj` cannot be serialized.
  """
  # if obj is a serializable Keras class instance
  # e.g. optimizer, layer
  if hasattr(obj, 'get_config'):
    return {'class_name': obj.__class__.__name__, 'config': obj.get_config()}

  # if obj is any numpy type
  if type(obj).__module__ == np.__name__:
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return obj.item()

  # misc functions (e.g. loss function)
  if callable(obj):
    return obj.__name__

  # if obj is a python 'type'
  if type(obj).__name__ == type.__name__:
    return obj.__name__

  if isinstance(obj, tensor_shape.Dimension):
    return obj.value

  if isinstance(obj, tensor_shape.TensorShape):
    return obj.as_list()

  if isinstance(obj, dtypes.DType):
    return obj.name

  if isinstance(obj, collections_abc.Mapping):
    return dict(obj)

  if obj is Ellipsis:
    return {'class_name': '__ellipsis__'}

  if isinstance(obj, wrapt.ObjectProxy):
    return obj.__wrapped__

  raise TypeError(f'Object {obj} is not JSON-serializable. You may implement '
                  'a `get_config()` method on the class '
                  '(returning a JSON-serializable dictionary) to make it '
                  'serializable.')
