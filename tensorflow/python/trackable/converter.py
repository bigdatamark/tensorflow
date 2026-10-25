# [MODULE DOC]
# HELPER: conversion utility for the trackable package.
# Exposes a single function that turns an arbitrary Python object into a Trackable, wrapping
# containers via the data-structures module and plain tensors as trackable constants from the
# polymorphic-function saved-model utilities. Used by saving/loading code. No import-time side
# effects.

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
"""Util for converting a Python object to a Trackable."""


from tensorflow.python.eager.polymorphic_function import saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures


def convert_to_trackable(obj, parent=None):
  # [FUNCTION DOC]
  # HELPER: used by saved-model and checkpoint code to coerce arbitrary attribute values into
  # trackable objects.
  # Inputs: obj is any Python object; parent is an optional owner object passed through when a
  # constant wrapper is created. Output: the object itself if it is already a Trackable; otherwise
  # the object is first passed through the data-structure wrap/unwrap step (so lists, dicts and
  # tuples become trackable containers). If the result is a tensor-like value whose dtype is
  # neither variant nor resource and which is not a resource variable, it is wrapped in a
  # trackable constant tied to the parent and returned. Otherwise, if the result is still not a
  # Trackable, raises a value error saying it cannot be converted. No side effects.
  """Converts `obj` to `Trackable`."""
  if isinstance(obj, base.Trackable):
    return obj
  obj = data_structures.wrap_or_unwrap(obj)
  if (tensor_util.is_tf_type(obj) and
      obj.dtype not in (dtypes.variant, dtypes.resource) and
      not resource_variable_ops.is_resource_variable(obj)):
    return saved_model_utils.TrackableConstant(obj, parent)
  if not isinstance(obj, base.Trackable):
    raise ValueError(f"Cannot convert {obj} to Trackable.")
  return obj
