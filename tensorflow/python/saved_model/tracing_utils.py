# [MODULE DOC]
# Helper module for SavedModel export that traces a trackable object's tensor serialization and
# restoration methods into concrete functions. Exposes a single function used by the saving
# code. No import-time side effects.
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
"""Tracing utilities used by SavedModel."""

from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun


def trace_save_and_restore(obj):
  # [FUNCTION DOC]
  # HELPER: Used by SavedModel saving to get graph-friendly save and restore functions.
  # KNOWN ISSUE: The docstring says it returns a single concrete function, but it returns a
  # pair: the concrete save function and the concrete restore function.
  # Takes a trackable object. Uses the object's tensor serialization and restore-from-tensors
  # methods; if they are already concrete functions they are returned as is, otherwise they are
  # wrapped in traced functions. The traced save function raises a not-implemented error if
  # any serialized value is a callable, and prefixes keys with the object's legacy saveable name
  # when one exists; the traced restore function strips that prefix before calling the object's
  # restore method. The restore function is traced with the save function's structured outputs
  # as its input signature. Returns the (save, restore) pair; has no other side effects.
  """Traces `Trackable` serialize- and restore-from-tensors functions.

  Args:
    obj: A `Trackable` object.

  Returns:
    A concrete Function.
  """
  legacy_name = saveable_compat.get_saveable_name(obj)

  obj_save_fn = obj._serialize_to_tensors  # pylint: disable=protected-access
  obj_restore_fn = obj._restore_from_tensors  # pylint: disable=protected-access

  if isinstance(obj_save_fn, defun.ConcreteFunction):
    concrete_save = obj_save_fn
  else:
    @def_function.function
    def save_fn():
      # [FUNCTION DOC]
      # HELPER: Traced wrapper around the object's tensor serialization method.
      # Takes no arguments. Calls the object's serialization method, raises a not-implemented
      # error if any returned value is a callable, and returns the mapping of names to tensors,
      # with the legacy saveable name prepended to each key when one exists.
      tensor_dict = obj_save_fn()
      if any(isinstance(v, tensor_callable.Callable)
             for v in tensor_dict.values()):
        raise NotImplementedError(
            f"Unable to export SavedModel with object of type {type(obj)} "
            "because it returns a Callable in `_serialize_to_tensors`. "
            "If you need this functionality please file a feature request.")

      if legacy_name:
        # If there is a legacy decorator, append the name to the keys.
        return {f"{legacy_name}{key}": value
                for key, value in tensor_dict.items()}
      return tensor_dict

    concrete_save = save_fn.get_concrete_function()

  if isinstance(obj_restore_fn, defun.ConcreteFunction):
    concrete_restore = obj_restore_fn
  else:
    @def_function.function
    def restore_fn(restored_tensors):
      # [FUNCTION DOC]
      # HELPER: Traced wrapper around the object's restore-from-tensors method.
      # Takes a mapping of names to restored tensors, removes the legacy saveable name prefix
      # from keys when one exists, and calls the object's restore method. Returns nothing (the
      # method's own result is discarded).
      if legacy_name:
        # Do the opposite operation of save_fn()
        restored_tensors = {key[len(legacy_name):]: value
                            for key, value in restored_tensors.items()}
      obj_restore_fn(restored_tensors)

    concrete_restore = restore_fn.get_concrete_function(
        concrete_save.structured_outputs)

  return concrete_save, concrete_restore
