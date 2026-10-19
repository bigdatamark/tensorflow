# [MODULE DOC]
# Compatibility helpers for the migration away from SaveableObject checkpointing. Provides a
# decorator that records a legacy saveable name on a class or object, a getter for that name, a
# module-level switch (default off) to force the new checkpoint conversion, and an exception type.
# Holds one mutable module-level flag; no other import-time side effects.
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
"""Checkpoint compatibility functions with SaveableObject.

Compatibility methods to ensure that checkpoints are saved with the same
metadata attributes before/after the SaveableObject deprecation.
"""

_LEGACY_SAVEABLE_NAME = "_LEGACY_SAVEABLE_NAME"


def legacy_saveable_name(name):
  # [FUNCTION DOC]
  # Decorator factory. Input: name, the string legacy saveable factory key. Returns a decorator
  # that stores the name as an attribute on the decorated class or object and returns it
  # unchanged. Used so new-style tensor serialization keeps the old checkpoint key naming.
  """Decorator to set the local name to use in the Checkpoint.

  Needed for migrating certain Trackables (see next paragraph) from the legacy
  `_gather_saveables_for_checkpoint` to the new `_serialize_to_tensors`
  function.

  This decorator should be used if the SaveableObject generates tensors with
  different names from the name that is passed to the factory.

  Example migration:

  *Before*

  ```
  class MyTrackable(Trackable):
    def _gather_saveables_for_checkpoint(self):
      return {"key": _MySaveable}

  class _MySaveable(SaveableObject):
    def __init__(self, name):
      specs = [
          SaveSpec(tensor1, "", name + "-1")
          SaveSpec(tensor2, "", name + "-2")
      ]
      super().__init__(None, specs, name)
  ```

  *After*

  ```
  @legacy_saveable_name("key")
  class MyTrackable(Trackable):

    def _serialize_to_tensors(self):
      return {"key-1": tensor1, "key-2": tensor2}
  ```

  Args:
    name: String name of the SaveableObject factory (the key returned in the
       `_gather_saveables_for_checkpoint` function)

  Returns:
    A decorator.
  """
  def decorator(cls_or_obj):
    # [FUNCTION DOC]
    # HELPER: the decorator returned by legacy_saveable_name.
    # Sets the legacy name attribute on the given class or object (mutating it) and returns the
    # same class or object.
    setattr(cls_or_obj, _LEGACY_SAVEABLE_NAME, name)
    return cls_or_obj
  return decorator


def get_saveable_name(cls_or_obj):
  # [FUNCTION DOC]
  # Returns the legacy saveable name previously set by the decorator on the given class or
  # object, or nothing if none was set. Read-only.
  return getattr(cls_or_obj, _LEGACY_SAVEABLE_NAME, None)


_FORCE_CHECKPOINT_CONVERSION = False


def force_checkpoint_conversion(value=True):
  # [FUNCTION DOC]
  # Sets the module-wide flag that forces checkpoints to use the new implementation (with
  # slightly different metadata). Input: value, boolean, default true. Returns nothing; mutates
  # module global state.
  """Forces checkpoint to use the new implementation.

  The new checkpoint implementation is changing the saved metadata slightly,
  and therefore may break forward compatibility in newly saved checkpoints. This
  means:

    - Previous versions of TensorFlow may not be able to load new checkpoints.
    - Backwards compatibility is unchanged: Old checkpoints can still be loaded.

  TensorFlow guarantees 3 weeks of forward compatibility, so this flag will be
  removed in the future weeks, after which checkpoint conversion will happen by
  default.

  **What happens when this flag is enabled?**

  The checkpoint will be saved with different metadata, meaning that previous
  versions of TensorFlow (<=2.10) will not be able to load this checkpoint.

  Args:
    value: Boolean value, whether or not to force checkpoint conversion to the
      new implementation.
  """
  # TODO(kathywu): Add definite date for flag removal.
  global _FORCE_CHECKPOINT_CONVERSION
  _FORCE_CHECKPOINT_CONVERSION = value


def force_checkpoint_conversion_enabled():
  # [FUNCTION DOC]
  # Returns the current value of the force-checkpoint-conversion flag (default false).
  return _FORCE_CHECKPOINT_CONVERSION


class CheckpointConversionError(Exception):
  # [CLASS DOC]
  # Plain exception type signalling a failure converting a checkpoint between the legacy and new
  # implementations. Adds no state or behavior.
  pass
