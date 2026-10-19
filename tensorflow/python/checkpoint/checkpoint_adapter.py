# [MODULE DOC]
# Experimental checkpoint adapter API. Defines ReshardCallback, a hook that lets a restore
# operation rename the object being restored, change which checkpoint keys and slice specs are
# read, and combine the values read into the final resharded value; and AbstractCheckpointAdapter,
# an abstract base for adapters that decide whether a trackable object needs resharding and
# supply its callback. Used by the checkpoint restore machinery. No import-time side effects.
#
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Experimental API for checkpoint adapter."""
import abc
from typing import List, Optional

from tensorflow.python.framework import tensor
from tensorflow.python.trackable import base


class ReshardCallback:
  # [CLASS DOC]
  # Base class for resharding a checkpoint value during restore. The default behavior is the
  # identity: same object name, single checkpoint key read as requested, single value returned.
  # Subclasses override methods to customize. Holds no state.
  """API to reshard a checkpoint value during restore.

  When a ReshardCallback is attached to a CheckpointPosition, the restored value
  of the checkpoint position is resharded based on this callback.
  """

  def object_name(self) -> str:
    # [FUNCTION DOC]
    # KNOWN ISSUE: annotated as returning a string but the default returns the null value.
    # Returns the local object name to restore into; the default returns nothing (null), meaning
    # the name is unchanged. Subclasses override to supply a different name. No side effects.
    """Returns the local name of the object being restored.

    Override this method when the local name of object is different than in the
    checkpoint.
    """
    return None

  def reshard(
      self,
      checkpoint_values: List[tensor.Tensor],
      shape_and_slice_spec: List[str],
  ) -> tensor.Tensor:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a list of tensors, but the default returns a
    # single tensor (the first and only checkpoint value).
    # Combines the values read from the checkpoint into the value to restore. Inputs: a list of
    # tensors as read from file, and the list of shape-and-slice specs requested by the caller
    # (ignored by the default). The default returns the sole value; raises a value error if the
    # list does not contain exactly one value. No side effects.
    """Reshards the checkpoint values as read from the checkpoint file.

    Override this to reshard/modify the restored values
    Args:
      checkpoint_values: The values returned by the restore op, as read from
        file.
      shape_and_slice_spec: The shape and slice spec required by the caller.

    Returns:
      List of restored Tensor values after being resharded.
    """
    del shape_and_slice_spec  # unused
    # Default reshard is a trivial one.
    if len(checkpoint_values) != 1:
      raise ValueError("Default reshard expects a single checkpoint value.")
    return checkpoint_values[0]

  def update_restore_inputs(
      self, checkpoint_key, shape_and_slice_spec
  ) -> tuple[List[str], List[str]]:
    # [FUNCTION DOC]
    # Lets the callback change what the restore operation reads. Inputs: the checkpoint key and
    # the shape-and-slice spec requested by the caller. Returns a pair of lists (checkpoint keys,
    # specs) of equal length; the default returns one-element lists with the inputs unchanged.
    # The number of keys matches the number of values later passed to reshard. No side effects.
    """Updates the specs to restore op.

    Override this method if the arguments to restore op need to be updated as
    per the resharding required.
    Args:
      checkpoint_key: The checkpoint key as requested by the caller
      shape_and_slice_spec: The shape and slice spec as requested by caller

    Returns:
    Tuple of list of checkpoint_keys and specs that the restore op should fetch
    as per the resharding requirement. The length of checkpoint keys returned by
    this method will match the length of checkpoint_values that are input to
    `reshard`.
    """
    return ([checkpoint_key], [shape_and_slice_spec])


class AbstractCheckpointAdapter(abc.ABC):
  # [CLASS DOC]
  # Abstract base for experimental checkpoint adapters. Concrete adapters are built from a
  # checkpoint path, say whether a trackable needs resharding, and return a ReshardCallback by
  # name. maybe_reshard combines these into a (name, callback) answer for the restore code.
  """Abstract API for checkpoint adapter.

  This is an experimental API that specifies how checkpoint restore should be
  adapted for specific trackable objects.
  """

  @classmethod
  @abc.abstractmethod
  def create_from_checkpoint(cls, path: str):
    # [FUNCTION DOC]
    # Abstract class-level factory: subclasses build an adapter by inspecting the checkpoint at
    # the given path (string). Declared to return an adapter instance; the base returns nothing.
    """Create factory to create an Adapter from checkpoint.

    Args:
      path: Path to checkpoint.
    """

  @abc.abstractmethod
  def is_applicable(self, trackable: base.Trackable) -> bool:
    # [FUNCTION DOC]
    # Abstract: subclasses return a boolean saying whether the given trackable being restored
    # should have its checkpoint value resharded by this adapter.
    """Returns whether the adapter is applicable to trackable for resharding.

    Args:
      trackable: A Trackable object that is being restored.

    Returns:
      A Boolean indicating if the checkpoint value for this Trackable should be
      resharded.
    """

  @abc.abstractmethod
  def get_reshard_callback(self, name: str) -> Optional[ReshardCallback]:
    # [FUNCTION DOC]
    # Abstract: subclasses return the ReshardCallback for the trackable with the given name, or
    # nothing (null) when no resharding applies.
    """Returns the reshard callback for the trackable with `name`."""

  def maybe_reshard(self, name: str) -> tuple[str, Optional[ReshardCallback]]:
    # [FUNCTION DOC]
    # Looks up the callback for the given name. Returns a pair (name, callback): with no callback
    # the original name and nothing; with a callback whose object name is non-empty, that object
    # name and the callback; otherwise the original name and the callback.
    """Returns the updated name and ReshardCallback applicable to it."""
    callback = self.get_reshard_callback(name)
    if callback is None:
      return name, None
    if callback.object_name():
      return callback.object_name(), callback
    return name, callback
