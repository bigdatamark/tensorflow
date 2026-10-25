# [MODULE DOC]
# Base types describing how objects are saved to and restored from checkpoints: a specification
# of one tensor (or tensor slice) to save, and an abstract base class for saveable objects that
# group such specifications under one name. Used by the saveable utilities, savers and
# variables in the training package. No import-time side effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Types for specifying saving and loading behavior."""


class SaveSpec:
  # [CLASS DOC]
  # Describes one tensor (or slice of a tensor) to be saved: the tensor or a callable producing
  # it, the slice specification, the checkpoint name, the data type and the device. Public
  # attributes are slice spec, name, dtype and device; the tensor is exposed through a property
  # that calls the producer when needed. Instances are created by saveable objects and consumed
  # by the savers.
  """Class used to describe tensor slices that need to be saved."""

  def __init__(self, tensor, slice_spec, name, dtype=None, device=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a null tensor means the spec is ignored, but the constructor
    # does not handle null: it would try to read the dtype of the null value and fail with an
    # attribute error.
    # Constructor. Inputs: tensor, a tensor or a callable producing one; slice_spec, the slice
    # to save; name, checkpoint name; dtype and device, optional, but both required when tensor is
    # callable. Stores tensor privately and slice_spec and name publicly.
    # If tensor is callable and dtype or device is missing, raises an assertion error. Otherwise
    # for a callable stores the given dtype and device; for a plain tensor takes dtype from the
    # tensor, and the device from the argument if given, else from the tensor.
    """Creates a `SaveSpec` object.

    Args:
      tensor: the tensor to save or callable that produces a tensor to save.
        If the value is `None`, the `SaveSpec` is ignored.
      slice_spec: the slice to be saved. See `Variable.SaveSliceInfo`.
      name: the name to save the tensor under.
      dtype: The data type of the Tensor. Required if `tensor` is callable.
        Used for error checking in the restore op.
      device: The device generating and consuming this tensor. Required if
        `tensor` is callable. Used to group objects to save by device.
    """
    self._tensor = tensor
    self.slice_spec = slice_spec
    self.name = name
    if callable(self._tensor):
      if dtype is None or device is None:
        raise AssertionError(
            "When passing a callable `tensor` to a SaveSpec, an explicit "
            "dtype and device must be provided.")
      self.dtype = dtype
      self.device = device
    else:
      self.dtype = tensor.dtype
      if device is not None:
        self.device = device
      else:
        self.device = tensor.device

  @property
  def tensor(self):
    # [FUNCTION DOC]
    # Property returning the tensor to save. If a callable was supplied, calls it on every access
    # (no caching) and returns the result; otherwise returns the stored tensor.
    return self._tensor() if callable(self._tensor) else self._tensor


class SaveableObject:
  # [CLASS DOC]
  # Base class for objects that can be saved to and restored from a checkpoint. Holds the
  # producing object, a list of save specs (all on one device) and the name to save under.
  # Subclasses override the restore method; the base version is abstract and always fails.
  """Base class for saving and restoring saveable objects."""

  def __init__(self, op, specs, name):
    # [FUNCTION DOC]
    # Constructor. Inputs: op, the producer object being wrapped (for example a variable);
    # specs, list of save specs, one per tensor, all on the same device; name, the name to save
    # under. Stores the three as attributes without validation. Returns nothing.
    """Creates a `SaveableObject` object.

    Args:
      op: the "producer" object that this class wraps; it produces a list of
        tensors to save.  E.g., a "Variable" object saving its backing tensor.
      specs: a list of SaveSpec, each element of which describes one tensor to
        save under this object. All Tensors must be on the same device.
      name: the name to save the object under.
    """
    self.op = op
    self.specs = specs
    self.name = name

  @property
  def device(self):
    # [FUNCTION DOC]
    # Property returning the device of the first save spec, taken to represent all specs. Raises an
    # index error if the specs list is empty.
    """The device for SaveSpec Tensors."""
    return self.specs[0].device

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # Abstract hook for restoring the object from loaded tensors. Inputs: restored_tensors, the
    # tensors read from a checkpoint; restored_shapes, shapes the object should conform to, or
    # null. The base implementation ignores both and always raises a value error saying an
    # abstract method was called; subclasses return an operation that restores the state.
    """Restores this object from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint
      restored_shapes: the shapes this object should conform to after
        restore, or None.

    Returns:
      An operation that restores the state of the object.

    Raises:
      ValueError: If the object cannot be restored using the provided
        parameters.
    """
    # pylint: disable=unused-argument
    raise ValueError("Calling an abstract method.")
