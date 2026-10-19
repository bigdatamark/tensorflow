# [MODULE DOC]
# Defines the Callable save-spec subclass: a placeholder that a trackable may return from its
# tensor serialization in place of a tensor, evaluated lazily on a given device when the
# checkpoint is written. No import-time side effects.
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
"""`Callable` class used for checkpointing."""

from tensorflow.python.training.saving import saveable_object


class Callable(saveable_object.SaveSpec):
  # [CLASS DOC]
  # A save spec whose tensor is produced by a no-argument function run at checkpoint write time
  # on a specified device. Holds the function as the spec's tensor plus dtype and device; there
  # is no name or slice spec. Not supported by SavedModel export for anything but resource
  # variables (export raises an error).
  """A callable that represents a Tensor that should be saved to checkpoint.

  This can be returned from `_serialize_to_tensor` in place of a Tensor. The
  callable will be executed on the specified device when the checkpoint is
  about to be written.

  Any class can use `Callable` for checkpointing, but for SavedModel export,
  only resource-type variables* are supported.

  * `resource_variable_ops.is_resource_variable(obj)` must return True.
  """

  def __init__(self, tensor_callable, dtype, device):
    # [FUNCTION DOC]
    # Inputs: tensor_callable (function with no arguments returning a tensor), dtype (data type
    # of that tensor), device (device string of that tensor). Initializes the base save spec with
    # the callable as its tensor and no name or slice spec. Returns nothing.
    """Initializes a `Callable` object.

    Args:
      tensor_callable: A callable that takes no arguments and returns a Tensor.
      dtype: Dtype of the tensor returned by the callable.
      device: Device of the tensor returned by the callable.
    """
    super().__init__(tensor_callable, None, None, dtype, device)
