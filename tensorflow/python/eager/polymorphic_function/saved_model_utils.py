# [MODULE DOC]
# HELPER: helper module (shim layer) for saving and restoring functions to and from saved models.
# Defines a trackable class representing constants captured by functions, which is registered with
# the saved model serialization registry at import time (a global side effect). Used by the
# polymorphic function and concrete function code when exporting captured constants.
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
# pylint: disable=unidiomatic-typecheck
"""A shim layer for working with functions exported/restored from saved models.

This functionality should ultimately be moved into a first-class core API.
"""

import numpy

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base as trackable


@registration.register_tf_serializable()
class TrackableConstant(trackable.Trackable):
  # [CLASS DOC]
  # Trackable object representing a constant tensor captured by a function from a parent graph,
  # so that the constant can be saved into and restored from a saved model. Holds the captured
  # tensor, the function that captured it, and (after export) the copied constant tensor. It is
  # registered as a serializable type with the saved model registration, which is a class-level
  # side effect at import time. Overrides the export, serialize and deserialize hooks.
  """Trackable class for captured constants."""
  __slots__ = ("capture", "function", "_exported_tensor")

  def __init__(self, capture, function):
    # [FUNCTION DOC]
    # Constructor. Stores the captured tensor and the owning function, and initializes the
    # exported tensor to nothing; it is filled in during export. Returns nothing.
    self.capture = capture
    self.function = function
    self._exported_tensor = None

  def _export_to_saved_model_graph(self, tensor_map, **unused_kwargs):
    # [FUNCTION DOC]
    # Export hook called when saving. Evaluates the captured tensor to a static constant value.
    # Inputs: tensor_map, a mapping from captured tensors to exported-graph tensors, which is
    # mutated here by adding an entry for the capture; extra keyword arguments are ignored.
    # If the capture is an array of more than one element whose entries are all equal, the
    # constant is rebuilt from the single repeated value plus the shape, to avoid writing the
    # whole array into the graph definition (size limits); otherwise the full value is copied.
    # Stores the new tensor as the exported tensor and returns a list holding the original
    # capture. Raises a value error if the capture cannot be converted to a static value.
    capture_constant_value = tensor_util.constant_value(self.capture)
    if capture_constant_value is None:
      raise ValueError(
          f"Unable to save function {self.function.name} because it "
          f"captures graph tensor {self.capture} from a parent function which "
          "cannot be converted to a constant with `tf.get_static_value`.")

    if numpy.prod(self.capture.shape.as_list()) > 1 and numpy.all(
        capture_constant_value == capture_constant_value.flat[0]):
      # For the common case of a constant array filled with the same
      # value, rebuild the constant op specifically with the shape arg,
      # since otherwise the whole array is written into the node def,
      # causing performance and graph proto size issues (protos cannot be
      # bigger than 2GB).
      copied_tensor = constant_op.constant(
          capture_constant_value.flat[0],
          dtype=self.capture.dtype,
          shape=self.capture.shape)
    else:
      copied_tensor = constant_op.constant(capture_constant_value)

    tensor_map[self.capture] = copied_tensor
    self._exported_tensor = copied_tensor
    return [self.capture]

  def _serialize_to_proto(self, object_proto=None, **kwargs):
    # [FUNCTION DOC]
    # Serialization hook. Records, in the constant field of the given object proto (mutated in
    # place), the name of the operation that produced the exported tensor. Extra keyword
    # arguments are ignored. Returns nothing. Must run after export; otherwise the exported
    # tensor is still nothing and an attribute error would result. The proto argument defaults to
    # nothing but is effectively required.
    object_proto.constant.operation = self._exported_tensor.op.name

  @classmethod
  def _deserialize_from_proto(cls, object_proto, operation_attributes,
                              **kwargs):
    # [FUNCTION DOC]
    # Class-level deserialization hook used when loading. Inputs: the object proto (naming the
    # constant's operation) and a mapping from operation names to their attributes; extra keyword
    # arguments are ignored. Looks up the value tensor proto for the operation, converts it to an
    # array, and creates a new constant tensor from it; string-typed constants are created under
    # a CPU device scope. Returns the constant tensor itself, not an instance of this class.
    tensor_proto = (
        operation_attributes[object_proto.constant.operation]["value"].tensor)
    ndarray = tensor_util.MakeNdarray(tensor_proto)
    if dtypes.as_dtype(tensor_proto.dtype) == dtypes.string:
      with ops.device("CPU"):
        # String operations should be done on the CPU.
        imported_constant = constant_op.constant(ndarray)
    else:
      imported_constant = constant_op.constant(ndarray)
    return imported_constant
