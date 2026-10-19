# [MODULE DOC]
# Defines NoneTensor, a composite tensor standing in for a None value, and NoneTensorSpec, its
# batchable type specification (exported publicly as NoneTensorSpec). Used mainly so tf.data
# pipelines can carry None elements through structures. Import-time side effects: the spec is
# registered in the type spec registry under the name for NoneTensorSpec, and a value converter is
# registered so that the None value maps to NoneTensorSpec.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""NoneTensor and NoneTensorSpec classes."""

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.util.tf_export import tf_export


# TODO(b/149584798): add tests for non-tf.data functionality.
class NoneTensor(composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # Composite tensor that represents a None value. It holds no state and no component tensors;
  # its type spec is always a new NoneTensorSpec. Created by tf.data machinery, not by users.
  """Composite tensor representation for `None` value."""

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Property returning the type specification of this tensor: a fresh NoneTensorSpec instance.
    # No inputs, no side effects.
    return NoneTensorSpec()


# TODO(b/149584798): add tests for non-tf.data functionality.
@tf_export("NoneTensorSpec")
@type_spec_registry.register("tf.NoneTensorSpec")
class NoneTensorSpec(type_spec.BatchableTypeSpec):
  # [CLASS DOC]
  # Batchable type specification describing a None value. It has no components, an empty
  # serialization, and batching or unbatching just returns another NoneTensorSpec. Registered in
  # the type spec registry and exported publicly. Stateless; legacy dataset type, shape and class
  # accessors return the spec itself.
  """Type specification for `None` value."""

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # Property returning the Python value type this spec describes: the NoneTensor class.
    return NoneTensor

  def _serialize(self):
    # [FUNCTION DOC]
    # Returns the serialized form of the spec: an empty tuple, since the spec has no parameters.
    return ()

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Property returning the component specs: always an empty list, as None has no components.
    return []

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Converts a value to its component tensors. The value argument is ignored; returns an empty
    # list.
    return []

  def _from_components(self, components):
    # [FUNCTION DOC]
    # Rebuilds a value from components. The components argument is ignored; returns nothing
    # (the null value), which is the value this spec represents.
    return

  def _to_tensor_list(self, value):
    # [FUNCTION DOC]
    # Flattens a value to a list of tensors. The value is ignored; returns an empty list.
    return []

  @staticmethod
  def from_value(value):
    # [FUNCTION DOC]
    # Static method creating a spec from a value. The value is ignored (it is expected to be the
    # null value); always returns a new NoneTensorSpec. Registered as the converter for None.
    return NoneTensorSpec()

  def _batch(self, batch_size):
    # [FUNCTION DOC]
    # Returns the spec for a batch of this spec; the batch size is ignored and a new
    # NoneTensorSpec is returned.
    return NoneTensorSpec()

  def _unbatch(self):
    # [FUNCTION DOC]
    # Returns the spec for one unbatched element: a new NoneTensorSpec. Never raises.
    return NoneTensorSpec()

  def _to_batched_tensor_list(self, value):
    # [FUNCTION DOC]
    # Flattens a batched value to a list of tensors. The value is ignored; returns an empty list.
    return []

  def _to_legacy_output_types(self):
    # [FUNCTION DOC]
    # Legacy tf.data output types accessor; returns the spec itself.
    return self

  def _to_legacy_output_shapes(self):
    # [FUNCTION DOC]
    # Legacy tf.data output shapes accessor; returns the spec itself.
    return self

  def _to_legacy_output_classes(self):
    # [FUNCTION DOC]
    # Legacy tf.data output classes accessor; returns the spec itself.
    return self

  def most_specific_compatible_shape(self, other):
    # [FUNCTION DOC]
    # Computes the most specific spec compatible with this one and another. Input other is the
    # other type spec. Returns this spec if other has exactly the same type; otherwise raises a
    # value error stating that no type spec is compatible with both.
    if type(self) is not type(other):
      raise ValueError("No `TypeSpec` is compatible with both {} and {}".format(
          self, other))
    return self


type_spec.register_type_spec_from_value_converter(type(None),
                                                  NoneTensorSpec.from_value)
