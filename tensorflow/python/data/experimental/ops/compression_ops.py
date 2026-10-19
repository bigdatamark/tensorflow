# [MODULE DOC]
# HELPER: small helper module with ops that compress a dataset element into a single variant
# tensor and uncompress it back given its type spec. Used by the tf.data service machinery.
# No import-time side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for compressing and uncompressing dataset elements."""
from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops


def compress(element):
  # [FUNCTION DOC]
  # Parameter: element (nested structure of supported values). Infers its type spec, flattens it
  # to a list of tensors and returns a single variant tensor holding the compressed element.
  """Compress a dataset element.

  Args:
    element: A nested structure of types supported by Tensorflow.

  Returns:
    A variant tensor representing the compressed element. This variant can be
    passed to `uncompress` to get back the original element.
  """
  element_spec = structure.type_spec_from_value(element)
  tensor_list = structure.to_tensor_list(element_spec, element)
  return ged_ops.compress_element(tensor_list)


def uncompress(element, output_spec):
  # [FUNCTION DOC]
  # Parameters: element (scalar variant tensor produced by compress); output_spec (nested type
  # specs of the original element). Uses the flattened tensor types and shapes of the spec to
  # uncompress, then rebuilds and returns the original nested structure.
  """Uncompress a compressed dataset element.

  Args:
    element: A scalar variant tensor to uncompress. The element should have been
      created by calling `compress`.
    output_spec: A nested structure of `tf.TypeSpec` representing the type(s) of
      the uncompressed element.

  Returns:
    The uncompressed element.
  """
  flat_types = structure.get_flat_tensor_types(output_spec)
  flat_shapes = structure.get_flat_tensor_shapes(output_spec)
  tensor_list = ged_ops.uncompress_element(
      element, output_types=flat_types, output_shapes=flat_shapes)
  return structure.from_tensor_list(output_spec, tensor_list)
