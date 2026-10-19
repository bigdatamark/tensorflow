# [MODULE DOC]
# HELPER: helper module for polymorphic_function. Provides a flatten utility that expands composite
# tensors into components while leaving resource variables intact. No import-time side effects.
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
"""Utility to manipulate CompositeTensors in tf.function."""

from tensorflow.python.framework import composite_tensor
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest


# TODO(b/240337581, b/240337099): Remove this function when we de-alias
# dt_resource tensors or tf.nest support is_leaf.
def flatten_with_variables(inputs):
  # [FUNCTION DOC]
  # HELPER: flattening utility for building flat call inputs for concrete functions.
  # KNOWN ISSUE: the docstring says resource variables are not expanded, but the check only
  # skips expansion for composite values that are resource variables; plain variables
  # that are not composite tensors are simply kept as leaves by the generic flatten.
  # Flattens a nested structure. Input: any nested structure. Composite tensors (other than
  # resource variables) are expanded into their components, recursively; everything else is kept
  # as a leaf. Returns a flat list. No side effects.
  """Flattens `inputs` but don't expand `ResourceVariable`s."""
  # We assume that any CompositeTensors have already converted their components
  # from numpy arrays to Tensors, so we don't need to expand composites here for
  # the numpy array conversion. Instead, we do so because the flattened inputs
  # are eventually passed to ConcreteFunction()._call_flat, which requires
  # expanded composites.
  flat_inputs = []
  for value in nest.flatten(inputs):
    if (isinstance(value, composite_tensor.CompositeTensor) and
        not _pywrap_utils.IsResourceVariable(value)):
      components = value._type_spec._to_components(value)  # pylint: disable=protected-access
      flat_inputs.extend(flatten_with_variables(components))
    else:
      flat_inputs.append(value)
  return flat_inputs
