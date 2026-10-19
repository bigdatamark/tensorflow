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
# [MODULE DOC]
# HELPER: thin Python wrapper over the native op_def_library extension, used by op_def_library.
# Exposes process_inputs, which calls the native implementation and converts the serialized
# attribute protos it returns into AttrValue messages. Import-time side effects: imports the
# native pywrap_tensorflow and _op_def_library_pybind extension modules.
"""Code to assist with the op_def_library."""

# pylint: disable=invalid-import-order, g-bad-import-order, unused-import
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import _op_def_library_pybind

from tensorflow.core.framework import attr_value_pb2


def process_inputs(op_name, producer_version, keywords):
  # [FUNCTION DOC]
  # HELPER: fast-path helper for _apply_op_helper in op_def_library.
  # Inputs: op_name (string name of the op type), producer_version (integer graph producer
  # version), keywords (mapping of argument names to values given to the op). Calls the native
  # processor, then parses each serialized attribute value in the returned mapping into an
  # attribute-value message in place. Returns a 4-tuple: attribute protos mapping, inputs, input
  # types, and output structure. Errors from the native layer propagate.
  """Helper method to speed up `_apply_op_helper` in op_def_library."""
  attr_protos, inputs, input_types, output_structure = (
      _op_def_library_pybind.process_inputs(op_name, producer_version,
                                            keywords))
  for k, attr in attr_protos.items():
    attr_protos[k] = attr_value_pb2.AttrValue.FromString(attr)
  return attr_protos, inputs, input_types, output_structure
