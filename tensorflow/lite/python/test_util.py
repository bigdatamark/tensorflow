# [MODULE DOC]
# HELPER: shared helper module for TFLite test files.
# Provides functions that parse a serialized TFLite model (flatbuffer bytes) and report the set
# of operator names used and the output tensor shapes of every subgraph. Used by converter and
# interpreter tests. No import-time side effects beyond imports.
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
"""Functions used by multiple tflite test files."""

from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.lite.tools import visualize


def get_ops_list(model_data):
  # [FUNCTION DOC]
  # HELPER: test utility for asserting which ops a converted model contains.
  # Input: model_data, the serialized TFLite flatbuffer bytes. Walks every operator of every
  # subgraph, resolves its operator code, and collects a name: for custom ops the custom code
  # string decoded as text, for builtin ops the human-readable builtin name from the visualize
  # tool. Returns a set of operator name strings (duplicates collapsed). Raises whatever the
  # flatbuffer parser raises for malformed data. No side effects.
  """Returns a set of ops in the tflite model data."""
  model = schema_fb.Model.GetRootAsModel(model_data, 0)
  op_set = set()

  for subgraph_idx in range(model.SubgraphsLength()):
    subgraph = model.Subgraphs(subgraph_idx)
    for op_idx in range(subgraph.OperatorsLength()):
      op = subgraph.Operators(op_idx)
      opcode = model.OperatorCodes(op.OpcodeIndex())
      builtin_code = schema_util.get_builtin_code_from_operator_code(opcode)
      if builtin_code == schema_fb.BuiltinOperator.CUSTOM:
        opname = opcode.CustomCode().decode("utf-8")
        op_set.add(opname)
      else:
        op_set.add(visualize.BuiltinCodeToName(builtin_code))
  return op_set


def get_output_shapes(model_data):
  # [FUNCTION DOC]
  # HELPER: test utility for asserting output shapes of a converted model.
  # Input: model_data, serialized TFLite flatbuffer bytes. For every subgraph in order, and for
  # each of its output tensors in order, appends that tensor's shape as a list of integers.
  # Returns the flat list of shapes (one entry per output across all subgraphs). Raises
  # whatever the flatbuffer parser raises for malformed data. No side effects.
  """Returns a list of output shapes in the tflite model data."""
  model = schema_fb.Model.GetRootAsModel(model_data, 0)

  output_shapes = []
  for subgraph_idx in range(model.SubgraphsLength()):
    subgraph = model.Subgraphs(subgraph_idx)
    for output_idx in range(subgraph.OutputsLength()):
      output_tensor_idx = subgraph.Outputs(output_idx)
      output_tensor = subgraph.Tensors(output_tensor_idx)
      output_shapes.append(output_tensor.ShapeAsNumpy().tolist())

  return output_shapes
