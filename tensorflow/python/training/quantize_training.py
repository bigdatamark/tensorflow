# [MODULE DOC]
# Thin Python wrapper around the native quantization-aware training graph rewriter.
# Exposes one deprecated, v1-only public function (exported under the train namespace) that
# serializes a graph definition, hands it to the compiled helper, and parses the rewritten result.
# Import-time side effects: registers the exported symbol via the export decorator and wraps it
# with a deprecation warning. Not a helper module; it is a public API shim.
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
"""Quantize training for TensorFlow."""
from tensorflow.core.framework import graph_pb2
from tensorflow.python._pywrap_quantize_training import DoQuantizeTrainingOnGraphDefHelper
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# Migrated this python code from deprecated quantize_training.i
@deprecation.deprecated(
    None,
    "GraphDef quantized training rewriter is deprecated in the long term.")
@tf_export(v1=["train.do_quantize_training_on_graphdef"])
def do_quantize_training_on_graphdef(input_graph, num_bits):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring summary talks about a contrib quantization scheme rather than
  # describing this function; the contrib namespace no longer exists in this release.
  # Rewrites a graph definition for quantization-aware training. Inputs: input_graph, a graph
  # definition message; num_bits, integer number of quantization bits. The graph is serialized to
  # bytes, passed to the native rewriting helper, and the returned bytes are parsed into a new
  # graph definition message which is returned. The input is not modified. Errors from the
  # native helper or from parsing propagate to the caller. Emits a deprecation warning when called.
  """A general quantization scheme is being developed in `tf.contrib.quantize`.

  Consider using that instead, though since it is in the tf.contrib namespace,
  it is not subject to backward compatibility guarantees.

  Args:
    input_graph: A `GraphDef`.
    num_bits: The number of bits for quantize training.

  Returns:
    The graph with quantize training done.
  """

  graph = graph_pb2.GraphDef()
  result_graph_string = DoQuantizeTrainingOnGraphDefHelper(
      input_graph.SerializeToString(), num_bits)

  graph.ParseFromString(result_graph_string)
  return graph
