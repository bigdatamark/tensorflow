# [MODULE DOC]
# HELPER: shared test utility module for the proto splitter tests. Provides a function that
# synthesizes a GraphDef made of constant nodes with approximately requested serialized sizes,
# including nodes inside library functions. No import-time side effects.
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
"""Utilities for Python tests."""

from collections.abc import Sequence
import math
from typing import Optional

import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util


def make_graph_def_with_constant_nodes(
    node_sizes: Sequence[int],
    dtype: Optional[dtypes.DType] = None,
    **function_node_sizes,
) -> graph_pb2.GraphDef:
  # [FUNCTION DOC]
  # HELPER: builds test graphs for the splitter tests.
  # KNOWN ISSUE: the dtype argument is ignored; it is unconditionally overwritten with float32,
  # so the documented float32/float64 choice has no effect.
  # Inputs: node_sizes, sequence of approximate serialized byte sizes, one constant node each;
  # dtype, optional element type (ignored, see above); keyword arguments mapping a function name
  # to a sequence of node sizes, each producing a library function with such nodes.
  # Output: a GraphDef with constant nodes named by a running counter, filled with random
  # floating-point values sized to approximate the request. Raises a value error if a requested
  # size is smaller than an empty constant node. Uses the random number generator.
  """Creates a GraphDef with approximate node sizes.

  Args:
    node_sizes: list of ints, the approximate desired sizes of the nodes in the
      GraphDef.
    dtype: Dtype of encoded constant values (float32 or float64).
    **function_node_sizes: Map of function name to FunctionDef node sizes (see
      `node_sizes`).

  Returns:
    A GraphDef proto.
  """
  dtype = dtypes.float32
  graph_def = graph_pb2.GraphDef()
  n = 0

  def add_nodes(node_list, sizes):
    # [FUNCTION DOC]
    # HELPER: nested worker of the graph builder. Appends one constant node per entry in sizes
    # to node_list, named with the shared running counter. Each node starts with a scalar value
    # to measure its base size, then the value is replaced with a random array whose element
    # count is chosen to approach the requested size. Raises a value error if a size is below the
    # base size. Updates the enclosing counter; returns nothing.
    nonlocal n
    for s in sizes:
      node = node_list.add(name=f"Const_{n}", op="Const")

      # Add an empty value to compute the approximate size of the constant
      # value that will be added to the proto.
      node.attr["value"].tensor.MergeFrom(
          tensor_util.make_tensor_proto(np.ones([]), dtype=dtype)
      )
      remaining_size = s - node.ByteSize()
      if remaining_size < 0:
        raise ValueError(f"Unable to create node of size {s} bytes.")

      constant_size = [math.ceil(remaining_size / dtype.size)]
      node.attr["value"].tensor.Clear()
      node.attr["value"].tensor.MergeFrom(
          tensor_util.make_tensor_proto(
              np.random.random_sample(constant_size), dtype=dtype
          )
      )
      n += 1

  add_nodes(graph_def.node, node_sizes)
  for fn_name, fn_sizes in function_node_sizes.items():
    fn = graph_def.library.function.add()
    fn.signature.name = fn_name
    add_nodes(fn.node_def, fn_sizes)

  return graph_def
