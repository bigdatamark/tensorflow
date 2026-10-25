# [MODULE DOC]
# Tests for the embedding operations module. Checks that looking up a variable that was created
# uninitialized uses a sparse gather (no dense variable read) and that gradients of the lookup
# with respect to a resource variable have a statically known shape. All tests run in both graph
# and eager modes and are launched through the test main.
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
"""Tests for tensorflow.ops.embedding_ops."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class EmbeddingLookupTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the embedding lookup function, run in graph and eager modes. No shared state.

  def testEmbeddingLookupOnUninitializedVariableDoesSparseRead(self):
    # [FUNCTION DOC]
    # Creates an uninitialized 3 by 3 float variable, initializes it to zeros via a compiled
    # function, then looks up row zero in another compiled function. Asserts the init result is
    # zeros and the lookup gives a row of zeros. Inspects the graph and its function library:
    # expects exactly one resource gather node and no variable read node, proving a sparse read
    # rather than a dense read of the whole variable.
    x = resource_variable_ops.UninitializedVariable(
        trainable=True, shape=[3, 3], dtype=dtypes.float32)

    @def_function.function(input_signature=[])
    def _init():
      # [FUNCTION DOC]
      # HELPER: compiled initializer for the test. Assigns a 3 by 3 zeros array to the variable
      # and returns the assignment result.
      return x.assign(np.zeros([3, 3]))

    @def_function.function(input_signature=[])
    def _call():
      # [FUNCTION DOC]
      # HELPER: compiled function that looks up row zero of the variable using the version 2
      # embedding lookup and returns it.
      return embedding_ops.embedding_lookup_v2(x, [0])

    self.assertAllClose(self.evaluate(_init()), np.zeros([3, 3]))

    concrete_call = _call.get_concrete_function()
    self.assertAllClose(self.evaluate(concrete_call()), [[0., 0., 0.]])

    resource_gather_node = []
    read_var_node = []
    graph = concrete_call.graph.as_graph_def()
    for n in graph.node:
      if n.op == "ResourceGather":
        resource_gather_node.append(n)
      if n.op == "ReadVariableOp":
        read_var_node.append(n)

    for f in graph.library.function:
      for n in f.node_def:
        if n.op == "ResourceGather":
          resource_gather_node.append(n)
        if n.op == "ReadVariableOp":
          read_var_node.append(n)
    # There should be a single ResourceGather, but no ReadVariableOp
    # (dense read).
    self.assertLen(resource_gather_node, 1)
    self.assertLen(read_var_node, 0)

  def testEmbeddingLookupGradientsHaveKnownShape(self):
    # [FUNCTION DOC]
    # Creates a 3 by 3 zero-initialized float resource variable and a compiled function that
    # looks up row zero, sums it and takes the gradient with respect to the variable. Inside the
    # function it asserts the gradient shape is 3 by 3; outside it asserts the init result is
    # zeros and the evaluated gradient has ones in row zero and zeros elsewhere.
    x = resource_variable_ops.ResourceVariable(
        initial_value=np.zeros([3, 3]),
        trainable=True,
        shape=[3, 3],
        dtype=dtypes.float32)

    @def_function.function(input_signature=[])
    def _init():
      # [FUNCTION DOC]
      # HELPER: compiled initializer for the test. Assigns zeros to the variable and returns the
      # assignment result.
      return x.assign(np.zeros([3, 3]))

    @def_function.function(input_signature=[])
    def _call():
      # [FUNCTION DOC]
      # HELPER: compiled function that computes the gradient of the summed lookup of row zero
      # with respect to the variable under a gradient tape, asserts its static shape is 3 by 3,
      # and returns it converted to a dense tensor.
      with gradients.GradientTape() as tape:
        y = embedding_ops.embedding_lookup_v2(x, [0])
        loss = math_ops.reduce_sum(y)
      grads = tape.gradient(loss, x)
      self.assertAllEqual(grads.shape, [3, 3])
      return ops.convert_to_tensor(grads)

    self.assertAllClose(self.evaluate(_init()), np.zeros([3, 3]))

    concrete_call = _call.get_concrete_function()
    self.assertAllClose(
        self.evaluate(concrete_call()),
        [[1., 1., 1.], [0., 0., 0.], [0., 0., 0.]])


if __name__ == "__main__":
  googletest.main()
