# [MODULE DOC]
# Unit tests for the flops statistics registered by the profiler's flops_registry module.
# Importing flops_registry (kept only for its registration side effect) makes the statistics
# available, and the tests query them through the generic per-node statistics lookup on graph
# mode (v1 only) graphs. Contains one test class, run via the standard TensorFlow test main.
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

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.profiler.internal import flops_registry  # pylint: disable=unused-import


class FlopsRegistryTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking the registered flops statistic for matrix multiplication. Each test
  # builds variables and a matmul in the default graph and inspects the node definitions.
  # No shared fixtures beyond the default test case setup.

  @test_util.run_v1_only('Test requires a Graph and NodeDef inspection')
  def testSimpleStatistics(self):
    # [FUNCTION DOC]
    # Scenario: multiply two random variables of shape 25x16 and 16x9 in the default graph.
    # Action: fetch the flops statistic for every operation in the graph. Expected: for the
    # operation named MatMul the flops value equals 7200 (25 times 9 outputs, inner size 16,
    # times two). Runs only in graph (v1) mode.
    a = variables.Variable(random_ops.random_normal([25, 16]))
    b = variables.Variable(random_ops.random_normal([16, 9]))
    math_ops.matmul(a, b)
    g = ops.get_default_graph()
    for op in g.get_operations():
      flops = ops.get_stats_for_node_def(g, op.node_def, 'flops').value
      if op.name == 'MatMul':
        self.assertEqual(7200, flops)

  @test_util.run_v1_only('Test requires a Graph and NodeDef inspection')
  def testTransposedStatistics(self):
    # [FUNCTION DOC]
    # Scenario: multiply variables of shape 16x25 and 16x9 with the first operand transposed.
    # Action: fetch the flops statistic for each graph operation. Expected: for the operation
    # named MatMul the flops value equals 7200, showing the transpose attribute selects the
    # right inner dimension. Runs only in graph (v1) mode.
    a = variables.Variable(random_ops.random_normal([16, 25]))
    b = variables.Variable(random_ops.random_normal([16, 9]))
    math_ops.matmul(a, b, transpose_a=True)
    g = ops.get_default_graph()
    for op in g.get_operations():
      flops = ops.get_stats_for_node_def(g, op.node_def, 'flops').value
      if op.name == 'MatMul':
        self.assertEqual(7200, flops)


if __name__ == '__main__':
  test.main()
