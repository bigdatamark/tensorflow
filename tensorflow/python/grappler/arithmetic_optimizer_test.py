# [MODULE DOC]
# Test for the graph optimizer that simplifies arithmetic. Checks that when a function whose
# argument shapes must be inferred is run in eager mode with optimized graph collection, the
# optimized graph has four nodes and the third node is the transpose-folded-into-matrix-multiply
# node. Standalone test program.
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
"""Tests for Grappler Arithmetic Optimizer."""

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ArithmeticOptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the arithmetic optimizer, using collected optimized graphs from eager runs.

  # See b/146524878.
  def testFunctionArgShapeInference(self):
    # [FUNCTION DOC]
    # Scenario (regression test for a function argument shape inference bug): a traced function
    # transposes a 1536 by 384 matrix, reshapes it to 384 by 1536 and multiplies a 1 by 384 matrix
    # by it. Run in eager mode while collecting optimized graphs. Asserts exactly one graph was
    # collected, it has four nodes, and its third node is named as the optimizer node that folds
    # the transpose into the matrix multiply.

    @def_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Traced function: transposes y, reshapes it to 384 by 1536, and returns the matrix product
      # of x with it.
      return math_ops.matmul(
          x, array_ops.reshape(array_ops.transpose(y), [384, 1536]))

    with context.eager_mode():
      x = array_ops.ones((1, 384))
      y = array_ops.ones((1536, 384))
      with context.collect_graphs(optimized=True) as graphs:
        f(x, y).numpy()
      self.assertLen(graphs, 1)
      self.assertLen(graphs[0].node, 4)
      self.assertEqual(graphs[0].node[2].name,
                       'ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul')


if __name__ == '__main__':
  test.main()
