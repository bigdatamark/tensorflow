# [MODULE DOC]
# Unit tests for Grappler constant folding. Verifies that a scan inside a while loop with CPU
# placement runs correctly on a CUDA GPU, and that graph optimization of a function with a
# gradient tape leaves only the expected variable update. Defines one test class; no import-time
# side effects; running it as a script starts the TensorFlow test runner.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Grappler Constant Folding."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


class ConstantFoldingTest(test.TestCase):
  # [CLASS DOC]
  # Regression tests for Grappler constant folding behavior, each tied to a tracked bug.
  # No shared fixtures beyond the base test case.

  # See b/76008022.
  def testScanInsideWhile(self):
    # [FUNCTION DOC]
    # Regression test: a scan inside a while loop body whose result is copied on the CPU. Only
    # runs when a CUDA GPU is available; then it executes a one-iteration loop over a zero
    # tensor of shape 10 by 20 by 30 and expects the result to be all zeros. Does nothing
    # otherwise.

    def loop_cond(idx_step, *unused_args):
      # [FUNCTION DOC]
      # HELPER: loop condition for the while loop in the scan-inside-while test.
      # Takes the step counter plus ignored extra loop variables; returns true while the
      # counter is below one, so the loop runs exactly once.
      return idx_step < 1

    def loop_body(idx_step, y):
      # [FUNCTION DOC]
      # HELPER: loop body for the while loop in the scan-inside-while test.
      # Takes the step counter and the previous value (ignored). Builds a zero tensor of shape
      # 10 by 20 by 30, runs a non-backprop, single-parallel-iteration scan with addition over
      # it, copies the scan result via an identity op pinned to the CPU, and returns the
      # incremented counter and that copy.
      x = array_ops.zeros([10, 20, 30], dtype=dtypes.float32)
      x = functional_ops.scan(
          math_ops.add,
          x,
          initializer=array_ops.zeros([20, 30], dtype=dtypes.float32),
          back_prop=False,
          parallel_iterations=1)

      with ops.device('/cpu:0'):
        y = array_ops.identity(x)

        return idx_step + 1, y

    if test.is_gpu_available(cuda_only=True):
      init_y = array_ops.zeros([10, 20, 30], dtype=dtypes.float32)
      _, y = while_loop.while_loop(
          loop_cond,
          loop_body,
          loop_vars=[0, init_y],
          back_prop=False,
          parallel_iterations=1)

      y_v = self.evaluate(y)
      self.assertAllEqual(np.zeros([10, 20, 30]), y_v)

  # See b/159753857.
  def testGradientGraphOptimization(self):
    # [FUNCTION DOC]
    # Regression test: traces a function that computes a gradient of a sum with respect to two
    # variables and adds the gradients back into them, runs it eagerly with optimized graph
    # collection, and inspects the optimized graph. Skipped under XLA. Expects exactly one
    # optimized graph, exactly one remaining assign-add variable op that reads input y, and
    # eleven nodes in total, i.e. the update of x is folded away.

    @def_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # HELPER: function under optimization in the gradient graph test.
      # Takes two variables. Records a product of x with zeros, adds y and sums, takes the
      # gradient with respect to both, assigns-adds each gradient into its variable (side
      # effect on the variables) and returns the sum of the two variables.
      with backprop.GradientTape() as tape:
        z = math_ops.mul(x, array_ops.zeros_like(x))
        l = math_ops.add(z, y)
        l = math_ops.reduce_sum(l)

      gx, gy = tape.gradient(l, [x, y])
      x.assign_add(gx)
      y.assign_add(gy)
      return x + y

    # XLA completely optimizes away the variable reads and
    # assignments, so skip the test.
    if test_util.is_xla_enabled():
      self.skipTest('Not relevant for XLA')
    with context.eager_mode():
      x = resource_variable_ops.ResourceVariable(
          np.random.uniform(size=[2, 2]), dtype=dtypes.float32)
      y = resource_variable_ops.ResourceVariable(
          np.random.uniform(size=[2, 2]), dtype=dtypes.float32)
      with context.collect_graphs(optimized=True) as graphs:
        f(x, y).numpy()
    self.assertLen(graphs, 1)
    assign_count = 0
    for node in graphs[0].node:
      if node.op == 'AssignAddVariableOp':
        self.assertEqual(node.input[0], 'y')
        assign_count += 1

    # Make sure that the only variable update that remains after
    # grappler optimization is that of y.
    self.assertEqual(assign_count, 1)
    self.assertLen(graphs[0].node, 11)


if __name__ == '__main__':
  test.main()
