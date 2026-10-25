# [MODULE DOC]
# Benchmark module measuring the runtime cost of a conditional whose branch produces many
# intermediate values, comparing the v1 and v2 control flow implementations, each inside a
# compiled function and in legacy graph mode. Defines one benchmark class and a main entry point
# that enables eager execution and runs the test/benchmark runner. No import-time side effects
# beyond imports.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark for control flow ops."""

import time

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class CondWithManyIntermediatesBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class that times a conditional op whose true branch sums a thousand random
  # values, so many intermediate tensors exist. Class constants set the intermediate count,
  # measured iteration count and warm-up iteration count. Public benchmark methods toggle the
  # global control-flow-v2 switch, run a helper, then restore the switch.
  """Checks the runtime performance of outputting all intermediates."""

  NUM_INTERMEDIATES = 1000
  NUM_ITERS = 500
  NUM_WARM_UP_ITERS = 50

  def _create_cond(self, x):
    # [FUNCTION DOC]
    # HELPER: builds the conditional under test for the benchmarks.
    # Takes x, a scalar tensor. Returns the output tensor of a conditional whose predicate is a
    # dynamic not-equal test against minus one (so it cannot be constant folded). The true branch
    # returns x plus the sum of many random normal scalars (count set by the class constant, so
    # they are not foldable); the false branch returns the constant 0.0. Adds ops to the current graph.

    def branch_fn():
      # [FUNCTION DOC]
      # HELPER: true-branch callback for the conditional. Takes no arguments; returns x plus the sum
      # of the configured number of random normal scalars. Creates that many random ops as a side effect.
      # Use a random value so the adds can't be constant folded.
      return x + sum(random_ops.random_normal([])
                     for _ in range(self.NUM_INTERMEDIATES))

    # Use a dynamic predicate to make sure the cond isn't constant folded.
    return cond.cond(math_ops.not_equal(x, -1),
                     branch_fn, lambda: 0.0)

  def _benchmark_defun(self):
    # [FUNCTION DOC]
    # HELPER: shared body for the defun benchmarks. Wraps the conditional in a compiled function,
    # runs warm-up calls, then times the configured number of calls and reports the wall time and
    # iteration count through the benchmark reporter. Takes no arguments; returns nothing.
    """Benchmarks cond in a defun."""

    @def_function.function
    def cond_fn(x):
      # [FUNCTION DOC]
      # HELPER: compiled function body; takes a scalar x and returns the conditional output built
      # by the helper that creates it.
      return self._create_cond(x)

    # Warm up
    for _ in range(self.NUM_WARM_UP_ITERS):
      cond_fn(0.0)

    start_time = time.time()

    for _ in range(self.NUM_ITERS):
      cond_fn(0.0)

    self.report_benchmark(
        wall_time=time.time() - start_time,
        iters=self.NUM_ITERS)

  def _benchmark_graph(self):
    # [FUNCTION DOC]
    # HELPER: shared body for the legacy graph benchmarks. In graph mode with a fresh graph, builds
    # a float placeholder and the conditional, opens a session, makes a callable, warms up, then
    # times the configured number of calls and reports wall time and iterations. Returns nothing.
    """Benchmarks cond in legacy graph mode."""
    with context.graph_mode():
      with ops.Graph().as_default():
        x = array_ops.placeholder(dtypes.float32)
        cond_val = self._create_cond(x)

        with session.Session() as sess:
          cond_fn = sess.make_callable(cond_val, [x])

          # Warm up
          for _ in range(self.NUM_WARM_UP_ITERS):
            cond_fn(0.0)

          start_time = time.time()

          for _ in range(self.NUM_ITERS):
            cond_fn(0.0)

          self.report_benchmark(
              wall_time=time.time() - start_time,
              iters=self.NUM_ITERS)

  def benchmark_cond_v1_defun(self):
    # [FUNCTION DOC]
    # Benchmark: conditional in a compiled function using control flow v1. Disables the global
    # control-flow-v2 flag, runs the defun helper, then restores the previous flag value. Returns
    # nothing. (The flag is not restored if the helper raises.)
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = False
    self._benchmark_defun()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val

  def benchmark_cond_v2_defun(self):
    # [FUNCTION DOC]
    # Benchmark: conditional in a compiled function using control flow v2. Enables the global
    # control-flow-v2 flag, runs the defun helper, then restores the previous flag value. Returns
    # nothing. (The flag is not restored if the helper raises.)
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
    self._benchmark_defun()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val

  def benchmark_cond_v1_graph(self):
    # [FUNCTION DOC]
    # Benchmark: conditional in legacy graph mode using control flow v1. Disables the global
    # control-flow-v2 flag, runs the graph helper, then restores the previous flag value. Returns
    # nothing. (The flag is not restored if the helper raises.)
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = False
    self._benchmark_graph()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val

  def benchmark_cond_v2_graph(self):
    # [FUNCTION DOC]
    # Benchmark: conditional in legacy graph mode using control flow v2. Enables the global
    # control-flow-v2 flag, runs the graph helper, then restores the previous flag value. Returns
    # nothing. (The flag is not restored if the helper raises.)
    old_val = control_flow_util.ENABLE_CONTROL_FLOW_V2
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
    self._benchmark_graph()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = old_val

if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
