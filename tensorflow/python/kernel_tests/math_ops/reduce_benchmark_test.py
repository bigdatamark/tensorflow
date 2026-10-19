# [MODULE DOC]
# Simple benchmarks measuring the speed of the sum reduction's gradient in eager mode and in graph
# mode, each on the default device and pinned to the CPU. Exposes one benchmark class; the
# benchmarks run when the file is executed as a script. No import-time side effects.
#
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
"""Simple benchmarks for reductions and their gradients."""

import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ReduceBenchmarks(test.Benchmark):
  # [CLASS DOC]
  # Benchmark suite timing the gradient of a sum reduction over a 100x1000 tensor in eager and
  # graph modes, on the default device and on the CPU. Each benchmark method reports its result
  # through the benchmark reporting facility. No shared state.
  """Benchmarks for reductions."""

  def _run(self, func, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared timing loop for all benchmarks in this class. Calls the given callable once
    # to warm up (for example the GPU), then times num_iters further calls with wall-clock time.
    # Reports a benchmark with the iteration count, the mean wall time per iteration in
    # microseconds, and an extra entry for iterations per second. Returns nothing.
    # call func to maybe warm up the GPU
    func()
    start = time.time()
    for _ in range(num_iters):
      func()
    end = time.time()
    mean_us = (end - start) * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={"examples_per_sec": num_iters / (end - start)})

  def benchmark_reduce_sum_grad_eager(self):
    # [FUNCTION DOC]
    # Benchmark: in eager mode on the default device, times 10000 computations of the gradient of
    # a sum reduction over a 100x1000 zero tensor, using the timing helper. Returns nothing.
    with context.eager_mode():
      tensor = array_ops.zeros([100, 1000])

      def fn():
        # [FUNCTION DOC]
        # HELPER: the timed workload for the eager benchmark. Computes the gradient of the sum
        # reduction with respect to the captured tensor; returns nothing.
        backprop.gradients_function(math_ops.reduce_sum, [0])(tensor)

      self._run(fn, 10000)

  def benchmark_reduce_sum_grad_eager_cpu(self):
    # [FUNCTION DOC]
    # Benchmark: in eager mode on the CPU device, times 10000 computations of the gradient of a
    # sum reduction over a 100x1000 zero tensor, using the timing helper. Returns nothing.
    with context.eager_mode(), ops.device("/cpu:0"):
      tensor = array_ops.zeros([100, 1000])

      def fn():
        # [FUNCTION DOC]
        # HELPER: the timed workload for the CPU eager benchmark. Computes the gradient of the sum
        # reduction with respect to the captured tensor; returns nothing.
        backprop.gradients_function(math_ops.reduce_sum, [0])(tensor)

      self._run(fn, 10000)

  def benchmark_reduce_sum_grad_graph(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the session is created but its variable is unused; evaluation relies on the
    # test's own default session instead of the one opened here.
    # Benchmark: builds a graph with a 100x1000 float32 zero constant, its sum reduction and the
    # gradient of the sum, in a session with graph optimizations disabled (level zero). Times 10000
    # evaluations of the gradient op via the timing helper and reports the result. Returns nothing.
    config = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0)))
    with ops.Graph().as_default(), session.Session(config=config) as sess:

      tensor = constant_op.constant(np.zeros([100, 1000], dtype=np.float32))
      reduction = math_ops.reduce_sum(tensor)
      grad, = gradients_impl.gradients(reduction, tensor)

      def fn():
        # [FUNCTION DOC]
        # HELPER: the timed workload for the graph benchmark. Evaluates the gradient op in the
        # test's default session; takes no inputs and returns nothing.
        self.evaluate(grad.op)

      self._run(fn, 10000)

  def benchmark_reduce_sum_grad_graph_cpu(self):
    # [FUNCTION DOC]
    # Benchmark: same as the plain graph benchmark but the constant, the sum and its gradient are
    # all placed on the CPU device. Graph optimizations are disabled. Runs the gradient 10000
    # times via the timing helper and reports the result. Returns nothing.
    config = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0)))
    with ops.Graph().as_default(), session.Session(config=config) as sess:

      with ops.device("/cpu:0"):
        tensor = constant_op.constant(np.zeros([100, 1000], dtype=np.float32))
        reduction = math_ops.reduce_sum(tensor)
        grad, = gradients_impl.gradients(reduction, tensor)

      def fn():
        # [FUNCTION DOC]
        # HELPER: the timed workload for the CPU graph benchmark. Evaluates the gradient op in
        # the enclosing session; takes no inputs and returns nothing.
        self.evaluate(grad.op)

      self._run(fn, 10000)


if __name__ == "__main__":
  test.main()
