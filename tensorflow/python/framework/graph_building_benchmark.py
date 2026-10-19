# [MODULE DOC]
# Benchmark module measuring how long it takes to add single operations (scalar add, batched
# matrix add, matrix multiply) to a graph in graph mode. It defines a timing helper and one
# benchmark class and is meant to be run as a script via the test main entry point. No import-time
# side effects other than imports; benchmarks only run when the benchmark runner selects them.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
r"""Benchmarks for low-level graph building primitives.

To run CPU benchmarks:
  bazel run -c opt graph_building_benchmarks -- --benchmark_filter=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" graph_building_benchmarks -- \
    --benchmark_filter=.

To run a subset of benchmarks using --benchmarks flag.
--benchmarks: the list of benchmarks to run. The specified value is interpreted
as a regular expression and any benchmark whose name contains a partial match
to the regular expression is executed.
e.g. --benchmark_filter=".*MatMul.*" will run all matmul related benchmarks.

"""
import time

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.platform import test


def run_benchmark(func, num_iters):
  # [FUNCTION DOC]
  # HELPER: timing utility for the graph-building benchmarks in this module.
  # Calls the given zero-argument callable num_iters times in a row and returns the total elapsed
  # wall-clock time in seconds (a float). Return values of the callable are discarded. Raises
  # whatever the callable raises; no other side effects.
  start = time.time()
  for _ in range(num_iters):
    func()
  end = time.time()
  return end - start


class SingleOpBenchmarks(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class with one method per single-operation graph-building scenario. Each benchmark
  # method creates placeholders in graph mode and measures the time to add one operation many
  # times. Holds no state of its own; results are reported through the benchmark framework.
  """Benchmark for graph building time of ops."""

  def _run_and_report(self, func, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared reporting step for the benchmark methods of this class.
    # Times num_iters calls of the given callable, computes the mean time per call in microseconds
    # and reports it to the benchmark framework as the wall time, together with the iteration
    # count and a rate figure (iterations per second, rounded to three decimals) as extra data.
    # Returns nothing. Side effect: emits a benchmark report entry.
    total_time = run_benchmark(func, num_iters)
    mean_us = total_time * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={
            "examples_per_sec": float("{0:.3f}".format(num_iters / total_time)),
        })

  def benchmarkAddScalars(self):
    # [FUNCTION DOC]
    # Benchmark: in graph mode, creates two scalar float placeholders and times adding them
    # together 1000 times via the generated add op, then reports the mean time per addition.
    with context.execution_mode(context.GRAPH_MODE):
      x = array_ops.placeholder(shape=[], dtype=dtypes.float32, name="x")
      y = array_ops.placeholder(shape=[], dtype=dtypes.float32, name="y")

      def bench():
        # [FUNCTION DOC]
        # HELPER: the timed workload; adds the two scalar placeholders and returns the new
        # add operation's output tensor, growing the graph by one node per call.
        return gen_math_ops.add(x, y)

      self._run_and_report(bench, 1000)

  def benchmarkAddBatchedMatrices(self):
    # [FUNCTION DOC]
    # Benchmark: in graph mode, creates two float placeholders of shape 32 by 784 by 1000 and
    # times adding them 1000 times, reporting the mean graph-building time per addition.
    with context.execution_mode(context.GRAPH_MODE):
      x = array_ops.placeholder(
          shape=[32, 784, 1000], dtype=dtypes.float32, name="x")
      y = array_ops.placeholder(
          shape=[32, 784, 1000], dtype=dtypes.float32, name="y")

      def bench():
        # [FUNCTION DOC]
        # HELPER: the timed workload; adds the two batched-matrix placeholders and returns the
        # resulting tensor, adding one node to the graph per call.
        return gen_math_ops.add(x, y)

      self._run_and_report(bench, 1000)

  def benchmarkMatMul(self):
    # [FUNCTION DOC]
    # Benchmark: in graph mode, creates float placeholders of shape 784 by 1000 and 1000 by 1000
    # and times multiplying them 1000 times, reporting the mean graph-building time per multiply.
    with context.execution_mode(context.GRAPH_MODE):
      x = array_ops.placeholder(
          shape=[784, 1000], dtype=dtypes.float32, name="x")
      y = array_ops.placeholder(
          shape=[1000, 1000], dtype=dtypes.float32, name="y")

      def bench():
        # [FUNCTION DOC]
        # HELPER: the timed workload; multiplies the two matrix placeholders and returns the
        # result tensor, adding one matrix-multiply node to the graph per call.
        return gen_math_ops.mat_mul(x, y)

      self._run_and_report(bench, 1000)


if __name__ == "__main__":
  test.main()
