# [MODULE DOC]
# Key-performance-indicator micro benchmarks for low-level eager execution primitives:
# creating a constant, converting a nested list to a tensor, a relu call, and invoking an
# identity compiled function, each on a two-by-two input and each wrapped in a profiler trace
# scope so it is tracked as a KPI trace. Exposes a timing helper and one benchmark class built
# on the micro-benchmark base; running the file as a script launches the test/benchmark main.
# No import-time side effects other than defining the iteration-count constant.
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
r"""KPI Benchmarks for low-level eager execution primitives.

This is a suite of full end-to-end integration benchmakr for low-level eager
execution APIs. Also tracks them as KPI Traceme.

To run CPU benchmarks:
  bazel run -c opt kpi_benchmarks_test -- --benchmark_filter=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" kpi_benchmarks_test -- \
    --benchmark_filter=.

To run a subset of benchmarks using --benchmarks flag.
--benchmarks: the list of benchmarks to run. The specified value is interpreted
as a regular expression and any benchmark whose name contains a partial match
to the regular expression is executed.
e.g. --benchmark_filter=".*matmul*." will run all matmul related benchmarks.

"""
import gc
import time

import tensorflow as tf

from tensorflow.python.eager import benchmarks_test_base
from tensorflow.python.eager import context
from tensorflow.python.profiler import trace

NUM_ITERATIONS = 30000


def _run_benchmark(func, num_iters, execution_mode=None):
  # [FUNCTION DOC]
  # HELPER: timing routine passed to the benchmark report machinery by KpiBenchmarks.
  # Runs the given callable once as warm-up, then num_iters times, under the chosen execution
  # mode (default none, meaning the context default; sync or async). In async mode it waits on
  # the executor after warm-up and after the loop so all work is finished. Returns the elapsed
  # wall-clock seconds of the timed loop (warm-up excluded).
  ctx = context.context()
  with context.execution_mode(execution_mode):
    # call func to warm up
    func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    start = time.time()
    for _ in range(num_iters):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    end = time.time()

    return end - start


class KpiBenchmarks(benchmarks_test_base.MicroBenchmarksBase):
  # [CLASS DOC]
  # Collection of KPI micro benchmarks of basic eager operations (constant creation, tensor
  # conversion, relu, compiled-function call). Each benchmark defines a small closure wrapped
  # in a profiler trace and runs it NUM_ITERATIONS times through the shared run helper.
  """A Collection of KPI benchmarks."""

  def _get_benchmark_name(self):
    # [FUNCTION DOC]
    # HELPER: naming hook for the micro-benchmark base class.
    # Returns the benchmark name by delegating to the base class's name lookup; no arguments
    # besides self and no side effects.
    return self._get_name()

  def _run(self, func, num_iters):
    # [FUNCTION DOC]
    # HELPER: common runner for the benchmark methods of this class.
    # Disables the garbage collector after forcing one collection, reports a benchmark by
    # timing the callable func for num_iters iterations through the report helper, then
    # re-enables garbage collection. Returns nothing. Side effect: toggles global GC state and
    # records benchmark results. If the report call raises, garbage collection stays disabled.
    gc.disable()
    gc.collect()
    self.run_report(_run_benchmark, func, num_iters)
    gc.enable()

  def benchmark_tf_constant_2x2(self):
    # [FUNCTION DOC]
    # Benchmark: creating a constant tensor from a two-by-two nested list of floats, repeated
    # NUM_ITERATIONS times inside a trace scope named for the constant op.
    x = [[1., 2.], [3., 4.]]

    def fn():
      # [FUNCTION DOC]
      # Timed body: opens a trace scope and creates a constant from the captured list; the
      # result is discarded.
      with trace.Trace("tf.constant-2x2"):
        tf.constant(x)

    self._run(fn, NUM_ITERATIONS)

  def benchmark_tf_convert_to_tensor_2x2(self):
    # [FUNCTION DOC]
    # Benchmark: converting a two-by-two nested list of floats to a tensor, repeated
    # NUM_ITERATIONS times inside a trace scope named for the conversion.
    x = [[1., 2.], [3., 4.]]

    def fn():
      # [FUNCTION DOC]
      # Timed body: opens a trace scope and converts the captured list to a tensor; the result
      # is discarded.
      with trace.Trace("tf.convert_to_tensor-2x2"):
        tf.convert_to_tensor(x)

    self._run(fn, NUM_ITERATIONS)

  def benchmark_tf_nn_relu_2x2(self):
    # [FUNCTION DOC]
    # Benchmark: applying relu to a pre-built two-by-two float tensor, repeated NUM_ITERATIONS
    # times inside a trace scope named for relu.
    x = tf.constant([[1., 2.], [3., 4.]])

    def fn():
      # [FUNCTION DOC]
      # Timed body: opens a trace scope and applies relu to the captured tensor; the result is
      # discarded.
      with trace.Trace("tf.nn.relu-2x2"):
        tf.nn.relu(x)

    self._run(fn, NUM_ITERATIONS)

  def benchmark_tf_function_invocation_identity(self):
    # [FUNCTION DOC]
    # Benchmark: calling a compiled identity function on a two-by-two float tensor, repeated
    # NUM_ITERATIONS times inside a trace scope named for the function call. The first call
    # (warm-up) traces the function, so the timed calls measure invocation overhead only.
    x = tf.constant([[1., 2.], [3., 4.]])

    @tf.function
    def identity(x):
      # [FUNCTION DOC]
      # Compiled function that returns its argument unchanged.
      return x

    def fn():
      # [FUNCTION DOC]
      # Timed body: opens a trace scope and calls the identity function on the captured
      # tensor; the result is discarded.
      with trace.Trace("tf.function-identity"):
        identity(x)

    self._run(fn, NUM_ITERATIONS)


if __name__ == "__main__":
  tf.test.main()
