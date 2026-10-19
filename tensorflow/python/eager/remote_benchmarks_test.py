# [MODULE DOC]
# Micro-benchmarks for eager execution against remote workers: sending a function call to a remote
# task, receiving a remote variable, and creating variables inside a remotely placed function.
# Uses in-process local gRPC servers as the "remote" workers. Run via the benchmark filter flag.
# No import-time side effects; the main guard runs the test/benchmark runner.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
r"""Benchmarks for remote worker eager execution.

To run CPU benchmarks:
  bazel run -c opt remote_benchmarks_test -- --benchmark_filter=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" remote_benchmarks_test -- \
    --benchmark_filter=.
"""

import gc
import time


from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import server_lib


def run_benchmark(func, num_iters, execution_mode=None):
  # [FUNCTION DOC]
  # HELPER: timing utility for the remote worker micro-benchmarks.
  # Runs func once as warm-up, then times num_iters further calls inside the given eager execution
  # mode (default none, meaning the current mode; async mode also waits on the executor before
  # starting and after finishing the timed loop). Returns the elapsed wall-clock seconds as a
  # number. Side effects: temporarily switches the eager execution mode; calls func many times.
  ctx = context.context()
  with context.execution_mode(execution_mode):
    # call func to maybe warm up the GPU
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


class Foo(object):
  # [CLASS DOC]
  # HELPER: small callable model used by the variable-creation benchmark.
  # Holds a count of variables and a lazily created list of scalar random variables; calling it
  # multiplies the input by every variable in turn.

  def __init__(self, num_vars):
    # [FUNCTION DOC]
    # Stores the number of variables to create (num_vars, integer) and starts with an empty list
    # of variables; no variables are created until the first call.
    self._num_vars = num_vars
    self._v = []

  def __call__(self, inputs):
    # [FUNCTION DOC]
    # On first call creates the configured number of scalar variables initialized from random
    # uniform values (side effect: variable creation on the current device). Then multiplies the
    # inputs tensor by each variable in sequence and returns the resulting tensor.
    if not self._v:
      for _ in range(self._num_vars):
        self._v.append(variables.Variable(
            random_ops.random_uniform([]), shape=[]))
    for v in self._v:
      inputs = inputs * v
    return inputs


class RemoteWorkerMicroBenchmarks(test.Benchmark):
  # [CLASS DOC]
  # Benchmark suite measuring eager remote-execution overhead. Holds two cached in-process local
  # servers (and their address targets without the grpc prefix) that act as remote workers.

  def __init__(self):
    # [FUNCTION DOC]
    # Starts two local in-process servers and stores each server and its target address with the
    # grpc scheme prefix removed. Side effect: opens two local server ports.
    # used for remote benchmarks
    self._cached_server1 = server_lib.Server.create_local_server()
    self._cached_server_target1 = self._cached_server1.target[len("grpc://"):]
    self._cached_server2 = server_lib.Server.create_local_server()
    self._cached_server_target2 = self._cached_server2.target[len("grpc://"):]

  def _run(self, func, num_iters=1000, execution_mode=context.ASYNC):
    # [FUNCTION DOC]
    # HELPER: runs and reports one benchmark.
    # Times num_iters (default 1000) calls of func in the given execution mode (default async),
    # then reports the iteration count, mean microseconds per iteration as wall time, and an
    # extra examples-per-second figure. Returns nothing; side effect is a benchmark report.
    total_time = run_benchmark(func, num_iters, execution_mode)
    mean_us = total_time * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={"examples_per_sec": num_iters / total_time})

  def benchmark_send(self):
    # [FUNCTION DOC]
    # Benchmark: connects to the first local server, then repeatedly sends a local 2x2 tensor to
    # a compiled function placed on the remote worker task 0 CPU that multiplies the matrix by
    # itself. Measures the cost of sending inputs and executing remotely. Forces garbage
    # collection at the end so remote resources are destroyed before the next server definition.
    remote.connect_to_remote_host(self._cached_server_target1)

    x = random_ops.random_uniform((2, 2)).cpu()

    @def_function.function
    def remote_func(m):
      # [FUNCTION DOC]
      # Compiled function returning the matrix product of the input with itself.
      return math_ops.matmul(m, m)

    def func(m):
      # [FUNCTION DOC]
      # Calls the compiled function under a device scope pinned to remote worker task 0 CPU and
      # returns its result.
      with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
        return remote_func(m)

    self._run(lambda: func(x))
    # NOTE(b/136184459): Force garbage collecting hanging resources before
    # subsequent calls to set_server_def, to ensure the destroy resource ops are
    # executed when their corresponding device and manager are still available.
    gc.collect()

  def benchmark_worker_recv(self):
    # [FUNCTION DOC]
    # Benchmark: connects to both local servers, creates a scalar variable on worker task 1, and
    # repeatedly runs on worker task 0 a compiled function that adds one to that variable, so the
    # value must be received from another worker. Forces garbage collection at the end.
    remote.connect_to_remote_host(
        [self._cached_server_target1, self._cached_server_target2])

    with ops.device("job:worker/replica:0/task:1/device:CPU:0"):
      v = variables.Variable(1.0)

    @def_function.function
    def remote_func():
      # [FUNCTION DOC]
      # Compiled function returning one plus the captured remote variable.
      return 1.0 + v

    def func():
      # [FUNCTION DOC]
      # Runs the compiled function under a device scope for worker task 0 CPU and returns the
      # result.
      with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
        return remote_func()

    self._run(func)
    # NOTE(b/136184459): Force garbage collecting hanging resources before
    # subsequent calls to set_server_def, to ensure the destroy resource ops are
    # executed when their corresponding device and manager are still available.
    gc.collect()

  def benchmark_create_vars_inside_function(self):
    # [FUNCTION DOC]
    # Benchmark: connects to the first local server and repeatedly (100 iterations, async mode)
    # builds a fifty-variable Foo layer on the remote worker and invokes it inside a compiled
    # function on a random scalar, measuring variable creation plus remote execution cost.
    # Forces garbage collection at the end.
    remote.connect_to_remote_host(self._cached_server_target1)

    def func():
      # [FUNCTION DOC]
      # Under the remote worker CPU scope, creates a new layer with fifty variables, defines a
      # compiled function applying it to a random scalar, and returns that function's result.
      with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
        layer = Foo(50)

        @def_function.function
        def remote_func():
          # [FUNCTION DOC]
          # Applies the layer to a random scalar under the remote worker CPU scope and returns
          # the result.
          with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
            return layer(random_ops.random_uniform([]))

        return remote_func()

    self._run(func, execution_mode=context.ASYNC, num_iters=100)
    # NOTE(b/136184459): Force garbage collecting hanging resources before
    # subsequent calls to set_server_def, to ensure the destroy resource ops are
    # executed when their corresponding device and manager are still available.
    gc.collect()


if __name__ == "__main__":
  test.main()
