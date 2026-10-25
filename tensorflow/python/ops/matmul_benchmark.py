# [MODULE DOC]
# Purpose: benchmark for the matrix multiplication operator on a GPU device. It builds graph-mode
# (session based) graphs that multiply two random variable matrices for many combinations of sizes,
# transpose flags and floating point types, times repeated runs, prints throughput and reports it.
# Exposes: a graph-building function and a benchmark class derived from the framework benchmark
# base class; running the file as a script starts the standard test/benchmark main.
# Import-time side effects: none beyond imports; benchmarks only run when invoked by the runner.
# Uses the GPU device name reported by the test platform, so it needs a GPU to be meaningful.
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
"""Benchmark for Matmul operator."""

import itertools
import time

import numpy as np

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def build_graph(device, n, m, k, transpose_a, transpose_b, dtype):
  # [FUNCTION DOC]
  # HELPER: graph construction helper for the matmul benchmark class below.
  # Builds, inside the given device scope, two random-uniform non-resource variables and a matmul of
  # them. Input device is a device name string; n, m, k are integer dimensions (A is n by m, B is
  # m by k); transpose_a / transpose_b say whether the stored operand is laid out transposed (so the
  # variable shape is swapped, m by n or k by m) and are also passed to the matmul; dtype is the
  # numeric element type used for the random values.
  # Returns a grouped no-output control operation depending on the matmul result, suitable for
  # running in a session. Side effects: adds ops and variables to the current default graph.
  """Build a graph containing a sequence of matmul operations.

  Args:
    device: String, the device to run on.
    n: tensor A's first dimension size.
    m: tensor A's second dimension size.
    k: tensor B's second dimension size.
    transpose_a: boolean value to show if tensor A is transposed.
    transpose_b: boolean value to show if tensor B is transposed.
    dtype: numpy data type of the input tensor.

  Returns:
    A matmul operation to run()
  """
  with ops.device('%s' % device):
    if not transpose_a:
      x = variable_v1.VariableV1(
          random_ops.random_uniform([n, m], dtype=dtype), use_resource=False)
    else:
      x = variable_v1.VariableV1(
          random_ops.random_uniform([m, n], dtype=dtype), use_resource=False)
    if not transpose_b:
      y = variable_v1.VariableV1(
          random_ops.random_uniform([m, k], dtype=dtype), use_resource=False)
    else:
      y = variable_v1.VariableV1(
          random_ops.random_uniform([k, m], dtype=dtype), use_resource=False)

    z = math_ops.matmul(x, y, transpose_a=transpose_a, transpose_b=transpose_b)
    return control_flow_ops.group(z)


class MatmulBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class for the matmul operator. It holds no state of its own. The benchmark entry point
  # runs a sweep of shapes, transpose settings and float types on the GPU; helper methods build and
  # time one configuration. Instances are created by the benchmark runner, which calls methods whose
  # names start with the benchmark prefix. Inherits report_benchmark from the framework base class.
  """Benchmark matmul!"""

  def run_graph(self, device, n, m, k, transpose_a, transpose_b, num_iters,
                dtype):
    # [FUNCTION DOC]
    # Builds a fresh graph for one configuration, runs 500 warm-up iterations, then times num_iters
    # iterations in a session. Inputs: device name string, matrix sizes n, m, k, the two transpose
    # flags, the iteration count and the numeric dtype.
    # Side effects: prints a line with device, dtype, shape info, duration and throughput in
    # giga-items per second (counting two operations per multiply-add), and reports the benchmark
    # result through the framework with a name encoding device, dtype, sizes and transposes.
    # Returns the measured wall time in seconds for the timed iterations.
    # Initializes global variables inside the session before running.
    """Run the graph and print its execution time.

    Args:
      device: String, the device to run on.
      n: tensor A's first dimension size.
      m: tensor A's second dimension size.
      k: tensor B's second dimension size.
      transpose_a: boolean value to show if tensor A is transposed.
      transpose_b: boolean value to show if tensor B is transposed.
      num_iters: number of iterations to run the benchmark.
      dtype: numpy data type of the input tensor.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      output = build_graph(device, n, m, k, transpose_a, transpose_b, dtype)
      with session_lib.Session(graph=graph) as session:
        variables.global_variables_initializer().run()
        for _ in range(500):
          session.run(output)
        start_time = time.time()
        for _ in range(num_iters):
          session.run(output)
        duration = (time.time() - start_time)
        num_items = n * m * k * 2
        throughput = num_items * num_iters / duration / 1e9
        print('%s %s input_info:%s %d %.4fsec, %.4fGitems/s.' %
              (device, str(dtype), str(n) + 'x' + str(m) + 'x' + str(k) +
               ',ta:' + str(transpose_a) + '.tb:' + str(transpose_b), num_iters,
               duration, throughput))

    name_template = ('matmul_{device}_{dtype}_input_info_{inputinfo}')

    self.report_benchmark(
        name=name_template.format(
            device=device,
            dtype=str(dtype).replace(' ', ''),
            inputinfo=str(n) + 'x' + str(m) + 'x' + str(k) + ',ta:' +
            str(transpose_a) + ',tb:' + str(transpose_b)).replace(' ', ''),
        iters=num_iters,
        wall_time=duration)
    return duration

  def run_test_gpu(self, n, m, k, transpose_a, transpose_b, dtype, num_iters):
    # [FUNCTION DOC]
    # HELPER: convenience wrapper used by the sweep method to run one configuration on the GPU.
    # Forwards the sizes, transpose flags, dtype and iteration count to the graph runner using the
    # GPU device name from the test platform. Returns nothing (the duration is discarded); side
    # effects are those of the graph runner (printing and benchmark reporting).
    self.run_graph(test.gpu_device_name(), n, m, k, transpose_a, transpose_b,
                   num_iters, dtype)

  def test_round(self, num_iters):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name starts with test but it takes a required argument, and it is a benchmark
    # sweep rather than a unit test; a test runner that collects it directly would fail on the
    # missing argument. It is only meant to be called from the benchmark method.
    # Runs the full sweep for single and double precision floats: square-ish sizes (512, 1024 with
    # second dimension 1, 8, 16, 128, and k equal to n), then n of 200 with k of 10000, then two
    # explicit large shapes, each with no-transpose, transpose-A and transpose-B variants.
    # Input num_iters is the timed iteration count per configuration. Returns nothing; prints and
    # reports results through the GPU runner.
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
      for n, m, (transpose_a, transpose_b) in itertools.product(
          [512, 1024], [1, 8, 16, 128], [(False, False), (True, False),
                                         (False, True)]):
        k = n
        self.run_test_gpu(n, m, k, transpose_a, transpose_b, dtype, num_iters)

      for n, m, k, (transpose_a, transpose_b) in itertools.product(
          [200], [1, 8, 20], [10000], [(False, False), (True, False),
                                       (False, True)]):
        self.run_test_gpu(n, m, k, transpose_a, transpose_b, dtype, num_iters)

      for (n, m, k), (transpose_a, transpose_b) in itertools.product(
          [(200, 20, 20000), (1, 10000, 200)], [(False, False), (True, False),
                                                (False, True)]):
        self.run_test_gpu(n, m, k, transpose_a, transpose_b, dtype, num_iters)

  def benchmark_matmul(self):
    # [FUNCTION DOC]
    # Benchmark entry point discovered by the benchmark runner. Runs the whole sweep with 200 timed
    # iterations per configuration. Takes no inputs and returns nothing; results are printed and
    # reported by the underlying calls.
    self.test_round(num_iters=200)


if __name__ == '__main__':
  test.main()
