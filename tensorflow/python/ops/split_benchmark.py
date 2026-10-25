# [MODULE DOC]
# Benchmark for the split operation. Builds graphs with 100 repeated splits of a zero tensor
# for many shapes, with fixed or randomly varying piece sizes, and times them in a session
# with graph optimizations turned off. Not a library: run directly through the test main
# entry point. No import-time side effects beyond imports.
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
"""Benchmark for split and grad of split."""

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


def build_graph(device, input_shape, output_sizes, axis):
  # [FUNCTION DOC]
  # HELPER: graph builder for the split benchmark.
  # KNOWN ISSUE: the docstring says it returns an array of tensors, but it returns a single
  # grouped no-output operation that depends on all split outputs.
  # Under the given device (index 0) creates a zeros tensor of input_shape, then repeats 100
  # times a split along axis into pieces of output_sizes (list of sizes or a count), collecting
  # all outputs. Returns the grouping op over all of them, so running it executes every split.
  """Build a graph containing a sequence of split operations.

  Args:
    device: string, the device to run on.
    input_shape: shape of the input tensor.
    output_sizes: size of each output along axis.
    axis: axis to be split along.

  Returns:
    An array of tensors to run()
  """
  with ops.device("/%s:0" % device):
    inp = array_ops.zeros(input_shape)

    outputs = []
    for _ in range(100):
      outputs.extend(array_ops.split(inp, output_sizes, axis))
    return control_flow_ops.group(*outputs)


class SplitBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class measuring forward split performance on many shapes. It has a private helper
  # that builds and times one graph, and a public benchmark method that sweeps configurations.
  # Holds no state of its own; created and run by the benchmark framework.
  """Benchmark split!"""

  def _run_graph(self, device, output_shape, variable, num_outputs, axis):
    # [FUNCTION DOC]
    # HELPER: runs one benchmark configuration for benchmark_split.
    # KNOWN ISSUE: the docstring says it returns the run duration in seconds, but it returns
    # nothing; results are only reported through the benchmark harness.
    # Builds the input shape and split sizes from output_shape, number of outputs and axis. With
    # variable false all sizes equal the output extent along the axis; with variable true sizes
    # are random integers near that extent, so input size is their sum. Builds a split graph on
    # the device, then runs it in a session with graph optimization disabled (level L0), sets
    # logging verbosity to info, initializes global variables, and records a benchmark with
    # throughput estimated from the input size plus the shape, variable flag and axis as extras.
    # Side effects: global logging verbosity changed, random numbers consumed, benchmark output.
    """Run the graph and print its execution time.

    Args:
      device: string, the device to run on.
      output_shape: shape of each output tensors.
      variable: whether or not the output shape should be fixed
      num_outputs: the number of outputs to split the input into
      axis: axis to be split

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      if not variable:
        if axis == 0:
          input_shape = [output_shape[0] * num_outputs, output_shape[1]]
          sizes = [output_shape[0] for _ in range(num_outputs)]
        else:
          input_shape = [output_shape[0], output_shape[1] * num_outputs]
          sizes = [output_shape[1] for _ in range(num_outputs)]
      else:
        sizes = np.random.randint(
            low=max(1, output_shape[axis] - 2),
            high=output_shape[axis] + 2,
            size=num_outputs)
        total_size = np.sum(sizes)
        if axis == 0:
          input_shape = [total_size, output_shape[1]]
        else:
          input_shape = [output_shape[0], total_size]

      outputs = build_graph(device, input_shape, sizes, axis)
    config = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(
        optimizer_options=config_pb2.OptimizerOptions(
            opt_level=config_pb2.OptimizerOptions.L0)))
    with session_lib.Session(graph=graph, config=config) as session:
      logging.set_verbosity("info")
      variables.global_variables_initializer().run()
      bench = benchmark.TensorFlowBenchmark()
      bench.run_op_benchmark(
          session,
          outputs,
          mbs=input_shape[0] * input_shape[1] * 4 * 2 * 100 / 1e6,
          extras={
              "input_shape": input_shape,
              "variable": variable,
              "axis": axis
          })

  def benchmark_split(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the device is hard-coded to gpu, so it fails or is meaningless without a GPU.
    # Benchmark entry point (takes no arguments, returns nothing). Sweeps a fixed list of nine 2-D
    # shapes, split along axis 1 only (axis 0 is skipped because it needs no copying), into 100
    # outputs, each with fixed and then variable output sizes. For every combination it calls the
    # graph runner on the "gpu" device. Prints a heading line and, via the runner, benchmark
    # results to the logs.
    print("Forward vs backward concat")
    shapes = [[2000, 8], [8, 2000], [100, 18], [1000, 18], [10000, 18],
              [100, 97], [1000, 97], [10000, 1], [1, 10000]]
    axis_ = [1]  # 0 is very fast because it doesn't actually do any copying
    num_outputs = 100
    variable = [False, True]  # fixed input size or not
    for shape in shapes:
      for axis in axis_:
        for v in variable:
          self._run_graph("gpu", shape, v, num_outputs, axis)


if __name__ == "__main__":
  test.main()
