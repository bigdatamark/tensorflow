# [MODULE DOC]
# KNOWN ISSUE: the module docstring says "split and grad of split", but the code benchmarks the
# concatenation operation and its gradient.
# Benchmark for concatenating many tensors and (optionally) differentiating the concatenation.
# Provides a graph builder and a benchmark class that runs a grid of shapes and axes on the CPU,
# prints timings and reports them to the benchmark framework. Running as a script starts the
# test runner. No import-time side effects.
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

import itertools
import random
import time

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def build_graph(device, input_shape, variable, num_inputs, axis, grad):
  # [FUNCTION DOC]
  # HELPER: graph builder for the concat benchmark.
  # Inputs: device (device type name such as cpu; zero-th device of that type is used),
  # input_shape (two-element shape), variable (when true, the size along the non-concat
  # dimension is randomized within five of the base size, using Python's random generator),
  # num_inputs (count of zero tensors), axis (zero or one; only the value one is treated
  # specially), grad (also compute gradients of each output with respect to the inputs).
  # Creates the zero input tensors and one hundred concat ops, then returns a single grouping
  # op depending on either all gradients (grad true) or all concat outputs. Adds ops to the
  # current default graph.
  """Build a graph containing a sequence of concat operations.

  Args:
    device: string, the device to run on.
    input_shape: shape of the input tensors.
    variable: whether or not to randomize the input shape
    num_inputs: the number of inputs to concat
    axis: axis to be concat'ed
    grad: if True compute the gradient

  Returns:
    An array of tensors to run()
  """
  with ops.device("/%s:0" % device):
    if not variable:
      inputs = [array_ops.zeros(input_shape) for _ in range(num_inputs)]
    else:
      if axis == 1:
        inputs = [
            array_ops.zeros([
                input_shape[0],
                random.randint(max(1, input_shape[1] - 5), input_shape[1] + 5)
            ]) for _ in range(num_inputs)
        ]
      else:
        inputs = [
            array_ops.zeros([
                random.randint(max(1, input_shape[0] - 5), input_shape[0] + 5),
                input_shape[1]
            ]) for _ in range(num_inputs)
        ]

    outputs = [array_ops.concat(inputs, axis) for _ in range(100)]
    if grad:
      return control_flow_ops.group(*list(
          itertools.chain.from_iterable([
              gradients_impl.gradients(output, inputs) for output in outputs
          ])))
    else:
      return control_flow_ops.group(*outputs)


class ConcatBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class for concatenation. Holds no state; a helper builds and times a graph for
  # one configuration and the benchmark method sweeps configurations.
  """Benchmark concat."""

  def _run_graph(self, device, input_shape, variable, num_inputs, axis, grad,
                 num_iters):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says variable means the input shape should be fixed, but in
    # the graph builder true means the shape is randomized. Also, the name template used for
    # the report has no placeholder for the iteration count, so the iterations value passed to
    # formatting is ignored. The returned duration is the total over all timed iterations, not
    # per iteration.
    # Inputs: device, input_shape, variable, num_inputs, axis, grad (see the graph builder) and
    # num_iters (number of timed runs). Builds the graph in a fresh graph object, creates a
    # session with the optimizer level set to zero, runs once to warm up, then times the
    # iterations. Prints a summary line with the per-iteration time and a derived bandwidth
    # figure, reports the benchmark under a generated name, and returns the total seconds.
    """Run the graph and print its execution time.

    Args:
      device: string, the device to run on.
      input_shape: shape of the input tensors.
      variable: whether or not the input shape should be fixed
      num_inputs: the number of inputs to concat
      axis: axis to be concat'ed
      grad: if True compute the gradient
      num_iters: number of steps to run.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      outputs = build_graph(device, input_shape, variable, num_inputs, axis,
                            grad)
    config = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(
        optimizer_options=config_pb2.OptimizerOptions(
            opt_level=config_pb2.OptimizerOptions.L0)))
    with session_lib.Session(graph=graph, config=config) as session:
      variables.global_variables_initializer().run()
      _ = session.run(outputs)  # warm up.
      start_time = time.time()
      for _ in range(num_iters):
        _ = session.run(outputs)
      duration = time.time() - start_time
      print("%s shape:%d/%d var: %r #inputs:%d axis:%d grad:%r - %f secs - %f "
            "GB/sec" % (device, input_shape[0], input_shape[1], variable,
                        num_inputs, axis, grad, duration / num_iters,
                        num_inputs * input_shape[0] * input_shape[1] * 4 * 2 *
                        100 / (duration / num_iters) / 1e9))

    name_template = (
        "concat_bench_{device}_input_shape_{shape}_variable_{variable}"
        "_num_inputs_{num_inputs}_axis_{axis}_grad_{grad}")

    self.report_benchmark(name=name_template.format(
        device=device,
        num_inputs=num_inputs,
        variable=variable,
        grad=grad,
        shape=str(input_shape).replace(" ", ""),
        axis=str(axis),
        iters=num_iters))

    return duration

  def benchmark_concat(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the printed title says "Forward vs backward concat", but every run passes the
    # gradient flag as true, so only the forward-plus-backward case is measured.
    # Takes no arguments, returns nothing. For eight shapes, both concat axes and fixed or
    # randomized shapes, runs the helper on the CPU with twenty inputs and ten iterations each.
    # Prints progress and reports each benchmark result.
    print("Forward vs backward concat")
    shapes = [[2000, 8], [8, 2000], [100, 18], [1000, 18], [100, 97],
              [1000, 97], [10000, 1], [1, 10000]]
    axis_ = [0, 1]
    num_inputs = 20
    num_iters = [10] * len(shapes)
    variable = [False, True]  # fixed input size or not
    for shape, iters in zip(shapes, num_iters):
      for axis in axis_:
        for v in variable:
          self._run_graph("cpu", shape, v, num_inputs, axis, True, iters)


if __name__ == "__main__":
  test.main()
