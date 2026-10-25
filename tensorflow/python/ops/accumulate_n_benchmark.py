# [MODULE DOC]
# Benchmark comparing the add-n operation against three hand-built accumulation strategies built
# from a temporary variable (initialized from the first input, from a merge of inputs, or from an
# explicit shape), across several input sizes, input counts and input orderings (replicated,
# unordered, ordered via a queue, reversed). Timings are printed to standard output as a table.
# Run as a test binary; no import-time side effects other than defining the benchmark class.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark for accumulate_n() in math_ops."""

import random
import time


from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test


class AccumulateNBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class. Builds graphs that sum lists of tensors with add-n and with three
  # temporary-variable accumulation variants, generates differently ordered inputs, runs each
  # op in a session many times and prints timings. Holds a row format template set by the
  # benchmark entry point.

  def _AccumulateNTemplate(self, inputs, init, shape, validate_shape):
    # [FUNCTION DOC]
    # HELPER: common graph builder for the accumulation variants.
    # Inputs: list of tensors to sum, init (initial value tensor), shape (shape of the temporary
    # variable) and validate_shape (whether the assignment checks shapes). Creates a temporary
    # variable, assigns init, adds every input with locking, and after all updates returns the op
    # that destroys the variable and yields the accumulated sum.
    var = gen_state_ops.temporary_variable(
        shape=shape, dtype=inputs[0].dtype.base_dtype)
    ref = state_ops.assign(var, init, validate_shape=validate_shape)
    update_ops = [
        state_ops.assign_add(
            ref, tensor, use_locking=True).op for tensor in inputs
    ]
    with ops.control_dependencies(update_ops):
      return gen_state_ops.destroy_temporary_variable(ref, var_name=var.op.name)

  def _AccumulateNInitializedWithFirst(self, inputs):
    # [FUNCTION DOC]
    # HELPER: accumulation variant initialized with zeros shaped like the first input, using the
    # first input's static shape with shape validation. Returns the accumulated tensor.
    return self._AccumulateNTemplate(
        inputs,
        init=array_ops.zeros_like(inputs[0]),
        shape=inputs[0].get_shape(),
        validate_shape=True)

  def _AccumulateNInitializedWithMerge(self, inputs):
    # [FUNCTION DOC]
    # HELPER: accumulation variant initialized with zeros shaped like the output of a merge of
    # all inputs (so no input is forced first), with a placeholder variable shape and shape
    # validation off. Returns the accumulated tensor.
    return self._AccumulateNTemplate(
        inputs,
        init=array_ops.zeros_like(gen_control_flow_ops.merge(inputs)[0]),
        shape=tensor_shape.TensorShape([0]),
        validate_shape=False)

  def _AccumulateNInitializedWithShape(self, inputs):
    # [FUNCTION DOC]
    # HELPER: accumulation variant initialized with an explicit zeros tensor built from the first
    # input's static shape and base dtype, with shape validation. Returns the accumulated tensor.
    return self._AccumulateNTemplate(
        inputs,
        init=array_ops.zeros(
            shape=inputs[0].get_shape(), dtype=inputs[0].dtype.base_dtype),
        shape=inputs[0].get_shape(),
        validate_shape=True)

  def _GenerateUnorderedInputs(self, size, n):
    # [FUNCTION DOC]
    # HELPER: input generator. Creates n independent random vectors of the given length and
    # returns them in a random order (shuffled in place using the standard random module).
    inputs = [random_ops.random_uniform(shape=[size]) for _ in range(n)]
    random.shuffle(inputs)
    return inputs

  def _GenerateReplicatedInputs(self, size, n):
    # [FUNCTION DOC]
    # HELPER: input generator. Creates one random vector and returns a list repeating that same
    # tensor n times.
    return n * self._GenerateUnorderedInputs(size, 1)

  def _GenerateOrderedInputs(self, size, n):
    # [FUNCTION DOC]
    # HELPER: input generator forcing a computation order. Starts with one random vector and
    # appends n minus one tensors, each computed from a hyperbolic tangent of one plus a value
    # dequeued from a single-slot queue after the previous tensor was enqueued, so each depends
    # on the one before. Returns the list in creation order.
    inputs = self._GenerateUnorderedInputs(size, 1)
    queue = data_flow_ops.FIFOQueue(
        capacity=1, dtypes=[inputs[0].dtype], shapes=[inputs[0].get_shape()])
    for _ in range(n - 1):
      op = queue.enqueue(inputs[-1])
      with ops.control_dependencies([op]):
        inputs.append(math_ops.tanh(1.0 + queue.dequeue()))
    return inputs

  def _GenerateReversedInputs(self, size, n):
    # [FUNCTION DOC]
    # HELPER: input generator. Builds the ordered inputs and returns them in reverse order.
    inputs = self._GenerateOrderedInputs(size, n)
    inputs.reverse()
    return inputs

  def _SetupAndRunBenchmark(self, graph, inputs, repeats, format_args):
    # [FUNCTION DOC]
    # HELPER: runs one measurement round. Inputs: graph holding the inputs, list of input tensors,
    # repeat count and a tuple of leading row values (tag, size, input count, repeats). In the
    # graph, builds add-n and the three accumulation variants. In a session on that graph, runs
    # each op one hundred times for warm up, then times the repeat count of runs, and prints a
    # formatted row with the method tag and elapsed seconds. Returns nothing.
    with graph.as_default():
      add_n = math_ops.add_n(inputs)
      acc_n_first = self._AccumulateNInitializedWithFirst(inputs)
      acc_n_merge = self._AccumulateNInitializedWithMerge(inputs)
      acc_n_shape = self._AccumulateNInitializedWithShape(inputs)

    test_ops = (("AddN", add_n.op),
                ("AccNFirst", acc_n_first.op),
                ("AccNMerge", acc_n_merge.op),
                ("AccNShape", acc_n_shape.op))

    with session.Session(graph=graph):
      for tag, op in test_ops:
        for _ in range(100):
          op.run()  # Run for warm up.
        start = time.time()
        for _ in range(repeats):
          op.run()
        duration = time.time() - start
        args = format_args + (tag, duration)
        print(self._template.format(*args))

  def _RunBenchmark(self, tag, input_fn, sizes, ninputs, repeats):
    # [FUNCTION DOC]
    # HELPER: sweeps a generator over all size and input-count combinations. For each pair it
    # creates a fresh graph, generates inputs with input_fn (called with size and count) inside
    # it, and runs the measurement round with the given repeat count. tag labels the rows.
    # Returns nothing; prints timing rows.
    for size in sizes:
      for ninput in ninputs:
        graph = ops.Graph()
        with graph.as_default():
          inputs = input_fn(size, ninput)

        format_args = (tag, size, ninput, repeats)
        self._SetupAndRunBenchmark(graph, inputs, repeats, format_args)

  def benchmarkAccumulateN(self):
    # [FUNCTION DOC]
    # Benchmark entry point. Sets the row template (six left-aligned columns of width fifteen),
    # prints a header and separator, then for each input ordering (replicated, unordered,
    # ordered, reversed) runs the sweep over sizes of 128 and 128 squared, input counts of 1, 10,
    # 100 and 300, with 100 repeats. Side effects: sets the template attribute, prints to
    # standard output, creates graphs and sessions.
    self._template = "{:<15}" * 6
    args = {
        "sizes": (128, 128**2),
        "ninputs": (1, 10, 100, 300),
        "repeats": 100
    }
    benchmarks = (("Replicated", self._GenerateReplicatedInputs),
                  ("Unordered", self._GenerateUnorderedInputs),
                  ("Ordered", self._GenerateOrderedInputs),
                  ("Reversed", self._GenerateReversedInputs))

    print(self._template.format("", "Size", "#Inputs", "#Repeat", "Method",
                                "Duration"))
    print("-" * 90)
    for benchmark in benchmarks:
      self._RunBenchmark(*benchmark, **args)


if __name__ == "__main__":
  test.main()
