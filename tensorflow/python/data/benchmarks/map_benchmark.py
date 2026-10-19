# [MODULE DOC]
# Benchmarks for the dataset map transformation: chains of maps, fan-out, control flow and nested
# parallel maps. Built on the shared dataset benchmark base class.
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
"""Benchmarks for `tf.data.Dataset.map()`."""
import numpy as np

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import map_op
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import while_loop


class MapBenchmark(benchmark_base.DatasetBenchmarkBase):
  # [CLASS DOC]
  # Benchmark suite for the map transformation: chained maps, wide tuple fan-out, sequential and
  # parallel control flow inside the mapped function, and nested parallel maps in an interleave.
  """Benchmarks for `tf.data.Dataset.map()`."""

  def benchmark_chain_of_maps(self):
    # [FUNCTION DOC]
    # For chain lengths 0 to 50, benchmarks 10,000-element range datasets passed through that
    # many internal map datasets: with an add-one function using inter-op parallelism, the same
    # without it, and with an identity function (short-circuit). Reports each. Returns nothing.

    def benchmark_helper(chain_length, fn, use_inter_op_parallelism, label,
                         benchmark_id):
      # [FUNCTION DOC]
      # HELPER: builds a range dataset and applies chain_length internal map datasets with fn and
      # the inter-op parallelism flag, then runs and reports it with a name containing the
      # chain length and label, and a model name from benchmark_id. Returns nothing.
      dataset = dataset_ops.Dataset.range(10000)
      for _ in range(chain_length):
        dataset = map_op._MapDataset(  # pylint: disable=protected-access
            dataset, fn, use_inter_op_parallelism=use_inter_op_parallelism)
      self.run_and_report_benchmark(
          dataset,
          num_elements=10000,
          extras={
              "model_name": "map.benchmark.%d" % benchmark_id,
              "parameters": "%d" % chain_length,
          },
          name="chain_length_%d%s" % (chain_length, label))

    chain_lengths = [0, 1, 2, 5, 10, 20, 50]
    for chain_length in chain_lengths:
      benchmark_helper(
          chain_length=chain_length,
          fn=lambda x: x + 1,
          use_inter_op_parallelism=True,
          label="",
          benchmark_id=1)
      benchmark_helper(
          chain_length=chain_length,
          fn=lambda x: x + 1,
          use_inter_op_parallelism=False,
          label="_single_threaded",
          benchmark_id=2)
      benchmark_helper(
          chain_length=chain_length,
          fn=lambda x: x,
          use_inter_op_parallelism=True,
          label="_short_circuit",
          benchmark_id=3)

  def benchmark_map_fan_out(self):
    # [FUNCTION DOC]
    # For tuple widths 1 to 100, benchmarks a map over endless tuples of zeros with an add-one
    # function (with and without inter-op parallelism) and a pass-through function (short-circuit).
    # Reports each. Returns nothing.
    fan_outs = [1, 2, 5, 10, 20, 50, 100]

    def benchmark_helper(fan_out, fn, use_inter_op_parallelism, label,
                         benchmark_id):
      # [FUNCTION DOC]
      # HELPER: builds an endless dataset of tuples of fan_out zeros, applies one internal map
      # dataset with fn and the inter-op flag, and runs and reports it for 10,000 elements.
      # Returns nothing.
      dataset = dataset_ops.Dataset.from_tensors(
          tuple(0 for _ in range(fan_out))).repeat(None)
      dataset = map_op._MapDataset(  # pylint: disable=protected-access
          dataset, fn, use_inter_op_parallelism=use_inter_op_parallelism)
      self.run_and_report_benchmark(
          dataset,
          num_elements=10000,
          extras={
              "model_name": "map.benchmark.%d" % benchmark_id,
              "parameters": "%d" % fan_out,
          },
          name="fan_out_%d%s" % (fan_out, label))

    for fan_out in fan_outs:
      benchmark_helper(
          fan_out=fan_out,
          fn=lambda *xs: [x + 1 for x in xs],
          use_inter_op_parallelism=True,
          label="",
          benchmark_id=4)
      benchmark_helper(
          fan_out=fan_out,
          fn=lambda *xs: [x + 1 for x in xs],
          use_inter_op_parallelism=False,
          label="_single_threaded",
          benchmark_id=5)
      benchmark_helper(
          fan_out=fan_out,
          fn=lambda *xs: xs,
          use_inter_op_parallelism=True,
          label="_short_circuit",
          benchmark_id=6)

  def benchmark_sequential_control_flow(self):
    # [FUNCTION DOC]
    # Maps a single-element dataset through a sequential while loop that counts up to the element
    # value of 100000, with default optimizations enabled, and runs and reports one element.
    dataset = dataset_ops.Dataset.from_tensors(100000)

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: map function; runs a while loop incrementing a counter from zero while it is less
      # than x, and returns the loop's final variables.
      i = constant_op.constant(0)

      def body(i, x):
        # [FUNCTION DOC]
        # HELPER: loop body; returns the counter plus one and x unchanged.
        return math_ops.add(i, 1), x

      return while_loop.while_loop(math_ops.less, body, [i, x])

    num_elements = 1
    dataset = dataset.map(fn)
    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "map.benchmark.8",
            "parameters": "%d" % num_elements,
        },
        name="sequential_control_flow",
        apply_default_optimizations=True)

  def benchmark_parallel_control_flow(self):
    # [FUNCTION DOC]
    # Maps a single large random matrix (100 by ten million) through a function using a parallel
    # map over rows, with default optimizations enabled, and runs and reports one element.
    dataset = dataset_ops.Dataset.from_tensors(
        random_ops.random_uniform([100, 10000000]))

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: map function; applies a map over the rows of x (ten parallel iterations)
      # multiplying each row by its transpose, and returns the result.
      return map_fn.map_fn(
          lambda y: y * array_ops.transpose(y), x, parallel_iterations=10)

    num_elements = 1
    dataset = dataset.map(fn)
    self.run_and_report_benchmark(
        dataset,
        num_elements=1,
        extras={
            "model_name": "map.benchmark.9",
            "parameters": "%d" % num_elements,
        },
        name="parallel_control_flow",
        apply_default_optimizations=True)

  def _benchmark_nested_parallel_map(self, cycle_length, num_parallel_calls):
    # [FUNCTION DOC]
    # HELPER: shared body of the nested parallel map benchmark.
    # Builds a range of 2000 elements interleaved (autotuned parallel calls, given cycle_length,
    # possibly null for default) with inner datasets of 10 elements each mapped by a CPU-heavy
    # function with num_parallel_calls (null for sequential, integer, or autotune). Runs and
    # reports 20,000 elements with a name describing the map type and cycle length.
    k = 1024 * 1024
    num_map_elements = 10
    num_range_elements = 2000

    def g(_):
      # [FUNCTION DOC]
      # HELPER: map function; ignores its input and returns the sum of about fifty million random
      # numbers (CPU load).
      return np.random.rand(50 * k).sum()

    def f(_):
      # [FUNCTION DOC]
      # HELPER: interleave function; ignores its input and returns a range dataset of 10 elements
      # mapped by the heavy function with the chosen parallelism.
      return dataset_ops.Dataset.range(num_map_elements).map(
          g, num_parallel_calls=num_parallel_calls)

    dataset = dataset_ops.Dataset.range(num_range_elements)
    dataset = dataset.interleave(
        f, cycle_length=cycle_length, num_parallel_calls=dataset_ops.AUTOTUNE)

    cycle_length_str = ("default"
                        if cycle_length is None else str(cycle_length))
    num_parallel_calls_str = ("autotune"
                              if num_parallel_calls == dataset_ops.AUTOTUNE else
                              str(num_parallel_calls))
    map_dataset_str = ("map" if num_parallel_calls is None else
                       "parallel_map_num_parallel_calls_%s" %
                       num_parallel_calls_str)

    self.run_and_report_benchmark(
        dataset,
        num_elements=num_map_elements * num_range_elements,
        extras={
            "model_name": "map.benchmark.10",
            "parameters": "%s_%s" % (cycle_length_str, num_parallel_calls_str),
        },
        name=("%s_cycle_length_%s" % (map_dataset_str, cycle_length_str)))

  def benchmark_nested_parallel_map(self):
    # [FUNCTION DOC]
    # Runs the nested parallel map benchmark for cycle lengths of default and 100 crossed with
    # parallel call settings of none, 1, 10, 100 and autotune.
    cycle_lengths = [None, 100]
    nums_parallel_calls = [None, 1, 10, 100, dataset_ops.AUTOTUNE]
    for cycle_length in cycle_lengths:
      for num_parallel_calls in nums_parallel_calls:
        self._benchmark_nested_parallel_map(cycle_length, num_parallel_calls)


if __name__ == "__main__":
  benchmark_base.test.main()
