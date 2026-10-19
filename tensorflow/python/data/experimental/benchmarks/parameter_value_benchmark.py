# [MODULE DOC]
# Benchmark module measuring how the parallelism and prefetch buffer size parameters affect
# tf.data pipelines (map, map-and-batch, interleave) when the consumer sleeps between elements and
# the map function sleeps a random time. Defines two module constants for sleep durations in
# milliseconds and two helper functions used as map functions. Not a library module; run directly it
# starts the benchmark runner.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks to compare effect of different parameter values on the performance."""
import time
import numpy as np

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops

# The maximum sleeping time for each output step and the input sleeping time in
# milliseconds.
max_output_sleep_ms = 0.5
input_sleep_ms = 1.5


def sleep_function(x):
  # [FUNCTION DOC]
  # HELPER: simulated work for the benchmarks. Sleeps for a random time drawn uniformly from zero
  # up to the maximum output sleep (given in milliseconds, converted to seconds), then returns x
  # unchanged. Uses numpy randomness; side effect is the sleep.
  time.sleep(np.random.uniform(max_output_sleep_ms) / 1000)
  return x


def map_function(x):
  # [FUNCTION DOC]
  # HELPER: dataset map function that wraps the sleeping function as a Python function op, keeping
  # the input tensor's data type. Takes a tensor x and returns the tensor produced by the op.
  return script_ops.py_func(sleep_function, [x], x.dtype)


class ParameterValueBenchmark(benchmark_base.DatasetBenchmarkBase):
  # [CLASS DOC]
  # Benchmark suite sweeping parallel-call counts and prefetch buffer sizes for three pipeline
  # shapes. Each public benchmark runs all combinations, records wall times in a mapping and prints
  # the combinations sorted from fastest to slowest. Stateless; uses the dataset benchmark base.
  """Benchmarks to compare effect of different parameter values on the performance."""

  def _benchmark_map(self, num_parallel_calls, buffer_size):
    # [FUNCTION DOC]
    # HELPER: times one map configuration. Builds an endless dataset of a large random matrix pair,
    # maps matrix multiply with num_parallel_calls (int) parallelism, maps the sleeping function,
    # prefetches buffer_size (int) elements, and applies a consumer-side sleep of the input sleep
    # time. Runs and reports 10000 elements under a name encoding the parameters. Returns the
    # measured wall time.
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(map_function)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    dataset = dataset.apply(testing.sleep(int(input_sleep_ms * 1000)))

    name_str = ("map_max_output_sleep_ms_%.2f_input_sleep_ms_%.2f_"
                "num_parallel_calls_%d_buffer_size_%d")
    return self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=10000,
        name=name_str %
        (max_output_sleep_ms, input_sleep_ms, num_parallel_calls, buffer_size))

  def benchmark_map(self):
    # [FUNCTION DOC]
    # Benchmark entry: tries parallel-call counts 4, 8, 12 against buffer sizes 10 to 300, times
    # each combination with the map helper, then prints each combination with its wall time, fastest
    # first. Returns nothing.
    nums_parallel_calls = [4, 8, 12]
    buffer_sizes = [10, 50, 100, 150, 200, 250, 300]

    parameters_list = []
    wall_time_map = {}

    for num_parallel_calls in nums_parallel_calls:
      for buffer_size in buffer_sizes:
        parameters = (num_parallel_calls, buffer_size)
        parameters_list.append(parameters)
        wall_time = self._benchmark_map(num_parallel_calls, buffer_size)
        wall_time_map[parameters] = wall_time

    parameters_list.sort(key=lambda x: wall_time_map[x])
    for parameters in parameters_list:
      print("num_parallel_calls_%d_buffer_size_%d_wall_time:" % parameters,
            wall_time_map[parameters])

  def _benchmark_map_and_batch(self, num_parallel_calls, buffer_size):
    # [FUNCTION DOC]
    # HELPER: times one map-and-batch configuration. Same as the map helper but additionally batches
    # 16 elements after the matrix-multiply map and before the sleeping map; reports 1000 elements.
    # num_parallel_calls (int) and buffer_size (int) are the swept parameters. Returns wall time.
    batch_size = 16
    k = 1024 * 1024

    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(map_function)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    dataset = dataset.apply(testing.sleep(int(input_sleep_ms * 1000)))

    name_str = ("map_and_batch_max_output_sleep_ms_%.2f_input_sleep_ms_%.2f"
                "_num_parallel_calls_%d_buffer_size_%d")
    return self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=1000,
        name=name_str %
        (max_output_sleep_ms, input_sleep_ms, num_parallel_calls, buffer_size))

  def benchmark_map_and_batch(self):
    # [FUNCTION DOC]
    # Benchmark entry: sweeps parallel-call counts 4, 8, 12 and buffer sizes 10 to 300 with the
    # map-and-batch helper and prints each combination with its wall time, fastest first. Returns
    # nothing.
    nums_parallel_calls = [4, 8, 12]
    buffer_sizes = [10, 50, 100, 150, 200, 250, 300]

    parameters_list = []
    wall_time_map = {}

    for num_parallel_calls in nums_parallel_calls:
      for buffer_size in buffer_sizes:
        parameters = (num_parallel_calls, buffer_size)
        parameters_list.append(parameters)
        wall_time = self._benchmark_map_and_batch(num_parallel_calls,
                                                  buffer_size)
        wall_time_map[parameters] = wall_time

    parameters_list.sort(key=lambda x: wall_time_map[x])
    for parameters in parameters_list:
      print("num_parallel_calls_%d_buffer_size_%d_wall_time:" % parameters,
            wall_time_map[parameters])

  def _benchmark_interleave(self, num_parallel_calls, buffer_size):
    # [FUNCTION DOC]
    # HELPER: times one interleave configuration. Inner dataset: endless large random matrix pair,
    # matrix-multiply map, sleeping map. Outer: endless range interleaved onto the inner dataset
    # with cycle length 10 and num_parallel_calls (int) parallelism, then prefetch of buffer_size
    # (int) and a consumer-side sleep. Reports 10000 elements. Returns wall time.
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(math_ops.matmul)
    dataset = dataset.map(map_function)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,  # pylint: disable=cell-var-from-loop
        cycle_length=10,
        num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    dataset = dataset.apply(testing.sleep(int(input_sleep_ms * 1000)))

    name_str = ("interleave_max_output_sleep_ms_%.2f_input_sleep_ms_%.2f"
                "_num_parallel_calls_%d_buffer_size_%d")
    return self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=10000,
        name=name_str %
        (max_output_sleep_ms, input_sleep_ms, num_parallel_calls, buffer_size))

  def benchmark_interleave(self):
    # [FUNCTION DOC]
    # Benchmark entry: sweeps parallel-call counts 4, 8, 10 and buffer sizes 10 to 300 with the
    # interleave helper and prints each combination with its wall time, fastest first. Returns
    # nothing.
    nums_parallel_calls = [4, 8, 10]
    buffer_sizes = [10, 50, 100, 150, 200, 250, 300]

    parameters_list = []
    wall_time_map = {}

    for num_parallel_calls in nums_parallel_calls:
      for buffer_size in buffer_sizes:
        parameters = (num_parallel_calls, buffer_size)
        parameters_list.append(parameters)
        wall_time = self._benchmark_interleave(num_parallel_calls, buffer_size)
        wall_time_map[parameters] = wall_time

    parameters_list.sort(key=lambda x: wall_time_map[x])
    for parameters in parameters_list:
      print("num_parallel_calls_%d_buffer_size_%d_wall_time:" % parameters,
            wall_time_map[parameters])


if __name__ == "__main__":
  benchmark_base.test.main()
