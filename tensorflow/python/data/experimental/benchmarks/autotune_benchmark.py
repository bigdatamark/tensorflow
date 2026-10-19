# [MODULE DOC]
# Benchmark module comparing tf.data pipelines with and without autotuning of parallelism knobs.
# Defines one benchmark class covering batch, map, map+batch, interleave and combined pipelines built
# from large random matrices multiplied by a matrix-multiply map function. Not a library module:
# run directly it starts the benchmark test runner. No import-time side effects besides imports.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for autotuning performance knobs."""

import numpy as np

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.ops import math_ops


class AutotuneBenchmark(benchmark_base.DatasetBenchmarkBase):
  # [CLASS DOC]
  # Benchmark suite measuring the speedup of autotuned parallelism over no autotuning for several
  # pipeline shapes. Each public benchmark method runs its pipeline twice (autotune off with id 1,
  # on with id 2) and prints the ratio of wall times. Inherits the reporting machinery from the
  # dataset benchmark base class. Holds no instance state.
  """Benchmarks for autotuning performance knobs."""

  def _run_benchmark(self, dataset, autotune, benchmark_iters, benchmark_label,
                     benchmark_id):
    # [FUNCTION DOC]
    # HELPER: shared runner for all autotune benchmarks in this class.
    # Applies options to the given dataset (default optimizations off, autotune enabled or not
    # according to the autotune flag), then runs and reports a benchmark with one timed iteration
    # after a warmup, over benchmark_iters elements. The reported name gets a suffix when autotune is
    # on; extras carry a model name built from the label and id and the autotune flag as parameter.
    # Returns the measured wall time (a number).
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.autotune.enabled = autotune
    dataset = dataset.with_options(options)

    autotune_string = "_autotune_parallelism_only"
    wall_time = self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=benchmark_iters,
        warmup=True,
        iters=1,
        extras={
            "model_name":
                "autotune.benchmark.%s.%d" % (benchmark_label, benchmark_id),
            "parameters":
                "%s" % autotune,
        },
        name=benchmark_label + (autotune_string if autotune else ""))
    return wall_time

  def benchmark_batch(self):
    # [FUNCTION DOC]
    # Benchmark entry: runs the batch pipeline without then with autotuning and prints the speedup
    # (time without divided by time with). Takes no arguments, returns nothing.
    a = self._benchmark_batch(autotune=False, benchmark_id=1)
    b = self._benchmark_batch(autotune=True, benchmark_id=2)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))

  def _benchmark_batch(self, autotune, benchmark_id):
    # [FUNCTION DOC]
    # HELPER: builds and runs the batch benchmark for benchmark_batch.
    # Builds an endless dataset of one random row/column matrix pair, maps matrix multiply, then
    # batches 128 elements with autotuned parallel calls. autotune (bool) toggles autotuning in the
    # run; benchmark_id (int) identifies the run in reports. Uses 10000 iterations. Returns wall time.
    batch_size = 128
    k = 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(math_ops.matmul)
    dataset = dataset.batch(
        batch_size=batch_size, num_parallel_calls=dataset_ops.AUTOTUNE)
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        benchmark_iters=10000,
        benchmark_label="batch",
        benchmark_id=benchmark_id)

  def benchmark_map(self):
    # [FUNCTION DOC]
    # Benchmark entry: runs the map pipeline without then with autotuning and prints the speedup
    # (time without divided by time with). Takes no arguments, returns nothing.
    a = self._benchmark_map(autotune=False, benchmark_id=1)
    b = self._benchmark_map(autotune=True, benchmark_id=2)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))

  def _benchmark_map(self, autotune, benchmark_id):
    # [FUNCTION DOC]
    # HELPER: builds and runs the map benchmark for benchmark_map.
    # Endless dataset of one large random matrix pair mapped through matrix multiply with autotuned
    # parallel calls; 10000 iterations. autotune (bool) toggles autotuning; benchmark_id (int)
    # labels the run. Returns wall time.
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        benchmark_iters=10000,
        benchmark_label="map",
        benchmark_id=benchmark_id)

  def benchmark_map_and_batch(self):
    # [FUNCTION DOC]
    # Benchmark entry: runs the map-then-batch pipeline without then with autotuning and prints the
    # speedup (time without divided by time with). Takes no arguments, returns nothing.
    a = self._benchmark_map_and_batch(autotune=False, benchmark_id=1)
    b = self._benchmark_map_and_batch(autotune=True, benchmark_id=2)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))

  def _benchmark_map_and_batch(self, autotune, benchmark_id):
    # [FUNCTION DOC]
    # HELPER: builds and runs the map-and-batch benchmark for benchmark_map_and_batch.
    # Endless dataset of a large random matrix pair, mapped through matrix multiply with autotuned
    # parallel calls, then batched by 16 (batch not parallel); 1000 iterations. autotune (bool)
    # toggles autotuning; benchmark_id (int) labels the run. Returns wall time.
    batch_size = 16
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        benchmark_iters=1000,
        benchmark_label="map_and_batch",
        benchmark_id=benchmark_id)

  def benchmark_interleave(self):
    # [FUNCTION DOC]
    # Benchmark entry: runs the interleave pipeline without then with autotuning and prints the
    # speedup (time without divided by time with). Takes no arguments, returns nothing.
    a = self._benchmark_interleave(autotune=False, benchmark_id=1)
    b = self._benchmark_interleave(autotune=True, benchmark_id=2)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))

  def _benchmark_interleave(self, autotune, benchmark_id):
    # [FUNCTION DOC]
    # HELPER: builds and runs the interleave benchmark for benchmark_interleave.
    # Inner dataset: endless large random matrix pair mapped through matrix multiply (sequential).
    # Outer: an endless range interleaved to that inner dataset with cycle length 10 and autotuned
    # parallel calls; 10000 iterations. autotune (bool) toggles autotuning; benchmark_id (int)
    # labels the run. Returns wall time.
    k = 1024 * 1024
    dataset = dataset_ops.Dataset.from_tensors(
        (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
    dataset = dataset.map(math_ops.matmul)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        benchmark_iters=10000,
        benchmark_label="interleave",
        benchmark_id=benchmark_id)

  def benchmark_map_and_interleave(self):
    # [FUNCTION DOC]
    # Benchmark entry: runs the nested map-and-interleave pipeline without then with autotuning and
    # prints the speedup (time without divided by time with). Takes no arguments, returns nothing.
    a = self._benchmark_map_and_interleave(autotune=False, benchmark_id=1)
    b = self._benchmark_map_and_interleave(autotune=True, benchmark_id=2)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))

  def _benchmark_map_and_interleave(self, autotune, benchmark_id):
    # [FUNCTION DOC]
    # HELPER: builds and runs the map-and-interleave benchmark for benchmark_map_and_interleave.
    # Creates three endless datasets from random matrix pairs of decreasing size, then chains: map
    # and interleave on the first, zip with the second and map, interleave again, zip with the third
    # and map; all with autotuned parallelism; 10000 iterations. autotune (bool) toggles autotuning;
    # benchmark_id (int) labels the run. Returns wall time.
    k = 1024 * 1024
    a = (np.random.rand(1, 8 * k), np.random.rand(8 * k, 1))
    b = (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))
    c = (np.random.rand(1, 2 * k), np.random.rand(2 * k, 1))
    dataset_a = dataset_ops.Dataset.from_tensors(a).repeat()
    dataset_b = dataset_ops.Dataset.from_tensors(b).repeat()
    dataset_c = dataset_ops.Dataset.from_tensors(c).repeat()

    def f1(x, y):
      # [FUNCTION DOC]
      # HELPER: map function for the first stage; multiplies its two matrix inputs and returns the
      # product.
      return math_ops.matmul(x, y)

    def f2(a, b):
      # [FUNCTION DOC]
      # HELPER: map function for zipped stages; takes the previous element a and a pair b of
      # matrices, multiplies the pair and returns a tuple of a unchanged and the product.
      x, y = b
      return a, math_ops.matmul(x, y)

    dataset = dataset_a
    dataset = dataset.map(f1, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=dataset_ops.AUTOTUNE,
        cycle_length=2)

    dataset = dataset_ops.Dataset.zip((dataset, dataset_b))
    dataset = dataset.map(f2, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=dataset_ops.AUTOTUNE,
        cycle_length=2)

    dataset = dataset_ops.Dataset.zip((dataset, dataset_c))
    dataset = dataset.map(f2, num_parallel_calls=dataset_ops.AUTOTUNE)
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        benchmark_iters=10000,
        benchmark_label="map_and_interleave",
        benchmark_id=benchmark_id)

  def benchmark_map_batch_and_interleave(self):
    # [FUNCTION DOC]
    # Benchmark entry: runs the combined map, batch and interleave pipeline without then with
    # autotuning and prints the speedup (time without divided by time with). No arguments, returns
    # nothing.
    a = self._benchmark_map_batch_and_interleave(autotune=False, benchmark_id=1)
    b = self._benchmark_map_batch_and_interleave(autotune=True, benchmark_id=2)
    print("autotune parallelism vs no autotuning speedup: {}".format(a / b))

  def _benchmark_map_batch_and_interleave(self, autotune, benchmark_id):
    # [FUNCTION DOC]
    # HELPER: builds and runs the map/batch/interleave benchmark for the matching entry method.
    # Three endless datasets from random matrix pairs; the first is mapped (matrix multiply),
    # batched by 16 and interleaved, zipped with the second and interleaved again, then zipped with
    # the third which is mapped and batched by 16. All with autotuned parallelism; 1000 iterations.
    # autotune (bool) toggles autotuning; benchmark_id (int) labels the run. Returns wall time.
    batch_size = 16
    k = 1024 * 1024
    a = (np.random.rand(1, 8 * k), np.random.rand(8 * k, 1))
    b = (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))
    c = (np.random.rand(1, 2 * k), np.random.rand(2 * k, 1))
    dataset_a = dataset_ops.Dataset.from_tensors(a).repeat()
    dataset_b = dataset_ops.Dataset.from_tensors(b).repeat()
    dataset_c = dataset_ops.Dataset.from_tensors(c).repeat()

    dataset = dataset_a
    dataset = dataset.map(
        math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=dataset_ops.AUTOTUNE,
        cycle_length=2)

    dataset = dataset_ops.Dataset.zip((dataset, dataset_b))
    dataset = dataset_ops.Dataset.range(1).repeat().interleave(
        lambda _: dataset,
        num_parallel_calls=dataset_ops.AUTOTUNE,
        cycle_length=2)

    dataset_c = dataset_c.map(
        math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset_c = dataset_c.batch(batch_size=batch_size)
    dataset = dataset_ops.Dataset.zip((dataset, dataset_c))
    return self._run_benchmark(
        dataset=dataset,
        autotune=autotune,
        benchmark_iters=1000,
        benchmark_label="map_and_batch_and_interleave",
        benchmark_id=benchmark_id)


if __name__ == "__main__":
  benchmark_base.test.main()
