# [MODULE DOC]
# Benchmarks for the dataset batch transformation (sparse, dense and parallel variants), built on
# the shared dataset benchmark base class. Runs via the test main entry point when executed.
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
"""Benchmarks for `tf.data.Dataset.batch()`."""
import numpy as np

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import random_ops


class BatchBenchmark(benchmark_base.DatasetBenchmarkBase):
  # [CLASS DOC]
  # Benchmark suite for the batch transformation: sparse elements, dense elements with optional
  # parallel copy, and parallel batching. Inherits the run-and-report machinery from the dataset
  # benchmark base class.
  """Benchmarks for `tf.data.Dataset.batch()`."""

  def benchmark_batch_sparse(self):
    # [FUNCTION DOC]
    # Benchmarks batching of repeated sparse tensors of dense length one thousand, sweeping the
    # number of non-zero entries per row (0 to 100) and the batch size (1 to 1024). Each
    # combination runs once for one hundred thousand divided by batch size elements and reports
    # extras with a model name and a parameter string. Returns nothing.
    non_zeros_per_row_values = [0, 1, 5, 10, 100]
    batch_size_values = [1, 32, 64, 128, 1024]

    for non_zeros_per_row in non_zeros_per_row_values:

      tensor = sparse_tensor.SparseTensor(
          indices=np.arange(non_zeros_per_row, dtype=np.int64)[:, np.newaxis],
          values=np.arange(non_zeros_per_row, dtype=np.int64),
          dense_shape=[1000])

      for batch_size in batch_size_values:
        dataset = dataset_ops.Dataset.from_tensors(tensor).repeat().batch(
            batch_size)
        self.run_and_report_benchmark(
            dataset,
            num_elements=100000 // batch_size,
            iters=1,
            extras={
                "model_name": "batch.benchmark.1",
                "parameters": "%d.%d" % (batch_size, non_zeros_per_row),
            },
            name="sparse_num_elements_%d_batch_size_%d" %
            (non_zeros_per_row, batch_size))

  def _benchmark_batch_dense(self, parallel_copy, benchmark_id):
    # [FUNCTION DOC]
    # HELPER: shared body of the dense batch benchmark.
    # Sweeps element sizes of two to the power 10 to 22 (step two) and batch sizes of 8, 64, 512 for
    # repeated random float vectors. Sets the parallel batch optimization option from
    # parallel_copy (boolean), runs and reports each combination once, with a model name built from
    # benchmark_id (integer) and a name carrying a parallel-copy tag when enabled. Returns nothing.
    for element_exp in [10, 12, 14, 16, 18, 20, 22]:
      for batch_exp in [3, 6, 9]:
        element_size = 1 << element_exp
        batch_size = 1 << batch_exp
        dataset = dataset_ops.Dataset.from_tensors(
            np.random.rand(element_size)).repeat().batch(batch_size)
        options = options_lib.Options()
        options.experimental_optimization.parallel_batch = parallel_copy
        dataset = dataset.with_options(options)
        tag = "_parallel_copy" if parallel_copy else ""
        self.run_and_report_benchmark(
            dataset,
            num_elements=(1 << (22 - ((batch_exp + element_exp) // 2))),
            iters=1,
            extras={
                "model_name": "batch.benchmark.%d" % benchmark_id,
                "parameters": "%d.%d" % (batch_size, element_size),
            },
            name="batch_element_size_%d_batch_size_%d%s" %
            (element_size, batch_size, tag))

  def benchmark_batch_dense(self):
    # [FUNCTION DOC]
    # Runs the dense batch benchmark without parallel copy (benchmark id two). The parallel copy
    # variant (id three) is commented out and therefore not run.
    self._benchmark_batch_dense(parallel_copy=False, benchmark_id=2)
    #self._benchmark_batch_dense(parallel_copy=True, benchmark_id=3)

  def benchmark_parallel_batch(self):
    # [FUNCTION DOC]
    # Benchmarks batching of 128 images (224 by 224 by 3 random values produced by a map over a
    # range of one hundred thousand) with no parallelism, 1, 4, 16 and autotuned parallel calls.
    # Each configuration is run and reported once. Returns nothing.
    batch_size = 128
    nums_parallel_calls = [None, 1, 4, 16, dataset_ops.AUTOTUNE]
    num_range = 100000

    def f(_):
      # [FUNCTION DOC]
      # HELPER: map function; ignores its input and returns a random uniform tensor of shape
      # 224 by 224 by 3.
      return random_ops.random_uniform([224, 224, 3])

    for num_parallel_calls in nums_parallel_calls:
      num_parallel_calls_str = ("autotune"
                                if num_parallel_calls == dataset_ops.AUTOTUNE
                                else str(num_parallel_calls))
      op_str = ("batch" if num_parallel_calls is None else
                ("parallel_batch_num_parallel_calls_%s" %
                 num_parallel_calls_str))

      dataset = dataset_ops.Dataset.range(num_range).map(f).batch(
          batch_size, num_parallel_calls=num_parallel_calls)
      self.run_and_report_benchmark(
          dataset,
          num_elements=num_range // batch_size,
          iters=1,
          extras={
              "model_name": "batch.benchmark.4",
              "parameters": "%d.%s" % (batch_size, num_parallel_calls_str),
          },
          name="batch_size_%d_%s" % (batch_size, op_str))


if __name__ == "__main__":
  benchmark_base.test.main()
