# [MODULE DOC]
# Benchmarks for the dataset filter transformation, built on the shared dataset benchmark base class.
# Runs via the test main entry point when executed.
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
"""Benchmarks for `tf.data.Dataset.filter()`."""
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import array_ops


class FilterBenchmark(benchmark_base.DatasetBenchmarkBase):
  # [CLASS DOC]
  # Benchmark suite for the filter transformation using a trivially true predicate in two forms.
  """Benchmarks for `tf.data.Dataset.filter()`."""

  def _benchmark(self, predicate, name, benchmark_id):
    # [FUNCTION DOC]
    # HELPER: shared body of the filter benchmarks.
    # Builds an endlessly repeated true-valued dataset filtered by predicate (callback), then
    # runs and reports one hundred thousand elements with a model name from benchmark_id and the
    # given name. Returns nothing.
    num_elements = 100000
    dataset = dataset_ops.Dataset.from_tensors(True)
    dataset = dataset.repeat().filter(predicate)
    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "filter.benchmark.%d" % benchmark_id,
            "parameters": "%d" % num_elements,
        },
        name=name)

  def benchmark_simple_function(self):
    # [FUNCTION DOC]
    # Benchmarks filtering with an identity op as the predicate.
    self._benchmark(array_ops.identity, "simple_function", benchmark_id=1)

  def benchmark_return_component_optimization(self):
    # [FUNCTION DOC]
    # Benchmarks filtering with a predicate that returns its input unchanged, which can be
    # short-circuited by the return-component optimization.
    self._benchmark(lambda x: x, "return_component", benchmark_id=2)


if __name__ == "__main__":
  benchmark_base.test.main()
