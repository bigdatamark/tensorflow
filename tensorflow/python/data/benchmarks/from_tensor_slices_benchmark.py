# [MODULE DOC]
# Benchmarks for creating datasets from tensor slices (batching, reshaping, sparse tensors, caching).
# Also defines a helper dataset class that runs a flat-map function on a single-threaded executor.
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
"""Benchmarks for `tf.data.Dataset.from_tensor_slices()`."""
import numpy as np

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import get_single_element
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.eager import def_function
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_dataset_ops


class SingleThreadedFlatMapDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # HELPER: dataset used by the sparse slicing benchmark.
  # Unary dataset that applies a function to each input element and flattens the resulting
  # datasets, like the standard flat-map, but with the function wrapped to run on a single-threaded
  # executor. Holds the input dataset, the wrapped function and the element structure.
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func):
    # [FUNCTION DOC]
    # Wraps map_func (callback from an element to a dataset) as a structured function using the
    # single-threaded executor, records the output element structure, creates the native flat-map
    # dataset variant from the input dataset's variant and the function, and initializes the base
    # unary dataset with it. Returns nothing.
    """See `Dataset.flat_map()` for details."""
    self._input_dataset = input_dataset
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        defun_kwargs={"_executor": "SINGLE_THREADED_EXECUTOR"})
    self._structure = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    variant_tensor = gen_dataset_ops.flat_map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        **self._flat_structure)
    super(SingleThreadedFlatMapDataset, self).__init__(input_dataset,
                                                       variant_tensor)

  def _functions(self):
    # [FUNCTION DOC]
    # HELPER: dataset internals hook. Returns a list holding the wrapped map function.
    return [self._map_func]

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element structure (type specification) of the flattened output.
    return self._structure

  def _transformation_name(self):
    # [FUNCTION DOC]
    # HELPER: dataset internals hook. Returns the fixed name string of this transformation.
    return "SingleThreadedFlatMapDataset"


class FromTensorSlicesBenchmark(benchmark_base.DatasetBenchmarkBase):
  # [CLASS DOC]
  # Benchmark suite for creating datasets from tensor slices, combined with batching, repeating,
  # reshaping, sparse inputs and caching. Inherits the run-and-report support from the base class.
  """Benchmarks for `tf.data.Dataset.from_tensor_slices()`."""

  def benchmark_slice_repeat_batch(self):
    # [FUNCTION DOC]
    # Slices ten thousand random numbers, repeats for 100 epochs and batches by 100, then runs and
    # reports the benchmark over the resulting number of batches. Returns nothing.
    input_size = 10000
    batch_size = 100
    num_epochs = 100
    num_elements = input_size * num_epochs // batch_size

    input_data = np.random.randn(input_size)

    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = dataset.repeat(num_epochs).batch(batch_size)

    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "from_tensor_slices.benchmark.1",
            "parameters": "%d.%d" % (input_size, batch_size),
        },
        name="slice_repeat_batch_input_%d_batch_%d" % (input_size, batch_size))

  def benchmark_reshape_slice_repeat(self):
    # [FUNCTION DOC]
    # Reshapes ten thousand random numbers to a 100 by 100 matrix, slices it by rows, repeats 100
    # epochs, and runs and reports the benchmark over 10,000 row elements. Returns nothing.
    input_size = 10000
    reshape_dim = [100, 100]
    num_epochs = 100

    num_elements = num_epochs * reshape_dim[0]

    data = np.random.randn(input_size).reshape(*reshape_dim)
    dataset = dataset_ops.Dataset.from_tensor_slices(data).repeat(num_epochs)

    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "from_tensor_slices.benchmark.2",
            "parameters": "%d" % input_size,
        },
        name="reshape_slice_repeat_input_%d" % input_size,
    )

  def benchmark_slice_repeat_sparse(self):
    # [FUNCTION DOC]
    # Sweeps non-zero counts per row (0 to 100) and row counts (32 to 1024). For each, builds in a
    # traced function a batched sparse tensor from repeated copies, then an endless dataset that
    # slices it via the single-threaded flat-map dataset, and runs and reports 100,000 elements
    # for five iterations. Returns nothing.
    non_zeros_per_row_values = [0, 1, 5, 10, 100]
    num_rows_values = [32, 64, 128, 1024]

    for non_zeros_per_row in non_zeros_per_row_values:
      tensor = sparse_tensor.SparseTensor(
          indices=np.arange(non_zeros_per_row, dtype=np.int64)[:, np.newaxis],
          values=np.arange(non_zeros_per_row, dtype=np.int64),
          dense_shape=[1000])

      for num_rows in num_rows_values:

        # TODO(b/147153744): Function-valued attributes with their own
        # attributes are currently only supported in graph mode.
        @def_function.function
        def make_dataset():
          # [FUNCTION DOC]
          # HELPER: traced function; captures the loop's sparse tensor and row count. Batches the
          # repeated tensor into one element, makes an endless dataset of it, and returns a
          # single-threaded flat-map dataset that slices each element.
          # pylint: disable=cell-var-from-loop
          dataset = dataset_ops.Dataset.from_tensors(tensor)
          dataset = dataset.repeat(num_rows).batch(num_rows)
          batched_tensor = get_single_element.get_single_element(dataset)

          dataset = dataset_ops.Dataset.from_tensors(batched_tensor).repeat()
          return SingleThreadedFlatMapDataset(
              dataset, dataset_ops.Dataset.from_tensor_slices)

        self.run_and_report_benchmark(
            make_dataset(),
            num_elements=100000,
            iters=5,
            extras={
                "model_name": "from_tensor_slices.benchmark.3",
                "parameters": "%d.%d" % (non_zeros_per_row, num_rows),
            },
            name="slice_repeat_sparse_elements_per_row_%d_num_rows_%d" %
            (non_zeros_per_row, num_rows))

  def benchmark_slice_batch_cache_repeat(self):
    # [FUNCTION DOC]
    # Slices ten thousand random numbers, batches by 100, caches, repeats 100 epochs, then runs and
    # reports the benchmark over the resulting number of batches. Returns nothing.
    input_size = 10000
    batch_size = 100
    num_epochs = 100
    num_elements = input_size * num_epochs // batch_size

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data).batch(
            batch_size).cache().repeat(num_epochs))

    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "from_tensor_slices.benchmark.4",
            "parameters": "%d.%d" % (input_size, batch_size),
        },
        name="slice_batch_cache_repeat_input_%d_batch_%d" %
        (input_size, batch_size))


if __name__ == "__main__":
  benchmark_base.test.main()
