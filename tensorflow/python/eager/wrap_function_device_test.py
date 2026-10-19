# [MODULE DOC]
# Test checking that a wrapped graph function that consumes a dataset can be built and run when the
# graph is imported under an explicit CPU or GPU device scope. Defines three module-level helpers
# that sum a range dataset in different ways (reduce, python loop, iterator loop) and one
# parameterized test class. When run as a script, eager execution is enabled first.
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

from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer as graph_def_importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def _dataset_reduce_sum(dataset):
  # [FUNCTION DOC]
  # HELPER: dataset-summing variant for the device test. Input: a dataset of 64-bit integers.
  # Uses the dataset's reduce method starting from a zero constant and adding each element.
  # Returns the scalar sum tensor.
  return dataset.reduce(
      constant_op.constant(0, dtype=dtypes.int64), lambda x, y: x + y)


def _loop_dataset_sum(dataset):
  # [FUNCTION DOC]
  # HELPER: dataset-summing variant for the device test. Input: a dataset of 64-bit integers.
  # Iterates over the dataset directly in a Python loop, accumulating into a zero-initialized
  # constant. Returns the scalar sum tensor.
  value = constant_op.constant(0, dtype=dtypes.int64)
  for d in dataset:
    value += d
  return value


def _iter_dataset_sum(dataset):
  # [FUNCTION DOC]
  # HELPER: dataset-summing variant for the device test. Input: a dataset of 64-bit integers.
  # Explicitly creates an iterator over the dataset and accumulates its elements into a
  # zero-initialized constant. Returns the scalar sum tensor.
  value = constant_op.constant(0, dtype=dtypes.int64)
  for d in iter(dataset):
    value += d
  return value


class WrappedGraphTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case covering wrapped-graph functions that use datasets under an explicit
  # device scope, over the three dataset-summing helpers and both CPU and GPU device types.

  @parameterized.named_parameters(
      ('cpu_reduce', 'CPU', _dataset_reduce_sum),
      ('gpu_reduce', 'GPU', _dataset_reduce_sum),
      ('cpu_loop', 'CPU', _loop_dataset_sum),
      ('gpu_loop', 'GPU', _loop_dataset_sum),
      ('cpu_iter', 'CPU', _iter_dataset_sum),
      ('gpu_iter', 'GPU', _iter_dataset_sum),
  )
  def testWrapFuncDatasetDevice(self, device_type, dataset_reduce_fn):
    # [FUNCTION DOC]
    # Scenario, parameterized by device type name and a dataset-summing function. Skipped if no
    # logical device of that type exists. Traces a function summing a ten element range dataset
    # with the given helper, takes its graph, wraps a function that imports the graph definition
    # under the first device of that type into a no-argument wrapped function (also created under
    # that device scope), and calls it. Passes if no error is raised; the sum is not asserted.

    devices = config.list_logical_devices(device_type=device_type)
    if not devices:
      self.skipTest('Skip when {} is not detected by TF'.format(device_type))

    @def_function.function
    def comp():
      # [FUNCTION DOC]
      # Traced function: applies the selected dataset-summing helper to a ten element range
      # dataset and returns the sum tensor.
      return dataset_reduce_fn(dataset_ops.Dataset.range(10))

    graph = comp.get_concrete_function().graph

    def function_to_wrap():
      # [FUNCTION DOC]
      # Function handed to the function wrapper: under the chosen device scope, imports the traced
      # graph's graph definition into the wrapper's graph and returns the import result.
      with ops.device(devices[0].name):
        return graph_def_importer.import_graph_def(graph.as_graph_def())

    with ops.device(devices[0].name):
      wrapped_noarg_fn = wrap_function.wrap_function(
          function_to_wrap, signature=[])

    wrapped_noarg_fn()


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
