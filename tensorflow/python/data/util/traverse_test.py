# [MODULE DOC]
# Tests for the dataset graph traversal helpers in traverse.py. Builds various dataset pipelines
# (source only, map, concatenate, zip, a custom dataset with several variant tensors, flat map with
# captured datasets, and the tf.data service distribute transformation) in graph mode and asserts
# the names of the operations found by the traversal functions.
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
"""Tests for utilities for traversing the dataset construction graph."""

from absl.testing import parameterized

from tensorflow.python.compat import compat
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import traverse
from tensorflow.python.framework import combinations
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class _TestDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # HELPER: test dataset wrapping an input dataset with two chained dataset operations (prefetch
  # with buffer size one, then model), so a single dataset contributes several variant tensor
  # operations to the graph. Keeps the input structure unchanged.

  def __init__(self, input_dataset):
    # [FUNCTION DOC]
    # Builds the prefetch and model dataset operations on top of the input dataset's variant tensor
    # and initializes the base class with the final variant tensor. Input: the dataset to wrap.
    self._input_dataset = input_dataset
    temp_variant_tensor = gen_dataset_ops.prefetch_dataset(
        input_dataset._variant_tensor,
        buffer_size=1,
        **self._flat_structure)
    variant_tensor = gen_dataset_ops.model_dataset(
        temp_variant_tensor, **self._flat_structure)
    super(_TestDataset, self).__init__(input_dataset, variant_tensor)


class TraverseTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the traversal helpers. All tests run in graph mode only and compare the names of
  # the discovered operations against expected sets.

  @combinations.generate(test_base.graph_only_combinations())
  def testOnlySource(self):
    # [FUNCTION DOC]
    # Scenario: a range dataset alone. Asserts the traversal finds exactly the range dataset op.
    ds = dataset_ops.Dataset.range(10)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertAllEqual(["RangeDataset"], [x.name for x in variant_tensor_ops])

  @combinations.generate(test_base.graph_only_combinations())
  def testSimplePipeline(self):
    # [FUNCTION DOC]
    # Scenario: range dataset followed by a map with a square function. Asserts the found ops are
    # exactly the map and range dataset ops.
    ds = dataset_ops.Dataset.range(10).map(math_ops.square)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["MapDataset", "RangeDataset"]),
        set(x.name for x in variant_tensor_ops))

  @combinations.generate(test_base.graph_only_combinations())
  def testConcat(self):
    # [FUNCTION DOC]
    # Scenario: concatenation of two range datasets. Asserts the found ops are the concatenate op
    # and both range ops (the second with a numeric suffix).
    ds1 = dataset_ops.Dataset.range(10)
    ds2 = dataset_ops.Dataset.range(10)
    ds = ds1.concatenate(ds2)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["ConcatenateDataset", "RangeDataset", "RangeDataset_1"]),
        set(x.name for x in variant_tensor_ops))

  @combinations.generate(test_base.graph_only_combinations())
  def testZip(self):
    # [FUNCTION DOC]
    # Scenario: zip of two range datasets. Asserts the found ops are the zip op and both range ops.
    ds1 = dataset_ops.Dataset.range(10)
    ds2 = dataset_ops.Dataset.range(10)
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["ZipDataset", "RangeDataset", "RangeDataset_1"]),
        set(x.name for x in variant_tensor_ops))

  @combinations.generate(test_base.graph_only_combinations())
  def testMultipleVariantTensors(self):
    # [FUNCTION DOC]
    # Scenario: a range dataset wrapped by the custom dataset that adds prefetch and model ops.
    # Asserts all three ops (range, model, prefetch) are found.
    ds = dataset_ops.Dataset.range(10)
    ds = _TestDataset(ds)
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds)
    self.assertSetEqual(
        set(["RangeDataset", "ModelDataset", "PrefetchDataset"]),
        set(x.name for x in variant_tensor_ops))

  @combinations.generate(test_base.graph_only_combinations())
  def testFlatMap(self):
    # [FUNCTION DOC]
    # Scenario: a flat map over a prefetched range dataset whose function captures another dataset
    # (a repeated range). Asserts the traversal finds the flat map, prefetch, repeat and both range
    # ops, showing that captured datasets are discovered through function inputs.
    ds1 = dataset_ops.Dataset.range(10).repeat(10)

    def map_fn(ds):
      # [FUNCTION DOC]
      # HELPER: given a dataset, returns a mapping function that batches that dataset by its
      # argument, thereby capturing the dataset in the flat map function.

      def _map(x):
        # [FUNCTION DOC]
        # HELPER: returns the captured dataset batched by the given batch size.
        return ds.batch(x)

      return _map

    ds2 = dataset_ops.Dataset.range(20).prefetch(1)
    ds2 = ds2.flat_map(map_fn(ds1))
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(ds2)
    self.assertSetEqual(
        set([
            "FlatMapDataset", "PrefetchDataset", "RepeatDataset",
            "RangeDataset", "RangeDataset_1"
        ]), set(x.name for x in variant_tensor_ops))

  @combinations.generate(test_base.graph_only_combinations())
  def testTfDataService(self):
    # [FUNCTION DOC]
    # Scenario: a range dataset distributed through the data service with a placeholder address.
    # Asserts the capture-by-value traversal includes the range op, the data service dataset op
    # (version four when forward compatible past the 2022-08-31 date, else version three) and the
    # dummy iteration counter op.
    ds = dataset_ops.Dataset.range(10)
    ds = ds.apply(
        data_service_ops.distribute("parallel_epochs", "grpc://foo:0"))
    ops = traverse.obtain_capture_by_value_ops(ds)
    data_service_dataset_op = ("DataServiceDatasetV4"
                               if compat.forward_compatible(2022, 8, 31) else
                               "DataServiceDatasetV3")
    self.assertContainsSubset(
        ["RangeDataset", data_service_dataset_op, "DummyIterationCounter"],
        set(x.name for x in ops))


if __name__ == "__main__":
  test.main()
