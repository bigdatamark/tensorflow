# [MODULE DOC]
# Tests for dynamic sharding in the tf.data service, where the dispatcher hands out source splits
# to workers on demand. Covers many dataset shapes (range, tensor slices, interleave, flat map,
# group by window, repeats, shuffle, zip of equal and unequal inputs, sampling and choosing from
# datasets, enumerate and zip with replicate-on-split, concatenate), unsupported cases (snapshot,
# nested distributed datasets), job sharing and dataset registration behavior, garbage collection
# of dynamic sharding jobs, and sharding of TFRecord file lists. Uses the shared data service
# test base. Runs under the standard test main entry point.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for dynamic sharding."""
import collections
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class DynamicShardingTest(data_service_test_base.TestBase,
                          parameterized.TestCase):
  # [CLASS DOC]
  # Test case for dynamic sharding of datasets through the tf.data service. Each test starts an
  # in-process test cluster (or dispatcher and worker servers), builds a dataset, distributes it
  # with the dynamic sharding policy and checks the produced elements or raised errors.

  def _make_dynamic_sharding_dataset(self, dataset, cluster):
    # [FUNCTION DOC]
    # HELPER: shared by the tests in this class to distribute a dataset.
    # Takes a dataset and a test cluster, and returns the dataset distributed through the cluster
    # with the dynamic sharding policy and the job name "job_name".
    return self.make_distributed_dataset(
        dataset,
        cluster,
        processing_mode=data_service_ops.ShardingPolicy.DYNAMIC,
        job_name="job_name")

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    # [FUNCTION DOC]
    # Distributes a 100-element range over two workers with dynamic sharding; expects every
    # number exactly once, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_elements = 100
    ds = dataset_ops.Dataset.range(num_elements)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    self.assertDatasetProduces(
        ds, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testNoJobName(self):
    # [FUNCTION DOC]
    # Same as the basic test but distributes with no job name; expects all 100 numbers once, in
    # any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_elements = 100
    ds = dataset_ops.Dataset.range(num_elements)
    ds = self.make_distributed_dataset(
        ds,
        cluster,
        processing_mode=data_service_ops.ShardingPolicy.DYNAMIC,
        job_name=None)
    self.assertDatasetProduces(
        ds, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testTensorSlices(self):
    # [FUNCTION DOC]
    # Distributes a dataset from tensor slices of four values over two workers; expects those
    # values, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    vals = [5, 1, 2, 4]
    ds = dataset_ops.Dataset.from_tensor_slices(vals)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    self.assertDatasetProduces(ds, vals, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testInterleave(self):
    # [FUNCTION DOC]
    # Distributes a tensor-slices dataset passed through interleave (each element becomes a
    # one-element dataset); expects the three source values, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    elements = [1, 5, 0]
    ds = dataset_ops.Dataset.from_tensor_slices(elements)
    ds = ds.interleave(lambda x: dataset_ops.Dataset.from_tensor_slices([x]))
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    self.assertDatasetProduces(ds, elements, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testParallelInterleave(self):
    # [FUNCTION DOC]
    # Like the interleave test but with automatic parallel calls; expects the three source values,
    # in any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    elements = [1, 5, 0]
    ds = dataset_ops.Dataset.from_tensor_slices(elements)
    ds = ds.interleave(
        lambda x: dataset_ops.Dataset.from_tensor_slices([x]),
        num_parallel_calls=dataset_ops.AUTOTUNE)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    self.assertDatasetProduces(ds, elements, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFlatMap(self):
    # [FUNCTION DOC]
    # Distributes a tensor-slices dataset passed through flat map into one-element datasets;
    # expects the three source values, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    elements = [1, 5, 0]
    ds = dataset_ops.Dataset.from_tensor_slices(elements)
    ds = ds.flat_map(lambda x: dataset_ops.Dataset.from_tensor_slices([x]))
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    self.assertDatasetProduces(ds, elements, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testGroupByWindow(self):
    # [FUNCTION DOC]
    # Regression test: groups three elements into one window whose reduce function zips the window
    # with a range dataset, then distributes with dynamic sharding and reads everything. Passes if
    # no error occurs, i.e. split providers are not wrongly propagated into the reduce datasets.
    # Verify that split providers are not propagated into iterators created for
    # the reduce datasets created by the reduce_fn in group_by_window.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    elements = [1, 5, 0]
    ds = dataset_ops.Dataset.from_tensor_slices(elements)

    def reduce_fn(_, window):
      # [FUNCTION DOC]
      # HELPER: reduce function for the group-by-window call. Ignores the key and returns the
      # window zipped with a 100-element range dataset.
      return dataset_ops.Dataset.zip((window, dataset_ops.Dataset.range(100)))

    ds = ds.group_by_window(lambda x: 0, reduce_fn, window_size=3)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    # This will fail if the tensor_slices split provider is propagated into the
    # `reduce_fn`, since the `zip` requires either 0 or 2 split providers.
    self.getDatasetOutput(ds)

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatBeforeDistribution(self):
    # [FUNCTION DOC]
    # Repeats a 20-element range five times, then distributes; expects the range five times, in
    # any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_repeats = 5
    num_elements = 20
    ds = dataset_ops.Dataset.range(num_elements).repeat(num_repeats)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    self.assertDatasetProduces(
        ds, num_repeats * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatAfterDistribution(self):
    # [FUNCTION DOC]
    # Distributes a 20-element range, then repeats the distributed dataset five times; expects the
    # range five times, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_repeats = 5
    num_elements = 20
    ds = dataset_ops.Dataset.range(num_elements)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    ds = ds.repeat(num_repeats)
    self.assertDatasetProduces(
        ds, num_repeats * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testForeverRepeat(self):
    # [FUNCTION DOC]
    # Distributes an endlessly repeated 20-element range over two workers and reads 1000
    # elements, counting occurrences. Expects every value to appear more than a quarter of the
    # average share (25 times), i.e. reasonably balanced.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_elements = 20
    elements_to_read = 1000
    ds = dataset_ops.Dataset.range(num_elements).repeat()
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    get_next = self.getNext(ds)
    results = {}
    for _ in range(elements_to_read):
      val = self.evaluate(get_next())
      if val not in results:
        results[val] = 0
      results[val] += 1
    for i in range(num_elements):
      self.assertGreater(results[i], elements_to_read / num_elements / 2)

  @combinations.generate(test_base.default_test_combinations())
  def testForeverRepeatFewElements(self):
    # [FUNCTION DOC]
    # Uses five workers and an endlessly repeated one-element dataset (fewer elements than
    # workers). Reads 20 elements, all must be zero. Then stops four of five workers and reads 20
    # more zeros, showing reading continues with one worker.
    num_workers = 5
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    # Less than the number of workers, so that some workers get zero elements on
    # the first repetition.
    num_elements = 1
    ds = dataset_ops.Dataset.range(num_elements).repeat()
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    get_next = self.getNext(ds)
    for _ in range(20):
      self.assertEqual(self.evaluate(get_next()), 0)

    # Stop all but one worker and check that we can still read.
    for i in range(num_workers - 1):
      cluster.workers[i].stop()
    for _ in range(20):
      self.assertEqual(self.evaluate(get_next()), 0)

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleAndRepeat(self):
    # [FUNCTION DOC]
    # Distributes a shuffled then repeated (five times) 20-element range; expects the range five
    # times, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_repeats = 5
    num_elements = 20
    ds = dataset_ops.Dataset.range(num_elements).shuffle(num_elements).repeat(
        num_repeats)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    self.assertDatasetProduces(
        ds, num_repeats * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testZip(self):
    # [FUNCTION DOC]
    # Distributes, on one worker, the zip of a ten-element range with itself; expects ordered
    # pairs of equal numbers 0 to 9.
    num_elements = 10
    cluster = data_service_test_base.TestCluster(num_workers=1)
    a = dataset_ops.Dataset.range(num_elements)

    ds = dataset_ops.Dataset.zip((a, a))
    ds = self._make_dynamic_sharding_dataset(ds, cluster)

    self.assertDatasetProduces(
        ds, list(zip(range(num_elements), range(num_elements))))

  @combinations.generate(test_base.default_test_combinations())
  def testNestedZip(self):
    # [FUNCTION DOC]
    # Distributes, on one worker, a zip containing another zip as one component (four components
    # in total); expects ordered tuples with the nested pair inside, for numbers 0 to 9.
    num_elements = 10
    cluster = data_service_test_base.TestCluster(num_workers=1)
    a = dataset_ops.Dataset.range(num_elements)

    ds = dataset_ops.Dataset.zip((a, a))
    ds = dataset_ops.Dataset.zip((a, a, ds, a))
    ds = self._make_dynamic_sharding_dataset(ds, cluster)

    b = list(range(10))
    self.assertDatasetProduces(ds, list(zip(b, b, zip(b, b), b)))

  @combinations.generate(test_base.default_test_combinations())
  def testImbalancedZip(self):
    # [FUNCTION DOC]
    # On one worker, zips a 200-element range with a 1000-element range and distributes it;
    # expects ordered pairs only for the first 200 numbers (the shorter input).
    smaller_num_elements = 200
    larger_num_elements = 1000

    cluster = data_service_test_base.TestCluster(num_workers=1)
    a = dataset_ops.Dataset.range(smaller_num_elements)
    b = dataset_ops.Dataset.range(larger_num_elements)

    ds = dataset_ops.Dataset.zip((a, b))
    ds = self._make_dynamic_sharding_dataset(ds, cluster)

    self.assertDatasetProduces(
        ds, list(zip(range(smaller_num_elements), range(smaller_num_elements))))

  @combinations.generate(test_base.default_test_combinations())
  def testImbalancedZipAndRepeat(self):
    # [FUNCTION DOC]
    # On one worker, zips a 200-element range with a 1000-element range, repeats it three times,
    # distributes it, and expects three repetitions of the 200 equal-number pairs, in order.
    smaller_num_elements = 200
    larger_num_elements = 1000
    repetitions = 3

    cluster = data_service_test_base.TestCluster(num_workers=1)
    a = dataset_ops.Dataset.range(smaller_num_elements)
    b = dataset_ops.Dataset.range(larger_num_elements)

    ds = dataset_ops.Dataset.zip((a, b))
    ds = ds.repeat(repetitions)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)

    expected = repetitions * (
        list(zip(range(smaller_num_elements), range(smaller_num_elements)))
    )

    self.assertDatasetProduces(ds, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testImbalancedZipMultiWorker(self):
    # [FUNCTION DOC]
    # With three workers, distributes the zip of a 200-element and a 1000-element range. Because
    # splits are nondeterministic, only asserts that exactly 200 elements are produced.
    smaller_num_elements = 200
    larger_num_elements = 1000
    cluster = data_service_test_base.TestCluster(num_workers=3)
    a = dataset_ops.Dataset.range(smaller_num_elements)
    b = dataset_ops.Dataset.range(larger_num_elements)

    ds = dataset_ops.Dataset.zip((a, b))
    ds = self._make_dynamic_sharding_dataset(ds, cluster)

    # Cannot assert specific elements because the range datasets are split
    # nondeterministically and may not line up.
    self.assertLen(self.getDatasetOutput(ds), smaller_num_elements)

  @combinations.generate(test_base.default_test_combinations())
  def testZipDifferentRates(self):
    # [FUNCTION DOC]
    # With three workers, zips a 100-element range with a copy filtered to multiples of ten and
    # distributes it; asserts exactly ten elements are produced.
    cluster = data_service_test_base.TestCluster(num_workers=3)
    a = dataset_ops.Dataset.range(100)
    b = dataset_ops.Dataset.range(100).filter(
        lambda x: math_ops.equal(x % 10, 0))

    ds = dataset_ops.Dataset.zip((a, b))
    ds = self._make_dynamic_sharding_dataset(ds, cluster)

    self.assertLen(self.getDatasetOutput(ds), 10)

  @combinations.generate(test_base.default_test_combinations())
  def testZipDifferentRepeats(self):
    # [FUNCTION DOC]
    # With three workers, zips a 50-element range with a ten-element range repeated ten times and
    # distributes it; asserts exactly 50 elements are produced.
    cluster = data_service_test_base.TestCluster(num_workers=3)
    a = dataset_ops.Dataset.range(50)
    b = dataset_ops.Dataset.range(10).repeat(10)

    ds = dataset_ops.Dataset.zip((a, b))
    ds = self._make_dynamic_sharding_dataset(ds, cluster)

    self.assertLen(self.getDatasetOutput(ds), 50)

  @combinations.generate(test_base.default_test_combinations())
  def testSampleFromDatasets(self):
    # [FUNCTION DOC]
    # With three workers, builds a dataset sampling endless constant datasets of classes 0, 1, 2
    # with weights 0.6, 0.3, 0.1, distributes it, takes 200 samples and counts frequencies.
    # Asserts class 0 appears more often than class 1, and class 1 more than class 2.
    cluster = data_service_test_base.TestCluster(num_workers=3)
    num_samples = 200
    weights = [.6, .3, .1]
    classes = len(weights)

    # Create a dataset that samples each integer in `[0, num_datasets)`
    # with probability given by `weights[i]`.
    ds = dataset_ops.Dataset.sample_from_datasets(
        [dataset_ops.Dataset.from_tensors(i).repeat() for i in range(classes)],
        weights)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    ds = ds.take(num_samples)

    freqs = np.zeros([classes])
    for v in self.getDatasetOutput(ds, requires_initialization=True):
      freqs[v] += 1

    self.assertGreater(freqs[0], freqs[1])
    self.assertGreater(freqs[1], freqs[2])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_workers=[1, 3])))
  def testChooseFromDatasets(self, num_workers):
    # [FUNCTION DOC]
    # Parameterized by worker count (1 or 3). Chooses among three endless word datasets using 15
    # random choices, distributes with dynamic sharding. Expects the chosen words repeated once per
    # worker; compared in any order when there is more than one worker, in order otherwise.
    # Uses unseeded random choices, so the inputs differ per run.
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    words = [b"foo", b"bar", b"baz"]
    datasets = [dataset_ops.Dataset.from_tensors(w).repeat() for w in words]
    choice_array = np.random.randint(3, size=(15,), dtype=np.int64)
    choice_dataset = dataset_ops.Dataset.from_tensor_slices(choice_array)
    ds = dataset_ops.Dataset.choose_from_datasets(datasets, choice_dataset)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    expected = [words[i] for i in choice_array] * num_workers

    assert_items_equal = (num_workers > 1)
    self.assertDatasetProduces(
        ds, expected, assert_items_equal=assert_items_equal)

  @combinations.generate(test_base.default_test_combinations())
  def testEnumerateReplicateOnSplit(self):
    # [FUNCTION DOC]
    # With three workers, distributes an endlessly repeated three-string dataset with enumerate.
    # Reads, counting how often each index appears, until index 10 has been seen once per worker.
    # Asserts indices 0 to 9 each appeared exactly once per worker (enumerate is replicated, not
    # split, across workers).
    num_workers = 3
    cluster = data_service_test_base.TestCluster(num_workers)
    ds = dataset_ops.Dataset.from_tensor_slices(["a", "b", "c"]).repeat()
    ds = ds.enumerate()
    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    get_next = self.getNext(ds)

    counts = collections.defaultdict(int)
    while True:
      i, _ = self.evaluate(get_next())
      counts[i] += 1
      # Read until all workers have reached enumeration index 10.
      if counts[10] == num_workers:
        break

    for i in range(10):
      self.assertEqual(counts[i], num_workers)

  @combinations.generate(test_base.default_test_combinations())
  def testZipReplicateOnSplit(self):
    # [FUNCTION DOC]
    # With three workers, zips a 900-element int32 range with a five-element dataset rewritten to
    # replicate on split, and distributes it with dynamic sharding under a shared job name.
    # Expects the second component output to be 0 to 4 once per worker.
    num_workers = 3
    cluster = data_service_test_base.TestCluster(num_workers)

    ds1 = dataset_ops.Dataset.range(100, 1000, output_type=dtypes.int32)
    ds2 = dataset_ops.Dataset.from_tensor_slices(range(0, 5))
    ds2 = dataset_ops.apply_rewrite(ds2, "replicate_on_split")

    ds = dataset_ops.Dataset.zip(ds1, ds2)
    ds = ds.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.DYNAMIC,
            cluster.dispatcher_address(),
            job_name="shared_job",
        )
    )
    ds2_output = [y for x, y in self.getDatasetOutput(ds)]
    self.assertCountEqual(ds2_output, [0, 1, 2, 3, 4] * num_workers)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_workers=[1, 3])))
  def testConcatenate(self, num_workers):
    # [FUNCTION DOC]
    # Parameterized by worker count (1 or 3). Concatenates ranges 0-99 and 100-199 and distributes
    # them; expects 0 to 199, in any order when more than one worker, in order for one worker.
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    a = dataset_ops.Dataset.range(100)
    b = dataset_ops.Dataset.range(100, 200)
    ds = a.concatenate(b)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)

    assert_items_equal = (num_workers > 1)
    self.assertDatasetProduces(
        ds, list(range(200)), assert_items_equal=assert_items_equal)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(already_written=[True, False])))
  def testSnapshot(self, already_written):
    # [FUNCTION DOC]
    # Parameterized by whether the snapshot has been materialized. Snapshots a 100-element range
    # into the test temp directory, optionally reading it fully first, then distributes with
    # dynamic sharding. Expects reading to raise an unimplemented error saying splitting is not
    # implemented for snapshot datasets.
    num_workers = 3
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    ds = dataset_ops.Dataset.range(100)
    ds = ds.snapshot(self.get_temp_dir())
    if already_written:
      # Materialize the snapshot.
      self.getDatasetOutput(ds)

    ds = self._make_dynamic_sharding_dataset(ds, cluster)
    error_regex = "Splitting is not implemented for snapshot datasets"
    with self.assertRaisesRegex(errors.UnimplementedError, error_regex):
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributedDataset(self):
    # [FUNCTION DOC]
    # Distributes a tensor-slices dataset through one cluster with sharding off, maps it, then
    # distributes the result through a second cluster with dynamic sharding. Expects an
    # unimplemented error saying split providers cannot be created for the data service dataset.
    cluster_1 = data_service_test_base.TestCluster(num_workers=1)
    cluster_2 = data_service_test_base.TestCluster(num_workers=1)
    num_sizes = 10
    size_repeats = 5
    numbers = [1 * i for i in range(num_sizes)] * size_repeats
    ds = dataset_ops.Dataset.from_tensor_slices(numbers)
    ds = self.make_distributed_dataset(
        ds, cluster_1, processing_mode=data_service_ops.ShardingPolicy.OFF)
    ds = ds.map(lambda x: x + 1)
    ds = self._make_dynamic_sharding_dataset(ds, cluster_2)

    error_regex = ("Cannot create split providers for dataset " +
                   "of type DataServiceDataset")
    with self.assertRaisesRegex(errors.UnimplementedError, error_regex):
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributedEpoch(self):
    # [FUNCTION DOC]
    # Distributes a 100-element range over two workers using the legacy string processing mode
    # for distributed epoch; expects each number once, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    num_elements = 100
    ds = dataset_ops.Dataset.range(num_elements)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(
        ds, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFlatMapWithRepeat(self):
    # [FUNCTION DOC]
    # With three workers, flat-maps a five-element range into a string dataset a, b, c repeated
    # ten times and distributes it; expects the three strings fifty times each pattern, in any
    # order.
    cluster = data_service_test_base.TestCluster(num_workers=3)
    ds = dataset_ops.Dataset.range(5)

    def flat_map_fn(_):
      # [FUNCTION DOC]
      # HELPER: flat map function for the test. Ignores its input and returns a dataset of the
      # strings a, b, c repeated ten times.
      return dataset_ops.Dataset.from_tensor_slices(["a", "b", "c"]).repeat(10)

    ds = ds.flat_map(flat_map_fn)
    ds = self._make_dynamic_sharding_dataset(ds, cluster)

    self.assertDatasetProduces(
        ds, [b"a", b"b", b"c"] * 50, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleRegistration(self):
    # [FUNCTION DOC]
    # Applies distribute twice to the same range dataset with the same shared job name, giving two
    # datasets that join one job. Reads the first fully, then the second, and expects the union
    # to be the numbers 0 to 99 exactly once (the job is shared, not duplicated).
    cluster = data_service_test_base.TestCluster(num_workers=2)
    dataset = dataset_ops.Dataset.range(100)
    dataset1 = dataset.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.DYNAMIC,
            cluster.dispatcher_address(),
            job_name="shared_job"
        )
    )
    dataset2 = dataset.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.DYNAMIC,
            cluster.dispatcher_address(),
            job_name="shared_job"
        )
    )

    output = []
    iter1 = self.getNext(dataset1)
    iter2 = self.getNext(dataset2)
    output.extend(self.getIteratorOutput(iter1))
    output.extend(self.getIteratorOutput(iter2))
    self.assertCountEqual(output, list(range(100)))

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentDatasetIdsForSameJob(self):
    # [FUNCTION DOC]
    # Registers the same range dataset twice under two explicit dataset ids, creates a dataset
    # from each id with dynamic sharding and the same job name, and alternates reading five
    # elements from each before draining both. Since the dataset id is prepended to the job name,
    # the jobs differ, so the output must contain two full copies of 0 to 99, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=2)
    dataset = dataset_ops.Dataset.range(100)
    dataset_id1 = data_service_ops.register_dataset(
        cluster.dispatcher_address(), dataset, dataset_id="dataset_id1"
    )
    dataset1 = data_service_ops.from_dataset_id(
        data_service_ops.ShardingPolicy.DYNAMIC,
        cluster.dispatcher_address(),
        dataset_id1,
        element_spec=dataset.element_spec,
        job_name="shared_job",
    )

    dataset_id2 = data_service_ops.register_dataset(
        cluster.dispatcher_address(), dataset, dataset_id="dataset_id2"
    )
    dataset2 = data_service_ops.from_dataset_id(
        data_service_ops.ShardingPolicy.DYNAMIC,
        cluster.dispatcher_address(),
        dataset_id2,
        element_spec=dataset.element_spec,
        job_name="shared_job",
    )

    output = []
    iter1 = self.getNext(dataset1)
    iter2 = self.getNext(dataset2)
    for _ in range(5):
      output.append(self.evaluate(iter1()))
      output.append(self.evaluate(iter2()))
    output.extend(self.getIteratorOutput(iter1))
    output.extend(self.getIteratorOutput(iter2))
    # Produces 2 copies of the dataset because `from_dataset_id` prepends the
    # dataset ID to the job name.
    self.assertCountEqual(output, list(range(100)) * 2)

  @combinations.generate(test_base.eager_only_combinations())
  def testGcDynamicShardingJobIfRequested(self):
    # [FUNCTION DOC]
    # Eager only. Starts a dispatcher with a short job garbage collection timeout and the option
    # to collect dynamic sharding jobs, plus a worker with a 100 ms heartbeat. Reads the first
    # element (0) of a distributed 1000-element range and checks the worker has one task. Drops
    # the iterator and waits until the worker has no tasks (job collected). A new iterator must
    # start again at 0, proving the job was garbage collected; then waits for tasks to drain
    # again and releases the worker. Uses private worker task counting.
    dispatcher = server_lib.DispatchServer(
        service_config_pb2.DispatcherConfig(
            protocol="grpc",
            job_gc_check_interval_ms=50,
            job_gc_timeout_ms=20,
            gc_dynamic_sharding_jobs=True,
        )
    )
    dispatcher_address = dispatcher.target.split("://")[1]
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(
            dispatcher_address=dispatcher_address, heartbeat_interval_ms=100
        )
    )

    num_elements = 1000
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.apply(
        data_service_ops._distribute(
            processing_mode=data_service_ops.ShardingPolicy.DYNAMIC,
            service=dispatcher.target,
        )
    )
    it = iter(dataset)
    self.assertEqual(self.evaluate(next(it)), 0)
    self.assertEqual(worker._num_tasks(), 1)
    del it
    while worker._num_tasks() > 0:
      time.sleep(0.1)

    # If re-creating the iterator with the same job, the elements are revisited.
    it = iter(dataset)
    self.assertEqual(self.evaluate(next(it)), 0)
    del it
    while worker._num_tasks() > 0:
      time.sleep(0.1)
    del worker


class DynamicShardingFilesTest(data_service_test_base.TestBase,
                               tf_record_test_base.TFRecordTestBase,
                               parameterized.TestCase):
  # [CLASS DOC]
  # Test case for dynamic sharding of datasets reading TFRecord files. Combines the data service
  # test base with the TFRecord test base; setUp creates five files of five records each.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the parent setup, then records that there are five files with five
    # records each and creates those TFRecord files, remembering their names for the tests.
    super(DynamicShardingFilesTest, self).setUp()
    self._num_files = 5
    self._num_records = 5
    self._filenames = self._createFiles()

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleFiles(self):
    # [FUNCTION DOC]
    # With three workers, shuffles the list of five record files, interleaves TFRecord readers
    # over them and distributes with dynamic sharding. Expects all 25 records (each file's five
    # records) exactly once, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=3)
    shuffled_filenames = random_ops.random_shuffle(self._filenames)
    dataset = dataset_ops.Dataset.from_tensor_slices(shuffled_filenames)
    dataset = dataset.interleave(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.DYNAMIC)
    # pylint:disable=g-complex-comprehension
    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 5)
        for record in range(0, 5)
    ]
    self.assertDatasetProduces(
        dataset,
        expected,
        requires_initialization=True,
        assert_items_equal=True)


if __name__ == "__main__":
  test.main()
