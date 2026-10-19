# [MODULE DOC]
# General tests for the tf.data service client ops (distribute, register dataset, from dataset id)
# running against in-process dispatcher and worker clusters built with the shared test base.
# Covers basic reads, sparse/ragged/lookup-table elements, shuffle order, shared and distinct job
# names, job and client garbage collection, determinism, stateful-op policies, argument
# validation, dataset registration and ids, nested distribution, large graphs and variables.
# Defines two module constants for the temporary and empty work directory settings taken from the
# test base. Some nested helper functions that are traced or converted by the framework are left
# without comment blocks on purpose (see the individual test docs).
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
"""Tests for tf.data service ops."""
import time

from absl.testing import parameterized

from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

TMP_WORK_DIR = data_service_test_base.TMP_WORK_DIR
NO_WORK_DIR = data_service_test_base.NO_WORK_DIR


class DataServiceOpsTest(
    data_service_test_base.TestBase, parameterized.TestCase
):
  # [CLASS DOC]
  # Main test case for tf.data service client behavior. Tests build a test cluster (dispatcher
  # plus workers) through the base class helpers, distribute datasets through it and check the
  # produced elements or the raised errors. Also holds the run_stateful helper.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          data_service_test_base.all_cluster_configurations(),
      )
  )
  def testDistributeBasic(self, work_dir, fault_tolerant_mode):
    # [FUNCTION DOC]
    # Parameterized over all cluster configurations (work directory, fault tolerant mode).
    # Distributes a range of 10 elements over one worker and expects 0..9.
    cluster = self.make_test_cluster(
        num_workers=1,
        work_dir=work_dir,
        fault_tolerant_mode=fault_tolerant_mode,
    )
    num_elements = 10
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeInvalidCompression(self):
    # [FUNCTION DOC]
    # Distributing with an unknown compression name must raise a value error about an invalid
    # compression argument.
    cluster = self.make_test_cluster(num_workers=1)
    with self.assertRaisesRegex(ValueError, "Invalid `compression` argument"):
      self.make_distributed_range_dataset(10, cluster, compression="foo")

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeSparse(self):
    # [FUNCTION DOC]
    # Eager only. Distributes a dataset holding one sparse tensor element and checks that
    # converting the received element to dense gives a single zero.
    cluster = self.make_test_cluster(num_workers=1)
    element = sparse_tensor.SparseTensor(
        indices=[[0]],
        values=constant_op.constant([0], dtype=dtypes.int32),
        dense_shape=[1],
    )
    ds = dataset_ops.Dataset.from_tensors(element)
    ds = self.make_distributed_dataset(ds, cluster)
    results = [sparse_ops.sparse_tensor_to_dense(elem) for elem in ds]
    self.assertAllEqual(results, [[0]])

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeRagged(self):
    # [FUNCTION DOC]
    # Eager only. Maps values 1,5,3,2,8 to ranges, batches into ragged batches of two, distributes
    # them, and checks the three padded dense batches received.
    cluster = self.make_test_cluster(num_workers=1)
    ds = dataset_ops.Dataset.from_tensor_slices([1, 5, 3, 2, 8])
    ds = ds.map(math_ops.range)
    ds = ds.apply(batching.dense_to_ragged_batch(2))
    ds = self.make_distributed_dataset(ds, cluster)
    results = [elem.to_tensor() for elem in ds]
    self.assertAllEqual(results[0], [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]])
    self.assertAllEqual(results[1], [[0, 1, 2], [0, 1, 0]])
    self.assertAllEqual(results[2], [[0, 1, 2, 3, 4, 5, 6, 7]])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              init_source=["textfile", "keyvaluetensor", "dataset"]
          ),
      )
  )
  def testDistributeLookupTable(self, init_source):
    # [FUNCTION DOC]
    # Parameterized over the table initializer source (text file, key/value tensors, dataset).
    # Maps a range of 3 through a static hash table holding values 10 and 11 (default -1),
    # distributes it, initializes tables and expects 10, 11, -1.
    cluster = self.make_test_cluster(num_workers=1)
    initializer = self.lookupTableInitializer(init_source, [10, 11])
    table = lookup_ops.StaticHashTable(initializer, -1)
    ds = dataset_ops.Dataset.range(3)
    ds = ds.map(table.lookup)
    ds = self.make_distributed_dataset(ds, cluster)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertDatasetProduces(ds, [10, 11, -1], requires_initialization=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(value_rank=[0, 1]),
      )
  )
  def testDistributeMutableHashTable(self, value_rank):
    # [FUNCTION DOC]
    # Parameterized over value rank 0 or 1. Inserts two entries into a mutable hash table,
    # looks up a range of 3 through it in a distributed dataset and expects the two stored
    # values followed by the default value.
    def value(v):
      # [FUNCTION DOC]
      # HELPER: builds table values for the enclosing test.
      # Wraps the input in a pair, repeated value_rank times; returns the input unchanged for
      # rank 0.
      for _ in range(value_rank):
        v = [v, v]
      return v

    v1 = value(10)
    v2 = value(11)
    default_value = value(-1)

    cluster = self.make_test_cluster(num_workers=1)
    table = lookup_ops.MutableHashTable(
        dtypes.int64, dtypes.int64, default_value
    )
    self.evaluate(table.insert([0, 1], [v1, v2]))
    ds = dataset_ops.Dataset.range(3)
    ds = ds.map(table.lookup)
    ds = self.make_distributed_dataset(ds, cluster)
    self.assertDatasetProduces(
        ds, [v1, v2, default_value], requires_initialization=True
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(shuffle_seed=[None, 10]),
      )
  )
  def testShuffleOrder(self, shuffle_seed):
    # [FUNCTION DOC]
    # Parameterized over shuffle seed (none or 10). Distributes a shuffled range of 100 across two
    # workers; each worker emits a full pass, so the output holds two interleaved passes. Records
    # the order of first and second occurrences of each element. With no seed the two orders must
    # differ; with a fixed seed they must be equal.
    random_seed.set_random_seed(None)
    num_elements = 100
    cluster = self.make_test_cluster(num_workers=2)
    ds = dataset_ops.Dataset.range(num_elements)
    ds = ds.shuffle(num_elements, seed=shuffle_seed)
    ds = self.make_distributed_dataset(ds, cluster)
    output = self.getDatasetOutput(ds)

    # The output will be two sequences of range(num_elements)
    # non-deterministically interleaved together. If the orders of the elements
    # were the same, first_order and second_order computed below will be equal.
    first_order = {}
    second_order = {}
    for element in output:
      if element in first_order:
        second_order[element] = len(second_order)
      else:
        first_order[element] = len(first_order)
    if shuffle_seed is None:
      self.assertNotEqual(first_order, second_order)
    else:
      self.assertEqual(first_order, second_order)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleEpochs(self):
    # [FUNCTION DOC]
    # Reads the same distributed range of 3 elements ten times in a row; every pass must yield
    # 0..2.
    cluster = self.make_test_cluster(num_workers=1)
    num_elements = 3
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    for _ in range(10):
      self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatedDataset(self):
    # [FUNCTION DOC]
    # Repeats a distributed range of 10 elements five times and expects the sequence 0..9
    # five times in order.
    cluster = self.make_test_cluster(num_workers=1)
    num_elements = 10
    num_repetitions = 5
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    ds = ds.repeat(num_repetitions)
    self.assertDatasetProduces(
        ds, expected_output=num_repetitions * list(range(num_elements))
    )

  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentEpoch(self):
    # [FUNCTION DOC]
    # Creates three distributed datasets over one worker and reads them in lockstep, one element
    # at a time. Each must independently produce 0..9.
    cluster = self.make_test_cluster(num_workers=1)
    num_elements = 10
    num_datasets = 3
    get_nexts = []
    results = []
    for _ in range(num_datasets):
      ds = self.make_distributed_range_dataset(num_elements, cluster)
      get_nexts.append(self.getNext(ds))
      results.append([])

    for _ in range(num_elements):
      for dataset_ind in range(num_datasets):
        result = self.evaluate(get_nexts[dataset_ind]())
        results[dataset_ind].append(result)
    for result in results:
      self.assertEqual(list(range(num_elements)), result)

  @combinations.generate(test_base.default_test_combinations())
  def testMultiWorker(self):
    # [FUNCTION DOC]
    # With three workers, each worker processes the full range of 10, so the output must contain
    # 0..9 three times (any order).
    num_workers = 3
    cluster = self.make_test_cluster(num_workers=num_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(
        ds, num_workers * list(range(num_elements)), assert_items_equal=True
    )

  @combinations.generate(test_base.default_test_combinations())
  def testFromGenerator(self):
    # [FUNCTION DOC]
    # Distributes (no sharding) a dataset built from a Python generator that yields 0..9 with a
    # scalar int64 signature, and expects 0..9.
    cluster = self.make_test_cluster(num_workers=1)

    def generator():
      # [FUNCTION DOC]
      # HELPER: element source for the enclosing test. Yields the integers 0 through 9.
      yield from range(10)

    dataset = dataset_ops.Dataset.from_generator(
        generator,
        output_signature=tensor_spec.TensorSpec(shape=(), dtype=dtypes.int64),
    )
    dataset = dataset.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.OFF, cluster.dispatcher_address()
        )
    )
    self.assertDatasetProduces(dataset, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testMaxOutstandingRequests(self):
    # [FUNCTION DOC]
    # Three workers, range of 10, with at most one outstanding request. Expects 0..9 three times
    # in any order.
    num_workers = 3
    cluster = self.make_test_cluster(num_workers=num_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, max_outstanding_requests=1
    )
    self.assertDatasetProduces(
        ds, num_workers * list(range(num_elements)), assert_items_equal=True
    )

  @combinations.generate(test_base.eager_only_combinations())
  def testInsideFunction(self):
    # [FUNCTION DOC]
    # Eager only. Iterates a distributed range of 10 (three workers) inside a compiled function
    # that collects elements into a growing array and returns them stacked. The elements must
    # equal 0..9 three times (any order). The nested compiled function is deliberately left
    # without a comment block because the framework parses its source.
    num_workers = 3
    cluster = self.make_test_cluster(num_workers=num_workers)
    num_elements = 10

    @def_function.function
    def f():
      ds = self.make_distributed_range_dataset(num_elements, cluster)
      result = tensor_array_ops.TensorArray(
          dtypes.int64, size=num_workers * num_elements, dynamic_size=True
      )
      i = 0
      for elem in ds:
        result = result.write(i, elem)
        i += 1
      return result.stack()

    result = list(f().numpy())
    self.assertCountEqual(num_workers * list(range(num_elements)), result)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyJobNameDistribute(self):
    # [FUNCTION DOC]
    # Calling distribute with an empty job name must raise a value error saying the job name must
    # not be empty.
    cluster = self.make_test_cluster(num_workers=1)
    with self.assertRaisesRegex(ValueError, "`job_name` must not be empty"):
      dataset_ops.Dataset.range(10).apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs",
              service=cluster.dispatcher.target,
              job_name="",
          )
      )

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyJobNameFromDatasetId(self):
    # [FUNCTION DOC]
    # Registers a dataset, then calling from-dataset-id with an empty job name must raise a value
    # error saying the job name must not be empty.
    cluster = self.make_test_cluster(num_workers=1)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset_ops.Dataset.range(10)
    )
    with self.assertRaisesRegex(ValueError, "`job_name` must not be empty"):
      data_service_ops.from_dataset_id(
          dataset_id=dataset_id,
          processing_mode="parallel_epochs",
          service=cluster.dispatcher.target,
          job_name="",
      )

  @combinations.generate(test_base.default_test_combinations())
  def testNonStringJobNameDistribute(self):
    # [FUNCTION DOC]
    # Passing a tensor as job name to distribute must raise a value error saying the job name
    # must be a string.
    cluster = self.make_test_cluster(num_workers=1)
    with self.assertRaisesRegex(ValueError, "`job_name` must be a string"):
      dataset_ops.Dataset.range(10).apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs",
              service=cluster.dispatcher.target,
              job_name=constant_op.constant("foo"),
          )
      )

  @combinations.generate(test_base.default_test_combinations())
  def testNonStringJobNameFromDatasetId(self):
    # [FUNCTION DOC]
    # Passing a tensor as job name to from-dataset-id (for a registered dataset) must raise a
    # value error saying the job name must be a string.
    cluster = self.make_test_cluster(num_workers=1)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset_ops.Dataset.range(10)
    )
    with self.assertRaisesRegex(ValueError, "`job_name` must be a string"):
      data_service_ops.from_dataset_id(
          dataset_id=dataset_id,
          processing_mode="parallel_epochs",
          service=cluster.dispatcher.target,
          job_name=constant_op.constant("foo"),
      )

  @combinations.generate(test_base.default_test_combinations())
  def testSharedJobName(self):
    # [FUNCTION DOC]
    # Two clients with the same job name over a shuffled range of 1000 split the job's data.
    # Reads alternate between them for a fifth of the elements, then drains both; together the
    # results must contain every element exactly once.
    cluster = self.make_test_cluster(num_workers=1)
    num_elements = 1000

    def make_ds():
      # [FUNCTION DOC]
      # HELPER: dataset factory for the enclosing test. Returns a freshly shuffled range dataset.
      return dataset_ops.Dataset.range(num_elements).shuffle(num_elements)

    ds1 = self.make_distributed_dataset(make_ds(), cluster, job_name="job_name")
    ds2 = self.make_distributed_dataset(make_ds(), cluster, job_name="job_name")
    get_next_1 = self.getNext(ds1)
    get_next_2 = self.getNext(ds2)
    results = []
    for _ in range(num_elements // 5):
      results.append(self.evaluate(get_next_1()))
      results.append(self.evaluate(get_next_2()))
    results += self.getIteratorOutput(get_next_1)
    results += self.getIteratorOutput(get_next_2)
    self.assertCountEqual(list(range(num_elements)), results)

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentJobNames(self):
    # [FUNCTION DOC]
    # Two clients with different job names each receive the full range 0..9.
    cluster = self.make_test_cluster(num_workers=1)
    num_elements = 10
    ds1 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name1"
    )
    ds2 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name2"
    )
    self.assertDatasetProduces(ds1, list(range(num_elements)))
    self.assertDatasetProduces(ds2, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedJobNameMultiIteration(self):
    # [FUNCTION DOC]
    # Eager only. Two datasets share a job name. In iteration one the first dataset yields 0..9
    # and the second yields nothing; in iteration two the second yields 0..9 and the first
    # yields nothing.
    cluster = self.make_test_cluster(num_workers=1)
    num_elements = 10
    ds1 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name"
    )
    ds2 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name"
    )
    # iteration 1
    self.assertDatasetProduces(ds1, list(range(num_elements)))
    self.assertDatasetProduces(ds2, [])
    # iteration 2
    self.assertDatasetProduces(ds2, list(range(num_elements)))
    self.assertDatasetProduces(ds1, [])

  @combinations.generate(test_base.default_test_combinations())
  def testSharedJobNameRepeat(self):
    # [FUNCTION DOC]
    # Two clients share a job name, each repeating a range of 100 three times. Reads a fifth of
    # the total from the first, then from the second, then drains both. The combined results must
    # equal three copies of 0..99 (any order).
    cluster = self.make_test_cluster(num_workers=1)
    num_elements = 100
    num_repetitions = 3
    ds1 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name"
    )
    ds1 = ds1.repeat(num_repetitions)
    ds2 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name"
    )
    ds2 = ds2.repeat(num_repetitions)
    results = []
    get_next_1 = self.getNext(ds1)
    get_next_2 = self.getNext(ds2)
    for _ in range((num_elements * num_repetitions) // 5):
      results.append(self.evaluate(get_next_1()))
    for _ in range((num_elements * num_repetitions) // 5):
      results.append(self.evaluate(get_next_2()))
    results += self.getIteratorOutput(get_next_1)
    results += self.getIteratorOutput(get_next_2)
    self.assertCountEqual(num_repetitions * list(range(num_elements)), results)

  @combinations.generate(test_base.default_test_combinations())
  def testSharedJobNameRepeatEmptyJob(self):
    # [FUNCTION DOC]
    # Two endlessly repeated clients share a job name with no sharding. The first reads three
    # passes of 0..9 in order. The second is then created and must still be able to read three
    # passes' worth of elements (it must not be empty).
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    ds1 = ds.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.OFF,
            cluster.dispatcher_address(),
            job_name="shared_job"))
    ds1 = ds1.repeat()
    ds2 = ds.apply(
        data_service_ops.distribute(
            data_service_ops.ShardingPolicy.OFF,
            cluster.dispatcher_address(),
            job_name="shared_job"))
    ds2 = ds2.repeat()

    get_next_1 = self.getNext(ds1)
    for i in list(range(num_elements)) * 3:
      self.assertEqual(self.evaluate(get_next_1()), i)
    # Verifies ds2 is non-empty.
    get_next_2 = self.getNext(ds2)
    for i in list(range(num_elements)) * 3:
      _ = self.evaluate(get_next_2())

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedJobNameMultipleEpochs(self):
    # [FUNCTION DOC]
    # Eager only. A named-job distributed range of 10 is iterated five times with a fresh
    # iterator each time; each iteration must yield 0..9.
    cluster = self.make_test_cluster(num_workers=1)
    dataset = self.make_distributed_range_dataset(
        10, cluster, job_name="job_name"
    )

    num_epochs = 5
    for _ in range(num_epochs):
      get_next = self.getNext(dataset)
      self.assertEqual(self.getIteratorOutput(get_next), list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testStringDatasetId(self):
    # [FUNCTION DOC]
    # Registers a dataset, converts the id to a string tensor if it is not already one, reads it
    # via from-dataset-id with no sharding and a job name, and expects 0..9.
    """Tests passing a dataset ID of string Tensor."""
    cluster = self.make_test_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset
    )
    dataset_id_str = (
        dataset_id
        if dataset_id.dtype == dtypes.string
        else string_ops.as_string(dataset_id)
    )
    dataset = data_service_ops.from_dataset_id(
        dataset_id=dataset_id_str,
        element_spec=dataset.element_spec,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
        job_name="job_name",
    )
    self.assertDatasetProduces(dataset, list(range(10)))

  @combinations.generate(test_base.eager_only_combinations())
  def testPyStringDatasetId(self):
    # [FUNCTION DOC]
    # Eager only. Registers a dataset, extracts the id's constant value as a plain string
    # (decoding bytes if needed), reads via from-dataset-id and expects 0..9.
    """Tests passing a dataset ID of Python string."""
    cluster = self.make_test_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset
    )
    dataset_id_val = tensor_util.constant_value(dataset_id)
    dataset_id_str = (
        dataset_id_val.decode()
        if isinstance(dataset_id_val, bytes)
        else str(dataset_id_val)
    )
    dataset = data_service_ops.from_dataset_id(
        dataset_id=dataset_id_str,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
        job_name="job_name",
    )
    self.assertDatasetProduces(dataset, list(range(10)))

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(job_name=[None, "test"]),
      )
  )
  def testGcUnusedJob(self, job_name):
    # [FUNCTION DOC]
    # Eager only; parameterized over no job name or a named job. With fast job garbage
    # collection settings, creates an iterator, reads the first element (0), checks one task
    # exists on the worker, drops the iterator and polls until the worker has no tasks left.
    cluster = self.make_test_cluster(
        num_workers=1, job_gc_check_interval_ms=50, job_gc_timeout_ms=20
    )
    num_elements = 100
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, job_name=job_name
    )
    it = iter(ds)
    self.assertEqual(next(it).numpy(), 0)
    self.assertEqual(cluster.workers[0].num_tasks(), 1)
    del it
    while cluster.workers[0].num_tasks() > 0:
      time.sleep(0.1)

  @combinations.generate(test_base.eager_only_combinations())
  def testDontGcUsedJob(self):
    # [FUNCTION DOC]
    # Eager only. Creates iterators for job "test1" and two for "test2" (the third iterator is
    # kept alive), expects two tasks, drops the first two and waits; only the unused "test1" job
    # is collected, leaving one task.
    cluster = self.make_test_cluster(
        num_workers=1, job_gc_check_interval_ms=50, job_gc_timeout_ms=20
    )
    num_elements = 1000
    it1 = iter(
        self.make_distributed_range_dataset(
            num_elements, cluster, job_name="test1"
        )
    )
    it2 = iter(
        self.make_distributed_range_dataset(
            num_elements, cluster, job_name="test2"
        )
    )
    it3 = iter(  # this iterator keeps the task alive. pylint: disable=unused-variable
        self.make_distributed_range_dataset(
            num_elements, cluster, job_name="test2"
        )
    )
    self.assertEqual(cluster.workers[0].num_tasks(), 2)
    del it1
    del it2
    # Check that only the first job is gced. The second job will not be gced
    # because there is still an outstanding iterator for it.
    while cluster.workers[0].num_tasks() > 1:
      time.sleep(0.1)
    self.assertEqual(cluster.workers[0].num_tasks(), 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testDontGcJobsWithVisitationGuarantees(self):
    # [FUNCTION DOC]
    # Eager only. Creates one job with the default policy and one with dynamic sharding, expects
    # two tasks, drops both iterators and waits; only the default job is collected because
    # dynamic sharding gives visitation guarantees, leaving one task.
    cluster = self.make_test_cluster(
        num_workers=1,
        job_gc_check_interval_ms=50,
        job_gc_timeout_ms=20,
    )
    num_elements = 1000
    it1 = iter(
        self.make_distributed_range_dataset(
            num_elements,
            cluster,
            job_name="test1",
        )
    )
    it2 = iter(
        self.make_distributed_range_dataset(
            num_elements,
            cluster,
            job_name="test2",
            processing_mode=data_service_ops.ShardingPolicy.DYNAMIC,
        )
    )
    self.assertEqual(cluster.workers[0].num_tasks(), 2)
    del it1
    del it2
    # Check that only the first job is gced. The second job will not be gced
    # because it has a sharding policy with visitation guarantees.
    while cluster.workers[0].num_tasks() > 1:
      time.sleep(0.1)
    self.assertEqual(cluster.workers[0].num_tasks(), 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testGcAndRecreate(self):
    # [FUNCTION DOC]
    # Eager only. Three times: creates the same named job on a three-worker cluster, reads 50
    # elements, drops the iterator and waits until all workers have garbage-collected its
    # tasks. Verifies a collected job can be recreated.
    cluster = self.make_test_cluster(
        num_workers=3, job_gc_check_interval_ms=50, job_gc_timeout_ms=20
    )
    num_elements = 1000
    # Repeatedly create and garbage-collect the same job.
    for _ in range(3):
      ds = self.make_distributed_range_dataset(
          num_elements, cluster, job_name="test"
      )
      it = iter(ds)
      for _ in range(50):
        next(it)
      del it
      # Wait for the task to be garbage-collected on all workers.
      while cluster.num_tasks_on_workers() > 0:
        time.sleep(0.1)

  @combinations.generate(test_base.eager_only_combinations())
  def testGcClient(self):
    # [FUNCTION DOC]
    # Eager only. Starts a dispatcher with a 50 ms client timeout and a worker, applies the
    # internal distribute op with a 10 second task refresh hint (so the client does not
    # heartbeat). Reads one element, sleeps 3 seconds and keeps reading; expects a not-found error
    # about an unknown iteration client id because the client was garbage collected.
    dispatcher = server_lib.DispatchServer(
        service_config_pb2.DispatcherConfig(
            protocol="grpc",
            job_gc_check_interval_ms=50,
            job_gc_timeout_ms=20,
            client_timeout_ms=50,
        )
    )
    dispatcher_address = dispatcher.target.split("://")[1]
    _ = server_lib.WorkerServer(
        server_lib.WorkerConfig(
            dispatcher_address=dispatcher_address, heartbeat_interval_ms=100
        )
    )

    num_elements = 1000
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.apply(
        data_service_ops._distribute(
            processing_mode=data_service_ops.ShardingPolicy.OFF,
            service=dispatcher.target,
            task_refresh_interval_hint_ms=10000,
        )
    )
    get_next = self.getNext(dataset)

    # The client does not heartbeat in 10 seconds. It will be garbage-collected.
    with self.assertRaisesRegex(
        errors.NotFoundError, "Unknown iteration client id"
    ):
      self.evaluate(get_next())
      time.sleep(3)
      self.getIteratorOutput(get_next)

  @combinations.generate(test_base.eager_only_combinations())
  def testKeepClientAliveBeforeReading(self):
    # [FUNCTION DOC]
    # Eager only. Dispatcher with a 1 second client timeout; the client heartbeats every 100 ms
    # (refresh hint 100 ms). Sleeps 3 seconds before the first read, then expects the whole
    # range of 1000 elements, showing the client was kept alive.
    dispatcher = server_lib.DispatchServer(
        service_config_pb2.DispatcherConfig(
            protocol="grpc",
            job_gc_check_interval_ms=50,
            job_gc_timeout_ms=20,
            client_timeout_ms=1000,
        )
    )
    dispatcher_address = dispatcher.target.split("://")[1]
    _ = server_lib.WorkerServer(
        server_lib.WorkerConfig(
            dispatcher_address=dispatcher_address, heartbeat_interval_ms=100
        )
    )

    num_elements = 1000
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.apply(
        data_service_ops._distribute(
            processing_mode=data_service_ops.ShardingPolicy.OFF,
            service=dispatcher.target,
            task_refresh_interval_hint_ms=100,
        )
    )
    get_next = self.getNext(dataset)

    # The client regularly heartbeats in 100 milliseconds. It should not be
    # garbage-collected even if it does not start reading in 3 seconds.
    time.sleep(3)
    self.assertEqual(
        self.getIteratorOutput(get_next), list(range(num_elements))
    )

  @combinations.generate(test_base.default_test_combinations())
  def testApplyDeterminismOption(self):
    # [FUNCTION DOC]
    # Builds a dataset factory whose interleave delays the element 0 and sets non-deterministic
    # option, distributes it, and uses the base class determinism check to confirm the
    # non-deterministic order is honored while the elements are still the 10 expected ones. The
    # nested compiled interleave function is deliberately left without a comment block.
    elements = list(range(10))
    cluster = self.make_test_cluster(num_workers=1)

    def dataset_fn(delay_ms):
      # [FUNCTION DOC]
      # HELPER: dataset factory passed to the determinism check of the enclosing test.
      # Takes the delay in milliseconds applied to element 0; returns a distributed dataset of
      # ten elements interleaved in parallel with non-deterministic option set.
      @def_function.function
      def interleave_fn(x):
        ds = dataset_ops.Dataset.from_tensors(x)
        if math_ops.equal(x, 0):
          ds = ds.apply(testing.sleep(delay_ms * 1000))
        else:
          ds = ds.apply(testing.sleep(0))
        return ds

      ds = dataset_ops.Dataset.from_tensor_slices(elements)
      ds = ds.interleave(interleave_fn, cycle_length=10, num_parallel_calls=10)
      opts = options_lib.Options()
      opts.deterministic = False
      ds = ds.with_options(opts)
      ds = self.make_distributed_dataset(ds, cluster)
      return ds

    self.checkDeterminism(
        dataset_fn=dataset_fn,
        expect_determinism=False,
        expected_elements=elements,
    )

  def run_stateful(self, external_state_policy):
    # [FUNCTION DOC]
    # HELPER: shared body of the stateful-dataset tests.
    # Builds a range of 10 mapped to random values, sets the given external state policy option,
    # distributes it over three workers and reads all output. Raises whatever error the policy
    # triggers (for the fail policy); returns nothing.
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements).map(
        lambda _: random_ops.random_uniform(())
    )

    options = options_lib.Options()
    options.experimental_external_state_policy = external_state_policy
    ds = ds.with_options(options)

    cluster = self.make_test_cluster(num_workers=3)
    ds = self.make_distributed_dataset(ds, cluster)
    self.getDatasetOutput(ds)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              external_state_policy=[
                  options_lib.ExternalStatePolicy.IGNORE,
                  options_lib.ExternalStatePolicy.WARN,
              ]
          ),
      )
  )
  def testStatefulNoError(self, external_state_policy):
    # [FUNCTION DOC]
    # Parameterized over the ignore and warn policies. Distributing a dataset with a random
    # (stateful) map must complete without error.
    self.run_stateful(external_state_policy)

  @combinations.generate(test_base.default_test_combinations())
  def testStatefulError(self):
    # [FUNCTION DOC]
    # With the fail policy, distributing a stateful dataset must raise a failed-precondition
    # error.
    with self.assertRaises(errors.FailedPreconditionError):
      self.run_stateful(options_lib.ExternalStatePolicy.FAIL)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeFromInterleave(self):
    # [FUNCTION DOC]
    # Interleaves (cycle length 2) over inner datasets, each a distributed two-element range
    # starting at 10 times the outer element. Expects 0, 10, 1, 11. The nested interleave
    # function is deliberately left without a comment block because the framework parses it.
    cluster = self.make_test_cluster(num_workers=1)
    ds = dataset_ops.Dataset.range(2)

    def interleave_fn(x):
      dataset = dataset_ops.Dataset.range(10 * x, 10 * x + 2)
      dataset = self.make_distributed_dataset(dataset, cluster)
      return dataset

    ds = ds.interleave(interleave_fn, cycle_length=2)
    self.assertDatasetProduces(ds, [0, 10, 1, 11])

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeNonStringAddresses(self):
    # [FUNCTION DOC]
    # Passing a non-string service address to distribute must raise a value error saying the
    # service must be a string.
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(ValueError, "`service` must be a string"):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs", service=1
          )
      )

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeEmptyAddress(self):
    # [FUNCTION DOC]
    # An empty service address must raise a value error with the exact message that the service
    # must not be empty.
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesWithLiteralMatch(
        ValueError, "`service` must not be empty"
    ):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs", service=""
          )
      )

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeInvalidProtocol(self):
    # [FUNCTION DOC]
    # Using a misspelled protocol prefix on the dispatcher address must raise a not-found error
    # saying no credentials factory is registered for that protocol, when the output is read.
    cluster = self.make_test_cluster(num_workers=1)
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        errors.NotFoundError,
        "No credentials factory has been registered for protocol grp",
    ):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs",
              service="grp://" + cluster.dispatcher_address(),
          )
      )
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeInvalidProcessingMode(self):
    # [FUNCTION DOC]
    # Eager only. An unrecognized processing mode string must raise a value error listing the
    # accepted sharding policy and mode names.
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        ValueError,
        (
            "should be a `tf.data.experimental.service.ShardingPolicy`, "
            '`"parallel_epochs"`, or '
            "`\"distributed_epoch\"`. Got 'invalid'."
        ),
    ):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="invalid", service="grpc://localhost:5000"
          )
      )

  @combinations.generate(test_base.default_test_combinations())
  def testZipDifferentProcessingModesDatasets(self):
    # [FUNCTION DOC]
    # Zips a range of 100 distributed with distributed-epoch mode and another with parallel-epochs
    # mode (no job names). Expects the pairs (i, i) for all i, in any order.
    cluster = self.make_test_cluster(num_workers=1)
    num_elements = 100
    ds1 = dataset_ops.Dataset.range(num_elements)
    ds1 = self.make_distributed_dataset(
        ds1, cluster, processing_mode="distributed_epoch"
    )
    ds2 = dataset_ops.Dataset.range(num_elements)
    ds2 = self.make_distributed_dataset(
        ds2, cluster, processing_mode="parallel_epochs"
    )
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    self.assertDatasetProduces(
        ds,
        list(zip(range(num_elements), range(num_elements))),
        assert_items_equal=True,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testZipDifferentProcessingModesDatasetsSharedJobName(self):
    # [FUNCTION DOC]
    # Same zip as the previous test but both use one job name; reading must raise an
    # invalid-argument error that an existing job has a different processing mode.
    cluster = self.make_test_cluster(num_workers=1)
    num_elements = 100
    ds1 = dataset_ops.Dataset.range(num_elements)
    ds1 = self.make_distributed_dataset(
        ds1, cluster, processing_mode="distributed_epoch", job_name="job_name"
    )
    ds2 = dataset_ops.Dataset.range(num_elements)
    ds2 = self.make_distributed_dataset(
        ds2, cluster, processing_mode="parallel_epochs", job_name="job_name"
    )
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "but found an existing job with diff"
    ):
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetId(self):
    # [FUNCTION DOC]
    # Registers a range of 10 with the dispatcher, reads it back via from-dataset-id with
    # parallel-epochs and the dataset's element spec, and expects 0..9.
    cluster = self.make_test_cluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher_address(), ds
    )
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs",
        cluster.dispatcher.target,
        dataset_id,
        ds.element_spec,
    )
    self.assertDatasetProduces(from_dataset_id_ds, list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetIdSharedJobs(self):
    # [FUNCTION DOC]
    # Registers two datasets with explicit ids (an int32 range of 20 and the values 20..39) on a
    # two-worker cluster. Creates two readers per dataset sharing one job name with distributed
    # epoch mode, interleaves all of them in parallel, and expects 0..39 (any order). The nested
    # lambda is not documented.
    cluster = self.make_test_cluster(num_workers=2)

    dataset_ids = ["dataset_1", "dataset_2"]
    datasets = [
        dataset_ops.Dataset.range(20, output_type=dtypes.int32),
        dataset_ops.Dataset.from_tensor_slices(list(range(20, 40))),
    ]

    for ds, dataset_id in zip(datasets, dataset_ids):
      # Evaluate to ensure that in graph mode `register_dataset` is called
      # before `from_dataset_id` below.
      self.evaluate(
          data_service_ops.register_dataset(
              cluster.dispatcher_address(), ds, dataset_id=dataset_id
          )
      )

    # Read from both jobs in parallel, with 2 consumers for each job.
    data_service_datasets = []
    for _ in range(2):
      for dataset, dataset_id in zip(datasets, dataset_ids):
        ds = data_service_ops.from_dataset_id(
            "distributed_epoch",
            cluster.dispatcher.target,
            dataset_id,
            dataset.element_spec,
            job_name="shared_job",
        )
        data_service_datasets.append(ds)
    ds = dataset_ops.Dataset.from_tensor_slices(data_service_datasets)
    ds = ds.interleave(lambda x: x, cycle_length=len(data_service_datasets))

    self.assertDatasetProduces(ds, list(range(40)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testRegisteringDatasetAsTfFunction(self):
    # [FUNCTION DOC]
    # Wraps the register-dataset op in a compiled function, passing the protocol and address as
    # constants, then reads the registered dataset via from-dataset-id and expects 0..9.
    cluster = self.make_test_cluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    register_func = def_function.function(data_service_ops.register_dataset)
    dataset_id = register_func(
        (
            constant_op.constant("grpc"),
            constant_op.constant(cluster.dispatcher_address()),
        ),
        ds,
    )
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs",
        cluster.dispatcher.target,
        dataset_id,
        ds.element_spec,
    )
    self.assertDatasetProduces(from_dataset_id_ds, list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetIdMultipleComponents(self):
    # [FUNCTION DOC]
    # Registers a zipped nested structure (a pair under key a and a single under key b, all from
    # a range of 10), reads it via from-dataset-id and checks every component of each element
    # equals its index.
    cluster = self.make_test_cluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    ds = dataset_ops.Dataset.zip({"a": (ds, ds), "b": ds})
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher_address(), ds
    )
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs",
        cluster.dispatcher.target,
        dataset_id,
        ds.element_spec,
    )
    output = self.getDatasetOutput(from_dataset_id_ds)
    for i in range(num_elements):
      self.assertEqual(i, output[i]["a"][0])
      self.assertEqual(i, output[i]["a"][1])
      self.assertEqual(i, output[i]["b"])

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetIdWrongElementSpec(self):
    # [FUNCTION DOC]
    # Reads a registered range of 10 using a wrong (variant scalar) element spec; fetching the
    # first element must raise some error.
    cluster = self.make_test_cluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher_address(), ds
    )
    wrong_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.variant)
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs", cluster.dispatcher.target, dataset_id, wrong_spec
    )

    with self.assertRaises(Exception):
      self.evaluate(self.getNext(from_dataset_id_ds)())

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetIdNotRegistered(self):
    # [FUNCTION DOC]
    # Reading with an id that was never registered must raise a not-found error naming the
    # dataset id.
    cluster = self.make_test_cluster(num_workers=1)
    dataset_id = "UnregisteredID"
    element_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.variant)
    with self.assertRaisesRegex(
        errors.NotFoundError, f"Dataset id {dataset_id} not found."
    ):
      from_dataset_id_ds = data_service_ops.from_dataset_id(
          data_service_ops.ShardingPolicy.OFF,
          cluster.dispatcher.target,
          dataset_id,
          element_spec,
      )
      self.evaluate(self.getNext(from_dataset_id_ds)())

  @combinations.generate(test_base.default_test_combinations())
  def testCancellation(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is skipped immediately (bug reference b/162521601), so the code after
    # the skip never runs.
    # Intended scenario: a dataset with a quick first element and a very slow second one is
    # distributed and prefetched; after reading the first element, destroying the iterator must
    # cancel the pending slow fetch instead of hanging.
    self.skipTest("b/162521601")
    sleep_microseconds = int(1e6) * 1000

    cluster = self.make_test_cluster(num_workers=1)
    # Create a dataset which produces the first element quickly, and the second
    # element slowly. Fetching the first element triggers prefetching of the
    # second element, which we should be able to cancel.
    slow = dataset_ops.Dataset.range(1)
    slow = slow.apply(testing.sleep(sleep_microseconds))
    ds = dataset_ops.Dataset.range(1).concatenate(slow)
    ds = self.make_distributed_dataset(ds, cluster)
    ds = ds.prefetch(1)
    get_next = self.getNext(ds)
    self.assertEqual(0, self.evaluate(get_next()))
    # Without properly implemented cancellation, we will hang here while trying
    # to garbage collect the dataset iterator.

  @combinations.generate(test_base.default_test_combinations())
  def testRegisterDifferentDatasets(self):
    # [FUNCTION DOC]
    # Registering a range of 10 and a range of 20 must yield different dataset ids.
    ds_1 = dataset_ops.Dataset.range(10)
    ds_2 = dataset_ops.Dataset.range(20)
    cluster = self.make_test_cluster(num_workers=1)
    id_1 = data_service_ops.register_dataset(cluster.dispatcher_address(), ds_1)
    id_2 = data_service_ops.register_dataset(cluster.dispatcher_address(), ds_2)
    self.assertNotEqual(self.evaluate(id_1), self.evaluate(id_2))

  @combinations.generate(test_base.default_test_combinations())
  def testRegisterWithExplicitDatasetId(self):
    # [FUNCTION DOC]
    # Registers a range of 10 under the explicit id "dataset_id", reads it with the returned id
    # and again with the literal string id; both reads must give 0..9.
    cluster = self.make_test_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset, dataset_id="dataset_id"
    )
    dataset = data_service_ops.from_dataset_id(
        dataset_id=dataset_id,
        element_spec=dataset.element_spec,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
    )
    self.assertDatasetProduces(dataset, list(range(10)))

    # Verifies the dataset ID is indeed "dataset_id".
    dataset = data_service_ops.from_dataset_id(
        dataset_id="dataset_id",
        element_spec=dataset.element_spec,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
    )
    self.assertDatasetProduces(dataset, list(range(10)))

  # Eager mode only: In the graph mode, `register_dataset` may not run before
  # `from_dataset_id` if `from_dataset_id` does not use its return value.
  @combinations.generate(test_base.eager_only_combinations())
  def testFromRegisteredStringDatasetId(self):
    # [FUNCTION DOC]
    # Eager only (graph mode may not run registration before use if the return value is
    # ignored). Registers a range of 10 under "dataset_id", discards the returned id, reads by
    # the literal id and expects 0..9.
    cluster = self.make_test_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(10)
    _ = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset, dataset_id="dataset_id"
    )
    dataset = data_service_ops.from_dataset_id(
        dataset_id="dataset_id",
        element_spec=dataset.element_spec,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
    )
    self.assertDatasetProduces(dataset, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testRegisterSameDatasetIds(self):
    # [FUNCTION DOC]
    # Registers two identical ranges under the same explicit id, then reads both with one shared
    # job name. The first reader gets 0..9 and the second gets nothing because the shared job
    # is exhausted.
    cluster = self.make_test_cluster(num_workers=1)
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10)
    dataset_id1 = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset1, dataset_id="dataset_id"
    )
    dataset_id2 = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset2, dataset_id="dataset_id"
    )
    dataset1 = data_service_ops.from_dataset_id(
        dataset_id=dataset_id1,
        element_spec=dataset1.element_spec,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
        job_name="job_name",
    )
    dataset2 = data_service_ops.from_dataset_id(
        dataset_id=dataset_id2,
        element_spec=dataset2.element_spec,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
        job_name="job_name",
    )
    # `dataset2` is empty because the datasets share the same job and `dataset1`
    # has exhausted the dataset.
    self.assertDatasetProduces(dataset1, list(range(10)))
    self.assertDatasetProduces(dataset2, list())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              different_dataset_id=[None, "another_dataset_id"]
          ),
      )
  )
  def testRegisterDifferentDatasetIds(self, different_dataset_id):
    # [FUNCTION DOC]
    # Parameterized over the second dataset's id (none, i.e. generated, or another explicit id).
    # Registers two ranges of 10 with different ids and reads both under one job name; since the
    # ids differ the datasets are distinct and each yields 0..9.
    cluster = self.make_test_cluster(num_workers=1)
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10)
    dataset_id1 = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset1, dataset_id="dataset_id"
    )
    dataset_id2 = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset2, dataset_id=different_dataset_id
    )
    dataset1 = data_service_ops.from_dataset_id(
        dataset_id=dataset_id1,
        element_spec=dataset1.element_spec,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
        job_name="job_name",
    )
    dataset2 = data_service_ops.from_dataset_id(
        dataset_id=dataset_id2,
        element_spec=dataset2.element_spec,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        service=cluster.dispatcher.target,
        job_name="job_name",
    )
    # `dataset1` and `dataset2` are different datasets.
    self.assertDatasetProduces(dataset1, list(range(10)))
    self.assertDatasetProduces(dataset2, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetsDoNotMatch(self):
    # [FUNCTION DOC]
    # Registers a numeric range and a string dataset under the same id without compression; the
    # registration must raise an invalid-argument error that datasets with the same id need the
    # same structure.
    cluster = self.make_test_cluster(num_workers=1)
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.from_tensor_slices(list("Test dataset."))

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Datasets with the same ID should have the same structure",
    ):
      dataset_id1 = data_service_ops.register_dataset(
          cluster.dispatcher.target,
          dataset1,
          compression=None,
          dataset_id="dataset_id",
      )
      dataset_id2 = data_service_ops.register_dataset(
          cluster.dispatcher.target,
          dataset2,
          compression=None,
          dataset_id="dataset_id",
      )
      dataset1 = data_service_ops.from_dataset_id(
          dataset_id=dataset_id1,
          element_spec=dataset1.element_spec,
          processing_mode=data_service_ops.ShardingPolicy.OFF,
          service=cluster.dispatcher.target,
      )
      dataset2 = data_service_ops.from_dataset_id(
          dataset_id=dataset_id2,
          element_spec=dataset2.element_spec,
          processing_mode=data_service_ops.ShardingPolicy.OFF,
          service=cluster.dispatcher.target,
      )
      self.getDatasetOutput(dataset1)
      self.getDatasetOutput(dataset2)

  @combinations.generate(test_base.default_test_combinations())
  def testDoubleDistribute(self):
    # [FUNCTION DOC]
    # Distributing an already distributed dataset a second time still yields 0..9.
    cluster = self.make_test_cluster(num_workers=1)
    ds = self.make_distributed_range_dataset(num_elements=10, cluster=cluster)
    ds = self.make_distributed_dataset(dataset=ds, cluster=cluster)
    self.assertDatasetProduces(ds, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testTwoLevelDistribute(self):
    # [FUNCTION DOC]
    # Shuffles strings of ten distinct lengths (five copies each), distributes over a three-worker
    # cluster, groups by string length into windows large enough to hold all copies from all
    # workers and batches them, then distributes over a second cluster. Each batch element read
    # must be repeated identically once per first-cluster worker, and nothing remains afterward.
    # The nested key function is deliberately left without a comment block because the
    # framework parses it.
    cluster_1_size = 3
    cluster_1 = self.make_test_cluster(num_workers=cluster_1_size)
    cluster_2 = self.make_test_cluster(num_workers=1)
    num_sizes = 10
    size_repeats = 5
    strings = ["a" * i for i in range(num_sizes)] * size_repeats
    ds = dataset_ops.Dataset.from_tensor_slices(strings)
    ds = ds.shuffle(len(strings))
    ds = self.make_distributed_dataset(ds, cluster_1)
    # Large enough so that all strings of the same size are windowed together.
    window_size = cluster_1_size * size_repeats
    batch_size = size_repeats

    def key_func(x):
      return math_ops.cast(string_ops.string_length_v2(x), dtypes.int64)

    ds = ds.apply(
        grouping.group_by_window(
            key_func=key_func,
            reduce_func=lambda _, x: x.batch(batch_size),
            window_size=window_size,
        )
    )
    ds = self.make_distributed_dataset(ds, cluster_2)

    get_next = self.getNext(ds)
    for _ in range(num_sizes):
      element = self.evaluate(get_next())
      for _ in range(1, cluster_1_size):
        self.assertAllEqual(self.evaluate(get_next()), element)
    self.assertEmpty(self.getIteratorOutput(get_next))

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeLargeGraph(self):
    # [FUNCTION DOC]
    # Distributes a dataset embedding a 2x1000x1000 float tensor (larger than the 4MB default
    # message limit) with no work directory and no fault tolerance, and expects that tensor.
    cluster = self.make_test_cluster(
        num_workers=1, work_dir=NO_WORK_DIR, fault_tolerant_mode=False
    )
    # Larger than default OSS grpc message size limit of 4MB.
    tensor = array_ops.ones((2, 1000, 1000), dtype=dtypes.float32)
    ds = dataset_ops.Dataset.from_tensors(tensor)
    ds = self.make_distributed_dataset(ds, cluster)
    self.assertDatasetProduces(ds, [tensor])

  @combinations.generate(test_base.default_test_combinations())
  def testBatchDropsAllElements(self):
    # [FUNCTION DOC]
    # Batching a range of 10 into batches of 1000 with remainder dropping produces nothing even
    # across two workers with no sharding; expects an empty output.
    cluster = self.make_test_cluster(
        num_workers=2, fault_tolerant_mode=False
    )
    dataset = dataset_ops.Dataset.range(10).batch(1000, drop_remainder=True)
    dataset = self.make_distributed_dataset(
        dataset, cluster, processing_mode=data_service_ops.ShardingPolicy.OFF
    )
    self.assertDatasetProduces(dataset, [])

  @combinations.generate(test_base.default_test_combinations())
  def testBatchDoesNotDropRemainder(self):
    # [FUNCTION DOC]
    # Batching a range of 10 into batches of 1000 keeping the remainder, with two workers and no
    # sharding, yields one batch of 0..9 per worker.
    num_workers = 2
    cluster = self.make_test_cluster(
        num_workers=num_workers, fault_tolerant_mode=False
    )
    dataset = dataset_ops.Dataset.range(10).batch(1000, drop_remainder=False)
    dataset = self.make_distributed_dataset(
        dataset, cluster, processing_mode=data_service_ops.ShardingPolicy.OFF
    )
    self.assertDatasetProduces(dataset, [list(range(10))] * num_workers)

  @combinations.generate(
      combinations.times(
          test_base.graph_only_combinations(),
          combinations.combine(use_resource=False),
      )
      + combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(use_resource=True),
      )
  )
  def testVariables(self, use_resource):
    # [FUNCTION DOC]
    # Parameterized over legacy (graph-only) or resource variables. Creates a variable with value
    # 10, maps a range of 3 by adding it, distributes, initializes the variable and expects
    # 10, 11, 12.
    cluster = self.make_test_cluster(num_workers=1)
    if not use_resource:
      with variable_scope.variable_scope("foo", use_resource=False):
        v = variable_v1.VariableV1(10, dtype=dtypes.int64)
    else:
      v = variables.Variable(10, dtype=dtypes.int64)

    ds = dataset_ops.Dataset.range(3)
    ds = ds.map(lambda x: x + v)
    ds = self.make_distributed_dataset(ds, cluster)
    self.evaluate(v.initializer)
    self.assertDatasetProduces(
        ds, list(range(10, 13)), requires_initialization=True
    )

  @combinations.generate(test_base.default_test_combinations())
  def testNoShardingPolicy(self):
    # [FUNCTION DOC]
    # A range of 20 distributed with sharding turned off over one worker yields 0..19.
    cluster = self.make_test_cluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
    )
    self.assertDatasetProduces(dataset, list(range(20)))


class DataServiceOpsGrpcDataTransferTest(
    data_service_test_base.TestBase, parameterized.TestCase
):
  # [CLASS DOC]
  # Test case for tf.data service reads that use the gRPC data transfer protocol explicitly or
  # compression settings, over small in-process clusters.

  @combinations.generate(test_base.default_test_combinations())
  def testExplicitProtocolFromDatasetId(self):
    # [FUNCTION DOC]
    # Cluster configured for gRPC data transfer. Registers a range of 10 and reads it via
    # from-dataset-id passing the gRPC transfer protocol explicitly; expects 0..9.
    cluster = self.make_test_cluster(
        num_workers=1, data_transfer_protocol="grpc"
    )
    range_ds = dataset_ops.Dataset.range(10)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher.target, range_ds
    )
    ds = data_service_ops.from_dataset_id(
        dataset_id=dataset_id,
        processing_mode="parallel_epochs",
        element_spec=range_ds.element_spec,
        service=cluster.dispatcher.target,
        data_transfer_protocol="grpc",
    )
    self.assertDatasetProduces(ds, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeExplicitProtocol(self):
    # [FUNCTION DOC]
    # Cluster configured for gRPC data transfer. Applies distribute with the service address
    # prefixed by the grpc scheme to a range of 10 and expects 0..9.
    cluster = self.make_test_cluster(
        num_workers=1, data_transfer_protocol="grpc"
    )
    ds = dataset_ops.Dataset.range(10)
    ds = ds.apply(
        data_service_ops.distribute(
            processing_mode="parallel_epochs",
            service="grpc://" + cluster.dispatcher_address(),
        )
    )
    self.assertDatasetProduces(ds, list(range(10)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[None, "AUTO"]),
      )
  )
  def testDistributeCompression(self, compression):
    # [FUNCTION DOC]
    # Parameterized over compression none or automatic. Distributes a range of 10 with that
    # setting and expects 0..9.
    cluster = self.make_test_cluster(num_workers=1)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, compression=compression
    )
    self.assertDatasetProduces(ds, list(range(num_elements)))


if __name__ == "__main__":
  test.main()
