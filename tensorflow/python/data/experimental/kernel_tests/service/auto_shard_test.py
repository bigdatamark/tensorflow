# [MODULE DOC]
# Tests for static auto-sharding of datasets with the tf.data service. Uses a multi-process
# cluster helper to start five (or other numbers of) workers with one in-test local worker at a
# chosen position, then checks which elements/records each sharding policy (OFF, DATA, FILE,
# FILE_OR_DATA, HINT) gives that worker and the errors raised on misconfiguration. Contains a
# module-level cluster-building helper, one test class and a main entry that uses the
# multi-process test main.
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
"""Tests auto-sharding datasets with tf.data service."""

from absl.testing import parameterized

from tensorflow.core.protobuf import data_service_pb2
from tensorflow.python.data.experimental.kernel_tests.service import multi_process_cluster
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.kernel_tests.service.multi_process_cluster import MultiProcessCluster
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors


def _make_service_cluster(
    num_workers,
    local_shard_index,
    worker_addresses=None,
    deployment_mode=data_service_pb2.DEPLOYMENT_MODE_COLOCATED):
  # [FUNCTION DOC]
  # HELPER: builds a multi-process tf.data service cluster for the auto-shard tests.
  # Parameters: num_workers is the total number of workers; local_shard_index is the position
  # (zero based) at which the single in-test local worker is started, remote workers being started
  # before and after it; worker_addresses is an optional list of worker address strings (defaults
  # to "localhost" repeated num_workers times); deployment_mode is the deployment mode enum
  # (default colocated; a null value means unspecified). Creates the cluster with no workers, then
  # starts local_shard_index remote workers, one local worker and the remaining remote workers.
  # Returns the cluster object. Side effect: spawns worker processes.
  if worker_addresses is None:
    worker_addresses = ["localhost" for _ in range(num_workers)]

  cluster = MultiProcessCluster(
      num_local_workers=0,
      num_remote_workers=0,
      worker_addresses=worker_addresses,
      deployment_mode=deployment_mode)
  for _ in range(local_shard_index):
    cluster.start_remote_worker()
  cluster.start_local_worker()
  for _ in range(num_workers - local_shard_index - 1):
    cluster.start_remote_worker()
  return cluster


# pylint:disable=g-complex-comprehension
class AutoShardTest(data_service_test_base.TestBase,
                    tf_record_test_base.TFRecordTestBase,
                    parameterized.TestCase):
  # [CLASS DOC]
  # Test case for static auto-sharding of datasets with the tf.data service. Each test builds a
  # five-worker (or other) cluster whose local worker has a specific shard index and checks which
  # elements, files or records that worker's shard produces, or which error is raised, for the
  # various sharding policies. Fixtures: ten TFRecord files with ten records each.
  """Tests auto-sharding datasets with tf.data service."""

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the parent setUp, sets file and record counts to ten each and creates
    # the TFRecord test files, storing their names. Takes no arguments, returns nothing.
    super(AutoShardTest, self).setUp()
    self._num_files = 10
    self._num_records = 10
    self._filenames = self._createFiles()

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.DATA,
              data_service_ops.ShardingPolicy.FILE_OR_DATA
          ])))
  def testRangeDataset_AutoShard(self, sharding_policy):
    # [FUNCTION DOC]
    # Scenario: range(20) distributed with a DATA or FILE_OR_DATA policy on a five-worker cluster
    # where the local worker has index 1. Expects the shard elements 1, 6, 11, 16.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)
    self.assertDatasetProduces(dataset, [1, 6, 11, 16])

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_FileShard(self):
    # [FUNCTION DOC]
    # Scenario: range(20) with the FILE policy (not shardable by file). Reading raises a
    # not-found error saying an unshardable source dataset was found.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE)
    with self.assertRaisesRegex(errors.NotFoundError,
                                "Found an unshardable source dataset"):
      self.getDatasetOutput(dataset)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(worker_index=[distribute.SHARD_HINT, 0, 5])))
  def testRangeDataset_ShardHint(self, worker_index):
    # [FUNCTION DOC]
    # Scenario: range(20) with an explicit shard transformation using the shard hint as the number
    # of shards and a given index (the hint, 0 or 5) under the HINT policy. The index is ignored,
    # so the local worker (index 1 of 5) produces 1, 6, 11, 16.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    # With HINT sharding, `num_shards` should be `SHARD_HINT`; `index` can be
    # any value.
    dataset = dataset.shard(
        num_shards=distribute.SHARD_HINT, index=worker_index)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT)
    self.assertDatasetProduces(dataset, [1, 6, 11, 16])

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_InvalidWorkerIndexUsingShardHint(self):
    # [FUNCTION DOC]
    # Scenario: the shard hint is wrongly passed as the shard index with five shards. Expects an
    # invalid-argument error saying the index must be between 0 and 4 (currently -1).
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    # With HINT sharding, `SHARD_HINT` should be passed to `num_shards`, not
    # `index`.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Index must be between 0 and 4 \(currently index = -1\)."):
      dataset = dataset.shard(num_shards=5, index=distribute.SHARD_HINT)
      dataset = self.make_distributed_dataset(
          dataset,
          cluster=cluster,
          processing_mode=data_service_ops.ShardingPolicy.HINT)
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_NoShardHint(self):
    # [FUNCTION DOC]
    # Scenario: HINT policy but the dataset has a normal shard (1 shard, index 0) and no hint, so
    # the given arguments apply: the full range 0..19 is produced.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    # No SHARD_HINT is provided. The given sharding arguments will be used.
    dataset = dataset.shard(num_shards=1, index=0)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT)
    self.assertDatasetProduces(dataset, list(range(20)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.OFF,
              data_service_ops.ShardingPolicy.FILE_OR_DATA
          ])))
  def testRangeDataset_ShardHintUsedInWrongShardingPolicy(
      self, sharding_policy):
    # [FUNCTION DOC]
    # Scenario: a shard-hint shard transformation used with the OFF or FILE_OR_DATA policy.
    # Reading raises a failed-precondition error saying the hint is only valid with the HINT
    # processing mode.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = dataset.shard(distribute.SHARD_HINT, distribute.SHARD_HINT)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)
    with self.assertRaisesRegex(
        errors.FailedPreconditionError, "tf.data service with "
        "`tf.data.experimental.service.ShardingPolicy.HINT` processing mode."):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_NoShard(self):
    # [FUNCTION DOC]
    # Scenario: policy OFF targeting local workers only: no sharding, the whole range 0..19 is
    # produced by the local worker.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        target_workers="LOCAL")
    self.assertDatasetProduces(dataset, list(range(20)))

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_OneWorker(self):
    # [FUNCTION DOC]
    # Scenario: one-worker cluster, FILE_OR_DATA policy: the single shard is the complete range
    # 0..19.
    """Makes sure shards from all workers form the complete dataset."""
    cluster = _make_service_cluster(num_workers=1, local_shard_index=0)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(dataset, list(range(20)))

  @combinations.generate(test_base.default_test_combinations())
  def testRangeDataset_ReadFromAnyWorker(self):
    # [FUNCTION DOC]
    # Scenario: cluster created with unspecified deployment mode, so the client reads from any
    # worker; with FILE_OR_DATA the full range 0..19 is produced (order ignored).
    # When deployment mode is unspecified, the client will read from any worker.
    cluster = _make_service_cluster(
        num_workers=5, local_shard_index=1, deployment_mode=None)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(
        dataset, list(range(20)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.FILE_OR_DATA,
              data_service_ops.ShardingPolicy.FILE
          ])))
  def testTFRecordDataset_AutoShard(self, sharding_policy):
    # [FUNCTION DOC]
    # Scenario: files listed unshuffled and read by a flat map of TFRecord readers, FILE_OR_DATA
    # or FILE policy, local worker index 3 of 5, local workers targeted. Expects all records of
    # files 3 and 8 in order.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=sharding_policy,
        target_workers="LOCAL")

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in (3, 8)
        for record in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.FILE_OR_DATA,
              data_service_ops.ShardingPolicy.FILE
          ])))
  def testTFRecordDataset_ShuffleFileList(self, sharding_policy):
    # [FUNCTION DOC]
    # Scenario: like auto shard but the file list is shuffled; the shard still holds files 3 and 8
    # and all their records (compared ignoring order).
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=True)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in (3, 8)
        for record in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_DataShard(self):
    # [FUNCTION DOC]
    # Scenario: DATA policy on TFRecord files: every file is read but only records 3 and 8 of each
    # (shard index 3 of 5), in file order.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.DATA)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 10)
        for record in (3, 8)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_HintDataShard(self):
    # [FUNCTION DOC]
    # Scenario: HINT policy with the shard hint placed after the record reading flat map, so the
    # shard is by record: records 3 and 8 of every file are produced in order.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = dataset.shard(distribute.SHARD_HINT, distribute.SHARD_HINT)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 10)
        for record in (3, 8)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_HintFileShard(self):
    # [FUNCTION DOC]
    # Scenario: HINT policy with the shard hint placed on the file list before reading, so the
    # shard is by file: all records of files 3 and 8 are produced in order.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.shard(distribute.SHARD_HINT, distribute.SHARD_HINT)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in (3, 8)
        for record in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_NoShard(self):
    # [FUNCTION DOC]
    # Scenario: OFF policy on local workers: every record of every file is produced in order.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.OFF,
        target_workers="LOCAL")

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 10)
        for record in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_ReadFromAnyWorker(self):
    # [FUNCTION DOC]
    # Scenario: unspecified deployment mode with FILE_OR_DATA: all records of all files are
    # produced (order ignored), i.e. the client reads from any worker.
    # When deployment mode is unspecified, the client will read from any worker.
    cluster = _make_service_cluster(
        num_workers=5, local_shard_index=3, deployment_mode=None)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 10)
        for record in range(0, 10)
    ]
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.FILE_OR_DATA,
              data_service_ops.ShardingPolicy.FILE
          ])))
  def testTFRecordDataset_FewerFilesThanWorkers(self, sharding_policy):
    # [FUNCTION DOC]
    # Scenario: only four files for five workers with FILE_OR_DATA or FILE policy. Reading raises
    # an invalid-argument error saying the files are not enough for the required 5 shards.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames[:4], shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "not enough for the required 5 shards/workers."):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_FewerFilesThanWorkers_HintShard(self):
    # [FUNCTION DOC]
    # Scenario: four files, shard hint applied on the file list, HINT policy: reading raises the
    # same not-enough-files invalid-argument error for 5 shards.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames[:4], shuffle=False)
    dataset = dataset.shard(distribute.SHARD_HINT, distribute.SHARD_HINT)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.HINT)

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "not enough for the required 5 shards/workers."):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testTFRecordDataset_FewerFilesThanWorkers_DataShard(self):
    # [FUNCTION DOC]
    # Scenario: four files with DATA policy works: records 3 and 8 of each of the four files are
    # produced (order ignored).
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames[:4], shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.DATA)

    expected = [
        b"Record %d of file %d" % (record, file)
        for file in range(0, 4)
        for record in (3, 8)
    ]
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(sharding_policy=[
              data_service_ops.ShardingPolicy.FILE_OR_DATA,
              data_service_ops.ShardingPolicy.DATA
          ])))
  def testBatchDataset(self, sharding_policy):
    # [FUNCTION DOC]
    # Scenario: range(20) batched by 3 (keeping the remainder) with FILE_OR_DATA or DATA policy;
    # the shard of worker 1 holds the batches [3,4,5] and [18,19].
    cluster = _make_service_cluster(num_workers=5, local_shard_index=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = dataset.batch(batch_size=3, drop_remainder=False)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)
    self.assertDatasetProduces(dataset, [[3, 4, 5], [18, 19]])

  @combinations.generate(test_base.default_test_combinations())
  def testInterleaveDataset(self):
    # [FUNCTION DOC]
    # Scenario: files read through a parallel interleave (cycle length 10), prefetched, distributed
    # with FILE_OR_DATA, prefetched again. Worker index 3 gets files 3 and 8; records interleave
    # across the two files, record number outer and file inner.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.interleave(
        readers.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)

    expected = [
        b"Record %d of file %d" % (record, file)
        for record in range(0, 10)
        for file in (3, 8)
    ]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testZipDataset(self):
    # [FUNCTION DOC]
    # Scenario: two identical interleaved TFRecord datasets zipped and distributed with
    # FILE_OR_DATA. Expects pairs of identical records from files 3 and 8, interleaved by record.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset1 = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset1 = dataset1.interleave(
        readers.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset2 = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset2 = dataset2.interleave(
        readers.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)

    expected = [(b"Record %d of file %d" % (record, file),
                 b"Record %d of file %d" % (record, file))
                for record in range(0, 10)
                for file in (3, 8)]
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testConcatenateDataset(self):
    # [FUNCTION DOC]
    # Scenario: two identical interleaved TFRecord datasets concatenated and distributed with
    # FILE_OR_DATA. Expects the shard records (files 3 and 8, interleaved) twice in a row.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset1 = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset1 = dataset1.interleave(
        readers.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset2 = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset2 = dataset2.interleave(
        readers.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset1.concatenate(dataset2)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)

    expected = [
        b"Record %d of file %d" % (record, file)
        for record in range(0, 10)
        for file in (3, 8)
    ]
    expected += expected
    self.assertDatasetProduces(dataset, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDataset(self):
    # [FUNCTION DOC]
    # Scenario: an empty range distributed with FILE_OR_DATA produces no elements.
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.range(0)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(dataset, [])

  @combinations.generate(test_base.default_test_combinations())
  def testAnonymousPorts(self):
    # [FUNCTION DOC]
    # Scenario: worker addresses use an anonymous port placeholder; the local worker (index 3)
    # must still resolve its position and produce 3, 8, 13, 18.
    cluster = _make_service_cluster(
        num_workers=5,
        local_shard_index=3,
        worker_addresses=["localhost:%port%" for _ in range(5)])
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(dataset, [3, 8, 13, 18])

  @combinations.generate(test_base.default_test_combinations())
  def testNamedPorts(self):
    # [FUNCTION DOC]
    # Scenario: worker addresses use a named port placeholder; the local worker (index 3) still
    # resolves its position and produces 3, 8, 13, 18.
    cluster = _make_service_cluster(
        num_workers=5,
        local_shard_index=3,
        worker_addresses=["localhost:%port_worker%" for _ in range(5)])
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    self.assertDatasetProduces(dataset, [3, 8, 13, 18])

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidPorts(self):
    # [FUNCTION DOC]
    # Scenario: worker addresses contain an invalid (non-numeric) port. Creating the cluster must
    # raise a runtime error saying the worker's address is not configured.
    with self.assertRaisesRegex(RuntimeError,
                                "The worker's address is not configured"):
      _ = _make_service_cluster(
          num_workers=5,
          local_shard_index=0,
          worker_addresses=["localhost:worker" for _ in range(5)])

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyWorkerList(self):
    # [FUNCTION DOC]
    # Scenario: an empty configured worker address list. Reading raises a not-found error that the
    # worker is not in the workers list.
    cluster = _make_service_cluster(
        num_workers=5, local_shard_index=1, worker_addresses=[])
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    with self.assertRaisesRegex(errors.NotFoundError,
                                "Worker .* is not in the workers list."):
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testWorkerNotFound(self):
    # [FUNCTION DOC]
    # Scenario: configured addresses name fake workers that do not match the real worker. Creating
    # the cluster raises a runtime error saying the worker's address is not configured.
    worker_addresses = [f"fake_worker_{i}" for i in range(5)]
    with self.assertRaisesRegex(RuntimeError,
                                "The worker's address is not configured"):
      _ = _make_service_cluster(
          num_workers=5, local_shard_index=0, worker_addresses=worker_addresses)

  @combinations.generate(test_base.default_test_combinations())
  def testMoreWorkersThanConfigured(self):
    # [FUNCTION DOC]
    # Scenario: five workers started but only one address configured. Creating the cluster raises
    # a runtime error saying other workers are already running at the configured host.
    worker_addresses = ["localhost:%port%"]
    with self.assertRaisesRegex(
        RuntimeError,
        "other workers are already running at the configured host"):
      _ = _make_service_cluster(
          num_workers=5, local_shard_index=1, worker_addresses=worker_addresses)

  @combinations.generate(test_base.default_test_combinations())
  def testNoLocalWorkers(self):
    # [FUNCTION DOC]
    # Scenario: a cluster with only remote workers and the FILE_OR_DATA policy. Reading raises an
    # invalid-argument error that static sharding requires local workers.
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=0, num_remote_workers=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Static sharding policy <FILE_OR_DATA> requires local tf.data workers"):
      self.getDatasetOutput(dataset)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              sharding_policy=list(data_service_ops.ShardingPolicy))))
  def testEnumerateShardingPolicies(self, sharding_policy):
    # [FUNCTION DOC]
    # Scenario: every sharding policy in the enumeration is tried on a TFRecord pipeline with
    # local worker index 3 of 5; reading all output must succeed without errors (no values checked).
    """Verifies tf.data service handles every sharding policy with no errors."""
    cluster = _make_service_cluster(num_workers=5, local_shard_index=3)
    dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
    dataset = dataset.flat_map(readers.TFRecordDataset)
    dataset = self.make_distributed_dataset(
        dataset, cluster=cluster, processing_mode=sharding_policy)
    self.getDatasetOutput(dataset)


if __name__ == "__main__":
  multi_process_cluster.test_main()
