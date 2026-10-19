# [MODULE DOC]
# Tests for the distributed save op of the tf.data service. Covers save followed by load for
# different worker counts, element counts, compression modes, repeated datasets, choose-from
# datasets, multiple snapshots, custom reader functions, distributed loading with sharding
# policies, zipped datasets of unequal size, error cases (missing snapshot, duplicate save,
# failing worker, bad dispatcher address, infinite dataset, wrong element spec or compression,
# non fault tolerant cluster) and checkpointing of loaded datasets. Also defines two private
# helper functions that poll for the snapshot done and error marker files. Uses the shared data
# service test base. Runs under the standard test main entry point.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.data.experimental.distributed_save."""

import os
import time

from absl.testing import parameterized

import numpy as np
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distributed_save_op
from tensorflow.python.data.experimental.service import _pywrap_snapshot_utils
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DistributedSaveTest(
    data_service_test_base.TestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the distributed save op. Each test starts an in-process test cluster, uses a
  # temporary snapshot directory, saves a dataset through the dispatcher and checks the loaded
  # output or the raised error.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3], num_elements=[0, 10, 1000])))
  def testSaveLoad(self, num_workers, num_elements):
    # [FUNCTION DOC]
    # Saves a range dataset with the given worker and element counts (including empty), waits for
    # the snapshot done marker, loads it and expects all elements in any order.
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(num_elements)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[None, "AUTO", "GZIP"])))
  def testCompression(self, compression):
    # [FUNCTION DOC]
    # Saves ten elements with the compression setting (none, automatic or gzip), waits for
    # completion, loads without specifying compression and expects 0-9 in any order.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(distributed_save_op.distributed_save(
        dataset,
        snapshot_dir.full_path,
        cluster.dispatcher_address(),
        compression=compression))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset, list(range(10)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3], num_repetitions=[1, 5])))
  def testRepeatedDataset(self, num_workers, num_repetitions):
    # [FUNCTION DOC]
    # Saves a 1000-element range repeated by the given count, waits for completion, loads and
    # expects the range repeated that many times, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.repeat(num_repetitions)
    self.evaluate(distributed_save_op.distributed_save(
        dataset,
        snapshot_dir.full_path,
        cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset, list(range(1000)) * num_repetitions, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromDatasets(self):
    # [FUNCTION DOC]
    # Saves a dataset that picks, in round-robin order from an endless choice stream, among three
    # five-element string datasets (a, b, c). Loads it without waiting for the done marker and
    # expects the three strings five times each, any order.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(["a", "a", "a", "a", "a"]),
        dataset_ops.Dataset.from_tensor_slices(["b", "b", "b", "b", "b"]),
        dataset_ops.Dataset.from_tensor_slices(["c", "c", "c", "c", "c"])]
    choice_dataset = dataset_ops.Dataset.range(3).repeat()
    dataset = dataset_ops.Dataset.choose_from_datasets(datasets, choice_dataset)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset, [b"a", b"b", b"c"] * 5, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromRepeatedDatasets(self):
    # [FUNCTION DOC]
    # Saves a choose-from dataset over repeated single-string datasets a (5), b (5), c (10), not
    # stopping on empty datasets. Waits for completion, loads and expects a, b, c five times plus
    # five more c, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    datasets = [
        dataset_ops.Dataset.from_tensors("a").repeat(5),
        dataset_ops.Dataset.from_tensors("b").repeat(5),
        dataset_ops.Dataset.from_tensors("c").repeat(10)]
    choice_dataset = dataset_ops.Dataset.range(3).repeat()
    dataset = dataset_ops.Dataset.choose_from_datasets(
        datasets, choice_dataset, stop_on_empty_dataset=False)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    self.assertDatasetProduces(
        dataset, [b"a", b"b", b"c"] * 5 + [b"c"] * 5, assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3])))
  def testWriteMultipleDatasets(self, num_workers):
    # [FUNCTION DOC]
    # Saves two different datasets (a 100-element range and a choose-from string dataset) to two
    # snapshot paths through the same cluster and waits for both. Loads the first and expects the
    # range; for the second, asserts directly on the original in-memory dataset.
    # KNOWN ISSUE: the second snapshot is never loaded; the final assertion checks the original
    # unsaved dataset, so the second snapshot's content is not actually verified.
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset1 = dataset_ops.Dataset.range(100)
    datasets = [
        dataset_ops.Dataset.from_tensors("a").repeat(5),
        dataset_ops.Dataset.from_tensors("b").repeat(5),
        dataset_ops.Dataset.from_tensors("c").repeat(5)]
    choice_dataset = dataset_ops.Dataset.range(3).repeat()
    dataset2 = dataset_ops.Dataset.choose_from_datasets(
        datasets, choice_dataset)

    snapshot_path1 = os.path.join(snapshot_dir.full_path, "snapshot1")
    snapshot_path2 = os.path.join(snapshot_dir.full_path, "snapshot2")
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset1, snapshot_path1, cluster.dispatcher_address()))
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset2, snapshot_path2, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_path1)
    _wait_for_snapshot(snapshot_path2)

    dataset1 = dataset_ops.Dataset.load(snapshot_path1)
    self.assertDatasetProduces(
        dataset1, list(range(100)), assert_items_equal=True)
    self.assertDatasetProduces(
        dataset2, [b"a", b"b", b"c"] * 5, assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_workers=[1, 3], snapshot_max_chunk_size_bytes=[1, 100])))
  def testLoadWithCustomReaderFunc(
      self, num_workers, snapshot_max_chunk_size_bytes):
    # [FUNCTION DOC]
    # Saves a ten-element range on a cluster with the given maximum chunk size, waits, then loads
    # with a custom reader function that shuffles the chunk datasets and interleaves them in
    # parallel. Expects 0-9 in any order.
    cluster = data_service_test_base.TestCluster(
        num_workers=num_workers,
        snapshot_max_chunk_size_bytes=snapshot_max_chunk_size_bytes)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    def custom_reader_func(datasets):
      # [FUNCTION DOC]
      # HELPER: reader function for the load call in the custom reader test.
      # Takes a dataset of chunk datasets, shuffles it with a buffer of three, interleaves the
      # chunks with automatic parallelism and returns the resulting dataset.
      datasets = datasets.shuffle(3)
      return datasets.interleave(
          lambda x: x, num_parallel_calls=dataset_ops.AUTOTUNE)

    dataset = dataset_ops.Dataset.load(
        snapshot_dir.full_path, reader_func=custom_reader_func)
    self.assertDatasetProduces(
        dataset, list(range(10)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_workers=[1, 3],
              repeated_load=[1, 5],
              sharding_policy=[
                  data_service_ops.ShardingPolicy.OFF,
                  data_service_ops.ShardingPolicy.DYNAMIC])))
  def testDistributedLoad(self, num_workers, repeated_load, sharding_policy):
    # [FUNCTION DOC]
    # Saves a ten-element range, waits, loads it (repeated when requested) and distributes it via
    # the service with the sharding policy (off or dynamic). Expects the range repeated by the
    # load count, multiplied by the worker count when sharding is off, in any order.
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    if repeated_load > 1:
      dataset = dataset.repeat(repeated_load)
    dataset = dataset.apply(
        data_service_ops.distribute(
            processing_mode=sharding_policy,
            service=cluster.dispatcher_address()))

    expected = list(range(10)) * repeated_load
    if sharding_policy == data_service_ops.ShardingPolicy.OFF:
      expected *= num_workers
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testImbalancedZipAndRepeat(self):
    # [FUNCTION DOC]
    # Zips a 200-element range with a 1000-element range, repeats it three times and saves it.
    # After completion, loads it and expects three repetitions of 200 identical pairs (zip stops
    # at the shorter input), in any order.
    smaller_num_elements = 200
    larger_num_elements = 1000
    repetitions = 3

    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset1 = dataset_ops.Dataset.range(smaller_num_elements)
    dataset2 = dataset_ops.Dataset.range(larger_num_elements)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    dataset = dataset.repeat(repetitions)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
    expected = repetitions * (
        list(zip(range(smaller_num_elements), range(smaller_num_elements))))
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSnapshotDoesNotExist(self):
    # [FUNCTION DOC]
    # Loads from an empty directory, distributes with sharding off and reads; expects a not-found
    # error.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    with self.assertRaises(errors.NotFoundError):
      dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
      dataset = dataset.apply(
          data_service_ops.distribute(
              data_service_ops.ShardingPolicy.OFF,
              cluster.dispatcher_address()))
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testDuplicateSnapshot(self):
    # [FUNCTION DOC]
    # Saves a dataset to the same directory twice; expects the second save to raise an
    # already-exists error saying the save was already started or completed.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        errors.AlreadyExistsError, "already started or completed"):
      self.evaluate(
          distributed_save_op.distributed_save(
              dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
      self.evaluate(
          distributed_save_op.distributed_save(
              dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

  @combinations.generate(test_base.default_test_combinations())
  def testWorkerFailure(self):
    # [FUNCTION DOC]
    # Saves a float dataset containing a not-a-number value mapped through a numeric check that
    # fails. Waits for the error marker file, then loading and reading must raise a value error
    # saying the save job failed to write it.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    components = np.array([1.0, 2.0, 3.0, np.nan, 5.0]).astype(np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.map(lambda x: array_ops.check_numerics(x, "message"))
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_error(snapshot_dir.full_path)

    with self.assertRaisesRegex(
        ValueError, "The save job failed to write it."):
      dataset = dataset_ops.Dataset.load(snapshot_dir.full_path)
      self.getDatasetOutput(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testBadDispatcherAddress(self):
    # [FUNCTION DOC]
    # Calls save with a non-string dispatcher address and expects a value error saying it must be
    # a string; then with an empty string and expects a value error saying it must not be empty.
    # No cluster is started.
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(ValueError, "must be a string"):
      self.evaluate(distributed_save_op.distributed_save(dataset, "", 1))
    with self.assertRaisesRegex(ValueError, "must not be empty"):
      self.evaluate(distributed_save_op.distributed_save(dataset, "", ""))

  @combinations.generate(test_base.default_test_combinations())
  def testBadCardinality(self):
    # [FUNCTION DOC]
    # Saves an infinitely repeated dataset; expects an invalid-argument error saying saving an
    # infinite dataset is not allowed.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10).repeat()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Saving an infinite dataset is not allowed"):
      self.evaluate(distributed_save_op.distributed_save(
          dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

  @combinations.generate(test_base.default_test_combinations())
  def testBadElementSpec(self):
    # [FUNCTION DOC]
    # Saves a ten-element range with automatic compression, waits, then loads passing a bogus
    # string as the element spec; expects a value error saying the specified spec differs from the
    # actual tensor spec.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path,
        cluster.dispatcher_address(),
        compression="AUTO"))
    _wait_for_snapshot(snapshot_dir.full_path)

    with self.assertRaisesRegex(
        ValueError,
        "User specified element_spec bad_element_spec, but the actual "
        "element_spec is TensorSpec"):
      _ = dataset_ops.Dataset.load(snapshot_dir.full_path,
                                   element_spec="bad_element_spec")

  @combinations.generate(test_base.default_test_combinations())
  def testBadCompression(self):
    # [FUNCTION DOC]
    # Saves with automatic compression (which resolves to snappy), waits, then loads asking for
    # zlib; expects a value error naming the mismatch between specified and actual compression.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(distributed_save_op.distributed_save(
        dataset, snapshot_dir.full_path,
        cluster.dispatcher_address(),
        compression="AUTO"))
    _wait_for_snapshot(snapshot_dir.full_path)

    with self.assertRaisesRegex(
        ValueError,
        "User specified compression ZLIB, but the actual compression is "
        "SNAPPY."):
      _ = dataset_ops.Dataset.load(snapshot_dir.full_path, compression="ZLIB")

  @combinations.generate(test_base.default_test_combinations())
  def testRequiresFaultTolerantMode(self):
    # [FUNCTION DOC]
    # Starts a cluster with fault tolerant mode disabled and tries to save; expects an
    # invalid-argument error saying distributed snapshot requires fault tolerant mode.
    cluster = data_service_test_base.TestCluster(
        num_workers=1, fault_tolerant_mode=False)
    snapshot_dir = data_service_test_base.TempDir()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "tf.data distributed snapshot requires running tf.data service in the "
        "fault tolerant mode."):
      self.evaluate(distributed_save_op.distributed_save(
          dataset_ops.Dataset.range(10), snapshot_dir.full_path,
          cluster.dispatcher_address(),
          compression="AUTO"))


class LoadCheckpointTest(
    data_service_test_base.TestBase,
    checkpoint_test_base.CheckpointTestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint test for datasets loaded from a distributed snapshot; combines the data service
  # test base with the generic checkpoint test base that supplies the verification function.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def testLoadCheckpoint(self, verify_fn):
    # [FUNCTION DOC]
    # Saves a ten-element range, waits for completion, then runs the checkpoint verification
    # function with a builder that loads the snapshot; expects ten outputs, compared in any order.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    _wait_for_snapshot(snapshot_dir.full_path)

    def _build_ds():
      # [FUNCTION DOC]
      # HELPER: dataset builder passed to the checkpoint verification function. Returns the
      # dataset loaded from the snapshot directory.
      return dataset_ops.Dataset.load(snapshot_dir.full_path)

    verify_fn(self, _build_ds, num_outputs=10, assert_items_equal=True)


def _wait_for_snapshot(snapshot_path):
  # [FUNCTION DOC]
  # HELPER: lets tests wait for a distributed save to finish.
  # Blocks, polling every 0.1 seconds, until the snapshot done marker file exists under the given
  # snapshot directory path (string). Returns nothing. Has no timeout, so it can hang forever if
  # the save never completes.
  while not os.path.exists(
      _pywrap_snapshot_utils.TF_DATA_SnapshotDoneFilePath(snapshot_path)):
    time.sleep(0.1)


def _wait_for_error(snapshot_path):
  # [FUNCTION DOC]
  # HELPER: lets tests wait for a distributed save to fail.
  # Blocks, polling every 0.1 seconds, until the snapshot error marker file exists under the given
  # snapshot directory path (string). Returns nothing. Has no timeout.
  while not os.path.exists(
      _pywrap_snapshot_utils.TF_DATA_SnapshotErrorFilePath(snapshot_path)):
    time.sleep(0.1)


if __name__ == "__main__":
  test.main()
