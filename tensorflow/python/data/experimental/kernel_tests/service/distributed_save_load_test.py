# [MODULE DOC]
# Tests for distributed save followed by load in the tf.data service, using the newer load
# algorithm that reads snapshot chunks directly. Covers round trips with different worker counts,
# repetitions, compression and chunk sizes; saving and loading concurrently; deterministic load
# order after a save finishes; shuffling chunks; distributed loading with sharding policies;
# sampling before and after saving; enumerate; worker failure; the optional dataset spec file;
# missing snapshots; the chunk listing dataset (cardinality and ordering); and checkpointing of
# loaded datasets. Uses the shared data service test base. Runs under the standard test main.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for distributed save/load with the new load algorithm."""

import multiprocessing
import os
import threading
import time
from typing import Callable, Optional

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distributed_save_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import load_op
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class DistributedSaveLoadTest(
    data_service_test_base.TestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for distributed save and load. Each test creates its own in-process test cluster
  # and a temporary snapshot directory, saves a dataset with the distributed save op, and loads
  # it back to compare output.
  """Tests for distributed save/load with the new load algorithm."""

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_workers=[1, 3],
              num_elements=[0, 10],
              num_repetitions=[1, 3],
              compression=[None, "AUTO", "GZIP"],
              max_chunk_size_bytes=[1, 16 << 10])))
  def test_save_load(
      self,
      num_workers: int,
      num_elements: int,
      num_repetitions: int,
      compression: Optional[str],
      max_chunk_size_bytes: int):
    # [FUNCTION DOC]
    # Parameterized round-trip test over worker count, element count (including zero),
    # repetitions, compression (none, automatic, gzip) and maximum chunk size. Note the
    # compression parameter is accepted but never used in the body. Saves a repeated range
    # dataset through a cluster configured with the chunk size, loads it (waiting for the save to
    # finish) and expects all elements and repetitions, ignoring order.
    # KNOWN ISSUE: compression is not passed to the save or load, so compression is not tested.
    cluster = data_service_test_base.TestCluster(
        num_workers=num_workers,
        snapshot_max_chunk_size_bytes=max_chunk_size_bytes)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.repeat(num_repetitions)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
    self.assertDatasetProduces(
        dataset,
        list(range(num_elements)) * num_repetitions,
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3])))
  def test_concurrent_save_load(self, num_workers: int):
    # [FUNCTION DOC]
    # Parameterized by worker count. Starts a loader thread that loads a snapshot directory
    # (waiting for it to appear) and expects the ten numbers 0-9 in any order, then a saver thread
    # that sleeps five seconds and saves a ten-element range dataset to the same directory. Joins
    # both threads. Verifies load can begin before the save has started.
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()

    def load_thread_fn():
      # [FUNCTION DOC]
      # Thread body: loads the snapshot (waiting for it) and asserts it produces 0-9 in any
      # order. Takes no arguments, returns nothing.
      dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
      self.assertDatasetProduces(
          dataset, list(range(10)), assert_items_equal=True)
    load_thread = threading.Thread(target=load_thread_fn, name="load_thread")
    load_thread.start()

    def save_thread_fn():
      # [FUNCTION DOC]
      # Thread body: waits five seconds, then distributed-saves a ten-element range dataset to
      # the snapshot directory using the cluster dispatcher. Takes no arguments, returns nothing.
      time.sleep(5)
      dataset = dataset_ops.Dataset.range(10)
      self.evaluate(
          distributed_save_op.distributed_save(
              dataset, snapshot_dir.full_path, cluster.dispatcher_address()))
    save_thread = threading.Thread(target=save_thread_fn, name="save_thread")
    save_thread.start()
    save_thread.join()
    load_thread.join()

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_workers=[1, 5],
              num_elements=[10],
              num_repetitions=[10],
              max_chunk_size_bytes=[1, 16 << 10])))
  def test_deterministic_load_order(
      self,
      num_workers: int,
      num_elements: int,
      num_repetitions: int,
      max_chunk_size_bytes: int):
    # [FUNCTION DOC]
    # Parameterized by worker count, element count, repetition count and chunk size. Saves a
    # shuffled range dataset, loads it repeated, splits the output into per-repetition lists
    # and asserts there are exactly num_repetitions of them and that repetitions from the third
    # onward equal their predecessor (the first repetition may be non-deterministic while the
    # snapshot is still being written).
    """Verifies `load` produces data deterministically after `save` finishes."""
    cluster = data_service_test_base.TestCluster(
        num_workers=num_workers,
        snapshot_max_chunk_size_bytes=max_chunk_size_bytes)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(num_elements).shuffle(
        buffer_size=num_elements)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
    dataset = dataset.repeat(num_repetitions)
    output = self.getDatasetOutput(dataset)
    output_per_repetition = [
        output[i : i + num_elements]
        for i in range(0, len(output), num_elements)]
    self.assertLen(output_per_repetition, num_repetitions)
    for i in range(2, num_repetitions):  # Starts from the second repetition.
      self.assertEqual(output_per_repetition[i], output_per_repetition[i - 1])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_elements=[20],
              num_repetitions=[10])))
  def test_shuffle_chunks(
      self, num_elements: int, num_repetitions: int):
    # [FUNCTION DOC]
    # Saves a range dataset on a five-worker cluster with tiny (8 byte) chunks, then loads it once
    # per repetition through a flat map using a custom reader function that shuffles the chunk
    # list (only when its cardinality is positive) and interleaves the chunks in parallel. Expects
    # num_repetitions groups of output, and each group to differ from the previous one.
    cluster = data_service_test_base.TestCluster(
        num_workers=5, snapshot_max_chunk_size_bytes=8)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(num_elements)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    def _interleave_shuffled_chunks(
        datasets: dataset_ops.Dataset) -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: reader function for the shuffle-chunks test's load call.
      # Takes a dataset of chunk datasets, shuffles it with a buffer of its cardinality when the
      # cardinality is positive (otherwise leaves it), then interleaves the chunks with cycle length
      # equal to the CPU count and automatic parallelism. Returns the resulting dataset.
      # Does not shuffle when the snapshot is being written (first repetition).
      # Otherwise, loading will be blocked to fill the shuffle buffer, not to
      # read the partial chunks.
      datasets = cond.cond(
          math_ops.greater(datasets.cardinality(), 0),
          lambda: datasets.shuffle(buffer_size=datasets.cardinality()),
          lambda: datasets)
      return datasets.interleave(
          lambda x: x,
          cycle_length=multiprocessing.cpu_count(),
          num_parallel_calls=dataset_ops.AUTOTUNE)

    dataset = dataset_ops.Dataset.range(num_repetitions).flat_map(
        lambda _: dataset_ops.Dataset.load(
            snapshot_dir.full_path,
            reader_func=_interleave_shuffled_chunks,
            wait=True))

    output = self.getDatasetOutput(dataset)
    output_per_repetition = [
        output[i : i + num_elements]
        for i in range(0, len(output), num_elements)]
    self.assertLen(output_per_repetition, num_repetitions)
    for i in range(1, num_repetitions):
      self.assertNotEqual(output_per_repetition[i],
                          output_per_repetition[i - 1])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_workers=[1, 3],
              num_elements=[0, 10],
              repeated_load=[1, 5],
              sharding_policy=[
                  data_service_ops.ShardingPolicy.OFF,
                  data_service_ops.ShardingPolicy.DYNAMIC])))
  def test_distributed_load(
      self,
      num_workers: int,
      num_elements: int,
      repeated_load: int,
      sharding_policy: data_service_ops.ShardingPolicy):
    # [FUNCTION DOC]
    # Parameterized by workers, element count, load repetitions and sharding policy (off or
    # dynamic). Saves a range dataset, loads it (repeated when requested) and distributes it
    # through the service. Expects the range repeated by the load count, multiplied by the number
    # of workers when sharding is off, ignoring order.
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(num_elements)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
    if repeated_load > 1:
      dataset = dataset.repeat(repeated_load)
    dataset = dataset.apply(
        data_service_ops.distribute(
            sharding_policy, cluster.dispatcher_address()))
    expected = list(range(num_elements)) * repeated_load
    if sharding_policy == data_service_ops.ShardingPolicy.OFF:
      expected *= num_workers
    self.assertDatasetProduces(dataset, expected, assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3])))
  def test_save_before_sample(self, num_workers: int):
    # [FUNCTION DOC]
    # Saves three separate ten-element range datasets into sub-directories of one snapshot
    # directory, loads each, mixes them with equal-weight sampling (not stopping on an empty
    # dataset) and expects thirty elements: the range repeated three times, in any order.
    num_elements = 10
    num_datasets = 3
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    datasets = [
        dataset_ops.Dataset.range(num_elements) for i in range(num_datasets)]
    for i, dataset in enumerate(datasets):
      self.evaluate(
          distributed_save_op.distributed_save(
              dataset,
              os.path.join(snapshot_dir.full_path, f"dataset_{i}"),
              cluster.dispatcher_address()))

    loaded_datasets = []
    for i in range(len(datasets)):
      snapshot_path = os.path.join(snapshot_dir.full_path, f"dataset_{i}")
      loaded_datasets.append(dataset_ops.Dataset.load(snapshot_path, wait=True))
    dataset = dataset_ops.Dataset.sample_from_datasets(
        loaded_datasets,
        weights=[1.0] * num_datasets,
        stop_on_empty_dataset=False)
    self.assertDatasetProduces(
        dataset,
        list(range(num_elements)) * num_datasets,
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3], num_repetitions=[1, 3])))
  def test_save_after_sample(self, num_workers: int, num_repetitions: int):
    # [FUNCTION DOC]
    # Builds three ten-element range datasets (each repeated when requested), mixes them by
    # equal-weight sampling, saves the mixture, loads it back and expects the range repeated three
    # times the repetition count, in any order.
    num_elements = 10
    num_datasets = 3
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    datasets = [
        dataset_ops.Dataset.range(num_elements) for i in range(num_datasets)]
    if num_repetitions > 1:
      datasets = [dataset.repeat(num_repetitions) for dataset in datasets]
    dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=[1.0] * num_datasets, stop_on_empty_dataset=False)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
    self.assertDatasetProduces(
        dataset,
        list(range(num_elements)) * num_datasets * num_repetitions,
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_workers=[1, 3])))
  def test_enumerate(self, num_workers: int):
    # [FUNCTION DOC]
    # Saves a dataset of the strings a, b, c repeated three times and enumerated, loads it and
    # splits output into indexes and elements. Expects elements to be a, b, c three times (as
    # byte strings) in any order; with a single worker also expects indexes 0 to 8.
    cluster = data_service_test_base.TestCluster(num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.from_tensor_slices(["a", "b", "c"])
    dataset = dataset.repeat(3)
    dataset = dataset.enumerate()
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
    indexes, elements = map(list, zip(*self.getDatasetOutput(dataset)))
    if num_workers == 1:
      self.assertCountEqual(indexes, list(range(9)))
    self.assertCountEqual(elements, [b"a", b"b", b"c"] * 3)

  @combinations.generate(test_base.default_test_combinations())
  def test_worker_failure(self):
    # [FUNCTION DOC]
    # Saves a dataset of floats containing a not-a-number value, mapped through a numeric check
    # that fails on it. Loading and reading the snapshot must raise an invalid-argument error,
    # propagated from the failed save.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    components = np.array([1.0, 2.0, 3.0, np.nan, 5.0]).astype(np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.map(lambda x: array_ops.check_numerics(x, "message"))
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
      self.getDatasetOutput(dataset)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_elements=[10])))
  def test_dataset_spec_file_is_optional(self, num_elements: int):
    # [FUNCTION DOC]
    # Saves a range dataset and loads it, checking the output. Then deletes the dataset spec file
    # from the snapshot directory and loads again, expecting the same output (the element spec is
    # recovered from other metadata).
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(num_elements)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
    self.assertDatasetProduces(
        dataset, list(range(num_elements)), assert_items_equal=True)

    # After removing the dataset_spec file, the loaded dataset should produce
    # the same output.
    os.remove(os.path.join(
        snapshot_dir.full_path, dataset_ops.DATASET_SPEC_FILENAME))
    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
    self.assertDatasetProduces(
        dataset, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_elements=[10])))
  def test_empty_dataset_spec_file(self, num_elements: int):
    # [FUNCTION DOC]
    # Saves a range dataset and loads it, checking the output. Then overwrites the dataset spec
    # file with empty content and loads again, expecting the same output (element spec is read
    # from the metadata file instead).
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(num_elements)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
    self.assertDatasetProduces(
        dataset, list(range(num_elements)), assert_items_equal=True)

    dataset_spec_file = os.path.join(
        snapshot_dir.full_path, dataset_ops.DATASET_SPEC_FILENAME)
    with open(dataset_spec_file, "w") as f:
      f.write("")

    # Reads element_spec from the metadata file.
    dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
    self.assertDatasetProduces(
        dataset, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def test_snapshot_does_not_exist(self):
    # [FUNCTION DOC]
    # Loads from an empty temporary directory without waiting; reading the dataset must raise a
    # not-found error.
    snapshot_dir = data_service_test_base.TempDir()
    with self.assertRaises(errors.NotFoundError):
      dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=False)
      self.getDatasetOutput(dataset)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_repetitions=[None, 0, 1, 3])))
  def test_snapshot_chunks_cardinality(self, num_repetitions: int):
    # [FUNCTION DOC]
    # Parameterized by repetitions (nothing meaning infinite, 0, 1, 3). Saves a ten-element
    # range, builds the internal snapshot chunk listing dataset, repeats it unless the count is
    # one, and polls every 0.1 seconds until the cardinality is known. Expects cardinality to be
    # infinite for the nothing case, otherwise the number of files in the chunks directory times
    # the repetition count.
    cluster = data_service_test_base.TestCluster(num_workers=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(10)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = load_op._ListSnapshotChunksDataset(snapshot_dir.full_path)
    if num_repetitions != 1:
      dataset = dataset.repeat(num_repetitions)
    while self.evaluate(dataset.cardinality()) == dataset_ops.UNKNOWN:
      time.sleep(.1)

    num_chunks = len(os.listdir(os.path.join(snapshot_dir.full_path, "chunks")))
    expected_cardinality = (
        dataset_ops.INFINITE
        if num_repetitions is None
        else num_chunks * num_repetitions)
    self.assertEqual(self.evaluate(dataset.cardinality()), expected_cardinality)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_repetitions=[50])))
  def test_snapshot_chunks_order(self, num_repetitions: int):
    # [FUNCTION DOC]
    # Saves 50 single-element repetitions with one-byte chunks, so many chunk files exist. Lists
    # chunks via the internal chunk listing dataset once cardinality is known, extracts the chunk
    # index from each file name, and expects the indices to be in sorted order.
    cluster = data_service_test_base.TestCluster(
        num_workers=1, snapshot_max_chunk_size_bytes=1)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(1)
    dataset = dataset.repeat(num_repetitions)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    dataset = load_op._ListSnapshotChunksDataset(snapshot_dir.full_path)
    while self.evaluate(dataset.cardinality()) == dataset_ops.UNKNOWN:
      time.sleep(.1)

    chunk_indices = [
        int(os.path.basename(str(chunk)).split("_")[2])
        for chunk in self.getDatasetOutput(dataset)]
    self.assertEqual(chunk_indices, sorted(chunk_indices),
                     "Snapshot chunks should be sorted by chunk indices.")


class SaveLoadCheckpointTest(
    data_service_test_base.TestBase,
    checkpoint_test_base.CheckpointTestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint tests for datasets loaded from a distributed snapshot, combining the data service
  # test base with the generic dataset checkpoint test base (which supplies the verification
  # function).

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              num_workers=[1, 3],
              num_elements=[0, 10],
              num_repetitions=[1, 5])))
  def test_save_load_checkpoint(
      self,
      verify_fn: Callable[..., None],
      num_workers: int,
      num_elements: int,
      num_repetitions: int,):
    # [FUNCTION DOC]
    # Saves a range dataset, then uses the checkpoint verification function with a builder that
    # loads the snapshot (repeated when requested). Expects num_elements times num_repetitions
    # outputs, compared ignoring order since the first repetition may be non-deterministic.
    cluster = data_service_test_base.TestCluster(
        num_workers=num_workers)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(num_elements)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    def _build_ds() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: dataset builder passed to the checkpoint verification function.
      # Loads the snapshot (waiting for it) and repeats it when the repetition count exceeds one.
      dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
      if num_repetitions > 1:
        dataset = dataset.repeat(num_repetitions)
      return dataset

    # Compares output ignoring order since the first repetition may be
    # non-deterministic.
    verify_fn(
        self,
        _build_ds,
        num_outputs=num_elements * num_repetitions,
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              num_workers=[1, 3],
              num_elements=[0, 10],
              num_repetitions=[5],
              max_chunk_size_bytes=[1, 16 << 10])))
  def test_skip_first_repetition(
      self,
      verify_fn: Callable[..., None],
      num_workers: int,
      num_elements: int,
      num_repetitions: int,
      max_chunk_size_bytes: int):
    # [FUNCTION DOC]
    # Saves a range dataset with the given chunk size, then uses the checkpoint verification
    # function on a builder that loads, repeats and skips the first repetition. Expects
    # num_elements times (num_repetitions - 1) outputs in deterministic order (items are not
    # compared ignoring order).
    cluster = data_service_test_base.TestCluster(
        num_workers=num_workers,
        snapshot_max_chunk_size_bytes=max_chunk_size_bytes)
    snapshot_dir = data_service_test_base.TempDir()
    dataset = dataset_ops.Dataset.range(num_elements)
    self.evaluate(
        distributed_save_op.distributed_save(
            dataset, snapshot_dir.full_path, cluster.dispatcher_address()))

    def _build_ds() -> dataset_ops.Dataset:
      # [FUNCTION DOC]
      # HELPER: dataset builder passed to the checkpoint verification function.
      # Loads the snapshot (waiting), repeats it by the repetition count, and skips the first
      # repetition so the remaining ones are deterministic.
      dataset = dataset_ops.Dataset.load(snapshot_dir.full_path, wait=True)
      dataset = dataset.repeat(num_repetitions)
      # Skips the first repetition. The remaining repetitions should be
      # deterministic.
      dataset = dataset.skip(num_elements)
      return dataset

    verify_fn(
        self,
        _build_ds,
        num_outputs=num_elements * (num_repetitions - 1),
        assert_items_equal=False)


if __name__ == "__main__":
  test.main()
