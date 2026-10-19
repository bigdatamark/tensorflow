# [MODULE DOC]
# Kernel tests for dataset save and load: round trips with and without compression, cardinality,
# custom shard and reader functions, saving inside a compiled function, optional element spec,
# waiting for a saved dataset to appear, and checkpointing during save and load.
# Test module only; no import-time side effects.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the `tf.data.experimental.{save,load}` operations."""

import os
import shutil
import threading
import time

from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class IOTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for saving datasets to disk and loading them back. Each test gets fresh temporary
  # directories for data, checkpoints and a separate save location. Also serves as a base class
  # for the checkpoint test classes below.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the parent setup, then creates three directories under the test temp
    # directory (data directory, checkpoint directory, save directory) and records their paths
    # on the instance. Takes no arguments, returns nothing.
    super(IOTest, self).setUp()
    tmpdir = self.get_temp_dir()
    tmpdir = os.path.join(tmpdir, "io_test")
    os.mkdir(tmpdir)
    self._test_dir = tmpdir
    self._checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    os.mkdir(self._checkpoint_prefix)
    self._save_dir = os.path.join(self.get_temp_dir(), "save")
    os.mkdir(self._save_dir)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture: calls the parent teardown, then recursively deletes the data, checkpoint and
    # save directories created by setup. Takes no arguments, returns nothing.
    super(IOTest, self).tearDown()
    shutil.rmtree(self._test_dir)
    shutil.rmtree(self._checkpoint_prefix)
    shutil.rmtree(self._save_dir)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(compression=[None, "GZIP"]),
      )
  )
  def testBasic(self, compression):
    # [FUNCTION DOC]
    # Scenario: save a 42-element range with no or GZIP compression, then load it with the
    # original element spec and the same compression. Expects the same 42 elements back.
    dataset = dataset_ops.Dataset.range(42)
    self.evaluate(dataset.save(self._test_dir, compression=compression))
    dataset2 = dataset_ops.Dataset.load(
        self._test_dir, dataset.element_spec, compression=compression
    )
    self.assertDatasetProduces(dataset2, range(42))

  @combinations.generate(test_base.default_test_combinations())
  def testCardinality(self):
    # [FUNCTION DOC]
    # Scenario: save then load a 42-element range. Expects the loaded dataset to report
    # cardinality 42.
    dataset = dataset_ops.Dataset.range(42)
    self.evaluate(dataset.save(self._test_dir))
    dataset2 = dataset_ops.Dataset.load(self._test_dir, dataset.element_spec)
    self.assertEqual(self.evaluate(dataset2.cardinality()), 42)

  @combinations.generate(test_base.default_test_combinations())
  def testCustomShardFunction(self):
    # [FUNCTION DOC]
    # Scenario: save a 42-element range with a shard function splitting it into two halves.
    # Loading with the default reader is expected to interleave the two shards, producing each
    # element of the first half followed by its counterpart from the second half.
    dataset = dataset_ops.Dataset.range(42)
    self.evaluate(dataset.save(self._test_dir, shard_func=lambda x: x // 21))
    dataset2 = dataset_ops.Dataset.load(self._test_dir, dataset.element_spec)
    expected = []
    for i in range(21):
      expected.extend([i, i + 21])
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testCustomReaderFunction(self):
    # [FUNCTION DOC]
    # Scenario: save a 42-element range sharded by value modulo 7, then load with a reader
    # function that simply concatenates shards in order. Expects the elements grouped by shard.
    dataset = dataset_ops.Dataset.range(42)
    self.evaluate(dataset.save(self._test_dir, shard_func=lambda x: x % 7))
    dataset2 = dataset_ops.Dataset.load(
        self._test_dir,
        dataset.element_spec,
        reader_func=lambda x: x.flat_map(lambda y: y),
    )
    expected = []
    for i in range(7):
      expected.extend(range(i, 42, 7))
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(compression=[None, "GZIP"]),
      )
  )
  def testSaveInsideFunction(self, compression):
    # [FUNCTION DOC]
    # Scenario (eager only): the save operation is invoked from inside a compiled function, with
    # no or GZIP compression. Loading afterwards must produce the 42 original elements.
    dataset = dataset_ops.Dataset.range(42)

    @def_function.function
    def save_fn():
      # [FUNCTION DOC]
      # HELPER: compiled function that saves the enclosing dataset to the test directory with
      # the enclosing compression setting. Takes no arguments, returns nothing.
      dataset.save(self._test_dir, compression=compression)

    save_fn()
    dataset = dataset_ops.Dataset.load(
        self._test_dir, dataset.element_spec, compression=compression
    )
    self.assertDatasetProduces(dataset, range(42))

  @combinations.generate(test_base.default_test_combinations())
  def testElementSpecOptional(self):
    # [FUNCTION DOC]
    # Scenario: save a zip of a range, a mapping-valued dataset and a tuple-valued dataset, then
    # load without giving an element spec. Expects the loaded dataset to equal the original
    # (structure inferred from saved metadata).
    range_dataset = dataset_ops.Dataset.range(42)
    dict_dataset = dataset_ops.Dataset.from_tensor_slices(
        {"a": [1, 2], "b": [3, 4]}
    )
    tuple_dataset = dataset_ops.Dataset.from_tensor_slices(([1, 2], [3, 4]))
    dataset = dataset_ops.Dataset.zip(
        (range_dataset, dict_dataset, tuple_dataset)
    )
    self.evaluate(dataset.save(self._test_dir))
    dataset_loaded = dataset_ops.Dataset.load(self._test_dir)
    self.assertDatasetsEqual(dataset, dataset_loaded)

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatAndPrefetch(self):
    # [FUNCTION DOC]
    # Regression scenario for a reported repeat-plus-prefetch problem: save random data, load it,
    # then shuffle, batch, repeat and prefetch it. Reads 30 batches and expects no errors.
    """This test reproduces github.com/tensorflow/tensorflow/issues/49165."""
    dataset1 = dataset_ops.Dataset.from_tensor_slices(np.random.rand(16, 32))
    self.evaluate(dataset1.save(self._test_dir))
    dataset = dataset_ops.Dataset.load(self._test_dir)
    dataset = dataset.shuffle(buffer_size=16)
    dataset = dataset.batch(16)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    next_element = self.getNext(dataset)
    for _ in range(30):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testWait(self):
    # [FUNCTION DOC]
    # Scenario: a loader thread starts loading before anything is saved, while a second thread
    # sleeps five seconds and then saves a 50-element range. Expects the loader to wait for the
    # saved data and produce all 50 elements (in any order). Both threads are joined.
    dataset_range = 50

    def load_thread_fn():
      # [FUNCTION DOC]
      # HELPER: thread body that loads from the test directory and asserts the dataset yields
      # the numbers 0 to 49 (order-insensitive). Takes no arguments.
      dataset = dataset_ops.Dataset.load(self._test_dir)
      self.assertDatasetProduces(
          dataset, list(range(dataset_range)), assert_items_equal=True)
    load_thread = threading.Thread(target=load_thread_fn, name="load_thread")
    load_thread.start()

    def save_thread_fn():
      # [FUNCTION DOC]
      # HELPER: thread body that sleeps five seconds, then saves a 50-element range to the test
      # directory. Takes no arguments.
      time.sleep(5)
      dataset = dataset_ops.Dataset.range(dataset_range)
      self.evaluate(dataset.save(self._test_dir))
    save_thread = threading.Thread(target=save_thread_fn, name="save_thread")
    save_thread.start()
    save_thread.join()
    load_thread.join()


class LoadCheckpointTest(IOTest, checkpoint_test_base.CheckpointTestBase):
  # [CLASS DOC]
  # Checkpoint tests for the load operation. Inherits the save/load tests from IOTest (so they
  # run again here) and the checkpoint verification helpers from the checkpoint test base.

  def _build_ds(self):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test by loading from the save directory; returns it.
    return dataset_ops.Dataset.load(self._save_dir)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
      )
  )
  def test(self, verify_fn):
    # [FUNCTION DOC]
    # Scenario: save a 42-element range to the save directory, then run the supplied checkpoint
    # verification function on the loading dataset builder expecting 42 outputs.
    dataset = dataset_ops.Dataset.range(42)
    self.evaluate(dataset.save(self._save_dir))
    verify_fn(self, self._build_ds, num_outputs=42)


class SaveCheckpointTest(IOTest, checkpoint_test_base.CheckpointTestBase):
  # [CLASS DOC]
  # Tests for the checkpointing options of the save operation (eager only), counting checkpoint
  # files written. Inherits all IOTest tests, which run again under this class.

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveCheckpointingAPI(self):
    # [FUNCTION DOC]
    # Scenario: save a 40-element range with checkpoint arguments (directory, keep up to 50).
    # With the default interval, expects 81 files: a data and an index file per element plus the
    # overall checkpoint file.
    dataset = dataset_ops.Dataset.range(40)
    checkpoint_args = {"directory": self._checkpoint_prefix, "max_to_keep": 50}
    dataset.save(self._save_dir, checkpoint_args=checkpoint_args)
    num_checkpoint_files = len(list(os.listdir(self._checkpoint_prefix)))
    # By default, we checkpoint every increment. Each checkpoint writes a
    # file containing the data and a file containing the index. There is
    # also an overall checkpoint file. Thus, we expect (2 * 40) + 1 files.
    self.assertEqual(81, num_checkpoint_files)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveCheckpointingAPICustomCheckpointInterval(self):
    # [FUNCTION DOC]
    # Scenario: save a 40-element range with a checkpoint interval of 5, a step counter variable
    # and keep up to 10 checkpoints. Expects 17 files (two per retained checkpoint of 8, plus the
    # overall checkpoint file).
    dataset = dataset_ops.Dataset.range(40)
    step_counter = variables.Variable(0, trainable=False)
    checkpoint_args = {
        "checkpoint_interval": 5,
        "step_counter": step_counter,
        "directory": self._checkpoint_prefix,
        "max_to_keep": 10,
    }
    dataset.save(self._save_dir, checkpoint_args=checkpoint_args)
    num_checkpoint_files = len(list(os.listdir(self._checkpoint_prefix)))
    # We expect (2 * 8) + 1 files.
    self.assertEqual(17, num_checkpoint_files)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveCheckpointingAPIIncorrectArgs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the save call also passes the dataset itself as the path argument, so a type
    # error may come from that misuse rather than from the unknown checkpoint argument.
    # Scenario: checkpoint arguments contain an unsupported key. Expects a type error.
    dataset = dataset_ops.Dataset.range(42)
    checkpoint_args = {
        "directory": self._checkpoint_prefix,
        "incorrect_arg": "incorrect_arg"
    }
    with self.assertRaises(TypeError):
      dataset.save(
          dataset, self._save_dir, checkpoint_args=checkpoint_args)


if __name__ == "__main__":
  test.main()
