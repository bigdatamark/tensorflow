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
# [MODULE DOC]
# Tests for saving datasets to disk and loading them back (the experimental io ops): compression,
# cardinality, custom shard and reader functions, saving inside a traced function, optional
# element spec, a prefetch/repeat regression, checkpoint-while-saving arguments, and
# checkpointing of loaded datasets. Each test uses temporary directories created in setup.
import os
import shutil

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import io
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class IOTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for save and load of datasets. Holds three temporary directories per test (data,
  # checkpoint prefix, save directory); also used as a base class by the checkpoint test classes.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates three fresh directories under the test temp directory (the data directory, a
    # checkpoint directory and a save directory) and stores their paths on the instance.
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
    # Removes the three temporary directories created by setup after calling the parent teardown.
    super(IOTest, self).tearDown()
    shutil.rmtree(self._test_dir)
    shutil.rmtree(self._checkpoint_prefix)
    shutil.rmtree(self._save_dir)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(compression=[None, "GZIP"])))
  def testBasic(self, compression):
    # [FUNCTION DOC]
    # Saves a range of 42 (no compression or GZIP), loads it with the element spec and asserts the
    # loaded dataset produces the same range.
    dataset = dataset_ops.Dataset.range(42)
    self.evaluate(io.save(dataset, self._test_dir, compression=compression))
    dataset2 = io.load(
        self._test_dir, dataset.element_spec, compression=compression)
    self.assertDatasetProduces(dataset2, range(42))

  @combinations.generate(test_base.default_test_combinations())
  def testCardinality(self):
    # [FUNCTION DOC]
    # Saves and loads a range of 42; asserts the loaded dataset reports cardinality 42.
    dataset = dataset_ops.Dataset.range(42)
    self.evaluate(io.save(dataset, self._test_dir))
    dataset2 = io.load(self._test_dir, dataset.element_spec)
    self.assertEqual(self.evaluate(dataset2.cardinality()), 42)

  @combinations.generate(test_base.default_test_combinations())
  def testCustomShardFunction(self):
    # [FUNCTION DOC]
    # Saves a range of 42 with a shard function splitting into two shards (first half and second
    # half); asserts loading yields elements interleaved across the shards (i, i + 21).
    dataset = dataset_ops.Dataset.range(42)
    self.evaluate(
        io.save(dataset, self._test_dir, shard_func=lambda x: x // 21)
    )
    dataset2 = io.load(self._test_dir, dataset.element_spec)
    expected = []
    for i in range(21):
      expected.extend([i, i + 21])
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testCustomReaderFunction(self):
    # [FUNCTION DOC]
    # Saves a range of 42 into seven shards by value modulo seven and loads with a reader function
    # that concatenates shards sequentially; asserts the output is grouped by shard.
    dataset = dataset_ops.Dataset.range(42)
    self.evaluate(io.save(dataset, self._test_dir, shard_func=lambda x: x % 7))
    dataset2 = io.load(
        self._test_dir,
        dataset.element_spec,
        reader_func=lambda x: x.flat_map(lambda y: y))
    expected = []
    for i in range(7):
      expected.extend(range(i, 42, 7))
    self.assertDatasetProduces(dataset2, expected)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(compression=[None, "GZIP"])))
  def testSaveInsideFunction(self, compression):
    # [FUNCTION DOC]
    # Eager only. Calls save from inside a traced function (with optional GZIP), then loads and
    # asserts the range of 42 is produced.

    dataset = dataset_ops.Dataset.range(42)

    @def_function.function
    def save_fn():
      # [FUNCTION DOC]
      # HELPER: traced function that saves the dataset to the test directory with the compression.
      io.save(dataset, self._test_dir, compression=compression)

    save_fn()
    dataset = io.load(
        self._test_dir, dataset.element_spec, compression=compression)
    self.assertDatasetProduces(dataset, range(42))

  @combinations.generate(test_base.default_test_combinations())
  def testElementSpecOptional(self):
    # [FUNCTION DOC]
    # Saves a zipped dataset of a range, a dict dataset and a tuple dataset, then loads without
    # giving an element spec; asserts the loaded dataset equals the original.
    range_dataset = dataset_ops.Dataset.range(42)
    dict_dataset = dataset_ops.Dataset.from_tensor_slices({"a": [1, 2],
                                                           "b": [3, 4]})
    tuple_dataset = dataset_ops.Dataset.from_tensor_slices(([1, 2], [3, 4]))
    dataset = dataset_ops.Dataset.zip((range_dataset, dict_dataset,
                                       tuple_dataset))
    self.evaluate(io.save(dataset, self._test_dir))
    dataset_loaded = io.load(self._test_dir)
    self.assertDatasetsEqual(dataset, dataset_loaded)

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatAndPrefetch(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue: saves random float data, loads it, then applies
    # shuffle, batch, infinite repeat and prefetch, and pulls thirty elements without error.
    """This test reproduces github.com/tensorflow/tensorflow/issues/49165."""
    dataset1 = dataset_ops.Dataset.from_tensor_slices(np.random.rand(16, 32))
    self.evaluate(io.save(dataset1, self._test_dir))
    dataset = io.load(self._test_dir)
    dataset = dataset.shuffle(buffer_size=16)
    dataset = dataset.batch(16)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    next_element = self.getNext(dataset)
    for _ in range(30):
      self.evaluate(next_element())


class LoadCheckpointTest(IOTest, checkpoint_test_base.CheckpointTestBase):
  # [CLASS DOC]
  # Checkpoint tests for datasets loaded from disk. Inherits (and so re-runs) all IOTest tests.

  def _build_ds(self):
    # [FUNCTION DOC]
    # HELPER: returns a dataset loaded from the save directory, for the checkpoint verification.
    return io.load(self._save_dir)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    # [FUNCTION DOC]
    # Saves a range of 42 to the save directory and runs the checkpoint verification callback on
    # the loaded dataset, expecting 42 outputs.
    dataset = dataset_ops.Dataset.range(42)
    self.evaluate(io.save(dataset, self._save_dir))
    verify_fn(self, self._build_ds, num_outputs=42)


class SaveCheckpointTest(IOTest, checkpoint_test_base.CheckpointTestBase):
  # [CLASS DOC]
  # Tests the checkpoint arguments of save (checkpoint during saving). Inherits IOTest tests.

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveCheckpointingAPI(self):
    # [FUNCTION DOC]
    # Eager only. Saves a range of 40 with checkpoint arguments (directory, keep up to 50) and
    # asserts 81 files exist in the checkpoint directory (data and index per step plus one
    # overall file).
    dataset = dataset_ops.Dataset.range(40)
    checkpoint_args = {"directory": self._checkpoint_prefix, "max_to_keep": 50}
    io.save(dataset, self._save_dir, checkpoint_args=checkpoint_args)
    num_checkpoint_files = len(list(os.listdir(self._checkpoint_prefix)))
    # By default, we checkpoint every increment. Each checkpoint writes a
    # file containing the data and a file containing the index. There is
    # also an overall checkpoint file. Thus, we expect (2 * 40) + 1 files.
    self.assertEqual(81, num_checkpoint_files)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveCheckpointingAPICustomCheckpointInterval(self):
    # [FUNCTION DOC]
    # Eager only. Saves a range of 40 with a checkpoint interval of 5, a step counter variable and
    # keep-at-most 10; asserts 17 files exist in the checkpoint directory.
    dataset = dataset_ops.Dataset.range(40)
    step_counter = variables.Variable(0, trainable=False)
    checkpoint_args = {
        "checkpoint_interval": 5,
        "step_counter": step_counter,
        "directory": self._checkpoint_prefix,
        "max_to_keep": 10,
    }
    io.save(dataset, self._save_dir, checkpoint_args=checkpoint_args)
    num_checkpoint_files = len(list(os.listdir(self._checkpoint_prefix)))
    # We expect (2 * 8) + 1 files.
    self.assertEqual(17, num_checkpoint_files)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveCheckpointingAPIIncorrectArgs(self):
    # [FUNCTION DOC]
    # Eager only. Passing an unknown key in the checkpoint arguments to save raises a type error.
    dataset = dataset_ops.Dataset.range(42)
    checkpoint_args = {
        "directory": self._checkpoint_prefix,
        "incorrect_arg": "incorrect_arg"
    }
    with self.assertRaises(TypeError):
      io.save(dataset, self._save_dir, checkpoint_args=checkpoint_args)


if __name__ == "__main__":
  test.main()
