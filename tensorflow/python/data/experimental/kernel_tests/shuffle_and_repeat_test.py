# [MODULE DOC]
# Tests for the experimental shuffle-and-repeat dataset transformation: output correctness,
# per-epoch reshuffling, seed determinism, count of none or minus one meaning infinite repeat,
# very large buffer sizes, re-randomization on graph round trip, and checkpoint save/restore.
# No import-time side effects.
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
"""Tests for `tf.data.experimental.shuffle_and_repeat()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import shuffle_ops
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test


class ShuffleAndRepeatTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Behavioral tests for shuffle-and-repeat on range datasets. Uses a shared dataset builder and
  # an output-collecting helper.

  def _build_ds(self, seed, count=5, num_elements=20):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the tests in this class.
    # Returns a range dataset of num_elements (default 20) with shuffle-and-repeat applied using
    # buffer size 5, the given repeat count (default 5; none or minus one means infinite) and seed.
    return dataset_ops.Dataset.range(num_elements).apply(
        shuffle_ops.shuffle_and_repeat(buffer_size=5, count=count, seed=seed))

  def _gen_outputs(self, ds_fn, num_outputs, verify_exhausted=True):
    # [FUNCTION DOC]
    # HELPER: collects outputs from a dataset. Takes a callback building the dataset, the number of
    # elements to read, and whether to verify exhaustion afterwards (default true). Returns the list
    # of evaluated elements. If verification is on, asserts that one more read raises an
    # out-of-range error (so the assertion itself fails for infinite datasets).
    get_next = self.getNext(ds_fn())
    outputs = []
    for _ in range(num_outputs):
      outputs.append(self.evaluate(get_next()))
    if verify_exhausted:
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())
    return outputs

  @combinations.generate(test_base.default_test_combinations())
  def testCorrectOutput(self):
    # [FUNCTION DOC]
    # Reads all 100 outputs (5 epochs of 20) and asserts the multiset equals five copies of 0..19
    # and that each consecutive block of 20 is a permutation of 0..19.
    output = self._gen_outputs(lambda: self._build_ds(10), 100)
    self.assertSequenceEqual(
        sorted(output), sorted(
            np.array([range(20) for _ in range(5)]).flatten()))
    for i in range(5):
      self.assertSequenceEqual(sorted(output[i * 20:(i + 1) * 20]), range(20))

  @combinations.generate(test_base.default_test_combinations())
  def testReshuffling(self):
    # [FUNCTION DOC]
    # Asserts that the element order of each epoch differs from the order of the next epoch.
    # Check that the output orders of different epochs are indeed different.
    output = self._gen_outputs(lambda: self._build_ds(10), 100)
    for i in range(4):
      epoch1 = output[i * 20:(i + 1) * 20]
      epoch2 = output[(i + 1) * 20:(i + 2) * 20]
      self.assertNotEqual(epoch1, epoch2)

  @combinations.generate(test_base.default_test_combinations())
  def testSameOrderForSameSeeds(self):
    # [FUNCTION DOC]
    # Builds the dataset twice with the same seed and asserts identical output sequences.
    output1 = self._gen_outputs(lambda: self._build_ds(10), 100)
    output2 = self._gen_outputs(lambda: self._build_ds(10), 100)
    self.assertEqual(output1, output2)

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentOrderForDifferentSeeds(self):
    # [FUNCTION DOC]
    # Uses two different seeds and asserts the outputs differ in order but contain the same
    # elements.
    output1 = self._gen_outputs(lambda: self._build_ds(10), 100)
    output2 = self._gen_outputs(lambda: self._build_ds(20), 100)
    self.assertNotEqual(output1, output2)
    self.assertCountEqual(output1, output2)

  @combinations.generate(test_base.default_test_combinations())
  def testCountNone(self):
    # [FUNCTION DOC]
    # With an unset repeat count (infinite repeat) and different seeds, reads 100 elements without
    # checking exhaustion; asserts different order but same elements.
    output1 = self._gen_outputs(
        lambda: self._build_ds(10, count=None), 100, verify_exhausted=False)
    output2 = self._gen_outputs(
        lambda: self._build_ds(20, count=None), 100, verify_exhausted=False)
    self.assertNotEqual(output1, output2)
    self.assertCountEqual(output1, output2)

  @combinations.generate(test_base.default_test_combinations())
  def testCountMinusOne(self):
    # [FUNCTION DOC]
    # Same as the unset-count test but with count minus one, which also means infinite repeat.
    output1 = self._gen_outputs(
        lambda: self._build_ds(10, count=-1), 100, verify_exhausted=False)
    output2 = self._gen_outputs(
        lambda: self._build_ds(20, count=-1), 100, verify_exhausted=False)
    self.assertNotEqual(output1, output2)
    self.assertCountEqual(output1, output2)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteOutputs(self):
    # [FUNCTION DOC]
    # Asserts that, for infinite repeat (count none or minus one), the exhaustion check after 100
    # items fails with an assertion error since the dataset never ends.
    # Asserting the iterator is exhausted after producing 100 items should fail.
    with self.assertRaises(AssertionError):
      self._gen_outputs(lambda: self._build_ds(10, count=None), 100)
    with self.assertRaises(AssertionError):
      self._gen_outputs(lambda: self._build_ds(10, count=-1), 100)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteEmpty(self):
    # [FUNCTION DOC]
    # An empty input with infinite repeat must end immediately: reading raises an out-of-range
    # error for both count none and count minus one.
    with self.assertRaises(errors.OutOfRangeError):
      self._gen_outputs(lambda: self._build_ds(10, count=None, num_elements=0),
                        100)
    with self.assertRaises(errors.OutOfRangeError):
      self._gen_outputs(lambda: self._build_ds(10, count=-1, num_elements=0),
                        100)

  @combinations.generate(test_base.default_test_combinations())
  def testLargeBufferSize(self):
    # [FUNCTION DOC]
    # Uses a buffer size (21) larger than the 20-element input and asserts that the first element
    # can still be read without error.
    ds = dataset_ops.Dataset.range(20).apply(
        shuffle_ops.shuffle_and_repeat(buffer_size=21))
    get_next = self.getNext(ds)
    self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testVeryLargeBufferSize(self):
    # [FUNCTION DOC]
    # Uses a huge buffer size and a million epochs over batched data; verifies that the buffer is
    # not eagerly allocated (no out-of-memory or timeout) by reading two epochs (10 batches) and
    # asserting each epoch, sorted by first value, equals the 500 numbers reshaped into 5 batches.
    num_epochs = 1000 * 1000
    # Each element being shuffled and repeated has shape (100,). This will OOM
    # or timeout if we actually load everything into the buffer.
    ds = dataset_ops.Dataset.range(500).batch(100).apply(
        shuffle_ops.shuffle_and_repeat(
            buffer_size=5 * num_epochs, count=num_epochs))
    # Verify two epochs worth of output.
    output = self._gen_outputs(lambda: ds, 2 * 5, verify_exhausted=False)
    for i in range(2):
      sorted_epoch = sorted(
          output[i * 5:(i + 1) * 5], key=lambda batch: batch[0])
      self.assertAllEqual(sorted_epoch, np.arange(500).reshape([5, 100]))

  @combinations.generate(test_base.default_test_combinations())
  def testRerandomizeOnReplicate(self):
    # [FUNCTION DOC]
    # Clears the global random seed, reads the full output, round-trips the dataset through a graph
    # (replication), reads again, and asserts the same elements in a different order.
    random_seed.set_random_seed(None)
    # When no seeds are fixed, each instantiation of the dataset should
    # produce elements in a different order.
    num_epochs = 2
    num_elements = 100
    ds = dataset_ops.Dataset.range(num_elements).apply(
        shuffle_ops.shuffle_and_repeat(
            buffer_size=num_elements, count=num_epochs))

    shuffle_1 = self.getDatasetOutput(ds)
    ds = self.graphRoundTrip(ds)
    shuffle_2 = self.getDatasetOutput(ds)

    self.assertCountEqual(shuffle_1, shuffle_2)
    self.assertNotEqual(shuffle_1, shuffle_2)


class ShuffleAndRepeatCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                     parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint (save/restore) tests for shuffle-and-repeat using the shared checkpoint test base.

  def _build_ds(self, seed):
    # [FUNCTION DOC]
    # HELPER: builds a range of 20 with shuffle-and-repeat (buffer 5, count 5) using the given seed.
    return dataset_ops.Dataset.range(20).apply(
        shuffle_ops.shuffle_and_repeat(buffer_size=5, count=5, seed=seed))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):
    # [FUNCTION DOC]
    # Parameterized over the standard checkpoint verification functions; runs the verifier on the
    # seeded dataset with 100 expected outputs.
    verify_fn(self, lambda: self._build_ds(10), num_outputs=100)


if __name__ == "__main__":
  test.main()
