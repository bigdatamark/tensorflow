# [MODULE DOC]
# Kernel tests for the dataset prefetch transformation. Covers valid and invalid buffer sizes
# (including automatic and negative values), how many upstream elements get fetched ahead of
# consumption, prefetching with a slack period, cancellation of a blocked prefetch when the
# session closes, the name argument, iterator checkpointing with and without symbolic
# checkpoints, and random-access reads (valid, out-of-range and empty cases). No import-time
# side effects other than the test entry point.
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
"""Tests for `tf.data.Dataset.prefetch()`."""
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import prefetch_op
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class PrefetchTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the prefetch transformation: buffer sizes, read-ahead behavior, slack,
  # cancellation and naming.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(buffer_size=[-1, None, 0, 42])))
  def testBufferSize(self, buffer_size):
    # [FUNCTION DOC]
    # Scenario: prefetch over a ten-element range with buffer size minus one, unset, zero or
    # forty-two. Expected: the dataset always produces zero through nine in order.
    dataset = dataset_ops.Dataset.range(10).prefetch(buffer_size=buffer_size)
    self.assertDatasetProduces(dataset, expected_output=range(10))

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(buffer_size=[0, 1, 2, 42])))
  def testPrefetching(self, buffer_size):
    # [FUNCTION DOC]
    # Eager-only scenario: a thousand-element range is mapped by a Python function that counts
    # its invocations, then prefetched with the given buffer size; ten elements are consumed.
    # Expected: after waiting for the buffer to fill and a short extra pause, the map function
    # was called exactly ten plus buffer-size times (reads ahead by exactly the buffer size).
    dataset = dataset_ops.Dataset.range(1000)

    calls = 0

    @script_ops.eager_py_func(Tout=[dtypes.int64])
    def map_fn(x):
      # [FUNCTION DOC]
      # HELPER: map function (run as an eager Python function returning a 64-bit integer) that
      # increments the enclosing call counter and returns its input unchanged.
      nonlocal calls
      calls += 1
      return x

    dataset = dataset.map(map_fn)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    it = iter(dataset)
    for _ in range(10):
      next(it)

    # Wait for the prefetch buffer to fill up.
    while calls != 10+buffer_size:
      time.sleep(0.1)
    # Wait some extra time to make sure the prefetch buffer isn't fetching more
    # elements than it should.
    time.sleep(0.5)
    self.assertEqual(calls, 10+buffer_size)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(buffer_size=[-2, -42])))
  def testInvalidBufferSize(self, buffer_size):
    # [FUNCTION DOC]
    # Scenario: prefetch with a buffer size of minus two or minus forty-two (below the allowed
    # minus-one sentinel). Expected: evaluating the dataset raises an invalid-argument error.
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).prefetch(buffer_size=buffer_size)
      self.evaluate(dataset._variant_tensor)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              buffer_size=[-1, None, 0, 42], slack_period=[1, 8])))
  def testPrefetchWithSlack(self, buffer_size, slack_period):
    # [FUNCTION DOC]
    # Scenario: the internal prefetch dataset class is built directly over a hundred-element
    # range with the given buffer size and slack period (one or eight). Expected: the dataset
    # still produces zero through ninety-nine in order.
    dataset = dataset_ops.Dataset.range(100)
    dataset = prefetch_op._PrefetchDataset(  # pylint: disable=protected-access
        dataset, buffer_size, slack_period=slack_period)
    self.assertDatasetProduces(dataset, expected_output=range(100))

  @combinations.generate(combinations.combine(tf_api_version=1, mode="graph"))
  def testPrefetchCancellation(self):
    # [FUNCTION DOC]
    # Graph-mode (version-one API) scenario: a map function that never terminates feeds a
    # prefetch of three. A background thread requests the next element; after two seconds the
    # session is closed. Expected: the blocked operation is cancelled and the thread finishes
    # (the helper asserts the cancelled status).

    def map_py_fn(x):
      # [FUNCTION DOC]
      # HELPER: deliberately non-terminating map function (busy loop while the value exceeds
      # minus one, which is always true for non-negative input) used to keep prefetch blocked
      # until cancelled. Never returns for the dataset's inputs.
      while x > -1:
        x = x * 1
      return x

    dataset = dataset_ops.Dataset.range(10).map(map_py_fn).prefetch(3)
    get_next = self.getNext(dataset)

    with self.cached_session() as sess:
      thread = self.checkedThread(self.assert_op_cancelled, args=(get_next(),))
      thread.start()
      time.sleep(2)
      sess.close()
      thread.join()

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Scenario: prefetch of one with an explicit name over a single-element dataset. Expected:
    # the dataset produces 42.
    dataset = dataset_ops.Dataset.from_tensors(42).prefetch(1, name="prefetch")
    self.assertDatasetProduces(dataset, [42])


class PrefetchCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                             parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint test case for prefetch; builds a prefetched range pipeline and uses the shared
  # checkpoint test base to verify save and restore of iterator state.

  def build_dataset(self, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the pipeline under test: a hundred-element range with prefetch of ten.
    # Parameter options is an optional dataset options object; when given it is applied.
    # Returns the dataset.
    dataset = dataset_ops.Dataset.range(100).prefetch(10)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Scenario: checkpoint verification of the prefetched pipeline with symbolic checkpointing
    # on or off. Parameters: verify_fn (checkpoint verification strategy from the base class)
    # and symbolic_checkpoint flag. Expected: verification passes with all hundred outputs.
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(self, lambda: self.build_dataset(options), num_outputs=100)


class PrefetchRandomAccessTest(test_base.DatasetTestBase,
                               parameterized.TestCase):
  # [CLASS DOC]
  # Test case for random access (reading an element by index) on prefetched datasets: invalid
  # indices, empty datasets and many size/buffer combinations.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 10, 11])))
  def testInvalidIndex(self, index):
    # [FUNCTION DOC]
    # Scenario: random access at index minus one, ten or eleven into a prefetched ten-element
    # dataset. Expected: an out-of-range error.
    dataset = dataset_ops.Dataset.range(10).prefetch(buffer_size=5)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-2, 0, 1])))
  def testEmptyDataset(self, index):
    # [FUNCTION DOC]
    # Scenario: random access at index minus two, zero or one into a prefetched empty dataset.
    # Expected: an out-of-range error.
    dataset = dataset_ops.Dataset.from_tensor_slices([]).prefetch(buffer_size=5)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(elements=[10, 50, 100], buffer_size=[0, 5, 10])))
  def testMultipleCombinations(self, elements, buffer_size):
    # [FUNCTION DOC]
    # Scenario: a prefetched range of the given length and buffer size is read by index.
    # Expected: every valid index returns the matching value and the index equal to the
    # cardinality raises an out-of-range error.
    dataset = dataset_ops.Dataset.range(elements).prefetch(
        buffer_size=buffer_size)
    len_dataset = self.evaluate(dataset.cardinality())
    expected_output = np.arange(elements)
    for i in range(len_dataset):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)), expected_output[i])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=len_dataset))


if __name__ == "__main__":
  test.main()
