# [MODULE DOC]
# Tests for the dataset range source. Covers output types, start/stop/step handling (including
# zero, negative and empty ranges), the name argument, checkpointing of range datasets (with and
# without symbolic checkpoints) and random access by index. Defines test classes only; no
# import-time side effects beyond the test main entry point.
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
"""Tests for `tf.data.Dataset.range()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class RangeTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for the range dataset source over several output types (32/64-bit integer
  # and float). Runs under the default graph/eager test combinations.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(output_type=[
              dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
          ])))
  def testStop(self, output_type):
    # [FUNCTION DOC]
    # Builds a range dataset with only a stop value of five and the given output type. Asserts
    # it produces zero through four in that dtype and that the dataset's element type matches.
    stop = 5
    dataset = dataset_ops.Dataset.range(stop, output_type=output_type)
    expected_output = np.arange(stop, dtype=output_type.as_numpy_dtype)
    self.assertDatasetProduces(dataset, expected_output=expected_output)
    self.assertEqual(output_type, dataset_ops.get_legacy_output_types(dataset))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(output_type=[
              dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
          ])))
  def testStartStop(self, output_type):
    # [FUNCTION DOC]
    # Builds a range dataset from start two to stop five with the given output type. Asserts the
    # produced values equal the numpy arange and the element type matches output_type.
    start, stop = 2, 5
    dataset = dataset_ops.Dataset.range(start, stop, output_type=output_type)
    expected_output = np.arange(start, stop, dtype=output_type.as_numpy_dtype)
    self.assertDatasetProduces(dataset, expected_output=expected_output)
    self.assertEqual(output_type, dataset_ops.get_legacy_output_types(dataset))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(output_type=[
              dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
          ])))
  def testStartStopStep(self, output_type):
    # [FUNCTION DOC]
    # Range from two to ten with step two for the given output type. Asserts the values match
    # numpy arange and the element type matches.
    start, stop, step = 2, 10, 2
    dataset = dataset_ops.Dataset.range(
        start, stop, step, output_type=output_type)
    expected_output = np.arange(
        start, stop, step, dtype=output_type.as_numpy_dtype)
    self.assertDatasetProduces(dataset, expected_output=expected_output)
    self.assertEqual(output_type, dataset_ops.get_legacy_output_types(dataset))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(output_type=[
              dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
          ])))
  def testZeroStep(self, output_type):
    # [FUNCTION DOC]
    # Builds a range dataset with step zero and evaluates its variant tensor. Asserts an
    # invalid-argument error is raised (during construction or evaluation).
    start, stop, step = 2, 10, 0
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(
          start, stop, step, output_type=output_type)
      self.evaluate(dataset._variant_tensor)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(output_type=[
              dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
          ])))
  def testNegativeStep(self, output_type):
    # [FUNCTION DOC]
    # Range from two down to ten with step minus one, which is empty. Asserts the output equals
    # the (empty) numpy arange and the element type matches.
    start, stop, step = 2, 10, -1
    dataset = dataset_ops.Dataset.range(
        start, stop, step, output_type=output_type)
    expected_output = np.arange(
        start, stop, step, dtype=output_type.as_numpy_dtype)
    self.assertDatasetProduces(dataset, expected_output=expected_output)
    self.assertEqual(output_type, dataset_ops.get_legacy_output_types(dataset))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(output_type=[
              dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
          ])))
  def testStopLessThanStart(self, output_type):
    # [FUNCTION DOC]
    # Range with start ten and stop two and default step. Asserts the result is empty, matching
    # numpy arange, with the right element type.
    start, stop = 10, 2
    dataset = dataset_ops.Dataset.range(start, stop, output_type=output_type)
    expected_output = np.arange(start, stop, dtype=output_type.as_numpy_dtype)
    self.assertDatasetProduces(dataset, expected_output=expected_output)
    self.assertEqual(output_type, dataset_ops.get_legacy_output_types(dataset))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(output_type=[
              dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
          ])))
  def testStopLessThanStartWithPositiveStep(self, output_type):
    # [FUNCTION DOC]
    # Range with start ten, stop two and positive step two. Asserts an empty output matching
    # numpy arange and the correct element type.
    start, stop, step = 10, 2, 2
    dataset = dataset_ops.Dataset.range(
        start, stop, step, output_type=output_type)
    expected_output = np.arange(
        start, stop, step, dtype=output_type.as_numpy_dtype)
    self.assertDatasetProduces(dataset, expected_output=expected_output)
    self.assertEqual(output_type, dataset_ops.get_legacy_output_types(dataset))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(output_type=[
              dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
          ])))
  def testStopLessThanStartWithNegativeStep(self, output_type):
    # [FUNCTION DOC]
    # Counting down: start ten, stop two, step minus one. Asserts the descending values match
    # numpy arange and the element type matches.
    start, stop, step = 10, 2, -1
    dataset = dataset_ops.Dataset.range(
        start, stop, step, output_type=output_type)
    expected_output = np.arange(
        start, stop, step, dtype=output_type.as_numpy_dtype)
    self.assertDatasetProduces(dataset, expected_output=expected_output)
    self.assertEqual(output_type, dataset_ops.get_legacy_output_types(dataset))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Builds a range of five with a custom dataset name and asserts it still yields zero to four.
    dataset = dataset_ops.Dataset.range(5, name="range")
    self.assertDatasetProduces(dataset, list(range(5)))


class RangeCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                          parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for the range dataset, using the shared checkpoint test base.

  def _build_range_dataset(self, start, stop, options=None):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint test.
    # Creates a range dataset from start to stop; if options (dataset options object, optional)
    # is given, applies them. Returns the dataset.
    dataset = dataset_ops.Dataset.range(start, stop)
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
    # Sets the symbolic-checkpoint option as parameterized, then runs the provided verification
    # callback on a range dataset from two to ten, expecting eight elements and correct
    # checkpoint round-tripping.
    start = 2
    stop = 10
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(self, lambda: self._build_range_dataset(start, stop, options),
              stop - start)


class RangeRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for index-based random access into range datasets via the random access helper.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 2, 3])))
  def testInvalidIndex(self, index):
    # [FUNCTION DOC]
    # Accessing a two-element range at a negative or too-large index must raise an out-of-range
    # error.
    dataset = dataset_ops.Dataset.range(2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 0])))
  def testEmptyDataset(self, index):
    # [FUNCTION DOC]
    # Random access into an empty range at index minus one or zero must raise an out-of-range
    # error.
    dataset = dataset_ops.Dataset.range(0)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testBasic(self):
    # [FUNCTION DOC]
    # For a range of ten, random access at each index from zero to nine returns the value equal
    # to that index.
    dataset = dataset_ops.Dataset.range(10)
    for i in range(10):
      self.assertEqual(self.evaluate(random_access.at(dataset, index=i)), i)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              start=[-1, 0, 5],
              stop=[-5, 0, 10],
              step=[-3, 1, 5],
              output_type=[
                  dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64
              ])))
  def testMultipleCombinations(self, start, stop, step, output_type):
    # [FUNCTION DOC]
    # For many combinations of start, stop, step and output type, builds a range and compares
    # random access at every valid index with numpy arange values; accessing the index equal to
    # the cardinality must raise an out-of-range error.
    dataset = dataset_ops.Dataset.range(
        start, stop, step, output_type=output_type)
    expected_output = np.arange(
        start, stop, step, dtype=output_type.as_numpy_dtype)
    len_dataset = self.evaluate(dataset.cardinality())
    for i in range(len_dataset):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)), expected_output[i])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=len_dataset))


if __name__ == "__main__":
  test.main()
