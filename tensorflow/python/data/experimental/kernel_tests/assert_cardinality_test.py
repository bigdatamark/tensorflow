# [MODULE DOC]
# Unit tests for the experimental assert-cardinality transformation, which declares the number of
# elements a dataset will produce and fails at runtime if the real count differs. Covers correct
# and incorrect assertions, interaction with global shuffle, random access, and checkpointing
# (regular and symbolic). Test module, no import-time side effects; run directly it starts the
# test runner.
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
"""Tests for `tf.data.experimental.assert_cardinality()`."""
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class AssertCardinalityTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the assert-cardinality transformation: valid and invalid cardinality assertions,
  # error messages, global shuffle behavior and random access. Uses dataset test base helpers and
  # parameterized combinations; no shared fixtures.
  """Tests for `tf.data.experimental.assert_cardinality()`."""

  @combinations.generate(test_base.default_test_combinations())
  def testCorrectCardinality(self):
    # [FUNCTION DOC]
    # Scenario: a range of ten elements passed through an always-true filter has unknown
    # cardinality. Asserts it is unknown and yields zero to nine; after applying assert-cardinality
    # of ten, asserts the cardinality is ten and the same elements are still produced.
    dataset = dataset_ops.Dataset.range(10).filter(lambda x: True)
    self.assertEqual(
        self.evaluate(cardinality.cardinality(dataset)), cardinality.UNKNOWN)
    self.assertDatasetProduces(dataset, expected_output=range(10))
    dataset = dataset.apply(cardinality.assert_cardinality(10))
    self.assertEqual(self.evaluate(cardinality.cardinality(dataset)), 10)
    self.assertDatasetProduces(dataset, expected_output=range(10))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_elements=10,
              asserted_cardinality=20,
              expected_error="Input dataset was expected to contain 20 "
              "elements but contained only 10 elements.") +
          combinations.combine(
              num_elements=1,
              asserted_cardinality=20,
              expected_error="Input dataset was expected to contain 20 "
              "elements but contained only 1 element.") +
          combinations.combine(
              num_elements=10,
              asserted_cardinality=cardinality.INFINITE,
              expected_error="Input dataset was expected to contain an "
              "infinite number of elements but contained only 10 elements.") +
          combinations.combine(
              num_elements=1,
              asserted_cardinality=cardinality.INFINITE,
              expected_error="Input dataset was expected to contain an "
              "infinite number of elements but contained only 1 element.") +
          combinations.combine(
              num_elements=10,
              asserted_cardinality=5,
              expected_error="Input dataset was expected to contain 5 "
              "elements but contained at least 6 elements.") +
          combinations.combine(
              num_elements=10,
              asserted_cardinality=1,
              expected_error="Input dataset was expected to contain 1 "
              "element but contained at least 2 elements.")))
  def testIncorrectCardinality(self, num_elements, asserted_cardinality,
                               expected_error):
    # [FUNCTION DOC]
    # Parameterized scenario: a range of num_elements (int) with an asserted cardinality
    # (int, possibly the infinite marker) that disagrees with reality. Iterates the dataset until
    # exhaustion and asserts a failed-precondition error whose message matches expected_error
    # (text pattern): too few elements ("contained only") or too many ("contained at least"),
    # with singular/plural wording for one element.
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.apply(
        cardinality.assert_cardinality(asserted_cardinality))
    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(errors.FailedPreconditionError, expected_error):
      while True:
        self.evaluate(get_next())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_elements=10,
              asserted_cardinality=100,
              expected_error=errors.FailedPreconditionError,
              expected_error_message=(
                  "Input dataset was expected to contain 100 elements.")) +
          combinations.combine(
              num_elements=10,
              asserted_cardinality=cardinality.INFINITE,
              expected_error=errors.InvalidArgumentError,
              expected_error_message=(
                  "`global_shuffle` requires the input dataset to have a "
                  "non-empty finite cardinality."))))
  def testIncorrectCardinalityForGlobalShuffle(
      self,
      num_elements: int,
      asserted_cardinality: int,
      expected_error: Exception,
      expected_error_message: str):
    # [FUNCTION DOC]
    # Parameterized scenario: global shuffle applied to a range of num_elements (int) with a wrong
    # asserted cardinality. Asserts the raised error type (expected_error) and message pattern
    # (expected_error_message): a failed-precondition error when the asserted count (100) exceeds
    # the real count, or an invalid-argument error saying global shuffle needs a non-empty finite
    # cardinality when the asserted cardinality is infinite.
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.apply(
        cardinality.assert_cardinality(asserted_cardinality))
    with self.assertRaisesRegex(
        expected_error, expected_error_message):
      dataset = global_shuffle_op._global_shuffle(dataset)
      self.getDatasetOutput(dataset, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testRandomAccess(self):
    # [FUNCTION DOC]
    # Scenario: a ten-element range with a correct asserted cardinality; asserts random access
    # verification returns each of the ten expected elements by index.
    num_elements = 10
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.apply(cardinality.assert_cardinality(num_elements))
    self.verifyRandomAccess(dataset, expected=range(num_elements))

  @combinations.generate(test_base.default_test_combinations())
  def testRandomAccessOutOfRange(self):
    # [FUNCTION DOC]
    # Scenario: a five-element range with an asserted cardinality of ten. Random access at index
    # ten is expected to raise an out-of-range error; at index five (inside the asserted size but
    # past the real data) a failed-precondition error.
    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.apply(cardinality.assert_cardinality(10))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=10))
    with self.assertRaises(errors.FailedPreconditionError):
      self.evaluate(random_access.at(dataset, index=5))


class AssertCardinalityCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                      parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint (save/restore) test for datasets using assert-cardinality, built on the checkpoint
  # test base class. Shared fixture: the dataset builder method.

  def build_dataset(self, num_elements, options=None):
    # [FUNCTION DOC]
    # HELPER: dataset factory for the checkpoint test. Builds a range of num_elements (int) with a
    # matching asserted cardinality and applies the given options (optional; skipped when absent).
    # Returns the dataset.
    dataset = dataset_ops.Dataset.range(num_elements).apply(
        cardinality.assert_cardinality(num_elements))
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
    # Parameterized scenario: sets the symbolic checkpoint option (bool) and runs the supplied
    # verification callback on a 200-element dataset, expecting 200 outputs to survive checkpoint
    # save and restore at various points.
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(self, lambda: self.build_dataset(200, options), num_outputs=200)


if __name__ == "__main__":
  test.main()
