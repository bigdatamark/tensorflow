# [MODULE DOC]
# Unit tests for the testing-only assert-prev transformation, which checks that the
# transformations preceding it have the listed names and attributes. Covers a match, version-suffix
# tolerance, a wrong name, a list longer than the pipeline and an unknown attribute name. Test
# module; run directly it starts the test runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.assert_prev()`."""
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class AssertPrevTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the assert-prev transformation, run across the default eager/graph test
  # combinations. No shared fixtures.

  @combinations.generate(test_base.default_test_combinations())
  def testAssertPrev(self):
    # [FUNCTION DOC]
    # Scenario: a deterministic parallel map (8 calls) followed by assert-prev naming a parallel map
    # dataset with a deterministic attribute; optimizations disabled. Asserts the single element
    # zero is produced.
    # KNOWN ISSUE: the attribute argument is written as a set of two strings rather than a mapping
    # from attribute name to value, so it does not actually express the intended attribute check.
    dataset = dataset_ops.Dataset.from_tensors(0).map(
        lambda x: x, deterministic=True, num_parallel_calls=8).apply(
            testing.assert_prev([("ParallelMapDataset",
                                  {"deterministic", "true"})]))
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])

  @combinations.generate(test_base.default_test_combinations())
  def testIgnoreVersionSuffix(self):
    # [FUNCTION DOC]
    # Scenario: deterministic parallel map then batch, then assert-prev expecting a batch followed
    # by a parallel map with deterministic true; the real batch op carries a version suffix but
    # must match. Optimizations disabled. Asserts the output is one batch containing zero.
    # The `batch` transformation creates a "BatchV2" dataset, but we should
    # still match that with "Batch".
    dataset = dataset_ops.Dataset.from_tensors(0).map(
        lambda x: x, deterministic=True, num_parallel_calls=8).batch(1).apply(
            testing.assert_prev([("BatchDataset", {}),
                                 ("ParallelMapDataset", {
                                     "deterministic": "true"
                                 })]))
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[[0]])

  @combinations.generate(test_base.default_test_combinations())
  def testAssertPrevInvalid(self):
    # [FUNCTION DOC]
    # Scenario: assert-prev names a non-existent transformation. Asserts an invalid-argument error
    # saying the asserted transformation did not match.
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_prev([("Whoops", {})]))
    self.assertDatasetProduces(
        dataset,
        expected_error=(errors.InvalidArgumentError,
                        "Asserted transformation matching 'Whoops'"))

  @combinations.generate(test_base.default_test_combinations())
  def testAssertPrevShort(self):
    # [FUNCTION DOC]
    # Scenario: assert-prev lists a tensor dataset and a non-existent transformation but only one
    # precedes it. Asserts an invalid-argument error stating two were asserted but only one found.
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_prev([("TensorDataset", {}), ("Whoops", {})]))
    self.assertDatasetProduces(
        dataset,
        expected_error=(
            errors.InvalidArgumentError,
            "Asserted previous 2 transformations but encountered only 1."))

  @combinations.generate(test_base.default_test_combinations())
  def testAssertBadAttributeName(self):
    # [FUNCTION DOC]
    # Scenario: assert-prev expects a tensor dataset with an attribute that does not exist.
    # Asserts an invalid-argument error saying no such attribute was found.
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_prev([("TensorDataset", {
            "whoops": "true"
        })]))
    self.assertDatasetProduces(
        dataset,
        expected_error=(errors.InvalidArgumentError, "found no such attribute"))


if __name__ == "__main__":
  test.main()
