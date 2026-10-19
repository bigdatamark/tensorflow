# [MODULE DOC]
# Unit tests for the testing-only assert-next transformation, which checks at graph build/run time
# that the transformations following it in the pipeline have the listed names. Covers a match,
# version-suffix tolerance, a wrong name and a list longer than the pipeline. Test module; run
# directly it starts the test runner.
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
"""Tests for `tf.data.experimental.assert_next()`."""
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class AssertNextTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the assert-next transformation, run across the default eager/graph test
  # combinations. No shared fixtures.

  @combinations.generate(test_base.default_test_combinations())
  def testAssertNext(self):
    # [FUNCTION DOC]
    # Scenario: assert-next expecting a map is followed by an identity map, with default
    # optimizations disabled so the map is not rewritten. Asserts the dataset produces the single
    # element zero.
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_next(["Map"])).map(lambda x: x)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])

  @combinations.generate(test_base.default_test_combinations())
  def testIgnoreVersionSuffix(self):
    # [FUNCTION DOC]
    # Scenario: assert-next expects a map then a batch; the real batch op is a versioned name but
    # must still match. Optimizations disabled. Asserts the output is one batch containing zero.
    # The `batch` transformation creates a "BatchV2" dataset, but we should
    # still match that with "Batch".
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_next(["Map", "Batch"])).map(lambda x: x).batch(1)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[[0]])

  @combinations.generate(test_base.default_test_combinations())
  def testAssertNextInvalid(self):
    # [FUNCTION DOC]
    # Scenario: assert-next names a non-existent transformation. Asserts iterating raises an
    # invalid-argument error saying the asserted transformation did not match.
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_next(["Whoops"]))
    self.assertDatasetProduces(
        dataset,
        expected_error=(errors.InvalidArgumentError,
                        "Asserted transformation matching Whoops"))

  @combinations.generate(test_base.default_test_combinations())
  def testAssertNextShort(self):
    # [FUNCTION DOC]
    # Scenario: assert-next lists two transformations but fewer follow it (optimizations
    # disabled). Asserts an invalid-argument error stating two were asserted but only one was found.
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_next(["Root", "Whoops"]))
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset,
        expected_error=(
            errors.InvalidArgumentError,
            "Asserted next 2 transformations but encountered only 1."))


if __name__ == "__main__":
  test.main()
