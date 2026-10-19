# [MODULE DOC]
# Tests for the experimental slack option of prefetch, which lets the prefetch buffer run ahead
# by a number of elements (period derived from the number of devices on a multi-device
# iterator, default one). Covers multi-device iteration, no iterator, passthrough datasets after
# prefetch, pipelines without prefetch, and nested datasets after prefetch (rewrite must not
# break results). No import-time side effects; runs the test main when executed directly.
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
"""Tests for `experimental_slack` option."""

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class PrefetchWithSlackTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the slack option of prefetch. Holds a list of three logical devices configured
  # in setUp for multi-device iterator tests.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the base setUp, then configures three logical devices for multi-device testing and
    # stores their names.
    super(PrefetchWithSlackTest, self).setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(3)

  @combinations.generate(test_base.default_test_combinations())
  def testPrefetchWithSlackOption(self):
    # [FUNCTION DOC]
    # Scenario: range of ten, prefetch of one, slack option enabled, consumed through a
    # multi-device iterator over the second and third devices. Expects pairs of consecutive
    # values per step (even on the first device, odd on the second), then out-of-range.
    """Determines slack_period based on num devices attached to iterator."""
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])
    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual(i, self.evaluate(elem_on_1))
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.default_test_combinations())
  def testPrefetchWithSlackOptionWithoutIterator(self):
    # [FUNCTION DOC]
    # Scenario: slack enabled with a plain dataset (no multi-device iterator); expects the
    # dataset still produces 0 to 9.
    """Defaults to slack period of 1 without iterator."""
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(10))

  @combinations.generate(test_base.default_test_combinations())
  def testWithPassthroughDataset(self):
    # [FUNCTION DOC]
    # Scenario: slack enabled, with a map (adding one) after prefetch. Expects values 1 to 10.
    """Should still work with a passthrough dataset after prefetch()."""
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    dataset = dataset.map(lambda x: x + 1)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(1, 11))

  @combinations.generate(test_base.default_test_combinations())
  def testNoErrorWithoutPrefetch(self):
    # [FUNCTION DOC]
    # Scenario: slack enabled but no prefetch in the pipeline; the rewrite must not fail and
    # the dataset must produce 0 to 9.
    """The rewrite should not fail if there is no prefetch() in the pipeline."""
    dataset = dataset_ops.Dataset.range(10)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(10))

  @combinations.generate(test_base.default_test_combinations())
  def testNoErrorWithInvalidDataset(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the rewrite should fail, but the test asserts the dataset
    # still produces all values, i.e. the failed rewrite must be harmless.
    # Scenario: slack enabled with a flat-map of nested datasets after prefetch; expects 0 to 9.
    """With a nested dataset op after prefetch, the rewrite should fail."""
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    dataset = dataset.flat_map(dataset_ops.Dataset.from_tensors)
    options = options_lib.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(10))


if __name__ == "__main__":
  test.main()
