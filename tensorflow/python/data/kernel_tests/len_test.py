# [MODULE DOC]
# Test module for the length protocol of datasets in the data kernel tests. It checks that taking
# the length of a dataset works in eager mode for datasets of known size, and raises a type error
# for infinite, unknown-size, or graph-mode datasets. Defines one test class and runs the test
# main entry point when executed as a script. No import-time side effects.
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
"""Tests for `tf.data.Dataset.__len__()`."""
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class LenTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the length operation of datasets. Uses the standard dataset test base and
  # parameterized test support; each test is restricted to eager or graph mode via combinations.

  @combinations.generate(test_base.eager_only_combinations())
  def testKnown(self):
    # [FUNCTION DOC]
    # Eager-mode test: a dataset of ten range elements has a known size, so its length is
    # asserted to be ten.
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    self.assertLen(ds, 10)

  @combinations.generate(test_base.eager_only_combinations())
  def testInfinite(self):
    # [FUNCTION DOC]
    # Eager-mode test: a ten-element range dataset repeated forever is infinite, so taking its
    # length must raise a type error whose message mentions "infinite".
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements).repeat()
    with self.assertRaisesRegex(TypeError, "infinite"):
      len(ds)

  @combinations.generate(test_base.eager_only_combinations())
  def testUnknown(self):
    # [FUNCTION DOC]
    # Eager-mode test: filtering a range dataset makes its size unknown, so taking its length
    # must raise a type error whose message mentions "unknown".
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements).filter(lambda x: True)
    with self.assertRaisesRegex(TypeError, "unknown"):
      len(ds)

  @combinations.generate(test_base.graph_only_combinations())
  def testGraphMode(self):
    # [FUNCTION DOC]
    # Graph-mode test: taking the length of a range dataset must raise a type error saying that
    # datasets only support length in eager mode and suggesting the cardinality method instead.
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    with self.assertRaisesRegex(
        TypeError,
        r"`tf.data.Dataset` only supports `len` in eager mode. Use "
        r"`tf.data.Dataset.cardinality\(\)` instead."):
      len(ds)


if __name__ == "__main__":
  test.main()
