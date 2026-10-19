# [MODULE DOC]
# Tests for the dataset fingerprint operation, which computes a hash identifying a dataset's
# graph. Provides two helper functions generating named parameter combinations (single datasets
# and pairs of datasets built from range, flat_map and TFRecord readers) and a test class checking
# that identical datasets have equal fingerprints and different datasets have different ones.
# No import-time side effects; running as a script starts the TensorFlow test runner.
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `dataset.fingerprint()`."""

from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


def dataset_fn_test_cases():
  # [FUNCTION DOC]
  # HELPER: produces parameter combinations for the fingerprint tests in this module.
  # Takes no inputs. Returns a combination of test parameters with one named entry per case, each
  # wrapping a factory that builds a dataset: a range, a range flat-mapped to ranges, a TFRecord
  # dataset over two file names, and the same repeated twice. The files are never read.
  cases = [
      ("range1", lambda: dataset_ops.Dataset.range(10)),
      (
          "flat_map1",
          lambda: dataset_ops.Dataset.range(10).flat_map(
              dataset_ops.Dataset.range
          ),
      ),
      ("tfrecord1", lambda: readers.TFRecordDataset(["f1.txt", "f2.txt"])),
      (
          "tfrecord2",
          lambda: readers.TFRecordDataset(["f1.txt", "f2.txt"]).repeat(2),
      ),
  ]

  named_cases = []
  for case in cases:
    name, dataset_fn = case
    named_cases.append(combinations.NamedObject(name=name, obj=dataset_fn))

  return combinations.combine(dataset_fn=named_cases)


def dataset_pair_fn_test_cases():
  # [FUNCTION DOC]
  # HELPER: produces parameter combinations for the fingerprint inequality test in this module.
  # Takes no inputs. Returns a combination of test parameters with one named entry per case, each
  # wrapping a factory that returns a pair of datasets that differ (ranges of different length,
  # flat-maps with different inner functions, TFRecord datasets over different file lists, and the
  # repeated variants of the latter). The files are never read.
  dataset = dataset_ops.Dataset

  cases = [
      ("range1", lambda: (dataset.range(10), dataset.range(11))),
      (
          "flat_map1",
          lambda: (
              dataset.range(10).flat_map(dataset.range),
              dataset.range(10).flat_map(lambda x: dataset.range(x + 1)),
          ),
      ),
      (
          "tfrecord1",
          lambda: (
              readers.TFRecordDataset(["f1.txt", "f2.txt"]),
              readers.TFRecordDataset(["f1.txt", "f3.txt"]),
          ),
      ),
      (
          "tfrecord2",
          lambda: (
              readers.TFRecordDataset(["f1.txt", "f2.txt"]).repeat(2),
              readers.TFRecordDataset(["f1.txt", "f3.txt"]).repeat(2),
          ),
      ),
  ]

  named_cases = []
  for case in cases:
    name, dataset_pair_fn = case
    named_cases.append(combinations.NamedObject(name=name, obj=dataset_pair_fn))

  return combinations.combine(dataset_pair_fn=named_cases)


class FingerprintTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the dataset fingerprint method. Parameterized over the case generators defined in
  # this module. No shared fixtures.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(), dataset_fn_test_cases()
      )
  )
  def testSameDatasetSameFingerprint(self, dataset_fn):
    # [FUNCTION DOC]
    # Scenario: calls the dataset factory twice to build two independent identical datasets,
    # evaluates the fingerprint of each, and asserts the fingerprints are equal. Input is the
    # factory supplied by the parameter combination. Returns nothing.
    fingerprint1 = self.evaluate(dataset_fn().fingerprint())
    fingerprint2 = self.evaluate(dataset_fn().fingerprint())
    self.assertEqual(fingerprint1, fingerprint2)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          dataset_pair_fn_test_cases(),
      )
  )
  def testDifferentDatasetDifferentFingerprint(self, dataset_pair_fn):
    # [FUNCTION DOC]
    # Scenario: calls the factory to get a pair of different datasets, evaluates the fingerprint
    # of each, and asserts the two fingerprints differ. Input is the pair factory supplied by the
    # parameter combination. Returns nothing.
    lhs, rhs = dataset_pair_fn()
    lhs_fingerprint = self.evaluate(lhs.fingerprint())
    rhs_fingerprint = self.evaluate(rhs.fingerprint())
    self.assertNotEqual(lhs_fingerprint, rhs_fingerprint)


if __name__ == "__main__":
  test.main()
