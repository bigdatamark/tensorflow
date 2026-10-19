# [MODULE DOC]
# Test for the MapAndBatchFusion tf.data optimization: a map followed by a batch should be fused
# into a single MapAndBatch kernel. Exposes one test class and a standard test main entry point.
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
"""Tests for the `MapAndBatchFusion` optimization."""
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class MapAndBatchFusionTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the map-and-batch fusion graph optimization of tf.data.

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatchFusion(self):
    # [FUNCTION DOC]
    # Scenario: range(10) mapped by squaring and batched with size 10, with default optimizations
    # off and only map_and_batch_fusion on. An assert_next check requires the next op to be a
    # MapAndBatch kernel; output must be a single batch of the squares of 0..9.
    dataset = dataset_ops.Dataset.range(10).apply(
        testing.assert_next(
            ["MapAndBatch"])).map(lambda x: x * x).batch(10)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_and_batch_fusion = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset, expected_output=[[x * x for x in range(10)]])


if __name__ == "__main__":
  test.main()
