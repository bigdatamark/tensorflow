# [MODULE DOC]
# Long-running tests for the multinomial sampling op. Checks that events with very small
# probabilities are not over-sampled when drawing hundreds of millions of samples. Running the file
# as a script starts the test runner.
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
"""Long tests for Multinomial."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class MultinomialTest(test.TestCase):
  # [CLASS DOC]
  # Test case with large-dynamic-range multinomial sampling checks. Each test draws one million
  # samples a hundred times with a fixed seed and accumulates per-index counts.
  # check that events with tiny probabilities are not over-sampled

  def testLargeDynamicRange(self):
    # [FUNCTION DOC]
    # Test: samples from logits of minus thirty and zero with fixed seeds, one hundred times one
    # million draws, and asserts that all one hundred million draws chose index one (the tiny
    # probability event never occurred).
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    with self.test_session():
      samples = random_ops.multinomial(
          constant_op.constant([[-30, 0]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=15)
      for _ in range(100):
        x = self.evaluate(samples)
        indices, counts = np.unique(x, return_counts=True)  # pylint: disable=unexpected-keyword-arg
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertEqual(counts_by_indices[1], 100000000)

  def testLargeDynamicRange2(self):
    # [FUNCTION DOC]
    # Test: same as the first large dynamic range test with the logits swapped (zero and minus
    # thirty). Asserts all one hundred million draws chose index zero.
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    with self.test_session():
      samples = random_ops.multinomial(
          constant_op.constant([[0, -30]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=15)
      for _ in range(100):
        x = self.evaluate(samples)
        indices, counts = np.unique(x, return_counts=True)  # pylint: disable=unexpected-keyword-arg
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertEqual(counts_by_indices[0], 100000000)

  @test_util.run_deprecated_v1
  def testLargeDynamicRange3(self):
    # [FUNCTION DOC]
    # Test (graph mode only): samples from logits of zero and minus seventeen, one hundred times one
    # million draws with another seed. Asserts index one was drawn at least once, i.e. the smaller
    # probability is not undersampled to zero. Needs large memory according to a comment in the code.
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    # here the cpu undersamples and won't pass this test either
    with self.test_session():
      samples = random_ops.multinomial(
          constant_op.constant([[0, -17]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=22)

      # we'll run out of memory if we try to draw 1e9 samples directly
      # really should fit in 12GB of memory...
      for _ in range(100):
        x = self.evaluate(samples)
        indices, counts = np.unique(x, return_counts=True)  # pylint: disable=unexpected-keyword-arg
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertGreater(counts_by_indices[1], 0)

if __name__ == "__main__":
  test.main()
