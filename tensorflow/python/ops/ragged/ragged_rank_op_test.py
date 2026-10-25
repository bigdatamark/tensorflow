# [MODULE DOC]
# Unit tests for the ragged rank operation, parameterized over nested lists of several depths
# and ragged ranks. Runs the test main when executed as a script.
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
"""Tests for ragged.rank op."""

from absl.testing import parameterized
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedRankOpTest(test_util.TensorFlowTestCase,
                       parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the rank operation on ragged tensors, run in graph and eager modes with
  # parameterized inputs.

  @parameterized.parameters([
      # Rank 0
      dict(
          test_input=1,
          expected_rank=0,
      ),
      # Rank 1
      dict(
          test_input=[1],
          expected_rank=1,
      ),
      dict(
          test_input=[1, 2, 3, 4],
          expected_rank=1,
      ),
      # Rank 2
      dict(
          test_input=[[1], [2], [3]],
          expected_rank=2,
      ),
      # Rank 3
      dict(
          test_input=[[[1], [2, 3]], [[4], [5, 6, 7]]],
          expected_rank=3,
      ),
      # Rank 3, ragged_rank=2
      dict(
          test_input=[[[1], [2, 3], [10, 20]],
                      [[4], [5, 6, 7]]],
          expected_rank=3,
          ragged_rank=2,
      ),
      # Rank 4, ragged_rank=3 with dimensions: {2, (1, 2), (2), (1, 2)}
      dict(
          test_input=[[[[1], [2]]],
                      [[[3, 4], [5, 6]], [[7, 8], [9, 10]]]],
          expected_rank=4,
      ),
      # Rank 4, ragged_rank=2 with dimensions: {2, (1, 2), (1, 2), 2}
      dict(
          test_input=[
              [[[1, 2]]],
              [[[5, 6], [7, 8]],
               [[9, 10], [11, 12]]]],
          expected_rank=4,
          ragged_rank=2,
      ),

  ])
  def testRaggedRank(self, test_input, expected_rank, ragged_rank=None):
    # [FUNCTION DOC]
    # Parameters: test_input (scalar or nested list), expected_rank (total number of dimensions)
    # and optional ragged_rank used when building the ragged constant. Builds the ragged
    # tensor and asserts its rank operation result equals the expected rank.
    test_input = ragged_factory_ops.constant(
        test_input, ragged_rank=ragged_rank)
    self.assertAllEqual(ragged_array_ops.rank(
        test_input), expected_rank)


if __name__ == '__main__':
  googletest.main()
