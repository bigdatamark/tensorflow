# [MODULE DOC]
# KNOWN ISSUE: the module docstring says it tests the ragged range operation, but the file
# actually tests the ragged segment reductions (sum, product, min, max, mean, square-root-n).
# Unit tests for ragged segment reductions, with plain-Python reference combiners (product,
# mean, square-root-n), a reference grouping routine, parameterized integer and float cases,
# ragged-rank-two data, ragged segment ids and shape-mismatch errors.
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
"""Tests for ragged_range op."""

import math

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


def prod(values):
  # [FUNCTION DOC]
  # HELPER: Reference combiner for the segment tests.
  # Takes a list of numbers and returns their product (one for an empty list).
  val = 1
  for v in values:
    val *= v
  return val
  # return reduce(lambda x, y: x * y, values, 1)


def mean(values):
  # [FUNCTION DOC]
  # HELPER: Reference combiner for the segment tests.
  # Takes a non-empty list of numbers and returns their arithmetic mean as a float.
  return 1.0 * sum(values) / len(values)


def sqrt_n(values):
  # [FUNCTION DOC]
  # HELPER: Reference combiner for the segment tests.
  # Takes a non-empty list of numbers and returns their sum divided by the square root of the
  # count, as a float.
  return 1.0 * sum(values) / math.sqrt(len(values))


@test_util.run_all_in_graph_and_eager_modes
class RaggedSegmentOpsTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Test case for ragged segment reduction operations, run in graph and eager modes. Provides a
  # reference-result helper method used by the parameterized tests.

  def expected_value(self, data, segment_ids, num_segments, combiner):
    # [FUNCTION DOC]
    # HELPER: Computes expected results for the segment tests.
    # Inputs: data (nested list of rows), segment_ids (one segment id per row), num_segments
    # and combiner (function reducing a list of values to one value). Asserts there is one id
    # per row, groups the values by segment and column, and returns a nested list with one row
    # per segment, applying the combiner to every non-empty group.
    """Find the expected value for a call to ragged_segment_<aggregate>.

    Args:
      data: The input RaggedTensor, expressed as a nested python list.
      segment_ids: The segment ids, as a python list of ints.
      num_segments: The number of segments, as a python int.
      combiner: The Python function used to combine values.
    Returns:
      The expected value, as a nested Python list.
    """
    self.assertLen(data, len(segment_ids))

    # Build an empty (num_segments x ncols) "grouped" matrix
    ncols = max(len(row) for row in data)
    grouped = [[[] for _ in range(ncols)] for row in range(num_segments)]

    # Append values from data[row] to grouped[segment_ids[row]]
    for row in range(len(data)):
      for col in range(len(data[row])):
        grouped[segment_ids[row]][col].append(data[row][col])

    # Combine the values.
    return [[combiner(values)
             for values in grouped_row
             if values]
            for grouped_row in grouped]

  @parameterized.parameters(
      (ragged_math_ops.segment_sum, sum, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_sum, sum, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_sum, sum, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_sum, sum, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_prod, prod, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_min, min, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_min, min, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_min, min, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_min, min, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_max, max, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_max, max, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_max, max, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_max, max, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_mean, mean, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 0, 10, 10, 10]),
  )
  def testRaggedSegment_Int(self, segment_op, combiner, segment_ids):
    # [FUNCTION DOC]
    # Parameterized by the segment operation, the reference combiner and the segment ids. Applies
    # the operation to a ragged integer tensor and asserts exact equality with the reference.
    rt_as_list = [[0, 1, 2, 3], [4], [], [5, 6], [7], [8, 9]]
    rt = ragged_factory_ops.constant(rt_as_list)
    num_segments = max(segment_ids) + 1
    expected = self.expected_value(rt_as_list, segment_ids, num_segments,
                                   combiner)

    segmented = segment_op(rt, segment_ids, num_segments)
    self.assertAllEqual(segmented, expected)

  @parameterized.parameters(
      (ragged_math_ops.segment_sum, sum, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_sum, sum, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_sum, sum, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_sum, sum, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_prod, prod, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_prod, prod, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_min, min, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_min, min, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_min, min, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_min, min, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_max, max, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_max, max, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_max, max, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_max, max, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_mean, mean, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_mean, mean, [0, 0, 0, 10, 10, 10]),
      (ragged_math_ops.segment_sqrt_n, sqrt_n, [0, 0, 1, 1, 2, 2]),
      (ragged_math_ops.segment_sqrt_n, sqrt_n, [0, 0, 0, 1, 1, 1]),
      (ragged_math_ops.segment_sqrt_n, sqrt_n, [5, 4, 3, 2, 1, 0]),
      (ragged_math_ops.segment_sqrt_n, sqrt_n, [0, 0, 0, 10, 10, 10]),
  )
  def testRaggedSegment_Float(self, segment_op, combiner, segment_ids):
    # [FUNCTION DOC]
    # Same as the integer test but with float data, also covering the square-root-n operation,
    # and asserting closeness rather than exact equality.
    rt_as_list = [[0., 1., 2., 3.], [4.], [], [5., 6.], [7.], [8., 9.]]
    rt = ragged_factory_ops.constant(rt_as_list)
    num_segments = max(segment_ids) + 1
    expected = self.expected_value(rt_as_list, segment_ids, num_segments,
                                   combiner)

    segmented = segment_op(rt, segment_ids, num_segments)
    self.assertAllClose(segmented, expected)

  def testRaggedRankTwo(self):
    # [FUNCTION DOC]
    # Applies segment sum to a ragged-rank-two tensor with two different segment id lists and
    # asserts the expected nested results, including empty segments.
    rt = ragged_factory_ops.constant([
        [[111, 112, 113, 114], [121],],  # row 0
        [],                              # row 1
        [[], [321, 322], [331]],         # row 2
        [[411, 412]]                     # row 3
    ])  # pyformat: disable
    segment_ids1 = [0, 2, 2, 2]
    segmented1 = ragged_math_ops.segment_sum(rt, segment_ids1, 3)
    expected1 = [[[111, 112, 113, 114], [121]],     # row 0
                 [],                                # row 1
                 [[411, 412], [321, 322], [331]]    # row 2
                ]  # pyformat: disable
    self.assertAllEqual(segmented1, expected1)

    segment_ids2 = [1, 2, 1, 1]
    segmented2 = ragged_math_ops.segment_sum(rt, segment_ids2, 3)
    expected2 = [[],
                 [[111+411, 112+412, 113, 114], [121+321, 322], [331]],
                 []]  # pyformat: disable
    self.assertAllEqual(segmented2, expected2)

  def testRaggedSegmentIds(self):
    # [FUNCTION DOC]
    # Applies segment sum where the segment ids are themselves a ragged tensor whose shape is a
    # prefix of the data's shape, and asserts the expected nested result.
    rt = ragged_factory_ops.constant([
        [[111, 112, 113, 114], [121],],  # row 0
        [],                              # row 1
        [[], [321, 322], [331]],         # row 2
        [[411, 412]]                     # row 3
    ])  # pyformat: disable
    segment_ids = ragged_factory_ops.constant([[1, 2], [], [1, 1, 2], [2]])
    segmented = ragged_math_ops.segment_sum(rt, segment_ids, 3)
    expected = [[],
                [111+321, 112+322, 113, 114],
                [121+331+411, 412]]  # pyformat: disable
    self.assertAllEqual(segmented, expected)

  def testShapeMismatchError1(self):
    # [FUNCTION DOC]
    # Asserts that segment sum with dense data but ragged segment ids raises a value error
    # saying the segment ids shape must be a prefix of the data shape.
    dt = constant_op.constant([1, 2, 3, 4, 5, 6])
    segment_ids = ragged_factory_ops.constant([[1, 2], []])
    self.assertRaisesRegex(
        ValueError, 'segment_ids.shape must be a prefix of data.shape, '
        'but segment_ids is ragged and data is not.',
        ragged_math_ops.segment_sum, dt, segment_ids, 3)

  def testShapeMismatchError2(self):
    # [FUNCTION DOC]
    # Asserts an invalid-argument error when ragged segment ids do not match the ragged data
    # rows: first raised when building the operation with statically known shapes, then, with
    # the segment ids rebuilt from placeholders with unknown shape, raised when evaluating.
    rt = ragged_factory_ops.constant([
        [[111, 112, 113, 114], [121]],  # row 0
        [],                             # row 1
        [[], [321, 322], [331]],        # row 2
        [[411, 412]]                    # row 3
    ])  # pyformat: disable
    segment_ids = ragged_factory_ops.constant([[1, 2], [1], [1, 1, 2], [2]])

    # Error is raised at graph-building time if we can detect it then.
    self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'segment_ids.shape must be a prefix of data.shape.*',
        ragged_math_ops.segment_sum, rt, segment_ids, 3)

    # Otherwise, error is raised when we run the graph.
    segment_ids2 = ragged_tensor.RaggedTensor.from_row_splits(
        array_ops.placeholder_with_default(segment_ids.values, None),
        array_ops.placeholder_with_default(segment_ids.row_splits, None))
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'segment_ids.shape must be a prefix of data.shape.*'):
      self.evaluate(ragged_math_ops.segment_sum(rt, segment_ids2, 3))


if __name__ == '__main__':
  googletest.main()
