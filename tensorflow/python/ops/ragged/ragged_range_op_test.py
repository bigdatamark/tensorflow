# [MODULE DOC]
# Unit tests for the ragged range operation: documentation examples, basic, float, negative
# delta, broadcast and empty ranges, shape-function and kernel error cases, static shape and an
# int32 overflow case. Runs the test main when executed as a script.
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

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedRangeOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the ragged range operation, run in graph and eager modes. No shared fixtures.

  def testDocStringExamples(self):
    # [FUNCTION DOC]
    # Reproduces the three documentation examples of the ragged range operation (limits only,
    # starts and limits, starts limits and a step) and asserts the expected ragged results.
    """Examples from ragged_range.__doc__."""
    rt1 = ragged_math_ops.range([3, 5, 2])
    self.assertAllEqual(rt1, [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1]])

    rt2 = ragged_math_ops.range([0, 5, 8], [3, 3, 12])
    self.assertAllEqual(rt2, [[0, 1, 2], [], [8, 9, 10, 11]])

    rt3 = ragged_math_ops.range([0, 5, 8], [3, 3, 12], 2)
    self.assertAllEqual(rt3, [[0, 2], [], [8, 10]])

  def testBasicRanges(self):
    # [FUNCTION DOC]
    # Checks integer ranges built from limits only, from starts and limits, and from starts,
    # limits and deltas, comparing each row with the equivalent standard range.
    # Specify limits only.
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5]),
        [list(range(0)), list(range(3)),
         list(range(5))])

    # Specify starts and limits.
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], [2, 3, 10]),
        [list(range(0, 2)),
         list(range(3, 3)),
         list(range(5, 10))])

    # Specify starts, limits, and deltas.
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], [4, 4, 15], [2, 3, 4]),
        [list(range(0, 4, 2)),
         list(range(3, 4, 3)),
         list(range(5, 15, 4))])

  def testFloatRanges(self):
    # [FUNCTION DOC]
    # Builds ranges from float starts, limits and deltas and asserts the result is close to the
    # expected rows of floats.
    expected = [[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6], [3.0],
                [5.0, 7.2, 9.4, 11.6, 13.8]]
    actual = ragged_math_ops.range([0.0, 3.0, 5.0], [3.9, 4.0, 15.0],
                                   [0.4, 1.5, 2.2])
    self.assertAllClose(actual, expected)

  def testNegativeDeltas(self):
    # [FUNCTION DOC]
    # Checks ranges with a scalar negative delta and with a per-row mix of negative and positive
    # deltas, comparing each row with the equivalent standard range.
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], limits=0, deltas=-1),
        [list(range(0, 0, -1)),
         list(range(3, 0, -1)),
         list(range(5, 0, -1))])

    self.assertAllEqual(
        ragged_math_ops.range([0, -3, 5], limits=0, deltas=[-1, 1, -2]),
        [list(range(0, 0, -1)),
         list(range(-3, 0, 1)),
         list(range(5, 0, -2))])

  def testBroadcast(self):
    # [FUNCTION DOC]
    # Checks that a scalar delta is broadcast across vector starts and limits, and that all-scalar
    # arguments produce a single-row result.
    # Specify starts and limits, broadcast deltas.
    self.assertAllEqual(
        ragged_math_ops.range([0, 3, 5], [4, 4, 15], 3),
        [list(range(0, 4, 3)),
         list(range(3, 4, 3)),
         list(range(5, 15, 3))])

    # Broadcast all arguments.
    self.assertAllEqual(ragged_math_ops.range(0, 5, 1), [list(range(0, 5, 1))])

  def testEmptyRanges(self):
    # [FUNCTION DOC]
    # Asserts rows whose start is not before the limit (for positive or negative step) come out
    # as empty lists while other rows are filled normally.
    rt1 = ragged_math_ops.range([0, 5, 3], [0, 3, 5])
    rt2 = ragged_math_ops.range([0, 5, 5], [0, 3, 5], -1)
    self.assertAllEqual(rt1, [[], [], [3, 4]])
    self.assertAllEqual(rt2, [[], [5, 4], []])

  def testShapeFnErrors(self):
    # [FUNCTION DOC]
    # Asserts that arguments of invalid rank (nested lists) or mismatched lengths raise either a
    # value error or an invalid-argument error.
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, [[0]], 5)
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, 0, [[5]])
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, 0, 5, [[0]])
    self.assertRaises((ValueError, errors.InvalidArgumentError),
                      ragged_math_ops.range, [0], [1, 2])

  def testKernelErrors(self):
    # [FUNCTION DOC]
    # Evaluates three invalid cases and asserts an invalid-argument error with a matching
    # message: zero delta, a range that would be too large, and (calling the generated raw op
    # with 32-bit row splits) a size overflow.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'Requires delta != 0'):
      self.evaluate(ragged_math_ops.range(0, 0, 0))

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'Requires \(\(limit - start\) / delta\) <='):
      self.evaluate(ragged_math_ops.range(0.1, 1e10, 1e-10))

    with self.assertRaisesRegex(errors.InvalidArgumentError, 'overflowed'):
      self.evaluate(
          gen_ragged_math_ops.ragged_range(
              starts=[0, 0],
              limits=[2**31 - 1, 1],
              deltas=[1, 1],
              Tsplits=dtypes.int32))

  def testShape(self):
    # [FUNCTION DOC]
    # Asserts the static shape of the result: a known row count (one for scalar arguments, the
    # vector length otherwise) and an unknown ragged dimension.
    self.assertAllEqual(
        ragged_math_ops.range(0, 0, 1).shape.as_list(), [1, None])
    self.assertAllEqual(
        ragged_math_ops.range([1, 2, 3]).shape.as_list(), [3, None])
    self.assertAllEqual(
        ragged_math_ops.range([1, 2, 3], [4, 5, 6]).shape.as_list(), [3, None])

  def testInt32Overflow(self):
    # [FUNCTION DOC]
    # Uses large start, end and negative step values that would overflow 32-bit arithmetic and
    # asserts the evaluated result equals the corresponding NumPy arange (one row).
    start = 1136033460
    end = -2110457150
    step = -1849827689
    expected = [np.arange(start, end, step)]
    actual = ragged_math_ops.range(start, end, step)
    self.assertAllEqual(expected, self.evaluate(actual))


if __name__ == '__main__':
  googletest.main()
