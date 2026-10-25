# [MODULE DOC]
# Unit tests for ragged math operations: string reduce-join on ragged input, ragged softmax
# (ordinary, large and short inputs compared with a numpy reference), and ragged cumulative sum
# (axis, exclusive and reverse variants, compared with a dense-based slow baseline). Run through
# the standard test runner entry point at the bottom.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests functionality of math operations on ragged tensors."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


class RaggedReduceTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering reductions of ragged tensors (currently only string reduce-join).

  @parameterized.parameters(
      [dict(original=['a b'.split(), 'c d e'.split()], expected='a b c d e')])
  @test_util.run_in_graph_and_eager_modes
  def testStringReduceJoin(self, original, expected, separator=' ', axis=None):
    # [FUNCTION DOC]
    # Test: joins the strings of a ragged tensor of two rows (a b / c d e) with a space separator
    # over all axes (default) and asserts the result equals the single string "a b c d e". Runs in
    # graph and eager modes. Parameters: original nested list, expected value, separator, axis.
    original_rt = ragged_factory_ops.constant(original)
    expected_rt = ragged_factory_ops.constant(expected)
    actual = ragged_string_ops.reduce_join(original_rt, axis=axis,
                                           separator=separator)
    self.assertAllEqual(actual, expected_rt)


class RaggedSoftmaxTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering softmax on ragged tensors via the dense softmax API, compared with a numpy
  # reference implemented as a helper method.

  def _softmax(self, x):
    # [FUNCTION DOC]
    # HELPER: numpy reference softmax for the softmax tests.
    # Takes a 2D numpy array, returns it unchanged if it has zero columns, otherwise returns the
    # row-wise softmax (max subtracted for stability). Asserts the input is 2D.
    assert len(x.shape) == 2
    if x.shape[1] == 0:
      return x
    m = x.max(1)[:, np.newaxis]
    u = np.exp(x - m)
    z = u.sum(1)[:, np.newaxis]
    return u / z

  @test_util.run_in_graph_and_eager_modes
  def testOrdinaryValues(self):
    # [FUNCTION DOC]
    # Test: softmax of a ragged tensor of log-probabilities (rows of 3 and 2 entries) should give
    # probabilities 0.5, 0.25, 0.25 and 0.5, 0.5. Checks the numpy reference against the literal
    # expectation and then the ragged softmax against the reference within a small tolerance.
    eps = 1e-5
    x_list = [np.log([0.5, 0.25, 0.25]), np.log([0.5, 0.5])]
    x_row_matrices = [[row] for row in x_list]
    y_row_matrices = [
        self._softmax(np.array(row_matrix)).tolist()
        for row_matrix in x_row_matrices
    ]
    y_list = [row_matrix[0] for row_matrix in y_row_matrices]
    y_expected_from_numpy = ragged_factory_ops.constant(
        y_list, dtype=dtypes.float32)
    y_expected = ragged_factory_ops.constant([[0.5, 0.25, 0.25], [0.5, 0.5]],
                                             dtype=dtypes.float32)
    self.assertAllClose(y_expected_from_numpy, y_expected, eps)
    x_tf = ragged_factory_ops.constant(x_list, dtype=dtypes.float32)
    y_tf = nn_ops.softmax_v2(x_tf)
    self.assertAllClose(y_tf, y_expected_from_numpy, eps)

  @test_util.run_in_graph_and_eager_modes
  def testLargeValues(self):
    # [FUNCTION DOC]
    # Test: ragged softmax on very large magnitude inputs (negative hundreds and equal 1729s) must
    # be numerically stable and match the numpy reference within a small tolerance.
    eps = 1e-5
    x_list = [[-500, -501, -502], [1729, 1729]]
    x_row_matrices = [[row] for row in x_list]
    y_row_matrices = [
        self._softmax(np.array(row_matrix)).tolist()
        for row_matrix in x_row_matrices
    ]
    y_list = [row_matrix[0] for row_matrix in y_row_matrices]
    y_expected_from_numpy = ragged_factory_ops.constant(
        y_list, dtype=dtypes.float32)
    x_tf = ragged_factory_ops.constant(x_list, dtype=dtypes.float32)
    y_tf = nn_ops.softmax_v2(x_tf)
    self.assertAllClose(y_tf, y_expected_from_numpy, eps)

  @test_util.run_in_graph_and_eager_modes
  def testShortTensors(self):
    # [FUNCTION DOC]
    # Test: ragged softmax on rows that are empty or have a single element must match the numpy
    # reference (empty stays empty, single element becomes one) within a small tolerance.
    eps = 1e-5
    x_list = [[], [1]]
    x_row_matrices = [[row] for row in x_list]
    y_row_matrices = [
        self._softmax(np.array(row_matrix)).tolist()
        for row_matrix in x_row_matrices
    ]
    y_list = [row_matrix[0] for row_matrix in y_row_matrices]
    y_expected_from_numpy = ragged_factory_ops.constant(
        y_list, dtype=dtypes.float32)
    x_tf = ragged_factory_ops.constant(x_list, dtype=dtypes.float32)
    y_tf = nn_ops.softmax_v2(x_tf)
    self.assertAllClose(y_tf, y_expected_from_numpy, eps)


def _cumsum_slow(rt, axis=0, exclusive=False, reverse=False, name=None):
  # [FUNCTION DOC]
  # HELPER: reference cumulative sum for the cumsum tests.
  # Densifies the ragged tensor, applies the dense cumsum with the given axis, exclusive, reverse
  # and name, and converts back to ragged using the original nested row lengths. Returns the ragged
  # result. Note padding positions are summed too but are dropped by the conversion.
  dense = rt.to_tensor()
  result = math_ops.cumsum(dense, axis=axis, exclusive=exclusive,
                           reverse=reverse, name=name)
  return ragged_tensor.RaggedTensor.from_tensor(
      result, lengths=rt.nested_row_lengths())


class RaggedCumsumTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the ragged cumulative sum on a two-row ragged tensor and on a deeper
  # ragged tensor, over several axes and exclusive/reverse settings.

  @parameterized.parameters([
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[1, 2], [4, 6, 5]]),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[4, 6], [3, 4, 5]],
           reverse=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[3, 4], [0, 0, 0]],
           reverse=True,
           exclusive=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[0, 0], [1, 2, 0]],
           exclusive=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[0, 0], [1, 2, 0]],
           exclusive=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[1, 3], [3, 7, 12]],
           axis=1),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[3, 2], [12, 9, 5]],
           axis=1, reverse=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[2, 0], [9, 5, 0]],
           axis=1, exclusive=True, reverse=True),
      dict(original=[[1, 2], [3, 4, 5]],
           expected=[[0, 1], [0, 3, 7]],
           axis=1, exclusive=True),
  ])
  def test_cumsum(self, original, expected, axis=0, exclusive=False,
                  reverse=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the parameter list contains the exclusive-only case twice, so one scenario is
    # tested redundantly.
    # Test: for each parameter set, builds a ragged tensor from the original nested list, runs the
    # ragged cumulative sum with the given axis, exclusive and reverse flags and asserts it equals
    # the expected ragged tensor; also asserts the dense-based slow baseline matches the expectation.
    original_rt = ragged_factory_ops.constant(original)
    expected_rt = ragged_factory_ops.constant(expected)
    actual = ragged_math_ops.ragged_cumsum(
        original_rt, axis=axis, exclusive=exclusive, reverse=reverse)
    self.assertAllEqual(actual, expected_rt)
    baseline = _cumsum_slow(original_rt, axis=axis, exclusive=exclusive,
                            reverse=reverse)
    self.assertAllEqual(baseline, expected_rt)

  @parameterized.parameters([
      dict(expected=[[[0, 1], [2, 5]], [[4, 9], [6, 13], [8, 17]]], axis=2),
      dict(expected=[[[0, 0], [0, 2]], [[0, 4], [0, 6], [0, 8]]],
           exclusive=True, axis=2),
      dict(expected=[[[1, 0], [3, 0]], [[5, 0], [7, 0], [9, 0]]],
           exclusive=True, reverse=True, axis=2),
      dict(expected=[[[1, 1], [5, 3]], [[9, 5], [13, 7], [17, 9]]],
           reverse=True, axis=2),
  ])
  def test_cumsum_deep(self, expected, axis=0, exclusive=False, reverse=False):
    # [FUNCTION DOC]
    # Test: builds a ragged tensor with a dense inner dimension of width 2 (row lengths 2 and 3)
    # from the numbers 0 to 9, runs the ragged cumulative sum along the innermost axis with
    # exclusive and reverse variants, and asserts both the ragged result and the slow baseline equal
    # the expected nested values.
    # [[[0, 1], [2, 3]], [[4, 5], [6, 7], [8, 9]]]

    original_rt = ragged_tensor.RaggedTensor.from_row_lengths(
        array_ops.reshape(math_ops.range(10), (5, 2)), [2, 3])

    actual = ragged_math_ops.ragged_cumsum(
        original_rt, axis=axis, exclusive=exclusive, reverse=reverse)
    self.assertAllEqual(actual, expected)
    baseline = _cumsum_slow(original_rt, axis=axis, exclusive=exclusive,
                            reverse=reverse)
    self.assertAllEqual(baseline, expected)

if __name__ == '__main__':
  googletest.main()
