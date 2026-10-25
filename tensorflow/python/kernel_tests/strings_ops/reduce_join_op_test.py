# [MODULE DOC]
# Tests for the reduce_join string operation, which concatenates strings along one or more axes
# with an optional separator. Defines two array-building helpers that construct a rank-n input
# array of binary-index strings and the expected joined result, a unicode-aware test base class,
# a test case that validates those helpers, and the main test case covering ranks one, two and
# five, negative and multiple axes, separators, keeping dimensions, unknown shapes and indices,
# zero-size dimensions, error handling and deprecated argument names. Run via the standard test
# main entry point. No import-time side effects.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ReduceJoin op from string_ops."""

import itertools

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


def _input_array(num_dims):
  # [FUNCTION DOC]
  # HELPER: builds test input for the reduce_join tests.
  # Takes the number of dimensions; returns a byte-string array of shape two along every axis
  # whose element at each position is the binary representation of its flattened index,
  # zero-padded to num_dims digits. No side effects.
  """Creates an ndarray where each element is the binary of its linear index.

  Args:
    num_dims: The number of dimensions to create.

  Returns:
    An ndarray of shape [2] * num_dims.
  """
  formatter = "{:0%db}" % num_dims
  strings = [formatter.format(i) for i in range(2**num_dims)]
  return np.array(strings, dtype="S%d" % num_dims).reshape([2] * num_dims)


def _joined_array(num_dims, reduce_dim):
  # [FUNCTION DOC]
  # HELPER: builds the expected output for the reduce_join tests.
  # Takes the number of dimensions of the original array and the axis to reduce; returns a
  # byte-string array with that axis removed (shape two along each remaining axis). Each element
  # is the concatenation of the two input strings that differ only by a zero or a one inserted
  # at the reduced axis position of the remaining binary index. No side effects.
  """Creates an ndarray with the result from reduce_join on input_array.

  Args:
    num_dims: The number of dimensions of the original input array.
    reduce_dim: The dimension to reduce.

  Returns:
    An ndarray of shape [2] * (num_dims - 1).
  """
  formatter = "{:0%db}" % (num_dims - 1)
  result = np.zeros(shape=[2] * (num_dims - 1), dtype="S%d" % (2 * num_dims))
  flat = result.ravel()
  for i in range(2**(num_dims - 1)):
    dims = formatter.format(i)
    flat[i] = "".join([
        (dims[:reduce_dim] + "%d" + dims[reduce_dim:]) % j for j in range(2)
    ])
  return result


class UnicodeTestCase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base test case for the reduce_join tests. Adds an assertion that compares
  # byte-string and text-string arrays by converting both sides to unicode text first.
  """Test case with Python3-compatible string comparator."""

  def assertAllEqualUnicode(self, truth, actual):
    # [FUNCTION DOC]
    # HELPER: comparison utility for the reduce_join tests.
    # Converts the expected and actual values to unicode text arrays and asserts they are equal
    # element by element and in shape. Fails the test on any difference. Returns nothing.
    self.assertAllEqual(
        np.array(truth).astype("U"), np.array(actual).astype("U"))


class ReduceJoinTestHelperTest(UnicodeTestCase):
  # [CLASS DOC]
  # Test case that checks the module's own array-building helpers against hand-written
  # expectations for three dimensions, so the main tests can trust them.
  """Tests for helper functions."""

  def testInputArray(self):
    # [FUNCTION DOC]
    # Builds the three-dimensional input array, flattens it, and asserts it equals the eight
    # three-digit binary strings in order. Returns nothing.
    num_dims = 3
    truth = ["{:03b}".format(i) for i in range(2**num_dims)]
    output_array = _input_array(num_dims).reshape([-1])
    self.assertAllEqualUnicode(truth, output_array)

  def testJoinedArray(self):
    # [FUNCTION DOC]
    # Builds the expected joined array for three dimensions reducing axis zero, one and two in
    # turn, and asserts each equals the corresponding hand-written two-by-two array of six-digit
    # strings. Returns nothing.
    num_dims = 3
    truth_dim_zero = [["000100", "001101"], ["010110", "011111"]]
    truth_dim_one = [["000010", "001011"], ["100110", "101111"]]
    truth_dim_two = [["000001", "010011"], ["100101", "110111"]]
    output_array_dim_zero = _joined_array(num_dims, reduce_dim=0)
    output_array_dim_one = _joined_array(num_dims, reduce_dim=1)
    output_array_dim_two = _joined_array(num_dims, reduce_dim=2)
    self.assertAllEqualUnicode(truth_dim_zero, output_array_dim_zero)
    self.assertAllEqualUnicode(truth_dim_one, output_array_dim_one)
    self.assertAllEqualUnicode(truth_dim_two, output_array_dim_two)


class ReduceJoinTest(UnicodeTestCase):
  # [CLASS DOC]
  # Main test case for the reduce_join operation. Provides two private comparison helpers (a
  # single-axis check against a given expected value and shape, and a multi-axis check against
  # nested single-axis reductions) used by the individual scenario tests.

  def _testReduceJoin(self,
                      input_array,
                      truth,
                      truth_shape,
                      axis,
                      keep_dims=False,
                      separator=""):
    # [FUNCTION DOC]
    # HELPER: shared single-reduction assertion for the reduce_join tests.
    # Inputs: the string input, the expected value, the expected static shape, the axis or axes
    # to reduce (may be nothing for all axes), whether to keep reduced dimensions (default no),
    # and the separator (default empty). Runs reduce_join in a cached session, evaluates it, and
    # asserts the value equals the expectation (unicode-aware) and the static output shape equals
    # the expected shape. Returns nothing.
    """Compares the output of reduce_join to an expected result.

    Args:
      input_array: The string input to be joined.
      truth: An array or np.array of the expected result.
      truth_shape: An array or np.array of the expected shape.
      axis: The indices to reduce over.
      keep_dims: Whether or not to retain reduced dimensions.
      separator: The separator to use for joining.
    """
    with self.cached_session():
      output = string_ops.reduce_join(
          inputs=input_array,
          axis=axis,
          keep_dims=keep_dims,
          separator=separator)
      output_array = self.evaluate(output)

    self.assertAllEqualUnicode(truth, output_array)
    self.assertAllEqual(truth_shape, output.get_shape())

  def _testMultipleReduceJoin(self, input_array, axis, separator=" "):
    # [FUNCTION DOC]
    # HELPER: shared multi-axis assertion for the reduce_join tests.
    # Inputs: the input array, a sequence of axes to reduce (possibly empty), and the separator
    # (default a single space). Computes reduce_join over all axes at once with and without
    # keeping dimensions, builds the expected value by reducing one axis at a time while keeping
    # dimensions (wrapping in a constant for an empty axis list) and squeezing it, then asserts
    # values and static shapes match. Returns nothing.
    """Tests reduce_join for one input and multiple axes.

    Does so by comparing the output to that from nested reduce_string_joins.
    The correctness of single-dimension reduce_join is verified by other
    tests below using _testReduceJoin.

    Args:
      input_array: The input to test.
      axis: The indices to reduce.
      separator: The separator to use when joining.
    """
    with self.cached_session():
      output = string_ops.reduce_join(
          inputs=input_array, axis=axis, keep_dims=False, separator=separator)
      output_keep_dims = string_ops.reduce_join(
          inputs=input_array, axis=axis, keep_dims=True, separator=separator)

      truth = input_array
      for index in axis:
        truth = string_ops.reduce_join(
            inputs=truth, axis=index, keep_dims=True, separator=separator)
      if not axis:
        truth = constant_op.constant(truth)
      truth_squeezed = array_ops.squeeze(truth, axis=axis)
      output_array = self.evaluate(output)
      output_keep_dims_array = self.evaluate(output_keep_dims)
      truth_array = self.evaluate(truth)
      truth_squeezed_array = self.evaluate(truth_squeezed)
    self.assertAllEqualUnicode(truth_array, output_keep_dims_array)
    self.assertAllEqualUnicode(truth_squeezed_array, output_array)
    self.assertAllEqual(truth.get_shape(), output_keep_dims.get_shape())
    self.assertAllEqual(truth_squeezed.get_shape(), output.get_shape())

  def testRankOne(self):
    # [FUNCTION DOC]
    # Joins a list of four words along axis zero and expects the single concatenated string
    # with an empty (scalar) static shape. Returns nothing.
    input_array = ["this", "is", "a", "test"]
    truth = "thisisatest"
    truth_shape = []
    self._testReduceJoin(input_array, truth, truth_shape, axis=0)

  def testRankTwo(self):
    # [FUNCTION DOC]
    # On a two-by-four matrix of words, checks reduction over axis zero (column-wise joins), axis
    # one (row-wise joins), no axis given (everything joined to a scalar), and an empty axis
    # list, which must leave the input unchanged with shape two by four. Returns nothing.
    input_array = [["this", "is", "a", "test"],
                   ["please", "do", "not", "panic"]]
    truth_dim_zero = ["thisplease", "isdo", "anot", "testpanic"]
    truth_shape_dim_zero = [4]
    truth_dim_one = ["thisisatest", "pleasedonotpanic"]
    truth_shape_dim_one = [2]
    self._testReduceJoin(
        input_array, truth_dim_zero, truth_shape_dim_zero, axis=0)
    self._testReduceJoin(
        input_array, truth_dim_one, truth_shape_dim_one, axis=1)

    expected_val = "thisisatestpleasedonotpanic"
    expected_shape = []
    self._testReduceJoin(input_array, expected_val, expected_shape, axis=None)

    # Using axis=[] is a no-op.
    expected_val = input_array
    expected_shape = [2, 4]
    self._testReduceJoin(input_array, expected_val, expected_shape, axis=[])

  def testRankFive(self):
    # [FUNCTION DOC]
    # Builds a five-dimensional binary-index string array and checks reduction over each of the
    # five axes against the helper-computed expectation with static shape two on four axes.
    # Returns nothing.
    input_array = _input_array(num_dims=5)
    truths = [_joined_array(num_dims=5, reduce_dim=i) for i in range(5)]
    truth_shape = [2] * 4
    for i in range(5):
      self._testReduceJoin(input_array, truths[i], truth_shape, axis=i)

  def testNegative(self):
    # [FUNCTION DOC]
    # Same as the rank-five test but addresses each axis with its negative index (index minus
    # five), asserting negative axes count from the end and give the same results.
    # Returns nothing.
    input_array = _input_array(num_dims=5)
    truths = [_joined_array(num_dims=5, reduce_dim=i) for i in range(5)]
    truth_shape = [2] * 4
    for i in range(5):
      self._testReduceJoin(input_array, truths[i], truth_shape, axis=i - 5)

  def testSingletonDimension(self):
    # [FUNCTION DOC]
    # Inserts a size-one dimension at each of six positions in a rank-five input and reduces
    # exactly that axis; asserts the result equals the original rank-five array, i.e. joining a
    # single element is the identity. Returns nothing.
    input_arrays = [
        _input_array(num_dims=5).reshape([2] * i + [1] + [2] * (5 - i))
        for i in range(6)
    ]
    truth = _input_array(num_dims=5)
    truth_shape = [2] * 5
    for i in range(6):
      self._testReduceJoin(input_arrays[i], truth, truth_shape, axis=i)

  def testSeparator(self):
    # [FUNCTION DOC]
    # Joins a two-by-four matrix of words along axis zero and one using a two-space separator and
    # asserts the separator appears between joined elements only, with the expected shapes.
    # Returns nothing.
    input_array = [["this", "is", "a", "test"],
                   ["please", "do", "not", "panic"]]
    truth_dim_zero = ["this  please", "is  do", "a  not", "test  panic"]
    truth_shape_dim_zero = [4]
    truth_dim_one = ["this  is  a  test", "please  do  not  panic"]
    truth_shape_dim_one = [2]

    self._testReduceJoin(
        input_array,
        truth_dim_zero,
        truth_shape_dim_zero,
        axis=0,
        separator="  ")
    self._testReduceJoin(
        input_array,
        truth_dim_one,
        truth_shape_dim_one,
        axis=1,
        separator="  ")

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Feeds a two-by-one string matrix through a placeholder of unknown shape, reduces axis zero
    # and asserts the value is a one-element list with the joined text and that the static
    # output shape is unknown. Graph mode only. Returns nothing.
    input_array = [["a"], ["b"]]
    truth = ["ab"]
    truth_shape = None
    with self.cached_session():
      placeholder = array_ops.placeholder(dtypes.string, name="placeholder")
      reduced = string_ops.reduce_join(placeholder, axis=0)
      output_array = reduced.eval(feed_dict={placeholder.name: input_array})
      self.assertAllEqualUnicode(truth, output_array)
      self.assertAllEqual(truth_shape, reduced.get_shape())

  @test_util.run_deprecated_v1
  def testUnknownIndices(self):
    # [FUNCTION DOC]
    # Passes the reduction axis through an integer placeholder, evaluating with axis zero and then
    # axis one, and asserts the column-wise and row-wise joined results while the static output
    # shape stays unknown. Graph mode only. Returns nothing.
    input_array = [["this", "is", "a", "test"],
                   ["please", "do", "not", "panic"]]
    truth_dim_zero = ["thisplease", "isdo", "anot", "testpanic"]
    truth_dim_one = ["thisisatest", "pleasedonotpanic"]
    truth_shape = None
    with self.cached_session():
      placeholder = array_ops.placeholder(dtypes.int32, name="placeholder")
      reduced = string_ops.reduce_join(input_array, axis=placeholder)
      output_array_dim_zero = reduced.eval(feed_dict={placeholder.name: [0]})
      output_array_dim_one = reduced.eval(feed_dict={placeholder.name: [1]})
      self.assertAllEqualUnicode(truth_dim_zero, output_array_dim_zero)
      self.assertAllEqualUnicode(truth_dim_one, output_array_dim_one)
      self.assertAllEqual(truth_shape, reduced.get_shape())

  def testKeepDims(self):
    # [FUNCTION DOC]
    # Reduces a two-by-four matrix with the keep-dimensions flag on over axis zero, axis one, no
    # axis (all, giving a one-by-one result) and an empty axis list (no-op), asserting values and
    # static shapes retain size-one reduced dimensions. Returns nothing.
    input_array = [["this", "is", "a", "test"],
                   ["please", "do", "not", "panic"]]
    truth_dim_zero = [["thisplease", "isdo", "anot", "testpanic"]]
    truth_shape_dim_zero = [1, 4]
    truth_dim_one = [["thisisatest"], ["pleasedonotpanic"]]
    truth_shape_dim_one = [2, 1]

    self._testReduceJoin(
        input_array,
        truth_dim_zero,
        truth_shape_dim_zero,
        axis=0,
        keep_dims=True)
    self._testReduceJoin(
        input_array,
        truth_dim_one,
        truth_shape_dim_one,
        axis=1,
        keep_dims=True)

    expected_val = [["thisisatestpleasedonotpanic"]]
    expected_shape = [1, 1]
    self._testReduceJoin(
        constant_op.constant(input_array), expected_val, expected_shape,
        keep_dims=True, axis=None)

    # Using axis=[] is a no-op.
    expected_val = input_array
    expected_shape = [2, 4]
    self._testReduceJoin(
        input_array, expected_val, expected_shape, keep_dims=True, axis=[])

  def testMultiIndex(self):
    # [FUNCTION DOC]
    # For a three-dimensional input, iterates over every ordered selection of zero to three
    # distinct axes and uses the multi-axis helper to assert that reducing them at once equals
    # reducing one at a time. Returns nothing.
    num_dims = 3
    input_array = _input_array(num_dims=num_dims)
    # Also tests [].
    for i in range(num_dims + 1):
      for permutation in itertools.permutations(range(num_dims), i):
        self._testMultipleReduceJoin(input_array, axis=permutation)

  @test_util.run_deprecated_v1
  def testInvalidReductionIndices(self):
    # [FUNCTION DOC]
    # Asserts that graph construction raises a value error with an invalid-reduction-dimension
    # message for axis zero on a scalar, for axes out of range below (minus three) and above
    # (two) for a rank-two input, and for the same out-of-range axes inside a list.
    # Graph mode only. Returns nothing.
    with self.cached_session():
      with self.assertRaisesRegex(ValueError, "Invalid reduction dim"):
        string_ops.reduce_join(inputs="", axis=0)
      with self.assertRaisesRegex(ValueError, "Invalid reduction dimension -3"):
        string_ops.reduce_join(inputs=[[""]], axis=-3)
      with self.assertRaisesRegex(ValueError, "Invalid reduction dimension 2"):
        string_ops.reduce_join(inputs=[[""]], axis=2)
      with self.assertRaisesRegex(ValueError, "Invalid reduction dimension -3"):
        string_ops.reduce_join(inputs=[[""]], axis=[0, -3])
      with self.assertRaisesRegex(ValueError, "Invalid reduction dimension 2"):
        string_ops.reduce_join(inputs=[[""]], axis=[0, 2])

  def testZeroDims(self):
    # [FUNCTION DOC]
    # Uses an input with shape zero by one. Reducing axis zero (the empty dimension) must give a
    # single empty string; reducing axis one must give an empty vector of shape zero.
    # Returns nothing.
    with self.cached_session():
      inputs = np.zeros([0, 1], dtype=str)

      # Reduction that drops the dim of size 0.
      output = string_ops.reduce_join(inputs=inputs, axis=0)
      self.assertAllEqualUnicode([""], self.evaluate(output))

      # Reduction that keeps the dim of size 0.
      output = string_ops.reduce_join(inputs=inputs, axis=1)
      output_shape = self.evaluate(output).shape
      self.assertAllEqual([0], output_shape)

  @test_util.run_deprecated_v1
  def testInvalidArgsUnknownShape(self):
    # [FUNCTION DOC]
    # With a placeholder input of unknown shape, builds reductions with an out-of-range axis and
    # with a duplicated axis (minus one and one on a matrix). Asserts evaluation raises op errors
    # reporting an invalid reduction dimension and a duplicate reduction dimension respectively.
    # Graph mode only. Returns nothing.
    with self.cached_session():
      placeholder = array_ops.placeholder(dtypes.string, name="placeholder")
      index_too_high = string_ops.reduce_join(placeholder, axis=1)
      duplicate_index = string_ops.reduce_join(placeholder, axis=[-1, 1])
      with self.assertRaisesOpError("Invalid reduction dimension 1"):
        index_too_high.eval(feed_dict={placeholder.name: [""]})
      with self.assertRaisesOpError("Duplicate reduction dimension 1"):
        duplicate_index.eval(feed_dict={placeholder.name: [[""]]})

  @test_util.run_deprecated_v1
  def testInvalidArgsUnknownIndices(self):
    # [FUNCTION DOC]
    # Reduces a two-element string vector with the axis supplied via an integer placeholder, and
    # asserts evaluating with minus two and with two raises op errors mentioning the invalid
    # reduction dimension. Graph mode only. Returns nothing.
    with self.cached_session():
      placeholder = array_ops.placeholder(dtypes.int32, name="placeholder")
      reduced = string_ops.reduce_join(["test", "test2"], axis=placeholder)

      with self.assertRaisesOpError("reduction dimension -2"):
        reduced.eval(feed_dict={placeholder.name: -2})
      with self.assertRaisesOpError("reduction dimension 2"):
        reduced.eval(feed_dict={placeholder.name: 2})

  def testDeprecatedArgs(self):
    # [FUNCTION DOC]
    # Asserts the deprecated argument names (reduction indices and keep dims) and the current
    # names (axis and keepdims) both produce the one-element result containing the joined text
    # of two words. Returns nothing.
    foobar = constant_op.constant(["foobar"])
    # Old names: keep_dims and reduction_indices
    output = string_ops.reduce_join(
        ["foo", "bar"], reduction_indices=0, keep_dims=True)
    self.assertAllEqual(foobar, output)
    # New names keepdims and axis.
    output = string_ops.reduce_join(["foo", "bar"], axis=0, keepdims=True)
    self.assertAllEqual(foobar, output)


if __name__ == "__main__":
  test.main()
