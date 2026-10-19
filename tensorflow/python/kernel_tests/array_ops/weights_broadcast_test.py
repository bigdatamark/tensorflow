# [MODULE DOC]
# Tests for the weights broadcasting utilities (assert-broadcastable and broadcast-weights) from the
# weights broadcast ops module. Two test cases cover the static (known shape) and dynamic
# (placeholder fed at run time) paths for valid weight shapes (scalar, singleton or full
# dimensions) and invalid ones (wrong rank, prefix/suffix-only matches, extra dimensions).
# Includes a small module-level helper that builds sequential test values of a given shape.
# No import-time side effects; running as a script starts the TensorFlow test runner.
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
"""Tests for broadcast rules."""

import numpy as np

from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import test


def _test_values(shape):
  # [FUNCTION DOC]
  # HELPER: builds the values tensor used by the broadcast tests.
  # Takes a shape (sequence of dimension sizes) and returns a numpy integer array of that shape
  # filled with 1, 2, 3, ... in row-major order. No side effects.
  return np.reshape(np.cumsum(np.ones(shape), dtype=np.int32), shape)


class AssertBroadcastableTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the assert-broadcastable op, which verifies that a weights tensor can be broadcast
  # to a values tensor. Each scenario is checked both with statically known inputs (error raised
  # at graph construction) and with placeholders (error raised when the op runs). Graph state is
  # reset before every test.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture hook run before each test: resets the default graph so tests do not share
    # graph state. Takes no inputs and returns nothing.
    ops.reset_default_graph()

  def _test_valid(self, weights, values):
    # [FUNCTION DOC]
    # HELPER: shared check that a weights/values pair is broadcastable.
    # Builds the assert-broadcastable op once from the concrete weights and values and once from
    # float placeholders, then in a cached session runs the static op and runs the dynamic op
    # feeding the real weights and values. Passes if neither raises. Returns nothing.
    static_op = weights_broadcast_ops.assert_broadcastable(
        weights=weights, values=values)
    weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
    values_placeholder = array_ops.placeholder(dtypes_lib.float32)
    dynamic_op = weights_broadcast_ops.assert_broadcastable(
        weights=weights_placeholder, values=values_placeholder)
    with self.cached_session():
      static_op.run()
      dynamic_op.run(feed_dict={
          weights_placeholder: weights,
          values_placeholder: values,
      })

  @test_util.run_deprecated_v1
  def testScalar(self):
    # [FUNCTION DOC]
    # Scenario: a scalar weight against a 3x2x4 values array is broadcastable. Returns nothing.
    self._test_valid(weights=5, values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1x1x1(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 1x1x1 against 3x2x4 values are broadcastable. Returns nothing.
    self._test_valid(
        weights=np.asarray((5,)).reshape((1, 1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1x1xN(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 1x1x4 against 3x2x4 values are broadcastable. Returns nothing.
    self._test_valid(
        weights=np.asarray((5, 7, 11, 3)).reshape((1, 1, 4)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1xNx1(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 1x2x1 against 3x2x4 values are broadcastable. Returns nothing.
    self._test_valid(
        weights=np.asarray((5, 11)).reshape((1, 2, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1xNxN(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 1x2x4 against 3x2x4 values are broadcastable. Returns nothing.
    self._test_valid(
        weights=np.asarray((5, 7, 11, 3, 2, 13, 7, 5)).reshape((1, 2, 4)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testNx1x1(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 3x1x1 against 3x2x4 values are broadcastable. Returns nothing.
    self._test_valid(
        weights=np.asarray((5, 7, 11)).reshape((3, 1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testNx1xN(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 3x1x4 against 3x2x4 values are broadcastable. Returns nothing.
    self._test_valid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3)).reshape((3, 1, 4)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testNxNxN(self):
    # [FUNCTION DOC]
    # Scenario: weights with exactly the values' shape 3x2x4 are broadcastable. Returns nothing.
    self._test_valid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4)),
        values=_test_values((3, 2, 4)))

  def _test_invalid(self, weights, values):
    # [FUNCTION DOC]
    # HELPER: shared check that a weights/values pair is NOT broadcastable.
    # With concrete inputs, asserts the assert-broadcastable call raises a value error whose
    # message says the weights cannot be broadcast to the values. Then builds the op with
    # placeholders and asserts that running it with the fed data raises an op error with the same
    # message. Returns nothing.
    error_msg = 'weights can not be broadcast to values'
    with self.assertRaisesRegex(ValueError, error_msg):
      weights_broadcast_ops.assert_broadcastable(weights=weights, values=values)
    weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
    values_placeholder = array_ops.placeholder(dtypes_lib.float32)
    dynamic_op = weights_broadcast_ops.assert_broadcastable(
        weights=weights_placeholder, values=values_placeholder)
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.OpError, error_msg):
        dynamic_op.run(feed_dict={
            weights_placeholder: weights,
            values_placeholder: values,
        })

  @test_util.run_deprecated_v1
  def testInvalid1(self):
    # [FUNCTION DOC]
    # Scenario: a one-element vector of weights is rejected against 3x2x4 values (rank mismatch
    # with no trailing match). Returns nothing.
    self._test_invalid(weights=np.asarray((5,)), values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalid1x1(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 1x1 (fewer dimensions than values) are rejected. Returns nothing.
    self._test_invalid(
        weights=np.asarray((5,)).reshape((1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidPrefixMatch(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 3x2 match only the leading dimensions of the 3x2x4 values and are
    # rejected. Returns nothing.
    self._test_invalid(
        weights=np.asarray((5, 7, 11, 3, 2, 12)).reshape((3, 2)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidSuffixMatch(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 2x4 match only the trailing dimensions of the 3x2x4 values and
    # are rejected. Returns nothing.
    self._test_invalid(
        weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5)).reshape((2, 4)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidOnesExtraDim(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 1x1x1x1 have more dimensions than the values and are rejected.
    # Returns nothing.
    self._test_invalid(
        weights=np.asarray((5,)).reshape((1, 1, 1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidPrefixMatchExtraDim(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 3x2x4x1 (values' shape plus a trailing extra dimension) are
    # rejected. Returns nothing.
    self._test_invalid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidSuffixMatchExtraDim(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 1x3x2x4 (values' shape plus a leading extra dimension) are
    # rejected. Returns nothing.
    self._test_invalid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((1, 3, 2, 4)),
        values=_test_values((3, 2, 4)))


class BroadcastWeightsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the broadcast-weights op, which expands a weights tensor to the shape of a values
  # tensor. Valid cases compare the result with the expected numpy-tiled array in static and
  # placeholder modes; invalid cases expect errors. Graph state is reset before every test.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture hook run before each test: resets the default graph so tests do not share
    # graph state. Takes no inputs and returns nothing.
    ops.reset_default_graph()

  def _test_valid(self, weights, values, expected):
    # [FUNCTION DOC]
    # HELPER: shared check that broadcasting weights to values gives the expected array.
    # Builds the broadcast-weights op from the concrete inputs and again from float placeholders;
    # in a cached session asserts the static result equals the expected array and that the
    # placeholder version, fed the real weights and values, also equals it. Returns nothing.
    static_op = weights_broadcast_ops.broadcast_weights(
        weights=weights, values=values)
    weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
    values_placeholder = array_ops.placeholder(dtypes_lib.float32)
    dynamic_op = weights_broadcast_ops.broadcast_weights(
        weights=weights_placeholder, values=values_placeholder)
    with self.cached_session():
      self.assertAllEqual(expected, self.evaluate(static_op))
      self.assertAllEqual(expected, dynamic_op.eval(feed_dict={
          weights_placeholder: weights,
          values_placeholder: values,
      }))

  @test_util.run_deprecated_v1
  def testScalar(self):
    # [FUNCTION DOC]
    # Scenario: a scalar weight of five is broadcast to 3x2x4 values; expects an all-fives array
    # of that shape. Returns nothing.
    self._test_valid(
        weights=5,
        values=_test_values((3, 2, 4)),
        expected=5 * np.ones((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1x1x1(self):
    # [FUNCTION DOC]
    # Scenario: 1x1x1 weights of five are broadcast to 3x2x4; expects an all-fives array.
    # Returns nothing.
    self._test_valid(
        weights=np.asarray((5,)).reshape((1, 1, 1)),
        values=_test_values((3, 2, 4)),
        expected=5 * np.ones((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1x1xN(self):
    # [FUNCTION DOC]
    # Scenario: 1x1x4 weights are broadcast to 3x2x4; expects the weights tiled three by two
    # times. Returns nothing.
    weights = np.asarray((5, 7, 11, 3)).reshape((1, 1, 4))
    self._test_valid(
        weights=weights,
        values=_test_values((3, 2, 4)),
        expected=np.tile(weights, reps=(3, 2, 1)))

  @test_util.run_deprecated_v1
  def test1xNx1(self):
    # [FUNCTION DOC]
    # Scenario: 1x2x1 weights are broadcast to 3x2x4; expects the weights tiled three and four
    # times along the first and last axes. Returns nothing.
    weights = np.asarray((5, 11)).reshape((1, 2, 1))
    self._test_valid(
        weights=weights,
        values=_test_values((3, 2, 4)),
        expected=np.tile(weights, reps=(3, 1, 4)))

  @test_util.run_deprecated_v1
  def test1xNxN(self):
    # [FUNCTION DOC]
    # Scenario: 1x2x4 weights are broadcast to 3x2x4; expects the weights tiled three times along
    # the first axis. Returns nothing.
    weights = np.asarray((5, 7, 11, 3, 2, 13, 7, 5)).reshape((1, 2, 4))
    self._test_valid(
        weights=weights,
        values=_test_values((3, 2, 4)),
        expected=np.tile(weights, reps=(3, 1, 1)))

  @test_util.run_deprecated_v1
  def testNx1x1(self):
    # [FUNCTION DOC]
    # Scenario: 3x1x1 weights are broadcast to 3x2x4; expects tiling two and four times along the
    # last two axes. Returns nothing.
    weights = np.asarray((5, 7, 11)).reshape((3, 1, 1))
    self._test_valid(
        weights=weights,
        values=_test_values((3, 2, 4)),
        expected=np.tile(weights, reps=(1, 2, 4)))

  @test_util.run_deprecated_v1
  def testNx1xN(self):
    # [FUNCTION DOC]
    # Scenario: 3x1x4 weights are broadcast to 3x2x4; expects tiling two times along the middle
    # axis. Returns nothing.
    weights = np.asarray((
        5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3)).reshape((3, 1, 4))
    self._test_valid(
        weights=weights,
        values=_test_values((3, 2, 4)),
        expected=np.tile(weights, reps=(1, 2, 1)))

  @test_util.run_deprecated_v1
  def testNxNxN(self):
    # [FUNCTION DOC]
    # Scenario: weights already of shape 3x2x4; expects the result to equal the weights
    # unchanged. Returns nothing.
    weights = np.asarray((
        5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
        2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4))
    self._test_valid(
        weights=weights, values=_test_values((3, 2, 4)), expected=weights)

  def _test_invalid(self, weights, values):
    # [FUNCTION DOC]
    # HELPER: shared check that weights which cannot be broadcast are rejected.
    # With concrete inputs asserts the broadcast-weights call raises a value error saying the
    # weights cannot be broadcast to the values. With placeholders, asserts evaluating the op
    # with the fed data raises an op error with the same message. Returns nothing.
    error_msg = 'weights can not be broadcast to values'
    with self.assertRaisesRegex(ValueError, error_msg):
      weights_broadcast_ops.broadcast_weights(weights=weights, values=values)
    weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
    values_placeholder = array_ops.placeholder(dtypes_lib.float32)
    dynamic_op = weights_broadcast_ops.broadcast_weights(
        weights=weights_placeholder, values=values_placeholder)
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.OpError, error_msg):
        dynamic_op.eval(feed_dict={
            weights_placeholder: weights,
            values_placeholder: values,
        })

  @test_util.run_deprecated_v1
  def testInvalid1(self):
    # [FUNCTION DOC]
    # Scenario: a one-element vector of weights is rejected against 3x2x4 values. Returns nothing.
    self._test_invalid(weights=np.asarray((5,)), values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalid1x1(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 1x1 are rejected against 3x2x4 values. Returns nothing.
    self._test_invalid(
        weights=np.asarray((5,)).reshape((1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidPrefixMatch(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 3x2 (leading-dimension match only) are rejected. Returns nothing.
    self._test_invalid(
        weights=np.asarray((5, 7, 11, 3, 2, 12)).reshape((3, 2)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidSuffixMatch(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 2x4 (trailing-dimension match only) are rejected. Returns
    # nothing.
    self._test_invalid(
        weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5)).reshape((2, 4)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidOnesExtraDim(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 1x1x1x1 (too many dimensions) are rejected. Returns nothing.
    self._test_invalid(
        weights=np.asarray((5,)).reshape((1, 1, 1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidPrefixMatchExtraDim(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 3x2x4x1 (extra trailing dimension) are rejected. Returns nothing.
    self._test_invalid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidSuffixMatchExtraDim(self):
    # [FUNCTION DOC]
    # Scenario: weights of shape 1x3x2x4 (extra leading dimension) are rejected. Returns nothing.
    self._test_invalid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((1, 3, 2, 4)),
        values=_test_values((3, 2, 4)))


if __name__ == '__main__':
  test.main()
