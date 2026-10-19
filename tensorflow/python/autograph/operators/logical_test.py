# [MODULE DOC]
# Unit tests for the logical operators module (and_, or_, not_): Python and tensor variants,
# checking values and lazy evaluation of the right operand. No import-time side effects.
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
"""Tests for logical module."""

from tensorflow.python.autograph.operators import logical
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class LogicalOperatorsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the logical operators. Provides helper methods: a function that fails the
  # test when called (to prove laziness) and functions returning constant true/false tensors.

  def assertNotCalled(self):
    # [FUNCTION DOC]
    # HELPER: callable used as a lazy operand; fails the test immediately with a message if it
    # is ever invoked.
    self.fail('this should not be called')

  def _tf_true(self):
    # [FUNCTION DOC]
    # HELPER: returns a constant true boolean tensor.
    return constant_op.constant(True)

  def _tf_false(self):
    # [FUNCTION DOC]
    # HELPER: returns a constant false boolean tensor.
    return constant_op.constant(False)

  def test_and_python(self):
    # [FUNCTION DOC]
    # Checks and_ on Python values: truthy results, returning the right operand's value, and
    # that a false left operand short-circuits (the right callable is never called).
    self.assertTrue(logical.and_(lambda: True, lambda: True))
    self.assertTrue(logical.and_(lambda: [1], lambda: True))
    self.assertListEqual(logical.and_(lambda: True, lambda: [1]), [1])

    self.assertFalse(logical.and_(lambda: False, lambda: True))
    self.assertFalse(logical.and_(lambda: False, self.assertNotCalled))

  @test_util.run_deprecated_v1
  def test_and_tf(self):
    # [FUNCTION DOC]
    # Runs in graph (v1) mode. Checks and_ with tensor left operands: true-and-true and
    # true-and-Python-true evaluate to true, false-and-true evaluates to false. A TODO notes
    # side-effecting ops are untested.
    with self.cached_session() as sess:
      t = logical.and_(self._tf_true, self._tf_true)
      self.assertEqual(self.evaluate(t), True)
      t = logical.and_(self._tf_true, lambda: True)
      self.assertEqual(self.evaluate(t), True)
      t = logical.and_(self._tf_false, lambda: True)
      self.assertEqual(self.evaluate(t), False)
      # TODO(mdan): Add a test for ops with side effects.

  def test_or_python(self):
    # [FUNCTION DOC]
    # Checks or_ on Python values: falsy results, returning the right operand's value, and that
    # a true left operand short-circuits (the right callable is never called).
    self.assertFalse(logical.or_(lambda: False, lambda: False))
    self.assertFalse(logical.or_(lambda: [], lambda: False))
    self.assertListEqual(logical.or_(lambda: False, lambda: [1]), [1])

    self.assertTrue(logical.or_(lambda: False, lambda: True))
    self.assertTrue(logical.or_(lambda: True, self.assertNotCalled))

  @test_util.run_deprecated_v1
  def test_or_tf(self):
    # [FUNCTION DOC]
    # Runs in graph (v1) mode. Checks or_ with tensor left operands: false-or-true, false-or-
    # Python-true and true-or-true all evaluate to true. A TODO notes side effects are untested.
    with self.cached_session() as sess:
      t = logical.or_(self._tf_false, self._tf_true)
      self.assertEqual(self.evaluate(t), True)
      t = logical.or_(self._tf_false, lambda: True)
      self.assertEqual(self.evaluate(t), True)
      t = logical.or_(self._tf_true, lambda: True)
      self.assertEqual(self.evaluate(t), True)
      # TODO(mdan): Add a test for ops with side effects.

  def test_not_python(self):
    # [FUNCTION DOC]
    # Checks not_ on Python values: negating true or a non-empty list is false, negating an
    # empty list is true.
    self.assertFalse(logical.not_(True))
    self.assertFalse(logical.not_([1]))
    self.assertTrue(logical.not_([]))

  def test_not_tf(self):
    # [FUNCTION DOC]
    # Checks not_ of a false tensor evaluates to true in a session.
    with self.cached_session() as sess:
      t = logical.not_(self._tf_false())
      self.assertEqual(self.evaluate(t), True)


if __name__ == '__main__':
  test.main()
