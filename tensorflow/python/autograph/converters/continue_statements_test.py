# [MODULE DOC]
# Unit tests for the autograph continue_statements converter, which rewrites continue statements
# in loops into equivalent guarded code. Each test defines a small function containing loops with
# continue statements, converts it with the converter, and checks that the converted function
# returns the same result as the original for several inputs. Importing has no side effects; the
# file runs the TensorFlow test main when executed directly.
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
"""Tests for continue_statements module."""

from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class ContinueCanonicalizationTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case covering the continue_statements converter. Inherits the transform helper from the
  # autograph converter testing base class. Has no state of its own; each test builds a local
  # function and compares original versus converted behavior.

  def assertTransformedEquivalent(self, f, *inputs):
    # [FUNCTION DOC]
    # HELPER: shared assertion used by every test in this class.
    # Converts function f with the continue_statements converter, calls both the original and the
    # converted function with the given positional inputs, and asserts the results are equal.
    # Returns nothing; fails the test on mismatch.
    tr = self.transform(f, continue_statements)
    self.assertEqual(f(*inputs), tr(*inputs))

  def test_basic(self):
    # [FUNCTION DOC]
    # Scenario: a while loop that skips even values with a single continue and collects the odd
    # ones. Expects converted and original functions to agree for inputs 0, 1, 3 and 4.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: a loop-based function with continue statements that is converted and compared
      # against its original form. Takes one input (a number or list) and returns the collected list(s).
      # Has no side effects beyond building local lists.
      v = []
      while x > 0:
        x -= 1
        if x % 2 == 0:
          continue
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 4)

  def test_multiple_continues(self):
    # [FUNCTION DOC]
    # Scenario: a while loop containing two separate conditional continue statements before the
    # final append. Expects equivalent results for inputs 0, 1, 3 and 4.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: a loop-based function with continue statements that is converted and compared
      # against its original form. Takes one input (a number or list) and returns the collected list(s).
      # Has no side effects beyond building local lists.
      v = []
      while x > 0:
        x -= 1
        if x > 1:
          continue
        if x > 2:
          continue
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 4)

  def test_multiple_continues_in_nested_scope(self):
    # [FUNCTION DOC]
    # Scenario: a for loop where one continue is in an if and another is inside an exception
    # handler after a deliberately raised error. Expects equivalent results for several lists
    # (empty, single elements, and a three element list).

    def f(a):
      # [FUNCTION DOC]
      # Test subject: a loop-based function with continue statements that is converted and compared
      # against its original form. Takes one input (a number or list) and returns the collected list(s).
      # Has no side effects beyond building local lists.
      v = []
      for x in a:
        x -= 1
        if x > 100:
          continue
        try:
          raise ValueError('intentional')
        except ValueError:
          continue
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, [])
    self.assertTransformedEquivalent(f, [1])
    self.assertTransformedEquivalent(f, [2])
    self.assertTransformedEquivalent(f, [1, 2, 3])

  def test_for_loop(self):
    # [FUNCTION DOC]
    # Scenario: a for loop over a list that decrements each item and skips even results with a
    # continue. Expects equivalent results for an empty list and several small lists.

    def f(a):
      # [FUNCTION DOC]
      # Test subject: a loop-based function with continue statements that is converted and compared
      # against its original form. Takes one input (a number or list) and returns the collected list(s).
      # Has no side effects beyond building local lists.
      v = []
      for x in a:
        x -= 1
        if x % 2 == 0:
          continue
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, [])
    self.assertTransformedEquivalent(f, [1])
    self.assertTransformedEquivalent(f, [2])
    self.assertTransformedEquivalent(f, [1, 2, 3])

  def test_nested_with(self):
    # [FUNCTION DOC]
    # Scenario: a continue located inside a name scope context manager block within a while loop.
    # Expects equivalent results for inputs 0, 1, 3 and 4.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: a loop-based function with continue statements that is converted and compared
      # against its original form. Takes one input (a number or list) and returns the collected list(s).
      # Has no side effects beyond building local lists.
      v = []
      while x > 0:
        x -= 1
        with ops.name_scope(''):
          if x % 2 == 0:
            continue
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 4)

  def test_nested_multiple_withs(self):
    # [FUNCTION DOC]
    # Scenario: a while loop with a continue inside one context manager block, followed by a
    # second context manager block and a plain statement. Expects equivalent results for inputs
    # 0, 1, 3 and 4.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: a loop-based function with continue statements that is converted and compared
      # against its original form. Takes one input (a number or list) and returns the collected list(s).
      # Has no side effects beyond building local lists.
      v = []
      while x > 0:
        x -= 1
        with ops.name_scope(''):
          if x % 2 == 0:
            continue
        with ops.name_scope(''):
          v.append(x)
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 4)

  def test_nested_multiple_withs_and_statements(self):
    # [FUNCTION DOC]
    # Scenario: like the multiple-withs case, but the first context manager block also has a
    # statement after the continue, plus further statements and a second block afterwards.
    # Expects equivalent results for inputs 0, 1, 3 and 4.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: a loop-based function with continue statements that is converted and compared
      # against its original form. Takes one input (a number or list) and returns the collected list(s).
      # Has no side effects beyond building local lists.
      v = []
      while x > 0:
        x -= 1
        with ops.name_scope(''):
          if x % 2 == 0:
            continue
          v.append(x)
        v.append(x)
        with ops.name_scope(''):
          v.append(x)
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 4)

  def test_nested_multiple_withs_and_nested_withs(self):
    # [FUNCTION DOC]
    # Scenario: a continue inside a context manager block that itself contains another nested
    # context manager block, followed by more blocks and statements. Expects equivalent results
    # for inputs 0, 1, 3 and 4.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: a loop-based function with continue statements that is converted and compared
      # against its original form. Takes one input (a number or list) and returns the collected list(s).
      # Has no side effects beyond building local lists.
      v = []
      while x > 0:
        x -= 1
        with ops.name_scope(''):
          if x % 2 == 0:
            continue
          with ops.name_scope(''):
            v.append(x)
        v.append(x)
        with ops.name_scope(''):
          v.append(x)
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 4)

  def test_nested(self):
    # [FUNCTION DOC]
    # Scenario: a while loop with nested conditionals where the continue sits in the inner else
    # branch, and three result lists are returned. Expects equivalent results for inputs 0, 1, 3
    # and 4.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: a loop-based function with continue statements that is converted and compared
      # against its original form. Takes one input (a number or list) and returns the collected list(s).
      # Has no side effects beyond building local lists.
      v = []
      u = []
      w = []
      while x > 0:
        x -= 1
        if x % 2 == 0:
          if x % 3 != 0:
            u.append(x)
          else:
            w.append(x)
            continue
        v.append(x)
      return v, u, w

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 4)

  def test_multiple_guarded_continues_with_side_effects(self):
    # [FUNCTION DOC]
    # Scenario: two continue conditions whose test expressions call a tracking function that
    # records each value in a list (a side effect). Verifies the converter evaluates each
    # condition exactly as many times as the original, by comparing both the tracked list and the
    # result list for inputs 3 and 2.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: a loop-based function with continue statements that is converted and compared
      # against its original form. Takes one input (a number or list) and returns the collected list(s).
      # Has no side effects beyond building local lists.
      def track(u, x):
        # [FUNCTION DOC]
        # HELPER: records a value for the test above. Appends x to list u (mutating it) and returns x.
        u.append(x)
        return x

      u = []
      v = []
      while x > 0:
        x -= 1
        if track(u, x) > 1:
          continue
        if track(u, x) > 2:
          continue
        v.append(x)
      return u, v

    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 2)


if __name__ == '__main__':
  test.main()
