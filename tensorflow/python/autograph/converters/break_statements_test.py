# [MODULE DOC]
# Tests for the AutoGraph break-statement converter: checks that converted while and for loops
# (plain, nested, with else clauses, with side effects) behave like the originals, and that
# directive annotations survive conversion.
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
"""Tests for break_statements module."""

from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.platform import test


class BreakCanonicalizationTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the break converter, built on the converter testing base class (provides the
  # transform helper). Defines an equivalence-check helper used by most tests.

  def assertTransformedEquivalent(self, f, *inputs):
    # [FUNCTION DOC]
    # HELPER: assertion shared by the tests. Converts function f with the break converter, runs
    # both original and converted with the given inputs, and asserts equal results.
    tr = self.transform(f, break_statements)
    self.assertEqual(f(*inputs), tr(*inputs))

  def test_while_loop(self):
    # [FUNCTION DOC]
    # A while loop that decrements and breaks on even values, collecting odd values; checks
    # converted behavior equals the original for inputs 0, 1 and 4.

    def f(x):
      # [FUNCTION DOC]
      # Sample function: loops while positive, breaks when even, returns collected values.
      v = []
      while x > 0:
        x -= 1
        if x % 2 == 0:
          break
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 4)

  def test_while_loop_preserves_directives(self):
    # [FUNCTION DOC]
    # Transforms a function with a breaking while loop to get its tree and context, attaches a
    # marker directive annotation to the original loop, applies the break converter, and asserts
    # the new loop node carries the same marker object.

    def f(x):
      # [FUNCTION DOC]
      # Sample function with a while loop that breaks on even values; returns nothing.
      while x > 0:
        x -= 1
        if x % 2 == 0:
          break

    _, node, ctx = self.transform(f, (), include_ast=True)
    fake_annotation = object()
    anno.setanno(node.body[0], anno.Basic.DIRECTIVES, fake_annotation)
    node = break_statements.transform(node, ctx)

    self.assertIs(
        anno.getanno(node.body[1], anno.Basic.DIRECTIVES), fake_annotation)

  def test_for_loop(self):
    # [FUNCTION DOC]
    # Converts a for loop that breaks on an even decremented value and asserts that for input
    # list of 5 and 4 the converted function returns a list holding only 3.

    def f(a):
      # [FUNCTION DOC]
      # Sample function: iterates, decrements each item, breaks on even, collects odd values.
      v = []
      for x in a:
        x -= 1
        if x % 2 == 0:
          break
        v.append(x)
      return v

    tr = self.transform(f, break_statements)

    self.assertEqual([3], tr([5, 4]))

  def test_for_loop_preserves_directives(self):
    # [FUNCTION DOC]
    # Same as the while-loop directive test but for a for loop: marker annotation on the original
    # loop must be present on the loop node after conversion.

    def f(a):
      # [FUNCTION DOC]
      # Sample function with a for loop that breaks on even items; returns nothing.
      for x in a:
        if x % 2 == 0:
          break

    _, node, ctx = self.transform(f, (), include_ast=True)
    fake_annotation = object()
    anno.setanno(node.body[0], anno.Basic.DIRECTIVES, fake_annotation)
    node = break_statements.transform(node, ctx)
    self.assertIs(
        anno.getanno(node.body[1], anno.Basic.DIRECTIVES), fake_annotation)

  def test_nested(self):
    # [FUNCTION DOC]
    # Breaks inside nested conditionals within a while loop; converted behavior must equal the
    # original for inputs 0, 3 and 11.

    def f(x):
      # [FUNCTION DOC]
      # Sample function: collects values into three lists depending on divisibility, breaking in
      # one branch; returns the three lists.
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
            break
        v.append(x)
      return v, u, w

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 11)

  def test_nested_loops(self):
    # [FUNCTION DOC]
    # Nested while loops each with their own break; converted behavior must equal the original for
    # inputs 0, 2, 3 and 5.

    def f(x):
      # [FUNCTION DOC]
      # Sample function with an inner and outer breaking while loop; returns two lists.
      v = []
      u = []
      while x > 0:
        x -= 1
        y = x
        while y > 0:
          y -= 1
          if y % 2 == 0:
            break
          u.append(y)
        if x == 0:
          break
        v.append(x)
      return v, u

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 5)

  def test_loop_orelse(self):
    # [FUNCTION DOC]
    # While loops with else clauses containing breaks (a break in the else belongs to the outer
    # loop); converted behavior must equal the original for inputs 0, 2 and 3.

    def f(x):
      # [FUNCTION DOC]
      # Sample function with an inner loop having an else clause that breaks the outer loop;
      # returns two lists.
      v = []
      u = []
      while x > 0:
        x -= 1
        y = x
        while y > 1:
          break
        else:
          u.append(y)
          break
        v.append(x)
      return v, u

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, 3)

  def test_multiple_correlated_breaks_with_side_effects(self):
    # [FUNCTION DOC]
    # Loop with two break points where a side effect (list append) precedes the second break;
    # checks that after the first break path no later side effect runs. Equivalence is asserted
    # for true and false inputs.
    def f(cond1):
      # [FUNCTION DOC]
      # Sample function: infinite loop appending to a list or breaking on the condition, then
      # breaking when the last element is positive; returns the list.
      lst = []
      while True:
        if cond1:
          lst.append(1)
        else:
          break
        if lst[-1] > 0:  # lst always has an element here
          break
      return lst

    self.assertTransformedEquivalent(f, True)
    self.assertTransformedEquivalent(f, False)


if __name__ == '__main__':
  test.main()
