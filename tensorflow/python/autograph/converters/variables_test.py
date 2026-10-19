# [MODULE DOC]
# Unit tests for the AutoGraph variables converter. Verifies that variable reads are wrapped in
# the overridden load runtime function (tests replace it with a function adding one), that
# augmented assignments produce an extra read, that deleting a plain name makes later use raise
# a name error, that composite deletes (indexing, slicing, attributes, list packing) are left
# untouched, and that attribute, subscript and call expressions read their base through the
# load function. Runs the TensorFlow test main when executed as a script.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for variables module."""

from tensorflow.python.autograph.converters import variables
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class VariablesTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the variables converter. Shares a helper that transforms with a test load
  # function overriding the AutoGraph load operator, so reads are observable as adding one.

  def transform_with_test_ld(self, f):
    # [FUNCTION DOC]
    # HELPER: shared by the tests in this class.
    # Transforms the given function with the variables converter, overriding the runtime load
    # function with one that adds one to the value read. Returns the converted function.
    """Generates code which adds 1 to all variable reads."""
    return self.transform(f, variables, ag_overrides={'ld': lambda x: x + 1})

  def test_read(self):
    # [FUNCTION DOC]
    # Scenario: a function returning its argument. With the add-one load override, calling the
    # converted function with one must return two (one read was wrapped).

    def f(l):
      # [FUNCTION DOC]
      # Test subject: returns its argument.
      return l

    tr = self.transform_with_test_ld(f)

    self.assertEqual(tr(1), 2)

  def test_aug_assign(self):
    # [FUNCTION DOC]
    # Scenario: multiply-assign on an argument then return it. With the add-one load override
    # there are two reads (one for the augmented assignment target, one for the return), so for
    # input one the expected result is ((1 plus 1) times 10) plus 1.

    def f(l):
      # [FUNCTION DOC]
      # Test subject: multiplies its argument by ten in place and returns it.
      l *= 10
      return l

    tr = self.transform_with_test_ld(f)

    self.assertEqual(tr(1), (1 + 1) * 10 + 1)  # two reads

  def test_del(self):
    # [FUNCTION DOC]
    # Scenario: a function deletes its argument and then returns it. The converted function must
    # raise a name error saying the variable is used before assignment.

    def f(l):
      # [FUNCTION DOC]
      # Test subject: deletes its argument, then returns it.
      del l
      return l

    tr = self.transform(f, variables)

    with self.assertRaisesRegex(NameError, "'l' is used before assignment"):
      tr(1)

  def test_del_getitem_ignored_basic_slice(self):
    # [FUNCTION DOC]
    # Scenario: deleting an indexed list item is not rewritten. Calling the converted function on
    # a two-element list must return the list with the first item removed.

    def f(l):
      # [FUNCTION DOC]
      # Test subject: deletes item zero of its list argument and returns the list.
      del l[0]
      return l

    tr = self.transform(f, variables)

    self.assertListEqual([2], tr([1, 2]))

  def test_del_getitem_ignored_range_slice(self):
    # [FUNCTION DOC]
    # Scenario: deleting a range slice of a list is not rewritten. Calling the converted function
    # on a two-element list must return an empty list.

    def f(l):
      # [FUNCTION DOC]
      # Test subject: deletes the slice from zero to two of its list argument and returns it.
      del l[0:2]
      return l

    tr = self.transform(f, variables)

    self.assertListEqual([], tr([1, 2]))

  def test_del_getattr_ignored(self):
    # [FUNCTION DOC]
    # Scenario: deleting an attribute of an object is not rewritten. The converted function is
    # called on fresh objects; the result must lack attribute a while attribute b remains two.

    def f(l):
      # [FUNCTION DOC]
      # Test subject: deletes attribute a of its argument and returns the argument.
      del l.a
      return l

    class TestClass(object):
      # [CLASS DOC]
      # Local fixture class: a plain object with two integer attributes, a (one) and b (two).

      def __init__(self):
        # [FUNCTION DOC]
        # Sets attribute a to one and attribute b to two.
        self.a = 1
        self.b = 2

    tr = self.transform(f, variables)

    self.assertFalse(hasattr(tr(TestClass()), 'a'))
    self.assertEqual(tr(TestClass()).b, 2)

  def test_del_packing_ignored_list(self):
    # [FUNCTION DOC]
    # Scenario: deleting a list-packed group of names is not rewritten, so the later use of a
    # deleted name raises the native unbound-local error rather than the AutoGraph name error.
    # Note: testing for UnboundLocalError, not NameError because in this case we
    # don't rewrite the del.

    def f(a, b):
      # [FUNCTION DOC]
      # Test subject: deletes a list of names a and b and returns a.
      del [a, b]
      return a

    tr = self.transform(f, variables)

    with self.assertRaises(UnboundLocalError):
      tr(1, 2)

  def test_del_packing_ignored_nested(self):
    # [FUNCTION DOC]
    # Scenario: deleting a nested packed group of names is not rewritten, so using a deleted
    # name raises the native unbound-local error.
    # Note: testing for UnboundLocalError, not NameError because in this case we
    # don't rewrite the del.

    def f(a, b, c):
      # [FUNCTION DOC]
      # Test subject: deletes a list holding a and a pair of b and c, then returns c.
      del [a, (b, c)]
      return c

    tr = self.transform(f, variables)

    with self.assertRaises(UnboundLocalError):
      tr(1, 2, 3)

  def test_del_item_multiple_mixed_used_after(self):
    # [FUNCTION DOC]
    # Scenario: one delete statement mixes plain names and an indexed item; name a is reassigned
    # later but name b is used afterwards without reassignment. Expects a name error saying b is
    # used before assignment.

    def f(a, b, c):
      # [FUNCTION DOC]
      # Test subject: deletes a, b and item zero of c, reassigns a, and returns all three.
      del a, b, c[0]
      a = 1
      return a, b, c

    tr = self.transform(f, variables)

    with self.assertRaisesRegex(NameError, "'b' is used before assignment"):
      tr(1, 2, [1, 2])

  def test_del_item_multiple_mixed_unused_after(self):
    # [FUNCTION DOC]
    # Scenario: same mixed delete, but both deleted names are reassigned before use. Expects the
    # converted function to succeed and return the list with its first item removed.

    def f(a, b, c):
      # [FUNCTION DOC]
      # Test subject: deletes a, b and item zero of c, reassigns a and b, and returns c.
      del a, b, c[0]
      a = 1
      b = 2
      return c

    tr = self.transform(f, variables)

    self.assertListEqual([2], tr(1, 2, [1, 2]))

  def test_attribute(self):
    # [FUNCTION DOC]
    # Scenario: reading an attribute of an object. The fixture object supports addition, so the
    # add-one load override applied to it increments its attribute. Expects result two.

    class TestClass(object):
      # [CLASS DOC]
      # Local fixture class: holds an integer value v starting at one; adding a number to it
      # increases v in place and returns the same object.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets attribute v to one.
        self.v = 1

      def __add__(self, other):
        # [FUNCTION DOC]
        # Adds the other value to attribute v in place and returns the same object.
        self.v += other
        return self

    def f(l):
      # [FUNCTION DOC]
      # Test subject: returns attribute v of its argument.
      return l.v

    tc = TestClass()
    tr = self.transform_with_test_ld(f)

    self.assertEqual(tr(tc), 2)

  def test_subscript(self):
    # [FUNCTION DOC]
    # Scenario: indexing an object. The fixture supports addition and indexing; the base object is
    # read through the add-one load override before indexing. Expects result two.

    class TestClass(object):
      # [CLASS DOC]
      # Local fixture class: holds integer v starting at one, supports in-place addition and
      # returns v for any index.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets attribute v to one.
        self.v = 1

      def __add__(self, other):
        # [FUNCTION DOC]
        # Adds the other value to attribute v in place and returns the same object.
        self.v += other
        return self

      def __getitem__(self, _):
        # [FUNCTION DOC]
        # Index access: ignores the index and returns attribute v.
        return self.v

    def f(l):
      # [FUNCTION DOC]
      # Test subject: returns the item at index zero of its argument.
      return l[0]

    tc = TestClass()
    tr = self.transform_with_test_ld(f)

    self.assertEqual(tr(tc), 2)

  def test_call(self):
    # [FUNCTION DOC]
    # Scenario: calling an object. The fixture supports addition and calling; the callee is read
    # through the add-one load override before being called. Expects result two.

    class TestClass(object):
      # [CLASS DOC]
      # Local fixture class: holds integer v starting at one, supports in-place addition and
      # returns v when called.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets attribute v to one.
        self.v = 1

      def __add__(self, other):
        # [FUNCTION DOC]
        # Adds the other value to attribute v in place and returns the same object.
        self.v += other
        return self

      def __call__(self):
        # [FUNCTION DOC]
        # Call operator: returns attribute v.
        return self.v

    def f(l):
      # [FUNCTION DOC]
      # Test subject: calls its argument with no arguments and returns the result.
      return l()

    tc = TestClass()
    tr = self.transform_with_test_ld(f)

    self.assertEqual(tr(tc), 2)


if __name__ == '__main__':
  test.main()
