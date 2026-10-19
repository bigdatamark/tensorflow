# [MODULE DOC]
# Unit tests for the test_combinations module: combine, times, addition of combination lists,
# the overlapping-keys error and the generate decorator applied to a parameterized test class.
# Contains one plain test class and one parameterized suite generated from a combination list.
# No import-time side effects beyond imports; run via the standard test main entry point.
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
"""Tests generating test combinations."""

from collections import OrderedDict

from absl.testing import parameterized

from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.eager import test


class TestingCombinationsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the combination helpers: combine, times, list addition and error handling
  # for overlapping keys. Each test compares the produced list of mappings against a literal.

  def test_combine(self):
    # [FUNCTION DOC]
    # Checks that combining two keyword lists yields the full cross product as a list of plain
    # mappings (a with 1,2 and b with 2,3 gives four entries). Asserts equality; returns nothing.
    self.assertEqual([{
        "a": 1,
        "b": 2
    }, {
        "a": 1,
        "b": 3
    }, {
        "a": 2,
        "b": 2
    }, {
        "a": 2,
        "b": 3
    }], combinations.combine(a=[1, 2], b=[2, 3]))

  def test_arguments_sorted(self):
    # [FUNCTION DOC]
    # Checks that combine orders the keys of each produced mapping alphabetically regardless of
    # the order the keyword arguments were given. Asserts equality with ordered mappings.
    self.assertEqual([
        OrderedDict([("aa", 1), ("ab", 2)]),
        OrderedDict([("aa", 1), ("ab", 3)]),
        OrderedDict([("aa", 2), ("ab", 2)]),
        OrderedDict([("aa", 2), ("ab", 3)])
    ], combinations.combine(ab=[2, 3], aa=[1, 2]))

  def test_combine_single_parameter(self):
    # [FUNCTION DOC]
    # Checks that a non-list (scalar) keyword value is treated as a single fixed choice, so only
    # the list-valued argument multiplies the combinations. Asserts equality; returns nothing.
    self.assertEqual([{
        "a": 1,
        "b": 2
    }, {
        "a": 2,
        "b": 2
    }], combinations.combine(a=[1, 2], b=2))

  def test_add(self):
    # [FUNCTION DOC]
    # Checks that adding two combination lists concatenates them (a union, not a product).
    # Asserts the result equals the first list followed by the second; returns nothing.
    self.assertEqual(
        [{
            "a": 1
        }, {
            "a": 2
        }, {
            "b": 2
        }, {
            "b": 3
        }],
        combinations.combine(a=[1, 2]) + combinations.combine(b=[2, 3]))

  def test_times(self):
    # [FUNCTION DOC]
    # Builds graph and eager combinations, adds them, and takes the product with a distribution
    # combination. Asserts the product has every distribution paired with every mode/loss entry,
    # in the expected order, with sorted keys.
    c1 = combinations.combine(mode=["graph"], loss=["callable", "tensor"])
    c2 = combinations.combine(mode=["eager"], loss=["callable"])
    c3 = combinations.combine(distribution=["d1", "d2"])
    c4 = combinations.times(c3, c1 + c2)
    self.assertEqual([
        OrderedDict([("distribution", "d1"), ("loss", "callable"),
                     ("mode", "graph")]),
        OrderedDict([("distribution", "d1"), ("loss", "tensor"),
                     ("mode", "graph")]),
        OrderedDict([("distribution", "d1"), ("loss", "callable"),
                     ("mode", "eager")]),
        OrderedDict([("distribution", "d2"), ("loss", "callable"),
                     ("mode", "graph")]),
        OrderedDict([("distribution", "d2"), ("loss", "tensor"),
                     ("mode", "graph")]),
        OrderedDict([("distribution", "d2"), ("loss", "callable"),
                     ("mode", "eager")])
    ], c4)

  def test_times_variable_arguments(self):
    # [FUNCTION DOC]
    # Checks that times accepts more than two combination lists and equals the single combine
    # call with all keywords. Asserts the ordered product and then equality with the combine result.
    c1 = combinations.combine(mode=["graph", "eager"])
    c2 = combinations.combine(optimizer=["adam", "gd"])
    c3 = combinations.combine(distribution=["d1", "d2"])
    c4 = combinations.times(c3, c1, c2)
    self.assertEqual([
        OrderedDict([("distribution", "d1"), ("mode", "graph"),
                     ("optimizer", "adam")]),
        OrderedDict([("distribution", "d1"), ("mode", "graph"),
                     ("optimizer", "gd")]),
        OrderedDict([("distribution", "d1"), ("mode", "eager"),
                     ("optimizer", "adam")]),
        OrderedDict([("distribution", "d1"), ("mode", "eager"),
                     ("optimizer", "gd")]),
        OrderedDict([("distribution", "d2"), ("mode", "graph"),
                     ("optimizer", "adam")]),
        OrderedDict([("distribution", "d2"), ("mode", "graph"),
                     ("optimizer", "gd")]),
        OrderedDict([("distribution", "d2"), ("mode", "eager"),
                     ("optimizer", "adam")]),
        OrderedDict([("distribution", "d2"), ("mode", "eager"),
                     ("optimizer", "gd")])
    ], c4)
    self.assertEqual(
        combinations.combine(
            mode=["graph", "eager"],
            optimizer=["adam", "gd"],
            distribution=["d1", "d2"]), c4)

  def test_overlapping_keys(self):
    # [FUNCTION DOC]
    # Checks that taking the product of two combination lists that share keys (mode and loss)
    # raises a value error whose message mentions overlapping keys.
    c1 = combinations.combine(mode=["graph"], loss=["callable", "tensor"])
    c2 = combinations.combine(mode=["eager"], loss=["callable"])
    with self.assertRaisesRegex(ValueError, ".*Keys.+overlap.+"):
      _ = combinations.times(c1, c2)


@combinations.generate(combinations.combine(a=[1, 0], b=[2, 3], c=[1]))
class CombineTheTestSuite(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized suite expanded by the generate decorator over a=1,0, b=2,3, c=1 (four
  # combinations). Public test methods receive a, b, c; non-test and underscore-prefixed
  # methods must be left untouched by the decorator and so are never run with arguments.

  def test_add_things(self, a, b, c):
    # [FUNCTION DOC]
    # Parameterized test run once per combination. Asserts the sum a+b+c lies between 3 and 5
    # inclusive, which holds for all generated combinations.
    self.assertLessEqual(3, a + b + c)
    self.assertLessEqual(a + b + c, 5)

  def test_add_things_one_more(self, a, b, c):
    # [FUNCTION DOC]
    # Second parameterized test with the same body as the first; verifies every public test
    # method of the class gets expanded. Asserts the sum a+b+c lies between 3 and 5 inclusive.
    self.assertLessEqual(3, a + b + c)
    self.assertLessEqual(a + b + c, 5)

  def not_a_test(self, a=0, b=0, c=0):
    # [FUNCTION DOC]
    # Method whose name does not start with the test prefix, so it must not be collected or
    # expanded. Arguments default to zero and are discarded; fails the test if ever invoked.
    del a, b, c
    self.fail()

  def _test_but_private(self, a=0, b=0, c=0):
    # [FUNCTION DOC]
    # Underscore-prefixed method that must not be treated as a test by the decorator. Arguments
    # default to zero and are discarded; fails the test if it is ever invoked.
    del a, b, c
    self.fail()

  # Check that nothing funny happens to a non-callable that starts with "_test".
  test_member = 0


if __name__ == "__main__":
  test.main()
