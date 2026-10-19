# [MODULE DOC]
# Python-3-only tests for the reaching definitions analysis (nonlocal declarations), extending
# the shared test base class from the main reaching definitions test module. Runs the test
# runner when executed as a script.
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
"""Tests for reaching_definitions module, that only run in Python 3."""

from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions_test
from tensorflow.python.platform import test


class ReachingDefinitionsAnalyzerTest(
    reaching_definitions_test.ReachingDefinitionsAnalyzerTestBase):
  # [CLASS DOC]
  # Test case covering reaching-definition analysis of functions using nonlocal declarations.
  # Inherits the parse-and-analyze helper and definition assertions from the shared base class.
  """Tests which can only run in Python 3."""

  def test_nonlocal(self):
    # [FUNCTION DOC]
    # Scenario: a function declares two outer variables nonlocal, tests one, conditionally
    # assigns the other, and returns both. Analyzes it and asserts that the tested name and the
    # assigned target each have one definition, the returned first name has one definition
    # identical to the tested one, the returned second name has two (outer and assigned), and
    # the if statement has both names defined on entry.

    a = 3
    b = 13

    def test_fn():
      # [FUNCTION DOC]
      # Analyzed function: declares both outer names nonlocal, assigns an empty list to the
      # second if the first is truthy, and returns both as a tuple.
      nonlocal a
      nonlocal b
      if a:
        b = []
      return a, b

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[2].test, 1)
    self.assertHasDefs(fn_body[2].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[1], 2)

    self.assertSameDef(fn_body[2].test, fn_body[3].value.elts[0])

    self.assertHasDefinedIn(fn_body[2], ('a', 'b'))

  def test_nonlocal_in_nested_function(self):
    # [FUNCTION DOC]
    # Scenario: an outer function defines two locals and a nested function that declares them
    # nonlocal, conditionally reassigns one, and returns both. Analyzes the outer function and
    # asserts, on the nested body, the same definition counts and identity as the plain nonlocal
    # test (one, one, one, two), and that both names are defined on entry to the nested if.

    a = 3
    b = 13

    def test_fn():
      # [FUNCTION DOC]
      # Analyzed outer function: defines two locals, a nested function, and returns its call.
      a = 3
      b = 13

      def local_fn():
        # [FUNCTION DOC]
        # Nested function: declares both outer names nonlocal, assigns an empty list to the
        # second if the first is truthy, and returns both as a tuple.
        nonlocal a, b
        if a:
          b = []
        return a, b

      return local_fn()

    node = self._parse_and_analyze(test_fn)
    local_body = node.body[2].body

    self.assertHasDefs(local_body[1].test, 1)
    self.assertHasDefs(local_body[1].body[0].targets[0], 1)
    self.assertHasDefs(local_body[2].value.elts[0], 1)
    self.assertHasDefs(local_body[2].value.elts[1], 2)

    self.assertSameDef(local_body[1].test, local_body[2].value.elts[0])

    # Note: the function name is visible inside the function body. But it's
    # a closure variable, not a local.
    #
    # Example:
    #
    #   >>> def f():
    #   ...  print(f)
    #   >>> g = f
    #   >>> f = 'something else'
    #   >>> g()
    #   something else
    #
    self.assertHasDefinedIn(local_body[1], ('a', 'b'))


if __name__ == '__main__':
  test.main()
