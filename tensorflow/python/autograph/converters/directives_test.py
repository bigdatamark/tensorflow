# [MODULE DOC]
# Unit tests for the autograph directives converter. Each test converts a small function that
# calls a directive (set element type or set loop options) and inspects the resulting AST
# annotations, or checks the errors raised for misplaced directives, or checks that resolving
# static values does not evaluate properties or custom attribute hooks. Runs the TensorFlow test
# main when executed directly.
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
"""Tests for directives module."""

from tensorflow.python.autograph.converters import directives as directives_converter
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.platform import test


class DirectivesTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the directives converter; uses the converter testing base class to transform
  # functions and return their AST.

  def test_local_target(self):
    # [FUNCTION DOC]
    # Scenario: the element type directive applied to a local list. Expects the definition of the
    # list to carry directive arguments with dtype constant 'a' and shape referring to the name of
    # another local variable.

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test; it is converted by the
      # directives converter and its resulting AST is inspected (it is not meant to be executed).
      l = []
      string_var = 0
      directives.set_element_type(l, 'a', string_var)

    _, node, _ = self.transform(f, directives_converter, include_ast=True)

    def_, = anno.getanno(node.body[0].targets[0],
                         anno.Static.DEFINITIONS)
    d = def_.directives[directives.set_element_type]
    self.assertEqual(d['dtype'].value, 'a')
    self.assertEqual(d['shape'].id, 'string_var')

  def test_argument_target(self):
    # [FUNCTION DOC]
    # Scenario: the element type directive applied to a function argument, with shape given by
    # keyword. Expects the argument definition to carry dtype 1 and shape 2.

    def f(a):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test; it is converted by the
      # directives converter and its resulting AST is inspected (it is not meant to be executed).
      directives.set_element_type(a, 1, shape=2)
      pass

    _, node, _ = self.transform(f, directives_converter, include_ast=True)

    def_, = anno.getanno(node.args.args[0], anno.Static.DEFINITIONS)
    d = def_.directives[directives.set_element_type]
    self.assertEqual(d['dtype'].value, 1)
    self.assertEqual(d['shape'].value, 2)

  def test_loop_target(self):
    # [FUNCTION DOC]
    # Scenario: the loop options directive as the first statement of a while loop. Expects the
    # loop node annotation to hold parallel iterations 10 and a back prop argument referring to a
    # variable, and no swap memory entry.

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test; it is converted by the
      # directives converter and its resulting AST is inspected (it is not meant to be executed).
      a = True
      while True:
        directives.set_loop_options(parallel_iterations=10, back_prop=a)  # pylint: disable=unexpected-keyword-arg
        pass

    _, node, _ = self.transform(f, directives_converter, include_ast=True)

    d = anno.getanno(node.body[1], anno.Basic.DIRECTIVES)
    d = d[directives.set_loop_options]
    self.assertEqual(d['parallel_iterations'].value, 10)
    self.assertEqual(d['back_prop'].id, 'a')
    self.assertNotIn('swap_memory', d)

  def test_loop_target_no_loop(self):
    # [FUNCTION DOC]
    # Scenario: the loop options directive used outside any loop. Expects a value error saying it
    # must be used inside a statement.

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test; it is converted by the
      # directives converter and its resulting AST is inspected (it is not meant to be executed).
      directives.set_loop_options()
      pass

    with self.assertRaisesRegex(ValueError, 'must be used inside a statement'):
      self.transform(f, directives_converter, include_ast=True)

  def test_loop_target_not_first(self):
    # [FUNCTION DOC]
    # Scenario: the loop options directive placed after another statement in the loop. Expects a
    # value error saying it must be the first statement.

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test; it is converted by the
      # directives converter and its resulting AST is inspected (it is not meant to be executed).
      a = 1
      while True:
        a = 2
        directives.set_loop_options(parallel_iterations=10, back_prop=a)  # pylint: disable=unexpected-keyword-arg

    with self.assertRaisesRegex(ValueError, 'must be the first statement'):
      self.transform(f, directives_converter, include_ast=True)

  def test_value_verification_does_not_trigger_properties(self):
    # [FUNCTION DOC]
    # Scenario: the function accesses a property of a captured object whose getter fails the test
    # if run. Verifies the conversion succeeds (returns a node) without evaluating the property.

    self_test = self

    class TestClass(object):
      # [CLASS DOC]
      # Local helper class whose special members are probed to ensure they are not evaluated.

      @property
      def b(self):
        # [FUNCTION DOC]
        # Property getter that fails the test if it is ever evaluated.
        self_test.fail('This should never be evaluated')

    tc = TestClass()

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test; it is converted by the
      # directives converter and its resulting AST is inspected (it is not meant to be executed).
      return tc.b + 1

    _, node, _ = self.transform(f, directives_converter, include_ast=True)

    self.assertIsNotNone(node)

  def test_value_verification_does_not_trigger_getattr(self):
    # [FUNCTION DOC]
    # Scenario: the function accesses a missing attribute of a captured object with a custom
    # attribute fallback that records being called. Verifies conversion succeeds and the fallback
    # was never called.

    class TestClass(object):
      # [CLASS DOC]
      # Local helper class whose special members are probed to ensure they are not evaluated.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor: initializes the flag recording whether the attribute fallback was called to false.
        self.getattr_called = False

      def __getattr__(self, _):
        # [FUNCTION DOC]
        # Attribute fallback hook: sets the flag recording that it was called; returns nothing.
        # Note: seems that any exception raised here is absorbed by hasattr.
        # So we can't call test.fail or raise.
        self.getattr_called = True

    tc = TestClass()

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test; it is converted by the
      # directives converter and its resulting AST is inspected (it is not meant to be executed).
      return tc.b + 1

    _, node, _ = self.transform(f, directives_converter, include_ast=True)

    self.assertIsNotNone(node)
    self.assertFalse(tc.getattr_called)


if __name__ == '__main__':
  test.main()
