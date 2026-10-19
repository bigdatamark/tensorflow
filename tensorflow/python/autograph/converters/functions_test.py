# [MODULE DOC]
# Unit tests for the autograph functions converter, which wraps converted functions and lambdas
# in a function scope. Tests check that tensor op names include the function name scope, that
# docstrings are preserved, that nested functions and methods get nested name scopes, that the
# conversion context is preserved in inner functions, and that lambdas returned from converted
# functions are marked as autograph artifacts. Runs the TensorFlow test main when executed.
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
"""Tests for functions module."""

from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.impl import api
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class FunctionTransformer(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the functions converter (named like the transformer it covers, though it is a
  # test class); uses the converter testing base class to transform and run small functions.

  def test_basic(self):
    # [FUNCTION DOC]
    # Scenario: a simple function with a docstring that adds to a tensor. Expects the result op
    # name to include the function name scope and the converted function to keep the docstring.

    def f(l):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the functions
      # converter; the test inspects the names of the ops it creates or its docstring.
      """Docstring."""
      a = 1
      l += a
      return l

    tr = self.transform(f, functions)

    result_op = tr(constant_op.constant(1))
    self.assertIn('f/', result_op.op.name)
    self.assertEqual('Docstring.', tr.__doc__)

  def test_multiline_docstring(self):
    # [FUNCTION DOC]
    # Scenario: a function with a multi-line docstring returning a constant tensor. Expects the op
    # name to contain the function scope and both docstring sentences to be preserved.

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the functions
      # converter; the test inspects the names of the ops it creates or its docstring.
      """First sentence.

      Second sentence.

      Returns:
        Something.
      """
      return constant_op.constant(1)

    tr = self.transform(f, functions)

    result_op = tr()
    self.assertIn('f/', result_op.op.name)
    self.assertIn('First sentence.', tr.__doc__)
    self.assertIn('Second sentence.', tr.__doc__)

  def test_nested_functions(self):
    # [FUNCTION DOC]
    # Scenario: a function with an inner function, converted with the functions and return
    # statement converters. Expects the first result op in the outer scope only, and the second
    # result derived from an input in the nested outer/inner scope.

    def f(l):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the functions
      # converter; the test inspects the names of the ops it creates or its docstring.

      def inner_fn(i):
        # [FUNCTION DOC]
        # Inner function that returns its argument plus one.
        return i + 1

      l += 1
      return l, inner_fn(l)

    tr = self.transform(f, (functions, return_statements))

    first, second = tr(constant_op.constant(1))
    self.assertIn('f/', first.op.name)
    self.assertNotIn('inner_fn', first.op.name)
    self.assertIn('f/inner_fn/', second.op.inputs[0].name)

  def test_conversion_context_preserves_in_inner_functions(self):
    # [FUNCTION DOC]
    # Scenario: an inner function calls a callee that asserts autograph conversion status is
    # disabled, run inside a context that disables it. Expects the disabled status to hold in the
    # callee after conversion (no assertion failure).

    def inner_fn_callee():
      # [FUNCTION DOC]
      # Asserts that the current autograph control status is disabled. Returns nothing.
      self.assertEqual(
          ag_ctx.control_status_ctx().status, ag_ctx.Status.DISABLED)

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the functions
      # converter; the test inspects the names of the ops it creates or its docstring.
      def inner_fn():
        # [FUNCTION DOC]
        # Inner function that calls the callee checking the conversion status. Returns nothing.
        inner_fn_callee()
      with ag_ctx.ControlStatusCtx(
          ag_ctx.Status.DISABLED, converter.ConversionOptions(recursive=True)):
        inner_fn()

    tr = self.transform(f, functions)

    tr()

  def test_method(self):
    # [FUNCTION DOC]
    # Scenario: a method with an inner function is converted and called on an instance. Expects the
    # same scope naming as for plain nested functions.

    class TestClass(object):
      # [CLASS DOC]
      # Local helper class with a method containing an inner function, used as the conversion target.

      def f(self, l):
        # [FUNCTION DOC]
        # Method under test: adds 1 to l and returns it together with the result of an inner function.

        def inner_fn(i):
          # [FUNCTION DOC]
          # Inner function that returns its argument plus one.
          return i + 1

        l += 1
        return l, inner_fn(l)

    tr = self.transform(TestClass.f, (functions, return_statements))

    first, second = tr(TestClass(), constant_op.constant(1))
    self.assertIn('f/', first.op.name)
    self.assertNotIn('inner_fn', first.op.name)
    self.assertIn('f/inner_fn/', second.op.inputs[0].name)

  def test_lambda_in_return_value(self):
    # [FUNCTION DOC]
    # Scenario: a function returning a lambda. Expects the returned lambda to be recognized as an
    # autograph artifact.

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the functions
      # converter; the test inspects the names of the ops it creates or its docstring.
      return lambda x: x + 1

    tr = self.transform(f, functions)

    result_l = tr()
    self.assertTrue(api.is_autograph_artifact(result_l))


if __name__ == '__main__':
  test.main()
