# [MODULE DOC]
# Test module for the function utilities: argument-name retrieval, keyword catch-all detection,
# callable name retrieval and code-object retrieval, covering functions, methods, callable
# instances, partials, lambdas and invalid input.
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
"""Tests for function args retrieval utils."""

import functools

from tensorflow.python.platform import test
from tensorflow.python.util import function_utils


def silly_example_function():
  # [FUNCTION DOC]
  # HELPER: module-level no-op function used as a plain function target in name/code tests.
  pass


class SillyCallableClass(object):
  # [CLASS DOC]
  # HELPER: test fixture class whose instances are callable (no arguments, no result), used
  # to test name and code retrieval for callable instances.

  def __call__(self):
    # [FUNCTION DOC]
    # HELPER: does nothing; makes instances callable.
    pass


class FnArgsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for function_utils.fn_args covering plain functions, callable instances, bound
  # methods, and (nested) partial objects with positional and keyword bindings.

  def test_simple_function(self):
    # [FUNCTION DOC]
    # Scenario: a plain function with two parameters.
    # Asserts fn_args returns both names, in order.
    def fn(a, b):
      # [FUNCTION DOC]
      # HELPER: sample function with parameters a and b; returns their sum.
      return a + b
    self.assertEqual(('a', 'b'), function_utils.fn_args(fn))

  def test_callable(self):
    # [FUNCTION DOC]
    # Scenario: an instance of a class whose call method takes a and b.
    # Asserts fn_args reports the call-method parameters without the self parameter.

    class Foo(object):
      # [CLASS DOC]
      # HELPER: local fixture class with a two-parameter call method.

      def __call__(self, a, b):
        # [FUNCTION DOC]
        # HELPER: returns the sum of its two arguments.
        return a + b

    self.assertEqual(('a', 'b'), function_utils.fn_args(Foo()))

  def test_bound_method(self):
    # [FUNCTION DOC]
    # Scenario: a method bound to an instance, with parameters a and b.
    # Asserts fn_args omits the self parameter and returns a and b.

    class Foo(object):
      # [CLASS DOC]
      # HELPER: local fixture class with one ordinary method.

      def bar(self, a, b):
        # [FUNCTION DOC]
        # HELPER: returns the sum of its two arguments.
        return a + b

    self.assertEqual(('a', 'b'), function_utils.fn_args(Foo().bar))

  def test_bound_method_no_self(self):
    # [FUNCTION DOC]
    # Scenario: a bound method declared with only variadic positional arguments (no named self).
    # Asserts fn_args returns an empty tuple.

    class Foo(object):
      # [CLASS DOC]
      # HELPER: local fixture class whose method takes only variadic positional arguments.

      def bar(*args):  # pylint:disable=no-method-argument
        # [FUNCTION DOC]
        # HELPER: returns the sum of the second and third positional arguments.
        return args[1] + args[2]

    self.assertEqual((), function_utils.fn_args(Foo().bar))

  def test_partial_function(self):
    # [FUNCTION DOC]
    # Scenario: a partial that binds the parameter test_arg by keyword.
    # Asserts fn_args returns only the remaining name a.
    expected_test_arg = 123

    def fn(a, test_arg):
      # [FUNCTION DOC]
      # HELPER: sample function; returns an error object if test_arg differs from the expected
      # value (123), otherwise returns a.
      if test_arg != expected_test_arg:
        return ValueError('partial fn does not work correctly')
      return a

    wrapped_fn = functools.partial(fn, test_arg=123)

    self.assertEqual(('a',), function_utils.fn_args(wrapped_fn))

  def test_partial_function_with_positional_args(self):
    # [FUNCTION DOC]
    # Scenario: a partial that binds the first parameter positionally.
    # Asserts fn_args returns only a, and that calling the partial positionally or by keyword
    # returns the given value (three).
    expected_test_arg = 123

    def fn(test_arg, a):
      # [FUNCTION DOC]
      # HELPER: sample function; returns an error object if test_arg differs from the expected
      # value (123), otherwise returns a.
      if test_arg != expected_test_arg:
        return ValueError('partial fn does not work correctly')
      return a

    wrapped_fn = functools.partial(fn, 123)

    self.assertEqual(('a',), function_utils.fn_args(wrapped_fn))

    self.assertEqual(3, wrapped_fn(3))
    self.assertEqual(3, wrapped_fn(a=3))

  def test_double_partial(self):
    # [FUNCTION DOC]
    # Scenario: a partial of a partial, both binding parameters by keyword.
    # Asserts fn_args returns only the unbound name a.
    expected_test_arg1 = 123
    expected_test_arg2 = 456

    def fn(a, test_arg1, test_arg2):
      # [FUNCTION DOC]
      # HELPER: sample function; returns an error object if either bound value differs from its
      # expected value, otherwise returns a.
      if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
        return ValueError('partial does not work correctly')
      return a

    wrapped_fn = functools.partial(fn, test_arg2=456)
    double_wrapped_fn = functools.partial(wrapped_fn, test_arg1=123)

    self.assertEqual(('a',), function_utils.fn_args(double_wrapped_fn))

  def test_double_partial_with_positional_args_in_outer_layer(self):
    # [FUNCTION DOC]
    # Scenario: inner partial binds the last parameter by keyword; the outer partial binds the
    # first parameter positionally.
    # Asserts fn_args returns only a, and calling the double partial positionally or by keyword
    # returns the given value (three).
    expected_test_arg1 = 123
    expected_test_arg2 = 456

    def fn(test_arg1, a, test_arg2):
      # [FUNCTION DOC]
      # HELPER: sample function; returns an error object if either bound value differs from its
      # expected value, otherwise returns a.
      if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
        return ValueError('partial fn does not work correctly')
      return a

    wrapped_fn = functools.partial(fn, test_arg2=456)
    double_wrapped_fn = functools.partial(wrapped_fn, 123)

    self.assertEqual(('a',), function_utils.fn_args(double_wrapped_fn))

    self.assertEqual(3, double_wrapped_fn(3))  # pylint: disable=no-value-for-parameter
    self.assertEqual(3, double_wrapped_fn(a=3))  # pylint: disable=no-value-for-parameter

  def test_double_partial_with_positional_args_in_both_layers(self):
    # [FUNCTION DOC]
    # Scenario: both partial layers bind parameters positionally (first and second parameter).
    # Asserts fn_args returns only a, and calling the double partial positionally or by keyword
    # returns the given value (three).
    expected_test_arg1 = 123
    expected_test_arg2 = 456

    def fn(test_arg1, test_arg2, a):
      # [FUNCTION DOC]
      # HELPER: sample function; returns an error object if either bound value differs from its
      # expected value, otherwise returns a.
      if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
        return ValueError('partial fn does not work correctly')
      return a

    wrapped_fn = functools.partial(fn, 123)  # binds to test_arg1
    double_wrapped_fn = functools.partial(wrapped_fn, 456)  # binds to test_arg2

    self.assertEqual(('a',), function_utils.fn_args(double_wrapped_fn))

    self.assertEqual(3, double_wrapped_fn(3))  # pylint: disable=no-value-for-parameter
    self.assertEqual(3, double_wrapped_fn(a=3))  # pylint: disable=no-value-for-parameter


class HasKwargsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for function_utils.has_kwargs on lambdas, callable instances, bound methods, partial
  # objects, and non-callable input.

  def test_simple_function(self):
    # [FUNCTION DOC]
    # Scenario: two anonymous functions, one with a variadic keyword parameter and one without.
    # Asserts has_kwargs returns true for the first and false for the second.

    fn_has_kwargs = lambda **x: x
    self.assertTrue(function_utils.has_kwargs(fn_has_kwargs))

    fn_has_no_kwargs = lambda x: x
    self.assertFalse(function_utils.has_kwargs(fn_has_no_kwargs))

  def test_callable(self):
    # [FUNCTION DOC]
    # Scenario: two callable-instance fixtures, whose call method has / lacks a keyword catch-all.
    # Asserts has_kwargs returns true for the first and false for the second.

    class FooHasKwargs(object):
      # [CLASS DOC]
      # HELPER: fixture whose call method accepts arbitrary keyword arguments.

      def __call__(self, **x):
        # [FUNCTION DOC]
        # HELPER: accepts and discards keyword arguments; returns nothing.
        del x
    self.assertTrue(function_utils.has_kwargs(FooHasKwargs()))

    class FooHasNoKwargs(object):
      # [CLASS DOC]
      # HELPER: fixture whose call method accepts one named argument and no keyword catch-all.

      def __call__(self, x):
        # [FUNCTION DOC]
        # HELPER: accepts and discards one argument; returns nothing.
        del x
    self.assertFalse(function_utils.has_kwargs(FooHasNoKwargs()))

  def test_bound_method(self):
    # [FUNCTION DOC]
    # Scenario: bound methods with and without a keyword catch-all.
    # Asserts has_kwargs returns true for the first and false for the second.

    class FooHasKwargs(object):
      # [CLASS DOC]
      # HELPER: fixture with a method accepting arbitrary keyword arguments.

      def fn(self, **x):
        # [FUNCTION DOC]
        # HELPER: accepts and discards keyword arguments; returns nothing.
        del x
    self.assertTrue(function_utils.has_kwargs(FooHasKwargs().fn))

    class FooHasNoKwargs(object):
      # [CLASS DOC]
      # HELPER: fixture with a method accepting one named argument and no catch-all.

      def fn(self, x):
        # [FUNCTION DOC]
        # HELPER: accepts and discards one argument; returns nothing.
        del x
    self.assertFalse(function_utils.has_kwargs(FooHasNoKwargs().fn))

  def test_partial_function(self):
    # [FUNCTION DOC]
    # Scenario: partials (keyword-bound) over a function with and without a keyword catch-all.
    # Asserts has_kwargs is true / false respectively, and that calling the partials still works
    # (extra keywords come back for the first; the plain argument comes back for the second).
    expected_test_arg = 123

    def fn_has_kwargs(test_arg, **x):
      # [FUNCTION DOC]
      # HELPER: returns an error object if test_arg is not the expected value (123); otherwise
      # returns the extra keyword arguments as a mapping.
      if test_arg != expected_test_arg:
        return ValueError('partial fn does not work correctly')
      return x

    wrapped_fn = functools.partial(fn_has_kwargs, test_arg=123)
    self.assertTrue(function_utils.has_kwargs(wrapped_fn))
    some_kwargs = dict(x=1, y=2, z=3)
    self.assertEqual(wrapped_fn(**some_kwargs), some_kwargs)

    def fn_has_no_kwargs(x, test_arg):
      # [FUNCTION DOC]
      # HELPER: returns an error object if test_arg is not the expected value (123); otherwise
      # returns x.
      if test_arg != expected_test_arg:
        return ValueError('partial fn does not work correctly')
      return x

    wrapped_fn = functools.partial(fn_has_no_kwargs, test_arg=123)
    self.assertFalse(function_utils.has_kwargs(wrapped_fn))
    some_arg = 1
    self.assertEqual(wrapped_fn(some_arg), some_arg)

  def test_double_partial(self):
    # [FUNCTION DOC]
    # Scenario: double partials (both keyword-bound) over a function with and without a keyword
    # catch-all.
    # Asserts has_kwargs is true / false respectively and the double partials remain callable.
    expected_test_arg1 = 123
    expected_test_arg2 = 456

    def fn_has_kwargs(test_arg1, test_arg2, **x):
      # [FUNCTION DOC]
      # HELPER: returns an error object if either bound value differs from expected; otherwise
      # returns the extra keyword arguments as a mapping.
      if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
        return ValueError('partial does not work correctly')
      return x

    wrapped_fn = functools.partial(fn_has_kwargs, test_arg2=456)
    double_wrapped_fn = functools.partial(wrapped_fn, test_arg1=123)

    self.assertTrue(function_utils.has_kwargs(double_wrapped_fn))
    some_kwargs = dict(x=1, y=2, z=3)
    self.assertEqual(double_wrapped_fn(**some_kwargs), some_kwargs)

    def fn_has_no_kwargs(x, test_arg1, test_arg2):
      # [FUNCTION DOC]
      # HELPER: returns an error object if either bound value differs from expected; otherwise
      # returns x.
      if test_arg1 != expected_test_arg1 or test_arg2 != expected_test_arg2:
        return ValueError('partial does not work correctly')
      return x

    wrapped_fn = functools.partial(fn_has_no_kwargs, test_arg2=456)
    double_wrapped_fn = functools.partial(wrapped_fn, test_arg1=123)

    self.assertFalse(function_utils.has_kwargs(double_wrapped_fn))
    some_arg = 1
    self.assertEqual(double_wrapped_fn(some_arg), some_arg)  # pylint: disable=no-value-for-parameter

  def test_raises_type_error(self):
    # [FUNCTION DOC]
    # Scenario: call has_kwargs with a string, which is not callable.
    # Asserts a type error whose message says the argument should be a callable.
    with self.assertRaisesRegex(TypeError,
                                'should be a callable'):
      function_utils.has_kwargs('not a function')


class GetFuncNameTest(test.TestCase):
  # [CLASS DOC]
  # Tests for function_utils.get_func_name on functions, methods, callable instances, partials,
  # lambdas, and non-callable input.

  def testWithSimpleFunction(self):
    # [FUNCTION DOC]
    # Scenario: a module-level function. Asserts the returned name equals the function name.
    self.assertEqual(
        'silly_example_function',
        function_utils.get_func_name(silly_example_function))

  def testWithClassMethod(self):
    # [FUNCTION DOC]
    # Scenario: a bound test method. Asserts the name is the class name and method name joined
    # with a dot.
    self.assertEqual(
        'GetFuncNameTest.testWithClassMethod',
        function_utils.get_func_name(self.testWithClassMethod))

  def testWithCallableClass(self):
    # [FUNCTION DOC]
    # Scenario: an instance of the callable fixture class. Asserts the name is the string form
    # of its type (matching a pattern containing the class name).
    callable_instance = SillyCallableClass()
    self.assertRegex(
        function_utils.get_func_name(callable_instance),
        '<.*SillyCallableClass.*>')

  def testWithFunctoolsPartial(self):
    # [FUNCTION DOC]
    # Scenario: a partial object. Asserts the name is the string form of the partial type.
    partial = functools.partial(silly_example_function)
    self.assertRegex(
        function_utils.get_func_name(partial), '<.*functools.partial.*>')

  def testWithLambda(self):
    # [FUNCTION DOC]
    # Scenario: an anonymous function. Asserts the name is the lambda placeholder name.
    anon_fn = lambda x: x
    self.assertEqual('<lambda>', function_utils.get_func_name(anon_fn))

  def testRaisesWithNonCallableObject(self):
    # [FUNCTION DOC]
    # Scenario: pass the null value. Asserts a value error is raised.
    with self.assertRaises(ValueError):
      function_utils.get_func_name(None)


class GetFuncCodeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for function_utils.get_func_code on functions, methods, callable instances, lambdas,
  # partials, and non-callable input.

  def testWithSimpleFunction(self):
    # [FUNCTION DOC]
    # Scenario: a module-level function. Asserts a code object is returned whose file name
    # refers to this test file.
    code = function_utils.get_func_code(silly_example_function)
    self.assertIsNotNone(code)
    self.assertRegex(code.co_filename, 'function_utils_test.py')

  def testWithClassMethod(self):
    # [FUNCTION DOC]
    # Scenario: a bound test method. Asserts a code object is returned whose file name refers
    # to this test file.
    code = function_utils.get_func_code(self.testWithClassMethod)
    self.assertIsNotNone(code)
    self.assertRegex(code.co_filename, 'function_utils_test.py')

  def testWithCallableClass(self):
    # [FUNCTION DOC]
    # Scenario: an instance of the callable fixture class. Asserts the code object of its call
    # method is returned and refers to this test file.
    callable_instance = SillyCallableClass()
    code = function_utils.get_func_code(callable_instance)
    self.assertIsNotNone(code)
    self.assertRegex(code.co_filename, 'function_utils_test.py')

  def testWithLambda(self):
    # [FUNCTION DOC]
    # Scenario: an anonymous function. Asserts a code object is returned that refers to this
    # test file.
    anon_fn = lambda x: x
    code = function_utils.get_func_code(anon_fn)
    self.assertIsNotNone(code)
    self.assertRegex(code.co_filename, 'function_utils_test.py')

  def testWithFunctoolsPartial(self):
    # [FUNCTION DOC]
    # Scenario: a partial object. Asserts that no code object is available (null result).
    partial = functools.partial(silly_example_function)
    code = function_utils.get_func_code(partial)
    self.assertIsNone(code)

  def testRaisesWithNonCallableObject(self):
    # [FUNCTION DOC]
    # Scenario: pass the null value. Asserts a value error is raised.
    with self.assertRaises(ValueError):
      function_utils.get_func_code(None)


if __name__ == '__main__':
  test.main()
