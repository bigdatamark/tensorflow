# [MODULE DOC]
# Unit tests for the AutoGraph api module (converted_call, convert, do_not_convert, to_graph,
# to_code, tf_convert, control-status context handling, error propagation and caching). Defines a
# single test case class plus two small fixtures. When run as a script, sets the strict-conversion
# environment variable to one and starts the test runner.
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
"""Tests for api module."""

import abc
import collections
import contextlib
import functools
import gc
import inspect
import io
import os
import re
import sys
import textwrap
import types

import numpy as np

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.impl import api
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import _errors_test_helper
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors as tf_errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

global_n = 2

DEFAULT_RECURSIVE = converter.ConversionOptions(recursive=True)


class TestResource:
  # [CLASS DOC]
  # HELPER: tiny marker object used by the memory-leak tests. It holds one integer attribute and
  # tests check that no instances of it survive garbage collection after a converted call.

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: sets the single attribute x to the integer three.
    self.x = 3


class ApiTest(test.TestCase):
  # [CLASS DOC]
  # Unit tests for the AutoGraph api module: converted_call on many kinds of callables
  # (functions, methods, builtins, partials, callable objects, namedtuples, native bindings),
  # the convert and do_not_convert decorators, to_graph and to_code, control-status context
  # tracking, tf_convert, super handling, error propagation, and allowlist caching. Provides two
  # shared helpers: an output-capture context manager and a memory-leak checker.

  @contextlib.contextmanager
  def assertPrints(self, expected, not_expected):
    # [FUNCTION DOC]
    # HELPER: context manager assertion for tests. Redirects standard output to an in-memory
    # buffer for the duration of the with-block; on normal exit asserts the captured text contains
    # the expected string and does not contain the not_expected string. Always restores the real
    # standard output afterwards. Inputs: expected and not_expected (strings). Yields nothing.
    try:
      out_capturer = io.StringIO()
      sys.stdout = out_capturer
      yield
      self.assertIn(expected, out_capturer.getvalue())
      self.assertNotIn(not_expected, out_capturer.getvalue())
    finally:
      sys.stdout = sys.__stdout__

  def assertNoMemoryLeaks(self, f):
    # [FUNCTION DOC]
    # HELPER: records the identities of all live objects, calls the zero-argument function f,
    # forces garbage collection, then asserts that no new TestResource instance remains alive.
    # Input: f (callable taking nothing). Fails the test if a leak is found.
    object_ids_before = {id(o) for o in gc.get_objects()}
    f()
    gc.collect()
    objects_after = tuple(
        o for o in gc.get_objects() if id(o) not in object_ids_before)
    self.assertEmpty(
        tuple(o for o in objects_after if isinstance(o, TestResource)))

  def test_converted_call_kwonly_args(self):
    # [FUNCTION DOC]
    # Scenario: a function with a keyword-only parameter is called through converted_call with the
    # argument given in the keyword mapping. Expects the result to equal the passed constant, -1.

    def test_fn(*, a):
      # [FUNCTION DOC]
      # HELPER: returns its keyword-only argument unchanged.
      return a

    x = api.converted_call(
        test_fn, (), {'a': constant_op.constant(-1)}, options=DEFAULT_RECURSIVE)
    self.assertEqual(-1, self.evaluate(x))

  def test_super_with_no_arg(self):
    # [FUNCTION DOC]
    # Scenario: a subclass method calls the zero-argument form of super to reach a base-class
    # method, and the class is instantiated through converted_call. Expects the base-class method
    # to be called (result 5 for input 2), never the overriding subclass method.
    test_case_self = self

    class TestBase:
      # [CLASS DOC]
      # HELPER: base class whose method adds three to its argument.

      def plus_three(self, x):
        # [FUNCTION DOC]
        # HELPER: returns the argument plus three.
        return x + 3

    class TestSubclass(TestBase):
      # [CLASS DOC]
      # HELPER: subclass that overrides the method with a failing version and reaches the base
      # method through zero-argument super.

      def plus_three(self, x):
        # [FUNCTION DOC]
        # HELPER: override that fails the test if ever called.
        test_case_self.fail('This should never be called.')

      def no_arg(self, x):
        # [FUNCTION DOC]
        # HELPER: calls the base-class method on the argument via zero-argument super.
        return super().plus_three(x)

    tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)

    self.assertEqual(5, tc.no_arg(2))

  def test_converted_call_avoids_triggering_operators(self):
    # [FUNCTION DOC]
    # Scenario: a callable namedtuple subclass whose equality operator fails the test is passed to
    # converted_call. Expects the call to work without ever invoking equality comparison (the
    # sum of the two tensor fields is evaluated; the final assertion is weak).

    test_self = self

    class Pair(collections.namedtuple('Pair', ['a', 'b'])):
      # [CLASS DOC]
      # HELPER: namedtuple subclass with two fields that is callable (returns their sum) and
      # whose equality operator fails the test if triggered.

      def __call__(self):
        # [FUNCTION DOC]
        # HELPER: returns the sum of fields a and b.
        return self.a + self.b

      def __eq__(self, other):
        # [FUNCTION DOC]
        # HELPER: fails the test immediately; it must never be invoked.
        test_self.fail('Triggered operator')

    p = Pair(constant_op.constant(1), constant_op.constant(2))

    x = api.converted_call(p, (), {}, options=DEFAULT_RECURSIVE)
    self.assertIsNotNone(self.evaluate(x), 3)

  @test_util.run_deprecated_v1
  def test_decorator_recursive(self):
    # [FUNCTION DOC]
    # Scenario: a method decorated with the recursive convert decorator loops while the sum of a
    # tensor exceeds a limit, dividing by the result of a helper method that takes absolute
    # values. Expects the helper to be converted too, giving [0, 1] for inputs [2, 4], 1, -2.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class with a data-dependent loop method and a helper method.

      def called_member(self, a):
        # [FUNCTION DOC]
        # HELPER: returns the absolute value of a using a conditional on a tensor.
        if a < 0:
          a = -a
        return a

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        # [FUNCTION DOC]
        # HELPER: converted method; repeatedly floor-divides x by the helper's result while the
        # sum of x is greater than s; returns x.
        while math_ops.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    x = tc.test_method(
        constant_op.constant([2, 4]), constant_op.constant(1),
        constant_op.constant(-2))
    self.assertListEqual([0, 1], self.evaluate(x).tolist())

  @test_util.run_deprecated_v1
  def test_decorator_not_recursive(self):
    # [FUNCTION DOC]
    # Scenario: same loop as the recursive test but the convert decorator is non-recursive and the
    # helper uses a plain negation op. Expects the result [0, 1] for inputs [2, 4], 1, -2.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class with a loop method and a negating helper method.

      def called_member(self, a):
        # [FUNCTION DOC]
        # HELPER: returns the numeric negation of a.
        return math_ops.negative(a)

      @api.convert(recursive=False)
      def test_method(self, x, s, a):
        # [FUNCTION DOC]
        # HELPER: converted (non-recursive) method; divides x by the helper's result while the sum
        # of x exceeds s; returns x.
        while math_ops.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    x = tc.test_method(
        constant_op.constant([2, 4]), constant_op.constant(1),
        constant_op.constant(-2))
    self.assertListEqual([0, 1], self.evaluate(x).tolist())

  @test_util.run_deprecated_v1
  def test_convert_then_do_not_convert(self):
    # [FUNCTION DOC]
    # Scenario: a recursively converted method calls a helper that is marked do-not-convert.
    # Expects the loop to still work and produce (0, 1) for inputs (2, 4), 1, -2.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class with a do-not-convert helper and a converted loop method.

      @api.do_not_convert
      def called_member(self, a):
        # [FUNCTION DOC]
        # HELPER: unconverted helper returning the numeric negation of a.
        return math_ops.negative(a)

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        # [FUNCTION DOC]
        # HELPER: converted method; divides x by the helper's result while the sum of x exceeds s.
        while math_ops.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    x = tc.test_method(
        constant_op.constant((2, 4)), constant_op.constant(1),
        constant_op.constant(-2))
    self.assertAllEqual((0, 1), self.evaluate(x))

  @test_util.run_deprecated_v1
  def test_decorator_calls_decorated(self):
    # [FUNCTION DOC]
    # Scenario: a converted method calls a helper that itself carries the convert decorator.
    # Expects it to work and give [0, 1] for inputs [2, 4], 1, -2.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class with a convert-decorated helper and a converted loop method.

      @api.convert()
      def called_member(self, a):
        # [FUNCTION DOC]
        # HELPER: decorated helper returning the absolute value of a.
        if a < 0:
          a = -a
        return a

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        # [FUNCTION DOC]
        # HELPER: converted method; divides x by the helper's result while the sum of x exceeds s.
        while math_ops.reduce_sum(x) > s:
          x //= self.called_member(a)
        return x

    tc = TestClass()
    x = tc.test_method(
        constant_op.constant([2, 4]), constant_op.constant(1),
        constant_op.constant(-2))
    self.assertListEqual([0, 1], self.evaluate(x).tolist())

  def test_decorator_preserves_argspec(self):
    # [FUNCTION DOC]
    # Scenario: a method is wrapped by the convert decorator and attached to the class as a second
    # attribute. Expects the wrapped method to report the same full argument specification as the
    # original.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture with a method and a convert-decorated alias of it.

      def test_method(self, a):
        # [FUNCTION DOC]
        # HELPER: returns the absolute value of a.
        if a < 0:
          a = -a
        return a

      test_method_converted = api.convert()(test_method)

    tc = TestClass()
    self.assertListEqual(
        list(tf_inspect.getfullargspec(tc.test_method)),
        list(tf_inspect.getfullargspec(tc.test_method_converted)))

  def test_do_not_convert_argspec(self):
    # [FUNCTION DOC]
    # Scenario: a method is wrapped with do-not-convert and exposed as a class attribute. Expects
    # the attribute to still be a method, and its reported argument names to be empty (the wrapper
    # generic signature, since the wrapped function is not generated).

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture with a method and a do-not-convert alias of it.

      def test_method(self, x, y):
        # [FUNCTION DOC]
        # HELPER: returns the sum of x and y.
        z = x + y
        return z

      test_method_allowlisted = api.do_not_convert(test_method)

    tc = TestClass()
    self.assertTrue(tf_inspect.ismethod(tc.test_method_allowlisted))
    # Because the wrapped function is not generated, we can't preserve its
    # arg spec.
    self.assertEqual((),
                     tuple(function_utils.fn_args(tc.test_method_allowlisted)))

  def test_do_not_convert_callable_object(self):
    # [FUNCTION DOC]
    # Scenario: do-not-convert is applied to a callable object (not a function). Expects the
    # wrapped object to be callable and return 1.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture callable class returning the constant one.

      def __call__(self):
        # [FUNCTION DOC]
        # HELPER: returns the integer one.
        return 1

    tc = TestClass()
    self.assertEqual(1, api.do_not_convert(tc)())

  @test_util.run_deprecated_v1
  def test_convert_call_site_decorator(self):
    # [FUNCTION DOC]
    # Scenario: a converted method explicitly calls its helper through converted_call (as
    # generated code would). Expects the same loop result [0, 1] for inputs [2, 4], 1, -2.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class with an absolute-value helper and a loop method that calls the
      # helper through converted_call.

      def called_member(self, a):
        # [FUNCTION DOC]
        # HELPER: returns the absolute value of a.
        if a < 0:
          a = -a
        return a

      @api.convert(recursive=True)
      def test_method(self, x, s, a):
        # [FUNCTION DOC]
        # HELPER: converted method; divides x by the helper's result (called via converted_call)
        # while the sum of x exceeds s; returns x.
        while math_ops.reduce_sum(x) > s:
          x //= api.converted_call(
              self.called_member, (a,), None, options=DEFAULT_RECURSIVE)
        return x

    tc = TestClass()
    x = tc.test_method(
        constant_op.constant([2, 4]), constant_op.constant(1),
        constant_op.constant(-2))
    self.assertListEqual([0, 1], self.evaluate(x).tolist())

  def test_converted_call_builtin(self):
    # [FUNCTION DOC]
    # Scenario: converted_call is applied to the range builtin and to a regular-expression compile
    # function. Expects range(3) to yield 0, 1, 2 and the compiled pattern to match a sample
    # weight name.
    x = api.converted_call(range, (3,), None, options=DEFAULT_RECURSIVE)
    self.assertEqual((0, 1, 2), tuple(x))

    x = api.converted_call(
        re.compile, ('mnas_v4_a.*\\/.*(weights|kernel):0$',),
        None,
        options=DEFAULT_RECURSIVE)
    self.assertIsNotNone(x.match('mnas_v4_a/weights:0'))

  def test_converted_call_function(self):
    # [FUNCTION DOC]
    # Scenario: a plain function with a tensor-dependent conditional is called via converted_call
    # with -1. Expects the result 1 (absolute value).

    def test_fn(x):
      # [FUNCTION DOC]
      # HELPER: returns the absolute value of x.
      if x < 0:
        return -x
      return x

    x = api.converted_call(
        test_fn, (constant_op.constant(-1),), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  @test_util.run_v1_only('b/120545219')
  def test_converted_call_functools_partial(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on a partial object (and on a partial of a partial) binding the
    # first positional argument and a keyword. Expects the negated values (1, 2, 3). Runs only in
    # version-1 graph mode.

    def test_fn(x, y, z):
      # [FUNCTION DOC]
      # HELPER: returns the three arguments, negated if x is negative.
      if x < 0:
        return -x, -y, -z
      return x, y, z

    x = api.converted_call(
        functools.partial(test_fn, constant_op.constant(-1), z=-3),
        (constant_op.constant(-2),),
        None,
        options=DEFAULT_RECURSIVE)
    self.assertEqual((1, 2, 3), self.evaluate(x))

    x = api.converted_call(
        functools.partial(
            functools.partial(test_fn, constant_op.constant(-1)), z=-3),
        (constant_op.constant(-2),),
        None,
        options=DEFAULT_RECURSIVE)
    self.assertEqual((1, 2, 3), self.evaluate(x))

  @test_util.run_v1_only('b/120545219')
  def test_converted_call_functools_partial_kwarg_mutation(self):
    # [FUNCTION DOC]
    # Scenario: the same partial is called twice, first supplying y as a keyword and then as a
    # positional argument. Expects (1, 2, 3) then (1, 4, 3), proving the partial's stored keyword
    # mapping is not mutated by the first call. Runs only in version-1 graph mode.

    def test_fn(x, y, z):
      # [FUNCTION DOC]
      # HELPER: returns the three arguments, negated if x is negative.
      if x < 0:
        return -x, -y, -z
      return x, y, z

    partial_fn = functools.partial(test_fn, constant_op.constant(-1), z=-3)
    # Call using kwargs to assign y first to ensure that partial_fn.keywords is
    # not mutated for subsequent calls (where y is assign through args).
    x = api.converted_call(
        partial_fn,
        args=(),
        kwargs={
            'y': constant_op.constant(-2),
        },
        options=DEFAULT_RECURSIVE)
    self.assertEqual((1, 2, 3), self.evaluate(x))

    x = api.converted_call(
        partial_fn,
        args=(constant_op.constant(-4),),
        kwargs=None,
        options=DEFAULT_RECURSIVE)
    self.assertEqual((1, 4, 3), self.evaluate(x))

  def test_converted_call_method(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on a bound method of an instance holding a tensor -1. Expects the
    # method's absolute-value result, 1.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class storing a value and returning its absolute value from a method.

      def __init__(self, x):
        # [FUNCTION DOC]
        # HELPER: stores x as an attribute.
        self.x = x

      def test_method(self):
        # [FUNCTION DOC]
        # HELPER: returns the absolute value of the stored attribute.
        if self.x < 0:
          return -self.x
        return self.x

    tc = TestClass(constant_op.constant(-1))
    x = api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  def test_converted_call_synthetic_method(self):
    # [FUNCTION DOC]
    # Scenario: a free function is bound to an instance by constructing a method object manually,
    # then called via converted_call. Expects the absolute value 1 of the instance's stored -1.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class that only stores a value.

      def __init__(self, x):
        # [FUNCTION DOC]
        # HELPER: stores x as an attribute.
        self.x = x

    def test_function(self):
      # [FUNCTION DOC]
      # HELPER: free function taking an instance; returns the absolute value of its stored value.
      if self.x < 0:
        return -self.x
      return self.x

    tc = TestClass(constant_op.constant(-1))
    test_method = types.MethodType(test_function, tc)

    x = api.converted_call(test_method, (), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  def test_converted_call_method_wrapper(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on the descriptor-binding method-wrapper of a bound method, passing
    # the instance. Expects the result to equal the bound method itself.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class with a do-nothing method.

      def foo(self):
        # [FUNCTION DOC]
        # HELPER: does nothing.
        pass

    tc = TestClass()

    # `method.__get__()` returns a so-called method-wrapper.
    wrapper = api.converted_call(
        tc.foo.__get__, (tc,), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(wrapper, tc.foo)

  def test_converted_call_method_as_object_attribute(self):
    # [FUNCTION DOC]
    # Scenario: a bound method of one object is stored as an attribute of another object and
    # called through that attribute via converted_call. Expects the result 2.

    class AnotherClass:
      # [CLASS DOC]
      # HELPER: fixture class holding a tensor attribute and a conditional method.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets the tensor attribute to the constant one.
        self.another_class_attr = constant_op.constant(1)

      def method(self):
        # [FUNCTION DOC]
        # HELPER: returns the attribute plus one if it is positive, otherwise plus ten.
        if self.another_class_attr > 0:
          return self.another_class_attr + 1
        return self.another_class_attr + 10

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class that stores a callable passed to its constructor.

      def __init__(self, another_obj_method):
        # [FUNCTION DOC]
        # HELPER: stores the given method as an attribute.
        self.another_obj_method = another_obj_method

    obj = AnotherClass()
    tc = TestClass(obj.method)

    x = api.converted_call(
        tc.another_obj_method, (), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(self.evaluate(x), 2)

  def test_converted_call_method_converts_recursively(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on a method that calls another method of the same instance.
    # Expects the nested call to work too, returning 1 for a stored -1.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class with a method that delegates to another conditional method.

      def __init__(self, x):
        # [FUNCTION DOC]
        # HELPER: stores x as an attribute.
        self.x = x

      def other_method(self):
        # [FUNCTION DOC]
        # HELPER: returns the absolute value of the stored attribute.
        if self.x < 0:
          return -self.x
        return self.x

      def test_method(self):
        # [FUNCTION DOC]
        # HELPER: returns the result of the other method.
        return self.other_method()

    tc = TestClass(constant_op.constant(-1))
    x = api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  def test_converted_call_method_by_class(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on the unbound method taken from the class, passing the instance as
    # the first argument. Expects the absolute value 1 of the stored -1.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class storing a value and returning its absolute value from a method.

      def __init__(self, x):
        # [FUNCTION DOC]
        # HELPER: stores x as an attribute.
        self.x = x

      def test_method(self):
        # [FUNCTION DOC]
        # HELPER: returns the absolute value of the stored attribute.
        if self.x < 0:
          return -self.x
        return self.x

    tc = TestClass(constant_op.constant(-1))
    x = api.converted_call(
        TestClass.test_method, (tc,), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  def test_converted_call_callable_object(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on an instance of a class defining a call method. Expects the call
    # method to run, returning 1 for a stored -1.

    class TestClass:
      # [CLASS DOC]
      # HELPER: callable fixture class returning the absolute value of its stored value.

      def __init__(self, x):
        # [FUNCTION DOC]
        # HELPER: stores x as an attribute.
        self.x = x

      def __call__(self):
        # [FUNCTION DOC]
        # HELPER: returns the absolute value of the stored attribute.
        if self.x < 0:
          return -self.x
        return self.x

    tc = TestClass(constant_op.constant(-1))
    x = api.converted_call(tc, (), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(1, self.evaluate(x))

  def test_converted_call_callable_metaclass(self):
    # [FUNCTION DOC]
    # Scenario: a class built from a metaclass that overrides the call operation is instantiated
    # through converted_call. Expects the metaclass call to run converted code (inside generated
    # code), and the result to be an instance of the class; the instance-level call attribute must
    # not be used.

    test_self = self

    class TestMetaclass(type):
      # [CLASS DOC]
      # HELPER: metaclass whose call operation creates instances manually and attaches a failing
      # instance-level call attribute.

      def __call__(cls):  # pylint: disable=method-hidden
        # [FUNCTION DOC]
        # HELPER: asserts it runs inside generated code, creates the instance, runs its
        # initializer, sets an instance call attribute that fails the test if used, and returns
        # the instance.
        self.assertTrue(converter_testing.is_inside_generated_code())
        inst = object.__new__(cls)
        inst.__init__()

        def instance_call(unused_self):
          # [FUNCTION DOC]
          # HELPER: fails the test; the class-bound call must be used instead.
          test_self.fail(
              'The class-bound __call__ should be called, not the instance'
              ' bound one.')

        inst.__call__ = instance_call
        return inst

    tmc = TestMetaclass('TestClass', (), {})
    tc = api.converted_call(tmc, (), None, options=DEFAULT_RECURSIVE)
    self.assertIsInstance(tc, tmc)

  def test_converted_call_callable_abc(self):
    # [FUNCTION DOC]
    # Scenario: a subclass of an abstract base class defining an abstract call method is
    # instantiated and then called through converted_call. Expects the constructor to run
    # unconverted, and the concrete call method to run converted (inside generated code).

    test_self = self

    class TestBase(metaclass=abc.ABCMeta):
      # [CLASS DOC]
      # HELPER: abstract base class declaring an abstract call method.

      @abc.abstractmethod
      def __call__(self):
        # [FUNCTION DOC]
        # HELPER: abstract call method that fails the test if invoked.
        test_self.fail('This should not be called')

    class TestSubclass(TestBase):
      # [CLASS DOC]
      # HELPER: concrete subclass whose constructor and call method assert whether they run
      # inside generated code.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: asserts the constructor is not running inside generated code.
        test_self.assertFalse(converter_testing.is_inside_generated_code())

      def __call__(self, expected):
        # [FUNCTION DOC]
        # HELPER: asserts expected is truthy and that this call runs inside generated code.
        test_self.assertTrue(expected)
        test_self.assertTrue(converter_testing.is_inside_generated_code())

    tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)
    api.converted_call(tc, (True,), None, options=DEFAULT_RECURSIVE)

  @test_util.run_deprecated_v1
  def test_converted_call_constructor(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on a class (its constructor). Expects the constructor to run
    # unconverted (not inside generated code) and the result to be an instance of the class.

    test_self = self

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class whose constructor asserts it is not run as generated code.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: asserts the constructor is not running inside generated code.
        test_self.assertFalse(converter_testing.is_inside_generated_code())

    tc = api.converted_call(TestClass, (), None, options=DEFAULT_RECURSIVE)
    self.assertIsInstance(tc, TestClass)

  def test_converted_call_mangled_properties(self):
    # [FUNCTION DOC]
    # Scenario: a method reads a name-mangled private attribute, which AutoGraph does not support.
    # With strict conversion mode on, expects an unsupported-language-element error mentioning
    # mangled names. Then, with verbosity zero and strict mode off, expects a warning containing
    # "could not transform" but not the bug-report text printed. Side effects: changes the
    # logging verbosity and the strict-conversion environment variable (leaves it set to one).

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class using a name-mangled private attribute.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets the private attribute to the constant -1.
        self.__private = constant_op.constant(-1)

      def test_method(self):
        # [FUNCTION DOC]
        # HELPER: returns the private attribute.
        return self.__private

    tc = TestClass()
    with self.assertRaisesRegex(errors.UnsupportedLanguageElementError,
                                'mangled names'):
      api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)

    # TODO(mdan): Refactor to avoid this use of global state.
    ag_logging.set_verbosity(0, True)
    os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '0'
    with self.assertPrints('could not transform', 'bug'):
      api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
    ag_logging.set_verbosity(0, False)
    os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '1'

  def test_converted_call_partial_of_allowlisted_function(self):
    # [FUNCTION DOC]
    # Scenario: a partial wraps a function that was explicitly allowlisted for testing. Expects the
    # function to run unconverted (not inside generated code) when called via converted_call.

    def test_fn(_):
      # [FUNCTION DOC]
      # HELPER: ignores its argument and asserts it is not running inside generated code.
      self.assertFalse(converter_testing.is_inside_generated_code())

    converter_testing.allowlist(test_fn)
    api.converted_call(
        functools.partial(test_fn, None), (), None, options=DEFAULT_RECURSIVE)

  def test_converted_call_already_converted(self):
    # [FUNCTION DOC]
    # Scenario: a function is called via converted_call, then converted explicitly with to_graph
    # (all optional features) and the result is again called via converted_call. Expects both
    # calls to return a true tensor value for input zero.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: returns whether x equals zero.
      return x == 0

    x = api.converted_call(
        f, (constant_op.constant(0),), None, options=DEFAULT_RECURSIVE)
    self.assertTrue(self.evaluate(x))

    converted_f = api.to_graph(
        f, experimental_optional_features=converter.Feature.ALL)
    x = api.converted_call(
        converted_f, (constant_op.constant(0),),
        None,
        options=DEFAULT_RECURSIVE)
    self.assertTrue(self.evaluate(x))

  def test_converted_call_then_already_converted_dynamic(self):
    # [FUNCTION DOC]
    # Scenario: a function that calls its function-valued argument is converted via
    # converted_call and given an already convert-decorated absolute-value function. Expects the
    # result 1 for input 1.

    @api.convert()
    def g(x):
      # [FUNCTION DOC]
      # HELPER: decorated function returning the absolute value of x.
      if x > 0:
        return x
      else:
        return -x

    def f(g, x):
      # [FUNCTION DOC]
      # HELPER: calls g on x and returns the result.
      return g(x)

    x = api.converted_call(
        f, (g, constant_op.constant(1)), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(self.evaluate(x), 1)

  def test_converted_call_forced_when_explicitly_allowlisted(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests conversion is forced, but the assertions only check that the
    # result is truthy and that a function converted via to_graph returns 1; the test never
    # verifies that conversion actually happened.
    # Scenario: a do-not-convert function is called via converted_call with user-requested
    # options, then converted via to_graph and called with the default recursive options.
    # Expects a truthy tensor result, then the value 1.

    @api.do_not_convert()
    def f(x):
      # [FUNCTION DOC]
      # HELPER: returns x plus one.
      return x + 1

    opts = converter.ConversionOptions(recursive=True, user_requested=True)
    x = api.converted_call(f, (constant_op.constant(0),), None, options=opts)
    self.assertTrue(self.evaluate(x))

    converted_f = api.to_graph(
        f, experimental_optional_features=converter.Feature.ALL)
    x = api.converted_call(converted_f, (0,), None, options=DEFAULT_RECURSIVE)
    self.assertEqual(x, 1)

  @test_util.run_deprecated_v1
  def test_converted_call_no_user_code(self):
    # [FUNCTION DOC]
    # Scenario: options disable converting user code. A function calling the length builtin on a
    # tensor is called via converted_call. Expects it to run unconverted and raise an error saying
    # length is not well defined; calling the length builtin directly via converted_call must work
    # and give 1 (static shape). Runs in version-1 mode.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: returns the length of x.
      return len(x)

    opts = converter.ConversionOptions(internal_convert_user_code=False)

    # f should not be converted, causing len to error out.
    with self.assertRaisesRegex(Exception, 'len is not well defined'):
      api.converted_call(f, (constant_op.constant([0]),), None, options=opts)

    # len on the other hand should work fine.
    x = api.converted_call(
        len, (constant_op.constant([0]),), None, options=opts)
    # The constant has static shape so the result is a primitive not a Tensor.
    self.assertEqual(x, 1)

  def test_converted_call_no_kwargs_allowed(self):
    # [FUNCTION DOC]
    # Scenario: with user-code conversion disabled, a function forwarding to a native routine that
    # rejects any keyword arguments is called via converted_call with no keywords. Expects the
    # call to succeed and return a non-null result (i.e. an empty keyword mapping is not splatted).

    def f(*args):
      # [FUNCTION DOC]
      # HELPER: passes the first positional argument as a one-element sequence to numpy's
      # broadcast helper and returns its result.
      # Note: np.broadcast rejects any **kwargs, even *{}
      return np.broadcast(args[:1])

    opts = converter.ConversionOptions(internal_convert_user_code=False)
    self.assertIsNotNone(
        api.converted_call(f, (1, 2, 3, 4), None, options=opts))

  def test_converted_call_allowlisted_method(self):
    # [FUNCTION DOC]
    # Scenario: a method's underlying function is allowlisted for testing, then called via
    # converted_call on a bound instance. Expects it to run unconverted, so the method (which
    # reports whether it runs in generated code) returns false.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class whose method reports whether it runs inside generated code.

      def method(self):
        # [FUNCTION DOC]
        # HELPER: returns whether this code runs inside generated code.
        return converter_testing.is_inside_generated_code()

    obj = TestClass()
    converter_testing.allowlist(obj.method.__func__)

    self.assertFalse(
        api.converted_call(obj.method, (), {}, options=DEFAULT_RECURSIVE))

  def test_converted_call_allowlisted_method_via_owner(self):
    # [FUNCTION DOC]
    # Scenario: the owner class is allowlisted for testing, then a bound method is called via
    # converted_call. Expects the method to run unconverted (reports false for generated code).

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class whose method reports whether it runs inside generated code.

      def method(self):
        # [FUNCTION DOC]
        # HELPER: returns whether this code runs inside generated code.
        return converter_testing.is_inside_generated_code()

    converter_testing.allowlist(TestClass)

    obj = TestClass()
    self.assertFalse(
        api.converted_call(obj.method, (), {}, options=DEFAULT_RECURSIVE))

  def test_converted_call_numpy(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on numpy's range-array function with 5. Expects the values 0..4.

    x = api.converted_call(np.arange, (5,), None, options=DEFAULT_RECURSIVE)

    self.assertAllEqual(x, list(range(5)))

  def test_converted_call_tf_op_forced(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on the TensorFlow add op with user-requested options (so the
    # allowlist is bypassed), arguments 1 and 1. Expects the value 2.

    # TODO(mdan): Add the missing level of support to LOGICAL_EXPRESSIONS.
    opts = converter.ConversionOptions(
        user_requested=True, optional_features=None)

    x = api.converted_call(math_ops.add, (1, 1), None, options=opts)

    self.assertAllEqual(self.evaluate(x), 2)

  def test_converted_call_exec_generated_code(self):
    # [FUNCTION DOC]
    # Scenario: a function defined by executing a dedented source string inside a fresh module is
    # called via converted_call. Expects it to run as-is (dynamic code has no accessible source)
    # and return 2 for input 1.

    temp_mod = types.ModuleType('test_module')
    dynamic_code = """
      def foo(x):
        return x + 1
    """
    exec(textwrap.dedent(dynamic_code), temp_mod.__dict__)  # pylint:disable=exec-used
    opts = converter.ConversionOptions(optional_features=None)

    x = api.converted_call(temp_mod.foo, (1,), None, options=opts)

    self.assertAllEqual(x, 2)

  def test_converted_call_namedtuple(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on the namedtuple factory to create a type with two fields.
    # Expects the result to be recognized as a namedtuple type.

    x = api.converted_call(
        collections.namedtuple, ('TestNamedtuple', ('a', 'b')),
        None,
        options=DEFAULT_RECURSIVE)

    self.assertTrue(inspect_utils.isnamedtuple(x))

  def test_converted_call_namedtuple_via_collections(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: body is identical to the previous namedtuple test (both use the factory from
    # the collections module), so the name suggests a distinction that does not exist.
    # Scenario: converted_call on the namedtuple factory; expects a namedtuple type.

    x = api.converted_call(
        collections.namedtuple, ('TestNamedtuple', ('a', 'b')),
        None,
        options=DEFAULT_RECURSIVE)

    self.assertTrue(inspect_utils.isnamedtuple(x))

  def test_converted_call_namedtuple_subclass_bound_method(self):
    # [FUNCTION DOC]
    # Scenario: a bound method of a namedtuple subclass instance (fields 5 and 2) is called via
    # converted_call with the tensor [2, 4]. Expects the loop to halve until the sum is not above
    # the first field, giving [1, 2].

    class TestClass(collections.namedtuple('TestNamedtuple', ('a', 'b'))):
      # [CLASS DOC]
      # HELPER: namedtuple subclass with a data-dependent loop method.

      def test_method(self, x):
        # [FUNCTION DOC]
        # HELPER: floor-divides x by field b while the sum of x exceeds field a; returns x.
        while math_ops.reduce_sum(x) > self.a:
          x //= self.b
        return x

    obj = TestClass(5, 2)
    x = api.converted_call(
        obj.test_method, (constant_op.constant([2, 4]),),
        None,
        options=DEFAULT_RECURSIVE)

    self.assertAllEqual(self.evaluate(x), [1, 2])

  def test_converted_call_namedtuple_method(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on the dictionary-conversion method inherited from a namedtuple
    # subclass instance (5, 2). Expects a mapping with a equals 5 and b equals 2.

    class TestClass(collections.namedtuple('TestNamedtuple', ('a', 'b'))):
      # [CLASS DOC]
      # HELPER: empty namedtuple subclass with two fields.
      pass

    obj = TestClass(5, 2)
    # _asdict is a documented method of namedtuple.
    x = api.converted_call(obj._asdict, (), None, options=DEFAULT_RECURSIVE)

    self.assertDictEqual(x, {'a': 5, 'b': 2})

  def test_converted_call_namedtuple_subclass_unbound_method(self):
    # [FUNCTION DOC]
    # Scenario: same as the bound-method case, but the method is taken from the class and the
    # instance is passed as the first argument. Expects the result [1, 2].

    class TestClass(collections.namedtuple('TestNamedtuple', ('a', 'b'))):
      # [CLASS DOC]
      # HELPER: namedtuple subclass with a data-dependent loop method.

      def test_method(self, x):
        # [FUNCTION DOC]
        # HELPER: floor-divides x by field b while the sum of x exceeds field a; returns x.
        while math_ops.reduce_sum(x) > self.a:
          x //= self.b
        return x

    obj = TestClass(5, 2)
    x = api.converted_call(
        TestClass.test_method, (obj, constant_op.constant([2, 4])),
        None,
        options=DEFAULT_RECURSIVE)

    self.assertAllEqual(self.evaluate(x), [1, 2])

  def test_converted_call_lambda(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on a lambda testing equality with zero, given tensor zero, after
    # initializing global variables. Expects a true result.

    l = lambda x: x == 0

    x = api.converted_call(
        l, (constant_op.constant(0),), None, options=DEFAULT_RECURSIVE)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(True, self.evaluate(x))

  def test_converted_call_function_object_method(self):
    # [FUNCTION DOC]
    # Scenario: an instance replaces its own method with a function object created by wrapping the
    # method with the TensorFlow function decorator, then converted_call invokes it. Expects the
    # value 1.

    # pylint:disable=method-hidden
    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class whose prepare step wraps its own method in a TensorFlow function.

      def method(self):
        # [FUNCTION DOC]
        # HELPER: returns the integer one.
        return 1

      def prepare(self):
        # [FUNCTION DOC]
        # HELPER: replaces the instance attribute for the method with a TensorFlow function
        # wrapping the original bound method.
        self.method = def_function.function(self.method)

    # pylint:enable=method-hidden

    tc = TestClass()
    tc.prepare()

    x = api.converted_call(tc.method, (), None, options=DEFAULT_RECURSIVE)

    self.assertAllEqual(1, self.evaluate(x))

  def test_converted_call_native_binding(self):
    # [FUNCTION DOC]
    # Scenario: converted_call on a native numpy function (power) with arguments 2 and 2.
    # Expects the value 4.
    x = api.converted_call(np.power, (2, 2), None, options=DEFAULT_RECURSIVE)
    self.assertAllEqual(x, 4)

  def test_converted_call_native_binding_errorneous(self):
    # [FUNCTION DOC]
    # Scenario: a native numpy function is given an object whose array conversion raises a value
    # error, and the AutoGraph warning logger is replaced by one that fails the test. Expects the
    # value error (message "fault") to propagate and no warning to be logged.

    class FaultyBinding:
      # [CLASS DOC]
      # HELPER: fixture whose array conversion always raises a value error.

      def __array__(self):
        # [FUNCTION DOC]
        # HELPER: always raises a value error with the message "fault".
        raise ValueError('fault')

    bad_obj = FaultyBinding()

    def fail_if_warning(*_):
      # [FUNCTION DOC]
      # HELPER: replacement warning logger; ignores arguments and fails the test.
      self.fail('No warning should be issued')

    with test.mock.patch.object(ag_logging, 'warning', fail_if_warning):
      with self.assertRaisesRegex(ValueError, 'fault'):
        api.converted_call(
            np.power, (bad_obj, 2), None, options=DEFAULT_RECURSIVE)

  def test_converted_call_through_tf_dataset(self):
    # [FUNCTION DOC]
    # Scenario: inside a TensorFlow function, a function that builds a dataset of the range -3 to
    # 3 and maps an absolute-value function over it is called via converted_call; the first three
    # elements are read. Expects (3, 2, 1).

    def other_fn(x):
      # [FUNCTION DOC]
      # HELPER: dataset mapping function returning the absolute value of x.
      if x > 0:
        return x
      return -x

    def f():
      # [FUNCTION DOC]
      # HELPER: returns a dataset over the integer range -3 to 3 mapped with the helper.
      return dataset_ops.Dataset.range(-3, 3).map(other_fn)

    # Dataset iteration only works inside math_ops.
    @def_function.function
    def graph_fn():
      # [FUNCTION DOC]
      # HELPER: calls f via converted_call, iterates the dataset and returns its first three
      # elements.
      ds = api.converted_call(f, (), None, options=DEFAULT_RECURSIVE)
      itr = iter(ds)
      return next(itr), next(itr), next(itr)

    self.assertAllEqual(self.evaluate(graph_fn()), (3, 2, 1))

  def test_converted_call_no_leaks_via_closure(self):
    # [FUNCTION DOC]
    # Scenario: a function closing over a resource object is called via converted_call inside a
    # leak-checking helper. Expects no resource instance to remain after garbage collection.

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: creates a resource, defines a closure over it and calls the closure via
      # converted_call, then returns nothing.
      res = TestResource()

      def f(y):
        # [FUNCTION DOC]
        # HELPER: closure returning the resource value plus y.
        return res.x + y

      api.converted_call(f, (1,), None, options=DEFAULT_RECURSIVE)

    self.assertNoMemoryLeaks(test_fn)

  def test_converted_call_no_leaks_via_inner_function_closure(self):
    # [FUNCTION DOC]
    # Scenario: like the closure leak test but the converted function returns an inner function
    # closing over the resource, which is then called. Expects no resource instance to remain
    # after garbage collection.

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: creates a resource, converts and calls a function returning an inner closure, and
      # calls that closure. Returns nothing.
      res = TestResource()

      def f(y):
        # [FUNCTION DOC]
        # HELPER: returns an inner function that closes over the resource and y.

        def inner_f():
          # [FUNCTION DOC]
          # HELPER: returns the resource value plus y.
          return res.x + y

        return inner_f

      api.converted_call(f, (1,), None, options=DEFAULT_RECURSIVE)()

    self.assertNoMemoryLeaks(test_fn)

  def test_converted_call_no_caching_on_abort(self):
    # [FUNCTION DOC]
    # Scenario: a function is first called via converted_call inside a disabled AutoGraph context
    # (so conversion is skipped), then inside an enabled context where it needs conversion.
    # Expects the first call to return 3 and the second to still be converted and return 1,
    # showing the first call did not cache the function as unconverted.

    def test_fn(needs_autograph):
      # [FUNCTION DOC]
      # HELPER: returns a tensor via a tensor-dependent conditional if needs_autograph is true
      # (requires conversion), otherwise returns the plain integer 3.
      if needs_autograph:
        if constant_op.constant(True):
          x = constant_op.constant(1)
        else:
          x = constant_op.constant(2)
      else:
        x = 3
      return x

    def call_in_disabled_context():
      # [FUNCTION DOC]
      # HELPER: calls the test function with false via converted_call under a disabled context.
      with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED):
        return api.converted_call(
            test_fn, (False,), None, options=DEFAULT_RECURSIVE)

    def call_in_default_context():
      # [FUNCTION DOC]
      # HELPER: calls the test function with true via converted_call under an enabled context.
      with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED):
        return api.converted_call(
            test_fn, (True,), None, options=DEFAULT_RECURSIVE)

    # Note: this is an invariant, not a test (see above).
    assert call_in_disabled_context() == 3

    # If api.convert placed test_fn in the unconverted cache, this second
    # invocation would fail.
    self.assertEqual(self.evaluate(call_in_default_context()), 1)

  def test_converted_call_caching_of_allowlisted_bound_methods(self):
    # [FUNCTION DOC]
    # Scenario: a bound method that cannot be converted (name-mangled attribute) is called with
    # fallback enabled, which records it in the allowlist cache; then with strict mode back on, a
    # fresh instance's method is called again. Expects the cache to grow by exactly one entry
    # after the first call and not to grow after the second (which must not raise). Side effects:
    # sets the strict-conversion environment variable to zero then one.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class using a name-mangled private attribute (unconvertible).

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: sets the private attribute to the constant -1.
        self.__private = constant_op.constant(-1)

      def test_method(self):
        # [FUNCTION DOC]
        # HELPER: returns the private attribute.
        return self.__private

    # TODO(mdan): Refactor to avoid this use of global state.
    cache_size_before = len(conversion._ALLOWLIST_CACHE)

    # First invocation with fallback on, to allow recording it into cache.
    os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '0'
    tc = TestClass()
    api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
    os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '1'

    # Entry should be added to the allowlist cache.
    self.assertEqual(len(conversion._ALLOWLIST_CACHE), cache_size_before + 1)

    # A second invocation should go through even with fallback off.
    tc = TestClass()
    api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)

    # No new entries should appear in the allowlist cache.
    self.assertEqual(len(conversion._ALLOWLIST_CACHE), cache_size_before + 1)

  def test_context_tracking_direct_calls(self):
    # [FUNCTION DOC]
    # Scenario: functions decorated with do-not-convert, convert and the unspecified-status
    # decorator are called directly. Expects the control status to be disabled inside the first,
    # enabled inside the second (and still enabled after it calls the first), unspecified
    # outside any, and unspecified inside the third.

    @api.do_not_convert()
    def unconverted_fn():
      # [FUNCTION DOC]
      # HELPER: asserts the control status is disabled.
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.DISABLED)

    @api.convert()
    def converted_fn():
      # [FUNCTION DOC]
      # HELPER: asserts status is enabled, calls the unconverted function, asserts status is
      # enabled again.
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.ENABLED)
      unconverted_fn()
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.ENABLED)

    self.assertEqual(ag_ctx.control_status_ctx().status,
                     ag_ctx.Status.UNSPECIFIED)
    converted_fn()
    self.assertEqual(ag_ctx.control_status_ctx().status,
                     ag_ctx.Status.UNSPECIFIED)

    @api.call_with_unspecified_conversion_status
    def unspecified_fn():
      # [FUNCTION DOC]
      # HELPER: asserts the control status is unspecified.
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.UNSPECIFIED)

    unspecified_fn()

  def test_to_graph_basic(self):
    # [FUNCTION DOC]
    # Scenario: a loop function is converted with to_graph and called inside a fresh graph with
    # [4, 8] and limit 4. Expects (1, 2).

    def test_fn(x, s):
      # [FUNCTION DOC]
      # HELPER: halves x (floor division) while the sum of x exceeds s; returns x.
      while math_ops.reduce_sum(x) > s:
        x //= 2
      return x

    compiled_fn = api.to_graph(test_fn)

    with ops.Graph().as_default():
      x = compiled_fn(constant_op.constant((4, 8)), 4)
      self.assertAllEqual(self.evaluate(x), (1, 2))

  @test_util.run_deprecated_v1
  def test_to_graph_with_defaults(self):
    # [FUNCTION DOC]
    # Scenario: a function with a default argument value taken from a local variable is converted
    # with to_graph and called without that argument. Expects the default (4) to be preserved,
    # giving [1, 2] for [4, 8]. Runs in version-1 mode.

    foo = 4

    def test_fn(x, s=foo):
      # [FUNCTION DOC]
      # HELPER: halves x (floor division) while the sum of x exceeds s; returns x.
      while math_ops.reduce_sum(x) > s:
        x //= 2
      return x

    compiled_fn = api.to_graph(test_fn)

    x = compiled_fn(constant_op.constant([4, 8]))
    self.assertListEqual([1, 2], self.evaluate(x).tolist())

  def test_to_graph_with_globals(self):
    # [FUNCTION DOC]
    # Scenario: a function that updates a module-level global is converted and called with 10.
    # Expects the module's global counter to increase. Side effect: mutates the global.

    def test_fn(x):
      # [FUNCTION DOC]
      # HELPER: adds x to the module-level global and returns the new value.
      global global_n
      global_n = x + global_n
      return global_n

    converted_fn = api.to_graph(test_fn)
    prev_val = global_n
    converted_fn(10)
    self.assertGreater(global_n, prev_val)

  def test_to_graph_with_kwargs_clashing_converted_call(self):
    # [FUNCTION DOC]
    # Scenario: a converted function calls another with keyword names that coincide with the
    # parameter names of converted_call (f and owner). Expects no clash; the result is 3.

    def called_fn(**kwargs):
      # [FUNCTION DOC]
      # HELPER: returns the sum of the keyword arguments named f and owner.
      return kwargs['f'] + kwargs['owner']

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: calls the other function with f=1 and owner=2.
      # These arg names intentionally match converted_call's
      return called_fn(f=1, owner=2)

    compiled_fn = api.to_graph(test_fn)

    self.assertEqual(compiled_fn(), 3)

  def test_to_graph_with_kwargs_clashing_unconverted_call(self):
    # [FUNCTION DOC]
    # Scenario: like the previous test but the callee is marked do-not-convert, so the unconverted
    # call path is used with keyword names that match its parameter names. Expects result 3.

    @api.do_not_convert
    def called_fn(**kwargs):
      # [FUNCTION DOC]
      # HELPER: returns the sum of the keyword arguments named f and owner.
      return kwargs['f'] + kwargs['owner']

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: calls the other function with f=1 and owner=2.
      # These arg names intentionally match _call_unconverted's
      return called_fn(f=1, owner=2)

    compiled_fn = api.to_graph(test_fn)

    self.assertEqual(compiled_fn(), 3)

  def test_to_graph_caching(self):
    # [FUNCTION DOC]
    # Scenario: the same function is converted three times with to_graph. Expects all results to
    # share one generated module (not the main module) while being three distinct function
    # objects.

    def test_fn(x):
      # [FUNCTION DOC]
      # HELPER: returns the absolute value of x.
      if x > 0:
        return x
      else:
        return -x

    converted_functions = tuple(api.to_graph(test_fn) for _ in (-1, 0, 1))

    # All outputs are from the same module. We can't use __module__ because
    # that's reset when we instantiate the function (see conversion.py).
    # TODO(mdan): Can and should we overwrite __module__ instead?
    module_names = frozenset(f.ag_module for f in converted_functions)
    self.assertEqual(len(module_names), 1)
    self.assertNotIn('__main__', module_names)

    self.assertEqual(len(frozenset(id(f) for f in converted_functions)), 3)

  def test_to_graph_caching_different_options(self):
    # [FUNCTION DOC]
    # Scenario: the same function is converted with recursive on and off. Expects different
    # generated modules, and the generated source of each to show the matching recursive setting
    # in its function scope.

    def called_fn():
      # [FUNCTION DOC]
      # HELPER: does nothing.
      pass

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: calls and returns the result of the other function.
      return called_fn()

    converted_recursive = api.to_graph(test_fn, recursive=True)
    converted_non_recursive = api.to_graph(test_fn, recursive=False)

    self.assertNotEqual(converted_recursive.ag_module,
                        converted_non_recursive.ag_module)
    self.assertRegex(
        tf_inspect.getsource(converted_recursive),
        'FunctionScope(.*recursive=True.*)')
    self.assertRegex(
        tf_inspect.getsource(converted_non_recursive),
        'FunctionScope(.*recursive=False.*)')

  def test_to_graph_preserves_bindings(self):
    # [FUNCTION DOC]
    # Scenario: a converted function reads a variable from its enclosing scope; the variable is
    # reassigned afterwards. Expects the converted function to return 3 and then 7, i.e. the
    # closure binding is live, not copied.
    y = 3

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: returns the enclosing variable y.
      return y

    converted = api.to_graph(test_fn)

    self.assertEqual(converted(), 3)

    y = 7

    self.assertEqual(converted(), 7)

  def test_to_graph_source_map(self):
    # [FUNCTION DOC]
    # Scenario: a function is converted with to_graph. Expects the result to carry a source map
    # attribute.

    def test_fn(y):
      # [FUNCTION DOC]
      # HELPER: returns y squared.
      return y**2

    self.assertTrue(hasattr(api.to_graph(test_fn), 'ag_source_map'))

  def test_to_graph_sets_conversion_context(self):
    # [FUNCTION DOC]
    # Scenario: inside a TensorFlow function with AutoGraph disabled, another function is
    # converted with to_graph and called. Expects the converted function to see the control
    # status as enabled, i.e. to_graph overrides the disabled context.

    def g():
      # [FUNCTION DOC]
      # HELPER: asserts the control status is enabled; returns 0.
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.ENABLED)
      return 0

    # Note: the autograph=False sets the connect to Status.DISABLED. The test
    # verifies that to_graph overrides that.
    @def_function.function(autograph=False)
    def f():
      # [FUNCTION DOC]
      # HELPER: converts g with to_graph and calls it (inside a disabled context).
      converted_g = api.to_graph(g)
      converted_g()

    f()

  def test_to_code_basic(self):
    # [FUNCTION DOC]
    # Scenario: to_code is applied to a loop function. Expects the returned text to parse as valid
    # Python source.

    def test_fn(x, s):
      # [FUNCTION DOC]
      # HELPER: halves x (true division) while the sum of x exceeds s; returns x.
      while math_ops.reduce_sum(x) > s:
        x /= 2
      return x

    # Just check that the output is parseable Python code.
    self.assertIsNotNone(parser.parse(api.to_code(test_fn)))

  def test_to_code_with_wrapped_function(self):
    # [FUNCTION DOC]
    # Scenario: to_code is applied to a function already wrapped by the TensorFlow function
    # decorator. Expects an error whose message advises passing the underlying Python function.

    @def_function.function
    def test_fn(x, s):
      # [FUNCTION DOC]
      # HELPER: halves x (true division) while the sum of x exceeds s; returns x.
      while math_ops.reduce_sum(x) > s:
        x /= 2
      return x

    with self.assertRaisesRegex(Exception, 'try passing.*python_function'):
      api.to_code(test_fn)

  def test_tf_convert_overrides_current_context(self):
    # [FUNCTION DOC]
    # Scenario: inside a do-not-convert function, tf_convert is applied to a function using a
    # supplied control context. Expects the function to run as generated code when the supplied
    # context is enabled and not when it is disabled, regardless of the surrounding context.

    def f(expect_converted):
      # [FUNCTION DOC]
      # HELPER: asserts whether it runs in generated code matches expect_converted.
      self.assertEqual(converter_testing.is_inside_generated_code(),
                       expect_converted)

    @api.do_not_convert
    def test_fn(ctx, expect_converted):
      # [FUNCTION DOC]
      # HELPER: applies tf_convert to f with context ctx and calls it with expect_converted.
      return api.tf_convert(f, ctx)(expect_converted)

    test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED), True)
    test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED), False)

  def test_tf_convert_unspecified_not_converted_by_default(self):
    # [FUNCTION DOC]
    # Scenario: tf_convert is applied with convert_by_default false and an unspecified context,
    # inside a TensorFlow function. Expects the function to run with unspecified status and not
    # as generated code.

    def f():
      # [FUNCTION DOC]
      # HELPER: asserts the status is unspecified and that it is not running as generated code.
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.UNSPECIFIED)
      self.assertFalse(converter_testing.is_inside_generated_code())

    @def_function.function
    def test_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: applies tf_convert to f without default conversion and calls it.
      return api.tf_convert(f, ctx, convert_by_default=False)()

    test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.UNSPECIFIED))

  def test_tf_convert_allowlisted_method(self):
    # [FUNCTION DOC]
    # Scenario: a method allowlisted for testing is passed to tf_convert with an enabled context.
    # Expects the unwrapped target of the result to wrap the very same underlying function as the
    # original bound method.

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class whose method reports whether it runs inside generated code.

      def method(self):
        # [FUNCTION DOC]
        # HELPER: returns whether this code runs inside generated code.
        return converter_testing.is_inside_generated_code()

    converter_testing.allowlist(TestClass.method)

    obj = TestClass()
    converted_call = api.tf_convert(
        obj.method, ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED))
    _, converted_target = tf_decorator.unwrap(converted_call)
    self.assertIs(converted_target.__func__, obj.method.__func__)

  def test_tf_convert_tf_decorator_unwrapping_context_enabled(self):
    # [FUNCTION DOC]
    # Scenario: a function wrapped by a TensorFlow-style decorator is passed to tf_convert with an
    # enabled context and called. Expects the inner function to run as generated code.

    def f():
      # [FUNCTION DOC]
      # HELPER: asserts it runs inside generated code.
      self.assertTrue(converter_testing.is_inside_generated_code())

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: decorator wrapper forwarding all arguments to the wrapped function.
      return wrapper.__wrapped__(*args, **kwargs)

    decorated_f = tf_decorator.make_decorator(f, wrapper)

    def test_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: applies tf_convert to the decorated function with ctx and calls it.
      return api.tf_convert(decorated_f, ctx)()

    test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED))

  def test_tf_convert_tf_decorator_unwrapping_context_disabled(self):
    # [FUNCTION DOC]
    # Scenario: same as the enabled variant, but with a disabled context. Expects the inner
    # function to not run as generated code.

    def f():
      # [FUNCTION DOC]
      # HELPER: asserts it does not run inside generated code.
      self.assertFalse(converter_testing.is_inside_generated_code())

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: decorator wrapper forwarding all arguments to the wrapped function.
      return wrapper.__wrapped__(*args, **kwargs)

    decorated_f = tf_decorator.make_decorator(f, wrapper)

    def test_fn(ctx):
      # [FUNCTION DOC]
      # HELPER: applies tf_convert to the decorated function with ctx and calls it.
      return api.tf_convert(decorated_f, ctx)()

    test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED))

  def test_tf_convert_tf_decorator_allowlist_method(self):
    # [FUNCTION DOC]
    # Scenario: a method decorated with a TensorFlow-style decorator is allowlisted for testing
    # and passed to tf_convert with an enabled context. Expects both the converted callable and
    # the original method (which tf_convert intentionally modifies) to return true, i.e. report
    # running inside generated code.

    def wrap(f):
      # [FUNCTION DOC]
      # HELPER: decorator factory that wraps f in a forwarding wrapper registered through the
      # TensorFlow decorator utility; returns the decorated callable.

      def wrapper(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: forwards all arguments to the wrapped function and returns its result.
        return wrapper.__wrapped__(*args, **kwargs)

      return tf_decorator.make_decorator(f, wrapper)

    class TestClass:
      # [CLASS DOC]
      # HELPER: fixture class with a decorated method reporting generated-code status.

      @wrap
      def method(self):
        # [FUNCTION DOC]
        # HELPER: returns whether this code runs inside generated code.
        return converter_testing.is_inside_generated_code()

    converter_testing.allowlist(TestClass.method)

    obj = TestClass()
    # It's intended that tf_convert modifies the original method in this case.
    # This is not desirable, but options are limited.
    converted = api.tf_convert(
        obj.method, ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED))
    self.assertTrue(converted())
    self.assertTrue(obj.method())

  def test_super_with_one_arg(self):
    # [FUNCTION DOC]
    # Scenario: a subclass method uses the one-argument (unbound) form of super, binds it to the
    # instance and calls the base method; the class is instantiated via converted_call. Expects
    # the base method's result 5 for input 2, never the failing override.
    test_case_self = self

    class TestBase:
      # [CLASS DOC]
      # HELPER: base class whose method adds three to its argument.

      def plus_three(self, x):
        # [FUNCTION DOC]
        # HELPER: returns the argument plus three.
        return x + 3

    class TestSubclass(TestBase):
      # [CLASS DOC]
      # HELPER: subclass with a failing override and a method using one-argument super.

      def plus_three(self, x):
        # [FUNCTION DOC]
        # HELPER: override that fails the test if ever called.
        test_case_self.fail('This should never be called.')

      def one_arg(self, x):
        # [FUNCTION DOC]
        # HELPER: creates an unbound super for the subclass, binds it to this instance and calls
        # the base-class method on x.
        test_base_unbound = super(TestSubclass)
        test_base = test_base_unbound.__get__(self, TestSubclass)
        return test_base.plus_three(x)

    tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)

    self.assertEqual(5, tc.one_arg(2))

  def test_super_with_two_args(self):
    # [FUNCTION DOC]
    # Scenario: a subclass method uses the two-argument form of super (class and instance) to
    # call the base method; the class is instantiated via converted_call. Expects 5 for input 2.
    test_case_self = self

    class TestBase:
      # [CLASS DOC]
      # HELPER: base class whose method adds three to its argument.

      def plus_three(self, x):
        # [FUNCTION DOC]
        # HELPER: returns the argument plus three.
        return x + 3

    class TestSubclass(TestBase):
      # [CLASS DOC]
      # HELPER: subclass with a failing override and a method using two-argument super.

      def plus_three(self, x):
        # [FUNCTION DOC]
        # HELPER: override that fails the test if ever called.
        test_case_self.fail('This should never be called.')

      def two_args(self, x):
        # [FUNCTION DOC]
        # HELPER: calls the base-class method on x via two-argument super.
        return super(TestSubclass, self).plus_three(x)

    tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)

    self.assertEqual(5, tc.two_args(2))

  def test_raise_from_func_graph(self):
    # [FUNCTION DOC]
    # Scenario: a TensorFlow function calls a native test helper that raises an error built from
    # a status code, for each of sixteen canonical status codes. Expects the matching specific
    # TensorFlow error class each time (cancelled, unknown, invalid argument, deadline exceeded,
    # not found, already exists, permission denied, unauthenticated, resource exhausted, failed
    # precondition, aborted, out of range, unimplemented, internal, unavailable, data loss), and
    # that the error keeps both experimental payload entries (key1 to value1, key2 to value2).

    @def_function.function
    def raise_from_tf_function(n):
      # [FUNCTION DOC]
      # HELPER: raises, through the native test helper, an error for status code n.
      _errors_test_helper.TestRaiseFromStatus(n)

    for code, expected_exception in [
        (1, tf_errors.CancelledError),
        (2, tf_errors.UnknownError),
        (3, tf_errors.InvalidArgumentError),
        (4, tf_errors.DeadlineExceededError),
        (5, tf_errors.NotFoundError),
        (6, tf_errors.AlreadyExistsError),
        (7, tf_errors.PermissionDeniedError),
        (16, tf_errors.UnauthenticatedError),
        (8, tf_errors.ResourceExhaustedError),
        (9, tf_errors.FailedPreconditionError),
        (10, tf_errors.AbortedError),
        (11, tf_errors.OutOfRangeError),
        (12, tf_errors.UnimplementedError),
        (13, tf_errors.InternalError),
        (14, tf_errors.UnavailableError),
        (15, tf_errors.DataLossError),
    ]:
      with self.assertRaises(expected_exception) as error:
        raise_from_tf_function(code)
      self.assertEqual(error.exception.experimental_payloads[b'key1'],
                       b'value1')
      self.assertEqual(error.exception.experimental_payloads[b'key2'],
                       b'value2')

  def test_inspect_source_unsupported(self):
    # [FUNCTION DOC]
    # Scenario: strict mode is turned off and source lookup is made to fail with an OS error. Two
    # TensorFlow functions with a tensor-dependent conditional are called. With source inspection
    # reported as unsupported, expects an operator-not-allowed error mentioning the source may
    # not be visible and no warning logged; with it reported as supported, expects an
    # operator-not-allowed error mentioning an unsupported feature and exactly one logged warning
    # starting with the AutoGraph could-not-transform text.

    @def_function.function
    def test_func(a):
      # [FUNCTION DOC]
      # HELPER: returns a if a true tensor constant, else a plus a (needs conversion).
      if constant_op.constant(True):
        return a
      else:
        return a + a

    @def_function.function
    def test_func2(a):
      # [FUNCTION DOC]
      # HELPER: same behavior as the first function, defined separately to avoid cache effects.
      if constant_op.constant(True):
        return a
      else:
        return a + a

    patch = test.mock.patch
    with patch.dict(os.environ, {'AUTOGRAPH_STRICT_CONVERSION': '0'}), \
        patch.object(inspect, 'findsource', side_effect=OSError()), \
        self.assertLogs(level='WARNING') as logs:

      with patch.object(ag_ctx, 'INSPECT_SOURCE_SUPPORTED', False), \
          self.assertRaisesRegex(tf_errors.OperatorNotAllowedInGraphError,
                                 'source code may not be visible'):
        test_func(2)
      self.assertEmpty(logs.output)

      with patch.object(ag_ctx, 'INSPECT_SOURCE_SUPPORTED', True), \
          self.assertRaisesRegex(tf_errors.OperatorNotAllowedInGraphError,
                                 'using an unsupported feature'):
        test_func2(2)
    self.assertLen(logs.output, 1)
    self.assertRegex(logs.output[0], r'^.+:AutoGraph could not transform.+')


if __name__ == '__main__':
  os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '1'
  test.main()
