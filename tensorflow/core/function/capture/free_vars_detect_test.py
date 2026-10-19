# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Unit tests for the free-variable detection module. Three groups: one for locating the object
# that self refers to in a closure, one for the breadth-first free-variable detection over many
# function shapes (arguments, locals, builtins, subscripts, lambdas, methods, class methods,
# decorators, partials), and one for the generated logging text (thresholds, caching, input
# validation). Defines a small module-level helper to extract variable names.
# ==============================================================================
"""Tests for detecting free vars in tf.function."""

import functools
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.core.function.capture import free_vars_detect
from tensorflow.python.util import tf_decorator


def get_var_name(d):
  # [FUNCTION DOC]
  # HELPER: shared test utility. Input: d, a list of free-variable records. Returns the list of
  # their names in the same order.
  return [var.name for var in d]


class GetSelfObjFromClosureTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the private helper that recovers the self object from a nested function closure.

  def test_single_enclosing_class(self):
    # [FUNCTION DOC]
    # Defines a class whose method returns a nested function using self and a local. Calls the
    # method on an instance and asserts the helper returns exactly that instance when given the
    # nested function.

    class Foo():
      # [CLASS DOC]
      # Local fixture class: stores a value and has a method returning a closure over self.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets the attribute val to one.
        self.val = 1

      def bar(self):
        # [FUNCTION DOC]
        # Returns a nested function that captures self and a local variable.
        x = 2

        def fn():
          # [FUNCTION DOC]
          # Returns the sum of the captured self attribute and the captured local.
          return self.val + x

        return fn

    foo = Foo()
    fn = foo.bar()
    self_obj = free_vars_detect._get_self_obj_from_closure(fn)
    self.assertIs(self_obj, foo)


class FreeVarDetectionTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the free-variable detection routine across many function shapes. Each test defines
  # local functions or classes, runs the detection and asserts on the resulting mapping from
  # callable name to sorted list of free variable names.

  def test_func_arg(self):
    # [FUNCTION DOC]
    # A function whose only variable is its own argument (shadowing an outer name) has no free
    # variables; asserts the result mapping is empty.
    x = 1  # pylint: disable=unused-variable

    def f(x):
      # [FUNCTION DOC]
      # Returns the argument plus one.
      return x + 1

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertEmpty(func_map)

  def test_func_local_var(self):
    # [FUNCTION DOC]
    # A function using only a local variable has no free variables; asserts an empty result.

    def f():
      # [FUNCTION DOC]
      # Assigns a local and returns it plus one.
      x = 1
      return x + 1

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertEmpty(func_map)

  def test_global_var_int(self):
    # [FUNCTION DOC]
    # A function reading an integer from the enclosing scope: asserts the result has an entry
    # for f whose free variables are exactly x.
    x = 1

    def f():
      # [FUNCTION DOC]
      # Returns the enclosing integer plus one.
      return x + 1

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_builtin_func(self):
    # [FUNCTION DOC]
    # A function calling a builtin (length) has no free variables; asserts an empty result.

    def f(x):
      # [FUNCTION DOC]
      # Returns the length of the argument.
      return len(x)

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertEmpty(func_map)

  def test_global_var_dict(self):
    # [FUNCTION DOC]
    # A function indexing an enclosing dictionary with a literal key: asserts only the base name
    # glob is reported (the subscripted form is dropped).
    glob = {"a": 1}

    def f():
      # [FUNCTION DOC]
      # Returns the dictionary value for key a plus one.
      return glob["a"] + 1

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_global_var_dict_w_var_index(self):
    # [FUNCTION DOC]
    # A function indexing an enclosing dictionary with an enclosing key variable: asserts both
    # glob and key are reported, in sorted order.
    glob = {"a": 1}
    key = "a"

    def f():
      # [FUNCTION DOC]
      # Returns the dictionary value at the enclosing key plus one.
      return glob[key] + 1

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["glob", "key"])

  def test_duplicate_global_var(self):
    # [FUNCTION DOC]
    # A function using the same enclosing variable twice: asserts it is listed only once.
    x = 1

    def f():
      # [FUNCTION DOC]
      # Returns the enclosing variable added to itself.
      return x + x

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["x"])

  @parameterized.named_parameters(
      ("lambda_1", lambda _x: 3,), ("lambda_2", lambda _x: 3,))
  def test_multiple_lambda_w_same_line_num_and_args(self, fn):
    # [FUNCTION DOC]
    # Parameterized over two lambdas written on the same line with the same argument name, which
    # the source analysis cannot tell apart. Asserts detection tolerates this and returns an
    # empty mapping.
    func_map = free_vars_detect._detect_function_free_vars(fn)
    self.assertEmpty(func_map)

  def test_lambda_wo_free_var(self):
    # [FUNCTION DOC]
    # A lambda using only its own argument: asserts an empty result.
    f = lambda x: x + x
    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertEmpty(func_map)

  def test_lambda_w_free_var(self):
    # [FUNCTION DOC]
    # A lambda assigned to f that uses an enclosing variable: asserts the entry for f lists glob.
    glob = 1
    f = lambda x: x + glob
    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_multi_lambda_w_free_var(self):
    # [FUNCTION DOC]
    # Two lambdas (g and h) use an enclosing variable and a function f calls both. Asserts the
    # result has three entries: f lists g and h, while g and h each list glob.
    glob = 1
    g = lambda x: x + glob
    h = lambda: glob + 1

    def f(x):
      # [FUNCTION DOC]
      # Returns the sum of calling the two enclosing lambdas.
      return g(x) + h()

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertLen(func_map, 3)
    self.assertIn("f", func_map.keys())
    self.assertIn("g", func_map.keys())
    self.assertIn("h", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["g", "h"])
    free_vars = get_var_name(func_map["g"])
    self.assertSequenceEqual(free_vars, ["glob"])
    free_vars = get_var_name(func_map["h"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_lambda_inline(self):
    # [FUNCTION DOC]
    # A function returning an inline lambda that uses its argument and an enclosing variable:
    # asserts f lists only glob (the argument is not free).
    glob = 1

    def f(x):
      # [FUNCTION DOC]
      # Returns a lambda adding the argument and the enclosing variable.
      return lambda: x + glob

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_glob_numpy_var(self):
    # [FUNCTION DOC]
    # A function using an enclosing integer, an enclosing numpy array and a local numpy array
    # built via the numpy module. Asserts the free variables are a and b only (the module and
    # the local are excluded).
    a = 0
    b = np.asarray(1)

    def f():
      # [FUNCTION DOC]
      # Builds a local numpy array and returns the sum of it with the two enclosing values.
      c = np.asarray(2)
      res = a + b + c
      return res

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["a", "b"])

  def test_global_var_in_nested_func(self):
    # [FUNCTION DOC]
    # A function defining and calling an inner function that uses an enclosing variable. Asserts
    # only f is reported (one entry), with x as its free variable; the inner function is not
    # searched separately because it is not an outside free variable.
    x = 1

    def f():
      # [FUNCTION DOC]
      # Defines an inner function using the enclosing variable and returns its result.

      def g():
        # [FUNCTION DOC]
        # Returns the enclosing variable plus one.
        return x + 1

      return g()

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    self.assertLen(func_map.keys(), 1)
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_global_var_from_outer_func(self):
    # [FUNCTION DOC]
    # Function f calls a sibling function g that uses an enclosing variable. Asserts two entries:
    # f lists g, and g lists x.
    x = 1

    def g():
      # [FUNCTION DOC]
      # Returns the enclosing variable plus one.
      return x + 1

    def f():
      # [FUNCTION DOC]
      # Returns the result of calling g.
      return g()

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    self.assertIn("g", func_map.keys())
    self.assertLen(func_map.keys(), 2)
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["g"])
    free_vars = get_var_name(func_map["g"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_method(self):
    # [FUNCTION DOC]
    # A bound method reading an enclosing variable. Asserts one entry keyed by class and method
    # name, listing x.
    x = 1

    class Foo():
      # [CLASS DOC]
      # Local fixture class with one method returning an enclosing variable.

      def f(self):
        # [FUNCTION DOC]
        # Returns the enclosing variable.
        return x

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 1)
    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_method_w_method_call(self):
    # [FUNCTION DOC]
    # Method f refers to sibling method g, which uses an enclosing variable. Asserts two entries:
    # the first method lists self.g and the second lists x.
    x = 0

    class Foo():
      # [CLASS DOC]
      # Local fixture class: one method referencing another, which returns a list with x.

      def f(self):
        # [FUNCTION DOC]
        # Returns the sibling method g (not called).
        return self.g

      def g(self):
        # [FUNCTION DOC]
        # Returns a list holding the enclosing variable.
        return [x]

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["self.g"])

    self.assertIn("Foo.g", func_map.keys())
    free_vars = get_var_name(func_map["Foo.g"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_method_w_self_as_arg(self):
    # [FUNCTION DOC]
    # Method f calls sibling g passing self as an argument; g compares it and returns an
    # enclosing variable or its negation. Asserts f lists only self.g and g lists x.
    x = 1

    class Foo():
      # [CLASS DOC]
      # Local fixture class: f calls g with self; g branches on identity of its argument.

      def f(self):
        # [FUNCTION DOC]
        # Calls g with self and returns the result.
        return self.g(self)

      def g(self, obj):
        # [FUNCTION DOC]
        # Returns the enclosing variable if the argument differs from self, else its negation.
        if obj != self:
          return x
        else:
          return -x

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["self.g"])

    self.assertIn("Foo.g", func_map.keys())
    free_vars = get_var_name(func_map["Foo.g"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_self_inside_method(self):
    # [FUNCTION DOC]
    # A nested function returned from a method uses self.val and an enclosing variable. Detection
    # on that nested function should yield one entry named after it, listing self, self.val, x.
    x = 1

    class Foo():
      # [CLASS DOC]
      # Local fixture class whose method returns a nested function closing over self.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets the attribute val to two.
        self.val = 2

      def bar(self):
        # [FUNCTION DOC]
        # Returns a nested function that uses self and the enclosing variable.

        def tf_func():
          # [FUNCTION DOC]
          # Returns the sum of the self attribute and the enclosing variable.
          return self.val + x

        return tf_func

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.bar())
    self.assertLen(func_map.keys(), 1)

    self.assertIn("tf_func", func_map.keys())
    free_vars = get_var_name(func_map["tf_func"])
    self.assertSequenceEqual(free_vars, ["self", "self.val", "x"])

  def test_self_inside_function_w_multiple_closures(self):
    # [FUNCTION DOC]
    # A nested function whose closure has two cells (self of one class and an instance of
    # another). Asserts the closure length is two, then that detection yields two entries: the
    # nested function (listing baz, self, self.bar_str) and the bar_str method (listing x).
    # Test when a function contins multiple closures

    class Foo():
      # [CLASS DOC]
      # Local fixture outer class; its method builds inner classes and returns a closure.

      def method(self):
        # [FUNCTION DOC]
        # Defines helper classes, creates one instance of each and returns the closure produced
        # by the second instance's method.

        class Baz():
          # [CLASS DOC]
          # Local fixture class with a string-returning method.

          def baz_str(self):
            # [FUNCTION DOC]
            # Returns the string Baz.
            return "Baz"

        baz = Baz()
        x = "x"

        class Bar():
          # [CLASS DOC]
          # Local fixture class: a method using an enclosing string and a method returning a
          # closure over self and a Baz instance.

          def bar_str(self):
            # [FUNCTION DOC]
            # Returns the enclosing string x followed by Bar.
            return x + "Bar"

          def method(self):
            # [FUNCTION DOC]
            # Returns a nested function closing over self and the Baz instance.

            def fn():
              # [FUNCTION DOC]
              # Returns the concatenation of this object's bar string and the Baz string.
              return self.bar_str() + baz.baz_str()

            return fn

        bar = Bar()
        return bar.method()

    foo = Foo()
    fn = foo.method()
    # cells for `self.bar_str()`, `baz.baz_str()`
    self.assertLen(fn.__closure__, 2)

    func_map = free_vars_detect._detect_function_free_vars(fn)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("fn", func_map.keys())
    free_vars = get_var_name(func_map["fn"])
    self.assertSequenceEqual(free_vars, ["baz", "self", "self.bar_str"])

    self.assertIn("Bar.bar_str", func_map.keys())
    free_vars = get_var_name(func_map["Bar.bar_str"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_method_w_self_attribute(self):
    # [FUNCTION DOC]
    # Method f uses sibling method g and instance attributes x and y; g returns an enclosing
    # variable. Asserts f lists self.g, self.x, self.y and g lists x.
    x = 0

    class Foo():
      # [CLASS DOC]
      # Local fixture class with two instance attributes and two methods.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets attributes x to one and y to two.
        self.x = 1
        self.y = 2

      def f(self):
        # [FUNCTION DOC]
        # Returns an expression combining method g and both attributes.
        return self.g + self.x + self.y

      def g(self):
        # [FUNCTION DOC]
        # Returns the enclosing variable.
        return x

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["self.g", "self.x", "self.y"])

    self.assertIn("Foo.g", func_map.keys())
    free_vars = get_var_name(func_map["Foo.g"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_method_w_multiple_attributes(self):
    # [FUNCTION DOC]
    # Method f accesses a long attribute chain starting at a sibling method. Asserts only the
    # first attribute (self.g) is reported for f, and g lists the enclosing string glob.
    glob = "dummy_value"

    class Foo():
      # [CLASS DOC]
      # Local fixture class with a method using a long attribute chain.

      def f(self):
        # [FUNCTION DOC]
        # Returns a deep attribute chain starting at method g.
        return self.g.h.x.y.z

      def g(self):
        # [FUNCTION DOC]
        # Returns the enclosing string.
        return glob

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["self.g"])

    self.assertIn("Foo.g", func_map.keys())
    free_vars = get_var_name(func_map["Foo.g"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_classmethod_decorator(self):
    # [FUNCTION DOC]
    # Detection on a class method reading an enclosing variable. Asserts one entry keyed by class
    # and method name, listing glob.
    glob = 1

    class Foo():
      # [CLASS DOC]
      # Local fixture class with a class method.

      @classmethod
      def f(cls):
        # [FUNCTION DOC]
        # Returns the enclosing variable.
        return glob

    func_map = free_vars_detect._detect_function_free_vars(Foo.f)
    self.assertLen(func_map.keys(), 1)
    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_method_call_classmethod(self):
    # [FUNCTION DOC]
    # An instance method calls a class method that reads an enclosing variable. Asserts two
    # entries: the instance method lists self.g and the class method lists glob.
    glob = 1

    class Foo():
      # [CLASS DOC]
      # Local fixture class: an instance method calling a class method.

      def f(self):
        # [FUNCTION DOC]
        # Calls the class method and returns its result.
        return self.g()

      @classmethod
      def g(cls):
        # [FUNCTION DOC]
        # Returns the enclosing variable.
        return glob

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["self.g"])

    self.assertIn("Foo.g", func_map.keys())
    free_vars = get_var_name(func_map["Foo.g"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_global_var_from_renamed_outer_func(self):
    # [FUNCTION DOC]
    # Function f aliases an outer function g to a local name and calls the alias. Asserts two
    # entries: f lists g, and g lists x.
    x = 1

    def g():
      # [FUNCTION DOC]
      # Returns the enclosing variable plus one.
      return x + 1

    def f():
      # [FUNCTION DOC]
      # Assigns g to a local name and returns the result of calling it.
      h = g
      return h()

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    self.assertIn("g", func_map.keys())
    self.assertLen(func_map.keys(), 2)
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["g"])
    free_vars = get_var_name(func_map["g"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_decorated_method_w_self_no_exception(self):
    # [FUNCTION DOC]
    # Defines a decorator that, at decoration time, runs detection on the decorated method and
    # asserts exactly one entry for foo listing the decorator name. Then applies it to a method
    # that calls a sibling method using an enclosing variable (which cannot be detected). The test
    # passes if instantiating the class raises nothing.
    """Test this pattern does not raise any exceptions."""

    def dummy_tf_function(func):
      # [FUNCTION DOC]
      # Test decorator. Input: func, the decorated function. Runs the detection and asserts on
      # it, then returns a pass-through wrapper.

      func_map = free_vars_detect._detect_function_free_vars(func)
      self.assertLen(func_map, 1)
      self.assertIn("foo", func_map.keys())
      free_vars = get_var_name(func_map["foo"])
      self.assertSequenceEqual(free_vars, ["dummy_tf_function"])

      def wrapper(*args, **kwargs):
        # [FUNCTION DOC]
        # Calls the wrapped function with all given arguments and returns its result.
        return func(*args, **kwargs)

      return wrapper

    glob = 1

    # This pattern is not fully supported yet in the sense that `self.bar()` is
    # not inspected so `glob` cannot be detected.
    # The reason is the neither `self` nor `self.bar` is accessible from the
    # perspective of dummy_tf_function decorator.
    # One possible solution is parsing the source code of the whole module,
    # instead of single function. And probably get the source of `self.bar`
    # from the AST of the module where `Foo` is defined. One potentail challenge
    # of this approach is how to locate the decorated function in the AST.
    class Foo():
      # [CLASS DOC]
      # Local fixture class: a decorated method calling a plain method that uses an enclosing
      # variable.

      @dummy_tf_function
      def foo(self):
        # [FUNCTION DOC]
        # Calls the sibling method and returns its result.
        return self.bar()

      def bar(self):
        # [FUNCTION DOC]
        # Returns the enclosing variable.
        return glob

    _ = Foo()

  # Use `wrapper_first` to control different arguments order
  @parameterized.parameters(
      (functools.update_wrapper, True),
      (tf_decorator.make_decorator, False))
  def test_func_w_decorator(self, make_decorator, wrapper_first):
    # [FUNCTION DOC]
    # Parameterized over the decorator-building function (standard wrapper update or the
    # framework decorator maker) and the argument order flag. A function decorated twice contains
    # a nested function also decorated twice that uses an enclosing variable. Asserts two entries
    # and that the outer function lists the decorator name and x.
    x = 1

    def decorator_foo(func):
      # [FUNCTION DOC]
      # Test decorator. Input: func, the function to wrap. Builds a pass-through wrapper and
      # returns it linked to func via make_decorator, in an argument order set by wrapper_first.

      def wrapper(*args, **kwargs):
        # [FUNCTION DOC]
        # Calls the wrapped function with all given arguments and returns its result.
        return func(*args, **kwargs)

      if wrapper_first:
        return make_decorator(wrapper, func)
      else:
        return make_decorator(func, wrapper)

    @decorator_foo
    @decorator_foo
    def f():
      # [FUNCTION DOC]
      # Defines a doubly decorated inner function and returns its result.

      @decorator_foo
      @decorator_foo
      def g():
        # [FUNCTION DOC]
        # Returns the enclosing variable plus one.
        return x + 1

      return g()

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    self.assertLen(func_map.keys(), 2)
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["decorator_foo", "x"])

  # TODO(panzf): test the pattern when callable function args are supported
  @unittest.skip("Feature not implemented")
  def test_global_var_from_arg_func(self):
    # [FUNCTION DOC]
    # Skipped placeholder test for detecting free variables of a function passed as an argument
    # (feature not implemented). The body only defines the functions and calls f with g and
    # asserts nothing.
    x = 1

    def g():
      # [FUNCTION DOC]
      # Returns the enclosing variable plus one.
      return x + 1

    def f(h):
      # [FUNCTION DOC]
      # Calls the function argument and returns its result.
      return h()

    _ = f(g)


class GenerateLoggingTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the public logging-text generator: unsupported inputs return null, functions
  # without free variables return null, and others yield text with an explanation followed by
  # one line per function, subject to thresholds and a once-per-function cache.

  def _remove_explanation(self, logging_txt):
    # [FUNCTION DOC]
    # HELPER: strips the leading explanation from generated text. Input: logging_txt, a
    # multi-line string. Asserts it has more than two lines, then returns it without its first
    # two lines (the explanation sentence and the top function line is NOT removed: only the
    # first two lines are dropped).
    # KNOWN ISSUE: the name suggests only the explanation is removed, but two lines are dropped;
    # this works because the explanation text itself spans two lines (one line plus the heading
    # sentence split by a newline), so the remaining lines are the per-function lines.
    free_vars = logging_txt.split("\n")
    self.assertGreater(len(free_vars), 2)
    return "\n".join(free_vars[2:])

  def test_none_input(self):
    # [FUNCTION DOC]
    # Passing a null value to the generator returns null.
    txt = free_vars_detect.generate_free_var_logging(None)
    self.assertIsNone(txt)

  def test_non_function_input(self):
    # [FUNCTION DOC]
    # Passing a class instance (not a function, method or partial) returns null.
    x = 1

    class Foo():
      # [CLASS DOC]
      # Local fixture class with one method returning an enclosing variable.

      def bar(self):
        # [FUNCTION DOC]
        # Returns the enclosing variable.
        return x

    foo = Foo()
    txt = free_vars_detect.generate_free_var_logging(foo)
    self.assertIsNone(txt)

  def test_func_wo_source_code(self):
    # [FUNCTION DOC]
    # Creates a function from a code string at runtime in the module globals, so it has no
    # retrievable source, and asserts the generator returns null.
    # Side effect: defines a module-level function named f_exec.
    code = "def f_exec():\n  return 1"
    # Use `exec` to generate a function without source code
    exec(code, globals())  # pylint: disable=exec-used
    txt = free_vars_detect.generate_free_var_logging(f_exec)  # pylint: disable=undefined-variable
    self.assertIsNone(txt)

  def test_no_free_var(self):
    # [FUNCTION DOC]
    # A function with no free variables makes the generator return null.

    def f(x):
      # [FUNCTION DOC]
      # Returns the argument plus one.
      return x + 1

    txt = free_vars_detect.generate_free_var_logging(f)
    self.assertIsNone(txt)

  def test_single_func(self):
    # [FUNCTION DOC]
    # A function using two enclosing variables: after removing the leading lines, the text is
    # exactly the line naming the function and listing x and y.
    x = 1
    y = 2

    def f(a):
      # [FUNCTION DOC]
      # Returns the argument plus both enclosing variables.
      return a + x + y

    txt = free_vars_detect.generate_free_var_logging(f)
    txt = self._remove_explanation(txt)
    self.assertEqual(txt, "Inside function f(): x, y")

  def test_nested_func(self):
    # [FUNCTION DOC]
    # Function f calls g, each using an enclosing variable. Asserts the remaining text has two
    # lines: one for f listing g and x, one for g listing y.
    x = 1
    y = 2

    def g():
      # [FUNCTION DOC]
      # Returns the enclosing variable y.
      return y

    def f():
      # [FUNCTION DOC]
      # Returns the result of g plus the enclosing variable x.
      return g() + x

    txt = free_vars_detect.generate_free_var_logging(f)
    txt = self._remove_explanation(txt)
    lines = txt.split("\n")
    self.assertLen(lines, 2)
    self.assertEqual(lines[0], "Inside function f(): g, x")
    self.assertEqual(lines[1], "Inside function g(): y")

  def test_method_w_method_call(self):
    # [FUNCTION DOC]
    # Logging for a bound method that references a sibling method using an enclosing variable.
    # Asserts two lines: one for the first method listing self.g and one for the second listing x.
    x = 0

    class Foo():
      # [CLASS DOC]
      # Local fixture class: one method referencing another, which returns a list with x.

      def f(self):
        # [FUNCTION DOC]
        # Returns the sibling method g (not called).
        return self.g

      def g(self):
        # [FUNCTION DOC]
        # Returns a list holding the enclosing variable.
        return [x]

    foo = Foo()
    txt = free_vars_detect.generate_free_var_logging(foo.f)
    txt = self._remove_explanation(txt)
    lines = txt.split("\n")
    self.assertLen(lines, 2)
    self.assertEqual(lines[0], "Inside function Foo.f(): self.g")
    self.assertEqual(lines[1], "Inside function Foo.g(): x")

  def test_partial_func(self):
    # [FUNCTION DOC]
    # Logging for a partially applied function: the partial is unwrapped, so the text names the
    # underlying function and lists x and y.
    x = 1
    y = 2

    def f(a):
      # [FUNCTION DOC]
      # Returns the argument plus both enclosing variables.
      return a + x + y

    partial_f = functools.partial(f, a=0)

    txt = free_vars_detect.generate_free_var_logging(partial_f)
    txt = self._remove_explanation(txt)
    self.assertEqual(txt, "Inside function f(): x, y")

  def test_partial_method(self):
    # [FUNCTION DOC]
    # Logging for a method exposed through a partial-method attribute: asserts two lines, naming
    # the underlying method (listing self.g) and the sibling method (listing x).
    x = 0

    class Foo():
      # [CLASS DOC]
      # Local fixture class with two methods and a partial-method alias of the first.

      def f(self):
        # [FUNCTION DOC]
        # Returns the sibling method g (not called).
        return self.g

      def g(self):
        # [FUNCTION DOC]
        # Returns a list holding the enclosing variable.
        return [x]

      partial_f = functools.partialmethod(f)

    foo = Foo()
    txt = free_vars_detect.generate_free_var_logging(foo.partial_f)
    txt = self._remove_explanation(txt)
    lines = txt.split("\n")
    self.assertLen(lines, 2)
    self.assertEqual(lines[0], "Inside function Foo.f(): self.g")
    self.assertEqual(lines[1], "Inside function Foo.g(): x")

  def test_partial_wrapped_partial_func(self):
    # [FUNCTION DOC]
    # A function wrapped as partial, then decorated, then partial again. Asserts the generator
    # unwraps all layers and the text names f and lists x and y.

    def decorator_foo(func):
      # [FUNCTION DOC]
      # Test decorator. Input: func. Returns a pass-through wrapper with metadata copied from it.

      def wrapper(*args, **kwargs):
        # [FUNCTION DOC]
        # Calls the wrapped function with all given arguments and returns its result.
        return func(*args, **kwargs)

      return functools.update_wrapper(wrapper, func)

    x = 1
    y = 2

    def f(a, b):
      # [FUNCTION DOC]
      # Returns the sum of both arguments and both enclosing variables.
      return a + b + x + y

    f = functools.partial(f, a=0)
    f = decorator_foo(f)
    f = functools.partial(f, b=0)

    txt = free_vars_detect.generate_free_var_logging(f)
    txt = self._remove_explanation(txt)
    self.assertEqual(txt, "Inside function f(): x, y")

  def test_freevar_threshold(self):
    # [FUNCTION DOC]
    # A function with five free variables logged with a variable threshold of three: asserts the
    # line lists the first three names followed by an ellipsis.
    a = b = c = d = e = 1

    def f():
      # [FUNCTION DOC]
      # Returns the sum of the five enclosing variables.
      return a + b + c + d + e

    txt = free_vars_detect.generate_free_var_logging(f, var_threshold=3)
    txt = self._remove_explanation(txt)
    self.assertEqual(txt, "Inside function f(): a, b, c...")

  def test_func_threshold(self):
    # [FUNCTION DOC]
    # Function f calls g and h, all using an enclosing variable, logged with a function threshold
    # of two. Asserts three remaining lines: f's line, g's line, and an ellipsis line (h is cut).
    x = 1

    def g():
      # [FUNCTION DOC]
      # Returns the enclosing variable.
      return x

    def h():
      # [FUNCTION DOC]
      # Returns the enclosing variable.
      return x

    def f():
      # [FUNCTION DOC]
      # Returns the sum of calling g and h.
      return g() + h()

    txt = free_vars_detect.generate_free_var_logging(f, fn_threshold=2)
    txt = self._remove_explanation(txt)
    lines = txt.split("\n")
    self.assertLen(lines, 3)
    self.assertEqual(lines[0], "Inside function f(): g, h")
    self.assertEqual(lines[1], "Inside function g(): x")
    self.assertEqual(lines[2], "...")

  def test_func_second_call_return_none(self):
    # [FUNCTION DOC]
    # Calls the generator twice for the same function: the first call returns text, the second
    # returns null because logging is only produced once per function.
    x = 1

    def f():
      # [FUNCTION DOC]
      # Returns the enclosing variable.
      return x

    logging_txt = free_vars_detect.generate_free_var_logging(f)
    self.assertIsNotNone(logging_txt)
    logging_txt = free_vars_detect.generate_free_var_logging(f)
    self.assertIsNone(logging_txt)


if __name__ == "__main__":
  unittest.main()
