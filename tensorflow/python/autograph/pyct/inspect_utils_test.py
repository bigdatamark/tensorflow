# [MODULE DOC]
# Unit tests for the autograph inspect_utils module (source/namespace/class introspection helpers).
# Defines small fixture decorators, a fixture class and free functions at module level, then a
# single test case class that checks lambda detection, named-tuple detection, retrieval of the
# immediate source of decorated functions, namespace construction from globals and closures,
# qualified-name search through module graphs, owning-class lookup for methods, builtin and
# constructor detection and future-import detection. No import-time side effects besides imports.
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
"""Tests for inspect_utils module."""

import abc
import collections
import functools
import textwrap
import types

from tensorflow.python import lib
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct.testing import basic_definitions
from tensorflow.python.autograph.pyct.testing import decorators
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


def decorator(f):
  # [FUNCTION DOC]
  # HELPER: fixture decorator used to build decorated methods for the tests.
  # A no-op decorator: takes a function and returns that same function unchanged.
  return f


def function_decorator():
  # [FUNCTION DOC]
  # HELPER: fixture decorator factory used by the tests.
  # Takes no arguments and returns a decorator that returns its input function unchanged.
  def dec(f):
    # [FUNCTION DOC]
    # HELPER: the decorator produced by the factory; returns the given function unchanged.
    return f
  return dec


def wrapping_decorator():
  # [FUNCTION DOC]
  # HELPER: fixture decorator factory used by the tests.
  # Takes no arguments and returns a decorator that wraps the decorated function in a
  # wrapper (which preserves metadata) that ignores the original and calls a replacement.
  def dec(f):
    # [FUNCTION DOC]
    # HELPER: the decorator produced by the factory.
    # Takes a function and returns a metadata-preserving wrapper that delegates to a
    # replacement function returning nothing, instead of calling the original function.
    def replacement(*_):
      # [FUNCTION DOC]
      # HELPER: stand-in body for the wrapper; accepts any positional arguments, returns nothing.
      return None

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: wrapper returned by the decorator; forwards all arguments to the replacement
      # function and returns its result (nothing).
      return replacement(*args, **kwargs)
    return wrapper
  return dec


class TestClass:
  # [CLASS DOC]
  # HELPER: fixture class for the method-class lookup tests.
  # Has one plain method, methods decorated with a no-op decorator, a decorator factory and a
  # wrapping decorator, plus a static method and a class method. Never instantiated with state.

  def member_function(self):
    # [FUNCTION DOC]
    # HELPER: fixture plain method; does nothing and returns nothing.
    pass

  @decorator
  def decorated_member(self):
    # [FUNCTION DOC]
    # HELPER: fixture method with a no-op decorator; does nothing.
    pass

  @function_decorator()
  def fn_decorated_member(self):
    # [FUNCTION DOC]
    # HELPER: fixture method decorated through a decorator factory; does nothing.
    pass

  @wrapping_decorator()
  def wrap_decorated_member(self):
    # [FUNCTION DOC]
    # HELPER: fixture method decorated with a wrapping decorator; does nothing.
    pass

  @staticmethod
  def static_method():
    # [FUNCTION DOC]
    # HELPER: fixture static method; does nothing.
    pass

  @classmethod
  def class_method(cls):
    # [FUNCTION DOC]
    # HELPER: fixture class method; does nothing.
    pass


def free_function():
  # [FUNCTION DOC]
  # HELPER: fixture module-level function; does nothing and returns nothing.
  pass


def factory():
  # [FUNCTION DOC]
  # HELPER: fixture function that returns the module-level free_function.
  return free_function


def free_factory():
  # [FUNCTION DOC]
  # HELPER: fixture function that returns a locally defined function (not reachable by name
  # from the module).
  def local_function():
    # [FUNCTION DOC]
    # HELPER: fixture nested function; does nothing.
    pass
  return local_function


class InspectUtilsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the inspect_utils module: lambda and named-tuple detection, immediate
  # source retrieval, namespace and qualified-name resolution, method class lookup, builtin and
  # constructor detection and future-import detection. Uses the module-level fixtures above.

  def test_islambda(self):
    # [FUNCTION DOC]
    # Checks that islambda is true for a lambda and false for a regular named function.
    def test_fn():
      # [FUNCTION DOC]
      # HELPER: a regular nested function used as the negative case; does nothing.
      pass

    self.assertTrue(inspect_utils.islambda(lambda x: x))
    self.assertFalse(inspect_utils.islambda(test_fn))

  def test_islambda_renamed_lambda(self):
    # [FUNCTION DOC]
    # Checks that a lambda whose name was changed to an ordinary name is still reported as a lambda.
    l = lambda x: 1
    l.__name__ = 'f'
    self.assertTrue(inspect_utils.islambda(l))

  def test_isnamedtuple(self):
    # [FUNCTION DOC]
    # Checks isnamedtuple is true for a genuine named-tuple type and false for a plain tuple subclass.
    nt = collections.namedtuple('TestNamedTuple', ['a', 'b'])

    class NotANamedTuple(tuple):
      # [CLASS DOC]
      # HELPER: plain tuple subclass without field names; negative case for isnamedtuple.
      pass

    self.assertTrue(inspect_utils.isnamedtuple(nt))
    self.assertFalse(inspect_utils.isnamedtuple(NotANamedTuple))

  def test_isnamedtuple_confounder(self):
    # [FUNCTION DOC]
    # Documents a known false positive: a tuple subclass that merely declares a field-names
    # attribute is reported as a named tuple, and the test asserts that (true) outcome.
    """This test highlights false positives when detecting named tuples."""

    class NamedTupleLike(tuple):
      # [CLASS DOC]
      # HELPER: tuple subclass with a field-names attribute but not created by the named-tuple
      # factory; used to show the detection heuristic is fooled.
      _fields = ('a', 'b')

    self.assertTrue(inspect_utils.isnamedtuple(NamedTupleLike))

  def test_isnamedtuple_subclass(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it highlights false positives, but a subclass of a real
    # named tuple is arguably a legitimate named tuple; the test just asserts it is detected.
    # Checks that a subclass of a genuine named-tuple type is reported as a named tuple.
    """This test highlights false positives when detecting named tuples."""

    class NamedTupleSubclass(collections.namedtuple('Test', ['a', 'b'])):
      # [CLASS DOC]
      # HELPER: subclass of a genuine named-tuple type with no additions.
      pass

    self.assertTrue(inspect_utils.isnamedtuple(NamedTupleSubclass))

  def assertSourceIdentical(self, actual, expected):
    # [FUNCTION DOC]
    # HELPER: custom assertion for the source-retrieval tests.
    # Takes two source strings, removes common leading indentation and surrounding whitespace
    # from each, and asserts they are equal. Fails the test otherwise; returns nothing.
    self.assertEqual(
        textwrap.dedent(actual).strip(),
        textwrap.dedent(expected).strip()
    )

  def test_getimmediatesource_basic(self):
    # [FUNCTION DOC]
    # Decorates a function with a locally defined decorator whose wrapper is a plain function and
    # checks that getimmediatesource returns the source of the wrapper (the immediate function),
    # not of the original decorated function.

    def test_decorator(f):
      # [FUNCTION DOC]
      # HELPER: local decorator that returns a plain wrapper function calling the original.

      def f_wrapper(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: wrapper whose source is the expected result; forwards all arguments to f.
        return f(*args, **kwargs)

      return f_wrapper

    expected = """
      def f_wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    """

    @test_decorator
    def test_fn(a):
      # [FUNCTION DOC]
      # HELPER: decorated function under inspection; returns a list holding its argument.
      """Test docstring."""
      return [a]

    self.assertSourceIdentical(
        inspect_utils.getimmediatesource(test_fn), expected)

  def test_getimmediatesource_noop_decorator(self):
    # [FUNCTION DOC]
    # Applies a decorator that returns the function unchanged and checks that the immediate source
    # is the original function's own source, including the decorator line and docstring.

    def test_decorator(f):
      # [FUNCTION DOC]
      # HELPER: local no-op decorator; returns the given function unchanged.
      return f

    expected = '''
      @test_decorator
      def test_fn(a):
        """Test docstring."""
        return [a]
    '''

    @test_decorator
    def test_fn(a):
      # [FUNCTION DOC]
      # HELPER: decorated function under inspection; returns a list holding its argument.
      """Test docstring."""
      return [a]

    self.assertSourceIdentical(
        inspect_utils.getimmediatesource(test_fn), expected)

  def test_getimmediatesource_functools_wrapper(self):
    # [FUNCTION DOC]
    # Applies a decorator whose wrapper uses the metadata-preserving wrapping helper and checks
    # that the immediate source is the wrapper's source (with its decorator line), not the
    # original function's.

    def wrapper_decorator(f):
      # [FUNCTION DOC]
      # HELPER: local decorator returning a metadata-preserving wrapper that calls f.

      @functools.wraps(f)
      def wrapper(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: wrapper whose source is expected; forwards all arguments to f.
        return f(*args, **kwargs)

      return wrapper

    expected = textwrap.dedent("""
      @functools.wraps(f)
      def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    """)

    @wrapper_decorator
    def test_fn(a):
      # [FUNCTION DOC]
      # HELPER: decorated function under inspection; returns a list holding its argument.
      """Test docstring."""
      return [a]

    self.assertSourceIdentical(
        inspect_utils.getimmediatesource(test_fn), expected)

  def test_getimmediatesource_functools_wrapper_different_module(self):
    # [FUNCTION DOC]
    # Same as the metadata-preserving wrapper case, but the decorator comes from the separate
    # testing decorators module; checks the wrapper's source is still returned.

    expected = textwrap.dedent("""
      @functools.wraps(f)
      def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    """)

    @decorators.wrapping_decorator
    def test_fn(a):
      # [FUNCTION DOC]
      # HELPER: decorated function under inspection; returns a list holding its argument.
      """Test docstring."""
      return [a]

    self.assertSourceIdentical(
        inspect_utils.getimmediatesource(test_fn), expected)

  def test_getimmediatesource_normal_decorator_different_module(self):
    # [FUNCTION DOC]
    # Uses a plain (non metadata-preserving) decorator from the testing decorators module and
    # checks the immediate source is that decorator's standalone wrapper function.

    expected = textwrap.dedent("""
      def standalone_wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    """)

    @decorators.standalone_decorator
    def test_fn(a):
      # [FUNCTION DOC]
      # HELPER: decorated function under inspection; returns a list holding its argument.
      """Test docstring."""
      return [a]

    self.assertSourceIdentical(
        inspect_utils.getimmediatesource(test_fn), expected)

  def test_getimmediatesource_normal_functional_decorator_different_module(
      self):
    # [FUNCTION DOC]
    # Uses a decorator factory from the testing decorators module and checks the immediate source
    # is the functional wrapper defined by that factory.

    expected = textwrap.dedent("""
      def functional_wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    """)

    @decorators.functional_decorator()
    def test_fn(a):
      # [FUNCTION DOC]
      # HELPER: decorated function under inspection; returns a list holding its argument.
      """Test docstring."""
      return [a]

    self.assertSourceIdentical(
        inspect_utils.getimmediatesource(test_fn), expected)

  def test_getnamespace_globals(self):
    # [FUNCTION DOC]
    # Checks getnamespace of a module-level function exposes the module's globals, e.g. the
    # free function is found under its name.
    ns = inspect_utils.getnamespace(factory)
    self.assertEqual(ns['free_function'], free_function)

  def test_getnamespace_closure_with_undefined_var(self):
    # [FUNCTION DOC]
    # A nested function closes over a variable that is not yet assigned. Checks the variable is
    # absent from the namespace while unbound, and present with its value once assigned.
    if False:  # pylint:disable=using-constant-test
      a = 1

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: closure that returns the captured variable.
      return a

    ns = inspect_utils.getnamespace(test_fn)
    self.assertNotIn('a', ns)

    a = 2
    ns = inspect_utils.getnamespace(test_fn)

    self.assertEqual(ns['a'], 2)

  def test_getnamespace_hermetic(self):
    # [FUNCTION DOC]
    # Shadows a global name with a local variable captured by a closure and checks the returned
    # namespace uses the local value while the function's real globals mapping is left unmodified.

    # Intentionally hiding the global function to make sure we don't overwrite
    # it in the global namespace.
    free_function = object()  # pylint:disable=redefined-outer-name

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: closure returning the shadowing local variable.
      return free_function

    ns = inspect_utils.getnamespace(test_fn)
    globs = test_fn.__globals__
    self.assertTrue(ns['free_function'] is free_function)
    self.assertFalse(globs['free_function'] is free_function)

  def test_getnamespace_locals(self):
    # [FUNCTION DOC]
    # Builds a nested function that closes over a function, a list and a number, and checks the
    # namespace contains exactly those captured values but not the function's own local variable.

    def called_fn():
      # [FUNCTION DOC]
      # HELPER: captured function; returns zero.
      return 0

    closed_over_list = []
    closed_over_primitive = 1

    def local_fn():
      # [FUNCTION DOC]
      # HELPER: function under inspection; appends to a captured list and returns a sum using the
      # captured function and number.
      closed_over_list.append(1)
      local_var = 1
      return called_fn() + local_var + closed_over_primitive

    ns = inspect_utils.getnamespace(local_fn)
    self.assertEqual(ns['called_fn'], called_fn)
    self.assertEqual(ns['closed_over_list'], closed_over_list)
    self.assertEqual(ns['closed_over_primitive'], closed_over_primitive)
    self.assertTrue('local_var' not in ns)

  def test_getqualifiedname(self):
    # [FUNCTION DOC]
    # Builds a namespace mapping with an object and two fake modules (one holding another object)
    # and checks getqualifiedname returns nothing for an unreachable object, the plain name for
    # top-level entries, and a dotted path for an attribute of a module.
    foo = object()
    qux = types.ModuleType('quxmodule')
    bar = types.ModuleType('barmodule')
    baz = object()
    bar.baz = baz

    ns = {
        'foo': foo,
        'bar': bar,
        'qux': qux,
    }

    self.assertIsNone(inspect_utils.getqualifiedname(ns, inspect_utils))
    self.assertEqual(inspect_utils.getqualifiedname(ns, foo), 'foo')
    self.assertEqual(inspect_utils.getqualifiedname(ns, bar), 'bar')
    self.assertEqual(inspect_utils.getqualifiedname(ns, baz), 'bar.baz')

  def test_getqualifiedname_efficiency(self):
    # [FUNCTION DOC]
    # Builds ten levels of ten fake modules where every module links to all modules of the next
    # level and hides a target object in the last module. Checks that the search returns nothing
    # for an absent object and still finds the target (with a huge depth limit) in reasonable
    # time, i.e. visited modules are skipped.
    foo = object()

    # We create a densely connected graph consisting of a relatively small
    # number of modules and hide our symbol in one of them. The path to the
    # symbol is at least 10, and each node has about 10 neighbors. However,
    # by skipping visited modules, the search should take much less.
    ns = {}
    prev_level = []
    for i in range(10):
      current_level = []
      for j in range(10):
        mod_name = 'mod_{}_{}'.format(i, j)
        mod = types.ModuleType(mod_name)
        current_level.append(mod)
        if i == 9 and j == 9:
          mod.foo = foo
      if prev_level:
        # All modules at level i refer to all modules at level i+1
        for prev in prev_level:
          for mod in current_level:
            prev.__dict__[mod.__name__] = mod
      else:
        for mod in current_level:
          ns[mod.__name__] = mod
      prev_level = current_level

    self.assertIsNone(inspect_utils.getqualifiedname(ns, inspect_utils))
    self.assertIsNotNone(
        inspect_utils.getqualifiedname(ns, foo, max_depth=10000000000))

  def test_getqualifiedname_cycles(self):
    # [FUNCTION DOC]
    # Builds a chain of ten fake modules in which each module refers to the next and to all
    # earlier ones (circular references), with a target hidden in the last. Checks the search
    # terminates, returns nothing for an absent object and finds the target.
    foo = object()

    # We create a graph of modules that contains circular references. The
    # search process should avoid them. The searched object is hidden at the
    # bottom of a path of length roughly 10.
    ns = {}
    mods = []
    for i in range(10):
      mod = types.ModuleType('mod_{}'.format(i))
      if i == 9:
        mod.foo = foo
      # Module i refers to module i+1
      if mods:
        mods[-1].__dict__[mod.__name__] = mod
      else:
        ns[mod.__name__] = mod
      # Module i refers to all modules j < i.
      for prev in mods:
        mod.__dict__[prev.__name__] = prev
      mods.append(mod)

    self.assertIsNone(inspect_utils.getqualifiedname(ns, inspect_utils))
    self.assertIsNotNone(
        inspect_utils.getqualifiedname(ns, foo, max_depth=10000000000))

  def test_getqualifiedname_finds_via_parent_module(self):
    # [FUNCTION DOC]
    # Searches the namespace of the TensorFlow lib package with a depth limit of one for a file
    # I/O class and checks the dotted result is the path through the parent submodule. Fragile:
    # depends on the current layout of the lib package (noted in the existing TODO).
    # TODO(mdan): This test is vulnerable to change in the lib module.
    # A better way to forge modules should be found.
    self.assertEqual(
        inspect_utils.getqualifiedname(
            lib.__dict__, lib.io.file_io.FileIO, max_depth=1),
        'io.file_io.FileIO')

  def test_getmethodclass(self):
    # [FUNCTION DOC]
    # Checks getmethodclass returns nothing for free and locally created functions, and returns
    # the fixture class for all its member kinds (plain, decorated in three ways, static, class
    # method), both when accessed on the class and on an instance.

    self.assertEqual(
        inspect_utils.getmethodclass(free_function), None)
    self.assertEqual(
        inspect_utils.getmethodclass(free_factory()), None)

    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.member_function),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.fn_decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.wrap_decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.static_method),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(TestClass.class_method),
        TestClass)

    test_obj = TestClass()
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.member_function),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.fn_decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.wrap_decorated_member),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.static_method),
        TestClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.class_method),
        TestClass)

  def test_getmethodclass_locals(self):
    # [FUNCTION DOC]
    # Same as the module-level method-class test but with a class defined inside the test:
    # checks nothing is returned for a local function and the local class is returned for its
    # plain and decorated members, accessed on the class and on an instance.

    def local_function():
      # [FUNCTION DOC]
      # HELPER: local function; does nothing.
      pass

    class LocalClass:
      # [CLASS DOC]
      # HELPER: locally defined fixture class with a plain method and three differently
      # decorated methods.

      def member_function(self):
        # [FUNCTION DOC]
        # HELPER: plain fixture method; does nothing.
        pass

      @decorator
      def decorated_member(self):
        # [FUNCTION DOC]
        # HELPER: fixture method with a no-op decorator; does nothing.
        pass

      @function_decorator()
      def fn_decorated_member(self):
        # [FUNCTION DOC]
        # HELPER: fixture method decorated via a decorator factory; does nothing.
        pass

      @wrapping_decorator()
      def wrap_decorated_member(self):
        # [FUNCTION DOC]
        # HELPER: fixture method with a wrapping decorator; does nothing.
        pass

    self.assertEqual(
        inspect_utils.getmethodclass(local_function), None)

    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.member_function),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.decorated_member),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.fn_decorated_member),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(LocalClass.wrap_decorated_member),
        LocalClass)

    test_obj = LocalClass()
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.member_function),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.decorated_member),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.fn_decorated_member),
        LocalClass)
    self.assertEqual(
        inspect_utils.getmethodclass(test_obj.wrap_decorated_member),
        LocalClass)

  def test_getmethodclass_callables(self):
    # [FUNCTION DOC]
    # Checks that for an instance of a class defining a call method, getmethodclass returns that class.

    class TestCallable:
      # [CLASS DOC]
      # HELPER: fixture class whose instances are callable.

      def __call__(self):
        # [FUNCTION DOC]
        # HELPER: fixture call method; does nothing.
        pass

    c = TestCallable()
    self.assertEqual(inspect_utils.getmethodclass(c), TestCallable)

  def test_getmethodclass_no_bool_conversion(self):
    # [FUNCTION DOC]
    # Takes a bound method of a tensor and checks getmethodclass returns the tensor's type,
    # i.e. the lookup works without converting the tensor to a boolean.

    tensor = constant_op.constant([1])
    self.assertEqual(
        inspect_utils.getmethodclass(tensor.get_shape), type(tensor))

  def test_getdefiningclass(self):
    # [FUNCTION DOC]
    # Builds a base class and a subclass overriding one method and checks getdefiningclass
    # returns the subclass for overridden and subclass-only methods and the base class for
    # inherited methods, including an inherited class method.

    class Superclass:
      # [CLASS DOC]
      # HELPER: fixture base class with two instance methods and a class method.

      def foo(self):
        # [FUNCTION DOC]
        # HELPER: base method that is overridden in the subclass; does nothing.
        pass

      def bar(self):
        # [FUNCTION DOC]
        # HELPER: base method inherited unchanged by the subclass; does nothing.
        pass

      @classmethod
      def class_method(cls):
        # [FUNCTION DOC]
        # HELPER: base class method inherited by the subclass; does nothing.
        pass

    class Subclass(Superclass):
      # [CLASS DOC]
      # HELPER: fixture subclass overriding one base method and adding another.

      def foo(self):
        # [FUNCTION DOC]
        # HELPER: override of the base method; does nothing.
        pass

      def baz(self):
        # [FUNCTION DOC]
        # HELPER: method defined only in the subclass; does nothing.
        pass

    self.assertIs(
        inspect_utils.getdefiningclass(Subclass.foo, Subclass), Subclass)
    self.assertIs(
        inspect_utils.getdefiningclass(Subclass.bar, Subclass), Superclass)
    self.assertIs(
        inspect_utils.getdefiningclass(Subclass.baz, Subclass), Subclass)
    self.assertIs(
        inspect_utils.getdefiningclass(Subclass.class_method, Subclass),
        Superclass)

  def test_isbuiltin(self):
    # [FUNCTION DOC]
    # Checks isbuiltin is true for several builtin functions and types and false for a user-defined function.
    self.assertTrue(inspect_utils.isbuiltin(enumerate))
    self.assertTrue(inspect_utils.isbuiltin(eval))
    self.assertTrue(inspect_utils.isbuiltin(float))
    self.assertTrue(inspect_utils.isbuiltin(int))
    self.assertTrue(inspect_utils.isbuiltin(len))
    self.assertTrue(inspect_utils.isbuiltin(range))
    self.assertTrue(inspect_utils.isbuiltin(zip))
    self.assertFalse(inspect_utils.isbuiltin(function_decorator))

  def test_isconstructor(self):
    # [FUNCTION DOC]
    # Checks isconstructor is true for ordinary classes, classes with callable instances,
    # metaclasses and instances of a plain metaclass (which are classes), but false for a class
    # created by a metaclass that overrides the call behaviour.

    class OrdinaryClass:
      # [CLASS DOC]
      # HELPER: empty fixture class.
      pass

    class OrdinaryCallableClass:
      # [CLASS DOC]
      # HELPER: fixture class whose instances are callable.

      def __call__(self):
        # [FUNCTION DOC]
        # HELPER: fixture call method; does nothing.
        pass

    class Metaclass(type):
      # [CLASS DOC]
      # HELPER: empty fixture metaclass.
      pass

    class CallableMetaclass(type):
      # [CLASS DOC]
      # HELPER: fixture metaclass that overrides the call behaviour for its classes.

      def __call__(cls):
        # [FUNCTION DOC]
        # HELPER: overridden call on the metaclass; does nothing and returns nothing.
        pass

    self.assertTrue(inspect_utils.isconstructor(OrdinaryClass))
    self.assertTrue(inspect_utils.isconstructor(OrdinaryCallableClass))
    self.assertTrue(inspect_utils.isconstructor(Metaclass))
    self.assertTrue(inspect_utils.isconstructor(Metaclass('TestClass', (), {})))
    self.assertTrue(inspect_utils.isconstructor(CallableMetaclass))

    self.assertFalse(inspect_utils.isconstructor(
        CallableMetaclass('TestClass', (), {})))

  def test_isconstructor_abc_callable(self):
    # [FUNCTION DOC]
    # Defines an abstract base class with an abstract call method and a concrete subclass, and
    # checks isconstructor is true for both classes.

    class AbcBase(metaclass=abc.ABCMeta):
      # [CLASS DOC]
      # HELPER: fixture abstract base class declaring an abstract call method.

      @abc.abstractmethod
      def __call__(self):
        # [FUNCTION DOC]
        # HELPER: abstract call method; does nothing.
        pass

    class AbcSubclass(AbcBase):
      # [CLASS DOC]
      # HELPER: concrete fixture subclass implementing the call method.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: constructor; does nothing.
        pass

      def __call__(self):
        # [FUNCTION DOC]
        # HELPER: call method implementation; does nothing.
        pass

    self.assertTrue(inspect_utils.isconstructor(AbcBase))
    self.assertTrue(inspect_utils.isconstructor(AbcSubclass))

  def test_getfutureimports_functions(self):
    # [FUNCTION DOC]
    # Checks getfutureimports for a function from the testing definitions module reports none of
    # the legacy future feature names (absolute import, division, print function, generators).
    imps = inspect_utils.getfutureimports(basic_definitions.function_with_print)
    self.assertNotIn('absolute_import', imps)
    self.assertNotIn('division', imps)
    self.assertNotIn('print_function', imps)
    self.assertNotIn('generators', imps)

  def test_getfutureimports_lambdas(self):
    # [FUNCTION DOC]
    # Same check as for functions but for a lambda from the testing definitions module: no legacy
    # future feature names are reported.
    imps = inspect_utils.getfutureimports(basic_definitions.simple_lambda)
    self.assertNotIn('absolute_import', imps)
    self.assertNotIn('division', imps)
    self.assertNotIn('print_function', imps)
    self.assertNotIn('generators', imps)

  def test_getfutureimports_methods(self):
    # [FUNCTION DOC]
    # Same check as for functions but for a method of a class from the testing definitions
    # module: no legacy future feature names are reported.
    imps = inspect_utils.getfutureimports(
        basic_definitions.SimpleClass.method_with_print)
    self.assertNotIn('absolute_import', imps)
    self.assertNotIn('division', imps)
    self.assertNotIn('print_function', imps)
    self.assertNotIn('generators', imps)


if __name__ == '__main__':
  test.main()
