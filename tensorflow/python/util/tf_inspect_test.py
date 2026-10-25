# [MODULE DOC]
# Unit tests for the tf_inspect module, which wraps the standard inspect module so that it
# understands tf_decorator-wrapped callables. The file defines a few decorated and undecorated
# sample functions and a decorated class (using a local decorator factory built on TFDecorator),
# then checks that argument-spec, full-argument-spec, signature, doc, file, source, member,
# module, type-predicate, stack and call-argument helpers return the same answers as the
# standard library (or the expected decorated-aware answers), including for partial functions,
# callable objects and classes with custom constructors. No import-time side effects beyond
# imports; run as a test binary through the TensorFlow test main.
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
"""Unit tests for tf_inspect."""

# pylint: disable=unused-import
import functools
import inspect

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect


def test_decorator(decorator_name, decorator_doc=None):
  # [FUNCTION DOC]
  # HELPER: test utility that builds decorators used to create decorated sample targets.
  # Decorator factory. Takes a decorator name (string) and an optional documentation string
  # and returns a decorator function; the returned decorator wraps its target in a
  # TFDecorator carrying that name and doc. No side effects.

  def make_tf_decorator(target):
    # [FUNCTION DOC]
    # HELPER: inner decorator produced by the decorator factory.
    # Wraps the given target callable in a TFDecorator using the captured name and doc and
    # returns the resulting TFDecorator object. No side effects.
    return tf_decorator.TFDecorator(decorator_name, target, decorator_doc)

  return make_tf_decorator


def test_undecorated_function():
  # [FUNCTION DOC]
  # HELPER: plain sample target used as the wrapped function in decorator tests.
  # Takes no arguments, does nothing and returns nothing.
  pass


@test_decorator('decorator 1')
@test_decorator('decorator 2')
@test_decorator('decorator 3')
def test_decorated_function(x):
  # [FUNCTION DOC]
  # HELPER: sample target wrapped by three nested test decorators.
  # Takes one argument x and returns it multiplied by two. Used to check that signature and
  # argument inspection follow nested decorators down to the innermost function.
  """Test Decorated Function Docstring."""
  return x * 2


@test_decorator('decorator')
def test_decorated_function_with_defaults(a, b=2, c='Hello'):
  """Test Decorated Function With Defaults Docstring."""
  return [a, b, c]


@test_decorator('decorator')
def test_decorated_function_with_varargs_and_kwonlyargs(*args, b=2, c='Hello'):
  # [FUNCTION DOC]
  # HELPER: sample target with variable positional arguments and two keyword-only arguments
  # (defaults 2 and 'Hello'), wrapped by a test decorator. Returns a list of the packed
  # positional arguments, b and c.
  """Test Decorated Function With both varargs and keyword args."""
  return [args, b, c]


@test_decorator('decorator')
class TestDecoratedClass(object):
  # [CLASS DOC]
  # HELPER: sample class (wrapped by a test decorator) used as an inspection target.
  # Has a no-argument constructor and a method returning two; carries no state.
  """Test Decorated Class."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor that takes no arguments and does nothing.
    pass

  def two(self):
    # [FUNCTION DOC]
    # Returns the integer two. No side effects.
    return 2


class TfInspectTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the tf_inspect wrappers: argument specs (including partials, callable
  # objects and classes), signature, doc, file, members, module, source, type predicates,
  # stack and decorator-target detection. Uses the module-level decorated sample functions
  # and class as shared fixtures.

  def testCurrentFrame(self):
    # [FUNCTION DOC]
    # Checks that tf_inspect current-frame returns the same frame as the standard inspect
    # module.
    self.assertEqual(inspect.currentframe(), tf_inspect.currentframe())

  def testGetArgSpecOnDecoratorsThatDontProvideArgspec(self):
    # [FUNCTION DOC]
    # Gets the argument spec of a decorated function whose decorator provides no spec and
    # asserts the arguments are a, b, c and the defaults are 2 and 'Hello' (taken from the
    # underlying function).
    argspec = tf_inspect.getargspec(test_decorated_function_with_defaults)
    self.assertEqual(['a', 'b', 'c'], argspec.args)
    self.assertEqual((2, 'Hello'), argspec.defaults)

  def testGetArgSpecOnDecoratorThatChangesArgspec(self):
    # [FUNCTION DOC]
    # Wraps an undecorated function in a TFDecorator that declares its own full argument spec
    # (a, b, c with two defaults) and asserts that the full-argspec lookup returns exactly
    # that declared spec.
    argspec = tf_inspect.FullArgSpec(
        args=['a', 'b', 'c'],
        varargs=None,
        varkw=None,
        defaults=(1, 'hello'),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={},
    )

    decorator = tf_decorator.TFDecorator('', test_undecorated_function, '',
                                         argspec)
    self.assertEqual(argspec, tf_inspect.getfullargspec(decorator))

  def testGetArgSpecIgnoresDecoratorsThatDontProvideArgspec(self):
    # [FUNCTION DOC]
    # Builds an inner decorator with a declared spec and an outer decorator with none, then
    # asserts the argspec lookup on the outer decorator skips it and returns the inner spec.
    argspec = tf_inspect.FullArgSpec(
        args=['a', 'b', 'c'],
        varargs=None,
        varkw=None,
        defaults=(1, 'hello'),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={},
    )

    inner_decorator = tf_decorator.TFDecorator(
        '', test_undecorated_function, '', argspec
    )
    outer_decorator = tf_decorator.TFDecorator('', inner_decorator)
    self.assertEqual(argspec, tf_inspect.getargspec(outer_decorator))

  def testGetArgSpecThatContainsVarargsAndKwonlyArgs(self):
    # [FUNCTION DOC]
    # Gets the argspec of a decorated function with varargs and keyword-only arguments and
    # asserts the legacy argspec reports the keyword-only names b, c as args with defaults
    # 2 and 'Hello'.
    argspec = tf_inspect.getargspec(
        test_decorated_function_with_varargs_and_kwonlyargs
    )
    self.assertEqual(['b', 'c'], argspec.args)
    self.assertEqual((2, 'Hello'), argspec.defaults)

  def testGetArgSpecReturnsOutermostDecoratorThatChangesArgspec(self):
    # [FUNCTION DOC]
    # Nests two decorators that each declare a different spec and asserts the full-argspec
    # lookup returns the spec of the outermost one.
    outer_argspec = tf_inspect.FullArgSpec(
        args=['a'],
        varargs=None,
        varkw=None,
        defaults=(),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={},
    )
    inner_argspec = tf_inspect.FullArgSpec(
        args=['b'],
        varargs=None,
        varkw=None,
        defaults=(),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={},
    )

    inner_decorator = tf_decorator.TFDecorator('', test_undecorated_function,
                                               '', inner_argspec)
    outer_decorator = tf_decorator.TFDecorator('', inner_decorator, '',
                                               outer_argspec)
    self.assertEqual(outer_argspec, tf_inspect.getfullargspec(outer_decorator))

  def testGetArgSpecOnPartialPositionalArgumentOnly(self):
    # [FUNCTION DOC]
    # Binds the first positional argument of a two-argument function with a partial and
    # asserts the argspec lists only the remaining argument n with no defaults.
    """Tests getargspec on partial function with only positional arguments."""

    def func(m, n):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns twice m plus n.
      return 2 * m + n

    partial_func = functools.partial(func, 7)
    argspec = tf_inspect.ArgSpec(
        args=['n'], varargs=None, keywords=None, defaults=None)

    self.assertEqual(argspec, tf_inspect.getargspec(partial_func))

  def testGetArgSpecOnPartialArgumentWithConvertibleToFalse(self):
    # [FUNCTION DOC]
    # Binds the first argument by keyword to a falsy value (zero) in a partial, leaving a
    # positional argument after a keyword-bound one, and asserts that the argspec lookup
    # raises a value error mentioning keyword-only arguments.
    """Tests getargspec on partial function with args that convert to False."""

    def func(m, n):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns twice m plus n.
      return 2 * m + n

    partial_func = functools.partial(func, m=0)

    with self.assertRaisesRegex(ValueError, 'keyword-only arguments'):
      tf_inspect.getargspec(partial_func)

  def testGetArgSpecOnPartialInvalidArgspec(self):
    # [FUNCTION DOC]
    # Binds a middle argument by keyword in a partial so the remaining positional arguments
    # follow a keyword-bound one, and asserts the argspec lookup raises a value error
    # mentioning keyword-only arguments.
    """Tests getargspec on partial function that doesn't have valid argspec."""

    def func(m, n, l, k=4):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns a weighted sum of its arguments.
      return 2 * m + l + n * k

    partial_func = functools.partial(func, n=7)

    with self.assertRaisesRegex(ValueError, 'keyword-only arguments'):
      tf_inspect.getargspec(partial_func)

  def testGetArgSpecOnPartialValidArgspec(self):
    # [FUNCTION DOC]
    # Binds n and l by keyword in a partial of a four-argument function and asserts the
    # argspec keeps all four names with defaults 7, 2 and 4 (the bound values replace the
    # original defaults).
    """Tests getargspec on partial function with valid argspec."""

    def func(m, n, l, k=4):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns a weighted sum of its arguments.
      return 2 * m + l + n * k

    partial_func = functools.partial(func, n=7, l=2)
    argspec = tf_inspect.ArgSpec(
        args=['m', 'n', 'l', 'k'],
        varargs=None,
        keywords=None,
        defaults=(7, 2, 4))

    self.assertEqual(argspec, tf_inspect.getargspec(partial_func))

  def testGetArgSpecOnPartialNoArgumentsLeft(self):
    # [FUNCTION DOC]
    # Binds both positional arguments in a partial and asserts the argspec has an empty
    # argument list, no varargs, no keywords and no defaults.
    """Tests getargspec on partial function that prunes all arguments."""

    def func(m, n):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns twice m plus n.
      return 2 * m + n

    partial_func = functools.partial(func, 7, 10)
    argspec = tf_inspect.ArgSpec(
        args=[], varargs=None, keywords=None, defaults=None)

    self.assertEqual(argspec, tf_inspect.getargspec(partial_func))

  def testGetArgSpecOnPartialKeywordArgument(self):
    # [FUNCTION DOC]
    # Binds n by keyword in a partial and asserts the argspec lists m and n with a single
    # default of 7 for n.
    """Tests getargspec on partial function that prunes some arguments."""

    def func(m, n):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns twice m plus n.
      return 2 * m + n

    partial_func = functools.partial(func, n=7)
    argspec = tf_inspect.ArgSpec(
        args=['m', 'n'], varargs=None, keywords=None, defaults=(7,))

    self.assertEqual(argspec, tf_inspect.getargspec(partial_func))

  def testGetArgSpecOnPartialKeywordArgumentWithDefaultValue(self):
    # [FUNCTION DOC]
    # Binds n by keyword (value 7) in a partial of a function whose arguments both have
    # defaults and asserts the argspec defaults become 1 and 7.
    """Tests getargspec on partial function that prunes argument by keyword."""

    def func(m=1, n=2):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns twice m plus n.
      return 2 * m + n

    partial_func = functools.partial(func, n=7)
    argspec = tf_inspect.ArgSpec(
        args=['m', 'n'], varargs=None, keywords=None, defaults=(1, 7))

    self.assertEqual(argspec, tf_inspect.getargspec(partial_func))

  def testGetArgSpecOnPartialWithVarargs(self):
    # [FUNCTION DOC]
    # Binds the fixed argument and one extra positional value in a partial of a function
    # with variable positional arguments and asserts the argspec has no named args and
    # reports the varargs name 'arg'.
    """Tests getargspec on partial function with variable arguments."""

    def func(m, *arg):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns m plus the count of extra arguments.
      return m + len(arg)

    partial_func = functools.partial(func, 7, 8)
    argspec = tf_inspect.ArgSpec(
        args=[], varargs='arg', keywords=None, defaults=None)

    self.assertEqual(argspec, tf_inspect.getargspec(partial_func))

  def testGetArgSpecOnPartialWithVarkwargs(self):
    # [FUNCTION DOC]
    # Binds the first argument in a partial of a function with variable keyword arguments
    # and asserts the argspec lists n and reports the keywords name 'kwarg'.
    """Tests getargspec on partial function with variable keyword arguments."""

    def func(m, n, **kwarg):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns m times n plus the keyword count.
      return m * n + len(kwarg)

    partial_func = functools.partial(func, 7)
    argspec = tf_inspect.ArgSpec(
        args=['n'], varargs=None, keywords='kwarg', defaults=None)

    self.assertEqual(argspec, tf_inspect.getargspec(partial_func))

  def testGetArgSpecOnPartialWithDecorator(self):
    # [FUNCTION DOC]
    # Applies a test decorator to a function, binds n by keyword in a partial and asserts the
    # argspec equals the one for the undecorated case (args m, n; defaults 1 and 7).
    """Tests getargspec on decorated partial function."""

    @test_decorator('decorator')
    def func(m=1, n=2):
      # [FUNCTION DOC]
      # HELPER: decorated target for the partial; returns twice m plus n.
      return 2 * m + n

    partial_func = functools.partial(func, n=7)
    argspec = tf_inspect.ArgSpec(
        args=['m', 'n'], varargs=None, keywords=None, defaults=(1, 7))

    self.assertEqual(argspec, tf_inspect.getargspec(partial_func))

  def testGetArgSpecOnPartialWithDecoratorThatChangesArgspec(self):
    # [FUNCTION DOC]
    # Creates a TFDecorator with a declared spec, wraps it in a partial binding a=2 and
    # asserts that the full-argspec lookup on the decorator returns the declared spec while
    # the standard signature of the partial shows a, b, c as keyword-only parameters with
    # defaults 2, 1 and 'hello'.
    """Tests getargspec on partial function with decorated argspec."""

    argspec = tf_inspect.FullArgSpec(
        args=['a', 'b', 'c'],
        varargs=None,
        varkw=None,
        defaults=(1, 'hello'),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={},
    )
    decorator = tf_decorator.TFDecorator('', test_undecorated_function, '',
                                         argspec)
    signature = inspect.Signature([
        inspect.Parameter(
            'a', inspect.Parameter.KEYWORD_ONLY, default=2
        ),
        inspect.Parameter(
            'b', inspect.Parameter.KEYWORD_ONLY, default=1
        ),
        inspect.Parameter(
            'c', inspect.Parameter.KEYWORD_ONLY, default='hello'
        ),
    ])
    partial_with_decorator = functools.partial(decorator, a=2)
    self.assertEqual(argspec, tf_inspect.getfullargspec(decorator))
    self.assertEqual(signature, inspect.signature(partial_with_decorator))

  def testGetArgSpecOnCallableObject(self):
    # [FUNCTION DOC]
    # Gets the argspec of an instance of a class defining a call method and asserts it lists
    # self, a, b, c with defaults 1 and 'hello'.

    class Callable(object):
      # [CLASS DOC]
      # HELPER: local sample class whose instances are callable with arguments a, b, c.

      def __call__(self, a, b=1, c='hello'):
        # [FUNCTION DOC]
        # Sample call method taking a, b (default 1) and c (default 'hello'); does nothing.
        pass

    argspec = tf_inspect.ArgSpec(
        args=['self', 'a', 'b', 'c'],
        varargs=None,
        keywords=None,
        defaults=(1, 'hello'))

    test_obj = Callable()
    self.assertEqual(argspec, tf_inspect.getargspec(test_obj))

  def testGetArgSpecOnInitClass(self):
    # [FUNCTION DOC]
    # Gets the argspec of a class (not an instance) with a custom constructor and asserts it
    # reflects the constructor: self, a, b, c with defaults 1 and 'hello'.

    class InitClass(object):
      # [CLASS DOC]
      # HELPER: local sample class with a constructor taking a, b, c.

      def __init__(self, a, b=1, c='hello'):
        # [FUNCTION DOC]
        # Sample constructor taking a, b (default 1) and c (default 'hello'); does nothing.
        pass

    argspec = tf_inspect.ArgSpec(
        args=['self', 'a', 'b', 'c'],
        varargs=None,
        keywords=None,
        defaults=(1, 'hello'))

    self.assertEqual(argspec, tf_inspect.getargspec(InitClass))

  def testGetArgSpecOnNewClass(self):
    # [FUNCTION DOC]
    # Gets the argspec of a class that defines only an instance-creation method and asserts
    # it reflects that method: cls, a, b, c with defaults 1 and 'hello'.

    class NewClass(object):
      # [CLASS DOC]
      # HELPER: local sample class defining an instance-creation method taking a, b, c.

      def __new__(cls, a, b=1, c='hello'):
        # [FUNCTION DOC]
        # Sample instance-creation method taking cls, a, b (default 1), c (default 'hello').
        # Does nothing and returns nothing (so it would not produce an instance if called).
        pass

    argspec = tf_inspect.ArgSpec(
        args=['cls', 'a', 'b', 'c'],
        varargs=None,
        keywords=None,
        defaults=(1, 'hello'))

    self.assertEqual(argspec, tf_inspect.getargspec(NewClass))

  def testGetFullArgSpecOnDecoratorsThatDontProvideFullArgSpec(self):
    # [FUNCTION DOC]
    # Gets the full argspec of a decorated function whose decorator declares none and asserts
    # args a, b, c and defaults 2 and 'Hello' from the underlying function.
    argspec = tf_inspect.getfullargspec(test_decorated_function_with_defaults)
    self.assertEqual(['a', 'b', 'c'], argspec.args)
    self.assertEqual((2, 'Hello'), argspec.defaults)

  def testGetFullArgSpecOnDecoratorThatChangesFullArgSpec(self):
    # [FUNCTION DOC]
    # Wraps a function in a TFDecorator declaring its own full spec and asserts the lookup
    # returns exactly the declared spec.
    argspec = tf_inspect.FullArgSpec(
        args=['a', 'b', 'c'],
        varargs=None,
        varkw=None,
        defaults=(1, 'hello'),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})

    decorator = tf_decorator.TFDecorator('', test_undecorated_function, '',
                                         argspec)
    self.assertEqual(argspec, tf_inspect.getfullargspec(decorator))

  def testGetFullArgSpecIgnoresDecoratorsThatDontProvideFullArgSpec(self):
    # [FUNCTION DOC]
    # Places a decorator without a declared spec around one with a declared spec and asserts
    # the full-argspec lookup on the outer one returns the inner declared spec.
    argspec = tf_inspect.FullArgSpec(
        args=['a', 'b', 'c'],
        varargs=None,
        varkw=None,
        defaults=(1, 'hello'),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})

    inner_decorator = tf_decorator.TFDecorator('', test_undecorated_function,
                                               '', argspec)
    outer_decorator = tf_decorator.TFDecorator('', inner_decorator)
    self.assertEqual(argspec, tf_inspect.getfullargspec(outer_decorator))

  def testGetFullArgSpecReturnsOutermostDecoratorThatChangesFullArgSpec(self):
    # [FUNCTION DOC]
    # Nests two decorators with different declared specs and asserts the lookup returns the
    # spec of the outermost decorator.
    outer_argspec = tf_inspect.FullArgSpec(
        args=['a'],
        varargs=None,
        varkw=None,
        defaults=None,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})
    inner_argspec = tf_inspect.FullArgSpec(
        args=['b'],
        varargs=None,
        varkw=None,
        defaults=None,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})

    inner_decorator = tf_decorator.TFDecorator('', test_undecorated_function,
                                               '', inner_argspec)
    outer_decorator = tf_decorator.TFDecorator('', inner_decorator, '',
                                               outer_argspec)
    self.assertEqual(outer_argspec, tf_inspect.getfullargspec(outer_decorator))

  def testGetFullArgsSpecForPartial(self):
    # [FUNCTION DOC]
    # Binds the first positional argument of a two-argument function in a partial and asserts
    # the full argspec lists only b, with no varargs, keywords, defaults or keyword-only
    # arguments.

    def func(a, b):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; deletes its arguments and returns nothing.
      del a, b

    partial_function = functools.partial(func, 1)
    argspec = tf_inspect.FullArgSpec(
        args=['b'],
        varargs=None,
        varkw=None,
        defaults=None,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})

    self.assertEqual(argspec, tf_inspect.getfullargspec(partial_function))

  def testGetFullArgSpecOnPartialNoArgumentsLeft(self):
    # [FUNCTION DOC]
    # Binds both arguments in a partial and asserts the full argspec has an empty argument
    # list and no other components.
    """Tests getfullargspec on partial function that prunes all arguments."""

    def func(m, n):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns twice m plus n.
      return 2 * m + n

    partial_func = functools.partial(func, 7, 10)
    argspec = tf_inspect.FullArgSpec(
        args=[],
        varargs=None,
        varkw=None,
        defaults=None,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})

    self.assertEqual(argspec, tf_inspect.getfullargspec(partial_func))

  def testGetFullArgSpecOnPartialWithVarargs(self):
    # [FUNCTION DOC]
    # Partial of a function with variable positional arguments, binding two values; asserts
    # the full argspec has no named args and varargs name 'arg'.
    """Tests getfullargspec on partial function with variable arguments."""

    def func(m, *arg):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns m plus the count of extra arguments.
      return m + len(arg)

    partial_func = functools.partial(func, 7, 8)
    argspec = tf_inspect.FullArgSpec(
        args=[],
        varargs='arg',
        varkw=None,
        defaults=None,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})

    self.assertEqual(argspec, tf_inspect.getfullargspec(partial_func))

  def testGetFullArgSpecOnPartialWithVarkwargs(self):
    # [FUNCTION DOC]
    # Partial of a function with variable keyword arguments binding the first argument;
    # asserts the full argspec lists n and has varkw name 'kwarg'.
    """Tests getfullargspec.

    Tests on partial function with variable keyword arguments.
    """

    def func(m, n, **kwarg):
      # [FUNCTION DOC]
      # HELPER: target function for the partial; returns m times n plus the keyword count.
      return m * n + len(kwarg)

    partial_func = functools.partial(func, 7)
    argspec = tf_inspect.FullArgSpec(
        args=['n'],
        varargs=None,
        varkw='kwarg',
        defaults=None,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})

    self.assertEqual(argspec, tf_inspect.getfullargspec(partial_func))

  def testGetFullArgSpecOnCallableObject(self):
    # [FUNCTION DOC]
    # Gets the full argspec of a callable object instance and asserts self, a, b, c with
    # defaults 1 and 'hello'.

    class Callable(object):
      # [CLASS DOC]
      # HELPER: local sample class whose instances are callable with arguments a, b, c.

      def __call__(self, a, b=1, c='hello'):
        # [FUNCTION DOC]
        # Sample call method taking a, b (default 1) and c (default 'hello'); does nothing.
        pass

    argspec = tf_inspect.FullArgSpec(
        args=['self', 'a', 'b', 'c'],
        varargs=None,
        varkw=None,
        defaults=(1, 'hello'),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})

    test_obj = Callable()
    self.assertEqual(argspec, tf_inspect.getfullargspec(test_obj))

  def testGetFullArgSpecOnInitClass(self):
    # [FUNCTION DOC]
    # Gets the full argspec of a class with a custom constructor and asserts it reflects the
    # constructor: self, a, b, c with defaults 1 and 'hello'.

    class InitClass(object):
      # [CLASS DOC]
      # HELPER: local sample class with a constructor taking a, b, c.

      def __init__(self, a, b=1, c='hello'):
        # [FUNCTION DOC]
        # Sample constructor taking a, b (default 1) and c (default 'hello'); does nothing.
        pass

    argspec = tf_inspect.FullArgSpec(
        args=['self', 'a', 'b', 'c'],
        varargs=None,
        varkw=None,
        defaults=(1, 'hello'),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})

    self.assertEqual(argspec, tf_inspect.getfullargspec(InitClass))

  def testGetFullArgSpecOnNewClass(self):
    # [FUNCTION DOC]
    # Gets the full argspec of a class that defines only an instance-creation method and
    # asserts cls, a, b, c with defaults 1 and 'hello'.

    class NewClass(object):
      # [CLASS DOC]
      # HELPER: local sample class defining an instance-creation method taking a, b, c.

      def __new__(cls, a, b=1, c='hello'):
        # [FUNCTION DOC]
        # Sample instance-creation method taking cls, a, b (default 1), c (default 'hello').
        # Does nothing and returns nothing.
        pass

    argspec = tf_inspect.FullArgSpec(
        args=['cls', 'a', 'b', 'c'],
        varargs=None,
        varkw=None,
        defaults=(1, 'hello'),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})

    self.assertEqual(argspec, tf_inspect.getfullargspec(NewClass))

  def testSignatureOnDecoratorsThatDontProvideFullArgSpec(self):
    # [FUNCTION DOC]
    # Gets the signature of a decorated function and asserts its parameters are a (required),
    # b (default 2) and c (default 'Hello'), all positional-or-keyword.
    signature = tf_inspect.signature(test_decorated_function_with_defaults)

    self.assertEqual([
        tf_inspect.Parameter('a', tf_inspect.Parameter.POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter(
            'b', tf_inspect.Parameter.POSITIONAL_OR_KEYWORD, default=2),
        tf_inspect.Parameter(
            'c', tf_inspect.Parameter.POSITIONAL_OR_KEYWORD, default='Hello')
    ], list(signature.parameters.values()))

  def testSignatureFollowsNestedDecorators(self):
    # [FUNCTION DOC]
    # Gets the signature of a function wrapped by three decorators and asserts the single
    # parameter x of the innermost function is reported.
    signature = tf_inspect.signature(test_decorated_function)

    self.assertEqual(
        [tf_inspect.Parameter('x', tf_inspect.Parameter.POSITIONAL_OR_KEYWORD)],
        list(signature.parameters.values()))

  def testGetDoc(self):
    # [FUNCTION DOC]
    # Asserts that the doc lookup on a decorated function returns the underlying function's
    # docstring text.
    self.assertEqual('Test Decorated Function With Defaults Docstring.',
                     tf_inspect.getdoc(test_decorated_function_with_defaults))

  def testGetFile(self):
    # [FUNCTION DOC]
    # Asserts the file lookup on a decorated function names this test file, and that for a
    # decorator applied to the decorator-unwrap utility it names the tf_decorator file
    # (i.e. the decorator wrapper is looked through).
    self.assertTrue('tf_inspect_test.py' in tf_inspect.getfile(
        test_decorated_function_with_defaults))
    self.assertTrue('tf_decorator.py' in tf_inspect.getfile(
        test_decorator('decorator')(tf_decorator.unwrap)))

  def testGetMembers(self):
    # [FUNCTION DOC]
    # Asserts the members listing of the decorated class equals the standard library result.
    self.assertEqual(
        inspect.getmembers(TestDecoratedClass),
        tf_inspect.getmembers(TestDecoratedClass))

  def testGetModule(self):
    # [FUNCTION DOC]
    # Asserts the module lookup matches the standard library for the decorated class, a
    # decorated function and an undecorated function.
    self.assertEqual(
        inspect.getmodule(TestDecoratedClass),
        tf_inspect.getmodule(TestDecoratedClass))
    self.assertEqual(
        inspect.getmodule(test_decorated_function),
        tf_inspect.getmodule(test_decorated_function))
    self.assertEqual(
        inspect.getmodule(test_undecorated_function),
        tf_inspect.getmodule(test_undecorated_function))

  def testGetSource(self):
    # [FUNCTION DOC]
    # Asserts the source lookup on a decorated function returns the exact source text of the
    # underlying function including its decorator line. The expected text is a literal copy
    # of the sample function's source, so that sample function must stay textually unchanged
    # (it is therefore left without a documentation block).
    expected = '''@test_decorator('decorator')
def test_decorated_function_with_defaults(a, b=2, c='Hello'):
  """Test Decorated Function With Defaults Docstring."""
  return [a, b, c]
'''
    self.assertEqual(
        expected, tf_inspect.getsource(test_decorated_function_with_defaults))

  def testGetSourceFile(self):
    # [FUNCTION DOC]
    # Asserts the source-file lookup on a decorated function equals this test module's file.
    self.assertEqual(
        __file__,
        tf_inspect.getsourcefile(test_decorated_function_with_defaults))

  def testGetSourceLines(self):
    # [FUNCTION DOC]
    # Asserts the source-lines lookup on a decorated function equals the standard library
    # result for the function wrapped inside the decorator.
    expected = inspect.getsourcelines(
        test_decorated_function_with_defaults.decorated_target)
    self.assertEqual(
        expected,
        tf_inspect.getsourcelines(test_decorated_function_with_defaults))

  def testIsBuiltin(self):
    # [FUNCTION DOC]
    # Asserts the built-in predicate agrees with the standard library for the decorated
    # class, decorated and undecorated functions, and the built-in range and max.
    self.assertEqual(
        tf_inspect.isbuiltin(TestDecoratedClass),
        inspect.isbuiltin(TestDecoratedClass))
    self.assertEqual(
        tf_inspect.isbuiltin(test_decorated_function),
        inspect.isbuiltin(test_decorated_function))
    self.assertEqual(
        tf_inspect.isbuiltin(test_undecorated_function),
        inspect.isbuiltin(test_undecorated_function))
    self.assertEqual(tf_inspect.isbuiltin(range), inspect.isbuiltin(range))
    self.assertEqual(tf_inspect.isbuiltin(max), inspect.isbuiltin(max))

  def testIsClass(self):
    # [FUNCTION DOC]
    # Asserts the class predicate is true for the decorated class and false for a decorated
    # function.
    self.assertTrue(tf_inspect.isclass(TestDecoratedClass))
    self.assertFalse(tf_inspect.isclass(test_decorated_function))

  def testIsFunction(self):
    # [FUNCTION DOC]
    # Asserts the function predicate is true for a decorated function and false for the
    # decorated class.
    self.assertTrue(tf_inspect.isfunction(test_decorated_function))
    self.assertFalse(tf_inspect.isfunction(TestDecoratedClass))

  def testIsMethod(self):
    # [FUNCTION DOC]
    # Asserts the method predicate is true for a bound method of the decorated class and
    # false for a decorated plain function.
    self.assertTrue(tf_inspect.ismethod(TestDecoratedClass().two))
    self.assertFalse(tf_inspect.ismethod(test_decorated_function))

  def testIsModule(self):
    # [FUNCTION DOC]
    # Asserts the module predicate is true for the module of the current frame and false for
    # a decorated function.
    self.assertTrue(
        tf_inspect.ismodule(inspect.getmodule(inspect.currentframe())))
    self.assertFalse(tf_inspect.ismodule(test_decorated_function))

  def testIsRoutine(self):
    # [FUNCTION DOC]
    # Asserts the routine predicate is true for the built-in len and false for the decorated
    # class.
    self.assertTrue(tf_inspect.isroutine(len))
    self.assertFalse(tf_inspect.isroutine(TestDecoratedClass))

  def testStack(self):
    # [FUNCTION DOC]
    # Compares the standard stack with the tf_inspect stack taken on consecutive lines: same
    # depth, same top frame object, filename and function name, a line number one greater
    # for the second call (relative offset, so independent of absolute line numbers), and
    # identical remaining frames.
    expected_stack = inspect.stack()
    actual_stack = tf_inspect.stack()
    self.assertEqual(len(expected_stack), len(actual_stack))
    self.assertEqual(expected_stack[0][0], actual_stack[0][0])  # Frame object
    self.assertEqual(expected_stack[0][1], actual_stack[0][1])  # Filename
    self.assertEqual(expected_stack[0][2],
                     actual_stack[0][2] - 1)  # Line number
    self.assertEqual(expected_stack[0][3], actual_stack[0][3])  # Function name
    self.assertEqual(expected_stack[1:], actual_stack[1:])

  def testIsAnyTargetMethod(self):
    # [FUNCTION DOC]
    # Checks the any-target-is-method predicate: true for an object with a call method, a
    # bound method, and that method wrapped in a partial and in nested test decorators; false
    # for an object without one, a plain function (also partial/decorated), a lambda, the
    # null value and an integer.
    class MyModule:
      # [CLASS DOC]
      # HELPER: local sample class with an ordinary method and a call method.

      def f(self, a):
        # [FUNCTION DOC]
        # Sample method taking one argument; does nothing.
        pass

      def __call__(self):
        # [FUNCTION DOC]
        # Sample call method with no arguments; does nothing.
        pass

    module = MyModule()
    self.assertTrue(tf_inspect.isanytargetmethod(module))
    f = module.f
    self.assertTrue(tf_inspect.isanytargetmethod(f))
    f = functools.partial(f, 1)
    self.assertTrue(tf_inspect.isanytargetmethod(f))
    f = test_decorator('tf_decorator1')(f)
    self.assertTrue(tf_inspect.isanytargetmethod(f))
    f = test_decorator('tf_decorator2')(f)
    self.assertTrue(tf_inspect.isanytargetmethod(f))

    class MyModule2:
      # [CLASS DOC]
      # HELPER: local sample class with no methods, used as a negative case.
      pass
    module = MyModule2()
    self.assertFalse(tf_inspect.isanytargetmethod(module))
    def f2(x):
      # [FUNCTION DOC]
      # HELPER: plain function used as a negative case; deletes its argument, returns nothing.
      del x
    self.assertFalse(tf_inspect.isanytargetmethod(f2))
    f2 = functools.partial(f2, 1)
    self.assertFalse(tf_inspect.isanytargetmethod(f2))
    f2 = test_decorator('tf_decorator1')(f2)
    self.assertFalse(tf_inspect.isanytargetmethod(f2))
    f2 = test_decorator('tf_decorator2')(f2)
    self.assertFalse(tf_inspect.isanytargetmethod(f2))
    self.assertFalse(tf_inspect.isanytargetmethod(lambda: None))
    self.assertFalse(tf_inspect.isanytargetmethod(None))
    self.assertFalse(tf_inspect.isanytargetmethod(1))


class TfInspectGetCallArgsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the tf_inspect call-arguments helper: it maps argument names to the
  # values bound for a call, for unbound functions, bound methods, class methods and
  # decorated functions, with positional, keyword and default values.

  def testReturnsEmptyWhenUnboundFuncHasNoParameters(self):
    # [FUNCTION DOC]
    # Asserts the call-arguments mapping for a function with no parameters is empty.

    def empty():
      # [FUNCTION DOC]
      # HELPER: parameterless sample function; does nothing.
      pass

    self.assertEqual({}, tf_inspect.getcallargs(empty))

  def testClashingParameterNames(self):
    # [FUNCTION DOC]
    # Uses a function whose parameter names clash with names used inside the helper (func,
    # kwargs, func_and_positional); asserts the mapping is correct with defaults and when the
    # clashing names are supplied as keyword values.

    def func(positional, func=1, func_and_positional=2, kwargs=3):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its four arguments as a tuple.
      return positional, func, func_and_positional, kwargs

    kwargs = {}
    self.assertEqual(
        tf_inspect.getcallargs(func, 0, **kwargs), {
            'positional': 0,
            'func': 1,
            'func_and_positional': 2,
            'kwargs': 3
        })
    kwargs = dict(func=4, func_and_positional=5, kwargs=6)
    self.assertEqual(
        tf_inspect.getcallargs(func, 0, **kwargs), {
            'positional': 0,
            'func': 4,
            'func_and_positional': 5,
            'kwargs': 6
        })

  def testUnboundFuncWithOneParamPositional(self):
    # [FUNCTION DOC]
    # Calls with one positional value for a one-parameter function; asserts a maps to 5.

    def func(a):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its argument.
      return a

    self.assertEqual({'a': 5}, tf_inspect.getcallargs(func, 5))

  def testUnboundFuncWithTwoParamsPositional(self):
    # [FUNCTION DOC]
    # Two positional values for a two-parameter function; asserts a maps to 10 and b to 20.

    def func(a, b):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its two arguments as a tuple.
      return (a, b)

    self.assertEqual({'a': 10, 'b': 20}, tf_inspect.getcallargs(func, 10, 20))

  def testUnboundFuncWithOneParamKeyword(self):
    # [FUNCTION DOC]
    # Supplies the single parameter by keyword; asserts a maps to 5.

    def func(a):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its argument.
      return a

    self.assertEqual({'a': 5}, tf_inspect.getcallargs(func, a=5))

  def testUnboundFuncWithTwoParamsKeyword(self):
    # [FUNCTION DOC]
    # Supplies both parameters by keyword; asserts a maps to 6 and b to 7.

    def func(a, b):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its two arguments as a tuple.
      return (a, b)

    self.assertEqual({'a': 6, 'b': 7}, tf_inspect.getcallargs(func, a=6, b=7))

  def testUnboundFuncWithOneParamDefault(self):
    # [FUNCTION DOC]
    # Calls with no values for a function whose parameter has a default; asserts a maps to
    # the default 13.

    def func(a=13):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its argument.
      return a

    self.assertEqual({'a': 13}, tf_inspect.getcallargs(func))

  def testUnboundFuncWithOneParamDefaultOnePositional(self):
    # [FUNCTION DOC]
    # Overrides a defaulted parameter with a positional value; asserts a maps to 1.

    def func(a=0):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its argument.
      return a

    self.assertEqual({'a': 1}, tf_inspect.getcallargs(func, 1))

  def testUnboundFuncWithTwoParamsDefaultOnePositional(self):
    # [FUNCTION DOC]
    # One positional value for two defaulted parameters; asserts a is 5 and b stays 2.

    def func(a=1, b=2):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its two arguments as a tuple.
      return (a, b)

    self.assertEqual({'a': 5, 'b': 2}, tf_inspect.getcallargs(func, 5))

  def testUnboundFuncWithTwoParamsDefaultTwoPositional(self):
    # [FUNCTION DOC]
    # Two positional values override both defaults; asserts a is 3 and b is 4.

    def func(a=1, b=2):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its two arguments as a tuple.
      return (a, b)

    self.assertEqual({'a': 3, 'b': 4}, tf_inspect.getcallargs(func, 3, 4))

  def testUnboundFuncWithOneParamDefaultOneKeyword(self):
    # [FUNCTION DOC]
    # Overrides a defaulted parameter by keyword; asserts a maps to 3.

    def func(a=1):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its argument.
      return a

    self.assertEqual({'a': 3}, tf_inspect.getcallargs(func, a=3))

  def testUnboundFuncWithTwoParamsDefaultOneKeywordFirst(self):
    # [FUNCTION DOC]
    # Overrides only the first of two defaulted parameters by keyword; asserts a is 3, b is 2.

    def func(a=1, b=2):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its two arguments as a tuple.
      return (a, b)

    self.assertEqual({'a': 3, 'b': 2}, tf_inspect.getcallargs(func, a=3))

  def testUnboundFuncWithTwoParamsDefaultOneKeywordSecond(self):
    # [FUNCTION DOC]
    # Overrides only the second of two defaulted parameters by keyword; asserts a is 1, b is 4.

    def func(a=1, b=2):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its two arguments as a tuple.
      return (a, b)

    self.assertEqual({'a': 1, 'b': 4}, tf_inspect.getcallargs(func, b=4))

  def testUnboundFuncWithTwoParamsDefaultTwoKeywords(self):
    # [FUNCTION DOC]
    # Overrides both defaulted parameters by keyword; asserts a is 3 and b is 4.

    def func(a=1, b=2):
      # [FUNCTION DOC]
      # HELPER: sample function; returns its two arguments as a tuple.
      return (a, b)

    self.assertEqual({'a': 3, 'b': 4}, tf_inspect.getcallargs(func, a=3, b=4))

  def testBoundFuncWithOneParam(self):
    # [FUNCTION DOC]
    # Gets the call arguments for a bound method taking only self; asserts the mapping holds
    # self mapped to the instance.

    class Test(object):
      # [CLASS DOC]
      # HELPER: local sample class with one method that takes no arguments besides self.

      def bound(self):
        # [FUNCTION DOC]
        # Sample method; does nothing.
        pass

    t = Test()
    self.assertEqual({'self': t}, tf_inspect.getcallargs(t.bound))

  def testBoundFuncWithManyParamsAndDefaults(self):
    # [FUNCTION DOC]
    # Calls a bound method with one positional and one keyword value; asserts the mapping has
    # self as the instance, a as 3, b as its default 2 and c as 'Goodbye'.

    class Test(object):
      # [CLASS DOC]
      # HELPER: local sample class with a method taking a, b (default 2) and c (default 'Hello').

      def bound(self, a, b=2, c='Hello'):
        # [FUNCTION DOC]
        # Sample method; returns a, b and c as a tuple.
        return (a, b, c)

    t = Test()
    self.assertEqual({
        'self': t,
        'a': 3,
        'b': 2,
        'c': 'Goodbye'
    }, tf_inspect.getcallargs(t.bound, 3, c='Goodbye'))

  def testClassMethod(self):
    # [FUNCTION DOC]
    # Calls a class method accessed through the class; asserts the mapping has cls as the
    # class, a as 5, b as default 3 and c as 'goodbye'.

    class Test(object):
      # [CLASS DOC]
      # HELPER: local sample class with a class method taking a, b (default 3), c (default
      # 'hello').

      @classmethod
      def test(cls, a, b=3, c='hello'):
        # [FUNCTION DOC]
        # Sample class method; returns a, b and c as a tuple.
        return (a, b, c)

    self.assertEqual({
        'cls': Test,
        'a': 5,
        'b': 3,
        'c': 'goodbye'
    }, tf_inspect.getcallargs(Test.test, 5, c='goodbye'))

  def testUsesOutermostDecoratorsArgSpec(self):
    # [FUNCTION DOC]
    # Builds a decorated function via the decorator-making utility with a declared argument
    # spec (a, b, c; defaults 3 and 'hello') differing from the real parameterless function,
    # then asserts the call-arguments mapping follows the declared spec (a 4, b 3, c 'goodbye').

    def func():
      # [FUNCTION DOC]
      # HELPER: parameterless sample target being decorated; does nothing.
      pass

    def wrapper(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: wrapper passed to the decorator maker; forwards all arguments to the target
      # function and returns its result.
      return func(*args, **kwargs)

    decorated = tf_decorator.make_decorator(
        func,
        wrapper,
        decorator_argspec=tf_inspect.FullArgSpec(
            args=['a', 'b', 'c'],
            varargs=None,
            kwonlyargs={},
            defaults=(3, 'hello'),
            kwonlydefaults=None,
            varkw=None,
            annotations=None))

    self.assertEqual({
        'a': 4,
        'b': 3,
        'c': 'goodbye'
    }, tf_inspect.getcallargs(decorated, 4, c='goodbye'))


if __name__ == '__main__':
  test.main()
