# [MODULE DOC]
# Unit tests for tf_decorator. Defines fixture decorators (a TFDecorator factory, a decorator that
# increments the first integer argument, and two "injectable" decorators that call through their
# wrapped reference so rewrap works), a plain function, a multiply-decorated function and a
# decorated class. Test cases then cover TFDecorator construction and delegation, make_decorator
# metadata, rewrap target replacement and unwrap behavior. No import-time side effects besides
# defining the fixtures.
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
"""Unit tests for tf_decorator."""

# pylint: disable=unused-import
import functools
import inspect

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect


def test_tfdecorator(decorator_name, decorator_doc=None):
  # [FUNCTION DOC]
  # HELPER: fixture factory for the tests in this module.
  # Takes a decorator name and optional doc and returns a decorator function which wraps its
  # target in a TFDecorator carrying that name and doc. No side effects.

  def make_tf_decorator(target):
    # [FUNCTION DOC]
    # HELPER: the decorator returned by test_tfdecorator. Given a target callable, returns a new
    # TFDecorator with the captured name and doc wrapping it.
    return tf_decorator.TFDecorator(decorator_name, target, decorator_doc)

  return make_tf_decorator


def test_decorator_increment_first_int_arg(target):
  # [FUNCTION DOC]
  # HELPER: fixture decorator. Wraps target so that the first integer positional argument is
  # incremented by one (skipping a leading self object) before the call, and registers the wrapper
  # through make_decorator so the decorator name is this function's name. Returns the wrapper.
  """This test decorator skips past `self` as args[0] in the bound case."""

  def wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: the wrapper built by the increment decorator. Copies the positional arguments,
    # adding one to the first integer found, then calls the target with them and the keyword
    # arguments and returns the target's result.
    new_args = []
    found = False
    for arg in args:
      if not found and isinstance(arg, int):
        new_args.append(arg + 1)
        found = True
      else:
        new_args.append(arg)
    return target(*new_args, **kwargs)

  return tf_decorator.make_decorator(target, wrapper)


def test_injectable_decorator_square(target):
  # [FUNCTION DOC]
  # HELPER: fixture decorator that squares the result of its target. The wrapper reads the target
  # through its own wrapped reference so rewrap can swap the target. Returns the decorated wrapper.

  def wrapper(x):
    # [FUNCTION DOC]
    # HELPER: calls the current wrapped target with x and returns the square of the result.
    return wrapper.__wrapped__(x)**2

  return tf_decorator.make_decorator(target, wrapper)


def test_injectable_decorator_increment(target):
  # [FUNCTION DOC]
  # HELPER: fixture decorator that adds one to the result of its target. The wrapper reads the
  # target through its own wrapped reference so rewrap can swap the target. Returns the wrapper.

  def wrapper(x):
    # [FUNCTION DOC]
    # HELPER: calls the current wrapped target with x and returns that result plus one.
    return wrapper.__wrapped__(x) + 1

  return tf_decorator.make_decorator(target, wrapper)


def test_function(x):
  # [FUNCTION DOC]
  # HELPER: plain undecorated fixture target. Returns its numeric argument x plus one.
  """Test Function Docstring."""
  return x + 1


@test_tfdecorator('decorator 1')
@test_decorator_increment_first_int_arg
@test_tfdecorator('decorator 3', 'decorator 3 documentation')
def test_decorated_function(x):
  # [FUNCTION DOC]
  # HELPER: fixture decorated by three decorators (outer TFDecorator named decorator 1, the
  # increment-first-integer decorator, and an inner TFDecorator named decorator 3 with its own
  # doc). The undecorated body returns x times two; the increment decorator makes the effective
  # result (x plus one) times two.
  """Test Decorated Function Docstring."""
  return x * 2


@test_tfdecorator('decorator')
class TestDecoratedClass(object):
  # [CLASS DOC]
  # HELPER: fixture class wrapped by a TFDecorator. Has an attribute set by the constructor, a
  # property, a plain method and a method decorated with the increment decorator, used to check
  # that decorated classes and bound methods still work and keep their metadata.
  """Test Decorated Class."""

  def __init__(self, two_attr=2):
    # [FUNCTION DOC]
    # Constructor storing the optional integer two_attr (default 2) as an attribute. Returns nothing.
    self.two_attr = two_attr

  @property
  def two_prop(self):
    # [FUNCTION DOC]
    # Property returning the constant 2.
    return 2

  def two_func(self):
    # [FUNCTION DOC]
    # Method returning the constant 2.
    return 2

  @test_decorator_increment_first_int_arg
  def return_params(self, a, b, c):
    # [FUNCTION DOC]
    # Method decorated with the increment decorator, so its first integer argument is incremented.
    # Returns the three arguments as a list.
    """Return parameters."""
    return [a, b, c]


class TfDecoratorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the TFDecorator class: constructor capture of target, name, doc and argspec,
  # copying of name and doc, call delegation, behavior on decorated functions and classes, and
  # proxying of the descriptor hook. Uses the module-level fixtures.

  def testInitCapturesTarget(self):
    # [FUNCTION DOC]
    # Builds a TFDecorator around the fixture function and asserts the decorated target is that
    # same object.
    self.assertIs(test_function,
                  tf_decorator.TFDecorator('', test_function).decorated_target)

  def testInitCapturesDecoratorName(self):
    # [FUNCTION DOC]
    # Builds a TFDecorator with a name and asserts the decorator name property returns it.
    self.assertEqual(
        'decorator name',
        tf_decorator.TFDecorator('decorator name',
                                 test_function).decorator_name)

  def testInitCapturesDecoratorDoc(self):
    # [FUNCTION DOC]
    # Builds a TFDecorator with a doc string and asserts the decorator doc property returns it.
    self.assertEqual(
        'decorator doc',
        tf_decorator.TFDecorator('', test_function,
                                 'decorator doc').decorator_doc)

  def testInitCapturesNonNoneArgspec(self):
    # [FUNCTION DOC]
    # Builds a TFDecorator with an explicit argument-spec record and asserts the argspec property
    # returns that identical object.
    argspec = tf_inspect.FullArgSpec(
        args=['a', 'b', 'c'],
        varargs=None,
        varkw=None,
        defaults=(1, 'hello'),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations=None,
    )
    self.assertIs(
        argspec,
        tf_decorator.TFDecorator('', test_function, '',
                                 argspec).decorator_argspec)

  def testInitSetsDecoratorNameToTargetName(self):
    # [FUNCTION DOC]
    # Asserts the decorator's name attribute equals the target function's name.
    self.assertEqual('test_function',
                     tf_decorator.TFDecorator('', test_function).__name__)

  def testInitSetsDecoratorQualNameToTargetQualName(self):
    # [FUNCTION DOC]
    # If the decorator has a qualified name attribute, asserts it equals the target's name (a
    # module-level function). Does nothing otherwise.
    if hasattr(tf_decorator.TFDecorator('', test_function), '__qualname__'):
      self.assertEqual('test_function',
                       tf_decorator.TFDecorator('', test_function).__qualname__)

  def testInitSetsDecoratorDocToTargetDoc(self):
    # [FUNCTION DOC]
    # Asserts the decorator's doc equals the target function's doc string when no decorator doc
    # is given.
    self.assertEqual('Test Function Docstring.',
                     tf_decorator.TFDecorator('', test_function).__doc__)

  def testCallingATFDecoratorCallsTheTarget(self):
    # [FUNCTION DOC]
    # Calls a TFDecorator around the fixture function with 123 and asserts the result is 124.
    self.assertEqual(124, tf_decorator.TFDecorator('', test_function)(123))

  def testCallingADecoratedFunctionCallsTheTarget(self):
    # [FUNCTION DOC]
    # Calls the triple-decorated fixture with 2 and asserts the result is six, reflecting the
    # increment decorator and the multiply-by-two body.
    self.assertEqual((2 + 1) * 2, test_decorated_function(2))

  def testInitializingDecoratedClassWithInitParamsDoesntRaise(self):
    # [FUNCTION DOC]
    # Instantiates the decorated class with a constructor argument and fails if a type error
    # is raised.
    # KNOWN ISSUE: on failure it calls a nonexistent assertFail method, which would itself raise
    # an attribute error instead of reporting a clean test failure.
    try:
      TestDecoratedClass(2)
    except TypeError:
      self.assertFail()

  def testReadingClassAttributeOnDecoratedClass(self):
    # [FUNCTION DOC]
    # Creates the decorated class and asserts its instance attribute equals 2.
    self.assertEqual(2, TestDecoratedClass().two_attr)

  def testCallingClassMethodOnDecoratedClass(self):
    # [FUNCTION DOC]
    # Creates the decorated class and asserts calling its plain method returns 2.
    self.assertEqual(2, TestDecoratedClass().two_func())

  def testReadingClassPropertyOnDecoratedClass(self):
    # [FUNCTION DOC]
    # Creates the decorated class and asserts its property value is 2.
    self.assertEqual(2, TestDecoratedClass().two_prop)

  def testNameOnBoundProperty(self):
    # [FUNCTION DOC]
    # Asserts the bound decorated method keeps the original method name.
    self.assertEqual('return_params',
                     TestDecoratedClass().return_params.__name__)

  def testQualNameOnBoundProperty(self):
    # [FUNCTION DOC]
    # If the bound decorated method has a qualified name, asserts it is the class-qualified
    # method name.
    if hasattr(TestDecoratedClass().return_params, '__qualname__'):
      self.assertEqual('TestDecoratedClass.return_params',
                       TestDecoratedClass().return_params.__qualname__)

  def testDocstringOnBoundProperty(self):
    # [FUNCTION DOC]
    # Asserts the bound decorated method keeps the original doc string.
    self.assertEqual('Return parameters.',
                     TestDecoratedClass().return_params.__doc__)

  def testTarget__get__IsProxied(self):
    # [FUNCTION DOC]
    # Defines a descriptor class whose get hook returns itself, stores a TFDecorator around an
    # instance of it as a class attribute, and asserts that reading it through the class returns
    # the descriptor instance, proving the get hook is forwarded to the target.

    class Descr(object):
      # [CLASS DOC]
      # HELPER: local descriptor whose get hook returns the descriptor itself; used to check
      # that TFDecorator forwards the descriptor protocol.

      def __get__(self, instance, owner):
        # [FUNCTION DOC]
        # HELPER: descriptor hook returning the descriptor object itself, ignoring its arguments.
        return self

    class Foo(object):
      # [CLASS DOC]
      # HELPER: local class holding a TFDecorator-wrapped descriptor as a class attribute.
      foo = tf_decorator.TFDecorator('Descr', Descr())

    self.assertIsInstance(Foo.foo, Descr)


def test_wrapper(*args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: generic wrapper fixture for make_decorator tests. Forwards all arguments to the
  # fixture function and returns its result.
  return test_function(*args, **kwargs)


class TfMakeDecoratorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for make_decorator: attached decorator record, wrapped reference, name and doc
  # arguments, dictionary copying, argspec override, default naming and handling of callables
  # without name attributes.

  def testAttachesATFDecoratorAttr(self):
    # [FUNCTION DOC]
    # Decorates the fixture and asserts the result carries a decorator record that is a
    # TFDecorator.
    decorated = tf_decorator.make_decorator(test_function, test_wrapper)
    decorator = getattr(decorated, '_tf_decorator')
    self.assertIsInstance(decorator, tf_decorator.TFDecorator)

  def testAttachesWrappedAttr(self):
    # [FUNCTION DOC]
    # Decorates the fixture and asserts the wrapped reference of the result is the original function.
    decorated = tf_decorator.make_decorator(test_function, test_wrapper)
    wrapped_attr = getattr(decorated, '__wrapped__')
    self.assertIs(test_function, wrapped_attr)

  def testSetsTFDecoratorNameToDecoratorNameArg(self):
    # [FUNCTION DOC]
    # Decorates with an explicit decorator name and asserts the record stores that name.
    decorated = tf_decorator.make_decorator(test_function, test_wrapper,
                                            'test decorator name')
    decorator = getattr(decorated, '_tf_decorator')
    self.assertEqual('test decorator name', decorator.decorator_name)

  def testSetsTFDecoratorDocToDecoratorDocArg(self):
    # [FUNCTION DOC]
    # Decorates with an explicit decorator doc and asserts the record stores that doc.
    decorated = tf_decorator.make_decorator(
        test_function, test_wrapper, decorator_doc='test decorator doc')
    decorator = getattr(decorated, '_tf_decorator')
    self.assertEqual('test decorator doc', decorator.decorator_doc)

  def testUpdatesDictWithMissingEntries(self):
    # [FUNCTION DOC]
    # Temporarily sets an extra attribute on the target, decorates, asserts the wrapper received
    # the attribute, then removes it from the target (cleanup of shared fixture state).
    test_function.foobar = True
    decorated = tf_decorator.make_decorator(test_function, test_wrapper)
    self.assertTrue(decorated.foobar)
    del test_function.foobar

  def testUpdatesDict_doesNotOverridePresentEntries(self):
    # [FUNCTION DOC]
    # Sets the same attribute on target (true) and wrapper (false), decorates, and asserts the
    # wrapper keeps its own value. Removes the attributes from both afterwards.
    test_function.foobar = True
    test_wrapper.foobar = False
    decorated = tf_decorator.make_decorator(test_function, test_wrapper)
    self.assertFalse(decorated.foobar)
    del test_function.foobar
    del test_wrapper.foobar

  def testSetsTFDecoratorArgSpec(self):
    # [FUNCTION DOC]
    # Decorates with an explicit argument-spec record and asserts the record stores it and that
    # the resulting signature has parameters a, b (default 1), c (default hello), variable
    # positional args and variable keyword kwargs.
    argspec = tf_inspect.FullArgSpec(
        args=['a', 'b', 'c'],
        varargs='args',
        kwonlyargs={},
        defaults=(1, 'hello'),
        kwonlydefaults=None,
        varkw='kwargs',
        annotations=None)
    decorated = tf_decorator.make_decorator(test_function, test_wrapper, '', '',
                                            argspec)
    decorator = getattr(decorated, '_tf_decorator')
    self.assertEqual(argspec, decorator.decorator_argspec)
    self.assertEqual(
        inspect.signature(decorated),
        inspect.Signature([
            inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                'b', inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1
            ),
            inspect.Parameter(
                'c',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default='hello',
            ),
            inspect.Parameter('args', inspect.Parameter.VAR_POSITIONAL),
            inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD),
        ]),
    )

  def testSetsDecoratorNameToFunctionThatCallsMakeDecoratorIfAbsent(self):
    # [FUNCTION DOC]
    # Calls make_decorator without a name from inside a local function and asserts the recorded
    # decorator name is that local function's name.

    def test_decorator_name(wrapper):
      # [FUNCTION DOC]
      # HELPER: local decorator calling make_decorator without a name, so its own name is
      # recorded. Returns the decorated wrapper.
      return tf_decorator.make_decorator(test_function, wrapper)

    decorated = test_decorator_name(test_wrapper)
    decorator = getattr(decorated, '_tf_decorator')
    self.assertEqual('test_decorator_name', decorator.decorator_name)

  def testCompatibleWithNamelessCallables(self):
    # [FUNCTION DOC]
    # Smoke test: make_decorator must not raise for a callable object lacking a name and for a
    # partial object lacking name, module and doc attributes. Nothing is asserted beyond that.

    class Callable(object):
      # [CLASS DOC]
      # HELPER: local callable class with no name attribute on its instances, used in a smoke test.

      def __call__(self):
        # [FUNCTION DOC]
        # HELPER: does nothing and returns nothing.
        pass

    callable_object = Callable()
    # Smoke test: This should not raise an exception, even though
    # `callable_object` does not have a `__name__` attribute.
    _ = tf_decorator.make_decorator(callable_object, test_wrapper)

    partial = functools.partial(test_function, x=1)
    # Smoke test: This should not raise an exception, even though `partial` does
    # not have `__name__`, `__module__`, and `__doc__` attributes.
    _ = tf_decorator.make_decorator(partial, test_wrapper)


class TfDecoratorRewrapTest(test.TestCase):
  # [CLASS DOC]
  # Tests for rewrap: replacing the innermost target of a decorator chain and rewrapping at the
  # decorator-function level.

  def testRewrapMutatesAffectedFunction(self):
    # [FUNCTION DOC]
    # Builds a function decorated by the increment and square decorators, checks its result, then
    # rewraps it with a new target obtained from unwrap and asserts the result reflects the new
    # target (multiply by three) under the same decorators.

    @test_injectable_decorator_square
    @test_injectable_decorator_increment
    def test_rewrappable_decorated(x):
      # [FUNCTION DOC]
      # HELPER: local target returning x times two, decorated with the injectable decorators.
      return x * 2

    def new_target(x):
      # [FUNCTION DOC]
      # HELPER: replacement target returning x times three.
      return x * 3

    self.assertEqual((1 * 2 + 1)**2, test_rewrappable_decorated(1))
    prev_target, _ = tf_decorator.unwrap(test_rewrappable_decorated)
    tf_decorator.rewrap(test_rewrappable_decorated, prev_target, new_target)
    self.assertEqual((1 * 3 + 1)**2, test_rewrappable_decorated(1))

  def testRewrapOfDecoratorFunction(self):
    # [FUNCTION DOC]
    # Builds a similarly decorated function and rewraps using the target recorded in the outer
    # decorator, so the increment decorator is dropped. Asserts only the square decorator is
    # preserved around the new target.
    # KNOWN ISSUE: the comment in the body says the outer decorator is preserved, which is
    # consistent, but the name suggests the decorator function is rewrapped rather than the target.

    @test_injectable_decorator_square
    @test_injectable_decorator_increment
    def test_rewrappable_decorated(x):
      # [FUNCTION DOC]
      # HELPER: local target returning x times two, decorated with the injectable decorators.
      return x * 2

    def new_target(x):
      # [FUNCTION DOC]
      # HELPER: replacement target returning x times three.
      return x * 3

    prev_target = test_rewrappable_decorated._tf_decorator._decorated_target
    # In this case, only the outer decorator (test_injectable_decorator_square)
    # should be preserved.
    tf_decorator.rewrap(test_rewrappable_decorated, prev_target, new_target)
    self.assertEqual((1 * 3)**2, test_rewrappable_decorated(1))


class TfDecoratorUnwrapTest(test.TestCase):
  # [CLASS DOC]
  # Tests for unwrap: results for undecorated functions, the final target, the list of
  # decorators and its ordering, and bound methods.

  def testUnwrapReturnsEmptyArrayForUndecoratedFunction(self):
    # [FUNCTION DOC]
    # Unwraps an undecorated function and asserts the decorator list is empty.
    decorators, _ = tf_decorator.unwrap(test_function)
    self.assertEqual(0, len(decorators))

  def testUnwrapReturnsUndecoratedFunctionAsTarget(self):
    # [FUNCTION DOC]
    # Unwraps an undecorated function and asserts the returned target is that same function.
    _, target = tf_decorator.unwrap(test_function)
    self.assertIs(test_function, target)

  def testUnwrapReturnsFinalFunctionAsTarget(self):
    # [FUNCTION DOC]
    # Checks the decorated fixture's effective result, unwraps it, and asserts the final target is
    # a plain function that, when called with 4, returns 8 (no decorators applied).
    self.assertEqual((4 + 1) * 2, test_decorated_function(4))
    _, target = tf_decorator.unwrap(test_decorated_function)
    self.assertTrue(tf_inspect.isfunction(target))
    self.assertEqual(4 * 2, target(4))

  def testUnwrapReturnsListOfUniqueTFDecorators(self):
    # [FUNCTION DOC]
    # Unwraps the triple-decorated fixture and asserts three TFDecorator objects are returned and
    # that all three are distinct objects.
    decorators, _ = tf_decorator.unwrap(test_decorated_function)
    self.assertEqual(3, len(decorators))
    self.assertTrue(isinstance(decorators[0], tf_decorator.TFDecorator))
    self.assertTrue(isinstance(decorators[1], tf_decorator.TFDecorator))
    self.assertTrue(isinstance(decorators[2], tf_decorator.TFDecorator))
    self.assertIsNot(decorators[0], decorators[1])
    self.assertIsNot(decorators[1], decorators[2])
    self.assertIsNot(decorators[2], decorators[0])

  def testUnwrapReturnsDecoratorListFromOutermostToInnermost(self):
    # [FUNCTION DOC]
    # Unwraps the triple-decorated fixture and asserts the decorator names appear outermost to
    # innermost and that the innermost doc is the one supplied.
    decorators, _ = tf_decorator.unwrap(test_decorated_function)
    self.assertEqual('decorator 1', decorators[0].decorator_name)
    self.assertEqual('test_decorator_increment_first_int_arg',
                     decorators[1].decorator_name)
    self.assertEqual('decorator 3', decorators[2].decorator_name)
    self.assertEqual('decorator 3 documentation', decorators[2].decorator_doc)

  def testUnwrapBoundMethods(self):
    # [FUNCTION DOC]
    # Calls the decorated bound method (first integer incremented), unwraps it, asserts the
    # decorator name, and asserts calling the undecorated target with the instance returns the
    # unmodified arguments.
    test_decorated_class = TestDecoratedClass()
    self.assertEqual([2, 2, 3], test_decorated_class.return_params(1, 2, 3))
    decorators, target = tf_decorator.unwrap(test_decorated_class.return_params)
    self.assertEqual('test_decorator_increment_first_int_arg',
                     decorators[0].decorator_name)
    self.assertEqual([1, 2, 3], target(test_decorated_class, 1, 2, 3))


if __name__ == '__main__':
  test.main()
