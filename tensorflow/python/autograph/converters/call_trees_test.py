# [MODULE DOC]
# Tests for the AutoGraph call-trees converter. Each test converts a small function with a mocked
# runtime converted-call function (which records the positional and keyword arguments of every
# call and then performs the call) and checks both the final result and the recorded calls:
# argument forms (none, plain, keyword, starred, double-starred, mixed), nested calls, calls
# inside lambdas, debugger break calls, and methods of local classes.
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
"""Tests for call_trees module."""

import types

from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class MockConvertedCall(object):
  # [CLASS DOC]
  # HELPER: test double standing in for the runtime converted-call function. Holds a list of
  # recorded (positional arguments, keyword arguments) pairs. Callable like the real function.

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes an empty list of recorded calls.
    self.calls = []

  def __call__(self, f, args, kwargs, caller_fn_scope=None, options=None):
    # [FUNCTION DOC]
    # Records and performs a call. Inputs: f (callee), args (tuple of positional arguments), kwargs
    # (mapping or null of keyword arguments), and the unused optional caller scope and options
    # (discarded). Appends the (args, kwargs) pair to the call list, then calls f with them (null
    # kwargs treated as empty) and returns its result.
    del caller_fn_scope, options
    self.calls.append((args, kwargs))
    kwargs = kwargs or {}
    return f(*args, **kwargs)


class CallTreesTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the call-trees converter, built on the converter testing base class. Shared
  # fixture: a helper that converts a function with the function and call-tree converters while
  # overriding the runtime converted-call with a recording mock.

  def _transform_with_mock(self, f):
    # [FUNCTION DOC]
    # HELPER: shared by the tests. Creates a call-recording mock, converts function f with the
    # functions and call-trees converters overriding converted-call with the mock, and returns a
    # pair of the converted function and the mock.
    mock = MockConvertedCall()
    tr = self.transform(
        f, (functions, call_trees),
        ag_overrides={'converted_call': mock})
    return tr, mock

  def test_function_no_args(self):
    # [FUNCTION DOC]
    # Calling a callable argument with no arguments: result is 21 and one call with empty
    # positional arguments and null keywords is recorded.

    def f(f):
      # [FUNCTION DOC]
      # Sample function: calls its argument with no arguments and adds 20.
      return f() + 20

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda: 1), 21)
    self.assertListEqual(mock.calls, [((), None)])

  def test_function_with_expression_in_argument(self):
    # [FUNCTION DOC]
    # An argument that is itself an expression with a call: result is 4321; two calls are
    # recorded, the inner no-argument call first, then the outer call with value 21.

    def f(f, g):
      # [FUNCTION DOC]
      # Sample function: calls f with (call of g plus 20), adds 4000.
      return f(g() + 20) + 4000

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda x: x + 300, lambda: 1), 4321)
    self.assertListEqual(mock.calls, [
        ((), None),
        ((21,), None),
    ])

  def test_function_with_call_in_argument(self):
    # [FUNCTION DOC]
    # A call used directly as an argument: result is 321; recorded calls are the inner
    # no-argument call then the outer call with value 1.

    def f(f, g):
      # [FUNCTION DOC]
      # Sample function: calls f with the result of calling g, adds 300.
      return f(g()) + 300

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda x: x + 20, lambda: 1), 321)
    self.assertListEqual(mock.calls, [
        ((), None),
        ((1,), None),
    ])

  def test_function_chaining(self):
    # [FUNCTION DOC]
    # Chained call on a call result (addition method on the result of a call): result is 21;
    # recorded calls are the no-argument call, then the method call with argument 20.

    def get_one():
      # [FUNCTION DOC]
      # Sample helper returning 1.
      return 1

    def f():
      # [FUNCTION DOC]
      # Sample function: calls get_one and invokes the addition method on the result with 20.
      return get_one().__add__(20)

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(), 21)
    self.assertListEqual(mock.calls, [
        ((), None),
        ((20,), None),
    ])

  def test_function_with_single_arg(self):
    # [FUNCTION DOC]
    # Call with one positional argument: result is 21; one call with arguments (1) recorded.

    def f(f, a):
      # [FUNCTION DOC]
      # Sample function: calls f with a and adds 20.
      return f(a) + 20

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda a: a, 1), 21)
    self.assertListEqual(mock.calls, [((1,), None)])

  def test_function_with_args_only(self):
    # [FUNCTION DOC]
    # Call with two positional arguments: result is 321; one call with arguments (1, 20) recorded
    # with null keywords.

    def f(f, a, b):
      # [FUNCTION DOC]
      # Sample function: calls f with a and b, adds 300.
      return f(a, b) + 300

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda a, b: a + b, 1, 20), 321)
    self.assertListEqual(mock.calls, [((1, 20), None)])

  def test_function_with_kwarg(self):
    # [FUNCTION DOC]
    # Call with one positional and one keyword argument: result is 321; recorded call has
    # arguments (1) and keywords mapping c to 20.

    def f(f, a, b):
      # [FUNCTION DOC]
      # Sample function: calls f with a and keyword c set to b, adds 300.
      return f(a, c=b) + 300

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda a, c: a + c, 1, 20), 321)
    self.assertListEqual(mock.calls, [((1,), {'c': 20})])

  def test_function_with_kwargs_starargs(self):
    # [FUNCTION DOC]
    # Forwarding variadic positional and keyword arguments: result is 12; recorded call has
    # positional (1, 2, 3) and keywords b=4, c=5.

    def f(f, a, *args, **kwargs):
      # [FUNCTION DOC]
      # Sample function: calls f forwarding a, extra positionals and keywords, adds 5.
      return f(a, *args, **kwargs) + 5

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(
        tr(lambda *args, **kwargs: 7, 1, *[2, 3], **{
            'b': 4,
            'c': 5
        }), 12)
    self.assertListEqual(mock.calls, [((1, 2, 3), {'b': 4, 'c': 5})])

  def test_function_with_starargs_only(self):
    # [FUNCTION DOC]
    # Call with only an unpacked list: result is 4321; recorded call has positional (1, 20, 300)
    # and null keywords.

    def g(*args):
      # [FUNCTION DOC]
      # Sample helper: returns the sum of its positional arguments.
      return sum(args)

    def f():
      # [FUNCTION DOC]
      # Sample function: unpacks a three-element list into g and adds 4000.
      args = [1, 20, 300]
      return g(*args) + 4000

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(), 4321)
    self.assertListEqual(mock.calls, [((1, 20, 300), None)])

  def test_function_with_starargs_mixed(self):
    # [FUNCTION DOC]
    # Call mixing unpacked sequences with plain arguments: result is 1234; recorded positional
    # arguments are (1, 2, 3, 4) in order.

    def g(a, b, c, d):
      # [FUNCTION DOC]
      # Sample helper: combines four digits into a four-digit number.
      return a * 1000 + b * 100 + c * 10 + d

    def f():
      # [FUNCTION DOC]
      # Sample function: calls g with an unpacked tuple, a plain value, an unpacked list, and a
      # plain value.
      args1 = (1,)
      args2 = [3]
      return g(*args1, 2, *args2, 4)

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(), 1234)
    self.assertListEqual(mock.calls, [((1, 2, 3, 4), None)])

  def test_function_with_kwargs_keywords(self):
    # [FUNCTION DOC]
    # Mixing a plain keyword with an unpacked keyword mapping: result is 12; recorded call has
    # positional (1) and keywords b=2, c=3.

    def f(f, a, b, **kwargs):
      # [FUNCTION DOC]
      # Sample function: calls f with a, keyword b and unpacked keywords, adds 5.
      return f(a, b=b, **kwargs) + 5

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(
        tr(lambda *args, **kwargs: 7, 1, 2, **{'c': 3}), 12)
    self.assertListEqual(mock.calls, [((1,), {'b': 2, 'c': 3})])

  def test_function_with_multiple_kwargs(self):
    # [FUNCTION DOC]
    # Several unpacked keyword mappings interleaved with plain keywords: result is 12; recorded
    # keywords merge to b=2, c=3, d=4, e=5.

    def f(f, a, b, c, kwargs1, kwargs2):
      # [FUNCTION DOC]
      # Sample function: calls f with a, keyword b, a first mapping, keyword c, a second mapping,
      # adds 5.
      return f(a, b=b, **kwargs1, c=c, **kwargs2) + 5

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(
        tr(lambda *args, **kwargs: 7, 1, 2, 3, {'d': 4}, {'e': 5}), 12)
    self.assertListEqual(mock.calls, [((1,), {
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5
    })])

  def test_function_with_call_in_lambda_argument(self):
    # [FUNCTION DOC]
    # A call inside a lambda passed as an argument: converted function called with helpers and
    # arguments 1, 20, 300 must return 4321 (no recorded-call assertion).

    def h(l, a):
      # [FUNCTION DOC]
      # Sample helper: calls callable l with a and adds 4000.
      return l(a) + 4000

    def g(a, *args):
      # [FUNCTION DOC]
      # Sample helper: returns a plus the sum of the extra arguments.
      return a + sum(args)

    def f(h, g, a, *args):
      # [FUNCTION DOC]
      # Sample function: calls h with a lambda that calls g with x and the extra args, and with a.
      return h(lambda x: g(x, *args), a)

    tr, _ = self._transform_with_mock(f)

    self.assertEqual(tr(h, g, 1, *(20, 300)), 4321)

  def test_debugger_set_trace(self):
    # [FUNCTION DOC]
    # A call to a debugger break function must not be wrapped by converted-call. A fake debugger
    # module whose break function appends 1 to a list is used; after running the converted
    # function the list must equal [1].

    tracking_list = []

    pdb = types.ModuleType('fake_pdb')
    pdb.set_trace = lambda: tracking_list.append(1)

    def f():
      # [FUNCTION DOC]
      # Sample function: calls the fake debugger break function and returns its result.
      return pdb.set_trace()

    tr, _ = self._transform_with_mock(f)

    tr()
    self.assertListEqual(tracking_list, [1])

  def test_class_method(self):
    # [FUNCTION DOC]
    # Converts an unbound method of a local class and calls it with an instance and 1: result is
    # 321 and one call with positional (1) is recorded.

    class TestClass(object):
      # [CLASS DOC]
      # Sample local class with a helper method and a method that calls it.

      def other_method(self, x):
        # [FUNCTION DOC]
        # Sample method: returns x plus 20.
        return x + 20

      def test_method(self, a):
        # [FUNCTION DOC]
        # Sample method: calls other_method on itself with a and adds 300.
        return self.other_method(a) + 300

    tc = TestClass()
    tr, mock = self._transform_with_mock(TestClass.test_method)

    self.assertEqual(321, tr(tc, 1))
    self.assertListEqual(mock.calls, [((1,), None)])

  def test_object_method(self):
    # [FUNCTION DOC]
    # Same scenario as the class-method test but converting the method bound to an instance (still
    # called with the instance and 1): result 321 and one call with positional (1) recorded.

    class TestClass(object):
      # [CLASS DOC]
      # Sample local class with a helper method and a method that calls it.

      def other_method(self, x):
        # [FUNCTION DOC]
        # Sample method: returns x plus 20.
        return x + 20

      def test_method(self, a):
        # [FUNCTION DOC]
        # Sample method: calls other_method on itself with a and adds 300.
        return self.other_method(a) + 300

    tc = TestClass()
    tr, mock = self._transform_with_mock(tc.test_method)

    self.assertEqual(321, tr(tc, 1))
    self.assertListEqual(mock.calls, [((1,), None)])


if __name__ == '__main__':
  test.main()
