# [MODULE DOC]
# Unit tests for the tf_contextlib module: checks that the TensorFlow-aware context manager
# decorator runs code before and after the yield, supports nesting, returns the yielded value,
# and that the result is introspectable through tf_decorator (unwrap) and tf_inspect (argspec).
# Defines three small decorated generator/stub helpers at import time and runs the test main.
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
"""Unit tests for tf_contextlib."""

# pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect


@tf_contextlib.contextmanager
def test_yield_append_before_and_after_yield(x, before, after):
  # [FUNCTION DOC]
  # HELPER: context manager fixture for the tests in this module.
  # Appends the given before value to the list x, yields control to the with-body, and after the
  # body finishes appends the after value to x. Returns nothing; mutates x.
  x.append(before)
  yield
  x.append(after)


@tf_contextlib.contextmanager
def test_yield_return_x_plus_1(x):
  # [FUNCTION DOC]
  # HELPER: context manager fixture for the tests in this module.
  # Yields the number x plus one, so the with-statement target is x plus one. No side effects.
  yield x + 1


@tf_contextlib.contextmanager
def test_params_and_defaults(a, b=2, c=True, d='hello'):
  # [FUNCTION DOC]
  # HELPER: fixture used to check that signature introspection sees through the decorator.
  # KNOWN ISSUE: despite being wrapped by the context manager decorator it is not a generator; it
  # returns a list of its four arguments (a, b, c, d), so it could not be used in a with statement.
  # Parameters a, b (default 2), c (default true) and d (default hello) are simply echoed back.
  return [a, b, c, d]


class TfContextlibTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering tf_contextlib.contextmanager: ordering of code around the yield, nesting,
  # repeated use, yielded values, unwrapping of the decorator and argument-spec reporting.
  # Uses the module-level decorated helper functions as fixtures.

  def testRunsCodeBeforeYield(self):
    # [FUNCTION DOC]
    # Enters the append-before/after context manager with a before marker and checks inside the
    # body that the last list entry is the before marker. Asserts equality only.
    x = []
    with test_yield_append_before_and_after_yield(x, 'before', ''):
      self.assertEqual('before', x[-1])

  def testRunsCodeAfterYield(self):
    # [FUNCTION DOC]
    # Enters and leaves the append-before/after context manager with an after marker, then
    # asserts the last list entry is the after marker, proving code after the yield ran on exit.
    x = []
    with test_yield_append_before_and_after_yield(x, '', 'after'):
      pass
    self.assertEqual('after', x[-1])

  def testNestedWith(self):
    # [FUNCTION DOC]
    # Nests two append-before/after managers and a plus-one manager, appending the yielded value
    # in the innermost body. Asserts the list order is before, inner, yielded value, outer, after.
    x = []
    with test_yield_append_before_and_after_yield(x, 'before', 'after'):
      with test_yield_append_before_and_after_yield(x, 'inner', 'outer'):
        with test_yield_return_x_plus_1(1) as var:
          x.append(var)
    self.assertEqual(['before', 'inner', 2, 'outer', 'after'], x)

  def testMultipleCallsOfSeparateInstances(self):
    # [FUNCTION DOC]
    # Uses the append-before/after manager twice in sequence with different markers and asserts
    # that each use ran independently, giving the list of four markers in order.
    x = []
    with test_yield_append_before_and_after_yield(x, 1, 2):
      pass
    with test_yield_append_before_and_after_yield(x, 3, 4):
      pass
    self.assertEqual([1, 2, 3, 4], x)

  def testReturnsResultFromYield(self):
    # [FUNCTION DOC]
    # Enters the plus-one manager with 3 and asserts the with-target equals 4.
    with test_yield_return_x_plus_1(3) as result:
      self.assertEqual(4, result)

  def testUnwrapContextManager(self):
    # [FUNCTION DOC]
    # Unwraps the decorated fixture using tf_decorator unwrap. Asserts exactly one decorator is
    # found, it is a TFDecorator named contextmanager, and the final target is not a TFDecorator.
    decorators, target = tf_decorator.unwrap(test_params_and_defaults)
    self.assertEqual(1, len(decorators))
    self.assertTrue(isinstance(decorators[0], tf_decorator.TFDecorator))
    self.assertEqual('contextmanager', decorators[0].decorator_name)
    self.assertFalse(isinstance(target, tf_decorator.TFDecorator))

  def testGetArgSpecReturnsWrappedArgSpec(self):
    # [FUNCTION DOC]
    # Asks tf_inspect for the argument spec of the decorated fixture and asserts it reports the
    # original argument names (a, b, c, d) and defaults (2, true, hello) of the wrapped function.
    argspec = tf_inspect.getargspec(test_params_and_defaults)
    self.assertEqual(['a', 'b', 'c', 'd'], argspec.args)
    self.assertEqual((2, True, 'hello'), argspec.defaults)


if __name__ == '__main__':
  test.main()
