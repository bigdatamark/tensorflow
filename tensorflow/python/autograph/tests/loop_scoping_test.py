# [MODULE DOC]
# Reference tests for AutoGraph's handling of variable scoping around for and while loops.
# Defines plain Python functions that exercise loop-local variables, variables defined only
# inside a loop body, loop iterate reuse/alteration, and closures (lambdas) capturing the loop
# variable, plus test classes that compare the compiled (tf.function) behavior against eager
# Python via the reference test base class. Several cases are skipped with TODO notes.
# Nothing is registered at import time; the module runs the TF test main when executed directly.
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
"""Tests that verify scoping around loops."""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def for_with_local_var(l):
  # [FUNCTION DOC]
  # Test subject: loops over the iterable l, computing a temporary x = i + 2 each step and
  # accumulating a decimal-digit style sum s = s * 10 + x. Returns s (0 for an empty input).
  # Verifies that a variable local to the loop body does not leak into loop state.
  s = 0
  for i in l:
    x = i + 2
    s = s * 10 + x
  return s


def while_with_local_var(x):
  # [FUNCTION DOC]
  # Test subject: while x is positive, computes a temporary y = x + 2, accumulates
  # s = s * 10 + y and decrements x. Returns s. Checks loop-local temporaries in while loops.
  s = 0
  while x > 0:
    y = x + 2
    s = s * 10 + y
    x -= 1
  return s


def for_with_lambda_iter(l):
  # [FUNCTION DOC]
  # Test subject: builds a list of lambdas that each return the loop variable (late-binding
  # closure), then calls them all after the loop and returns the list of results. In eager
  # Python every call yields the last element. Only used by skipped tests.
  fns = []
  results = []
  for i in l:
    fns.append(lambda: i)
  for f in fns:
    results.append(f())
  return results


def for_with_lambda_object():
  # [FUNCTION DOC]
  # Test subject (no parameters): defines a local class with a method adding one, loops over a
  # tensor range creating an instance each step and storing a lambda that calls the method of
  # the current instance with a default-bound loop value, then calls all lambdas and returns the
  # results list. Only used by a skipped test.

  class SomeRandomObject:
    # [CLASS DOC]
    # Local helper class for the test subject; its only role is to provide a method bar.

    def bar(self, n):
      # [FUNCTION DOC]
      # HELPER: local helper class method used by the lambda-object test subject.
      # Returns the argument n plus one.
      return n + 1

  def foo_init():
    # [FUNCTION DOC]
    # HELPER: local helper that returns a scalar int32 tensor with value zero, used to initialize
    # the variable that is later rebound inside the loop.
    return tf.constant(0)

  fns = []
  results = []
  foo = foo_init()
  for i in tf.range(3):
    foo = SomeRandomObject()
    fns.append(lambda i=i: foo.bar(i))
  for f in fns:
    results.append(f())
  return results


def for_with_lambda_iter_local_var(l):
  # [FUNCTION DOC]
  # Test subject: like the lambda-iterate subject but each lambda binds the current loop value
  # as a default argument, so each returns its own element. Returns the list of results.
  # Only used by a skipped test.
  fns = []
  results = []
  for i in l:
    fns.append(lambda i=i: i)
  for f in fns:
    results.append(f())
  return results

def for_initializes_local_var(l):
  # [FUNCTION DOC]
  # Test subject: inside a loop over l, a variable x is set to 0 on the first element and
  # incremented on later iterations; s accumulates s * 10 + x. Returns s. In graph mode with
  # tensor inputs x would be undefined before the loop (the illegal-case test is skipped).
  s = 0
  for i in l:
    if i == l[0]:
      x = 0
    else:
      x += 1
    s = s * 10 + x
  return s


def while_initializes_local_var(x):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the inner condition repeats the loop condition (x > 0), so the else branch that
  # increments y can never run; the name suggests a conditional first-iteration initialization.
  # Test subject: while x is positive, sets y to 0 (the else branch increments y), accumulates
  # s = s * 10 + y and decrements x. Returns s.
  s = 0
  while x > 0:
    if x > 0:
      y = 0
    else:
      y += 1
    s = s * 10 + y
    x -= 1
  return s


def for_defines_var(l):
  # [FUNCTION DOC]
  # Test subject: a variable x is defined only inside the loop body (i + 2) and returned after
  # the loop. For an empty iterable eager Python raises an unbound-variable error.
  for i in l:
    x = i + 2
  return x


def while_defines_var(x):
  # [FUNCTION DOC]
  # Test subject: a variable y is defined only inside the while body (x + 2) while x is
  # decremented; y is returned after the loop (undefined if the loop never runs).
  while x > 0:
    y = x + 2
    x -= 1
  return y


def for_defines_iterate(n, fn):
  # [FUNCTION DOC]
  # Test subject: loops over fn(n) (fn is a range-like callable), accumulating s = s * 10 + i,
  # and returns the pair of the last loop variable and s after the loop. With zero iterations
  # eager Python fails because the loop variable is unbound.
  s = 0
  for i in fn(n):
    s = s * 10 + i
  return i, s  # pylint:disable=undefined-loop-variable


def for_reuses_iterate(n, fn):
  # [FUNCTION DOC]
  # Test subject: the loop variable i is initialized to 7 before the loop, then the loop over
  # fn(n) overwrites it and accumulates s. Returns the pair (i, s); i keeps 7 if no iteration.
  i = 7
  s = 0
  for i in fn(n):
    s = s * 10 + i
  return i, s


def for_alters_iterate(n, fn):
  # [FUNCTION DOC]
  # Test subject: like the reuse subject, but the loop body also reassigns the loop variable to
  # three times itself plus one before accumulating. Returns the pair (i, s).
  i = 7
  s = 0
  for i in fn(n):
    i = 3 * i + 1
    s = s * 10 + i
  return i, s


def _int_tensor(x):
  # [FUNCTION DOC]
  # HELPER: converts a Python value to an int32 constant tensor; used to parameterize tests.
  return tf.constant(x, dtype=tf.int32)


class LoopScopingTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests that run the loop-scoping subject functions both eagerly and under
  # tf.function and require identical results, over Python lists/ranges and tensor equivalents.
  # Includes legal-case tests and illegal-case tests (the latter currently skipped).

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_with_local_var(self, l, type_):
    # [FUNCTION DOC]
    # Parameterized over lists of length 0-2 and input converters (list or int32 tensor).
    # Converts l with type_ and asserts the loop-local-variable for-loop matches eager.
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_local_var, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range),
  ))
  def test_for_with_local_var_range(self, l, type_):
    # [FUNCTION DOC]
    # Parameterized over counts 0-2 and range constructors (Python range or tf.range). Builds the
    # range and asserts the loop-local-variable for-loop matches eager.
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_local_var, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, _int_tensor),
  ))
  def test_while_with_local_var(self, x, type_):
    # [FUNCTION DOC]
    # Parameterized over x in 0-2 and converters (int or int32 tensor). Asserts the while
    # loop with a loop-local temporary matches eager.
    x = type_(x)
    self.assertFunctionMatchesEager(while_with_local_var, x)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_initializes_local_var_legal_cases(self, l):
    # [FUNCTION DOC]
    # Parameterized over lists of length 0-2 (Python lists only). Asserts the for-loop that
    # initializes a variable on its first iteration matches eager.
    self.assertFunctionMatchesEager(for_initializes_local_var, l)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_initializes_local_var_illegal_cases(self, l):
    # [FUNCTION DOC]
    # Currently skipped immediately (TODO note). The intended check: with a tensor input, the
    # compiled function raises a value error saying the variable must be defined.
    self.skipTest('TODO(mdanatg): Check')
    l = tf.constant(l)
    with self.assertRaisesRegex(ValueError, '"x" must be defined'):
      tf.function(for_initializes_local_var)(l)

  @parameterized.parameters(
      0,
      1,
      2,
  )
  def test_while_initializes_local_var_legal_cases(self, x):
    # [FUNCTION DOC]
    # Parameterized over x in 0-2 (Python ints). Asserts the while loop with an initialized local
    # matches eager.
    self.assertFunctionMatchesEager(while_initializes_local_var, x)

  @parameterized.parameters(
      0,
      1,
      2,
  )
  def test_while_initializes_local_var_illegal_cases(self, x):
    # [FUNCTION DOC]
    # Currently skipped immediately (TODO note). The intended check: with a tensor input the
    # compiled function raises a value error saying that y must be defined.
    self.skipTest('TODO(mdanatg): check')
    x = tf.constant(x)
    with self.assertRaisesRegex(ValueError, '"y" must be defined'):
      tf.function(while_initializes_local_var)(x)

  @parameterized.parameters(
      # TODO(b/155171694): Enable once the error message here is corrected.
      # ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_defines_var_legal_cases(self, l):
    # [FUNCTION DOC]
    # Parameterized over non-empty lists (the empty case is disabled pending a better error
    # message). Asserts the for-loop that defines a variable inside the body matches eager.
    self.assertFunctionMatchesEager(for_defines_var, l)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_defines_var_illegal_cases(self, l):
    # [FUNCTION DOC]
    # Currently skipped immediately (TODO note). The intended check: with a tensor input the
    # compiled function raises a value error saying that x must be defined.
    self.skipTest('TODO(mdanatg): check')
    l = tf.constant(l)
    with self.assertRaisesRegex(ValueError, '"x" must be defined'):
      tf.function(for_defines_var)(l)

  @parameterized.parameters(
      # TODO(b/155171694): Enable once the error message here is corrected.
      # (0,),
      (1,),
      (2,),
  )
  def test_while_defines_var_legal_cases(self, x):
    # [FUNCTION DOC]
    # Parameterized over x of 1 and 2 (zero disabled pending an error-message fix). Asserts the
    # while loop that defines a variable inside the body matches eager.
    self.assertFunctionMatchesEager(while_defines_var, x)

  @parameterized.parameters(
      (0,),
      (1,),
      (2,),
  )
  def test_while_defines_var_illegal_cases(self, x):
    # [FUNCTION DOC]
    # Currently skipped immediately (TODO note). The intended check: with a tensor input the
    # compiled function raises a value error saying that y must be defined.
    self.skipTest('TODO(mdanatg): check')
    x = tf.constant(x)
    with self.assertRaisesRegex(ValueError, '"y" must be defined'):
      tf.function(while_defines_var)(x)

  @parameterized.parameters(*itertools.product(
      (1, 2),
      (range, tf.range),
  ))
  def test_for_defines_iterate_legal_cases(self, n, fn):
    # [FUNCTION DOC]
    # Parameterized over n of 1 and 2 and range constructors (range, tf.range). Asserts the loop
    # that reads its iterate after the loop matches eager.
    self.assertFunctionMatchesEager(for_defines_iterate, n, fn)

  def test_for_defines_iterate_range(self):
    # [FUNCTION DOC]
    # Placeholder test that is skipped, referring to a tracked bug about the zero-iteration case.
    self.skipTest('b/155171694')

  def test_for_defines_iterate_tf_range(self):
    # [FUNCTION DOC]
    # Calls the compiled iterate-defining loop with zero iterations over a tensor range and
    # asserts the result equals the pair (0, 0). This deliberately deviates from Python
    # semantics (which would raise) to avoid inserting an extra assert operation.
    # Deviating from the normal Python semantics here to avoid inserting
    # an extra assert op. If needed, we can insert it and raise an error
    # to mimic the eager behavior, but this is an exceptionally uncummon
    # use case.
    self.assertAllEqual(tf.function(for_defines_iterate)(0, tf.range), (0, 0))

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_reuses_iterate(self, l, fn):
    # [FUNCTION DOC]
    # Parameterized over lists of length 0-2 and converters (list or int32 tensor); the second
    # parameter is named fn but acts as a converter. Asserts the iterate-reuse loop matches eager.
    self.assertFunctionMatchesEager(for_reuses_iterate, l, fn)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range),
  ))
  def test_for_reuses_iterate_range(self, n, fn):
    # [FUNCTION DOC]
    # Parameterized over n in 0-2 and range constructors (range, tf.range). Asserts the
    # iterate-reuse loop matches eager.
    self.assertFunctionMatchesEager(for_reuses_iterate, n, fn)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_alters_iterate(self, l, fn):
    # [FUNCTION DOC]
    # Parameterized over lists of length 0-2 and converters (list or int32 tensor). Asserts the
    # loop that reassigns its iterate matches eager.
    self.assertFunctionMatchesEager(for_alters_iterate, l, fn)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (range, tf.range),
  ))
  def test_for_alters_iterate_range(self, n, fn):
    # [FUNCTION DOC]
    # Parameterized over n in 0-2 and range constructors. Asserts the iterate-altering loop
    # matches eager.
    self.assertFunctionMatchesEager(for_alters_iterate, n, fn)


class LoopLambdaScopingTest(reference_test_base.TestCase,
                            parameterized.TestCase):
  # [CLASS DOC]
  # Tests for lambdas capturing loop variables. Every test is currently skipped with a reference
  # to a TensorFlow GitHub issue, so no assertions actually run.

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2], [(1, 2), (3, 4)]),
      (list, list),
  ))
  def test_for_with_lambda_iter(self, l, type_):
    # [FUNCTION DOC]
    # Skipped immediately (known issue link). Intended: convert l and assert the late-binding
    # lambda loop matches eager.
    self.skipTest('https://github.com/tensorflow/tensorflow/issues/56089')
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_lambda_iter, l)

  def test_for_with_lambda_object(self):
    # [FUNCTION DOC]
    # Skipped immediately (known issue link). Intended: assert the lambda-object loop matches eager.
    self.skipTest('https://github.com/tensorflow/tensorflow/issues/56089')
    self.assertFunctionMatchesEager(for_with_lambda_object)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2], [(1, 2), (3, 4)]),
      (list, list),
  ))
  def test_for_with_lambda_iter_local_var(self, l, type_):
    # [FUNCTION DOC]
    # Skipped immediately (known issue link). Intended: convert l and assert the loop with
    # default-bound lambdas matches eager.
    self.skipTest('https://github.com/tensorflow/tensorflow/issues/56089')
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_lambda_iter_local_var, l)

if __name__ == '__main__':
  tf.test.main()
