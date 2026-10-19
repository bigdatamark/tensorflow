# [MODULE DOC]
# Reference tests for functions with multiple return statements, including returns nested in
# conditionals, loops, context managers and try/finally blocks, run through AutoGraph and compared
# with eager execution (or checked for the expected error). Defines subject functions and one test
# class. No import-time side effects.
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
"""Multiple returns, some in conditionals."""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def return_with_default(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # If x is positive, prints it and returns x; otherwise returns x squared.
  if x > 0:
    tf.print('x', x)
    return x
  return x * x


def return_dependent_on_local(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns a constant one if c holds; otherwise stacks it with itself and returns the sum (two).
  t = tf.constant(1)
  if c:
    return t
  t = tf.stack([t, t])
  return tf.reduce_sum(t)


def return_possibly_undefined(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns x if 0 < x < 5, x cubed if x is not positive, and implicitly nothing when x is at
  # least five. Tensor conditions reject the missing return.
  if x > 0:
    if x < 5:
      return x
  else:
    return x * x * x


def nested_ifs(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Nested conditionals returning x, x squared or x cubed depending on the range of x.
  if x > 0:
    if x < 5:
      return x
    else:
      return x * x
  else:
    return x * x * x


def possible_return_before_loop(c1, c2, n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns one if c1 and c2 both hold; otherwise runs an empty loop n times and returns two.
  if c1:
    if c2:
      return 1
  for _ in range(n):
    pass
  return 2


def nested_ifs_and_context_managers(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Within a name scope, nested conditionals (one inside another name scope) return x, x squared
  # or x cubed.
  with tf.name_scope(''):
    if x > 0:
      if x < 5:
        with tf.name_scope(''):
          return x
      else:
        return x * x
    else:
      return x * x * x


def unreachable_return(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Same as the nested context-manager subject but with a final return of x to the fourth power
  # that can never be reached because every path returns earlier.
  with tf.name_scope(''):
    if x > 0:
      if x < 5:
        with tf.name_scope(''):
          return x
      else:
        return x * x
    else:
      return x * x * x
  return x * x * x * x


def return_with_default_in_contexmanager(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # In a name scope returns one if x is positive, else zero (default return after the if).
  with tf.name_scope(''):
    if x > 0:
      return 1
    return 0


def return_in_try_with_finally(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns one or zero from if/else in a try; the finally block only increments x.
  try:
    if x > 0:
      return 1
    else:
      return 0
  finally:
    x = x + 1


def return_with_default_in_try_with_finally(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns one if x is positive, otherwise zero (default return) inside a try with a finally
  # that increments x.
  try:
    if x > 0:
      return 1
    return 0
  finally:
    x = x + 1


def return_in_finally(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # The try returns two, but the finally block overrides it, returning one or zero by x.
  try:
    return 2
  finally:
    if x > 0:
      return 1  # pylint: disable=lost-exception
    else:
      return 0  # pylint: disable=lost-exception


def return_with_default_in_finally(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Like the previous but the finally block uses a default return of zero after an if.
  try:
    return 2
  finally:
    if x > 0:
      return 1  # pylint: disable=lost-exception
    return 0  # pylint: disable=lost-exception


def return_in_finally_default_in_try(x):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # The try returns zero if x is positive; the finally block always returns one, overriding it.
  try:
    if x > 0:
      return 0
  finally:
    return 1  # pylint: disable=lost-exception


def _raising_helper():
  # [FUNCTION DOC]
  # HELPER: raises an error for the raise-during-return subjects.
  # Always raises a value error.
  raise ValueError()


def raise_during_return_caught():
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Tries to return the result of the raising helper, catches the value error and returns one.
  try:
    return _raising_helper()
  except ValueError:
    pass
  return 1


def raise_during_return_caught_in_tail_branch(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns two if c holds; otherwise as above: the failing return is caught and one returned.
  if c:
    return 2
  try:
    return _raising_helper()
  except ValueError:
    pass
  return 1


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case comparing graph and eager results for the multiple-return subjects.
  # KNOWN ISSUE: the existing docstring says base class, but this class merely derives from the
  # shared reference test base and holds the concrete tests.
  """Base class for the reference tests."""

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_with_default(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n of 0 or 1 and plain int or tensor; asserts graph/eager results match.
    self.assertFunctionMatchesEager(return_with_default, type_(n))

  @parameterized.parameters(*itertools.product(
      (True, False),
      (int, tf.constant),
  ))
  def test_return_dependent_on_local(self, c, type_):
    # [FUNCTION DOC]
    # Parameterized over condition true/false and type; asserts graph and eager results match.
    self.assertFunctionMatchesEager(return_dependent_on_local, type_(c))

  @parameterized.parameters((0,), (3,), (5,))
  def test_return_possibly_undefined_legal(self, n):
    # [FUNCTION DOC]
    # With plain Python values 0, 3 and 5, asserts graph and eager results match.
    self.assertFunctionMatchesEager(return_possibly_undefined, n)

  @parameterized.parameters((0,), (3,), (5,))
  def test_return_possibly_undefined_illegal(self, n):
    # [FUNCTION DOC]
    # With tensor inputs expects a value error saying the else branch must also have a return.
    with self.assertRaisesRegex(
        ValueError, 'else branch must also have a return'):
      tf.function(return_possibly_undefined)(tf.constant(n))

  @parameterized.parameters(*itertools.product(
      (-1, 3, 6),
      (int, tf.constant),
  ))
  def test_nested_ifs(self, n, type_):
    # [FUNCTION DOC]
    # Over values -1, 3, 6 as int or tensor, asserts graph and eager results match.
    self.assertFunctionMatchesEager(nested_ifs, type_(n))

  @parameterized.parameters(*itertools.product(
      (True, False),
      (True, False),
      (0, 1, 2),
  ))
  def test_possible_return_before_loop(self, c1, c2, n):
    # [FUNCTION DOC]
    # Over boolean pairs and loop counts, asserts graph and eager results match.
    self.assertFunctionMatchesEager(possible_return_before_loop, c1, c2, n)

  @parameterized.parameters(*itertools.product(
      (0, 3, 5),
      (int, tf.constant),
  ))
  def test_nested_ifs_and_context_managers(self, x, type_):
    # [FUNCTION DOC]
    # Over values 0, 3, 5 as int or tensor, asserts graph and eager results match.
    self.assertFunctionMatchesEager(nested_ifs_and_context_managers, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 3, 5),
      (int, tf.constant),
  ))
  def test_unreachable_return(self, x, type_):
    # [FUNCTION DOC]
    # Over values 0, 3, 5 as int or tensor, asserts graph and eager results match.
    self.assertFunctionMatchesEager(unreachable_return, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_with_default_in_contexmanager(self, x, type_):
    # [FUNCTION DOC]
    # Over 0 and 1 as int or tensor, asserts graph and eager results match.
    self.assertFunctionMatchesEager(
        return_with_default_in_contexmanager, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_in_try_finally(self, x, type_):
    # [FUNCTION DOC]
    # Over 0 and 1 as int or tensor, asserts results match for return inside try with finally.
    self.assertFunctionMatchesEager(return_in_try_with_finally, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_with_default_try_finally(self, x, type_):
    # [FUNCTION DOC]
    # Over 0 and 1 as int or tensor, asserts results match for default return in try/finally.
    self.assertFunctionMatchesEager(
        return_with_default_in_try_with_finally, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_in_finally(self, x, type_):
    # [FUNCTION DOC]
    # Over 0 and 1 as int or tensor, asserts results match when finally returns.
    self.assertFunctionMatchesEager(return_in_finally, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_with_default_in_finally(self, x, type_):
    # [FUNCTION DOC]
    # Over 0 and 1 as int or tensor, asserts results match for default return in finally.
    self.assertFunctionMatchesEager(return_with_default_in_finally, type_(x))

  @parameterized.parameters(*itertools.product(
      (0, 1),
      (int, tf.constant),
  ))
  def test_return_in_finally_default_in_try(self, x, type_):
    # [FUNCTION DOC]
    # Over 0 and 1 as int or tensor, asserts results match when finally overrides a try return.
    self.assertFunctionMatchesEager(return_in_finally_default_in_try, type_(x))

  def test_raise_during_return_caught(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match when an error raised during return is caught.
    self.assertFunctionMatchesEager(raise_during_return_caught)

  @parameterized.parameters(*itertools.product(
      (True, False),
      (int, tf.constant),
  ))
  def test_raise_during_return_caught_in_tail_branch(self, c, type_):
    # [FUNCTION DOC]
    # Over condition true/false as bool or tensor, asserts results match for the tail-branch case.
    self.assertFunctionMatchesEager(
        raise_during_return_caught_in_tail_branch, type_(c))


if __name__ == '__main__':
  tf.test.main()
