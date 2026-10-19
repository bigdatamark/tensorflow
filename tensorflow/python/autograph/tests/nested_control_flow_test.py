# [MODULE DOC]
# Reference tests for AutoGraph conversion of arbitrarily nested control flow: if inside if,
# loops inside loops, ifs inside loops, break/continue inside context managers and try blocks,
# modification of module globals from nested conditionals, and loop variables first defined
# inside the loop. Contains the subject functions and one parameterized test class that
# compares tf.function execution against eager Python. Module state: a global used by the
# global-modification subject. No other import-time side effects.
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
"""Nested loops and conditional statements (e.g. while, for, if).

Meant to verify that arbitrarily nested statements are processed correctly.
"""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def independent_ifs(x, y):
  # [FUNCTION DOC]
  # Test subject: z starts at 0; if x is positive and, nested, y is positive, z becomes x + y.
  # Returns z.
  z = 0
  if x > 0:
    if y > 0:
      z = x + y
  return z


def dependent_inner_if(x):
  # [FUNCTION DOC]
  # Test subject: if x is positive, y becomes -2x and, if that is positive (it never is for
  # positive x), x becomes -3x; otherwise y becomes 4x. Returns the pair (x, y).
  y = 0
  if x > 0:
    y = -2 * x
    if y > 0:
      x = -3 * x
  else:
    y = 4 * x
  return x, y


def dependent_imbalanced_inner_if(x):
  # [FUNCTION DOC]
  # Test subject: only when x is positive and below three, sets y = -2x and x = -3x (no else
  # branch at either level). Returns the pair (x, y).
  y = 0
  if x > 0:
    if x < 3:
      y = -2 * x
      x = -3 * x
  return x, y


def _hidden_raise():
  # [FUNCTION DOC]
  # HELPER: always raises a value error; hides the raise from static analysis in the subject.
  raise ValueError('exception used for control flow')


def if_with_local_modification_masked_by_exception(x):
  # [FUNCTION DOC]
  # Test subject: when x is positive, calls the raising helper if x > 1, otherwise sets y to 1;
  # a value error is swallowed; then if y is still zero sets it to 2. Returns y (0 if x <= 0).
  y = 0
  if x > 0:
    try:
      if x > 1:
        _hidden_raise()
      y = 1
    except ValueError:
      pass
    if y == 0:
      y = 2
  return y


_test_global = None


def if_nested_with_modification_of_global(x):
  # [FUNCTION DOC]
  # Test subject: when x is positive (checked twice, nested), initializes the module global
  # counter to 1 if it is unset, otherwise increments it, and adds it to y. Returns y.
  # Side effect: mutates the module-level global.
  y = 0
  if x > 0:
    if x > 0:
      global _test_global
      if _test_global is None:
        _test_global = 1
      else:
        _test_global += 1
      y += _test_global
  return y


def independent_inner_for(a, b):
  # [FUNCTION DOC]
  # Test subject: for each item of a, iterates over b (via a temporary) summing the items into p.
  # Returns p.
  p = 0
  for _ in a:
    tmp = b
    for j in tmp:
      p += j
  return p


def independent_inner_while(a, b):
  # [FUNCTION DOC]
  # Test subject: while a > 0, runs an inner loop counting tmp from b down to 0 and adds one per
  # inner step to p, then decrements a. Returns p. The inner loop does not depend on outer state.
  p = 0
  while a > 0:
    tmp = b
    while tmp > 0:
      p += 1
      tmp -= 1
    a -= 1
  return p


def dependent_inner_for(a, b):
  # [FUNCTION DOC]
  # Test subject: for each item of a, adds the running sum s to r, then iterates over b adding
  # items to s. Returns r (the inner loop state feeds the outer one).
  r = 1
  s = 0
  for _ in a:
    r += s
    tmp = b
    for j in tmp:
      s += j
  return r


def dependent_inner_while(a, b):
  # [FUNCTION DOC]
  # Test subject: while a > 0, increments r, runs an inner loop that decrements both a and a
  # temporary copy of b, then decrements a again. Returns r. The inner loop modifies the outer
  # loop's condition variable.
  r = 1
  while a > 0:
    r += 1
    tmp = b
    while tmp > 0:
      a -= 1
      tmp -= 1
    a -= 1
  return r


def if_in_for(a):
  # [FUNCTION DOC]
  # Test subject: for each i in a, if i is odd adds i integer-divided by two to k. Returns k.
  k = 0
  for i in a:
    if i % 2 > 0:
      j = i // 2
      k += j
  return k


def while_with_continue_in_context_manager(x):
  # [FUNCTION DOC]
  # Test subject: while x > 0, inside a name scope decrements x, continues if x < 5, else
  # increments z. Returns z.
  z = 0
  while x > 0:
    with tf.name_scope(''):
      x = x - 1
      if x < 5:
        continue
      z = z + 1
  return z


def while_continue_in_try(x):
  # [FUNCTION DOC]
  # Test subject: while x > 0, decrements x, then in a try block continues if x < 5 else
  # increments z; the finally block always adds ten to z. Returns z.
  z = 0
  while x > 0:
    x = x - 1
    try:
      if x < 5:
        continue
      z = z + 1
    finally:
      z = z + 10
  return z


def while_break_in_context_manager(x):
  # [FUNCTION DOC]
  # Test subject: while x > 0, inside a name scope decrements x, breaks if x < 5, else
  # increments z. Returns z.
  z = 0
  while x > 0:
    with tf.name_scope(''):
      x = x - 1
      if x < 5:
        break
      z = z + 1
  return z


def while_break_in_try(x):
  # [FUNCTION DOC]
  # Test subject: while x > 0, decrements x, then in a try block breaks if x < 5 else
  # increments z; the finally block adds ten to z, also when breaking. Returns z.
  z = 0
  while x > 0:
    x = x - 1
    try:
      if x < 5:
        break
      z = z + 1
    finally:
      z = z + 10
  return z


def loop_initializing_invariant_variable(n):
  # [FUNCTION DOC]
  # Test subject: loops over a range of n, assigning a = 1 on the first iteration and a = 2
  # afterwards. Returns a, which is unbound when n is zero.
  for i in range(n):
    if i == 0:
      a = 1
    else:
      a = 2
  return a


def loop_initializing_variant_variable(n):
  # [FUNCTION DOC]
  # Test subject: loops over a range of n; a is 1 on the first iteration and a + 1 afterwards,
  # so it depends on its own earlier value before being defined. Returns a.
  for i in range(n):
    if i == 0:
      a = 1
    else:
      a = a + 1
  return a


def _int_tensor(x):
  # [FUNCTION DOC]
  # HELPER: converts a Python value to an int32 constant tensor; used to parameterize tests.
  return tf.constant(x, dtype=tf.int32)


class NestedControlFlowTest(
    reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests asserting that nested control-flow subject functions behave the same when
  # compiled by tf.function as in eager Python, plus tests of expected failures.

  @parameterized.parameters(*itertools.product(
      (
          -1, 1,
      ),
      (
          -1, 1,
      ),
      (
          int,
          _int_tensor,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_independent_ifs(self, x, y, type_x, type_y):
    # [FUNCTION DOC]
    # Parameterized over x and y in {-1, 1} and converters (int, int32 tensor). KNOWN ISSUE: both
    # x and y are converted with type_x; type_y is never used. Asserts the nested ifs match eager.
    x = type_x(x)
    y = type_x(y)
    self.assertFunctionMatchesEager(independent_ifs, x, y)

  @parameterized.parameters(*itertools.product(
      (
          -1, 1,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_dependent_inner_if(self, x, type_):
    # [FUNCTION DOC]
    # Parameterized over x in {-1, 1} and converters. Asserts the dependent nested ifs match eager.
    x = type_(x)
    self.assertFunctionMatchesEager(dependent_inner_if, x)

  @parameterized.parameters(*itertools.product(
      (
          -1, 1,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_dependent_imbalanced_inner_if(self, x, type_):
    # [FUNCTION DOC]
    # Parameterized over x in {-1, 1} and converters. Asserts the nested ifs without else match
    # eager.
    x = type_(x)
    self.assertFunctionMatchesEager(dependent_imbalanced_inner_if, x)

  @parameterized.parameters(
      (-1,),
      (0,),
      (1,),
      (2,),
  )
  def test_if_with_local_modification_masked_by_exception(self, x):
    # [FUNCTION DOC]
    # Parameterized over Python ints -1 to 2 only (tensor inputs have undefined behavior).
    # Asserts the exception-masked modification subject matches eager.
    # Note: If the input is a Tensor, the behavior is undefined.
    self.assertFunctionMatchesEager(
        if_with_local_modification_masked_by_exception, x)

  def test_if_nested_with_modification_of_global(self):
    # [FUNCTION DOC]
    # Resets the module global to unset, runs the compiled subject with x = 1 and asserts the
    # return value is 1 and the global ends at 1. Side effect: modifies the module global.
    global _test_global
    _test_global = None
    self.assertEqual(tf.function(if_nested_with_modification_of_global)(1), 1)
    self.assertEqual(_test_global, 1)

  def test_if_nested_with_modification_of_global_not_executed(self):
    # [FUNCTION DOC]
    # Resets the module global to unset, runs the compiled subject with x = 0 and asserts the
    # return value is 0 and the global remains unset. Side effect: resets the module global.
    global _test_global
    _test_global = None
    self.assertEqual(tf.function(if_nested_with_modification_of_global)(0), 0)
    self.assertIsNone(_test_global)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2,
      ),
      (
          0, 1, 2,
      ),
      (
          range,
          tf.range,
      ),
      (
          range,
          tf.range,
      ),
  ))
  def test_independent_inner_for(self, a, b, type_a, type_b):
    # [FUNCTION DOC]
    # Parameterized over a and b in 0-2 and range constructors (range, tf.range). Asserts the
    # nested for loops match eager.
    a = type_a(a)
    b = type_b(b)
    self.assertFunctionMatchesEager(independent_inner_for, a, b)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2,
      ),
      (
          0, 1, 2,
      ),
      (
          int,
          _int_tensor,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_independent_inner_while(self, a, b, type_a, type_b):
    # [FUNCTION DOC]
    # Parameterized over a and b in 0-2 and converters (int, int32 tensor). Asserts the nested
    # while loops match eager.
    a = type_a(a)
    b = type_b(b)
    self.assertFunctionMatchesEager(independent_inner_while, a, b)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2,
      ),
      (
          0, 1, 2,
      ),
      (
          range,
          tf.range,
      ),
      (
          range,
          tf.range,
      ),
  ))
  def test_dependent_inner_for(self, a, b, type_a, type_b):
    # [FUNCTION DOC]
    # Parameterized over a and b in 0-2 and range constructors. Asserts the dependent nested for
    # loops match eager.
    a = type_a(a)
    b = type_b(b)
    self.assertFunctionMatchesEager(dependent_inner_for, a, b)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2, 3, 4,
      ),
      (
          0, 1, 2, 3, 4,
      ),
      (
          int,
          _int_tensor,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_dependent_inner_while(self, a, b, type_a, type_b):
    # [FUNCTION DOC]
    # Parameterized over a and b in 0-4 and converters. Skips the combination of a Python int
    # for a with a tensor for b (known bug). Asserts the dependent nested while loops match eager.
    if (type_a is int) and (type_b is _int_tensor):
      self.skipTest('b/124378596')
    a = type_a(a)
    b = type_b(b)
    self.assertFunctionMatchesEager(dependent_inner_while, a, b)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2,
      ),
      (
          range,
          tf.range,
      ),
  ))
  def test_if_in_for(self, a, type_):
    # [FUNCTION DOC]
    # Parameterized over a in 0-2 and range constructors. Asserts the if-in-for subject matches
    # eager.
    a = type_(a)
    self.assertFunctionMatchesEager(if_in_for, a)

  @parameterized.parameters(*itertools.product(
      (
          0, 4, 10,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_continue_in_context_manager(self, x, type_):
    # [FUNCTION DOC]
    # Parameterized over x in {0, 4, 10} and converters. Asserts the continue-in-context-manager
    # loop matches eager.
    x = type_(x)
    self.assertFunctionMatchesEager(while_with_continue_in_context_manager, x)

  @parameterized.parameters(*itertools.product(
      (
          0, 4, 10,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_continue_in_try(self, x, type_):
    # [FUNCTION DOC]
    # Parameterized over x in {0, 4, 10} and converters. Asserts the continue-in-try loop matches
    # eager.
    x = type_(x)
    self.assertFunctionMatchesEager(while_continue_in_try, x)

  @parameterized.parameters(*itertools.product(
      (
          0, 4, 10,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_break_in_context_manager(self, x, type_):
    # [FUNCTION DOC]
    # Parameterized over x in {0, 4, 10} and converters. Asserts the break-in-context-manager
    # loop matches eager.
    x = type_(x)
    self.assertFunctionMatchesEager(while_break_in_context_manager, x)

  @parameterized.parameters(*itertools.product(
      (
          0, 4, 10,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_break_in_try(self, x, type_):
    # [FUNCTION DOC]
    # Parameterized over x in {0, 4, 10} and converters. Asserts the break-in-try loop matches eager.
    x = type_(x)
    self.assertFunctionMatchesEager(while_break_in_try, x)

  @parameterized.parameters(*itertools.product(
      (
          1, 2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_loop_initializing_invariant_variable_legal(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n in {1, 2} and converters. Asserts the loop initializing a variable
    # inside matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(loop_initializing_invariant_variable, n)

  def test_loop_initializing_invariant_variable_illegal(self):
    # [FUNCTION DOC]
    # With a Python zero, the compiled function must raise an unbound-local error. With a tensor
    # zero, it must raise an invalid-argument error saying the loop must iterate at least once.
    with self.assertRaises(UnboundLocalError):
      tf.function(loop_initializing_invariant_variable)(0)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'loop must iterate at least once'):
      tf.function(loop_initializing_invariant_variable)(tf.constant(0))

  @parameterized.parameters(
      (1,),
      (2,),
  )
  def test_loop_initializing_variant_variable_legal(self, n):
    # [FUNCTION DOC]
    # Parameterized over Python n in {1, 2}. Simply runs the compiled function; asserts only that
    # no error is raised.
    tf.function(loop_initializing_variant_variable)(n)

  @parameterized.parameters(
      (0,),
      (1,),
      (2,),
  )
  def test_loop_initializing_variant_variable_illegal(self, n):
    # [FUNCTION DOC]
    # Parameterized over n in 0-2 passed as tensors. Asserts a value error saying the variable
    # must be defined before the loop.
    with self.assertRaisesRegex(ValueError, 'must be defined before the loop'):
      tf.function(loop_initializing_variant_variable)(tf.constant(n))


if __name__ == '__main__':
  tf.test.main()
