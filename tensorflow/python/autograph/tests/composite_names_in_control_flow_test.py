# [MODULE DOC]
# Reference test for autograph conversion of control flow (if, while, for) in which the loop or
# branch state is a composite symbol: an entry of a dictionary or list, possibly nested. Sample
# functions cover balanced and imbalanced initialization of entries across branches, loops with
# state only in composite entries, and deliberately buggy code whose error messages must be
# clear. Tests are parameterized over the sample function, input values and whether the
# condition is a Python value or a tensor. No import-time side effects.
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
"""Composite names (attributes) in control flow.

Generally, composite symbols should be treated like regular ones.
"""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def if_basic_dict(a):
  # [FUNCTION DOC]
  # Function under test: builds a dictionary with key a set to a, and sets key b to one or minus
  # one in the true or false branch of an if on whether a is positive. Returns the dictionary.
  x = {'a': a}

  if x['a'] > 0:
    x['b'] = 1
  else:
    x['b'] = -1
  return x


def if_basic_list(a):
  # [FUNCTION DOC]
  # Function under test: builds the list of a and zero, and sets its second entry to one or
  # minus one depending on whether its first entry is positive. Returns the list.
  x = [a, 0]

  if x[0] > 0:
    x[1] = 1
  else:
    x[1] = -1
  return x


def if_imbalanced(a):
  # [FUNCTION DOC]
  # Function under test: sets dictionary key b only in the true branch of an if on whether
  # the entry a is positive (no else branch). Returns the dictionary.
  x = {'a': a}

  if x['a'] > 0:
    x['b'] = 1
  return x


def else_imbalanced(a):
  # [FUNCTION DOC]
  # Function under test: sets dictionary key b only in the else branch (the true branch does
  # nothing). Returns the dictionary.
  x = {'a': a}

  if x['a'] > 0:
    pass
  else:
    x['b'] = 1
  return x


def if_imbalanced_nested(a):
  # [FUNCTION DOC]
  # Function under test: like the imbalanced if sample, but the entries live in a nested
  # dictionary; sets the inner key b only in the true branch. Returns the outer dictionary.
  x = {'a': {'a': a}}

  if x['a']['a'] > 0:
    x['a']['b'] = 1
  return x


def else_imbalanced_nested(a):
  # [FUNCTION DOC]
  # Function under test: nested-dictionary variant that sets the inner key b only in the else
  # branch. Returns the outer dictionary.
  x = {'a': {'a': a}}

  if x['a']['a'] > 0:
    pass
  else:
    x['a']['b'] = 1
  return x


# Note: this is incorrect code. We test that the error message is clear about
# that.
def if_buggy(a):
  # [FUNCTION DOC]
  # KNOWN ISSUE: intentionally incorrect code. It writes into the inner dictionary under key b
  # although key b was never created, so it fails with a missing-key error even in plain Python.
  # Function under test for error-message clarity: assigns one or minus one to that entry in
  # both branches of an if. Never returns normally.
  x = {'a': {'a': a}}

  if x['a']['a'] > 0:
    x['b']['a'] = 1
  else:
    x['b']['a'] = -1
  return x


# Note: this is incorrect code. We test that the error message is clear about
# that.
def if_imbalanced_buggy(a):
  # [FUNCTION DOC]
  # KNOWN ISSUE: intentionally incorrect code. It writes into the never-created inner dictionary
  # under key b, so it fails with a missing-key error whenever the branch is taken.
  # Function under test for error-message clarity: assigns one to that entry only in the true
  # branch (no else branch).
  x = {'a': {'a': a}}

  if x['a']['a'] > 0:
    x['b']['a'] = 1
  return x


def while_basic_dict(x, a, b):
  # [FUNCTION DOC]
  # Function under test: dictionary with keys a and b; loops while x is positive, decrementing x
  # and incrementing the entry a. Returns the dictionary.
  y = {'a': a, 'b': b}
  while x > 0:
    x -= 1
    y['a'] += 1
  return y


def while_basic_list(x, a, b):
  # [FUNCTION DOC]
  # Function under test: list of a and b; loops while x is positive, decrementing x and
  # incrementing the first list entry. Returns the list.
  y = [a, b]
  while x > 0:
    x -= 1
    y[0] += 1
  return y


def while_state_only_dict(a, b):
  # [FUNCTION DOC]
  # Function under test: dictionary with keys a and b; loops while entry b is at most ten,
  # incrementing entry a and doubling entry b, so the loop state lives only in the dictionary.
  # Returns the dictionary.
  y = {'a': a, 'b': b}
  while y['b'] <= 10:
    y['a'] += 1
    y['b'] *= 2
  return y


def while_state_only_list(a, b):
  # [FUNCTION DOC]
  # Function under test: list of a and b; loops while the second entry is at most ten,
  # incrementing the first and doubling the second. Returns the list.
  y = [a, b]
  while y[1] <= 10:
    y[0] += 1
    y[1] *= 2
  return y


def while_imbalanced(b):
  # [FUNCTION DOC]
  # Function under test: dictionary holding only key b; loops while entry b is at most ten,
  # creating key a inside the loop as b plus one and doubling b. Key a is therefore undefined
  # if the loop never runs. Returns the dictionary.
  y = {'b': b}
  while y['b'] <= 10:
    y['a'] = y['b'] + 1
    y['b'] *= 2
  return y


def for_basic_dict(n, x, a, b):
  # [FUNCTION DOC]
  # Function under test: dictionary with keys a and b; for each index below n, decrements x and
  # adds the index to entry a. Returns the dictionary.
  y = {'a': a, 'b': b}
  for i in range(n):
    x -= 1
    y['a'] += i
  return y


def for_basic_list(n, x, a, b):
  # [FUNCTION DOC]
  # Function under test: list of a and b; for each index below n, decrements x and adds the
  # index to the first list entry. Returns the list.
  y = [a, b]
  for i in range(n):
    x -= 1
    y[0] += i
  return y


def for_state_only_dict(n, a, b):
  # [FUNCTION DOC]
  # Function under test: dictionary with keys a and b; increments entry a once for each of n
  # iterations, so the loop state lives only in the dictionary. Returns the dictionary.
  y = {'a': a, 'b': b}
  for _ in range(n):
    y['a'] += 1
  return y


def for_state_only_list(n, a, b):
  # [FUNCTION DOC]
  # Function under test: list of a and b; increments the first entry once for each of n
  # iterations. Returns the list.
  y = [a, b]
  for _ in range(n):
    y[0] += 1
  return y


def for_imbalanced(n, x):
  # [FUNCTION DOC]
  # Function under test: starts with an empty dictionary; for each index below n, subtracts the
  # index from x and stores x under key a, so key a is undefined if n is zero. Returns the
  # dictionary.
  y = {}
  for i in range(n):
    x -= i
    y['a'] = x
  return y


# TODO(mdan): More tests needed. Many pitfalls around mutating objects this way.


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized reference test case for control flow over composite (dictionary or list entry)
  # symbols. Combines the reference test base class helpers with parameterized test generation.

  @parameterized.parameters(*itertools.product(
      (
          if_basic_dict,
          if_basic_list,
      ),
      (
          0,
          1,
      ),
      (
          bool,
          tf.constant,
      ),
  ))
  def test_if_basic(self, target, a, type_):
    # [FUNCTION DOC]
    # Parameters: the sample function, the input value (zero or one) and a converter (boolean or
    # tensor constant). Converts the input with the converter and asserts the sample matches
    # eager results.
    a = type_(a)
    self.assertFunctionMatchesEager(target, a)

  @parameterized.parameters(*itertools.product(
      (
          if_imbalanced,
          else_imbalanced,
          if_imbalanced_nested,
          else_imbalanced_nested,
      ),
      (
          0,
          1,
      )
  ))
  def test_if_imbalanced_legal(self, target, a):
    # [FUNCTION DOC]
    # Parameters: an imbalanced-if sample function and an input value (zero or one), using plain
    # Python values. Asserts compiled behavior matches eager results.
    self.assertFunctionMatchesEager(target, a)

  @parameterized.parameters(
      (if_imbalanced, 0, tf.constant, ValueError,
       r"'x\['b'\]' must also be initialized in the else"),
      (if_imbalanced, 1, tf.constant, ValueError,
       r"'x\['b'\]' must also be initialized in the else"),
      (else_imbalanced, 0, tf.constant, ValueError,
       r"'x\['b'\]' must also be initialized in the main"),
      (else_imbalanced, 1, tf.constant, ValueError,
       r"'x\['b'\]' must also be initialized in the main"),
      (if_imbalanced_nested, 0, tf.constant, ValueError,
       r"'x\['a'\]\['b'\]' must also be initialized in the else"),
      (if_imbalanced_nested, 1, tf.constant, ValueError,
       r"'x\['a'\]\['b'\]' must also be initialized in the else"),
      (else_imbalanced_nested, 0, tf.constant, ValueError,
       r"'x\['a'\]\['b'\]' must also be initialized in the main"),
      (else_imbalanced_nested, 1, tf.constant, ValueError,
       r"'x\['a'\]\['b'\]' must also be initialized in the main"),
      (if_buggy, 1, int, KeyError, "'b'"),
      (if_buggy, 1, tf.constant, KeyError, "'b'"),
      (if_buggy, 0, tf.constant, KeyError, "'b'"),
      (if_imbalanced_buggy, 1, int, KeyError, "'b'"),
      (if_imbalanced_buggy, 1, tf.constant, KeyError, "'b'"),
      (if_imbalanced_buggy, 0, tf.constant, KeyError, "'b'"),
  )
  def test_if_imbalanced_illegal(self, target, a, type_, exc_type, exc_regex):
    # [FUNCTION DOC]
    # Parameters: sample function, input value, converter, expected exception type and a pattern
    # for its message. Converts the input and runs the sample under graph compilation, asserting
    # it raises the expected exception: a value error naming the composite symbol that must also
    # be initialized in the other branch (tensor conditions), or a missing-key error for the
    # buggy samples.
    a = type_(a)
    with self.assertRaisesRegex(exc_type, exc_regex):
      tf.function(target)(a)

  @parameterized.parameters(*itertools.product(
      (
          while_basic_dict,
          while_basic_list,
      ),
      (
          0,
          1,
          2,
      ),
      (
          bool,
          tf.constant,
      ),
      (
          3,
          7,
      ),
  ))
  def test_while_basic(self, target, x, type_, a):
    # [FUNCTION DOC]
    # Parameters: while-loop sample, loop count x (0 to 2), converter (boolean or tensor
    # constant) and initial value a (3 or 7). Converts x and asserts the sample, called with a and
    # zero, matches eager results.
    x = type_(x)
    self.assertFunctionMatchesEager(target, x, a, 0)

  @parameterized.parameters(*itertools.product(
      (
          while_state_only_dict,
          while_state_only_list,
      ),
      (
          bool,
          tf.constant,
      ),
      (
          3,
          4,
          5,
          6,
      ),
  ))
  def test_while_state_only(self, target, type_, b):
    # [FUNCTION DOC]
    # Parameters: state-only while sample, converter and initial b (3 to 6). Converts b and
    # asserts the sample, called with zero and b, matches eager results.
    b = type_(b)
    self.assertFunctionMatchesEager(target, 0, b)

  @parameterized.parameters(*itertools.product(
      (
          5,
          10,
          11,
      ),
      (
          int,
          tf.constant,
      ),
  ))
  def test_while_imbalanced_legal(self, b, type_):
    # [FUNCTION DOC]
    # Parameters: initial b (5, 10 or 11) and converter (integer or tensor constant). Skips the
    # case b equal to 11 with a tensor, since a tensor loop must initialize the new key. Otherwise
    # converts b and asserts the imbalanced while sample matches eager results.
    if b == 11 and type_ is tf.constant:
      self.skipTest("TF loop must initialize y['a']")

    b = type_(b)
    self.assertFunctionMatchesEager(while_imbalanced, b)

  def test_while_imbalanced_illegal(self):
    # [FUNCTION DOC]
    # Runs the imbalanced while sample under graph compilation with a tensor of eleven, so the
    # loop body never runs, and asserts an invalid-argument error saying the loop must iterate
    # at least once to initialize the new key.
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r"loop must iterate at least once to initialize y\[\\'a\\'\]"):
      tf.function(while_imbalanced)(tf.constant(11))

  @parameterized.parameters(*itertools.product(
      (
          for_basic_dict,
          for_basic_list,
      ),
      (
          0,
          1,
          2,
      ),
      (
          bool,
          tf.constant,
      ),
  ))
  def test_for_basic(self, target, n, type_):
    # [FUNCTION DOC]
    # Parameters: for-loop sample, iteration count n (0 to 2) and converter (boolean or tensor
    # constant). Converts n and asserts the sample, called with n and three ones, matches eager
    # results.
    n = type_(n)
    self.assertFunctionMatchesEager(target, n, 1, 1, 1)

  @parameterized.parameters(*itertools.product(
      (
          for_state_only_dict,
          for_state_only_list,
      ),
      (
          0,
          1,
          2,
      ),
      (
          bool,
          tf.constant,
      ),
  ))
  def test_for_state_only(self, target, n, type_):
    # [FUNCTION DOC]
    # Parameters: state-only for sample, iteration count n (0 to 2) and converter. Converts n and
    # asserts the sample, called with n and two ones, matches eager results.
    n = type_(n)
    self.assertFunctionMatchesEager(target, n, 1, 1)

  @parameterized.parameters(*itertools.product(
      (
          0, 1, 2,
      ),
      (
          int,
          tf.constant,
      ),
  ))
  def test_for_imbalanced_legal(self, n, type_):
    # [FUNCTION DOC]
    # Parameters: iteration count n (0 to 2) and converter (integer or tensor constant). Skips
    # n equal to zero with a tensor (the loop must initialize the new key). Otherwise converts n
    # and asserts the imbalanced for sample, called with n and zero, matches eager results.
    if n == 0 and type_ is tf.constant:
      self.skipTest("TF loop must initialize y['a']")

    n = type_(n)
    self.assertFunctionMatchesEager(for_imbalanced, n, 0)

  def test_for_imbalanced_illegal(self):
    # [FUNCTION DOC]
    # Runs the imbalanced for sample under graph compilation with a tensor zero count, so the
    # loop never runs, and asserts an invalid-argument error saying the loop must iterate at
    # least once to initialize the new key.
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        r"loop must iterate at least once to initialize y\[\\'a\\'\]"):
      tf.function(for_imbalanced)(tf.constant(0), 0)


if __name__ == '__main__':
  tf.test.main()
