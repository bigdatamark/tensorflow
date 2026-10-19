# [MODULE DOC]
# Reference tests for basic AutoGraph conditionals (if / if-else) run through tf.function.
# Defines many small subject functions (variables created, destroyed, set to None, local to a
# branch, successive and nested ifs) and one parameterized test class that checks each either
# matches eager execution or raises the expected error when the condition is a tensor.
# Uses the shared reference_test_base test case. No import-time side effects.
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
"""Basic conditionals."""

import itertools
import re

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def if_no_vars(c, v):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Resets variable v to zero, adds one to it if condition c holds, returns the variable value.
  v.assign(0)
  if c:
    v.assign_add(1)
  return v.read_value()


def if_else_no_vars(c, v):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Resets variable v to zero, then adds one if c holds, otherwise adds two; returns the value.
  v.assign(0)
  if c:
    v.assign_add(1)
  else:
    v.assign_add(2)
  return v.read_value()


def if_one_var(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Starts a counter at zero and increments it if zero is less than n; returns the counter.
  i = 0
  if i < n:
    i += 1
  return i


def if_else_one_var(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Counter starts at zero; adds one if zero is less than n, otherwise adds two; returns it.
  i = 0
  if i < n:
    i += 1
  else:
    i += 2
  return i


def if_two_vars(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Two variables (0 and 1); if zero is less than n, adds one to the first and multiplies the
  # second by ten. Returns both.
  i = 0
  j = 1
  if i < n:
    i += 1
    j *= 10
  return i, j


def if_else_two_vars(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Two variables (0 and 1); if zero is less than n, adds one and multiplies by ten, otherwise adds
  # two and multiplies by twenty. Returns both.
  i = 0
  j = 1
  if i < n:
    i += 1
    j *= 10
  else:
    i += 2
    j *= 20
  return i, j


def if_creates_var(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Defines a variable only inside the if branch and returns it. With a false plain condition this
  # raises an unbound-local error; with a tensor condition AutoGraph rejects it.
  if c:
    i = 1
  return i


def if_else_creates_var(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Defines a variable in both branches (1 or 2) and returns it; legal for any condition type.
  if c:
    i = 1
  else:
    i = 2
  return i


def else_creates_var(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Defines a variable only in the else branch and returns it; invalid when the else branch is
  # skipped or the condition is a tensor.
  if c:
    pass
  else:
    i = 2
  return i


def if_destroys_var(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Defines a variable, deletes it in the if branch, then returns it (fails if branch taken).
  i = 1
  if c:
    del i
  return i


def if_else_destroys_var(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Defines a variable and deletes it in both branches, then returns it; always fails on return.
  i = 1
  if c:
    del i
  else:
    del i
  return i


def else_destroys_var(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Defines a variable and deletes it only in the else branch, then returns it.
  i = 2
  if c:
    pass
  else:
    del i
  return i


def if_returns_none(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Two variables; if c holds, the first becomes nothing (None) and the second 2. Returns both.
  i = 0
  j = 1
  if c:
    i = None
    j = 2
  return i, j


def if_else_returns_none(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Sets the first variable to None in both branches and the second to 1 or 2; returns both.
  if c:
    i = None
    j = 1
  else:
    i = None
    j = 2
  return i, j


def else_returns_none(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Two variables start at 1; the else branch sets the first to None and the second to 2.
  # Returns both.
  i = 1
  j = 1
  if c:
    pass
  else:
    i = None
    j = 2
  return i, j


def if_local_var(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # If c holds, creates a branch-local variable j and sets i to j plus one. Returns i.
  i = 0
  if c:
    j = 1
    i = j + 1
  return i


def if_else_local_var(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Branch-local j is 1 in the if branch; in the else branch j is 2 and i becomes j plus one.
  # Returns i.
  i = 0
  if c:
    j = 1
  else:
    j = 2
    i = j + 1
  return i


def if_locally_modified_var(c):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # If c holds, increments j (starting at 2) and sets i to j plus one. Returns i.
  i = 0
  j = 2
  if c:
    j = j + 1
    i = j + 1
  return i


def successive_ifs(n1, n2):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Two consecutive if statements, conditioned on n1 and n2, each appending a digit to an
  # accumulator s (s times ten plus i) and incrementing i. Returns s.
  s = 0
  i = 0
  if i < n1:
    s = s * 10 + i
    i += 1
  i = 0
  if i < n2:
    s = s * 10 + i
    i += 1
  return s


def successive_if_elses(n1, n2):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Two consecutive if-else statements on n1 and n2 updating accumulator s (times ten in the if
  # branch, times eleven in else) and counter i. Returns s.
  s = 0
  i = 0
  if i < n1:
    s = s * 10 + i
    i += 1
  else:
    s = s * 11 + i
    i += 2
  i = 0
  if i < n2:
    s = s * 10 + i
    i += 1
  else:
    s = s * 11 + i
    i += 2
  return s


def nested_ifs(n1, n2):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # An if on n1 containing a nested if on n2, accumulating digits into l. Returns l.
  i = 0
  l = 0
  if i < n1:
    j = 0
    s = 0
    if j < n2:
      s = s * 10 + i * j
      j += 1
    l = l * 1000 + s
    i += 1
  return l


def nested_if_temporarily_undefined_return(c1, c2):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Returns 1 if both c1 and c2 hold, otherwise 2. The inner return makes the result temporarily
  # undefined, which tensor conditions reject.
  if c1:
    if c2:
      return 1
  return 2


def nested_if_elses(n1, n2):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest conditional tests.
  # Nested if-else statements on n1 and n2 with different multipliers and increments in each of
  # the four paths, accumulating into l. Returns l.
  i = 0
  l = 0
  if i < n1:
    j = 0
    s = 0
    if j < n2:
      s = s * 10 + i * j
      j += 1
    else:
      s = s * 11 + i * j
      j += 2
    l = l * 1000 + s
    i += 1
  else:
    j = 0
    s = 0
    if j < n2:
      s = s * 12 + i * j
      j += 3
    else:
      s = s * 13 + i * j
      j += 4
    l = l * 2000 + s
    i += 1
  return l


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case comparing AutoGraph-converted conditionals (tf.function) against eager
  # execution for the subject functions above, and checking expected errors. No shared fixtures.

  @parameterized.parameters(*itertools.product(
      (
          if_no_vars,
          if_else_no_vars,
      ),
      (
          True,
          False,
      ),
      (
          bool,
          tf.constant,
      ),
  ))
  def test_no_vars(self, target, c, type_):
    # [FUNCTION DOC]
    # Parameterized over the no-variable subjects, a true/false condition and a condition type
    # (plain bool or tensor). Converts the condition and asserts graph and eager results match.
    c = type_(c)
    self.assertFunctionMatchesEager(target, c, tf.Variable(0))

  @parameterized.parameters(*itertools.product(
      (
          if_one_var,
          if_else_one_var,
          if_two_vars,
          if_else_two_vars,
      ),
      (
          0,
          1,
      ),
      (
          int,
          tf.constant,
      ),
  ))
  def test_several_vars(self, target, n, type_):
    # [FUNCTION DOC]
    # Parameterized over the one- and two-variable subjects, n of 0 or 1, and plain int or tensor.
    # Asserts graph and eager results match.
    n = type_(n)
    self.assertFunctionMatchesEager(target, n)

  def test_var_lifetime_imbalanced_legal(self):
    # [FUNCTION DOC]
    # Checks four subjects with plain boolean conditions where a variable exists on only one path
    # but is used only when defined; asserts graph and eager results match.
    self.assertFunctionMatchesEager(if_creates_var, True)
    self.assertFunctionMatchesEager(else_creates_var, False)
    self.assertFunctionMatchesEager(if_destroys_var, False)
    self.assertFunctionMatchesEager(else_destroys_var, True)

  @parameterized.parameters(*itertools.product(
      (
          True,
          False,
      ),
      (
          int,
          tf.constant,
      ),
  ))
  def test_if_else_var_lifetime(self, c, type_):
    # [FUNCTION DOC]
    # Parameterized over condition value and type. The creating subject must match eager. For the
    # destroying subject a plain int condition must raise an unbound-local error, while a tensor
    # condition must raise a value error saying the variable must also be initialized.
    c = type_(c)
    self.assertFunctionMatchesEager(if_else_creates_var, c)
    if type_ is int:
      with self.assertRaisesRegex(UnboundLocalError, "'i'"):
        tf.function(if_else_destroys_var)(c)
    else:
      with self.assertRaisesRegex(ValueError, "'i' must also be initialized"):
        tf.function(if_else_destroys_var)(c)

  @parameterized.parameters(
      (if_creates_var, False, bool, UnboundLocalError,
       "'i' is used before assignment"),
      (if_creates_var, True, tf.constant, ValueError,
       "'i' must also be initialized in the else branch"),
      (if_creates_var, False, tf.constant, ValueError,
       "'i' must also be initialized in the else branch"),
      (else_creates_var, True, bool, UnboundLocalError,
       "'i' is used before assignment"),
      (else_creates_var, True, tf.constant, ValueError,
       "'i' must also be initialized in the main branch"),
      (else_creates_var, False, tf.constant, ValueError,
       "'i' must also be initialized in the main branch"),
  )
  def test_creates_var_imbalanced_illegal(self, target, c, type_, exc_type,
                                          exc_regex):
    # [FUNCTION DOC]
    # Parameterized over subject, condition, condition type, expected error type and message. Runs
    # the subject through tf.function and asserts the expected error is raised for imbalanced
    # variable creation.
    c = type_(c)
    with self.assertRaisesRegex(exc_type, exc_regex):
      tf.function(target)(c)

  def test_returns_none_legal(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for subjects that set variables to None under plain
    # boolean conditions.
    self.assertFunctionMatchesEager(if_returns_none, True)
    self.assertFunctionMatchesEager(if_else_returns_none, False)
    self.assertFunctionMatchesEager(else_returns_none, False)

  @parameterized.parameters(
      (if_returns_none, True),
      (if_returns_none, False),
      (else_returns_none, True),
      (else_returns_none, False),
      (if_else_returns_none, True),
      (if_else_returns_none, False),
  )
  def test_returns_none_illegal(self, target, c):
    # [FUNCTION DOC]
    # Parameterized over None-producing subjects and condition values. With a tensor condition
    # tf.function must raise a value error saying the variable is None.
    c = tf.constant(c)
    with self.assertRaisesRegex(ValueError, re.compile("'i' is None",
                                                       re.DOTALL)):
      tf.function(target)(c)

  @parameterized.parameters(*itertools.product(
      (
          if_local_var,
          if_else_local_var,
          if_locally_modified_var,
      ),
      (
          True,
          False,
      ),
      (
          bool,
          tf.constant,
      ),
  ))
  def test_local_vars(self, target, c, type_):
    # [FUNCTION DOC]
    # Parameterized over branch-local-variable subjects, condition value and type. Asserts graph and
    # eager results match.
    c = type_(c)
    self.assertFunctionMatchesEager(target, c)

  @parameterized.parameters(*itertools.product(
      (True, False),
      (True, False),
  ))
  def test_nested_if_temporarily_undefined_return_legal(self, c1, c2):
    # [FUNCTION DOC]
    # For all combinations of two plain booleans, asserts graph and eager results match.
    self.assertFunctionMatchesEager(
        nested_if_temporarily_undefined_return, c1, c2)

  @parameterized.parameters(*itertools.product(
      (True, False),
      (True, False),
  ))
  def test_nested_if_temporarily_undefined_return_illegal(self, c1, c2):
    # [FUNCTION DOC]
    # Converts both booleans to tensors and asserts tf.function raises a value error saying the
    # branch must also have a return.
    c1 = tf.constant(c1)
    c2 = tf.constant(c2)
    with self.assertRaisesRegex(ValueError, "must also have a return"):
      tf.function(nested_if_temporarily_undefined_return)(c1, c2)

  @parameterized.parameters(*itertools.product(
      (
          successive_ifs,
          successive_if_elses,
          nested_ifs,
          nested_if_elses,
      ),
      (
          0,
          1,
      ),
      (
          bool,
          tf.constant,
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
  def test_composition(self, target, n1, n1_type, n2, n2_type):
    # [FUNCTION DOC]
    # Parameterized over successive and nested if subjects, n1 and n2 of 0 or 1 and their types
    # (bool or tensor). Asserts graph and eager results match.
    n1 = n1_type(n1)
    n2 = n2_type(n2)
    self.assertFunctionMatchesEager(target, n1, n2)


if __name__ == "__main__":
  tf.test.main()
