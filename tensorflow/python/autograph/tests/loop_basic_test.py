# [MODULE DOC]
# Reference tests for basic AutoGraph loops (while, for, while/else, for/else, loop-created
# variables, successive and nested loops) over Python lists, tensors, datasets and dataset
# iterators, optionally compiled with XLA. Defines subject functions, small input-conversion
# helpers, a product generator for parameterization and one test class. No import-time side effects.
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
"""Basic loops iterating over various types."""

import functools
import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def while_no_vars(n, v):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Resets variable v to zero and increments it while its value is below n. Returns the value.
  v.assign(0)
  while v.read_value() < n:
    v.assign_add(1)
  return v.read_value()


def for_no_vars(l, v):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Resets variable v to zero and increments it once per element of iterable l. Returns the value.
  v.assign(0)
  for _ in l:
    v.assign_add(1)
  return v.read_value()


def while_one_var(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Starts i at zero and, while i is below n, sets i to ten times i plus one. Returns i.
  i = 0
  while i < n:
    i = i * 10 + 1
  return i


def for_one_var(l):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Starts i at zero and iterates over l assigning each element to i. Returns the last element
  # (or zero when l is empty).
  i = 0
  for i in l:
    pass
  return i


def while_two_vars(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # While counter i is below n, appends i as a decimal digit to accumulator s and increments i.
  # Returns s.
  s = 0
  i = 0
  while i < n:
    s = s * 10 + i
    i += 1
  return s


def for_two_vars(l):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Appends each element of l as a decimal digit to accumulator s. Returns s.
  s = 0
  for i in l:
    s = s * 10 + i
  return s


def while_else(x, y):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # While x is positive, decrements it, breaks if it exceeds y, else adds it to s. The else clause
  # sets s to minus one hundred when no break happened. AutoGraph does not support while/else.
  s = 0
  while x > 0:
    x -= 1
    if x > y:
      break
    s += x
  else:
    s = -100
  return s


def for_else(l1, l2):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Iterates over l1, breaking when an element is in l2, otherwise appending it as a digit to s.
  # The else clause sets s to minus one thousand when no break happened. Unsupported by AutoGraph.
  s = 0
  for c in l1:
    if c in l2:
      break
    s = s * 10 + c
  else:
    s = -1000
  return s


def while_creates_var(n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # While i is below n, sets i to n plus one and creates variable j inside the loop. Returns
  # both; j is undefined when the loop never runs.
  i = 0
  while i < n:
    i = n + 1
    j = i + 1
  return i, j


def for_creates_var(l):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Iterates over l doing nothing and returns the loop variable, which is undefined when l is
  # empty.
  for i in l:
    pass
  return i  # pylint:disable=undefined-loop-variable


def successive_while_loops(n1, n2):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Two consecutive while loops bounded by n1 and n2, each appending its counter as a digit to s.
  # Returns s.
  s = 0
  i = 0
  while i < n1:
    s = s * 10 + i
    i += 1
  i = 0
  while i < n2:
    s = s * 10 + i
    i += 1
  return s


def successive_for_loops(l1, l2):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Two consecutive for loops over l1 and l2 appending each element as a digit to s. Returns s.
  s = 0
  for i in l1:
    s = s * 10 + i
  for i in l2:
    s = s * 10 + i
  return s


def nested_while_loops(n1, n2):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Outer loop over i below n1 and inner loop over j below n2 build a number per outer iteration,
  # written into a dynamically sized integer tensor array. Returns the stacked tensor.
  i = 0
  l = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=())
  while i < n1:
    j = 0
    s = 0
    while j < n2:
      s = s * 10 + i * j
      j += 1
    l = l.write(i, s)
    i += 1
  return l.stack()


def nested_for_loops(m):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # For each inner sequence in m, builds a number from its elements as digits and appends it to a
  # dynamically sized integer tensor array. Returns the stacked tensor.
  l = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=())
  for i in m:
    s = 0
    for j in i:
      s = s * 10 + j
    l = l.write(l.size(), s)
  return l.stack()


def _int_tensor(x):
  # [FUNCTION DOC]
  # HELPER: converts test inputs for ReferenceTest parameterization.
  # Converts x to a 32-bit integer constant tensor.
  return tf.constant(x, dtype=tf.int32)


def _int_dataset(l):
  # [FUNCTION DOC]
  # HELPER: converts test inputs for ReferenceTest parameterization.
  # Converts list l to a 32-bit integer tensor and returns a dataset yielding its slices.
  return tf.data.Dataset.from_tensor_slices(tf.constant(l, dtype=tf.int32))


def double_product(l1, l2):
  # [FUNCTION DOC]
  # HELPER: builds parameter tuples for ReferenceTest.
  # Generator yielding every combination (i, j, k, l) with i and k drawn from l1 and j and l drawn
  # from l2, i.e. the product of l1, l2, l1, l2 flattened into four-element tuples.
  for i in l1:
    for j in l2:
      for k in l1:
        for l in l2:
          yield i, j, k, l


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case comparing graph (optionally XLA) and eager results of the loop subject
  # functions over several input kinds, and checking expected errors for unsupported constructs.

  @parameterized.parameters(*itertools.product(
      (
          0,
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_no_vars(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n of 0 to 2 and plain int or tensor. Asserts graph and eager results match
    # for the no-variable while loop with a fresh variable.
    n = type_(n)
    self.assertFunctionMatchesEager(while_no_vars, n, tf.Variable(0))

  @parameterized.parameters(*itertools.product(
      (
          [],
          [1],
          [1, 2],
      ),
      (
          list,
          _int_tensor,
          # TODO(mdan): Enable this once #35335 is fixed.
          # _int_dataset,
      ),
      (
          True,
          False,
      ),
  ))
  def test_for_no_vars(self, l, type_, xla):
    # [FUNCTION DOC]
    # Parameterized over list contents, input kind (list or tensor) and XLA flag. Skips empty
    # tensors under XLA. Asserts graph and eager results match for the no-variable for loop.
    if type_ is _int_tensor and xla and not l:
      self.skipTest('Empty loops not supported in XLA')

    l = type_(l)
    self.assertFunctionMatchesEager(for_no_vars, l, tf.Variable(0), xla=xla)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_no_vars_ds_iterator(self, l):
    # [FUNCTION DOC]
    # For several lists, builds a dataset iterator plus variable per call and asserts graph and
    # eager results match using the stateful-input comparison helper.
    inputs_ = lambda: (iter(_int_dataset(l)), tf.Variable(0))
    self.assertFunctionMatchesEagerStatefulInput(for_no_vars, inputs_)

  @parameterized.parameters(*itertools.product(
      (
          0,
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_one_var(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and plain int or tensor; asserts graph and eager results match.
    n = type_(n)
    self.assertFunctionMatchesEager(while_one_var, n)

  @parameterized.parameters(*itertools.product(
      (
          [],
          [1],
          [1, 2],
      ),
      (
          list,
          _int_tensor,
          _int_dataset,
      ),
      (
          True,
          False,
      ),
  ))
  def test_for_one_var(self, l, type_, xla):
    # [FUNCTION DOC]
    # Parameterized over list contents, input kind (list, tensor, dataset) and XLA flag. Skips
    # datasets under XLA and empty tensors under XLA; asserts graph and eager results match.
    if type_ is _int_dataset and xla:
      self.skipTest('Datasets not supported in XLA')
    if type_ is _int_tensor and xla and not l:
      self.skipTest('Empty loops not supported in XLA')

    l = type_(l)
    self.assertFunctionMatchesEager(for_one_var, l, xla=xla)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_one_var_ds_iterator(self, l):
    # [FUNCTION DOC]
    # Uses dataset iterators as stateful inputs and asserts graph and eager results match.
    # KNOWN ISSUE: the input factory also supplies a variable, but the subject for_one_var accepts
    # only one argument, so the call with two inputs looks wrong (copied from the no-vars test).
    inputs_ = lambda: (iter(_int_dataset(l)), tf.Variable(0))
    self.assertFunctionMatchesEagerStatefulInput(for_one_var, inputs_)

  @parameterized.parameters(*itertools.product(
      (
          0,
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_two_vars(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and plain int or tensor; asserts graph and eager results match.
    n = type_(n)
    self.assertFunctionMatchesEager(while_two_vars, n)

  @parameterized.parameters(*itertools.product(
      (
          [],
          [1],
          [1, 2],
      ),
      (
          list,
          _int_tensor,
          _int_dataset,
      ),
      (
          True,
          False,
      ),
  ))
  def test_for_two_vars(self, l, type_, xla):
    # [FUNCTION DOC]
    # Parameterized over list contents, input kind and XLA flag; skips unsupported XLA cases
    # (datasets, empty tensors); asserts graph and eager results match.
    if type_ is _int_dataset and xla:
      self.skipTest('Datasets not supported in XLA')
    if type_ is _int_tensor and xla and not l:
      self.skipTest('Empty loops not supported in XLA')

    l = type_(l)
    self.assertFunctionMatchesEager(for_two_vars, l, xla=xla)

  @parameterized.parameters(
      ([],),
      ([1],),
      ([1, 2],),
  )
  def test_for_two_vars_ds_iterator(self, l):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the input factory also supplies a variable, but for_two_vars accepts only one
    # argument, so the call with two inputs looks wrong (copied from the no-vars test).
    # Uses dataset iterators as stateful inputs; asserts graph and eager results match.
    inputs_ = lambda: (iter(_int_dataset(l)), tf.Variable(0))
    self.assertFunctionMatchesEagerStatefulInput(for_two_vars, inputs_)

  @parameterized.parameters(*itertools.product(
      (
          [],
          [1, 2],
      ),
      (
          [],
          [1, 2],
      ),
      (list, _int_tensor),
  ))
  def test_for_else(self, l1, l2, type_):
    # [FUNCTION DOC]
    # Converts two lists to the chosen kind and asserts tf.function raises a not-implemented
    # error mentioning for/else.
    l1 = type_(l1)
    l2 = type_(l2)
    with self.assertRaisesRegex(NotImplementedError, 'for/else'):
      tf.function(for_else)(l1, l2)

  @parameterized.parameters(*itertools.product(
      (0, 2),
      (0, 2),
      (int, _int_tensor),
  ))
  def test_while_else(self, n1, n2, type_):
    # [FUNCTION DOC]
    # Converts two numbers to the chosen kind and asserts tf.function raises a not-implemented
    # error mentioning while/else.
    n1 = type_(n1)
    n2 = type_(n2)
    with self.assertRaisesRegex(NotImplementedError, 'while/else'):
      tf.function(while_else)(n1, n2)

  @parameterized.parameters(*itertools.product(
      (
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_while_creates_var_legal(self, n, type_):
    # [FUNCTION DOC]
    # With n of 1 or 2 (loop runs), asserts graph and eager results match.
    n = type_(n)
    self.assertFunctionMatchesEager(while_creates_var, n)

  def test_while_creates_var_illegal(self):
    # [FUNCTION DOC]
    # With zero: a plain int must raise an unbound-local error, and a tensor must raise an
    # invalid-argument error saying the loop must iterate at least once.
    with self.assertRaisesRegex(UnboundLocalError, 'used before assignment'):
      tf.function(while_creates_var)(0)
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'loop must iterate at least once'):
      tf.function(while_creates_var)(tf.constant(0))

  @parameterized.parameters(*itertools.product(
      (
          [1],
          [1, 2],
      ),
      (
          list,
          _int_tensor,
      ),
  ))
  def test_for_creates_var_legal(self, l, type_):
    # [FUNCTION DOC]
    # With non-empty lists as list or tensor, asserts graph and eager results match.
    l = type_(l)
    self.assertFunctionMatchesEager(for_creates_var, l)

  def test_for_creates_var_illegal(self):
    # [FUNCTION DOC]
    # With an empty list expects an unbound-local error; with an empty tensor expects an
    # invalid-argument error saying the loop must iterate at least once.
    with self.assertRaisesRegex(UnboundLocalError, 'used before assignment'):
      tf.function(for_creates_var)([])
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'loop must iterate at least once'):
      tf.function(for_creates_var)(tf.constant([]))

  @parameterized.parameters(*double_product(
      (
          0,
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_successive_while_loops(self, n1, type1, n2, type2):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second parameter type (type2) is never used; both counts are converted with
    # the first type.
    # Asserts graph and eager results match for two consecutive while loops.
    n1 = type1(n1)
    n2 = type1(n2)
    self.assertFunctionMatchesEager(successive_while_loops, n1, n2)

  @parameterized.parameters(*double_product(
      (
          [],
          [1],
          [1, 2],
      ),
      (
          list,
          _int_tensor,
          _int_dataset,
      ),
  ))
  def test_successive_for_loops(self, l1, type1, l2, type2):
    # [FUNCTION DOC]
    # KNOWN ISSUE: type2 is never used; both lists are converted with the first type.
    # Asserts graph and eager results match for two consecutive for loops.
    l1 = type1(l1)
    l2 = type1(l2)
    self.assertFunctionMatchesEager(successive_for_loops, l1, l2)

  @parameterized.parameters(*double_product(
      (
          [],
          [1],
          [1, 2],
      ),
      (
          list,
          _int_dataset,
      ),
  ))
  def test_successive_for_loops_iterators(self, l1, type1, l2, type2):
    # [FUNCTION DOC]
    # Wraps both inputs in iterators of the given kinds as stateful inputs and asserts graph and
    # eager results match for the two consecutive for loops.
    inputs_ = lambda: (iter(type1(l1)), iter(type2(l2)))
    self.assertFunctionMatchesEagerStatefulInput(successive_for_loops, inputs_)

  @parameterized.parameters(*double_product(
      (
          0,
          1,
          2,
      ),
      (
          int,
          _int_tensor,
      ),
  ))
  def test_nested_while_loops(self, n1, type1, n2, type2):
    # [FUNCTION DOC]
    # KNOWN ISSUE: type2 is never used; both counts are converted with the first type.
    # Asserts graph and eager results match for nested while loops.
    n1 = type1(n1)
    n2 = type1(n2)
    self.assertFunctionMatchesEager(nested_while_loops, n1, n2)

  @parameterized.parameters(*itertools.product(
      (
          [[]],
          [[], []],
          [[1]],
          [[1], [2]],
          [[1, 2]],
          [[1, 2], [3, 4]],
      ),
      (
          _int_tensor,
          _int_dataset,
      ),
  ))
  def test_nested_for_loops_dense(self, m, type_):
    # [FUNCTION DOC]
    # Parameterized over dense nested lists as tensor or dataset; asserts graph and eager results
    # match for nested for loops.
    m = type_(m)
    self.assertFunctionMatchesEager(nested_for_loops, m)

  @parameterized.parameters(*itertools.product(
      (
          [[]],
          [[], [1]],
          [[], [1], [1, 2]],
      ),
      (
          list,
          functools.partial(tf.ragged.constant, dtype=tf.int32),
      ),
  ))
  def test_nested_for_loops_ragged(self, m, type_):
    # [FUNCTION DOC]
    # Parameterized over ragged nested lists as plain list or ragged tensor; asserts graph and
    # eager results match for nested for loops.
    m = type_(m)
    self.assertFunctionMatchesEager(nested_for_loops, m)

  def test_nested_for_loops_mixed_list(self):
    # [FUNCTION DOC]
    # Passes a list mixing plain lists and tensors as inner sequences; asserts graph and eager
    # results match.
    m = [[], _int_tensor([]), [1], _int_tensor([1]), [1, 2]]
    self.assertFunctionMatchesEager(nested_for_loops, m)


if __name__ == '__main__':
  tf.test.main()
