# [MODULE DOC]
# Reference tests for AutoGraph conversion of loops whose conditions, iteration targets or bodies
# contain function calls: global functions, lambdas, bound methods, builtins, local (nested)
# functions, closures over loop variables, lambda closures and callable-object closures.
# Defines the subject functions plus one parameterized test class that compares tf.function
# results against eager execution through the reference test base class.
# No import-time side effects; runs the TF test main when executed directly.
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
"""Function calls inside the while loop body."""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def while_with_call_in_cond(n, fn):
  # [FUNCTION DOC]
  # Test subject: while the counter i is less than fn(n) (fn is called on every condition
  # check), accumulates s = s * 10 + i and increments i. Returns s.
  i = 0
  s = 0
  while i < fn(n):
    s = s * 10 + i
    i += 1
  return s


def for_with_call_in_target(l, fn):
  # [FUNCTION DOC]
  # Test subject: iterates over the result of calling fn on l, accumulating s = s * 10 + i.
  # Returns s.
  s = 0
  for i in fn(l):
    s = s * 10 + i
  return s


def while_with_local_call_in_cond(n):
  # [FUNCTION DOC]
  # Test subject: like the call-in-condition subject but uses a nested helper multiplying by three
  # as the callee. Returns the accumulated s. (Not referenced by any test in this module.)

  def local_fn(x):
    # [FUNCTION DOC]
    # HELPER: nested callee used inside the loop condition of the enclosing test subject.
    # Returns x times three.
    return x * 3

  i = 0
  s = 0
  while i < local_fn(n):
    s = s * 10 + i
    i += 1
  return s


def for_with_local_call_in_target(l):
  # [FUNCTION DOC]
  # Test subject: iterates over the result of a nested helper applied to l (the helper returns l
  # times one) accumulating s = s * 10 + i. Returns s. (Not referenced by any test here.)

  def local_fn(l):
    # [FUNCTION DOC]
    # HELPER: nested callee used to build the iteration target; returns l times one.
    return l * 1

  s = 0
  for i in local_fn(l):
    s = s * 10 + i
  return s


def while_with_call(n, fn):
  # [FUNCTION DOC]
  # Test subject: while i < n, accumulates s = s * 10 + fn(i) and increments i; fn is called in
  # the loop body. Returns s.
  i = 0
  s = 0
  while i < n:
    s = s * 10 + fn(i)
    i += 1
  return s


def for_with_call(l, fn):
  # [FUNCTION DOC]
  # Test subject: iterates over l accumulating s = s * 10 + fn(i). Returns s.
  s = 0
  for i in l:
    s = s * 10 + fn(i)
  return s


def while_with_local_call(n):
  # [FUNCTION DOC]
  # Test subject: while i < n, accumulates s = s * 10 + local_fn(i), where local_fn is a nested
  # function multiplying by three. Returns s.

  def local_fn(x):
    # [FUNCTION DOC]
    # HELPER: nested callee for the enclosing test subject; returns x times three.
    return x * 3

  i = 0
  s = 0
  while i < n:
    s = s * 10 + local_fn(i)
    i += 1
  return s


def for_with_local_call(l):
  # [FUNCTION DOC]
  # Test subject: iterates over l accumulating s = s * 10 + local_fn(i) with a nested function
  # multiplying by three. Returns s.

  def local_fn(x):
    # [FUNCTION DOC]
    # HELPER: nested callee for the enclosing test subject; returns x times three.
    return x * 3

  s = 0
  for i in l:
    s = s * 10 + local_fn(i)
  return s


def while_with_closure_call(n):
  # [FUNCTION DOC]
  # Test subject: defines a nested function that reads the loop counter i from the enclosing
  # scope and adds two; the while loop (i < n) accumulates s = s * 10 + that closure result
  # and increments i. Returns s.
  i = 0

  def i_via_closure():
    # [FUNCTION DOC]
    # HELPER: nested closure that returns the enclosing variable i plus two.
    return i + 2

  i = 0
  s = 0
  while i < n:
    s = s * 10 + i_via_closure()
    i += 1
  return s


def for_with_closure_call(l):
  # [FUNCTION DOC]
  # Test subject: a nested closure reads the enclosing variable i (plus two); the loop over l
  # rebinds i and accumulates s = s * 10 + closure result. Returns the pair (s, i); the extra i
  # is a known workaround noted by a TODO.
  i = 0

  def i_via_closure():
    # [FUNCTION DOC]
    # HELPER: nested closure that returns the enclosing variable i plus two.
    return i + 2

  s = 0
  for i in l:
    s = s * 10 + i_via_closure()
  # TODO(b/134822197): Remove i from return values.
  return s, i


def while_with_lambda_closure_call(n):
  # [FUNCTION DOC]
  # Test subject: like the closure-call while loop but the closure is a lambda returning i + 2.
  # Accumulates s = s * 10 + closure result while i < n, incrementing i. Returns s.
  i = 0
  s = 0
  i_via_closure = lambda: i + 2
  while i < n:
    s = s * 10 + i_via_closure()
    i += 1
  return s


def for_with_lambda_closure_call(l):
  # [FUNCTION DOC]
  # Test subject: for loop over l calling a lambda closure over i (returns i + 2) and
  # accumulating s. Returns the pair (s, i); the extra i is a workaround noted by a TODO.
  i = 0
  s = 0
  i_via_closure = lambda: i + 2
  for i in l:
    s = s * 10 + i_via_closure()
  # TODO(b/134822197): Remove i from return values.
  return s, i


def while_with_method_closure_call(n):
  # [FUNCTION DOC]
  # Test subject: closure implemented as an instance of a locally defined callable class that
  # returns the enclosing i; while i < n accumulates s = s * 10 + the call result and increments
  # i. Returns s. Its test is skipped (static analysis of nested classes unsupported).
  i = 0

  class Callable(object):
    # [CLASS DOC]
    # Local helper class whose instances are callable and return the enclosing variable i.

    def __call__(self):
      # [FUNCTION DOC]
      # HELPER: returns the value of the enclosing scope variable i (closure over the loop var).
      return i

  i_via_closure = Callable()
  i = 0
  s = 0
  while i < n:
    s = s * 10 + i_via_closure()
    i += 1
  return s


def for_with_method_closure_call(l):
  # [FUNCTION DOC]
  # Test subject: like the while version but iterating over l and rebinding i; accumulates
  # s = s * 10 + the callable object result. Returns the pair (s, i). Its test is skipped.
  i = 0

  class Callable(object):
    # [CLASS DOC]
    # Local helper class whose instances are callable and return the enclosing variable i.

    def __call__(self):
      # [FUNCTION DOC]
      # HELPER: returns the value of the enclosing scope variable i (closure over the loop var).
      return i

  i_via_closure = Callable()
  i = 0
  s = 0
  for i in l:
    s = s * 10 + i_via_closure()
  # TODO(b/134822197): Remove i from return values.
  return s, i


def global_fn(x):
  # [FUNCTION DOC]
  # HELPER: module-level callee used as a parameter value in tests; returns x times two.
  return x * 2


class TestClass(object):
  # [CLASS DOC]
  # HELPER: tiny class providing a bound method so tests can pass a method as the callee.

  def method(self, x):
    # [FUNCTION DOC]
    # HELPER: returns x times four; used as a bound-method callee in tests.
    return x * 4


def _int_tensor(x):
  # [FUNCTION DOC]
  # HELPER: converts a Python value to an int32 constant tensor; used to parameterize tests.
  return tf.constant(x, dtype=tf.int32)


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests asserting that loops with calls in conditions, targets and bodies
  # behave the same when compiled by tf.function as in eager Python.

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
      (global_fn, lambda x: x * 1, TestClass().method, abs),
  ))
  def test_while_with_call_in_cond(self, n, type_, fn):
    # [FUNCTION DOC]
    # Parameterized over n in 0-2, converters (int, tf.constant) and callees (global function,
    # lambda, bound method, builtin abs). Asserts the call-in-condition while loop matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_call_in_cond, n, fn)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
      (global_fn, lambda x: x * 1, TestClass().method, tf.abs),
  ))
  def test_for_with_call_in_target(self, l, type_, fn):
    # [FUNCTION DOC]
    # Parameterized over lists of length 0-2, converters (list, int32 tensor) and callees
    # (global function, lambda, bound method, tf.abs). Skips the combination of tf.abs with
    # a Python list (empty list would default to float). Asserts the loop matches eager.
    if fn is tf.abs and type_ is list:
      self.skipTest('tf.abs([]) defaults to float32')
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_call_in_target, l, fn)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, _int_tensor),
      (range, tf.range),
  ))
  def test_for_with_range_call_in_target(self, l, type_, fn):
    # [FUNCTION DOC]
    # Parameterized over n in 0-2, converters (int, int32 tensor) and range constructors (range,
    # tf.range) used as the callee. Asserts the for loop over a range call matches eager.
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_call_in_target, l, fn)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
      (global_fn, lambda x: x * 1, TestClass().method, abs),
  ))
  def test_while_with_call(self, n, type_, fn):
    # [FUNCTION DOC]
    # Parameterized over n, converters and callees (global function, lambda, method, abs).
    # Asserts the while loop with a call in its body matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_call, n, fn)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
      (global_fn, lambda x: x * 1, TestClass().method, abs),
  ))
  def test_for_with_call(self, l, type_, fn):
    # [FUNCTION DOC]
    # Parameterized over lists, converters and callees. Asserts the for loop with a call in its
    # body matches eager.
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_call, l, fn)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_local_call(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n in 0-2 and converters (int, tf.constant). Asserts the while loop that
    # calls a nested function matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_local_call, n)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_with_local_call(self, l, type_):
    # [FUNCTION DOC]
    # Parameterized over lists and converters. Asserts the for loop calling a nested function
    # matches eager.
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_local_call, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_closure_call(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and converters. Asserts the while loop calling a closure over the loop
    # variable matches eager.
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_closure_call, n)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_with_closure_call(self, l, type_):
    # [FUNCTION DOC]
    # Parameterized over lists and converters. Asserts the for loop calling a closure over the
    # loop variable matches eager.
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_closure_call, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_lambda_closure_call(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over n and converters. Asserts the while loop using a lambda closure matches
    # eager.
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_lambda_closure_call, n)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_with_lambda_closure_call(self, l, type_):
    # [FUNCTION DOC]
    # Parameterized over lists and converters. Asserts the for loop using a lambda closure
    # matches eager.
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_lambda_closure_call, l)

  @parameterized.parameters(*itertools.product(
      (0, 1, 2),
      (int, tf.constant),
  ))
  def test_while_with_method_closure_call(self, n, type_):
    # [FUNCTION DOC]
    # Skipped immediately because static analysis of nested classes is not supported. Intended:
    # assert the while loop calling a callable-object closure matches eager.
    self.skipTest('fix static analysis for nested classes')
    n = type_(n)
    self.assertFunctionMatchesEager(while_with_method_closure_call, n)

  @parameterized.parameters(*itertools.product(
      ([], [1], [1, 2]),
      (list, _int_tensor),
  ))
  def test_for_with_method_closure_call(self, l, type_):
    # [FUNCTION DOC]
    # Skipped immediately because static analysis of nested classes is not supported. Intended:
    # assert the for loop calling a callable-object closure matches eager.
    self.skipTest('fix static analysis for nested classes')
    l = type_(l)
    self.assertFunctionMatchesEager(for_with_method_closure_call, l)


if __name__ == '__main__':
  tf.test.main()
