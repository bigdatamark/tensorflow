# [MODULE DOC]
# Reference tests for iterating over tf.data datasets and dataset iterators inside
# AutoGraph-converted functions (for loops, break, return, next calls, stop-iteration handling).
# Defines subject functions and one test class comparing with eager execution or checking the
# expected errors. No import-time side effects.
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
"""Tests involving the tf.data.Datasets API."""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def dataset_no_vars_loop(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Loops over dataset ds printing each element. Returns nothing.
  for e in ds:
    tf.print(e)


def iterator_no_vars_loop(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Loops over an iterator made from ds printing each element. Returns nothing.
  for e in iter(ds):
    tf.print(e)


def dataset_single_var_loop(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Folds the dataset into a 64-bit integer s, appending each element as a decimal digit.
  s = tf.constant(0, dtype=tf.int64)
  for e in ds:
    s = s * 10 + e
  return s


def iterator_single_var_loop(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Same as the dataset version but loops over an iterator of ds. Returns s.
  s = tf.constant(0, dtype=tf.int64)
  for e in iter(ds):
    s = s * 10 + e
  return s


def dataset_two_vars_loop(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Loops over ds computing the sum and product of its elements. Returns both.
  s = tf.constant(0, dtype=tf.int64)
  p = tf.constant(1, dtype=tf.int64)
  for e in ds:
    s += e
    p *= e
  return s, p


def iterator_two_vars_loop(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Same as the dataset version but over an iterator of ds. Returns sum and product.
  s = tf.constant(0, dtype=tf.int64)
  p = tf.constant(1, dtype=tf.int64)
  for e in iter(ds):
    s += e
    p *= e
  return s, p


def dataset_loop_with_break(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Appends elements as digits to s, stopping with a break once s exceeds one hundred. Returns s.
  s = tf.constant(0, dtype=tf.int64)
  for e in ds:
    s = s * 10 + e
    if s > 100:
      break
  return s


def iterator_loop_with_break(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Over an iterator of ds, sums elements, breaking once the sum exceeds one hundred. Returns it.
  # (Unlike the dataset version it adds rather than appending digits.)
  s = tf.constant(0, dtype=tf.int64)
  for e in iter(ds):
    s = s + e
    if s > 100:
      break
  return s


def iterator_resuming_loop(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Creates one iterator and consumes it in three successive loops: the first two take one
  # element each and break, the third drains the rest, appending elements as digits to s.
  s = tf.constant(0, dtype=tf.int64)
  itr = iter(ds)
  for e in itr:
    s = s * 10 + e
    break
  for e in itr:
    s = s * 10 + e
    break
  for e in itr:
    s = s * 10 + e
  return s


def dataset_loop_with_return(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns the first element of ds from inside the loop (unsupported in graph loops).
  y = tf.constant(0, dtype=tf.int64)
  for e in ds:
    y = e
    return y
  return y


def iterator_loop_with_return(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns the first element of an iterator of ds from inside the loop (unsupported).
  y = tf.constant(0, dtype=tf.int64)
  for e in iter(ds):
    y = e
    return y
  return y


def iterator_next(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Makes an iterator over ds and returns its first element.
  itr = iter(ds)
  return next(itr)


def iterator_next_multiple_calls(ds):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Takes the first two elements of an iterator of ds and returns ten times the first plus the
  # second.
  itr = iter(ds)
  return 10 * next(itr) + next(itr)


def iterator_next_in_loop(ds, n):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Calls next on an iterator of ds n times, appending each element as a digit to a 64-bit
  # integer. Returns it.
  itr = iter(ds)
  s = tf.constant(0, dtype=tf.int64)
  for _ in range(n):
    s = s * 10 + next(itr)
  return s


def iterator_next_stopping(ds, cond):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # While cond holds, keeps calling next on an iterator of ds, so it runs off the end. In graph
  # mode this raises an out-of-range error rather than a stop-iteration error. Returns nothing.
  # This case will raise, but not the expected StopIteration error.
  itr = iter(ds)
  while cond:
    next(itr)


def iterator_next_with_catching_stop_iteration(ds, cond):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Like the stopping subject but wrapped in a try that catches stop-iteration. In graph mode the
  # catch never triggers and an out-of-range error escapes. Returns nothing.
  # This is the only instance when the use of TF iterators does not work as
  # intended. In graph mode, the `except` below will never catch, and the
  # tf.function will raise the error instead.
  # TODO(b/132311724): The error should be friendlier here.
  # Note: b/132298783 covers actually supporting this pattern.
  itr = iter(ds)
  try:
    while cond:
      next(itr)
  except StopIteration:
    pass


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Test case for dataset iteration under AutoGraph. Fixture: a dataset of integers zero to six
  # stored as the ds attribute in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp and creates the range-of-seven dataset used by all tests.
    super(ReferenceTest, self).setUp()
    self.ds = tf.data.Dataset.range(7)

  def test_dataset_no_vars_loop(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for the print-only dataset loop.
    self.assertFunctionMatchesEager(dataset_no_vars_loop, self.ds)

  def test_iterator_no_vars_loop(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for the print-only iterator loop.
    self.assertFunctionMatchesEager(iterator_no_vars_loop, self.ds)

  def test_dataset_single_var_loop(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for the single-variable dataset loop.
    self.assertFunctionMatchesEager(dataset_single_var_loop, self.ds)

  def test_iterator_single_var_loop(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for the single-variable iterator loop.
    self.assertFunctionMatchesEager(iterator_single_var_loop, self.ds)

  def test_dataset_two_vars_loop(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for the two-variable dataset loop.
    self.assertFunctionMatchesEager(dataset_two_vars_loop, self.ds)

  def test_iterator_two_vars_loop(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for the two-variable iterator loop.
    self.assertFunctionMatchesEager(iterator_two_vars_loop, self.ds)

  def test_dataset_loop_with_break(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for the dataset loop with break.
    self.assertFunctionMatchesEager(dataset_loop_with_break, self.ds)

  def test_iterator_loop_with_break(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for the iterator loop with break.
    self.assertFunctionMatchesEager(iterator_loop_with_break, self.ds)

  def test_dataset_loop_with_return_raises(self):
    # [FUNCTION DOC]
    # Expects tf.function on the dataset loop containing a return to raise a not-implemented error
    # saying a return cannot be placed inside this TensorFlow loop.
    # This is for the same reason why returns in loops aren't allowed.
    # TODO(mdan): This might be resolved by unrolling the loop once.
    with self.assertRaisesRegex(
        NotImplementedError,
        'a return statement cannot be placed inside this TensorFlow loop'):
      tf.function(dataset_loop_with_return)(self.ds)

  def test_iterator_loop_with_return_raises(self):
    # [FUNCTION DOC]
    # Same expectation as the dataset variant but for the iterator loop with a return.
    # This is for the same reason why returns in loops aren't allowed.
    # TODO(mdan): This might be resolved by unrolling the loop once.
    with self.assertRaisesRegex(
        NotImplementedError,
        'a return statement cannot be placed inside this TensorFlow loop'):
      tf.function(iterator_loop_with_return)(self.ds)

  def test_iterator_next(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match when taking one element from an iterator.
    self.assertFunctionMatchesEager(iterator_next, self.ds)

  def test_iterator_next_multiple_calls(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match for two successive next calls.
    self.assertFunctionMatchesEager(iterator_next_multiple_calls, self.ds)

  def test_iterator_next_in_loop(self):
    # [FUNCTION DOC]
    # Asserts graph and eager results match when calling next seven times in a loop.
    self.assertFunctionMatchesEager(iterator_next_in_loop, self.ds, 7)

  def test_iterator_next_stopping(self):
    # [FUNCTION DOC]
    # Running off the end of the iterator in graph mode must raise an out-of-range error (eager
    # would raise stop-iteration), with condition passed as a true tensor.
    # Graph ops raise OutOfRangeError, but eager ops raise StopIteration
    with self.assertRaises(tf.errors.OutOfRangeError):
      tf.function(iterator_next_stopping)(self.ds, tf.constant(True))

  def test_iterator_next_with_catching_stop_iteration(self):
    # [FUNCTION DOC]
    # Even with a stop-iteration handler in the subject, graph mode must raise an out-of-range
    # error since the handler cannot catch it.
    # Graph ops raise OutOfRangeError, but eager ops raise StopIteration
    with self.assertRaises(tf.errors.OutOfRangeError):
      tf.function(iterator_next_with_catching_stop_iteration)(
          self.ds, tf.constant(True))


if __name__ == '__main__':
  tf.test.main()
