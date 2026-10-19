# [MODULE DOC]
# Reference tests for AutoGraph loops over tf.distribute (mirrored strategy) distributed datasets
# and iterators: for loops, break, continue, enumerate and explicit next calls. Defines subject
# functions, helpers that build a two-virtual-CPU strategy with a distributed dataset, and a test
# class. Helpers configure virtual CPU devices (global device state).
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
"""Tests involving the tf.distributed datasets."""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def no_vars_loop(strat, iterable):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Prints the summed per-replica value of each element of the iterable. Returns nothing.
  for pr in iterable:
    tf.print(strat.reduce('SUM', pr, axis=0))


def single_var_loop(strat, iterable):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Appends each reduced element as a decimal digit to s. Returns s.
  s = 0
  for pr in iterable:
    # TODO(mdan): It would be nice to be able to write s = s * 10 + pr.
    s = s * 10 + strat.reduce('SUM', pr, axis=0)
  return s


def loop_with_break(strat, iterable):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Like the single-variable loop but breaks when the reduced element is divisible by five.
  s = 0
  for pr in iterable:
    if strat.reduce('SUM', pr, axis=0) % 5 == 0:
      break
    s = s * 10 + strat.reduce('SUM', pr, axis=0)
  return s


def loop_with_continue(strat, iterable):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Skips elements whose reduction is even; appends odd ones as digits to s. Returns s.
  s = 0
  for pr in iterable:
    if strat.reduce('SUM', pr, axis=0) % 2 == 0:
      continue
    s = s * 10 + strat.reduce('SUM', pr, axis=0)
  return s


def two_vars_loop(strat, iterable):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Computes the sum and product of the reduced elements. Returns both.
  s = 0
  p = 1
  for pr in iterable:
    e = strat.reduce('SUM', pr, axis=0)
    s += e
    p *= e
  return s, p


def enumeration(strat, iterable):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Enumerates the iterable, appending reduced elements as digits to s and multiplying p by the
  # index (so p ends up zero whenever the first element is visited). Returns s and p.
  s = 0
  p = 1
  for i, pr in enumerate(iterable):
    e = strat.reduce('SUM', pr, axis=0)
    s = s * 10 + e
    p *= i
  return s, p


def iterator_next(strat, iterable):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns the reduced first element of an iterator made from the iterable.
  itr = iter(iterable)
  return strat.reduce('SUM', next(itr), axis=0)


def iterator_next_multiple_calls(strat, iterable):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Reduces the first two elements and returns ten times the first plus the second.
  itr = iter(iterable)
  a = strat.reduce('SUM', next(itr), axis=0)
  b = strat.reduce('SUM', next(itr), axis=0)
  return a * 10 + b


def iterator_next_in_limited_loop(strat, iterable, l):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # For each item in l, takes the next element and appends its reduction as a digit to s.
  # Returns s.
  itr = iter(iterable)
  s = 0
  for _ in l:
    s = s * 10 + strat.reduce('SUM', next(itr), axis=0)
  return s


def iterator_next_stopping(strat, iterable, cond):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # While cond holds, keeps taking elements, running past the end; in graph mode an out-of-range
  # error is raised instead of stop-iteration. Returns nothing.
  # This case will raise, but not the expected StopIteration error.
  itr = iter(iterable)
  while cond:
    strat.reduce('SUM', next(itr), axis=0)


def iterator_next_with_catching_stop_iteration(strat, iterable, cond):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Same as the stopping subject but wrapped in a catch for stop-iteration, which never fires in
  # graph mode. Returns nothing.
  # This is the one instance when the use of TF iterators does not work as
  # intended. In graph mode, the `except` below will never catch, and the
  # tf.function will raise the error instead.
  # TODO(b/132311724): The error should be friendlier here.
  # Note: b/132298783 covers actually supporting this pattern.
  itr = iter(iterable)
  try:
    while cond:
      strat.reduce('SUM', next(itr), axis=0)
  except StopIteration:
    pass


def _distributed_dataset():
  # [FUNCTION DOC]
  # HELPER: builds the distributed dataset fixture for ReferenceTest.
  # Splits the first physical CPU into two virtual devices (global device configuration, fails if
  # already initialized differently), creates a mirrored strategy and a dataset of three rows of
  # four integers, distributes it. Returns the strategy and the distributed dataset.
  cpus = tf.config.experimental.list_physical_devices('CPU')
  tf.config.experimental.set_virtual_device_configuration(
      cpus[0], [tf.config.experimental.VirtualDeviceConfiguration()] * 2)

  strat = tf.distribute.MirroredStrategy()
  ds = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.range(12), (3, 4)))

  return strat, strat.experimental_distribute_dataset(ds)


def _distributed_iterator():
  # [FUNCTION DOC]
  # HELPER: builds the distributed iterator fixture for ReferenceTest.
  # Calls the dataset builder and returns the strategy with an iterator over the dataset.
  strat, ds = _distributed_dataset()
  return strat, iter(ds)


class ReferenceTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case comparing graph and eager results of loops over distributed datasets
  # and iterators. Fixtures are built per test via the module helpers.

  @parameterized.parameters(*itertools.product(
      (
          no_vars_loop,
          single_var_loop,
          two_vars_loop,
          loop_with_break,
          loop_with_continue,
      ),
      (
          _distributed_dataset,
          _distributed_iterator,
      ),
  ))
  def test_basic(self, test_fn, target):
    # [FUNCTION DOC]
    # Parameterized over loop subjects and the dataset/iterator builder. Skips break and continue
    # subjects with the dataset builder (known bug). Compares using the stateful-input helper.
    if (test_fn in (loop_with_break, loop_with_continue) and
        target is _distributed_dataset):
      self.skipTest('b/162250181')
    self.assertFunctionMatchesEagerStatefulInput(test_fn, target)

  def test_iterator_next(self):
    # [FUNCTION DOC]
    # Builds the strategy and dataset, asserts graph and eager results match for one next call.
    strat, ds = _distributed_dataset()
    self.assertFunctionMatchesEager(iterator_next, strat, ds)

  def test_iterator_next_multiple_calls(self):
    # [FUNCTION DOC]
    # Builds the strategy and dataset, asserts graph and eager results match for two next calls.
    strat, ds = _distributed_dataset()
    self.assertFunctionMatchesEager(iterator_next_multiple_calls, strat, ds)

  @parameterized.parameters(*itertools.product(
      (
          0,
          1,
          2,
      ),
      (
          range,
          tf.range,
      ),
  ))
  def test_iterator_next_in_limited_loop(self, n, type_):
    # [FUNCTION DOC]
    # Parameterized over count 0 to 2 as a Python or tensor range; asserts graph and eager results
    # match for taking that many elements.
    n = type_(n)
    strat, ds = _distributed_dataset()
    self.assertFunctionMatchesEager(iterator_next_in_limited_loop, strat, ds, n)

  @parameterized.parameters(
      (iterator_next_stopping,),
      # Note that `except` has no effect in graph mode.
      (iterator_next_with_catching_stop_iteration,),
  )
  def test_iterator_next_stopping(self, test_fn):
    # [FUNCTION DOC]
    # For both stopping subjects, expects tf.function to raise an out-of-range error when running
    # past the end with a true tensor condition.
    strat, ds = _distributed_dataset()
    with self.assertRaises(tf.errors.OutOfRangeError):
      tf.function(test_fn)(strat, ds, tf.constant(True))


if __name__ == '__main__':
  tf.test.main()
