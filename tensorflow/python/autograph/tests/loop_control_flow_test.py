# [MODULE DOC]
# Reference tests for loop control statements (break, continue, early return) in single and nested
# for and while loops under AutoGraph, over Python lists, tensors and datasets, compared with eager
# execution. Defines many subject functions, small input-conversion helpers and one test class.
# No import-time side effects.
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
"""Nested loops and loop control statements (e.g. break and continue).

Meant to verify that:
  * break/continue in the inner loop does not affect outer loop
  * break/continue inside nested conditionals still works
"""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def continue_in_single_for(l):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Sums the even elements of l, skipping odd ones with continue. Returns the sum.
  s = 0
  for c in l:
    if c % 2 > 0:
      continue
    s += c
  return s


def continue_in_single_while(x):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Counts x down, adding each even value to s and skipping odd ones with continue. Returns s.
  s = 0
  while x > 0:
    x -= 1
    if x % 2 > 0:
      continue
    s += x
  return s


def continue_in_inner_for(m):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Sums even elements over a sequence of sequences, using continue in the inner loop.
  s = 0
  for l in m:
    for c in l:
      if c % 2 > 0:
        continue
      s += c
  return s


def continue_in_inner_while(x, y):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Nested while loops on x and y; adds x plus y to s when it is even, otherwise continues.
  s = 0
  while x > 0:
    x -= 1
    while y > 0:
      y -= 1
      if (x + y) % 2 > 0:
        continue
      s += x + y
  return s


def break_in_single_for(l):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Sums elements of l until the first odd one, which triggers break. Returns the sum.
  s = 0
  for c in l:
    if c % 2 > 0:
      break
    s += c
  return s


def unconditional_return_in_single_for(l):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Adds the first element of l to s and returns inside the loop; with empty l returns zero.
  s = 0
  for c in l:
    s += c
    return s
  return s


def effectively_unconditional_return_in_single_for(l):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Adds the first element to s and returns s if it is odd, else minus s; both branches return.
  s = 0
  for c in l:
    s += c
    if c % 2 > 0:
      return s
    else:
      return -s
  return s


def unconditional_return_in_single_while(x):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # While x is positive, decrements it, adds it to s and returns s at once. Else returns s.
  s = 0
  while x > 0:
    x -= 1
    s += x
    return s
  return s


def effectively_unconditional_return_in_single_while(x):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Decrements x, adds to s, then returns s or minus s depending on parity of x; always returns
  # in the first iteration.
  s = 0
  while x > 0:
    x -= 1
    s += x
    if x % 2 > 0:
      return s
    else:
      return -s
  return s


def break_in_single_while(x):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Counts x down, breaking at the first odd value, otherwise adding it to s. Returns s.
  s = 0
  while x > 0:
    x -= 1
    if x % 2 > 0:
      break
    s += x
  return s


def break_in_inner_for(m):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Nested for loops; the inner loop breaks at an odd element, otherwise summing. Returns the sum.
  s = 0
  for l in m:
    for c in l:
      if c % 2 > 0:
        break
      s += c
  return s


def break_in_inner_while(x, y):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Nested while loops on x and y; the inner loop breaks when x plus y is even, else adds it.
  s = 0
  while x > 0:
    x -= 1
    while y > 0:
      y -= 1
      if ((x + y) % 2) == 0:
        break
      s += x + y
  return s


def break_continue_in_inner_for(m):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # KNOWN ISSUE: both branches break or continue, so the sum update is unreachable and s stays 0.
  # Nested for loops; inner loop breaks on odd elements and continues otherwise.
  s = 0
  for l in m:
    for c in l:
      if c % 2 > 0:
        break
      else:
        continue
      s += c
  return s


def break_continue_in_inner_while(x, y):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # KNOWN ISSUE: both branches break or continue, so the sum update is unreachable and s stays 0.
  # Nested while loops; the inner loop breaks if x plus y is odd, else continues.
  s = 0
  while x > 0:
    x -= 1
    while y > 0:
      y -= 1
      if (x + y) % 2 > 0:
        break
      else:
        continue
      s += x + y
  return s


def break_followed_by_cond_in_single_for(x, y):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Loops i over range y, breaking at 2, otherwise decrementing x if positive. Returns x.
  for i in range(y):
    if i == 2:
      break
    if x > 0:
      x -= 1
  return x


def break_followed_by_cond_in_single_while(x):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # While x is positive, breaks when x is 2, otherwise decrements x. Returns x.
  while x > 0:
    if x == 2:
      break
    if x > 0:
      x -= 1
  return x


def multiple_breaks_in_single_while(n):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowTest.
  # Counts i up to n; breaks if i exceeds ten times n or equals n, else appends i as a digit to s
  # (starting at one). Returns i and s.
  s = 1
  i = 0
  while i < n:
    i += 1
    if i > 10 * n:
      break
    if i == n:
      break
    s = s * 10 + i
  return i, s


def _int_tensor(x):
  # [FUNCTION DOC]
  # HELPER: converts test inputs for LoopControlFlowTest parameterization.
  # Converts x to a 32-bit integer constant tensor.
  return tf.constant(x, dtype=tf.int32)


def _list_of_int_tensor(l):
  # [FUNCTION DOC]
  # HELPER: converts test inputs for LoopControlFlowTest parameterization.
  # Converts each element of l into its own 32-bit integer constant tensor; returns a list.
  return [_int_tensor(x) for x in l]


def _int_dataset(l):
  # [FUNCTION DOC]
  # HELPER: converts test inputs for LoopControlFlowTest parameterization.
  # Returns a dataset over the slices of list l as a 32-bit integer tensor.
  return tf.data.Dataset.from_tensor_slices(tf.constant(l, dtype=tf.int32))


class LoopControlFlowTest(reference_test_base.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case comparing graph and eager results of single and nested loop control
  # flow subjects over several input kinds. No shared fixtures.

  @parameterized.parameters(*itertools.product(
      (
          [],
          [1],
          [1, 2],
          [1, 2, 3],
          [1, 2, 3, 4],
      ),
      (
          list,
          _int_tensor,
          _int_dataset,
      ),
      (
          continue_in_single_for,
          break_in_single_for,
          unconditional_return_in_single_for,
          effectively_unconditional_return_in_single_for,
      ),
  ))
  def test_single_for(self, l, type_, target):
    # [FUNCTION DOC]
    # Parameterized over list contents, input kind (list, tensor, dataset) and subject. Skips
    # dataset inputs for the return subjects and empty inputs for them. Asserts graph and eager
    # results match.
    if ((type_ is _int_dataset) and
        (target in (unconditional_return_in_single_for,
                    effectively_unconditional_return_in_single_for))):
      # TODO(mdan): Enable in a separate improvement.
      self.skipTest('Creating symbols in dataset loops.')

    if ((not l) and
        ((target in (unconditional_return_in_single_for,
                     effectively_unconditional_return_in_single_for)))):
      self.skipTest('Undefined symbols require at least one iteration.')

    l = type_(l)
    self.assertFunctionMatchesEager(target, l)

  @parameterized.parameters(*itertools.product(
      (
          0,
          1,
          2,
          3,
          4,
      ),
      (
          int,
          _int_tensor,
      ),
      (
          continue_in_single_while,
          break_in_single_while,
          multiple_breaks_in_single_while,
          break_followed_by_cond_in_single_while,
          unconditional_return_in_single_while,
          effectively_unconditional_return_in_single_while,
      ),
  ))
  def test_single_while(self, n, type_, target):
    # [FUNCTION DOC]
    # Parameterized over n, int or tensor and single-while subjects. Skips return subjects with
    # zero. Asserts graph and eager results match.
    if ((not n) and
        ((target in (unconditional_return_in_single_while,
                     effectively_unconditional_return_in_single_while)))):
      self.skipTest('Undefined symbols require at least one iteration.')

    n = type_(n)
    self.assertFunctionMatchesEager(target, n)

  @parameterized.parameters(
      (unconditional_return_in_single_for, _int_tensor, []),
      (effectively_unconditional_return_in_single_for, _int_tensor, []),
      (unconditional_return_in_single_while, _int_tensor, 0),
      (effectively_unconditional_return_in_single_while, _int_tensor, 0),
  )
  def test_single_loop_illegal_return(self, target, type_, l):
    # [FUNCTION DOC]
    # For return-in-loop subjects with empty input as tensors, expects an invalid-argument error
    # saying the loop must iterate at least once to initialize.
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'must iterate at least once to initialize'):
      tf.function(target)(type_(l))

  @parameterized.parameters(*itertools.product(
      (
          [[], []],
          [[1], [2]],
          [[1, 2], [3, 4]],
          [[1, 2, 3], [4, 5, 6]],
          # TODO(mdan): Add ragged tensors / variable-shape datasets.
      ),
      (
          list,
          _int_tensor,
          _list_of_int_tensor,
          _int_dataset,
      ),
      (
          continue_in_inner_for,
          break_in_inner_for,
          break_continue_in_inner_for,
      ),
  ))
  def test_nested_for(self, a, type_, target):
    # [FUNCTION DOC]
    # Parameterized over nested lists, input kind and nested-for subjects; asserts graph and eager
    # results match.
    a = type_(a)
    self.assertFunctionMatchesEager(target, a)

  @parameterized.parameters(*itertools.product(
      (
          0,
          1,
          2,
          3,
          4,
      ),
      (
          0,
          1,
          2,
          3,
          4,
      ),
      (
          int,
          _int_tensor,
      ),
      (
          int,
          _int_tensor,
      ),
      (
          continue_in_inner_while,
          break_in_inner_while,
          break_continue_in_inner_while,
      ),
  ))
  def test_nested_while(self, m, n, m_type, n_type, target):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second count is converted with the first type (m_type); n_type is unused.
    # Parameterized over counts, types and nested-while subjects; asserts graph/eager match.
    m = m_type(m)
    n = m_type(n)
    self.assertFunctionMatchesEager(target, m, n)


if __name__ == '__main__':
  tf.test.main()
