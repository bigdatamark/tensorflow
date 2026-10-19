# [MODULE DOC]
# Tests that AutoGraph rejects break and return statements guarded by a tensor condition inside
# Python-native (non-tensor) for and while loops. Defines four subject functions and one test
# class asserting a not-implemented error is raised. No import-time side effects.
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
"""Loop control statements (e.g. break, return) in illegal patterns.

Meant to verify that:
  * break/return on a dynamic condition raises error inside Python loop
"""

import itertools
import re

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def tf_break_in_py_for(l):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowIllegalCasesTest.
  # Python for loop over l that breaks when a tensor comparison says the element is odd,
  # otherwise accumulates the element. Returns the sum.
  s = 0
  for c in l:
    if tf.greater(c % 2, 0):
      break
    s += c
  return s


def tf_return_in_py_for(l):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowIllegalCasesTest.
  # KNOWN ISSUE: both branches return, so the accumulation line is unreachable dead code.
  # Python for loop over l that returns the running sum from either branch of a tensor condition.
  s = 0
  for c in l:
    if tf.greater(c % 2, 0):
      return s
    else:
      return s
    s += c
  return s


def tf_break_in_py_while(x):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowIllegalCasesTest.
  # While x is positive, decrements it and breaks if a tensor comparison says it is odd, else
  # adds it to s. Returns s.
  s = 0
  while x > 0:
    x -= 1
    if tf.greater(x % 2, 0):
      break
    s += x
  return s


def tf_return_in_py_while(x):
  # [FUNCTION DOC]
  # HELPER: test subject for LoopControlFlowIllegalCasesTest.
  # KNOWN ISSUE: both branches return, so the accumulation line is unreachable dead code.
  # While x is positive, decrements it and returns s from either branch of a tensor condition.
  s = 0
  while x > 0:
    x -= 1
    if tf.greater(x % 2, 0):
      return s
    else:
      return s
    s += x
  return s


class LoopControlFlowIllegalCasesTest(reference_test_base.TestCase,
                                      parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case verifying that tensor-conditioned break/return in Python loops raises
  # a not-implemented error. No shared fixtures.

  @parameterized.parameters(*itertools.product(
      (
          [1],
          [1, 2],
          [1, 2, 3],
      ),
      (
          tf_break_in_py_for,
          tf_return_in_py_for,
      ),
  ))
  def test_tf_control_flow_in_py_for(self, l, target):
    # [FUNCTION DOC]
    # Parameterized over short lists and the for-loop break/return subjects. Asserts tf.function
    # raises a not-implemented error saying the construct is not supported in Python for loops.
    with self.assertRaisesRegex(NotImplementedError,
                                'not supported in Python for'):
      tf.function(target)(l)

  @parameterized.parameters(*itertools.product(
      (
          1,
          2,
          3,
      ),
      (
          tf_break_in_py_while,
          tf_return_in_py_while,
      ),
  ))
  def test_tf_control_flow_in_py_while(self, n, target):
    # [FUNCTION DOC]
    # Parameterized over n and the while-loop break/return subjects. Asserts tf.function raises a
    # not-implemented error saying the while condition started as non-tensor then became a tensor.
    with self.assertRaisesRegex(
        NotImplementedError,
        re.compile(
            r'.*condition of while loop started as non\-Tensor,'
            r' then changed to Tensor.*', re.DOTALL)):
      tf.function(target)(n)


if __name__ == '__main__':
  tf.test.main()
