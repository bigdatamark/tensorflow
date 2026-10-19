# [MODULE DOC]
# Reference tests for logical and comparison expressions (or, and, equality, inequality, chained
# comparisons) in AutoGraph-converted functions, with plain Python values and with all inputs
# converted to tensors. Contains subject functions and one test class. No import-time side effects.
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
"""Basic logical expressions that are not autoboxed to TF."""

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def composite_ors_with_callable(x, y, z):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns x or y or the result of calling a zero-argument function that returns z.
  z1 = lambda: z
  return x or y or z1()


def composite_ors(x, y, z):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns the logical or of x, y and z.
  return x or y or z


def composite_ands(x, y, z):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns the logical and of x, y and z.
  return x and y and z


def composite_mixed(x, y, z):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns x or y or (z and y and z), mixing or and and with and binding tighter.
  return x or y or z and y and z


def equality(x, y):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns whether x equals y.
  return x == y


def inequality(x, y):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns whether x differs from y.
  return x != y


def multiple_equality(x, y, z):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns the chained comparison x equals y equals z.
  return x == y == z


def comparison(x, y, z):
  # [FUNCTION DOC]
  # HELPER: test subject for ReferenceTest.
  # Returns whether x is less than y and y is less than z.
  return x < y and y < z


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Test case comparing graph and eager results for the logical-expression subjects, using the
  # base class settings for tensor inputs and autograph options.

  def test_basic(self):
    # [FUNCTION DOC]
    # With plain Python inputs, asserts graph and eager results match for or, and, mixed, callable
    # and ordering-comparison subjects over several truth combinations and orderings.
    self.assertFunctionMatchesEager(composite_ors, False, True, False)
    self.assertFunctionMatchesEager(composite_ors, False, False, False)
    self.assertFunctionMatchesEager(composite_ands, True, True, True)
    self.assertFunctionMatchesEager(composite_ands, True, False, True)
    self.assertFunctionMatchesEager(composite_mixed, False, True, True)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, True,
                                    False)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, False,
                                    True)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, False,
                                    False)

    self.assertFunctionMatchesEager(comparison, 1, 2, 3)
    self.assertFunctionMatchesEager(comparison, 2, 1, 3)
    self.assertFunctionMatchesEager(comparison, 3, 2, 1)
    self.assertFunctionMatchesEager(comparison, 3, 1, 2)
    self.assertFunctionMatchesEager(comparison, 1, 3, 2)
    self.assertFunctionMatchesEager(comparison, 2, 3, 1)

  def test_basic_tensor(self):
    # [FUNCTION DOC]
    # Same checks as the basic test but sets the base class flag so all inputs are converted to
    # tensors first. Asserts graph and eager results match.
    self.all_inputs_tensors = True
    self.assertFunctionMatchesEager(composite_ors, False, True, False)
    self.assertFunctionMatchesEager(composite_ors, False, False, False)
    self.assertFunctionMatchesEager(composite_ands, True, True, True)
    self.assertFunctionMatchesEager(composite_ands, True, False, True)
    self.assertFunctionMatchesEager(composite_mixed, False, True, True)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, True,
                                    False)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, False,
                                    True)
    self.assertFunctionMatchesEager(composite_ors_with_callable, False, False,
                                    False)

    self.assertFunctionMatchesEager(comparison, 1, 2, 3)
    self.assertFunctionMatchesEager(comparison, 2, 1, 3)
    self.assertFunctionMatchesEager(comparison, 3, 2, 1)
    self.assertFunctionMatchesEager(comparison, 3, 1, 2)
    self.assertFunctionMatchesEager(comparison, 1, 3, 2)
    self.assertFunctionMatchesEager(comparison, 2, 3, 1)

  def test_equality(self):
    # [FUNCTION DOC]
    # With plain integers, asserts graph and eager results match for equality, inequality and
    # chained equality.
    self.assertFunctionMatchesEager(equality, 1, 1)
    self.assertFunctionMatchesEager(equality, 1, 2)
    self.assertFunctionMatchesEager(inequality, 1, 1)
    self.assertFunctionMatchesEager(inequality, 1, 2)
    self.assertFunctionMatchesEager(multiple_equality, 1, 1, 2)
    self.assertFunctionMatchesEager(multiple_equality, 1, 1, 1)

  def test_equality_tensor(self):
    # [FUNCTION DOC]
    # Enables the equality-operators AutoGraph feature and tensor inputs (by setting attributes
    # on the test case), then asserts graph and eager results match for equality subjects.
    self.autograph_opts = tf.autograph.experimental.Feature.EQUALITY_OPERATORS
    self.all_inputs_tensors = True

    self.assertFunctionMatchesEager(equality, 1, 1)
    self.assertFunctionMatchesEager(equality, 1, 2)
    self.assertFunctionMatchesEager(inequality, 1, 1)
    self.assertFunctionMatchesEager(inequality, 1, 2)
    self.assertFunctionMatchesEager(multiple_equality, 1, 1, 2)
    self.assertFunctionMatchesEager(multiple_equality, 1, 1, 1)


if __name__ == '__main__':
  tf.test.main()
