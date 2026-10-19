# [MODULE DOC]
# Reference test for autograph handling of print calls (the Python builtin and the TensorFlow
# print operation): single and multiple values, non-tensor values, printing inside a conditional.
# The comparison covers both results and captured standard output. Builtin-function conversion
# is enabled for these tests. No import-time side effects.
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
"""Simple call to a print function preceding other computations.

The call may be wrapped inside a py_func, but tf.Print should be used if
possible. The subsequent computations will be gated by the print function
execution.
"""

import numpy as np
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def lone_print(x):
  # [FUNCTION DOC]
  # Function under test: prints x with the builtin print; returns nothing.
  print(x)


def print_multiple_values(x):
  # [FUNCTION DOC]
  # Function under test: prints a label string followed by x with the builtin print.
  print('x is', x)


def multiple_prints(x, y):
  # [FUNCTION DOC]
  # Function under test: prints x then y, each with a label, using the TensorFlow print
  # operation (two consecutive print operations).
  tf.print('x is', x)
  tf.print('y is', y)


def print_with_nontf_values(x):
  # [FUNCTION DOC]
  # Function under test: prints a label, x and a plain dictionary (a non-tensor value) with the
  # builtin print.
  print('x is', x, {'foo': 'bar'})


def print_in_cond(x):
  # [FUNCTION DOC]
  # Function under test: prints x with the builtin print only when x equals zero.
  if x == 0:
    print(x)


def tf_print(x):
  # [FUNCTION DOC]
  # Function under test: prints x with the TensorFlow print operation.
  tf.print(x)


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Reference test case for print calls, with only builtin-function conversion enabled.

  def setUp(self):
    # [FUNCTION DOC]
    # Runs base setup, then enables only the builtin-functions conversion feature.
    super(ReferenceTest, self).setUp()
    self.autograph_opts = tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS

  def test_lone_print(self):
    # [FUNCTION DOC]
    # Asserts the single print sample matches eager results and output for a number and an array.
    self.assertFunctionMatchesEager(lone_print, 1)
    self.assertFunctionMatchesEager(lone_print, np.array([1, 2, 3]))

  def test_print_multiple_values(self):
    # [FUNCTION DOC]
    # Asserts the multi-value print sample matches eager results and output for a number and an
    # array.
    self.assertFunctionMatchesEager(print_multiple_values, 1)
    self.assertFunctionMatchesEager(print_multiple_values, np.array([1, 2, 3]))

  def test_multiple_prints(self):
    # [FUNCTION DOC]
    # Asserts the two-print sample matches eager results and output for number and array inputs.
    self.assertFunctionMatchesEager(multiple_prints, 1, 2)
    self.assertFunctionMatchesEager(multiple_prints, np.array([1, 2, 3]), 4)

  def test_print_with_nontf_values(self):
    # [FUNCTION DOC]
    # Asserts the sample printing a dictionary alongside x matches eager results and output for
    # a number and an array.
    self.assertFunctionMatchesEager(print_with_nontf_values, 1)
    self.assertFunctionMatchesEager(print_with_nontf_values, np.array([1, 2,
                                                                       3]))

  def test_print_in_cond(self):
    # [FUNCTION DOC]
    # Asserts the conditional print sample matches eager results and output when the condition
    # is true (zero) and false (one).
    self.assertFunctionMatchesEager(print_in_cond, 0)
    self.assertFunctionMatchesEager(print_in_cond, 1)

  def test_tf_print(self):
    # [FUNCTION DOC]
    # Asserts the TensorFlow print sample matches eager results and output for the input zero.
    self.assertFunctionMatchesEager(tf_print, 0)


if __name__ == '__main__':
  tf.test.main()
