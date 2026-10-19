# [MODULE DOC]
# Unit tests for the AutoGraph logical_expressions converter. Each test converts a small function
# using the converter_testing harness and evaluates the converted function on tensor and plain
# Python inputs. Covers equality, boolean operators, chained comparisons, default (unconverted)
# operators and unary operators. Runs the TensorFlow test main when executed as a script.
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
"""Tests for logical_expressions module."""

from tensorflow.python.autograph.converters import logical_expressions
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class LogicalExpressionTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case covering the logical_expressions converter (boolean, comparison and unary
  # operators). Inherits the transform and evaluate helpers from the converter testing base class.

  def test_equals(self):
    # [FUNCTION DOC]
    # Converts a function returning equality of its two arguments, then checks that a tensor
    # compared with an equal number gives true and with a different number gives false.

    def f(a, b):
      # [FUNCTION DOC]
      # Test subject: returns whether the two arguments are equal.
      return a == b

    tr = self.transform(f, logical_expressions)

    self.assertTrue(self.evaluate(tr(constant_op.constant(1), 1)))
    self.assertFalse(self.evaluate(tr(constant_op.constant(1), 2)))

  @test_util.run_deprecated_v1
  def test_bool_ops(self):
    # [FUNCTION DOC]
    # Converts a function combining or, and and not over three arguments; with a true tensor
    # first argument, checks the result is true when the third is false and false when it is
    # true. Runs under the deprecated v1 graph mode decorator.

    def f(a, b, c):
      # [FUNCTION DOC]
      # Test subject: returns the logical combination of or, and and not over the three arguments.
      return (a or b) and (a or b or c) and not c

    tr = self.transform(f, logical_expressions)

    self.assertTrue(self.evaluate(tr(constant_op.constant(True), False, False)))
    self.assertFalse(self.evaluate(tr(constant_op.constant(True), False, True)))

  def test_comparison(self):
    # [FUNCTION DOC]
    # Converts a function with a chained comparison of four arguments and checks it with only the
    # first argument a tensor: true when the chain holds, false when the last link fails. This
    # verifies the comparisons are evaluated in the right order.

    def f(a, b, c, d):
      # [FUNCTION DOC]
      # Test subject: returns a chained comparison (less, equal, greater) over four arguments.
      return a < b == c > d

    tr = self.transform(f, logical_expressions)

    # Note: having just the first constant a tensor tests that the
    # operations execute in the correct order. If anything other than
    # a < b executed first, the result would be a Python scalar and not a
    # Tensor. This is valid as long as the dispat is automatic based on
    # type.
    self.assertTrue(self.evaluate(tr(constant_op.constant(1), 2, 2, 1)))
    self.assertFalse(self.evaluate(tr(constant_op.constant(1), 2, 2, 3)))

  def test_default_ops(self):
    # [FUNCTION DOC]
    # Converts a function using the membership operator, which has no special conversion, and
    # checks it still works on plain strings and a tuple, returning true.

    def f(a, b):
      # [FUNCTION DOC]
      # Test subject: returns whether the first argument is a member of the second.
      return a in b

    tr = self.transform(f, logical_expressions)

    self.assertTrue(tr('a', ('a',)))

  def test_unary_ops(self):
    # [FUNCTION DOC]
    # Converts a function returning the bitwise invert, negation and unary plus of its argument
    # and checks that for the integer one the result is the tuple of minus two, minus one, one.

    def f(a):
      # [FUNCTION DOC]
      # Test subject: returns a tuple of the bitwise inverse, negation and unary plus of the argument.
      return ~a, -a, +a

    tr = self.transform(f, logical_expressions)

    self.assertEqual(tr(1), (-2, -1, 1))


if __name__ == '__main__':
  test.main()
