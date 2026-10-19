# [MODULE DOC]
# Tests for the AutoGraph conditional-expression converter: converted functions using simple and
# nested ternary expressions must behave like the originals.
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
"""Tests for conditional_expressions module."""

from tensorflow.python.autograph.converters import conditional_expressions
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class ConditionalExpressionsTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the conditional-expression converter, built on the converter testing base class
  # (provides the transform helper). Defines an equivalence-check helper.

  def assertTransformedEquivalent(self, f, *inputs):
    # [FUNCTION DOC]
    # HELPER: assertion shared by the tests. Converts f with the conditional-expression converter
    # and asserts the original and converted functions return equal results for the inputs.
    tr = self.transform(f, conditional_expressions)
    self.assertEqual(f(*inputs), tr(*inputs))

  def test_basic(self):
    # [FUNCTION DOC]
    # A simple ternary choosing 1 or 0; equivalence checked for inputs 0 and 3.

    def f(x):
      # [FUNCTION DOC]
      # Sample function: returns 1 if x is truthy, else 0.
      return 1 if x else 0

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 3)

  def test_nested_orelse(self):
    # [FUNCTION DOC]
    # A ternary nested in the else branch of another; equivalence checked for inputs -2, 0, 2.

    def f(x):
      # [FUNCTION DOC]
      # Sample function: square of x if positive, else x if truthy, else 1.
      y = x * x if x > 0 else x if x else 1
      return y

    self.assertTransformedEquivalent(f, -2)
    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 2)


if __name__ == '__main__':
  test.main()
