# [MODULE DOC]
# Unit tests for the multiplication dispatch helper in the math operator overrides module,
# covering numeric tensors, boolean tensors and mixed boolean/numeric operands. Test module only.
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
"""Tests for the math operator overrides."""


from tensorflow.python.framework import constant_op
from tensorflow.python.ops import tensor_math_operator_overrides as tmoo
from tensorflow.python.platform import test


class SortTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the name suggests sorting tests, but it actually tests the multiplication
  # dispatch helper of the math operator overrides module (name looks copy-pasted).
  # Test case for the multiplication dispatch factory function; no shared fixtures other than
  # one private assertion helper.

  def _test_mul_dispatch_factory(self, x, y, expected, name=None):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the multiplication dispatch tests in this class.
    # Calls the module's multiplication dispatch function on x and y with the optional op name
    # and asserts the result equals expected. Returns nothing; fails the test on mismatch.
    self.assertAllEqual(expected, tmoo._mul_dispatch_factory(x, y, name=name))

  def testNonBooleanTensor(self):
    # [FUNCTION DOC]
    # Multiplies two integer vectors and expects elementwise products 4, 10, 18.
    x = constant_op.constant([1, 2, 3])
    y = constant_op.constant([4, 5, 6])
    expected = constant_op.constant([4, 10, 18])
    self._test_mul_dispatch_factory(x, y, expected)

  def testBooleanTensor(self):
    # [FUNCTION DOC]
    # Multiplies two boolean vectors and expects the elementwise logical AND of them.
    x = constant_op.constant([True, False, True])
    y = constant_op.constant([False, True, True])
    expected = constant_op.constant([False, False, True])
    self._test_mul_dispatch_factory(x, y, expected)

  def testBooleanMix(self):
    # [FUNCTION DOC]
    # Multiplies an integer vector with a boolean vector in both operand orders and expects the
    # boolean vector back (boolean operand treated as a mask/logical result) in each case.
    # Non-boolean tensor is first.
    x = constant_op.constant([1, 2, 3])
    y = constant_op.constant([False, True, True])
    expected = constant_op.constant([False, True, True])
    self._test_mul_dispatch_factory(x, y, expected)

    # Boolean tensor is first.
    x = constant_op.constant([False, True, True])
    y = constant_op.constant([1, 2, 3])
    expected = constant_op.constant([False, True, True])
    self._test_mul_dispatch_factory(x, y, expected)


if __name__ == "__main__":
  test.main()
