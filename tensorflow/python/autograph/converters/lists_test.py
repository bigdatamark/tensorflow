# [MODULE DOC]
# Unit tests for the autograph lists converter. Each test converts a small function using list
# literals or the append, pop and stack operations, runs it, and checks the result is a tensor
# list or has the expected values. Runs the TensorFlow test main when executed directly.
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
"""Tests for lists module."""

from tensorflow.python.autograph.converters import directives as directives_converter
from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class ListTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the lists converter; uses the converter testing base class to transform and run
  # functions and to evaluate tensors.

  def test_empty_list(self):
    # [FUNCTION DOC]
    # Scenario: a function returning an empty list literal. Expects a tensor result of the variant
    # data type (an empty tensor list).

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the lists
      # converter; builds or manipulates a list and returns it or a derived value.
      return []

    tr = self.transform(f, lists)

    tl = tr()
    # Empty tensor lists cannot be evaluated or stacked.
    self.assertIsInstance(tl, tensor.Tensor)
    self.assertEqual(tl.dtype, dtypes.variant)

  def test_initialized_list(self):
    # [FUNCTION DOC]
    # Scenario: a function returning a list literal of 1, 2, 3. Expects the values 1, 2, 3.

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the lists
      # converter; builds or manipulates a list and returns it or a derived value.
      return [1, 2, 3]

    tr = self.transform(f, lists)

    self.assertAllEqual(tr(), [1, 2, 3])

  def test_list_append(self):
    # [FUNCTION DOC]
    # Scenario: a tensor list initialized with 1 and appended with 2 and 3. Stacks the result and
    # expects the values 1, 2, 3.

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the lists
      # converter; builds or manipulates a list and returns it or a derived value.
      l = special_functions.tensor_list([1])
      l.append(2)
      l.append(3)
      return l

    tr = self.transform(f, lists)

    tl = tr()
    r = list_ops.tensor_list_stack(tl, dtypes.int32)
    self.assertAllEqual(self.evaluate(r), [1, 2, 3])

  def test_list_pop(self):
    # [FUNCTION DOC]
    # Scenario: pop on a tensor list of 1, 2, 3 with an element type directive (int32, scalar
    # shape), converted with the directives and lists converters. Expects popped value 3 and
    # remaining list 1, 2.

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the lists
      # converter; builds or manipulates a list and returns it or a derived value.
      l = special_functions.tensor_list([1, 2, 3])
      directives.set_element_type(l, dtype=dtypes.int32, shape=())
      s = l.pop()
      return s, l

    tr = self.transform(f, (directives_converter, lists))

    ts, tl = tr()
    r = list_ops.tensor_list_stack(tl, dtypes.int32)
    self.assertAllEqual(self.evaluate(r), [1, 2])
    self.assertAllEqual(self.evaluate(ts), 3)

  def test_double_list_pop(self):
    # [FUNCTION DOC]
    # Scenario: two chained pops on a plain Python list of lists as input. Verifies the pops are
    # sequenced correctly: for input 1, 2 and a nested list 1, 2, 3 the result is 3.

    def f(l):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the lists
      # converter; builds or manipulates a list and returns it or a derived value.
      s = l.pop().pop()
      return s

    tr = self.transform(f, lists)

    test_input = [1, 2, [1, 2, 3]]
    # TODO(mdan): Pass a list of lists of tensor when we fully support that.
    # For now, we just pass a regular Python list of lists just to verify that
    # the two pop calls are sequenced properly.
    self.assertAllEqual(tr(test_input), 3)

  def test_list_stack(self):
    # [FUNCTION DOC]
    # Scenario: stacking a list literal of 1, 2, 3 with the stack function. Expects values 1, 2, 3.

    def f():
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test and converted by the lists
      # converter; builds or manipulates a list and returns it or a derived value.
      l = [1, 2, 3]
      return array_ops_stack.stack(l)

    tr = self.transform(f, lists)

    self.assertAllEqual(self.evaluate(tr()), [1, 2, 3])


if __name__ == '__main__':
  test.main()
