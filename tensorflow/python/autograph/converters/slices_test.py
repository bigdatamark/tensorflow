# [MODULE DOC]
# Unit tests for the AutoGraph slices converter. Converts small functions that index a list
# argument after declaring its element type, using the directives and slices converters, and
# checks the generated code works on tensor lists. Runs the TensorFlow test main as a script.
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
"""Tests for slices module."""

from tensorflow.python.autograph.converters import directives as directives_converter
from tensorflow.python.autograph.converters import slices
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.lang import directives
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class SliceTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the slices converter, using the converter testing base class transform and
  # evaluate helpers.

  def test_index_access(self):
    # [FUNCTION DOC]
    # Scenario: a function sets the element type of its list argument to 32-bit integers and
    # returns the item at index one. The test transforms it with the directives and slices
    # converters, passes a tensor list built from the values one and two, and asserts the
    # evaluated result is two.

    def f(l):
      # [FUNCTION DOC]
      # Test subject: declares the element type of the argument as int32 and returns item one.
      directives.set_element_type(l, dtypes.int32)
      return l[1]

    tr = self.transform(f, (directives_converter, slices))

    tl = list_ops.tensor_list_from_tensor(
        [1, 2], element_shape=constant_op.constant([], dtype=dtypes.int32))
    y = tr(tl)
    self.assertEqual(2, self.evaluate(y))

  def test_index_access_multiple_definitions(self):
    # [FUNCTION DOC]
    # Scenario: the indexed variable is redefined conditionally, so it has multiple definitions
    # after the element type directive. The test only checks that the transformation completes
    # without raising; the converted function is not called and nothing is asserted.

    def f(l):
      # [FUNCTION DOC]
      # Test subject: declares the element type, conditionally rebinds the list to an empty
      # list, and returns item one.
      directives.set_element_type(l, dtypes.int32)
      if l:
        l = []
      return l[1]

    self.transform(f, (directives_converter, slices))


if __name__ == '__main__':
  test.main()
