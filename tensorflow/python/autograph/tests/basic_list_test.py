# [MODULE DOC]
# Reference test for autograph conversion of Python lists into tensor lists. Sample functions
# build, read, update, fill and pop lists (annotating element types with directives, then
# stacking into tensors); the test compares native versus compiled results and checks the
# expected errors for unannotated or empty lists. No import-time side effects.
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
"""Basic list operations."""

import tensorflow as tf

from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.tests import reference_test_base


def type_not_annotated(n):
  # [FUNCTION DOC]
  # Function under test: appends the integers below n to a list with no element type annotation
  # and stacks it. Converted, this fails because stacking needs a known element type.
  l = []
  # TODO(mdan): Here, we ought to infer the dtype and shape when i is staged.
  for i in range(n):
    l.append(i)
  return special_functions.stack(l, strict=False)


def element_access():
  # [FUNCTION DOC]
  # Function under test: builds the list one, two, three, annotates its element type as 32-bit
  # integer, and returns twice the element at index one.
  l = []
  l.append(1)
  l.append(2)
  l.append(3)
  directives.set_element_type(l, tf.int32)
  return 2 * l[1]


def element_update():
  # [FUNCTION DOC]
  # Function under test: builds the list one, two, three with an integer element type, sets the
  # element at index one to five, and returns the stacked list.
  l = []
  l.append(1)
  l.append(2)
  l.append(3)
  directives.set_element_type(l, tf.int32)
  l[1] = 5
  return special_functions.stack(l, strict=False)


def simple_fill(n):
  # [FUNCTION DOC]
  # Function under test: fills an integer-typed list with 0 to n-1 in a loop and returns it
  # stacked. For n equal to zero the stacking of an empty list fails.
  l = []
  directives.set_element_type(l, tf.int32)
  for i in range(n):
    l.append(i)
  return special_functions.stack(l, strict=False)


def nested_fill(m, n):
  # [FUNCTION DOC]
  # Function under test: builds a list of m rows, each a list of 0 to n-1, with integer element
  # types, and returns the stacked matrix. Stacking fails when m or n is zero.
  mat = []
  directives.set_element_type(mat, tf.int32)
  for _ in range(m):
    l = []
    directives.set_element_type(l, tf.int32)
    for j in range(n):
      l.append(j)
    mat.append(special_functions.stack(l, strict=False))
  return special_functions.stack(mat, strict=False)


def read_write_loop(n):
  # [FUNCTION DOC]
  # Function under test: starts with a list of two ones and, for each index from two to n-1,
  # appends the sum of the previous two elements and negates the element two positions back.
  # Returns the stacked list.
  l = []
  l.append(1)
  l.append(1)
  directives.set_element_type(l, tf.int32)
  for i in range(2, n):
    l.append(l[i-1] + l[i-2])
    l[i-2] = -l[i-2]
  return special_functions.stack(l, strict=False)


def simple_empty(n):
  # [FUNCTION DOC]
  # Function under test: builds the list one to four, then pops n elements from the end summing
  # them. Returns a pair: the stacked remaining list and the sum. The element type annotation
  # includes an empty element shape.
  l = []
  l.append(1)
  l.append(2)
  l.append(3)
  l.append(4)
  directives.set_element_type(l, tf.int32, ())
  s = 0
  for _ in range(n):
    s += l.pop()
  return special_functions.stack(l, strict=False), s


def mutation(t, n):
  # [FUNCTION DOC]
  # Function under test: sets t at each index i below n to i, in place, and returns t.
  for i in range(n):
    t[i] = i
  return t


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Reference test case with only the list-conversion feature enabled.

  def setUp(self):
    # [FUNCTION DOC]
    # Runs base setup, then sets autograph options to enable only list conversion.
    super(ReferenceTest, self).setUp()
    self.autograph_opts = tf.autograph.experimental.Feature.LISTS

  def test_tensor_mutation(self):
    # [FUNCTION DOC]
    # Converts the mutation function and checks it matches native execution for a list of ten
    # zeros and n equal to ten.
    self.assertConvertedMatchesNative(mutation, [0] * 10, 10)

  def test_basic(self):
    # [FUNCTION DOC]
    # With all inputs converted to tensors: checks element access, update, simple and nested
    # fill, read/write loop and pop samples match eager results; checks the unannotated list
    # raises a value error about unknown element type; and checks filling with zero elements
    # raises a value error for both the simple and the nested fill.
    self.all_inputs_tensors = True
    self.assertFunctionMatchesEager(element_access)
    self.assertFunctionMatchesEager(element_update)

    # TODO(mdan): This should raise a compilation, not runtime, error.
    with self.assertRaisesRegex(
        ValueError,
        'cannot stack a list without knowing its element type; '
        'use set_element_type to annotate it'):
      self.function(type_not_annotated)(3)

    self.assertFunctionMatchesEager(simple_fill, 5)
    self.assertFunctionMatchesEager(nested_fill, 5, 3)
    self.assertFunctionMatchesEager(read_write_loop, 4)
    self.assertFunctionMatchesEager(simple_empty, 0)
    self.assertFunctionMatchesEager(simple_empty, 2)
    self.assertFunctionMatchesEager(simple_empty, 4)

    # TODO(mdan): Allow explicitly setting the element shape to mitigate these.
    # TODO(mdan): This should raise a friendlier runtime error.
    # The error should spell out that empty lists cannot be stacked.
    # Alternatively, we can also insert conditionals that construct a zero-sized
    # Tensor of the appropriate type and shape, but we first want to make sure
    # that doesn't degrade performance.
    with self.assertRaises(ValueError):
      self.function(simple_fill)(0)
    with self.assertRaises(ValueError):
      self.function(nested_fill)(0, 3)


if __name__ == '__main__':
  tf.test.main()
