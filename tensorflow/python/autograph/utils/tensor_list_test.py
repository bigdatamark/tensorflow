# [MODULE DOC]
# Unit tests for the tensor_list utility module (dynamic_list_append and the TensorList wrapper),
# covering eager mode and deprecated graph mode. One test case class. No import-time side effects;
# runs the TF test main when executed directly.
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
"""Tests for Autograph lists."""

from tensorflow.python.autograph.utils import tensor_list as tl
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class TensorListTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the tensor_list module with one shared helper for building shape constants.

  def _shape(self, shape_tuple):
    # [FUNCTION DOC]
    # HELPER: converts a tuple of dimensions into an int32 constant tensor used as an element shape.
    return constant(shape_tuple, dtypes.int32)

  def test_dynamic_list_append(self):
    # [FUNCTION DOC]
    # Appends the value 1 through the dynamic append function to a Python list, an empty tensor
    # list, a dynamic-size tensor array and the TensorList wrapper, asserting each ends up holding
    # exactly that element (list equality, stacked tensor values, or first-element lookup).
    l = []
    l = tl.dynamic_list_append(l, 1)
    self.assertListEqual(l, [1])

    l = list_ops.empty_tensor_list(self._shape(()), dtypes.int32)
    l = tl.dynamic_list_append(l, 1)
    s = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    self.assertAllEqual(s, [1])

    l = tensor_array_ops.TensorArray(dtypes.int32, size=0, dynamic_size=True)
    l = tl.dynamic_list_append(l, 1)
    s = l.stack()
    self.assertAllEqual(s, [1])

    l = tl.TensorList(self._shape(()), dtypes.int32)
    l = tl.dynamic_list_append(l, 1)
    self.assertAllEqual(l[0], 1)

  def test_list_append_python(self):
    # [FUNCTION DOC]
    # In eager mode, appends a constant twice to a TensorList, pops twice, checking the count
    # after each step (1, 2, 1, 0) and that the popped value equals the original.
    with context.eager_mode():
      a = constant(3.0)
      l = tl.TensorList(a.shape, a.dtype)
      l.append(a)
      self.assertEqual(l.count().numpy(), 1)
      l.append(a)
      self.assertEqual(l.count().numpy(), 2)
      _ = l.pop()
      self.assertEqual(l.count().numpy(), 1)
      a2 = l.pop()
      self.assertEqual(l.count().numpy(), 0)
      self.assertEqual(a.numpy(), a2.numpy())

  def test_list_index_python(self):
    # [FUNCTION DOC]
    # In eager mode, appends a constant to a TensorList, asserts reading index 0 gives it, assigns a
    # second constant at index 0 and asserts reading it back gives the new value.
    with context.eager_mode():
      a = constant(3.0)
      b = constant(2.0)
      l = tl.TensorList(a.shape, a.dtype)
      l.append(a)
      self.assertEqual(l[0].numpy(), a.numpy())
      l[0] = ops.convert_to_tensor(b)
      self.assertEqual(l[0].numpy(), b.numpy())

  @test_util.run_deprecated_v1
  def test_list_append_tf(self):
    # [FUNCTION DOC]
    # Graph-mode version of the append/pop test (deprecated v1 mode): builds count and pop
    # operations, evaluates them together and asserts the counts are 1, 2, 1, 0 and that the
    # popped value equals the original.
    a = constant(3.0)
    l = tl.TensorList(a.shape, a.dtype)
    l.append(a)
    c1 = l.count()
    l.append(a)
    c2 = l.count()
    _ = l.pop()
    c3 = l.count()
    a2 = l.pop()
    c4 = l.count()
    c1, c2, c3, c4, a, a2 = self.evaluate([c1, c2, c3, c4, a, a2])
    self.assertEqual(c1, 1)
    self.assertEqual(c2, 2)
    self.assertEqual(c3, 1)
    self.assertEqual(c4, 0)
    self.assertEqual(a, a2)

  def test_list_index_tf(self):
    # [FUNCTION DOC]
    # Appends a constant, reads index 0, overwrites it with a second constant and reads again;
    # evaluates everything and asserts the first read equals the original and the second equals
    # the replacement.
    a = constant(3.0)
    b = constant(2.0)
    l = tl.TensorList(a.shape, a.dtype)
    l.append(a)
    l0 = l[0]
    l[0] = b
    l1 = l[0]
    l0, l1, a, b = self.evaluate([l0, l1, a, b])
    self.assertEqual(l0, a)
    self.assertEqual(l1, b)


if __name__ == '__main__':
  test.main()
