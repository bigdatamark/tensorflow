# [MODULE DOC]
# Unit tests for the list operators in the data_structures operators module: creating lists
# (Python, tensor list and TensorArray variants), appending, popping and stacking, including
# illegal-input error cases and the Python fallback for stacking. Contains one test class and
# no import-time side effects; running as a script starts the test runner.
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
"""Tests for data_structures module."""

from tensorflow.python.autograph.operators import data_structures
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class ListTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the list operators of the data_structures module. No shared fixtures;
  # each test builds its own lists and uses cached sessions to evaluate tensors.

  def test_new_list_empty(self):
    # [FUNCTION DOC]
    # Creates a list with no elements and asserts the result is a tensor (an empty tensor list).
    l = data_structures.new_list()
    # Can't evaluate an empty list.
    # TODO(mdan): sess.run should allow tf.variant maybe?
    self.assertTrue(isinstance(l, tensor.Tensor))

  def test_new_list_tensor(self):
    # [FUNCTION DOC]
    # Creates a list from three integers and asserts the value equals those three integers.
    l = data_structures.new_list([3, 4, 5])
    self.assertAllEqual(l, [3, 4, 5])

  def test_tf_tensor_list_new(self):
    # [FUNCTION DOC]
    # Builds a tensor list from three integers, stacks it, and asserts the evaluated result is
    # those three integers.
    l = data_structures.tf_tensor_list_new([3, 4, 5])
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [3, 4, 5])

  def test_tf_tensor_list_new_empty(self):
    # [FUNCTION DOC]
    # Builds an empty tensor list with explicit integer dtype and scalar element shape, stacks it,
    # and asserts the evaluated result is empty.
    l = data_structures.tf_tensor_list_new([],
                                           element_dtype=dtypes.int32,
                                           element_shape=())
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [])

  def test_tf_tensor_list_new_from_tensor(self):
    # [FUNCTION DOC]
    # Builds a tensor list from a constant tensor, stacks it and asserts the original values
    # come back.
    l = data_structures.tf_tensor_list_new(constant_op.constant([3, 4, 5]))
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [3, 4, 5])

  @test_util.run_deprecated_v1
  def test_tf_tensor_list_new_illegal_input(self):
    # [FUNCTION DOC]
    # Runs in graph (v1) mode. Asserts value errors for: mixed int/float elements, an integer
    # list with a conflicting float dtype, and a given element shape with list elements or with a
    # tensor source. Asserts heterogeneous nested lists are accepted (non-null result).
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_list_new([3, 4.0])
    # TODO(mdan): It might make more sense to type cast in this case.
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_list_new([3, 4], element_dtype=dtypes.float32)
    # Tensor lists do support heterogeneous lists.
    self.assertIsNot(data_structures.tf_tensor_list_new([3, [4, 5]]), None)
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_list_new([3, 4], element_shape=(2,))
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_list_new(
          constant_op.constant([1, 2, 3]), element_shape=[1])

  def test_tf_tensor_array_new(self):
    # [FUNCTION DOC]
    # Builds a TensorArray from three integers, stacks it and asserts the evaluated values.
    l = data_structures.tf_tensor_array_new([3, 4, 5])
    t = l.stack()
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [3, 4, 5])

  def test_tf_tensor_array_new_illegal_input(self):
    # [FUNCTION DOC]
    # Asserts value errors for TensorArray creation with: mixed dtypes, conflicting dtype,
    # ragged element shapes, conflicting shape, and an empty list without dtype. Asserts an empty
    # list with an explicit dtype is accepted (shape can be inferred).
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_array_new([3, 4.0])
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_array_new([3, 4], element_dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_array_new([3, [4, 5]])
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_array_new([3, 4], element_shape=(2,))
    with self.assertRaises(ValueError):
      data_structures.tf_tensor_array_new([], element_shape=(2,))
    # TAs can infer the shape.
    self.assertIsNot(
        data_structures.tf_tensor_array_new([], element_dtype=dtypes.float32),
        None)

  def test_append_tensor_list(self):
    # [FUNCTION DOC]
    # Appends a three-element tensor to a new empty list, stacks it and asserts the result is a
    # single row holding those values.
    l = data_structures.new_list()
    x = constant_op.constant([1, 2, 3])
    l = data_structures.list_append(l, x)

    t = list_ops.tensor_list_stack(l, element_dtype=x.dtype)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [[1, 2, 3]])

  @test_util.run_deprecated_v1
  def test_append_tensorarray(self):
    # [FUNCTION DOC]
    # Runs in graph (v1) mode. Appends 1 then 2 to a dynamic empty TensorArray and asserts the
    # first returned array stacks to one item and the second to two items.
    l = tensor_array_ops.TensorArray(dtypes.int32, size=0, dynamic_size=True)
    l1 = data_structures.list_append(l, 1)
    l2 = data_structures.list_append(l1, 2)

    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(l1.stack()), [1])
      self.assertAllEqual(self.evaluate(l2.stack()), [1, 2])

  def test_append_python(self):
    # [FUNCTION DOC]
    # Appends 1 and then 2 to a plain Python list and asserts the results are one and then two
    # items (the list is mutated in place).
    l = []
    self.assertAllEqual(data_structures.list_append(l, 1), [1])
    self.assertAllEqual(data_structures.list_append(l, 2), [1, 2])

  def test_pop_tensor_list(self):
    # [FUNCTION DOC]
    # Creates a tensor list from a 2x2 tensor with pop options (dtype and element shape).
    # Asserts popping with an explicit index raises a not-implemented error. Then pops from the
    # end and asserts the popped element is the last row and the remaining list stacks to the
    # first row.
    initial_list = constant_op.constant([[1, 2], [3, 4]])
    elem_shape = constant_op.constant([2])
    l = list_ops.tensor_list_from_tensor(initial_list, element_shape=elem_shape)

    opts = data_structures.ListPopOpts(
        element_dtype=initial_list.dtype,
        element_shape=(2,))

    with self.assertRaises(NotImplementedError):
      data_structures.list_pop(l, 0, opts)

    with self.cached_session() as sess:
      l, x = data_structures.list_pop(l, None, opts)
      self.assertAllEqual(self.evaluate(x), [3, 4])

      t = list_ops.tensor_list_stack(l, element_dtype=initial_list.dtype)
      self.assertAllEqual(self.evaluate(t), [[1, 2]])

  def test_pop_python(self):
    # [FUNCTION DOC]
    # Pops twice from a plain Python list with no index and asserts each result pair (remaining
    # list, removed element) matches ([1,2],3) then ([1],2).
    l = [1, 2, 3]
    opts = data_structures.ListPopOpts(element_dtype=None, element_shape=())
    self.assertAllEqual(data_structures.list_pop(l, None, opts), ([1, 2], 3))
    self.assertAllEqual(data_structures.list_pop(l, None, opts), ([1], 2))

  def test_stack_tensor_list(self):
    # [FUNCTION DOC]
    # Stacks a tensor list built from a 2x2 tensor with stack options and asserts the result
    # equals the original tensor.
    initial_list = constant_op.constant([[1, 2], [3, 4]])
    elem_shape = constant_op.constant([2])
    l = list_ops.tensor_list_from_tensor(initial_list, element_shape=elem_shape)

    opts = data_structures.ListStackOpts(
        element_dtype=initial_list.dtype, original_call=None)

    with self.cached_session() as sess:
      t = data_structures.list_stack(l, opts)
      self.assertAllEqual(self.evaluate(t), self.evaluate(initial_list))

  @test_util.run_deprecated_v1
  def test_stack_tensor_list_empty(self):
    # [FUNCTION DOC]
    # Runs in graph (v1) mode. Asserts stacking an empty variant-typed tensor list raises a
    # value error (a TODO notes empty lists could be allowed when dtype and shape are known).
    l = list_ops.empty_tensor_list(
        element_shape=None, element_dtype=dtypes.variant)

    opts = data_structures.ListStackOpts(
        element_dtype=dtypes.int32, original_call=None)

    # TODO(mdan): Allow stacking empty lists if the dtype and shape are known.
    with self.assertRaises(ValueError):
      data_structures.list_stack(l, opts)

  def test_stack_fallback(self):
    # [FUNCTION DOC]
    # Stacks a plain Python list using options whose original call doubles every element.
    # Asserts the result is the doubled list, proving the fallback callable was used.

    def dummy_function(l):
      # [FUNCTION DOC]
      # HELPER: fake original call; returns a new list with each element doubled.
      # Lazy person's mock: just transform the argument in a way in which we
      # can check that this function was indeed called.
      return [x * 2 for x in l]

    opts = data_structures.ListStackOpts(
        element_dtype=None, original_call=dummy_function)

    self.assertAllEqual(data_structures.list_stack([1, 2], opts), [2, 4])


if __name__ == '__main__':
  test.main()
