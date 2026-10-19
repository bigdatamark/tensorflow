# [MODULE DOC]
# Test module for tensor-list (variant list) ops compiled through the XLA bridge: element shape
# queries, push/pop, get/set, reserve, stack, gather, zeros-like and split error handling, plus
# behavior without a maximum element count. Contains one test class. When run as a script it
# prepends a flag to the XLA flags environment variable setting the minimum cluster size to two,
# then starts the test runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ops which manipulate lists of tensors via bridge."""

# pylint: disable=g-bad-name
import os

from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class ListOpsTest(parameterized.TestCase, xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for tensor-list ops on the XLA test device. Inherits parameterized test support and
  # XLA fixtures (session, test scope). Holds no state; each test builds a list inside the test
  # scope and checks values, shapes or raised errors.

  def testElementShape(self):
    # [FUNCTION DOC]
    # Test: a list created with an element shape whose first dimension is a fed placeholder and
    # second is 15 reports that element shape as 32-bit and 64-bit integer tensors. Asserts the
    # result is (10, 15) when the placeholder is 10 and (7, 15) when it is 7.
    with self.session() as sess, self.test_scope():
      dim = array_ops.placeholder(dtypes.int32)
      l = list_ops.empty_tensor_list(
          element_shape=(dim, 15),
          element_dtype=dtypes.float32,
          max_num_elements=20)
      e32 = list_ops.tensor_list_element_shape(l, shape_type=dtypes.int32)
      e64 = list_ops.tensor_list_element_shape(l, shape_type=dtypes.int64)
      self.assertAllEqual(sess.run(e32, {dim: 10}), (10, 15))
      self.assertAllEqual(sess.run(e64, {dim: 7}), (7, 15))

  def testPushPop(self):
    # [FUNCTION DOC]
    # Test: push two 7 by 15 tensors (filled with 1 and 2) onto an empty list of capacity 10, pop
    # twice, and assert pops return the values in last-in-first-out order (2 then 1).
    with self.session() as sess, self.test_scope():
      l = list_ops.empty_tensor_list(
          element_shape=(7, 15),
          element_dtype=dtypes.float32,
          max_num_elements=10)
      l = list_ops.tensor_list_push_back(
          l, constant_op.constant(1.0, shape=(7, 15)))
      l = list_ops.tensor_list_push_back(
          l, constant_op.constant(2.0, shape=(7, 15)))
      l, e2 = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      _, e1 = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      self.assertAllEqual(sess.run(e2), 2.0 * np.ones((7, 15)))
      self.assertAllEqual(sess.run(e1), 1.0 * np.ones((7, 15)))

  def testDoNotConstantFoldVariants(self):
    # [FUNCTION DOC]
    # Test: pushing a tensor that depends on a placeholder onto a list must not be constant-folded
    # into a variant-typed constant (XLA must supply a consider-folding callback that prevents it).
    # Pushes a fed-value tensor and a constant 2, pops both and asserts 2 then the fed value 1.
    with self.session() as sess, self.test_scope():
      val = array_ops.placeholder(dtype=dtypes.float32)
      l = list_ops.empty_tensor_list(
          element_shape=(7, 15),
          element_dtype=dtypes.float32,
          max_num_elements=10)
      # Note: Pushing a Placeholder will force the constant folding code
      # to build a Const node with a DT_VARIANT output. This tests that XLA
      # passes a cf_consider_fn which prevent folding such nodes.
      l = list_ops.tensor_list_push_back(
          l, array_ops.fill(value=val, dims=(7, 15)))
      l = list_ops.tensor_list_push_back(
          l, constant_op.constant(2.0, shape=(7, 15)))
      l, e2 = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      _, e1 = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      self.assertAllEqual(sess.run(e2, {val: 1.0}), 2.0 * np.ones((7, 15)))
      self.assertAllEqual(sess.run(e1, {val: 1.0}), 1.0 * np.ones((7, 15)))

  def testPushPopSeparateLists(self):
    # [FUNCTION DOC]
    # Test: lists are values, so extending one list in two different ways yields independent
    # lists. Pushes 1, then derives two lists by pushing 2 and 3. Pops from each and asserts the
    # base list yields 1, the second yields 2 then 1, and the third yields 3 then 1.
    with self.session() as sess, self.test_scope():
      l = list_ops.empty_tensor_list(
          element_shape=[],
          element_dtype=dtypes.float32,
          max_num_elements=20)
      l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
      l2 = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
      l3 = list_ops.tensor_list_push_back(l, constant_op.constant(3.0))
      _, e11 = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      l2, e21 = list_ops.tensor_list_pop_back(l2, element_dtype=dtypes.float32)
      l2, e22 = list_ops.tensor_list_pop_back(l2, element_dtype=dtypes.float32)
      l3, e31 = list_ops.tensor_list_pop_back(l3, element_dtype=dtypes.float32)
      l3, e32 = list_ops.tensor_list_pop_back(l3, element_dtype=dtypes.float32)
      result = sess.run([e11, [e21, e22], [e31, e32]])
      self.assertEqual(result, [1.0, [2.0, 1.0], [3.0, 1.0]])

  def testEmptyTensorListNoMax(self):
    # [FUNCTION DOC]
    # Test: an empty list created without a maximum element count cannot be used under XLA;
    # running a push followed by a pop must raise an invalid-argument error telling the user to set
    # the maximum number of elements.
    with self.session() as sess, self.test_scope():
      l = list_ops.empty_tensor_list(
          element_shape=(7, 15), element_dtype=dtypes.float32)
      l = list_ops.tensor_list_push_back(
          l, constant_op.constant(1.0, shape=(7, 15)))
      _, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Set the max number of elements"):
        self.assertAllEqual(sess.run(e), 1.0 * np.ones((7, 15)))

  def testEmptyTensorListMax(self):
    # [FUNCTION DOC]
    # Test: an empty list with maximum two elements of shape 10 by 15 accepts a push and a pop;
    # the popped tensor equals the pushed one (filled with 3).
    with self.session() as sess, self.test_scope():
      l = list_ops.empty_tensor_list(
          element_shape=(10, 15), element_dtype=dtypes.float32,
          max_num_elements=2)
      l = list_ops.tensor_list_push_back(
          l, array_ops.fill(value=3.0, dims=(10, 15)))
      _, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      self.assertAllEqual(sess.run(e), 3.0 * np.ones((10, 15)))

  def testListFromTensor(self):
    # [FUNCTION DOC]
    # Test: build a list from a two-element vector, read item 0 (1.0), pop twice (2.0 then 1.0) and
    # check the length of the original list is 2.
    with self.session(), self.test_scope():
      t = constant_op.constant([1.0, 2.0])
      l = list_ops.tensor_list_from_tensor(t, element_shape=[])
      e = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      self.assertAllEqual(e, 1.0)
      l, e0 = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      self.assertAllEqual(e0, 2.0)
      l, e1 = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      self.assertAllEqual(e1, 1.0)
      self.assertAllEqual(list_ops.tensor_list_length(l), 2)

  def testGetSet(self):
    # [FUNCTION DOC]
    # Test: from a list of 1.0 and 2.0, item 0 reads 1.0; after setting item 0 to 3.0, stacking the
    # list gives 3.0 and 2.0.
    with self.session(), self.test_scope():
      t = constant_op.constant([1.0, 2.0])
      l = list_ops.tensor_list_from_tensor(t, element_shape=[])
      e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      self.assertAllEqual(e0, 1.0)
      l = list_ops.tensor_list_set_item(l, 0, 3.0)
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.assertAllEqual(t, [3.0, 2.0])

  def testSetDoesNotUpdatePushIndex(self):
    # [FUNCTION DOC]
    # Test: setting an item in an empty list of capacity two does not move the push position.
    # Sets index 1 to 3, then pushes 5 and 7; the stacked result is 5 and 7.
    with self.session(), self.test_scope():
      l = list_ops.empty_tensor_list(
          element_shape=[], element_dtype=dtypes.float32, max_num_elements=2)
      # SetItem should not change the push index.
      l = list_ops.tensor_list_set_item(l, 1, 3.)
      l = list_ops.tensor_list_push_back(l, 5.)
      l = list_ops.tensor_list_push_back(l, 7.)
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.assertAllEqual(t, [5., 7.])

  def testGetSetReserved(self):
    # [FUNCTION DOC]
    # Test: a reserved list of two scalars reads zero for unset item 0; after setting item 0 to
    # 3.0, stacking gives 3.0 and 0.0.
    with self.session(), self.test_scope():
      l = list_ops.tensor_list_reserve(
          element_dtype=dtypes.float32, element_shape=[], num_elements=2)
      e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      self.assertAllEqual(e0, 0.0)
      l = list_ops.tensor_list_set_item(l, 0, 3.0)
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.assertAllEqual(t, [3.0, 0.0])

  def testSetStackReservedUnknownElementShape(self):
    # [FUNCTION DOC]
    # Test: a reserved list of two items with unknown element shape; setting item 0 to a two-element
    # vector and stacking gives that vector followed by a zero vector of the same length.
    with self.session(), self.test_scope():
      l = list_ops.tensor_list_reserve(
          element_dtype=dtypes.float32, element_shape=None, num_elements=2)
      l = list_ops.tensor_list_set_item(l, 0, [3.0, 4.0])
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.assertAllEqual(t, [[3.0, 4.0], [0., 0.]])

  def testPushInEmptyListWithUnknownElementShape(self):
    # [FUNCTION DOC]
    # Test: in a list with unknown element shape, the first push fixes the element shape (a
    # two-element vector); pushing a scalar afterwards and evaluating the stack must raise an
    # internal error whose message mentions shape.
    with self.session(), self.test_scope():
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32, element_shape=None, max_num_elements=2)
      l = list_ops.tensor_list_push_back(l, [3.0, 4.0])
      # Pushing an element with a different shape should raise an error.
      with self.assertRaisesRegex(errors.InternalError, "shape"):
        l = list_ops.tensor_list_push_back(l, 5.)
        self.evaluate(
            list_ops.tensor_list_stack(l, element_dtype=dtypes.float32))

  def testGetSetReservedNonScalar(self):
    # [FUNCTION DOC]
    # Test: a reserved list of two 7 by 15 elements; set item 0 to ones, then item 0 reads all ones
    # and the unset item 1 reads all zeros.
    with self.session() as sess, self.test_scope():
      l = list_ops.tensor_list_reserve(
          element_dtype=dtypes.float32,
          element_shape=(7, 15),
          num_elements=2)
      l = list_ops.tensor_list_set_item(
          l, 0, constant_op.constant(1.0, shape=(7, 15)))
      e1 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      e2 = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
      self.assertAllEqual(sess.run(e1), np.ones((7, 15)))
      self.assertAllEqual(sess.run(e2), np.zeros((7, 15)))

  def testStack(self):
    # [FUNCTION DOC]
    # Test: push 1.0 and 2.0 into a list of capacity two; item 0 reads 1.0, and stacking returns a
    # vector with unknown static length containing 1.0 and 2.0.
    with self.session(), self.test_scope():
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=[],
          max_num_elements=2)
      l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
      e = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      self.assertAllEqual(e, 1.0)
      l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.assertAllEqual(t.shape.as_list(), [None])
      self.assertAllEqual(t, [1.0, 2.0])

  @parameterized.named_parameters(
      ("FlatList", [1.0, 2.0, 3.0], [], [0, 2], [1.0, 3.0]),
      ("NestedList", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]
                     ], [2], [1], [[3.0, 4.0]]),
      ("EmptyIndices", [1.0, 2.0, 3.0], [], [], []),
  )
  def testGather(self, input_list, element_shape, indices, output):
    # [FUNCTION DOC]
    # Parameterized test of gathering. Inputs: input_list (values to build a list from),
    # element_shape (shape of each element), indices (positions to gather) and output (expected
    # result). Cases: flat list with indices 0 and 2; nested list with index 1; and empty indices
    # giving an empty result. Asserts the gathered tensor equals the expected output.
    with self.session(), self.test_scope():
      tensor_list = list_ops.tensor_list_from_tensor(
          input_list, element_shape=element_shape)
      gather_t = list_ops.tensor_list_gather(
          tensor_list, indices, element_dtype=dtypes.float32)
      self.assertAllEqual(gather_t, output)

  def testStackWithUninitializedTensors(self):
    # [FUNCTION DOC]
    # Test: stacking a reserved list of three scalars that were never set yields three zeros.
    with self.session(), self.test_scope():
      l = list_ops.tensor_list_reserve(
          element_dtype=dtypes.float32, element_shape=[], num_elements=3)
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.assertAllEqual(t, [0., 0., 0.])

  def testZerosLikeForTensorList(self):
    # [FUNCTION DOC]
    # Test: the zeros-like operation applied to a list holding one pushed element (list capacity
    # two) gives a list that stacks to a vector of unknown static length containing two zeros.
    with self.session(), self.test_scope():
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=[],
          max_num_elements=2)
      l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
      z = array_ops.zeros_like(l)
      z = list_ops.tensor_list_stack(z, element_dtype=dtypes.float32)
      self.assertAllEqual(z.shape.as_list(), [None])
      self.assertAllEqual(z, [0.0, 0.0])

  def testInvalidSplitLength(self):
    # [FUNCTION DOC]
    # Test: splitting a tensor into a list with a length of zero is not supported by XLA; evaluating
    # it must raise an unimplemented error saying all lengths must be positive.
    with self.session(), self.test_scope():
      tensor_list_split = list_ops.tensor_list_split(
          tensor=[1], element_shape=[-1], lengths=[0]
      )
      with self.assertRaisesRegex(
          errors.UnimplementedError, "All lengths must be positive"
      ):
        self.evaluate(tensor_list_split)


if __name__ == "__main__":
  os.environ["TF_XLA_FLAGS"] = ("--tf_xla_min_cluster_size=2 " +
                                os.environ.get("TF_XLA_FLAGS", ""))
  test.main()
