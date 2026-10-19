# [MODULE DOC]
# Tests for the tensor-list ops exposed by the list_ops module and the generated list op wrappers
# (empty, reserve, push/pop, get/set, stack, gather, scatter, concat, split, resize, add_n,
# zeros_like, serialization across devices, and gradients through them).
# A single test class is defined; it is run in both graph and eager modes. Several tests are
# graph-only or GPU-only or need a local multi-process cluster.
# Importing the module has no side effects; running it as a script starts the TensorFlow test runner.
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
"""Tests for ops which manipulate lists of tensors."""

# pylint: disable=g-bad-name
from absl.testing import parameterized
import numpy as np  # pylint: disable=unused-import

from tensorflow.core.framework import full_type_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ListOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering tensor lists (variant-typed list handles holding tensors) in list_ops.
  # The whole class is decorated so every test runs in both graph and eager modes unless a test
  # overrides that. Tests exercise happy paths, error messages for invalid input (shape, rank,
  # index, dtype, size problems), uninitialized elements of reserved lists, gradients through list
  # ops, nested lists, resource variables holding lists, and device copies. Methods whose names
  # start with an underscore are shared helpers called by CPU and GPU variants. Inherits the
  # TensorFlow test case helpers and the parameterized test machinery; has no instance state.

  def _testPushPop(self, max_num_elements):
    # [FUNCTION DOC]
    # HELPER: shared body for the push/pop test and its GPU variant.
    # Creates an empty float list with scalar elements and the given maximum length (nothing means
    # unbounded), pushes one value, pops it, stacks the remaining list, and evaluates both.
    # Asserts that the remaining stacked list is empty and the popped value equals one.
    # Input: max_num_elements, an integer or nothing. Returns nothing.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[],
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    l = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    l, e = self.evaluate((l, e))
    self.assertAllEqual(l, [])
    self.assertAllEqual(e, 1.0)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  def testPushPop(self, max_num_elements):
    # [FUNCTION DOC]
    # Parameterized over no limit and a limit of two elements; runs the shared push/pop scenario on
    # the default device. Expects the list to be empty after one push and one pop and the popped
    # value to be the pushed one.
    self._testPushPop(max_num_elements)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  @test_util.run_gpu_only
  def testPushPopGPU(self, max_num_elements):
    # [FUNCTION DOC]
    # GPU-only variant of the push/pop test, parameterized over no limit and a limit of two.
    # Runs the shared push/pop scenario inside a GPU device scope; same expectations as the CPU test.
    with context.device("gpu:0"):
      self._testPushPop(max_num_elements)

  @test_util.run_deprecated_v1
  def testPushInFullListFails(self):
    # [FUNCTION DOC]
    # Graph-mode only. Builds an empty list limited to one element, pushes one value, then pushes a
    # second value. Expects an invalid-argument error saying an item was pushed into a full list
    # when the result is evaluated.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[], max_num_elements=1)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Tried to push item into a full list"):
      l = list_ops.tensor_list_push_back(l, 2.)
      self.evaluate(l)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  @test_util.run_deprecated_v1
  def testPopFromEmptyTensorListFails(self, max_num_elements):
    # [FUNCTION DOC]
    # Graph-mode only, parameterized over no limit and a limit of two. Pops from an empty list and
    # expects an invalid-argument error saying the code tried to pop from an empty list.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[],
        max_num_elements=max_num_elements)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Trying to pop from an empty list"):
      l = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      self.evaluate(l)

  def testTensorListReserveWithNonScalarNumElements(self):
    # [FUNCTION DOC]
    # Reserves a list whose number-of-elements argument is a two-element vector instead of a scalar.
    # Expects either an invalid-argument error (from the kernel) or an unknown error (from the MLIR
    # op definition) when evaluated; both are accepted.
    # list_kernels.cc in tf/core/kernels raises InvalidArgumentError, and
    # tf_ops_n_z.cc in tf/compiler/mlir/tf/ir raises UnknownError.
    with self.assertRaises((errors.InvalidArgumentError, errors.UnknownError)):
      l = list_ops.tensor_list_reserve(
          element_dtype=dtypes.float32,
          element_shape=[2, 3],
          num_elements=constant_op.constant([1, 1]))
      self.evaluate(l)

  def testPopUninitializedTensorUseListElementShape(self):
    # [FUNCTION DOC]
    # Reserves a list of three float elements with a fully defined element shape of two by three,
    # pops the last element and stacks the rest. Expects the popped uninitialized element to be a
    # zero tensor of the element shape and the remaining stack to be zeros of the matching shape.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[2, 3], num_elements=3)
    _, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    l = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    l, e = self.evaluate((l, e))
    self.assertAllEqual(e, np.zeros((2, 3)))
    self.assertAllEqual(l, np.zeros((3, 2, 3)))

  def testPopUninitializedTensorUseSpecifiedElementShape(self):
    # [FUNCTION DOC]
    # Reserves a list whose element shape is only partially defined, then pops using the raw op with
    # an explicit, fully defined element shape. Expects the popped uninitialized element to be zeros
    # with the explicitly specified shape.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[None, 3], num_elements=3)
    _, e = gen_list_ops.tensor_list_pop_back(
        l, element_dtype=dtypes.float32, element_shape=[4, 3])
    self.assertAllEqual(e, np.zeros((4, 3)))

  def testPopUninitializedTensorWithInvalidElementShapeFails(self):
    # [FUNCTION DOC]
    # Two failure scenarios when popping an uninitialized element. First, a list with unknown element
    # shape: expects an invalid-argument error that the shape is not fully defined. Second, a list
    # with a partially defined element shape and an incompatible explicit shape passed to the raw
    # pop op: expects an invalid-argument error about incompatible shapes during merge.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Trying to read an uninitialized tensor but "
        "element_shape is not fully defined"):
      _, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      self.evaluate(e)

    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[None, 2], num_elements=3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Incompatible shapes during merge: \[1,3\] vs. \[\?,2\]"):
      _, e = gen_list_ops.tensor_list_pop_back(
          l, element_dtype=dtypes.float32, element_shape=[1, 3])
      self.evaluate(e)

  def testPushGetGrad(self):
    # [FUNCTION DOC]
    # Under a gradient tape, pushes a scalar and a vector onto a list with unknown element shape and
    # reads the second element. Checks the read value, then checks the gradient with respect to the
    # two pushed values: the vector gets ones and the scalar gets zero. This verifies that the
    # gradient of a push converts invalid (unset) gradient entries to zeros.
    with backprop.GradientTape() as tape:
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32, element_shape=None)
      c0 = constant_op.constant(5.0)
      c1 = constant_op.constant([10.0, 20.0])
      tape.watch(c0)
      tape.watch(c1)
      l = list_ops.tensor_list_push_back(l, c0)
      l = list_ops.tensor_list_push_back(l, c1)
      t1 = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(t1), [10.0, 20.0])
      # t1 == c1 so the gradient should be [0., [1., 1.]]
      # This tests that the gradient of push_back correctly converts DT_INVALID
      # tensors to zeros. The list returned by the gradient of GetItem will
      # have only have tensor at index 1 set and others set to DT_INVALID.
      dt0, dt1 = tape.gradient(t1, [c0, c1])
      self.assertAllEqual(self.evaluate(dt1), [1.0, 1.0])
      self.assertEqual(self.evaluate(dt0), 0.0)

  def _testStack(self, max_num_elements):
    # [FUNCTION DOC]
    # HELPER: shared body for the stack test and its GPU variant.
    # Pushes two scalars onto an empty list with the given maximum size, stacks it, checks that in
    # graph mode the static shape has an unknown length, and checks the values are one and two.
    # Input: max_num_elements, integer or nothing. Returns nothing.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[],
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    if not context.executing_eagerly():
      self.assertAllEqual(t.shape.as_list(), [None])
    self.assertAllEqual(self.evaluate(t), [1.0, 2.0])

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  def testStack(self, max_num_elements):
    # [FUNCTION DOC]
    # Parameterized over no limit and a limit of two; runs the shared stack scenario on the default
    # device and expects a stacked result of the two pushed values.
    self._testStack(max_num_elements)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  @test_util.run_gpu_only
  def testStackGPU(self, max_num_elements):
    # [FUNCTION DOC]
    # GPU-only variant of the stack test, parameterized over no limit and a limit of two; runs the
    # shared stack scenario inside a GPU device scope.
    with context.device("gpu:0"):
      self._testStack(max_num_elements)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 3))
  @test_util.run_deprecated_v1
  def testStackWithUnknownElementShape(self, max_num_elements):
    # [FUNCTION DOC]
    # Graph-mode only, parameterized over no limit and a limit of three. Stacks a list with unknown
    # element shape holding two scalars and expects the values. Then pushes a vector and stacks again
    # and expects an invalid-argument error about incompatible ranks during merge.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=None,
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))

    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [1.0, 2.0])

    # Should raise an error when the element tensors do not all have the same
    # shape.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Incompatible ranks during merge: 0 vs. 1"):
      l = list_ops.tensor_list_push_back(l, constant_op.constant([3.0, 4.0]))
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 3))
  @test_util.run_deprecated_v1
  def testStackWithPartiallyDefinedElementShape(self, max_num_elements):
    # [FUNCTION DOC]
    # Graph-mode only, parameterized over no limit and a limit of three. Stacks a list with element
    # shape of one unknown dimension holding two single-element vectors and expects a two by one
    # result. Then pushes a vector of a different length and expects an invalid-argument error about
    # incompatible shapes during merge.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[None],
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant([1.0]))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([2.0]))

    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[1.0], [2.0]])

    # Should raise an error when the element tensors do not all have the same
    # shape.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Incompatible shapes during merge: \[1\] vs. \[2\]"):
      l = list_ops.tensor_list_push_back(l, constant_op.constant([2.0, 3.0]))
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  @test_util.run_deprecated_v1
  def testStackEmptyList(self, max_num_elements):
    # [FUNCTION DOC]
    # Graph-mode only, parameterized over no limit and a limit of two. Stacking an empty list whose
    # element shape is fully defined yields shape zero by one by two. Stacking an empty list with a
    # partially defined or unknown element shape must raise an invalid-argument error mentioning
    # non-fully-defined shapes.
    # Should be able to stack empty lists with fully defined element_shape.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[1, 2],
        max_num_elements=max_num_elements)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t).shape, (0, 1, 2))

    # Should not be able to stack empty lists with partially defined
    # element_shape.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "non-fully-defined"):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=[None, 2],
          max_num_elements=max_num_elements)
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.evaluate(t)

    # Should not be able to stack empty lists with undefined element_shape.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "non-fully-defined"):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=None,
          max_num_elements=max_num_elements)
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def _testStackWithUninitializedTensors(self):
    # [FUNCTION DOC]
    # HELPER: shared body for the uninitialized-stack test and its GPU variant.
    # Reserves a list of three scalar floats without setting any element and stacks it; asserts the
    # result is three zeros.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=3)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(t, [0., 0., 0.])

  def testStackWithUninitializedTensors(self):
    # [FUNCTION DOC]
    # Runs the shared scenario of stacking a reserved, never-written list on the default device;
    # expects zeros.
    self._testStackWithUninitializedTensors()

  @test_util.run_gpu_only
  def testStackWithUninitializedTensorsGpu(self):
    # [FUNCTION DOC]
    # GPU-only. Runs the shared scenario of stacking a reserved, never-written list inside a GPU
    # device scope; expects zeros.
    with context.device("gpu:0"):
      self._testStackWithUninitializedTensors()

  def _testStackWithUninitializedTensorsInferShape(self):
    # [FUNCTION DOC]
    # HELPER: shared body for the infer-shape stack test and its GPU variant.
    # Reserves a three-element list with unknown element shape, sets only the middle element to a
    # two-element vector, and stacks. Asserts the unset elements are filled with zeros of the shape
    # inferred from the set element.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    l = list_ops.tensor_list_set_item(l, 1, [1., 2.])
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(t, [[0., 0.], [1., 2.], [0., 0.]])

  def testStackWithUninitializedTensorsInferShape(self):
    # [FUNCTION DOC]
    # Runs the shared scenario where the element shape is inferred from the one initialized element
    # when stacking; expects zero vectors around the set vector, on the default device.
    self._testStackWithUninitializedTensorsInferShape()

  @test_util.run_gpu_only
  def testStackWithUninitializedTensorsInferShapeGpu(self):
    # [FUNCTION DOC]
    # GPU-only variant of the infer-shape stacking scenario, run inside a GPU device scope.
    with context.device("gpu:0"):
      self._testStackWithUninitializedTensorsInferShape()

  def testStackReservedListWithNoElementsAndPartialElementShapeFails(self):
    # [FUNCTION DOC]
    # Reserves a list with unknown element shape and no initialized elements, then stacks it.
    # Expects an invalid-argument error that a list of only uninitialized tensors with a
    # non-fully-defined element shape cannot be stacked.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "Tried to stack list which only contains "
        "uninitialized tensors and has a "
        "non-fully-defined element_shape: <unknown>"):
      t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testStackUsingSpecifiedElementShape(self):
    # [FUNCTION DOC]
    # Reserves a three-element list with unknown element shape and stacks it with the raw op using an
    # explicit scalar element shape. Expects a static shape of three in eager mode (unknown length in
    # graph mode) and a value of three zeros.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    t = gen_list_ops.tensor_list_stack(
        l, element_dtype=dtypes.float32, element_shape=[])
    if context.executing_eagerly():
      self.assertEqual(t.shape.as_list(), [3])
    else:
      self.assertEqual(t.shape.as_list(), [None])
    self.assertAllEqual(self.evaluate(t), np.zeros((3,)))

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 2))
  def testGatherGrad(self, max_num_elements):
    # [FUNCTION DOC]
    # Parameterized over no limit and a limit of two. Pushes two scalars, gathers them in reversed
    # order and checks the values. Builds the square of the sum of the gathered values and checks
    # the gradient with respect to the first pushed value equals six.
    with backprop.GradientTape() as tape:
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=[],
          max_num_elements=max_num_elements)
      c0 = constant_op.constant(1.0)
      tape.watch(c0)
      l = list_ops.tensor_list_push_back(l, c0)
      l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
      t = list_ops.tensor_list_gather(l, [1, 0], element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(t), [2.0, 1.0])
      s = (t[0] + t[1]) * (t[0] + t[1])
    dt = tape.gradient(s, c0)
    self.assertAllEqual(self.evaluate(dt), 6.0)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 3))
  @test_util.run_deprecated_v1
  def testGatherWithUnknownElementShape(self, max_num_elements):
    # [FUNCTION DOC]
    # Graph-mode only, parameterized over no limit and a limit of three. Builds a list with unknown
    # element shape holding two scalars and one vector. Gathering scalars or only the vector works.
    # Gathering a scalar together with the vector must raise an invalid-argument error about
    # incompatible ranks during merge.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=None,
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([3.0, 4.0]))

    t = list_ops.tensor_list_gather(l, [1, 0], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [2.0, 1.0])

    t = list_ops.tensor_list_gather(l, [2], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[3.0, 4.0]])

    # Should raise an error when the requested tensors do not all have the same
    # shape.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Incompatible ranks during merge: 0 vs. 1"):
      t = list_ops.tensor_list_gather(l, [0, 2], element_dtype=dtypes.float32)
      self.evaluate(t)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 3))
  @test_util.run_deprecated_v1
  def testGatherWithPartiallyDefinedElementShape(self, max_num_elements):
    # [FUNCTION DOC]
    # Graph-mode only, parameterized over no limit and a limit of three. List elements have a
    # one-dimensional unknown-length element shape. Gathering a single element or two equal-length
    # elements works. Gathering elements of different lengths must raise an invalid-argument error
    # about incompatible shapes during merge.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[None],
        max_num_elements=max_num_elements)
    l = list_ops.tensor_list_push_back(l, constant_op.constant([1.0]))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([2.0, 3.0]))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([4.0, 5.0]))

    t = list_ops.tensor_list_gather(l, [0], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[1.0]])

    t = list_ops.tensor_list_gather(l, [1, 2], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[2.0, 3.0], [4.0, 5.0]])

    # Should raise an error when the requested tensors do not all have the same
    # shape.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Incompatible shapes during merge: \[1\] vs. \[2\]"):
      t = list_ops.tensor_list_gather(l, [0, 2], element_dtype=dtypes.float32)
      self.evaluate(t)

  @parameterized.named_parameters(("NoMaxNumElements", None),
                                  ("WithMaxNumElements", 3))
  @test_util.run_deprecated_v1
  def testGatherEmptyList(self, max_num_elements):
    # [FUNCTION DOC]
    # Graph-mode only, parameterized over no limit and a limit of three. Gathering no indices from an
    # empty list with a fully defined element shape yields shape zero by one by two. Doing so with a
    # partially defined or unknown element shape must raise an invalid-argument error mentioning
    # non-fully-defined shapes.
    # Should be able to gather from empty lists with fully defined
    # element_shape.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=[1, 2],
        max_num_elements=max_num_elements)
    t = list_ops.tensor_list_gather(l, [], element_dtype=dtypes.float32)
    self.assertAllEqual((0, 1, 2), self.evaluate(t).shape)

    # Should not be able to gather from empty lists with partially defined
    # element_shape.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "non-fully-defined"):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=[None, 2],
          max_num_elements=max_num_elements)
      t = list_ops.tensor_list_gather(l, [], element_dtype=dtypes.float32)
      self.evaluate(t)

    # Should not be able to gather from empty lists with undefined
    # element_shape.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "non-fully-defined"):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=None,
          max_num_elements=max_num_elements)
      t = list_ops.tensor_list_gather(l, [], element_dtype=dtypes.float32)
      self.evaluate(t)

  def testGatherGradWithNonContiguousIndices(self):
    # [FUNCTION DOC]
    # Under a persistent gradient tape, builds a three-element list from a tensor, overwrites the
    # middle element with a watched scalar, and gathers only that index. Checks the gathered value,
    # the gradient of the squared value with respect to the scalar (ten), and that the gradient of
    # the gathered tensor with respect to the list is a list of length three.
    with backprop.GradientTape(persistent=True) as tape:
      t = constant_op.constant([1.0, 2.0, 3.0])
      l = list_ops.tensor_list_from_tensor(t, element_shape=[])
      c = constant_op.constant(5.0)
      tape.watch(c)
      l = list_ops.tensor_list_set_item(l, 1, c)
      t = list_ops.tensor_list_gather(l, [1], element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(t), [5.0])
      s = t[0] * t[0]
    dt = tape.gradient(s, c)
    self.assertAllEqual(self.evaluate(dt), 10.0)
    dl = tape.gradient(t, l)
    dl_length = list_ops.tensor_list_length(dl)
    self.assertAllEqual(self.evaluate(dl_length), 3)

  def _testGatherWithUninitializedTensors(self):
    # [FUNCTION DOC]
    # HELPER: shared body for the uninitialized-gather test and its GPU variant.
    # Reserves a three-element scalar list, gathers indices zero and two, and asserts two zeros.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=3)
    t = list_ops.tensor_list_gather(l, [0, 2], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [0., 0.])

  def testGatherWithUninitializedTensors(self):
    # [FUNCTION DOC]
    # Runs the shared scenario of gathering never-written elements on the default device; expects
    # zeros.
    self._testGatherWithUninitializedTensors()

  @test_util.run_gpu_only
  def testGatherWithUninitializedTensorsGpu(self):
    # [FUNCTION DOC]
    # GPU-only variant of the uninitialized-gather scenario, run inside a GPU device scope.
    with context.device("gpu:0"):
      self._testGatherWithUninitializedTensors()

  def _testGatherWithUninitializedTensorsInferShape(self):
    # [FUNCTION DOC]
    # HELPER: shared body for the infer-shape gather test and its GPU variant.
    # Reserves a list with unknown element shape, sets element one to a two-element vector, gathers
    # elements one and two, and asserts the second row is zeros of the inferred shape.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    l = list_ops.tensor_list_set_item(l, 1, [1., 2.])
    t = list_ops.tensor_list_gather(l, [1, 2], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[1., 2.], [0., 0.]])

  def testGatherWithUninitializedTensorsInferShape(self):
    # [FUNCTION DOC]
    # Runs the shared infer-shape gather scenario on the default device.
    self._testGatherWithUninitializedTensorsInferShape()

  @test_util.run_gpu_only
  def testGatherWithUninitializedTensorsInferShapeGpu(self):
    # [FUNCTION DOC]
    # GPU-only variant of the infer-shape gather scenario, run inside a GPU device scope.
    with context.device("gpu:0"):
      self._testGatherWithUninitializedTensorsInferShape()

  def testGatherReservedListWithNoElementsAndPartialElementShapeFails(self):
    # [FUNCTION DOC]
    # Gathers from a reserved list with unknown element shape and no initialized elements. Expects an
    # invalid-argument error saying uninitialized tensors cannot be gathered from a list with a
    # non-fully-defined element shape.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Tried to gather uninitialized tensors from a"
        " list with non-fully-defined element_shape"):
      t = list_ops.tensor_list_gather(l, [0], element_dtype=dtypes.float32)
      self.evaluate(t)

  def testGatherUsingSpecifiedElementShape(self):
    # [FUNCTION DOC]
    # Gathers all three elements of a reserved list with unknown element shape using the raw op with
    # an explicit scalar element shape. Expects static shape of three and zero values.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    t = gen_list_ops.tensor_list_gather(
        l, [0, 1, 2], element_dtype=dtypes.float32, element_shape=[])
    self.assertEqual(t.shape.as_list(), [3])
    self.assertAllEqual(self.evaluate(t), np.zeros((3,)))

  def testGatherWithInvalidIndicesFails(self):
    # [FUNCTION DOC]
    # Gathers from a reserved three-element list with a negative index and then with an index equal
    # to the list length. Each must raise an invalid-argument error stating which element could not
    # be gathered from a list of three elements.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3
    )

    # Should raise an error when the input index is negative.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Trying to gather element -1 in a list with 3 elements.",
    ):
      t = list_ops.tensor_list_gather(l, [-1], element_dtype=dtypes.float32)
      self.evaluate(t)

    # Should raise an error when the input index is larger than the number of
    # elements in the list.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Trying to gather element 3 in a list with 3 elements.",
    ):
      t = list_ops.tensor_list_gather(l, [3], element_dtype=dtypes.float32)
      self.evaluate(t)

  def testScatterOutputListSize(self):
    # [FUNCTION DOC]
    # Scatters two values at indices one and three. Expects the resulting list length to be four,
    # that is the largest index plus one.
    c0 = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_scatter(c0, [1, 3], [])
    # TensorListScatter should return a list with size largest index + 1.
    self.assertAllEqual(list_ops.tensor_list_length(l), 4)

  def testScatterOutputListSizeWithNumElementsSpecified(self):
    # [FUNCTION DOC]
    # Scatters two values at indices one and three using the second-version raw op with an explicit
    # number of elements of five. Expects the resulting list length to equal five.
    c0 = constant_op.constant([1.0, 2.0])
    l = gen_list_ops.tensor_list_scatter_v2(
        c0, [1, 3], list_ops._build_element_shape([]), num_elements=5)
    # TensorListScatter should return a list with size num_elements.
    self.assertAllEqual(list_ops.tensor_list_length(l), 5)

  def testScatterFailsWhenElementShapeIsNotVector(self):
    # [FUNCTION DOC]
    # Passes an element shape of rank two to the raw scatter op. Expects an invalid-argument error
    # (eager mode, from the kernel) or a value error (graph mode, from shape inference) stating the
    # shape must be at most rank one.
    c0 = constant_op.constant([1.0, 2.0])
    # In Eager mode, InvalidArgumentError is generated by the Compute function.
    # In graph mode, ValueError is generated by the shape function.
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "must be at most rank 1"):
      l = gen_list_ops.tensor_list_scatter(
          # Wrong element_shape. Should be at most rank 1.
          c0, [1, 3], element_shape=[[1]])
      self.evaluate(l)

  def testScatterV2FailsWhenElementShapeIsNotVector(self):
    # [FUNCTION DOC]
    # Same as the previous scatter test but for the second-version raw op with an explicit number of
    # elements: an element shape of rank two must raise an invalid-argument or value error
    # saying it must be at most rank one.
    c0 = constant_op.constant([1.0, 2.0])
    # In Eager mode, InvalidArgumentError is generated by the Compute function.
    # In graph mode, ValueError is generated by the shape function.
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "must be at most rank 1"):
      l = gen_list_ops.tensor_list_scatter_v2(
          # Wrong element_shape. Should be at most rank 1.
          c0, [1, 3], element_shape=[[1]], num_elements=2)
      self.evaluate(l)

  def testScatterFailsWhenIndexLargerThanNumElements(self):
    # [FUNCTION DOC]
    # Scatters at index three with the number of elements fixed at three. Expects an invalid-argument
    # error that the scatter index is out of range for a list of that size.
    c0 = constant_op.constant([1.0, 2.0])
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "TensorListScatter: Trying to scatter at index 3 in list with size 3"):
      l = gen_list_ops.tensor_list_scatter_v2(
          c0, [1, 3], list_ops._build_element_shape([]), num_elements=3)
      self.evaluate(l)

  def testScatterFailsWithInvalidNumElements(self):
    # [FUNCTION DOC]
    # Scatters with a number of elements of negative two. Expects an invalid-argument error saying
    # the number of elements must be at least negative one.
    c0 = constant_op.constant([1.0, 2.0])
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "TensorListScatter expects num_elements >= -1, found: -2"):
      l = gen_list_ops.tensor_list_scatter_v2(
          c0, [1, 3], list_ops._build_element_shape([]), num_elements=-2)
      self.evaluate(l)

  def testScatterWithInvalidRowsInInputTensorFails(self):
    # [FUNCTION DOC]
    # Scatters a tensor of two rows using three indices. Expects an invalid-argument error that the
    # number of rows in the input tensor is wrong (expected three, actual two).
    c0 = constant_op.constant([1.0, 2.0])
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Invalid number of rows in input tensor. Expected: 3 Actual: 2"):
      l = list_ops.tensor_list_scatter(c0, [1, 0, 2], [])
      self.evaluate(l)

  def testScatterWithNegativeIndicesFails(self):
    # [FUNCTION DOC]
    # Scatters using negative indices. Expects an invalid-argument error that indices must all be
    # non-negative.
    c0 = constant_op.constant([1.0, 2.0])
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Indices in TensorListScatter must all be non-negative."):
      l = list_ops.tensor_list_scatter(c0, [-1, -2], element_shape=[])
      self.evaluate(l)

  @test_util.run_in_graph_and_eager_modes
  def testScatterWithNonScalarFails(self):
    # [FUNCTION DOC]
    # Calls the raw second-version scatter op with a non-scalar number of elements. Expects a value
    # error or invalid-argument error saying the argument must be a scalar or have rank zero.
    c = constant_op.constant(value=[2])
    num_elements = np.array([[], [], []], dtype=np.float32)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r"Shape must be rank 0 but is rank \d+|"
                                r"\w+ must be a scalar"):
      self.evaluate(
          gen_list_ops.TensorListScatterV2(
              tensor=c, indices=c, element_shape=c, num_elements=num_elements))

  def testScatterIntoExistingList(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the list created by the reserve call is immediately overwritten by the next scatter
    # call, which is not given an input list; the reserved list is therefore never used.
    # Scatters one value at index zero, then scatters two values at indices one and two into the
    # resulting list as the input handle, and stacks. Expects the three values in order.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=3)
    l = list_ops.tensor_list_scatter(tensor=[1.], indices=[0], element_shape=[])
    l = list_ops.tensor_list_scatter(
        tensor=[2., 3.], indices=[1, 2], element_shape=[], input_handle=l)
    self.assertAllEqual(
        list_ops.tensor_list_stack(l, element_dtype=dtypes.float32),
        [1., 2., 3.])

  def testScatterGrad(self):
    # [FUNCTION DOC]
    # Under a gradient tape, scatters a two-element vector with reversed indices, reads both elements
    # and checks they are swapped. Checks the gradient of the sum of their squares with respect to
    # the input vector.
    with backprop.GradientTape() as tape:
      c0 = constant_op.constant([1.0, 2.0])
      tape.watch(c0)
      l = list_ops.tensor_list_scatter(c0, [1, 0], element_shape=[])
      t0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      t1 = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(t0), 2.0)
      self.assertAllEqual(self.evaluate(t1), 1.0)
      loss = t0 * t0 + t1 * t1
    dt = tape.gradient(loss, c0)
    self.assertAllEqual(self.evaluate(dt), [2., 4.])

  def testScatterWithPartialReadGrad(self):
    # [FUNCTION DOC]
    # Like the scatter gradient test but reads only one element. Checks the read value and that the
    # gradient is zero for the unread element and twice the value for the read one.
    with backprop.GradientTape() as tape:
      c0 = constant_op.constant([1.0, 2.0])
      tape.watch(c0)
      l = list_ops.tensor_list_scatter(c0, [1, 0], element_shape=[])
      t0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(t0), 2.0)
      loss = t0 * t0
    dt = tape.gradient(loss, c0)
    self.assertAllEqual(self.evaluate(dt), [0., 4.])

  def testTensorListFromTensor(self):
    # [FUNCTION DOC]
    # Creates a list from a two-element vector, reads index zero, then pops twice (getting the values
    # in reverse order) and checks the final list length is zero.
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    e = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    self.assertAllEqual(e, 1.0)
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(e, 2.0)
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(e, 1.0)
    self.assertAllEqual(list_ops.tensor_list_length(l), 0)

  def testTensorListFromTensorFailsWhenElementShapeIsNotVector(self):
    # [FUNCTION DOC]
    # Creates a list from a tensor with an element shape of rank two. Expects an invalid-argument
    # error (eager) or value error (graph) saying the shape must be at most rank one.
    t = constant_op.constant([1.0, 2.0])
    # In Eager mode, InvalidArgumentError is generated by the Compute function.
    # In graph mode, ValueError is generated by the shape function.
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError),
        "must be at most rank 1"):
      # Wrong element_shape. Should be at most rank 1.
      l = list_ops.tensor_list_from_tensor(t, element_shape=[[1]])
      self.evaluate(l)

  @test_util.run_gpu_only
  def testFromTensorGPU(self):
    # [FUNCTION DOC]
    # GPU-only. Runs the list-from-tensor test inside a GPU device scope by calling it directly.
    with context.device("gpu:0"):
      self.testTensorListFromTensor()

  def testGetSetBool(self):
    # [FUNCTION DOC]
    # Creates a boolean list, reads the first element (true), sets it to false and stacks.
    # Expects both elements to be false.
    t = constant_op.constant([True, False])
    l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.bool)
    self.assertAllEqual(self.evaluate(e0), True)
    l = list_ops.tensor_list_set_item(l, 0, False)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.bool)
    self.assertAllEqual(self.evaluate(t), [False, False])

  @test_util.run_gpu_only
  def testGetSetBoolGPU(self):
    # [FUNCTION DOC]
    # GPU-only. Runs the boolean get/set test inside a GPU device scope.
    with context.device("gpu:0"):
      self.testGetSetBool()

  def _testGetSetNumeric(self, dtype):
    # [FUNCTION DOC]
    # HELPER: shared body for the numeric get/set test and its GPU variant.
    # Input: dtype, a floating or complex data type. Creates a list of two values of that type, reads
    # the first, sets it to three, stacks, and asserts the read value and final list.
    t = constant_op.constant([1.0, 2.0], dtype=dtype)
    l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtype)
    self.assertAllEqual(self.evaluate(e0), 1.0)
    l = list_ops.tensor_list_set_item(
        l, 0, constant_op.constant(3.0, dtype=dtype))
    t = list_ops.tensor_list_stack(l, element_dtype=dtype)
    self.assertAllEqual(self.evaluate(t), [3.0, 2.0])

  @parameterized.parameters([dtypes.float32, dtypes.float64,
                             dtypes.complex64, dtypes.complex128])
  def testGetSetNumeric(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float32, float64, complex64 and complex128; runs the shared numeric get/set
    # scenario for each.
    self._testGetSetNumeric(dtype)

  @parameterized.parameters([dtypes.float32, dtypes.float64,
                             dtypes.complex64, dtypes.complex128])
  @test_util.run_gpu_only
  def testGetSetNumericGPU(self, dtype):
    # [FUNCTION DOC]
    # GPU-only, parameterized over the same four numeric types; runs the shared get/set scenario
    # inside a GPU device scope.
    with context.device("gpu:0"):
      self._testGetSetNumeric(dtype)

  def testGetSetReserved(self):
    # [FUNCTION DOC]
    # Reserves a two-element scalar list, reads an unset element (zero), sets the first to three and
    # stacks. Expects three and zero.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=2)
    e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    self.assertAllEqual(e0, 0.0)
    l = list_ops.tensor_list_set_item(l, 0, 3.0)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(t, [3.0, 0.0])

  @test_util.run_gpu_only
  def testGetSetReservedGPU(self):
    # [FUNCTION DOC]
    # GPU-only. Runs the reserved-list get/set test inside a GPU device scope.
    with context.device("gpu:0"):
      self.testGetSetReserved()

  def testSetGetGrad(self):
    # [FUNCTION DOC]
    # Under a gradient tape, sets one element of a reserved list to twice a watched scalar and reads
    # it back. Expects the value ten and a gradient of two with respect to the scalar.
    with backprop.GradientTape() as tape:
      t = constant_op.constant(5.)
      tape.watch(t)
      l = list_ops.tensor_list_reserve(
          element_dtype=dtypes.float32, element_shape=[], num_elements=3)
      l = list_ops.tensor_list_set_item(l, 1, 2. * t)
      e = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
      self.assertAllEqual(self.evaluate(e), 10.0)
    self.assertAllEqual(self.evaluate(tape.gradient(e, t)), 2.0)

  def testGetUninitializedTensorUseListElementShape(self):
    # [FUNCTION DOC]
    # Reserves a scalar list of three, sets index zero, and reads indices one and two which were never
    # set. Expects both to be zero.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=3)
    l = list_ops.tensor_list_set_item(l, 0, 5.)
    e1 = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
    e2 = list_ops.tensor_list_get_item(l, 2, element_dtype=dtypes.float32)
    self.assertEqual(self.evaluate(e1), 0.)
    self.assertEqual(self.evaluate(e2), 0.)

  def testGetUninitializedTensorUseSpecifiedElementShape(self):
    # [FUNCTION DOC]
    # Reserves lists with unknown or partially defined element shape and reads unset elements with
    # the raw op using an explicit element shape (a list, or a tensor of integers). Checks the static
    # shapes of the results and that the values are zeros of the requested shape.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    e0 = gen_list_ops.tensor_list_get_item(
        l, 0, element_shape=[], element_dtype=dtypes.float32)
    e1 = gen_list_ops.tensor_list_get_item(
        l,
        1,
        element_shape=constant_op.constant([2, 3], dtype=dtypes.int64),
        element_dtype=dtypes.float32,
    )
    self.assertEqual(e0.shape.as_list(), [])
    self.assertEqual(e1.shape.as_list(), [2, 3])
    self.assertEqual(self.evaluate(e0), 0.)
    self.assertAllEqual(self.evaluate(e1), np.zeros((2, 3)))

    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[None, 3], num_elements=3)
    e1 = gen_list_ops.tensor_list_get_item(
        l, 1, element_shape=[2, 3], element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e1), np.zeros((2, 3)))

  def testGetUninitializedTensorWithInvalidElementShapeFails(self):
    # [FUNCTION DOC]
    # Reading an unset element of a list with unknown element shape must raise an invalid-argument
    # error that the shape is not fully defined. For a list with partially defined element shape and
    # an incompatible explicit shape, expects an invalid-argument error in eager mode and a value
    # error in graph mode, both mentioning shapes.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Trying to read an uninitialized tensor but "
        "element_shape is not fully defined"):
      e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      self.evaluate(e0)

    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[None, 2], num_elements=3)

    # In eager mode the shape mismatch is caught in the TensorListGetItem
    # kernel which raises an InvalidArgumentError.
    # In graph mode the shape mismatch is caught in the C++ shape inference
    # which raises a ValueError.
    if context.executing_eagerly():
      error_type = errors.InvalidArgumentError
    else:
      error_type = ValueError
    with self.assertRaisesRegex(error_type, r"shapes"):
      e0 = gen_list_ops.tensor_list_get_item(
          l, 0, element_dtype=dtypes.float32, element_shape=[1, 3])
      self.evaluate(e0)

  @test_util.run_deprecated_v1
  @test_util.enable_control_flow_v2
  def testSkipEagerSetItemIndexOutOfBounds(self):
    # [FUNCTION DOC]
    # Graph-only with control flow version two enabled. Sets elements zero and one of an empty list
    # with automatic resizing allowed, stacks, and takes the gradient with respect to a source value
    # that was doubled before being set. Expects the gradient to be two.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[])
    e0 = constant_op.constant(5.)
    l = list_ops.tensor_list_set_item(
        l, 0, 2. * e0, resize_if_index_out_of_bounds=True)
    l = list_ops.tensor_list_set_item(
        l, 1, 1., resize_if_index_out_of_bounds=True)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    grad = gradients_impl.gradients(t, e0)[0]
    self.assertAllEqual(self.evaluate(grad), 2.)

  @test_util.run_deprecated_v1
  def testSetOnEmptyListWithMaxNumElementsFails(self):
    # [FUNCTION DOC]
    # Graph-mode only. Sets index zero in an empty list that has a maximum size of three but no
    # elements. Expects an invalid-argument error that the element cannot be modified in a list with
    # zero elements.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[], max_num_elements=3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Trying to modify element 0 in a list with 0 elements."):
      l = list_ops.tensor_list_set_item(l, 0, 1.)
      self.evaluate(l)

  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Pushes a scalar and then a vector onto a list with unknown element shape and pops both back,
    # checking the vector comes out first and then the scalar.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=None)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([1.0, 2.0]))
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e), [1.0, 2.0])
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e), 1.0)

  @test_util.run_gpu_only
  def testCPUGPUCopy(self):
    # [FUNCTION DOC]
    # GPU-only. Copies a list to a GPU with an identity op, pops from it, then copies it back to the
    # CPU and pops again. Both pops must produce the last element, two.
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    with context.device("gpu:0"):
      l_gpu = array_ops.identity(l)
      self.assertAllEqual(
          self.evaluate(
              list_ops.tensor_list_pop_back(
                  l_gpu, element_dtype=dtypes.float32)[1]), 2.0)
    l_cpu = array_ops.identity(l_gpu)
    self.assertAllEqual(
        self.evaluate(
            list_ops.tensor_list_pop_back(
                l_cpu, element_dtype=dtypes.float32)[1]), 2.0)

  @test_util.run_gpu_only
  def testCPUGPUCopyNested(self):
    # [FUNCTION DOC]
    # GPU-only. Same as the copy test, but uses a list whose single element is itself a list. The
    # nested list is copied to GPU and back by identity ops and in each place the inner list is
    # popped; the last inner value must be two.
    t = constant_op.constant([1.0, 2.0])
    child_l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    l = list_ops.empty_tensor_list(
        element_shape=constant_op.constant([], dtype=dtypes.int32),
        element_dtype=dtypes.variant)
    l = list_ops.tensor_list_push_back(l, child_l)
    with context.device("gpu:0"):
      l_gpu = array_ops.identity(l)
      _, child_l_gpu = list_ops.tensor_list_pop_back(
          l_gpu, element_dtype=dtypes.variant)
      self.assertAllEqual(
          self.evaluate(
              list_ops.tensor_list_pop_back(
                  child_l_gpu, element_dtype=dtypes.float32)[1]), 2.0)
    l_cpu = array_ops.identity(l_gpu)
    _, child_l_cpu = list_ops.tensor_list_pop_back(
        l_cpu, element_dtype=dtypes.variant)
    self.assertAllEqual(
        self.evaluate(
            list_ops.tensor_list_pop_back(
                child_l_cpu, element_dtype=dtypes.float32)[1]), 2.0)

  def testGraphStack(self):
    # [FUNCTION DOC]
    # Within a cached session, pushes a one-element vector to a list with a known element shape,
    # stacks, and expects a one by one result.
    with self.cached_session():
      tl = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([1], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      tl = list_ops.tensor_list_push_back(tl, [1])
      self.assertAllEqual(
          self.evaluate(
              list_ops.tensor_list_stack(tl, element_dtype=dtypes.int32)),
          [[1]])

  def testSkipEagerStackInLoop(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says to skip eager mode but there is no skip decorator; the test runs in
    # both modes because of the class-level decorator.
    # Within a cached session, a while loop with four iterations pushes the loop counter onto an
    # initially empty integer list. After the loop the list is stacked and must equal zero to three.
    with self.cached_session():
      t1 = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      i = constant_op.constant(0, dtype=dtypes.int32)

      def body(i, t1):
        # [FUNCTION DOC]
        # HELPER: loop body for the stack-in-loop test.
        # Inputs: the loop counter and a list. Pushes the counter onto the list and increments it.
        # Returns the new counter and list.
        t1 = list_ops.tensor_list_push_back(t1, i)
        i += 1
        return i, t1

      i, t1 = while_loop.while_loop(lambda i, t1: math_ops.less(i, 4), body,
                                    [i, t1])
      s1 = list_ops.tensor_list_stack(t1, element_dtype=dtypes.int32)
      self.assertAllEqual(self.evaluate(s1), [0, 1, 2, 3])

  def testSkipEagerStackSwitchDtype(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests a graph loop and skipping eager mode, but the body function is called from an
    # ordinary Python loop and nothing skips eager mode.
    # Starts with an empty integer list. Twice runs a body that replaces the list by a fresh empty
    # float list if the current one is empty, then pushes a float vector. Stacks and expects two rows
    # of the vector, showing that the list element type can change when starting from an empty list.
    with self.cached_session():
      list_ = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      m = constant_op.constant([1, 2, 3], dtype=dtypes.float32)

      def body(list_, m):
        # [FUNCTION DOC]
        # HELPER: step function for the switch-dtype test.
        # Inputs: a list and a float vector. If the list is empty, swaps it for an empty list typed and
        # shaped like the vector (through a conditional); then pushes the vector. Returns the list and the
        # vector.
        list_ = cond.cond(
            math_ops.equal(list_ops.tensor_list_length(list_), 0),
            lambda: list_ops.empty_tensor_list(m.shape, m.dtype), lambda: list_)
        list_ = list_ops.tensor_list_push_back(list_, m)
        return list_, m

      for _ in range(2):
        list_, m = body(list_, m)

      s1 = list_ops.tensor_list_stack(list_, element_dtype=dtypes.float32)
      np_s1 = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
      self.assertAllEqual(self.evaluate(s1), np_s1)

  def testSkipEagerStackInLoopSwitchDtype(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says to skip eager mode, but no skip decorator exists, so it also runs in
    # eager mode.
    # Within a cached session, a while loop runs four times; each iteration conditionally replaces an
    # empty integer list with an empty float list and pushes the vector scaled by the counter.
    # The stacked result must equal the matrix whose rows are the vector times zero through three.
    with self.cached_session():
      t1 = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      i = constant_op.constant(0, dtype=dtypes.float32)
      m = constant_op.constant([1, 2, 3], dtype=dtypes.float32)

      def body(i, m, t1):
        # [FUNCTION DOC]
        # HELPER: loop body for the loop switch-dtype test.
        # Inputs: float counter, float vector, list. Replaces an empty list by an empty list of the
        # vector's type and shape, pushes the vector multiplied by the counter, adds one to the counter.
        # Returns counter, vector and list.
        t1 = cond.cond(
            math_ops.equal(list_ops.tensor_list_length(t1), 0),
            lambda: list_ops.empty_tensor_list(m.shape, m.dtype), lambda: t1)

        t1 = list_ops.tensor_list_push_back(t1, m * i)
        i += 1.0
        return i, m, t1

      i, m, t1 = while_loop.while_loop(lambda i, m, t1: math_ops.less(i, 4),
                                       body, [i, m, t1])
      s1 = list_ops.tensor_list_stack(t1, element_dtype=dtypes.float32)
      np_s1 = np.vstack([np.arange(1, 4) * i for i in range(4)])
      self.assertAllEqual(self.evaluate(s1), np_s1)

  def testSerialize(self):
    # [FUNCTION DOC]
    # Starts a local cluster with one worker and one parameter server and works in a graph bound to
    # the worker. Creates a list from a tensor on the worker, copies it to the parameter server with
    # identity, pops there, and copies the popped value back to the worker. Expects the value two,
    # checking that lists are serialized across devices.
    worker = test_util.create_local_cluster(num_workers=1, num_ps=1)[0][0]
    with ops.Graph().as_default(), session.Session(target=worker.target):
      with ops.device("/job:worker"):
        t = constant_op.constant([[1.0], [2.0]])
        l = list_ops.tensor_list_from_tensor(t, element_shape=[1])
      with ops.device("/job:ps"):
        l_ps = array_ops.identity(l)
        l_ps, e = list_ops.tensor_list_pop_back(
            l_ps, element_dtype=dtypes.float32)
      with ops.device("/job:worker"):
        worker_e = array_ops.identity(e)
      self.assertAllEqual(self.evaluate(worker_e), [2.0])

  def testSerializeListWithInvalidTensors(self):
    # [FUNCTION DOC]
    # Local cluster test. Creates a reserved two-element list on the worker with one element set,
    # copies it to the parameter server, sets the other element there, stacks and brings the result
    # back to the worker. Expects both values; this verifies that unset elements survive serialization.
    worker = test_util.create_local_cluster(num_workers=1, num_ps=1)[0][0]
    with ops.Graph().as_default(), session.Session(target=worker.target):
      with ops.device("/job:worker"):
        l = list_ops.tensor_list_reserve(
            element_dtype=dtypes.float32, element_shape=[], num_elements=2)
        l = list_ops.tensor_list_set_item(l, 0, 1.)
      with ops.device("/job:ps"):
        l_ps = array_ops.identity(l)
        l_ps = list_ops.tensor_list_set_item(l_ps, 1, 2.)
        t = list_ops.tensor_list_stack(l_ps, element_dtype=dtypes.float32)
      with ops.device("/job:worker"):
        worker_t = array_ops.identity(t)
      self.assertAllEqual(self.evaluate(worker_t), [1.0, 2.0])

  def testSerializeListWithUnknownRank(self):
    # [FUNCTION DOC]
    # Local cluster test. Creates a list with unknown element shape on the worker, copies it to the
    # parameter server and reads its element shape, then moves that back to the worker. Expects the
    # marker value for unknown shape (negative one).
    worker = test_util.create_local_cluster(num_workers=1, num_ps=1)[0][0]
    with ops.Graph().as_default(), session.Session(target=worker.target):
      with ops.device("/job:worker"):
        t = constant_op.constant([[1.0], [2.0]])
        l = list_ops.tensor_list_from_tensor(t, element_shape=None)
      with ops.device("/job:ps"):
        l_ps = array_ops.identity(l)
        element_shape = list_ops.tensor_list_element_shape(
            l_ps, shape_type=dtypes.int32)
      with ops.device("/job:worker"):
        element_shape = array_ops.identity(element_shape)
      self.assertEqual(self.evaluate(element_shape), -1)

  def testSerializeListWithMaxNumElements(self):
    # [FUNCTION DOC]
    # Local cluster test. Creates a list limited to two elements and pushes on the worker, copies to
    # the parameter server and pushes again, then copies back to the worker and pushes a third value.
    # Expects an invalid-argument error that the list is full, showing the size limit is preserved
    # across devices.
    worker = test_util.create_local_cluster(num_workers=1, num_ps=1)[0][0]
    with ops.Graph().as_default(), session.Session(target=worker.target):
      with ops.device("/job:worker"):
        l = list_ops.empty_tensor_list(
            element_shape=None,
            element_dtype=dtypes.float32,
            max_num_elements=2)
        l = list_ops.tensor_list_push_back(l, 1.)
      with ops.device("/job:ps"):
        l_ps = array_ops.identity(l)
        l_ps = list_ops.tensor_list_push_back(l_ps, 2.)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Tried to push item into a full list"):
        with ops.device("/job:worker"):
          l_worker = array_ops.identity(l_ps)
          l_worker = list_ops.tensor_list_push_back(l_worker, 3.0)
          self.evaluate(l_worker)

  def testPushPopGradients(self):
    # [FUNCTION DOC]
    # Under a gradient tape, pushes a watched value and pops it, and doubles the result. Expects a
    # gradient of two with respect to the pushed value.
    with backprop.GradientTape() as tape:
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32, element_shape=[])
      c = constant_op.constant(1.0)
      tape.watch(c)
      l = list_ops.tensor_list_push_back(l, c)
      l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      e = 2 * e
    self.assertAllEqual(self.evaluate(tape.gradient(e, [c])[0]), 2.0)

  def testStackFromTensorGradients(self):
    # [FUNCTION DOC]
    # Under a gradient tape, builds a list from a watched vector, stacks it and doubles the result.
    # Expects a gradient of two for each element.
    with backprop.GradientTape() as tape:
      c = constant_op.constant([1.0, 2.0])
      tape.watch(c)
      l = list_ops.tensor_list_from_tensor(c, element_shape=[])
      c2 = list_ops.tensor_list_stack(
          l, element_dtype=dtypes.float32, num_elements=2)
      result = c2 * 2.0
    grad = tape.gradient(result, [c])[0]
    self.assertAllEqual(self.evaluate(grad), [2.0, 2.0])

  def testGetSetGradients(self):
    # [FUNCTION DOC]
    # Under a gradient tape, builds a list from a watched vector, overwrites element zero with
    # another watched scalar, reads both elements and forms the sum of their squares. Expects the
    # vector gradient to be zero for the overwritten element and four for the other, and the scalar
    # gradient to be six.
    with backprop.GradientTape() as tape:
      c = constant_op.constant([1.0, 2.0])
      tape.watch(c)
      l = list_ops.tensor_list_from_tensor(c, element_shape=[])
      c2 = constant_op.constant(3.0)
      tape.watch(c2)
      l = list_ops.tensor_list_set_item(l, 0, c2)
      e = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      ee = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
      y = e * e + ee * ee
    grad_c, grad_c2 = tape.gradient(y, [c, c2])
    self.assertAllEqual(self.evaluate(grad_c), [0.0, 4.0])
    self.assertAllEqual(self.evaluate(grad_c2), 6.0)

  @test_util.run_deprecated_v1
  def testSetOutOfBounds(self):
    # [FUNCTION DOC]
    # Graph-mode only. Sets index twenty in a two-element list and expects an invalid-argument error
    # on evaluation.
    c = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(c, element_shape=[])
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(list_ops.tensor_list_set_item(l, 20, 3.0))

  @test_util.run_deprecated_v1
  def testSkipEagerSetItemWithMismatchedShapeFails(self):
    # [FUNCTION DOC]
    # Graph-mode only. Sets an element of a scalar-shaped list to a placeholder of unknown shape so
    # that graph building succeeds, reads it back, and runs with a one-element vector fed in.
    # Expects an invalid-argument error about an incompatible shape.
    with self.cached_session() as sess:
      ph = array_ops.placeholder(dtypes.float32)
      c = constant_op.constant([1.0, 2.0])
      l = list_ops.tensor_list_from_tensor(c, element_shape=[])
      # Set a placeholder with unknown shape to satisfy the shape inference
      # at graph building time.
      l = list_ops.tensor_list_set_item(l, 0, ph)
      l_0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "incompatible shape"):
        sess.run(l_0, {ph: [3.0]})

  def testResourceVariableScatterGather(self):
    # [FUNCTION DOC]
    # Creates a resource variable holding ten copies of a float list. Checks that indexing and a
    # sparse read yield the original list values. Then scatter-updates two rows (three and five) with
    # new lists, unstacks the variable, stacks each list and compares with the expected nested values.
    c = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
    l = list_ops.tensor_list_from_tensor(c, element_shape=[])
    v = vs.get_variable("var", initializer=[l] * 10, use_resource=True)
    v_r_0_stacked = list_ops.tensor_list_stack(v[0], dtypes.float32)
    self.evaluate(v.initializer)
    self.assertAllEqual([1.0, 2.0], self.evaluate(v_r_0_stacked))
    v_r_sparse_stacked = list_ops.tensor_list_stack(
        v.sparse_read(0), dtypes.float32)
    self.assertAllEqual([1.0, 2.0], self.evaluate(v_r_sparse_stacked))
    l_new_0 = list_ops.tensor_list_from_tensor([3.0, 4.0], element_shape=[])
    l_new_1 = list_ops.tensor_list_from_tensor([5.0, 6.0], element_shape=[])
    updated_v = state_ops.scatter_update(v, [3, 5], [l_new_0, l_new_1])
    updated_v_elems = array_ops_stack.unstack(updated_v)
    updated_v_stacked = [
        list_ops.tensor_list_stack(el, dtypes.float32) for el in updated_v_elems
    ]
    expected = ([[1.0, 2.0]] * 3 + [[3.0, 4.0], [1.0, 2.0], [5.0, 6.0]] +
                [[1.0, 2.0]] * 4)
    self.assertAllEqual(self.evaluate(updated_v_stacked), expected)

  def testResourceVariableScatterGatherInt64(self):
    # [FUNCTION DOC]
    # Same as the resource variable scatter/gather test but with int64 list elements.
    c = constant_op.constant([1, 2], dtype=dtypes.int64)
    l = list_ops.tensor_list_from_tensor(c, element_shape=[])
    v = vs.get_variable("var", initializer=[l] * 10, use_resource=True)
    v_r_0_stacked = list_ops.tensor_list_stack(v[0], dtypes.int64)
    self.evaluate(v.initializer)
    self.assertAllEqual([1, 2], self.evaluate(v_r_0_stacked))
    v_r_sparse_stacked = list_ops.tensor_list_stack(
        v.sparse_read(0), dtypes.int64)
    self.assertAllEqual([1, 2], self.evaluate(v_r_sparse_stacked))
    c34 = constant_op.constant([3, 4], dtype=dtypes.int64)
    l_new_0 = list_ops.tensor_list_from_tensor(c34, element_shape=[])
    c56 = constant_op.constant([5, 6], dtype=dtypes.int64)
    l_new_1 = list_ops.tensor_list_from_tensor(c56, element_shape=[])
    updated_v = state_ops.scatter_update(v, [3, 5], [l_new_0, l_new_1])
    updated_v_elems = array_ops_stack.unstack(updated_v)
    updated_v_stacked = [
        list_ops.tensor_list_stack(el, dtypes.int64) for el in updated_v_elems
    ]
    expected = ([[1, 2]] * 3 + [[3, 4], [1, 2], [5, 6]] +
                [[1, 2]] * 4)
    self.assertAllEqual(self.evaluate(updated_v_stacked), expected)

  @test_util.run_deprecated_v1
  def testConcat(self):
    # [FUNCTION DOC]
    # Graph-mode only. Builds two batches of two lists with different lengths and concatenates them
    # pairwise in all four combinations, unstacking and comparing every result with expected values
    # (printing each case). Then checks that concatenating with a differently shaped batch fails with
    # an invalid-argument or value error, with mismatched element ranks fails with specific messages
    # that differ by eager or graph mode, and with a batch of the wrong element dtype fails likewise.
    c = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
    l0 = list_ops.tensor_list_from_tensor(c, element_shape=[])
    l1 = list_ops.tensor_list_from_tensor([-1.0], element_shape=[])
    l_batch_0 = array_ops_stack.stack([l0, l1])
    l_batch_1 = array_ops_stack.stack([l1, l0])

    l_concat_01 = list_ops.tensor_list_concat_lists(
        l_batch_0, l_batch_1, element_dtype=dtypes.float32)
    l_concat_10 = list_ops.tensor_list_concat_lists(
        l_batch_1, l_batch_0, element_dtype=dtypes.float32)
    l_concat_00 = list_ops.tensor_list_concat_lists(
        l_batch_0, l_batch_0, element_dtype=dtypes.float32)
    l_concat_11 = list_ops.tensor_list_concat_lists(
        l_batch_1, l_batch_1, element_dtype=dtypes.float32)

    expected_0 = [[1.0, 2.0], [-1.0]]
    expected_1 = [[-1.0], [1.0, 2.0]]
    expected_00 = [[1.0, 2.0, 1.0, 2.0], [-1.0, -1.0]]
    expected_01 = [[1.0, 2.0, -1.0], [-1.0, 1.0, 2.0]]
    expected_10 = [[-1.0, 1.0, 2.0], [1.0, 2.0, -1.0]]
    expected_11 = [[-1.0, -1.0], [1.0, 2.0, 1.0, 2.0]]

    for i, (concat, expected) in enumerate(zip(
        [l_batch_0, l_batch_1,
         l_concat_00, l_concat_01, l_concat_10, l_concat_11],
        [expected_0, expected_1,
         expected_00, expected_01, expected_10, expected_11])):
      splitted = array_ops_stack.unstack(concat)                # go/LEGACY_TYPO
      splitted_stacked_ret = self.evaluate(                     # go/LEGACY_TYPO
          (list_ops.tensor_list_stack(splitted[0], dtypes.float32),
           list_ops.tensor_list_stack(splitted[1], dtypes.float32)))
      print("Test concat %d: %s, %s, %s, %s"
            % (i, expected[0], splitted_stacked_ret[0],
               expected[1], splitted_stacked_ret[1]))
      self.assertAllClose(expected[0], splitted_stacked_ret[0])
      self.assertAllClose(expected[1], splitted_stacked_ret[1])

    # Concatenating mismatched shapes fails.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(
          list_ops.tensor_list_concat_lists(
              l_batch_0,
              list_ops.empty_tensor_list([], dtypes.float32),
              element_dtype=dtypes.float32))

    if context.executing_eagerly():
      expected_error = (
          errors.InvalidArgumentError,
          "element shapes are not identical at index 0")
    else:
      expected_error = (ValueError, "Shapes must be equal rank")
    with self.assertRaisesRegex(*expected_error):
      l_batch_of_vec_tls = array_ops_stack.stack(
          [list_ops.tensor_list_from_tensor([[1.0]], element_shape=[1])] * 2)
      self.evaluate(
          list_ops.tensor_list_concat_lists(l_batch_0, l_batch_of_vec_tls,
                                            element_dtype=dtypes.float32))

    if context.executing_eagerly():
      expected_error = (errors.InvalidArgumentError,
                        r"input_b\[0\].dtype != element_dtype.")
    else:
      expected_error = (ValueError, "input_b.type != element_dtype")
    with self.assertRaisesRegex(*expected_error):
      l_batch_of_int_tls = array_ops_stack.stack(
          [list_ops.tensor_list_from_tensor([1], element_shape=[])] * 2)
      self.evaluate(
          list_ops.tensor_list_concat_lists(l_batch_0, l_batch_of_int_tls,
                                            element_dtype=dtypes.float32))

  @test_util.run_deprecated_v1
  def testPushBackBatch(self):
    # [FUNCTION DOC]
    # Graph-mode only. Pushes one value onto each list of a batch of two lists and checks the results
    # and that the original batch is unchanged. Then verifies failures for a wrong-length value
    # list, wrongly shaped values (incompatible shape at index zero), and wrong dtype (message differs
    # between eager and graph mode).
    c = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
    l0 = list_ops.tensor_list_from_tensor(c, element_shape=[])
    l1 = list_ops.tensor_list_from_tensor([-1.0], element_shape=[])
    l_batch = array_ops_stack.stack([l0, l1])
    l_push = list_ops.tensor_list_push_back_batch(l_batch, [3.0, 4.0])
    l_unstack = array_ops_stack.unstack(l_push)
    l0_ret = list_ops.tensor_list_stack(l_unstack[0], dtypes.float32)
    l1_ret = list_ops.tensor_list_stack(l_unstack[1], dtypes.float32)
    self.assertAllClose([1.0, 2.0, 3.0], self.evaluate(l0_ret))
    self.assertAllClose([-1.0, 4.0], self.evaluate(l1_ret))

    with ops.control_dependencies([l_push]):
      l_unstack_orig = array_ops_stack.unstack(l_batch)
      l0_orig_ret = list_ops.tensor_list_stack(l_unstack_orig[0],
                                               dtypes.float32)
      l1_orig_ret = list_ops.tensor_list_stack(l_unstack_orig[1],
                                               dtypes.float32)

    # Check that without aliasing, push_back_batch still works; and
    # that it doesn't modify the input.
    l0_r_v, l1_r_v, l0_orig_v, l1_orig_v = self.evaluate(
        (l0_ret, l1_ret, l0_orig_ret, l1_orig_ret))
    self.assertAllClose([1.0, 2.0, 3.0], l0_r_v)
    self.assertAllClose([-1.0, 4.0], l1_r_v)
    self.assertAllClose([1.0, 2.0], l0_orig_v)
    self.assertAllClose([-1.0], l1_orig_v)

    # Pushing back mismatched shapes fails.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(list_ops.tensor_list_push_back_batch(l_batch, []))

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "incompatible shape to a list at index 0"):
      self.evaluate(
          list_ops.tensor_list_push_back_batch(l_batch, [[3.0], [4.0]]))

    if context.executing_eagerly():
      expected_error = (errors.InvalidArgumentError, "Invalid data type")
    else:
      expected_error = (ValueError, "wrong element dtype")
    with self.assertRaisesRegex(*expected_error):
      self.evaluate(list_ops.tensor_list_push_back_batch(l_batch, [3, 4]))

  def testZerosLike(self):
    # [FUNCTION DOC]
    # For each of twelve numeric and boolean types, makes an empty list and a two-element list and
    # applies zeros-like to both. Expects the stacked result to be empty and an all-zero vector of
    # length two respectively.
    for dtype in (dtypes.uint8, dtypes.uint16, dtypes.int8, dtypes.int16,
                  dtypes.int32, dtypes.int64, dtypes.float16, dtypes.float32,
                  dtypes.float64, dtypes.complex64, dtypes.complex128,
                  dtypes.bool):
      l_empty = list_ops.empty_tensor_list(
          element_dtype=dtype, element_shape=[])
      l_empty_zeros = array_ops.zeros_like(l_empty)
      t_empty_zeros = list_ops.tensor_list_stack(
          l_empty_zeros, element_dtype=dtype)

      l_full = list_ops.tensor_list_push_back(l_empty,
                                              math_ops.cast(0, dtype=dtype))
      l_full = list_ops.tensor_list_push_back(l_full,
                                              math_ops.cast(1, dtype=dtype))
      l_full_zeros = array_ops.zeros_like(l_full)
      t_full_zeros = list_ops.tensor_list_stack(
          l_full_zeros, element_dtype=dtype)

      self.assertAllEqual(self.evaluate(t_empty_zeros), [])
      self.assertAllEqual(
          self.evaluate(t_full_zeros), np.zeros(
              (2,), dtype=dtype.as_numpy_dtype))

  def testZerosLikeNested(self):
    # [FUNCTION DOC]
    # For each of twelve numeric and boolean types, builds a list whose elements are lists of
    # different lengths (zero, one and two items). Applies zeros-like and pops the three inner lists,
    # expecting an empty list, one zero, and two zeros (popped in reverse order).
    for dtype in (dtypes.uint8, dtypes.uint16, dtypes.int8, dtypes.int16,
                  dtypes.int32, dtypes.int64, dtypes.float16, dtypes.float32,
                  dtypes.float64, dtypes.complex64, dtypes.complex128,
                  dtypes.bool):
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.variant, element_shape=[])

      sub_l = list_ops.empty_tensor_list(element_dtype=dtype, element_shape=[])
      l = list_ops.tensor_list_push_back(l, sub_l)
      sub_l = list_ops.tensor_list_push_back(sub_l, math_ops.cast(
          1, dtype=dtype))
      l = list_ops.tensor_list_push_back(l, sub_l)
      sub_l = list_ops.tensor_list_push_back(sub_l, math_ops.cast(
          2, dtype=dtype))
      l = list_ops.tensor_list_push_back(l, sub_l)

      # l : [[],
      #      [1],
      #      [1, 2]]
      #
      # l_zeros : [[],
      #            [0],
      #            [0, 0]]
      l_zeros = array_ops.zeros_like(l)

      outputs = []
      for _ in range(3):
        l_zeros, out = list_ops.tensor_list_pop_back(
            l_zeros, element_dtype=dtypes.variant)
        outputs.append(list_ops.tensor_list_stack(out, element_dtype=dtype))

      # Note: `outputs` contains popped values so the order is reversed.
      self.assertAllEqual(self.evaluate(outputs[2]), [])
      self.assertAllEqual(
          self.evaluate(outputs[1]), np.zeros((1,), dtype=dtype.as_numpy_dtype))
      self.assertAllEqual(
          self.evaluate(outputs[0]), np.zeros((2,), dtype=dtype.as_numpy_dtype))

  def testElementShape(self):
    # [FUNCTION DOC]
    # Reads the element shape of a list that has unknown element shape. Expects negative one.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=None)
    shape = list_ops.tensor_list_element_shape(l, shape_type=dtypes.int32)
    self.assertEqual(self.evaluate(shape), -1)

  def testZerosLikeUninitialized(self):
    # [FUNCTION DOC]
    # Reserves a three-element list, sets some elements, and applies zeros-like. Gathers the set
    # indices of the result and expects zeros; unset slots stay uninitialized.
    l0 = list_ops.tensor_list_reserve([], 3, element_dtype=dtypes.float32)
    l1 = list_ops.tensor_list_set_item(l0, 0, 1.)  # [1., _, _]
    zeros_1 = array_ops.zeros_like(l1)  # [0., _, _]
    l2 = list_ops.tensor_list_set_item(l1, 2, 2.)  # [1., _, 2.]
    zeros_2 = array_ops.zeros_like(l2)  # [0., _, 0.]

    # Gather indices with zeros in `zeros_1`.
    res_1 = list_ops.tensor_list_gather(
        zeros_1, [0], element_dtype=dtypes.float32)
    # Gather indices with zeros in `zeros_2`.
    res_2 = list_ops.tensor_list_gather(
        zeros_2, [0, 2], element_dtype=dtypes.float32)

    self.assertAllEqual(self.evaluate(res_1), [0.])
    self.assertAllEqual(self.evaluate(res_2), [0., 0.])

  @test_util.run_deprecated_v1
  def testSkipEagerTensorListGetItemGradAggregation(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a session variable is opened but never used directly; evaluation goes through the
    # test helper.
    # Graph-only. Reads the same list element twice and takes the gradient of both reads with respect
    # to the stored value. Expects the gradients to be aggregated into two.
    l = list_ops.tensor_list_reserve(
        element_shape=[], num_elements=1, element_dtype=dtypes.float32)
    x = constant_op.constant(1.0)
    l = list_ops.tensor_list_set_item(l, 0, x)
    l_read1 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    l_read2 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    grad = gradients_impl.gradients([l_read1, l_read2], [x])
    with self.cached_session() as sess:
      self.assertSequenceEqual(self.evaluate(grad), [2.])

  @test_util.run_deprecated_v1
  def testSkipEagerBuildElementShape(self):
    # [FUNCTION DOC]
    # Graph-only. Tests the private element-shape builder of list_ops. Unknown shapes become negative
    # one; scalar shapes become an empty int32 tensor; a tensor is returned unchanged; shapes with
    # unknown dimensions become lists with negative one for unknown entries; a mix with a tensor
    # dimension keeps the tensor object in place.
    fn = list_ops._build_element_shape
    # Unknown shape -> -1.
    self.assertEqual(fn(None), -1)
    self.assertEqual(fn(tensor_shape.unknown_shape()), -1)
    # Scalar shape -> [] with type int32.
    self.assertEqual(fn([]).dtype, dtypes.int32)
    self.assertEqual(fn(tensor_shape.TensorShape([])).dtype, dtypes.int32)
    self.assertAllEqual(self.evaluate(fn([])), np.array([], np.int32))
    self.assertAllEqual(
        self.evaluate(fn(tensor_shape.TensorShape([]))), np.array([], np.int32))
    # Tensor -> Tensor
    shape = constant_op.constant(1)
    self.assertIs(fn(shape), shape)
    # Shape with unknown dims -> shape list with -1's.
    shape = [None, 5]
    self.assertAllEqual(fn(shape), [-1, 5])
    self.assertAllEqual(fn(tensor_shape.TensorShape(shape)), [-1, 5])
    # Shape with unknown dims and tensor dims -> shape list with -1's and tensor
    # dims.
    t = array_ops.placeholder(dtypes.int32)
    shape = [None, 5, t]
    result = fn(shape)
    self.assertAllEqual(result[:2], [-1, 5])
    self.assertIs(result[2], t)

  def testAddN(self):
    # [FUNCTION DOC]
    # Adds three tensor lists elementwise with the add-n op, stacks the sum and expects the
    # elementwise sums.
    l1 = list_ops.tensor_list_from_tensor([1.0, 2.0], element_shape=[])
    l2 = list_ops.tensor_list_from_tensor([3.0, 4.0], element_shape=[])
    l3 = list_ops.tensor_list_from_tensor([5.0, 6.0], element_shape=[])
    result = math_ops.add_n((l1, l2, l3))
    result_t = list_ops.tensor_list_stack(result, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(result_t), [9., 12.])

  def testAddNNestedList(self):
    # [FUNCTION DOC]
    # Adds two lists whose elements are themselves lists with the add-n op. Reads each inner list of
    # the sum, stacks it and expects the elementwise sums.
    l1 = list_ops.tensor_list_from_tensor([1.0, 2.0], element_shape=[])
    l2 = list_ops.tensor_list_from_tensor([3.0, 4.0], element_shape=[])
    l3 = list_ops.tensor_list_from_tensor([5.0, 6.0], element_shape=[])
    l4 = list_ops.tensor_list_from_tensor([7.0, 8.0], element_shape=[])
    a = list_ops.empty_tensor_list(
        element_dtype=dtypes.variant, element_shape=[])
    a = list_ops.tensor_list_push_back(a, l1)
    a = list_ops.tensor_list_push_back(a, l2)
    b = list_ops.empty_tensor_list(
        element_dtype=dtypes.variant, element_shape=[])
    b = list_ops.tensor_list_push_back(b, l3)
    b = list_ops.tensor_list_push_back(b, l4)
    result = math_ops.add_n((a, b))
    result_0 = list_ops.tensor_list_stack(
        list_ops.tensor_list_get_item(result, 0, element_dtype=dtypes.variant),
        element_dtype=dtypes.float32)
    result_1 = list_ops.tensor_list_stack(
        list_ops.tensor_list_get_item(result, 1, element_dtype=dtypes.variant),
        element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(result_0), [6., 8.])
    self.assertAllEqual(self.evaluate(result_1), [10., 12.])

  def testAddTensorListsFailsIfLeadingDimsMismatch(self):
    # [FUNCTION DOC]
    # Adds two reserved lists of lengths two and three. Expects an invalid-argument error that
    # lists with different lengths cannot be added.
    l1 = list_ops.tensor_list_reserve(
        element_shape=[], element_dtype=dtypes.float32, num_elements=2)
    l2 = list_ops.tensor_list_reserve(
        element_shape=[], element_dtype=dtypes.float32, num_elements=3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Trying to add two lists of tensors with different lengths"):
      l = math_ops.add_n([l1, l2])
      self.evaluate(list_ops.tensor_list_stack(l, element_dtype=dtypes.float32))

  @test_util.run_v1_only("Uses placeholders")
  def testSkipEagerAddTensorListsFailsIfElementShapesMismatch(self):
    # [FUNCTION DOC]
    # Graph-only (version one only). Reserves two lists whose element shapes come from placeholders
    # to avoid early shape inference, adds them, and feeds different shapes at run time. Expects an
    # invalid-argument error about incompatible element shapes.
    with self.cached_session() as sess:
      # Use placeholders instead of constant values for shapes to prevent TF's
      # shape inference from catching this early.
      l1_element_shape = array_ops.placeholder(dtype=dtypes.int32)
      l2_element_shape = array_ops.placeholder(dtype=dtypes.int32)
      l1 = list_ops.tensor_list_reserve(
          element_shape=l1_element_shape,
          element_dtype=dtypes.float32,
          num_elements=3)
      l2 = list_ops.tensor_list_reserve(
          element_shape=l2_element_shape,
          element_dtype=dtypes.float32,
          num_elements=3)
      l = math_ops.add_n([l1, l2])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "Trying to add two lists of tensors with incompatible element shapes"
      ):
        sess.run(
            list_ops.tensor_list_stack(l, element_dtype=dtypes.float32), {
                l1_element_shape: [],
                l2_element_shape: [2]
            })

  @test_util.run_deprecated_v1
  def testSkipEagerConcatShapeInference(self):
    # [FUNCTION DOC]
    # Graph-only. Checks the static shape produced by the list concat op for different element
    # shapes: unknown shape gives unknown rank, partially defined shapes keep known dimensions, and
    # the first dimension is always unknown.

    def BuildTensor(element_shape):
      # [FUNCTION DOC]
      # HELPER: builds an empty list with the given element shape and returns the concatenation of
      # it. Input: element shape. Returns the concatenated tensor (graph mode, for shape checks).
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32, element_shape=element_shape)
      return list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)

    self.assertIsNone(BuildTensor(None).shape.rank)
    self.assertAllEqual(BuildTensor([None, 2, 3]).shape.as_list(), [None, 2, 3])
    self.assertAllEqual(
        BuildTensor([None, 2, None]).shape.as_list(), [None, 2, None])
    self.assertAllEqual(BuildTensor([1, 2, 3]).shape.as_list(), [None, 2, 3])

  def testConcatWithFullyDefinedElementShape(self):
    # [FUNCTION DOC]
    # Concatenates a list of two two by two elements and expects a four by two result in order.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[2, 2])
    l = list_ops.tensor_list_push_back(l, [[0., 1.], [2., 3.]])
    l = list_ops.tensor_list_push_back(l, [[4., 5.], [6., 7.]])
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual(
        self.evaluate(t), [[0., 1.], [2., 3.], [4., 5.], [6., 7.]])

  def testConcatWithNonFullyDefinedElementShape(self):
    # [FUNCTION DOC]
    # Concatenates two elements with different first dimensions but a fixed second dimension and
    # expects the rows joined in order.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[None, 2])
    l = list_ops.tensor_list_push_back(l, [[0., 1.]])
    l = list_ops.tensor_list_push_back(l, [[2., 3.], [4., 5.]])
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [[0., 1.], [2., 3.], [4., 5.]])

  def testConcatWithMismatchingTensorShapesFails(self):
    # [FUNCTION DOC]
    # Concatenates elements whose trailing dimensions differ; expects an invalid-argument error about
    # incompatible shapes during merge.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=None)
    l = list_ops.tensor_list_push_back(l, [[0., 1.]])
    l = list_ops.tensor_list_push_back(l, [[2.], [4.]])
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, r"Incompatible shapes during merge: "
        r"\[2\] vs. \[1\]"):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatEmptyListWithFullyDefinedElementShape(self):
    # [FUNCTION DOC]
    # Concatenating an empty list whose trailing dimension is known (first dimension fixed or
    # unknown) yields a tensor of shape zero by two.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[5, 2])
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t).shape, (0, 2))
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[None, 2])
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t).shape, (0, 2))

  def testConcatEmptyListWithUnknownElementShapeFails(self):
    # [FUNCTION DOC]
    # Concatenating an empty list with unknown element shape must raise an invalid-argument error
    # saying all but the first dimension must be fully defined.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=None)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "All except the first dimension must be fully"
        " defined when concating an empty tensor list"):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatEmptyListWithPartiallyDefinedElementShapeFails(self):
    # [FUNCTION DOC]
    # Concatenating an empty list whose trailing dimension is unknown must raise the same
    # invalid-argument error about fully defined dimensions.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=[2, None])
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "All except the first dimension must be fully"
        " defined when concating an empty tensor list"):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatListWithScalarElementShapeFails(self):
    # [FUNCTION DOC]
    # Concatenating a list whose element shape is scalar must raise an invalid-argument error that
    # elements must be at least vectors.
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32,
        element_shape=tensor_shape.TensorShape([]))
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Concat requires elements to be at least vectors, "
        "found scalars instead"):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatListWithScalarElementsFails(self):
    # [FUNCTION DOC]
    # Concatenating lists containing a scalar must raise an invalid-argument error naming the index of
    # the scalar (index zero for a list of one scalar, index one when the scalar comes after a vector).
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=None)
    l1 = list_ops.tensor_list_push_back(l, 1.)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "Concat saw a scalar shape at index 0"
        " but requires at least vectors"):
      t = list_ops.tensor_list_concat(l1, element_dtype=dtypes.float32)
      self.evaluate(t)
    l1 = list_ops.tensor_list_push_back(l, [1.])
    l1 = list_ops.tensor_list_push_back(l1, 2.)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "Concat saw a scalar shape at index 1"
        " but requires at least vectors"):
      t = list_ops.tensor_list_concat(l1, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatWithUninitializedTensorsUseListElementShape(self):
    # [FUNCTION DOC]
    # Concatenates a reserved list of three with element shape two by three; expects zeros of shape
    # six by three.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[2, 3], num_elements=3)
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual(np.zeros((6, 3)), t)

  def testConcatWithUninitializedTensorsUseProvidedElementShape(self):
    # [FUNCTION DOC]
    # Concatenates a reserved list with unknown element shape using a provided element shape of two by
    # three. Expects zeros of shape six by three.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    t = list_ops.tensor_list_concat(
        l, element_dtype=dtypes.float32, element_shape=(2, 3))
    self.assertAllEqual(np.zeros((6, 3)), t)

  def testConcatWithUninitializedTensorsUseProvidedElementShapeAndLengths(self):
    # [FUNCTION DOC]
    # Uses the raw second-version concat op with an element shape with unknown first dimension and an
    # explicit list of leading dimensions. Unset elements become zeros with those leading sizes (a
    # ten by three zeros result). After setting the middle element, unset elements before and after it
    # use the given leading sizes, giving a nine by two result.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    t, _ = gen_list_ops.tensor_list_concat_v2(
        l,
        element_dtype=dtypes.float32,
        element_shape=list_ops._build_element_shape((None, 3)),
        leading_dims=[2, 3, 5])
    self.assertAllEqual(np.zeros((10, 3)), t)
    l = list_ops.tensor_list_set_item(l, 1, [[2., 3.], [4., 5.], [6., 7.]])
    t, _ = gen_list_ops.tensor_list_concat_v2(
        l,
        element_dtype=dtypes.float32,
        element_shape=list_ops._build_element_shape((None, 2)),
        leading_dims=[2, 3, 4])
    self.assertAllEqual([[0., 0.], [0., 0.], [2., 3.], [4., 5.], [6., 7.],
                         [0., 0.], [0., 0.], [0., 0.], [0., 0.]], t)

  def testConcatWithUninitializedTensorsInferShapeFromElements(self):
    # [FUNCTION DOC]
    # Reserves a list with unknown element shape, sets only the middle element to a three by two
    # matrix, and concatenates. Unset elements become zeros of the inferred shape.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    l = list_ops.tensor_list_set_item(l, 1, [[2., 3.], [4., 5.], [6., 7.]])
    t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
    self.assertAllEqual([[0., 0.], [0., 0.], [0., 0.], [2., 3.], [4., 5.],
                         [6., 7.], [0., 0.], [0., 0.], [0., 0.]], t)

  def testConcatWithUninitializedTensorsFailsIfNoElementShape(self):
    # [FUNCTION DOC]
    # Concatenating a fully unset list with unknown element shape must raise an invalid-argument error
    # that the element shape except the first dimension is not fully defined.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=None, num_elements=3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Trying to concat list with only uninitialized tensors "
        r"but element_shape_except_first_dim is not fully defined"):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  def testConcatWithUninitializedTensorsFailsIfNoInputLengths(self):
    # [FUNCTION DOC]
    # Concatenating a fully unset list with partially defined element shape and no leading dimension
    # lengths must raise an invalid-argument error that leading dimensions have too few entries.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[None, 3], num_elements=3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"List contains uninitialized tensor at index 0"
        r" but leading_dims has only 0 elements."):
      t = list_ops.tensor_list_concat(l, element_dtype=dtypes.float32)
      self.evaluate(t)

  @test_util.run_in_graph_and_eager_modes
  def testConcatWithInvalidElementShape(self):
    # [FUNCTION DOC]
    # Runs in both modes. Calls the raw concat op with an empty element shape. Expects a value error or
    # invalid-argument error saying the element shape must not be empty.
    l = list_ops.tensor_list_reserve(
        element_dtype=dtypes.float32, element_shape=[], num_elements=0)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r"element_shape must not be empty"):
      self.evaluate(gen_list_ops.tensor_list_concat(
          input_handle=l, element_dtype=dtypes.float32, element_shape=[]))

  def testEmptyTensorListInvalidShape(self):
    # [FUNCTION DOC]
    # Creates an empty list with an element shape tensor of rank two. Expects a value error or
    # invalid-argument error that the shape must be at most rank one.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r"Shape must be at most rank 1 but is rank 2"):
      t = gen_list_ops.EmptyTensorList(
          element_shape=array_ops.ones(dtype=dtypes.int32, shape=[1, 0]),
          max_num_elements=constant_op.constant(1),
          element_dtype=dtypes.int32)
      self.evaluate(t)

  def testEvenSplit(self):
    # [FUNCTION DOC]
    # Splits tensors into lists with equal-length pieces and stacks each, checking the result through
    # a local helper for three input cases (scalars, vectors, and matrices).

    def RunTest(input_tensor, lengths, expected_stacked_output):
      # [FUNCTION DOC]
      # HELPER: local checker for the even-split test. Inputs: tensor to split, list of piece lengths,
      # expected stacked result. Splits with unknown element shape, stacks as float and asserts equality.
      l = list_ops.tensor_list_split(
          input_tensor, element_shape=None, lengths=lengths)
      self.assertAllEqual(
          list_ops.tensor_list_stack(l, element_dtype=dtypes.float32),
          expected_stacked_output)

    RunTest([1., 2., 3.], [1, 1, 1], [[1.], [2.], [3.]])
    RunTest([1., 2., 3., 4.], [2, 2], [[1., 2.], [3., 4.]])
    RunTest([[1., 2.], [3., 4.]], [1, 1], [[[1., 2.]], [[3., 4.]]])

  def testUnevenSplit(self):
    # [FUNCTION DOC]
    # Splits five values into pieces of lengths three and two. Expects a list of length two whose
    # elements are the first three and the last two values.
    l = list_ops.tensor_list_split([1., 2., 3., 4., 5],
                                   element_shape=None,
                                   lengths=[3, 2])
    self.assertAllEqual(list_ops.tensor_list_length(l), 2)
    self.assertAllEqual(
        list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32),
        [1., 2., 3.])
    self.assertAllEqual(
        list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32),
        [4., 5.])

  @test_util.run_deprecated_v1
  def testSkipEagerSplitWithInvalidTensorShapeFails(self):
    # [FUNCTION DOC]
    # Graph-only. Splits a placeholder and feeds a scalar. Expects an invalid-argument error that the
    # tensor must be at least a vector.
    with self.cached_session():
      tensor = array_ops.placeholder(dtype=dtypes.float32)
      l = list_ops.tensor_list_split(tensor, element_shape=None, lengths=[1])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"Tensor must be at least a vector, but saw shape: \[\]"):
        l.eval({tensor: 1})

  @test_util.run_deprecated_v1
  def testSkipEagerSplitWithInvalidLengthsShapeFails(self):
    # [FUNCTION DOC]
    # Graph-only. Uses a placeholder for lengths and feeds a scalar. Expects an invalid-argument
    # error that lengths must be a vector.
    with self.cached_session():
      lengths = array_ops.placeholder(dtype=dtypes.int64)
      l = list_ops.tensor_list_split([1., 2.],
                                     element_shape=None,
                                     lengths=lengths)
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"Expected lengths to be a vector, received shape: \[\]"):
        l.eval({lengths: 1})

  def testSplitWithInvalidLengthsFails(self):
    # [FUNCTION DOC]
    # Three failing splits: a negative length, a length beyond the tensor size, and lengths that leave
    # values unused. Each must raise an invalid-argument error with the corresponding message.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r"Invalid value in lengths: -1"):
      l = list_ops.tensor_list_split([1., 2.],
                                     element_shape=None,
                                     lengths=[1, -1])
      self.evaluate(l)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Attempting to slice \[0, 3\] from tensor with length 2"):
      l = list_ops.tensor_list_split([1., 2.], element_shape=None, lengths=[3])
      self.evaluate(l)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Unused values in tensor. Length of tensor: 2 Values used: 1"):
      l = list_ops.tensor_list_split([1., 2.], element_shape=None, lengths=[1])
      self.evaluate(l)

  @test_util.run_deprecated_v1
  def testSkipEagerSplitWithScalarElementShapeFails(self):
    # [FUNCTION DOC]
    # Graph-only. A scalar static element shape raises a value error about rank mismatch at graph
    # construction. A scalar element shape fed through a placeholder at run time raises an
    # invalid-argument error that element shape must have rank at least one.
    with self.assertRaisesRegex(ValueError,
                                r"Shapes must be equal rank, but are 1 and 0"):
      l = list_ops.tensor_list_split([1., 2.], element_shape=[], lengths=[1, 1])
    with self.cached_session():
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"TensorListSplit requires element_shape to be at least of rank 1, "
          r"but saw: \[\]"):
        element_shape = array_ops.placeholder(dtype=dtypes.int32)
        l = list_ops.tensor_list_split([1., 2.],
                                       element_shape=element_shape,
                                       lengths=[1, 1])
        l.eval({element_shape: []})

  def testEagerOnlySplitWithScalarElementShapeFails(self):
    # [FUNCTION DOC]
    # Only acts in eager mode (does nothing in graph mode). Splitting with a scalar element shape
    # must raise an invalid-argument error that the element shape needs rank at least one.
    if context.executing_eagerly():
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"TensorListSplit requires element_shape to be at least of rank 1, "
          r"but saw: \[\]"):
        list_ops.tensor_list_split([1., 2.], element_shape=[], lengths=[1, 1])

  @test_util.run_deprecated_v1
  def testSkipEagerSplitWithIncompatibleTensorShapeAndElementShapeFails(self):
    # [FUNCTION DOC]
    # Graph-only. A static element shape of the wrong rank raises a value error at graph construction;
    # an element shape supplied at run time that is incompatible with the tensor raises an
    # invalid-argument error about incompatible shapes.
    with self.assertRaisesRegex(ValueError,
                                r"Shapes must be equal rank, but are 2 and 1"):
      l = list_ops.tensor_list_split([[1.], [2.]],
                                     element_shape=[1],
                                     lengths=[1, 1])

    with self.cached_session():
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"tensor shape \[2,1\] is not compatible with element_shape \[1\]"):
        element_shape = array_ops.placeholder(dtype=dtypes.int32)
        l = list_ops.tensor_list_split([[1.], [2.]],
                                       element_shape=element_shape,
                                       lengths=[1, 1])
        l.eval({element_shape: [1]})

  def testEagerOnlySplitWithIncompatibleTensorShapeAndElementShapeFails(self):
    # [FUNCTION DOC]
    # Only acts in eager mode. Splitting a two by one tensor with an incompatible element shape must
    # raise an invalid-argument error about the shapes being incompatible.
    if context.executing_eagerly():
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"tensor shape \[2,1\] is not compatible with element_shape \[1\]"):
        list_ops.tensor_list_split([[1.], [2.]],
                                   element_shape=[1],
                                   lengths=[1, 1])

  def testResizeGrow(self):
    # [FUNCTION DOC]
    # Grows a two-element list to four. Expects length four and the original elements preserved.
    l = list_ops.tensor_list_from_tensor([1., 2.], element_shape=[])
    l = list_ops.tensor_list_resize(l, 4)
    self.assertEqual(self.evaluate(list_ops.tensor_list_length(l)), 4)
    self.assertEqual(
        self.evaluate(
            list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)),
        1.)
    self.assertEqual(
        self.evaluate(
            list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)),
        2.)

  def testResizeShrink(self):
    # [FUNCTION DOC]
    # Shrinks a three-element list to two. Expects length two and the first two values retained.
    l = list_ops.tensor_list_from_tensor([1., 2., 3.], element_shape=[])
    l = list_ops.tensor_list_resize(l, 2)
    self.assertEqual(self.evaluate(list_ops.tensor_list_length(l)), 2)
    self.assertAllEqual(
        self.evaluate(
            list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)),
        [1., 2.])

  def testResizeWithInvalidSizeFails(self):
    # [FUNCTION DOC]
    # Resizing to a negative size must raise an invalid-argument error that the size must be
    # non-negative.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "TensorListSlice expects size to be non-negative"):
      l = list_ops.tensor_list_from_tensor([1., 2., 3.], element_shape=[])
      l = list_ops.tensor_list_resize(l, -1)
      self.evaluate(l)

  @test_util.run_in_graph_and_eager_modes
  def testResizeWithNonScalarFails(self):
    # [FUNCTION DOC]
    # Calls the raw resize op with a non-scalar size. Expects a value error or invalid-argument error
    # that the size must be a scalar or have rank zero.
    l = list_ops.tensor_list_from_tensor([3, 4, 5], element_shape=[])
    size = np.zeros([0, 2, 3, 3])
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r"Shape must be rank 0 but is rank \d+|"
                                r"\w+ must be a scalar"):
      self.evaluate(gen_list_ops.TensorListResize(input_handle=l, size=size))

  @test_util.run_deprecated_v1
  @test_util.enable_control_flow_v2
  def testSkipEagerResizeGrad(self):
    # [FUNCTION DOC]
    # Graph-only with control flow version two. Sets an element beyond the list end with resizing
    # allowed, stacks, and takes the gradient with respect to the source tensor. Expects all ones.
    t = constant_op.constant([1., 2., 3.])
    l = list_ops.tensor_list_from_tensor(t, element_shape=[])
    l = list_ops.tensor_list_set_item(
        l, 3, 4., resize_if_index_out_of_bounds=True)
    t1 = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    grad = gradients_impl.gradients(t1, t)[0]
    self.assertAllEqual(self.evaluate(grad), [1., 1., 1.])

  def testHandleDataAcrossFunctionCall(self):
    # [FUNCTION DOC]
    # Builds a list inside a compiled function and returns it. Checks that handle data (type
    # information) is set both inside the function and on the returned list outside, with the float
    # dtype and the array full type, and that the element read has a scalar shape.

    @def_function.function
    def func():
      # [FUNCTION DOC]
      # HELPER: compiled function that builds a list from a vector, asserts handle data is set with the
      # array full type, and returns the list.
      t = constant_op.constant([1., 2., 3.])
      l = list_ops.tensor_list_from_tensor(t, element_shape=[])
      handle_data = resource_variable_ops.get_eager_safe_handle_data(l)
      self.assertTrue(handle_data.is_set)
      self.assertEqual(handle_data.shape_and_type[0].type.type_id,
                       full_type_pb2.TFT_ARRAY)
      return l

    tensor_list = func()
    handle_data = resource_variable_ops.get_eager_safe_handle_data(tensor_list)
    self.assertTrue(handle_data.is_set)
    self.assertEqual(dtypes.float32, handle_data.shape_and_type[0].dtype)
    self.assertEqual(handle_data.shape_and_type[0].type.type_id,
                     full_type_pb2.TFT_ARRAY)
    element = list_ops.tensor_list_get_item(
        tensor_list, 0, element_dtype=dtypes.float32)
    self.assertAllEqual(element.shape.as_list(), [])

  @test_util.run_gpu_only
  def testNestedListDevicetoDeviceCopy(self):
    # [FUNCTION DOC]
    # GPU-only; skipped when fewer than two GPUs exist. Builds a nested list on the first GPU, copies
    # it back and forth between two GPUs one thousand twenty-four times, then pops the inner list on
    # the second GPU and checks it still holds the original values.
    if context.num_gpus() < 2:
      self.skipTest("Need at least 2 GPUs for this test, found %d" %
                    context.num_gpus())
    with ops.device("gpu:0"):
      t = constant_op.constant([1.0, 2.0, 3.0])
      inner_l = list_ops.tensor_list_from_tensor(t, element_shape=[])
      outer_l = list_ops.empty_tensor_list(
          element_dtype=dtypes.variant, element_shape=[])
      outer_l = list_ops.tensor_list_push_back(outer_l, inner_l)

    # Stress test.
    for _ in range(1024):
      with ops.device("gpu:1"):
        outer_l = array_ops.identity(outer_l)
      with ops.device("gpu:0"):
        outer_l = array_ops.identity(outer_l)

    with ops.device("gpu:1"):
      _, inner_l = list_ops.tensor_list_pop_back(
          outer_l, element_dtype=dtypes.variant)
      t = list_ops.tensor_list_stack(inner_l, element_dtype=dtypes.float32)
      self.assertAllEqual(t, [1.0, 2.0, 3.0])

  def testTensorListStrings(self):
    # [FUNCTION DOC]
    # Maps a string upper-casing function over a string vector inside a compiled function (which
    # internally uses lists) and expects the upper-cased byte strings.
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: compiled function mapping the upper-case string op over three strings; returns results.
      return map_fn.map_fn(string_ops.string_upper,
                           constant_op.constant(["a", "b", "c"]))

    self.assertAllEqual(f(), [b"A", b"B", b"C"])

  def testTensorListStringsNoInline(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is unconditionally skipped on its first statement (bug reference
    # 150742232), so everything after the skip call is dead code and never runs.
    # Intended scenario: a no-inline generator function returns a list built from a string vector, and
    # another function maps an upper-casing op over indexes reading that list; expects upper-cased
    # strings.
    # Generator function output type is a variant with a host-only underlying
    # data type. "ColocationGraph::AddHostOnlyDataTypesConstraints" needs to
    # have "deep op inspection" to be able to correctly place the while loop
    # generated from map_fn.
    self.skipTest("b/150742232")

    @def_function.function(experimental_attributes={"_noinline": True})
    def generator(c):
      # [FUNCTION DOC]
      # HELPER: compiled function (marked no-inline) that builds a list from a string tensor. Input: the
      # tensor. Returns the list. Never executed since the enclosing test is skipped.
      return list_ops.tensor_list_from_tensor(c, element_shape=[])

    @def_function.function
    def f(c):
      # [FUNCTION DOC]
      # HELPER: compiled function that gets the list from the generator and maps the upper helper over
      # indexes zero to two as strings. Input: string tensor. Returns the mapped strings.
      l = generator(c)

      def upper(i):
        # [FUNCTION DOC]
        # HELPER: reads a string element from the captured list at the given index and returns it
        # upper-cased. Never executed since the enclosing test is skipped.
        e = list_ops.tensor_list_get_item(l, i, element_dtype=dtypes.string)
        return string_ops.string_upper(e)

      return map_fn.map_fn(
          upper, constant_op.constant([0, 1, 2]), dtype=dtypes.string)

    c = constant_op.constant(["a", "b", "c"])
    self.assertAllEqual(f(c), [b"A", b"B", b"C"])

  def testPopBackGrad(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name refers to pop-back, but no list pop is used; the test is a regression test
    # for a TensorFlow issue (37230) on second derivatives through a loop inside a compiled function.
    # Computes the Hessian of a function multiplying its input three times via nested gradient tapes
    # and expects six.
    # https://github.com/tensorflow/tensorflow/issues/37230

    @def_function.function
    def g(x):
      # [FUNCTION DOC]
      # HELPER: compiled function that multiplies a vector accumulator by the input three times in a
      # loop and returns it. Input: scalar. Returns the product.
      x_prod = constant_op.constant([1.])
      for unused_i in math_ops.range(3):
        x_prod = x_prod * x
      return x_prod

    x = constant_op.constant(1.)
    with backprop.GradientTape() as t:
      t.watch(x)
      with backprop.GradientTape() as tt:
        tt.watch(x)
        loss = g(x)
      jac = tt.gradient(loss, x)
    hess = t.gradient(jac, x)
    self.assertAllEqual(hess, 6.)

  def testTensorListElementShapeShapeInference(self):
    # [FUNCTION DOC]
    # Inside a compiled function, creates a list with unknown element shape, reads its element shape
    # and checks the static rank is unknown. Pushes that shape tensor onto an int32 list and pops it.
    # Expects negative one as the final result.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: compiled function that creates a list with unknown shape, asserts the element shape
      # tensor has unknown rank, round-trips it through another list and returns the popped value.
      l = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32, element_shape=None)
      l_element_shape = list_ops.tensor_list_element_shape(l, dtypes.int32)
      self.assertIsNone(l_element_shape.shape.rank)
      shape_l = list_ops.empty_tensor_list(
          element_dtype=dtypes.int32, element_shape=l_element_shape.shape)
      shape_l = list_ops.tensor_list_push_back(shape_l, l_element_shape)
      return list_ops.tensor_list_pop_back(shape_l, dtypes.int32)[1]

    self.assertAllEqual(f(), -1)

  def testElementShapeArgOfTensorListFromTensor(self):
    # [FUNCTION DOC]
    # Inside a compiled function, builds a list from a three by three tensor with an element shape
    # of one unknown dimension, pushes a four-element vector, reads index three and checks the static
    # shape has an unknown dimension. Expects the four ones.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: compiled function that builds a list with an unknown-length element shape, pushes a
      # longer vector, reads it back, asserts its static shape is unknown-length and returns it.
      t = array_ops.ones([3, 3])
      l = list_ops.tensor_list_from_tensor(t, element_shape=[-1])
      l = list_ops.tensor_list_push_back(l, array_ops.ones([4]))
      read_val = list_ops.tensor_list_get_item(
          l, 3, element_dtype=dtypes.float32)
      self.assertAllEqual(read_val.shape.as_list(), [None])
      return read_val

    self.assertAllEqual(f(), [1.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
  test.main()
