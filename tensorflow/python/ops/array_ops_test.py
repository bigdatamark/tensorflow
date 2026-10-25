# [MODULE DOC]
# Unit tests for selected array operations: static shape inference through gather, reshape and
# slicing of shapes inside traced functions, the empty meshgrid case, regression tests for
# rank-mismatch errors in the parallel concat, lower bound and upper bound kernels (formerly
# crashes), and the default 32-bit dtype of shape results when the int64 shape flag is off.
# Complements the shape-int64 test module. No import-time side effects; running as a script
# starts the standard test runner.
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
"""Tests for array operations."""

from tensorflow.core.config import flags
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ArrayOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case grouping the array operation regression and shape-inference tests of this module.
  # Has no shared fixtures; every test builds its own tensors, and several run in both graph and
  # eager modes through the test utility decorator.

  def testGatherGradHasPartialStaticShape(self):
    # [FUNCTION DOC]
    # Test: builds a random 4x10x10 tensor and gathers a data-dependent subset along axis 1 so
    # that dimension becomes unknown. Under a gradient tape, takes the gradient of a double gather
    # along the first axis and of a gather along the last-but-one axis, and asserts the gradient
    # shapes keep the partial static shape (unknown dims where expected, known dims preserved).
    # Takes no inputs, returns nothing.
    # Create a tensor with an unknown dim 1.
    x = random_ops.random_normal([4, 10, 10])
    x = array_ops.gather(
        x, array_ops.reshape(array_ops.where_v2(x[0, :, 0] > 0.5), [-1]), axis=1
    )
    x.shape.assert_is_compatible_with([4, None, 10])

    with backprop.GradientTape() as tape:
      tape.watch(x)
      a = array_ops.gather(array_ops.gather(x, [0, 1]), [0, 1])
    grad_a = tape.gradient(a, x)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      b = array_ops.gather(array_ops.gather(x, [2, 3], axis=2), [0, 1])
    grad_b = tape.gradient(b, x)

    # We make sure that the representation of the shapes are correct; the shape
    # equality check will always eval to false due to the shapes being partial.
    grad_a.shape.assert_is_compatible_with([None, None, 10])
    grad_b.shape.assert_is_compatible_with([4, None, 10])

  def testReshapeShapeInference(self):
    # [FUNCTION DOC]
    # Test: creates a tensor with an unknown second dimension via a data-dependent gather, then
    # reshapes it to its own dynamic shape (as 32-bit and as 64-bit integers) and asserts the
    # static shape [4, unknown, 10] is preserved. Also asserts that a shape routed through a cast
    # to float and back is not used for inference, so the result is fully unknown in each
    # dimension. Takes no inputs, returns nothing.
    # Create a tensor with an unknown dim 1.
    x = random_ops.random_normal([4, 10, 10])
    x = array_ops.gather(
        x, array_ops.reshape(array_ops.where_v2(x[0, :, 0] > 0.5), [-1]), axis=1
    )
    x.shape.assert_is_compatible_with([4, None, 10])
    a = array_ops.reshape(x, array_ops.shape(x))
    a.shape.assert_is_compatible_with([4, None, 10])
    b = array_ops.reshape(x, math_ops.cast(array_ops.shape(x), dtypes.int64))
    b.shape.assert_is_compatible_with([4, None, 10])

    # We do not shape-infer across a tf.cast into anything that's not tf.int32
    # or tf.int64, since they might end up mangling the shape.
    c = array_ops.reshape(
        x,
        math_ops.cast(
            math_ops.cast(array_ops.shape(x), dtypes.float32), dtypes.int32
        ),
    )
    c.shape.assert_is_compatible_with([None, None, None])

  def testEmptyMeshgrid(self):
    # [FUNCTION DOC]
    # Test: calls meshgrid with no inputs and asserts it returns an empty list. No inputs, no
    # return value.
    self.assertEqual(array_ops.meshgrid(), [])

  def testSlicedPartialShapeInference(self):
    # [FUNCTION DOC]
    # Test: defines a traced function (autograph off) that builds a zeros tensor whose shape is the
    # first dimension of the input's dynamic shape, obtains its concrete function for an input
    # spec of shape 10 by unknown, and asserts the output static shape is [10]. Takes no inputs,
    # returns nothing.
    @def_function.function(autograph=False)
    def g(x):
      # [FUNCTION DOC]
      # HELPER: traced function used by the enclosing shape-inference test.
      # Takes a tensor x and returns a zeros vector whose length is the first entry of x's
      # dynamic shape; no side effects.
      return array_ops.zeros([array_ops.shape(x)[0]])

    conc = g.get_concrete_function(tensor_spec.TensorSpec([10, None]))
    self.assertAllEqual(conc.output_shapes.as_list(), [10])

  def testIdentityOnSlicedPartialShapeInference(self):
    # [FUNCTION DOC]
    # Test: like the sliced partial shape test, but the sliced dimension value is passed through an
    # identity operation before being used as the zeros shape. Asserts the concrete function's
    # output static shape is still [10]. Takes no inputs, returns nothing.
    @def_function.function(autograph=False)
    def g(x):
      # [FUNCTION DOC]
      # HELPER: traced function used by the enclosing shape-inference test.
      # Takes a tensor x and returns a zeros vector whose length is the first entry of x's
      # dynamic shape passed through identity; no side effects.
      return array_ops.zeros([array_ops.identity(array_ops.shape(x)[0])])

    conc = g.get_concrete_function(tensor_spec.TensorSpec([10, None]))
    self.assertAllEqual(conc.output_shapes.as_list(), [10])

  @test_util.run_in_graph_and_eager_modes
  def testParallelConcatFailsWithRankZeroShape(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): calls the parallel concat operation with a rank-zero shape
    # argument and a one-element value list, and asserts it raises an error whose message says the
    # rank or dimension must be greater than zero. Takes no inputs, returns nothing.
    op = array_ops.ParallelConcat
    para = {"shape": 0, "values": [1]}

    def func():
      # [FUNCTION DOC]
      # HELPER: wrapper invoking the parallel concat operation with the prepared arguments so the
      # enclosing test can assert on the raised error. Returns the operation result if it succeeds.
      y = op(**para)
      return y

    with self.assertRaisesRegex(
        Exception, "(rank|dimension) of .* must be greater than .* 0"
    ):
      func()

  @test_util.run_in_graph_and_eager_modes
  def testUpperBoundValuesWrongRank(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): regression for a former crash. Passes sorted inputs of rank 2
    # and a values tensor of rank 3 to the upper bound kernel and asserts an error stating the
    # shape must be rank 2 but is rank 3. Takes no inputs, returns nothing.
    # Used to cause a segfault, b/266336058
    arg0 = array_ops.zeros([2, 3], dtype=dtypes.float32)
    arg1 = array_ops.zeros([2, 1, 0], dtype=dtypes.float32)
    with self.assertRaisesRegex(
        Exception, "Shape must be rank 2 but is rank 3"
    ):
      gen_array_ops.upper_bound(arg0, arg1)

  def testLowerBoundValuesWrongRank(self):
    # [FUNCTION DOC]
    # Test: regression for a former crash. Passes sorted inputs of rank 2 and a values tensor of
    # rank 3 to the lower bound kernel and asserts an error stating the shape must be rank 2 but
    # is rank 3. Takes no inputs, returns nothing.
    # Used to cause a segfault, b/266336058
    arg0 = array_ops.zeros([2, 3], dtype=dtypes.float32)
    arg1 = array_ops.zeros([2, 1, 0], dtype=dtypes.float32)
    with self.assertRaisesRegex(
        Exception, "Shape must be rank 2 but is rank 3"
    ):
      gen_array_ops.lower_bound(arg0, arg1)

  def testUpperBoundInputsWrongRank(self):
    # [FUNCTION DOC]
    # Test: regression for a former crash. Passes a rank 3 sorted-inputs tensor and rank 2 values
    # to the upper bound kernel and asserts an error stating the shape must be rank 2 but is
    # rank 3. Takes no inputs, returns nothing.
    # Used to cause a segfault, b/266336058
    arg0 = array_ops.zeros([2, 1, 0], dtype=dtypes.float32)
    arg1 = array_ops.zeros([2, 3], dtype=dtypes.float32)
    with self.assertRaisesRegex(
        Exception, "Shape must be rank 2 but is rank 3"
    ):
      gen_array_ops.upper_bound(arg0, arg1)

  def testLowerBoundInputsWrongRank(self):
    # [FUNCTION DOC]
    # Test: regression for a former crash. Passes a rank 3 sorted-inputs tensor and rank 2 values
    # to the lower bound kernel and asserts an error stating the shape must be rank 2 but is
    # rank 3. Takes no inputs, returns nothing.
    # Used to cause a segfault, b/266336058
    arg0 = array_ops.zeros([2, 1, 0], dtype=dtypes.float32)
    arg1 = array_ops.zeros([2, 3], dtype=dtypes.float32)
    with self.assertRaisesRegex(
        Exception, "Shape must be rank 2 but is rank 3"
    ):
      gen_array_ops.lower_bound(arg0, arg1)

  def testShapeDefaultIn32(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says "In32" but means 32-bit integer (int32); likely a typo for Int32.
    # Test: asserts the shape-default-int64 flag is NOT enabled in this test's run, then checks
    # the version 2 shape operation on a 1 by 2 zeros tensor yields 32-bit integer dtype. Takes no
    # inputs, returns nothing.
    # The tf_shape_default_int64 flag should NOT be set when this test runs
    self.assertFalse(flags.config().tf_shape_default_int64.value())
    s1 = array_ops.shape_v2(array_ops.zeros([1, 2]))
    self.assertEqual(s1.dtype, dtypes.int32)


if __name__ == "__main__":
  test.main()
