# [MODULE DOC]
# Unit tests for data_flow_ops.dynamic_stitch and parallel_dynamic_stitch (the gradient module is
# imported for registration). A shared mixin class defines the common scenarios and takes the
# stitch function as a constructor argument; two concrete test cases bind it to the sequential and
# the parallel stitch ops. Covers scalars, vectors, matrices, empty inputs, higher ranks with
# gradients, shape errors and out-of-range indices. Not a helper module.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.data_flow_ops.{,parallel_}dynamic_stitch."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gradients_impl
import tensorflow.python.ops.data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class DynamicStitchTestBase(object):
  # [CLASS DOC]
  # Mixin (plain object, not a test case) holding the scenarios shared by both stitch ops. It
  # relies on the test-case base of the concrete subclass for assertions and for evaluation, and
  # stores the op under test in an attribute set by its constructor. Not collected on its own.

  def __init__(self, stitch_op):
    # [FUNCTION DOC]
    # Constructor: stores the stitch function to test as the attribute holding the op. Input: a
    # callable taking a list of index tensors and a list of data tensors. Returns nothing.
    self.stitch_op = stitch_op

  def testScalar(self):
    # [FUNCTION DOC]
    # Test: stitches two scalar data values with scalar indices 0 and 1, in forward and reversed
    # order, on the GPU if available; checks the output values follow the index order and the static
    # shape is a vector of 2.
    with test_util.use_gpu():
      indices = [constant_op.constant(0), constant_op.constant(1)]
      data = [constant_op.constant(40), constant_op.constant(60)]
      for step in -1, 1:
        stitched_t = self.stitch_op(indices[::step], data)
        stitched_val = self.evaluate(stitched_t)
        self.assertAllEqual([40, 60][::step], stitched_val)
        # Dimension 0 is max(flatten(indices))+1.
        self.assertEqual([2], stitched_t.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testShapeInferenceForScalarWithNonConstantIndices(self):
    # [FUNCTION DOC]
    # Test (graph mode): one index is a placeholder, so the output length cannot be inferred; the
    # static shape must be a vector of unknown length, for both orders of the inputs.
    with test_util.use_gpu():
      indices = [
          array_ops.placeholder(dtype=dtypes.int32),
          constant_op.constant(1)
      ]
      data = [constant_op.constant(40), constant_op.constant(60)]
      for step in -1, 1:
        stitched_t = self.stitch_op(indices[::step], data)
        # Dimension 0 is max(flatten(indices))+1, but the first indices input is
        # not a constant tensor, so we can only infer it as a vector of unknown
        # length.
        self.assertEqual([None], stitched_t.get_shape().as_list())

  @test_util.disable_tfrt("b/169901260")
  def testSimpleOneDimensional(self):
    # [FUNCTION DOC]
    # Test (skipped under TFRT): for several float and quantized dtypes, stitches two vectors with
    # interleaved indices into an 8-element vector and checks values and static shape, verifying
    # the op is registered for each dtype.
    # Test various datatypes in the simple case to ensure that the op was
    # registered under those types.
    dtypes_to_test = [
        dtypes.float32,
        dtypes.float16,
        dtypes.bfloat16,
        dtypes.qint8,
        dtypes.quint8,
        dtypes.qint32,
    ]
    for dtype in dtypes_to_test:
      indices = [
          constant_op.constant([0, 4, 7]),
          constant_op.constant([1, 6, 2, 3, 5])
      ]
      data = [
          math_ops.cast(constant_op.constant([0, 40, 70]), dtype=dtype),
          math_ops.cast(
              constant_op.constant([10, 60, 20, 30, 50]), dtype=dtype)
      ]
      stitched_t = self.stitch_op(indices, data)
      stitched_val = self.evaluate(stitched_t)
      self.assertAllEqual([0, 10, 20, 30, 40, 50, 60, 70], stitched_val)
      # Dimension 0 is max(flatten(indices))+1.
      self.assertEqual([8], stitched_t.get_shape().as_list())

  def testOneListOneDimensional(self):
    # [FUNCTION DOC]
    # Test: a single index list and single data list that are shuffled permutations produce the
    # sorted 8-element output with static shape of 8.
    indices = [constant_op.constant([1, 6, 2, 3, 5, 0, 4, 7])]
    data = [constant_op.constant([10, 60, 20, 30, 50, 0, 40, 70])]
    stitched_t = self.stitch_op(indices, data)
    stitched_val = self.evaluate(stitched_t)
    self.assertAllEqual([0, 10, 20, 30, 40, 50, 60, 70], stitched_val)
    # Dimension 0 is max(flatten(indices))+1.
    self.assertEqual([8], stitched_t.get_shape().as_list())

  def testSimpleTwoDimensional(self):
    # [FUNCTION DOC]
    # Test: stitches three groups of rows (2-column) by interleaved indices into an 8x2 matrix and
    # checks values and static shape.
    indices = [
        constant_op.constant([0, 4, 7]),
        constant_op.constant([1, 6]),
        constant_op.constant([2, 3, 5])
    ]
    data = [
        constant_op.constant([[0, 1], [40, 41], [70, 71]]),
        constant_op.constant([[10, 11], [60, 61]]),
        constant_op.constant([[20, 21], [30, 31], [50, 51]])
    ]
    stitched_t = self.stitch_op(indices, data)
    stitched_val = self.evaluate(stitched_t)
    self.assertAllEqual([[0, 1], [10, 11], [20, 21], [30, 31], [40, 41],
                         [50, 51], [60, 61], [70, 71]], stitched_val)
    # Dimension 0 is max(flatten(indices))+1.
    self.assertEqual([8, 2], stitched_t.get_shape().as_list())

  def testZeroSizeTensor(self):
    # [FUNCTION DOC]
    # Test: like the two-dimensional case but with an extra empty index/data pair; result must be
    # unchanged 8x2.
    indices = [
        constant_op.constant([0, 4, 7]),
        constant_op.constant([1, 6]),
        constant_op.constant([2, 3, 5]),
        array_ops.zeros([0], dtype=dtypes.int32)
    ]
    data = [
        constant_op.constant([[0, 1], [40, 41], [70, 71]]),
        constant_op.constant([[10, 11], [60, 61]]),
        constant_op.constant([[20, 21], [30, 31], [50, 51]]),
        array_ops.zeros([0, 2], dtype=dtypes.int32)
    ]
    stitched_t = self.stitch_op(indices, data)
    stitched_val = self.evaluate(stitched_t)
    self.assertAllEqual([[0, 1], [10, 11], [20, 21], [30, 31], [40, 41],
                         [50, 51], [60, 61], [70, 71]], stitched_val)
    # Dimension 0 is max(flatten(indices))+1.
    self.assertEqual([8, 2], stitched_t.get_shape().as_list())

  def testAllZeroSizeTensor(self):
    # [FUNCTION DOC]
    # Test: all indices and data are empty; the output is an empty array with 0 rows and 2 columns
    # and a matching static shape.
    indices = [
        array_ops.zeros([0], dtype=dtypes.int32),
        array_ops.zeros([0], dtype=dtypes.int32)
    ]
    data = [
        array_ops.zeros([0, 2], dtype=dtypes.int32),
        array_ops.zeros([0, 2], dtype=dtypes.int32)
    ]
    stitched_t = self.stitch_op(indices, data)
    stitched_val = self.evaluate(stitched_t)
    self.assertAllEqual(np.zeros((0, 2)), stitched_val)
    self.assertEqual([0, 2], stitched_t.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testHigherRank(self):
    # [FUNCTION DOC]
    # Test (graph mode): indices of rank 0, 1 and 2 with matching data; checks the 7x2 output and
    # gradients: index gradients are nothing, and data gradients equal the incoming gradient (scaled
    # by 7) sliced back per input.
    indices = [
        constant_op.constant(6),
        constant_op.constant([4, 1]),
        constant_op.constant([[5, 2], [0, 3]])
    ]
    data = [
        constant_op.constant([61., 62.]),
        constant_op.constant([[41., 42.], [11., 12.]]),
        constant_op.constant([[[51., 52.], [21., 22.]],
                              [[1., 2.], [31., 32.]]])
    ]
    stitched_t = self.stitch_op(indices, data)
    stitched_val = self.evaluate(stitched_t)
    correct = 10. * np.arange(7)[:, None] + [1., 2.]
    self.assertAllEqual(correct, stitched_val)
    self.assertEqual([7, 2], stitched_t.get_shape().as_list())
    # Test gradients
    stitched_grad = 7. * stitched_val
    grads = gradients_impl.gradients(stitched_t, indices + data,
                                     stitched_grad)
    self.assertEqual(grads[:3], [None] * 3)  # Indices have no gradients
    for datum, grad in zip(data, self.evaluate(grads[3:])):
      self.assertAllEqual(7. * self.evaluate(datum), grad)

  @test_util.run_deprecated_v1
  def testErrorIndicesMultiDimensional(self):
    # [FUNCTION DOC]
    # Test (graph mode): a rank-2 index tensor with mismatched data ranks must raise a value error
    # when the op is built.
    indices = [
        constant_op.constant([0, 4, 7]),
        constant_op.constant([[1, 6, 2, 3, 5]])
    ]
    data = [
        constant_op.constant([[0, 40, 70]]),
        constant_op.constant([10, 60, 20, 30, 50])
    ]
    with self.assertRaises(ValueError):
      self.stitch_op(indices, data)

  @test_util.run_deprecated_v1
  def testErrorDataNumDimsMismatch(self):
    # [FUNCTION DOC]
    # Test (graph mode): data tensors with different ranks must raise a value error.
    indices = [
        constant_op.constant([0, 4, 7]),
        constant_op.constant([1, 6, 2, 3, 5])
    ]
    data = [
        constant_op.constant([0, 40, 70]),
        constant_op.constant([[10, 60, 20, 30, 50]])
    ]
    with self.assertRaises(ValueError):
      self.stitch_op(indices, data)

  @test_util.run_deprecated_v1
  def testErrorDataDimSizeMismatch(self):
    # [FUNCTION DOC]
    # Test (graph mode): data tensors whose trailing dimension sizes differ (1 vs 2 columns) must
    # raise a value error.
    indices = [
        constant_op.constant([0, 4, 5]),
        constant_op.constant([1, 6, 2, 3])
    ]
    data = [
        constant_op.constant([[0], [40], [70]]),
        constant_op.constant([[10, 11], [60, 61], [20, 21], [30, 31]])
    ]
    with self.assertRaises(ValueError):
      self.stitch_op(indices, data)

  @test_util.run_deprecated_v1
  def testErrorDataAndIndicesSizeMismatch(self):
    # [FUNCTION DOC]
    # Test (graph mode): data of length 4 paired with indices of length 5 must raise a value error.
    indices = [
        constant_op.constant([0, 4, 7]),
        constant_op.constant([1, 6, 2, 3, 5])
    ]
    data = [
        constant_op.constant([0, 40, 70]),
        constant_op.constant([10, 60, 20, 30])
    ]
    with self.assertRaises(ValueError):
      self.stitch_op(indices, data)

  def testOutOfBoundsIndexRaisesInvalidArgument(self):
    # [FUNCTION DOC]
    # Test: indices including a large negative value and values larger than the data size must make
    # evaluation raise an invalid-argument error whose message mentions out of range.
    with self.assertRaisesRegex(errors.InvalidArgumentError, "out of range"):
      indices = [[-1000], [405], [519], [758], [1015]]
      data = [
          [110.27793884277344],
          [120.29475402832031],
          [157.2418212890625],
          [157.2626953125],
          [188.45382690429688],
      ]

      self.evaluate(self.stitch_op(indices, data))


class DynamicStitchTest(DynamicStitchTestBase, test.TestCase):
  # [CLASS DOC]
  # Concrete test case that runs all shared stitch scenarios against the sequential dynamic stitch
  # op. Combines the mixin with the framework test case.

  def __init__(self, *test_case_args):
    # [FUNCTION DOC]
    # Constructor: initializes the test-case base with the received test arguments and then the mixin
    # with the sequential dynamic stitch function. Inputs: the arguments forwarded to the test case
    # (typically the test method name). Returns nothing.
    test.TestCase.__init__(self, *test_case_args)
    DynamicStitchTestBase.__init__(self, data_flow_ops.dynamic_stitch)


class ParallelDynamicStitchTest(DynamicStitchTestBase, test.TestCase):
  # [CLASS DOC]
  # Concrete test case that runs the shared scenarios against the parallel dynamic stitch op, and
  # adds scalar and higher-rank tests (including GPU-oriented variants) that override the mixin's.

  def __init__(self, *test_case_args):
    # [FUNCTION DOC]
    # Constructor: initializes the test-case base with the received arguments and then the mixin with
    # the parallel dynamic stitch function. Returns nothing.
    test.TestCase.__init__(self, *test_case_args)
    DynamicStitchTestBase.__init__(self, data_flow_ops.parallel_dynamic_stitch)

  def testScalar(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: overrides the shared scalar test but calls the sequential dynamic stitch function
    # directly instead of the parallel op under test, so the parallel op is not exercised here.
    # Test: stitches two float scalars with indices 0 and 1 in both orders and checks values and
    # the static shape of 2.
    with test_util.use_gpu():
      indices = [constant_op.constant(0), constant_op.constant(1)]
      data = [constant_op.constant(40.0), constant_op.constant(60.0)]
      for step in -1, 1:
        stitched_t = data_flow_ops.dynamic_stitch(indices[::step], data)
        stitched_val = self.evaluate(stitched_t)
        self.assertAllEqual([40.0, 60.0][::step], stitched_val)
        # Dimension 0 is max(flatten(indices))+1.
        self.assertEqual([2], stitched_t.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testHigherRank(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: overrides the shared higher-rank test but calls the sequential dynamic stitch
    # function directly instead of the parallel op under test.
    # Test (graph mode): float data with indices of ranks 0 to 2; checks 7x2 output, static shape,
    # and gradients (index gradients nothing, data gradients scaled by 7).
    indices = [
        constant_op.constant(6),
        constant_op.constant([4, 1]),
        constant_op.constant([[5, 2], [0, 3]])
    ]
    data = [
        constant_op.constant([61, 62], dtype=dtypes.float32),
        constant_op.constant([[41, 42], [11, 12]], dtype=dtypes.float32),
        constant_op.constant(
            [[[51, 52], [21, 22]], [[1, 2], [31, 32]]], dtype=dtypes.float32)
    ]
    stitched_t = data_flow_ops.dynamic_stitch(indices, data)
    stitched_val = self.evaluate(stitched_t)
    correct = 10 * np.arange(7)[:, None] + [1.0, 2.0]
    self.assertAllEqual(correct, stitched_val)
    self.assertEqual([7, 2], stitched_t.get_shape().as_list())
    # Test gradients
    stitched_grad = 7 * stitched_val
    grads = gradients_impl.gradients(stitched_t, indices + data,
                                     stitched_grad)
    self.assertEqual(grads[:3], [None] * 3)  # Indices have no gradients
    for datum, grad in zip(data, self.evaluate(grads[3:])):
      self.assertAllEqual(7.0 * self.evaluate(datum), grad)

  # GPU version unit tests
  def testScalarGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name it does not request a GPU explicitly and calls the sequential
    # stitch function rather than the parallel op; it duplicates the scalar test without GPU scope.
    # Test: stitches two float scalars in both orders and checks values and static shape.
    indices = [constant_op.constant(0), constant_op.constant(1)]
    data = [constant_op.constant(40.0), constant_op.constant(60.0)]
    for step in -1, 1:
      stitched_t = data_flow_ops.dynamic_stitch(indices[::step], data)
      stitched_val = self.evaluate(stitched_t)
      self.assertAllEqual([40.0, 60.0][::step], stitched_val)
      # Dimension 0 is max(flatten(indices))+1.
      self.assertEqual([2], stitched_t.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testHigherRankGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name it does not request a GPU and calls the sequential stitch
    # function rather than the parallel op; it is an exact duplicate of the higher-rank test above.
    # Test (graph mode): same checks as the higher-rank test, including gradients.
    indices = [
        constant_op.constant(6),
        constant_op.constant([4, 1]),
        constant_op.constant([[5, 2], [0, 3]])
    ]
    data = [
        constant_op.constant([61, 62], dtype=dtypes.float32),
        constant_op.constant([[41, 42], [11, 12]], dtype=dtypes.float32),
        constant_op.constant(
            [[[51, 52], [21, 22]], [[1, 2], [31, 32]]], dtype=dtypes.float32)
    ]
    stitched_t = data_flow_ops.dynamic_stitch(indices, data)
    stitched_val = self.evaluate(stitched_t)
    correct = 10 * np.arange(7)[:, None] + [1.0, 2.0]
    self.assertAllEqual(correct, stitched_val)
    self.assertEqual([7, 2], stitched_t.get_shape().as_list())
    # Test gradients
    stitched_grad = 7 * stitched_val
    grads = gradients_impl.gradients(stitched_t, indices + data,
                                     stitched_grad)
    self.assertEqual(grads[:3], [None] * 3)  # Indices have no gradients
    for datum, grad in zip(data, self.evaluate(grads[3:])):
      self.assertAllEqual(7.0 * self.evaluate(datum), grad)

  @test_util.run_in_graph_and_eager_modes
  def testMismatchedDataAndIndexListSizes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: like the other overrides it calls the sequential dynamic stitch function, not the
    # parallel one, although it lives in the parallel test case.
    # Test (graph and eager modes): passes index and data lists of different lengths (indices cut to
    # 2 vs 4 data, then 4 indices vs 2 data) and expects a value or invalid-argument error whose
    # message says the list sizes do not match.
    indices = [
        constant_op.constant([2]),
        constant_op.constant([1]),
        constant_op.constant([0]),
        constant_op.constant([3]),
    ]
    data = [
        constant_op.constant([1.0]),
        constant_op.constant([2.0]),
        constant_op.constant([3.0]),
        constant_op.constant([4.0])
    ]
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "expected inputs .* do not match|List argument .* must match"):
      self.evaluate(data_flow_ops.dynamic_stitch(indices[0:2], data))

    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "expected inputs .* do not match|List argument .* must match"):
      self.evaluate(data_flow_ops.dynamic_stitch(indices, data[0:2]))

if __name__ == "__main__":
  test.main()
