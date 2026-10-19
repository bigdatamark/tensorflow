# [MODULE DOC]
# Functional tests for the batch-to-space op and its N-dimensional variant. Covers a consistency
# check against depth-to-space, argument validation errors, static/dynamic shape inference and
# numerical gradient checks. The shared test classes are written against a mixin that supplies the
# op implementation, and each is subclassed once for the Python wrapper and once for the raw
# generated (C++-backed) op. No import-time side effects.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for BatchToSpace op.

Additional tests are included in spacetobatch_op_test.py, where the BatchToSpace
op is tested in tandem with its reverse SpaceToBatch op.
"""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test


class PythonOpImpl(object):
  # [CLASS DOC]
  # HELPER: mixin that selects the public Python wrapper of the batch-to-space op for the test
  # classes that inherit from it. Stateless.

  @staticmethod
  def batch_to_space(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: forwards all positional and keyword arguments unchanged to the public array ops
    # batch-to-space function and returns its resulting tensor. Static method; no side effects.
    return array_ops.batch_to_space(*args, **kwargs)


class CppOpImpl(object):
  # [CLASS DOC]
  # HELPER: mixin that selects the raw generated batch-to-space op (the one backed directly by
  # the native kernel) for the test classes that inherit from it. Stateless.

  @staticmethod
  def batch_to_space(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: forwards all positional and keyword arguments unchanged to the generated array ops
    # batch-to-space function and returns its resulting tensor. Static method; no side effects.
    return gen_array_ops.batch_to_space(*args, **kwargs)


class BatchToSpaceDepthToSpace(test.TestCase, parameterized.TestCase,
                               PythonOpImpl):
  # [CLASS DOC]
  # Test case checking that batch-to-space agrees with a transpose, depth-to-space, transpose
  # composition. Uses the op implementation supplied by the mixin base (Python wrapper here;
  # subclassed below for the generated op). Parameterized over value dtype.

  # Verifies that: batch_to_space(x) = transpose(depth_to_space(transpose(x)))
  @parameterized.parameters(np.float32, dtypes.bfloat16.as_numpy_dtype)
  @test_util.run_deprecated_v1
  def testDepthToSpaceTranspose(self, dtype):
    # [FUNCTION DOC]
    # Parameterized by numpy value dtype (32-bit float or bfloat16). Builds a 20x5x8x7 array and,
    # for 64-bit and 32-bit integer all-zero crops, computes batch-to-space with block size 2 and
    # the equivalent transposed depth-to-space result, then asserts both are equal in a cached
    # session. Runs under the deprecated graph-mode decorator. Returns nothing.
    x = np.arange(20 * 5 * 8 * 7, dtype=dtype).reshape([20, 5, 8, 7])
    block_size = 2
    for crops_dtype in [dtypes.int64, dtypes.int32]:
      crops = array_ops.zeros((2, 2), dtype=crops_dtype)
      y1 = self.batch_to_space(x, crops, block_size=block_size)
      y2 = array_ops.transpose(
          array_ops.depth_to_space(
              array_ops.transpose(x, [3, 1, 2, 0]), block_size=block_size),
          [3, 1, 2, 0])
      with self.cached_session():
        self.assertAllEqual(y1, y2)


class BatchToSpaceDepthToSpaceCpp(BatchToSpaceDepthToSpace, CppOpImpl):
  # [CLASS DOC]
  # Re-runs the depth-to-space consistency test using the generated op implementation instead of
  # the Python wrapper. Adds no new tests or state.
  pass


class BatchToSpaceErrorHandlingTest(test.TestCase, PythonOpImpl):
  # [CLASS DOC]
  # Test case verifying that batch-to-space rejects invalid inputs (missing batch dimension, bad
  # block sizes, indivisible batch) and that unknown input shapes still infer the output rank.
  # Uses the op implementation supplied by the mixin.

  @test_util.run_deprecated_v1
  def testInputWrongDimMissingBatch(self):
    # [FUNCTION DOC]
    # Passes a rank-3 input (no batch dimension) with zero crops and block size 2 and asserts a
    # value error is raised when the op is built. Returns nothing.
    # The input is missing the first dimension ("batch")
    x_np = [[[1], [2]], [[3], [4]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 2
    with self.assertRaises(ValueError):
      _ = self.batch_to_space(x_np, crops, block_size)

  @test_util.run_deprecated_v1
  def testBlockSize0(self):
    # [FUNCTION DOC]
    # Uses block size 0 on a valid 4-D input with zero crops and asserts a value error is raised
    # while building or evaluating the op. Returns nothing.
    # The block size is 0.
    x_np = [[[[1], [2]], [[3], [4]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 0
    with self.assertRaises(ValueError):
      out_tf = self.batch_to_space(x_np, crops, block_size)
      self.evaluate(out_tf)

  @test_util.run_deprecated_v1
  def testBlockSizeOne(self):
    # [FUNCTION DOC]
    # Uses block size 1 (which must be greater than 1) and asserts a value error is raised
    # while building or evaluating the op. Returns nothing.
    # The block size is 1. The block size needs to be > 1.
    x_np = [[[[1], [2]], [[3], [4]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 1
    with self.assertRaises(ValueError):
      out_tf = self.batch_to_space(x_np, crops, block_size)
      out_tf.eval()

  @test_util.run_deprecated_v1
  def testBlockSizeLarger(self):
    # [FUNCTION DOC]
    # Uses block size 10, too large for the 2x2 spatial input with zero crops, and asserts a
    # value error is raised. Returns nothing.
    # The block size is too large for this input.
    x_np = [[[[1], [2]], [[3], [4]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 10
    with self.assertRaises(ValueError):
      out_tf = self.batch_to_space(x_np, crops, block_size)
      self.evaluate(out_tf)

  @test_util.run_deprecated_v1
  def testBlockSizeSquaredNotDivisibleBatch(self):
    # [FUNCTION DOC]
    # Uses a batch of 1 with block size 3 (block size squared does not divide the batch) and
    # asserts a value error is raised. Returns nothing.
    # The block size squared does not divide the batch.
    x_np = [[[[1], [2], [3]], [[3], [4], [7]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(ValueError):
      _ = self.batch_to_space(x_np, crops, block_size)

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Applies batch-to-space with block size 4 to placeholders of fully unknown shape and
    # asserts the static output rank is still inferred as 4. Returns nothing.
    t = self.batch_to_space(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32),
        block_size=4)
    self.assertEqual(4, t.get_shape().ndims)


class BatchToSpaceErrorHandlingCppTest(BatchToSpaceErrorHandlingTest,
                                       CppOpImpl):
  # [CLASS DOC]
  # Re-runs the error handling tests using the generated op implementation instead of the
  # Python wrapper. Adds no new tests or state.
  pass


class BatchToSpaceNDErrorHandlingTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that the N-dimensional batch-to-space op raises errors for invalid
  # shapes, block shapes and crops, both when sizes are static and when supplied dynamically, and
  # that shape inference works with partially known shapes.

  def _testStaticShape(self, input_shape, block_shape, paddings, error):
    # [FUNCTION DOC]
    # HELPER: static-shape half of the error check. Builds the op from numpy zeros of the given
    # input shape with the given block shape and paddings (crops) and asserts that building it
    # raises the expected error type. Returns nothing.
    block_shape = np.array(block_shape)
    paddings = np.array(paddings)

    # Try with sizes known at graph construction time.
    with self.assertRaises(error):
      _ = array_ops.batch_to_space_nd(
          np.zeros(input_shape, np.float32), block_shape, paddings)

  def _testDynamicShape(self, input_shape, block_shape, paddings):
    # [FUNCTION DOC]
    # HELPER: dynamic-shape half of the error check. Builds the op from placeholders (input of
    # unknown shape, block shape of known shape, paddings of unknown shape), feeds numpy values
    # for the given shapes and asserts that evaluation raises a value error. Returns nothing.
    block_shape = np.array(block_shape)
    paddings = np.array(paddings)

    # Try with sizes unknown at graph construction time.
    input_placeholder = array_ops.placeholder(dtypes.float32)
    block_shape_placeholder = array_ops.placeholder(
        dtypes.int32, shape=block_shape.shape)
    paddings_placeholder = array_ops.placeholder(dtypes.int32)
    t = array_ops.batch_to_space_nd(input_placeholder, block_shape_placeholder,
                                    paddings_placeholder)

    with self.assertRaises(ValueError):
      _ = t.eval({
          input_placeholder: np.zeros(input_shape, np.float32),
          block_shape_placeholder: block_shape,
          paddings_placeholder: paddings
      })

  def _testShape(self, input_shape, block_shape, paddings, error):
    # [FUNCTION DOC]
    # HELPER: runs both the static-shape check (expecting the given error type) and the
    # dynamic-shape check (expecting a value error) for the same input shape, block shape and
    # paddings. Returns nothing.
    self._testStaticShape(input_shape, block_shape, paddings, error)
    self._testDynamicShape(input_shape, block_shape, paddings)

  @test_util.run_deprecated_v1
  def testInputWrongDimMissingBatch(self):
    # [FUNCTION DOC]
    # Checks that inputs of too low rank for the given block shape (rank 2 with two block
    # dimensions, rank 3 with three) produce a value error in both static and dynamic modes.
    # Returns nothing.
    self._testShape([2, 2], [2, 2], [[0, 0], [0, 0]], ValueError)
    self._testShape([2, 2, 3], [2, 2, 3], [[0, 0], [0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testBlockSize0(self):
    # [FUNCTION DOC]
    # Checks that a block shape containing 0 produces a value error in both static and dynamic
    # modes. Returns nothing.
    # The block size is 0.
    self._testShape([1, 2, 2, 1], [0, 1], [[0, 0], [0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testBlockSizeNegative(self):
    # [FUNCTION DOC]
    # Checks that a negative block size produces a value error in both static and dynamic
    # modes. Returns nothing.
    self._testShape([1, 2, 2, 1], [-1, 1], [[0, 0], [0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testNegativePadding(self):
    # [FUNCTION DOC]
    # Checks that a negative crop amount produces a value error in both static and dynamic
    # modes. Returns nothing.
    self._testShape([1, 2, 2], [1, 1], [[0, -1], [0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testCropTooLarge(self):
    # [FUNCTION DOC]
    # Checks that crops whose total exceeds the expanded dimension size produce a value error in
    # both static and dynamic modes. Returns nothing.
    # The amount to crop exceeds the padded size.
    self._testShape([1 * 2 * 2, 2, 3, 1], [2, 2], [[3, 2], [0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testBlockSizeSquaredNotDivisibleBatch(self):
    # [FUNCTION DOC]
    # Checks that a batch size not divisible by the product of the block shape produces a value
    # error in both static and dynamic modes. Returns nothing.
    # The batch dimension is not divisible by the product of the block_shape.
    self._testShape([3, 1, 1, 1], [2, 3], [[0, 0], [0, 0]], ValueError)

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Verifies shape inference of the N-dimensional op with placeholders: fully unknown input
    # builds without error; known rank gives rank 4; partially known input shapes with a constant
    # block shape and crops give the expected partially or fully known output dimensions (batch
    # divided by the block product, spatial dimensions multiplied then cropped). Returns nothing.
    # Verify that input shape and paddings shape can be unknown.
    _ = array_ops.batch_to_space_nd(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))

    # Only number of input dimensions is known.
    t = array_ops.batch_to_space_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(None, None, None, None)),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))
    self.assertEqual(4, t.get_shape().ndims)

    # Dimensions are partially known.
    t = array_ops.batch_to_space_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(None, None, None, 2)),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))
    self.assertEqual([None, None, None, 2], t.get_shape().as_list())

    # Dimensions are partially known.
    t = array_ops.batch_to_space_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3 * 2 * 3, None, None, 2)), [2, 3],
        array_ops.placeholder(dtypes.int32))
    self.assertEqual([3, None, None, 2], t.get_shape().as_list())

    # Dimensions are partially known.
    t = array_ops.batch_to_space_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3 * 2 * 3, None, 2, 2)), [2, 3],
        [[1, 1], [0, 1]])
    self.assertEqual([3, None, 5, 2], t.get_shape().as_list())

    # Dimensions are fully known.
    t = array_ops.batch_to_space_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3 * 2 * 3, 2, 1, 2)), [2, 3],
        [[1, 1], [0, 0]])
    self.assertEqual([3, 2, 3, 2], t.get_shape().as_list())


class BatchToSpaceGradientTest(test.TestCase, PythonOpImpl):
  # [CLASS DOC]
  # Test case comparing theoretical and numerical gradients of batch-to-space for small inputs
  # and crops. Uses the op implementation supplied by the mixin.

  # Check the gradients.
  def _checkGrad(self, x, crops, block_size):
    # [FUNCTION DOC]
    # HELPER: gradient checker for the gradient tests. Takes a 4-D numpy array x, crops and a
    # block size; asserts that x has rank 4, applies batch-to-space in a cached session, computes
    # theoretical and numerical Jacobians with a small epsilon, and asserts they are close.
    # Returns nothing.
    assert 4 == x.ndim
    with self.cached_session():
      tf_x = ops.convert_to_tensor(x)
      tf_y = self.batch_to_space(tf_x, crops, block_size)
      epsilon = 1e-5
      ((x_jacob_t, x_jacob_n)) = gradient_checker.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a gradient for batch_to_space of x which is a four dimensional
  # tensor of shape [b * block_size * block_size, h, w, d].
  def _compare(self, b, h, w, d, block_size, crop_beg, crop_end):
    # [FUNCTION DOC]
    # HELPER: builds a random normal float32 input whose batch dimension is b times the block
    # size squared, with spatial sizes h and w and depth d, and crops of crop_beg and crop_end on
    # both spatial axes, then runs the gradient check. Returns nothing.
    block_size_sq = block_size * block_size
    x = np.random.normal(0, 1, b * h * w * d *
                         block_size_sq).astype(np.float32).reshape(
                             [b * block_size * block_size, h, w, d])
    crops = np.array(
        [[crop_beg, crop_end], [crop_beg, crop_end]], dtype=np.int32)

    self._checkGrad(x, crops, block_size)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  @test_util.run_deprecated_v1
  def testSmall(self):
    # [FUNCTION DOC]
    # Gradient check with block size 2, no cropping, base batch 1 and spatial size 2x3, depth 5.
    # Returns nothing.
    block_size = 2
    crop_beg = 0
    crop_end = 0
    self._compare(1, 2, 3, 5, block_size, crop_beg, crop_end)

  @test_util.run_deprecated_v1
  def testSmall2(self):
    # [FUNCTION DOC]
    # Gradient check with block size 2, no cropping, base batch 2 and spatial size 4x3, depth 2.
    # Returns nothing.
    block_size = 2
    crop_beg = 0
    crop_end = 0
    self._compare(2, 4, 3, 2, block_size, crop_beg, crop_end)

  @test_util.run_deprecated_v1
  def testSmallCrop1x1(self):
    # [FUNCTION DOC]
    # Gradient check with block size 2 and a crop of 1 at both ends of both spatial axes, base
    # batch 1, spatial size 2x3, depth 5. Returns nothing.
    block_size = 2
    crop_beg = 1
    crop_end = 1
    self._compare(1, 2, 3, 5, block_size, crop_beg, crop_end)


class BatchToSpaceGradientCppTest(BatchToSpaceGradientTest, CppOpImpl):
  # [CLASS DOC]
  # Re-runs the gradient tests using the generated op implementation instead of the Python
  # wrapper. Adds no new tests or state.
  pass


class BatchToSpaceNDGradientTest(test.TestCase):
  # [CLASS DOC]
  # Test case comparing theoretical and numerical gradients of the N-dimensional batch-to-space
  # op for small inputs, with both 32-bit and 64-bit integer crops.

  # Check the gradients.
  def _checkGrad(self, x, block_shape, crops, crops_dtype):
    # [FUNCTION DOC]
    # HELPER: gradient checker. Takes input array x, block shape, crops (reshaped to one pair per
    # block dimension) and the integer dtype for the crops constant. Applies the N-dimensional op
    # in a cached session, computes theoretical and numerical Jacobians and asserts they are
    # close. Returns nothing.
    block_shape = np.array(block_shape)
    crops = constant_op.constant(
        np.array(crops).reshape((len(block_shape), 2)), crops_dtype)
    with self.cached_session():
      tf_x = ops.convert_to_tensor(x)
      tf_y = array_ops.batch_to_space_nd(tf_x, block_shape, crops)
      epsilon = 1e-5
      ((x_jacob_t, x_jacob_n)) = gradient_checker.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  def _compare(self, input_shape, block_shape, crops, crops_dtype):
    # [FUNCTION DOC]
    # HELPER: scales the leading dimension of the given base input shape by the product of the
    # block shape, fills a float32 array of that shape with random normal values, and runs the
    # gradient check with the given block shape, crops and crops dtype. Returns nothing.
    input_shape = list(input_shape)
    input_shape[0] *= np.prod(block_shape)
    x = np.random.normal(
        0, 1, np.prod(input_shape)).astype(np.float32).reshape(input_shape)
    self._checkGrad(x, block_shape, crops, crops_dtype)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  @test_util.run_deprecated_v1
  def testSmall(self):
    # [FUNCTION DOC]
    # For each of 64-bit and 32-bit integer crops dtypes, gradient check with block shape 2x2,
    # zero crops and base shape 1x2x3x5. Returns nothing.
    for dtype in [dtypes.int64, dtypes.int32]:
      self._compare([1, 2, 3, 5], [2, 2], [[0, 0], [0, 0]], dtype)

  @test_util.run_deprecated_v1
  def testSmall2(self):
    # [FUNCTION DOC]
    # For each of 64-bit and 32-bit integer crops dtypes, gradient check with block shape 2x2,
    # zero crops and base shape 2x4x3x2. Returns nothing.
    for dtype in [dtypes.int64, dtypes.int32]:
      self._compare([2, 4, 3, 2], [2, 2], [[0, 0], [0, 0]], dtype)

  @test_util.run_deprecated_v1
  def testSmallCrop1x1(self):
    # [FUNCTION DOC]
    # For each of 64-bit and 32-bit integer crops dtypes, gradient check with block shape 2x2,
    # crops of 1 on both sides of both axes and base shape 1x2x3x5. Returns nothing.
    for dtype in [dtypes.int64, dtypes.int32]:
      self._compare([1, 2, 3, 5], [2, 2], [[1, 1], [1, 1]], dtype)


if __name__ == "__main__":
  test.main()
