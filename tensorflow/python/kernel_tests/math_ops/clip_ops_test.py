# [MODULE DOC]
# Unit tests for the clipping ops (clip by value, by norm, by global norm, by average norm).
# Covers scalar and tensor bounds, several dtypes, gradients of clip by value, sparse
# (indexed slices) inputs, bad shapes, non-finite values, zero norms and empty tensors.
# No import-time side effects; run as a script it starts the TensorFlow test runner.
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
"""Tests for tensorflow.ops.clip_ops."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ClipTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the tensor clipping functions in the clip ops module. Has no shared fixtures;
  # each test builds constants inside its own session and compares against hand-computed results.

  # ClipByValue test
  def testClipByValue(self):
    # [FUNCTION DOC]
    # Test clip by value on a 2x3 float tensor with scalar bounds of minus and plus 4.4.
    # Asserts the output equals the input with entries limited to that range.
    with self.session():
      x = constant_op.constant([-5.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
      np_ans = [[-4.4, 2.0, 3.0], [4.0, 4.4, 4.4]]
      clip_value = 4.4
      ans = clip_ops.clip_by_value(x, -clip_value, clip_value)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Scalar, Scalar]
  def testClipByValue0Type(self):
    # [FUNCTION DOC]
    # Test clip by value with a tensor input and scalar min (2) and max (4) for each of eight
    # dtypes (float16/32/64, bfloat16, int16/32/64, uint8). Asserts values are limited to [2, 4].
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.bfloat16,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint8,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[2, 2, 3], [4, 4, 4]]
        clip_value_min = 2
        clip_value_max = 4
        ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Tensor, Scalar]
  def testClipByValue1Type(self):
    # [FUNCTION DOC]
    # Test clip by value with a tensor input, an elementwise tensor min and a scalar max (4),
    # across the same eight dtypes. Asserts the result honors per-element minimums and the max.
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.bfloat16,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint8,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[2, 2, 3], [4, 4, 4]]
        clip_value_min = constant_op.constant(
            [2, 2, 2, 3, 3, 3], shape=[2, 3], dtype=dtype)
        clip_value_max = 4
        ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Scalar, Tensor]
  def testClipByValue2Type(self):
    # [FUNCTION DOC]
    # Test clip by value with a tensor input, a scalar min (4) and an elementwise tensor max,
    # across the same eight dtypes. Asserts the expected clipped output.
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.bfloat16,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint8,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[4, 4, 4], [4, 5, 6]]
        clip_value_min = 4
        clip_value_max = constant_op.constant(
            [6, 6, 6, 6, 6, 6], shape=[2, 3], dtype=dtype)
        ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Tensor, Tensor]
  def testClipByValue3Type(self):
    # [FUNCTION DOC]
    # Test clip by value when input, min and max are all tensors of the same shape, across the
    # same eight dtypes. Asserts the expected elementwise clipped output.
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.bfloat16,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint8,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[2, 2, 3], [5, 5, 6]]
        clip_value_min = constant_op.constant(
            [2, 2, 2, 5, 5, 5], shape=[2, 3], dtype=dtype)
        clip_value_max = constant_op.constant(
            [5, 5, 5, 7, 7, 7], shape=[2, 3], dtype=dtype)
        ans = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  def testClipByValueGradient(self):
    # [FUNCTION DOC]
    # Test the gradients of clip by value with respect to input, min and max, using a gradient
    # tape. Runs for both the public op and the raw generated op, and covers scalar, vector and
    # mixed shapes; the final matrix/vector/matrix broadcasting case is checked only with the
    # public op because the raw op needs scalar or same-shape bounds. Asserts that the input
    # gradient is one where the input lies inside the bounds, and that min and max gradients
    # collect (sum over broadcast dimensions) the positions where they were selected.
    def grad(x, y, z, clip_fn):
      # [FUNCTION DOC]
      # HELPER: computes gradients of a clip function. Takes x, y, z (input, min, max values,
      # converted to float32 constants) and clip_fn (callable taking the three tensors). Records
      # the call on a gradient tape and returns the list of gradients of the output with respect
      # to x, y and z.
      x = constant_op.constant(x, dtype=dtypes.float32)
      y = constant_op.constant(y, dtype=dtypes.float32)
      z = constant_op.constant(z, dtype=dtypes.float32)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        tape.watch(y)
        tape.watch(z)
        output = clip_fn(x, y, z)
      return tape.gradient(output, [x, y, z])

    for f in (clip_ops.clip_by_value, gen_math_ops._clip_by_value):
      with self.subTest(f=f):
        # Input: [Scalar, Scalar, Scalar]
        xg, yg, zg = grad(0, -1, 1, clip_fn=f)
        self.assertEqual(self.evaluate(xg), 1)
        self.assertEqual(self.evaluate(yg), 0)
        self.assertEqual(self.evaluate(zg), 0)

        # Input: [Scalar, Scalar, Scalar]
        xg, yg, zg = grad(2, -1, 1, clip_fn=f)
        self.assertEqual(self.evaluate(xg), 0)
        self.assertEqual(self.evaluate(yg), 0)
        self.assertEqual(self.evaluate(zg), 1)

        # Input: [Vector, Scalar, Scalar]
        xg, yg, zg = grad([0, -2, 2, -2], -1, 1, clip_fn=f)
        self.assertAllEqual(self.evaluate(xg), [1, 0, 0, 0])
        self.assertEqual(self.evaluate(yg), 2)
        self.assertEqual(self.evaluate(zg), 1)

        # Input: [Vector, Vector, Scalar]
        xg, yg, zg = grad([-1, -2, 0, 2], [-2, -1, -3, 0], 1, clip_fn=f)
        self.assertAllEqual(self.evaluate(xg), [1, 0, 1, 0])
        self.assertAllEqual(self.evaluate(yg), [0, 1, 0, 0])
        self.assertEqual(self.evaluate(zg), 1)

        # Input: [Vector, Vector, Vector]
        xg, yg, zg = grad(
            [-1, -2, 0, 2], [-2, -1, -3, 0], [1, 2, -1, 1], clip_fn=f
        )
        self.assertAllEqual(self.evaluate(xg), [1, 0, 0, 0])
        self.assertAllEqual(self.evaluate(yg), [0, 1, 0, 0])
        self.assertAllEqual(self.evaluate(zg), [0, 0, 1, 1])

    # Only test the following with `clip_ops.clip_by_value`, as
    # `gen_math_ops._clip_by_value` requires the min and max values to be
    # scalar or the same shape as the input.

    # Input: [Matrix, Vector, Matrix]
    xg, yg, zg = grad([[-2, 3], [2, -1]], [-1, -2], [[1, 2], [3, 4]],
                      clip_fn=clip_ops.clip_by_value)
    self.assertAllEqual(self.evaluate(xg), [[0, 0], [1, 1]])
    self.assertAllEqual(self.evaluate(yg), [1, 0])
    self.assertAllEqual(self.evaluate(zg), [[0, 1], [0, 0]])

  def testClipByValueBadShape(self):
    # [FUNCTION DOC]
    # Test that clip by value raises a value error when the bounds have an incompatible shape
    # (a length-2 vector against a 2x3x1 input), both when both bounds are that vector (negated
    # and plain) and when only the max is.
    with self.session():
      x = constant_op.constant([-5.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3, 1])
      # Use a nonsensical shape.
      clip = constant_op.constant([1.0, 2.0])
      with self.assertRaises(ValueError):
        _ = clip_ops.clip_by_value(x, -clip, clip)  # pylint: disable=invalid-unary-operand-type
      with self.assertRaises(ValueError):
        _ = clip_ops.clip_by_value(x, 1.0, clip)

  def testClipByValueNonFinite(self):
    # [FUNCTION DOC]
    # Test clip by value on not-a-number, positive infinity and negative infinity with bounds of
    # minus and plus 4. Asserts not-a-number passes through and infinities become the bounds.
    # Not run on GPU (a TODO references an open bug).
    # TODO(b/78016351): Enable test on GPU once the bug is fixed.
    with self.cached_session():
      x = constant_op.constant([float('NaN'), float('Inf'), -float('Inf')])
      np_ans = [float('NaN'), 4.0, -4.0]
      clip_value = 4.0
      ans = clip_ops.clip_by_value(x, -clip_value, clip_value)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def _testClipIndexedSlicesByValue(self, values, indices, shape,
                                    clip_value_min, clip_value_max, expected):
    # [FUNCTION DOC]
    # HELPER: shared check for clipping sparse inputs. Inputs: values, indices and dense shape
    # (nested lists), scalar min and max bounds, and the expected clipped values. Builds a sparse
    # indexed-slices object, clips it by value, asserts the result is still an indexed-slices
    # object and that its values are close to expected. Returns nothing.
    with self.session():
      values = constant_op.constant(values)
      indices = constant_op.constant(indices)
      shape = constant_op.constant(shape)
      # IndexedSlices mode
      indexed_slices = indexed_slices_lib.IndexedSlices(values, indices, shape)
      clipped = clip_ops.clip_by_value(indexed_slices, clip_value_min,
                                       clip_value_max)
      # clipped should be IndexedSlices
      self.assertIsInstance(clipped, indexed_slices_lib.IndexedSlices)

    self.assertAllClose(clipped.values, expected)

  def testClipByValueWithIndexedSlicesClipped(self):
    # [FUNCTION DOC]
    # Test clip by value on a sparse indexed-slices input with three bound pairs (-2..2, 1..2,
    # -2..-1) using the shared helper; each case checks the clipped slice values.
    values = [[[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
              [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]]
    indices = [2, 6]
    shape = [10, 2, 3]
    # [-2.0, 2.0]
    self._testClipIndexedSlicesByValue(values, indices, shape, -2.0, 2.0,
                                       [[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                                        [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]])
    # [1.0, 2.0]
    self._testClipIndexedSlicesByValue(values, indices, shape, 1.0, 2.0,
                                       [[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                                        [[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]])
    # [-2.0, -1.0]
    self._testClipIndexedSlicesByValue(
        values, indices, shape, -2.0, -1.0,
        [[[-2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
         [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]])

  # ClipByNorm tests
  def testClipByNormClipped(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the clip by norm call is executed twice with identical arguments and both
    # results are checked against the same answer; the second ("tensor") variant never actually
    # uses a tensor clip norm, so it duplicates the first.
    # Test clip by norm on a 2x3 tensor of norm 5 with clip norm 4. Asserts the result is the
    # input scaled by four fifths.
    # Norm clipping when clip_norm < 5
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Norm of x = sqrt(3^2 + 4^2) = 5
      np_ans = [[-2.4, 0.0, 0.0], [3.2, 0.0, 0.0]]
      clip_norm = 4.0
      ans = clip_ops.clip_by_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

      ans = clip_ops.clip_by_norm(x, clip_norm)
      tf_ans_tensor = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)
    self.assertAllClose(np_ans, tf_ans_tensor)

  @test_util.run_deprecated_v1
  def testClipByNormGradientZeros(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated-v1 decorator) that the gradient of clip by norm with respect
    # to an all-zero length-3 input with clip norm 1 is all ones (no division-by-zero NaNs).
    with self.session():
      x = array_ops.zeros([3])
      b = clip_ops.clip_by_norm(x, 1.)
      grad, = gradients_impl.gradients(b, x)
      self.assertAllEqual(grad, [1., 1., 1.])

  def testClipByNormBadShape(self):
    # [FUNCTION DOC]
    # Test that clip by norm raises a value error when the clip norm is a length-2 vector that
    # is not broadcast-compatible with a 2x3x1 input.
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3, 1])
      # Use a nonsensical shape.
      clip = constant_op.constant([1.0, 2.0])
      with self.assertRaises(ValueError):
        _ = clip_ops.clip_by_norm(x, clip)

  def testClipByNormNotClipped(self):
    # [FUNCTION DOC]
    # Test clip by norm leaves a tensor of norm 5 unchanged when the clip norm is 6.
    # No norm clipping when clip_norm >= 5
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Norm of x = sqrt(3^2 + 4^2) = 5
      np_ans = [[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
      clip_norm = 6.0
      ans = clip_ops.clip_by_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByNormZero(self):
    # [FUNCTION DOC]
    # Test clip by norm on an all-zero tensor with clip norm 6; the output must stay all zeros.
    # No norm clipping when norm = 0
    with self.session():
      x = constant_op.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
      # Norm = 0, no changes
      np_ans = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      clip_norm = 6.0
      ans = clip_ops.clip_by_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByNormClippedWithDim0(self):
    # [FUNCTION DOC]
    # Test clip by norm computed along axis 0 with clip norm 4: the first column (norm 5) is
    # scaled down, the last column (norm 3) is untouched. Asserts the expected values.
    # Norm clipping when clip_norm < 5
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 3.0], shape=[2, 3])
      # Norm of x[:, 0] = sqrt(3^2 + 4^2) = 5, x[:, 2] = 3
      np_ans = [[-2.4, 0.0, 0.0], [3.2, 0.0, 3.0]]
      clip_norm = 4.0
      ans = clip_ops.clip_by_norm(x, clip_norm, [0])
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByNormClippedWithDim1(self):
    # [FUNCTION DOC]
    # Test clip by norm computed along axis 1 with clip norm 4: only the second row (norm 5) is
    # scaled down. Asserts the expected values.
    # Norm clipping when clip_norm < 5
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 3.0], shape=[2, 3])
      # Norm of x[0, :] = 3, x[1, :] = sqrt(3^2 + 4^2) = 5
      np_ans = [[-3.0, 0.0, 0.0], [3.2, 0.0, 2.4]]
      clip_norm = 4.0
      ans = clip_ops.clip_by_norm(x, clip_norm, [1])
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByNormNotClippedWithAxes(self):
    # [FUNCTION DOC]
    # Test clip by norm along axis 1 with clip norm 6 (above every row norm): input unchanged.
    # No norm clipping when clip_norm >= 5
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 3.0], shape=[2, 3])
      # Norm of x[0, :] = 3, x[1, :] = sqrt(3^2 + 4^2) = 5
      np_ans = [[-3.0, 0.0, 0.0], [4.0, 0.0, 3.0]]
      clip_norm = 6.0
      ans = clip_ops.clip_by_norm(x, clip_norm, [1])
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  # ClipByGlobalNorm tests
  def testClipByGlobalNormClipped(self):
    # [FUNCTION DOC]
    # Test clip by global norm on a tuple of two tensors with global norm 5 and a float clip norm
    # of 4. Asserts the reported global norm is 5 and each tensor is scaled by four fifths.
    # Norm clipping when clip_norm < 5
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = constant_op.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = 4.0

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = clip_ops.clip_by_global_norm((x0, x1), clip_norm)
      tf_ans_1 = self.evaluate(ans[0])
      tf_ans_2 = self.evaluate(ans[1])
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  def testClipByGlobalNormClippedTensor(self):
    # [FUNCTION DOC]
    # Same scenario as the clipped global norm test but the clip norm is given as a tensor
    # constant. Asserts global norm 5 and scaling by four fifths.
    # Norm clipping when clip_norm < 5
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = constant_op.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = constant_op.constant(4.0)

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = clip_ops.clip_by_global_norm((x0, x1), clip_norm)
      tf_ans_1 = self.evaluate(ans[0])
      tf_ans_2 = self.evaluate(ans[1])
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  def testClipByGlobalNormSupportsNone(self):
    # [FUNCTION DOC]
    # Test clip by global norm accepts a list containing null entries: null entries stay null in
    # the output, and the non-null tensors are clipped and the norm is 5 as usual.
    # Norm clipping when clip_norm < 5
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = constant_op.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = 4.0

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = clip_ops.clip_by_global_norm((x0, None, x1, None), clip_norm)
      self.assertTrue(ans[1] is None)
      self.assertTrue(ans[3] is None)
      tf_ans_1 = self.evaluate(ans[0])
      tf_ans_2 = self.evaluate(ans[2])
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  @test_util.run_deprecated_v1
  def testClipByGlobalNormWithIndexedSlicesClipped(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated-v1 decorator) of clip by global norm mixing a dense tensor and a
    # sparse indexed-slices value. Asserts global norm 5, scaling by four fifths of the dense
    # tensor, and of the sparse values.
    # Norm clipping when clip_norm < 5
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = indexed_slices_lib.IndexedSlices(
          constant_op.constant([1.0, -2.0]), constant_op.constant([3, 4]))
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = 4.0

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0], [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = self.evaluate(ans[0])
      tf_ans_2 = self.evaluate(ans[1].values)
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  def testClipByGlobalNormPreservesDenseShape(self):
    # [FUNCTION DOC]
    # Test that clip by global norm keeps the dense shape attribute of an indexed-slices input:
    # both the original and the clipped object report the shape of length one.
    dense_shape = (1,)
    slices = indexed_slices_lib.IndexedSlices(
        constant_op.constant([1.0]),
        constant_op.constant([0]),
        dense_shape=dense_shape)
    ans, _ = clip_ops.clip_by_global_norm([slices], 1.0)
    modified_slices = ans[0]
    self.assertEqual(dense_shape, slices.dense_shape)
    self.assertEqual(dense_shape, modified_slices.dense_shape)

  def testClipByGlobalNormNotClipped(self):
    # [FUNCTION DOC]
    # Test clip by global norm with clip norm 6 above the global norm 5: tensors are unchanged
    # and the reported norm is 5.
    # No norm clipping when clip_norm >= 5
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = constant_op.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      np_ans_0 = [[-2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
      np_ans_1 = [1.0, -2.0]
      clip_norm = 6.0

      ans, norm = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = self.evaluate(ans[0])
      tf_ans_2 = self.evaluate(ans[1])
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  def testClipByGlobalNormZero(self):
    # [FUNCTION DOC]
    # Test clip by global norm on all-zero tensors: the reported norm is 0 and outputs are zeros.
    # No norm clipping when norm = 0
    with self.session():
      x0 = constant_op.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
      x1 = constant_op.constant([0.0, 0.0])
      # Norm = 0, no changes
      np_ans_0 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      np_ans_1 = [0.0, 0.0]
      clip_norm = 6.0

      ans, norm = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = self.evaluate(ans[0])
      tf_ans_2 = self.evaluate(ans[1])
      tf_norm = self.evaluate(norm)

    self.assertAllClose(tf_norm, 0.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  def testClipByGlobalNormInf(self):
    # [FUNCTION DOC]
    # Test clip by global norm when one input contains infinity: the global norm is infinite and
    # every output element in every tensor becomes not-a-number.
    # Expect all NaNs when global norm is inf.
    with self.session():
      x0 = constant_op.constant([-2.0, 0.0, np.inf, 4.0, 0.0, 0.0],
                                shape=[2, 3])
      x1 = constant_op.constant([1.0, -2.0])
      clip_norm = 6.0

      ans, norm = clip_ops.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = self.evaluate(ans[0])
      tf_ans_2 = self.evaluate(ans[1])
      tf_norm = self.evaluate(norm)
      self.assertAllEqual(tf_norm, float('inf'))
      self.assertAllEqual(tf_ans_1, np.full([2, 3], float('nan')))
      self.assertAllEqual(tf_ans_2, np.full([2], float('nan')))

  def testClipByAverageNormClipped(self):
    # [FUNCTION DOC]
    # Test clip by average norm with clip norm 0.8 below the average norm (about 0.833): the
    # tensor is scaled down to the expected values.
    # Norm clipping when average clip_norm < 0.83333333
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Average norm of x = sqrt(3^2 + 4^2) / 6 = 0.83333333
      np_ans = [[-2.88, 0.0, 0.0], [3.84, 0.0, 0.0]]
      clip_norm = 0.8
      ans = clip_ops.clip_by_average_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByAverageNormClippedTensor(self):
    # [FUNCTION DOC]
    # Same as the clipped average norm test but the clip norm is passed as a tensor constant.
    # Norm clipping when average clip_norm < 0.83333333
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Average norm of x = sqrt(3^2 + 4^2) / 6 = 0.83333333
      np_ans = [[-2.88, 0.0, 0.0], [3.84, 0.0, 0.0]]
      clip_norm = constant_op.constant(0.8)
      ans = clip_ops.clip_by_average_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByAverageNormNotClipped(self):
    # [FUNCTION DOC]
    # Test clip by average norm with clip norm 0.9 above the average norm: input unchanged.
    # No norm clipping when average clip_norm >= 0.83333333
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Average norm of x = sqrt(3^2 + 4^2) / 6 = 0.83333333
      np_ans = [[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
      clip_norm = 0.9
      ans = clip_ops.clip_by_average_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByAverageNormZero(self):
    # [FUNCTION DOC]
    # Test clip by average norm on an all-zero tensor: output is all zeros.
    # No norm clipping when average clip_norm = 0
    with self.session():
      x = constant_op.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
      # Average norm = 0, no changes
      np_ans = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      clip_norm = 0.9
      ans = clip_ops.clip_by_average_norm(x, clip_norm)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  def testClipByAverageNormReplacedWithClipByNorm(self):
    # [FUNCTION DOC]
    # Test the equivalence of clip by average norm with clip norm c and clip by norm with clip
    # norm c times the element count (cast to float32). Asserts both outputs are close.
    # Check clip_by_average_norm(t) is the same as
    # clip_by_norm(t, clip_norm * tf.compat.v1.to_float(tf.size(t)))
    with self.session():
      x = constant_op.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Average norm of x = sqrt(3^2 + 4^2) / 6 = 0.83333333
      # expected answer [[-2.88, 0.0, 0.0], [3.84, 0.0, 0.0]]
      clip_norm = constant_op.constant(0.8)
      with_norm = clip_ops.clip_by_average_norm(x, clip_norm)
      without_norm = clip_ops.clip_by_norm(
          x, clip_norm * math_ops.cast(array_ops.size(x), dtypes.float32))
      clip_by_average_norm_ans = self.evaluate(with_norm)
      clip_by_norm_ans = self.evaluate(without_norm)
      self.assertAllClose(clip_by_average_norm_ans, clip_by_norm_ans)

  @test_util.run_deprecated_v1
  def testClipByValueEmptyTensor(self):
    # [FUNCTION DOC]
    # Graph-mode regression test (deprecated-v1 decorator; GitHub issue 19337) that clip by value
    # works on placeholders fed with an empty (7 by 0) array, using placeholder, scalar and mixed
    # bounds. Only checks that running the graph does not raise; no values are asserted.
    # Test case for GitHub issue 19337
    zero = array_ops.placeholder(dtype=dtypes.float32, shape=None)
    x = clip_ops.clip_by_value(zero, zero, zero)
    y = clip_ops.clip_by_value(zero, 1.0, 1.0)
    z = clip_ops.clip_by_value(zero, zero, 1.0)
    w = clip_ops.clip_by_value(zero, 1.0, zero)
    with self.session() as sess:
      sess.run([x, y, z, w], feed_dict={zero: np.zeros((7, 0))})


if __name__ == '__main__':
  test.main()
