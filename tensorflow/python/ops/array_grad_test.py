# [MODULE DOC]
# Unit tests for the gradient functions in array_grad.py. Each test builds a small float64
# input, wraps an array op (gather, broadcast, slice, reshape) in a function, and uses the
# numeric gradient checker to compare analytic and numeric Jacobians. Runs in graph and eager
# modes. No import-time side effects beyond test registration.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for array_grad."""

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class ArrayGradTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the array-op gradients (gather, broadcast-to, slice, reshape). Decorated
  # to run every test in both graph and eager modes, with eager ops executed as functions. The
  # shared fixture is the gradient-check helper method.

  def _testGrad(self, f, x):
    # [FUNCTION DOC]
    # HELPER: shared assertion used by all tests in this class.
    # Computes theoretical and numeric gradients of function f at input x using the gradient
    # checker, and asserts that the maximum error is below 1e-4. Returns nothing.
    max_error = gradient_checker_v2.max_error(
        *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(max_error, 1e-4)

  def test_gather_v2_simple(self):
    # [FUNCTION DOC]
    # Gathers elements [2, 0, 2, 4] (with a repeated index) from a 1-D float64 vector and checks
    # the gradient numerically.
    x = constant_op.constant([1., 2., 3., 4., 5.], dtype=dtypes.float64)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; gathers the fixed int32 indices from x.
      return array_ops.gather_v2(
          x, constant_op.constant([2, 0, 2, 4], dtype=dtypes.int32))

    self._testGrad(f, x)

  def test_gather_v2_more_index_dims(self):
    # [FUNCTION DOC]
    # Gathers from a 1-D vector using a 2-D index matrix and checks the gradient numerically.
    x = constant_op.constant([1., 2., 3., 4., 5.], dtype=dtypes.float64)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; gathers with a 2x2 int32 index matrix.
      return array_ops.gather_v2(
          x, constant_op.constant([[2, 0], [2, 4]], dtype=dtypes.int32))

    self._testGrad(f, x)

  def test_gather_v2_more_param_dims(self):
    # [FUNCTION DOC]
    # Gathers rows [1, 0] from a 2x2 matrix (axis 0) and checks the gradient numerically.
    x = constant_op.constant([[1., 2.], [3., 4.]], dtype=dtypes.float64)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; gathers rows with int32 indices.
      return array_ops.gather_v2(
          x, constant_op.constant([1, 0], dtype=dtypes.int32))

    self._testGrad(f, x)

  def test_gather_v2_axis(self):
    # [FUNCTION DOC]
    # Gathers columns [1, 0] (axis 1) from a 2x2 matrix, exercising the non-zero-axis path of the
    # gather gradient, and checks the gradient numerically.
    x = constant_op.constant([[1., 2.], [3., 4.]], dtype=dtypes.float64)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; gathers along axis 1.
      return array_ops.gather_v2(
          x, constant_op.constant([1, 0], dtype=dtypes.int32), axis=1)

    self._testGrad(f, x)

  def test_gather_v2_batch_dims(self):
    # [FUNCTION DOC]
    # Gathers along axis 1 with one batch dimension from a 2x2 matrix using a 2x2 index matrix
    # and checks the gradient numerically.
    x = constant_op.constant([[1., 2.], [3., 4.]], dtype=dtypes.float64)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; batched gather with axis 1 and batch_dims 1.
      return array_ops.gather_v2(
          x,
          constant_op.constant([[1, 0], [0, 0]], dtype=dtypes.int32),
          axis=1,
          batch_dims=1)

    self._testGrad(f, x)

  def test_gather_v2_2batch_dims(self):
    # [FUNCTION DOC]
    # Gathers along axis 2 with two batch dimensions from a 1x2x2 tensor and checks the gradient
    # numerically.
    x = constant_op.constant([[[1., 2.], [3., 4.]]], dtype=dtypes.float64)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; batched gather with axis 2 and batch_dims 2.
      return array_ops.gather_v2(
          x,
          constant_op.constant([[[1, 0], [0, 0]]], dtype=dtypes.int32),
          axis=2,
          batch_dims=2)

    self._testGrad(f, x)

  def test_gather_v2_batch_dims_with_axis(self):
    # [FUNCTION DOC]
    # Gathers along axis 2 with one batch dimension from a 2x1x2 tensor (axis differs from the
    # first non-batch dimension) and checks the gradient numerically.
    x = constant_op.constant([[[1., 2.]], [[3., 4.]]], dtype=dtypes.float64)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; gather with axis 2 and batch_dims 1.
      return array_ops.gather_v2(
          x,
          constant_op.constant([[0], [0]], dtype=dtypes.int32),
          axis=2,
          batch_dims=1)

    self._testGrad(f, x)

  def test_gather_v2_zero_bsize_grad_has_matching_shapes(self):
    # [FUNCTION DOC]
    # Regression test for zero-size batches: gathers with axis 2 and two batch dims from an
    # all-zeros params tensor of shape 0x1x8x16 with 0x1x3 indices, computes the gradient with
    # the eager gradients function, and asserts there is exactly one gradient whose shape equals
    # the params shape.
    params = array_ops.zeros(shape=[0, 1, 8, 16], dtype=dtypes.float64)
    indices = array_ops.zeros(shape=[0, 1, 3], dtype=dtypes.int32)

    def f(params):
      # [FUNCTION DOC]
      # HELPER: function under test; batched gather on the closed-over indices.
      return array_ops.gather_v2(params, indices, axis=2, batch_dims=2)

    grads = backprop.gradients_function(f)(params)
    self.assertLen(grads, 1)
    self.assertAllEqual(params.shape, grads[0].shape)

  def test_broadcast_to(self):
    # [FUNCTION DOC]
    # Broadcasts a length-3 vector to shape 2x3 using an int32 shape tensor and checks the
    # gradient numerically.
    x = constant_op.constant([1., 2., 3.], dtype=dtypes.float64)
    y = constant_op.constant([2, 3], dtype=dtypes.int32)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; broadcasts x to the closed-over shape.
      return array_ops.broadcast_to(
          x,
          y)

    self._testGrad(f, x)

  def test_broadcast_to_int64(self):
    # [FUNCTION DOC]
    # Same as the broadcast test but with an int64 shape tensor, verifying the gradient handles
    # the shape dtype.
    x = constant_op.constant([1., 2., 3.], dtype=dtypes.float64)
    y = constant_op.constant([2, 3], dtype=dtypes.int64)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; broadcasts x to the closed-over int64 shape.
      return array_ops.broadcast_to(
          x,
          y)

    self._testGrad(f, x)

  def test_slice_int64(self):
    # [FUNCTION DOC]
    # Slices one element at offset 1 from a vector using int64 begin and size tensors and checks
    # the gradient numerically.
    x = constant_op.constant([1., 2., 3.], dtype=dtypes.float64)
    begin = constant_op.constant([1], dtype=dtypes.int64)
    size = constant_op.constant([1], dtype=dtypes.int64)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; slices x with the closed-over begin and size.
      return array_ops.slice(x, begin, size)

    self._testGrad(f, x)

  def test_reshape_simple(self):
    # [FUNCTION DOC]
    # Reshapes a length-3 vector to 3x1 using an int64 shape tensor and checks the gradient
    # numerically.
    x = constant_op.constant([1., 2., 3.], dtype=dtypes.float64)
    y = constant_op.constant([3, 1], dtype=dtypes.int64)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; reshapes x to the closed-over shape.
      return array_ops.reshape(x, y)

    self._testGrad(f, x)

  def test_reshape_one_unknown_dim(self):
    # [FUNCTION DOC]
    # Reshapes a 3x2 matrix whose first dimension is hidden (unknown) by a placeholder with
    # default, exercising the wildcard-dimension path of the reshape gradient; checks the
    # gradient numerically.
    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; hides the first dimension, then reshapes to 3x2.
      x_without_shape = array_ops.placeholder_with_default(x, shape=[None, 2])
      return array_ops.reshape(x_without_shape, [3, 2])

    x = constant_op.constant([[1., 2.], [3., 4.], [5., 6.]],
                             dtype=dtypes.float64)
    self._testGrad(f, x)

  def test_reshape_two_unknown_dims(self):
    # [FUNCTION DOC]
    # Reshapes a 2x3 matrix with both dimensions hidden (known rank, unknown sizes) to a vector
    # of 6, exercising the dynamic-shape path of the reshape gradient; checks the gradient
    # numerically.
    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; hides both dimensions, then reshapes to length 6.
      x_without_shape = array_ops.placeholder_with_default(x,
                                                           shape=[None, None])
      return array_ops.reshape(x_without_shape, [6])

    x = constant_op.constant([[1., 2., 3.], [4., 5., 6.]], dtype=dtypes.float64)
    self._testGrad(f, x)

  def test_reshape_one_unknown_dim_and_zero_elements(self):
    # [FUNCTION DOC]
    # Reshapes a 3x0 (zero-element) tensor whose first dimension is hidden, to an empty vector;
    # the gradient must avoid the wildcard dimension in this case. Checks the gradient
    # numerically.
    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; hides the first dimension, then reshapes to length 0.
      x_without_shape = array_ops.placeholder_with_default(x, shape=[None, 0])
      return array_ops.reshape(x_without_shape, [0])

    x = constant_op.constant([], shape=[3, 0], dtype=dtypes.float64)
    self._testGrad(f, x)


if __name__ == "__main__":
  test.main()
