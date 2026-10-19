# [MODULE DOC]
# Unit tests for the broadcast-to op. Compares TensorFlow results with numpy broadcasting for many
# numeric types, strings, booleans and scalars, checks shape inference with 32/64-bit shape
# tensors, verifies error reporting for incompatible or huge target shapes, and runs numerical
# gradient checks (scalar, same rank, increasing rank, all dimensions broadcast, many
# dimensions). Defines one test case class; no import-time side effects.
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
"""Tests for broadcast_to ops."""
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test as test_lib


class BroadcastToTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the broadcast-to op, covering forward values against numpy, shape inference,
  # error handling and gradients. Holds no shared state; each test builds its own inputs.

  def testBroadcastToBasic(self):
    # [FUNCTION DOC]
    # For each of twelve numeric dtypes (unsigned and signed integers, three float widths,
    # bfloat16 and two 8-bit float formats), broadcasts a length-3 vector to 3x3 and asserts it
    # equals numpy's result. Returns nothing.
    for dtype in [
        np.uint8,
        np.uint16,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
        dtypes.float8_e5m2.as_numpy_dtype,
        dtypes.float8_e4m3fn.as_numpy_dtype,
    ]:
      with self.session():
        x = np.array([1, 2, 3], dtype=dtype)
        v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf, v_np)

  def testBroadcastToString(self):
    # [FUNCTION DOC]
    # Broadcasts a length-3 vector of byte strings to 3x3 and asserts equality with numpy.
    # Returns nothing.
    with self.session():
      x = np.array([b"1", b"2", b"3"])
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToBool(self):
    # [FUNCTION DOC]
    # Broadcasts a length-3 boolean vector to 3x3 and asserts equality with numpy. Returns nothing.
    with self.session():
      x = np.array([True, False, True], dtype=np.bool_)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToShape(self):
    # [FUNCTION DOC]
    # For every input rank from 1 to 5 and every output rank from the input rank to 5, builds a
    # random integer array with all dimensions 2, broadcasts it to the all-2 output shape and
    # asserts equality with numpy. Returns nothing.
    for input_dim in range(1, 6):
      for output_dim in range(input_dim, 6):
        with self.cached_session():
          input_shape = [2] * input_dim
          output_shape = [2] * output_dim
          x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
          v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
          v_np = np.broadcast_to(x, output_shape)
          self.assertAllEqual(v_tf, v_np)

  def testBroadcastToShapeInnerDim(self):
    # [FUNCTION DOC]
    # Broadcasts a random 2x1x3 integer array to 2x5x3 (an inner dimension expanded) and asserts
    # equality with numpy. Returns nothing.
    input_shape = [2, 1, 3]
    output_shape = [2, 5, 3]
    with self.cached_session():
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToShapeLargerDim(self):
    # [FUNCTION DOC]
    # Broadcasts a random rank-6 integer array to a rank-15 shape with many leading ones and an
    # expanded middle dimension, and asserts equality with numpy. Returns nothing.
    input_shape = [2, 1, 3, 2, 2, 2]
    output_shape = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 15, 3, 2, 2, 2]
    with self.cached_session():
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToShapeLargerDim2(self):
    # [FUNCTION DOC]
    # Broadcasts a random rank-9 integer array with trailing ones to a rank-12 shape, expanding
    # several dimensions, and asserts equality with numpy. Returns nothing.
    input_shape = [2, 1, 3, 2, 2, 2, 1, 1, 1]
    output_shape = [1, 1, 1, 2, 5, 3, 2, 2, 2, 3, 3, 3]
    with self.cached_session():
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToScalar(self):
    # [FUNCTION DOC]
    # Broadcasts a rank-0 integer scalar to 3x3 and asserts equality with numpy. Returns nothing.
    with self.session():
      x = np.array(1, dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastScalarToNonScalar(self):
    # [FUNCTION DOC]
    # Broadcasts a float scalar constant to the rank-6 shape 2x3x4x1x1x1 and compares with
    # numpy's broadcast of the equal double-precision scalar. Returns nothing.
    with self.session():
      x = np.array(1.0, dtype=np.float64)
      v_tf = array_ops.broadcast_to(
          constant_op.constant(1.0), [2, 3, 4, 1, 1, 1])
      v_np = np.broadcast_to(x, [2, 3, 4, 1, 1, 1])
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToShapeTypeAndInference(self):
    # [FUNCTION DOC]
    # For both 32-bit and 64-bit integer target shape tensors, broadcasts a length-3 vector to
    # 3x3, asserts the values match numpy and that the statically inferred shape (known because
    # the shape input is constant) equals the numpy shape. Returns nothing.
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.cached_session():
        x = np.array([1, 2, 3])
        v_tf = array_ops.broadcast_to(
            constant_op.constant(x), constant_op.constant([3, 3], dtype=dtype))
        shape = v_tf.get_shape().as_list()
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf, v_np)
        # check shape inference when shape input is constant
        self.assertAllEqual(shape, v_np.shape)

  def testBroadcastToBadOutputShape(self):
    # [FUNCTION DOC]
    # In eager mode, tries to broadcast a length-2 vector to shape 2x1 (incompatible) and asserts
    # an invalid-argument error whose message says it was unable to broadcast. Returns nothing.
    with context.eager_mode():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Unable to broadcast tensor of shape"):
        self.evaluate(
            array_ops.broadcast_to(
                constant_op.constant([0, 1]), constant_op.constant([2, 1])))

  def testGradientForScalar(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient of twice a scalar broadcast to 2x4x3 using the version-2
    # gradient checker, and asserts the maximum error is below 1e-4. Returns nothing.
    x = constant_op.constant(1, dtype=dtypes.float32)

    def func(x):
      # [FUNCTION DOC]
      # HELPER: function under differentiation. Broadcasts x to 2x4x3 and returns twice the
      # result. No side effects.
      v = array_ops.broadcast_to(x, [2, 4, 3])
      return 2 * v

    with self.cached_session():
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(func, [x]))

    self.assertLess(err, 1e-4)

  def testGradientWithSameRank(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient (step 1e-2) of twice a 2x1x3 input broadcast to 2x5x3
    # (same rank) and asserts the maximum error is below 1e-4. Returns nothing.
    x = constant_op.constant(
        np.reshape(np.arange(6), (2, 1, 3)), dtype=dtypes.float32)

    def func(x):
      # [FUNCTION DOC]
      # HELPER: function under differentiation. Broadcasts x to 2x5x3 and returns twice the
      # result. No side effects.
      v = array_ops.broadcast_to(x, [2, 5, 3])
      return 2 * v

    with self.cached_session():
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(func, [x], delta=1e-2))

    self.assertLess(err, 1e-4)

  def testGradientWithIncreasingRank(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient of twice a 2x1 input broadcast to 5x2x3 (rank increases)
    # and asserts the maximum error is below 1e-4. Returns nothing.
    x = constant_op.constant([[1], [2]], dtype=dtypes.float32)

    def func(x):
      # [FUNCTION DOC]
      # HELPER: function under differentiation. Broadcasts x to 5x2x3 and returns twice the
      # result. No side effects.
      v = array_ops.broadcast_to(x, [5, 2, 3])
      return 2 * v

    with self.cached_session():
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(func, [x]))

    self.assertLess(err, 1e-4)

  def testGradientWithBroadcastAllDimensions(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient of twice a one-element vector broadcast across all
    # dimensions of 5x2x3 and asserts the maximum error is below 1e-4. Returns nothing.
    x = constant_op.constant([1], dtype=dtypes.float32)

    def func(x):
      # [FUNCTION DOC]
      # HELPER: function under differentiation. Broadcasts x to 5x2x3 and returns twice the
      # result. No side effects.
      v = array_ops.broadcast_to(x, [5, 2, 3])
      return 2 * v

    with self.cached_session():
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(func, [x]))

    self.assertLess(err, 1e-4)

  def testGradientWithLargeDim(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient (step 1e-2) of twice a random rank-9 input broadcast to a
    # rank-12 shape and asserts the maximum error is below 1e-4. Returns nothing.
    input_shape = [2, 1, 3, 2, 2, 2, 1, 1, 1]
    output_shape = [1, 1, 1, 2, 5, 3, 2, 2, 2, 3, 3, 3]
    x = constant_op.constant(
        np.array(np.random.randn(*input_shape), dtype=np.float32))

    def func(x):
      # [FUNCTION DOC]
      # HELPER: function under differentiation. Broadcasts x to the enclosing test's output shape
      # and returns twice the result. No side effects.
      v = array_ops.broadcast_to(x, output_shape)
      return 2 * v

    with self.cached_session():
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(func, [x], delta=1e-2))

    self.assertLess(err, 1e-4)

  def testBroadcastToInvalidShape(self):
    # [FUNCTION DOC]
    # Broadcasts a length-3 vector to an absurdly large twelve-dimensional shape and asserts a
    # value error or invalid-argument error whose message contains the requested shape.
    # Returns nothing.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "110,53,104,147,157,123,5,24,188,40,5,2"):
      output_shape = [110, 53, 104, 147, 157, 123, 5, 24, 188, 40, 5, 2]
      x = np.array([1, 2, 3], dtype=np.int32)
      v = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      self.evaluate(v)

  def testBroadcastToInvalidShapeForEmpty(self):
    # [FUNCTION DOC]
    # Broadcasts an empty 3x0x5 tensor to shape 3x0x3 (incompatible last dimension) and asserts
    # the framework's incompatible-shapes error (value error or invalid-argument error).
    # Returns nothing.
    with self.assertRaisesIncompatibleShapesError(
        (ValueError, errors.InvalidArgumentError)):
      output_shape = [3, 0, 3]
      x = constant_op.constant(value=[], shape=(3, 0, 5), dtype=np.int32)
      v = array_ops.broadcast_to(x, output_shape)
      self.evaluate(v)

if __name__ == "__main__":
  test_lib.main()
