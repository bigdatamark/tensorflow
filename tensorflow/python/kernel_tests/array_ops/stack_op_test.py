# [MODULE DOC]
# Functional tests for the stack and parallel-stack operations and for automatic stacking of
# nested Python lists in tensor conversion. Covers many dtypes (including quantized and
# complex), negative axes, constant-folded stacks, CPU and GPU parallel stacking (graph mode
# only), zero-size inputs, gradients along two axes, comparison with NumPy, error messages for
# out-of-range axes and mismatched shapes, and dtype handling when converting nested lists that
# mix Python numbers, tensors and NumPy arrays.
# Defines a NumPy split-and-squeeze helper function. No import-time side effects.
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
"""Functional tests for Stack and ParallelStack Ops."""

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def np_split_squeeze(array, axis):
  # [FUNCTION DOC]
  # HELPER: test utility producing the inputs for stacking tests.
  # Inputs: array (NumPy array) and axis (integer, may be negative). Splits the array into one
  # piece per element along that axis and removes that axis from each piece. Returns a list of
  # NumPy arrays with one fewer dimension. No side effects.
  axis_len = array.shape[axis]
  return [
      np.squeeze(
          arr, axis=(axis,)) for arr in np.split(
              array, axis_len, axis=axis)
  ]


class StackOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the stack and parallel-stack operations. Most tests build NumPy data, split it
  # into pieces with the module helper, stack the pieces back (using sub-tests per shape, axis and
  # dtype) and compare with the original. Holds no state apart from a random data helper method.

  def randn(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: random data generator. Inputs: shape (sequence of sizes) and dtype (NumPy dtype).
    # Draws standard normal values; for the boolean dtype returns whether each value is negative
    # (a roughly even mix of true and false), otherwise returns the values cast to the dtype.
    # Returns a NumPy array. Uses the global NumPy random state.
    data = np.random.randn(*shape)
    if dtype == np.bool_:
      return data < 0  # Naive casting yields True with P(1)!
    else:
      return data.astype(dtype)

  def testSimple(self):
    # [FUNCTION DOC]
    # Test: for several shapes, every axis (including negative) and dtypes bool, float32, int32,
    # int64 and two 8-bit float types, splits random data along the axis, stacks the pieces back
    # along it, and asserts equality with the original. Seeds the global random generator.
    # Returns nothing.
    np.random.seed(7)
    for shape in (2,), (3,), (2, 3), (3, 2), (8, 2, 10):
      rank = len(shape)
      for axis in range(-rank, rank):
        for dtype in [
            np.bool_,
            np.float32,
            np.int32,
            np.int64,
            dtypes.float8_e5m2.as_numpy_dtype,
            dtypes.float8_e4m3fn.as_numpy_dtype,
        ]:
          data = self.randn(shape, dtype)
          xs = np_split_squeeze(data, axis)
          # Stack back into a single tensorflow tensor
          with self.subTest(shape=shape, axis=axis, dtype=dtype):
            c = array_ops_stack.stack(xs, axis=axis)
            self.assertAllEqual(c, data)

  def testSimpleParallelCPU(self):
    # [FUNCTION DOC]
    # Test (graph mode only): parallel-stacks lists of per-row constants of several shapes on the
    # CPU and asserts the result equals the original float32 data. Seeds the random generator.
    # Returns nothing.
    # tf.parallel_stack is only supported in graph mode.
    with ops.Graph().as_default():
      np.random.seed(7)
      with test_util.device(use_gpu=False):
        for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2), (100, 24, 24, 3):
          with self.subTest(shape=shape):
            data = self.randn(shape, np.float32)
            xs = list(map(constant_op.constant, data))
            c = array_ops.parallel_stack(xs)
            self.assertAllEqual(c, data)

  def testParallelConcatShapeZero(self):
    # [FUNCTION DOC]
    # Test: skipped unless TF2 behavior is enabled. A compiled function that calls the raw
    # parallel-concat op with a shape of zero must raise an invalid-argument error saying the
    # zeroth dimension must be greater than zero. Returns nothing.
    if not tf2.enabled():
      self.skipTest("only fails in TF2")

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Nested function compiled into a graph function: calls the raw parallel-concat op on a
      # single string element with shape zero and returns its output. Takes no inputs; raises the
      # op's invalid-argument error when called.
      y = gen_array_ops.parallel_concat(values=[["tf"]], shape=0)
      return y

    with self.assertRaisesRegex(
        errors.InvalidArgumentError, r"0th dimension .* must be greater than"
    ):
      f()

  def testSimpleParallelGPU(self):
    # [FUNCTION DOC]
    # Test (graph mode only): same as the CPU parallel-stack test but on a GPU-preferred device.
    # Returns nothing.
    # tf.parallel_stack is only supported in graph mode.
    with ops.Graph().as_default():
      with test_util.device(use_gpu=True):
        for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2), (100, 24, 24, 3):
          with self.subTest(shape=shape):
            data = self.randn(shape, np.float32)
            xs = list(map(constant_op.constant, data))
            c = array_ops.parallel_stack(xs)
            self.assertAllEqual(c, data)

  def testConst(self):
    # [FUNCTION DOC]
    # Test: stacking constants. Checks that a stacked constant list can drive shape inference for
    # a reshape (result shape 2x3). For several shapes and dtypes, stacks a NumPy array (and a
    # Python list for 1-D data) and asserts equality with the data; in graph mode also asserts the
    # result is produced by a constant operation (constant folding). Runs on a GPU-preferred
    # device. Returns nothing.
    np.random.seed(7)
    with test_util.use_gpu():
      # Verify that shape induction works with shapes produced via const stack
      a = constant_op.constant([1, 2, 3, 4, 5, 6])
      b = array_ops.reshape(a, array_ops_stack.stack([2, 3]))
      self.assertAllEqual(b.get_shape(), [2, 3])

      # Check on a variety of shapes and types
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2), (8, 2, 10):
        for dtype in [np.bool_, np.float32, np.int16, np.int32, np.int64]:
          with self.subTest(shape=shape, dtype=dtype):
            data = self.randn(shape, dtype)
            # Stack back into a single tensorflow tensor directly using np array
            c = array_ops_stack.stack(data)
            if not context.executing_eagerly():
              # This is implemented via a Const:
              self.assertEqual(c.op.type, "Const")
            self.assertAllEqual(c, data)

            # Python lists also work for 1-D case:
            if len(shape) == 1:
              data_list = list(data)
              cl = array_ops_stack.stack(data_list)
              if not context.executing_eagerly():
                self.assertEqual(cl.op.type, "Const")
              self.assertAllEqual(cl, data)

  def testConstParallelCPU(self):
    # [FUNCTION DOC]
    # Test (graph mode only): parallel-stacking constant NumPy data (and a list of elements for
    # 1-D data) on the CPU and asserting equality with the data. Returns nothing.
    # tf.parallel_stack is only supported in graph mode.
    with ops.Graph().as_default():
      np.random.seed(7)
      with test_util.device(use_gpu=False):
        for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2), (8, 2, 10):
          with self.subTest(shape=shape):
            data = self.randn(shape, np.float32)
            if len(shape) == 1:
              data_list = list(data)
              cl = array_ops.parallel_stack(data_list)
              self.assertAllEqual(cl, data)

            data = self.randn(shape, np.float32)
            c = array_ops.parallel_stack(data)
            self.assertAllEqual(c, data)

  def testConstParallelGPU(self):
    # [FUNCTION DOC]
    # Test (graph mode only): same as the CPU constant parallel-stack test but on a GPU-preferred
    # device and with one fewer shape. Returns nothing.
    # tf.parallel_stack is only supported in graph mode.
    with ops.Graph().as_default():
      np.random.seed(7)
      with test_util.device(use_gpu=True):
        for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
          with self.subTest(shape=shape):
            data = self.randn(shape, np.float32)
            if len(shape) == 1:
              data_list = list(data)
              cl = array_ops.parallel_stack(data_list)
              self.assertAllEqual(cl, data)

            data = self.randn(shape, np.float32)
            c = array_ops.parallel_stack(data)
            self.assertAllEqual(c, data)

  def testGradientsAxis0(self):
    # [FUNCTION DOC]
    # Test: numerical gradient check of stacking along axis 0 for several shapes; compares
    # theoretical and numerical Jacobians with the default tolerance. Returns nothing.
    np.random.seed(7)
    for shape in (2,), (3,), (2, 3), (3, 2), (8, 2, 10):
      data = np.random.randn(*shape)
      with self.subTest(shape=shape):
        with self.cached_session():

          def func(*xs):
            # [FUNCTION DOC]
            # HELPER: function under test for the gradient checker; stacks all its tensor
            # arguments along axis 0 and returns the stacked tensor.
            return array_ops_stack.stack(xs)
          # TODO(irving): Remove list() once we handle maps correctly
          xs = list(map(constant_op.constant, data))
          theoretical, numerical = gradient_checker_v2.compute_gradient(
              func, xs)
          self.assertAllClose(theoretical, numerical)

  def testGradientsAxis1(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the computed expected output shape variable is never used.
    # Test: numerical gradient check of stacking along axis 1 for several shapes; compares
    # theoretical and numerical Jacobians. Returns nothing.
    np.random.seed(7)
    for shape in (2, 3), (3, 2), (8, 2, 10):
      data = np.random.randn(*shape)
      out_shape = list(shape[1:])
      out_shape.insert(1, shape[0])
      with self.subTest(shape=shape):
        with self.cached_session():

          def func(*inp):
            # [FUNCTION DOC]
            # HELPER: function under test for the gradient checker; stacks all its tensor
            # arguments along axis 1 and returns the stacked tensor.
            return array_ops_stack.stack(inp, axis=1)
          # TODO(irving): Remove list() once we handle maps correctly
          xs = list(map(constant_op.constant, data))
          theoretical, numerical = gradient_checker_v2.compute_gradient(
              func, xs)
          self.assertAllClose(theoretical, numerical)

  def testZeroSizeCPU(self):
    # [FUNCTION DOC]
    # Test (graph mode only): on the CPU, stacking and parallel-stacking two zero-size integer
    # arrays of shapes with a zero dimension must not crash and must equal the input. Returns
    # nothing.
    # tf.parallel_stack is only supported in graph mode.
    with ops.Graph().as_default():
      # Verify that stack doesn't crash for zero size inputs
      with test_util.device(use_gpu=False):
        for shape in (0,), (3, 0), (0, 3):
          with self.subTest(shape=shape):
            x = np.zeros((2,) + shape).astype(np.int32)
            p = self.evaluate(array_ops_stack.stack(list(x)))
            self.assertAllEqual(p, x)

            p = self.evaluate(array_ops.parallel_stack(list(x)))
            self.assertAllEqual(p, x)

  def testZeroSizeGPU(self):
    # [FUNCTION DOC]
    # Test (graph mode only): same as the CPU zero-size test but on a GPU-preferred device.
    # Returns nothing.
    # tf.parallel_stack is only supported in graph mode.
    with ops.Graph().as_default():
      # Verify that stack doesn't crash for zero size inputs
      with test_util.device(use_gpu=True):
        for shape in (0,), (3, 0), (0, 3):
          with self.subTest(shape=shape):
            x = np.zeros((2,) + shape).astype(np.int32)
            p = self.evaluate(array_ops_stack.stack(list(x)))
            self.assertAllEqual(p, x)

            p = self.evaluate(array_ops.parallel_stack(list(x)))
            self.assertAllEqual(p, x)

  def testAxis0DefaultCPU(self):
    # [FUNCTION DOC]
    # Test (graph mode only): on the CPU, stacking and parallel-stacking two constant vectors
    # without an axis gives the same 2x3 matrix (stack along axis 0). Returns nothing.
    # tf.parallel_stack is only supported in graph mode.
    with ops.Graph().as_default():
      with test_util.device(use_gpu=False):
        t = [constant_op.constant([1, 2, 3]), constant_op.constant([4, 5, 6])]
        stacked = self.evaluate(array_ops_stack.stack(t))
        parallel_stacked = self.evaluate(array_ops.parallel_stack(t))

      expected = np.array([[1, 2, 3], [4, 5, 6]])
      self.assertAllEqual(stacked, expected)
      self.assertAllEqual(parallel_stacked, expected)

  def testAxis0DefaultGPU(self):
    # [FUNCTION DOC]
    # Test (graph mode only): same as the CPU default-axis test but on a GPU-preferred device.
    # Returns nothing.
    # tf.parallel_stack is only supported in graph mode.
    with ops.Graph().as_default():
      with test_util.device(use_gpu=True):
        t = [constant_op.constant([1, 2, 3]), constant_op.constant([4, 5, 6])]
        stacked = self.evaluate(array_ops_stack.stack(t))
        parallel_stacked = self.evaluate(array_ops.parallel_stack(t))

      expected = np.array([[1, 2, 3], [4, 5, 6]])
      self.assertAllEqual(stacked, expected)
      self.assertAllEqual(parallel_stacked, expected)

  def testAgainstNumpy(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variables named pack and stack both come from the same stack function
    # called with identical arguments (leftover from a pack/stack rename), so the second check
    # is redundant, and the dtype loop does not actually change the data (always float32).
    # Test: for several shapes and every axis (including negative), splits random float32 data,
    # stacks the pieces, asserts the static shape equals the original shape, evaluates, and
    # asserts the values are nearly equal (tolerance 1e-6). Returns nothing.
    # For 1 to 5 dimensions.
    for shape in (3,), (2, 2, 3), (4, 1, 2, 2), (8, 2, 10):
      rank = len(shape)
      expected = self.randn(shape, np.float32)
      for dtype in [np.bool_, np.float32, np.int32, np.int64]:
        # For all the possible axis to split it, including negative indices.
        for axis in range(-rank, rank):
          test_arrays = np_split_squeeze(expected, axis)

          with self.cached_session():
            with self.subTest(shape=shape, dtype=dtype, axis=axis):
              actual_pack = array_ops_stack.stack(test_arrays, axis=axis)
              self.assertEqual(expected.shape, actual_pack.get_shape())
              actual_pack = self.evaluate(actual_pack)

              actual_stack = array_ops_stack.stack(test_arrays, axis=axis)
              self.assertEqual(expected.shape, actual_stack.get_shape())
              actual_stack = self.evaluate(actual_stack)

              self.assertNDArrayNear(expected, actual_stack, 1e-6)

  def testDimOutOfRange(self):
    # [FUNCTION DOC]
    # Test: stacking two vectors along axis 2 raises a value error saying the axis is not in the
    # allowed half-open range from minus 2 up to 2. Returns nothing.
    t = [constant_op.constant([1, 2, 3]), constant_op.constant([4, 5, 6])]
    with self.assertRaisesRegex(ValueError,
                                r"Argument `axis` = 2 not in range \[-2, 2\)"):
      array_ops_stack.stack(t, axis=2)

  def testDimOutOfNegativeRange(self):
    # [FUNCTION DOC]
    # Test: stacking two vectors along axis minus 3 raises a value error saying the axis is not in
    # the allowed range from minus 2 up to 2. Returns nothing.
    t = [constant_op.constant([1, 2, 3]), constant_op.constant([4, 5, 6])]
    with self.assertRaisesRegex(ValueError,
                                r"Argument `axis` = -3 not in range \[-2, 2\)"):
      array_ops_stack.stack(t, axis=-3)

  def testComplex(self):
    # [FUNCTION DOC]
    # Test: stacking per-row constants of complex64 and complex128 data of several shapes
    # reproduces the data. Returns nothing.
    np.random.seed(7)
    with self.session():
      for shape in (2,), (3,), (2, 3), (3, 2), (8, 2, 10):
        for dtype in [np.complex64, np.complex128]:
          with self.subTest(shape=shape, dtype=dtype):
            data = self.randn(shape, dtype)
            xs = list(map(constant_op.constant, data))
            c = array_ops_stack.stack(xs)
            self.assertAllEqual(self.evaluate(c), data)

  def testZeroDimUnmatch(self):
    # [FUNCTION DOC]
    # Test (regression for a GitHub issue): stacking a 0x3 and a 1x3 tensor must raise an
    # invalid-argument or value error whose message mentions shapes (the exact text differs
    # between eager and graph mode). Returns nothing.
    # Test case for GitHub issue 53300.
    # Error message is `Shapes of all inputs must match` in eager mode,
    # and `Shapes ...` in graph mode. Below is to capture both:
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                r"Shapes"):
      with self.session():
        t = [array_ops.zeros([0, 3]), array_ops.zeros([1, 3])]
        self.evaluate(array_ops_stack.stack(t))

  def testQTypes(self):
    # [FUNCTION DOC]
    # Test: stacks two-element constants of each quantized type (quint8, quint16, qint8, qint16,
    # qint32), then checks element-wise equality with the original data gives all true. Runs on a
    # GPU-preferred session. Returns nothing.
    np.random.seed(7)
    with self.session(use_gpu=True):
      shape = [2]
      for dtype in [
          dtypes.quint8, dtypes.quint16, dtypes.qint8, dtypes.qint16,
          dtypes.qint32
      ]:
        with self.subTest(shape=shape, dtype=dtype):
          data = self.randn(shape, dtype.as_numpy_dtype)
          xs = list(map(constant_op.constant, data))
          c = math_ops.equal(array_ops_stack.stack(xs), data)
          self.assertAllEqual(self.evaluate(c), [True, True])


class AutomaticStackingTest(test.TestCase):
  # [CLASS DOC]
  # Tests the automatic stacking performed by the generic tensor conversion function when given
  # nested Python lists that mix numbers, tensors and NumPy arrays, including result dtype rules.

  def testSimple(self):
    # [FUNCTION DOC]
    # Test: converting nested lists that contain scalar tensors or vector tensors in various
    # positions yields the same dense tensors as the all-Python-number lists. Returns nothing.
    self.assertAllEqual([1, 0, 2],
                        ops.convert_to_tensor([1, constant_op.constant(0), 2]))
    self.assertAllEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        ops.convert_to_tensor([[0, 0, 0],
                                               [0,
                                                constant_op.constant(1), 0],
                                               [0, 0, 0]]))
    self.assertAllEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        ops.convert_to_tensor([[0, 0, 0],
                                               constant_op.constant([0, 1, 0]),
                                               [0, 0, 0]]))
    self.assertAllEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        ops.convert_to_tensor([
                            constant_op.constant([0, 0, 0]),
                            constant_op.constant([0, 1, 0]),
                            constant_op.constant([0, 0, 0])
                        ]))

  def testWithNDArray(self):
    # [FUNCTION DOC]
    # Test: converting a nested list that mixes Python numbers, a tensor and a NumPy array gives
    # the expected 2x2x2 tensor. Returns nothing.
    with self.session():
      result = ops.convert_to_tensor([[[0., 0.],
                                       constant_op.constant([1., 1.])],
                                      np.array(
                                          [[2., 2.], [3., 3.]],
                                          dtype=np.float32)])
      self.assertAllEqual([[[0., 0.], [1., 1.]], [[2., 2.], [3., 3.]]],
                          self.evaluate(result))

  def testDtype(self):
    # [FUNCTION DOC]
    # Test: dtype rules of list conversion. Plain floats give 32-bit float; a list containing a
    # 64-bit float tensor gives 64-bit float; an explicit requested dtype wins (including
    # narrowing a 64-bit tensor to 32-bit); a list mixing 32-bit and 64-bit float tensors without
    # a requested dtype raises a type error. Returns nothing.
    t_0 = ops.convert_to_tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    self.assertEqual(dtypes.float32, t_0.dtype)

    t_1 = ops.convert_to_tensor([[0., 0., 0.], constant_op.constant(
        [0., 0., 0.], dtype=dtypes.float64), [0., 0., 0.]])
    self.assertEqual(dtypes.float64, t_1.dtype)

    t_2 = ops.convert_to_tensor(
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, t_2.dtype)

    t_3 = ops.convert_to_tensor(
        [[0., 0., 0.],
         constant_op.constant([0., 0., 0.], dtype=dtypes.float64), [0., 0., 0.]
        ],
        dtype=dtypes.float32)
    self.assertEqual(dtypes.float32, t_3.dtype)

    t_4 = ops.convert_to_tensor(
        [constant_op.constant([0., 0., 0.], dtype=dtypes.float64)],
        dtype=dtypes.float32)
    self.assertEqual(dtypes.float32, t_4.dtype)

    with self.assertRaises(TypeError):
      ops.convert_to_tensor([
          constant_op.constant(
              [0., 0., 0.], dtype=dtypes.float32), constant_op.constant(
                  [0., 0., 0.], dtype=dtypes.float64), [0., 0., 0.]
      ])

  def testDtypeConversionWhenTensorDtypeMismatch(self):
    # [FUNCTION DOC]
    # Test: converts a float32 vector and an int32 vector to tensors with expected dtypes, then
    # converts a list holding the float tensor twice and the int tensor with a requested 64-bit
    # float dtype and asserts the result dtype is 64-bit float (mismatched tensor dtypes are
    # converted). Returns nothing.
    t_0 = ops.convert_to_tensor([0., 0., 0.])
    self.assertEqual(dtypes.float32, t_0.dtype)

    t_1 = ops.convert_to_tensor([0, 0, 0])
    self.assertEqual(dtypes.int32, t_1.dtype)

    t_2 = ops.convert_to_tensor([t_0, t_0, t_1], dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, t_2.dtype)


if __name__ == "__main__":
  test.main()
