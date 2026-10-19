# [MODULE DOC]
# Functional tests for the tensor slicing operation, covering both the explicit slice function and
# Python indexing/slicing syntax on tensors. Exercises empty slices, int32 and int64 indices,
# many dtypes, shape inference, error cases for scalar or out-of-range input, gradients in graph
# mode and with a gradient tape, and non-iterability of graph tensors.
# Defines one test case class and runs the test main entry point when executed as a script.
# No import-time side effects beyond imports.
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
"""Functional tests for slice op."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class SliceTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering tensor slicing through the slice function and Python indexing syntax.
  # Holds no state; each test builds its own random or constant inputs, compares results with
  # NumPy slicing, and also checks static shape inference and gradients (graph and tape modes).
  # Private helper methods with a leading underscore are shared by the gradient tests.

  def testEmpty(self):
    # [FUNCTION DOC]
    # Test: slicing a random 4x4 float matrix with a fixed row and an empty column range
    # (start equals stop) for each start in 0..3. Asserts the evaluated tensor equals the
    # equivalent empty NumPy slice. Returns nothing.
    inp = np.random.rand(4, 4).astype("f")
    for k in range(4):
      with self.cached_session():
        a = constant_op.constant(inp, shape=[4, 4], dtype=dtypes.float32)
        slice_t = a[2, k:k]
        slice_val = self.evaluate(slice_t)
      self.assertAllEqual(slice_val, inp[2, k:k])

  def testInt32(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the input is made by casting random floats in [0, 1) to a 32-bit integer
    # type, so every element is zero and the comparison is weak.
    # Test: same as the empty-slice test but with an int32 constant; slices one row with an empty
    # column range and asserts equality with the NumPy slice. Returns nothing.
    inp = np.random.rand(4, 4).astype("i")
    for k in range(4):
      with self.cached_session():
        a = constant_op.constant(inp, shape=[4, 4], dtype=dtypes.int32)
        slice_t = a[2, k:k]
        slice_val = self.evaluate(slice_t)
      self.assertAllEqual(slice_val, inp[2, k:k])

  def testSlicingWithInt64Index(self):
    # [FUNCTION DOC]
    # Test: slicing an int32 vector using int64 indices, both as an int64 tensor and as a NumPy
    # int64 scalar, through indexing and range syntax; also calls the slice function with int64
    # begin/size arrays on int32 and float32 vectors. Asserts the expected element or sub-vector
    # each time. Uses a GPU session if one is available. Returns nothing.
    with self.cached_session(force_gpu=test.is_gpu_available()):
      a = constant_op.constant([0, 1, 2], dtype=dtypes.int32)

      # Slice using int64 Tensor.
      i = constant_op.constant(1, dtype=dtypes.int64)
      slice_t = a[i]
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual(1, slice_val)
      slice_t = a[i:i+1]
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([1], slice_val)

      # Slice using int64 integer.
      i = np.asarray(1).astype(np.int64)
      slice_t = a[i]
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual(1, slice_val)
      slice_t = a[i:i+1]
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([1], slice_val)

      a_int32 = constant_op.constant([0, 1, 2], dtype=dtypes.int32)
      slice_t = array_ops.slice(a_int32,
                                np.asarray([1]).astype(np.int64),
                                np.asarray([2]).astype(np.int64))
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([1, 2], slice_val)

      a_float32 = constant_op.constant([0, 1, 2], dtype=dtypes.float32)
      slice_t = array_ops.slice(a_float32,
                                np.asarray([1]).astype(np.int64),
                                np.asarray([2]).astype(np.int64))
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([1, 2], slice_val)

  def testSlicingInt64Tensor(self):
    # [FUNCTION DOC]
    # Test: slicing an int64 vector using int32 indices (tensor and NumPy scalar) via indexing and
    # range syntax, then with the slice function using plain lists. Asserts the expected element
    # or sub-vector each time. Uses a GPU session if one is available. Returns nothing.
    with self.cached_session(force_gpu=test.is_gpu_available()):
      a = constant_op.constant([0, 1, 2], dtype=dtypes.int64)

      # Slice using int32 Tensor.
      i = constant_op.constant(1, dtype=dtypes.int32)
      slice_t = a[i]
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual(1, slice_val)
      slice_t = a[i:i + 1]
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([1], slice_val)

      # Slice using int32 integer.
      i = np.asarray(1).astype(np.int32)
      slice_t = a[i]
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual(1, slice_val)
      slice_t = a[i:i + 1]
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([1], slice_val)

      slice_t = array_ops.slice(a, [1], [2])
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([1, 2], slice_val)

  def testSelectAll(self):
    # [FUNCTION DOC]
    # Test: ten times, takes a random 4-D float tensor and selects everything, once with the
    # slice function (begin zeros, size all minus one) and once with full-range indexing.
    # Asserts both results equal the input and that both static shapes equal the input shape.
    # Returns nothing.
    for _ in range(10):
      with self.cached_session():
        inp = np.random.rand(4, 4, 4, 4).astype("f")
        a = constant_op.constant(inp, shape=[4, 4, 4, 4], dtype=dtypes.float32)

        slice_explicit_t = array_ops.slice(a, [0, 0, 0, 0], [-1, -1, -1, -1])
        slice_implicit_t = a[:, :, :, :]

        self.assertAllEqual(inp, self.evaluate(slice_explicit_t))
        self.assertAllEqual(inp, self.evaluate(slice_implicit_t))
        self.assertEqual(inp.shape, slice_explicit_t.get_shape())
        self.assertEqual(inp.shape, slice_implicit_t.get_shape())

  def testSingleDimension(self):
    # [FUNCTION DOC]
    # Test: ten times, on a random length-10 float vector, takes a scalar element at a random
    # index and a range from a random lower bound to that index. Asserts both match NumPy
    # indexing and slicing. Returns nothing.
    for _ in range(10):
      with self.cached_session():
        inp = np.random.rand(10).astype("f")
        a = constant_op.constant(inp, shape=[10], dtype=dtypes.float32)

        hi = np.random.randint(0, 9)
        scalar_t = a[hi]
        scalar_val = self.evaluate(scalar_t)
        self.assertAllEqual(scalar_val, inp[hi])

        if hi > 0:
          lo = np.random.randint(0, hi)
        else:
          lo = 0
        slice_t = a[lo:hi]
        slice_val = self.evaluate(slice_t)
        self.assertAllEqual(slice_val, inp[lo:hi])

  @test_util.run_without_tensor_float_32("Use FP32 in conv3d.")
  def test3Dimension(self):
    # [FUNCTION DOC]
    # Test: builds a 5-D input and a 1x1x1 filter with deterministic ramp values, runs a 3-D
    # convolution (valid padding, unit strides), then slices a single position with all 8
    # channels from the result. Asserts the 8 values are close to hard-coded expected numbers
    # (relative tolerance 1e-6). The decorator forces full float32 precision (no TensorFloat-32).
    # Returns nothing.
    with self.cached_session():
      input_shape = [8, 16, 16, 16, 8]
      total_input_size = 1
      for s in input_shape:
        total_input_size *= s
      inputs = [
          i * 1.0 / total_input_size for i in range(1, total_input_size + 1)
      ]
      a = constant_op.constant(inputs, shape=input_shape, dtype=dtypes.float32)

      filter_shape = [1, 1, 1, 8, 8]
      total_filter_size = 1
      for s in filter_shape:
        total_filter_size *= s
      filters = [
          i * 1.0 / total_filter_size for i in range(1, total_filter_size + 1)
      ]
      f = constant_op.constant(
          filters, shape=filter_shape, dtype=dtypes.float32)

      conv_t = nn_ops.conv3d(
          a, filter=f, strides=[1, 1, 1, 1, 1], padding="VALID")
      slice_t = array_ops.slice(conv_t, [0, 1, 1, 1, 0], [1, 1, 1, 1, 8])
      result = self.evaluate(slice_t)
      expected = [
          0.03028321, 0.03132677, 0.03237033, 0.03341389, 0.03445745, 0.035501,
          0.03654456, 0.03758812
      ]
      self.assertAllClose(expected, result.flatten(), rtol=1e-6)

  def testScalarInput(self):
    # [FUNCTION DOC]
    # Test: slicing a scalar. A full-range slice of a constant scalar must raise a value error or
    # invalid-argument error mentioning an attempt to slice scalar input. Then a compiled
    # function that slices its argument, called with a plain Python integer, must raise a type
    # error saying the object is not subscriptable. Returns nothing.
    input_val = 0
    # Test with constant input; shape inference fails.
    with self.assertRaisesWithPredicateMatch(
        (ValueError, errors_impl.InvalidArgumentError),
        "Attempting to slice scalar input."):
      constant_op.constant(input_val)[:].get_shape()

    # Test evaluating with non-constant input; kernel execution fails.
    @def_function.function
    def func(input_t):
      # [FUNCTION DOC]
      # Nested function compiled into a graph function: takes one input and returns its
      # full-range slice. Used to check that slicing a Python integer argument raises a type
      # error. No side effects.
      slice_t = input_t[:]
      return slice_t

    with self.assertRaisesWithPredicateMatch(TypeError, "not subscriptable"):
      self.evaluate(func(input_val))

  def testInvalidIndex(self):
    # [FUNCTION DOC]
    # Test: a one-dimensional constant sliced with two range indices must raise a value error or
    # invalid-argument error mentioning out of range. A compiled function doing the same slice
    # on a plain Python list must raise a type error saying indices must be integers or slices,
    # not tuple. Returns nothing.
    input_val = [1, 2]
    # Test with constant input; shape inference fails.
    with self.assertRaisesWithPredicateMatch(
        (ValueError, errors_impl.InvalidArgumentError), "out of range"):
      constant_op.constant(input_val)[1:, 1:].get_shape()

    # Test evaluating with non-constant input; kernel execution fails.
    @def_function.function
    def func(input_t):
      # [FUNCTION DOC]
      # Nested function compiled into a graph function: takes one input and returns a
      # two-dimensional range slice of it. Used to check the type error raised when the
      # argument is a plain Python list. No side effects.
      slice_t = input_t[1:, 1:]
      return slice_t

    with self.assertRaisesWithPredicateMatch(
        TypeError, "must be integers or slices, not tuple"):
      self.evaluate(func(input_val))

  def _testSliceMatrixDim0(self, x, begin, size):
    # [FUNCTION DOC]
    # HELPER: shared check for the matrix first-dimension slice test.
    # Slices rows from begin for size rows (all columns) of the NumPy matrix x with the slice
    # function, and asserts the result equals the same NumPy row range. Returns nothing.
    tf_ans = self.evaluate(array_ops.slice(x, [begin, 0], [size, x.shape[1]]))
    np_ans = x[begin:begin + size, :]
    self.assertAllEqual(tf_ans, np_ans)

  def testSliceMatrixDim0(self):
    # [FUNCTION DOC]
    # Test: slices row ranges out of a random 8x4 and an 8x7 float matrix (the latter has a row
    # byte size not aligned to the element width) using the shared helper. Returns nothing.
    x = np.random.rand(8, 4).astype("f")
    self._testSliceMatrixDim0(x, 1, 2)
    self._testSliceMatrixDim0(x, 3, 3)
    y = np.random.rand(8, 7).astype("f")  # 7 * sizeof(float) is not aligned
    self._testSliceMatrixDim0(y, 1, 2)
    self._testSliceMatrixDim0(y, 3, 3)

  def testSingleElementAll(self):
    # [FUNCTION DOC]
    # Test: ten times, on a random 4x4 float matrix, indexes a random row with a column range from
    # zero to a random bound and asserts equality with the NumPy result. Returns nothing.
    for _ in range(10):
      with self.cached_session():
        inp = np.random.rand(4, 4).astype("f")
        a = constant_op.constant(inp, shape=[4, 4], dtype=dtypes.float32)

        x, y = np.random.randint(0, 3, size=2).tolist()
        slice_t = a[x, 0:y]
        slice_val = self.evaluate(slice_t)
      self.assertAllEqual(slice_val, inp[x, 0:y])

  def testSimple(self):
    # [FUNCTION DOC]
    # Test: for each of many numeric dtypes (integers, bool, floats, complex, bfloat16, float8
    # variants), builds a random 4x4 array of that dtype, feeds its values as float32, and takes
    # the top-left 2x2 block via the slice function and via indexing. Asserts both match the
    # NumPy block and that static shapes equal evaluated shapes. Runs on GPU if available.
    # Returns nothing.
    with test_util.use_gpu():
      for dtype in [
          np.uint8,
          np.int8,
          np.uint16,
          np.int16,
          np.int32,
          np.int64,
          np.bool_,
          np.float16,
          np.float32,
          np.float64,
          np.complex64,
          np.complex128,
          dtypes.bfloat16.as_numpy_dtype,
          dtypes.float8_e5m2.as_numpy_dtype,
          dtypes.float8_e4m3fn.as_numpy_dtype,
      ]:
        inp = np.random.rand(4, 4).astype(dtype)
        a = constant_op.constant(
            [float(x) for x in inp.ravel(order="C")],
            shape=[4, 4],
            dtype=dtypes.float32)
        slice_t = array_ops.slice(a, [0, 0], [2, 2])
        slice2_t = a[:2, :2]
        slice_val, slice2_val = self.evaluate([slice_t, slice2_t])
        self.assertAllEqual(slice_val, np.array(inp[:2, :2], dtype=np.float32))
        self.assertAllEqual(slice2_val, np.array(inp[:2, :2], dtype=np.float32))
        self.assertEqual(slice_val.shape, slice_t.get_shape())
        self.assertEqual(slice2_val.shape, slice2_t.get_shape())

  def testComplex(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name it does not test complex numbers; it tests a mixed
    # index-and-range slice of a 4-D float tensor.
    # Test: slices a random 4-D tensor with a full first axis, a scalar index, a random range and
    # a full last axis, and asserts equality with the NumPy slice. Returns nothing.
    inp = np.random.rand(4, 10, 10, 4).astype("f")
    a = constant_op.constant(inp, dtype=dtypes.float32)

    x = np.random.randint(0, 9)
    z = np.random.randint(0, 9)
    if z > 0:
      y = np.random.randint(0, z)
    else:
      y = 0
    slice_t = a[:, x, y:z, :]
    self.assertAllEqual(slice_t, inp[:, x, y:z, :])

  def testRandom(self):
    # [FUNCTION DOC]
    # Test: builds a random rank-6 float tensor with random dimensions (possibly zero), picks a
    # random valid begin and size per dimension, and slices with both the slice function and
    # range indexing. Asserts both results and their static shapes match the NumPy slice.
    # Returns nothing.
    # Random dims of rank 6
    input_shape = np.random.randint(0, 20, size=6)
    inp = np.random.rand(*input_shape).astype("f")
    a = constant_op.constant([float(x) for x in inp.ravel(order="C")],
                             shape=input_shape,
                             dtype=dtypes.float32)
    indices = [0 if x == 0 else np.random.randint(x) for x in input_shape]
    sizes = [
        np.random.randint(0, input_shape[i] - indices[i] + 1) for i in range(6)
    ]
    slice_t = array_ops.slice(a, indices, sizes)
    slice2_t = a[indices[0]:indices[0] + sizes[0],
                 indices[1]:indices[1] + sizes[1],
                 indices[2]:indices[2] + sizes[2],
                 indices[3]:indices[3] + sizes[3],
                 indices[4]:indices[4] + sizes[4],
                 indices[5]:indices[5] + sizes[5]]

    slice_val, slice2_val = self.evaluate([slice_t, slice2_t])

    expected_val = inp[indices[0]:indices[0] + sizes[0],
                       indices[1]:indices[1] + sizes[1],
                       indices[2]:indices[2] + sizes[2],
                       indices[3]:indices[3] + sizes[3],
                       indices[4]:indices[4] + sizes[4],
                       indices[5]:indices[5] + sizes[5]]
    self.assertAllEqual(slice_val, expected_val)
    self.assertAllEqual(slice2_val, expected_val)
    self.assertEqual(expected_val.shape, slice_t.get_shape())
    self.assertEqual(expected_val.shape, slice2_t.get_shape())

  def testPartialShapeInference(self):
    # [FUNCTION DOC]
    # Test: static shape inference of slicing a 1x2x3 zeros tensor, with all-minus-one sizes and
    # with a size given by a computed tensor for the first dimension. Asserts the inferred shape
    # is 1x2x3 in both cases. Returns nothing.
    z = array_ops.zeros((1, 2, 3))
    self.assertAllEqual(z.get_shape().as_list(), [1, 2, 3])

    m1 = array_ops.slice(z, [0, 0, 0], [-1, -1, -1])
    self.assertAllEqual(m1.get_shape().as_list(), [1, 2, 3])

    m2 = array_ops.slice(z, [0, 0, 0], [constant_op.constant(1) + 0, 2, -1])
    self.assertAllEqual(m2.get_shape().as_list(), [1, 2, 3])

  def _testGradientSlice(self, input_shape, slice_begin, slice_size):
    # [FUNCTION DOC]
    # HELPER: graph-mode gradient check for the gradient tests.
    # Builds a random float tensor of input_shape, slices it with begin and size lists, computes
    # the gradient of the slice with respect to the input using random upstream gradients, and
    # asserts it equals a zero tensor with those gradients placed in the sliced region. Returns
    # nothing.
    with self.cached_session():
      num_inputs = np.prod(input_shape)
      num_grads = np.prod(slice_size)
      inp = np.random.rand(num_inputs).astype("f").reshape(input_shape)
      a = constant_op.constant(
          [float(x) for x in inp.ravel(order="C")],
          shape=input_shape,
          dtype=dtypes.float32)
      slice_t = array_ops.slice(a, slice_begin, slice_size)
      grads = np.random.rand(num_grads).astype("f").reshape(slice_size)
      grad_tensor = constant_op.constant(grads)
      grad = gradients_impl.gradients(slice_t, [a], grad_tensor)[0]
      result = self.evaluate(grad)

    # Create a zero tensor of the input shape ane place
    # the grads into the right location to compare against TensorFlow.
    np_ans = np.zeros(input_shape)
    slices = []
    for i in range(len(input_shape)):
      slices.append(slice(slice_begin[i], slice_begin[i] + slice_size[i]))
    np_ans[tuple(slices)] = grads

    self.assertAllClose(np_ans, result)

  def _testGradientSliceTape(self, input_shape, slice_begin, slice_size):
    # [FUNCTION DOC]
    # HELPER: same check as the graph-mode gradient helper, but uses a gradient tape that watches
    # the input tensor. Compares the tape gradient of the slice with a zero tensor holding the
    # random upstream gradients in the sliced region. Returns nothing.
    with backprop.GradientTape() as tape:
      num_inputs = np.prod(input_shape)
      num_grads = np.prod(slice_size)
      inp = np.random.rand(num_inputs).astype("f").reshape(input_shape)
      a = constant_op.constant([float(x) for x in inp.ravel(order="C")],
                               shape=input_shape,
                               dtype=dtypes.float32)
      tape.watch(a)
      slice_t = array_ops.slice(a, slice_begin, slice_size)
      grads = np.random.rand(num_grads).astype("f").reshape(slice_size)
      grad_tensor = constant_op.constant(grads)
    grad = tape.gradient(slice_t, [a], grad_tensor)[0]
    result = self.evaluate(grad)

    # Create a zero tensor of the input shape ane place
    # the grads into the right location to compare against TensorFlow.
    np_ans = np.zeros(input_shape)
    slices = []
    for i in range(len(input_shape)):
      slices.append(slice(slice_begin[i], slice_begin[i] + slice_size[i]))
    np_ans[tuple(slices)] = grads

    self.assertAllClose(np_ans, result)

  def _testGradientVariableSize(self):
    # [FUNCTION DOC]
    # HELPER: graph-mode gradient check for a slice whose size is minus one (rest of the vector).
    # Slices a 3-element vector from index 1 to the end and asserts the gradient is zero then
    # ones. Returns nothing.
    with self.cached_session():
      inp = constant_op.constant([1.0, 2.0, 3.0], name="in")
      out = array_ops.slice(inp, [1], [-1])
      grad_actual = self.evaluate(gradients_impl.gradients(out, inp)[0])
    self.assertAllClose([0., 1., 1.], grad_actual)

  def _testGradientVariableSizeTape(self):
    # [FUNCTION DOC]
    # HELPER: tape-based version of the variable-size gradient check. Slices a watched 3-element
    # vector from index 1 to the end and asserts the tape gradient is zero then ones. Returns
    # nothing.
    with backprop.GradientTape() as tape:
      inp = constant_op.constant([1.0, 2.0, 3.0], name="in")
      tape.watch(inp)
      out = array_ops.slice(inp, [1], [-1])
    grad_actual = self.evaluate(tape.gradient(out, inp))
    self.assertAllClose([0., 1., 1.], grad_actual)

  def _testGradientVariableSize2D(self):
    # [FUNCTION DOC]
    # HELPER: graph-mode regression check for negative indices in multi-dimensional slices (an
    # Eigen bug). Computes gradients of two equivalent sums over a 3x3 matrix (a combined slice
    # and a chained slice) and asserts the gradients are equal. Returns nothing.
    # Regression test for bug in slice. A low-level bug in Eigen was causing
    # incorrect results for negative indices in multi-dimensional tensors.
    # See b/114318298.
    with self.cached_session():
      x = constant_op.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 7]])
      loss1 = math_ops.reduce_sum(x[:-1, :-1] * 1.0)
      loss2 = math_ops.reduce_sum(x[:-1][:, :-1])

      g1 = gradients_impl.gradients(loss1, x)[0]
      g2 = gradients_impl.gradients(loss2, x)[0]

      g1_val, g2_val = self.evaluate([g1, g2])
    self.assertAllEqual(g1_val, g2_val)

  def _testGradientVariableSize2DTape(self):
    # [FUNCTION DOC]
    # HELPER: tape-based version of the 2-D negative-index regression check, using a persistent
    # tape so two gradients can be taken. Asserts the gradients of the combined and chained
    # slice sums are equal. Returns nothing.
    # Regression test for bug in slice. A low-level bug in Eigen was causing
    # incorrect results for negative indices in multi-dimensional tensors.
    # See b/114318298.
    with backprop.GradientTape(persistent=True) as tape:
      x = constant_op.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 7]])
      tape.watch(x)
      loss1 = math_ops.reduce_sum(x[:-1, :-1] * 1.0)
      loss2 = math_ops.reduce_sum(x[:-1][:, :-1])

    g1 = tape.gradient(loss1, x)
    g2 = tape.gradient(loss2, x)
    g1_val, g2_val = self.evaluate([g1, g2])
    self.assertAllEqual(g1_val, g2_val)

  def testGradientsAll(self):
    # [FUNCTION DOC]
    # Test: in a fresh graph, runs the graph-mode gradient helper for a middle square, an upper
    # left square, a non-square slice, and a 3-D slice, then the two variable-size helpers.
    # Returns nothing.
    with ops.Graph().as_default():
      # Slice the middle square out of a 4x4 input
      self._testGradientSlice([4, 4], [1, 1], [2, 2])

      # Slice the upper left square out of a 4x4 input
      self._testGradientSlice([4, 4], [0, 0], [2, 2])

      # Slice a non-square input starting from (2,1)
      self._testGradientSlice([4, 4], [2, 1], [1, 2])

      # Slice a 3D tensor
      self._testGradientSlice([3, 3, 3], [0, 1, 0], [2, 1, 1])

      # Use -1 as a slice dimension.
      self._testGradientVariableSize()

      # Use -1 as a slice dimension on a 2D tensor.
      self._testGradientVariableSize2D()

  def testGradientsAllTape(self):
    # [FUNCTION DOC]
    # Test: same scenarios as the graph-mode gradient test, but using the tape-based helpers.
    # Returns nothing.
    # Slice the middle square out of a 4x4 input
    self._testGradientSliceTape([4, 4], [1, 1], [2, 2])

    # Slice the upper left square out of a 4x4 input
    self._testGradientSliceTape([4, 4], [0, 0], [2, 2])

    # Slice a non-square input starting from (2,1)
    self._testGradientSliceTape([4, 4], [2, 1], [1, 2])

    # Slice a 3D tensor
    self._testGradientSliceTape([3, 3, 3], [0, 1, 0], [2, 1, 1])

    # Use -1 as a slice dimension.
    self._testGradientVariableSizeTape()

    # Use -1 as a slice dimension on a 2D tensor.
    self._testGradientVariableSize2DTape()

  def testNotIterable(self):
    # [FUNCTION DOC]
    # Test: in graph mode, iterating over a symbolic scalar tensor must raise the
    # operator-not-allowed-in-graph error mentioning iterating over a symbolic tensor (guards
    # against Python looping forever through the item-access operator). Returns nothing.
    # Tensor iteration is disabled explicitly for only graph mode.
    with ops.Graph().as_default():
      # NOTE(mrry): If we register __getitem__ as an overloaded
      # operator, Python will valiantly attempt to iterate over the
      # Tensor from 0 to infinity.  This test ensures that this
      # unintended behavior is prevented.
      c = constant_op.constant(5.0)
      with self.assertRaisesRegex(errors_impl.OperatorNotAllowedInGraphError,
                                  "Iterating over a symbolic `tf.Tensor`"):
        for _ in c:
          pass

  def testComputedShape(self):
    # [FUNCTION DOC]
    # Test: static shape inference when begin and size are computed tensors. With constant begin
    # and size on a 2x3 constant the shape is 1x2. In a graph with a placeholder begin and a
    # size of minus one, the shape is unknown rows by 2 columns. Returns nothing.
    # NOTE(mrry): We cannot currently handle partially-known values,
    # because `tf.slice()` uses -1 to specify a wildcard size, and
    # this can't be handled using the
    # `tensor_util.constant_value_as_shape()` trick.
    a = constant_op.constant([[1, 2, 3], [4, 5, 6]])
    begin = constant_op.constant(0)
    size = constant_op.constant(1)
    b = array_ops.slice(a, [begin, 0], [size, 2])
    self.assertEqual([1, 2], b.get_shape())

    # placeholders only make sense in a graph.
    with ops.Graph().as_default():
      a = constant_op.constant([[1, 2, 3], [4, 5, 6]])
      begin = array_ops.placeholder(dtypes.int32, shape=())
      c = array_ops.slice(a, [begin, 0], [-1, 2])
      self.assertEqual([None, 2], c.get_shape().as_list())

  def testSliceOfSlice(self):
    # [FUNCTION DOC]
    # Test: repeatedly slices a 4x3 constant (slice of a slice) and combines the pieces in an
    # arithmetic expression that algebraically cancels. Asserts the evaluated result is all
    # zeros. Returns nothing.
    with self.session():
      a = constant_op.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
      b = a[1:, :]
      c = b[:-1, :]
      d = c[1, :]
      res = 2 * d - c[1, :] + a[2, :] - 2 * b[-2, :]
      self.assertAllEqual([0, 0, 0], self.evaluate(res))


if __name__ == "__main__":
  test.main()
