# [MODULE DOC]
# Functional tests for the concat op and its relatives. Covers stacking along different axes,
# many dtypes (including bfloat16 and 8-bit floats on GPU), random shapes, argument validation
# errors, gradients (including unknown dimensions and negative axes), shape inference with an
# unknown concat axis and zero-size inputs, plus a second test class for the concat offset op.
# No import-time side effects; running the file as a script launches the TensorFlow test runner.
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
"""Functional tests for Concat Op."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ConcatOpTest(test.TestCase):
  # [CLASS DOC]
  # Main test case for the concat op: forward results for several dtypes and axes, random
  # shape/axis combinations, error checking, gradient correctness and shape inference. Holds no
  # shared state; helper methods generate random inputs and verify results.

  @test_util.run_deprecated_v1
  def testHStack(self):
    # [FUNCTION DOC]
    # Concatenates two 4x4 float placeholders along axis 0, feeds random values, and asserts the
    # static shape matches and that the top and bottom halves equal the fed inputs. Returns
    # nothing.
    with self.session():
      p1 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      p2 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      c = array_ops.concat([p1, p2], 0)
      params = {
          p1: np.random.rand(4, 4).astype("f"),
          p2: np.random.rand(4, 4).astype("f")
      }
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:4, :], params[p1])
    self.assertAllEqual(result[4:, :], params[p2])

  @test_util.run_deprecated_v1
  def testVStack(self):
    # [FUNCTION DOC]
    # Concatenates two 4x4 float placeholders along axis 1, feeds random values, and asserts the
    # static shape matches and that the left and right halves equal the fed inputs. Returns
    # nothing.
    with self.session():
      p1 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      p2 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      c = array_ops.concat([p1, p2], 1)
      params = {
          p1: np.random.rand(4, 4).astype("f"),
          p2: np.random.rand(4, 4).astype("f")
      }
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:, :4], params[p1])
    self.assertAllEqual(result[:, 4:], params[p2])

  @test_util.run_deprecated_v1
  def test4DStack(self):
    # [FUNCTION DOC]
    # Concatenates rank-4 float placeholders of shapes 2x3x1x1 and 2x3x4x1 along axis 2 and
    # asserts the static shape and that each slice of the result equals the matching input.
    # Returns nothing.
    with self.session():
      p1 = array_ops.placeholder(dtypes.float32, shape=[2, 3, 1, 1])
      p2 = array_ops.placeholder(dtypes.float32, shape=[2, 3, 4, 1])
      c = array_ops.concat([p1, p2], 2)
      params = {
          p1: np.random.rand(2, 3, 1, 1).astype("f"),
          p2: np.random.rand(2, 3, 4, 1).astype("f")
      }
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:, :, :1, :], params[p1])
    self.assertAllEqual(result[:, :, 1:, :], params[p2])

  def testInt32GPU(self):
    # [FUNCTION DOC]
    # With GPU preferred, concatenates two random 2x3 32-bit integer constants along axis 0 and
    # asserts each half equals the corresponding input. Returns nothing.
    with test_util.use_gpu():
      p1 = np.random.rand(2, 3).astype("i")
      p2 = np.random.rand(2, 3).astype("i")
      x1 = constant_op.constant(p1)
      x2 = constant_op.constant(p2)
      c = array_ops.concat([x1, x2], 0)
      result = self.evaluate(c)
    self.assertAllEqual(result[:2, :], p1)
    self.assertAllEqual(result[2:, :], p2)

  def testBfloat16GPU(self):
    # [FUNCTION DOC]
    # With GPU preferred, concatenates two random 2x3 bfloat16 constants along axis 0 and
    # asserts each half equals the corresponding input. Returns nothing.
    with test_util.use_gpu():
      p1 = np.random.rand(2, 3).astype(dtypes.bfloat16.as_numpy_dtype)
      p2 = np.random.rand(2, 3).astype(dtypes.bfloat16.as_numpy_dtype)
      x1 = constant_op.constant(p1)
      x2 = constant_op.constant(p2)
      c = array_ops.concat([x1, x2], 0)
      result = self.evaluate(c)
    self.assertAllEqual(result[:2, :], p1)
    self.assertAllEqual(result[2:, :], p2)

  def testFloat8E5M2GPU(self):
    # [FUNCTION DOC]
    # With GPU preferred, concatenates two random 2x3 constants of the 5-exponent-bit 8-bit float
    # type along axis 0 and asserts each half equals the corresponding input. Returns nothing.
    with test_util.use_gpu():
      p1 = np.random.rand(2, 3).astype(dtypes.float8_e5m2.as_numpy_dtype)
      p2 = np.random.rand(2, 3).astype(dtypes.float8_e5m2.as_numpy_dtype)
      x1 = constant_op.constant(p1)
      x2 = constant_op.constant(p2)
      c = array_ops.concat([x1, x2], 0)
      result = self.evaluate(c)
    self.assertAllEqual(result[:2, :], p1)
    self.assertAllEqual(result[2:, :], p2)

  def testFloat8E4M3FNGPU(self):
    # [FUNCTION DOC]
    # With GPU preferred, concatenates two random 2x3 constants of the 4-exponent-bit finite-only
    # 8-bit float type along axis 0 and asserts each half equals the corresponding input.
    # Returns nothing.
    with test_util.use_gpu():
      p1 = np.random.rand(2, 3).astype(dtypes.float8_e4m3fn.as_numpy_dtype)
      p2 = np.random.rand(2, 3).astype(dtypes.float8_e4m3fn.as_numpy_dtype)
      x1 = constant_op.constant(p1)
      x2 = constant_op.constant(p2)
      c = array_ops.concat([x1, x2], 0)
      result = self.evaluate(c)
    self.assertAllEqual(result[:2, :], p1)
    self.assertAllEqual(result[2:, :], p2)

  def testRefType(self):
    # [FUNCTION DOC]
    # With GPU preferred, concatenates two 4x4 float variables (reference-typed inputs) along
    # axis 0 after initializing variables, and asserts the static shape and that each half
    # equals the initial value. Returns nothing.
    with test_util.use_gpu():
      p1 = np.random.rand(4, 4).astype("f")
      p2 = np.random.rand(4, 4).astype("f")
      v1 = variables.Variable(p1)
      v2 = variables.Variable(p2)
      c = array_ops.concat([v1, v2], 0)
      self.evaluate(variables.global_variables_initializer())
      result = self.evaluate(c)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:4, :], p1)
    self.assertAllEqual(result[4:, :], p2)

  def _testRandom(self, dtype):
    # [FUNCTION DOC]
    # HELPER: randomized concat check for testRandom. For the given TensorFlow dtype, picks a
    # random rank-5 shape with dimensions 1 to 4, a random count of 2 to 9 tensors and a random
    # concat axis, varies the axis size per tensor, and feeds random placeholders. Types without
    # direct feed support (bfloat16, and via the same cast path any type that differs from the
    # feed dtype) are fed as float32 and cast to dtype before and back after the concat. Asserts
    # the static shape matches the result and that every input appears at the right offset along
    # the axis (exact equality, or closeness with tolerance 0.01 when a cast was involved).
    # Uses numpy's random generator (unseeded here). Returns nothing.
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    # Random number of tensors, but always > 1.
    num_tensors = np.random.randint(2, 10)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    params = {}
    if dtype == dtypes.bfloat16:
      dtype_feed = dtypes.float32
    else:
      dtype_feed = dtype
    with self.session():
      p = []
      for i in np.arange(num_tensors):
        input_shape = shape
        input_shape[concat_dim] = np.random.randint(1, 5)
        placeholder = array_ops.placeholder(dtype_feed, shape=input_shape)
        p.append(placeholder)

        t = dtype_feed.as_numpy_dtype
        params[placeholder] = np.random.rand(*input_shape).astype(t)

      if dtype != dtype_feed:
        concat_inputs = [math_ops.cast(p_i, dtype) for p_i in p]
      else:
        concat_inputs = p
      c = array_ops.concat(concat_inputs, concat_dim)
      if dtype != dtype_feed:
        c = math_ops.cast(c, dtype_feed)
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    cur_offset = 0

    for i in np.arange(num_tensors):
      # The index into the result is the ':' along all dimensions
      # except the concat_dim. slice(0, size) is used for ':', and
      # a list of slices is used to index into result.
      ind = [slice(0, params[p[i]].shape[j]) for j in np.arange(5)]
      ind[concat_dim] = slice(cur_offset,
                              cur_offset + params[p[i]].shape[concat_dim])
      cur_offset += params[p[i]].shape[concat_dim]
      if dtype == dtype_feed:
        self.assertAllEqual(result[tuple(ind)], params[p[i]])
      else:
        self.assertAllClose(result[tuple(ind)], params[p[i]], 0.01)

  @test_util.run_deprecated_v1
  def testRandom(self):
    # [FUNCTION DOC]
    # Runs the randomized concat check for ten dtypes: bool, float32, 16/32/64-bit integers,
    # bfloat16, both 8-bit floats, and both complex types. Returns nothing.
    self._testRandom(dtypes.bool)
    self._testRandom(dtypes.float32)
    self._testRandom(dtypes.int16)
    self._testRandom(dtypes.int32)
    self._testRandom(dtypes.int64)
    self._testRandom(dtypes.bfloat16)
    self._testRandom(dtypes.float8_e5m2)
    self._testRandom(dtypes.float8_e4m3fn)
    self._testRandom(dtypes.complex64)
    self._testRandom(dtypes.complex128)

  @test_util.run_deprecated_v1
  def testInvalidConcatDimTypeAndShape(self):
    # [FUNCTION DOC]
    # Checks argument validation of concat using float variables: swapped arguments, non-list
    # values, a non-scalar axis and an empty list must raise value or type errors as appropriate;
    # a scalar integer tensor as the axis must raise nothing; a one-element (non-scalar) tensor
    # axis must raise a value error. Returns nothing.
    a = variables.Variable(constant_op.constant(1.0, shape=[1]))
    b = variables.Variable(constant_op.constant(2.0, shape=[1]))
    with self.assertRaises(ValueError):
      array_ops.concat(b, a)
    with self.assertRaises(TypeError):
      array_ops.concat(1, 4.2)
    with self.assertRaises(ValueError):
      array_ops.concat(1, a)
    with self.assertRaises(TypeError):
      array_ops.concat([a, b], a)
    with self.assertRaises(ValueError):
      array_ops.concat([a, b], [3])
    with self.assertRaises(ValueError):
      array_ops.concat([], 0)
    # An integer tensor for shape dim should throw no error.
    array_ops.concat(1, constant_op.constant(0, shape=[]))
    # A non-scalar tensor for shape should throw ValueError.
    with self.assertRaises(ValueError):
      array_ops.concat(1, constant_op.constant(0, shape=[1]))

  def testScalars(self):
    # [FUNCTION DOC]
    # Builds a list of rank-0 tensors by squaring the elements of a two-element tensor and
    # asserts that concatenating them along axis 0 raises a value error or invalid-argument error
    # because scalars cannot be concatenated. Returns nothing.
    arr = ops.convert_to_tensor([0.2, 0.3])
    outs = []
    for i in range(arr.shape[0]):
      outs.append(arr[i]**2)
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      _ = array_ops.concat(outs, axis=0)

  def _testGradientsSimple(self, dtype):
    # [FUNCTION DOC]
    # HELPER: gradient check for the given real or complex dtype. For both a negative and a
    # positive spelling of the same middle axis, concatenates three random 10xNx2 tensors
    # (N of 1, 2 and 6), back-propagates a random gradient of the output shape through the
    # concat, re-concatenates the per-input gradients and asserts they equal the supplied gradient.
    # Complex inputs get a conjugate-style imaginary part. Runs with GPU preferred. Returns
    # nothing.
    # Test both positive and negative concat axis.
    # -2 and 1 correspond to the same axis for 3-dimensional tensors.
    for axis in [-2, 1]:
      with test_util.use_gpu():
        inp = []
        inp_tensors = []
        for x in [1, 2, 6]:
          shape = [10, x, 2]
          t = np.random.rand(*shape).astype(dtype.as_numpy_dtype)
          if dtype.is_complex:
            t += -1j * t
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  t.flatten(),
                  shape=shape,
                  dtype=dtype))
        c = array_ops.concat(inp_tensors, axis)
        output_shape = [10, 9, 2]
        grad_inp = np.random.rand(*output_shape).astype(dtype.as_numpy_dtype)
        if dtype.is_complex:
          grad_inp += -1j * grad_inp
        grad_tensor = constant_op.constant(
            grad_inp.flatten(), shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, axis)
        result = self.evaluate(concated_grad)
    self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testGradientsSimple(self):
    # [FUNCTION DOC]
    # Runs the simple gradient check for 32-bit float and 64-bit complex. Returns nothing.
    self._testGradientsSimple(dtypes.float32)
    self._testGradientsSimple(dtypes.complex64)

  @test_util.run_deprecated_v1
  def testGradientsFirstDim(self):
    # [FUNCTION DOC]
    # Concatenates three random float tensors of shapes Nx10x2 (N of 1, 2, 6) along axis 0, pulls
    # a random output-shaped gradient back through the concat and asserts the re-concatenated
    # input gradients equal it. Runs with GPU preferred. Returns nothing.
    with test_util.use_gpu():
      inp = []
      inp_tensors = []
      for x in [1, 2, 6]:
        shape = [x, 10, 2]
        t = np.random.rand(*shape).astype("f")
        inp.append(t)
        inp_tensors.append(
            constant_op.constant(
                t.flatten(),
                shape=shape,
                dtype=dtypes.float32))
      c = array_ops.concat(inp_tensors, 0)
      output_shape = [9, 10, 2]
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = constant_op.constant(
          grad_inp.flatten(), shape=output_shape)
      grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = array_ops.concat(grad, 0)
      result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testGradientsLastDim(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the loop variable axis (negative and positive spellings of the last axis) is
    # only used when re-concatenating the gradients; the forward concat inside the loop always
    # uses the literal axis 2, so the negative axis is never exercised for the forward op.
    # Concatenates three random float tensors of shapes 10x2xN (N of 1, 2, 6) along the last
    # axis, pulls a random gradient back through the concat and asserts the re-concatenated
    # input gradients equal it. Runs with GPU preferred. Returns nothing.
    # Test both positive and negative concat axis.
    # -1 and 2 correspond to the same axis for 3-dimensional tensors.
    for axis in [-1, 2]:
      with test_util.use_gpu():
        inp = []
        inp_tensors = []
        for x in [1, 2, 6]:
          shape = [10, 2, x]
          t = np.random.rand(*shape).astype("f")
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  t.flatten(),
                  shape=shape,
                  dtype=dtypes.float32))
        c = array_ops.concat(inp_tensors, 2)
        output_shape = [10, 2, 9]
        grad_inp = np.random.rand(*output_shape).astype("f")
        grad_tensor = constant_op.constant(
            grad_inp.flatten(), shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, axis)
        result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  def _RunAndVerifyGradientsRandom(self):
    # [FUNCTION DOC]
    # HELPER: one randomized gradient check. Picks a random rank-5 shape with dimensions 1 to 4,
    # 12 to 19 input tensors, a random concat axis and a random axis size per tensor, concatenates
    # random float tensors (GPU preferred), pulls a random gradient back through the concat and
    # asserts the re-concatenated input gradients equal the supplied gradient. Uses numpy's
    # unseeded random generator. Returns nothing.
    # Random dims of rank 5
    input_shape = np.random.randint(1, 5, size=5)
    # Random number of tensors
    num_tensors = np.random.randint(12, 20)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    concat_dim_sizes = np.random.randint(1, 5, size=num_tensors)
    with test_util.use_gpu():
      inp = []
      inp_tensors = []
      for x in concat_dim_sizes:
        shape = input_shape
        shape[concat_dim] = x
        t = np.random.rand(*shape).astype("f")
        inp.append(t)
        inp_tensors.append(
            constant_op.constant(t.flatten(), shape=shape,
                                 dtype=dtypes.float32))
      c = array_ops.concat(inp_tensors, concat_dim)
      output_shape = input_shape
      output_shape[concat_dim] = concat_dim_sizes.sum()
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = constant_op.constant(grad_inp.flatten(), shape=output_shape)
      grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = array_ops.concat(grad, concat_dim)
      result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testGradientsRandom(self):
    # [FUNCTION DOC]
    # Repeats the randomized gradient check five times. Returns nothing.
    for _ in range(5):
      self._RunAndVerifyGradientsRandom()

  @test_util.run_deprecated_v1
  def testGradientWithUnknownInputDim(self):
    # [FUNCTION DOC]
    # Concatenates two unknown-shape float placeholders along axis 2, pulls a random 10x2x9
    # gradient back through the concat, feeds actual inputs of shapes 10x2x3 and 10x2x6 and
    # asserts the re-concatenated gradients equal the supplied gradient. Returns nothing.
    with self.session():
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.placeholder(dtypes.float32)
      c = array_ops.concat([x, y], 2)

      output_shape = [10, 2, 9]
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = constant_op.constant(
          grad_inp.flatten(), shape=output_shape)

      grad = gradients_impl.gradients([c], [x, y], [grad_tensor])
      concated_grad = array_ops.concat(grad, 2)
      params = {
          x: np.random.rand(10, 2, 3).astype("f"),
          y: np.random.rand(10, 2, 6).astype("f")
      }
      result = concated_grad.eval(feed_dict=params)

      self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testShapeError(self):
    # [FUNCTION DOC]
    # Asserts that concat raises a value error for mismatched ranks, for a size mismatch in a
    # non-concat dimension, and for concat axes out of range (3 and -4 for rank-3 inputs).
    # Returns nothing.
    # Rank doesn't match.
    with self.assertRaises(ValueError):
      array_ops.concat(
          [constant_op.constant(10.0, shape=[4, 4, 4, 4]),
           constant_op.constant(20.0, shape=[4, 4, 4])
          ], 1)

    # Dimensions don't match in a non-concat dim.
    with self.assertRaises(ValueError):
      array_ops.concat(
          [constant_op.constant(10.0, shape=[1, 2, 1]),
           constant_op.constant(20.0, shape=[3, 2, 1])
          ], 1)

    # concat_dim out of range.
    with self.assertRaises(ValueError):
      array_ops.concat(
          [constant_op.constant(10.0, shape=[4, 4, 4]),
           constant_op.constant(20.0, shape=[4, 4, 4])
          ], 3)

    # concat_dim out of range
    with self.assertRaises(ValueError):
      array_ops.concat(
          [constant_op.constant(10.0, shape=[4, 4, 4]),
           constant_op.constant(20.0, shape=[4, 4, 4])
          ], -4)

  @test_util.run_deprecated_v1
  def testShapeWithUnknownConcatDim(self):
    # [FUNCTION DOC]
    # Uses a placeholder for the concat axis. With a mix of unknown-shape placeholders and
    # rank-4 constants, asserts the output rank is inferred as 4; with only unknown-shape inputs
    # asserts the static shape is entirely unknown; with a rank-3 constant among rank-4 inputs
    # asserts a value error. Returns nothing.
    p1 = array_ops.placeholder(dtypes.float32)
    c1 = constant_op.constant(10.0, shape=[4, 4, 4, 4])
    p2 = array_ops.placeholder(dtypes.float32)
    c2 = constant_op.constant(20.0, shape=[4, 4, 4, 4])
    dim = array_ops.placeholder(dtypes.int32)
    concat = array_ops.concat([p1, c1, p2, c2], dim)
    self.assertEqual(4, concat.get_shape().ndims)

    # All dimensions unknown.
    concat2 = array_ops.concat([p1, p2], dim)
    self.assertEqual(None, concat2.get_shape())

    # Rank doesn't match.
    c3 = constant_op.constant(30.0, shape=[4, 4, 4])
    with self.assertRaises(ValueError):
      array_ops.concat([p1, c1, p2, c3], dim)

  @test_util.run_deprecated_v1
  def testZeroSize(self):
    # [FUNCTION DOC]
    # Seeds numpy's random generator with 7. For leading shapes empty or of length 2 and
    # trailing shapes empty or of length 3, and for zero, one or two entries in the concat
    # dimension of each of two random inputs, concatenates them on GPU if available and asserts
    # the result equals numpy's concatenation, then back-propagates a random gradient and asserts
    # the input gradients re-concatenate to it. Verifies concat handles zero-size inputs. Returns
    # nothing.
    # Verify that concat doesn't crash and burn for zero size inputs
    np.random.seed(7)
    with test_util.use_gpu():
      for shape0 in (), (2,):
        axis = len(shape0)
        for shape1 in (), (3,):
          for n0 in 0, 1, 2:
            for n1 in 0, 1, 2:
              x0 = np.random.randn(*(shape0 + (n0,) + shape1))
              x1 = np.random.randn(*(shape0 + (n1,) + shape1))
              correct = np.concatenate([x0, x1], axis=axis)
              # TODO(irving): Make tf.concat handle map, then drop list().
              xs = list(map(constant_op.constant, [x0, x1]))
              c = array_ops.concat(xs, axis)
              self.assertAllEqual(self.evaluate(c), correct)
              # Check gradients
              dc = np.random.randn(*c.get_shape().as_list())
              dxs = self.evaluate(gradients_impl.gradients(c, xs, dc))
              self.assertAllEqual(dc, np.concatenate(dxs, axis=axis))

  @test_util.run_deprecated_v1
  def testTensorConcatDim0Grad(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient of concat along axis 0 for three random 64-bit float inputs
    # (leading sizes 20, 10, 14) and asserts the gradient error is below 1e-11. Returns nothing.
    x_shapes = [[20, 7, 3], [10, 7, 3], [14, 7, 3]]
    output_shape = [44, 7, 3]
    x_vals = [
        np.random.random_sample(x_shape).astype(np.float64)
        for x_shape in x_shapes
    ]
    with self.cached_session():
      xs = [constant_op.constant(x_val) for x_val in x_vals]
      output = array_ops.concat(xs, 0)
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape)
    self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testTensorConcatDim1Grad(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient of concat along axis 1 for three random 64-bit float inputs
    # (middle sizes 7, 3, 1) and asserts the gradient error is below 1e-11. Returns nothing.
    x_shapes = [[20, 7, 3], [20, 3, 3], [20, 1, 3]]
    output_shape = [20, 11, 3]
    x_vals = [
        np.random.random_sample(x_shape).astype(np.float64)
        for x_shape in x_shapes
    ]
    with self.cached_session():
      xs = [constant_op.constant(x_val) for x_val in x_vals]
      output = array_ops.concat(xs, 1)
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape)
    self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testIndexedSlicesConcatDim0Grad(self):
    # [FUNCTION DOC]
    # Concatenates three random inputs along axis 0, gathers four rows from the result (so the
    # gradient arrives as sparse indexed slices) and numerically checks the gradient with respect
    # to the inputs, asserting the error is below 1e-11. Returns nothing.
    x_shapes = [[20, 7, 3], [10, 7, 3], [14, 7, 3]]
    output_shape = [4, 7, 3]
    x_vals = [
        np.random.random_sample(x_shape).astype(np.float64)
        for x_shape in x_shapes
    ]
    with self.cached_session():
      xs = [constant_op.constant(x_val) for x_val in x_vals]
      x_concat = array_ops.concat(xs, 0)
      output = array_ops.gather(x_concat, [1, 2, 0, 5])
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape)
    self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testIndexedSlicesConcatDim1Grad(self):
    # [FUNCTION DOC]
    # Like the axis-0 version but concatenating along axis 1: gathers four rows of the
    # concatenation and numerically checks the gradient to the inputs, asserting the error is
    # below 1e-11. Returns nothing.
    x_shapes = [[20, 7, 3], [20, 3, 3], [20, 1, 3]]
    output_shape = [4, 11, 3]
    x_vals = [
        np.random.random_sample(x_shape).astype(np.float64)
        for x_shape in x_shapes
    ]
    with self.cached_session():
      xs = [constant_op.constant(x_val) for x_val in x_vals]
      x_concat = array_ops.concat(xs, 1)
      output = array_ops.gather(x_concat, [1, 2, 0, 5])
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape)
    self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testIndexedSlicesConcatDim2Grad(self):
    # [FUNCTION DOC]
    # Like the axis-0 version but concatenating along axis 2: gathers four rows of the
    # concatenation and numerically checks the gradient to the inputs, asserting the error is
    # below 1e-11. Returns nothing.
    x_shapes = [[20, 7, 3], [20, 7, 1], [20, 7, 2]]
    output_shape = [4, 7, 6]
    x_vals = [
        np.random.random_sample(x_shape).astype(np.float64)
        for x_shape in x_shapes
    ]
    with self.cached_session():
      xs = [constant_op.constant(x_val) for x_val in x_vals]
      x_concat = array_ops.concat(xs, 2)
      output = array_ops.gather(x_concat, [1, 2, 0, 5])
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape)
    self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testIndexedSlicesConcatDim1Grad_UnknownInputDim(self):
    # [FUNCTION DOC]
    # Same as the axis-1 sparse-gradient check, but the three inputs are float placeholders of
    # unknown shape; actual values are supplied through an extra feed mapping to the gradient
    # checker. Asserts the gradient error is below 1e-11. Returns nothing.
    x_shapes = [[20, 7, 3], [20, 3, 3], [20, 1, 3]]
    output_shape = [4, 11, 3]
    with self.cached_session():
      x_1 = array_ops.placeholder(dtypes.float64)
      x_2 = array_ops.placeholder(dtypes.float64)
      x_3 = array_ops.placeholder(dtypes.float64)
      xs = [x_1, x_2, x_3]

      x_concat = array_ops.concat(xs, 1)
      output = array_ops.gather(x_concat, [1, 2, 0, 5])
      params = {
          x_1: np.random.random_sample(x_shapes[0]).astype(np.float64),
          x_2: np.random.random_sample(x_shapes[1]).astype(np.float64),
          x_3: np.random.random_sample(x_shapes[2]).astype(np.float64)
      }
      err = gradient_checker.compute_gradient_error(xs, x_shapes, output,
                                                    output_shape,
                                                    extra_feed_dict=params)
    self.assertLess(err, 1e-11)

  def testConcatTuple(self):
    # [FUNCTION DOC]
    # Asserts that concatenating two random arrays given as a tuple yields the same result as
    # when given as a list. Returns nothing.
    c1 = np.random.rand(4, 4)
    c2 = np.random.rand(4, 4)
    concat_list_t = array_ops.concat([c1, c2], 0)
    concat_tuple_t = array_ops.concat((c1, c2), 0)
    self.assertAllEqual(
        self.evaluate(concat_list_t), self.evaluate(concat_tuple_t))

  @test_util.run_deprecated_v1
  def testConcatNoScalars(self):
    # [FUNCTION DOC]
    # Asserts that concatenating three scalar tensors with a placeholder axis raises a value
    # error whose message says scalars cannot be concatenated and suggests stacking. Returns
    # nothing.
    scalar = constant_op.constant(7)
    dim = array_ops.placeholder(dtypes.int32)
    with self.assertRaisesRegex(
        ValueError, r"Can't concatenate scalars \(use tf\.stack instead\)"):
      array_ops.concat([scalar, scalar, scalar], dim)

  # important as gpu implementation could fail if
  # shared memory is not large for all the inputs
  @test_util.run_deprecated_v1
  def testConcatLargeNumberOfTensors(self):
    # [FUNCTION DOC]
    # Stress test with many inputs (5000 when a GPU is available, otherwise 500) of shape 7x13.
    # For each of axis 0 and 1, concatenates placeholders fed with random values and asserts the
    # static shape matches and each input appears at the correct offset. Guards against a GPU
    # kernel failing when shared memory is insufficient for all inputs. Returns nothing.
    with self.session():
      for concat_dim in range(2):
        params = {}
        p = []
        shape = np.array([7, 13])
        if test.is_gpu_available():
          num_tensors = 5000
        else:
          num_tensors = 500
        for i in np.arange(num_tensors):
          input_shape = shape
          placeholder = array_ops.placeholder(dtypes.float32, shape=input_shape)
          p.append(placeholder)

          params[placeholder] = np.random.rand(*input_shape).astype(np.float32)

        concat_inputs = p
        c = array_ops.concat(concat_inputs, concat_dim)
        result = c.eval(feed_dict=params)

        self.assertEqual(result.shape, c.get_shape())
        cur_offset = 0

        for i in np.arange(num_tensors):
          # The index into the result is the ':' along all dimensions
          # except the concat_dim. slice(0, size) is used for ':', and
          # a list of slices is used to index into result.
          index = [slice(0, params[p[i]].shape[j]) for j in np.arange(2)]
          index[concat_dim] = slice(cur_offset,
                                    cur_offset + params[p[i]].shape[concat_dim])
          cur_offset += params[p[i]].shape[concat_dim]
          self.assertAllEqual(result[tuple(index)], params[p[i]])

  def testConcatEmpty(self):
    # [FUNCTION DOC]
    # Concatenates two empty lists with the raw version-2 concat op along axis 0 (GPU preferred)
    # and asserts the output has zero elements. Returns nothing.
    with test_util.use_gpu():
      t1 = []
      t2 = []
      output = gen_array_ops.concat_v2([t1, t2], 0)
      # Checks that output is empty
      self.assertEqual(self.evaluate(output).size, 0)

  @test_util.run_deprecated_v1
  def testConcatInvalidAxis(self):
    # [FUNCTION DOC]
    # Concatenates two one-element lists along axis 1 (out of range for rank 1) with the raw
    # version-2 concat op and asserts a value error is raised. Returns nothing.
    with self.assertRaises(ValueError):
      with test_util.use_gpu():
        t1 = [1]
        t2 = [2]
        gen_array_ops.concat_v2([t1, t2], 1).eval()

  def testConcatInvalidAxisInTfFunction(self):
    # [FUNCTION DOC]
    # Defines a compiled function that concatenates two lists using a huge out-of-range axis
    # value with the raw version-2 concat op, then asserts calling it raises a value error
    # (regression check that bad axes are rejected rather than crashing). Returns nothing.

    @def_function.function
    def concat_wrapper():
      # [FUNCTION DOC]
      # HELPER: the compiled function under test. Takes no arguments, concatenates the lists
      # [1, 2, 3] and [4, 5, 6] along an invalid axis (a very large constant) and returns the
      # result; tracing fails with a value error.
      y = gen_array_ops.concat_v2(
          values=[[1, 2, 3], [4, 5, 6]], axis=0xb500005b)
      return y

    with self.assertRaises(ValueError):
      concat_wrapper()

  def testConcatNegativeAxis(self):
    # [FUNCTION DOC]
    # Concatenates two 2x3 integer lists with the raw version-2 concat op using axis -2 and then
    # -1 (GPU preferred), asserting the static shapes (4x3 and 2x6) and the concatenated values.
    # Returns nothing.
    with test_util.use_gpu():
      t1 = [[1, 2, 3], [4, 5, 6]]
      t2 = [[7, 8, 9], [10, 11, 12]]

      c = gen_array_ops.concat_v2([t1, t2], -2)
      self.assertEqual([4, 3], c.get_shape().as_list())
      output = self.evaluate(c)
      self.assertAllEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                          output)

      c = gen_array_ops.concat_v2([t1, t2], -1)
      self.assertEqual([2, 6], c.get_shape().as_list())
      output = self.evaluate(c)
      self.assertAllEqual([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]], output)

  def _testGradientsForAxis(
      self, inp_tensors, axis, output_shape, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: dense gradient check for testGradientsNegativeAxis. Takes the list of input
    # tensors, the concat axis, the shape of the concatenated output, and an optional mapping of
    # placeholder feeds (default none). Concatenates the inputs, pulls a random float gradient of
    # the output shape back through the concat, re-concatenates the per-input gradients along the
    # same axis and asserts they equal the supplied gradient. Returns nothing.
    with self.cached_session():
      c = array_ops.concat(inp_tensors, axis)
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = constant_op.constant(
          grad_inp.flatten(), shape=output_shape)
      grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = array_ops.concat(grad, axis)
      result = concated_grad.eval(feed_dict=feed_dict)
      self.assertAllEqual(result, grad_inp)

  def _testIndexedSlicesGradientsForAxis(
      self, inp_tensors, axis, output_shape, gather_indexes, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: sparse (indexed slices) gradient check for testGradientsNegativeAxis. Like the dense
    # version but gathers the rows given by gather_indexes from the concatenation, so gradients
    # are sparse; the output shape is that of the gathered result. Re-concatenates the per-input
    # gradients, gathers the same rows and asserts equality with the supplied random gradient.
    # Optional feed mapping defaults to none. Returns nothing.
    with self.cached_session():
      c = array_ops.gather(
          array_ops.concat(inp_tensors, axis), gather_indexes)
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = constant_op.constant(
          grad_inp.flatten(), shape=output_shape)
      grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = array_ops.gather(
          array_ops.concat(grad, axis), gather_indexes)
      result = concated_grad.eval(feed_dict=feed_dict)
      self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testGradientsNegativeAxis(self):
    # [FUNCTION DOC]
    # Checks concat gradients with negative axes: dense gradient with axis -2 on known shapes,
    # dense gradient with axis -1 on unknown-shape placeholders, sparse gradient with axis -2 on
    # known shapes, and asserts a value error for the sparse gradient with a negative axis when
    # the rank is unknown. Returns nothing.
    x1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x2 = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    inp_tensors = [constant_op.constant(x1, shape=(2, 3), dtype=dtypes.float32),
                   constant_op.constant(x2, shape=(2, 3), dtype=dtypes.float32)]

    # Test concat gradient with axis == -2
    self._testGradientsForAxis(inp_tensors, -2, output_shape=[4, 3])

    # Test concat gradient with unknown-shape tensors.
    x1_placeholder = array_ops.placeholder(dtypes.float32)
    x2_placeholder = array_ops.placeholder(dtypes.float32)
    inp_tensors_placeholders = [x1_placeholder, x2_placeholder]
    feed_dict = {x1_placeholder: x1, x2_placeholder: x2}
    self._testGradientsForAxis(
        inp_tensors_placeholders, -1, output_shape=[2, 6], feed_dict=feed_dict)

    # Test IndexedSlices concat gradient.
    self._testIndexedSlicesGradientsForAxis(
        inp_tensors, -2, output_shape=[2, 3], gather_indexes=[2, 0])

    # We don't support calculating IndexedSlices concat gradient for
    # negative indexes when rank is not known.
    with self.assertRaises(ValueError):
      self._testIndexedSlicesGradientsForAxis(
          inp_tensors_placeholders, -2, output_shape=[2, 3],
          gather_indexes=[2, 0], feed_dict=feed_dict)

  def testConcatDtype(self):
    # [FUNCTION DOC]
    # For signed and unsigned 32 and 64-bit integer types, concatenates two 2x3 constants along
    # axis 1 with the raw version-2 concat op (GPU preferred) and asserts the static shape 2x6
    # and the values. Returns nothing.
    for dtype in [dtypes.int32, dtypes.int64, dtypes.uint32, dtypes.uint64]:
      with test_util.use_gpu():
        t1 = constant_op.constant([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        t2 = constant_op.constant([[7, 8, 9], [10, 11, 12]], dtype=dtype)

        c = gen_array_ops.concat_v2([t1, t2], 1)
        self.assertEqual([2, 6], c.get_shape().as_list())
        output = self.evaluate(c)
        self.assertAllEqual([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]], output)

  def testConcatAxisType(self):
    # [FUNCTION DOC]
    # For both 32-bit and 64-bit integer axis tensors, concatenates two 2x3 integer lists along
    # axis 1 with the raw version-2 concat op (GPU preferred) and asserts the static shape 2x6
    # and the values. Returns nothing.
    for dtype in [dtypes.int32, dtypes.int64]:
      with test_util.use_gpu():
        t1 = [[1, 2, 3], [4, 5, 6]]
        t2 = [[7, 8, 9], [10, 11, 12]]

        c = gen_array_ops.concat_v2([t1, t2],
                                    constant_op.constant(1, dtype=dtype))
        self.assertEqual([2, 6], c.get_shape().as_list())
        output = self.evaluate(c)
        self.assertAllEqual([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]], output)


class ConcatOffsetTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the concat offset op, which computes the starting offset of each input within
  # the concatenated output along a given dimension. Covers normal use, error conditions,
  # negative dimensions and 64-bit shapes.

  def testBasic(self):
    # [FUNCTION DOC]
    # Computes offsets along dimension 1 for three shapes (2x3x5, 2x7x5, 2x20x5) with GPU
    # preferred and asserts the offsets are zero except 0, 3 and 10 in the second position.
    # Returns nothing.
    with test_util.use_gpu():
      cdim = constant_op.constant(1, dtypes.int32)
      s0 = constant_op.constant([2, 3, 5], dtypes.int32)
      s1 = constant_op.constant([2, 7, 5], dtypes.int32)
      s2 = constant_op.constant([2, 20, 5], dtypes.int32)
      off = gen_array_ops.concat_offset(cdim, [s0, s1, s2])
      ans = self.evaluate(off)
      self.assertAllEqual(ans, [[0, 0, 0], [0, 3, 0], [0, 10, 0]])

  @test_util.run_deprecated_v1
  def testNotVector(self):
    # [FUNCTION DOC]
    # Passes shape inputs that are matrices instead of vectors and asserts an invalid-argument
    # error saying the input should be a vector. Returns nothing.
    cdim = constant_op.constant(1, dtypes.int32)
    s0 = constant_op.constant([[2, 3, 5]], dtypes.int32)
    s1 = constant_op.constant([[2, 7, 5]], dtypes.int32)
    off = gen_array_ops.concat_offset(cdim, [s0, s1])
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                r"should be a vector"):
      self.evaluate(off)

  @test_util.run_deprecated_v1
  def testConcatDimOutOfRange(self):
    # [FUNCTION DOC]
    # Uses concat dimension 4 with rank-3 shapes and asserts an invalid-argument error saying the
    # concat dimension is out of range. Returns nothing.
    cdim = constant_op.constant(4, dtypes.int32)
    s0 = constant_op.constant([2, 3, 5], dtypes.int32)
    s1 = constant_op.constant([2, 7, 5], dtypes.int32)
    off = gen_array_ops.concat_offset(cdim, [s0, s1])
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                r"Concat dim is out of range: 4 vs. 3"):
      self.evaluate(off)

  @test_util.run_deprecated_v1
  def testDimMismatch(self):
    # [FUNCTION DOC]
    # Passes one rank-3 and one rank-4 shape vector and asserts an invalid-argument error saying
    # the vector should contain three elements. Returns nothing.
    cdim = constant_op.constant(1, dtypes.int32)
    s0 = constant_op.constant([2, 3, 5], dtypes.int32)
    s1 = constant_op.constant([2, 7, 5, 10], dtypes.int32)
    off = gen_array_ops.concat_offset(cdim, [s0, s1])
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                r"should contain 3 elem"):
      self.evaluate(off)

  @test_util.run_deprecated_v1
  def testSizeMismatch(self):
    # [FUNCTION DOC]
    # Passes shapes that differ in a non-concat dimension (concat along 1, last dimension 5 vs.
    # 10) and asserts an invalid-argument error saying all dimensions except the concat one must
    # match and naming both shapes. Returns nothing.
    cdim = constant_op.constant(1, dtypes.int32)
    s0 = constant_op.constant([2, 3, 5], dtypes.int32)
    s1 = constant_op.constant([2, 7, 10], dtypes.int32)
    off = gen_array_ops.concat_offset(cdim, [s0, s1])
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        r"All dimensions except 1 must match. Input 1 has shape \[2 7 10\] "
        r"and doesn't match input 0 with shape \[2 3 5\]."):
      self.evaluate(off)

  def testNegativeDim(self):
    # [FUNCTION DOC]
    # Computes offsets with concat dimension -2 (equivalent to 1) and -3 (equivalent to 0) for
    # rank-3 shapes with GPU preferred, and asserts the offsets match those of the positive
    # spellings. Returns nothing.
    with test_util.use_gpu():
      cdim = constant_op.constant(-2, dtypes.int32)
      s0 = constant_op.constant([2, 3, 5], dtypes.int32)
      s1 = constant_op.constant([2, 7, 5], dtypes.int32)
      s2 = constant_op.constant([2, 20, 5], dtypes.int32)
      off = gen_array_ops.concat_offset(cdim, [s0, s1, s2])
      ans = self.evaluate(off)
      self.assertAllEqual(ans, [[0, 0, 0], [0, 3, 0], [0, 10, 0]])

      cdim = constant_op.constant(-3, dtypes.int32)
      s0 = constant_op.constant([2, 3, 5], dtypes.int32)
      s1 = constant_op.constant([1, 3, 5], dtypes.int32)
      s2 = constant_op.constant([3, 3, 5], dtypes.int32)
      off = gen_array_ops.concat_offset(cdim, [s0, s1, s2])
      ans = self.evaluate(off)
      self.assertAllEqual(ans, [[0, 0, 0], [2, 0, 0], [3, 0, 0]])

  def testCreateMemDecBlockedFormat(self):
    # [FUNCTION DOC]
    # Only does anything when the MKL (oneDNN) backend is enabled; otherwise it silently passes.
    # Builds a large chain of strided slice, slice and convolution operations so that one concat
    # input has a blocked memory format, concatenates the two branches along the channel axis and
    # evaluates it. The only expectation is that creating and running the op does not crash.
    # Returns nothing.
    """Try to create the mkl concat operation

    when one of the input's memory descriptor is in blocked format
    """
    if test_util.IsMklEnabled():
      s0 = np.ones((1, 8188, 4092, 1), dtype=np.uint8).astype(np.float32)
      s1 = array_ops.strided_slice(
          s0, [0, 1, 1, 0], [0, -1, -1, 0], [1, 1, 1, 1],
          begin_mask=9,
          end_mask=9)
      s2 = array_ops.slice(s1, [0, 0, 0, 0], [-1, -1, -1, 1])
      s3_1 = array_ops.slice(s2, [0, 4, 4, 0], [-1, 8178, 4082, 1])
      s3_2 = array_ops.slice(s2, [0, 4, 4, 0], [-1, 8178, 4082, 1])
      filter4_1 = constant_op.constant([[[[1.18, -0.51]]]])
      s4_1 = nn_ops.conv2d(
          s3_1, filter4_1, strides=[1, 1, 1, 1], padding="VALID")
      filter4_2 = constant_op.constant([[[[1.38, -0.11]]]])
      s4_2 = nn_ops.conv2d(
          s3_2, filter4_2, strides=[1, 1, 1, 1], padding="VALID")
      s5_1 = array_ops.slice(s4_1, [0, 6, 6, 0], [-1, 1, 1, -1])
      s5_2 = array_ops.slice(s4_2, [0, 6, 6, 0], [-1, 1, 1, -1])
      x_concat = array_ops.concat([s5_1, s5_2], 3)
      self.evaluate(
          x_concat
      )  # This test is only meant to check the creation is not crashed

  def testInt64Shape(self):
    # [FUNCTION DOC]
    # Computes offsets along dimension 1 for 64-bit shape vectors where the first has a dimension
    # of five billion, with GPU preferred, and asserts the offsets (0, 5000000000, 5000000007)
    # and that the result dtype is 64-bit integer. Returns nothing.
    with test_util.use_gpu():
      cdim = constant_op.constant(1, dtypes.int32)
      s0 = constant_op.constant([2, 5000000000, 5], dtypes.int64)
      s1 = constant_op.constant([2, 7, 5], dtypes.int64)
      s2 = constant_op.constant([2, 20, 5], dtypes.int64)
      off = gen_array_ops.concat_offset(cdim, [s0, s1, s2])
      ans = self.evaluate(off)
      self.assertAllEqual(
          ans, [[0, 0, 0], [0, 5000000000, 0], [0, 5000000007, 0]])
      self.assertEqual(ans[0].dtype, dtypes.int64)


if __name__ == "__main__":
  test.main()
