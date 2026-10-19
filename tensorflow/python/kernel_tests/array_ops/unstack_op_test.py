# [MODULE DOC]
# Functional tests for the unstack operation, which splits a tensor along an axis into a list of
# lower-rank tensors. Contains a numpy-based reference helper and one test case covering many
# dtypes, shapes and axes, gradient checks, inference of the output count, error handling for
# unknown shapes and bad axes, and zero-length dimensions. No import-time side effects; running as
# a script starts the TensorFlow test runner.
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
"""Functional tests for Unstack Op."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test


def np_split_squeeze(array, axis):
  # [FUNCTION DOC]
  # HELPER: numpy reference implementation of unstack used by the against-numpy test.
  # Takes an array and an axis (may be negative); splits the array into as many pieces as that
  # axis is long and removes the split axis from each piece. Returns a list of arrays. Raises
  # whatever numpy raises for an out-of-range axis. No side effects.
  axis_len = array.shape[axis]
  return [
      np.squeeze(
          arr, axis=(axis,)) for arr in np.split(
              array, axis_len, axis=axis)
  ]


class UnstackOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the unstack op (array_ops_stack.unstack) on CPU and optionally GPU. Provides helper
  # methods for generating random data and computing reference results with numpy.

  def randn(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: generates random test data for the unstack tests.
    # Takes a shape (sequence of dimension sizes) and a numpy dtype. Draws standard-normal values;
    # for the boolean dtype returns whether each is negative (so about half are true), otherwise
    # casts to the dtype. Returns a numpy array. Uses the global numpy random state.
    data = np.random.randn(*shape)
    if dtype == np.bool_:
      return data < 0  # Naive casting yields True with P(1)!
    else:
      return data.astype(dtype)

  def unstackReference(self, data, axis):
    # [FUNCTION DOC]
    # HELPER: numpy reference for unstack used to check expected results.
    # Takes a numpy array and an axis (negative allowed). Returns a list with one entry per index
    # along that axis, each being the array slice at that index with the axis removed. No side
    # effects.
    """Use numpy primitives to implement unstack equivalent."""
    result = []
    rank = len(data.shape)
    axis = axis + rank if axis < 0 else axis
    for k in range(data.shape[axis]):
      axis = rank + axis if axis < 0 else axis
      # Slice in axis dimension of k'th slice.
      # e.g. if rank=4 k=2, axis=2 then equivalent of data[:,:,2,:]
      # Give error with loop context
      slice_spec = tuple(
          slice(None) if i != axis else k for i in range(rank))
      result.append(data.__getitem__(slice_spec))
    return result

  def testSimple(self):
    # [FUNCTION DOC]
    # Scenario: with a fixed random seed, for several shapes, every valid axis (positive and
    # negative) and many dtypes (bool, floats including float8 types, unsigned and signed ints),
    # random data is converted to a constant tensor and unstacked. Asserts the result is a list of
    # length equal to the axis size and each element equals the numpy reference slice (checked in
    # sub-tests). Side effect: seeds the global numpy random generator. Returns nothing.
    np.random.seed(7)
    for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
      rank = len(shape)
      for axis in range(-rank, rank):
        for dtype in [
            np.bool_,
            np.float16,
            np.float32,
            np.float64,
            np.uint8,
            np.int32,
            np.int64,
            dtypes.float8_e5m2.as_numpy_dtype,
            dtypes.float8_e4m3fn.as_numpy_dtype,
        ]:
          data = self.randn(shape, dtype)
          # Convert data to a single tensorflow tensor
          x = constant_op.constant(data)

          # Unstack into a list of tensors
          ref = self.unstackReference(data, axis)
          cs = array_ops_stack.unstack(x, axis=axis)
          self.assertEqual(type(cs), list)
          self.assertEqual(len(cs), shape[axis])
          for k, c in enumerate(cs):
            with self.subTest(shape=shape, k=k, axis=axis, dtype=dtype):
              self.assertAllEqual(ref[k], self.evaluate(c))

  def testSimpleGpu(self):
    # [FUNCTION DOC]
    # Same scenario as the basic test but restricted to a smaller dtype list and executed with
    # operations forced onto a GPU. Skipped when no GPU is available. Asserts list type, length and
    # element equality with the numpy reference. Seeds the global numpy random generator. Returns
    # nothing.
    if not test_util.is_gpu_available():
      self.skipTest('No GPU available')

    np.random.seed(7)
    with test_util.force_gpu():
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        rank = len(shape)
        for axis in range(-rank, rank):
          for dtype in [
              np.bool_, np.float16, np.float32, np.float64, np.uint8, np.int32,
              np.int64
          ]:
            data = self.randn(shape, dtype)
            # Convert data to a single tensorflow tensor
            x = constant_op.constant(data)
            # Unstack into a list of tensors
            ref = self.unstackReference(data, axis)
            cs = array_ops_stack.unstack(x, axis=axis)
            self.assertEqual(type(cs), list)
            self.assertEqual(len(cs), shape[axis])
            for k, c in enumerate(cs):
              # Give error with loop context
              with self.subTest(shape=shape, k=k, axis=axis, dtype=dtype):
                self.assertAllEqual(ref[k], self.evaluate(c))

  def testGradientsAxis0(self):
    # [FUNCTION DOC]
    # Scenario: for several shapes and each index along the first axis, a nested function that
    # unstacks its input along axis zero (with explicit count) and picks that one output is
    # numerically gradient-checked using the v2 gradient checker inside a cached session. Asserts
    # the maximum error between theoretical and numeric gradients is below one millionth. Returns
    # nothing.
    for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
      data = np.random.randn(*shape)
      x = constant_op.constant(data)

      for i in range(shape[0]):
        def func(x, shape=shape, i=i):
          # [FUNCTION DOC]
          # HELPER: function under gradient test for axis zero.
          # Takes the input tensor x plus the shape and index bound as defaults; unstacks x along
          # axis zero into shape-first-dimension pieces and returns the piece at index i.
          return array_ops_stack.unstack(x, num=shape[0])[i]

        with self.cached_session():
          err = gradient_checker_v2.max_error(
              *gradient_checker_v2.compute_gradient(func, [x]))
          self.assertLess(err, 1e-6)

  def testGradientsAxis1(self):
    # [FUNCTION DOC]
    # Scenario: like the axis-zero gradient test but unstacking along axis one for shapes of rank
    # two or more, checking each output slice's gradient numerically. Asserts max gradient error is
    # below one millionth. Returns nothing.
    for shape in (2, 3), (3, 2), (4, 3, 2):
      data = np.random.randn(*shape)
      x = constant_op.constant(data)

      for i in range(shape[1]):
        def func(x, shape=shape, i=i):
          # [FUNCTION DOC]
          # HELPER: function under gradient test for axis one.
          # Takes input tensor x plus the shape and index bound as defaults; unstacks x along axis
          # one into as many pieces as that dimension has and returns the piece at index i.
          return array_ops_stack.unstack(x, num=shape[1], axis=1)[i]

        with self.cached_session():
          err = gradient_checker_v2.max_error(
              *gradient_checker_v2.compute_gradient(func, [x]))
          self.assertLess(err, 1e-6)

  def testInferNum(self):
    # [FUNCTION DOC]
    # Scenario: unstack is called on a tensor of ones with no explicit count, so the count must be
    # inferred from the first dimension. Asserts a list is returned whose length equals that
    # dimension. Returns nothing.
    for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
      x = array_ops.ones(shape, dtype=np.float32)
      cs = array_ops_stack.unstack(x)
      self.assertEqual(type(cs), list)
      self.assertEqual(len(cs), shape[0])

  def testCannotInferNumFromUnknownShape(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a placeholder of completely unknown shape is unstacked without a
    # count. Asserts a value error saying the count cannot be inferred from an unknown shape.
    # Returns nothing.
    # Testing unknown shape in graph mode.
    with ops.Graph().as_default():
      x = array_ops.placeholder(np.float32)
      with self.assertRaisesRegex(
          ValueError, r'Cannot infer argument `num` from shape <unknown>'):
        array_ops_stack.unstack(x)

  def testUnknownShapeOkWithNum(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a placeholder of unknown shape is unstacked with an explicit count of
    # two. Asserts only that no error is raised. Returns nothing.
    # Testing unknown shape in graph mode.
    with ops.Graph().as_default():
      x = array_ops.placeholder(np.float32)
      array_ops_stack.unstack(x, num=2)

  def testCannotInferNumFromNoneShape(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): a placeholder with a one-dimensional shape whose size is unknown is
    # unstacked without a count. Asserts a value error saying the count cannot be inferred from
    # that partially unknown shape. Returns nothing.
    # Testing unknown shape in graph mode.
    with ops.Graph().as_default():
      x = array_ops.placeholder(np.float32, shape=(None,))
      with self.assertRaisesRegex(
          ValueError, r'Cannot infer argument `num` from shape \((\?|None),\)'):
        array_ops_stack.unstack(x)

  def testAgainstNumpy(self):
    # [FUNCTION DOC]
    # Scenario: for random arrays of rank one to five with randomly permuted dimension sizes and
    # every axis (including negative), unstack is compared with the numpy split-and-squeeze
    # helper. Asserts the evaluated results equal the expected list. Uses the global numpy random
    # state. Returns nothing.
    # For 1 to 5 dimensions.
    for i in range(1, 6):
      a = np.random.random(np.random.permutation(i) + 1)

      # For all the possible axis to split it, including negative indices.
      for j in range(-i, i):
        expected = np_split_squeeze(a, j)

        actual_unstack = self.evaluate(array_ops_stack.unstack(a, axis=j))

        self.assertAllEqual(expected, actual_unstack)

  def testAxis0Default(self):
    # [FUNCTION DOC]
    # Scenario: a 2x3 constant is unstacked with default axis. Asserts two outputs equal to the
    # first and second rows. Returns nothing.
    a = constant_op.constant([[1, 2, 3], [4, 5, 6]], name='a')
    unstacked = self.evaluate(array_ops_stack.unstack(a))

    self.assertEqual(len(unstacked), 2)
    self.assertAllEqual(unstacked[0], [1, 2, 3])
    self.assertAllEqual(unstacked[1], [4, 5, 6])

  def testAxisOutOfRange(self):
    # [FUNCTION DOC]
    # Scenario: a rank-two constant is unstacked with axis two. Asserts a value error stating the
    # axis is not in the valid half-open range from minus two to two. Returns nothing.
    a = constant_op.constant([[1, 2, 3], [4, 5, 6]], name='a')
    with self.assertRaisesRegex(ValueError,
                                r'Argument `axis` = 2 not in range \[-2, 2\)'):
      array_ops_stack.unstack(a, axis=2)

  def testAxisOutOfNegativeRange(self):
    # [FUNCTION DOC]
    # Scenario: a rank-two constant is unstacked with axis minus three. Asserts a value error
    # stating the axis is not in the valid range from minus two up to but excluding two. Returns
    # nothing.
    a = constant_op.constant([[1, 2, 3], [4, 5, 6]], name='a')
    with self.assertRaisesRegex(ValueError,
                                r'Argument `axis` = -3 not in range \[-2, 2\)'):
      array_ops_stack.unstack(a, axis=-3)

  def testZeroLengthDim(self):
    # [FUNCTION DOC]
    # Scenario: a zeros tensor with a zero-length first dimension is unstacked along axis one.
    # Asserts the first output has shape zero by two. Returns nothing.
    x = array_ops.zeros(shape=(0, 1, 2))
    y = self.evaluate(array_ops_stack.unstack(x, axis=1)[0])
    self.assertEqual(y.shape, (0, 2))

  def testComplexGpu(self):
    # [FUNCTION DOC]
    # Scenario: complex64 and complex128 random data of several shapes is unstacked with an
    # explicit count on a GPU (operations forced onto the GPU). Skipped when no GPU is available.
    # Asserts list type, length and that the evaluated pieces equal the original data. Seeds the
    # global numpy random generator. Returns nothing.
    if not test_util.is_gpu_available():
      self.skipTest('No GPU available')

    np.random.seed(7)
    with test_util.force_gpu():
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        for dtype in [np.complex64, np.complex128]:
          data = np.random.randn(*shape).astype(dtype)
          # Convert data to a single tensorflow tensor
          x = constant_op.constant(data)
          # Unstack into a list of tensors
          cs = array_ops_stack.unstack(x, num=shape[0])
          self.assertEqual(type(cs), list)
          self.assertEqual(len(cs), shape[0])
          cs = [self.evaluate(c) for c in cs]
          self.assertAllEqual(cs, data)


if __name__ == '__main__':
  test.main()
