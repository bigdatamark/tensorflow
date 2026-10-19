# [MODULE DOC]
# Tests for the tensor padding op in its constant, reflect and symmetric modes. A single test
# case compares results against numpy padding for many element types (integers, floats, bfloat16,
# complex, quantized-style structured types, strings), checks numerical gradients, error handling
# for malformed paddings, static shape inference with partially unknown shapes and paddings, scalar
# inputs, and adjacent-dimension collapsing behavior. No import-time side effects besides running
# the test main when executed as a script.
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
"""Tests for tensorflow.ops.nn_ops.Pad."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test


class PadOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the pad op. Holds numpy dtypes standing in for quantized types (created in
  # setUp) and private helpers that compute numpy reference padding, compare TensorFlow results
  # on GPU and CPU placements, and check gradients; test methods exercise specific scenarios.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture run before each test: calls the base setup, then stores as instance attributes three
    # numpy structured dtypes (unsigned 8-bit, signed 8-bit and signed 32-bit, each wrapped in a
    # single named field) that serve as stand-ins for the quantized TensorFlow types.
    super().setUp()
    # Create mapping between TensorFlow quantized types and numpy types.
    self._quint8 = np.dtype([("quint8", np.uint8)])
    self._qint8 = np.dtype([("qint8", np.int8)])
    self._qint32 = np.dtype([("qint32", np.int32)])

  def _npPad(self, inp, paddings, mode, constant_values=0):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation of padding used to compute expected values.
    # Inputs: inp array, paddings (list of before/after pairs per dimension), mode name (case
    # insensitive; constant, reflect or symmetric) and constant_values fill (default zero, only
    # used in constant mode). Returns the padded numpy array. No side effects.
    mode = mode.lower()
    if mode == "constant":
      return np.pad(inp, paddings, mode=mode, constant_values=constant_values)
    else:
      return np.pad(inp, paddings, mode=mode)

  def testNpPad(self):
    # [FUNCTION DOC]
    # Sanity test of the numpy reference helper itself: asserts exact expected arrays for padding
    # a small matrix in constant mode with fill zero, constant mode with fill one, reflect mode
    # and symmetric mode.
    self.assertAllEqual(
        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 3, 3, 0, 0, 0],
                  [0, 4, 4, 0, 0, 0],
                  [0, 5, 5, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]),
        self._npPad(
            np.array([[3, 3], [4, 4], [5, 5]]),
            [[1, 2], [1, 3]],
            mode="constant"))

    self.assertAllEqual(
        np.array([[1, 1, 1, 1, 1, 1],
                  [1, 3, 3, 1, 1, 1],
                  [1, 4, 4, 1, 1, 1],
                  [1, 5, 5, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1]]),
        self._npPad(
            np.array([[3, 3], [4, 4], [5, 5]]),
            [[1, 2], [1, 3]],
            mode="constant", constant_values=1))

    self.assertAllEqual(
        np.array([[4, 3, 4, 9, 4, 3],
                  [1, 0, 1, 2, 1, 0],
                  [4, 3, 4, 9, 4, 3],
                  [1, 0, 1, 2, 1, 0]]),
        self._npPad(
            np.array([[0, 1, 2], [3, 4, 9]]),
            [[1, 1], [1, 2]],
            mode="reflect"))

    self.assertAllEqual(
        np.array([[0, 0, 1, 2, 2, 1],
                  [0, 0, 1, 2, 2, 1],
                  [3, 3, 4, 9, 9, 4],
                  [3, 3, 4, 9, 9, 4]]),
        self._npPad(
            np.array([[0, 1, 2], [3, 4, 9]]),
            [[1, 1], [1, 2]],
            mode="symmetric"))

  def _testPad(self, np_inputs, paddings, mode, constant_values):
    # [FUNCTION DOC]
    # HELPER: compares the TensorFlow pad op against the numpy reference. Inputs: numpy input
    # array, paddings, mode name and fill value. For GPU and then CPU placement, pads the input
    # with the op and evaluates it; for the structured stand-in dtypes it converts the expected
    # array to the underlying scalar type. Asserts values equal and static shape matches.
    np_val = self._npPad(np_inputs, paddings, mode=mode,
                         constant_values=constant_values)
    for use_gpu in [True, False]:
      with test_util.device(use_gpu=use_gpu):
        tf_val = array_ops.pad(np_inputs, paddings, mode=mode,
                               constant_values=constant_values)
        out = self.evaluate(tf_val)

        if np_inputs.dtype in [self._qint8, self._quint8, self._qint32]:
          # Cast quantized types back to their numpy equivalents.
          np_val = np_val.astype(np_inputs.dtype[0])

      self.assertAllEqual(np_val, out)
      self.assertShapeEqual(np_val, tf_val)

  def _testGradient(self,
                    x,
                    a,
                    mode,
                    constant_values,
                    paddings_dtype=dtypes.int32):
    # [FUNCTION DOC]
    # HELPER: gradient check for the pad op. Inputs: x input array, a paddings list, mode name,
    # constant fill value and paddings_dtype (default 32-bit integer) for the paddings tensor.
    # Computes theoretical and numerical Jacobians with the version-2 gradient checker in a cached
    # session; for bfloat16 inputs, recomputes the numerical Jacobian using a float32 copy. Asserts
    # they are close with tolerance 1e-3 for float16, otherwise 4e-5.

    def pad(x):
      # [FUNCTION DOC]
      # HELPER: closure wrapping the pad op with the enclosing paddings, mode and fill value so
      # the gradient checker can call it on an input tensor. Returns the padded tensor.
      return array_ops.pad(
          x,
          ops.convert_to_tensor(a, paddings_dtype),
          mode=mode,
          constant_values=constant_values)

    with self.cached_session():
      jacob_t, jacob_n = gradient_checker_v2.compute_gradient(pad, [x])
      if x.dtype == dtypes.bfloat16.as_numpy_dtype:
        # Compare bf16 analytical gradients to fp32 numerical gradients.
        x_fp32 = constant_op.constant(x, shape=x.shape, dtype=dtypes.float32)
        _, jacob_n = gradient_checker_v2.compute_gradient(pad, [x_fp32])
      tol = 1e-3 if x.dtype == np.float16 else 4e-5
      self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _testAll(self, np_inputs, paddings, constant_values):
    # [FUNCTION DOC]
    # HELPER: runs the value check, and the gradient check for float32, float16 and bfloat16
    # inputs, over all six spellings of the three padding modes (upper and lower case). Skips
    # reflect mode for zero-size inputs and reflect/symmetric for any input with a zero-length
    # dimension, since those are not allowed. Inputs: numpy array, paddings, fill value.
    for mode in ("CONSTANT", "REFLECT", "SYMMETRIC", "reflect", "symmetric",
                 "constant"):
      # Zero-sized input is not allowed for REFLECT mode, but we still want
      # zero-sized input test cases for the other modes.
      if not np_inputs.size and mode.upper() == "REFLECT":
        continue
      # Empty tensor is not allowed for MirrorPad.
      if 0 in np_inputs.shape and mode.upper() in ["REFLECT", "SYMMETRIC"]:
        continue
      self._testPad(
          np_inputs, paddings, mode=mode, constant_values=constant_values)
      if np_inputs.dtype in [
          np.float32, np.float16, dtypes.bfloat16.as_numpy_dtype
      ]:
        self._testGradient(
            np_inputs, paddings, mode=mode, constant_values=constant_values)

  def testInputDims(self):
    # [FUNCTION DOC]
    # Asserts that padding a rank-6 tensor with a paddings tensor of shape two by two raises a
    # value error or invalid-argument error about the paddings rank not matching the input rank.
    with test_util.use_gpu():
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "Shape must be rank 1 but is rank 6|"
                                  "paddings must be the rank of inputs"):
        array_ops.pad(array_ops.reshape(
            [1, 2], shape=[1, 2, 1, 1, 1, 1]),
                      array_ops.reshape(
                          [1, 2], shape=[1, 2]))

  def testPaddingsDim(self):
    # [FUNCTION DOC]
    # Asserts that a rank-1 paddings tensor (shape two) for a rank-2 input raises a value error or
    # invalid-argument error saying paddings must be a matrix with two columns.
    with test_util.use_gpu():
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "Shape must be rank 2 but is rank 1|"
                                  "paddings must be a matrix with 2 columns"):
        array_ops.pad(array_ops.reshape(
            [1, 2], shape=[1, 2]),
                      array_ops.reshape(
                          [1, 2], shape=[2]))

  def testPaddingsDim2(self):
    # [FUNCTION DOC]
    # Asserts that a paddings tensor of shape two by one (one column instead of two) raises a
    # value error or invalid-argument error about the column dimension.
    with test_util.use_gpu():
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "Dimension must be 2 but is 1|"
                                  "paddings must be a matrix with 2 columns"):
        array_ops.pad(array_ops.reshape(
            [1, 2], shape=[1, 2]),
                      array_ops.reshape(
                          [1, 2], shape=[2, 1]))

  def testPaddingsDim3(self):
    # [FUNCTION DOC]
    # Asserts that a paddings tensor of shape one by two for a rank-2 input (too few rows) raises
    # a value error or invalid-argument error about the paddings rank not matching the input.
    with test_util.use_gpu():
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "Shape must be rank 1 but is rank 2|"
                                  "paddings must be the rank of inputs"):
        array_ops.pad(array_ops.reshape(
            [1, 2], shape=[1, 2]),
                      array_ops.reshape(
                          [1, 2], shape=[1, 2]))

  def testPaddingsDim4(self):
    # [FUNCTION DOC]
    # Asserts that a paddings tensor of shape three by two (too many rows) for a rank-2 input
    # raises a value error or invalid-argument error about the paddings rank.
    with test_util.use_gpu():
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "Shape must be rank 3 but is rank 2|"
                                  "paddings must be the rank of inputs"):
        array_ops.pad(array_ops.reshape(
            [1, 2], shape=[1, 2]),
                      array_ops.reshape(
                          [1, 2, 3, 4, 5, 6], shape=[3, 2]))

  def testPaddingsNonNegative(self):
    # [FUNCTION DOC]
    # Asserts that a negative padding amount raises a value error or invalid-argument error
    # stating paddings must be non-negative.
    with test_util.use_gpu():
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "must be non-negative"):
        array_ops.pad(constant_op.constant(
            [1], shape=[1]),
                      constant_op.constant(
                          [-1, 0], shape=[1, 2]))

  def testPaddingsNonNegative2(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the body is an exact duplicate of the non-negative paddings test above; the
    # name suggests a different scenario but nothing different is exercised.
    # Asserts that a negative padding amount raises a value error or invalid-argument error
    # stating paddings must be non-negative.
    with test_util.use_gpu():
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "must be non-negative"):
        array_ops.pad(constant_op.constant(
            [1], shape=[1]),
                      constant_op.constant(
                          [-1, 0], shape=[1, 2]))

  def testPaddingsMaximum(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the op results are evaluated with the implicit-session eval method with no
    # session set up, so the exception raised may come from that missing session (or eager mode)
    # rather than from the padding limit; the generic exception assertion cannot tell them apart.
    # Asserts that some exception is raised when reflect mode pads by more than allowed (before
    # padding two) or symmetric mode pads after by three, on a length-two vector.
    with test_util.use_gpu():
      with self.assertRaises(Exception):
        array_ops.pad(constant_op.constant(
            [1], shape=[2]),
                      constant_op.constant(
                          [2, 0], shape=[1, 2]),
                      mode="REFLECT").eval()
      with self.assertRaises(Exception):
        array_ops.pad(constant_op.constant(
            [1], shape=[2]),
                      constant_op.constant(
                          [0, 3], shape=[1, 2]),
                      mode="SYMMETRIC").eval()

  def testInvalid(self):
    # [FUNCTION DOC]
    # Asserts that an unrecognized mode name raises a value error whose message says the mode
    # argument was expected to be one of the valid options and received the bad value.
    with self.cached_session():
      x = [[1, 2, 3], [4, 5, 6]]
      with self.assertRaisesRegex(
          ValueError,
          "Value of argument `mode` expected to be .* Received `mode` = WEIRD"):
        self.evaluate(array_ops.pad(x, [[1, 0], [2, 1]], mode="weird"))

  def testPaddingTypes(self):
    # [FUNCTION DOC]
    # For every mode spelling and for both 32-bit and 64-bit integer paddings tensors, pads a
    # random float matrix, asserts equality with the numpy reference and shape match, and for
    # non-reflect modes also runs the gradient check in a fresh graph.
    paddings = [[1, 0], [2, 0]]
    inputs = np.random.rand(2, 5).astype(np.float32)
    for mode in ("CONSTANT", "REFLECT", "SYMMETRIC", "reflect", "symmetric",
                 "constant"):
      for paddings_dtype in [dtypes.int32, dtypes.int64]:
        np_val = self._npPad(inputs,
                             paddings,
                             mode=mode,
                             constant_values=0)

        with test_util.use_gpu():
          tf_val = array_ops.pad(
              inputs,
              constant_op.constant(paddings, paddings_dtype),
              mode=mode,
              constant_values=0)
          out = self.evaluate(tf_val)

        self.assertAllEqual(np_val, out)
        self.assertShapeEqual(np_val, tf_val)

        if mode.upper() != "REFLECT":
          with ops.Graph().as_default():
            self._testGradient(
                inputs,
                paddings,
                mode=mode,
                constant_values=0,
                paddings_dtype=paddings_dtype)

  def testQuantizedTypes(self):
    # [FUNCTION DOC]
    # For each structured stand-in quantized dtype, runs the full value/gradient check on random
    # integer data, once with nonzero paddings and fill zero, and once with zero paddings on a
    # rank-4 input and a fill value of 123.
    for t in [self._qint8, self._quint8, self._qint32]:
      self._testAll(
          np.random.randint(-100, 100, (4, 4, 3)).astype(t),
          [[1, 0], [2, 3], [0, 2]], 0)
      self._testAll(
          np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t),
          [[0, 0], [0, 0], [0, 0], [0, 0]],
          np.array(123).astype(t))

  def testIntTypes(self):
    # [FUNCTION DOC]
    # Runs the full check for int8, uint8, int32 and int64 random data with nonzero paddings and
    # fill zero, and with zero paddings on a rank-4 input and fill value minus 123.
    # TODO(touts): Figure out why the padding tests do not work on GPU
    # for int types and rank > 2.
    for t in [np.int8, np.uint8, np.int32, np.int64]:
      self._testAll(
          np.random.randint(-100, 100, (4, 4, 3)).astype(t),
          [[1, 0], [2, 3], [0, 2]], 0)
      self._testAll(
          np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t),
          [[0, 0], [0, 0], [0, 0], [0, 0]], -123)

  def testFloatTypes(self):
    # [FUNCTION DOC]
    # Runs the full value and gradient check for float16, float32, float64 and bfloat16 random
    # inputs of several shapes with different paddings and fill values (0.0, -12.34, 1.41).
    for t in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._testAll(np.random.rand(2, 5).astype(t), [[1, 0], [2, 0]], 0.0)
      self._testAll(
          np.random.rand(2, 3, 4).astype(t), [[0, 0], [0, 0], [0, 0]], -12.34)
      self._testAll(
          np.random.rand(1, 3, 4).astype(t), [[0, 0], [1, 1], [2, 2]], 1.41)

  def testEmptyTensor(self):
    # [FUNCTION DOC]
    # Runs the full check on a float input with a zero-length first dimension for four float
    # dtypes, padding every dimension, verifying constant padding of empty inputs works.
    for t in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._testAll(np.random.rand(0, 3, 4).astype(t),
                    [[0, 0], [2, 1], [2, 3]], 0.0)

  def testComplexTypes(self):
    # [FUNCTION DOC]
    # For complex64 and complex128, builds random complex inputs of rank 2 and rank 4 and runs the
    # full check with complex fill values and nonzero or zero paddings.
    for t in [np.complex64, np.complex128]:
      x = np.random.rand(2, 5).astype(t)
      self._testAll(x + 1j * x, [[1, 0], [2, 0]], 1234.0 - 1234.0j)
      x = np.random.rand(3, 2, 1, 1).astype(t)
      self._testAll(x + 1j * x, [[0, 0], [0, 0], [0, 0], [0, 0]], 0 + 0j)

  def testString(self):
    # [FUNCTION DOC]
    # Pads a two by two string tensor in constant (fill "PAD"), reflect and symmetric modes and
    # asserts the exact expected byte-string matrices, since numpy cannot pad strings.
    # Numpy does not support padding strings so we compare padding manually.
    x = ops.convert_to_tensor([["Hello", "World"],
                               ["Goodnight", "Moon"]])

    constant = array_ops.pad(x, [[1, 0], [0, 1]], mode="CONSTANT",
                             constant_values="PAD")
    reflect = array_ops.pad(x, [[1, 0], [0, 1]], mode="REFLECT",
                            constant_values="PAD")
    symmetric = array_ops.pad(x, [[1, 0], [0, 1]], mode="SYMMETRIC",
                              constant_values="PAD")
    with test_util.use_gpu():
      self.assertAllEqual(
          [[b"PAD", b"PAD", b"PAD"], [b"Hello", b"World", b"PAD"],
           [b"Goodnight", b"Moon", b"PAD"]], self.evaluate(constant))
      self.assertAllEqual([[b"Goodnight", b"Moon", b"Goodnight"],
                           [b"Hello", b"World", b"Hello"],
                           [b"Goodnight", b"Moon", b"Goodnight"]],
                          self.evaluate(reflect))
      self.assertAllEqual(
          [[b"Hello", b"World", b"World"], [b"Hello", b"World", b"World"],
           [b"Goodnight", b"Moon", b"Moon"]], self.evaluate(symmetric))

  def testShapeFunctionEdgeCases(self):
    # [FUNCTION DOC]
    # In a graph, checks static output shape inference: unknown paddings give all-unknown
    # dimensions for a known rank, unknown input shape with known paddings gives rank two unknown,
    # and both unknown gives unknown rank.
    # Shape function requires placeholders and a graph
    with ops.Graph().as_default():
      # Unknown paddings shape.
      inp = constant_op.constant(0.0, shape=[4, 4, 4, 4])
      padded = array_ops.pad(inp, array_ops.placeholder(dtypes.int32))
      self.assertEqual([None, None, None, None], padded.get_shape().as_list())

      # Unknown input shape.
      inp = array_ops.placeholder(dtypes.float32)
      padded = array_ops.pad(inp, [[2, 2], [2, 2]])
      self.assertEqual([None, None], padded.get_shape().as_list())

      # Unknown input and paddings shape.
      inp = array_ops.placeholder(dtypes.float32)
      padded = array_ops.pad(inp, array_ops.placeholder(dtypes.int32))
      self.assertAllEqual(None, padded.get_shape().ndims)

  def testPartialShapeInformation(self):
    # [FUNCTION DOC]
    # In a graph, checks static shape inference when paddings are partially unknown: unknown
    # amounts in one dimension, in the begin or the end, unknown input shape with and without
    # known rank, constant begin or whole-dimension paddings, and zero padding keeping a known
    # dimension. Asserts the inferred shape lists with unknown entries where expected.
    # Partial shapes requires placeholders and a graph
    with ops.Graph().as_default():
      unknown = array_ops.placeholder(dtypes.int32)

      # Known input shape, partial unknown padding (one dimension).
      inp = constant_op.constant(0.0, shape=[4, 4])
      padded = array_ops.pad(inp, [[1, 2], unknown])
      self.assertEqual([7, None], padded.get_shape().as_list())

      # Known input shape, partial unknown padding (begin).
      inp = constant_op.constant(0.0, shape=[4, 4])
      padded = array_ops.pad(inp, [[unknown, 0], [1, 2]])
      self.assertEqual([None, 7], padded.get_shape().as_list())

      # Known input shape, partial unknown padding (end).
      inp = constant_op.constant(0.0, shape=[4, 4])
      padded = array_ops.pad(inp, [[1, 2], [0, unknown]])
      self.assertEqual([7, None], padded.get_shape().as_list())

      # Unknown input shape, partial unknown padding (one dimension).
      padded = array_ops.pad(unknown, [[1, 2], unknown])
      self.assertEqual([None, None], padded.get_shape().as_list())

      # Unknown input shape (rank known), partial unknown padding (one dim).
      rank_known = array_ops.placeholder(dtypes.int32)
      rank_known.set_shape([None, None])
      padded = array_ops.pad(rank_known, [[1, 2], unknown])
      self.assertEqual([None, None], padded.get_shape().as_list())

      # Known input shape, partial unknown padding (begin), with constant begin.
      inp = constant_op.constant(0.0, shape=[4, 4])
      padded = array_ops.pad(
          inp, [[constant_op.constant(1, shape=[]), 2], [0, unknown]])
      self.assertEqual([7, None], padded.get_shape().as_list())

      # Known input shape, partial unknown padding (begin), with constant dim.
      inp = constant_op.constant(0.0, shape=[4, 4])
      padded = array_ops.pad(inp,
                             [constant_op.constant(1, shape=[2]), [0, unknown]])
      self.assertEqual([6, None], padded.get_shape().as_list())

      # Zero padding on a known dimension.
      inp = array_ops.placeholder(dtypes.int32, [None, None, 20])
      padded = array_ops.pad(inp, [[0, 0], [0, unknown], [0, 0]])
      self.assertEqual([None, None, 20], padded.get_shape().as_list())

  def testScalars(self):
    # [FUNCTION DOC]
    # Pads a scalar with an empty (zero by two) paddings array and asserts the result equals the
    # input and the static shape matches.
    paddings = np.zeros((0, 2), dtype=np.int32)
    inp = np.asarray(7)
    with test_util.use_gpu():
      tf_val = array_ops.pad(inp, paddings)
      out = self.evaluate(tf_val)
    self.assertAllEqual(inp, out)
    self.assertShapeEqual(inp, tf_val)

  def testPadTypes(self):
    # [FUNCTION DOC]
    # Pads a scalar with an empty paddings tensor of 32-bit then 64-bit integer type and asserts
    # the value and static shape are unchanged.
    for dtype in [dtypes.int32, dtypes.int64]:
      paddings = np.zeros((0, 2))
      inp = np.asarray(7)
      with self.cached_session():
        tf_val = array_ops.pad(inp, constant_op.constant(paddings, dtype=dtype))
        out = self.evaluate(tf_val)
      self.assertAllEqual(inp, out)
      self.assertShapeEqual(inp, tf_val)

  def testCollapseAdjacentNonPaddedDimensions(self):
    # [FUNCTION DOC]
    # For three paddings patterns on a four-dimensional tensor of ones (float32 and int32),
    # pads, then slices out the middle (original data), the leading border and the trailing
    # border, asserting the middle equals the input and the borders are all zeros of the padded
    # sizes. Exercises the kernel path that merges adjacent unpadded dimensions.
    # pyformat: disable
    paddings_values = [[[0, 0], [0, 0], [0, 0], [0, 1]],
                       [[0, 0], [2, 3], [0, 0], [0, 0]],
                       [[0, 0], [0, 0], [0, 0], [0, 0]]]
    # pyformat: enable
    for paddings_value in paddings_values:
      for dtype in [dtypes.float32, dtypes.int32]:
        inp = constant_op.constant(1, shape=[8, 28, 28, 3], dtype=dtype)
        paddings = constant_op.constant(paddings_value, dtype=dtypes.int32)
        padded = array_ops.pad(inp, paddings)
        middle = array_ops.slice(padded, [row[0] for row in paddings_value],
                                 [dim.value for dim in inp.shape.dims])
        left = array_ops.slice(padded, [0, 0, 0, 0],
                               [row[0] for row in paddings_value])
        right = array_ops.slice(
            padded,
            [paddings_value[i][0] + inp.shape.dims[i].value for i in range(4)],
            [-1, -1, -1, -1])
        with self.cached_session():
          self.assertAllEqual(inp, self.evaluate(middle))
          self.assertAllEqual(
              np.zeros([row[0] for row in paddings_value]), self.evaluate(left))
          self.assertAllEqual(
              np.zeros([row[1] for row in paddings_value]),
              self.evaluate(right))


if __name__ == "__main__":
  test.main()
