# [MODULE DOC]
# Purpose: unit tests for the math operations module (reductions, log-sum-exp, rounding, matrix
# multiplication including mixed integer and bfloat16 variants, sampled sparse-dense matrix
# multiply-add, modulus, squared difference, approximate equality, scalar multiply, n-ary add,
# division and modulo variants, divide-no-nan, and further sections below).
# Each area is a separate test class, most of them run in both graph and eager modes. Results are
# compared against numpy reference computations or hand-computed expected values.
# Exposes only test classes; running the file as a script starts the standard test runner.
# Import-time side effects: none beyond imports.
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
"""Tests for tensorflow.ops.math_ops."""
from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class ReduceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the reduction ops: sum, count of non-zero entries, variance and standard deviation
  # (including complex and ragged inputs). Runs in both graph and eager modes. No shared fixtures.

  def testReduceAllDims(self):
    # [FUNCTION DOC]
    # Sums a 2 by 3 integer array with no axis given on the GPU-preferred device and asserts the
    # total over all elements is 21.
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with test_util.device(use_gpu=True):
      y_tf = self.evaluate(math_ops.reduce_sum(x))
      self.assertEqual(y_tf, 21)

  def testReduceExtendType(self):
    # [FUNCTION DOC]
    # Sums a large random float32 matrix directly and after casting to bfloat16, then asserts the
    # bfloat16 sum is close (relative tolerance 1e-3) to the float32 sum cast to bfloat16. Checks
    # that low-precision sums accumulate in a wider type.
    in_f32 = np.random.randn(1000, 1000).astype(np.float32)
    in_bf16 = math_ops.cast(in_f32, dtypes.bfloat16)

    out_f32 = self.evaluate(math_ops.reduce_sum(in_f32))
    out_bf16 = self.evaluate(math_ops.reduce_sum(in_bf16))
    expected = math_ops.cast(out_f32, dtypes.bfloat16)

    self.assertAllClose(out_bf16, expected, 1e-3)

  def testCountNonzero(self):
    # [FUNCTION DOC]
    # Checks the non-zero counting op on three inputs: an integer matrix (expects 2), a boolean
    # mask derived from it (expects 2), and 512 int8 values of 2 (expects 512, which would overflow
    # if an 8-bit accumulator were used).
    # simple case
    x = np.array([[0, -2, 0], [4, 0, 0]], dtype=np.int32)
    self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 2)

    # boolean input
    x = math_ops.not_equal(x, 0)
    self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 2)

    # would overflow if int8 would be used for internal calculations
    x = 2 * np.ones(512, dtype=np.int8)
    self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 512)

  def testReduceExplicitAxes(self):
    # [FUNCTION DOC]
    # Sums a 2 by 3 integer array with explicit positive and negative single axes (expects column
    # sums 5, 7, 9 and row sums 6, 15) and with no axis or axis tuples covering both dimensions in
    # any order and sign (expects 21).
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with test_util.device(use_gpu=True):
      for axis in (0, -2):
        self.assertAllEqual(
            self.evaluate(math_ops.reduce_sum(x, axis=axis)), [5, 7, 9])
      for axis in (1, -1):
        self.assertAllEqual(
            self.evaluate(math_ops.reduce_sum(x, axis=axis)), [6, 15])
      for axis in (None, (0, 1), (1, 0), (-1, 0), (0, -1), (-2, 1), (1, -2),
                   (-1, -2), (-2, -1)):
        self.assertEqual(self.evaluate(math_ops.reduce_sum(x, axis=axis)), 21)

  def testReduceInvalidAxis(self):
    # [FUNCTION DOC]
    # Graph mode only (returns immediately in eager mode because the rank check happens at graph
    # construction time). Passes a rank-2 axis argument to the sum and asserts a value error whose
    # message says the axis must be at most rank 1.
    if context.executing_eagerly():
      # The shape check is in run a graph construction time. In eager mode,
      # it misses the check, magically return result given wrong shape.
      return
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([[0], [1]])
    with self.assertRaisesRegex(ValueError, "must be at most rank 1"):
      math_ops.reduce_sum(x, axis)

  def testReduceVar(self):
    # [FUNCTION DOC]
    # Variance op scenarios: all-zero float input (overall and along axis 0) gives zeros; integer
    # input raises a type error saying real or complex is needed; a float list, and its numpy
    # equivalent, give 0.25 (matching numpy); a ragged tensor reduced along axis 0 gives
    # 0, 16, 1, 0.
    x = np.array([[0, 0, 0], [0, 0, 0]], "float32")
    self.assertAllClose(self.evaluate(math_ops.reduce_variance(x)), 0)
    self.assertAllClose(
        self.evaluate(math_ops.reduce_variance(x, axis=0)), [0, 0, 0])

    x = [[1, 2, 1, 1], [1, 1, 0, 1]]
    with self.assertRaisesRegex(TypeError, "must be either real or complex"):
      math_ops.reduce_variance(x)

    x = [[1., 2., 1., 1.], [1., 1., 0., 1.]]
    self.assertEqual(self.evaluate(math_ops.reduce_variance(x)), 0.25)
    x_np = np.array(x)
    self.assertEqual(np.var(x_np), 0.25)
    self.assertEqual(self.evaluate(math_ops.reduce_variance(x_np)), 0.25)

    x = ragged_factory_ops.constant([[5., 1., 4., 1.], [], [5., 9., 2.], [5.],
                                     []])
    self.assertAllClose(math_ops.reduce_variance(x, axis=0), [0., 16., 1., 0.])

  def testReduceVarComplex(self):
    # [FUNCTION DOC]
    # For a python complex list and complex64/complex128 arrays holding minus i and plus i, asserts
    # numpy's variance is 1.0, the variance op evaluates to 1.0, and the result dtype is the
    # matching real type (float64 for list and complex128, float32 for complex64).
    # Ensure that complex values are handled to be consistent with numpy
    complex_ys = [([0 - 1j, 0 + 1j], dtypes.float64),
                  (np.array([0 - 1j, 0 + 1j], "complex64"), dtypes.float32),
                  (np.array([0 - 1j, 0 + 1j], "complex128"), dtypes.float64)]
    for y, dtype in complex_ys:
      y_result = math_ops.reduce_variance(y)
      self.assertEqual(np.var(y), 1.0)
      self.assertEqual(self.evaluate(y_result), 1.0)
      self.assertEqual(y_result.dtype, dtype)

  def testReduceStd(self):
    # [FUNCTION DOC]
    # Standard deviation analog of the variance test: zeros give zeros; integer input raises a type
    # error; a float list and numpy array give 0.5 (matching numpy); a ragged tensor along axis 0
    # gives 0, 4, 1, 0.
    x = np.array([[0, 0, 0], [0, 0, 0]], "float32")
    self.assertAllClose(self.evaluate(math_ops.reduce_std(x)), 0)
    self.assertAllClose(
        self.evaluate(math_ops.reduce_std(x, axis=0)), [0, 0, 0])

    x = [[1, 2, 1, 1], [1, 1, 0, 1]]
    with self.assertRaisesRegex(TypeError, "must be either real or complex"):
      math_ops.reduce_std(x)

    x = [[1., 2., 1., 1.], [1., 1., 0., 1.]]
    self.assertEqual(self.evaluate(math_ops.reduce_std(x)), 0.5)
    x_np = np.array(x)
    self.assertEqual(np.std(x_np), 0.5)
    self.assertEqual(self.evaluate(math_ops.reduce_std(x_np)), 0.5)

    x = ragged_factory_ops.constant([[5., 1., 4., 1.], [], [5., 9., 2.], [5.],
                                     []])
    self.assertAllClose(math_ops.reduce_std(x, axis=0), [0., 4., 1., 0.])

  def testReduceStdComplex(self):
    # [FUNCTION DOC]
    # Same as the complex variance test but for standard deviation: expects 1.0 for the pure
    # imaginary pairs and the real dtype matching the complex input width.
    # Ensure that complex values are handled to be consistent with numpy
    complex_ys = [([0 - 1j, 0 + 1j], dtypes.float64),
                  (np.array([0 - 1j, 0 + 1j], "complex64"), dtypes.float32),
                  (np.array([0 - 1j, 0 + 1j], "complex128"), dtypes.float64)]
    for y, dtype in complex_ys:
      y_result = math_ops.reduce_std(y)
      self.assertEqual(np.std(y), 1.0)
      self.assertEqual(self.evaluate(y_result), 1.0)
      self.assertEqual(y_result.dtype, dtype)


@test_util.run_all_in_graph_and_eager_modes
class LogSumExpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the log-sum-exp reduction: plain, axis and keep-dimension forms, numerical overflow
  # and underflow stability, negative infinity input, and ragged input. Runs in graph and eager
  # modes over half, single and double precision numpy data.

  def testReduceLogSumExp(self):
    # [FUNCTION DOC]
    # For float16/32/64 random 5 by 5 data, asserts the op over all elements is close to the naive
    # numpy log of summed exponentials.
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf_np = math_ops.reduce_logsumexp(x_np)
        y_np = np.log(np.sum(np.exp(x_np)))
        self.assertAllClose(y_tf_np, y_np)

  def testReductionIndices(self):
    # [FUNCTION DOC]
    # Reduces over axis given as a one-element list; asserts the result's static shape matches
    # numpy's and the values are close to the numpy reference, for three float types.
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf = math_ops.reduce_logsumexp(x_np, axis=[0])
        y_np = np.log(np.sum(np.exp(x_np), axis=0))
        self.assertShapeEqual(y_np, y_tf)
        y_tf_np = self.evaluate(y_tf)
        self.assertAllClose(y_tf_np, y_np)

  def testReductionIndices2(self):
    # [FUNCTION DOC]
    # Same as the list-axis test but passes axis as a plain integer 0; asserts shape and values
    # match the numpy reference for three float types.
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf = math_ops.reduce_logsumexp(x_np, axis=0)
        y_np = np.log(np.sum(np.exp(x_np), axis=0))
        self.assertShapeEqual(y_np, y_tf)
        y_tf_np = self.evaluate(y_tf)
        self.assertAllClose(y_tf_np, y_np)

  def testKeepDims(self):
    # [FUNCTION DOC]
    # Reduces with the keep-dimensions option on; asserts the output rank equals the input rank and
    # the values are close to the numpy reference with kept dimensions.
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf_np = math_ops.reduce_logsumexp(x_np, keepdims=True)
        self.assertEqual(y_tf_np.shape.rank, x_np.ndim)
        y_np = np.log(np.sum(np.exp(x_np), keepdims=True))
        self.assertAllClose(y_tf_np, y_np)

  def testOverflow(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the assertion on the naive numpy computation is tautological: the test itself
    # raises the expected warning exception inside the block when the naive result is infinite, so
    # it never exercises numpy's own warning behavior; it only proves the naive form overflows.
    # Uses values around 1000, where the naive log of summed exponentials overflows, and asserts
    # the op instead matches the max-shifted stable formula for each float type.
    x = [1000, 1001, 1002, 1003]
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegex(RuntimeWarning,
                                  "overflow encountered in exp"):
        out = np.log(np.sum(np.exp(x_np)))
        if out == np.inf:
          raise RuntimeWarning("overflow encountered in exp")

      with test_util.use_gpu():
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf)
        y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testUnderflow(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: as in the overflow test, the expected warning is raised by the test body itself
    # when the naive result is negative infinity, so that block merely confirms the naive form
    # underflows rather than checking a real warning.
    # Uses values around minus 1000 and asserts the op equals the max-shifted stable formula for
    # each float type.
    x = [-1000, -1001, -1002, -1003]
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegex(RuntimeWarning,
                                  "divide by zero encountered in log"):
        out = np.log(np.sum(np.exp(x_np)))
        if out == -np.inf:
          raise RuntimeWarning("divide by zero encountered in log")

      with test_util.use_gpu():
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf)
        y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testInfinity(self):
    # [FUNCTION DOC]
    # Applies the op to scalar negative infinity and asserts the evaluated result is negative
    # infinity.
    with test_util.use_gpu():
      res = math_ops.reduce_logsumexp(-np.inf)
      self.assertEqual(-np.inf, self.evaluate(res))

  def testRaggedTensor(self):
    # [FUNCTION DOC]
    # Builds a ragged tensor with rows [1,2], empty and [3,4,5] in three float types, reduces it
    # fully, and asserts the result matches the stable max-shifted numpy formula over the flat
    # values.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.double]:
      x_rt = ragged_factory_ops.constant([[1, 2], [], [3, 4, 5]], dtype=dtype)
      x_np = np.array(self.evaluate(x_rt.flat_values))
      with test_util.use_gpu():
        y_rt = math_ops.reduce_logsumexp(x_rt)
        y_np = np.log(np.sum(np.exp(x_np - np.max(x_np)))) + np.max(x_np)
        self.assertAllClose(y_rt, y_np)


@test_util.run_all_in_graph_and_eager_modes
class RoundTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the rounding op, run in graph and eager modes.

  def testRounding(self):
    # [FUNCTION DOC]
    # Rounds values from minus 5 to 5 in steps of 0.25 as float32, float64 and int32 on the
    # preferred device and asserts agreement with numpy rounding (tolerance 1e-2).
    x = np.arange(-5.0, 5.0, .25)
    for dtype in [np.float32, np.double, np.int32]:
      x_np = np.array(x, dtype=dtype)
      with test_util.device(use_gpu=True):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf = math_ops.round(x_tf)
        y_tf_np = self.evaluate(y_tf)
        y_np = np.round(x_np)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class MatMulTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for matrix multiplication: 2-D and batched products across supported dtypes, unsupported
  # type errors, mixed-precision products with a requested output type, and invalid output types.
  # Class attribute SUPPORTED_DTYPES lists the dtypes (half, single, double, int32, int64, bfloat16,
  # complex64, complex128) looped over. Runs with ops-as-functions enabled in eager mode and in
  # both graph and eager modes; supports parameterized tests.
  """Test for matmul."""

  SUPPORTED_DTYPES = [
      dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32,
      dtypes.int64, dtypes.bfloat16, dtypes.complex64, dtypes.complex128
  ]

  def testMatMul2D(self):
    # [FUNCTION DOC]
    # For each supported dtype multiplies a 2 by 3 matrix of 1..6 by a 3 by 2 matrix of 7..12 and
    # asserts the result equals [[58, 64], [139, 154]] within tolerance 1e-2.
    for dtype in self.SUPPORTED_DTYPES:
      a = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
      b = constant_op.constant([7, 8, 9, 10, 11, 12], shape=[3, 2], dtype=dtype)
      c = math_ops.matmul(a, b)
      c_np = constant_op.constant([[58, 64], [139, 154]],
                                  shape=(2, 2),
                                  dtype=dtype)
      self.assertAllClose(c, c_np, atol=1e-2)

  def testBatchMatMul(self):
    # [FUNCTION DOC]
    # For each supported dtype multiplies batches of two 2 by 3 and 3 by 2 matrices (values 1..12
    # and 13..24) and asserts the batch result equals the hand-computed 2 by 2 by 2 values
    # within tolerance 1e-2.
    for dtype in self.SUPPORTED_DTYPES:
      a = constant_op.constant(np.arange(1, 13), shape=[2, 2, 3], dtype=dtype)
      b = constant_op.constant(np.arange(13, 25), shape=[2, 3, 2], dtype=dtype)
      c = math_ops.matmul(a, b)
      c_np = constant_op.constant(
          [[[94, 100], [229, 244]], [[508, 532], [697, 730]]],
          shape=[2, 2, 2],
          dtype=dtype)
      self.assertAllClose(c, c_np, atol=1e-2)

  def testUnsupportedtypeMatmul(self):
    # [FUNCTION DOC]
    # Batched int8 by int8 product without an output type is not supported: asserts a type error or
    # invalid-argument error whose message lists the allowed values.
    a = constant_op.constant(
        np.arange(1, 13), shape=[2, 2, 3], dtype=dtypes.int8)
    b = constant_op.constant(
        np.arange(13, 25), shape=[2, 3, 2], dtype=dtypes.int8)
    with self.assertRaisesRegex((TypeError, errors.InvalidArgumentError),
                                "list of allowed values:"):
      math_ops.matmul(a, b)

  @parameterized.parameters((dtypes.int8, dtypes.int8),
                            (dtypes.int8, dtypes.uint8),
                            (dtypes.uint8, dtypes.int8))
  # TODO(shivaniagrawal): matmul (dtypes.uint8, dtypes.uint8) fails in xla_gpu.
  def testInt8MatMul2D(self, a_dtype, b_dtype):
    # [FUNCTION DOC]
    # Parameterized over pairs of 8-bit signed/unsigned input types (unsigned by unsigned is
    # excluded because it fails on one compiled GPU backend). Multiplies the standard 2 by 3 and
    # 3 by 2 matrices requesting int32 output and asserts the result is [[58, 64], [139, 154]].
    a = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=a_dtype)
    b = constant_op.constant([7, 8, 9, 10, 11, 12], shape=[3, 2], dtype=b_dtype)
    c = math_ops.matmul(a, b, output_type=dtypes.int32)
    c_np = constant_op.constant([[58, 64], [139, 154]],
                                shape=(2, 2),
                                dtype=dtypes.int32)
    self.assertAllClose(c, c_np)

  @parameterized.parameters((dtypes.int8), (dtypes.uint8))
  def testMixPrecMatMul2D(self, b_dtype):
    # [FUNCTION DOC]
    # Parameterized over int8 and uint8 for the right operand. Multiplies a bfloat16 left matrix
    # by an 8-bit integer right matrix with bfloat16 output type and asserts the expected product
    # [[58, 64], [139, 154]] within tolerance 1e-2.
    a = constant_op.constant([1, 2, 3, 4, 5, 6],
                             shape=[2, 3],
                             dtype=dtypes.bfloat16)
    b = constant_op.constant([7, 8, 9, 10, 11, 12], shape=[3, 2], dtype=b_dtype)
    c = math_ops.matmul(a, b, output_type=dtypes.bfloat16)
    c_np = constant_op.constant([[58, 64], [139, 154]],
                                shape=(2, 2),
                                dtype=dtypes.bfloat16)
    self.assertAllClose(c, c_np, atol=1e-2)

  @parameterized.parameters((dtypes.int8, dtypes.int8),
                            (dtypes.int8, dtypes.uint8),
                            (dtypes.uint8, dtypes.int8))
  # TODO(shivaniagrawal): matmul (dtypes.uint8, dtypes.uint8) fails in xla_gpu.
  def testInt8BatchMatmul(self, a_dtype, b_dtype):
    # [FUNCTION DOC]
    # Batched version of the 8-bit integer product test: parameterized over signed/unsigned input
    # pairs (unsigned by unsigned excluded for a GPU backend failure), requests int32 output, and
    # asserts exact equality with the hand-computed batch result.
    a = constant_op.constant(np.arange(1, 13), shape=[2, 2, 3], dtype=a_dtype)
    b = constant_op.constant(np.arange(13, 25), shape=[2, 3, 2], dtype=b_dtype)
    c_np = constant_op.constant(
        [[[94, 100], [229, 244]], [[508, 532], [697, 730]]],
        shape=[2, 2, 2],
        dtype=dtypes.int32)
    c = math_ops.matmul(a, b, output_type=dtypes.int32)
    self.assertAllEqual(c, c_np)

  @parameterized.parameters((dtypes.int8), (dtypes.uint8))
  def testMixPrecBatchMatmul(self, b_dtype):
    # [FUNCTION DOC]
    # Batched mixed-precision product: bfloat16 left operand by int8 or uint8 right operand with
    # bfloat16 output type; asserts closeness to the expected batch result within tolerance 1e-2.
    a = constant_op.constant(
        np.arange(1, 13), shape=[2, 2, 3], dtype=dtypes.bfloat16)
    b = constant_op.constant(np.arange(13, 25), shape=[2, 3, 2], dtype=b_dtype)
    c_np = constant_op.constant(
        [[[94, 100], [229, 244]], [[508, 532], [697, 730]]],
        shape=[2, 2, 2],
        dtype=dtypes.bfloat16)
    c = math_ops.matmul(a, b, output_type=dtypes.bfloat16)
    self.assertAllClose(c, c_np, atol=1e-2)

  def testInvalidOutputTypeMatmul(self):
    # [FUNCTION DOC]
    # For int8 and bfloat16 left operands with an int8 right operand, requests float32 output,
    # which has no kernel. Eager mode expects an invalid-argument error (when the TFRT runtime is
    # enabled, about node definition inputs) or a not-found error about finding a device for the
    # node; graph mode expects an invalid-argument error saying no kernel is registered.
    for dtype in [dtypes.int8, dtypes.bfloat16]:
      a = constant_op.constant(np.arange(1, 13), shape=[2, 2, 3], dtype=dtype)
      b = constant_op.constant(
          np.arange(13, 25), shape=[2, 3, 2], dtype=dtypes.int8)
      if context.executing_eagerly():
        if context.is_tfrt_enabled():
          with self.assertRaisesRegex(errors.InvalidArgumentError,
                                      "NodeDef expected inputs"):
            math_ops.matmul(a, b, output_type=dtypes.float32)
        else:
          with self.assertRaisesRegex(errors.NotFoundError,
                                      "Could not find device for node:"):
            math_ops.matmul(a, b, output_type=dtypes.float32)
      else:
        with self.assertRaisesRegex(errors.InvalidArgumentError,
                                    "No OpKernel was registered to support Op"):
          self.evaluate(math_ops.matmul(a, b, output_type=dtypes.float32))


@test_util.run_all_in_graph_and_eager_modes
class SampledADDMMTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the sampled dense-dense matrix multiply that adds into sparse values at given
  # indices. Class attribute SUPPORTED_DTYPES lists bfloat16, half, single and double. Provides a
  # reference-implementation helper used by the tests. Runs in graph and eager modes.
  """Test for sampled_addmm."""

  SUPPORTED_DTYPES = [
      dtypes.bfloat16,
      dtypes.float16,
      dtypes.float32,
      dtypes.float64,
  ]

  def sampledADDMMRef(
      self,
      indices,
      values,
      dense_shape,
      mat1,
      mat2,
      beta=1.0,
      alpha=1.0,
      output_type=dtypes.float32,
  ):
    # [FUNCTION DOC]
    # HELPER: reference implementation for the sampled matrix-multiply tests.
    # Computes the full dense product of mat1 and mat2 with the given output type, gathers the
    # entries at the given indices (treating leading dimensions as batch dimensions), and returns
    # alpha times the gathered values plus beta times the supplied values. The dense_shape argument
    # is accepted but unused. Defaults: beta and alpha 1.0, output type float32.
    dense = math_ops.matmul(mat1, mat2, output_type=output_type)
    dense_vals = array_ops.gather_nd(dense, indices, batch_dims=dense.ndim - 2)
    return alpha * dense_vals + beta * values

  def testSampledADDMM2D(self):
    # [FUNCTION DOC]
    # For each supported dtype, runs the sampled op on 2 by 3 and 3 by 2 matrices with two sample
    # indices on the diagonal, alpha 0.75 and beta 0.25, and asserts its values output (the middle
    # element of the returned triple) is close (tolerance 1e-2) to the reference helper.
    for dtype in self.SUPPORTED_DTYPES:
      indices = constant_op.constant([[0, 0], [1, 1]])
      values = constant_op.constant([0.5, 0.3], dtype=dtype)
      dense_shape = constant_op.constant([2, 2])
      mat1 = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
      mat2 = constant_op.constant(
          [7, 8, 9, 10, 11, 12], shape=[3, 2], dtype=dtype
      )
      alpha = 0.75
      beta = 0.25

      _, res, _ = math_ops.sampled_addmm(
          indices,
          values,
          dense_shape,
          mat1,
          mat2,
          beta=beta,
          alpha=alpha,
          output_type=dtype,
      )
      ref = self.sampledADDMMRef(
          indices,
          values,
          dense_shape,
          mat1,
          mat2,
          beta=beta,
          alpha=alpha,
          output_type=dtype,
      )
      self.assertAllClose(res, ref, atol=1e-2)

  def testBatchSampledADDMM(self):
    # [FUNCTION DOC]
    # Batched variant with two batches of 2 by 3 and 3 by 2 matrices, per-batch index pairs,
    # alpha 0.4 and beta 0.6; for each supported dtype asserts the values output is close to the
    # reference helper (tolerance 1e-2).
    for dtype in self.SUPPORTED_DTYPES:
      indices = constant_op.constant([[[0, 1], [1, 0]], [[0, 0], [1, 0]]])
      values = constant_op.constant([[3, 5], [2, 7]], dtype=dtype)
      dense_shape = constant_op.constant([2, 2])
      mat1 = constant_op.constant(
          np.arange(1, 13), shape=[2, 2, 3], dtype=dtype
      )
      mat2 = constant_op.constant(
          np.arange(13, 25), shape=[2, 3, 2], dtype=dtype
      )
      alpha = 0.4
      beta = 0.6

      _, res, _ = math_ops.sampled_addmm(
          indices,
          values,
          dense_shape,
          mat1,
          mat2,
          beta=beta,
          alpha=alpha,
          output_type=dtype,
      )
      ref = self.sampledADDMMRef(
          indices,
          values,
          dense_shape,
          mat1,
          mat2,
          beta=beta,
          alpha=alpha,
          output_type=dtype,
      )
      self.assertAllClose(res, ref, atol=1e-2)

  def testInvalidDenseShape(self):
    # [FUNCTION DOC]
    # Passes a dense shape of 1 by 2 that is inconsistent with the 2 by 2 product of the batched
    # operands and asserts a value error saying the shape does not match the output shape.
    for dtype in self.SUPPORTED_DTYPES:
      indices = constant_op.constant([[[0, 1], [1, 0]], [[0, 0], [1, 0]]])
      values = constant_op.constant([[3, 5], [2, 7]], dtype=dtype)
      dense_shape = constant_op.constant([1, 2])
      mat1 = constant_op.constant(
          np.arange(1, 13), shape=[2, 2, 3], dtype=dtype
      )
      mat2 = constant_op.constant(
          np.arange(13, 25), shape=[2, 3, 2], dtype=dtype
      )

      with self.assertRaisesRegex(ValueError, "does not match output shape"):
        math_ops.sampled_addmm(
            indices, values, dense_shape, mat1, mat2, output_type=dtype
        )


@test_util.run_all_in_graph_and_eager_modes
class ModTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the modulo op on floating point and integer data with scalar and vector divisors,
  # run in graph and eager modes.

  def testFloat(self):
    # [FUNCTION DOC]
    # For float32 and float64 data 0.5, 0.7, 0.3 with a scalar divisor and a vector divisor,
    # asserts the op is close (tolerance 1e-2) to numpy's C-style remainder.
    x = [0.5, 0.7, 0.3]
    for dtype in [np.float32, np.double]:
      # Test scalar and vector versions.
      for denom in [x[0], [x[0]] * 3]:
        x_np = np.array(x, dtype=dtype)
        with test_util.use_gpu():
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = self.evaluate(y_tf)
          y_np = np.fmod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)

  def testFixed(self):
    # [FUNCTION DOC]
    # For int32 and int64 data 5, 10, 23 with a scalar divisor and a vector divisor, asserts the
    # op equals numpy's floor-style modulo.
    x = [5, 10, 23]
    for dtype in [np.int32, np.int64]:
      # Test scalar and vector versions.
      for denom in [x[0], x]:
        x_np = np.array(x, dtype=dtype)
        with test_util.use_gpu():
          x_tf = constant_op.constant(x_np, shape=x_np.shape)
          y_tf = math_ops.mod(x_tf, denom)
          y_tf_np = self.evaluate(y_tf)
          y_np = np.mod(x_np, denom)
        self.assertAllClose(y_tf_np, y_np)


@test_util.run_all_in_graph_and_eager_modes
class SquaredDifferenceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the squared difference op on real and complex inputs with broadcasting, run in graph
  # and eager modes.

  def testSquaredDifference(self):
    # [FUNCTION DOC]
    # Broadcasts a 2 by 3 array against a length-3 vector for half, single, double, bfloat16,
    # int32 and int64, and asserts the op equals the explicit square of the difference.
    for dtype in [
        np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype,
        np.int32, np.int64
    ]:
      x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
      y = np.array([-3, -2, -1], dtype=dtype)
      z = (x - y) * (x - y)
      with test_util.device(use_gpu=True):
        z_tf = self.evaluate(math_ops.squared_difference(x, y))
        self.assertAllClose(z, z_tf)

  def testComplexSquaredDifference(self):
    # [FUNCTION DOC]
    # For complex64 and complex128 inputs (run on CPU), asserts the op equals the conjugate of the
    # difference times the difference, i.e. a real-valued squared magnitude.
    for dtype in [np.complex64, np.complex128]:
      x = np.array([[1 + 3j, 2 + 2j, 3 + 1j], [4 - 1j, 5 - 2j, 6 - 3j]],
                   dtype=dtype)
      y = np.array([-3 + 1j, -2 + 2j, -1 + 3j], dtype=dtype)
      z = np.conj(x - y) * (x - y)
      with test_util.device(use_gpu=False):
        z_tf = self.evaluate(math_ops.squared_difference(x, y))
        self.assertAllClose(z, z_tf)


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class ApproximateEqualTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the approximate equality op, including shape mismatch errors in normal and
  # compiled (XLA) execution. Runs with ops-as-functions in eager mode and in graph and eager
  # modes.

  def testApproximateEqual(self):
    # [FUNCTION DOC]
    # Checks three cases for float32 and float64: values differing by 9e-5 are not equal under the
    # default tolerance of 1e-5; values differing by 9e-6 are equal; and an elementwise array
    # comparison with explicit tolerance 1e-4 gives the expected boolean pattern.
    for dtype in [np.float32, np.double]:
      x = dtype(1)
      y = dtype(1.00009)
      z = False
      with test_util.device(use_gpu=True):
        # Default tolerance is 0.00001
        z_tf = self.evaluate(math_ops.approximate_equal(x, y))
        self.assertAllEqual(z, z_tf)

    for dtype in [np.float32, np.double]:
      x = dtype(1)
      y = dtype(1.000009)
      z = True
      with test_util.device(use_gpu=True):
        # Default tolerance is 0.00001
        z_tf = self.evaluate(math_ops.approximate_equal(x, y))
        self.assertAllEqual(z, z_tf)

    for dtype in [np.float32, np.double]:
      x = np.array([[[[-1, 2.00009999], [-3, 4.01]]]], dtype=dtype)
      y = np.array([[[[-1.001, 2], [-3.00009, 4]]]], dtype=dtype)
      z = np.array([[[[False, True], [True, False]]]], dtype=np.bool_)
      with test_util.device(use_gpu=True):
        z_tf = self.evaluate(math_ops.approximate_equal(x, y, tolerance=0.0001))
        self.assertAllEqual(z, z_tf)

  def testApproximateEqualShape(self):
    # [FUNCTION DOC]
    # Passes inputs of different rank (length-2 vector vs 1 by 2 matrix) and asserts a value error
    # or invalid-argument error about equal rank or identical shapes.
    for dtype in [np.float32, np.double]:
      x = np.array([1, 2], dtype=dtype)
      y = np.array([[1, 2]], dtype=dtype)
      # The inputs 'x' and 'y' must have the same shape.
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError),
          "Shapes must be equal rank|must be of the same shape"):
        math_ops.approximate_equal(x, y)

  def testApproximateEqualShapeXla(self):
    # [FUNCTION DOC]
    # Same mismatched-shape scenario as the plain shape test, but the op is wrapped in a function
    # compiled with the XLA JIT; asserts the same value error or invalid-argument error.

    @def_function.function(jit_compile=True)
    def approximate_equal(x, y):
      # [FUNCTION DOC]
      # HELPER: JIT-compiled wrapper around the approximate equality op for the shape test.
      # Takes two tensors and returns the boolean elementwise comparison.
      return math_ops.approximate_equal(x, y)

    for dtype in [np.float32, np.double]:
      x = np.array([1, 2], dtype=dtype)
      y = np.array([[1, 2]], dtype=dtype)
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError),
          "Shapes must be equal rank|must be of the same shape"):
        approximate_equal(x, y)


@test_util.run_all_in_graph_and_eager_modes
class ScalarMulTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that the scalar multiply op accepts variables, constants, tensors and indexed slices.
  # Runs in graph and eager modes.

  def testAcceptsRefs(self):
    # [FUNCTION DOC]
    # Creates a variable of value 10 (resource variable in eager mode, regular variable in graph
    # mode), multiplies it by 3 after initialization and asserts the result is 30.
    if context.executing_eagerly():
      var = resource_variable_ops.ResourceVariable(10, name="var")
    else:
      var = variables.Variable(10)
    result = math_ops.scalar_mul(3, var)
    init = variables.global_variables_initializer()
    with test_util.device(use_gpu=True):
      self.evaluate(init)
      self.assertEqual(30, self.evaluate(result))

  def testAcceptsConstant(self):
    # [FUNCTION DOC]
    # Multiplies a constant 10 by scalar 3 and asserts the result is 30.
    const = constant_op.constant(10)
    result = math_ops.scalar_mul(3, const)
    with test_util.device(use_gpu=True):
      self.assertEqual(30, self.evaluate(result))

  def testAcceptsTensor(self):
    # [FUNCTION DOC]
    # Multiplies a 10 by 10 tensor of ones by scalar 3 and asserts it equals ones times 3.
    tensor = array_ops.ones([10, 10])
    result = math_ops.scalar_mul(3, tensor)
    expected = array_ops.ones([10, 10]) * 3

    with test_util.device(use_gpu=True):
      self.assertAllEqual(self.evaluate(expected), self.evaluate(result))

  def testAcceptsIndexedSlices(self):
    # [FUNCTION DOC]
    # Multiplies an indexed-slices value (three rows at indices 0, 2, 5) by scalar minus 3 and
    # asserts the values are scaled and the indices are unchanged.
    values = constant_op.constant([2, 3, 5, 7, 0, -1], shape=[3, 2])
    indices = constant_op.constant([0, 2, 5])
    x = math_ops.scalar_mul(-3, indexed_slices.IndexedSlices(values, indices))
    with test_util.device(use_gpu=True):
      self.assertAllEqual(
          self.evaluate(x.values), [[-6, -9], [-15, -21], [0, 3]])
      self.assertAllEqual(self.evaluate(x.indices), [0, 2, 5])


@test_util.run_all_in_graph_and_eager_modes
class AddNTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the n-ary add op: buffer forwarding, float and integer inputs, gradients, indexed
  # slices and iterable inputs. Runs in graph and eager modes.

  def testPartials(self):
    # [FUNCTION DOC]
    # Regression test for a buffer-forwarding bug: builds 98 single-input sums of the constant 1
    # plus one two-input sum, adds all of them plus a zero constant, and asserts the total is 100.
    """Test that previously revealed a bug in buffer forwarding for AddN."""
    partials = []
    for _ in range(98):
      partials.append(math_ops.add_n([constant_op.constant(1)]))
    partials.append(
        math_ops.add_n([constant_op.constant(1),
                        constant_op.constant(1)]))

    res = math_ops.add_n(partials) + constant_op.constant(0)
    with test_util.use_gpu():
      self.assertAllEqual(res, 100)

  def testFloat(self):
    # [FUNCTION DOC]
    # With a fixed random seed, adds 1 to 9 random 5-dimensional float arrays and asserts the op
    # matches the python sum; also adds the same tensor repeatedly and asserts it equals the
    # multiple.
    np.random.seed(12345)
    for num_inputs in range(1, 10):
      x = [np.random.random((1, 2, 3, 4, 5)) - 0.5 for _ in range(num_inputs)]
      tf_x = ops.convert_n_to_tensor(x)
      with test_util.use_gpu():
        self.assertAllClose(sum(x), math_ops.add_n(tf_x))
        self.assertAllClose(x[0] * num_inputs,
                            math_ops.add_n([tf_x[0]] * num_inputs))

  def testInt(self):
    # [FUNCTION DOC]
    # Same as the float test but with random integer arrays in the signed 8-bit range (fixed
    # seed); asserts exact equality with the python sum and with repeated addition.
    np.random.seed(54321)
    for num_inputs in range(1, 10):
      x = [
          np.random.randint(-128, 128, (5, 4, 3, 2, 1))
          for _ in range(num_inputs)
      ]
      tf_x = ops.convert_n_to_tensor(x)
      with test_util.use_gpu():
        self.assertAllEqual(sum(x), math_ops.add_n(tf_x))
        self.assertAllEqual(x[0] * num_inputs,
                            math_ops.add_n([tf_x[0]] * num_inputs))

  def testGrad(self):
    # [FUNCTION DOC]
    # Creates 1 to 9 random scalar variables, initializes them, adds them and computes the
    # gradient (gradient tape in eager mode, graph gradients otherwise); asserts every gradient is
    # exactly 1.
    np.random.seed(42)
    for num_inputs in range(1, 10):
      with test_util.use_gpu():
        input_vars = [
            variables.Variable(10.0 * np.random.random())
            for _ in range(0, num_inputs)
        ]
        self.evaluate(variables.global_variables_initializer())
        if context.executing_eagerly():
          with backprop.GradientTape() as tape:
            tape.watch(input_vars)
            addn = math_ops.add_n(input_vars)
            add_n_grad = tape.gradient(addn, input_vars)
        else:
          addn = math_ops.add_n(input_vars)
          add_n_grad = gradients.gradients(addn, input_vars)

        self.assertAllEqual(
            np.repeat(1.0, num_inputs),  # d/dx (x + y + ...) = 1
            [self.evaluate(g) for g in add_n_grad])

  def testIndexedSlices(self):
    # [FUNCTION DOC]
    # Adds one and two copies of an indexed-slices value and asserts the results equal the dense
    # form and twice the dense form, since the op always converts slices to dense.
    slc = indexed_slices.IndexedSlices(
        array_ops.constant([1, 2], shape=[1, 2]), array_ops.constant([1]),
        array_ops.constant([2, 2]))
    slc_as_dense = np.array([[0, 0], [1, 2]])
    with test_util.use_gpu():
      # add_n currently always converts IndexedSlices to dense
      self.assertAllEqual(slc_as_dense, math_ops.add_n([slc]))
      self.assertAllEqual(2 * slc_as_dense, math_ops.add_n([slc, slc]))

  def test_iterable(self):
    # [FUNCTION DOC]
    # Asserts the op accepts a generator yielding 1 and 2, and the values view of a mapping with
    # values 1 and 2, each producing 3.
    """Test that add_n supports iterables (e.g. generators and dict values)."""

    def fn():
      # [FUNCTION DOC]
      # HELPER: generator used as an iterable input; yields the integers 1 and then 2.
      yield 1
      yield 2

    values_dict = {"a": 1, "b": 2}
    with test_util.use_gpu():
      self.assertAllEqual(3, math_ops.add_n(fn()))
      self.assertAllEqual(3, math_ops.add_n(values_dict.values()))


@test_util.run_all_in_graph_and_eager_modes
class DivAndModTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the division and modulo families: floor and truncate division and remainder on
  # integers and floats, true division type promotion, real division, gradients, consistency of
  # the quotient/remainder identity, python value inputs, and integer edge cases. Provides
  # helper methods for test data and for numpy reference results that avoid numpy's integer
  # overflow quirks. Runs in graph and eager modes.
  # TODO(aselle): Test more types before exposing new division operators.

  def intTestData(self):
    # [FUNCTION DOC]
    # HELPER: returns a pair of integer arrays: numerators of minus 10 to 9 as a column of 20, and
    # divisors minus 3, minus 1, 1, 3 as a row of 4, so that combining them broadcasts to a grid.
    nums = np.arange(-10, 10, 1).reshape(20, 1)
    divs = np.arange(-3, 4, 2).reshape(1, 4)
    return nums, divs

  def floatTestData(self):
    # [FUNCTION DOC]
    # HELPER: returns a pair of float arrays: numerators from minus 10 up to (excluding) 10 in
    # steps of 0.25 as a column of 80, and divisors from minus 3 up to (excluding) 0 in steps of
    # 0.25 as a row of 12 (all negative, never zero).
    nums = np.arange(-10, 10, .25).reshape(80, 1)
    divs = np.arange(-3, 0, .25).reshape(1, 12)
    return nums, divs

  def numpySafeFloorDivInt(self, x, y):
    # [FUNCTION DOC]
    # HELPER: numpy reference for integer floor division. Computes floor division of the integer
    # arrays and then fixes the overflow case where the numerator is the minimum value of its type
    # and the divisor is minus one: numpy gives 0 but the expected result (matching the op) is the
    # minimum value itself, so that quotient times divisor plus remainder still equals the
    # numerator. Modifies the freshly computed result in place and returns it.
    z = x // y
    # Numpy produces 0 for INT_MIN/-1, but we expect an overflow to INT_MIN
    # so that (INT_MIN/-1) + (INT_MIN % -1) = INT_MIN + 0 = INT_MIN.
    z[(x == np.iinfo(x.dtype).min) & (y == -1)] = np.iinfo(x.dtype).min
    return z

  def numpySafeFloorModInt(self, x, y):
    # [FUNCTION DOC]
    # HELPER: numpy reference for integer floor remainder, computed as numerator minus the safe
    # floor quotient times the divisor, because numpy's own remainder crashes with a floating point
    # exception for the minimum value modulo minus one. Returns the remainder array.
    # Numpy crashes with a FPE for INT_MIN % -1.
    z = self.numpySafeFloorDivInt(x, y)
    return x - z * y

  def numpySafeTruncateDivInt(self, x, y):
    # [FUNCTION DOC]
    # HELPER: numpy reference for integer division truncating toward zero. Starts from the safe
    # floor quotient and adds one wherever the remainder is non-zero and the operands have opposite
    # signs (floor rounded away from zero there). Modifies the array in place and returns it.
    z = self.numpySafeFloorDivInt(x, y)
    # Round up if non-zero remainder and inputs have opposite signs.
    z[(x != z * y) & ((x < 0) != (y < 0))] += 1
    return z

  def numpySafeTruncateModInt(self, x, y):
    # [FUNCTION DOC]
    # HELPER: numpy reference for the integer remainder matching truncating division, computed as
    # numerator minus the safe truncating quotient times the divisor (avoids numpy's crash for the
    # minimum value modulo minus one). Returns the remainder array.
    # Numpy crashes with a FPE for INT_MIN % -1.
    z = self.numpySafeTruncateDivInt(x, y)
    return x - z * y

  def testFloorModInt(self):
    # [FUNCTION DOC]
    # For int32 and int64 grid data, asserts the floor-modulo op equals the safe numpy reference
    # and that the python percent operator on tensors gives the same result.
    nums, divs = self.intTestData()
    for dtype in [np.int32, np.int64]:
      x = nums.astype(dtype)
      y = divs.astype(dtype)
      tf_result = math_ops.floormod(x, y)
      np_result = self.numpySafeFloorModInt(x, y)
      self.assertAllEqual(tf_result, np_result)
      tf2_result = (array_ops.constant(x) % array_ops.constant(y))
      self.assertAllEqual(tf2_result, tf_result)

  def testFloorModFloat(self):
    # [FUNCTION DOC]
    # For float16, float32 and float64 grid data, asserts the floor-modulo op exactly equals numpy's
    # remainder and the tensor percent operator.
    nums, divs = self.floatTestData()
    for dtype in [np.float16, np.float32, np.float64]:
      x = nums.astype(dtype)
      y = divs.astype(dtype)
      tf_result = math_ops.floormod(x, y)
      np_result = x % y
      self.assertAllEqual(tf_result, np_result)
      tf2_result = (array_ops.constant(x) % array_ops.constant(y))
      self.assertAllEqual(tf2_result, tf_result)

  def testFloorModBfloat16(self):
    # [FUNCTION DOC]
    # Casts float grid data to bfloat16, applies floor-modulo and asserts exact equality with the
    # numpy remainder of the original float data.
    nums, divs = self.floatTestData()
    tf_result = math_ops.floormod(
        math_ops.cast(nums, dtypes.bfloat16),
        math_ops.cast(divs, dtypes.bfloat16))
    np_result = nums % divs
    self.assertAllEqual(tf_result, np_result)

  def testTruncateModInt(self):
    # [FUNCTION DOC]
    # Asserts truncate-modulo on integer grid data equals numpy's C-style remainder.
    nums, divs = self.intTestData()
    tf_result = math_ops.truncatemod(nums, divs)
    np_result = np.fmod(nums, divs)
    self.assertAllEqual(tf_result, np_result)

  def testTruncateModFloat(self):
    # [FUNCTION DOC]
    # Asserts truncate-modulo on float grid data equals numpy's C-style remainder.
    nums, divs = self.floatTestData()
    tf_result = math_ops.truncatemod(nums, divs)
    np_result = np.fmod(nums, divs)
    self.assertAllEqual(tf_result, np_result)

  def testFloorDivideInt(self):
    # [FUNCTION DOC]
    # Asserts floor division on integer grid data equals the safe numpy reference and the tensor
    # double-slash operator.
    nums, divs = self.intTestData()
    tf_result = math_ops.floor_div(nums, divs)
    np_result = self.numpySafeFloorDivInt(nums, divs)
    self.assertAllEqual(tf_result, np_result)
    tf2_result = (array_ops.constant(nums) // array_ops.constant(divs))
    self.assertAllEqual(tf2_result, tf_result)

  def testTruncateDivideInt(self):
    # [FUNCTION DOC]
    # Asserts truncating division on integer grid data equals the safe numpy truncating reference.
    nums, divs = self.intTestData()
    tf_result = math_ops.truncatediv(nums, divs)
    np_result = self.numpySafeTruncateDivInt(nums, divs)
    self.assertAllEqual(tf_result, np_result)

  def testTruncateDivideFloat(self):
    # [FUNCTION DOC]
    # Asserts truncating division on float grid data equals numpy's truncation of the true quotient.
    nums, divs = self.floatTestData()
    tf_result = math_ops.truncatediv(nums, divs)
    np_result = np.trunc(nums / divs)
    self.assertAllEqual(tf_result, np_result)

  @test_util.deprecated_graph_mode_only
  def testDivideName(self):
    # [FUNCTION DOC]
    # Graph mode only. Divides constants 3 and 4 with an explicit op name and asserts the output
    # tensor name is that name with output index zero appended.
    op = math_ops.divide(
        array_ops.constant(3), array_ops.constant(4), name="my_cool_divide")
    self.assertEqual(op.name, "my_cool_divide:0")

  def testRealDiv(self):
    # [FUNCTION DOC]
    # Asserts real division on float grid data is close to numpy's division.
    nums, divs = self.floatTestData()
    tf_result = math_ops.realdiv(nums, divs)
    np_result = np.divide(nums, divs)
    self.assertAllClose(tf_result, np_result)

  def testDivideType(self):
    # [FUNCTION DOC]
    # Divides an int32 tensor by python integers 1 and 4 and asserts the results are float64 with
    # values 2.0 and 0.5 (true division promotes to double).
    a = array_ops.constant([2], dtype=dtypes.int32)
    # Since __future__.division is effect, we should always upgrade to float64
    b = math_ops.divide(a, 1)
    self.assertEqual(b.dtype, dtypes.float64)
    self.assertEqual(2.0, self.evaluate(b))
    c = math_ops.divide(a, 4)
    self.assertEqual(c.dtype, dtypes.float64)
    self.assertEqual(0.5, self.evaluate(c))

  def testComplexDiv(self):
    # [FUNCTION DOC]
    # Smoke test: divides a complex tensor by python floats using both the divide and the legacy
    # div function; passes if no error is raised (nothing is asserted about values).
    foo = array_ops.constant([1. + 3.j])
    _ = math_ops.divide(foo, 1.)
    _ = math_ops.div(foo, 2.)

  def testFloorDivGrad(self):
    # [FUNCTION DOC]
    # With variables 2 and 4, computes gradients of divide, legacy div and floor-division with
    # respect to both (gradient tape in eager mode, graph gradients otherwise). Asserts the first
    # two give 0.25 and minus 0.125 and that floor division gives no gradient (null) for both.
    a = variables.Variable(2.)
    b = variables.Variable(4.)
    input_vars = [a, b]
    self.evaluate(variables.global_variables_initializer())
    if context.executing_eagerly():
      # TDOO(rmlarsen): Is there a more compact way of
      # writing this for multiple expressions?
      with backprop.GradientTape() as tape:
        tape.watch(input_vars)
        c_grad0 = tape.gradient(math_ops.divide(a, b), input_vars)
      with backprop.GradientTape() as tape:
        tape.watch(input_vars)
        c_grad1 = tape.gradient(math_ops.div(a, b), input_vars)
      with backprop.GradientTape() as tape:
        tape.watch(input_vars)
        c_grad2 = tape.gradient(math_ops.floordiv(a, b), input_vars)
    else:
      c_grad0 = gradients.gradients(math_ops.divide(a, b), input_vars)
      c_grad1 = gradients.gradients(math_ops.div(a, b), input_vars)
      c_grad2 = gradients.gradients(math_ops.floordiv(a, b), input_vars)
    self.assertAllEqual([self.evaluate(x) for x in c_grad0], [.25, -.125])
    self.assertAllEqual([self.evaluate(x) for x in c_grad1], [.25, -.125])
    self.assertAllEqual(
        [None if x is None else self.evaluate(x) for x in c_grad2],
        [None, None])

  def testConsistent(self):
    # [FUNCTION DOC]
    # Checks the identity quotient times divisor plus remainder equals the numerator: for floor
    # forms (op functions and operators, and numpy) and for truncation forms, asserting each
    # equals the numerator broadcast to the full grid.
    nums, divs = self.intTestData()
    tf_result = (
        math_ops.floor_div(nums, divs) * divs + math_ops.floormod(nums, divs))
    tf_nums = array_ops.constant(nums)
    tf_divs = array_ops.constant(divs)
    tf2_result = (tf_nums // tf_divs * tf_divs + tf_nums % tf_divs)
    np_result = (nums // divs) * divs + (nums % divs)
    # Consistent with numpy
    self.assertAllEqual(tf_result, np_result)
    # Consistent with two forms of divide
    self.assertAllEqual(tf_result, tf2_result)
    # consistency for truncation form
    tf3_result = (
        math_ops.truncatediv(nums, divs) * divs +
        math_ops.truncatemod(nums, divs))
    expanded_nums = np.reshape(
        np.tile(nums, divs.shape[1]), (nums.shape[0], divs.shape[1]))
    # Consistent with desire to get numerator
    self.assertAllEqual(tf3_result, expanded_nums)
    # Consistent with desire to get numerator
    self.assertAllEqual(tf_result, expanded_nums)

  def testWithPythonValue(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue: dividing python numbers (and a python number by a
    # tensor) must return a tensor object.
    # Test case for https://github.com/tensorflow/tensorflow/issues/39475
    x = math_ops.divide(5, 2)
    self.assertIsInstance(x, tensor_lib.Tensor)
    x = math_ops.divide(5, array_ops.constant(2.0))
    self.assertIsInstance(x, tensor_lib.Tensor)

  def intEdgeTestData(self, dtype):
    # [FUNCTION DOC]
    # HELPER: builds 4 by 4 integer numerator and divisor arrays of the given numpy integer type
    # filled with the type's extreme values, minus one, and one, avoiding the
    # minimum-divided-by-minus-one overflow (uses minimum plus one there). Returns the pair.
    """Edge-case test data for integer types."""
    # INT_MIN/-1 will produce signed-integer overflow, so we instead test
    # (INT_MIN + 1) / -1.
    nums = np.array(
        [
            [np.iinfo(dtype).min, -1, 1, np.iinfo(dtype).max],
            [np.iinfo(dtype).min + 1, -1, 1, np.iinfo(dtype).max],
            [np.iinfo(dtype).min, -1, 1, np.iinfo(dtype).max],
            [np.iinfo(dtype).min, -1, 1, np.iinfo(dtype).max],
        ],
        dtype=dtype,
    )
    divs = np.array(
        [
            [
                np.iinfo(dtype).min,
                np.iinfo(dtype).min,
                np.iinfo(dtype).min,
                np.iinfo(dtype).min,
            ],
            [-1, -1, -1, -1],
            [1, 1, 1, 1],
            [
                np.iinfo(dtype).max,
                np.iinfo(dtype).max,
                np.iinfo(dtype).max,
                np.iinfo(dtype).max,
            ],
        ],
        dtype=dtype,
    )
    return nums, divs

  def testFloorDivModIntEdges(self):
    # [FUNCTION DOC]
    # For int32 and int64 edge-case grids, asserts the floor-division and floor-modulo ops equal the
    # safe numpy references.
    for dtype in [np.int32, np.int64]:
      x, y = self.intEdgeTestData(dtype)
      tf_floor_div = math_ops.floor_div(x, y)
      np_floor_div = self.numpySafeFloorDivInt(x, y)
      self.assertAllEqual(tf_floor_div, np_floor_div)
      tf_floor_mod = math_ops.floormod(x, y)
      np_floor_mod = self.numpySafeFloorModInt(x, y)
      self.assertAllEqual(tf_floor_mod, np_floor_mod)

  def testTruncateDivModIntEdges(self):
    # [FUNCTION DOC]
    # For int32 and int64 edge-case grids, asserts the truncating division and modulo ops equal the
    # safe numpy references.
    for dtype in [np.int32, np.int64]:
      x, y = self.intEdgeTestData(dtype)
      tf_truncate_div = math_ops.truncatediv(x, y)
      np_truncate_div = self.numpySafeTruncateDivInt(x, y)
      self.assertAllEqual(tf_truncate_div, np_truncate_div)
      tf_truncate_mod = math_ops.truncatemod(x, y)
      np_truncate_mod = self.numpySafeTruncateModInt(x, y)
      self.assertAllEqual(tf_truncate_mod, np_truncate_mod)


@test_util.run_all_in_graph_and_eager_modes
class DivNoNanTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for division that returns zero when the divisor is zero. Class attribute
  # _SUPPORTED_DTYPES lists integer, bfloat16, float and complex types used for parameterization.
  # Runs in graph and eager modes.

  _SUPPORTED_DTYPES = [dtypes.int8, dtypes.uint8,
                       dtypes.int16, dtypes.uint16,
                       dtypes.int32, dtypes.uint32,
                       dtypes.int64, dtypes.uint64,
                       dtypes.bfloat16, dtypes.float16,
                       dtypes.float32, dtypes.float64,
                       dtypes.complex64, dtypes.complex128]

  @parameterized.parameters(*_SUPPORTED_DTYPES)
  def testBasic(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over every supported dtype. Builds numerator and divisor grids suited to the
    # dtype (non-negative for unsigned, wider integer range, or fractions), always including a zero
    # divisor. Expected values are true division of the evaluated tensors with entries for zero
    # divisors set to zero; asserts the op matches within a type-dependent tolerance.
    if dtype.is_unsigned:
      nums = np.arange(0, 120, 3).reshape(40, 1)
      divs = np.arange(0, 48, 4).reshape(1, 12)
    elif dtype.is_integer:
      nums = np.arange(-120, 120, 3).reshape(80, 1)
      divs = np.arange(-48, 48, 4).reshape(1, 24)
    else:
      nums = np.arange(-10, 10, .25).reshape(80, 1)
      divs = np.arange(-3, 3, .25).reshape(1, 24)
    assert 0 in divs, "Bad test set-up"

    tf_nums = constant_op.constant(nums, dtype=dtype)
    tf_divs = constant_op.constant(divs, dtype=dtype)

    # Use tf versions for expected value to ensure inputs are identical
    # (e.g. in the case of bfloat16).
    np_nums = self.evaluate(tf_nums)
    np_divs = self.evaluate(tf_divs)
    np_result = np.true_divide(np_nums, np_divs)
    np_result[:, np_divs[0] == 0] = 0

    with test_util.use_gpu():
      tf_result = math_ops.div_no_nan(tf_nums, tf_divs)
      self.assertAllCloseAccordingToType(tf_result, np_result)

  @parameterized.product(
      type_x=_SUPPORTED_DTYPES + [float, int],
      type_y=_SUPPORTED_DTYPES + [float, int])
  def testSameSupportedTypesAsDivide(self, type_x, type_y):
    # [FUNCTION DOC]
    # Parameterized over the product of dtypes (plus python float and int) for both operands. Divides
    # one by one with the regular divide function. If that raises a type error, asserts the
    # no-nan version raises a type error too; otherwise asserts it returns the same dtype and
    # value. Verifies both accept exactly the same type combinations.
    def one(type_):
      # [FUNCTION DOC]
      # HELPER: returns the value one in the requested kind: python integer for the integer type,
      # python float for the float type, otherwise a constant tensor of that dtype.
      if type_ is int:
        return 1
      elif type_ is float:
        return 1.0
      else:
        return constant_op.constant(1, dtype=type_)

    x = one(type_x)
    y = one(type_y)

    divide_raises = False
    try:
      divide_result = math_ops.divide(x, y)
    except TypeError:
      divide_raises = True

    if divide_raises:
      with self.assertRaises(TypeError):
        _ = math_ops.div_no_nan(x, y)
    else:
      divide_no_nan_result = math_ops.div_no_nan(x, y)
      self.assertEqual(divide_no_nan_result.dtype, divide_result.dtype)
      self.assertAllEqual(divide_no_nan_result, divide_result)

  @parameterized.parameters((dtypes.bfloat16), (dtypes.float16),
                            (dtypes.float32), (dtypes.float64),
                            (dtypes.complex64), (dtypes.complex128))
  def testSmall(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over bfloat16, half, single, double and complex types. Divides zeros by tiny
    # divisors whose squared magnitude underflows and asserts the result is exactly zero (no
    # spurious not-a-number from complex division).
    # Choose values whose squared magnitude underflows to zero/subnormal.
    zero = constant_op.constant([0, 0, 0, 0], dtype=dtype)
    divs = constant_op.constant([1e-25, -1e-20, 1e-165, -1e-160], dtype=dtype)
    tf_result = math_ops.div_no_nan(zero, divs)

    # Results should always be exactly zero.
    self.assertAllEqual(tf_result, zero)

  @parameterized.parameters((dtypes.bfloat16), (dtypes.float16),
                            (dtypes.float32), (dtypes.float64),
                            (dtypes.complex64), (dtypes.complex128))
  def testNonFiniteInNumerator(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over bfloat16, half, single, double and complex types. With numerators
    # not-a-number, plus and minus infinity: dividing by zero must give zeros; dividing by one must
    # equal ordinary division (non-finite values preserved).
    nums = constant_op.constant([np.nan, np.inf, -np.inf], dtype=dtype)
    zeros = constant_op.constant([0, 0, 0], dtype=dtype)
    ones = constant_op.constant([1, 1, 1], dtype=dtype)
    with test_util.use_gpu():
      tf_result_zeros = math_ops.div_no_nan(nums, zeros)
      self.assertAllEqual([0, 0, 0], tf_result_zeros)
      tf_result_ones = math_ops.div_no_nan(nums, ones)
      self.assertAllEqual(nums / ones, tf_result_ones)


@test_util.run_all_in_graph_and_eager_modes
class MultiplyNoNanTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for multiplication that returns zero when the second operand is zero, even if the first
  # is not finite. Runs in graph and eager modes.

  def testBasic(self):
    # [FUNCTION DOC]
    # For float32 and float64 with first operand zero, one, not-a-number and both infinities:
    # multiplying by zeros gives zeros; by ones gives the operand unchanged; with arguments
    # reversed (non-finite values in the second argument) normal arithmetic applies.
    for dtype in [np.float32, np.float64]:
      values = [0, 1, np.nan, np.inf, -np.inf]
      x = constant_op.constant(values, dtype=dtype)
      zeros = constant_op.constant(np.zeros((5,)), dtype=dtype)
      ones = constant_op.constant(np.ones((5,)), dtype=dtype)
      with test_util.use_gpu():
        tf_result_zeros = math_ops.multiply_no_nan(x, zeros)
        self.assertAllEqual(tf_result_zeros, zeros)
        tf_result_ones = math_ops.multiply_no_nan(x, ones)
        self.assertAllEqual(tf_result_ones, x)
        # Normal floating point arithmetic if nonfinite values are in the
        # second argument.
        tf_result_reverseargs = math_ops.multiply_no_nan(zeros, x)
        self.assertAllEqual(zeros * x, tf_result_reverseargs)


@test_util.run_all_in_graph_and_eager_modes
class XlogyTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the x-times-log-y op, which returns zero when x is zero. Runs in graph and eager
  # modes over half, single and double precision.

  def testXlogyNoZero(self):
    # [FUNCTION DOC]
    # With no zero in x, asserts the op is close to the plain product of x and the natural log of y.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.1, 0.2, 3.5], [-2., -5., 30.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [3.1, 4., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlogy = self.evaluate(math_ops.xlogy(x, y))
        xtimeslogy = self.evaluate(x * math_ops.log(y))
        self.assertAllClose(xlogy, xtimeslogy)

  def testXlogyWithZero(self):
    # [FUNCTION DOC]
    # With x all zeros and y containing a zero, asserts the result is all zeros (not not-a-number).
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(np.zeros((2, 3)), dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlogy_tf_np = self.evaluate(math_ops.xlogy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y))
        self.assertAllClose(xlogy_tf_np, zeros_np)

  def testXlogyWithZeroBroadcast(self):
    # [FUNCTION DOC]
    # Broadcasts a column x of 0 and 1 against a 2 by 3 y whose first row contains a zero. Asserts
    # the first output row is zeros and the second row equals the log of the second row of y.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.], [1.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlogy_tf_np = self.evaluate(math_ops.xlogy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y[0]))
        xtimes_logy = self.evaluate(math_ops.log(y[1]))
        self.assertAllClose(zeros_np, xlogy_tf_np[0])
        self.assertAllClose(xtimes_logy, xlogy_tf_np[1])


@test_util.run_all_in_graph_and_eager_modes
class Xlog1pyTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the x-times-log-of-one-plus-y op, which returns zero when x is zero (even if y is
  # minus one). Runs in graph and eager modes over half, single and double precision.

  def testXlog1pyNoNeg1(self):
    # [FUNCTION DOC]
    # With y never equal to minus one, asserts the op is close to x times the log of one plus y.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.1, 0.2, 3.5], [-2., -5., 30.]], dtype=dtype)
      y = constant_op.constant([[-0.1, -0.2, 3.5], [3.1, -0.9, 2.]],
                               dtype=dtype)
      with test_util.use_gpu():
        xlog1py = self.evaluate(math_ops.xlog1py(x, y))
        xtimeslog1py = self.evaluate(x * math_ops.log1p(y))
        self.assertAllClose(xlog1py, xtimeslog1py)

  def testXlog1pyWithNegOne(self):
    # [FUNCTION DOC]
    # With x all zeros and y containing minus one, asserts the output is all zeros.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(np.zeros((2, 3)), dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [-1., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlog1py_tf_np = self.evaluate(math_ops.xlog1py(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y))
        self.assertAllClose(xlog1py_tf_np, zeros_np)

  def testXlog1pyWithZeroBroadcast(self):
    # [FUNCTION DOC]
    # Broadcasts column x of 0 and 1 against y whose first row contains minus one. Asserts the first
    # output row is zeros and the second row equals the log of one plus the second row of y.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.], [1.]], dtype=dtype)
      y = constant_op.constant([[-0.1, -0.2, -1.], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xlog1py_tf_np = self.evaluate(math_ops.xlog1py(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y[0]))
        xtimes_log1py = self.evaluate(math_ops.log1p(y[1]))
        self.assertAllClose(zeros_np, xlog1py_tf_np[0])
        self.assertAllClose(xtimes_log1py, xlog1py_tf_np[1])


@test_util.run_all_in_graph_and_eager_modes
class XdivyTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the x-divided-by-y op, which returns zero when x is zero. Runs in graph and eager
  # modes over half, single and double precision.

  def testXdivyNoZero(self):
    # [FUNCTION DOC]
    # With no zero in x, asserts the op is close to ordinary division of x by y.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.1, 0.2, 3.5], [-2., -5., 30.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [3.1, 4., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xdivy = self.evaluate(math_ops.xdivy(x, y))
        x_over_y = self.evaluate(x / y)
        self.assertAllClose(xdivy, x_over_y)

  def testXdivyWithZero(self):
    # [FUNCTION DOC]
    # With x all zeros and y containing a zero, asserts the output is all zeros (no division error).
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant(np.zeros((2, 3)), dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xdivy_tf_np = self.evaluate(math_ops.xdivy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y))
        self.assertAllClose(xdivy_tf_np, zeros_np)

  def testXdivyWithZeroBroadcast(self):
    # [FUNCTION DOC]
    # Broadcasts column x of 0 and 1 against y whose first row contains a zero. Asserts the first
    # output row is zeros and the second row equals the reciprocal of the second row of y.
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      x = constant_op.constant([[0.], [1.]], dtype=dtype)
      y = constant_op.constant([[0.1, 0.2, 3.5], [0., 1., 2.]], dtype=dtype)
      with test_util.use_gpu():
        xdivy_tf_np = self.evaluate(math_ops.xdivy(x, y))
        zeros_np = self.evaluate(array_ops.zeros_like(y[0]))
        x_over_y = self.evaluate(1 / y[1])
        self.assertAllClose(zeros_np, xdivy_tf_np[0])
        self.assertAllClose(x_over_y, xdivy_tf_np[1])


@test_util.run_all_in_graph_and_eager_modes
class NextAfterTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the next-representable-floating-point-value op in single and double precision,
  # replicating numpy's own checks. Runs in graph and eager modes.

  # Basic NextAfter tests that replicate numpy nextafter tests.
  def testBasic(self):
    # [FUNCTION DOC]
    # For float32 and float64: the next value after one toward two exceeds one by machine epsilon;
    # toward zero it is below one; a not-a-number in either argument gives not-a-number; and
    # next after a value toward itself returns that value.

    for dtype in [dtypes.float32, dtypes.float64]:
      one = constant_op.constant([1], dtype=dtype)
      two = constant_op.constant([2], dtype=dtype)
      zero = constant_op.constant([0], dtype=dtype)
      nan = constant_op.constant([np.nan], dtype=dtype)

      eps = constant_op.constant([np.finfo(dtype.as_numpy_dtype).eps],
                                 dtype=dtype)

      self.assertAllEqual(math_ops.nextafter(one, two) - one, eps)
      self.assertAllLess(math_ops.nextafter(one, zero) - one, 0)
      self.assertAllEqual(math_ops.is_nan(math_ops.nextafter(nan, one)), [True])
      self.assertAllEqual(math_ops.is_nan(math_ops.nextafter(one, nan)), [True])
      self.assertAllEqual(math_ops.nextafter(one, one), one)

  def testBroadcasting(self):
    # [FUNCTION DOC]
    # Broadcasts a two-element vector of ones against a one-element target of two and asserts each
    # output element exceeds one by machine epsilon, for float32 and float64.

    for dtype in [dtypes.float32, dtypes.float64]:
      one = constant_op.constant([1, 1], dtype=dtype)
      two = constant_op.constant([2], dtype=dtype)

      eps = np.finfo(dtype.as_numpy_dtype).eps

      eps_const = constant_op.constant([eps, eps], dtype=dtype)

      self.assertAllEqual(math_ops.nextafter(one, two) - one, eps_const)


@test_util.run_all_in_graph_and_eager_modes
class BinaryOpsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for error behavior and operand dispatching of overloaded binary operators on tensors.
  # Runs in graph and eager modes.

  def testErrorReceivedIfDtypeMismatchFromOp(self):
    # [FUNCTION DOC]
    # Adds an int32 tensor and a python float. Expects an invalid-argument error about the second
    # input being expected as int32 in eager mode, or a type error saying the second input's float32
    # type does not match the first argument's int32 in graph mode.
    if context.executing_eagerly():
      error = errors_impl.InvalidArgumentError
      error_message = (
          r"cannot compute Add(V2)? as input #1\(zero-based\) was expected to "
          r"be a int32 tensor but is a float tensor \[Op:Add(V2)?\]")
    else:
      error = TypeError
      error_message = (
          "Input 'y' of 'Add(V2)?' Op has type float32 that does not "
          "match type int32 of argument 'x'.")
    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + 1.0
      self.evaluate(a)

  def testRHSDispatchingAndErrorRaising(self):
    # [FUNCTION DOC]
    # Checks how adding a tensor to a foreign right-hand object behaves. An object whose reflected
    # addition returns true makes the result true. Objects whose reflected addition raises a type
    # error, returns the not-implemented marker, or does not define addition at all must produce a
    # value error (eager) or type error (graph) about being unable to convert to a tensor.
    if context.executing_eagerly():
      error = ValueError
      error_message = (
          r"Attempt to convert a value .* with an unsupported type")
    else:
      error = TypeError
      error_message = (r"Failed to convert elements of .* to Tensor")

    class RHSReturnsTrue:
      # [CLASS DOC]
      # HELPER: stand-in right-hand operand whose reflected addition always returns true.

      def __radd__(self, other):
        # [FUNCTION DOC]
        # HELPER: reflected addition; ignores the left operand and returns true.
        return True

    a = array_ops.ones([1], dtype=dtypes.int32) + RHSReturnsTrue()
    self.assertEqual(a, True)

    class RHSRaisesError:
      # [CLASS DOC]
      # HELPER: stand-in right-hand operand whose reflected addition raises a type error.

      def __radd__(self, other):
        # [FUNCTION DOC]
        # HELPER: reflected addition; always raises a type error saying the right-hand side is not
        # implemented.
        raise TypeError("RHS not implemented")

    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + RHSRaisesError()
      self.evaluate(a)

    class RHSReturnsNotImplemented:
      # [CLASS DOC]
      # HELPER: stand-in right-hand operand whose reflected addition declines via the
      # not-implemented marker.

      def __radd__(self, other):
        # [FUNCTION DOC]
        # HELPER: reflected addition; returns the not-implemented marker so the tensor conversion
        # error path is taken.
        return NotImplemented

    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + RHSReturnsNotImplemented()
      self.evaluate(a)

    class RHSNotImplemented:
      # [CLASS DOC]
      # HELPER: stand-in right-hand operand with no addition support at all; has no members.
      pass

    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + RHSNotImplemented()
      self.evaluate(a)


class SignTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the sign op's gradient on complex input. Not decorated, so each test runs only in the
  # mode it sets itself.

  def test_complex_sign_gradient(self):
    # [FUNCTION DOC]
    # In eager mode, takes the sign of the complex number one plus one i, records it with a
    # gradient tape and asserts the gradient is approximately 0.353553 minus 0.353553 i.
    with context.eager_mode():
      x = math_ops.complex(1., 1.)
      with backprop.GradientTape() as t:
        t.watch(x)
        y = math_ops.sign(x)
      self.assertAllClose(
          t.gradient(y, x), math_ops.complex(0.353553, -0.353553))


@test_util.run_all_in_graph_and_eager_modes
class ReciprocalNoNanTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the reciprocal op that maps zero to zero. Class attribute allowed_dtypes lists half,
  # single, double, complex64 and complex128. Runs in graph and eager modes.

  allowed_dtypes = [
      dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64,
      dtypes.complex128
  ]

  def testBasic(self):
    # [FUNCTION DOC]
    # For each allowed dtype, applies the op to 1, 2, 0, 4 and asserts the result is 1, 0.5, 0,
    # 0.25 with the same base dtype.
    for dtype in self.allowed_dtypes:
      x = constant_op.constant([1.0, 2.0, 0.0, 4.0], dtype=dtype)

      y = math_ops.reciprocal_no_nan(x)

      target = constant_op.constant([1.0, 0.5, 0.0, 0.25], dtype=dtype)

      self.assertAllEqual(y, target)
      self.assertEqual(y.dtype.base_dtype, target.dtype.base_dtype)

  def testInverse(self):
    # [FUNCTION DOC]
    # For each allowed dtype and a random 5 by 5 by 5 array drawn from 0, 1, 2, 4, 5, applies the
    # op twice and asserts the original values come back (zeros stay zeros) with unchanged dtype.
    for dtype in self.allowed_dtypes:
      x = np.random.choice([0, 1, 2, 4, 5], size=(5, 5, 5))
      x = constant_op.constant(x, dtype=dtype)

      y = math_ops.reciprocal_no_nan(math_ops.reciprocal_no_nan(x))

      self.assertAllClose(y, x)
      self.assertEqual(y.dtype.base_dtype, x.dtype.base_dtype)


class EqualityTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for tensor equality and inequality ops against null values and against python or numpy
  # float literals. Supports parameterized tests; modes are chosen per method.

  @test_util.run_all_in_graph_and_eager_modes
  def testEqualityNone(self):
    # [FUNCTION DOC]
    # Compares a float tensor with the null value in both argument orders. Asserts the tensors are
    # not equal to null under python comparison, the equals op returns false and the not-equals
    # op returns true.
    x = constant_op.constant([1.0, 2.0, 0.0, 4.0], dtype=dtypes.float32)
    self.assertNotEqual(x, None)
    self.assertNotEqual(None, x)
    self.assertFalse(math_ops.tensor_equals(x, None))
    self.assertTrue(math_ops.tensor_not_equals(x, None))

  @parameterized.named_parameters(
      (f"-is_equals={is_equals}-float_literal_type={type(float_literal)}"  # pylint: disable=g-complex-comprehension
       f"-float_literal={float_literal}", is_equals, float_literal)
      for float_literal in [4.6, np.float32(4.6), 4.4, np.float32(4.4)]
      for is_equals in [True, False])
  def testEqualityNoDowncast(self, is_equals, float_literal):
    # [FUNCTION DOC]
    # Parameterized over equality vs inequality and over float literals 4.6 and 4.4 as python float
    # and numpy float32. Compares an integer tensor 4 with the literal; the comparison must not
    # downcast the literal to an integer. Either a type error is raised (accepted, test returns) or
    # the result must be the opposite of the equality flag (not equal for the equals op, equal for
    # the not-equals op). Skipped for numpy float32 literals when the second API version is
    # enabled, and for python floats otherwise, due to a known type promotion bug.
    if (tf2.enabled() and isinstance(float_literal, np.float32) or
        not tf2.enabled() and isinstance(float_literal, float)):
      # TODO(b/199262800): Remove this skip
      self.skipTest("There is a bug in type promotion.")
    if is_equals:
      op = math_ops.tensor_equals
    else:
      op = math_ops.tensor_not_equals
    x = constant_op.constant(4)
    try:
      result = op(x, float_literal)
      if isinstance(result, tensor_lib.Tensor):
        result = self.evaluate(result)
    except TypeError:
      # Throwing a TypeError is OK
      return
    self.assertEqual(result, not is_equals)


@test_util.run_all_in_graph_and_eager_modes
class RangeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the range op and for converting python range objects to tensors, including overflow
  # edge cases. Runs in graph and eager modes.

  def testConvertToTensorRange(self):
    # [FUNCTION DOC]
    # Converts a python range of 5 elements to a tensor and asserts its static shape is a vector of
    # length 5 and its values equal the range.
    values = range(5)
    tensor = ops.convert_to_tensor(values)
    self.assertAllEqual((5,), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))

  def testInputsNearInt64Max(self):
    # [FUNCTION DOC]
    # Builds int64 ranges with a step close to the largest int64 value: with limit 201 (just below
    # potential overflow) and 202 (smallest limit with potential overflow) both must yield only the
    # single element zero.
    int64_t_max = 2**63 - 1
    x = math_ops.range(0, 201, int64_t_max - 200, dtype=dtypes.int64)
    self.assertAllEqual((0,), self.evaluate(x))  # just below potential overflow
    x = math_ops.range(0, 202, int64_t_max - 200, dtype=dtypes.int64)
    self.assertAllEqual(
        (0,), self.evaluate(x))  # smallest input with potential overflow

  def testInt32Overflow(self):
    # [FUNCTION DOC]
    # Uses a start, end and large negative step whose size calculation could overflow 32-bit
    # integers and asserts the range op matches numpy's arange.
    start = 1136033460
    end = -2110457150
    step = -1849827689
    expected = np.arange(start, end, step)
    actual = math_ops.range(start, end, step)
    self.assertAllEqual(expected, self.evaluate(actual))


@test_util.run_all_in_graph_and_eager_modes
class ErfcinvTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the inverse complementary error function, run in graph and eager modes.

  def testErfcinv(self):
    # [FUNCTION DOC]
    # Draws 10000 random float32 values in 0.1 to 1.9, applies the inverse function and then the
    # complementary error function, and asserts the original values are recovered.
    values = np.random.uniform(0.1, 1.9, size=int(1e4)).astype(np.float32)
    approx_id = math_ops.erfc(math_ops.erfcinv(values))
    self.assertAllClose(values, self.evaluate(approx_id))


@test_util.run_all_in_graph_and_eager_modes
class ArgMaxMinTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the index-of-maximum and index-of-minimum ops across dtypes, axes, output types and
  # invalid axis types, with helpers producing random inputs and the dtype list. Supports
  # parameterized tests; runs in graph and eager modes.

  def _generateRandomTensor(self, dtype, shape):
    # [FUNCTION DOC]
    # HELPER: random input generator for the arg-max and arg-min tests.
    # Inputs: a tensor dtype and a shape. For integer dtypes draws integers across the dtype's full
    # range inclusive of both ends; otherwise draws uniform values from minus one to one. Returns a
    # constant tensor of that dtype. Uses a fresh unseeded generator each call (nondeterministic).
    if dtype.is_integer:
      array = np.random.default_rng().integers(
          low=dtype.min, high=dtype.max, size=shape, endpoint=True)
      return constant_op.constant(array, dtype=dtype)
    else:
      array = np.random.default_rng().uniform(low=-1.0, high=1.0, size=shape)
      return constant_op.constant(array, dtype=dtype)

  def _getValidDtypes(self):
    # [FUNCTION DOC]
    # HELPER: returns the tuple of dtypes (bfloat16, half, single, double, int32, int64) iterated
    # over by the arg-max and arg-min tests.
    return (dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64,
            dtypes.int32, dtypes.int64)

  def testArgMax(self):
    # [FUNCTION DOC]
    # For each valid dtype, builds a random 24 by 8 tensor and asserts that arg-max along each axis
    # equals numpy's.
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_max = np.argmax(np_values, axis=axis)
        tf_max = math_ops.argmax(tf_values, axis=axis)
        self.assertAllEqual(tf_max, np_max)

  def testArgMaxReturnsFirstOccurence(self):
    # [FUNCTION DOC]
    # With ties in the data (rows with repeated maxima) asserts arg-max picks the first occurrence
    # like numpy; also checks a very long (193681 element) all-zero vector gives index 0, covering
    # multi-threaded and GPU code paths.
    for dtype in self._getValidDtypes():
      values = constant_op.constant(
          [[10, 11, 15, 15, 10], [12, 12, 10, 10, 12]], dtype=dtype)
      self.assertAllEqual(
          math_ops.argmax(values, axis=1),
          np.argmax(self.evaluate(values), axis=1))

      # Long tensor to ensure works with multithreading/GPU
      values = array_ops.zeros(shape=(193681,), dtype=dtype)
      self.assertAllEqual(math_ops.argmax(values), 0)

  def testArgMaxUint16(self):
    # [FUNCTION DOC]
    # Same as the basic arg-max test but requests unsigned 16-bit output indices and asserts the
    # results equal numpy's.
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_max = np.argmax(np_values, axis=axis)
        tf_max = math_ops.argmax(
            tf_values, axis=axis, output_type=dtypes.uint16)
        self.assertAllEqual(tf_max, np_max)

  @parameterized.parameters(
      dtypes.int8,
      dtypes.uint8,
      dtypes.int16,
      dtypes.uint16,
      dtypes.int32,
      dtypes.int64,
  )
  def testArgMaxWithIntegerDtypes(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over 8, 16, 32 and 64-bit signed and unsigned integer dtypes. Passes the axis
    # as a constant tensor of that integer dtype, requests int32 output, and asserts the result
    # equals numpy's arg-max along each axis of a random 24 by 8 tensor.
    shape = (24, 8)
    tf_values = self._generateRandomTensor(dtype, shape)
    np_values = self.evaluate(tf_values)
    for axis in range(len(shape)):
      np_max = np.argmax(np_values, axis=axis)
      tf_max = math_ops.argmax(
          tf_values,
          axis=constant_op.constant(axis, dtype=dtype),
          output_type=dtypes.int32,
      )
      self.assertAllEqual(tf_max, np_max)

  def testArgMaxRaisesInvalidDtype(self):
    # [FUNCTION DOC]
    # Passes a float32 constant as the axis and asserts a type error is raised.
    invalid_axis = constant_op.constant(1, dtype=dtypes.float32)
    with self.assertRaises(TypeError):
      math_ops.argmax(
          constant_op.constant([1, 2, 3]),
          axis=invalid_axis,
          output_type=dtypes.int32,
      )

  def testArgMin(self):
    # [FUNCTION DOC]
    # For each valid dtype, builds a random 24 by 8 tensor and asserts arg-min along each axis
    # equals numpy's.
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_min = np.argmin(np_values, axis=axis)
        tf_min = math_ops.argmin(tf_values, axis=axis)
        self.assertAllEqual(tf_min, np_min)

  def testArgMinReturnsFirstOccurence(self):
    # [FUNCTION DOC]
    # With ties at the minimum in each row, asserts arg-min along axis 1 picks the first occurrence
    # like numpy; also checks a long (193681 element) all-zero vector gives index 0.
    for dtype in self._getValidDtypes():
      values = constant_op.constant(
          [[10, 11, 15, 15, 10], [12, 12, 10, 10, 12]], dtype=dtype)
      self.assertAllEqual(
          math_ops.argmin(values, axis=1),
          np.argmin(self.evaluate(values), axis=1))

      # Long tensor to ensure works with multithreading/GPU
      values = array_ops.zeros(shape=(193681,), dtype=dtype)
      self.assertAllEqual(math_ops.argmin(values), 0)


class CastTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the cast op's interaction with full type information on ops. Not decorated for
  # graph/eager modes.

  def testCastWithFullType(self):
    # [FUNCTION DOC]
    # Traces a function that creates a tensor array, casts its flow value to the variant dtype,
    # attaches a full-type annotation marking it as a product holding an array type to the cast
    # op, builds a second tensor array from that flow, writes 1, and stacks it. Asserts the
    # evaluated result is a vector containing 1.

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # HELPER: traced function for the cast test; see the test for the scenario. Takes no inputs
      # and returns the stacked tensor array contents (a vector with the single value 1).
      ta = tensor_array_ops.TensorArray(dtypes.int32, size=1)
      h = math_ops.cast(ta.flow, dtypes.variant)

      t = full_type_pb2.FullTypeDef(
          type_id=full_type_pb2.TFT_PRODUCT,
          args=[full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_ARRAY)])
      h.op.experimental_set_type(t)

      ta = tensor_array_ops.TensorArray(dtypes.int32, flow=h)
      ta = ta.write(0, constant_op.constant(1))
      return ta.stack()

    self.assertAllEqual(self.evaluate(test_fn()), [1])

if __name__ == "__main__":
  googletest.main()
