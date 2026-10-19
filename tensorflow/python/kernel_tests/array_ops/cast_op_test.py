# [MODULE DOC]
# Unit tests for the cast op and the saturating cast. Checks numeric conversions against numpy
# across float, integer, bool, complex, bfloat16 and 8-bit float types on CPU and GPU, boundary and
# special values (infinity, not-a-number), unsupported casts, casting to a variable's dtype,
# gradients through casts, reference dtypes, sparse tensor casts, and the clipping behavior of the
# saturating cast between many type pairs. Defines three test case classes; no import-time side
# effects.
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
"""Tests for tensorflow.ops.tf.cast."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class CastOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the general cast op. Provides helpers that convert numpy types to TensorFlow
  # dtypes, run a cast on a chosen device and compare against numpy conversion, then tests for
  # ordinary values, small and random values, special values, error cases, variables, gradients
  # and reference dtypes.

  def _toDataType(self, dtype):
    # [FUNCTION DOC]
    # HELPER: maps a numpy scalar type (32/64-bit float, 32/64-bit integer, bool, 64/128-bit
    # complex) to the matching TensorFlow dtype. Returns nothing/null value for any other type
    # instead of raising an error.
    """Returns TensorFlow data type for numpy type."""
    if dtype == np.float32:
      return dtypes.float32
    elif dtype == np.float64:
      return dtypes.float64
    elif dtype == np.int32:
      return dtypes.int32
    elif dtype == np.int64:
      return dtypes.int64
    elif dtype == np.bool_:
      return dtypes.bool
    elif dtype == np.complex64:
      return dtypes.complex64
    elif dtype == np.complex128:
      return dtypes.complex128
    else:
      return None

  def _cast(self, x, dtype, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: runs a TensorFlow cast. Takes a value or array x, the target numpy type and a flag
    # for GPU use (default false). Creates a constant of x with the dtype inferred from x, casts
    # it to the target type on the requested device and returns the evaluated numpy result.
    with test_util.device(use_gpu):
      val = constant_op.constant(x, self._toDataType(np.array([x]).dtype))
      cast = math_ops.cast(val, self._toDataType(dtype), name="cast")
      return self.evaluate(cast)

  def _test(self, x, dtype, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares a TensorFlow cast of x to the target numpy type (optionally on GPU) with
    # numpy's own conversion of x and asserts the results are equal. Returns nothing.
    """Tests cast(x) to dtype behaves the same as numpy.astype."""
    np_ans = x.astype(dtype)
    tf_ans = self._cast(x, dtype, use_gpu)
    self.assertAllEqual(np_ans, tf_ans)

  def _testTypes(self, x, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: checks casts of x over a matrix of types. Uses a list of float, integer and complex
    # types (without 32-bit integers when on GPU), converts x to each source type and casts to
    # each target type, then additionally checks bool and unsigned 8-bit sources to 32-bit float
    # and, on CPU only, bool and 32-bit integer sources to 32-bit integer. Returns nothing.
    """Tests cast(x) to different tf."""
    if use_gpu:
      type_list = [
          np.float32, np.float64, np.int64, np.complex64, np.complex128
      ]
    else:
      type_list = [
          np.float32, np.float64, np.int32, np.int64, np.complex64,
          np.complex128
      ]
    for from_type in type_list:
      for to_type in type_list:
        self._test(x.astype(from_type), to_type, use_gpu)

    self._test(x.astype(np.bool_), np.float32, use_gpu)
    self._test(x.astype(np.uint8), np.float32, use_gpu)
    if not use_gpu:
      self._test(x.astype(np.bool_), np.int32, use_gpu)
      self._test(x.astype(np.int32), np.int32, use_gpu)

  def _testAll(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the type-matrix check for x on CPU, and also on GPU when x is a 32-bit or
    # 64-bit float array. Returns nothing.
    self._testTypes(x, use_gpu=False)
    if x.dtype == np.float32 or x.dtype == np.float64:
      self._testTypes(x, use_gpu=True)

  def testBasic(self):
    # [FUNCTION DOC]
    # Runs the all-types cast check on a 2x10 integer range from -10 to 9 and on 17 evenly spaced
    # floats between -10 and 10. Returns nothing.
    self._testAll(np.arange(-10, 10).reshape(2, 10))
    self._testAll(np.linspace(-10, 10, 17))

  def testSmallValues(self):
    # [FUNCTION DOC]
    # Runs the all-types cast check on a vector of zero, plus and minus one and the float32 and
    # float64 machine resolutions. Returns nothing.
    f4 = np.finfo(np.float32)
    f8 = np.finfo(np.float64)
    self._testAll(
        np.array([
            0, -1, 1, -f4.resolution, f4.resolution, f8.resolution,
            -f8.resolution
        ]))

  def testBfloat16(self):
    # [FUNCTION DOC]
    # Casts 100 random float32 values in [-100, 100] to bfloat16 and back to float32, first on
    # CPU only and then on the default device, and asserts the round trip is close to the
    # original within a relative tolerance of 1/128. Returns nothing.
    a = np.random.uniform(-100, 100, 100).astype(np.float32)
    with self.cached_session(use_gpu=False):
      b = math_ops.cast(math_ops.cast(a, dtypes.bfloat16), dtypes.float32)
      self.assertAllClose(a, self.evaluate(b), rtol=1 / 128.)
    with self.cached_session():
      b = math_ops.cast(math_ops.cast(a, dtypes.bfloat16), dtypes.float32)
      self.assertAllClose(a, self.evaluate(b), rtol=1 / 128.)

  def testFloat8(self):
    # [FUNCTION DOC]
    # For both 8-bit float formats, converts random float32 values to that format, casts to each
    # of several wider float types (including the e4m3 format, which should hold all 8-bit values
    # exactly) and back to the 8-bit format, and asserts the round trip is lossless. Returns
    # nothing.
    a = np.random.uniform(-100, 100, 100).astype(np.float32)
    for float8 in (dtypes.float8_e4m3fn, dtypes.float8_e5m2):
      # Including float8_e4m3fn should cover the float8 combinations without
      # loss of precision.
      for dtype in (dtypes.float64, dtypes.float32, dtypes.bfloat16,
                    dtypes.float16, dtypes.float8_e4m3fn):
        with self.cached_session(use_gpu=True):
          b = ops.convert_to_tensor(a, float8)
          c = math_ops.cast(math_ops.cast(b, dtype), float8)
          self.assertAllEqual(b, c)

  def testRandom(self):
    # [FUNCTION DOC]
    # Runs the all-types cast check on 2x3x5x7 arrays of random normal values with standard
    # deviation 10 and 1e6. Returns nothing.
    self._testAll(np.random.normal(0, 10, 210).reshape([2, 3, 5, 7]))
    self._testAll(np.random.normal(0, 1e6, 210).reshape([2, 3, 5, 7]))

  # Special values like int32max, int64min, inf, -inf, nan casted to
  # integer values in somewhat unexpected ways. And they behave
  # differently on CPU and GPU.
  def _compare(self, x, dst_dtype, expected, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: casts x to the numpy type dst_dtype (optionally on GPU) and asserts it equals the
    # expected value converted to that same type. Returns nothing.
    np.testing.assert_equal(
        self._cast(
            x, dst_dtype, use_gpu=use_gpu), dst_dtype(expected))

  def testIntToFloatBoundary(self):
    # [FUNCTION DOC]
    # On CPU, casts the minimum and maximum 32-bit and 64-bit integers to 32-bit and 64-bit floats
    # and asserts each equals the same value converted to float. GPU is not tested because it
    # does not support integer casts. Returns nothing.
    i4 = np.iinfo(np.int32)
    i8 = np.iinfo(np.int64)

    self._compare(i4.min, np.float32, i4.min, False)
    self._compare(i4.max, np.float32, i4.max, False)
    self._compare(i8.min, np.float32, i8.min, False)
    self._compare(i8.max, np.float32, i8.max, False)
    self._compare(i4.min, np.float64, i4.min, False)
    self._compare(i4.max, np.float64, i4.max, False)
    self._compare(i8.min, np.float64, i8.min, False)
    self._compare(i8.max, np.float64, i8.max, False)
    # NOTE: GPU does not support int32/int64 for casting.

  def testInfNan(self):
    # [FUNCTION DOC]
    # Casts positive infinity, negative infinity and not-a-number to 32-bit and 64-bit floats,
    # on CPU then on GPU, asserting infinities are preserved and not-a-number stays not-a-number.
    # Returns nothing.
    self._compare(np.inf, np.float32, np.inf, False)
    self._compare(np.inf, np.float64, np.inf, False)
    self._compare(-np.inf, np.float32, -np.inf, False)
    self._compare(-np.inf, np.float64, -np.inf, False)
    self.assertAllEqual(np.isnan(self._cast(np.nan, np.float32, False)), True)
    self.assertAllEqual(np.isnan(self._cast(np.nan, np.float64, False)), True)

    self._compare(np.inf, np.float32, np.inf, True)
    self._compare(np.inf, np.float64, np.inf, True)
    self._compare(-np.inf, np.float32, -np.inf, True)
    self._compare(-np.inf, np.float64, -np.inf, True)
    self.assertAllEqual(np.isnan(self._cast(np.nan, np.float32, True)), True)
    self.assertAllEqual(np.isnan(self._cast(np.nan, np.float64, True)), True)

  def _OpError(self, x, dtype, err):
    # [FUNCTION DOC]
    # HELPER: casts x to dtype and evaluates it, asserting that an op error whose message matches
    # the pattern err is raised. Returns nothing.
    with self.assertRaisesOpError(err):
      self.evaluate(math_ops.cast(x, dtype))

  def testNotImplemented(self):
    # [FUNCTION DOC]
    # Casting an integer array to string is unsupported; asserts the op error says that a cast
    # from integer to string is not implemented. Returns nothing.
    self._OpError(np.arange(0, 10), dtypes.string, "Cast.*int.*string.*")

  def testCastToTypeOfVariable(self):
    # [FUNCTION DOC]
    # Casts a boolean variable (true) to the dtype of a float32 variable, after initializing
    # variables, and asserts the result equals 1.0. Returns nothing.
    with self.cached_session():
      x = variables.Variable(5, dtype=dtypes.float32)
      y = variables.Variable(True, dtype=dtypes.bool)
      cast = math_ops.cast(y, x.dtype)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(cast))

  def testGradients(self):
    # [FUNCTION DOC]
    # For every ordered pair of float and complex types, numerically checks the gradient of an
    # identity followed by a cast from the source type to the destination type, asserting the
    # maximum error is below 1e-3. Returns nothing.
    t = [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]
    for src_t in t:
      for dst_t in t:
        with self.cached_session():
          x = constant_op.constant(1.0, src_t)

          def cast(x, dst_t=dst_t):
            # [FUNCTION DOC]
            # HELPER: function under differentiation. Applies an identity to x, then casts the
            # result to the destination type (bound at definition time through the dst_t default
            # argument) and returns it. No side effects.
            x = array_ops.identity(x)
            x = math_ops.cast(x, dst_t)
            return x

          err = gradient_checker_v2.max_error(
              *gradient_checker_v2.compute_gradient(cast, [x]))
          self.assertLess(err, 1e-3)

  def testRefDtype(self):
    # [FUNCTION DOC]
    # In graph mode, creates a raw reference-typed float32 variable, casts it to float32 and
    # asserts the input has the float32 reference dtype while the cast result has the plain
    # float32 dtype. Returns nothing.
    with context.graph_mode(), self.cached_session():
      x = gen_state_ops.variable(shape=[1], dtype=dtypes.float32)
      result = math_ops.cast(x, dtypes.float32)
      self.assertEqual(x.dtype, dtypes.float32_ref)
      self.assertEqual(result.dtype, dtypes.float32)


class SparseTensorCastTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that the cast op also accepts sparse tensors.

  def testCast(self):
    # [FUNCTION DOC]
    # Builds a sparse tensor of three 64-bit integer values in a one-dimensional dense shape of
    # 3, casts it to 32-bit float and asserts the indices and dense shape are unchanged and the
    # values are the converted floats. Returns nothing.
    indices = constant_op.constant([[0], [1], [2]], dtypes.int64)
    values = constant_op.constant(np.array([1, 2, 3], np.int64))
    shape = constant_op.constant([3], dtypes.int64)
    st = sparse_tensor.SparseTensor(indices, values, shape)
    st_cast = math_ops.cast(st, dtypes.float32)

    self.assertAllEqual(st_cast.indices, [[0], [1], [2]])
    self.assertAllEqual(st_cast.values,
                        np.array([1, 2, 3], np.float32))
    self.assertAllEqual(st_cast.dense_shape, [3])


class SaturateCastTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the saturating cast, which clips values into the range of the target type
  # instead of overflowing. Covers real to real, real to complex, complex to real, complex to
  # complex, avoidance of undefined behavior near range limits, and infinity/not-a-number handling.

  def testSaturate(self):
    # [FUNCTION DOC]
    # Saturating-casts extreme 32-bit float values (minimum, maximum and neighbors) to several
    # integer types and 32-bit float, asserts the result dtype and that the values equal the
    # inputs clipped to the target type's range. Returns nothing.
    in_types = dtypes.float32,
    out_types = dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.float32
    for in_type in in_types:
      for out_type in out_types:
        lo, hi = in_type.min, in_type.max
        x = constant_op.constant(
            [lo, lo + 1, lo // 2, hi // 2, hi - 1, hi], dtype=in_type)
        y = math_ops.saturate_cast(x, dtype=out_type)
        self.assertEqual(y.dtype, out_type)
        x, y = self.evaluate([x, y])
        correct = np.maximum(out_type.min, np.minimum(out_type.max, x))
        self.assertAllEqual(correct, y)

  def testSaturateRealToComplex(self):
    # [FUNCTION DOC]
    # Saturating-casts extreme 32/64-bit float values to 64/128-bit complex types and asserts
    # the result dtype and that values equal the inputs clipped to the target's real range.
    # Returns nothing.
    in_types = (dtypes.float32, dtypes.float64)
    out_types = (dtypes.complex64, dtypes.complex128)
    for in_type in in_types:
      for out_type in out_types:
        lo, hi = in_type.min, in_type.max
        x = constant_op.constant([lo, lo + 1, lo // 2, hi // 2, hi - 1, hi],
                                 dtype=in_type)
        y = math_ops.saturate_cast(x, dtype=out_type)
        self.assertEqual(y.dtype, out_type)
        x, y = self.evaluate([x, y])
        correct = np.maximum(out_type.real_dtype.min,
                             np.minimum(out_type.real_dtype.max, x))
        self.assertAllEqual(correct, y)

  def testSaturateComplexToReal(self):
    # [FUNCTION DOC]
    # Saturating-casts extreme values stored in 64/128-bit complex types to 32/64-bit floats and
    # asserts the result dtype and the clipping to the target range. Returns nothing.
    in_types = (dtypes.complex64, dtypes.complex128)
    out_types = (dtypes.float32, dtypes.float64)
    for in_type in in_types:
      for out_type in out_types:
        lo, hi = in_type.real_dtype.min, in_type.real_dtype.max
        x = constant_op.constant([lo, lo + 1, lo // 2, hi // 2, hi - 1, hi],
                                 dtype=in_type)
        y = math_ops.saturate_cast(x, dtype=out_type)
        self.assertEqual(y.dtype, out_type)
        x, y = self.evaluate([x, y])
        correct = np.maximum(out_type.min, np.minimum(out_type.max, x))
        self.assertAllEqual(correct, y)

  def testSaturateAvoidsUndefinedBehavior(self):
    # [FUNCTION DOC]
    # For float32, float16 and bfloat16 inputs and several 8, 16 and 32-bit integer outputs,
    # saturating-casts values just beyond and within the representable limits (where the input
    # type cannot represent the output maximum exactly) and asserts each result is within one
    # representable input value of the correctly clipped answer, i.e. no out-of-range cast
    # artifacts. Returns nothing.
    # E.g. float32 -> uint32, float32 cannot represent uint32 max, so trying
    # to clip to that range can introduce values still outside those
    # representable by uint32, which result in UB when followed by a cast.
    for in_type in [dtypes.float32, dtypes.float16, dtypes.bfloat16]:
      for out_type in [
          dtypes.int32,
          dtypes.uint32,
          dtypes.int16,
          dtypes.uint16,
          dtypes.int8,
          dtypes.uint8,
      ]:
        # Avoid casting issues by using comparisons instead of min/max to
        # enable proper type promotion.
        lo = (
            out_type.min
            if float(out_type.min) > (in_type.min + 1)
            else (in_type.min + 1)
        )
        hi = (
            out_type.max
            if float(out_type.max) < (in_type.max - 1)
            else (in_type.max - 1)
        )
        x = constant_op.constant(
            [lo - 1, lo, lo + 1, lo // 2, hi // 2, hi - 1, hi, hi + 1],
            dtype=in_type,
        )
        y = math_ops.saturate_cast(x, out_type)
        x, y = self.evaluate([x, y])

        # Ensure that we are at most one representable input element away from
        # the true answer.
        np_in_type = in_type.as_numpy_dtype
        expected = np.clip(x, out_type.min, out_type.max).astype(np_in_type)
        u = y.astype(np_in_type)
        v = expected
        self.assertTrue(np.all(v == np.nextafter(u, v)))

  @test_util.disable_xla("Clamp is not implemented for C128 in XLA")
  def testSaturateComplexToComplex(self):
    # [FUNCTION DOC]
    # Skipped under XLA. Saturating-casts complex values (real part extremes with a transposed
    # copy as imaginary part) between 64 and 128-bit complex types and asserts the result dtype
    # and that real and imaginary parts are clipped to the target range. Returns nothing.
    in_types = (dtypes.complex64, dtypes.complex128)
    out_types = (dtypes.complex64, dtypes.complex128)
    for in_type in in_types:
      for out_type in out_types:
        lo, hi = in_type.real_dtype.min, in_type.real_dtype.max
        x_real = constant_op.constant(
            [lo, lo + 1, lo // 2, hi // 2, hi - 1, hi],
            dtype=in_type.real_dtype)
        x = math_ops.complex(x_real, array_ops.transpose(x_real))
        y = math_ops.saturate_cast(x, dtype=out_type)
        self.assertEqual(y.dtype, out_type)
        x, y = self.evaluate([x, y])
        correct = np.maximum(
            out_type.real_dtype.min,
            np.minimum(out_type.real_dtype.max, np.real(x))) + 1j * np.maximum(
                out_type.real_dtype.min,
                np.minimum(out_type.real_dtype.max, np.imag(x)))
        self.assertAllEqual(correct, y)

  @test_util.with_forward_compatibility_horizons([2024, 11, 2])
  def testSaturateHandlesInfs(self):
    # [FUNCTION DOC]
    # Runs with a forward compatibility horizon set to a date in November 2024. For all pairs of
    # 64, 32, 16-bit float and bfloat16 types, saturating-casts positive infinity, negative
    # infinity and not-a-number values and asserts the infinities become finite while the
    # not-a-number values stay non-finite. Returns nothing.
    for in_type in [
        dtypes.float64,
        dtypes.float32,
        dtypes.float16,
        dtypes.bfloat16,
    ]:
      for out_type in [
          dtypes.float64,
          dtypes.float32,
          dtypes.float16,
          dtypes.bfloat16,
      ]:
        inf = float("inf")
        nan = float("nan")
        x = constant_op.constant([inf, -inf, nan, -nan], dtype=in_type)
        y = math_ops.saturate_cast(x, dtype=out_type)
        self.assertAllEqual(math_ops.is_finite(y), [True, True, False, False])


if __name__ == "__main__":
  test.main()
