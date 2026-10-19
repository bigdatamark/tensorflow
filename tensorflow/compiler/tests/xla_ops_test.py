# [MODULE DOC]
# Tests for the Python wrappers of XLA-specific operations (add, broadcast, gather, shifts,
# convolution, dot, pad, dynamic slice and update, reduce, variadic reduce, scatter,
# select-and-scatter, transpose, random bit generator, optimization barrier, reduce precision, ...).
# Two test classes: one runs the ops on the XLA test device and compares numeric results with
# expected arrays; the other checks only static shape inference and argument validation
# errors at graph-construction time. Built on the XLA test fixture module in this directory.
# Running the file as a program disables eager execution (sessions are used) and starts the runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for XLA op wrappers."""

import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.tf2xla.python import xla
from xla import xla_data_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import random_ops_util
from tensorflow.python.platform import googletest


class XlaOpsNumericalTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Numerical tests: each test builds inputs, runs one XLA op wrapper on the test device inside
  # a session and compares to hand-computed expected values. Combines the XLA test fixture
  # (device, type sets) with parameterized test support. Holds a class-level tuple of
  # precision settings (including none) and a list of boolean choices used for parameterizing.

  def _assertOpOutputMatchesExpected(
      self, op, args, expected, equality_fn=None
  ):
    # [FUNCTION DOC]
    # HELPER: shared assertion used by most tests in this class.
    # Inputs: op (callable taking tensors and returning a tensor or tuple), args (sequence of
    # numpy arrays or scalars fed through placeholders of matching type and shape), expected
    # (value or tuple of values), equality_fn (optional callable of result and expected; default
    # is a closeness check with relative tolerance 1e-3).
    # Opens a session, builds placeholders and the op under the test device scope, runs it with
    # the arguments fed, and compares. Returns nothing; fails the test on mismatch.
    with self.session() as session:
      with self.test_scope():
        placeholders = [
            array_ops.placeholder(dtypes.as_dtype(arg.dtype), arg.shape)
            for arg in args
        ]
        feeds = {placeholders[i]: args[i] for i in range(0, len(args))}
        output = op(*placeholders)
      result = session.run(output, feeds)
      if not equality_fn:
        equality_fn = lambda x, y: self.assertAllClose(x, y, rtol=1e-3)
      equality_fn(result, expected)

  def testAdd(self):
    # [FUNCTION DOC]
    # Skipped on ROCm builds. For each numeric type: adds two equal-length vectors, and adds a 2x2
    # matrix to a vector with the vector broadcast along dimension 0 and then dimension 1.
    # Expects element-wise sums with the corresponding broadcasting.
    if xla_test.test.is_built_with_rocm():
      self.skipTest('Broken with rocm')
    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          xla.add,
          args=(
              np.array([1, 2, 3], dtype=dtype),
              np.array([4, 5, 6], dtype=dtype),
          ),
          expected=np.array([5, 7, 9], dtype=dtype),
      )

      self._assertOpOutputMatchesExpected(
          lambda x, y: xla.add(x, y, broadcast_dims=(0,)),
          args=(
              np.array([[1, 2], [3, 4]], dtype=dtype),
              np.array([7, 11], dtype=dtype),
          ),
          expected=np.array([[8, 9], [14, 15]], dtype=dtype),
      )

      self._assertOpOutputMatchesExpected(
          lambda x, y: xla.add(x, y, broadcast_dims=(1,)),
          args=(
              np.array([[1, 2], [3, 4]], dtype=dtype),
              np.array([7, 11], dtype=dtype),
          ),
          expected=np.array([[8, 13], [10, 15]], dtype=dtype),
      )

  def testBroadcast(self):
    # [FUNCTION DOC]
    # For each numeric type: broadcasts a 2x2 matrix by prepending dimensions of sizes 7 and 42.
    # Expects the matrix tiled into a 7x42x2x2 result.
    for dtype in self.numeric_types:
      v = np.arange(4, dtype=np.int32).astype(dtype).reshape([2, 2])
      self._assertOpOutputMatchesExpected(
          lambda x: xla.broadcast(x, (7, 42)),
          args=(v,),
          expected=np.tile(v, (7, 42, 1, 1)),
      )

  @test_util.disable_mlir_bridge('Not supported yet')
  def testGather(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge. Gathers a slice of three elements at index 2 on a 2x5 integer
    # matrix with a hand-built gather dimension description (offset dim 1, collapsed dim 0,
    # index map 0, index vector dim 1). Expects the single row of values 5, 6, 7.
    operand = np.arange(10, dtype=np.int32).reshape([2, 5])
    start_indices = np.array([2], np.int32)
    slice_sizes = np.array([1, 3], np.int32)

    def gather(operand, start_indices):
      # [FUNCTION DOC]
      # Builds the gather dimension description and calls the XLA gather wrapper using the
      # enclosing test's slice sizes. Returns the gathered tensor.
      dimension_numbers = xla_data_pb2.GatherDimensionNumbers()
      dimension_numbers.offset_dims.extend([1])
      dimension_numbers.collapsed_slice_dims.extend([0])
      dimension_numbers.start_index_map.extend([0])
      dimension_numbers.index_vector_dim = 1
      return xla.gather(operand, start_indices, dimension_numbers, slice_sizes)

    self._assertOpOutputMatchesExpected(
        gather, args=(operand, start_indices), expected=np.array([[5, 6, 7]])
    )

  def testShiftRightLogical(self):
    # [FUNCTION DOC]
    # Logical right shift by 4 on signed 32-bit values (negative one and 16) and on unsigned
    # 32-bit values (all ones and 16). Expects zero-filled high bits: 0x0FFFFFFF and 1 in both.
    self._assertOpOutputMatchesExpected(
        xla.shift_right_logical,
        args=(np.array([-1, 16], dtype=np.int32), np.int32(4)),
        expected=np.array([0x0FFFFFFF, 1], dtype=np.int32),
    )

    self._assertOpOutputMatchesExpected(
        xla.shift_right_logical,
        args=(np.array([0xFFFFFFFF, 16], dtype=np.uint32), np.uint32(4)),
        expected=np.array([0x0FFFFFFF, 1], dtype=np.uint32),
    )

  def testShiftRightArithmetic(self):
    # [FUNCTION DOC]
    # Arithmetic right shift by 4 on signed 32-bit values (negative one and 16) and on unsigned
    # 32-bit values (all ones and 16). Expects sign extension for the signed case (negative one, 1)
    # and, for unsigned input, all-ones and 1 as the expected result.
    self._assertOpOutputMatchesExpected(
        xla.shift_right_arithmetic,
        args=(np.array([-1, 16], dtype=np.int32), np.int32(4)),
        expected=np.array([-1, 1], dtype=np.int32),
    )

    self._assertOpOutputMatchesExpected(
        xla.shift_right_arithmetic,
        args=(np.array([0xFFFFFFFF, 16], dtype=np.uint32), np.uint32(4)),
        expected=np.array([0xFFFFFFFF, 1], dtype=np.uint32),
    )

  PRECISION_VALUES = (
      None,
      xla_data_pb2.PrecisionConfig.DEFAULT,
      xla_data_pb2.PrecisionConfig.HIGH,
      xla_data_pb2.PrecisionConfig.HIGHEST,
  )

  @parameterized.parameters(*PRECISION_VALUES)
  def testConv(self, precision):
    # [FUNCTION DOC]
    # Parameterized over precision settings (none, default, high, highest). For bfloat16 and
    # float32 (when supported) runs a 1-D convolution of a length-4 input with a length-2 kernel,
    # stride 1, padding (2, 1), kernel dilation 2, and optionally a precision config applied to
    # both operands. Expects the five values -9, -12, -21, -26, -10.
    for dtype in set(self.float_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])
    ):
      def conv_1d_fn(lhs, rhs):
        # [FUNCTION DOC]
        # Builds batch/feature/spatial dimension numbers for a 1-D convolution and the optional
        # precision config from the enclosing parameter, then calls the XLA convolution wrapper.
        # Returns the convolution result.
        dnums = xla_data_pb2.ConvolutionDimensionNumbers()
        num_spatial_dims = 1
        dnums.input_batch_dimension = 0
        dnums.input_feature_dimension = 1
        dnums.output_batch_dimension = 0
        dnums.output_feature_dimension = 1
        dnums.kernel_output_feature_dimension = 0
        dnums.kernel_input_feature_dimension = 1
        dnums.input_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
        dnums.kernel_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
        dnums.output_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
        precision_config = None
        if precision:
          precision_config = xla_data_pb2.PrecisionConfig()
          precision_config.operand_precision.extend([precision, precision])
        return xla.conv(
            lhs,
            rhs,
            window_strides=(1,),
            padding=((2, 1),),
            lhs_dilation=(1,),
            rhs_dilation=(2,),
            dimension_numbers=dnums,
            precision_config=precision_config,
        )

      self._assertOpOutputMatchesExpected(
          conv_1d_fn,
          args=(
              np.array([[[3, 4, 5, 6]]], dtype=dtype),
              np.array([[[-2, -3]]], dtype=dtype),
          ),
          expected=np.array([[[-9, -12, -21, -26, -10]]], dtype=dtype),
      )

  def testConvPreferredElementType(self):
    # [FUNCTION DOC]
    # Runs the same 1-D convolution as the main conv test on float16 inputs, requesting a float32
    # result type. Expects the same five values (-9, -12, -21, -26, -10) typed float32.
    dtype = np.float16
    preferred_element_type = np.float32

    def conv_1d_fn(lhs, rhs):
      # [FUNCTION DOC]
      # Builds 1-D convolution dimension numbers (no precision config) and calls the XLA
      # convolution wrapper with the preferred element type float32. Returns the result.
      dnums = xla_data_pb2.ConvolutionDimensionNumbers()
      num_spatial_dims = 1
      dnums.input_batch_dimension = 0
      dnums.input_feature_dimension = 1
      dnums.output_batch_dimension = 0
      dnums.output_feature_dimension = 1
      dnums.kernel_output_feature_dimension = 0
      dnums.kernel_input_feature_dimension = 1
      dnums.input_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
      dnums.kernel_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
      dnums.output_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
      precision_config = None
      return xla.conv(
          lhs,
          rhs,
          window_strides=(1,),
          padding=((2, 1),),
          lhs_dilation=(1,),
          rhs_dilation=(2,),
          dimension_numbers=dnums,
          precision_config=precision_config,
          preferred_element_type=preferred_element_type,
      )

    self._assertOpOutputMatchesExpected(
        conv_1d_fn,
        args=(
            np.array([[[3, 4, 5, 6]]], dtype=dtype),
            np.array([[[-2, -3]]], dtype=dtype),
        ),
        expected=np.array(
            [[[-9, -12, -21, -26, -10]]], dtype=preferred_element_type
        ),
    )

  @parameterized.parameters(*PRECISION_VALUES)
  def testDotGeneral(self, precision):
    # [FUNCTION DOC]
    # Parameterized over precision settings. For each float type: batched matrix product of a
    # 2x2x2 and a 2x2x3 array, batch dimension 0, contracting lhs dim 2 with rhs dim 1,
    # with optional precision config. Expects the 2x2x3 result of per-batch matrix products.
    for dtype in self.float_types:

      def dot_fn(lhs, rhs):
        # [FUNCTION DOC]
        # Builds the dot dimension numbers and optional precision config from the enclosing
        # parameter and calls the XLA general dot wrapper. Returns the product.
        dnums = xla_data_pb2.DotDimensionNumbers()
        dnums.lhs_contracting_dimensions.append(2)
        dnums.rhs_contracting_dimensions.append(1)
        dnums.lhs_batch_dimensions.append(0)
        dnums.rhs_batch_dimensions.append(0)
        precision_config = None
        if precision:
          precision_config = xla_data_pb2.PrecisionConfig()
          precision_config.operand_precision.extend([precision, precision])
        return xla.dot_general(
            lhs, rhs, dimension_numbers=dnums, precision_config=precision_config
        )

      lhs = np.array(
          [
              [[1, 2], [3, 4]],
              [[5, 6], [7, 8]],
          ],
          dtype=dtype,
      )
      rhs = np.array(
          [
              [[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]],
          ],
          dtype=dtype,
      )
      self._assertOpOutputMatchesExpected(
          dot_fn,
          args=(lhs, rhs),
          expected=np.array(
              [
                  [[9, 12, 15], [19, 26, 33]],
                  [[95, 106, 117], [129, 144, 159]],
              ],
              dtype=dtype,
          ),
      )

  def testDotGeneralInt8xInt8ToInt32(self):
    # [FUNCTION DOC]
    # Same batched dot as the float test but with signed 8-bit operands and a requested 32-bit
    # integer result type. Expects the same product values typed as 32-bit integers.
    def dot_fn(lhs, rhs):
      # [FUNCTION DOC]
      # Builds the dot dimension numbers and calls the XLA general dot wrapper with the preferred
      # element type 32-bit integer. Returns the product.
      dnums = xla_data_pb2.DotDimensionNumbers()
      dnums.lhs_contracting_dimensions.append(2)
      dnums.rhs_contracting_dimensions.append(1)
      dnums.lhs_batch_dimensions.append(0)
      dnums.rhs_batch_dimensions.append(0)
      return xla.dot_general(
          lhs, rhs, dimension_numbers=dnums, preferred_element_type=np.int32
      )

    lhs = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int8,
    )
    rhs = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.int8,
    )
    self._assertOpOutputMatchesExpected(
        dot_fn,
        args=(lhs, rhs),
        expected=np.array(
            [
                [[9, 12, 15], [19, 26, 33]],
                [[95, 106, 117], [129, 144, 159]],
            ],
            dtype=np.int32,
        ),
    )

  def testNeg(self):
    # [FUNCTION DOC]
    # Negates the vector 1, 2, 3 for every numeric type except the 8-bit integer types.
    # Expects negative one, two, three.
    for dtype in self.numeric_types - {np.uint8, np.int8}:
      self._assertOpOutputMatchesExpected(
          xla.neg,
          args=(np.array([1, 2, 3], dtype=dtype),),
          expected=np.array([-1, -2, -3], dtype=dtype),
      )

  def testPad(self):
    # [FUNCTION DOC]
    # For each numeric type: pads a 2x2 matrix (0..3) with value 7, low padding (2, 1), high
    # padding (1, 2), interior padding (1, 0). Expects a 6x5 matrix with the original values
    # separated by a row of padding and surrounded by padding.
    for dtype in self.numeric_types:

      def pad_fn(x):
        # [FUNCTION DOC]
        # Calls the XLA pad wrapper with the fixed padding value and low/high/interior padding
        # described by the enclosing test. Returns the padded tensor.
        return xla.pad(
            x,
            padding_value=7,
            padding_low=[2, 1],
            padding_high=[1, 2],
            padding_interior=[1, 0],
        )

      self._assertOpOutputMatchesExpected(
          pad_fn,
          args=(np.arange(4, dtype=np.int32).astype(dtype).reshape([2, 2]),),
          expected=np.array(
              [
                  [7, 7, 7, 7, 7],
                  [7, 7, 7, 7, 7],
                  [7, 0, 1, 7, 7],
                  [7, 7, 7, 7, 7],
                  [7, 2, 3, 7, 7],
                  [7, 7, 7, 7, 7],
              ],
              dtype=dtype,
          ),
      )

  def testSetDynamicDimensionSize(self):
    # [FUNCTION DOC]
    # For int32, float32, float64 and complex64 (bfloat16 is unsupported), marks dimension 0 of a
    # ten-element vector as having dynamic size 7. Expects the first seven elements.
    dynamic_size = 7

    # XLA doesn't support this for bfloat16.
    for dtype in set(self.numeric_types).intersection(
        set([np.int32, np.float32, np.float64, np.complex64])
    ):
      def xla_set_dynamic_dimension_size_fn(x):
        # [FUNCTION DOC]
        # Applies the raw set-dynamic-dimension-size op to dimension 0 with the enclosing size.
        # Returns the resulting tensor.
        # Tell XLA to cut the array to size=dynamic_size.
        return gen_xla_ops.xla_set_dynamic_dimension_size(
            x, dim_index=0, size=dynamic_size
        )

      a = np.arange(10, dtype=np.int32).astype(dtype)
      expected = a[:dynamic_size]

      self._assertOpOutputMatchesExpected(
          xla_set_dynamic_dimension_size_fn, args=(a,), expected=expected
      )

  def testPadNegative(self):
    # [FUNCTION DOC]
    # For each numeric type: pads a 2x3 matrix (0..5) with value 7 using negative low and high
    # padding on the second dimension (which crops) plus interior padding (1, 2). Expects a
    # 4x4 result with the cropped and spaced values.
    for dtype in self.numeric_types:

      def pad_fn(x):
        # [FUNCTION DOC]
        # Calls the XLA pad wrapper with padding value 7, low (0, -1), high (1, -2), interior
        # (1, 2). Returns the padded tensor.
        return xla.pad(
            x,
            padding_value=7,
            padding_low=[0, -1],
            padding_high=[1, -2],
            padding_interior=[1, 2],
        )

      self._assertOpOutputMatchesExpected(
          pad_fn,
          args=(np.arange(6, dtype=np.int32).astype(dtype).reshape([2, 3]),),
          expected=np.array(
              [[7, 7, 1, 7], [7, 7, 7, 7], [7, 7, 4, 7], [7, 7, 7, 7]],
              dtype=dtype,
          ),
      )

  @parameterized.parameters(
      random_ops_util.Algorithm.THREEFRY,
      random_ops_util.Algorithm.PHILOX,
      random_ops_util.Algorithm.AUTO_SELECT,
  )
  def testRngBitGeneratorIsDeterministic(self, algorithm):
    # [FUNCTION DOC]
    # Parameterized over three random algorithms (Threefry, Philox, automatic). Calls the
    # random bit generator twice with the same two-element key and a 10x12 unsigned 32-bit output
    # shape and subtracts the corresponding outputs. Expects all zeros for both the new state and
    # the generated bits, showing the generator is deterministic.
    dtype = np.uint32
    key = np.array([1, 2], dtype=np.uint64)
    shape = (10, 12)

    def rng_fun_is_deterministic(k):
      # [FUNCTION DOC]
      # Runs the generator twice with key k and returns the differences of the two state outputs
      # and of the two bit outputs.
      res1 = xla.rng_bit_generator(algorithm, k, shape, dtype=dtype)
      res2 = xla.rng_bit_generator(algorithm, k, shape, dtype=dtype)
      return (res1[0] - res2[0], res1[1] - res2[1])

    self._assertOpOutputMatchesExpected(
        rng_fun_is_deterministic,
        args=(key,),
        expected=(
            np.zeros(key.shape, dtype=key.dtype),
            np.zeros(shape, dtype=dtype),
        ),
    )

  def testReduce(self):
    # [FUNCTION DOC]
    # For bfloat16 and float32 (when numeric): reduces a 3x4 matrix (0..11) with a defined
    # addition function over no dims, dim 0, dim 1 and both dims with initial value 0,
    # expecting the input, column sums (12, 15, 18, 21), row sums (6, 22, 38) and 66; then
    # reduces dim 0 with a multiplication function and initial value 1, expecting (0, 45, 120, 231).
    for dtype in set(self.numeric_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])
    ):

      @function.Defun(dtype, dtype)  # pylint: disable=cell-var-from-loop
      def sum_reducer(x, y):
        # [FUNCTION DOC]
        # Reducer function: returns the sum of its two scalar arguments.
        return x + y

      def sum_reduction(dims):
        # [FUNCTION DOC]
        # Factory: given the dimensions to reduce, returns a callable that applies the XLA reduce
        # wrapper with initial value 0 and the addition reducer.
        def fn(x):
          # [FUNCTION DOC]
          # Reduces x over the captured dimensions with the addition reducer.
          return xla.reduce(
              x, init_value=0, dimensions_to_reduce=dims, reducer=sum_reducer  # pylint: disable=cell-var-from-loop
          )

        return fn

      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),
      )
      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[0]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.array([12, 15, 18, 21], dtype=dtype),
      )
      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[1]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.array([6, 22, 38], dtype=dtype),
      )
      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[0, 1]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=dtype(66),
      )

      @function.Defun(dtype, dtype)  # pylint: disable=cell-var-from-loop
      def mul_reducer(x, y):
        # [FUNCTION DOC]
        # Reducer function: returns the product of its two scalar arguments.
        return x * y

      def mul_reduction(dims):
        # [FUNCTION DOC]
        # Factory: given the dimensions to reduce, returns a callable that applies the XLA reduce
        # wrapper with initial value 1 and the multiplication reducer.
        def fn(x):
          # [FUNCTION DOC]
          # Reduces x over the captured dimensions with the multiplication reducer.
          return xla.reduce(
              x, init_value=1, dimensions_to_reduce=dims, reducer=mul_reducer  # pylint: disable=cell-var-from-loop
          )

        return fn

      self._assertOpOutputMatchesExpected(
          mul_reduction(dims=[0]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.array([0, 45, 120, 231], dtype=dtype),
      )

  IS_XLA_VARIADIC_REDUCE_V2 = [True, False]

  @parameterized.parameters(IS_XLA_VARIADIC_REDUCE_V2)
  def testVariadicReduceKahanSum(self, is_v2):
    # [FUNCTION DOC]
    # Parameterized on whether the version-2 wrapper or the raw generated op is used. For float32
    # and complex64: performs compensated (Kahan) summation, reducing a pair of (sum, error
    # compensation) tensors over a 3x4 array of values of widely different magnitudes, with rows
    # and columns randomly shuffled to check order independence. Checks reducing no dims (inputs
    # unchanged, compensation zero), dim 0, dim 1 and both dims against closed-form sums, and checks
    # the error term is close to zero (with a looser absolute tolerance).
    # Uses random shuffles from numpy's global generator (seeded in setUp).
    for dtype in set(self.numeric_types).intersection(
        set([np.float32, np.complex64])
    ):
      @def_function.function
      def kahan_sum_reducer(t0, t1):
        # [FUNCTION DOC]
        # Reducer: combines two (sum, compensation) pairs with the Kahan update and returns the new
        # (sum, compensation) pair.
        (s0, c0), (s1, c1) = t0, t1
        s0minusc = s0 - (c0 + c1)
        t = s1 + s0minusc
        c = (t - s1) - s0minusc
        s = t
        return s, c

      def kahan_sum_reduction(dims, output_idx):
        # [FUNCTION DOC]
        # Factory: given dimensions to reduce and which output (0 sum, 1 compensation) to pick,
        # returns a callable performing the variadic reduce.
        def fn(x):
          # [FUNCTION DOC]
          # Builds scalar specs, gets a concrete reducer function, and runs the variadic reduce on
          # x paired with zeros (using the v2 wrapper or the raw op depending on the parameter).
          # Returns the selected output.
          arg = array_ops.zeros([], dtype)  # pylint: disable=cell-var-from-loop
          reducer = kahan_sum_reducer.get_concrete_function(  # pylint: disable=cell-var-from-loop
              (arg, arg), (arg, arg)
          )

          if is_v2:
            return xla.variadic_reduce(
                (x, array_ops.zeros_like(x)),
                init_values=(arg, arg),
                dimensions_to_reduce=dims,
                reducer=reducer,
            )[output_idx]
          else:
            return gen_xla_ops.xla_variadic_reduce(
                (x, array_ops.zeros_like(x)),
                init_value=(arg, arg),
                dimensions_to_reduce=dims,
                reducer=reducer,
            )[output_idx]

        return fn

      xs = np.array([1e5, np.pi, -1e5, np.exp(1.0)])
      xs = np.array([xs, xs[::-1] / 3, xs / 7], dtype)
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[], output_idx=0), args=(xs,), expected=xs
      )
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[], output_idx=1),
          args=(xs,),
          expected=np.zeros_like(xs),
      )
      shuffle_indices = np.argsort(np.random.randn(xs.shape[0]))
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[0], output_idx=0),
          args=(xs[shuffle_indices],),
          expected=np.array(
              [
                  np.exp(1) / 3 + 1e5 * 8 / 7,
                  np.pi * 8 / 7 - 1e5 / 3,
                  -1e5 * 8 / 7 + np.pi / 3,
                  np.exp(1) * 8 / 7 + 1e5 / 3,
              ],
              dtype=dtype,
          ),
      )
      error_term_equality = functools.partial(
          self.assertAllClose, rtol=1e-3, atol=0.005
      )
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[0], output_idx=1),
          args=(xs[shuffle_indices],),
          expected=np.zeros_like(xs[0]),
          equality_fn=error_term_equality,
      )
      shuffle_indices = np.argsort(np.random.randn(xs.shape[1]))
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[1], output_idx=0),
          args=(xs[:, shuffle_indices],),
          expected=np.array(
              [
                  np.pi + np.exp(1.0),
                  (np.pi + np.exp(1.0)) / 3,
                  (np.pi + np.exp(1.0)) / 7,
              ],
              dtype=dtype,
          ),
      )
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[1], output_idx=1),
          args=(xs[:, shuffle_indices],),
          expected=np.zeros_like(xs[:, 0]),
          equality_fn=error_term_equality,
      )
      # Now, shuffle both dims.
      xs = xs[np.argsort(np.random.randn(xs.shape[0]))]
      xs = xs[:, np.argsort(np.random.randn(xs.shape[1]))]
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[0, 1], output_idx=0),
          args=(xs,),
          expected=dtype((np.pi + np.exp(1.0)) * 31 / 21),
      )
      self._assertOpOutputMatchesExpected(
          kahan_sum_reduction(dims=[0, 1], output_idx=1),
          args=(xs,),
          expected=dtype(0),
          equality_fn=error_term_equality,
      )

  @parameterized.parameters(IS_XLA_VARIADIC_REDUCE_V2)
  def testVariadicReduceSingleOp(self, is_v2):
    # [FUNCTION DOC]
    # Parameterized on v2 wrapper versus raw op. For every numeric type: variadic reduce with a
    # single input, an add reducer and initial value 0 over a 2x3 matrix, reducing dim 0
    # (5, 9, 13), dim 1 (9, 18) and both dims (27).
    @def_function.function
    def reducer_add(op_element, acc_val):
      # [FUNCTION DOC]
      # Reducer: returns a one-element tuple with the sum of the element and accumulator.
      return (op_element + acc_val,)

    for dtype in set(self.numeric_types):
      values = np.array([[1, 3, 5], [4, 6, 8]], dtype=dtype)
      init_val = np.array(0, dtype=dtype)
      arg_spec = array_ops.zeros([], dtype)  # pylint: disable=cell-var-from-loop
      reducer_func = reducer_add.get_concrete_function(arg_spec, arg_spec)

      def reduce(values, *, dimensions_to_reduce):
        # [FUNCTION DOC]
        # Runs the single-input variadic reduce (v2 wrapper or raw op depending on the parameter)
        # with the enclosing initial value and reducer; dimensions_to_reduce is a keyword-only
        # sequence of dimension indices. Returns the first (only) output.
        if is_v2:
          return xla.variadic_reduce(
              (values,),
              (init_val,),  # pylint: disable=cell-var-from-loop
              dimensions_to_reduce=dimensions_to_reduce,
              reducer=reducer_func,  # pylint: disable=cell-var-from-loop
          )[0]
        else:
          return gen_xla_ops.xla_variadic_reduce(
              (values,),
              (init_val,),  # pylint: disable=cell-var-from-loop
              dimensions_to_reduce=dimensions_to_reduce,
              reducer=reducer_func,  # pylint: disable=cell-var-from-loop
          )[0]

      # Reduce dimension 0
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(0,)),
          args=(values,),
          expected=np.array([5, 9, 13], dtype=dtype),
      )

      # Reduce dimension 1
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(1,)),
          args=(values,),
          expected=np.array([9, 18], dtype=dtype),
      )

      # Reduce dimensions 0 and 1
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(0, 1)),
          args=(values,),
          expected=np.array(27, dtype=dtype),
      )

  def testVariadicReduceV2DifferentTypes(self):
    # [FUNCTION DOC]
    # For every numeric type: v2 variadic reduce over two inputs of different types (the type under
    # test and 32-bit integer) with an add reducer on each. Reduces dim 0, dim 1, both dims and no
    # dims over a 2x3 matrix; expects the sums (or unchanged inputs) with the proper type for each.
    # Two ops, with different dtypes
    @def_function.function
    def reducer_add(op_element_1, op_element_2, acc_val_1, acc_val_2):
      # [FUNCTION DOC]
      # Reducer: returns a pair of the element-plus-accumulator sums for each of the two inputs.
      return (op_element_1 + acc_val_1, op_element_2 + acc_val_2)

    for dtype in set(self.numeric_types):
      values_1 = np.array([[1, 3, 5], [4, 6, 8]], dtype=dtype)
      values_2 = values_1.astype(np.int32)

      init_val_1 = np.array(0, dtype=dtype)  # pylint: disable=cell-var-from-loop
      init_val_2 = init_val_1.astype(np.int32)

      arg_spec_1 = array_ops.zeros([], dtype)  # pylint: disable=cell-var-from-loop
      arg_spec_2 = array_ops.zeros([], np.int32)
      reducer_func = reducer_add.get_concrete_function(
          arg_spec_1, arg_spec_2, arg_spec_1, arg_spec_2
      )  # pylint: disable=cell-var-from-loop

      def reduce(*values, dimensions_to_reduce):
        # [FUNCTION DOC]
        # Runs the v2 variadic reduce on the given input tensors with the two enclosing initial
        # values and reducer; dimensions_to_reduce is keyword-only. Returns all outputs.
        return xla.variadic_reduce(
            values,
            (
                init_val_1,  # pylint: disable=cell-var-from-loop
                init_val_2,  # pylint: disable=cell-var-from-loop
            ),
            dimensions_to_reduce=dimensions_to_reduce,
            reducer=reducer_func,  # pylint: disable=cell-var-from-loop
        )

      # Reduce dimension 0
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(0,)),
          args=(values_1, values_2),
          expected=(
              np.array([5, 9, 13], dtype=dtype),
              np.array([5, 9, 13], dtype=np.int32),
          ),
      )

      # Reduce dimension 1
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(1,)),
          args=(values_1, values_2),
          expected=(
              np.array([9, 18], dtype=dtype),
              np.array([9, 18], dtype=np.int32),
          ),
      )

      # Reduce dimensions 0 and 1
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=(0, 1)),
          args=(values_1, values_2),
          expected=(np.array(27, dtype=dtype), np.array(27, dtype=np.int32)),
      )

      # Reduce not dimensions
      self._assertOpOutputMatchesExpected(
          functools.partial(reduce, dimensions_to_reduce=()),
          args=(values_1, values_2),
          expected=(values_1, values_2),
      )

  @test_util.disable_mlir_bridge('Not supported yet')
  def testScatter(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge. Scatters two rows of updates into rows 0 and 2 of a 3x3
    # integer matrix (0..8) using an addition combiner and serialized scatter dimension numbers.
    # Expects the updates added to those rows: rows (10,21,32), (3,4,5), (76,87,98).
    test_array = np.arange(9).astype(np.int32).reshape((3, 3))
    scatter_indices = np.array([0, 2], dtype=np.int32)
    updates = np.array([[10, 20, 30], [70, 80, 90]], dtype=np.int32)

    dnums = xla_data_pb2.ScatterDimensionNumbers()
    dnums.update_window_dims.append(1)
    dnums.inserted_window_dims.append(0)
    dnums.scatter_dims_to_operand_dims.append(0)
    dnums.index_vector_dim = 1

    add_numbers = function.Defun(np.int32, np.int32)(lambda x, y: x + y)

    def test_fn(
        scatter_input,
        scatter_indices,
        scatter_updates,
    ):
      # [FUNCTION DOC]
      # Calls the raw scatter op with the addition combiner and serialized dimension numbers,
      # indices not assumed sorted. Returns the scattered tensor.
      return gen_xla_ops.xla_scatter(
          scatter_input,
          scatter_indices,
          scatter_updates,
          add_numbers,
          dnums.SerializeToString(),
          indices_are_sorted=False,
      )

    expected = np.array([[10, 21, 32], [3, 4, 5], [76, 87, 98]], dtype=np.int32)
    self._assertOpOutputMatchesExpected(
        test_fn,
        args=(test_array, scatter_indices, updates),
        expected=expected,
    )

  def testSelectAndScatter(self):
    # [FUNCTION DOC]
    # For bfloat16 and float32 (when numeric): select-and-scatter on a 4x5 operand with 2x3
    # windows and strides (2, 2), selecting the greater-or-equal element in each window and adding
    # the corresponding source value at that position. Expects zeros except the selected
    # maxima positions holding the source values 8, 3 and 1 (second row/third column, etc.).
    for dtype in set(self.numeric_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])
    ):

      @function.Defun(dtype, dtype)  # pylint: disable=cell-var-from-loop
      def add_scatter(x, y):
        # [FUNCTION DOC]
        # Scatter combiner: returns the sum of its two arguments.
        return x + y

      @function.Defun(dtype, dtype)  # pylint: disable=cell-var-from-loop
      def ge_select(x, y):
        # [FUNCTION DOC]
        # Select function: returns whether the first argument is greater than or equal to the
        # second.
        return x >= y

      def test_fn(operand, source):
        # [FUNCTION DOC]
        # Calls the XLA select-and-scatter wrapper with the fixed windows, strides, zero padding,
        # initial value 0 and the select and scatter functions. Returns the result.
        return xla.select_and_scatter(
            operand,
            window_dimensions=[2, 3, 1, 1],
            window_strides=[2, 2, 1, 1],
            padding=[[0, 0]] * 4,
            source=source,
            init_value=0,
            select=ge_select,  # pylint: disable=cell-var-from-loop
            scatter=add_scatter,  # pylint: disable=cell-var-from-loop
        )

      self._assertOpOutputMatchesExpected(
          test_fn,
          args=(
              np.array(
                  [
                      [7, 2, 5, 3, 8],
                      [3, 8, 9, 3, 4],
                      [1, 5, 7, 5, 6],
                      [0, 6, 2, 10, 2],
                  ],
                  dtype=dtype,
              ).reshape((4, 5, 1, 1)),
              np.array([[2, 6], [3, 1]], dtype=dtype).reshape((2, 2, 1, 1)),
          ),
          expected=np.array(
              [
                  [0, 0, 0, 0, 0],
                  [0, 0, 8, 0, 0],
                  [0, 0, 3, 0, 0],
                  [0, 0, 0, 1, 0],
              ],
              dtype=dtype,
          ).reshape((4, 5, 1, 1)),
      )

  def testTranspose(self):
    # [FUNCTION DOC]
    # For each numeric type: transposes a 2x2 matrix with permutation (1, 0). Expects the
    # matrix transpose.
    for dtype in self.numeric_types:
      v = np.arange(4, dtype=np.int32).astype(dtype).reshape([2, 2])
      self._assertOpOutputMatchesExpected(
          lambda x: xla.transpose(x, [1, 0]), args=(v,), expected=v.T
      )

  def testDynamicSlice(self):
    # [FUNCTION DOC]
    # For each numeric type: slices a 10x10x10 array of 0..999 starting at (5, 7, 3) with
    # sizes (2, 3, 2). Expects the 2x3x2 block of values such as 573, 574, 583, ... 694.
    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          xla.dynamic_slice,
          args=(
              np.arange(1000, dtype=np.int32)
              .astype(dtype)
              .reshape([10, 10, 10]),
              np.array([5, 7, 3]),
              np.array([2, 3, 2]),
          ),
          expected=np.array(
              np.array([
                  [[573, 574], [583, 584], [593, 594]],
                  [[673, 674], [683, 684], [693, 694]],
              ]),
              dtype=dtype,
          ),
      )

  def testDynamicSliceWithIncorrectStartIndicesShape(self):
    # [FUNCTION DOC]
    # Dynamic slice of a rank-3 array with only two start indices but three slice sizes. Running
    # the session is expected to raise an invalid-argument error whose message reports a mismatch
    # between 3 slice sizes and 2 start indices.
    with self.session() as session:
      with self.test_scope():
        output = xla.dynamic_slice(
            np.arange(1000, dtype=np.int32).reshape([10, 10, 10]),
            np.array([5, 7]),
            np.array([2, 3, 4]),
        )
      with self.assertRaises(errors.InvalidArgumentError) as invalid_arg_error:
        session.run(output)
      self.assertRegex(
          invalid_arg_error.exception.message,
          (
              r'has mismatched number of slice sizes \(3\) and number of start'
              r' indices \(2\)'
          ),
      )

  def testDynamicSliceWithIncorrectSizeIndicesShape(self):
    # [FUNCTION DOC]
    # Dynamic slice of a rank-3 array with three start indices but only two slice sizes. Running
    # the session is expected to raise an invalid-argument error whose message reports a mismatch
    # between 2 slice sizes and 3 start indices.
    with self.session() as session:
      with self.test_scope():
        output = xla.dynamic_slice(
            np.arange(1000, dtype=np.int32).reshape([10, 10, 10]),
            np.array([5, 7, 3]),
            np.array([2, 3]),
        )
      with self.assertRaises(errors.InvalidArgumentError) as invalid_arg_error:
        session.run(output)
      self.assertRegex(
          invalid_arg_error.exception.message,
          (
              r'has mismatched number of slice sizes \(2\) and number of start'
              r' indices \(3\)'
          ),
      )

  def test_optimization_barrier(self):
    # [FUNCTION DOC]
    # Passes a float32 and an integer array through the optimization barrier wrapper. Expects the
    # outputs to equal the inputs unchanged.
    args = (
        np.array([[5, 6, 7]], dtype=np.float32),
        np.array([[1, 2, 3]], dtype=int),
    )

    self._assertOpOutputMatchesExpected(
        xla.optimization_barrier, args=args, expected=args
    )

  def test_reduce_precision(self):
    # [FUNCTION DOC]
    # Reduces float32 values to 4 exponent bits and 2 mantissa bits: expects the extra mantissa bit
    # rounded away, 128 kept and 256 overflowing to infinity. Then passes bit counts of 2 to the
    # 33rd power (too big for 32-bit integers) for a value 4 and expects it unchanged, checking that
    # large counts are accepted. Uses exact equality.
    arg = np.array([1 + 2**-2 + 2**-4, 128, 256], dtype=np.float32)
    expected = np.array([1 + 2**-2, 128, float('Inf')], dtype=np.float32)
    exponent_bits = 4
    mantissa_bits = 2
    self._assertOpOutputMatchesExpected(
        lambda x: xla.reduce_precision(x, exponent_bits, mantissa_bits),
        args=(arg,),
        expected=expected,
        equality_fn=self.assertAllEqual,
    )

    arg = np.array([4], dtype=np.float32)
    expected = np.array([4], dtype=np.float32)
    # Test passing numbers that cannot fit in a 32-bit integer.
    exponent_bits = 2**33
    mantissa_bits = 2**33
    self._assertOpOutputMatchesExpected(
        lambda x: xla.reduce_precision(x, exponent_bits, mantissa_bits),
        args=(arg,),
        expected=expected,
        equality_fn=self.assertAllEqual,
    )


class XlaOpsShapeInferenceTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests static shape inference and argument validation of XLA op wrappers (dot, dynamic slice,
  # dynamic update slice, pad, variadic reduce, random bit generator, gather). No session is
  # run; tests inspect shapes of symbolic tensors or expect errors at graph-construction time.

  def testDotShapeInference(self):
    # [FUNCTION DOC]
    # Dot of shapes (1,2,3,4) and (4,5,2,6) with lhs batch dim 3 and rhs batch dim 0, contracting
    # lhs dim 1 with rhs dim 2. Expects the result shape (4, 1, 3, 5, 6).
    a = array_ops.placeholder(np.float32, shape=(1, 2, 3, 4))
    b = array_ops.placeholder(np.float32, shape=(4, 5, 2, 6))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(1)
    dim_nums.rhs_contracting_dimensions.append(2)
    dim_nums.lhs_batch_dimensions.append(3)
    dim_nums.rhs_batch_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape, tensor_shape.TensorShape([4, 1, 3, 5, 6]))

  def testDotDifferentNumberOfContractingDimensions(self):
    # [FUNCTION DOC]
    # Dot with one lhs contracting dim but two rhs contracting dims. Expects a value error saying
    # the number of contracting dimensions must match (got 1 and 2).
    a = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(3)

    with self.assertRaisesRegex(
        ValueError,
        'Must specify the same number of contracting '
        'dimensions for lhs and rhs. Got: 1 and 2',
    ):
      xla.dot_general(a, b, dim_nums)

  def testDotDifferentContractingDimensionsSizes(self):
    # [FUNCTION DOC]
    # Dot whose contracting dimensions have sizes 2 and 4. Expects a value error that dimensions
    # must be equal but are 2 and 4.
    a = array_ops.placeholder(np.float32, shape=(2, 2, 2, 2))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(3)

    with self.assertRaisesRegex(
        ValueError, 'Dimensions must be equal, but are 2 and 4'
    ):
      xla.dot_general(a, b, dim_nums)

  def testDotDifferentNumberOfBatchDimensions(self):
    # [FUNCTION DOC]
    # Dot with one lhs batch dim but two rhs batch dims. Expects a value error saying the number of
    # batch dimensions must match (got 1 and 2).
    a = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_batch_dimensions.append(2)
    dim_nums.rhs_batch_dimensions.append(2)
    dim_nums.rhs_batch_dimensions.append(3)

    with self.assertRaisesRegex(
        ValueError,
        'Must specify the same number of batch '
        'dimensions for lhs and rhs. Got: 1 and 2',
    ):
      xla.dot_general(a, b, dim_nums)

  def testDotDifferentBatchDimensionsSizes(self):
    # [FUNCTION DOC]
    # Dot whose batch dimensions have sizes 2 and 4. Expects a value error that dimensions must be
    # equal but are 2 and 4.
    a = array_ops.placeholder(np.float32, shape=(2, 2, 2, 2))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 2))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(3)
    dim_nums.lhs_batch_dimensions.append(0)
    dim_nums.rhs_batch_dimensions.append(0)

    with self.assertRaisesRegex(
        ValueError, 'Dimensions must be equal, but are 2 and 4'
    ):
      xla.dot_general(a, b, dim_nums)

  def testDotUnknownNonContractingDimension(self):
    # [FUNCTION DOC]
    # Matrix product of (unknown, 16) by (16, 2). Expects result shape (unknown, 2).
    a = array_ops.placeholder(np.float32, shape=(None, 16))
    b = array_ops.placeholder(np.float32, shape=(16, 2))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(1)
    dim_nums.rhs_contracting_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape.as_list(), [None, 2])

  def testDotUnknownContractingDimension(self):
    # [FUNCTION DOC]
    # Matrix product of (3, unknown) by (unknown, 2), both contracting dims unknown. Expects
    # result shape (3, 2).
    a = array_ops.placeholder(np.float32, shape=(3, None))
    b = array_ops.placeholder(np.float32, shape=(None, 2))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(1)
    dim_nums.rhs_contracting_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape.as_list(), [3, 2])

  def testDotUnknownAndKnownContractingDimension(self):
    # [FUNCTION DOC]
    # Matrix product of (3, 4) by (unknown, 2): one contracting dim known, the other unknown.
    # Expects result shape (3, 2).
    a = array_ops.placeholder(np.float32, shape=(3, 4))
    b = array_ops.placeholder(np.float32, shape=(None, 2))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(1)
    dim_nums.rhs_contracting_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape.as_list(), [3, 2])

  def testDotUnknownBatchDimension(self):
    # [FUNCTION DOC]
    # Batched dot of (unknown, 3, 4) by (unknown, 4) with both batch dims unknown. Expects result
    # shape (unknown, 3).
    a = array_ops.placeholder(np.float32, shape=(None, 3, 4))
    b = array_ops.placeholder(np.float32, shape=(None, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(1)
    dim_nums.lhs_batch_dimensions.append(0)
    dim_nums.rhs_batch_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape.as_list(), [None, 3])

  def testDotUnknownAndKnownBatchDimension(self):
    # [FUNCTION DOC]
    # Batched dot of (2, 3, 4) by (unknown, 4): one batch dim known (2), other unknown. Expects
    # result shape (2, 3), taking the known size.
    a = array_ops.placeholder(np.float32, shape=(2, 3, 4))
    b = array_ops.placeholder(np.float32, shape=(None, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(1)
    dim_nums.lhs_batch_dimensions.append(0)
    dim_nums.rhs_batch_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape.as_list(), [2, 3])

  def testDynamicSlice(self):
    # [FUNCTION DOC]
    # Checks inferred output shape of dynamic slice for various slice-size inputs and operand
    # shapes (fully known, partly unknown, unknown rank): constant sizes give exactly those sizes;
    # sizes with one unknown element give unknown only in that position; a size vector of known
    # length but non-constant, or of unknown length (with known operand rank), gives all-unknown
    # dims of the right rank; with both unknown the rank is unknown.
    start = array_ops.placeholder(np.int32, shape=(2, 3, 4))

    # If slice_sizes are known, the operand shape does not matter.
    # The shape of the output is equal to slice_sizes.
    slice_sizes = np.array([1, 2, 4], dtype=np.int32)
    for a_shape in [(2, 3, 4), (None, 3, 4), None]:
      a = array_ops.placeholder(np.float32, shape=a_shape)
      res = xla.dynamic_slice(a, start, slice_sizes)
      self.assertEqual(res.shape.as_list(), [1, 2, 4])

    # The first two dimension slice sizes are known
    slice_sizes = array_ops_stack.stack(
        [1, 2, array_ops.placeholder(np.int32, [])]
    )
    for a_shape in [(2, 3, 4), (None, 3, 4), None]:
      a = array_ops.placeholder(np.float32, shape=a_shape)
      res = xla.dynamic_slice(a, start, slice_sizes)
      self.assertEqual(res.shape.as_list(), [1, 2, None])

    # If slice_sizes has known rank and dimension, but is not a constant
    # then output has the same rank, but with unknown dimensions.
    slice_sizes = array_ops.placeholder(np.int32, [3])
    for a_shape in [(2, 3, 4), (None, 3, 4), None]:
      a = array_ops.placeholder(np.float32, shape=a_shape)
      res = xla.dynamic_slice(a, start, slice_sizes)
      self.assertEqual(res.shape.as_list(), [None, None, None])

    # slice sizes has known rank, but unknown dimensions.
    # then the output has the same rank as the operand, but with unknown
    # dimensions.
    slice_sizes = array_ops.placeholder(np.int32, [None])
    for a_shape in [(2, 3, 4), (None, 3, 4)]:
      a = array_ops.placeholder(np.float32, shape=a_shape)
      res = xla.dynamic_slice(a, start, slice_sizes)
      self.assertEqual(res.shape.as_list(), [None, None, None])

    a = array_ops.placeholder(np.float32, shape=None)
    slice_sizes = array_ops.placeholder(np.int32, [None])
    res = xla.dynamic_slice(a, start, slice_sizes)
    self.assertIsNone(res.shape.rank)

  def testDynamicUpdateSlice(self):
    # [FUNCTION DOC]
    # Dynamic update slice with a (2,3,4) operand, update (1,2,3) and a 3-element start vector.
    # Expects the output shape to equal the operand shape, including unknown dims when the operand
    # has them (unknown, 3, unknown).
    a = array_ops.placeholder(np.float32, shape=(2, 3, 4))
    upd = array_ops.placeholder(np.float32, shape=(1, 2, 3))
    start_indices = array_ops.placeholder(np.int32, shape=(3,))

    res = xla.dynamic_update_slice(a, upd, start_indices)
    self.assertEqual(res.shape.as_list(), [2, 3, 4])

    a = array_ops.placeholder(np.float32, shape=(None, 3, None))
    res = xla.dynamic_update_slice(a, upd, start_indices)
    self.assertEqual(res.shape.as_list(), [None, 3, None])

  def testPadShapeInference(self):
    # [FUNCTION DOC]
    # Checks pad output shapes: (2,3) with the given paddings gives (6,14); negative padding gives
    # (6,3); an unknown first dim stays unknown and the second becomes 9; a zero-sized dimension
    # with interior padding gives (6,2). Also expects value errors for: non-scalar padding value,
    # wrong lengths of the low, high and interior padding lists, negative interior padding, and
    # a padding amount that makes a dimension negative.
    a = array_ops.placeholder(np.float32, shape=(2, 3))

    c = xla.pad(
        a,
        padding_value=7,
        padding_low=[2, 1],
        padding_high=[1, 2],
        padding_interior=[1, 4],
    )

    self.assertEqual(c.shape, tensor_shape.TensorShape([6, 14]))

    c = xla.pad(
        a,
        padding_value=7,
        padding_low=[2, -2],
        padding_high=[1, -2],
        padding_interior=[1, 2],
    )

    self.assertEqual(c.shape, tensor_shape.TensorShape([6, 3]))

    c = xla.pad(
        array_ops.placeholder(np.float32, shape=(None, 2)),
        padding_value=7,
        padding_low=[0, 1],
        padding_high=[0, 2],
        padding_interior=[0, 4],
    )
    self.assertEqual(c.shape.as_list(), [None, 9])

    # 0-sized input dimension and interior padding
    c = xla.pad(
        array_ops.placeholder(np.float32, shape=(2, 0)),
        padding_value=7,
        padding_low=[2, 1],
        padding_high=[1, 1],
        padding_interior=[1, 2],
    )

    self.assertEqual(c.shape, tensor_shape.TensorShape([6, 2]))

    with self.assertRaisesRegex(
        ValueError, 'padding_value input must be scalar, found rank 1 '
    ):
      xla.pad(
          a,
          padding_value=[0, 1],
          padding_low=[0, 0],
          padding_high=[0, 0],
          padding_interior=[0, 0],
      )

    with self.assertRaisesRegex(
        ValueError, 'padding_low must be a 1D tensor of size 2 '
    ):
      xla.pad(
          a,
          padding_value=7,
          padding_low=[0, 0, 0],
          padding_high=[0, 0],
          padding_interior=[0, 0],
      )

    with self.assertRaisesRegex(
        ValueError, 'padding_high must be a 1D tensor of size 2 '
    ):
      xla.pad(
          a,
          padding_value=7,
          padding_low=[0, 0],
          padding_high=[0, 0, 0],
          padding_interior=[0, 0],
      )

    with self.assertRaisesRegex(
        ValueError, 'padding_interior must be a 1D tensor of size 2 '
    ):
      xla.pad(
          a,
          padding_value=7,
          padding_low=[0, 0],
          padding_high=[0, 0],
          padding_interior=[0],
      )

    with self.assertRaisesRegex(
        ValueError,
        'padding_interior must contain only non-negative values, found -2 ',
    ):
      xla.pad(
          a,
          padding_value=7,
          padding_low=[0, 0],
          padding_high=[0, 0],
          padding_interior=[-2, 0],
      )

    with self.assertRaisesRegex(
        ValueError, 'resulting padded dimension has negative size -1 '
    ):
      xla.pad(
          a,
          padding_value=7,
          padding_low=[-3, 0],
          padding_high=[0, 0],
          padding_interior=[0, 0],
      )

  def testVariadicReduceV2SingleArg(self):
    # [FUNCTION DOC]
    # Variadic reduce (v2) with one float32 input of shape (3,4,5), reducing dim 1 with an add
    # reducer. Expects one output of shape (3, 5).
    @def_function.function
    def reducer_add(op_element, acc_val):
      # [FUNCTION DOC]
      # Reducer: returns a one-element tuple with the sum of the element and accumulator.
      return (op_element + acc_val,)

    dtype = np.float32
    arg_spec = array_ops.zeros([], dtype)  # pylint: disable=cell-var-from-loop
    reducer_func = reducer_add.get_concrete_function(arg_spec, arg_spec)

    res = xla.variadic_reduce(
        (array_ops.placeholder(np.float32, shape=(3, 4, 5)),),
        (array_ops.placeholder(np.float32, shape=()),),
        dimensions_to_reduce=(1,),
        reducer=reducer_func,
    )
    self.assertLen(res, 1)
    self.assertEqual(res[0].shape, tensor_shape.TensorShape([3, 5]))

  def testVariadicReduceV2MultipleArgs(self):
    # [FUNCTION DOC]
    # Variadic reduce (v2) with three inputs (one float32, two int32) and an add reducer.
    # Checks inferred output shapes for known shapes (reducing dim 1 or none), partly unknown
    # shapes (known dims are merged across inputs), and completely unknown shapes (unknown rank
    # outputs). Also expects value errors saying all inputs must have the same shape when known
    # dims conflict.
    @def_function.function
    def reducer_adds(
        op_element_1,
        op_element_2,
        op_element_3,
        acc_val_1,
        acc_val_2,
        acc_val_3,
    ):
      # [FUNCTION DOC]
      # Reducer: returns a triple of element-plus-accumulator sums, one per input.
      return (
          op_element_1 + acc_val_1,
          op_element_2 + acc_val_2,
          op_element_3 + acc_val_3,
      )

    dtype = np.float32
    arg1_spec = array_ops.zeros([], dtype)  # pylint: disable=cell-var-from-loop
    arg2_spec = array_ops.zeros([], np.int32)
    arg3_spec = array_ops.zeros([], np.int32)
    reducer_func = reducer_adds.get_concrete_function(
        arg1_spec, arg2_spec, arg3_spec, arg1_spec, arg2_spec, arg3_spec
    )

    def reduce_with_shapes(shape1, shape2, shape3, dimensions_to_reduce=(1,)):
      # [FUNCTION DOC]
      # HELPER: builds placeholders of the three given shapes (possibly partly or fully unknown)
      # and scalar initial-value placeholders, then calls the v2 variadic reduce with the
      # enclosing reducer. dimensions_to_reduce defaults to dim 1. Returns the outputs.
      inputs = (
          array_ops.placeholder(np.float32, shape=shape1),
          array_ops.placeholder(np.int32, shape=shape2),
          array_ops.placeholder(np.int32, shape=shape3),
      )
      init_values = (
          array_ops.placeholder(np.float32, shape=()),
          array_ops.placeholder(np.int32, shape=()),
          array_ops.placeholder(np.int32, shape=()),
      )

      return xla.variadic_reduce(
          inputs,
          init_values,
          dimensions_to_reduce=dimensions_to_reduce,
          reducer=reducer_func,
      )

    def assert_output_shapes(output, expected_shape):
      # [FUNCTION DOC]
      # HELPER: asserts there are three outputs, each with the expected shape (list form).
      self.assertLen(output, 3)
      self.assertEqual(output[0].shape.as_list(), list(expected_shape))
      self.assertEqual(output[1].shape.as_list(), list(expected_shape))
      self.assertEqual(output[2].shape.as_list(), list(expected_shape))

    output = reduce_with_shapes((3, 4, 5), (3, 4, 5), (3, 4, 5))
    assert_output_shapes(output, (3, 5))

    output = reduce_with_shapes(
        (3, 4, 5), (3, 4, 5), (3, 4, 5), dimensions_to_reduce=()
    )
    assert_output_shapes(output, (3, 4, 5))

    output = reduce_with_shapes(None, (3, None, 5), (None, 4, 5))
    assert_output_shapes(output, (3, 5))

    output = reduce_with_shapes(None, (3, None, 5), None)
    assert_output_shapes(output, (3, 5))

    output = reduce_with_shapes(None, (None, None, 5), None)
    assert_output_shapes(output, (None, 5))

    output = reduce_with_shapes(None, None, None)
    self.assertLen(output, 3)
    self.assertIsNone(output[0].shape.rank)
    self.assertIsNone(output[1].shape.rank)
    self.assertIsNone(output[2].shape.rank)

    with self.assertRaisesRegex(
        ValueError, 'All inputs must have the same shape'
    ):
      reduce_with_shapes((3, 4, 5), (13, 4, 5), (3, 4, 5))

    with self.assertRaisesRegex(
        ValueError, 'All inputs must have the same shape'
    ):
      reduce_with_shapes((None, 4, 5), (3, None, 5), (13, 4, 5))

    with self.assertRaisesRegex(
        ValueError, 'All inputs must have the same shape'
    ):
      reduce_with_shapes((None, 4, 5), (3, None, 5), (13, 4, 5))

  @parameterized.product(
      algorithm=[random_ops_util.Algorithm.THREEFRY,
                 random_ops_util.Algorithm.PHILOX,
                 random_ops_util.Algorithm.AUTO_SELECT],
      dtype=[np.uint8, np.uint64],
  )
  def testRngBitGenerator(self, algorithm, dtype):
    # [FUNCTION DOC]
    # Parameterized over three algorithms and two output types (8-bit and 64-bit unsigned).
    # Checks the shape of the two outputs (new state, bits) for a known-shape state, a state of
    # unknown length, and a state of unknown rank (state shape then reported as one unknown dim);
    # output shape (2,3) is kept. With an unknown output dimension, expects a type error about
    # failing to convert elements to a tensor.
    initial_state = array_ops.placeholder(np.uint64, shape=(2,))
    shape = (2, 3)
    res = xla.rng_bit_generator(algorithm, initial_state, shape, dtype=dtype)

    self.assertEqual(res[0].shape, initial_state.shape)
    self.assertEqual(res[1].shape, shape)

    # The initial_state has unknown dimension size
    initial_state = array_ops.placeholder(np.uint64, shape=(None,))
    shape = (2, 3)
    res = xla.rng_bit_generator(algorithm, initial_state, shape, dtype=dtype)

    self.assertEqual(res[0].shape.as_list(), initial_state.shape.as_list())
    self.assertEqual(res[1].shape, shape)

    # The initial_state has unknown rank
    initial_state = array_ops.placeholder(np.uint64, shape=None)
    shape = (2, 3)
    res = xla.rng_bit_generator(algorithm, initial_state, shape, dtype=dtype)

    self.assertEqual(res[0].shape.as_list(), [None])
    self.assertEqual(res[1].shape, shape)

    # The output shape has unknown dimension
    initial_state = array_ops.placeholder(np.uint64, shape=(None,))
    shape = (None, 3)
    with self.assertRaisesRegex(
        TypeError, 'Failed to convert elements .* to Tensor'
    ):
      _ = xla.rng_bit_generator(algorithm, initial_state, shape, dtype=dtype)

  def testGatherShapeInference(self):
    # [FUNCTION DOC]
    # Gather on a constant 2x5 operand with index 2, slice sizes (1, 3) and fully specified
    # dimension numbers. Expects output shape (1, 3).
    operand = np.arange(10, dtype=np.int32).reshape([2, 5])
    start_indices = np.array([2], np.int32)
    slice_sizes = np.array([1, 3], np.int32)
    dimension_numbers = xla_data_pb2.GatherDimensionNumbers(
        offset_dims=[1],
        collapsed_slice_dims=[0],
        start_index_map=[0],
        index_vector_dim=1,
    )

    res = xla.gather(operand, start_indices, dimension_numbers, slice_sizes)
    self.assertEqual(res.shape, tensor_shape.TensorShape([1, 3]))

  def testGatherShapeInferenceDynamicSlice(self):
    # [FUNCTION DOC]
    # Gather on a constant 3x2x2 operand with indices of shape (3, unknown, 2), slice sizes
    # (1, 2, 2). Expects output shape (3, unknown, 2, 2).
    operand = np.arange(12, dtype=np.int32).reshape([3, 2, 2])
    start_indices = array_ops.placeholder(np.int32, shape=(3, None, 2))
    slice_sizes = np.array([1, 2, 2], np.int32)
    dimension_numbers = xla_data_pb2.GatherDimensionNumbers(
        offset_dims=[2, 3],
        collapsed_slice_dims=[0],
        start_index_map=[0, 1],
        index_vector_dim=2,
    )

    res = xla.gather(operand, start_indices, dimension_numbers, slice_sizes)
    self.assertEqual(res.shape, tensor_shape.TensorShape([3, None, 2, 2]))

  def testGatherShapeInferenceDynamicInput(self):
    # [FUNCTION DOC]
    # Gather with an operand of shape (unknown, 5) and empty dimension numbers. Expects a fully
    # unknown output shape (inference gives up).
    operand = array_ops.placeholder(np.int32, shape=(None, 5))
    start_indices = np.array([2], np.int32)
    slice_sizes = np.array([1, 3], np.int32)
    dimension_numbers = xla_data_pb2.GatherDimensionNumbers()

    res = xla.gather(operand, start_indices, dimension_numbers, slice_sizes)
    self.assertEqual(res.shape, tensor_shape.unknown_shape())

  def testGatherShapeInferenceUnknownSliceSizes(self):
    # [FUNCTION DOC]
    # Gather with slice sizes given by a non-constant tensor and empty dimension numbers. Expects
    # a fully unknown output shape.
    operand = np.arange(10, dtype=np.int32).reshape([2, 5])
    start_indices = np.array([2], np.int32)
    slice_sizes = array_ops.placeholder(np.int32, shape=(2,))
    dimension_numbers = xla_data_pb2.GatherDimensionNumbers()

    res = xla.gather(operand, start_indices, dimension_numbers, slice_sizes)
    self.assertEqual(res.shape, tensor_shape.unknown_shape())


if __name__ == '__main__':
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  googletest.main()
