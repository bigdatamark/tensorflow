# [MODULE DOC]
# Unit tests for the array manipulation operations in the TensorFlow Python array_ops module
# (plus a few generated array ops), run through the standard TensorFlow test framework.
# Covers matrix transpose, boolean masking, shape/squeeze/expand checks, reverse, meshgrid,
# strided slicing (reading, shape inference, gradients, assignment, benchmarks), shape/size/rank,
# sequence masks, identity, padding (mirror pad gradients), invert permutation, unravel index,
# guarantee-const, snapshot, quantize/dequantize, sorted search, batched gather-nd, repeat,
# tiling of variant tensors and stop-gradient.
# Several small checker/benchmark helper classes are defined here and shared by the slicing tests.
# The module defines a module-level list of data types used by the strided slice tests.
# No flags or registrations happen at import time; running the file as a script starts the
# TensorFlow test runner.
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
"""Tests for array_ops."""
import re
import time
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_getitem_override
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.platform import test as test_lib


@test_util.run_all_in_graph_and_eager_modes
class BatchMatrixTransposeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the matrix transpose operation on plain and batched matrices, with optional
  # conjugation and with statically unknown shapes. The decorator runs every test method in both
  # graph mode and eager mode. It holds no fixtures of its own.

  def testNonBatchMatrix(self):
    # [FUNCTION DOC]
    # Transposes a plain 2 by 3 nested-list matrix and asserts the static shape becomes 3 by 2 and
    # the values equal the hand-written transpose. Returns nothing; failures raise assertion errors.
    matrix = [[1, 2, 3], [4, 5, 6]]  # Shape (2, 3)
    expected_transposed = [[1, 4], [2, 5], [3, 6]]  # Shape (3, 2)
    transposed = array_ops.matrix_transpose(matrix)
    self.assertEqual((3, 2), transposed.get_shape())
    self.assertAllEqual(expected_transposed, transposed)

  def testConjugate(self):
    # [FUNCTION DOC]
    # Transposes a 2 by 3 complex matrix with conjugation enabled and asserts the result has shape
    # 3 by 2 and equals the transpose with each imaginary part negated. Returns nothing.
    m = [[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]]
    expected_transposed = [[1 - 1j, 4 - 4j], [2 - 2j, 5 - 5j], [3 - 3j, 6 - 6j]]
    matrix = ops.convert_to_tensor(m)
    transposed = array_ops.matrix_transpose(matrix, conjugate=True)
    self.assertEqual((3, 2), transposed.get_shape())
    self.assertAllEqual(expected_transposed, transposed)

  def testBatchMatrix(self):
    # [FUNCTION DOC]
    # Transposes a batch of two 2 by 3 matrices (rank-3 input) and asserts that each matrix is
    # transposed independently, giving shape 2 by 3 by 2 and the expected values. Returns nothing.
    matrix_0 = [[1, 2, 3], [4, 5, 6]]
    matrix_0_t = [[1, 4], [2, 5], [3, 6]]
    matrix_1 = [[11, 22, 33], [44, 55, 66]]
    matrix_1_t = [[11, 44], [22, 55], [33, 66]]
    batch_matrix = [matrix_0, matrix_1]  # Shape (2, 2, 3)
    expected_transposed = [matrix_0_t, matrix_1_t]  # Shape (2, 3, 2)
    transposed = array_ops.matrix_transpose(batch_matrix)
    self.assertEqual((2, 3, 2), transposed.get_shape())
    self.assertAllEqual(expected_transposed, transposed)

  def testNonBatchMatrixDynamicallyDefined(self):
    # [FUNCTION DOC]
    # Builds a graph function whose input signature is an integer tensor of unknown rank, asserts
    # inside that the static rank is unknown, and transposes it. Calls it with a 2 by 3 constant and
    # asserts the result equals the transposed list. Checks transpose works with dynamic rank.
    # needs explicit `constant` because lists are not automatically
    # converted to sensors when applying `transpose` below
    matrix = constant_op.constant([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
    expected_transposed = [[1, 4], [2, 5], [3, 6]]  # Shape (3, 2)

    @def_function.function(input_signature=[
        tensor_lib.TensorSpec(shape=None, dtype=dtypes.int32)
    ])
    def transpose(matrix):
      # [FUNCTION DOC]
      # Nested graph function (input signature: int32 tensor of unknown shape). Asserts the static rank
      # of its argument is unknown while tracing, then returns the matrix-transposed argument.
      self.assertIs(matrix.shape.ndims, None)
      return array_ops.matrix_transpose(matrix)

    self.assertAllEqual(expected_transposed, transpose(matrix))

  def testBatchMatrixDynamicallyDefined(self):
    # [FUNCTION DOC]
    # Same as the non-batch dynamic test but with a batch of two 2 by 3 matrices: wraps transpose in
    # a graph function taking an int32 tensor of unknown rank, calls it with the batch constant and
    # asserts the result equals the per-matrix transposes. Returns nothing.
    matrix_0 = [[1, 2, 3], [4, 5, 6]]
    matrix_0_t = [[1, 4], [2, 5], [3, 6]]
    matrix_1 = [[11, 22, 33], [44, 55, 66]]
    matrix_1_t = [[11, 44], [22, 55], [33, 66]]
    # needs explicit `constant` because lists are not automatically
    # converted to sensors when applying `transpose` below
    batch_matrix = constant_op.constant([matrix_0, matrix_1])  # Shape (2, 2, 3)
    expected_transposed = [matrix_0_t, matrix_1_t]  # Shape (2, 3, 2)

    @def_function.function(input_signature=[
        tensor_lib.TensorSpec(shape=None, dtype=dtypes.int32)
    ])
    def transpose(matrix):
      # [FUNCTION DOC]
      # Nested graph function (input signature: int32 tensor of unknown shape). Asserts the static rank
      # of its argument is unknown while tracing, then returns the matrix-transposed argument.
      self.assertIs(matrix.shape.ndims, None)
      return array_ops.matrix_transpose(matrix)

    self.assertAllEqual(expected_transposed, transpose(batch_matrix))

  def testTensorWithStaticRankLessThanTwoRaisesBecauseNotAMatrix(self):
    # [FUNCTION DOC]
    # Passes a rank-1 list to matrix transpose and asserts a value error whose message contains
    # "should be a ", since a vector is not a matrix. Returns nothing.
    vector = [1, 2, 3]
    with self.assertRaisesRegex(ValueError, "should be a "):
      array_ops.matrix_transpose(vector)

  def testNarrowMatrixConjugateTranspose(self):
    # [FUNCTION DOC]
    # For float32 and float64 component types and both conjugate settings (each a sub-test), builds a
    # complex vector of length 96 with zero real part and increasing imaginary part, tiles it into a
    # 96 by 2 narrow matrix, transposes it (with or without conjugation) and asserts the shape is
    # 2 by 96 and the values equal the tiled row vector (negated when conjugating). Targets the
    # narrow-matrix code path.
    for dtype in (dtypes.float32, dtypes.float64):
      for conjugate in (True, False):
        with self.subTest(complex_type=dtype, conjugate=conjugate):
          vector = math_ops.complex(
              constant_op.constant(0, dtype=dtype),
              math_ops.range(96, dtype=dtype))
          column_vector = array_ops.expand_dims(vector, axis=-1)
          row_vector = array_ops.expand_dims(vector, axis=0)
          narrow_matrix = array_ops.tile(column_vector, [1, 2])  # [96, 2]
          expected_transposed = array_ops.tile(row_vector, [2, 1])  # [2, 96]
          if conjugate:
            expected_transposed = -expected_transposed

          transposed = array_ops.matrix_transpose(
              narrow_matrix, conjugate=conjugate)

          self.assertEqual((2, 96), transposed.get_shape())
          self.assertAllEqual(expected_transposed, transposed)


class BooleanMaskTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the boolean mask operation, compared against numpy boolean indexing and for its
  # error handling. Fixture: setUp creates a seeded random number generator (seed 42) used to build
  # random masks. Not wrapped in the both-modes decorator.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp and stores a numpy random state seeded with 42 as the test's random
    # number generator, so generated masks are reproducible. Returns nothing.
    super().setUp()
    self.rng = np.random.RandomState(42)

  def CheckVersusNumpy(self, ndims_mask, arr_shape, make_mask=None, axis=None):
    # [FUNCTION DOC]
    # HELPER: shared checker for the boolean mask tests in this class.
    # Compares boolean mask against numpy masking. Inputs: ndims_mask (number of leading dimensions
    # the mask covers), arr_shape (shape of a random array), make_mask (optional callable producing
    # a boolean mask for a shape; defaults to random booleans from the test generator) and axis
    # (optional starting axis for the mask). Builds a random array and mask, computes the numpy result
    # for axis none/0/1/2, runs the mask op, asserts the dimensions after the leading masked one have
    # the same static shape and that values are close. Returns nothing; raises assertion errors.
    """Check equivalence between boolean_mask and numpy masking."""
    if make_mask is None:
      make_mask = lambda shape: self.rng.randint(0, 2, size=shape).astype(bool)
    arr = np.random.rand(*arr_shape)
    mask = make_mask(arr_shape[:ndims_mask])
    if axis is not None:
      mask = make_mask(arr_shape[axis:ndims_mask + axis])
    if axis is None or axis == 0:
      masked_arr = arr[mask]
    elif axis == 1:
      masked_arr = arr[:, mask]
    elif axis == 2:
      masked_arr = arr[:, :, mask]
    masked_tensor = array_ops.boolean_mask(arr, mask, axis=axis)

    # Leading dimension size of masked_tensor is always unknown until runtime
    # since we don't how many elements will be kept.
    leading = 1 if axis is None else axis + 1
    self.assertAllEqual(masked_tensor.get_shape()[leading:],
                        masked_arr.shape[leading:])

    self.assertAllClose(masked_arr, masked_tensor)

  def testMaskDim1ArrDim2Axis1(self):
    # [FUNCTION DOC]
    # One-dimensional mask applied along axis 1 of arrays with shapes 1x1, 2x2 and 2x5 (one sub-test
    # each); each case is checked against numpy through the shared helper.
    ndims_mask = 1
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape, axis=1)

  def testMaskDim2ArrDim2Axis1(self):
    # [FUNCTION DOC]
    # Two-dimensional mask on two-dimensional arrays (shapes 1x1, 2x2, 2x5) with axis 1, checked
    # against numpy through the shared helper, one sub-test per shape.
    ndims_mask = 2
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape, axis=1)

  def testMaskDim1ArrDim1(self):
    # [FUNCTION DOC]
    # One-dimensional mask on one-dimensional arrays of sizes 1, 2, 3 and 10 with default axis,
    # checked against numpy through the shared helper, one sub-test per shape.
    ndims_mask = 1
    for arr_shape in [(1,), (2,), (3,), (10,)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testMaskDim1ArrDim2(self):
    # [FUNCTION DOC]
    # One-dimensional mask on two-dimensional arrays (1x1, 2x2, 2x5), default axis, checked against
    # numpy through the shared helper, one sub-test per shape.
    ndims_mask = 1
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testMaskDim2ArrDim2(self):
    # [FUNCTION DOC]
    # Two-dimensional mask on two-dimensional arrays (1x1, 2x2, 2x5), default axis, checked against
    # numpy through the shared helper, one sub-test per shape.
    ndims_mask = 2
    for arr_shape in [(1, 1), (2, 2), (2, 5)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testMaskDim2ArrDim3(self):
    # [FUNCTION DOC]
    # Two-dimensional mask on three-dimensional arrays (1x1x1, 1x2x2, 2x2x1), default axis, checked
    # against numpy through the shared helper, one sub-test per shape.
    ndims_mask = 2
    for arr_shape in [(1, 1, 1), (1, 2, 2), (2, 2, 1)]:
      with self.subTest(arr_shape=arr_shape):
        self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testEmptyInput2D(self):
    # [FUNCTION DOC]
    # Applies boolean mask to an empty 2 by 0 float array with a two-element mask. Asserts the static
    # shape after the first dimension equals numpy's and that values are close inside a cached session.
    mask = np.array([True, False])
    arr = np.array([[], []]).astype(np.float32)
    numpy_result = arr[mask]
    tf_result = array_ops.boolean_mask(arr, mask)
    self.assertAllEqual(numpy_result.shape[1:], tf_result.get_shape()[1:])
    with self.cached_session():
      self.assertAllClose(numpy_result, tf_result)

  def testEmptyInput1D(self):
    # [FUNCTION DOC]
    # Applies boolean mask to an empty float vector with an empty boolean mask; asserts the trailing
    # static shape matches numpy and the (empty) values are close inside a cached session.
    mask = np.array([]).astype(bool)
    arr = np.array([]).astype(np.float32)
    numpy_result = arr[mask]
    tf_result = array_ops.boolean_mask(arr, mask)
    self.assertAllEqual(numpy_result.shape[1:], tf_result.get_shape()[1:])
    with self.cached_session():
      self.assertAllClose(numpy_result, tf_result)

  def testEmptyOutput(self):
    # [FUNCTION DOC]
    # Uses an all-false mask generator and loops over mask ranks 1..3 and array ranks from the mask
    # rank up to mask rank plus 2, three random array shapes each (sub-test per combination). Asserts
    # via the shared helper that selecting nothing matches numpy's empty result.
    make_mask = lambda shape: np.zeros(shape, dtype=bool)
    for ndims_mask in range(1, 4):
      for ndims_arr in range(ndims_mask, ndims_mask + 3):
        for _ in range(3):
          with self.subTest(ndims_mask=ndims_mask, ndims_arr=ndims_arr, _=_):
            arr_shape = np.random.randint(1, 5, size=ndims_arr)
            self.CheckVersusNumpy(ndims_mask, arr_shape, make_mask=make_mask)

  def testWorksWithDimensionsEqualToNoneDuringGraphBuild(self):
    # [FUNCTION DOC]
    # Builds a concrete graph function for boolean mask whose tensor has unknown rank and whose mask
    # has rank 1 but unknown length (the mask rank must be known). Calls it with a 2 by 2 array and a
    # two-element mask and asserts the result equals numpy indexing. Checks it works with unknown
    # dimensions at graph build time.
    # The rank of the mask tensor must be specified. This is explained
    # in the docstring as well.
    @def_function.function
    def func(ph_tensor, ph_mask):
      # [FUNCTION DOC]
      # Nested graph function with two arguments (tensor and mask); returns the boolean-masked tensor.
      return array_ops.boolean_mask(ph_tensor, ph_mask)

    f = func.get_concrete_function(
        tensor_lib.TensorSpec(None, dtypes.int32),
        tensor_lib.TensorSpec([None], dtypes.bool))
    arr = np.array([[1, 2], [3, 4]], np.int32)
    mask = np.array([False, True])
    masked_tensor = f(arr, mask)
    self.assertAllEqual(masked_tensor, arr[mask])

  def testMaskDimensionsSetToNoneRaises(self):
    # [FUNCTION DOC]
    # Tries to build a concrete function whose mask has completely unknown rank and asserts a value
    # error whose message contains "dimensions must be specified". Returns nothing.
    # The rank of the mask tensor must be specified. This is explained
    # in the docstring as well.
    @def_function.function
    def func(tensor, mask):
      # [FUNCTION DOC]
      # Nested graph function with two arguments (tensor and mask); returns the boolean-masked tensor.
      # It is only traced, expected to raise during tracing.
      return array_ops.boolean_mask(tensor, mask)

    with self.assertRaisesRegex(ValueError, "dimensions must be specified"):
      _ = func.get_concrete_function(
          tensor_lib.TensorSpec([None, 2], dtypes.int32),
          tensor_lib.TensorSpec(None, dtypes.bool))

  def testMaskHasMoreDimsThanTensorRaises(self):
    # [FUNCTION DOC]
    # Uses a 2-D mask on a 1-D tensor inside a cached session and asserts a value error mentioning
    # "incompatible" when the masking op is built and evaluated.
    mask = [[True, True], [False, False]]
    tensor = [1, 2, 3, 4]
    with self.cached_session():
      with self.assertRaisesRegex(ValueError, "incompatible"):
        self.evaluate(array_ops.boolean_mask(tensor, mask))

  def testMaskIsScalarRaises(self):
    # [FUNCTION DOC]
    # Uses a scalar boolean mask and a scalar tensor in a cached session; asserts a value error whose
    # message mentions the mask being a scalar.
    mask = True
    tensor = 1
    with self.cached_session():
      with self.assertRaisesRegex(ValueError, "mask.*scalar"):
        self.evaluate(array_ops.boolean_mask(tensor, mask))

  def testMaskShapeDifferentThanFirstPartOfTensorShapeRaises(self):
    # [FUNCTION DOC]
    # Uses a length-3 mask on a 2 by 2 tensor (mismatch with the leading dimension) in a cached
    # session; asserts a value error mentioning "incompatible".
    mask = [True, True, True]
    tensor = [[1, 2], [3, 4]]
    with self.cached_session():
      with self.assertRaisesRegex(ValueError, "incompatible"):
        self.evaluate(array_ops.boolean_mask(tensor, mask))

  def testStringMask(self):
    # [FUNCTION DOC]
    # Regression test for a placement bug (optimized boolean mask graph incorrectly put on GPU).
    # Side effect: enables the shape optimization experimental optimizer option globally and does not
    # restore it. Defines a graph function that tiles a string tensor and a boolean tensor by a given
    # multiples vector and masks them; calls it with 2 by 2 and asserts four "hello" byte strings.
    # Reproduces b/111171330, where the optimized boolean_mask graph would
    # be incorrectly placed on GPU.
    config.set_optimizer_experimental_options({"shape_optimization": True})

    @def_function.function
    def func(tile_input):
      # [FUNCTION DOC]
      # Nested graph function taking the tile multiples tensor: tiles a 1x1 string tensor and a 1x1
      # true tensor by it, then returns the boolean-masked string tensor.
      string_tensor = array_ops.tile([["hello"]], tile_input)
      bool_tensor = array_ops.tile([[True]], tile_input)
      masked_tensor = array_ops.boolean_mask(string_tensor, bool_tensor)
      return masked_tensor

    result = func([2, 2])
    self.assertAllEqual([b"hello", b"hello", b"hello", b"hello"], result)

  def testMaskWithAxisTensor(self):
    # [FUNCTION DOC]
    # Defines a graph function (autograph off) that masks a 3-element list with a boolean list using
    # the axis given as a constant tensor, runs it and asserts the result is the first and third
    # elements. Checks that a tensor-valued constant axis is accepted.

    @def_function.function(autograph=False)
    def f():
      # [FUNCTION DOC]
      # Nested graph function with no arguments (autograph disabled); returns the boolean mask of a
      # fixed list with a constant int32 tensor as the axis.
      return array_ops.boolean_mask([1, 2, 3], [True, False, True],
                                    axis=constant_op.constant(
                                        0, dtype=dtypes.int32))

    self.assertAllEqual(self.evaluate(f()), [1, 3])

  def testMaskWithAxisNonConstTensor(self):
    # [FUNCTION DOC]
    # Defines a graph function with an int32 tensor-of-unknown-shape axis argument (autograph off)
    # that masks a fixed list; calls it with a constant zero and asserts the first and third elements
    # are returned. Checks that a non-constant axis tensor is accepted.

    @def_function.function(
        autograph=False,
        input_signature=[
            tensor_lib.TensorSpec(shape=None, dtype=dtypes.int32)
        ])
    def f(axis):
      # [FUNCTION DOC]
      # Nested graph function (autograph disabled) taking an axis tensor; returns the boolean mask of a
      # fixed three-element list using that axis.
      return array_ops.boolean_mask([1, 2, 3], [True, False, True], axis=axis)

    self.assertAllEqual(
        self.evaluate(f(constant_op.constant(0, dtype=dtypes.int32))), [1, 3])


@test_util.run_all_in_graph_and_eager_modes
class OperatorShapeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for static shape results and errors of expand_dims, squeeze and reshape. Run in both
  # graph and eager mode through the class decorator. No shared fixtures.

  def testExpandScalar(self):
    # [FUNCTION DOC]
    # Expands a scalar string along axis 0 and asserts the static shape is a one-element vector.
    scalar = "hello"
    scalar_expanded = array_ops.expand_dims(scalar, [0])
    self.assertEqual(scalar_expanded.get_shape(), (1,))

  def testSqueezeScalar(self):
    # [FUNCTION DOC]
    # Squeezes a scalar string with an empty axis tuple and asserts the static shape stays scalar.
    scalar = "hello"
    scalar_squeezed = array_ops.squeeze(scalar, ())
    self.assertEqual(scalar_squeezed.get_shape(), ())

  def testSqueezeMatrix(self):
    # [FUNCTION DOC]
    # Squeezes a 1 by 3 matrix on axis 0 and asserts the shape equals 3. Then squeezes axis 1 (size
    # 3) and asserts an error whose message says it cannot squeeze that dimension because it is 3.
    matrix = [[1, 2, 3]]
    matrix_squeezed = array_ops.squeeze(matrix, [0])
    self.assertEqual(matrix_squeezed.get_shape(), (3))

    with self.assertRaisesRegex(
        Exception, "Can not squeeze dim.1., expected a dimension of 1, got 3"):
      matrix_squeezed = array_ops.squeeze(matrix, [1])

  def testSqueezeScalarDim(self):
    # [FUNCTION DOC]
    # Squeezes a 1 by 3 matrix using a scalar axis (not a list) and asserts the static shape equals 3.
    matrix = [[1, 2, 3]]
    matrix_squeezed = array_ops.squeeze(matrix, 0)
    self.assertEqual(matrix_squeezed.get_shape(), (3))

  def testExpandDimsWithNonScalarDim(self):
    # [FUNCTION DOC]
    # Calls expand_dims with a two-element axis list and asserts an error saying the axis must be a
    # tensor with a single value.
    with self.assertRaisesRegex(Exception,
                                "must be a tensor with a single value"):
      array_ops.expand_dims(1, axis=[0, 1])

  def testReshapeWithManyDims(self):
    # [FUNCTION DOC]
    # Reshapes a 1x1 tensor to a shape vector of 254 ones (int64) and asserts an invalid-argument
    # error mentioning "too many dimensions" when evaluated.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "too many dimensions"):
      self.evaluate(
          array_ops.reshape(
              tensor=[[1]],
              shape=constant_op.constant([1 for i in range(254)],
                                         dtype=dtypes.int64)))


@test_util.with_eager_op_as_function
class ReverseV2Test(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the reverse (reverse_v2 and legacy reverse) operation: various dtypes, axis
  # dtypes, negative axes, large sizes and invalid axes. Decorated to run ops as functions in eager
  # mode. Contains private helpers that take a numpy dtype.

  def testReverse0DimAuto(self):
    # [FUNCTION DOC]
    # Reverses a Python scalar with no axes on CPU and GPU (sub-test each) and asserts the value is
    # unchanged.
    x_np = 4
    for use_gpu in [False, True]:
      with self.subTest(use_gpu=use_gpu):
        with self.cached_session(use_gpu=use_gpu):
          x_tf = self.evaluate(array_ops.reverse_v2(x_np, []))
          self.assertAllEqual(x_tf, x_np)

  def _reverse1DimAuto(self, np_dtype):
    # [FUNCTION DOC]
    # HELPER: shared body for testReverse1DimAuto, parameterized by numpy dtype.
    # Builds a five-element vector of the given dtype; for CPU and GPU and both int32/int64 axis
    # dtypes (sub-tests) reverses axis 0 and asserts the result equals the numpy reversed vector.
    x_np = np.array([1, 120, 3, 40, 5], dtype=np_dtype)

    for use_gpu in [False, True]:
      for axis_dtype in [dtypes.int32, dtypes.int64]:
        with self.subTest(use_gpu=use_gpu, axis_dtype=axis_dtype):
          x_tf = self.evaluate(
              array_ops.reverse_v2(x_np,
                                   constant_op.constant([0], dtype=axis_dtype)))
          self.assertAllEqual(x_tf, np.asarray(x_np)[::-1])

  def _reverse2DimAuto(self, np_dtype):
    # [FUNCTION DOC]
    # HELPER: shared body for testReverse2DimAuto, parameterized by numpy dtype.
    # Builds a 2 by 3 matrix of the dtype; for both reverse functions, CPU/GPU and both axis dtypes it
    # reverses axes 0, -2, 1, -1 and [1, 0] and compares with numpy flips of rows, columns or both.
    x_np = np.array([[1, 120, 3], [4, 5, 60]], dtype=np_dtype)

    for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
      for use_gpu in [False, True]:
        for axis_dtype in [dtypes.int32, dtypes.int64]:
          with self.subTest(
              reverse_f=reverse_f, use_gpu=use_gpu, axis_dtype=axis_dtype):
            x_tf_1 = self.evaluate(
                reverse_f(x_np, constant_op.constant([0], dtype=axis_dtype)))
            x_tf_2 = self.evaluate(
                reverse_f(x_np, constant_op.constant([-2], dtype=axis_dtype)))
            x_tf_3 = self.evaluate(
                reverse_f(x_np, constant_op.constant([1], dtype=axis_dtype)))
            x_tf_4 = self.evaluate(
                reverse_f(x_np, constant_op.constant([-1], dtype=axis_dtype)))
            x_tf_5 = self.evaluate(
                reverse_f(x_np, constant_op.constant([1, 0], dtype=axis_dtype)))
            self.assertAllEqual(x_tf_1, np.asarray(x_np)[::-1, :])
            self.assertAllEqual(x_tf_2, np.asarray(x_np)[::-1, :])
            self.assertAllEqual(x_tf_3, np.asarray(x_np)[:, ::-1])
            self.assertAllEqual(x_tf_4, np.asarray(x_np)[:, ::-1])
            self.assertAllEqual(x_tf_5, np.asarray(x_np)[::-1, ::-1])

  # This test covers the axis validation in the shape function
  # (no eval())
  def testInvalidAxis(self):
    # [FUNCTION DOC]
    # Checks shape-function axis validation (without evaluating): reversing a 2 by 3 matrix with
    # axis -30 or 2 raises a value or invalid-argument error saying the axis is out of range, and
    # axes 0 and -2 together raise an error about the axis being repeated.
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "is out of.* range"):
      array_ops.reverse_v2(x_np, [-30])
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "is out of.* range"):
      array_ops.reverse_v2(x_np, [2])
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        r"axis 0 specified more than once|axis 0 was repeated"):
      array_ops.reverse_v2(x_np, [0, -2])

  # This is the version of reverse that uses axis indices rather than
  # bool tensors
  # TODO(b/32254538): Change this test to use array_ops.reverse
  #
  # Note: this test passes placeholder as constant axis is validated
  # in shape function (see testInvalidAxis)
  def testInvalid(self):
    # [FUNCTION DOC]
    # Same invalid-axis checks as testInvalidAxis, but axes come from a placeholder argument of a
    # graph function so validation happens at run time: out-of-range axes -30 and 2, and duplicate
    # axes 0 and -2, must raise a value or invalid-argument error with the matching message.
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    @def_function.function
    def func(ax):
      # [FUNCTION DOC]
      # Nested graph function taking an axis argument; reverses a fixed 2 by 3 float matrix along it.
      return array_ops.reverse_v2(x_np, ax)

    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "is out of.*range"):
      func([-30])
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "is out of.*range"):
      func([2])
    with self.assertRaisesRegex(
        (ValueError, errors_impl.InvalidArgumentError),
        "(axis 0 specified more than once|canonicalized axis 0 was repeated.)"):
      func([0, -2])

  def testReverse1DimAuto(self):
    # [FUNCTION DOC]
    # Runs the private 1-D reverse helper for unsigned and signed integer types, bool, floats,
    # complex types, bytes strings and bfloat16.
    for dtype in [
        np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64,
        np.int64, np.bool_, np.float16, np.float32, np.float64, np.complex64,
        np.complex128,
        np.array(b"").dtype.type, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._reverse1DimAuto(dtype)

  def testReverse2DimAuto(self):
    # [FUNCTION DOC]
    # Runs the private 2-D reverse helper for unsigned and signed integer types, bool, floats,
    # complex types, bytes strings and bfloat16.
    for dtype in [
        np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64,
        np.int64, np.bool_, np.float16, np.float32, np.float64, np.complex64,
        np.complex128,
        np.array(b"").dtype.type, dtypes.bfloat16.as_numpy_dtype
    ]:
      self._reverse2DimAuto(dtype)

  def testReverseRowsOf3Channels(self):
    # [FUNCTION DOC]
    # Exercises the optimized path for reversing the middle axis of arrays whose last dimension is 3.
    # For both reverse functions, outer sizes 1 and 2 and middle sizes 0..49 plus 100000 (sub-tests)
    # it reverses axis 1 of a float32 array and asserts it equals the numpy flip.
    """Tests optimized code for reversing rows with last dim size = 3."""
    for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
      for outer_size in (1, 2):
        for middle_size in list(range(50)) + [100000]:
          with self.subTest(
              reverse_f=reverse_f,
              outer_size=outer_size,
              middle_size=middle_size,
              use_gpu=True):
            x_np = np.reshape(
                np.arange(outer_size * middle_size * 3, dtype=np.float32),
                (outer_size, middle_size, 3),
            )
            x_tf = self.evaluate(reverse_f(x_np, [1]))
            np_answer = x_np[:, ::-1, :]
            self.assertAllEqual(x_tf, np_answer)

  def testReverseRowsOf4Channels(self):
    # [FUNCTION DOC]
    # Same as the 3-channel row test but with last dimension 4: reverses axis 1 for many middle sizes
    # and both reverse functions, comparing to the numpy flip.
    for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
      for outer_size in (1, 2):
        for middle_size in list(range(50)) + [100000]:
          with self.subTest(
              reverse_f=reverse_f,
              outer_size=outer_size,
              middle_size=middle_size,
              use_gpu=True):
            x_np = np.reshape(
                np.arange(outer_size * middle_size * 4, dtype=np.float32),
                (outer_size, middle_size, 4),
            )
            x_tf = self.evaluate(reverse_f(x_np, [1]))
            np_answer = x_np[:, ::-1, :]
            self.assertAllEqual(x_tf, np_answer)

  def testReverseColumnsOf3Channels(self):
    # [FUNCTION DOC]
    # Reverses axis 0 of float32 arrays whose last dimension is 3 and middle dimension is 1 or 2, for
    # many outer sizes (0..49 and 100000) and both reverse functions, comparing with numpy flips.
    for reverse_f in [array_ops.reverse_v2, array_ops.reverse]:
      for outer_size in list(range(50)) + [100000]:
        for middle_size in (1, 2):
          with self.subTest(
              reverse_f=reverse_f,
              outer_size=outer_size,
              middle_size=middle_size,
              use_gpu=True):
            x_np = np.reshape(
                np.arange(outer_size * middle_size * 3, dtype=np.float32),
                (outer_size, middle_size, 3),
            )
            x_tf = self.evaluate(reverse_f(x_np, [0]))
            np_answer = x_np[::-1, :, :]
            self.assertAllEqual(x_tf, np_answer)

  def testReverseInvalidShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests an error case, but the test expects success: it reverses an array
    # of shape 0 by 1 by 1 along axis 1 and only asserts the evaluated result equals the op output.
    # The comparison is against the same op's output, so it checks little beyond not crashing.
    # Verifies that reversing a zero-element array does not fail.
    x = np.ndarray(shape=[0, 1, 1])
    v = array_ops.reverse_v2(x, axis=[1])
    self.assertAllEqual(self.evaluate(v), v)


class MeshgridTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case comparing the meshgrid operation against numpy meshgrid for both indexing modes
  # (matrix and Cartesian), several dtypes and CPU/GPU placement. Defines private comparison helpers.

  def _compareDiff(self, x, y, use_gpu):
    # [FUNCTION DOC]
    # HELPER: comparison helper for testCompare.
    # Inputs: x and y (two coordinate vectors, lists or arrays) and use_gpu (whether to place on GPU).
    # For both index modes computes numpy and TensorFlow meshgrids and asserts each output pair
    # matches. Returns nothing.
    for index in ("ij", "xy"):
      numpy_out = np.meshgrid(x, y, indexing=index)
      tf_out = array_ops.meshgrid(x, y, indexing=index)
      with self.cached_session(use_gpu=use_gpu):
        for xx, yy in zip(numpy_out, tf_out):
          self.assertAllEqual(xx, yy)

  def _compareDiffType(self, n, np_dtype, use_gpu):
    # [FUNCTION DOC]
    # HELPER: comparison helper for testCompare.
    # KNOWN ISSUE: the input list is created once outside the index-mode loop and not reset, so the
    # second mode receives 2n inputs rather than n (the vectors are also appended inside that loop).
    # Inputs: n (number of input vectors), np_dtype (numpy dtype) and use_gpu. For each index mode it
    # appends n new linspace vectors (cast to dtype, plus 1j for complex types) to the input list,
    # then compares numpy and TensorFlow meshgrid outputs of all inputs on the chosen device.
    inputs = []
    for index in ("ij", "xy"):
      for _ in range(n):
        x = np.linspace(-10, 10, 5).astype(np_dtype)
        if np_dtype in (np.complex64, np.complex128):
          x += 1j
        inputs.append(x)
      numpy_out = np.meshgrid(*inputs, indexing=index)
      with test_util.device(use_gpu=use_gpu):
        tf_out = array_ops.meshgrid(*inputs, indexing=index)
        for x_np, x_tf in zip(numpy_out, tf_out):
          self.assertAllEqual(x_np, x_tf)

  def testCompare(self):
    # [FUNCTION DOC]
    # Loops over float16/32/64, int32/64 and complex64/128 (sub-test each). Runs the typed helper with
    # 2 and 3 inputs on CPU, then compares meshgrid of two small lists and of a list with a nested
    # 2x2 list through the plain helper, all with GPU disabled.
    for t in (np.float16, np.float32, np.float64, np.int32, np.int64,
              np.complex64, np.complex128):
      with self.subTest(t=t):
        self._compareDiffType(2, t, False)
        self._compareDiffType(3, t, False)

        x = [1, 2, 3]
        y = [4, 5]

        a = [[1, 1], [1, 1]]

        self._compareDiff(x, y, False)
        self._compareDiff(x, a, False)


class StridedSliceChecker(object):
  # [CLASS DOC]
  # HELPER: test utility that checks tensor indexing results against numpy.
  # Constructed with the test case, source data, tensor dtype and a flag for static shape checking.
  # Stores the numpy array x_np (cast to the dtype; booleans derived from the data modulo 3;
  # complex types get a nonzero imaginary part), the matching constant tensor x, the test case and
  # the shape-check flag. Index syntax on the object evaluates the slice and compares to numpy.
  # Class constants REF_TENSOR (3x2x3) and REF_TENSOR_ALIGNED (3x4x8) are reference float32 arrays.
  """Check a given tensor against the numpy result."""

  REF_TENSOR = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
  REF_TENSOR_ALIGNED = np.arange(1, 97, dtype=np.float32).reshape(3, 4, 8)

  def __init__(self, test, x, tensor_type=dtypes.int32, check_type_infer=True):
    # [FUNCTION DOC]
    # Stores the owning test case and builds data: x_np is the data cast to the dtype's numpy type
    # (for booleans the data modulo 3 cast to bool; for complex types the imaginary part is set to
    # the negative of the real part). Creates the constant tensor x of that dtype and records whether
    # static shape inference should be compared (check_type_infer, default true). Returns nothing.
    self.x_np = np.array(x).astype(tensor_type.as_numpy_dtype)
    if tensor_type.is_bool:
      self.x_np = np.array(x % 3).astype(np.bool_)
    # Give the value a non-zero imaginary component for complex types.
    if tensor_type.is_complex:
      self.x_np -= 1j * self.x_np
    self.test = test
    self.x = constant_op.constant(self.x_np, dtype=tensor_type)
    self.check_type_infer = check_type_infer

  def __getitem__(self, spec):
    # [FUNCTION DOC]
    # Index operator. Applies the given spec (any slice spec, including tensors, ellipsis, new axis,
    # masks) to the tensor, evaluates the result with the test case and compares against numpy
    # indexing of the numpy array. Boolean specs (bool scalar, bool tensor/array or list of bools)
    # use direct numpy boolean indexing. Other specs are normalized to a tuple, slice components that
    # are tensors are evaluated first. Asserts values equal and, when check_type_infer is true, that
    # the static shape matches the evaluated shape. Returns the evaluated numpy result.
    op = self.x.__getitem__(spec)

    def eval_if_tensor(x):
      # [FUNCTION DOC]
      # Nested helper: tries to evaluate its argument with the test case; if that fails with an
      # attribute, type or value error (plain Python value) returns the argument unchanged.
      try:
        return self.test.evaluate(x)
      except (AttributeError, TypeError, ValueError):
        return x

    def casts_to_bool_nparray(x):
      # [FUNCTION DOC]
      # Nested helper: returns true if the argument converts to a numpy array of bool dtype, false if
      # conversion raises a not-implemented error.
      try:
        return np.asarray(x).dtype == bool
      except NotImplementedError:
        return False

    if (
        isinstance(spec, bool)
        or (isinstance(spec, tensor_lib.Tensor) and spec.dtype == dtypes.bool)
        or (isinstance(spec, np.ndarray) and spec.dtype == bool)
        or (isinstance(spec, (list, tuple)) and casts_to_bool_nparray(spec))
    ):
      tensor = self.test.evaluate(op)
      np_spec = eval_if_tensor(spec)
      self.test.assertAllEqual(self.x_np[np_spec], tensor)
      return tensor

    if not isinstance(spec, (list, tuple)):
      spec = [spec]

    tensor = self.test.evaluate(op)

    # Make a numpy spec that pre-evals the tensors
    np_specs = []

    for s in spec:
      if isinstance(s, slice):
        start = eval_if_tensor(s.start)
        stop = eval_if_tensor(s.stop)
        step = eval_if_tensor(s.step)
        np_specs.append(slice(start, stop, step))
      else:
        np_specs.append(eval_if_tensor(s))

    self.test.assertAllEqual(self.x_np[tuple(np_specs)], tensor)
    if self.check_type_infer:
      self.test.assertAllEqual(tensor.shape, op.get_shape())
    return tensor


STRIDED_SLICE_TYPES = [
    dtypes.int32,
    dtypes.int64,
    dtypes.int16,
    dtypes.int8,
    dtypes.uint8,
    dtypes.float32,
    dtypes.float64,
    dtypes.complex64,
    dtypes.complex128,
    dtypes.bool,
    dtypes.bfloat16,
    dtypes.float8_e5m2,
    dtypes.float8_e4m3fn,
]


class StridedSliceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for strided slicing through Python indexing of tensors: identity, reverse, negative
  # indices, degenerate intervals, ellipsis, shrink, new axis, tensor indices, masks, error cases
  # and memory-leak checks. Uses the checker helper and the module list of data types.
  """Test the strided slice operation with variants of slices."""

  def test_basic_slice(self):
    # [FUNCTION DOC]
    # For every type in the module's data type list (sub-test each), checks many slice forms with
    # the checker against numpy: identity forms, steps, reversed dimensions, negative start with
    # negative step, and rank-0 indexing with new axis, ellipsis and empty tuple.
    for tensor_type in STRIDED_SLICE_TYPES:
      with self.subTest(tensor_type=tensor_type, use_gpu=True):
        checker = StridedSliceChecker(
            self, StridedSliceChecker.REF_TENSOR, tensor_type=tensor_type)
        _ = checker[:, :, :]
        # Various ways of representing identity slice
        _ = checker[:, :, :]
        _ = checker[::, ::, ::]
        _ = checker[::1, ::1, ::1]
        # Not zero slice
        _ = checker[::1, ::5, ::2]
        # Reverse in each dimension independently
        _ = checker[::-1, :, :]
        _ = checker[:, ::-1, :]
        _ = checker[:, :, ::-1]
        ## negative index tests i.e. n-2 in first component
        _ = checker[-2::-1, :, ::1]
        # negative index tests i.e. n-2 in first component, non-unit stride
        _ = checker[-2::-1, :, ::2]

        # Check rank-0 examples
        checker2 = StridedSliceChecker(self, 5, tensor_type=tensor_type)
        _ = checker2[None]
        _ = checker2[...]
        _ = checker2[tuple()]

  def testInt64GPU(self):
    # [FUNCTION DOC]
    # Skipped when no GPU is available. Forces GPU placement and runs strided_slice with int64 begin,
    # end and strides over a float vector; asserts the result is the element 3.0.
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    with test_util.force_gpu():
      x = constant_op.constant([1., 2., 3.])
      begin = constant_op.constant([2], dtype=dtypes.int64)
      end = constant_op.constant([3], dtype=dtypes.int64)
      strides = constant_op.constant([1], dtype=dtypes.int64)
      s = array_ops.strided_slice(x, begin, end, strides)
      self.assertAllEqual([3.], self.evaluate(s))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  @test_util.assert_no_garbage_created
  def testTensorSliceEagerMemory(self):
    # [FUNCTION DOC]
    # Eager-only leak check (decorators assert no new Python objects or garbage). Slices an eager
    # tensor by an integer index to verify no memory is leaked.
    with context.eager_mode():
      inputs = constant_op.constant([[[1], [2], [3], [4]]],
                                    dtype=dtypes.float32)
      # Tests that slicing an EagerTensor doesn't leak memory
      inputs[0]  # pylint: disable=pointless-statement

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  @test_util.assert_no_garbage_created
  def testVariableSliceEagerMemory(self):
    # [FUNCTION DOC]
    # Eager-only leak check (decorators assert no new Python objects or garbage). Indexes a variable
    # element to verify no memory is leaked.
    with context.eager_mode():
      v = variables.Variable([1., 2.])
      v[0]  # pylint: disable=pointless-statement

  def testDegenerateSlices(self):
    # [FUNCTION DOC]
    # On GPU-capable device placement, slices a reference tensor with degenerate/empty intervals
    # (forward interval with negative stride, reverse interval with positive stride, empty in every
    # dimension) and an aligned tensor with an empty first dimension, comparing each with numpy.
    with test_util.device(use_gpu=True):
      checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
      # degenerate by offering a forward interval with a negative stride
      _ = checker[0:-1:-1, :, :]
      # degenerate with a reverse interval with a positive stride
      _ = checker[-1:0, :, :]
      # empty interval in every dimension
      _ = checker[-1:0, 2:2, 2:3:-1]
      # empty first dimension only (used to break for aligned tensors).
      checker = StridedSliceChecker(self,
                                    StridedSliceChecker.REF_TENSOR_ALIGNED)
      _ = checker[1:0]

  def testSliceWithUndefinedDimension(self):
    # [FUNCTION DOC]
    # Slices a constant vector using an undefined Dimension as start, stop and step and asserts the
    # result equals the original tensor.
    t = constant_op.constant([1, 2, 3])
    d = tensor_shape.Dimension(None)
    self.assertAllEqual(t[d:d:d], t)

  def testEllipsis(self):
    # [FUNCTION DOC]
    # Checks ellipsis slicing on a 5-D nested list for several positions (alone, start, end, middle,
    # implicit) against numpy. Also asserts that using two ellipses raises a value or invalid-argument
    # error mentioning multiple ellipses.
    with test_util.device(use_gpu=True):
      raw = [[[[[1, 2], [3, 4], [5, 6]]], [[[7, 8], [9, 10], [11, 12]]]]]
      checker = StridedSliceChecker(self, raw)

      _ = checker[0:]
      # implicit ellipsis
      _ = checker[0:, ...]
      # ellipsis alone
      _ = checker[...]
      # ellipsis at end
      _ = checker[0:1, ...]
      # ellipsis at begin
      _ = checker[..., 0:1]
      # ellipsis at middle
      _ = checker[0:1, ..., 0:1]
      # multiple ellipses not allowed
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "Multiple ellipses"):
        _ = checker[..., :, ...].eval()

  def testShrink(self):
    # [FUNCTION DOC]
    # Checks shrink-axis (integer index) slices on a 5-D nested list, with ellipsis combinations,
    # against numpy via the checker.
    with test_util.device(use_gpu=True):
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw)
      _ = checker[:, :, :, :, 3]
      _ = checker[..., 3]
      _ = checker[:, 0]
      _ = checker[:, :, 0]

  def testBothNewAxisAndShrink(self):
    # [FUNCTION DOC]
    # Traces a graph function that indexes a 2 by 2 int16 input with a new axis, all of axis 0 and
    # index 0 of axis 1, creates the input constant on CPU (workaround for a GPU copy bug) and asserts
    # the result equals one row of ones.
    with test_util.device(use_gpu=True):

      @def_function.function
      def func(inp):
        # [FUNCTION DOC]
        # Nested graph function: indexes its input with a new axis, a full slice and the integer 0.
        return inp[array_ops.newaxis, :, 0]

      f = func.get_concrete_function(
          tensor_lib.TensorSpec([2, 2], dtypes.int16))

      # TODO(b/190416665): Allow the constant to be eagerly copied/created on
      # the GPU.
      with ops.device("CPU"):
        ones = constant_op.constant([[1, 1], [1, 1]], dtypes.int16)
      self.assertAllEqual([[1, 1]], self.evaluate(f(ones)))

  def testTensorIndexing(self):
    # [FUNCTION DOC]
    # Checks indexing using tensor-valued indices (slice bounds and scalar index) combined with
    # ellipsis and also a large Python integer, with static shape checking disabled.
    with test_util.device(use_gpu=True):
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw, check_type_infer=False)
      bar = constant_op.constant(2)
      bar2 = constant_op.constant(3)
      _ = checker[..., bar:bar2]
      _ = checker[..., bar]
      _ = checker[..., 3]
      _ = checker[..., 2**64 // 2**63]  # Test longs in Python 2

  def testTensorIndexingTypeError(self):
    # [FUNCTION DOC]
    # Asserts that invalid index types (string, string tensor, float, float tensor, integer vector
    # tensor and list of floats) raise a type error carrying the module's slice type error message.
    with self.session():
      checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
      expected = re.escape(tensor_getitem_override._SLICE_TYPE_ERROR)
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker["foo"]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[constant_op.constant("foo")]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[0.0]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[constant_op.constant(0.0)]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[constant_op.constant([1, 2, 3])]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[[2.1, -0.7, 1.5]]

  def testExpand(self):
    # [FUNCTION DOC]
    # Checks new axis indexing in various positions (leading, after ellipsis, between ellipsis and a
    # range, mixed with reversed final dimension, around an ellipsis) against numpy.
    with test_util.device(use_gpu=True):
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw)
      # new axis (followed by implicit ellipsis)
      _ = checker[np.newaxis]
      # newaxis after ellipsis
      _ = checker[..., np.newaxis]
      # newaxis in between ellipsis and explicit range
      _ = checker[..., np.newaxis, :]
      _ = checker[:, ..., np.newaxis, :, :]
      # Reverse final dimension with new axis
      _ = checker[:, :, np.newaxis, :, 2::-1]
      # Ellipsis in middle of two newaxis
      _ = checker[np.newaxis, ..., np.newaxis]

  def testExpandVariable(self):
    # [FUNCTION DOC]
    # Indexes a scalar int32 variable with a new axis after initializing it; asserts the result has
    # shape one and value 7.
    with test_util.device(use_gpu=True):
      x = variables.Variable(7, dtype=dtypes.int32)
      self.evaluate(x.initializer)
      y = self.evaluate(x[None])
      self.assertEqual(y.shape, (1,))
      self.assertAllEqual(y, (7,))

  def testOptimizedCases(self):
    # [FUNCTION DOC]
    # Checks slicing patterns that take optimized paths (identity forms, first axis slice, with new
    # axes) on an aligned reference tensor against numpy.
    with test_util.device(use_gpu=True):
      checker = StridedSliceChecker(self,
                                    StridedSliceChecker.REF_TENSOR_ALIGNED)
      # Identity
      _ = checker[:]
      # Identity
      _ = checker[...]
      # Identity
      _ = checker[np.newaxis, ..., np.newaxis]
      # First axis slice
      _ = checker[1:]
      # First axis slice
      _ = checker[np.newaxis, 1:]

  def testMasks(self):
    # [FUNCTION DOC]
    # Checks boolean-mask indexing: tensor comparison masks, a scalar tensor, numpy array masks on a
    # 5-D array, and a numpy boolean vector plus its tensor version, each against numpy results.
    with test_util.device(use_gpu=True):
      scalar = np.array(0)
      # Test tensor type mask
      checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
      _ = checker[checker.x > 2]
      _ = checker[checker.x <= 5]
      _ = checker[ops.convert_to_tensor(scalar)]

      # Test numpy array type mask
      raw = np.array([[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
                       [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23,
                                                              24]]]]])
      checker1 = StridedSliceChecker(self, raw)
      _ = checker1[raw >= 4]
      _ = checker1[raw < 19]
      _ = checker1[scalar]

      # Test boolean and non boolean cases
      mask = np.array([True, False, True])
      raw1 = np.array([[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]])
      checker2 = StridedSliceChecker(self, raw1)
      _ = checker2[mask]
      _ = checker2[ops.convert_to_tensor(mask)]

  def test_int16_indices(self):
    # [FUNCTION DOC]
    # Checks mixed int16/int64 slice bounds for each type in the module data type list. A single
    # int16 index and all-int16 slice bounds must work, while slices mixing int16 and int64 bounds
    # must raise some exception.

    def _int16(i):
      # [FUNCTION DOC]
      # HELPER: nested helper making an int16 scalar constant from the given value.
      return constant_op.constant(i, dtype=dtypes.int16)

    def _int64(i):
      # [FUNCTION DOC]
      # HELPER: nested helper making an int64 scalar constant from the given value.
      return constant_op.constant(i, dtype=dtypes.int64)

    for tensor_type in STRIDED_SLICE_TYPES:
      with self.subTest(tensor_type=tensor_type, use_gpu=True):
        checker = StridedSliceChecker(
            self, StridedSliceChecker.REF_TENSOR, tensor_type=tensor_type)

        _ = checker[_int16(1)]

        with self.assertRaises(Exception):
          _ = checker[_int16(1)::1, :, 1:_int64(3):2]
        with self.assertRaises(Exception):
          _ = checker[:, _int16(1):_int16(5):-1, :]
        with self.assertRaises(Exception):
          _ = checker[::_int64(1), _int64(1):10:_int16(3), ::_int64(2)]

        _ = checker[::_int16(1), _int16(1)::_int16(5), ::2]
        _ = checker[_int16(1):_int16(5):_int16(2), 1:2, :]


class StridedSliceShapeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for static shape inference of strided slice via tensor indexing, using graph
  # function tracing with partially unknown shapes.
  """Test the shape inference of StridedSliceShapes."""

  def testUnknown(self):
    # [FUNCTION DOC]
    # Traces a graph function on an input of unknown rank where the body slices with ellipsis and
    # asserts the result's static rank is unknown.
    with test_util.device(use_gpu=True):

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # Nested graph function: slices its argument with an ellipsis and asserts the static rank of the
        # result is unknown. Returns nothing.
        y = x[...]
        self.assertAllEqual(y.get_shape().ndims, None)

      _ = f.get_concrete_function(tensor_lib.TensorSpec(None, dtypes.float32))

  def testScalarInput(self):
    # [FUNCTION DOC]
    # Calls strided_slice on a scalar and asserts an error saying it is attempting to slice a scalar.
    c = constant_op.constant(3)
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "Attempting to slice scalar input.",
    ):
      array_ops.strided_slice(c, [0], [1])

  def tensorShapeEqual(self, x, y):
    # [FUNCTION DOC]
    # HELPER: shape comparison helper for the shape tests.
    # KNOWN ISSUE: the none-handling is dead or misleading: if both shapes are none the first
    # assertion passes but the next line calls the list conversion on none and raises an attribute
    # error, so the none case can never succeed.
    # Takes two tensor shapes, asserts either both or neither are none, and asserts their dimension
    # lists are equal. Returns nothing.
    self.assertTrue(x is not None and y is not None or x is None and y is None)
    self.assertEqual(x.as_list(), y.as_list())

  def testTensorShapeUncertain(self):
    # [FUNCTION DOC]
    # Traces ten graph functions (f1 to f10) with input shape 5 by unknown by 7 float32, each slicing
    # differently (ranges, scalar index, new axis, empty ranges, steps and reversal). Each body asserts
    # the inferred static shape equals the expected partial shape. Only tracing is performed.
    with test_util.device(use_gpu=True):

      @def_function.function
      def f1(x):
        # [FUNCTION DOC]
        # Nested traced function: slices the first dimension 3..5 and asserts the shape is 2, unknown, 7.
        y = x[3:5]
        self.tensorShapeEqual(y.get_shape(),
                              tensor_shape.TensorShape([2, None, 7]))

      _ = f1.get_concrete_function(
          tensor_lib.TensorSpec((5, None, 7), dtypes.float32))

      @def_function.function
      def f2(x):
        # [FUNCTION DOC]
        # Nested traced function: slices 3..5 on axis 0, all of axis 1 and index 4 on axis 2; asserts
        # shape 2, unknown.
        y = x[3:5, :, 4]
        self.tensorShapeEqual(y.get_shape(), tensor_shape.TensorShape([2,
                                                                       None]))

      _ = f2.get_concrete_function(
          tensor_lib.TensorSpec((5, None, 7), dtypes.float32))

      @def_function.function
      def f3(x):
        # [FUNCTION DOC]
        # Nested traced function: slices 3..5, 3..4 and index 4; asserts shape 2, unknown.
        y = x[3:5, 3:4, 4]
        self.tensorShapeEqual(y.get_shape(), tensor_shape.TensorShape([2,
                                                                       None]))

      _ = f3.get_concrete_function(
          tensor_lib.TensorSpec((5, None, 7), dtypes.float32))

      @def_function.function
      def f4(x):
        # [FUNCTION DOC]
        # Nested traced function: slices 3..5, all, and 5..10; asserts shape 2, unknown, 2.
        y = x[3:5, :, 5:10]
        self.tensorShapeEqual(y.get_shape(),
                              tensor_shape.TensorShape([2, None, 2]))

      _ = f4.get_concrete_function(
          tensor_lib.TensorSpec((5, None, 7), dtypes.float32))

      @def_function.function
      def f5(x):
        # [FUNCTION DOC]
        # Nested traced function: slices 3..5, all, and an empty range 50..3; asserts shape 2, unknown, 0.
        y = x[3:5, :, 50:3]
        self.tensorShapeEqual(y.get_shape(),
                              tensor_shape.TensorShape([2, None, 0]))

      _ = f5.get_concrete_function(
          tensor_lib.TensorSpec((5, None, 7), dtypes.float32))

      @def_function.function
      def f6(x):
        # [FUNCTION DOC]
        # Nested traced function: slices 3..5, all, new axis, empty range; asserts shape 2, unknown, 1, 0.
        y = x[3:5, :, array_ops.newaxis, 50:3,]
        self.tensorShapeEqual(y.get_shape(),
                              tensor_shape.TensorShape([2, None, 1, 0]))

      _ = f6.get_concrete_function(
          tensor_lib.TensorSpec((5, None, 7), dtypes.float32))

      @def_function.function
      def f7(x):
        # [FUNCTION DOC]
        # Nested traced function: step-2 slice of 1..5, all, new axis, empty range; asserts 2, unknown,
        # 1, 0.
        y = x[1:5:2, :, array_ops.newaxis, 50:3,]
        self.tensorShapeEqual(y.get_shape(),
                              tensor_shape.TensorShape([2, None, 1, 0]))

      _ = f7.get_concrete_function(
          tensor_lib.TensorSpec((5, None, 7), dtypes.float32))

      @def_function.function
      def f8(x):
        # [FUNCTION DOC]
        # Nested traced function: slice up to 5 step 3, all, new axis, empty range; asserts 2, unknown,
        # 1, 0.
        y = x[:5:3, :, array_ops.newaxis, 50:3,]
        self.tensorShapeEqual(y.get_shape(),
                              tensor_shape.TensorShape([2, None, 1, 0]))

      _ = f8.get_concrete_function(
          tensor_lib.TensorSpec((5, None, 7), dtypes.float32))

      @def_function.function
      def f9(x):
        # [FUNCTION DOC]
        # Nested traced function: slice up to 2 step 3, all, new axis, empty range; asserts 1, unknown,
        # 1, 0.
        y = x[:2:3, :, array_ops.newaxis, 50:3,]
        self.tensorShapeEqual(y.get_shape(),
                              tensor_shape.TensorShape([1, None, 1, 0]))

      _ = f9.get_concrete_function(
          tensor_lib.TensorSpec((5, None, 7), dtypes.float32))

      @def_function.function
      def f10(x):
        # [FUNCTION DOC]
        # Nested traced function: fully reversed first axis, all, new axis, reversed step-2 last axis;
        # asserts shape 5, unknown, 1, 4.
        y = x[::-1, :, array_ops.newaxis, ::-2]
        self.tensorShapeEqual(y.get_shape(),
                              tensor_shape.TensorShape([5, None, 1, 4]))

      _ = f10.get_concrete_function(
          tensor_lib.TensorSpec((5, None, 7), dtypes.float32))

  def testTensorValuedIndexShape(self):
    # [FUNCTION DOC]
    # Traces four graph functions (f1 to f4) over a 5 by 3 by 7 input with tensor-valued scalar
    # indices or slice bounds, asserting inferred static shapes (3,7), (3,7), (2,7) and (unknown,7).
    with self.session():

      @def_function.function
      def f1(x, y):
        # [FUNCTION DOC]
        # Nested traced function (inputs x and index y): selects x at y; asserts shape 3 by 7.
        z = x[y]
        self.tensorShapeEqual(z.get_shape(), tensor_shape.TensorShape([3, 7]))

      _ = f1.get_concrete_function(
          tensor_lib.TensorSpec((5, 3, 7)),
          tensor_lib.TensorSpec((), dtypes.int32))

      @def_function.function
      def f2(x, y):
        # [FUNCTION DOC]
        # Nested traced function: selects x at y and reverses the next axis; asserts shape 3 by 7.
        z = x[y, ::-1]
        self.tensorShapeEqual(z.get_shape(), tensor_shape.TensorShape([3, 7]))

      _ = f2.get_concrete_function(
          tensor_lib.TensorSpec((5, 3, 7)),
          tensor_lib.TensorSpec((), dtypes.int32))

      @def_function.function
      def f3(x, y):
        # [FUNCTION DOC]
        # Nested traced function: selects x at y and takes the next axis with step -2; asserts shape 2 by 7.
        z = x[y, ::-2]
        self.tensorShapeEqual(z.get_shape(), tensor_shape.TensorShape([2, 7]))

      _ = f3.get_concrete_function(
          tensor_lib.TensorSpec((5, 3, 7)),
          tensor_lib.TensorSpec((), dtypes.int32))

      @def_function.function
      def f4(x, y, s):
        # [FUNCTION DOC]
        # Nested traced function (inputs x, y and s): selects x at y and slices from tensor s to 2 on the
        # next axis; asserts shape unknown by 7.
        z = x[y, s:2]
        self.tensorShapeEqual(z.get_shape(), tensor_shape.TensorShape([None,
                                                                       7]))

      _ = f4.get_concrete_function(
          tensor_lib.TensorSpec((5, 3, 7)),
          tensor_lib.TensorSpec((), dtypes.int32),
          tensor_lib.TensorSpec((), dtypes.int32))


class GradSliceChecker(object):
  # [CLASS DOC]
  # HELPER: test utility that verifies gradients of sliced squared variables.
  # Constructed with the test case, a variable, its numpy value and a flag for using real gradient
  # tape or the abstract tape. Indexing it computes first and second gradients and compares them
  # with analytic values.
  """Tests that we can compute a gradient for var^2."""

  def __init__(self, test, var, varnp, use_tape):
    # [FUNCTION DOC]
    # Stores the test case, the variable, its numpy value and the use-tape flag as attributes.
    # Returns nothing.
    self.test = test
    self.var = var
    self.varnp = varnp
    self.use_tape = use_tape

  def __getitem__(self, spec):
    # [FUNCTION DOC]
    # Index operator. Under a persistent (abstract) gradient tape watching the variable computes the
    # square of the variable, slices both the variable and the square with the given spec, assigns the
    # sliced variable into an auxiliary variable of ones, and takes the gradient of the sliced square
    # and then its gradient with respect to the auxiliary variable. Evaluates everything and asserts
    # that the second derivative equals twice the sliced square and that the first gradient equals
    # twice the squared numpy value inside the slice and zero outside. Tensor specs are evaluated first.
    # Returns nothing.
    with test_util.AbstractGradientTape(
        use_tape=self.use_tape, persistent=True) as tape:
      tape.watch(self.var)
      val = self.var * self.var
      slice_var = self.var[spec]
      slice_val = val[spec]

      # compute analytic 2nd derivative
      analytic_grad2 = 2 * slice_val

      dy = variables.Variable(
          array_ops.ones_like(slice_var, dtype=dtypes.float32))
      assign = dy.assign(slice_var)

      slice_val_grad = tape.gradient(slice_val, self.var, [dy])
      slice_val_grad2 = tape.gradient(slice_val_grad, dy, [self.var])
    self.test.evaluate(assign)
    slice_val_grad_evaled, slice_val_grad2_evaled = (
        self.test.evaluate([slice_val_grad, slice_val_grad2]))
    analytic_grad2_evaled = self.test.evaluate(analytic_grad2)
    self.test.assertAllEqual(slice_val_grad2_evaled, analytic_grad2_evaled)

    # compute analytic gradient for slice
    np_val_grad = (2 * self.varnp * self.varnp)
    np_sliceval_grad = np.zeros(self.var.get_shape())
    if isinstance(spec, tensor_lib.Tensor):
      spec = self.test.evaluate(spec)
    np_sliceval_grad[spec] = np_val_grad[spec]
    # verify gradient
    self.test.assertAllEqual(slice_val_grad_evaled, np_sliceval_grad)


class StridedSliceGradTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Test case verifying the custom gradient of strided slice, parameterized over whether a real
  # gradient tape is used. Uses the gradient checker helper class.
  """Test that strided slice's custom gradient produces correct gradients."""

  @parameterized.parameters(set((True, context.executing_eagerly())))
  @test_util.disable_xla(
      "b/210077724: Auto-clustering with where op isn't supported. Has loose "
      "output shape bounds")
  def testGradient(self, use_tape):
    # [FUNCTION DOC]
    # Parameterized by use_tape (set of true and whether executing eagerly); XLA disabled due to a
    # known bug. On a 6 by 4 by 4 variable checks gradients of strided slices with positive and
    # negative steps, new axis, shrink, negative indices, and numpy and tensor boolean masks. Asserts
    # that indices out of bounds (-200 and 200) raise value or invalid-argument errors.
    with test_util.device(use_gpu=True):
      var = variables.Variable(
          array_ops.reshape(
              math_ops.range(1, 97, 1, dtype=dtypes.float32), shape=(6, 4, 4)))
      self.evaluate(var.initializer)

      raw = np.array(range(1, 97, 1)).reshape((6, 4, 4))
      grad = GradSliceChecker(self, var, raw, use_tape)
      _ = grad[2:6:2, 1:3, 1:3]
      _ = grad[3:0:-2, 1:3, 1:3]
      _ = grad[3:0:-2, array_ops.newaxis, 1:3, 2, array_ops.newaxis]
      _ = grad[3:0:-2, 1:3, 2]
      _ = grad[:, -1, :]
      _ = grad[:, -2, :]
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "out of bounds"):
        _ = grad[:, -200, :]
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "out of bounds"):
        _ = grad[:, 200, :]

      # Test numpy array type mask
      _ = grad[raw > 51]
      # Test tensor type mask
      _ = grad[ops.convert_to_tensor(raw) <= 76]

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradientZero(self, use_tape):
    # [FUNCTION DOC]
    # Parameterized by use_tape. Checks the gradient of indexing a scalar variable with an empty tuple.
    with test_util.device(use_gpu=True):
      var = variables.Variable(8.)
      self.evaluate(var.initializer)
      grad = GradSliceChecker(self, var, np.array(8), use_tape)
      _ = grad[tuple()]

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testInt64Indices(self, use_tape):
    # [FUNCTION DOC]
    # Parameterized by use_tape. Indexes a float vector with an int64 scalar tensor under a tape and
    # asserts the gradient of twice the element is the vector 0, 2, 0.
    with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
      a = math_ops.range(3, dtype=dtypes.float32)
      tape.watch(a)
      index = constant_op.constant(1, dtype=dtypes.int64)
      b = 2. * a[index]
    grad = tape.gradient(b, a)
    self.assertAllEqual(self.evaluate(grad), [0., 2., 0.])


class StridedSliceGradTypeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the strided slice gradient op with differing index dtypes and host or device
  # placed arguments.
  """Test varied index types and host located memory."""

  def testHostVsDevice(self):
    # [FUNCTION DOC]
    # Calls the strided slice gradient op with a variable holding the original shape (device
    # memory) and a variable gradient, initializes variables and evaluates; passes when no error occurs.
    var2 = variables.Variable(
        array_ops.reshape(
            math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32),
            shape=(4, 1, 1)))
    varshape = variables.Variable([6, 4, 4], dtype=dtypes.int32)
    begin = constant_op.constant([0, 0, 0])
    end = constant_op.constant([4, 1, 1])
    strides = constant_op.constant([1, 1, 1])
    foo = array_ops.strided_slice_grad(varshape, begin, end, strides, var2)
    self.evaluate(var2.initializer)
    self.evaluate(varshape.initializer)
    self.evaluate(foo)

  def testInt64Shape(self):
    # [FUNCTION DOC]
    # Calls the strided slice gradient op with int64 shape, begin, end and strides and evaluates it;
    # passes when no error occurs.
    original_dy = array_ops.reshape(
        math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32), shape=(4, 1, 1))
    original_shape = constant_op.constant([6, 4, 4], dtype=dtypes.int64)
    begin = constant_op.constant([0, 0, 0], dtype=dtypes.int64)
    end = constant_op.constant([4, 1, 1], dtype=dtypes.int64)
    strides = constant_op.constant([1, 1, 1], dtype=dtypes.int64)
    dx = array_ops.strided_slice_grad(original_shape, begin, end, strides,
                                      original_dy)
    self.evaluate(dx)

  def testMixedIndexTypes(self):
    # [FUNCTION DOC]
    # Calls the strided slice gradient op with int64 shape and int32 begin but int64 end and strides;
    # asserts a type error or invalid-argument error is raised.
    original_dy = array_ops.reshape(
        math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32), shape=(4, 1, 1))
    original_shape = constant_op.constant([6, 4, 4], dtype=dtypes.int64)
    begin = constant_op.constant([0, 0, 0], dtype=dtypes.int32)
    end = constant_op.constant([4, 1, 1], dtype=dtypes.int64)
    strides = constant_op.constant([1, 1, 1], dtype=dtypes.int64)
    with self.assertRaises((TypeError, errors_impl.InvalidArgumentError)):
      dx = array_ops.strided_slice_grad(original_shape, begin, end, strides,
                                        original_dy)
      self.evaluate(dx)


class BenchmarkSlice(object):
  # [CLASS DOC]
  # HELPER: tiny indexing wrapper used by the slice benchmarks. Holds a tensor and forwards the
  # index operator to it so slice syntax can be used.

  def __init__(self, tensor):
    # [FUNCTION DOC]
    # Stores the wrapped tensor as an attribute. Returns nothing.
    self.tensor = tensor

  def __getitem__(self, x):
    # [FUNCTION DOC]
    # Index operator: returns the wrapped tensor indexed with the given spec.
    return self.tensor[x]


class StridedSliceBenchmark(test_lib.Benchmark):
  # [CLASS DOC]
  # Benchmark class timing strided slice versus slice on a large 256 by 256 by 256 float variable.
  # Subclass of the TensorFlow benchmark base class; benchmark methods report wall time.
  """Benchmark new strided slice operation on non-trivial case."""

  def run_and_time(self, slice_op):
    # [FUNCTION DOC]
    # HELPER: timing helper for the benchmarks in this class.
    # Input: slice_op (the operation to time). Initializes global variables, runs 10 warm-up
    # evaluations, then times 1000 evaluations and reports a benchmark with iteration count 1000 and
    # average wall time per iteration. Returns nothing.
    self.evaluate(variables.global_variables_initializer())
    for _ in range(10):
      _ = self.evaluate(slice_op)
    iters = 1000
    t0 = time.time()
    for _ in range(iters):
      self.evaluate(slice_op)
    t1 = time.time()
    self.report_benchmark(iters=iters, wall_time=(t1 - t0) / 1000.0)

  def make_variable(self):
    # [FUNCTION DOC]
    # HELPER: creates the large benchmark variable.
    # Returns a float32 variable of shape 256 cubed filled with a linear sequence from 1 up to the
    # element count.
    n = 256
    shape = (n, n, n)
    items = n**3
    var = variables.Variable(
        array_ops.reshape(math_ops.linspace(1., float(items), items), shape),
        dtype=dtypes.float32)
    return var

  def benchmark_strided_slice_skip(self):
    # [FUNCTION DOC]
    # In a new session builds the large variable and times a strided slice that skips elements on
    # axes 0 and 2 (step 2).
    with session.Session():
      var = self.make_variable()
      helper = BenchmarkSlice(var)
      slice_op = helper[::2, ::1, ::2]
      self.run_and_time(slice_op)

  def benchmark_strided_slice_easy(self):
    # [FUNCTION DOC]
    # In a new session builds the large variable and times an easy strided slice starting at 3 on
    # every axis, through the wrapper object.
    with session.Session():
      var = self.make_variable()
      helper = BenchmarkSlice(var)
      slice_op = helper[3::1, 3::1, 3::1]
      self.run_and_time(slice_op)

  def benchmark_slice_easy(self):
    # [FUNCTION DOC]
    # In a new session builds the large variable and times indexing it directly with a start of 3 on
    # every axis (compare with the wrapper-based variant).
    with session.Session():
      var = self.make_variable()
      slice_op = var[3::1, 3::1, 3::1]
      self.run_and_time(slice_op)


class StridedSliceAssignChecker(object):
  # [CLASS DOC]
  # HELPER: test utility checking slice assignment against numpy.
  # Constructed with the test case, source data, tensor dtype (default float32) and whether to use
  # resource variables. Assigning through index syntax performs the assignment and compares.
  # Holds the dtype, test case, resource flag, numpy array and constant tensor.

  def __init__(self, test, x, tensor_type=dtypes.float32, use_resource=False):
    # [FUNCTION DOC]
    # Stores the dtype, test case and resource flag; builds the numpy array from data cast to the
    # dtype (complex types get nonzero imaginary part) and the constant tensor x. Returns nothing.
    self.tensor_type = tensor_type
    self.test = test
    self._use_resource = use_resource

    self.x_np = np.array(x).astype(tensor_type.as_numpy_dtype)
    # Give the value a non-zero imaginary component for complex types.
    if tensor_type.is_complex:
      self.x_np -= 1j * self.x_np
    self.x = constant_op.constant(self.x_np, dtype=tensor_type)

  def __setitem__(self, index, value):
    # [FUNCTION DOC]
    # Assignment operator. Casts the value to the dtype (complex gets nonzero imaginary part), enables
    # soft device placement globally, creates a variable (resource variable if requested) from the
    # tensor on GPU-capable placement, initializes it, assigns the value into the indexed slice
    # with the variable method and again with the state assign function, evaluating both. Asserts both
    # results equal the numpy array with the same assignment applied. Returns nothing.
    value = np.array(value).astype(self.tensor_type.as_numpy_dtype)
    # Give the value a non-zero imaginary component for complex types.
    if self.tensor_type.is_complex:
      value -= 1j * value

    config.set_soft_device_placement(True)
    with test_util.device(use_gpu=True):
      if self._use_resource:
        var = resource_variable_ops.ResourceVariable(self.x)
      else:
        var = variables.Variable(self.x)
      self.test.evaluate(var.initializer)
      val = self.test.evaluate(var[index].assign(value))
      # val_copy is used to check that tf.compat.v1.assign works equivalently
      # to the assign method above.
      val_copy = self.test.evaluate(state_ops.assign(var[index], value))
      valnp = np.copy(self.x_np)
      valnp[index] = np.array(value)
      self.test.assertAllEqual(val, valnp)
      self.test.assertAllEqual(val_copy, valnp)


class SliceAssignTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for assigning into slices of variables: success cases for several dtypes with and
  # without broadcasting, resource variables, type errors and tensor strided slice update op
  # (forwarding and gradients). Parameterized-test capable.

  def testInvalidSlice(self):
    # [FUNCTION DOC]
    # Assigns into the slice of a constant tensor and asserts an attribute error saying there is no
    # assign attribute.
    foo = constant_op.constant([1, 2, 3])
    with self.assertRaisesRegex(AttributeError, "no attribute 'assign'"):
      bar = foo[:2].assign(constant_op.constant([1, 2]))
      self.evaluate(bar)

  def doTestSliceAssign(self, use_resource):
    # [FUNCTION DOC]
    # HELPER: shared body of the slice assignment tests. Input use_resource (use resource variables).
    # For each type in the module data type list (sub-tests) uses the assign checker on a 2 by 3
    # matrix to try whole assignment, trivial shapes, shrink, new axis, combinations, non-unit
    # strides and degenerate intervals; then assigns to a scalar via empty index, ellipsis and new axis.
    for dtype in STRIDED_SLICE_TYPES:
      with self.subTest(dtype=dtype):
        checker = StridedSliceAssignChecker(
            self, [[1, 2, 3], [4, 5, 6]],
            use_resource=use_resource,
            tensor_type=dtype)
        # Check if equal
        checker[:] = [[10, 20, 30], [40, 50, 60]]
        # Check trivial (1,1) shape tensor
        checker[1:2, 1:2] = [[66]]
        # shrinks shape changes
        checker[1:2, 1] = [66]
        checker[1, 1:2] = [66]
        checker[1, 1] = 66
        # newaxis shape changes
        checker[:, None, :] = [[[10, 20, 30]], [[40, 50, 50]]]
        # shrink and newaxis
        checker[None, None, 0, 0:1] = [[[99]]]
        # Non unit strides
        checker[::1, ::-2] = [[3, 33], [4, 44]]
        # degenerate interval
        checker[8:10, 0] = []
        checker[8:10, 8:10] = [[]]
    # Assign vector to scalar (rank-0) using newaxis
    checker2 = StridedSliceAssignChecker(self, 222)
    checker2[()] = 6  # no indices
    checker2[...] = 6  # ellipsis
    checker2[None] = [6]  # new axis

  def doTestSliceAssignWithBroadcasting(self, use_resource):
    # [FUNCTION DOC]
    # HELPER: shared body of the broadcasting assignment tests. Input use_resource. For each type in
    # the module data type list (sub-tests) uses the assign checker with values that need broadcasting
    # (full, trivial, shrink, new axis, combined, non-unit strides, degenerate interval).
    for dtype in STRIDED_SLICE_TYPES:
      with self.subTest(dtype=dtype):
        checker = StridedSliceAssignChecker(
            self, [[1, 2, 3], [4, 5, 6]],
            use_resource=use_resource,
            tensor_type=dtype)
        # Broadcast to full LHS.
        checker[:] = [[40, 50, 60]]
        # Assign a trivial (1,1) tensor.
        checker[1:2, 1:2] = 66
        # Broadcast with shrink axis shape changes.
        checker[1:2, 1] = 66
        checker[1, 1:2] = 66
        # Broadcast with newaxis shape changes.
        checker[:, None, :] = [10, 20, 30]
        # Broadcast with both shrink and newaxis.
        checker[None, None, 0, 0:1] = 99
        # Broadcast with non-unit strides.
        checker[::1, ::-2] = [[4, 44]]
        # Broadcast a degenerate interval.
        checker[8:10, 8:10] = []

  @test_util.disable_xla("b/123559667")
  def testSliceAssign(self):
    # [FUNCTION DOC]
    # XLA disabled. Runs the slice assignment helper with ordinary variables.
    self.doTestSliceAssign(use_resource=False)

  @test_util.disable_xla("b/123559667")
  def testSliceAssignWithBroadcasting(self):
    # [FUNCTION DOC]
    # XLA disabled. Runs the broadcasting slice assignment helper with ordinary variables.
    self.doTestSliceAssignWithBroadcasting(use_resource=False)

  @test_util.disable_xla("b/123559667")
  def testSliceAssignResource(self):
    # [FUNCTION DOC]
    # XLA disabled. Runs the slice assignment helper with resource variables.
    self.doTestSliceAssign(use_resource=True)

  def testTypeError(self):
    # [FUNCTION DOC]
    # Assigns int8 and int64 values into a slice of an int32 reference-style variable and asserts a
    # value or type error each time (no implicit dtype conversion).
    init_val = constant_op.constant([1, 2], dtype=dtypes.int32)
    too_small_val = constant_op.constant([3, 4], dtype=dtypes.int8)
    too_large_val = constant_op.constant([3, 4], dtype=dtypes.int64)
    v = variable_v1.VariableV1(init_val)
    with self.assertRaises((ValueError, TypeError)):
      self.evaluate(v[:].assign(too_small_val))
    with self.assertRaises((ValueError, TypeError)):
      self.evaluate(v[:].assign(too_large_val))

  def testTypeErrorResource(self):
    # [FUNCTION DOC]
    # Same as testTypeError for a resource variable: assigning int64 or int8 values into an int32
    # resource variable slice must raise a value error.
    init_val = constant_op.constant([1, 2], dtype=dtypes.int32)
    too_small_val = constant_op.constant([3, 4], dtype=dtypes.int8)
    too_large_val = constant_op.constant([3, 4], dtype=dtypes.int64)
    v = resource_variable_ops.ResourceVariable(init_val)
    self.evaluate(v.initializer)
    with self.assertRaises(ValueError):
      self.evaluate(v[:].assign(too_large_val))
    with self.assertRaises(ValueError):
      self.evaluate(v[:].assign(too_small_val))

  @test_util.disable_xla("b/123559667")
  @test_util.run_in_graph_and_eager_modes
  def testTensorStridedSliceUpdateWithInputForward(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes, XLA disabled. A graph function adds one to its input and updates
    # element 0 using the tensor strided slice update op, so the input buffer can be forwarded. Asserts
    # the result for a zero vector is 0 then 1.
    """Tests tensor_strided_slice_update with input-forwarding taking effect."""
    @def_function.function
    def assign(x):
      # [FUNCTION DOC]
      # Nested graph function: adds one to the input then replaces its first element with 0 using the
      # generated tensor strided slice update op (begin 0, end 1, stride 1); returns the result.
      y = x + 1
      return gen_array_ops.tensor_strided_slice_update(y, [0], [1], [1], [0])
    self.assertAllEqual([0, 1], self.evaluate(assign(array_ops.zeros([2]))))

  @test_util.run_in_graph_and_eager_modes
  def testTensorStridedSliceUpdateWithInputForwardInt32(self):
    # [FUNCTION DOC]
    # Same as the forwarding test but with an int32 zero vector input; asserts the result is 0, 1.
    """Tests tensor_strided_slice_update with int32."""
    @def_function.function
    def assign(x):
      # [FUNCTION DOC]
      # Nested graph function: adds one to the input then replaces its first element with 0 using the
      # generated tensor strided slice update op; returns the result.
      y = x + 1
      return gen_array_ops.tensor_strided_slice_update(y, [0], [1], [1], [0])
    self.assertAllEqual(
        [0, 1], self.evaluate(assign(array_ops.zeros([2], dtype=dtypes.int32))))

  @test_util.disable_xla("b/123559667")
  @test_util.run_in_graph_and_eager_modes
  def testTensorStridedSliceUpdateNoInputForward(self):
    # [FUNCTION DOC]
    # Runs both modes, XLA disabled. Updates the first element of a shifted tensor while the original
    # shifted tensor is still needed later (so the buffer cannot be forwarded) and asserts the sum of
    # the shifted tensor and the update result is 1.6 and 2.6.
    """Tests tensor_strided_slice_update with no input-forwarding."""
    x = constant_op.constant([0.2, 0.3])
    y = x + 1
    # y's buffer won't be forwarded to z because y and z will be alive at the
    # same time later.
    z = gen_array_ops.tensor_strided_slice_update(y, [0], [1], [1], [0.4])
    ans = y + z
    self.assertAllClose([1.6, 2.6], self.evaluate(ans))

  @test_util.disable_xla("b/123559667")
  def testTensorStridedSliceUpdateGradSimple(self):
    # [FUNCTION DOC]
    # XLA disabled. Updates one element of a vector with the update op under a gradient tape and asserts
    # gradients for the original (0 at updated position, 3 elsewhere) and for the update (2).
    original = constant_op.constant([0.2, 0.3])
    updates = constant_op.constant([0.4])
    with backprop.GradientTape() as tape:
      tape.watch([original, updates])
      updated = gen_array_ops.tensor_strided_slice_update(
          original, [0], [1], [1], updates)
    d1, d2 = tape.gradient(updated, [original, updates],
                           output_gradients=constant_op.constant([2.0, 3.0]))
    self.assertAllClose([0.0, 3.0], d1)
    self.assertAllClose([2.0], d2)

  @parameterized.named_parameters(
      ("_%s" % i, *args) for i, args in enumerate([  # pylint:disable=g-complex-comprehension
          ([2, 5], [0, 1], [1, 0], [1, 2], [2], 0, 2, 0, 0, 1),
          ([4], [5], [3], [1], [3], 1, 0, 0, 0, 0),
          ([2, 2, 3, 2], [0, 0, 1], [1, 0, 2], [1, 0, 1], [2, 3], 0, 0, 2, 0, 5)
      ]))
  @test_util.disable_xla("b/123559667")
  def testTensorStridedSliceUpdateGrad(
      self, shape, begin, end, strides, updates_shape, *args):
    # [FUNCTION DOC]
    # Parameterized over three named cases giving shape, begin, end, strides, update shape and extra
    # mask arguments; XLA disabled. In a cached session numerically checks that the theoretical and
    # numerical gradients of the update op agree.
    with self.cached_session():
      def f(a, b):
        # [FUNCTION DOC]
        # Nested function of two arguments: applies the tensor strided slice update op to the first
        # argument with the captured begin, end, strides and extra mask arguments, using the second as
        # updates; returns the result.
        return gen_array_ops.tensor_strided_slice_update(
            a, begin, end, strides, b, *args)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [array_ops.zeros(shape), array_ops.ones(updates_shape)], delta=1.0)
      self.assertAllClose(theoretical, numerical)

  @parameterized.named_parameters(("_%s" % i, *args) for i, args in enumerate([  # pylint:disable=g-complex-comprehension
      ([2, 5], [0, 1], [1, 0], [1, 2], [1], 0, 2, 0, 0,
       1), ([4], [5], [3], [1], [], 1, 0, 0, 0, 0),
      ([2, 2, 3, 2], [0, 0, 1], [1, 0, 2], [1, 0, 1], [2, 1], 0, 0, 2, 0, 5)
  ]))
  @test_util.disable_xla("b/123559667")
  def testTensorStridedSliceUpdateWithBroadcastingGrad(self, shape, begin, end,
                                                       strides, updates_shape,
                                                       *args):
    # [FUNCTION DOC]
    # Same as the grad test but the update shapes require broadcasting (including empty updates);
    # parameterized over three named cases, XLA disabled. Asserts theoretical and numerical gradients
    # agree in a cached session.
    with self.cached_session():

      def f(a, b):
        # [FUNCTION DOC]
        # Nested function of two arguments: applies the tensor strided slice update op to the first
        # argument with captured begin, end, strides and extra arguments using the second as updates.
        return gen_array_ops.tensor_strided_slice_update(
            a, begin, end, strides, b, *args)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [array_ops.zeros(shape),
              array_ops.ones(updates_shape)], delta=1.0)
      self.assertAllClose(theoretical, numerical)


class ShapeSizeRankTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the shape, size and rank operations on dense and sparse inputs.

  @test_util.run_in_graph_and_eager_modes
  def testDenseShape(self):
    # [FUNCTION DOC]
    # Both modes. For a nested list and for a constant, asserts shape is 2 by 2, size 4 and rank 2.
    t_value = [[0, 42], [24, 0]]
    self.assertAllEqual((2, 2), self.evaluate(array_ops.shape(t_value)))
    self.assertEqual(4, self.evaluate(array_ops.size(t_value)))
    self.assertEqual(2, self.evaluate(array_ops.rank(t_value)))

    t = constant_op.constant(t_value)
    self.assertAllEqual((2, 2), self.evaluate(array_ops.shape(t)))
    self.assertEqual(4, self.evaluate(array_ops.size(t)))
    self.assertEqual(2, self.evaluate(array_ops.rank(t)))

  @test_util.run_in_graph_and_eager_modes
  def testSparseShape(self):
    # [FUNCTION DOC]
    # Both modes. For a sparse tensor value and a sparse tensor, asserts dense shape 2 by 2, size 4
    # and rank 2.
    sp_value = sparse_tensor.SparseTensorValue(
        indices=((0, 1), (1, 0)), values=(42, 24), dense_shape=(2, 2))
    self.assertAllEqual((2, 2), self.evaluate(array_ops.shape(sp_value)))
    self.assertEqual(4, self.evaluate(array_ops.size(sp_value)))
    self.assertEqual(2, self.evaluate(array_ops.rank(sp_value)))

    sp = sparse_tensor.SparseTensor.from_value(sp_value)
    self.assertAllEqual((2, 2), self.evaluate(array_ops.shape(sp)))
    self.assertEqual(4, self.evaluate(array_ops.size(sp)))
    self.assertEqual(2, self.evaluate(array_ops.rank(sp)))

  @test_util.run_in_graph_and_eager_modes
  def testSizeDtype(self):
    # [FUNCTION DOC]
    # Both modes. Asserts size defaults to an int32 result and an int64 result when out type is int64.
    tensor = [1]
    self.assertEqual(dtypes.int32, self.evaluate(array_ops.size(tensor)).dtype)
    self.assertEqual(
        dtypes.int64,
        self.evaluate(array_ops.size(tensor, out_type=dtypes.int64)).dtype)


class SequenceMaskTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the sequence mask operation: shapes, dtypes, optional maximum length and errors.

  def testExceptions(self):
    # [FUNCTION DOC]
    # Passes a non-scalar maxlen and asserts a value error saying maxlen must be scalar.
    with self.cached_session():
      with self.assertRaisesRegex(ValueError, "`maxlen` must be scalar"):
        array_ops.sequence_mask([10, 20], [10, 20])

  def testOneDimensionalWithMaxlen(self):
    # [FUNCTION DOC]
    # Masks lengths 1, 3, 2 with maxlen 5; asserts the shape 3 by 5 and expected boolean rows.
    res = array_ops.sequence_mask(constant_op.constant([1, 3, 2]), 5)
    self.assertAllEqual(res.get_shape(), [3, 5])
    self.assertAllEqual(
        res,
        [[True, False, False, False, False], [True, True, True, False, False],
         [True, True, False, False, False]])

  def testOneDimensionalDtypeWithoutMaxlen(self):
    # [FUNCTION DOC]
    # Masks lengths 0, 1, 4 with output dtype float32 and default maxlen; asserts shape 3 by 4 and
    # expected float rows (maxlen defaults to the largest length).
    # test dtype and default maxlen:
    res = array_ops.sequence_mask(
        constant_op.constant([0, 1, 4]), dtype=dtypes.float32)
    self.assertAllEqual(res.get_shape().as_list(), [3, 4])
    self.assertAllEqual(
        res, [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

  def testOneDimensionalWithoutMaxlen(self):
    # [FUNCTION DOC]
    # Masks lengths 0, 1, 4 with default maxlen and bool output; asserts shape 3 by 4 and rows.
    res = array_ops.sequence_mask(constant_op.constant([0, 1, 4]))
    self.assertAllEqual(res.get_shape().as_list(), [3, 4])
    self.assertAllEqual(res,
                        [[False, False, False, False],
                         [True, False, False, False], [True, True, True, True]])

  def testTwoDimensional(self):
    # [FUNCTION DOC]
    # Masks a 2-D lengths tensor: first with maxlen 5 (shape 1 by 3 by 5, bool) then a 2 by 3 lengths
    # tensor with float32 dtype and default maxlen (shape 2 by 3 by 4), asserting values.
    res = array_ops.sequence_mask(constant_op.constant([[1, 3, 2]]), 5)
    self.assertAllEqual(res.get_shape(), [1, 3, 5])
    self.assertAllEqual(
        res,
        [[[True, False, False, False, False], [True, True, True, False, False],
          [True, True, False, False, False]]])

    # test dtype and default maxlen:
    res = array_ops.sequence_mask(
        constant_op.constant([[0, 1, 4], [1, 2, 3]]), dtype=dtypes.float32)
    self.assertAllEqual(res.get_shape().as_list(), [2, 3, 4])
    self.assertAllEqual(
        res,
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
         [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]])

  def testDtypes(self):
    # [FUNCTION DOC]
    # Checks all four combinations of int32/int64 lengths and int32/int64 maxlen produce the same
    # expected boolean mask.

    def check_dtypes(lengths_dtype, maxlen_dtype):
      # [FUNCTION DOC]
      # Nested checker with inputs lengths_dtype and maxlen_dtype: masks lengths 1, 3, 2 with maxlen 5
      # using these dtypes and asserts shape 3 by 5 and expected boolean values.
      res = array_ops.sequence_mask(
          constant_op.constant([1, 3, 2], dtype=lengths_dtype),
          constant_op.constant(5, dtype=maxlen_dtype))
      self.assertAllEqual(res.get_shape(), [3, 5])
      self.assertAllEqual(
          res,
          [[True, False, False, False, False], [True, True, True, False, False],
           [True, True, False, False, False]])

    check_dtypes(dtypes.int32, dtypes.int32)
    check_dtypes(dtypes.int32, dtypes.int64)
    check_dtypes(dtypes.int64, dtypes.int32)
    check_dtypes(dtypes.int64, dtypes.int64)

  def testOutputDtype(self):
    # [FUNCTION DOC]
    # Checks the output dtype option accepts TensorFlow dtypes, strings and numpy types for bool,
    # int32, float32, int64 and float64, comparing against the cast expected mask.

    def check_output_dtype(output_dtype):
      # [FUNCTION DOC]
      # Nested checker with input output_dtype: evaluates the sequence mask of lengths 1, 3, 2 with
      # maxlen 5 and that dtype and asserts it equals the boolean mask cast to the dtype.
      res = self.evaluate(
          array_ops.sequence_mask(
              constant_op.constant([1, 3, 2], dtype=dtypes.int32),
              constant_op.constant(5, dtype=dtypes.int32),
              dtype=output_dtype))
      self.assertAllEqual(
          res,
          self.evaluate(
              math_ops.cast([[True, False, False, False, False],
                             [True, True, True, False, False],
                             [True, True, False, False, False]], output_dtype)))

    check_output_dtype(dtypes.bool)
    check_output_dtype("bool")
    check_output_dtype(np.bool_)
    check_output_dtype(dtypes.int32)
    check_output_dtype("int32")
    check_output_dtype(np.int32)
    check_output_dtype(dtypes.float32)
    check_output_dtype("float32")
    check_output_dtype(np.float32)
    check_output_dtype(dtypes.int64)
    check_output_dtype("float64")
    check_output_dtype(np.float64)


class ConcatSliceResourceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case verifying resource handles survive stacking and slicing.

  @test_util.run_in_graph_and_eager_modes
  def testConcatSlice(self):
    # [FUNCTION DOC]
    # Both modes. Stacks two stub resource handles, slices the second, creates a resource through the
    # slice and asserts that creating the same resource again via the original handle raises an
    # already-exists error. Side effect: creates a test resource.
    r1 = test_ops.stub_resource_handle_op(container="a", shared_name="b")
    r2 = test_ops.stub_resource_handle_op(container="a", shared_name="c")
    c = array_ops_stack.stack([r1, r2])
    s = array_ops.strided_slice(c, [1], [2])
    self.evaluate(test_ops.resource_create_op(s))
    with self.assertRaises(errors.AlreadyExistsError):
      self.evaluate(test_ops.resource_create_op(r2))


class IdentityTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the identity operation, including device placement and variable inputs.

  @test_util.run_gpu_only
  def testEagerIdentity(self):
    # [FUNCTION DOC]
    # GPU-only test in eager mode. Alternates identity ops under forced GPU and CPU placement and
    # checks values stay equal and the output's device string contains the requested device.
    with context.eager_mode():

      def _test(x, y, device):
        # [FUNCTION DOC]
        # HELPER: nested check helper. Inputs x, y tensors and device substring; asserts the numpy values
        # are equal and the device name of y contains the substring.
        self.assertAllEqual(x.numpy(), y.numpy())
        self.assertIn(device, y.device.lower())

      with test_util.force_gpu():
        a = constant_op.constant([[2], [3]], dtype=dtypes.float32)
      with test_util.force_gpu():
        b = array_ops.identity(a)
        _test(a, b, "gpu")
      with test_util.force_cpu():
        c = array_ops.identity(b)
        _test(b, c, "cpu")
      with test_util.force_cpu():
        d = array_ops.identity(c)
        _test(c, d, "cpu")
      with test_util.force_gpu():
        e = array_ops.identity(d)
        _test(d, e, "gpu")

  def testIdentityVariable(self):
    # [FUNCTION DOC]
    # Applies identity to a resource variable and asserts the result is a tensor with the same value.
    v = resource_variable_ops.ResourceVariable(1.0)
    self.evaluate(v.initializer)
    result = array_ops.identity(v)
    self.assertIsInstance(result, tensor_lib.Tensor)
    self.assertAllEqual(result, v)


class PadTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the pad op and mirror-pad gradient op.

  def testEager(self):
    # [FUNCTION DOC]
    # Eager mode: pads a 2 by 3 int constant with one row and two columns of constant padding on each
    # side and asserts the 4 by 7 result with zero padding.
    with context.eager_mode():
      t = constant_op.constant([[1, 2, 3], [4, 5, 6]])
      paddings = constant_op.constant([[
          1,
          1,
      ], [2, 2]])
      padded = array_ops.pad(t, paddings, "CONSTANT")
      self.assertAllEqual(padded.numpy(),
                          [[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 0, 0],
                           [0, 0, 4, 5, 6, 0, 0], [0, 0, 0, 0, 0, 0, 0]])

  # b/246325518: Bad shape size. Explicitly testing different execution paths.
  def testInvalidMirrorPadGradEagerMode(self):
    # [FUNCTION DOC]
    # Regression test for bad shape size. In eager mode calls the mirror pad gradient op with huge
    # paddings and asserts that an exception is raised.
    with context.eager_mode():
      with self.assertRaises(Exception):
        gen_array_ops.MirrorPadGrad(
            input=[1], paddings=[[0x77f00000, 0xa000000]], mode="REFLECT")

  # b/246325518: Bad shape size. Explicitly testing different execution paths.
  def testInvalidMirrorPadGradGraphMode(self):
    # [FUNCTION DOC]
    # Same regression in graph mode: builds the gradient op with huge paddings and evaluates it,
    # expecting some exception.
    with context.graph_mode():
      with self.assertRaises(Exception):
        result = gen_array_ops.MirrorPadGrad(
            input=[1], paddings=[[0x77f00000, 0xa000000]], mode="REFLECT")
        self.evaluate(result)

  def testSymmetricMirrorPadGrad(self):
    # [FUNCTION DOC]
    # Applies the mirror pad gradient in symmetric mode to a broadcast arange tensor and asserts the
    # expected folded sums with shape 1 by 2 by 1 by 3.
    t = np.broadcast_to(np.arange(0, 7), (3, 2, 1, 7))
    paddings = constant_op.constant([
        [1, 1],
        [0, 0],
        [0, 0],
        [2, 2],
    ])
    expected = np.broadcast_to(np.array([9, 27, 27]), (1, 2, 1, 3))
    result = gen_array_ops.mirror_pad_grad(t, paddings, "SYMMETRIC")
    self.assertAllEqual(result, expected)

  def testReflectMirrorPadGrad(self):
    # [FUNCTION DOC]
    # Applies the mirror pad gradient in reflect mode to a broadcast arange tensor and asserts the
    # expected folded values of shape 1 by 2 by 3 by 1.
    t = np.broadcast_to(np.reshape(np.arange(0, 7), (7, 1)), (1, 4, 7, 1))
    paddings = constant_op.constant([
        [0, 0],
        [1, 1],
        [2, 2],
        [0, 0],
    ])
    expected = np.broadcast_to(
        np.reshape(np.array([16, 18, 8]), (3, 1)), (1, 2, 3, 1))
    result = gen_array_ops.mirror_pad_grad(t, paddings, "REFLECT")
    self.assertAllEqual(result, expected)


class InvertPermutationTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the invert permutation op.

  def testInvertPermutation(self):
    # [FUNCTION DOC]
    # For int32 and int64 (sub-tests), inverts permutation 3, 4, 0, 2, 1; asserts shape 5 and the
    # inverse 2, 4, 3, 0, 1.
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.subTest(dtype=dtype, use_gpu=True):
        x = constant_op.constant([3, 4, 0, 2, 1], dtype=dtype)
        y = array_ops.invert_permutation(x)
        self.assertAllEqual(y.get_shape(), [5])
        self.assertAllEqual(y, [2, 4, 3, 0, 1])


class UnravelIndexTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the unravel index op.

  # TODO(b/73086570): Reenable test.
  @unittest.skip("Test does not pass internally.")
  def testUnravelIndex(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is unconditionally skipped (decorator says it does not pass internally),
    # so its assertions never run.
    # Would check unravel index for scalar, one-element vector and vector indices with dims and both
    # integer dtypes, comparing against hand-computed coordinates.
    with self.cached_session():
      for dtype in [dtypes.int32, dtypes.int64]:
        with self.subTest(dtype=dtype):
          indices_1 = constant_op.constant(1621, dtype=dtype)
          dims_1 = constant_op.constant([6, 7, 8, 9], dtype=dtype)
          out_1 = array_ops.unravel_index(indices_1, dims_1)
          self.assertAllEqual(out_1, [3, 1, 4, 1])

          indices_2 = constant_op.constant([1621], dtype=dtype)
          dims_2 = constant_op.constant([6, 7, 8, 9], dtype=dtype)
          out_2 = array_ops.unravel_index(indices_2, dims_2)
          self.assertAllEqual(out_2, [[3], [1], [4], [1]])

          indices_3 = constant_op.constant([22, 41, 37], dtype=dtype)
          dims_3 = constant_op.constant([7, 6], dtype=dtype)
          out_3 = array_ops.unravel_index(indices_3, dims_3)
          self.assertAllEqual(out_3, [[3, 6, 6], [4, 5, 1]])

  # Test case for GitHub issue 40204.
  def testUnravelIndexZeroDim(self):
    # [FUNCTION DOC]
    # Regression for a reported issue. For int32 and int64 asserts that a dims vector containing zero
    # raises an invalid-argument error saying dims cannot contain a dim of zero.
    with self.cached_session():
      for dtype in [dtypes.int32, dtypes.int64]:
        with self.assertRaisesRegex(errors.InvalidArgumentError,
                                    "dims cannot contain a dim of zero"):
          indices = constant_op.constant([2, 5, 7], dtype=dtype)
          dims = constant_op.constant([3, 0], dtype=dtype)
          self.evaluate(array_ops.unravel_index(indices=indices, dims=dims))

  def testUnravelIndexIntegerOverflow(self):
    # [FUNCTION DOC]
    # For int32 and int64 uses dims whose product overflows and asserts an invalid-argument error
    # about integer overflow.
    with self.cached_session():
      for dtype in [dtypes.int32, dtypes.int64]:
        with self.assertRaisesRegex(
            errors.InvalidArgumentError,
            r"Input dims product is causing integer overflow"):
          indices = constant_op.constant(-0x100000, dtype=dtype)
          if dtype == dtypes.int32:
            value = 0x10000000
          else:
            value = 0x7FFFFFFFFFFFFFFF
          dims = constant_op.constant([value, value], dtype=dtype)
          self.evaluate(array_ops.unravel_index(indices=indices, dims=dims))


class GuaranteeConstOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the guarantee-const op.

  def testSimple(self):
    # [FUNCTION DOC]
    # Applies guarantee const to a constant 10 and asserts the evaluated value is 10.
    a = array_ops.constant(10)
    guarantee_a = array_ops.guarantee_const(a)
    self.assertEqual(10, self.evaluate(guarantee_a))

  def testVariables(self):
    # [FUNCTION DOC]
    # For non-resource and resource variables (sub-tests) creates a variable with initial value 10,
    # guarantees it constant, initializes and asserts the value 10. Side effect: creates named
    # variables in the default variable scope.
    for use_resource in [False, True]:
      with self.subTest(use_resource=use_resource):
        a = variable_scope.get_variable(
            "var_{}".format(use_resource), [],
            initializer=init_ops.constant_initializer(10.0),
            use_resource=use_resource)
        guarantee_a = array_ops.guarantee_const(a)
        self.evaluate(a.initializer)
        self.assertEqual(10.0, self.evaluate(guarantee_a))

  def testResourceRejection(self):
    # [FUNCTION DOC]
    # Creates a resource variable on CPU and asserts that applying guarantee const to its handle
    # raises an invalid-argument error saying it cannot be a resource variable.
    with ops.device("/cpu:0"):
      a = variable_scope.get_variable(
          "resource_var", [],
          initializer=init_ops.constant_initializer(10.0),
          use_resource=True)
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             "cannot be a resource variable"):
      guarantee_a = array_ops.guarantee_const(a.handle)
      self.evaluate(a.initializer)
      self.evaluate(guarantee_a)


class SnapshotOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the snapshot op.

  def testInvertPermutation(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name (apparently copied from the invert permutation test) does not match the
    # behavior: it tests the snapshot op, not invert permutation.
    # For int32, int64, float32 and float64 (sub-tests) snapshots 0..3 and asserts the values are
    # unchanged.
    for dtype in [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64]:
      with self.subTest(dtype=dtype, use_gpu=True):
        x = constant_op.constant([0, 1, 2, 3], dtype=dtype)
        y = gen_array_ops.snapshot(x)
        self.assertAllEqual(y, [0, 1, 2, 3])


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class QuantizeAndDequantizeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for quantize-and-dequantize ops: per-axis behavior, bad axes, gradient and invalid
  # axis errors. Both-modes and op-as-function decorators. Defines a private data helper.

  # Generates a tensor of the specified `shape` using values from `values`
  # scaled by (slice_idx + 1) along `axis` dimension.
  def _scale_per_slice(self, shape, axis, values):
    # [FUNCTION DOC]
    # HELPER: test data generator. Inputs: shape (list of sizes), axis (none or an axis) and values
    # (1-D numpy array). Returns an array of the given shape filled by repeating values cyclically;
    # if an axis is given, each slice along that axis is multiplied by its one-based index.
    # Note: repeats the values if the shape is larger than values.
    out = np.take(values, np.remainder(np.arange(np.prod(shape)),
                                       len(values))).reshape(shape)
    if axis is not None:
      scale_shape = [1] * len(shape)
      scale_shape[axis] = shape[axis]
      out *= np.arange(1, shape[axis] + 1).reshape(scale_shape)
    return out

  def testAxis(self):
    # [FUNCTION DOC]
    # For per-tensor (no axis) and axes 0..3 (sub-tests) on shape 2, 3, 4, 5 builds scaled inputs and
    # expected quantized values (signed 8-bit style with half-up rounding) and asserts the
    # quantize-and-dequantize op with unused min and max values and range not given matches. For
    # non-null axes also checks the negative axis equivalent (axis minus 4) with close values.
    shape = np.array([2, 3, 4, 5])
    values = np.array([-1, -0.5, 0, 0.3, 0.8, 0.555, 0.5], dtype=np.float32)
    quant_values = np.array(
        [-1, -0.5, 0, 38.0 / 128, 102.0 / 128, 71.0 / 128, 0.5],
        dtype=np.float32)
    for axis in [None, 0, 1, 2, 3]:
      with self.subTest(axis=axis):
        inputs = constant_op.constant(
            self._scale_per_slice(shape, axis, values))
        expected = self._scale_per_slice(shape, axis, quant_values)
        unused_minmax_value = 0 if axis is None else [0] * shape[axis]
        fake_quantized = self.evaluate(
            array_ops.quantize_and_dequantize_v2(
                inputs,
                unused_minmax_value,
                unused_minmax_value,
                range_given=False,
                round_mode="HALF_UP",
                axis=axis))
        self.assertAllEqual(fake_quantized, expected)
        if axis is not None:
          fake_quantized = self.evaluate(
              array_ops.quantize_and_dequantize_v2(
                  inputs,
                  unused_minmax_value,
                  unused_minmax_value,
                  range_given=False,
                  axis=(axis - 4)))
          self.assertAllClose(fake_quantized, expected)

  def testBadAxis(self):
    # [FUNCTION DOC]
    # Calls quantize-and-dequantize with axis 10 on a rank-1 input and asserts an invalid-argument,
    # value or unimplemented error (type differs by mode/XLA) whose message mentions the rank or
    # invalid axis.
    input_tensor = [2.5, 2.5]
    input_min = [0, 0]
    input_max = [1, 1]
    # When eager_op_as_function mode is enabled XLA auto-clustering kicks in.
    # XLA raises an UnimplementedError on invalid axis.
    error_message_pattern = (r"Shape must be at least rank 11 but is rank "
                             r"1|invalid axis")
    # TODO(b/171260356): Eager mode and graph mode throw different error types
    error = (errors.InvalidArgumentError, ValueError, errors.UnimplementedError)
    with self.assertRaisesRegex(error, error_message_pattern):
      self.evaluate(
          array_ops.quantize_and_dequantize_v2(
              input=input_tensor,
              input_min=input_min,
              input_max=input_max,
              axis=10))

  def testQuantizeDequantizeGrad(self):
    # [FUNCTION DOC]
    # Checks the numerical gradient of quantize and dequantize with a given range (min -10, max 0) on
    # a random 2 by 2 input in a cached session; asserts the theoretical gradient is all zeros of
    # shape 1 by 4 by 4.
    shape = (2, 2)
    max_threshold = 0
    min_threshold = -10
    input_value = np.random.rand(2, 2) * 40.0 - 20.0
    input_tensor = constant_op.constant(input_value, shape=shape,
                                        name="input_tensor")
    with self.cached_session():
      def f(a):
        # [FUNCTION DOC]
        # Nested function: applies quantize-and-dequantize to its argument with fixed min and max
        # thresholds and range given; returns the result.
        return array_ops.quantize_and_dequantize_v2(
            a,
            input_min=min_threshold,
            input_max=max_threshold,
            range_given=True)
      output_grad = gradient_checker_v2.compute_gradient(f, [input_tensor])
      self.assertAllClose(output_grad[0], np.zeros([1, 4, 4]))

  def testOutOfBoundAxis(self):
    # [FUNCTION DOC]
    # Quantizes a two-element vector then dequantizes with axis of the maximum 32-bit value and
    # asserts an error saying the axis must be less than the input dimension.
    input_tensor = constant_op.constant([1., 1.])
    input_min = [0]
    input_max = [1]
    q_input, _, _ = array_ops.quantize(input_tensor, 0, 1, dtypes.qint32)
    error = (errors.InvalidArgumentError, ValueError)
    with self.assertRaisesRegex(error,
                                r".*Axis must be less than input dimension.*"):
      self.evaluate(
          gen_array_ops.dequantize(
              input=q_input,
              min_range=input_min,
              max_range=input_max,
              axis=2**31 - 1))

  @test_util.run_v2_only
  def testInvalidAxis(self):
    # [FUNCTION DOC]
    # Only runs in v2 behavior. Defines four graph functions each calling a generated quantize op
    # variant (v2, v3, v4, v4 gradient) with axis 2147483647 and asserts each raises a value error
    # saying the axis cannot be at least the 32-bit maximum.

    @def_function.function
    def test_quantize_and_dequantize_v2():
      # [FUNCTION DOC]
      # Nested graph function calling the generated v2 op with fixed arguments and axis 2147483647.
      gen_array_ops.quantize_and_dequantize_v2(
          input=[2.5],
          input_min=[1.0],
          input_max=[10.0],
          signed_input=True,
          num_bits=1,
          range_given=True,
          round_mode="HALF_TO_EVEN",
          narrow_range=True,
          axis=0x7fffffff)

    @def_function.function
    def test_quantize_and_dequantize_v3():
      # [FUNCTION DOC]
      # Nested graph function calling the generated v3 op with fixed arguments and axis 2147483647.
      gen_array_ops.quantize_and_dequantize_v3(
          input=[2.5],
          input_min=[1.0],
          input_max=[10.0],
          num_bits=1,
          signed_input=True,
          range_given=True,
          narrow_range=True,
          axis=0x7fffffff)

    @def_function.function
    def test_quantize_and_dequantize_v4():
      # [FUNCTION DOC]
      # Nested graph function calling the generated v4 op with fixed arguments and axis 2147483647.
      gen_array_ops.quantize_and_dequantize_v4(
          input=[2.5],
          input_min=[1.0],
          input_max=[10.0],
          signed_input=True,
          num_bits=1,
          range_given=True,
          round_mode="HALF_TO_EVEN",
          narrow_range=True,
          axis=0x7fffffff)

    @def_function.function
    def test_quantize_and_dequantize_v4_grad():
      # [FUNCTION DOC]
      # Nested graph function calling the generated v4 gradient op with fixed arguments and axis
      # 2147483647.
      gen_array_ops.quantize_and_dequantize_v4_grad(
          gradients=[2.5],
          input=[2.5],
          input_min=[1.0],
          input_max=[10.0],
          axis=0x7fffffff)

    with self.assertRaisesRegex(
        ValueError, "Axis cannot be >= kint32max value, got 2147483647"):
      test_quantize_and_dequantize_v2()

    with self.assertRaisesRegex(
        ValueError, "Axis cannot be >= kint32max value, got 2147483647"):
      test_quantize_and_dequantize_v3()

    with self.assertRaisesRegex(
        ValueError, "Axis cannot be >= kint32max value, got 2147483647"):
      test_quantize_and_dequantize_v4()

    with self.assertRaisesRegex(
        ValueError, "Axis cannot be >= kint32max value, got 2147483647"):
      test_quantize_and_dequantize_v4_grad()


@test_util.run_all_in_graph_and_eager_modes
class SortedSearchTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the sorted search op (upper and lower bounds) against numpy for floats and
  # integers, random n-dimensional data, uneven shapes, empty sizes and large sizes. Both-modes
  # decorator.

  def testUpperBoundFloatHandCoded(self):
    # [FUNCTION DOC]
    # Searches a small sorted float vector with right side and compares with numpy.
    cdf = np.array([0, .2, .5, .6, .8, 1.], dtype=np.float32)
    arr = np.array([.04, .99, .53, .58, .31, .01, .79, .8, .21],
                   dtype=np.float32)
    result = np.searchsorted(cdf, arr, side="right")
    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))
    self.assertAllEqual(result, tf_result)

  def testUpperBoundFloatRandomNd(self):
    # [FUNCTION DOC]
    # For ranks 1 to 4 with size 7 builds cumulative-sum float32 sorted arrays and random queries,
    # runs right-side search and compares to row-wise numpy results.
    dim_size = 7
    for d in range(1, 5):
      shape = [dim_size] * d
      cdf = np.cumsum(
          np.random.uniform(size=shape).astype(np.float32), axis=(d - 1))
      arr = np.random.uniform(size=shape).astype(np.float32) * dim_size

      tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))

      cdf = cdf.reshape([-1, dim_size])
      arr = arr.reshape([-1, dim_size])
      result = np.zeros(arr.shape, dtype=np.int32)
      for i in range(dim_size**(d - 1)):
        result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="right")

      result = result.reshape(shape)

      self.assertAllEqual(result, tf_result)

  def testUpperBoundFloatUneven(self):
    # [FUNCTION DOC]
    # Batch 7, sorted arrays of 1000 and 47 query values (float32), right side; compares per-row
    # numpy results.
    batch_size = 7
    size_search_array = 1000
    size_values = 47
    cdf = np.cumsum(
        np.random.uniform(size=[batch_size, size_search_array]).astype(
            np.float32),
        axis=1)
    arr = np.random.uniform(size=[batch_size, size_values]).astype(
        np.float32) * size_search_array

    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))

    result = np.zeros(arr.shape, dtype=np.int32)
    for i in range(batch_size):
      result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="right")

    self.assertAllEqual(result, tf_result)

  def testLowerBoundFloatHandCoded(self):
    # [FUNCTION DOC]
    # Searches a small sorted float vector with left side and compares with numpy.
    cdf = np.array([0, .2, .5, .6, .8, 1.], dtype=np.float32)
    arr = np.array([.04, .99, .53, .58, .31, .01, .79, .8, .21],
                   dtype=np.float32)
    result = np.searchsorted(cdf, arr, side="left")
    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))
    self.assertAllEqual(result, tf_result)

  def testLowerBoundFloatRandomNd(self):
    # [FUNCTION DOC]
    # Same as the upper bound random test but with left side.
    dim_size = 7
    for d in range(1, 5):
      shape = [dim_size] * d
      cdf = np.cumsum(
          np.random.uniform(size=shape).astype(np.float32), axis=(d - 1))
      arr = np.random.uniform(size=shape).astype(np.float32) * dim_size

      tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))

      cdf = cdf.reshape([-1, dim_size])
      arr = arr.reshape([-1, dim_size])
      result = np.zeros(arr.shape, dtype=np.int32)
      for i in range(dim_size**(d - 1)):
        result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="left")

      result = result.reshape(shape)

      self.assertAllEqual(result, tf_result)

  def testLowerBoundFloatUneven(self):
    # [FUNCTION DOC]
    # Same as the upper bound uneven test but with left side.
    batch_size = 7
    size_search_array = 1000
    size_values = 47
    cdf = np.cumsum(
        np.random.uniform(size=[batch_size, size_search_array]).astype(
            np.float32),
        axis=1)
    arr = np.random.uniform(size=[batch_size, size_values]).astype(
        np.float32) * size_search_array

    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))

    result = np.zeros(arr.shape, dtype=np.int32)
    for i in range(batch_size):
      result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="left")

    self.assertAllEqual(result, tf_result)

  def testUpperBoundIntHandCoded(self):
    # [FUNCTION DOC]
    # Searches a small sorted int64 vector with right side and compares with numpy.
    cdf = np.array([0, 20, 50, 60, 80, 100], dtype=np.int64)
    arr = np.array([4, 99, 53, 58, 31, 1, 79, 8, 21], dtype=np.int64)
    result = np.searchsorted(cdf, arr, side="right")
    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))
    self.assertAllEqual(result, tf_result)

  def testUpperBoundIntRandomNd(self):
    # [FUNCTION DOC]
    # Random int64 data for ranks 1 to 4, right side, compared to row-wise numpy.
    dim_size = 7
    for d in range(1, 5):
      shape = [dim_size] * d
      cdf = np.cumsum(
          np.random.randint(low=0, high=10, size=shape).astype(np.int64),
          axis=(d - 1))
      arr = np.random.randint(
          low=0, high=10 * dim_size, size=shape).astype(np.int64)

      tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))

      cdf = cdf.reshape([-1, dim_size])
      arr = arr.reshape([-1, dim_size])
      result = np.zeros(arr.shape, dtype=np.int32)
      for i in range(dim_size**(d - 1)):
        result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="right")

      result = result.reshape(shape)

      self.assertAllEqual(result, tf_result)

  def testUpperBoundIntUneven(self):
    # [FUNCTION DOC]
    # Random int64 data batch 7 with 1000 sorted values and 47 queries, right side, compared to numpy.
    batch_size = 7
    size_search_array = 1000
    size_values = 47
    cdf = np.cumsum(
        np.random.randint(low=0, high=10,
                          size=[batch_size,
                                size_search_array]).astype(np.int64),
        axis=1)
    arr = np.random.randint(
        low=0, high=10 * size_search_array, size=[batch_size,
                                                  size_values]).astype(np.int64)

    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="right"))

    result = np.zeros(arr.shape, dtype=np.int32)
    for i in range(batch_size):
      result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="right")

    self.assertAllEqual(result, tf_result)

  def testLowerBoundIntHandCoded(self):
    # [FUNCTION DOC]
    # Searches a small sorted int64 vector with left side and compares with numpy.
    cdf = np.array([0, 20, 50, 60, 80, 100], dtype=np.int64)
    arr = np.array([4, 99, 53, 58, 31, 1, 79, 8, 21], dtype=np.int64)
    result = np.searchsorted(cdf, arr, side="left")
    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))
    self.assertAllEqual(result, tf_result)

  def testLowerBoundIntRandomNd(self):
    # [FUNCTION DOC]
    # Random int64 data for ranks 1 to 4, left side, compared to row-wise numpy.
    dim_size = 7
    for d in range(1, 5):
      shape = [dim_size] * d
      cdf = np.cumsum(
          np.random.randint(low=0, high=10, size=shape).astype(np.int64),
          axis=(d - 1))
      arr = np.random.randint(
          low=0, high=10 * dim_size, size=shape).astype(np.int64)

      tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))

      cdf = cdf.reshape([-1, dim_size])
      arr = arr.reshape([-1, dim_size])
      result = np.zeros(arr.shape, dtype=np.int32)
      for i in range(dim_size**(d - 1)):
        result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="left")

      result = result.reshape(shape)

      self.assertAllEqual(result, tf_result)

  def testLowerBoundIntUneven(self):
    # [FUNCTION DOC]
    # Random int64 data batch 7 with 1000 sorted values and 47 queries, left side, compared to numpy.
    batch_size = 7
    size_search_array = 1000
    size_values = 47
    cdf = np.cumsum(
        np.random.randint(low=0, high=10,
                          size=[batch_size,
                                size_search_array]).astype(np.int64),
        axis=1)
    arr = np.random.randint(
        low=0, high=10 * size_search_array, size=[batch_size,
                                                  size_values]).astype(np.int64)

    tf_result = self.evaluate(array_ops.searchsorted(cdf, arr, side="left"))

    result = np.zeros(arr.shape, dtype=np.int32)
    for i in range(batch_size):
      result[i, :] = np.searchsorted(cdf[i, :], arr[i, :], side="left")

    self.assertAllEqual(result, tf_result)

  def testZeroSequenceSize(self):
    # [FUNCTION DOC]
    # For both sides (sub-tests) searches a 2 by 0 sorted sequence with 2 by 3 values and int32 output
    # and asserts the result is all zeros of shape 2 by 3.
    dtype = dtypes.int32
    for side in ("left", "right"):
      with self.subTest(side=side):
        self.assertAllEqual(
            array_ops.searchsorted(
                array_ops.ones([2, 0]),
                array_ops.ones([2, 3]),
                side=side,
                out_type=dtype), array_ops.zeros([2, 3], dtype))

  def testZeroValueSize(self):
    # [FUNCTION DOC]
    # For both sides searches a 2 by 3 sequence with 2 by 0 values and asserts an empty 2 by 0 result.
    dtype = dtypes.int32
    for side in ("left", "right"):
      with self.subTest(side=side):
        self.assertAllEqual(
            array_ops.searchsorted(
                array_ops.ones([2, 3]),
                array_ops.ones([2, 0]),
                side=side,
                out_type=dtype), array_ops.zeros([2, 0], dtype))

  def testZeroInputSize(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this body is identical to testZeroSequenceSize (same empty 2 by 0 sorted sequence
    # and 2 by 3 values), so it duplicates that test instead of covering a distinct case.
    # For both sides asserts a search of an empty sequence returns zeros of shape 2 by 3.
    dtype = dtypes.int32
    for side in ("left", "right"):
      with self.subTest(side=side):
        self.assertAllEqual(
            array_ops.searchsorted(
                array_ops.ones([2, 0]),
                array_ops.ones([2, 3]),
                side=side,
                out_type=dtype), array_ops.zeros([2, 3], dtype))

  def testInt64(self):
    # [FUNCTION DOC]
    # Traces a graph function that searches 1e10-element tensors with an int64 output type; only
    # builds the concrete function (checks it traces without error and without running).

    @def_function.function
    def g():
      # [FUNCTION DOC]
      # Nested graph function: builds a 1e10-element random normal tensor and ones tensor and returns
      # their sorted search with int64 output type.
      x = random_ops.random_normal(shape=[int(1e10)])
      y = array_ops.ones(shape=[int(1e10)])
      return array_ops.searchsorted(x, y, out_type=dtypes.int64)

    _ = g.get_concrete_function()

  def testInt64UnspecifiedOutType(self):
    # [FUNCTION DOC]
    # Traces a graph function that searches 1e10-element tensors without specifying output type, to
    # check the library picks a usable type; only builds the concrete function.

    @def_function.function
    def g():
      # [FUNCTION DOC]
      # Nested graph function: builds 1e10-element random normal and ones tensors and returns their
      # sorted search with default output type.
      x = random_ops.random_normal(shape=[int(1e10)])
      y = array_ops.ones(shape=[int(1e10)])
      return array_ops.searchsorted(x, y)

    _ = g.get_concrete_function()

  def testInvalidValuesLowerBound(self):
    # [FUNCTION DOC]
    # Passes a 3 by 3 sequence, a rank-1 values tensor and wrong arguments (dtype and boolean) to the
    # generated lower bound op and asserts some exception is raised.
    arg_0_tensor = random_ops.random_uniform([3, 3], dtype=dtypes.float32)
    arg_0 = array_ops.identity(arg_0_tensor)
    arg_1_tensor = random_ops.random_uniform([3], dtype=dtypes.float32)
    arg_1 = array_ops.identity(arg_1_tensor)
    arg_2 = dtypes.int32
    arg_3 = False
    with self.assertRaises(Exception):
      gen_array_ops.lower_bound(arg_0, arg_1, arg_2, arg_3,)


class BatchGatherNdTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for batched gather-nd and gather-nd with batch dimensions, checking shapes,
  # equivalence to plain gather-nd and to map-fn, tensor batch dims, invalid batch dims and
  # unknown dimensions.

  def testShapesMatch(self):
    # [FUNCTION DOC]
    # Over a list of (params shape, indices shape, batch dims) combinations (sub-test each) runs
    # batch gather-nd on constant arrays and asserts the output shape equals the indices shape without
    # its last dimension plus the remaining row dimensions of params.
    """Tests for various different shape combinations."""
    shapes = []
    # params_shape, indices_shape, batch_dims
    shapes.append(((2, 2, 2), (2, 1), 1),)
    shapes.append(((2, 2, 2), (2, 2), 1),)
    shapes.append(((2, 2, 2), (2, 3), 0),)
    shapes.append(((2, 2, 2), (3,), 0),)
    shapes.append(((2, 2, 2), (1,), 0),)
    shapes.append(((2, 2, 3, 2), (2, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 3, 1), 1),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 2),)

    for params_shape, indices_shape, batch_dims in shapes:
      with self.subTest(
          params_shape=params_shape,
          indices_shape=indices_shape,
          batch_dims=batch_dims):
        params = constant_op.constant(1.0, shape=(params_shape))
        indices = constant_op.constant(
            1, shape=(indices_shape), dtype=dtypes.int32)
        out = array_ops.batch_gather_nd(
            params=params, indices=indices, batch_dims=batch_dims)
        ndims_params = len(params_shape) - batch_dims
        ndims_rows = ndims_params - indices_shape[-1]
        expected_out_shape = indices_shape[:-1]
        if ndims_rows > 0:
          expected_out_shape += params_shape[-ndims_rows:]
        self.assertSequenceEqual(out.shape, expected_out_shape)

  def testReducesToGatherNDWhenBatchDimIsZero(self):
    # [FUNCTION DOC]
    # For nine index shapes (sub-tests) asserts batch gather-nd with zero batch dims equals the
    # generated gather-nd on a random 7 by 8 by 9 tensor.
    """Confirms setting batch_dims to zero reduces to tf.gather_nd."""
    params = constant_op.constant(np.random.uniform(0.0, 1.0, size=(7, 8, 9)))
    indices_shapes = []
    indices_shapes.append((1,))
    indices_shapes.append((3, 1))
    indices_shapes.append((3, 3, 1))
    indices_shapes.append((2,))
    indices_shapes.append((3, 2))
    indices_shapes.append((3, 3, 2))
    indices_shapes.append((3,))
    indices_shapes.append((3, 3))
    indices_shapes.append((3, 3, 3))

    for indices_shape in indices_shapes:
      with self.subTest(indices_shape=indices_shape):
        indices = np.random.randint(0, 7, size=indices_shape)
        gather_nd_result = gen_array_ops.gather_nd(params, indices)
        batch_gather_nd_result = array_ops.batch_gather_nd(
            params=params, indices=indices, batch_dims=0)
        self.assertAllEqual(gather_nd_result, batch_gather_nd_result)

  def testSameResultAsMapFn(self):
    # [FUNCTION DOC]
    # For many shape combinations compares batch gather-nd with a map-fn applying gather-nd per batch
    # element (flattening extra batch dimensions and restoring the shape), asserting equality.
    """Compares results with gather_nd called on every element with map_fn."""
    shapes = []
    # params_shape, indices_shape, batch_dims
    shapes.append(((2, 2, 2), (2, 1), 1),)
    shapes.append(((2, 2, 2), (2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 3, 1), 1),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 2),)

    for params_shape, indices_shape, batch_dims in shapes:
      with self.subTest(
          params_shape=params_shape,
          indices_shape=indices_shape,
          batch_dims=batch_dims):
        params = constant_op.constant(
            np.random.uniform(0.0, 1.0, size=(params_shape)))
        indices = np.random.randint(0, 2, size=indices_shape)
        batch_gather_nd_result = array_ops.batch_gather_nd(
            params=params, indices=indices, batch_dims=batch_dims)

        if batch_dims > 1:
          params = array_ops.reshape(
              params, shape=[-1] + list(params_shape[batch_dims:]))
          indices = array_ops.reshape(
              indices, shape=[-1] + list(indices_shape[batch_dims:]))

        map_fn_gather_nd_result = map_fn.map_fn(
            fn=self._map_fn_body, elems=(params, indices), dtype=dtypes.float64)

        if batch_dims > 1:
          out_shape = map_fn_gather_nd_result.shape.as_list()
          out_shape = list(params_shape[:batch_dims]) + out_shape[1:]
          map_fn_gather_nd_result = array_ops.reshape(
              map_fn_gather_nd_result, shape=out_shape)

        self.assertAllEqual(map_fn_gather_nd_result, batch_gather_nd_result)

  def _map_fn_body(self, elems):
    # [FUNCTION DOC]
    # HELPER: map function used by testSameResultAsMapFn; takes a pair of tensors (params slice,
    # indices slice) and returns the gather-nd result of them.
    return gen_array_ops.gather_nd(elems[0], elems[1])

  def testBatchDimsAsTensor(self):
    # [FUNCTION DOC]
    # For three batch dim values (sub-tests) checks gather-nd with batch dims given as Python integer
    # equals gather-nd with batch dims given as a one-element tensor.
    """Tests Tensor batch_dims as input works as intended."""
    shapes = []
    # params_shape, indices_shape, batch_dims
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 0),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 1),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 2),)

    for params_shape, indices_shape, batch_dims in shapes:
      with self.subTest(
          params_shape=params_shape,
          indices_shape=indices_shape,
          batch_dims=batch_dims):
        params = constant_op.constant(
            np.random.uniform(0.0, 1.0, size=(params_shape)))
        indices = np.random.randint(0, 2, size=indices_shape)
        batch_gather_nd_result = array_ops.gather_nd(
            params=params, indices=indices, batch_dims=batch_dims)
        batch_dims_tensor = constant_op.constant([batch_dims])
        batch_gather_nd_tensor_batch_dims_result = array_ops.gather_nd(
            params=params, indices=indices, batch_dims=batch_dims_tensor)

        self.assertAllEqual(batch_gather_nd_tensor_batch_dims_result,
                            batch_gather_nd_result)

  def testInvalidBatchDimsRaisesException(self):
    # [FUNCTION DOC]
    # Asserts batch gather-nd raises a type error for a two-element tensor batch dims, and value errors
    # for batch dims -1 and 4.
    """Tests whether invalid batch_dims raise expected exceptions."""
    params = constant_op.constant(
        np.random.uniform(0.0, 1.0, size=(3, 2, 2, 3, 4)))
    indices = np.random.randint(0, 2, size=(3, 2, 3))

    with self.assertRaises(TypeError):
      array_ops.batch_gather_nd(
          params=params,
          indices=indices,
          batch_dims=constant_op.constant((0, 1)))

    with self.assertRaises(ValueError):
      array_ops.batch_gather_nd(params=params, indices=indices, batch_dims=-1)

    with self.assertRaises(ValueError):
      array_ops.batch_gather_nd(params=params, indices=indices, batch_dims=4)

  def testNoneBatchDimensions(self):
    # [FUNCTION DOC]
    # For shape combinations makes graph functions whose batch dimensions of params and indices are
    # unknown, runs them on ones and asserts the output shape equals the expected shape. Redefines the
    # traced function per loop iteration capturing the loop variable.
    """Tests gather_nd works with None dimensions."""
    shapes = []
    # params_shape, indices_shape, batch_dims
    shapes.append(((2, 2, 2), (2, 1), 1),)
    shapes.append(((2, 2, 2), (2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1), 1),)
    shapes.append(((2, 2, 3, 2), (2, 1, 3), 1),)
    shapes.append(((2, 2, 3, 2), (2, 2, 2), 1),)
    shapes.append(((2, 2, 3, 2), (2, 3, 1), 1),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 1, 3), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 2, 2), 2),)
    shapes.append(((3, 2, 2, 3, 4), (3, 2, 3, 1), 2),)

    for params_shape, indices_shape, batch_dims in shapes:
      params_ph_shape = list(params_shape)
      indices_ph_shape = list(indices_shape)
      for i in range(batch_dims):
        params_ph_shape[i] = None
        indices_ph_shape[i] = None

      @def_function.function
      def func(params, indices):
        # [FUNCTION DOC]
        # Nested graph function (params, indices): returns batch gather-nd with the batch dims of the
        # current loop iteration captured from the enclosing scope.
        return array_ops.batch_gather_nd(
            params=params, indices=indices, batch_dims=batch_dims)  # pylint: disable=cell-var-from-loop

      f = func.get_concrete_function(
          tensor_lib.TensorSpec(params_ph_shape, dtypes.float32),
          tensor_lib.TensorSpec(indices_ph_shape, dtypes.int32))

      params_val = np.ones(dtype=np.float32, shape=params_shape)
      indices_val = np.ones(dtype=np.int32, shape=indices_shape)
      res = f(params_val, indices_val)
      row_ndims = len(params_shape) - batch_dims - indices_shape[-1]
      expected_out_shape = indices_shape[:-1]
      if row_ndims > 0:
        expected_out_shape += params_shape[-row_ndims:]

      self.assertSequenceEqual(res.shape, expected_out_shape)


@test_util.run_all_in_graph_and_eager_modes
class RepeatTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the repeat op, parameterized with several array and repeat combinations. Both-modes
  # decorator.

  @parameterized.parameters(
      (3, 4, None),
      ([[1, 2], [3, 4]], 2, None),
      ([[1, 2], [3, 4]], [1, 2], 0),
      ([[1, 2], [3, 4]], [1, 2], 1),
      ([[1, 2], [3, 4]], 3, 1),
      ([[1, 2], [3, 4]], [1, 2, 3, 4], None),
      (np.ones([0, 4]), 0, 1),
      (np.ones([1, 2]), [2], None),
  )
  @test_util.with_forward_compatibility_horizons(None, [2052, 2, 7])
  def testRepeat(self, array, repeats, axis):
    # [FUNCTION DOC]
    # Parameterized with (array, repeats, axis). Compares repeat computed eagerly and through a graph
    # function with int32 input signatures to numpy repeat. Runs under several forward compatibility
    # horizons.
    array = np.array(array)

    @def_function.function(
        input_signature=[tensor_lib.TensorSpec(None, dtypes.int32)] * 2)
    def repeat_fn(array, repeats):
      # [FUNCTION DOC]
      # Nested graph function (array, repeats; both int32 tensors of unknown shape): returns the repeat of
      # the array along the captured axis.
      return array_ops.repeat(array, repeats, axis)

    v_tf = array_ops.repeat(constant_op.constant(array), repeats, axis)
    v_tf_fn = repeat_fn(
        constant_op.constant(array, dtype=dtypes.int32), repeats)
    v_np = np.repeat(array, repeats, axis)
    self.assertAllEqual(v_tf, v_np)
    self.assertAllEqual(v_tf_fn, v_np)


class RepeatBenchmark(test_lib.Benchmark):
  # [CLASS DOC]
  # Benchmark class for repeat with different shapes, repeat counts and axes.
  """Benchmark the repeat implementation."""

  def run_and_time(self, op, iters=100, warmup_iters=10):
    # [FUNCTION DOC]
    # HELPER: timing helper. Inputs: op, iters (default 100) and warmup_iters (default 10).
    # Initializes variables, warms up, times iterations and reports a benchmark with the average wall
    # time. Returns nothing.
    self.evaluate(variables.global_variables_initializer())
    for _ in range(warmup_iters):
      _ = self.evaluate(op)
    t0 = time.time()
    for _ in range(iters):
      self.evaluate(op)
    t1 = time.time()
    self.report_benchmark(iters=iters, wall_time=(t1 - t0) / float(iters))

  def make_variable(self, shape, dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: creates a float variable (dtype optional, default float32) of the given shape filled with
    # a linear sequence from 1 up to the element count; returns the variable.
    items = 1
    for dim in shape:
      items *= dim
    var = variables.Variable(
        array_ops.reshape(math_ops.linspace(1., float(items), items), shape),
        dtype=dtype)
    return var

  def run_benchmark(self, shape, max_repeats, axis=None):
    # [FUNCTION DOC]
    # HELPER: shared driver for the repeat benchmarks. Inputs: shape, max_repeats (exclusive upper
    # bound for random repeat counts) and axis (optional; none flattens). In a new session creates the
    # variable and random int64 repeat counts of axis length, builds the repeat op, reduces to a scalar
    # element to reduce copy overhead, and times it.
    with session.Session():
      var = self.make_variable(shape)
      if axis is None:
        axis_size = 1
        for dim in shape:
          axis_size *= dim
      else:
        axis_size = shape[axis]
      repeats = constant_op.constant(
          np.random.randint(max_repeats, size=[axis_size]), dtype=dtypes.int64)
      repeat_op = array_ops.repeat(var, repeats, axis=axis)
      # Return a scalar to reduce the device-to-host memcopy overhead.
      repeat_op = repeat_op[(0,) * len(shape)]
      self.run_and_time(repeat_op)

  def benchmark_repeat_few_1d(self):
    # [FUNCTION DOC]
    # Benchmarks repeat on a 1M-element vector with up to 8 repeats along axis 0.
    self.run_benchmark(shape=[1024 * 1024], max_repeats=8, axis=0)

  def benchmark_repeat_many_1d(self):
    # [FUNCTION DOC]
    # Benchmarks repeat on an 8K-element vector with up to 1024 repeats along axis 0.
    self.run_benchmark(shape=[8 * 1024], max_repeats=1024, axis=0)

  def benchmark_repeat_few_2d_axis0(self):
    # [FUNCTION DOC]
    # Benchmarks repeat on an 8 by 128K matrix with up to 8 repeats along axis 0.
    self.run_benchmark(shape=[8, 128 * 1024], max_repeats=8, axis=0)

  def benchmark_repeat_many_2d_axis0(self):
    # [FUNCTION DOC]
    # Benchmarks repeat on an 8 by 1024 matrix with up to 1024 repeats along axis 0.
    self.run_benchmark(shape=[8, 1024], max_repeats=1024, axis=0)

  def benchmark_repeat_many_2d_axis0_big(self):
    # [FUNCTION DOC]
    # Benchmarks repeat on a 1024 by 32 matrix with up to 1024 repeats along axis 0.
    self.run_benchmark(shape=[1024, 32], max_repeats=1024, axis=0)

  def benchmark_repeat_few_2d_axis1(self):
    # [FUNCTION DOC]
    # Benchmarks repeat on an 8 by 128K matrix with up to 8 repeats along axis 1.
    self.run_benchmark(shape=[8, 128 * 1024], max_repeats=8, axis=1)

  def benchmark_repeat_many_2d_axis1(self):
    # [FUNCTION DOC]
    # Benchmarks repeat on an 8 by 1024 matrix with up to 1024 repeats along axis 1.
    self.run_benchmark(shape=[8, 1024], max_repeats=1024, axis=1)


@test_util.run_all_in_graph_and_eager_modes
class TileVariantTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for tiling variant (tensor list) handles. Both-modes decorator.

  def test_tile_tensor_list(self):
    # [FUNCTION DOC]
    # Makes a tensor list from a random 2 by 3 by 4 tensor, tiles its handle twice on CPU and asserts
    # both stacked copies equal the original. Then mutates the original and one tiled list under
    # control dependencies and asserts the tiled lists are still equal to the original (copy-on-write).
    t = constant_op.constant(np.random.uniform(size=[2, 3, 4]))
    handle = list_ops.tensor_list_from_tensor(t, element_shape=None)
    with ops.device("CPU:0"):
      tiled_handles = array_ops.tile(array_ops.reshape(handle, [1]), [2])
    tiled_tensor_0 = list_ops.tensor_list_stack(tiled_handles[0], t.dtype, 2,
                                                [3, 4])
    tiled_tensor_1 = list_ops.tensor_list_stack(tiled_handles[1], t.dtype, 2,
                                                [3, 4])
    self.assertAllEqual(t, tiled_tensor_0)
    self.assertAllEqual(t, tiled_tensor_1)
    # Now mutate some of the lists and make sure the changes are not reflected
    # in the tiled handles.
    with ops.control_dependencies([
        list_ops.tensor_list_scatter([t[0] + 1], [0], input_handle=handle),
        list_ops.tensor_list_set_item(tiled_handles[0], 0, t[0] + 2)]):
      tiled_tensor_0 = list_ops.tensor_list_stack(tiled_handles[0], t.dtype, 2,
                                                  [3, 4])
      tiled_tensor_1 = list_ops.tensor_list_stack(tiled_handles[1], t.dtype, 2,
                                                  [3, 4])
    self.assertAllEqual(t, tiled_tensor_0)
    self.assertAllEqual(t, tiled_tensor_1)


class StopGradientTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for stop-gradient on tensors, ragged tensors and variables, in eager tape mode and
  # inside functions. Parameterized-test capable.

  def testStopGradient(self):
    # [FUNCTION DOC]
    # Applies stop gradient to a zeros vector and asserts the values are unchanged.
    x = array_ops.zeros(3)
    y = array_ops.stop_gradient(x)
    self.assertAllEqual(x, y)

  def testStopGradientRaggedTensor(self):
    # [FUNCTION DOC]
    # Applies stop gradient to a ragged tensor and asserts the result equals the input.
    x = RaggedTensor.from_row_splits(values=[1, 2, 3], row_splits=[0, 1, 1, 3])
    y = array_ops.stop_gradient(x)
    self.assertAllEqual(x, y)

  def testStopGradientGradientTape(self):
    # [FUNCTION DOC]
    # Under a gradient tape applies stop gradient to a vector and asserts the gradient with respect
    # to the input is nothing.
    x = array_ops.zeros(3)
    with backprop.GradientTape() as tape:
      y = array_ops.stop_gradient(x)

    self.assertIsNone(tape.gradient(y, x))

  def testStopGradientGradientTapeRaggedTensor(self):
    # [FUNCTION DOC]
    # Under a gradient tape applies stop gradient to a ragged tensor and asserts the gradient with
    # respect to the input is nothing.
    x = RaggedTensor.from_row_splits(values=[1, 2, 3], row_splits=[0, 1, 1, 3])
    with backprop.GradientTape() as tape:
      y = array_ops.stop_gradient(x)

    self.assertIsNone(tape.gradient(y, x))

  @parameterized.named_parameters([
      ("TFFunction", def_function.function),
      ("PythonFunction", lambda f: f),
  ])
  def test_stop_gradient_resource_variable(self, decorator):
    # [FUNCTION DOC]
    # Parameterized by a decorator: a graph function wrapper or the identity wrapper. Wraps a function
    # applying stop gradient to a resource variable, calls it under a tape and asserts the gradient is
    # nothing, the output is a tensor and its value equals the variable.
    x = resource_variable_ops.ResourceVariable([1.0])
    self.evaluate(x.initializer)

    @decorator
    def stop_gradient_f(x):
      # [FUNCTION DOC]
      # Nested function (decorated with the parameterized wrapper); returns the stop gradient of its
      # argument.
      return array_ops.stop_gradient(x)

    with backprop.GradientTape() as tape:
      y = stop_gradient_f(x)
    self.assertIsNone(tape.gradient(y, x))
    # stop_gradient converts ResourceVariable to Tensor
    self.assertIsInstance(y, tensor_lib.Tensor)
    self.assertAllEqual(y, x)

if __name__ == "__main__":
  test_lib.main()
