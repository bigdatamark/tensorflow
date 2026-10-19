# [MODULE DOC]
# Test module for the gather operation (including axis and batch_dims support): scalar and
# vector indices, higher-rank params/indices, gradients in graph mode and with a gradient tape,
# strings and unsigned types, unknown-shape placeholders, invalid indices and axes, empty
# slices, batched gathers checked against numpy, and gathers from reference and resource
# variables. Defines module constants listing the dtypes and index dtypes exercised, a small
# string-conversion helper, and one parameterized test class. No import-time side effects
# beyond imports; run through the test runner entry point when executed as a script.
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
"""Tests for tensorflow.ops.tf.gather."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

_TEST_TYPES = (
    dtypes.int64,
    dtypes.bfloat16,
    dtypes.float32,
    dtypes.complex64,
    dtypes.complex128,
    dtypes.float8_e5m2,
    dtypes.float8_e4m3fn,
)
_INDEX_TYPES = (dtypes.int16, dtypes.int32, dtypes.int64)

# TODO(virimia): Add a benchmark for gather_v2, with batch_dims and axis set.


def _to_str_elements(values):
  # [FUNCTION DOC]
  # HELPER: converts numeric test data into byte-string data for the batched gather tests.
  # Recursively walks a possibly nested list and replaces every non-list leaf with its decimal
  # text form encoded as UTF-8 bytes; lists are rebuilt (input not mutated). Input: a value or a
  # nested list of values. Returns a nested list of the same structure with byte strings (or a
  # single byte string for a non-list input).
  """Converts the inner list elements to strings."""
  if isinstance(values, list):
    return [_to_str_elements(value) for value in values]
  else:
    return str(values).encode("utf-8")


class GatherTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the gather operation across dtypes, index types, axes and batch
  # dimensions, using numpy take (and a recursive numpy batch gather) as the reference. Provides
  # helpers for building typed test data. No shared setup.

  def _buildParams(self, data, dtype):
    # [FUNCTION DOC]
    # HELPER: builds typed input data for the gather tests.
    # Inputs: a numpy array and a TensorFlow dtype. Converts the array to the dtype's numpy type;
    # for complex dtypes adds ten times the value as the imaginary part so each element is
    # distinguishable. Returns the converted numpy array; no side effects.
    data = data.astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testScalar1D(self):
    # [FUNCTION DOC]
    # Test scenario: for every test dtype and index dtype, gather from a six-element vector with
    # a scalar index four and with an index list. Expects values equal to numpy indexing and the
    # static shape of the result to equal the numpy result shape. Returns nothing.
    with self.cached_session():
      data = np.array([0, 1, 2, 3, 7, 5])
      for dtype in _TEST_TYPES:
        for itype in _INDEX_TYPES:
          for indices in 4, [1, 2, 2, 4, 5]:
            with self.subTest(dtype=dtype, itype=itype, indices=indices):
              params_np = self._buildParams(data, dtype)
              params = constant_op.constant(params_np)
              indices_tf = constant_op.constant(indices, dtype=itype)
              gather_t = array_ops.gather(params, indices_tf)
              gather_val = self.evaluate(gather_t)
              np_val = params_np[indices]
              self.assertAllEqual(np_val, gather_val)
              self.assertEqual(np_val.shape, gather_t.get_shape())

  def testScalar2D(self):
    # [FUNCTION DOC]
    # Test scenario: for every test dtype, index dtype and each axis of a five-by-three matrix,
    # gather with scalar index two. Expects values equal to numpy take and a static shape with
    # the gathered axis removed. Returns nothing.
    with self.session():
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [9, 10, 11], [12, 13, 14]])
      for dtype in _TEST_TYPES:
        for itype in _INDEX_TYPES:
          for axis in range(data.ndim):
            with self.subTest(dtype=dtype, itype=itype, axis=axis):
              params_np = self._buildParams(data, dtype)
              params = constant_op.constant(params_np)
              indices = constant_op.constant(2, dtype=itype)
              gather_t = array_ops.gather(params, indices, axis=axis)
              gather_val = self.evaluate(gather_t)
              self.assertAllEqual(np.take(params_np, 2, axis=axis), gather_val)
              expected_shape = data.shape[:axis] + data.shape[axis + 1 :]
              self.assertEqual(expected_shape, gather_t.get_shape())

  def testSimpleTwoD32(self):
    # [FUNCTION DOC]
    # Test scenario: for every test dtype, index dtype and axis of a five-by-three matrix, gather
    # the index list zero, one, zero, two (in bounds for either axis). Expects values equal to
    # numpy take and the gathered axis replaced by size four in the static shape. Returns nothing.
    with self.session():
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [9, 10, 11], [12, 13, 14]])
      for dtype in _TEST_TYPES:
        for itype in _INDEX_TYPES:
          for axis in range(data.ndim):
            with self.subTest(dtype=dtype, itype=itype, axis=axis):
              params_np = self._buildParams(data, dtype)
              params = constant_op.constant(params_np)
              # The indices must be in bounds for any axis.
              indices = constant_op.constant([0, 1, 0, 2], dtype=itype)
              gather_t = array_ops.gather(params, indices, axis=axis)
              gather_val = self.evaluate(gather_t)
              self.assertAllEqual(
                  np.take(params_np, [0, 1, 0, 2], axis=axis), gather_val
              )
              expected_shape = data.shape[:axis] + (4,) + data.shape[axis + 1 :]
              self.assertEqual(expected_shape, gather_t.get_shape())

  def testHigherRank(self):
    # [FUNCTION DOC]
    # Test scenario (graph mode): params of shape two, one, three, two filled with random values;
    # random indices of several shapes (scalar, empty, two-dimensional); every test dtype and
    # every axis, given both as a non-negative and as the equivalent negative axis. Expects equal
    # values to numpy take and static shape params-before-axis + indices shape + params-after.
    # Gradients with respect to params, indices and axis are then computed with an upstream
    # gradient of random values: indices and axis gradients must be nothing; params gradient must
    # be nothing for integer and 8-bit float dtypes (the case is then skipped); otherwise for
    # axis zero it must be an indexed-slices object, and its dense form must match a numpy
    # scatter-add of the upstream gradient within 2e-6 tolerance. Returns nothing.
    with ops.Graph().as_default():
      # We check that scalar and empty indices shapes work as well
      shape = (2, 1, 3, 2)
      for indices_shape in (), (0,), (2, 0), (2, 3):
        for dtype in _TEST_TYPES:
          for axis in range(len(shape)):
            params = self._buildParams(np.random.randn(*shape), dtype)
            indices = np.random.randint(shape[axis], size=indices_shape)
            with self.subTest(
                indices_shape=indices_shape,
                dtype=dtype,
                axis=axis,
                indices=indices,
            ):
              tf_params = constant_op.constant(params)
              tf_indices = constant_op.constant(indices)
              # Check that both positive and negative indices for axis work.
              tf_axis = constant_op.constant(axis)
              tf_negative_axis = constant_op.constant(-len(shape) + axis)
              gather = array_ops.gather(tf_params, tf_indices, axis=tf_axis)
              gather_negative_axis = array_ops.gather(
                  tf_params, tf_indices, axis=tf_negative_axis
              )
              gather_value, gather_negative_axis_value = self.evaluate(
                  [gather, gather_negative_axis]
              )
              gather_np = np.take(params, indices, axis)
              self.assertAllEqual(gather_np, gather_value)
              self.assertAllEqual(gather_np, gather_negative_axis_value)
              expected_shape = (
                  params.shape[:axis] + indices.shape + params.shape[axis + 1 :]
              )
              self.assertEqual(expected_shape, gather.shape)
              self.assertEqual(expected_shape, gather_negative_axis.shape)

              # Test gradients
              gather_grad = np.random.randn(
                  *gather.get_shape().as_list()
              ).astype(dtype.as_numpy_dtype)
              if dtype.is_complex:
                gather_grad -= 1j * gather_grad
              params_grad, indices_grad, axis_grad = gradients_impl.gradients(
                  gather, [tf_params, tf_indices, tf_axis], gather_grad
              )
              self.assertIsNone(indices_grad)
              self.assertIsNone(axis_grad)
              if dtype.is_integer or dtype in [
                  dtypes.float8_e5m2,
                  dtypes.float8_e4m3fn,
              ]:
                self.assertIsNone(params_grad)
                continue
              # For axis 0, we are able to create an efficient IndexedSlices
              # for the gradient.
              if axis == 0:
                self.assertEqual(
                    type(params_grad), indexed_slices.IndexedSlices
                )
                params_grad = ops.convert_to_tensor(params_grad)
              correct_params_grad = np.zeros(shape).astype(dtype.as_numpy_dtype)
              outer_dims = axis
              inner_dims = len(shape) - axis - 1
              gather_grad = gather_grad.reshape(
                  shape[:axis] + (indices.size,) + shape[axis + 1 :]
              )
              for source_index, dest_index in enumerate(indices.flat):
                dest_slice = (
                    (slice(None),) * outer_dims
                    + (dest_index,)
                    + (slice(None),) * inner_dims
                )
                source_slice = (
                    (slice(None),) * outer_dims
                    + (source_index,)
                    + (slice(None),) * inner_dims
                )
                correct_params_grad[dest_slice] += gather_grad[source_slice]
              self.assertAllCloseAccordingToType(
                  correct_params_grad,
                  self.evaluate(params_grad),
                  atol=2e-6,
                  rtol=2e-6,
              )

  def testHigherRankGradientTape(self):
    # [FUNCTION DOC]
    # Test scenario: same coverage as the graph-mode higher-rank test (random params, indices of
    # several shapes, all dtypes, all axes in positive and negative form, value/shape checks and
    # gradient checks) but gradients come from a gradient tape that watches params, indices and
    # axis, with the upstream gradient supplied explicitly. Expects no gradient for indices and
    # axis, no params gradient for integer/8-bit float dtypes, an indexed-slices params gradient
    # for axis zero, and dense values matching a numpy scatter-add within 2e-6. Returns nothing.
    # We check that scalar and empty indices shapes work as well
    shape = (2, 1, 3, 2)
    for indices_shape in (), (0,), (2, 0), (2, 3):
      for dtype in _TEST_TYPES:
        for axis in range(len(shape)):
          params = self._buildParams(np.random.randn(*shape), dtype)
          indices = np.random.randint(shape[axis], size=indices_shape)
          with self.subTest(
              indices_shape=indices_shape,
              dtype=dtype,
              axis=axis,
              indices=indices,
          ):
            with backprop.GradientTape() as tape:
              tf_params = constant_op.constant(params)
              tf_indices = constant_op.constant(indices)
              # Check that both positive and negative indices for axis work.
              tf_axis = constant_op.constant(axis)
              tape.watch(tf_params)
              tape.watch(tf_indices)
              tape.watch(tf_axis)
              tf_negative_axis = constant_op.constant(-len(shape) + axis)
              gather = array_ops.gather(tf_params, tf_indices, axis=tf_axis)
              gather_negative_axis = array_ops.gather(
                  tf_params, tf_indices, axis=tf_negative_axis
              )
              gather_value, gather_negative_axis_value = self.evaluate(
                  [gather, gather_negative_axis]
              )
              gather_np = np.take(params, indices, axis)
              self.assertAllEqual(gather_np, gather_value)
              self.assertAllEqual(gather_np, gather_negative_axis_value)
              expected_shape = (
                  params.shape[:axis] + indices.shape + params.shape[axis + 1 :]
              )
              self.assertEqual(expected_shape, gather.shape)
              self.assertEqual(expected_shape, gather_negative_axis.shape)

              # Test gradients
              gather_grad = np.random.randn(
                  *gather.get_shape().as_list()
              ).astype(dtype.as_numpy_dtype)
              if dtype.is_complex:
                gather_grad -= 1j * gather_grad
            params_grad, indices_grad, axis_grad = tape.gradient(
                gather, [tf_params, tf_indices, tf_axis], gather_grad
            )
            self.assertIsNone(indices_grad)
            self.assertIsNone(axis_grad)
            if dtype.is_integer or dtype in [
                dtypes.float8_e5m2,
                dtypes.float8_e4m3fn,
            ]:
              self.assertIsNone(params_grad)
              continue
            # For axis 0, we are able to create an efficient IndexedSlices for
            # the gradient.
            if axis == 0:
              self.assertEqual(type(params_grad), indexed_slices.IndexedSlices)
              params_grad = ops.convert_to_tensor(params_grad)
            correct_params_grad = np.zeros(shape).astype(dtype.as_numpy_dtype)
            outer_dims = axis
            inner_dims = len(shape) - axis - 1
            gather_grad = gather_grad.reshape(
                shape[:axis] + (indices.size,) + shape[axis + 1 :]
            )
            for source_index, dest_index in enumerate(indices.flat):
              dest_slice = ((slice(None),) * outer_dims + (dest_index,) +
                            (slice(None),) * inner_dims)
              source_slice = ((slice(None),) * outer_dims + (source_index,) +
                              (slice(None),) * inner_dims)
              correct_params_grad[dest_slice] += gather_grad[source_slice]
            self.assertAllCloseAccordingToType(
                correct_params_grad,
                self.evaluate(params_grad),
                atol=2e-6,
                rtol=2e-6,
            )

  def testString(self):
    # [FUNCTION DOC]
    # Test scenario: gather from a two-by-two array of byte strings with scalar index one on axis
    # zero (expects the second row) and scalar index zero on axis one (expects the first column).
    # Returns nothing.
    params = np.array([[b"asdf", b"zxcv"], [b"qwer", b"uiop"]])
    self.assertAllEqual([b"qwer", b"uiop"], array_ops.gather(params, 1, axis=0))
    self.assertAllEqual([b"asdf", b"qwer"], array_ops.gather(params, 0, axis=1))

  def testUInt32AndUInt64(self):
    # [FUNCTION DOC]
    # Test scenario: for 32-bit and 64-bit unsigned integers, gather from a two-by-three matrix
    # with scalar index one on axis zero (expects seven, eight, nine) and index zero on axis one
    # (expects one, seven). Returns nothing.
    for unsigned_type in (dtypes.uint32, dtypes.uint64):
      with self.subTest(unsigned_type=unsigned_type):
        params = self._buildParams(
            np.array([[1, 2, 3], [7, 8, 9]]), unsigned_type)
        with self.cached_session():
          self.assertAllEqual([7, 8, 9], array_ops.gather(params, 1, axis=0))
          self.assertAllEqual([1, 7], array_ops.gather(params, 0, axis=1))

  def testUnknownIndices(self):
    # [FUNCTION DOC]
    # Test scenario (graph mode only, uses a placeholder): gather from a constant with an indices
    # placeholder of unknown shape. Expects the static result shape to be completely unknown.
    # Returns nothing.
    # This test is purely a test for placeholder inputs which is only applicable
    # in graph mode.
    with ops.Graph().as_default():
      params = constant_op.constant([[0, 1, 2]])
      indices = array_ops.placeholder(dtypes.int32)
      gather_t = array_ops.gather(params, indices)
      self.assertEqual(None, gather_t.get_shape())

  def testUnknownAxis(self):
    # [FUNCTION DOC]
    # Test scenario (graph mode only): the axis is a placeholder of unknown value. With rank-two
    # params and rank-two indices, expects a result of rank three with all dimensions unknown.
    # When indices is also an unknown-shape placeholder, expects the result shape to be fully
    # unknown. Returns nothing.
    # This test is purely a test for placeholder inputs which is only applicable
    # in graph mode.
    with ops.Graph().as_default():
      params = constant_op.constant([[0, 1, 2]])
      indices = constant_op.constant([[0, 0], [0, 0]])
      axis = array_ops.placeholder(dtypes.int32)
      gather_t = array_ops.gather(params, indices, axis=axis)
      # Rank 2 params with rank 2 indices results in a rank 3 shape.
      self.assertEqual([None, None, None], gather_t.shape.as_list())

      # If indices is also unknown the result rank is unknown.
      indices = array_ops.placeholder(dtypes.int32)
      gather_t = array_ops.gather(params, indices, axis=axis)
      self.assertEqual(None, gather_t.shape)

  def testBadIndicesType(self):
    # [FUNCTION DOC]
    # Test scenario: gathering with a floating-point scalar index must raise a type error or an
    # invalid-argument error whose message lists the allowed index types (16-, 32-, 64-bit
    # integers). Returns nothing.
    with self.assertRaisesRegex(
        (TypeError, errors.InvalidArgumentError),
        "float.* not in.* list of allowed values: int16, int32, int64"):
      self.evaluate(array_ops.gather([0], 0.))

  @test_util.disable_xla(
      "Assertion inside an op is not supported in XLA. Instead XLA clamps the "
      "index to be in bounds and returns the indexed value there (Don't rely "
      "on this behavior).")
  def testBadIndicesCPU(self):
    # [FUNCTION DOC]
    # Test scenario (forced onto the CPU, disabled under XLA because XLA clamps indices instead of
    # failing): gather with out-of-range index seven from a two-by-three matrix on axis zero and
    # on axis one. Expects an op error stating the index is not in the half-open range of the
    # axis size (two, then three). Returns nothing.
    with test_util.force_cpu():
      params = [[0, 1, 2], [3, 4, 5]]
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 2\)"):
        self.evaluate(array_ops.gather(params, [[7]], axis=0))
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 3\)"):
        self.evaluate(array_ops.gather(params, [[7]], axis=1))

  def _disabledTestBadIndicesGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead code. The leading underscore keeps the test runner from collecting it; it
    # is disabled because GPU gather returns zeros for bad indices instead of raising an error.
    # Intended scenario: when a GPU is available, gather index seven from a two-by-three matrix
    # on axes zero and one and expect an out-of-range op error. Does nothing without a GPU.
    # TODO disabled due to different behavior on GPU and CPU
    # On GPU the bad indices do not raise error but fetch 0 values
    if not test.is_gpu_available():
      return
    with self.session():
      params = [[0, 1, 2], [3, 4, 5]]
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 2\)"):
        array_ops.gather(params, [[7]], axis=0).eval()
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 3\)"):
        array_ops.gather(params, [[7]], axis=1).eval()

  def testBadAxis(self):
    # [FUNCTION DOC]
    # Test scenario: builds four callables doing the same gather (the plain op, a compiled
    # function, a compiled function with unknown-shape input signature, and the same with XLA
    # compilation) and calls each with a rank-one params vector, scalar index, and invalid axes
    # one, two and minus two. Expects each to raise a value error or invalid-argument error
    # saying the shape must be at least a larger rank but is rank one. Returns nothing.

    @def_function.function(autograph=False, jit_compile=False)
    def gather(x, indices, axis):
      # [FUNCTION DOC]
      # Nested helper: compiled function (no autograph, no XLA) that gathers from x using
      # indices along axis and returns the result.
      return array_ops.gather(x, indices, axis=axis)

    @def_function.function(
        autograph=False,
        jit_compile=False,
        input_signature=[
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
        ] * 3)
    def gather_shape_inf_disabled(x, indices, axis):
      # [FUNCTION DOC]
      # Nested helper: compiled function with unknown-shape 32-bit integer inputs so static shape
      # inference cannot catch a bad axis; gathers x by indices along axis and returns the result.
      return array_ops.gather(x, indices, axis=axis)

    @def_function.function(
        autograph=False,
        jit_compile=True,
        input_signature=[
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
        ] * 3)
    def xla_gather(x, indices, axis):
      # [FUNCTION DOC]
      # Nested helper: like the unknown-shape variant but compiled with XLA; gathers x by indices
      # along axis and returns the result.
      return array_ops.gather(x, indices, axis=axis)

    params = [0, 1, 2]
    indices = 0
    functions = [("array_ops.gather", array_ops.gather), ("gather", gather),
                 ("gather_shape_inf_disabled", gather_shape_inf_disabled),
                 ("xla_gather", xla_gather)]
    for bad_axis in (1, 2, -2):
      for fn_name, fn in functions:
        # Shape inference can validate axis for known params rank.
        with self.subTest(bad_axis=bad_axis, msg=fn_name, fn=fn):
          with self.assertRaisesRegex(
              (ValueError, errors.InvalidArgumentError),
              "Shape must be at least rank .* but is rank 1"):
            fn(params, indices, axis=bad_axis)

  def testEmptySlices(self):
    # [FUNCTION DOC]
    # Test scenario: for every test dtype and index dtype, gather indices three and four from
    # zero-filled arrays with shapes seven-zero-zero (axis zero), zero-seven-zero (axis one) and
    # zero-zero-seven (axis two). Expects zero arrays whose gathered axis has size two.
    # Returns nothing.
    for dtype in _TEST_TYPES:
      for itype in _INDEX_TYPES:
        # Leading axis gather.
        with self.subTest(dtype=dtype, itype=itype):
          params = np.zeros((7, 0, 0), dtype=dtype.as_numpy_dtype)
          indices = np.array([3, 4], dtype=itype.as_numpy_dtype)
          gather = array_ops.gather(params, indices, axis=0)
          self.assertAllEqual(gather, np.zeros((2, 0, 0)))

          # Middle axis gather.
          params = np.zeros((0, 7, 0), dtype=dtype.as_numpy_dtype)
          gather = array_ops.gather(params, indices, axis=1)
          self.assertAllEqual(gather, np.zeros((0, 2, 0)))

          # Trailing axis gather.
          params = np.zeros((0, 0, 7), dtype=dtype.as_numpy_dtype)
          gather = array_ops.gather(params, indices, axis=2)
          self.assertAllEqual(gather, np.zeros((0, 0, 2)))

  @parameterized.parameters([
      # batch_dims=0 (equivalent to tf.gather)
      dict(  # 2D indices
          batch_dims=0,
          params=[6, 7, 8, 9],
          indices=[[2, 1], [0, 3]],
          expected=[[8, 7], [6, 9]]),
      dict(  # 3D indices
          batch_dims=0,
          params=[6, 7, 8, 9],
          indices=[[[3, 1], [2, 0]], [[0, 3], [2, 2]]],
          expected=[[[9, 7], [8, 6]], [[6, 9], [8, 8]]]),
      dict(  # 4D indices
          batch_dims=0,
          params=[8, 9],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[8, 9], [9, 8]], [[8, 8], [9, 9]]],
                    [[[9, 9], [8, 8]], [[8, 9], [9, 8]]]]),

      # batch_dims=indices.shape.ndims - 1
      # (equivalent to tf.compat.v1.batch_gather)
      dict(  # 2D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[2, 1], [0, 3]],
          expected=[[12, 11], [20, 23]]),
      dict(  # 3D indices (2 batch dims)
          batch_dims=2,
          params=[[[100, 101], [110, 111]], [[200, 201], [210, 211]]],
          indices=[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
          expected=[[[100, 101], [111, 110]], [[200, 200], [211, 211]]]),
      dict(  # 2D indices (1 batch dim)
          batch_dims=-1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[2, 1], [0, 3]],
          expected=[[12, 11], [20, 23]]),
      dict(  # 3D indices (2 batch dims)
          batch_dims=-1,
          params=[[[100, 101], [110, 111]], [[200, 201], [210, 211]]],
          indices=[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
          expected=[[[100, 101], [111, 110]], [[200, 200], [211, 211]]]),

      # batch_dims=indices.shape.ndims
      dict(  # 1D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[2, 1],
          expected=[12, 21]),
      dict(  # 2D indices (2 batch dim)
          batch_dims=2,
          params=[[[100, 101, 102, 103], [110, 111, 112, 113]],
                  [[200, 201, 202, 203], [210, 211, 212, 213]]],
          indices=[[2, 1], [0, 3]],
          expected=[[102, 111], [200, 213]]),

      # 0 < batch_dims < indices.shape.ndims - 1
      dict(  # 3D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[[3, 1], [2, 0]], [[0, 3], [2, 2]]],
          expected=[[[13, 11], [12, 10]], [[20, 23], [22, 22]]]),
      dict(  # 4D indices (1 batch dim)
          batch_dims=1,
          params=[[6, 7], [8, 9]],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[6, 7], [7, 6]], [[6, 6], [7, 7]]],
                    [[[9, 9], [8, 8]], [[8, 9], [9, 8]]]]),
      dict(  # 4D indices (2 batch dims)
          batch_dims=2,
          params=[[[2, 3], [4, 5]], [[6, 7], [8, 9]]],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[2, 3], [3, 2]], [[4, 4], [5, 5]]],
                    [[[7, 7], [6, 6]], [[8, 9], [9, 8]]]]),

      # axis > 0
      dict(  # 3D indices, batch_dims=1, axis=2
          # params.shape  = [I1, J1, J2] = [2, 2, 3]
          # indices.shape = [I1, K1, K2] = [2, 1, 5]
          # result.shape  = [I1, J1, K1, K2] = [2, 2, 1, 5]
          batch_dims=1,
          axis=2,
          params=[[[10, 11, 12], [13, 14, 15]], [[20, 21, 22], [23, 24, 25]]],
          indices=[[[0, 1, 2, 1, 0]], [[0, 1, 2, 1, 0]]],
          expected=[[[[10, 11, 12, 11, 10]], [[13, 14, 15, 14, 13]]],
                    [[[20, 21, 22, 21, 20]], [[23, 24, 25, 24, 23]]]]),
      dict(  # 3D indices, batch_dims=None, axis=1
          batch_dims=None,
          axis=1,
          params=[[10, 11, 12], [13, 14, 15]],
          indices=[1, 0],
          expected=[[11, 10], [14, 13]]),
      dict(  # 3D indices, batch_dims=-3, axis=1
          batch_dims=-3,
          axis=1,
          params=[[0, 1, 2], [3, 4, 5]],
          indices=[[[0, 1], [1, 0]]],
          expected=[[[[0, 1], [1, 0]]], [[[3, 4], [4, 3]]]]),
  ])
  @test_util.run_in_graph_and_eager_modes
  def testBatchDims(self, params, indices, batch_dims, expected=None,
                    axis=None):
    # [FUNCTION DOC]
    # Parameterized test (runs in graph and eager modes) over many cases. Inputs: params and
    # indices (nested lists), batch_dims (integer, negative, or nothing), expected result
    # (optional), and axis (optional). Checks: (1) gather with batch_dims gives the expected
    # values; (2) numeric and theoretical gradients agree for a 64-bit float copy of params;
    # (3) the same gradient check through a function with unknown-shape inputs, which for negative
    # batch_dims must instead raise a value error saying gradients of gather with such inputs
    # are unsupported; (4) the gradient of the result with respect to a tape-watched zero tensor
    # has the same shape as params; (5) the same gather on string versions of params returns
    # string versions of expected. Returns nothing.
    result = array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    self.assertAllEqual(expected, result)

    # Test gradients
    f64_params = math_ops.cast(params, dtypes.float64)
    def gather(params):
      # [FUNCTION DOC]
      # Nested helper: gathers from the given params using the enclosing test's indices, axis and
      # batch_dims; used as the function whose gradient is checked. Returns the gathered tensor.
      return array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        gather, [f64_params])
    self.assertAllClose(theoretical, numerical)

    # Test gradients when input shapes are unknown
    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.float64),
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
    ])
    def gather_unknown_shapes(params, indices):
      # [FUNCTION DOC]
      # Nested helper: compiled function with unknown-shape float64 params and 32-bit integer
      # indices that gathers using the enclosing test's axis and batch_dims. Returns the result.
      return array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    if batch_dims is None or batch_dims >= 0:
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda p: gather_unknown_shapes(p, indices), [f64_params])
      self.assertAllClose(theoretical, numerical)
    else:
      with self.assertRaisesRegex(
          ValueError,
          "Currently, it is unsupported to take the gradient of tf.gather"):
        gradient_checker_v2.compute_gradient(
            lambda p: gather_unknown_shapes(p, indices), [f64_params])

    # Test the gradients shape.
    with backprop.GradientTape() as tape:
      zeros = array_ops.zeros_like(params, dtype=dtypes.float32)
      tape.watch(zeros)
      values = zeros * 2 + zeros
      result = array_ops.gather(
          values, indices, axis=axis, batch_dims=batch_dims)
    gradients = tape.gradient(result, zeros)

    self.assertAllEqual(array_ops.shape(params), array_ops.shape(gradients))

    # Run the same test for strings.
    params = _to_str_elements(params)
    expected = _to_str_elements(expected)
    result = array_ops.gather(
        params, indices, axis=axis, batch_dims=batch_dims)

    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=2,
          output_shape=[2, 3, 8, 9, 10, 5, 6, 7]
          # = params.shape[:2] + indices.shape[2:] + params.shape[3:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=3,
          output_shape=[2, 3, 4, 8, 9, 10, 6, 7]
          # = params.shape[:3] + indices.shape[2:] + params.shape[4:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=4,
          output_shape=[2, 3, 4, 5, 8, 9, 10, 7]
          # = params.shape[:4] + indices.shape[2:] + params.shape[5:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=5,
          output_shape=[2, 3, 4, 5, 6, 8, 9, 10]
          # = params.shape[:5] + indices.shape[2:] + params.shape[6:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=-4,
          output_shape=[2, 3, 8, 9, 10, 5, 6, 7]
          # = params.shape[:2] + indices.shape[2:] + params.shape[3:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=-3,
          output_shape=[2, 3, 4, 8, 9, 10, 6, 7]
          # = params.shape[:3] + indices.shape[2:] + params.shape[4:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=-2,
          output_shape=[2, 3, 4, 5, 8, 9, 10, 7]
          # = params.shape[:4] + indices.shape[2:] + params.shape[5:]
          ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          axis=-1,
          output_shape=[2, 3, 4, 5, 6, 8, 9, 10]
          # = params.shape[:5] + indices.shape[2:] + params.shape[6:]
          ),
  ])
  @test_util.run_in_graph_and_eager_modes
  def testBatchDimsMatchesPythonBatching(self, params_shape, indices_shape,
                                         batch_dims, axis, output_shape):
    # [FUNCTION DOC]
    # Parameterized test (graph and eager modes). Inputs: params_shape, indices_shape,
    # batch_dims, axis (possibly negative) and the expected output shape. Builds params as a
    # reshaped range and indices as a reshaped range reduced modulo the axis size, computes the
    # expected result with the numpy batch-gather helper, converts both back to lists, runs gather
    # with batch_dims and axis, and asserts the output shape and values. Repeats with string data.
    # Returns nothing.
    """Checks that batch_dims matches multiple calls to tf.gather()."""
    # Generate a `params` tensor with the indicated shape.
    params_size = np.prod(params_shape)
    params = np.reshape(np.arange(params_size), params_shape)

    # Generate an `indices` tensor with the indicated shape, where each index
    # is within the appropriate range.
    indices_size = np.prod(indices_shape)
    indices = np.reshape(np.arange(indices_size), indices_shape)
    indices = indices % params_shape[axis]

    # Perform repeated (batched) gather operations with numpy, to find the
    # expected result.
    expected = self._batchNumpyGather(params, indices, axis, batch_dims)

    # On Windows, we get an exception if we pass in the transformed numpy
    # arrays ("Failed to convert numpy ndarray to a Tensor (Unsupported
    # feed type)."); so convert them back to lists before calling tf.gather.
    params = params.tolist()
    indices = indices.tolist()

    result = array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    self.assertAllEqual(output_shape, result.shape.as_list())
    self.assertAllEqual(expected, result)

    # Run the same test for strings.
    params = _to_str_elements(params)
    expected = _to_str_elements(expected.tolist())
    result = array_ops.gather(
        params, indices, axis=axis, batch_dims=batch_dims)

    self.assertAllEqual(output_shape, result.shape.as_list())
    self.assertAllEqual(expected, result)

  def _batchNumpyGather(self, params, indices, axis, batch_dims):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation of batched gather for the batch-dims tests.
    # KNOWN ISSUE: the docstring says it is used by the batch-dims test, but it is actually
    # called only by the test comparing batch_dims with Python batching.
    # Inputs: params and indices numpy arrays, axis (integer), batch_dims (integer, not
    # negative). With zero batch dims uses numpy take; otherwise asserts the leading dimensions of
    # params and indices match (failing the test if not), reduces the axis by one when positive,
    # recurses per leading element with one fewer batch dim and stacks the results.
    # Returns a numpy array.
    """Performs a batch gather by making recursive calls to np.take().

    This is used by testBatchDims() to construct the expected value.

    Args:
      params: A numpy array
      indices: A numpy array
      axis: An integer
      batch_dims: An integer
    Returns:
      A numpy array
    """
    if batch_dims == 0:
      return np.take(params, indices, axis=axis)
    self.assertEqual(params.shape[0], indices.shape[0])
    if axis > 0:
      axis -= 1
    return np.stack([
        self._batchNumpyGather(params[i], indices[i], axis, batch_dims - 1)
        for i in range(params.shape[0])
    ])

  @test_util.run_v1_only("RefVariable is not supported in v2")
  def testGatherRefVariable(self):
    # [FUNCTION DOC]
    # Test scenario (graph-mode v1 only): gather rows zero and two from a reference variable
    # holding a three-by-two integer matrix, after initializing variables. Expects rows one-two
    # and five-six. Returns nothing.
    with self.cached_session():
      v = ref_variable.RefVariable(
          constant_op.constant([[1, 2], [3, 4], [5, 6]]))
      self.evaluate(variables.global_variables_initializer())
      gather = array_ops.gather(v, [0, 2])
      self.assertAllEqual([[1, 2], [5, 6]], gather)

  @test_util.run_in_graph_and_eager_modes
  def testGatherResourceVariable(self):
    # [FUNCTION DOC]
    # Test scenario (graph and eager modes): same as the reference-variable test but with a
    # resource variable; gathers rows zero and two and expects rows one-two and five-six.
    # Returns nothing.
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(
          constant_op.constant([[1, 2], [3, 4], [5, 6]]))
      self.evaluate(variables.global_variables_initializer())
      gather = array_ops.gather(v, [0, 2])
      self.assertAllEqual([[1, 2], [5, 6]], gather)

if __name__ == "__main__":
  test.main()
