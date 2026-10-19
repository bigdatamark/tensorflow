# [MODULE DOC]
# Tests for the gather_nd operation, which collects slices of a tensor addressed by index tuples.
# Covers many element types and index types, empty inputs, higher-rank inputs, bad-index policies,
# gradients (dense and indexed-slices), variables, and includes a benchmark class.
# Defines module-level tuples of tested value types and index types. No other import-time side
# effects; runs the TensorFlow test runner when executed as a script.
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
"""Tests for tensorflow.ops.tf.gather_nd."""

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


_TEST_DTYPES = (
    dtypes.bfloat16.as_numpy_dtype,
    dtypes.float8_e5m2.as_numpy_dtype,
    dtypes.float8_e4m3fn.as_numpy_dtype,
    np.float16,
    np.float32,
    np.float64,
    np.int16,
    np.int32,
    np.int64,
    np.complex64,
    np.complex128,
    "|S",  # byte strings in python2 + 3
)
_TEST_ITYPES = (np.int16, np.int32, np.int64)


class GatherNdTest(test.TestCase):
  # [CLASS DOC]
  # Test case for gather_nd covering values, shapes, errors on invalid indices, and gradients.

  def _testSimpleDtype(self, dtype, itype):
    # [FUNCTION DOC]
    # HELPER: dtype-parametrised body of the simple gather test.
    # Takes a value dtype and an index dtype. Gathers three single positions from a vector of six values
    # and asserts the values and the static shape of the result.
    with self.cached_session():
      params = constant_op.constant(np.array([8, 1, 2, 3, 7, 5], dtype=dtype))
      indices = constant_op.constant([[4], [4], [0]], dtype=itype)
      gather_nd_t = array_ops.gather_nd(params, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    self.assertAllEqual(np.array([7, 7, 8], dtype=dtype), gather_nd_val)
    self.assertEqual([3], gather_nd_t.get_shape())

  def testSimpleDtype(self):
    # [FUNCTION DOC]
    # Runs the helper for every combination of tested value types (including bfloat16, float8, floats,
    # ints, complex and byte strings) and index types (int16, int32, int64).
    for dtype in _TEST_DTYPES:
      for itype in _TEST_ITYPES:
        self._testSimpleDtype(dtype, itype)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123337890")  # Error messages differ
  def testEmptyIndicesAndParamsOKButJustEmptyParamsFails(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the last assertion re-checks the earlier, stale gather result instead of anything from
    # the failing gather, so it is redundant copy-paste.
    # Checks empty index lists against non-empty params (shapes zero and zero by three), empty params with
    # empty indices (allowed), and finally empty params with non-empty indices, which must raise an
    # operation error saying more than zero entries were requested but params is empty.
    with self.session():
      params = np.ones((3, 3), dtype=np.float32)

      indices_empty = np.empty((0, 2), dtype=np.int32)
      gather_nd_ok_t = array_ops.gather_nd(params, indices_empty)
      gather_nd_ok_val = self.evaluate(gather_nd_ok_t)
      self.assertEqual([0], gather_nd_ok_t.get_shape())
      self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)

      indices_empty = np.empty((0, 1), dtype=np.int32)
      gather_nd_ok_t = array_ops.gather_nd(params, indices_empty)
      gather_nd_ok_val = self.evaluate(gather_nd_ok_t)
      self.assertEqual([0, 3], gather_nd_ok_t.get_shape())
      self.assertAllClose(np.empty((0, 3), dtype=np.float32), gather_nd_ok_val)

      params_empty = np.empty((0, 3), dtype=np.float32)
      indices_empty = np.empty((0, 2), dtype=np.int32)
      gather_nd_ok_t = array_ops.gather_nd(params_empty, indices_empty)
      gather_nd_ok_val = self.evaluate(gather_nd_ok_t)
      self.assertEqual([0], gather_nd_ok_t.get_shape())
      self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)

      params_empty = np.empty((0, 3), dtype=np.float32)
      indices_nonempty = np.zeros((1, 2), dtype=np.int32)
      gather_nd_break_t = array_ops.gather_nd(params_empty, indices_nonempty)
      with self.assertRaisesOpError(
          r"Requested more than 0 entries, but params is empty."):
        self.evaluate(gather_nd_break_t)
      self.assertAllClose(np.empty((0,), dtype=np.float32), gather_nd_ok_val)

  def testIndexScalar(self):
    # [FUNCTION DOC]
    # Gathers one element using a full-length index vector; expects a scalar of the right value.
    with self.session():
      params = np.array(
          [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
      indices = constant_op.constant([4, 1])
      gather_nd_t = array_ops.gather_nd(params, indices)
      gather_nd_val = self.evaluate(gather_nd_t)
      self.assertEqual([], gather_nd_t.get_shape())
      self.assertAllEqual(np.array(7), gather_nd_val)

  def testParamsRankLargerThanIndexIndexScalarSlices(self):
    # [FUNCTION DOC]
    # Uses a single-element index into a matrix; expects the whole selected row with shape two.
    with self.session():
      params = np.array(
          [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
      indices = constant_op.constant([4])
      gather_nd_t = array_ops.gather_nd(params, indices)
      gather_nd_val = self.evaluate(gather_nd_t)
      self.assertEqual([2], gather_nd_t.get_shape())
      self.assertAllEqual(np.array([-7, 7]), gather_nd_val)

  def testParamsRankLargerThanIndexSlices(self):
    # [FUNCTION DOC]
    # Uses a list of single-element indices into a matrix; expects stacked rows with shape three by two.
    with self.session():
      params = np.array(
          [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32).T
      indices = constant_op.constant([[4], [4], [0]])
      gather_nd_t = array_ops.gather_nd(params, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    self.assertEqual([3, 2], gather_nd_t.get_shape())
    self.assertAllEqual(np.array([[-7, 7], [-7, 7], [-8, 8]]), gather_nd_val)

  def testHigherRankParamsLargerThanIndexSlices(self):
    # [FUNCTION DOC]
    # Gathers slices from a rank 3 tensor using single-element indices; expects shape three by two by two
    # and values matching numpy fancy indexing.
    with self.session():
      params = np.array(
          [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
           [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
          dtype=np.float32).T
      params_t = constant_op.constant(params)
      indices = constant_op.constant([[4], [4], [0]])
      gather_nd_t = array_ops.gather_nd(params_t, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    self.assertEqual([3, 2, 2], gather_nd_t.get_shape())
    self.assertAllEqual(params[[4, 4, 0]], gather_nd_val)

  def testEmptyIndicesLastRankMeansCopyEntireTensor(self):
    # [FUNCTION DOC]
    # Uses an index tensor of shape two by zero, meaning each output is a full copy of the input. Expects
    # shape two by the input shape and values equal to two stacked copies.
    with self.session():
      params = np.array(
          [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
           [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
          dtype=np.float32).T
      params_t = constant_op.constant(params)
      indices = constant_op.constant(
          [[], []], dtype=dtypes.int32)  # Size (2, 0)
      gather_nd_t = array_ops.gather_nd(params_t, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    self.assertEqual([2, 6, 2, 2], gather_nd_t.get_shape())
    self.assertAllEqual(
        np.vstack((params[np.newaxis, :], params[np.newaxis, :])),
        gather_nd_val)

  def testHigherRankParamsAndIndicesLargerThanIndexSlices(self):
    # [FUNCTION DOC]
    # Uses a rank 3 index tensor with single-element indices into a rank 3 tensor; expects shape two by
    # three by two by two and values matching numpy indexing reshaped.
    with self.session():
      params = np.array(
          [[[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]],
           [[-80, -10, -20, -30, -70, -50], [80, 10, 20, 30, 70, 50]]],
          dtype=np.float32).T
      params_t = constant_op.constant(params)
      indices = constant_op.constant([[[3], [2], [1]], [[4], [4], [0]]])
      gather_nd_t = array_ops.gather_nd(params_t, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    self.assertEqual([2, 3, 2, 2], gather_nd_t.get_shape())
    self.assertAllEqual(params[[3, 2, 1, 4, 4, 0]].reshape(2, 3, 2, 2),
                        gather_nd_val)

  def testHigherRankParams(self):
    # [FUNCTION DOC]
    # Gathers 2000 random full indices from a random five-dimensional tensor; asserts values equal numpy
    # indexing and the static shape is 2000.
    with self.session():
      shape = (10, 20, 5, 1, 17)
      params = np.random.rand(*shape)
      indices = np.vstack([np.random.randint(0, s, size=2000) for s in shape]).T
      gather_nd_t = array_ops.gather_nd(params, indices)
      gather_nd_val = self.evaluate(gather_nd_t)

    expected = params[tuple(indices.T)]
    self.assertAllEqual(expected, gather_nd_val)
    self.assertEqual([2000], gather_nd_t.get_shape())

  def testHigherRankParamsAndIndices(self):
    # [FUNCTION DOC]
    # Same as above but indices are reshaped to a rank 4 index tensor; asserts values and shape ten by ten
    # by twenty.
    with self.session():
      shape = (10, 20, 5, 1, 17)
      params = np.random.rand(*shape)
      indices = np.vstack([np.random.randint(0, s, size=2000) for s in shape]).T
      indices_reshaped = indices.reshape([10, 10, 20, 5])
      gather_nd_t = array_ops.gather_nd(params, indices_reshaped)
      gather_nd_val = self.evaluate(gather_nd_t)

    expected = params[tuple(indices.T)]
    self.assertAllEqual(expected.reshape([10, 10, 20]), gather_nd_val)
    self.assertEqual([10, 10, 20], gather_nd_t.get_shape())

  def assertIndexedSlices(self, t):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the gradient tests. Takes a value and asserts it is an indexed-slices
    # object. Returns nothing.
    self.assertIsInstance(t, indexed_slices.IndexedSlices)

  @test_util.run_deprecated_v1
  def testUnknownIndices(self):
    # [FUNCTION DOC]
    # Uses a placeholder with unknown shape for the indices and asserts the result has unknown rank and
    # unknown first dimension.
    params = constant_op.constant([[0, 1, 2]])
    indices = array_ops.placeholder(dtypes.int32)
    gather_nd_t = array_ops.gather_nd(params, indices)
    shape = gather_nd_t.get_shape()
    self.assertEqual(None, shape.ndims)
    self.assertEqual(None, tensor_shape.dimension_value(shape[0]))

  @test_util.run_deprecated_v1
  @test_util.disable_xla("XLA does not have assertions in kernels.")
  def testBadIndicesCPU(self):
    # [FUNCTION DOC]
    # On CPU (XLA disabled) gathers an out-of-range index with the default policy and expects an
    # operation error naming the offending index and the params shape.
    with self.session(use_gpu=False):
      params = [0, 1, 2]
      indices = [[[0], [7]]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(params, indices)
      with self.assertRaisesOpError(
          r"indices\[0,1\] = \[7\] does not index into param shape \[3\]"):
        self.evaluate(gather_nd)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("XLA does not have assertions in kernels.")
  def testBadIndicesCPUWithDefaultPolicy(self):
    # [FUNCTION DOC]
    # Same as the previous test but passes the DEFAULT bad-indices policy explicitly; expects the same
    # operation error.
    with self.session(use_gpu=False):
      params = [0, 1, 2]
      indices = [[[0], [7]]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(
          params, indices, bad_indices_policy="DEFAULT"
      )
      with self.assertRaisesOpError(
          r"indices\[0,1\] = \[7\] does not index into param shape \[3\]"
      ):
        self.evaluate(gather_nd)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("XLA does not have assertions in kernels.")
  def testBadIndicesCPUWithExplicitErrorPolicy(self):
    # [FUNCTION DOC]
    # Same but with the ERROR bad-indices policy; expects the same operation error.
    with self.session(use_gpu=False):
      params = [0, 1, 2]
      indices = [[[0], [7]]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(
          params, indices, bad_indices_policy="ERROR"
      )
      with self.assertRaisesOpError(
          r"indices\[0,1\] = \[7\] does not index into param shape \[3\]"
      ):
        self.evaluate(gather_nd)

  @test_util.disable_xla("XLA does not have assertions in kernels.")
  def testBadIndicesCPUIgnore(self):
    # [FUNCTION DOC]
    # With the IGNORE policy an out-of-range index produces zero instead of an error; asserts the result
    # is ten, zero, one for params ten, one, two with indices zero, seven, one.
    with self.session(use_gpu=False):
      params = [10, 1, 2]
      indices = [[0], [7], [1]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(
          params, indices, bad_indices_policy="IGNORE"
      )
      gather_nd_val = self.evaluate(gather_nd)
    self.assertAllEqual(gather_nd_val, [10, 0, 1])

  def _disabledTestBadIndicesGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead test. The name lacks the test prefix so the runner never runs it, and its comment
    # says it is disabled because GPU and CPU behaviors differ (GPU returns zeros).
    # Returns early without a GPU; otherwise expects an operation error for an out-of-range index.
    # TODO disabled due to different behavior on GPU and CPU
    # On GPU the bad indices do not raise error but fetch 0 values
    if not test.is_gpu_available():
      return
    with self.session():
      params = [0, 1, 2]
      indices = [[[0], [7]]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(params, indices)
      with self.assertRaisesOpError(
          r"indices\[0,1\] = \[7\] does not index into param shape \[3\]"):
        self.evaluate(gather_nd)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("XLA does not have assertions in kernels.")
  def testBadIndicesWithSlicesCPU(self):
    # [FUNCTION DOC]
    # On CPU, an out-of-range index into a 1 by 3 matrix gathering slices raises an operation error naming
    # the index and shape.
    with self.session(use_gpu=False):
      params = [[0, 1, 2]]
      indices = [[[0], [0], [1]]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(params, indices)
      with self.assertRaisesOpError(
          r"indices\[0,2\] = \[1\] does not index into param shape \[1,3\]"):
        self.evaluate(gather_nd)

  def _disabledTestBadIndicesWithSlicesGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead test. The name lacks the test prefix so it is never run; disabled because GPU and
    # CPU behavior differ (GPU returns zeros instead of raising).
    # Returns early without a GPU; otherwise expects an operation error for an out-of-range slice index.
    # TODO disabled due to different behavior on GPU and CPU
    # On GPU the bad indices do not raise error but fetch 0 values
    if not test.is_gpu_available():
      return
    with self.session():
      params = [[0, 1, 2]]
      indices = [[[0], [0], [1]]]  # Make this one higher rank
      gather_nd = array_ops.gather_nd(params, indices)
      with self.assertRaisesOpError(
          r"indices\[0,2\] = \[1\] does not index into param shape \[1,3\]"):
        self.evaluate(gather_nd)

  @test_util.run_deprecated_v1
  def testGradientsRank2Elements(self):
    # [FUNCTION DOC]
    # Computes the gradient of gather_nd of a 2x2 matrix with element indices, using given upstream
    # gradients; asserts the gradient is a dense matrix with the upstream values scattered at the indices.
    # KNOWN ISSUE: uses a bare assert statement instead of the test case assertion helpers.
    indices = constant_op.constant([[0, 0], [1, 1]], dtype=dtypes.int32)
    inputs = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)

    grad_vals = constant_op.constant([1, 2], dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array([[1, 0], [0, 2]], dtype=np.float64)
    with self.session():
      assert np.array_equal(expected_grads, self.evaluate(grads))

  @test_util.run_deprecated_v1
  def testGradientsRank2Slices(self):
    # [FUNCTION DOC]
    # Gradient with row-slice indices; asserts it is an indexed-slices object whose dense form has the
    # upstream rows scattered at the indexed rows.
    indices = constant_op.constant([[1], [0]], dtype=dtypes.int32)
    inputs = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)

    grad_vals = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array([[3, 4], [1, 2]], dtype=np.float64)
    with self.session():
      self.assertIndexedSlices(grads)
      self.assertAllEqual(expected_grads, ops.convert_to_tensor(grads))

  @test_util.run_deprecated_v1
  def testGradientsRank3Elements(self):
    # [FUNCTION DOC]
    # Gradient with a rank 3 index tensor of element indices on a 2x2x2 input; asserts the dense gradient.
    indices = constant_op.constant(
        [[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=dtypes.int32)
    inputs = constant_op.constant(
        [[[1, 3], [5, 7]], [[2, 4], [6, 8]]], dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)

    grad_vals = constant_op.constant(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array(
        [[[5, 6], [1, 2]], [[3, 4], [7, 8]]], dtype=np.float64)
    with self.session():
      self.assertAllEqual(expected_grads, self.evaluate(grads))

  @test_util.run_deprecated_v1
  def testGradientsRank7Elements(self):
    # [FUNCTION DOC]
    # Gradient with a rank 7 index tensor (shape one, one, two, one, one, two, two) over a rank 7 input;
    # asserts the dense gradient equals the upstream values scattered back.
    # Shape [1,1,2,1,1,2,2]
    indices = constant_op.constant(
        [[[
            [[[[0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]]]],
            [[[[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1]]]]
        ]]],
        dtype=dtypes.int32)
    inputs = constant_op.constant(
        [[[
            [[[[1, 3], [5, 7]]]],
            [[[[2, 4], [6, 8]]]]
        ]]], dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)

    grad_vals = constant_op.constant(
        [[[
            [[[[1, 2], [3, 4]]]],
            [[[[5, 6], [7, 8]]]]
        ]]], dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array(
        [[[
            [[[[5, 6], [1, 2]]]],
            [[[[3, 4], [7, 8]]]]
        ]]], dtype=np.float64)
    with self.session():
      self.assertAllEqual(expected_grads, self.evaluate(grads))

  @test_util.run_deprecated_v1
  def testGradientsInt64Indices(self):
    # [FUNCTION DOC]
    # Same as the rank 3 gradient test but with 64-bit integer indices.
    indices = constant_op.constant(
        [[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=dtypes.int64)
    inputs = constant_op.constant(
        [[[1, 3], [5, 7]], [[2, 4], [6, 8]]], dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)

    grad_vals = constant_op.constant(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array(
        [[[5, 6], [1, 2]], [[3, 4], [7, 8]]], dtype=np.float64)
    with self.session():
      self.assertAllEqual(expected_grads, self.evaluate(grads))

  @test_util.run_deprecated_v1
  def testGradientsRank2SlicesWithEmptySpace(self):
    # [FUNCTION DOC]
    # Gradient with row indices on a 6x9 input where only rows two, zero and five are gathered; asserts an
    # indexed-slices gradient whose dense form is zero on the rows never gathered.
    indices = constant_op.constant([[2], [0], [5]], dtype=dtypes.int32)
    inputs = constant_op.constant(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9],
         [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9],
         [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
        dtype=dtypes.float64)
    outputs = array_ops.gather_nd(inputs, indices)
    grad_vals = constant_op.constant(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3, 3, 3, 3, 3]],
        dtype=dtypes.float64)
    grads = gradients_impl.gradients([outputs], [inputs], [grad_vals])[0]
    expected_grads = np.array(
        [[2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3, 3]],
        dtype=np.float64)
    with self.session():
      self.assertIndexedSlices(grads)
      self.assertAllEqual(expected_grads, ops.convert_to_tensor(grads))

  @test_util.run_v1_only("RefVariable is not supported in v2")
  def testGatherNdRefVariable(self):
    # [FUNCTION DOC]
    # Graph-mode only (skipped in v2). Gathers from a reference variable after initialization; checks the
    # operation name when not eager and that values are two and five.
    with self.cached_session():
      v = ref_variable.RefVariable(
          constant_op.constant([[1, 2], [3, 4], [5, 6]]))
      self.evaluate(variables.global_variables_initializer())
      gather = array_ops.gather_nd(v, [[0, 1], [2, 0]])
      if not context.executing_eagerly():  # .op doesn't make sense in Eager
        self.assertEqual("GatherNd", gather.op.name)
      self.assertAllEqual([2, 5], gather)

  @test_util.run_in_graph_and_eager_modes
  def testGatherNdResourceVariable(self):
    # [FUNCTION DOC]
    # In graph and eager modes, gathers from a resource variable after initialization; when not eager
    # checks the underlying operation is the resource gather op; asserts values two and five.
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(
          constant_op.constant([[1, 2], [3, 4], [5, 6]]))
      self.evaluate(variables.global_variables_initializer())
      gather = array_ops.gather_nd(v, [[0, 1], [2, 0]])
      if not context.executing_eagerly():  # .op doesn't make sense in Eager
        self.assertEqual("ResourceGatherNd", gather.op.inputs[0].op.type)
      self.assertAllEqual([2, 5], gather)


class GatherNdOpBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class measuring the speed of gather_nd on a large random five-dimensional tensor.

  def benchmark_gather_nd_op(self):
    # [FUNCTION DOC]
    # Seeds numpy, builds variables for random params and 10000 random indices, builds the gather op,
    # initializes variables, runs it ten times as warm-up and one thousand times timed, then reports the
    # iteration count and the average wall time per iteration through the benchmark reporting facility.
    shape = (100, 47, 18, 170, 13)
    np.random.seed(127)
    params = np.random.rand(*shape)
    indices = np.vstack([np.random.randint(0, s, size=10000) for s in shape]).T

    with session.Session():
      t_params = variables.Variable(params)
      t_indices = variables.Variable(indices)
      gather_op = array_ops.gather_nd(t_params, t_indices)
      self.evaluate(variables.global_variables_initializer())
      for _ in range(10):
        self.evaluate(gather_op)
      t1 = time.time()
      for _ in range(1000):
        self.evaluate(gather_op)
      t2 = time.time()
      self.report_benchmark(iters=1000, wall_time=(t2 - t1) / 1000.0)


if __name__ == "__main__":
  test.main()
