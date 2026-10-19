# [MODULE DOC]
# Test module for the shape-related array ops: shape, shape_n, rank and size (dense and sparse
# inputs), expand_dims, squeeze, and tile (values, dtypes, shape inference, gradients, error
# handling and a very large output). Contains a helper that turns a dense numpy array into a
# sparse tensor, a test class for the shape/expand/squeeze ops and a parameterized test class for
# tile. No import-time side effects beyond definitions; run as a test program.
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
"""Tests for various tensorflow.ops.tf."""

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import importer
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  # [FUNCTION DOC]
  # HELPER: converts a dense numpy array into a sparse tensor for the sparse shape/rank/size
  # comparisons. Inputs: x (numeric numpy array), thresh (default 0.5; entries below it are zeroed),
  # index_dtype (default 64-bit integer type used for the index matrix). Zeroes the small entries
  # of x IN PLACE (mutating the argument), then collects the coordinates and values of the
  # remaining non-zero entries. Returns a pair: the sparse tensor with the original dense shape,
  # and the number of non-zero values.
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values)


class ShapeOpsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the shape, shape_n, rank, size, expand_dims and squeeze ops. Private compare
  # helpers evaluate the TensorFlow op and the numpy equivalent on CPU and GPU and assert that
  # values and static shapes agree; the test methods feed them many ranks, dtypes, axis choices,
  # gradients and error cases. No shared fixtures.

  def _compareShape(self, x, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the shape op with numpy. Inputs: x (array-like), use_gpu (default false).
    # Evaluates shape with default (32-bit) and 64-bit output types and asserts both equal the
    # numpy shape, and that the static shape of the result matches. Returns nothing.
    np_ans = np.array(np.shape(x))
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.shape(x)
      tf_ans_64 = array_ops.shape(x, out_type=dtypes.int64)
      result = self.evaluate(tf_ans)
      result_64 = self.evaluate(tf_ans_64)
    self.assertAllEqual(np_ans, result)
    self.assertAllEqual(np_ans, result_64)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareShapeSparse(self, x_np, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the shape op on a sparse tensor with numpy. Inputs: x_np (numeric numpy
    # array; note it is modified in place by the sparse conversion), use_gpu (default false).
    # Builds a sparse tensor from it, evaluates its shape and asserts it equals the dense shape.
    np_ans = np.array(np.shape(x_np))
    x_tf, unused_nnz = _sparsify(x_np)
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.shape(x_tf)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareShapeN(self, x, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the multi-input shape op with numpy. Inputs: x (array-like), use_gpu
    # (default false). Calls shape_n on three copies of x with default and 64-bit output types and
    # asserts each of the three results equals the numpy shape with matching static shape.
    np_ans = np.array(np.shape(x))
    with self.cached_session(use_gpu=use_gpu) as sess:
      tf_ans = array_ops.shape_n([x, x, x])
      tf_ans_64 = array_ops.shape_n([x, x, x], out_type=dtypes.int64)
      result = self.evaluate(tf_ans)
      result_64 = self.evaluate(tf_ans_64)
    for i in range(3):
      self.assertAllEqual(np_ans, result[i])
      self.assertAllEqual(np_ans, result_64[i])
      self.assertShapeEqual(np_ans, tf_ans[i])

  def _compareRank(self, x, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the rank op with numpy. Inputs: x (array-like), use_gpu (default false).
    # Asserts the evaluated rank equals the number of dimensions and the static shape matches.
    np_ans = np.asarray(np.ndim(x))
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.rank(x)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareRankSparse(self, x_np, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the rank op on a sparse tensor with numpy. Inputs: x_np (numeric numpy
    # array, modified in place by the sparse conversion), use_gpu (default false). Asserts the
    # evaluated rank equals the dense number of dimensions.
    np_ans = np.asarray(np.ndim(x_np))
    x_tf, unused_nnz = _sparsify(x_np)
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.rank(x_tf)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareSize(self, x, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the size op with numpy. Inputs: x (array-like), use_gpu (default false).
    # Evaluates size with default and 64-bit output types and asserts both equal the element
    # count, and that the static shape of the result matches.
    np_ans = np.asarray(np.size(x))
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.size(x)
      result = self.evaluate(tf_ans)
      tf_ans_64 = array_ops.size(x, out_type=dtypes.int64)
      result_64 = self.evaluate(tf_ans_64)
    self.assertAllEqual(np_ans, result)
    self.assertAllEqual(np_ans, result_64)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareSizeSparse(self, x_np, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the size op on a sparse tensor with numpy. Inputs: x_np (numeric numpy
    # array, modified in place by the sparse conversion), use_gpu (default false). Asserts the
    # evaluated size equals the dense element count.
    np_ans = np.asarray(np.size(x_np))
    x_tf, unused_nnz = _sparsify(x_np)
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.size(x_tf)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _testCpu(self, x, compare_sparse):
    # [FUNCTION DOC]
    # HELPER: runs the shape, shape_n, rank and size comparisons on CPU for input x, plus the
    # sparse shape, rank and size comparisons when compare_sparse is true. Returns nothing.
    self._compareShape(x, use_gpu=False)
    self._compareShapeN(x, use_gpu=False)
    self._compareRank(x, use_gpu=False)
    self._compareSize(x, use_gpu=False)
    if compare_sparse:
      self._compareShapeSparse(x, use_gpu=False)
      self._compareRankSparse(x, use_gpu=False)
      self._compareSizeSparse(x, use_gpu=False)

  def _testGpu(self, x, compare_sparse):
    # [FUNCTION DOC]
    # HELPER: same set of comparisons as the CPU variant but requesting GPU placement (which
    # falls back to CPU when no GPU exists). Takes input x and the compare_sparse flag.
    self._compareShape(x, use_gpu=True)
    self._compareShapeN(x, use_gpu=True)
    self._compareRank(x, use_gpu=True)
    self._compareSize(x, use_gpu=True)
    if compare_sparse:
      self._compareShapeSparse(x, use_gpu=True)
      self._compareRankSparse(x, use_gpu=True)
      self._compareSizeSparse(x, use_gpu=True)

  def _testAll(self, x, compare_sparse=True):
    # [FUNCTION DOC]
    # HELPER: runs both the CPU and GPU comparison suites on input x. compare_sparse (default
    # true) controls whether the sparse-tensor comparisons run; string inputs turn it off.
    self._testCpu(x, compare_sparse)
    self._testGpu(x, compare_sparse)

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: random float arrays of rank one through six (dimensions 2, 3, 5, 7, 11, 13).
    # Expects shape, shape_n, rank and size, dense and sparse, to agree with numpy.
    self._testAll(np.random.randn(2))
    self._testAll(np.random.randn(2, 3))
    self._testAll(np.random.randn(2, 3, 5))
    self._testAll(np.random.randn(2, 3, 5, 7))
    self._testAll(np.random.randn(2, 3, 5, 7, 11))
    self._testAll(np.random.randn(2, 3, 5, 7, 11, 13))

  def testBool(self):
    # [FUNCTION DOC]
    # Scenario: random boolean arrays of rank one through six. Expects shape, shape_n, rank and
    # size, dense and sparse, to agree with numpy.
    self._testAll(np.random.choice((False, True), size=(2,)))
    self._testAll(np.random.choice((False, True), size=(2, 3)))
    self._testAll(np.random.choice((False, True), size=(2, 3, 5)))
    self._testAll(np.random.choice((False, True), size=(2, 3, 5, 7)))
    self._testAll(np.random.choice((False, True), size=(2, 3, 5, 7, 11)))
    self._testAll(np.random.choice((False, True), size=(2, 3, 5, 7, 11, 13)))

  def testString(self):
    # [FUNCTION DOC]
    # Scenario: random string arrays of rank one through six. Expects dense shape, shape_n, rank
    # and size to agree with numpy; sparse comparisons are skipped for strings.
    self._testAll(
        np.random.choice(["abcd", "efgh"], size=(2,)), compare_sparse=False
    )
    self._testAll(
        np.random.choice(["abcd", "efgh"], size=(2, 3)), compare_sparse=False
    )
    self._testAll(
        np.random.choice(["abcd", "efgh"], size=(2, 3, 5)), compare_sparse=False
    )
    self._testAll(
        np.random.choice(["abcd", "efgh"], size=(2, 3, 5, 7)),
        compare_sparse=False,
    )
    self._testAll(
        np.random.choice(["abcd", "efgh"], size=(2, 3, 5, 7, 11)),
        compare_sparse=False,
    )
    self._testAll(
        np.random.choice(["abcd", "efgh"], size=(2, 3, 5, 7, 11, 13)),
        compare_sparse=False,
    )

  # Disabled because it takes too long to run, but manually verified
  # as passing at time of writing.
  def _test64BitOutput(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead code. The leading underscore keeps the runner from collecting it, so it
    # never runs (the comment says it is too slow, manually verified once).
    # Intended scenario: a vector of two to the power thirty-one zeros. Its size with 64-bit
    # output must equal that count, while requesting 32-bit output must raise an invalid-argument
    # error because the count overflows.
    with self.cached_session():
      inp = array_ops.zeros([2**31])
      num_elements = array_ops.size_internal(
          inp, optimize=False, out_type=dtypes.int64)
      self.assertEqual(2**31, self.evaluate(num_elements))

    # Too large for tf.int32 output.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      with self.cached_session():
        inp = array_ops.zeros([2**31])
        num_elements = array_ops.size_internal(
            inp, optimize=False, out_type=dtypes.int32)
        self.assertEqual(2**31, self.evaluate(num_elements))

  def _compareExpandDims(self, x, dim, use_gpu):
    # [FUNCTION DOC]
    # HELPER: compares expand_dims with numpy. Inputs: x (array-like), dim (axis at which to
    # insert a size-one dimension, may be negative), use_gpu (whether to request GPU placement).
    # Asserts the evaluated result equals the numpy result and the static shapes match.
    np_ans = np.expand_dims(x, axis=dim)
    with self.cached_session(use_gpu=use_gpu):
      tensor = array_ops.expand_dims(x, dim)
      tf_ans = self.evaluate(tensor)
    self.assertShapeEqual(np_ans, tensor)
    self.assertAllEqual(np_ans, tf_ans)

  def _compareExpandDimsAll(self, x, dim):
    # [FUNCTION DOC]
    # HELPER: runs the expand_dims comparison for input x and axis dim on CPU and then GPU.
    self._compareExpandDims(x, dim, False)
    self._compareExpandDims(x, dim, True)

  def testExpandDims(self):
    # [FUNCTION DOC]
    # Scenario: expand_dims on zero arrays of rank one, two and three for every valid positive and
    # negative axis (from minus rank-minus-one to rank). Expects values and shapes to match numpy.
    self._compareExpandDimsAll(np.zeros([2]), 0)
    self._compareExpandDimsAll(np.zeros([2]), 1)
    self._compareExpandDimsAll(np.zeros([2]), -1)

    self._compareExpandDimsAll(np.zeros([2, 3]), 0)
    self._compareExpandDimsAll(np.zeros([2, 3]), 1)
    self._compareExpandDimsAll(np.zeros([2, 3]), 2)
    self._compareExpandDimsAll(np.zeros([2, 3]), -1)
    self._compareExpandDimsAll(np.zeros([2, 3]), -2)

    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 0)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 1)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 2)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 3)

    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -1)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -2)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -3)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -4)

  def testExpandDimsBool(self):
    # [FUNCTION DOC]
    # Same scenario as the numeric expand_dims test but with random boolean arrays of rank one,
    # two and three (a small local lambda generates the random input).
    choice = lambda s: np.random.choice((False, True), size=s)
    self._compareExpandDimsAll(choice([2]), 0)
    self._compareExpandDimsAll(choice([2]), 1)
    self._compareExpandDimsAll(choice([2]), -1)

    self._compareExpandDimsAll(choice([2, 3]), 0)
    self._compareExpandDimsAll(choice([2, 3]), 1)
    self._compareExpandDimsAll(choice([2, 3]), 2)
    self._compareExpandDimsAll(choice([2, 3]), -1)
    self._compareExpandDimsAll(choice([2, 3]), -2)

    self._compareExpandDimsAll(choice([2, 3, 5]), 0)
    self._compareExpandDimsAll(choice([2, 3, 5]), 1)
    self._compareExpandDimsAll(choice([2, 3, 5]), 2)
    self._compareExpandDimsAll(choice([2, 3, 5]), 3)

    self._compareExpandDimsAll(choice([2, 3, 5]), -1)
    self._compareExpandDimsAll(choice([2, 3, 5]), -2)
    self._compareExpandDimsAll(choice([2, 3, 5]), -3)
    self._compareExpandDimsAll(choice([2, 3, 5]), -4)

  @test_util.run_deprecated_v1
  def testExpandDimsErrors(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): expand_dims with an axis outside the allowed range, minus five
    # or four, on a rank-three numpy array and on a three-element boolean list. Expects a value
    # error in each case.
    with self.cached_session():
      self.assertRaises(ValueError, array_ops.expand_dims,
                        np.zeros([2, 3, 5]), -5)
      self.assertRaises(ValueError, array_ops.expand_dims,
                        [False, True, True], -5)
      self.assertRaises(ValueError, array_ops.expand_dims,
                        np.zeros([2, 3, 5]), 4)
      self.assertRaises(ValueError, array_ops.expand_dims,
                        [False, True, True], 4)

  @test_util.run_deprecated_v1
  def testExpandDimsGradient(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): expand_dims at axis one of a random 4x2 float tensor. Expects
    # the numerically estimated gradient error to be below 1e-3.
    with self.cached_session():
      inp = constant_op.constant(
          np.random.rand(4, 2).astype("f"), dtype=dtypes.float32)
      squeezed = array_ops.expand_dims(inp, 1)

      err = gradient_checker.compute_gradient_error(inp, [4, 2], squeezed,
                                                    [4, 1, 2])
    self.assertLess(err, 1e-3)

  @test_util.run_deprecated_v1
  def testExpandDimsScalar(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): expand_dims at axis zero and minus one on a scalar integer and
    # a scalar boolean. Expects a one-element vector holding the same value each time.
    with self.cached_session():
      inp = constant_op.constant(7)
      self.assertAllEqual([7], array_ops.expand_dims(inp, 0))
      self.assertAllEqual([7], array_ops.expand_dims(inp, -1))

      inp = constant_op.constant(True)
      self.assertAllEqual([True], array_ops.expand_dims(inp, 0))
      self.assertAllEqual([True], array_ops.expand_dims(inp, -1))

  def testExpandDimsDimType(self):
    # [FUNCTION DOC]
    # Scenario: expand_dims at axis zero of a length-two zero vector where the axis is a tensor
    # constant of 32-bit or 64-bit integer type. Expects the same result as numpy for both.
    for dtype in [dtypes.int32, dtypes.int64]:
      x = np.zeros([2])
      np_ans = np.expand_dims(x, axis=0)
      with self.cached_session():
        tensor = array_ops.expand_dims(x, constant_op.constant(0, dtype))
        tf_ans = self.evaluate(tensor)
      self.assertShapeEqual(np_ans, tensor)
      self.assertAllEqual(np_ans, tf_ans)

  def _compareSqueeze(self, x, squeeze_dims, use_gpu):
    # [FUNCTION DOC]
    # HELPER: compares squeeze with numpy. Inputs: x (array-like), squeeze_dims (list of axes to
    # squeeze; an empty list means squeeze all size-one dimensions), use_gpu (GPU placement flag).
    # Asserts the evaluated result equals numpy and the static shapes match.
    with self.cached_session(use_gpu=use_gpu):
      if squeeze_dims:
        np_ans = np.squeeze(x, axis=tuple(squeeze_dims))
        tensor = array_ops.squeeze(x, squeeze_dims)
        tf_ans = self.evaluate(tensor)
      else:
        np_ans = np.squeeze(x)
        tensor = array_ops.squeeze(x)
        tf_ans = self.evaluate(tensor)
    self.assertShapeEqual(np_ans, tensor)
    self.assertAllEqual(np_ans, tf_ans)

  def _compareSqueezeAll(self, x, squeeze_dims=None):
    # [FUNCTION DOC]
    # HELPER: runs the squeeze comparison on CPU and GPU. squeeze_dims is optional (default
    # nothing, treated as an empty list meaning squeeze every size-one dimension).
    if squeeze_dims is None:
      squeeze_dims = []
    self._compareSqueeze(x, squeeze_dims, False)
    self._compareSqueeze(x, squeeze_dims, True)

  def testSqueeze(self):
    # [FUNCTION DOC]
    # Scenario: squeeze with no axes on arrays with nothing to squeeze, a middle size-one
    # dimension, and size-one dimensions at both ends. Expects numpy-identical results.
    # Nothing to squeeze.
    self._compareSqueezeAll(np.zeros([2]))
    self._compareSqueezeAll(np.zeros([2, 3]))

    # Squeeze the middle element away.
    self._compareSqueezeAll(np.zeros([2, 1, 2]))

    # Squeeze on both ends.
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]))

  def testSqueezeBool(self):
    # [FUNCTION DOC]
    # Same scenarios as the numeric squeeze test but with random boolean arrays.
    choice = lambda s: np.random.choice((False, True), size=s)
    # Nothing to squeeze.
    self._compareSqueezeAll(choice([2]))
    self._compareSqueezeAll(choice([2, 3]))

    # Squeeze the middle element away.
    self._compareSqueezeAll(choice([2, 1, 2]))

    # Squeeze on both ends.
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]))

  def testSqueezeSpecificDimension(self):
    # [FUNCTION DOC]
    # Scenario: squeeze of a 1x2x1x3x1 zero array with explicit axis lists, positive (such as
    # zero; two and four) and negative (such as minus one; minus three and minus five). Expects
    # numpy-identical results.
    # Positive squeeze dim index.
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [0])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [2, 4])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [0, 4, 2])

    # Negative squeeze dim index.
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [-1])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [-3, -5])
    self._compareSqueezeAll(np.zeros([1, 2, 1, 3, 1]), [-3, -5, -1])

  def testSqueezeSpecificDimensionBool(self):
    # [FUNCTION DOC]
    # Same scenario as the numeric explicit-axis squeeze test but with random boolean arrays.
    choice = lambda s: np.random.choice((False, True), size=s)
    # Positive squeeze dim index.
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [0])
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [2, 4])
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [0, 4, 2])

    # Negative squeeze dim index.
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [-1])
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [-3, -5])
    self._compareSqueezeAll(choice([1, 2, 1, 3, 1]), [-3, -5, -1])

  def testSqueezeAllOnes(self):
    # [FUNCTION DOC]
    # Scenario: squeeze a 1x1x1 zero array with an empty axis list on CPU and GPU. Expects a
    # zero-dimensional result both statically and after evaluation, as numpy does.
    # Numpy squeezes a 1 element tensor into a zero dimensional tensor.
    # Verify that we do the same.
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        tensor = array_ops.squeeze(np.zeros([1, 1, 1]), [])
        self.assertEqual(np.shape(1), tensor.get_shape())
        tf_ans = self.evaluate(tensor)
        self.assertEqual(np.shape(1), tf_ans.shape)

  def testSqueezeAllOnesBool(self):
    # [FUNCTION DOC]
    # Same as the all-ones squeeze test but with a nested single-element boolean list: expects a
    # zero-dimensional result statically and after evaluation, on CPU and GPU.
    # Numpy squeezes a 1 element tensor into a zero dimensional tensor.
    # Verify that we do the same.
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        tensor = array_ops.squeeze([[[False]]], [])
        self.assertEqual(np.shape(1), tensor.get_shape())
        tf_ans = self.evaluate(tensor)
        self.assertEqual(np.shape(1), tf_ans.shape)

  @test_util.run_deprecated_v1
  def testSqueezeOnlyOnes(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): on a 1x1x3 zero array squeeze with no axes, with axis zero and
    # with axis one must match numpy; squeezing axis two (size three) must raise a value error.
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        input_1x1x3 = np.zeros([1, 1, 3])
        self._compareSqueezeAll(input_1x1x3)
        self._compareSqueezeAll(input_1x1x3, [0])
        self._compareSqueezeAll(input_1x1x3, [1])
        self.assertRaises(ValueError, array_ops.squeeze, input_1x1x3, [2])

  @test_util.run_deprecated_v1
  def testSqueezeErrors(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): squeeze of a 1x2x1 array with out-of-range axes (minus four,
    # three, and lists containing them). Expects a value error each time.
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        self.assertRaises(ValueError, array_ops.squeeze,
                          np.zeros([1, 2, 1]), [-4])
        self.assertRaises(ValueError, array_ops.squeeze,
                          np.zeros([1, 2, 1]), [0, -4])
        self.assertRaises(ValueError, array_ops.squeeze,
                          np.zeros([1, 2, 1]), [3])
        self.assertRaises(ValueError, array_ops.squeeze,
                          np.zeros([1, 2, 1]), [2, 3])

  @test_util.run_deprecated_v1
  def testSqueezeGradient(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): squeeze of a 4x1x2 tensor with no axes. Expects the numerically
    # estimated gradient error to be below 1e-3.
    with self.cached_session():
      inp = np.random.rand(4, 2).astype("f")
      a = array_ops.reshape(inp, [4, 1, 2])
      squeezed = array_ops.squeeze(a, [])

      err = gradient_checker.compute_gradient_error(a, [4, 1, 2], squeezed,
                                                    [4, 2])
    self.assertLess(err, 1e-3)

  @test_util.run_deprecated_v1
  def testSqueezeGradientWithSqueezeDims(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): squeeze axis one of a 4x1x2x1 tensor. Expects the numerically
    # estimated gradient error to be below 1e-3.
    with self.cached_session():
      inp = np.random.rand(4, 2).astype("f")
      a = array_ops.reshape(inp, [4, 1, 2, 1])
      squeezed = array_ops.squeeze(a, [1])

      err = gradient_checker.compute_gradient_error(a, [4, 1, 2, 1], squeezed,
                                                    [4, 2, 1])
    self.assertLess(err, 1e-3)

  @test_util.run_deprecated_v1
  def testSqueezeWithUnknownShape(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): squeeze of a float placeholder of shape two by unknown.
    # Squeezing axis one gives static shape of length two; squeezing with no axes gives a fully
    # unknown shape; squeezing axis zero (size two) or an out-of-range axis raises a value error.
    with self.cached_session():
      a = array_ops.placeholder(dtypes.float32, shape=[2, None])

      squeezed = array_ops.squeeze(a, [1])
      self.assertEqual([2], squeezed.get_shape().as_list())

      squeezed = array_ops.squeeze(a)
      self.assertEqual(None, squeezed.get_shape())

      self.assertRaises(ValueError, array_ops.squeeze, a, [0])
      self.assertRaises(ValueError, array_ops.squeeze, a, [100])


class TileTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the tile op: scalars, simple and empty cases, multiples of either integer type,
  # shape inference with unknown shapes (including importing a graph with a reordered tile
  # node), all common dtypes, invalid multiples, random ranks, gradients (analytic and
  # numerical, with sparse upstream gradients), and an oversized output error.

  def testScalar(self):
    # [FUNCTION DOC]
    # Scenario: tile a scalar float with an empty multiples list on CPU and GPU. Expects a
    # scalar result of the same value with an empty static shape.
    for use_gpu in False, True:
      with self.cached_session(use_gpu=use_gpu):
        a = constant_op.constant(7, shape=[], dtype=dtypes.float32)
        tiled = array_ops.tile(a, [])
        result = self.evaluate(tiled)
      self.assertEqual(result.shape, ())
      self.assertEqual([], tiled.get_shape())
      self.assertEqual(7, result)

  def testSimple(self):
    # [FUNCTION DOC]
    # Scenario: tile a random 4x1 float array with multiples one and four given as 32-bit or
    # 64-bit integer tensors. Expects a 4x4 result equal to numpy tiling, with static shape 4x4.
    # multiples could be int32 or int64
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.cached_session():
        inp = np.random.rand(4, 1).astype(np.float32)
        a = constant_op.constant(inp)
        tiled = array_ops.tile(a, constant_op.constant([1, 4], dtype=dtype))
        result = self.evaluate(tiled)
      self.assertEqual(result.shape, (4, 4))
      self.assertEqual([4, 4], tiled.get_shape())
      self.assertTrue((result == np.tile(inp, (1, 4))).all())

  def testIdentityTileAndGrad(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name promises a gradient check, but the body only checks the forward
    # result; no gradient is computed.
    # Scenario: tile a random 4x1 float array with multiples of all ones. Expects an unchanged
    # 4x1 result equal to numpy tiling, with static shape 4x1.
    with self.cached_session():
      inp = np.random.rand(4, 1).astype(np.float32)
      a = constant_op.constant(inp)
      tiled = array_ops.tile(a, [1, 1])
      result = self.evaluate(tiled)
    self.assertEqual(result.shape, (4, 1))
    self.assertEqual([4, 1], tiled.get_shape())
    self.assertTrue((result == np.tile(inp, (1, 1))).all())

  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: tile a random 2x3 float array with multiples five and zero. Expects an empty
    # result of shape 10x0 with matching static shape.
    with self.cached_session():
      inp = np.random.rand(2, 3).astype(np.float32)
      a = constant_op.constant(inp)
      tiled = array_ops.tile(a, [5, 0])
      result = self.evaluate(tiled)
    self.assertEqual(result.shape, (10, 0))
    self.assertEqual([10, 0], tiled.get_shape())

  @test_util.run_deprecated_v1
  def testUnknownInputShape(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): tile a placeholder of unknown shape with constant multiples of
    # four entries, serialize the graph, and swap the tile node to the front of the node list so
    # its inputs are not yet known when the shape function runs during import. Asserts the swap
    # happened, imports the graph, and expects the imported tile output to have rank four.
    """Importing can call _TileShape without shape of <multiples> known."""
    with self.cached_session():
      inp = array_ops.placeholder(dtypes.float32)  # unknown shape
      multiples = constant_op.constant([1, 2, 3, 4], dtype=np.int32)
      tiled = array_ops.tile(inp, multiples)
      gdef = tiled.graph.as_graph_def()

      # Move the tile op to the start of the graph so that shapes of its inputs
      # are not available when the shape function runs on import.
      swapped = False
      for i, n in enumerate(gdef.node):
        if n.op == "Tile":
          # Swap tile op to be first in gdef.node
          assert i != 0
          new_node = node_def_pb2.NodeDef()
          new_node.CopyFrom(gdef.node[i])
          gdef.node[i].CopyFrom(gdef.node[0])
          gdef.node[0].CopyFrom(new_node)
          swapped = True
      assert swapped

      tiled_imported, = importer.import_graph_def(
          gdef, return_elements=[tiled.name])
      self.assertEqual(4, tiled_imported.get_shape().ndims)

  def testTypes(self):
    # [FUNCTION DOC]
    # Scenario: tile a 4x1 array with multiples one and four for each of bool, bfloat16, float16,
    # float32, float64, complex64, complex128, eight to sixty-four bit signed and unsigned integers
    # and strings, using a table mapping numpy type to TensorFlow dtype and a Python conversion
    # function. Expects a 4x4 result equal to numpy tiling with matching static shape.
    types_to_test = {
        "bool": (dtypes.bool, bool),
        "bfloat16": (dtypes.bfloat16, float),
        "float16": (dtypes.float16, float),
        "float32": (dtypes.float32, float),
        "float64": (dtypes.float64, float),
        "complex64": (dtypes.complex64, complex),
        "complex128": (dtypes.complex128, complex),
        "uint8": (dtypes.uint8, int),
        "int8": (dtypes.int8, int),
        "int16": (dtypes.int16, int),
        "int32": (dtypes.int32, int),
        "int64": (dtypes.int64, int),
        "uint32": (dtypes.uint32, int),
        "uint64": (dtypes.uint64, int),
        bytes: (dtypes.string, bytes)
    }
    for dtype_np, (dtype_tf, cast) in types_to_test.items():
      with self.cached_session():
        inp = np.random.rand(4, 1).astype(dtype_np)
        a = constant_op.constant(
            [cast(x) for x in inp.ravel(order="C")],
            shape=[4, 1],
            dtype=dtype_tf)
        tiled = array_ops.tile(a, [1, 4])
        result = self.evaluate(tiled)
      self.assertEqual(result.shape, (4, 4))
      self.assertEqual([4, 4], tiled.get_shape())
      self.assertAllEqual(result, np.tile(inp, (1, 4)))

  @test_util.run_deprecated_v1
  def testInvalidDim(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): tile a 4x1 float tensor with multiples of the wrong length
    # (three entries) and with a rank-two multiples argument. Expects a value error in each case.
    with self.cached_session():
      inp = np.random.rand(4, 1).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.ravel(order="C")],
          shape=[4, 1],
          dtype=dtypes.float32)
      # Wrong length of multiples.
      with self.assertRaises(ValueError):
        array_ops.tile(a, [1, 4, 2])
      # Wrong rank for multiples.
      with self.assertRaises(ValueError):
        array_ops.tile(a, [[2, 3], [3, 4]]).eval()

  def _RunAndVerifyResult(self, rank, use_gpu):
    # [FUNCTION DOC]
    # HELPER: random tile check. Inputs: rank (number of dimensions) and use_gpu (placement flag).
    # Draws random dimensions (one to three each) and random multiples (one to three each),
    # tiles a random float tensor, and asserts the result shape is the elementwise product of
    # dimensions and multiples, the values equal numpy tiling, and static shape matches.
    # Uses unseeded random numbers.
    with self.cached_session(use_gpu=use_gpu):
      # Random dims of given rank
      input_shape = np.random.randint(1, 4, size=rank)
      inp = np.random.rand(*input_shape).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.ravel(order="C")],
          shape=input_shape,
          dtype=dtypes.float32)
      multiples = np.random.randint(1, 4, size=rank).astype(np.int32)
      tiled = array_ops.tile(a, multiples)
      result = self.evaluate(tiled)
    self.assertTrue((np.array(multiples) * np.array(inp.shape) == np.array(
        result.shape)).all())
    self.assertAllEqual(result, np.tile(inp, tuple(multiples)))
    self.assertShapeEqual(result, tiled)

  def testRandom(self):
    # [FUNCTION DOC]
    # Scenario: five random tile checks each on CPU and GPU at rank five, then again at rank ten.
    # Expects every check to pass.
    # test low rank, like 5
    for _ in range(5):
      self._RunAndVerifyResult(5, use_gpu=False)
    for _ in range(5):
      self._RunAndVerifyResult(5, use_gpu=True)
    # test high rank, like 10
    for _ in range(5):
      self._RunAndVerifyResult(10, use_gpu=False)
    for _ in range(5):
      self._RunAndVerifyResult(10, use_gpu=True)

  @parameterized.parameters(dtypes.int32, dtypes.int64)
  @test_util.run_deprecated_v1
  def testGradientSimpleReduction(self, multiples_dtype):
    # [FUNCTION DOC]
    # Parameterized by the multiples integer type (32-bit or 64-bit); graph-mode only.
    # Scenario: tile a 4x1 float tensor by one and four, then back-propagate a random 4x4
    # upstream gradient. Expects the input gradient to have the input's shape and to equal the
    # row sums of the upstream gradient (within 1e-3).
    with self.cached_session():
      inp = np.random.rand(4, 1).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.flatten()], shape=[4, 1], dtype=dtypes.float32)
      multiples = constant_op.constant([1, 4], dtype=multiples_dtype)
      tiled = array_ops.tile(a, multiples)
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = constant_op.constant(
          [float(x) for x in grad_inp.flatten()], shape=grad_shape)
      grad = gradients_impl.gradients([tiled], [a], [grad_tensor])[0]
      self.assertShapeEqual(inp, grad)
      result = self.evaluate(grad)
    self.assertAllClose(np.sum(grad_inp, axis=1).reshape(4, 1), result, 1e-3)

  @test_util.run_deprecated_v1
  def testGradientStridedReduction(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): tile a 4x2 float tensor by one and two and back-propagate a
    # random 4x4 upstream gradient. Expects the input gradient to have the input's shape, with
    # column zero equal to upstream columns zero plus two and column one equal to columns one
    # plus three (within 1e-3).
    with self.cached_session():
      inp = np.random.rand(4, 2).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.flatten()], shape=[4, 2], dtype=dtypes.float32)
      tiled = array_ops.tile(a, [1, 2])
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = constant_op.constant(
          [float(x) for x in grad_inp.flatten()], shape=grad_shape)
      grad = gradients_impl.gradients([tiled], [a], [grad_tensor])[0]
      self.assertShapeEqual(inp, grad)
      result = self.evaluate(grad)
    expected_shape = [4, 2]
    expected = np.zeros(expected_shape)
    expected[:, 0] = grad_inp[:, 0] + grad_inp[:, 2]
    expected[:, 1] = grad_inp[:, 1] + grad_inp[:, 3]
    self.assertTrue((np.abs(expected - result) < 1e-3).all())

  @test_util.run_deprecated_v1
  def testGradientSimpleReductionOnGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name it never forces GPU placement; it uses a default session, so
    # it runs on whatever device the placer picks (usually GPU only if one is available).
    # Scenario (graph-mode only): same as the simple reduction gradient test with multiples one
    # and four on a 4x1 tensor; expects the input gradient to equal the row sums of the random
    # upstream gradient (within 1e-3). It does not check the gradient's static shape.
    with self.session():
      inp = np.random.rand(4, 1).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.flatten()], shape=[4, 1], dtype=dtypes.float32)
      tiled = array_ops.tile(a, [1, 4])
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = constant_op.constant(
          [float(x) for x in grad_inp.flatten()], shape=grad_shape)
      grad = gradients_impl.gradients([tiled], [a], [grad_tensor])[0]
      result = self.evaluate(grad)
    self.assertAllClose(np.sum(grad_inp, axis=1).reshape(4, 1), result, 1e-3)

  @test_util.run_deprecated_v1
  def testGradientStridedReductionOnGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name it never forces GPU placement; it uses a default session.
    # Scenario (graph-mode only): tile a 4x2 tensor by one and two and back-propagate a random
    # 4x4 gradient; expects column zero to equal upstream columns zero plus two and column one to
    # equal columns one plus three (within 1e-3).
    with self.session():
      inp = np.random.rand(4, 2).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.flatten()], shape=[4, 2], dtype=dtypes.float32)
      tiled = array_ops.tile(a, [1, 2])
      grad_shape = [4, 4]
      grad_inp = np.random.rand(*grad_shape).astype("f")
      grad_tensor = constant_op.constant(
          [float(x) for x in grad_inp.flatten()], shape=grad_shape)
      grad = gradients_impl.gradients([tiled], [a], [grad_tensor])[0]
      result = self.evaluate(grad)
    expected_shape = [4, 2]
    expected = np.zeros(expected_shape)
    expected[:, 0] = grad_inp[:, 0] + grad_inp[:, 2]
    expected[:, 1] = grad_inp[:, 1] + grad_inp[:, 3]
    self.assertAllClose(expected, result, 1e-3)

  def _RunAndVerifyGradientResult(self, input_shape, multiples):
    # [FUNCTION DOC]
    # HELPER: numerical gradient check for tile. Inputs: input_shape (list of dimensions of the
    # random double-precision input) and multiples (list of repeat counts, same length). On CPU
    # then GPU it tiles the input, computes the numerical-versus-analytic gradient error and
    # asserts it is below 1e-3. Side effect: prints the error to standard output.
    for use_gpu in False, True:
      with self.cached_session(use_gpu=use_gpu):
        # Random values
        inp = np.asarray(np.random.rand(*input_shape))
        a = constant_op.constant(inp, dtype=dtypes.float64)
        tiled = array_ops.tile(a, multiples)
        grad_shape = list(np.array(multiples) * np.array(inp.shape))
        err = gradient_checker.compute_gradient_error(
            a, list(input_shape), tiled, grad_shape, x_init_value=inp)
      print("tile(float) error = ", err)
      self.assertLess(err, 1e-3)

  @test_util.run_deprecated_v1
  def testGradientRandomScalar(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): numerical gradient check of tile on a scalar input with empty
    # multiples. Expects the gradient error to be below 1e-3.
    self._RunAndVerifyGradientResult([], [])

  @test_util.run_deprecated_v1
  def testGradientRandom(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): numerical gradient checks of tile on rank-five inputs with
    # four different multiples lists (including all ones). Expects errors below 1e-3.
    self._RunAndVerifyGradientResult([2, 2, 1, 1, 3], [1, 1, 1, 1, 1])
    self._RunAndVerifyGradientResult([2, 2, 1, 1, 3], [1, 2, 1, 3, 1])
    self._RunAndVerifyGradientResult([2, 3, 1, 1, 3], [3, 1, 1, 2, 2])
    self._RunAndVerifyGradientResult([2, 1, 3, 3, 2], [1, 3, 3, 1, 2])

  @test_util.run_deprecated_v1
  def testGradientStridedReductionGC(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): numerical gradient check of tiling a 4x2 float tensor by one and
    # two (output 4x4) using the gradient checker. Expects the error to be below 1e-3.
    with self.cached_session():
      inp = np.random.rand(4, 2).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.flatten()], shape=[4, 2], dtype=dtypes.float32)
      tiled = array_ops.tile(a, [1, 2])
      err = gradient_checker.compute_gradient_error(a, [4, 2], tiled, [4, 4])
    self.assertLess(err, 1e-3)

  @parameterized.parameters(dtypes.int32, dtypes.int64)
  @test_util.run_deprecated_v1
  def testGradientWithSparseGradWithRank1(self, multiples_dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the parameterized multiples_dtype argument is never used; the multiples
    # constant is always 64-bit, so both parameter cases run the identical test.
    # Scenario (graph-mode only): tile a four-element vector by three and gather eight entries
    # (with a repeated index) from the result, so the incoming gradient is sparse. Expects the
    # numerical gradient error to be below 1e-4.
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0],
                                  dtype=dtypes.float32)
    multiples = constant_op.constant([3], dtype=dtypes.int64)
    outputs = array_ops.gather(array_ops.tile(inputs, multiples),
                               [1, 5, 9, 3, 7, 2, 2, 2])
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithSparseGradWithRank3(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): reshape a four-element vector to 4x1x1, tile by three, four and
    # two, and gather eight rows (with a repeated index) so the incoming gradient is sparse.
    # Expects the numerical gradient error to be below 1e-4.
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0],
                                  dtype=dtypes.float32)
    inputs = array_ops.reshape(inputs, [-1, 1, 1])
    outputs = array_ops.gather(array_ops.tile(inputs, [3, 4, 2]),
                               [1, 5, 9, 3, 7, 2, 2, 2])
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only): static shape inference of tile with partial information.
    # Unknown multiples gives four unknown dimensions for a known rank-four input; unknown input
    # shape with four multiples gives four unknown dimensions; both unknown gives unknown rank;
    # known 1x1 input with multiples of an unknown value and seven gives unknown by seven;
    # a rank-two input with a length-three multiples placeholder raises a value error.
    # Unknown multiples shape.
    inp = constant_op.constant(0.0, shape=[4, 4, 4, 4])
    tiled = array_ops.tile(inp, array_ops.placeholder(dtypes.int32))
    self.assertEqual([None, None, None, None], tiled.get_shape().as_list())

    # Unknown input shape.
    inp = array_ops.placeholder(dtypes.float32)
    tiled = array_ops.tile(inp, [2, 2, 2, 2])
    self.assertEqual([None, None, None, None], tiled.get_shape().as_list())

    # Unknown input and multiples shape.
    inp = array_ops.placeholder(dtypes.float32)
    tiled = array_ops.tile(inp, array_ops.placeholder(dtypes.int32))
    self.assertIs(None, tiled.get_shape().ndims)

    # Known input and partially known multiples.
    inp = constant_op.constant(0.0, shape=[1, 1])
    tiled = array_ops.tile(inp, [array_ops.placeholder(dtypes.int32), 7])
    self.assertEqual([None, 7], tiled.get_shape().as_list())

    # Mismatched input rank and multiples length.
    inp = array_ops.placeholder(dtypes.float32, shape=[None, None])
    with self.assertRaises(ValueError):
      tiled = array_ops.tile(
          inp, array_ops.placeholder(
              dtypes.int32, shape=[3]))

  def testLargeTensor(self):
    # [FUNCTION DOC]
    # Regression test for a reported GitHub issue (46911). Returns immediately (does nothing)
    # when XLA is enabled because it fails there. Otherwise tiling a 1x1x1 array by one hundred
    # million along each axis must raise an invalid-argument error when evaluated.
    # Test case for GItHub issue 46911.
    if test_util.is_xla_enabled():
      # The following test fails with XLA enabled.
      return
    with self.assertRaises(errors_impl.InvalidArgumentError):
      with self.cached_session():
        tiled = array_ops.tile(
            np.ones((1, 1, 1)), [100000000, 100000000, 100000000])
        self.evaluate(tiled)


if __name__ == "__main__":
  test.main()
