# [MODULE DOC]
# Functional tests of the reduction ops: sum, mean, Euclidean norm, product, min, max, logical all
# and any, count-nonzero, and the reduced-shape helper. Tests compare TensorFlow results with
# numpy references across dtypes and all axes combinations, and check gradients, shape inference,
# invalid arguments and empty inputs. Shared comparison logic lives in a base test class that
# several classes extend. No import-time side effects.
#
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
"""Functional tests for reduction ops."""

import itertools
import numbers

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

# The maximum input rank to test.
_MAX_RANK = 5


def _powerset(iterable):
  # [FUNCTION DOC]
  # HELPER: generates every possible reduction-axes choice for the comparison tests.
  # Input: any iterable of items. Returns a lazy iterator over all subsets (as tuples) of the
  # items, ordered by increasing size from the empty tuple up to the full set. Items are first
  # copied into a list, so one-shot iterables are fine. No side effects.
  """Helper for generating all possible reduction_axes arguments.

  Example:
  powerset([0,1,2]): () (0,) (1,) (2,) (0,1) (0,2) (1,2) (0,1,2)

  Args:
    iterable: An iterable of items to generate the powerset of.

  Returns:
    The powerset of all items in iterable.
  """
  s = list(iterable)
  return itertools.chain.from_iterable(
      itertools.combinations(s, r) for r in range(len(s) + 1))


class ReducedShapeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the helper that computes the shape of a tensor after a reduction (reduced
  # dimensions become size one). Covers simple, zero-sized and negative-axis cases.

  def _check(self, shape, axes, result):
    # [FUNCTION DOC]
    # HELPER: asserts the reduced-shape computation. Inputs: an input shape list, a list of axes
    # to reduce, and the expected resulting shape. Evaluates the shape helper and asserts the
    # output equals the expected shape; raises a test failure otherwise. Returns nothing.
    output = math_ops.reduced_shape(shape, axes=axes)
    self.assertAllEqual(output, result)

  @test_util.run_deprecated_v1
  def testSimple(self):
    # [FUNCTION DOC]
    # Checks reduced shapes for rank-one and rank-two shapes with no axes, a single axis and
    # all axes; reduced axes become size one and no axes leaves the shape unchanged.
    with self.cached_session():
      self._check([3], [], [3])
      self._check([3], [0], [1])
      self._check([5, 3], [], [5, 3])
      self._check([5, 3], [0], [1, 3])
      self._check([5, 3], [1], [5, 1])
      self._check([5, 3], [0, 1], [1, 1])

  @test_util.run_deprecated_v1
  def testZeros(self):
    # [FUNCTION DOC]
    # Checks the reduced-shape helper when some dimensions are zero: reduced axes still become
    # size one even if their length was zero, while unreduced zero dimensions are preserved.
    """Check that reduced_shape does the right thing with zero dimensions."""
    with self.cached_session():
      self._check([0], [], [0])
      self._check([0], [0], [1])
      self._check([0, 3], [], [0, 3])
      self._check([0, 3], [0], [1, 3])
      self._check([0, 3], [1], [0, 1])
      self._check([0, 3], [0, 1], [1, 1])
      self._check([3, 0], [], [3, 0])
      self._check([3, 0], [0], [1, 0])
      self._check([3, 0], [1], [3, 1])
      self._check([3, 0], [0, 1], [1, 1])

  @test_util.run_deprecated_v1
  def testNegAxes(self):
    # [FUNCTION DOC]
    # Checks reduced shapes of a 10x10x10 shape when axes are given as negative indices, repeated
    # or mixed with equivalent positive indices; duplicates collapse to a single reduced axis.
    with self.cached_session():
      self._check([10, 10, 10], [-1], [10, 10, 1])
      self._check([10, 10, 10], [-1, 2], [10, 10, 1])
      self._check([10, 10, 10], [-1, -1], [10, 10, 1])
      self._check([10, 10, 10], [-1, 0], [1, 10, 1])
      self._check([10, 10, 10], [-3], [1, 10, 10])


class ReductionUnknownShape(test.TestCase):
  # [CLASS DOC]
  # Test case checking that reductions over a tensor of completely unknown shape still yield a
  # statically known scalar shape.

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Graph-mode test. For float32 inputs it applies sum, mean, product, max, min and Euclidean
    # norm, and for boolean inputs all and any, each on a placeholder whose shape is unknown.
    # Asserts every result has the empty (scalar) shape.
    with self.cached_session():
      for dtype, reductions in [(dtypes.float32,
                                 (math_ops.reduce_sum, math_ops.reduce_mean,
                                  math_ops.reduce_prod, math_ops.reduce_max,
                                  math_ops.reduce_min,
                                  math_ops.reduce_euclidean_norm)),
                                (dtypes.bool, (math_ops.reduce_all,
                                               math_ops.reduce_any))]:
        for reduction in reductions:
          x = array_ops.placeholder(
              dtype=dtype, shape=None)  # Some tensor w/ unknown shape.
          y = reduction(x)
          self.assertEqual(y.shape, ())


class ReductionInvalidKeepdims(test.TestCase):
  # [CLASS DOC]
  # Test case for rejection of an invalid keepdims argument by the reduction functions
  # (regression test for a GitHub issue).

  def testBasic(self):
    # [FUNCTION DOC]
    # For each reduction (sum, mean, product, max, min, Euclidean norm on numbers; all and any on
    # booleans) passes a two-element float16 array as the keepdims argument and asserts a
    # value error mentioning ambiguity of a truth value is raised. The evaluate call after the
    # reduction is never reached when the error is raised. Returns nothing.
    # Test case for GitHub issue 46700.
    for dtype, reductions in [
        (dtypes.float32, (math_ops.reduce_sum, math_ops.reduce_mean,
                          math_ops.reduce_prod, math_ops.reduce_max,
                          math_ops.reduce_min, math_ops.reduce_euclidean_norm)),
        (dtypes.bool, (math_ops.reduce_all, math_ops.reduce_any))
    ]:
      for reduction in reductions:
        with self.assertRaisesRegex(ValueError, "The truth value"):
          x = True if dtype == dtypes.bool else 1
          y = reduction(
              input_tensor=x, keepdims=np.array([63600, 1], dtype=np.float16))
          self.evaluate(y)


class BaseReductionTest(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the sum, mean, product and Euclidean norm test cases. Subclasses
  # supply the TensorFlow reduction and the matching numpy reference; this class provides data
  # builders and comparison helpers that evaluate both and compare values and shapes, as well as
  # gradient-checking helpers. Holds no state of its own.

  def _tf_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: abstract hook returning the TensorFlow reduction of x over the given axes with the
    # keep-dimensions flag. This base version always raises a not-implemented error.
    raise NotImplementedError()

  def _np_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: abstract hook returning the numpy reference reduction of x. This base version
    # always raises a not-implemented error; subclasses override it.
    raise NotImplementedError()

  def _makeIncremental(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: builds deterministic test data. Inputs: a shape and a TensorFlow data type.
    # Returns a numpy array of that shape holding consecutive integers starting at zero, cast to
    # the type; for complex types each value v is replaced by v minus twice-imaginary v so the
    # imaginary parts are nonzero.
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype.as_numpy_dtype)
    if dtype.is_complex:
      data -= 2j * data
    return data

  def _makeRandom(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: builds random test data. Inputs: a shape and a TensorFlow data type. Returns a
    # numpy array of that shape with uniform random values in zero to one cast to the type; for
    # complex types the imaginary parts are made nonzero the same way as in the incremental
    # builder. Consumes numpy global random state.
    data = np.random.rand(*shape).astype(dtype.as_numpy_dtype)
    if dtype.is_complex:
      data -= 2j * data
    return data

  def _compare(self,
               x,
               reduction_axes,
               keepdims,
               feed_dict=None,
               rtol=1e-6,
               atol=1e-6):
    # [FUNCTION DOC]
    # HELPER: core comparison of the TensorFlow and numpy reductions.
    # KNOWN ISSUE: rtol and atol are accepted but never used; the closeness check uses the default
    # type-based tolerances.
    # Inputs: array x, reduction axes (or nothing for all axes),
    # keepdims flag, optional feed mapping for placeholders, and rtol and atol tolerances.
    # Computes the numpy reference, evaluates the TensorFlow reduction in a cached session, and
    # asserts the values are close (using type-dependent tolerances) and the shapes agree.
    np_ans = self._np_reduce(x, reduction_axes, keepdims)
    with self.cached_session() as sess:
      tf_ans = self._tf_reduce(x, reduction_axes, keepdims)
      out = sess.run(tf_ans, feed_dict)
    self.assertAllCloseAccordingToType(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self,
                  x,
                  reduction_axes,
                  feed_dict=None,
                  rtol=1e-6,
                  atol=1e-6):
    # [FUNCTION DOC]
    # HELPER: compares the reduction with keepdims false and then true.
    # KNOWN ISSUE: the scalar-axis recursion does not forward the feed mapping, so it is only
    # valid for tests that do not use placeholders.
    # Inputs: array x, axes
    # (nothing means all), optional feed mapping, rtol and atol. When the axes form a length-one
    # sequence it additionally tests the plain scalar axis by recursing.
    if reduction_axes is not None and np.shape(reduction_axes) == (1,):
      # Test scalar reduction_axes argument
      self._compareAll(x, reduction_axes[0], rtol=rtol, atol=atol)
    self._compare(
        x,
        reduction_axes,
        keepdims=False,
        feed_dict=feed_dict,
        rtol=rtol,
        atol=atol)
    self._compare(
        x,
        reduction_axes,
        keepdims=True,
        feed_dict=feed_dict,
        rtol=rtol,
        atol=atol)

  def _compareAllAxes(self, x, feed_dict=None, rtol=1e-6, atol=1e-6):
    # [FUNCTION DOC]
    # HELPER: compares reductions over every possible axes choice. Inputs: array x, optional feed
    # mapping, rtol and atol. Runs the comparison for a nothing axes value (reduce everything)
    # and then for every subset of the axes of x, including the empty subset. Returns nothing.
    self._compareAll(x, None, rtol=rtol, atol=atol)
    for axes in _powerset(range(x.ndim)):
      self._compareAll(x, axes, feed_dict, rtol=rtol, atol=atol)

  def _compareGradient(self, x, reduction_axes, rtol=1e-8, atol=1e-8):
    # [FUNCTION DOC]
    # HELPER: checks the gradient of the reduction. Inputs: array x, axes (or nothing), and rtol
    # and atol. Also recurses on the scalar form of a length-one axes sequence. Converts x to a
    # tensor, reduces without keepdims, then asserts the theoretical and numerical Jacobians from
    # the gradient checker are close (using a large step size of one).
    if reduction_axes is not None and np.shape(reduction_axes) == (1,):
      # Test scalar reduction_axes argument
      self._compareGradient(x, reduction_axes[0], rtol=rtol, atol=atol)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = self._tf_reduce(t, reduction_axes, False)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, x.shape, su, su.get_shape().as_list(), x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=rtol, atol=atol)

  def _compareGradientAxes(self, x, rtol=1e-8, atol=1e-8):
    # [FUNCTION DOC]
    # HELPER: runs the gradient check for a fixed set of axes: all, none, a single axis, pairs
    # and all four axes. Expects a rank-four input. Inputs: array x, rtol and atol.
    self._compareGradient(x, None, rtol=rtol, atol=atol)
    self._compareGradient(x, [], rtol=rtol, atol=atol)
    self._compareGradient(x, 0, rtol=rtol, atol=atol)
    self._compareGradient(x, [1], rtol=rtol, atol=atol)
    self._compareGradient(x, [2], rtol=rtol, atol=atol)
    self._compareGradient(x, [1, 2], rtol=rtol, atol=atol)
    self._compareGradient(x, [0, 1, 2, 3], rtol=rtol, atol=atol)


class SumReductionTest(BaseReductionTest):
  # [CLASS DOC]
  # Tests for the sum reduction across many dtypes, axes combinations, error cases, partial shape
  # inference, gradients and degenerate empty inputs. Uses numpy sum as the reference.

  def _tf_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: returns the TensorFlow sum of x over the axes with the keepdims flag.
    return math_ops.reduce_sum(x, reduction_axes, keepdims)

  def _np_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: numpy reference sum. Lists and arrays of axes are converted to tuples; returns the
    # numpy sum of x over those axes honoring keepdims.
    if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                      np.ndarray):
      reduction_axes = tuple(reduction_axes)
    return np.sum(x, axis=reduction_axes, keepdims=keepdims)

  def testAxesType(self):
    # [FUNCTION DOC]
    # Checks that the sum accepts the axes argument as a 32-bit or 64-bit integer constant: sums a
    # two-element list of zeros over axis zero and asserts the result is zero.
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.cached_session():
        v = math_ops.reduce_sum([0, 0], constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, 0)

  @test_util.run_deprecated_v1
  def testInfinity(self):
    # [FUNCTION DOC]
    # Graph-mode test: sums two-element arrays made from every combination of negative and
    # positive infinity in float32 and float64 and compares with numpy (including infinity and
    # not-a-number results).
    for dtype in [np.float32, np.float64]:
      for special_value_x in [-np.inf, np.inf]:
        for special_value_y in [-np.inf, np.inf]:
          np_arr = np.array([special_value_x, special_value_y]).astype(dtype)
          self._compareAll(np_arr, None)

  @test_util.run_deprecated_v1
  def testInt32(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares sums of 32-bit integer incremental arrays of rank one to five,
    # each of shape two in every dimension, over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.int32)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testFloat16(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares sums of half-precision incremental arrays of ranks one to five
    # over all axes combinations. Then, only if a GPU is available (otherwise the test silently
    # returns early), averages 68000 ones stored in a half-precision variable on the GPU and
    # asserts the mean is one, i.e. the accumulation does not overflow.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float16)
      self._compareAllAxes(np_arr)

    # test that mean doesn't overflow
    # only on GPU, since it has the more accurate implementation
    if not test.is_gpu_available():
      return

    arr = np.ones([68000], dtype=np.float16)

    with self.session(graph=ops.Graph(), use_gpu=True) as sess:
      tf_arr = variables.Variable(arr)
      self.evaluate(variables.global_variables_initializer())
      tf_mean = math_ops.reduce_mean(tf_arr, 0, False)
      tf_out_mean = self.evaluate(tf_mean)
    self.assertAllClose(tf_out_mean, 1.)

  @test_util.run_deprecated_v1
  def testBfloat16(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares sums of bfloat16 incremental arrays of ranks one to five over all
    # axes combinations using a loose absolute tolerance.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.bfloat16)
      self._compareAllAxes(np_arr, rtol=1e-3, atol=5.)

  @test_util.run_deprecated_v1
  def testFloat32(self):
    # [FUNCTION DOC]
    # Graph-mode test. First compares float32 incremental arrays of ranks one to five over all
    # axes. Then ten times draws random large 2-D sizes (capped at about ten million elements),
    # sums an all-ones array over rows and columns on GPU-enabled sessions and compares with
    # numpy. Finally checks many small 3-D all-ones arrays reduced over the outer axes and the
    # middle axis.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
      self._compareAllAxes(np_arr)

    for _ in range(10):
      size_x = int(2**np.random.uniform(0, 15))
      size_y = int(2**np.random.uniform(0, 15))

      if size_x * size_y > 1e7:
        size_y = int(1e7 / size_x)

      arr = np.ones([size_x, size_y], dtype=np.float32)
      col_sum = np.sum(arr, axis=0)
      row_sum = np.sum(arr, axis=1)

      with self.session(graph=ops.Graph(), use_gpu=True) as sess:
        tf_row_sum = self._tf_reduce(arr, 1, False)
        tf_col_sum = self._tf_reduce(arr, 0, False)
        tf_out_row, tf_out_col = self.evaluate([tf_row_sum, tf_col_sum])
      self.assertAllClose(col_sum, tf_out_col)
      self.assertAllClose(row_sum, tf_out_row)

    for size_x in [1, 3, 16, 33]:
      for size_y in [1, 3, 16, 33]:
        for size_z in [1, 3, 16, 33]:
          arr = np.ones([size_x, size_y, size_z], dtype=np.float32)
          sum_y = np.sum(arr, axis=1)
          sum_xz = np.sum(arr, axis=(0, 2))

          with self.session(graph=ops.Graph(), use_gpu=True) as sess:
            tf_sum_xz = self._tf_reduce(arr, [0, 2], False)
            tf_sum_y = self._tf_reduce(arr, 1, False)
            tf_out_sum_xz, tf_out_sum_y = self.evaluate([tf_sum_xz, tf_sum_y])
          self.assertAllClose(sum_y, tf_out_sum_y)
          self.assertAllClose(sum_xz, tf_out_sum_xz)

  @test_util.run_deprecated_v1
  def testFloat32BFloat16(self):
    # [FUNCTION DOC]
    # Graph-mode test run for float32 and bfloat16. Compares incremental arrays of ranks one to
    # five over all axes, then random small 2-D all-ones arrays reduced over rows and columns,
    # then small 3-D all-ones arrays reduced over several axes, using type-aware tolerances.
    for dtype in [dtypes.float32, dtypes.bfloat16]:
      dtype_np = np.float32 if dtype == dtypes.float32 else dtype.as_numpy_dtype
      for rank in range(1, _MAX_RANK + 1):
        np_arr = self._makeIncremental((2,) * rank, dtype)
        self._compareAllAxes(np_arr)

      for _ in range(10):
        size_x = int(2 ** np.random.uniform(0, 7))
        size_y = int(2 ** np.random.uniform(0, 7))

        if size_x * size_y > 1e7:
          size_y = int(1e7 / size_x)

        arr = np.ones([size_x, size_y], dtype=dtype_np)
        col_sum = np.sum(arr, axis=0)
        row_sum = np.sum(arr, axis=1)

        tf_row_sum = self._tf_reduce(arr, 1, False)
        tf_col_sum = self._tf_reduce(arr, 0, False)
        tf_out_row, tf_out_col = self.evaluate([tf_row_sum, tf_col_sum])
        self.assertAllCloseAccordingToType(col_sum, tf_out_col)
        self.assertAllCloseAccordingToType(row_sum, tf_out_row)

      for size_x in [1, 3, 16]:
        for size_y in [1, 3, 16]:
          for size_z in [1, 3, 16]:
            arr = np.ones([size_x, size_y, size_z], dtype=dtype_np)
            sum_y = np.sum(arr, axis=1)
            sum_xz = np.sum(arr, axis=(0, 2))

            tf_sum_xz = self._tf_reduce(arr, [0, 2], False)
            tf_sum_y = self._tf_reduce(arr, 1, False)
            tf_out_sum_xz, tf_out_sum_y = self.evaluate([tf_sum_xz, tf_sum_y])
            self.assertAllCloseAccordingToType(sum_y, tf_out_sum_y)
            self.assertAllCloseAccordingToType(sum_xz, tf_out_sum_xz)

  @test_util.run_deprecated_v1
  def testFloat64(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares sums of double-precision incremental arrays of ranks one to five
    # over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float64)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testComplex64(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares sums of single-precision complex incremental arrays of ranks one
    # to five over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex64)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testComplex128(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares sums of double-precision complex incremental arrays of ranks one
    # to five over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex128)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testInvalidIndex(self):
    # [FUNCTION DOC]
    # Graph-mode test: summing a 2x5 tensor with out-of-range axes (minus three, two, and zero
    # with two) must raise a value error whose message mentions an invalid reduction dimension.
    np_arr = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    input_tensor = ops.convert_to_tensor(np_arr)
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "Invalid reduction dimension" in str(e)):
      math_ops.reduce_sum(input_tensor, [-3])
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "Invalid reduction dimension" in str(e)):
      math_ops.reduce_sum(input_tensor, [2])
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "Invalid reduction dimension" in str(e)):
      math_ops.reduce_sum(input_tensor, [0, 2])

  @test_util.run_deprecated_v1
  def testPartialShapes(self):
    # [FUNCTION DOC]
    # Graph-mode test of static shape inference for the sum. Seeds numpy randomness. Checks that
    # an unknown-shape input gives an unknown-shape result, a known-rank input with keepdims gives
    # a result of that rank, and unknown reduction indices give an unknown shape (or the input
    # rank when keepdims is true). Also compares numeric results through fed placeholders.
    np.random.seed(1618)

    # Input shape is unknown.
    reduction_axes = [1, 2]
    c_unknown = array_ops.placeholder(dtypes.float32)
    s_unknown = math_ops.reduce_sum(c_unknown, reduction_axes)
    self.assertEqual(tensor_shape.unknown_shape(), s_unknown.get_shape())

    np_input = np.random.randn(3, 3, 3)
    self._compareAll(np_input, reduction_axes, {c_unknown: np_input})

    # Input shape only has known rank.
    c_known_rank = array_ops.placeholder(dtypes.float32)
    c_known_rank.set_shape(tensor_shape.unknown_shape(rank=3))
    s_known_rank = math_ops.reduce_sum(
        c_known_rank, reduction_axes, keepdims=True)
    self.assertEqual(3, s_known_rank.get_shape().rank)

    np_input = np.random.randn(3, 3, 3)
    self._compareAll(np_input, reduction_axes, {c_known_rank: np_input})

    # Reduction indices are unknown.
    unknown_indices = array_ops.placeholder(dtypes.int32)
    c_unknown_indices = constant_op.constant([[10.0], [20.0]])
    s_unknown_indices = math_ops.reduce_sum(
        c_unknown_indices, unknown_indices, keepdims=False)
    self.assertEqual(tensor_shape.unknown_shape(),
                     s_unknown_indices.get_shape())
    s_unknown_indices_keep = math_ops.reduce_sum(
        c_unknown_indices, unknown_indices, keepdims=True)
    self.assertEqual(2, s_unknown_indices_keep.get_shape().rank)

  @test_util.run_deprecated_v1
  def testWrongShapeForReductionIndices(self):
    # [FUNCTION DOC]
    # Graph-mode test: reduction axes given as a nested two-dimensional list must raise a value
    # error saying the indices must be at most rank one.
    reduction_axes = [[1], [2]]
    c_unknown = array_ops.placeholder(dtypes.float32)
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             ".*must be at most rank 1.*"):
      math_ops.reduce_sum(c_unknown, reduction_axes)

  def testInvalidRepeatedReductionIndices(self):
    # [FUNCTION DOC]
    # Sums a two-element vector with the axes tensor containing zero twice and asserts an
    # invalid-argument error saying the axes contain a duplicate dimension.
    reduction_axes = constant_op.constant([0, 0])
    c = constant_op.constant([1.0, 2.0])
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        ".*Axes contains duplicate dimension: 0.*"):
      self.evaluate(math_ops.reduce_sum(c, reduction_axes))

  # Int64??

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Graph-mode test: checks the sum gradient for float32, float64, complex64 and complex128
    # inputs of shape two by three by four by two over the standard axes choices.
    for dtype in [
        dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128
    ]:
      x = self._makeIncremental([2, 3, 4, 2], dtype)
      self._compareGradientAxes(x)

  @test_util.run_deprecated_v1
  def testHighRank(self):
    # [FUNCTION DOC]
    # Graph-mode test with a fixed seed: twenty random reductions on integer arrays of rank four
    # to ten with random subsets of axes, then specific axes patterns (none, all, even, odd) for
    # ranks four, seven and ten.
    # Do a bunch of random high dimensional reductions
    np.random.seed(42)
    for _ in range(20):
      rank = np.random.randint(4, 10 + 1)
      axes, = np.nonzero(np.random.randint(2, size=rank))
      shape = tuple(np.random.randint(1, 3 + 1, size=rank))
      data = np.random.randint(1024, size=shape)
      self._compareAll(data, axes)
    # Check some particular axis patterns
    for rank in 4, 7, 10:
      shape = tuple(np.random.randint(1, 3 + 1, size=rank))
      data = np.random.randint(1024, size=shape)
      for axes in ([], np.arange(rank), np.arange(0, rank, 2),
                   np.arange(1, rank, 2)):
        self._compareAll(data, axes)

  @test_util.run_deprecated_v1
  def testExpand(self):
    # [FUNCTION DOC]
    # Graph-mode test: sums a 5x0 array over axis one, reducing an empty tensor to a nonempty
    # (zero-filled) one, and compares with numpy.
    # Reduce an empty tensor to a nonempty tensor
    x = np.zeros((5, 0))
    self._compareAll(x, [1])

  @test_util.run_deprecated_v1
  def testEmptyGradients(self):
    # [FUNCTION DOC]
    # Graph-mode test: the gradient error for the sum of a zero-by-three tensor over axis one is
    # asserted to be zero.
    with self.session():
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_sum(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)

  @test_util.run_deprecated_v1
  def testDegenerate(self):
    # [FUNCTION DOC]
    # Graph-mode test: for several float and complex types, sums a zero-by-9938 tensor over axis
    # zero (a size chosen to trigger an Eigen crash historically) and asserts the result is a
    # vector of zeros.
    with self.session():
      for dtype in (dtypes.bfloat16, dtypes.float16, dtypes.float32,
                    dtypes.float64, dtypes.complex64, dtypes.complex128):
        # A large number is needed to get Eigen to die
        x = array_ops.zeros((0, 9938), dtype=dtype)
        y = math_ops.reduce_sum(x, [0])
        self.assertAllEqual(y, np.zeros(9938))


class MeanReductionTest(BaseReductionTest):
  # [CLASS DOC]
  # Tests for the mean reduction: dtypes including small integer types and size-overflow
  # regression cases, infinities, gradients and degenerate empty inputs. The numpy reference
  # emulates TensorFlow's truncating integer mean.

  def _tf_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: returns the TensorFlow mean of x over the axes with the keepdims flag.
    return math_ops.reduce_mean(x, reduction_axes, keepdims)

  def _np_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: numpy reference mean. Converts list or array axes to a tuple and a single integer
    # to a one-element tuple. Computes the element count over the reduced axes (all elements if
    # axes are nothing), sums, and divides by the count; for integer inputs it uses truncating
    # division to mimic TensorFlow, since numpy would otherwise produce floats.
    if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                      np.ndarray):
      reduction_axes = tuple(reduction_axes)
    elif isinstance(reduction_axes, numbers.Integral):
      reduction_axes = (reduction_axes,)

    if reduction_axes is None:
      count = np.prod(x.shape)
    else:
      count = np.prod([x.shape[ax] for ax in reduction_axes])
    # np.mean automatically converts integer inputs to float, while TensorFlow's
    # reduce_mean does not. For integer inputs, we emulate TensorFlow's behavior
    # using np.sum and truncating division.
    np_sum = np.sum(x, axis=reduction_axes, keepdims=keepdims)
    if np.issubdtype(x.dtype, np.integer):
      return np_sum // count
    return np_sum / count

  def testAxesType(self):
    # [FUNCTION DOC]
    # Checks the mean accepts 32-bit and 64-bit integer constant axes: the mean of two zeros over
    # axis zero equals zero.
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.cached_session():
        v = math_ops.reduce_mean([0, 0], constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, 0)

  @test_util.run_deprecated_v1
  def testInfinity(self):
    # [FUNCTION DOC]
    # Graph-mode test: means of two-element arrays of every combination of plus and minus
    # infinity in float32 and float64 compared against the reference.
    for dtype in [np.float32, np.float64]:
      for special_value_x in [-np.inf, np.inf]:
        for special_value_y in [-np.inf, np.inf]:
          np_arr = np.array([special_value_x, special_value_y]).astype(dtype)
          self._compareAll(np_arr, None)

  @test_util.run_deprecated_v1
  def testInt32(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares means of 32-bit integer incremental arrays of ranks one to five
    # over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.int32)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testUint8(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares means of random unsigned 8-bit arrays of ranks one to five over
    # all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeRandom((2,) * rank, dtypes.uint8)
      self._compareAllAxes(np_arr)

  # This tests the issue reported in b/145030710.
  @test_util.run_deprecated_v1
  def testSizeOverflowUint8(self):
    # [FUNCTION DOC]
    # Graph-mode regression test for a reported bug (size overflow): mean of a 256-element
    # random unsigned 8-bit vector over all axes compared with the reference.
    np_arr = self._makeRandom((2**8,), dtypes.uint8)
    self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testSizeOverflowInt8(self):
    # [FUNCTION DOC]
    # Graph-mode test: mean of a 128-element random signed 8-bit vector (count overflowing the
    # type) over all axes compared with the reference.
    np_arr = self._makeRandom((2**7,), dtypes.int8)
    self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testSizeOverflowUint16(self):
    # [FUNCTION DOC]
    # Graph-mode test: mean of a 65536-element random unsigned 16-bit vector over all axes
    # compared with the reference, checking count overflow handling.
    np_arr = self._makeRandom((2**16,), dtypes.uint16)
    self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testSizeOverflowInt16(self):
    # [FUNCTION DOC]
    # Graph-mode test: mean of a 32768-element random signed 16-bit vector over all axes compared
    # with the reference, checking count overflow handling.
    np_arr = self._makeRandom((2**15,), dtypes.int16)
    self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testFloat32(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares means of float32 incremental arrays of ranks one to five over all
    # axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testBFloat16(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares means of bfloat16 incremental arrays of ranks one to five over
    # all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.bfloat16)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testFloat64(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares means of double-precision incremental arrays of ranks one to
    # five over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float64)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testComplex64(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares means of single-precision complex incremental arrays of ranks one
    # to five over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex64)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testComplex128(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares means of double-precision complex incremental arrays of ranks one
    # to five over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex128)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Graph-mode test: checks the mean gradient for float32 and float64 inputs of shape two by
    # three by four by two over the standard axes choices with relaxed tolerances.
    s = [2, 3, 4, 2]
    for dtype in [dtypes.float32, dtypes.float64]:
      x = self._makeIncremental(s, dtype)
      self._compareGradientAxes(x, rtol=1e-3, atol=1e-3)

  @test_util.run_deprecated_v1
  def testEmptyGradients(self):
    # [FUNCTION DOC]
    # Graph-mode test: the gradient error for the mean of a zero-by-three tensor over axis one is
    # asserted to be zero.
    with self.session():
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_mean(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)

  @test_util.run_deprecated_v1
  def testDegenerate(self):
    # [FUNCTION DOC]
    # Graph-mode test: the mean of a zero-by-9938 tensor over axis zero (float types) must be a
    # vector of length 9938 where every entry is not-a-number.
    with self.session():
      for dtype in (dtypes.bfloat16, dtypes.float16, dtypes.float32,
                    dtypes.float64):
        # A large number is needed to get Eigen to die
        x = array_ops.zeros((0, 9938), dtype=dtype)
        y = math_ops.reduce_mean(x, [0]).eval()
        self.assertEqual(y.shape, (9938,))
        self.assertTrue(np.all(np.isnan(y)))


class EuclideanNormReductionTest(BaseReductionTest, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the Euclidean norm reduction (square root of sum of squared magnitudes), with
  # parameterized dtype coverage, infinities, singletons, gradients and empty inputs. Also uses
  # the parameterized test case base for the type-parameterized test.

  def _tf_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: returns the TensorFlow Euclidean norm of x over the axes with the keepdims flag.
    return math_ops.reduce_euclidean_norm(x, reduction_axes, keepdims)

  def _np_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: numpy reference norm. Converts list or array axes to a tuple, then takes the
    # square root of the sum of x times its complex conjugate over those axes; for integer inputs
    # the result is floored to mimic TensorFlow integer arithmetic.
    if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                      np.ndarray):
      reduction_axes = tuple(reduction_axes)
    np_fro = np.sqrt(
        np.sum(x * np.conj(x), axis=reduction_axes, keepdims=keepdims))
    if np.issubdtype(x.dtype, np.integer):
      np_fro = np.floor(np_fro)
    return np_fro

  @test_util.run_deprecated_v1
  def testAxesType(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite living in the Euclidean norm tests, it calls the mean reduction, not
    # the Euclidean norm, so the norm's axes-type handling is not actually tested.
    # Graph-mode test with 32-bit and 64-bit integer constant axes.
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.cached_session():
        v = math_ops.reduce_mean([0, 0], constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, 0)

  @test_util.run_deprecated_v1
  def testInfinity(self):
    # [FUNCTION DOC]
    # Graph-mode test: norms of two-element arrays of every combination of plus and minus
    # infinity in float32 and float64 compared against the reference.
    for dtype in [np.float32, np.float64]:
      for special_value_x in [-np.inf, np.inf]:
        for special_value_y in [-np.inf, np.inf]:
          np_arr = np.array([special_value_x, special_value_y]).astype(dtype)
          self._compareAll(np_arr, None)

  @test_util.run_deprecated_v1
  def testSingleton(self):
    # [FUNCTION DOC]
    # Graph-mode test: norm of the one-element array containing minus one in float32 and float64,
    # over all axes, compared against the reference.
    for dtype in [np.float32, np.float64]:
      np_arr = np.array([-1.]).astype(dtype)
      self._compareAll(np_arr, None)

  @parameterized.parameters(
      dtypes.bfloat16,
      dtypes.float32,
      dtypes.float64,
      dtypes.int32,
      dtypes.complex64,
      dtypes.complex128,
  )
  @test_util.run_deprecated_v1
  def testTypes(self, dtype):
    # [FUNCTION DOC]
    # Parameterized graph-mode test, run for bfloat16, float32, float64, int32, complex64 and
    # complex128 (the dtype parameter). Compares norms of incremental arrays of ranks one to five
    # over all axes, with loose tolerances for bfloat16 and tight ones otherwise.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtype)
      rtol, atol = (1e-2, 5e-1) if dtype == dtypes.bfloat16 else (1e-6, 1e-6)
      self._compareAllAxes(np_arr, rtol=rtol, atol=atol)

  @test_util.run_deprecated_v1
  def testDegenerate(self):
    # [FUNCTION DOC]
    # Graph-mode test: the norm of a zero-by-9938 tensor over axis zero (float types) must have
    # shape 9938 and be all zeros.
    with self.session():
      for dtype in (dtypes.bfloat16, dtypes.float16, dtypes.float32,
                    dtypes.float64):
        # A large number is needed to get Eigen to die
        x = array_ops.zeros((0, 9938), dtype=dtype)
        y = math_ops.reduce_euclidean_norm(x, [0]).eval()
        self.assertEqual(y.shape, (9938,))
        self.assertAllEqual(y, np.zeros(9938))

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Graph-mode test: checks the norm gradient for float32 and float64 inputs of shape two by
    # three by four by two with one added to every entry (to avoid a not-a-number gradient at
    # zero), over the standard axes choices with a loose tolerance.
    shape = [2, 3, 4, 2]
    for dtype in [dtypes.float32, dtypes.float64]:
      # zero value entry will result NaN gradient if reduction doesn't happen.
      # e.g., `tf.math.reduce_sum([0, 1], axis=[])` so add one to avoid it.
      x = self._makeIncremental(shape, dtype) + 1.0
      self._compareGradientAxes(x, rtol=1e-2, atol=1e-2)


class ProdReductionTest(BaseReductionTest):
  # [CLASS DOC]
  # Tests for the product reduction across dtypes (with value limiting to avoid integer
  # overflow), axes combinations, gradients with zeros in various positions, and degenerate
  # empty inputs. Uses numpy product as the reference.

  def _tf_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: returns the TensorFlow product of x over the axes with the keepdims flag.
    return math_ops.reduce_prod(x, reduction_axes, keepdims)

  def _np_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # HELPER: numpy reference product. Converts list or array axes to a tuple and returns the
    # numpy product of x over those axes honoring keepdims.
    if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                      np.ndarray):
      reduction_axes = tuple(reduction_axes)
    return np.prod(x, axis=reduction_axes, keepdims=keepdims)

  def testAxesType(self):
    # [FUNCTION DOC]
    # Checks the product accepts 32-bit and 64-bit integer constant axes: the product of two
    # zeros over axis zero equals zero.
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.cached_session():
        v = math_ops.reduce_prod([0, 0], constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, 0)

  @test_util.run_deprecated_v1
  def testInfinity(self):
    # [FUNCTION DOC]
    # Graph-mode test: products of two-element arrays of every combination of plus and minus
    # infinity in float32 and float64 compared against the reference.
    for dtype in [np.float32, np.float64]:
      for special_value_x in [-np.inf, np.inf]:
        for special_value_y in [-np.inf, np.inf]:
          np_arr = np.array([special_value_x, special_value_y]).astype(dtype)
          self._compareAll(np_arr, None)

  @test_util.run_deprecated_v1
  def testInt32(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares products of 32-bit integer arrays for ranks one to four (the
    # maximum rank is excluded) with values reduced modulo five plus one to avoid overflow, since
    # numpy would silently widen the type.
    # Numpy automatically upgrades the type of np.prod from int32 to int64, so
    # Numpy does not overflow an int32 np.prod while TensorFlow does. To avoid
    # overflow, limit array values.
    for rank in range(1, _MAX_RANK):
      np_arr = self._makeIncremental((2,) * rank, dtypes.int32) % 5 + 1
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testInt64(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares products of 64-bit integer arrays for ranks one to four with
    # values reduced modulo eleven plus one to avoid overflow.
    for rank in range(1, _MAX_RANK):
      # Avoid overflow by limiting array values.
      np_arr = self._makeIncremental((2,) * rank, dtypes.int64) % 11 + 1
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testFloat32(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares products of float32 incremental arrays of ranks one to five over
    # all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testBfloat16(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares products of bfloat16 incremental arrays scaled by 0.01 (to keep
    # magnitudes small) for ranks one to five over all axes, with loose tolerances.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.bfloat16) * \
               np.array([0.01]).astype(dtypes.bfloat16.as_numpy_dtype)
      self._compareAllAxes(np_arr, rtol=1e-2, atol=1e-2)

  @test_util.run_deprecated_v1
  def testFloat64(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares products of double-precision incremental arrays of ranks one to
    # five over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float64)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testComplex64(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares products of single-precision complex incremental arrays of ranks
    # one to five over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex64)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testComplex128(self):
    # [FUNCTION DOC]
    # Graph-mode test: compares products of double-precision complex incremental arrays of ranks
    # one to five over all axes combinations.
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.complex128)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testGradientWithZeros(self):
    # [FUNCTION DOC]
    # Graph-mode test: product gradient checks on a scaled float32 array of shape two by three by
    # four by two, first with no zeros, then with a zero slice at the beginning, end and middle
    # of the third axis, and finally with all zeros, each over the standard axes choices.
    s = [2, 3, 4, 2]
    x = self._makeIncremental(s, dtypes.float32) / 20.
    # No zeros in input
    self._compareGradientAxes(x, rtol=1e-3, atol=1e-3)
    # Zero at beginning
    x1 = x.copy()
    x1[:, :, 0, :] = 0
    self._compareGradientAxes(x1, rtol=1e-3, atol=1e-3)
    # Zero at end
    x2 = x.copy()
    x2[:, :, -1, :] = 0
    self._compareGradientAxes(x2, rtol=1e-3, atol=1e-3)
    # Zero in middle
    x3 = x.copy()
    x3[:, :, 2, :] = 0
    self._compareGradientAxes(x3, rtol=1e-3, atol=1e-3)
    # All zeros
    x4 = x.copy()
    x4[:, :, :, :] = 0
    self._compareGradientAxes(x4, rtol=1e-3, atol=1e-3)

  @test_util.run_deprecated_v1
  def testEmptyGradients(self):
    # [FUNCTION DOC]
    # Graph-mode test: the gradient error for the product of a zero-by-three tensor over axis one
    # is asserted to be zero.
    with self.session():
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_prod(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)

  @test_util.run_deprecated_v1
  def testDegenerate(self):
    # [FUNCTION DOC]
    # Graph-mode test: the product of a zero-by-9938 tensor over axis zero (float types) must be
    # a vector of ones, the multiplicative identity.
    with self.session():
      for dtype in (dtypes.bfloat16, dtypes.float16, dtypes.float32,
                    dtypes.float64):
        # A large number is needed to get Eigen to die
        x = array_ops.zeros((0, 9938), dtype=dtype)
        y = math_ops.reduce_prod(x, [0])
        self.assertAllEqual(y, np.ones(9938))


class MinReductionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the minimum reduction, comparing against numpy minimum applied axis by axis, and
  # checking special values, 3-D reductions, gradients and empty gradients. Self-contained; it
  # does not use the shared base class.

  def _compare(self, x, reduction_axes, keepdims, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the TensorFlow minimum with a numpy reference. Inputs: array x, axes (or
    # nothing for all), keepdims flag, and whether to allow the GPU. The reference applies the
    # numpy minimum one axis at a time starting from the last. The TensorFlow result is evaluated
    # in a cached session; asserts values are close and shapes equal.
    np_ans = x
    if reduction_axes is None:
      np_ans = np.amin(np_ans, keepdims=keepdims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.amin(np_ans, axis=ra, keepdims=keepdims)
    with self.cached_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_min(x, reduction_axes, keepdims)
      out = self.evaluate(tf_ans)
    self.assertAllClose(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    # [FUNCTION DOC]
    # HELPER: runs the comparison for keepdims false and true, both with the GPU allowed.
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=True)

  def testAxesType(self):
    # [FUNCTION DOC]
    # Checks the minimum accepts 32-bit and 64-bit integer constant axes: the minimum of two
    # zeros over axis zero equals zero.
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.cached_session():
        v = math_ops.reduce_min([0, 0], constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, 0)

  @test_util.disable_xla("b/168718272")  # XLA handling of NaN is inconsistent
  def testSpecialValues(self):
    # [FUNCTION DOC]
    # Compares minimum over all axes for every vector of length one to three drawn from negative
    # infinity, one, not-a-number and positive infinity, in float32 and float64. Disabled under
    # XLA because its not-a-number handling is inconsistent.
    for dtype in [np.float32, np.float64]:
      for size in range(1, 4):
        for arr in itertools.product([-np.inf, 1., np.nan, np.inf],
                                     repeat=size):
          self._compareAll(np.array(arr, dtype=dtype), None)

  def testFloatReduce3D(self):
    # [FUNCTION DOC]
    # Compares minimum over every possible axes choice (none, empty, each single axis, each pair
    # and all three) for a 2x3x5 float32 array of increasing numbers.
    # Create a 3D array of floats and reduce across all possible
    # dimensions
    np_arr = np.arange(1, 31).reshape([2, 3, 5]).astype(np.float32)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testDoubleReduce3D(self):
    # [FUNCTION DOC]
    # Compares minimum over every possible axes choice (none, empty, each single axis, each pair
    # and all three) for a 2x3x5 double-precision array of increasing numbers.
    # Create a 3D array of doubles and reduce across all possible
    # dimensions
    np_arr = np.arange(1, 31).reshape([2, 3, 5]).astype(np.float64)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Graph-mode test: gradient check of the minimum over axes one and two of a 2x3x4x2 array of
    # increasing doubles; theoretical and numerical Jacobians must be close.
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_min(t, [1, 2])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient2(self):
    # [FUNCTION DOC]
    # Graph-mode test: gradient check of the minimum over axis one of a 2x3x4x2 array of
    # increasing doubles.
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_min(t, [1])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 4, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient3(self):
    # [FUNCTION DOC]
    # Graph-mode test: gradient check of the minimum over axis two of a 2x3x4x2 array of
    # increasing doubles.
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_min(t, [2])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 3, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient4(self):
    # [FUNCTION DOC]
    # Graph-mode test: gradient check of the minimum over all axes (scalar result) of a 2x3x4x2
    # array of increasing doubles.
    s = [2, 3, 4, 2]
    x = np.arange(1.0, 49.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_min(t)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [1], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testEmptyGradients(self):
    # [FUNCTION DOC]
    # Graph-mode test: the gradient error for the minimum of a zero-by-three tensor over axis one
    # is asserted to be zero.
    with self.cached_session():
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_min(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)


class MaxReductionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the maximum reduction, comparing against numpy maximum applied axis by axis, and
  # checking special values, 3-D reductions over several dtypes, gradients and empty gradients.
  # Self-contained; it does not use the shared base class.

  def _compare(self, x, reduction_axes, keepdims, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the TensorFlow maximum with a numpy reference. Inputs: array x, axes (or
    # nothing for all), keepdims flag, and whether to allow the GPU. The reference applies the
    # numpy maximum one axis at a time starting from the last. The TensorFlow result is evaluated
    # in a cached session; asserts values are close and shapes equal.
    np_ans = x
    if reduction_axes is None:
      np_ans = np.amax(np_ans, keepdims=keepdims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.amax(np_ans, axis=ra, keepdims=keepdims)
    with self.cached_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_max(x, reduction_axes, keepdims)
      out = self.evaluate(tf_ans)
    self.assertAllClose(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    # [FUNCTION DOC]
    # HELPER: runs the comparison for keepdims false and true, both with the GPU allowed.
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=True)

  def testAxesType(self):
    # [FUNCTION DOC]
    # Checks the maximum accepts 32-bit and 64-bit integer constant axes: the maximum of two
    # zeros over axis zero equals zero.
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.cached_session():
        v = math_ops.reduce_max([0, 0], constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, 0)

  @test_util.disable_xla("b/168718272")  # XLA handling of NaN is inconsistent
  def testSpecialValues(self):
    # [FUNCTION DOC]
    # Compares maximum over all axes for every vector of length one to three drawn from negative
    # infinity, one, not-a-number and positive infinity, in float32 and float64. Disabled under
    # XLA because its not-a-number handling is inconsistent.
    for dtype in [np.float32, np.float64]:
      for size in range(1, 4):
        for arr in itertools.product([-np.inf, 1., np.nan, np.inf],
                                     repeat=size):
          self._compareAll(np.array(arr, dtype=dtype), None)

  def testInt64Reduce3D(self):
    # [FUNCTION DOC]
    # Compares maximum over every possible axes choice for a 2x3x5 array of negative 64-bit
    # integers.
    # Create a 3D array of int64s and reduce across all possible
    # dimensions
    np_arr = np.arange(-31, -1).reshape([2, 3, 5]).astype(np.int64)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testFloatReduce3D(self):
    # [FUNCTION DOC]
    # Compares maximum over every possible axes choice for a 2x3x5 array of negative float32
    # numbers.
    # Create a 3D array of floats and reduce across all possible
    # dimensions
    np_arr = np.arange(-31, -1).reshape([2, 3, 5]).astype(np.float32)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testDoubleReduce3D(self):
    # [FUNCTION DOC]
    # Compares maximum over every possible axes choice for a 2x3x5 array of negative
    # double-precision numbers.
    # Create a 3D array of doubles and reduce across all possible
    # dimensions
    np_arr = np.arange(-31, -1).reshape([2, 3, 5]).astype(np.float64)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testBfloat16Reduce3D(self):
    # [FUNCTION DOC]
    # Compares maximum over every possible axes choice for a 2x3x5 array of negative bfloat16
    # numbers.
    # Create a 3D array of floats and reduce across all possible
    # dimensions
    np_arr = np.arange(-31,
                       -1).reshape([2, 3,
                                    5]).astype(dtypes.bfloat16.as_numpy_dtype)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Graph-mode test: gradient check of the maximum over axes one and two of a 2x3x4x2 array of
    # negative doubles; theoretical and numerical Jacobians must be close.
    s = [2, 3, 4, 2]
    x = np.arange(-49.0, -1.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t, [1, 2])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient2(self):
    # [FUNCTION DOC]
    # Graph-mode test: gradient check of the maximum over axis one of a 2x3x4x2 array of negative
    # doubles.
    s = [2, 3, 4, 2]
    x = np.arange(-49.0, -1.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t, [1])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 4, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient3(self):
    # [FUNCTION DOC]
    # Graph-mode test: gradient check of the maximum over axis two of a 2x3x4x2 array of negative
    # doubles.
    s = [2, 3, 4, 2]
    x = np.arange(-49.0, -1.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t, [2])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 3, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient4(self):
    # [FUNCTION DOC]
    # Graph-mode test: gradient check of the maximum over all axes (scalar result) of a 2x3x4x2
    # array of negative doubles.
    s = [2, 3, 4, 2]
    x = np.arange(-49.0, -1.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [1], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testEmptyGradients(self):
    # [FUNCTION DOC]
    # Graph-mode test: the gradient error for the maximum of a zero-by-three tensor over axis one
    # is asserted to be zero.
    with self.cached_session():
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_max(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)


class AllReductionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the logical-and reduction on boolean arrays, compared with numpy all applied axis
  # by axis, on both GPU-enabled and CPU-only sessions.

  def _compare(self, x, reduction_axes, keepdims, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the TensorFlow logical-and reduction with a numpy reference applied one
    # axis at a time from the last. Inputs: array x, axes (or nothing), keepdims flag, and
    # whether to allow the GPU. Asserts exact equality of values and equal shapes.
    np_ans = x
    if reduction_axes is None:
      np_ans = np.all(np_ans, keepdims=keepdims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.all(np_ans, axis=ra, keepdims=keepdims)
    with self.cached_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_all(x, reduction_axes, keepdims)
      out = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    # [FUNCTION DOC]
    # HELPER: runs the comparison for keepdims false and true, each with and without the GPU.
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, False, use_gpu=False)
    self._compare(x, reduction_axes, True, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=False)

  def testAxesType(self):
    # [FUNCTION DOC]
    # Checks the logical-and reduction accepts 32-bit and 64-bit integer constant axes: reducing
    # two true values over axis zero gives true.
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.session():
        v = math_ops.reduce_all([True, True],
                                constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, True)

  def testAll3D(self):
    # [FUNCTION DOC]
    # Compares logical-and over every possible axes choice for a random 2x3x5 boolean array in
    # which each element is true with probability about ninety percent (uses random numbers).
    # Create a 3D array of bools and reduce across all possible
    # dimensions
    np_arr = (np.random.uniform(0, 1, 30) > 0.1).reshape([2, 3, 5])
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testEmpty(self):
    # [FUNCTION DOC]
    # Compares logical-and of an empty list over axis zero with the numpy result (true).
    self._compareAll([], [0])


class AnyReductionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the logical-or reduction on boolean arrays, compared with numpy any applied axis
  # by axis, on both GPU-enabled and CPU-only sessions.

  def _compare(self, x, reduction_axes, keepdims, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the TensorFlow logical-or reduction with a numpy reference applied one
    # axis at a time from the last. Inputs: array x, axes (or nothing), keepdims flag, and
    # whether to allow the GPU. Asserts exact equality of values and equal shapes.
    np_ans = x
    if reduction_axes is None:
      np_ans = np.any(np_ans, keepdims=keepdims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.any(np_ans, axis=ra, keepdims=keepdims)
    with self.cached_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_any(x, reduction_axes, keepdims)
      out = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    # [FUNCTION DOC]
    # HELPER: runs the comparison for keepdims false and true, each with and without the GPU.
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, False, use_gpu=False)
    self._compare(x, reduction_axes, True, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=False)

  def testAxesType(self):
    # [FUNCTION DOC]
    # Checks the logical-or reduction accepts 32-bit and 64-bit integer constant axes: reducing
    # two true values over axis zero gives true.
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.session():
        v = math_ops.reduce_any([True, True],
                                constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, True)

  def testAll3D(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is named for the all reduction (copied from the all-reduction tests)
    # but exercises the any reduction.
    # Compares logical-or over every possible axes choice for a random 2x3x5 boolean array in
    # which each element is true with probability about ten percent (uses random numbers).
    # Create a 3D array of bools and reduce across all possible
    # dimensions
    np_arr = (np.random.uniform(0, 1, 30) > 0.9).reshape([2, 3, 5])
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testEmpty(self):
    # [FUNCTION DOC]
    # Compares logical-or of an empty list over axis zero with the numpy result (false).
    self._compareAll([], [0])


class CountNonzeroReductionTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the count-nonzero reduction on booleans, floats and strings, compared with a numpy
  # reference that counts elements different from a given zero value. Covers axes combinations,
  # empty and degenerate inputs.

  def _compare(self, x, reduction_axes, keepdims, use_gpu=False, zero=0,
               feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: compares count-nonzero with a numpy reference. Inputs: array x, axes (or nothing),
    # keepdims flag, whether to allow the GPU, the value regarded as zero (default numeric zero,
    # strings use an empty string), and an optional feed mapping. The reference marks elements
    # different from zero and sums over each axis from the last. Runs the TensorFlow op in a
    # cached session and asserts values are close and shapes equal.
    np_ans = (x != zero).astype(np.int32)
    if reduction_axes is None:
      np_ans = np.sum(np_ans, keepdims=keepdims)
    else:
      reduction_axes = np.array(reduction_axes).astype(np.int32)
      for ra in reduction_axes.ravel()[::-1]:
        np_ans = np.sum(np_ans, axis=ra, keepdims=keepdims)
    with self.cached_session(use_gpu=use_gpu) as sess:
      tf_ans = math_ops.count_nonzero(x, reduction_axes, keepdims)
      out = sess.run(tf_ans, feed_dict)
    self.assertAllClose(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes, feed_dict=None):
    # [FUNCTION DOC]
    # HELPER: runs the comparison for keepdims false and true, each with and without the GPU,
    # forwarding the feed mapping. A length-one axes sequence is also tested in scalar form by
    # recursing (without the feed mapping).
    if reduction_axes is not None and np.shape(reduction_axes) == (1,):
      # Test scalar reduction_axes argument
      self._compareAll(x, reduction_axes[0])
    self._compare(x, reduction_axes, False, use_gpu=True, feed_dict=feed_dict)
    self._compare(x, reduction_axes, False, use_gpu=False, feed_dict=feed_dict)
    self._compare(x, reduction_axes, True, use_gpu=True, feed_dict=feed_dict)
    self._compare(x, reduction_axes, True, use_gpu=False, feed_dict=feed_dict)

  @test_util.run_deprecated_v1
  def testBoolReduce1D(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the in-body comment says the array is floats, but the array is boolean.
    # Graph-mode test: counts true values in a one-dimensional boolean array over no axes, an
    # empty axes list and axis zero.
    # Create a 1D array of floats
    np_arr = np.asarray([False, False, True, False, False, True])
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])

  @test_util.run_deprecated_v1
  def testFloatReduce1D(self):
    # [FUNCTION DOC]
    # Graph-mode test: counts nonzero entries of a one-dimensional float32 array over axis zero.
    # Create a 1D array of floats
    np_arr = np.asarray([0.0, 1.0, -1.0, 0.0, 0.0, 3.0]).astype(np.float32)
    self._compareAll(np_arr, [0])

  @test_util.run_deprecated_v1
  def testFloatReduce4D(self):
    # [FUNCTION DOC]
    # Graph-mode test: counts nonzero entries in a 2x3x5x7 float32 array of zeros and ones over
    # several axes combinations. The case of axes zero and two is deliberately omitted, noted as
    # needing a specialization.
    # Create a 4D array of floats and reduce across some
    # dimensions
    np_arr = np.floor(np.arange(0.0, 210.0) / 100.0).reshape([2, 3, 5,
                                                              7]).astype(
                                                                  np.float32)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    # Need specialization for reduce(4D, [0, 2])
    # self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])
    self._compareAll(np_arr, [1, 2, 3])
    self._compareAll(np_arr, [0, 1, 2, 3])

  @test_util.run_deprecated_v1
  def testExpand(self):
    # [FUNCTION DOC]
    # Graph-mode test: counts nonzero entries of a 5x0 array over axis one, reducing an empty
    # tensor to a nonempty zero-filled one.
    # Reduce an empty tensor to a nonempty tensor
    x = np.zeros((5, 0))
    self._compareAll(x, [1])

  @test_util.run_deprecated_v1
  def testDegenerate(self):
    # [FUNCTION DOC]
    # Graph-mode test: for CPU and GPU sessions, counts nonzero entries of a zero-by-9938
    # boolean tensor over axis zero (a size once crashing Eigen) and asserts the result is zeros.
    for use_gpu in False, True:
      with self.cached_session(use_gpu=use_gpu):
        for dtype in (dtypes.bool,):
          # A large number is needed to get Eigen to die
          x = array_ops.zeros((0, 9938), dtype=dtype)
          y = math_ops.count_nonzero(x, [0])
          self.assertAllEqual(y, np.zeros(9938))

  def testStringReduce(self):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: counts nonzero entries of a one-element string tensor
    # (nonempty string) without specifying axes and asserts the count is one.
    # Test case for GitHub issue 18712
    with self.cached_session() as sess:
      v = math_ops.count_nonzero(constant_op.constant(["test"]))
      self.assertAllClose(self.evaluate(v), 1)

  @test_util.run_deprecated_v1
  def testStringReduce1D(self):
    # [FUNCTION DOC]
    # Graph-mode test: counts nonempty strings in a one-dimensional string array, with an empty
    # string as the zero value, for no axes, empty axes and axis zero, with keepdims false and
    # true.
    # Create a 1D array of strings
    x = np.asarray(["", "", "a", "", "", "b"])
    self._compare(x, None, keepdims=False, zero=np.str_(""))
    self._compare(x, [], keepdims=False, zero=np.str_(""))
    self._compare(x, [0], keepdims=False, zero=np.str_(""))
    self._compare(x, None, keepdims=True, zero=np.str_(""))
    self._compare(x, [], keepdims=True, zero=np.str_(""))
    self._compare(x, [0], keepdims=True, zero=np.str_(""))

  @test_util.run_deprecated_v1
  def testStringReduce2D(self):
    # [FUNCTION DOC]
    # Graph-mode test: counts nonempty strings in a 3x6 string array, with an empty string as the
    # zero value, for no axes, empty axes, each axis and both axes, with keepdims false and true
    # (some combinations are only tested with keepdims false or true).
    # Create a 2D array of strings
    x = np.asarray([["", "", "a", "", "", "b"],
                    ["", "c", "", "d", "", ""],
                    ["e", "", "f", "", "", ""]])
    self._compare(x, None, keepdims=False, zero=np.str_(""))
    self._compare(x, [], keepdims=False, zero=np.str_(""))
    self._compare(x, [0], keepdims=False, zero=np.str_(""))
    self._compare(x, [1], keepdims=False, zero=np.str_(""))
    self._compare(x, [0, 1], keepdims=False, zero=np.str_(""))
    self._compare(x, None, keepdims=True, zero=np.str_(""))
    self._compare(x, [], keepdims=True, zero=np.str_(""))
    self._compare(x, [0], keepdims=True, zero=np.str_(""))
    self._compare(x, [0, 1], keepdims=True, zero=np.str_(""))


if __name__ == "__main__":
  test.main()
