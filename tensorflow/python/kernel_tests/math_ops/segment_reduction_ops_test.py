# [MODULE DOC]
# Functional tests for the segment reduction ops in the math ops module: sorted segment ops (sum, mean,
# prod, min, max), unsorted segment ops (sum, mean, sqrt_n, prod, min, max), and sparse segment ops
# (sum, mean, sqrt_n, their with-num-segments variants and their gradient ops, including the v2
# gradients). Results are compared against a pure numpy reference implementation, gradients are
# checked numerically, and invalid segment ids / indices are checked to raise the expected errors.
# The module also contains a GPU-only benchmark class for sorted and unsorted segment sum.
# It defines helper base classes (reference implementations and input builders) shared by the test
# classes. No registrations or flags are defined at import time; the test runner starts when the
# module is executed as a script.
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
"""Functional tests for segment reduction ops."""

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class SegmentReductionHelper(test.TestCase):
  # [CLASS DOC]
  # HELPER: base class providing input builders and numpy reference reductions for the segment tests.
  # Holds no state. Subclasses use _input to create test data, _segmentReduce as the numpy oracle,
  # and the mean / sqrt_n callbacks to emulate averaging reductions. Contains no tests of its own.

  def _input(self, input_shape, dtype=dtypes_lib.int32):
    # [FUNCTION DOC]
    # HELPER: builds test data for the segment reduction tests.
    # Takes a shape (list of ints) and a TensorFlow dtype (default int32). Creates the numbers
    # 1..N in row-major order reshaped to the shape and cast to the dtype. For bfloat16 the values
    # are scaled down by a small factor to keep absolute error small; for complex dtypes a
    # non-zero imaginary part (the negative of the real part) is added.
    # Returns a pair: a constant tensor and the equivalent numpy array. No side effects.
    num_elem = 1
    for x in input_shape:
      num_elem *= x
    values = np.arange(1, num_elem + 1)
    np_values = values.reshape(input_shape).astype(dtype.as_numpy_dtype)
    if dtype == dtypes_lib.bfloat16:
      # Large numbers from arange lead to high absolute diff with bfloat16, so
      # scale down.
      np_values *= np.array(0.00001, dtype=dtype.as_numpy_dtype)
    # Add a non-zero imaginary component to complex types.
    if dtype.is_complex:
      np_values -= 1j * np_values
    return constant_op.constant(
        np_values, shape=input_shape, dtype=dtype), np_values

  def _segmentReduce(
      self,
      indices,
      x,
      op1,
      op2=None,
      num_segments=None,
      initial_value=0,
      empty_value=0,
  ):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation of segment reductions used as the expected result.
    # KNOWN ISSUE: the initial_value parameter is accepted but never used; callers pass it
    # (e.g. for product or max tests) but only empty_value affects the output.
    # Inputs: indices (segment id per leading row, any integer array-like), x (numpy array of
    # data), op1 (binary combining callback applied cumulatively), op2 (optional finishing
    # callback applied to each segment result, used for mean / sqrt_n), num_segments (optional;
    # defaults to the last index plus one), and empty_value (fill for segments that received no
    # rows). When op1 is the numpy max function, the reduction is done element by element.
    # Returns a numpy array with one reduced slice per segment, or an empty array if x is empty.
    # No side effects.
    if not x.size:
      return np.array([])
    indices = np.asarray(indices)
    if num_segments is None:
      num_segments = indices[-1] + 1
    output = [None] * num_segments
    slice_shape = x.shape[indices.ndim:]
    x_flat = x.reshape((indices.size,) + slice_shape)
    for i, index in enumerate(indices.ravel()):
      if (output[index] is not None) and op1 == np.max:
        for j in range(0, output[index].shape[0]):
          output[index][j] = op1([output[index][j], x_flat[i][j]])
      elif output[index] is not None:
        output[index] = op1(output[index], x_flat[i])
      else:
        output[index] = x_flat[i]
    # zero initialize values that are still uncalculated.
    empty_value_slice = np.ones(slice_shape) * empty_value
    output = [o if o is not None else empty_value_slice for o in output]
    if op2 is not None:
      output = [op2(o) for o in output]
    output = [o.reshape(slice_shape) for o in output]
    return np.array(output)

  def _mean_cum_op(self, x, y):
    # [FUNCTION DOC]
    # HELPER: cumulative combining callback for the numpy mean reference.
    # Takes an accumulator x and a new row y. If x is already a (running sum, count) pair, returns
    # the pair with y added and the count incremented; otherwise treats x as the first row and
    # returns (x plus y, 2). Pure function, no side effects.
    return (x[0] + y, x[1] + 1) if isinstance(x, tuple) else (x + y, 2)

  def _mean_reduce_op(self, x):
    # [FUNCTION DOC]
    # HELPER: finishing callback for the numpy mean reference.
    # If x is a (sum, count) pair returns sum divided by count; otherwise (a segment with a single
    # row) returns x unchanged. Pure function.
    return x[0] / x[1] if isinstance(x, tuple) else x

  def _sqrt_n_reduce_op(self, x):
    # [FUNCTION DOC]
    # HELPER: finishing callback for the numpy sqrt_n reference.
    # If x is a (sum, count) pair returns sum divided by the square root of count; otherwise
    # returns x unchanged (single-row segment, square root of one). Pure function.
    return x[0] / np.sqrt(x[1]) if isinstance(x, tuple) else x


class SegmentReductionOpTest(SegmentReductionHelper, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the sorted segment ops (segment sum, mean, prod, min, max). Covers value correctness
  # across dtypes against the numpy oracle, segment-id validation errors, gradients, and invalid
  # input regressions. Uses the helper base class for data and reference computation.

  @parameterized.parameters(
      (dtypes_lib.float32),
      (dtypes_lib.float64),
      (dtypes_lib.int64),
      (dtypes_lib.int32),
      (dtypes_lib.complex64),
      (dtypes_lib.complex128),
      (dtypes_lib.bfloat16),
  )
  def testValues(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float32/64, int32/64, complex64/128 and bfloat16. For several inner sizes
    # (1, 2, 3, 32) and on both GPU and CPU placement, builds a [10, inner] input with segment ids
    # of three rows each, runs segment sum, mean, prod, min and max (complex dtypes: only sum, prod
    # and mean), and asserts the result is close to the numpy reference (tolerance per dtype) and
    # that the static shape after the first dimension matches.
    # Each item is np_op1, np_op2, tf_op
    ops_list = [(np.add, None, math_ops.segment_sum),
                (self._mean_cum_op, self._mean_reduce_op,
                 math_ops.segment_mean),
                (np.ndarray.__mul__, None, math_ops.segment_prod),
                (np.minimum, None, math_ops.segment_min),
                (np.maximum, None, math_ops.segment_max)]

    # A subset of ops has been enabled for complex numbers
    complex_ops_list = [(np.add, None, math_ops.segment_sum),
                        (np.ndarray.__mul__, None, math_ops.segment_prod),
                        (self._mean_cum_op, self._mean_reduce_op,
                         math_ops.segment_mean)]

    n = 10
    # Note that the GPU implem has different paths for different inner sizes.
    for shape in [[n, 1], [n, 2], [n, 3], [n, 32]]:
      indices = [i // 3 for i in range(n)]
      if dtype in (dtypes_lib.complex64, dtypes_lib.complex128):
        curr_ops_list = complex_ops_list
      else:
        curr_ops_list = ops_list
      for use_gpu in [True, False]:
        with self.cached_session(use_gpu=use_gpu):
          tf_x, np_x = self._input(shape, dtype=dtype)
          for np_op1, np_op2, tf_op in curr_ops_list:
            initial_value = 1 if tf_op is math_ops.segment_prod else 0
            np_ans = self._segmentReduce(
                indices, np_x, np_op1, np_op2, initial_value=initial_value)
            s = tf_op(data=tf_x, segment_ids=indices)
            tf_ans = self.evaluate(s)
            self.assertAllCloseAccordingToType(np_ans, tf_ans)
            # NOTE(mrry): The static shape inference that computes
            # `tf_ans.shape` can only infer that sizes from dimension 1
            # onwards, because the size of dimension 0 is data-dependent
            # and may therefore vary dynamically.
            self.assertAllEqual(np_ans.shape[1:], tf_ans.shape[1:])

  @test_util.run_deprecated_v1
  def testSegmentIdsShape(self):
    # [FUNCTION DOC]
    # Runs in deprecated graph mode. Passing a rank-2 segment_ids tensor to segment sum on a 4x4
    # input is expected to raise a value error at graph construction time.
    shape = [4, 4]
    tf_x, _ = self._input(shape)
    indices = constant_op.constant([0, 1, 2, 2], shape=[2, 2])
    with self.assertRaises(ValueError):
      math_ops.segment_sum(data=tf_x, segment_ids=indices)

  @test_util.run_deprecated_v1
  def testSegmentIdsSize(self):
    # [FUNCTION DOC]
    # Graph mode, on GPU and CPU. A 4x4 input with only two segment ids must make evaluation fail
    # with an op error saying segment_ids should be the same size as the first dimension.
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape)
        indices = [0, 1]
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        with self.assertRaisesOpError("segment_ids should be the same size"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsValid(self):
    # [FUNCTION DOC]
    # Baseline for the invalid-segment-id tests. Graph mode, GPU and CPU: segment sum over a 4x4
    # float input with sorted ids (0,0,0,1) is evaluated and compared to the exact expected sums
    # (first segment is the sum of three rows, second is the last row).
    # This is a baseline for the following SegmentIdsInvalid* tests.
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 0, 1]
        result = math_ops.segment_sum(data=tf_x, segment_ids=indices).eval()
        self.assertAllEqual([[15, 18, 21, 24], [13, 14, 15, 16]], result)

  def testSegmentIdsGreaterThanZero(self):
    # [FUNCTION DOC]
    # Segment ids that start at 1 (ids 1,1,2,2) instead of 0: segment sum on GPU and CPU must
    # match the numpy reference, meaning segment 0 is empty and filled with zeros.
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float32)
        indices = [1, 1, 2, 2]
        np_ans = self._segmentReduce(indices, np_x, np.add)
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  def testSegmentIdsHole(self):
    # [FUNCTION DOC]
    # Segment ids with a gap (0,0,3,3): segment sum on GPU and CPU must produce zero rows for the
    # skipped segments 1 and 2 and match the numpy reference.
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 3, 3]
        np_ans = self._segmentReduce(indices, np_x, np.add)
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  def testSegmentIdsHoleEmptyValue(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: passes an initial_value to the numpy reference, but that parameter is ignored
    # by the reference, so it has no effect on the expected result.
    # Segment ids with a gap (0,0,3,3) on GPU and CPU: segment max must give zeros (the empty
    # value) for the missing segments and match the numpy max reference elsewhere.
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 3, 3]
        np_ans = self._segmentReduce(
            indices,
            np_x,
            np.max,
            initial_value=-dtypes_lib.float32.min,
            empty_value=0,
        )
        s = math_ops.segment_max(data=tf_x, segment_ids=indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid1(self):
    # [FUNCTION DOC]
    # Graph mode. Ids (-1,-1,0,0) must make segment sum fail on evaluation with an op error saying
    # segment id -1 is out of range [0, 1) possibly because the ids are not sorted.
    shape = [4, 4]
    with self.cached_session():
      tf_x, _ = self._input(shape)
      indices = [-1, -1, 0, 0]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError(
          r"Segment id -1 out of range \[0, 1\), possibly because "
          "'segment_ids' input is not sorted."):
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid2(self):
    # [FUNCTION DOC]
    # Graph mode. Ids (0,1,0,1) are not sorted; evaluating segment sum must raise an op error
    # saying segment ids are not increasing.
    shape = [4, 4]
    with self.cached_session():
      tf_x, _ = self._input(shape)
      indices = [0, 1, 0, 1]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment ids are not increasing"):
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid3(self):
    # [FUNCTION DOC]
    # Graph mode. Ids (0,1,2,0) decrease at the end; evaluating segment sum must raise an op error
    # saying segment id 1 is out of range [0, 1) possibly because the ids are not sorted.
    shape = [4, 4]
    with self.cached_session():
      tf_x, _ = self._input(shape)
      indices = [0, 1, 2, 0]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError(
          r"Segment id 1 out of range \[0, 1\), possibly "
          "because 'segment_ids' input is not sorted."):
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid4(self):
    # [FUNCTION DOC]
    # Graph mode, GPU and CPU. Ids (0,0,0,-1) contain a negative id; evaluating segment sum must
    # raise an op error saying segment ids must be non-negative.
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 0, -1]
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid5(self):
    # [FUNCTION DOC]
    # Same as the previous negative-id test but with ids (0,0,0,-2): evaluation must raise an op
    # error saying segment ids must be non-negative. Graph mode, GPU and CPU.
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 0, -2]
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Graph mode. For segment sum, mean, min and max on a 4x4 double input with ids (0,1,2,2),
    # computes the theoretical and numerical Jacobians with the gradient checker (output shape
    # 3x4) and asserts they are close.
    shape = [4, 4]
    indices = [0, 1, 2, 2]
    for tf_op in [
        math_ops.segment_sum, math_ops.segment_mean, math_ops.segment_min,
        math_ops.segment_max
    ]:
      with self.cached_session():
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float64)
        s = tf_op(data=tf_x, segment_ids=indices)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [3, 4],
            x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n)

  def testDataInvalid(self):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. A scalar (rank 0) data input to segment mean with an
    # empty segment id array must raise either a value error or an invalid-argument error whose
    # message says the input must be at least rank 1. Runs on GPU and CPU.
    # Test case for GitHub issue 40653.
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        with self.assertRaisesRegex(
            (ValueError, errors_impl.InvalidArgumentError),
            "must be at least rank 1"):
          s = math_ops.segment_mean(
              data=np.uint16(10), segment_ids=np.array([]).astype("int64"))
          self.evaluate(s)

  def testInvalidIds(self):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue. For segment max, min, mean, sum and prod, a single huge
    # segment id (about 1.6e18) with a 1x10x1 data array must raise a value error or an
    # invalid-argument error rather than crashing or allocating. CPU only.
    # Test case for GitHub issue 46888.
    for op in [
        math_ops.segment_max,
        math_ops.segment_min,
        math_ops.segment_mean,
        math_ops.segment_sum,
        math_ops.segment_prod,
    ]:
      with self.cached_session(use_gpu=False):
        with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
          s = op(data=np.ones((1, 10, 1)), segment_ids=[1676240524292489355])
          self.evaluate(s)


class UnsortedSegmentTest(SegmentReductionHelper, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the unsorted segment ops (sum, mean, sqrt_n, prod, min, max). Instance state is set
  # in the constructor: ops_list and complex_ops_list (tuples of numpy combine callback, numpy
  # finish callback, TensorFlow op, and a callback giving the initial value per dtype),
  # differentiable_dtypes and all_dtypes. Covers values, index / num_segments dtypes, gradients
  # (graph and tape), negative-id dropping, and error cases.

  def __init__(self, methodName='runTest'):
    # [FUNCTION DOC]
    # Test case constructor. Takes the test method name (default is the standard run entry) and
    # builds the instance attributes listed in the class doc: the op tables for all dtypes and for
    # complex dtypes (only sum and prod), and the dtype lists. Then calls the parent constructor.
    # Each item is np_op1, np_op2, tf_op, initial_value functor
    self.ops_list = [(np.add, None,
                      math_ops.unsorted_segment_sum, lambda t: 0),
                     (self._mean_cum_op, self._mean_reduce_op,
                      math_ops.unsorted_segment_mean, lambda t: 0),
                     (self._mean_cum_op, self._sqrt_n_reduce_op,
                      math_ops.unsorted_segment_sqrt_n, lambda t: 0),
                     (np.ndarray.__mul__, None,
                      math_ops.unsorted_segment_prod, lambda t: 1),
                     (np.minimum, None,
                      math_ops.unsorted_segment_min, lambda t: t.max),
                     (np.maximum, None,
                      math_ops.unsorted_segment_max, lambda t: t.min)]

    # A subset of ops has been enabled for complex numbers
    self.complex_ops_list = [(np.add, None,
                              math_ops.unsorted_segment_sum, lambda t: 0),
                             (np.ndarray.__mul__, None,
                              math_ops.unsorted_segment_prod, lambda t: 1)]
    self.differentiable_dtypes = [dtypes_lib.float16, dtypes_lib.float32,
                                  dtypes_lib.float64]
    self.all_dtypes = (self.differentiable_dtypes +
                       [dtypes_lib.bfloat16,
                        dtypes_lib.int64, dtypes_lib.int32,
                        dtypes_lib.complex64, dtypes_lib.complex128])
    super(UnsortedSegmentTest, self).__init__(methodName=methodName)

  @parameterized.parameters((dtypes_lib.float16), (dtypes_lib.float32),
                            (dtypes_lib.float64), (dtypes_lib.bfloat16),
                            (dtypes_lib.int64), (dtypes_lib.int32),
                            (dtypes_lib.complex64), (dtypes_lib.complex128))
  def testValues(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float16/32/64, bfloat16, int32/64 and complex64/128. Uses unsorted ids
    # (flat and reshaped to 5x2) with 12 segments, inner sizes 1, 2, 3, 32, and GPU/CPU flags.
    # For each op in the table (complex: sum and prod only; integer dtypes skip sqrt_n) compares
    # the TensorFlow result to the numpy reference with per-dtype tolerance and checks the static
    # shape equals the reference shape.
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      # Note that the GPU implem has different paths for different inner sizes.
      for inner_size in [1, 2, 3, 32]:
        shape = indices.shape + (inner_size,)

        ops_list = (
            self.complex_ops_list if dtype.is_complex else self.ops_list)
        tf_x, np_x = self._input(shape, dtype=dtype)
        for use_gpu in [True, False]:
          with self.cached_session():
            for np_op1, np_op2, tf_op, init_op in ops_list:
              # sqrt_n doesn't support integers
              if (np_op2 == self._sqrt_n_reduce_op and dtype.is_integer):
                continue
              np_ans = self._segmentReduce(
                  indices,
                  np_x,
                  np_op1,
                  np_op2,
                  num_segments=num_segments,
                  initial_value=init_op(dtype),
                  empty_value=init_op(dtype),
              )
              s = tf_op(tf_x, segment_ids=indices, num_segments=num_segments)
              tf_ans = self.evaluate(s)
              self.assertAllCloseAccordingToType(np_ans, tf_ans)
              self.assertShapeEqual(np_ans, s)

  def testNumSegmentsTypes(self):
    # [FUNCTION DOC]
    # Checks that unsorted segment sum accepts num_segments given as a 32-bit or 64-bit integer
    # constant, for flat and 5x2 id arrays, and matches the numpy reference in value and shape.
    dtypes = [dtypes_lib.int32, dtypes_lib.int64]
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in dtypes:
        with self.cached_session():
          tf_x, np_x = self._input(shape)
          num_segments_constant = constant_op.constant(
              num_segments, dtype=dtype)
          np_ans = self._segmentReduce(
              indices, np_x, np.add, op2=None, num_segments=num_segments)
          s = math_ops.unsorted_segment_sum(
              data=tf_x,
              segment_ids=indices,
              num_segments=num_segments_constant)
          tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, s)

  def testIndicesTypes(self):
    # [FUNCTION DOC]
    # Checks that unsorted segment sum accepts segment ids of 16-, 32- and 64-bit integer dtypes
    # (flat and 5x2 shapes) and matches the numpy reference in value and static shape.
    dtypes = [dtypes_lib.int16, dtypes_lib.int32, dtypes_lib.int64]
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in dtypes:
        with self.cached_session():
          tf_x, np_x = self._input(shape)
          indices_constant = constant_op.constant(indices, dtype=dtype)
          np_ans = self._segmentReduce(
              indices, np_x, np.add, op2=None, num_segments=num_segments
          )
          s = math_ops.unsorted_segment_sum(
              data=tf_x, segment_ids=indices_constant, num_segments=num_segments
          )
          tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, s)

  @test_util.run_deprecated_v1
  def testGradientsTFGradients(self):
    # [FUNCTION DOC]
    # Graph mode. For each differentiable dtype, id layout (with negative ids that must be
    # ignored), GPU/CPU choice and each op in the table, compares the theoretical and numerical
    # Jacobians from the graph gradient checker (half precision uses a looser absolute tolerance).
    num_cols = 2
    indices_flat = np.array([0, 4, 0, -1, 3, -1, 4, 7, 7, 3])
    num_segments = max(indices_flat) + 3
    for dtype in self.differentiable_dtypes:
      ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
      for indices in indices_flat, indices_flat.reshape(5, 2):
        shape = indices.shape + (num_cols,)
        # test CPU and GPU as tf.gather behaves differently on each device
        for use_gpu in [False, True]:
          with self.cached_session(use_gpu=use_gpu):
            for _, _, tf_op, _ in ops_list:
              tf_x, np_x = self._input(shape, dtype=dtype)
              s = tf_op(tf_x, indices, num_segments)
              jacob_t, jacob_n = gradient_checker.compute_gradient(
                  tf_x,
                  shape,
                  s, [num_segments, num_cols],
                  x_init_value=np_x,
                  delta=1.)
              self.assertAllCloseAccordingToType(jacob_t, jacob_n,
                                                 half_atol=1e-2)

  @parameterized.parameters(
      (dtypes_lib.float16),
      (dtypes_lib.float32),
      (dtypes_lib.float64),
      (dtypes_lib.bfloat16),
  )
  @test_util.run_in_graph_and_eager_modes
  def testGradientsGradientTape(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float16/32/64 and bfloat16, in both graph and eager modes. Same scenario
    # as the graph gradient test but using the tape-based gradient checker: for each op, id layout
    # (including negative ids) and forced device, compares tape and numerical Jacobians with
    # relaxed tolerances for half and bfloat16.
    num_cols = 2
    indices_flat = np.array([0, 4, 0, -1, 3, -1, 4, 7, 7, 3])
    num_segments = max(indices_flat) + 3

    ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (num_cols,)
      # test CPU and GPU as tf.gather behaves differently on each device
      for use_gpu in [test_util.use_gpu, test_util.force_cpu]:
        with use_gpu():
          for _, _, tf_op, _ in ops_list:
            _, np_x = self._input(shape, dtype=dtype)

            # pylint: disable=cell-var-from-loop
            def f(x):
              # [FUNCTION DOC]
              # Closure for the gradient checker: applies the current unsorted segment op to the
              # given tensor x using the enclosing ids and number of segments, and returns the
              # result. Captures loop variables by reference; no side effects.
              return tf_op(x, indices, num_segments)

            gradient_tape_jacob_t, jacob_n = (
                gradient_checker_v2.compute_gradient(f, [np_x], delta=1.))
            # pylint: enable=cell-var-from-loop
            self.assertAllCloseAccordingToType(
                jacob_n,
                gradient_tape_jacob_t,
                half_atol=1e-2,
                bfloat16_rtol=5e-1,
                bfloat16_atol=5e-1,
            )

  @test_util.run_deprecated_v1
  def testProdGrad(self):
    # [FUNCTION DOC]
    # Graph mode. Checks the unsorted segment product gradient with inputs containing zeros, for
    # ids without and with negative (ignored) entries, on CPU and GPU. Asserts that numerical and
    # theoretical Jacobians agree and that both equal hand-computed ground-truth derivatives.
    # additional test for the prod gradient to ensure correct handling of zeros
    values = np.array([0, 0, 1, 0, 2, 2, 3, 3, 3], dtype=np.float32)
    indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    indices_neg = np.array([-1, 0, 0, -1, 1, 1, -1, 2, 2], dtype=np.int32)
    values_tf = constant_op.constant(values)
    # ground truth partial derivatives
    gradients_indices = np.zeros((9, 3), dtype=np.float32)
    gradients_indices_neg = np.zeros((9, 3), dtype=np.float32)
    # the derivative w.r.t. to the other segments is zero, so here we only
    # explicitly set the grad values for the corresponding segment
    gradients_indices[range(9), indices] = [0, 0, 0, 4, 0, 0, 9, 9, 9]
    gradients_indices_neg[range(9), indices_neg] = [0, 1, 0, 0, 2, 2, 0, 3, 3]
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        for ind, grad_gt in [(indices, gradients_indices),
                             (indices_neg, gradients_indices_neg)]:
          s = math_ops.unsorted_segment_prod(values_tf,
                                             constant_op.constant(ind), 3)
          jacob_t, jacob_n = gradient_checker.compute_gradient(
              values_tf, (9,), s, (3,), x_init_value=values, delta=1)
          self.assertAllClose(jacob_t, jacob_n)
          self.assertAllClose(jacob_t, grad_gt)

  @test_util.run_deprecated_v1
  def testGradientMatchesSegmentSum(self):
    # [FUNCTION DOC]
    # Graph mode. For valid sorted contiguous ids, computes Jacobians of unsorted segment sum and
    # of sorted segment sum for each differentiable dtype and asserts the theoretical and the
    # numerical Jacobians are respectively identical between the two ops.
    # Strategy: compute the gradient for UnsortedSegmentSum and SegmentSum
    # and compare the outputs, which should be identical.
    # NB: for this test to work, indices must be valid for SegmentSum, namely
    # it must be sorted, the indices must be contiguous, and num_segments
    # must be max(indices) + 1.
    indices = [0, 0, 1, 1, 1, 2, 3, 4, 5]
    n = len(indices)
    num_cols = 2
    shape = [n, num_cols]
    num_segments = max(indices) + 1
    for dtype in self.differentiable_dtypes:
      with self.cached_session():
        tf_x, np_x = self._input(shape, dtype=dtype)
        # Results from UnsortedSegmentSum
        unsorted_s = math_ops.unsorted_segment_sum(
            data=tf_x, segment_ids=indices, num_segments=num_segments)
        unsorted_jacob_t, unsorted_jacob_n = (
            gradient_checker.compute_gradient(tf_x, shape, unsorted_s,
                                              [num_segments, num_cols],
                                              x_init_value=np_x, delta=1))

        # Results from SegmentSum
        sorted_s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        sorted_jacob_t, sorted_jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            sorted_s, [num_segments, num_cols],
            x_init_value=np_x,
            delta=1)
      self.assertAllClose(unsorted_jacob_t, sorted_jacob_t)
      self.assertAllClose(unsorted_jacob_n, sorted_jacob_n)

  @test_util.run_deprecated_v1
  def testBadIndices(self):
    # [FUNCTION DOC]
    # Graph mode, CPU only. Out-of-range positive segment ids (2 and 7 with two segments) must make
    # unsorted segment sum fail on evaluation with an op error naming the out-of-range id.
    # Note: GPU kernel does not return the out-of-range error needed for this
    # test, so this test is marked as cpu-only.
    # Note: With PR #13055 a negative index will be ignored silently.
    with self.session(use_gpu=False):
      for bad in [[2]], [[7]]:
        unsorted = math_ops.unsorted_segment_sum([[17]], bad, num_segments=2)
        with self.assertRaisesOpError(
            r"segment_ids\[0,0\] = %d is out of range \[0, 2\)" % bad[0][0]):
          self.evaluate(unsorted)

  @test_util.run_deprecated_v1
  def testEmptySecondDimension(self):
    # [FUNCTION DOC]
    # Graph mode. For several numeric dtypes and both 32- and 64-bit id dtypes, unsorted segment
    # sum over a 2x0 data array must return a 2x0 array of zeros.
    dtypes = [np.float16, np.float32, np.float64, np.int64, np.int32,
              np.complex64, np.complex128]
    with self.session():
      for dtype in dtypes:
        for itype in (np.int32, np.int64):
          data = np.zeros((2, 0), dtype=dtype)
          segment_ids = np.array([0, 1], dtype=itype)
          unsorted = math_ops.unsorted_segment_sum(data, segment_ids, 2)
          self.assertAllEqual(unsorted, np.zeros((2, 0), dtype=dtype))

  def testDropNegatives(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the in-place replacement of id 8 by -1 mutates the shared id array (the 5x2
    # variant is a view of the flat array), so later iterations and dtypes see already modified ids;
    # the expected output zeroing is applied once per iteration regardless.
    # For every dtype, unsorted segment sum is run with some ids replaced by -1 and the result must
    # equal the reference with the rows of the dropped segment zeroed, i.e. negative ids are
    # ignored. Also checks the static shape.
    # Note: the test is done by replacing segment_ids with 8 to -1
    # for index  and replace values generated by numpy with 0.
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in self.all_dtypes:
        with self.session():
          tf_x, np_x = self._input(shape, dtype=dtype)
          np_ans = self._segmentReduce(
              indices, np_x, np.add, op2=None, num_segments=num_segments)
          # Replace np_ans[8] with 0 for the value
          np_ans[8:] = 0
          # Replace 8 with -1 in indices
          np.place(indices, indices == 8, [-1])
          s = math_ops.unsorted_segment_sum(
              data=tf_x, segment_ids=indices, num_segments=num_segments)
          tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, s)

  @test_util.run_deprecated_v1
  def testAllNegatives(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. When every segment id is -1, unsorted segment sum of a 2x1 float array into
    # two segments must return zeros of shape 2x1.
    with self.session(use_gpu=False):
      data = np.ones((2, 1), dtype=np.float32)
      segment_ids = np.array([-1, -1], dtype=np.int32)
      unsorted = math_ops.unsorted_segment_sum(data, segment_ids, 2)
      self.assertAllClose(unsorted.eval(), np.zeros((2, 1), dtype=np.float32))

  @test_util.run_deprecated_v1
  def testBadNumSegments(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. A gigantic num_segments (about 8e18) with a scalar segment id must fail on
    # evaluation with an op error about overflow when multiplying or about a negative value.
    with self.session(use_gpu=False):
      num_segments = 8327099846119777499
      unsorted = math_ops.unsorted_segment_sum(
          np.ones((3)), segment_ids=898042203, num_segments=num_segments)
      with self.assertRaisesOpError(
          "Encountered overflow when multiplying | must not be negative"
      ):
        self.evaluate(unsorted)


class SparseSegmentReductionHelper(SegmentReductionHelper):
  # [CLASS DOC]
  # HELPER: base class adding input builders and numpy reference computations (forward and
  # gradient, v1 and v2) for the sparse segment tests. Holds no state and has no tests.

  def _sparse_input(self, input_shape, num_indices, dtype=dtypes_lib.int32):
    # [FUNCTION DOC]
    # HELPER: builds data plus random gather indices for sparse segment tests.
    # Takes the data shape, the number of indices to draw and a dtype (default int32). Builds the
    # data with the parent input builder, draws indices uniformly at random in the range of the
    # first dimension (randomness uses the global numpy generator). Returns a tuple: constant
    # int32 index tensor, numpy index array, data tensor, data numpy array.
    a, b = super(SparseSegmentReductionHelper, self)._input(input_shape, dtype)
    indices = np.random.randint(0, input_shape[0], num_indices).astype(np.int32)
    return (constant_op.constant(
        indices, dtype=dtypes_lib.int32), indices, a, b)

  def _sparseSegmentReduce(self,
                           x,
                           indices,
                           segment_indices,
                           op1,
                           op2=None,
                           num_segments=None):
    # [FUNCTION DOC]
    # HELPER: numpy reference for sparse segment reductions.
    # Gathers the rows of x selected by indices, then reduces them by segment_indices using the
    # dense reference with the combine callback op1, optional finishing callback op2 and optional
    # num_segments. Returns a numpy array with one row per segment. No side effects.
    return self._segmentReduce(
        segment_indices, x[indices], op1, op2, num_segments=num_segments)

  def _sparseSegmentReduceGradWeights(self, ygrad, segment_ids, mode):
    # [FUNCTION DOC]
    # HELPER: computes per-segment gradient scaling weights for the numpy gradient references.
    # Inputs: ygrad (numpy output gradient, one row per segment), segment_ids (segment of each
    # gathered row) and mode, which must be sum, mean or sqrtn (otherwise an assertion fails).
    # Sum gives weight one for every segment; mean gives the reciprocal of the segment's row count;
    # sqrtn gives the reciprocal of the square root of the count. Returns a numpy vector of
    # weights. Segments without rows get an infinite weight (division by zero) but are not used.
    assert mode in ("sum", "mean", "sqrtn")
    if mode == "sum":
      weights = np.ones(ygrad.shape[0], ygrad.dtype)
    else:
      weights = np.zeros(ygrad.shape[0], ygrad.dtype)
      for segment in segment_ids:
        weights[segment] += 1
      weights = 1. / weights if mode == "mean" else 1. / np.sqrt(weights)
    return weights

  def _sparseSegmentReduceGrad(
      self, ygrad, indices, segment_ids, output_dim0, mode
  ):
    # [FUNCTION DOC]
    # HELPER: numpy reference for the gradient of a sparse segment reduction (v1 form).
    # Inputs: ygrad (output gradient array), indices (source row of each gathered element),
    # segment_ids (segment of each element), output_dim0 (number of rows of the result) and mode
    # (sum, mean or sqrtn; anything else fails an assertion). Scatter-adds each segment's gradient
    # row times its weight into the source row. Returns a dense gradient with output_dim0 rows.
    assert mode in ("sum", "mean", "sqrtn")
    weights = self._sparseSegmentReduceGradWeights(ygrad, segment_ids, mode)
    xgrad = np.zeros([output_dim0, ygrad.shape[1]], ygrad.dtype)
    for segment, index in zip(segment_ids, indices):
      xgrad[index] += ygrad[segment] * weights[segment]
    return xgrad

  def _sparseSegmentReduceGradV2(self, ygrad, indices, segment_ids, mode):
    # [FUNCTION DOC]
    # HELPER: numpy reference for the v2 gradient of a sparse segment reduction, which returns
    # gradients only for the unique source indices.
    # Inputs as in the v1 reference but without an output size; mode must be sum, mean or sqrtn
    # (assertion otherwise). Returns a pair: the gradient rows for each unique index (in sorted
    # order of unique indices) and the sorted array of unique indices.
    assert mode in ("sum", "mean", "sqrtn")
    unique_indices, unique_indices_inverse = np.unique(
        indices, return_inverse=True
    )
    weights = self._sparseSegmentReduceGradWeights(ygrad, segment_ids, mode)
    xgrad_unique = np.zeros(
        [unique_indices.shape[0], ygrad.shape[1]], ygrad.dtype
    )
    for segment, unique_index in zip(segment_ids, unique_indices_inverse):
      xgrad_unique[unique_index] += ygrad[segment] * weights[segment]
    return xgrad_unique, unique_indices


class SparseSegmentReductionOpTest(SparseSegmentReductionHelper):
  # [CLASS DOC]
  # Tests for the sparse segment ops (sum, mean, sqrt_n, with-num-segments variants) and their
  # gradient ops (v1 and v2). Covers values, empty inputs, validation errors for indices and
  # segment ids, and gradients against numpy references and the gradient checker.

  def testValues(self):
    # [FUNCTION DOC]
    # For several dtypes (mean only for float32/64), index dtypes and segment id dtypes (int32,
    # int64), inner sizes 1, 2, 3, 32 and 400 rows with 20 growing segments, runs sparse segment
    # sum and mean and compares with the numpy reference (per-dtype tolerance) and the static
    # shape after the first dimension.
    dtypes = [
        dtypes_lib.float32,
        dtypes_lib.float64,
        dtypes_lib.int64,
        dtypes_lib.int32,
        dtypes_lib.bfloat16,
    ]

    index_dtypes = [dtypes_lib.int32, dtypes_lib.int64]
    segment_ids_dtypes = [dtypes_lib.int32, dtypes_lib.int64]

    mean_dtypes = [dtypes_lib.float32, dtypes_lib.float64]

    # Each item is np_op1, np_op2, tf_op
    ops_list = [(np.add, None, math_ops.sparse_segment_sum),
                (self._mean_cum_op, self._mean_reduce_op,
                 math_ops.sparse_segment_mean)]

    n = 400
    # Note that the GPU implem has different paths for different inner sizes.
    for inner_size in [1, 2, 3, 32]:
      shape = [n, inner_size]
      segment_indices = []
      for i in range(20):
        for _ in range(i + 1):
          segment_indices.append(i)
      num_indices = len(segment_indices)
      for dtype in dtypes:
        for index_dtype in index_dtypes:
          for segment_ids_dtype in segment_ids_dtypes:
            with self.cached_session():
              tf_indices, np_indices, tf_x, np_x = self._sparse_input(
                  shape, num_indices, dtype=dtype)
              for np_op1, np_op2, tf_op in ops_list:
                if (tf_op == math_ops.sparse_segment_mean and
                    dtype not in mean_dtypes):
                  continue
                np_ans = self._sparseSegmentReduce(np_x, np_indices,
                                                   segment_indices, np_op1,
                                                   np_op2)
                s = tf_op(
                    data=tf_x,
                    indices=math_ops.cast(tf_indices, index_dtype),
                    segment_ids=math_ops.cast(segment_indices,
                                              segment_ids_dtype))
                tf_ans = self.evaluate(s)
                self.assertAllCloseAccordingToType(np_ans, tf_ans)
                # NOTE(mrry): The static shape inference that computes
                # `tf_ans.shape` can only infer that sizes from dimension 1
                # onwards, because the size of dimension 0 is data-dependent
                # and may therefore vary dynamically.
                self.assertAllEqual(np_ans.shape[1:], tf_ans.shape[1:])

  def testSegmentIdsHole(self):
    # [FUNCTION DOC]
    # Sparse segment sum and mean with gather indices (8,3,0,9) and segment ids (0,2,2,2): the
    # skipped segment 1 must be zero and results must match the numpy reference.
    tf_x, np_x = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [(np.add, None, math_ops.sparse_segment_sum), (
        self._mean_cum_op, self._mean_reduce_op, math_ops.sparse_segment_mean)]
    segment_indices = [0, 2, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.session():
      for np_op1, np_op2, tf_op in ops_list:
        np_ans = self._sparseSegmentReduce(np_x, tf_indices, segment_indices,
                                           np_op1, np_op2)
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  def testWithNumSegments(self):
    # [FUNCTION DOC]
    # Sparse segment sum and mean with an explicit num_segments of 5 (more than the largest id
    # plus one): the output must have five rows, with empty segments zero, matching the reference.
    tf_x, np_x = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [(np.add, None, math_ops.sparse_segment_sum_with_num_segments),
                (self._mean_cum_op, self._mean_reduce_op,
                 math_ops.sparse_segment_mean_with_num_segments)]
    segment_indices = [0, 2, 2, 2]
    tf_indices = [8, 3, 0, 9]
    num_segments = 5
    for np_op1, np_op2, tf_op in ops_list:
      np_ans = self._sparseSegmentReduce(
          np_x,
          tf_indices,
          segment_indices,
          np_op1,
          np_op2,
          num_segments=num_segments,
      )
      s = tf_op(
          data=tf_x,
          indices=tf_indices,
          segment_ids=segment_indices,
          num_segments=num_segments,
      )
      tf_ans = self.evaluate(s)
      self.assertAllClose(np_ans, tf_ans)

  def testWithEmptySegments(self):
    # [FUNCTION DOC]
    # Sparse segment sum and mean on a 0x4 data tensor with empty indices and ids must return an
    # empty 0x4 result.
    tf_x = constant_op.constant([], shape=[0, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    for tf_op in ops_list:
      s = tf_op(data=tf_x, indices=[], segment_ids=[])
      tf_ans = self.evaluate(s)
      self.assertAllClose(np.zeros([0, 4]), tf_ans)

  def testWithEmptySegmentsWithNumSegments(self):
    # [FUNCTION DOC]
    # Sparse segment sum and mean with-num-segments variants on a 0x4 tensor with empty indices
    # and ids and num_segments of 5 must return a 5x4 tensor of zeros.
    tf_x = constant_op.constant([], shape=[0, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_with_num_segments,
        math_ops.sparse_segment_mean_with_num_segments,
    ]
    for tf_op in ops_list:
      s = tf_op(data=tf_x, indices=[], segment_ids=[], num_segments=5)
      tf_ans = self.evaluate(s)
      self.assertAllClose(np.zeros([5, 4]), tf_ans)

  @test_util.run_in_graph_and_eager_modes
  def testSegmentScalarIdiRaisesInvalidArgumentError(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name has a typo ("Idi", presumably "Ids"); it does not describe
    # what is tested beyond that.
    # Runs in graph and eager modes. Regression test: sparse segment sum, mean and sqrt_n called
    # with a scalar float as data must raise a value error or invalid-argument error whose message
    # says the shape must be at least rank 1.
    """Test for github #46897."""
    ops_list = [
        math_ops.sparse_segment_sum,
        math_ops.sparse_segment_mean,
        math_ops.sparse_segment_sqrt_n,
    ]
    for op in ops_list:
      with self.assertRaisesRegex(
          (ValueError, errors_impl.InvalidArgumentError),
          "Shape must be at least rank 1"):
        op(data=1.0, indices=[0], segment_ids=[3])

  def testSegmentIdsGreaterThanZero(self):
    # [FUNCTION DOC]
    # Sparse segment sum and mean with ids (1,2,2,2) that start above zero: segment 0 must be
    # zero and the rest must match the numpy reference.
    tf_x, np_x = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [(np.add, None, math_ops.sparse_segment_sum), (
        self._mean_cum_op, self._mean_reduce_op, math_ops.sparse_segment_mean)]
    segment_indices = [1, 2, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.session():
      for np_op1, np_op2, tf_op in ops_list:
        np_ans = self._sparseSegmentReduce(np_x, tf_indices, segment_indices,
                                           np_op1, np_op2)
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  def testValid(self):
    # [FUNCTION DOC]
    # Baseline for the invalid-input tests: sparse segment sum and mean with valid indices and
    # sorted ids (0,1,2,2) must evaluate without raising. Nothing is compared.
    # Baseline for the test*Invalid* methods below.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.session():
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testIndicesInvalid1(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. A gather index of -1 must make sparse segment sum and mean fail on
    # evaluation with an op error saying indices[1] is out of range [0, 10).
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, -1, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"indices\[1\] == -1 out of range \[0, 10\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testIndicesInvalid2(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. A gather index of 10 (equal to the first dimension) must make sparse segment
    # sum and mean fail with an op error saying indices[3] is out of range [0, 10).
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 10]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"indices\[3\] == 10 out of range \[0, 10\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid2(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. Ids (0,1,0,1) are not sorted; sparse segment sum and mean must fail on
    # evaluation with an op error saying segment ids are not increasing.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 0, 1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids are not increasing"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid3(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. Ids (0,1,2,0) end with a decrease; sparse segment sum and mean must fail
    # with an op error saying segment id 1 is out of range [0, 1) possibly because unsorted.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 0]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"Segment id 1 out of range \[0, 1\), possibly because "
            "'segment_ids' input is not sorted"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid4(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. Ids (-1,0,1,1) start with a negative id; sparse segment sum and mean must
    # fail with an op error saying segment id -1 is out of range [0, 2) possibly because unsorted.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [-1, 0, 1, 1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"Segment id -1 out of range \[0, 2\), possibly because "
            "'segment_ids' input is not sorted"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid6(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. Ids (0,0,0,-1) with a trailing negative id; sparse segment sum and mean
    # must fail with an op error saying segment ids must be non-negative.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 0, 0, -1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid7(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. Same as the previous negative-id test but with ids (0,0,0,-2): an op error
    # saying segment ids must be non-negative is expected for sparse segment sum and mean.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 0, 0, -2]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid8(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. A single huge index and segment id (2 to the 62 minus 1) must make sparse
    # segment sum and mean fail with an op error about overflow when multiplying.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [2**62 - 1]
    tf_indices = [2**62 - 1]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            "Encountered overflow when multiplying"):
          self.evaluate(s)

  def testSegmentWithNumSegmentsValid(self):
    # [FUNCTION DOC]
    # Baseline for the with-num-segments invalid tests: sparse segment sum and mean with
    # num_segments of 5 and ids (0,1,3,3) must evaluate without error. Nothing is compared.
    # Baseline for the test*WithNumSegmentsInvalid* methods below.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_with_num_segments,
        math_ops.sparse_segment_mean_with_num_segments,
    ]
    num_segments = 5
    segment_indices = [0, 1, 3, 3]
    tf_indices = [8, 3, 0, 9]
    with self.session():
      for tf_op in ops_list:
        s = tf_op(
            data=tf_x,
            indices=tf_indices,
            segment_ids=segment_indices,
            num_segments=num_segments)
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentWithNumSegmentsInvalid1(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. A segment id (5) not below num_segments (5) must make the with-num-segments
    # sparse sum and mean fail with an op error saying ids must be less than num_segments.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_with_num_segments,
        math_ops.sparse_segment_mean_with_num_segments,
    ]
    num_segments = 5
    segment_indices = [0, 1, 3, 5]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(
            data=tf_x,
            indices=tf_indices,
            segment_ids=segment_indices,
            num_segments=num_segments)
        with self.assertRaisesOpError("segment ids must be < num_segments"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentWithNumSegmentsInvalid2(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. A negative num_segments (-2) must raise a value error at graph construction
    # saying a negative value cannot be specified for num_segments (no evaluation needed).
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_with_num_segments,
        math_ops.sparse_segment_mean_with_num_segments,
    ]
    num_segments = -2
    segment_indices = [0, 1, 3, 3]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        with self.assertRaisesRegex(
            ValueError, "Cannot specify a negative value for num_segments"):
          tf_op(
              data=tf_x,
              indices=tf_indices,
              segment_ids=segment_indices,
              num_segments=num_segments)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Graph mode. For sparse segment sum and mean with ids (0,1,2,2) over double data of shape
    # 10x4, compares theoretical and numerical Jacobians from the gradient checker (output 3x4).
    shape = [10, 4]

    segment_indices = [0, 1, 2, 2]
    num_indices = len(segment_indices)
    for tf_op in [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]:
      with self.cached_session():
        tf_indices, _, tf_x, np_x = self._sparse_input(
            shape, num_indices, dtype=dtypes_lib.float64)
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [3, 4],
            x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n)

  @test_util.run_deprecated_v1
  def testGradientWithEmptySegmentsAtEnd(self):
    # [FUNCTION DOC]
    # Graph mode. Gradient check of the with-num-segments sparse sum and mean when num_segments (5)
    # exceeds the used ids, leaving empty trailing segments; Jacobians (output 5x4) must agree.
    shape = [10, 4]

    num_segments = 5
    segment_indices = [0, 1, 2, 2]
    num_indices = len(segment_indices)
    for tf_op in [
        math_ops.sparse_segment_sum_with_num_segments,
        math_ops.sparse_segment_mean_with_num_segments,
    ]:
      with self.cached_session():
        tf_indices, _, tf_x, np_x = self._sparse_input(
            shape, num_indices, dtype=dtypes_lib.float64)
        s = tf_op(
            data=tf_x,
            indices=tf_indices,
            segment_ids=segment_indices,
            num_segments=num_segments)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [5, 4],
            x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n)

  def testGradientExplicit(self):
    # [FUNCTION DOC]
    # Compares the sparse segment sum, mean and sqrt_n gradient ops (v1) with the numpy reference
    # for several inner sizes, output dimension 10, repeated and unsorted gather indices.
    # Note that the GPU implem has different paths for different inner sizes.
    for inner_size in (1, 2, 3, 32):
      with self.session():
        tf_ygrad, np_ygrad = self._input([3, inner_size],
                                         dtype=dtypes_lib.float32)
        segment_ids = [0, 1, 2, 2, 2]
        indices = [8, 3, 0, 9, 3]
        output_dim0 = 10
        ops_list = [
            (math_ops.sparse_segment_sum_grad, "sum"),
            (math_ops.sparse_segment_mean_grad, "mean"),
            (math_ops.sparse_segment_sqrt_n_grad, "sqrtn"),
        ]
        for tf_op, mode in ops_list:
          np_xgrad = self._sparseSegmentReduceGrad(np_ygrad, indices,
                                                   segment_ids, output_dim0,
                                                   mode)
          tf_xgrad = tf_op(tf_ygrad, indices, segment_ids, output_dim0)
          self.assertAllClose(tf_xgrad, np_xgrad)

  def testGradientExplicitSingleOutput(self):
    # [FUNCTION DOC]
    # Same as the explicit gradient test, but all gather indices are zero and the output has a
    # single row (a special case in the GPU implementation).
    # The GPU implem has a special case when there is a single output.
    for inner_size in (1, 2, 3, 32):
      with self.session():
        tf_ygrad, np_ygrad = self._input([3, inner_size],
                                         dtype=dtypes_lib.float32)
        segment_ids = [0, 1, 2, 2, 2]
        indices = [0, 0, 0, 0, 0]
        output_dim0 = 1
        ops_list = [
            (math_ops.sparse_segment_sum_grad, "sum"),
            (math_ops.sparse_segment_mean_grad, "mean"),
            (math_ops.sparse_segment_sqrt_n_grad, "sqrtn"),
        ]
        for tf_op, mode in ops_list:
          np_xgrad = self._sparseSegmentReduceGrad(np_ygrad, indices,
                                                   segment_ids, output_dim0,
                                                   mode)
          tf_xgrad = tf_op(tf_ygrad, indices, segment_ids, output_dim0)
          self.assertAllClose(tf_xgrad, np_xgrad)

  def testGradientV2Explicit(self):
    # [FUNCTION DOC]
    # Compares the v2 sparse segment sum, mean and sqrt_n gradient ops with the numpy v2 reference
    # for several inner sizes; checks both the gradient rows and the returned unique indices.
    # Note that the GPU implem has different paths for different inner sizes.
    for inner_size in (1, 2, 3, 32):
      with self.session():
        tf_ygrad, np_ygrad = self._input(
            [3, inner_size], dtype=dtypes_lib.float32
        )
        segment_ids = [0, 1, 2, 2, 2]
        indices = [8, 3, 0, 9, 3]
        output_dim0 = 10
        ops_list = [
            (math_ops.sparse_segment_sum_grad_v2, "sum"),
            (math_ops.sparse_segment_mean_grad_v2, "mean"),
            (math_ops.sparse_segment_sqrt_n_grad_v2, "sqrtn"),
        ]
        for tf_op, mode in ops_list:
          np_xgrad, np_unique_indices = self._sparseSegmentReduceGradV2(
              np_ygrad, indices, segment_ids, mode
          )
          tf_xgrad, tf_unique_indices = tf_op(
              tf_ygrad, indices, segment_ids, output_dim0
          )
          self.assertAllClose(tf_xgrad, np_xgrad)
          self.assertAllClose(tf_unique_indices, np_unique_indices)

  def testGradientV2ExplicitSingleOutput(self):
    # [FUNCTION DOC]
    # Same as the v2 explicit gradient test but with all gather indices zero and output dimension
    # one, so there is a single unique index; checks gradient rows and unique indices.
    # The GPU implem has a special case when there is a single output.
    for inner_size in (1, 2, 3, 32):
      with self.session():
        tf_ygrad, np_ygrad = self._input(
            [3, inner_size], dtype=dtypes_lib.float32
        )
        segment_ids = [0, 1, 2, 2, 2]
        indices = [0, 0, 0, 0, 0]
        output_dim0 = 1
        ops_list = [
            (math_ops.sparse_segment_sum_grad_v2, "sum"),
            (math_ops.sparse_segment_mean_grad_v2, "mean"),
            (math_ops.sparse_segment_sqrt_n_grad_v2, "sqrtn"),
        ]
        for tf_op, mode in ops_list:
          np_xgrad, np_unique_indices = self._sparseSegmentReduceGradV2(
              np_ygrad, indices, segment_ids, mode
          )
          tf_xgrad, tf_unique_indices = tf_op(
              tf_ygrad, indices, segment_ids, output_dim0
          )
          self.assertAllClose(tf_xgrad, np_xgrad)
          self.assertAllClose(tf_unique_indices, np_unique_indices)

  @test_util.run_deprecated_v1
  def testSparseGradient(self):
    # [FUNCTION DOC]
    # Graph mode. For sparse segment sum and mean with the sparse-gradient option on, records the
    # operation with a gradient tape, asserts the gradient with respect to the data is a sparse
    # indexed-slices object, and compares Jacobians from the gradient checker.
    shape = [10, 4]

    segment_indices = [0, 1, 2, 2]
    num_indices = len(segment_indices)
    for tf_op in [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]:
      with self.cached_session():
        tf_indices, _, tf_x, np_x = self._sparse_input(
            shape, num_indices, dtype=dtypes_lib.float64
        )
        with gradients.GradientTape() as g:
          g.watch(tf_x)
          s = tf_op(
              data=tf_x,
              indices=tf_indices,
              segment_ids=segment_indices,
              sparse_gradient=True,
          )
        tf_dx = g.gradient(s, tf_x)
        self.assertIsInstance(tf_dx, indexed_slices.IndexedSlices)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x, shape, s, [3, 4], x_init_value=np_x.astype(np.double), delta=1
        )
      self.assertAllClose(jacob_t, jacob_n)

  def testGradientValid(self):
    # [FUNCTION DOC]
    # Baseline for the v1 gradient invalid tests: the sum, mean and sqrt_n gradient ops with valid
    # indices and ids (0,1,2,2) and output dimension 10 must evaluate without error.
    # Baseline for the testGradient*Invalid* methods below.
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientIndicesInvalid1(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. For the v1 sum, mean and sqrt_n gradient ops, an index equal to the output
    # dimension (10) must make evaluation fail with an op error saying index 10 is out of range.
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 10]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Index 10 out of range \[0, 10\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientIndicesInvalid2(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. For the v1 gradient ops, a negative index (-1) must make evaluation fail
    # with an op error saying index -1 is out of range [0, 10).
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, -1, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Index -1 out of range \[0, 10\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientSegmentsInvalid1(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. The gradient data has 3 rows but ids (0,1,1,4) imply 5 segments; the v1
    # gradient ops must fail with an op error about an invalid number of segments.
    tf_x, _ = self._input(
        [3, 4], dtype=dtypes_lib.float32)  # expecting 3 segments
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 1, 4]  # 5 segments
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError("Invalid number of segments"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientSegmentsInvalid2(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. One-row gradient data with unsorted ids (0,1,2,0): the v1 gradient ops must
    # fail with an op error saying segment id 1 is out of range [0, 1).
    tf_x, _ = self._input([1, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 0]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id 1 out of range \[0, 1\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientSegmentsInvalid3(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. Two-row gradient data with ids (-1,0,1,1): the v1 gradient ops must fail
    # with an op error saying segment id -1 is out of range [0, 2).
    tf_x, _ = self._input([2, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [-1, 0, 1, 1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id -1 out of range \[0, 2\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientSegmentsInvalid4(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. Empty (0x4) gradient data with ids (0,1,2,-1): the v1 gradient ops must
    # fail with an op error saying segment id 0 is out of range [0, 0).
    tf_x, _ = self._input([0, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, -1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id 0 out of range \[0, 0\)"):
          self.evaluate(s)

  def testGradientV2Valid(self):
    # [FUNCTION DOC]
    # Baseline for the v2 gradient invalid tests: the v2 sum, mean and sqrt_n gradient ops with
    # valid input must evaluate both outputs (gradient and unique indices) without error.
    # Baseline for the testGradientV2*Invalid* methods below.
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad_v2,
        math_ops.sparse_segment_mean_grad_v2,
        math_ops.sparse_segment_sqrt_n_grad_v2,
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s, j = tf_op(tf_x, tf_indices, segment_indices, 10)
        self.evaluate([s, j])

  @test_util.run_deprecated_v1
  def testGradientV2IndicesInvalid1(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. For the v2 gradient ops, an index equal to the output dimension (10) must
    # make evaluation fail with an op error saying index 10 is out of range [0, 10).
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad_v2,
        math_ops.sparse_segment_mean_grad_v2,
        math_ops.sparse_segment_sqrt_n_grad_v2,
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 10]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s, j = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Index 10 out of range \[0, 10\)"):
          self.evaluate([s, j])

  @test_util.run_deprecated_v1
  def testGradientV2IndicesInvalid2(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. For the v2 gradient ops, a negative index (-1) must make evaluation fail
    # with an op error saying index -1 is out of range [0, 10).
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad_v2,
        math_ops.sparse_segment_mean_grad_v2,
        math_ops.sparse_segment_sqrt_n_grad_v2,
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, -1, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s, j = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Index -1 out of range \[0, 10\)"):
          self.evaluate([s, j])

  @test_util.run_deprecated_v1
  def testGradientV2SegmentsInvalid1(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. Three-row gradient data but ids (0,1,1,4) imply 5 segments: the v2 gradient
    # ops must fail with an op error about an invalid number of segments.
    tf_x, _ = self._input(
        [3, 4], dtype=dtypes_lib.float32
    )  # expecting 3 segments
    ops_list = [
        math_ops.sparse_segment_sum_grad_v2,
        math_ops.sparse_segment_mean_grad_v2,
        math_ops.sparse_segment_sqrt_n_grad_v2,
    ]
    segment_indices = [0, 1, 1, 4]  # 5 segments
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s, j = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError("Invalid number of segments"):
          self.evaluate([s, j])

  @test_util.run_deprecated_v1
  def testGradientV2SegmentsInvalid2(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. One-row gradient data with ids (0,1,2,0): the v2 gradient ops must fail with
    # an op error saying segment id 1 is out of range [0, 1).
    tf_x, _ = self._input([1, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad_v2,
        math_ops.sparse_segment_mean_grad_v2,
        math_ops.sparse_segment_sqrt_n_grad_v2,
    ]
    segment_indices = [0, 1, 2, 0]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s, j = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id 1 out of range \[0, 1\)"):
          self.evaluate([s, j])

  @test_util.run_deprecated_v1
  def testGradientV2SegmentsInvalid3(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. Two-row gradient data with ids (-1,0,1,1): the v2 gradient ops must fail
    # with an op error saying segment id -1 is out of range [0, 2).
    tf_x, _ = self._input([2, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad_v2,
        math_ops.sparse_segment_mean_grad_v2,
        math_ops.sparse_segment_sqrt_n_grad_v2,
    ]
    segment_indices = [-1, 0, 1, 1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s, j = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id -1 out of range \[0, 2\)"):
          self.evaluate([s, j])

  @test_util.run_deprecated_v1
  def testGradientV2SegmentsInvalid4(self):
    # [FUNCTION DOC]
    # Graph mode, CPU. Empty (0x4) gradient data with ids (0,1,2,-1): the v2 gradient ops must fail
    # with an op error saying segment id 0 is out of range [0, 0).
    tf_x, _ = self._input([0, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad_v2,
        math_ops.sparse_segment_mean_grad_v2,
        math_ops.sparse_segment_sqrt_n_grad_v2,
    ]
    segment_indices = [0, 1, 2, -1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s, j = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id 0 out of range \[0, 0\)"):
          self.evaluate([s, j])


class SegmentReductionOpBenchmark(test.Benchmark):
  # [CLASS DOC]
  # GPU benchmark for sorted and unsorted segment sum. Class attributes hold the option grids:
  # outer dimensions, outer-to-output ratios, inner dimensions (powers of two plus some unaligned
  # sizes), dtypes (float32, float64), their cross product, two op builder callbacks (sorted and
  # unsorted) and the repeat count of 10 iterations. Benchmark methods are skipped without CUDA.
  outer_dim_options = [2**x for x in range(9, 14, 2)]
  ratio_options = [2**x for x in range(1, 6, 2)]
  inner_dim_options = [2**x for x in range(9, 14, 2)]
  # randomly generated sizes with less alignments
  inner_dim_options += [
      1120, 1215, 1856, 1302, 1329, 1531, 1313, 1672, 1851, 1584
  ]
  dtype_options = [np.float32, np.float64]
  options = (outer_dim_options, ratio_options, inner_dim_options, dtype_options)
  # pylint: disable=g-long-lambda
  op_functors = [lambda vc, vs, seg_ids:
                 ("sorted", math_ops.segment_sum(vc, vs)),
                 lambda vc, vs, seg_ids:
                 ("unsorted",
                  math_ops.unsorted_segment_sum(vc, vs, seg_ids[-1]+1))]
  # pylint: enable=g-long-lambda
  repeat = 10

  def _npTypeToStr(self, t):
    # [FUNCTION DOC]
    # HELPER: names a numpy dtype for benchmark labels.
    # KNOWN ISSUE: only float32 and float64 are handled; any other type falls through and the
    # function returns nothing (null value).
    # Returns the short string for float32 or float64.
    if t == np.float32:
      return "fp32"
    if t == np.float64:
      return "fp64"

  def _runGraph(self, op_functor, outer_dim, ratio, inner_dim, dtype):
    # [FUNCTION DOC]
    # HELPER: runs one benchmark configuration.
    # KNOWN ISSUE: variables are initialized with the test's evaluate helper rather than the
    # session created just above, so the benchmark session may not see initialized variables.
    # Inputs: op_functor (callback building the op from data variable, id variable and ids),
    # outer_dim (rows), ratio (rows per segment), inner_dim (columns) and dtype. Creates random
    # integer data on the first GPU, sorted random ids, builds the op, runs the benchmark helper
    # with a minimum iteration count and a descriptive name. Returns a pair of op kind name and
    # measured wall time.
    output_outer_dim = int(outer_dim / ratio)
    const = np.random.randint(5, size=(outer_dim, inner_dim))
    seg_ids = np.sort(np.random.randint(output_outer_dim, size=outer_dim))
    vs = variables.Variable(seg_ids.astype(np.int32))
    with ops.device("/gpu:0"):
      vc = variables.Variable(const.astype(dtype))
    name, op = op_functor(vc, vs, seg_ids)
    with session.Session() as sess:
      self.evaluate(variables.global_variables_initializer())
      r = self.run_op_benchmark(
          sess,
          op,
          min_iters=self.repeat,
          name="_".join(
              map(str,
                  [name, outer_dim, ratio, inner_dim,
                   self._npTypeToStr(dtype)])))
    return name, r["wall_time"]

  def benchmarkSegmentSumGPU(self):
    # [FUNCTION DOC]
    # Benchmarks sorted segment sum over the whole option grid, each in a fresh graph. Returns
    # immediately (does nothing) if no CUDA GPU is available. Records benchmark results as a side
    # effect.
    if not test.is_gpu_available(cuda_only=True):
      return
    for outer_dim, ratio, inner_dim, dtype in itertools.product(*self.options):
      op_functor = self.op_functors[0]
      with ops.Graph().as_default():
        self._runGraph(op_functor, outer_dim, ratio, inner_dim, dtype)

  def benchmarkUnsortedSegmentSumGPU(self):
    # [FUNCTION DOC]
    # Benchmarks unsorted segment sum over the whole option grid, each in a fresh graph. Returns
    # immediately if no CUDA GPU is available. Records benchmark results as a side effect.
    if not test.is_gpu_available(cuda_only=True):
      return
    for outer_dim, ratio, inner_dim, dtype in itertools.product(*self.options):
      op_functor = self.op_functors[1]
      with ops.Graph().as_default():
        self._runGraph(op_functor, outer_dim, ratio, inner_dim, dtype)


if __name__ == "__main__":
  test.main()
