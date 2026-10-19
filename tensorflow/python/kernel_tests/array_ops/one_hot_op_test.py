# [MODULE DOC]
# Tests for the one-hot encoding op. Defines a single parameterized test case that checks the op
# on CPU and GPU placements across numeric, complex and string value types, default and explicit
# on/off values, batched and empty indices, different axes, index dtypes, large prefix dimensions,
# and error handling for mismatched value types. No import-time side effects besides running the
# test main when executed as a script.
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
"""Tests for tensorflow.ops.one_hot_op."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class OneHotTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the one-hot op. Combines the standard test base class with parameterized
  # testing so dtype-dependent tests run once per element type. Private helpers build the op call,
  # evaluate it and compare against an expected array; public methods supply the scenarios.

  def _testOneHot(self,
                  truth,
                  use_gpu=False,
                  expected_err_re=None,
                  raises=None,
                  dtype=None,
                  **inputs):
    # [FUNCTION DOC]
    # HELPER: core assertion helper used by the one-hot tests.
    # Runs the one-hot op with the given keyword inputs (indices, depth, values, axis) and the
    # optional output dtype inside a cached session on CPU or GPU as chosen by use_gpu. If raises
    # is given, asserts that building the op raises that exception type. Otherwise, if no
    # expected_err_re is given, evaluates the result and asserts it equals truth, that its dtype
    # matches dtype when one was requested, and that its shape equals the static shape. If
    # expected_err_re is given, asserts that evaluation fails with an op error matching it.
    with self.cached_session(use_gpu=use_gpu):
      if raises is not None:
        with self.assertRaises(raises):
          array_ops.one_hot(dtype=dtype, **inputs)
      else:
        ans = array_ops.one_hot(dtype=dtype, **inputs)
        if expected_err_re is None:
          tf_ans = self.evaluate(ans)
          self.assertAllEqual(tf_ans, truth)
          if dtype:
            self.assertEqual(tf_ans.dtype, dtype)
          self.assertEqual(tf_ans.shape, ans.get_shape())
        else:
          with self.assertRaisesOpError(expected_err_re):
            self.evaluate(ans)

  def _testBothOneHot(self, truth, expected_err_re=None, raises=None, **inputs):
    # [FUNCTION DOC]
    # HELPER: runs the core one-hot check twice, once with GPU placement allowed and once on CPU,
    # forwarding truth, the optional expected error pattern, the optional expected exception type
    # and the op keyword inputs. Returns nothing; failures surface as assertion errors.
    self._testOneHot(truth, True, expected_err_re, raises, **inputs)
    self._testOneHot(truth, False, expected_err_re, raises, **inputs)

  def _testBasic(self, dtype):
    # [FUNCTION DOC]
    # HELPER: checks one-hot of a 1-D index list (including a negative index that yields an all-off
    # row) with depth 3 and explicit on value 1 and off value -1 of the given numpy dtype, for the
    # last axis and for axis zero (expected output is the transpose).
    indices = np.asarray([0, 2, -1, 1], dtype=np.int64)
    depth = 3
    on_value = np.asarray(1.0, dtype=dtype)
    off_value = np.asarray(-1.0, dtype=dtype)

    truth = np.asarray(
        [[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0],
         [-1.0, 1.0, -1.0]],
        dtype=dtype)

    # axis == -1
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        dtype=dtype,
        truth=truth)

    # axis == 0
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        axis=0,
        dtype=dtype,
        truth=truth.T)  # Output is transpose version in this case

  def _testDefaultBasic(self, dtype):
    # [FUNCTION DOC]
    # HELPER: like the basic check but with default on/off values (one and zero) and the given
    # dtype, which may be nothing so the default float type is used. Checks axis last and axis
    # zero (transposed expectation) with depth 3 and a negative index producing a zero row.
    indices = np.asarray([0, 2, -1, 1], dtype=np.int64)
    depth = 3

    truth = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype)

    # axis == -1
    self._testBothOneHot(indices=indices, depth=depth, dtype=dtype, truth=truth)

    # axis == 0
    self._testBothOneHot(
        indices=indices, depth=depth, axis=0, dtype=dtype,
        truth=truth.T)  # Output is transpose version in this case

  def testDefaultNoDtype(self):
    # [FUNCTION DOC]
    # Runs the default-value one-hot check with no dtype specified, asserting the op's default
    # output behavior is correct for both axes.
    self._testDefaultBasic(None)

  @parameterized.parameters(
      np.float16,
      dtypes.bfloat16.as_numpy_dtype,
      np.float32,
      np.float64,
      np.int8,
      np.int32,
      np.int64,
      np.complex64,
      np.complex128,
  )
  def testBasic(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float16, bfloat16, float32, float64, int8, int32, int64, complex64 and
    # complex128. For each dtype runs the explicit-value and default-value 1-D one-hot checks.
    self._testBasic(dtype)
    self._testDefaultBasic(dtype)

  def _testBatch(self, dtype):
    # [FUNCTION DOC]
    # HELPER: checks one-hot on a 2-D batch of indices with depth 3, on value 1 and off value -1
    # of the given dtype, for axis last and axis one (each batch element transposed, batch not).
    indices = np.asarray([[0, 2, -1, 1], [1, 0, 1, -1]], dtype=np.int64)
    depth = 3
    on_value = np.asarray(1.0, dtype=dtype)
    off_value = np.asarray(-1.0, dtype=dtype)

    truth = np.asarray(
        [[[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0],
          [-1.0, 1.0, -1.0]], [[-1.0, 1.0, -1.0], [1.0, -1.0, -1.0],
                               [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]],
        dtype=dtype)

    # axis == -1
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        dtype=dtype,
        truth=truth)

    # axis == 1
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        axis=1,
        dtype=dtype,
        truth=[truth[0].T, truth[1].T])  # Do not transpose the batch

  def _testDefaultValuesBatch(self, dtype):
    # [FUNCTION DOC]
    # HELPER: batch one-hot check using default on/off values and the given dtype, for axis last
    # and axis one, depth 3, with negative indices producing all-zero rows.
    indices = np.asarray([[0, 2, -1, 1], [1, 0, 1, -1]], dtype=np.int64)
    depth = 3

    truth = np.asarray(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
         [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=dtype)

    # axis == -1
    self._testBothOneHot(indices=indices, depth=depth, dtype=dtype, truth=truth)

    # axis == 1
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        axis=1,
        dtype=dtype,
        truth=[truth[0].T, truth[1].T])  # Do not transpose the batch

  def _testValueTypeBatch(self, dtype):
    # [FUNCTION DOC]
    # HELPER: batch one-hot check where on/off values are numpy scalars of the given dtype and the
    # same dtype is requested for the output, for axis last and axis one. Expected values are
    # identical to the explicit-value batch check.
    indices = np.asarray([[0, 2, -1, 1], [1, 0, 1, -1]], dtype=np.int64)
    depth = 3

    on_value = np.asarray(1.0, dtype=dtype)
    off_value = np.asarray(-1.0, dtype=dtype)

    truth = np.asarray(
        [[[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0],
          [-1.0, 1.0, -1.0]], [[-1.0, 1.0, -1.0], [1.0, -1.0, -1.0],
                               [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]],
        dtype=dtype)

    # axis == -1
    self._testBothOneHot(
        indices=indices,
        on_value=on_value,
        off_value=off_value,
        depth=depth,
        dtype=dtype,
        truth=truth)

    # axis == 1
    self._testBothOneHot(
        indices=indices,
        on_value=on_value,
        off_value=off_value,
        depth=depth,
        axis=1,
        dtype=dtype,
        truth=[truth[0].T, truth[1].T])  # Do not transpose the batch

  def _testEmpty(self, dtype):
    # [FUNCTION DOC]
    # HELPER: checks one-hot on an empty index array of shape zero by sixteen with depth 3 and
    # explicit values of the given dtype; expects an empty result of shape zero, sixteen, three.
    indices = np.zeros((0, 16), dtype=np.int64)
    depth = 3
    on_value = np.asarray(1.0, dtype=dtype)
    off_value = np.asarray(-1.0, dtype=dtype)
    truth = np.empty((0, 16, 3), dtype=dtype)

    # axis == -1
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        dtype=dtype,
        truth=truth)

  @parameterized.parameters(
      np.float16,
      dtypes.bfloat16.as_numpy_dtype,
      np.float32,
      np.float64,
      np.int8,
      np.int32,
      np.int64,
      np.complex64,
  )
  def testBatch(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over several real and complex64 dtypes. Runs the empty, batch and
    # value-type batch checks for each, and the default-values batch check for all but complex64.
    self._testEmpty(dtype)
    self._testBatch(dtype)
    if dtype != np.complex64:
      self._testDefaultValuesBatch(dtype)
    self._testValueTypeBatch(dtype)

  def testSimpleCases(self):
    # [FUNCTION DOC]
    # Checks one-hot of indices zero to two with depth 3 yielding an identity matrix: default
    # float output, an explicit integer output dtype, and integer on/off values one and minus one
    # with the dtype inferred.
    indices = [0, 1, 2]
    depth = 3
    truth = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    self._testBothOneHot(indices=indices, depth=depth, truth=truth)

    indices = [0, 1, 2]
    depth = 3
    truth = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32)
    self._testBothOneHot(
        indices=indices, depth=depth, dtype=np.int32, truth=truth)

    indices = [0, 1, 2]
    depth = 3
    truth = np.asarray([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.int32)
    self._testBothOneHot(
        indices=indices, depth=depth, on_value=1, off_value=-1, truth=truth)

  def testSingleValueGiven(self):
    # [FUNCTION DOC]
    # Checks that supplying only the on value (integer) or only the off value (float zero) works,
    # with the other defaulting, and the result is an identity matrix of the inferred type.
    # Only on_value provided
    indices = [0, 1, 2]
    depth = 3
    truth = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32)
    self._testBothOneHot(indices=indices, depth=depth, on_value=1, truth=truth)

    # Only off_value provided
    indices = [0, 1, 2]
    depth = 3
    truth = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    self._testBothOneHot(
        indices=indices, depth=depth, off_value=0.0, truth=truth)

  def testString(self):
    # [FUNCTION DOC]
    # Checks one-hot with byte-string on/off values supplied as numpy scalars, as constant tensors
    # and as raw byte strings; each must produce the identity pattern of strings.
    indices = [0, 1, 2]
    depth = 3
    truth = np.asarray([[b"1.0", b"0.0", b"0.0"], [b"0.0", b"1.0", b"0.0"],
                        [b"0.0", b"0.0", b"1.0"]])
    on_value = np.asarray(b"1.0")
    off_value = np.asarray(b"0.0")

    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        truth=truth)

    on_value = constant_op.constant(b"1.0")
    off_value = constant_op.constant(b"0.0")
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        truth=truth)

    on_value = b"1.0"
    off_value = b"0.0"
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        truth=truth)

  def testIndicesTypes(self):
    # [FUNCTION DOC]
    # Checks that batched indices of tensor types (uint8, int32, int64) and numpy types (int32,
    # int64) all work, with float on/off values, for axis last and axis one. For uint8 the index
    # minus one becomes 255, which is out of range and so also yields an all-off row.
    tf_types = [dtypes.uint8, dtypes.int32, dtypes.int64]
    np_types = [np.int32, np.int64]
    for itype in tf_types + np_types:
      # Note: to keep the tests simple in the case of uint8 the index -1 below
      # maps to 255 which is out of the depth range, just like -1.
      if itype in tf_types:
        indices = constant_op.constant(
            [[0, 2, -1, 1], [1, 0, 1, -1]], dtype=itype)
      elif itype in np_types:
        indices = np.asarray([[0, 2, -1, 1], [1, 0, 1, -1]], dtype=itype)
      depth = 3

      on_value = np.asarray(1.0, dtype=np.float32)
      off_value = np.asarray(-1.0, dtype=np.float32)

      truth = np.asarray(
          [[[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0]], [[-1.0, 1.0, -1.0], [1.0, -1.0, -1.0],
                                 [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]],
          dtype=np.float32)

      # axis == -1
      self._testBothOneHot(
          indices=indices,
          on_value=on_value,
          off_value=off_value,
          depth=depth,
          truth=truth)

      # axis == 1
      self._testBothOneHot(
          indices=indices,
          on_value=on_value,
          off_value=off_value,
          depth=depth,
          axis=1,
          truth=[truth[0].T, truth[1].T])  # Do not transpose the batch

  def testPrefixDimOverflow(self):
    # [FUNCTION DOC]
    # For index dtypes int32, int64 and uint8, one-hot encodes 65536 alternating indices with
    # depth 2 (a large prefix dimension that could overflow narrow types) and asserts the exact
    # expected encoding.
    for itype in [dtypes.int32, dtypes.int64, dtypes.uint8]:
      prefix_dim_size = 65536
      depth = 2
      x = [i % depth for i in range(prefix_dim_size)]
      indices = constant_op.constant(x, dtype=itype)

      truth = np.zeros((prefix_dim_size, depth), np.float32)
      for i in range(prefix_dim_size):
        truth[i, x[i]] = 1.0

      self._testBothOneHot(
          indices=indices,
          depth=depth,
          on_value=1.0,
          off_value=0.0,
          truth=truth)

  def testOnOffMismatchTypeError(self):
    # [FUNCTION DOC]
    # Asserts that a type error is raised when the on value is float64 and the off value is
    # float32 and no dtype is given.
    indices = [0, 1, 2]
    depth = 3
    on_value = np.asarray(1.0, np.float64)
    off_value = np.asarray(0.0, np.float32)

    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        truth=None,
        raises=TypeError)

  def testDtypeMismatchTypeError(self):
    # [FUNCTION DOC]
    # Asserts a type error is raised when an explicit integer output dtype conflicts with a
    # float32 tensor given as the on value, and likewise when given as the off value.
    indices = [0, 1, 2]
    depth = 3
    on_value = constant_op.constant(1.0, dtypes.float32)
    off_value = constant_op.constant(0.0, dtypes.float32)
    dtype = np.int32

    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        dtype=dtype,
        truth=None,
        raises=TypeError)

    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=off_value,
        dtype=dtype,
        truth=None,
        raises=TypeError)

  def testConvertToTensorOfCorrectDtype(self):
    # [FUNCTION DOC]
    # Checks that Python float on/off values are converted to the requested float16 dtype when
    # mixed with a float16 tensor for the other value, or when both are plain Python floats; the
    # result must be the identity pattern.
    indices = [0, 1, 2]
    depth = 3
    dtype = np.float16
    truth = np.asarray([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    self._testBothOneHot(
        truth=truth,
        indices=indices,
        depth=depth,
        on_value=1.0,
        off_value=constant_op.constant(0.0, dtype),
        dtype=dtype)

    self._testBothOneHot(
        truth=truth,
        indices=indices,
        depth=depth,
        on_value=constant_op.constant(1.0, dtype),
        off_value=0.,
        dtype=dtype)

    self._testBothOneHot(
        truth=truth,
        indices=indices,
        depth=depth,
        on_value=1.0,
        off_value=0.,
        dtype=dtype)

  def testOneHotUint8WithLargeArray(self):
    # [FUNCTION DOC]
    # Builds a uint8 constant from 256 random values scaled to under ten, applies one-hot with
    # depth 10 on axis zero on CPU, and asserts the result shape is ten by 256.
    with self.cached_session(use_gpu=False) as sess:
      matrix = np.random.rand(256) * 10
      tensor = constant_op.constant(matrix, dtypes.uint8, shape=matrix.shape)
      tensor_one_hot = array_ops.one_hot(tensor, depth=10, axis=0)
      self.assertEqual(sess.run(tensor_one_hot).shape, (10, 256))


if __name__ == "__main__":
  test.main()
