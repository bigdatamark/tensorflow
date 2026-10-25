# [MODULE DOC]
# Tests for the python wrapper of the sparse-to-dense op (and a conversion round trip with the
# from-dense helper). Covers integer, floating, complex and string values, scalar and list
# values, custom and default fill values, one- to three-dimensional outputs, argument validation
# errors, index validation switches (out of bounds, repeated and unsorted indices, with the
# different CPU and GPU messages), and static shape inference. The test class is also run with
# the eager-op-as-function mode. Running it as a script starts the TensorFlow test runner.
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
"""Tests for tensorflow.kernels.sparse_op."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
class SparseToDenseTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the sparse-to-dense wrapper, decorated so that eager ops run as
  # functions. Holds no state; each test builds a small dense result and compares it to numpy.

  def testInt(self):
    # [FUNCTION DOC]
    # Scenario: scatter the scalar 1 at positions 1 and 3 of a length-5 vector with default 0.
    # Expected: the vector of zero, one, zero, one, zero as integers.
    tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], 1, 0)
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  @parameterized.parameters(
      dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64
  )
  def testFloatTypes(self, dtype):
    # [FUNCTION DOC]
    # Scenario (parameterized over bfloat16, half, single and double precision): scatter the
    # scalar one of the given type at positions 1 and 3 of a length-5 vector with default 0.0.
    # Input: dtype, the tensor floating type being tested.
    # Expected: the vector zero, one, zero, one, zero in that type.
    tf_ans = sparse_ops.sparse_to_dense(
        [1, 3], [5], array_ops.constant(1.0, dtype=dtype), 0.0
    )
    np_ans = np.array([0, 1, 0, 1, 0]).astype(dtype.as_numpy_dtype)
    self.assertAllClose(np_ans, tf_ans)

  def testComplex(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the loop variable over complex64 and complex128 is never used; both iterations
    # run with complex128 data, so complex64 is not tested. The test also goes through from-dense
    # and sparse-to-dense-tensor conversion rather than the sparse-to-dense wrapper itself.
    # Scenario: build a complex vector of two entries, convert it to a sparse tensor and back.
    # Expected: the round trip equals the original vector.
    for dtype in [dtypes.complex64, dtypes.complex128]:
      tf_val = math_ops.cast(
          constant_op.constant([1.0 + 1.0j, 2.0 - 2.0j]), dtypes.complex128)
      tf_ans = sparse_ops.sparse_tensor_to_dense(sparse_ops.from_dense(tf_val))
      self.assertAllClose(tf_val, tf_ans)

  def testEmptyNonZeros(self):
    # [FUNCTION DOC]
    # Scenario: scatter an empty list of indices and values into a length-5 float vector with
    # default 0.0.
    # Expected: a vector of five zeros.
    indices = array_ops.constant([], dtype=dtypes.int32)
    values = array_ops.constant([], dtype=dtypes.float32)
    tf_ans = sparse_ops.sparse_to_dense(indices, [5], values, 0.0)
    np_ans = np.array([0, 0, 0, 0, 0]).astype(np.float32)
    self.assertAllClose(np_ans, tf_ans)

  def testString(self):
    # [FUNCTION DOC]
    # Scenario: scatter the string a at positions 1 and 3 of a length-5 vector with default b.
    # Expected: the byte-string vector b, a, b, a, b.
    tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], "a", "b")
    np_ans = np.array(["b", "a", "b", "a", "b"]).astype(np.bytes_)
    self.assertAllEqual(np_ans, tf_ans)

  def testSetValue(self):
    # [FUNCTION DOC]
    # Scenario: scatter the list of values one and two at positions 1 and 3 of a length-5 vector
    # with default minus one.
    # Expected: the vector minus one, one, minus one, two, minus one.
    tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], [1, 2], -1)
    np_ans = np.array([-1, 1, -1, 2, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testSetSingleValue(self):
    # [FUNCTION DOC]
    # Scenario: scatter the single scalar one (broadcast to all positions) at positions 1 and 3 of
    # a length-5 vector with default minus one.
    # Expected: the vector minus one, one, minus one, one, minus one.
    tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], 1, -1)
    np_ans = np.array([-1, 1, -1, 1, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def test2d(self):
    # [FUNCTION DOC]
    # Scenario: scatter the scalar one at coordinates (1, 3) and (2, 0) of a 3 by 4 matrix with
    # default minus one.
    # Expected: a matrix of minus ones with ones at those two positions.
    tf_ans = sparse_ops.sparse_to_dense([[1, 3], [2, 0]], [3, 4], 1, -1)
    np_ans = np.array([[-1, -1, -1, -1],
                       [-1, -1, -1, 1],
                       [1, -1, -1, -1]]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testZeroDefault(self):
    # [FUNCTION DOC]
    # Scenario: scatter the value 7 at the scalar index 2 of a length-4 vector without giving a
    # default value.
    # Expected: the default is zero, so the result is zero, zero, seven, zero.
    x = sparse_ops.sparse_to_dense(2, [4], 7)
    self.assertAllEqual(x, [0, 0, 7, 0])

  def test3d(self):
    # [FUNCTION DOC]
    # Scenario: scatter the scalar one at coordinates (1, 3, 0) and (2, 0, 1) of a 3 by 4 by 2
    # tensor with default minus one.
    # Expected: a tensor of minus ones with ones at those two positions.
    tf_ans = sparse_ops.sparse_to_dense([[1, 3, 0], [2, 0, 1]], [3, 4, 2], 1,
                                        -1)
    np_ans = np.ones((3, 4, 2), dtype=np.int32) * -1
    np_ans[1, 3, 0] = 1
    np_ans[2, 0, 1] = 1
    self.assertAllClose(np_ans, tf_ans)

  def testBadShape(self):
    # [FUNCTION DOC]
    # Scenario: pass an output shape that is a matrix (rank 2) instead of a vector.
    # Expected: a value error or invalid-argument error saying the shape must be rank 1.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 1"):
      sparse_ops.sparse_to_dense([1, 3], [[5], [3]], 1, -1)

  def testBadValue(self):
    # [FUNCTION DOC]
    # Scenario: pass the values as a column of shape 2 by 1 for two indices.
    # Expected: an error saying the sparse values have an incorrect shape and should be a scalar
    # or a vector of length 2.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r"sparse_values has incorrect shape \[2,1\], "
                                r"should be \[\] or \[2\]"):
      self.evaluate(sparse_ops.sparse_to_dense([1, 3], [5], [[5], [3]], -1))

  def testBadNumValues(self):
    # [FUNCTION DOC]
    # Scenario: pass three values for two indices.
    # Expected: an error saying the sparse values have shape 3 and should be a scalar or a vector
    # of length 2.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        r"sparse_values has incorrect shape \[3\], should be \[\] or \[2\]"):
      self.evaluate(sparse_ops.sparse_to_dense([1, 3], [5], [1, 2, 3], -1))

  def testBadDefault(self):
    # [FUNCTION DOC]
    # Scenario: pass the default value as a one-element list instead of a scalar.
    # Expected: an error saying the default value should be a scalar.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "default_value should be a scalar"):
      self.evaluate(sparse_ops.sparse_to_dense([1, 3], [5], [1, 2], [0]))

  @test_util.disable_xla("XLA does not check validity for SparseToDense")
  def testOutOfBoundsIndicesWithWithoutValidation(self):
    # [FUNCTION DOC]
    # Scenario (skipped under XLA, which does not validate): scatter into a length-5 vector with
    # an index of 10. First with default validation; then, only when no GPU is available, with
    # index validation disabled.
    # Expected: an error each time. With validation the message depends on the device: a short
    # out-of-bounds message on GPU (which avoids copying inputs to the host) and a detailed one
    # naming the offending index and allowed range on CPU. With validation disabled on CPU the
    # allocation or write still fails with an out-of-bounds error.
    # The GPU implementation doesn't print the contents of the invalid inputs,
    # since the overhead of memory copy between device to host is large.
    # Therefore, the following three tests on invalid inputs will distinguish
    # the reference error messages between GPUs and CPUs.
    error_msg = (r"out of bounds" if test_util.is_gpu_available() else
                 r"indices\[1\] = \[10\] is out of bounds: need 0 <= "
                 "index < \[5\]")
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                error_msg):
      self.evaluate(
          sparse_ops.sparse_to_dense([[1], [10]], [5], [1.0, 1.0], 0.0))
    # When validate_indices=False, the GPU kernel won't check out-of-bound
    # access. Therefore, we skip the following test.
    if not test_util.is_gpu_available():
      # Disable checks, the allocation should still fail.
      with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                  "out of bounds"):
        self.evaluate(
            sparse_ops.sparse_to_dense([[1], [10]], [5], [-1.0, 1.0],
                                       0.0,
                                       validate_indices=False))

  @test_util.disable_xla("XLA does not check validity for SparseToDense")
  def testRepeatingIndicesWithWithoutValidation(self):
    # [FUNCTION DOC]
    # Scenario (skipped under XLA): scatter two values at the same index of a length-5 vector.
    # Expected: with default validation an error saying the second index is repeated (more
    # detailed text on CPU than on GPU); with index validation turned off, the call succeeds.
    error_msg = (r"indices\[1\] is repeated" if test_util.is_gpu_available()
                 else r"indices\[1\] = \[1\] is repeated")
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                error_msg):
      self.evaluate(
          sparse_ops.sparse_to_dense([[1], [1]], [5], [-1.0, 1.0], 0.0))
    # Disable checks
    self.evaluate(
        sparse_ops.sparse_to_dense([[1], [1]], [5], [-1.0, 1.0],
                                   0.0,
                                   validate_indices=False))

  @test_util.disable_xla("XLA does not check validity for SparseToDense")
  def testUnsortedIndicesWithWithoutValidation(self):
    # [FUNCTION DOC]
    # Scenario (skipped under XLA): scatter two values at indices 2 then 1 (decreasing order) into
    # a length-5 vector.
    # Expected: with default validation an error saying the second index is out of order (more
    # detailed text on CPU than on GPU); with index validation turned off, the call succeeds.
    error_msg = (r"indices\[1\] is out of order"
                 if test_util.is_gpu_available() else
                 r"indices\[1\] = \[1\] is out of order")
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                error_msg):
      self.evaluate(
          sparse_ops.sparse_to_dense([[2], [1]], [5], [-1.0, 1.0], 0.0))
    # Disable checks
    self.evaluate(
        sparse_ops.sparse_to_dense([[2], [1]], [5], [-1.0, 1.0],
                                   0.0,
                                   validate_indices=False))

  def testShapeInferenceKnownShape(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph with a placeholder for the indices, build the op with a constant
    # output shape of 4, 5, 6 and then with a placeholder shape vector of length 3.
    # Expected: static shape 4 by 5 by 6 in the first case; three dimensions of unknown size in
    # the second.
    with ops.Graph().as_default():
      indices = array_ops.placeholder(dtypes.int64)

      shape = [4, 5, 6]
      output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape(), [4, 5, 6])

      shape = array_ops.placeholder(dtypes.int64, shape=(3,))
      output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
      self.assertEqual(output.get_shape().as_list(), [None, None, None])

  def testShapeInferenceUnknownShape(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph, build the op with placeholders of unknown shape for both the
    # indices and the output shape.
    # Expected: the result has unknown rank.
    with ops.Graph().as_default():
      indices = array_ops.placeholder(dtypes.int64)
      shape = array_ops.placeholder(dtypes.int64)
      output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
      self.assertIsNone(output.get_shape().ndims)


if __name__ == "__main__":
  test.main()
