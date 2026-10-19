# [MODULE DOC]
# Tests for the sparse-to-dense op on the XLA test device: scalar and vector value fills,
# multi-dimensional index matrices, default values, and error handling for bad shapes and values.
# Defines a small module-level helper that evaluates the op on fed indices. Run as a test binary.
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

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def _SparseToDense(sparse_indices,
                   output_size,
                   sparse_values,
                   default_value,
                   validate_indices=True):
  # [FUNCTION DOC]
  # HELPER: runs the sparse-to-dense op for the tests in this module.
  # Inputs: sparse_indices (scalar, list or matrix of integer positions, fed through an int32
  # placeholder), output_size (shape of the dense result), sparse_values (a scalar or one value
  # per index), default_value (fill for unset positions), validate_indices (optional, defaults to
  # true; passed to the op). Evaluates the op in the currently active default session and returns
  # the dense numpy result. Must be called inside a session scope.
  feed_sparse_indices = array_ops.placeholder(dtypes.int32)
  feed_dict = {feed_sparse_indices: sparse_indices}
  return sparse_ops.sparse_to_dense(
      feed_sparse_indices,
      output_size,
      sparse_values,
      default_value=default_value,
      validate_indices=validate_indices).eval(feed_dict=feed_dict)


class SparseToDenseTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the sparse-to-dense op on the XLA test device. Each test builds a dense result
  # from sparse indices and values and compares it with a numpy-built expectation, or checks
  # the error raised for invalid arguments. No instance state.

  def testInt(self):
    # [FUNCTION DOC]
    # Sets positions 1 and 3 of a length-5 vector to integer 1 with default 0; expects 0,1,0,1,0.
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([1, 3], [5], 1, 0)
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testFloat(self):
    # [FUNCTION DOC]
    # Same as the integer case with float value 1.0 and default 0.0; expects 0,1,0,1,0 as floats.
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([1, 3], [5], 1.0, 0.0)
    np_ans = np.array([0, 1, 0, 1, 0]).astype(np.float32)
    self.assertAllClose(np_ans, tf_ans)

  def testSetValue(self):
    # [FUNCTION DOC]
    # Sets positions 1 and 3 of a length-5 vector to the distinct values 1 and 2 with default -1;
    # expects -1,1,-1,2,-1.
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([1, 3], [5], [1, 2], -1)
    np_ans = np.array([-1, 1, -1, 2, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testSetSingleValue(self):
    # [FUNCTION DOC]
    # Sets positions 1 and 3 of a length-5 vector using one scalar value (1) broadcast to both,
    # with default -1; expects -1,1,-1,1,-1.
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([1, 3], [5], 1, -1)
    np_ans = np.array([-1, 1, -1, 1, -1]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def test2d(self):
    # [FUNCTION DOC]
    # Uses a matrix of two-dimensional indices into a 3x4 output, scalar value 1 and default -1;
    # expects ones at positions (1,3) and (2,0) and -1 elsewhere.
    # pylint: disable=bad-whitespace
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([[1, 3], [2, 0]], [3, 4], 1, -1)
    np_ans = np.array([[-1, -1, -1, -1],
                       [-1, -1, -1,  1],
                       [ 1, -1, -1, -1]]).astype(np.int32)
    self.assertAllClose(np_ans, tf_ans)

  def testZeroDefault(self):
    # [FUNCTION DOC]
    # Calls the op directly with a scalar index 2, output size 4 and value 7 and no default
    # (so default zero); expects 0,0,7,0. Note: unlike the other tests, this one does not enter the
    # XLA test device scope.
    with self.session():
      x = sparse_ops.sparse_to_dense(2, [4], 7).eval()
      self.assertAllEqual(x, [0, 0, 7, 0])

  def test3d(self):
    # [FUNCTION DOC]
    # Uses three-dimensional indices into a 3x4x2 output with value 1 and default -1; expects ones
    # at positions (1,3,0) and (2,0,1) and -1 elsewhere.
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([[1, 3, 0], [2, 0, 1]], [3, 4, 2], 1, -1)
    np_ans = np.ones((3, 4, 2), dtype=np.int32) * -1
    np_ans[1, 3, 0] = 1
    np_ans[2, 0, 1] = 1
    self.assertAllClose(np_ans, tf_ans)

  def testDegenerateIndexMatrix(self):
    # [FUNCTION DOC]
    # Uses an index matrix with a single column (eight rows) into a length-10 output with eight
    # distinct values and default -1; expects the values at positions 2 to 9 and -1 at 0 and 1.
    with self.session(), self.test_scope():
      tf_ans = _SparseToDense([[2], [3], [4], [5], [6], [7], [8], [9]], [10],
                              [1, 2, 3, 4, 5, 6, 7, 8], -1)
    self.assertAllClose([-1, -1, 1, 2, 3, 4, 5, 6, 7, 8], tf_ans)

  def testBadShape(self):
    # [FUNCTION DOC]
    # Passes a two-dimensional output shape; expects a value error saying the shape must be rank 1.
    with self.session(), self.test_scope():
      with self.assertRaisesWithPredicateMatch(ValueError, "must be rank 1"):
        _SparseToDense([1, 3], [[5], [3]], 1, -1)

  @test_util.disable_mlir_bridge("Error handling")
  def testBadValue(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge. Passes values of shape 2x1 for two indices; expects an op
    # error that the values have an incorrect shape and should be scalar or of length 2.
    with self.session(), self.test_scope():
      with self.assertRaisesOpError(
          r"sparse_values has incorrect shape \[2,1\], "
          r"should be \[\] or \[2\]"):
        _SparseToDense([1, 3], [5], [[5], [3]], -1)

  @test_util.disable_mlir_bridge("Error handling")
  def testBadNumValues(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge. Passes three values for two indices; expects an op error that
    # the values have an incorrect shape and should be scalar or of length 2.
    with self.session(), self.test_scope():
      with self.assertRaisesOpError(
          r"sparse_values has incorrect shape \[3\], should be \[\] or \[2\]"):
        _SparseToDense([1, 3], [5], [1, 2, 3], -1)

  @test_util.disable_mlir_bridge("Error handling")
  def testBadDefault(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge. Passes a one-element list as the default value; expects an op
    # error that the default value should be a scalar.
    with self.session(), self.test_scope():
      with self.assertRaisesOpError("default_value should be a scalar"):
        _SparseToDense([1, 3], [5], [1, 2], [0])


if __name__ == "__main__":
  test.main()
