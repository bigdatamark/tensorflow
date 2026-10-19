# [MODULE DOC]
# Unit tests for the bitcast op, which reinterprets tensor bytes as another dtype. Covers
# smaller, larger, same-size and same dtype targets, empty (scalar) input, invalid size
# combinations, unknown input shape, and quantized and unsigned target types. Defines one test
# case class (run with the eager-op-as-function variant). No import-time side effects.
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
"""Tests for tf.bitcast."""

import sys

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
class BitcastTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the bitcast op, decorated so that eager ops run as functions. A shared helper
  # verifies byte-for-byte equality, static shape and dtype of the result.

  def _testBitcast(self, x, datatype, shape):
    # [FUNCTION DOC]
    # HELPER: shared verification for the bitcast tests. Takes a numpy array x, the target
    # TensorFlow dtype and the expected static result shape. Bitcasts x (preferring GPU), then
    # compares the raw bytes of the output and input (byte-swapping first on big-endian hosts)
    # and asserts they are identical, that the static shape equals the expected shape and that
    # the result dtype is the target. Returns nothing.
    with test_util.use_gpu():
      tf_ans = array_ops.bitcast(x, datatype)
      out = self.evaluate(tf_ans)
      if sys.byteorder == "little":
        buff_after = memoryview(out).tobytes()
        buff_before = memoryview(x).tobytes()
      else:
        buff_after = memoryview(out.byteswap()).tobytes()
        buff_before = memoryview(x.byteswap()).tobytes()
      self.assertEqual(buff_before, buff_after)
      self.assertEqual(tf_ans.get_shape(), shape)
      self.assertEqual(tf_ans.dtype, datatype)

  def testSmaller(self):
    # [FUNCTION DOC]
    # Bitcasts a random 3x2 float64 array to 8-bit integers; expects an extra trailing dimension
    # of 8 (shape 3x2x8). Returns nothing.
    x = np.random.rand(3, 2)
    datatype = dtypes.int8
    shape = [3, 2, 8]
    self._testBitcast(x, datatype, shape)

  def testLarger(self):
    # [FUNCTION DOC]
    # Bitcasts a 4x4 int8 array to 32-bit integers; expects the last dimension of 4 to collapse
    # (shape 4). Returns nothing.
    x = np.arange(16, dtype=np.int8).reshape([4, 4])
    datatype = dtypes.int32
    shape = [4]
    self._testBitcast(x, datatype, shape)

  def testSameDtype(self):
    # [FUNCTION DOC]
    # Bitcasts a random 3x4 array to its own dtype; expects identical bytes and shape 3x4.
    # Returns nothing.
    x = np.random.rand(3, 4)
    shape = [3, 4]
    self._testBitcast(x, x.dtype, shape)

  def testSameSize(self):
    # [FUNCTION DOC]
    # Bitcasts a random 3x4 float64 array to 64-bit integers (same element size); expects shape
    # 3x4 unchanged. Returns nothing.
    x = np.random.rand(3, 4)
    shape = [3, 4]
    self._testBitcast(x, dtypes.int64, shape)

  def testErrors(self):
    # [FUNCTION DOC]
    # Bitcasts a 1x1 int8 array to 32-bit integers, where the last dimension does not match the
    # size ratio, and asserts a value error or invalid-argument error whose message matches
    # either the classic size message or the more informative message produced when eager ops
    # run as functions. Returns nothing.
    x = np.zeros([1, 1], np.int8)
    datatype = dtypes.int32
    # When eager_op_as_function is enabled shape inference will raise
    # a different more informative error message.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "Cannot bitcast from 6 to 3|convert from s8.* to S32"):
      array_ops.bitcast(x, datatype, None)

  def testEmpty(self):
    # [FUNCTION DOC]
    # Bitcasts a scalar 32-bit integer to 8-bit integers; expects shape 4. Returns nothing.
    x = np.ones([], np.int32)
    datatype = dtypes.int8
    shape = [4]
    self._testBitcast(x, datatype, shape)

  def testUnknownShape(self):
    # [FUNCTION DOC]
    # In a fresh graph, bitcasts a float32 placeholder of unknown shape to 8-bit integers and
    # asserts only that graph construction succeeds. Returns nothing.
    # Need to use placeholder for unknown shape
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32)
      datatype = dtypes.int8
      array_ops.bitcast(x, datatype, None)

  @test_util.disable_tfrt("b/169901260")
  def testQuantizedType(self):
    # [FUNCTION DOC]
    # Skipped under the TFRT runtime. Bitcasts a 3x4 array of unsigned 16-bit zeros to the
    # quantized unsigned 16-bit type; expects identical bytes, shape 3x4 and the target dtype.
    # Returns nothing.
    shape = [3, 4]
    x = np.zeros(shape, np.uint16)
    datatype = dtypes.quint16
    self._testBitcast(x, datatype, shape)

  def testUnsignedType(self):
    # [FUNCTION DOC]
    # Bitcasts a 3x4 array of signed 64-bit zeros to unsigned 64-bit integers; expects identical
    # bytes, shape 3x4 and the target dtype. Returns nothing.
    shape = [3, 4]
    x = np.zeros(shape, np.int64)
    datatype = dtypes.uint64
    self._testBitcast(x, datatype, shape)


if __name__ == "__main__":
  test.main()
