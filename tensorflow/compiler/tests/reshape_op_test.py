# [MODULE DOC]
# XLA compiler tests for the reshape op: a basic reshape with 32-bit and 64-bit shape tensors
# across all numeric types, and a check that very large dimensions (beyond 32-bit range) are
# handled in static shape inference. No import-time side effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for reshape."""

from absl.testing import parameterized

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class ReshapeTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized XLA test case for the reshape op. No shared fixtures.

  @parameterized.named_parameters(('32_bit_index', dtypes.int32),
                                  ('64_bit_index', dtypes.int64))
  def testBasic(self, index_dtype):
    # [FUNCTION DOC]
    # Parameterized over a 32-bit and a 64-bit integer type for the shape constant. For every
    # numeric type of the device, reshapes a 2x3 placeholder holding 1..6 to the shape 3x2 on the
    # test device and asserts the result is [[1,2],[3,4],[5,6]].
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[2, 3])
        with self.test_scope():
          shape = constant_op.constant([3, 2], dtype=index_dtype)
          o = array_ops.reshape(i, shape)
        params = {
            i: [[1, 2, 3], [4, 5, 6]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[1, 2], [3, 4], [5, 6]], result)

  def testInt64(self):
    # [FUNCTION DOC]
    # Creates a 50000 by 50000 boolean tensor of zeros on the test device (graph construction
    # only, never run) and reshapes it to one dimension of 50000 squared elements, and to 1 by
    # 50000 squared; asserts the inferred static shapes equal the requested ones, showing that
    # dimensions beyond 32-bit range are handled.
    with self.session():
      with self.test_scope():
        x = array_ops.zeros([50000, 50000], dtype=dtypes.bool)
        # Provide dimension larger than int32
        y = array_ops.reshape(x, [50000**2])
        self.assertEqual([50000**2], y.get_shape().as_list())
        # Even if first dimension is within int32, ensure we correctly go to
        # int64
        y = array_ops.reshape(x, [1, 50000**2])
        self.assertEqual([1, 50000**2], y.get_shape().as_list())


if __name__ == '__main__':
  googletest.main()
