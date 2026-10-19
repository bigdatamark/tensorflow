# [MODULE DOC]
# Test module for the reduce-mean op under XLA compilation (overflow check on a very large
# tensor, requires a TPU device). Runs the test runner when executed as a script.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Mean operators."""

from tensorflow.python.eager.def_function import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class MeanOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the reduce-mean op under XLA compilation; currently holds one overflow test
  # that requires a TPU device.
  def testReduceMeanOverflow(self):
    # [FUNCTION DOC]
    # Scenario: reduce-mean over a very large float32 tensor of ones (about 805 million
    # elements) inside an XLA-compiled function placed on the first TPU device. Expects a mean
    # of one within 0.0001, i.e. the sum must not overflow or lose precision. Needs a TPU.
    @function(jit_compile=True)
    def tpu_computation():
      # [FUNCTION DOC]
      # Nested function compiled with XLA: builds a ones tensor of shape four by 32 by 512 by
      # 512 by 96 in float32 and returns its mean as a scalar tensor.
      shape = (4, 32, 512, 512, 96)
      dtype = dtypes.float32
      test_ones = array_ops.ones(shape=shape, dtype=dtype)
      return math_ops.reduce_mean(test_ones)
    with ops.device("TPU:0"):
      result = tpu_computation()
      self.assertAllClose(
          result, 1.0, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
  test.main()
