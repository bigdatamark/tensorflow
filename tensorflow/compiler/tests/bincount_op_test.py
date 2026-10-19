# [MODULE DOC]
# Test module for the bincount operation compiled through XLA. Contains one test class with a
# single error-path test; it relies on the XLA test base class to pick the device and compile
# scope. No import-time side effects; running it as a script starts the test runner.
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
"""Tests for bincount using the XLA JIT."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import errors
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.platform import googletest


class BincountTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case covering the bincount operation. Inherits device selection, session and
  # test-scope helpers from the XLA test base class. Holds no state of its own.

  def testInputRank0(self):
    # [FUNCTION DOC]
    # Scenario: bincount is built (inside the XLA test scope) with a scalar (rank 0) input array,
    # a size of 804 and a weights list of two elements, so weights do not match the input shape.
    # Evaluating the result is expected to raise an invalid-argument error whose message says
    # weights must have the same shape as the array or be an empty tensor. Returns nothing.
    with self.session():
      with self.test_scope():
        bincount = gen_math_ops.bincount(arr=6, size=804, weights=[52, 351])

      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          (
              "`weights` must be the same shape as `arr` or a length-0"
              " `Tensor`, in which case it acts as all weights equal to 1."
          ),
      ):
        self.evaluate(bincount)


if __name__ == "__main__":
  googletest.main()
